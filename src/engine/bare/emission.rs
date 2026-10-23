//! Observer and hook fan-out — the single home for every lifecycle notification.
//!
//! The lifecycle notifications the driver fires are dispatched from this
//! module: run start/end, turn start/end, response, tool-call-received,
//! tool pre/post, stream success/failure, model fallback, and transport
//! fallback. Two per-event families live with their natural home instead —
//! the streaming deltas (`on_text_delta`, `on_thinking_delta`) and
//! `on_model_switched` fire from `llm_turn` — but everything else routes
//! through the `notify_*` / `record_*` helpers here, so a reader looking
//! for "where does `on_turn_end` fire" finds it in one place and the
//! driver modules (`llm_turn`, `dispatch`, `compact`) never fire those
//! observers directly.

use super::{ApiClient, BareLoop, Duration, LoopError, Run, ToolCall, UsageAnchor};
use crate::capabilities::FallbackCapable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    RunEndContext as HookRunEndContext, RunEndReason, RunStartContext as HookRunStartContext,
};
use crate::message::Message;
use crate::observer::{ContextFigureInfo, FigureSource};
use crate::observer::{
    FallbackContext, ResponseContext, RunEndContext, RunStartContext, StreamContext,
    StreamFailureContext, ToolCallReceivedContext, TurnEndContext, TurnStartContext,
};
use crate::stream::{StreamStopReason, Usage};
use std::sync::Arc;

/// Data for an `on_turn_end` notification.
///
/// Bundles everything [`notify_turn_end`](BareLoop::notify_turn_end) forwards to
/// observers so call sites name each field by key instead of lining up seven
/// positional arguments. The wall-clock [`Duration`] is converted to
/// milliseconds when the observer context is built.
pub(super) struct TurnEnd<'a> {
    /// The 0-indexed turn that ended, matching the `turn` field the machine
    /// emitted on the [`MachineStep::CallLLM`](crate::engine::core::MachineStep::CallLLM)
    /// that began this turn.
    ///
    /// Lets observers pair an `on_turn_end` with its earlier
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) by
    /// index. One turn-end fires per serving attempt — the LLM-phase
    /// event for a tool-free turn, the tool-phase event for a turn whose
    /// model call requested tools — and all kinds share this number.
    /// A turn the overflow salvage retried is served twice: its
    /// rejected attempt and its retry each fire their own pair under
    /// this number, so observers pair by index and attempt order, not
    /// by event count.
    pub turn: usize,

    /// Whether the turn reached its intended completion without a hard error.
    ///
    /// `true` for a normal completion (model produced text, or all tool calls
    /// dispatched without a hard failure). `false` when the turn is being
    /// reported because something went wrong: cancellation, dispatch error,
    /// or a recovery-exhausted tool. Soft tool errors (`is_error: true` on a
    /// single result that the model will see) do **not** flip this — they are
    /// surfaced through the result payload, and the turn still "succeeded"
    /// from the loop's perspective.
    pub success: bool,

    /// Human-readable error description, present only when [`success`](Self::success)
    /// is `false`.
    ///
    /// Borrowed for the lifetime of the notification: callers pass either a
    /// `&'static str` (e.g. `"cancelled"`) or a borrow of an owned
    /// [`LoopError::to_string()`] held on the stack for the duration of the
    /// call. [`notify_turn_end`](BareLoop::notify_turn_end) copies it into an
    /// owned `String` before forwarding to observers, so the borrow does not
    /// need to outlive the notification. `None` on the success path.
    pub error: Option<&'a str>,

    /// Wall-clock duration of the phase this turn-end event describes.
    ///
    /// Measured from the start of the relevant handler — for the LLM
    /// phase, from `handle_call_llm`'s entry to the response being
    /// recorded; for the tool phase, from the turn's serving start
    /// (the model-phase entry, carried across the phase boundary)
    /// through dispatch completion, so a tool turn's figure covers
    /// the whole attempt. One turn-end fires per serving attempt:
    /// the LLM phase's for a tool-free turn (covering the model
    /// call), the tool phase's for a turn whose model call requested
    /// tools (covering the model call plus dispatch), and a turn the
    /// overflow salvage retried fires one per attempt — the rejected
    /// attempt's failed event and the retry's own pair. The phase
    /// that does not fire contributes no event, so the two durations
    /// never double-count. Converted to `duration_ms` (via
    /// [`millis_u64`](BareLoop::millis_u64)) when the observer
    /// context is built.
    pub duration: Duration,

    /// Prompt-side token count reported by the provider for the model call
    /// associated with this turn.
    ///
    /// Sourced from the provider's [`Usage`](crate::stream::Usage) on the
    /// LLM phase, and *forwarded unchanged* on the tool phase — tool
    /// dispatch does not consume model tokens, so the same count is repeated
    /// to let an observer compute full-turn cost from either event without
    /// cross-referencing. Defaults to `0` on the cancelled path, where no
    /// provider response was received.
    pub input_tokens: u64,

    /// Completion-side token count for the same model call.
    ///
    /// Same provenance and forwarding semantics as
    /// [`input_tokens`](Self::input_tokens): provider-reported on the LLM
    /// phase, repeated verbatim on the tool phase, `0` on the cancelled path.
    /// Kept as a separate field so the pair travels together onto a single
    /// observer event — a host billing per-turn reads both from one callback
    /// rather than correlating across `on_response` and `on_turn_end`.
    pub output_tokens: u64,

    /// Cached-share token count reported by the provider for the same
    /// model call.
    ///
    /// Same provenance and forwarding semantics as
    /// [`input_tokens`](Self::input_tokens): the turn's cached share on
    /// the LLM phase, repeated verbatim from the recorded
    /// [`Turn`](crate::engine::core::Turn) on the tool phase. `0` means
    /// the provider reported no cached share — unreported, not
    /// zero-known.
    pub cached_input_tokens: u64,

    /// Cache-write token count reported by the provider for the same
    /// model call.
    ///
    /// The cache-creation spend, forwarded with the same provenance as
    /// [`cached_input_tokens`](Self::cached_input_tokens); `0` means
    /// unreported.
    pub cache_write_tokens: u64,

    /// Reasoning-share token count reported by the provider for the
    /// same model call.
    ///
    /// The reasoning share inside [`output_tokens`](Self::output_tokens)
    /// — a subset, never an addition — forwarded with the same
    /// provenance as [`cached_input_tokens`](Self::cached_input_tokens);
    /// `0` means unreported.
    pub reasoning_tokens: u64,

    /// Why the model stopped producing output for this turn.
    ///
    /// The provider's stop field, forwarded onto
    /// [`TurnEndContext::stop_reason`](crate::observer::TurnEndContext::stop_reason).
    /// On the LLM phase it is the stop reason the turn's model call returned;
    /// on the tool phase it is forwarded from the recorded
    /// [`Turn`](crate::engine::core::Turn), matching the token-pair
    /// provenance. Turns that failed before the model finished (cancellation,
    /// stream error) carry the [`EndTurn`](StreamStopReason::EndTurn)
    /// default — the `success`/`error` fields carry that story.
    pub stop_reason: StreamStopReason,

    /// The engine's context-size estimate at the moment of the event.
    ///
    /// Read from the machine's
    /// [`context_tokens`](crate::engine::core::LoopMachine::context_tokens)
    /// at event time — the numerator of the utilization pair; the
    /// denominator arrives as [`context_window`](Self::context_window).
    ///
    /// The figure is the one the compaction trigger evaluates,
    /// reported to observers without a re-estimate. The tool-turn
    /// success event fires after this turn's tool results are recorded
    /// and the estimate refreshed, so its figure already includes
    /// them — including on the run's final tool turn, where no later
    /// event would otherwise report the growth.
    pub context_tokens: u64,

    /// The compaction denominator the run measured against.
    ///
    /// The resolved window (client-disclosed or declared), forwarded
    /// onto
    /// [`TurnEndContext::context_window`](crate::observer::TurnEndContext::context_window);
    /// `None` when the window policy is disabled. Read from the driver
    /// at event time via
    /// [`turn_end_context_window`](BareLoop::turn_end_context_window).
    pub context_window: Option<u64>,

    /// The machine's history snapshot this event reports.
    ///
    /// Built by the firing site: a successful turn end carries
    /// [`full_history`](crate::engine::core::LoopMachine::full_history)
    /// — the committed history plus the run's turns so far, ending with
    /// this turn's complete exchange; a failed turn end carries the
    /// committed [`history`](crate::engine::core::LoopMachine::history)
    /// only, so an interrupted exchange never rides. Shared behind an
    /// [`Arc`] so every observer receives the same snapshot without a
    /// copy each.
    pub history: Arc<Vec<Message>>,
}

/// The provider-reported usage of one served turn, in observer units.
///
/// The five counts travel together from a turn's [`Usage`] report to
/// every surface that forwards them — the
/// [`StreamContext`](crate::observer::StreamContext) fired per request,
/// the [`Turn`](crate::engine::core::Turn) record the run carries, and
/// the [`TurnEnd`] built for the turn-end event — so no call site
/// transposes a pair or drops a detail class along the way. The base
/// counts arrive already normalized by the provider adapters (the
/// Anthropic wire family folds its cache traffic into the input total;
/// see [`Usage`] for the per-family rules), and every detail field
/// follows the crate-wide convention: `0` means not reported.
pub(super) struct TurnUsage {
    /// Prompt-side token count of the turn's model call.
    ///
    /// The provider's after-the-fact count of the prompt it served
    /// (from [`Usage::input_tokens`]), widened from the report's `u32`
    /// into the observer contexts' `u64`. On the Anthropic wire family
    /// this is the cache-aware total — raw input plus cache read plus
    /// cache write — so it never reads as the shrinking uncached
    /// remainder.
    pub input_tokens: u64,

    /// Completion-side token count of the turn's model call.
    ///
    /// Everything the model generated for the turn (from
    /// [`Usage::output_tokens`]): answer text, tool calls, and any
    /// reasoning share reported inside it. A host billing per turn
    /// reads this together with [`input_tokens`](Self::input_tokens)
    /// from one record rather than correlating events.
    pub output_tokens: u64,

    /// Cached share of the turn's input, `0` when unreported.
    ///
    /// The read share of the provider's prompt cache (from
    /// [`Usage::cached_input_tokens`]); a subset of
    /// [`input_tokens`](Self::input_tokens), never additional to it.
    /// `0` means the provider reported no cached share — unreported,
    /// not zero-known, so a hit-rate display treats it as no data.
    pub cached_input_tokens: u64,

    /// Cache-write spend of the turn's input, `0` when unreported.
    ///
    /// The tokens the turn stored into the provider's cache (from
    /// [`Usage::cache_write_tokens`]) — the pricing class that makes
    /// the next turn's reads cheap — with the same subset relationship
    /// as [`cached_input_tokens`](Self::cached_input_tokens) and the
    /// same `0`-means-unreported convention.
    pub cache_write_tokens: u64,

    /// Reasoning share of the turn's output, `0` when unreported.
    ///
    /// The thinking share inside [`output_tokens`](Self::output_tokens)
    /// (from [`Usage::reasoning_tokens`]) — a subset, never an
    /// addition. `0` covers both a non-thinking model and a wire that
    /// reports no reasoning class; consumers computing a reasoning
    /// share skip the row rather than divide by it.
    pub reasoning_tokens: u64,
}

impl TurnUsage {
    /// Map an optional provider [`Usage`] report into observer units.
    ///
    /// The single conversion between the wire-side `u32` counters and
    /// the `u64` observer units every forwarding surface carries:
    /// [`record_turn_success`](BareLoop::record_turn_success) builds
    /// the per-request context from it and `handle_call_llm` builds the
    /// turn record and turn-end event from it, so both paths report one
    /// identical story. A missing report (`None`) maps to
    /// [`zeros`](Self::zeros) — the shape of a backend that reports no
    /// usage at all — rather than propagating an optionality every
    /// consumer would have to re-handle.
    pub(super) fn from_report(usage: Option<&Usage>) -> Self {
        match usage {
            Some(u) => Self {
                input_tokens: u64::from(u.input_tokens),
                output_tokens: u64::from(u.output_tokens),
                cached_input_tokens: u64::from(u.cached_input_tokens),
                cache_write_tokens: u64::from(u.cache_write_tokens),
                reasoning_tokens: u64::from(u.reasoning_tokens),
            },
            None => Self::zeros(),
        }
    }

    /// The all-zero report: no usage, no detail.
    ///
    /// The stand-in for a turn that never received a provider report —
    /// the cancelled and salvage-failure arms of `handle_call_llm`,
    /// which fire their failed turn-end without a response in hand, and
    /// `handle_call_tools`' path for a turn whose record never landed.
    /// Every forwarding surface still receives a complete five-count
    /// record, so an observer never distinguishes "no usage" from
    /// "absent event payload".
    pub(super) fn zeros() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
        }
    }
}

impl<C: ApiClient> BareLoop<C> {
    /// Notify all observers and hooks that a run has started.
    ///
    /// Observers always fire; the hook half is feature-gated and
    /// skipped when no executor is configured.
    pub(super) fn notify_run_start(&self) {
        self.managers.observers().on_run_start(&RunStartContext {
            session_id: self.session.id,
        });
        #[cfg(feature = "hooks")]
        self.notify_run_start_hook();
    }

    /// Notify hooks and observers that a run has ended.
    ///
    /// Carries the finished run, its duration, and the terminal error
    /// when the run failed.
    pub(super) fn notify_run_end(
        &self,
        result: &Run,
        duration: Duration,
        error: Option<&LoopError>,
    ) {
        #[cfg(feature = "hooks")]
        self.notify_run_end_hook(result, error, duration);
        let context_figure = match &self.anchor {
            Some(anchor) => Some(
                ContextFigureInfo::new(FigureSource::Anchored).with_anchor_age_turns(Some(
                    self.session
                        .total_turns()
                        .saturating_sub(anchor.anchored_at_turn),
                )),
            ),
            None => Some(ContextFigureInfo::new(FigureSource::Estimated)),
        };
        let unfinished_input = error
            .filter(|e| super::conversation_cannot_fit(e))
            .and_then(|_| self.session.current_run().map(|run| run.input.clone()));
        let usage = Self::run_usage_totals(result);
        self.managers.observers().on_run_end(&RunEndContext {
            success: error.is_none(),
            error: error.map(std::string::ToString::to_string),
            total_turns: result.turn_count(),
            duration_ms: Self::millis_u64(duration),
            calibration_ratio_bp: self.calibration.report(),
            context_figure,
            unfinished_input,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            cache_write_tokens: usage.cache_write_tokens,
            reasoning_tokens: usage.reasoning_tokens,
        });
    }

    /// Fire [`on_turn_start`](crate::observer::LoopObserver::on_turn_start).
    ///
    /// Called once per served turn — after the turn's transient context
    /// is gathered (contributors, memories) and after the window
    /// re-decision, right before the provider is contacted. A turn that
    /// defers to compaction fires no turn events at all; the retried
    /// turn fires them once. Every fired start has a matching
    /// [`notify_turn_end`](Self::notify_turn_end) later in the same
    /// turn (on the success, soft-error, or cancelled path).
    ///
    /// `query` is **not** the outbound request body — it is a text summary of
    /// what the model is being asked to respond to on this turn, useful for
    /// logging and UI display rather than for replaying the request. On turn 0
    /// it is the user's input verbatim (the `Run::input` string); on later
    /// turns it is the concatenated text of the last message in the machine's
    /// history (typically the prior assistant response, or a tool result on a
    /// turn that follows tool dispatch). Tool-only turns do not produce a
    /// separate `on_turn_start` — one turn is one model call, regardless of
    /// how many tools it triggered.
    ///
    /// The `query` is copied into an owned [`String`](TurnStartContext::query)
    /// when the observer context is built, so the borrow only needs to live
    /// for the duration of this call.
    pub(super) fn notify_turn_start(&self, turn: usize, query: &str) {
        self.managers.observers().on_turn_start(&TurnStartContext {
            turn,
            query: query.to_string(),
        });
    }

    /// Fire [`on_turn_end`](crate::observer::LoopObserver::on_turn_end).
    ///
    /// Takes the turn-end data as a single [`TurnEnd`] so call sites name every
    /// field by key — eliminating the transposition risk of seven positional
    /// arguments. `duration_ms` is derived from the [`Duration`] inside the
    /// struct when the observer context is built.
    pub(super) fn notify_turn_end(&self, data: &TurnEnd) {
        if data.success
            && let Some(gate) = &self.budget_gate
        {
            gate.observe_turn_end();
        }
        self.managers.observers().on_turn_end(&TurnEndContext {
            turn: data.turn,
            success: data.success,
            error: data.error.map(str::to_owned),
            duration_ms: Self::millis_u64(data.duration),
            input_tokens: data.input_tokens,
            output_tokens: data.output_tokens,
            cached_input_tokens: data.cached_input_tokens,
            cache_write_tokens: data.cache_write_tokens,
            reasoning_tokens: data.reasoning_tokens,
            stop_reason: data.stop_reason,
            context_tokens: data.context_tokens,
            context_window: data.context_window,
            history: Arc::clone(&data.history),
        });
    }

    /// Fire [`on_response`](crate::observer::LoopObserver::on_response).
    ///
    /// Called once per turn, mid-`handle_call_llm`, after the provider call
    /// ([`do_turn`](BareLoop::do_turn)) has returned a complete assistant
    /// message and the text has been extracted via
    /// [`Message::text_content`](crate::message::Message::text_content).
    /// Sits between response-side loop detection
    /// ([`record_response`](crate::detection::LoopDetector::record_response))
    /// and the turn-end event, so an observer sees the model's text *before*
    /// the turn is reported as ended — useful for streaming UIs that want to
    /// render the answer as soon as it is final.
    ///
    /// `text` is the concatenated text content of the assistant message only;
    /// tool-call parts are excluded (they surface through the separate
    /// [`on_tool_call_received`](Self::notify_tool_calls_received) /
    /// [`on_tool_post`](Self::notify_tool_post) events). May be empty if the
    /// model's response carried only tool calls and no text.
    ///
    /// `usage` is the provider-reported token pair, forwarded unchanged. It
    /// is `Option` because not every provider or response includes usage — a
    /// provider that omits it yields `None`, and observers that depend on
    /// token counts must handle that case. The same usage is also split into
    /// `input_tokens`/`output_tokens` and recorded on the run's `Turn`, so
    /// the [`on_turn_end`](Self::notify_turn_end) event carries the counts as
    /// concrete `u64`s for observers that prefer the later, more complete
    /// event.
    pub(super) fn notify_response(&self, turn: usize, text: &str, usage: Option<Usage>) {
        self.managers.observers().on_response(&ResponseContext {
            turn,
            text: text.to_string(),
            usage,
        });
    }

    /// Fire [`on_tool_call_received`](crate::observer::LoopObserver::on_tool_call_received)
    /// once per tool call the model requested this turn.
    ///
    /// Called from [`handle_call_tools`](BareLoop::handle_call_tools) after the
    /// pending calls have been split into preresolved (unknown-tool) and
    /// dispatch-bound buckets, but **before** any tool actually executes. This
    /// makes it the earliest per-call observer signal in the dispatch phase —
    /// earlier than [`on_tool_pre`](Self::notify_tool_pre), which fires only
    /// for calls that get dispatched.
    ///
    /// Fires for **every** call the model emitted, including preresolved
    /// unknown-tool calls that will never be dispatched. This is the only
    /// observer event those calls produce: they have no
    /// [`on_tool_pre`](crate::observer::LoopObserver::on_tool_pre) /
    /// [`on_tool_post`](crate::observer::LoopObserver::on_tool_post) pair,
    /// because no execution happens — the driver feeds back a synthetic
    /// error result without running a tool. An observer correlating
    /// `on_tool_call_received` against `on_tool_pre` will see a strict
    /// subset: every dispatched call appears in both, but unknown-tool calls
    /// appear only here.
    ///
    /// `turn` matches the value on the preceding
    /// [`on_response`](crate::observer::LoopObserver::on_response) for the
    /// same assistant message, so an observer can pair each requested call
    /// back to the response that asked for it. The calls fire in the order
    /// the model emitted them (input order), regardless of how they are
    /// later dispatched — sequential, parallel-waves, or preresolved-shortcut.
    pub(super) fn notify_tool_calls_received(&self, turn: usize, tool_calls: &[ToolCall]) {
        for tc in tool_calls {
            self.managers
                .observers()
                .on_tool_call_received(&ToolCallReceivedContext {
                    turn,
                    tool: tc.tool.clone(),
                    call_id: tc.id.clone(),
                    input: tc.input.clone(),
                });
        }
    }

    /// The model that served the current turn, for observer contexts.
    ///
    /// Reads the breaker's current routing (a fallback while it is
    /// open, the primary otherwise) and falls back to the client's
    /// static model when no manager routes models. Callers capture it
    /// before fallback state changes: a success may close the circuit
    /// and a failure may advance the chain, and the observer context
    /// should name the model that was serving, not the one that serves
    /// next.
    /// # Errors
    ///
    /// Returns [`LockPoisoned`](crate::error::LoopError::LockPoisoned) when the fallback state lock is poisoned.
    pub(super) fn routed_or_client_model(&self) -> Result<String, LoopError> {
        Ok(self.routed_model()?.unwrap_or_else(|| self.client.model()))
    }

    /// Fire [`on_transport_fallback`](crate::observer::LoopObserver::on_transport_fallback).
    ///
    /// Reports that this turn was served by the non-streaming transport
    /// fallback — streaming exhausted its retry ceiling and the handler's
    /// last-chance `create_message` produced the answer. Fired from the
    /// engine's stream path the moment the fallback response is in hand,
    /// before the turn's success bookkeeping and well before its
    /// `on_turn_end`, so a live consumer hears about the degradation as it
    /// happens rather than re-reading the run record. Streaming-only: the
    /// fallback is a stream-handler behavior, so no other turn path calls
    /// this.
    #[cfg(feature = "streaming")]
    pub(super) fn notify_transport_fallback(&self, turn: usize, stop_reason: StreamStopReason) {
        self.managers.observers().on_transport_fallback(
            &crate::observer::TransportFallbackContext { turn, stop_reason },
        );
    }

    /// Fire [`on_attempt_reset`](crate::observer::LoopObserver::on_attempt_reset)
    /// for the retried stream attempt the engine is entering.
    ///
    /// Called right after the engine discarded the failed attempt's
    /// accumulator — the observer's cue to discard its own per-turn delta
    /// buffer for the same turn. Streaming-only: only the stream-handler
    /// retry path resets attempts.
    #[cfg(feature = "streaming")]
    pub(super) fn notify_attempt_reset(&self, attempt: usize) {
        let turn = self.session.current_run().map_or(0, Run::turn_count);
        self.managers
            .observers()
            .on_attempt_reset(&crate::observer::AttemptResetContext { turn, attempt });
    }

    /// Record a successful LLM turn: tells the fallback manager the model is
    /// healthy and fires
    /// [`on_stream_success`](crate::observer::LoopObserver::on_stream_success).
    ///
    /// # Errors
    ///
    /// Returns [`LockPoisoned`](crate::error::LoopError::LockPoisoned) when the fallback state lock is poisoned.
    pub(super) fn record_turn_success(
        &mut self,
        turn: usize,
        usage: Option<&Usage>,
    ) -> Result<(), LoopError> {
        let served = self.routed_or_client_model()?;
        self.managers.fallback().record_success()?;
        let usage = TurnUsage::from_report(usage);
        let TurnUsage {
            input_tokens: in_tok,
            output_tokens: out_tok,
            cached_input_tokens,
            cache_write_tokens,
            reasoning_tokens,
        } = usage;
        let armed_prediction = self.calibration.prediction;
        if in_tok > 0
            && let Some(raw_payload) = armed_prediction
            && raw_payload > 0
        {
            self.anchor = Some(UsageAnchor {
                input_tokens: in_tok,
                raw_payload,
                anchored_at_turn: self.session.total_turns().saturating_add(1),
            });
        }
        let ratio_before = self.calibration.ratio_bp;
        self.calibration.observe_usage(in_tok);
        if self.calibration.ratio_bp != ratio_before && self.effective_context_window() > 0 {
            self.sync_manager_window(self.effective_context_window());
        }
        if let Some(gate) = &self.budget_gate {
            gate.observe_usage(in_tok, out_tok);
        }
        self.managers.observers().on_stream_success(
            &StreamContext::new(turn, served, in_tok, out_tok).with_usage_detail(
                cached_input_tokens,
                cache_write_tokens,
                reasoning_tokens,
            ),
        );
        Ok(())
    }

    /// Record an LLM-turn failure and return the error to propagate.
    ///
    /// A [`LoopError::Cancelled`] short-circuits without touching the breaker
    /// or firing `on_stream_failure`. A rate-limit escalation trips the breaker
    /// as a rate-limit failure; anything else is a transient failure. When the
    /// breaker trips and a fallback model is configured, fires
    /// [`on_fallback`](crate::observer::LoopObserver::on_fallback). Always fires
    /// [`on_stream_failure`](crate::observer::LoopObserver::on_stream_failure)
    /// (except for the cancel short-circuit) and returns the original error.
    /// # Errors
    ///
    /// Returns [`LockPoisoned`](crate::error::LoopError::LockPoisoned) when the fallback state lock is poisoned.
    pub(super) fn record_turn_failure(&mut self, turn: usize, e: LoopError) -> LoopError {
        if matches!(e, LoopError::Cancelled) {
            return e;
        }
        // A poisoned breaker while recording a failure is the more
        // severe condition — surface it instead of the original error.
        self.record_turn_failure_inner(turn, e)
            .unwrap_or_else(|poison| poison)
    }

    /// The `?`-propagating body of [`record_turn_failure`](Self::record_turn_failure):
    /// `Ok` carries the error to propagate (the original, normally);
    /// `Err` carries a poison that supersedes it.
    ///
    /// # Errors
    ///
    /// Returns the original error (`Ok`) or a poison that supersedes it
    /// (`Err`).
    fn record_turn_failure_inner(
        &mut self,
        turn: usize,
        e: LoopError,
    ) -> Result<LoopError, LoopError> {
        let served = self.routed_or_client_model()?;
        let was_in_fallback =
            self.managers.fallback().state()? == crate::fallback::FallbackState::Fallback;
        let tripped = if matches!(e, LoopError::RateLimitEscalation { .. }) {
            self.managers
                .fallback()
                .record_failure(crate::fallback::FailureKind::RateLimit)
        } else {
            self.managers
                .fallback()
                .record_failure(crate::fallback::FailureKind::Transient)
        }?;

        if tripped && let Some(to) = self.managers.fallback().fallback_model()? {
            tracing::warn!(from = %served, to = %to, "fallback manager tripped");
            self.managers.observers().on_fallback(&FallbackContext {
                from: served.clone(),
                to,
            });
        }

        if was_in_fallback && let Some(active) = self.managers.fallback().fallback_model()? {
            self.managers.fallback().mark_fallback_failed(&active)?;
        }

        self.managers
            .observers()
            .on_stream_failure(&StreamFailureContext {
                turn,
                model: served,
                error: e.clone(),
            });

        Ok(e)
    }

    /// Sum a run's per-turn usage into the run-end totals.
    ///
    /// Saturating sums over the recorded [`Turn`](crate::engine::core::Turn)
    /// entries, so a run whose backend never reported usage totals zero and
    /// a backend that did carries its detail classes through.
    fn run_usage_totals(result: &Run) -> TurnUsage {
        let mut totals = TurnUsage::zeros();
        for turn in &result.turns {
            totals.input_tokens = totals.input_tokens.saturating_add(turn.input_tokens);
            totals.output_tokens = totals.output_tokens.saturating_add(turn.output_tokens);
            totals.cached_input_tokens = totals
                .cached_input_tokens
                .saturating_add(turn.cached_input_tokens);
            totals.cache_write_tokens = totals
                .cache_write_tokens
                .saturating_add(turn.cache_write_tokens);
            totals.reasoning_tokens = totals
                .reasoning_tokens
                .saturating_add(turn.reasoning_tokens);
        }
        totals
    }

    /// Convert a [`Duration`] to milliseconds as a `u64`.
    ///
    /// Saturates at `u64::MAX` on overflow rather than truncating or
    /// panicking.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// Derive the structured [`RunEndReason`] from the terminal error.
    ///
    /// Cancellation (signalled or carried by [`LoopError::Cancelled`]) takes
    /// precedence; then the context-overflow family —
    /// [`LoopError::ContextExceeded`], [`LoopError::CompactionFailed`],
    /// [`LoopError::CompactionStalled`] — maps to
    /// [`ContextOverflow`](RunEndReason::ContextOverflow),
    /// [`LoopError::MaxTurnsExceeded`] to
    /// [`MaxTurns`](RunEndReason::MaxTurns), any other error to
    /// [`Error`](RunEndReason::Error), and `None` to
    /// [`Complete`](RunEndReason::Complete).
    #[cfg(feature = "hooks")]
    fn run_end_reason(&self, error: Option<&LoopError>) -> RunEndReason {
        if self.is_cancelled() {
            return RunEndReason::Cancelled;
        }
        match error {
            None => RunEndReason::Complete,
            Some(LoopError::Cancelled) => RunEndReason::Cancelled,
            Some(
                LoopError::ContextExceeded { .. }
                | LoopError::CompactionFailed { .. }
                | LoopError::CompactionStalled { .. },
            ) => RunEndReason::ContextOverflow,
            Some(LoopError::MaxTurnsExceeded { .. }) => RunEndReason::MaxTurns,
            Some(_) => RunEndReason::Error,
        }
    }

    /// Fire the `on_run_start` hook when a hook executor is configured.
    ///
    /// A missing executor is a no-op for feature-enabled builds without
    /// one; builds without the `hooks` feature do not compile this call
    /// at all.
    #[cfg(feature = "hooks")]
    fn notify_run_start_hook(&self) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let ctx = HookRunStartContext {
            session_id: self.session.id,
            model: self.client.model(),
            working_directory: std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        executor.notify_run_start(&ctx);
    }

    /// Fire the `on_run_end` hook when a hook executor is configured.
    ///
    /// The context carries the end reason derived from the run's
    /// outcome and error.
    #[cfg(feature = "hooks")]
    fn notify_run_end_hook(&self, result: &Run, error: Option<&LoopError>, duration: Duration) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let reason = self.run_end_reason(error);
        let ctx = HookRunEndContext {
            session_id: self.session.id,
            reason,
            total_turns: result.turn_count(),
            total_tokens: result.total_tokens(),
            duration_secs: duration.as_secs(),
        };
        executor.notify_run_end(&ctx);
    }
}
