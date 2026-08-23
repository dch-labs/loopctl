//! Observer and hook fan-out — the single home for every lifecycle notification.
//!
//! Every [`LoopObserver`](crate::observer::LoopObserver) callback and every
//! hook notification fired by the driver is dispatched from this module: run
//! start/end, turn start/end, response, tool-call-received, tool pre/post,
//! stream success/failure, and fallback. Co-locating them here means a reader
//! looking for "where does `on_turn_end` fire" finds it in one place, and the
//! driver modules (`llm_turn`, `dispatch`, `compact`) never fire observers
//! directly — they call into the `notify_*` / `record_*` helpers here.

use super::{ApiClient, BareLoop, Duration, LoopError, Run, ToolCall};
use crate::capabilities::FallbackCapable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    RunEndContext as HookRunEndContext, RunEndReason, RunStartContext as HookRunStartContext,
};
use crate::observer::{
    FallbackContext, ResponseContext, RunEndContext, RunStartContext, StreamContext,
    StreamFailureContext, ToolCallReceivedContext, TurnEndContext, TurnStartContext,
};
use crate::stream::Usage;

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
    /// index. Stable across both the LLM-phase and tool-phase turn-end
    /// events for the same turn — they share this number so an observer can
    /// tell which model call a dispatch belonged to.
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
    /// Measured from the start of the relevant handler — for the LLM phase,
    /// from `handle_call_llm`'s entry to the response being recorded; for the
    /// tool phase, from `handle_call_tools`'s entry through dispatch
    /// completion. The two phases time separately, so a single model turn
    /// that triggers tools produces two turn-end events with disjoint
    /// durations (one per phase), not one combined figure. Converted to
    /// `duration_ms` (via [`millis_u64`](BareLoop::millis_u64)) when the
    /// observer context is built.
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
}

impl<C: ApiClient> BareLoop<C> {
    /// Notify all observers and hooks that a run has started.
    pub(super) fn notify_run_start(&self) {
        self.managers.observers().on_run_start(&RunStartContext {
            session_id: self.session.id,
        });
        #[cfg(feature = "hooks")]
        self.notify_run_start_hook();
    }

    /// Notify hooks and observers that a run has ended.
    pub(super) fn notify_run_end(
        &self,
        result: &Run,
        duration: Duration,
        error: Option<&LoopError>,
    ) {
        #[cfg(feature = "hooks")]
        self.notify_run_end_hook(result, error, duration);
        self.managers.observers().on_run_end(&RunEndContext {
            success: error.is_none(),
            error: error.map(std::string::ToString::to_string),
            total_turns: result.turn_count(),
            duration_ms: Self::millis_u64(duration),
        });
    }

    /// Fire [`on_turn_start`](crate::observer::LoopObserver::on_turn_start).
    ///
    /// Called once per turn, at the very top of [`handle_call_llm`](BareLoop::handle_call_llm)
    /// — before contributor messages are gathered, before memory is retrieved,
    /// and before the provider is contacted. This makes it the earliest signal
    /// an observer receives that a new turn has begun, and it always has a
    /// matching [`notify_turn_end`](Self::notify_turn_end) later in the same
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
        self.managers.observers().on_turn_end(&TurnEndContext {
            turn: data.turn,
            success: data.success,
            error: data.error.map(str::to_owned),
            duration_ms: Self::millis_u64(data.duration),
            input_tokens: data.input_tokens,
            output_tokens: data.output_tokens,
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

    /// Record a successful LLM turn: tells the fallback manager the model is
    /// healthy and fires
    /// [`on_stream_success`](crate::observer::LoopObserver::on_stream_success).
    pub(super) fn record_turn_success(&mut self, turn: usize, usage: Option<&Usage>) {
        self.managers.fallback().record_success();
        let (in_tok, out_tok) = Self::usage_tokens(usage);
        self.managers.observers().on_stream_success(&StreamContext {
            turn,
            model: self.client.model(),
            input_tokens: in_tok,
            output_tokens: out_tok,
        });
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
    pub(super) fn record_turn_failure(&mut self, turn: usize, e: LoopError) -> LoopError {
        if matches!(e, LoopError::Cancelled) {
            return e;
        }
        let was_in_fallback =
            self.managers.fallback().state() == crate::fallback::FallbackState::Fallback;
        let tripped = if matches!(e, LoopError::RateLimitEscalation { .. }) {
            self.managers
                .fallback()
                .record_failure(crate::fallback::FailureKind::RateLimit)
        } else {
            self.managers
                .fallback()
                .record_failure(crate::fallback::FailureKind::Transient)
        };

        if tripped {
            let from = self.client.model();
            if let Some(to) = self.managers.fallback().fallback_model() {
                tracing::warn!(from = %from, to = %to, "fallback manager tripped");
                self.managers
                    .observers()
                    .on_fallback(&FallbackContext { from, to });
            }
        }

        if was_in_fallback && let Some(active) = self.managers.fallback().fallback_model() {
            self.managers.fallback().mark_fallback_failed(&active);
        }

        self.managers
            .observers()
            .on_stream_failure(&StreamFailureContext {
                turn,
                model: self.client.model(),
                error: e.clone(),
            });

        e
    }

    /// Pull per-turn `(input_tokens, output_tokens)` from optional [`Usage`].
    ///
    /// Returns `(0, 0)` when the provider did not report usage for the turn.
    pub(super) fn usage_tokens(usage: Option<&Usage>) -> (u64, u64) {
        match usage {
            Some(u) => (u64::from(u.input_tokens), u64::from(u.output_tokens)),
            None => (0, 0),
        }
    }

    /// Convert a [`Duration`] to milliseconds as a `u64`, saturating at
    /// `u64::MAX` on overflow.
    pub(super) fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// Derive the structured [`RunEndReason`] from the terminal error.
    ///
    /// Cancellation (signalled or carried by [`LoopError::Cancelled`]) takes
    /// precedence; then [`LoopError::ContextExceeded`] maps to
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
            Some(LoopError::ContextExceeded { .. }) => RunEndReason::ContextOverflow,
            Some(LoopError::MaxTurnsExceeded { .. }) => RunEndReason::MaxTurns,
            Some(_) => RunEndReason::Error,
        }
    }

    /// Fire the `on_run_start` hook when a hook executor is configured.
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
