//! The sans-IO agent-loop state machine.
//!
//! [`LoopMachine`] owns every *decision* the loop makes — turn counting,
//! max-turn enforcement, tool-call validity, stop-reason routing, compaction
//! triggering, history accumulation, and the cancellation flag — and performs
//! zero IO. The machine is policy-free and serializes as pure state; the turn
//! budget and compaction thresholds are passed in fresh on each
//! [`next_step`](LoopMachine::next_step) call via [`MachinePolicy`].

use serde::{Deserialize, Serialize};

use super::lifecycle::{StopReason, ToolCall};
use crate::compact::types::CompactReason;
use crate::error::LoopError;
use crate::message::{Message, MessagePart, Role, ToolContent};

/// One unit of work the driver must perform before feeding an outcome back.
///
/// Returned by [`LoopMachine::next_step`]. The driver matches on this and, for
/// each non-terminal variant, performs the requested work and feeds the result
/// back into the machine via the matching method ([`LoopMachine::model_response`],
/// [`LoopMachine::tool_results`], [`LoopMachine::compaction_result`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MachineStep {
    /// Request an LLM call over the machine's current [`history`](LoopMachine::history).
    ///
    /// The driver builds the feed (the messages actually sent to the LLM) from
    /// [`LoopMachine::history`], calls the provider, and feeds the completed
    /// [`ModelResponse`] back via [`LoopMachine::model_response`]. `turn` is the
    /// 0-indexed number of the turn being requested.
    CallLLM {
        /// The 0-indexed turn number being requested.
        ///
        /// Starts at `0` on the first call after construction and increments
        /// with each completed model response. The driver can use it to tag
        /// observer events, logs, and rate-limit bookkeeping so each turn is
        /// correlatable back to its request.
        turn: usize,
    },

    /// Request that the driver dispatch these tool calls.
    ///
    /// Each [`PendingToolCall`] may carry a [`preresolved_result`](PendingToolCall::preresolved_result):
    /// when set, the driver returns that content as the tool result *without*
    /// dispatching (used for invalid-tool-call recovery). After dispatch, the
    /// driver feeds the tool-result [`Message`]s back via
    /// [`LoopMachine::tool_results`].
    CallTools {
        /// The 0-indexed turn number whose tool calls are being dispatched.
        ///
        /// Matches the `turn` of the preceding [`MachineStep::CallLLM`] — the
        /// tools belong to the model response that just completed. The driver
        /// uses it to tag observer events so the LLM and tool events for the
        /// same turn correlate.
        turn: usize,

        /// The tool calls awaiting dispatch, with any preresolved results.
        ///
        /// Exactly the calls the model requested in the preceding
        /// [`MachineStep::CallLLM`], in order. Entries whose
        /// [`preresolved_result`](PendingToolCall::preresolved_result) is set
        /// should be fed straight back rather than dispatched to a tool.
        calls: Vec<PendingToolCall>,
    },

    /// Request a context compaction pass over the history.
    ///
    /// The machine decides *when* to compact (this step); the driver's
    /// `ContextCompactor` decides *how*. After compacting, the driver feeds the
    /// compacted history back via [`LoopMachine::compaction_result`].
    Compact {
        /// Why compaction was triggered.
        ///
        /// The [`CompactReason`] that caused the machine to emit this step, which
        /// a driver may use to pick a compaction strategy.
        reason: CompactReason,
    },

    /// Terminal — the run is complete.
    ///
    /// The wrapped [`MachineOutcome`] describes *how* the run ended: normal
    /// completion, the max-turn limit, cancellation, or a failure. Once the
    /// machine emits this, every subsequent [`LoopMachine::next_step`] repeats
    /// the same outcome — the machine is finished and accepts no further input.
    Done(MachineOutcome),
}

/// A tool call awaiting dispatch, with an optional preresolved result.
///
/// The machine classifies each model-emitted tool call against the names the
/// driver reported as available ([`ModelResponse::available_tools`]). For a name the
/// driver did not advertise, the machine sets
/// [`preresolved_result`](Self::preresolved_result) to a synthetic error
/// [`Message`]; the driver feeds that back as the tool result without
/// dispatching, so the model learns the name is unknown and can correct itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PendingToolCall {
    /// The tool call requested by the model.
    ///
    /// Exactly as the model emitted it — call id, tool name, and JSON input —
    /// to be dispatched against the driver's [`ToolRegistry`](crate::tool::ToolRegistry).
    pub call: ToolCall,

    /// A pre-resolved result for calls suppressed by invalid-tool-call recovery.
    ///
    /// `Some(message)` when the tool name was not in the call's
    /// [`available_tools`](ModelResponse::available_tools): the machine has built a
    /// synthetic error [`Message`] and the driver should feed it back as the
    /// tool result *without* dispatching, so the model learns the name is
    /// unknown and can correct itself. `None` for valid calls, which the driver
    /// dispatches normally.
    pub preresolved_result: Option<Message>,
}

/// A completed model call, fed back to the machine by the driver.
///
/// The driver consumes the provider's stream, accumulates the final assistant
/// [`Message`], and packages it with its usage and stop reason into this
/// struct. Streaming stays driver-side; the machine only ever sees a completed
/// response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelResponse {
    /// The assistant message the model produced.
    ///
    /// The fully-accumulated response, including any text and tool-call [`MessagePart`] parts.
    pub message: Message,

    /// Input tokens reported for this call.
    ///
    /// The prompt-side token count the provider returned for this request.
    pub input_tokens: u64,

    /// Output tokens reported for this call.
    ///
    /// The completion-side token count the provider returned.
    pub output_tokens: u64,

    /// Why the model stopped.
    ///
    /// The provider's [`StopReason`] for this response.
    pub stop_reason: StopReason,

    /// Tool names the driver advertised as available for this call.
    ///
    /// The set of registered tool names the driver sent to the provider for
    /// this turn (e.g. from [`ToolRegistry::tool_names`](crate::tool::ToolRegistry::tool_names)).
    /// A tool call whose name is not in this list is treated as unknown and
    /// given a preresolved error result (see
    /// [`PendingToolCall::preresolved_result`]).
    pub available_tools: Vec<String>,
}

/// The terminal outcome of a run.
///
/// Carried by [`MachineStep::Done`] once [`LoopMachine::next_step`] decides the
/// run is over. Describes *how* it ended and carries the accounting a host needs
/// to log, bill, or report it. Each variant corresponds to one of the machine's
/// terminal conditions; once produced, the outcome is repeated unchanged on
/// every subsequent `next_step` call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MachineOutcome {
    /// The model finished without requesting further tools.
    ///
    /// The normal end state: the last model response carried no tool calls, so
    /// the machine treated it as the final answer. Token and turn totals are
    /// not carried here — the owning `Run` is the
    /// single source for accounting, derived from its `turns` list.
    Completed {
        /// The final text the model produced.
        ///
        /// Concatenation of the text parts of the last assistant message. May be
        /// empty if the model's final message carried only non-text parts.
        final_text: String,
    },

    /// The run was halted because `max_turns` was reached.
    ///
    /// The model kept requesting tools (or otherwise not finishing) until the
    /// turn budget was exhausted, so the machine stopped it rather than loop
    /// indefinitely. Usually a sign the agent isn't converging — raise
    /// `max_turns` or investigate the loop.
    MaxTurnsExceeded,

    /// The run was cancelled.
    ///
    /// A clean, cooperative termination caused by [`LoopMachine::cancel`] (the
    /// driver calls it when its cancellation signal fires). Distinct from
    /// [`Failed`](Self::Failed): cancellation is expected, not an error, and the
    /// conversation may hold partial results in the history. Carries no extra
    /// fields because there is nothing further to report.
    Cancelled,

    /// The run failed with an error.
    ///
    /// A driver-side failure the machine was told about (for example a tool
    /// dispatch or LLM call that errored terminally). The wrapped error is typed
    /// so a host can match on it programmatically rather than parsing a string.
    Failed {
        /// The error that terminated the run.
        ///
        /// A host can pattern-match on the [`LoopError`] to recover or report
        /// the specific failure.
        error: LoopError,
    },
}

/// Where the machine is in its request/respond cycle.
///
/// One value is produced at each point in the loop and is also exposed by
/// [`LoopMachine::state`] and the trait method for inspection.
/// It is the machine's own step-protocol bookkeeping, surfaced to observers so
/// a host can see where a run currently is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineState {
    /// Ready to request the next model call.
    ///
    /// The starting state, and the state the machine returns to after tool
    /// results or a compaction result are fed back — i.e. whenever it is ready
    /// to call the model again. The driver also reports this as the idle
    /// pre-run state before the first turn.
    Start,

    /// A model call has been requested; awaiting the response.
    ///
    /// Entered when the machine emits [`MachineStep::CallLLM`] and left when the
    /// driver feeds the [`ModelResponse`] back via [`LoopMachine::model_response`].
    AwaitingModel {
        /// The 0-indexed turn number in flight.
        ///
        /// Matches the `turn` carried on the outstanding [`MachineStep::CallLLM`].
        turn: usize,
    },

    /// Tool dispatch has been requested; awaiting the results.
    ///
    /// Entered when the machine emits [`MachineStep::CallTools`] and left when
    /// the driver feeds the tool-result messages back via
    /// [`LoopMachine::tool_results`].
    AwaitingTools {
        /// The 0-indexed turn number the tool calls belong to.
        ///
        /// Lets a host correlate a dispatch back to the model response that
        /// requested it.
        turn: usize,
    },

    /// A compaction pass has been requested; awaiting the compacted history.
    ///
    /// Entered when the machine emits [`MachineStep::Compact`] and left when the
    /// driver feeds the compacted history back via
    /// [`LoopMachine::compaction_result`].
    AwaitingCompaction {
        /// Why compaction was triggered.
        ///
        /// The [`CompactReason`] carried on the outstanding [`MachineStep::Compact`].
        reason: CompactReason,
    },

    /// The run has terminated.
    ///
    /// The wrapped [`MachineOutcome`] describes how it ended. Once in this
    /// state, the machine accepts no further input.
    Terminal(MachineOutcome),
}

/// Policy inputs passed to [`LoopMachine::next_step`] on each call.
///
/// The machine is policy-free: it tracks state and reports facts. The turn
/// budget and compaction thresholds live here, not on the machine, so the
/// machine serializes as pure state. Build one from your session/run config
/// and pass it fresh each `next_step` call.
#[derive(Debug, Clone, Copy)]
pub struct MachinePolicy {
    /// Maximum turns before the run is capped.
    ///
    /// The turn budget for this run. The machine checks `turns_taken` against
    /// this on every `next_step` call; reaching the cap ends the run with
    /// [`MachineOutcome::MaxTurnsExceeded`].
    pub max_turns: usize,

    /// Model context window in tokens.
    ///
    /// The compaction trigger compares the current context size against this
    /// value (via `compact_threshold`) and the 95% emergency line.
    pub context_window: u64,

    /// Compaction threshold as a percentage of the context window (0–100).
    ///
    /// When `context_tokens` reaches this percentage of `context_window`, the
    /// machine emits a [`MachineStep::Compact`] with a threshold-exceeded
    /// reason (if `auto_compact` is `true`).
    pub compact_threshold: u8,

    /// Whether automatic compaction is enabled.
    ///
    /// When `true` the machine emits `Compact` steps once the threshold (or the
    /// 95% emergency line) is reached. When `false` only the emergency line
    /// fires; threshold-compaction is left to the driver.
    pub auto_compact: bool,
}

/// The sans-IO agent-loop state machine — pure state, no policy.
///
/// Owns the append-only history and tracks where the loop is in its
/// request/respond cycle. Construct one with
/// [`LoopMachine::from_history`], then repeatedly call
/// [`Self::next_step`] (passing a fresh [`MachinePolicy`]) and feed the
/// result back. The machine performs no IO and stores no policy — it
/// serializes as pure state.
///
/// # Example
///
/// Driving a machine to completion (illustrative — a real driver performs IO
/// in each arm):
///
/// ```no_run
/// # use loopctl::engine::core::{LoopMachine, MachinePolicy, ModelResponse, MachineStep, MachineOutcome};
/// # use loopctl::message::Message;
/// # fn build_response(_: &LoopMachine) -> ModelResponse { unimplemented!() }
/// let policy = MachinePolicy { max_turns: 10, context_window: 200_000, compact_threshold: 80, auto_compact: true };
/// let mut machine = LoopMachine::from_history(vec![Message::user("Hello")]);
/// loop {
///     match machine.next_step(policy) {
///         MachineStep::CallLLM { .. } => {
///             let response = build_response(&machine);
///             machine.model_response(response, 0);
///         }
///         MachineStep::CallTools { .. } => {
///             machine.tool_results(Vec::new());
///         }
///         MachineStep::Compact { .. } => {
///             let measured_before = 80;
///             machine.compaction_result(machine.history().to_vec(), measured_before, 60);
///         }
///         MachineStep::Done(MachineOutcome::Completed { final_text }) => {
///             assert_eq!(final_text, "Hi there");
///             break;
///         }
///         _ => break,
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopMachine {
    /// The committed conversation history from previous runs.
    ///
    /// Immutable during a run — the driver derives the LLM feed from
    /// `history + pending`. Only modified by compaction (which replaces it
    /// wholesale with a compacted version) and by [`commit_pending`]
    /// (which appends the current run's messages on success).
    ///
    /// [`commit_pending`]: Self::commit_pending
    history: Vec<Message>,

    /// Messages accumulated by the current run.
    ///
    /// The user input, assistant responses, and tool results from the
    /// in-flight run. Cleared by [`accept_input`] at the start of each
    /// run. On success, [`commit_pending`] moves these into `history`; on
    /// failure they are discarded so `history` stays clean.
    ///
    /// [`accept_input`]: Self::accept_input
    /// [`commit_pending`]: Self::commit_pending
    pending: Vec<Message>,

    /// Where the machine is in its request/respond cycle.
    ///
    /// One of [`MachineState`]: ready to call the model, awaiting a model
    /// response, awaiting tool results, awaiting a compaction result, or
    /// terminal. Determines which step [`Self::next_step`] emits next.
    state: MachineState,

    /// How many turns have completed.
    ///
    /// The count of model responses accepted so far. The driver checks this
    /// against the turn budget passed to [`next_step`](Self::next_step);
    /// reaching the cap ends the run with
    /// [`MachineOutcome::MaxTurnsExceeded`].
    turns_taken: usize,

    /// Current context-size estimate, in tokens.
    ///
    /// The machine's view of how large the conversation currently is.
    /// [`next_step`](Self::next_step) compares it against the compaction
    /// policy to decide whether to emit a [`MachineStep::Compact`].
    context_tokens: u64,

    /// Whether [`Self::cancel`] has been called.
    ///
    /// A cancellation request from the driver. Once set, the next
    /// [`Self::next_step`] ends the run with [`MachineOutcome::Cancelled`].
    cancelled: bool,

    /// Tool calls queued for the next [`MachineStep::CallTools`].
    ///
    /// The classified calls (valid vs. unknown tool name) the most recent model
    /// response requested, awaiting dispatch. Emitted as a `CallTools` step on
    /// the next [`Self::next_step`].
    pending_tools: Vec<PendingToolCall>,
}

impl LoopMachine {
    /// Construct a machine seeded with an existing conversation history.
    ///
    /// The driver calls this at the top of every `run()`, passing the
    /// session's accumulated messages so the model sees prior turns. For a
    /// fresh single-message conversation, pass `vec![Message::user(prompt)]`.
    ///
    /// The machine starts in pure state: zero turns taken this run, zero
    /// context tokens, not cancelled. The turn budget, compaction window,
    /// and policy knobs are not stored — they are passed to
    /// [`next_step`](Self::next_step) on each call so the machine stays
    /// policy-free and serializes as pure state.
    #[must_use]
    pub fn from_history(history: Vec<Message>) -> Self {
        Self {
            history,
            pending: Vec::new(),
            state: MachineState::Start,
            turns_taken: 0,
            context_tokens: 0,
            cancelled: false,
            pending_tools: Vec::new(),
        }
    }

    /// Begin a new run on this machine, preserving the accumulated history.
    ///
    /// Pushes the user's input message onto the history, then resets the
    /// per-run state (turn counter, context tokens, cancellation flag,
    /// pending tools, state machine position) so the machine is ready for
    /// a fresh `CallLLM` → tool → ... → `Done` cycle. The conversation
    /// history from previous runs is untouched — the model sees the full
    /// cross-run context.
    ///
    /// This replaces the old pattern of cloning `session.history` into a
    /// fresh `LoopMachine::from_history` at every `run()` call.
    pub fn accept_input(&mut self, input: &str) {
        self.pending.clear();
        self.pending.push(Message::user(input));
        self.state = MachineState::Start;
        self.turns_taken = 0;
        self.context_tokens = 0;
        self.cancelled = false;
        self.pending_tools.clear();
    }

    /// Feed a driver-measured context estimate into the machine.
    ///
    /// The estimate is the payload the provider would receive — history
    /// plus the per-request overhead (system prompt, tool schemas) and,
    /// when known ahead of the call, the turn's transient messages
    /// (contributors, retrieved memories). The driver calls this
    /// whenever that payload grows outside a model response — after
    /// [`Self::accept_input`] at run start, after tool results land,
    /// and before a model call that carries fresh transients — so the
    /// first [`Self::next_step`] sees the true size instead of the zero
    /// `accept_input` reset to. Replaces the current estimate with
    /// `tokens`; performs no state transition, so the compaction
    /// trigger evaluates it on the next [`Self::next_step`]. No effect
    /// once the machine is terminal.
    pub fn set_context_tokens(&mut self, tokens: u64) {
        if self.is_terminal() {
            return;
        }
        self.context_tokens = tokens;
    }

    /// Return the next step the driver must perform.
    ///
    /// `policy` supplies the turn budget and compaction thresholds the machine
    /// consults to decide between `CallLLM`, `Compact`, and `MaxTurnsExceeded`.
    /// It is not stored — pass it fresh each call from the session/run config.
    ///
    /// This is pure and idempotent: calling it twice with no intervening feed
    /// method ([`Self::model_response`], [`Self::tool_results`],
    /// [`Self::compaction_result`], [`Self::compaction_noop`],
    /// [`Self::cancel`]) returns an equal step.
    /// Once the machine is terminal, every subsequent call returns
    /// [`MachineStep::Done`] with the same [`MachineOutcome`].
    ///
    /// # Cancellation
    ///
    /// If [`Self::cancel`] has been called, the next call returns
    /// [`MachineStep::Done`] with [`MachineOutcome::Cancelled`].
    pub fn next_step(&mut self, policy: MachinePolicy) -> MachineStep {
        if let MachineState::Terminal(outcome) = &self.state {
            let outcome = outcome.clone();
            return MachineStep::Done(outcome);
        }

        if self.cancelled {
            let outcome = MachineOutcome::Cancelled;
            self.state = MachineState::Terminal(outcome.clone());
            return MachineStep::Done(outcome);
        }

        let turn = match &self.state {
            MachineState::AwaitingModel { turn } => *turn,
            _ => self.turns_taken,
        };

        match self.state.clone() {
            MachineState::Start | MachineState::AwaitingModel { .. } => {
                self.request_model(turn, policy)
            }
            MachineState::AwaitingTools { turn, .. } => {
                let calls = self.pending_tools.clone();
                MachineStep::CallTools { turn, calls }
            }
            MachineState::AwaitingCompaction { reason } => MachineStep::Compact { reason },
            MachineState::Terminal(outcome) => MachineStep::Done(outcome),
        }
    }

    /// Produce the step for a model-call request, honoring the turn budget and
    /// the compaction trigger.
    ///
    /// Returns [`MachineOutcome::MaxTurnsExceeded`] when the budget is spent,
    /// [`MachineStep::Compact`] when the context size has crossed the threshold
    /// and auto-compaction is on, or [`MachineStep::CallLLM`] otherwise.
    fn request_model(&mut self, turn: usize, policy: MachinePolicy) -> MachineStep {
        if self.turns_taken >= policy.max_turns {
            let outcome = MachineOutcome::MaxTurnsExceeded;
            self.state = MachineState::Terminal(outcome.clone());
            return MachineStep::Done(outcome);
        }
        if self.is_emergency(policy) {
            let reason = CompactReason::Emergency;
            self.state = MachineState::AwaitingCompaction { reason };
            return MachineStep::Compact { reason };
        }
        if policy.auto_compact && self.should_compact(policy) {
            let reason = CompactReason::ThresholdExceeded;
            self.state = MachineState::AwaitingCompaction { reason };
            return MachineStep::Compact { reason };
        }
        self.state = MachineState::AwaitingModel { turn };
        MachineStep::CallLLM { turn }
    }

    /// Decide whether the current context size warrants a compaction pass.
    ///
    /// Compares the machine's current context-size estimate against the
    /// configured `compact_threshold` of the model's context window: returns
    /// `true` when the estimate has grown past that fraction of the window,
    /// signalling that the next step should be [`MachineStep::Compact`] rather
    /// than another model call. Returns `false` when compaction is disabled by
    /// a zero threshold or window, or when the context still fits comfortably.
    fn should_compact(&self, policy: MachinePolicy) -> bool {
        if policy.context_window == 0 || policy.compact_threshold == 0 {
            return false;
        }
        let limit = policy
            .context_window
            .saturating_mul(u64::from(policy.compact_threshold))
            / 100;
        self.context_tokens > limit
    }

    /// Whether the context is in the emergency zone (≥ 95% of the window).
    ///
    /// Fires regardless of `auto_compact` or the configured threshold — a
    /// safety net against context overflow.
    fn is_emergency(&self, policy: MachinePolicy) -> bool {
        if policy.context_window == 0 {
            return false;
        }
        self.context_tokens >= policy.context_window.saturating_mul(95) / 100
    }

    /// Feed a completed model call back into the machine.
    ///
    /// The driver calls this after servicing a [`MachineStep::CallLLM`], passing
    /// the model's response. If the response carried no tool calls the run is
    /// complete and the machine becomes terminal with [`MachineOutcome::Completed`];
    /// otherwise the calls are classified (valid vs. unknown tool name) and the
    /// next [`Self::next_step`] will request their dispatch via
    /// [`MachineStep::CallTools`].
    ///
    /// Has no effect once the machine is terminal.
    pub fn model_response(&mut self, response: ModelResponse, context_tokens: u64) {
        if self.is_terminal() {
            return;
        }

        let message = response.message;
        let tool_calls: Vec<ToolCall> = message
            .tool_call_parts()
            .into_iter()
            .map(|(id, tool, input)| ToolCall {
                id: id.to_string(),
                tool: tool.to_string(),
                input: input.clone(),
            })
            .collect();
        let turn_number = self.turns_taken;
        self.pending.push(message);
        self.context_tokens = context_tokens;
        self.turns_taken = self.turns_taken.saturating_add(1);

        if tool_calls.is_empty() {
            let final_text = self
                .pending
                .last()
                .map(Message::text_content)
                .unwrap_or_default();
            let outcome = MachineOutcome::Completed { final_text };
            self.state = MachineState::Terminal(outcome);
            return;
        }

        self.pending_tools = tool_calls
            .into_iter()
            .map(|call| Self::classify(call, &response.available_tools))
            .collect();
        self.state = MachineState::AwaitingTools { turn: turn_number };
    }

    /// Feed tool-result messages back into the machine.
    ///
    /// Each provided [`Message`] is appended to the history, and the pending
    /// tool calls are consumed — this is the one feed that clears them, so a
    /// [`next_step`](Self::next_step) re-poll before it returns the same calls
    /// again, while the step after it advances. After the results are
    /// recorded, the next [`Self::next_step`] requests another
    /// [`MachineStep::CallLLM`] (subject to the max-turn budget and compaction
    /// trigger). Has no effect once the machine is terminal.
    pub fn tool_results(&mut self, messages: Vec<Message>) {
        if self.is_terminal() {
            return;
        }
        self.pending.extend(messages);
        self.pending_tools.clear();
        self.state = MachineState::Start;
    }

    /// Inject an arbitrary message into the history.
    ///
    /// The driver calls this to add a message that did not come from a model
    /// response, a tool result, or a compaction pass — for example a host
    /// steering message. The message becomes part
    /// of the record the driver builds the feed from on the next
    /// [`MachineStep::CallLLM`]. Has no effect once the machine is terminal.
    ///
    /// The context estimate is not refreshed here — the machine keeps no
    /// counter. A caller injecting enough content to matter should pair
    /// this with [`set_context_tokens`](Self::set_context_tokens),
    /// measured by whoever owns the token counter; otherwise the next
    /// step's compaction decision runs against the pre-inject estimate.
    pub fn inject(&mut self, message: Message) {
        if self.is_terminal() {
            return;
        }
        self.pending.push(message);
    }

    /// Feed compacted history back into the machine.
    ///
    /// The driver calls this after servicing a [`MachineStep::Compact`] that
    /// rewrote the history, passing the compacted history plus two measured
    /// estimates from the same counter: `tokens_before` — the payload the
    /// provider would have received ahead of the compaction pass
    /// (history plus the per-request overhead) — and `tokens_after` —
    /// the same measure of `compacted`. The machine adopts
    /// `tokens_after` as the current context size so it does not
    /// immediately request another compaction. The next
    /// [`Self::next_step`] then requests the deferred
    /// [`MachineStep::CallLLM`]. Has no effect once the machine is terminal.
    pub fn compaction_result(
        &mut self,
        compacted: Vec<Message>,
        tokens_before: u64,
        tokens_after: u64,
    ) {
        if self.is_terminal() {
            return;
        }
        if self.terminate_on_no_progress(tokens_before, tokens_after) {
            return;
        }
        self.history = compacted;
        self.pending.clear();
        self.context_tokens = tokens_after;
        self.state = MachineState::Start;
    }

    /// Feed an unchanged compaction result back into the machine.
    ///
    /// The driver calls this after servicing a [`MachineStep::Compact`] that
    /// changed nothing — no compactor ran, a pre-compact hook vetoed the pass,
    /// or the compactor returned the conversation unchanged. The committed
    /// history and the pending buffer are left untouched: feeding the
    /// uncompacted conversation through [`Self::compaction_result`] would
    /// commit the current run's partial messages mid-run, so a later failure
    /// could no longer discard them.
    ///
    /// `tokens_before` and `tokens_after` are the driver's measured estimates
    /// of the conversation ahead of and after the pass (equal in practice,
    /// since nothing changed); the machine adopts `tokens_after` as the
    /// current context size. The same no-progress guard as
    /// [`Self::compaction_result`] applies: when nothing was shaved off, the
    /// machine transitions to [`MachineOutcome::Failed`] with
    /// [`LoopError::ContextExceeded`] — compaction cannot shrink this
    /// conversation, and another model call would exceed the context window.
    /// Has no effect once the machine is terminal.
    pub fn compaction_noop(&mut self, tokens_before: u64, tokens_after: u64) {
        if self.is_terminal() {
            return;
        }
        if self.terminate_on_no_progress(tokens_before, tokens_after) {
            return;
        }
        self.context_tokens = tokens_after;
        self.state = MachineState::Start;
    }

    /// Fail the run when a compaction feed made no progress.
    ///
    /// Shared guard behind [`Self::compaction_result`] and
    /// [`Self::compaction_noop`]: compares the driver's measured post-pass
    /// token count against its measured pre-pass count of the full history.
    /// When nothing was shaved off, transitions to [`MachineOutcome::Failed`]
    /// with [`LoopError::ContextExceeded`] (preventing an infinite compaction
    /// cycle) and returns `true`; returns `false` when the feed may proceed.
    fn terminate_on_no_progress(&mut self, tokens_before: u64, tokens_after: u64) -> bool {
        if tokens_after < tokens_before {
            return false;
        }
        self.state = MachineState::Terminal(MachineOutcome::Failed {
            error: LoopError::ContextExceeded {
                used: tokens_after,
                limit: tokens_before,
            },
        });
        true
    }

    /// Mark the run as cancelled.
    ///
    /// The next [`Self::next_step`] returns [`MachineStep::Done`] with
    /// [`MachineOutcome::Cancelled`]. This is the state-correctness half of
    /// cancellation: a driver that aborts an in-flight step calls this so the
    /// machine's state reflects the abort rather than resuming the partial step.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Record a terminal failure in the machine's state.
    ///
    /// A driver that hits an unrecoverable error (stream failure, dispatch
    /// error, compaction overflow) calls this so the machine transitions to
    /// [`MachineState::Terminal`] with [`MachineOutcome::Failed`] carrying
    /// `error`. Without this, a serialized machine that errored has no failure
    /// record and would resume as if the error never happened. The driver
    /// still returns the error from `run()`; this call ensures the machine's
    /// own state agrees. Has no effect once the machine is already terminal.
    pub fn fail(&mut self, error: LoopError) {
        if self.is_terminal() {
            return;
        }
        self.state = MachineState::Terminal(MachineOutcome::Failed { error });
    }

    /// Whether [`Self::cancel`] has been called.
    ///
    /// When `true`, the next [`Self::next_step`] goes terminal with
    /// [`MachineOutcome::Cancelled`]. Reports the cancellation request, not
    /// whether the machine is already terminal — use [`Self::is_terminal`] for
    /// that.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// The committed conversation history from previous runs.
    ///
    /// Does not include the current run's pending messages. Use
    /// [`full_history`](Self::full_history) when you need the complete
    /// conversation for an API call.
    #[must_use]
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// The full conversation: committed history plus the current run's pending messages.
    ///
    /// Allocates a merged `Vec` each call — use when sending the outbound
    /// `StreamRequest` or when a contributor needs to see the full context.
    #[must_use]
    pub fn full_history(&self) -> Vec<Message> {
        let mut merged = self.history.clone();
        merged.extend_from_slice(&self.pending);
        merged
    }

    /// Move the current run's pending messages into the committed history.
    ///
    /// Called by the driver on successful run completion. After this,
    /// the pending buffer is empty and the committed history contains the
    /// full conversation. On failure the driver simply clears pending
    /// via [`accept_input`](Self::accept_input) on the next run,
    /// leaving the committed history untouched.
    pub fn commit_pending(&mut self) {
        self.history.append(&mut self.pending);
    }

    /// Discard the current run's pending messages without committing.
    ///
    /// Called by the driver on run failure. The committed history stays
    /// clean — no orphaned tool calls, model responses, or tool results
    /// from the abandoned run leak into the next run's context.
    pub fn discard_pending(&mut self) {
        self.pending.clear();
    }

    /// The current state of the step protocol.
    ///
    /// A snapshot of where the machine is in its request/respond cycle — useful
    /// for diagnostics, or for a driver that needs to know which feed method is
    /// currently expected.
    #[must_use]
    pub fn state(&self) -> MachineState {
        self.state.clone()
    }

    /// Number of turns completed so far.
    ///
    /// How many model responses the machine has accepted. Also carried in
    /// [`MachineOutcome::Completed`] and [`MachineOutcome::MaxTurnsExceeded`]
    /// once the run ends.
    #[must_use]
    pub fn turns_taken(&self) -> usize {
        self.turns_taken
    }

    /// Whether the machine has reached a terminal outcome.
    ///
    /// `true` once the machine has emitted a [`MachineOutcome`] (completion,
    /// max-turns, cancellation, or failure). Once terminal, the feed methods
    /// (`model_response`, `tool_results`, `compaction_result`) are no-ops and
    /// [`Self::next_step`] keeps returning [`MachineStep::Done`] with the same
    /// outcome.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, MachineState::Terminal(_))
    }

    /// Classify a tool call against the advertised available tool names.
    ///
    /// Known names yield a plain [`PendingToolCall`] (`preresolved_result:
    /// None`); unknown names get a synthetic error result so the driver can feed
    /// it back without dispatching.
    fn classify(call: ToolCall, available: &[String]) -> PendingToolCall {
        let known = available.iter().any(|name| name == &call.tool);
        if known {
            return PendingToolCall {
                call,
                preresolved_result: None,
            };
        }
        let result = Self::unknown_tool_result(&call);
        PendingToolCall {
            call,
            preresolved_result: Some(result),
        }
    }

    /// Build the tool-result message returned for an unknown tool call.
    ///
    /// Constructs a user-role [`Message`] carrying a single error tool-result
    /// part that echoes the call's id and names the tool as unavailable. This is
    /// the value placed on [`PendingToolCall::preresolved_result`] so a driver
    /// can feed it straight back to the model — without dispatching — letting
    /// the model learn the name is unknown and correct itself on the next turn.
    fn unknown_tool_result(call: &ToolCall) -> Message {
        let message = format!("tool '{}' is not available", call.tool);
        Message::new(
            Role::User,
            vec![MessagePart::tool_result(
                call.id.clone(),
                call.tool.clone(),
                ToolContent::Text(message),
                true,
            )],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn small_machine() -> LoopMachine {
        LoopMachine::from_history(vec![Message::user("hello")])
    }

    fn test_policy(max_turns: usize) -> MachinePolicy {
        MachinePolicy {
            max_turns,
            context_window: 200_000,
            compact_threshold: 80,
            auto_compact: true,
        }
    }

    fn text_response(text: &str, input_tokens: u64, output_tokens: u64) -> ModelResponse {
        ModelResponse {
            message: Message::assistant(text),
            input_tokens,
            output_tokens,
            stop_reason: StopReason::EndTurn,
            available_tools: Vec::new(),
        }
    }

    fn tool_response(tool: &str, available: &[&str], input_tokens: u64) -> ModelResponse {
        let part = MessagePart::tool_call("call_1", tool, Value::Object(serde_json::Map::new()));
        ModelResponse {
            message: Message::new(Role::Assistant, vec![part]),
            input_tokens,
            output_tokens: 10,
            stop_reason: StopReason::ToolCall,
            available_tools: available.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// A string long enough to exceed the compaction threshold in tests that
    /// use `context_window: 100, compact_threshold: 50` (trigger at 50 tokens
    /// = 200 chars). The machine estimates tokens from message text now, not
    /// from the provider's `input_tokens` field.
    fn long_text(n: usize) -> String {
        "x".repeat(n)
    }

    fn count_tokens(machine: &LoopMachine) -> u64 {
        use crate::compact::TokenCounter;
        crate::compact::HeuristicTokenCounter.count(&machine.full_history())
    }

    fn count_vec_tokens(messages: &[Message]) -> u64 {
        crate::compact::CompactionOutcome::estimate_tokens(messages)
    }

    fn same_step(a: &MachineStep, b: &MachineStep) -> bool {
        serde_json::to_string(a).unwrap_or_default() == serde_json::to_string(b).unwrap_or_default()
    }

    #[test]
    fn calling_llm_from_new_emits_call_llm_step() {
        let mut machine = LoopMachine::from_history(vec![Message::user("hello")]);
        let step = machine.next_step(test_policy(5));
        let MachineStep::CallLLM { turn } = step else {
            panic!("expected CallLLM, got {step:?}");
        };
        assert_eq!(turn, 0);
        assert_eq!(machine.state(), MachineState::AwaitingModel { turn: 0 });
    }

    #[test]
    fn machine_api_has_no_async_no_tokio_no_apiclient() {
        let mut machine = LoopMachine::from_history(vec![Message::user("hello")]);
        assert!(matches!(
            machine.next_step(test_policy(5)),
            MachineStep::CallLLM { .. }
        ));
        machine.model_response(text_response("hi", 5, 3), 0);
        // A text-only turn completes the run, so the machine is terminal.
        assert!(machine.is_terminal());
        assert_eq!(machine.turns_taken(), 1);
        assert_eq!(machine.full_history().len(), 2);
        assert!(matches!(
            machine.state(),
            MachineState::Terminal(MachineOutcome::Completed { .. })
        ));
        machine.cancel();
        assert!(machine.is_cancelled());
    }

    #[test]
    fn resume_after_model_response_round_trips() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.model_response(tool_response("echo", &["echo"], 10), 0);
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step(test_policy(5));
        let b = restored.next_step(test_policy(5));
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn resume_after_tool_results_round_trips() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.model_response(tool_response("echo", &["echo"], 10), 0);
        let step = machine.next_step(test_policy(5));
        let MachineStep::CallTools { turn: _, calls } = &step else {
            panic!("expected CallTools, got {step:?}");
        };
        let results: Vec<Message> = calls
            .iter()
            .map(|c| {
                Message::new(
                    Role::User,
                    vec![MessagePart::tool_result(
                        c.call.id.clone(),
                        c.call.tool.clone(),
                        ToolContent::Text("ok".to_string()),
                        false,
                    )],
                )
            })
            .collect();
        machine.tool_results(results);
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step(test_policy(5));
        let b = restored.next_step(test_policy(5));
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn resume_after_compaction_result_round_trips() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        let _ = machine.next_step(policy); // CallLLM
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine)); // AwaitingTools
        let _ = machine.next_step(policy); // CallTools
        machine.tool_results(vec![Message::user(long_text(250))]); // → Start
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::Compact { .. }
        ));
        let compacted = vec![Message::user("compacted")];
        let tokens_after = count_vec_tokens(&compacted);
        machine.compaction_result(compacted, count_tokens(&machine), tokens_after);
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step(policy);
        let b = restored.next_step(policy);
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn max_turns_enforced_by_machine() {
        let mut machine = small_machine();
        // Turn 0.
        assert!(matches!(
            machine.next_step(test_policy(2)),
            MachineStep::CallLLM { turn: 0 }
        ));
        machine.model_response(tool_response("echo", &["echo"], 0), 0);
        let _ = machine.next_step(test_policy(2));
        machine.tool_results(vec![Message::user("r")]);
        // Turn 1.
        assert!(matches!(
            machine.next_step(test_policy(2)),
            MachineStep::CallLLM { turn: 1 }
        ));
        machine.model_response(tool_response("echo", &["echo"], 1), 0);
        let _ = machine.next_step(test_policy(2));
        machine.tool_results(vec![Message::user("r")]);
        // Turn 3 must be denied.
        match machine.next_step(test_policy(2)) {
            MachineStep::Done(MachineOutcome::MaxTurnsExceeded) => {}
            other => panic!("expected MaxTurnsExceeded, got {other:?}"),
        }
        assert!(machine.is_terminal());
    }

    #[test]
    fn cancel_returns_done_cancelled_at_next_step() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.cancel();
        match machine.next_step(test_policy(5)) {
            MachineStep::Done(MachineOutcome::Cancelled) => {}
            other => panic!("expected Done(Cancelled), got {other:?}"),
        }
        // Idempotent: stays terminal-cancelled.
        assert!(matches!(
            machine.next_step(test_policy(5)),
            MachineStep::Done(MachineOutcome::Cancelled)
        ));
    }

    #[test]
    fn fail_returns_done_failed_at_next_step() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        let err = LoopError::Api("stream failed".to_string());
        machine.fail(err.clone());
        match machine.next_step(test_policy(5)) {
            MachineStep::Done(MachineOutcome::Failed { error }) => {
                assert_eq!(error, err);
            }
            other => panic!("expected Done(Failed), got {other:?}"),
        }
        assert!(machine.is_terminal());
    }

    #[test]
    fn failed_outcome_survives_round_trip() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        let err = LoopError::Api("stream failed".to_string());
        machine.fail(err.clone());
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        match restored.next_step(test_policy(5)) {
            MachineStep::Done(MachineOutcome::Failed { error }) => {
                assert_eq!(error, err, "failure record survives round-trip");
            }
            other => panic!("expected Done(Failed) after resume, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tool_call_gets_preresolved_result() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.model_response(tool_response("ghost", &["echo", "ls"], 3), 0);
        let step = machine.next_step(test_policy(5));
        let MachineStep::CallTools { turn: _, calls } = step else {
            panic!("expected CallTools, got {step:?}");
        };
        let call = calls.first().expect("one call");
        let result = call
            .preresolved_result
            .as_ref()
            .expect("unknown tool has a preresolved result");
        assert_eq!(result.role, Role::User);
        assert!(result.parts.iter().any(|p| match p {
            MessagePart::ToolResult { is_error, .. } => *is_error == Some(true),
            _ => false,
        }));
    }

    #[test]
    fn known_tool_call_emits_plain_pending_call() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.model_response(tool_response("echo", &["echo", "ls"], 3), 0);
        let step = machine.next_step(test_policy(5));
        let MachineStep::CallTools { turn: _, calls } = step else {
            panic!("expected CallTools, got {step:?}");
        };
        let call = calls.first().expect("one call");
        assert!(
            call.preresolved_result.is_none(),
            "known tool has no preresolved result"
        );
    }

    #[test]
    fn compaction_triggered_when_tokens_exceed_threshold() {
        // window = 100, threshold = 50% ⇒ compact once estimate > 50 tokens.
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::CallLLM { .. }
        ));
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::CallTools { .. }
        ));
        machine.tool_results(vec![Message::user(long_text(250))]);
        match machine.next_step(policy) {
            MachineStep::Compact { reason } => {
                assert_eq!(reason, CompactReason::ThresholdExceeded);
            }
            other => panic!("expected Compact, got {other:?}"),
        }
    }

    #[test]
    fn emergency_compaction_fires_at_95_percent() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: false,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(400))]);
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::CallLLM { .. }
        ));
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::CallTools { .. }
        ));
        machine.tool_results(vec![Message::user(long_text(400))]);
        match machine.next_step(policy) {
            MachineStep::Compact { reason } => {
                assert_eq!(reason, CompactReason::Emergency);
            }
            other => panic!("expected emergency Compact, got {other:?}"),
        }
    }

    #[test]
    fn compaction_result_replaces_history() {
        // window = 100, threshold = 50 ⇒ compact once tokens > 50.
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        // Drive a tool-call turn past the threshold so the next step compacts.
        let _ = machine.next_step(policy); // CallLLM
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine)); // AwaitingTools
        let _ = machine.next_step(policy); // CallTools
        machine.tool_results(vec![Message::user(long_text(250))]); // → Start
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::Compact { .. }
        ));
        let compacted = vec![Message::user("compacted-only")];
        let tokens_after = count_vec_tokens(&compacted);
        machine.compaction_result(compacted.clone(), count_tokens(&machine), tokens_after);
        // Compare by serialized form: Message is not PartialEq.
        let got = serde_json::to_string(&machine.full_history()).expect("serialize history");
        let want = serde_json::to_string(&compacted).expect("serialize expected");
        assert_eq!(got, want, "history must be replaced by the compacted slice");
        // After compaction the next step is the deferred CallLLM.
        assert!(matches!(
            machine.next_step(test_policy(5)),
            MachineStep::CallLLM { .. }
        ));
    }

    #[test]
    fn compaction_includes_pending_messages() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user("from previous run")]);
        machine.accept_input("current run input");
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        let _ = machine.next_step(policy);
        machine.tool_results(vec![Message::user("tool-out")]);

        let full = machine.full_history();
        assert!(
            full.len() >= 4,
            "full_history must include both committed history and pending messages; \
             got {} messages",
            full.len()
        );
    }

    #[test]
    fn compaction_result_clears_pending() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        let _ = machine.next_step(policy);
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        let _ = machine.next_step(policy);
        machine.tool_results(vec![Message::user(long_text(250))]);
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::Compact { .. }
        ));
        let compacted = vec![Message::user("compacted")];
        let tokens_after = count_vec_tokens(&compacted);
        machine.compaction_result(compacted, count_tokens(&machine), tokens_after);

        assert_eq!(
            machine.history().len(),
            1,
            "history must contain only the compacted message"
        );
        assert_eq!(
            machine.full_history().len(),
            1,
            "pending must be cleared after compaction"
        );
    }

    #[test]
    fn compaction_no_progress_terminates_not_loops() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        let _ = machine.next_step(policy);
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        let _ = machine.next_step(policy);
        machine.tool_results(vec![Message::user(long_text(250))]);
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::Compact { .. }
        ));
        let tokens_before = count_tokens(&machine);
        machine.compaction_result(
            vec![Message::user("compacted")],
            tokens_before,
            tokens_before,
        );

        match machine.next_step(policy) {
            MachineStep::Done(MachineOutcome::Failed {
                error: LoopError::ContextExceeded { .. },
            }) => {}
            other => {
                panic!("no-progress compaction must terminate with ContextExceeded, got {other:?}")
            }
        }
    }

    #[test]
    fn compaction_progress_continues_normally() {
        let policy = MachinePolicy {
            max_turns: 5,
            context_window: 100,
            compact_threshold: 50,
            auto_compact: true,
        };
        let mut machine = LoopMachine::from_history(vec![Message::user(long_text(250))]);
        let _ = machine.next_step(policy);
        machine.model_response(tool_response("echo", &["echo"], 0), count_tokens(&machine));
        let _ = machine.next_step(policy);
        machine.tool_results(vec![Message::user(long_text(250))]);
        assert!(matches!(
            machine.next_step(policy),
            MachineStep::Compact { .. }
        ));
        let tokens_before = count_tokens(&machine);
        machine.compaction_result(vec![Message::user("compacted")], tokens_before, 30);

        assert!(
            matches!(machine.next_step(policy), MachineStep::CallLLM { .. }),
            "compaction that reduced tokens must continue to CallLLM"
        );
    }

    #[test]
    fn history_accumulates_user_assistant_tool_round() {
        let mut machine = small_machine();
        let _ = machine.next_step(test_policy(5));
        machine.model_response(tool_response("echo", &["echo"], 1), 0);
        let step = machine.next_step(test_policy(5));
        let MachineStep::CallTools { turn: _, calls } = step else {
            panic!("expected CallTools, got {step:?}");
        };
        let result = Message::new(
            Role::User,
            calls
                .iter()
                .map(|c| {
                    MessagePart::tool_result(
                        c.call.id.clone(),
                        c.call.tool.clone(),
                        ToolContent::Text("ok".to_string()),
                        false,
                    )
                })
                .collect(),
        );
        machine.tool_results(vec![result]);

        let roles: Vec<Role> = machine.full_history().iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![Role::User, Role::Assistant, Role::User],
            "history must be [user, assistant, tool_result(user)]"
        );
    }

    #[test]
    fn compact_reason_is_serde() {
        for reason in [
            CompactReason::ThresholdExceeded,
            CompactReason::Emergency,
            CompactReason::Manual,
        ] {
            let text = serde_json::to_string(&reason).expect("serialize");
            let back: CompactReason = serde_json::from_str(&text).expect("deserialize");
            assert_eq!(back, reason);
        }
    }

    fn make_call() -> ToolCall {
        ToolCall {
            id: "test".to_string(),
            tool: "Read".to_string(),
            input: serde_json::json!({"path": "/tmp"}),
        }
    }

    #[test]
    fn input_fix_accepts_json_object() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = ToolCall {
            id: "test".to_string(),
            tool: "Read".to_string(),
            input: serde_json::json!({"path": "/tmp"}),
        };
        let correction = Correction {
            correction_type: CorrectionType::InputFix,
            description: "fix path".into(),
            modified_input: Some(serde_json::json!({"path": "/tmp/fixed"})),
            alternative_tool: None,
            guidance: None,
        };
        let result = call.apply_correction(&correction);
        assert!(matches!(result, CorrectionResult::Applied));
        assert_eq!(call.input, serde_json::json!({"path": "/tmp/fixed"}));
    }

    #[test]
    fn input_fix_fails_when_modified_input_missing() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = make_call();
        let correction = Correction {
            correction_type: CorrectionType::InputFix,
            description: "fix path".into(),
            modified_input: None,
            alternative_tool: None,
            guidance: None,
        };
        let result = call.apply_correction(&correction);
        assert!(matches!(result, CorrectionResult::Failed(_)));
    }

    #[test]
    fn input_fix_fails_when_modified_input_not_object() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = make_call();
        let correction = Correction {
            correction_type: CorrectionType::InputFix,
            description: "fix path".into(),
            modified_input: Some(serde_json::json!("not an object")),
            alternative_tool: None,
            guidance: None,
        };
        let result = call.apply_correction(&correction);
        assert!(matches!(result, CorrectionResult::Failed(_)));
    }

    #[test]
    fn tool_change_swaps_tool_name() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = make_call();
        let correction = Correction {
            correction_type: CorrectionType::ToolChange,
            description: "use alt tool".into(),
            modified_input: None,
            alternative_tool: Some("Write".into()),
            guidance: None,
        };
        let result = call.apply_correction(&correction);
        assert!(matches!(result, CorrectionResult::Applied));
        assert_eq!(call.tool, "Write");
    }

    #[test]
    fn tool_change_fails_without_alternative() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = make_call();
        let correction = Correction {
            correction_type: CorrectionType::ToolChange,
            description: "use alt tool".into(),
            modified_input: None,
            alternative_tool: None,
            guidance: None,
        };
        let result = call.apply_correction(&correction);
        assert!(matches!(result, CorrectionResult::Failed(_)));
    }

    #[test]
    fn prerequisite_fix_approach_change_escalate_all_skip() {
        use crate::reflection::{Correction, CorrectionResult, CorrectionType};
        let mut call = make_call();
        for ct in [
            CorrectionType::PrerequisiteFix,
            CorrectionType::ApproachChange,
            CorrectionType::Escalate,
        ] {
            let correction = Correction {
                correction_type: ct,
                description: "n/a".into(),
                modified_input: None,
                alternative_tool: None,
                guidance: None,
            };
            let result = call.apply_correction(&correction);
            assert!(
                matches!(result, CorrectionResult::Skipped),
                "{ct:?} should skip"
            );
        }
    }

    #[test]
    fn configs_carry_no_model_field() {
        use crate::config::SessionConfig;
        let session = SessionConfig::default();
        let _: &Option<String> = &session.system_prompt;
        let _: u64 = session.context_window;

        let run = crate::engine::RunConfig::default();
        let _: usize = run.max_turns;
        let _ = run.parallel_tool_dispatch;
    }

    #[test]
    fn repolling_next_step_returns_the_same_call_tools_step() {
        let mut machine = small_machine();
        let step = machine.next_step(test_policy(5));
        assert!(matches!(step, MachineStep::CallLLM { .. }));
        machine.model_response(tool_response("Read", &["Read"], 10), 10);

        let first = machine.next_step(test_policy(5));
        let MachineStep::CallTools { calls, .. } = &first else {
            panic!("expected CallTools, got {first:?}");
        };
        assert_eq!(calls.len(), 1);

        let second = machine.next_step(test_policy(5));
        assert!(
            same_step(&first, &second),
            "next_step doc: pure and idempotent — a repoll returned {second:?}"
        );

        machine.tool_results(vec![Message::user("result")]);
        let after_feed = machine.next_step(test_policy(5));
        assert!(
            matches!(after_feed, MachineStep::CallLLM { .. }),
            "tool_results consumes the pending calls — the next step advances, got {after_feed:?}"
        );
    }
}
