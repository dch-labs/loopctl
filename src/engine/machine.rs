//! Sans-IO agent-loop state machine.
//!
//! [`LoopMachine`] owns every *decision* the agent loop makes — turn counting,
//! max-turn enforcement, tool-call validity, stop-reason routing, compaction
//! triggering, history accumulation, and the cancellation flag — and performs
//! zero IO. It is advanced by feeding it outcomes via its methods; a thin IO
//! driver (the `BareLoop` in [`crate::engine`]) runs a `match
//! machine.next_step()` loop, performs the side effect each step requests, and
//! feeds the result back.
//!
//! Because the machine is pure and [`Serialize`] + [`Deserialize`], a run can be
//! serialized mid-flight and resumed in another process.
//!
//! [`Serialize`]: serde::Serialize
//! [`Deserialize`]: serde::Deserialize

use serde::{Deserialize, Serialize};

use crate::compact::types::CompactReason;
use crate::config::ParallelDispatchConfig;
use crate::engine::loop_core::{StopReason, ToolCall};
use crate::error::LoopError;
use crate::message::{Message, MessagePart, Role, ToolContent};

/// Per-run configuration.
///
/// The slice of agent configuration that varies across `run()` calls on the
/// same agent (turn/token budgets, compaction policy, dispatch mode). It is
/// owned by the machine so that the machine's decisions are self-contained and
/// a serialized machine carries its configuration with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Maximum number of turns before forcing completion.
    ///
    /// A safety cap on runaway loops: once the machine has taken this many
    /// turns without the model finishing, the run ends with
    /// [`MachineOutcome::MaxTurnsExceeded`]. Defaults to `200`. Set it per-run
    /// to bound long-running tool loops.
    pub max_turns: usize,

    /// Maximum tokens for each API response.
    ///
    /// The per-response cap the driver sends to the provider, limiting the
    /// length of a single model reply. Defaults to `16_384`.
    pub max_tokens: u32,

    /// Context window size in tokens.
    ///
    /// The hard upper bound on tokens the model accepts in one request. The
    /// machine uses it as the denominator for the compaction trigger: the
    /// [`MachineStep::Compact`] step fires once estimated context size reaches
    /// [`compact_threshold`](Self::compact_threshold) of this window. Defaults
    /// to `200_000`; should match the configured model's real window.
    pub context_window: u64,

    /// Threshold to trigger auto-compaction, in basis points (`0–10_000`;
    /// `10_000` = 100% of the context window).
    ///
    /// The machine emits [`MachineStep::Compact`] once the current context size
    /// exceeds this fraction of [`context_window`](Self::context_window).
    /// Defaults to `8_000` (80%). Only consulted when
    /// [`auto_compact`](Self::auto_compact) is `true`.
    pub compact_threshold: u16,

    /// Whether auto-compaction is enabled.
    ///
    /// When `false`, the machine never emits [`MachineStep::Compact`] on its
    /// own — the host must manage context size manually (useful in tests or
    /// fixed-length runs). When `true` (the default), the machine gates
    /// compaction with [`compact_threshold`](Self::compact_threshold).
    pub auto_compact: bool,

    /// How independent tool calls within a single turn are dispatched.
    ///
    /// Controls whether the driver runs a turn's independent tool calls
    /// sequentially or in parallel, up to a concurrency limit. See
    /// [`ParallelDispatchConfig`].
    pub parallel_tool_dispatch: ParallelDispatchConfig,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_turns: 200,
            max_tokens: 16_384,
            context_window: 200_000,
            compact_threshold: 8_000,
            auto_compact: true,
            parallel_tool_dispatch: ParallelDispatchConfig::default(),
        }
    }
}

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
    /// [`ModelTurn`] back via [`LoopMachine::model_response`]. `turn` is the
    /// 1-indexed number of the turn being requested.
    CallLLM {
        /// The 1-indexed turn number being requested.
        ///
        /// Starts at `1` on the first call after construction and increments
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
/// driver reported as available ([`ModelTurn::available_tools`]). For a name the
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
    /// [`available_tools`](ModelTurn::available_tools): the machine has built a
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
/// turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelTurn {
    /// The assistant message the model produced.
    ///
    /// The fully-accumulated response, including any text and tool-call
    /// [`MessagePart`] parts.
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
    /// the machine treated it as the final answer. The captured totals let a
    /// host record exactly what the completed run cost.
    Completed {
        /// The final text the model produced.
        ///
        /// Concatenation of the text parts of the last assistant message. May be
        /// empty if the model's final message carried only non-text parts.
        final_text: String,

        /// Total input tokens consumed across the run.
        ///
        /// Sum of every turn's [`ModelTurn::input_tokens`].
        total_input_tokens: u64,

        /// Total output tokens consumed across the run.
        ///
        /// Sum of every turn's [`ModelTurn::output_tokens`].
        total_output_tokens: u64,

        /// Number of turns executed.
        ///
        /// How many model calls the run made before finishing.
        turns_taken: usize,
    },

    /// The run was halted because `max_turns` was reached.
    ///
    /// The model kept requesting tools (or otherwise not finishing) until the
    /// turn budget was exhausted, so the machine stopped it rather than loop
    /// indefinitely. Usually a sign the agent isn't converging — raise
    /// [`max_turns`](RunConfig::max_turns) or investigate the loop.
    MaxTurnsExceeded {
        /// Number of turns executed before the limit was hit.
        ///
        /// Equal to the configured [`RunConfig::max_turns`].
        turns_taken: usize,
    },

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
/// [`LoopMachine::state`] for inspection. It is the machine's own step-protocol
/// bookkeeping — distinct from the public `LoopState` observation surface, which
/// a driver derives for observers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineState {
    /// Ready to request the next model call.
    ///
    /// The starting state, and the state the machine returns to after tool
    /// results or a compaction result are fed back — i.e. whenever it is ready
    /// to call the model again.
    Start,

    /// A model call has been requested; awaiting the response.
    ///
    /// Entered when the machine emits [`MachineStep::CallLLM`] and left when the
    /// driver feeds the [`ModelTurn`] back via [`LoopMachine::model_response`].
    AwaitingModel {
        /// The 1-indexed turn number in flight.
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
        /// The 1-indexed turn number the tool calls belong to.
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

/// The sans-IO agent-loop state machine.
///
/// Owns the append-only history and every decision the loop makes. Construct
/// one with [`LoopMachine::new`], then repeatedly call [`Self::next_step`] and
/// feed the result back. The machine performs no IO; a driver is required to
/// actually call the LLM, dispatch tools, and compact context.
///
/// # Example
///
/// Driving a machine to completion (illustrative — a real driver performs IO in
/// each arm):
///
/// ```no_run
/// # use loopctl::engine::machine::{LoopMachine, RunConfig, ModelTurn, MachineStep, MachineOutcome};
/// # fn build_turn(_: &LoopMachine) -> ModelTurn { unimplemented!() }
/// let mut machine = LoopMachine::new(RunConfig::default(), "Hello");
/// loop {
///     match machine.next_step() {
///         MachineStep::CallLLM { .. } => {
///             let turn = build_turn(&machine);
///             machine.model_response(turn);
///         }
///         MachineStep::CallTools { .. } => {
///             machine.tool_results(Vec::new());
///         }
///         MachineStep::Compact { .. } => {
///             machine.compaction_result(machine.history().to_vec(), 0);
///         }
///         MachineStep::Done(MachineOutcome::Completed { final_text, .. }) => {
///             assert_eq!(final_text, "Hi there");
///             break;
///         }
///         _ => break,
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopMachine {
    /// The per-run configuration the machine decides against.
    ///
    /// The turn budget, compaction policy, and dispatch mode that govern this
    /// run. Owned by the machine so it travels with a serialized instance.
    config: RunConfig,

    /// The append-only conversation history.
    ///
    /// The complete record of the run: the opening user message, every
    /// assistant response, and every tool result. The driver derives the LLM
    /// feed from it on each call; compaction replaces it wholesale.
    history: Vec<Message>,

    /// Where the machine is in its request/respond cycle.
    ///
    /// One of [`MachineState`]: ready to call the model, awaiting a model
    /// response, awaiting tool results, awaiting a compaction result, or
    /// terminal. Determines which step [`Self::next_step`] emits next.
    state: MachineState,

    /// How many turns have completed.
    ///
    /// The count of model responses accepted so far. Bounded by
    /// [`RunConfig::max_turns`]; reaching that cap ends the run with
    /// [`MachineOutcome::MaxTurnsExceeded`].
    turns_taken: usize,

    /// Cumulative input tokens across the run, for accounting.
    ///
    /// Total prompt-side token consumption from every accepted turn. Reported
    /// in [`MachineOutcome::Completed`] when the run finishes. Distinct from
    /// [`context_tokens`](Self::context_tokens), which tracks current size for
    /// the compaction trigger rather than lifetime consumption.
    total_input_tokens: u64,

    /// Cumulative output tokens across the run, for accounting.
    ///
    /// Total generation-side token consumption from every accepted turn.
    /// Reported in [`MachineOutcome::Completed`] when the run finishes.
    total_output_tokens: u64,

    /// Current context-size estimate, in tokens.
    ///
    /// The machine's view of how large the conversation currently is, used to
    /// decide when to compact. Compared against
    /// [`compact_threshold`](RunConfig::compact_threshold) of
    /// [`context_window`](RunConfig::context_window); crossing it triggers a
    /// [`MachineStep::Compact`].
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
    /// Construct a machine for a new run with the given `user_input`.
    ///
    /// The user's message is appended to the (initially empty) history as the
    /// first entry, and the first [`Self::next_step`] call requests the initial
    /// [`MachineStep::CallLLM`] at turn 1.
    #[must_use]
    pub fn new(config: RunConfig, user_input: impl Into<String>) -> Self {
        Self {
            config,
            history: vec![Message::user(user_input)],
            state: MachineState::Start,
            turns_taken: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            context_tokens: 0,
            cancelled: false,
            pending_tools: Vec::new(),
        }
    }

    /// The configuration the machine makes its decisions against.
    ///
    /// The driver reads it to build provider requests (e.g.
    /// [`RunConfig::max_tokens`], [`RunConfig::parallel_tool_dispatch`]).
    #[must_use]
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// Return the next step the driver must perform.
    ///
    /// This is pure and idempotent: calling it twice with no intervening feed
    /// method ([`Self::model_response`], [`Self::tool_results`],
    /// [`Self::compaction_result`], [`Self::cancel`]) returns an equal step.
    /// Once the machine is terminal, every subsequent call returns
    /// [`MachineStep::Done`] with the same [`MachineOutcome`].
    ///
    /// # Cancellation
    ///
    /// If [`Self::cancel`] has been called, the next call returns
    /// [`MachineStep::Done`] with [`MachineOutcome::Cancelled`].
    pub fn next_step(&mut self) -> MachineStep {
        if let MachineState::Terminal(outcome) = &self.state {
            let outcome = outcome.clone();
            return MachineStep::Done(outcome);
        }

        if self.cancelled {
            let outcome = MachineOutcome::Cancelled;
            self.state = MachineState::Terminal(outcome.clone());
            return MachineStep::Done(outcome);
        }

        let turn = self.turns_taken.saturating_add(1);
        match self.state.clone() {
            MachineState::Start | MachineState::AwaitingModel { .. } => self.request_model(turn),
            MachineState::AwaitingTools { .. } => {
                let calls = std::mem::take(&mut self.pending_tools);
                MachineStep::CallTools { calls }
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
    fn request_model(&mut self, turn: usize) -> MachineStep {
        if self.turns_taken >= self.config.max_turns {
            let turns_taken = self.turns_taken;
            let outcome = MachineOutcome::MaxTurnsExceeded { turns_taken };
            self.state = MachineState::Terminal(outcome.clone());
            return MachineStep::Done(outcome);
        }
        if self.config.auto_compact && self.should_compact() {
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
    /// configured [`compact_threshold`](RunConfig::compact_threshold) of the
    /// [`context_window`](RunConfig::context_window): returns `true` when the
    /// estimate has grown past that fraction of the window, signalling that the
    /// next step should be [`MachineStep::Compact`] rather than another model
    /// call. Returns `false` when compaction is disabled by a zero threshold or
    /// window, or when the context still fits comfortably.
    fn should_compact(&self) -> bool {
        const BASIS: u128 = 10_000;
        if self.config.context_window == 0 || self.config.compact_threshold == 0 {
            return false;
        }
        let limit = u128::from(self.config.compact_threshold)
            .saturating_mul(u128::from(self.config.context_window));
        u128::from(self.context_tokens).saturating_mul(BASIS) > limit
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
    pub fn model_response(&mut self, turn: ModelTurn) {
        if self.is_terminal() {
            return;
        }

        let message = turn.message;
        let tool_calls = Self::extract_tool_calls(&message);
        self.history.push(message);
        self.total_input_tokens = self.total_input_tokens.saturating_add(turn.input_tokens);
        self.total_output_tokens = self.total_output_tokens.saturating_add(turn.output_tokens);
        self.context_tokens = turn.input_tokens;
        self.turns_taken = self.turns_taken.saturating_add(1);

        if tool_calls.is_empty() {
            let final_text = self
                .history
                .last()
                .map(Self::extract_text)
                .unwrap_or_default();
            let outcome = MachineOutcome::Completed {
                final_text,
                total_input_tokens: self.total_input_tokens,
                total_output_tokens: self.total_output_tokens,
                turns_taken: self.turns_taken,
            };
            self.state = MachineState::Terminal(outcome);
            return;
        }

        let turn_number = self.turns_taken;
        self.pending_tools = tool_calls
            .into_iter()
            .map(|call| Self::classify(call, &turn.available_tools))
            .collect();
        self.state = MachineState::AwaitingTools { turn: turn_number };
    }

    /// Feed tool-result messages back into the machine.
    ///
    /// Each provided [`Message`] is appended to the history. After the results
    /// are recorded, the next [`Self::next_step`] requests another
    /// [`MachineStep::CallLLM`] (subject to the max-turn budget and compaction
    /// trigger). Has no effect once the machine is terminal.
    pub fn tool_results(&mut self, messages: Vec<Message>) {
        if self.is_terminal() {
            return;
        }
        self.history.extend(messages);
        self.pending_tools.clear();
        self.state = MachineState::Start;
    }

    /// Feed compacted history back into the machine.
    ///
    /// The driver calls this after servicing a [`MachineStep::Compact`], passing
    /// the compacted history and `tokens_after` — its estimate of that history's
    /// token size, which the machine adopts as the current context size so it
    /// does not immediately request another compaction. The next
    /// [`Self::next_step`] then requests the deferred [`MachineStep::CallLLM`].
    /// Has no effect once the machine is terminal.
    pub fn compaction_result(&mut self, compacted: Vec<Message>, tokens_after: u64) {
        if self.is_terminal() {
            return;
        }
        self.history = compacted;
        self.context_tokens = tokens_after;
        self.state = MachineState::Start;
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

    /// The append-only conversation history.
    ///
    /// Read-only access to the record the machine owns. The driver builds the
    /// LLM feed from this on each [`MachineStep::CallLLM`].
    #[must_use]
    pub fn history(&self) -> &[Message] {
        &self.history
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

    /// Total input tokens accumulated so far.
    ///
    /// The run's lifetime input consumption, for accounting. Also carried in
    /// [`MachineOutcome::Completed`] once the run ends.
    #[must_use]
    pub fn total_input_tokens(&self) -> u64 {
        self.total_input_tokens
    }

    /// Total output tokens accumulated so far.
    ///
    /// The run's total generation cost, for accounting. Also carried in
    /// [`MachineOutcome::Completed`] once the run ends.
    #[must_use]
    pub fn total_output_tokens(&self) -> u64 {
        self.total_output_tokens
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
                ToolContent::Text(message),
                true,
            )],
        )
    }

    /// Concatenate every text part of a message into a single string.
    ///
    /// Walks the message's parts in order and joins all text content, skipping
    /// non-text parts (tool calls, tool results, images) entirely. Used to
    /// derive the final answer text from the model's last response.
    fn extract_text(message: &Message) -> String {
        message
            .parts
            .iter()
            .filter_map(|part| part.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Collect the tool calls a message requests, in order.
    ///
    /// Scans the message's parts for tool-call parts and maps each to a
    /// [`ToolCall`] (carrying its id, tool name, and JSON input). Returns an
    /// empty vector when the message contains no tool calls — the signal that
    /// the turn is complete rather than a request to dispatch tools.
    fn extract_tool_calls(message: &Message) -> Vec<ToolCall> {
        message
            .parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::ToolCall { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    tool: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn small_machine(max_turns: usize) -> LoopMachine {
        LoopMachine::new(
            RunConfig {
                max_turns,
                ..RunConfig::default()
            },
            "hello",
        )
    }

    fn text_turn(text: &str, input_tokens: u64, output_tokens: u64) -> ModelTurn {
        ModelTurn {
            message: Message::assistant(text),
            input_tokens,
            output_tokens,
            stop_reason: StopReason::EndTurn,
            available_tools: Vec::new(),
        }
    }

    fn tool_turn(tool: &str, available: &[&str], input_tokens: u64) -> ModelTurn {
        let part = MessagePart::tool_call("call_1", tool, Value::Object(serde_json::Map::new()));
        ModelTurn {
            message: Message::new(Role::Assistant, vec![part]),
            input_tokens,
            output_tokens: 10,
            stop_reason: StopReason::ToolCall,
            available_tools: available.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn same_step(a: &MachineStep, b: &MachineStep) -> bool {
        serde_json::to_string(a).unwrap_or_default() == serde_json::to_string(b).unwrap_or_default()
    }

    #[test]
    fn calling_llm_from_new_emits_call_llm_step() {
        let mut machine = LoopMachine::new(RunConfig::default(), "hello");
        let step = machine.next_step();
        let MachineStep::CallLLM { turn } = step else {
            panic!("expected CallLLM, got {step:?}");
        };
        assert_eq!(turn, 1);
        assert_eq!(machine.state(), MachineState::AwaitingModel { turn: 1 });
    }

    #[test]
    fn machine_api_has_no_async_no_tokio_no_apiclient() {
        // If any driving method were async or needed a client/runtime, this
        // body would not compile or would require a tokio context. It touches
        // only the pure machine surface.
        let mut machine = LoopMachine::new(RunConfig::default(), "hello");
        assert!(matches!(machine.next_step(), MachineStep::CallLLM { .. }));
        machine.model_response(text_turn("hi", 5, 3));
        // A text-only turn completes the run, so the machine is terminal.
        assert!(machine.is_terminal());
        assert_eq!(machine.turns_taken(), 1);
        assert_eq!(machine.total_input_tokens(), 5);
        assert_eq!(machine.total_output_tokens(), 3);
        assert_eq!(machine.history().len(), 2);
        assert!(matches!(
            machine.state(),
            MachineState::Terminal(MachineOutcome::Completed { .. })
        ));
        assert_eq!(machine.config().max_turns, RunConfig::default().max_turns);
        machine.cancel();
        assert!(machine.is_cancelled());
    }

    #[test]
    fn resume_after_model_response_round_trips() {
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.model_response(tool_turn("echo", &["echo"], 10));
        // Now AwaitingTools.
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step();
        let b = restored.next_step();
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn resume_after_tool_results_round_trips() {
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.model_response(tool_turn("echo", &["echo"], 10));
        let step = machine.next_step();
        let MachineStep::CallTools { calls } = &step else {
            panic!("expected CallTools, got {step:?}");
        };
        // Synthesize the tool-result message the driver would build.
        let results: Vec<Message> = calls
            .iter()
            .map(|c| {
                Message::new(
                    Role::User,
                    vec![MessagePart::tool_result(
                        c.call.id.clone(),
                        ToolContent::Text("ok".to_string()),
                        false,
                    )],
                )
            })
            .collect();
        machine.tool_results(results);
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step();
        let b = restored.next_step();
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn resume_after_compaction_result_round_trips() {
        // window = 100, threshold = 0.5 ⇒ compact once tokens > 50.
        let mut machine = LoopMachine::new(
            RunConfig {
                max_turns: 5,
                context_window: 100,
                compact_threshold: 5_000,
                auto_compact: true,
                ..RunConfig::default()
            },
            "hello",
        );
        // Drive a tool-call turn past the threshold so the next step compacts.
        let _ = machine.next_step(); // CallLLM
        machine.model_response(tool_turn("echo", &["echo"], 60)); // AwaitingTools
        let _ = machine.next_step(); // CallTools
        machine.tool_results(vec![Message::user("tool-out")]); // → Start
        assert!(matches!(machine.next_step(), MachineStep::Compact { .. }));
        machine.compaction_result(vec![Message::user("compacted")], 0);
        let snapshot = serde_json::to_string(&machine).expect("serialize");
        let mut restored: LoopMachine = serde_json::from_str(&snapshot).expect("deserialize");
        let a = machine.next_step();
        let b = restored.next_step();
        assert!(same_step(&a, &b), "steps diverged after round-trip");
    }

    #[test]
    fn max_turns_enforced_by_machine() {
        let mut machine = small_machine(2);
        // Turn 1.
        assert!(matches!(
            machine.next_step(),
            MachineStep::CallLLM { turn: 1 }
        ));
        machine.model_response(tool_turn("echo", &["echo"], 1));
        let _ = machine.next_step();
        machine.tool_results(vec![Message::user("r")]);
        // Turn 2.
        assert!(matches!(
            machine.next_step(),
            MachineStep::CallLLM { turn: 2 }
        ));
        machine.model_response(tool_turn("echo", &["echo"], 1));
        let _ = machine.next_step();
        machine.tool_results(vec![Message::user("r")]);
        // Turn 3 must be denied.
        match machine.next_step() {
            MachineStep::Done(MachineOutcome::MaxTurnsExceeded { turns_taken }) => {
                assert_eq!(turns_taken, 2);
            }
            other => panic!("expected MaxTurnsExceeded, got {other:?}"),
        }
        assert!(machine.is_terminal());
    }

    #[test]
    fn cancel_returns_done_cancelled_at_next_step() {
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.cancel();
        match machine.next_step() {
            MachineStep::Done(MachineOutcome::Cancelled) => {}
            other => panic!("expected Done(Cancelled), got {other:?}"),
        }
        // Idempotent: stays terminal-cancelled.
        assert!(matches!(
            machine.next_step(),
            MachineStep::Done(MachineOutcome::Cancelled)
        ));
    }

    #[test]
    fn unknown_tool_call_gets_preresolved_result() {
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.model_response(tool_turn("ghost", &["echo", "ls"], 3));
        let step = machine.next_step();
        let MachineStep::CallTools { calls } = step else {
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
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.model_response(tool_turn("echo", &["echo", "ls"], 3));
        let step = machine.next_step();
        let MachineStep::CallTools { calls } = step else {
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
        // window = 100, threshold = 0.5 ⇒ compact once tokens > 50.
        let mut machine = LoopMachine::new(
            RunConfig {
                max_turns: 5,
                context_window: 100,
                compact_threshold: 5_000,
                auto_compact: true,
                ..RunConfig::default()
            },
            "hello",
        );
        // First step: no tokens yet (0 > 50 is false) ⇒ CallLLM, not Compact.
        assert!(matches!(machine.next_step(), MachineStep::CallLLM { .. }));
        // A tool-call turn keeps the run going and accumulates 60 input tokens
        // (exceeds 50). A text-only turn would complete the run instead.
        machine.model_response(tool_turn("echo", &["echo"], 60));
        assert!(matches!(machine.next_step(), MachineStep::CallTools { .. }));
        machine.tool_results(vec![Message::user("tool-out")]);
        // The next step must compact before calling the model again.
        match machine.next_step() {
            MachineStep::Compact { reason } => {
                assert_eq!(reason, CompactReason::ThresholdExceeded);
            }
            other => panic!("expected Compact, got {other:?}"),
        }
    }

    #[test]
    fn compaction_result_replaces_history() {
        // window = 100, threshold = 0.5 ⇒ compact once tokens > 50.
        let mut machine = LoopMachine::new(
            RunConfig {
                max_turns: 5,
                context_window: 100,
                compact_threshold: 5_000,
                auto_compact: true,
                ..RunConfig::default()
            },
            "hello",
        );
        // Drive a tool-call turn past the threshold so the next step compacts.
        let _ = machine.next_step(); // CallLLM
        machine.model_response(tool_turn("echo", &["echo"], 60)); // AwaitingTools
        let _ = machine.next_step(); // CallTools
        machine.tool_results(vec![Message::user("tool-out")]); // → Start
        assert!(matches!(machine.next_step(), MachineStep::Compact { .. }));
        let compacted = vec![Message::user("compacted-only")];
        machine.compaction_result(compacted.clone(), 0);
        // Compare by serialized form: Message is not PartialEq.
        let got = serde_json::to_string(machine.history()).expect("serialize history");
        let want = serde_json::to_string(&compacted).expect("serialize expected");
        assert_eq!(got, want, "history must be replaced by the compacted slice");
        // After compaction the next step is the deferred CallLLM.
        assert!(matches!(machine.next_step(), MachineStep::CallLLM { .. }));
    }

    #[test]
    fn history_accumulates_user_assistant_tool_round() {
        let mut machine = small_machine(5);
        let _ = machine.next_step();
        machine.model_response(tool_turn("echo", &["echo"], 1));
        let step = machine.next_step();
        let MachineStep::CallTools { calls } = step else {
            panic!("expected CallTools, got {step:?}");
        };
        let result = Message::new(
            Role::User,
            calls
                .iter()
                .map(|c| {
                    MessagePart::tool_result(
                        c.call.id.clone(),
                        ToolContent::Text("ok".to_string()),
                        false,
                    )
                })
                .collect(),
        );
        machine.tool_results(vec![result]);

        let roles: Vec<Role> = machine.history().iter().map(|m| m.role).collect();
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
}
