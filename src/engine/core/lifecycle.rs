//! Foundational lifecycle types and the core agent trait.
//!
//! Fundamental operations every agent must support, plus the data types that
//! flow through the agent lifecycle: per-run configuration
//! ([`RunConfig`]), lifecycle state ([`MachineState`]),
//! turn and run results ([`Run`]), and tool call representations
//! ([`ToolCall`]).
//!
//! # Lifecycle
//!
//! ```text
//! run(input, run_config)
//!   → should_continue() → true   [repeated]
//!   → should_continue() → false
//! finalize()
//! ```
//!
//! # Implementing
//!
//! At a minimum you must provide [`run`](Loop::run),
//! [`should_continue`](Loop::should_continue),
//! [`finalize`](Loop::finalize),
//! [`state`](Loop::state), and
//! [`cancel`](Loop::cancel).
//!
//! # Quick Start
//!
//! ```
//! use loopctl::config::SessionConfig;
//! use loopctl::engine::core::{MachineState, Run};
//!
//! let _config = SessionConfig::default();
//!
//! let run = Run::new("Task done.", &Default::default());
//! assert_eq!(run.turn_count(), 0);
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::ParallelDispatchConfig;
use crate::engine::core::machine::MachineState;
use crate::error::LoopError;

use crate::reflection::{Correction, CorrectionResult, CorrectionType};

/// Provide a fresh `Instant` for serde-deserialized fields.
///
/// `Instant` has no `Default` (its zero point is unspecified), so
/// `#[serde(skip, default = "instant_now")]` is used on the
/// `Instant`-typed fields of [`Run`] to give deserialized runs a
/// sensible monotonic starting point.
fn instant_now() -> Instant {
    Instant::now()
}

/// Per-run configuration.
///
/// The slice of agent configuration that varies across `run()` calls on the
/// same agent (turn/token budgets, compaction policy, dispatch mode, manager
/// reset). It is owned by the machine so that the machine's decisions are
/// self-contained and a serialized machine carries its configuration with
/// it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunConfig {
    /// Maximum number of turns before forcing completion.
    ///
    /// A safety cap on runaway loops: once the machine has taken this many
    /// turns without the model finishing, the run ends with
    /// `MachineOutcome::MaxTurnsExceeded`. Defaults to `200`. Set it per-run
    /// to bound long-running tool loops.
    pub max_turns: usize,

    /// How independent tool calls within a single turn are dispatched.
    ///
    /// Controls whether the driver runs a turn's independent tool calls
    /// sequentially or in parallel, up to a concurrency limit. See
    /// [`ParallelDispatchConfig`].
    pub parallel_tool_dispatch: ParallelDispatchConfig,

    /// Whether to reset all managers to their initial state before this run.
    ///
    /// When `true`, [`LoopManagers::reset_all`] is called at the top of
    /// `run()`, clearing the fallback circuit breaker, loop/convergence
    /// detection history, and per-observer accumulators. Use this when a
    /// run is logically independent from the previous one (different task,
    /// fresh context) and you do not want accumulated state to carry over.
    /// Defaults to `false` — manager state persists across runs within a
    /// session, which is usually the right behavior (e.g. a tripped
    /// circuit breaker should stay tripped).
    ///
    /// [`LoopManagers::reset_all`]: crate::managers::LoopManagers::reset_all
    pub reset_managers: bool,

    /// How many memory entries to retrieve and inject at the top of each turn.
    ///
    /// When a [`LoopMemory`](crate::memory::LoopMemory) backend is configured,
    /// the driver retrieves this many relevant entries before each model call
    /// and appends them as a reference user message. Defaults to `3`. Set to
    /// `0` to disable memory retrieval entirely for the run.
    pub memory_top_k: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_turns: 200,
            parallel_tool_dispatch: ParallelDispatchConfig::default(),
            reset_managers: false,
            memory_top_k: 3,
        }
    }
}

impl RunConfig {
    /// Set the parallel tool-dispatch configuration for the run.
    ///
    /// `RunConfig` is `#[non_exhaustive]`, so this builder is the one
    /// way a host opts a run into
    /// [`ParallelMode::Parallel`](crate::config::ParallelMode::Parallel)
    /// (or back to sequential) — the struct literal is not available
    /// outside the crate.
    ///
    /// ```
    /// use loopctl::config::{ParallelDispatchConfig, ParallelMode};
    /// use loopctl::engine::RunConfig;
    ///
    /// let config = RunConfig::default().with_parallel_dispatch(
    ///     ParallelDispatchConfig {
    ///         mode: ParallelMode::Parallel,
    ///         ..Default::default()
    ///     },
    /// );
    /// assert!(matches!(
    ///     config.parallel_tool_dispatch.mode,
    ///     ParallelMode::Parallel
    /// ));
    /// ```
    #[must_use]
    pub fn with_parallel_dispatch(mut self, config: crate::config::ParallelDispatchConfig) -> Self {
        self.parallel_tool_dispatch = config;
        self
    }

    /// Set the maximum number of turns for this run.
    ///
    /// Builder-style convenience for `#[non_exhaustive]` compliance.
    ///
    /// ```
    /// use loopctl::engine::RunConfig;
    ///
    /// let config = RunConfig::default().with_max_turns(50);
    /// assert_eq!(config.max_turns, 50);
    /// ```
    #[must_use]
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }
}

/// Why the API stopped generating.
///
/// Mirrors the stop reasons returned by LLM APIs. The framework uses this
/// to determine the next step: dispatch tools, continue the conversation,
/// or end the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StopReason {
    /// The model decided to stop (natural end of turn).
    ///
    /// The LLM finished its response without requesting tools or hitting
    /// any limit. The framework treats this as a completed run.
    EndTurn,

    /// The model requested tool execution.
    ///
    /// The LLM response contains one or more tool calls. The framework should
    /// dispatch them and feed the results back.
    ToolCall,

    /// The maximum token limit was reached.
    ///
    /// The LLM hit the model's max-tokens limit before
    /// finishing. The response may be truncated. The framework may
    /// choose to continue the turn to let the model complete its output.
    MaxTokens,

    /// The stop sequence was encountered.
    ///
    /// The model generated a configured stop sequence. Rare in
    /// standard usage; typically indicates custom API configuration.
    StopSequence,
}

/// How the engine fulfils each LLM turn.
///
/// `BareLoop` drives every turn by asking the [`ApiClient`](crate::api::ApiClient)
/// for a response and folding the result into the conversation. Two mechanisms
/// are available, selected per turn from this enum:
///
/// - `NonStreaming` calls [`ApiClient::create_message`](crate::api::ApiClient::create_message)
///   and receives a single complete [`Message`](crate::message::Message). It
///   compiles and runs with no streaming dependencies, so it is the default
///   under `default = []`.
/// - `Streaming` calls [`ApiClient::stream_messages`](crate::api::ApiClient::stream_messages)
///   through the resilient `StreamHandler`, emitting per-delta observer
///   callbacks. Requires the `streaming` feature.
///
/// The constructor default is feature-dependent: `Streaming` when `streaming`
/// is compiled in, otherwise `NonStreaming` (see
/// [`BareLoop::turn_mode`](crate::engine::BareLoop::turn_mode)). It is
/// intentionally *not* a `Default` impl on this enum, because a single fixed
/// `Default` could not express that feature-dependent choice. Switch modes on
/// a constructed loop with
/// [`set_turn_mode`](crate::engine::BareLoop::set_turn_mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnMode {
    /// Fulfil each turn via [`ApiClient::create_message`](crate::api::ApiClient::create_message).
    ///
    /// No streaming code is exercised; `on_text_delta`, `on_thinking_delta`,
    /// and the text streamer never fire. The full assistant text still
    /// surfaces through [`on_response`](crate::observer::LoopObserver::on_response).
    NonStreaming,

    /// Fulfil each turn via [`ApiClient::stream_messages`](crate::api::ApiClient::stream_messages)
    /// wrapped in [`StreamHandler`](crate::stream::handler::StreamHandler).
    ///
    /// Requires the `streaming` feature: this variant only exists when the
    /// feature is enabled, so it cannot be constructed or selected without it.
    #[cfg(feature = "streaming")]
    Streaming,
}

/// Resolve the constructor default for [`TurnMode`].
///
/// Streaming when the `streaming` feature is compiled in, non-streaming
/// otherwise. Kept as a free function so both constructors share one
/// definition and the `cfg` lives in exactly one place.
pub(crate) fn default_turn_mode() -> TurnMode {
    #[cfg(feature = "streaming")]
    {
        TurnMode::Streaming
    }
    #[cfg(not(feature = "streaming"))]
    {
        TurnMode::NonStreaming
    }
}

/// A tool call requested by the agent.
///
/// Represents a single tool invocation that the LLM has requested during a
/// turn. The framework matches each `ToolCall` to a registered tool, executes
/// it, and produces a tool dispatch result with the output.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence and inter-process
/// communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The call identifier assigned by the LLM API.
    ///
    /// Used to correlate this request with the tool-result message that the
    /// framework produces after dispatching the tool. Unique within a single
    /// model response.
    pub id: String,

    /// The name of the tool to invoke.
    ///
    /// Must match a tool registered in the
    /// [`ToolRegistry`](crate::tool::ToolRegistry). If the name is not
    /// registered, the machine classifies the call as unknown and attaches a
    /// preresolved error result instead of dispatching.
    pub tool: String,

    /// The JSON input arguments for the tool.
    ///
    /// A free-form JSON value whose shape depends on the tool's input schema.
    /// The framework passes this verbatim to the tool's `call` method; the
    /// tool is responsible for validating and parsing it.
    pub input: serde_json::Value,
}

impl ToolCall {
    /// Apply a [`Correction`] from the reflection system in place.
    ///
    /// Modifies `self` according to the correction strategy:
    ///
    /// - [`InputFix`](CorrectionType::InputFix) — replaces
    ///   `self.input` with the corrected input.
    /// - [`ToolChange`](CorrectionType::ToolChange) — replaces
    ///   `self.tool` with an alternative tool name.
    /// - Other types — no mutation; the retry proceeds with unchanged parameters.
    ///
    /// Returns a [`CorrectionResult`] indicating whether the correction
    /// was applied, failed, or skipped.
    pub fn apply_correction(&mut self, correction: &Correction) -> CorrectionResult {
        match correction.correction_type {
            CorrectionType::InputFix => {
                if let Some(ref modified) = correction.modified_input {
                    if modified.is_object() {
                        tracing::debug!(
                            tool = %self.tool,
                            "applying InputFix correction from reflector"
                        );
                        self.input = modified.clone();
                        CorrectionResult::Applied
                    } else {
                        CorrectionResult::Failed(
                            "InputFix correction modified_input must be a JSON object".to_string(),
                        )
                    }
                } else {
                    CorrectionResult::Failed(
                        "InputFix correction missing modified_input".to_string(),
                    )
                }
            }
            CorrectionType::ToolChange => {
                if let Some(ref alt) = correction.alternative_tool {
                    tracing::debug!(
                        old_tool = %self.tool,
                        new_tool = %alt,
                        "applying ToolChange correction from reflector"
                    );
                    self.tool.clone_from(alt);
                    CorrectionResult::Applied
                } else {
                    CorrectionResult::Failed(
                        "ToolChange correction missing alternative_tool".to_string(),
                    )
                }
            }
            CorrectionType::PrerequisiteFix | CorrectionType::ApproachChange => {
                CorrectionResult::Skipped
            }
            CorrectionType::Escalate => CorrectionResult::Skipped,
        }
    }
}

/// One iteration of the agent loop — a single LLM call and any tools it
/// triggered.
///
/// Each entry in [`Run::turns`] records what happened during one loop
/// iteration: the model's response text, any tool calls it requested, and the
/// token cost of that call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// The 0-indexed position of this turn within its run.
    ///
    /// Counts from zero within the owning [`Run`]. Useful for
    /// correlating a turn with observer events and logs.
    pub turn: usize,

    /// What went into this turn's model call.
    ///
    /// For turn 0 this is the user's prompt. For subsequent turns it's the
    /// tool-result text from the previous turn's dispatch — the context the
    /// model sees when continuing the loop.
    pub input: String,

    /// The text the model produced (concatenated text parts).
    ///
    /// May be empty if the turn was a pure tool-call response with no
    /// accompanying text. On the final turn (no tool calls) this is the
    /// run's answer.
    pub output: String,

    /// The tool calls the model requested during this turn, if any.
    ///
    /// Empty when the turn ended with a plain text response. Each entry is
    /// one tool invocation the loop dispatched (or would have dispatched).
    pub tool_calls: Vec<ToolCall>,

    /// Input tokens reported by the provider for this turn.
    ///
    /// The prompt-side token count — the size of the context the model read
    /// to produce this turn's response.
    pub input_tokens: u64,

    /// Output tokens reported by the provider for this turn.
    ///
    /// The completion-side token count — the size of the response the model
    /// produced for this turn.
    pub output_tokens: u64,
}

/// One prompt → loop → final answer.
///
/// Owned by a [`Session`]; a session accumulates one `Run` per `run()` call.
/// The turn list carries per-turn detail; aggregate totals are derived from it.
///
/// # Example
///
/// ```
/// use loopctl::engine::core::Run;
///
/// let mut run = Run::new("What is 2+2?", &Default::default());
/// run.output = Some("4".to_string());
///
/// assert_eq!(run.output.as_deref(), Some("4"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// A fresh identifier unique to this run.
    ///
    /// Minted when the run begins; differs across successive `run()` calls on
    /// the same session, so a caller can tell runs apart.
    pub id: Uuid,

    /// When this run started.
    ///
    /// Captured at the top of the run. Combined with [`end`](Self::end) by
    /// [`duration`](Self::duration) to report the run's wall-clock span.
    /// Skipped during serialization (monotonic `Instant` is not portable);
    /// deserialized runs get a fresh `Instant::now()`.
    #[serde(skip, default = "instant_now")]
    pub start: Instant,

    /// When this run ended.
    ///
    /// `None` while the run is still in flight, `Some` once it terminates
    /// (completed, cancelled, or failed). Skipped during serialization;
    /// deserialized runs get `None` (treated as still in-flight).
    #[serde(skip)]
    pub end: Option<Instant>,

    /// The turns executed during this run, in order.
    ///
    /// Each entry records one model call and any tools it triggered: the
    /// input text the model saw, the output it produced, the tool calls it
    /// requested, and the token cost of that call. Derived totals
    /// ([`turn_count`](Self::turn_count), [`input_tokens`](Self::input_tokens),
    /// [`output_tokens`](Self::output_tokens),
    /// [`tool_call_count`](Self::tool_call_count)) aggregate over this list.
    pub turns: Vec<Turn>,

    /// The user prompt that started this run.
    ///
    /// The exact text passed to `run()`, recorded for replay and reporting.
    /// This is the `input` of the first [`Turn`] in [`turns`](Self::turns).
    pub input: String,

    /// The run's final output, if it completed.
    ///
    /// The assistant's final answer on a successful run; `None` while in flight
    /// or when the run ended without producing a final message. For a
    /// completed run this matches the `output` of the last
    /// [`Turn`](Self::turns) that carried no tool calls.
    pub output: Option<String>,

    /// The per-run configuration that governed this run.
    ///
    /// The turn budget, dispatch mode, and compaction policy passed to the
    /// `run()` call that started this run. Captured so a serialized run
    /// carries its governing config.
    pub config: RunConfig,

    /// Why the run ended, if it has terminated.
    ///
    /// `None` while the run is in flight or completed normally. Set to the
    /// terminal [`LoopError`] (`Cancelled`, `MaxTurnsExceeded`, etc.) when
    /// the run ended abnormally. Populated by the engine in `finalize`.
    #[serde(skip)]
    pub stop_reason: Option<LoopError>,
}

impl Run {
    /// Begin a fresh run for the given `input`, governed by `config`.
    ///
    /// Mints a new [`id`](Self::id), captures [`start`](Self::start)
    /// as now, leaves [`end`](Self::end) as `None` (the run is in flight),
    /// stores the per-run [`config`](Self::config), and zeroes all accounting
    /// fields. The framework fills in turns, token totals, and the terminal
    /// outcome as the run progresses.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::engine::core::Run;
    /// use loopctl::engine::RunConfig;
    ///
    /// let run = Run::new("Hello, agent!", &RunConfig::default());
    /// assert_eq!(run.input, "Hello, agent!");
    /// assert_eq!(run.turn_count(), 0);
    /// assert!(run.end.is_none());
    /// ```
    #[must_use]
    pub fn new(input: impl Into<String>, config: &RunConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            start: Instant::now(),
            end: None,
            turns: Vec::new(),
            input: input.into(),
            output: None,
            config: config.clone(),
            stop_reason: None,
        }
    }

    /// Number of turns executed during this run.
    ///
    /// Equivalent to the length of [`turns`](Self::turns). Each turn
    /// corresponds to one model call.
    #[must_use]
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Total input tokens consumed across all turns.
    ///
    /// Sums the `input_tokens` field of every [`Turn`] in
    /// [`turns`](Self::turns) — the total prompt-side cost of the run.
    #[must_use]
    pub fn input_tokens(&self) -> u64 {
        self.turns.iter().map(|t| t.input_tokens).sum()
    }

    /// Total output tokens consumed across all turns.
    ///
    /// Sums the `output_tokens` field of every [`Turn`] in
    /// [`turns`](Self::turns) — the total completion-side cost of the run.
    #[must_use]
    pub fn output_tokens(&self) -> u64 {
        self.turns.iter().map(|t| t.output_tokens).sum()
    }

    /// Total number of tool calls dispatched across all turns.
    ///
    /// Sums the tool-call count of every [`Turn`] in
    /// [`turns`](Self::turns).
    #[must_use]
    pub fn tool_call_count(&self) -> usize {
        self.turns.iter().map(|t| t.tool_calls.len()).sum()
    }

    /// Wall-clock duration of this run.
    ///
    /// Elapsed time from [`start`](Self::start) to [`end`](Self::end).
    /// While the run is in flight (`end` is `None`), counts up from
    /// `start` to now.
    #[must_use]
    pub fn duration(&self) -> Duration {
        match self.end {
            Some(end) => end.saturating_duration_since(self.start),
            None => self.start.elapsed(),
        }
    }

    /// Total tokens (input + output) consumed by this run.
    ///
    /// Convenience: sums [`input_tokens`](Self::input_tokens) and
    /// [`output_tokens`](Self::output_tokens) with saturating addition.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens().saturating_add(self.output_tokens())
    }
}

impl Default for Run {
    fn default() -> Self {
        Self::new(String::new(), &RunConfig::default())
    }
}

/// One agent identity, stable across the process lifetime.
///
/// A multi-turn REPL is one `Session` made of many [`Run`]s: the `session_id`
/// and start time are stable across `run()` calls, while each run rotates its
/// own identity and accounting. Per-session totals are derived from the run
/// list, never stored separately.
///
/// # Example
///
/// ```
/// use loopctl::config::SessionConfig;
/// use loopctl::engine::core::{Run, Session};
///
/// let config = SessionConfig::default();
/// let mut session = Session::new(config);
/// session.runs.push(Run::new("first prompt", &Default::default()));
/// session.runs.push(Run::new("second prompt", &Default::default()));
/// assert_eq!(session.runs.len(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    ///
    /// A random UUID v4 generated on construction. Two sessions with the same
    /// `id` are considered the same logical session. Stable across `run()`
    /// calls on the same agent.
    pub id: Uuid,

    /// Session-scoped configuration.
    ///
    /// The system prompt, context window, and compaction settings — set once
    /// at construction and unchanged across `run()` calls. See
    /// [`SessionConfig`](crate::config::SessionConfig).
    pub config: crate::config::SessionConfig,

    /// When the session started, or `None` before the first `run()` call.
    ///
    /// Captured once on the first run; unlike a run's start time it does not
    /// rotate between `run()` calls. Skipped during serialization (monotonic
    /// `Instant` is not portable); deserialized sessions get `None`.
    #[serde(skip)]
    pub session_start: Option<Instant>,

    /// The runs executed in this session, in order.
    ///
    /// One [`Run`] per `run()` call. The session's derived totals
    /// (turns, tokens, duration) sum over this list. The last entry is the
    /// in-flight run (or the most recently completed one).
    pub runs: Vec<Run>,
}

impl Session {
    /// Create a new session with a fresh random id and the given config.
    ///
    /// The session starts with `session_start = None` (set on the first
    /// `run()` call) and an empty run list.
    #[must_use]
    pub fn new(config: crate::config::SessionConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            session_start: None,
            runs: Vec::new(),
        }
    }

    /// Borrow the most recent run (the in-flight or just-completed one).
    ///
    /// Returns `None` when no `run()` call has been made yet.
    #[must_use]
    pub fn current_run(&self) -> Option<&Run> {
        self.runs.last()
    }

    /// Mutably borrow the most recent run (the in-flight or just-completed
    /// one).
    ///
    /// Returns `None` when no `run()` call has been made yet.
    #[must_use]
    pub fn current_run_mut(&mut self) -> Option<&mut Run> {
        self.runs.last_mut()
    }

    /// Total turns across every run in this session.
    ///
    /// The sum of each [`Run`]'s turn count. Derived from `runs`, so it stays
    /// correct as runs are added or mutated.
    #[must_use]
    pub fn total_turns(&self) -> usize {
        self.runs.iter().map(Run::turn_count).sum()
    }

    /// Total wall-clock duration across every run in this session.
    ///
    /// The sum of each [`Run::duration`].
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.runs.iter().map(Run::duration).sum()
    }

    /// Total input tokens across every run in this session.
    ///
    /// The sum of each [`Run::input_tokens`].
    #[must_use]
    pub fn total_input_tokens(&self) -> u64 {
        self.runs.iter().map(Run::input_tokens).sum()
    }

    /// Total output tokens across every run in this session.
    ///
    /// The sum of each [`Run::output_tokens`].
    #[must_use]
    pub fn total_output_tokens(&self) -> u64 {
        self.runs.iter().map(Run::output_tokens).sum()
    }
}

/// The result of a `run()` call — either the completed [`Run`] or a [`LoopError`].
pub type RunResult = Result<Run, LoopError>;

/// The core agent lifecycle trait.
///
/// Implement this trait to create a new type of agent. The framework
/// provides shared infrastructure for context management, tool execution,
/// reflection, and observability, so implementations only need to define
/// the core processing logic.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::engine::core::Loop;
/// use loopctl::engine::RunConfig;
/// use loopctl::engine::core::{MachineState, Run};
/// use loopctl::error::LoopError;
///
/// struct MyAgent {
///     state: MyState,
/// }
///
/// impl Loop for MyAgent {
///     fn run<'a>(
///         &'a mut self,
///         input: &'a str,
///         run_config: &'a RunConfig,
///     ) -> Pin<Box<dyn Future<Output = RunResult> + Send + 'a>> {
///         Box::pin(async { Ok(loopctl::engine::core::Run::new(input, &Default::default())) })
///     }
///     fn should_continue(&self) -> bool {
///         !self.state.is_complete
///     }
///     fn finalize<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = RunResult> + Send + 'a>> {
///         Box::pin(async { Ok(loopctl::engine::core::Run::new("", &Default::default())) })
///     }
///     fn state(&self) -> MachineState {
///         MachineState::Start
///     }
///     fn cancel(&self) {}
/// }
/// ```
pub trait Loop: Send + Sync {
    /// Drive one run of the agent loop for the given user prompt.
    ///
    /// This is the main entry point for running an agent. Session-scoped state
    /// (identity, start time, system prompt) is established once at
    /// construction and stays stable across calls; each `run()` receives a
    /// fresh [`RunConfig`] for the per-run budget and policy, mints a new
    /// [`Run`], and drives the turn loop until
    /// the model finishes or the budget is exhausted.
    ///
    /// `input` is the prompt for this run. It is passed to the first turn;
    /// continuation turns receive an empty string (tool results are already in
    /// the conversation history).
    ///
    /// # Errors
    ///
    /// - [`LoopError::Cancelled`] — if the session was cancelled.
    /// - [`LoopError::MaxTurnsExceeded`] — if the turn limit was reached.
    /// - Any error returned by the turn loop or [`finalize`](Self::finalize).
    fn run<'a>(
        &'a mut self,
        input: &'a str,
        run_config: &'a RunConfig,
    ) -> Pin<Box<dyn Future<Output = RunResult> + Send + 'a>>;

    /// Check whether the agent should continue processing turns.
    ///
    /// Called after each turn. Return `false` to end the session.
    fn should_continue(&self) -> bool;

    /// Finalize the agent run and produce its result.
    ///
    /// Called once after the last turn. Use this to clean up resources
    /// and produce a final [`Run`].
    fn finalize<'a>(
        &'a mut self,
        error: Option<&'a LoopError>,
    ) -> Pin<Box<dyn Future<Output = RunResult> + Send + 'a>>;

    /// Get the current state of the agent.
    ///
    /// Used by the framework to drive the state machine and by observers
    /// to report status.
    fn state(&self) -> MachineState;

    /// Cancel the agent's current operation.
    ///
    /// Implementations must use thread-safe interior mutability (e.g.
    /// [`AtomicBool`](std::sync::atomic::AtomicBool), `Mutex<bool>`) to
    /// store the cancellation flag, since this method takes `&self`. The
    /// flag should be set in a non-blocking fashion so that
    /// [`run`](Loop::run) and [`should_continue`](Loop::should_continue)
    /// can observe it and return promptly across threads.
    fn cancel(&self);

    /// Explain *why* [`should_continue`](Self::should_continue) returned `false`.
    ///
    /// Return `None` for normal completion (the model finished) or `Some(err)`
    /// when the session was forced to stop (`Cancelled`, `MaxTurnsExceeded`,
    /// etc.). The default implementation returns `None` (normal completion).
    fn stop_reason(&self) -> Option<LoopError> {
        None
    }
}
