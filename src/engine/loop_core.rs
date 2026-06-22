//! Agent core trait and foundational lifecycle types.
//!
//! Fundamental operations every agent must support, plus the data types
//! that flow through the agent lifecycle: configuration ([`LoopConfig`]),
//! lifecycle state ([`LoopState`]), turn and session results
//! ([`TurnResult`], [`SessionResult`]), and tool call representations
//! ([`ToolCall`]).
//!
//! # Lifecycle
//!
//! ```text
//! initialize(config)
//!   → process_turn(input)   [repeated]
//!   → process_turn(input)
//!   → ...
//!   → should_continue() → false
//! finalize()
//! ```
//!
//! # Implementing
//!
//! At a minimum you must provide [`initialize`](Loop::initialize),
//! [`process_turn`](Loop::process_turn),
//! [`should_continue`](Loop::should_continue),
//! [`finalize`](Loop::finalize),
//! [`state`](Loop::state), and
//! [`cancel`](Loop::cancel).
//!
//! # Quick Start
//!
//! ```
//! use loopctl::engine::loop_core::{LoopConfig, LoopState, TurnResult, SessionResult};
//!
//! let config = LoopConfig::default();
//! assert_eq!(config.max_turns, 200);
//!
//! let turn = TurnResult::completed("Task done.");
//! assert!(turn.is_complete);
//!
//! let session = SessionResult::success(config.session_id);
//! assert!(session.success);
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::LoopError;

// Re-export LoopConfig for convenience — it lives in crate::config.
pub use crate::config::LoopConfig;

// Re-export ToolDispatchResult so consumers of this module see the full
// tool-call type family in one place.
pub use crate::tool::ToolDispatchResult;

// ==================================================
// LoopState
// ==================================================

/// The lifecycle state of an agent.
///
/// Models the agent as an explicit state machine, making transitions clear
/// and invalid states unrepresentable. The framework reads and writes this
/// enum to drive the agent loop and report status to observers.
///
/// ```text
/// Idle → Processing → WaitingForTool → Processing → ... → Completed/Failed
///                  ↘ Compacting ↗
///                  ↘ Reflecting ↗
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopState {
    /// The agent is idle, waiting for a user message.
    ///
    /// Initial state after initialization and the state
    /// the agent returns to between user inputs.
    ///
    /// No background work is performed while idle.
    Idle,

    /// The agent is actively processing a turn.
    ///
    /// Entered when the agent core begins processing a user message
    /// or a tool result. The `turn` field tracks progress for observers.
    ///
    /// The agent transitions here from [`Idle`](LoopState::Idle) or
    /// [`WaitingForTool`](LoopState::WaitingForTool).
    Processing {
        /// Current turn number (0-indexed). Used to enforce [`LoopConfig::max_turns`].
        turn: usize,
    },

    /// The agent is waiting for a tool to complete.
    ///
    /// Entered after the LLM requests a tool call. The framework
    /// dispatches the tool and waits for its result before returning
    /// to [`Processing`](LoopState::Processing).
    WaitingForTool {
        /// Name of the tool being executed. Never empty — always matches a registered tool name.
        tool: String,

        /// When the tool call started. Used to compute execution duration.
        started_at: SystemTime,
    },

    /// The agent is compacting its conversation context.
    ///
    /// Entered when token usage exceeds the
    /// [`compact_threshold`](LoopConfig::compact_threshold) or when
    /// compaction is explicitly requested. The agent summarizes older
    /// messages to free context space, then returns to
    /// [`Processing`](LoopState::Processing).
    Compacting {
        /// Why compaction was triggered. See
        /// [`CompactReason`](crate::compact::types::CompactReason).
        reason: crate::compact::types::CompactReason,
    },

    /// The agent is reflecting on a failure and preparing a correction.
    ///
    /// Entered when a tool call fails and the reflection system is
    /// enabled (via configuration).
    /// The agent analyzes the error and produces a
    /// [`Correction`](crate::reflection::Correction) before retrying.
    Reflecting {
        /// Number of errors being analyzed; higher counts may need
        /// [`ApproachChange`](crate::reflection::CorrectionType::ApproachChange).
        error_count: usize,
    },

    /// The agent has completed its task.
    ///
    /// Terminal state. The framework calls
    /// `Loop::finalize` to produce
    /// a [`SessionResult`].
    Completed {
        /// May be empty if the agent produced only tool calls.
        summary: String,
    },

    /// The agent has failed with an unrecoverable error.
    ///
    /// Terminal state. The error is propagated through
    /// [`SessionResult::error`].
    ///
    /// No further turns will be executed after entering this state.
    Failed { error: String },
}

// ==================================================
// TurnResult
// ==================================================

/// Result of a single agent turn (one API call → response cycle).
///
/// Produced by `Loop::process_turn`
/// after each LLM interaction. Contains the response text, any tool calls
/// requested, token usage, and timing information.
///
/// # Construction
///
/// Use [`TurnResult::completed`] for a terminal response or
/// [`TurnResult::continuing`] for a response that should keep the loop running.
/// Production code typically constructs this from the raw API response.
///
/// ```
/// use loopctl::engine::loop_core::TurnResult;
///
/// let done = TurnResult::completed("All tasks finished.");
/// assert!(done.is_complete);
///
/// let more = TurnResult::continuing("Still working...");
/// assert!(!more.is_complete);
/// ```
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// May be empty if the response consists entirely of tool calls.
    pub text: String,
    /// Tool calls requested by the LLM in this turn.
    pub tool_calls: Vec<ToolCall>,
    /// Results from dispatching the requested tool calls.
    pub tool_results: Vec<ToolDispatchResult>,
    /// Input tokens used (system prompt + history + user message). Reported by the provider.
    pub input_tokens: u64,
    /// Output tokens in the API response. Reported by the provider.
    pub output_tokens: u64,
    /// Wall-clock duration (API request → full response + tool execution).
    pub duration: Duration,
    /// When `true`, the framework skips `should_continue` and proceeds to finalization.
    pub is_complete: bool,
    /// Used by the framework to decide whether to dispatch tools ([`StopReason::ToolCall`]) or continue.
    pub stop_reason: StopReason,
}

impl TurnResult {
    /// Create a completed turn result with a simple text response.
    ///
    /// Sets [`is_complete`](TurnResult::is_complete) to `true` and all
    /// token counters to zero. Use this for the final turn of a session.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::engine::loop_core::TurnResult;
    ///
    /// let result = TurnResult::completed("The file has been written successfully.");
    /// assert!(result.is_complete);
    /// assert_eq!(result.tool_calls.len(), 0);
    /// ```
    #[must_use]
    pub fn completed(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            duration: Duration::ZERO,
            is_complete: true,
            stop_reason: StopReason::EndTurn,
        }
    }

    /// Create a turn result that should continue with more turns.
    ///
    /// Sets [`is_complete`](TurnResult::is_complete) to `false`, indicating
    /// that the agent loop should keep running.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::engine::loop_core::TurnResult;
    ///
    /// let result = TurnResult::continuing("I need to read the file first...");
    /// assert!(!result.is_complete);
    /// ```
    #[must_use]
    pub fn continuing(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            duration: Duration::ZERO,
            is_complete: false,
            stop_reason: StopReason::EndTurn,
        }
    }

    /// Check if this turn included any tool calls.
    ///
    /// Returns `true` when [`tool_calls`](TurnResult::tool_calls) is
    /// non-empty, indicating that the LLM requested tool execution.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Total tokens (input + output) for this turn.
    ///
    /// Sums [`input_tokens`](TurnResult::input_tokens)
    /// and [`output_tokens`](TurnResult::output_tokens).
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ==================================================
// StopReason
// ==================================================

/// Why the API stopped generating.
///
/// Mirrors the stop reasons returned by LLM APIs. The framework uses this
/// to determine the next step: dispatch tools, continue the conversation,
/// or end the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The model decided to stop (natural end of turn).
    ///
    /// The LLM finished its response without requesting tools or hitting
    /// any limit. The framework should check
    /// [`TurnResult::is_complete`] to decide whether to continue.
    EndTurn,

    /// The model requested tool execution.
    ///
    /// The LLM response contains one or more tool calls in
    /// [`TurnResult::tool_calls`]. The framework should dispatch them
    /// and feed the results back.
    ToolCall,

    /// The maximum token limit was reached.
    ///
    /// The LLM hit the [`LoopConfig::max_tokens`] limit before
    /// finishing. The response may be truncated. The framework may
    /// choose to continue the turn to let the model complete its output.
    MaxTokens,

    /// The stop sequence was encountered.
    ///
    /// The model generated a configured stop sequence. Rare in
    /// standard usage; typically indicates custom API configuration.
    StopSequence,
}

// ==================================================
// ToolCall
// ==================================================

/// A tool call requested by the agent.
///
/// Represents a single tool invocation that the LLM has requested during a
/// turn. The framework matches each `ToolCall` to a registered tool, executes
/// it, and produces a [`ToolDispatchResult`] with the output.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence and inter-process
/// communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Assigned by the LLM API. Correlates with [`ToolDispatchResult::tool_call_id`].
    pub id: String,

    /// Must match a tool registered in the `ToolRegistry`.
    pub tool: String,

    /// JSON object whose schema depends on the tool.
    pub input: serde_json::Value,
}

impl ToolCall {
    /// Apply a [`Correction`](crate::reflection::Correction) from the reflection system in place.
    ///
    /// Modifies `self` according to the correction strategy:
    ///
    /// - [`InputFix`](crate::reflection::CorrectionType::InputFix) — replaces
    ///   `self.input` with the corrected input.
    /// - [`ToolChange`](crate::reflection::CorrectionType::ToolChange) — replaces
    ///   `self.tool` with an alternative tool name.
    /// - Other types — no mutation; the retry proceeds with unchanged parameters.
    ///
    /// Returns a [`CorrectionResult`](crate::reflection::CorrectionResult) indicating whether the correction
    /// was applied, failed, or skipped.
    pub fn apply_correction(
        &mut self,
        correction: &crate::reflection::Correction,
        _prior_result: &crate::tool::ToolDispatchResult,
    ) -> crate::reflection::CorrectionResult {
        use crate::reflection::CorrectionType;
        match correction.correction_type {
            CorrectionType::InputFix => {
                if let Some(ref modified) = correction.modified_input {
                    tracing::debug!(
                        tool = %self.tool,
                        "applying InputFix correction from reflector"
                    );
                    self.input = modified.clone();
                    crate::reflection::CorrectionResult::Applied
                } else {
                    crate::reflection::CorrectionResult::Failed(
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
                    crate::reflection::CorrectionResult::Applied
                } else {
                    crate::reflection::CorrectionResult::Failed(
                        "ToolChange correction missing alternative_tool".to_string(),
                    )
                }
            }
            CorrectionType::PrerequisiteFix | CorrectionType::ApproachChange => {
                crate::reflection::CorrectionResult::Skipped
            }
            CorrectionType::Escalate => crate::reflection::CorrectionResult::Skipped,
        }
    }
}

// ==================================================
// SessionResult
// ==================================================

/// Summary of a complete agent session.
///
/// Produced by `Loop::finalize`
/// after the last turn. Aggregates all session-level metrics: total turns,
/// tokens, duration, tool calls, and final output.
///
/// # Construction
///
/// Use [`SessionResult::success`] for a completed session or
/// [`SessionResult::failed`] for a session that ended with an error.
///
/// ```
/// use loopctl::engine::loop_core::SessionResult;
/// use uuid::Uuid;
///
/// let session_id = Uuid::new_v4();
///
/// let ok = SessionResult::success(session_id);
/// assert!(ok.success);
///
/// let err = SessionResult::failed(session_id, "API rate limit exceeded");
/// assert!(!err.success);
/// assert_eq!(err.error.unwrap(), "API rate limit exceeded");
/// ```
#[derive(Debug, Clone)]
pub struct SessionResult {
    /// Matches [`LoopConfig::session_id`].
    pub session_id: Uuid,
    /// Total turns executed. Compared against [`LoopConfig::max_turns`].
    pub total_turns: usize,
    /// Sum of [`TurnResult::input_tokens`] across all turns.
    pub input_tokens: u64,
    /// Sum of [`TurnResult::output_tokens`] across all turns.
    pub output_tokens: u64,
    /// Wall-clock time from session start to finalization.
    pub total_duration: Duration,
    /// Total number of tool calls executed across all turns.
    pub tool_calls: usize,
    /// `true` on [`LoopState::Completed`], `false` on [`LoopState::Failed`].
    pub success: bool,
    /// Last meaningful text response from the agent. `None` if no final message.
    pub final_output: Option<String>,
    /// `Some` on failure with a human-readable error description.
    pub error: Option<String>,
}

impl Default for SessionResult {
    fn default() -> Self {
        Self {
            session_id: Uuid::nil(),
            total_turns: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_duration: Duration::ZERO,
            tool_calls: 0,
            success: false,
            final_output: None,
            error: None,
        }
    }
}

impl SessionResult {
    /// Create a successful session result.
    ///
    /// Initializes all counters to zero and sets [`success`](SessionResult::success)
    /// to `true`. The framework or production code should fill in the actual
    /// counters before returning.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::engine::loop_core::SessionResult;
    /// use uuid::Uuid;
    ///
    /// let session_id = Uuid::new_v4();
    /// let result = SessionResult::success(session_id);
    /// assert!(result.success);
    /// assert!(result.error.is_none());
    /// ```
    #[must_use]
    pub fn success(session_id: Uuid) -> Self {
        Self {
            session_id,
            total_turns: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_duration: Duration::ZERO,
            tool_calls: 0,
            success: true,
            final_output: None,
            error: None,
        }
    }

    /// Create a failed session result.
    ///
    /// Sets [`success`](SessionResult::success) to `false` and records the
    /// error message. All counters are initialized to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::engine::loop_core::SessionResult;
    /// use uuid::Uuid;
    ///
    /// let session_id = Uuid::new_v4();
    /// let result = SessionResult::failed(session_id, "API rate limit exceeded");
    /// assert!(!result.success);
    /// assert_eq!(result.error.unwrap(), "API rate limit exceeded");
    /// ```
    #[must_use]
    pub fn failed(session_id: Uuid, error: impl Into<String>) -> Self {
        Self {
            session_id,
            total_turns: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_duration: Duration::ZERO,
            tool_calls: 0,
            success: false,
            final_output: None,
            error: Some(error.into()),
        }
    }

    /// Total tokens (input + output) for this session.
    ///
    /// Sums [`input_tokens`](SessionResult::input_tokens)
    /// and [`output_tokens`](SessionResult::output_tokens).
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ==================================================
// Loop trait
// ==================================================

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
/// use loopctl::engine::loop_core::Loop;
/// use loopctl::error::LoopError;
/// use loopctl::engine::loop_core::{LoopConfig, LoopState, SessionResult, TurnResult};
///
/// struct MyAgent {
///     state: MyState,
/// }
///
/// impl Loop for MyAgent {
///     fn initialize<'a>(&'a mut self, config: &'a LoopConfig) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
///         Box::pin(async { Ok(()) })
///     }
///     fn process_turn<'a>(&'a mut self, input: &'a str) -> Pin<Box<dyn Future<Output = Result<TurnResult, LoopError>> + Send + 'a>> {
///         Box::pin(async { Ok(TurnResult::completed("Done!")) })
///     }
///     fn should_continue(&self) -> bool {
///         !self.state.is_complete
///     }
///     fn finalize<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Result<SessionResult, LoopError>> + Send + 'a>> {
///         Box::pin(async { Ok(SessionResult::success(self.state.session_id)) })
///     }
///     fn state(&self) -> LoopState {
///         LoopState::Idle
///     }
///     fn cancel(&self) {}
/// }
/// ```
pub trait Loop: Send + Sync {
    /// Initialize the agent with the given configuration.
    ///
    /// Called once before any turns are processed. Use this to set up
    /// internal state, validate configuration, and prepare resources.
    fn initialize<'a>(
        &'a mut self,
        config: &'a LoopConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>>;

    /// Process a single user message / turn.
    ///
    /// Main entry point for agent logic. It receives the user's
    /// input and returns a [`TurnResult`] describing what happened.
    fn process_turn<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult, LoopError>> + Send + 'a>>;

    /// Check whether the agent should continue processing turns.
    ///
    /// Called after each turn. Return `false` to end the session.
    fn should_continue(&self) -> bool;

    /// Finalize the agent session and produce a summary.
    ///
    /// Called once after the last turn. Use this to clean up resources
    /// and produce a final [`SessionResult`].
    fn finalize<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<SessionResult, LoopError>> + Send + 'a>>;

    /// Get the current state of the agent.
    ///
    /// Used by the framework to drive the state machine and by observers
    /// to report status.
    fn state(&self) -> LoopState;

    /// Cancel the agent's current operation.
    ///
    /// Implementations must use thread-safe interior mutability (e.g.
    /// [`AtomicBool`](std::sync::atomic::AtomicBool), `Mutex<bool>`) to
    /// store the cancellation flag, since this method takes `&self`. The
    /// flag should be set in a non-blocking fashion so that
    /// [`process_turn`](Loop::process_turn) and
    /// [`should_continue`](Loop::should_continue) can observe it
    /// and return promptly across threads.
    fn cancel(&self);

    /// Explain *why* [`should_continue`](Loop::should_continue) returned `false`.
    ///
    /// Called by [`run`](Loop::run) after the drive loop exits. Return:
    ///
    /// - `None` — the session ended normally (model finished).
    /// - `Some(err)` — the session was forced to stop (`Cancelled`,
    ///   `MaxTurnsExceeded`, etc.).
    ///
    /// The default implementation returns `None` (normal completion).
    fn stop_reason(&self) -> Option<LoopError> {
        None
    }

    /// Drive the full agent session: initialize → turn loop → finalize.
    ///
    /// This is the main entry point for running an agent. It calls
    /// [`initialize`](Loop::initialize) with the agent's stored config,
    /// then repeatedly calls [`process_turn`](Loop::process_turn) until either:
    ///
    /// - The turn result is marked `is_complete` (the model finished), or
    /// - [`should_continue`](Loop::should_continue) returns `false`.
    ///
    /// When `should_continue` returns `false`,
    /// [`stop_reason`](Loop::stop_reason) is consulted to distinguish
    /// normal completion from an error (cancellation, max-turns, etc.).
    ///
    /// # Errors
    ///
    /// - [`LoopError::Cancelled`] — if the session was cancelled.
    /// - [`LoopError::MaxTurnsExceeded`] — if the turn limit was reached.
    /// - Any error returned by [`process_turn`](Loop::process_turn) or
    ///   [`finalize`](Loop::finalize).
    fn run<'a>(
        &'a mut self,
        user_input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SessionResult, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            self.initialize(&self.config()).await?;

            loop {
                if !self.should_continue() {
                    break;
                }

                match self.process_turn(user_input).await {
                    Ok(turn_result) if turn_result.is_complete => {
                        return self.finalize().await;
                    }
                    Ok(_) => { /* turn produced tool calls — continue */ }
                    Err(e) => {
                        self.finalize().await?;
                        return Err(e);
                    }
                }
            }

            // should_continue() returned false — ask the impl why.
            if let Some(err) = self.stop_reason() {
                self.finalize().await?;
                return Err(err);
            }

            self.finalize().await
        })
    }

    /// Return the configuration that [`run`](Loop::run) passes to
    /// [`initialize`](Loop::initialize).
    ///
    /// Implementors should return the [`LoopConfig`] they want to use
    /// for the session.
    fn config(&self) -> LoopConfig;
}
