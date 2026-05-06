//! Common types shared across the agent framework.
//!
//! This module defines the foundational data types that every other module in
//! the framework depends on: configuration ([`AgentConfig`]), lifecycle state
//! ([`AgentState`]), turn and session results ([`TurnResult`], [`SessionResult`]),
//! tool call representations ([`ToolCall`], [`ToolCallResult`]), and the
//! reflection/correction system ([`Correction`], [`CorrectionType`],
//! [`CorrectionResult`]).
//!
//! These types are intentionally framework-level — they contain no
//! domain-specific or production-specific logic. Production crates should
//! extend them via composition rather than modification.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::core::types::{AgentConfig, AgentState, TurnResult, SessionResult};
//!
//! let config = AgentConfig::default();
//! assert_eq!(config.max_turns, 200);
//!
//! let turn = TurnResult::completed("Task done.");
//! assert!(turn.is_complete);
//!
//! let session = SessionResult::success(config.session_id);
//! assert!(session.success);
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

// ===================================================
// Agent configuration
// ===================================================

/// Configuration for an agent session.
///
/// Holds generic agent configuration fields that apply to every session
/// regardless of the specific agent type. Domain-specific configuration
/// (e.g., ITR engine settings, `ToolShield` rules, fallback model chains)
/// should live in production-specific config types that embed or wrap
/// this struct.
///
/// # Construction
///
/// Use [`AgentConfig::default`] for sensible defaults or override individual
/// fields through the builder via
/// `AgentBuilder::with_config`.
///
/// ```
/// use loopctl::core::types::AgentConfig;
///
/// let config = AgentConfig {
///     max_turns: 50,
///     model: "default".to_string(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Unique session identifier.
    ///
    /// Automatically generated as a UUID v4 on construction. Used for
    /// correlating logs, metrics, and observer events across the
    /// lifetime of a single agent session.
    pub session_id: Uuid,

    /// Model identifier (e.g. `"default"`).
    ///
    /// Production consumers should override this via the builder, but the
    /// default provides a reasonable fallback. The value is passed to the
    /// API client on each request.
    pub model: String,

    /// Optional system prompt override.
    ///
    /// When `Some`, replaces the agent's default system prompt for this
    /// session. When `None`, the agent core decides its own system prompt.
    /// Set via `AgentBuilder::system_prompt`.
    pub system_prompt: Option<String>,

    /// Maximum number of turns before forcing completion.
    ///
    /// Prevents runaway sessions. After `max_turns` turns, the framework
    /// forces the session to end regardless of
    /// `AgentCore::should_continue`.
    /// Defaults to `200`.
    pub max_turns: usize,

    /// Maximum tokens for each API response.
    ///
    /// Controls the length of each individual LLM response. Lower values
    /// reduce latency and cost; higher values allow longer responses.
    /// Defaults to `16_384`.
    pub max_tokens: u32,

    /// Context window size for the model (in tokens).
    ///
    /// Used by the auto-compaction system to estimate when the conversation
    /// is approaching the model's context limit. Defaults to `200_000`.
    ///
    /// Must match the actual context window of the configured [`model`](AgentConfig::model).
    pub context_window: u64,

    /// Threshold percentage to trigger auto-compaction (0.0–1.0).
    ///
    /// When the estimated token usage exceeds `compact_threshold * context_window`,
    /// the auto-compaction system triggers a context compaction pass.
    /// Defaults to `0.80` (80% of the context window).
    pub compact_threshold: f64,

    /// Whether auto-compaction is enabled.
    ///
    /// When `true`, the framework automatically compacts the conversation
    /// context when usage exceeds [`compact_threshold`](AgentConfig::compact_threshold).
    /// Set via `AgentBuilder::auto_compact`.
    /// Defaults to `true`.
    pub auto_compact: bool,
}

impl Default for AgentConfig {
    /// Produce a configuration with production-ready defaults.
    ///
    /// | Field | Default |
    /// |-------|---------|
    /// | [`session_id`](AgentConfig::session_id) | Random UUID v4 |
    /// | [`model`](AgentConfig::model) | `"default"` |
    /// | [`system_prompt`](AgentConfig::system_prompt) | `None` |
    /// | [`max_turns`](AgentConfig::max_turns) | `200` |
    /// | [`max_tokens`](AgentConfig::max_tokens) | `16_384` |
    /// | [`context_window`](AgentConfig::context_window) | `200_000` |
    /// | [`compact_threshold`](AgentConfig::compact_threshold) | `0.80` |
    /// | [`auto_compact`](AgentConfig::auto_compact) | `true` |
    ///
    /// # When called
    ///
    /// By `AgentBuilder::new` to seed the builder, or by consumers who need a
    /// baseline before overriding fields.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::types::AgentConfig;
    ///
    /// let config = AgentConfig::default();
    /// assert_eq!(config.max_turns, 200);
    /// assert_eq!(config.model, "default");
    /// ```
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            model: "default".to_string(),
            system_prompt: None,
            max_turns: 200,
            max_tokens: 16_384,
            context_window: 200_000,
            compact_threshold: 0.80,
            auto_compact: true,
        }
    }
}

// ===================================================
// Agent lifecycle state
// ===================================================

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
pub enum AgentState {
    /// The agent is idle, waiting for a user message.
    ///
    /// This is the initial state after initialization and the state
    /// the agent returns to between user inputs.
    ///
    /// No background work is performed while idle.
    Idle,

    /// The agent is actively processing a turn.
    ///
    /// Entered when the agent core begins processing a user message
    /// or a tool result. The `turn` field tracks progress for observers.
    ///
    /// The agent transitions here from [`Idle`](AgentState::Idle) or
    /// [`WaitingForTool`](AgentState::WaitingForTool).
    Processing {
        /// Current turn number (0-indexed).
        ///
        /// Incremented at the start of each new API call cycle. Used
        /// by observers to track session progress and by the framework
        /// to enforce [`AgentConfig::max_turns`].
        turn: usize,
    },

    /// The agent is waiting for a tool to complete.
    ///
    /// Entered after the LLM requests a tool call. The framework
    /// dispatches the tool and waits for its result before returning
    /// to [`Processing`](AgentState::Processing).
    WaitingForTool {
        /// Name of the tool being executed.
        ///
        /// Corresponds to the `name` field of [`ToolCall`]. Used by observers for
        /// latency tracking and by the detection system for loop analysis.
        ///
        /// Never empty — always matches a registered tool name.
        tool: String,

        /// When the tool call started.
        ///
        /// Recorded as [`SystemTime::now`] when the tool dispatch begins.
        /// Used to compute tool execution duration for metrics and
        /// timeout enforcement.
        started_at: SystemTime,
    },

    /// The agent is compacting its conversation context.
    ///
    /// Entered when token usage exceeds the
    /// [`compact_threshold`](AgentConfig::compact_threshold) or when
    /// compaction is explicitly requested. The agent summarizes older
    /// messages to free context space, then returns to
    /// [`Processing`](AgentState::Processing).
    Compacting {
        /// Why compaction was triggered.
        ///
        /// See [`CompactReason`] for the possible triggers.
        ///
        /// Carried through to observers for compaction analytics.
        reason: CompactReason,
    },

    /// The agent is reflecting on a failure and preparing a correction.
    ///
    /// Entered when a tool call fails and the reflection system is
    /// enabled (via `Feature::Reflection`).
    /// The agent analyzes the error and produces a [`Correction`] before
    /// retrying.
    Reflecting {
        /// Number of errors being analyzed.
        ///
        /// Helps the agent core gauge how many past failures to consider
        /// when formulating a correction strategy.
        ///
        /// A higher count may indicate a systematic issue requiring
        /// an [`ApproachChange`](CorrectionType::ApproachChange).
        error_count: usize,
    },

    /// The agent has completed its task.
    ///
    /// Terminal state. The framework calls
    /// `AgentCore::finalize` to produce
    /// a [`SessionResult`].
    Completed {
        /// Summary of what was accomplished.
        ///
        /// Typically the last text output from the agent core. Included
        /// in [`SessionResult::final_output`] for consumer inspection.
        ///
        /// May be empty if the agent produced only tool calls.
        summary: String,
    },

    /// The agent has failed with an unrecoverable error.
    ///
    /// Terminal state. The error is propagated through
    /// [`SessionResult::error`].
    ///
    /// No further turns will be executed after entering this state.
    Failed {
        /// The error that caused the failure.
        ///
        /// A human-readable description of the unrecoverable error.
        /// Used for logging and for inclusion in
        /// [`SessionResult::error`].
        error: String,
    },
}

/// Reason for triggering context compaction.
///
/// Indicates why the framework decided to compact the conversation context.
/// Carried inside [`AgentState::Compacting`] and used by observers for
/// compaction analytics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactReason {
    /// Token usage crossed the auto-compaction threshold.
    ///
    /// Triggered when estimated token usage exceeds
    /// [`AgentConfig::compact_threshold`] × [`AgentConfig::context_window`].
    ///
    /// This is the normal, expected compaction path.
    AutoThreshold,

    /// An explicit user request to compact.
    ///
    /// Triggered by an external API call or control signal asking the
    /// agent to compact immediately, regardless of current token usage.
    ///
    /// Useful when the consumer knows the conversation is about to grow significantly.
    UserRequested,

    /// Emergency compaction due to context window overflow.
    ///
    /// Triggered when the API rejects a request because the input exceeds
    /// the model's context window. This is a last-resort compaction that
    /// aggressively summarizes to recover.
    Emergency,
}

// ===================================================
// Turn result
// ===================================================

/// Result of a single agent turn (one API call → response cycle).
///
/// Produced by `AgentCore::process_turn`
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
/// use loopctl::core::types::TurnResult;
///
/// let done = TurnResult::completed("All tasks finished.");
/// assert!(done.is_complete);
///
/// let more = TurnResult::continuing("Still working...");
/// assert!(!more.is_complete);
/// ```
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// The text content of the assistant's response.
    ///
    /// May be empty if the response consists entirely of tool calls.
    /// Combined with tool results in the session's final output.
    ///
    /// See [`TurnResult::tool_calls`] for the accompanying tool invocations.
    pub text: String,

    /// Tool calls made during this turn.
    ///
    /// Each entry represents a tool the LLM requested to execute. The
    /// framework dispatches them and collects results into
    /// [`tool_results`](TurnResult::tool_results).
    pub tool_calls: Vec<ToolCall>,

    /// Results from tool executions during this turn.
    ///
    /// Populated after tool dispatch. Each result corresponds to a
    /// [`ToolCall`] by matching [`tool_call_id`](ToolCallResult::tool_call_id)
    /// to [`ToolCall::id`].
    pub tool_results: Vec<ToolCallResult>,

    /// How many tokens were used in the API request.
    ///
    /// Includes the system prompt, conversation history, and the user
    /// message. Used for cost tracking and compaction decisions.
    ///
    /// Reported by the provider in the response metadata.
    pub input_tokens: u64,

    /// How many tokens were in the API response.
    ///
    /// Includes response text and tool call definitions. Used for cost
    /// tracking and [`total_tokens`](TurnResult::total_tokens).
    ///
    /// Reported by the provider in the response metadata.
    pub output_tokens: u64,

    /// Wall-clock duration of this turn.
    ///
    /// Measures the full time from sending the API request through receiving
    /// the complete response and executing any tool calls.
    pub duration: Duration,

    /// Whether the agent considers the task complete.
    ///
    /// When `true`, the framework will not invoke
    /// `should_continue` and
    /// will proceed to finalization.
    pub is_complete: bool,

    /// The stop reason reported by the API.
    ///
    /// Indicates why the LLM stopped generating. Used by the framework
    /// to decide whether to dispatch tools ([`StopReason::ToolCall`]) or
    /// continue the conversation.
    pub stop_reason: StopReason,
}

impl TurnResult {
    /// Create a completed turn result with a simple text response.
    ///
    /// Sets [`is_complete`](TurnResult::is_complete) to `true` and all
    /// token counters to zero. Use this for the final turn of a session.
    ///
    /// # When called
    ///
    /// Called by agent core implementations to signal that the task is
    /// done and no further turns are needed.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::types::TurnResult;
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
    /// # When called
    ///
    /// Called by agent core implementations when a turn produces intermediate
    /// output (e.g., tool results to process) and the session is not yet done.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::types::TurnResult;
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
    ///
    /// # When called
    ///
    /// Called by the framework to decide whether to enter the tool-dispatch
    /// path or proceed to the next turn.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Total tokens (input + output) for this turn.
    ///
    /// Convenience method that sums [`input_tokens`](TurnResult::input_tokens)
    /// and [`output_tokens`](TurnResult::output_tokens). Used for session-level
    /// token accounting.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ===================================================
// Stop reason
// ===================================================

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
    /// The LLM hit the [`AgentConfig::max_tokens`] limit before
    /// finishing. The response may be truncated. The framework may
    /// choose to continue the turn to let the model complete its output.
    MaxTokens,

    /// The stop sequence was encountered.
    ///
    /// The model generated a configured stop sequence. Rare in
    /// standard usage; typically indicates custom API configuration.
    StopSequence,
}

// ===================================================
// Tool call
// ===================================================

/// A tool call requested by the agent.
///
/// Represents a single tool invocation that the LLM has requested during a
/// turn. The framework matches each `ToolCall` to a registered tool, executes
/// it, and produces a [`ToolCallResult`] with the output.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence and inter-process
/// communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    ///
    /// Assigned by the LLM API. Used to correlate the call with its
    /// [`ToolCallResult`] via [`ToolCallResult::tool_call_id`].
    pub id: String,

    /// Name of the tool to invoke.
    ///
    /// Must match a tool registered in the
    /// `ToolRegistry`. Examples: `"Read"`,
    /// `"Bash"`, `"Write"`.
    pub tool: String,

    /// Input parameters for the tool.
    ///
    /// A JSON object whose schema depends on the tool. For example,
    /// a `Read` tool might have `{"file_path": "/path/to/file"}`.
    /// The framework passes this directly to the tool implementation.
    pub input: serde_json::Value,
}

/// The result of a single tool execution.
///
/// Produced after the framework dispatches a [`ToolCall`] and collects the
/// tool's output. Contains the result text (or error message), a flag
/// indicating success or failure, and the execution duration.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// The tool call this result is for.
    ///
    /// Matches [`ToolCall::id`] to correlate results back to their
    /// originating requests.
    pub tool_call_id: String,

    /// The tool's output.
    ///
    /// On success, contains the tool's return value (e.g., file contents,
    /// command output). On error, contains a human-readable error message.
    pub output: String,

    /// Whether the tool execution resulted in an error.
    ///
    /// When `true`, [`output`](ToolCallResult::output) contains the error
    /// message and the framework may trigger reflection or retry logic.
    pub is_error: bool,

    /// Duration of tool execution.
    ///
    /// Measures wall-clock time from tool dispatch start to completion.
    /// Used by observers for latency tracking and by the detection system
    /// for timeout analysis.
    pub duration: Duration,
}

// ===================================================
// Session result
// ===================================================

/// Summary of a complete agent session.
///
/// Produced by `AgentCore::finalize`
/// after the last turn. Aggregates all session-level metrics: total turns,
/// tokens, duration, tool calls, and final output.
///
/// # Construction
///
/// Use [`SessionResult::success`] for a completed session or
/// [`SessionResult::failed`] for a session that ended with an error.
///
/// ```
/// use loopctl::core::types::SessionResult;
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
    /// The session identifier.
    ///
    /// Matches [`AgentConfig::session_id`]. Used for correlating this
    /// result with logs, metrics, and observer events.
    pub session_id: Uuid,

    /// Total number of turns executed.
    ///
    /// Counts every API call cycle, including tool-dispatch turns.
    /// Compared against [`AgentConfig::max_turns`] to detect runaway
    /// sessions.
    pub total_turns: usize,

    /// Total input tokens consumed across all turns.
    ///
    /// Sum of [`TurnResult::input_tokens`] for every turn in the session.
    /// Used for cost reporting (input tokens are typically cheaper).
    pub input_tokens: u64,

    /// Total output tokens consumed across all turns.
    ///
    /// Sum of [`TurnResult::output_tokens`] for every turn in the session.
    /// Used for cost reporting (output tokens are typically more expensive).
    pub output_tokens: u64,

    /// Total wall-clock time.
    ///
    /// Measures from session start to finalization. Includes all API
    /// calls, tool executions, and compaction passes.
    pub total_duration: Duration,

    /// Number of tool calls made.
    ///
    /// Counts every [`ToolCall`] dispatched during the session. Useful
    /// for understanding agent behavior and cost.
    pub tool_calls: usize,

    /// Whether the session completed successfully.
    ///
    /// `true` when the agent reached [`AgentState::Completed`], `false`
    /// when it reached [`AgentState::Failed`].
    pub success: bool,

    /// Final text output from the agent (if any).
    ///
    /// Contains the last meaningful text response from the agent core,
    /// typically from [`AgentState::Completed`]. `None` if the session
    /// ended without a final message.
    pub final_output: Option<String>,

    /// Error message if the session failed.
    ///
    /// `Some` when [`success`](SessionResult::success) is `false`,
    /// containing a human-readable description of what went wrong.
    /// `None` for successful sessions.
    pub error: Option<String>,
}

impl SessionResult {
    /// Create a successful session result.
    ///
    /// Initializes all counters to zero and sets [`success`](SessionResult::success)
    /// to `true`. The framework or production code should fill in the actual
    /// counters before returning.
    ///
    /// # When called
    ///
    /// Called by `AgentCore::finalize`
    /// implementations when the session completed without error.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::types::SessionResult;
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
    /// # When called
    ///
    /// Called by `AgentCore::finalize`
    /// implementations when the session ended due to an unrecoverable error.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::core::types::SessionResult;
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
    /// Convenience method that sums [`input_tokens`](SessionResult::input_tokens)
    /// and [`output_tokens`](SessionResult::output_tokens).
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ===================================================
// Correction system
// ===================================================

/// A correction produced by the reflection system.
///
/// When a tool call fails and reflection is enabled (via
/// `Feature::Reflection`), the agent
/// analyzes the error and produces a `Correction` that describes how to fix
/// the problem. The framework applies the correction and retries.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// Type of correction to apply.
    ///
    /// Categorizes the fix strategy. See [`CorrectionType`] for the
    /// available strategies and their semantics.
    pub correction_type: CorrectionType,

    /// Human-readable description of the correction.
    ///
    /// Explains *what* went wrong and *how* the correction addresses it.
    /// Used for logging, observability, and agent guidance.
    pub description: String,

    /// Modified tool input (if applicable).
    ///
    /// When [`correction_type`](Correction::correction_type) is
    /// [`InputFix`](CorrectionType::InputFix), contains the corrected
    /// JSON input that should replace the original. `None` when the
    /// correction does not modify tool input.
    pub modified_input: Option<serde_json::Value>,

    /// Alternative tool to use (if applicable).
    ///
    /// When [`correction_type`](Correction::correction_type) is
    /// [`ToolChange`](CorrectionType::ToolChange), contains the name
    /// of the tool that should be used instead. `None` when the
    /// correction keeps the same tool.
    pub alternative_tool: Option<String>,

    /// Additional guidance for the agent.
    ///
    /// Free-form text that provides extra context or instructions to
    /// help the agent avoid the same failure in future turns.
    pub guidance: Option<String>,
}

/// Type of correction to apply.
///
/// Categorizes the fix strategy that the reflection system has determined
/// is most appropriate for the observed failure. Each variant maps to a
/// different retry approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionType {
    /// Fix the input to the tool.
    ///
    /// The tool was correct but its input parameters were wrong (e.g., a
    /// typo in a file path). The correction provides a fixed input via
    /// [`Correction::modified_input`].
    InputFix,

    /// Use a different tool.
    ///
    /// The chosen tool was inappropriate for the task. The correction
    /// specifies an alternative via [`Correction::alternative_tool`].
    ToolChange,

    /// Fix a dependency or prerequisite.
    ///
    /// The tool call failed because a prerequisite was not met (e.g.,
    /// a directory doesn't exist). The correction describes what needs
    /// to be done first.
    PrerequisiteFix,

    /// Change the approach entirely.
    ///
    /// The current strategy is fundamentally flawed. The correction
    /// provides high-level guidance for a different approach via
    /// [`Correction::guidance`].
    ApproachChange,

    /// No fix possible, escalate.
    ///
    /// The reflection system cannot determine a correction. The
    /// framework should propagate the error to the user or higher-level
    /// handler.
    Escalate,
}

/// Result of applying a correction.
///
/// Indicates whether the reflection system's correction was successfully
/// applied, failed, or was skipped. Produced after attempting to retry
/// with the corrected parameters.
#[derive(Debug, Clone)]
pub enum CorrectionResult {
    /// Correction was applied successfully.
    ///
    /// The retry with corrected parameters succeeded and the agent can
    /// continue processing normally.
    Applied,

    /// Correction failed.
    ///
    /// The retry also failed. Contains a human-readable error message
    /// describing what went wrong with the corrected attempt.
    Failed(String),

    /// No correction was needed or possible.
    ///
    /// The reflection system decided not to apply a correction (e.g.,
    /// the error is transient or the correction type was
    /// [`Escalate`](CorrectionType::Escalate)).
    Skipped,
}
