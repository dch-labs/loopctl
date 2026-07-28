//! Reflection and recovery for failed agent turns.
//!
//! When a tool call fails or a turn produces an unexpected result, the
//! framework needs to decide what to do. A pluggable
//! two-layer system:
//!
//! 1. **[`Reflector`]** — Analyses the failure and produces a
//!    [`FailureAnalysis`] describing what went wrong and whether it's
//!    recoverable.
//!
//! 2. **[`RecoveryStrategy`]** — Takes the analysis and decides on a
//!    [`RecoveryAction`] (retry, skip, ask user, or fail).
//!
//! Both layers are trait-based so agents can plug in their own strategies.
//! The framework provides reference implementations:
//!
//! - [`NoopReflector`] — marks everything as non-recoverable (safe default).
//! - [`ExponentialBackoffRecovery`] — retries with exponential backoff up to
//!   a configurable limit.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::reflection::{
//!     NoopReflector, ExponentialBackoffRecovery, ReflectionContext,
//! };
//! use std::sync::Arc;
//!
//! let reflector = NoopReflector;
//! let strategy = ExponentialBackoffRecovery::new(3);
//!
//! // Usage (in BareLoop):
//! // let analysis = reflector.analyze(error, tool_name, input, &context).await?;
//! // let action = strategy.decide(&analysis, attempt, max_attempts).await;
//! ```

pub mod backoff;
pub use backoff::ExponentialBackoffRecovery;

pub mod llm;
pub use llm::LlmReflector;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// How severe a failure is.
///
/// Ordered from least to most severe. The [`RecoveryStrategy`] may use
/// severity to decide whether retrying is worthwhile — a `Low` severity
/// issue (e.g., a transient network blip) is more retryable than a
/// `Critical` one (e.g., invalid API key).
///
/// # Ordering and comparison
///
/// Derives [`Ord`], so variants can be compared directly: `Low < Medium <
/// High < Critical`. Strategies commonly write thresholds like
/// `analysis.severity >= FailureSeverity::High` to gate retries.
///
/// # Serialization
///
/// Serializes to and from the lowercase snake-case form of the variant
/// name (`"low"`, `"medium"`, `"high"`, `"critical"`) via
/// `#[serde(rename_all = "snake_case")]`. The same four strings are the
/// `enum` values in the [`FailureAnalysis`] JSON Schema, so model output
/// round-trips through deserialization without renaming.
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::FailureSeverity;
///
/// assert!(FailureSeverity::Low < FailureSeverity::Critical);
/// ```
///
/// [`FailureAnalysis`]: crate::reflection::FailureAnalysis
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    /// Minor issue — a retry will likely fix it.
    ///
    /// Typical causes: a transient network blip, a momentary rate-limit,
    /// or a tool that succeeded on a second attempt without any input
    /// changes. Strategies usually retry immediately on `Low` severity.
    Low,

    /// Moderate issue — may need a correction before retrying.
    ///
    /// The failure looks recoverable, but a bare retry is less likely to
    /// succeed than at `Low`: the input may have a small mistake (a typo
    /// in a path, a slightly wrong argument), or the tool's
    /// preconditions may need a step first. Strategies commonly consult
    /// [`FailureAnalysis::correction`] before retrying at this severity.
    ///
    /// [`FailureAnalysis::correction`]: crate::reflection::FailureAnalysis::correction
    Medium,

    /// Serious issue — retrying without changes is unlikely to help.
    ///
    /// The call is fundamentally off: the wrong tool was chosen, the
    /// argument type is incorrect, or the task itself is misframed. A
    /// retry that doesn't first apply a [`Correction`] is probably
    /// wasted. Strategies may still retry when a correction is supplied
    /// and the attempt budget allows, but should treat `High` as a
    /// signal to slow down rather than retry blindly.
    ///
    /// [`Correction`]: crate::reflection::Correction
    High,

    /// Unrecoverable — the agent should stop or escalate.
    ///
    /// No correction can rescue this turn. Typical causes: an invalid
    /// API key, a permissions failure the agent can't self-resolve, or a
    /// bug in the tool itself. Strategies usually map `Critical` to
    /// [`RecoveryAction::Fail`] (or
    /// [`RecoveryAction::AskUser`] when interactive recovery is an
    /// option) rather than retry.
    ///
    /// [`RecoveryAction::Fail`]: crate::reflection::RecoveryAction::Fail
    /// [`RecoveryAction::AskUser`]: crate::reflection::RecoveryAction::AskUser
    Critical,
}

impl fmt::Display for FailureSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Context provided to [`Reflector::analyze()`] describing the retry state.
///
/// Built by the framework before invoking the reflector. Carries what the
/// agent was trying to do and where it is in the retry budget, so a
/// reflector can factor both into its analysis — e.g., be more conservative
/// with suggested corrections on the last permitted attempt.
///
/// # Lifecycle
///
/// The engine constructs a fresh `ReflectionContext` for each failure
/// (see `recover_tool_error` in `engine/bare/dispatch.rs`) and passes it by
/// reference to [`Reflector::analyze`]. It is not stored across calls;
/// reflectors that want to track cross-failure history must keep their own
/// state.
///
/// # Attempt indexing
///
/// `attempt` is 0-indexed: the first try of a tool is `attempt = 0`. A
/// reflector rendering the value for a model prompt should add 1 (the
/// framework's built-in `LlmReflector` does this — "Attempt: 1 of N").
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::ReflectionContext;
///
/// let context = ReflectionContext {
///     task: "Fix the bug in main.rs".to_string(),
///     attempt: 2,
///     max_attempts: 5,
/// };
/// assert_eq!(context.attempt, 2);
/// assert_eq!(context.max_attempts, 5);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReflectionContext {
    /// What the agent was trying to accomplish when the failure occurred.
    ///
    /// Free-form text — typically the original user message, a summary of
    /// the current step, or an empty string when the engine has no task
    /// description to share. A reflector may include this in its prompt so
    /// the model can reason about whether the failure is relevant to the
    /// stated goal.
    pub task: String,

    /// Current attempt number for this tool call, 0-indexed.
    ///
    /// `0` is the first attempt; the framework increments this on each
    /// retry within the recovery loop. Compare against
    /// [`max_attempts`](Self::max_attempts) to know how much budget
    /// remains. Render as `attempt + 1` when showing the value to a user
    /// or model.
    pub attempt: u32,

    /// Maximum attempts allowed before the framework gives up on this
    /// tool call.
    ///
    /// Set by the engine to its configured recovery ceiling
    /// (`BareLoop::MAX_RECOVERY_ATTEMPTS`). When `attempt >= max_attempts`,
    /// a [`RecoveryStrategy`] should typically return
    /// [`RecoveryAction::Fail`] rather than schedule another retry.
    ///
    /// [`RecoveryAction::Fail`]: crate::reflection::RecoveryAction::Fail
    pub max_attempts: u32,
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

/// A correction produced by the reflection system.
///
/// When a tool call fails and reflection is enabled (via configuration),
/// the agent analyzes the error and produces a `Correction` that describes
/// how to fix the problem. The framework applies the correction and
/// retries — `recover_tool_error` in `engine/bare/dispatch.rs` clones the
/// `Correction` out of the [`FailureAnalysis`] and feeds it back into the
/// retry loop.
///
/// # Which fields apply when
///
/// The fields are deliberately permissive (four `Option`s + one enum) so a
/// single shape covers every [`CorrectionType`] strategy. The convention is
/// that only the fields named by the variant's doc are meaningful for a
/// given `correction_type`; consumers should consult `correction_type`
/// first and read the relevant fields accordingly rather than treating
/// every `Some` as load-bearing. There is no runtime enforcement of this
/// pairing — a reflector that fills `modified_input` while declaring
/// `correction_type: Escalate` will not be rejected.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence (e.g., writing
/// analyses to a session log) and so an [`LlmReflector`] can request it
/// back as a nested object inside a [`FailureAnalysis`] via the
/// [`StructuredOutput`] trait.
///
/// [`LlmReflector`]: crate::reflection::LlmReflector
/// [`StructuredOutput`]: crate::structured::StructuredOutput
/// [`FailureAnalysis`]: crate::reflection::FailureAnalysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// Which fix strategy this correction represents.
    ///
    /// Drives which of the remaining fields the framework will actually
    /// consume on retry. See [`CorrectionType`] for the five strategies
    /// and the field each one pairs with.
    pub correction_type: CorrectionType,

    /// Human-readable explanation of *what* went wrong and *how* this
    /// correction addresses it.
    ///
    /// Always populated — even an [`CorrectionType::Escalate`] correction
    /// carries a description so the framework can surface it to the user
    /// or a higher-level handler. The string is free-form; some
    /// reflectors include a short root-cause summary here in addition to
    /// [`FailureAnalysis::root_cause`].
    ///
    /// [`FailureAnalysis::root_cause`]: crate::reflection::FailureAnalysis::root_cause
    pub description: String,

    /// Corrected JSON input to pass on retry, when the fix is to change
    /// the arguments rather than the tool.
    ///
    /// Set only for [`CorrectionType::InputFix`] corrections, where the
    /// retry should substitute this value for the original
    /// [`MessagePart::ToolCall::input`]. The shape must conform to the
    /// tool's `input_schema`; an `LlmReflector` with the
    /// `schema_validation` feature enabled will reject a non-conforming
    /// suggestion before it reaches the retry.
    ///
    /// [`MessagePart::ToolCall::input`]: crate::message::MessagePart::ToolCall
    pub modified_input: Option<serde_json::Value>,

    /// Name of a different tool to call instead, when the fix is to swap
    /// tools rather than rewrite arguments.
    ///
    /// Set only for [`CorrectionType::ToolChange`] corrections. Must
    /// match a tool name the registry knows; the retry loop will fail
    /// normally if it does not. Leave `None` for all other correction
    /// types.
    pub alternative_tool: Option<String>,

    /// Free-form instructions to help avoid the same failure on a future
    /// turn.
    ///
    /// Used by [`CorrectionType::ApproachChange`] (high-level
    /// re-strategizing) and optionally by other variants as a sidecar
    /// note. Not consumed mechanically by the retry loop — it is
    /// advisory, typically surfaced to a user or appended to context for
    /// the next model turn.
    pub guidance: Option<String>,
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

/// Result of analyzing a failure via [`Reflector::analyze()`].
///
/// Describes what went wrong, how severe it is, whether it's worth
/// retrying, and optionally provides a [`Correction`] the agent can apply
/// before retrying. Produced by a [`Reflector`] and consumed by a
/// [`RecoveryStrategy`] to decide the next action.
///
/// # How the engine consumes it
///
/// `recover_tool_error` in `engine/bare/dispatch.rs` calls
/// [`Reflector::analyze`] to get a `FailureAnalysis`, hands it to
/// [`RecoveryStrategy::decide`] for the action, and clones `correction`
/// out separately so the retry loop can apply it. If the reflector
/// itself errors, the engine conservatively fails the turn rather than
/// guessing — see [`ReflectionError`].
///
/// # Structured output
///
/// Implements [`StructuredOutput`] so an [`LlmReflector`] can request it
/// back from a model via [`request_structured`] with a guaranteed-schema
/// response. The hand-written schema enumerates the five fields below,
/// pins [`FailureSeverity`] as a four-value string enum, and embeds the
/// [`Correction`] shape under `correction`.
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::{FailureAnalysis, FailureSeverity};
///
/// let analysis = FailureAnalysis {
///     is_recoverable: true,
///     root_cause: "file not found".to_string(),
///     severity: FailureSeverity::Medium,
///     correction: None,
///     context: "Attempted to read config.yaml".to_string(),
/// };
/// assert!(analysis.is_recoverable);
/// ```
///
/// [`StructuredOutput`]: crate::structured::StructuredOutput
/// [`LlmReflector`]: crate::reflection::LlmReflector
/// [`request_structured`]: crate::structured::request_structured
/// [`RecoveryStrategy::decide`]: crate::reflection::RecoveryStrategy::decide
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailureAnalysis {
    /// Whether the failure can be recovered from.
    ///
    /// The headline signal. A [`RecoveryStrategy`] typically maps `false`
    /// to [`RecoveryAction::Fail`] (or [`RecoveryAction::Skip`] when the
    /// step is optional) and `true` to [`RecoveryAction::Retry`] when the
    /// attempt budget allows. This is independent of [`severity`](Self::severity):
    /// a `Low`-severity failure may still be non-recoverable if the
    /// reflector can't suggest a fix, and a `Critical`-severity failure
    /// may technically be recoverable if a correction is supplied.
    ///
    /// [`RecoveryAction::Fail`]: crate::reflection::RecoveryAction::Fail
    /// [`RecoveryAction::Skip`]: crate::reflection::RecoveryAction::Skip
    /// [`RecoveryAction::Retry`]: crate::reflection::RecoveryAction::Retry
    pub is_recoverable: bool,

    /// Description of what went wrong.
    ///
    /// Free-form text identifying the root cause — e.g., `"file not
    /// found"`, `"401 Unauthorized"`, `"tool input did not match schema"`.
    /// Often echoes the tool's error message but may be rephrased or
    /// sharpened by the reflector. Surfaced to users in failure output
    /// and included by an [`LlmReflector`] in its analysis of subsequent
    /// failures.
    ///
    /// [`LlmReflector`]: crate::reflection::LlmReflector
    pub root_cause: String,

    /// How severe the failure is.
    ///
    /// See [`FailureSeverity`] for the four levels and their typical
    /// recovery implications. Strategies commonly use this to gate
    /// retries — e.g., refusing to retry `Critical` even when
    /// [`is_recoverable`](Self::is_recoverable) is `true`.
    pub severity: FailureSeverity,

    /// Suggested correction for the agent to apply before retrying.
    ///
    /// `None` when the reflector has no concrete fix to suggest (the
    /// failure is either non-recoverable, or recoverable by a bare retry
    /// with no input changes). When `Some`, the engine clones the
    /// [`Correction`] out and feeds it into the retry loop, which
    /// substitutes `modified_input` / `alternative_tool` as the
    /// correction directs. Reflectors that populate this field should
    /// keep [`Correction::correction_type`] consistent with the fields
    /// they fill.
    pub correction: Option<Correction>,

    /// Additional context the reflector wants the framework or a future
    /// turn to see.
    ///
    /// Free-form; common uses are environment state at the time of
    /// failure (cwd, available tools), the original task description, or
    /// a short note about what the reflector considered. The framework
    /// does not parse this field — it is advisory, typically logged
    /// alongside the analysis or surfaced to a user when the turn fails.
    pub context: String,
}

impl crate::structured::StructuredOutput for FailureAnalysis {
    fn name() -> &'static str {
        "failure_analysis"
    }

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "is_recoverable": {"type": "boolean"},
                "root_cause": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "correction": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "correction_type": {
                                    "type": "string",
                                    "enum": [
                                        "input_fix",
                                        "tool_change",
                                        "prerequisite_fix",
                                        "approach_change",
                                        "escalate"
                                    ]
                                },
                                "description": {"type": "string"},
                                "modified_input": {"type": ["object", "null"]},
                                "alternative_tool": {"type": ["string", "null"]},
                                "guidance": {"type": ["string", "null"]}
                            },
                            "required": [
                                "correction_type",
                                "description",
                                "modified_input",
                                "alternative_tool",
                                "guidance"
                            ],
                            "additionalProperties": false
                        },
                        {"type": "null"}
                    ]
                },
                "context": {"type": "string"}
            },
            "required": [
                "is_recoverable",
                "root_cause",
                "severity",
                "correction",
                "context"
            ],
            "additionalProperties": false
        })
    }
}

/// Errors produced by [`Reflector::analyze()`].
///
/// The reflector can either produce a valid analysis, skip analysis
/// (letting the framework use its default behaviour), or fail
/// during analysis.
#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
    /// The reflector opted out of analyzing this failure.
    ///
    /// The framework should fall back to its default error handling.
    #[error("reflection skipped: {0}")]
    Skipped(String),

    /// The reflector itself encountered an error.
    ///
    /// Distinct from the tool failure being analyzed — it means
    /// the reflector's own logic broke (e.g., an LLM call for
    /// summarisation failed).
    #[error("reflection internal error: {0}")]
    Internal(String),
}

/// What the framework should do after a failure.
///
/// Produced by [`RecoveryStrategy::decide()`] after the
/// [`Reflector::analyze()`] step. Each variant maps to a different action
/// in the agent loop — the strategy decides which one; the engine
/// executes it.
///
/// # Variants in rough order of severity
///
/// [`Retry`] is the most permissive (try again, optionally with a
/// correction); [`Skip`] continues past the failed step; [`AskUser`]
/// yields control for human input; [`Fail`] terminates the operation and
/// propagates the error. A typical strategy progresses through these as
/// the attempt budget drains: early attempts → `Retry`, late attempts →
/// `Fail` or `AskUser`.
///
/// # Equality and ordering
///
/// Derives [`Eq`] so two actions compare equal when their payloads
/// match (same `delay`, same string). Useful in tests that assert a
/// strategy picked a specific action; not meaningful for runtime
/// prioritization — there is no `Ord` impl, by design.
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::RecoveryAction;
/// use std::time::Duration;
///
/// let action = RecoveryAction::Retry {
///     delay: Duration::from_millis(500),
/// };
/// assert!(action.is_retry());
/// assert_eq!(action.delay(), Some(Duration::from_millis(500)));
///
/// let fail = RecoveryAction::Fail("unrecoverable".to_string());
/// assert!(fail.is_fail());
/// ```
///
/// [`Retry`]: Self::Retry
/// [`Skip`]: Self::Skip
/// [`AskUser`]: Self::AskUser
/// [`Fail`]: Self::Fail
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the failed operation.
    ///
    /// Wait for `delay` before retrying. If the
    /// [`FailureAnalysis::correction`] that produced this action was
    /// `Some`, the engine applies it (substituting
    /// [`Correction::modified_input`] or swapping to
    /// [`Correction::alternative_tool`]) before re-dispatching the tool
    /// call. The strategy is responsible for choosing a sensible `delay`
    /// — typically backoff that grows with the attempt number.
    ///
    /// [`FailureAnalysis::correction`]: crate::reflection::FailureAnalysis::correction
    /// [`Correction::modified_input`]: crate::reflection::Correction::modified_input
    /// [`Correction::alternative_tool`]: crate::reflection::Correction::alternative_tool
    Retry {
        /// Duration to wait before retrying, chosen by the strategy.
        ///
        /// Strategies commonly use exponential backoff here. A `delay`
        /// of zero is permitted (immediate retry) but should be reserved
        /// for cases where the failure is known to be transient and the
        /// retry is cheap.
        delay: Duration,
    },

    /// Skip the failed operation and continue with the rest of the turn.
    ///
    /// The framework logs the reason and moves on rather than
    /// retrying. Use when the step is optional or the failure is
    /// non-fatal — e.g., a metric-emitting tool that the agent can
    /// safely proceed without. The carried string is the human-readable
    /// reason to log.
    Skip(String),

    /// Ask the user for input before continuing.
    ///
    /// Yields control to the caller with a prompt string. The framework
    /// surfaces this in whatever interaction model it runs under
    /// (headless: prints the prompt and waits on stdin; TUI: renders a
    /// prompt and waits for input). Use when the strategy cannot decide
    /// autonomously — e.g., a permission-style failure that a human
    /// should adjudicate, or a `ToolChange` correction that requires a
    /// choice between plausible alternatives.
    AskUser(String),

    /// Fail the operation and propagate the error.
    ///
    /// No further retries — the framework should report this failure and
    /// stop the recovery loop. The carried string is the error message
    /// to surface. Use when the failure is unrecoverable, when the
    /// attempt budget is exhausted, or when a [`Reflector`] returned
    /// [`ReflectionError::Internal`] (the engine conservatively maps
    /// reflector failure to `Fail`).
    ///
    /// [`ReflectionError::Internal`]: crate::reflection::ReflectionError::Internal
    Fail(String),
}

impl RecoveryAction {
    /// Returns the retry delay, if this is a [`Retry`](Self::Retry) action.
    ///
    /// Lets callers branch on the wait without a full `match`. Returns
    /// `None` for the other three variants, so a strategy can write
    /// `action.delay().unwrap_or(Duration::ZERO)` to default a non-retry
    /// action to immediate handling.
    #[must_use]
    pub fn delay(&self) -> Option<Duration> {
        match self {
            Self::Retry { delay } => Some(*delay),
            _ => None,
        }
    }

    /// Returns `true` if this is a [`Retry`](Self::Retry) action.
    ///
    /// Convenience predicate; equivalent to
    /// `matches!(action, RecoveryAction::Retry { .. })`. Useful in
    /// engine code that gates on retry vs. non-retry without caring
    /// about the delay.
    #[must_use]
    pub fn is_retry(&self) -> bool {
        matches!(self, Self::Retry { .. })
    }

    /// Returns `true` if this is a [`Fail`](Self::Fail) action.
    ///
    /// Convenience predicate. Engine code commonly checks this to decide
    /// whether to terminate the recovery loop and propagate the error.
    #[must_use]
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }

    /// Returns `true` if this is a [`Skip`](Self::Skip) action.
    ///
    /// Convenience predicate. Use to distinguish "move on silently" from
    /// the harder-failure variants when logging.
    #[must_use]
    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip(_))
    }

    /// Returns `true` if this is an [`AskUser`](Self::AskUser) action.
    ///
    /// Convenience predicate. Engine code checks this to know it must
    /// yield control to the caller (headless: stdin; TUI: prompt) rather
    /// than continue autonomously.
    #[must_use]
    pub fn is_ask_user(&self) -> bool {
        matches!(self, Self::AskUser(_))
    }
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry { delay } => {
                write!(f, "retry after {delay:?}")
            }
            Self::Skip(reason) => write!(f, "skip: {reason}"),
            Self::AskUser(prompt) => write!(f, "ask user: {prompt}"),
            Self::Fail(reason) => write!(f, "fail: {reason}"),
        }
    }
}

/// Analyses failed tool calls to determine recoverability and corrections.
///
/// Implement this trait to provide custom failure analysis — for example,
/// an LLM-based reflector that reads error messages and suggests fixes,
/// or a rule-based reflector that matches known error patterns.
///
/// The trait is [`Send + Sync`] so it can be stored in `Arc<dyn Reflector>`
/// and shared across threads.
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::{
///     Reflector, ReflectionContext, FailureAnalysis, FailureSeverity, ReflectionError,
/// };
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct PatternReflector;
///
/// impl Reflector for PatternReflector {
///     fn analyze(
///         &self,
///         error: &str,
///         tool_name: &str,
///         _tool_input: &serde_json::Value,
///         _tool_schema: Option<&loopctl::tool::ToolSchema>,
///         _context: &ReflectionContext,
///     ) -> Pin<Box<dyn Future<Output = Result<FailureAnalysis, ReflectionError>> + Send + '_>> {
///         let error = error.to_string();
///         let tool_name = tool_name.to_string();
///         Box::pin(async move {
///             if error.contains("not found") {
///                 Ok(FailureAnalysis {
///                     is_recoverable: true,
///                     root_cause: error,
///                     severity: FailureSeverity::Medium,
///                     correction: None,
///                     context: format!("tool: {tool_name}"),
///                 })
///             } else {
///                 Ok(FailureAnalysis {
///                     is_recoverable: false,
///                     root_cause: error,
///                     severity: FailureSeverity::High,
///                     correction: None,
///                     context: String::new(),
///                 })
///             }
///         })
///     }
/// }
/// ```
pub trait Reflector: Send + Sync {
    /// Analyse a failed tool call.
    ///
    /// # Arguments
    ///
    /// - `error` — The error message from the failed call.
    /// - `tool_name` — Which tool was called.
    /// - `tool_input` — The JSON input that was passed.
    /// - `tool_schema` — The schema of the tool that failed, when the
    ///   engine can resolve it. `None` if the tool isn't in the registry
    ///   or the schema is otherwise unavailable. Reflectors that want to
    ///   validate a suggested `modified_input` should skip validation
    ///   when this is `None`.
    /// - `context` — Retry state and task description.
    ///
    /// # Errors
    ///
    /// Returns [`ReflectionError::Skipped`] if the reflector opts out,
    /// or [`ReflectionError::Internal`] if the reflector itself fails.
    fn analyze(
        &self,
        error: &str,
        tool_name: &str,
        tool_input: &serde_json::Value,
        tool_schema: Option<&crate::tool::ToolSchema>,
        context: &ReflectionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FailureAnalysis, ReflectionError>> + Send + '_>>;
}

/// Decides what to do after a failure has been analyzed.
///
/// Takes a [`FailureAnalysis`] and the current retry state, returns a
/// [`RecoveryAction`]. Implement this trait to provide custom recovery
/// policies — for example, circuit-breaker patterns, rate-limit-aware
/// backoff, or user-prompting strategies.
///
/// The trait is [`Send + Sync`] so it can be stored in
/// `Arc<dyn RecoveryStrategy>` and shared across threads.
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::{
///     RecoveryStrategy, FailureAnalysis, FailureSeverity, RecoveryAction,
/// };
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct AlwaysRetryStrategy;
///
/// impl RecoveryStrategy for AlwaysRetryStrategy {
///     fn decide(
///         &self,
///         _analysis: &FailureAnalysis,
///         attempt: u32,
///         max_attempts: u32,
///     ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
///         let action = if attempt >= max_attempts {
///             RecoveryAction::Fail("max retries exceeded".to_string())
///         } else {
///             RecoveryAction::Retry {
///                 delay: std::time::Duration::from_secs(1),
///             }
///         };
///         Box::pin(async move { action })
///     }
/// }
/// ```
pub trait RecoveryStrategy: Send + Sync {
    /// Decide what to do after a failure.
    ///
    /// # Arguments
    ///
    /// - `analysis` — The reflector's analysis of the failure.
    /// - `attempt` — Current attempt number (0-indexed).
    /// - `max_attempts` — Maximum attempts before giving up.
    ///
    /// # Returns
    ///
    /// A [`RecoveryAction`] telling the framework what to do next.
    fn decide(
        &self,
        analysis: &FailureAnalysis,
        attempt: u32,
        max_attempts: u32,
    ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>>;
}

/// A no-op reflector that marks everything as non-recoverable.
///
/// Useful as a safe default when reflection is disabled, or as a base
/// for building composite reflectors via the decorator pattern.
///
/// # Example
///
/// ```rust,ignore
/// # tokio_test::block_on(async {
/// let reflector = NoopReflector;
/// let context = ReflectionContext {
///     task: "test".to_string(),
///     attempt: 0,
///     max_attempts: 3,
/// };
///
/// let analysis = reflector.analyze("error", "tool", &json!({}), &context).await.unwrap();
/// assert!(!analysis.is_recoverable);
/// # });
/// ```
pub struct NoopReflector;

impl Reflector for NoopReflector {
    fn analyze(
        &self,
        error: &str,
        _tool_name: &str,
        _tool_input: &serde_json::Value,
        _tool_schema: Option<&crate::tool::ToolSchema>,
        _context: &ReflectionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FailureAnalysis, ReflectionError>> + Send + '_>> {
        let root_cause = error.to_string();
        Box::pin(async move {
            Ok(FailureAnalysis {
                is_recoverable: false,
                root_cause,
                severity: FailureSeverity::Medium,
                correction: None,
                context: String::new(),
            })
        })
    }
}

impl fmt::Debug for NoopReflector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NoopReflector").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_ordering() {
        assert!(FailureSeverity::Low < FailureSeverity::Medium);
        assert!(FailureSeverity::Medium < FailureSeverity::High);
        assert!(FailureSeverity::High < FailureSeverity::Critical);
    }

    #[test]
    fn severity_display() {
        assert_eq!(FailureSeverity::Low.to_string(), "low");
        assert_eq!(FailureSeverity::Medium.to_string(), "medium");
        assert_eq!(FailureSeverity::High.to_string(), "high");
        assert_eq!(FailureSeverity::Critical.to_string(), "critical");
    }

    #[test]
    fn severity_serde_round_trip() {
        let severity = FailureSeverity::High;
        let json = serde_json::to_string(&severity).unwrap();
        assert_eq!(json, "\"high\"");
        let deserialized: FailureSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, severity);
    }

    #[test]
    fn reflection_context_fields() {
        let ctx = ReflectionContext {
            task: "fix bug".to_string(),
            attempt: 2,
            max_attempts: 5,
        };
        assert_eq!(ctx.task, "fix bug");
        assert_eq!(ctx.attempt, 2);
        assert_eq!(ctx.max_attempts, 5);
    }

    #[test]
    fn failure_analysis_recoverable() {
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Low,
            correction: None,
            context: "network".to_string(),
        };
        assert!(analysis.is_recoverable);
        assert_eq!(analysis.root_cause, "timeout");
    }

    #[test]
    fn failure_analysis_with_correction() {
        let correction = Correction {
            correction_type: CorrectionType::InputFix,
            description: "fix path".to_string(),
            modified_input: Some(serde_json::json!({"path": "/correct/path"})),
            alternative_tool: None,
            guidance: Some("check paths".to_string()),
        };
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "file not found".to_string(),
            severity: FailureSeverity::Medium,
            correction: Some(correction),
            context: String::new(),
        };
        assert!(analysis.correction.is_some());
        let c = analysis.correction.unwrap();
        assert_eq!(c.description, "fix path");
    }

    #[test]
    fn reflection_error_skipped_display() {
        let err = ReflectionError::Skipped("not applicable".to_string());
        let s = err.to_string();
        assert!(s.contains("skipped"));
        assert!(s.contains("not applicable"));
    }

    #[test]
    fn reflection_error_internal_display() {
        let err = ReflectionError::Internal("llm timeout".to_string());
        let s = err.to_string();
        assert!(s.contains("internal"));
        assert!(s.contains("llm timeout"));
    }

    #[test]
    fn action_retry_accessors() {
        let action = RecoveryAction::Retry {
            delay: Duration::from_secs(2),
        };
        assert!(action.is_retry());
        assert!(!action.is_fail());
        assert_eq!(action.delay(), Some(Duration::from_secs(2)));
    }

    #[test]
    fn action_skip_accessors() {
        let action = RecoveryAction::Skip("not important".to_string());
        assert!(!action.is_retry());
        assert!(!action.is_fail());
        assert_eq!(action.delay(), None);
    }

    #[test]
    fn action_ask_user() {
        let action = RecoveryAction::AskUser("choose option".to_string());
        assert!(!action.is_retry());
        assert_eq!(action.delay(), None);
    }

    #[test]
    fn action_fail_accessors() {
        let action = RecoveryAction::Fail("unrecoverable".to_string());
        assert!(action.is_fail());
        assert!(!action.is_retry());
        assert_eq!(action.delay(), None);
    }

    #[test]
    fn action_display() {
        let retry = RecoveryAction::Retry {
            delay: Duration::from_secs(1),
        };
        assert!(retry.to_string().contains("retry"));

        let skip = RecoveryAction::Skip("reason".to_string());
        assert!(skip.to_string().contains("skip: reason"));

        let ask = RecoveryAction::AskUser("prompt".to_string());
        assert!(ask.to_string().contains("ask user: prompt"));

        let fail = RecoveryAction::Fail("bad".to_string());
        assert!(fail.to_string().contains("fail: bad"));
    }

    #[tokio::test]
    async fn noop_reflector_marks_non_recoverable() {
        let reflector = NoopReflector;
        let ctx = ReflectionContext {
            task: "test".to_string(),
            attempt: 0,
            max_attempts: 3,
        };
        let analysis = reflector
            .analyze("some error", "tool", &serde_json::json!({}), None, &ctx)
            .await
            .unwrap();
        assert!(!analysis.is_recoverable);
        assert_eq!(analysis.root_cause, "some error");
        assert_eq!(analysis.severity, FailureSeverity::Medium);
        assert!(analysis.correction.is_none());
    }

    #[test]
    fn noop_reflector_debug() {
        let reflector = NoopReflector;
        let debug = format!("{reflector:?}");
        assert!(debug.contains("NoopReflector"));
    }

    #[test]
    fn failure_analysis_structured_round_trip() {
        use crate::structured::StructuredOutput;
        let v = serde_json::json!({
            "is_recoverable": true,
            "root_cause": "timeout",
            "severity": "low",
            "correction": {
                "correction_type": "input_fix",
                "description": "fix the path",
                "modified_input": {"path": "/x"},
                "alternative_tool": null,
                "guidance": null
            },
            "context": "open call"
        });
        let analysis = FailureAnalysis::from_value(v).expect("should deserialize");
        assert!(analysis.is_recoverable);
        assert_eq!(analysis.root_cause, "timeout");
        assert_eq!(analysis.severity, FailureSeverity::Low);
        let correction = analysis.correction.expect("correction");
        assert_eq!(correction.description, "fix the path");
        assert_eq!(
            correction.modified_input,
            Some(serde_json::json!({"path": "/x"}))
        );
    }

    #[test]
    fn failure_analysis_schema_is_valid_json() {
        let schema = <FailureAnalysis as crate::structured::StructuredOutput>::schema();
        let obj = schema.as_object().expect("schema must be a JSON object");
        // Five top-level properties.
        assert_eq!(obj["type"], "object");
        let required = obj["required"]
            .as_array()
            .expect("required must be an array");
        assert_eq!(required.len(), 5);
    }

    #[test]
    fn failure_analysis_schema_correction_type_enum() {
        let schema = <FailureAnalysis as crate::structured::StructuredOutput>::schema();
        let enum_values = schema
            .pointer("/properties/correction/anyOf/0/properties/correction_type/enum")
            .expect("correction_type enum must be present")
            .as_array()
            .expect("enum must be an array");
        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        // Pin the 5 snake_case variants matching CorrectionType's serde rename.
        assert_eq!(
            values,
            vec![
                "input_fix",
                "tool_change",
                "prerequisite_fix",
                "approach_change",
                "escalate"
            ]
        );
    }

    #[test]
    fn failure_analysis_name_is_stable() {
        use crate::structured::StructuredOutput;
        let name = FailureAnalysis::name();
        assert_eq!(name, "failure_analysis");
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "name must match ^[a-zA-Z0-9_-]+$: {name}"
        );
    }

    #[tokio::test]
    async fn noop_reflector_accepts_new_signature() {
        // Pins the breaking trait change: NoopReflector implements the
        // 5-arg analyze and its semantics are unchanged.
        let reflector = NoopReflector;
        let ctx = ReflectionContext {
            task: "t".to_string(),
            attempt: 0,
            max_attempts: 1,
        };
        let analysis = reflector
            .analyze("err", "tool", &serde_json::json!({}), None, &ctx)
            .await
            .unwrap();
        assert!(!analysis.is_recoverable);
        assert_eq!(analysis.severity, FailureSeverity::Medium);
    }
}
