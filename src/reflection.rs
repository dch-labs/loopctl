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

use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// ===================================================
// FailureSeverity
// ===================================================

/// How severe a failure is.
///
/// Ordered from least to most severe. The [`RecoveryStrategy`] may use
/// severity to decide whether retrying is worthwhile — a `Low` severity
/// issue (e.g., a transient network blip) is more retryable than a
/// `Critical` one (e.g., invalid API key).
///
/// # Example
///
/// ```rust
/// use loopctl::reflection::FailureSeverity;
///
/// assert!(FailureSeverity::Low < FailureSeverity::Critical);
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    /// Minor issue — a retry will likely fix it.
    Low,
    /// Moderate issue — may need a correction before retrying.
    Medium,
    /// Serious issue — retrying without changes is unlikely to help.
    High,
    /// Unrecoverable — the agent should stop or escalate.
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

// ===================================================
// ReflectionContext
// ===================================================

/// Context provided to [`Reflector::analyze()`] describing the retry state.
///
/// Built by the framework before invoking the reflector. Contains
/// information about what the agent was doing and how many attempts
/// have been made so far.
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
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReflectionContext {
    /// What the agent was trying to accomplish.
    pub task: String,
    /// Current attempt number (0-indexed).
    pub attempt: u32,
    /// Maximum attempts allowed before giving up.
    pub max_attempts: u32,
}

// ===================================================
// Correction
// ===================================================

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
/// the agent analyzes the error and produces a `Correction` that describes how to fix
/// the problem. The framework applies the correction and retries.
///
/// # Serialization
///
/// Implements `Serialize` and `Deserialize` for persistence and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// See [`CorrectionType`] for available strategies.
    pub correction_type: CorrectionType,
    /// Explains *what* went wrong and *how* the correction addresses it.
    pub description: String,
    /// Corrected JSON input when [`CorrectionType::InputFix`]. `None` otherwise.
    pub modified_input: Option<serde_json::Value>,
    /// Alternative tool name when [`CorrectionType::ToolChange`]. `None` otherwise.
    pub alternative_tool: Option<String>,
    /// Extra context or instructions to help avoid the same failure.
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

// ===================================================
// FailureAnalysis
// ===================================================

/// Result of analysing a failure via [`Reflector::analyze()`].
///
/// Describes what went wrong, how severe it is, whether it's worth
/// retrying, and optionally provides a [`Correction`] the agent can
/// apply before retrying.
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailureAnalysis {
    /// Whether the failure can be recovered from.
    pub is_recoverable: bool,
    /// Description of what went wrong.
    pub root_cause: String,
    /// How severe the failure is.
    pub severity: FailureSeverity,
    /// Suggested correction for the agent to apply before retrying.
    pub correction: Option<Correction>,
    /// Additional context (e.g., environment state at time of failure).
    pub context: String,
}

// ===================================================
// ReflectionError
// ===================================================

/// Errors produced by [`Reflector::analyze()`].
///
/// The reflector can either produce a valid analysis, skip analysis
/// (letting the framework use its default behaviour), or fail
/// during analysis.
#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
    /// The reflector opted out of analysing this failure.
    ///
    /// The framework should fall back to its default error handling.
    #[error("reflection skipped: {0}")]
    Skipped(String),

    /// The reflector itself encountered an error.
    ///
    /// Distinct from the tool failure being analysed — it means
    /// the reflector's own logic broke (e.g., an LLM call for
    /// summarisation failed).
    #[error("reflection internal error: {0}")]
    Internal(String),
}

// ===================================================
// RecoveryAction
// ===================================================

/// What the framework should do after a failure.
///
/// Produced by [`RecoveryStrategy::decide()`]. Each variant maps to a
/// different action in the agent loop.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the failed operation.
    ///
    /// Wait for `delay` before retrying. If `correction` is `Some`,
    /// apply it before the retry.
    Retry {
        /// Duration to wait before retrying.
        delay: Duration,
    },

    /// Skip the failed operation and continue.
    ///
    /// The framework should log the reason and move to the next turn.
    Skip(String),

    /// Ask the user for input.
    ///
    /// The framework should return control to the caller with a prompt.
    AskUser(String),

    /// Fail the operation and propagate the error.
    ///
    /// No further retries — the framework should report this failure.
    Fail(String),
}

impl RecoveryAction {
    /// Returns the retry delay, if this is a [`Retry`](Self::Retry) action.
    #[must_use]
    pub fn delay(&self) -> Option<Duration> {
        match self {
            Self::Retry { delay } => Some(*delay),
            _ => None,
        }
    }

    /// Returns `true` if this is a [`Retry`](Self::Retry) action.
    #[must_use]
    pub fn is_retry(&self) -> bool {
        matches!(self, Self::Retry { .. })
    }

    /// Returns `true` if this is a [`Fail`](Self::Fail) action.
    #[must_use]
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }

    /// Returns `true` if this is a [`Skip`](Self::Skip) action.
    #[must_use]
    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip(_))
    }

    /// Returns `true` if this is an [`AskUser`](Self::AskUser) action.
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

// ===================================================
// Reflector trait
// ===================================================

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
        context: &ReflectionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FailureAnalysis, ReflectionError>> + Send + '_>>;
}

// ===================================================
// RecoveryStrategy trait
// ===================================================

/// Decides what to do after a failure has been analysed.
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

// ===================================================
// NoopReflector
// ===================================================

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
            .analyze("some error", "tool", &serde_json::json!({}), &ctx)
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
}
