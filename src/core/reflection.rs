//! Reflection and recovery for failed agent turns.
//!
//! When a tool call fails or a turn produces an unexpected result, the
//! framework needs to decide what to do. This module provides a pluggable
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
//! # Architecture
//!
//! ```text
//!       Tool call fails
//!             │
//!             ▼
//! ┌───────────────────────┐
//! │  Reflector::analyze() │
//! │  → FailureAnalysis    │
//! │    (recoverable?)     │
//! │    (correction?)      │
//! │    (severity)         │
//! └──────────┬────────────┘
//!            │
//!            ▼
//! ┌───────────────────────────────┐
//! │ RecoveryStrategy::decide()    │
//! │ → RecoveryAction              │
//! │   Retry / Skip / AskUser /    │
//! │   Fail                        │
//! └───────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::core::reflection::{
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

use crate::core::types::Correction;
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
/// use loopctl::core::reflection::FailureSeverity;
///
/// assert!(FailureSeverity::Low < FailureSeverity::Critical);
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    /// Minor issue — a simple retry will likely fix it.
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
/// use loopctl::core::reflection::ReflectionContext;
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
/// use loopctl::core::reflection::{FailureAnalysis, FailureSeverity};
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
    /// Whether the framework should attempt recovery.
    pub is_recoverable: bool,
    /// Human-readable description of the root cause.
    pub root_cause: String,
    /// How severe the failure is.
    pub severity: FailureSeverity,
    /// Optional correction the agent can apply before retrying.
    pub correction: Option<Correction>,
    /// Additional context about the failure (e.g., environment state).
    pub context: String,
}

// ===================================================
// ReflectionError
// ===================================================

/// Errors produced by [`Reflector::analyze()`].
///
/// The reflector can either produce a valid analysis, skip analysis
/// (letting the framework use its default behaviour), or fail
/// internally.
#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
    /// The reflector opted out of analysing this failure.
    ///
    /// The framework should fall back to its default error handling.
    #[error("reflection skipped: {0}")]
    Skipped(String),

    /// The reflector itself encountered an error.
    ///
    /// This is distinct from the tool failure being analysed — it means
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
/// use loopctl::core::reflection::RecoveryAction;
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
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::core::reflection::RecoveryAction;
    /// use std::time::Duration;
    ///
    /// let action = RecoveryAction::Retry { delay: Duration::from_secs(2) };
    /// assert_eq!(action.delay(), Some(Duration::from_secs(2)));
    ///
    /// assert_eq!(RecoveryAction::Fail("bad".into()).delay(), None);
    /// ```
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
/// use loopctl::core::reflection::{
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
/// use loopctl::core::reflection::{
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

// ===================================================
// ExponentialBackoffRecovery
// ===================================================

/// Recovery strategy using exponential backoff.
///
/// Retries up to `max_retries` times with exponential delays. If the
/// [`FailureAnalysis`] says the failure is not recoverable, returns
/// [`RecoveryAction::Fail`] immediately.
///
/// # Backoff Formula
///
/// `delay = min(base_delay × 2^attempt, max_delay)`
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::core::reflection::{ExponentialBackoffRecovery, RecoveryAction, FailureAnalysis, FailureSeverity, RecoveryStrategy};
/// use std::time::Duration;
///
/// let strategy = ExponentialBackoffRecovery::new(3)
///     .with_base_delay(Duration::from_millis(100))
///     .with_max_delay(Duration::from_secs(10));
///
/// let recoverable = FailureAnalysis {
///     is_recoverable: true,
///     root_cause: "timeout".to_string(),
///     severity: FailureSeverity::Low,
///     correction: None,
///     context: String::new(),
/// };
/// let action = strategy.decide(&recoverable, 0, 5).await;
/// assert!(action.is_retry());
/// assert_eq!(action.delay(), Some(Duration::from_millis(100)));
///
/// let unrecoverable = FailureAnalysis {
///     is_recoverable: false,
///     root_cause: "invalid key".to_string(),
///     severity: FailureSeverity::Critical,
///     correction: None,
///     context: String::new(),
/// };
/// let action = strategy.decide(&unrecoverable, 0, 5).await;
/// assert!(action.is_fail());
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialBackoffRecovery {
    /// Maximum number of retry attempts.
    max_retries: u32,
    /// Base delay before the first retry.
    base_delay: Duration,
    /// Maximum delay between retries.
    max_delay: Duration,
}

impl ExponentialBackoffRecovery {
    /// Create a new strategy with the given maximum retries.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::core::reflection::ExponentialBackoffRecovery;
    ///
    /// let strategy = ExponentialBackoffRecovery::new(5);
    /// ```
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
        }
    }

    /// Set the base delay (delay before the first retry).
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set the maximum delay between retries.
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Returns the configured max retries.
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Returns the configured base delay.
    #[must_use]
    pub fn base_delay(&self) -> Duration {
        self.base_delay
    }

    /// Returns the configured max delay.
    #[must_use]
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Calculate the backoff delay for a given attempt.
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms = self
            .base_delay
            .as_millis()
            .saturating_mul(1u128.checked_shl(attempt).unwrap_or(u128::MAX));
        let delay_ms = u64::try_from(delay_ms.min(self.max_delay.as_millis())).unwrap_or(u64::MAX);
        Duration::from_millis(delay_ms)
    }
}

impl RecoveryStrategy for ExponentialBackoffRecovery {
    fn decide(
        &self,
        analysis: &FailureAnalysis,
        attempt: u32,
        _max_attempts: u32,
    ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
        let action = if !analysis.is_recoverable {
            RecoveryAction::Fail(analysis.root_cause.clone())
        } else if attempt >= self.max_retries {
            RecoveryAction::Fail(format!("max retries ({}) exceeded", self.max_retries))
        } else if analysis.severity >= FailureSeverity::High && analysis.correction.is_some() {
            RecoveryAction::AskUser(format!(
                "high-severity failure with correction available: {}",
                analysis.root_cause
            ))
        } else {
            RecoveryAction::Retry {
                delay: self.delay_for_attempt(attempt),
            }
        };
        Box::pin(async move { action })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::CorrectionType;

    // ===================================================
    // FailureSeverity tests
    // ===================================================

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

    // ===================================================
    // ReflectionContext tests
    // ===================================================

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

    // ===================================================
    // FailureAnalysis tests
    // ===================================================

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

    // ===================================================
    // ReflectionError tests
    // ===================================================

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

    // ===================================================
    // RecoveryAction tests
    // ===================================================

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

    // ===================================================
    // NoopReflector tests
    // ===================================================

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

    // ===================================================
    // ExponentialBackoffRecovery tests
    // ===================================================

    #[test]
    fn backoff_builder_defaults() {
        let strategy = ExponentialBackoffRecovery::new(3);
        assert_eq!(strategy.max_retries(), 3);
        assert_eq!(strategy.base_delay(), Duration::from_millis(100));
        assert_eq!(strategy.max_delay(), Duration::from_secs(30));
    }

    #[test]
    fn backoff_builder_custom() {
        let strategy = ExponentialBackoffRecovery::new(5)
            .with_base_delay(Duration::from_millis(200))
            .with_max_delay(Duration::from_secs(60));
        assert_eq!(strategy.max_retries(), 5);
        assert_eq!(strategy.base_delay(), Duration::from_millis(200));
        assert_eq!(strategy.max_delay(), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn backoff_recoverable_first_attempt() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Medium,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 0, 5).await;
        assert!(action.is_retry());
        assert_eq!(action.delay(), Some(Duration::from_millis(100)));
    }

    #[tokio::test]
    async fn backoff_recoverable_second_attempt() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Medium,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 1, 5).await;
        assert!(action.is_retry());
        assert_eq!(action.delay(), Some(Duration::from_millis(200)));
    }

    #[tokio::test]
    async fn backoff_recoverable_third_attempt() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Medium,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 2, 5).await;
        assert!(action.is_retry());
        assert_eq!(action.delay(), Some(Duration::from_millis(400)));
    }

    #[tokio::test]
    async fn backoff_max_retries_exceeded() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Low,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 3, 5).await;
        assert!(action.is_fail());
        let RecoveryAction::Fail(reason) = action else {
            unreachable!()
        };
        assert!(reason.contains("max retries"));
    }

    #[tokio::test]
    async fn backoff_unrecoverable_fails_immediately() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: false,
            root_cause: "invalid api key".to_string(),
            severity: FailureSeverity::Critical,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 0, 5).await;
        assert!(action.is_fail());
        let RecoveryAction::Fail(reason) = action else {
            unreachable!()
        };
        assert_eq!(reason, "invalid api key");
    }

    #[tokio::test]
    async fn backoff_delay_capped_at_max() {
        let strategy = ExponentialBackoffRecovery::new(10)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(5));
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::Medium,
            correction: None,
            context: String::new(),
        };
        // 1 * 2^5 = 32s, capped at 5s
        let action = strategy.decide(&analysis, 5, 10).await;
        assert_eq!(action.delay(), Some(Duration::from_secs(5)));
    }

    #[tokio::test]
    async fn backoff_low_severity_retries() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "transient hiccup".to_string(),
            severity: FailureSeverity::Low,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 0, 5).await;
        assert!(action.is_retry());
    }

    #[tokio::test]
    async fn backoff_high_severity_with_correction_asks_user() {
        use crate::core::types::CorrectionType;
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "bad parameter".to_string(),
            severity: FailureSeverity::High,
            correction: Some(Correction {
                correction_type: CorrectionType::InputFix,
                description: "fix the file path".to_string(),
                modified_input: None,
                alternative_tool: None,
                guidance: None,
            }),
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 0, 5).await;
        assert!(action.is_ask_user());
    }

    #[tokio::test]
    async fn backoff_high_severity_without_correction_retries() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "timeout".to_string(),
            severity: FailureSeverity::High,
            correction: None,
            context: String::new(),
        };
        let action = strategy.decide(&analysis, 0, 5).await;
        assert!(action.is_retry());
    }

    #[test]
    fn backoff_debug_format() {
        let strategy = ExponentialBackoffRecovery::new(3);
        let debug = format!("{strategy:?}");
        assert!(debug.contains("ExponentialBackoffRecovery"));
        assert!(debug.contains("max_retries"));
    }
}
