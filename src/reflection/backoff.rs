//! Exponential backoff recovery strategy.
//!
//! [`ExponentialBackoffRecovery`] retries recoverable failures with
//! exponentially increasing delays, up to a configurable maximum.
//! This is the default [`RecoveryStrategy`] used by `BareLoop`.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::reflection::backoff::ExponentialBackoffRecovery;
//! use loopctl::reflection::{RecoveryAction, FailureAnalysis, FailureSeverity, RecoveryStrategy};
//!
//! let strategy = ExponentialBackoffRecovery::new(3);
//! ```

use super::{FailureAnalysis, FailureSeverity, RecoveryAction, RecoveryStrategy};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// ===================================================
// ExponentialBackoffRecovery
// ===================================================

/// A recovery strategy that retries with exponential backoff.
///
/// When a tool call fails with a recoverable error, this strategy
/// retries up to [`max_retries`](ExponentialBackoffRecovery::max_retries) times,
/// with delays that grow exponentially: `base_delay * 2^attempt`.
///
/// # Behaviour by severity
///
/// | Severity   | Recoverable | Correction | Action                                                            |
/// |------------|-------------|------------|-------------------------------------------------------------------|
/// | Low/Medium | ✓           | —          | Retry with backoff                                                |
/// | High       | ✓           | Some       | [`AskUser`](super::RecoveryAction::AskUser) (correction provided) |
/// | High       | ✓           | None       | Retry with backoff                                                |
/// | Critical   | ✗           | —          | Fail immediately                                                  |
/// | any        | ✗           | —          | Fail immediately                                                  |
///
/// # Example
///
/// ```
/// use loopctl::reflection::backoff::ExponentialBackoffRecovery;
///
/// let strategy = ExponentialBackoffRecovery::new(5);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialBackoffRecovery {
    /// Maximum number of retry attempts before giving up.
    max_retries: u32,
    /// Initial delay applied to the first retry, doubled on each subsequent attempt.
    base_delay: Duration,
    /// Upper bound on the computed delay, regardless of exponential growth.
    max_delay: Duration,
}

impl ExponentialBackoffRecovery {
    /// Create a new strategy with the given maximum retry count.
    ///
    /// Defaults: `base_delay` = 100ms, `max_delay` = 30s.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
        }
    }

    /// Set a custom base delay for the exponential backoff.
    ///
    /// The delay for attempt `n` is `base_delay * 2^n`, capped at `max_delay`.
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set a maximum delay cap.
    ///
    /// Even as the exponential grows, the delay never exceeds this value.
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Maximum retry attempts before giving up.
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Base delay for exponential backoff calculation.
    #[must_use]
    pub fn base_delay(&self) -> Duration {
        self.base_delay
    }

    /// Maximum delay cap.
    #[must_use]
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Calculate the backoff delay for a given attempt.
    ///
    /// Computes `base_delay * 2^attempt`, clamped to [`max_delay`](Self::max_delay).
    ///
    /// Attempt 0 yields `base_delay`, attempt 1 yields `2 * base_delay`, and
    /// so on. Once the exponential exceeds `max_delay`, every subsequent
    /// attempt returns `max_delay`.
    ///
    /// Uses saturating arithmetic to avoid overflow on very large attempt
    /// numbers — once `2^attempt` would overflow, the delay is clamped to
    /// `max_delay`.
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
        use super::super::{Correction, CorrectionType};
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

    #[test]
    fn delay_for_attempt_zero_yields_base() {
        let strategy = ExponentialBackoffRecovery::new(3);
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_millis(100));
    }

    #[test]
    fn delay_for_attempt_one_doubles() {
        let strategy = ExponentialBackoffRecovery::new(3);
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_millis(200));
    }

    #[test]
    fn delay_for_attempt_two_quadruples() {
        let strategy = ExponentialBackoffRecovery::new(3);
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn delay_for_attempt_capped_at_max() {
        let strategy = ExponentialBackoffRecovery::new(10)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(5));
        // 1 * 2^5 = 32s, capped at 5s
        assert_eq!(strategy.delay_for_attempt(5), Duration::from_secs(5));
    }

    #[test]
    fn delay_for_attempt_exactly_at_max() {
        let strategy = ExponentialBackoffRecovery::new(10)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(8));
        // 1 * 2^3 = 8s, exactly at cap
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(8));
    }

    #[test]
    fn delay_for_attempt_stays_capped_for_large_attempts() {
        let strategy = ExponentialBackoffRecovery::new(100)
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(30));
        let late = strategy.delay_for_attempt(50);
        assert_eq!(late, Duration::from_secs(30));
    }

    #[test]
    fn delay_for_attempt_custom_base() {
        let strategy =
            ExponentialBackoffRecovery::new(5).with_base_delay(Duration::from_millis(250));
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_millis(250));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_millis(500));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_secs(1));
    }

    #[test]
    fn delay_for_attempt_overflow_does_not_panic() {
        let strategy = ExponentialBackoffRecovery::new(3)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(10));
        // attempt = u32::MAX would overflow 2^n; should saturate, not panic
        let delay = strategy.delay_for_attempt(u32::MAX);
        assert_eq!(delay, Duration::from_secs(10));
    }
}
