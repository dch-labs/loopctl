//! Configuration, result types, and error types for resilient LLM stream handling.
//!
//! This module defines the types that underpin [`StreamHandler`] — the framework's
//! production-grade streaming resilience layer. The handler will wrap
//! [`ApiClient::stream_messages`](crate::api_client::ApiClient::stream_messages)
//! with retry, timeout, and fallback behaviour.
//!
//! **Current state:** config structs, result types, error types, and constructors
//! are implemented here. The runtime lifecycle methods (`stream_turn`,
//! `init_with_retry`, `process_events`, `fallback_non_streaming`) will be wired
//! into `BareLoop` in a future phase, using these types.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │        StreamHandler (config + types)           │
//! |                                                 │
//! |   Phase 1: init_with_retry()          (planned) │
//! │    └─ stream_messages() → first event timeout   │
//! │    └─ retry with backoff on failure             │
//! │                                                 │
//! |    Phase 2: process_events()          (planned) │
//! │    └─ per-event timeout + total timeout         │
//! │    └─ progress callbacks at intervals           │
//! │                                                 │
//! |   Phase 3: fallback_non_streaming()   (planned) │
//! │    └─ create_message() if streaming exhausted   │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Configuration
//!
//! | Config                  | Purpose                          | Default                              |
//! |-------------------------|----------------------------------|--------------------------------------|
//! | [`StreamTimeoutConfig`] | Timeout durations and thresholds | See [`StreamTimeoutConfig::default`] |
//! | [`StreamRetryConfig`]   | Retry count, backoff, jitter     | See [`StreamRetryConfig::default`]   |
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
//!
//! let handler = StreamHandler::new();
//!
//! // Or with custom config:
//! let handler = StreamHandler::with_config(
//!     StreamTimeoutConfig {
//!         initial_event_timeout: std::time::Duration::from_secs(60),
//!         ..Default::default()
//!     },
//!     Default::default(),
//! );
//! ```

use std::fmt;
use std::time::Duration;

// ===================================================
// StreamTimeoutConfig
// ===================================================

/// Configuration for the [`StreamHandler`]'s timeout behaviour.
///
/// Controls how long the handler waits for events at each phase of the
/// streaming lifecycle. The defaults are production-ready for typical
/// LLM API interactions.
///
/// # Timeouts
///
/// | Timeout                 | Phase   | Default | Purpose                       |
/// |-------------------------|---------|---------|-------------------------------|
/// | `initial_event_timeout` | Init    | 120s    | First event after stream open |
/// | `per_event_timeout`     | Process | 300s    | Between consecutive events    |
/// | `total_stream_timeout`  | Process | 900s    | Maximum total stream duration |
///
/// # Example
///
/// ```rust
/// use loopctl::stream::handler::StreamTimeoutConfig;
/// use std::time::Duration;
///
/// let config = StreamTimeoutConfig {
///     initial_event_timeout: Duration::from_secs(60),
///     per_event_timeout: Duration::from_secs(180),
///     ..Default::default()
/// };
/// assert_eq!(config.total_stream_timeout, Duration::from_secs(900));
/// ```
#[derive(Debug, Clone)]
pub struct StreamTimeoutConfig {
    /// Timeout for the first event after opening the stream.
    ///
    /// This is the most critical timeout — if the API server never sends
    /// the first event, the stream hangs forever. Set to a generous value
    /// since the model may need time to begin generating.
    pub initial_event_timeout: Duration,

    /// Timeout between consecutive events during normal processing.
    ///
    /// If no event arrives within this window, the handler increments a
    /// consecutive-timeout counter. If [`max_consecutive_timeouts`](Self::max_consecutive_timeouts)
    /// is reached, the handler triggers recovery.
    pub per_event_timeout: Duration,

    /// Maximum total duration for a single stream, regardless of activity.
    ///
    /// Even if events are flowing, the stream is terminated after this
    /// duration. Prevents runaway streams from very long model responses.
    pub total_stream_timeout: Duration,

    /// Maximum consecutive per-event timeouts before triggering recovery.
    ///
    /// When zero events have been received (empty stream), a lower
    /// threshold is used: `min(2, max_consecutive_timeouts)`.
    pub max_consecutive_timeouts: u32,

    /// Interval for progress callbacks during long streams.
    ///
    /// The handler calls the progress callback at this interval to report
    /// elapsed time and event count.
    pub progress_interval: Duration,

    /// Whether to fall back to [`ApiClient::create_message`](crate::api_client::ApiClient::create_message)
    /// when streaming exhausts all retries.
    ///
    /// When `true`, the handler will attempt a non-streaming request as a
    /// last resort. When `false`, the handler returns an error instead.
    pub fallback_to_non_streaming: bool,
}

impl Default for StreamTimeoutConfig {
    fn default() -> Self {
        Self {
            initial_event_timeout: Duration::from_secs(120),
            per_event_timeout: Duration::from_secs(300),
            total_stream_timeout: Duration::from_secs(900),
            max_consecutive_timeouts: 10,
            progress_interval: Duration::from_secs(30),
            fallback_to_non_streaming: true,
        }
    }
}

impl StreamTimeoutConfig {
    /// Validates the configuration, returning an error message if invalid.
    ///
    /// Checks that all timeout durations are non-zero and that
    /// `total_stream_timeout` ≥ `initial_event_timeout`.
    ///
    /// # Errors
    ///
    /// Returns a string describing the first validation failure.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::StreamTimeoutConfig;
    /// use std::time::Duration;
    ///
    /// assert!(StreamTimeoutConfig::default().validate().is_ok());
    ///
    /// let bad = StreamTimeoutConfig {
    ///     initial_event_timeout: Duration::ZERO,
    ///     ..Default::default()
    /// };
    /// assert!(bad.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_event_timeout.is_zero() {
            return Err("initial_event_timeout must be non-zero".to_string());
        }
        if self.per_event_timeout.is_zero() {
            return Err("per_event_timeout must be non-zero".to_string());
        }
        if self.total_stream_timeout.is_zero() {
            return Err("total_stream_timeout must be non-zero".to_string());
        }
        if self.total_stream_timeout < self.initial_event_timeout {
            return Err(format!(
                "total_stream_timeout ({:?}) must be >= initial_event_timeout ({:?})",
                self.total_stream_timeout, self.initial_event_timeout
            ));
        }
        if self.progress_interval.is_zero() {
            return Err("progress_interval must be non-zero".to_string());
        }

        if self.max_consecutive_timeouts == 0 {
            return Err("max_consecutive_timeouts must be >= 1".to_string());
        }
        Ok(())
    }
}

// ===================================================
// StreamRetryConfig
// ===================================================

/// Configuration for retry behaviour when stream initialization fails.
///
/// Uses exponential backoff with jitter to avoid thundering herd when
/// multiple agents retry simultaneously.
///
/// # Backoff Formula
///
/// ```text
/// delay = min(base_delay * 2^attempt, max_delay) * (1.0 ± jitter)
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::stream::handler::StreamRetryConfig;
///
/// let config = StreamRetryConfig {
///     max_retries: 5,
///     base_delay_ms: 200,
///     ..Default::default()
/// };
/// assert_eq!(config.max_delay_ms, 10_000);
/// ```
#[derive(Debug, Clone)]
pub struct StreamRetryConfig {
    /// Maximum number of retry attempts for stream initialization.
    ///
    /// Each attempt opens a new stream and waits for the first event
    /// with the [`initial_event_timeout`](StreamTimeoutConfig::initial_event_timeout).
    pub max_retries: u32,

    /// Base delay in milliseconds before the first retry.
    ///
    /// Doubled on each subsequent retry attempt.
    pub base_delay_ms: u64,

    /// Maximum delay in milliseconds between retries.
    ///
    /// Caps the exponential growth so retries don't take too long.
    pub max_delay_ms: u64,

    /// Jitter factor (0.0 to 1.0) applied to the delay.
    ///
    /// Prevents thundering herd when multiple agents retry
    /// simultaneously. A factor of 0.1 means the delay varies
    /// by ±10%.
    pub jitter_factor: f64,
}

impl Default for StreamRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            jitter_factor: 0.1,
        }
    }
}

impl StreamRetryConfig {
    /// Calculate the backoff delay for a given attempt number (0-indexed).
    ///
    /// Returns the delay as a [`Duration`], capped at
    /// [`max_delay_ms`](Self::max_delay_ms). Does not apply jitter —
    /// callers should add jitter based on [`jitter_factor`](Self::jitter_factor).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::StreamRetryConfig;
    /// use std::time::Duration;
    ///
    /// let config = StreamRetryConfig::default();
    /// assert_eq!(config.base_delay(0), Duration::from_millis(100));
    /// assert_eq!(config.base_delay(1), Duration::from_millis(200));
    /// assert_eq!(config.base_delay(2), Duration::from_millis(400));
    /// ```
    #[must_use]
    pub fn base_delay(&self, attempt: u32) -> Duration {
        let delay_ms = self
            .base_delay_ms
            .saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
        Duration::from_millis(delay_ms.min(self.max_delay_ms))
    }

    /// Validates the configuration, returning an error message if invalid.
    ///
    /// Checks that `jitter_factor` is finite and within `0.0..=1.0`,
    /// and that all delay values are non-zero with `max_delay_ms` ≥ `base_delay_ms`.
    ///
    /// # Errors
    ///
    /// Returns a string describing the first validation failure.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::StreamRetryConfig;
    ///
    /// assert!(StreamRetryConfig::default().validate().is_ok());
    ///
    /// let bad = StreamRetryConfig { jitter_factor: 1.5, ..Default::default() };
    /// assert!(bad.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.base_delay_ms == 0 {
            return Err("base_delay_ms must be non-zero".to_string());
        }
        if self.max_delay_ms == 0 {
            return Err("max_delay_ms must be non-zero".to_string());
        }
        if self.max_delay_ms < self.base_delay_ms {
            return Err(format!(
                "max_delay_ms ({}) must be >= base_delay_ms ({})",
                self.max_delay_ms, self.base_delay_ms
            ));
        }
        if !self.jitter_factor.is_finite() {
            return Err(format!(
                "jitter_factor must be finite, got {}",
                self.jitter_factor
            ));
        }
        if !(0.0..=1.0).contains(&self.jitter_factor) {
            return Err(format!(
                "jitter_factor must be in 0.0..=1.0, got {}",
                self.jitter_factor
            ));
        }
        Ok(())
    }
}

// ===================================================
// StreamOutcome
// ===================================================

/// Why the stream ended.
///
/// Each variant captures the relevant context for how streaming
/// terminated. This allows callers to make informed decisions about
/// whether to retry, use partial data, or report an error.
///
/// # Ordering
///
/// The variants are ordered by severity:
///
/// ```text
/// Completed < TotalTimeout < EventTimeout < InitFailed < FallbackToNonStreaming < Cancelled
/// ```
#[derive(Debug, Clone)]
pub enum StreamOutcome {
    /// Stream completed normally — all events received, `MessageStop` seen.
    ///
    /// This is the happy path. The [`StreamAccumulator`](super::StreamAccumulator)
    /// contains the full response.
    Completed {
        /// Number of SSE events processed.
        events_processed: u64,
        /// Wall-clock duration of the stream.
        duration: Duration,
    },

    /// Total-stream timeout exceeded.
    ///
    /// The stream was active for longer than
    /// [`total_stream_timeout`](StreamTimeoutConfig::total_stream_timeout).
    /// Partial data may be available in the accumulator.
    TotalTimeout {
        /// Whether partial content was accumulated before timeout.
        has_partial_data: bool,
        /// Events processed before timeout.
        events_processed: u64,
        /// Duration before timeout was triggered.
        duration: Duration,
    },

    /// Per-event timeouts exhausted.
    ///
    /// Too many consecutive events failed to arrive within
    /// [`per_event_timeout`](StreamTimeoutConfig::per_event_timeout).
    /// Partial data may be available.
    EventTimeout {
        /// Whether partial content was accumulated.
        has_partial_data: bool,
        /// Consecutive timeouts that triggered the failure.
        consecutive_timeouts: u32,
    },

    /// Stream initialization failed after all retries.
    ///
    /// The handler could not get a first event from any retry attempt.
    /// No data was accumulated.
    InitFailed {
        /// The last error from the final retry attempt.
        last_error: String,
        /// Number of retry attempts made.
        attempts: u32,
    },

    /// Fell back to non-streaming [`create_message`](crate::api_client::ApiClient::create_message).
    ///
    /// Streaming failed, but a non-streaming request succeeded.
    /// The response is complete but was not streamed incrementally.
    FallbackToNonStreaming,

    /// Cancelled by the user via [`CancelSignal`](crate::cancel::CancelSignal).
    ///
    /// The stream was terminated because the user requested cancellation.
    /// Partial data may be available.
    Cancelled,
}

impl fmt::Display for StreamOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed {
                events_processed,
                duration,
            } => {
                write!(
                    f,
                    "stream completed ({events_processed} events in {:.1}s)",
                    duration.as_secs_f64()
                )
            }
            Self::TotalTimeout {
                has_partial_data,
                events_processed,
                duration,
            } => {
                let partial = if *has_partial_data {
                    " (partial data)"
                } else {
                    ""
                };
                write!(
                    f,
                    "total timeout after {:.1}s, {events_processed} events{partial}",
                    duration.as_secs_f64()
                )
            }
            Self::EventTimeout {
                has_partial_data,
                consecutive_timeouts,
            } => {
                let partial = if *has_partial_data {
                    " (partial data)"
                } else {
                    ""
                };
                write!(
                    f,
                    "event timeout after {consecutive_timeouts} consecutive timeouts{partial}"
                )
            }
            Self::InitFailed {
                last_error,
                attempts,
            } => {
                write!(f, "init failed after {attempts} attempts: {last_error}")
            }
            Self::FallbackToNonStreaming => {
                write!(f, "fell back to non-streaming request")
            }
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ===================================================
// StreamHandlerError
// ===================================================

/// Errors produced by [`StreamHandler`].
///
/// Each variant captures the specific failure mode, allowing callers
/// to distinguish between transient failures (retryable) and permanent
/// errors (non-retryable).
#[derive(Debug)]
pub enum StreamHandlerError {
    /// Streaming initialization failed after all retries.
    ///
    /// No data was accumulated. The [`StreamOutcome`] contains the
    /// last error and number of attempts.
    InitFailed(StreamOutcome),

    /// Streaming failed mid-stream.
    ///
    /// Some data may have been accumulated. The [`StreamOutcome`]
    /// describes the specific failure mode (timeout, error, etc.).
    StreamFailed(StreamOutcome),

    /// Both streaming and non-streaming fallback failed.
    ///
    /// The handler attempted a fallback `create_message()` call after
    /// streaming failed, but the fallback also produced an error.
    FallbackFailed {
        /// The streaming failure that triggered the fallback.
        stream_outcome: StreamOutcome,
        /// The error from the fallback request.
        fallback_error: String,
    },

    /// The operation was cancelled.
    ///
    /// The [`CancelSignal`](crate::cancel::CancelSignal) was triggered
    /// before the stream completed. Partial data may be available.
    Cancelled,
}

impl fmt::Display for StreamHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitFailed(outcome) => write!(f, "stream init failed: {outcome}"),
            Self::StreamFailed(outcome) => write!(f, "stream failed: {outcome}"),
            Self::FallbackFailed {
                stream_outcome,
                fallback_error,
            } => {
                write!(
                    f,
                    "stream failed ({stream_outcome}) and fallback also failed: {fallback_error}"
                )
            }
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for StreamHandlerError {}

// ===================================================
// StreamProgress
// ===================================================

/// Progress data emitted during long-running streams.
///
/// Passed to the progress callback at regular intervals
/// ([`progress_interval`](StreamTimeoutConfig::progress_interval))
/// to report stream health.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::handler::StreamProgress;
/// use std::time::Duration;
///
/// let progress = StreamProgress {
///     elapsed: Duration::from_secs(45),
///     events_processed: 127,
/// };
/// assert_eq!(progress.events_processed, 127);
/// ```
#[derive(Debug, Clone)]
pub struct StreamProgress {
    /// Time elapsed since the stream started.
    pub elapsed: Duration,
    /// Number of SSE events processed so far.
    pub events_processed: u64,
}

// ===================================================
// StreamHandler
// ===================================================

/// Holds configuration for the streaming resilience layer.
///
/// `StreamHandler` stores [`StreamTimeoutConfig`] and [`StreamRetryConfig`]
/// for use by the planned three-phase streaming lifecycle:
///
/// 1. **Initialize** *(planned)* — Opens a stream, waits for the first
///    event with a timeout, retries with exponential backoff on failure.
///
/// 2. **Process** *(planned)* — Consumes events with per-event and total
///    timeouts. Emits progress callbacks at regular intervals.
///
/// 3. **Recover** *(planned)* — If streaming fails, falls back to
///    [`ApiClient::create_message`](crate::api_client::ApiClient::create_message)
///    for a non-streaming response.
///
/// **Current state:** only config accessors are implemented. Runtime methods
/// (`stream_turn`, `init_with_retry`, `process_events`, `fallback_non_streaming`)
/// will be added in a future phase.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
///
/// let handler = StreamHandler::new();
/// assert_eq!(handler.timeout_config().initial_event_timeout, std::time::Duration::from_secs(120));
///
/// let handler = StreamHandler::with_config(
///     StreamTimeoutConfig {
///         initial_event_timeout: std::time::Duration::from_secs(60),
///         ..Default::default()
///     },
///     Default::default(),
/// );
/// assert_eq!(handler.timeout_config().initial_event_timeout, std::time::Duration::from_secs(60));
/// ```
pub struct StreamHandler {
    /// Timeout configuration for all phases.
    timeout_config: StreamTimeoutConfig,
    /// Retry configuration for stream initialization.
    retry_config: StreamRetryConfig,
}

impl fmt::Debug for StreamHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamHandler")
            .field("timeout_config", &self.timeout_config)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Default for StreamHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamHandler {
    /// Create a handler with default configuration.
    ///
    /// The defaults are suitable for production LLM API usage:
    /// - 120s initial event timeout
    /// - 300s per-event timeout
    /// - 900s total stream timeout
    /// - 3 retries with 100ms base delay
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::StreamHandler;
    ///
    /// let handler = StreamHandler::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout_config: StreamTimeoutConfig::default(),
            retry_config: StreamRetryConfig::default(),
        }
    }

    /// Create a handler with custom configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig, StreamRetryConfig};
    /// use std::time::Duration;
    ///
    /// let handler = StreamHandler::with_config(
    ///     StreamTimeoutConfig {
    ///         initial_event_timeout: Duration::from_secs(60),
    ///         ..Default::default()
    ///     },
    ///     StreamRetryConfig {
    ///         max_retries: 5,
    ///         ..Default::default()
    ///     },
    /// );
    /// ```
    #[must_use]
    pub fn with_config(timeout: StreamTimeoutConfig, retry: StreamRetryConfig) -> Self {
        Self {
            timeout_config: timeout,
            retry_config: retry,
        }
    }

    /// Returns a reference to the timeout configuration.
    #[must_use]
    pub fn timeout_config(&self) -> &StreamTimeoutConfig {
        &self.timeout_config
    }

    /// Returns a reference to the retry configuration.
    #[must_use]
    pub fn retry_config(&self) -> &StreamRetryConfig {
        &self.retry_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_config_default_values() {
        let config = StreamTimeoutConfig::default();
        assert_eq!(config.initial_event_timeout, Duration::from_secs(120));
        assert_eq!(config.per_event_timeout, Duration::from_secs(300));
        assert_eq!(config.total_stream_timeout, Duration::from_secs(900));
        assert_eq!(config.max_consecutive_timeouts, 10);
        assert_eq!(config.progress_interval, Duration::from_secs(30));
        assert!(config.fallback_to_non_streaming);
    }

    #[test]
    fn timeout_config_custom_values() {
        let config = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(30),
            per_event_timeout: Duration::from_secs(60),
            total_stream_timeout: Duration::from_secs(300),
            max_consecutive_timeouts: 5,
            progress_interval: Duration::from_secs(10),
            fallback_to_non_streaming: false,
        };
        assert_eq!(config.initial_event_timeout, Duration::from_secs(30));
        assert!(!config.fallback_to_non_streaming);
    }

    #[test]
    fn retry_config_default_values() {
        let config = StreamRetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 10_000);
        assert!((config.jitter_factor - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_config_base_delay_exponential() {
        let config = StreamRetryConfig::default();
        assert_eq!(config.base_delay(0), Duration::from_millis(100));
        assert_eq!(config.base_delay(1), Duration::from_millis(200));
        assert_eq!(config.base_delay(2), Duration::from_millis(400));
        assert_eq!(config.base_delay(3), Duration::from_millis(800));
    }

    #[test]
    fn retry_config_base_delay_capped_at_max() {
        let config = StreamRetryConfig {
            base_delay_ms: 1000,
            max_delay_ms: 5000,
            ..Default::default()
        };
        // 1000 * 2^3 = 8000, capped at 5000
        assert_eq!(config.base_delay(3), Duration::from_millis(5000));
    }

    #[test]
    fn outcome_completed_display() {
        let outcome = StreamOutcome::Completed {
            events_processed: 42,
            duration: Duration::from_secs(5),
        };
        let s = outcome.to_string();
        assert!(s.contains("42 events"));
        assert!(s.contains("5.0s"));
    }

    #[test]
    fn outcome_total_timeout_display() {
        let outcome = StreamOutcome::TotalTimeout {
            has_partial_data: true,
            events_processed: 10,
            duration: Duration::from_secs(900),
        };
        let s = outcome.to_string();
        assert!(s.contains("partial data"));
        assert!(s.contains("900.0s"));
    }

    #[test]
    fn outcome_event_timeout_display() {
        let outcome = StreamOutcome::EventTimeout {
            has_partial_data: false,
            consecutive_timeouts: 10,
        };
        let s = outcome.to_string();
        assert!(s.contains("10 consecutive"));
        assert!(!s.contains("partial data"));
    }

    #[test]
    fn outcome_init_failed_display() {
        let outcome = StreamOutcome::InitFailed {
            last_error: "connection refused".to_string(),
            attempts: 3,
        };
        let s = outcome.to_string();
        assert!(s.contains("3 attempts"));
        assert!(s.contains("connection refused"));
    }

    #[test]
    fn outcome_fallback_display() {
        let outcome = StreamOutcome::FallbackToNonStreaming;
        let s = outcome.to_string();
        assert!(s.contains("non-streaming"));
    }

    #[test]
    fn outcome_cancelled_display() {
        let outcome = StreamOutcome::Cancelled;
        assert_eq!(outcome.to_string(), "cancelled");
    }

    #[test]
    fn error_init_failed_display() {
        let outcome = StreamOutcome::InitFailed {
            last_error: "timeout".to_string(),
            attempts: 3,
        };
        let err = StreamHandlerError::InitFailed(outcome);
        let s = err.to_string();
        assert!(s.contains("init failed"));
    }

    #[test]
    fn error_stream_failed_display() {
        let outcome = StreamOutcome::EventTimeout {
            has_partial_data: true,
            consecutive_timeouts: 5,
        };
        let err = StreamHandlerError::StreamFailed(outcome);
        let s = err.to_string();
        assert!(s.contains("stream failed"));
    }

    #[test]
    fn error_fallback_failed_display() {
        let stream_outcome = StreamOutcome::TotalTimeout {
            has_partial_data: false,
            events_processed: 0,
            duration: Duration::from_secs(900),
        };
        let err = StreamHandlerError::FallbackFailed {
            stream_outcome,
            fallback_error: "api error 429".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("fallback also failed"));
        assert!(s.contains("429"));
    }

    #[test]
    fn error_cancelled_display() {
        let err = StreamHandlerError::Cancelled;
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn progress_fields() {
        let progress = StreamProgress {
            elapsed: Duration::from_secs(45),
            events_processed: 127,
        };
        assert_eq!(progress.elapsed, Duration::from_secs(45));
        assert_eq!(progress.events_processed, 127);
    }

    #[test]
    fn handler_new_defaults() {
        let handler = StreamHandler::new();
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_secs(120),
        );
        assert_eq!(handler.retry_config().max_retries, 3);
    }

    #[test]
    fn handler_with_config() {
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(60),
                ..Default::default()
            },
            StreamRetryConfig {
                max_retries: 5,
                ..Default::default()
            },
        );
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_secs(60),
        );
        assert_eq!(handler.retry_config().max_retries, 5);
    }

    #[test]
    fn handler_default_trait() {
        let handler = StreamHandler::default();
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_secs(120),
        );
    }

    #[test]
    fn handler_debug_format() {
        let handler = StreamHandler::new();
        let debug = format!("{handler:?}");
        assert!(debug.contains("StreamHandler"));
        assert!(debug.contains("timeout_config"));
    }

    // ===================================================
    // StreamTimeoutConfig::validate tests
    // ===================================================

    #[test]
    fn timeout_config_validate_default_ok() {
        assert!(StreamTimeoutConfig::default().validate().is_ok());
    }

    #[test]
    fn timeout_config_validate_zero_initial() {
        let config = StreamTimeoutConfig {
            initial_event_timeout: Duration::ZERO,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("initial_event_timeout"));
    }

    #[test]
    fn timeout_config_validate_zero_per_event() {
        let config = StreamTimeoutConfig {
            per_event_timeout: Duration::ZERO,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("per_event_timeout"));
    }

    #[test]
    fn timeout_config_validate_zero_total() {
        let config = StreamTimeoutConfig {
            total_stream_timeout: Duration::ZERO,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("total_stream_timeout"));
    }

    #[test]
    fn timeout_config_validate_total_less_than_initial() {
        let config = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(120),
            total_stream_timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("total_stream_timeout"));
        assert!(err.contains("initial_event_timeout"));
    }

    #[test]
    fn timeout_config_validate_zero_progress() {
        let config = StreamTimeoutConfig {
            progress_interval: Duration::ZERO,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("progress_interval"));
    }

    // ===================================================
    // StreamRetryConfig::validate tests
    // ===================================================

    #[test]
    fn retry_config_validate_default_ok() {
        assert!(StreamRetryConfig::default().validate().is_ok());
    }

    #[test]
    fn retry_config_validate_zero_base_delay() {
        let config = StreamRetryConfig {
            base_delay_ms: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("base_delay_ms"));
    }

    #[test]
    fn retry_config_validate_zero_max_delay() {
        let config = StreamRetryConfig {
            max_delay_ms: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("max_delay_ms"));
    }

    #[test]
    fn retry_config_validate_max_less_than_base() {
        let config = StreamRetryConfig {
            base_delay_ms: 1000,
            max_delay_ms: 500,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("max_delay_ms"));
        assert!(err.contains("base_delay_ms"));
    }

    #[test]
    fn retry_config_validate_jitter_nan() {
        let config = StreamRetryConfig {
            jitter_factor: f64::NAN,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("finite"));
    }

    #[test]
    fn retry_config_validate_jitter_infinity() {
        let config = StreamRetryConfig {
            jitter_factor: f64::INFINITY,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("finite"));
    }

    #[test]
    fn retry_config_validate_jitter_above_one() {
        let config = StreamRetryConfig {
            jitter_factor: 1.5,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("0.0..=1.0"));
    }

    #[test]
    fn retry_config_validate_jitter_negative() {
        let config = StreamRetryConfig {
            jitter_factor: -0.1,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("0.0..=1.0"));
    }

    #[test]
    fn retry_config_validate_jitter_boundaries() {
        // 0.0 and 1.0 are valid boundaries.
        let config = StreamRetryConfig {
            jitter_factor: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        let config = StreamRetryConfig {
            jitter_factor: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
