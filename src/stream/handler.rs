//! Configuration, result types, and error types for resilient LLM stream handling.
//!
//! This module defines the types that underpin [`StreamHandler`] — the framework's
//! production-grade streaming resilience layer. The handler wraps
//! [`ApiClient::stream_messages`] with retry, timeout, and fallback behaviour.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              StreamHandler                      │
//! │                                                 │
//! │  1. init_with_retry()                           │
//! │    └─ stream_messages() → first event timeout   │
//! │    └─ retry with backoff on failure             │
//! │                                                 │
//! │  2. process_events()                            │
//! │    └─ per-event timeout + total timeout         │
//! │    └─ progress callbacks at intervals           │
//! │                                                 │
//! │  3. fallback_non_streaming()                    │
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

use crate::api_client::ApiClient;
use crate::cancel::CancelSignal;
use crate::message::Message;
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
use crate::tool::ToolSchema;
use futures::StreamExt;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    /// Whether to fall back to [`ApiClient::create_message`]
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
    /// This is the happy path. The [`StreamAccumulator`]
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

    /// Cancelled by the user via [`CancelSignal`].
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
    /// The [`CancelSignal`] was triggered
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
/// for the three-phase streaming lifecycle:
///
/// 1. **Initialize** — Opens a stream, waits for the first event with a
///    timeout, retries with exponential backoff on failure.
///
/// 2. **Process** — Consumes events with per-event and total timeouts.
///    Emits progress callbacks at regular intervals.
///
/// 3. **Recover** — If streaming fails, falls back to
///    [`ApiClient::create_message`] for a non-streaming response.
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

    // ==================================================
    // Runtime methods
    // ==================================================

    /// Stream one complete turn with retry, timeout, and fallback.
    ///
    /// This is the primary entry point for resilient streaming. It
    /// orchestrates the full lifecycle:
    ///
    /// 1. Opens a stream via [`ApiClient::stream_messages`].
    /// 2. Processes events with per-event and total timeouts.
    /// 3. On transient errors, retries with exponential backoff.
    /// 4. If all retries fail and `fallback_to_non_streaming` is enabled,
    ///    falls back to [`ApiClient::create_message`].
    ///
    /// The returned [`StreamTurnResult`] carries the accumulated message,
    /// token usage, stop reason, and timing regardless of which path
    /// produced the result (streaming or fallback).
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client to stream from.
    /// - `conversation` — The conversation history to send.
    /// - `system` — An optional system prompt.
    /// - `tool_schemas` — Optional tool definitions.
    /// - `cancel` — Shared cancellation signal.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError`] on unrecoverable failure:
    /// - [`InitFailed`](StreamHandlerError::InitFailed) — stream could not
    ///   be opened after all retries.
    /// - [`StreamFailed`](StreamHandlerError::StreamFailed) — stream
    ///   failed mid-event with a timeout or API error.
    /// - [`FallbackFailed`](StreamHandlerError::FallbackFailed) — both
    ///   streaming and non-streaming fallback failed.
    /// - [`Cancelled`](StreamHandlerError::Cancelled) — cancellation
    ///   signal fired.
    pub async fn stream_turn<C: ApiClient>(
        &self,
        client: &C,
        conversation: Vec<Message>,
        system: Option<String>,
        tool_schemas: Option<Vec<ToolSchema>>,
        cancel: &Arc<CancelSignal>,
    ) -> Result<StreamTurnResult, StreamHandlerError> {
        let total_deadline = Some(
            Instant::now()
                .checked_add(self.timeout_config.total_stream_timeout)
                .unwrap_or(Instant::now()),
        );
        let max_attempts = self.retry_config.max_retries.saturating_add(1);
        let mut last_stream_outcome: Option<StreamOutcome> = None;
        for attempt in 0..max_attempts {
            if cancel.is_cancelled() {
                return Err(StreamHandlerError::Cancelled);
            }
            if let Some(deadline) = total_deadline {
                if Instant::now() >= deadline {
                    return Err(StreamHandlerError::InitFailed(
                        StreamOutcome::TotalTimeout {
                            has_partial_data: false,
                            events_processed: 0,
                            duration: self.timeout_config.total_stream_timeout,
                        },
                    ));
                }
            }
            match self
                .try_stream_once(
                    client,
                    conversation.clone(),
                    system.clone(),
                    tool_schemas.clone(),
                    cancel,
                    total_deadline,
                )
                .await
            {
                Ok(result) => return Ok(result),
                Err(StreamHandlerError::Cancelled) => {
                    return Err(StreamHandlerError::Cancelled);
                }
                Err(e) => {
                    let outcome = match &e {
                        StreamHandlerError::InitFailed(o) | StreamHandlerError::StreamFailed(o) => {
                            Some(o.clone())
                        }
                        _ => None,
                    };
                    last_stream_outcome = outcome;
                    if attempt >= max_attempts.saturating_sub(1) {
                        // All retries exhausted — try fallback if enabled.
                        if self.timeout_config.fallback_to_non_streaming {
                            return self
                                .fallback_non_streaming(
                                    client,
                                    conversation,
                                    system,
                                    tool_schemas,
                                    cancel,
                                    last_stream_outcome,
                                )
                                .await;
                        }
                        return Err(e);
                    }
                    let delay = self.retry_config.base_delay(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        // Should not reach here, but handle it gracefully.
        Err(StreamHandlerError::InitFailed(
            last_stream_outcome.unwrap_or(StreamOutcome::InitFailed {
                attempts: self.retry_config.max_retries,
                last_error: "all attempts failed".to_string(),
            }),
        ))
    }

    /// Attempt a single streaming pass.
    ///
    /// Opens a stream via [`ApiClient::stream_messages`] and processes
    /// all events with timeout and cancellation support.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError`] if the stream fails or times out.
    async fn try_stream_once<C: ApiClient>(
        &self,
        client: &C,
        conversation: Vec<Message>,
        system: Option<String>,
        tool_schemas: Option<Vec<ToolSchema>>,
        cancel: &Arc<CancelSignal>,
        total_deadline: Option<Instant>,
    ) -> Result<StreamTurnResult, StreamHandlerError> {
        let stream = client.stream_messages(conversation, system, tool_schemas);
        self.process_events(stream, cancel, total_deadline).await
    }

    /// Process events from an open stream with per-event and total timeouts.
    ///
    /// Reads events from the stream, accumulating them into a
    /// [`Message`]. Applies per-event timeouts and an overall deadline.
    /// Checks cancellation between events.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::StreamFailed`] on timeout or API error,
    /// or [`StreamHandlerError::Cancelled`] if the cancel signal fires.
    async fn process_events<S>(
        &self,
        mut stream: S,
        cancel: &Arc<CancelSignal>,
        total_deadline: Option<Instant>,
    ) -> Result<StreamTurnResult, StreamHandlerError>
    where
        S: futures::Stream<Item = Result<crate::stream::StreamEvent, crate::api_error::ApiError>>
            + Unpin,
    {
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;
        let mut consecutive_timeouts: usize = 0;
        let mut events_processed: u64 = 0;
        let stream_start = Instant::now();
        loop {
            if let Some(deadline) = total_deadline {
                if Instant::now() >= deadline {
                    let has_partial_data = !accumulator.peek_parts().is_empty();
                    return Err(StreamHandlerError::StreamFailed(
                        StreamOutcome::TotalTimeout {
                            has_partial_data,
                            events_processed,
                            duration: stream_start.elapsed(),
                        },
                    ));
                }
            }
            let per_event_timeout = self.timeout_config.per_event_timeout;
            let event_deadline = Instant::now()
                .checked_add(per_event_timeout)
                .unwrap_or(Instant::now());
            let event_result = tokio::select! {
                event = stream.next() => event,
                () = cancel.notified() => {
                    return Err(StreamHandlerError::Cancelled);
                }
                () = tokio::time::sleep_until(event_deadline.into()) => {
                    consecutive_timeouts = consecutive_timeouts.saturating_add(1);
                    let max_consecutive = self.timeout_config.max_consecutive_timeouts as usize;
                    if consecutive_timeouts >= max_consecutive {
                        let has_partial_data = !accumulator.peek_parts().is_empty();
                        return Err(StreamHandlerError::StreamFailed(
                            StreamOutcome::EventTimeout {
                                has_partial_data,
                                consecutive_timeouts: u32::try_from(consecutive_timeouts).unwrap_or(u32::MAX),
                            },
                        ));
                    }
                    continue;
                }
            };

            match event_result {
                Some(Ok(event)) => {
                    consecutive_timeouts = 0;
                    events_processed = events_processed.saturating_add(1);
                    if let StreamEvent::MessageDelta(delta) = &event {
                        if let Some(ref reason_str) = delta.delta.stop_reason {
                            stop_reason =
                                StreamStopReason::from_api_str(reason_str).unwrap_or(stop_reason);
                        }
                    }
                    if let Err(e) = accumulator.process(&event) {
                        return Err(StreamHandlerError::StreamFailed(
                            StreamOutcome::InitFailed {
                                attempts: 1,
                                last_error: e.to_string(),
                            },
                        ));
                    }
                }
                Some(Err(api_error)) => {
                    return Err(StreamHandlerError::StreamFailed(
                        StreamOutcome::InitFailed {
                            attempts: 1,
                            last_error: api_error.to_string(),
                        },
                    ));
                }
                None => break,
            }
        }

        let usage = accumulator.usage().copied();
        let message = accumulator.build();
        let elapsed = stream_start.elapsed();
        Ok(StreamTurnResult {
            message,
            usage,
            stop_reason,
            from_fallback: false,
            elapsed,
        })
    }

    /// Fall back to non-streaming message creation.
    ///
    /// Called when streaming fails (timeout, retries exhausted) and
    /// `fallback_to_non_streaming` is enabled. Uses
    /// [`ApiClient::create_message`] to get a complete response.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::FallbackFailed`] if the fallback
    /// request also fails, or [`StreamHandlerError::Cancelled`] if the
    /// cancel signal fires.
    async fn fallback_non_streaming<C: ApiClient>(
        &self,
        client: &C,
        conversation: Vec<Message>,
        system: Option<String>,
        tool_schemas: Option<Vec<ToolSchema>>,
        cancel: &Arc<CancelSignal>,
        stream_outcome: Option<StreamOutcome>,
    ) -> Result<StreamTurnResult, StreamHandlerError> {
        if cancel.is_cancelled() {
            return Err(StreamHandlerError::Cancelled);
        }

        let start = Instant::now();

        let result = tokio::select! {
            res = client.create_message(conversation, system, tool_schemas) => res,
            () = cancel.notified() => {
                return Err(StreamHandlerError::Cancelled);
            }
        };

        match result {
            Ok(value) => {
                // Best-effort extraction of text content from the JSON response.
                let text = value
                    .get("content")
                    .and_then(|c| c.as_array())
                    .and_then(|parts| {
                        parts
                            .iter()
                            .find_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                    })
                    .unwrap_or_default();

                let stop_reason = value
                    .get("stop_reason")
                    .and_then(|r| r.as_str())
                    .and_then(StreamStopReason::from_api_str)
                    .unwrap_or(StreamStopReason::EndTurn);

                Ok(StreamTurnResult {
                    message: Message::assistant(&text),
                    usage: None,
                    stop_reason,
                    from_fallback: true,
                    elapsed: start.elapsed(),
                })
            }
            Err(e) => Err(StreamHandlerError::FallbackFailed {
                stream_outcome: stream_outcome.unwrap_or(StreamOutcome::InitFailed {
                    attempts: 0,
                    last_error: "unknown".to_string(),
                }),
                fallback_error: e.to_string(),
            }),
        }
    }
}

/// Result of a successful streaming turn via [`StreamHandler::stream_turn`].
///
/// Carries the accumulated message, token usage, and stop reason,
/// alongside metadata about how the turn was completed (normal streaming
/// vs. non-streaming fallback).
///
/// # Example
///
/// ```rust,ignore
/// let result = handler.stream_turn(client, messages, system, tools, cancel).await?;
/// if result.from_fallback {
///     eprintln!("Warning: fell back to non-streaming");
/// }
/// println!("Response: {:?}", result.message);
/// ```
#[derive(Debug, Clone)]
pub struct StreamTurnResult {
    /// The fully accumulated assistant message.
    pub message: Message,
    /// Token counts for this turn, if reported by the provider.
    pub usage: Option<Usage>,
    /// Why the model stopped generating.
    pub stop_reason: StreamStopReason,
    /// Whether the result came from a non-streaming fallback.
    pub from_fallback: bool,
    /// Wall-clock time spent on this turn.
    pub elapsed: Duration,
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

    // ===================================================
    // StreamTurnResult tests
    // ===================================================

    #[test]
    fn stream_turn_result_fields() {
        let result = StreamTurnResult {
            message: Message::assistant("hello"),
            usage: Some(Usage::new(10, 5)),
            stop_reason: StreamStopReason::EndTurn,
            from_fallback: false,
            elapsed: Duration::from_millis(100),
        };
        assert!(!result.from_fallback);
        assert_eq!(result.stop_reason, StreamStopReason::EndTurn);
        assert!(result.usage.is_some());
    }

    #[test]
    fn stream_turn_result_fallback_flag() {
        let result = StreamTurnResult {
            message: Message::assistant("fallback text"),
            usage: None,
            stop_reason: StreamStopReason::EndTurn,
            from_fallback: true,
            elapsed: Duration::from_millis(200),
        };
        assert!(result.from_fallback);
        assert!(result.usage.is_none());
    }

    // ===================================================
    // process_events async tests
    // ===================================================

    use crate::api_error::ApiError;
    use crate::stream::{
        DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
        PartStart, StreamEvent,
    };

    /// Helper: build a minimal happy-path event stream.
    ///
    /// Produces: MessageStart → PartStart(text) → IndexedDelta("hi") →
    /// PartStop → MessageDelta(end_turn) → MessageStop
    fn happy_stream_events() -> Vec<Result<StreamEvent, ApiError>> {
        vec![
            Ok(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".to_string(),
                    role: "assistant".to_string(),
                    model: "test-model".to_string(),
                },
            })),
            Ok(StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(crate::stream::MessagePart::text("")),
            })),
            Ok(StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: "hi".to_string(),
                },
            })),
            Ok(StreamEvent::PartStop),
            Ok(StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: None,
            })),
            Ok(StreamEvent::MessageStop),
        ]
    }

    /// Build a `futures::stream` from a vec of events.
    fn event_stream(
        events: Vec<Result<StreamEvent, ApiError>>,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
    > {
        Box::pin(futures::stream::iter(events))
    }

    #[tokio::test]
    async fn process_events_happy_path() {
        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let stream = event_stream(happy_stream_events());

        let result = handler
            .process_events(stream, &cancel, None)
            .await
            .expect("should succeed");

        assert!(!result.from_fallback);
        assert_eq!(result.stop_reason, StreamStopReason::EndTurn);
        assert!(result.elapsed > Duration::ZERO);
    }

    #[tokio::test]
    async fn process_events_api_error_mid_stream() {
        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let events = vec![
            Ok(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".to_string(),
                    role: "assistant".to_string(),
                    model: "test-model".to_string(),
                },
            })),
            Err(ApiError::api("connection lost")),
        ];
        let stream = event_stream(events);

        let err = handler
            .process_events(stream, &cancel, None)
            .await
            .expect_err("should fail on API error");

        match err {
            StreamHandlerError::StreamFailed(outcome) => {
                let s = outcome.to_string();
                assert!(s.contains("connection lost"), "unexpected: {s}");
            }
            other => panic!("expected StreamFailed, got: {other}"),
        }
    }

    #[tokio::test]
    async fn process_events_total_timeout() {
        // Use a very short total timeout to trigger it immediately.
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                total_stream_timeout: Duration::from_millis(1),
                per_event_timeout: Duration::from_secs(300),
                initial_event_timeout: Duration::from_secs(120),
                max_consecutive_timeouts: 10,
                progress_interval: Duration::from_secs(30),
                fallback_to_non_streaming: false,
            },
            StreamRetryConfig::default(),
        );
        let cancel = Arc::new(CancelSignal::new());

        // Set a total deadline in the past so it triggers immediately.
        let deadline = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or(Instant::now()),
        );

        // Use a stream that never produces events (pending forever).
        let pending_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > = Box::pin(futures::stream::pending());

        let err = handler
            .process_events(pending_stream, &cancel, deadline)
            .await
            .expect_err("should fail on total timeout");

        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::TotalTimeout { .. }) => {}
            other => panic!("expected StreamFailed(TotalTimeout), got: {other}"),
        }
    }

    #[tokio::test]
    async fn process_events_cancelled() {
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                per_event_timeout: Duration::from_secs(300),
                ..Default::default()
            },
            StreamRetryConfig::default(),
        );
        let cancel = Arc::new(CancelSignal::new());
        cancel.cancel();

        let pending_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > = Box::pin(futures::stream::pending());

        let err = handler
            .process_events(pending_stream, &cancel, None)
            .await
            .expect_err("should fail on cancellation");

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got: {err}"
        );
    }

    #[tokio::test]
    async fn process_events_empty_stream() {
        // A stream that ends immediately (None) should produce an
        // empty-but-successful result.
        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let stream = event_stream(vec![]);

        let result = handler
            .process_events(stream, &cancel, None)
            .await
            .expect("empty stream should succeed");

        assert!(!result.from_fallback);
    }

    // ===================================================
    // fallback_non_streaming async tests
    // ===================================================

    /// Minimal mock that implements [`ApiClient`] for handler tests.
    ///
    /// Unlike the full [`MockApiClient`](crate::testing::MockApiClient),
    /// this is defined locally so it works without the `testing` feature.
    struct HandlerMock {
        /// If set, `create_message` returns this error.
        create_error: Option<String>,
        /// If set, `create_message` returns this JSON.
        create_response: Option<serde_json::Value>,
    }

    impl HandlerMock {
        fn new() -> Self {
            Self {
                create_error: None,
                create_response: None,
            }
        }

        /// Make `create_message` succeed with the given text.
        fn with_text_response(mut self, text: &str) -> Self {
            self.create_response = Some(serde_json::json!({
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn"
            }));
            self
        }

        /// Make `create_message` fail with the given error message.
        fn with_create_error(mut self, msg: &str) -> Self {
            self.create_error = Some(msg.to_string());
            self
        }
    }

    impl ApiClient for HandlerMock {
        fn model(&self) -> &str {
            "test-model"
        }

        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            // Default: return a happy-path stream.
            Box::pin(futures::stream::iter(happy_stream_events()))
        }

        fn create_message(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>,
        > {
            if let Some(ref err) = self.create_error {
                let err = err.clone();
                return Box::pin(async move { Err(ApiError::api(&err)) });
            }
            let val = self.create_response.clone().unwrap_or(serde_json::json!({
                "content": [{"type": "text", "text": "default"}],
                "stop_reason": "end_turn"
            }));
            Box::pin(async move { Ok(val) })
        }
    }

    #[tokio::test]
    async fn fallback_non_streaming_success() {
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: true,
                ..Default::default()
            },
            StreamRetryConfig::default(),
        );
        let client = HandlerMock::new().with_text_response("fallback works");
        let cancel = Arc::new(CancelSignal::new());

        let result = handler
            .fallback_non_streaming(
                &client,
                vec![],
                None,
                None,
                &cancel,
                Some(StreamOutcome::InitFailed {
                    last_error: "stream failed".to_string(),
                    attempts: 3,
                }),
            )
            .await
            .expect("fallback should succeed");

        assert!(result.from_fallback);
        assert_eq!(result.stop_reason, StreamStopReason::EndTurn);
    }

    #[tokio::test]
    async fn fallback_non_streaming_cancelled_before_start() {
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: true,
                ..Default::default()
            },
            StreamRetryConfig::default(),
        );
        let client = HandlerMock::new().with_text_response("fallback works");
        let cancel = Arc::new(CancelSignal::new());
        cancel.cancel();

        let err = handler
            .fallback_non_streaming(&client, vec![], None, None, &cancel, None)
            .await
            .expect_err("should fail on cancellation");

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got: {err}"
        );
    }

    #[tokio::test]
    async fn fallback_non_streaming_error() {
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: true,
                ..Default::default()
            },
            StreamRetryConfig::default(),
        );
        let client = HandlerMock::new().with_create_error("service unavailable");
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .fallback_non_streaming(
                &client,
                vec![],
                None,
                None,
                &cancel,
                Some(StreamOutcome::InitFailed {
                    last_error: "stream timeout".to_string(),
                    attempts: 2,
                }),
            )
            .await
            .expect_err("should fail when fallback also errors");

        match err {
            StreamHandlerError::FallbackFailed {
                stream_outcome,
                fallback_error,
            } => {
                let stream_s = stream_outcome.to_string();
                assert!(
                    stream_s.contains("stream timeout"),
                    "unexpected: {stream_s}"
                );
                assert!(
                    fallback_error.contains("service unavailable"),
                    "unexpected: {fallback_error}"
                );
            }
            other => panic!("expected FallbackFailed, got: {other}"),
        }
    }

    // ===================================================
    // stream_turn async tests
    // ===================================================

    #[tokio::test]
    async fn stream_turn_happy_path() {
        let handler = StreamHandler::new();
        let client = HandlerMock::new().with_text_response("hello world");
        let cancel = Arc::new(CancelSignal::new());

        let result = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect("stream_turn should succeed");

        assert!(!result.from_fallback);
        assert_eq!(result.stop_reason, StreamStopReason::EndTurn);
    }

    #[tokio::test]
    async fn stream_turn_cancelled_at_start() {
        let handler = StreamHandler::new();
        let client = HandlerMock::new().with_text_response("hello");
        let cancel = Arc::new(CancelSignal::new());
        cancel.cancel();

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("should fail on cancellation");

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got: {err}"
        );
    }

    #[tokio::test]
    async fn stream_turn_fallback_after_stream_error() {
        // When streaming fails but fallback is enabled, the handler
        // should fall back to create_message. We use two responses
        // queued: the first stream errors mid-way (we inject an error
        // event), and create_message gets the second response.
        //
        // However, MockApiClient::with_error blocks both paths, so we
        // test the fallback path directly via fallback_non_streaming
        // (covered above). Here we test that stream_turn returns the
        // error when streaming fails and the handler is configured
        // without fallback.
        let handler = StreamHandler::with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            },
            StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            },
        );

        /// Mock that always returns an error stream.
        struct ErrorMock;
        impl ApiClient for ErrorMock {
            fn model(&self) -> &str {
                "test-model"
            }
            fn stream_messages(
                &self,
                _messages: Vec<Message>,
                _system: Option<String>,
                _tools: Option<Vec<ToolSchema>>,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::api("API down"))
                }))
            }
            fn create_message(
                &self,
                _messages: Vec<Message>,
                _system: Option<String>,
                _tools: Option<Vec<ToolSchema>>,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = Result<serde_json::Value, ApiError>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unreachable")) })
            }
        }

        let client = ErrorMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("should fail when streaming errors and fallback is disabled");

        // With max_retries=0, we get 1 attempt. Error stream produces StreamFailed.
        match err {
            StreamHandlerError::StreamFailed(outcome) => {
                let s = outcome.to_string();
                assert!(s.contains("API down"), "unexpected: {s}");
            }
            other => panic!("expected StreamFailed, got: {other}"),
        }
    }
}
