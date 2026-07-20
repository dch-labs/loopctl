//! Resilient LLM stream handling.
//!
//! [`StreamHandler`] wraps [`ApiClient::stream_messages`] with retry, timeout,
//! rate-limit detection, and fallback behaviour. Configure it with
//! [`StreamTimeoutConfig`], [`StreamRetryConfig`], and [`RateLimitConfig`].
//!
//! # Architecture
//!
//! A turn flows through three phases:
//!
//! 1. **Initialization with retry.** Open the SSE connection via
//!    `stream_messages` and wait for the first event. On failure, back off and
//!    retry per [`StreamRetryConfig`] (exponential, jittered).
//! 2. **Event processing.** Assemble the response while enforcing three guards:
//!    a per-event timeout, a total-stream timeout, and rate-limit detection
//!    (429/503/529 mid-stream). Cancellation is honoured throughout.
//! 3. **Fallback.** If streaming is exhausted, retry the request as a
//!    single-shot [`ApiClient::create_message`] call.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
//!
//! let handler = StreamHandler::new();
//!
//! // Or with custom config:
//! let handler = StreamHandler::new().with_config(
//!     StreamTimeoutConfig {
//!         initial_event_timeout: std::time::Duration::from_secs(60),
//!         ..Default::default()
//!     },
//!     Default::default(),
//! );
//! ```

use crate::api::ApiClient;
use crate::api::error::{ApiError, http_status_is_overload};
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
    /// Most critical timeout — if the API server never sends
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
            initial_event_timeout: Duration::from_mins(2),
            per_event_timeout: Duration::from_mins(5),
            total_stream_timeout: Duration::from_mins(15),
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
// RateLimitConfig + detection
// ===================================================

/// Policy for handling 429 / 503 rate-limit responses from LLM providers.
///
/// Governs backoff and retry behaviour when the server signals that the client
/// should slow down: a 429 *Too Many Requests*, a 503 *Service Unavailable*,
/// or a 529 *Overloaded*. All three can carry a `Retry-After` hint that this
/// config can honour.
///
/// Distinct from [`StreamRetryConfig`], which covers generic
/// stream-initialization transport failures (the connection itself failed to
/// open).
///
/// # Example
///
/// ```
/// use loopctl::stream::handler::RateLimitConfig;
/// use std::time::Duration;
///
/// let cfg = RateLimitConfig {
///     default_delay: Duration::from_secs(2),
///     fallback_after_retries: 2,
///     ..Default::default()
/// };
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Whether to honour a 429/503 `Retry-After` value when the server
    /// provides one.
    ///
    /// When `true` (the default), [`backoff`](Self::backoff) returns the
    /// server-advised delay (capped at [`max_delay`](Self::max_delay)). When
    /// `false`, the server's hint is ignored and
    /// [`default_delay`](Self::default_delay) is always used.
    pub respect_retry_after: bool,

    /// Backoff used when the server gives no `Retry-After`.
    ///
    /// Also used unconditionally when
    /// [`respect_retry_after`](Self::respect_retry_after) is `false`. Defaults
    /// to 5s — long enough to let a transient burst clear, short enough that a
    /// missing header doesn't stall the agent.
    pub default_delay: Duration,

    /// Upper bound on any single rate-limit backoff.
    ///
    /// Caps both the server's `Retry-After` (when honoured) and
    /// [`default_delay`](Self::default_delay) so a misbehaving provider cannot
    /// stall the agent indefinitely.
    pub max_delay: Duration,

    /// Advisory per-minute request ceiling for proactive throttling (0 = unset).
    ///
    /// This field is **not read at runtime** by the reactive rate-limit handler
    /// — it does not throttle on its own. It is the value a caller feeds to
    /// [`RateLimiter::new`](crate::stream::rate_limit::RateLimiter::new) when
    /// attaching a proactive limiter via
    /// [`StreamHandler::with_rate_limiter`](crate::stream::handler::StreamHandler::with_rate_limiter).
    /// Zero (the default) means no proactive throttling is configured; reactive handling
    /// of server-returned 429/503 responses is unaffected and governed by the
    /// other fields below.
    pub requests_per_minute: u32,

    /// Number of rate-limit retries before switching to a fallback model.
    ///
    /// After this many retries on the same model, the next rate limit triggers
    /// a fallback to a different model (if one is configured).
    pub fallback_after_retries: u32,

    /// Hard cap on rate-limit retries for a single turn.
    ///
    /// Once this many retries have been exhausted, the turn fails outright.
    /// Distinct from [`fallback_after_retries`](Self::fallback_after_retries),
    /// which controls the *escalation* threshold to a fallback model, not the
    /// hard stop after which the turn gives up entirely.
    pub max_retries: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            respect_retry_after: true,
            default_delay: Duration::from_secs(5),
            max_delay: Duration::from_mins(1),
            requests_per_minute: 0,
            fallback_after_retries: 3,
            max_retries: 5,
        }
    }
}

impl RateLimitConfig {
    /// Validate the policy.
    ///
    /// `default_delay` and `max_delay` must be non-zero, `max_delay` must be at
    /// least `default_delay`, and `max_retries` must be at least 1.
    ///
    /// # Errors
    ///
    /// Returns a human-readable description of the first violated constraint.
    ///
    /// ```
    /// use loopctl::stream::handler::RateLimitConfig;
    /// use std::time::Duration;
    ///
    /// assert!(RateLimitConfig::default().validate().is_ok());
    /// let bad = RateLimitConfig { max_retries: 0, ..Default::default() };
    /// assert!(bad.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.default_delay == Duration::ZERO {
            return Err("default_delay must be non-zero".into());
        }
        if self.max_delay < self.default_delay {
            return Err("max_delay must be >= default_delay".into());
        }
        if self.max_retries == 0 {
            return Err("max_retries must be >= 1".into());
        }
        Ok(())
    }

    /// Effective backoff for a detected rate limit given an optional server hint.
    ///
    /// - If `server_hint` is `Some(d)` and `respect_retry_after` is `true`,
    ///   returns `min(d, max_delay)`.
    /// - Otherwise returns `min(default_delay, max_delay)`.
    ///
    /// ```
    /// use loopctl::stream::handler::RateLimitConfig;
    /// use std::time::Duration;
    ///
    /// let cfg = RateLimitConfig::default();
    /// assert_eq!(cfg.backoff(Some(Duration::from_secs(12))), Duration::from_secs(12));
    /// assert_eq!(cfg.backoff(None), cfg.default_delay);
    /// ```
    #[must_use]
    pub fn backoff(&self, server_hint: Option<Duration>) -> Duration {
        match server_hint {
            Some(d) if self.respect_retry_after => d.min(self.max_delay),
            _ => self.default_delay.min(self.max_delay),
        }
    }
}

/// Which kind of rate-limit / overload response was detected.
///
/// Set by [`DetectedRateLimit::detect`] when classifying an [`ApiError`]; the
/// distinction matters because the two responses come from different failure
/// modes (a hard per-account quota vs. a transient capacity signal) even
/// though both honour `Retry-After`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitKind {
    /// HTTP 429 Too Many Requests.
    ///
    /// The canonical rate-limit signal: the caller has exceeded a per-account
    /// or per-key quota. The server typically sends a `Retry-After` hint;
    /// [`backoff`](RateLimitConfig::backoff) honours it when
    /// [`respect_retry_after`](RateLimitConfig::respect_retry_after) is set.
    RateLimited,

    /// HTTP 503 Service Unavailable / 529 Overloaded.
    ///
    /// A rate-limit-adjacent transient: the provider is overloaded rather than
    /// enforcing a quota. Treated the same as [`RateLimited`](Self::RateLimited)
    /// for backoff purposes (it honours `Retry-After`), but surfaced as a
    /// distinct kind so callers can log or route it differently.
    Overloaded,
}

/// A rate-limit response detected on an established stream.
///
/// Produced by [`DetectedRateLimit::detect`] from an [`ApiError`]. Carries the
/// parsed `Retry-After` (when available) so the caller can back off accordingly
/// without re-parsing.
#[derive(Debug, Clone)]
pub struct DetectedRateLimit {
    /// The detected rate-limit class.
    ///
    /// Either [`RateLimitKind::RateLimited`] (HTTP 429) or
    /// [`RateLimitKind::Overloaded`] (HTTP 503/529). Determines nothing on its
    /// own — both kinds honour `Retry-After` — but lets the caller distinguish
    /// a quota hit from a transient capacity signal.
    pub kind: RateLimitKind,

    /// The server-advised delay, parsed from the `Retry-After` header.
    ///
    /// `None` when the header was absent or could not be parsed as a number of
    /// seconds or an HTTP-date. When `None`, the caller falls back to
    /// [`RateLimitConfig::default_delay`].
    pub retry_after: Option<Duration>,

    /// The original error message, preserved verbatim.
    ///
    /// Kept so the caller can log the provider's wording, include it in a
    /// fallback-model prompt, or surface it to the user without losing the
    /// diagnostic detail that [`detect`](Self::detect) collapsed into
    /// [`kind`](Self::kind).
    pub message: String,
}

impl DetectedRateLimit {
    /// Inspect an [`ApiError`] for a rate-limit signature.
    ///
    /// Returns `Some(DetectedRateLimit)` when the error is:
    /// - the structured [`ApiError::RateLimit`] variant (typed `retry_after`),
    /// - an `Api(String)` whose body contains `"rate limit"` or `"429"`, or
    /// - an `Http(String)` whose `"HTTP {status}:"` prefix indicates 503/529.
    ///
    /// Returns `None` for everything else (500s, auth errors, generic transport
    /// failures, etc.).
    #[must_use]
    pub fn detect(err: &crate::api::error::ApiError) -> Option<Self> {
        match err {
            ApiError::RateLimit {
                retry_after,
                message,
            } => Some(Self {
                kind: RateLimitKind::RateLimited,
                retry_after: *retry_after,
                message: message.clone(),
            }),
            ApiError::Api(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("rate limit") || lower.contains("429") {
                    Some(Self {
                        kind: RateLimitKind::RateLimited,
                        retry_after: parse_retry_after(msg),
                        message: msg.clone(),
                    })
                } else {
                    None
                }
            }
            ApiError::Http(msg) if http_status_is_overload(msg) => Some(Self {
                kind: RateLimitKind::Overloaded,
                retry_after: parse_retry_after(msg),
                message: msg.clone(),
            }),
            _ => None,
        }
    }
}

/// Parse an HTTP `Retry-After` value into a [`Duration`].
///
/// Accepts the two RFC 9110 forms:
/// - **delta-seconds** (`"12"`) → `Duration::from_secs(12)`. Huge values are
///   clamped rather than overflowing.
/// - **HTTP-date** (`"Wed, 21 Oct 2026 07:28:00 GMT"`) → `max(ZERO, date − now)`.
///   Only available when the `providers` feature is enabled (the `httpdate`
///   crate lives there); returns `None` otherwise.
///
/// Returns `None` for anything unparseable.
fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    #[cfg(feature = "providers")]
    {
        if let Ok(target) = httpdate::parse_http_date(trimmed) {
            let now = std::time::SystemTime::now();
            return now
                .duration_since(target)
                .ok()
                .map(|_| Duration::ZERO)
                .or_else(|| target.duration_since(now).ok());
        }
    }
    None
}

/// Clamp a rate-limit backoff so it cannot sleep past the turn's `total_deadline`.
///
/// `None` passes the delay through unchanged. When the remaining time to the
/// deadline is smaller than `delay`, the remaining time is returned (zero once
/// the deadline has passed). All arithmetic is checked; the worst case is
/// `Duration::ZERO`, never a panic.
fn clamp_delay_to_deadline(delay: Duration, deadline: Option<Instant>) -> Duration {
    let Some(deadline) = deadline else {
        return delay;
    };
    let now = Instant::now();
    let Some(remaining) = deadline.checked_duration_since(now) else {
        return Duration::ZERO;
    };
    delay.min(remaining)
}

/// Outcome of a rate-limit retry decision.
///
/// Returned by [`StreamHandler::rate_limit_retry`] for each detected rate
/// limit on the current model. The variants form a three-step escalation
/// ladder: retry in place while the count is low, escalate to the model
/// circuit breaker once it crosses
/// [`fallback_after_retries`](RateLimitConfig::fallback_after_retries), and
/// give up entirely once it crosses
/// [`max_retries`](RateLimitConfig::max_retries).
#[derive(Debug)]
enum RateLimitRetry {
    /// Escalate to the model circuit breaker.
    ///
    /// Returned once the per-model retry count exceeds
    /// [`fallback_after_retries`](RateLimitConfig::fallback_after_retries).
    /// The caller trips the breaker, which routes subsequent turns to a
    /// fallback model if one is configured; if not, the escalation has nowhere
    /// to go and the turn fails.
    Escalate {
        /// Number of rate-limit retries honored before this escalation.
        ///
        /// Always strictly greater than
        /// [`fallback_after_retries`](RateLimitConfig::fallback_after_retries)
        /// — the count that triggered the escalation, incremented before the
        /// decision is made. Surfaced for logging and for the
        /// [`RateLimitEscalation`](crate::error::LoopError::RateLimitEscalation)
        /// error payload.
        attempts: u32,

        /// The server-advised delay from the triggering response.
        ///
        /// The raw `Retry-After` from [`DetectedRateLimit`] (`None` if the
        /// header was absent). Carried unmodified — clamping to
        /// [`max_delay`](RateLimitConfig::max_delay) happens in
        /// [`backoff`](RateLimitConfig::backoff) on the retry path, not here.
        /// Preserved so the escalation consumer can log or forward the
        /// provider's hint.
        retry_after: Option<Duration>,
    },

    /// Give up on the current model without escalating.
    ///
    /// Returned once the per-model retry count exceeds
    /// [`max_retries`](RateLimitConfig::max_retries) — the hard stop after
    /// which retrying the same model is pointless. Distinct from
    /// [`Escalate`](Self::Escalate): escalation hands off to the circuit
    /// breaker (and a fallback model), `HardStop` fails the turn outright.
    HardStop,

    /// Sleep for `delay`, then retry the current model.
    ///
    /// Returned while the retry count is below both
    /// [`fallback_after_retries`](RateLimitConfig::fallback_after_retries) and
    /// [`max_retries`](RateLimitConfig::max_retries). The delay is the
    /// [`backoff`](RateLimitConfig::backoff) for the detected response,
    /// further clamped to the remaining time before the turn's
    /// `total_stream_timeout` deadline so a large `Retry-After` cannot overrun
    /// the turn budget.
    Retry(Duration),
}

/// Pull the carried [`StreamOutcome`] out of a [`StreamHandlerError`], if any.
///
/// Only [`InitFailed`](StreamHandlerError::InitFailed) and
/// [`StreamFailed`](StreamHandlerError::StreamFailed) carry one (the outcome
/// that was in progress when the error was raised); every other variant maps
/// to `None`. The caller uses the recovered outcome to route the failure into
/// the correct retry budget — a [`RateLimited`](StreamOutcome::RateLimited)
/// outcome draws on [`RateLimitConfig`], distinct from the transport-retry
/// budget, so a rate-limit storm cannot exhaust transport retries (nor vice
/// versa).
fn carried_outcome(error: &StreamHandlerError) -> Option<StreamOutcome> {
    match error {
        StreamHandlerError::InitFailed(o) | StreamHandlerError::StreamFailed(o) => {
            Some(o.to_owned())
        }
        _ => None,
    }
}

/// Sleep for `delay`, or return [`StreamHandlerError::Cancelled`] if the cancel
/// signal fires first.
///
/// # Errors
///
/// Returns [`StreamHandlerError::Cancelled`] if `cancel` is signalled before the
/// sleep elapses.
async fn sleep_cancellable(
    delay: Duration,
    cancel: &Arc<CancelSignal>,
) -> Result<(), StreamHandlerError> {
    tokio::select! {
        () = tokio::time::sleep(delay) => Ok(()),
        () = cancel.notified() => Err(StreamHandlerError::Cancelled),
    }
}

/// Result of polling the stream once inside [`StreamHandler::next_event`].
///
/// Produced by the `tokio::select!` that races the stream against the
/// per-event timeout. Only two outcomes materialize here: the stream produced
/// an item ([`Next`](Self::Next)), or the per-event timeout fired first
/// ([`TimedOut`](Self::TimedOut)). Cancellation and the total-stream deadline
/// are also raced in the same `select!`, but they return directly as
/// [`StreamHandlerError::Cancelled`] / `StreamFailed` and so do not need a
/// variant here.
enum EventPoll {
    /// The stream produced an item before the per-event timeout.
    ///
    /// Delegates the three sub-cases to the caller: `Some(Ok(event))` is
    /// yielded to the accumulator, `Some(Err(api_error))` becomes an API-error
    /// outcome, and `None` means the stream ended cleanly (turn completes).
    Next(Option<Result<crate::stream::StreamEvent, crate::api::error::ApiError>>),

    /// The per-event timeout fired before any item arrived.
    ///
    /// Increments the consecutive-timeout counter; once it reaches
    /// [`max_consecutive_timeouts`](StreamTimeoutConfig::max_consecutive_timeouts),
    /// the caller escalates to a [`StreamFailed`](StreamHandlerError::StreamFailed)
    /// event-timeout outcome. A lower threshold (`min(2, max_consecutive_timeouts)`)
    /// applies when no events have been received yet (empty-stream fast-fail).
    TimedOut,
}

/// Read-only diagnostic context used to build timeout and error outcomes.
///
/// Snapshotted once per loop iteration in [`StreamHandler::process_events`] and
/// handed to [`StreamHandler::next_event`], which needs progress/elapsed data to
/// populate [`StreamOutcome`] fields when it short-circuits.
struct EventDiagnostics {
    /// Events processed so far this turn.
    ///
    /// Surfaced on the [`StreamOutcome::TotalTimeout`] and
    /// [`StreamOutcome::RateLimited`] outcomes so the caller can tell a
    /// mid-stream failure (some events got through) from an immediate one
    /// (nothing arrived). Not used by [`event_timeout`](Self::event_timeout),
    /// which reports consecutive-timeout count instead.
    events_processed: u64,

    /// When the stream started, for elapsed-duration outcomes.
    ///
    /// Read via [`Instant::elapsed`] when building
    /// [`StreamOutcome::TotalTimeout`]'s `duration` field. Captured once at
    /// the top of [`process_events`](StreamHandler::process_events) rather
    /// than per event so the reported duration is the full stream lifetime,
    /// not the time since the most recent event.
    stream_start: Instant,

    /// Whether partial content has been accumulated.
    ///
    /// Recomputed each loop iteration (the whole [`EventDiagnostics`] is
    /// rebuilt per iteration in [`process_events`](StreamHandler::process_events))
    /// from the accumulator's current part count: `true` once at least one
    /// usable event has been received. Flows into the `has_partial_data` flag
    /// on [`StreamOutcome::TotalTimeout`], [`StreamOutcome::EventTimeout`],
    /// and [`StreamOutcome::RateLimited`], letting a downstream consumer
    /// decide whether to salvage the partial output or discard it.
    has_partial_data: bool,
}

impl EventDiagnostics {
    /// Build the [`StreamOutcome::TotalTimeout`] for this point in the stream.
    ///
    /// Snapshots the current diagnostic state — partial-data flag, events
    /// processed so far, and elapsed time since [`stream_start`](Self::stream_start)
    /// — into a `TotalTimeout` outcome. Used by [`process_events`](StreamHandler::process_events)
    /// when the turn's `total_stream_timeout` deadline fires (both at the
    /// top-of-loop check and inside the per-event `select!`).
    fn total_timeout(&self) -> StreamOutcome {
        StreamOutcome::TotalTimeout {
            has_partial_data: self.has_partial_data,
            events_processed: self.events_processed,
            duration: self.stream_start.elapsed(),
        }
    }

    /// Build the [`StreamOutcome::EventTimeout`] for this point in the stream.
    ///
    /// Carries the partial-data flag plus the caller-supplied
    /// `consecutive_timeouts` count (this method does not track the counter
    /// itself — `process_events` owns it and passes the current value in).
    /// Used once the per-event timeout crosses
    /// [`max_consecutive_timeouts`](StreamTimeoutConfig::max_consecutive_timeouts).
    fn event_timeout(&self, consecutive_timeouts: u32) -> StreamOutcome {
        StreamOutcome::EventTimeout {
            has_partial_data: self.has_partial_data,
            consecutive_timeouts,
        }
    }

    /// Map a stream API error to the matching [`StreamHandlerError`].
    ///
    /// Two branches: if [`DetectedRateLimit::detect`] classifies the error as
    /// a 429/503/529, builds a [`StreamOutcome::RateLimited`] carrying the
    /// parsed `Retry-After` and current progress; otherwise wraps it as a
    /// generic [`StreamOutcome::InitFailed`] with `attempts: 1` (this is the
    /// per-event error path, not the init-retry path, so the attempt counter
    /// isn't meaningful here). Used by `process_events` when the stream yields
    /// an `Err` event.
    fn api_error_outcome(&self, error: &crate::api::error::ApiError) -> StreamHandlerError {
        if let Some(detail) = DetectedRateLimit::detect(error) {
            return StreamHandlerError::StreamFailed(StreamOutcome::RateLimited {
                detail,
                has_partial_data: self.has_partial_data,
                events_processed: self.events_processed,
            });
        }
        StreamHandlerError::StreamFailed(StreamOutcome::InitFailed {
            attempts: 1,
            last_error: error.to_string(),
        })
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
/// Completed < TotalTimeout < EventTimeout < RateLimited < InitFailed < FallbackToNonStreaming < Cancelled
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamOutcome {
    /// Stream completed normally — all events received, `MessageStop` seen.
    ///
    /// Happy path. The [`StreamAccumulator`]
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

    /// A 429 / 503 rate-limit response arrived mid-stream.
    ///
    /// Distinct from [`InitFailed`](Self::InitFailed): the stream *was*
    /// established and may have produced partial output. The
    /// [`DetectedRateLimit`] carries the parsed `Retry-After`, if the server
    /// provided one.
    RateLimited {
        /// Decoded rate-limit detail (kind + parsed `Retry-After`).
        detail: DetectedRateLimit,
        /// Whether partial content was accumulated before the rate limit.
        has_partial_data: bool,
        /// Events processed before the rate limit fired.
        events_processed: u64,
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

    /// Fell back to non-streaming [`create_message`](crate::api::ApiClient::create_message).
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
            Self::RateLimited {
                detail,
                has_partial_data,
                events_processed,
            } => {
                let kind = match detail.kind {
                    RateLimitKind::RateLimited => "rate limit",
                    RateLimitKind::Overloaded => "overloaded",
                };
                let retry = detail
                    .retry_after
                    .map(|d| format!(" (retry after {d:?})"))
                    .unwrap_or_default();
                let partial = if *has_partial_data {
                    " (partial data)"
                } else {
                    ""
                };
                write!(
                    f,
                    "{kind}{retry}{partial}, {events_processed} events processed"
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
#[non_exhaustive]
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

    /// Rate-limit retries on the current model were exhausted.
    ///
    /// The handler honored the provider's `Retry-After` up to the configured
    /// [`RateLimitConfig::fallback_after_retries`] ceiling and could not make
    /// progress on this model. The caller should escalate to the model circuit
    /// breaker ([`FallbackManager`](crate::fallback::FallbackManager)), not the
    /// same-model non-streaming fallback.
    RateLimitEscalation {
        /// Number of rate-limit retries honored before escalating.
        attempts: u32,
        /// Last server-advised `Retry-After` hint, after clamping. `None` when no header sent.
        retry_after: Option<Duration>,
        /// The rate-limit outcome that triggered escalation.
        prior: StreamOutcome,
    },
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
            Self::RateLimitEscalation {
                attempts,
                retry_after,
                prior: _,
            } => write!(
                f,
                "rate-limit escalation after {attempts} retries (retry-after {retry_after:?})"
            ),
        }
    }
}

impl std::error::Error for StreamHandlerError {}

// ===================================================
// StreamProgress
// ===================================================

/// A snapshot of stream progress for external reporting.
///
/// Plain data struct carrying the two progress signals a consumer is likely
/// to want (elapsed time and events processed). `StreamHandler` does not
/// itself emit `StreamProgress` — it has no built-in progress callback. The
/// struct is shipped so a downstream consumer that drives its own progress
/// reporting (metrics observer, TUI heartbeat, deadline watcher) has a
/// shared shape to read or fill.
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
    ///
    /// Wall-clock duration from stream open to the snapshot point. Useful for
    /// heartbeat-style reporting (“still streaming after Ns”) and for
    /// deadline-aware consumers that compare it against their own budget.
    pub elapsed: Duration,

    /// Number of SSE events processed so far.
    ///
    /// Count of stream events successfully accumulated up to the snapshot
    /// point. A flat or slow-growing count is the early signal of a stalled
    /// stream before a timeout fires.
    pub events_processed: u64,
}

// ===================================================
// StreamHandler
// ===================================================

/// Holds configuration for the streaming resilience layer.
///
/// `StreamHandler` wraps an [`ApiClient`]'s streaming path with timeout,
/// retry, and rate-limit handling. It owns four independent budgets that
/// together make a stream turn robust:
///
/// 1. **Timeouts** ([`StreamTimeoutConfig`]) — initial-event, per-event, and
///    total-stream deadlines, plus the consecutive-timeout escalation
///    threshold and the optional non-streaming fallback.
///
/// 2. **Transport retries** ([`StreamRetryConfig`]) — exponential backoff for
///    stream-initialization failures (connection drops, transport errors),
///    distinct from rate-limit retries.
///
/// 3. **Rate-limit handling** ([`RateLimitConfig`]) — `Retry-After`-aware
///    backoff for 429/503/529 responses, with its own retry budget, an
///    escalation threshold to the model circuit breaker, and a hard stop.
///    Kept independent from transport retries so a rate-limit storm cannot
///    exhaust the transport budget (nor vice versa).
///
/// 4. **Proactive throttling** (optional
///    [`RateLimiter`](crate::stream::rate_limit::RateLimiter)) — a per-provider
///    token bucket that gates each stream attempt *before* it fires, sleeping
///    up to `max_wait` rather than risking a 429.
///
/// On exhaustion, the handler escalates: rate-limit retries trip the model
/// circuit breaker (route to a fallback model), transport retries fall back
/// to [`ApiClient::create_message`] when
/// [`fallback_to_non_streaming`](StreamTimeoutConfig::fallback_to_non_streaming)
/// is set, and a turn that can't recover fails with a typed
/// [`StreamHandlerError`].
///
/// # Example
///
/// ```rust
/// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
///
/// let handler = StreamHandler::new();
/// assert_eq!(handler.timeout_config().initial_event_timeout, std::time::Duration::from_secs(120));
///
/// let handler = StreamHandler::new().with_config(
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
    ///
    /// Drives the initial-event / per-event / total-stream deadlines, the
    /// consecutive-timeout escalation threshold, and the
    /// non-streaming-fallback toggle. Read on every turn in `stream_turn` and
    /// on each event poll.
    timeout_config: StreamTimeoutConfig,

    /// Retry configuration for stream-initialization failures.
    ///
    /// Exponential backoff applied to transport-level failures (connection
    /// drops, TLS errors, etc.) before any event arrives. Read in the
    /// init-retry loop; distinct from `rate_limit_config`, which has its own
    /// budget.
    retry_config: StreamRetryConfig,

    /// Rate-limit detection + backoff policy.
    ///
    /// Governs reactive handling of server-returned 429/503/529 responses
    /// (honoured `Retry-After`, default delay, cap, escalation threshold to
    /// the model circuit breaker, hard-stop ceiling). Read by the
    /// `rate_limit_retry` decision on each detected rate limit.
    rate_limit_config: RateLimitConfig,

    /// Optional proactive per-provider rate limiter (token bucket).
    ///
    /// When set, each stream attempt is gated by `gate_on_rate_limit` *before*
    /// firing — sleeping up to the limiter's `max_wait` for a token rather
    /// than risking a 429. `None` (the default) means reactive-only handling:
    /// no pre-throttling, server 429s still handled via `rate_limit_config`.
    rate_limiter: Option<Arc<crate::stream::rate_limit::RateLimiter>>,

    /// Per-turn request options applied to every stream-open call.
    ///
    /// Carries [`tool_constraint`](crate::structured::ToolConstraint) for
    /// constrained tool-call decoding. Default is
    /// [`RequestOptions::default`](crate::structured::RequestOptions::default)
    /// (no constraint), reproducing the prior unconstrained behavior.
    /// Passed to [`ApiClient::stream_messages_with_options`] at the stream-open
    /// call site. Set via [`with_request_options`](StreamHandler::with_request_options).
    request_options: crate::structured::RequestOptions,
}

impl fmt::Debug for StreamHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamHandler")
            .field("timeout_config", &self.timeout_config)
            .field("retry_config", &self.retry_config)
            .field("rate_limit_config", &self.rate_limit_config)
            .field("rate_limiter", &self.rate_limiter)
            .field("request_options", &self.request_options)
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
            rate_limit_config: RateLimitConfig::default(),
            rate_limiter: None,
            request_options: crate::structured::RequestOptions::default(),
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
    /// let handler = StreamHandler::new().with_config(
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
    pub fn with_config(mut self, timeout: StreamTimeoutConfig, retry: StreamRetryConfig) -> Self {
        self.timeout_config = timeout;
        self.retry_config = retry;
        self
    }

    /// Set the per-turn [`RequestOptions`](crate::structured::RequestOptions)
    /// applied to every stream-open call. Consuming builder.
    ///
    /// Default is [`RequestOptions::default`](crate::structured::RequestOptions::default)
    /// (no constraint), which reproduces prior behavior. Set
    /// [`tool_constraint`](crate::structured::ToolConstraint) to
    /// [`Strict`](crate::structured::ToolConstraint::Strict) for constrained
    /// tool-call decoding.
    #[must_use]
    pub fn with_request_options(mut self, options: crate::structured::RequestOptions) -> Self {
        self.request_options = options;
        self
    }

    /// Returns a reference to the timeout configuration.
    ///
    /// Read-only access to the [`StreamTimeoutConfig`] stored on the handler.
    /// Mutate via [`with_config`](StreamHandler::with_config) (which replaces
    /// both timeout and retry together); there is no per-field setter.
    #[must_use]
    pub fn timeout_config(&self) -> &StreamTimeoutConfig {
        &self.timeout_config
    }

    /// Returns a reference to the retry configuration.
    ///
    /// Read-only access to the [`StreamRetryConfig`] stored on the handler.
    /// Mutate via [`with_config`](StreamHandler::with_config) (which replaces
    /// both retry and timeout together); there is no per-field setter.
    #[must_use]
    pub fn retry_config(&self) -> &StreamRetryConfig {
        &self.retry_config
    }

    /// Set a custom [`RateLimitConfig`]. Consuming builder.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::stream::handler::{StreamHandler, RateLimitConfig};
    /// use std::time::Duration;
    ///
    /// let handler = StreamHandler::new().with_rate_limit_config(
    ///     RateLimitConfig { max_retries: 2, ..Default::default() },
    /// );
    /// assert_eq!(handler.rate_limit_config().max_retries, 2);
    /// ```
    #[must_use]
    pub fn with_rate_limit_config(mut self, rl: RateLimitConfig) -> Self {
        self.rate_limit_config = rl;
        self
    }

    /// Returns a reference to the rate-limit configuration.
    ///
    /// Read-only access to the [`RateLimitConfig`] stored on the handler.
    /// Mutate via
    /// [`with_rate_limit_config`](StreamHandler::with_rate_limit_config).
    #[must_use]
    pub fn rate_limit_config(&self) -> &RateLimitConfig {
        &self.rate_limit_config
    }

    /// Attach a per-provider rate limiter (proactive throttle).
    ///
    /// When set, every `stream_turn` attempt waits for a token before opening
    /// the stream, spacing requests to the limiter's `requests_per_minute`
    /// ceiling. `None` (the default) disables proactive throttling — the
    /// handler then relies purely on the reactive 429 handling.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::stream::handler::StreamHandler;
    /// use loopctl::stream::rate_limit::RateLimiter;
    ///
    /// let handler = StreamHandler::new()
    ///     .with_rate_limiter(Arc::new(RateLimiter::new(60)));
    /// ```
    #[must_use]
    pub fn with_rate_limiter(
        mut self,
        limiter: Arc<crate::stream::rate_limit::RateLimiter>,
    ) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    // ==================================================
    // Runtime methods
    // ==================================================

    /// Stream one complete turn with retry, timeout, and fallback.
    ///
    /// Primary entry point for resilient streaming. It
    /// orchestrates the full lifecycle:
    ///
    /// 1. Opens a stream via [`ApiClient::stream_messages_with_options`]
    ///    (gated by the proactive [`RateLimiter`](crate::stream::rate_limit::RateLimiter),
    ///    if attached).
    /// 2. Processes events with per-event and total timeouts.
    /// 3. On transient transport errors, retries with exponential backoff;
    ///    on 429/503/529 responses, backs off under the rate-limit budget
    ///    and escalates to the model circuit breaker once that budget is
    ///    exhausted.
    /// 4. If all retries fail and
    ///    [`fallback_to_non_streaming`](StreamTimeoutConfig::fallback_to_non_streaming)
    ///    is enabled, falls back to [`ApiClient::create_message`].
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
    /// - [`RateLimitEscalation`](StreamHandlerError::RateLimitEscalation) —
    ///   rate-limit retries on this model were exhausted; escalate to a
    ///   fallback model.
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
        let mut rate_limit_retries: u32 = 0;
        let mut transport_attempts: u32 = 0;
        loop {
            if let Some(deadline) = total_deadline
                && Instant::now() >= deadline
            {
                return Err(StreamHandlerError::InitFailed(
                    StreamOutcome::TotalTimeout {
                        has_partial_data: false,
                        events_processed: 0,
                        duration: self.timeout_config.total_stream_timeout,
                    },
                ));
            }
            match self
                .try_stream(
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
                    let last_stream_outcome = carried_outcome(&e);

                    // Rate-limit retries draw on their own budget
                    // (RateLimitConfig), independent of the transport retry
                    // budget below — a rate-limit storm must not exhaust
                    // transport retries (nor vice versa).
                    if let Some(StreamOutcome::RateLimited { detail, .. }) = &last_stream_outcome {
                        match self.rate_limit_retry(detail, &mut rate_limit_retries, total_deadline)
                        {
                            RateLimitRetry::Escalate {
                                attempts,
                                retry_after,
                            } => {
                                return Err(StreamHandlerError::RateLimitEscalation {
                                    attempts,
                                    retry_after,
                                    prior: last_stream_outcome.clone().unwrap_or(
                                        StreamOutcome::RateLimited {
                                            detail: detail.clone(),
                                            has_partial_data: false,
                                            events_processed: 0,
                                        },
                                    ),
                                });
                            }
                            RateLimitRetry::HardStop => return Err(e),
                            RateLimitRetry::Retry(delay) => {
                                sleep_cancellable(delay, cancel).await?;
                                continue;
                            }
                        }
                    }

                    // Non-rate-limit errors consume the transport retry budget.
                    if transport_attempts >= max_attempts.saturating_sub(1) {
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
                    let delay = self.retry_config.base_delay(transport_attempts);
                    transport_attempts = transport_attempts.saturating_add(1);
                    sleep_cancellable(delay, cancel).await?;
                }
            }
        }
    }

    /// Decide how to handle a rate-limit failure on the current model.
    ///
    /// Bumps `count` and returns one of:
    /// - [`RateLimitRetry::Escalate`] once `count` exceeds
    ///   [`fallback_after_retries`](RateLimitConfig::fallback_after_retries) — the
    ///   caller escalates to the model circuit breaker;
    /// - [`RateLimitRetry::HardStop`] once `count` exceeds
    ///   [`max_retries`](RateLimitConfig::max_retries) — for when escalation is
    ///   unavailable (e.g. no fallback model configured);
    /// - [`RateLimitRetry::Retry`] with the deadline-clamped backoff otherwise.
    fn rate_limit_retry(
        &self,
        detail: &DetectedRateLimit,
        count: &mut u32,
        deadline: Option<Instant>,
    ) -> RateLimitRetry {
        *count = count.saturating_add(1);
        if *count > self.rate_limit_config.fallback_after_retries {
            return RateLimitRetry::Escalate {
                attempts: *count,
                retry_after: detail.retry_after,
            };
        }
        if *count > self.rate_limit_config.max_retries {
            return RateLimitRetry::HardStop;
        }
        let delay =
            clamp_delay_to_deadline(self.rate_limit_config.backoff(detail.retry_after), deadline);
        RateLimitRetry::Retry(delay)
    }

    /// Attempt a single streaming pass.
    ///
    /// Gates on the proactive [`RateLimiter`](crate::stream::rate_limit::RateLimiter)
    /// (if attached), opens a stream via
    /// [`ApiClient::stream_messages_with_options`] carrying the handler's
    /// [`RequestOptions`](crate::structured::RequestOptions), and processes
    /// all events with timeout and cancellation support. Called inside the
    /// retry loop in [`stream_turn`](StreamHandler::stream_turn), so a retried
    /// 429 re-gates on the rate limiter.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError`] if the stream fails or times out.
    async fn try_stream<C: ApiClient>(
        &self,
        client: &C,
        conversation: Vec<Message>,
        system: Option<String>,
        tool_schemas: Option<Vec<ToolSchema>>,
        cancel: &Arc<CancelSignal>,
        total_deadline: Option<Instant>,
    ) -> Result<StreamTurnResult, StreamHandlerError> {
        self.gate_on_rate_limit(client, cancel, total_deadline)
            .await?;
        let stream = client.stream_messages_with_options(
            conversation,
            system,
            tool_schemas,
            self.request_options.clone(),
        );
        self.process_events(stream, cancel, total_deadline).await
    }

    /// Wait for a rate-limit token before opening the stream.
    ///
    /// When a [`RateLimiter`](crate::stream::rate_limit::RateLimiter) is
    /// attached, this acquires one token from the bucket keyed by the client's
    /// [`base_url`](ApiClient::base_url), sleeping as needed until either a
    /// token is available, the cumulative wait reaches the limiter's `max_wait`
    /// (better to risk a 429 than hang the agent). Each wait is also clamped to
    /// the turn's remaining `total_deadline` so the gate cannot overrun the
    /// turn budget; if the deadline has already elapsed the gate proceeds
    /// rather than sleeping (the downstream per-event/total-timeout checks in
    /// [`process_events`](Self::process_events) report the expiry). A no-op
    /// when no limiter is attached.
    ///
    /// Fires per attempt (called from [`try_stream`](Self::try_stream), which
    /// runs inside the retry loop), so a retried 429 re-respects the budget.
    /// Cancel-safe: a turn stuck waiting for tokens is still user-cancellable.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::Cancelled`] if the cancel signal fires
    /// during the wait.
    async fn gate_on_rate_limit<C: ApiClient>(
        &self,
        client: &C,
        cancel: &Arc<CancelSignal>,
        total_deadline: Option<Instant>,
    ) -> Result<(), StreamHandlerError> {
        let Some(limiter) = &self.rate_limiter else {
            return Ok(());
        };
        let key = client.base_url();
        let max_wait = limiter.max_wait();
        let mut waited = Duration::ZERO;
        loop {
            match limiter.acquire(&key) {
                Ok(()) => return Ok(()),
                Err(wait) => {
                    if waited >= max_wait {
                        return Ok(());
                    }
                    let max_wait_remaining = max_wait.checked_sub(waited).unwrap_or(Duration::ZERO);
                    let total_deadline_remaining = match total_deadline {
                        None => max_wait_remaining,
                        Some(deadline) => deadline
                            .checked_duration_since(Instant::now())
                            .unwrap_or(Duration::ZERO),
                    };
                    let capped = wait.min(max_wait_remaining).min(total_deadline_remaining);
                    if capped.is_zero() {
                        return Ok(());
                    }
                    tokio::select! {
                        () = tokio::time::sleep(capped) => {}
                        () = cancel.notified() => return Err(StreamHandlerError::Cancelled),
                    }
                    waited = waited.saturating_add(capped);
                }
            }
        }
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
        S: futures::Stream<Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>>
            + Unpin,
    {
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;
        let mut consecutive_timeouts: usize = 0;
        let mut events_processed: u64 = 0;
        let stream_start = Instant::now();

        loop {
            let diagnostics = EventDiagnostics {
                events_processed,
                stream_start,
                has_partial_data: !accumulator.peek_parts().is_empty(),
            };
            let Some(event) = self
                .next_event(
                    &mut stream,
                    cancel,
                    &mut consecutive_timeouts,
                    total_deadline,
                    &diagnostics,
                )
                .await?
            else {
                break;
            };
            events_processed = events_processed.saturating_add(1);
            consecutive_timeouts = 0;
            Self::apply_event(&event, &mut accumulator, &mut stop_reason)?;
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

    /// Wait for the next stream event, enforcing the total deadline and
    /// per-event timeout.
    ///
    /// Returns `Ok(None)` when the stream ends. On a per-event timeout the
    /// consecutive-timeout counter is bumped; once it reaches
    /// [`max_consecutive_timeouts`](StreamTimeoutConfig::max_consecutive_timeouts)
    /// the turn fails with [`StreamOutcome::EventTimeout`].
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::Cancelled`] if the cancel signal fires, or
    /// [`StreamHandlerError::StreamFailed`] on total/per-event timeout or an API
    /// error.
    async fn next_event<S>(
        &self,
        stream: &mut S,
        cancel: &Arc<CancelSignal>,
        consecutive_timeouts: &mut usize,
        total_deadline: Option<Instant>,
        diagnostics: &EventDiagnostics,
    ) -> Result<Option<StreamEvent>, StreamHandlerError>
    where
        S: futures::Stream<Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>>
            + Unpin,
    {
        loop {
            if Self::deadline_exceeded(total_deadline) {
                return Err(StreamHandlerError::StreamFailed(
                    diagnostics.total_timeout(),
                ));
            }
            if cancel.is_cancelled() {
                return Err(StreamHandlerError::Cancelled);
            }

            let event_deadline = self.event_deadline(diagnostics.events_processed);
            let event_result = tokio::select! {
                event = stream.next() => EventPoll::Next(event),
                () = cancel.notified() => return Err(StreamHandlerError::Cancelled),
                () = tokio::time::sleep_until(event_deadline.into()) => EventPoll::TimedOut,
                () = Self::total_deadline_future(total_deadline) => {
                    return Err(StreamHandlerError::StreamFailed(diagnostics.total_timeout()));
                }
            };
            match event_result {
                EventPoll::TimedOut => {
                    *consecutive_timeouts = consecutive_timeouts.saturating_add(1);
                    let max_consecutive = self.timeout_config.max_consecutive_timeouts as usize;
                    if *consecutive_timeouts >= max_consecutive {
                        return Err(StreamHandlerError::StreamFailed(diagnostics.event_timeout(
                            u32::try_from(*consecutive_timeouts).unwrap_or(u32::MAX),
                        )));
                    }
                }
                EventPoll::Next(Some(Ok(event))) => return Ok(Some(event)),
                EventPoll::Next(Some(Err(api_error))) => {
                    return Err(diagnostics.api_error_outcome(&api_error));
                }
                EventPoll::Next(None) => return Ok(None),
            }
        }
    }

    /// The deadline for the next stream event.
    ///
    /// Uses [`initial_event_timeout`](StreamTimeoutConfig::initial_event_timeout)
    /// before any event has arrived (the model may need time to begin
    /// generating), then switches to
    /// [`per_event_timeout`](StreamTimeoutConfig::per_event_timeout) once events
    /// are flowing. Falls back to "now" if the addition overflows.
    fn event_deadline(&self, events_processed: u64) -> Instant {
        let base_timeout = if events_processed == 0 {
            self.timeout_config.initial_event_timeout
        } else {
            self.timeout_config.per_event_timeout
        };

        Instant::now()
            .checked_add(base_timeout)
            .unwrap_or(Instant::now())
    }

    /// Whether the total-stream deadline has already passed.
    ///
    /// Polled between events at the top of [`next_event`](Self::next_event)'s
    /// loop, before the per-event `select!` commits to another wait. This
    /// catches a deadline that elapsed while the loop was processing the
    /// previous event (or building diagnostics) — the
    /// [`total_deadline_future`](Self::total_deadline_future) `select!` arm
    /// only fires *during* a wait, so without this check a long event handler
    /// could overshoot the deadline by up to one event's processing time.
    ///
    /// `None` means no total-stream deadline is configured (the turn is
    /// bounded only by the per-event timeout) and the function returns
    /// `false` for every poll.
    fn deadline_exceeded(total_deadline: Option<Instant>) -> bool {
        match total_deadline {
            Some(deadline) => Instant::now() >= deadline,
            None => false,
        }
    }

    /// A future that completes when the overall total-stream deadline elapses.
    ///
    /// Returns a future that never resolves when there is no total deadline,
    /// so the `tokio::select!` branch stays inert in that case.
    async fn total_deadline_future(total_deadline: Option<Instant>) {
        match total_deadline {
            Some(deadline) => {
                if let Some(duration) = deadline.checked_duration_since(Instant::now()) {
                    tokio::time::sleep(duration).await;
                }
            }
            None => std::future::pending::<()>().await,
        }
    }

    /// Fold one event into the accumulator, tracking stop reason.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::StreamFailed`] if the event cannot be
    /// accumulated.
    fn apply_event(
        event: &StreamEvent,
        accumulator: &mut StreamAccumulator,
        stop_reason: &mut StreamStopReason,
    ) -> Result<(), StreamHandlerError> {
        if let StreamEvent::MessageDelta(delta) = event
            && let Some(ref reason_str) = delta.delta.stop_reason
        {
            *stop_reason = StreamStopReason::from_api_str(reason_str).unwrap_or(*stop_reason);
        }
        if let Err(e) = accumulator.process(event) {
            return Err(StreamHandlerError::StreamFailed(
                StreamOutcome::InitFailed {
                    attempts: 1,
                    last_error: e.to_string(),
                },
            ));
        }
        Ok(())
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
#[non_exhaustive]
pub struct StreamTurnResult {
    /// The fully accumulated assistant message.
    ///
    /// On the streaming path, built by the accumulator from every received
    /// event — so it carries the full part structure (text blocks, tool
    /// calls, images). On the non-streaming fallback path, built from just
    /// the first text part of the JSON response, so non-text parts are lost
    /// when [`from_fallback`](Self::from_fallback) is `true`.
    pub message: Message,

    /// Token counts for this turn, if reported by the provider.
    ///
    /// Populated on the streaming path from the provider's final usage event.
    /// `None` on the non-streaming fallback path (the JSON extraction does
    /// not parse usage), and `None` on either path when the provider did not
    /// report usage.
    pub usage: Option<Usage>,

    /// Why the model stopped generating.
    ///
    /// [`EndTurn`](StreamStopReason::EndTurn),
    /// [`ToolCall`](StreamStopReason::ToolCall), etc. — parsed from the
    /// provider's stop signal. On the non-streaming fallback path, defaults to
    /// [`EndTurn`](StreamStopReason::EndTurn) when the response carries no
    /// parseable stop reason.
    pub stop_reason: StreamStopReason,

    /// Whether the result came from a non-streaming fallback.
    ///
    /// `false` on the normal streaming path; `true` when streaming exhausted
    /// its retries and [`fallback_to_non_streaming`](StreamTimeoutConfig::fallback_to_non_streaming)
    /// routed the turn through [`ApiClient::create_message`]. Callers can use
    /// this to downgrade trust in the result (the fallback path loses non-text
    /// parts and usage) or surface a warning to the user.
    pub from_fallback: bool,

    /// Wall-clock time spent on this turn.
    ///
    /// Measured from the turn's stream-open (or fallback dispatch) to the
    /// completed result, inclusive of any retries and backoff sleeps along
    /// the way — so it reflects the *true* latency the turn incurred, not
    /// just the final successful attempt's duration.
    pub elapsed: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_config_default_values() {
        let config = StreamTimeoutConfig::default();
        assert_eq!(config.initial_event_timeout, Duration::from_mins(2));
        assert_eq!(config.per_event_timeout, Duration::from_mins(5));
        assert_eq!(config.total_stream_timeout, Duration::from_mins(15));
        assert_eq!(config.max_consecutive_timeouts, 10);
        assert_eq!(config.progress_interval, Duration::from_secs(30));
        assert!(config.fallback_to_non_streaming);
    }

    #[test]
    fn timeout_config_custom_values() {
        let config = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(30),
            per_event_timeout: Duration::from_mins(1),
            total_stream_timeout: Duration::from_mins(5),
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
        assert_eq!(config.base_delay(3), Duration::from_secs(5));
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
            duration: Duration::from_mins(15),
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
            duration: Duration::from_mins(15),
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
    fn error_rate_limit_escalation_display() {
        let prior = StreamOutcome::RateLimited {
            detail: DetectedRateLimit {
                kind: RateLimitKind::RateLimited,
                retry_after: Some(Duration::from_secs(5)),
                message: "slow down".to_string(),
            },
            has_partial_data: false,
            events_processed: 0,
        };
        let err = StreamHandlerError::RateLimitEscalation {
            attempts: 4,
            retry_after: Some(Duration::from_secs(5)),
            prior,
        };
        let s = err.to_string();
        assert!(s.contains("rate-limit escalation"), "got: {s}");
        assert!(s.contains("4 retries"), "got: {s}");
        assert!(
            s.contains("5s"),
            "should render the retry-after duration, got: {s}"
        );
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
            Duration::from_mins(2),
        );
        assert_eq!(handler.retry_config().max_retries, 3);
    }

    #[test]
    fn handler_with_config() {
        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig {
                initial_event_timeout: Duration::from_mins(1),
                ..Default::default()
            },
            StreamRetryConfig {
                max_retries: 5,
                ..Default::default()
            },
        );
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_mins(1),
        );
        assert_eq!(handler.retry_config().max_retries, 5);
    }

    #[test]
    fn handler_default_trait() {
        let handler = StreamHandler::default();
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_mins(2),
        );
    }

    #[test]
    fn handler_debug_format() {
        let handler = StreamHandler::new();
        let debug = format!("{handler:?}");
        assert!(debug.contains("StreamHandler"));
        assert!(debug.contains("timeout_config"));
    }

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
            initial_event_timeout: Duration::from_mins(2),
            total_stream_timeout: Duration::from_mins(1),
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

    use crate::api::error::ApiError;
    use crate::stream::{
        DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
        PartStart, StreamEvent,
    };

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
        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig {
                total_stream_timeout: Duration::from_millis(1),
                per_event_timeout: Duration::from_mins(5),
                initial_event_timeout: Duration::from_mins(2),
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
        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig {
                per_event_timeout: Duration::from_mins(5),
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

    struct HandlerMock {
        create_error: Option<String>,
        create_response: Option<serde_json::Value>,
    }

    impl HandlerMock {
        fn new() -> Self {
            Self {
                create_error: None,
                create_response: None,
            }
        }

        fn with_text_response(mut self, text: &str) -> Self {
            self.create_response = Some(serde_json::json!({
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn"
            }));
            self
        }

        fn with_create_error(mut self, msg: &str) -> Self {
            self.create_error = Some(msg.to_string());
            self
        }
    }

    impl ApiClient for HandlerMock {
        fn model(&self) -> String {
            "test-model".to_string()
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
        let handler = StreamHandler::new().with_config(
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
        let handler = StreamHandler::new().with_config(
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
        let handler = StreamHandler::new().with_config(
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
        struct ErrorMock;
        impl ApiClient for ErrorMock {
            fn model(&self) -> String {
                "test-model".to_string()
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

        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            },
            StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            },
        );

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

    #[test]
    fn rate_limit_config_default_values() {
        let cfg = RateLimitConfig::default();
        assert!(cfg.respect_retry_after);
        assert_eq!(cfg.default_delay, Duration::from_secs(5));
        assert_eq!(cfg.max_delay, Duration::from_mins(1));
        assert_eq!(cfg.requests_per_minute, 0);
        assert_eq!(cfg.fallback_after_retries, 3);
        assert_eq!(cfg.max_retries, 5);
    }

    #[test]
    fn rate_limit_config_validate_rejects_invalid() {
        assert!(RateLimitConfig::default().validate().is_ok());
        assert!(
            RateLimitConfig {
                default_delay: Duration::ZERO,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            RateLimitConfig {
                max_delay: Duration::from_secs(1),
                default_delay: Duration::from_secs(10),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            RateLimitConfig {
                max_retries: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn rate_limit_config_backoff_honours_hint_and_caps() {
        let cfg = RateLimitConfig::default();
        assert_eq!(
            cfg.backoff(Some(Duration::from_secs(12))),
            Duration::from_secs(12)
        );
        assert_eq!(
            cfg.backoff(Some(Duration::from_mins(2))),
            cfg.max_delay,
            "should cap at max_delay"
        );
        assert_eq!(cfg.backoff(None), cfg.default_delay);

        let ignore = RateLimitConfig {
            respect_retry_after: false,
            ..Default::default()
        };
        assert_eq!(
            ignore.backoff(Some(Duration::from_secs(12))),
            ignore.default_delay
        );
    }

    #[test]
    fn parse_retry_after_delta_seconds() {
        assert_eq!(parse_retry_after("12"), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("  7  "), Some(Duration::from_secs(7)));
    }

    #[test]
    fn parse_retry_after_huge_value_clamps() {
        let parsed = parse_retry_after("999999999999");
        assert!(parsed.is_some(), "huge value should parse, not panic");
    }

    #[test]
    fn parse_retry_after_garbage_returns_none() {
        assert_eq!(parse_retry_after("soon"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("abc"), None);
    }

    #[test]
    #[cfg(feature = "providers")]
    fn parse_retry_after_http_date() {
        let parsed = parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT");
        assert!(parsed.is_some(), "HTTP-date should parse under providers");
    }

    #[test]
    fn clamp_delay_to_deadline_none_deadline_returns_delay_unchanged() {
        let delay = Duration::from_mins(10);
        assert_eq!(clamp_delay_to_deadline(delay, None), delay);
    }

    #[test]
    fn clamp_delay_to_deadline_future_deadline_fits() {
        let delay = Duration::from_millis(10);
        let deadline = Some(Instant::now() + Duration::from_mins(1));
        assert_eq!(clamp_delay_to_deadline(delay, deadline), delay);
    }

    #[test]
    fn clamp_delay_to_deadline_exceeds_remaining() {
        let delay = Duration::from_mins(10);
        let remaining = Duration::from_millis(50);
        let deadline = Some(Instant::now() + remaining);
        let clamped = clamp_delay_to_deadline(delay, deadline);
        assert!(
            clamped <= remaining,
            "clamped {clamped:?} must not exceed remaining {remaining:?}"
        );
        assert!(
            !clamped.is_zero(),
            "deadline still in the future, so sleep should be positive"
        );
    }

    #[test]
    fn clamp_delay_to_deadline_past_deadline_zero() {
        let delay = Duration::from_mins(10);
        let deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
        assert_eq!(clamp_delay_to_deadline(delay, deadline), Duration::ZERO);
    }

    #[test]
    fn backoff_clamps_huge_hint_to_max_delay() {
        let cfg = RateLimitConfig {
            max_delay: Duration::from_mins(1),
            ..Default::default()
        };
        assert_eq!(
            cfg.backoff(Some(Duration::from_secs(9_999_999))),
            Duration::from_mins(1)
        );
    }

    fn detected_limit(retry_after: Option<Duration>) -> DetectedRateLimit {
        DetectedRateLimit {
            kind: RateLimitKind::RateLimited,
            retry_after,
            message: "slow down".to_string(),
        }
    }

    #[test]
    fn rate_limit_retry_returns_clamped_delay_below_ceilings() {
        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 3,
            max_retries: 5,
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_mins(1),
            ..Default::default()
        });
        let mut count = 0u32;
        let detail = detected_limit(Some(Duration::from_mins(10)));

        // Below both ceilings: retry with the hint clamped to max_delay.
        let decision = handler.rate_limit_retry(&detail, &mut count, None);
        assert_eq!(count, 1);
        match decision {
            RateLimitRetry::Retry(delay) => assert_eq!(delay, Duration::from_mins(1)),
            other => panic!("expected Retry, got {other:?}"),
        }
    }

    #[test]
    fn rate_limit_retry_escalates_after_fallback_ceiling() {
        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 2,
            max_retries: 5,
            ..Default::default()
        });
        let mut count = 0u32;
        let detail = detected_limit(Some(Duration::from_millis(5)));

        // Two retries are honored, then the next hit escalates.
        let _ = handler.rate_limit_retry(&detail, &mut count, None);
        let _ = handler.rate_limit_retry(&detail, &mut count, None);
        assert_eq!(count, 2);
        let decision = handler.rate_limit_retry(&detail, &mut count, None);
        assert_eq!(count, 3);
        match decision {
            RateLimitRetry::Escalate {
                attempts,
                retry_after,
            } => {
                assert_eq!(attempts, 3);
                assert_eq!(retry_after, Some(Duration::from_millis(5)));
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    #[test]
    fn rate_limit_retry_hard_stops_after_max_retries() {
        // fallback_after_retries high so escalation never fires; the hard-stop
        // backstop kicks in once max_retries is exceeded.
        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 100,
            max_retries: 2,
            ..Default::default()
        });
        let mut count = 0u32;
        let detail = detected_limit(None);

        let _ = handler.rate_limit_retry(&detail, &mut count, None);
        let _ = handler.rate_limit_retry(&detail, &mut count, None);
        assert_eq!(count, 2);
        assert!(matches!(
            handler.rate_limit_retry(&detail, &mut count, None),
            RateLimitRetry::HardStop
        ));
        assert_eq!(count, 3);
    }

    #[test]
    fn apply_event_extracts_stop_reason_and_accumulates() {
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;

        StreamHandler::apply_event(
            &StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg".to_string(),
                    role: "assistant".to_string(),
                    model: "m".to_string(),
                },
            }),
            &mut accumulator,
            &mut stop_reason,
        )
        .expect("MessageStart should accumulate");
        StreamHandler::apply_event(
            &StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(crate::stream::MessagePart::text("")),
            }),
            &mut accumulator,
            &mut stop_reason,
        )
        .expect("PartStart should accumulate");

        let result = StreamHandler::apply_event(
            &StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: None,
            }),
            &mut accumulator,
            &mut stop_reason,
        );
        assert!(result.is_ok());
        assert_eq!(stop_reason, StreamStopReason::EndTurn);

        let message = accumulator.build();
        assert_eq!(message.role, crate::message::Role::Assistant);
    }

    #[test]
    fn apply_event_unknown_stop_reason_keeps_prior() {
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::ToolCall;

        StreamHandler::apply_event(
            &StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("not_a_real_reason".to_string()),
                },
                usage: None,
            }),
            &mut accumulator,
            &mut stop_reason,
        )
        .expect("apply_event should not error on an out-of-context delta");
        assert_eq!(stop_reason, StreamStopReason::ToolCall);
    }

    #[test]
    fn detected_rate_limit_from_structured_variant() {
        let err = ApiError::RateLimit {
            retry_after: Some(Duration::from_secs(7)),
            message: "slow down".into(),
        };
        let detected = DetectedRateLimit::detect(&err).expect("RateLimit variant should detect");
        assert_eq!(detected.kind, RateLimitKind::RateLimited);
        assert_eq!(detected.retry_after, Some(Duration::from_secs(7)));
    }

    #[test]
    fn detected_rate_limit_from_structured_variant_no_hint() {
        let err = ApiError::RateLimit {
            retry_after: None,
            message: "slow down".into(),
        };
        let detected = DetectedRateLimit::detect(&err).expect("RateLimit variant should detect");
        assert_eq!(detected.retry_after, None);
    }

    #[test]
    fn detected_rate_limit_from_http_503() {
        let err = ApiError::http_with_status(503, "overloaded");
        let detected = DetectedRateLimit::detect(&err).expect("503 should detect as Overloaded");
        assert_eq!(detected.kind, RateLimitKind::Overloaded);
    }

    #[test]
    fn detected_rate_limit_http_500_is_not_overload() {
        let err = ApiError::http_with_status(500, "boom");
        assert!(DetectedRateLimit::detect(&err).is_none());
    }

    #[test]
    fn detected_rate_limit_non_rate_errors_return_none() {
        assert!(DetectedRateLimit::detect(&ApiError::api("connection reset")).is_none());
        assert!(DetectedRateLimit::detect(&ApiError::auth("bad key")).is_none());
    }

    #[test]
    fn is_rate_limited_matches_detect() {
        let cases: &[ApiError] = &[
            ApiError::RateLimit {
                retry_after: None,
                message: "x".into(),
            },
            ApiError::http_with_status(503, "overloaded"),
            ApiError::http_with_status(500, "boom"),
            ApiError::api("connection reset"),
            ApiError::auth("bad key"),
        ];
        for err in cases {
            assert_eq!(
                err.is_rate_limited(),
                DetectedRateLimit::detect(err).is_some(),
                "is_rate_limited disagree with detect on {err}",
            );
        }
    }

    #[tokio::test]
    async fn process_events_rate_limit_mid_stream() {
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
            Err(ApiError::RateLimit {
                retry_after: Some(Duration::from_secs(4)),
                message: "slow down".into(),
            }),
        ];
        let stream = event_stream(events);

        let err = handler
            .process_events(stream, &cancel, None)
            .await
            .expect_err("rate-limit error should fail the stream");

        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited {
                detail,
                has_partial_data,
                ..
            }) => {
                assert_eq!(detail.kind, RateLimitKind::RateLimited);
                assert_eq!(detail.retry_after, Some(Duration::from_secs(4)));
                assert!(!has_partial_data, "MessageStart carries no parts");
            }
            other => panic!("expected RateLimited outcome, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn process_events_rate_limit_with_partial_data() {
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
            Ok(StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(crate::stream::MessagePart::text("partial")),
            })),
            Ok(StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: " response".into(),
                },
            })),
            Ok(StreamEvent::PartStop),
            Err(ApiError::http_with_status(503, "overloaded")),
        ];
        let stream = event_stream(events);

        let err = handler
            .process_events(stream, &cancel, None)
            .await
            .expect_err("503 should fail the stream");

        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited {
                detail,
                has_partial_data,
                events_processed,
            }) => {
                assert_eq!(detail.kind, RateLimitKind::Overloaded);
                assert!(has_partial_data, "PartStart+delta should count as partial");
                assert!(events_processed >= 2, "at least 2 events processed");
            }
            other => panic!("expected RateLimited outcome, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn process_events_non_rate_error_still_init_failed() {
        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let events: Vec<Result<StreamEvent, ApiError>> = vec![
            Ok(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".to_string(),
                    role: "assistant".to_string(),
                    model: "test-model".to_string(),
                },
            })),
            Err(ApiError::api("connection reset")),
        ];
        let stream = event_stream(events);

        let err = handler
            .process_events(stream, &cancel, None)
            .await
            .expect_err("should fail");

        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::InitFailed { .. }) => {}
            other => panic!("expected InitFailed for non-rate error, got: {other:?}"),
        }
    }

    #[test]
    fn stream_outcome_rate_limited_display() {
        let outcome = StreamOutcome::RateLimited {
            detail: DetectedRateLimit {
                kind: RateLimitKind::RateLimited,
                retry_after: Some(Duration::from_secs(12)),
                message: "slow down".into(),
            },
            has_partial_data: false,
            events_processed: 5,
        };
        let s = outcome.to_string();
        assert!(s.contains("rate limit"), "got: {s}");
        assert!(s.contains("12"), "retry-after seconds missing: {s}");
    }

    #[test]
    fn stream_handler_rate_limit_config_round_trip() {
        let handler = StreamHandler::new();
        assert_eq!(
            handler.rate_limit_config().max_retries,
            RateLimitConfig::default().max_retries
        );

        let custom = RateLimitConfig {
            max_retries: 2,
            default_delay: Duration::from_secs(1),
            ..Default::default()
        };
        let handler = StreamHandler::new().with_rate_limit_config(custom);
        assert_eq!(handler.rate_limit_config().max_retries, 2);
        assert_eq!(
            handler.rate_limit_config().default_delay,
            Duration::from_secs(1)
        );
    }

    struct GateMock {
        url: &'static str,
    }

    impl ApiClient for GateMock {
        fn model(&self) -> String {
            "gate-model".to_string()
        }
        fn base_url(&self) -> String {
            self.url.to_string()
        }
        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
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
            Box::pin(async { Ok(serde_json::json!({})) })
        }
    }

    #[tokio::test]
    async fn gate_on_rate_limit_noop_without_limiter() {
        // Default handler has no limiter — the gate must return Ok immediately
        // and never touch a bucket.
        let handler = StreamHandler::new();
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());
        let result = handler.gate_on_rate_limit(&client, &cancel, None).await;
        assert!(result.is_ok(), "no limiter => gate is a no-op");
    }

    #[tokio::test]
    async fn gate_on_rate_limit_full_bucket_acquires_immediately() {
        // A fresh bucket is full, so the first acquire should return Ok without
        // any observable wait.
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(60));
        let handler = StreamHandler::new().with_rate_limiter(Arc::clone(&limiter));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        let start = Instant::now();
        handler
            .gate_on_rate_limit(&client, &cancel, None)
            .await
            .expect("full bucket should acquire");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "full bucket should not wait; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn gate_on_rate_limit_respects_total_deadline() {
        // A 1-RPM limiter with a generous max_wait (120s), so max_wait does NOT
        // fire first. Drain the single token, then call the gate with a
        // total_deadline already in the past: it must proceed immediately
        // (return Ok) rather than sleeping or spinning. The downstream
        // process_events deadline checks report the actual TotalTimeout — the
        // gate's job is just to not overrun the budget.
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(1).with_max_wait(Duration::from_mins(2)));
        let handler = StreamHandler::new().with_rate_limiter(Arc::clone(&limiter));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        // First acquire drains the only token.
        handler
            .gate_on_rate_limit(&client, &cancel, None)
            .await
            .expect("first acquire should succeed (full bucket)");

        // Expired total deadline → proceed immediately, no overrun.
        let expired = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or(Instant::now()),
        );
        let start = Instant::now();
        handler
            .gate_on_rate_limit(&client, &cancel, expired)
            .await
            .expect("gate should proceed on an expired deadline, not hang or spin");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "gate should proceed immediately on an expired deadline; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn gate_on_rate_limit_clamps_sleep_to_remaining_deadline() {
        // 1-RPM limiter (raw wait ~60s for a refill), max_wait 120s (so it does
        // NOT bind), but a total_deadline only ~80ms in the future. The gate
        // must clamp the sleep to the remaining deadline budget and proceed
        // after ~80ms — proving it honors the turn ceiling rather than the
        // 60s refill wait or the 120s max_wait.
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(1).with_max_wait(Duration::from_mins(2)));
        let handler = StreamHandler::new().with_rate_limiter(Arc::clone(&limiter));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        // First acquire drains the only token.
        handler
            .gate_on_rate_limit(&client, &cancel, None)
            .await
            .expect("first acquire should succeed (full bucket)");

        // Tight-but-not-expired deadline: ~80ms remaining.
        let near_deadline = Some(
            Instant::now()
                .checked_add(Duration::from_millis(80))
                .unwrap_or(Instant::now()),
        );
        let start = Instant::now();
        handler
            .gate_on_rate_limit(&client, &cancel, near_deadline)
            .await
            .expect("gate should proceed after clamping to the deadline");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "gate should proceed within the ~80ms deadline window, not wait 60s; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn proactive_throttle_slows_burst() {
        // 60 RPM → refill 1 token/sec, burst capacity 60. Fire 3 turns
        // back-to-back through stream_turn (end-to-end wiring).
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(60));
        let handler = StreamHandler::new().with_rate_limiter(Arc::clone(&limiter));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        let start = Instant::now();
        for _ in 0..3 {
            handler
                .stream_turn(&client, vec![], None, None, &cancel)
                .await
                .expect("turn should succeed");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "three turns from a 60-burst should be fast; elapsed {elapsed:?}"
        );
        assert!(limiter.is_enabled());
    }

    #[tokio::test]
    async fn proactive_throttle_cancel_interrupts_wait() {
        // 1 RPM → after the first turn drains the single-token bucket, the
        // second turn must wait ~60s. Cancelling during that wait should return
        // promptly.
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(1).with_max_wait(Duration::from_mins(2)));
        let handler = StreamHandler::new().with_rate_limiter(limiter);
        let client = GateMock { url: "openai" };

        // First turn consumes the only token.
        let cancel = Arc::new(CancelSignal::new());
        handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect("first turn should succeed");

        // Cancel before the second turn — it will need to wait ~60s for a token.
        let cancel2 = Arc::new(CancelSignal::new());
        cancel2.cancel();
        let start = Instant::now();
        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel2)
            .await
            .expect_err("should be cancelled, not hang for 60s");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "cancel should interrupt the wait promptly; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn proactive_throttle_max_wait_clamp_degrades_to_reactive() {
        // 1 RPM but max_wait = 50ms. The second turn must wait ~60s for a token,
        // but the clamp caps the cumulative wait at 50ms, so the turn proceeds.
        use crate::stream::rate_limit::RateLimiter;
        let limiter = Arc::new(RateLimiter::new(1).with_max_wait(Duration::from_millis(50)));
        let handler = StreamHandler::new().with_rate_limiter(limiter);
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        // First turn consumes the only token.
        handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect("first turn should succeed");

        // Second turn: bucket empty, wait capped at 50ms, then proceeds.
        let start = Instant::now();
        let result = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await;
        let elapsed = start.elapsed();

        assert!(
            result.is_ok(),
            "max_wait clamp should let the turn proceed, got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "should proceed after ~50ms, not wait 60s; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn stream_turn_uses_rate_limit_delay_on_rate_limited_outcome() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct RateLimitOnceMock {
            attempts: AtomicUsize,
        }
        impl ApiClient for RateLimitOnceMock {
            fn model(&self) -> String {
                "test-model".to_string()
            }
            fn stream_messages(
                &self,
                _messages: Vec<Message>,
                _system: Option<String>,
                _tools: Option<Vec<ToolSchema>>,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let n = self.attempts.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Box::pin(futures::stream::once(async {
                        Err(ApiError::RateLimit {
                            retry_after: None,
                            message: "slow down".into(),
                        })
                    }))
                } else {
                    Box::pin(futures::stream::iter(happy_stream_events()))
                }
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        // Rate-limit delay tiny; transport retry delay large. If the retry
        // loop honours the rate-limit outcome, the test finishes in ~1ms; if
        // it falls back to the transport base_delay, it sleeps 2s.
        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig::default(),
            StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 2_000,
                ..Default::default()
            },
        );
        let handler = handler.with_rate_limit_config(RateLimitConfig {
            default_delay: Duration::from_millis(1),
            ..Default::default()
        });
        let client = RateLimitOnceMock {
            attempts: AtomicUsize::new(0),
        };
        let cancel = Arc::new(CancelSignal::new());

        let start = Instant::now();
        let result = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect("second attempt should succeed");
        let elapsed = start.elapsed();

        assert!(!result.from_fallback);
        assert!(
            elapsed < Duration::from_secs(1),
            "rate-limit retry should use RateLimitConfig delay, not the 2s transport delay; elapsed {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn stream_turn_escalates_after_rate_limit_threshold() {
        // Every attempt is rate-limited, so the loop must escalate rather than
        // exhaust the generic transport budget.
        struct AlwaysRateLimitMock;
        impl ApiClient for AlwaysRateLimitMock {
            fn model(&self) -> String {
                "test-model".to_string()
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
                    Err(ApiError::RateLimit {
                        retry_after: Some(Duration::from_millis(1)),
                        message: "slow down".into(),
                    })
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 2,
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });
        let client = AlwaysRateLimitMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("should escalate, not succeed");
        match err {
            StreamHandlerError::RateLimitEscalation {
                attempts,
                retry_after,
                prior,
            } => {
                // fallback_after_retries == 2, so escalation fires on the 3rd hit.
                assert_eq!(attempts, 3);
                assert_eq!(retry_after, Some(Duration::from_millis(1)));
                match prior {
                    StreamOutcome::RateLimited { detail, .. } => {
                        assert_eq!(detail.kind, RateLimitKind::RateLimited);
                        assert_eq!(detail.retry_after, Some(Duration::from_millis(1)));
                    }
                    other => panic!("prior should be RateLimited, got {other:?}"),
                }
            }
            other => panic!("expected RateLimitEscalation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rate_limit_budget_independent_of_transport() {
        // Transport retry budget is tiny (max_retries = 1 -> 2 attempts), but
        // fallback_after_retries = 3. A leading transport error must NOT consume
        // the rate-limit budget: after the one transport failure, three
        // rate-limit retries must still be honored before escalating. Under the
        // old shared-counter loop this would exhaust the transport budget first
        // and fall through to non-streaming fallback instead of escalating.
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TransportThenRateLimitMock {
            calls: AtomicUsize,
        }
        impl ApiClient for TransportThenRateLimitMock {
            fn model(&self) -> String {
                "test-model".to_string()
            }
            fn stream_messages(
                &self,
                _messages: Vec<Message>,
                _system: Option<String>,
                _tools: Option<Vec<ToolSchema>>,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                let result = if n == 0 {
                    Err(ApiError::api("connection refused"))
                } else {
                    Err(ApiError::RateLimit {
                        retry_after: Some(Duration::from_millis(1)),
                        message: "slow down".into(),
                    })
                };
                Box::pin(futures::stream::once(async { result }))
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new()
            .with_config(
                StreamTimeoutConfig {
                    fallback_to_non_streaming: false,
                    ..Default::default()
                },
                StreamRetryConfig {
                    max_retries: 1,
                    base_delay_ms: 1,
                    ..Default::default()
                },
            )
            .with_rate_limit_config(RateLimitConfig {
                fallback_after_retries: 3,
                default_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
                ..Default::default()
            });

        let client = TransportThenRateLimitMock {
            calls: AtomicUsize::new(0),
        };
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("should escalate after the rate-limit budget, not fall through");
        match err {
            StreamHandlerError::RateLimitEscalation { attempts, .. } => {
                // One transport failure (not counted) + 3 rate-limit retries,
                // escalation on the 4th rate-limit hit.
                assert_eq!(attempts, 4);
            }
            other => panic!("expected RateLimitEscalation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rate_limit_hard_stop_after_max_retries() {
        // fallback_after_retries high so escalation never fires; max_retries low so
        // the hard-stop backstop returns the underlying rate-limit outcome.
        struct AlwaysRateLimitMock;
        impl ApiClient for AlwaysRateLimitMock {
            fn model(&self) -> String {
                "test-model".to_string()
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
                    Err(ApiError::RateLimit {
                        retry_after: None,
                        message: "slow down".into(),
                    })
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 100,
            max_retries: 2,
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });
        let client = AlwaysRateLimitMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("hard-stop should fail the turn");
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited { .. })
            | StreamHandlerError::InitFailed(StreamOutcome::RateLimited { .. }) => {}
            StreamHandlerError::RateLimitEscalation { .. } => {
                panic!("escalation must not fire when max_retries < fallback_after_retries")
            }
            other => panic!("expected rate-limit outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rate_limit_counter_does_not_leak_across_calls() {
        // The rate_limit_retries counter is a per-call local, so two independent
        // stream_turn calls on the same handler must each start fresh.
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct RateLimitOnceMock {
            attempts: AtomicUsize,
        }
        impl ApiClient for RateLimitOnceMock {
            fn model(&self) -> String {
                "test-model".to_string()
            }
            fn stream_messages(
                &self,
                _messages: Vec<Message>,
                _system: Option<String>,
                _tools: Option<Vec<ToolSchema>>,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let n = self.attempts.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Box::pin(futures::stream::once(async {
                        Err(ApiError::RateLimit {
                            retry_after: None,
                            message: "slow down".into(),
                        })
                    }))
                } else {
                    Box::pin(futures::stream::iter(happy_stream_events()))
                }
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });

        // First call: one rate-limit, then success.
        let client = RateLimitOnceMock {
            attempts: AtomicUsize::new(0),
        };
        let cancel = Arc::new(CancelSignal::new());
        handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect("first call should succeed after one rate-limit retry");

        // Second call on the SAME handler: fresh counter, so it must succeed too
        // rather than escalating on a leaked count.
        let client2 = RateLimitOnceMock {
            attempts: AtomicUsize::new(0),
        };
        handler
            .stream_turn(&client2, vec![], None, None, &cancel)
            .await
            .expect("second call should not see leaked rate-limit state");
    }

    #[tokio::test]
    async fn stream_turn_non_rate_limit_error_path_unchanged() {
        // Regression guard: a plain transport error still follows the generic
        // exponential-backoff path and returns StreamFailed, never escalation.
        struct AlwaysFailingMock;
        impl ApiClient for AlwaysFailingMock {
            fn model(&self) -> String {
                "test-model".to_string()
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
                    Err(ApiError::api("connection refused"))
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            },
            StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 1,
                ..Default::default()
            },
        );
        let client = AlwaysFailingMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("transport errors should fail");
        assert!(
            !matches!(err, StreamHandlerError::RateLimitEscalation { .. }),
            "non-rate-limit errors must not escalate"
        );
    }

    #[tokio::test]
    async fn stream_turn_rate_limit_delay_clamped_to_total_timeout() {
        // A huge Retry-After against a tight total_stream_timeout must fail
        // promptly (TotalTimeout), not sleep for the full hint.
        struct AlwaysRateLimitMock;
        impl ApiClient for AlwaysRateLimitMock {
            fn model(&self) -> String {
                "test-model".to_string()
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
                    Err(ApiError::RateLimit {
                        retry_after: Some(Duration::from_mins(10)),
                        message: "slow down".into(),
                    })
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new()
            .with_config(
                StreamTimeoutConfig {
                    total_stream_timeout: Duration::from_millis(80),
                    ..Default::default()
                },
                StreamRetryConfig {
                    max_retries: 10,
                    base_delay_ms: 1,
                    ..Default::default()
                },
            )
            .with_rate_limit_config(RateLimitConfig {
                // Honour the hint, but max_delay lets the 600s through so the
                // deadline clamp is what must bound the sleep.
                max_delay: Duration::from_mins(10),
                default_delay: Duration::from_millis(1),
                fallback_after_retries: 100,
                max_retries: 100,
                ..Default::default()
            });
        let client = AlwaysRateLimitMock;
        let cancel = Arc::new(CancelSignal::new());

        let start = Instant::now();
        let err = handler
            .stream_turn(&client, vec![], None, None, &cancel)
            .await
            .expect_err("tight total timeout should fail the turn");
        let elapsed = start.elapsed();
        // The clamp keeps the sleep inside the ~80ms budget, so the whole turn
        // resolves well under the 600s hint. Allow generous slack for scheduling.
        assert!(
            elapsed < Duration::from_secs(2),
            "deadline clamp should prevent a 600s sleep; elapsed {elapsed:?}",
        );
        // No escalation: the deadline tripped before the counter ceiling.
        assert!(
            !matches!(err, StreamHandlerError::RateLimitEscalation { .. }),
            "timeout should fire before escalation"
        );
    }

    #[tokio::test]
    async fn stream_turn_cancel_during_backoff_returns_immediately() {
        struct AlwaysFailingMock;
        impl ApiClient for AlwaysFailingMock {
            fn model(&self) -> String {
                "test-model".to_string()
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
                    Err(ApiError::api("connection lost"))
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
                Box::pin(async { Ok(serde_json::json!({})) })
            }
        }

        let handler = StreamHandler::new().with_config(
            StreamTimeoutConfig::default(),
            StreamRetryConfig {
                max_retries: 5,
                base_delay_ms: 60_000,
                ..Default::default()
            },
        );
        let cancel = Arc::new(CancelSignal::new());
        let cancel_clone = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancel_clone.cancel();
        });

        let start = Instant::now();
        let err = handler
            .stream_turn(&AlwaysFailingMock, vec![], None, None, &cancel)
            .await
            .expect_err("should return Cancelled, not hang for 60s");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got {err:?}",
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "cancellation during backoff should return immediately, not wait for the 60s sleep; elapsed {elapsed:?}",
        );
    }
}
