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
//! let handler = StreamHandler::new().with_timeout_config(
//!     StreamTimeoutConfig {
//!         initial_event_timeout: std::time::Duration::from_secs(60),
//!         ..Default::default()
//!     },
//! );
//! ```

use crate::api::ApiClient;
use crate::api::error::{ApiError, http_status_from_message, parse_retry_after};
use crate::cancel::CancelSignal;
use crate::message::Message;
use crate::stream::rate_limit;
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
use futures::StreamExt;
use futures::stream::Stream;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
/// | `per_event_timeout`     | Process | 180s    | Between consecutive events    |
/// | `total_stream_timeout`  | Process | 300s    | Maximum total stream duration |
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
/// assert_eq!(config.total_stream_timeout, Duration::from_secs(300));
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
            per_event_timeout: Duration::from_mins(3),
            total_stream_timeout: Duration::from_mins(5),
            max_consecutive_timeouts: 10,
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
        if self.total_stream_timeout == Duration::MAX {
            return Err(
                "total_stream_timeout must be finite — Duration::MAX silently disables the \
                 total deadline, both backoff clamps, and the non-streaming fallback deadline; \
                 construct the config directly (as passthrough does) to opt into that"
                    .to_string(),
            );
        }
        if self.total_stream_timeout < self.initial_event_timeout {
            return Err(format!(
                "total_stream_timeout ({:?}) must be >= initial_event_timeout ({:?})",
                self.total_stream_timeout, self.initial_event_timeout
            ));
        }

        if self.max_consecutive_timeouts == 0 {
            return Err("max_consecutive_timeouts must be >= 1".to_string());
        }
        Ok(())
    }
}

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
    /// [`max_delay_ms`](Self::max_delay_ms). This is the *raw* exponential
    /// backoff with no jitter; for the jittered delay used by
    /// [`StreamHandler`] on transport retries, use
    /// [`jittered_base_delay`](Self::jittered_base_delay).
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

    /// The raw exponential backoff with [`jitter_factor`](Self::jitter_factor) applied.
    ///
    /// Returns [`base_delay`](Self::base_delay)`(attempt)` scaled by a random
    /// factor in `[1 - jitter_factor, 1 + jitter_factor]`, drawn from
    /// [`fastrand`]. Concurrent retries with the same attempt number get
    /// different delays — avoiding a thundering herd where every client
    /// retries on the same tick. When [`jitter_factor`](Self::jitter_factor)
    /// is `0.0`, returns [`base_delay`](Self::base_delay) unchanged (no
    /// randomness, no allocation).
    ///
    /// This is the delay [`StreamHandler`] sleeps between transport-retry
    /// attempts; [`base_delay`](Self::base_delay) is the deterministic core
    /// it composes on.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::StreamRetryConfig;
    /// use std::time::Duration;
    ///
    /// let config = StreamRetryConfig { jitter_factor: 0.0, ..Default::default() };
    /// // With no jitter, the jittered delay equals the raw backoff exactly.
    /// assert_eq!(config.jittered_base_delay(1), config.base_delay(1));
    /// ```
    #[must_use]
    pub fn jittered_base_delay(&self, attempt: u32) -> Duration {
        let base = self.base_delay(attempt);
        if self.jitter_factor == 0.0 {
            return base;
        }
        let f = Self::random_signed_fraction() * self.jitter_factor;
        base.mul_f64(1.0 + f)
    }

    /// A random signed fraction in `[-1.0, 1.0)` from [`fastrand`].
    ///
    /// Draws a uniform `f64` in `[0.0, 1.0)` from fastrand's thread-local
    /// Wyrand PRNG and remaps it to `[-1.0, 1.0)`. Each call produces a
    /// different result, so concurrent retries spread their backoffs.
    #[must_use]
    fn random_signed_fraction() -> f64 {
        (fastrand::f64() - 0.5) * 2.0
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
        if self.fallback_after_retries > self.max_retries {
            return Err(format!(
                "fallback_after_retries ({}) must be <= max_retries ({})",
                self.fallback_after_retries, self.max_retries
            ));
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
    /// - the structured [`ApiError::RateLimit`] variant (typed `retry_after`;
    ///   kind `RateLimited` for 429-shaped messages, `Overloaded` when the
    ///   message carries a 503/529 status),
    /// - an `Api(String)` whose body contains `"rate limit"` or `"429"`, or
    /// - an `Http(String)` whose `"HTTP {status}:"` prefix indicates 429
    ///   (kind [`RateLimitKind::RateLimited`]) or 503/529 (kind
    ///   [`RateLimitKind::Overloaded`]).
    ///
    /// Returns `None` for everything else (500s, auth errors, generic
    /// transport failures, etc.).
    #[must_use]
    pub fn detect(err: &crate::api::error::ApiError) -> Option<Self> {
        match err {
            ApiError::RateLimit {
                retry_after,
                message,
            } => Some(Self {
                kind: match http_status_from_message(message) {
                    Some(503 | 529) => RateLimitKind::Overloaded,
                    _ => RateLimitKind::RateLimited,
                },
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
            ApiError::Http(msg) => {
                let kind = match http_status_from_message(msg) {
                    Some(429) => RateLimitKind::RateLimited,
                    Some(503 | 529) => RateLimitKind::Overloaded,
                    _ => return None,
                };
                Some(Self {
                    kind,
                    retry_after: parse_retry_after(msg),
                    message: msg.clone(),
                })
            }
            _ => None,
        }
    }
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
///
/// Exactly one of the two terminal rungs is reachable per configuration:
/// [`Escalate`](Self::Escalate) ends the turn, so the count never grows
/// past it while it sits below
/// [`max_retries`](RateLimitConfig::max_retries). The default config
/// (`fallback_after_retries: 3 < max_retries: 5`) therefore always
/// escalates; setting `fallback_after_retries == max_retries` removes the
/// escalation rung and retries to the hard ceiling instead.
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

    /// Give up on retrying the current model.
    ///
    /// Returned once the per-model retry count exceeds
    /// [`max_retries`](RateLimitConfig::max_retries) — the hard stop after
    /// which retrying the same model is pointless. Reachable only when
    /// [`fallback_after_retries`](RateLimitConfig::fallback_after_retries)
    /// equals [`max_retries`](RateLimitConfig::max_retries) (see the enum
    /// docs for why). Distinct from [`Escalate`](Self::Escalate):
    /// escalation hands off to the circuit breaker (and a fallback model);
    /// `HardStop` skips that hand-off — when
    /// [`fallback_to_non_streaming`](StreamTimeoutConfig::fallback_to_non_streaming)
    /// is enabled the turn gets one last-chance non-streaming request,
    /// otherwise it fails outright.
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

/// What `stream_turn`'s error arm should do after a failed stream event.
///
/// Produced by [`decide_rate_limit_error`](StreamHandler::decide_rate_limit_error)
/// and [`decide_transport_error`](StreamHandler::decide_transport_error). The
/// generator body matches on this to propagate the error, try a non-streaming
/// fallback, or sleep and retry — keeping the decision logic out of the
/// `async_stream` body.
///
/// This indirection exists because `async_stream::try_stream!` forbids
/// extracting `yield` / `?` into helper functions. The helpers return a
/// plain enum; the generator body is the only place the actual side-effect
/// (`yield`, `return`, `continue`) happens.
enum ErrorAction {
    /// Propagate the error and end the stream immediately.
    ///
    /// Carries the [`StreamHandlerError`] the caller propagates via `?`. The
    /// generator yields nothing further — the stream terminates with this
    /// error as the final item.
    Fail(StreamHandlerError),

    /// Attempt a non-streaming fallback before giving up.
    ///
    /// Carries the [`StreamOutcome`] from the failed stream attempt, which
    /// [`fallback_non_streaming`](StreamHandler::fallback_non_streaming) uses
    /// to build a diagnostic if the fallback also fails. The generator calls
    /// the fallback, yields a [`HandlerEvent::Fallback`] on success, and
    /// returns. On fallback failure the error propagates as if `Fail`.
    TryFallback(Option<StreamOutcome>),

    /// Sleep for the delay, then retry the outer stream loop.
    ///
    /// The delay is already clamped to the total-stream deadline by the
    /// decision method, so the generator just calls `sleep_cancellable` and
    /// `continue 'outer`. The retry counter has already been incremented.
    Retry(Duration),
}

/// A failed event poll paired with its retry classification.
///
/// Carries the [`StreamHandlerError`] the generator propagates plus the
/// [`ApiError::is_retryable`] verdict captured while the originating provider
/// error was still typed — before it was flattened into an outcome's message
/// string. [`decide_transport_error`](StreamHandler::decide_transport_error)
/// reads the verdict to fail fast on permanent errors instead of spending the
/// transport-retry ladder on them.
struct StreamFailure {
    /// The error to propagate when the failure is terminal.
    ///
    /// Built from the failing outcome exactly as it flows to the consumer;
    /// the retry verdict never alters the error's shape.
    error: StreamHandlerError,

    /// Whether the underlying failure is transient.
    ///
    /// `true` for timeouts, 5xx responses, 408 request timeouts, 429 rate
    /// limits, and connection-level transport errors; `false` for permanent
    /// classes (authentication rejections, other 4xx) and cancellation.
    retryable: bool,
}

impl StreamFailure {
    /// Wrap a transient failure whose full retry treatment applies.
    ///
    /// Used for the timeout outcomes, which have no [`ApiError`] to consult —
    /// a deadline that fired is by definition worth another attempt while the
    /// budget lasts.
    fn transient(error: StreamHandlerError) -> Self {
        Self {
            error,
            retryable: true,
        }
    }
}

/// Recover the [`StreamOutcome`] a [`StreamHandlerError`] carries, if any.
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

/// A future that completes at `deadline`, or never if it is `None`.
///
/// Shared by the deadline-driven arms of
/// [`next_event`](StreamHandler::next_event)'s `tokio::select!` (the
/// per-event timeout and the total-stream deadline). Each arm computes its
/// [`Option<Instant>`] deadline and hands it here, so this function owns the
/// single definition of "sleep until the instant, or stay pending forever
/// when disabled."
///
/// `None` disables the arm: the returned future never resolves, so the
/// `select!` branch stays inert. This is how
/// [`passthrough`](StreamHandler::passthrough) (which sets every timeout to
/// [`Duration::MAX`], yielding `None` deadlines) disables resilience without
/// a separate code path. A `Some(deadline)` already in the past resolves
/// immediately, letting a lapsed deadline fire on the next poll rather than
/// being missed.
async fn deadline_future(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
        None => std::future::pending::<()>().await,
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
/// Snapshotted once per loop iteration in [`StreamHandler::stream_turn`] and
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
    /// the top of [`stream_turn`](Self::stream_turn) rather
    /// than per event so the reported duration is the full stream lifetime,
    /// not the time since the most recent event.
    stream_start: Instant,

    /// Whether partial content has been accumulated.
    ///
    /// Recomputed each loop iteration (the whole [`EventDiagnostics`] is
    /// rebuilt per iteration in [`stream_turn`](Self::stream_turn))
    /// from the accumulator's current part count: `true` once at least one
    /// usable event has been received. Flows into the `has_partial_data` flag
    /// on [`StreamOutcome::TotalTimeout`], [`StreamOutcome::EventTimeout`],
    /// and [`StreamOutcome::RateLimited`], letting a downstream consumer
    /// decide whether to salvage the partial output or discard it.
    has_partial_data: bool,

    /// Stream attempts started so far this turn, counting both ladders.
    ///
    /// `transport_attempts + rate_limit_retries + 1` — the `+1` is the
    /// attempt in flight when this snapshot was taken. Mid-stream failures
    /// report it as [`StreamOutcome::InitFailed`]'s `attempts` so the
    /// count reflects every request the turn has issued, including the
    /// rate-limit ladder's backoffs (which never increment the transport
    /// counter).
    attempts_so_far: u32,
}

impl EventDiagnostics {
    /// Snapshot the diagnostic context for one event poll.
    ///
    /// Convenience constructor for the per-iteration snapshot
    /// [`next_event`](StreamHandler::next_event) receives: progress counts,
    /// the turn-level start instant, partial-data presence derived from
    /// the shadow accumulator's current part list, and the attempts
    /// started so far this turn (both retry ladders counted, plus the
    /// in-flight attempt).
    fn new(
        events_processed: u64,
        stream_start: Instant,
        shadow: &StreamAccumulator,
        attempts_so_far: u32,
    ) -> Self {
        Self {
            events_processed,
            stream_start,
            has_partial_data: !shadow.peek_parts().is_empty(),
            attempts_so_far,
        }
    }

    /// Build the [`StreamOutcome::TotalTimeout`] for this point in the stream.
    ///
    /// Snapshots the current diagnostic state — partial-data flag, events
    /// processed so far, and elapsed time since [`stream_start`](Self::stream_start)
    /// — into a `TotalTimeout` outcome. Used by [`stream_turn`](Self::stream_turn)
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
    /// itself — `stream_turn` owns it and passes the current value in).
    /// Used once the per-event timeout crosses
    /// [`max_consecutive_timeouts`](StreamTimeoutConfig::max_consecutive_timeouts).
    fn event_timeout(&self, consecutive_timeouts: u32) -> StreamOutcome {
        StreamOutcome::EventTimeout {
            has_partial_data: self.has_partial_data,
            consecutive_timeouts,
        }
    }

    /// Map a stream API error to the matching [`StreamFailure`].
    ///
    /// Two branches: if [`DetectedRateLimit::detect`] classifies the error as
    /// a 429/503/529, builds a [`StreamOutcome::RateLimited`] carrying the
    /// parsed `Retry-After` and current progress; otherwise wraps it as a
    /// generic [`StreamOutcome::InitFailed`] whose `attempts` is
    /// [`attempts_so_far`](Self::attempts_so_far) — every request the turn
    /// has issued, both ladders counted. The retry verdict is
    /// [`ApiError::is_retryable`] consulted while the error is still typed —
    /// the outcome flattens it to a message string, after which the
    /// classification would be unrecoverable. Used by `stream_turn` when the
    /// stream yields an `Err` event.
    fn api_error_failure(&self, error: &crate::api::error::ApiError) -> StreamFailure {
        let retryable = error.is_retryable();
        if let Some(detail) = DetectedRateLimit::detect(error) {
            return StreamFailure {
                error: StreamHandlerError::StreamFailed(StreamOutcome::RateLimited {
                    detail,
                    has_partial_data: self.has_partial_data,
                    events_processed: self.events_processed,
                }),
                retryable,
            };
        }
        StreamFailure {
            error: StreamHandlerError::StreamFailed(StreamOutcome::InitFailed {
                attempts: self.attempts_so_far,
                last_error: error.to_string(),
            }),
            retryable,
        }
    }
}

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
        /// Number of SSE events processed before `MessageStop`.
        ///
        /// Counts every event the accumulator accepted, including
        /// keep-alive heartbeats and metadata events. Useful as a
        /// throughput signal alongside `duration`.
        events_processed: u64,

        /// Wall-clock duration of the stream from first byte to
        /// `MessageStop`.
        ///
        /// Measured inside the handler; divide `events_processed` by
        /// this to get the average events-per-second rate for
        /// telemetry.
        duration: Duration,
    },

    /// Total-stream timeout exceeded.
    ///
    /// The stream was active for longer than
    /// [`total_stream_timeout`](StreamTimeoutConfig::total_stream_timeout).
    /// Partial data may be available in the accumulator.
    TotalTimeout {
        /// Whether partial content was accumulated before the timeout.
        ///
        /// `true` when at least one content event arrived before the
        /// deadline; the caller may inspect the accumulator to decide
        /// whether the partial result is usable.
        has_partial_data: bool,

        /// Events processed before the timeout fired.
        ///
        /// How many SSE events the accumulator accepted before the
        /// overall stream deadline elapsed — zero implies the stream
        /// stalled immediately.
        events_processed: u64,

        /// Elapsed time from stream start to the timeout trigger.
        ///
        /// Approximately equal to
        /// [`total_stream_timeout`](StreamTimeoutConfig::total_stream_timeout),
        /// reported for diagnostics so callers can correlate the
        /// observed wait with the configured ceiling.
        duration: Duration,
    },

    /// Per-event timeouts exhausted.
    ///
    /// Too many consecutive events failed to arrive within
    /// [`per_event_timeout`](StreamTimeoutConfig::per_event_timeout).
    /// Partial data may be available.
    EventTimeout {
        /// Whether partial content was accumulated before the failure.
        ///
        /// `true` when at least one content event arrived before the
        /// consecutive-timeout threshold was reached; the caller may
        /// inspect the accumulator to decide whether the partial result
        /// is usable.
        has_partial_data: bool,

        /// Consecutive per-event timeouts that triggered the failure.
        ///
        /// Reaches
        /// [`max_consecutive_timeouts`](StreamTimeoutConfig::max_consecutive_timeouts)
        /// when the handler gives up. Each individual timeout equals
        /// [`per_event_timeout`](StreamTimeoutConfig::per_event_timeout);
        /// this count tells the caller how many gaps were observed.
        consecutive_timeouts: u32,
    },

    /// A 429 / 503 rate-limit response arrived mid-stream.
    ///
    /// Distinct from [`InitFailed`](Self::InitFailed): the stream *was*
    /// established and may have produced partial output. The
    /// [`DetectedRateLimit`] carries the parsed `Retry-After`, if the server
    /// provided one.
    RateLimited {
        /// Decoded rate-limit detail.
        ///
        /// Carries the rate-limit kind (429 / 503 / 529) and the parsed
        /// `Retry-After` hint, if the server provided one. Downstream
        /// layers (the fallback manager, the retry loop) read this to
        /// honour the provider's back-off guidance without re-parsing
        /// the response.
        detail: DetectedRateLimit,

        /// Whether partial content was accumulated before the rate limit.
        ///
        /// `true` when the stream produced content events before the
        /// provider started rate-limiting; the caller may inspect the
        /// accumulator to decide whether the partial result is usable.
        has_partial_data: bool,

        /// Events processed before the rate limit fired.
        ///
        /// How many SSE events the accumulator accepted before the
        /// 429/503/529 response arrived — zero implies the provider
        /// rejected the stream early.
        events_processed: u64,
    },

    /// The stream failed before completing — the name is historical.
    ///
    /// Covers both a true initialization failure (no first event from
    /// any retry attempt) and, despite the name, mid-stream failures
    /// surfaced by the event loop (API error events, malformed
    /// accumulator events): those may have already delivered partial
    /// data to the consumer, which the retry caveat on
    /// [`on_text_delta`](crate::observer::LoopObserver::on_text_delta)
    /// documents. The `attempts` field counts every request the turn
    /// issued, both retry ladders included.
    InitFailed {
        /// The last error from the final retry attempt.
        ///
        /// Rendered to a string for diagnostics; typically the
        /// underlying transport or HTTP error that prevented the
        /// handler from receiving a first event. Surface this in logs
        /// so the caller can see why the stream never started.
        last_error: String,

        /// Number of retry attempts made before giving up.
        ///
        /// Counts the (re-)connection attempts up to
        /// [`StreamRetryConfig`]'s ceiling; reaching this count without
        /// a first event means the provider was unreachable or
        /// rejecting the request outright.
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
                write!(
                    f,
                    "stream failed before completing after {attempts} attempts: {last_error}"
                )
            }
            Self::FallbackToNonStreaming => {
                write!(f, "fell back to non-streaming request")
            }
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Errors produced by [`StreamHandler`].
///
/// Each variant captures the specific failure mode, allowing callers
/// to distinguish between transient failures (retryable) and permanent
/// errors (non-retryable).
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamHandlerError {
    /// The stream failed before completing — the name is historical.
    ///
    /// Covers initialization failures and mid-stream failures alike;
    /// partial data may have been delivered to the consumer (see the
    /// [`StreamOutcome`] and the retry caveat on
    /// [`on_text_delta`](crate::observer::LoopObserver::on_text_delta)).
    /// The outcome carries the last error and the attempt count.
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
        /// The streaming failure that triggered the fallback attempt.
        ///
        /// Preserved verbatim so the caller can see both halves of the
        /// double failure — why streaming gave up (timeout, mid-stream
        /// error, rate limit) and why the non-streaming retry then
        /// failed — without losing the original context.
        stream_outcome: StreamOutcome,

        /// The error returned by the non-streaming fallback request.
        ///
        /// Rendered to a string for diagnostics; typically the
        /// underlying transport or HTTP error from the
        /// `create_message()` call. Surface both this and
        /// `stream_outcome` in logs so the caller can see the full
        /// chain of failures.
        fallback_error: String,
    },

    /// The operation was cancelled.
    ///
    /// The [`CancelSignal`] was triggered
    /// before the stream completed. Partial data may be available.
    Cancelled,

    /// A mutex protecting pacing or routing state was found poisoned.
    ///
    /// Carries the subsystem label (e.g. `"rate_limit"`). Pacing or
    /// routing decisions cannot be made safely from desynchronised
    /// state; the caller must surface this rather than continue.
    Poisoned(&'static str),

    /// Rate-limit retries on the current model were exhausted.
    ///
    /// The handler honored the provider's `Retry-After` up to the configured
    /// [`RateLimitConfig::fallback_after_retries`] ceiling and could not make
    /// progress on this model. The caller should escalate to the model circuit
    /// breaker ([`FallbackManager`](crate::fallback::FallbackManager)), not the
    /// same-model non-streaming fallback.
    RateLimitEscalation {
        /// Number of rate-limit retries honored before escalating.
        ///
        /// Counts the 429/503/529 responses the handler retried
        /// (honoring the provider's `Retry-After`) before giving up on
        /// the current model and handing control to the model circuit
        /// breaker. Reaching
        /// [`RateLimitConfig::fallback_after_retries`] triggers this
        /// variant.
        attempts: u32,

        /// Last server-advised `Retry-After` hint, after clamping.
        ///
        /// Preserved for diagnostics and back-off tuning so the caller
        /// can correlate the escalation with the provider's last
        /// guidance. `None` when the provider sent no `Retry-After`
        /// header on the final rate-limited response.
        retry_after: Option<Duration>,
    },
}

impl fmt::Display for StreamHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitFailed(outcome) => write!(f, "stream failed before completing: {outcome}"),
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
            Self::Poisoned(what) => write!(f, "lock poisoned: {what}"),
            Self::RateLimitEscalation {
                attempts,
                retry_after,
            } => write!(
                f,
                "rate-limit escalation after {attempts} retries (retry-after {retry_after:?})"
            ),
        }
    }
}

impl std::error::Error for StreamHandlerError {}

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
/// let handler = StreamHandler::new().with_timeout_config(
///     StreamTimeoutConfig {
///         initial_event_timeout: std::time::Duration::from_secs(60),
///         ..Default::default()
///     },
/// );
/// assert_eq!(handler.timeout_config().initial_event_timeout, std::time::Duration::from_secs(60));
/// ```
pub struct StreamHandler {
    /// Timeout configuration for all phases.
    ///
    /// Drives the initial-event / per-event / total-stream deadlines, the
    /// consecutive-timeout escalation threshold, and the
    /// non-streaming-fallback toggle. Read on every turn in
    /// [`stream_turn`](Self::stream_turn) and on each event poll.
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

    /// Upper bound on how long `gate_on_rate_limit` blocks for a token.
    ///
    /// When the limiter's `acquire` returns a wait exceeding this, the
    /// handler proceeds anyway rather than hanging the agent. Defaults
    /// to 30 seconds.
    rate_limit_max_wait: Duration,
}

impl fmt::Debug for StreamHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamHandler")
            .field("timeout_config", &self.timeout_config)
            .field("retry_config", &self.retry_config)
            .field("rate_limit_config", &self.rate_limit_config)
            .field("rate_limiter", &self.rate_limiter)
            .field("rate_limit_max_wait", &self.rate_limit_max_wait)
            .finish()
    }
}

impl Default for StreamHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamHandler {
    /// Create a no-resilience handler — yields the raw provider stream with
    /// no retries, no timeouts, no fallback, no rate-limit handling.
    ///
    /// Used as the default when no `StreamHandler` is configured: the engine
    /// always routes streaming through a handler, and `passthrough()` is what
    /// you get when you haven't opted into resilience. Behaves like the
    /// pre-redesign inline streaming path — one attempt, no retry, surface
    /// the underlying [`ApiClient`] errors directly.
    ///
    /// Also useful as an explicit baseline for callers who want to start
    /// fresh and reconfigure only the fields they care about:
    ///
    /// ```rust,no_run
    /// # use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
    /// let handler = StreamHandler::passthrough()
    ///     .with_timeout_config(
    ///         StreamTimeoutConfig {
    ///             total_stream_timeout: std::time::Duration::from_secs(60),
    ///             ..Default::default()
    ///         },
    ///     );
    /// ```
    ///
    /// Returned by [`passthrough_default`](Self::passthrough_default) as a
    /// shared static, so [`StreamCapable::stream_handler`](crate::capabilities::StreamCapable::stream_handler)
    /// can return `&Self` (never `Option<&Self>`).
    #[must_use]
    pub fn passthrough() -> Self {
        // Duration::MAX overflows Instant::now() + it; stream_turn maps that
        // overflow to None (no deadline), so this never fires spuriously.
        const NEVER_TIME_OUT: Duration = Duration::MAX;
        Self {
            timeout_config: StreamTimeoutConfig {
                initial_event_timeout: NEVER_TIME_OUT,
                per_event_timeout: NEVER_TIME_OUT,
                total_stream_timeout: NEVER_TIME_OUT,
                // validate() rejects 0; value is irrelevant since timeouts never fire.
                max_consecutive_timeouts: 1,
                fallback_to_non_streaming: false,
            },
            retry_config: StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            },
            rate_limit_config: RateLimitConfig {
                max_retries: 0,
                fallback_after_retries: 0,
                ..Default::default()
            },
            rate_limiter: None,
            rate_limit_max_wait: Duration::from_secs(30),
        }
    }

    /// Return a shared reference to the default passthrough handler.
    ///
    /// Constructed lazily on first access (via `std::sync::OnceLock`) and
    /// shared by all callers that don't configure their own [`StreamHandler`].
    /// Used by
    /// [`StreamCapable::stream_handler`](crate::capabilities::StreamCapable::stream_handler)
    /// to always return `&Self` regardless of configuration.
    #[must_use]
    pub fn passthrough_default() -> &'static Self {
        static PASSTHROUGH: std::sync::OnceLock<StreamHandler> = std::sync::OnceLock::new();
        PASSTHROUGH.get_or_init(Self::passthrough)
    }

    /// Create a handler with default configuration.
    ///
    /// The defaults are suitable for production LLM API usage:
    /// - 120s initial event timeout
    /// - 180s per-event timeout
    /// - 300s total stream timeout
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
            rate_limit_max_wait: Duration::from_secs(30),
        }
    }

    /// Set the timeout configuration, consuming `self`.
    ///
    /// Each field that violates a [`validate`](StreamTimeoutConfig::validate)
    /// constraint is substituted with the default's value and named in a
    /// warning; the caller's valid fields are kept, and the sanitized
    /// result always satisfies every `validate` rule. Constructing the
    /// config directly bypasses this — call
    /// [`validate`](StreamTimeoutConfig::validate) on hand-built configs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
    /// use std::time::Duration;
    ///
    /// let handler = StreamHandler::new().with_timeout_config(
    ///     StreamTimeoutConfig {
    ///         initial_event_timeout: Duration::from_secs(60),
    ///         ..Default::default()
    ///     },
    /// );
    /// ```
    #[must_use]
    pub fn with_timeout_config(mut self, timeout: StreamTimeoutConfig) -> Self {
        self.timeout_config = Self::sanitized_timeout_config(timeout);
        self
    }

    /// Repair an invalid [`StreamTimeoutConfig`] field by field.
    ///
    /// Each field that violates a [`validate`](StreamTimeoutConfig::validate)
    /// constraint — including infinite (`Duration::MAX`) event timeouts,
    /// which silently disable the deadline they name — is substituted
    /// with the default's value and named in a warning; every valid
    /// field the caller supplied is kept. The ordering rule runs once
    /// after substitution: a `total_stream_timeout` below the (possibly
    /// sanitized) `initial_event_timeout` is raised to it, so the
    /// sanitized result always satisfies every `validate` rule.
    /// Constructing the config directly bypasses this — call
    /// [`validate`](StreamTimeoutConfig::validate) yourself on hand-built
    /// configs.
    fn sanitized_timeout_config(timeout: StreamTimeoutConfig) -> StreamTimeoutConfig {
        let default = StreamTimeoutConfig::default();
        let mut sanitized = timeout;
        let mut repaired: Vec<&'static str> = Vec::new();

        if sanitized.initial_event_timeout.is_zero()
            || sanitized.initial_event_timeout == Duration::MAX
        {
            sanitized.initial_event_timeout = default.initial_event_timeout;
            repaired.push("initial_event_timeout");
        }
        if sanitized.per_event_timeout.is_zero() || sanitized.per_event_timeout == Duration::MAX {
            sanitized.per_event_timeout = default.per_event_timeout;
            repaired.push("per_event_timeout");
        }
        if sanitized.total_stream_timeout.is_zero()
            || sanitized.total_stream_timeout == Duration::MAX
        {
            sanitized.total_stream_timeout = default.total_stream_timeout;
            repaired.push("total_stream_timeout");
        }
        if sanitized.total_stream_timeout < sanitized.initial_event_timeout {
            sanitized.total_stream_timeout = sanitized.initial_event_timeout;
            repaired.push("total_stream_timeout");
        }
        if sanitized.max_consecutive_timeouts == 0 {
            sanitized.max_consecutive_timeouts = default.max_consecutive_timeouts;
            repaired.push("max_consecutive_timeouts");
        }
        if !repaired.is_empty() {
            tracing::warn!(
                fields = repaired.join(","),
                "invalid StreamTimeoutConfig fields substituted with defaults"
            );
        }
        sanitized
    }

    /// Set the retry configuration, consuming `self`.
    ///
    /// Validates `retry`: if it violates any constraint, the invalid
    /// value is logged and the previously configured (initially default)
    /// config is kept instead. See
    /// [`StreamRetryConfig::validate`] for the constraints enforced.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::handler::{StreamHandler, StreamRetryConfig};
    ///
    /// let handler = StreamHandler::new().with_retry_config(
    ///     StreamRetryConfig {
    ///         max_retries: 5,
    ///         ..Default::default()
    ///     },
    /// );
    /// ```
    #[must_use]
    pub fn with_retry_config(mut self, retry: StreamRetryConfig) -> Self {
        if let Err(e) = retry.validate() {
            tracing::warn!(error = %e, "invalid StreamRetryConfig, falling back to default");
        } else {
            self.retry_config = retry;
        }
        self
    }

    /// Returns a reference to the timeout configuration.
    ///
    /// Read-only access to the [`StreamTimeoutConfig`] stored on the handler.
    /// Mutate via
    /// [`with_timeout_config`](Self::with_timeout_config).
    #[must_use]
    pub fn timeout_config(&self) -> &StreamTimeoutConfig {
        &self.timeout_config
    }

    /// Returns a reference to the retry configuration.
    ///
    /// Read-only access to the [`StreamRetryConfig`] stored on the handler.
    /// Mutate via [`with_retry_config`](Self::with_retry_config).
    #[must_use]
    pub fn retry_config(&self) -> &StreamRetryConfig {
        &self.retry_config
    }

    /// Set a custom [`RateLimitConfig`]. Consuming builder.
    ///
    /// Validates the config: if it violates any constraint (zero delay,
    /// inverted ceilings, etc.), the invalid value is logged and the
    /// default config is used instead. This prevents silently storing a
    /// config that would invert retry/escalation behavior.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::stream::handler::{StreamHandler, RateLimitConfig};
    ///
    /// let handler = StreamHandler::new().with_rate_limit_config(
    ///     RateLimitConfig { max_retries: 5, fallback_after_retries: 2, ..Default::default() },
    /// );
    /// assert_eq!(handler.rate_limit_config().max_retries, 5);
    /// ```
    #[must_use]
    pub fn with_rate_limit_config(mut self, rl: RateLimitConfig) -> Self {
        if let Err(e) = rl.validate() {
            tracing::warn!(error = %e, "invalid RateLimitConfig, falling back to default");
            return self;
        }
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
    /// When set, every `stream_turn` attempt waits for a token before
    /// opening the stream, spacing requests to the limiter's `requests_per_minute`
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

    /// Set the max-wait ceiling for `gate_on_rate_limit` (builder style).
    ///
    /// Defaults to 30 seconds. When the limiter's `acquire` returns a
    /// wait exceeding this, the handler proceeds anyway rather than
    /// hanging the agent.
    #[must_use]
    pub fn with_rate_limit_max_wait(mut self, max_wait: Duration) -> Self {
        self.rate_limit_max_wait = max_wait;
        self
    }

    /// Drive one turn as a stream of [`HandlerEvent`]s.
    ///
    /// The engine consumes this stream directly: real stream events flow to
    /// observers and the engine's accumulator (identical to the inline
    /// streaming path), [`HandlerEvent::AttemptReset`] signals a retry so the
    /// engine can discard partial state, and [`HandlerEvent::Fallback`]
    /// delivers the non-streaming fallback message when retries exhaust.
    ///
    /// The handler keeps its retry/rate-limit/timeout/fallback machinery —
    /// this method is the retry loop reshaped as a stream. Events from failed
    /// attempts are yielded as `HandlerEvent::Stream` to the engine before the
    /// attempt fails; the engine then receives `AttemptReset` and discards
    /// them. Consumers that want only committed output must reset their state
    /// on `AttemptReset`.
    ///
    /// # Errors
    ///
    /// Item errors carry the same [`StreamHandlerError`] variants the
    /// pre-redesign `stream_turn` returned: cancellation, timeout, transport
    /// retry exhaustion, rate-limit escalation, and non-streaming fallback
    /// failure.
    /// A successful turn ends with `None` from the stream once the
    /// provider's terminal `MessageStop` has been seen (or after
    /// [`HandlerEvent::Fallback`]); a stream that ends without the
    /// terminal event is treated as truncated and routed through the
    /// retry ladder like any other mid-stream failure.
    pub fn stream_turn<'a, C: ApiClient>(
        &'a self,
        client: &'a C,
        request: &'a crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
        cancel: &'a Arc<CancelSignal>,
    ) -> Pin<Box<dyn Stream<Item = Result<HandlerEvent, StreamHandlerError>> + Send + 'a>> {
        let total_deadline = Instant::now().checked_add(self.timeout_config.total_stream_timeout);
        let stream_start = Instant::now();
        let max_attempts = self.retry_config.max_retries.saturating_add(1);

        Box::pin(async_stream::try_stream! {
            let mut rate_limit_retries: u32 = 0;
            let mut transport_attempts: u32 = 0;
            let mut first_attempt = true;
            // Shadow accumulator tracks partial-data presence for diagnostics.
            let mut shadow = StreamAccumulator::new();

            loop {
                if !first_attempt {
                    shadow = StreamAccumulator::new();
                    yield HandlerEvent::AttemptReset;
                }
                first_attempt = false;

                self.gate_on_rate_limit(client, cancel, total_deadline).await?;
                let mut stream =
                    client.stream_messages_with_options(request, options.clone());

                let mut consecutive_timeouts: usize = 0;
                let mut events_processed: u64 = 0;
                let mut saw_terminal = false;

                let action = loop {
                    let diagnostics = EventDiagnostics::new(
                        events_processed,
                        stream_start,
                        &shadow,
                        transport_attempts
                            .saturating_add(rate_limit_retries)
                            .saturating_add(1),
                    );
                    match self
                        .next_event(
                            &mut stream,
                            cancel,
                            &mut consecutive_timeouts,
                            total_deadline,
                            &diagnostics,
                        )
                        .await
                    {
                        Ok(Some(event)) => {
                            events_processed = events_processed.saturating_add(1);
                            consecutive_timeouts = 0;
                            if matches!(event, StreamEvent::MessageStop) {
                                saw_terminal = true;
                            }
                            if let Err(failure) =
                                Self::accumulate_event(&diagnostics, &mut shadow, &event)
                            {
                                break self.decide_failure_action(
                                    failure,
                                    &mut rate_limit_retries,
                                    &mut transport_attempts,
                                    max_attempts,
                                    total_deadline,
                                );
                            }
                            yield HandlerEvent::Stream(event);
                        }
                        Ok(None) => {
                            if saw_terminal {
                                return;
                            }
                            let failure = StreamFailure::transient(
                                StreamHandlerError::StreamFailed(StreamOutcome::InitFailed {
                                    attempts: diagnostics.attempts_so_far,
                                    last_error: format!(
                                        "stream ended without a terminal event after \
                                         {events_processed} events (truncated?)"
                                    ),
                                }),
                            );
                            break self.decide_failure_action(
                                failure,
                                &mut rate_limit_retries,
                                &mut transport_attempts,
                                max_attempts,
                                total_deadline,
                            );
                        }
                        Err(failure) => break self.decide_failure_action(
                            failure,
                            &mut rate_limit_retries,
                            &mut transport_attempts,
                            max_attempts,
                            total_deadline,
                        ),
                    }
                };

                match action {
                    ErrorAction::Fail(e) => {
                        Err(e)?;
                        return;
                    }
                    ErrorAction::TryFallback(outcome) => {
                        let (message, stop_reason, usage) = self
                            .fallback_non_streaming(
                                client,
                                request,
                                &options,
                                cancel,
                                total_deadline,
                                outcome,
                            )
                            .await?;
                        yield HandlerEvent::Fallback {
                            message,
                            stop_reason,
                            usage,
                        };
                        return;
                    }
                    ErrorAction::Retry(delay) => {
                        sleep_cancellable(delay, cancel).await?;
                    }
                }
            }
        })
    }

    /// Route a failed event poll to the matching retry ladder.
    ///
    /// The single dispatch point for the generator's error arm: when the
    /// failure's carried outcome is [`StreamOutcome::RateLimited`] it draws on
    /// the rate-limit budget via
    /// [`decide_rate_limit_error`](Self::decide_rate_limit_error); every other
    /// failure draws on the transport budget via
    /// [`decide_transport_error`](Self::decide_transport_error) (including the
    /// retryability fast-fail and the terminal total-timeout route). Keeping
    /// the outcome-inspection here leaves the generator body a flat
    /// decision-and-act sequence.
    fn decide_failure_action(
        &self,
        failure: StreamFailure,
        rate_limit_retries: &mut u32,
        transport_attempts: &mut u32,
        max_attempts: u32,
        total_deadline: Option<Instant>,
    ) -> ErrorAction {
        let last_stream_outcome = carried_outcome(&failure.error);
        if let Some(StreamOutcome::RateLimited { detail, .. }) = &last_stream_outcome {
            self.decide_rate_limit_error(
                failure.error,
                detail,
                rate_limit_retries,
                total_deadline,
                last_stream_outcome.clone(),
            )
        } else {
            self.decide_transport_error(
                failure.error,
                failure.retryable,
                transport_attempts,
                max_attempts,
                last_stream_outcome.clone(),
                total_deadline,
            )
        }
    }

    /// Accumulate one accepted event into the shadow accumulator.
    ///
    /// Wraps [`StreamAccumulator::process`] for the generator's happy-path
    /// arm: a malformed event becomes a transient [`StreamFailure`] with the
    /// attempt count carried as [`EventDiagnostics::attempts_so_far`], so the
    /// generator routes it through [`decide_failure_action`](Self::decide_failure_action)
    /// like every other mid-stream failure — it draws on the retry budget
    /// and, at exhaustion with the fallback enabled, gets the last-chance
    /// non-streaming request instead of failing the turn on first
    /// occurrence. Wire-level protocol violations are usually transient
    /// (proxy corruption, truncated chunks), and a fresh attempt replays
    /// the whole stream.
    ///
    /// # Errors
    ///
    /// Returns the wrapped accumulation failure for the generator to hand
    /// to the decision ladder.
    fn accumulate_event(
        diagnostics: &EventDiagnostics,
        shadow: &mut StreamAccumulator,
        event: &StreamEvent,
    ) -> Result<(), StreamFailure> {
        shadow.process(event).map_err(|e| {
            tracing::warn!(
                error = %e,
                attempts = diagnostics.attempts_so_far,
                events_processed = diagnostics.events_processed,
                "malformed accumulator event rejected"
            );
            StreamFailure::transient(StreamHandlerError::StreamFailed(
                StreamOutcome::InitFailed {
                    attempts: diagnostics.attempts_so_far,
                    last_error: e.to_string(),
                },
            ))
        })
    }

    /// Decide how to handle a rate-limit stream error.
    ///
    /// Delegates to [`rate_limit_retry`](Self::rate_limit_retry) for the
    /// retry/escalate/hard-stop decision, then maps the result to an
    /// [`ErrorAction`] the generator body can act on.
    ///
    /// `Escalate` fails the turn with
    /// [`RateLimitEscalation`](StreamHandlerError::RateLimitEscalation) —
    /// under the default config this is the ladder's only terminal outcome.
    /// A rate limit is charged against the model's quota, so a same-model
    /// non-streaming request is not attempted; the engine records the
    /// escalation against the circuit breaker, which routes subsequent
    /// turns to a fallback model. `HardStop` (reachable only when
    /// `fallback_after_retries == max_retries`; see the
    /// [`RateLimitRetry`] docs) behaves differently: when
    /// [`fallback_to_non_streaming`](StreamTimeoutConfig::fallback_to_non_streaming)
    /// is enabled it returns [`ErrorAction::TryFallback`] instead of
    /// failing, so a host that configures the ceiling-equal ladder opts
    /// into the last-chance non-streaming request.
    fn decide_rate_limit_error(
        &self,
        err: StreamHandlerError,
        detail: &DetectedRateLimit,
        rate_limit_retries: &mut u32,
        total_deadline: Option<Instant>,
        last_outcome: Option<StreamOutcome>,
    ) -> ErrorAction {
        match self.rate_limit_retry(detail, rate_limit_retries, total_deadline) {
            RateLimitRetry::Escalate {
                attempts,
                retry_after,
            } => ErrorAction::Fail(StreamHandlerError::RateLimitEscalation {
                attempts,
                retry_after,
            }),
            RateLimitRetry::HardStop => {
                if self.timeout_config.fallback_to_non_streaming {
                    ErrorAction::TryFallback(last_outcome)
                } else {
                    ErrorAction::Fail(err)
                }
            }
            RateLimitRetry::Retry(delay) => ErrorAction::Retry(delay),
        }
    }

    /// Decide how to handle a non-rate-limit transport stream error.
    ///
    /// Fails fast — one attempt, no backoff, no non-streaming fallback — when
    /// `retryable` is `false`: the verdict is [`ApiError::is_retryable`]
    /// consulted while the provider error was still typed, so permanent
    /// classes (authentication rejections, other 4xx the retryable set
    /// excludes) never enter the retry math.
    ///
    /// An outcome carrying [`StreamOutcome::TotalTimeout`] is equally
    /// terminal but takes the budget-exhaustion route instead: the total
    /// budget is already spent, so a second streaming attempt can only fail
    /// against the same expired deadline — the non-streaming fallback runs
    /// when one is configured, otherwise the error fails the turn. The
    /// already-built outcome (real `events_processed`, real
    /// `has_partial_data`) propagates verbatim.
    ///
    /// For other retryable errors, checks whether the transport-retry budget
    /// is exhausted:
    ///
    /// - **Exhausted + fallback enabled** → [`ErrorAction::TryFallback`]:
    ///   the non-streaming path gets one last chance, carrying the stream
    ///   outcome for diagnostics if it also fails.
    /// - **Exhausted + fallback disabled** → [`ErrorAction::Fail`]:
    ///   propagate the error.
    /// - **Retries remaining** → [`ErrorAction::Retry`]: sleep for the
    ///   jittered backoff (clamped to the total-stream deadline), then
    ///   retry. Increments `transport_attempts` so the next call knows
    ///   how many attempts have been spent.
    fn decide_transport_error(
        &self,
        err: StreamHandlerError,
        retryable: bool,
        transport_attempts: &mut u32,
        max_attempts: u32,
        last_outcome: Option<StreamOutcome>,
        total_deadline: Option<Instant>,
    ) -> ErrorAction {
        if !retryable {
            return ErrorAction::Fail(err);
        }
        if matches!(last_outcome, Some(StreamOutcome::TotalTimeout { .. })) {
            if self.timeout_config.fallback_to_non_streaming {
                return ErrorAction::TryFallback(last_outcome);
            }
            return ErrorAction::Fail(err);
        }
        if *transport_attempts >= max_attempts.saturating_sub(1) {
            if self.timeout_config.fallback_to_non_streaming {
                return ErrorAction::TryFallback(last_outcome);
            }
            return ErrorAction::Fail(err);
        }
        let delay = self.retry_config.jittered_base_delay(*transport_attempts);
        let delay = clamp_delay_to_deadline(delay, total_deadline);
        *transport_attempts = transport_attempts.saturating_add(1);
        ErrorAction::Retry(delay)
    }

    /// Decide how to handle a rate-limit failure on the current model.
    ///
    /// Bumps `count` and returns one of:
    /// - [`RateLimitRetry::HardStop`] once `count` exceeds
    ///   [`max_retries`](RateLimitConfig::max_retries) — the absolute ceiling;
    /// - [`RateLimitRetry::Escalate`] once `count` exceeds
    ///   [`fallback_after_retries`](RateLimitConfig::fallback_after_retries)
    ///   but is still within `max_retries` — the caller escalates to the
    ///   model circuit breaker;
    /// - [`RateLimitRetry::Retry`] with the deadline-clamped backoff otherwise.
    ///
    /// `max_retries` is checked first so it is always enforced as the hard
    /// ceiling, regardless of `fallback_after_retries`.
    fn rate_limit_retry(
        &self,
        detail: &DetectedRateLimit,
        count: &mut u32,
        deadline: Option<Instant>,
    ) -> RateLimitRetry {
        *count = count.saturating_add(1);
        if *count > self.rate_limit_config.max_retries {
            return RateLimitRetry::HardStop;
        }
        if *count > self.rate_limit_config.fallback_after_retries {
            return RateLimitRetry::Escalate {
                attempts: *count,
                retry_after: detail.retry_after,
            };
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
    /// retry loop in [`stream_turn`](Self::stream_turn), so a
    /// retried 429 re-gates on the rate limiter.
    ///
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
    /// [`stream_turn`](Self::stream_turn) report the expiry). A no-op
    /// when no limiter is attached.
    ///
    /// Fires per attempt (called from
    /// [`stream_turn`](Self::stream_turn), which runs the retry
    /// loop), so a retried 429 re-respects the budget.
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
        let max_wait = self.rate_limit_max_wait;
        let mut waited = Duration::ZERO;
        loop {
            match limiter.acquire(&key) {
                Ok(()) => return Ok(()),
                Err(rate_limit::RateLimitError::Poisoned) => {
                    tracing::warn!("rate-limit bucket poisoned; pacing unavailable");
                    return Err(StreamHandlerError::Poisoned("rate_limit"));
                }
                Err(rate_limit::RateLimitError::Wait(wait)) => {
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
    /// Returns a [`StreamFailure`] carrying [`StreamHandlerError::Cancelled`]
    /// if the cancel signal fires, or
    /// [`StreamHandlerError::StreamFailed`] on total/per-event timeout or an
    /// API error — paired with the error's retry classification where the
    /// provider error was still typed.
    async fn next_event<S>(
        &self,
        stream: &mut S,
        cancel: &Arc<CancelSignal>,
        consecutive_timeouts: &mut usize,
        total_deadline: Option<Instant>,
        diagnostics: &EventDiagnostics,
    ) -> Result<Option<StreamEvent>, StreamFailure>
    where
        S: futures::Stream<Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>>
            + Unpin,
    {
        loop {
            if Self::deadline_exceeded(total_deadline) {
                return Err(StreamFailure::transient(StreamHandlerError::StreamFailed(
                    diagnostics.total_timeout(),
                )));
            }
            if cancel.is_cancelled() {
                return Err(StreamFailure {
                    error: StreamHandlerError::Cancelled,
                    retryable: false,
                });
            }

            let event_deadline = self.event_deadline(diagnostics.events_processed);
            let event_result = tokio::select! {
                event = stream.next() => EventPoll::Next(event),
                () = cancel.notified() => return Err(StreamFailure {
                    error: StreamHandlerError::Cancelled,
                    retryable: false,
                }),
                () = deadline_future(event_deadline) => EventPoll::TimedOut,
                () = deadline_future(total_deadline) => {
                    return Err(StreamFailure::transient(
                        StreamHandlerError::StreamFailed(diagnostics.total_timeout()),
                    ));
                }
            };
            match event_result {
                EventPoll::TimedOut => {
                    *consecutive_timeouts = consecutive_timeouts.saturating_add(1);
                    let max_consecutive = if diagnostics.events_processed == 0 {
                        self.timeout_config.max_consecutive_timeouts.min(2) as usize
                    } else {
                        self.timeout_config.max_consecutive_timeouts as usize
                    };
                    if *consecutive_timeouts >= max_consecutive {
                        return Err(StreamFailure::transient(StreamHandlerError::StreamFailed(
                            diagnostics.event_timeout(
                                u32::try_from(*consecutive_timeouts).unwrap_or(u32::MAX),
                            ),
                        )));
                    }
                }
                EventPoll::Next(Some(Ok(event))) => return Ok(Some(event)),
                EventPoll::Next(Some(Err(api_error))) => {
                    return Err(diagnostics.api_error_failure(&api_error));
                }
                EventPoll::Next(None) => return Ok(None),
            }
        }
    }

    /// The deadline for the next stream event, or `None` if disabled.
    ///
    /// Computes the instant at which the per-event timeout fires for the
    /// current poll: [`initial_event_timeout`](StreamTimeoutConfig::initial_event_timeout)
    /// before any event has arrived (the model may need time to begin
    /// generating), then [`per_event_timeout`](StreamTimeoutConfig::per_event_timeout)
    /// once events are flowing. A disabled timeout ([`Duration::MAX`])
    /// overflows `Instant::now() + timeout`, so `checked_add` returns `None`
    /// and the caller arms a never-firing `select!` branch. `events_processed`
    /// is the same counter [`next_event`](Self::next_event) maintains, so the
    /// deadline always matches the timeout phase the stream is in.
    fn event_deadline(&self, events_processed: u64) -> Option<Instant> {
        let base_timeout = if events_processed == 0 {
            self.timeout_config.initial_event_timeout
        } else {
            self.timeout_config.per_event_timeout
        };
        Instant::now().checked_add(base_timeout)
    }

    /// Whether the total-stream deadline has already passed.
    ///
    /// Polled between events at the top of [`next_event`](Self::next_event)'s
    /// loop, before the per-event `select!` commits to another wait. This
    /// catches a deadline that elapsed while the loop was processing the
    /// previous event (or building diagnostics) — the
    /// [`deadline_future`] `select!` arm only fires *during* a wait, so
    /// without this check a long event handler could overshoot the deadline
    /// by up to one event's processing time.
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

    /// Fall back to non-streaming message creation.
    ///
    /// Called when streaming fails (timeout, retries exhausted) and
    /// `fallback_to_non_streaming` is enabled. Uses
    /// [`ApiClient::create_message_with_options`] with the turn's
    /// [`RequestOptions`](crate::structured::RequestOptions) — a configured
    /// `response_format` or `tool_constraint` applies to the fallback exactly
    /// as it did to the streaming attempt — to get a complete typed response:
    /// the message, stop reason, and token usage are returned directly, with
    /// no JSON parsing at this layer.
    ///
    /// While streaming-budget time remains, the request is raced against the
    /// turn's `total_deadline` (the same one that bounded the streaming
    /// attempts), so a hanging non-streaming call cannot outlive the budget
    /// the stream already spent. When the deadline has already expired by
    /// the time the fallback starts — the terminal total-timeout paths,
    /// where the expiry itself is what triggered the fallback — racing
    /// against it would kill the request on its first poll and make the
    /// configured fallback unreachable; instead the fallback gets one fresh
    /// budget of [`initial_event_timeout`] to produce its answer. In both
    /// cases the deadline bounds *waiting*, not completion: a response that
    /// has resolved by the time the select is polled is accepted even when
    /// it lands at or past the deadline — the answer exists and its tokens
    /// are already spent, so discarding it would trade finished work for
    /// wall-clock bookkeeping. `None` means no deadline is configured and
    /// the call is bounded only by cancellation and any client-level limits.
    ///
    /// # Errors
    ///
    /// Returns [`StreamHandlerError::FallbackFailed`] if the fallback request
    /// also fails or its deadline expires before it completes, or
    /// [`StreamHandlerError::Cancelled`] if the cancel signal fires.
    ///
    /// [`initial_event_timeout`]: StreamTimeoutConfig::initial_event_timeout
    async fn fallback_non_streaming<C: ApiClient>(
        &self,
        client: &C,
        request: &crate::api::StreamRequest,
        options: &crate::structured::RequestOptions,
        cancel: &Arc<CancelSignal>,
        total_deadline: Option<Instant>,
        stream_outcome: Option<StreamOutcome>,
    ) -> Result<(Message, StreamStopReason, Option<Usage>), StreamHandlerError> {
        if cancel.is_cancelled() {
            return Err(StreamHandlerError::Cancelled);
        }

        let fallback_deadline = match total_deadline {
            Some(deadline) if deadline > Instant::now() => total_deadline,
            Some(_) => Instant::now().checked_add(self.timeout_config.initial_event_timeout),
            None => None,
        };
        let result = tokio::select! {
            biased;

            () = cancel.notified() => {
                return Err(StreamHandlerError::Cancelled);
            }
            res = client.create_message_with_options(request, options.clone()) => res,
            () = deadline_future(fallback_deadline) => {
                return Err(StreamHandlerError::FallbackFailed {
                    stream_outcome: stream_outcome.unwrap_or(StreamOutcome::InitFailed {
                        attempts: 0,
                        last_error: "unknown".to_string(),
                    }),
                    fallback_error: "fallback request exceeded its deadline".to_string(),
                });
            }
        };

        match result {
            Ok(response) => Ok((response.message, response.stop_reason, response.usage)),
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

/// Event yielded by [`StreamHandler::stream_turn`].
///
/// The engine drives the handler's turn as a stream of these events: real
/// stream events flow through to observers and the engine's accumulator;
/// retry boundaries and the non-streaming fallback are surfaced as
/// first-class signals so the engine can react (reset state, swap in the
/// fallback message).
///
/// See [`StreamHandler::stream_turn`] for the contract.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HandlerEvent {
    /// A raw event from the provider's stream (text delta, thinking delta,
    /// tool call, etc.).
    ///
    /// The engine forwards these to observers (`on_text_delta`,
    /// `on_thinking_delta`, `text_streamer`) exactly like the inline path,
    /// then feeds them to its own [`StreamAccumulator`].
    Stream(StreamEvent),

    /// The handler is starting a new attempt after a retry decision.
    ///
    /// Fired before the first `Stream` event of attempts 2, 3, … (never on
    /// the first attempt). The engine must reset any per-attempt state —
    /// including its [`StreamAccumulator`] — so events from the failed
    /// attempt are discarded rather than concatenated with the retry's
    /// events. Observers that concatenate deltas per turn must reset too.
    AttemptReset,

    /// Streaming retries are exhausted and the non-streaming fallback
    /// succeeded.
    ///
    /// Carries the final message, stop reason, and token usage from the
    /// non-streaming
    /// [`create_message_with_options`](crate::api::ApiClient::create_message_with_options)
    /// typed response, bounded by the turn's total deadline. The engine
    /// should stop accumulating and use these directly —
    /// the streaming accumulator's partial state from failed attempts is
    /// irrelevant on this path.
    ///
    /// Always the last event before the stream ends (when the fallback path
    /// is taken).
    Fallback {
        /// The fallback assistant message produced by the non-streaming
        /// request.
        ///
        /// Built from the typed
        /// [`NonStreamingResponse`](crate::api::NonStreamingResponse) returned
        /// by [`create_message`](crate::api::ApiClient::create_message). The
        /// engine should treat this as the authoritative turn output — the
        /// streaming accumulator's partial state from failed attempts is
        /// discarded on this path.
        message: Message,

        /// Stop reason mapped from the provider's native finish/stop field.
        ///
        /// Defaults to [`EndTurn`](StreamStopReason::EndTurn) when the
        /// field is absent or holds an unrecognized value, so the engine
        /// always has a concrete reason to act on. Drives the same
        /// downstream behaviour as a streaming `MessageStop`.
        stop_reason: StreamStopReason,

        /// Token usage reported by the provider for the fallback request.
        ///
        /// `None` when the provider omits usage from its non-streaming
        /// response. The engine threads this into the turn's usage totals
        /// exactly like the `MessageDelta` usage on the streaming path.
        usage: Option<Usage>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only result shape matching the old `StreamTurnResult`, used by
    /// [`StreamHandler::drive_turn`] to keep existing tests' assertions working.
    #[derive(Debug)]
    #[allow(dead_code)]
    struct DriveResult {
        message: Message,
        usage: Option<Usage>,
        stop_reason: StreamStopReason,
        from_fallback: bool,
    }

    impl StreamHandler {
        /// Drive `stream_turn` to completion and return the assembled
        /// `(Message, Option<Usage>, StreamStopReason, from_fallback)` tuple —
        /// the same shape `stream_turn` used to return directly. Mirrors how the
        /// engine consumes the stream: events accumulate, `AttemptReset`
        /// discards partial state, `Fallback` short-circuits with the fallback
        /// message.
        async fn drive_turn<C: ApiClient>(
            &self,
            client: &C,
            request: &crate::api::StreamRequest,
            cancel: &Arc<CancelSignal>,
        ) -> Result<DriveResult, StreamHandlerError> {
            let mut stream = self.stream_turn(
                client,
                request,
                crate::structured::RequestOptions::default(),
                cancel,
            );
            let mut accumulator = StreamAccumulator::new();
            let mut stop_reason = StreamStopReason::EndTurn;
            let mut from_fallback = false;
            while let Some(item) = stream.next().await {
                match item? {
                    HandlerEvent::Stream(ev) => {
                        if let StreamEvent::MessageDelta(delta) = &ev
                            && let Some(ref reason_str) = delta.delta.stop_reason
                        {
                            stop_reason =
                                StreamStopReason::from_api_str(reason_str).unwrap_or(stop_reason);
                        }
                        accumulator.process(&ev).map_err(|e| {
                            StreamHandlerError::StreamFailed(StreamOutcome::InitFailed {
                                attempts: 1,
                                last_error: e.to_string(),
                            })
                        })?;
                    }
                    HandlerEvent::AttemptReset => {
                        accumulator = StreamAccumulator::new();
                        stop_reason = StreamStopReason::EndTurn;
                    }
                    HandlerEvent::Fallback {
                        message,
                        stop_reason: fallback_stop_reason,
                        usage: fallback_usage,
                    } => {
                        from_fallback = true;
                        return Ok(DriveResult {
                            message,
                            usage: fallback_usage,
                            stop_reason: fallback_stop_reason,
                            from_fallback,
                        });
                    }
                }
            }
            let usage = accumulator.usage().copied();
            Ok(DriveResult {
                message: accumulator.build(),
                usage,
                stop_reason,
                from_fallback,
            })
        }
    }

    #[test]
    fn timeout_config_default_values() {
        let config = StreamTimeoutConfig::default();
        assert_eq!(config.initial_event_timeout, Duration::from_mins(2));
        assert_eq!(config.per_event_timeout, Duration::from_mins(3));
        assert_eq!(config.total_stream_timeout, Duration::from_mins(5));
        assert_eq!(config.max_consecutive_timeouts, 10);
        assert!(config.fallback_to_non_streaming);
    }

    #[test]
    fn passthrough_sets_no_resilience_config() {
        let h = StreamHandler::passthrough();
        assert_eq!(h.timeout_config().initial_event_timeout, Duration::MAX);
        assert_eq!(h.timeout_config().per_event_timeout, Duration::MAX);
        assert_eq!(h.timeout_config().total_stream_timeout, Duration::MAX);
        assert!(!h.timeout_config().fallback_to_non_streaming);
        assert_eq!(h.retry_config().max_retries, 0);
        assert_eq!(h.rate_limit_config().max_retries, 0);
        assert_eq!(h.rate_limit_config().fallback_after_retries, 0);
    }

    #[test]
    fn passthrough_default_returns_shared_static() {
        let a = StreamHandler::passthrough_default();
        let b = StreamHandler::passthrough_default();
        assert!(
            std::ptr::eq(a, b),
            "passthrough_default must return the same static"
        );
    }

    #[test]
    fn timeout_config_custom_values() {
        let config = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(30),
            per_event_timeout: Duration::from_mins(1),
            total_stream_timeout: Duration::from_mins(5),
            max_consecutive_timeouts: 5,
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
        assert_eq!(config.base_delay(3), Duration::from_secs(5));
    }

    #[test]
    fn jittered_base_delay_zero_jitter_equals_raw() {
        let config = StreamRetryConfig {
            jitter_factor: 0.0,
            ..Default::default()
        };
        for attempt in 0..5 {
            assert_eq!(
                config.jittered_base_delay(attempt),
                config.base_delay(attempt),
                "zero jitter must reproduce the raw backoff exactly"
            );
        }
    }

    #[test]
    fn jittered_base_delay_stays_within_jitter_band() {
        let config = StreamRetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            jitter_factor: 0.2,
            ..Default::default()
        };
        for attempt in 0..64 {
            let base = config.base_delay(attempt);
            let delay = config.jittered_base_delay(attempt);
            let lo = base.mul_f64(0.8);
            let hi = base.mul_f64(1.2);
            assert!(
                delay >= lo && delay <= hi,
                "attempt {attempt}: jittered delay {delay:?} outside [{lo:?}, {hi:?}]"
            );
        }
    }

    #[test]
    fn jittered_base_delay_concurrent_calls_produce_different_delays() {
        let config = StreamRetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            jitter_factor: 0.5,
            ..Default::default()
        };
        let attempt = 1;
        let mut delays: Vec<_> = (0..10)
            .map(|_| config.jittered_base_delay(attempt))
            .collect();
        delays.sort();
        delays.dedup();
        assert!(
            delays.len() > 1,
            "concurrent calls with the same attempt must produce varied delays"
        );
    }

    #[test]
    fn jittered_base_delay_max_jitter_stays_non_negative() {
        let config = StreamRetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            jitter_factor: 1.0,
            ..Default::default()
        };
        for attempt in 0..256 {
            let delay = config.jittered_base_delay(attempt);
            let hi = config.base_delay(attempt).mul_f64(2.0);
            assert!(
                delay <= hi,
                "attempt {attempt}: delay {delay:?} exceeds 2x base under max jitter"
            );
        }
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
        assert!(
            !s.contains("init failed"),
            "the historical variant name must not leak into the rendered \
             message — a mid-stream truncation is not an init failure: {s}"
        );
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
        assert!(
            s.contains("stream failed before completing"),
            "the historical variant name must not leak into the message: {s}"
        );
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
        let err = StreamHandlerError::RateLimitEscalation {
            attempts: 4,
            retry_after: Some(Duration::from_secs(5)),
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
    fn handler_new_defaults() {
        let handler = StreamHandler::new();
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_mins(2),
        );
        assert_eq!(handler.retry_config().max_retries, 3);
    }

    #[test]
    fn handler_with_timeout_and_retry_config() {
        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_mins(1),
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 5,
                ..Default::default()
            });
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
    fn timeout_config_validate_rejects_infinite_total_timeout() {
        let config = StreamTimeoutConfig {
            total_stream_timeout: Duration::MAX,
            ..Default::default()
        };
        let err = config
            .validate()
            .expect_err("Duration::MAX must be rejected");
        assert!(
            err.contains("finite"),
            "the error must name the silent-disable hazard: {err}"
        );
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

    use crate::api::error::ApiError;
    use crate::stream::{
        DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
        PartStart, StreamEvent, Usage,
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
            Ok(StreamEvent::PartStop { index: None }),
            Ok(StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".to_string()),
                },
                usage: None,
            })),
            Ok(StreamEvent::MessageStop),
        ]
    }

    struct HandlerMock {
        create_error: Option<String>,
        create_response: Option<Message>,
    }

    impl HandlerMock {
        fn new() -> Self {
            Self {
                create_error: None,
                create_response: None,
            }
        }

        fn with_text_response(mut self, text: &str) -> Self {
            self.create_response = Some(Message::assistant(text));
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
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            // Default: return a happy-path stream.
            Box::pin(futures::stream::iter(happy_stream_events()))
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            if let Some(ref err) = self.create_error {
                let err = err.clone();
                return Box::pin(async move { Err(ApiError::api(&err)) });
            }
            let message = self
                .create_response
                .clone()
                .unwrap_or_else(|| Message::assistant("default"));
            Box::pin(async move {
                Ok(crate::api::NonStreamingResponse {
                    message,
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    #[tokio::test]
    async fn fallback_non_streaming_success() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let client = HandlerMock::new().with_text_response("fallback works");
        let cancel = Arc::new(CancelSignal::new());

        let (message, stop_reason, usage) = handler
            .fallback_non_streaming(
                &client,
                &crate::api::StreamRequest::new(vec![]),
                &crate::structured::RequestOptions::default(),
                &cancel,
                None,
                Some(StreamOutcome::InitFailed {
                    last_error: "stream failed".to_string(),
                    attempts: 3,
                }),
            )
            .await
            .expect("fallback should succeed");

        // The fallback returns a Message built from the first text part of the
        // non-streaming JSON response, plus the stop_reason from the JSON.
        let text: String = message
            .parts
            .iter()
            .filter_map(|p| match p {
                crate::stream::MessagePart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("fallback works"), "got: {text:?}");
        // HandlerMock::with_text_response sets stop_reason: "end_turn".
        assert_eq!(stop_reason, StreamStopReason::EndTurn);
        // HandlerMock returns Usage::default() (zero tokens).
        assert_eq!(usage, Some(Usage::default()));
    }

    #[tokio::test]
    async fn fallback_non_streaming_cancelled_before_start() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let client = HandlerMock::new().with_text_response("fallback works");
        let cancel = Arc::new(CancelSignal::new());
        cancel.cancel();

        let err = handler
            .fallback_non_streaming(
                &client,
                &crate::api::StreamRequest::new(vec![]),
                &crate::structured::RequestOptions::default(),
                &cancel,
                None,
                None,
            )
            .await
            .expect_err("should fail on cancellation");

        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "expected Cancelled, got: {err}"
        );
    }

    /// Mock recording every `create_message_with_options` invocation, so the
    /// fallback's options forwarding is observable.
    struct OptionsRecordingMock {
        seen: std::sync::Mutex<Vec<crate::structured::RequestOptions>>,
    }

    impl ApiClient for OptionsRecordingMock {
        fn model(&self) -> String {
            "test-model".to_string()
        }

        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: Message::assistant("unused"),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }

        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            options: crate::structured::RequestOptions,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            self.seen.lock().unwrap().push(options);
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: Message::assistant("fallback works"),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    #[tokio::test]
    async fn fallback_non_streaming_forwards_request_options() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let client = OptionsRecordingMock {
            seen: std::sync::Mutex::new(Vec::new()),
        };
        let cancel = Arc::new(CancelSignal::new());

        let mut options = crate::structured::RequestOptions::default();
        options.response_format = Some(crate::structured::ResponseFormat::new(
            "probe",
            serde_json::json!({"type": "object"}),
        ));

        let (message, _stop, _usage) = handler
            .fallback_non_streaming(
                &client,
                &crate::api::StreamRequest::new(vec![]),
                &options,
                &cancel,
                None,
                Some(StreamOutcome::InitFailed {
                    last_error: "stream failed".to_string(),
                    attempts: 1,
                }),
            )
            .await
            .expect("fallback should succeed");

        assert!(
            message.text_content().contains("fallback works"),
            "the options-aware response is the one used"
        );
        let seen = client.seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "exactly one options-aware call");
        assert!(
            seen[0]
                .response_format
                .as_ref()
                .is_some_and(|format| format.name == "probe"),
            "the fallback must receive the turn's RequestOptions verbatim"
        );
    }

    /// Mock whose non-streaming call never resolves — the deadline arm must
    /// win.
    struct HangingFallbackMock;

    impl ApiClient for HangingFallbackMock {
        fn model(&self) -> String {
            "test-model".to_string()
        }

        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(std::future::pending())
        }
    }

    #[tokio::test]
    async fn fallback_non_streaming_honors_the_total_deadline() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let deadline = Instant::now() + Duration::from_millis(10);

        let err = handler
            .fallback_non_streaming(
                &HangingFallbackMock,
                &crate::api::StreamRequest::new(vec![]),
                &crate::structured::RequestOptions::default(),
                &cancel,
                Some(deadline),
                None,
            )
            .await
            .expect_err("a hanging fallback must be cut by the deadline");

        match err {
            StreamHandlerError::FallbackFailed { fallback_error, .. } => {
                assert!(
                    fallback_error.contains("deadline"),
                    "the deadline arm must be the failure cause: {fallback_error}"
                );
            }
            other => panic!("expected FallbackFailed, got: {other}"),
        }
    }

    #[tokio::test]
    async fn completed_fallback_response_racing_the_deadline_is_accepted() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let client = HandlerMock::new().with_text_response("worth keeping");
        let cancel = Arc::new(CancelSignal::new());
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("a past instant");

        let (message, _stop_reason, _usage) = handler
            .fallback_non_streaming(
                &client,
                &crate::api::StreamRequest::new(vec![]),
                &crate::structured::RequestOptions::default(),
                &cancel,
                Some(deadline),
                None,
            )
            .await
            .expect("a completed response outranks the expired deadline");
        assert!(
            message.text_content().contains("worth keeping"),
            "the completed response is returned, not discarded"
        );
    }

    #[tokio::test]
    async fn fallback_non_streaming_error() {
        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..Default::default()
        });
        let client = HandlerMock::new().with_create_error("service unavailable");
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .fallback_non_streaming(
                &client,
                &crate::api::StreamRequest::new(vec![]),
                &crate::structured::RequestOptions::default(),
                &cancel,
                None,
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
    async fn stream_turn_yields_handler_event_stream_per_event() {
        // Each raw stream event must arrive as a HandlerEvent::Stream, in
        // arrival order, when driving the handler as a stream.
        let handler = StreamHandler::new();
        let client = HandlerMock::new().with_text_response("hello");
        let cancel = Arc::new(CancelSignal::new());

        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &client,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut saw_stream_events = 0;
        let mut saw_attempt_reset = false;
        let mut saw_fallback = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item ok") {
                HandlerEvent::Stream(_) => saw_stream_events += 1,
                HandlerEvent::AttemptReset => saw_attempt_reset = true,
                HandlerEvent::Fallback { .. } => saw_fallback = true,
            }
        }
        assert!(saw_stream_events > 0, "should yield Stream events");
        assert!(!saw_attempt_reset, "happy path must not emit AttemptReset");
        assert!(!saw_fallback, "happy path must not emit Fallback");
    }

    #[tokio::test]
    async fn empty_stream_fast_fails_after_lower_threshold() {
        struct NeverYieldingMock;
        impl ApiClient for NeverYieldingMock {
            fn model(&self) -> String {
                "stuck".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::pending())
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_millis(10),
                per_event_timeout: Duration::from_millis(10),
                total_stream_timeout: Duration::from_secs(10),
                max_consecutive_timeouts: 10,
                fallback_to_non_streaming: false,
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            });
        let client = NeverYieldingMock;
        let cancel = Arc::new(CancelSignal::new());
        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &client,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let start = Instant::now();
        let mut got = None;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                got = Some(item);
                break;
            }
        }
        let elapsed = start.elapsed();
        match got.expect("stream must terminate with an error") {
            Err(StreamHandlerError::StreamFailed(StreamOutcome::EventTimeout { .. })) => {}
            other => panic!("expected EventTimeout on dead stream, got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_millis(60),
            "empty-stream fast-fail (2×10ms) must beat the full threshold (10×10ms); \
             elapsed {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn fallback_preserves_tool_call_parts() {
        struct ToolFallbackMock;
        impl ApiClient for ToolFallbackMock {
            fn model(&self) -> String {
                "test".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::api("connection refused"))
                }))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::new(
                            crate::message::Role::Assistant,
                            vec![
                                crate::message::MessagePart::text("Let me search"),
                                crate::message::MessagePart::tool_call(
                                    "tc_1",
                                    "search",
                                    serde_json::json!({"q": "hello"}),
                                ),
                            ],
                        ),
                        stop_reason: crate::stream::StreamStopReason::ToolCall,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: true,
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &ToolFallbackMock,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut got_fallback = false;
        while let Some(item) = stream.next().await {
            if let Ok(HandlerEvent::Fallback { message, .. }) = item {
                got_fallback = true;
                let has_tool = message
                    .parts
                    .iter()
                    .any(|p| matches!(p, crate::message::MessagePart::ToolCall { name, .. } if name == "search"));
                assert!(
                    has_tool,
                    "fallback message must preserve the tool-call part, got: {:?}",
                    message.parts
                );
                let has_text = message
                    .parts
                    .iter()
                    .any(|p| matches!(p, crate::message::MessagePart::Text { text } if text == "Let me search"));
                assert!(has_text, "fallback message must preserve the text part");
            }
        }
        assert!(got_fallback, "must emit a Fallback event");
    }

    #[tokio::test]
    async fn rate_limit_hard_stop_tries_fallback_when_enabled() {
        struct RateLimitThenOkMock;
        impl ApiClient for RateLimitThenOkMock {
            fn model(&self) -> String {
                "test".to_string()
            }

            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant("fallback ok"),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 2,
            max_retries: 2,
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let req = crate::api::StreamRequest::new(vec![]);
        let result = handler
            .drive_turn(&RateLimitThenOkMock, &req, &cancel)
            .await;
        assert!(
            result.is_ok(),
            "hard-stop must try fallback when enabled, got: {:?}",
            result.err()
        );
        let drive = result.unwrap();
        assert!(drive.from_fallback);
        assert!(drive.message.text_content().contains("fallback ok"));
    }

    /// Mock that fails its first streaming attempt with a transport error,
    /// then succeeds on the second. Used by the AttemptReset test to verify
    /// the handler emits `AttemptReset` before the retried attempt's events.
    struct RetryingMock {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ApiClient for RetryingMock {
        fn model(&self) -> String {
            "retry-test".to_string()
        }
        fn base_url(&self) -> String {
            "retry-test".to_string()
        }
        fn set_model(&self, _: &str) -> bool {
            false
        }
        fn stream_messages(
            &self,
            request: &crate::api::StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            self.stream_messages_with_options(request, crate::structured::RequestOptions::default())
        }
        fn create_message(
            &self,
            request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            self.create_message_with_options(request, crate::structured::RequestOptions::default())
        }
        fn stream_messages_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: crate::structured::RequestOptions,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            use std::sync::atomic::Ordering;
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                // First attempt: one event then transport error.
                Box::pin(futures::stream::iter(vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: String::new(),
                            role: "assistant".into(),
                            model: String::new(),
                        },
                    })),
                    Err(ApiError::api("transient")),
                ]))
            } else {
                // Second attempt: clean happy-path events.
                Box::pin(futures::stream::iter(happy_stream_events()))
            }
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: crate::structured::RequestOptions,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
        fn extract_structured(&self, _: &crate::message::Message) -> serde_json::Value {
            serde_json::Value::Null
        }
    }

    #[tokio::test]
    async fn stream_turn_yields_attempt_reset_on_retry() {
        // A retried transport failure must emit AttemptReset before the
        // retried attempt's events. Engine uses this to discard partial state.
        use std::sync::atomic::AtomicUsize;

        let attempts = Arc::new(AtomicUsize::new(0));
        let handler = StreamHandler::new();
        let client = RetryingMock { attempts };
        let cancel = Arc::new(CancelSignal::new());

        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &client,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut saw_attempt_reset = false;
        while let Some(item) = stream.next().await {
            if let HandlerEvent::AttemptReset = item.expect("stream item ok") {
                saw_attempt_reset = true;
            }
        }
        assert!(
            saw_attempt_reset,
            "second attempt must be preceded by AttemptReset"
        );
    }

    #[tokio::test]
    async fn stream_turn_happy_path() {
        let handler = StreamHandler::new();
        let client = HandlerMock::new().with_text_response("hello world");
        let cancel = Arc::new(CancelSignal::new());

        let result = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::api("API down"))
                }))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unreachable")) })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            });

        let client = ErrorMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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

    /// Mock that always fails streaming but succeeds on non-streaming
    /// `create_message`. Used by the fallback regression test to verify the
    /// handler yields `HandlerEvent::Fallback` with the message and stop_reason
    /// extracted from the JSON response.
    struct StreamingFailingFallbackMock;
    impl ApiClient for StreamingFailingFallbackMock {
        fn model(&self) -> String {
            "fallback-test".to_string()
        }
        fn base_url(&self) -> String {
            "fallback-test".to_string()
        }
        fn set_model(&self, _: &str) -> bool {
            false
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            // Always fail streaming — forces the handler to retry, then
            // fall back to create_message.
            Box::pin(futures::stream::once(async {
                Err(ApiError::api("stream down"))
            }))
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant("fallback answer"),
                    stop_reason: crate::stream::StreamStopReason::MaxTokens,
                    usage: Some(crate::stream::Usage::new(42, 13)),
                })
            })
        }
        fn stream_messages_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: crate::structured::RequestOptions,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            Box::pin(futures::stream::once(async {
                Err(ApiError::api("stream down"))
            }))
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: crate::structured::RequestOptions,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant("fallback answer"),
                    stop_reason: crate::stream::StreamStopReason::MaxTokens,
                    usage: Some(crate::stream::Usage::new(42, 13)),
                })
            })
        }
        fn extract_structured(&self, _: &crate::message::Message) -> serde_json::Value {
            serde_json::Value::Null
        }
    }

    #[tokio::test]
    async fn drive_turn_returns_fallback_message_and_stop_reason() {
        // When streaming fails after retries and fallback is enabled, the
        // engine should see the fallback message and the stop_reason from the
        // non-streaming JSON response (not the streaming accumulator's stale
        // values). Regression test for an earlier bug where Fallback dropped
        // stop_reason.
        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: true,
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 0,
                ..Default::default()
            });
        let client = StreamingFailingFallbackMock;
        let cancel = Arc::new(CancelSignal::new());

        let result = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect("fallback should succeed");

        assert!(
            result.from_fallback,
            "result should be marked from_fallback"
        );
        // Stop reason comes from the JSON's "max_tokens", not the streaming
        // default "end_turn".
        assert_eq!(
            result.stop_reason,
            StreamStopReason::MaxTokens,
            "fallback stop_reason must come from the JSON response"
        );
        let text: String = result
            .message
            .parts
            .iter()
            .filter_map(|p| match p {
                crate::stream::MessagePart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            text.contains("fallback answer"),
            "fallback message text, got {text:?}"
        );
        assert_eq!(
            result.usage,
            Some(Usage::new(42, 13)),
            "fallback path must propagate usage from the non-streaming response"
        );
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
    fn with_timeout_config_substitutes_only_invalid_fields() {
        let bad_total = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(45),
            per_event_timeout: Duration::from_secs(45),
            total_stream_timeout: Duration::MAX,
            max_consecutive_timeouts: 7,
            ..Default::default()
        };
        let handler = StreamHandler::new().with_timeout_config(bad_total);
        let config = handler.timeout_config();
        assert_eq!(
            config.initial_event_timeout,
            Duration::from_secs(45),
            "valid fields the caller supplied must survive an invalid sibling"
        );
        assert_eq!(config.per_event_timeout, Duration::from_secs(45));
        assert_eq!(config.max_consecutive_timeouts, 7);
        assert_eq!(
            config.total_stream_timeout,
            StreamTimeoutConfig::default().total_stream_timeout,
            "an infinite total timeout is substituted with the default, not silently disabling every deadline"
        );

        let unordered = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(400),
            ..Default::default()
        };
        let handler = StreamHandler::new().with_timeout_config(unordered);
        assert_eq!(
            handler.timeout_config().total_stream_timeout,
            Duration::from_secs(400),
            "a default total below a custom initial timeout is raised to it, \
             keeping the caller's initial customization"
        );
    }

    #[test]
    fn sanitized_config_always_validates() {
        let adversarial = [
            StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(600),
                total_stream_timeout: Duration::ZERO,
                ..Default::default()
            },
            StreamTimeoutConfig {
                initial_event_timeout: Duration::MAX,
                ..Default::default()
            },
            StreamTimeoutConfig {
                per_event_timeout: Duration::MAX,
                ..Default::default()
            },
            StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(45),
                per_event_timeout: Duration::from_secs(45),
                total_stream_timeout: Duration::MAX,
                max_consecutive_timeouts: 7,
                ..Default::default()
            },
        ];
        for config in adversarial {
            let handler = StreamHandler::new().with_timeout_config(config);
            assert!(
                handler.timeout_config().validate().is_ok(),
                "the sanitized builder output must satisfy every validate rule: {:?}",
                handler.timeout_config()
            );
        }
        let zero_total = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(600),
            total_stream_timeout: Duration::ZERO,
            ..Default::default()
        });
        assert_eq!(
            zero_total.timeout_config().total_stream_timeout,
            Duration::from_secs(600),
            "a repaired total must still honor the ordering rule against a large initial"
        );
    }

    #[test]
    fn handler_error_display_never_says_init_failed() {
        let outcome = StreamOutcome::InitFailed {
            last_error: "stream ended without a terminal event".to_string(),
            attempts: 3,
        };
        let rendered = StreamHandlerError::InitFailed(outcome).to_string();
        assert!(
            rendered.contains("without a terminal event"),
            "the wrapper names the failure cause: {rendered}"
        );
        assert!(
            !rendered.contains("init failed"),
            "the historical variant name must not leak into the rendered message: {rendered}"
        );
    }

    #[test]
    fn with_timeout_config_keeps_valid() {
        let good = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(45),
            ..Default::default()
        };
        let handler = StreamHandler::new().with_timeout_config(good);
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_secs(45)
        );
    }

    #[test]
    fn with_retry_config_rejects_invalid_falls_back_to_default() {
        let bad = StreamRetryConfig {
            base_delay_ms: 0,
            ..Default::default()
        };
        let handler = StreamHandler::new().with_retry_config(bad);
        assert_eq!(
            handler.retry_config().base_delay_ms,
            StreamRetryConfig::default().base_delay_ms,
            "invalid retry config must fall back to default"
        );
    }

    #[test]
    fn with_retry_config_keeps_valid() {
        let good = StreamRetryConfig {
            max_retries: 7,
            ..Default::default()
        };
        let handler = StreamHandler::new().with_retry_config(good);
        assert_eq!(handler.retry_config().max_retries, 7);
    }

    #[test]
    fn with_timeout_and_retry_config_are_independent() {
        let good_timeout = StreamTimeoutConfig {
            initial_event_timeout: Duration::from_mins(1),
            ..Default::default()
        };
        let bad_retry = StreamRetryConfig {
            jitter_factor: 2.0,
            ..Default::default()
        };
        let handler = StreamHandler::new()
            .with_timeout_config(good_timeout)
            .with_retry_config(bad_retry);
        assert_eq!(
            handler.timeout_config().initial_event_timeout,
            Duration::from_mins(1),
            "valid timeout must be kept when retry config is invalid"
        );
        assert_eq!(
            handler.retry_config().max_retries,
            StreamRetryConfig::default().max_retries,
            "invalid retry config must fall back to default"
        );
    }

    #[test]
    fn with_rate_limit_config_rejects_invalid_falls_back_to_default() {
        let bad = RateLimitConfig {
            max_retries: 0,
            ..Default::default()
        };
        let handler = StreamHandler::new().with_rate_limit_config(bad);
        assert_eq!(
            handler.rate_limit_config().max_retries,
            RateLimitConfig::default().max_retries,
            "invalid rate-limit config must fall back to default"
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
        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            fallback_after_retries: 1,
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
    fn rate_limit_retry_max_retries_should_be_enforced_under_valid_config() {
        let handler = StreamHandler::new();
        let detail = detected_limit(None);
        let mut count = 0u32;

        for _ in 0..(handler.rate_limit_config().max_retries + 2) {
            let _ = handler.rate_limit_retry(&detail, &mut count, None);
        }
        let max = handler.rate_limit_config().max_retries;
        assert!(
            count > max,
            "count {count} must exceed max_retries {max} after enough calls"
        );
        let decision = handler.rate_limit_retry(&detail, &mut count, None);
        assert!(
            matches!(decision, RateLimitRetry::HardStop),
            "max_retries={max} should be enforced as a hard ceiling, \
             but Escalate shadows it — HardStop is dead code under valid config"
        );
    }

    #[test]
    fn with_rate_limit_config_should_reject_invalid() {
        let invalid = RateLimitConfig {
            fallback_after_retries: 10,
            max_retries: 3,
            ..Default::default()
        };
        let result = StreamHandler::new().with_rate_limit_config(invalid);
        let detail = detected_limit(None);
        let mut count = 0u32;
        for _ in 0..4 {
            let _ = result.rate_limit_retry(&detail, &mut count, None);
        }
        let decision = result.rate_limit_retry(&detail, &mut count, None);
        assert!(
            !matches!(decision, RateLimitRetry::HardStop),
            "invalid config (fallback_after=10 > max_retries=3) must not \
             silently invert behavior — HardStop should never fire before Escalation"
        );
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
            fallback_after_retries: 1,
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
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::iter(happy_stream_events()))
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
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
        let limiter = Arc::new(RateLimiter::new(1));
        let handler = StreamHandler::new()
            .with_rate_limiter(Arc::clone(&limiter))
            .with_rate_limit_max_wait(Duration::from_mins(2));
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
        let limiter = Arc::new(RateLimiter::new(1));
        let handler = StreamHandler::new()
            .with_rate_limiter(Arc::clone(&limiter))
            .with_rate_limit_max_wait(Duration::from_mins(2));
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
        let handler = StreamHandler::new()
            .with_rate_limiter(Arc::clone(&limiter))
            .with_rate_limit_max_wait(Duration::from_mins(2));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        let start = Instant::now();
        for _ in 0..3 {
            handler
                .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
        let limiter = Arc::new(RateLimiter::new(1));
        let handler = StreamHandler::new()
            .with_rate_limiter(limiter)
            .with_rate_limit_max_wait(Duration::from_millis(50));
        let client = GateMock { url: "openai" };

        // First turn consumes the only token.
        let cancel = Arc::new(CancelSignal::new());
        handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect("first turn should succeed");

        // Cancel before the second turn — it will need to wait ~60s for a token.
        let cancel2 = Arc::new(CancelSignal::new());
        cancel2.cancel();
        let start = Instant::now();
        let err = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel2)
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
        let limiter = Arc::new(RateLimiter::new(1));
        let handler = StreamHandler::new()
            .with_rate_limiter(limiter)
            .with_rate_limit_max_wait(Duration::from_millis(50));
        let client = GateMock { url: "openai" };
        let cancel = Arc::new(CancelSignal::new());

        // First turn consumes the only token.
        handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect("first turn should succeed");

        // Second turn: bucket empty, wait capped at 50ms, then proceeds.
        let start = Instant::now();
        let result = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        // Rate-limit delay tiny; transport retry delay large. If the retry
        // loop honours the rate-limit outcome, the test finishes in ~1ms; if
        // it falls back to the transport base_delay, it sleeps 2s.
        let handler = StreamHandler::new().with_retry_config(StreamRetryConfig {
            max_retries: 1,
            base_delay_ms: 2_000,
            ..Default::default()
        });
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
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
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
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect_err("should escalate, not succeed");
        match err {
            StreamHandlerError::RateLimitEscalation {
                attempts,
                retry_after,
            } => {
                // fallback_after_retries == 2, so escalation fires on the 3rd hit.
                assert_eq!(attempts, 3);
                assert_eq!(retry_after, Some(Duration::from_millis(1)));
            }
            other => panic!("expected RateLimitEscalation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn default_rate_limit_config_escalates_without_a_non_streaming_attempt() {
        struct Counting429Mock {
            non_streaming_calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for Counting429Mock {
            fn model(&self) -> String {
                "test".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                self.non_streaming_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant("fallback ok"),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let client = Counting429Mock {
            non_streaming_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let handler = StreamHandler::new().with_rate_limit_config(RateLimitConfig {
            default_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let err = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect_err("the default ladder escalates rather than exhausting");
        assert!(
            matches!(err, StreamHandlerError::RateLimitEscalation { .. }),
            "the default ladder (fallback_after=3 < max=5) escalates to the model \
             breaker, got: {err:?}"
        );
        assert_eq!(
            client
                .non_streaming_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a rate limit is charged against the model's quota — a same-model \
             non-streaming request is deliberately not attempted (the ceiling-equal \
             ladder opts into it)"
        );
    }

    #[tokio::test]
    async fn rate_limit_after_partial_data_reports_has_partial_data() {
        struct PartialThenRateLimitMock;
        impl ApiClient for PartialThenRateLimitMock {
            fn model(&self) -> String {
                "partial-then-429".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "partial-then-429".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::text("")),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStop { index: Some(0) }),
                    Err(ApiError::RateLimit {
                        retry_after: None,
                        message: "slow down".into(),
                    }),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unused")) })
            }
        }

        let handler = StreamHandler::new()
            .with_rate_limit_config(RateLimitConfig {
                fallback_after_retries: 1,
                max_retries: 1,
                default_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let err = handler
            .drive_turn(
                &PartialThenRateLimitMock,
                &crate::api::StreamRequest::new(vec![]),
                &cancel,
            )
            .await
            .expect_err("the disabled fallback makes the hard stop terminal");
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited {
                has_partial_data,
                events_processed,
                ..
            }) => {
                assert!(
                    has_partial_data,
                    "a 429 after accepted events must report salvageable partial data"
                );
                assert_eq!(
                    events_processed, 4,
                    "the outcome counts the events that got through before the 429"
                );
            }
            other => panic!("expected a RateLimited terminal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn event_timeout_after_partial_data_reports_has_partial_data() {
        struct PartialThenHangMock;
        impl ApiClient for PartialThenHangMock {
            fn model(&self) -> String {
                "partial-then-hang".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "partial-then-hang".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::text("")),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStop { index: Some(0) }),
                ];
                let pending = futures::stream::pending();
                Box::pin(futures::stream::iter(events).chain(pending))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unused")) })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_millis(50),
            per_event_timeout: Duration::from_millis(50),
            max_consecutive_timeouts: 1,
            fallback_to_non_streaming: false,
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let err = handler
            .drive_turn(
                &PartialThenHangMock,
                &crate::api::StreamRequest::new(vec![]),
                &cancel,
            )
            .await
            .expect_err("the hang must terminate via the event timeout");
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::EventTimeout {
                has_partial_data,
                consecutive_timeouts,
            }) => {
                assert!(
                    has_partial_data,
                    "a hang after accepted events must report salvageable partial data"
                );
                assert_eq!(consecutive_timeouts, 1);
            }
            other => panic!("expected an EventTimeout terminal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn retried_attempt_re_gates_on_the_rate_limiter() {
        struct FailOnceThenAnswerMock {
            calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for FailOnceThenAnswerMock {
            fn model(&self) -> String {
                "fail-once".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let calls = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if calls == 0 {
                    return Box::pin(futures::stream::once(async {
                        Err(ApiError::http("transient transport failure"))
                    }));
                }
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "fail-once".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::text("")),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "recovered".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStop { index: Some(0) }),
                    Ok(StreamEvent::MessageStop),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unused")) })
            }
        }

        let handler = StreamHandler::new()
            .with_rate_limiter(Arc::new(crate::stream::rate_limit::RateLimiter::new(1)))
            .with_rate_limit_max_wait(Duration::from_millis(1200))
            .with_retry_config(crate::stream::handler::StreamRetryConfig {
                base_delay_ms: 1,
                max_delay_ms: 1,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let client = FailOnceThenAnswerMock {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let started = std::time::Instant::now();
        handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect("the retried attempt must succeed");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(1000),
            "the retried attempt must re-gate on the limiter and wait out the max-wait \
             ceiling (1 rpm = the first attempt drains the bucket); elapsed {elapsed:?}"
        );
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 1,
                ..Default::default()
            })
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
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            })
            .with_rate_limit_config(RateLimitConfig {
                fallback_after_retries: 2,
                max_retries: 2,
                default_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
                ..Default::default()
            });
        let client = AlwaysRateLimitMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect_err("hard-stop should fail the turn");
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited { .. })
            | StreamHandlerError::InitFailed(StreamOutcome::RateLimited { .. }) => {}
            StreamHandlerError::RateLimitEscalation { .. } => {
                panic!("escalation must not fire when max_retries == fallback_after_retries")
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
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
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await
            .expect("first call should succeed after one rate-limit retry");

        // Second call on the SAME handler: fresh counter, so it must succeed too
        // rather than escalating on a leaked count.
        let client2 = RateLimitOnceMock {
            attempts: AtomicUsize::new(0),
        };
        handler
            .drive_turn(&client2, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::api("connection refused"))
                }))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                fallback_to_non_streaming: false,
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 1,
                ..Default::default()
            });
        let client = AlwaysFailingMock;
        let cancel = Arc::new(CancelSignal::new());

        let err = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
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
                _request: &crate::api::StreamRequest,
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_millis(40),
                per_event_timeout: Duration::from_millis(40),
                total_stream_timeout: Duration::from_millis(80),
                ..Default::default()
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 10,
                base_delay_ms: 1,
                ..Default::default()
            })
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
        let result = handler
            .drive_turn(&client, &crate::api::StreamRequest::new(vec![]), &cancel)
            .await;
        let elapsed = start.elapsed();
        // The clamp keeps the sleep inside the ~80ms budget, so the whole turn
        // resolves well under the 600s hint. Allow generous slack for scheduling.
        assert!(
            elapsed < Duration::from_secs(2),
            "deadline clamp should prevent a 600s sleep; elapsed {elapsed:?}",
        );
        // The expiry is terminal: with the fallback enabled by default the
        // turn resolves through it, otherwise it fails — never the 600s hint,
        // never escalation (the deadline trips before the counter ceiling).
        match result {
            Ok(done) => assert!(
                done.from_fallback,
                "a prompt success here can only be the non-streaming fallback"
            ),
            Err(err) => assert!(
                !matches!(err, StreamHandlerError::RateLimitEscalation { .. }),
                "timeout should fire before escalation"
            ),
        }
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
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::api("connection lost"))
                }))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new().with_retry_config(StreamRetryConfig {
            max_retries: 5,
            base_delay_ms: 60_000,
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let cancel_clone = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancel_clone.cancel();
        });

        let start = Instant::now();
        let err = handler
            .drive_turn(
                &AlwaysFailingMock,
                &crate::api::StreamRequest::new(vec![]),
                &cancel,
            )
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

    #[tokio::test]
    async fn malformed_event_fails_the_stream_with_attempt_context() {
        struct GarbageToolInputMock;
        impl ApiClient for GarbageToolInputMock {
            fn model(&self) -> String {
                "garbage-input".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "garbage-input".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::tool_call(
                            "t1",
                            "search",
                            serde_json::json!({}),
                        )),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::InputJson {
                            partial_json: "not json".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStop { index: None }),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http("unused")) })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(5),
            per_event_timeout: Duration::from_secs(5),
            total_stream_timeout: Duration::from_secs(60),
            max_consecutive_timeouts: 3,
            fallback_to_non_streaming: false,
        });
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &GarbageToolInputMock,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut yielded = 0usize;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::Stream(_)) => yielded += 1,
                Ok(_) => {}
                Err(e) => {
                    terminal = Some(e);
                    break;
                }
            }
        }
        assert_eq!(
            yielded, 12,
            "each ladder attempt replays the accepted events before the malformed one (4 attempts × 3)"
        );
        match terminal.expect("the malformed event must fail the stream") {
            StreamHandlerError::StreamFailed(StreamOutcome::InitFailed {
                attempts,
                last_error,
            }) => {
                assert_eq!(
                    attempts, 4,
                    "the failure counts every attempt the ladder made"
                );
                assert!(
                    last_error.contains("invalid tool input JSON"),
                    "the accumulator's rejection surfaces verbatim, got: {last_error}"
                );
            }
            other => panic!("expected a StreamFailed InitFailed terminal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn truncated_stream_is_not_a_completed_turn() {
        struct CutStreamMock;
        impl ApiClient for CutStreamMock {
            fn model(&self) -> String {
                "cut-stream".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "cut-stream".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::text("")),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::api("unused")) })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: false,
            ..Default::default()
        });
        let cancel = Arc::new(CancelSignal::new());
        let err = handler
            .drive_turn(
                &CutStreamMock,
                &crate::api::StreamRequest::new(vec![]),
                &cancel,
            )
            .await
            .expect_err("a stream that ends without a terminal event is truncated");
        let rendered = err.to_string();
        assert!(
            rendered.contains("without a terminal event") && !rendered.contains("init failed"),
            "the engine-facing message must name the truncation, not the \
             historical init framing: {rendered}"
        );
    }

    #[tokio::test]
    async fn truncated_stream_with_fallback_enabled_gets_the_ladder() {
        struct CutStreamThenAnswerMock;
        impl ApiClient for CutStreamThenAnswerMock {
            fn model(&self) -> String {
                "cut-then-answer".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "cut-then-answer".to_string(),
                        },
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant("fallback ok"),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &CutStreamThenAnswerMock,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut fallback_message = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::Fallback { message, .. }) => fallback_message = Some(message),
                Err(e) => panic!(
                    "a truncated stream with the fallback enabled must not fail the turn: {e}"
                ),
                _ => {}
            }
        }
        assert_eq!(
            fallback_message
                .expect("the non-streaming fallback must serve the truncated turn")
                .text_content(),
            "fallback ok"
        );
    }

    #[tokio::test]
    async fn malformed_event_with_fallback_enabled_gets_the_ladder() {
        struct GarbageThenAnswerMock;
        impl ApiClient for GarbageThenAnswerMock {
            fn model(&self) -> String {
                "garbage-then-answer".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "garbage-then-answer".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(crate::stream::MessagePart::tool_call(
                            "t1",
                            "search",
                            serde_json::json!({}),
                        )),
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::InputJson {
                            partial_json: "not json".to_string(),
                        },
                    })),
                    Ok(StreamEvent::PartStop { index: Some(0) }),
                ];
                Box::pin(futures::stream::iter(events))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: crate::message::Message::assistant("fallback ok"),
                        stop_reason: crate::stream::StreamStopReason::EndTurn,
                        usage: Some(crate::stream::Usage::default()),
                    })
                })
            }
        }

        let handler = StreamHandler::new();
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &GarbageThenAnswerMock,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut fallback_message = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::Fallback { message, .. }) => fallback_message = Some(message),
                Err(e) => panic!(
                    "a malformed event with the fallback enabled must not fail the turn: {e}"
                ),
                _ => {}
            }
        }
        assert_eq!(
            fallback_message
                .expect("the non-streaming fallback must serve the turn")
                .text_content(),
            "fallback ok",
            "exhausting the retry ladder on accumulation failures routes to the fallback"
        );
    }

    #[test]
    fn http_429_is_classified_as_rate_limited() {
        let detected =
            DetectedRateLimit::detect(&ApiError::http_with_status(429, "Too Many Requests"))
                .expect("429 must be detected as a rate limit");
        assert_eq!(
            detected.kind,
            RateLimitKind::RateLimited,
            "doc: RateLimited is the HTTP 429 Too Many Requests kind"
        );
    }

    #[test]
    fn rate_limit_variant_kind_splits_by_message_status() {
        let overload =
            DetectedRateLimit::detect(&ApiError::rate_limited("HTTP 503: unavailable", None))
                .expect("a 503-shaped RateLimit must be detected");
        assert!(
            matches!(overload.kind, RateLimitKind::Overloaded),
            "503 is the Overloaded kind, got {:?}",
            overload.kind
        );
        let overloaded_529 =
            DetectedRateLimit::detect(&ApiError::rate_limited("HTTP 529: overloaded", None))
                .expect("a 529-shaped RateLimit must be detected");
        assert!(matches!(overloaded_529.kind, RateLimitKind::Overloaded));
        let quota = DetectedRateLimit::detect(&ApiError::rate_limited("HTTP 429: slow down", None))
            .expect("a 429-shaped RateLimit must be detected");
        assert!(matches!(quota.kind, RateLimitKind::RateLimited));
        let untyped =
            DetectedRateLimit::detect(&ApiError::rate_limited("provider quota text", None))
                .expect("a statusless RateLimit must be detected");
        assert!(
            matches!(untyped.kind, RateLimitKind::RateLimited),
            "without an embedded status the default kind is RateLimited"
        );
    }

    /// A stream client whose every attempt fails with the given error.
    ///
    /// Builds the error per call from a factory (the error type is not
    /// `Clone`) and counts `stream_messages` / `create_message` calls so the
    /// permanent-error contracts can assert exactly how many attempts the
    /// handler spent before giving up.
    struct FailingStreamClient {
        make_error: fn() -> ApiError,
        stream_calls: std::sync::atomic::AtomicUsize,
        non_streaming_calls: std::sync::atomic::AtomicUsize,
    }

    impl FailingStreamClient {
        fn failing_with(make_error: fn() -> ApiError) -> Self {
            Self {
                make_error,
                stream_calls: std::sync::atomic::AtomicUsize::new(0),
                non_streaming_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn stream_calls(&self) -> usize {
            self.stream_calls.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn non_streaming_calls(&self) -> usize {
            self.non_streaming_calls
                .load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl ApiClient for FailingStreamClient {
        fn model(&self) -> String {
            "failing".to_string()
        }

        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            self.stream_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(futures::stream::iter(vec![Err((self.make_error)())]))
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            self.non_streaming_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Err((self.make_error)()) })
        }
    }

    /// Consume `stream_turn` to its terminal item, returning the error.
    async fn terminal_error<C: ApiClient>(
        handler: &StreamHandler,
        client: &C,
        cancel: &Arc<CancelSignal>,
    ) -> StreamHandlerError {
        let request = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            client,
            &request,
            crate::structured::RequestOptions::default(),
            cancel,
        );
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                return e;
            }
        }
        panic!("the stream must terminate with an error");
    }

    #[tokio::test]
    async fn unauthorized_stream_errors_are_not_retried() {
        let client = FailingStreamClient::failing_with(|| {
            ApiError::auth_invalid_key("HTTP 401: invalid api key")
        });
        let handler = StreamHandler::new()
            .with_retry_config(StreamRetryConfig {
                max_retries: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_secs(60),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: true,
            });
        let cancel = Arc::new(CancelSignal::new());
        let err = terminal_error(&handler, &client, &cancel).await;
        assert_eq!(
            client.stream_calls(),
            1,
            "a permanent 401 must cost exactly one streaming attempt"
        );
        assert_eq!(
            client.non_streaming_calls(),
            0,
            "a permanent 401 must not get a non-streaming fallback attempt"
        );
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::InitFailed { last_error, .. }) => {
                assert!(
                    last_error.contains("Invalid API key"),
                    "the auth failure must surface verbatim, got: {last_error}"
                );
            }
            other => panic!("the 401 must fail the stream, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn internal_server_error_is_still_retried() {
        let client = FailingStreamClient::failing_with(|| ApiError::http_with_status(500, "boom"));
        let handler = StreamHandler::new()
            .with_retry_config(StreamRetryConfig {
                max_retries: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_secs(60),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: false,
            });
        let cancel = Arc::new(CancelSignal::new());
        let err = terminal_error(&handler, &client, &cancel).await;
        assert_eq!(
            client.stream_calls(),
            4,
            "a 500-class error keeps the full ladder: initial + max_retries retries"
        );
        assert!(
            matches!(
                err,
                StreamHandlerError::StreamFailed(StreamOutcome::InitFailed { .. })
            ),
            "with the fallback disabled the exhausted ladder fails the stream, got {err:?}"
        );
    }

    #[tokio::test]
    async fn not_found_stream_errors_are_not_retried() {
        let client =
            FailingStreamClient::failing_with(|| ApiError::http_with_status(404, "unknown model"));
        let handler = StreamHandler::new()
            .with_retry_config(StreamRetryConfig {
                max_retries: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_secs(60),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: true,
            });
        let cancel = Arc::new(CancelSignal::new());
        let err = terminal_error(&handler, &client, &cancel).await;
        assert_eq!(
            client.stream_calls(),
            1,
            "a permanent 404 must cost exactly one streaming attempt"
        );
        assert_eq!(
            client.non_streaming_calls(),
            0,
            "a permanent 404 must not get a non-streaming fallback attempt"
        );
        match err {
            StreamHandlerError::StreamFailed(StreamOutcome::InitFailed { last_error, .. }) => {
                assert!(
                    last_error.contains("HTTP 404"),
                    "the permanent status must surface verbatim, got: {last_error}"
                );
            }
            other => panic!("the 404 must fail the stream, got {other:?}"),
        }
    }

    /// A stream client that accepts the request and then never produces an
    /// event, counting calls so the cancellation contract can assert the
    /// handler never re-entered the retry ladder.
    struct StalledStreamClient {
        stream_calls: std::sync::atomic::AtomicUsize,
    }

    impl ApiClient for StalledStreamClient {
        fn model(&self) -> String {
            "stalled".to_string()
        }

        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            self.stream_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(futures::stream::pending())
        }

        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Err(ApiError::http("no non-streaming path")) })
        }
    }

    #[tokio::test]
    async fn mid_stream_total_timeout_takes_the_fallback_path() {
        struct StallThenFallbackMock {
            stream_calls: std::sync::atomic::AtomicUsize,
            non_streaming_calls: std::sync::atomic::AtomicUsize,
        }
        impl StallThenFallbackMock {
            fn counting() -> Self {
                Self {
                    stream_calls: std::sync::atomic::AtomicUsize::new(0),
                    non_streaming_calls: std::sync::atomic::AtomicUsize::new(0),
                }
            }
        }
        impl ApiClient for StallThenFallbackMock {
            fn model(&self) -> String {
                "stall-fallback".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                self.stream_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "stall-fallback".to_string(),
                        },
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                ];
                Box::pin(futures::stream::iter(events).chain(futures::stream::pending()))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                self.non_streaming_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async {
                    // A real fallback request is pending on its first poll,
                    // unlike an instantly-ready future — the sleep makes the
                    // test discriminate a fallback killed by the already
                    // expired streaming deadline.
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(crate::api::NonStreamingResponse {
                        message: Message::new(
                            crate::message::Role::Assistant,
                            vec![crate::stream::MessagePart::text("fallback answer")],
                        ),
                        stop_reason: StreamStopReason::EndTurn,
                        usage: None,
                    })
                })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_millis(200),
            per_event_timeout: Duration::from_millis(200),
            total_stream_timeout: Duration::from_millis(400),
            max_consecutive_timeouts: 10,
            fallback_to_non_streaming: true,
        });
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        let client = StallThenFallbackMock::counting();
        let started = Instant::now();
        let mut stream = handler.stream_turn(
            &client,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut fell_back = false;
        while let Some(item) = stream.next().await {
            match item.expect("an expired deadline with fallback configured must not error") {
                HandlerEvent::Fallback { .. } => fell_back = true,
                HandlerEvent::Stream(_) | HandlerEvent::AttemptReset => {}
            }
        }
        assert!(
            fell_back,
            "a mid-stream total timeout must reach the non-streaming fallback, not a retry or a bare failure"
        );
        assert!(
            started.elapsed() >= Duration::from_millis(400),
            "the fallback must complete after the streaming deadline expired, at {started:?}+{elapsed:?}",
            elapsed = started.elapsed()
        );
        assert_eq!(
            client
                .stream_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the expired deadline must cost exactly one streaming attempt"
        );
        assert_eq!(
            client
                .non_streaming_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the fallback must run exactly once"
        );
    }

    #[tokio::test]
    async fn mid_stream_total_timeout_is_not_retried() {
        struct CountingStallMock {
            stream_calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for CountingStallMock {
            fn model(&self) -> String {
                "counting-stall".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                self.stream_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "counting-stall".to_string(),
                        },
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "partial".to_string(),
                        },
                    })),
                ];
                Box::pin(futures::stream::iter(events).chain(futures::stream::pending()))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http("unused")) })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_millis(200),
                per_event_timeout: Duration::from_millis(200),
                total_stream_timeout: Duration::from_millis(400),
                max_consecutive_timeouts: 10,
                fallback_to_non_streaming: false,
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        let client = CountingStallMock {
            stream_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut stream = handler.stream_turn(
            &client,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut resets = 0usize;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::AttemptReset) => resets += 1,
                Ok(_) => {}
                Err(e) => {
                    terminal = Some(e);
                    break;
                }
            }
        }
        assert_eq!(
            client
                .stream_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "an expired total deadline must never trigger a second streaming attempt"
        );
        assert_eq!(
            resets, 0,
            "no AttemptReset may be emitted when the timeout is terminal"
        );
        assert!(
            matches!(
                terminal,
                Some(StreamHandlerError::StreamFailed(StreamOutcome::TotalTimeout {
                    events_processed,
                    ..
                })) if events_processed >= 2
            ),
            "the terminal error must be the mid-stream TotalTimeout with real progress, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn hanging_fallback_is_cut_by_the_fresh_budget() {
        struct StallWithHangingFallbackMock {
            stream_calls: std::sync::atomic::AtomicUsize,
            non_streaming_calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for StallWithHangingFallbackMock {
            fn model(&self) -> String {
                "stall-hanging-fallback".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                self.stream_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let events = vec![Ok(StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "m1".to_string(),
                        role: "assistant".to_string(),
                        model: "stall-hanging-fallback".to_string(),
                    },
                }))];
                Box::pin(futures::stream::iter(events).chain(futures::stream::pending()))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                self.non_streaming_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(std::future::pending())
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_millis(200),
            per_event_timeout: Duration::from_millis(200),
            total_stream_timeout: Duration::from_millis(400),
            max_consecutive_timeouts: 10,
            fallback_to_non_streaming: true,
        });
        let cancel = Arc::new(CancelSignal::new());
        let client = StallWithHangingFallbackMock {
            stream_calls: std::sync::atomic::AtomicUsize::new(0),
            non_streaming_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let started = Instant::now();
        let err = terminal_error(&handler, &client, &cancel).await;
        let elapsed = started.elapsed();
        assert_eq!(
            client
                .stream_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the stalled stream costs one attempt"
        );
        assert_eq!(
            client
                .non_streaming_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the fallback must actually start"
        );
        assert!(
            elapsed >= Duration::from_millis(550),
            "the fallback must run its fresh initial_event_timeout budget (200ms) after the \
             streaming deadline (400ms), not be cut instantly by the expired deadline; elapsed {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the fresh budget must still bound a hanging fallback; elapsed {elapsed:?}"
        );
        match err {
            StreamHandlerError::FallbackFailed { fallback_error, .. } => assert!(
                fallback_error.contains("deadline"),
                "the fresh budget's expiry must be the failure cause: {fallback_error}"
            ),
            other => panic!("a hanging fallback must fail as FallbackFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn expired_deadline_before_retry_takes_the_fallback() {
        struct RetryErrorThenFallbackMock {
            stream_calls: std::sync::atomic::AtomicUsize,
            non_streaming_calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for RetryErrorThenFallbackMock {
            fn model(&self) -> String {
                "retry-then-fallback".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                self.stream_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(futures::stream::iter(vec![Err(ApiError::http(
                    "connection reset",
                ))]))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                self.non_streaming_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async {
                    Ok(crate::api::NonStreamingResponse {
                        message: Message::new(
                            crate::message::Role::Assistant,
                            vec![crate::stream::MessagePart::text("fallback answer")],
                        ),
                        stop_reason: StreamStopReason::EndTurn,
                        usage: None,
                    })
                })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_millis(150),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: true,
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 400,
                max_delay_ms: 400,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let client = RetryErrorThenFallbackMock {
            stream_calls: std::sync::atomic::AtomicUsize::new(0),
            non_streaming_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let request = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &client,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut fell_back = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::Fallback { .. }) => fell_back = true,
                Ok(_) => {}
                Err(e) => panic!("the expiry must take the fallback, got {e:?}"),
            }
        }
        assert!(
            fell_back,
            "a deadline expiring before the next retry must reach the non-streaming fallback"
        );
        let calls = client
            .stream_calls
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            calls, 2,
            "the retried attempt starts and is cut on its first poll — the expiry is terminal"
        );
        assert_eq!(
            client
                .non_streaming_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the fallback must run exactly once"
        );
    }

    #[tokio::test]
    async fn per_event_timeout_exhaustion_still_retries() {
        struct AlwaysStalledMock {
            stream_calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for AlwaysStalledMock {
            fn model(&self) -> String {
                "always-stalled".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                self.stream_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(futures::stream::pending())
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http("no non-streaming path")) })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_millis(50),
                per_event_timeout: Duration::from_millis(50),
                total_stream_timeout: Duration::from_secs(60),
                max_consecutive_timeouts: 2,
                fallback_to_non_streaming: false,
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 1,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            });
        let cancel = Arc::new(CancelSignal::new());
        let client = AlwaysStalledMock {
            stream_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let err = terminal_error(&handler, &client, &cancel).await;
        assert_eq!(
            client
                .stream_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "per-event timeout exhaustion keeps the retry ladder: initial + one retry"
        );
        assert!(
            matches!(
                err,
                StreamHandlerError::StreamFailed(StreamOutcome::EventTimeout { .. })
            ),
            "the exhausted ladder terminates with the EventTimeout outcome, got {err:?}"
        );
    }

    #[tokio::test]
    async fn per_event_stall_still_uses_the_total_deadline() {
        struct SlowButHealthyStream;
        impl ApiClient for SlowButHealthyStream {
            fn model(&self) -> String {
                "slow-healthy".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                Box::pin(async_stream::stream! {
                    yield Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "slow-healthy".to_string(),
                        },
                    }));
                    for _ in 0..10 {
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        yield Ok(StreamEvent::IndexedDelta(IndexedDelta {
                            index: 0,
                            delta: DeltaPart::Text { text: "chunk".to_string() },
                        }));
                    }
                    yield Ok(StreamEvent::MessageStop);
                })
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http("unused")) })
            }
        }

        struct FlakyThenOkStream {
            calls: std::sync::atomic::AtomicUsize,
        }
        impl ApiClient for FlakyThenOkStream {
            fn model(&self) -> String {
                "flaky-then-ok".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call == 0 {
                    return Box::pin(futures::stream::iter(vec![Err(ApiError::http(
                        "connection reset",
                    ))]));
                }
                Box::pin(futures::stream::iter(vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m2".to_string(),
                            role: "assistant".to_string(),
                            model: "flaky-then-ok".to_string(),
                        },
                    })),
                    Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text {
                            text: "recovered".to_string(),
                        },
                    })),
                    Ok(StreamEvent::MessageStop),
                ]))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http("unused")) })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_secs(1),
            per_event_timeout: Duration::from_secs(1),
            total_stream_timeout: Duration::from_secs(2),
            max_consecutive_timeouts: 3,
            fallback_to_non_streaming: false,
        });
        let cancel = Arc::new(CancelSignal::new());
        let request = crate::api::StreamRequest::new(vec![]);
        {
            let mut stream = handler.stream_turn(
                &SlowButHealthyStream,
                &request,
                crate::structured::RequestOptions::default(),
                &cancel,
            );
            let mut stopped = false;
            while let Some(item) = stream.next().await {
                if let HandlerEvent::Stream(StreamEvent::MessageStop) =
                    item.expect("a healthy stream within both budgets must not error")
                {
                    stopped = true;
                }
            }
            assert!(
                stopped,
                "a stream producing events under the total budget must complete, not be cut"
            );
        }

        let client = FlakyThenOkStream {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let handler = handler.with_retry_config(StreamRetryConfig {
            max_retries: 1,
            base_delay_ms: 1,
            max_delay_ms: 2,
            ..Default::default()
        });
        let mut stream = handler.stream_turn(
            &client,
            &request,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut stopped = false;
        while let Some(item) = stream.next().await {
            if let HandlerEvent::Stream(StreamEvent::MessageStop) =
                item.expect("a retried-then-successful stream must not error")
            {
                stopped = true;
            }
        }
        assert!(stopped, "the recovered attempt must complete the turn");
        assert_eq!(
            client.calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "exactly one retry, then success"
        );
    }

    #[tokio::test]
    async fn cancelled_stream_is_not_retried() {
        let client = StalledStreamClient {
            stream_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let handler = StreamHandler::new()
            .with_retry_config(StreamRetryConfig {
                max_retries: 3,
                base_delay_ms: 1,
                max_delay_ms: 2,
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_secs(60),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: true,
            });
        let cancel = Arc::new(CancelSignal::new());
        let cancel_for_task = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancel_for_task.cancel();
        });
        let err = terminal_error(&handler, &client, &cancel).await;
        assert_eq!(
            client
                .stream_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "cancellation mid-stream must not re-enter the retry ladder"
        );
        assert!(
            matches!(err, StreamHandlerError::Cancelled),
            "the terminal error must be the cancellation, got {err:?}"
        );
    }

    #[tokio::test]
    async fn mid_stream_total_timeout_reports_real_progress() {
        struct StallAfterEventsMock;

        impl ApiClient for StallAfterEventsMock {
            fn model(&self) -> String {
                "stall".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let events = vec![
                    Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "m1".to_string(),
                            role: "assistant".to_string(),
                            model: "stall".to_string(),
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
                ];
                Box::pin(futures::stream::iter(events).chain(futures::stream::pending()))
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http_with_status(500, "no non-streaming")) })
            }
        }

        let handler = StreamHandler::new().with_timeout_config(StreamTimeoutConfig {
            initial_event_timeout: Duration::from_millis(100),
            per_event_timeout: Duration::from_millis(100),
            total_stream_timeout: Duration::from_millis(500),
            max_consecutive_timeouts: 10,
            fallback_to_non_streaming: false,
        });
        let cancel = Arc::new(CancelSignal::new());
        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &StallAfterEventsMock,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut streamed = 0usize;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(HandlerEvent::Stream(_)) => streamed += 1,
                Err(e) => {
                    terminal = Some(e);
                    break;
                }
                Ok(_) => {}
            }
        }
        assert!(streamed >= 3, "the stream processed real events first");
        match terminal.expect("stream must terminate with an error") {
            StreamHandlerError::StreamFailed(StreamOutcome::TotalTimeout {
                events_processed,
                ..
            }) => assert!(
                events_processed >= 3,
                "doc: events_processed counts accepted events before the deadline — zero implies an immediate stall"
            ),
            other => panic!(
                "a mid-stream deadline is a StreamFailed TotalTimeout, got {other:?} after {streamed} events"
            ),
        }
    }

    #[tokio::test]
    async fn total_timeout_duration_covers_retried_attempts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FailThenStallMock {
            calls: AtomicUsize,
        }

        impl ApiClient for FailThenStallMock {
            fn model(&self) -> String {
                "flaky".to_string()
            }
            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
            > {
                let call = self.calls.fetch_add(1, Ordering::SeqCst);
                if call == 0 {
                    let opening = futures::stream::once(async {
                        Ok(StreamEvent::MessageStart(MessageStart {
                            message: MessageMetadata {
                                id: "m1".to_string(),
                                role: "assistant".to_string(),
                                model: "flaky".to_string(),
                            },
                        }))
                    });
                    let kept_alive = opening.chain(futures::stream::once(async {
                        tokio::time::sleep(Duration::from_millis(150)).await;
                        Ok(StreamEvent::IndexedDelta(IndexedDelta {
                            index: 0,
                            delta: DeltaPart::Text {
                                text: "chunk".to_string(),
                            },
                        }))
                    }));
                    Box::pin(kept_alive.chain(futures::stream::once(async {
                        tokio::time::sleep(Duration::from_millis(1200)).await;
                        Err(ApiError::http_with_status(500, "transient boom"))
                    })))
                } else {
                    Box::pin(futures::stream::pending())
                }
            }
            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<crate::api::NonStreamingResponse, ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async { Err(ApiError::http_with_status(500, "no non-streaming")) })
            }
        }

        let handler = StreamHandler::new()
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_millis(500),
                per_event_timeout: Duration::from_millis(500),
                total_stream_timeout: Duration::from_secs(2),
                max_consecutive_timeouts: 10,
                fallback_to_non_streaming: false,
            })
            .with_retry_config(StreamRetryConfig {
                max_retries: 1,
                ..Default::default()
            });
        let client = FailThenStallMock {
            calls: AtomicUsize::new(0),
        };
        let cancel = Arc::new(CancelSignal::new());
        let req = crate::api::StreamRequest::new(vec![]);
        let mut stream = handler.stream_turn(
            &client,
            &req,
            crate::structured::RequestOptions::default(),
            &cancel,
        );
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                terminal = Some(e);
                break;
            }
        }
        match terminal.expect("stream must terminate with an error") {
            StreamHandlerError::StreamFailed(StreamOutcome::TotalTimeout { duration, .. }) => {
                assert!(
                    duration >= Duration::from_millis(1500),
                    "doc: duration is the full stream lifetime, approximately the configured total (2s); got {duration:?}"
                );
            }
            other => panic!("expected StreamFailed TotalTimeout, got {other:?}"),
        }
    }
}
