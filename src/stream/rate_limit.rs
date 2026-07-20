//! Proactive client-side rate limiting (token bucket).
//!
//! A [`TokenBucket`] is a continuous-fill bucket: it allows a burst equal to its
//! capacity, then refills at `capacity / 60` tokens per second. [`RateLimiter`]
//! holds one bucket per provider identity (`base_url`) so distinct providers get
//! independent budgets.
//!
//! This is the proactive complement to the reactive 429 handling in
//! [`handler`](super::handler): the bucket gates a request *before* it fires,
//! smoothing bursty multi-turn loops so most provider-imposed rate limits are
//! never hit.
//!
//! # Quick Start
//!
//! ```rust
//! use std::sync::Arc;
//! use loopctl::stream::rate_limit::RateLimiter;
//!
//! let limiter = Arc::new(RateLimiter::new(60)); // 60 requests/minute
//! // Each acquire() takes one token; the first 60 succeed instantly (burst),
//! // then one token refills per second.
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Default upper bound on how long a single turn waits for a token (30s).
const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(30);

/// A continuous-fill token bucket: the standard client-side rate-limiting algorithm.
///
/// - **Capacity** = max burst size.
/// - **Refill rate** = `capacity / 60` tokens per second (smooth, not tick-based).
/// - Each [`take`](Self::take) consumes exactly one token; if none is available it
///   returns the [`Duration`] the caller should wait before retrying.
///
/// Time is read lazily on each call: there is no background refill task. A long
/// idle period always tops the bucket back to capacity. `Send + Sync` via a short
/// `Mutex` critical section (a few float ops + one `Instant::now()`).
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_sec: f64,
    state: Mutex<BucketState>,
}

#[derive(Debug, Clone, Copy)]
struct BucketState {
    /// Current token count (float for sub-second refill precision).
    tokens: f64,
    /// Last instant at which `tokens` was updated.
    last_refill: Instant,
}

impl TokenBucket {
    /// Build a bucket that bursts `capacity` requests, then refills at
    /// `capacity / 60` tokens per second.
    ///
    /// A capacity of `0` produces a bucket that is always empty — callers should
    /// treat `0` as "disabled" and skip the bucket entirely (see [`RateLimiter`]).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::rate_limit::TokenBucket;
    ///
    /// let bucket = TokenBucket::new(5);
    /// assert!(bucket.take().is_ok(), "first token from a full bucket");
    /// ```
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        let cap = f64::from(capacity);
        Self {
            capacity: cap,
            refill_per_sec: cap / 60.0,
            state: Mutex::new(BucketState {
                tokens: cap,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Consume one token at a known instant.
    ///
    /// Refill is applied *before* the decision, so a long idle period always tops
    /// the bucket to capacity.
    ///
    /// Taking `at` as a parameter keeps the bucket fully deterministic under test
    /// (no wall-clock dependency).
    ///
    /// # Errors
    ///
    /// Returns `Err(wait)` with the [`Duration`] until one token will have
    /// refilled when no token is currently available.
    pub fn take_at(&self, at: Instant) -> Result<(), Duration> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let refill = elapsed_refill(&mut state, at, self.capacity, self.refill_per_sec);
        if refill >= 1.0 {
            state.tokens = refill - 1.0;
            Ok(())
        } else if self.refill_per_sec <= 0.0 {
            Err(Duration::MAX)
        } else {
            let wait_secs = (1.0 - refill) / self.refill_per_sec;
            Err(Duration::from_secs_f64(wait_secs))
        }
    }

    /// Consume one token now. See [`take_at`](Self::take_at).
    ///
    /// # Errors
    ///
    /// Returns `Err(wait)` when no token is currently available; see
    /// [`take_at`](Self::take_at).
    pub fn take(&self) -> Result<(), Duration> {
        self.take_at(Instant::now())
    }

    /// Tokens available at a known instant (after a lazy refill). Non-consuming.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::rate_limit::TokenBucket;
    ///
    /// let bucket = TokenBucket::new(10);
    /// // A fresh bucket is full.
    /// assert!((bucket.available() - 10.0).abs() < 0.5);
    /// ```
    #[must_use]
    pub fn available_at(&self, at: Instant) -> f64 {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        elapsed_refill(&mut state, at, self.capacity, self.refill_per_sec)
    }

    /// Tokens available right now. See [`available_at`](Self::available_at).
    #[must_use]
    pub fn available(&self) -> f64 {
        self.available_at(Instant::now())
    }
}

/// Apply the elapsed-time refill to `state` in place and return the new token count.
///
/// Caps at `capacity` so a long idle does not overflow. Leaves `last_refill` at
/// `at` so the next caller only accounts for the gap since this call.
fn elapsed_refill(state: &mut BucketState, at: Instant, capacity: f64, refill_per_sec: f64) -> f64 {
    let elapsed = at.saturating_duration_since(state.last_refill);
    if elapsed.is_zero() {
        return state.tokens;
    }
    let added = elapsed.as_secs_f64() * refill_per_sec;
    let topped = (state.tokens + added).min(capacity);
    state.tokens = topped;
    state.last_refill = at;
    topped
}

/// One [`TokenBucket`] per distinct provider identity (`base_url`).
///
/// "Provider identity" is the provider's base URL: OpenAI and Ollama (different
/// URLs) get independent budgets; two clients pointed at the same endpoint share
/// a bucket. Set `requests_per_minute` to `0` to disable limiting entirely —
/// [`acquire`](Self::acquire) then returns `Ok(())` immediately and never
/// allocates a bucket.
///
/// This is the type [`StreamHandler`](super::handler::StreamHandler) holds.
#[derive(Debug)]
pub struct RateLimiter {
    buckets: Mutex<HashMap<String, Arc<TokenBucket>>>,
    requests_per_minute: u32,
    max_wait: Duration,
}

impl RateLimiter {
    /// Build a limiter allowing `requests_per_minute` requests per minute per
    /// provider. `0` disables. The `max_wait` ceiling defaults to 30s.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::rate_limit::RateLimiter;
    ///
    /// let limiter = RateLimiter::new(60);
    /// assert!(limiter.is_enabled());
    /// ```
    #[must_use]
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            requests_per_minute,
            max_wait: DEFAULT_MAX_WAIT,
        }
    }

    /// Override the max-wait ceiling: the longest a single turn will block
    /// waiting for a token before proceeding anyway ("better to risk a 429 than
    /// hang the agent").
    #[must_use]
    pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }

    /// Get (or lazily create) the bucket for `base_url`, then take one token.
    ///
    /// When the limiter is disabled (`requests_per_minute == 0`) this returns
    /// `Ok(())` immediately and never touches the bucket map.
    ///
    /// # Errors
    ///
    /// Returns `Err(wait)` with the [`Duration`] until one token will have
    /// refilled when no token is currently available for `base_url`.
    pub fn acquire(&self, base_url: &str) -> Result<(), Duration> {
        if self.requests_per_minute == 0 {
            return Ok(());
        }
        let bucket = {
            let mut map = self
                .buckets
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Arc::clone(
                map.entry(base_url.to_owned())
                    .or_insert_with(|| Arc::new(TokenBucket::new(self.requests_per_minute))),
            )
        };
        bucket.take()
    }

    /// Whether this limiter is active (`requests_per_minute > 0`).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.requests_per_minute > 0
    }

    /// The configured max-wait ceiling.
    #[must_use]
    pub fn max_wait(&self) -> Duration {
        self.max_wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_then_throttle() {
        let bucket = TokenBucket::new(5);
        let now = Instant::now();
        // Five take_at calls at the same instant all succeed (burst).
        for i in 0..5 {
            assert!(bucket.take_at(now).is_ok(), "burst slot {i} should succeed");
        }
        // The sixth at the same instant must wait.
        let wait = bucket
            .take_at(now)
            .expect_err("sixth take should wait for a refill");
        // refill_per_sec = 5/60 ≈ 0.0833, so one token takes ≈ 12s.
        assert!(
            wait >= Duration::from_secs(10) && wait <= Duration::from_secs(14),
            "expected ~12s wait, got {wait:?}"
        );
    }

    #[test]
    fn refill_after_idle() {
        let bucket = TokenBucket::new(5);
        let t0 = Instant::now();
        // Drain the bucket.
        for _ in 0..5 {
            bucket.take_at(t0).expect("drain should succeed while full");
        }
        assert!(bucket.available_at(t0) < 1.0, "bucket should be drained");
        // After 60s it should be back at capacity.
        let t1 = t0 + Duration::from_mins(1);
        let available = bucket.available_at(t1);
        assert!(
            (available - 5.0).abs() < 0.5,
            "expected ~5.0 after 60s idle, got {available}"
        );
    }

    #[test]
    fn subsecond_refill_precision() {
        let bucket = TokenBucket::new(10);
        let t0 = Instant::now();
        // Drain.
        for _ in 0..10 {
            bucket.take_at(t0).expect("drain should succeed while full");
        }
        // Advance 6s: refill = 6 * (10/60) = 1.0 token.
        let t1 = t0 + Duration::from_secs(6);
        let available = bucket.available_at(t1);
        assert!(
            (available - 1.0).abs() < 0.1,
            "expected ~1.0 after 6s, got {available}"
        );
    }

    #[test]
    fn available_is_non_consuming() {
        let bucket = TokenBucket::new(5);
        let now = Instant::now();
        let a = bucket.available_at(now);
        let b = bucket.available_at(now);
        assert!(
            (a - b).abs() < f64::EPSILON,
            "available must be non-consuming"
        );
        // A take_at decrements.
        bucket
            .take_at(now)
            .expect("take should succeed on a full bucket");
        let c = bucket.available_at(now);
        assert!(c < a, "take must decrement available");
    }

    #[test]
    fn disabled_short_circuits() {
        let limiter = RateLimiter::new(0);
        assert!(!limiter.is_enabled());
        assert!(limiter.acquire("anywhere").is_ok());
        // Map stays empty: no bucket allocated.
        assert!(limiter.buckets.lock().unwrap().is_empty());
    }

    #[test]
    fn per_provider_isolation() {
        let limiter = RateLimiter::new(1);
        // Drain the "openai" bucket.
        assert!(limiter.acquire("openai").is_ok());
        assert!(
            limiter.acquire("openai").is_err(),
            "openai bucket should be empty"
        );
        // "ollama" has its own bucket — still full.
        assert!(
            limiter.acquire("ollama").is_ok(),
            "ollama bucket must be independent"
        );
    }

    #[test]
    fn zero_capacity_bucket_does_not_panic() {
        // A zero-capacity bucket can never refill. take() must return Err
        // (not panic by dividing by zero / passing inf to from_secs_f64).
        let bucket = TokenBucket::new(0);
        let result = bucket.take();
        assert!(result.is_err(), "empty bucket should return Err");
    }
}
