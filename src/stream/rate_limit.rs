//! Proactive client-side rate limiting (token bucket).
//!
//! A [`TokenBucket`] allows a burst equal to its capacity, then refills at
//! `capacity / 60` tokens per second. [`RateLimiter`] holds one bucket per
//! provider identity (`base_url`) so distinct providers get independent budgets.
//!
//! This is the proactive complement to the reactive 429 handling in
//! `stream::handler`: the bucket gates a request *before* it fires,
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
    /// Maximum burst size, in tokens.
    ///
    /// The bucket starts full at this value and never refills above it,
    /// so a long idle period always restores the full burst budget.
    capacity: f64,

    /// Continuous refill rate in tokens per second.
    ///
    /// Computed once at construction as `capacity / 60.0`, so a
    /// 60-token bucket refills at one token per second. Float-valued
    /// to preserve sub-second refill precision on each lazy update.
    refill_per_sec: f64,

    /// Mutable bucket state behind a short-lived mutex.
    ///
    /// Held only for the few float ops and one `Instant::now()` of each
    /// [`take`](Self::take) / [`available`](Self::available) call, so
    /// contention is negligible under normal turn rates.
    state: Mutex<BucketState>,
}

#[derive(Debug, Clone, Copy)]
struct BucketState {
    /// Current token count.
    ///
    /// Float-valued for sub-second refill precision: a partial token
    /// accrues between calls and is banked on the next one. Capped at
    /// the bucket's `capacity` by [`elapsed_refill`].
    tokens: f64,

    /// Instant at which `tokens` was last reconciled with elapsed time.
    ///
    /// Each call to [`elapsed_refill`] advances this to the current
    /// `at`, so the next caller only accounts for the gap since the
    /// previous call rather than recomputing from construction.
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

    /// Tokens available at a known instant (after a lazy refill).
    ///
    /// Advances `last_refill` to `at`, banking the elapsed refill.
    pub fn available_at(&self, at: Instant) -> f64 {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        elapsed_refill(&mut state, at, self.capacity, self.refill_per_sec)
    }

    /// Tokens available right now (after a lazy refill).
    pub fn available(&self) -> f64 {
        self.available_at(Instant::now())
    }
}

/// Apply the elapsed-time refill to `state` in place and return the new token count.
///
/// Caps at `capacity` so a long idle does not overflow. When the bucket fills
/// before `at` (the elapsed time exceeded what was needed to reach capacity),
/// `last_refill` advances only to the fill point — not to `at`. This prevents
/// a caller that passes a far-future instant from freezing the bucket: the
/// excess time beyond capacity is irrelevant, so the clock ignores it.
fn elapsed_refill(state: &mut BucketState, at: Instant, capacity: f64, refill_per_sec: f64) -> f64 {
    let elapsed = at.saturating_duration_since(state.last_refill);
    if elapsed.is_zero() {
        return state.tokens;
    }
    if refill_per_sec <= 0.0 {
        return state.tokens;
    }
    let added = elapsed.as_secs_f64() * refill_per_sec;
    let raw = state.tokens + added;
    if raw >= capacity {
        let needed = capacity - state.tokens;
        let secs_to_fill = needed / refill_per_sec;
        state.last_refill = state
            .last_refill
            .checked_add(Duration::from_secs_f64(secs_to_fill))
            .unwrap_or(at);
        state.tokens = capacity;
    } else {
        state.tokens = raw;
        state.last_refill = at;
    }
    state.tokens
}

/// One [`TokenBucket`] per distinct provider identity (`base_url`).
///
/// "Provider identity" is the provider's base URL: OpenAI and Ollama (different
/// URLs) get independent budgets; two clients pointed at the same endpoint share
/// a bucket. Set `requests_per_minute` to `0` to disable limiting entirely —
/// [`acquire`](Self::acquire) then returns `Ok(())` immediately and never
/// allocates a bucket.
///
/// This is the type `StreamHandler` holds when the `streaming` feature is
/// enabled.
#[derive(Debug)]
pub struct RateLimiter {
    /// Per-provider token buckets, keyed by base URL.
    ///
    /// Lazily populated: the first [`acquire`](Self::acquire) for a
    /// given `base_url` creates its bucket at the configured
    /// `requests_per_minute`; later acquires for the same URL share it
    /// via [`Arc`]. Guarded by a `Mutex` so concurrent turns acquire
    /// safely.
    buckets: Mutex<HashMap<String, Arc<TokenBucket>>>,

    /// Configured request budget per provider, in requests per minute.
    ///
    /// `0` disables the limiter entirely — [`acquire`](Self::acquire)
    /// short-circuits to `Ok(())` and never allocates a bucket. Stored
    /// as `u32` because it doubles as each bucket's capacity.
    requests_per_minute: u32,
}

impl RateLimiter {
    /// Build a limiter allowing `requests_per_minute` requests per minute per
    /// provider. `0` disables.
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
        }
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

    /// Whether this limiter is active.
    ///
    /// Returns `true` when `requests_per_minute > 0`. When `false`,
    /// [`acquire`](Self::acquire) short-circuits to `Ok(())` and never
    /// allocates a bucket, so the limiter is effectively a no-op.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.requests_per_minute > 0
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

    #[test]
    fn future_instant_breaks_rate_limit() {
        let bucket = TokenBucket::new(10);
        let now = Instant::now();
        let one_hour_later = now + Duration::from_hours(1);
        let one_min_later = now + Duration::from_mins(1);

        // Drain all 10 tokens at t=now.
        for _ in 0..10 {
            bucket.take_at(now).expect("burst tokens");
        }

        // Poison: take_at with a future instant refills and sets
        // last_refill = one_hour_later. The bucket now thinks time is
        // 1 hour ahead of reality.
        let _poison = bucket.take_at(one_hour_later);

        // Drain any remaining tokens at the future instant.
        for _ in 0..9 {
            let _drain = bucket.take_at(one_hour_later);
        }

        // Now 1 minute has passed in real time. Normally 1 min of refill
        // would restore tokens. After the fix, take_at should succeed
        // because the bucket must not allow a future instant to freeze
        // the refill clock.
        let result = bucket.take_at(one_min_later);
        assert!(
            result.is_ok(),
            "take_at must succeed — 1 min of refill should restore a token; the future-instant call must not freeze the bucket"
        );
    }

    #[test]
    fn past_instant_does_not_refund_or_corrupt() {
        // A take_at called with an instant earlier than last_refill must not
        // fabricate tokens from negative elapsed time, and must not rewind the
        // clock. The probe uses a small forward step so the partial refill is
        // stable (a far-future step would re-fill on every call).
        let bucket = TokenBucket::new(5);
        let t0 = Instant::now();
        // 6s ≈ 0.5 token at capacity 5 (refill 5/60 per second).
        let t1 = t0 + Duration::from_secs(6);

        // Drain at t0.
        for _ in 0..5 {
            bucket.take_at(t0).expect("drain while full");
        }
        assert!(bucket.take_at(t0).is_err(), "drained at t0");

        // Advance to t1: 6s of refill ≈ 0.5 token — still under one, take fails.
        // last_refill is now t1, tokens ≈ 0.5.
        assert!(
            bucket.take_at(t1).is_err(),
            "partial refill under one token"
        );

        // Probe: take at a past instant (t0 < last_refill = t1).
        // Correct: saturating_duration_since returns ZERO → early return, no
        // token granted, last_refill untouched.
        assert!(
            bucket.take_at(t0).is_err(),
            "past-instant take must not refund tokens from negative elapsed time"
        );

        // Integrity: take_at(t1) again must observe the same 0.5-token state.
        // A bug that rewound last_refill to t0 would make this see a fresh 6s
        // gap (0.5 + 0.5 = 1.0 token) and succeed.
        assert!(
            bucket.take_at(t1).is_err(),
            "last_refill must not rewind — bucket state unchanged by the past-instant probe"
        );
    }
}
