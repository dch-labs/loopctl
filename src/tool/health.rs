//! Tool health monitoring, circuit breakers, and self-healing routing.
//!
//! Per-tool health tracking using lock-free atomic counters,
//! circuit-breaker state machines to prevent repeated calls to failing tools,
//! and a registry that combines both into a unified health picture. A routing
//! middleware uses the registry to redirect tool calls away from unhealthy tools
//! toward healthy alternatives.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::tool::health::{ToolHealthRegistry, HealthStatus};
//! use std::time::Duration;
//!
//! let registry = ToolHealthRegistry::new();
//!
//! // Record outcomes
//! registry.record_success("bash", Duration::from_millis(150));
//! registry.record_success("bash", Duration::from_millis(200));
//! registry.record_failure("bash", Duration::from_secs(5));
//!
//! // Check health — mostly-successful tool stays at Degraded or better
//! let status = registry.get_health_status("bash");
//! assert!(status == HealthStatus::Healthy || status == HealthStatus::Degraded);
//! assert!(registry.is_tool_available("bash"));
//!
//! // Get a snapshot for observability
//! let summary = registry.health_summary();
//! assert!(summary.contains_key("bash"));
//! ```
//!
//! # Architecture
//!
//! The module has four layers:
//!
//! 1. **[`ToolStats`]** — Lock-free atomic counters for per-tool call counts,
//!    success/failure rates, and latency tracking. Updated on every tool call
//!    without blocking.
//!
//! 2. **[`ToolCircuitBreaker`]** — A three-state machine (`Closed` → `Open` → `HalfOpen`)
//!    per tool. After `failure_threshold` consecutive failures the breaker opens,
//!    blocking further calls until `recovery_duration` elapses. Then a single
//!    "probe" call is allowed (`HalfOpen`). If it succeeds the breaker closes;
//!    if it fails the breaker reopens.
//!
//! 3. **[`ToolHealthRegistry`]** — A concrete struct combining per-tool stats and
//!    circuit breakers. Every agent uses the same health tracking mechanics,
//!    so a trait boundary would add complexity without value.
//!
//! 4. **[`HealthRouter`]** — Inspects tool health before dispatch and
//!    routes calls to healthy alternatives when the primary tool is degraded or
//!    unhealthy.
//!
//! # Thread Safety
//!
//! All hot-path operations (recording success/failure, checking health) use
//! lock-free atomics. The only `Mutex` is in the registry's name → stats/breaker
//! maps, which are locked only when a new tool name is first seen (cold path).

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Classified health status for a tool.
///
/// Determined by combining the tool's composite [`ToolStats::health_score`]
/// with the [`ToolCircuitBreaker`] state. Used by [`HealthRouter`]
/// to decide whether to route calls to an alternative.
///
/// # Classification Thresholds
///
/// | Score range | Circuit breaker | Status |
/// |-------------|-----------------|--------|
/// | ≥ 0.8 | Closed | [`Healthy`](HealthStatus::Healthy) |
/// | 0.5–0.8 | Closed | [`Degraded`](HealthStatus::Degraded) |
/// | < 0.5 | Any | [`Unhealthy`](HealthStatus::Unhealthy) |
/// | Any | Open | [`Unhealthy`](HealthStatus::Unhealthy) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthStatus {
    /// Tool is operating normally.
    ///
    /// Assigned when the composite health score is ≥ 0.8 and the
    /// circuit breaker is `Closed`. The router routes calls to this
    /// tool without hesitation.
    Healthy,

    /// Tool is experiencing elevated errors or latency.
    ///
    /// Assigned when the health score is in the 0.5–0.8 band. The tool
    /// is still called (the breaker has not tripped), but the router
    /// may prefer a healthy alternative when one is available.
    Degraded,

    /// Tool is failing frequently or its circuit breaker is open.
    ///
    /// Assigned when the health score drops below 0.5, or whenever the
    /// breaker is `Open` regardless of score. The router treats the
    /// tool as unavailable and redirects to a fallback.
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Fixed-point scale for the EWMA success rate.
///
/// Stored as `u64` where `1.0` = `1_000_000`. This avoids floating-point
/// atomics (which don't exist in `std`) while maintaining sufficient
/// precision for health scoring.
const EWMA_SCALE: u64 = 1_000_000;

/// Per-tool statistics using lock-free atomic counters.
///
/// All counters are `AtomicU64` so that recording success/failure never
/// blocks. The [`health_score`](Self::health_score) computation reads all
/// counters once; slight skew between reads is acceptable because the score
/// is used for routing hints, not for exact correctness.
///
/// # Exponentially-Weighted Moving Average (EWMA)
///
/// The `ewma_success` counter tracks recent success rate with a decay
/// factor of 0.7. On every call:
///
/// ```text
/// ewma = 0.7 * ewma + 0.3 * (success ? 1.0 : 0.0)
/// ```
///
/// This makes the health score respond quickly to degradation while
/// retaining long-term history.
///
/// # Example
///
/// ```
/// use loopctl::tool::health::ToolStats;
/// use std::time::Duration;
///
/// let stats = ToolStats::new();
///
/// stats.record_success(Duration::from_millis(100));
/// stats.record_success(Duration::from_millis(200));
/// stats.record_failure(Duration::from_millis(5000));
///
/// assert_eq!(stats.total_calls(), 3);
/// assert_eq!(stats.success_count(), 2);
/// assert_eq!(stats.failure_count(), 1);
/// assert!(stats.success_rate() > 0.0);
/// assert!(stats.health_score() > 0.0);
/// ```
pub struct ToolStats {
    /// Total number of calls recorded (successes + failures).
    ///
    /// Incremented once per `record_success` / `record_failure` call;
    /// the denominator for [`success_rate`](Self::success_rate).
    total_calls: AtomicU64,

    /// Number of calls that completed successfully.
    ///
    /// Incremented by [`record_success`](Self::record_success); paired
    /// with `total_calls` to compute the all-time success rate.
    success_count: AtomicU64,

    /// Number of calls that failed.
    ///
    /// Incremented by [`record_failure`](Self::record_failure). Kept
    /// separately from `success_count` so both rates are available
    /// without re-deriving from the total.
    failure_count: AtomicU64,

    /// Sum of per-call durations in nanoseconds, across all calls.
    ///
    /// Accumulated by both record methods; divided by `total_calls` in
    /// [`avg_duration`](Self::avg_duration). Saturates at `u64::MAX` on
    /// overflow rather than wrapping.
    total_duration_ns: AtomicU64,

    /// High-water mark for the longest single-call duration, in
    /// nanoseconds.
    ///
    /// Updated via `fetch_max` so it only ever grows; exposed by
    /// [`max_duration`](Self::max_duration). Useful for spotting
    /// tail-latency outliers.
    max_duration_ns: AtomicU64,

    /// Exponentially-weighted moving average of success, in fixed-point.
    ///
    /// Stored as `u64` on a `1.0 = EWMA_SCALE` scale so it can be
    /// updated atomically without floating-point atomics. Decays with a
    /// 0.7 factor on every call, making the health score responsive to
    /// recent degradation.
    ewma_success: AtomicU64,
}

impl Default for ToolStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolStats {
    /// Create a new stats instance with all counters at zero.
    ///
    /// The EWMA starts at 1.0 (healthy) so that new tools are not
    /// penalized before they have any call history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            total_duration_ns: AtomicU64::new(0),
            max_duration_ns: AtomicU64::new(0),
            ewma_success: AtomicU64::new(EWMA_SCALE),
        }
    }

    /// Record a successful tool execution.
    ///
    /// Increments the total and success counters, accumulates the
    /// duration, updates the max-duration high-water mark, and pushes
    /// the EWMA toward 1.0.
    pub fn record_success(&self, duration: Duration) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
        let ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.total_duration_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_duration_ns.fetch_max(ns, Ordering::Relaxed);
        self.ewma_success
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
                Some(update_ewma(prev, true))
            })
            .ok();
    }

    /// Record a failed tool execution.
    ///
    /// Increments the total and failure counters, accumulates the
    /// duration, and pushes the EWMA toward 0.0.
    pub fn record_failure(&self, duration: Duration) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        let ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.total_duration_ns.fetch_add(ns, Ordering::Relaxed);
        self.ewma_success
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
                Some(update_ewma(prev, false))
            })
            .ok();
    }

    /// Total number of calls recorded (successes + failures).
    ///
    /// Lock-free relaxed load of the counter incremented on every
    /// record call. Use as the denominator when computing custom rates.
    #[must_use]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::Relaxed)
    }

    /// Number of calls that completed successfully.
    ///
    /// Lock-free relaxed load. Pair with [`total_calls`](Self::total_calls)
    /// to derive the all-time success rate, or use
    /// [`success_rate`](Self::success_rate) directly.
    #[must_use]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Number of calls that failed.
    ///
    /// Lock-free relaxed load. Pair with [`total_calls`](Self::total_calls)
    /// to derive the all-time failure rate.
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// All-time success rate (0.0–1.0).
    ///
    /// Returns 1.0 when no calls have been recorded (new tools start
    /// optimistic).
    ///
    /// The ratio is computed as `(successes * EWMA_SCALE) / total / EWMA_SCALE`
    /// rather than `successes / total` directly so the intermediate result
    /// keeps six digits of integer precision before the final narrowing to
    /// `f64` — small success rates over very large call counts would otherwise
    /// floor to zero in pure integer arithmetic.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_calls.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successes = self.success_count.load(Ordering::Relaxed);
        let rate = successes
            .saturating_mul(EWMA_SCALE)
            .checked_div(total)
            .unwrap_or(0);
        crate::numeric::unit_ratio(rate, EWMA_SCALE)
    }

    /// Composite health score (0.0–1.0) blending success rate with EWMA.
    ///
    /// The score weights the EWMA at 70% and the all-time success rate
    /// at 30% so that recent failures have a larger impact than ancient
    /// successes:
    ///
    /// ```text
    /// health_score = 0.3 * success_rate + 0.7 * ewma_success
    /// ```
    #[must_use]
    pub fn health_score(&self) -> f64 {
        let v = self.ewma_success.load(Ordering::Relaxed).min(EWMA_SCALE);
        let ewma = crate::numeric::unit_ratio(v, EWMA_SCALE);
        0.3 * self.success_rate() + 0.7 * ewma
    }

    /// Average call duration across all recorded calls.
    ///
    /// Returns [`Duration::ZERO`] when no calls have been recorded.
    #[must_use]
    pub fn avg_duration(&self) -> Duration {
        let total = self.total_calls.load(Ordering::Relaxed);
        if total == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.total_duration_ns.load(Ordering::Relaxed);
        // total is guaranteed > 0 (checked above), so checked_div always returns Some.
        let avg_ns = u128::from(total_ns)
            .checked_div(u128::from(total))
            .unwrap_or(0);
        Duration::from_nanos(u64::try_from(avg_ns).unwrap_or(u64::MAX))
    }

    /// Maximum single-call duration recorded so far.
    ///
    /// Returns [`Duration::ZERO`] when no calls have been recorded.
    #[must_use]
    pub fn max_duration(&self) -> Duration {
        Duration::from_nanos(self.max_duration_ns.load(Ordering::Relaxed))
    }
}

/// Update the EWMA in fixed-point representation.
///
/// - `prev` is the current EWMA value in `[0, EWMA_SCALE]`.
/// - `is_success` determines whether the new sample is 1.0 or 0.0.
/// - Returns the updated EWMA clamped to `[0, EWMA_SCALE]`.
fn update_ewma(prev: u64, is_success: bool) -> u64 {
    let sample = u64::from(is_success).saturating_mul(EWMA_SCALE);
    // 0.7 * prev + 0.3 * sample
    let next = (prev
        .saturating_mul(7)
        .saturating_add(sample.saturating_mul(3)))
        / 10;
    next.min(EWMA_SCALE)
}

/// Circuit breaker state.
///
/// The breaker follows the standard three-state pattern. Starting from
/// `Closed`, `failure_threshold` consecutive failures transition it to
/// `Open`, where requests are blocked. Once `recovery_duration`
/// elapses the next `allow_request` moves it to `HalfOpen` and lets a
/// single probe call through. A successful probe closes the breaker; a
/// failed probe reopens it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum CircuitState {
    /// Normal operation — requests are allowed.
    ///
    /// The resting state. Consecutive failures are counted toward
    /// `failure_threshold`; reaching it transitions to [`Open`](Self::Open).
    Closed = 0,

    /// Too many failures — requests are blocked.
    ///
    /// Entered after `failure_threshold` consecutive failures (or when
    /// a [`HalfOpen`](Self::HalfOpen) probe fails). Requests are
    /// refused until `recovery_duration` elapses, after which the next
    /// `allow_request` transitions to `HalfOpen`.
    Open = 1,

    /// Recovery probe — one request is allowed to test recovery.
    ///
    /// A single probe call runs; a success closes the breaker, a
    /// failure reopens it. Additional probes are refused to avoid a
    /// thundering herd.
    HalfOpen = 2,
}

impl From<u32> for CircuitState {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Closed,
            1 => Self::Open,
            _ => Self::HalfOpen,
        }
    }
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "closed"),
            Self::Open => write!(f, "open"),
            Self::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Configuration for a [`ToolCircuitBreaker`].
///
/// Controls how many consecutive failures trigger the breaker and how
/// long to wait before allowing a recovery probe.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before the breaker opens.
    ///
    /// Defaults to 3. `0` disables tripping — the crate-wide "zero
    /// disables" sentinel — so a breaker configured with `0` records
    /// failures and telemetry but never opens.
    pub failure_threshold: u64,

    /// How long to wait in the `Open` state before transitioning to `HalfOpen`.
    ///
    /// Defaults to 30 seconds.
    pub recovery_duration: Duration,

    /// How long the single `HalfOpen` probe may stay in flight before the
    /// breaker gives up on it.
    ///
    /// A probe whose result never arrives (the dispatch was cancelled or
    /// the task died mid-flight) would otherwise strand the breaker in
    /// `HalfOpen` forever — no probe result is ever recorded, so nothing
    /// closes or reopens it. After the timeout the breaker returns to
    /// `Open` with a freshly armed cooldown, and a later request probes
    /// again. Defaults to 30 seconds.
    pub probe_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            recovery_duration: Duration::from_secs(30),
            probe_timeout: Duration::from_secs(30),
        }
    }
}

/// Per-tool circuit breaker.
///
/// When a tool fails `failure_threshold` times consecutively the breaker
/// opens. After `recovery_duration` it transitions to `HalfOpen` and allows
/// one probe call. If the probe succeeds the breaker closes; if it fails
/// the breaker reopens.
///
/// All mutable state is behind a single `Mutex`, making every method's
/// read-modify-write atomic with respect to every other method — no
/// window for a counter/state race between concurrent `record_success`
/// and `record_failure` calls.
///
/// # Example
///
/// ```
/// use loopctl::tool::health::ToolCircuitBreaker;
/// use std::time::Duration;
///
/// let breaker = ToolCircuitBreaker::new(Duration::from_millis(100), 2);
///
/// // Initially closed — requests are allowed
/// assert!(breaker.allow_request());
///
/// // Record failures until it opens
/// breaker.record_failure();
/// assert!(breaker.allow_request()); // 1 failure < threshold 2
///
/// breaker.record_failure();
/// // Now open — requests blocked
/// assert!(!breaker.allow_request());
/// ```
pub struct ToolCircuitBreaker {
    /// Number of consecutive failures that trips the breaker.
    ///
    /// Compared against [`consecutive_failures`](BreakerState::consecutive_failures)
    /// on each `record_failure` call; reaching this value while in
    /// `Closed` transitions to `Open`. Set at construction and immutable
    /// thereafter.
    failure_threshold: u64,

    /// How long the breaker stays `Open` before allowing a `HalfOpen` probe.
    ///
    /// Compared against the elapsed time since the last failure in
    /// [`allow_request`](Self::allow_request). Set at construction and
    /// immutable thereafter.
    recovery_duration: Duration,

    /// How long a `HalfOpen` probe may stay in flight before the breaker
    /// re-arms recovery (see [`CircuitBreakerConfig::probe_timeout`]).
    ///
    /// Compared against the elapsed time since the probe was granted in
    /// [`allow_request`](Self::allow_request). Set at construction and
    /// immutable thereafter.
    probe_timeout: Duration,

    /// All mutable state, behind a single lock.
    ///
    /// Holding the counter, state, and last-failure timestamp together
    /// prevents the TOCTOU race where a concurrent `record_success`
    /// could reset the counter while a `record_failure` is mid-transition,
    /// or vice versa.
    state: Mutex<BreakerState>,
}

/// The complete mutable state of a [`ToolCircuitBreaker`].
///
/// Lives behind a single [`Mutex`] on the breaker so that every method's
/// read-modify-write is atomic. All three fields are always observed
/// together — no partial-state reads.
struct BreakerState {
    /// Current circuit-breaker phase.
    ///
    /// `Closed` at construction; transitions through `Open` (failures
    /// exceeded threshold) and `HalfOpen` (recovery probe in flight) as
    /// failures and successes are recorded.
    circuit: CircuitState,

    /// Consecutive failures since the last success.
    ///
    /// Reset to zero by success; reaching the threshold while in
    /// `Closed` transitions to `Open`.
    consecutive_failures: u64,

    /// When the current `HalfOpen` probe was granted, or `None`.
    ///
    /// Set when [`allow_request`](ToolCircuitBreaker::allow_request)
    /// transitions `Open`→`HalfOpen`; cleared by the probe's
    /// `record_success`/`record_failure`. When the probe outlives
    /// [`probe_timeout`](ToolCircuitBreaker::probe_timeout) without a
    /// record, the next `allow_request` re-arms recovery instead of
    /// stranding the breaker.
    probe_started_at: Option<Instant>,

    /// When the most recent failure occurred, or `None` if none yet.
    ///
    /// Compared against `recovery_duration` in `allow_request` to decide
    /// whether enough time has passed to allow a `HalfOpen` probe.
    last_failure_time: Option<Instant>,
}

impl Default for BreakerState {
    fn default() -> Self {
        Self {
            circuit: CircuitState::Closed,
            consecutive_failures: 0,
            last_failure_time: None,
            probe_started_at: None,
        }
    }
}

/// Whether the in-flight `HalfOpen` probe has outlived its timeout.
///
/// A probe whose result never arrives (cancelled dispatch, dead task)
/// is stranded: no `record_*` will ever run for it, so the expiry check
/// is what lets the breaker move on — re-arming the `Open` cooldown on
/// the next `allow_request`, and refusing a *late* result that arrives
/// after expiry (it describes a world the breaker has already left
/// behind).
fn probe_expired(state: &BreakerState, probe_timeout: Duration) -> bool {
    state
        .probe_started_at
        .is_some_and(|started| started.elapsed() >= probe_timeout)
}

/// The instant an in-flight `HalfOpen` probe's lease ends, if one is in flight.
///
/// The re-arm clock anchors here — not to the moment the expiry was
/// noticed — so a real [`allow_request`](ToolCircuitBreaker::allow_request)
/// and the pure availability reads agree on when the next probe becomes
/// grantable, no matter when each first observes the expiry.
fn probe_expires_at(state: &BreakerState, probe_timeout: Duration) -> Option<std::time::Instant> {
    state
        .probe_started_at
        .and_then(|started| started.checked_add(probe_timeout))
}

impl ToolCircuitBreaker {
    /// Create a new circuit breaker with the given recovery duration and
    /// failure threshold.
    ///
    /// The breaker starts in the Closed state.
    #[must_use]
    pub fn new(recovery_duration: Duration, failure_threshold: u64) -> Self {
        Self {
            failure_threshold,
            recovery_duration,
            probe_timeout: recovery_duration,
            state: Mutex::new(BreakerState::default()),
        }
    }

    /// Override the `HalfOpen` probe timeout (builder style).
    ///
    /// [`new`](Self::new) defaults the probe timeout to the recovery
    /// duration; use this when a probe should be given more (or less)
    /// time than a full recovery window before the breaker re-arms.
    #[must_use]
    pub fn with_probe_timeout(mut self, probe_timeout: Duration) -> Self {
        self.probe_timeout = probe_timeout;
        self
    }

    /// Create a circuit breaker from a [`CircuitBreakerConfig`].
    ///
    /// Convenience constructor that unpacks the threshold and recovery
    /// duration from a config struct, delegating to [`new`](Self::new).
    /// Useful when many breakers share a single config.
    #[must_use]
    pub fn from_config(config: &CircuitBreakerConfig) -> Self {
        Self {
            failure_threshold: config.failure_threshold,
            recovery_duration: config.recovery_duration,
            probe_timeout: config.probe_timeout,
            state: Mutex::new(BreakerState::default()),
        }
    }

    /// Whether a request is allowed to proceed.
    ///
    /// - **Closed**: always allowed.
    /// - **Open**: allowed only if `recovery_duration` has elapsed since
    ///   the last failure, in which case the breaker transitions to
    ///   `HalfOpen` and the caller becomes the sole probe.
    /// - **`HalfOpen`**: already probing — no additional probes allowed
    ///   (returns `false` to prevent thundering-herd).
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = crate::error::recover_guard(self.state.lock());
        match state.circuit {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => {
                if probe_expired(&state, self.probe_timeout) {
                    state.circuit = CircuitState::Open;
                    state.last_failure_time = probe_expires_at(&state, self.probe_timeout)
                        .or_else(|| Some(Instant::now()));
                    state.probe_started_at = None;
                }
                false
            }
            CircuitState::Open => {
                let recovered = state
                    .last_failure_time
                    .is_some_and(|t| t.elapsed() >= self.recovery_duration);
                if recovered {
                    state.circuit = CircuitState::HalfOpen;
                    state.probe_started_at = Some(Instant::now());
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Whether a request *would* be allowed, without the `Open`→`HalfOpen` side effect.
    ///
    /// Pure read mirroring [`allow_request`](Self::allow_request)'s decision
    /// logic: returns `true` for `Closed`, `false` for `HalfOpen`, and for
    /// `Open` returns `true` only if the recovery duration has elapsed (i.e.
    /// the next [`allow_request`](Self::allow_request) call would transition
    /// to `HalfOpen` and grant the probe). Crucially, this performs **no**
    /// state transition — use it for availability checks
    /// ([`is_tool_available`](ToolHealthRegistry::is_tool_available)) so a
    /// read does not consume the single `HalfOpen` probe slot that belongs to
    /// the real dispatch path.
    #[must_use]
    pub fn would_allow_request(&self) -> bool {
        let state = crate::error::recover_guard(self.state.lock());
        match state.circuit {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => {
                probe_expired(&state, self.probe_timeout)
                    && state.probe_started_at.is_some_and(|started| {
                        started
                            .checked_add(self.probe_timeout)
                            .and_then(|expired_at| expired_at.checked_add(self.recovery_duration))
                            .is_some_and(|available_at| available_at <= Instant::now())
                    })
            }
            CircuitState::Open => state
                .last_failure_time
                .is_some_and(|t| t.elapsed() >= self.recovery_duration),
        }
    }

    /// Whether the next [`allow_request`](Self::allow_request) call would
    /// transition an `Open` breaker into `HalfOpen`.
    ///
    /// Pure read: `true` only when the breaker is `Open` and the recovery
    /// duration has elapsed — i.e. the next `allow_request` would perform the
    /// `Open`→`HalfOpen` transition and grant the probe slot. Returns `false`
    /// for `HalfOpen` (a probe is already in flight; the next
    /// `allow_request` refuses to avoid a thundering herd) and for `Closed`
    /// (requests are allowed unconditionally, no transition pending).
    /// Complements [`would_allow_request`](Self::would_allow_request).
    #[must_use]
    pub fn would_be_half_open(&self) -> bool {
        let state = crate::error::recover_guard(self.state.lock());
        match state.circuit {
            CircuitState::Open => state
                .last_failure_time
                .is_some_and(|t| t.elapsed() >= self.recovery_duration),
            CircuitState::HalfOpen | CircuitState::Closed => false,
        }
    }

    /// Record a successful call.
    ///
    /// Resets consecutive failures to zero and transitions the breaker
    /// to Closed.
    pub fn record_success(&self) {
        let mut state = crate::error::recover_guard(self.state.lock());
        if matches!(state.circuit, CircuitState::HalfOpen)
            && probe_expired(&state, self.probe_timeout)
        {
            state.circuit = CircuitState::Open;
            state.last_failure_time =
                probe_expires_at(&state, self.probe_timeout).or_else(|| Some(Instant::now()));
            state.probe_started_at = None;
            return;
        }
        state.consecutive_failures = 0;
        state.probe_started_at = None;
        state.circuit = CircuitState::Closed;
    }

    /// Record a failed call.
    ///
    /// Increments the consecutive-failure counter. If the count reaches
    /// `failure_threshold`, the breaker transitions to Open. In the
    /// `HalfOpen` state, a single failure reopens the breaker.
    pub fn record_failure(&self) {
        let mut state = crate::error::recover_guard(self.state.lock());
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.last_failure_time = Some(Instant::now());
        state.probe_started_at = None;
        match state.circuit {
            CircuitState::Closed => {
                if self.failure_threshold > 0
                    && state.consecutive_failures >= self.failure_threshold
                {
                    state.circuit = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                state.circuit = CircuitState::Open;
            }
            CircuitState::Open => {}
        }
    }

    /// Current state of the breaker as a human-readable string.
    ///
    /// Returns `"closed"`, `"open"`, or `"half-open"`. Intended for
    /// logs and metrics where a string label is preferable to the
    /// numeric encoding.
    #[must_use]
    pub fn state_label(&self) -> &'static str {
        match crate::error::recover_guard(self.state.lock()).circuit {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half-open",
        }
    }

    /// Number of consecutive failures recorded since the last success.
    ///
    /// Reset to zero on every success; reaching `failure_threshold`
    /// trips the breaker.
    #[must_use]
    pub fn consecutive_failures(&self) -> u64 {
        crate::error::recover_guard(self.state.lock()).consecutive_failures
    }

    /// Whether the breaker is currently in the Closed (healthy) state.
    ///
    /// `true` when requests are allowed unconditionally.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        crate::error::recover_guard(self.state.lock()).circuit == CircuitState::Closed
    }

    /// Whether the breaker is currently in the Open (blocking) state.
    ///
    /// `true` when requests are refused outright (subject to the
    /// recovery-duration transition handled inside
    /// [`allow_request`](Self::allow_request)).
    #[must_use]
    pub fn is_open(&self) -> bool {
        crate::error::recover_guard(self.state.lock()).circuit == CircuitState::Open
    }

    /// Whether the breaker is currently in the `HalfOpen` (probing)
    /// state.
    ///
    /// `true` when a single probe call is in flight and additional
    /// probes are refused to avoid a thundering herd.
    #[must_use]
    pub fn is_half_open(&self) -> bool {
        crate::error::recover_guard(self.state.lock()).circuit == CircuitState::HalfOpen
    }
}

/// Global health registry for all tools.
///
/// A concrete struct (not a trait) because every agent uses the same health
/// tracking mechanics. The registry provides:
///
/// - Per-tool [`ToolStats`] (lock-free atomic counters)
/// - Per-tool [`ToolCircuitBreaker`] (atomic state machine)
/// - [`is_tool_available`](Self::is_tool_available) — quick check combining
///   health + breaker state
/// - [`health_summary`](Self::health_summary) — snapshot for observability
///
/// Uses `Mutex<HashMap>` for the tool-name → stats/breaker maps. Only
/// the cold path — locks are only taken when a new tool name is first seen.
/// Poisoned mutex recovery follows the pattern: `unwrap_or_else(std::sync::PoisonError::into_inner)`.
///
/// # Thread Safety
///
/// All recording methods (`record_success`, `record_failure`) are `&self`
/// and never block on each other. The internal `Mutex` is only taken to
/// insert a new entry for a previously-unseen tool name.
///
/// # Example
///
/// ```
/// use loopctl::tool::health::{ToolHealthRegistry, HealthStatus};
/// use std::time::Duration;
///
/// let registry = ToolHealthRegistry::new();
///
/// // Simulate some calls
/// registry.record_success("grep", Duration::from_millis(50));
/// registry.record_success("grep", Duration::from_millis(75));
///
/// assert!(registry.is_tool_available("grep"));
/// assert_eq!(registry.get_health_status("grep"), HealthStatus::Healthy);
///
/// // Observability snapshot
/// let summary = registry.health_summary();
/// assert!(summary.contains_key("grep"));
/// let (status, score) = &summary["grep"];
/// assert_eq!(*status, HealthStatus::Healthy);
/// assert!(*score > 0.9);
/// ```
pub struct ToolHealthRegistry {
    /// Per-tool statistics, keyed by tool name.
    ///
    /// Lazily populated on first sighting of a tool; entries hold
    /// `Arc<ToolStats>` so callers can read counters without holding the
    /// lock. The `Mutex` is only acquired on the cold path of inserting
    /// a previously-unseen tool.
    stats: Mutex<HashMap<String, Arc<ToolStats>>>,

    /// Per-tool circuit breakers, keyed by tool name.
    ///
    /// Lazily populated alongside `stats`; each entry is configured from
    /// `breaker_config` at insertion time. Same cold-path locking
    /// strategy as `stats`.
    breakers: Mutex<HashMap<String, Arc<ToolCircuitBreaker>>>,

    /// Configuration applied to every newly-created circuit breaker.
    ///
    /// Set at construction (default 3 failures / 30 s recovery) and
    /// cloned into each breaker on first sight of a tool; changing it
    /// after the fact does not retroactively update existing breakers.
    breaker_config: CircuitBreakerConfig,
}

impl Default for ToolHealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolHealthRegistry {
    /// Create a new empty registry with default circuit-breaker settings.
    ///
    /// Default: 3 consecutive failures to open, 30-second recovery duration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
            breakers: Mutex::new(HashMap::new()),
            breaker_config: CircuitBreakerConfig::default(),
        }
    }

    /// Set custom circuit-breaker configuration (builder style).
    ///
    /// Overrides the default (3 failures / 30 s recovery). Applied only
    /// to breakers created *after* this call — existing per-tool
    /// breakers keep their original thresholds.
    #[must_use]
    pub fn with_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.breaker_config = config;
        self
    }

    /// Get or create stats for a tool.
    ///
    /// Auto-registers on first call. Returns a cloned `Arc<ToolStats>`
    /// so the caller can read counters without holding any lock.
    #[must_use]
    pub fn get_stats(&self, tool_name: &str) -> Arc<ToolStats> {
        let guard = crate::error::recover_guard(self.stats.lock());
        if let Some(stats) = guard.get(tool_name) {
            return Arc::clone(stats);
        }
        drop(guard);
        let mut guard = crate::error::recover_guard(self.stats.lock());
        Arc::clone(
            guard
                .entry(tool_name.to_string())
                .or_insert_with(|| Arc::new(ToolStats::new())),
        )
    }

    /// Get or create a circuit breaker for a tool.
    ///
    /// Auto-registers on first call. The breaker is configured with the
    /// registry's [`CircuitBreakerConfig`].
    #[must_use]
    pub fn get_circuit_breaker(&self, tool_name: &str) -> Arc<ToolCircuitBreaker> {
        let guard = crate::error::recover_guard(self.breakers.lock());
        if let Some(cb) = guard.get(tool_name) {
            return Arc::clone(cb);
        }
        drop(guard);
        let mut guard = crate::error::recover_guard(self.breakers.lock());
        Arc::clone(
            guard
                .entry(tool_name.to_string())
                .or_insert_with(|| Arc::new(ToolCircuitBreaker::from_config(&self.breaker_config))),
        )
    }

    /// Quick health check: is this tool available for use?
    ///
    /// Combines the circuit-breaker state (`Open` = unavailable) with the
    /// health score (`Unhealthy` = unavailable). Returns `true` when:
    /// - the breaker is not `Open` and the health score is not `Unhealthy`, or
    /// - the breaker would treat the next call as a `HalfOpen` recovery probe
    ///   (available even if the health score is still `Unhealthy`).
    ///
    /// This is a **pure read** — it observes whether a request would be
    /// allowed without performing the `Open`→`HalfOpen` transition, so a
    /// bare availability check does not consume the single probe slot that
    /// belongs to the real dispatch path
    /// ([`allow_request`](ToolCircuitBreaker::allow_request)).
    #[must_use]
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        let breaker = self.get_circuit_breaker(tool_name);
        if !breaker.would_allow_request() {
            return false;
        }
        if breaker.would_be_half_open() {
            return true;
        }
        self.get_health_status(tool_name) != HealthStatus::Unhealthy
    }

    /// Get classified health status for a tool.
    ///
    /// Classifies based on the composite health score and circuit-breaker
    /// state using the thresholds documented on [`HealthStatus`].
    #[must_use]
    pub fn get_health_status(&self, tool_name: &str) -> HealthStatus {
        let breaker = self.get_circuit_breaker(tool_name);
        if breaker.is_open() {
            return HealthStatus::Unhealthy;
        }
        let score = self.get_stats(tool_name).health_score();
        if score >= 0.8 {
            HealthStatus::Healthy
        } else if score >= 0.5 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }

    /// Record a successful tool execution.
    ///
    /// Updates both the per-tool stats (success count, latency) and the
    /// circuit breaker (resets consecutive failures).
    pub fn record_success(&self, tool_name: &str, duration: Duration) {
        self.get_stats(tool_name).record_success(duration);
        self.get_circuit_breaker(tool_name).record_success();
    }

    /// Record a failed tool execution.
    ///
    /// Updates both the per-tool stats (failure count, latency) and the
    /// circuit breaker (increments consecutive failures, may open the
    /// breaker).
    pub fn record_failure(&self, tool_name: &str, duration: Duration) {
        self.get_stats(tool_name).record_failure(duration);
        self.get_circuit_breaker(tool_name).record_failure();
    }

    /// Snapshot of all tools' health for observability.
    ///
    /// Returns a map from tool name to `(HealthStatus, health_score)`.
    /// The snapshot is point-in-time and may be slightly inconsistent
    /// across tools (each tool's counters are read independently).
    #[must_use]
    pub fn health_summary(&self) -> HashMap<String, (HealthStatus, f64)> {
        let entries: Vec<(String, Arc<ToolStats>)> = {
            let guard = crate::error::recover_guard(self.stats.lock());
            guard
                .iter()
                .map(|(n, s)| (n.clone(), Arc::clone(s)))
                .collect()
        };
        entries
            .into_iter()
            .map(|(name, stats)| {
                let score = stats.health_score();
                let status = self.get_health_status(&name);
                (name, (status, score))
            })
            .collect()
    }

    /// Number of distinct tools currently tracked.
    ///
    /// Lock-free in the sense that it briefly acquires the stats map's
    /// `Mutex` to read its length. Returns the count of tools seen at
    /// least once via `record_*` / `get_stats` / `get_circuit_breaker`.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        crate::error::recover_guard(self.stats.lock()).len()
    }
}

/// A mapping from primary tool names to alternative tool names.
///
/// When the primary tool is [`Unhealthy`](HealthStatus::Unhealthy) or
/// [`Degraded`](HealthStatus::Degraded), the router attempts to redirect
/// to an alternative in preference order. This allows the agent to
/// continue operating even when a preferred tool is malfunctioning.
///
/// # Example
///
/// ```
/// use loopctl::tool::health::{HealthRouter, HealthRouterBuilder};
///
/// let router = HealthRouterBuilder::new()
///     .add_fallback("bash", vec!["sh".to_string(), "python".to_string()])
///     .add_fallback("edit", vec!["write".to_string()])
///     .build();
///
/// // Find an alternative for "bash" (no registry, so returns the fallbacks)
/// let alternatives = router.fallbacks_for("bash");
/// assert_eq!(alternatives, vec!["sh", "python"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct HealthRouter {
    /// Primary tool name → ordered list of fallback tool names.
    ///
    /// Populated via [`HealthRouterBuilder`]; the router consults this
    /// map (in order) when the primary tool is unavailable, returning
    /// the first healthy alternative.
    fallbacks: HashMap<String, Vec<String>>,
}

impl HealthRouter {
    /// Create an empty router with no fallback mappings.
    ///
    /// Equivalent to [`HealthRouter::default`]. An empty router always
    /// resolves to the primary tool name since no fallbacks are
    /// configured.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallbacks: HashMap::new(),
        }
    }

    /// Get the ordered list of fallback tool names for a primary tool.
    ///
    /// Returns an empty slice if no fallbacks are configured for the
    /// given tool name. The order matches the order supplied to
    /// [`HealthRouterBuilder::add_fallback`].
    #[must_use]
    pub fn fallbacks_for(&self, tool_name: &str) -> &[String] {
        self.fallbacks.get(tool_name).map_or(&[], Vec::as_slice)
    }

    /// Choose the best available tool from a primary name and its fallbacks.
    ///
    /// Returns the primary name if [`is_tool_available`](ToolHealthRegistry::is_tool_available)
    /// reports it as available. Otherwise iterates through the fallbacks and
    /// returns the first one that is available. If no alternative is available,
    /// returns the primary name (letting the caller decide how to handle the
    /// failure).
    #[must_use]
    pub fn resolve_tool(&self, tool_name: &str, registry: &ToolHealthRegistry) -> String {
        if registry.is_tool_available(tool_name) {
            return tool_name.to_string();
        }

        for fallback in self.fallbacks_for(tool_name) {
            if registry.is_tool_available(fallback) {
                return fallback.clone();
            }
        }

        // No healthy alternative — return the original and let the caller handle it
        tool_name.to_string()
    }
}

/// Builder for [`HealthRouter`] with a fluent API.
///
/// # Example
///
/// ```
/// use loopctl::tool::health::HealthRouterBuilder;
///
/// let router = HealthRouterBuilder::new()
///     .add_fallback("bash", vec!["sh".to_string()])
///     .add_fallback("edit", vec!["write".to_string(), "sed".to_string()])
///     .build();
///
/// assert_eq!(router.fallbacks_for("edit"), vec!["write", "sed"]);
/// assert!(router.fallbacks_for("unknown").is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct HealthRouterBuilder {
    /// In-progress fallback map, mutated by each `add_fallback` call.
    ///
    /// Consumed by [`build`](Self::build) to produce the immutable
    /// [`HealthRouter`].
    fallbacks: HashMap<String, Vec<String>>,
}

impl HealthRouterBuilder {
    /// Create an empty builder.
    ///
    /// Equivalent to [`HealthRouterBuilder::default`]; every primary
    /// starts with no fallbacks until `add_fallback` is called.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallbacks: HashMap::new(),
        }
    }

    /// Register a list of fallback tools for a primary tool name.
    ///
    /// Replaces any previously-registered fallbacks for `primary`.
    /// Fallbacks are tried in the order provided when
    /// [`HealthRouter::resolve_tool`] walks the list.
    #[must_use]
    pub fn add_fallback(mut self, primary: &str, alternatives: Vec<String>) -> Self {
        self.fallbacks.insert(primary.to_string(), alternatives);
        self
    }

    /// Build the [`HealthRouter`].
    ///
    /// Consumes the builder and freezes the fallback map into an
    /// immutable router. The returned router is cheap to clone for
    /// sharing across dispatch paths.
    #[must_use]
    pub fn build(self) -> HealthRouter {
        HealthRouter {
            fallbacks: self.fallbacks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_stats_starts_healthy() {
        let stats = ToolStats::new();
        assert_eq!(stats.total_calls(), 0);
        assert_eq!(stats.success_count(), 0);
        assert_eq!(stats.failure_count(), 0);
        assert!((stats.success_rate() - 1.0).abs() < f64::EPSILON);
        assert!(stats.health_score() > 0.9);
        assert_eq!(stats.avg_duration(), Duration::ZERO);
        assert_eq!(stats.max_duration(), Duration::ZERO);
    }

    #[test]
    fn tool_stats_records_success() {
        let stats = ToolStats::new();
        stats.record_success(Duration::from_millis(100));
        stats.record_success(Duration::from_millis(200));

        assert_eq!(stats.total_calls(), 2);
        assert_eq!(stats.success_count(), 2);
        assert_eq!(stats.failure_count(), 0);
        assert!(stats.success_rate() > 0.99);
        assert_eq!(stats.max_duration(), Duration::from_millis(200));
    }

    #[test]
    fn tool_stats_records_failure() {
        let stats = ToolStats::new();
        stats.record_failure(Duration::from_secs(5));

        assert_eq!(stats.total_calls(), 1);
        assert_eq!(stats.success_count(), 0);
        assert_eq!(stats.failure_count(), 1);
        assert!(stats.success_rate() < 0.01);
    }

    #[test]
    fn tool_stats_avg_duration() {
        let stats = ToolStats::new();
        stats.record_success(Duration::from_millis(100));
        stats.record_success(Duration::from_millis(300));

        let avg = stats.avg_duration();
        assert!(avg >= Duration::from_millis(199) && avg <= Duration::from_millis(201));
    }

    #[test]
    fn tool_stats_ewma_responds_to_failures() {
        let stats = ToolStats::new();

        // Start healthy
        let initial = stats.health_score();
        assert!(initial > 0.9);

        // Pound with failures
        for _ in 0..10 {
            stats.record_failure(Duration::from_millis(100));
        }

        let after_failures = stats.health_score();
        assert!(
            after_failures < 0.3,
            "expected score < 0.3, got {after_failures}"
        );
    }

    #[test]
    fn tool_stats_ewma_recovers_on_success() {
        let stats = ToolStats::new();

        // Drive EWMA down
        for _ in 0..10 {
            stats.record_failure(Duration::from_millis(100));
        }
        let low = stats.health_score();
        assert!(low < 0.3);

        // Recover with successes
        for _ in 0..20 {
            stats.record_success(Duration::from_millis(100));
        }
        let recovered = stats.health_score();
        assert!(
            recovered > low,
            "expected recovery: {recovered} should be > {low}"
        );
    }

    #[test]
    fn update_ewma_function() {
        // Start at 1.0
        let mut ewma = EWMA_SCALE;

        // One failure: 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        ewma = update_ewma(ewma, false);
        assert_eq!(ewma, 700_000);

        // Another failure: 0.7 * 0.7 + 0.0 = 0.49
        ewma = update_ewma(ewma, false);
        assert_eq!(ewma, 490_000);

        // One success: 0.7 * 0.49 + 0.3 * 1.0 = 0.643
        ewma = update_ewma(ewma, true);
        assert_eq!(ewma, 643_000);
    }

    #[test]
    fn circuit_state_from_u32() {
        assert_eq!(CircuitState::from(0u32), CircuitState::Closed);
        assert_eq!(CircuitState::from(1u32), CircuitState::Open);
        assert_eq!(CircuitState::from(2u32), CircuitState::HalfOpen);
        assert_eq!(CircuitState::from(99u32), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_state_display() {
        assert_eq!(format!("{}", CircuitState::Closed), "closed");
        assert_eq!(format!("{}", CircuitState::Open), "open");
        assert_eq!(format!("{}", CircuitState::HalfOpen), "half-open");
    }

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = ToolCircuitBreaker::new(Duration::from_secs(30), 3);
        assert!(cb.is_closed());
        assert!(!cb.is_open());
        assert_eq!(cb.state_label(), "closed");
        assert!(cb.allow_request());
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let cb = ToolCircuitBreaker::new(Duration::from_secs(30), 3);

        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_closed(), "2 failures < threshold 3");

        cb.record_failure();
        assert!(cb.is_open(), "3 failures should open breaker");
        assert!(!cb.allow_request());
        assert_eq!(cb.state_label(), "open");
    }

    #[test]
    fn circuit_breaker_success_resets() {
        let cb = ToolCircuitBreaker::new(Duration::from_secs(30), 2);

        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 1);

        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert!(cb.is_closed());
    }

    #[test]
    fn circuit_breaker_half_open_probe() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(50), 1);

        cb.record_failure();
        assert!(cb.is_open());

        // Wait for recovery duration
        std::thread::sleep(Duration::from_millis(60));

        // Should transition to HalfOpen and allow probe
        assert!(cb.allow_request());
        assert_eq!(cb.state_label(), "half-open");
    }

    #[test]
    fn circuit_breaker_half_open_success_closes() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(50), 1);

        cb.record_failure();
        assert!(cb.is_open());

        // Wait for recovery
        std::thread::sleep(Duration::from_millis(60));
        assert!(cb.allow_request()); // HalfOpen

        // Probe succeeds
        cb.record_success();
        assert!(cb.is_closed());
    }

    #[test]
    fn circuit_breaker_half_open_failure_reopens() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(50), 1);

        cb.record_failure();
        assert!(cb.is_open());

        // Wait for recovery
        std::thread::sleep(Duration::from_millis(60));
        assert!(cb.allow_request()); // HalfOpen

        // Probe fails — reopen
        cb.record_failure();
        assert!(cb.is_open());
        assert!(!cb.allow_request());
    }

    #[test]
    fn availability_recovers_after_a_stranded_probe_without_a_dispatch() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(40), 1)
            .with_probe_timeout(Duration::from_millis(60));
        cb.record_failure();
        assert!(!cb.would_allow_request());
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            cb.allow_request(),
            "after the recovery window the probe is granted"
        );
        assert!(
            !cb.would_allow_request(),
            "an in-flight probe holds availability"
        );

        std::thread::sleep(Duration::from_millis(140));
        assert!(
            cb.would_allow_request(),
            "after the probe times out and the re-armed cooldown elapses, \
             availability recovers without any allow_request call re-arming it"
        );
    }

    #[test]
    fn late_success_after_probe_expiry_does_not_close_the_breaker() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(40), 1)
            .with_probe_timeout(Duration::from_millis(60));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(50));
        assert!(cb.allow_request(), "the probe is granted");

        std::thread::sleep(Duration::from_millis(80));
        cb.record_success();
        assert_eq!(
            cb.state_label(),
            "open",
            "a success arriving after the probe lease expired describes a stale \
             probe — it re-arms recovery instead of closing the breaker"
        );
        assert!(!cb.would_allow_request(), "the re-armed cooldown holds");

        std::thread::sleep(Duration::from_millis(50));
        assert!(
            cb.allow_request(),
            "a fresh probe is granted after the re-armed cooldown"
        );
    }

    #[test]
    fn zero_failure_threshold_disables_tripping() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(10), 0);
        for _ in 0..10 {
            cb.record_failure();
        }
        assert!(
            cb.allow_request(),
            "a zero threshold follows the crate-wide zero-disables sentinel: \
             failures are counted but the breaker never opens"
        );
        assert_eq!(cb.consecutive_failures(), 10);
    }

    #[test]
    fn stranded_half_open_probe_re_arms_recovery_after_the_timeout() {
        let cb = ToolCircuitBreaker::new(Duration::from_millis(40), 1)
            .with_probe_timeout(Duration::from_millis(60));
        cb.record_failure();
        assert!(!cb.allow_request());
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            cb.allow_request(),
            "after the recovery window the probe is granted"
        );
        // The probe never records (cancelled or lost). Without the probe
        // timeout the breaker would answer false forever.
        std::thread::sleep(Duration::from_millis(80));
        assert!(
            !cb.allow_request(),
            "an expired probe re-arms the Open cooldown instead of stranding HalfOpen"
        );
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            cb.allow_request(),
            "a fresh probe is granted after the re-armed cooldown"
        );
        cb.record_success();
        assert!(cb.allow_request(), "the recorded probe closes the breaker");
    }

    #[test]
    fn circuit_breaker_from_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_duration: Duration::from_mins(1),
            probe_timeout: Duration::from_mins(1),
        };
        let cb = ToolCircuitBreaker::from_config(&config);

        // Should need 5 failures
        for _ in 0..4 {
            cb.record_failure();
        }
        assert!(cb.is_closed(), "4 failures < threshold 5");

        cb.record_failure();
        assert!(cb.is_open(), "5 failures should open breaker");
    }

    #[test]
    fn health_status_display() {
        assert_eq!(format!("{}", HealthStatus::Healthy), "healthy");
        assert_eq!(format!("{}", HealthStatus::Degraded), "degraded");
        assert_eq!(format!("{}", HealthStatus::Unhealthy), "unhealthy");
    }

    #[test]
    fn registry_starts_empty() {
        let registry = ToolHealthRegistry::new();
        assert_eq!(registry.tool_count(), 0);
    }

    #[test]
    fn registry_auto_registers_on_record() {
        let registry = ToolHealthRegistry::new();

        registry.record_success("bash", Duration::from_millis(100));
        assert_eq!(registry.tool_count(), 1);

        registry.record_failure("grep", Duration::from_millis(50));
        assert_eq!(registry.tool_count(), 2);
    }

    #[test]
    fn registry_tracks_per_tool_stats() {
        let registry = ToolHealthRegistry::new();

        registry.record_success("bash", Duration::from_millis(100));
        registry.record_success("bash", Duration::from_millis(200));
        registry.record_failure("bash", Duration::from_secs(5));

        let stats = registry.get_stats("bash");
        assert_eq!(stats.total_calls(), 3);
        assert_eq!(stats.success_count(), 2);
        assert_eq!(stats.failure_count(), 1);
    }

    #[test]
    fn registry_health_status_classification() {
        let registry = ToolHealthRegistry::new();

        // Fresh tool — healthy (EWMA starts at 1.0)
        registry.record_success("tool_a", Duration::from_millis(10));
        assert_eq!(registry.get_health_status("tool_a"), HealthStatus::Healthy);

        // Drive tool_b down with failures (with low threshold)
        let low_threshold_registry = ToolHealthRegistry::new().with_config(CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_duration: Duration::from_secs(30),
            probe_timeout: Duration::from_secs(30),
        });
        for _ in 0..5 {
            low_threshold_registry.record_failure("tool_b", Duration::from_millis(10));
        }
        let status = low_threshold_registry.get_health_status("tool_b");
        assert!(
            status == HealthStatus::Unhealthy,
            "expected Unhealthy, got {status}"
        );
    }

    #[test]
    fn registry_is_tool_available() {
        let registry = ToolHealthRegistry::new().with_config(CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_duration: Duration::from_secs(30),
            probe_timeout: Duration::from_secs(30),
        });

        // Healthy tool — available
        registry.record_success("tool_a", Duration::from_millis(10));
        assert!(registry.is_tool_available("tool_a"));

        // Open breaker — unavailable
        registry.record_failure("tool_b", Duration::from_millis(10));
        registry.record_failure("tool_b", Duration::from_millis(10));
        assert!(!registry.is_tool_available("tool_b"));
    }

    #[test]
    fn is_tool_available_does_not_consume_half_open_probe() {
        let registry = ToolHealthRegistry::new().with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_duration: Duration::from_millis(40),
            probe_timeout: Duration::from_secs(30),
        });
        registry.record_failure("tool", Duration::from_millis(1));
        assert!(
            !registry.is_tool_available("tool"),
            "Open breaker must be unavailable"
        );
        std::thread::sleep(Duration::from_millis(50));

        // An availability check on the recovered-Open breaker must report
        // available (the next dispatch would probe) WITHOUT performing the
        // Open→HalfOpen transition — otherwise the read consumes the probe
        // and the real dispatch is blocked.
        assert!(
            registry.is_tool_available("tool"),
            "recovered breaker must report available"
        );
        let breaker = registry.get_circuit_breaker("tool");
        assert!(
            breaker.allow_request(),
            "is_tool_available must not consume the HalfOpen probe slot; \
             the real dispatch path must still get it"
        );
        assert_eq!(breaker.state_label(), "half-open");
    }

    #[test]
    fn registry_health_summary() {
        let registry = ToolHealthRegistry::new();

        registry.record_success("bash", Duration::from_millis(100));
        registry.record_failure("grep", Duration::from_millis(50));

        let summary = registry.health_summary();
        assert_eq!(summary.len(), 2);
        assert!(summary.contains_key("bash"));
        assert!(summary.contains_key("grep"));

        let (bash_status, bash_score) = &summary["bash"];
        assert_eq!(*bash_status, HealthStatus::Healthy);
        assert!(*bash_score > 0.5);
    }

    #[test]
    fn health_router_no_fallbacks() {
        let router = HealthRouter::new();
        assert!(router.fallbacks_for("unknown").is_empty());
    }

    #[test]
    fn health_router_with_fallbacks() {
        let router = HealthRouterBuilder::new()
            .add_fallback("bash", vec!["sh".to_string(), "python".to_string()])
            .build();

        assert_eq!(router.fallbacks_for("bash"), vec!["sh", "python"]);
        assert!(router.fallbacks_for("unknown").is_empty());
    }

    #[test]
    fn health_router_resolve_healthy_primary() {
        let registry = ToolHealthRegistry::new();
        registry.record_success("bash", Duration::from_millis(10));

        let router = HealthRouterBuilder::new()
            .add_fallback("bash", vec!["sh".to_string()])
            .build();

        // Primary is healthy — should return primary
        assert_eq!(router.resolve_tool("bash", &registry), "bash");
    }

    #[test]
    fn health_router_resolve_falls_back_to_alternative() {
        let registry = ToolHealthRegistry::new().with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_duration: Duration::from_secs(30),
            probe_timeout: Duration::from_secs(30),
        });

        // Make "bash" unhealthy
        registry.record_failure("bash", Duration::from_millis(10));
        // Make "sh" healthy
        registry.record_success("sh", Duration::from_millis(10));

        let router = HealthRouterBuilder::new()
            .add_fallback("bash", vec!["sh".to_string()])
            .build();

        let resolved = router.resolve_tool("bash", &registry);
        assert_eq!(resolved, "sh", "should fall back to 'sh'");
    }

    #[test]
    fn health_router_resolve_returns_primary_when_no_healthy_alternative() {
        let registry = ToolHealthRegistry::new().with_config(CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_duration: Duration::from_secs(30),
            probe_timeout: Duration::from_secs(30),
        });

        // Both unhealthy
        registry.record_failure("bash", Duration::from_millis(10));
        registry.record_failure("sh", Duration::from_millis(10));

        let router = HealthRouterBuilder::new()
            .add_fallback("bash", vec!["sh".to_string()])
            .build();

        // No healthy alternative — returns primary
        assert_eq!(router.resolve_tool("bash", &registry), "bash");
    }

    #[test]
    fn registry_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let registry = Arc::new(ToolHealthRegistry::new());
        let mut handles = vec![];

        for i in 0..4 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let tool_name = format!("tool_{i}");
                for j in 0..100 {
                    if j % 3 == 0 {
                        reg.record_failure(&tool_name, Duration::from_millis(j));
                    } else {
                        reg.record_success(&tool_name, Duration::from_millis(j));
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All 4 tools registered
        assert_eq!(registry.tool_count(), 4);

        // Each tool should have 100 calls
        for i in 0..4u64 {
            let tool_name = format!("tool_{i}");
            let stats = registry.get_stats(&tool_name);
            assert_eq!(stats.total_calls(), 100);
            // 33 failures (j % 3 == 0 for j in 0..100 → indices 0,3,6,...,99 = 34 failures)
            let failures = stats.failure_count();
            assert!(
                (33..=35).contains(&failures),
                "tool_{i}: expected ~34 failures, got {failures}"
            );
        }
    }
}
