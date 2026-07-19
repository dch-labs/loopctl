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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ===================================================
// HealthStatus
// ===================================================

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
    /// Tool is operating normally (health score ≥ 0.8, breaker closed).
    Healthy,
    /// Tool is experiencing elevated errors or latency (score 0.5–0.8).
    Degraded,
    /// Tool is failing frequently or circuit breaker is open (score < 0.5).
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

// ===================================================
// ToolStats — Lock-free Atomic Counters
// ===================================================

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
    total_calls: AtomicU64,
    success_count: AtomicU64,
    failure_count: AtomicU64,
    total_duration_ns: AtomicU64,
    max_duration_ns: AtomicU64,
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
    #[must_use]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::Relaxed)
    }

    /// Number of successful calls.
    #[must_use]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Relaxed)
    }

    /// Number of failed calls.
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// All-time success rate (0.0–1.0).
    ///
    /// Returns 1.0 when no calls have been recorded (new tools start
    /// optimistic).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_calls.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let successes = self.success_count.load(Ordering::Relaxed);
        // Compute `successes * EWMA_SCALE / total` entirely in integers.
        // `successes <= total`, so the result is in `[0, EWMA_SCALE]` (i.e. fits in u32).
        let rate = successes
            .saturating_mul(EWMA_SCALE)
            .checked_div(total)
            .unwrap_or(0);
        // Result is in `[0, EWMA_SCALE]` (≤ 1_000_000), so `u32` is safe.
        f64::from(u32::try_from(rate).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(EWMA_SCALE).unwrap_or(u32::MAX))
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
        // `v` is clamped to `EWMA_SCALE` (1_000_000), so `u32` is safe.
        let ewma = f64::from(u32::try_from(v).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(EWMA_SCALE).unwrap_or(u32::MAX));
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

// ===================================================
// CircuitState + ToolCircuitBreaker
// ===================================================

/// Circuit breaker state.
///
/// The breaker follows the standard three-state pattern:
///
/// ```text
/// Closed ──(threshold failures)──► Open
///    ▲                               │
///    │                               │ (recovery_duration elapsed)
///    │                               ▼
///    └──(probe succeeds)──────── HalfOpen
///                                    │
///                  (probe fails) ────►│──► Open
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum CircuitState {
    /// Normal operation — requests are allowed.
    Closed = 0,
    /// Too many failures — requests are blocked.
    Open = 1,
    /// Recovery probe — one request is allowed to test recovery.
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
    /// Defaults to 3.
    pub failure_threshold: u64,
    /// How long to wait in the `Open` state before transitioning to `HalfOpen`.
    ///
    /// Defaults to 30 seconds.
    pub recovery_duration: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            recovery_duration: Duration::from_secs(30),
        }
    }
}

/// Per-tool circuit breaker using atomic state.
///
/// When a tool fails `failure_threshold` times consecutively the breaker
/// opens. After `recovery_duration` it transitions to `HalfOpen` and allows
/// one probe call. If the probe succeeds the breaker closes; if it fails
/// the breaker reopens.
///
/// All state is atomic — no locks needed on the hot path. The only
/// `Mutex` is for `last_failure_time`, which is updated only when a
/// failure occurs (cold path relative to the read-only `allow_request`
/// check).
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
    state: AtomicU32,
    consecutive_failures: AtomicU64,
    failure_threshold: u64,
    recovery_duration: Duration,
    last_failure_time: Mutex<Option<Instant>>,
}

impl ToolCircuitBreaker {
    /// Create a new circuit breaker with the given recovery duration and
    /// failure threshold.
    ///
    /// The breaker starts in the Closed state.
    #[must_use]
    pub fn new(recovery_duration: Duration, failure_threshold: u64) -> Self {
        Self {
            state: AtomicU32::new(CircuitState::Closed as u32),
            consecutive_failures: AtomicU64::new(0),
            failure_threshold,
            recovery_duration,
            last_failure_time: Mutex::new(None),
        }
    }

    /// Create a circuit breaker from a [`CircuitBreakerConfig`].
    #[must_use]
    pub fn from_config(config: &CircuitBreakerConfig) -> Self {
        Self::new(config.recovery_duration, config.failure_threshold)
    }

    /// Whether a request is allowed to proceed.
    ///
    /// - **Closed**: always allowed.
    /// - **Open**: allowed only if `recovery_duration` has elapsed since
    ///   the last failure, in which case the calling thread atomically
    ///   transitions to `HalfOpen` and becomes the sole probe.
    /// - **`HalfOpen`**: already probing — no additional probes allowed
    ///   (returns `false` to prevent thundering-herd).
    #[must_use]
    pub fn allow_request(&self) -> bool {
        match self.state.load(Ordering::Acquire).into() {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => false,
            CircuitState::Open => {
                let recovered = crate::error::recover_guard(self.last_failure_time.lock())
                    .map(|t| t.elapsed() >= self.recovery_duration)
                    .unwrap_or(false);
                if recovered {
                    self.state
                        .compare_exchange(
                            CircuitState::Open as u32,
                            CircuitState::HalfOpen as u32,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful call.
    ///
    /// Resets consecutive failures to zero and transitions the breaker
    /// to Closed.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
        self.state
            .store(CircuitState::Closed as u32, Ordering::Release);
    }

    /// Record a failed call.
    ///
    /// Increments the consecutive-failure counter. If the count reaches
    /// `failure_threshold`, the breaker transitions to Open. In the
    /// `HalfOpen` state, a single failure reopens the breaker.
    pub fn record_failure(&self) {
        let failures = self
            .consecutive_failures
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if let Ok(mut guard) = self.last_failure_time.lock() {
            *guard = Some(Instant::now());
        }
        let current_state = self.state.load(Ordering::Acquire).into();
        match current_state {
            CircuitState::Closed => {
                if failures >= self.failure_threshold {
                    self.state
                        .store(CircuitState::Open as u32, Ordering::Release);
                }
            }
            CircuitState::HalfOpen => {
                // Probe failed — reopen immediately
                self.state
                    .store(CircuitState::Open as u32, Ordering::Release);
            }
            CircuitState::Open => {
                // Already open; no state change needed
            }
        }
    }

    /// Current state of the breaker as a string.
    ///
    /// Returns `"closed"`, `"open"`, or `"half-open"`.
    #[must_use]
    pub fn state_label(&self) -> &'static str {
        match self.state.load(Ordering::Acquire).into() {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half-open",
        }
    }

    /// Number of consecutive failures recorded since the last success.
    #[must_use]
    pub fn consecutive_failures(&self) -> u64 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Whether the breaker is currently in the Closed (healthy) state.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire).into(),
            CircuitState::Closed
        )
    }

    /// Whether the breaker is currently in the Open (blocking) state.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire).into(),
            CircuitState::Open
        )
    }

    /// Whether the breaker is currently in the `HalfOpen` (probing) state.
    #[must_use]
    pub fn is_half_open(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire).into(),
            CircuitState::HalfOpen
        )
    }
}

// ===================================================
// ToolHealthRegistry
// ===================================================

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
    stats: Mutex<HashMap<String, Arc<ToolStats>>>,
    breakers: Mutex<HashMap<String, Arc<ToolCircuitBreaker>>>,
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

    /// Set custom circuit-breaker configuration.
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
    /// Combines the circuit-breaker state (Open = unavailable) with the
    /// health score (Unhealthy = unavailable). Returns `true` when:
    /// - the breaker is not Open and the health score is not Unhealthy, or
    /// - the breaker has transitioned to `HalfOpen` (allowing a recovery probe
    ///   even if the health score is still Unhealthy).
    #[must_use]
    pub fn is_tool_available(&self, tool_name: &str) -> bool {
        let breaker = self.get_circuit_breaker(tool_name);
        if !breaker.allow_request() {
            return false;
        }
        if breaker.is_half_open() {
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
    #[must_use]
    pub fn tool_count(&self) -> usize {
        crate::error::recover_guard(self.stats.lock()).len()
    }
}

// ===================================================
// HealthRouterMiddleware
// ===================================================

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
    fallbacks: HashMap<String, Vec<String>>,
}

impl HealthRouter {
    /// Create an empty router with no fallback mappings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallbacks: HashMap::new(),
        }
    }

    /// Get the ordered list of fallback tool names for a primary tool.
    ///
    /// Returns an empty slice if no fallbacks are configured for the
    /// given tool name.
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
    fallbacks: HashMap<String, Vec<String>>,
}

impl HealthRouterBuilder {
    /// Create an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallbacks: HashMap::new(),
        }
    }

    /// Register a list of fallback tools for a primary tool name.
    ///
    /// Fallbacks are tried in the order provided.
    #[must_use]
    pub fn add_fallback(mut self, primary: &str, alternatives: Vec<String>) -> Self {
        self.fallbacks.insert(primary.to_string(), alternatives);
        self
    }

    /// Build the [`HealthRouter`].
    #[must_use]
    pub fn build(self) -> HealthRouter {
        HealthRouter {
            fallbacks: self.fallbacks,
        }
    }
}

// ===================================================
// Tests
// ===================================================

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
    fn circuit_breaker_from_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_duration: Duration::from_mins(1),
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
