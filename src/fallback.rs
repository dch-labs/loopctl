//! Fallback manager — circuit breaker pattern for automatic API model fallback.
//!
//! When the primary LLM API begins failing repeatedly (rate limits, server errors,
//! timeouts), this module automatically switches to a fallback model and later
//! attempts to recover the primary once it appears healthy again. It implements the
//! classic three-state circuit breaker: **Closed** (primary) → **Open** (fallback) →
//! **Half-Open** (recovering) → back to **Closed**.
//!
//! # Why a circuit breaker?
//!
//! Calling a degraded API repeatedly wastes tokens, increases latency, and can
//! cascade failures. By tripping the circuit after a configurable number of
//! consecutive failures, the agent loop immediately routes subsequent requests to
//! a working fallback model. After a cooldown period, the manager probes the
//! primary with a few trial requests; if they succeed, the circuit closes and
//! normal operation resumes.
//!
//! # Provided types
//!
//! - **[`FallbackState`]** — The three circuit-breaker states (`Primary`, `Fallback`, `Recovering`).
//! - **[`FallbackConfig`]** — Configuration struct with thresholds and timeouts.
//! - **[`FallbackManager`]** — The state machine itself; thread-safe via atomics and `Mutex`.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::fallback::{FallbackManager, FallbackConfig};
//!
//! // Create a manager with a trip threshold of 3 failures
//! let mgr = FallbackManager::new(3, 2);
//!
//! // Simulate failures until the circuit trips
//! mgr.record_model_failure();
//! mgr.record_model_failure();
//! assert!(mgr.record_model_failure()); // 3rd failure trips the circuit
//! assert!(mgr.is_using_fallback());
//!
//! // Manually transition to recovering, then record successes to resume primary
//! mgr.transition_to_recovering();
//! mgr.record_model_success(); // 1st success
//! mgr.record_model_success(); // 2nd → back to Primary
//! assert!(!mgr.is_using_fallback());
//! ```

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Circuit breaker state for LLM model fallback.
///
/// Models the three classic circuit-breaker phases.
///
/// # Transitions
///
/// | From          | To            | Trigger                                                                                |
/// |---------------|---------------|----------------------------------------------------------------------------------------|
/// | `Primary`     | `Fallback`    | Consecutive failures ≥ [`FallbackConfig::trip_threshold`]                              |
/// | `Fallback`    | `Recovering`  | [`transition_to_recovering`](FallbackManager::transition_to_recovering) after cooldown |
/// | `Recovering`  | `Primary`     | Successes ≥ [`FallbackConfig::recovery_successes_needed`]                              |
/// | `Recovering`  | `Fallback`    | Any single failure during recovery                                                     |
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackState;
///
/// let state = FallbackState::Primary;
/// assert_eq!(state as u8, 0);
/// assert_eq!(FallbackState::from(1), FallbackState::Fallback);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackState {
    /// Operating on the primary model — no failures have tripped the
    /// circuit breaker yet.
    Primary = 0,

    /// A fallback model is active — the primary model failed and the
    /// breaker tripped. Subsequent failures on the fallback model are
    /// tracked separately.
    Fallback = 1,

    /// Between models — the primary failed, no fallback has been
    /// selected yet, or a fallback also failed and the manager is
    /// searching for another candidate.
    Recovering = 2,
}

/// Converts a raw `u8` back into a [`FallbackState`].
///
/// Inverse of `state as u8`. Unknown values default to
/// [`FallbackState::Primary`] for safety.
///
/// # Safety
///
/// The conversion is infallible — any out-of-range `u8` maps to
/// [`FallbackState::Primary`] so that a corrupted value cannot
/// cause a panic.
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackState;
///
/// assert_eq!(FallbackState::from(0u8), FallbackState::Primary);
/// assert_eq!(FallbackState::from(1u8), FallbackState::Fallback);
/// assert_eq!(FallbackState::from(2u8), FallbackState::Recovering);
/// assert_eq!(FallbackState::from(255u8), FallbackState::Primary); // unknown → safe default
/// ```
impl From<u8> for FallbackState {
    fn from(value: u8) -> Self {
        match value {
            1 => FallbackState::Fallback,
            2 => FallbackState::Recovering,
            _ => FallbackState::Primary,
        }
    }
}

/// A single failed attempt on a fallback model.
///
/// Records *when* a failure occurred and an optional reason string
/// (e.g. `"rate_limit"`, `"timeout"`, `"500 Internal Server Error"`).
/// Entries accumulate in [`FallbackEntry::attempts`]; once the count
/// reaches [`FallbackEntry::max_fail_count`], the model is considered
/// failed and skipped by [`FallbackManager::fallback_model`].
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::AttemptRecord;
///
/// let record = AttemptRecord::new("rate_limit");
/// assert_eq!(record.reason(), Some("rate_limit"));
/// ```
#[derive(Debug, Clone)]
pub struct AttemptRecord {
    failed_at: Instant,
    reason: Option<String>,
}

impl AttemptRecord {
    /// Create a new attempt record with the given reason.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::AttemptRecord;
    /// let record = AttemptRecord::new("rate_limit");
    /// assert_eq!(record.reason(), Some("rate_limit"));
    /// ```
    #[must_use]
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            failed_at: Instant::now(),
            reason: Some(reason.into()),
        }
    }

    /// Create a new attempt record without a reason.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::AttemptRecord;
    /// let record = AttemptRecord::anonymous();
    /// assert!(record.reason().is_none());
    /// ```
    #[must_use]
    pub fn anonymous() -> Self {
        Self {
            failed_at: Instant::now(),
            reason: None,
        }
    }

    /// Set the reason for this failure record.
    ///
    /// Pass `None` for an anonymous record, or `Some("reason")` for
    /// a labelled one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::AttemptRecord;
    /// let record = AttemptRecord::new("timeout").with_reason(Some("timeout".to_string()));
    /// assert_eq!(record.reason(), Some("timeout"));
    /// ```
    #[must_use]
    pub fn with_reason(mut self, reason: Option<String>) -> Self {
        self.reason = reason;
        self
    }

    /// When this failure was recorded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::AttemptRecord;
    /// let record = AttemptRecord::new("timeout");
    /// // failed_at is close to now
    /// assert!(record.failed_at().elapsed().as_secs() < 1);
    /// ```
    #[must_use]
    pub fn failed_at(&self) -> Instant {
        self.failed_at
    }

    /// The optional reason for this failure.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::AttemptRecord;
    /// let record = AttemptRecord::new("rate_limit");
    /// assert_eq!(record.reason(), Some("rate_limit"));
    /// ```
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

/// A single model in the fallback chain, with attempt history.
///
/// Each entry has a model name, a list of recorded failure attempts,
/// and a `max_fail_count` threshold. When [`attempts`](Self::attempts)
/// grows to `max_fail_count` entries, the entry is considered
/// [`failed`](Self::failed) and is automatically skipped by
/// [`FallbackManager::fallback_model`].
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackEntry;
///
/// let mut entry = FallbackEntry::new("llm-70b");
/// assert_eq!(entry.name(), "llm-70b");
/// assert!(!entry.failed());
///
/// entry.record_attempt("timeout");
/// entry.record_attempt("timeout");
/// assert_eq!(entry.attempt_count(), 2);
/// assert!(entry.failed()); // max_fail_count defaults to 2
/// ```
#[derive(Debug, Clone)]
pub struct FallbackEntry {
    /// Model identifier (e.g. `"llm-70b"`). Must match the API client's routing identifier.
    name: String,
    /// Set to `false` to take a model out of rotation independently of failure tracking.
    available: bool,
    /// Recorded failure attempts for this model.
    attempts: Vec<AttemptRecord>,
    /// When `attempts.len()` reaches this threshold the model is taken out of rotation.
    max_fail_count: usize,
}

impl FallbackEntry {
    /// Create a new entry for the given model, not yet failed.
    ///
    /// Uses a default `max_fail_count` of `2`
    /// failure marks the model as failed. Use
    /// [`with_max_fail_count`](Self::with_max_fail_count) to customise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let entry = FallbackEntry::new("llm-70b");
    /// assert_eq!(entry.name(), "llm-70b");
    /// assert!(!entry.failed());
    /// assert_eq!(entry.max_fail_count(), 2);
    /// ```
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            available: true,
            attempts: Vec::new(),
            max_fail_count: 2,
        }
    }

    /// Set a custom `max_fail_count` threshold.
    ///
    /// The model is only considered failed after this many attempts
    /// have been recorded via [`record_attempt`](Self::record_attempt).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b").with_max_fail_count(3);
    /// assert_eq!(entry.max_fail_count(), 3);
    ///
    /// entry.record_attempt("timeout");
    /// entry.record_attempt("rate_limit");
    /// assert!(!entry.failed()); // only 2 of 3
    ///
    /// entry.record_attempt("server_error");
    /// assert!(entry.failed()); // 3 of 3
    /// ```
    #[must_use]
    pub fn with_max_fail_count(mut self, max_fail_count: usize) -> Self {
        let new_max = max_fail_count.max(1);
        // If the entry was already failed, add attempts to keep it failed
        // under the new threshold.
        while self.attempts.len() < new_max && self.failed() {
            self.attempts.push(AttemptRecord::anonymous());
        }
        self.max_fail_count = new_max;
        self
    }

    /// Create a new entry already marked as failed.
    ///
    /// Useful when initializing from a known-degraded model.
    /// [`failed()`](Self::failed) returns `true` immediately.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let entry = FallbackEntry::new_failed("llm-70b");
    /// assert!(entry.failed());
    /// ```
    #[must_use]
    pub fn new_failed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            available: true,
            attempts: vec![AttemptRecord::anonymous(), AttemptRecord::anonymous()],
            max_fail_count: 2,
        }
    }

    /// The model name.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let entry = FallbackEntry::new("llm-70b");
    /// assert_eq!(entry.name(), "llm-70b");
    /// ```
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this model should be skipped.
    ///
    /// Returns `true` when the model is not [`available`](Self::available)
    /// or when [`attempt_count`](Self::attempt_count) has reached
    /// [`max_fail_count`](Self::max_fail_count).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// assert!(!entry.failed());
    /// entry.record_attempt("timeout");
    /// entry.record_attempt("timeout"); // max_fail_count defaults to 2
    /// assert!(entry.failed());
    /// ```
    #[must_use]
    pub fn failed(&self) -> bool {
        !self.available || self.attempts.len() >= self.max_fail_count
    }

    /// The configured maximum failure count before this model is skipped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let entry = FallbackEntry::new("llm-70b").with_max_fail_count(3);
    /// assert_eq!(entry.max_fail_count(), 3);
    /// ```
    #[must_use]
    pub fn max_fail_count(&self) -> usize {
        self.max_fail_count
    }

    /// Whether this model is available for use.
    ///
    /// A model that is not available is always skipped by
    /// [`FallbackManager::fallback_model`], regardless of failure count.
    /// Set to `false` via [`set_available`](Self::set_available) to take
    /// a model out of rotation (e.g. API key revoked, model decommissioned).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// assert!(entry.available());
    /// entry.set_available(false);
    /// assert!(!entry.available());
    /// assert!(entry.failed()); // unavailable ⇒ failed
    /// ```
    #[must_use]
    pub fn available(&self) -> bool {
        self.available
    }

    /// Set whether this model is available for use.
    ///
    /// When set to `false`, [`failed()`](Self::failed) returns `true`
    /// regardless of the attempt count, and the manager skips this model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// entry.set_available(false);
    /// assert!(entry.failed());
    /// entry.set_available(true);
    /// assert!(!entry.failed()); // no attempts recorded
    /// ```
    pub fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    /// How many failure attempts have been recorded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// assert_eq!(entry.attempt_count(), 0);
    /// entry.record_attempt("timeout");
    /// assert_eq!(entry.attempt_count(), 1);
    /// ```
    #[must_use]
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }

    /// Access the recorded failure attempts.
    ///
    /// Returns a slice of [`AttemptRecord`] in chronological order
    /// (oldest first).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// entry.record_attempt("timeout");
    /// entry.record_attempt("rate_limit");
    /// let attempts = entry.attempts();
    /// assert_eq!(attempts.len(), 2);
    /// assert_eq!(attempts[0].reason(), Some("timeout"));
    /// ```
    #[must_use]
    pub fn attempts(&self) -> &[AttemptRecord] {
        &self.attempts
    }

    /// Record a new failure attempt with an optional reason.
    ///
    /// If this causes [`attempt_count`](Self::attempt_count) to reach
    /// [`max_fail_count`](Self::max_fail_count), subsequent calls to
    /// [`failed()`](Self::failed) will return `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// entry.record_attempt("timeout");
    /// entry.record_attempt("timeout"); // exceeds max_fail_count (default = 2)
    /// assert!(entry.failed());
    /// ```
    pub fn record_attempt(&mut self, reason: impl Into<String>) {
        self.attempts.push(AttemptRecord::new(reason));
    }

    /// Record a new failure attempt without a reason.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// entry.record_attempt_anonymous();
    /// entry.record_attempt_anonymous(); // exceeds max_fail_count (default = 2)
    /// assert!(entry.failed());
    /// ```
    pub fn record_attempt_anonymous(&mut self) {
        self.attempts.push(AttemptRecord::anonymous());
    }

    /// Clear all recorded attempts, resetting [`failed()`](Self::failed)
    /// to `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackEntry;
    /// let mut entry = FallbackEntry::new("llm-70b");
    /// entry.record_attempt("timeout");
    /// entry.record_attempt("timeout"); // exceeds max_fail_count (default = 2)
    /// assert!(entry.failed());
    /// entry.clear_attempts();
    /// assert!(!entry.failed());
    /// ```
    pub fn clear_attempts(&mut self) {
        self.attempts.clear();
    }
}

/// Configuration for the [`FallbackManager`] circuit breaker.
///
/// Controls how aggressively the circuit trips and how cautiously it
/// recovers. These values are typically loaded from a config file or
/// environment variables and passed to [`FallbackManager::with_config`].
///
/// # Defaults
///
/// | Field                        | Default |
/// |------------------------------|---------|
/// | `trip_threshold`             | `3`     |
/// | `recovery_timeout`           | 60 s    |
/// | `recovery_successes_needed`  | `2`     |
/// | `max_fail_count`             | `2`     |
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackConfig;
/// use std::time::Duration;
///
/// let config = FallbackConfig {
///     trip_threshold: 5,
///     recovery_timeout: Duration::from_secs(120),
///     recovery_successes_needed: 3,
///     max_fail_count: 2,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Consecutive API failures required to trip the circuit open. Defaults to `3`.
    ///
    /// Once [`FallbackManager`] records this many failures in a row on the primary
    /// model, it transitions from [`FallbackState::Primary`] to
    /// [`FallbackState::Fallback`]. A higher value tolerates transient blips; a
    /// lower value reacts faster to a degrading provider.
    pub trip_threshold: usize,

    /// Minimum time in fallback before probing the primary model again. Defaults to 60 s.
    ///
    /// Cooldown enforced by [`FallbackManager::should_try_resume_primary`]: the
    /// manager stays on the fallback model for at least this long before it is
    /// willing to probe the primary again via [`FallbackState::Recovering`].
    pub recovery_timeout: Duration,

    /// Consecutive successes during recovering before the circuit fully closes. Defaults to `2`.
    ///
    /// Number of healthy responses the primary model must produce while in
    /// [`FallbackState::Recovering`] before the manager transitions back to
    /// [`FallbackState::Primary`]. Requiring more than one guards against a
    /// single lucky success masking an ongoing outage.
    pub recovery_successes_needed: usize,

    /// Per-model failure threshold before a fallback model is skipped. Defaults to `2`.
    ///
    /// Applied to each [`FallbackEntry`] in the fallback chain: once a single
    /// model accumulates this many recorded attempts, it is taken out of
    /// rotation and [`FallbackManager::active_model`] advances to the next
    /// candidate.
    pub max_fail_count: usize,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            trip_threshold: 3,
            recovery_timeout: Duration::from_mins(1),
            recovery_successes_needed: 2,
            max_fail_count: 2,
        }
    }
}

/// Manages circuit breaker state and model fallback transitions.
///
/// Tracks failures, transitions between models, and recovers back to
/// the primary model. Supports both config-driven construction via
/// [`FallbackManager::with_config`] and direct threshold control via
/// [`FallbackManager::new`].
///
/// # Thread safety
///
/// `&FallbackManager` is `Send + Sync` and can be freely shared across
/// threads (e.g. via `Arc<FallbackManager>`). No `&mut self` is needed
/// for any public method.
///
/// # Construction
///
/// Prefer [`FallbackManager::with_config`] for production use or
/// [`FallbackManager::for_model`] when you only need the framework-style
/// API. Use [`FallbackManager::new`] when you want fine-grained control
/// over thresholds.
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::{FallbackManager, FallbackConfig};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let mgr = Arc::new(FallbackManager::new(5, 3).with_config(&FallbackConfig::default()));
///
/// // Simulate failures
/// assert!(!mgr.record_model_failure()); // 1
/// assert!(!mgr.record_model_failure()); // 2
/// assert!(mgr.record_model_failure());  // 3 → circuit trips
///
/// assert!(mgr.is_using_fallback());
///
/// // ... later, after cooldown ...
/// if mgr.should_try_resume_primary(Duration::from_secs(60)) {
///     mgr.transition_to_recovering();
///     mgr.record_model_success();
///     mgr.record_model_success(); // → back to Primary
/// }
/// ```
/// Consolidated mutex-protected fallback state.
///
/// All mutable fallback bookkeeping lives behind a single lock so that
/// related fields are always observed together, preventing partial-state
/// reads that could occur when acquiring separate locks sequentially.
#[derive(Default)]
struct FallbackInner {
    /// Original model name, before any fallback was activated.
    ///
    /// Captured when the manager first trips so the recovering state
    /// knows which model to resume to. `None` until the circuit breaker
    /// has switched away from the primary at least once.
    original_model: Option<String>,

    /// Ordered list of fallback models with their per-model failure
    /// status.
    ///
    /// Each entry pairs a model name with whether it has already been
    /// tried-and-failed. The manager walks this list in order when
    /// selecting the next fallback, skipping entries marked failed.
    fallback_models: Vec<FallbackEntry>,

    /// Cached name of the first non-failed fallback model.
    ///
    /// Computed once and reused while the manager is in the fallback
    /// state, so callers asking "which model am I on now?" get an
    /// `O(1)` answer without re-scanning `fallback_models`. Recomputed
    /// whenever the set of failed entries changes.
    active_fallback: Option<String>,

    /// Instant at which fallback was activated.
    ///
    /// Recorded when the manager transitions from `Primary` to
    /// `Fallback`, and compared against the current time to compute
    /// cooldown eligibility for
    /// [`should_try_resume_primary`](FallbackManager::should_try_resume_primary).
    /// `None` while the manager has never left the primary.
    fallback_switched_at: Option<Instant>,
}

/// Circuit breaker for API model fallback.
///
/// `&FallbackManager` is `Send + Sync` and can be freely shared across
/// threads (e.g. via `Arc<FallbackManager>`). No `&mut self` is needed
/// for any operation.
pub struct FallbackManager {
    /// Number of consecutive failures on the primary that trips the
    /// circuit.
    ///
    /// Once [`consecutive_failures`](Self::consecutive_failures) reaches
    /// this value the manager transitions from `Primary` to `Fallback`.
    /// Set at construction and immutable thereafter.
    fallback_threshold: usize,

    /// Number of consecutive successes required on the primary during
    /// recovery before transitioning back.
    ///
    /// Counts towards this via
    /// [`record_model_success`](FallbackManager::record_model_success)
    /// while in the `Recovering` state; reaching it transitions to
    /// `Primary`. Set at construction and immutable thereafter.
    primary_resume_threshold: usize,

    /// Per-model max failure count seeded into new
    /// [`FallbackEntry`] instances.
    ///
    /// When a fallback model is added without an explicit cap, it
    /// inherits this value as its `max_fail_count`. Set at construction
    /// and immutable thereafter.
    default_max_fail_count: usize,

    /// Consecutive API failure counter on the current model.
    ///
    /// Incremented by
    /// [`record_model_failure`](FallbackManager::record_model_failure),
    /// reset to zero on any success. Hitting `fallback_threshold` trips
    /// the circuit. Atomic so it updates lock-free on the hot path.
    consecutive_failures: AtomicUsize,

    /// Sticky flag recording whether fallback has ever been activated.
    ///
    /// `true` once the circuit has tripped at least once, and never
    /// reset for the life of the manager. Lets observers tell "still
    /// on primary, never failed" from "recovered back to primary".
    fallback_activated: AtomicBool,

    /// Current circuit-breaker state, encoded as an atomic.
    ///
    /// `0` = `Primary`, `1` = `Fallback`, `2` = `Recovering` (see
    /// [`FallbackState`]). Stored as `AtomicU8` for lock-free reads on
    /// every request; mutated only on state transitions.
    fallback_state: AtomicU8,

    /// Consecutive successes on the primary accumulated during the
    /// `Recovering` state.
    ///
    /// Incremented by
    /// [`record_model_success`](FallbackManager::record_model_success);
    /// reaching `primary_resume_threshold` transitions back to
    /// `Primary` and resets this to zero. Atomic so it updates
    /// lock-free.
    primary_success_count: AtomicUsize,

    /// Consolidated mutex-protected fallback state.
    ///
    /// Holding all related fields behind a single lock prevents partial-state
    /// reads that could occur when acquiring the (formerly separate) locks one
    /// at a time.
    inner: Mutex<FallbackInner>,

    /// How long to remain in fallback before attempting primary
    /// recovery.
    ///
    /// Compared against the elapsed time since
    /// [`fallback_switched_at`](FallbackInner::fallback_switched_at)
    /// by
    /// [`should_try_resume_primary`](FallbackManager::should_try_resume_primary)
    /// to decide when a cooldown has elapsed. Set at construction and
    /// immutable thereafter.
    recovery_timeout: Duration,
}

impl FallbackManager {
    /// Create a new fallback manager with the given thresholds.
    ///
    /// Starts in [`FallbackState::Primary`] with all counters zeroed.
    /// Use this constructor when you want to control the exact numeric
    /// thresholds. For config-file-driven construction see
    /// [`FallbackManager::with_config`].
    ///
    /// # Parameters
    ///
    /// * `fallback_threshold` — number of consecutive failures before the
    ///   circuit trips to [`FallbackState::Fallback`].
    /// * `primary_resume_threshold` — number of consecutive successes on
    ///   the primary model during [`FallbackState::Recovering`] needed to
    ///   close the circuit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(5, 3);
    /// // Trip after 5 failures, resume after 3 consecutive successes
    /// ```
    #[must_use]
    pub fn new(fallback_threshold: usize, primary_resume_threshold: usize) -> Self {
        Self {
            fallback_threshold,
            primary_resume_threshold,
            default_max_fail_count: 2,
            consecutive_failures: AtomicUsize::new(0),
            fallback_activated: AtomicBool::new(false),
            fallback_state: AtomicU8::new(FallbackState::Primary as u8),
            primary_success_count: AtomicUsize::new(0),
            inner: Mutex::new(FallbackInner::default()),
            recovery_timeout: Duration::from_mins(1),
        }
    }

    /// Apply configuration from a [`FallbackConfig`] struct.
    ///
    /// Sets the failure threshold, recovery parameters, and per-model
    /// max fail count from the config.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackConfig};
    /// use std::time::Duration;
    ///
    /// let config = FallbackConfig {
    ///     trip_threshold: 5,
    ///     recovery_timeout: Duration::from_secs(120),
    ///     recovery_successes_needed: 3,
    ///     max_fail_count: 2,
    /// };
    /// let mgr = FallbackManager::new(5, 3).with_config(&config);
    /// ```
    #[must_use]
    pub fn with_config(mut self, config: &FallbackConfig) -> Self {
        self.fallback_threshold = config.trip_threshold;
        self.primary_resume_threshold = config.recovery_successes_needed;
        self.default_max_fail_count = config.max_fail_count;
        self.recovery_timeout = config.recovery_timeout;
        self
    }

    /// Set the recovery timeout (builder style).
    ///
    /// This is how long the manager stays in fallback before it is willing
    /// to probe the primary model again via
    /// [`should_try_resume_primary`](Self::should_try_resume_primary).
    ///
    /// Mirrors [`FallbackConfig::recovery_timeout`] for cases where a full
    /// [`with_config`](Self::with_config) is not desired.
    #[must_use]
    pub fn with_recovery_timeout(mut self, recovery_timeout: Duration) -> Self {
        self.recovery_timeout = recovery_timeout;
        self
    }

    /// Configured recovery timeout.
    ///
    /// Returns the duration the manager will remain in fallback before it
    /// is willing to probe the primary model again. Set via
    /// [`with_config`](Self::with_config) (from
    /// [`FallbackConfig::recovery_timeout`]) or
    /// [`with_recovery_timeout`](Self::with_recovery_timeout).
    ///
    /// Pass this to
    /// [`should_try_resume_primary`](Self::should_try_resume_primary) to
    /// honour the configured timeout without hard-coding a value.
    #[must_use]
    pub fn recovery_timeout(&self) -> Duration {
        self.recovery_timeout
    }

    /// Create with fallback already activated.
    ///
    /// Useful when a new manager should start in the
    /// [`FallbackState::Fallback`] state — for instance when
    /// the primary model is already known to be degraded.
    /// The `original_model` is stored for later recovery via
    /// [`should_try_resume_primary`](Self::should_try_resume_primary).
    ///
    /// # Parameters
    ///
    /// * `original_model` — the model name to remember for future recovery.
    /// * `fallback_threshold` — used for future re-tripping if recovery fails.
    ///
    /// # Initial state
    ///
    /// - [`fallback_activated`](Self::is_fallback_active) = `true`
    /// - [`state()`](Self::state) = [`FallbackState::Fallback`]
    /// - [`consecutive_failures()`](Self::consecutive_failures) = `0`
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new_with_fallback("llm-70b".into(), 3);
    /// assert!(mgr.is_using_fallback());
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    #[must_use]
    pub fn new_with_fallback(original_model: String, fallback_threshold: usize) -> Self {
        let mgr = Self::new(fallback_threshold, 2);
        if let Ok(mut inner) = mgr.inner.lock() {
            inner.original_model = Some(original_model);
            inner.fallback_switched_at = Some(Instant::now());
        }
        mgr.fallback_activated.store(true, Ordering::Relaxed);
        mgr.consecutive_failures.store(0, Ordering::Relaxed);
        mgr.fallback_state
            .store(FallbackState::Fallback as u8, Ordering::Relaxed);
        mgr
    }

    /// Create a new manager with a primary model name.
    ///
    /// The model name is stored as
    /// [`original_model`](Self::original_model) for later retrieval via
    /// [`active_model`](Self::active_model). Uses default thresholds
    /// (trip after 3 failures, resume after 2 successes).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// assert!(!mgr.is_using_fallback());
    /// ```
    pub fn for_model(primary_model: impl Into<String>) -> Self {
        let mgr = Self::new(3, 2);
        if let Ok(mut inner) = mgr.inner.lock() {
            inner.original_model = Some(primary_model.into());
        }
        mgr
    }

    /// Get the current circuit breaker state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// ```
    pub fn state(&self) -> FallbackState {
        FallbackState::from(self.fallback_state.load(Ordering::Relaxed))
    }

    /// Check if we're currently using a fallback model.
    ///
    /// Returns `true` only when the circuit is in
    /// [`FallbackState::Fallback`]. During
    /// [`FallbackState::Recovering`] the manager is probing the primary
    /// model (half-open state), so this returns `false` — callers should
    /// route requests to the primary.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.is_using_fallback());
    /// ```
    pub fn is_using_fallback(&self) -> bool {
        matches!(self.state(), FallbackState::Fallback)
    }

    /// Check if fallback has ever been activated (sticky flag).
    ///
    /// Unlike [`is_using_fallback`](Self::is_using_fallback), this flag
    /// remains `true` even after the circuit recovers — it records
    /// whether the circuit *has ever* tripped during this session.
    /// Useful for diagnostics and metrics. Only cleared by
    /// [`reset`](Self::reset) or [`transition_to_primary`](Self::transition_to_primary).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.is_fallback_active());
    /// ```
    pub fn is_fallback_active(&self) -> bool {
        self.fallback_activated.load(Ordering::Relaxed)
    }

    /// Get the number of consecutive failures.
    ///
    /// The count since the last success (when in
    /// [`FallbackState::Primary`]) or since the circuit tripped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// mgr.record_model_failure();
    /// assert_eq!(mgr.consecutive_failures(), 1);
    /// ```
    pub fn consecutive_failures(&self) -> usize {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Get the original model name (before fallback).
    ///
    /// Returns the model name stored at construction time via
    /// [`for_model`](Self::for_model) or
    /// [`new_with_fallback`](Self::new_with_fallback), or set later
    /// with [`set_original_model`](Self::set_original_model).
    /// Returns `None` if no model name has been set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn original_model(&self) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.original_model.clone())
    }

    /// Set the original model name.
    ///
    /// Overwrites the stored primary model name. Useful when the model
    /// is resolved from configuration or changed mid-session.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.original_model(), None);
    /// mgr.set_original_model("llm-70b".into());
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn set_original_model(&self, model: String) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.original_model = Some(model);
        }
    }

    /// Get the time when fallback was activated.
    ///
    /// Returns `Some(Instant)` if the circuit has transitioned to
    /// [`FallbackState::Fallback`] at least once (and has not yet
    /// recovered). Returns `None` if the circuit has never tripped or
    /// has already recovered back to [`FallbackState::Primary`].
    /// Used by [`should_try_resume_primary`](Self::should_try_resume_primary)
    /// to enforce the cooldown period.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// use std::time::Duration;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(mgr.fallback_switched_at().is_none());
    /// ```
    pub fn fallback_switched_at(&self) -> Option<Instant> {
        self.inner.lock().ok().and_then(|i| i.fallback_switched_at)
    }

    /// Get the model that should be used for the next request.
    ///
    /// When the circuit is in [`FallbackState::Fallback`], returns the
    /// fallback model (if one has been set via
    /// [`set_fallback_model`](Self::set_fallback_model) or
    /// [`add_fallback_model`](Self::add_fallback_model)),
    /// falling back to the original model if no dedicated fallback is
    /// configured. When the circuit is in [`FallbackState::Primary`] or
    /// [`FallbackState::Recovering`] (half-open probe), always returns the
    /// original model — the manager is testing whether the primary has
    /// recovered.
    ///
    /// # Returns
    ///
    /// * `Some(model_name)` — the model the caller should use for the next
    ///   LLM request.
    /// * `None` — no model has been configured (neither primary nor fallback).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.active_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn active_model(&self) -> Option<String> {
        match self.state() {
            FallbackState::Primary | FallbackState::Recovering => self.original_model(),
            FallbackState::Fallback => self.fallback_model().or_else(|| self.original_model()),
        }
    }

    /// Set a single fallback model, clearing any existing chain.
    ///
    /// Resets the fallback chain to contain only the given model. Use this
    /// for simple single-fallback setups. For multi-model chains, use
    /// [`add_fallback_model`](Self::add_fallback_model) to build up the
    /// chain incrementally, or [`set_fallback_models`](Self::set_fallback_models)
    /// to set the entire chain at once.
    ///
    /// # Parameters
    ///
    /// * `model` — the fallback model identifier (e.g. `"llm-70b"`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::for_model("llm-1");
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    /// assert_eq!(mgr.fallback_models(), vec!["llm-2", "llm-3"]);
    ///
    /// mgr.set_fallback_model("llm-4"); // clears chain, sets single model
    /// assert_eq!(mgr.fallback_models(), vec!["llm-4"]);
    ///
    /// // Simulate failures until circuit trips
    /// mgr.record_model_failure();
    /// mgr.record_model_failure();
    /// mgr.record_model_failure(); // trips to Fallback
    ///
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// assert_eq!(mgr.active_model(), Some("llm-4".to_string()));
    /// ```
    pub fn set_fallback_model(&self, model: impl Into<String>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.fallback_models.clear();
            inner
                .fallback_models
                .push(FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count));
        }
        self.recompute_active_fallback();
    }

    /// Get the first fallback model, if any are configured.
    ///
    /// Returns the model name at index `0` of the fallback chain (the
    /// highest-priority fallback), or `None` if no fallback models have
    /// been configured.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(mgr.fallback_model().is_none());
    ///
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    /// assert_eq!(mgr.fallback_model(), Some("llm-70b".to_string())); // first in chain
    /// ```
    pub fn fallback_model(&self) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.active_fallback.clone())
    }

    /// Get the full fallback model chain.
    ///
    /// Returns all configured fallback models in priority order (index `0`
    /// is the first fallback tried).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    /// mgr.add_fallback_model("llm-32b");
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b", "llm-32b"]);
    /// ```
    pub fn fallback_models(&self) -> Vec<String> {
        self.inner
            .lock()
            .ok()
            .map(|i| i.fallback_models.iter().map(|e| e.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Add a fallback model to the end of the fallback chain.
    ///
    /// Appends the model as the lowest-priority fallback (last resort).
    /// Use [`insert_fallback_model`](Self::insert_fallback_model) to add
    /// at a specific position in the chain.
    ///
    /// # Parameters
    ///
    /// * `model` — the fallback model identifier.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b"]);
    /// ```
    pub fn add_fallback_model(&self, model: impl Into<String>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .fallback_models
                .push(FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count));
        }
        self.recompute_active_fallback();
    }

    /// Insert a fallback model at a specific position in the chain.
    ///
    /// Models at and after the insertion index are shifted to the right.
    /// If `index` is beyond the end of the chain, the model is appended.
    ///
    /// # Parameters
    ///
    /// * `index` — the position in the chain (0 = highest priority).
    /// * `model` — the fallback model identifier.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-32b");
    /// mgr.insert_fallback_model(1, "llm-120b"); // insert between them
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b", "llm-32b"]);
    /// ```
    pub fn insert_fallback_model(&self, index: usize, model: impl Into<String>) {
        if let Ok(mut inner) = self.inner.lock() {
            let entry = FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count);
            if index >= inner.fallback_models.len() {
                inner.fallback_models.push(entry);
            } else {
                inner.fallback_models.insert(index, entry);
            }
        }
        self.recompute_active_fallback();
    }

    /// Remove a fallback model by name.
    ///
    /// Removes the first occurrence of the given model name from the chain.
    /// Returns `true` if a model was removed, `false` if the name was not
    /// found in the chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    ///
    /// assert!(mgr.remove_fallback_model("llm-70b"));
    /// assert!(!mgr.remove_fallback_model("nonexistent"));
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-120b"]);
    /// ```
    pub fn remove_fallback_model(&self, model: &str) -> bool {
        let removed = if let Ok(mut inner) = self.inner.lock() {
            if let Some(pos) = inner.fallback_models.iter().position(|x| x.name == model) {
                inner.fallback_models.remove(pos);
                true
            } else {
                false
            }
        } else {
            false
        };
        if removed {
            self.recompute_active_fallback();
        }
        removed
    }

    /// Replace the entire fallback model chain.
    ///
    /// Clears any existing fallback models and sets the provided list
    /// as the new chain, in the given order.
    ///
    /// # Parameters
    ///
    /// * `models` — fallback model identifiers in priority order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.set_fallback_models(vec!["llm-70b".into(), "llm-120b".into(), "llm-32b".into()]);
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b", "llm-32b"]);
    /// ```
    pub fn set_fallback_models(&self, models: Vec<String>) {
        let max_fc = self.default_max_fail_count;
        if let Ok(mut inner) = self.inner.lock() {
            inner.fallback_models = models
                .into_iter()
                .map(|name| FallbackEntry::new(name).with_max_fail_count(max_fc))
                .collect();
        }
        self.recompute_active_fallback();
    }

    /// Mark a fallback model as failed by name.
    ///
    /// Call this when the fallback model itself returns errors. The model
    /// is not removed from the chain — it is flagged so that
    /// [`active_model`](Self::active_model) skips it and returns the next
    /// non-failed model instead. Returns `true` if the model was found
    /// and marked, `false` if no model with that name exists in the chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    ///
    /// assert!(mgr.mark_fallback_failed("llm-2"));
    /// assert!(!mgr.mark_fallback_failed("nonexistent"));
    ///
    /// // Mark again to exceed max_fail_count (default = 2)
    /// mgr.mark_fallback_failed("llm-2");
    ///
    /// // active_model skips failed, returns next available
    /// assert_eq!(mgr.fallback_model(), Some("llm-3".to_string()));
    /// ```
    pub fn mark_fallback_failed(&self, model: &str) -> bool {
        let found = if let Ok(mut inner) = self.inner.lock() {
            if let Some(entry) = inner.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.record_attempt("marked_failed");
                true
            } else {
                false
            }
        } else {
            false
        };
        if found {
            self.recompute_active_fallback();
        }
        found
    }

    /// Clear the failed flag on a fallback model by name.
    ///
    /// Use this to retry a previously failed fallback model — for example
    /// after a cooldown period. Returns `true` if the model was found and
    /// its flag cleared, `false` if no model with that name exists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.mark_fallback_failed("llm-2");
    /// mgr.mark_fallback_failed("llm-2"); // exceeds max_fail_count (default = 2)
    /// assert!(mgr.failed_fallbacks().contains(&"llm-2".to_string()));
    ///
    /// mgr.clear_fallback_failed("llm-2");
    /// assert!(mgr.failed_fallbacks().is_empty());
    /// ```
    pub fn clear_fallback_failed(&self, model: &str) -> bool {
        let found = if let Ok(mut inner) = self.inner.lock() {
            if let Some(entry) = inner.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.clear_attempts();
                true
            } else {
                false
            }
        } else {
            false
        };
        if found {
            self.recompute_active_fallback();
        }
        found
    }

    /// Clear all failed flags, making every fallback model available again.
    ///
    /// Called by [`reset`](Self::reset) and when the circuit recovers
    /// back to the primary model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    /// mgr.mark_fallback_failed("llm-2");
    /// mgr.mark_fallback_failed("llm-2"); // exceeds max_fail_count
    /// mgr.mark_fallback_failed("llm-3");
    /// mgr.mark_fallback_failed("llm-3"); // exceeds max_fail_count
    /// assert_eq!(mgr.failed_fallbacks().len(), 2);
    ///
    /// mgr.clear_all_fallback_failed();
    /// assert!(mgr.failed_fallbacks().is_empty());
    /// ```
    pub fn clear_all_fallback_failed(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            for entry in &mut inner.fallback_models {
                entry.clear_attempts();
            }
        }
        self.recompute_active_fallback();
    }

    /// Get the names of all fallback models marked as failed.
    ///
    /// Returns model names in chain order, filtered to only those with
    /// the failed flag set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    /// mgr.mark_fallback_failed("llm-3");
    /// mgr.mark_fallback_failed("llm-3"); // exceeds max_fail_count (default = 2)
    ///
    /// let failed = mgr.failed_fallbacks();
    /// assert_eq!(failed, vec!["llm-3"]);
    /// ```
    pub fn failed_fallbacks(&self) -> Vec<String> {
        self.inner
            .lock()
            .ok()
            .map(|i| {
                i.fallback_models
                    .iter()
                    .filter(|e| e.failed())
                    .map(|e| e.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the names of all non-failed (available) fallback models.
    ///
    /// Returns model names in chain order, filtered to only those
    /// without the failed flag.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    /// mgr.mark_fallback_failed("llm-3");
    /// mgr.mark_fallback_failed("llm-3"); // exceeds max_fail_count (default = 2)
    ///
    /// let available = mgr.available_fallbacks();
    /// assert_eq!(available, vec!["llm-2"]);
    /// ```
    pub fn available_fallbacks(&self) -> Vec<String> {
        self.inner
            .lock()
            .ok()
            .map(|i| {
                i.fallback_models
                    .iter()
                    .filter(|e| !e.failed())
                    .map(|e| e.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Look up a fallback entry by name.
    ///
    /// Returns a cloned [`FallbackEntry`] for the first model in the chain
    /// whose name matches, or `None` if the name is not present. Use this
    /// to inspect a specific model's failure status without iterating the
    /// full chain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    ///
    /// let entry = mgr.fallback_entry("llm-70b");
    /// assert!(entry.is_some());
    /// let e = entry.unwrap();
    /// assert_eq!(e.name(), "llm-70b");
    /// assert!(!e.failed()); // not failed yet
    ///
    /// assert!(mgr.fallback_entry("nonexistent").is_none());
    /// ```
    pub fn fallback_entry(&self, name: &str) -> Option<FallbackEntry> {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.fallback_models.iter().find(|e| e.name == name).cloned())
    }

    /// Set the [`available`](FallbackEntry::available) flag on a fallback model.
    ///
    /// When set to `false`, the entry is considered [`failed`](FallbackEntry::failed)
    /// regardless of its attempt count, and [`active_model`](Self::active_model)
    /// will skip it. When set back to `true`, the entry becomes eligible again
    /// (unless its attempt count has also reached [`FallbackEntry::max_fail_count`]).
    ///
    /// Returns `true` if the model was found in the chain and updated,
    /// `false` if no model with that name exists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-2");
    /// mgr.add_fallback_model("llm-3");
    ///
    /// // Take llm-2 out of rotation (e.g. API key revoked)
    /// assert!(mgr.set_fallback_available("llm-2", false));
    ///
    /// // active_model now skips llm-2
    /// assert_eq!(mgr.fallback_model(), Some("llm-3".to_string()));
    ///
    /// // Bring it back
    /// mgr.set_fallback_available("llm-2", true);
    /// assert_eq!(mgr.fallback_model(), Some("llm-2".to_string()));
    /// ```
    pub fn set_fallback_available(&self, model: &str, available: bool) -> bool {
        let found = if let Ok(mut inner) = self.inner.lock() {
            if let Some(entry) = inner.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.set_available(available);
                true
            } else {
                false
            }
        } else {
            false
        };
        if found {
            self.recompute_active_fallback();
        }
        found
    }

    /// Record an API failure and check if fallback should be triggered.
    ///
    /// Called by the agent loop each time an LLM API call fails (e.g.
    /// rate limit, server error, timeout). Returns `true` when the
    /// consecutive failure count reaches the configured
    /// [`fallback_threshold`](FallbackManager::new) and the circuit
    /// has not already been activated.
    ///
    /// # Returns
    ///
    /// * `true` — the failure count reached the threshold for
    ///   the first time; the caller should switch to the fallback model.
    /// * `false` — either the threshold hasn't been reached yet, or the
    ///   circuit has already been tripped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.record_api_failure()); // 1
    /// assert!(!mgr.record_api_failure()); // 2
    /// assert!(mgr.record_api_failure());  // 3 — threshold reached, now activated
    /// assert!(!mgr.record_api_failure()); // 4 — already activated, no re-trip
    /// ```
    pub fn record_api_failure(&self) -> bool {
        match self.state() {
            FallbackState::Primary => {
                let failures = self
                    .consecutive_failures
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1);
                if failures >= self.fallback_threshold {
                    warn!(
                        consecutive_failures = failures,
                        threshold = self.fallback_threshold,
                        "Fallback threshold reached"
                    );
                    self.transition_to_fallback();
                    return true;
                }
                false
            }
            FallbackState::Fallback | FallbackState::Recovering => {
                let failures = self.consecutive_failures.load(Ordering::Relaxed);
                warn!(
                    consecutive_failures = failures,
                    "API failure recorded while not in Primary state; counter unchanged"
                );
                false
            }
        }
    }

    /// Alias for [`record_api_failure`](Self::record_api_failure) — framework-style name.
    ///
    /// Provided for callers that prefer the shorter `record_failure` name.
    /// Delegates directly to [`record_api_failure`](Self::record_api_failure).
    pub fn record_failure(&self) -> bool {
        self.record_api_failure()
    }

    /// Reset the consecutive failure counter.
    ///
    /// Sets [`consecutive_failures`](Self::consecutive_failures) back to
    /// `0` without changing the circuit state. Typically called by the
    /// framework when a manual reset is desired (e.g. user intervention)
    /// rather than through the normal success/failure recording flow.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.record_api_failure();
    /// mgr.record_api_failure();
    /// assert_eq!(mgr.consecutive_failures(), 2);
    /// mgr.reset_failure_counter();
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn reset_failure_counter(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    /// Record a success on the current model.
    ///
    /// Called by the agent loop after a successful LLM response. The
    /// effect depends on the current circuit state:
    ///
    /// - **[`Primary`](FallbackState::Primary)**: Resets the failure
    ///   counter to `0`, confirming the primary model is healthy.
    /// - **[`Fallback`](FallbackState::Fallback)**: No-op. Successes on
    ///   the fallback model don't affect recovery — the manager waits
    ///   for the cooldown period to expire first.
    /// - **[`Recovering`](FallbackState::Recovering)**: Increments the
    ///   success counter. If it reaches the configured threshold, the
    ///   circuit closes via [`transition_to_primary`](Self::transition_to_primary).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::FallbackManager;
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.record_api_failure();
    /// assert_eq!(mgr.consecutive_failures(), 1);
    /// mgr.record_model_success(); // resets failures to 0
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn record_model_success(&self) {
        match self.state() {
            FallbackState::Primary => {
                self.consecutive_failures.store(0, Ordering::Relaxed);
            }
            FallbackState::Fallback => {
                // Success on fallback, stay in fallback
            }
            FallbackState::Recovering => {
                let successes = self
                    .primary_success_count
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1);
                debug!(
                    successes,
                    threshold = self.primary_resume_threshold,
                    "Primary model success during recovery test"
                );
                if successes >= self.primary_resume_threshold {
                    self.transition_to_primary();
                }
            }
        }
    }

    /// Alias for [`record_model_success`](Self::record_model_success) — framework-style name.
    ///
    /// Provided for callers that prefer the shorter `record_success` name.
    /// Delegates directly to [`record_model_success`](Self::record_model_success).
    pub fn record_success(&self) {
        self.record_model_success();
    }

    /// Record a failure on the current model.
    ///
    /// Called by the agent loop when an LLM request fails. The effect
    /// depends on the current circuit state:
    ///
    /// - **[`Primary`](FallbackState::Primary)**: Increments the failure
    ///   counter. If the count reaches the threshold, transitions to
    ///   [`Fallback`](FallbackState::Fallback) via
    ///   [`transition_to_fallback`](Self::transition_to_fallback).
    /// - **[`Fallback`](FallbackState::Fallback)**: Logs a warning —
    ///   the fallback model itself is experiencing failures. The circuit
    ///   stays open.
    /// - **[`Recovering`](FallbackState::Recovering)**: Immediately
    ///   reopens the circuit back to [`Fallback`](FallbackState::Fallback)
    ///   — the primary is not yet healthy.
    ///
    /// # Returns
    ///
    /// `true` if this specific failure caused the circuit to trip from
    /// `Primary` to `Fallback`; `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.record_model_failure()); // 1
    /// assert!(!mgr.record_model_failure()); // 2
    /// assert!(mgr.record_model_failure());  // 3 → trips to Fallback
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// ```
    pub fn record_model_failure(&self) -> bool {
        match self.state() {
            FallbackState::Primary => {
                let failures = self
                    .consecutive_failures
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1);
                if failures >= self.fallback_threshold {
                    self.transition_to_fallback();
                    return true;
                }
                false
            }
            FallbackState::Fallback => {
                let fb_name = self
                    .fallback_model()
                    .unwrap_or_else(|| "unknown".to_string());
                warn!(
                    "Fallback model \"{fb_name}\" also experiencing failures; consider calling mark_fallback_failed(\"{fb_name}\") to skip it"
                );
                false
            }
            FallbackState::Recovering => {
                warn!("Primary model failed during recovery test, staying on fallback");
                self.transition_to_fallback();
                false
            }
        }
    }

    /// Check if we should try resuming the primary model.
    ///
    /// Returns `true` if **both** conditions hold:
    ///
    /// 1. The circuit is currently in [`FallbackState::Fallback`].
    /// 2. The elapsed time since [`fallback_switched_at`](Self::fallback_switched_at)
    ///    is at least `min_fallback_duration`.
    ///
    /// Called by the agent loop before each turn while in the fallback
    /// state. When this returns `true`, the caller should call
    /// [`transition_to_recovering`](Self::transition_to_recovering) to
    /// begin probing the primary model.
    ///
    /// # Parameters
    ///
    /// * `min_fallback_duration` — minimum time to stay in fallback
    ///   before attempting recovery (typically from
    ///   [`FallbackConfig::recovery_timeout`]).
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::time::Duration;
    /// use loopctl::fallback::FallbackManager;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// // Not in fallback state → false
    /// assert!(!mgr.should_try_resume_primary(Duration::from_secs(10)));
    /// ```
    pub fn should_try_resume_primary(&self, min_fallback_duration: Duration) -> bool {
        if self.state() != FallbackState::Fallback {
            return false;
        }
        if let Some(switched_at) = self.fallback_switched_at() {
            switched_at.elapsed() >= min_fallback_duration
        } else {
            false
        }
    }

    /// Transition to using fallback model (circuit open).
    ///
    /// Moves the circuit breaker to [`FallbackState::Fallback`], records
    /// the current time as [`fallback_switched_at`](Self::fallback_switched_at),
    /// and resets the recovery success counter. Called automatically by
    /// [`record_model_failure`](Self::record_model_failure) when the
    /// failure threshold is reached, or manually when the circuit needs
    /// to trip immediately.
    ///
    /// After this call, [`is_using_fallback`](Self::is_using_fallback)
    /// returns `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// ```
    pub fn transition_to_fallback(&self) {
        self.fallback_state
            .store(FallbackState::Fallback as u8, Ordering::Relaxed);
        self.fallback_activated.store(true, Ordering::Relaxed);
        if let Ok(mut inner) = self.inner.lock() {
            inner.fallback_switched_at = Some(Instant::now());
        }
        self.primary_success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker: transitioned to Fallback state");
    }

    /// Transition to testing primary model (half-open state).
    ///
    /// Moves the circuit breaker from [`FallbackState::Fallback`] to
    /// [`FallbackState::Recovering`] and resets the recovery success
    /// counter. Called by the agent loop after
    /// [`should_try_resume_primary`](Self::should_try_resume_primary)
    /// returns `true`. Subsequent calls to
    /// [`record_model_success`](Self::record_model_success) will count
    /// toward the recovery threshold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// mgr.transition_to_recovering();
    /// assert_eq!(mgr.state(), FallbackState::Recovering);
    /// ```
    pub fn transition_to_recovering(&self) {
        self.fallback_state
            .store(FallbackState::Recovering as u8, Ordering::Relaxed);
        self.primary_success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker: transitioned to Recovering state (testing primary)");
    }

    /// Transition back to primary model (circuit closed).
    ///
    /// Moves the circuit breaker to [`FallbackState::Primary`], clears
    /// the fallback timestamp, and resets all counters (failures,
    /// successes, and the `fallback_activated` flag). Called
    /// automatically when the recovery success threshold is reached
    /// inside [`record_model_success`](Self::record_model_success), or
    /// manually to force an immediate return to primary.
    ///
    /// After this call, [`is_using_fallback`](Self::is_using_fallback)
    /// returns `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// mgr.transition_to_primary();
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// assert!(!mgr.is_using_fallback());
    /// ```
    pub fn transition_to_primary(&self) {
        self.fallback_state
            .store(FallbackState::Primary as u8, Ordering::Relaxed);
        if let Ok(mut inner) = self.inner.lock() {
            inner.fallback_switched_at = None;
        }
        self.primary_success_count.store(0, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.fallback_activated.store(false, Ordering::Relaxed);
        self.clear_all_fallback_failed();
        info!("Circuit breaker: transitioned to Primary state (primary model recovered)");
    }

    /// Reset the circuit breaker to [`FallbackState::Primary`] state.
    ///
    /// Performs a full reset: sets the state to `Primary`, zeros all
    /// counters, clears the `fallback_activated` flag, and clears the
    /// fallback timestamp. Hard reset — erases all
    /// failure history and is equivalent to creating a new manager.
    ///
    /// Use this when you want to force the circuit back to its initial
    /// state (e.g. after a configuration change or user intervention)
    /// rather than going through the normal recovery flow.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// for _ in 0..3 { mgr.record_model_failure(); }
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    ///
    /// mgr.reset();
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// assert!(!mgr.is_fallback_active());
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn reset(&self) {
        self.fallback_state
            .store(FallbackState::Primary as u8, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.primary_success_count.store(0, Ordering::Relaxed);
        self.fallback_activated.store(false, Ordering::Relaxed);
        if let Ok(mut inner) = self.inner.lock() {
            inner.fallback_switched_at = None;
        }
        self.clear_all_fallback_failed();
    }

    /// Recompute the cached [`active_fallback`](Self::active_fallback) from
    /// the current fallback chain.
    ///
    /// [`fallback_models`]: Self::fallback_models
    /// [`active_fallback`]: Self::active_fallback
    fn recompute_active_fallback(&self) {
        let active = self.inner.lock().ok().and_then(|i| {
            i.fallback_models
                .iter()
                .find(|e| !e.failed())
                .map(|e| e.name.clone())
        });
        if let Ok(mut inner) = self.inner.lock() {
            inner.active_fallback = active;
        }
    }
}

impl Default for FallbackManager {
    fn default() -> Self {
        Self::new(3, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let mgr = FallbackManager::new(3, 2);
        assert_eq!(mgr.state(), FallbackState::Primary);
        assert!(!mgr.is_using_fallback());
        assert!(!mgr.is_fallback_active());
        assert_eq!(mgr.consecutive_failures(), 0);
    }

    #[test]
    fn test_failure_threshold() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.record_api_failure()); // 1
        assert!(!mgr.record_api_failure()); // 2
        assert!(mgr.record_api_failure()); // 3 — threshold reached
    }

    #[test]
    fn test_model_failure_triggers_fallback() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.record_model_failure()); // 1
        assert!(!mgr.record_model_failure()); // 2
        assert!(mgr.record_model_failure()); // 3 — triggers fallback
        assert_eq!(mgr.state(), FallbackState::Fallback);
    }

    #[test]
    fn test_recovery() {
        let mgr = FallbackManager::new(3, 2);
        // Trigger fallback
        for _ in 0..3 {
            mgr.record_model_failure();
        }
        assert_eq!(mgr.state(), FallbackState::Fallback);

        // Transition to recovering
        mgr.transition_to_recovering();
        assert_eq!(mgr.state(), FallbackState::Recovering);

        // Recover after enough successes
        mgr.record_model_success(); // 1
        mgr.record_model_success(); // 2 — threshold reached
        assert_eq!(mgr.state(), FallbackState::Primary);
    }

    #[test]
    fn test_recovery_failure_goes_back_to_fallback() {
        let mgr = FallbackManager::new(3, 2);
        for _ in 0..3 {
            mgr.record_model_failure();
        }
        mgr.transition_to_recovering();
        mgr.record_model_failure(); // failure during recovery
        assert_eq!(mgr.state(), FallbackState::Fallback);
    }

    #[test]
    fn test_should_try_resume_primary() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.should_try_resume_primary(Duration::from_secs(10)));

        // Trigger fallback
        for _ in 0..3 {
            mgr.record_model_failure();
        }
        // Not enough time
        assert!(!mgr.should_try_resume_primary(Duration::from_hours(1)));
        // Enough time (0s timeout)
        assert!(mgr.should_try_resume_primary(Duration::from_secs(0)));
    }

    #[test]
    fn test_new_with_fallback() {
        let mgr = FallbackManager::new_with_fallback("llm-70b".into(), 3);
        assert!(mgr.is_fallback_active());
        assert!(mgr.is_using_fallback());
        assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    }

    #[test]
    fn test_reset() {
        let mgr = FallbackManager::new(3, 2);
        for _ in 0..3 {
            mgr.record_model_failure();
        }
        assert_eq!(mgr.state(), FallbackState::Fallback);

        mgr.reset();
        assert_eq!(mgr.state(), FallbackState::Primary);
        assert!(!mgr.is_fallback_active());
        assert_eq!(mgr.consecutive_failures(), 0);
    }

    #[test]
    fn test_api_failure_does_not_retrip() {
        let mgr = FallbackManager::new(3, 2);
        // Trip the circuit
        for _ in 0..3 {
            mgr.record_api_failure();
        }
        // Activate fallback
        mgr.transition_to_fallback();
        mgr.fallback_activated.store(true, Ordering::Relaxed);

        // Further failures should not return true (already activated)
        assert!(!mgr.record_api_failure());
    }

    #[test]
    fn test_record_success_resets_on_primary() {
        let mgr = FallbackManager::new(3, 2);
        mgr.record_api_failure();
        mgr.record_api_failure();
        assert_eq!(mgr.consecutive_failures(), 2);

        mgr.record_model_success();
        assert_eq!(mgr.consecutive_failures(), 0);
    }

    #[test]
    fn test_for_model() {
        let mgr = FallbackManager::for_model("llm-70b");
        assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
        assert_eq!(mgr.state(), FallbackState::Primary);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(FallbackManager::new(3, 2));
        let mut handles = Vec::new();

        for _ in 0..10 {
            let mgr = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                mgr.record_api_failure();
                mgr.record_model_success();
                mgr.state();
                mgr.consecutive_failures();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_consolidated_mutex_fields_are_consistent() {
        let mgr = FallbackManager::for_model("primary-model");
        mgr.add_fallback_model("fallback-model");

        // Before transition: using primary model, no switch time.
        assert_eq!(mgr.active_model(), Some("primary-model".to_string()));
        assert!(mgr.fallback_switched_at().is_none());

        // Transition to fallback — updates multiple fields.
        mgr.transition_to_fallback();

        // After transition: both fields should be set together.
        // This verifies the consolidated Mutex prevents partial reads.
        assert_eq!(mgr.active_model(), Some("fallback-model".to_string()));
        assert!(
            mgr.fallback_switched_at().is_some(),
            "switch time should be set after transition"
        );
    }

    #[test]
    fn test_consolidated_mutex_clears_fields_together() {
        let mgr = FallbackManager::for_model("primary-model");
        mgr.add_fallback_model("fallback-model");
        mgr.transition_to_fallback();

        // Both fields are set.
        assert_eq!(mgr.active_model(), Some("fallback-model".to_string()));
        assert!(mgr.fallback_switched_at().is_some());

        // Transition back to primary.
        mgr.transition_to_primary();

        // Both fields should be cleared together.
        assert!(
            mgr.fallback_switched_at().is_none(),
            "switch time should be cleared after transition to primary"
        );
    }

    #[test]
    fn test_consolidated_mutex_reset_clears_all() {
        let mgr = FallbackManager::for_model("primary-model");
        mgr.add_fallback_model("fallback-model");
        // Record a failure while in Primary to dirty the counter, then trip.
        mgr.record_failure();
        mgr.transition_to_fallback();

        // State is dirty.
        assert!(mgr.consecutive_failures() > 0);
        assert!(mgr.fallback_switched_at().is_some());

        // Full reset.
        mgr.reset();

        // Everything cleared.
        assert_eq!(mgr.consecutive_failures(), 0);
        assert!(mgr.fallback_switched_at().is_none());
    }

    #[test]
    fn with_max_fail_count_no_padding_when_not_failed() {
        let entry = FallbackEntry::new("model-a").with_max_fail_count(5);
        assert!(!entry.failed());
        assert_eq!(entry.attempt_count(), 0);
        assert_eq!(entry.max_fail_count, 5);
    }

    #[test]
    fn with_max_fail_count_pads_already_failed_entry() {
        let mut entry = FallbackEntry::new("model-b");
        entry.record_attempt("timeout");
        entry.record_attempt("timeout");
        assert!(entry.failed());
        assert_eq!(entry.attempt_count(), 2);

        let entry = entry.with_max_fail_count(5);
        assert_eq!(entry.max_fail_count, 5);
        assert_eq!(entry.attempt_count(), 5);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_pads_exactly_to_new_threshold() {
        let mut entry = FallbackEntry::new("model-c");
        entry.record_attempt("err");
        entry.record_attempt("err");
        assert!(entry.failed());

        let entry = entry.with_max_fail_count(3);
        assert_eq!(entry.attempt_count(), 3);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_no_padding_when_lowering() {
        let mut entry = FallbackEntry::new("model-d");
        entry.record_attempt("err");
        entry.record_attempt("err");
        assert!(entry.failed());

        let entry = entry.with_max_fail_count(1);
        assert_eq!(entry.max_fail_count, 1);
        assert_eq!(entry.attempt_count(), 2);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_clamps_to_minimum_one() {
        let entry = FallbackEntry::new("model-e").with_max_fail_count(0);
        assert_eq!(entry.max_fail_count, 1);
    }
}
