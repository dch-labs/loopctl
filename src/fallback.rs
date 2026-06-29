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
        self.max_fail_count = max_fail_count.max(1);
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
    pub trip_threshold: usize,
    /// Minimum time in fallback before probing the primary model again. Defaults to 60 s.
    pub recovery_timeout: Duration,
    /// Consecutive successes during recovering before the circuit fully closes. Defaults to `2`.
    pub recovery_successes_needed: usize,
    /// Per-model failure threshold before a fallback model is skipped. Defaults to `2`.
    pub max_fail_count: usize,
}

/// Produces a [`FallbackConfig`] with sensible production defaults.
///
/// Defaults: `trip_threshold = 3`, `recovery_timeout = 60 s`,
/// `recovery_successes_needed = 2`, `max_fail_count = 2`.
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackConfig;
///
/// let config = FallbackConfig::default();
/// assert_eq!(config.trip_threshold, 3);
/// assert_eq!(config.max_fail_count, 2);
/// ```
impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            trip_threshold: 3,
            recovery_timeout: Duration::from_secs(60),
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
pub struct FallbackManager {
    /// Failures before switching to fallback.
    fallback_threshold: usize,
    /// Successes needed on primary before resuming.
    primary_resume_threshold: usize,
    /// Per-model max failure count for new [`FallbackEntry`] instances.
    default_max_fail_count: usize,
    /// Consecutive API failure counter.
    consecutive_failures: AtomicUsize,
    /// Whether fallback has been activated (sticky flag).
    fallback_activated: AtomicBool,
    /// Circuit breaker state (0=Primary, 1=Fallback, 2=Recovering).
    fallback_state: AtomicU8,
    /// Consecutive successes on primary during recovery.
    primary_success_count: AtomicUsize,
    /// Original model name (before fallback).
    original_model: Mutex<Option<String>>,
    /// Ordered fallback models with failure status.
    fallback_models: Mutex<Vec<FallbackEntry>>,
    /// Cached first non-failed fallback model name.
    active_fallback: Mutex<Option<String>>,
    /// Time when fallback was activated.
    fallback_switched_at: Mutex<Option<Instant>>,
    /// How long to remain in fallback before attempting primary recovery.
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
            original_model: Mutex::new(None),
            fallback_models: Mutex::new(Vec::new()),
            active_fallback: Mutex::new(None),
            fallback_switched_at: Mutex::new(None),
            recovery_timeout: Duration::from_secs(60),
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
        if let Ok(mut m) = mgr.original_model.lock() {
            *m = Some(original_model);
        }
        mgr.fallback_activated.store(true, Ordering::Relaxed);
        mgr.consecutive_failures.store(0, Ordering::Relaxed);
        mgr.fallback_state
            .store(FallbackState::Fallback as u8, Ordering::Relaxed);
        if let Ok(mut t) = mgr.fallback_switched_at.lock() {
            *t = Some(Instant::now());
        }
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
        if let Ok(mut m) = mgr.original_model.lock() {
            *m = Some(primary_model.into());
        }
        mgr
    }

    // ==================================================
    // Accessors
    // ==================================================

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
        self.original_model.lock().ok().and_then(|m| m.clone())
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
        if let Ok(mut m) = self.original_model.lock() {
            *m = Some(model);
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
        self.fallback_switched_at.lock().ok().and_then(|t| *t)
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
        if let Ok(mut m) = self.fallback_models.lock() {
            m.clear();
            m.push(FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count));
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
        self.active_fallback.lock().ok().and_then(|m| m.clone())
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
        self.fallback_models
            .lock()
            .ok()
            .map(|m| m.iter().map(|e| e.name.clone()).collect())
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
        if let Ok(mut m) = self.fallback_models.lock() {
            m.push(FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count));
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
        if let Ok(mut m) = self.fallback_models.lock() {
            let entry = FallbackEntry::new(model).with_max_fail_count(self.default_max_fail_count);
            if index >= m.len() {
                m.push(entry);
            } else {
                m.insert(index, entry);
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
        let removed = if let Ok(mut m) = self.fallback_models.lock() {
            if let Some(pos) = m.iter().position(|x| x.name == model) {
                m.remove(pos);
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
        if let Ok(mut m) = self.fallback_models.lock() {
            *m = models
                .into_iter()
                .map(|name| FallbackEntry::new(name).with_max_fail_count(max_fc))
                .collect();
        }
        self.recompute_active_fallback();
    }

    // ==================================================
    // Fallback failure tracking
    // ==================================================

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
        let found = if let Ok(mut m) = self.fallback_models.lock() {
            if let Some(entry) = m.iter_mut().find(|e| e.name == model) {
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
        let found = if let Ok(mut m) = self.fallback_models.lock() {
            if let Some(entry) = m.iter_mut().find(|e| e.name == model) {
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
        if let Ok(mut m) = self.fallback_models.lock() {
            for entry in m.iter_mut() {
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
        self.fallback_models
            .lock()
            .ok()
            .map(|m| {
                m.iter()
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
        self.fallback_models
            .lock()
            .ok()
            .map(|m| {
                m.iter()
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
        self.fallback_models
            .lock()
            .ok()
            .and_then(|m| m.iter().find(|e| e.name == name).cloned())
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
        let found = if let Ok(mut m) = self.fallback_models.lock() {
            if let Some(entry) = m.iter_mut().find(|e| e.name == model) {
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

    // ==================================================
    // Recording
    // ==================================================

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
        let failures = self
            .consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if failures >= self.fallback_threshold && !self.fallback_activated.load(Ordering::Relaxed) {
            warn!(
                consecutive_failures = failures,
                threshold = self.fallback_threshold,
                "Fallback threshold reached"
            );
            self.fallback_activated.store(true, Ordering::Relaxed);
            true
        } else {
            false
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

    // ==================================================
    // State transitions
    // ==================================================

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
        if let Ok(mut t) = self.fallback_switched_at.lock() {
            *t = Some(Instant::now());
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
        if let Ok(mut t) = self.fallback_switched_at.lock() {
            *t = None;
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
        if let Ok(mut t) = self.fallback_switched_at.lock() {
            *t = None;
        }
        self.clear_all_fallback_failed();
    }

    // ==================================================
    // Private helpers
    // ==================================================

    /// Recompute the cached [`active_fallback`](Self::active_fallback) from
    /// the current fallback chain.
    ///
    /// [`fallback_models`]: Self::fallback_models
    /// [`active_fallback`]: Self::active_fallback
    fn recompute_active_fallback(&self) {
        let active = self
            .fallback_models
            .lock()
            .ok()
            .and_then(|m| m.iter().find(|e| !e.failed()).map(|e| e.name.clone()));
        if let Ok(mut cached) = self.active_fallback.lock() {
            *cached = active;
        }
    }
}

/// Produces a [`FallbackManager`] with production defaults.
///
/// Equivalent to `FallbackManager::new(3, 2)` — trips after 3
/// consecutive failures and resumes after 2 consecutive successes.
/// No model name is stored; use [`FallbackManager::for_model`] or
/// [`FallbackManager::set_original_model`] to configure one.
///
/// # Example
///
/// ```rust
/// use loopctl::fallback::FallbackManager;
///
/// let mgr = FallbackManager::default();
/// assert_eq!(mgr.consecutive_failures(), 0);
/// assert!(!mgr.is_using_fallback());
/// ```
impl Default for FallbackManager {
    fn default() -> Self {
        Self::new(3, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Verify that a freshly-constructed manager starts in [`FallbackState::Primary`]
    /// with no failures and no fallback activation.
    ///
    /// Asserts initial values for [`state()`](FallbackManager::state),
    /// [`is_using_fallback()`](FallbackManager::is_using_fallback),
    /// [`is_fallback_active()`](FallbackManager::is_fallback_active), and
    /// [`consecutive_failures()`](FallbackManager::consecutive_failures).
    fn test_initial_state() {
        let mgr = FallbackManager::new(3, 2);
        assert_eq!(mgr.state(), FallbackState::Primary);
        assert!(!mgr.is_using_fallback());
        assert!(!mgr.is_fallback_active());
        assert_eq!(mgr.consecutive_failures(), 0);
    }

    #[test]
    /// Verify that [`record_api_failure`](FallbackManager::record_api_failure) returns
    /// `true` only when the failure count first reaches the threshold.
    ///
    /// Calls [`record_api_failure`](FallbackManager::record_api_failure) three times
    /// and asserts that only the third call returns `true`.
    fn test_failure_threshold() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.record_api_failure()); // 1
        assert!(!mgr.record_api_failure()); // 2
        assert!(mgr.record_api_failure()); // 3 — threshold reached
    }

    #[test]
    /// Verify that [`record_model_failure`](FallbackManager::record_model_failure)
    /// transitions to [`FallbackState::Fallback`] when the threshold is reached.
    ///
    /// Also checks that the method returns `true` only on the threshold-crossing call.
    fn test_model_failure_triggers_fallback() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.record_model_failure()); // 1
        assert!(!mgr.record_model_failure()); // 2
        assert!(mgr.record_model_failure()); // 3 — triggers fallback
        assert_eq!(mgr.state(), FallbackState::Fallback);
    }

    #[test]
    /// Verify the full recovery cycle: Primary → Fallback → Recovering → Primary.
    ///
    /// Trips the circuit with three failures, then transitions to recovering
    /// and records two successes to close the circuit back to primary.
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
    /// Verify that a failure during [`FallbackState::Recovering`] reopens the circuit.
    ///
    /// After transitioning to recovering, a single call to
    /// [`record_model_failure`](FallbackManager::record_model_failure) should
    /// move the state back to [`FallbackState::Fallback`].
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
    /// Verify [`should_try_resume_primary`](FallbackManager::should_try_resume_primary)
    /// enforces both the state check and the cooldown duration.
    ///
    /// Asserts `false` when in [`FallbackState::Primary`], `false` when in
    /// fallback but the cooldown hasn't elapsed, and `true` when the cooldown
    /// has passed (using a 0-second timeout).
    fn test_should_try_resume_primary() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.should_try_resume_primary(Duration::from_secs(10)));

        // Trigger fallback
        for _ in 0..3 {
            mgr.record_model_failure();
        }
        // Not enough time
        assert!(!mgr.should_try_resume_primary(Duration::from_secs(3600)));
        // Enough time (0s timeout)
        assert!(mgr.should_try_resume_primary(Duration::from_secs(0)));
    }

    #[test]
    /// Verify that [`new_with_fallback`](FallbackManager::new_with_fallback)
    /// starts in [`FallbackState::Fallback`] with the model name stored.
    ///
    /// Checks [`is_fallback_active()`](FallbackManager::is_fallback_active),
    /// [`is_using_fallback()`](FallbackManager::is_using_fallback), and
    /// [`original_model()`](FallbackManager::original_model).
    fn test_new_with_fallback() {
        let mgr = FallbackManager::new_with_fallback("llm-70b".into(), 3);
        assert!(mgr.is_fallback_active());
        assert!(mgr.is_using_fallback());
        assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    }

    #[test]
    /// Verify that [`reset`](FallbackManager::reset) clears all state back to
    /// [`FallbackState::Primary`], including counters and flags.
    ///
    /// Trips the circuit first, then asserts that [`reset`] restores
    /// every field to its initial value.
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
    /// Verify that [`record_api_failure`](FallbackManager::record_api_failure)
    /// does not re-trip the circuit after it has already been activated.
    ///
    /// The [`fallback_activated`](FallbackManager::is_fallback_active) flag
    /// prevents the sticky `true` return on every subsequent failure.
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
    /// Verify that [`record_model_success`](FallbackManager::record_model_success)
    /// resets the failure counter when in [`FallbackState::Primary`].
    ///
    /// After recording two failures, a single success should zero the counter.
    fn test_record_success_resets_on_primary() {
        let mgr = FallbackManager::new(3, 2);
        mgr.record_api_failure();
        mgr.record_api_failure();
        assert_eq!(mgr.consecutive_failures(), 2);

        mgr.record_model_success();
        assert_eq!(mgr.consecutive_failures(), 0);
    }

    #[test]
    /// Verify [`for_model`](FallbackManager::for_model) stores the model name.
    ///
    /// Asserts that [`FallbackManager::original_model`] returns the provided
    /// model string and that the initial state is [`FallbackState::Primary`].
    fn test_for_model() {
        let mgr = FallbackManager::for_model("llm-70b");
        assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
        assert_eq!(mgr.state(), FallbackState::Primary);
    }

    #[test]
    /// Verify thread safety — concurrent reads and writes via `Arc<FallbackManager>`.
    ///
    /// Spawns 10 threads that each call [`record_api_failure`], [`record_model_success`],
    /// [`state`], and [`consecutive_failures`] concurrently. The test passes if no
    /// thread panics due to data races (guaranteed by atomic operations).
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
}
