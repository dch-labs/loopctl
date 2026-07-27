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
//! - **[`FallbackManager`]** — The state machine itself; thread-safe via a single `Mutex`.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::fallback::{FallbackManager, FallbackConfig, FailureKind};
//!
//! // Create a manager with a trip threshold of 3 failures
//! let mgr = FallbackManager::new(3, 2);
//!
//! // Simulate failures until the circuit trips
//! mgr.record_failure(FailureKind::Transient);
//! mgr.record_failure(FailureKind::Transient);
//! assert!(mgr.record_failure(FailureKind::Transient)); // 3rd failure trips the circuit
//! assert!(mgr.is_using_fallback());
//!
//! // Manually transition to recovering, then record successes to resume primary
//! mgr.transition_to_recovering();
//! mgr.record_success(); // 1st success
//! mgr.record_success(); // 2nd → back to Primary
//! assert!(!mgr.is_using_fallback());
//! ```

use std::sync::Mutex;
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
/// assert_eq!(state, FallbackState::Primary);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackState {
    /// The circuit is closed and the primary model is in use.
    ///
    /// This is the initial state and the steady-state target: requests
    /// route to the primary model, the failure counter increments on
    /// each failure, and a single success resets it. When
    /// [`consecutive_failures`](FallbackManager::consecutive_failures)
    /// reaches [`trip_threshold`](FallbackConfig::trip_threshold), the
    /// manager transitions to [`Fallback`](Self::Fallback).
    Primary,

    /// The circuit is open and a fallback model is in use.
    ///
    /// Entered when the primary model's consecutive failures exceeded
    /// the trip threshold. Requests route to the active fallback model
    /// (the first non-failed entry in the chain). The manager stays here
    /// for at least
    /// [`recovery_timeout`](FallbackConfig::recovery_timeout) before it
    /// is willing to probe the primary again, at which point it
    /// transitions to [`Recovering`](Self::Recovering).
    Fallback,

    /// Half-open: the manager is probing whether the primary has recovered.
    ///
    /// Entered from [`Fallback`](Self::Fallback) after the cooldown
    /// elapses (see
    /// [`should_try_resume_primary`](FallbackManager::should_try_resume_primary)).
    /// Requests route back to the primary while a recovery success
    /// counter accrues; reaching
    /// [`recovery_successes_needed`](FallbackConfig::recovery_successes_needed)
    /// successes closes the circuit back to [`Primary`](Self::Primary).
    /// A single sustained failure ([`FailureKind::RateLimit`])
    /// during this probe re-opens the circuit to [`Fallback`](Self::Fallback).
    Recovering,
}

/// What kind of failure triggered a [`FallbackManager::record_failure`] call.
///
/// The engine routes different failure causes through one recorder, and the
/// cause changes how a failure during [`FallbackState::Recovering`] is
/// treated: a sustained rate-limit means the primary is still degraded, so
/// the circuit re-trips to [`FallbackState::Fallback`]; a transient error
/// leaves the half-open probe in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// A transient API error (timeout, 5xx, network).
    ///
    /// During [`Recovering`](FallbackState::Recovering) the circuit stays
    /// half-open — one transient blip does not by itself prove the primary
    /// is still bad.
    Transient,

    /// A sustained rate-limit escalation.
    ///
    /// During [`Recovering`](FallbackState::Recovering) the circuit
    /// re-trips to [`Fallback`](FallbackState::Fallback): the primary is
    /// still actively refusing load, so probing it further is pointless.
    RateLimit,
}

/// A single model in the fallback chain, with a failure counter.
///
/// Each entry pairs a model name with a consecutive-failure count and a
/// `max_fail_count` threshold. When [`attempt_count`](Self::attempt_count)
/// reaches `max_fail_count`, the entry is considered [`failed`](Self::failed)
/// and is skipped by [`FallbackManager::fallback_model`].
///
/// Internal to the fallback manager — not part of the public API.
#[derive(Debug, Clone)]
pub(crate) struct FallbackEntry {
    /// The model identifier this entry represents.
    ///
    /// Must match the routing identifier the API client expects (e.g.
    /// `"llm-70b"`, `"gpt-4o"`). Used both to look the entry up in the
    /// chain and as the value returned by
    /// [`fallback_model`](FallbackManager::fallback_model) when this
    /// entry is the active fallback. Set at construction and never
    /// mutated.
    name: String,

    /// Whether this model is eligible to serve requests.
    ///
    /// An out-of-band kill switch separate from failure tracking:
    /// `false` forces [`failed`](Self::failed) to return `true`
    /// regardless of the attempt count, so the manager skips this
    /// entry. Use [`set_available`](Self::set_available) to take a
    /// model out of rotation (e.g. an API key was revoked, the model
    /// was decommissioned) without recording spurious failures.
    available: bool,

    /// How many consecutive failures this model has recorded.
    ///
    /// Incremented by [`record_attempt`](Self::record_attempt) and
    /// compared against [`max_fail_count`](Self::max_fail_count) to
    /// decide [`failed`](Self::failed). Reset to zero by
    /// [`clear_attempts`](Self::clear_attempts) when the circuit
    /// recovers, giving a degraded model a fresh start.
    attempt_count: usize,

    /// The failure count at which this model is taken out of rotation.
    ///
    /// Once [`attempt_count`](Self::attempt_count) reaches this value
    /// the entry is [`failed`](Self::failed) and the manager advances
    /// to the next candidate in the chain. Defaults to the manager's
    /// [`max_fail_count`](FallbackConfig::max_fail_count) at
    /// construction; override per-entry with
    /// [`with_max_fail_count`](Self::with_max_fail_count).
    max_fail_count: usize,
}

impl FallbackEntry {
    /// Create a new entry for the given model, not yet failed.
    ///
    /// Uses a default `max_fail_count` of `2`. Use
    /// [`with_max_fail_count`](Self::with_max_fail_count) to customise.
    pub(crate) fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            available: true,
            attempt_count: 0,
            max_fail_count: 2,
        }
    }

    /// Set a custom `max_fail_count` threshold.
    ///
    /// If raising the threshold on an already-failed entry, the attempt
    /// count is padded up so it stays failed under the new threshold.
    #[must_use]
    pub(crate) fn with_max_fail_count(mut self, max_fail_count: usize) -> Self {
        let new_max = max_fail_count.max(1);
        if self.failed() && self.attempt_count < new_max {
            self.attempt_count = new_max;
        }
        self.max_fail_count = new_max;
        self
    }

    /// Whether this model should be skipped.
    ///
    /// `true` when the model is not `available` or when its attempt count
    /// has reached `max_fail_count`.
    pub(crate) fn failed(&self) -> bool {
        !self.available || self.attempt_count >= self.max_fail_count
    }

    /// Set whether this model is available for use.
    ///
    /// When set to `false`, [`failed`](Self::failed) returns `true`
    /// regardless of the attempt count, and the manager skips this model.
    pub(crate) fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    /// Record a new failure attempt.
    pub(crate) fn record_attempt(&mut self) {
        self.attempt_count = self.attempt_count.saturating_add(1);
    }

    /// Clear all recorded attempts, resetting [`failed`](Self::failed) to `false`.
    pub(crate) fn clear_attempts(&mut self) {
        self.attempt_count = 0;
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
    /// Applied to each fallback model in the chain: once a single
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
/// use loopctl::fallback::{FallbackManager, FallbackConfig, FailureKind};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let mgr = Arc::new(FallbackManager::new(5, 3).with_config(FallbackConfig::default()));
///
/// // Simulate failures
/// assert!(!mgr.record_failure(FailureKind::Transient)); // 1
/// assert!(!mgr.record_failure(FailureKind::Transient)); // 2
/// assert!(mgr.record_failure(FailureKind::Transient));  // 3 → circuit trips
///
/// assert!(mgr.is_using_fallback());
///
/// // ... later, after cooldown ...
/// if mgr.should_try_resume_primary(Duration::from_secs(60)) {
///     mgr.transition_to_recovering();
///     mgr.record_success();
///     mgr.record_success(); // → back to Primary
/// }
/// ```
/// Consolidated mutex-protected fallback state.
///
/// The complete mutable circuit-breaker state.
///
/// Every field that can change over the lifetime of a
/// [`FallbackManager`] lives here, behind a single
/// [`Mutex`](std::sync::Mutex). Holding the whole state under one lock
/// makes each public method's read-modify-write atomic with respect to
/// every other method: there is no window in which one caller can
/// observe a half-applied transition or race a counter reset. Plain
/// fields, no atomics — the lock is the only synchronization.
struct BreakerState {
    /// Current circuit-breaker phase.
    ///
    /// `Primary` at construction; transitions through `Fallback` and
    /// `Recovering` as failures and recoveries are recorded.
    state: FallbackState,

    /// Consecutive failures on the current model.
    ///
    /// Incremented by the `record_*_failure` methods while in `Primary`;
    /// reaching [`FallbackManager::fallback_threshold`] trips the
    /// circuit. Reset to `0` on trip and on any success in `Primary`.
    consecutive_failures: usize,

    /// Consecutive successes on the primary accumulated during
    /// `Recovering`.
    ///
    /// Incremented by [`record_success`](FallbackManager::record_success)
    /// while in `Recovering`; reaching
    /// [`primary_resume_threshold`](FallbackManager::primary_resume_threshold)
    /// closes the circuit back to `Primary`. Reset on entry to
    /// `Recovering` and on trip.
    primary_success_count: usize,

    /// Sticky flag recording whether fallback has ever been activated.
    ///
    /// `true` once the circuit has tripped at least once, and never
    /// reset for the life of the manager. Lets observers tell "still on
    /// primary, never failed" from "recovered back to primary".
    fallback_activated: bool,

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
    /// in place whenever the set of failed entries changes.
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

impl Default for BreakerState {
    fn default() -> Self {
        Self {
            state: FallbackState::Primary,
            consecutive_failures: 0,
            primary_success_count: 0,
            fallback_activated: false,
            original_model: None,
            fallback_models: Vec::new(),
            active_fallback: None,
            fallback_switched_at: None,
        }
    }
}

/// Circuit breaker for API model fallback.
///
/// `&FallbackManager` is `Send + Sync` and can be freely shared across
/// threads (e.g. via `Arc<FallbackManager>`). No `&mut self` is needed
/// for any operation.
pub struct FallbackManager {
    /// Immutable configuration: thresholds and timeouts.
    ///
    /// Single source of truth for the trip threshold, recovery
    /// threshold, per-model max-fail count, and recovery timeout —
    /// mirrors the [`DetectionManager::config`](crate::detection::DetectionManager::config)
    /// pattern. Held by value, outside the state lock.
    config: FallbackConfig,

    /// All mutable circuit-breaker state, behind a single lock.
    ///
    /// Holding every changeable field ([`BreakerState`]) under one
    /// [`Mutex`](std::sync::Mutex) makes each public method's
    /// read-modify-write atomic relative to every other method — no
    /// window for a partial-state read or a counter/transition race.
    /// Immutable config ([`config`](Self::config)) stays outside the
    /// lock.
    state: Mutex<BreakerState>,
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::new(5, 3);
    /// // Trip after 5 failures, resume after 3 consecutive successes
    /// ```
    #[must_use]
    pub fn new(fallback_threshold: usize, primary_resume_threshold: usize) -> Self {
        Self {
            config: FallbackConfig {
                trip_threshold: fallback_threshold,
                recovery_successes_needed: primary_resume_threshold,
                ..FallbackConfig::default()
            },
            state: Mutex::new(BreakerState::default()),
        }
    }

    /// Apply configuration from a [`FallbackConfig`] struct.
    ///
    /// Replaces the manager's entire config. Prefer
    /// [`new_with_config`](Self::new_with_config) for config-driven
    /// construction; use this builder when chaining off [`new`](Self::new).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackConfig, FailureKind};
    /// use std::time::Duration;
    ///
    /// let config = FallbackConfig {
    ///     trip_threshold: 5,
    ///     recovery_timeout: Duration::from_secs(120),
    ///     recovery_successes_needed: 3,
    ///     max_fail_count: 2,
    /// };
    /// let mgr = FallbackManager::new(5, 3).with_config(config);
    /// ```
    #[must_use]
    pub fn with_config(mut self, config: FallbackConfig) -> Self {
        self.config = config;
        self
    }

    /// The immutable [`FallbackConfig`] this manager was constructed with.
    ///
    /// Read thresholds and timeouts (trip threshold, recovery timeout,
    /// per-model max-fail count, recovery successes needed) from here
    /// rather than from duplicated accessors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackConfig};
    /// use std::time::Duration;
    ///
    /// let cfg = FallbackConfig { trip_threshold: 5, ..FallbackConfig::default() };
    /// let mgr = FallbackManager::new_with_config(cfg);
    /// assert_eq!(mgr.config().trip_threshold, 5);
    /// ```
    #[must_use]
    pub fn config(&self) -> &FallbackConfig {
        &self.config
    }

    /// Create a new manager directly from a [`FallbackConfig`].
    ///
    /// The config-driven counterpart of [`new`](Self::new): build the
    /// whole config struct (typically from a file or env) and construct
    /// in one step instead of overriding fields after the fact.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackConfig};
    ///
    /// let cfg = FallbackConfig::default();
    /// let mgr = FallbackManager::new_with_config(cfg);
    /// ```
    #[must_use]
    pub fn new_with_config(config: FallbackConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BreakerState::default()),
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::new_with_fallback("llm-70b".into(), 3);
    /// assert!(mgr.is_using_fallback());
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    #[must_use]
    pub fn new_with_fallback(original_model: String, fallback_threshold: usize) -> Self {
        let mgr = Self::new(fallback_threshold, 2);
        if let Ok(mut state) = mgr.state.lock() {
            state.original_model = Some(original_model);
            Self::transition_to_fallback_impl(&mut state);
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// assert!(!mgr.is_using_fallback());
    /// ```
    pub fn for_model(primary_model: impl Into<String>) -> Self {
        let mgr = Self::new(3, 2);
        if let Ok(mut state) = mgr.state.lock() {
            state.original_model = Some(primary_model.into());
        }
        mgr
    }

    /// Get the current circuit breaker state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// ```
    pub fn state(&self) -> FallbackState {
        self.state
            .lock()
            .map_or(FallbackState::Primary, |s| s.state)
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
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.is_fallback_active());
    /// ```
    pub fn is_fallback_active(&self) -> bool {
        self.state.lock().is_ok_and(|s| s.fallback_activated)
    }

    /// Get the number of consecutive failures.
    ///
    /// The count since the last success (when in
    /// [`FallbackState::Primary`]) or since the circuit tripped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// mgr.record_failure(FailureKind::Transient);
    /// assert_eq!(mgr.consecutive_failures(), 1);
    /// ```
    pub fn consecutive_failures(&self) -> usize {
        self.state.lock().map_or(0, |s| s.consecutive_failures)
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn original_model(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|s| s.original_model.clone())
    }

    /// Set the original model name.
    ///
    /// Overwrites the stored primary model name. Useful when the model
    /// is resolved from configuration or changed mid-session.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert_eq!(mgr.original_model(), None);
    /// mgr.set_original_model("llm-70b".into());
    /// assert_eq!(mgr.original_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn set_original_model(&self, model: String) {
        if let Ok(mut state) = self.state.lock() {
            state.original_model = Some(model);
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// use std::time::Duration;
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(mgr.fallback_switched_at().is_none());
    /// ```
    pub fn fallback_switched_at(&self) -> Option<Instant> {
        self.state.lock().ok().and_then(|s| s.fallback_switched_at)
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::for_model("llm-70b");
    /// assert_eq!(mgr.active_model(), Some("llm-70b".to_string()));
    /// ```
    pub fn active_model(&self) -> Option<String> {
        let Ok(state) = self.state.lock() else {
            return None;
        };
        match state.state {
            FallbackState::Primary | FallbackState::Recovering => state.original_model.clone(),
            FallbackState::Fallback => state
                .active_fallback
                .clone()
                .or_else(|| state.original_model.clone()),
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
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
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
    /// mgr.record_failure(FailureKind::Transient);
    /// mgr.record_failure(FailureKind::Transient);
    /// mgr.record_failure(FailureKind::Transient); // trips to Fallback
    ///
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// assert_eq!(mgr.active_model(), Some("llm-4".to_string()));
    /// ```
    pub fn set_fallback_model(&self, model: impl Into<String>) {
        if let Ok(mut state) = self.state.lock() {
            state.fallback_models.clear();
            state
                .fallback_models
                .push(FallbackEntry::new(model).with_max_fail_count(self.config.max_fail_count));
            Self::recompute_active_fallback_impl(&mut state);
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(mgr.fallback_model().is_none());
    ///
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    /// assert_eq!(mgr.fallback_model(), Some("llm-70b".to_string())); // first in chain
    /// ```
    pub fn fallback_model(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|s| s.active_fallback.clone())
    }

    /// Get the full fallback model chain.
    ///
    /// Returns all configured fallback models in priority order (index `0`
    /// is the first fallback tried).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        self.state
            .lock()
            .ok()
            .map(|s| s.fallback_models.iter().map(|e| e.name.clone()).collect())
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.add_fallback_model("llm-70b");
    /// mgr.add_fallback_model("llm-120b");
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b"]);
    /// ```
    pub fn add_fallback_model(&self, model: impl Into<String>) {
        if let Ok(mut state) = self.state.lock() {
            state
                .fallback_models
                .push(FallbackEntry::new(model).with_max_fail_count(self.config.max_fail_count));
            Self::recompute_active_fallback_impl(&mut state);
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        if let Ok(mut state) = self.state.lock() {
            let entry = FallbackEntry::new(model).with_max_fail_count(self.config.max_fail_count);
            if index >= state.fallback_models.len() {
                state.fallback_models.push(entry);
            } else {
                state.fallback_models.insert(index, entry);
            }
            Self::recompute_active_fallback_impl(&mut state);
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        let mut removed = false;
        if let Ok(mut state) = self.state.lock() {
            if let Some(pos) = state.fallback_models.iter().position(|x| x.name == model) {
                state.fallback_models.remove(pos);
                removed = true;
            }
            if removed {
                Self::recompute_active_fallback_impl(&mut state);
            }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.set_fallback_models(vec!["llm-70b".into(), "llm-120b".into(), "llm-32b".into()]);
    ///
    /// let chain = mgr.fallback_models();
    /// assert_eq!(chain, vec!["llm-70b", "llm-120b", "llm-32b"]);
    /// ```
    pub fn set_fallback_models(&self, models: Vec<String>) {
        let max_fc = self.config.max_fail_count;
        if let Ok(mut state) = self.state.lock() {
            state.fallback_models = models
                .into_iter()
                .map(|name| FallbackEntry::new(name).with_max_fail_count(max_fc))
                .collect();
            Self::recompute_active_fallback_impl(&mut state);
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        let mut found = false;
        if let Ok(mut state) = self.state.lock() {
            if let Some(entry) = state.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.record_attempt();
                found = true;
            }
            if found {
                Self::recompute_active_fallback_impl(&mut state);
            }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        let mut found = false;
        if let Ok(mut state) = self.state.lock() {
            if let Some(entry) = state.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.clear_attempts();
                found = true;
            }
            if found {
                Self::recompute_active_fallback_impl(&mut state);
            }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        if let Ok(mut state) = self.state.lock() {
            for entry in &mut state.fallback_models {
                entry.clear_attempts();
            }
            Self::recompute_active_fallback_impl(&mut state);
        }
    }

    /// Get the names of all fallback models marked as failed.
    ///
    /// Returns model names in chain order, filtered to only those with
    /// the failed flag set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        self.state
            .lock()
            .ok()
            .map(|s| {
                s.fallback_models
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        self.state
            .lock()
            .ok()
            .map(|s| {
                s.fallback_models
                    .iter()
                    .filter(|e| !e.failed())
                    .map(|e| e.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Set the `available` flag on a fallback model.
    ///
    /// When set to `false`, the model is considered failed regardless of
    /// its attempt count, and [`active_model`](Self::active_model) will
    /// skip it. When set back to `true`, the model becomes eligible again
    /// (unless its attempt count has also reached the per-model
    /// `max_fail_count`).
    ///
    /// Returns `true` if the model was found in the chain and updated,
    /// `false` if no model with that name exists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
        let mut found = false;
        if let Ok(mut state) = self.state.lock() {
            if let Some(entry) = state.fallback_models.iter_mut().find(|e| e.name == model) {
                entry.set_available(available);
                found = true;
            }
            if found {
                Self::recompute_active_fallback_impl(&mut state);
            }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.record_failure(FailureKind::Transient)); // 1
    /// assert!(!mgr.record_failure(FailureKind::Transient)); // 2
    /// assert!(mgr.record_failure(FailureKind::Transient));  // 3 — threshold reached, now activated
    /// assert!(!mgr.record_failure(FailureKind::Transient)); // 4 — already activated, no re-trip
    /// ```
    /// Record a failure on the current model and trip the circuit if the
    /// threshold is reached.
    ///
    /// Called by the agent loop each time an LLM request fails. The
    /// effect depends on the current circuit state and the failure
    /// [`kind`](FailureKind):
    ///
    /// - **[`Primary`](FallbackState::Primary)**: Increments the failure
    ///   counter. If it reaches
    ///   [`fallback_threshold`](Self::new), transitions to
    ///   [`Fallback`](FallbackState::Fallback) and returns `true`.
    /// - **[`Fallback`](FallbackState::Fallback)**: Logs a warning — the
    ///   fallback model itself is failing. Consider calling
    ///   [`mark_fallback_failed`](Self::mark_fallback_failed) to skip it.
    ///   Returns `false`; the counter is unchanged.
    /// - **[`Recovering`](FallbackState::Recovering)**: A
    ///   [`FailureKind::RateLimit`] re-trips the circuit to
    ///   [`Fallback`](FallbackState::Fallback) — the primary is still
    ///   rate-limited, so probing it further is pointless. A
    ///   [`FailureKind::Transient`] error leaves the half-open probe in
    ///   place. Returns `false` either way.
    ///
    /// # Returns
    ///
    /// `true` only when this call tripped the circuit from `Primary` to
    /// `Fallback`; `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// assert!(!mgr.record_failure(FailureKind::Transient)); // 1
    /// assert!(!mgr.record_failure(FailureKind::Transient)); // 2
    /// assert!(mgr.record_failure(FailureKind::Transient));  // 3 → trips to Fallback
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// ```
    pub fn record_failure(&self, kind: FailureKind) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        match state.state {
            FallbackState::Primary => {
                state.consecutive_failures = state.consecutive_failures.saturating_add(1);
                if state.consecutive_failures >= self.config.trip_threshold {
                    warn!(
                        consecutive_failures = state.consecutive_failures,
                        threshold = self.config.trip_threshold,
                        "Fallback threshold reached"
                    );
                    Self::transition_to_fallback_impl(&mut state);
                    true
                } else {
                    false
                }
            }
            FallbackState::Fallback => {
                let fb_name = state
                    .active_fallback
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                warn!(
                    "Fallback model \"{fb_name}\" also experiencing failures; consider calling mark_fallback_failed(\"{fb_name}\") to skip it"
                );
                false
            }
            FallbackState::Recovering => match kind {
                FailureKind::RateLimit => {
                    warn!(
                        "Primary model rate-limited during recovery test, re-tripping to fallback"
                    );
                    Self::transition_to_fallback_impl(&mut state);
                    false
                }
                FailureKind::Transient => {
                    warn!(
                        consecutive_failures = state.consecutive_failures,
                        "Transient failure recorded during recovery; staying half-open"
                    );
                    false
                }
            },
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.record_failure(FailureKind::Transient);
    /// mgr.record_failure(FailureKind::Transient);
    /// assert_eq!(mgr.consecutive_failures(), 2);
    /// mgr.reset_failure_counter();
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn reset_failure_counter(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.consecutive_failures = 0;
        }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.record_failure(FailureKind::Transient);
    /// assert_eq!(mgr.consecutive_failures(), 1);
    /// mgr.record_success(); // resets failures to 0
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn record_success(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match state.state {
            FallbackState::Primary => {
                state.consecutive_failures = 0;
            }
            FallbackState::Fallback => {}
            FallbackState::Recovering => {
                state.primary_success_count = state.primary_success_count.saturating_add(1);
                debug!(
                    successes = state.primary_success_count,
                    threshold = self.config.recovery_successes_needed,
                    "Primary model success during recovery test"
                );
                if state.primary_success_count >= self.config.recovery_successes_needed {
                    Self::transition_to_primary_impl(&mut state);
                }
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
    /// use loopctl::fallback::{FallbackManager, FailureKind};
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
    /// [`record_failure`](Self::record_failure) when the
    /// failure threshold is reached, or manually when the circuit needs
    /// to trip immediately.
    ///
    /// After this call, [`is_using_fallback`](Self::is_using_fallback)
    /// returns `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    /// ```
    pub fn transition_to_fallback(&self) {
        if let Ok(mut state) = self.state.lock() {
            Self::transition_to_fallback_impl(&mut state);
        }
    }

    /// [`transition_to_fallback`](Self::transition_to_fallback) assuming
    /// the caller already holds the state lock.
    ///
    /// Called from the `record_*` methods so the trip runs while their
    /// guard is held, making the read-decide-transition atomic with
    /// respect to other callers.
    fn transition_to_fallback_impl(state: &mut BreakerState) {
        state.state = FallbackState::Fallback;
        state.fallback_activated = true;
        state.fallback_switched_at = Some(Instant::now());
        state.primary_success_count = 0;
        info!("Circuit breaker: transitioned to Fallback state");
    }

    /// Transition to testing primary model (half-open state).
    ///
    /// Moves the circuit breaker from [`FallbackState::Fallback`] to
    /// [`FallbackState::Recovering`] and resets the recovery success
    /// counter. Called by the agent loop after
    /// [`should_try_resume_primary`](Self::should_try_resume_primary)
    /// returns `true`. Subsequent calls to
    /// [`record_success`](Self::record_success) will count
    /// toward the recovery threshold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// mgr.transition_to_recovering();
    /// assert_eq!(mgr.state(), FallbackState::Recovering);
    /// ```
    pub fn transition_to_recovering(&self) {
        if let Ok(mut state) = self.state.lock() {
            Self::transition_to_recovering_impl(&mut state);
        }
    }

    /// [`transition_to_recovering`](Self::transition_to_recovering)
    /// assuming the caller already holds the state lock.
    fn transition_to_recovering_impl(state: &mut BreakerState) {
        state.state = FallbackState::Recovering;
        state.primary_success_count = 0;
        info!("Circuit breaker: transitioned to Recovering state (testing primary)");
    }

    /// Transition back to primary model (circuit closed).
    ///
    /// Moves the circuit breaker to [`FallbackState::Primary`], clears
    /// the fallback timestamp, and resets all counters (failures,
    /// successes, and the `fallback_activated` flag). Called
    /// automatically when the recovery success threshold is reached
    /// inside [`record_success`](Self::record_success), or
    /// manually to force an immediate return to primary.
    ///
    /// After this call, [`is_using_fallback`](Self::is_using_fallback)
    /// returns `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// mgr.transition_to_fallback();
    /// mgr.transition_to_primary();
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// assert!(!mgr.is_using_fallback());
    /// ```
    pub fn transition_to_primary(&self) {
        if let Ok(mut state) = self.state.lock() {
            Self::transition_to_primary_impl(&mut state);
        }
    }

    /// [`transition_to_primary`](Self::transition_to_primary) assuming
    /// the caller already holds the state lock.
    ///
    /// Inlines the [`clear_all_fallback_failed`](Self::clear_all_fallback_failed)
    /// and active-fallback recompute against the held guard, because the
    /// public versions acquire the same lock themselves and
    /// `std::sync::Mutex` is not reentrant.
    fn transition_to_primary_impl(state: &mut BreakerState) {
        state.state = FallbackState::Primary;
        state.fallback_switched_at = None;
        state.primary_success_count = 0;
        state.consecutive_failures = 0;
        state.fallback_activated = false;
        for entry in &mut state.fallback_models {
            entry.clear_attempts();
        }
        Self::recompute_active_fallback_impl(state);
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
    /// use loopctl::fallback::{FallbackManager, FallbackState, FailureKind};
    ///
    /// let mgr = FallbackManager::new(3, 2);
    /// for _ in 0..3 { mgr.record_failure(FailureKind::Transient); }
    /// assert_eq!(mgr.state(), FallbackState::Fallback);
    ///
    /// mgr.reset();
    /// assert_eq!(mgr.state(), FallbackState::Primary);
    /// assert!(!mgr.is_fallback_active());
    /// assert_eq!(mgr.consecutive_failures(), 0);
    /// ```
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            Self::transition_to_primary_impl(&mut state);
        }
    }

    /// Recompute the cached [`active_fallback`](BreakerState::active_fallback)
    /// from the current fallback chain, in place against a held guard.
    ///
    /// Walks [`fallback_models`](BreakerState::fallback_models) for the
    /// first non-failed entry and stores its name, so
    /// [`fallback_model`](Self::fallback_model) stays `O(1)`. Called by
    /// every method that mutates the chain or a model's failed flag, and
    /// by [`transition_to_primary_impl`](Self::transition_to_primary_impl).
    fn recompute_active_fallback_impl(state: &mut BreakerState) {
        state.active_fallback = state
            .fallback_models
            .iter()
            .find(|e| !e.failed())
            .map(|e| e.name.clone());
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
        assert!(!mgr.record_failure(FailureKind::Transient)); // 1
        assert!(!mgr.record_failure(FailureKind::Transient)); // 2
        assert!(mgr.record_failure(FailureKind::Transient)); // 3 — threshold reached
    }

    #[test]
    fn test_model_failure_triggers_fallback() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.record_failure(FailureKind::Transient)); // 1
        assert!(!mgr.record_failure(FailureKind::Transient)); // 2
        assert!(mgr.record_failure(FailureKind::Transient)); // 3 — triggers fallback
        assert_eq!(mgr.state(), FallbackState::Fallback);
    }

    #[test]
    fn test_recovery() {
        let mgr = FallbackManager::new(3, 2);
        // Trigger fallback
        for _ in 0..3 {
            mgr.record_failure(FailureKind::Transient);
        }
        assert_eq!(mgr.state(), FallbackState::Fallback);

        // Transition to recovering
        mgr.transition_to_recovering();
        assert_eq!(mgr.state(), FallbackState::Recovering);

        // Recover after enough successes
        mgr.record_success(); // 1
        mgr.record_success(); // 2 — threshold reached
        assert_eq!(mgr.state(), FallbackState::Primary);
    }

    #[test]
    fn test_recovery_failure_goes_back_to_fallback() {
        let mgr = FallbackManager::new(3, 2);
        for _ in 0..3 {
            mgr.record_failure(FailureKind::Transient);
        }
        mgr.transition_to_recovering();
        mgr.record_failure(FailureKind::RateLimit); // sustained failure during recovery re-trips
        assert_eq!(mgr.state(), FallbackState::Fallback);
    }

    #[test]
    fn test_should_try_resume_primary() {
        let mgr = FallbackManager::new(3, 2);
        assert!(!mgr.should_try_resume_primary(Duration::from_secs(10)));

        // Trigger fallback
        for _ in 0..3 {
            mgr.record_failure(FailureKind::Transient);
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
            mgr.record_failure(FailureKind::Transient);
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
            mgr.record_failure(FailureKind::Transient);
        }
        // Activate fallback
        mgr.transition_to_fallback();

        // Further failures should not return true (already activated)
        assert!(!mgr.record_failure(FailureKind::Transient));
    }

    #[test]
    fn test_record_success_resets_on_primary() {
        let mgr = FallbackManager::new(3, 2);
        mgr.record_failure(FailureKind::Transient);
        mgr.record_failure(FailureKind::Transient);
        assert_eq!(mgr.consecutive_failures(), 2);

        mgr.record_success();
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
                mgr.record_failure(FailureKind::Transient);
                mgr.record_success();
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
        mgr.record_failure(FailureKind::Transient);
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
        assert_eq!(entry.attempt_count, 0);
        assert_eq!(entry.max_fail_count, 5);
    }

    #[test]
    fn with_max_fail_count_pads_already_failed_entry() {
        let mut entry = FallbackEntry::new("model-b");
        entry.record_attempt();
        entry.record_attempt();
        assert!(entry.failed());
        assert_eq!(entry.attempt_count, 2);

        let entry = entry.with_max_fail_count(5);
        assert_eq!(entry.max_fail_count, 5);
        assert_eq!(entry.attempt_count, 5);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_pads_exactly_to_new_threshold() {
        let mut entry = FallbackEntry::new("model-c");
        entry.record_attempt();
        entry.record_attempt();
        assert!(entry.failed());

        let entry = entry.with_max_fail_count(3);
        assert_eq!(entry.attempt_count, 3);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_no_padding_when_lowering() {
        let mut entry = FallbackEntry::new("model-d");
        entry.record_attempt();
        entry.record_attempt();
        assert!(entry.failed());

        let entry = entry.with_max_fail_count(1);
        assert_eq!(entry.max_fail_count, 1);
        assert_eq!(entry.attempt_count, 2);
        assert!(entry.failed());
    }

    #[test]
    fn with_max_fail_count_clamps_to_minimum_one() {
        let entry = FallbackEntry::new("model-e").with_max_fail_count(0);
        assert_eq!(entry.max_fail_count, 1);
    }
}
