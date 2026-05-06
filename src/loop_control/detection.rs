//! Detection manager — loop and convergence detection for agent behavior.
//!
//! This module provides the [`DetectionManager`] which unifies two complementary
//! detection strategies into a single manager that agents consult after each turn
//! to decide whether they are making progress or spinning in circles:
//!
//! - **Loop detection** (delegated to [`LoopDetector`]) — identifies repeated
//!   sequences of tool calls using tool-specific JSON parsing, result-aware
//!   comparison, Edit-recovery logic, warning deduplication, and configurable
//!   thresholds.
//! - **Convergence detection** (delegated to [`ConvergenceDetector`]) — detects
//!   when the agent's free-text responses have become semantically similar using
//!   Jaccard similarity on word-level tokens, with a configurable window size,
//!   similarity threshold, and convergence action.
//!
//! Together these detectors power the agent framework's "stuck detection" layer.
//! When either detector fires, the framework can inject a warning, force a
//! strategy change, or terminate the session outright.
//!
//! # Architecture
//!
//! ```text
//!                          DetectionManager
//!                    ┌───────────────────────────┐
//!                    │  config: DetectionConfig  │
//!                    │  stats: DetectionStats    │
//!                    └─────┬────────────┬────────┘
//!                          │            │
//!            ┌─────────────┘            └───────────┐
//!            ▼                                      ▼
//!    ┌──────────────┐                       ┌──────────────────┐
//!    │ LoopDetector │                       │ ConvergenceDet.  │
//!    │  (tool call  │                       │  (response text  │
//!    │   cycles)    │                       │   similarity)    │
//!    └──────┬───────┘                       └───────┬──────────┘
//!           │                                       │
//!   record_tool_call()                       record_response()
//!   record_operation()                      check_convergence()
//!      check_loop()                                 │
//!           │                                       │
//!           └──────────────────┬────────────────────┘
//!                              ▼
//!                       DetectedPattern
//!               ┌──────────────┼──────────────────┐
//!               ▼              ▼                  ▼
//!            NoPattern   LoopDetected   ConvergenceDetected
//! ```
//!
//! # Provided Types
//!
//! - [`DetectionManager`] — facade that owns and delegates to both detectors.
//! - [`DetectionConfig`] — unified configuration for loop + convergence tuning.
//! - [`DetectedPattern`] — summary enum returned by every check method.
//! - [`DetectionStats`] — cumulative statistics exposed for observability.
//!
//! # Re-exports
//!
//! This module re-exports key types from the underlying detector modules so
//! consumers can import everything from a single path:
//!
//! - [`ConvergenceAction`] — what to do when convergence is detected.
//! - [`LoopStatus`] — detailed loop status from [`DetectionManager::check_loop`].
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::loop_control::detection::{
//!     DetectionConfig, DetectionManager, DetectedPattern,
//! };
//!
//! // Create with defaults (loop threshold 3, convergence threshold 0.95)
//! let dm = DetectionManager::new();
//!
//! // Record tool calls each turn
//! dm.record_tool_call("Read", 42);
//! let pattern = dm.record_tool_call("Read", 42);
//! assert!(matches!(pattern, DetectedPattern::NoPattern));
//!
//! // Record assistant responses for convergence checking
//! dm.record_response("I'm working on the file.");
//!
//! // Inspect current state at any time
//! if let DetectedPattern::LoopDetected { repetitions, .. } = dm.check_current_pattern() {
//!     println!("Agent is looping — {repetitions} repetitions seen");
//! }
//!
//! // Reset between tasks
//! dm.reset();
//! ```
//!
//! # See Also
//!
//! - [`ConvergenceDetector`] — the standalone convergence detector module.
//! - [`LoopDetector`] — the standalone loop detector module.

use std::sync::{Arc, Mutex};

use super::convergence::{ConvergenceConfig, ConvergenceDetector, ConvergenceStatus};
use super::loop_detector::{LoopDetector, LoopDetectorConfig, Operation, ToolSignature};

pub use super::convergence::ConvergenceAction;
pub use super::convergence::ConvergenceConfigError;
pub use super::loop_detector::LoopStatus;

// ==================================================
// Detected Pattern
// ==================================================

/// Represents a pattern detected in agent behavior.
///
/// Returned by [`DetectionManager::record_operation`],
/// [`DetectionManager::record_response`], and
/// [`DetectionManager::check_current_pattern`] to summarise what — if
/// anything — the detector found. Callers typically match on this enum to
/// decide whether to inject a warning, switch strategies, or halt the
/// session.
///
/// # Variants
///
/// - [`LoopDetected`](Self::LoopDetected) — repeated tool-call pattern.
/// - [`ConvergenceDetected`](Self::ConvergenceDetected) — semantically similar responses.
/// - [`NoPattern`](Self::NoPattern) — agent is making progress.
///
/// # Matching Strategy
///
/// The recommended approach is to handle each variant explicitly:
///
/// ```rust,ignore
/// match pattern {
///     DetectedPattern::LoopDetected { repetitions, pattern_description } => {
///         tracing::warn!("Loop detected: {pattern_description} (×{repetitions})");
///     }
///     DetectedPattern::ConvergenceDetected { similarity, consecutive_count } => {
///         tracing::info!("Responses converged (similarity={similarity:.2})");
///     }
///     DetectedPattern::NoPattern => { /* agent is making progress */ }
/// }
/// ```
///
/// # See Also
///
/// - [`DetectionManager::check_current_pattern`] — unified check.
/// - [`DetectionStats`] — cumulative detection counters.
#[derive(Debug, Clone)]
pub enum DetectedPattern {
    /// The agent is repeating the same sequence of tool calls.
    ///
    /// Emitted by [`DetectionManager::record_operation`] (and the
    /// convenience wrappers [`DetectionManager::record_tool_call`] /
    /// [`DetectionManager::record_tool_call_with_result`]) when the
    /// internal [`LoopDetector`] observes the same operation at least
    /// [`DetectionConfig::loop_threshold`] times in a row.
    ///
    /// A "loop" does not mean the agent is calling the *exact same*
    /// tool every time — it means the operation signature (tool name +
    /// primary parameter + optional result hash) has been seen repeatedly.
    /// The loop detector uses result-aware comparison so that calling
    /// the same file with different results is *not* flagged.
    ///
    /// # Fields
    ///
    /// - `repetitions` — how many times the pattern has repeated so far.
    /// - `pattern_description` — a human-readable summary like `"Read(/etc/hosts)"`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After calling Read("/etc/hosts") 3 times with identical results:
    /// if let DetectedPattern::LoopDetected { repetitions, pattern_description } = pattern {
    ///     assert_eq!(pattern_description, "Read(/etc/hosts)");
    ///     assert!(repetitions >= 3);
    /// }
    /// ```
    LoopDetected {
        /// Number of times the pattern has repeated.
        ///
        /// Equal to [`LoopStatus::repetition_count`] at the time of
        /// detection. When this value reaches
        /// [`DetectionConfig::stop_threshold`] the agent should terminate.
        ///
        /// This is a `usize` ≥ 1. A value of 1 means the loop was just
        /// detected (the repetition count equals the loop threshold).
        repetitions: usize,
        /// Human-readable description of the repeating pattern.
        ///
        /// Typically formatted as `"ToolName(primary_param)"` extracted
        /// from the first entry in [`LoopStatus::repeated_operations`].
        /// For example, if the agent keeps calling `Read("/etc/hosts")`,
        /// this field would contain `"Read(/etc/hosts)"`.
        ///
        /// Can be empty if the loop detector did not provide any
        /// repeated operations (which should not happen in practice).
        pattern_description: String,
    },
    /// The agent's responses have become semantically similar.
    ///
    /// Emitted by [`DetectionManager::record_response`] when the internal
    /// [`ConvergenceDetector`] finds that the last *N* assistant messages
    /// exceed [`DetectionConfig::convergence_threshold`] in Jaccard
    /// similarity, where *N* equals [`DetectionConfig::convergence_count`].
    ///
    /// Unlike [`LoopDetected`](Self::LoopDetected), which monitors tool
    /// call patterns, this variant tracks the *content* of the agent's
    /// free-text replies. It fires when the agent keeps saying essentially
    /// the same thing in different words — a strong signal that it is
    /// stuck even if it is calling different tools each time.
    ///
    /// # Fields
    ///
    /// - `similarity` — the Jaccard similarity score of the most recent
    ///   pair of responses (0.0–1.0).
    /// - `consecutive_count` — how many consecutive response pairs exceeded
    ///   the threshold.
    ConvergenceDetected {
        /// Jaccard similarity score (0.0–1.0) of the most recent pair of
        /// responses.
        ///
        /// A value of `1.0` means the responses are identical at the
        /// word-token level. The default threshold is `0.95`.
        ///
        /// Note: this score is computed on word-level token sets after
        /// lowercasing, not on raw character sequences. Two responses
        /// that are paraphrases with different vocabulary may have a low
        /// score even if they are semantically equivalent.
        similarity: f32,
        /// Number of consecutive responses that exceeded the similarity
        /// threshold.
        ///
        /// Equal to [`ConvergenceStatus::consecutive_count`]. When this
        /// value reaches [`DetectionConfig::convergence_count`], the
        /// convergence detector fires.
        ///
        /// A value of `1` means this is the first time the threshold was
        /// exceeded; higher values indicate a sustained period of
        /// similarity.
        consecutive_count: usize,
    },
    /// No pattern detected — the agent appears to be making progress.
    ///
    /// This is the "healthy" variant. It is returned whenever neither the
    /// loop detector nor the convergence detector has fired. The agent's
    /// tool calls are varying and/or its responses are diverging, which
    /// indicates forward progress.
    ///
    /// Callers should simply continue the turn loop when they receive
    /// this variant.
    NoPattern,
}

// ==================================================
// Configuration
// ==================================================

/// Configuration for the [`DetectionManager`].
///
/// Groups all tunables for loop detection and convergence detection in a
/// single struct so consumers can construct a [`DetectionManager`] with
/// one call to [`DetectionManager::with_config`].
///
/// The [`Default`] implementation provides sensible production values:
/// loop threshold 3, stop threshold 10, convergence threshold 0.95, and
/// convergence count 3.
///
/// # Sections
///
/// The configuration is divided into two groups:
///
/// - **Loop detection** — [`loop_threshold`](Self::loop_threshold),
///   [`stop_threshold`](Self::stop_threshold),
///   [`enable_loop_detection`](Self::enable_loop_detection),
///   [`max_history`](Self::max_history).
/// - **Convergence detection** —
///   [`convergence_threshold`](Self::convergence_threshold),
///   [`convergence_count`](Self::convergence_count),
///   [`enable_convergence_detection`](Self::enable_convergence_detection),
///   [`on_converge`](Self::on_converge),
///   [`max_response_history`](Self::max_response_history).
///
/// # Example
///
/// ```rust,ignore
/// let config = DetectionConfig {
///     loop_threshold: 5,          // allow more repetition before flagging
///     convergence_threshold: 0.9, // looser similarity threshold
///     ..DetectionConfig::default()
/// };
/// let dm = DetectionManager::with_config(config);
/// ```
///
/// # See Also
///
/// - [`DetectionConfig::to_convergence_config`] — converts to [`ConvergenceConfig`].
/// - [`DetectionConfig::to_loop_detector_config`] — converts to [`LoopDetectorConfig`].
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    // ==================================================
    // Loop detection (forwarded to LoopDetectorConfig)
    // ==================================================
    /// Number of consecutive similar operations before declaring a loop.
    ///
    /// When the [`LoopDetector`] sees the same operation this many times
    /// in a row, [`DetectionManager::check_loop`] will report
    /// [`LoopStatus::is_looping`] as `true`.
    ///
    /// This threshold applies *per tool*, so if a custom [`ToolSignature`]
    /// provides per-tool overrides via [`ToolSignature::tool_thresholds`],
    /// those values take precedence for the matching tool name.
    ///
    /// Default: **3**.
    pub loop_threshold: usize,

    /// Number of repetitions that triggers a forced stop (0 = disabled).
    ///
    /// Once the loop detector has seen this many consecutive identical
    /// operations the framework should terminate the session. Set to `0`
    /// to disable forced stopping and rely solely on warnings.
    ///
    /// This maps to [`LoopDetectorConfig::stop_threshold`] during
    /// construction via [`DetectionConfig::to_loop_detector_config`].
    /// When the stop threshold is reached, [`LoopStatus::should_stop`]
    /// is set to `true` and the detector generates a `"STOPPING"` warning.
    ///
    /// Default: **10**.
    pub stop_threshold: usize,

    /// Whether loop detection is enabled.
    ///
    /// When `false`, [`DetectionManager::record_operation`] and
    /// [`DetectionManager::record_tool_call`] return
    /// [`DetectedPattern::NoPattern`] immediately without forwarding to
    /// the [`LoopDetector`]. This is useful in testing scenarios or when
    /// the consumer wants to rely solely on convergence detection.
    ///
    /// Default: **true**.
    pub enable_loop_detection: bool,

    /// Maximum number of operations to keep in the loop detector's history.
    ///
    /// Older operations are evicted once the ring buffer exceeds this
    /// size. A larger history lets the detector recognise longer cycles,
    /// at the cost of more memory.
    ///
    /// This maps to [`LoopDetectorConfig::window_size`] during
    /// construction. For most agents the default of 100 is more than
    /// sufficient — typical loops repeat within 3–10 operations.
    ///
    /// Default: **100**.
    pub max_history: usize,

    // ==================================================
    // Convergence detection
    // ==================================================
    /// Similarity threshold (0.0–1.0) for convergence detection.
    ///
    /// The [`ConvergenceDetector`] compares the Jaccard similarity of the
    /// most recent pair of responses. If the score is at or above this
    /// value for [`convergence_count`](Self::convergence_count) consecutive
    /// turns, convergence is declared.
    ///
    /// A higher threshold (e.g., 0.99) requires near-identical responses,
    /// while a lower threshold (e.g., 0.80) catches paraphrasing but may
    /// produce false positives.
    ///
    /// Default: **0.95**.
    pub convergence_threshold: f32,

    /// Number of consecutive similar responses required for convergence.
    ///
    /// A higher value makes the detector more tolerant — the agent can
    /// produce several similar responses before being flagged. Setting
    /// this to 1 would flag any two similar responses immediately.
    ///
    /// Maps to [`ConvergenceConfig::window_size`] during construction.
    ///
    /// Default: **3**.
    pub convergence_count: usize,

    /// Whether convergence detection is enabled.
    ///
    /// When `false`, [`DetectionManager::record_response`] returns
    /// [`DetectedPattern::NoPattern`] immediately without forwarding to
    /// the [`ConvergenceDetector`]. This is useful when the consumer
    /// wants to rely solely on loop detection, or in unit tests where
    /// convergence noise would be distracting.
    ///
    /// Default: **true**.
    pub enable_convergence_detection: bool,

    /// Action to take when convergence is detected.
    ///
    /// Determines whether the framework should emit a warning, force a
    /// stop, or silently note the event. See [`ConvergenceAction`] for
    /// available options.
    ///
    /// The action is forwarded to the internal [`ConvergenceDetector`]
    /// during construction via [`DetectionConfig::to_convergence_config`].
    /// It is consulted by the framework when [`DetectionManager::check_convergence`]
    /// returns a [`ConvergenceStatus`] with `detected == true`.
    ///
    /// Default: [`ConvergenceAction::default()`].
    pub on_converge: ConvergenceAction,

    /// Maximum number of responses to keep for convergence checking.
    ///
    /// The [`ConvergenceDetector`] maintains a sliding window of responses.
    /// Older responses are evicted once this limit is reached.
    ///
    /// A larger window retains more history (useful for spotting slow
    /// drift), while a smaller window reduces memory usage and makes the
    /// detector more responsive to recent changes.
    ///
    /// Default: **20**.
    pub max_response_history: usize,
}

impl Default for DetectionConfig {
    /// Produce a configuration with production-ready defaults.
    ///
    /// The defaults are chosen to be conservative enough for most agent
    /// workloads while still catching genuine stuck behaviours:
    ///
    /// | Field                          | Default                          |
    /// |--------------------------------|----------------------------------|
    /// | `loop_threshold`               | 3                                |
    /// | `stop_threshold`               | 10                               |
    /// | `enable_loop_detection`        | `true`                           |
    /// | `max_history`                  | 100                              |
    /// | `convergence_threshold`        | 0.95                             |
    /// | `convergence_count`            | 3                                |
    /// | `enable_convergence_detection` | `true`                           |
    /// | `on_converge`                  | [`ConvergenceAction::default()`] |
    /// | `max_response_history`         | 20                               |
    ///
    /// # When called
    ///
    /// By [`DetectionManager::new`] and anywhere a default config is needed.
    fn default() -> Self {
        Self {
            loop_threshold: 3,
            stop_threshold: 10,
            enable_loop_detection: true,
            max_history: 100,
            convergence_threshold: 0.95,
            convergence_count: 3,
            enable_convergence_detection: true,
            on_converge: ConvergenceAction::default(),
            max_response_history: 20,
        }
    }
}

impl DetectionConfig {
    /// Convert convergence settings into a [`ConvergenceConfig`].
    ///
    /// Maps the subset of fields that belong to the convergence subsystem
    /// (`enable_convergence_detection` → `enabled`, `convergence_count` →
    /// `window_size`, etc.). Called by [`DetectionManager::with_config`]
    /// during construction.
    ///
    /// # Field Mapping
    ///
    /// | `DetectionConfig` field        | [`ConvergenceConfig`] field            |
    /// |--------------------------------|----------------------------------------|
    /// | `enable_convergence_detection` | `enabled`                              |
    /// | `convergence_count`            | `window_size`                          |
    /// | `convergence_threshold`        | `similarity_threshold`                 |
    /// | `on_converge`                  | `on_converge`                          |
    ///
    /// # When called
    ///
    /// Internally by [`DetectionManager`] constructors; generally not
    /// needed by external callers.
    #[must_use]
    pub fn to_convergence_config(&self) -> ConvergenceConfig {
        ConvergenceConfig {
            enabled: self.enable_convergence_detection,
            window_size: self.convergence_count,
            similarity_threshold: self.convergence_threshold,
            on_converge: self.on_converge,
        }
    }

    /// Convert the loop-related settings into a [`LoopDetectorConfig`].
    ///
    /// Maps `max_history` → `window_size`, `loop_threshold` →
    /// `repetition_threshold`, and `stop_threshold` directly. All other
    /// [`LoopDetectorConfig`] fields inherit their defaults.
    ///
    /// # Field Mapping
    ///
    /// | `DetectionConfig` field | [`LoopDetectorConfig`] field |
    /// |-------------------------|------------------------------|
    /// | `max_history`           | `window_size`                |
    /// | `loop_threshold`        | `repetition_threshold`       |
    /// | `stop_threshold`        | `stop_threshold`             |
    ///
    /// # When called
    ///
    /// Internally by [`DetectionManager`] constructors; generally not
    /// needed by external callers.
    #[must_use]
    pub fn to_loop_detector_config(&self) -> LoopDetectorConfig {
        LoopDetectorConfig {
            window_size: self.max_history,
            repetition_threshold: self.loop_threshold,
            stop_threshold: self.stop_threshold,
            ..LoopDetectorConfig::default()
        }
    }
}

// ==================================================
// Statistics
// ==================================================

/// Cumulative statistics from the [`DetectionManager`].
///
/// Returned by [`DetectionManager::stats`] for observability and
/// debugging. All counters are monotonically increasing within a session
/// (until [`DetectionManager::reset`] is called).
///
/// This struct is [`Clone`] and [`Default`] so it can be cheaply snapshot
/// and serialised for logging or UI display.
///
/// # Example
///
/// ```rust,ignore
/// let stats = dm.stats();
/// println!("Turns: {}", stats.turns_analyzed);
/// println!("Loops: {}", stats.loops_detected);
/// println!("Convergences: {}", stats.convergences_detected);
/// println!("Current streak: {}", stats.current_streak);
/// ```
///
/// # See Also
///
/// - [`DetectionManager::stats`] — the accessor that returns snapshots of this type.
/// - [`DetectionManager::reset`] — zeroes all counters.
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    /// Total number of turns (operations + responses) analysed.
    ///
    /// Incremented by [`DetectionManager::record_operation`] each time an
    /// operation is recorded, regardless of whether a loop was found.
    /// This counter does **not** include responses recorded via
    /// [`DetectionManager::record_response`] — only tool-call operations.
    ///
    /// Reset to `0` by [`DetectionManager::reset`].
    pub turns_analyzed: usize,

    /// Number of times a loop was detected.
    ///
    /// Incremented only when [`DetectionManager::record_operation`] returns
    /// [`DetectedPattern::LoopDetected`]. This is a subset of
    /// [`turns_analyzed`](Self::turns_analyzed).
    ///
    /// A high ratio of `loops_detected` to `turns_analyzed` suggests the
    /// agent is frequently stuck. The framework can use this metric to
    /// decide whether to terminate the session early.
    ///
    /// Reset to `0` by [`DetectionManager::reset`].
    pub loops_detected: usize,

    /// Number of times convergence was detected.
    ///
    /// Incremented only when [`DetectionManager::record_response`] returns
    /// [`DetectedPattern::ConvergenceDetected`]. Unlike `loops_detected`,
    /// this counter tracks semantic similarity of free-text responses,
    /// not tool-call repetition.
    ///
    /// Reset to `0` by [`DetectionManager::reset`].
    pub convergences_detected: usize,

    /// Current streak of consecutive similar operations.
    ///
    /// Mirrors [`LoopStatus::repetition_count`] after the most recent
    /// [`DetectionManager::record_operation`] call. When this value reaches
    /// [`DetectionConfig::loop_threshold`], the next call to
    /// [`DetectionManager::record_operation`] will return
    /// [`DetectedPattern::LoopDetected`].
    ///
    /// Useful for progress bars or warning indicators that show "how close"
    /// the agent is to triggering a loop detection.
    ///
    /// Reset to `0` by [`DetectionManager::reset`].
    pub current_streak: usize,
}

// ==================================================
// Detection Manager
// ==================================================

/// Unified manager combining loop detection and convergence detection.
///
/// [`DetectionManager`] is the primary entry point for the detection
/// subsystem. It owns a [`LoopDetector`] (wrapped in `Arc` for cheap
/// sharing across compaction and response-analysis phases) and a
/// [`ConvergenceDetector`] (wrapped in `Mutex` for interior mutability).
///
/// # Construction
///
/// Choose a constructor based on your needs:
///
/// | Constructor | Use case |
/// |---|---|
/// | [`DetectionManager::new`] | Quick start with defaults |
/// | [`DetectionManager::with_config`] | Custom thresholds via [`DetectionConfig`] |
/// | [`DetectionManager::with_loop_detector`] | Inject a pre-built [`LoopDetector`] |
/// | [`DetectionManager::with_signature`] | Custom [`ToolSignature`] for JSON parsing |
///
/// # Loop Detection
///
/// Use [`Self::record_operation`] to record tool calls. Loop status is
/// checked via [`Self::check_loop`], which delegates to the internal
/// [`LoopDetector`].
///
/// # Convergence Detection
///
/// Record assistant responses via [`Self::record_response`] and check
/// for semantic similarity using [`Self::check_convergence`]. Both
/// delegate to the internal [`ConvergenceDetector`].
///
/// # Direct Access
///
/// For consumers that need direct access to the underlying detectors
/// (e.g., for compaction phases or response analysis phases), use
/// [`Self::loop_detector`] and [`Self::convergence_detector`].
///
/// # Lifecycle
///
/// ```text
/// new() / with_config()
///   → record_operation(op)    [each tool call]
///   → check_loop()            [after each turn]
///   → record_response(text)   [each assistant reply]
///   → check_convergence()     [after each turn]
///   → stats()                 [observability]
///   → reset()                 [between tasks]
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let dm = DetectionManager::new();
///
/// // Record tool calls
/// let pattern = dm.record_tool_call("Read", file_hash);
/// if let DetectedPattern::LoopDetected { repetitions, .. } = pattern {
///     tracing::warn!("Loop after {repetitions} reps");
/// }
///
/// // Record responses
/// dm.record_response("Working on step 1...");
///
/// // Query at any time
/// let stats = dm.stats();
/// println!("Turns: {}, Loops: {}", stats.turns_analyzed, stats.loops_detected);
///
/// // Reset between tasks
/// dm.reset();
/// ```
///
/// # Thread Safety
///
/// All public methods take `&self`. The two `Mutex` fields
/// (`convergence_detector` and `stats`) provide interior mutability,
/// so the manager can be shared across threads without a `mut` reference.
///
/// # See Also
///
/// - [`DetectionConfig`] — construction configuration.
/// - [`DetectedPattern`] — the result type returned by check methods.
/// - [`DetectionStats`] — cumulative observability counters.
pub struct DetectionManager {
    config: DetectionConfig,
    loop_detector: Arc<LoopDetector>,
    convergence_detector: Mutex<ConvergenceDetector>,
    stats: Mutex<DetectionStats>,
}

impl DetectionManager {
    /// Create a new detection manager with default configuration.
    ///
    /// Convenience wrapper around [`Self::with_config`] that passes
    /// [`DetectionConfig::default`]. Suitable for quick prototyping or
    /// when the default thresholds (loop=3, stop=10, convergence=0.95)
    /// are acceptable.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dm = DetectionManager::new();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn new() -> Result<Self, ConvergenceConfigError> {
        Self::with_config(DetectionConfig::default())
    }

    /// Create a new detection manager with custom configuration.
    ///
    /// Builds a [`LoopDetector`] using a [`NoOpToolSignature`](super::loop_detector::NoOpToolSignature)
    /// and a [`ConvergenceDetector`] from the relevant config fields.
    /// For tool-specific JSON parsing, use [`Self::with_signature`] or
    /// [`Self::with_loop_detector`] instead.
    ///
    /// # Construction Flow
    ///
    /// ```text
    /// with_config(config)
    ///   ├─ config.to_loop_detector_config() → LoopDetectorConfig
    ///   │    └─ LoopDetector::new(ldc, NoOpToolSignature)
    ///   ├─ config.to_convergence_config()  → ConvergenceConfig
    ///   │    └─ ConvergenceDetector::new(cc)
    ///   └─ DetectionStats::default()
    /// ```
    ///
    /// # When called
    ///
    /// Typically during agent initialisation, once per session.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = DetectionConfig {
    ///     loop_threshold: 5,
    ///     ..DetectionConfig::default()
    /// };
    /// let dm = DetectionManager::with_config(config);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn with_config(config: DetectionConfig) -> Result<Self, ConvergenceConfigError> {
        let loop_detector = Arc::new(LoopDetector::new(
            config.to_loop_detector_config(),
            Arc::new(super::loop_detector::NoOpToolSignature),
        ));
        let convergence_detector =
            Mutex::new(ConvergenceDetector::new(config.to_convergence_config())?);
        Ok(Self {
            config,
            loop_detector,
            convergence_detector,
            stats: Mutex::new(DetectionStats::default()),
        })
    }

    /// Create with an explicit [`LoopDetector`] instance.
    ///
    /// Use this when the caller has already constructed a [`LoopDetector`]
    /// with a custom [`ToolSignature`] or non-default thresholds and wants
    /// to inject it directly. The detector is wrapped in `Arc` internally
    /// for cheap sharing.
    ///
    /// This constructor **does not** use the `loop_threshold`, `stop_threshold`,
    /// or `max_history` fields from `config` — those are already baked into
    /// the provided `loop_detector`. Only the convergence-related fields
    /// are read from `config`.
    ///
    /// # When called
    ///
    /// During agent initialisation when the caller needs fine-grained
    /// control over the loop detector's internals.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ld = LoopDetector::new(ld_config, my_signature);
    /// let dm = DetectionManager::with_loop_detector(config, ld);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn with_loop_detector(
        config: DetectionConfig,
        loop_detector: LoopDetector,
    ) -> Result<Self, ConvergenceConfigError> {
        let convergence_detector =
            Mutex::new(ConvergenceDetector::new(config.to_convergence_config())?);
        Ok(Self {
            config,
            loop_detector: Arc::new(loop_detector),
            convergence_detector,
            stats: Mutex::new(DetectionStats::default()),
        })
    }

    /// Create with a specific [`ToolSignature`] for tool-specific parsing.
    ///
    /// Constructs a [`LoopDetector`] using the provided `signature` and
    /// its associated [`tool_thresholds`](ToolSignature::tool_thresholds).
    /// Useful when the consumer provides its own tool-aware signature
    /// (e.g., `DchToolSignature` for dch.sh tools that extracts
    /// `file_path` from JSON inputs).
    ///
    /// The tool thresholds from the signature are merged into a
    /// [`LoopDetectorConfig`] derived from [`DetectionConfig::default`],
    /// overriding the generic [`DetectionConfig::loop_threshold`] for any
    /// tool listed in [`ToolSignature::tool_thresholds`].
    ///
    /// # When called
    ///
    /// During agent initialisation when the agent runtime has a domain-
    /// specific [`ToolSignature`] implementation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let signature = Arc::new(MyToolSignature);
    /// let dm = DetectionManager::with_signature(signature);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`ToolSignature`] — the trait for tool-specific JSON parsing.
    /// - [`Operation::from_input_with_signature`] — creates operations using a signature.
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn with_signature(
        signature: Arc<dyn ToolSignature>,
    ) -> Result<Self, ConvergenceConfigError> {
        let config = DetectionConfig::default();
        let mut ldc = config.to_loop_detector_config();
        ldc.tool_thresholds = signature.tool_thresholds();
        let loop_detector = Arc::new(LoopDetector::new(ldc, signature));
        let convergence_detector =
            Mutex::new(ConvergenceDetector::new(config.to_convergence_config())?);
        let stats = Mutex::new(DetectionStats::default());
        Ok(Self {
            config,
            loop_detector,
            convergence_detector,
            stats,
        })
    }

    // ==================================================
    // Loop detection (delegated to LoopDetector)
    // ==================================================

    /// Record an [`Operation`] for loop detection and return the current
    /// [`DetectedPattern`].
    ///
    /// This is the primary entry point for feeding data into the loop
    /// detector. It performs three steps:
    ///
    /// 1. Forwards `operation` to [`LoopDetector::record`].
    /// 2. Calls [`LoopDetector::check_loop`] to evaluate the new state.
    /// 3. Updates [`DetectionStats`] and returns the appropriate
    ///    [`DetectedPattern`] variant.
    ///
    /// When [`DetectionConfig::enable_loop_detection`] is `false`, this
    /// method short-circuits and returns [`DetectedPattern::NoPattern`]
    /// without touching the loop detector or statistics.
    ///
    /// # When called
    ///
    /// After every tool invocation during an agent turn — typically by the
    /// framework's turn-processing loop or by the convenience wrappers
    /// [`Self::record_tool_call`] / [`Self::record_tool_call_with_result`].
    ///
    /// # Returns
    ///
    /// - [`DetectedPattern::LoopDetected`] if the same operation has been
    ///   seen ≥ [`DetectionConfig::loop_threshold`] times consecutively.
    /// - [`DetectedPattern::NoPattern`] otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let op = Operation::new("Read", "/etc/hosts");
    /// match dm.record_operation(op) {
    ///     DetectedPattern::LoopDetected { repetitions, pattern_description } => {
    ///         tracing::warn!("Loop: {pattern_description} (×{repetitions})");
    ///     }
    ///     _ => { /* no loop yet */ }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::record_tool_call`] — convenience wrapper for hash-based calls.
    /// - [`Self::record_tool_call_with_result`] — result-aware variant.
    /// - [`Self::check_loop`] — read-only query without recording.
    pub fn record_operation(&self, operation: Operation) -> DetectedPattern {
        if !self.config.enable_loop_detection {
            return DetectedPattern::NoPattern;
        }

        self.loop_detector.record(operation);

        // Check if this triggered a loop
        let status = self.loop_detector.check_loop();
        if status.is_looping {
            let mut guard = self.stats.lock().unwrap_or_else(|e| {
                tracing::warn!("stats lock poisoned, recovering");
                e.into_inner()
            });
            guard.turns_analyzed = guard.turns_analyzed.saturating_add(1);
            guard.loops_detected = guard.loops_detected.saturating_add(1);
            guard.current_streak = status.repetition_count;
            DetectedPattern::LoopDetected {
                repetitions: status.repetition_count,
                pattern_description: status
                    .repeated_operations
                    .first()
                    .map(|op| format!("{}({})", op.tool, op.primary_param))
                    .unwrap_or_default(),
            }
        } else {
            let mut guard = self.stats.lock().unwrap_or_else(|e| {
                tracing::warn!("stats lock poisoned, recovering");
                e.into_inner()
            });
            guard.turns_analyzed = guard.turns_analyzed.saturating_add(1);
            DetectedPattern::NoPattern
        }
    }

    /// Record a tool call for loop detection by tool name and input hash.
    ///
    /// Creates an [`Operation`] from a `tool` name and `input_hash`, then
    /// delegates to [`Self::record_operation`]. This is the easiest way to
    /// feed data into the loop detector when you only need a numeric hash
    /// as the primary parameter rather than a full JSON-based signature.
    ///
    /// The resulting [`Operation`] will have `primary_param` set to
    /// `"hash:{input_hash}"` and no result hash.
    ///
    /// For richer tool-specific parsing (e.g., extracting `file_path` from
    /// a JSON input), construct an [`Operation`] with
    /// [`Operation::from_input_with_signature`] and call
    /// [`Self::record_operation`] directly.
    ///
    /// # When called
    ///
    /// After each tool invocation during an agent turn when only a hash
    /// identity is available.
    ///
    /// # Returns
    ///
    /// Same as [`Self::record_operation`] — either
    /// [`DetectedPattern::LoopDetected`] or [`DetectedPattern::NoPattern`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pattern = dm.record_tool_call("Bash", 0xABCD);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::record_tool_call_with_result`] — adds result hashing for progress detection.
    /// - [`Self::record_operation`] — full API with arbitrary [`Operation`] values.
    pub fn record_tool_call(&self, tool: &str, input_hash: u64) -> DetectedPattern {
        let operation = Operation::new(tool, format!("hash:{input_hash}"));
        self.record_operation(operation)
    }

    /// Record a tool call with a result hash for result-aware loop detection.
    ///
    /// Behaves like [`Self::record_tool_call`] but also accepts an optional
    /// `result_hash`. When the same tool + input produces *different* result
    /// hashes across turns, the loop detector considers the agent to be
    /// making progress — even though the tool and input are identical. Only
    /// when both input and result match across consecutive calls is a loop
    /// flagged.
    ///
    /// This is essential for tools like `Read` where calling the same file
    /// is perfectly fine if the file content is changing (e.g., the agent
    /// is editing it).
    ///
    /// # How it works
    ///
    /// ```text
    /// Turn 1: Read("/foo.txt") → result_hash=0xA  ─┐
    /// Turn 2: Read("/foo.txt") → result_hash=0xA  ─┤ Same hash → loop candidate
    /// Turn 3: Read("/foo.txt") → result_hash=0xA  ─┘
    ///   → LoopDetected after loop_threshold repetitions
    ///
    /// Turn 1: Read("/foo.txt") → result_hash=0xA
    /// Turn 2: Read("/foo.txt") → result_hash=0xB  ← Different hash → not a loop
    /// ```
    ///
    /// # When called
    ///
    /// After each tool invocation when the caller has access to the tool's
    /// output and can compute a representative hash (e.g., via
    /// [`hash_result`](super::loop_detector::hash_result)).
    ///
    /// # Returns
    ///
    /// Same as [`Self::record_operation`] — either
    /// [`DetectedPattern::LoopDetected`] or [`DetectedPattern::NoPattern`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Same tool + input, different results → not a loop
    /// for i in 0..5 {
    ///     let result = hash_result(&format!("output {i}"));
    ///     dm.record_tool_call_with_result("Bash", 42, result);
    /// }
    /// assert!(matches!(dm.check_loop().is_looping, false));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Operation::with_result_hash`] — attaches a result hash to an operation.
    /// - [`hash_result`] — utility for computing result hashes.
    ///
    /// [`hash_result`]: super::loop_detector::hash_result
    pub fn record_tool_call_with_result(
        &self,
        tool: &str,
        input_hash: u64,
        result_hash: Option<u64>,
    ) -> DetectedPattern {
        let operation =
            Operation::new(tool, format!("hash:{input_hash}")).with_result_hash(result_hash);
        self.record_operation(operation)
    }

    /// Query the internal [`LoopDetector`] for the current loop status.
    ///
    /// Returns a full [`LoopStatus`] snapshot including repetition counts,
    /// the `should_stop` flag, and any warning message. Unlike
    /// [`Self::record_operation`], this method does **not** modify any
    /// state — it is a pure read.
    ///
    /// # When called
    ///
    /// By the framework's turn-processing loop after each turn, or by
    /// any observability layer that needs to report loop status without
    /// side effects.
    ///
    /// # Returns
    ///
    /// A [`LoopStatus`] with [`LoopStatus::is_looping`] set to `true` if
    /// the repetition count has reached [`DetectionConfig::loop_threshold`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = dm.check_loop();
    /// if status.should_stop {
    ///     tracing::error!("Forcing stop — loop detected: {:?}", status.warning);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::record_operation`] — records an operation *and* checks for loops.
    /// - [`Self::check_current_pattern`] — checks both loop and convergence.
    #[must_use]
    pub fn check_loop(&self) -> LoopStatus {
        self.loop_detector.check_loop()
    }

    /// Obtain a shared reference to the internal [`LoopDetector`].
    ///
    /// Useful for direct access to loop state, e.g. inspecting
    /// the turn count or history during diagnostics.
    ///
    /// # When called
    ///
    /// By internal subsystems that need direct access to the loop
    /// detector's methods (e.g., to query [`LoopDetector::turn_count`])
    /// without going through the [`DetectionManager`] facade.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ld = dm.loop_detector();
    /// println!("Turns seen by loop detector: {}", ld.turn_count());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::convergence_detector`] — access to the convergence detector.
    #[must_use]
    pub fn loop_detector(&self) -> &LoopDetector {
        &self.loop_detector
    }

    /// Obtain a shared reference to the internal [`ConvergenceDetector`].
    ///
    /// Returns `&Mutex<ConvergenceDetector>` so callers can lock and
    /// invoke methods on the convergence detector directly — for example
    /// to call [`ConvergenceDetector::check_convergence`] or inspect its
    /// internal window during a response-analysis phase.
    ///
    /// # Thread safety
    ///
    /// The returned `Mutex` guards the detector's mutable state. Callers
    /// must lock before accessing. If the lock is poisoned (due to a panic
    /// in another thread), the framework recovers automatically in
    /// [`Self::record_response`] and [`Self::check_convergence`].
    ///
    /// # When called
    ///
    /// By internal subsystems that need direct access to the convergence
    /// detector without going through the [`DetectionManager`] facade.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cd = dm.convergence_detector();
    /// let status = cd.lock().unwrap().check_convergence();
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::loop_detector`] — access to the loop detector.
    #[must_use]
    pub fn convergence_detector(&self) -> &Mutex<ConvergenceDetector> {
        &self.convergence_detector
    }

    // ==================================================
    // Convergence detection (handled internally)
    // ==================================================

    /// Record an assistant response for convergence detection.
    ///
    /// Forwards `response` to the internal [`ConvergenceDetector`] via
    /// [`ConvergenceDetector::add_response`], which tokenises the text
    /// into word-level tokens, computes Jaccard similarity against the
    /// previous response, and updates the consecutive-similarity counter.
    ///
    /// If the similarity score exceeds
    /// [`DetectionConfig::convergence_threshold`] for
    /// [`DetectionConfig::convergence_count`] consecutive turns, this
    /// method returns [`DetectedPattern::ConvergenceDetected`] and
    /// increments [`DetectionStats::convergences_detected`].
    ///
    /// When [`DetectionConfig::enable_convergence_detection`] is `false`,
    /// this method short-circuits and returns [`DetectedPattern::NoPattern`]
    /// without touching the convergence detector or statistics.
    ///
    /// # Tokenisation
    ///
    /// Responses are split on whitespace into word-level tokens,
    /// lowercased, and compared using Jaccard similarity (intersection
    /// over union of token sets). This is fast and language-agnostic
    /// but does not capture semantic equivalence — two paraphrased
    /// sentences with different vocabulary will have a low score.
    ///
    /// # When called
    ///
    /// After each assistant response during an agent turn — typically by
    /// the framework's turn-processing loop.
    ///
    /// # Returns
    ///
    /// - [`DetectedPattern::ConvergenceDetected`] if the last *N*
    ///   responses exceed the similarity threshold.
    /// - [`DetectedPattern::NoPattern`] otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pattern = dm.record_response("I'm still working on the task.");
    /// if let DetectedPattern::ConvergenceDetected { similarity, .. } = pattern {
    ///     tracing::warn!("Responses converging (similarity={similarity:.2})");
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::check_convergence`] — read-only check without recording.
    /// - [`Self::record_operation`] — the loop-detection counterpart for tool calls.
    pub fn record_response(&self, response: &str) -> DetectedPattern {
        if !self.config.enable_convergence_detection {
            return DetectedPattern::NoPattern;
        }
        let status = self
            .convergence_detector
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("convergence detector lock poisoned, recovering");
                e.into_inner()
            })
            .add_response(response);
        if status.detected {
            let mut guard = self.stats.lock().unwrap_or_else(|e| {
                tracing::warn!("stats lock poisoned, recovering");
                e.into_inner()
            });
            guard.convergences_detected = guard.convergences_detected.saturating_add(1);
            return DetectedPattern::ConvergenceDetected {
                similarity: status.similarity_score,
                consecutive_count: status.consecutive_count,
            };
        }
        DetectedPattern::NoPattern
    }

    /// Query the internal [`ConvergenceDetector`] for the current
    /// convergence status.
    ///
    /// Returns a full [`ConvergenceStatus`] snapshot including the
    /// similarity score, consecutive count, and detection flag. Unlike
    /// [`Self::record_response`], this method does **not** add a new
    /// response — it is a pure read of the detector's current state.
    ///
    /// # When called
    ///
    /// By the framework's turn-processing loop or observability layers
    /// that need to inspect convergence state without side effects.
    ///
    /// # Returns
    ///
    /// A [`ConvergenceStatus`] with [`ConvergenceStatus::detected`] set
    /// to `true` if the similarity threshold has been exceeded for enough
    /// consecutive turns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = dm.check_convergence();
    /// println!("Similarity: {:.2}", status.similarity_score);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::record_response`] — records a response *and* checks convergence.
    /// - [`Self::check_current_pattern`] — checks both loop and convergence.
    #[must_use]
    pub fn check_convergence(&self) -> ConvergenceStatus {
        self.convergence_detector
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("convergence detector lock poisoned, recovering");
                e.into_inner()
            })
            .check_convergence()
    }

    // ==================================================
    // Pattern check
    // ==================================================

    /// Inspect the current detection state without recording new data.
    ///
    /// Checks both the loop detector and the convergence detector in
    /// sequence and returns the first non-`NoPattern` result (loop takes
    /// priority over convergence). This is a **read-only** operation —
    /// no statistics are updated and no new data is recorded.
    ///
    /// # Priority order
    ///
    /// ```text
    /// check_current_pattern()
    ///   ├─ check_loop()           → LoopDetected?  (priority 1)
    ///   └─ check_convergence()    → ConvergenceDetected?  (priority 2)
    ///      └─ NoPattern  (fallback)
    /// ```
    ///
    /// Loop detection takes priority because a looping agent is more
    /// urgently stuck than a converging one.
    ///
    /// # When called
    ///
    /// By the framework when it needs a quick snapshot of whether anything
    /// has been detected so far — for example before deciding whether to
    /// inject a warning into the agent's context or terminate the session.
    ///
    /// # Returns
    ///
    /// - [`DetectedPattern::LoopDetected`] if the loop detector is
    ///   currently flagging a loop.
    /// - [`DetectedPattern::ConvergenceDetected`] if no loop was found
    ///   but the convergence detector has fired.
    /// - [`DetectedPattern::NoPattern`] if neither detector has fired.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// match dm.check_current_pattern() {
    ///     DetectedPattern::LoopDetected { repetitions, .. } => {
    ///         tracing::warn!("Loop detected: {repetitions} reps");
    ///     }
    ///     DetectedPattern::ConvergenceDetected { similarity, .. } => {
    ///         tracing::info!("Converging: similarity={similarity:.2}");
    ///     }
    ///     DetectedPattern::NoPattern => { /* all clear */ }
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::check_loop`] — loop-only check.
    /// - [`Self::check_convergence`] — convergence-only check.
    #[must_use]
    pub fn check_current_pattern(&self) -> DetectedPattern {
        let loop_status = self.loop_detector.check_loop();
        if loop_status.is_looping {
            return DetectedPattern::LoopDetected {
                repetitions: loop_status.repetition_count,
                pattern_description: loop_status
                    .repeated_operations
                    .first()
                    .map(|op| format!("{}({})", op.tool, op.primary_param))
                    .unwrap_or_default(),
            };
        }
        let convergence = self.check_convergence();
        if convergence.detected {
            return DetectedPattern::ConvergenceDetected {
                similarity: convergence.similarity_score,
                consecutive_count: convergence.consecutive_count,
            };
        }
        DetectedPattern::NoPattern
    }

    // ==================================================
    // Accessors
    // ==================================================

    /// Take a snapshot of the cumulative detection statistics.
    ///
    /// Locks the internal `Mutex`, clones the [`DetectionStats`] struct,
    /// and returns it. The snapshot reflects all operations recorded since
    /// the last [`Self::reset`] call (or since construction).
    ///
    /// # When called
    ///
    /// By observability layers, logging, or UI code that needs to display
    /// detection metrics.
    ///
    /// # Returns
    ///
    /// A cloned [`DetectionStats`] with current counter values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = dm.stats();
    /// println!("Turns: {}, Loops: {}, Convergences: {}",
    ///     stats.turns_analyzed,
    ///     stats.loops_detected,
    ///     stats.convergences_detected,
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// - [`DetectionStats`] — the struct returned by this method.
    /// - [`Self::reset`] — zeroes all counters.
    #[must_use]
    pub fn stats(&self) -> DetectionStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Obtain a reference to the configuration snapshot.
    ///
    /// The returned [`DetectionConfig`] is the same one passed at
    /// construction time. Because it is stored immutably inside the
    /// [`DetectionManager`], it can be read without taking any locks.
    ///
    /// # When called
    ///
    /// By any code that needs to inspect the current thresholds or
    /// feature flags — for example to log them at startup or to decide
    /// whether to skip a detection step.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cfg = dm.config();
    /// if !cfg.enable_loop_detection {
    ///     tracing::info!("Loop detection is disabled");
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`DetectionConfig`] — full description of all config fields.
    /// - [`Self::with_config`] — the constructor that accepts a config.
    #[must_use]
    pub fn config(&self) -> &DetectionConfig {
        &self.config
    }

    // ==================================================
    // Reset
    // ==================================================

    /// Reset all internal detection state, as if the manager were freshly
    /// constructed.
    ///
    /// Clears the [`LoopDetector`] history via [`LoopDetector::reset`],
    /// wipes the [`ConvergenceDetector`] window via
    /// [`ConvergenceDetector::clear`], and zeroes out all counters in
    /// [`DetectionStats`]. Use this between independent tasks or when the
    /// agent starts a new sub-goal.
    ///
    /// # What is reset
    ///
    /// | Component | Effect |
    /// |-----------|--------|
    /// | [`LoopDetector`] | History, repetition counts, warnings |
    /// | [`ConvergenceDetector`] | Response window, similarity scores |
    /// | [`DetectionStats`] | All counters zeroed |
    /// | [`DetectionConfig`] | **Not** reset — preserved from construction |
    ///
    /// # When called
    ///
    /// By the framework when transitioning between tasks, or manually by
    /// test code that needs a clean slate.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// dm.record_tool_call("Read", 42);
    /// dm.record_response("Working on step 1...");
    /// dm.reset();
    /// assert!(matches!(dm.check_current_pattern(), DetectedPattern::NoPattern));
    /// assert_eq!(dm.stats().turns_analyzed, 0);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Self::stats`] — check the stats before resetting.
    /// - [`LoopDetector::reset`] — the underlying reset.
    pub fn reset(&self) {
        self.loop_detector.reset();
        self.convergence_detector
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("convergence detector lock poisoned, recovering");
                e.into_inner()
            })
            .clear();
        *self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = DetectionStats::default();
    }
}

// ==================================================
// Tests
// ==================================================

/// Tests for [`DetectionManager`].
///
/// Verifies loop detection, convergence detection, result-aware detection,
/// reset behaviour, feature-flag toggling, stop thresholds, rich [`Operation`]
/// recording, and config mapping.
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a freshly constructed [`DetectionManager`] reports [`DetectedPattern::NoPattern`].
    ///
    /// Creates a new manager and immediately calls [`check_current_pattern`](DetectionManager::check_current_pattern).
    /// Asserts that no pattern is detected before any data has been recorded.
    #[test]
    fn test_no_pattern_initially() {
        let dm = DetectionManager::new().unwrap();
        assert!(matches!(
            dm.check_current_pattern(),
            DetectedPattern::NoPattern
        ));
    }

    /// Verify that repeating the same tool call triggers [`DetectedPattern::LoopDetected`].
    ///
    /// Calls [`record_tool_call`](DetectionManager::record_tool_call) with identical arguments
    /// five times. Asserts that at least one call returns a loop detection result,
    /// confirming the default [`DetectionConfig::loop_threshold`] of 3 is respected.
    #[test]
    fn test_loop_detection() {
        let dm = DetectionManager::new().unwrap();
        // Same operation repeated many times
        for _ in 0..5 {
            let result = dm.record_tool_call("read_file", 42);
            if matches!(result, DetectedPattern::LoopDetected { .. }) {
                return; // test passes
            }
        }
        panic!("Expected loop detection after 5 identical calls");
    }

    /// Verify that submitting identical responses triggers [`DetectedPattern::ConvergenceDetected`].
    ///
    /// Calls [`record_response`](DetectionManager::record_response) with the same string five
    /// times. Asserts that convergence is detected, validating the Jaccard similarity
    /// check and the [`DetectionConfig::convergence_count`] threshold.
    #[test]
    fn test_convergence_detection() {
        let dm = DetectionManager::new().unwrap();
        let response = "I am working on the task and making progress";
        for _ in 0..5 {
            let result = dm.record_response(response);
            if matches!(result, DetectedPattern::ConvergenceDetected { .. }) {
                return; // test passes
            }
        }
        panic!("Expected convergence detection after 5 identical responses");
    }

    /// Verify that [`reset`](DetectionManager::reset) clears all detection state.
    ///
    /// Records a tool call and a response, then calls [`reset`](DetectionManager::reset).
    /// Asserts that [`check_current_pattern`](DetectionManager::check_current_pattern) returns
    /// [`DetectedPattern::NoPattern`] and that [`stats`](DetectionManager::stats) shows zero turns.
    #[test]
    fn test_reset() {
        let dm = DetectionManager::new().unwrap();
        dm.record_tool_call("read_file", 42);
        dm.record_response("hello");
        dm.reset();
        assert!(matches!(
            dm.check_current_pattern(),
            DetectedPattern::NoPattern
        ));
        let stats = dm.stats();
        assert_eq!(stats.turns_analyzed, 0);
    }

    /// Verify that disabling both detectors causes all calls to return [`DetectedPattern::NoPattern`].
    ///
    /// Constructs a [`DetectionManager`] with `enable_loop_detection: false` and
    /// `enable_convergence_detection: false`. Records 10 identical tool calls and asserts
    /// none trigger a detection, confirming the feature-flag short-circuits work.
    #[test]
    fn test_no_detection_when_disabled() {
        let config = DetectionConfig {
            enable_loop_detection: false,
            enable_convergence_detection: false,
            ..Default::default()
        };
        let dm = DetectionManager::with_config(config).unwrap();
        for _ in 0..10 {
            let result = dm.record_tool_call("read_file", 42);
            assert!(matches!(result, DetectedPattern::NoPattern));
        }
    }

    /// Verify that the [`LoopStatus::should_stop`] flag activates after `stop_threshold` repetitions.
    ///
    /// Configures `loop_threshold: 3` and `stop_threshold: 5`, then records 5 identical
    /// operations. Asserts that [`check_loop`](DetectionManager::check_loop) reports
    /// [`is_looping`](LoopStatus::is_looping), [`should_stop`](LoopStatus::should_stop), and a
    /// warning containing `"STOPPING"`.
    #[test]
    fn test_loop_status_should_stop() {
        let config = DetectionConfig {
            loop_threshold: 3,
            stop_threshold: 5,
            ..Default::default()
        };
        let dm = DetectionManager::with_config(config).unwrap();

        // Record 5 identical operations
        for _ in 0..5 {
            dm.record_tool_call("read_file", 42);
        }

        let status = dm.check_loop();
        assert!(status.is_looping);
        assert!(status.should_stop);
        assert!(status.warning.is_some());
        assert!(status.warning.unwrap().contains("STOPPING"));
    }

    /// Verify loop detection using the rich [`Operation`] API.
    ///
    /// Creates [`Operation`] values with [`Operation::new`] and records them via
    /// [`record_operation`](DetectionManager::record_operation). Asserts that the loop
    /// detector correctly identifies repeated tool + parameter pairs and reports
    /// a repetition count of at least 3.
    #[test]
    fn test_record_operation_with_operation_struct() {
        let dm = DetectionManager::new().unwrap();

        // Record operations using the rich Operation API
        for _ in 0..5 {
            dm.record_operation(Operation::new("Read", "/test/file.txt"));
        }

        let status = dm.check_loop();
        assert!(status.is_looping);
        assert!(status.repetition_count >= 3);
    }

    /// Verify that [`Operation::from_input_with_signature`] integrates with loop detection.
    ///
    /// Parses a JSON input using [`NoOpToolSignature`](super::loop_detector::NoOpToolSignature)
    /// and records 5 operations. Because `NoOpToolSignature` produces an empty `primary_param`,
    /// all `"Read"` operations are considered identical and a loop is detected.
    #[test]
    fn test_record_operation_from_input() {
        use super::super::loop_detector::NoOpToolSignature;
        let dm = DetectionManager::new().unwrap();
        let input = serde_json::json!({"file_path": "/test/file.txt"});

        for _ in 0..5 {
            dm.record_operation(Operation::from_input_with_signature(
                "Read",
                &input,
                &NoOpToolSignature,
            ));
        }

        let status = dm.check_loop();
        // With NoOpToolSignature, primary_param is empty, so all "Read" ops
        // with empty param will be identical — loop is detected.
        assert!(status.is_looping);
    }

    /// Verify that changing result hashes prevent loop detection.
    ///
    /// Calls [`record_tool_call_with_result`](DetectionManager::record_tool_call_with_result)
    /// with the same tool and input but a different result hash each time. Asserts that
    /// [`check_loop`](DetectionManager::check_loop) reports `is_looping: false`, confirming
    /// that the result-aware logic treats varying outputs as progress.
    #[test]
    fn test_result_aware_detection() {
        let dm = DetectionManager::new().unwrap();

        // Same operation, different results = not a loop
        for i in 0..5 {
            let hash = super::super::loop_detector::hash_result(&format!("output {i}"));
            dm.record_tool_call_with_result("Bash", 42, hash);
        }

        let status = dm.check_loop();
        assert!(!status.is_looping, "Different results should not be a loop");
    }

    /// Verify that identical result hashes trigger loop detection.
    ///
    /// Calls [`record_tool_call_with_result`](DetectionManager::record_tool_call_with_result)
    /// with the same tool, input, *and* result hash. Asserts that the loop detector
    /// identifies this as a genuine loop, confirming the result-aware path works when
    /// results do not change.
    #[test]
    fn test_result_aware_same_result_is_loop() {
        let dm = DetectionManager::new().unwrap();

        let hash = super::super::loop_detector::hash_result("same output");
        for _ in 0..5 {
            dm.record_tool_call_with_result("Bash", 42, hash);
        }

        let status = dm.check_loop();
        assert!(status.is_looping, "Same results should be detected as loop");
    }

    /// Verify that [`loop_detector`](DetectionManager::loop_detector) provides access
    /// to the inner [`LoopDetector`].
    ///
    /// Calls [`loop_detector()`](DetectionManager::loop_detector) on a fresh manager
    /// and asserts that [`turn_count`](LoopDetector::turn_count) returns 0, confirming
    /// the accessor returns a usable reference.
    #[test]
    fn test_access_loop_detector() {
        let dm = DetectionManager::new().unwrap();
        // Should be able to access the inner LoopDetector
        assert_eq!(dm.loop_detector().turn_count(), 0);
    }

    /// Verify that [`DetectionConfig::to_loop_detector_config`] correctly maps fields.
    ///
    /// Creates a [`DetectionConfig`] with custom `loop_threshold`, `stop_threshold`, and
    /// `max_history`, then converts it to a [`LoopDetectorConfig`]. Asserts that each field
    /// maps to the expected value (`repetition_threshold`, `stop_threshold`, `window_size`).
    #[test]
    fn test_config_to_loop_detector_config() {
        let config = DetectionConfig {
            loop_threshold: 5,
            stop_threshold: 15,
            max_history: 200,
            ..Default::default()
        };
        let ldc = config.to_loop_detector_config();
        assert_eq!(ldc.repetition_threshold, 5);
        assert_eq!(ldc.stop_threshold, 15);
        assert_eq!(ldc.window_size, 200);
    }
}
