//! Detection manager — loop and convergence detection for agent behavior.
//!
//! [`DetectionManager`] unifies two complementary
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
//! `DetectionManager` is a **facade** that owns one `LoopDetector` and one
//! `ConvergenceDetector`, forwarding calls to each and merging their results
//! into a unified [`DetectedPattern`] enum.
//!
//! **Data flow** — On every agent turn the framework feeds two kinds of
//! telemetry into the manager:
//!
//! 1. *Tool calls* are sent to `record_operation` (or its convenience
//!    wrappers `record_tool_call` / `record_tool_call_with_result`), which
//!    hands the operation to the `LoopDetector`. The loop detector compares
//!    consecutive operations by tool name, primary parameter, and optional
//!    result hash; when the same signature repeats ≥ `loop_threshold`
//!    times it reports a loop.
//!
//! 2. *Assistant responses* (free text) are sent to `record_response`,
//!    which hands the text to the `ConvergenceDetector`. The convergence
//!    detector tokenises the text into words, computes Jaccard similarity
//!    against the previous response, and fires when similarity stays above
//!    `convergence_threshold` for `convergence_count` consecutive turns.
//!
//! **Merging** — `check_current_pattern` queries both detectors and
//! returns the first non-`NoPattern` result. Loop detection takes priority
//! over convergence because a tool-calling loop is a stronger signal of
//! being stuck.
//!
//! **Outcome** — The three possible results are carried by [`DetectedPattern`]:
//! `NoPattern` (agent is making progress), `LoopDetected` (repeated tool
//! calls), or `ConvergenceDetected` (semantically similar responses).
//!
//! # Provided Types
//!
//! - [`DetectionManager`] — facade that owns and delegates to both detectors.
//! - [`DetectionConfig`] — unified configuration for loop + convergence tuning.
//! - [`DetectedPattern`] — summary enum returned by every check method.
//! - [`DetectionStats`] — cumulative statistics exposed for observability.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::detection::manager::{
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
    /// Repeated tool-call pattern detected.
    ///
    /// Emitted by [`DetectionManager::record_operation`] (and the
    /// convenience wrappers [`DetectionManager::record_tool_call`] /
    /// [`DetectionManager::record_tool_call_with_result`]) when the
    /// [`LoopDetector`] observes the same operation at least
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
        /// How many times the repeated operation has been observed.
        ///
        /// Equal to [`LoopStatus::repetition_count`] at the moment the
        /// loop crossed
        /// [`DetectionConfig::loop_threshold`]. Callers can compare it
        /// against the threshold to decide between a soft warning and a
        /// hard stop.
        repetitions: usize,

        /// Human-readable summary of the repeated operation.
        ///
        /// Formatted as `"ToolName(primary_param)"` (for example
        /// `"Read(/etc/hosts)"` or `"Bash(ls -la)"`) using the
        /// tool-signature extractor. Suitable for log lines and
        /// user-facing diagnostics.
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
    /// free-text replies. It fires when the agent keeps saying
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
        /// Jaccard similarity of the most recent response pair.
        ///
        /// A value in `[0.0, 1.0]` where `1.0` means identical
        /// token sets. The detector fires once this score meets or
        /// exceeds
        /// [`DetectionConfig::convergence_threshold`]; callers can log
        /// it to show *how* similar the converged responses were.
        similarity: f32,

        /// Number of consecutive response pairs above the threshold.
        ///
        /// Reaches
        /// [`DetectionConfig::convergence_count`] when convergence is
        /// declared. A higher count means a longer run of near-identical
        /// replies, which is a stronger signal that the agent is stuck.
        consecutive_count: usize,
    },

    /// Neither the loop detector nor the convergence detector has fired.
    ///
    /// The agent's tool calls are varying and/or its responses are diverging,
    /// which indicates forward progress. Callers should continue the
    /// turn loop when they receive this variant.
    NoPattern,
}

/// Configuration for the [`DetectionManager`].
///
/// Groups all tunables for loop detection and convergence detection in a
/// single struct so consumers can construct a [`DetectionManager`] with
/// one call to [`DetectionManager::new_with_config`].
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
///   [`on_converge`](Self::on_converge).
///
/// # Example
///
/// ```rust,ignore
/// let config = DetectionConfig {
///     loop_threshold: 5,          // allow more repetition before flagging
///     convergence_threshold: 0.9, // looser similarity threshold
///     ..DetectionConfig::default()
/// };
/// let dm = DetectionManager::new_with_config(config);
/// ```
///
/// # See Also
///
/// - [`DetectionConfig::to_convergence_config`] — converts to [`ConvergenceConfig`].
/// - [`DetectionConfig::to_loop_detector_config`] — converts to [`LoopDetectorConfig`].
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Consecutive similar operations before declaring a loop. Default: **3**.
    ///
    /// Number of times the same `(tool, primary_param, result_hash)`
    /// signature must recur within the window before
    /// [`DetectedPattern::LoopDetected`] is reported. Lower it to catch loops
    /// sooner at the cost of more false positives.
    pub loop_threshold: usize,

    /// Repetitions triggering forced stop (0 = disabled). Default: **10**.
    ///
    /// When a single operation's repetition count reaches this value, the
    /// detector sets [`LoopStatus::should_stop`] so the caller halts the
    /// agent. Setting it to `0` disables forced stops entirely while still
    /// issuing warnings. The count is window-relative, not cumulative:
    /// [`max_history`](Self::max_history) eviction can reset it, so a
    /// stuck operation interleaved with enough distinct traffic may never
    /// reach the threshold — raise `max_history` if slow loops must be
    /// caught.
    pub stop_threshold: usize,

    /// Whether loop detection is enabled. Default: **true**.
    ///
    /// Master switch for the tool-loop subsystem. When `false`,
    /// [`DetectionManager::record_operation`] short-circuits and returns
    /// [`DetectedPattern::NoPattern`] without consulting the loop detector.
    pub enable_loop_detection: bool,

    /// Max operations kept in loop detector history. Default: **100**.
    ///
    /// Size of the sliding window the loop detector retains. A larger window
    /// catches slower loops that span many interleaved operations; a smaller
    /// one uses less memory and forgets stale patterns faster.
    pub max_history: usize,

    /// Jaccard similarity threshold for convergence (0.0–1.0). Default: **0.95**.
    ///
    /// Minimum Jaccard similarity a response must share with its predecessor
    /// to extend the convergence streak. Lower it to catch paraphrased
    /// repetition; raise it toward `1.0` to demand near-verbatim matches.
    pub convergence_threshold: f32,

    /// Consecutive similar responses for convergence. Default: **3**.
    ///
    /// Streak length of consecutive similar responses required before
    /// [`DetectedPattern::ConvergenceDetected`] is reported. Maps to the
    /// convergence detector's window size.
    pub convergence_count: usize,

    /// Whether convergence detection is enabled. Default: **true**.
    ///
    /// Master switch for the response-convergence subsystem. When `false`,
    /// [`DetectionManager::record_response`] short-circuits and returns
    /// [`DetectedPattern::NoPattern`] without consulting the convergence
    /// detector.
    pub enable_convergence_detection: bool,

    /// Action on convergence. Default: [`ConvergenceAction::Warn`] —
    /// text similarity is a heuristic with irreducible false positives, so
    /// its default punishment is not execution. Set
    /// [`ConvergenceAction::Stop`] to restore stop-on-converge.
    ///
    /// The [`ConvergenceAction`] forwarded to callers once convergence is
    /// detected — stop, warn, switch phase, ask the user, or compact the
    /// history. Drives how the agent responds to a detected stall.
    pub on_converge: ConvergenceAction,
}

impl Default for DetectionConfig {
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
        }
    }
}

impl DetectionConfig {
    /// Convert convergence settings into a [`ConvergenceConfig`].
    ///
    /// # Field Mapping
    ///
    /// | `DetectionConfig` field        | [`ConvergenceConfig`] field |
    /// |--------------------------------|-----------------------------|
    /// | `enable_convergence_detection` | `enabled`                   |
    /// | `convergence_count`            | `window_size`               |
    /// | `convergence_threshold`        | `similarity_threshold`      |
    /// | `on_converge`                  | `on_converge`               |
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
    /// # Field Mapping
    ///
    /// | `DetectionConfig` field | [`LoopDetectorConfig`] field |
    /// |-------------------------|------------------------------|
    /// | `max_history`           | `window_size`                |
    /// | `loop_threshold`        | `repetition_threshold`       |
    /// | `stop_threshold`        | `stop_threshold`             |
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

/// Cumulative statistics from the [`DetectionManager`].
///
/// Returned by [`DetectionManager::stats`] for observability and
/// debugging. All counters are monotonically increasing within a session
/// (until [`DetectionManager::reset`] is called).
///
/// [`Clone`] and [`Default`] — can be cheaply snapshot
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
    /// Tool-call operations recorded via `record_operation`. Reset by `reset`.
    ///
    /// Total count of operations fed into the loop detector since the last
    /// [`DetectionManager::reset`], giving the denominator for loop-rate
    /// calculations.
    pub turns_analyzed: usize,

    /// Subset of `turns_analyzed` that returned `LoopDetected`. Reset by `reset`.
    ///
    /// How many of the recorded operations resulted in a
    /// [`DetectedPattern::LoopDetected`] verdict, so callers can gauge how
    /// frequently the agent gets stuck in tool loops.
    pub loops_detected: usize,

    /// Times `record_response` returned `ConvergenceDetected`. Reset by `reset`.
    ///
    /// How many assistant responses triggered a
    /// [`DetectedPattern::ConvergenceDetected`] verdict, indicating how often
    /// the agent's replies collapsed into repetition.
    pub convergences_detected: usize,

    /// Mirrors `LoopStatus::repetition_count`. Triggers `LoopDetected` at `loop_threshold`. Reset by `reset`.
    ///
    /// Live snapshot of the most-repeated operation's count, updated on
    /// every [`record_operation`](DetectionManager::record_operation) —
    /// below the threshold (the build-up phase) as well as at the peak,
    /// so observers can genuinely watch a potential loop build up before
    /// it trips the [`DetectionConfig::loop_threshold`]. The streak
    /// recedes only when the window itself does: a differing result
    /// flushes a *warned* pattern's entries, and window eviction drops
    /// old counts.
    pub current_streak: usize,
}

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
/// | Constructor                                  | Use case                                  |
/// |----------------------------------------------|-------------------------------------------|
/// | [`DetectionManager::new`]                    | Quick start with defaults                 |
/// | [`DetectionManager::new_with_config`]        | Custom thresholds via [`DetectionConfig`] |
/// | [`DetectionManager::new_with_loop_detector`] | Inject a pre-built [`LoopDetector`]       |
/// | [`DetectionManager::new_with_signature`]     | Custom [`ToolSignature`] for JSON parsing |
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
/// delegate to the [`ConvergenceDetector`].
///
/// # Direct Access
///
/// For consumers that need direct access to the underlying detectors
/// (e.g., for compaction phases or response analysis phases), use
/// [`Self::loop_detector`] and [`Self::convergence_detector`].
///
/// # Lifecycle
///
/// The manager progresses through a simple per-turn cycle:
///
/// 1. **Construct** — `new()` or `new_with_config()`.
/// 2. **Feed** — call `record_operation(op)` for each tool invocation and
///    `record_response(text)` for each assistant reply.
/// 3. **Check** — call `check_loop()`, `check_convergence()`, or the
///    combined `check_current_pattern()` after each turn.
/// 4. **Observe** — call `stats()` at any time for cumulative counters.
/// 5. **Reset** — call `reset()` between tasks to clear all history.
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
    /// Configuration thresholds and feature flags.
    ///
    /// Holds the validated [`DetectionConfig`] (loop threshold, stop
    /// threshold, convergence settings) for the manager's lifetime. It
    /// is set in the constructor and never mutated, so threshold checks
    /// are a single field access with no locking.
    config: DetectionConfig,

    /// Loop detector shared across compaction and analysis phases.
    ///
    /// Held behind [`Arc`] so callers that need direct access to the
    /// tool-call window (for example, an observer that reports on
    /// repetition independently of the manager) can clone the handle
    /// rather than routing every query through the manager. The
    /// detector's own internal state is `Mutex`-guarded.
    loop_detector: Arc<LoopDetector>,

    /// Convergence detector guarded by interior mutability.
    ///
    /// [`ConvergenceDetector`] mutates its sliding window on every
    /// recorded response, so it is wrapped in a [`Mutex`] to keep the
    /// manager's public methods `&self`. This lets the manager be
    /// shared across threads without an outer `mut` reference.
    convergence_detector: Mutex<ConvergenceDetector>,

    /// Cumulative detection statistics.
    ///
    /// Counters for turns analysed, loops detected, and convergences
    /// observed since construction or the last
    /// [`reset`](DetectionManager::reset). Updated under the
    /// [`Mutex`] and snapshotted via
    /// [`stats`](DetectionManager::stats) for observability dashboards.
    stats: Mutex<DetectionStats>,
}

impl DetectionManager {
    /// Create a new detection manager with default configuration.
    ///
    /// Convenience wrapper around [`Self::new_with_config`] that passes
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
        Self::new_with_config(DetectionConfig::default())
    }

    /// Create a new detection manager with custom configuration.
    ///
    /// For tool-specific JSON parsing, use [`Self::new_with_signature`] or
    /// [`Self::new_with_loop_detector`] instead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = DetectionConfig {
    ///     loop_threshold: 5,
    ///     ..DetectionConfig::default()
    /// };
    /// let dm = DetectionManager::new_with_config(config);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn new_with_config(config: DetectionConfig) -> Result<Self, ConvergenceConfigError> {
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
    /// to inject it directly.
    ///
    /// This constructor **does not** use the `loop_threshold`, `stop_threshold`,
    /// or `max_history` fields from `config` — those are already baked into
    /// the provided `loop_detector`. Only the convergence-related fields
    /// are read from `config`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ld = LoopDetector::new(ld_config, my_signature);
    /// let dm = DetectionManager::new_with_loop_detector(config, ld);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the convergence configuration
    /// is invalid (e.g., threshold out of range, window too small).
    pub fn new_with_loop_detector(
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
    /// Useful when the consumer provides its own tool-aware signature
    /// (e.g., `DchToolSignature` for dch.sh tools that extracts
    /// `file_path` from JSON inputs).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let signature = Arc::new(MyToolSignature);
    /// let dm = DetectionManager::new_with_signature(signature);
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
    pub fn new_with_signature(
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

    /// Record an [`Operation`] for loop detection and return the current
    /// [`DetectedPattern`].
    ///
    /// Primary entry point for feeding data into the loop
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

        let status = self.loop_detector.check_loop();
        if status.should_stop {
            self.loop_detector.mark_warned(&status.repeated_operations);
        }
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
            guard.current_streak = self.loop_detector.max_operation_count();
            DetectedPattern::NoPattern
        }
    }

    /// Acknowledge a delivered loop warning so it is not rebuilt on
    /// every subsequent poll.
    ///
    /// Call this once you have actually *delivered* a non-stopping
    /// warning (logged it, surfaced it to a user, fed it to a monitor)
    /// — the acknowledgement marks the pattern's first repeated
    /// operation as warned, suppressing the same warning until the
    /// pattern changes. Pure queries
    /// ([`check_loop`](Self::check_loop),
    /// [`check_loop_pattern`](Self::check_loop_pattern)) never
    /// acknowledge implicitly; the stopping path marks automatically
    /// inside [`record_operation`](Self::record_operation).
    ///
    /// Acknowledging is orthogonal to *detection*: the repetition count
    /// and `should_stop` are unaffected, and a host cannot ack its way
    /// out of a stop. The pattern itself dissolves only through progress
    /// semantics — a differing result hash for the same operation
    /// flushes its window entries — so a host acknowledging while the
    /// agent alternates two outputs for one operation never sees the
    /// warning rebuilt, but also never sees a stop; that is the
    /// documented "different output = progress" rule at work.
    ///
    /// # Arguments
    ///
    /// * `repeated_operations` — the slice carried on the
    ///   [`LoopStatus`] whose warning was delivered.
    pub fn acknowledge_loop_warning(&self, repeated_operations: &[Operation]) {
        self.loop_detector.mark_warned(repeated_operations);
    }

    /// Returns the tool signature used for extracting primary parameters.
    ///
    /// Useful when callers need to construct [`Operation`]s directly using
    /// the configured [`ToolSignature`].
    pub fn signature(&self) -> &dyn ToolSignature {
        self.loop_detector.signature()
    }

    /// Record a tool call for loop detection by tool name and input hash.
    ///
    /// Creates an [`Operation`] from a `tool` name and `input_hash`, then
    /// delegates to [`Self::record_operation`]. Easiest way to
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
    /// Important for tools like `Read` where calling the same file
    /// is perfectly fine if the file content is changing (e.g., the agent
    /// is editing it).
    ///
    /// # How it works
    ///
    /// With result hashing, the detector distinguishes between a tool
    /// that returns the same output every time (a genuine loop) and one
    /// whose output changes between calls (the agent is making progress).
    /// For example, calling `Read("/foo.txt")` three times with the same
    /// result hash is flagged as a loop, but if the file is being edited
    /// between reads the result hashes will differ and no loop is
    /// reported.
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

    /// Query the [`LoopDetector`] for the current loop status.
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

    /// Build the convergence-only [`DetectedPattern`] as a pure query.
    ///
    /// The read side of [`record_response`](Self::record_response); used
    /// directly when loop detection is disabled so
    /// [`check_current_pattern`](Self::check_current_pattern) still reports
    /// convergence.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::detection::{DetectedPattern, DetectionManager};
    ///
    /// let manager = DetectionManager::new().unwrap();
    /// assert!(matches!(
    ///     manager.check_convergence_pattern(),
    ///     DetectedPattern::NoPattern
    /// ));
    /// ```
    #[must_use]
    pub fn check_convergence_pattern(&self) -> DetectedPattern {
        let convergence = self.check_convergence();
        if convergence.detected {
            return DetectedPattern::ConvergenceDetected {
                similarity: convergence.similarity_score,
                consecutive_count: convergence.consecutive_count,
            };
        }
        DetectedPattern::NoPattern
    }

    /// Build the loop-only [`DetectedPattern`] without recording anything.
    ///
    /// A pure query over the recorded window — the read side of
    /// [`record_operation`](Self::record_operation), which owns the single
    /// write per invocation. Callers that drive dispatch (pre-checks,
    /// idempotent re-derivations) use this so a step examined twice is
    /// counted once.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::detection::{DetectedPattern, DetectionManager, Operation};
    ///
    /// let manager = DetectionManager::new().unwrap();
    /// assert!(matches!(
    ///     manager.check_loop_pattern(),
    ///     DetectedPattern::NoPattern
    /// ));
    ///
    /// for _ in 0..3 {
    ///     manager.record_operation(Operation::new("Read", "/f.txt"));
    /// }
    /// assert!(matches!(
    ///     manager.check_loop_pattern(),
    ///     DetectedPattern::LoopDetected { repetitions, .. } if repetitions >= 3
    /// ));
    /// ```
    #[must_use]
    pub fn check_loop_pattern(&self) -> DetectedPattern {
        if !self.config.enable_loop_detection {
            return DetectedPattern::NoPattern;
        }
        let status = self.loop_detector.check_loop();
        if status.is_looping {
            return DetectedPattern::LoopDetected {
                repetitions: status.repetition_count,
                pattern_description: status
                    .repeated_operations
                    .first()
                    .map(|op| format!("{}({})", op.tool, op.primary_param))
                    .unwrap_or_default(),
            };
        }
        DetectedPattern::NoPattern
    }

    /// Consume a never-fired loop stop so the next run starts clean.
    ///
    /// A loop stop fires only at the next tool dispatch's pre-check: a run
    /// whose identical calls reach `stop_threshold` and then goes terminal
    /// (the model stops calling tools) never fires it, and the stale window
    /// would kill the *next* run's first dispatch with repetitions the new
    /// run never produced. The engine calls this at every run end: when the
    /// current pattern would stop — the detector's own `should_stop`
    /// rule (the single stop-rule source, zero sentinel included), the
    /// same rule the pre-dispatch check consults — the loop window is
    /// cleared, the
    /// same consumption a fired stop performs. Below the stop threshold
    /// nothing is cleared: sub-threshold history is harmless and may still
    /// warn. Convergence state is deliberately untouched — cross-run
    /// terminal-answer streaks are load-bearing for the opt-in
    /// [`Stop`](crate::detection::ConvergenceAction::Stop)/[`AskUser`](crate::detection::ConvergenceAction::AskUser)
    /// actions.
    pub fn consume_pending_loop_stop(&self) {
        if !self.config.enable_loop_detection {
            return;
        }
        if self.loop_detector.check_loop().should_stop {
            self.loop_detector.clear();
        }
    }

    /// Obtain a shared reference to the [`LoopDetector`].
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

    /// Obtain a shared reference to the [`ConvergenceDetector`].
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

    /// Record an assistant response for convergence detection.
    ///
    /// Forwards `response` to the [`ConvergenceDetector`] via
    /// [`ConvergenceDetector::add_response`], which computes Jaccard
    /// similarity against the previous response and updates the
    /// consecutive-similarity counter.
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
    /// # When called
    ///
    /// After each *terminal* assistant response — a response with no tool
    /// calls — typically by the framework's turn-processing loop. Acting
    /// turns (tool calls pending) do not feed convergence: their preamble
    /// text is by definition not a converged final answer.
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

    /// Query the [`ConvergenceDetector`] for the current
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

    /// Inspect the current detection state without recording new data.
    ///
    /// Checks both the loop detector and the convergence detector in
    /// sequence and returns the first non-`NoPattern` result (loop takes
    /// priority over convergence). **Read-only** operation —
    /// no statistics are updated and no new data is recorded.
    ///
    /// # Priority order
    ///
    /// Loop detection is checked first. If the loop detector reports a
    /// loop, `LoopDetected` is returned immediately. Only when no loop is
    /// found does the manager query the convergence detector. If neither
    /// detector has fired, `NoPattern` is returned. Loop takes priority
    /// because a tool-calling loop is a stronger and more urgent signal
    /// that the agent is stuck.
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
        if !self.config.enable_loop_detection {
            return self.check_convergence_pattern();
        }
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
        self.check_convergence_pattern()
    }

    /// Take a snapshot of the cumulative detection statistics.
    ///
    /// Clones the [`DetectionStats`] struct via the `Mutex`,
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
    /// - [`Self::new_with_config`] — the constructor that accepts a config.
    #[must_use]
    pub fn config(&self) -> &DetectionConfig {
        &self.config
    }

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

impl Default for DetectionManager {
    fn default() -> Self {
        let config = DetectionConfig::default();
        let loop_config = config.to_loop_detector_config();
        let loop_detector = Arc::new(LoopDetector::new(
            loop_config,
            Arc::new(super::loop_detector::NoOpToolSignature),
        ));
        let convergence_detector = Mutex::new(ConvergenceDetector::default());
        Self {
            config,
            loop_detector,
            convergence_detector,
            stats: Mutex::new(DetectionStats::default()),
        }
    }
}

/// Tests for [`DetectionManager`].
///
/// Verifies loop detection, convergence detection, result-aware detection,
/// reset behaviour, feature-flag toggling, stop thresholds, rich [`Operation`]
/// recording, and config mapping.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_pattern_initially() {
        let dm = DetectionManager::new().unwrap();
        assert!(matches!(
            dm.check_current_pattern(),
            DetectedPattern::NoPattern
        ));
    }

    #[test]
    fn test_loop_detection() {
        let dm = DetectionManager::new().unwrap();
        for _ in 0..5 {
            let result = dm.record_tool_call("read_file", 42);
            if matches!(result, DetectedPattern::LoopDetected { .. }) {
                return; // test passes
            }
        }
        panic!("Expected loop detection after 5 identical calls");
    }

    #[test]
    fn test_convergence_detection() {
        let dm = DetectionManager::new().unwrap();
        let response = "I am working on the task and making progress";
        for _ in 0..5 {
            let result = dm.record_response(response);
            if matches!(result, DetectedPattern::ConvergenceDetected { .. }) {
                return;
            }
        }
        panic!("Expected convergence detection after 5 identical responses");
    }

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

    #[test]
    fn test_no_detection_when_disabled() {
        let config = DetectionConfig {
            enable_loop_detection: false,
            enable_convergence_detection: false,
            ..Default::default()
        };
        let dm = DetectionManager::new_with_config(config).unwrap();
        for _ in 0..10 {
            let result = dm.record_tool_call("read_file", 42);
            assert!(matches!(result, DetectedPattern::NoPattern));
        }
    }

    #[test]
    fn test_loop_status_should_stop() {
        let config = DetectionConfig {
            loop_threshold: 3,
            stop_threshold: 5,
            ..Default::default()
        };
        let dm = DetectionManager::new_with_config(config).unwrap();

        for _ in 0..5 {
            dm.record_tool_call("read_file", 42);
        }

        let status = dm.check_loop();
        assert!(status.is_looping);
        assert!(status.should_stop);
        assert!(status.warning.is_some());
        assert!(status.warning.unwrap().contains("STOPPING"));
    }

    #[test]
    fn test_record_operation_with_operation_struct() {
        let dm = DetectionManager::new().unwrap();

        for _ in 0..5 {
            dm.record_operation(Operation::new("Read", "/test/file.txt"));
        }

        let status = dm.check_loop();
        assert!(status.is_looping);
        assert!(status.repetition_count >= 3);
    }

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

    #[test]
    fn test_access_loop_detector() {
        let dm = DetectionManager::new().unwrap();
        // Should be able to access the inner LoopDetector
        assert_eq!(dm.loop_detector().turn_count(), 0);
    }

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

    #[test]
    fn test_detection_config_default_has_no_max_response_history() {
        let config = DetectionConfig {
            loop_threshold: 3,
            stop_threshold: 5,
            ..Default::default()
        };
        assert_eq!(config.loop_threshold, 3);
        assert_eq!(config.stop_threshold, 5);
        let default = DetectionConfig::default();
        assert_eq!(default.loop_threshold, 3);
        assert_eq!(default.stop_threshold, 10);
        assert!(default.enable_loop_detection);
        assert_eq!(default.max_history, 100);
        assert!((default.convergence_threshold - 0.95).abs() < f32::EPSILON);
        assert_eq!(default.convergence_count, 3);
        assert!(default.enable_convergence_detection);
    }

    #[test]
    fn check_loop_pattern_respects_disabled_loop_detection() {
        let config = DetectionConfig {
            enable_loop_detection: false,
            ..DetectionConfig::default()
        };
        let manager = DetectionManager::new_with_config(config).unwrap();
        for _ in 0..12 {
            manager.record_operation(Operation::new("Read", "/f.txt"));
        }
        assert!(
            matches!(manager.check_loop_pattern(), DetectedPattern::NoPattern),
            "with loop detection disabled the pure query must never report a loop"
        );
    }

    #[test]
    fn check_loop_does_not_consume_the_warning() {
        let manager = DetectionManager::new().unwrap();
        for _ in 0..3 {
            manager.record_operation(Operation::new("Read", "/f.txt"));
        }
        let first = manager.check_loop();
        let second = manager.check_loop();
        assert!(first.warning.is_some(), "first poll must warn");
        assert!(
            second.warning.is_some(),
            "doc: any observability layer may poll without side effects"
        );
    }

    #[test]
    fn acknowledge_loop_warning_suppresses_rebuilds() {
        let manager = DetectionManager::new().unwrap();
        for _ in 0..3 {
            manager.record_operation(Operation::new("Read", "/f.txt"));
        }

        let delivered = manager.check_loop();
        let repeated = delivered.repeated_operations.clone();
        assert!(
            delivered.warning.is_some(),
            "precondition: a warning is pending"
        );

        manager.acknowledge_loop_warning(&repeated);

        let after = manager.check_loop();
        assert!(
            after.warning.is_none(),
            "an acknowledged warning must not rebuild while the pattern is unchanged"
        );
        assert!(
            after.is_looping,
            "the acknowledgement suppresses the warning only — the pattern itself is still live"
        );
    }

    #[test]
    fn current_streak_tracks_below_the_threshold_and_after_recovery() {
        let manager = DetectionManager::new_with_config(DetectionConfig {
            loop_threshold: 4,
            stop_threshold: 10,
            ..DetectionConfig::default()
        })
        .unwrap();
        manager.record_operation(Operation::new("Read", "/f.txt"));
        assert_eq!(
            manager.stats().current_streak,
            1,
            "the streak must be live in the build-up phase, not frozen at zero"
        );
        manager.record_operation(Operation::new("Read", "/f.txt"));
        assert_eq!(manager.stats().current_streak, 2);

        manager.record_operation(Operation::new("Read", "/other.txt"));
        assert_eq!(
            manager.stats().current_streak,
            2,
            "an unrelated operation leaves the most-repeated count alone — \
             the streak is the window's live maximum"
        );
    }

    #[test]
    fn consume_pending_loop_stop_clears_only_at_the_stop_threshold() {
        let manager = DetectionManager::new_with_config(DetectionConfig {
            loop_threshold: 2,
            stop_threshold: 5,
            ..DetectionConfig::default()
        })
        .unwrap();
        for _ in 0..3 {
            manager.record_operation(Operation::new("Read", "/f.txt"));
        }
        manager.consume_pending_loop_stop();
        let below = manager.check_loop_pattern();
        assert!(
            matches!(&below, DetectedPattern::LoopDetected { repetitions: 3, .. }),
            "below the stop threshold the window must survive — it may still warn: {below:?}"
        );

        for _ in 0..2 {
            manager.record_operation(Operation::new("Read", "/f.txt"));
        }
        manager.consume_pending_loop_stop();
        assert!(
            matches!(manager.check_loop_pattern(), DetectedPattern::NoPattern),
            "at the stop threshold the never-fired stop state must be consumed"
        );
    }
}
