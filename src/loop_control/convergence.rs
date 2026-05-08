//! Convergence detection for agent loops.
//!
//! This module detects when an agent's responses become semantically similar
//! across multiple consecutive turns, indicating that the agent has converged —
//! i.e., it is stuck repeating the same type of response without making
//! meaningful progress toward its goal.
//!
//! Loop detection (handled by the `loop_detector` module)
//! checks for repeated *operations* with the same result, while convergence
//! detection focuses on the *textual content* of the agent's responses, using
//! lightweight Jaccard similarity on word-level tokens.
//!
//! # How It Works
//!
//! The [`ConvergenceDetector`] maintains a sliding window of recent response
//! strings. Each new response is compared against every prior response in the
//! window using [`ConvergenceDetector::compute_similarity`]. When the number
//! of consecutive similar responses reaches the configured
//! [`ConvergenceConfig::window_size`], convergence is flagged and the
//! configured [`ConvergenceAction`] is returned.
//!
//! # Detection Flow
//!
//! ```text
//! add_response(text)
//!   ├─ disabled or empty? → return no_convergence()
//!   ├─ for each prev in window:
//!   │     compute Jaccard similarity(text, prev)
//!   │     if similarity >= threshold → increment consecutive_count
//!   │     else → reset consecutive_count to 1
//!   ├─ push text into window (evict oldest if full)
//!   └─ if consecutive_count >= window_size → return converged status
//! ```
//!
//! # Provided Types
//!
//! - **[`ConvergenceConfig`]** — Configuration: window size, similarity
//!   threshold, and action.
//! - **[`ConvergenceDetector`]** — The detector itself; feeds responses and
//!   checks convergence.
//! - **[`ConvergenceStatus`]** — Result of a convergence check, including
//!   similarity score and action.
//! - **[`ConvergenceAction`]** — What to do when convergence is detected.
//! - **[`ConvergenceConfigError`]** — Validation errors from invalid config.
//!
//! # Relationship to Other Detection Modules
//!
//! | Module                | What it detects         | Granularity |
//! |-----------------------|-------------------------|-------------|
//! | `convergence` (this)  | Similar *response text* | Turn-level  |
//! | `loop_detector`       | Repeated *operations*   | Tool-level  |
//! | `detection_manager`   | Orchestrates both       | Top-level   |
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::loop_control::convergence::{
//!     ConvergenceConfig, ConvergenceDetector, ConvergenceAction,
//! };
//!
//! let config = ConvergenceConfig {
//!     enabled: true,
//!     window_size: 3,
//!     similarity_threshold: 0.9,
//!     on_converge: ConvergenceAction::Warn,
//! };
//! let mut detector = ConvergenceDetector::new(config)?;
//!
//! // Feed identical responses to trigger convergence
//! let status = detector.add_response("I am working on the task.");
//! assert!(!status.detected);
//! let status = detector.add_response("I am working on the task.");
//! let status = detector.add_response("I am working on the task.");
//! assert!(status.detected); // 3 consecutive similar responses
//! # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
//! ```

use std::collections::HashSet;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// ===================================================
// ConvergenceAction
// ===================================================

/// Action to take when convergence is detected.
///
/// When the [`ConvergenceDetector`] determines that the agent's recent responses
/// are semantically similar, it returns a [`ConvergenceStatus`] carrying one
/// of these actions. The caller (typically the `DetectionManager`) interprets
/// the action to decide how to proceed.
///
/// The default is [`ConvergenceAction::Stop`], which halts the agent loop.
///
/// # Choosing an Action
///
/// | Scenario                       | Recommended Variant                             |
/// |--------------------------------|-------------------------------------------------|
/// | Unattended batch processing    | [`Stop`](ConvergenceAction::Stop)               |
/// | Long-running daemon            | [`Warn`](ConvergenceAction::Warn)               |
/// | Multi-phase pipeline           | [`SwitchPhase`](ConvergenceAction::SwitchPhase) |
/// | Interactive / REPL session     | [`AskUser`](ConvergenceAction::AskUser)         |
/// | Long context / token budget    | [`Compact`](ConvergenceAction::Compact)         |
///
/// # Serde
///
/// This enum derives [`Serialize`] and [`Deserialize`] so it can be
/// loaded from TOML/JSON configuration files. Variant names are serialized
/// in `snake_case` (e.g., `"switch_phase"`).
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::convergence::ConvergenceAction;
///
/// let action = ConvergenceAction::Warn;
/// match action {
///     ConvergenceAction::Stop => println!("Halting"),
///     ConvergenceAction::Warn => println!("Continuing with warning"),
///     ConvergenceAction::SwitchPhase => println!("Switching phase"),
///     ConvergenceAction::AskUser => println!("Asking user"),
///     ConvergenceAction::Compact => println!("Compacting history"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceAction {
    /// Stop the agent loop entirely.
    ///
    /// The agent has converged and is unlikely to make further progress.
    /// This is the safest default — the caller should report the situation
    /// to the user and await new instructions.
    ///
    /// The `DetectionManager` will
    /// propagate this as a terminal signal, causing the outer agent loop
    /// to exit cleanly.
    #[default]
    Stop,

    /// Continue execution but emit a warning log.
    ///
    /// Useful in long-running sessions where occasional convergence is
    /// acceptable but worth flagging for diagnostics. The agent loop
    /// continues running; the warning is surfaced through the
    /// logging pipeline.
    Warn,

    /// Switch to a different agent phase or strategy.
    ///
    /// The caller should transition the agent to an alternative processing
    /// phase (e.g., from "exploration" to "refinement") to break out of
    /// the converged pattern.
    ///
    /// The specific phase transition is left to the caller's discretion;
    /// the detector simply signals that the current strategy has stagnated.
    SwitchPhase,

    /// Ask the user for guidance before continuing.
    ///
    /// Pauses execution and prompts the user to provide direction. Best
    /// suited for interactive agent sessions where human oversight is
    /// available.
    ///
    /// When this action is returned, the agent loop should yield control
    /// back to the UI layer so the user can provide additional context
    /// or modify the task prompt.
    AskUser,

    /// Compact the conversation history to free context window space.
    ///
    /// Triggers a summarization or truncation of the conversation history
    /// to reduce token usage and break the converged pattern. The agent
    /// loop should invoke its compaction strategy (e.g., summarizing older
    /// turns, keeping only the most recent N exchanges, or pruning
    /// low-relevance messages).
    ///
    /// This is useful when convergence is caused by the agent repeatedly
    /// revisiting earlier context — compaction removes the redundant
    /// history that may be driving the repetition.
    Compact,
}

// ===================================================
// ConvergenceConfigError
// ===================================================

/// Error returned when [`ConvergenceConfig`] validation fails.
///
/// [`ConvergenceDetector::new`] validates the configuration before
/// constructing a detector. If any constraint is violated, it returns
/// one of these variants with enough context to produce an actionable
/// diagnostic message.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::convergence::{ConvergenceConfig, ConvergenceDetector, ConvergenceConfigError};
///
/// let bad_config = ConvergenceConfig {
///     window_size: 1,
///     ..Default::default()
/// };
/// let err = ConvergenceDetector::new(bad_config).unwrap_err();
/// assert!(matches!(err, ConvergenceConfigError::WindowTooSmall { actual: 1 }));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConvergenceConfigError {
    /// `window_size` is below the minimum of 2.
    ///
    /// Convergence requires at least one pair of consecutive responses,
    /// so a window of 1 (or 0) is meaningless.
    #[error("window_size must be at least 2, got {actual}")]
    WindowTooSmall {
        /// The invalid value that was provided.
        actual: usize,
    },

    /// `similarity_threshold` is outside the valid range `[0.0, 1.0]`.
    ///
    /// Jaccard similarity always produces a value in this range; a
    /// threshold outside it would never (or always) trigger.
    #[error("similarity_threshold must be in [0.0, 1.0], got {actual}")]
    ThresholdOutOfRange {
        /// The invalid value that was provided.
        actual: f32,
    },
}

// ===================================================
// ConvergenceConfig
// ===================================================

/// Configuration for convergence detection.
///
/// Controls the sensitivity and behavior of the [`ConvergenceDetector`].
/// Passed to [`ConvergenceDetector::new`] at construction time.
///
/// # Derives
///
/// [`Debug`] for logging, [`Clone`] for sharing config across detectors,
/// and [`Serialize`]/[`Deserialize`] for loading from config files.
///
/// # Defaults
///
/// | Field                 | Default  |
/// |-----------------------|----------|
/// | `enabled`             | `true`   |
/// | `window_size`         | `3`      |
/// | `similarity_threshold`| `0.95`   |
/// | `on_converge`         | `Stop`   |
///
/// # Tuning Guidelines
///
/// - **Lower `similarity_threshold`** (e.g., `0.8`) catches paraphrased
///   repetitions but may produce false positives when the agent naturally
///   revisits topics.
/// - **Larger `window_size`** (e.g., `5`) requires longer streaks before
///   declaring convergence, reducing sensitivity to brief repetitions.
/// - **Combine with [`ConvergenceAction::Warn`]** during development to
///   calibrate thresholds before switching to [`ConvergenceAction::Stop`].
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::convergence::{ConvergenceConfig, ConvergenceAction, ConvergenceDetector};
///
/// let config = ConvergenceConfig {
///     enabled: true,
///     window_size: 5,
///     similarity_threshold: 0.85,
///     on_converge: ConvergenceAction::Warn,
/// };
/// let detector = ConvergenceDetector::new(config)?;
/// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Whether convergence detection is active.
    ///
    /// When `false`, [`ConvergenceDetector::add_response`] and
    /// [`ConvergenceDetector::check_convergence`] return
    /// [`ConvergenceStatus::no_convergence`] immediately without performing
    /// any computation.
    ///
    /// Defaults to `true`.
    ///
    /// # When to Disable
    ///
    /// Set to `false` in performance-critical paths where similarity
    /// computation overhead is unacceptable, or when the agent's task
    /// domain makes convergence impossible by design.
    pub enabled: bool,

    /// Number of consecutive similar responses required to declare convergence.
    ///
    /// The detector maintains a sliding window of this size. When every
    /// response in the window exceeds [`ConvergenceConfig::similarity_threshold`]
    /// relative to its neighbours, convergence is detected.
    ///
    /// Defaults to `3`.
    ///
    /// # Constraints
    ///
    /// Must be at least `2`. Values below `2` are meaningless since
    /// convergence requires at least one pair of consecutive responses.
    pub window_size: usize,

    /// Similarity threshold (0.0–1.0) above which two responses are considered
    /// "similar".
    ///
    /// Computed via Jaccard similarity on word-level tokens (see
    /// [`ConvergenceDetector::compute_similarity`]). A value of `1.0` requires
    /// exact word-set matches; `0.0` considers any two non-empty strings similar.
    ///
    /// Defaults to `0.95`.
    ///
    /// # Invariant
    ///
    /// Must be in the range `0.0..=1.0`. Values outside this range will
    /// not panic but produce nonsensical results.
    pub similarity_threshold: f32,

    /// Action to take when convergence is detected.
    ///
    /// Carried through to [`ConvergenceStatus::action`] so the caller can
    /// respond appropriately. See [`ConvergenceAction`] for available options.
    ///
    /// Defaults to [`ConvergenceAction::Stop`].
    ///
    /// # Serde
    ///
    /// Uses `#[serde(default)]` so missing fields in TOML/JSON deserialize
    /// to [`ConvergenceAction::Stop`].
    #[serde(default)]
    pub on_converge: ConvergenceAction,
}

impl Default for ConvergenceConfig {
    /// Produce a configuration with production-ready defaults.
    ///
    /// The defaults are tuned for typical agent workloads:
    ///
    /// | Field | Default |
    /// |-------|---------|
    /// | [`enabled`](ConvergenceConfig::enabled) | `true` |
    /// | [`window_size`](ConvergenceConfig::window_size) | `3` |
    /// | [`similarity_threshold`](ConvergenceConfig::similarity_threshold) | `0.95` |
    /// | [`on_converge`](ConvergenceConfig::on_converge) | [`ConvergenceAction::Stop`] |
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::ConvergenceConfig;
    ///
    /// let config = ConvergenceConfig::default();
    /// assert!(config.enabled);
    /// assert_eq!(config.window_size, 3);
    /// ```
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 3,
            similarity_threshold: 0.95,
            on_converge: ConvergenceAction::Stop,
        }
    }
}

// ===================================================
// ConvergenceStatus
// ===================================================

/// Convergence status with details.
///
/// Returned by [`ConvergenceDetector::add_response`] and
/// [`ConvergenceDetector::check_convergence`] to communicate whether
/// convergence was detected and, if so, the evidence and recommended action.
///
/// # Construction
///
/// Use [`ConvergenceStatus::no_convergence`] to create a "no convergence"
/// sentinel. A "converged" status is constructed internally by the detector
/// when the [`ConvergenceConfig::window_size`] threshold is met.
///
/// # Derives
///
/// [`Debug`] for diagnostic output, [`Clone`] for sharing results, and
/// [`Default`] which produces the same value as [`no_convergence`](ConvergenceStatus::no_convergence).
///
/// # Fields Overview
///
/// ```text
/// detected ──────────── true if convergence reached
/// consecutive_count ─── how many similar responses in a row
/// similarity_score ──── highest Jaccard score among compared pairs
/// similar_responses ─── the response texts that triggered detection
/// action ────────────── what the caller should do (stop, warn, ...)
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::convergence::{ConvergenceConfig, ConvergenceDetector};
///
/// let config = ConvergenceConfig {
///     window_size: 3,
///     similarity_threshold: 0.5,
///     ..Default::default()
/// };
/// let mut detector = ConvergenceDetector::new(config)?;
///
/// let response = "same text";
/// let status = detector.add_response(response);
/// if status.detected {
///     println!("Converged after {} turns", status.consecutive_count);
///     println!("Similarity: {:.2}%", status.similarity_score * 100.0);
///     println!("Action: {:?}", status.action);
/// }
/// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct ConvergenceStatus {
    /// Whether convergence was detected.
    ///
    /// `true` when [`ConvergenceStatus::consecutive_count`] is at least
    /// [`ConvergenceConfig::window_size`], meaning the agent has produced
    /// enough consecutive similar responses to warrant action.
    ///
    /// When `false`, the other fields still contain valid data (streak count,
    /// latest similarity score, etc.) but should not be treated as a
    /// convergence signal.
    pub detected: bool,

    /// Number of consecutive similar responses observed so far.
    ///
    /// Resets to `1` whenever a response falls below
    /// [`ConvergenceConfig::similarity_threshold`]. Monotonically increases
    /// while responses remain similar.
    ///
    /// Compare against [`ConvergenceConfig::window_size`] to determine
    /// how close the detector is to declaring convergence.
    pub consecutive_count: usize,

    /// Highest Jaccard similarity score (0.0–1.0) among the compared pairs.
    ///
    /// Useful for diagnostics: a score of `0.99` means the responses are
    /// nearly identical; a lower score like `0.75` suggests the detector
    /// is barely triggering and the threshold may need tuning.
    ///
    /// Computed by [`ConvergenceDetector::compute_similarity`] during
    /// [`ConvergenceDetector::add_response`]. Always `0.0` when returned
    /// from [`ConvergenceDetector::check_convergence`] since no new
    /// comparison is performed.
    pub similarity_score: f32,

    /// The response strings that contributed to the convergence detection.
    ///
    /// Contains deduplicated copies of the recent responses that exceeded
    /// the similarity threshold. Useful for logging and user-facing reports.
    ///
    /// Cleared whenever a dissimilar response breaks the streak, so the
    /// collection only contains responses from the *current* streak.
    pub similar_responses: Vec<String>,

    /// The configured action to take, forwarded from
    /// [`ConvergenceConfig::on_converge`].
    ///
    /// The caller should inspect this field to decide how to respond (stop,
    /// warn, switch phase, or ask the user). See [`ConvergenceAction`] for
    /// the full list of variants and their semantics.
    pub action: ConvergenceAction,
}

impl ConvergenceStatus {
    /// Create a status indicating no convergence.
    ///
    /// Returns a [`ConvergenceStatus`] with `detected = false`,
    /// `consecutive_count = 0`, `similarity_score = 0.0`, and empty
    /// `similar_responses`. The `action` defaults to
    /// [`ConvergenceAction::Stop`] but is irrelevant since `detected` is
    /// `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::ConvergenceStatus;
    ///
    /// let status = ConvergenceStatus::no_convergence();
    /// assert!(!status.detected);
    /// assert_eq!(status.consecutive_count, 0);
    /// ```
    #[must_use]
    pub fn no_convergence() -> Self {
        Self {
            detected: false,
            consecutive_count: 0,
            similarity_score: 0.0,
            similar_responses: Vec::new(),
            action: ConvergenceAction::Stop,
        }
    }
}

// ===================================================
// ConvergenceDetector
// ===================================================

/// Detects when agent responses have converged (become semantically similar).
///
/// Maintains a sliding window of recent response strings and computes
/// pairwise Jaccard similarity to determine whether the agent is stuck
/// producing near-identical output. When enough consecutive responses
/// exceed the configured similarity threshold, convergence is reported.
///
/// # Construction
///
/// Prefer [`ConvergenceDetector::new`] with a custom [`ConvergenceConfig`],
/// or [`ConvergenceDetector::default_detector`] for sensible defaults.
///
/// # Internal Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────┐
/// │           ConvergenceDetector                   │
/// │                                                 │
/// │  config: ConvergenceConfig                      │
/// │  window: VecDeque<String>  ←─ sliding window    │
/// │  consecutive_count: usize                       │
/// │  similar_responses: Vec<String>                 │
/// │                                                 │
/// │  Methods:                                       │
/// │    add_response() → ConvergenceStatus           │
/// │    check_convergence() → ConvergenceStatus      │
/// │    compute_similarity() → f32                   │
/// │    clear()                                      │
/// └─────────────────────────────────────────────────┘
/// ```
///
/// # Derives
///
/// [`Debug`] for diagnostic output. Not [`Clone`] — the detector holds
/// mutable state and should be used as a single instance.
///
/// # Lifecycle
///
/// ```text
/// new(config)
///   → add_response(response)   [repeated, returns ConvergenceStatus]
///   → check_convergence()      [optional, peek without adding]
///   → clear()                  [reset for a new task]
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::convergence::ConvergenceDetector;
///
/// let mut detector = ConvergenceDetector::default_detector()?;
///
/// // Feed responses — identical content triggers convergence
/// let s1 = detector.add_response("Reading file contents...");
/// assert!(!s1.detected);
/// let s2 = detector.add_response("Reading file contents...");
/// let s3 = detector.add_response("Reading file contents...");
/// assert!(s3.detected);
///
/// // Reset for a fresh start
/// detector.clear();
/// assert!(detector.window().is_empty());
/// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug)]
pub struct ConvergenceDetector {
    /// Configuration controlling thresholds and actions.
    ///
    /// Set once at construction via [`ConvergenceDetector::new`] and read
    /// on every call to [`ConvergenceDetector::add_response`]. Immutable
    /// for the lifetime of the detector — create a new detector to change
    /// configuration.
    config: ConvergenceConfig,

    /// Sliding window of recent response strings.
    ///
    /// Bounded by [`ConvergenceConfig::window_size`]. When full, the oldest
    /// entry is evicted before a new one is appended. Accessed by the
    /// `DetectionManager` for
    /// direct inspection during testing.
    ///
    /// The deque is ordered chronologically: index `0` is the oldest
    /// response, and the last index is the most recent.
    ///
    /// # Capacity
    ///
    /// Pre-allocated to [`ConvergenceConfig::window_size`] at construction
    /// to avoid frequent heap allocations during steady-state operation.
    pub(super) window: VecDeque<String>,

    /// Number of consecutive responses that exceeded the similarity threshold.
    ///
    /// Resets to `1` when a response is sufficiently different from its
    /// predecessor. When this reaches [`ConvergenceConfig::window_size`],
    /// convergence is declared.
    ///
    /// This field is the internal counterpart of
    /// [`ConvergenceStatus::consecutive_count`].
    pub(super) consecutive_count: usize,

    /// Deduplicated collection of responses deemed "similar" so far.
    ///
    /// Grows as consecutive similar responses are observed and is cleared
    /// whenever a dissimilar response breaks the streak. Reported in
    /// [`ConvergenceStatus::similar_responses`] for diagnostics.
    ///
    /// Only unique strings are stored — duplicate responses are filtered
    /// to keep the collection compact.
    similar_responses: Vec<String>,
}

impl ConvergenceDetector {
    /// Create a new convergence detector with the given configuration.
    ///
    /// Validates the configuration before constructing the detector.
    /// Returns [`ConvergenceConfigError::WindowTooSmall`] if
    /// [`ConvergenceConfig::window_size`] is less than 2, or
    /// [`ConvergenceConfigError::ThresholdOutOfRange`] if
    /// [`ConvergenceConfig::similarity_threshold`] is outside `[0.0, 1.0]`.
    ///
    /// Pre-allocates the internal window to
    /// [`ConvergenceConfig::window_size`] capacity.
    ///
    /// The detector starts with an empty window and a zero consecutive
    /// count — no convergence can be detected until at least
    /// `window_size` responses have been added.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::{ConvergenceConfig, ConvergenceDetector};
    ///
    /// let config = ConvergenceConfig {
    ///     window_size: 5,
    ///     similarity_threshold: 0.85,
    ///     ..Default::default()
    /// };
    /// let detector = ConvergenceDetector::new(config)?;
    /// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError::WindowTooSmall`] if
    /// `window_size < 2`, or [`ConvergenceConfigError::ThresholdOutOfRange`]
    /// if `similarity_threshold` is outside `[0.0, 1.0]`.
    pub fn new(config: ConvergenceConfig) -> Result<Self, ConvergenceConfigError> {
        if config.window_size < 2 {
            return Err(ConvergenceConfigError::WindowTooSmall {
                actual: config.window_size,
            });
        }
        if !(0.0..=1.0).contains(&config.similarity_threshold) {
            return Err(ConvergenceConfigError::ThresholdOutOfRange {
                actual: config.similarity_threshold,
            });
        }
        let capacity = config.window_size;
        Ok(Self {
            config,
            window: VecDeque::with_capacity(capacity),
            consecutive_count: 0,
            similar_responses: Vec::new(),
        })
    }

    /// Create a detector with default configuration.
    ///
    /// Equivalent to `ConvergenceDetector::new(ConvergenceConfig::default())`.
    /// Uses a window of 3 and a similarity threshold of 0.95.
    ///
    /// Prefer this for quick prototyping; switch to
    /// [`ConvergenceDetector::new`] when you need custom thresholds.
    ///
    /// # Default Values
    ///
    /// | Setting               | Value  |
    /// |-----------------------|--------|
    /// | `enabled`             | `true` |
    /// | `window_size`         | `3`    |
    /// | `similarity_threshold`| `0.95` |
    /// | `on_converge`         | `Stop` |
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::ConvergenceDetector;
    ///
    /// let detector = ConvergenceDetector::default_detector()?;
    /// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the default config fails
    /// validation. This should never happen — the defaults are hard-coded
    /// to be valid.
    pub fn default_detector() -> Result<Self, ConvergenceConfigError> {
        Self::new(ConvergenceConfig::default())
    }

    /// Add a response and check for convergence in one step.
    ///
    /// This is the primary entry point. The response is compared against
    /// every prior response in the window. If any comparison exceeds
    /// [`ConvergenceConfig::similarity_threshold`], the consecutive count
    /// is incremented; otherwise it resets to `1`.
    ///
    /// When the consecutive count reaches [`ConvergenceConfig::window_size`],
    /// `detected` is set to `true` in the returned [`ConvergenceStatus`].
    ///
    /// # Returns
    ///
    /// A [`ConvergenceStatus`] indicating whether convergence was detected,
    /// the current streak count, the highest similarity score, and the
    /// recommended action.
    ///
    /// # Early Returns
    ///
    /// If [`ConvergenceConfig::enabled`] is `false` or `response` is empty,
    /// returns [`ConvergenceStatus::no_convergence`] immediately.
    ///
    /// # Side Effects
    ///
    /// Modifies the internal sliding window, consecutive counter, and
    /// similar-responses collection. If the window is full, the oldest
    /// entry is evicted.
    ///
    /// # Performance
    ///
    /// Each call performs `O(window_size)` similarity comparisons.
    /// For a typical window of 3, this is negligible; for large windows
    /// (e.g., 50), consider whether the overhead is acceptable.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::ConvergenceDetector;
    ///
    /// let mut detector = ConvergenceDetector::default_detector()?;
    /// let status = detector.add_response("Working on task...");
    /// assert!(!status.detected); // First response, no comparison possible
    /// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
    /// ```
    pub fn add_response(&mut self, response: &str) -> ConvergenceStatus {
        if !self.config.enabled || response.is_empty() {
            return ConvergenceStatus::no_convergence();
        }

        // Check similarity with previous responses
        let mut max_similarity = 0.0;
        for prev_response in &self.window {
            let similarity = Self::compute_similarity(response, prev_response);
            if similarity > max_similarity {
                max_similarity = similarity;
            }

            if similarity >= self.config.similarity_threshold {
                self.consecutive_count = self.consecutive_count.saturating_add(1);
                if !self.similar_responses.contains(&response.to_string()) {
                    self.similar_responses.push(response.to_string());
                }
            } else {
                // Response is different, reset counter
                self.consecutive_count = 1;
                self.similar_responses.clear();
                self.similar_responses.push(response.to_string());
            }
        }

        // Add to window
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(response.to_string());

        // Check if converged
        let detected = self.consecutive_count >= self.config.window_size;

        ConvergenceStatus {
            detected,
            consecutive_count: self.consecutive_count,
            similarity_score: max_similarity,
            similar_responses: self.similar_responses.clone(),
            action: self.config.on_converge,
        }
    }

    /// Check for convergence without adding a new response.
    ///
    /// Inspects the current window and consecutive count to determine
    /// whether convergence has already been reached. Does not modify
    /// internal state.
    ///
    /// # Returns
    ///
    /// A [`ConvergenceStatus`] reflecting the current detector state.
    /// Note that `similarity_score` is `0.0` because no new comparison
    /// is performed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::{ConvergenceConfig, ConvergenceDetector};
    ///
    /// let config = ConvergenceConfig {
    ///     window_size: 3,
    ///     similarity_threshold: 0.95,
    ///     ..Default::default()
    /// };
    /// let mut detector = ConvergenceDetector::new(config)?;
    /// detector.add_response("Response one about apples");
    /// detector.add_response("Response two about oranges");
    /// detector.add_response("Response three about bananas");
    /// let status = detector.check_convergence();
    /// assert!(!status.detected);
    /// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
    /// ```
    #[must_use]
    pub fn check_convergence(&self) -> ConvergenceStatus {
        if !self.config.enabled {
            return ConvergenceStatus::no_convergence();
        }

        if self.window.len() < self.config.window_size {
            return ConvergenceStatus::no_convergence();
        }

        ConvergenceStatus {
            detected: self.consecutive_count >= self.config.window_size,
            consecutive_count: self.consecutive_count,
            similarity_score: 0.0,
            similar_responses: self.similar_responses.clone(),
            action: self.config.on_converge,
        }
    }

    /// Clear all detection state.
    ///
    /// Resets the sliding window, consecutive count, and accumulated
    /// similar responses. Useful at the start of a new task or after
    /// the agent has been re-prompted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::convergence::ConvergenceDetector;
    ///
    /// let mut detector = ConvergenceDetector::default_detector()?;
    /// detector.add_response("task in progress");
    /// detector.clear();
    /// assert!(detector.window().is_empty());
    /// assert_eq!(detector.consecutive_count(), 0);
    /// # Ok::<(), loopctl::loop_control::convergence::ConvergenceConfigError>(())
    /// ```
    pub fn clear(&mut self) {
        self.window.clear();
        self.consecutive_count = 0;
        self.similar_responses.clear();
    }

    /// Get a reference to the current sliding window of responses.
    ///
    /// Primarily useful for testing and diagnostics — consumers can inspect
    /// what the detector has seen without modifying state.
    ///
    /// # Returns
    ///
    /// A reference to the internal [`VecDeque`] of response strings.
    /// The deque is ordered from oldest to newest.
    #[must_use]
    pub fn window(&self) -> &VecDeque<String> {
        &self.window
    }

    /// Get a reference to the current configuration.
    ///
    /// Allows consumers to inspect thresholds and the configured action
    /// without needing to store a separate copy of the config.
    ///
    /// # Returns
    ///
    /// A reference to the [`ConvergenceConfig`] this detector was created with.
    /// The config is immutable for the lifetime of the detector.
    #[must_use]
    pub fn config(&self) -> &ConvergenceConfig {
        &self.config
    }

    /// Get the number of consecutive similar responses.
    ///
    /// This is the current streak counter. When it reaches
    /// [`ConvergenceConfig::window_size`], convergence is detected.
    ///
    /// # Returns
    ///
    /// The current consecutive-similar count. Returns `0` if no responses
    /// have been added yet.
    #[must_use]
    pub fn consecutive_count(&self) -> usize {
        self.consecutive_count
    }

    /// Compute the Jaccard similarity between two strings.
    ///
    /// Normalizes both strings (lowercased, non-alphanumeric replaced with
    /// spaces), splits into word sets, then computes the ratio of
    /// intersection size to union size.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    ///
    /// Example:  A = {hello, world}
    ///           B = {hello, there}
    ///           Intersection = {hello}     → |1|
    ///           Union       = {hello, world, there} → |3|
    ///           Jaccard = 1/3 ≈ 0.33
    /// ```
    ///
    /// Returns `0.0` if either input is empty.
    ///
    /// # Normalization
    ///
    /// Before comparison, both strings are passed through
    /// the internal `normalize_text` helper to produce a canonical
    /// lowercased, alphanumeric-only form.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_similarity(a: &str, b: &str) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let a_norm = Self::normalize_text(a);
        let b_norm = Self::normalize_text(b);

        // Use Jaccard similarity on words
        let a_words: HashSet<&str> = a_norm.split_whitespace().collect();
        let b_words: HashSet<&str> = b_norm.split_whitespace().collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Normalize text for similarity comparison.
    ///
    /// Lowercases the input and replaces non-alphanumeric characters with
    /// spaces, producing a canonical form suitable for word-level Jaccard
    /// comparison via [`ConvergenceDetector::compute_similarity`].
    fn normalize_text(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detection() {
        let config = ConvergenceConfig {
            window_size: 3,
            similarity_threshold: 0.5,
            ..Default::default()
        };
        let mut detector = ConvergenceDetector::new(config).unwrap();
        let status1 = detector.add_response("alpha");
        assert!(!status1.detected, "first response: no comparison possible");
        assert_eq!(
            status1.consecutive_count, 0,
            "first response: window is empty"
        );

        let status2 = detector.add_response("beta");
        assert!(!status2.detected, "second response: different from first");
        assert_eq!(
            status2.consecutive_count, 1,
            "second response: new streak of 1"
        );

        let status3 = detector.add_response("beta");
        assert!(!status3.detected, "third response: streak of 2, need 3");
        assert_eq!(status3.consecutive_count, 2);

        let status4 = detector.add_response("beta");
        assert!(
            status4.detected,
            "fourth response: three consecutive 'beta'"
        );
        assert!(status4.consecutive_count >= 3);
    }

    #[test]
    fn test_no_convergence_different_responses() {
        let config = ConvergenceConfig {
            window_size: 3,
            similarity_threshold: 0.95,
            ..Default::default()
        };
        let mut detector = ConvergenceDetector::new(config).unwrap();
        detector.add_response("Response one about apples");
        detector.add_response("Response two about oranges");
        detector.add_response("Response three about bananas");
        let status = detector.check_convergence();
        assert!(!status.detected);
    }

    #[test]
    fn test_convergence_disabled() {
        let config = ConvergenceConfig {
            enabled: false,
            ..Default::default()
        };
        let mut detector = ConvergenceDetector::new(config).unwrap();
        detector.add_response("Same");
        detector.add_response("Same");
        detector.add_response("Same");
        let status = detector.check_convergence();
        assert!(!status.detected);
    }

    #[test]
    fn test_similarity_computation() {
        let sim = ConvergenceDetector::compute_similarity("hello world", "hello world");
        assert!(sim > 0.99);
        let sim = ConvergenceDetector::compute_similarity("hello world", "hello there");
        assert!(sim > 0.3 && sim < 0.7);
        let sim = ConvergenceDetector::compute_similarity("hello world", "goodbye moon");
        assert!(sim < 0.3);
    }

    #[test]
    fn test_clear_detector() {
        let mut detector = ConvergenceDetector::default_detector().unwrap();
        detector.add_response("Test");
        detector.add_response("Test");
        assert!(!detector.window.is_empty());
        detector.clear();
        assert!(detector.window.is_empty());
        assert_eq!(detector.consecutive_count, 0);
    }

    #[test]
    fn test_config_window_too_small() {
        let config = ConvergenceConfig {
            window_size: 1,
            ..Default::default()
        };
        let err = ConvergenceDetector::new(config).unwrap_err();
        assert!(matches!(
            err,
            ConvergenceConfigError::WindowTooSmall { actual: 1 }
        ));
        assert!(err.to_string().contains("at least 2"));
    }

    #[test]
    fn test_config_window_zero() {
        let config = ConvergenceConfig {
            window_size: 0,
            ..Default::default()
        };
        let err = ConvergenceDetector::new(config).unwrap_err();
        assert!(matches!(
            err,
            ConvergenceConfigError::WindowTooSmall { actual: 0 }
        ));
    }

    #[test]
    fn test_config_threshold_too_high() {
        let config = ConvergenceConfig {
            similarity_threshold: 1.5,
            ..Default::default()
        };
        let err = ConvergenceDetector::new(config).unwrap_err();
        assert!(matches!(
            err,
            ConvergenceConfigError::ThresholdOutOfRange { .. }
        ));
        assert!(err.to_string().contains("[0.0, 1.0]"));
    }

    #[test]
    fn test_config_threshold_negative() {
        let config = ConvergenceConfig {
            similarity_threshold: -0.1,
            ..Default::default()
        };
        let err = ConvergenceDetector::new(config).unwrap_err();
        assert!(matches!(
            err,
            ConvergenceConfigError::ThresholdOutOfRange { .. }
        ));
    }

    #[test]
    fn test_config_threshold_boundary_valid() {
        // 0.0 and 1.0 are valid boundary values
        let config_zero = ConvergenceConfig {
            similarity_threshold: 0.0,
            ..Default::default()
        };
        assert!(ConvergenceDetector::new(config_zero).is_ok());

        let config_one = ConvergenceConfig {
            similarity_threshold: 1.0,
            ..Default::default()
        };
        assert!(ConvergenceDetector::new(config_one).is_ok());
    }
}
