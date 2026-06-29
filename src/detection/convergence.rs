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
//! use loopctl::detection::convergence::{
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
//! # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
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
///
/// # Example
///
/// ```rust
/// use loopctl::detection::convergence::ConvergenceAction;
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
    /// Default action — halt the agent loop and report the situation
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
    /// the detector signals that the current strategy has stagnated.
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
    /// Useful when convergence is caused by the agent repeatedly
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
/// use loopctl::detection::convergence::{ConvergenceConfig, ConvergenceDetector, ConvergenceConfigError};
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
    WindowTooSmall { actual: usize },

    /// `similarity_threshold` is outside the valid range `[0.0, 1.0]`.
    ///
    /// Jaccard similarity always produces a value in this range; a
    /// threshold outside it would never (or always) trigger.
    #[error("similarity_threshold must be in [0.0, 1.0], got {actual}")]
    ThresholdOutOfRange { actual: f32 },
}

// ===================================================
// ConvergenceConfig
// ===================================================

/// Configuration for convergence detection.
///
/// Controls the sensitivity and behavior of the [`ConvergenceDetector`].
/// Passed to [`ConvergenceDetector::new`] at construction time.
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
/// use loopctl::detection::convergence::{ConvergenceConfig, ConvergenceAction, ConvergenceDetector};
///
/// let config = ConvergenceConfig {
///     enabled: true,
///     window_size: 5,
///     similarity_threshold: 0.85,
///     on_converge: ConvergenceAction::Warn,
/// };
/// let detector = ConvergenceDetector::new(config)?;
/// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Whether convergence detection is active. Defaults to `true`.
    pub enabled: bool,
    /// Consecutive similar responses required to declare convergence. Must be ≥ 2. Defaults to `3`.
    pub window_size: usize,
    /// Jaccard similarity threshold (0.0–1.0). Defaults to `0.95`.
    pub similarity_threshold: f32,
    /// Action on convergence. Defaults to [`ConvergenceAction::Stop`].
    #[serde(default)]
    pub on_converge: ConvergenceAction,
}

impl Default for ConvergenceConfig {
    /// Produce a configuration with production-ready defaults.
    ///
    /// The defaults are tuned for typical agent workloads:
    ///
    /// | Field                                                             | Default                     |
    /// |-------------------------------------------------------------------|-----------------------------|
    /// | [`enabled`](ConvergenceConfig::enabled)                           | `true`                      |
    /// | [`window_size`](ConvergenceConfig::window_size)                   | `3`                         |
    /// | [`similarity_threshold`](ConvergenceConfig::similarity_threshold) | `0.95`                      |
    /// | [`on_converge`](ConvergenceConfig::on_converge)                   | [`ConvergenceAction::Stop`] |
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::detection::convergence::ConvergenceConfig;
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
/// sentinel. A "converged" status is returned by the detector
/// when the [`ConvergenceConfig::window_size`] threshold is met.
///
/// # Example
///
/// ```rust
/// use loopctl::detection::convergence::{ConvergenceConfig, ConvergenceDetector};
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
/// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct ConvergenceStatus {
    /// `true` when `consecutive_count >= window_size`. Other fields still valid when `false`.
    pub detected: bool,
    /// Resets to `1` when similarity falls below threshold.
    pub consecutive_count: usize,
    /// Highest Jaccard similarity (0.0–1.0). `0.0` when no comparison was made.
    pub similarity_score: f32,
    /// Responses that exceeded the similarity threshold. Cleared on dissimilar response.
    pub similar_responses: Vec<String>,
    /// Forwarded from [`ConvergenceConfig::on_converge`]; see [`ConvergenceAction`].
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
    /// use loopctl::detection::convergence::ConvergenceStatus;
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
/// # Example
///
/// ```rust
/// use loopctl::detection::convergence::ConvergenceDetector;
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
/// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
/// ```
#[derive(Debug)]
pub struct ConvergenceDetector {
    /// Immutable config set at construction.
    config: ConvergenceConfig,
    /// Bounded by `window_size`, ordered oldest→newest.
    pub(super) window: VecDeque<String>,
    /// Resets to `1` on dissimilar response; convergence at `window_size`.
    pub(super) consecutive_count: usize,
    /// Deduplicated similar responses; cleared when streak breaks.
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
    /// The detector starts with an empty window and a zero consecutive
    /// count — no convergence can be detected until at least
    /// `window_size` responses have been added.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::detection::convergence::{ConvergenceConfig, ConvergenceDetector};
    ///
    /// let config = ConvergenceConfig {
    ///     window_size: 5,
    ///     similarity_threshold: 0.85,
    ///     ..Default::default()
    /// };
    /// let detector = ConvergenceDetector::new(config)?;
    /// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError::WindowTooSmall`] if `window_size < 2`,
    /// or [`ConvergenceConfigError::ThresholdOutOfRange`]
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
    /// use loopctl::detection::convergence::ConvergenceDetector;
    ///
    /// let detector = ConvergenceDetector::default_detector()?;
    /// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ConvergenceConfigError`] if the default config fails
    /// validation.
    pub fn default_detector() -> Result<Self, ConvergenceConfigError> {
        Self::new(ConvergenceConfig::default())
    }

    /// Add a response and check for convergence in one step.
    ///
    /// Primary entry point. The response is compared against
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
    /// # Example
    ///
    /// ```rust
    /// use loopctl::detection::convergence::ConvergenceDetector;
    ///
    /// let mut detector = ConvergenceDetector::default_detector()?;
    /// let status = detector.add_response("Working on task...");
    /// assert!(!status.detected); // First response, no comparison possible
    /// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
    /// ```
    pub fn add_response(&mut self, response: &str) -> ConvergenceStatus {
        if !self.config.enabled || response.is_empty() {
            return ConvergenceStatus::no_convergence();
        }

        let mut max_similarity = 0.0;
        let mut any_similar = false;
        for prev_response in &self.window {
            let similarity = Self::compute_similarity(response, prev_response);
            if similarity > max_similarity {
                max_similarity = similarity;
            }

            if similarity >= self.config.similarity_threshold {
                any_similar = true;
                if !self.similar_responses.contains(&response.to_string()) {
                    self.similar_responses.push(response.to_string());
                }
            }
        }

        // Update consecutive count once per add_response call
        if self.window.is_empty() {
            self.consecutive_count = 1;
            self.similar_responses.push(response.to_string());
        } else if any_similar {
            self.consecutive_count = self.consecutive_count.saturating_add(1);
        } else {
            self.consecutive_count = 1;
            self.similar_responses.clear();
            self.similar_responses.push(response.to_string());
        }

        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(response.to_string());

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
    /// detector state.
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
    /// use loopctl::detection::convergence::{ConvergenceConfig, ConvergenceDetector};
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
    /// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
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
    /// use loopctl::detection::convergence::ConvergenceDetector;
    ///
    /// let mut detector = ConvergenceDetector::default_detector()?;
    /// detector.add_response("task in progress");
    /// detector.clear();
    /// assert!(detector.window().is_empty());
    /// assert_eq!(detector.consecutive_count(), 0);
    /// # Ok::<(), loopctl::detection::convergence::ConvergenceConfigError>(())
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
    /// A reference to the [`VecDeque`] of response strings.
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
    /// Current streak counter. When it reaches
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
    /// Returns `0.0` if either input is empty.
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

    fn normalize_text(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect()
    }
}

impl Default for ConvergenceDetector {
    /// Produce a [`ConvergenceDetector`] with the default [`ConvergenceConfig`].
    ///
    /// This impl exists so that parent types (e.g. [`DetectionManager`]) can
    /// derive or delegate `Default` without going through a fallible constructor.
    ///
    /// [`DetectionManager`]: crate::detection::manager::DetectionManager
    fn default() -> Self {
        let config = ConvergenceConfig::default();
        let window_capacity = config.window_size;
        Self {
            config,
            window: VecDeque::with_capacity(window_capacity),
            consecutive_count: 0,
            similar_responses: Vec::new(),
        }
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
            status1.consecutive_count, 1,
            "first response: starts a streak of 1"
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
