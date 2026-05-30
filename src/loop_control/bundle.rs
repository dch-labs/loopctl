//! Manager bundle — aggregate struct for the agent's infrastructure managers.
//!
//! The [`ManagerBundle`] owns the managers that the framework builder produces,
//! and provides a single `reset_all()` call to reinitialise every manager at the
//! start of a new task.
//!
//! # Provided Managers
//!
//! - **[`FallbackManager`]** — Circuit breaker for automatic API model fallback.
//! - **[`DetectionManager`]** — Loop and convergence detection orchestrator.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::loop_control::bundle::ManagerBundle;
//! use loopctl::loop_control::fallback::FallbackManager;
//!
//! // Create with defaults
//! let bundle = ManagerBundle::new();
//!
//! // Or with a custom fallback manager
//! let bundle = ManagerBundle::new()
//!     .with_fallback(FallbackManager::for_model("llm-70b"));
//!
//! // Reset all managers at the start of a new task
//! bundle.reset_all();
//! ```

use crate::loop_control::detection::DetectionManager;
use crate::loop_control::fallback::FallbackManager;

/// Bundle of framework-provided manager instances.
///
/// This struct is constructed by the `AgentBuilder` and passed to production
/// `AgentLoopBuilder` via `into_raw_parts()`. Each manager in the bundle
/// handles a specific cross-cutting concern (fallback, detection, etc.) so
/// that the agent core can remain focused on turn processing.
///
/// # Construction
///
/// Prefer [`ManagerBundle::new`] for defaults or the builder-style
/// [`ManagerBundle::with_fallback`] to override individual managers.
///
/// ```
/// # use loopctl::loop_control::bundle::ManagerBundle;
/// let bundle = ManagerBundle::new();
/// assert!(bundle.fallback.active_model().is_none());
/// ```
pub struct ManagerBundle {
    /// Circuit breaker for API model fallback.
    ///
    /// Manages automatic failover from a primary LLM model to a fallback
    /// model when consecutive API failures exceed a threshold. Created
    /// with [`FallbackManager::default`] which uses 3 failures to trip
    /// and 2 successes to recover.
    ///
    /// See [`FallbackManager`] for the full state-machine documentation.
    pub fallback: FallbackManager,

    /// Loop and convergence detection orchestrator.
    ///
    /// Wraps [`LoopDetector`](crate::loop_control::loop_detector::LoopDetector)
    /// and [`ConvergenceDetector`](crate::loop_control::convergence::ConvergenceDetector)
    /// behind a single interface. Records tool operations and agent
    /// responses, then checks for repeated patterns or semantically
    /// similar outputs.
    ///
    /// See [`DetectionManager`] for the full API documentation.
    pub detection: DetectionManager,
}

impl ManagerBundle {
    /// Create a new bundle with default managers.
    ///
    /// Each manager is initialized with its default configuration.
    /// Use the `with_*` builder methods to override individual managers.
    ///
    /// Called when building a new agent to seed the bundle with defaults.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::loop_control::bundle::ManagerBundle;
    /// let bundle = ManagerBundle::new();
    /// assert!(bundle.fallback.active_model().is_none());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallback: FallbackManager::default(),
            detection: DetectionManager::default(),
        }
    }

    /// Replace the fallback manager with a custom instance.
    ///
    /// Consumes the current bundle and returns a new one with the given
    /// [`FallbackManager`]. This is the builder-style way to override the
    /// default fallback behavior.
    ///
    /// Typically used by callers who need custom fallback thresholds or a
    /// pre-configured model name, via
    /// `AgentBuilder::with_fallback`.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::loop_control::bundle::ManagerBundle;
    /// # use loopctl::loop_control::fallback::FallbackManager;
    /// let fallback = FallbackManager::for_model("llm-70b");
    /// let bundle = ManagerBundle::new().with_fallback(fallback);
    /// assert_eq!(bundle.fallback.active_model().as_deref(), Some("llm-70b"));
    /// ```
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackManager) -> Self {
        self.fallback = fallback;
        self
    }

    /// Replace the detection manager with a custom instance.
    ///
    /// Consumes the current bundle and returns a new one with the given
    /// [`DetectionManager`]. Use this when you need custom loop or
    /// convergence thresholds, e.g. via
    /// [`DetectionManager::with_config`].
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::loop_control::bundle::ManagerBundle;
    /// # use loopctl::loop_control::detection::{DetectionManager, DetectionConfig};
    /// let config = DetectionConfig {
    ///     loop_threshold: 5,
    ///     ..Default::default()
    /// };
    /// let detection = DetectionManager::with_config(config).unwrap();
    /// let bundle = ManagerBundle::new().with_detection(detection);
    /// assert_eq!(bundle.detection.config().loop_threshold, 5);
    /// ```
    #[must_use]
    pub fn with_detection(mut self, detection: DetectionManager) -> Self {
        self.detection = detection;
        self
    }

    /// Reset all managers to their initial state.
    ///
    /// Delegates to each manager's `reset()` method. Typically called at the
    /// start of a new agent task or session to clear any accumulated state
    /// from a previous run.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::loop_control::bundle::ManagerBundle;
    /// let bundle = ManagerBundle::new();
    /// // ... after a session ...
    /// bundle.reset_all();
    /// // All managers are back to their initial state
    /// ```
    pub fn reset_all(&self) {
        self.fallback.reset();
        self.detection.reset();
    }
}

impl Default for ManagerBundle {
    /// Produce a [`ManagerBundle`] with default managers.
    ///
    /// Equivalent to [`ManagerBundle::new`]. Exists so that generic code
    /// can write `ManagerBundle::default()` or derive `Default` on parent
    /// structs that contain a [`ManagerBundle`].
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::loop_control::bundle::ManagerBundle;
    /// let bundle = ManagerBundle::default();
    /// // identical to ManagerBundle::new()
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_control::detection::{DetectedPattern, DetectionManager};

    #[test]
    fn test_bundle_default() {
        let bundle = ManagerBundle::default();
        assert!(bundle.fallback.active_model().is_none());
    }

    #[test]
    fn test_bundle_with_custom_fallback() {
        let fallback = FallbackManager::for_model("my-model");
        let bundle = ManagerBundle::new().with_fallback(fallback);
        assert_eq!(bundle.fallback.active_model().as_deref(), Some("my-model"));
    }

    #[test]
    fn test_reset_all() {
        let bundle = ManagerBundle::new();
        bundle.reset_all();
        // No panic
    }

    #[test]
    fn test_bundle_contains_detection_manager() {
        let bundle = ManagerBundle::new();
        // Default detection config values
        assert_eq!(bundle.detection.config().loop_threshold, 3);
        assert_eq!(bundle.detection.config().stop_threshold, 10);
    }

    #[test]
    fn test_bundle_with_custom_detection() {
        use crate::loop_control::detection::DetectionConfig;

        let config = DetectionConfig {
            loop_threshold: 7,
            stop_threshold: 20,
            ..Default::default()
        };
        let detection = DetectionManager::with_config(config).unwrap();
        let bundle = ManagerBundle::new().with_detection(detection);
        assert_eq!(bundle.detection.config().loop_threshold, 7);
        assert_eq!(bundle.detection.config().stop_threshold, 20);
        // Fallback is still the default
        assert!(bundle.fallback.active_model().is_none());
    }

    #[test]
    fn test_reset_all_clears_detection() {
        let bundle = ManagerBundle::new();
        // Record a tool call to populate detection state
        let _ = bundle.detection.record_tool_call("Read", 12345);
        // Reset should clear it
        bundle.reset_all();
        // After reset, recording the same call should not trigger detection
        let pattern = bundle.detection.record_tool_call("Read", 12345);
        assert!(matches!(pattern, DetectedPattern::NoPattern));
    }
}
