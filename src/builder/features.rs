//! Builder feature flags for agent construction.
//!
//! The [`Feature`] enum defines named feature flags that control builder
//! behaviour, and [`FeatureSet`] manages an enabled subset. Features are
//! enabled via `AgentBuilder::enable_feature()` and queried at build time
//! or during agent execution.
//!
//! # Provided Types
//!
//! - **[`Feature`]** — Named feature flag enum with 15 variants.
//! - **[`FeatureSet`]** — Compact set of enabled features with `O(1)` lookup.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::builder::features::{Feature, FeatureSet};
//!
//! let mut fs = FeatureSet::new();
//! fs.enable(Feature::ToolShield);
//! assert!(fs.is_enabled(Feature::ToolShield));
//!
//! fs.disable(Feature::ToolShield);
//! assert!(!fs.is_enabled(Feature::ToolShield));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// ==================================================
// Feature
// ==================================================

/// A named feature flag that controls agent builder behaviour.
///
/// Each variant corresponds to a discrete piece of functionality that
/// can be enabled or disabled independently. Features are collected
/// into a [`FeatureSet`] by the builder and queried at build time or
/// during agent execution.
///
/// # Example
///
/// ```rust
/// use loopctl::builder::features::Feature;
///
/// let feature = Feature::ToolShield;
/// assert_eq!(feature.name(), "ToolShield");
/// assert!(!feature.description().is_empty());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Feature {
    /// Sandbox tool execution behind a permission boundary.
    ///
    /// When enabled, tool invocations are wrapped in a sandbox policy
    /// that restricts file-system access, network calls, and
    /// environment-variable reads to an allow-list.
    ToolShield,

    /// Detect and break out of repetitive agent loops.
    ///
    /// Monitors the conversation history for cycles (repeated tool
    /// calls, identical assistant messages, or oscillating tool
    /// arguments) and injects a corrective prompt or halts the loop.
    LoopDetection,

    /// Enable short-term conversation memory management.
    ///
    /// Maintains a sliding window of past turns and can summarise
    /// older context to keep the prompt within the model's context
    /// window.
    MemoryManagement,

    /// Validate tool inputs against their declared JSON Schema.
    ///
    /// Before dispatching a tool call, the framework validates the
    /// arguments against the tool's `parameters` schema. Invalid
    /// inputs are rejected before the tool is invoked.
    ToolInputValidation,

    /// Record detailed execution traces for debugging.
    ///
    /// Captures per-turn timing, token counts, tool call arguments
    /// and results, and other diagnostic data into a trace log that
    /// can be inspected after the agent run completes.
    ExecutionTracing,

    /// Automatic retry of transient API failures.
    ///
    /// When the LLM API returns a retryable error (rate limit, timeout,
    /// server error), the framework retries with exponential back-off
    /// up to a configurable maximum.
    AutoRetry,

    /// Convergence detection — halt when the agent reaches a fixpoint.
    ///
    /// Compares consecutive assistant outputs for semantic equivalence
    /// and stops the loop when the agent's response stops changing.
    ConvergenceDetection,

    /// Fallback to a secondary model on persistent failure.
    ///
    /// If the primary model fails repeatedly, the framework switches
    /// to a configured fallback model for subsequent turns.
    ModelFallback,

    /// Prompt injection detection.
    ///
    /// Scans user messages for common injection patterns (jailbreak
    /// prompts, role-reset attempts, etc.) and either rejects them or
    /// wraps them in a safety preamble.
    PromptInjectionDetection,

    /// Stream partial results to observers in real time.
    ///
    /// When enabled, the framework emits `on_stream_delta` events to
    /// registered observers as the model generates tokens, rather than
    /// waiting for the full response.
    Streaming,

    /// Cache identical tool calls within a single run.
    ///
    /// If a tool is called with the same arguments multiple times
    /// within one agent loop, the cached result is returned instead
    /// of re-executing the tool.
    ToolCallCaching,

    /// Rate-limit tool invocations per tool name.
    ///
    /// Prevents a runaway agent from overwhelming a tool (e.g. a web
    /// search API) by capping the number of invocations per tool per
    /// run.
    ToolRateLimiting,

    /// Emit structured events to an external audit log.
    ///
    /// Each tool call, API request, and observer callback is logged
    /// as a structured JSON event suitable for ingestion by an audit
    /// pipeline.
    AuditLogging,

    /// Enable cost tracking for API usage.
    ///
    /// Tracks token counts per model and estimates cost based on a
    /// configured price table. The accumulated cost is available after
    /// the agent run completes.
    CostTracking,

    /// Parallel tool execution where safe.
    ///
    /// When multiple tool calls are requested in a single assistant
    /// turn and none have data dependencies, the framework dispatches
    /// them concurrently.
    ParallelToolExecution,
}

impl Feature {
    /// Return all defined features as a slice.
    ///
    /// Useful for iterating over the full feature set or building a
    /// [`FeatureSet::all()`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::Feature;
    ///
    /// let all = Feature::all();
    /// assert!(!all.is_empty());
    /// for feature in all {
    ///     println!("- {}: {}", feature.name(), feature.description());
    /// }
    /// ```
    #[must_use]
    pub const fn all() -> &'static [Feature] {
        &[
            Feature::ToolShield,
            Feature::LoopDetection,
            Feature::MemoryManagement,
            Feature::ToolInputValidation,
            Feature::ExecutionTracing,
            Feature::AutoRetry,
            Feature::ConvergenceDetection,
            Feature::ModelFallback,
            Feature::PromptInjectionDetection,
            Feature::Streaming,
            Feature::ToolCallCaching,
            Feature::ToolRateLimiting,
            Feature::AuditLogging,
            Feature::CostTracking,
            Feature::ParallelToolExecution,
        ]
    }

    /// The short, `PascalCase` name of this feature.
    ///
    /// Matches the variant name exactly. Used as a key for
    /// serialisation and display.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::Feature;
    ///
    /// assert_eq!(Feature::ToolShield.name(), "ToolShield");
    /// assert_eq!(Feature::AutoRetry.name(), "AutoRetry");
    /// ```
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Feature::ToolShield => "ToolShield",
            Feature::LoopDetection => "LoopDetection",
            Feature::MemoryManagement => "MemoryManagement",
            Feature::ToolInputValidation => "ToolInputValidation",
            Feature::ExecutionTracing => "ExecutionTracing",
            Feature::AutoRetry => "AutoRetry",
            Feature::ConvergenceDetection => "ConvergenceDetection",
            Feature::ModelFallback => "ModelFallback",
            Feature::PromptInjectionDetection => "PromptInjectionDetection",
            Feature::Streaming => "Streaming",
            Feature::ToolCallCaching => "ToolCallCaching",
            Feature::ToolRateLimiting => "ToolRateLimiting",
            Feature::AuditLogging => "AuditLogging",
            Feature::CostTracking => "CostTracking",
            Feature::ParallelToolExecution => "ParallelToolExecution",
        }
    }

    /// A one-line human-readable description of what this feature does.
    ///
    /// Suitable for display in `--help` output, UI tooltips, or debug logs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::Feature;
    ///
    /// let desc = Feature::LoopDetection.description();
    /// assert!(!desc.is_empty());
    /// ```
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Feature::ToolShield => "Sandbox tool execution behind a permission boundary",
            Feature::LoopDetection => "Detect and break out of repetitive agent loops",
            Feature::MemoryManagement => "Enable short-term conversation memory management",
            Feature::ToolInputValidation => {
                "Validate tool inputs against their declared JSON Schema"
            }
            Feature::ExecutionTracing => "Record detailed execution traces for debugging",
            Feature::AutoRetry => "Automatic retry of transient API failures",
            Feature::ConvergenceDetection => "Halt when the agent reaches a fixpoint",
            Feature::ModelFallback => "Fallback to a secondary model on persistent failure",
            Feature::PromptInjectionDetection => "Scan user messages for injection patterns",
            Feature::Streaming => "Stream partial results to observers in real time",
            Feature::ToolCallCaching => "Cache identical tool calls within a single run",
            Feature::ToolRateLimiting => "Rate-limit tool invocations per tool name",
            Feature::AuditLogging => "Emit structured events to an external audit log",
            Feature::CostTracking => "Track token counts and estimate API cost",
            Feature::ParallelToolExecution => "Execute independent tools in parallel",
        }
    }

    /// Check whether this feature conflicts with another.
    ///
    /// Some features are mutually exclusive (e.g. `Streaming` and
    /// `ToolCallCaching` may interfere because streaming bypasses the
    /// caching layer).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::Feature;
    ///
    /// // Streaming conflicts with ToolCallCaching
    /// assert!(Feature::Streaming.conflicts_with(Feature::ToolCallCaching));
    /// // ToolShield does not conflict with LoopDetection
    /// assert!(!Feature::ToolShield.conflicts_with(Feature::LoopDetection));
    /// ```
    #[must_use]
    pub const fn conflicts_with(self, other: Feature) -> bool {
        matches!(
            (self, other),
            (Feature::Streaming, Feature::ToolCallCaching)
                | (Feature::ToolCallCaching, Feature::Streaming)
        )
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ==================================================
// FeatureSet
// ==================================================

/// A compact set of enabled [`Feature`] flags.
///
/// `FeatureSet` tracks which features are enabled for a given agent
/// build. It provides average `O(1)` enable/disable/query operations backed
/// by an internal `HashSet`.
///
/// # Construction
///
/// ```rust
/// use loopctl::builder::features::{Feature, FeatureSet};
///
/// // Empty set
/// let mut fs = FeatureSet::new();
///
/// // Enable individual features
/// fs.enable(Feature::ToolShield);
/// fs.enable(Feature::LoopDetection);
///
/// // Or start with all features enabled
/// let all = FeatureSet::all();
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureSet {
    /// Enabled features stored as a set for O(1) lookup.
    enabled: std::collections::HashSet<Feature>,
}

impl FeatureSet {
    /// Create an empty feature set (no features enabled).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let fs = FeatureSet::new();
    /// assert!(!fs.is_enabled(Feature::ToolShield));
    /// assert_eq!(fs.len(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: std::collections::HashSet::new(),
        }
    }

    /// Create a feature set with all features enabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let fs = FeatureSet::all();
    /// for feature in Feature::all() {
    ///     assert!(fs.is_enabled(*feature));
    /// }
    /// ```
    #[must_use]
    pub fn all() -> Self {
        let mut set = Self::new();
        for feature in Feature::all() {
            set.enable(*feature);
        }
        set
    }

    /// Enable a feature.
    ///
    /// Idempotent — enabling an already-enabled feature is a no-op.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::new();
    /// fs.enable(Feature::ToolShield);
    /// assert!(fs.is_enabled(Feature::ToolShield));
    /// ```
    pub fn enable(&mut self, feature: Feature) {
        self.enabled.insert(feature);
    }

    /// Disable a feature.
    ///
    /// Idempotent — disabling a feature that is not enabled is a no-op.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::all();
    /// fs.disable(Feature::ToolShield);
    /// assert!(!fs.is_enabled(Feature::ToolShield));
    /// ```
    pub fn disable(&mut self, feature: Feature) {
        self.enabled.remove(&feature);
    }

    /// Check whether a feature is enabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::new();
    /// fs.enable(Feature::AutoRetry);
    /// assert!(fs.is_enabled(Feature::AutoRetry));
    /// assert!(!fs.is_enabled(Feature::ToolShield));
    /// ```
    #[must_use]
    pub fn is_enabled(&self, feature: Feature) -> bool {
        self.enabled.contains(&feature)
    }

    /// Return the number of enabled features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::new();
    /// assert_eq!(fs.len(), 0);
    /// fs.enable(Feature::ToolShield);
    /// fs.enable(Feature::AutoRetry);
    /// assert_eq!(fs.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.enabled.len()
    }

    /// Check whether no features are enabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::FeatureSet;
    ///
    /// let fs = FeatureSet::new();
    /// assert!(fs.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.enabled.is_empty()
    }

    /// Return an iterator over the enabled features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::new();
    /// fs.enable(Feature::ToolShield);
    /// fs.enable(Feature::AutoRetry);
    ///
    /// let names: Vec<&str> = fs.iter().map(|f| f.name()).collect();
    /// assert!(names.contains(&"ToolShield"));
    /// assert!(names.contains(&"AutoRetry"));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Feature> {
        self.enabled.iter()
    }

    /// Check whether any enabled feature conflicts with the given feature.
    ///
    /// Returns the first conflicting enabled feature, if any.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::features::{Feature, FeatureSet};
    ///
    /// let mut fs = FeatureSet::new();
    /// fs.enable(Feature::Streaming);
    /// let conflict = fs.find_conflict(Feature::ToolCallCaching);
    /// assert_eq!(conflict, Some(Feature::Streaming));
    /// ```
    #[must_use]
    pub fn find_conflict(&self, feature: Feature) -> Option<Feature> {
        self.enabled
            .iter()
            .find(|enabled| enabled.conflicts_with(feature))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name_not_empty() {
        for feature in Feature::all() {
            assert!(!feature.name().is_empty());
        }
    }

    #[test]
    fn test_feature_description_not_empty() {
        for feature in Feature::all() {
            assert!(!feature.description().is_empty());
        }
    }

    #[test]
    fn test_feature_display() {
        assert_eq!(format!("{}", Feature::ToolShield), "ToolShield");
        assert_eq!(format!("{}", Feature::AutoRetry), "AutoRetry");
    }

    #[test]
    fn test_feature_conflicts() {
        assert!(Feature::Streaming.conflicts_with(Feature::ToolCallCaching));
        assert!(Feature::ToolCallCaching.conflicts_with(Feature::Streaming));
        assert!(!Feature::ToolShield.conflicts_with(Feature::LoopDetection));
        assert!(!Feature::AutoRetry.conflicts_with(Feature::ModelFallback));
    }

    #[test]
    fn test_feature_set_new() {
        let fs = FeatureSet::new();
        assert!(fs.is_empty());
        assert_eq!(fs.len(), 0);
        for feature in Feature::all() {
            assert!(!fs.is_enabled(*feature));
        }
    }

    #[test]
    fn test_feature_set_enable_disable() {
        let mut fs = FeatureSet::new();
        fs.enable(Feature::ToolShield);
        assert!(fs.is_enabled(Feature::ToolShield));
        assert_eq!(fs.len(), 1);

        fs.disable(Feature::ToolShield);
        assert!(!fs.is_enabled(Feature::ToolShield));
        assert!(fs.is_empty());
    }

    #[test]
    fn test_feature_set_all() {
        let fs = FeatureSet::all();
        assert_eq!(fs.len(), Feature::all().len());
        for feature in Feature::all() {
            assert!(fs.is_enabled(*feature));
        }
    }

    #[test]
    fn test_feature_names() {
        for feature in Feature::all() {
            assert!(!feature.name().is_empty());
        }
    }

    #[test]
    fn test_feature_set_iter() {
        let mut fs = FeatureSet::new();
        fs.enable(Feature::ToolShield);
        fs.enable(Feature::AutoRetry);

        let features: Vec<Feature> = fs.iter().copied().collect();
        assert_eq!(features.len(), 2);
        assert!(features.contains(&Feature::ToolShield));
        assert!(features.contains(&Feature::AutoRetry));
    }

    #[test]
    fn test_feature_set_find_conflict() {
        let mut fs = FeatureSet::new();
        fs.enable(Feature::Streaming);
        assert_eq!(
            fs.find_conflict(Feature::ToolCallCaching),
            Some(Feature::Streaming)
        );
        assert_eq!(fs.find_conflict(Feature::ToolShield), None);
    }

    #[test]
    fn test_feature_set_serialization() {
        let mut fs = FeatureSet::new();
        fs.enable(Feature::ToolShield);
        fs.enable(Feature::AutoRetry);

        let json = serde_json::to_string(&fs).unwrap();
        let back: FeatureSet = serde_json::from_str(&json).unwrap();
        assert!(back.is_enabled(Feature::ToolShield));
        assert!(back.is_enabled(Feature::AutoRetry));
        assert!(!back.is_enabled(Feature::LoopDetection));
    }
}
