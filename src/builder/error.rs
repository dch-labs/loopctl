//! Builder error types — failures that can occur during agent construction.
//!
//! The [`BuildError`] enum enumerates every way an `AgentBuilder`
//! call to `build()` can fail. Because the builder uses
//! **type-state generics** to enforce the presence of an `AgentCore`
//! at compile time, some classes of errors (e.g. "no core set") can only be triggered
//! through `into_raw_parts()` or dynamic construction paths.
//!
//! # Provided Types
//!
//! - **[`BuildError`]** — Enum of all construction-time validation failures.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::builder::BuildError;
//!
//! // Create a missing-dependency error with a helpful hint:
//! let err = BuildError::missing_dependency(
//!     "ApiClient",
//!     "Call .with_api_client() before building.",
//! );
//!
//! // Create a feature-conflict error:
//! let err = BuildError::feature_conflict("fast_mode", "safe_mode");
//! ```

/// Errors that can occur during agent construction.
///
/// Returned by `AgentBuilder::build()` when validation of the accumulated
/// builder state fails. Each variant captures enough context to produce an
/// actionable error message (e.g. which features conflict, how many
/// observers exceeded the limit).
///
/// # Validation invariants
///
/// The builder checks the following at build time:
///
/// - A core implementation must be present (enforced statically in most cases,
///   but checked dynamically for edge cases).
/// - Observer count must not exceed the framework's internal cap (32).
/// - User-provided validation closures, if any, must pass.
///
/// # Example
///
/// ```rust
/// use loopctl::builder::BuildError;
///
/// // Each variant can be constructed directly or via convenience methods
/// let err = BuildError::MissingCore;
/// assert!(err.to_string().contains("AgentCore"));
///
/// let err = BuildError::feature_conflict("fast_mode", "safe_mode");
/// assert!(matches!(err, BuildError::FeatureConflict { .. }));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum BuildError {
    /// No agent core was provided.
    ///
    /// The builder requires an `AgentCore` implementation
    /// before it can produce a runnable agent. In the standard type-state flow this
    /// is caught at compile time (the `NoCore` type parameter lacks the
    /// `CoreSet` bound), but this variant covers dynamic
    /// construction paths.
    ///
    /// **Fix:** Call `.with_core()` before `.build()`.
    #[error("AgentBuilder requires an AgentCore implementation. Call .with_core() before .build()")]
    MissingCore,

    /// Configuration is invalid.
    ///
    /// Wraps a human-readable description of what makes the current
    /// `LoopConfig` invalid — for example, a
    /// `max_turns` value of zero or a malformed model identifier.
    ///
    /// **Fix:** Adjust the config passed to `.with_config()`.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// A required dependency is missing.
    ///
    /// Some features or managers need additional components to function. This
    /// variant carries the dependency name and a hint about how to provide it.
    ///
    /// **Fix:** Add the missing dependency before calling `.build()`.
    #[error("Missing dependency: {name}. {hint}")]
    MissingDependency {
        /// Name of the missing dependency (e.g. `"ApiClient"`, `"ToolRegistry"`).
        name: String,
        /// Human-readable hint describing how to supply the dependency.
        hint: String,
    },

    /// Feature conflict — two mutually exclusive features enabled.
    ///
    /// Some features are logically incompatible (e.g. a "fast mode" and a
    /// "safe mode" that trade off against each other). This variant names both
    /// conflicting features so the caller can disable one.
    ///
    /// **Fix:** Disable one of the conflicting features via `.disable_feature()`.
    #[error("Feature conflict: {feature_a} and {feature_b} are mutually exclusive")]
    FeatureConflict {
        /// Name of the first conflicting feature.
        ///
        /// Set via [`BuildError::feature_conflict()`]. Matches the
        /// `Feature::name()` of the feature.
        feature_a: String,
        /// Name of the second conflicting feature.
        ///
        /// Set via [`BuildError::feature_conflict()`]. Matches the
        /// `Feature::name()` of the feature.
        feature_b: String,
    },

    /// Too many observers registered.
    ///
    /// The framework caps observers (currently 32) to prevent unbounded
    /// memory growth and O(n) dispatch overhead on every lifecycle event.
    ///
    /// **Fix:** Remove observers or consolidate them into a single multiplexing
    /// observer.
    #[error("Too many observers: {count} registered, maximum is {max}")]
    TooManyObservers {
        /// Current number of observers that were registered.
        ///
        /// Will always be greater than [`max`](Self::TooManyObservers.max).
        count: usize,
        /// Maximum number of observers allowed.
        ///
        /// Defaults to the framework's internal cap (32).
        max: usize,
    },

    /// A custom validation error from user-provided validation logic.
    ///
    /// Production builders can register arbitrary validation closures via
    /// extension traits. When such a closure returns `Err`, this variant
    /// wraps the returned message.
    ///
    /// **Fix:** Address the specific validation failure described in the message.
    #[error("Validation failed: {0}")]
    Validation(String),
}

impl BuildError {
    /// Create a missing-dependency error with a name and a hint.
    ///
    /// Convenience constructor for the [`MissingDependency`](BuildError::MissingDependency)
    /// variant. Use this instead of constructing the tuple variant directly so the
    /// caller's intent is self-documenting.
    ///
    /// # Arguments
    ///
    /// - `name` — Name of the missing dependency
    ///   (e.g. `"ApiClient"`).
    /// - `hint` — Description of how to supply the dependency
    ///   (e.g. `"Call .with_api_client() before building."`).
    ///
    /// Accepts any `Into<String>` so callers can pass `&str`, `String`,
    /// or string literals interchangeably.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::BuildError;
    ///
    /// let err = BuildError::missing_dependency(
    ///     "ToolRegistry",
    ///     "Call .with_tool_registry() with a configured registry.",
    /// );
    /// assert!(matches!(err, BuildError::MissingDependency { .. }));
    /// ```
    #[must_use]
    pub fn missing_dependency(name: impl Into<String>, hint: impl Into<String>) -> Self {
        Self::MissingDependency {
            name: name.into(),
            hint: hint.into(),
        }
    }

    /// Create a feature-conflict error naming two incompatible features.
    ///
    /// Convenience constructor for the [`FeatureConflict`](BuildError::FeatureConflict)
    /// variant. Accepts any `Into<String>` so callers can pass `&str`, `String`,
    /// or the result of `Feature::name()`.
    ///
    /// # Arguments
    ///
    /// - `a` — Name of the first feature (order does not matter).
    /// - `b` — Name of the second feature.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::builder::BuildError;
    ///
    /// let err = BuildError::feature_conflict("fast_mode", "safe_mode");
    /// if let BuildError::FeatureConflict { feature_a, feature_b } = &err {
    ///     assert_eq!(feature_a, "fast_mode");
    ///     assert_eq!(feature_b, "safe_mode");
    /// }
    /// ```
    #[must_use]
    pub fn feature_conflict(a: impl Into<String>, b: impl Into<String>) -> Self {
        Self::FeatureConflict {
            feature_a: a.into(),
            feature_b: b.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_core_display() {
        let err = BuildError::MissingCore;
        assert!(err.to_string().contains("AgentCore"));
        assert!(err.to_string().contains(".with_core()"));
    }

    #[test]
    fn test_invalid_config() {
        let err = BuildError::InvalidConfig("max_turns must be > 0".into());
        assert!(err.to_string().contains("max_turns"));
    }

    #[test]
    fn test_missing_dependency_constructor() {
        let err = BuildError::missing_dependency("ApiClient", "Call .with_api_client()");
        assert!(matches!(err, BuildError::MissingDependency { .. }));
        let msg = err.to_string();
        assert!(msg.contains("ApiClient"));
        assert!(msg.contains(".with_api_client()"));

        // Also accepts String
        let err = BuildError::missing_dependency(
            String::from("ToolRegistry"),
            format!("Call .with_tool_registry()"),
        );
        assert!(matches!(err, BuildError::MissingDependency { .. }));
    }

    #[test]
    fn test_feature_conflict_constructor() {
        let err = BuildError::feature_conflict("fast_mode", "safe_mode");
        if let BuildError::FeatureConflict {
            feature_a,
            feature_b,
        } = &err
        {
            assert_eq!(feature_a, "fast_mode");
            assert_eq!(feature_b, "safe_mode");
        } else {
            panic!("expected FeatureConflict variant");
        }
        let msg = err.to_string();
        assert!(msg.contains("fast_mode"));
        assert!(msg.contains("safe_mode"));
        assert!(msg.contains("mutually exclusive"));
    }

    #[test]
    fn test_too_many_observers() {
        let err = BuildError::TooManyObservers { count: 40, max: 32 };
        let msg = err.to_string();
        assert!(msg.contains('4') && msg.contains('0') && msg.contains("40"));
        assert!(msg.contains("32"));
    }

    #[test]
    fn test_validation() {
        let err = BuildError::Validation("custom check failed".into());
        assert!(err.to_string().contains("custom check failed"));
    }
}
