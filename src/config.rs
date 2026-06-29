//! Agent session configuration.
//!
//! Defines [`LoopConfig`] — the configuration struct that controls agent
//! session parameters such as turn limits, model selection, and context
//! window size.

use uuid::Uuid;

/// Configuration for an agent session.
///
/// Holds generic agent configuration fields that apply to every session
/// regardless of the specific agent type. Domain-specific configuration
/// (e.g., ITR engine settings, `ToolShield` rules, fallback model chains)
/// should live in production-specific config types that embed or wrap
/// this struct.
///
/// # Construction
///
/// Use [`LoopConfig::default`] for sensible defaults or override individual
/// fields through the builder via
/// `LoopBuilder::with_config`.
///
/// ```
/// use loopctl::config::LoopConfig;
///
/// let config = LoopConfig {
///     max_turns: 50,
///     model: "default".to_string(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoopConfig {
    /// Unique session identifier (random UUID v4).
    pub session_id: Uuid,
    /// Model identifier (e.g. `"default"`). Passed to the API client on each request.
    pub model: String,
    /// Optional system prompt override. `None` means the agent core decides.
    pub system_prompt: Option<String>,
    /// Maximum number of turns before forcing completion. Defaults to `200`.
    pub max_turns: usize,
    /// Maximum tokens for each API response. Defaults to `16_384`.
    pub max_tokens: u32,
    /// Context window size in tokens. Must match the actual window of [`model`](LoopConfig::model). Defaults to `200_000`.
    pub context_window: u64,
    /// Threshold to trigger auto-compaction (0.0–1.0). Defaults to `0.80`.
    pub compact_threshold: f64,
    /// Whether auto-compaction is enabled. Defaults to `true`.
    pub auto_compact: bool,
}

impl Default for LoopConfig {
    /// Produce a configuration with production-ready defaults.
    ///
    /// | Field | Default |
    /// |-------|---------|
    /// | [`session_id`](LoopConfig::session_id) | Random UUID v4 |
    /// | [`model`](LoopConfig::model) | `"default"` |
    /// | [`system_prompt`](LoopConfig::system_prompt) | `None` |
    /// | [`max_turns`](LoopConfig::max_turns) | `200` |
    /// | [`max_tokens`](LoopConfig::max_tokens) | `16_384` |
    /// | [`context_window`](LoopConfig::context_window) | `200_000` |
    /// | [`compact_threshold`](LoopConfig::compact_threshold) | `0.80` |
    /// | [`auto_compact`](LoopConfig::auto_compact) | `true` |
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::config::LoopConfig;
    ///
    /// let config = LoopConfig::default();
    /// assert_eq!(config.max_turns, 200);
    /// assert_eq!(config.model, "default");
    /// ```
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            model: "default".to_string(),
            system_prompt: None,
            max_turns: 200,
            max_tokens: 16_384,
            context_window: 200_000,
            compact_threshold: 0.80,
            auto_compact: true,
        }
    }
}

impl LoopConfig {
    /// Validate the configuration fields.
    ///
    /// Checks that:
    /// - `compact_threshold` is in the range `[0.0, 1.0]` (not `NaN`).
    /// - `max_turns` is greater than zero.
    /// - `context_window` is greater than zero.
    /// - `max_tokens` is greater than zero.
    /// - `model` is not empty.
    ///
    /// # Errors
    ///
    /// Returns a [`String`] describing the first invalid field, or `Ok(())`
    /// if all fields are valid.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::config::LoopConfig;
    ///
    /// let config = LoopConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad = LoopConfig { compact_threshold: 1.5, ..config };
    /// assert!(bad.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.max_turns == 0 {
            return Err("max_turns must be greater than 0".to_string());
        }
        if self.context_window == 0 {
            return Err("context_window must be greater than 0".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
        if self.model.is_empty() {
            return Err("model must not be empty".to_string());
        }
        if self.compact_threshold.is_nan()
            || self.compact_threshold < 0.0
            || self.compact_threshold > 1.0
        {
            return Err(format!(
                "compact_threshold must be in [0.0, 1.0], got {}",
                self.compact_threshold
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_default_config_is_ok() {
        let config = LoopConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_max_tokens() {
        let config = LoopConfig {
            max_tokens: 0,
            ..LoopConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.contains("max_tokens"),
            "error should mention max_tokens: {err}"
        );
    }

    #[test]
    fn validate_accepts_one_max_tokens() {
        let config = LoopConfig {
            max_tokens: 1,
            ..LoopConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_model() {
        let config = LoopConfig {
            model: String::new(),
            ..LoopConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("model"), "error should mention model: {err}");
    }

    #[test]
    fn validate_accepts_nonempty_model() {
        let config = LoopConfig {
            model: "gpt-4".to_string(),
            ..LoopConfig::default()
        };
        assert!(config.validate().is_ok());
    }
}
