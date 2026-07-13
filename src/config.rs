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
/// let config = LoopConfig::default()
///     .with_max_turns(50)
///     .with_model("default");
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct LoopConfig {
    /// Unique session identifier.
    ///
    /// A random UUID v4 generated on construction. Used to tag tool contexts,
    /// session save/load paths, and observer events so a host app can correlate
    /// turns across logs and persisted state. Two configs with the same
    /// `session_id` are considered the same logical session.
    pub session_id: Uuid,

    /// Model identifier passed to the API client on each request.
    ///
    /// Provider-specific (e.g. `"gpt-4o"`, `"claude-3.5-sonnet"`). The client
    /// may override this at runtime via
    /// [`ApiClient::set_model`](crate::api::ApiClient::set_model) when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips. Must be
    /// non-empty ([`validate`](LoopConfig::validate) rejects whitespace-only).
    pub model: String,

    /// Optional system prompt override.
    ///
    /// When `None`, the agent core assembles its own system prompt from the
    /// registered tools' `system_prompt()` contributions. When `Some(text)`,
    /// that text replaces the default entirely. Set this to inject a custom
    /// persona or instruction set.
    pub system_prompt: Option<String>,

    /// Maximum number of turns before forcing completion.
    ///
    /// A safety cap: the agent loop halts with
    /// [`LoopError::MaxTurnsExceeded`](crate::error::LoopError::MaxTurnsExceeded)
    /// if it reaches this count without the model emitting a stop. Defaults to
    /// `200`. Must be at least `1` ([`validate`](LoopConfig::validate)).
    pub max_turns: usize,

    /// Maximum tokens for each API response.
    ///
    /// Sent to the provider as the `max_tokens` parameter on every request,
    /// capping the length of a single model response. Defaults to `16_384`.
    /// Must be at least `1` ([`validate`](LoopConfig::validate)).
    pub max_tokens: u32,

    /// Context window size in tokens.
    ///
    /// Must match the actual window of [`model`](LoopConfig::model). Used by
    /// the compaction subsystem to decide when conversation history exceeds
    /// the budget (see [`compact_threshold`](LoopConfig::compact_threshold)).
    /// Defaults to `200_000`. Must be at least `1`
    /// ([`validate`](LoopConfig::validate)).
    pub context_window: u64,

    /// Threshold to trigger auto-compaction, as a fraction of the context
    /// window (0.0–1.0).
    ///
    /// When the estimated token usage of the conversation exceeds
    /// `compact_threshold * context_window`, the compactor runs before the next
    /// turn to avoid an overflow. Defaults to `0.80`. Must be finite and in
    /// `[0.0, 1.0]` ([`validate`](LoopConfig::validate)).
    pub compact_threshold: f64,
    /// Whether auto-compaction is enabled.
    ///
    /// When `true` (the default), the compactor runs automatically when
    /// [`compact_threshold`](LoopConfig::compact_threshold) is reached. When
    /// `false`, the agent never auto-compacts — the host app must manage
    /// context size manually (useful for tests or fixed-length sessions).
    pub auto_compact: bool,

    /// How independent tool calls within a single turn are dispatched.
    ///
    /// Defaults to [`ParallelMode::Sequential`] (one at a time).
    /// Set [`ParallelMode::Parallel`] to run independent,
    /// concurrency-safe calls concurrently up to
    /// [`max_concurrency`](ParallelDispatchConfig::max_concurrency).
    ///
    /// Parallel mode changes observer/detection event ordering (PRE events are
    /// batched, then POST events batched, rather than strictly paired per
    /// call) and adds finer-grained cancellation. See
    /// [`ParallelDispatchConfig`] and the dispatch module docs.
    pub parallel_tool_dispatch: ParallelDispatchConfig,
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
    /// | [`parallel_tool_dispatch`](LoopConfig::parallel_tool_dispatch) | `Sequential`, `max_concurrency: 8` |
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
            parallel_tool_dispatch: ParallelDispatchConfig::default(),
        }
    }
}

impl LoopConfig {
    /// Set the session ID.
    #[must_use]
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = session_id;
        self
    }

    /// Set the model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Set the maximum number of turns.
    #[must_use]
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the maximum tokens per API response.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the context window size in tokens.
    #[must_use]
    pub fn with_context_window(mut self, context_window: u64) -> Self {
        self.context_window = context_window;
        self
    }

    /// Set the auto-compaction threshold (0.0–1.0).
    #[must_use]
    pub fn with_compact_threshold(mut self, compact_threshold: f64) -> Self {
        self.compact_threshold = compact_threshold;
        self
    }

    /// Enable or disable auto-compaction.
    #[must_use]
    pub fn with_auto_compact(mut self, auto_compact: bool) -> Self {
        self.auto_compact = auto_compact;
        self
    }

    /// Set the parallel tool dispatch policy.
    #[must_use]
    pub fn with_parallel_tool_dispatch(mut self, config: ParallelDispatchConfig) -> Self {
        self.parallel_tool_dispatch = config;
        self
    }

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
    /// Returns a [`Config`](crate::error::LoopError::Config) variant describing
    /// or `Ok(())` if all fields are valid.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::config::LoopConfig;
    ///
    /// let config = LoopConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad = LoopConfig::default().with_compact_threshold(1.5);
    /// assert!(bad.validate().is_err());
    /// ```
    #[must_use = "validation errors should not be silently ignored"]
    pub fn validate(&self) -> Result<(), crate::error::LoopError> {
        if self.max_turns == 0 {
            return Err(crate::error::LoopError::Config(
                "max_turns must be greater than 0".to_string(),
            ));
        }
        if self.context_window == 0 {
            return Err(crate::error::LoopError::Config(
                "context_window must be greater than 0".to_string(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(crate::error::LoopError::Config(
                "max_tokens must be greater than 0".to_string(),
            ));
        }
        if self.model.trim().is_empty() {
            return Err(crate::error::LoopError::Config(
                "model must not be empty".to_string(),
            ));
        }
        if self.compact_threshold.is_nan()
            || self.compact_threshold < 0.0
            || self.compact_threshold > 1.0
        {
            return Err(crate::error::LoopError::Config(format!(
                "compact_threshold must be in [0.0, 1.0], got {}",
                self.compact_threshold
            )));
        }
        if self.parallel_tool_dispatch.max_concurrency == 0 {
            return Err(crate::error::LoopError::Config(
                "parallel_tool_dispatch.max_concurrency must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// How independent tool calls within a single turn are dispatched.
///
/// Defaults to [`Sequential`](ParallelMode::Sequential);
/// opt into [`Parallel`](ParallelMode::Parallel)
/// via [`LoopConfig::parallel_tool_dispatch`].
///
/// Parallel mode runs independent, concurrency-safe calls concurrently (up to
/// [`ParallelDispatchConfig::max_concurrency`]). Sequential mode dispatches one
/// call at a time. Both modes check the cancel signal between calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParallelMode {
    /// Dispatch tool calls one at a time.
    ///
    /// Each call in a turn runs to completion before the next begins.
    /// Observers see strictly paired `on_tool_pre` / `on_tool_post`
    /// events in call order, and the loop detector sees the classic
    /// `[pre A, post A, pre B, post B, …]` interleaving.
    /// Choose this when call ordering matters or when
    /// concurrency is unnecessary (e.g. single-call turns, write-heavy
    /// workloads).
    Sequential,

    /// Dispatch independent, concurrency-safe calls concurrently.
    ///
    /// Calls that are safe for concurrent execution (per
    /// [`Tool::is_safe_for_concurrent_execution`](crate::tool::Tool::is_safe_for_concurrent_execution))
    /// run in parallel up to
    /// [`max_concurrency`](ParallelDispatchConfig::max_concurrency), while
    /// non-safe calls and resource-conflicting calls are serialized into
    /// separate waves. Observers see batched PRE events (all `on_tool_pre`
    /// in input order) then batched POST events — see the dispatch module
    /// docs for the ordering invariant. Choose this for read-heavy,
    /// multi-call turns where latency is the sum of independent operations.
    Parallel,
}

/// Configuration for parallel tool dispatch.
///
/// Defaults to [`ParallelMode::Sequential`] (off) with a
/// `max_concurrency` of 8. Parallel dispatch is strictly opt-in — set
/// [`mode`](Self::mode) to [`Parallel`](ParallelMode::Parallel) to enable.
///
/// # Example
///
/// ```
/// use loopctl::config::{LoopConfig, ParallelDispatchConfig, ParallelMode};
///
/// let config = LoopConfig::default().with_parallel_tool_dispatch(
///     ParallelDispatchConfig { mode: ParallelMode::Parallel, max_concurrency: 4 },
/// );
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParallelDispatchConfig {
    /// Sequential vs parallel dispatch.
    ///
    /// Defaults to [`ParallelMode::Sequential`].
    ///  Set to [`ParallelMode::Parallel`] to enable concurrent dispatch of
    /// independent, concurrency-safe tool calls.
    ///
    /// See [`ParallelMode`] for the observer/detection ordering
    /// implications of each variant.
    pub mode: ParallelMode,

    /// Maximum number of tool calls executing at once under parallel mode.
    ///
    /// Defaults to `8`. Clamped to `[1, eligible_count]` at dispatch time (a
    /// 3-call batch never tries to acquire 8 permits). Setting this to `1`
    /// makes "parallel" mode behave like sequential — useful for debugging
    /// (same code path, no concurrency). Must be at least 1
    /// ([`LoopConfig::validate`] rejects `0`).
    pub max_concurrency: usize,
}

impl Default for ParallelDispatchConfig {
    fn default() -> Self {
        Self {
            mode: ParallelMode::Sequential,
            max_concurrency: 8,
        }
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
        let msg = err.to_string();
        assert!(
            msg.contains("max_tokens"),
            "error should mention max_tokens: {msg}"
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
        let msg = err.to_string();
        assert!(msg.contains("model"), "error should mention model: {msg}");
    }

    #[test]
    fn validate_accepts_nonempty_model() {
        let config = LoopConfig {
            model: "gpt-4".to_string(),
            ..LoopConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_whitespace_only_model() {
        let config = LoopConfig {
            model: "   ".to_string(),
            ..LoopConfig::default()
        };
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("model"), "error should mention model: {msg}");
    }
}
