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
/// Use [`LoopConfig::default`] for sensible defaults, then override individual
/// fields either with the fluent `with_*()` builders or direct struct-literal
/// assignment — both are supported.
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
    /// When `Some`, this text is sent to the provider as the turn's system
    /// prompt. When `None` (the default), the provider receives no system
    /// prompt for the turn. Set this to inject a custom persona or instruction
    /// set.
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

    /// Threshold to trigger auto-compaction, in basis points (`0–10_000`;
    /// `10_000` = 100% of the context window).
    ///
    /// When the estimated token usage of the conversation exceeds
    /// `compact_threshold * context_window / 10_000`, the compactor runs before
    /// the next turn to avoid an overflow. Defaults to `8_000` (80%). Must be in
    /// `[0, 10_000]` ([`validate`](LoopConfig::validate)).
    pub compact_threshold: u16,

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
    /// | [`compact_threshold`](LoopConfig::compact_threshold) | `8_000` (80%) |
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
            compact_threshold: 8_000,
            auto_compact: true,
            parallel_tool_dispatch: ParallelDispatchConfig::default(),
        }
    }
}

impl LoopConfig {
    /// Set the session identifier, overriding the random UUID v4 that
    /// [`Default`](LoopConfig::default) generates.
    ///
    /// Use this to resume a prior session under its existing ID, or to pin a
    /// deterministic ID in tests.
    #[must_use]
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = session_id;
        self
    }

    /// Set the model identifier passed to the API client on each request.
    ///
    /// Accepts anything that converts into a `String` (for example `"gpt-4o"`),
    /// so a `&str` literal works without an explicit `.to_string()`. The value
    /// must be non-empty; see [`validate`](LoopConfig::validate).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the optional system-prompt override.
    ///
    /// When `Some`, this text is sent to the provider as the turn's system
    /// prompt (verbatim). When `None` (the default), the provider receives no
    /// system prompt for the turn. Pass `Some(prompt)` to inject a custom
    /// persona or instructions, or `None` to revert an earlier override.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.system_prompt = system_prompt;
        self
    }

    /// Set the maximum number of turns before the loop forces completion.
    ///
    /// A safety cap: the loop halts with
    /// [`LoopError::MaxTurnsExceeded`](crate::error::LoopError::MaxTurnsExceeded)
    /// once it is reached. Must be at least `1`; see
    /// [`validate`](LoopConfig::validate).
    #[must_use]
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the maximum tokens for each API response, sent to the provider as
    /// `max_tokens` on every request.
    ///
    /// Must be at least `1`; see [`validate`](LoopConfig::validate).
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the context window size in tokens.
    ///
    /// Should match the actual window of the configured
    /// [`model`](LoopConfig::model); the compaction subsystem uses it together
    /// with [`compact_threshold`](LoopConfig::compact_threshold) to decide when
    /// to run. Must be at least `1`; see [`validate`](LoopConfig::validate).
    #[must_use]
    pub fn with_context_window(mut self, context_window: u64) -> Self {
        self.context_window = context_window;
        self
    }

    /// Set the auto-compaction trigger threshold in basis points (`0–10_000`;
    /// `10_000` = 100% of the context window).
    ///
    /// When estimated token usage exceeds
    /// `compact_threshold * context_window / 10_000`, compaction runs before the
    /// next turn. This builder stores the value verbatim — it does **not** clamp;
    /// the range is enforced by [`validate`](LoopConfig::validate), which is the
    /// single source of truth for the `[0, 10_000]` bound.
    #[must_use]
    pub fn with_compact_threshold(mut self, compact_threshold: u16) -> Self {
        self.compact_threshold = compact_threshold;
        self
    }

    /// Enable or disable automatic compaction.
    ///
    /// When `true` (the default), compaction runs automatically once
    /// [`compact_threshold`](LoopConfig::compact_threshold) is reached. When
    /// `false`, the host application must manage context size itself.
    #[must_use]
    pub fn with_auto_compact(mut self, auto_compact: bool) -> Self {
        self.auto_compact = auto_compact;
        self
    }

    /// Set how independent tool calls within a single turn are dispatched.
    ///
    /// Defaults to sequential. See [`ParallelDispatchConfig`] and
    /// [`ParallelMode`] for the observer and loop-detection ordering
    /// implications of enabling parallel dispatch.
    #[must_use]
    pub fn with_parallel_tool_dispatch(mut self, config: ParallelDispatchConfig) -> Self {
        self.parallel_tool_dispatch = config;
        self
    }

    /// Validate the configuration fields.
    ///
    /// Checks that:
    /// - `compact_threshold` is in the range `[0, 10_000]`.
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
    /// let bad = LoopConfig::default().with_compact_threshold(15_000);
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
        if self.compact_threshold > 10_000 {
            return Err(crate::error::LoopError::Config(format!(
                "compact_threshold must be in [0, 10_000], got {}",
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

    #[test]
    fn with_model_sets_field_and_accepts_str() {
        let config = LoopConfig::default().with_model("gpt-4o");
        assert_eq!(config.model, "gpt-4o");
    }

    #[test]
    fn with_system_prompt_sets_some() {
        let config = LoopConfig::default().with_system_prompt(Some("p".to_string()));
        assert_eq!(config.system_prompt, Some("p".to_string()));
    }

    #[test]
    fn with_system_prompt_none_clears_override() {
        let config = LoopConfig::default().with_system_prompt(None);
        assert!(config.system_prompt.is_none());
    }

    #[test]
    fn with_max_turns_sets_field() {
        let config = LoopConfig::default().with_max_turns(7);
        assert_eq!(config.max_turns, 7);
    }

    #[test]
    fn with_max_tokens_sets_field() {
        let config = LoopConfig::default().with_max_tokens(123);
        assert_eq!(config.max_tokens, 123);
    }

    #[test]
    fn with_context_window_sets_field() {
        let config = LoopConfig::default().with_context_window(8192);
        assert_eq!(config.context_window, 8192);
    }

    #[test]
    fn with_compact_threshold_stores_value_without_clamping() {
        let config = LoopConfig::default().with_compact_threshold(15_000);
        assert_eq!(config.compact_threshold, 15_000);
    }

    #[test]
    fn with_auto_compact_flips_default() {
        let config = LoopConfig::default().with_auto_compact(false);
        assert!(!config.auto_compact);
    }

    #[test]
    fn with_session_id_round_trips() {
        let id = Uuid::new_v4();
        let config = LoopConfig::default().with_session_id(id);
        assert_eq!(config.session_id, id);
    }

    #[test]
    fn with_parallel_tool_dispatch_round_trips() {
        let dispatch = ParallelDispatchConfig {
            mode: ParallelMode::Parallel,
            max_concurrency: 4,
        };
        let config = LoopConfig::default().with_parallel_tool_dispatch(dispatch);
        assert_eq!(config.parallel_tool_dispatch.mode, ParallelMode::Parallel);
        assert_eq!(config.parallel_tool_dispatch.max_concurrency, 4);
    }

    #[test]
    fn builder_chain_composes_without_clobbering() {
        let config = LoopConfig::default().with_model("x").with_max_turns(5);
        assert_eq!(config.model, "x");
        assert_eq!(config.max_turns, 5);
        let defaults = LoopConfig::default();
        assert_eq!(config.max_tokens, defaults.max_tokens);
        assert_eq!(config.context_window, defaults.context_window);
        assert_eq!(config.compact_threshold, defaults.compact_threshold);
        assert_eq!(config.auto_compact, defaults.auto_compact);
    }

    #[test]
    fn builder_does_not_mutate_source() {
        let original = LoopConfig::default();
        let _modified = original.clone().with_max_turns(1);
        assert_eq!(original.max_turns, 200);
    }

    #[test]
    fn validate_still_rejects_builder_built_invalid_config() {
        let config = LoopConfig::default().with_compact_threshold(15_000);
        assert!(config.validate().is_err());
    }
}
