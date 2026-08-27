//! Agent session configuration.
//!
//! Defines the configuration types that control agent parameters:
//! [`SessionConfig`] for session-scoped settings and
//! [`RunConfig`](crate::engine::RunConfig) for per-run budgets.

use serde::Deserialize;

/// Session-scoped agent configuration.
///
/// The slice of agent configuration that is stable across `run()` calls on the
/// same agent: the session identity, system prompt, and the model's context
/// window. These do not vary from one prompt to the next within a session, so
/// they live here rather than on the per-run [`RunConfig`](crate::engine::core::RunConfig).
///
/// # Construction
///
/// Use [`SessionConfig::default`] for sensible defaults, then override
/// individual fields with the `with_*()` builders.
///
/// ```
/// use loopctl::config::SessionConfig;
///
/// let config = SessionConfig::default().with_context_window(128_000);
/// assert_eq!(config.context_window, 128_000);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Optional system prompt override.
    ///
    /// When `Some`, this text is sent to the provider as the turn's system
    /// prompt. When `None` (the default), the provider receives no system prompt.
    pub system_prompt: Option<String>,

    /// Context window size in tokens.
    ///
    /// Must match the actual window of the configured model. Used by the
    /// compaction subsystem to decide when the request payload exceeds the
    /// budget. A window of `0` disables the window policy entirely — no
    /// threshold and no emergency check, every request is served.
    pub context_window: u64,

    /// Threshold to trigger auto-compaction, as a fraction of the context
    /// window (0–100). Defaults to `80`.
    ///
    /// When the estimated payload size exceeds this percentage of
    /// [`context_window`](Self::context_window), the compaction subsystem is
    /// invoked to summarize or truncate older messages before the next model
    /// call — the comparison is strict, so a payload sitting exactly at the
    /// threshold does not trigger a pass. Lower it to compact more
    /// aggressively; raise it to defer compaction and preserve more raw
    /// history. A threshold of `0` disables this trigger; the emergency
    /// line (at or above 95%) still applies.
    ///
    /// The `0..=100` invariant is enforced by [`Default`], by
    /// [`with_compact_threshold`](Self::with_compact_threshold), and by the
    /// serde deserialize path — values above `100` from any of those routes
    /// are silently clamped to `100`. Direct struct-literal construction
    /// (`SessionConfig { compact_threshold: 200, .. }`) bypasses the clamp
    /// because the field is `pub`; callers using that form are responsible for
    /// honoring the documented range.
    #[serde(deserialize_with = "deserialize_compact_threshold")]
    pub compact_threshold: u8,

    /// Whether auto-compaction is enabled. Defaults to `true`.
    ///
    /// When `true` the loop automatically runs the compactor once the
    /// [`compact_threshold`](Self::compact_threshold) is reached. When
    /// `false` the threshold-based compaction is off, leaving routine
    /// context management to the caller — but the 95%-of-window
    /// emergency safety net still fires (see
    /// [`LoopMachine`](crate::engine::core::LoopMachine)), so a run
    /// never serves a payload it believes is over the window.
    pub auto_compact: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        let mut config = Self {
            system_prompt: None,
            context_window: 200_000,
            compact_threshold: 80,
            auto_compact: true,
        };
        config.clamp_compact_threshold();
        config
    }
}

impl SessionConfig {
    /// Clamp `compact_threshold` into the documented `0..=100` range.
    ///
    /// Single canonical clamp point called by [`Default`],
    /// [`with_compact_threshold`](Self::with_compact_threshold), and the serde
    /// deserialize path. Values above `100` are silently lowered to `100`.
    fn clamp_compact_threshold(&mut self) {
        if self.compact_threshold > 100 {
            self.compact_threshold = 100;
        }
    }

    /// Set the optional system prompt.
    ///
    /// Stores `Some(prompt)` on the config; the provider receives it as
    /// the turn's system prompt. Accepts any `Into<String>` so string
    /// literals work directly. Pass `None` (or skip this builder) to
    /// leave the prompt unset, in which case the provider gets no
    /// system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Set the model's context window size in tokens.
    ///
    /// Must match the actual window of the configured model — the
    /// compaction subsystem compares estimated token usage against this
    /// value to decide when to compact. Setting it too low triggers
    /// premature compaction; too high risks a provider-side context
    /// overflow.
    #[must_use]
    pub fn with_context_window(mut self, context_window: u64) -> Self {
        self.context_window = context_window;
        self
    }

    /// Set the compaction threshold as a percentage of the context
    /// window (0–100).
    ///
    /// When estimated token usage reaches this percentage of
    /// [`context_window`](Self::context_window), the compaction
    /// subsystem summarizes or truncates older messages before the next
    /// model call. Lower it to compact more aggressively; raise it to
    /// defer compaction and preserve more raw history.
    #[must_use]
    pub fn with_compact_threshold(mut self, compact_threshold: u8) -> Self {
        self.compact_threshold = compact_threshold;
        self.clamp_compact_threshold();
        self
    }

    /// Enable or disable auto-compaction.
    ///
    /// When `true` (the default) the loop runs the compactor
    /// automatically once [`compact_threshold`](Self::compact_threshold)
    /// is reached. When `false` the loop never compacts on its own,
    /// even under token pressure, leaving context management entirely
    /// to the caller.
    #[must_use]
    pub fn with_auto_compact(mut self, auto_compact: bool) -> Self {
        self.auto_compact = auto_compact;
        self
    }
}

/// Deserialize helper that clamps [`SessionConfig::compact_threshold`] into
/// `0..=100`, so configs deserialized from disk cannot carry an out-of-range
/// value through to the compaction subsystem.
///
/// Values above `100` are silently lowered to `100` to match the
/// [`clamp_compact_threshold`](SessionConfig::clamp_compact_threshold) clamp used by the other construction
/// paths. Serialization still emits a JSON number, so an in-range value
/// round-trips unchanged; an out-of-range value is normalized to `100` on
/// deserialization and will not round-trip back to the original number.
///
/// # Errors
///
/// Returns the deserializer's error if the input is not a valid `u8` (e.g. a
/// string, a negative number, or out of `u8` range before clamping).
fn deserialize_compact_threshold<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u8::deserialize(deserializer)?;
    Ok(value.min(100))
}

/// How independent tool calls within a single turn are dispatched.
///
/// Defaults to [`Sequential`](ParallelMode::Sequential);
/// opt into [`Parallel`](ParallelMode::Parallel)
/// via [`ParallelDispatchConfig::mode`].
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
    ///
    /// # Side-effects (same granularity in both modes)
    ///
    /// Detection, observer, hook, and health side-effects all fire on **every**
    /// retry attempt in both modes. A tool that fails twice then succeeds
    /// produces 3 detection operations, 3 observer PRE+POST pairs, and 3 health
    /// recordings regardless of `ParallelMode`. All four side-effect targets are
    /// thread-safe ([`DetectionManager`](crate::detection::DetectionManager)
    /// and the observer registry use
    /// `Mutex`/immutable-`Vec` interiors;
    /// [`ToolHealthRegistry`](crate::tool::health::ToolHealthRegistry) uses
    /// atomic counters), so concurrent retry attempts in Parallel mode dispatch
    /// side-effects safely without serialization.
    ///
    /// The only retry-related difference between modes is **interleaving**, not
    /// granularity: in Sequential the classic `[pre A, post A, pre B, post B]`
    /// order is strict, while in Parallel the PRE/POST events for independent
    /// calls in the same wave interleave as those calls progress concurrently.
    /// Observers that pair `on_tool_pre`/`on_tool_post` should key on
    /// `tool_call_id` (carried in both contexts), not on arrival order.
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
/// use loopctl::config::{ParallelDispatchConfig, ParallelMode};
///
/// let dispatch = ParallelDispatchConfig { mode: ParallelMode::Parallel, max_concurrency: 4 };
/// assert_eq!(dispatch.max_concurrency, 4);
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
    /// (same code path, no concurrency). Must be at least 1.
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
    fn session_config_default_context_window() {
        let config = SessionConfig::default();
        assert_eq!(config.context_window, 200_000);
        assert!(config.system_prompt.is_none());
    }

    #[test]
    fn session_config_with_system_prompt_sets_some() {
        let config = SessionConfig::default().with_system_prompt("be helpful");
        assert_eq!(config.system_prompt.as_deref(), Some("be helpful"));
    }

    #[test]
    fn session_config_with_context_window_sets_field() {
        let config = SessionConfig::default().with_context_window(8192);
        assert_eq!(config.context_window, 8192);
    }

    #[test]
    fn session_config_builder_chain_composes_without_clobbering() {
        let config = SessionConfig::default()
            .with_system_prompt("p")
            .with_context_window(128_000);
        assert_eq!(config.system_prompt.as_deref(), Some("p"));
        assert_eq!(config.context_window, 128_000);
    }

    #[test]
    fn session_config_builder_does_not_mutate_source() {
        let original = SessionConfig::default();
        let _modified = original.clone().with_context_window(1);
        assert_eq!(original.context_window, 200_000);
    }

    #[test]
    fn parallel_dispatch_default_is_sequential_concurrency_8() {
        let dispatch = ParallelDispatchConfig::default();
        assert_eq!(dispatch.mode, ParallelMode::Sequential);
        assert_eq!(dispatch.max_concurrency, 8);
    }

    #[test]
    fn parallel_dispatch_struct_literal_round_trips() {
        let dispatch = ParallelDispatchConfig {
            mode: ParallelMode::Parallel,
            max_concurrency: 4,
        };
        assert_eq!(dispatch.mode, ParallelMode::Parallel);
        assert_eq!(dispatch.max_concurrency, 4);
    }

    #[test]
    fn default_compact_threshold_is_valid() {
        let config = SessionConfig::default();
        assert!(
            config.compact_threshold <= 100,
            "default compact_threshold must be in range; got {}",
            config.compact_threshold
        );
    }

    #[test]
    fn with_compact_threshold_clamps_high() {
        let config = SessionConfig::default().with_compact_threshold(200);
        assert_eq!(config.compact_threshold, 100);
    }

    #[test]
    fn with_compact_threshold_preserves_low() {
        let config = SessionConfig::default().with_compact_threshold(50);
        assert_eq!(config.compact_threshold, 50);
    }

    #[test]
    fn deserialize_clamps_high() {
        let json = r#"{"system_prompt":null,"context_window":200000,"compact_threshold":200,"auto_compact":true}"#;
        let config: SessionConfig = serde_json::from_str(json).expect("deserialize should succeed");
        assert_eq!(
            config.compact_threshold, 100,
            "deserialize must clamp out-of-range compact_threshold to 100"
        );
    }

    #[test]
    fn deserialize_preserves_valid() {
        let json = r#"{"system_prompt":null,"context_window":200000,"compact_threshold":80,"auto_compact":true}"#;
        let config: SessionConfig = serde_json::from_str(json).expect("deserialize should succeed");
        assert_eq!(config.compact_threshold, 80);
    }
}
