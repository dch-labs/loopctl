//! Manager bundle — capability traits and runtime infrastructure for agent loops.
//!
//! Two categories of types govern how agent loops
//! interact with their infrastructure:
//!
//! # Capability Traits
//!
//! Capability traits are composable interfaces that represent distinct
//! infrastructure concerns. Each trait describes a single ability —
//! observing lifecycle events, detecting loops, falling back to alternate
//! models, etc. The [`LoopManagers`] struct implements all capability traits,
//! providing a concrete, all-in-one infrastructure bundle.
//!
//! | Trait                    | Purpose                                          |
//! |--------------------------|--------------------------------------------------|
//! | [`Observable`]           | Emit lifecycle events to registered observers    |
//! | [`Detectable`]           | Detect repetitive loops and convergence          |
//! | [`FallbackCapable`]      | Circuit-breaker fallback to alternate models     |
//! | [`Compactable`]          | Automatic context compaction when tokens exceed  |
//! | `StreamCapable`          | Resilient streaming with retries and timeouts    |
//! | `Hookable`               | Bidirectional hooks that can block actions       |
//! | [`PipelineAware`]        | Dispatch tools through a middleware pipeline     |
//! | `HealthTrackable`        | Per-tool health tracking with circuit breakers   |
//!
//! # `LoopManagers`
//!
//! [`LoopManagers`] is the framework's default infrastructure bundle.
//! It bundles all managers, observers, hooks, and middleware into a single
//! struct that can be passed to any agent loop implementation:
//!
//! ```text
//! LoopManagers
//!   ├── ObserverHost           → Observable
//!   ├── DetectionManager       → Detectable
//!   ├── FallbackManager        → FallbackCapable
//!   ├── Option<ContextManager> → Compactable
//!   ├── Option<StreamHandler>  → StreamCapable
//!   ├── Option<HookExecutor>   → Hookable
//!   ├── Option<ToolPipeline>   → PipelineAware
//!   └── Option<ToolHealthReg>  → HealthTrackable
//! ```
//!
//! # Design Philosophy
//!
//! The trait hierarchy separates **what the managers can do** (capability
//! traits) from **how it's composed** (the `LoopManagers` struct). This
//! allows:
//!
//! - **Generic programming** — agent loops can be written against
//!   `impl Observable + Detectable` rather than a concrete type.
//! - **Testing** — swap `LoopManagers` for a stub that implements only the
//!   traits under test.
//! ```rust,ignore
//! let managers = LoopManagers::new()
//!     .with_fallback(FallbackManager::for_model("llm-70b"))
//!     .with_detection(DetectionManager::default())
//!     .with_observer(Arc::new(logging_observer));
//!
//! let agent = BareLoop::new_with_managers(client, tools, config, managers);
//! ```
//!
//! Every capability is optional. A bundle with no `.with_*()` calls still
//! works — it just has no observers, no hooks, no pipeline, etc.  This means
//! you only pay for (and configure) the infrastructure you actually use.

use std::sync::Arc;

use crate::compact::ContextManager;
use crate::detection::DetectionManager;
use crate::detection::{ConvergenceAction, DetectedPattern};

use crate::fallback::FallbackManager;
#[cfg(feature = "hooks")]
use crate::hooks::HookExecutor;
use crate::middleware::ToolPipeline;
use crate::observer::{ConvergenceDetectedContext, LoopDetectedContext};
use crate::observer::{LoopObserver, ObserverHost};
#[cfg(feature = "streaming")]
use crate::stream::handler::StreamHandler;
#[cfg(feature = "tool_health")]
use crate::tool::health::ToolHealthRegistry;

pub use crate::capabilities::*;

/// The framework's default infrastructure bundle for agent loops.
///
/// `LoopManagers` bundles all the cross-cutting infrastructure an agent
/// loop needs: observers, detection, fallback, hooks, middleware pipeline,
/// and health tracking. It implements all capability traits so that agent
/// loops can be written against trait bounds rather than a concrete type.
///
/// # Construction
///
/// Use [`LoopManagers::new()`] then chain `.with_*()` methods to compose
/// only the capabilities you need:
///
/// ```
/// # use loopctl::managers::LoopManagers;
/// # use loopctl::managers::FallbackCapable;
/// use loopctl::fallback::FallbackManager;
///
/// let managers = LoopManagers::new()
///     .with_fallback(FallbackManager::for_model("llm-70b"));
/// assert_eq!(managers.fallback().active_model().as_deref(), Some("llm-70b"));
/// ```
///
/// # Capability Traits
///
/// `LoopManagers` implements all capability traits defined in this module:
///
/// - [`Observable`] — via the internal [`ObserverHost`]
/// - [`Detectable`] — via the internal [`DetectionManager`]
/// - [`FallbackCapable`] — via the internal [`FallbackManager`]
/// - [`Compactable`] — via an optional [`ContextManager`]
/// - `StreamCapable` — via an optional `StreamHandler`
/// - `Hookable` — via an optional `HookExecutor` *(requires `hooks` feature)*
/// - [`PipelineAware`] — via an optional [`ToolPipeline`]
/// - `HealthTrackable` — via an optional `ToolHealthRegistry` *(requires `tool_health` feature)*
///
/// # Reset
///
/// Call [`reset_all`](LoopManagers::reset_all) at the start of a new
/// session to reinitialise every manager and observer to its default state.
pub struct LoopManagers {
    /// Circuit breaker for API model fallback.
    ///
    /// Tracks consecutive API failures and, when the threshold is exceeded,
    /// switches to a configured backup model. Fresh on construction; call
    /// [`reset_all`](Self::reset_all) to reinitialise mid-session.
    fallback: FallbackManager,

    /// Loop and convergence detection orchestrator.
    ///
    /// Monitors the agent's responses for repetitive patterns (identical tool
    /// calls, text convergence) and fires observer events. When thresholds
    /// are exceeded, returns an error that aborts the run.
    detection: DetectionManager,

    /// Observer host for lifecycle event fan-out.
    ///
    /// Holds the list of registered [`LoopObserver`](crate::observer::LoopObserver)
    /// instances and dispatches every lifecycle event (turn start/end, stream
    /// deltas, tool dispatch, compaction) to all of them.
    observer_host: ObserverHost,

    /// Optional middleware pipeline wrapping tool dispatch.
    ///
    /// When set, every tool call passes through the pipeline before reaching
    /// the tool itself. Middleware can modify inputs, cache results, enforce
    /// permissions, or cap output size.
    tool_pipeline: Option<ToolPipeline>,

    /// Optional context manager for automatic compaction.
    ///
    /// When set, the driver runs the context manager's compactor after the
    /// machine's compaction trigger fires, reducing the history to fit within
    /// the context window budget.
    context_manager: Option<Arc<ContextManager>>,

    /// Optional stream handler for resilient streaming.
    ///
    /// When set, wraps streaming calls with retry logic, timeout enforcement,
    /// and automatic fallback to non-streaming when the provider drops the
    /// connection mid-stream. Absent under `default = []`; requires the
    /// `streaming` feature.
    #[cfg(feature = "streaming")]
    stream_handler: Option<StreamHandler>,

    /// Optional hook executor for bidirectional lifecycle interception.
    ///
    /// When set, hooks can inspect and approve or reject actions before they
    /// execute (pre-tool, pre-compact) and observe outcomes after (post-tool,
    /// post-compact). Unlike observers, hooks can control flow.
    #[cfg(feature = "hooks")]
    hook_executor: Option<Arc<HookExecutor>>,

    /// Optional per-tool health tracker with circuit breakers.
    ///
    /// When set, records success/failure counts and latency for every tool
    /// dispatch. Tools that exceed the failure threshold have their circuit
    /// breaker opened, blocking subsequent calls until recovery.
    #[cfg(feature = "tool_health")]
    health_registry: Option<Arc<ToolHealthRegistry>>,

    /// Optional agent memory backend.
    ///
    /// When set, the engine stores a trajectory entry after each successful
    /// tool call, retrieves relevant entries before each turn to inject as
    /// context, and consolidates (prunes) the store at the end of a
    /// successful run. Persists across manager resets — memory is meant to
    /// survive a `reset_all`.
    memory: Option<Arc<dyn crate::memory::LoopMemory>>,
}

impl LoopManagers {
    /// Create a new runtime with default managers and no optional components.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::managers::LoopManagers;
    /// # use loopctl::managers::FallbackCapable;
    /// let managers = LoopManagers::new();
    /// assert!(managers.fallback().active_model().is_none());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallback: FallbackManager::default(),
            detection: DetectionManager::default(),
            observer_host: ObserverHost::new(),
            tool_pipeline: None,
            context_manager: None,
            #[cfg(feature = "streaming")]
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
            memory: None,
        }
    }

    /// Replace the fallback manager with a custom instance.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::managers::LoopManagers;
    /// # use loopctl::managers::FallbackCapable;
    /// # use loopctl::fallback::FallbackManager;
    /// let managers = LoopManagers::new()
    ///     .with_fallback(FallbackManager::for_model("llm-70b"));
    /// assert_eq!(managers.fallback().active_model().as_deref(), Some("llm-70b"));
    /// ```
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackManager) -> Self {
        self.fallback = fallback;
        self
    }

    /// Replace the detection manager with a custom instance.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::managers::LoopManagers;
    /// # use loopctl::managers::Detectable;
    /// # use loopctl::detection::{DetectionManager, DetectionConfig};
    /// let config = DetectionConfig {
    ///     loop_threshold: 5,
    ///     ..Default::default()
    /// };
    /// let managers = LoopManagers::new()
    ///     .with_detection(DetectionManager::new_with_config(config).unwrap());
    /// assert_eq!(managers.detection().config().loop_threshold, 5);
    /// ```
    #[must_use]
    pub fn with_detection(mut self, detection: DetectionManager) -> Self {
        self.detection = detection;
        self
    }

    /// Register an observer and return `self` for chaining.
    ///
    /// Builder-style alias for [`register_observer`](Self::register_observer).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = LoopManagers::new()
    ///     .with_observer(Arc::new(logging_observer))
    ///     .with_observer(Arc::new(metrics_observer))
    /// ```
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<dyn LoopObserver>) -> Self {
        self.observer_host.register(observer);
        self
    }

    /// Access the observer host directly.
    ///
    /// The observer host manages fan-out to all registered observers.
    /// Use this to fire events or inspect the registered observer list.
    pub fn observers(&self) -> &ObserverHost {
        &self.observer_host
    }

    /// Set the middleware pipeline for tool dispatch (builder-style).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = LoopManagers::new()
    ///     .with_pipeline(builder.build()?)
    /// ```
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: ToolPipeline) -> Self {
        self.tool_pipeline = Some(pipeline);
        self
    }

    /// Set the tool health registry for per-tool health tracking (builder-style).
    ///
    /// When set, records success/failure counts and latency for every
    /// tool dispatch. Tools that exceed the failure threshold have their
    /// circuit breaker opened, blocking subsequent calls until recovery.
    ///
    /// *Requires `tool_health` feature.*
    #[must_use]
    #[cfg(feature = "tool_health")]
    pub fn with_health_registry(mut self, registry: Arc<ToolHealthRegistry>) -> Self {
        self.health_registry = Some(registry);
        self
    }

    /// Set the stream handler for resilient streaming (builder-style).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = LoopManagers::new()
    ///     .with_stream_handler(handler)
    /// ```
    #[must_use]
    #[cfg(feature = "streaming")]
    pub fn with_stream_handler(mut self, handler: StreamHandler) -> Self {
        self.stream_handler = Some(handler);
        self
    }

    /// Set the hook executor for bidirectional lifecycle interception (builder-style).
    ///
    /// *Requires `hooks` feature.*
    #[must_use]
    #[cfg(feature = "hooks")]
    pub fn with_hook_executor(mut self, executor: Arc<HookExecutor>) -> Self {
        self.hook_executor = Some(executor);
        self
    }

    /// Register an observer with the observer host.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::observer::LoopObserver;
    /// use std::sync::Arc;
    ///
    /// let mut managers = LoopManagers::new();
    /// managers.register_observer(Arc::new(MyObserver));
    /// ```
    pub fn register_observer(&mut self, observer: Arc<dyn LoopObserver>) {
        self.observer_host.register(observer);
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// Non-consuming variant of [`with_pipeline`](Self::with_pipeline) for
    /// cases where a `LoopManagers` is already constructed. If you are installing
    /// onto a [`BareLoop`](crate::engine::BareLoop), prefer
    /// [`BareLoop::set_pipeline`](crate::engine::BareLoop::set_pipeline), which
    /// shares the loop's tool registry with the pipeline core automatically.
    pub fn set_pipeline(&mut self, pipeline: ToolPipeline) {
        self.tool_pipeline = Some(pipeline);
    }

    /// Set the context manager for automatic compaction (builder-style).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = LoopManagers::new()
    ///     .with_context_manager(Arc::new(manager))
    /// ```
    #[must_use]
    pub fn with_context_manager(mut self, manager: Arc<ContextManager>) -> Self {
        self.context_manager = Some(manager);
        self
    }

    /// Set the context manager for automatic compaction.
    ///
    /// Non-consuming variant of [`with_context_manager`](Self::with_context_manager).
    /// The context manager's context window is synced from the session
    /// config when attached via [`BareLoop::set_context_manager`](crate::engine::BareLoop::set_context_manager).
    pub fn set_context_manager(&mut self, manager: Arc<ContextManager>) {
        self.context_manager = Some(manager);
    }

    /// Set the stream handler for resilient streaming.
    ///
    /// Non-consuming variant of [`with_stream_handler`](Self::with_stream_handler).
    #[cfg(feature = "streaming")]
    pub fn set_stream_handler(&mut self, handler: StreamHandler) {
        self.stream_handler = Some(handler);
    }

    /// Set the hook executor for bidirectional lifecycle interception.
    ///
    /// Non-consuming variant of [`with_hook_executor`](Self::with_hook_executor).
    ///
    /// *Requires `hooks` feature.*
    #[cfg(feature = "hooks")]
    pub fn set_hook_executor(&mut self, executor: Arc<HookExecutor>) {
        self.hook_executor = Some(executor);
    }

    /// Set the tool health registry for per-tool health tracking.
    ///
    /// Non-consuming variant of [`with_health_registry`](Self::with_health_registry).
    ///
    /// *Requires `tool_health` feature.*
    #[cfg(feature = "tool_health")]
    pub fn set_health_registry(&mut self, registry: Arc<ToolHealthRegistry>) {
        self.health_registry = Some(registry);
    }

    /// Set the agent memory backend (builder-style).
    ///
    /// When set, the engine stores tool-execution trajectories, retrieves
    /// relevant entries as context before each turn, and consolidates the
    /// store at the end of a successful run.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::memory::InMemoryStore;
    /// use std::sync::Arc;
    ///
    /// let managers = LoopManagers::new()
    ///     .with_memory(Arc::new(InMemoryStore::new()));
    /// ```
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn crate::memory::LoopMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the agent memory backend.
    ///
    /// Non-consuming variant of [`with_memory`](Self::with_memory).
    pub fn set_memory(&mut self, memory: Arc<dyn crate::memory::LoopMemory>) {
        self.memory = Some(memory);
    }

    /// Borrow the memory backend, if configured.
    ///
    /// Returns `None` when no memory store is attached.
    #[must_use]
    pub fn memory(&self) -> Option<&Arc<dyn crate::memory::LoopMemory>> {
        self.memory.as_ref()
    }

    /// Reset all managers and observers to their initial state.
    ///
    /// Clears the fallback circuit breaker, loop/convergence detection
    /// history, and per-observer accumulators. Call this when you want a
    /// clean slate mid-session — for example after a provider outage
    /// resolves (so the circuit breaker does not stay tripped) or when
    /// switching to an unrelated task (so stale detection history does not
    /// skew the next run). The engine does not call this automatically;
    /// manager state persists across `run()` calls within a session.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::managers::LoopManagers;
    /// let managers = LoopManagers::new();
    /// // ... after several runs ...
    /// managers.reset_all();
    /// // All managers are back to their initial state
    /// ```
    pub fn reset_all(&self) {
        self.fallback.reset();
        self.detection.reset();
        self.observer_host.reset_all();
    }

    /// Fire tracing log lines and observer events for a detected pattern.
    ///
    /// Called by the agent loop after each response or tool operation is
    /// recorded with the [`DetectionManager`]. Does **not** decide whether
    /// to abort — that is the engine's job.
    pub fn notify_detected_pattern(
        &self,
        pattern: &crate::detection::DetectedPattern,
        turn: usize,
    ) {
        match pattern {
            DetectedPattern::NoPattern => {}

            DetectedPattern::LoopDetected {
                repetitions,
                pattern_description,
            } => {
                tracing::warn!(
                    repetitions,
                    pattern = %pattern_description,
                    turn,
                    "loop detected"
                );

                self.observer_host.on_loop_detected(&LoopDetectedContext {
                    pattern: pattern_description.clone(),
                    repetitions: *repetitions,
                });
            }

            DetectedPattern::ConvergenceDetected {
                similarity,
                consecutive_count,
            } => {
                tracing::warn!(similarity, consecutive_count, turn, "convergence detected");
                let action = self.detection.config().on_converge;
                let action_str = match action {
                    ConvergenceAction::Stop => "stop",
                    ConvergenceAction::Warn => "warn",
                    ConvergenceAction::Compact => "compact",
                    ConvergenceAction::AskUser => "ask_user",
                    ConvergenceAction::SwitchPhase => "switch_phase",
                };

                self.observer_host
                    .on_convergence_detected(&ConvergenceDetectedContext {
                        action: action_str.to_string(),
                    });
            }
        }
    }
}

impl Default for LoopManagers {
    /// Produce a [`LoopManagers`] with default managers and no optional components.
    ///
    /// Equivalent to [`LoopManagers::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl crate::capabilities::Observable for LoopManagers {
    fn observers(&self) -> &ObserverHost {
        &self.observer_host
    }
}

impl crate::capabilities::Detectable for LoopManagers {
    fn detection(&self) -> &DetectionManager {
        &self.detection
    }
}

impl crate::capabilities::FallbackCapable for LoopManagers {
    fn fallback(&self) -> &FallbackManager {
        &self.fallback
    }
}

impl crate::capabilities::Compactable for LoopManagers {
    fn context_manager(&self) -> Option<&Arc<ContextManager>> {
        self.context_manager.as_ref()
    }
}

impl crate::capabilities::RememberCapable for LoopManagers {
    fn memory(&self) -> Option<&Arc<dyn crate::memory::LoopMemory>> {
        self.memory.as_ref()
    }
}

#[cfg(feature = "streaming")]
impl crate::capabilities::StreamCapable for LoopManagers {
    fn stream_handler(&self) -> &StreamHandler {
        self.stream_handler
            .as_ref()
            .unwrap_or(StreamHandler::passthrough_default())
    }
}

#[cfg(feature = "hooks")]
impl crate::capabilities::Hookable for LoopManagers {
    fn hook_executor(&self) -> Option<&HookExecutor> {
        self.hook_executor.as_deref()
    }
}

impl crate::capabilities::PipelineAware for LoopManagers {
    fn pipeline(&self) -> Option<&ToolPipeline> {
        self.tool_pipeline.as_ref()
    }
}

#[cfg(feature = "tool_health")]
impl crate::capabilities::HealthTrackable for LoopManagers {
    fn health_registry(&self) -> Option<&ToolHealthRegistry> {
        self.health_registry.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::{DetectedPattern, DetectionManager};

    #[test]
    fn test_runtime_default() {
        let managers = LoopManagers::default();
        assert!(managers.fallback().active_model().is_none());
    }

    #[test]
    fn test_runtime_with_custom_fallback() {
        let fallback = FallbackManager::for_model("my-model");
        let managers = LoopManagers::new().with_fallback(fallback);
        assert_eq!(
            managers.fallback().active_model().as_deref(),
            Some("my-model")
        );
    }

    #[test]
    fn test_reset_all() {
        let managers = LoopManagers::new();
        managers.reset_all();
    }

    #[test]
    fn test_runtime_contains_detection_manager() {
        let managers = LoopManagers::new();
        assert_eq!(managers.detection().config().loop_threshold, 3);
        assert_eq!(managers.detection().config().stop_threshold, 10);
    }

    #[test]
    fn test_runtime_with_custom_detection() {
        use crate::detection::DetectionConfig;

        let config = DetectionConfig {
            loop_threshold: 7,
            stop_threshold: 20,
            ..Default::default()
        };
        let detection = DetectionManager::new_with_config(config).unwrap();
        let managers = LoopManagers::new().with_detection(detection);
        assert_eq!(managers.detection().config().loop_threshold, 7);
        assert_eq!(managers.detection().config().stop_threshold, 20);
        assert!(managers.fallback().active_model().is_none());
    }

    #[test]
    fn test_reset_all_clears_detection() {
        let managers = LoopManagers::new();
        let _ = managers.detection().record_tool_call("Read", 12345);
        managers.reset_all();
        let pattern = managers.detection().record_tool_call("Read", 12345);
        assert!(matches!(pattern, DetectedPattern::NoPattern));
    }

    #[test]
    fn test_capability_traits_are_object_safe() {
        fn _assert_observable(_: &dyn Observable) {}
        fn _assert_detectable(_: &dyn Detectable) {}
        fn _assert_fallback(_: &dyn FallbackCapable) {}
        fn _assert_pipeline(_: &dyn PipelineAware) {}
        fn _assert_compactable(_: &dyn Compactable) {}
        #[cfg(feature = "streaming")]
        fn _assert_stream_capable(_: &dyn StreamCapable) {}

        let managers = LoopManagers::new();
        _assert_observable(&managers);
        _assert_detectable(&managers);
        _assert_fallback(&managers);
        _assert_pipeline(&managers);
        _assert_compactable(&managers);
        #[cfg(feature = "streaming")]
        _assert_stream_capable(&managers);
    }

    #[test]
    fn test_pipeline_defaults_to_none() {
        let managers = LoopManagers::new();
        assert!(managers.pipeline().is_none());
    }

    #[test]
    fn test_observers_accessible_via_trait() {
        let managers = LoopManagers::new();
        let _: &ObserverHost = managers.observers();
    }

    #[test]
    fn test_context_manager_defaults_to_none() {
        let managers = LoopManagers::new();
        assert!(managers.context_manager().is_none());
    }

    #[test]
    #[cfg(feature = "streaming")]
    fn test_stream_handler_defaults_to_passthrough() {
        let managers = LoopManagers::new();
        let handler = managers.stream_handler();
        assert_eq!(
            handler.timeout_config().total_stream_timeout,
            std::time::Duration::MAX
        );
    }

    #[test]
    fn test_set_pipeline_returns_some() {
        use crate::middleware::ToolPipeline;
        use crate::tool::ToolRegistry;
        use std::sync::Arc;

        let registry = Arc::new(ToolRegistry::new());
        let pipeline = ToolPipeline::new(registry);
        let mut managers = LoopManagers::new();
        assert!(managers.pipeline().is_none());
        managers.set_pipeline(pipeline);
        assert!(managers.pipeline().is_some());
    }

    #[test]
    fn test_set_context_manager_returns_some() {
        use crate::compact::{ContextManager, TruncatingCompactor};

        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        let mut managers = LoopManagers::new();
        assert!(managers.context_manager().is_none());
        managers.set_context_manager(Arc::new(manager));
        assert!(managers.context_manager().is_some());
    }

    #[test]
    #[cfg(feature = "streaming")]
    fn test_set_stream_handler_overrides_passthrough() {
        use crate::stream::handler::StreamHandler;

        let handler = StreamHandler::new();
        let mut managers = LoopManagers::new();
        assert_eq!(
            managers
                .stream_handler()
                .timeout_config()
                .total_stream_timeout,
            std::time::Duration::MAX
        );
        managers.set_stream_handler(handler);
        assert_eq!(
            managers
                .stream_handler()
                .timeout_config()
                .total_stream_timeout,
            std::time::Duration::from_mins(5)
        );
    }

    #[test]
    fn test_register_observer_increments_count() {
        use crate::observer::LoopObserver;
        use std::sync::Arc;

        struct NopObserver;
        impl LoopObserver for NopObserver {
            fn name(&self) -> &'static str {
                "NopObserver"
            }
        }

        let mut managers = LoopManagers::new();
        assert!(managers.observers().is_empty());
        managers.register_observer(Arc::new(NopObserver));
        assert_eq!(managers.observers().len(), 1);
        managers.register_observer(Arc::new(NopObserver));
        assert_eq!(managers.observers().len(), 2);
    }

    #[test]
    fn test_reset_all_clears_fallback() {
        let managers = LoopManagers::new();
        let _ = managers
            .fallback()
            .record_failure(crate::fallback::FailureKind::Transient);
        managers.reset_all();
        assert!(managers.fallback().active_model().is_none());
    }

    #[test]
    fn test_reset_all_clears_observers() {
        use crate::observer::LoopObserver;
        use std::sync::Arc;

        struct NopObserver;
        impl LoopObserver for NopObserver {
            fn name(&self) -> &'static str {
                "NopObserver"
            }
        }

        let mut managers = LoopManagers::new();
        managers.register_observer(Arc::new(NopObserver));
        assert_eq!(managers.observers().len(), 1);
        managers.reset_all();
        // Observer count persists across reset — reset_all calls
        // reset_all on each observer, it doesn't remove them.
        assert_eq!(managers.observers().len(), 1);
    }

    #[test]
    fn test_generic_bounds_accept_runtime() {
        fn accepts_observable(_: &impl Observable) {}
        fn accepts_detectable(_: &impl Detectable) {}
        fn accepts_fallback(_: &impl FallbackCapable) {}
        fn accepts_compactable(_: &impl Compactable) {}
        #[cfg(feature = "streaming")]
        fn accepts_stream_capable(_: &impl StreamCapable) {}
        fn accepts_pipeline(_: &impl PipelineAware) {}
        fn accepts_multi_bound(_: &(impl Observable + Detectable + FallbackCapable)) {}

        let managers = LoopManagers::new();
        accepts_observable(&managers);
        accepts_detectable(&managers);
        accepts_fallback(&managers);
        accepts_compactable(&managers);
        #[cfg(feature = "streaming")]
        accepts_stream_capable(&managers);
        accepts_pipeline(&managers);
        accepts_multi_bound(&managers);
    }
}
