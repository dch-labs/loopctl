//! Loop runtime — capability traits and runtime infrastructure for agent loops.
//!
//! Two categories of types govern how agent loops
//! interact with their infrastructure:
//!
//! # Capability Traits
//!
//! Capability traits are composable interfaces that represent distinct
//! infrastructure concerns. Each trait describes a single ability —
//! observing lifecycle events, detecting loops, falling back to alternate
//! models, etc. The [`LoopRuntime`] struct implements all capability traits,
//! providing a concrete, all-in-one infrastructure bundle.
//!
//! | Trait                    | Purpose                                          |
//! |--------------------------|--------------------------------------------------|
//! | [`Observable`]           | Emit lifecycle events to registered observers    |
//! | [`Detectable`]           | Detect repetitive loops and convergence          |
//! | [`FallbackCapable`]      | Circuit-breaker fallback to alternate models     |
//! | [`Compactable`]          | Automatic context compaction when tokens exceed  |
//! | [`StreamCapable`]        | Resilient streaming with retries and timeouts    |
//! | [`Hookable`]             | Bidirectional hooks that can block actions       |
//! | [`PipelineAware`]        | Dispatch tools through a middleware pipeline     |
//! | [`HealthTrackable`]      | Per-tool health tracking with circuit breakers   |
//!
//! # `LoopRuntime`
//!
//! [`LoopRuntime`] is the framework's default infrastructure bundle.
//! It bundles all managers, observers, hooks, and middleware into a single
//! struct that can be passed to any agent loop implementation:
//!
//! ```text
//! LoopRuntime
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
//! The trait hierarchy separates **what the runtime can do** (capability
//! traits) from **how it's composed** (the `LoopRuntime` struct). This
//! allows:
//!
//! - **Generic programming** — agent loops can be written against
//!   `impl Observable + Detectable` rather than a concrete type.
//! - **Testing** — swap `LoopRuntime` for a stub that implements only the
//!   traits under test.
//! - **Incremental adoption** — start with `LoopRuntime::new()` and add
//!   capabilities via builder methods as needed.

use std::sync::Arc;

use crate::compact::ContextManager;
use crate::detection::DetectionManager;
use crate::fallback::FallbackManager;
#[cfg(feature = "hooks")]
use crate::hooks::HookExecutor;
use crate::middleware::ToolPipeline;
use crate::observer::{LoopObserver, ObserverHost};
use crate::stream::handler::StreamHandler;
#[cfg(feature = "tool_health")]
use crate::tool::health::ToolHealthRegistry;

pub use crate::capabilities::*;

// ==================================================
// LoopRuntime
// ==================================================

/// The framework's default infrastructure bundle for agent loops.
///
/// `LoopRuntime` bundles all the cross-cutting infrastructure an agent
/// loop needs: observers, detection, fallback, hooks, middleware pipeline,
/// and health tracking. It implements all capability traits so that agent
/// loops can be written against trait bounds rather than a concrete type.
///
/// # Construction
///
/// Use [`LoopRuntime::new`] for defaults or the builder-style `with_*`
/// methods to configure individual components:
///
/// ```rust,ignore
/// use loopctl::runtime::LoopRuntime;
/// use loopctl::fallback::FallbackManager;
///
/// let runtime = LoopRuntime::new()
///     .with_fallback(FallbackManager::for_model("llm-70b"));
/// ```
///
/// # Capability Traits
///
/// `LoopRuntime` implements all capability traits defined in this module:
///
/// - [`Observable`] — via the internal [`ObserverHost`]
/// - [`Detectable`] — via the internal [`DetectionManager`]
/// - [`FallbackCapable`] — via the internal [`FallbackManager`]
/// - [`Compactable`] — via an optional [`ContextManager`]
/// - [`StreamCapable`] — via an optional [`StreamHandler`]
/// - [`Hookable`] — via an optional [`HookExecutor`] **
/// - [`PipelineAware`] — via an optional [`ToolPipeline`]
/// - [`HealthTrackable`] — via an optional [`ToolHealthRegistry`] *(requires `tool_health` feature)*
///
/// # Reset
///
/// Call [`reset_all`](LoopRuntime::reset_all) at the start of a new
/// session to reinitialise every manager and observer to its default state.
pub struct LoopRuntime {
    /// Circuit breaker for API model fallback.
    pub fallback: FallbackManager,
    /// Loop and convergence detection orchestrator.
    pub detection: DetectionManager,
    /// Observer host for lifecycle event fan-out.
    observer_host: ObserverHost,
    /// Optional middleware pipeline wrapping tool dispatch.
    tool_pipeline: Option<ToolPipeline>,
    /// Optional context manager for automatic compaction.
    context_manager: Option<Arc<ContextManager>>,
    /// Optional stream handler for resilient streaming with retries.
    stream_handler: Option<StreamHandler>,
    /// Optional hook executor for bidirectional lifecycle interception.
    #[cfg(feature = "hooks")]
    hook_executor: Option<Arc<HookExecutor>>,
    /// Optional per-tool health tracker with circuit breakers. *Requires `tool_health` feature.*
    #[cfg(feature = "tool_health")]
    health_registry: Option<Arc<ToolHealthRegistry>>,
}

impl LoopRuntime {
    /// Create a new runtime with default managers and no optional components.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::runtime::LoopRuntime;
    /// # use loopctl::runtime::FallbackCapable;
    /// let runtime = LoopRuntime::new();
    /// assert!(runtime.fallback().active_model().is_none());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            fallback: FallbackManager::default(),
            detection: DetectionManager::default(),
            observer_host: ObserverHost::new(),
            tool_pipeline: None,
            context_manager: None,
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
        }
    }

    // ==================================================
    // Builder methods
    // ==================================================

    /// Replace the fallback manager with a custom instance.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::runtime::LoopRuntime;
    /// # use loopctl::runtime::FallbackCapable;
    /// # use loopctl::fallback::FallbackManager;
    /// let runtime = LoopRuntime::new()
    ///     .with_fallback(FallbackManager::for_model("llm-70b"));
    /// assert_eq!(runtime.fallback().active_model().as_deref(), Some("llm-70b"));
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
    /// # use loopctl::runtime::LoopRuntime;
    /// # use loopctl::runtime::Detectable;
    /// # use loopctl::detection::{DetectionManager, DetectionConfig};
    /// let config = DetectionConfig {
    ///     loop_threshold: 5,
    ///     ..Default::default()
    /// };
    /// let runtime = LoopRuntime::new()
    ///     .with_detection(DetectionManager::new_with_config(config).unwrap());
    /// assert_eq!(runtime.detection().config().loop_threshold, 5);
    /// ```
    #[must_use]
    pub fn with_detection(mut self, detection: DetectionManager) -> Self {
        self.detection = detection;
        self
    }

    // ==================================================
    // Setters for optional components
    // ==================================================

    /// Register an observer with the observer host.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::observer::LoopObserver;
    /// use std::sync::Arc;
    ///
    /// let mut runtime = LoopRuntime::new();
    /// runtime.register_observer(Arc::new(MyObserver));
    /// ```
    pub fn register_observer(&mut self, observer: Arc<dyn LoopObserver>) {
        self.observer_host.register(observer);
    }

    /// Access the observer host directly.
    pub fn observers(&self) -> &ObserverHost {
        &self.observer_host
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut runtime = LoopRuntime::new();
    /// runtime.set_pipeline(builder.build()?);
    /// ```
    pub fn set_pipeline(&mut self, pipeline: ToolPipeline) {
        self.tool_pipeline = Some(pipeline);
    }

    /// Set the context manager for automatic compaction.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::compact::{ContextManager, TruncatingCompactor};
    /// use std::sync::Arc;
    ///
    /// let compactor = TruncatingCompactor::new()
    ///     .with_preserve_recent(4)
    ///     .with_min_messages(6);
    /// let manager = ContextManager::new(Arc::new(compactor))
    ///     .with_context_window(200_000)
    ///     .with_threshold(0.80);
    ///
    /// let mut runtime = LoopRuntime::new();
    /// runtime.set_context_manager(Arc::new(manager));
    /// ```
    pub fn set_context_manager(&mut self, manager: Arc<ContextManager>) {
        self.context_manager = Some(manager);
    }

    /// Set the stream handler for resilient streaming with retries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
    /// use std::time::Duration;
    ///
    /// let handler = StreamHandler::with_config(
    ///     StreamTimeoutConfig {
    ///         initial_event_timeout: Duration::from_secs(60),
    ///         ..Default::default()
    ///     },
    ///     Default::default(),
    /// );
    ///
    /// let mut runtime = LoopRuntime::new();
    /// runtime.set_stream_handler(handler);
    /// ```
    pub fn set_stream_handler(&mut self, handler: StreamHandler) {
        self.stream_handler = Some(handler);
    }

    /// Set the hook executor for bidirectional lifecycle interception.
    #[cfg(feature = "hooks")]
    pub fn set_hook_executor(&mut self, executor: Arc<HookExecutor>) {
        self.hook_executor = Some(executor);
    }

    /// Set the tool health registry for per-tool health tracking.
    ///
    /// When set, records success/failure counts and latency for every
    /// tool dispatch. Tools that exceed the failure threshold have their
    /// circuit breaker opened, blocking subsequent calls until recovery.
    ///
    /// *Requires `tool_health` feature.*
    #[cfg(feature = "tool_health")]
    pub fn set_health_registry(&mut self, registry: Arc<ToolHealthRegistry>) {
        self.health_registry = Some(registry);
    }

    // ==================================================
    // Lifecycle
    // ==================================================

    /// Reset all managers and observers to their initial state.
    ///
    /// Delegates to each manager's `reset()` method and calls
    /// [`ObserverHost::reset_all`]. Typically called at the start of
    /// a new agent task or session.
    ///
    /// # Example
    ///
    /// ```
    /// # use loopctl::runtime::LoopRuntime;
    /// let runtime = LoopRuntime::new();
    /// // ... after a session ...
    /// runtime.reset_all();
    /// // All managers are back to their initial state
    /// ```
    pub fn reset_all(&self) {
        self.fallback.reset();
        self.detection.reset();
        self.observer_host.reset_all();
    }
}

impl Default for LoopRuntime {
    /// Produce a [`LoopRuntime`] with default managers and no optional components.
    ///
    /// Equivalent to [`LoopRuntime::new`].
    fn default() -> Self {
        Self::new()
    }
}

// ==================================================
// Capability trait implementations
// ==================================================

impl crate::capabilities::Observable for LoopRuntime {
    fn observers(&self) -> &ObserverHost {
        &self.observer_host
    }
}

impl crate::capabilities::Detectable for LoopRuntime {
    fn detection(&self) -> &DetectionManager {
        &self.detection
    }
}

impl crate::capabilities::FallbackCapable for LoopRuntime {
    fn fallback(&self) -> &FallbackManager {
        &self.fallback
    }
}

impl crate::capabilities::Compactable for LoopRuntime {
    fn context_manager(&self) -> Option<&Arc<ContextManager>> {
        self.context_manager.as_ref()
    }
}

impl crate::capabilities::StreamCapable for LoopRuntime {
    fn stream_handler(&self) -> Option<&StreamHandler> {
        self.stream_handler.as_ref()
    }
}

#[cfg(feature = "hooks")]
impl crate::capabilities::Hookable for LoopRuntime {
    fn hook_executor(&self) -> Option<&HookExecutor> {
        self.hook_executor.as_deref()
    }
}

impl crate::capabilities::PipelineAware for LoopRuntime {
    fn pipeline(&self) -> Option<&ToolPipeline> {
        self.tool_pipeline.as_ref()
    }
}

#[cfg(feature = "tool_health")]
impl crate::capabilities::HealthTrackable for LoopRuntime {
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
        let runtime = LoopRuntime::default();
        assert!(runtime.fallback().active_model().is_none());
    }

    #[test]
    fn test_runtime_with_custom_fallback() {
        let fallback = FallbackManager::for_model("my-model");
        let runtime = LoopRuntime::new().with_fallback(fallback);
        assert_eq!(
            runtime.fallback().active_model().as_deref(),
            Some("my-model")
        );
    }

    #[test]
    fn test_reset_all() {
        let runtime = LoopRuntime::new();
        runtime.reset_all();
    }

    #[test]
    fn test_runtime_contains_detection_manager() {
        let runtime = LoopRuntime::new();
        assert_eq!(runtime.detection().config().loop_threshold, 3);
        assert_eq!(runtime.detection().config().stop_threshold, 10);
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
        let runtime = LoopRuntime::new().with_detection(detection);
        assert_eq!(runtime.detection().config().loop_threshold, 7);
        assert_eq!(runtime.detection().config().stop_threshold, 20);
        assert!(runtime.fallback().active_model().is_none());
    }

    #[test]
    fn test_reset_all_clears_detection() {
        let runtime = LoopRuntime::new();
        let _ = runtime.detection().record_tool_call("Read", 12345);
        runtime.reset_all();
        let pattern = runtime.detection().record_tool_call("Read", 12345);
        assert!(matches!(pattern, DetectedPattern::NoPattern));
    }

    #[test]
    fn test_capability_traits_are_object_safe() {
        // Verify that trait objects can be created
        fn _assert_observable(_: &dyn Observable) {}
        fn _assert_detectable(_: &dyn Detectable) {}
        fn _assert_fallback(_: &dyn FallbackCapable) {}
        fn _assert_pipeline(_: &dyn PipelineAware) {}
        fn _assert_compactable(_: &dyn Compactable) {}
        fn _assert_stream_capable(_: &dyn StreamCapable) {}

        let runtime = LoopRuntime::new();
        _assert_observable(&runtime);
        _assert_detectable(&runtime);
        _assert_fallback(&runtime);
        _assert_pipeline(&runtime);
        _assert_compactable(&runtime);
        _assert_stream_capable(&runtime);
    }

    #[test]
    fn test_pipeline_defaults_to_none() {
        let runtime = LoopRuntime::new();
        assert!(runtime.pipeline().is_none());
    }

    #[test]
    fn test_observers_accessible_via_trait() {
        let runtime = LoopRuntime::new();
        let _: &ObserverHost = runtime.observers();
    }

    #[test]
    fn test_context_manager_defaults_to_none() {
        let runtime = LoopRuntime::new();
        assert!(runtime.context_manager().is_none());
    }

    #[test]
    fn test_stream_handler_defaults_to_none() {
        let runtime = LoopRuntime::new();
        assert!(runtime.stream_handler().is_none());
    }

    #[test]
    fn test_set_pipeline_returns_some() {
        use crate::middleware::ToolPipeline;
        use crate::tool::ToolRegistry;
        use std::sync::Arc;

        let registry = Arc::new(ToolRegistry::new());
        let pipeline = ToolPipeline::new(registry);
        let mut runtime = LoopRuntime::new();
        assert!(runtime.pipeline().is_none());
        runtime.set_pipeline(pipeline);
        assert!(runtime.pipeline().is_some());
    }

    #[test]
    fn test_set_context_manager_returns_some() {
        use crate::compact::{ContextManager, TruncatingCompactor};

        let compactor = TruncatingCompactor::new();
        let manager = ContextManager::new(Arc::new(compactor));
        let mut runtime = LoopRuntime::new();
        assert!(runtime.context_manager().is_none());
        runtime.set_context_manager(Arc::new(manager));
        assert!(runtime.context_manager().is_some());
    }

    #[test]
    fn test_set_stream_handler_returns_some() {
        use crate::stream::handler::StreamHandler;

        let handler = StreamHandler::new();
        let mut runtime = LoopRuntime::new();
        assert!(runtime.stream_handler().is_none());
        runtime.set_stream_handler(handler);
        assert!(runtime.stream_handler().is_some());
    }

    #[test]
    fn test_register_observer_increments_count() {
        use crate::observer::LoopObserver;
        use std::sync::Arc;

        struct NopObserver;
        impl LoopObserver for NopObserver {
            fn name(&self) -> &str {
                "NopObserver"
            }
        }

        let mut runtime = LoopRuntime::new();
        assert!(runtime.observers().is_empty());
        runtime.register_observer(Arc::new(NopObserver));
        assert_eq!(runtime.observers().len(), 1);
        runtime.register_observer(Arc::new(NopObserver));
        assert_eq!(runtime.observers().len(), 2);
    }

    #[test]
    fn test_reset_all_clears_fallback() {
        let runtime = LoopRuntime::new();
        let _ = runtime.fallback().record_api_failure();
        runtime.reset_all();
        assert!(runtime.fallback().active_model().is_none());
    }

    #[test]
    fn test_reset_all_clears_observers() {
        use crate::observer::LoopObserver;
        use std::sync::Arc;

        struct NopObserver;
        impl LoopObserver for NopObserver {
            fn name(&self) -> &str {
                "NopObserver"
            }
        }

        let mut runtime = LoopRuntime::new();
        runtime.register_observer(Arc::new(NopObserver));
        assert_eq!(runtime.observers().len(), 1);
        runtime.reset_all();
        // Observer count persists across reset — reset_all calls
        // reset_all on each observer, it doesn't remove them.
        assert_eq!(runtime.observers().len(), 1);
    }

    #[test]
    fn test_generic_bounds_accept_runtime() {
        fn accepts_observable(_: &impl Observable) {}
        fn accepts_detectable(_: &impl Detectable) {}
        fn accepts_fallback(_: &impl FallbackCapable) {}
        fn accepts_compactable(_: &impl Compactable) {}
        fn accepts_stream_capable(_: &impl StreamCapable) {}
        fn accepts_pipeline(_: &impl PipelineAware) {}
        fn accepts_multi_bound(_: &(impl Observable + Detectable + FallbackCapable)) {}

        let runtime = LoopRuntime::new();
        accepts_observable(&runtime);
        accepts_detectable(&runtime);
        accepts_fallback(&runtime);
        accepts_compactable(&runtime);
        accepts_stream_capable(&runtime);
        accepts_pipeline(&runtime);
        accepts_multi_bound(&runtime);
    }
}
