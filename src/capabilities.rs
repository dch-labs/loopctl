//! Capability traits for the agent loop runtime.
//!
//! Each trait represents a single infrastructure capability that the
//! agent loop can depend on. [`LoopRuntime`](crate::runtime::LoopRuntime)
//! implements all of them, but consumers can narrow their bounds to
//! only the capabilities they need.
//!
//! # Traits
//!
//! | Trait | Purpose |
//! |-------|---------|
//! | [`Observable`] | Lifecycle event observation |
//! | [`Detectable`] | Loop and convergence detection |
//! | [`FallbackCapable`] | Model fallback / circuit breaker |
//! | [`Compactable`] | Context compaction |
//! | [`StreamCapable`] | Resilient LLM streaming |
//! | [`Hookable`] | Bidirectional lifecycle hooks |
//! | [`PipelineAware`] | Middleware pipeline dispatch |
//! | [`HealthTrackable`] | Per-tool health monitoring *(requires `tool_health` feature)* |
//!
//! # When to use
//!
//! Use these traits as bounds when you need a specific capability
//! without pulling in the full [`LoopRuntime`](crate::runtime::LoopRuntime):
//!
//! ```rust,ignore
//! fn check_patterns(runtime: &impl Detectable) {
//!     let pattern = runtime.detection().record_tool_call("Read", hash);
//! }
//! ```

use std::sync::Arc;

use crate::compact::ContextManager;
use crate::detection::DetectionManager;
use crate::fallback::FallbackManager;
#[cfg(feature = "hooks")]
use crate::hooks::HookExecutor;
use crate::middleware::ToolPipeline;
use crate::observer::ObserverHost;
use crate::stream::handler::StreamHandler;
#[cfg(feature = "tool_health")]
use crate::tool::health::ToolHealthRegistry;

// ==================================================
// Capability Traits
// ==================================================

/// Capability to emit lifecycle events to registered observers.
///
/// Observers receive read-only notifications at well-defined hook points
/// in the agent loop. They cannot influence control flow — for that, see
/// [`Hookable`].
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to notify observers of lifecycle
/// events but don't need any other infrastructure capabilities.
///
/// ```rust,ignore
/// fn process_turn(runtime: &impl Observable) {
///     runtime.observers().on_turn_start(&ctx);
///     // ... do work ...
///     runtime.observers().on_turn_end(&ctx);
/// }
/// ```
pub trait Observable {
    fn observers(&self) -> &ObserverHost;
}

/// Capability to detect repetitive loops and semantic convergence.
///
/// Loop detection catches when the agent repeats the same tool operations
/// in a cycle. Convergence detection catches when successive assistant
/// responses become semantically similar. Both are handled by
/// [`DetectionManager`].
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to record operations or responses
/// and check whether a loop or convergence pattern has been detected.
///
/// ```rust,ignore
/// fn check_patterns(runtime: &impl Detectable) {
///     let pattern = runtime.detection().record_tool_call("Read", hash);
///     if let DetectedPattern::LoopDetected { .. } = pattern {
///         // intervention needed
///     }
/// }
/// ```
pub trait Detectable {
    fn detection(&self) -> &DetectionManager;
}

/// Capability to fall back to an alternate model when the primary fails.
///
/// Wraps a [`FallbackManager`] that acts as a circuit breaker: after
/// consecutive API failures exceed a threshold, requests are rerouted
/// to a fallback model until the primary stabilises.
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to record API failures or check
/// whether the circuit breaker has tripped.
///
/// ```rust,ignore
/// fn handle_stream_error(runtime: &impl FallbackCapable) {
///     let tripped = runtime.fallback().record_api_failure();
///     if tripped {
///         if let Some(model) = runtime.fallback().fallback_model() {
///             // switch to fallback model
///         }
///     }
/// }
/// ```
pub trait FallbackCapable {
    fn fallback(&self) -> &FallbackManager;
}

/// Capability to compact conversation context when token usage exceeds a threshold.
///
/// When a [`ContextManager`] is configured, the loop checks token usage
/// after each turn and triggers compaction when usage exceeds the
/// configured threshold. Compaction replaces conversation messages with
/// a compressed version, preserving the most recent context.
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to inspect or trigger context
/// compaction during the agent loop. Useful for custom loop
/// implementations that need to manage the context window directly.
pub trait Compactable {
    fn context_manager(&self) -> Option<&Arc<ContextManager>>;
}

/// Capability to stream LLM responses with retry, timeout, and fallback.
///
/// When a [`StreamHandler`] is configured, the loop delegates streaming
/// to it instead of using the basic inline logic. The handler provides
/// automatic retries, per-event timeouts, and fallback to non-streaming
/// mode.
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to access the stream handler for
/// resilient streaming. Useful for custom loop implementations
/// that need to control streaming behaviour (timeouts, retries, fallback
/// to non-streaming mode).
pub trait StreamCapable {
    fn stream_handler(&self) -> Option<&StreamHandler>;
}

/// Capability to run bidirectional hooks that can block actions.
///
/// Hooks differ from observers ([`Observable`]) in that they return
/// [`HookAction`](crate::hooks::HookAction) to control whether an action
/// proceeds. The executor stops at the first hook that returns a blocking
/// result.
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to run hooks before or after
/// tool dispatch, compaction, or session start/end.
#[cfg(feature = "hooks")]
pub trait Hookable {
    fn hook_executor(&self) -> Option<&HookExecutor>;
}

/// Capability to dispatch tools through a middleware pipeline.
///
/// When a pipeline is configured, tool calls flow through middleware
/// layers (timeouts, output limiting, etc.) before reaching the registry.
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to dispatch a tool call through
/// optional middleware.
///
/// ```rust,ignore
/// async fn dispatch(runtime: &impl PipelineAware, ctx: ToolDispatchContext) {
///     if let Some(pipeline) = runtime.pipeline() {
///         pipeline.invoke(ctx).await
///     } else {
///         // direct dispatch
///     }
/// }
/// ```
pub trait PipelineAware {
    fn pipeline(&self) -> Option<&ToolPipeline>;
}

/// Capability to track per-tool health with circuit breakers.
///
/// Records success/failure and latency for every tool dispatch.
/// Tools that exceed the failure threshold have their circuit breaker
/// opened, blocking subsequent calls until recovery.
///
/// *Requires `tool_health` feature.*
///
/// # Implementors
///
/// - [`LoopRuntime`](crate::runtime::LoopRuntime) — the framework's default implementation.
#[cfg(feature = "tool_health")]
pub trait HealthTrackable {
    fn health_registry(&self) -> Option<&ToolHealthRegistry>;
}
