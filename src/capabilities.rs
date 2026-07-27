//! Capability traits for the agent loop runtime.
//!
//! Each trait represents a single infrastructure capability that the
//! agent loop can depend on. [`LoopManagers`](crate::managers::LoopManagers)
//! implements all of them, but consumers can narrow their bounds to
//! only the capabilities they need.
//!
//! # Traits
//!
//! | Trait               | Purpose |
//! |---------------------|---------------------------------------------------------------|
//! | [`Observable`]      | Lifecycle event observation                                   |
//! | [`Detectable`]      | Loop and convergence detection                                |
//! | [`FallbackCapable`] | Model fallback / circuit breaker                              |
//! | [`Compactable`]     | Context compaction                                            |
//! | [`StreamCapable`]   | Resilient LLM streaming                                       |
//! | [`PipelineAware`]   | Middleware pipeline dispatch                                  |
//! | `Hookable`          | Bidirectional lifecycle hooks *(requires `hooks` feature)*    |
//! | `HealthTrackable`   | Per-tool health monitoring *(requires `tool_health` feature)* |
//!
//! # When to use
//!
//! Use these traits as bounds when you need a specific capability
//! without pulling in the full [`LoopManagers`](crate::managers::LoopManagers):
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

/// Capability to emit lifecycle events to registered observers.
///
/// Observers receive read-only notifications at well-defined hook points
/// in the agent loop. They cannot influence control flow — for that, see
/// the `Hookable` trait *(requires `hooks` feature)*.
///
/// # Implementors
///
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to notify observers of lifecycle
/// events but don't need any other infrastructure capabilities.
///
/// ```rust,ignore
/// fn run_step(runtime: &impl Observable) {
///     runtime.observers().on_turn_start(&ctx);
///     // ... do work ...
///     runtime.observers().on_turn_end(&ctx);
/// }
/// ```
pub trait Observable {
    /// Returns the observer host for lifecycle event fan-out.
    ///
    /// The observer host holds every registered observer and dispatches
    /// all lifecycle events (turn start/end, stream deltas, tool dispatch,
    /// compaction) to them.
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
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
    /// Returns the detection manager for loop and convergence detection.
    ///
    /// Use this to record tool calls or model responses and check whether
    /// the agent is repeating itself or converging on a stable answer.
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
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
    /// Returns the fallback manager (circuit breaker) for API model fallback.
    ///
    /// Use this to record API failures and check whether the circuit
    /// breaker has tripped, indicating the primary model is unavailable.
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to inspect or trigger context
/// compaction during the agent loop. Useful for custom loop
/// implementations that need to manage the context window directly.
pub trait Compactable {
    /// Returns the context manager, if compaction is configured.
    ///
    /// Returns `None` when no compactor is set — the loop will not
    /// auto-compact, and the host must manage context size manually.
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to access the stream handler for
/// resilient streaming. Useful for custom loop implementations
/// that need to control streaming behaviour (timeouts, retries, fallback
/// to non-streaming mode).
pub trait StreamCapable {
    /// Returns the stream handler.
    ///
    /// Always returns a handler — when no resilient handler is configured,
    /// returns a shared reference to [`StreamHandler::passthrough_default`]
    /// (a no-resilience handler that yields the raw provider stream with no
    /// retries, timeouts, or fallback). The engine's `stream_turn` always
    /// routes through a handler; this never returns `None`.
    fn stream_handler(&self) -> &StreamHandler;
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
///
/// # When to use
///
/// Use this trait bound when you need to run hooks before or after
/// tool dispatch, compaction, or session start/end.
#[cfg(feature = "hooks")]
pub trait Hookable {
    /// Returns the hook executor, if hooks are configured.
    ///
    /// Returns `None` when no hook executor is set — no pre/post hooks
    /// will fire. Hooks can approve or reject actions, unlike observers.
    fn hook_executor(&self) -> Option<&HookExecutor>;
}

/// Capability to dispatch tools through a middleware pipeline.
///
/// When a pipeline is configured, tool calls flow through middleware
/// layers (timeouts, output limiting, etc.) before reaching the registry.
///
/// # Implementors
///
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
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
    /// Returns the tool middleware pipeline, if configured.
    ///
    /// Returns `None` when no pipeline is set — tool calls go directly
    /// to the registry. When set, every tool call passes through the
    /// pipeline's middleware layers before reaching the tool.
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
/// - [`LoopManagers`](crate::managers::LoopManagers) — the framework's default implementation.
#[cfg(feature = "tool_health")]
pub trait HealthTrackable {
    /// Returns the tool health registry, if health tracking is configured.
    ///
    /// Returns `None` when no health registry is set — per-tool circuit
    /// breakers will not fire. When set, records success/failure counts
    /// and opens breakers for unhealthy tools.
    fn health_registry(&self) -> Option<&ToolHealthRegistry>;
}
