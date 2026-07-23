//! `BareLoop` — the framework's default agent loop implementation.
//!
//! [`BareLoop`] orchestrates the full lifecycle of an LLM-based agent session:
//! sending messages to an LLM API, accumulating streaming responses,
//! dispatching tool calls, and feeding results back into the conversation
//! until the model ends its turn or a configured limit is reached.
//!
//! # Architecture
//!
//! [`BareLoop`] ties together four key components:
//!
//! - An [`ApiClient`](crate::api::ApiClient) for communicating with
//!   the LLM provider.
//! - A [`ToolRegistry`](crate::tool::ToolRegistry) for dispatching tool
//!   calls the model requests.
//! - An [`LoopConfig`] governing session parameters (max turns, system
//!   prompt, session ID).
//! - Optional [`LoopObserver`](crate::observer::LoopObserver) registrations for lifecycle instrumentation.
//!
//! # Key Design Decisions
//!
//! - **Static dispatch** — `BareLoop<C>` is generic over the
//!   [`ApiClient`](crate::api::ApiClient) type parameter `C`,
//!   avoiding `dyn` overhead for the hot path.
//! - **Sequential tool dispatch** — tools within a single turn are
//!   executed one after another.
//! - **Soft tool errors** — when a tool is not found or returns an error,
//!   the loop records the error as a tool result and continues, letting
//!   the model decide how to recover. Only hard errors (API failures,
//!   max-turns exceeded, cancellation) terminate the session.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::engine::BareLoop;
//! use loopctl::tool::ToolRegistry;
//! use loopctl::config::LoopConfig;
//! use std::sync::Arc;
//!
//! // 1. Build components
//! let client = Arc::new(my_api_client);
//! let registry = ToolRegistry::new();
//! let config = LoopConfig::default();
//!
//! // 2. Create the loop
//! let mut agent = BareLoop::new(client, registry, config);
//!
//! // 3. Run
//! let result = agent.run("Hello, agent!").await?;
//! println!("Agent responded in {} turns", result.total_turns);
//! ```

use crate::api::ApiClient;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cancel::CancelSignal;
use crate::compact::ContextManager;
use crate::config::LoopConfig;

use crate::error::LoopError;

use crate::engine::loop_core::{LoopState, SessionResult, StopReason, ToolCall, TurnResult};
use crate::engine::{ContextContributor, ContributorContext};
#[cfg(all(test, feature = "hooks"))]
use crate::hooks::Hook;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    CompactTrigger, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
};
#[cfg(all(test, feature = "hooks"))]
use crate::hooks::context::{SessionEndContext as HookSessionEndContext, SessionEndReason};
#[cfg(feature = "hooks")]
use crate::hooks::{HookAction, HookExecutor};
use crate::message::{Message, MessagePart, Role, ToolContent};
use crate::middleware::{ToolDispatchContext, ToolPipeline, ToolPipelineBuilder};
use crate::observer::{
    FallbackContext, ModelSwitchedContext, ResponseContext, StreamContext, StreamFailureContext,
    ToolCallReceivedContext, TurnEndContext, TurnStartContext,
};
use crate::reflection::{
    ExponentialBackoffRecovery, NoopReflector, RecoveryAction, RecoveryStrategy, ReflectionContext,
    Reflector,
};
use crate::runtime::LoopRuntime;
use crate::stream::handler::StreamHandler;
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
use crate::structured::RequestOptions;
#[cfg(feature = "tool_health")]
use crate::tool::health::ToolHealthRegistry;
use crate::tool::{PermissionCheck, ToolContext, ToolDispatchResult, ToolRegistry, ToolSchema};

mod compact;
mod dispatch;
mod emission;
mod message;
mod stream;

// ==================================================
// BareLoop
// ==================================================

/// The framework's default agent loop implementation.
///
/// `BareLoop` ties together an [`ApiClient`], [`ToolRegistry`],
/// configuration, and optional observers into a complete agent loop.
/// It handles the full lifecycle: streaming responses, tool dispatch,
/// conversation management, and termination.
///
/// # Type Parameters
///
/// - `C` — The concrete [`ApiClient`] implementation. Uses static dispatch
///   (generics) for zero-cost abstraction. Wrap in `Arc` if you need
///   shared ownership.
///
/// # Construction
///
/// Use one of the constructors based on what components you have:
///
/// - [`new()`](BareLoop::new) — client + tools + config.
/// - [`new_with_managers()`](BareLoop::new_with_managers) — full control,
///   including a [`LoopRuntime`].
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::engine::BareLoop;
/// use loopctl::tool::ToolRegistry;
/// use loopctl::config::LoopConfig;
/// use std::sync::Arc;
///
/// let registry = ToolRegistry::new();
/// let config = LoopConfig::default();
///
/// let mut agent = BareLoop::new(
///     Arc::new(my_client),
///     registry,
///     config,
/// );
///
/// let result = agent.run("Hello, agent!").await?;
/// println!("Agent responded in {} turns", result.total_turns);
/// ```
pub struct BareLoop<C: ApiClient> {
    /// The LLM API client used to send conversation turns.
    ///
    /// Generic over the concrete [`ApiClient`] implementation. Wrapped
    /// in `Arc` for shared ownership across components.
    client: Arc<C>,

    /// Registered tools available to the agent.
    ///
    /// When the model emits a tool-call part, the loop looks up the tool
    /// by name in this registry and invokes it.
    tools: Arc<ToolRegistry>,

    /// Session parameters (max turns, model, system prompt).
    ///
    /// See [`LoopConfig`] for the full set of options.
    config: LoopConfig,

    /// Conversation history (system + user + assistant + tool results).
    ///
    /// Grows over the session lifetime. Each call to [`run()`](crate::engine::loop_core::Loop::run)
    /// appends the user message, then alternates between assistant responses
    /// and tool-result messages until the model signals `end_turn`.
    conversation: Vec<Message>,

    /// Framework runtime bundle — holds all cross-cutting infrastructure.
    ///
    /// This is the single source of truth for:
    ///
    /// - [`FallbackManager`] — circuit breaker for API model fallback.
    /// - [`DetectionManager`] — loop and convergence detection.
    /// - [`ObserverHost`] — lifecycle event fan-out.
    /// - Optional [`ToolPipeline`] — middleware pipeline for tool dispatch.
    /// - Optional [`ContextManager`] — automatic context compaction.
    /// - Optional [`StreamHandler`] — resilient streaming with retries.
    /// - Optional [`HookExecutor`] — bidirectional lifecycle hooks.
    /// - Optional [`ToolHealthRegistry`] — per-tool health tracking.
    ///
    /// Reset at the start of every session via [`LoopRuntime::reset_all`].
    ///
    /// [`FallbackManager`]: crate::fallback::FallbackManager
    /// [`DetectionManager`]: crate::detection::DetectionManager
    /// [`ObserverHost`]: crate::observer::ObserverHost
    /// [`ToolPipeline`]: crate::middleware::ToolPipeline
    /// [`ContextManager`]: crate::compact::ContextManager
    /// [`StreamHandler`]: crate::stream::handler::StreamHandler
    /// [`HookExecutor`]: crate::hooks::HookExecutor
    /// [`ToolHealthRegistry`]: crate::tool::health::ToolHealthRegistry
    managers: LoopRuntime,

    /// Failure analyser for tool errors.
    ///
    /// When a tool call returns an error, the reflector analyses the
    /// failure and produces a [`FailureAnalysis`] describing
    /// recoverability, severity, and optional corrections.
    reflector: Arc<dyn Reflector>,

    /// Recovery policy for tool errors.
    ///
    /// Takes the [`FailureAnalysis`] and decides on a [`RecoveryAction`]
    /// (retry, skip, ask user, or fail). Defaults to
    /// [`ExponentialBackoffRecovery`] with 3 retries.
    recovery: Arc<dyn RecoveryStrategy>,

    /// Shared cancellation signal.
    ///
    /// Set via [`cancel()`](BareLoop::cancel). Checked at the top of every
    /// loop iteration (before streaming) and between tool dispatches.
    /// Streaming is also cancel-aware via `tokio::select!` — the loop
    /// will wake up mid-stream when cancelled.
    cancelled: Arc<CancelSignal>,

    /// Current lifecycle state, exposed via the [`Loop`](crate::engine::loop_core::Loop) trait.
    ///
    /// Drives the engine's state machine
    /// (`Idle → Processing → WaitingForTool → Processing → … → Completed` /
    /// `Failed` / `Cancelled`, plus `Compacting`/`Reflecting` side-states).
    /// Set throughout the turn loop and read by [`finalize`](crate::engine::loop_core::Loop::finalize)
    /// to decide success vs. failure and to populate the
    /// [`SessionResult`](crate::engine::loop_core::SessionResult) fields; also
    /// returned from [`state`](crate::engine::loop_core::Loop::state) for
    /// outside observers. The [`Idle`](LoopState::Idle) variant additionally
    /// gates configuration setters via [`debug_assert_idle`](Self::debug_assert_idle).
    state: LoopState,

    /// Session-level accumulator for turn counts, token usage, and tool calls.
    ///
    /// Reused across turns in a single [`run()`](crate::engine::loop_core::Loop::run) call. Reset
    /// to `SessionResult::default()` in [`initialize`](crate::engine::loop_core::Loop::initialize).
    budget: SessionResult,

    /// Session start time, set by [`initialize`](crate::engine::loop_core::Loop::initialize).
    ///
    /// `None` only before the first `initialize()` call; `Some` thereafter.
    /// Read in [`finalize`](crate::engine::loop_core::Loop::finalize) to
    /// compute [`SessionResult::total_duration`](crate::engine::loop_core::SessionResult)
    /// via `elapsed()`. Captured once per session (not per turn) so the
    /// reported duration is the wall-clock lifetime of the whole session,
    /// not a single turn.
    session_start: Option<Instant>,

    /// Optional callback invoked for each text delta during streaming.
    ///
    /// Set via [`set_text_streamer`](BareLoop::set_text_streamer).
    /// When set, called from `stream_turn` on every `IndexedDelta` with
    /// a `Text` payload, enabling real-time token display.
    #[allow(clippy::type_complexity)]
    text_streamer: Option<Arc<dyn Fn(&str) + Send + Sync>>,

    /// Turn-boundary context contributors.
    ///
    /// Each registered contributor is consulted at the top of every turn
    /// (after [`on_turn_start`](crate::observer::LoopObserver::on_turn_start),
    /// before the model call); any message it returns is appended to the
    /// conversation in registration order. Register via
    /// [`add_contributor`](BareLoop::add_contributor).
    contributors: Vec<Box<dyn ContextContributor>>,

    /// Per-turn `RequestOptions` applied to every provider call.
    ///
    /// Carries `tool_constraint` (strict/grammar-constrained tool-call
    /// decoding). Default is [`RequestOptions::default`], which reproduces
    /// the prior (unconstrained) behavior. Set via
    /// [`set_request_options`](BareLoop::set_request_options).
    request_options: RequestOptions,
}

// ==================================================
// Run-loop helpers
// ==================================================

impl<C: ApiClient> BareLoop<C> {
    /// Maximum retry attempts for tool recovery before giving up.
    ///
    /// This is the engine-level safety ceiling passed to the
    /// [`RecoveryStrategy`](crate::reflection::RecoveryStrategy) as
    /// `max_attempts`. The strategy's own `max_retries` limit (typically
    /// stricter) is the effective limit; this constant prevents a
    /// misconfigured strategy from retrying indefinitely.
    const MAX_RECOVERY_ATTEMPTS: u32 = 5;

    /// Create a new `BareLoop` with the given components.
    ///
    /// Initializes an empty conversation history and a
    /// fresh [`LoopRuntime`]. The cancellation signal starts as non-cancelled.
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client, wrapped in `Arc`.
    /// - `tools` — The [`ToolRegistry`] containing available tools.
    /// - `config` — Session parameters (max turns, system prompt, etc.).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(
    ///     Arc::new(my_client),
    ///     ToolRegistry::new(),
    ///     LoopConfig::default(),
    /// );
    /// ```
    pub fn new(client: Arc<C>, tools: ToolRegistry, config: LoopConfig) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            config,
            conversation: Vec::new(),
            managers: LoopRuntime::new(),
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            state: LoopState::Idle,
            budget: SessionResult::default(),
            session_start: None,
            text_streamer: None,
            contributors: Vec::new(),
            request_options: RequestOptions::default(),
        }
    }

    /// Create a new `BareLoop` with all components including managers.
    ///
    /// Use this constructor when you need to supply a pre-configured
    /// [`LoopRuntime`] — for example, to enable loop detection or
    /// circuit-breaker policies.
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client, wrapped in `Arc`.
    /// - `tools` — The [`ToolRegistry`] containing available tools.
    /// - `config` — Session parameters.
    /// - `managers` — A pre-built [`LoopRuntime`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = LoopRuntime::builder()
    ///     .with_detection(DetectionManager::default())
    ///     .with_fallback(FallbackManager::default())
    ///     .build();
    ///
    /// let mut agent = BareLoop::new_with_managers(
    ///     Arc::new(my_client),
    ///     registry,
    ///     config,
    ///     managers,
    /// );
    /// ```
    pub fn new_with_managers(
        client: Arc<C>,
        tools: ToolRegistry,
        config: LoopConfig,
        managers: LoopRuntime,
    ) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            config,
            conversation: Vec::new(),
            managers,
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            state: LoopState::Idle,
            budget: SessionResult::default(),
            session_start: None,
            text_streamer: None,
            contributors: Vec::new(),
            request_options: RequestOptions::default(),
        }
    }

    // ==================================================
    // Accessors
    // ==================================================

    /// Get the conversation history.
    ///
    /// Returns a slice of [`Message`] representing the full conversation
    /// so far: system prompt (if applied), user messages, assistant
    /// responses, and tool-result messages.
    pub fn conversation(&self) -> &[Message] {
        &self.conversation
    }

    /// Get the agent configuration.
    ///
    /// Returns a reference to the [`LoopConfig`] that governs session
    /// parameters such as max turns, system prompt, and session ID.
    /// The config is immutable for the lifetime of the loop.
    pub fn config(&self) -> &LoopConfig {
        &self.config
    }

    /// Get the tool registry.
    ///
    /// Returns a reference to the [`ToolRegistry`] containing all tools
    /// available to the agent. Tools are looked up by name during
    /// `dispatch_tools()`.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// Check if the loop has been cancelled.
    ///
    /// Returns `true` if [`cancel()`](BareLoop::cancel) was called
    /// by any task holding a clone of the [`CancelSignal`].
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    /// Cancel the agent loop.
    ///
    /// Fires the shared [`CancelSignal`], which sets the internal flag
    /// and wakes any task awaiting cancellation via `tokio::select!`.
    ///
    /// Cancellation is cooperative. Check points are:
    ///
    /// - Top of the main loop (before streaming starts)
    /// - Between individual tool dispatches
    /// - During streaming (via `tokio::select!` with `CancelSignal::notified()`)
    ///
    /// **Not** cancellable during:
    ///
    /// - Inside a running tool invocation
    pub fn cancel(&self) {
        self.cancelled.cancel();
    }

    /// Get the cancellation signal for external monitoring.
    ///
    /// Returns a clone of the `Arc<CancelSignal>` so callers can poll
    /// cancellation state from another task or thread without needing
    /// a reference to the loop itself.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let signal = agent.cancel_signal();
    ///
    /// // From another task:
    /// signal.cancel();
    /// ```
    pub fn cancel_signal(&self) -> Arc<CancelSignal> {
        Arc::clone(&self.cancelled)
    }

    // ==================================================
    // Dependency setters
    // ==================================================

    /// Assert that the loop has not started running yet.
    ///
    /// Configuration setters must be called before [`run()`](crate::engine::loop_core::Loop::run).
    /// Calling them during a running session is a logic bug — the new value
    /// takes effect immediately but parts of the session may have already
    /// been initialised with the old value, leading to subtle inconsistencies.
    ///
    /// This check is only active in debug builds (`debug_assertions`).
    #[inline]
    fn debug_assert_idle(&self) {
        debug_assert!(
            matches!(self.state, LoopState::Idle),
            "BareLoop configuration setters must be called before run() — \
             current state is {:?}, expected Idle",
            self.state
        );
    }

    /// Set the [`Reflector`] for tool-error analysis.
    ///
    /// Replaces the default [`NoopReflector`] with a caller-supplied
    /// implementation. Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., when [`state`](LoopState) is not [`Idle`](LoopState::Idle)).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_reflector(Arc::new(MyReflector));
    /// ```
    pub fn set_reflector(&mut self, reflector: Arc<dyn Reflector>) {
        self.debug_assert_idle();
        self.reflector = reflector;
    }

    /// Set the [`RecoveryStrategy`] for tool-error recovery.
    ///
    /// Replaces the default [`ExponentialBackoffRecovery`] with a
    /// caller-supplied implementation. Must be called before
    /// [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_recovery_strategy(Arc::new(MyStrategy));
    /// ```
    pub fn set_recovery_strategy(&mut self, strategy: Arc<dyn RecoveryStrategy>) {
        self.debug_assert_idle();
        self.recovery = strategy;
    }

    /// Set the [`ContextManager`] for automatic context compaction.
    ///
    /// When set, the loop checks token usage after each turn and
    /// triggers compaction when usage exceeds the configured threshold.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
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
    ///     .with_threshold(8_000);
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_context_manager(Arc::new(manager));
    /// ```
    pub fn set_context_manager(&mut self, manager: Arc<ContextManager>) {
        self.debug_assert_idle();
        self.managers.set_context_manager(manager);
    }

    /// Set the [`StreamHandler`] for resilient streaming with retries,
    /// timeouts, and fallback to non-streaming.
    ///
    /// When set, the loop delegates streaming to the handler instead of
    /// using the inline streaming logic. Must be called before
    /// [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::stream::handler::{StreamHandler, StreamTimeoutConfig};
    ///
    /// let handler = StreamHandler::with_config(
    ///     StreamTimeoutConfig {
    ///         initial_event_timeout: Duration::from_secs(60),
    ///         ..Default::default()
    ///     },
    ///     Default::default(),
    /// );
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_stream_handler(handler);
    /// ```
    pub fn set_stream_handler(&mut self, handler: StreamHandler) {
        self.debug_assert_idle();
        self.managers.set_stream_handler(handler);
    }

    /// Set the [`HookExecutor`] for lifecycle interception.
    ///
    /// When set, the executor runs registered hooks before and after
    /// tool dispatch, compaction, and session start/end. Hooks can
    /// short-circuit with [`HookAction::Block`].
    /// [`HookAction::Ask`] is automatically downgraded to `Block` by the
    /// executor in [`crate::hooks::Interactivity::Headless`] mode (the default).
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// *Requires `hooks` feature.*
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::hooks::HookExecutor;
    /// use std::sync::Arc;
    ///
    /// let executor = HookExecutor::new();
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_hook_executor(Arc::new(executor));
    /// ```
    #[cfg(feature = "hooks")]
    pub fn set_hook_executor(&mut self, executor: Arc<HookExecutor>) {
        self.debug_assert_idle();
        self.managers.set_hook_executor(executor);
    }

    /// Set the [`ToolHealthRegistry`] for per-tool health tracking.
    ///
    /// When set, records success/failure and latency for every tool
    /// dispatch. Tools that exceed the failure threshold have their
    /// circuit breaker opened, blocking subsequent calls until recovery.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// *Requires `tool_health` feature.*
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::tool::health::ToolHealthRegistry;
    /// use std::sync::Arc;
    ///
    /// let registry = ToolHealthRegistry::new();
    /// let mut agent = BareLoop::new(client, tools, config);
    /// agent.set_health_registry(Arc::new(registry));
    /// ```
    #[cfg(feature = "tool_health")]
    pub fn set_health_registry(&mut self, registry: Arc<ToolHealthRegistry>) {
        self.debug_assert_idle();
        self.managers.set_health_registry(registry);
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// Replaces the default (no pipeline) with a caller-supplied
    /// [`ToolPipeline`]. When set, tool calls flow through the
    /// pipeline's middleware chain before reaching the registry.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// Build the pipeline using [`ToolPipeline::builder()`], adding middleware
    /// layers **without** calling `.with_core()` — the registry is injected
    /// automatically from `self.tools` so that schema generation and dispatch
    /// always share the same underlying registry:
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::engine::middleware::{ToolPipeline, TimeoutMiddleware};
    ///
    /// let builder = ToolPipeline::builder()
    ///     .with_middleware(TimeoutMiddleware::from_secs(30));
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_pipeline(builder)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Config`] if the builder fails to produce a valid
    /// pipeline (e.g. internal invariant violated).
    pub fn set_pipeline(&mut self, builder: ToolPipelineBuilder) -> Result<(), LoopError> {
        self.debug_assert_idle();
        let pipeline = builder
            .with_core(Arc::clone(&self.tools))
            .build()
            .map_err(|e| LoopError::Config(e.to_string()))?;
        self.managers.set_pipeline(pipeline);
        Ok(())
    }

    /// Register a [`LoopObserver`](crate::observer::LoopObserver) with the manager bundle's observer host.
    ///
    /// Plugins are called at lifecycle hook points inside the agent loop,
    /// in registration order. See [`LoopObserver`](crate::observer::LoopObserver)
    /// for the trait definition and available hooks.
    ///
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::observer::LoopObserver;
    /// use std::sync::Arc;
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.register_observer(Arc::new(MyObserver));
    /// ```
    pub fn register_observer(&mut self, observer: Arc<dyn crate::observer::LoopObserver>) {
        self.debug_assert_idle();
        self.managers.register_observer(observer);
    }

    /// Set a real-time text streaming callback.
    ///
    /// The callback is invoked for each text delta token as it arrives
    /// from the API during [`run`](crate::engine::loop_core::Loop::run).
    /// This enables real-time display of the model's output without
    /// waiting for the full turn to complete.
    ///
    /// The callback receives a `&str` containing the delta text fragment.
    /// It must be `Send + Sync` as it may be called from an async context.
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::{Arc, Mutex};
    ///
    /// let buffer = Arc::new(Mutex::new(String::new()));
    /// let buf = Arc::clone(&buffer);
    /// agent.set_text_streamer(Arc::new(move |delta| {
    ///     print!("{delta}");
    ///     buf.lock().unwrap_or_else(|e| e.into_inner()).push_str(delta);
    /// }));
    /// ```
    pub fn set_text_streamer(&mut self, f: Arc<dyn Fn(&str) + Send + Sync>) {
        self.debug_assert_idle();
        self.text_streamer = Some(f);
    }

    /// Register a [`ContextContributor`] consulted at the top of every turn.
    ///
    /// Contributors are consulted in registration order after
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) and
    /// before the model call. Each contributor that returns [`Some`] message
    /// has that message appended to the conversation (in registration order)
    /// so it reaches the model this turn and persists into later turns subject
    /// to compaction.
    ///
    /// With no contributors registered, the loop behaves identically to a loop
    /// built without any — the turn-top consultation is a single cheap branch.
    ///
    /// Must be called before
    /// [`run`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., when [`state`](LoopState) is not [`Idle`](LoopState::Idle)).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.add_contributor(Box::new(GoalReminder::new("ship the demo")));
    /// ```
    pub fn add_contributor(&mut self, contributor: Box<dyn ContextContributor>) {
        self.debug_assert_idle();
        self.contributors.push(contributor);
    }

    /// Set the per-turn [`RequestOptions`] applied to every provider call.
    ///
    /// Carries [`tool_constraint`](crate::structured::ToolConstraint) — set to
    /// [`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)
    /// for strict tool-call decoding (small-model reliability), or leave at the
    /// default ([`RequestOptions::default`]) for unconstrained behavior.
    ///
    /// Must be called before
    /// [`run`](crate::engine::loop_core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., when [`state`](LoopState) is not [`Idle`](LoopState::Idle)).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::structured::{RequestOptions, ToolConstraint};
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_request_options(
    ///     RequestOptions::new().with_tool_constraint(ToolConstraint::Strict),
    /// );
    /// ```
    pub fn set_request_options(&mut self, options: RequestOptions) {
        self.debug_assert_idle();
        self.request_options = options;
    }

    /// Set the reflector, consuming `self`. Fluent mirror of
    /// [`set_reflector`](BareLoop::set_reflector).
    #[must_use]
    pub fn with_reflector(mut self, reflector: Arc<dyn Reflector>) -> Self {
        self.set_reflector(reflector);
        self
    }

    /// Set the recovery strategy, consuming `self`. Fluent mirror of
    /// [`set_recovery_strategy`](BareLoop::set_recovery_strategy).
    #[must_use]
    pub fn with_recovery_strategy(mut self, strategy: Arc<dyn RecoveryStrategy>) -> Self {
        self.set_recovery_strategy(strategy);
        self
    }

    /// Set the context manager, consuming `self`. Fluent mirror of
    /// [`set_context_manager`](BareLoop::set_context_manager).
    #[must_use]
    pub fn with_context_manager(mut self, manager: Arc<ContextManager>) -> Self {
        self.set_context_manager(manager);
        self
    }

    /// Set the stream handler, consuming `self`. Fluent mirror of
    /// [`set_stream_handler`](BareLoop::set_stream_handler).
    #[must_use]
    pub fn with_stream_handler(mut self, handler: StreamHandler) -> Self {
        self.set_stream_handler(handler);
        self
    }

    /// Set the hook executor, consuming `self`. Fluent mirror of
    /// [`set_hook_executor`](BareLoop::set_hook_executor).
    ///
    /// *Requires `hooks` feature.*
    #[cfg(feature = "hooks")]
    #[must_use]
    pub fn with_hook_executor(mut self, executor: Arc<HookExecutor>) -> Self {
        self.set_hook_executor(executor);
        self
    }

    /// Set the tool health registry, consuming `self`. Fluent mirror of
    /// [`set_health_registry`](BareLoop::set_health_registry).
    ///
    /// *Requires `tool_health` feature.*
    #[cfg(feature = "tool_health")]
    #[must_use]
    pub fn with_health_registry(mut self, registry: Arc<ToolHealthRegistry>) -> Self {
        self.set_health_registry(registry);
        self
    }

    /// Set the middleware pipeline, consuming `self`. Fluent mirror of
    /// [`set_pipeline`](BareLoop::set_pipeline).
    ///
    /// Because building the pipeline can fail, this returns `Result<Self,
    /// LoopError>` — chain it with `?`.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Config`] if the builder fails to produce a valid
    /// pipeline. See [`set_pipeline`](BareLoop::set_pipeline).
    pub fn with_pipeline(mut self, builder: ToolPipelineBuilder) -> Result<Self, LoopError> {
        self.set_pipeline(builder)?;
        Ok(self)
    }

    /// Register an observer, consuming `self`. Fluent mirror of
    /// [`register_observer`](BareLoop::register_observer).
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<dyn crate::observer::LoopObserver>) -> Self {
        self.register_observer(observer);
        self
    }

    /// Set the real-time text streaming callback, consuming `self`. Fluent
    /// mirror of [`set_text_streamer`](BareLoop::set_text_streamer).
    #[must_use]
    pub fn with_text_streamer(mut self, f: Arc<dyn Fn(&str) + Send + Sync>) -> Self {
        self.set_text_streamer(f);
        self
    }

    /// Register a context contributor, consuming `self`. Fluent mirror of
    /// [`add_contributor`](BareLoop::add_contributor).
    #[must_use]
    pub fn with_contributor(mut self, contributor: Box<dyn ContextContributor>) -> Self {
        self.add_contributor(contributor);
        self
    }

    /// Set the per-turn request options, consuming `self`. Fluent mirror of
    /// [`set_request_options`](BareLoop::set_request_options).
    #[must_use]
    pub fn with_request_options(mut self, options: RequestOptions) -> Self {
        self.set_request_options(options);
        self
    }

    /// Begin a model switch operation.
    ///
    /// Returns a [`ModelSwitch`] builder that lets you optionally update
    /// the context window and max tokens before calling `.apply()`.
    ///
    /// This is the preferred way to switch models when the new model has
    /// a different context window or token limit:
    ///
    /// ```rust,ignore
    /// # use loopctl::engine::BareLoop;
    /// # use loopctl::config::LoopConfig;
    /// # use loopctl::tool::registry::ToolRegistry;
    /// # use loopctl::testing::MockApiClient;
    /// # let client = std::sync::Arc::new(MockApiClient::new("model-a"));
    /// # let tools = ToolRegistry::new();
    /// # let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());
    /// loop_.switch_model("model-b").with_context_window(8192).apply().unwrap();
    /// assert_eq!(loop_.config().model, "model-b");
    /// assert_eq!(loop_.config().context_window, 8192);
    /// ```
    ///
    /// For simple cases where you just want to swap the model name:
    ///
    /// ```rust,ignore
    /// # use loopctl::engine::BareLoop;
    /// # use loopctl::config::LoopConfig;
    /// # use loopctl::tool::registry::ToolRegistry;
    /// # use loopctl::testing::MockApiClient;
    /// # let client = std::sync::Arc::new(MockApiClient::new("a"));
    /// # let tools = ToolRegistry::new();
    /// # let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());
    /// loop_.switch_model("b").apply().unwrap();
    /// ```
    pub fn switch_model(&mut self, model: &str) -> ModelSwitch<'_, C> {
        ModelSwitch {
            loop_: self,
            target_model: model.to_string(),
            context_window: None,
            max_tokens: None,
        }
    }

    // ==================================================
    // Run helpers
    // ==================================================

    /// Pull per-turn `(input_tokens, output_tokens)` from optional [`Usage`].
    ///
    /// Returns `(0, 0)` when the provider did not report usage for the turn.
    fn usage_tokens(usage: Option<&Usage>) -> (u64, u64) {
        match usage {
            Some(u) => (u64::from(u.input_tokens), u64::from(u.output_tokens)),
            None => (0, 0),
        }
    }

    /// Dispatch a batch of tool calls, append the results to the conversation,
    /// record the call count on `budget`, and fire `on_turn_end`.
    ///
    /// `on_turn_end` fires on both success and error paths with the
    /// corresponding `success` flag. On error the conversation is *not*
    /// extended — the caller's error handling owns the terminal state.
    ///
    /// Takes `budget` by mutable reference (rather than `&mut self`) because
    /// the caller has already `mem::take`n it to avoid a double mutable borrow
    /// while dispatch runs.
    ///
    /// # Errors
    ///
    /// Propagates [`LoopError::Cancelled`] if cancellation fired during
    /// dispatch, or any error the recovery system escalates to a hard failure
    /// (e.g. loop detection aborts, exhaustion of retry budget).
    async fn dispatch_and_record(
        &mut self,
        tool_calls: &[ToolCall],
        turn_index: usize,
        turn_duration: Duration,
        turn_input_tokens: u64,
        turn_output_tokens: u64,
        budget: &mut SessionResult,
    ) -> Result<(), LoopError> {
        match self.dispatch_tools(tool_calls, turn_index).await {
            Ok(results) => {
                budget.tool_calls = budget.tool_calls.saturating_add(results.len());
                let tool_result_msg = Self::build_tool_result_message(results);
                self.conversation.push(tool_result_msg);
                self.managers.observers().on_turn_end(&TurnEndContext {
                    turn: turn_index,
                    success: true,
                    error: None,
                    duration_ms: Self::millis_u64(turn_duration),
                    input_tokens: turn_input_tokens,
                    output_tokens: turn_output_tokens,
                });
                Ok(())
            }
            Err(e) => {
                let err_str = e.to_string();
                self.managers.observers().on_turn_end(&TurnEndContext {
                    turn: turn_index,
                    success: false,
                    error: Some(err_str),
                    duration_ms: Self::millis_u64(turn_duration),
                    input_tokens: turn_input_tokens,
                    output_tokens: turn_output_tokens,
                });
                Err(e)
            }
        }
    }
    // ==================================================
    // Turn helpers (used by process_turn)
    // ==================================================

    /// Stream one assistant response from the API and apply post-stream bookkeeping.
    ///
    /// On success: records the success with the fallback manager and fires
    /// [`on_stream_success`](crate::observer::LoopObserver::on_stream_success).
    /// On failure: records the failure (firing
    /// [`on_fallback`](crate::observer::LoopObserver::on_fallback) if the
    /// circuit breaker trips), fires
    /// [`on_stream_failure`](crate::observer::LoopObserver::on_stream_failure),
    /// sets the terminal state, and returns the error.
    ///
    /// `LoopError::Cancelled` short-circuits the failure bookkeeping:
    /// cancellation is a clean termination, so it sets [`LoopState::Cancelled`]
    /// without tripping the fallback or firing `on_stream_failure`.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`stream_turn`](Self::stream_turn) returned.
    async fn do_stream(&mut self) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        match self.stream_turn().await {
            Ok((msg, usage, stop)) => {
                self.record_stream_success(usage.as_ref());
                Ok((msg, usage, stop))
            }
            Err(e) => Err(self.record_stream_failure(e)),
        }
    }

    /// Record a successful stream completion.
    ///
    /// Fires [`on_stream_success`](crate::observer::LoopObserver::on_stream_success)
    /// with this turn's token counts and tells the
    /// [`FallbackManager`](crate::fallback::FallbackManager) the current model
    /// is healthy (so a transient failure earlier in the session doesn't keep
    /// the circuit breaker tripped forever).
    ///
    /// Called from [`do_stream`](Self::do_stream) on the `Ok` branch only;
    /// has no return value because the caller already holds the successful
    /// `(Message, Option<Usage>, StreamStopReason)` and just needs the
    /// side-effects.
    fn record_stream_success(&mut self, usage: Option<&Usage>) {
        self.managers.fallback.record_model_success();
        let (in_tok, out_tok) = Self::usage_tokens(usage);

        // TODO: fire_stream_success
        self.managers.observers().on_stream_success(&StreamContext {
            turn: self.budget.total_turns,
            model: self.client.model(),
            input_tokens: in_tok,
            output_tokens: out_tok,
        });
    }

    /// Record a stream failure and return the error to propagate.
    ///
    /// Distinguishes [`LoopError::RateLimitEscalation`] (which trips the
    /// model circuit breaker via
    /// [`record_model_failure`](crate::fallback::FallbackManager::record_model_failure))
    /// from other stream errors (which only count as a generic API failure
    /// via
    /// [`record_api_failure`](crate::fallback::FallbackManager::record_api_failure)).
    /// When the breaker trips and a fallback model is configured, fires
    /// [`on_fallback`](crate::observer::LoopObserver::on_fallback).
    ///
    /// Then fires
    /// [`on_stream_failure`](crate::observer::LoopObserver::on_stream_failure)
    /// (regardless of breaker outcome), sets the terminal
    /// [`LoopState::Failed`], and returns the original error so the caller
    /// can propagate it from [`do_stream`](Self::do_stream).
    ///
    /// `LoopError::Cancelled` is **not** routed through here — cancellation
    /// is a clean termination that sets [`LoopState::Cancelled`] without
    /// tripping the breaker or firing `on_stream_failure`. The caller
    /// ([`run_turn_body`](Self::run_turn_body)) handles cancellation before
    /// reaching [`do_stream`](Self::do_stream).
    fn record_stream_failure(&mut self, e: LoopError) -> LoopError {
        let tripped = if matches!(e, LoopError::RateLimitEscalation { .. }) {
            self.managers.fallback.record_model_failure()
        } else {
            self.managers.fallback.record_api_failure()
        };

        if tripped {
            let from = self.client.model();
            if let Some(to) = self.managers.fallback.fallback_model() {
                tracing::warn!(from = %from, to = %to, "fallback manager tripped");
                self.managers
                    .observers()
                    .on_fallback(&FallbackContext { from, to });
            }
        }

        // TODO: fire_stream_failure
        self.managers
            .observers()
            .on_stream_failure(&StreamFailureContext {
                turn: self.budget.total_turns,
                model: self.client.model(),
                error: e.clone(),
            });

        self.state = LoopState::Failed {
            error: e.to_string(),
        };
        e
    }

    /// Fold this turn's token counts into the running session totals on
    /// `budget`. No-op when the provider did not report usage.
    ///
    /// Uses saturating add so a runaway session cannot overflow the counters.
    fn accumulate_usage(&mut self, usage: Option<&Usage>) {
        if let Some(u) = usage {
            self.budget.input_tokens = self
                .budget
                .input_tokens
                .saturating_add(u64::from(u.input_tokens));
            self.budget.output_tokens = self
                .budget
                .output_tokens
                .saturating_add(u64::from(u.output_tokens));
        }
    }

    /// Fire [`on_turn_end`](crate::observer::LoopObserver::on_turn_end) for the
    /// turn that just completed.
    ///
    /// Reports `total_turns - 1` as the turn number: callers invoke this after
    /// the per-turn increment, so subtracting one recovers the 0-indexed turn
    /// that just ran.
    fn finish_turn(&mut self, turn_in: u64, turn_out: u64, duration: Duration) {
        self.managers.observers().on_turn_end(&TurnEndContext {
            turn: self.budget.total_turns.saturating_sub(1),
            success: true,
            error: None,
            duration_ms: Self::millis_u64(duration),
            input_tokens: turn_in,
            output_tokens: turn_out,
        });
    }

    /// Build the [`TurnResult`] returned by a turn that completed the session
    /// (`is_complete: true`, [`StopReason::EndTurn`]). Used by the no-tool-calls
    /// and loop-detection success paths, both of which end the session from
    /// inside `process_turn`.
    fn turn_complete(text: String, turn_in: u64, turn_out: u64, duration: Duration) -> TurnResult {
        TurnResult {
            text,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            input_tokens: turn_in,
            output_tokens: turn_out,
            duration,
            is_complete: true,
            stop_reason: StopReason::EndTurn,
        }
    }

    /// Run context compaction if a [`ContextManager`] is configured.
    ///
    /// Best-effort: failures are logged and the turn continues with the
    /// un-compacted history rather than failing the session.
    async fn try_compact_context(&mut self) {
        if let Err(e) = self.maybe_compact_context(self.budget.total_turns).await {
            tracing::warn!(
                error = %e,
                turn = self.budget.total_turns,
                "context compaction failed; continuing with uncompactd history"
            );
        }
    }

    /// Push the user's message onto the conversation (first turn only).
    ///
    /// Continuation turns receive `input == ""` because the previous turn's
    /// tool results are already in the history.
    fn record_user_input(&mut self, input: &str) {
        if !input.is_empty() {
            self.conversation.push(Message::user(input));
        }
    }

    /// Fire [`on_turn_start`](crate::observer::LoopObserver::on_turn_start)
    /// for the turn about to stream.
    ///
    /// `turn` is the 0-indexed current turn (the same value `on_response` and
    /// `on_tool_call_received` will report for this turn — captured before the
    /// per-turn counter increment). `query` is the user's message on the first
    /// turn and `""` on continuation turns (the previous turn's tool results
    /// are already in the conversation history).
    fn fire_turn_start(&self, turn: usize, query: &str) {
        self.managers.observers().on_turn_start(&TurnStartContext {
            turn,
            query: query.to_string(),
        });
    }

    /// Fire [`on_response`](crate::observer::LoopObserver::on_response) with
    /// the assembled assistant text for the turn that just streamed.
    ///
    /// `turn` is the same 0-indexed current turn passed to
    /// [`fire_turn_start`](Self::fire_turn_start). `usage` is `None` when the
    /// provider did not report token counts for the turn.
    fn fire_response(&self, turn: usize, text: &str, usage: Option<crate::stream::Usage>) {
        self.managers.observers().on_response(&ResponseContext {
            turn,
            text: text.to_string(),
            usage,
        });
    }

    /// Fire [`on_tool_call_received`](crate::observer::LoopObserver::on_tool_call_received)
    /// for each accumulated tool call, before dispatch begins.
    ///
    /// Fires once per call regardless of how many recovery retries the call
    /// later undergoes (the retry loop lives in dispatch and re-fires only
    /// `on_tool_pre`/`on_tool_post`). `turn` is the same 0-indexed current
    /// turn passed to [`fire_turn_start`](Self::fire_turn_start); it reaches
    /// the dispatch path as `turn_idx`, so the two events correlate.
    fn fire_tool_calls_received(&self, turn: usize, tool_calls: &[ToolCall]) {
        for tc in tool_calls {
            self.managers
                .observers()
                .on_tool_call_received(&ToolCallReceivedContext {
                    turn,
                    tool: tc.tool.clone(),
                    call_id: tc.id.clone(),
                    input: tc.input.clone(),
                });
        }
    }

    /// Consult the detection manager and, if a pattern fired, produce the
    /// early-exit outcome for `process_turn` to return.
    ///
    /// Returns `None` when no pattern was detected — the caller continues with
    /// tool extraction and dispatch. Returns `Some(Ok(..))` when detection
    /// ended the turn softly (caller returns the `TurnResult`), or
    /// `Some(Err(..))` when detection forced a hard failure (caller
    /// propagates). Either `Some` arm is a terminal transition; both set the
    /// appropriate [`LoopState`] before returning.
    /// Consult the detection manager and, if a pattern forced a hard stop,
    /// produce the abort outcome for `process_turn` to return.
    ///
    /// Returns `None` when no pattern fired (caller continues with tool
    /// extraction and dispatch), or `Some(Err(..))` with the propagated error
    /// when detection aborted the session. The terminal state is set via
    /// [`set_error_state`](Self::set_error_state) before returning.
    fn apply_loop_detection(
        &mut self,
        current_turn: usize,
        pattern: &crate::detection::DetectedPattern,
    ) -> Option<LoopError> {
        let e = self
            .managers
            .handle_detected_pattern(pattern, current_turn)?;
        self.set_error_state(&e);
        Some(e)
    }

    /// End the session because the model finished its turn without requesting
    /// any tool calls.
    ///
    /// Fires [`on_turn_end`](crate::observer::LoopObserver::on_turn_end)
    /// (via [`finish_turn`](Self::finish_turn)), transitions to
    /// [`LoopState::Completed`], and returns a session-completing
    /// [`TurnResult`] (`is_complete: true`) for `process_turn` to return.
    ///
    /// Distinct from the success arm of
    /// [`apply_loop_detection`](Self::apply_loop_detection), which transitions
    /// to `Completed` *without* firing `on_turn_end` — that path ends the
    /// session from detection, not from natural turn completion, so the
    /// turn-end bookkeeping is intentionally skipped there.
    fn complete_session(
        &mut self,
        text: String,
        turn_in: u64,
        turn_out: u64,
        turn_start: Instant,
    ) -> TurnResult {
        // TODO: fire_complete_turn
        self.finish_turn(turn_in, turn_out, turn_start.elapsed());
        self.state = LoopState::Completed {
            summary: text.clone(),
        };
        Self::turn_complete(text, turn_in, turn_out, turn_start.elapsed())
    }

    /// Set the terminal state for a propagated error.
    ///
    /// All errors from the turn body are recorded as [`LoopState::Failed`].
    /// Cancellation is handled separately by the `select!` in `process_turn`,
    /// which sets [`LoopState::Cancelled`] directly and never reaches this
    /// method.
    fn set_error_state(&mut self, e: &LoopError) {
        self.state = LoopState::Failed {
            error: e.to_string(),
        };
    }
}

// ==================================================
// ModelSwitch builder
// ==================================================

/// Builder for a runtime model switch on [`BareLoop`].
///
/// Created by [`BareLoop::switch_model`]. Allows updating
/// context-window and max-tokens alongside the model name, then applies
/// all changes atomically via [`apply`](Self::apply).
///
/// The switch resets the fallback circuit breaker (stale failure counts
/// from the old model are meaningless for the new one) and fires
/// [`on_model_switched`](crate::observer::LoopObserver::on_model_switched)
/// to all observers.
pub struct ModelSwitch<'a, C: ApiClient> {
    loop_: &'a mut BareLoop<C>,
    target_model: String,
    context_window: Option<u64>,
    max_tokens: Option<u32>,
}

impl<C: ApiClient> ModelSwitch<'_, C> {
    /// Set the context window (in tokens) for the new model.
    ///
    /// If omitted, the existing `LoopConfig::context_window` is kept.
    /// Updating this is important when switching to a model with a
    /// significantly different context window — otherwise the
    /// auto-compactor will use the wrong threshold.
    #[must_use]
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Set the max output tokens for the new model.
    ///
    /// If omitted, the existing `LoopConfig::max_tokens` is kept.
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Apply the model switch.
    ///
    /// Performs the following atomically:
    /// 1. Validates the target model is non-empty.
    /// 2. Delegates to [`ApiClient::set_model`] on the underlying client.
    /// 3. Updates `LoopConfig::model`, `context_window`, and `max_tokens`.
    /// 4. Resets the [`FallbackManager`](crate::fallback::FallbackManager)
    ///    circuit breaker to `Primary` and updates the original-model
    ///    tracker to the new model.
    /// 5. Fires [`on_model_switched`](crate::observer::LoopObserver::on_model_switched).
    ///
    /// # Errors
    ///
    /// - [`LoopError::Config`] if the model name is empty/whitespace.
    pub fn apply(self) -> Result<(), LoopError> {
        let Self {
            loop_,
            target_model,
            context_window,
            max_tokens,
        } = self;

        let trimmed = target_model.trim();
        if trimmed.is_empty() {
            return Err(LoopError::Config(
                "model name must not be empty or whitespace".into(),
            ));
        }

        let from = loop_.config.model.clone();
        loop_.client.set_model(trimmed);
        loop_.config.model = trimmed.to_string();

        if let Some(cw) = context_window {
            loop_.config.context_window = cw;
        }

        if let Some(mt) = max_tokens {
            loop_.config.max_tokens = mt;
        }

        loop_.managers.fallback.reset();
        loop_
            .managers
            .fallback
            .set_original_model(trimmed.to_string());
        loop_
            .managers
            .observers()
            .on_model_switched(&ModelSwitchedContext {
                from,
                to: trimmed.to_string(),
            });

        Ok(())
    }
}

impl<C: ApiClient> BareLoop<C> {
    /// Execute the turn body without cancellation awareness.
    ///
    /// Cancellation is handled by the `select!` in `process_turn`, which
    /// drops this future if `cancel.notified()` fires.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] on streaming failure, loop detection abort, tool
    /// dispatch error, or compaction failure.
    async fn run_turn_body(
        &mut self,
        current_turn: usize,
        turn_start: Instant,
    ) -> Result<TurnResult, LoopError> {
        let (msg, usage, _stream_stop) = self.do_stream().await?;
        self.accumulate_usage(usage.as_ref());

        let text = Self::extract_text(&msg);
        let (turn_in, turn_out) = Self::usage_tokens(usage.as_ref());
        let pattern = self.managers.detection.record_response(&text);
        self.fire_response(current_turn, &text, usage);

        if let Some(e) = self.apply_loop_detection(current_turn, &pattern) {
            return Err(e);
        }

        let tool_calls = Self::extract_tool_calls(&msg);
        self.conversation.push(msg);
        self.budget.total_turns = self.budget.total_turns.saturating_add(1);

        if tool_calls.is_empty() {
            // TODO: complete turn
            return Ok(self.complete_session(text, turn_in, turn_out, turn_start));
        }

        self.fire_tool_calls_received(current_turn, &tool_calls);
        self.state = LoopState::WaitingForTool {
            tool: tool_calls
                .first()
                .map(|tc| tc.tool.clone())
                .unwrap_or_default(),
            started_at: std::time::SystemTime::now(),
        };

        let mut budget = std::mem::take(&mut self.budget);
        let turn_duration = turn_start.elapsed();
        let dispatch_result = self
            .dispatch_and_record(
                &tool_calls,
                current_turn,
                turn_duration,
                turn_in,
                turn_out,
                &mut budget,
            )
            .await;

        self.budget = budget;
        if let Err(e) = dispatch_result {
            self.set_error_state(&e);
            return Err(e);
        }

        self.try_compact_context().await;
        self.state = LoopState::Processing {
            turn: self.budget.total_turns,
        };

        Ok(TurnResult {
            text,
            tool_calls,
            tool_results: Vec::new(),
            input_tokens: turn_in,
            output_tokens: turn_out,
            duration: turn_start.elapsed(),
            is_complete: false,
            stop_reason: StopReason::ToolCall,
        })
    }
}

impl<C: ApiClient> crate::engine::loop_core::Loop for BareLoop<C> {
    fn initialize<'a>(
        &'a mut self,
        config: &'a crate::config::LoopConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        Box::pin(async move {
            config.validate()?;

            self.state = LoopState::Processing { turn: 0 };
            self.budget = SessionResult::default();
            self.session_start = Some(Instant::now());
            self.config = config.clone();
            self.managers.reset_all();
            self.notify_session_start();
            Ok(())
        })
    }

    fn process_turn<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            let turn_start = Instant::now();
            let current_turn = self.budget.total_turns;

            self.record_user_input(input);
            self.fire_turn_start(current_turn, input);

            if !self.contributors.is_empty() {
                let ctx = ContributorContext::new(current_turn, &self.conversation);
                let injected: Vec<Message> = self
                    .contributors
                    .iter()
                    .filter_map(|contributor| contributor.contribute(&ctx))
                    .collect();
                self.conversation.extend(injected);
            }

            let cancel = Arc::clone(&self.cancelled);
            tokio::select! {
                biased;
                () = cancel.notified() => {
                    self.state = LoopState::Cancelled;
                    // TODO: fire_turn_cancelled
                    self.managers.observers().on_turn_end(&TurnEndContext {
                        turn: current_turn,
                        success: false,
                        error: Some("cancelled".into()),
                        duration_ms: Self::millis_u64(turn_start.elapsed()),
                        input_tokens: 0,
                        output_tokens: 0,
                    });
                    Err(LoopError::Cancelled)
                }
                result = self.run_turn_body(current_turn, turn_start) => result,
            }
        })
    }

    fn should_continue(&self) -> bool {
        if self.is_cancelled() {
            return false;
        }
        self.budget.total_turns < self.config.max_turns
    }

    fn finalize<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<SessionResult, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            let duration = self.session_start.map(|s| s.elapsed()).unwrap_or_default();

            let success = !matches!(self.state, LoopState::Failed { .. } | LoopState::Cancelled);

            // Fill in the final fields on the budget accumulator, then
            // use it as the SessionResult.
            self.budget.success = success;
            self.budget.session_id = self.config.session_id;
            self.budget.total_duration = duration;

            if success {
                self.budget.final_output = match &self.state {
                    LoopState::Completed { summary } => Some(summary.clone()),
                    _ => Some(String::new()),
                };
            } else {
                self.budget.error = Some(match &self.state {
                    LoopState::Failed { error } => error.clone(),
                    LoopState::Cancelled => "session cancelled".to_string(),
                    _ => "session failed".to_string(),
                });
            }

            // Notify observers/hooks using the final SessionResult.
            self.notify_session_end(&self.budget, duration);

            Ok(self.budget.clone())
        })
    }

    fn state(&self) -> LoopState {
        self.state.clone()
    }

    fn cancel(&self) {
        BareLoop::cancel(self);
    }

    fn stop_reason(&self) -> Option<LoopError> {
        if self.is_cancelled() {
            return Some(LoopError::Cancelled);
        }
        if self.budget.total_turns >= self.config.max_turns {
            return Some(LoopError::MaxTurnsExceeded {
                max: self.config.max_turns,
            });
        }
        None
    }

    fn config(&self) -> &LoopConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::ApiError;
    use crate::engine::loop_core::Loop;
    use crate::stream::{
        DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
        PartStart, Usage,
    };
    use crate::tool::ToolRegistry;
    use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};
    use serde_json::{Value, json};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use std::sync::Mutex;

    #[derive(Clone)]
    struct MockClient {
        responses: Arc<Mutex<Vec<Vec<StreamEvent>>>>,

        model_name: Arc<std::sync::Mutex<String>>,
    }

    impl MockClient {
        fn new(model: &str) -> Self {
            Self {
                responses: Arc::new(Mutex::new(Vec::new())),
                model_name: Arc::new(std::sync::Mutex::new(model.to_string())),
            }
        }

        fn add_text_response(&self, text: &str) {
            let events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_test".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::text(text)),
                }),
                StreamEvent::IndexedDelta(IndexedDelta {
                    index: 0,
                    delta: DeltaPart::Text {
                        text: text.to_string(),
                    },
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("end_turn".to_string()),
                    },
                    usage: Some(Usage::new(10, 20)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(events);
        }

        fn add_events(&self, events: Vec<StreamEvent>) {
            crate::error::recover_guard(self.responses.lock()).push(events);
        }

        fn add_tool_then_text(
            &self,
            tool_id: &str,
            tool_name: &str,
            tool_input: Value,
            final_text: &str,
        ) {
            // First response: tool_call
            let tool_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_tool".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("tool_call".to_string()),
                    },
                    usage: Some(Usage::new(50, 10)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(tool_events);

            // Second response: end_turn with text
            let text_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_final".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::text(final_text)),
                }),
                StreamEvent::IndexedDelta(IndexedDelta {
                    index: 0,
                    delta: DeltaPart::Text {
                        text: final_text.to_string(),
                    },
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("end_turn".to_string()),
                    },
                    usage: Some(Usage::new(30, 15)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(text_events);
        }

        fn add_tool_only_response(&self, tool_id: &str, tool_name: &str, tool_input: Value) {
            let tool_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: format!("msg_{tool_id}"),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("tool_call".to_string()),
                    },
                    usage: Some(Usage::new(50, 10)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(tool_events);
        }

        #[expect(dead_code)]
        fn add_error_response(&self) {
            // Return an empty response that will cause the stream to error
            // We'll handle this by having the stream return an error event
            let events = vec![StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_err".into(),
                    role: "assistant".into(),
                    model: crate::error::recover_guard(self.model_name.lock()).clone(),
                },
            })];
            crate::error::recover_guard(self.responses.lock()).push(events);
        }
    }

    impl ApiClient for MockClient {
        fn model(&self) -> String {
            crate::error::recover_guard(self.model_name.lock()).clone()
        }

        fn set_model(&self, model: &str) -> bool {
            if model.trim().is_empty() {
                return false;
            }
            *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
            true
        }

        fn stream_messages(
            &self,
            _request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            let mut guard = crate::error::recover_guard(self.responses.lock());
            if let Some(events) = guard.pop_front() {
                let events: Vec<Result<StreamEvent, ApiError>> =
                    events.into_iter().map(Ok).collect();
                Box::pin(futures::stream::iter(events))
            } else {
                // No more responses — return an error
                let err = ApiError::api("No more mock responses");
                Box::pin(futures::stream::iter(vec![Err(err)]))
            }
        }

        fn create_message(
            &self,
            _request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
            Box::pin(async { Ok(json!({"content": []})) })
        }
    }

    // Helper trait for Vec-like pop_front on Vec
    trait PopFront<T> {
        fn pop_front(&mut self) -> Option<T>;
    }

    impl<T> PopFront<T> for Vec<T> {
        fn pop_front(&mut self) -> Option<T> {
            if self.is_empty() {
                None
            } else {
                Some(self.remove(0))
            }
        }
    }

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }

        fn description(&self) -> &'static str {
            "Echoes back the input"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "echo".into(),
                description: "Echoes back the input".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"]
                }),
            }
        }

        fn call(
            &self,
            input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let msg = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutput::text(format!("Echo: {msg}"))) })
        }
    }

    struct FailingTool;

    impl Tool for FailingTool {
        fn name(&self) -> &'static str {
            "fail"
        }

        fn description(&self) -> &'static str {
            "Always fails"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "fail".into(),
                description: "Always fails".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            }
        }

        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async move { Err(ToolError::Execution("Tool intentionally failed".into())) })
        }
    }

    struct FlakyTool {
        fail_threshold: usize,
        attempts: AtomicUsize,
    }

    impl FlakyTool {
        fn new(fail_threshold: usize) -> Self {
            Self {
                fail_threshold,
                attempts: AtomicUsize::new(0),
            }
        }
    }

    impl Tool for FlakyTool {
        fn name(&self) -> &'static str {
            "flaky"
        }

        fn description(&self) -> &'static str {
            "Fails the first N calls, then succeeds"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "flaky".into(),
                description: "Fails the first N calls, then succeeds".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            }
        }

        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if attempt < self.fail_threshold {
                    Err(ToolError::Execution("Flaky tool failing".into()))
                } else {
                    Ok(ToolOutput::text("Flaky tool succeeded"))
                }
            })
        }
    }

    struct CountingObserver {
        session_starts: AtomicUsize,
        session_ends: AtomicUsize,
        turn_starts: AtomicUsize,
        turn_ends: AtomicUsize,
        tool_calls_received: AtomicUsize,
        tool_pres: AtomicUsize,
        tool_posts: AtomicUsize,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self {
                session_starts: AtomicUsize::new(0),
                session_ends: AtomicUsize::new(0),
                turn_starts: AtomicUsize::new(0),
                turn_ends: AtomicUsize::new(0),
                tool_calls_received: AtomicUsize::new(0),
                tool_pres: AtomicUsize::new(0),
                tool_posts: AtomicUsize::new(0),
            }
        }
    }

    impl crate::observer::LoopObserver for CountingObserver {
        fn name(&self) -> &'static str {
            "counting"
        }

        fn on_session_start(&self, _ctx: &crate::observer::SessionStartContext) {
            self.session_starts.fetch_add(1, Ordering::SeqCst);
        }

        fn on_session_end(&self, _ctx: &crate::observer::SessionEndContext) {
            self.session_ends.fetch_add(1, Ordering::SeqCst);
        }

        fn on_turn_start(&self, _ctx: &crate::observer::TurnStartContext) {
            self.turn_starts.fetch_add(1, Ordering::SeqCst);
        }

        fn on_turn_end(&self, _ctx: &crate::observer::TurnEndContext) {
            self.turn_ends.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_call_received(&self, _ctx: &crate::observer::ToolCallReceivedContext) {
            self.tool_calls_received.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_pre(&self, _ctx: &crate::observer::ToolPreContext) {
            self.tool_pres.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_post(&self, _ctx: &crate::observer::ToolPostContext) {
            self.tool_posts.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn make_config() -> LoopConfig {
        LoopConfig {
            max_turns: 10,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_bare_loop_single_turn() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello! I'm done.");

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 1);
        assert_eq!(result.final_output.as_deref(), Some("Hello! I'm done."));
    }

    #[tokio::test]
    async fn test_bare_loop_with_tool_call() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text(
            "tool_1",
            "echo",
            json!({"message": "hello"}),
            "I echoed your message.",
        );

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let result = agent.run("Echo hello").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 2); // tool_call turn + end_turn
        assert_eq!(result.tool_calls, 1);
    }

    #[tokio::test]
    async fn test_bare_loop_max_turns_exceeded() {
        let client = MockClient::new("test-model");
        // Return only tool_call responses so the loop never gets an end_turn
        for i in 0..20 {
            client.add_tool_only_response(
                &format!("tool_{i}"),
                "echo",
                json!({"message": format!("msg_{i}")}),
            );
        }

        let mut config = make_config();
        config.max_turns = 3;

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let result = agent.run("Keep going").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LoopError::MaxTurnsExceeded { max } => assert_eq!(max, 3),
            other => panic!("Expected MaxTurnsExceeded, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_bare_loop_cancellation() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello!");

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

        // Cancel before running
        agent.cancel();
        assert!(agent.is_cancelled());

        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LoopError::Cancelled => {}
            other => panic!("Expected Cancelled error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_bare_loop_api_error() {
        // The mock will return an error
        let client = MockClient::new("test-model");
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LoopError::Api(msg) => assert!(msg.contains("No more mock responses"), "got: {msg}"),
            other => panic!("Expected Api error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_tool_not_found_returns_error_result() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "I see the tool failed.");

        // Empty registry — tool won't be found
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Use nonexistent tool").await.unwrap();

        // The tool-not-found should be returned as an error result in the conversation,
        // not as a hard error. The loop should continue and eventually get the end_turn.
        assert!(result.success);
        assert_eq!(result.total_turns, 2);
    }

    #[tokio::test]
    async fn test_tool_execution_failure() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "fail", json!({}), "The tool failed, moving on.");

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let result = agent.run("Use failing tool").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 2);
    }

    #[tokio::test]
    async fn test_observer_lifecycle_events() {
        let client = MockClient::new("test-model");
        client.add_text_response("Done!");

        let plugin = Arc::new(CountingObserver::new());
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        agent.register_observer(plugin.clone());

        let result = agent.run("Hi").await.unwrap();

        assert!(result.success);
        assert_eq!(plugin.session_starts.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.session_ends.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.turn_starts.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.turn_ends.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_observer_tool_events() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "test"}), "All done!");

        let plugin = Arc::new(CountingObserver::new());
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        agent.register_observer(plugin.clone());

        let result = agent.run("Echo test").await.unwrap();
        assert!(result.success);

        assert_eq!(plugin.tool_pres.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.tool_posts.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.turn_starts.load(Ordering::SeqCst), 2);
        assert_eq!(plugin.turn_ends.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_conversation_built_correctly() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text(
            "tool_1",
            "echo",
            json!({"message": "hello"}),
            "Final answer.",
        );

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);

        // Pre-push the user message manually to inspect conversation after
        agent.conversation.push(Message::user("Echo hello"));

        // Can't run directly since we manually pushed, but let's verify the helpers
        let msg = Message::assistant("test");
        let tool_calls = BareLoop::<MockClient>::extract_tool_calls(&msg);
        assert!(tool_calls.is_empty());

        let msg_with_tools = Message::new(
            Role::Assistant,
            vec![
                MessagePart::text("Using tool..."),
                MessagePart::tool_call("id1", "echo", json!({"message": "hi"})),
            ],
        );
        let tool_calls = BareLoop::<MockClient>::extract_tool_calls(&msg_with_tools);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool, "echo");
    }

    #[tokio::test]
    async fn test_tool_result_message_format() {
        let results = vec![super::ToolDispatchResult {
            tool_call_id: "tool_123".to_string(),
            output: ToolContent::Text("Echo: hello".to_string()),
            is_error: false,
            duration: Duration::from_millis(100),
            resolved_tool_name: String::new(),
            display_hint: None,
        }];

        let msg = BareLoop::<MockClient>::build_tool_result_message(results);
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.parts.len(), 1);

        match &msg.parts[0] {
            MessagePart::ToolResult {
                call_id,
                output,
                is_error,
            } => {
                assert_eq!(call_id, "tool_123");
                assert!(!is_error.unwrap_or(true));
                let text = output.to_string();
                assert_eq!(text, "Echo: hello");
            }
            other => panic!("Expected ToolResult part, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_multiple_tool_calls_in_one_turn() {
        let client = MockClient::new("test-model");

        // First response: two tool_call parts
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_multi".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call(
                    "t1",
                    "echo",
                    json!({"message": "first"}),
                )),
            }),
            StreamEvent::PartStop,
            StreamEvent::PartStart(PartStart {
                index: 1,
                part: Some(MessagePart::tool_call(
                    "t2",
                    "echo",
                    json!({"message": "second"}),
                )),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 20)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(client.responses.lock()).push(tool_events);

        // Second response: end_turn
        client.add_text_response("Both tools executed.");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);

        let result = agent.run("Echo twice").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 2);
        assert_eq!(result.tool_calls, 2);
    }

    #[tokio::test]
    async fn test_text_streamer_fires_on_text_delta() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello world");
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let received = Arc::new(Mutex::new(Vec::new()));
        let buf = Arc::clone(&received);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            crate::error::recover_guard(buf.lock()).push(delta.to_string());
        }));

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let received = crate::error::recover_guard(received.lock());
        assert!(!received.is_empty(), "streamer should have fired");
        assert!(
            received.join("").contains("Hello world"),
            "got: {received:?}",
        );
    }

    #[tokio::test]
    async fn test_text_streamer_fires_when_stream_handler_configured() {
        // Regression: when a StreamHandler is attached, the engine must still
        // fire text_streamer / on_text_delta for each streamed text delta. The
        // handler path used to bypass observers entirely.
        let client = MockClient::new("test-model");
        client.add_text_response("via handler");
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        agent.set_stream_handler(StreamHandler::new());

        let received = Arc::new(Mutex::new(Vec::new()));
        let buf = Arc::clone(&received);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            crate::error::recover_guard(buf.lock()).push(delta.to_string());
        }));

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let received = crate::error::recover_guard(received.lock());
        assert!(
            !received.is_empty(),
            "streamer should fire even with a StreamHandler configured"
        );
        assert!(
            received.join("").contains("via handler"),
            "got: {received:?}",
        );
    }

    #[tokio::test]
    async fn test_text_streamer_none_works() {
        let client = MockClient::new("test-model");
        client.add_text_response("No streamer");
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_text_streamer_ignores_non_text_deltas() {
        let client = MockClient::new("test-model");

        // Build a response with tool-call events (no text).
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg-1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::ToolCall {
                    id: "call_1".into(),
                    name: "echo".into(),
                    input: Value::Null,
                }),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::InputJson {
                    partial_json: "{}".into(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".into()),
                },
                usage: None,
            }),
            StreamEvent::MessageStop,
        ];
        client.add_events(events);

        // Second turn: plain text response.
        client.add_text_response("Done");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let received = Arc::new(Mutex::new(String::new()));
        let buf = Arc::clone(&received);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            crate::error::recover_guard(buf.lock()).push_str(delta);
        }));

        agent.run("Use tool").await.unwrap();

        // The InputJson delta should NOT have triggered the streamer.
        // Only the "Done" text response in the second turn should.
        let received = crate::error::recover_guard(received.lock());
        assert_eq!(&*received, "Done", "only text deltas should fire streamer");
    }

    #[tokio::test]
    async fn test_on_text_delta_fires_per_sse_chunk_in_order() {
        struct DeltaRecorder {
            deltas: Arc<Mutex<Vec<(usize, String)>>>,
        }
        impl crate::observer::LoopObserver for DeltaRecorder {
            fn name(&self) -> &'static str {
                "delta-recorder"
            }
            fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
            }
        }

        let client = MockClient::new("test-model");
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg-1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text("ignored")),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: "Hello".into(),
                },
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text { text: " ".into() },
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: "world".into(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                },
                usage: None,
            }),
            StreamEvent::MessageStop,
        ];
        client.add_events(events);

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(DeltaRecorder {
            deltas: Arc::clone(&captured),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let captured = crate::error::recover_guard(captured.lock());
        assert_eq!(captured.len(), 3, "one on_text_delta per SSE text chunk");
        let joined: String = captured.iter().map(|(_, d)| d.as_str()).collect();
        assert_eq!(joined, "Hello world");
    }

    #[tokio::test]
    async fn test_text_delta_turn_number_matches_surrounding_turn() {
        struct TurnRecorder {
            deltas: Arc<Mutex<Vec<(usize, String)>>>,
            response_turns: Arc<Mutex<Vec<usize>>>,
        }
        impl crate::observer::LoopObserver for TurnRecorder {
            fn name(&self) -> &'static str {
                "turn-recorder"
            }
            fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
            }
            fn on_response(&self, ctx: &crate::observer::ResponseContext) {
                crate::error::recover_guard(self.response_turns.lock()).push(ctx.turn);
            }
        }

        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "All done");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let deltas = Arc::new(Mutex::new(Vec::new()));
        let response_turns = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(TurnRecorder {
            deltas: Arc::clone(&deltas),
            response_turns: Arc::clone(&response_turns),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Use echo then finish").await.unwrap();
        assert!(result.success);
        assert_eq!(result.total_turns, 2);

        let response_turns = crate::error::recover_guard(response_turns.lock());
        let deltas = crate::error::recover_guard(deltas.lock());

        assert_eq!(
            response_turns.len(),
            2,
            "both turns should fire on_response",
        );
        assert!(!deltas.is_empty(), "text turn should produce deltas");
        for (turn, _) in deltas.iter() {
            assert!(
                response_turns.contains(turn),
                "on_text_delta turn {turn} must match an on_response turn",
            );
        }

        let text_turn = deltas.iter().map(|(t, _)| *t).next().unwrap();
        let joined: String = deltas
            .iter()
            .filter(|(t, _)| *t == text_turn)
            .map(|(_, d)| d.as_str())
            .collect();
        assert_eq!(joined, "All done");
        assert_eq!(
            text_turn, 1,
            "text deltas belong to the second turn (the text turn), not the tool turn",
        );
    }

    #[tokio::test]
    async fn test_on_text_delta_ignores_non_text_deltas() {
        struct DeltaRecorder {
            count: Arc<AtomicUsize>,
        }
        impl crate::observer::LoopObserver for DeltaRecorder {
            fn name(&self) -> &'static str {
                "delta-recorder"
            }
            fn on_text_delta(&self, _ctx: &crate::observer::TextDeltaContext) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }

        let client = MockClient::new("test-model");
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg-1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::ToolCall {
                    id: "call_1".into(),
                    name: "echo".into(),
                    input: Value::Null,
                }),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::InputJson {
                    partial_json: "{}".into(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".into()),
                },
                usage: None,
            }),
            StreamEvent::MessageStop,
        ];
        client.add_events(events);
        client.add_text_response("Done");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let count = Arc::new(AtomicUsize::new(0));
        let recorder = Arc::new(DeltaRecorder {
            count: Arc::clone(&count),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        agent.run("Use tool").await.unwrap();

        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "only the text delta should fire on_text_delta",
        );
    }

    #[tokio::test]
    async fn test_on_text_delta_fires_without_streamer() {
        struct DeltaRecorder {
            deltas: Arc<Mutex<Vec<String>>>,
        }
        impl crate::observer::LoopObserver for DeltaRecorder {
            fn name(&self) -> &'static str {
                "delta-recorder"
            }
            fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
            }
        }

        let client = MockClient::new("test-model");
        client.add_text_response("Hello world");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(DeltaRecorder {
            deltas: Arc::clone(&captured),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let captured = crate::error::recover_guard(captured.lock());
        assert!(
            !captured.is_empty(),
            "observer should receive deltas with no streamer set"
        );
        let joined: String = captured.iter().map(String::as_str).collect();
        assert!(joined.contains("Hello world"), "got: {joined:?}");
    }

    #[tokio::test]
    async fn test_on_text_delta_and_streamer_coexist() {
        struct DeltaRecorder {
            deltas: Arc<Mutex<Vec<String>>>,
        }
        impl crate::observer::LoopObserver for DeltaRecorder {
            fn name(&self) -> &'static str {
                "delta-recorder"
            }
            fn on_text_delta(&self, ctx: &crate::observer::TextDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push(ctx.delta.clone());
            }
        }

        let client = MockClient::new("test-model");
        client.add_text_response("Hello world");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let streamer_buf = Arc::new(Mutex::new(Vec::new()));
        let buf = Arc::clone(&streamer_buf);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            crate::error::recover_guard(buf.lock()).push(delta.to_string());
        }));

        let observer_buf = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(DeltaRecorder {
            deltas: Arc::clone(&observer_buf),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let streamer_buf = crate::error::recover_guard(streamer_buf.lock());
        let observer_buf = crate::error::recover_guard(observer_buf.lock());
        assert!(!streamer_buf.is_empty(), "streamer should fire");
        assert!(!observer_buf.is_empty(), "observer should fire");
        assert_eq!(
            streamer_buf.len(),
            observer_buf.len(),
            "both paths receive the same number of deltas",
        );
        assert_eq!(
            *streamer_buf, *observer_buf,
            "both paths receive identical chunks"
        );
    }

    #[tokio::test]
    async fn test_on_tool_call_received_fires_once_per_call() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());

        let result = agent.run("Use echo").await.unwrap();
        assert!(result.success);

        assert_eq!(
            observer.tool_calls_received.load(Ordering::SeqCst),
            1,
            "one accumulated call → one received event",
        );
        assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_on_tool_call_received_fires_per_call_for_multiple_calls() {
        let client = MockClient::new("test-model");
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_multi".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call(
                    "t1",
                    "echo",
                    json!({"message": "first"}),
                )),
            }),
            StreamEvent::PartStop,
            StreamEvent::PartStart(PartStart {
                index: 1,
                part: Some(MessagePart::tool_call(
                    "t2",
                    "echo",
                    json!({"message": "second"}),
                )),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 20)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(client.responses.lock()).push(tool_events);
        client.add_text_response("All done");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());

        let result = agent.run("Echo twice").await.unwrap();
        assert!(result.success);

        assert_eq!(
            observer.tool_calls_received.load(Ordering::SeqCst),
            2,
            "two accumulated calls → two received events",
        );
        assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_on_tool_call_received_not_fired_for_text_only_turn() {
        let client = MockClient::new("test-model");
        client.add_text_response("Just text, no tools");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        assert_eq!(
            observer.tool_calls_received.load(Ordering::SeqCst),
            0,
            "no tool calls → no received event",
        );
        assert_eq!(observer.tool_pres.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_on_tool_call_received_turn_matches_other_events() {
        struct TurnCapture {
            received_turns: Arc<Mutex<Vec<usize>>>,
            response_turns: Arc<Mutex<Vec<usize>>>,
            pre_turns: Arc<Mutex<Vec<usize>>>,
        }
        impl crate::observer::LoopObserver for TurnCapture {
            fn name(&self) -> &'static str {
                "turn-capture"
            }
            fn on_response(&self, ctx: &crate::observer::ResponseContext) {
                crate::error::recover_guard(self.response_turns.lock()).push(ctx.turn);
            }
            fn on_tool_call_received(&self, ctx: &crate::observer::ToolCallReceivedContext) {
                crate::error::recover_guard(self.received_turns.lock()).push(ctx.turn);
            }
            fn on_tool_pre(&self, ctx: &crate::observer::ToolPreContext) {
                crate::error::recover_guard(self.pre_turns.lock()).push(ctx.turn);
            }
        }

        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let received = Arc::new(Mutex::new(Vec::new()));
        let response = Arc::new(Mutex::new(Vec::new()));
        let pre = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(TurnCapture {
            received_turns: Arc::clone(&received),
            response_turns: Arc::clone(&response),
            pre_turns: Arc::clone(&pre),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Use echo").await.unwrap();
        assert!(result.success);

        let received = crate::error::recover_guard(received.lock());
        let response = crate::error::recover_guard(response.lock());
        let pre = crate::error::recover_guard(pre.lock());
        assert_eq!(received.len(), 1, "one tool call → one received event");
        for turn in received.iter() {
            assert!(
                response.contains(turn),
                "received turn {turn} must match an on_response turn",
            );
            assert!(
                pre.contains(turn),
                "received turn {turn} must match an on_tool_pre turn",
            );
        }
    }

    #[tokio::test]
    async fn test_on_tool_call_received_does_not_refire_on_retry() {
        struct AlwaysRecoverable;
        impl crate::reflection::Reflector for AlwaysRecoverable {
            fn analyze(
                &self,
                error: &str,
                tool_name: &str,
                _tool_input: &serde_json::Value,
                _tool_schema: Option<&crate::tool::ToolSchema>,
                _context: &crate::reflection::ReflectionContext,
            ) -> Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                crate::reflection::FailureAnalysis,
                                crate::reflection::ReflectionError,
                            >,
                        > + Send
                        + '_,
                >,
            > {
                let error = error.to_string();
                let tool_name = tool_name.to_string();
                Box::pin(async move {
                    Ok(crate::reflection::FailureAnalysis {
                        is_recoverable: true,
                        root_cause: error,
                        severity: crate::reflection::FailureSeverity::Medium,
                        correction: None,
                        context: format!("tool: {tool_name}"),
                    })
                })
            }
        }

        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "flaky", json!({}), "Recovered");

        let mut registry = ToolRegistry::new();
        registry.register(FlakyTool::new(2));

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        agent.set_reflector(Arc::new(AlwaysRecoverable));
        agent.set_recovery_strategy(Arc::new(
            crate::reflection::ExponentialBackoffRecovery::new(3)
                .with_base_delay(std::time::Duration::ZERO),
        ));
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());

        let result = agent.run("Use flaky").await.unwrap();
        assert!(result.success);

        assert_eq!(
            observer.tool_calls_received.load(Ordering::SeqCst),
            1,
            "received fires once per call regardless of retries",
        );
        assert!(
            observer.tool_pres.load(Ordering::SeqCst) >= 2,
            "tool_pre must re-fire on each retry attempt",
        );
    }

    #[test]
    fn test_accessors() {
        let client = MockClient::new("test-model");
        let config = make_config();
        let session_id = config.session_id;
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

        assert_eq!(agent.config().session_id, session_id);
        assert!(agent.conversation().is_empty());
        assert!(!agent.is_cancelled());
    }

    #[test]
    fn test_cancel_signal_shared() {
        let client = MockClient::new("test-model");
        let config = make_config();
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let signal = agent.cancel_signal();
        assert!(!signal.is_cancelled());

        agent.cancel();
        assert!(signal.is_cancelled());
        assert!(agent.is_cancelled());
    }

    #[tokio::test]
    async fn test_session_result_fields() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello!");

        let config = make_config();
        let session_id = config.session_id;
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await.unwrap();

        assert_eq!(result.session_id, session_id);
        assert!(result.total_duration > Duration::ZERO);
        assert!(result.input_tokens > 0 || result.output_tokens > 0); // from mock usage
    }

    #[tokio::test]
    async fn test_loop_terminates_with_max_turns_1() {
        let client = MockClient::new("test-model");
        client.add_text_response("One and done.");

        let mut config = make_config();
        config.max_turns = 1;

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 1);
    }

    #[tokio::test]
    async fn test_loop_terminates_with_max_turns_0() {
        let client = MockClient::new("test-model");
        client.add_text_response("Should not be reached.");

        let mut config = make_config();
        config.max_turns = 0;

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LoopError::Config(msg) => assert!(msg.contains("max_turns")),
            other => panic!("Expected Config error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_tool_error_is_soft_not_hard() {
        let client = MockClient::new("test-model");

        // Response: request a nonexistent tool
        let tool_events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::tool_call("t1", "nonexistent", json!({}))),
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_call".to_string()),
                },
                usage: Some(Usage::new(50, 10)),
            }),
            StreamEvent::MessageStop,
        ];
        crate::error::recover_guard(client.responses.lock()).push(tool_events);

        // Second response: end_turn after seeing error result
        client.add_text_response("Tool wasn't found, but I'll handle it.");

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Use missing tool").await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_loop_detection_hard_stop_propagates_loop_error() {
        use crate::detection::{DetectionConfig, DetectionManager};
        use crate::runtime::LoopRuntime;

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let client = MockClient::new("test");
        for i in 0..10 {
            client.add_tool_only_response(&format!("call_{i}"), "echo", json!({ "message": "hi" }));
        }

        let runtime = LoopRuntime::new().with_detection(
            DetectionManager::new_with_config(DetectionConfig {
                loop_threshold: 2,
                stop_threshold: 2,
                ..Default::default()
            })
            .expect("valid detection config"),
        );

        let mut agent =
            BareLoop::new_with_managers(Arc::new(client), registry, make_config(), runtime);
        let result = agent.run("test").await;

        assert!(
            matches!(result, Err(LoopError::LoopDetected { .. })),
            "expected Err(LoopError::LoopDetected), got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_loop_detection_soft_block_before_stop_threshold() {
        use crate::detection::{DetectionConfig, DetectionManager};
        use crate::runtime::LoopRuntime;

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let client = MockClient::new("test");
        client.add_tool_only_response("c1", "echo", json!({ "message": "hi" }));
        client.add_tool_only_response("c2", "echo", json!({ "message": "hi" }));
        client.add_text_response("Done");

        let runtime = LoopRuntime::new().with_detection(
            DetectionManager::new_with_config(DetectionConfig {
                loop_threshold: 2,
                stop_threshold: 10,
                ..Default::default()
            })
            .expect("valid detection config"),
        );

        let mut agent =
            BareLoop::new_with_managers(Arc::new(client), registry, make_config(), runtime);
        let result = agent.run("test").await;

        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }

    #[tokio::test]
    async fn test_cancelled_before_run_returns_cancelled() {
        let client = MockClient::new("test");
        client.add_text_response("Hello");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        agent.cancel();
        let result = agent.run("test").await;

        assert!(
            matches!(result, Err(LoopError::Cancelled)),
            "expected Err(LoopError::Cancelled), got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_default_recovery_on_tool_error_returns_soft_result() {
        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "fail", json!({}), "Moving on");

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
    }

    #[tokio::test]
    async fn test_recovery_on_missing_tool_returns_soft_result() {
        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "OK");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
    }

    #[tokio::test]
    async fn test_recovery_noop_reflector_no_retries() {
        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "fail", json!({}), "OK");

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
    }

    #[tokio::test]
    async fn test_recovery_respects_cancellation() {
        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_only_response("tc-1", "fail", json!({}));

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());

        // Cancel before running
        agent.cancel();

        let result = agent.run("Test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cancel_during_dispatch_lands_in_cancelled_state() {
        // Cancellation fired after process_turn has begun dispatching flows
        // through LoopState::Cancelled (not Failed). Uses AlwaysRecoverable so
        // FailingTool's error triggers a retry; the retry loop polls
        // is_cancelled() at the top of each iteration (dispatch.rs), so the
        // cancel signal set here is observed on the next retry attempt.
        struct AlwaysRecoverable;
        impl crate::reflection::Reflector for AlwaysRecoverable {
            fn analyze(
                &self,
                error: &str,
                tool_name: &str,
                _tool_input: &serde_json::Value,
                _tool_schema: Option<&crate::tool::ToolSchema>,
                _context: &crate::reflection::ReflectionContext,
            ) -> Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                crate::reflection::FailureAnalysis,
                                crate::reflection::ReflectionError,
                            >,
                        > + Send
                        + '_,
                >,
            > {
                let error = error.to_string();
                let tool_name = tool_name.to_string();
                Box::pin(async move {
                    Ok(crate::reflection::FailureAnalysis {
                        is_recoverable: true,
                        root_cause: error,
                        severity: crate::reflection::FailureSeverity::Medium,
                        correction: None,
                        context: format!("tool: {tool_name}"),
                    })
                })
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_only_response("tc-1", "fail", json!({}));

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        agent.set_reflector(Arc::new(AlwaysRecoverable));
        agent.set_recovery_strategy(Arc::new(
            crate::reflection::ExponentialBackoffRecovery::new(5)
                .with_base_delay(std::time::Duration::ZERO),
        ));
        let signal = agent.cancel_signal();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            signal.cancel();
        });

        let result = agent.run("Test").await;
        match result {
            Err(LoopError::Cancelled) => {}
            other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
        }
        assert_eq!(
            agent.state(),
            LoopState::Cancelled,
            "cancellation must land in LoopState::Cancelled, not Failed",
        );
    }

    struct StreamingMockClient {
        model: String,
        rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Result<StreamEvent, ApiError>>>>,
    }

    impl StreamingMockClient {
        fn new(
            model: &str,
        ) -> (
            Self,
            tokio::sync::mpsc::Sender<Result<StreamEvent, ApiError>>,
        ) {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent, ApiError>>(8);
            (
                Self {
                    model: model.to_string(),
                    rx: std::sync::Mutex::new(Some(rx)),
                },
                tx,
            )
        }
    }

    impl ApiClient for StreamingMockClient {
        fn model(&self) -> String {
            self.model.clone()
        }

        fn set_model(&self, _model: &str) -> bool {
            false
        }

        fn stream_messages(
            &self,
            _request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            let rx = crate::error::recover_guard(self.rx.lock())
                .take()
                .expect("stream_messages called twice");
            Box::pin(ReceiverStream { rx })
        }

        fn create_message(
            &self,
            _request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
            Box::pin(async { Err(ApiError::api("not implemented")) })
        }
    }

    struct ReceiverStream<T> {
        rx: tokio::sync::mpsc::Receiver<T>,
    }

    impl<T> futures::Stream for ReceiverStream<T> {
        type Item = T;

        fn poll_next(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            self.rx.poll_recv(cx)
        }
    }

    #[tokio::test]
    async fn test_stream_turn_cancelled_mid_stream() {
        let (client, tx) = StreamingMockClient::new("test-model");
        let model = client.model.clone();
        tx.send(Ok(StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model,
            },
        })))
        .await
        .unwrap();

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let signal = agent.cancel_signal();

        let handle = tokio::spawn(async move { agent.run("Hi").await });

        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        signal.cancel();

        // `tx` stays open until function exit, so the channel never closes —
        // the only way `run()` returns is via the cancel signal.
        let result = handle.await.unwrap();
        match result {
            Err(LoopError::Cancelled) => {}
            other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_set_pipeline_injects_self_tools_registry() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hello"}), "done");
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        // Build a builder WITHOUT calling .with_core() — set_pipeline must inject it.
        let builder = ToolPipeline::builder();
        agent.set_pipeline(builder).unwrap();

        let result = agent.run("Echo hello").await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }

    struct TurnNumberCapture {
        turns: Arc<Mutex<Vec<usize>>>,
    }

    impl TurnNumberCapture {
        fn new(shared: Arc<Mutex<Vec<usize>>>) -> Self {
            Self { turns: shared }
        }
    }

    impl crate::middleware::ToolMiddleware for TurnNumberCapture {
        fn name(&self) -> &'static str {
            "turn_capture"
        }

        fn dispatch<'a>(
            &'a self,
            ctx: &'a mut ToolDispatchContext,
            next: &'a ToolPipeline,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = crate::middleware::ToolDispatchResult> + Send + 'a,
            >,
        > {
            crate::error::recover_guard(self.turns.lock()).push(ctx.turn_number);
            next.dispatch(ctx)
        }
    }

    #[tokio::test]
    async fn test_turn_number_is_actual_turn_index() {
        let client = MockClient::new("test-model");
        // Turn 0: model requests tool call, then turn 1: model requests another
        client.add_tool_only_response("tool_0", "echo", json!({"message": "a"}));
        client.add_tool_only_response("tool_1", "echo", json!({"message": "b"}));
        client.add_text_response("done");

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let mut config = make_config();
        config.max_turns = 10;

        let capture = Arc::new(Mutex::new(Vec::<usize>::new()));
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let builder =
            ToolPipeline::builder().with_middleware(TurnNumberCapture::new(Arc::clone(&capture)));
        agent.set_pipeline(builder).unwrap();

        let result = agent.run("test").await;
        assert!(result.is_ok());

        let turns = crate::error::recover_guard(capture.lock()).clone();
        // Tool was called on turn 0 (first turn) and turn 1 (second turn).
        assert_eq!(
            turns.len(),
            2,
            "expected tool calls on 2 turns: got {turns:?}"
        );
        assert_eq!(turns[0], 0, "first tool call should be on turn 0");
        assert_eq!(turns[1], 1, "second tool call should be on turn 1");
        assert!(
            turns.iter().all(|&t| t < 10),
            "turn_number must be actual index, not max_turns (10): got {turns:?}"
        );
    }

    #[tokio::test]
    async fn switch_model_updates_config_and_client() {
        let client = MockClient::new("model-a");
        let client_arc = std::sync::Arc::new(client);
        let tools = ToolRegistry::new();
        let mut config = LoopConfig::default();
        config.model = "model-a".to_string();

        let mut loop_ = BareLoop::new(client_arc.clone(), tools, config);

        loop_.switch_model("model-b").apply().unwrap();

        // Config was updated.
        assert_eq!(loop_.config().model, "model-b");

        // Client was also updated via set_model.
        assert_eq!(client_arc.model(), "model-b");
    }

    #[tokio::test]
    async fn switch_model_notifies_observers() {
        #[derive(Default)]
        struct RecordingObserver {
            switches: Mutex<Vec<(String, String)>>,
        }

        impl crate::observer::LoopObserver for RecordingObserver {
            fn name(&self) -> &'static str {
                "recording"
            }

            fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
                crate::error::recover_guard(self.switches.lock())
                    .push((ctx.from.clone(), ctx.to.clone()));
            }
        }

        let client = std::sync::Arc::new(MockClient::new("m1"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());
        let obs = std::sync::Arc::new(RecordingObserver::default());
        let obs_clone = obs.clone();
        loop_.register_observer(obs);

        loop_.switch_model("m2").apply().unwrap();
        loop_.switch_model("m3").apply().unwrap();

        // Observer should have received both switches.
        let recorded = crate::error::recover_guard(obs_clone.switches.lock());
        assert_eq!(recorded.len(), 2, "should have 2 model-switch events");
        assert_eq!(recorded[0], ("default".to_string(), "m2".to_string()));
        assert_eq!(recorded[1], ("m2".to_string(), "m3".to_string()));
    }

    #[tokio::test]
    async fn switch_model_unsupported_client() {
        struct StaticClient {
            model_name: Arc<std::sync::Mutex<String>>,
        }

        impl ApiClient for StaticClient {
            fn model(&self) -> String {
                crate::error::recover_guard(self.model_name.lock()).clone()
            }
            // Uses default set_model which returns false.

            fn stream_messages(
                &self,
                _request: crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn futures::stream::Stream<Item = Result<StreamEvent, ApiError>>
                        + Send
                        + 'static,
                >,
            > {
                Box::pin(futures::stream::empty())
            }

            fn create_message(
                &self,
                _request: crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<Output = Result<serde_json::Value, ApiError>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async { Ok(serde_json::Value::Null) })
            }
        }

        let client = std::sync::Arc::new(StaticClient {
            model_name: std::sync::Arc::new(std::sync::Mutex::new("static".to_string())),
        });
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        // set_model returns false (unsupported), but apply() is best-effort
        // and still updates config-level state.
        loop_.switch_model("new-model").apply().unwrap();

        // Config was updated even though client didn't support it.
        assert_eq!(loop_.config().model, "new-model");

        // Client's model remains unchanged (no interior mutability).
        assert_eq!(loop_.client.model(), "static");
    }

    #[tokio::test]
    async fn switch_model_updates_fallback_original() {
        let client = std::sync::Arc::new(MockClient::new("primary"));
        let tools = ToolRegistry::new();
        let mut config = LoopConfig::default();
        config.model = "primary".to_string();

        let mut loop_ = BareLoop::new(client, tools, config);

        // Before switch, fallback manager has no original model set.
        assert_eq!(loop_.managers.fallback.original_model(), None);

        loop_.switch_model("new-primary").apply().unwrap();

        // After switch, fallback manager tracks the new primary.
        assert_eq!(
            loop_.managers.fallback.original_model(),
            Some("new-primary".to_string())
        );
    }

    #[tokio::test]
    async fn switch_model_rejects_empty() {
        let client = std::sync::Arc::new(MockClient::new("model"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        let result = loop_.switch_model("").apply();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));

        let result = loop_.switch_model("   ").apply();
        assert!(result.is_err());

        // Model should remain unchanged.
        assert_eq!(loop_.config().model, "default");
    }

    #[tokio::test]
    async fn switch_model_chained() {
        let client = std::sync::Arc::new(MockClient::new("a"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        loop_.switch_model("b").apply().unwrap();
        assert_eq!(loop_.config().model, "b");

        loop_.switch_model("c").apply().unwrap();
        assert_eq!(loop_.config().model, "c");

        loop_.switch_model("d").apply().unwrap();
        assert_eq!(loop_.config().model, "d");
    }

    #[tokio::test]
    async fn switch_model_updates_context_window() {
        let client = std::sync::Arc::new(MockClient::new("big-model"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        let original_cw = loop_.config().context_window;
        assert_ne!(original_cw, 8192);

        loop_
            .switch_model("small-model")
            .with_context_window(8192)
            .apply()
            .unwrap();

        assert_eq!(loop_.config().model, "small-model");
        assert_eq!(loop_.config().context_window, 8192);
    }

    #[tokio::test]
    async fn switch_model_updates_max_tokens() {
        let client = std::sync::Arc::new(MockClient::new("m"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        loop_
            .switch_model("m2")
            .with_max_tokens(4096)
            .apply()
            .unwrap();

        assert_eq!(loop_.config().model, "m2");
        assert_eq!(loop_.config().max_tokens, 4096);
    }

    #[tokio::test]
    async fn switch_model_trims_whitespace() {
        let client = std::sync::Arc::new(MockClient::new("m"));
        let tools = ToolRegistry::new();
        let mut loop_ = BareLoop::new(client, tools, LoopConfig::default());

        loop_.switch_model("  gpt-4o  ").apply().unwrap();
        assert_eq!(loop_.config().model, "gpt-4o");
    }

    #[tokio::test]
    async fn switch_model_resets_fallback_circuit() {
        use crate::fallback::FallbackState;

        let client = std::sync::Arc::new(MockClient::new("primary"));
        let tools = ToolRegistry::new();
        let mut config = LoopConfig::default();
        config.model = "primary".to_string();

        let mut loop_ = BareLoop::new(client, tools, config);

        // Trip the circuit breaker.
        loop_.managers.fallback.set_original_model("primary".into());
        loop_.managers.fallback.set_fallback_model("backup");
        loop_.managers.fallback.transition_to_fallback();
        assert_eq!(loop_.managers.fallback.state(), FallbackState::Fallback);

        // Switch model — circuit should reset to Primary.
        loop_.switch_model("new-primary").apply().unwrap();

        assert_eq!(loop_.managers.fallback.state(), FallbackState::Primary);
        assert_eq!(
            loop_.managers.fallback.original_model(),
            Some("new-primary".to_string())
        );
    }

    #[cfg(feature = "hooks")]
    struct ReasonCaptureHook {
        reason: Mutex<Option<SessionEndReason>>,
    }

    #[cfg(feature = "hooks")]
    impl ReasonCaptureHook {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                reason: Mutex::new(None),
            })
        }

        fn captured(&self) -> Option<SessionEndReason> {
            *crate::error::recover_guard(self.reason.lock())
        }
    }

    #[cfg(feature = "hooks")]
    impl Hook for ReasonCaptureHook {
        fn name(&self) -> &'static str {
            "ReasonCaptureHook"
        }

        fn on_session_end(&self, ctx: &HookSessionEndContext) {
            *crate::error::recover_guard(self.reason.lock()) = Some(ctx.reason);
        }
    }

    #[cfg(feature = "hooks")]
    fn loop_with_reason_hook() -> (BareLoop<MockClient>, Arc<ReasonCaptureHook>) {
        let hook = ReasonCaptureHook::new();
        let executor = Arc::new(HookExecutor::new().with_hook(hook.clone()));
        let config = LoopConfig {
            max_turns: 5,
            ..LoopConfig::default()
        };
        let mut loop_ = BareLoop::new(
            Arc::new(MockClient::new("test")),
            ToolRegistry::new(),
            config,
        );
        loop_.set_hook_executor(executor);
        (loop_, hook)
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn session_end_reason_complete() {
        let (mut loop_, hook) = loop_with_reason_hook();
        // Normal completion: state is Completed, not cancelled, under max_turns.
        loop_.budget.success = true;
        loop_.budget.total_turns = 2;

        loop_.notify_session_end(&loop_.budget.clone(), Duration::from_millis(100));

        assert_eq!(hook.captured(), Some(SessionEndReason::Complete));
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn session_end_reason_cancelled() {
        let (mut loop_, hook) = loop_with_reason_hook();
        // Cancel signal fired — success is true (not Failed) but cancelled.
        loop_.budget.success = true;
        loop_.budget.total_turns = 2;
        loop_.cancelled.cancel();

        loop_.notify_session_end(&loop_.budget.clone(), Duration::from_millis(100));

        assert_eq!(hook.captured(), Some(SessionEndReason::Cancelled));
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn session_end_reason_max_turns() {
        let (mut loop_, hook) = loop_with_reason_hook();
        // Hit max_turns: total_turns == max_turns, not cancelled, success true.
        loop_.budget.success = true;
        loop_.budget.total_turns = 5; // equals config.max_turns

        loop_.notify_session_end(&loop_.budget.clone(), Duration::from_millis(100));

        assert_eq!(hook.captured(), Some(SessionEndReason::MaxTurns));
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn session_end_reason_error() {
        let (mut loop_, hook) = loop_with_reason_hook();
        // Generic failure: success false, no context-overflow keyword.
        loop_.budget.success = false;
        loop_.budget.error = Some("API connection refused".to_string());

        loop_.notify_session_end(&loop_.budget.clone(), Duration::from_millis(100));

        assert_eq!(hook.captured(), Some(SessionEndReason::Error));
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn session_end_reason_context_overflow() {
        let (mut loop_, hook) = loop_with_reason_hook();
        // Failure with context-overflow keyword in the error message.
        loop_.budget.success = false;
        loop_.budget.error = Some("context length exceeded".to_string());

        loop_.notify_session_end(&loop_.budget.clone(), Duration::from_millis(100));

        assert_eq!(hook.captured(), Some(SessionEndReason::ContextOverflow));
    }

    #[tokio::test]
    async fn process_turn_cancel_during_streaming_returns_fast() {
        let (client, tx) = StreamingMockClient::new("test-model");
        tx.send(Ok(StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg-1".into(),
                role: "assistant".into(),
                model: "test-model".into(),
            },
        })))
        .await
        .unwrap();

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());
        let signal = agent.cancel_signal();

        let handle = tokio::spawn(async move { agent.run("Hi").await });

        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        let start = Instant::now();
        signal.cancel();

        let result = handle.await.unwrap();
        let elapsed = start.elapsed();

        match result {
            Err(LoopError::Cancelled) => {}
            other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_secs(5),
            "cancel during streaming should return fast; elapsed {elapsed:?}",
        );
        assert_eq!(
            observer.turn_ends.load(Ordering::SeqCst),
            1,
            "on_turn_end should fire once on cancel",
        );
    }

    #[tokio::test]
    async fn process_turn_cancel_during_dispatch_fires_turn_end() {
        struct SlowTool {
            notify: Arc<tokio::sync::Notify>,
        }
        impl Tool for SlowTool {
            fn name(&self) -> &'static str {
                "slow"
            }
            fn description(&self) -> &'static str {
                "Blocks until notified"
            }
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    tool: "slow".into(),
                    description: "Blocks until notified".into(),
                    input_schema: json!({"type": "object", "properties": {}}),
                }
            }
            fn call(
                &self,
                _input: Value,
                _ctx: &ToolContext,
            ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
            {
                let notify = self.notify.clone();
                Box::pin(async move {
                    notify.notified().await;
                    Ok(ToolOutput::text("done"))
                })
            }
        }

        let notify = Arc::new(tokio::sync::Notify::new());
        let mut registry = ToolRegistry::new();
        registry.register(SlowTool {
            notify: notify.clone(),
        });

        let client = MockClient::new("test");
        client.add_tool_only_response("tc-1", "slow", json!({}));

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        let observer = Arc::new(CountingObserver::new());
        agent.register_observer(observer.clone());
        let signal = agent.cancel_signal();

        let handle = tokio::spawn(async move { agent.run("Use slow tool").await });

        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        signal.cancel();

        let result = handle.await.unwrap();
        match result {
            Err(LoopError::Cancelled) => {}
            other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
        }
        assert_eq!(
            observer.turn_ends.load(Ordering::SeqCst),
            1,
            "on_turn_end(false) must fire on cancel during dispatch",
        );
        assert_eq!(
            observer.session_ends.load(Ordering::SeqCst),
            1,
            "on_session_end must fire via finalize after cancel",
        );
    }

    #[tokio::test]
    async fn process_turn_cancel_during_recovery_backoff_returns_fast() {
        struct AlwaysRecoverable;
        impl crate::reflection::Reflector for AlwaysRecoverable {
            fn analyze(
                &self,
                error: &str,
                tool_name: &str,
                _tool_input: &serde_json::Value,
                _tool_schema: Option<&crate::tool::ToolSchema>,
                _context: &crate::reflection::ReflectionContext,
            ) -> Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                crate::reflection::FailureAnalysis,
                                crate::reflection::ReflectionError,
                            >,
                        > + Send
                        + '_,
                >,
            > {
                let error = error.to_string();
                let tool_name = tool_name.to_string();
                Box::pin(async move {
                    Ok(crate::reflection::FailureAnalysis {
                        is_recoverable: true,
                        root_cause: error,
                        severity: crate::reflection::FailureSeverity::Medium,
                        correction: None,
                        context: format!("tool: {tool_name}"),
                    })
                })
            }
        }

        let client = MockClient::new("test");
        client.add_tool_only_response("tc-1", "fail", json!({}));

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        agent.set_reflector(Arc::new(AlwaysRecoverable));
        agent.set_recovery_strategy(Arc::new(
            crate::reflection::ExponentialBackoffRecovery::new(5)
                .with_base_delay(Duration::from_mins(1)),
        ));
        let signal = agent.cancel_signal();

        let handle = tokio::spawn(async move { agent.run("Use failing tool").await });

        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        let start = Instant::now();
        signal.cancel();

        let result = handle.await.unwrap();
        let elapsed = start.elapsed();

        match result {
            Err(LoopError::Cancelled) => {}
            other => panic!("expected Err(LoopError::Cancelled), got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_secs(5),
            "cancel during recovery backoff should return fast, not wait 60s; elapsed {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn test_rate_limit_escalation_feeds_circuit_breaker() {
        use crate::fallback::FallbackManager;
        use crate::runtime::LoopRuntime;
        use crate::stream::handler::{
            RateLimitConfig, StreamHandler, StreamRetryConfig, StreamTimeoutConfig,
        };

        // Every stream attempt is rate-limited, so the handler escalates on the
        // first 429 (fallback_after_retries = 0).
        struct AlwaysRateLimitClient;
        impl ApiClient for AlwaysRateLimitClient {
            fn model(&self) -> String {
                "primary-model".to_string()
            }
            fn stream_messages(
                &self,
                _request: crate::api::StreamRequest,
            ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
            {
                Box::pin(futures::stream::once(async {
                    Err(ApiError::RateLimit {
                        retry_after: None,
                        message: "slow down".into(),
                    })
                }))
            }
            fn create_message(
                &self,
                _request: crate::api::StreamRequest,
            ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
                Box::pin(async { Ok(json!({})) })
            }
        }

        let handler = StreamHandler::new()
            .with_config(
                StreamTimeoutConfig {
                    fallback_to_non_streaming: false,
                    ..Default::default()
                },
                StreamRetryConfig::default(),
            )
            .with_rate_limit_config(RateLimitConfig {
                fallback_after_retries: 0,
                default_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
                ..Default::default()
            });

        // Circuit breaker: trips on a single model failure (threshold = 1) and
        // has a fallback model configured.
        let mut runtime = LoopRuntime::new();
        runtime.fallback = FallbackManager::new_with_fallback("primary-model".to_string(), 1);
        runtime.fallback.set_fallback_model("fallback-model");
        runtime.set_stream_handler(handler);

        let config = make_config();
        let client = Arc::new(AlwaysRateLimitClient);
        let mut agent = BareLoop::new_with_managers(client, ToolRegistry::new(), config, runtime);

        let result = agent.run("Hi").await;
        assert!(result.is_err(), "rate-limited turn should fail");

        // The escalation arm called record_model_failure(); with threshold 1 the
        // breaker tripped into Fallback state.
        assert!(
            agent.managers.fallback.is_using_fallback(),
            "escalation should trip the circuit breaker to the fallback model"
        );
    }

    // ==========================================================
    // ContextContributor turn-boundary injection
    // ==========================================================
    //
    // A recording client that snapshots the inbound conversation on every
    // stream_messages call, so the tests can assert exactly what the model
    // received (including contributor-injected messages).

    #[derive(Clone)]
    struct RecordingClient {
        responses: Arc<Mutex<Vec<Vec<StreamEvent>>>>,
        seen: Arc<Mutex<Vec<Vec<Message>>>>,
        seen_options: Arc<Mutex<Vec<crate::structured::RequestOptions>>>,
        model_name: Arc<Mutex<String>>,
    }

    impl RecordingClient {
        fn new(model: &str) -> Self {
            Self {
                responses: Arc::new(Mutex::new(Vec::new())),
                seen: Arc::new(Mutex::new(Vec::new())),
                seen_options: Arc::new(Mutex::new(Vec::new())),
                model_name: Arc::new(Mutex::new(model.to_string())),
            }
        }

        fn add_text_response(&self, text: &str) {
            let events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_test".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::text(text)),
                }),
                StreamEvent::IndexedDelta(IndexedDelta {
                    index: 0,
                    delta: DeltaPart::Text {
                        text: text.to_string(),
                    },
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("end_turn".to_string()),
                    },
                    usage: Some(Usage::new(10, 20)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(events);
        }

        fn first_seen(&self) -> Vec<Message> {
            crate::error::recover_guard(self.seen.lock())
                .first()
                .expect("at least one stream_messages call")
                .clone()
        }

        fn add_tool_then_text(
            &self,
            tool_id: &str,
            tool_name: &str,
            tool_input: Value,
            final_text: &str,
        ) {
            let tool_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_tool".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::tool_call(tool_id, tool_name, tool_input)),
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("tool_call".to_string()),
                    },
                    usage: Some(Usage::new(50, 10)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(tool_events);

            let text_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_final".into(),
                        role: "assistant".into(),
                        model: crate::error::recover_guard(self.model_name.lock()).clone(),
                    },
                }),
                StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::text(final_text)),
                }),
                StreamEvent::IndexedDelta(IndexedDelta {
                    index: 0,
                    delta: DeltaPart::Text {
                        text: final_text.to_string(),
                    },
                }),
                StreamEvent::PartStop,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("end_turn".to_string()),
                    },
                    usage: Some(Usage::new(30, 15)),
                }),
                StreamEvent::MessageStop,
            ];
            crate::error::recover_guard(self.responses.lock()).push(text_events);
        }

        fn call_count(&self) -> usize {
            crate::error::recover_guard(self.seen.lock()).len()
        }

        fn first_options(&self) -> crate::structured::RequestOptions {
            crate::error::recover_guard(self.seen_options.lock())
                .first()
                .expect("at least one stream_messages_with_options call")
                .clone()
        }
    }

    impl ApiClient for RecordingClient {
        fn model(&self) -> String {
            crate::error::recover_guard(self.model_name.lock()).clone()
        }

        fn set_model(&self, model: &str) -> bool {
            if model.trim().is_empty() {
                return false;
            }
            *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
            true
        }

        fn stream_messages(
            &self,
            request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            let messages = request.messages;
            crate::error::recover_guard(self.seen.lock()).push(messages);
            let mut guard = crate::error::recover_guard(self.responses.lock());
            if let Some(events) = guard.pop_front() {
                let events: Vec<Result<StreamEvent, ApiError>> =
                    events.into_iter().map(Ok).collect();
                Box::pin(futures::stream::iter(events))
            } else {
                let err = ApiError::api("No more mock responses");
                Box::pin(futures::stream::iter(vec![Err(err)]))
            }
        }

        fn stream_messages_with_options(
            &self,
            request: crate::api::StreamRequest,
            options: crate::structured::RequestOptions,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            let messages = request.messages;
            crate::error::recover_guard(self.seen.lock()).push(messages);
            crate::error::recover_guard(self.seen_options.lock()).push(options);
            let mut guard = crate::error::recover_guard(self.responses.lock());
            if let Some(events) = guard.pop_front() {
                let events: Vec<Result<StreamEvent, ApiError>> =
                    events.into_iter().map(Ok).collect();
                Box::pin(futures::stream::iter(events))
            } else {
                let err = ApiError::api("No more mock responses");
                Box::pin(futures::stream::iter(vec![Err(err)]))
            }
        }

        fn create_message(
            &self,
            _request: crate::api::StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
            Box::pin(async { Ok(json!({"content": []})) })
        }
    }

    // A contributor that always returns the same System reminder.
    struct StaticReminder(String);
    impl ContextContributor for StaticReminder {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            Some(Message::new(
                Role::System,
                vec![MessagePart::text(self.0.clone())],
            ))
        }
    }

    // A contributor that never injects.
    struct NeverContributor;
    impl ContextContributor for NeverContributor {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            None
        }
    }

    // A contributor that counts how many times it was consulted, injecting
    // nothing. Shared by the cadence tests.
    struct CountingContributor {
        calls: Arc<AtomicUsize>,
    }
    impl ContextContributor for CountingContributor {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    // A contributor that records the turn number it was consulted at.
    struct CapturingContributor {
        seen_turns: Arc<Mutex<Vec<usize>>>,
    }
    impl ContextContributor for CapturingContributor {
        fn contribute(&self, ctx: &ContributorContext<'_>) -> Option<Message> {
            crate::error::recover_guard(self.seen_turns.lock()).push(ctx.turn);
            None
        }
    }

    // A contributor that counts how many times it was consulted.
    fn contributor_config() -> LoopConfig {
        LoopConfig {
            max_turns: 10,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_contributor_message_prepended() {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        agent.add_contributor(Box::new(StaticReminder("stay on task".into())));
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        // The injected message reached the model: it's in the captured inbound
        // conversation after the user message.
        let seen = client.first_seen();
        let texts: Vec<&str> = seen
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| {
                m.parts.iter().filter_map(|p| match p {
                    MessagePart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("stay on task")));

        // And it persists in the loop's own conversation history.
        let persisted = agent.conversation();
        assert!(persisted.iter().any(|m| m.role == Role::System
            && m.parts.iter().any(
                |p| matches!(p, MessagePart::Text { text } if text.contains("stay on task"))
            )));
    }

    #[tokio::test]
    async fn test_no_contributors_no_change() {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        // No add_contributor call.
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let seen = client.first_seen();
        // No System messages reached the model.
        assert!(
            !seen.iter().any(|m| m.role == Role::System),
            "no contributor registered, so no System message should appear"
        );
        // Exactly one user message (the "Hi").
        let user_count = seen.iter().filter(|m| m.role == Role::User).count();
        assert_eq!(user_count, 1, "baseline conversation has one user message");
    }

    #[tokio::test]
    async fn test_contributor_returning_none_injects_nothing() {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        agent.add_contributor(Box::new(NeverContributor));
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let seen = client.first_seen();
        assert!(
            !seen.iter().any(|m| m.role == Role::System),
            "None-returning contributor must inject nothing"
        );
    }

    #[tokio::test]
    async fn test_multiple_contributors_order_preserved() {
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        agent.add_contributor(Box::new(StaticReminder("first".into())));
        agent.add_contributor(Box::new(StaticReminder("second".into())));
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        // Find positions of the two reminders in the persisted history.
        let persisted = agent.conversation();
        let pos = |needle: &str| -> Option<usize> {
            persisted.iter().position(|m| {
                m.role == Role::System
                    && m.parts
                        .iter()
                        .any(|p| matches!(p, MessagePart::Text { text } if text == needle))
            })
        };
        let first = pos("first").expect("'first' reminder persisted");
        let second = pos("second").expect("'second' reminder persisted");
        assert!(first < second, "registration order must be preserved");
    }

    #[tokio::test]
    async fn test_contributor_does_not_affect_turn_count() {
        // Two-turn session: tool call then end_turn.
        let with_contrib = {
            let client = RecordingClient::new("test-model");
            client.add_text_response("done");
            let config = contributor_config();
            let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
            agent.add_contributor(Box::new(StaticReminder("remind".into())));
            agent.run("Hi").await.unwrap().total_turns
        };
        let without_contrib = {
            let client = RecordingClient::new("test-model");
            client.add_text_response("done");
            let config = contributor_config();
            let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
            agent.run("Hi").await.unwrap().total_turns
        };
        assert_eq!(
            with_contrib, without_contrib,
            "injection must not perturb turn counting"
        );
    }

    #[tokio::test]
    async fn test_contributor_fires_every_turn() {
        // A single contributor + a single-turn run must show exactly one call.
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let counter = Arc::new(AtomicUsize::new(0));
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let c = Arc::clone(&counter);
        agent.add_contributor(Box::new(CountingContributor { calls: c }));
        agent.run("Hi").await.unwrap();

        // One turn ran; the contributor was consulted once.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        // And the model was called exactly once (proving the single turn).
        assert_eq!(agent.budget.total_turns, 1);
    }

    #[tokio::test]
    async fn test_contributor_fires_across_two_turns() {
        // Two-turn session via a tool: turn 1 = tool_call, turn 2 = end_turn.
        // The contributor must be consulted on BOTH turns.
        let client = RecordingClient::new("test-model");
        client.add_tool_then_text("t1", "echo", json!({"message": "hi"}), "all done");
        let counter = Arc::new(AtomicUsize::new(0));

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let c = Arc::clone(&counter);
        agent.add_contributor(Box::new(CountingContributor { calls: c }));
        let result = agent.run("Echo hi").await.unwrap();
        assert_eq!(result.total_turns, 2, "tool_call turn + end_turn");
        assert_eq!(
            counter.load(Ordering::Relaxed),
            2,
            "contributor must fire on every turn"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "configuration setters must be called before run()")]
    fn test_add_contributor_panics_after_session_start() {
        let client = MockClient::new("test-model");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        // initialize() moves the loop out of Idle — subsequent add_contributor
        // must panic in debug builds (matches set_reflector's contract).
        let cfg = LoopConfig {
            max_turns: 10,
            ..Default::default()
        };
        // Box the future so we can drop it without awaiting; initialize is the
        // state transition under test.
        {
            let fut = agent.initialize(&cfg);
            let mut fut = std::pin::pin!(fut);
            futures::executor::block_on(fut.as_mut()).unwrap();
        }
        agent.add_contributor(Box::new(StaticReminder("late".into())));
    }

    #[tokio::test]
    async fn test_contributor_sees_turn_number() {
        // Assert the ContributorContext.turn matches the engine's turn counter
        // at consultation time. Captures the value across a 2-turn session.
        let client = RecordingClient::new("test-model");
        client.add_tool_then_text("t1", "echo", json!({"message": "x"}), "done");
        let seen_turns = Arc::new(Mutex::new(Vec::<usize>::new()));

        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let s = Arc::clone(&seen_turns);
        agent.add_contributor(Box::new(CapturingContributor { seen_turns: s }));
        agent.run("go").await.unwrap();

        let turns = crate::error::recover_guard(seen_turns.lock()).clone();
        assert_eq!(turns, vec![0, 1], "turn numbers are 0-indexed and per-turn");
    }

    // Suppress unused-warning for RecordingClient::call_count when no test
    // references it; kept for future diagnostics.
    #[allow(dead_code)]
    fn _suppress_recording_client_dead_code(c: &RecordingClient) {
        let _ = c.call_count();
    }

    // ==========================================================
    // RequestOptions engine plumbing
    // ==========================================================

    #[tokio::test]
    async fn test_request_options_default_is_unconstrained() {
        // A fresh BareLoop has default RequestOptions — the engine reproduces
        // v0.1.0 behavior (no tool_constraint).
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        // No set_request_options call — default path.
        agent.run("Hi").await.unwrap();

        let opts = client.first_options();
        assert!(
            matches!(
                opts.tool_constraint,
                crate::structured::ToolConstraint::None
            ),
            "default request options must be unconstrained"
        );
    }

    #[tokio::test]
    async fn test_request_options_strict_reaches_provider() {
        // The critical end-to-end proof: a tool_constraint: Strict set on the
        // loop reaches the provider's stream_messages_with_options call.
        let client = RecordingClient::new("test-model");
        client.add_text_response("done");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), config);
        agent.set_request_options(
            crate::structured::RequestOptions::new()
                .with_tool_constraint(crate::structured::ToolConstraint::Strict),
        );
        agent.run("Hi").await.unwrap();

        let opts = client.first_options();
        assert!(
            matches!(
                opts.tool_constraint,
                crate::structured::ToolConstraint::Strict
            ),
            "Strict set on the loop must reach the provider"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "configuration setters must be called before run()")]
    fn test_set_request_options_panics_after_session_start() {
        let client = MockClient::new("test-model");
        let config = contributor_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let cfg = LoopConfig {
            max_turns: 10,
            ..Default::default()
        };
        {
            let fut = agent.initialize(&cfg);
            let mut fut = std::pin::pin!(fut);
            futures::executor::block_on(fut.as_mut()).unwrap();
        }
        agent.set_request_options(crate::structured::RequestOptions::default());
    }

    // ==========================================================
    // ConstrainedProfile::apply
    // ==========================================================

    #[tokio::test]
    async fn test_constrained_apply_wires_pipeline_and_contributor() {
        // Apply() sets the small-model pipeline and registers a GoalReminder. To prove
        // the contributor wiring without driving 5 turns (each turn ends on
        // end_turn, so reaching turn 5 needs a long tool-call chain), we add
        // a cadence-1 GoalReminder on top: it fires on turn 1, so a single
        // tool-then-text session (2 turns) is enough.
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let client = RecordingClient::new("test-model");
        client.add_tool_then_text("t1", "echo", json!({"message": "x"}), "done");

        let mut agent = BareLoop::new(Arc::new(client.clone()), registry, contributor_config());
        // apply() wires the pipeline + a cadence-5 GoalReminder.
        crate::presets::ConstrainedProfile::apply(&mut agent).unwrap();
        // Add a cadence-1 reminder so it fires this session.
        agent.add_contributor(Box::new(crate::presets::GoalReminder::new(1)));

        let result = agent.run("ship the demo goal").await.unwrap();
        // Tool-call turn + end_turn = 2 turns.
        assert!(result.total_turns >= 1);

        // The contributor fired: a Role::System message carrying the first
        // user message text reached the provider on some turn's outbound
        // conversation. Scan all recorded calls (the reminder fires on turn 1,
        // not turn 0).
        let all_seen = crate::error::recover_guard(client.seen.lock()).clone();
        let has_reminder = all_seen.iter().flatten().any(|m| {
            m.role == Role::System
                && m.parts.iter().any(
                    |p| matches!(p, MessagePart::Text { text } if text.contains("ship the demo goal")),
                )
        });
        assert!(
            has_reminder,
            "GoalReminder (cadence 1) should have injected the goal text as a System message"
        );
    }

    #[tokio::test]
    async fn test_on_thinking_delta_fires_per_thinking_delta() {
        struct ThinkingRecorder {
            deltas: Arc<Mutex<Vec<(usize, String)>>>,
        }
        impl crate::observer::LoopObserver for ThinkingRecorder {
            fn name(&self) -> &'static str {
                "thinking-recorder"
            }
            fn on_thinking_delta(&self, ctx: &crate::observer::ThinkingDeltaContext) {
                crate::error::recover_guard(self.deltas.lock()).push((ctx.turn, ctx.delta.clone()));
            }
        }

        let client = MockClient::new("test-model");
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg-1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::PartStart(PartStart {
                index: 1,
                part: None,
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 1,
                delta: DeltaPart::Thinking {
                    text: "First reasoning".into(),
                },
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 1,
                delta: DeltaPart::Thinking {
                    text: " chunk".into(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(MessagePart::text("ignored")),
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: "final answer".into(),
                },
            }),
            StreamEvent::PartStop,
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                },
                usage: None,
            }),
            StreamEvent::MessageStop,
        ];
        client.add_events(events);

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::new(ThinkingRecorder {
            deltas: Arc::clone(&captured),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let captured = crate::error::recover_guard(captured.lock());
        assert_eq!(
            captured.len(),
            2,
            "one on_thinking_delta per Thinking delta"
        );
        let joined: String = captured.iter().map(|(_, d)| d.as_str()).collect();
        assert_eq!(joined, "First reasoning chunk");
        assert_eq!(captured[0].0, 0, "turn number matches budget.total_turns");
    }

    #[tokio::test]
    async fn test_on_thinking_delta_independent_of_text_delta() {
        struct MixedRecorder {
            text_calls: Arc<Mutex<usize>>,
            thinking_calls: Arc<Mutex<usize>>,
        }
        impl crate::observer::LoopObserver for MixedRecorder {
            fn name(&self) -> &'static str {
                "mixed-recorder"
            }
            fn on_text_delta(&self, _ctx: &crate::observer::TextDeltaContext) {
                *crate::error::recover_guard(self.text_calls.lock()) += 1;
            }
            fn on_thinking_delta(&self, _ctx: &crate::observer::ThinkingDeltaContext) {
                *crate::error::recover_guard(self.thinking_calls.lock()) += 1;
            }
        }

        let client = MockClient::new("test-model");
        let events = vec![
            StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg-1".into(),
                    role: "assistant".into(),
                    model: "test-model".into(),
                },
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 1,
                delta: DeltaPart::Thinking {
                    text: "reasoning".into(),
                },
            }),
            StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text {
                    text: "answer".into(),
                },
            }),
            StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                },
                usage: None,
            }),
            StreamEvent::MessageStop,
        ];
        client.add_events(events);

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let text_calls = Arc::new(Mutex::new(0usize));
        let thinking_calls = Arc::new(Mutex::new(0usize));
        let recorder = Arc::new(MixedRecorder {
            text_calls: Arc::clone(&text_calls),
            thinking_calls: Arc::clone(&thinking_calls),
        });
        agent.register_observer(recorder as Arc<dyn crate::observer::LoopObserver>);

        agent.run("Hi").await.unwrap();

        assert_eq!(
            *crate::error::recover_guard(text_calls.lock()),
            1,
            "text callback fires once (for the Text delta)"
        );
        assert_eq!(
            *crate::error::recover_guard(thinking_calls.lock()),
            1,
            "thinking callback fires once (for the Thinking delta)"
        );
    }

    #[tokio::test]
    async fn fluent_with_chain_builds_a_working_loop() {
        let client = MockClient::new("test-model");
        client.add_text_response("done");

        let observer = Arc::new(CountingObserver::new());
        let registered: Arc<dyn crate::observer::LoopObserver> = observer.clone();

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config())
            .with_observer(registered)
            .with_reflector(Arc::new(NoopReflector))
            .with_request_options(RequestOptions::default());

        let result = agent.run("Hi").await.unwrap();

        assert!(result.success, "the fluent-built loop runs a turn");
        assert_eq!(
            observer.turn_starts.load(Ordering::SeqCst),
            1,
            "with_observer registered the observer (it received the turn event)"
        );
    }

    #[test]
    fn fluent_with_observer_equivalent_to_register_observer() {
        let client = MockClient::new("test-model");
        let observer: Arc<dyn crate::observer::LoopObserver> = Arc::new(CountingObserver::new());

        let fluent = BareLoop::new(Arc::new(client.clone()), ToolRegistry::new(), make_config())
            .with_observer(Arc::clone(&observer));

        let mut imperative = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        imperative.register_observer(Arc::clone(&observer));

        assert_eq!(
            fluent.managers.observers().len(),
            imperative.managers.observers().len(),
            "both paths register the same number of observers"
        );
    }
}
