//! `BareLoop` — the framework's default agent loop implementation.
//!
//! [`BareLoop`] — a generic, framework-level agent
//! loop that orchestrates the full lifecycle of an LLM-based agent session:
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
//!   executed one after another so cancellation is checked between each.
//!   Parallel execution may be added in a future release.
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
#[cfg(feature = "hooks")]
use crate::hooks::HookAction;
#[cfg(feature = "hooks")]
use crate::hooks::HookExecutor;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    CompactTrigger, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
};
use crate::message::{Message, MessagePart, Role, ToolContent};
use crate::middleware::{ToolDispatchContext, ToolPipeline, ToolPipelineBuilder};
use crate::observer::{
    FallbackContext, ResponseContext, StreamContext, StreamFailureContext, TurnEndContext,
    TurnStartContext,
};
use crate::reflection::{
    ExponentialBackoffRecovery, NoopReflector, RecoveryAction, RecoveryStrategy, ReflectionContext,
    Reflector,
};
use crate::runtime::LoopRuntime;
use crate::stream::handler::StreamHandler;
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
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
    state: LoopState,

    /// Session-level accumulator for turn counts, token usage, and tool calls.
    ///
    /// Reused across turns in a single [`run()`](crate::engine::loop_core::Loop::run) call. Reset
    /// to `SessionResult::default()` in [`initialize`](crate::engine::loop_core::Loop::initialize).
    budget: SessionResult,

    /// Session start time, set by [`initialize`](crate::engine::loop_core::Loop::initialize).
    session_start: Option<Instant>,

    /// Optional callback invoked for each text delta during streaming.
    ///
    /// Set via [`set_text_streamer`](BareLoop::set_text_streamer).
    /// When set, called from `stream_turn` on every `IndexedDelta` with
    /// a `Text` payload, enabling real-time token display.
    #[allow(clippy::type_complexity)]
    text_streamer: Option<Arc<dyn Fn(&str) + Send + Sync>>,
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

    /// Set the [`Reflector`] for tool-error analysis.
    ///
    /// Replaces the default [`NoopReflector`] with a caller-supplied
    /// implementation. Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_reflector(Arc::new(MyReflector));
    /// ```
    pub fn set_reflector(&mut self, reflector: Arc<dyn Reflector>) {
        self.reflector = reflector;
    }

    /// Set the [`RecoveryStrategy`] for tool-error recovery.
    ///
    /// Replaces the default [`ExponentialBackoffRecovery`] with a
    /// caller-supplied implementation. Must be called before
    /// [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_recovery_strategy(Arc::new(MyStrategy));
    /// ```
    pub fn set_recovery_strategy(&mut self, strategy: Arc<dyn RecoveryStrategy>) {
        self.recovery = strategy;
    }

    /// Set the [`ContextManager`] for automatic context compaction.
    ///
    /// When set, the loop checks token usage after each turn and
    /// triggers compaction when usage exceeds the configured threshold.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
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
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_context_manager(Arc::new(manager));
    /// ```
    pub fn set_context_manager(&mut self, manager: Arc<ContextManager>) {
        self.managers.set_context_manager(manager);
    }

    /// Set the [`StreamHandler`] for resilient streaming with retries,
    /// timeouts, and fallback to non-streaming.
    ///
    /// When set, the loop delegates streaming to the handler instead of
    /// using the inline streaming logic. Must be called before
    /// [`run()`](crate::engine::loop_core::Loop::run).
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
        self.managers.set_hook_executor(executor);
    }

    /// Set the [`ToolHealthRegistry`] for per-tool health tracking.
    ///
    /// When set, records success/failure and latency for every tool
    /// dispatch. Tools that exceed the failure threshold have their
    /// circuit breaker opened, blocking subsequent calls until recovery.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
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
        self.managers.set_health_registry(registry);
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// Replaces the default (no pipeline) with a caller-supplied
    /// [`ToolPipeline`]. When set, tool calls flow through the
    /// pipeline's middleware chain before reaching the registry.
    /// Must be called before [`run()`](crate::engine::loop_core::Loop::run).
    ///
    /// Build the pipeline using [`ToolPipeline::builder()`], adding middleware
    /// layers **without** calling `.core()` — the registry is injected
    /// automatically from `self.tools` so that schema generation and dispatch
    /// always share the same underlying registry:
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::engine::middleware::{ToolPipeline, TimeoutMiddleware};
    ///
    /// let builder = ToolPipeline::builder()
    ///     .with(TimeoutMiddleware::from_secs(30));
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
        let pipeline = builder
            .core(Arc::clone(&self.tools))
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
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::{Arc, Mutex};
    ///
    /// let buffer = Arc::new(Mutex::new(String::new()));
    /// let buf = Arc::clone(&buffer);
    /// agent.set_text_streamer(Arc::new(move |delta| {
    ///     print!("{delta}");
    ///     buf.lock().unwrap().push_str(delta);
    /// }));
    /// ```
    pub fn set_text_streamer(&mut self, f: Arc<dyn Fn(&str) + Send + Sync>) {
        self.text_streamer = Some(f);
    }

    // ==================================================
    // Run helpers
    // ==================================================

    /// Extract per-turn token counts from optional [`Usage`].
    fn usage_tokens(usage: Option<&Usage>) -> (u64, u64) {
        match usage {
            Some(u) => (u64::from(u.input_tokens), u64::from(u.output_tokens)),
            None => (0, 0),
        }
    }

    /// Dispatch tool calls, push the result message, and record the count.
    ///
    /// Notifies observers: turn-end on success, turn-end on error.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if the cancellation token is set.
    /// Returns [`LoopError::Api`] if loop or convergence detection forces
    /// an abort, or if the underlying tool dispatch fails.
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
                    turn: budget.total_turns,
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
                    turn: budget.total_turns,
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
}

// ==================================================
// Loop trait implementation
// ==================================================

impl<C: ApiClient> crate::engine::loop_core::Loop for BareLoop<C> {
    fn initialize<'a>(
        &'a mut self,
        config: &'a crate::config::LoopConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        Box::pin(async move {
            config.validate().map_err(LoopError::Config)?;

            self.state = LoopState::Processing { turn: 0 };
            self.budget = SessionResult::default();
            self.session_start = Some(Instant::now());
            self.config = config.clone();
            self.managers.reset_all();
            self.notify_session_start();
            Ok(())
        })
    }

    #[allow(clippy::too_many_lines)]
    fn process_turn<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            // On the first turn, `input` is the user's message (non-empty).
            // On continuation turns, `input` is "" — the conversation already
            // has the tool results appended from the previous turn.
            if !input.is_empty() {
                self.conversation.push(Message::user(input));
            }

            let turn_start = Instant::now();

            self.managers.observers().on_turn_start(&TurnStartContext {
                turn: self.budget.total_turns,
                query: input.to_string(),
            });

            // Check cancellation before the API call.
            if self.is_cancelled() {
                self.state = LoopState::Failed {
                    error: "cancelled".into(),
                };
                return Err(LoopError::Cancelled);
            }

            let stream_result = self.stream_turn().await;
            let (assistant_msg, usage, _stream_stop) = match stream_result {
                Ok(value) => {
                    let (msg, usage, stop) = value;
                    self.managers.fallback.record_model_success();
                    let (in_tok, out_tok) = Self::usage_tokens(usage.as_ref());
                    self.managers.observers().on_stream_success(&StreamContext {
                        turn: self.budget.total_turns,
                        model: self.client.model().to_string(),
                        input_tokens: in_tok,
                        output_tokens: out_tok,
                    });
                    (msg, usage, stop)
                }
                Err(e) => {
                    // Record the failure with the fallback circuit breaker.
                    //
                    // Note: BareLoop records API failures and trips the
                    // circuit breaker but does **not** automatically retry
                    // with the fallback model. The `FallbackManager` is
                    // infrastructure for downstream consumers that hold
                    // multiple API clients. BareLoop has a single client,
                    // so the error is propagated after recording.
                    let tripped = self.managers.fallback.record_api_failure();
                    if tripped {
                        let from = self.client.model();
                        if let Some(to) = self.managers.fallback.fallback_model() {
                            tracing::warn!(from, to, "fallback manager tripped");
                            self.managers.observers().on_fallback(&FallbackContext {
                                from: from.to_string(),
                                to,
                            });
                        }
                    }

                    self.managers
                        .observers()
                        .on_stream_failure(&StreamFailureContext {
                            turn: self.budget.total_turns,
                            model: self.client.model().to_string(),
                            error: e.clone(),
                        });

                    self.state = LoopState::Failed {
                        error: e.to_string(),
                    };
                    return Err(e);
                }
            };

            // Accumulate usage into session budget.
            if let Some(u) = &usage {
                self.budget.input_tokens = self
                    .budget
                    .input_tokens
                    .saturating_add(u64::from(u.input_tokens));
                self.budget.output_tokens = self
                    .budget
                    .output_tokens
                    .saturating_add(u64::from(u.output_tokens));
            }

            let text = Self::extract_text(&assistant_msg);
            let (turn_in, turn_out) = Self::usage_tokens(usage.as_ref());

            // Record the response text with the detection manager and check
            // for loop/convergence patterns.
            let pattern = self.managers.detection.record_response(&text);

            self.managers.observers().on_response(&ResponseContext {
                turn: self.budget.total_turns,
                text: text.clone(),
                usage,
            });

            if let Some(result) = self
                .managers
                .handle_detected_pattern(&pattern, self.budget.total_turns)
            {
                match result {
                    Ok(_) => {
                        self.state = LoopState::Completed {
                            summary: text.clone(),
                        };
                        return Ok(TurnResult {
                            text,
                            tool_calls: Vec::new(),
                            tool_results: Vec::new(),
                            input_tokens: turn_in,
                            output_tokens: turn_out,
                            duration: turn_start.elapsed(),
                            is_complete: true,
                            stop_reason: StopReason::EndTurn,
                        });
                    }
                    Err(e) => {
                        self.state = LoopState::Failed {
                            error: e.to_string(),
                        };
                        return Err(e);
                    }
                }
            }

            let tool_calls = Self::extract_tool_calls(&assistant_msg);
            self.conversation.push(assistant_msg);
            self.budget.total_turns = self.budget.total_turns.saturating_add(1);

            // No tool calls → this turn is complete.
            if tool_calls.is_empty() {
                self.managers.observers().on_turn_end(&TurnEndContext {
                    turn: self.budget.total_turns.saturating_sub(1),
                    success: true,
                    error: None,
                    duration_ms: Self::millis_u64(turn_start.elapsed()),
                    input_tokens: turn_in,
                    output_tokens: turn_out,
                });
                self.state = LoopState::Completed {
                    summary: text.clone(),
                };
                return Ok(TurnResult {
                    text,
                    tool_calls: Vec::new(),
                    tool_results: Vec::new(),
                    input_tokens: turn_in,
                    output_tokens: turn_out,
                    duration: turn_start.elapsed(),
                    is_complete: true,
                    stop_reason: StopReason::EndTurn,
                });
            }

            // Dispatch tool calls.
            self.state = LoopState::WaitingForTool {
                tool: tool_calls
                    .first()
                    .map(|tc| tc.tool.clone())
                    .unwrap_or_default(),
                started_at: std::time::SystemTime::now(),
            };

            // Temporarily extract budget to avoid double mutable borrow.
            let mut budget = std::mem::take(&mut self.budget);
            let turn_index = budget.total_turns.saturating_sub(1);
            let turn_duration = turn_start.elapsed();
            if let Err(e) = self
                .dispatch_and_record(
                    &tool_calls,
                    turn_index,
                    turn_duration,
                    turn_in,
                    turn_out,
                    &mut budget,
                )
                .await
            {
                self.budget = budget;
                self.state = LoopState::Failed {
                    error: e.to_string(),
                };
                return Err(e);
            }
            self.budget = budget;

            // Attempt context compaction.
            //
            // Compaction is best-effort: if it fails (e.g. the compactor
            // cannot reduce the conversation enough), we log a warning and
            // continue. The next API call may still succeed, and if it
            // doesn't, the provider's context-overflow error will surface
            // naturally at that point.
            if let Err(e) = self.maybe_compact_context(self.budget.total_turns).await {
                tracing::warn!(
                    error = %e,
                    turn = self.budget.total_turns,
                    "context compaction failed; continuing with uncompactd history"
                );
            }

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

            let success = !matches!(self.state, LoopState::Failed { .. });

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

    fn config(&self) -> LoopConfig {
        self.config.clone()
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

    // ==================================================
    // Mock ApiClient
    // ==================================================

    /// A mock API client that returns configurable responses.
    ///
    /// Used exclusively in tests. Stores a queue of response vectors,
    /// where each response is a `Vec<StreamEvent>`. Each call to
    /// [`stream_messages`](MockClient::stream_messages) pops the next
    /// response from the front of the queue.
    ///
    /// When the queue is empty, `stream_messages` returns a single
    /// [`ApiError`] — this lets tests verify error-handling paths.
    #[derive(Clone)]
    struct MockClient {
        /// Responses to return, in order.
        ///
        /// Each entry is a `Vec<StreamEvent>` representing one complete
        /// streaming response from the API. Popped from the front by
        /// [`stream_messages`](MockClient::stream_messages).
        responses: Arc<std::sync::Mutex<Vec<Vec<StreamEvent>>>>,

        /// Model name reported by [`ApiClient::model()`].
        ///
        /// Copied into mock response metadata so that assertions can
        /// verify the model field.
        model_name: String,
    }

    impl MockClient {
        /// Create a new mock client with the given model name.
        ///
        /// The response queue starts empty. Add responses with
        /// [`add_text_response()`](MockClient::add_text_response),
        /// [`add_tool_then_text()`](MockClient::add_tool_then_text), or
        /// [`add_tool_only_response()`](MockClient::add_tool_only_response).
        fn new(model: &str) -> Self {
            Self {
                responses: Arc::new(std::sync::Mutex::new(Vec::new())),
                model_name: model.to_string(),
            }
        }

        /// Add a simple text response that ends the turn.
        ///
        /// Generates a complete stream: `MessageStart` →
        /// `PartStart(text)` → `IndexedDelta(text)` →
        /// `PartStop` → `MessageDelta(end_turn, usage)` →
        /// `MessageStop`. The model will emit no tool calls, so the
        /// loop terminates after this response.
        ///
        /// # Parameters
        ///
        /// - `text` — The text the model will "say".
        fn add_text_response(&self, text: &str) {
            let events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_test".into(),
                        role: "assistant".into(),
                        model: self.model_name.clone(),
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
            self.responses.lock().unwrap().push(events);
        }

        /// Add a raw sequence of stream events as a single response turn.
        fn add_events(&self, events: Vec<StreamEvent>) {
            self.responses.lock().unwrap().push(events);
        }

        /// Add a tool_call response followed by an end_turn response.
        ///
        /// The first response contains a single `tool_call` content part
        /// (causing the loop to dispatch the tool), and the second
        /// response is a plain text `end_turn` (causing the loop to
        /// terminate). This simulates the common two-turn pattern:
        /// model requests tool → model sees result → model responds.
        ///
        /// # Parameters
        ///
        /// - `tool_id` — Unique ID for the tool call.
        /// - `tool_name` — Name of the tool to invoke.
        /// - `tool_input` — JSON input for the tool.
        /// - `final_text` — Text the model says after seeing the result.
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
                        model: self.model_name.clone(),
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
            self.responses.lock().unwrap().push(tool_events);

            // Second response: end_turn with text
            let text_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_final".into(),
                        role: "assistant".into(),
                        model: self.model_name.clone(),
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
            self.responses.lock().unwrap().push(text_events);
        }

        /// Add a tool_call-only response (no end_turn).
        ///
        /// The response contains a single `tool_call` content part with
        /// stop reason `tool_call`. After the tool is dispatched, the
        /// loop will request another turn from the API. Useful for
        /// testing max-turns and multi-turn tool chains.
        ///
        /// # Parameters
        ///
        /// - `tool_id` — Unique ID for the tool call.
        /// - `tool_name` — Name of the tool to invoke.
        /// - `tool_input` — JSON input for the tool.
        fn add_tool_only_response(&self, tool_id: &str, tool_name: &str, tool_input: Value) {
            let tool_events = vec![
                StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: format!("msg_{tool_id}"),
                        role: "assistant".into(),
                        model: self.model_name.clone(),
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
            self.responses.lock().unwrap().push(tool_events);
        }

        /// Add an error response. Reserved for future use.
        ///
        /// Currently pushes an incomplete response (only `MessageStart`)
        /// which does not trigger an error on its own. Reserved for
        /// testing streaming-error scenarios once the accumulator
        /// handles partial messages.
        #[expect(dead_code)]
        fn add_error_response(&self) {
            // Return an empty response that will cause the stream to error
            // We'll handle this by having the stream return an error event
            let events = vec![StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_err".into(),
                    role: "assistant".into(),
                    model: self.model_name.clone(),
                },
            })];
            self.responses.lock().unwrap().push(events);
        }
    }

    impl ApiClient for MockClient {
        /// Return the model name configured at construction.
        fn model(&self) -> &str {
            &self.model_name
        }

        /// Pop the next queued response and return it as a stream.
        ///
        /// If the response queue is empty, returns a single-element
        /// stream containing an [`ApiError`]. This enables tests that
        /// exhaust all responses to verify error handling.
        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            let mut guard = self.responses.lock().unwrap();
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

        /// Non-streaming message creation — not used in these tests.
        ///
        /// Returns an empty JSON object. The `BareLoop` tests
        /// exercise only the streaming path.
        fn create_message(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
            Box::pin(async { Ok(json!({"content": []})) })
        }
    }

    // Helper trait for Vec-like pop_front on Vec
    trait PopFront<T> {
        /// Remove and return the first element, shifting the rest left.
        ///
        /// Returns `None` if the vector is empty. O(n)
        /// operation because it calls `Vec::remove(0)`. Acceptable
        /// for test-only code with small queues.
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

    // ==================================================
    // Mock Tool
    // ==================================================

    /// A test tool that echoes back its input.
    ///
    /// Implements [`Tool`] with a single `message` string parameter.
    /// Returns `ToolOutput::text(format!("Echo: {msg}"))` so callers
    /// can verify round-trip data flow.
    struct EchoTool;

    impl Tool for EchoTool {
        /// Return the tool name `"echo"`.
        fn name(&self) -> &'static str {
            "echo"
        }

        /// Return a human-readable description.
        fn description(&self) -> &'static str {
            "Echoes back the input"
        }

        /// Return the JSON schema for this tool.
        ///
        /// Requires a single string property `message`.
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

        /// Execute the tool: extract `message` from input and echo it.
        ///
        /// If the `message` field is missing or not a string, defaults
        /// to an empty string.
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

    /// A test tool that always fails.
    ///
    /// Used to verify that tool-execution errors are handled gracefully:
    /// the loop should record the error as a soft tool result and
    /// continue, not abort the session.
    struct FailingTool;

    impl Tool for FailingTool {
        /// Return the tool name `"fail"`.
        fn name(&self) -> &'static str {
            "fail"
        }

        /// Return a human-readable description.
        fn description(&self) -> &'static str {
            "Always fails"
        }

        /// Return the JSON schema for this tool.
        ///
        /// Accepts an empty object (no parameters).
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "fail".into(),
                description: "Always fails".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            }
        }

        /// Execute the tool: always returns an execution error.
        ///
        /// Returns [`ToolError::Execution`] with a fixed message so
        /// tests can assert on the error path without triggering
        /// panics or unwinds.
        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async move { Err(ToolError::Execution("Tool intentionally failed".into())) })
        }
    }

    // ==================================================
    // Counting Plugin (test helper)
    // ==================================================

    /// A [`LoopObserver`](crate::observer::LoopObserver) that counts
    /// how many times each hook fires.
    ///
    /// Uses [`AtomicUsize`] counters with `SeqCst` ordering so that
    /// test assertions can read the counts from any thread after the
    /// agent loop completes.
    struct CountingObserver {
        /// Number of times `on_session_start` was called.
        session_starts: AtomicUsize,
        /// Number of times `on_session_end` was called.
        session_ends: AtomicUsize,
        /// Number of times `on_turn_start` was called.
        turn_starts: AtomicUsize,
        /// Number of times `on_turn_end` was called.
        turn_ends: AtomicUsize,
        /// Number of times `on_tool_pre` was called.
        tool_pres: AtomicUsize,
        /// Number of times `on_tool_post` was called.
        tool_posts: AtomicUsize,
    }

    impl CountingObserver {
        /// Create a new observer with all counters initialized to zero.
        fn new() -> Self {
            Self {
                session_starts: AtomicUsize::new(0),
                session_ends: AtomicUsize::new(0),
                turn_starts: AtomicUsize::new(0),
                turn_ends: AtomicUsize::new(0),
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

        fn on_tool_pre(&self, _ctx: &crate::observer::ToolPreContext) {
            self.tool_pres.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_post(&self, _ctx: &crate::observer::ToolPostContext) {
            self.tool_posts.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ==================================================
    // Test Helpers
    // ==================================================

    /// Create a default [`LoopConfig`] with `max_turns = 10`.
    ///
    /// Most tests use this as a baseline. Tests that need a different
    /// max-turns value mutate the returned config before constructing
    /// the loop.
    fn make_config() -> LoopConfig {
        LoopConfig {
            max_turns: 10,
            ..Default::default()
        }
    }

    // ==================================================
    // Tests: Basic lifecycle
    // ==================================================

    /// Verify that a single-turn session (text response, no tool calls)
    /// completes successfully and returns the model's text output.
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

    /// Verify that a two-turn session (tool call → tool result → end turn)
    /// completes successfully and records the tool invocation.
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

    /// Verify that exceeding `max_turns` returns
    /// [`LoopError::MaxTurnsExceeded`] and reports `success = false`.
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

    /// Verify that calling [`cancel()`](BareLoop::cancel) mid-session
    /// returns [`LoopError::Cancelled`].
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

    /// Verify that an API error during streaming propagates as
    /// [`LoopError::Api`] and marks the session as failed.
    #[tokio::test]
    async fn test_bare_loop_api_error() {
        // The mock will return an error
        let client = MockClient::new("test-model");
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LoopError::Api(msg) => assert!(msg.contains("No more mock responses")),
            other => panic!("Expected Api error, got: {other}"),
        }
    }

    // ==================================================
    // Tests: Tool dispatch
    // ==================================================

    /// Verify that requesting a tool not present in the registry produces
    /// a soft error result (not a hard [`LoopError`]), allowing the model
    /// to see the failure and adapt.
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

    /// Verify that a tool returning an execution error produces a soft
    /// error result and the session continues to completion.
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

    // ==================================================
    // Tests: Observers
    // ==================================================

    /// Verify that a single-turn session fires `session_start`,
    /// `turn_start`, `turn_end`, and `session_end` on the observer.
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

    /// Verify that a tool-using session fires `on_tool_pre` and
    /// `on_tool_post` observer hooks in addition to the turn hooks.
    ///
    /// A two-turn session (tool_call + end_turn) should produce:
    /// - 2 turn starts, 2 turn ends
    /// - 1 tool pre, 1 tool post
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

    // ==================================================
    // Tests: Conversation management
    // ==================================================

    /// Verify that `extract_tool_calls` returns an empty list for a
    /// text-only message and correctly parses tool_call parts when
    /// present.
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

    /// Verify that `build_tool_result_message` produces a user message
    /// with the correct `tool_result` content parts, including the
    /// tool_call_id, output text, and is_error flag.
    #[tokio::test]
    async fn test_tool_result_message_format() {
        let results = vec![super::ToolDispatchResult {
            tool_call_id: "tool_123".to_string(),
            output: ToolContent::Text("Echo: hello".to_string()),
            is_error: false,
            duration: Duration::from_millis(100),
            resolved_tool_name: String::new(),
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

    // ==================================================
    // Tests: Multiple tools in one turn
    // ==================================================

    /// Verify that multiple tool_call parts in a single assistant message
    /// are all dispatched and counted.
    ///
    /// The mock emits two `tool_call` parts in one response, followed by
    /// an `end_turn` response. The session should report 2 turns and 2
    /// tool calls.
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
        client.responses.lock().unwrap().push(tool_events);

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

    // ==================================================
    // Tests: text streamer callback
    // ==================================================

    /// Verify that `set_text_streamer` fires the callback for each text
    /// delta during streaming.
    #[tokio::test]
    async fn test_text_streamer_fires_on_text_delta() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello world");
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let buf = Arc::clone(&received);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            buf.lock().unwrap().push(delta.to_string());
        }));

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let received = received.lock().unwrap();
        assert!(!received.is_empty(), "streamer should have fired");
        assert!(
            received.join("").contains("Hello world"),
            "got: {received:?}",
        );
    }

    /// Verify that a run works fine without a text streamer set.
    #[tokio::test]
    async fn test_text_streamer_none_works() {
        let client = MockClient::new("test-model");
        client.add_text_response("No streamer");
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());

        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);
    }

    /// Verify the streamer only fires for text deltas, not for tool-call
    /// deltas or metadata events.
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

        let received = Arc::new(std::sync::Mutex::new(String::new()));
        let buf = Arc::clone(&received);
        agent.set_text_streamer(Arc::new(move |delta: &str| {
            buf.lock().unwrap().push_str(delta);
        }));

        agent.run("Use tool").await.unwrap();

        // The InputJson delta should NOT have triggered the streamer.
        // Only the "Done" text response in the second turn should.
        let received = received.lock().unwrap();
        assert_eq!(&*received, "Done", "only text deltas should fire streamer");
    }

    // ==================================================
    // Tests: Accessors
    // ==================================================

    /// Verify that accessor methods return the values passed at
    /// construction.
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

    /// Verify that `cancel_signal()` returns a shared reference to the
    /// same signal used by `cancel()` and `is_cancelled()`.
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

    // ==================================================
    // Tests: Session result fields
    // ==================================================

    /// Verify that the returned [`SessionResult`] has the correct
    /// session ID, positive duration, and non-zero token count.
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

    // ==================================================
    // Tests: Property — loop always terminates
    // ==================================================

    /// Verify that setting `max_turns = 1` still allows a single-turn
    /// session to complete normally.
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

    /// Verify that setting `max_turns = 0` immediately triggers a
    /// configuration error before any API call.
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

    // ==================================================
    // Tests: Error in tool-not-found returns error result, not hard error
    // ==================================================

    /// Verify that requesting a nonexistent tool produces a soft error
    /// result (not a hard [`LoopError`]), allowing the model to see
    /// the error and respond gracefully.
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
        client.responses.lock().unwrap().push(tool_events);

        // Second response: end_turn after seeing error result
        client.add_text_response("Tool wasn't found, but I'll handle it.");

        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Use missing tool").await.unwrap();
        assert!(result.success);
    }

    // ==================================================
    // ==================================================
    // Recovery wiring tests
    // ==================================================

    /// Verify the default `NoopReflector` + `ExponentialBackoffRecovery`
    /// wiring returns soft errors (no infinite loop, no panic).
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

    /// Verify that when a tool is not found, the recovery wiring still
    /// produces a soft error result (no hard error propagated).
    #[tokio::test]
    async fn test_recovery_on_missing_tool_returns_soft_result() {
        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "OK");

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
    }

    /// Verify that a failing tool with the default recovery produces
    /// exactly one tool dispatch (NoopReflector marks everything as
    /// non-recoverable, so no retries).
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

    /// Verify cancellation is still respected during tool recovery.
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

    // ==================================================
    // Tests: set_pipeline uses self.tools registry
    // ==================================================

    /// Verify that `set_pipeline` automatically injects `self.tools` as the
    /// pipeline's core registry, so dispatch never diverges from schema
    /// generation.
    #[tokio::test]
    async fn test_set_pipeline_injects_self_tools_registry() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hello"}), "done");
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let config = make_config();
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        // Build a builder WITHOUT calling .core() — set_pipeline must inject it.
        let builder = ToolPipeline::builder();
        agent.set_pipeline(builder).unwrap();

        let result = agent.run("Echo hello").await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }

    // ==================================================
    // Tests: turn_number is actual turn index
    // ==================================================

    /// A middleware that records the `turn_number` from each dispatch context.
    struct TurnNumberCapture {
        turns: Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl TurnNumberCapture {
        fn new(shared: Arc<std::sync::Mutex<Vec<usize>>>) -> Self {
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
            self.turns.lock().unwrap().push(ctx.turn_number);
            next.dispatch(ctx)
        }
    }

    /// Verify that `turn_number` reflects the actual turn index, not
    /// `config.max_turns`.
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

        let capture = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        let builder = ToolPipeline::builder().with(TurnNumberCapture::new(Arc::clone(&capture)));
        agent.set_pipeline(builder).unwrap();

        let result = agent.run("test").await;
        assert!(result.is_ok());

        let turns = capture.lock().unwrap().clone();
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
}
