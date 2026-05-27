//! `BareLoop` — the framework's default agent loop implementation.
//!
//! This module provides [`BareLoop`], a generic, framework-level agent
//! loop that orchestrates the full lifecycle of an LLM-based agent session:
//! sending messages to an LLM API, accumulating streaming responses,
//! dispatching tool calls, and feeding results back into the conversation
//! until the model ends its turn or a configured limit is reached.
//!
//! # Architecture
//!
//! [`BareLoop`] ties together four key components:
//!
//! - An [`ApiClient`](crate::api_client::ApiClient) for communicating with
//!   the LLM provider.
//! - A [`ToolRegistry`](crate::tool::ToolRegistry) for dispatching tool
//!   calls the model requests.
//! - An [`AgentConfig`] governing session parameters (max turns, system
//!   prompt, session ID).
//! - Optional [`AgentObserver`] implementations for lifecycle instrumentation.
//!
//! ```text
//! BareLoop
//!   ┌─────────────────────────────────────────────────────────┐
//!   │  run(user_input)                                        │
//!   │    1. Push user message to conversation                 │
//!   │    2. Loop:                                             │
//!   │       a. stream_messages(conversation) → StreamEvents   │
//!   │       b. accumulate into Message (assistant)            │
//!   │       c. Extract tool calls from Message                │
//!   │       d. Execute tools via ToolRegistry                 │
//!   │       e. Push tool_result messages to conversation      │
//!   │       f. If no tool calls → break                       │
//!   │    3. Return SessionResult                              │
//!   └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Design Decisions
//!
//! - **Static dispatch** — `BareLoop<C>` is generic over the
//!   [`ApiClient`](crate::api_client::ApiClient) type parameter `C`,
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
//! use loopctl::loop_::BareLoop;
//! use loopctl::tool::ToolRegistry;
//! use loopctl::core::AgentConfig;
//! use std::sync::Arc;
//!
//! // 1. Build components
//! let client = Arc::new(my_api_client);
//! let registry = ToolRegistry::new();
//! let config = AgentConfig::default();
//!
//! // 2. Create the loop
//! let agent = BareLoop::new(client, registry, config);
//!
//! // 3. Run
//! let result = agent.run("Hello, agent!").await?;
//! println!("Agent responded in {} turns", result.total_turns);
//! ```

use crate::api_client::ApiClient;
use crate::cancel::CancelSignal;
use crate::compact::{ContextManager, EnsureContextResult};
use crate::core::reflection::{
    ExponentialBackoffRecovery, NoopReflector, RecoveryAction, RecoveryStrategy, ReflectionContext,
    Reflector,
};
use crate::core::{AgentConfig, AgentError, AgentObserver, SessionResult, ToolDispatchResult};
use crate::engine::middleware::{ToolDispatchContext, ToolPipeline, ToolPipelineBuilder};
#[cfg(feature = "hooks")]
use crate::hooks::HookAction;
#[cfg(feature = "hooks")]
use crate::hooks::HookExecutor;
#[cfg(feature = "hooks")]
use crate::hooks::context::{
    CompactTrigger, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
    SessionEndContext, SessionEndReason, SessionStartContext,
};
use crate::loop_control::bundle::ManagerBundle;
use crate::message::{Message, MessagePart, Role, ToolContent, ToolContentPart};
use crate::observability::{EventSink, NullSink, ObserveEvent};
use crate::stream::handler::{StreamHandler, StreamHandlerError};
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
#[cfg(feature = "tool_health")]
use crate::tool::health::ToolHealthRegistry;
use crate::tool::{PermissionCheck, ToolContext, ToolRegistry, ToolSchema};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

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
/// - [`with_observers()`](BareLoop::with_observers) — adds lifecycle
///   observers for instrumentation.
/// - [`with_managers()`](BareLoop::with_managers) — full control,
///   including a [`ManagerBundle`].
/// - [`from_parts()`](BareLoop::from_parts) — re-assembles from the
///   output of `AgentBuilder::into_raw_parts()`.
///
/// # Lifecycle
///
/// ```text
/// new() / with_observers() / from_parts()
///   → run(user_input)
///       → stream_turn() → dispatch_tools() → stream_turn()
///       → … (repeat until end_turn or max_turns)
///   → SessionResult
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::loop_::BareLoop;
/// use loopctl::tool::ToolRegistry;
/// use loopctl::core::AgentConfig;
/// use std::sync::Arc;
///
/// let registry = ToolRegistry::new();
/// let config = AgentConfig::default();
///
/// let agent = BareLoop::new(
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

    /// Optional middleware pipeline wrapping tool dispatch.
    ///
    /// When `Some`, tool calls flow through the pipeline's middleware
    /// chain (timeouts, output limiting, etc.) before reaching the
    /// registry. When `None`, dispatches go directly to the registry.
    pipeline: Option<ToolPipeline>,

    /// Session parameters (max turns, model, system prompt).
    ///
    /// See [`AgentConfig`] for the full set of options.
    config: AgentConfig,

    /// Conversation history (system + user + assistant + tool results).
    ///
    /// Grows over the session lifetime. Each call to [`run()`](BareLoop::run)
    /// appends the user message, then alternates between assistant responses
    /// and tool-result messages until the model signals `end_turn`.
    conversation: Vec<Message>,

    /// Lifecycle observers for session/turn/tool events.
    ///
    /// All observers are notified synchronously; long-running work should
    /// be offloaded to a channel or thread pool inside the observer.
    observers: Vec<Arc<dyn AgentObserver>>,

    /// Manager bundle (fallback, loop detection, convergence).
    #[expect(dead_code)]
    managers: ManagerBundle,

    /// Structured event sink for observability.
    ///
    /// Emits [`ObserveEvent`] variants at each lifecycle point.
    event_sink: Arc<dyn EventSink>,

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

    /// Optional context manager for automatic compaction.
    ///
    /// When `Some`, the loop checks token usage after each turn and
    /// triggers compaction when usage exceeds the configured threshold.
    /// Compaction replaces the conversation messages and emits
    /// [`on_compaction()`](AgentObserver::on_compaction) to observers
    /// and [`ObserveEvent::ContextCompacted`] to the event sink.
    context_manager: Option<Arc<ContextManager>>,

    /// Optional stream handler for resilient streaming.
    ///
    /// When `Some`, replaces the inline [`stream_turn()`](BareLoop::stream_turn)
    /// logic with the handler's retry, timeout, and fallback capabilities.
    /// When `None`, streaming uses the basic inline logic with no retries.
    stream_handler: Option<StreamHandler>,

    /// Ordered hook executor for lifecycle interception.
    ///
    /// When `Some`, the executor runs registered hooks before and after
    /// tool dispatch, compaction, and session start/end. Hooks can
    /// short-circuit with [`HookAction::Block`].
    /// [`HookAction::Ask`] is automatically downgraded to `Block` by the
    /// executor in [`crate::hooks::Interactivity::Headless`] mode (the default).
    /// When `None`, no lifecycle interception occurs.
    ///
    /// *Requires `hooks` feature.*
    #[cfg(feature = "hooks")]
    hook_executor: Option<Arc<HookExecutor>>,

    /// Per-tool health tracker with circuit breakers.
    ///
    /// When `Some`, records success/failure and latency for every tool
    /// dispatch. Tools that exceed the failure threshold have their circuit
    /// breaker opened, blocking subsequent calls until recovery.
    ///
    /// *Requires `tool_health` feature.*
    #[cfg(feature = "tool_health")]
    health_registry: Option<Arc<ToolHealthRegistry>>,
}

// ==================================================
// Run-loop bookkeeping types
// ==================================================

/// Accumulated token counts, tool-call count, and turn count for a session.
///
/// Mutable state that flows through the [`run()`](BareLoop::run) loop.
/// Using a struct avoids scattering loose counters across the method.
#[derive(Default)]
struct SessionBudget {
    input_tokens: u64,
    output_tokens: u64,
    total_tool_calls: usize,
    turn_count: usize,
}

impl SessionBudget {
    /// Accumulate token usage from a single turn into the running totals.
    fn accumulate_usage(&mut self, usage: Option<&Usage>) {
        if let Some(u) = usage {
            self.input_tokens = self.input_tokens.saturating_add(u64::from(u.input_tokens));
            self.output_tokens = self
                .output_tokens
                .saturating_add(u64::from(u.output_tokens));
        }
    }
}

/// Token counts for a single turn, captured before tool dispatch.
///
/// Needed because [`SessionBudget::accumulate_usage`] mutates the running
/// totals, but the per-turn values must be reported separately in
/// [`emit_turn_complete`](BareLoop::emit_turn_complete).
#[derive(Clone, Copy)]
struct TurnTokens {
    input: u64,
    output: u64,
}

impl TurnTokens {
    fn from_usage(usage: Option<&Usage>) -> Self {
        match usage {
            Some(u) => Self {
                input: u64::from(u.input_tokens),
                output: u64::from(u.output_tokens),
            },
            None => Self {
                input: 0,
                output: 0,
            },
        }
    }
}

/// Per-turn context passed to helper methods during the [`run()`](BareLoop::run) loop.
///
/// Bundles the zero-based turn index, wall-clock duration, and token
/// counts so that extracted methods don't need long parameter lists.
struct TurnContext {
    idx: usize,
    duration: Duration,
    tokens: TurnTokens,
}

/// Reason the session was aborted before normal completion.
///
/// Used by [`abort_session`](BareLoop::abort_session) to select the
/// correct [`AgentError`] variant without string matching.
#[derive(Clone, Copy)]
enum AbortReason {
    /// User or external signal requested cancellation.
    Cancelled,
    /// The turn budget was exhausted.
    MaxTurnsExceeded,
}

/// Aggregated session metrics passed to [`notify_session_end`](BareLoop::notify_session_end).
///
/// Captures completion status, an [`EndReason`] discriminant, turn/token
/// counters, and wall-clock duration — everything a hook needs to log or
/// react to session termination without pulling data from other sources.
#[allow(dead_code)]
struct SessionEndInfo {
    /// Whether the session completed normally.
    success: bool,
    /// Structured reason for the session end.
    reason: EndReason,
    /// Total turns executed.
    total_turns: usize,
    /// Total tokens consumed (input + output).
    total_tokens: u64,
    /// Wall-clock session duration in seconds.
    duration_secs: u64,
}

/// Discriminant for how a session terminated.
///
/// Mapped to [`SessionEndReason`] inside the `#[cfg(feature = "hooks")]`
/// path so the enum itself remains feature-independent.
enum EndReason {
    Complete,
    Cancelled,
    Error,
    MaxTurns,
}

impl<C: ApiClient> BareLoop<C> {
    /// Maximum retry attempts for tool recovery before giving up.
    const MAX_RECOVERY_ATTEMPTS: u32 = 5;

    /// Create a new `BareLoop` with the given components.
    ///
    /// Initializes an empty conversation history, no observers, and a
    /// fresh [`ManagerBundle`]. The cancellation signal starts as non-cancelled.
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
    /// let agent = BareLoop::new(
    ///     Arc::new(my_client),
    ///     ToolRegistry::new(),
    ///     AgentConfig::default(),
    /// );
    /// ```
    pub fn new(client: Arc<C>, tools: ToolRegistry, config: AgentConfig) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            pipeline: None,
            config,
            conversation: Vec::new(),
            observers: Vec::new(),
            managers: ManagerBundle::new(),
            event_sink: Arc::new(NullSink),
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            context_manager: None,
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
        }
    }

    /// Create a new `BareLoop` with lifecycle observers.
    ///
    /// Identical to [`new()`](BareLoop::new) but accepts a `Vec` of
    /// [`AgentObserver`] implementations. Observers receive callbacks for
    /// session start/end, turn start/end, and tool call/complete events.
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client, wrapped in `Arc`.
    /// - `tools` — The [`ToolRegistry`] containing available tools.
    /// - `config` — Session parameters.
    /// - `observers` — Lifecycle observers for instrumentation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = BareLoop::with_observers(
    ///     Arc::new(my_client),
    ///     registry,
    ///     config,
    ///     vec![Arc::new(LoggingObserver)],
    /// );
    /// ```
    pub fn with_observers(
        client: Arc<C>,
        tools: ToolRegistry,
        config: AgentConfig,
        observers: Vec<Arc<dyn AgentObserver>>,
    ) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            pipeline: None,
            config,
            conversation: Vec::new(),
            observers,
            managers: ManagerBundle::new(),
            event_sink: Arc::new(NullSink),
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            context_manager: None,
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
        }
    }

    /// Create a new `BareLoop` with all components including managers.
    ///
    /// Use this constructor when you need to supply a pre-configured
    /// [`ManagerBundle`] — for example, to enable loop detection or
    /// circuit-breaker policies.
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client, wrapped in `Arc`.
    /// - `tools` — The [`ToolRegistry`] containing available tools.
    /// - `config` — Session parameters.
    /// - `observers` — Lifecycle observers.
    /// - `managers` — A pre-built [`ManagerBundle`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let managers = ManagerBundle::builder()
    ///     .with_loop_detection(10)
    ///     .build();
    ///
    /// let agent = BareLoop::with_managers(
    ///     Arc::new(my_client),
    ///     registry,
    ///     config,
    ///     observers,
    ///     managers,
    /// );
    /// ```
    pub fn with_managers(
        client: Arc<C>,
        tools: ToolRegistry,
        config: AgentConfig,
        observers: Vec<Arc<dyn AgentObserver>>,
        managers: ManagerBundle,
    ) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            pipeline: None,
            config,
            conversation: Vec::new(),
            observers,
            managers,
            event_sink: Arc::new(NullSink),
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            context_manager: None,
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
        }
    }

    /// Create from builder parts (produced by `AgentBuilder::into_raw_parts()`).
    ///
    /// This is the most flexible constructor. It accepts all components
    /// individually, making it suitable for re-assembly after a builder
    /// has been consumed via `into_raw_parts()`.
    ///
    /// # Parameters
    ///
    /// - `client` — The LLM API client, wrapped in `Arc`.
    /// - `tools` — The [`ToolRegistry`].
    /// - `managers` — A [`ManagerBundle`].
    /// - `observers` — Lifecycle observers.
    /// - `config` — Session parameters.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (client, tools, managers, observers, config) = builder.into_raw_parts();
    /// let agent = BareLoop::from_parts(client, tools, managers, observers, config);
    /// ```
    pub fn from_parts(
        client: Arc<C>,
        tools: ToolRegistry,
        managers: ManagerBundle,
        observers: Vec<Arc<dyn AgentObserver>>,
        config: AgentConfig,
    ) -> Self {
        Self {
            client,
            tools: Arc::new(tools),
            pipeline: None,
            config,
            conversation: Vec::new(),
            observers,
            managers,
            event_sink: Arc::new(NullSink),
            reflector: Arc::new(NoopReflector),
            recovery: Arc::new(ExponentialBackoffRecovery::new(3)),
            cancelled: Arc::new(CancelSignal::new()),
            context_manager: None,
            stream_handler: None,
            #[cfg(feature = "hooks")]
            hook_executor: None,
            #[cfg(feature = "tool_health")]
            health_registry: None,
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
    /// Returns a reference to the [`AgentConfig`] that governs session
    /// parameters such as max turns, system prompt, and session ID.
    /// The config is immutable for the lifetime of the loop.
    pub fn config(&self) -> &AgentConfig {
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

    /// Set the [`EventSink`] for structured observability events.
    ///
    /// Replaces the default [`NullSink`] with a caller-supplied
    /// implementation. Must be called before [`run()`](BareLoop::run).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_event_sink(Arc::new(MySink));
    /// ```
    pub fn set_event_sink(&mut self, sink: Arc<dyn EventSink>) {
        self.event_sink = sink;
    }

    /// Set the [`Reflector`] for tool-error analysis.
    ///
    /// Replaces the default [`NoopReflector`] with a caller-supplied
    /// implementation. Must be called before [`run()`](BareLoop::run).
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
    /// [`run()`](BareLoop::run).
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
    /// Must be called before [`run()`](BareLoop::run).
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
        self.context_manager = Some(manager);
    }

    /// Set the [`StreamHandler`] for resilient streaming with retries,
    /// timeouts, and fallback to non-streaming.
    ///
    /// When set, the loop delegates streaming to the handler instead of
    /// using the inline streaming logic. Must be called before
    /// [`run()`](BareLoop::run).
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
        self.stream_handler = Some(handler);
    }

    /// Set the [`HookExecutor`] for lifecycle interception.
    ///
    /// When set, the executor runs registered hooks before and after
    /// tool dispatch, compaction, and session start/end. Hooks can
    /// short-circuit with [`HookAction::Block`].
    /// [`HookAction::Ask`] is automatically downgraded to `Block` by the
    /// executor in [`crate::hooks::Interactivity::Headless`] mode (the default).
    /// Must be called before [`run()`](BareLoop::run).
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
        self.hook_executor = Some(executor);
    }

    /// Set the [`ToolHealthRegistry`] for per-tool health tracking.
    ///
    /// When set, records success/failure and latency for every tool
    /// dispatch. Tools that exceed the failure threshold have their
    /// circuit breaker opened, blocking subsequent calls until recovery.
    /// Must be called before [`run()`](BareLoop::run).
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
        self.health_registry = Some(registry);
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// Replaces the default (no pipeline) with a caller-supplied
    /// [`ToolPipeline`]. When set, tool calls flow through the
    /// pipeline's middleware chain before reaching the registry.
    /// Must be called before [`run()`](BareLoop::run).
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
    /// Returns [`AgentError::Config`] if the builder fails to produce a valid
    /// pipeline (e.g. internal invariant violated).
    pub fn set_pipeline(&mut self, builder: ToolPipelineBuilder) -> Result<(), AgentError> {
        let pipeline = builder
            .core(Arc::clone(&self.tools))
            .build()
            .map_err(|e| AgentError::Config(e.to_string()))?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    // ==================================================
    // Main run loop
    // ==================================================

    /// Run the agent loop with the given user input.
    ///
    /// This is the primary entry point. It:
    /// 1. Pushes the user message into the conversation
    /// 2. Loops: stream → accumulate → tool dispatch → feedback
    /// 3. Returns a [`SessionResult`] when done
    ///
    /// The loop terminates when one of these conditions is met:
    ///
    /// - **End turn** — the model emits `end_turn` with no tool calls.
    /// - **Max turns exceeded** — [`config.max_turns`](AgentConfig::max_turns)
    ///   is reached, producing [`AgentError::MaxTurnsExceeded`].
    /// - **Cancellation** — [`cancel()`](BareLoop::cancel) was called,
    ///   producing [`AgentError::Cancelled`]; the caller should handle this
    ///   variant to distinguish user-initiated cancellation from other errors.
    /// - **API error** — the streaming request fails, producing
    ///   [`AgentError::Api`].
    ///
    /// # Observers
    ///
    /// Observers are notified at the following points:
    ///
    /// ```text
    /// on_session_start(session_id)
    ///   for each turn:
    ///     on_turn_start(user_input)
    ///     [stream from API]
    ///     for each tool_call:
    ///       on_tool_call(name, input)
    ///       on_tool_complete(name, input, output, duration, success, error)
    ///     on_turn_end(success, error)
    /// on_session_end(success, error)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] if:
    /// - The API call fails (after any retries)
    /// - Max turns is exceeded
    /// - A tool execution fails critically
    /// - The loop is cancelled
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = agent.run("Summarize this article").await?;
    /// if result.success {
    ///     println!("Output: {:?}", result.final_output);
    ///     println!("Turns: {}", result.total_turns);
    ///     println!("Input tokens: {}", result.input_tokens);
    ///     println!("Output tokens: {}", result.output_tokens);
    /// }
    /// ```
    pub async fn run(mut self, user_input: &str) -> Result<SessionResult, AgentError> {
        let session_id = self.config.session_id;
        let max_turns = self.config.max_turns;
        let start = Instant::now();
        let mut budget = SessionBudget::default();

        self.notify_session_start();
        self.conversation.push(Message::user(user_input));

        // Main agent loop
        loop {
            if self.is_cancelled() {
                return self.abort_session(&budget, start.elapsed(), AbortReason::Cancelled);
            }
            if budget.turn_count >= max_turns {
                return self.abort_session(&budget, start.elapsed(), AbortReason::MaxTurnsExceeded);
            }

            self.emit_turn_start(budget.turn_count, user_input);
            self.notify_turn_start(user_input);
            let turn_start = Instant::now();

            match self.stream_turn().await {
                Ok((assistant_msg, usage, stop_reason)) => {
                    budget.accumulate_usage(usage.as_ref());
                    let text = Self::extract_text(&assistant_msg);
                    let tool_calls = Self::extract_tool_calls(&assistant_msg);
                    self.conversation.push(assistant_msg);
                    budget.turn_count = budget.turn_count.saturating_add(1);

                    let turn = TurnContext {
                        idx: budget.turn_count.saturating_sub(1),
                        duration: turn_start.elapsed(),
                        tokens: TurnTokens::from_usage(usage.as_ref()),
                    };

                    if tool_calls.is_empty() {
                        return Ok(self.finalise_session(
                            session_id,
                            text,
                            stop_reason,
                            &turn,
                            start.elapsed(),
                            &budget,
                        ));
                    }

                    if let Err(e) = self
                        .dispatch_and_record(&tool_calls, &turn, &mut budget)
                        .await
                    {
                        return self.abort_session_from_error(e, start.elapsed(), &budget);
                    }

                    // After tool dispatch, check if context compaction is needed.
                    if let Err(e) = self.maybe_compact_context(budget.turn_count).await {
                        return self.abort_turn_and_session(
                            &budget,
                            turn_start.elapsed(),
                            start.elapsed(),
                            &e.to_string(),
                            e,
                        );
                    }
                }
                Err(e) => {
                    let err_str = e.to_string();
                    return self.abort_turn_and_session(
                        &budget,
                        turn_start.elapsed(),
                        start.elapsed(),
                        &err_str,
                        e,
                    );
                }
            }
        }
    }

    // ==================================================
    // Run helpers
    // ==================================================

    /// Dispatch tool calls, push the result message, and record the count.
    ///
    /// Emits turn-complete on success, turn-failed on error.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] or a tool-dispatch error
    /// propagated from [`dispatch_tools`](BareLoop::dispatch_tools).
    async fn dispatch_and_record(
        &mut self,
        tool_calls: &[ToolCallInfo],
        turn: &TurnContext,
        budget: &mut SessionBudget,
    ) -> Result<(), AgentError> {
        match self.dispatch_tools(tool_calls, turn.idx).await {
            Ok(results) => {
                budget.total_tool_calls = budget.total_tool_calls.saturating_add(results.len());
                let tool_result_msg = Self::build_tool_result_message(results);
                self.conversation.push(tool_result_msg);
                self.emit_turn_complete(
                    turn.idx,
                    turn.duration,
                    turn.tokens.input,
                    turn.tokens.output,
                );
                self.notify_turn_end(true, None);
                Ok(())
            }
            Err(e) => {
                let err_str = e.to_string();
                self.emit_turn_failed(turn.idx, turn.duration, &err_str);
                self.notify_turn_end(false, Some(&err_str));
                Err(e)
            }
        }
    }

    /// Build the final [`SessionResult`] when the model ends its turn.
    ///
    /// Called when streaming completes with no tool calls. Emits
    /// turn-complete/failed and session-stop events, notifies observers,
    /// and returns the assembled result.
    fn finalise_session(
        &self,
        session_id: Uuid,
        text: String,
        stop_reason: StreamStopReason,
        turn: &TurnContext,
        session_duration: Duration,
        budget: &SessionBudget,
    ) -> SessionResult {
        let success = stop_reason == StreamStopReason::EndTurn;
        let error = if success {
            None
        } else {
            Some(format!("Stream stopped with reason: {stop_reason:?}"))
        };

        if success {
            self.emit_turn_complete(
                turn.idx,
                turn.duration,
                turn.tokens.input,
                turn.tokens.output,
            );
        } else {
            self.emit_turn_failed(
                turn.idx,
                turn.duration,
                error.as_deref().unwrap_or("unknown"),
            );
        }
        self.notify_turn_end(success, error.as_deref());

        self.emit_session_stop(
            budget.turn_count,
            session_duration,
            success,
            error.as_deref().unwrap_or("completed"),
        );
        let end_reason = if success {
            EndReason::Complete
        } else {
            EndReason::Error
        };
        self.notify_session_end(&SessionEndInfo {
            success,
            reason: end_reason,
            total_turns: budget.turn_count,
            total_tokens: budget.input_tokens.saturating_add(budget.output_tokens),
            duration_secs: session_duration.as_secs(),
        });

        SessionResult {
            session_id,
            total_turns: budget.turn_count,
            input_tokens: budget.input_tokens,
            output_tokens: budget.output_tokens,
            total_duration: session_duration,
            tool_calls: budget.total_tool_calls,
            success,
            final_output: Some(text),
            error,
        }
    }

    /// Abort the session with an error — emits turn-failed + session-stop.
    ///
    /// Used when the streaming call itself fails (API error, timeout, etc.).
    ///
    /// # Errors
    ///
    /// Always returns `Err(error)`, passing through the original [`AgentError`].
    fn abort_turn_and_session(
        &self,
        budget: &SessionBudget,
        turn_duration: Duration,
        session_duration: Duration,
        reason: &str,
        error: AgentError,
    ) -> Result<SessionResult, AgentError> {
        self.emit_turn_failed(budget.turn_count, turn_duration, reason);
        self.notify_turn_end(false, Some(reason));
        self.emit_session_stop(budget.turn_count, session_duration, false, reason);
        let end_reason = if matches!(error, AgentError::Cancelled) {
            EndReason::Cancelled
        } else {
            EndReason::Error
        };
        self.notify_session_end(&SessionEndInfo {
            success: false,
            reason: end_reason,
            total_turns: budget.turn_count,
            total_tokens: budget.input_tokens.saturating_add(budget.output_tokens),
            duration_secs: session_duration.as_secs(),
        });
        Err(error)
    }

    /// Abort the session after a tool-dispatch error.
    ///
    /// Handles both [`AgentError::Cancelled`] and other errors uniformly.
    /// Turn-level events were already emitted inside [`dispatch_and_record`].
    ///
    /// # Errors
    ///
    /// Always returns `Err(error)`, passing through the original [`AgentError`].
    fn abort_session_from_error(
        &self,
        error: AgentError,
        session_duration: Duration,
        budget: &SessionBudget,
    ) -> Result<SessionResult, AgentError> {
        let reason = error.to_string();
        self.emit_session_stop(budget.turn_count, session_duration, false, &reason);
        let end_reason = if matches!(error, AgentError::Cancelled) {
            EndReason::Cancelled
        } else {
            EndReason::Error
        };
        self.notify_session_end(&SessionEndInfo {
            success: false,
            reason: end_reason,
            total_turns: budget.turn_count,
            total_tokens: budget.input_tokens.saturating_add(budget.output_tokens),
            duration_secs: session_duration.as_secs(),
        });
        Err(error)
    }

    /// Abort the session with a known reason string (cancel / max-turns).
    ///
    /// Does not emit turn-level events since no turn was started.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] or [`AgentError::MaxTurnsExceeded`]
    /// depending on the `reason` string.
    fn abort_session(
        &self,
        budget: &SessionBudget,
        session_duration: Duration,
        reason: AbortReason,
    ) -> Result<SessionResult, AgentError> {
        let reason_str = match &reason {
            AbortReason::Cancelled => "Cancelled",
            AbortReason::MaxTurnsExceeded => "Max turns exceeded",
        };
        self.emit_session_stop(budget.turn_count, session_duration, false, reason_str);
        let end_reason = match &reason {
            AbortReason::Cancelled => EndReason::Cancelled,
            AbortReason::MaxTurnsExceeded => EndReason::MaxTurns,
        };
        self.notify_session_end(&SessionEndInfo {
            success: false,
            reason: end_reason,
            total_turns: budget.turn_count,
            total_tokens: budget.input_tokens.saturating_add(budget.output_tokens),
            duration_secs: session_duration.as_secs(),
        });
        match reason {
            AbortReason::Cancelled => Err(AgentError::Cancelled),
            AbortReason::MaxTurnsExceeded => Err(AgentError::MaxTurnsExceeded {
                max: self.config.max_turns,
            }),
        }
    }

    // ==================================================
    // Context compaction
    // ==================================================

    /// Check if context compaction is needed and perform it if so.
    ///
    /// When a [`ContextManager`] is configured, this method:
    /// 1. Calls [`ContextManager::ensure_context_fits()`] to check token usage.
    /// 2. If compaction occurred, replaces `self.conversation` with the compacted messages.
    /// 3. Notifies observers via [`on_compaction`](AgentObserver::on_compaction).
    /// 4. Emits [`ObserveEvent::ContextCompacted`] to the event sink.
    ///
    /// When no `ContextManager` is set, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Context`] if compaction was needed but failed
    /// (i.e. the conversation exceeds the context window and the compactor
    /// could not reduce it sufficiently).
    async fn maybe_compact_context(&mut self, turn: usize) -> Result<(), AgentError> {
        let Some(ref ctx_manager) = self.context_manager else {
            return Ok(());
        };

        let messages_before = self.conversation.len();

        // Pre-compact hook check
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let tokens_before =
                crate::compact::CompactionOutcome::estimate_tokens(&self.conversation);
            let ctx = PreCompactContext {
                trigger: CompactTrigger::Auto,
                custom_instructions: None,
                message_count: messages_before,
                tokens_before,
                context_window: self.config.context_window,
                session_id: self.config.session_id,
            };
            let hook_result = executor.check_pre_compact(&ctx);
            if hook_result.abort {
                // Hook aborted compaction — return Ok, conversation unchanged.
                return Ok(());
            }
            // Note: hook_result.new_instructions and hook_result.additional_context
            // are available for future use with a hook-aware compactor.
        }

        let compact_start = Instant::now();
        let result = ctx_manager
            .ensure_context_fits(std::mem::take(&mut self.conversation), turn)
            .await;
        #[cfg(feature = "hooks")]
        let compact_duration_ms = u64::try_from(compact_start.elapsed().as_millis()).unwrap_or(0);
        #[cfg(not(feature = "hooks"))]
        let _ = compact_start;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                self.conversation = outcome.messages;
                let messages_after = self.conversation.len();
                for obs in &self.observers {
                    obs.on_compaction(messages_before, messages_after);
                }
                self.event_sink.on_event(&ObserveEvent::ContextCompacted {
                    messages_before,
                    messages_after,
                    tokens_saved: outcome.tokens_saved,
                });

                // Post-compact hook notification
                #[cfg(feature = "hooks")]
                if let Some(ref executor) = self.hook_executor {
                    let messages_compacted = messages_before.saturating_sub(messages_after);
                    let ctx = PostCompactContext {
                        trigger: CompactTrigger::Auto,
                        messages_compacted,
                        tokens_saved: outcome.tokens_saved,
                        tokens_after: outcome.tokens_after,
                        duration_ms: compact_duration_ms,
                        session_id: self.config.session_id,
                    };
                    executor.notify_post_compact(&ctx);
                }

                Ok(())
            }
            Ok(EnsureContextResult::NoAction(messages)) => {
                self.conversation = messages;
                Ok(())
            }
            Err(overflow) => Err(AgentError::ContextExceeded {
                used: overflow.tokens_used,
                limit: overflow.context_window,
            }),
        }
    }

    // ==================================================
    // Streaming
    // ==================================================

    /// Stream one turn from the API, accumulating the response.
    ///
    /// When a [`StreamHandler`] is configured (via
    /// [`set_stream_handler()`](BareLoop::set_stream_handler)), delegates
    /// to the handler for resilient streaming with retry, timeout, and
    /// fallback capabilities. Otherwise, uses the basic inline logic
    /// with no retries.
    ///
    /// Sends the current conversation history to the LLM API via
    /// [`ApiClient::stream_messages`] and uses a [`StreamAccumulator`]
    /// to collect the events into a single [`Message`].
    ///
    /// Also captures the stop reason (e.g. `end_turn`, `tool_call`) and
    /// token [`Usage`] from the stream's final `MessageDelta` event.
    ///
    /// # Returns
    ///
    /// A tuple of `(Message, Option<Usage>, StreamStopReason)`:
    ///
    /// - **[`Message`]** — the fully accumulated assistant message, including
    ///   any text and `tool_call` content parts.
    /// - **Option<[`Usage`]>** — token counts for this turn, if reported.
    /// - **[`StreamStopReason`]** — why the model stopped generating.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Api`] if any stream event is an error.
    /// Returns [`AgentError::Cancelled`] if the cancellation signal fires mid-stream.
    async fn stream_turn(&self) -> Result<(Message, Option<Usage>, StreamStopReason), AgentError> {
        // Delegate to StreamHandler if configured.
        if let Some(ref handler) = self.stream_handler {
            return self.stream_turn_via_handler(handler).await;
        }

        // Inline streaming (no handler).
        let system = self.config.system_prompt.clone();
        let tool_schemas = self.build_tool_schemas();
        let mut stream =
            self.client
                .stream_messages(self.conversation.clone(), system, tool_schemas);
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;
        loop {
            let event_result = tokio::select! {
                event = stream.next() => event,
                () = self.cancelled.notified() => {
                    return Err(AgentError::Cancelled);
                }
            };

            match event_result {
                Some(Ok(event)) => {
                    if let StreamEvent::MessageDelta(delta) = &event {
                        if let Some(ref reason_str) = delta.delta.stop_reason {
                            stop_reason =
                                StreamStopReason::from_api_str(reason_str).unwrap_or(stop_reason);
                        }
                    }
                    drop(accumulator.process(&event));
                }
                Some(Err(api_error)) => {
                    return Err(AgentError::Api(api_error.to_string()));
                }
                None => break,
            }
        }

        let usage = accumulator.usage().copied();
        let message = accumulator.build();

        Ok((message, usage, stop_reason))
    }

    /// Stream one turn via the [`StreamHandler`].
    ///
    /// Delegates streaming to the handler, which manages retries,
    /// timeouts, and fallback to non-streaming. Maps the handler's
    /// result/error types back to the `(Message, Option<Usage>,
    /// StreamStopReason)` tuple expected by the run loop.
    ///
    /// # Errors
    ///
    /// Maps [`StreamHandlerError`] variants to the appropriate
    /// [`AgentError`] variants:
    /// - [`Cancelled`](StreamHandlerError::Cancelled) → [`AgentError::Cancelled`]
    /// - [`InitFailed`](StreamHandlerError::InitFailed) → [`AgentError::Api`]
    /// - [`StreamFailed`](StreamHandlerError::StreamFailed) → [`AgentError::Api`]
    /// - [`FallbackFailed`](StreamHandlerError::FallbackFailed) → [`AgentError::Api`]
    async fn stream_turn_via_handler(
        &self,
        handler: &StreamHandler,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), AgentError> {
        let system = self.config.system_prompt.clone();
        let tool_schemas = self.build_tool_schemas();
        let result = handler
            .stream_turn(
                &*self.client,
                self.conversation.clone(),
                system,
                tool_schemas,
                &self.cancelled,
            )
            .await
            .map_err(Self::map_handler_error)?;
        Ok((result.message, result.usage, result.stop_reason))
    }

    /// Map a [`StreamHandlerError`] to an [`AgentError`].
    ///
    /// Preserves cancellation semantics —
    /// [`StreamHandlerError::Cancelled`] maps to [`AgentError::Cancelled`].
    /// All other variants map to [`AgentError::Api`] with a descriptive
    /// message.
    fn map_handler_error(error: StreamHandlerError) -> AgentError {
        match error {
            StreamHandlerError::Cancelled => AgentError::Cancelled,
            StreamHandlerError::InitFailed(outcome) => {
                AgentError::Api(format!("stream init failed: {outcome}"))
            }
            StreamHandlerError::StreamFailed(outcome) => {
                AgentError::Api(format!("stream failed: {outcome}"))
            }
            StreamHandlerError::FallbackFailed {
                stream_outcome,
                fallback_error,
            } => AgentError::Api(format!(
                "stream ({stream_outcome}) and fallback failed: {fallback_error}"
            )),
        }
    }

    // ==================================================
    // Tool dispatch
    // ==================================================

    /// Execute tool calls and return results.
    ///
    /// Iterates over each [`ToolCallInfo`] extracted from the assistant
    /// message, looks up the corresponding tool in the [`ToolRegistry`],
    /// and invokes it. Each result is wrapped in a [`ToolDispatchResult`].
    ///
    /// Tool execution is **sequential** so that cancellation can be
    /// checked between invocations. A tool that is not found in the
    /// registry produces a soft error result (not a hard [`AgentError`]),
    /// allowing the model to recover.
    ///
    /// When a tool returns an error (execution failure or not-found),
    /// the framework consults the [`Reflector`] and [`RecoveryStrategy`]
    /// to decide whether to retry, skip, ask user, or fail. Retry
    /// attempts use the delay specified by the [`RecoveryAction`].
    ///
    /// Observers are notified before and after each tool invocation via
    /// [`on_tool_call`](AgentObserver::on_tool_call) and
    /// [`on_tool_complete`](AgentObserver::on_tool_complete).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancellation flag is set
    /// between tool invocations.
    async fn dispatch_tools(
        &self,
        tool_calls: &[ToolCallInfo],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for tc in tool_calls {
            if self.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            let result = self.dispatch_tool_with_recovery(tc, turn_idx).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// Dispatch a single tool call, using reflector + recovery on errors.
    ///
    /// If the tool call succeeds, returns the result immediately. If it
    /// fails, calls [`Reflector::analyze()`] and [`RecoveryStrategy::decide()`]
    /// to determine the next action:
    ///
    /// - [`Retry`](RecoveryAction::Retry) — re-dispatch the tool after the
    ///   specified delay, up to the recovery strategy's retry limit.
    /// - [`Skip`](RecoveryAction::Skip) — produce a soft error result and
    ///   continue to the next tool.
    /// - [`Fail`](RecoveryAction::Fail) — produce a soft error result (the
    ///   model sees the failure and can decide how to respond).
    /// - [`AskUser`](RecoveryAction::AskUser) — treated as `Skip` (interactive
    ///   recovery not yet supported in `BareLoop`).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancellation signal fires
    /// during tool execution or between retry attempts.
    async fn dispatch_tool_with_recovery(
        &self,
        tc: &ToolCallInfo,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, AgentError> {
        let tool_context = self.build_tool_context();
        let mut attempt: u32 = 0;

        loop {
            if self.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            if let Some(blocked) = self.check_pre_tool_use_hooks(tc, turn_idx) {
                return Ok(blocked);
            }

            self.notify_tool_call(&tc.name, &tc.input.to_string());
            self.emit_tool_start(&tc.name, &tc.input.to_string());
            let start = Instant::now();

            let tool_result = if let Some(ref pipeline) = self.pipeline {
                self.dispatch_via_pipeline(pipeline, tc, &tool_context, start, turn_idx)
                    .await?
            } else if let Some(tool) = self.tools.get(&tc.name) {
                let cancel = Arc::clone(&self.cancelled);
                let call_result = tokio::select! {
                    r = tool.call(tc.input.clone(), &tool_context) => r,
                    () = cancel.notified() => {
                        let dur = start.elapsed();
                        self.notify_tool_complete(
                            &tc.name,
                            &tc.input.to_string(),
                            "",
                            dur,
                            false,
                            Some("cancelled"),
                        );
                        self.emit_tool_complete(&tc.name, "", true, dur);
                        return Err(AgentError::Cancelled);
                    }
                };
                match call_result {
                    Ok(result) => {
                        let duration = start.elapsed();
                        let output_text = result.text_content();
                        let success = !result.is_error;
                        self.notify_tool_complete(
                            &tc.name,
                            &tc.input.to_string(),
                            &output_text,
                            duration,
                            success,
                            None,
                        );
                        self.emit_tool_complete(&tc.name, &output_text, !success, duration);
                        ToolDispatchResult {
                            tool_call_id: Some(tc.id.clone()),
                            output: result.payload,
                            is_error: result.is_error,
                            duration,
                            resolved_tool_name: tc.name.clone(),
                        }
                    }
                    Err(e) => {
                        let duration = start.elapsed();
                        let error_msg = e.to_string();
                        self.notify_tool_complete(
                            &tc.name,
                            &tc.input.to_string(),
                            &error_msg,
                            duration,
                            false,
                            Some(&error_msg),
                        );
                        self.emit_tool_complete(&tc.name, &error_msg, true, duration);
                        ToolDispatchResult {
                            tool_call_id: Some(tc.id.clone()),
                            output: ToolContent::Text(error_msg.clone()),
                            is_error: true,
                            duration,
                            resolved_tool_name: tc.name.clone(),
                        }
                    }
                }
            } else {
                let available: Vec<String> = self.tools.tool_names().clone();
                let available_refs: Vec<&str> = available.iter().map(String::as_str).collect();
                let error = AgentError::tool_not_found(&tc.name, &available_refs);
                let error_msg = error.to_string();
                self.notify_tool_complete(
                    &tc.name,
                    &tc.input.to_string(),
                    &error_msg,
                    Duration::ZERO,
                    false,
                    Some(&error_msg),
                );
                self.emit_tool_complete(&tc.name, &error_msg, true, Duration::ZERO);
                ToolDispatchResult {
                    tool_call_id: Some(tc.id.clone()),
                    output: ToolContent::Text(error_msg.clone()),
                    is_error: true,
                    duration: Duration::ZERO,
                    resolved_tool_name: tc.name.clone(),
                }
            };

            self.notify_post_tool_use_hooks(tc, &tool_result, turn_idx);
            self.record_tool_health(tc.name.as_str(), &tool_result);

            // If the tool succeeded, return immediately.
            if !tool_result.is_error {
                return Ok(tool_result);
            }

            // Tool failed — consult reflector + recovery strategy.
            let recovery_action = self.recover_tool_error(tc, &tool_result, attempt).await;

            match recovery_action {
                RecoveryAction::Retry { delay } => {
                    attempt = attempt.saturating_add(1);
                    if attempt >= Self::MAX_RECOVERY_ATTEMPTS {
                        return Ok(tool_result);
                    }
                    tokio::select! {
                        () = tokio::time::sleep(delay) => {},
                        () = self.cancelled.notified() => {
                            return Err(AgentError::Cancelled);
                        }
                    }
                    // Loop to retry the tool call.
                }
                RecoveryAction::Skip(_) | RecoveryAction::AskUser(_) | RecoveryAction::Fail(_) => {
                    // All non-retry actions: return the error result as a
                    // soft error so the model can see it and respond.
                    return Ok(tool_result);
                }
            }
        }
    }

    /// Check pre-tool-use hooks and return a blocked result if any hook
    /// blocks or asks.
    ///
    /// Returns `Some(ToolDispatchResult)` with an error result if a hook
    /// blocked the call, or `None` if the call should proceed.
    ///
    /// *Requires `hooks` feature; returns `None` otherwise.*
    fn check_pre_tool_use_hooks(
        &self,
        tc: &ToolCallInfo,
        turn_idx: usize,
    ) -> Option<ToolDispatchResult> {
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let ctx = PreToolUseContext {
                tool_name: tc.name.clone(),
                input: tc.input.clone(),
                session_id: self.config.session_id,
                turn_number: turn_idx,
            };
            match executor.check_pre_tool_use(&ctx) {
                HookAction::Allow => None,
                HookAction::Block { reason } => {
                    self.emit_tool_complete(&tc.name, &reason, true, Duration::ZERO);
                    Some(ToolDispatchResult {
                        tool_call_id: Some(tc.id.clone()),
                        output: ToolContent::Text(reason),
                        is_error: true,
                        duration: Duration::ZERO,
                        resolved_tool_name: tc.name.clone(),
                    })
                }
                HookAction::Ask { message } => {
                    // In Headless mode (the default) the executor already
                    // downgrades Ask → Block. If we reach this arm the
                    // executor is Interactive, but BareLoop has no UI to
                    // show a prompt, so we still treat it as Block.
                    self.emit_tool_complete(&tc.name, &message, true, Duration::ZERO);
                    Some(ToolDispatchResult {
                        tool_call_id: Some(tc.id.clone()),
                        output: ToolContent::Text(message),
                        is_error: true,
                        duration: Duration::ZERO,
                        resolved_tool_name: tc.name.clone(),
                    })
                }
            }
        } else {
            None
        }
        #[cfg(not(feature = "hooks"))]
        {
            let _ = (tc, turn_idx);
            None
        }
    }

    /// Notify post-tool-use hooks with the execution result.
    ///
    /// *Requires `hooks` feature; no-op otherwise.*
    fn notify_post_tool_use_hooks(
        &self,
        tc: &ToolCallInfo,
        tool_result: &ToolDispatchResult,
        turn_idx: usize,
    ) {
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let output_text = match &tool_result.output {
                ToolContent::Text(t) => t.clone(),
                ToolContent::Multipart(_) => String::new(),
            };
            let ctx = PostToolUseContext {
                tool_name: tc.name.clone(),
                input: tc.input.clone(),
                output: output_text,
                is_error: tool_result.is_error,
                duration_ms: tool_result
                    .duration
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
                session_id: self.config.session_id,
                turn_number: turn_idx,
            };
            executor.notify_post_tool_use(&ctx);
        }
        #[cfg(not(feature = "hooks"))]
        {
            let _ = (tc, tool_result, turn_idx);
        }
    }

    /// Record tool health (success or failure) in the health registry.
    ///
    /// *Requires `tool_health` feature; no-op otherwise.*
    fn record_tool_health(&self, tool_name: &str, tool_result: &ToolDispatchResult) {
        #[cfg(feature = "tool_health")]
        if let Some(ref health) = self.health_registry {
            if tool_result.is_error {
                health.record_failure(tool_name, tool_result.duration);
            } else {
                health.record_success(tool_name, tool_result.duration);
            }
        }
        #[cfg(not(feature = "tool_health"))]
        {
            let _ = (tool_name, tool_result);
        }
    }

    /// Dispatch a tool call through the middleware pipeline.
    ///
    /// Builds a [`ToolDispatchContext`] from the tool call info, delegates
    /// to the pipeline's middleware chain, and converts the
    /// [`ToolDispatchResult`] back to a [`ToolDispatchResult`] with proper
    /// event emission. Handles cancellation via `tokio::select!`.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancel signal fires
    /// during pipeline dispatch.
    async fn dispatch_via_pipeline(
        &self,
        pipeline: &ToolPipeline,
        tc: &ToolCallInfo,
        tool_context: &ToolContext,
        start: Instant,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, AgentError> {
        let ctx = ToolDispatchContext {
            tool_name: tc.name.clone(),
            input: tc.input.clone(),
            call_id: tc.id.clone(),
            turn_number: turn_idx,
            cancel: Arc::clone(&self.cancelled),
            permission: PermissionCheck::Allow,
            tool_context: tool_context.clone(),
        };
        let cancel = Arc::clone(&self.cancelled);
        let dispatch_result = tokio::select! {
            r = pipeline.invoke(ctx) => r,
            () = cancel.notified() => {
                let dur = start.elapsed();
                self.notify_tool_complete(
                    &tc.name,
                    &tc.input.to_string(),
                    "",
                    dur,
                    false,
                    Some("cancelled"),
                );
                self.emit_tool_complete(&tc.name, "", true, dur);
                return Err(AgentError::Cancelled);
            }
        };
        let output_text = match &dispatch_result.output {
            ToolContent::Text(t) => t.clone(),
            ToolContent::Multipart(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ToolContentPart::Text { text } => Some(text.as_str()),
                    ToolContentPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };
        self.notify_tool_complete(
            &tc.name,
            &tc.input.to_string(),
            &output_text,
            dispatch_result.duration,
            !dispatch_result.is_error,
            if dispatch_result.is_error {
                Some(&output_text)
            } else {
                None
            },
        );
        self.emit_tool_complete(
            &tc.name,
            &output_text,
            dispatch_result.is_error,
            dispatch_result.duration,
        );
        Ok(ToolDispatchResult {
            tool_call_id: Some(tc.id.clone()),
            output: dispatch_result.output,
            is_error: dispatch_result.is_error,
            duration: dispatch_result.duration,
            resolved_tool_name: dispatch_result.resolved_tool_name,
        })
    }

    /// Analyse a tool error and decide on a recovery action.
    ///
    /// Calls [`Reflector::analyze()`] and then [`RecoveryStrategy::decide()`].
    /// If the reflector itself fails, logs the error and returns
    /// [`RecoveryAction::Fail`] (conservative default).
    async fn recover_tool_error(
        &self,
        tc: &ToolCallInfo,
        result: &ToolDispatchResult,
        attempt: u32,
    ) -> RecoveryAction {
        let error_msg = match &result.output {
            ToolContent::Text(msg) => msg.clone(),
            ToolContent::Multipart(_) => result.output.to_string(),
        };
        let context = ReflectionContext {
            task: String::new(),
            attempt,
            max_attempts: Self::MAX_RECOVERY_ATTEMPTS,
        };

        let Ok(analysis) = self
            .reflector
            .analyze(&error_msg, &tc.name, &tc.input, &context)
            .await
        else {
            // Reflector failed — conservatively fail.
            return RecoveryAction::Fail(error_msg);
        };

        self.recovery
            .decide(&analysis, attempt, Self::MAX_RECOVERY_ATTEMPTS)
            .await
    }

    // ==================================================
    // Helpers
    // ==================================================

    /// Extract all text content from a message.
    ///
    /// Iterates over the message's [`MessagePart`]s and concatenates
    /// every text part into a single `String`. Non-text parts (e.g.
    /// `ToolCall`, `ToolResult`) are silently skipped.
    ///
    /// Used to produce the [`SessionResult::final_output`] string when
    /// the model ends its turn.
    fn extract_text(msg: &Message) -> String {
        msg.parts
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract tool call information from a message.
    ///
    /// Scans the message's [`MessagePart`]s for `ToolCall` variants and
    /// maps each one to a [`ToolCallInfo`] containing the call ID, tool
    /// name, and JSON input. Non-`ToolCall` parts are silently skipped.
    ///
    /// Returns an empty `Vec` when the message contains no tool calls
    /// (i.e. the model ended with plain text).
    fn extract_tool_calls(msg: &Message) -> Vec<ToolCallInfo> {
        msg.parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::ToolCall { id, name, input } => Some(ToolCallInfo {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Build the tool result message from executed tool results.
    ///
    /// Per API convention, tool results are sent as a **user** message
    /// containing `tool_result` content parts. Each part pairs the
    /// `tool_call_id` with the tool's output (wrapped in a
    /// [`ToolContent`](ToolContent)) and an `is_error`
    /// flag so the model can distinguish successes from failures.
    ///
    /// # Parameters
    ///
    /// - `results` — The [`ToolDispatchResult`]s produced by
    ///   [`dispatch_tools()`](BareLoop::dispatch_tools).
    ///
    /// # Returns
    ///
    /// A [`Message`] with [`Role::User`] and one `tool_result`
    /// [`MessagePart`] per result.
    fn build_tool_result_message(results: Vec<ToolDispatchResult>) -> Message {
        let parts: Vec<MessagePart> = results
            .into_iter()
            .map(|r| {
                MessagePart::tool_result(r.tool_call_id.unwrap_or_default(), r.output, r.is_error)
            })
            .collect();
        Message::new(Role::User, parts)
    }

    /// Build tool schemas for the API request.
    ///
    /// Collects all tool schemas from the [`ToolRegistry`] and returns
    /// them as `Some(Vec<ToolSchema>)`, or `None` if the registry is
    /// empty (i.e. the agent has no tools). The API uses these schemas
    /// to inform the model what tools are available and their expected
    /// input shapes.
    fn build_tool_schemas(&self) -> Option<Vec<ToolSchema>> {
        let schemas = self.tools.all_schemas();
        if schemas.is_empty() {
            None
        } else {
            Some(schemas)
        }
    }

    /// Build a tool context for tool invocations.
    ///
    /// Creates a [`ToolContext`] pre-populated with the current session
    /// ID. Tools can use the context to correlate their work with the
    /// enclosing session (e.g. for logging, tracing, or storage).
    fn build_tool_context(&self) -> ToolContext {
        ToolContext {
            session_id: self.config.session_id,
            ..ToolContext::default()
        }
    }

    // ==================================================
    // Observer notifications
    // ==================================================

    /// Notify all observers that the session has started.
    ///
    /// Called once at the beginning of [`run()`](BareLoop::run),
    /// before the first turn. Iterates over every registered
    /// [`AgentObserver`] and calls
    /// [`on_session_start()`](AgentObserver::on_session_start) with the
    /// session ID from [`AgentConfig`].
    fn notify_session_start(&self) {
        for obs in &self.observers {
            obs.on_session_start(self.config.session_id);
        }
        self.event_sink.on_event(&ObserveEvent::SessionStart {
            session_id: self.config.session_id,
        });

        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let ctx = SessionStartContext {
                session_id: self.config.session_id,
                model: self.config.model.clone(),
                working_directory: std::env::current_dir()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            };
            executor.notify_session_start(&ctx);
        }
    }

    /// Notify all observers that the session has ended.
    ///
    /// Called once when [`run()`](BareLoop::run) returns — whether
    /// successfully, due to an error, or because of cancellation.
    fn notify_session_end(&self, info: &SessionEndInfo) {
        let reason_str = match &info.reason {
            EndReason::Complete => None,
            EndReason::Cancelled => Some("cancelled"),
            EndReason::MaxTurns => Some("max turns exceeded"),
            EndReason::Error => Some("session ended with error"),
        };
        for obs in &self.observers {
            obs.on_session_end(info.success, reason_str);
        }

        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let reason = match &info.reason {
                EndReason::Complete => SessionEndReason::Complete,
                EndReason::Cancelled => SessionEndReason::Cancelled,
                EndReason::Error => SessionEndReason::Error,
                EndReason::MaxTurns => SessionEndReason::MaxTurns,
            };
            let ctx = SessionEndContext {
                session_id: self.config.session_id,
                reason,
                total_turns: info.total_turns,
                total_tokens: info.total_tokens,
                duration_secs: info.duration_secs,
            };
            executor.notify_session_end(&ctx);
        }
    }

    /// Notify all observers that a turn has started.
    ///
    /// Called at the top of every iteration of the main loop, before
    /// the API streaming request. The `query` parameter is the original
    /// user input (the same for every turn within a single
    /// [`run()`](BareLoop::run) call).
    fn notify_turn_start(&self, query: &str) {
        for obs in &self.observers {
            obs.on_turn_start(query);
        }
    }

    /// Notify all observers that a turn has ended.
    ///
    /// Called after each turn completes — whether it produced a tool
    /// call, ended with text, or encountered an error. The `success`
    /// flag is `true` for normal turns and `false` when the API stream
    /// returned an error.
    fn notify_turn_end(&self, success: bool, error: Option<&str>) {
        for obs in &self.observers {
            obs.on_turn_end(success, error);
        }
    }

    /// Notify all observers that a tool is about to be invoked.
    ///
    /// Called just before the tool's [`call()`](crate::tool::Tool::call)
    /// method is invoked. The `tool` parameter is the tool name and
    /// `input` is the JSON input serialized to a string.
    fn notify_tool_call(&self, tool: &str, input: &str) {
        for obs in &self.observers {
            obs.on_tool_call(tool, input);
        }
    }

    /// Notify all observers that a tool invocation has completed.
    ///
    /// Called after the tool's [`call()`](crate::tool::Tool::call)
    /// method returns — whether successfully or with an error.
    /// Includes the tool's output, execution `duration`, a `success`
    /// flag, and an optional `error` message.
    ///
    /// Parameter count is dictated by the [`AgentObserver::on_tool_complete`]
    /// trait method.
    fn notify_tool_complete(
        &self,
        tool: &str,
        input: &str,
        output: &str,
        duration: Duration,
        success: bool,
        error: Option<&str>,
    ) {
        for obs in &self.observers {
            obs.on_tool_complete(tool, input, output, duration, success, error);
        }
    }

    // ==================================================
    // EventSink emissions
    // ==================================================

    /// Convert a [`Duration`] to milliseconds as `u64`.
    ///
    /// Clamps at `u64::MAX` if the duration exceeds ~584 million years,
    /// which is safe for any practical agent session.
    fn millis_u64(duration: Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }

    /// Emit a [`TurnStart`](ObserveEvent::TurnStart) event.
    fn emit_turn_start(&self, turn: usize, query: &str) {
        self.event_sink.on_event(&ObserveEvent::TurnStart {
            turn,
            query: query.to_string(),
        });
    }

    /// Emit a [`TurnComplete`](ObserveEvent::TurnComplete) event.
    fn emit_turn_complete(
        &self,
        turn: usize,
        duration: Duration,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        self.event_sink.on_event(&ObserveEvent::TurnComplete {
            turn,
            duration_ms: Self::millis_u64(duration),
            input_tokens,
            output_tokens,
        });
    }

    /// Emit a [`TurnFailed`](ObserveEvent::TurnFailed) event.
    fn emit_turn_failed(&self, turn: usize, duration: Duration, error: &str) {
        self.event_sink.on_event(&ObserveEvent::TurnFailed {
            turn,
            duration_ms: Self::millis_u64(duration),
            error: error.to_string(),
        });
    }

    /// Emit a [`ToolStart`](ObserveEvent::ToolStart) event.
    fn emit_tool_start(&self, name: &str, input: &str) {
        self.event_sink.on_event(&ObserveEvent::ToolStart {
            name: name.to_string(),
            input: input.to_string(),
        });
    }

    /// Emit a [`ToolComplete`](ObserveEvent::ToolComplete) event.
    fn emit_tool_complete(&self, name: &str, output: &str, is_error: bool, duration: Duration) {
        self.event_sink.on_event(&ObserveEvent::ToolComplete {
            name: name.to_string(),
            output: output.to_string(),
            is_error,
            duration_ms: Self::millis_u64(duration),
        });
    }

    /// Emit a [`SessionStop`](ObserveEvent::SessionStop) event.
    fn emit_session_stop(
        &self,
        total_turns: usize,
        duration: Duration,
        success: bool,
        reason: &str,
    ) {
        self.event_sink.on_event(&ObserveEvent::SessionStop {
            session_id: self.config.session_id,
            success,
            reason: reason.to_string(),
            total_turns,
            duration_ms: Self::millis_u64(duration),
        });
    }
}

// ==================================================
// ToolCallInfo
// ==================================================

/// Internal representation of a tool call extracted from a message.
///
/// When the LLM emits a `tool_call` content part, the loop extracts
/// its fields into this struct for convenient passing to
/// [`dispatch_tools()`](BareLoop::dispatch_tools).
///
/// This type is private to the module because external consumers
/// interact with tool results via [`SessionResult`] or the
/// [`AgentObserver`] callbacks.
///
/// # Fields
///
/// - [`id`](ToolCallInfo::id) — The unique identifier assigned by the
///   API to this tool call. Used to correlate the result back to the
///   request.
/// - [`name`](ToolCallInfo::name) — The tool name. Must match a tool
///   registered in the [`ToolRegistry`].
/// - [`input`](ToolCallInfo::input) — The JSON input provided by the
///   model. Deserialized into a `serde_json::Value`.
#[derive(Debug, Clone)]
struct ToolCallInfo {
    /// The tool call ID assigned by the API.
    ///
    /// Used to correlate the tool result message back to the original
    /// tool call. Copied into [`ToolDispatchResult::tool_call_id`] after
    /// execution.
    id: String,

    /// The tool name requested by the model.
    ///
    /// Must exactly match a name returned by a registered
    /// [`Tool::name()`](crate::tool::Tool::name). If no match is found,
    /// a soft error result is produced instead of a hard error.
    name: String,

    /// The tool input as a JSON value.
    ///
    /// Deserialized from the API's `tool_call` content part. Passed
    /// directly to [`Tool::call()`](crate::tool::Tool::call).
    input: serde_json::Value,
}

// ==================================================
// Tests
// ==================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_error::ApiError;
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
        /// Returns `None` if the vector is empty. This is an O(n)
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
        fn name(&self) -> &str {
            "echo"
        }

        /// Return a human-readable description.
        fn description(&self) -> &str {
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
        fn name(&self) -> &str {
            "fail"
        }

        /// Return a human-readable description.
        fn description(&self) -> &str {
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
    // Counting Observer
    // ==================================================

    /// An [`AgentObserver`] that counts how many times each callback fires.
    ///
    /// Uses [`AtomicUsize`] counters with `SeqCst` ordering so that
    /// test assertions can read the counts from any thread after the
    /// agent loop completes.
    ///
    /// # Counters
    ///
    /// - [`session_starts`](CountingObserver::session_starts) — incremented
    ///   by [`on_session_start`](AgentObserver::on_session_start).
    /// - [`session_ends`](CountingObserver::session_ends) — incremented
    ///   by [`on_session_end`](AgentObserver::on_session_end).
    /// - [`turn_starts`](CountingObserver::turn_starts) — incremented
    ///   by [`on_turn_start`](AgentObserver::on_turn_start).
    /// - [`turn_ends`](CountingObserver::turn_ends) — incremented
    ///   by [`on_turn_end`](AgentObserver::on_turn_end).
    /// - [`tool_calls`](CountingObserver::tool_calls) — incremented
    ///   by [`on_tool_call`](AgentObserver::on_tool_call).
    /// - [`tool_completes`](CountingObserver::tool_completes) — incremented
    ///   by [`on_tool_complete`](AgentObserver::on_tool_complete).
    struct CountingObserver {
        /// Number of times `on_session_start` was called.
        session_starts: AtomicUsize,
        /// Number of times `on_session_end` was called.
        session_ends: AtomicUsize,
        /// Number of times `on_turn_start` was called.
        turn_starts: AtomicUsize,
        /// Number of times `on_turn_end` was called.
        turn_ends: AtomicUsize,
        /// Number of times `on_tool_call` was called.
        tool_calls: AtomicUsize,
        /// Number of times `on_tool_complete` was called.
        tool_completes: AtomicUsize,
    }

    impl CountingObserver {
        /// Create a new observer with all counters initialized to zero.
        fn new() -> Self {
            Self {
                session_starts: AtomicUsize::new(0),
                session_ends: AtomicUsize::new(0),
                turn_starts: AtomicUsize::new(0),
                turn_ends: AtomicUsize::new(0),
                tool_calls: AtomicUsize::new(0),
                tool_completes: AtomicUsize::new(0),
            }
        }
    }

    impl AgentObserver for CountingObserver {
        /// Increment the session-start counter.
        fn on_session_start(&self, _session_id: uuid::Uuid) {
            self.session_starts.fetch_add(1, Ordering::SeqCst);
        }

        /// Increment the session-end counter.
        fn on_session_end(&self, _success: bool, _error: Option<&str>) {
            self.session_ends.fetch_add(1, Ordering::SeqCst);
        }

        /// Increment the turn-start counter.
        fn on_turn_start(&self, _query: &str) {
            self.turn_starts.fetch_add(1, Ordering::SeqCst);
        }

        /// Increment the turn-end counter.
        fn on_turn_end(&self, _success: bool, _error: Option<&str>) {
            self.turn_ends.fetch_add(1, Ordering::SeqCst);
        }

        /// Increment the tool-call counter.
        fn on_tool_call(&self, _tool: &str, _input: &str) {
            self.tool_calls.fetch_add(1, Ordering::SeqCst);
        }

        /// Increment the tool-complete counter.
        fn on_tool_complete(
            &self,
            _tool: &str,
            _input: &str,
            _output: &str,
            _duration: Duration,
            _success: bool,
            _error: Option<&str>,
        ) {
            self.tool_completes.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ==================================================
    // Test Helpers
    // ==================================================

    /// Create a default [`AgentConfig`] with `max_turns = 10`.
    ///
    /// Most tests use this as a baseline. Tests that need a different
    /// max-turns value mutate the returned config before constructing
    /// the loop.
    fn make_config() -> AgentConfig {
        AgentConfig {
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
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
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
        let agent = BareLoop::new(Arc::new(client), registry, config);
        let result = agent.run("Echo hello").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 2); // tool_call turn + end_turn
        assert_eq!(result.tool_calls, 1);
    }

    /// Verify that exceeding `max_turns` returns
    /// [`AgentError::MaxTurnsExceeded`] and reports `success = false`.
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

        let agent = BareLoop::new(Arc::new(client), registry, config);
        let result = agent.run("Keep going").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::MaxTurnsExceeded { max } => assert_eq!(max, 3),
            other => panic!("Expected MaxTurnsExceeded, got: {other}"),
        }
    }

    /// Verify that calling [`cancel()`](BareLoop::cancel) mid-session
    /// returns [`AgentError::Cancelled`].
    #[tokio::test]
    async fn test_bare_loop_cancellation() {
        let client = MockClient::new("test-model");
        client.add_text_response("Hello!");

        let config = make_config();
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

        // Cancel before running
        agent.cancel();
        assert!(agent.is_cancelled());

        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Cancelled => {}
            other => panic!("Expected Cancelled error, got: {other}"),
        }
    }

    /// Verify that an API error during streaming propagates as
    /// [`AgentError::Api`] and marks the session as failed.
    #[tokio::test]
    async fn test_bare_loop_api_error() {
        // The mock will return an error
        let client = MockClient::new("test-model");
        let config = make_config();
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Api(msg) => assert!(msg.contains("No more mock responses")),
            other => panic!("Expected Api error, got: {other}"),
        }
    }

    // ==================================================
    // Tests: Tool dispatch
    // ==================================================

    /// Verify that requesting a tool not present in the registry produces
    /// a soft error result (not a hard [`AgentError`]), allowing the model
    /// to see the failure and adapt.
    #[tokio::test]
    async fn test_tool_not_found_returns_error_result() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "I see the tool failed.");

        // Empty registry — tool won't be found
        let config = make_config();
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
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
        let agent = BareLoop::new(Arc::new(client), registry, config);
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

        let observer = Arc::new(CountingObserver::new());
        let config = make_config();
        let agent = BareLoop::with_observers(
            Arc::new(client),
            ToolRegistry::new(),
            config,
            vec![observer.clone()],
        );

        let result = agent.run("Hi").await.unwrap();

        assert!(result.success);
        assert_eq!(observer.session_starts.load(Ordering::SeqCst), 1);
        assert_eq!(observer.session_ends.load(Ordering::SeqCst), 1);
        assert_eq!(observer.turn_starts.load(Ordering::SeqCst), 1);
        assert_eq!(observer.turn_ends.load(Ordering::SeqCst), 1);
    }

    /// Verify that a tool-using session fires `tool_call` and
    /// `tool_complete` callbacks in addition to the turn callbacks.
    ///
    /// A two-turn session (tool_call + end_turn) should produce:
    /// - 2 turn starts, 2 turn ends
    /// - 1 tool call, 1 tool complete
    #[tokio::test]
    async fn test_observer_tool_events() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "test"}), "All done!");

        let observer = Arc::new(CountingObserver::new());
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let config = make_config();
        let agent =
            BareLoop::with_observers(Arc::new(client), registry, config, vec![observer.clone()]);

        let result = agent.run("Echo test").await.unwrap();
        assert!(result.success);

        assert_eq!(observer.tool_calls.load(Ordering::SeqCst), 1);
        assert_eq!(observer.tool_completes.load(Ordering::SeqCst), 1);
        assert_eq!(observer.turn_starts.load(Ordering::SeqCst), 2);
        assert_eq!(observer.turn_ends.load(Ordering::SeqCst), 2);
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
        assert_eq!(tool_calls[0].name, "echo");
    }

    /// Verify that `build_tool_result_message` produces a user message
    /// with the correct `tool_result` content parts, including the
    /// tool_call_id, output text, and is_error flag.
    #[tokio::test]
    async fn test_tool_result_message_format() {
        let results = vec![super::ToolDispatchResult {
            tool_call_id: Some("tool_123".to_string()),
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
        let agent = BareLoop::new(Arc::new(client), registry, config);

        let result = agent.run("Echo twice").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 2);
        assert_eq!(result.tool_calls, 2);
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
    // Tests: from_parts constructor
    // ==================================================

    /// Verify that `from_parts` produces a loop with an empty
    /// conversation.
    #[test]
    fn test_from_parts() {
        let client = MockClient::new("test-model");
        let config = make_config();
        let managers = ManagerBundle::new();
        let observers: Vec<Arc<dyn AgentObserver>> = vec![];
        let agent = BareLoop::from_parts(
            Arc::new(client),
            ToolRegistry::new(),
            managers,
            observers,
            config,
        );

        assert!(agent.conversation().is_empty());
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
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
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

        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await.unwrap();

        assert!(result.success);
        assert_eq!(result.total_turns, 1);
    }

    /// Verify that setting `max_turns = 0` immediately triggers
    /// [`AgentError::MaxTurnsExceeded`] before any API call.
    #[tokio::test]
    async fn test_loop_terminates_with_max_turns_0() {
        let client = MockClient::new("test-model");
        client.add_text_response("Should not be reached.");

        let mut config = make_config();
        config.max_turns = 0;

        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Hi").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::MaxTurnsExceeded { max } => assert_eq!(max, 0),
            other => panic!("Expected MaxTurnsExceeded, got: {other}"),
        }
    }

    // ==================================================
    // Tests: Error in tool-not-found returns error result, not hard error
    // ==================================================

    /// Verify that requesting a nonexistent tool produces a soft error
    /// result (not a hard [`AgentError`]), allowing the model to see
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
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);
        let result = agent.run("Use missing tool").await.unwrap();
        assert!(result.success);
    }

    // ==================================================
    // EventSink + Recovery wiring tests
    // ==================================================

    /// A recording [`EventSink`] that captures all emitted events.
    ///
    /// Uses `Mutex<Vec<ObserveEvent>>` so it's `Send + Sync`.
    struct RecordingSink {
        events: std::sync::Mutex<Vec<ObserveEvent>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                events: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Return a snapshot of all captured events.
        fn events(&self) -> Vec<ObserveEvent> {
            self.events.lock().expect("lock").clone()
        }

        fn count_matching(&self, pred: impl Fn(&ObserveEvent) -> bool) -> usize {
            self.events
                .lock()
                .expect("lock")
                .iter()
                .filter(|e| pred(e))
                .count()
        }

        /// Return true if any event matches the predicate.
        fn any(&self, pred: impl Fn(&ObserveEvent) -> bool) -> bool {
            self.events.lock().expect("lock").iter().any(|e| pred(e))
        }

        /// Return the first event matching the predicate, if any.
        fn find(&self, pred: impl Fn(&ObserveEvent) -> bool) -> Option<ObserveEvent> {
            self.events
                .lock()
                .expect("lock")
                .iter()
                .find(|e| pred(e))
                .cloned()
        }
    }

    impl EventSink for RecordingSink {
        fn on_event(&self, event: &ObserveEvent) {
            self.events.lock().expect("lock").push(event.clone());
        }
    }

    /// Helpers for matching [`ObserveEvent`] variants in assertions.
    mod event_match {
        use crate::observability::ObserveEvent;

        pub fn is_session_start(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::SessionStart { .. })
        }
        pub fn is_session_stop(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::SessionStop { .. })
        }
        pub fn is_turn_start(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::TurnStart { .. })
        }
        pub fn is_turn_complete(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::TurnComplete { .. })
        }
        pub fn is_tool_start(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::ToolStart { .. })
        }
        pub fn is_tool_complete(e: &ObserveEvent) -> bool {
            matches!(e, ObserveEvent::ToolComplete { .. })
        }
    }

    /// Build a [`BareLoop`] with a [`RecordingSink`] wired in.
    ///
    /// Returns the loop and an `Arc<RecordingSink>` for asserting events.
    fn agent_with_recording_sink(
        client: MockClient,
        registry: ToolRegistry,
        config: AgentConfig,
    ) -> (BareLoop<MockClient>, Arc<RecordingSink>) {
        let sink = Arc::new(RecordingSink::new());
        let mut agent = BareLoop::new(Arc::new(client), registry, config);
        agent.event_sink = Arc::clone(&sink) as Arc<dyn EventSink>;
        (agent, sink)
    }

    // ==================================================
    // EventSink emission tests
    // ==================================================

    /// Verify that a successful single-turn session emits
    /// [`SessionStart`](ObserveEvent::SessionStart) and
    /// [`SessionStop`](ObserveEvent::SessionStop) events.
    #[tokio::test]
    async fn test_sink_emits_session_start_stop_on_success() {
        let client = MockClient::new("test");
        client.add_text_response("Hello!");

        let (agent, sink) = agent_with_recording_sink(client, ToolRegistry::new(), make_config());
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);
        assert!(
            sink.any(event_match::is_session_start),
            "missing session_start"
        );
        assert!(
            sink.any(event_match::is_session_stop),
            "missing session_stop"
        );
        assert!(sink.any(event_match::is_turn_start), "missing turn_start");
        assert!(
            sink.any(event_match::is_turn_complete),
            "missing turn_complete"
        );
    }

    /// Verify that exceeding `max_turns` emits a
    /// [`SessionStop`](ObserveEvent::SessionStop) with `success = false`.
    #[tokio::test]
    async fn test_sink_emits_session_stop_on_max_turns_error() {
        let client = MockClient::new("test");
        // Queue a response so the mock has something to return, but
        // max_turns=0 means the loop aborts before streaming.
        client.add_text_response("turn 1");

        let mut config = make_config();
        config.max_turns = 0;

        let (agent, sink) = agent_with_recording_sink(client, ToolRegistry::new(), config);

        // max_turns=0 → immediate abort, no turn executes.
        let err = agent.run("Hi").await.unwrap_err();
        assert!(
            matches!(err, AgentError::MaxTurnsExceeded { .. }),
            "expected MaxTurnsExceeded, got {err:?}"
        );
        assert!(
            sink.any(event_match::is_session_start),
            "missing session_start"
        );
        let stop_evt = sink
            .find(event_match::is_session_stop)
            .expect("missing session_stop");
        assert!(
            matches!(stop_evt, ObserveEvent::SessionStop { success: false, .. }),
            "expected SessionStop with success=false, got {stop_evt:?}"
        );
    }

    /// Verify that a tool-using session emits
    /// [`ToolStart`](ObserveEvent::ToolStart) and
    /// [`ToolComplete`](ObserveEvent::ToolComplete) events.
    #[tokio::test]
    async fn test_sink_emits_tool_events_on_tool_call() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "hi"}), "Done");

        let (agent, sink) = agent_with_recording_sink(client, registry, make_config());
        let result = agent.run("Test").await.unwrap();
        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
        assert!(sink.any(event_match::is_tool_start), "missing tool_start");
        assert!(
            sink.any(event_match::is_tool_complete),
            "missing tool_complete"
        );
    }

    /// Verify that dispatching a missing tool emits
    /// [`ToolComplete`](ObserveEvent::ToolComplete) with `is_error = true`.
    #[tokio::test]
    async fn test_sink_emits_tool_complete_with_error_on_missing_tool() {
        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "missing_tool", json!({}), "OK");

        let (agent, sink) = agent_with_recording_sink(client, ToolRegistry::new(), make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert!(sink.any(event_match::is_tool_start), "missing tool_start");
        assert!(
            sink.any(event_match::is_tool_complete),
            "missing tool_complete"
        );

        // The tool_complete should have is_error=true
        let error_completes = sink.events().iter().any(|e| {
            if let ObserveEvent::ToolComplete { is_error, .. } = e {
                *is_error
            } else {
                false
            }
        });
        assert!(
            error_completes,
            "expected at least one tool_complete with is_error=true"
        );
    }

    /// Verify that a single-turn session emits the full event sequence:
    /// `SessionStart → TurnStart → TurnComplete → SessionStop`.
    #[tokio::test]
    async fn test_event_sequence_single_turn() {
        let client = MockClient::new("test");
        client.add_text_response("Hello!");

        let (agent, sink) = agent_with_recording_sink(client, ToolRegistry::new(), make_config());
        let result = agent.run("Hi").await.unwrap();
        assert!(result.success);

        let events = sink.events();
        // Sequence: SessionStart, TurnStart, TurnComplete, SessionStop
        assert!(
            event_match::is_session_start(&events[0]),
            "expected SessionStart, got {:?}",
            events[0]
        );
        assert!(
            event_match::is_turn_start(&events[1]),
            "expected TurnStart, got {:?}",
            events[1]
        );
        assert!(
            event_match::is_turn_complete(&events[2]),
            "expected TurnComplete, got {:?}",
            events[2]
        );
        assert!(
            event_match::is_session_stop(&events[3]),
            "expected SessionStop, got {:?}",
            events[3]
        );
        assert_eq!(events.len(), 4, "expected exactly 4 events, got {events:?}");
    }

    /// Verify that a tool-using session emits the full event sequence:
    /// `SessionStart → TurnStart → ToolStart → ToolComplete → TurnComplete
    /// → TurnStart → TurnComplete → SessionStop`.
    #[tokio::test]
    async fn test_event_sequence_with_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "echo", json!({"message": "x"}), "Done");

        let (agent, sink) = agent_with_recording_sink(client, registry, make_config());
        let result = agent.run("Test").await.unwrap();
        assert!(result.success);

        let events = sink.events();
        // Sequence: SessionStart, TurnStart, ToolStart, ToolComplete,
        // TurnComplete, TurnStart, TurnComplete, SessionStop
        assert!(
            event_match::is_session_start(&events[0]),
            "event[0] not SessionStart"
        );
        assert!(
            event_match::is_turn_start(&events[1]),
            "event[1] not TurnStart"
        );
        assert!(
            event_match::is_tool_start(&events[2]),
            "event[2] not ToolStart"
        );
        assert!(
            event_match::is_tool_complete(&events[3]),
            "event[3] not ToolComplete"
        );
        assert!(
            event_match::is_turn_complete(&events[4]),
            "event[4] not TurnComplete"
        );
        assert!(
            event_match::is_turn_start(&events[5]),
            "event[5] not TurnStart"
        );
        assert!(
            event_match::is_turn_complete(&events[6]),
            "event[6] not TurnComplete"
        );
        assert!(
            event_match::is_session_stop(&events[7]),
            "event[7] not SessionStop"
        );
        assert_eq!(
            events.len(),
            8,
            "expected exactly 8 events, got {}",
            events.len()
        );
    }

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

        let (agent, sink) = agent_with_recording_sink(client, registry, make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
        assert!(sink.any(event_match::is_tool_start), "missing tool_start");
        assert!(
            sink.any(event_match::is_tool_complete),
            "missing tool_complete"
        );
    }

    /// Verify that when a tool is not found, the recovery wiring still
    /// produces a soft error result (no hard error propagated).
    #[tokio::test]
    async fn test_recovery_on_missing_tool_returns_soft_result() {
        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "OK");

        let (agent, sink) = agent_with_recording_sink(client, ToolRegistry::new(), make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        assert_eq!(result.tool_calls, 1);
        // Should still emit tool_start and tool_complete (even for missing tools)
        assert!(sink.any(event_match::is_tool_start), "missing tool_start");
        assert!(
            sink.any(event_match::is_tool_complete),
            "missing tool_complete"
        );
    }

    /// Verify that a failing tool with the default recovery produces
    /// exactly one tool_start and one tool_complete event (NoopReflector
    /// marks everything as non-recoverable, so no retries).
    #[tokio::test]
    async fn test_recovery_noop_reflector_no_retries() {
        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_then_text("tool_1", "fail", json!({}), "OK");

        let (agent, sink) = agent_with_recording_sink(client, registry, make_config());
        let result = agent.run("Test").await.unwrap();

        assert!(result.success);
        // NoopReflector marks everything non-recoverable → Fail → no retry
        assert_eq!(
            sink.count_matching(event_match::is_tool_start),
            1,
            "expected exactly 1 tool_start (no retries)"
        );
        assert_eq!(
            sink.count_matching(event_match::is_tool_complete),
            1,
            "expected exactly 1 tool_complete (no retries)"
        );
    }

    /// Verify cancellation is still respected during tool recovery.
    #[tokio::test]
    async fn test_recovery_respects_cancellation() {
        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);

        let client = MockClient::new("test");
        client.add_tool_only_response("tc-1", "fail", json!({}));

        let sink = Arc::new(RecordingSink::new());
        let mut agent = BareLoop::new(Arc::new(client), registry, make_config());
        agent.event_sink = Arc::clone(&sink) as Arc<dyn EventSink>;

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

    impl crate::engine::middleware::ToolMiddleware for TurnNumberCapture {
        fn name(&self) -> &str {
            "turn_capture"
        }

        fn dispatch<'a>(
            &'a self,
            ctx: &'a mut ToolDispatchContext,
            next: &'a ToolPipeline,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = crate::engine::middleware::ToolDispatchResult>
                    + Send
                    + 'a,
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
