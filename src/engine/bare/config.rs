//! Configuration builders for [`BareLoop`] — the `set_*` / `with_*` methods.
//!
//! Every method here mutates a [`BareLoop`] field or forwards to a
//! [`LoopManagers`](crate::managers::LoopManagers) setter, gated by
//! [`debug_assert_idle`](BareLoop::debug_assert_idle). The fluent `with_*`
//! builders mirror the `set_*` mutators for chained construction.

#[cfg(feature = "hooks")]
use super::HookExecutor;

#[cfg(feature = "streaming")]
use super::StreamHandler;
#[cfg(feature = "tool_health")]
use super::ToolHealthRegistry;
use super::ToolPipelineBuilder;
use super::{
    ApiClient, Arc, BareLoop, ContextContributor, ContextManager, LoopError, RecoveryStrategy,
    Reflector, RequestOptions, TurnMode,
};
use crate::capabilities::PipelineAware;
use std::path::PathBuf;

/// Shared callback invoked once per text delta during streaming.
///
/// A clonable, thread-safe closure stored in [`BareLoop`] via
/// [`set_text_streamer`](BareLoop::set_text_streamer) and invoked from the
/// streaming engine path on every [`IndexedDelta`](crate::stream::IndexedDelta)
/// whose payload is [`Text`](crate::stream::DeltaPart::Text). The bounds
/// mirror the requirements of that path: `Send + Sync` because the engine may
/// dispatch deltas from an async task, and `Arc` so the same callback can be
/// shared across the engine and any observer without copying the closure.
#[cfg(feature = "streaming")]
pub(super) type TextStreamer = Arc<dyn Fn(&str) + Send + Sync>;

impl<C: ApiClient> BareLoop<C> {
    /// Set the [`Reflector`] for tool-error analysis.
    ///
    /// Replaces the default [`NoopReflector`](crate::reflection::NoopReflector) with a caller-supplied
    /// implementation. Must be called before [`run()`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., once the machine has advanced past [`MachineState::Start`](crate::engine::core::MachineState::Start)).
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
    /// Replaces the default [`ExponentialBackoffRecovery`](crate::reflection::ExponentialBackoffRecovery) with a
    /// caller-supplied implementation. Must be called before
    /// [`run()`](crate::engine::core::Loop::run).
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
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
    ///
    /// The session's `context_window` and `compact_threshold` are synced
    /// onto the manager (the same sync the default manager gets): the
    /// session config owns the window policy, and a host-installed
    /// manager that disagrees would trigger at the wrong point. Other
    /// manager settings (compactor, target base, counter) are the
    /// host's.
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
    ///     .with_threshold(80);
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_context_manager(Arc::new(manager));
    /// ```
    pub fn set_context_manager(&mut self, manager: Arc<ContextManager>) {
        self.debug_assert_idle();
        let synced = Arc::try_unwrap(manager)
            .unwrap_or_else(|arc| (*arc).clone())
            .with_context_window(self.session.config.context_window)
            .with_threshold(self.session.config.compact_threshold);
        self.managers.set_context_manager(Arc::new(synced));
    }

    /// Set the token counter for context-size estimates.
    ///
    /// The counter is used to estimate the conversation's token cost after
    /// each model response, which drives the compaction trigger. Defaults to
    /// [`HeuristicTokenCounter`](crate::compact::HeuristicTokenCounter) (a
    /// characters-per-token heuristic); swap in a real tokenizer (e.g.
    /// `tiktoken` for OpenAI) for better accuracy.
    ///
    /// This counter is the **fallback** used only when no [`ContextManager`] is
    /// configured. When a `ContextManager` is set (via
    /// [`set_context_manager`](Self::set_context_manager)), its own counter is
    /// the single source of truth for both the compaction trigger and the
    /// driver's context-size estimate (see `count_context`). The two counters
    /// are **not** kept in sync — each layer owns its own. To change the
    /// counter that the compactor uses, configure it on the `ContextManager`
    /// before passing it to `set_context_manager`.
    ///
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::compact::RatioTokenCounter;
    /// use std::sync::Arc;
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_token_counter(Arc::new(RatioTokenCounter::anthropic()));
    /// ```
    pub fn set_token_counter(&mut self, counter: Arc<dyn crate::compact::TokenCounter>) {
        self.debug_assert_idle();
        self.token_counter = counter;
    }

    /// Set the token counter, consuming `self`. Fluent mirror of
    /// [`set_token_counter`](Self::set_token_counter).
    #[must_use]
    pub fn with_token_counter(mut self, counter: Arc<dyn crate::compact::TokenCounter>) -> Self {
        self.set_token_counter(counter);
        self
    }

    /// Override the base directory under which the per-session temp
    /// subdir is created.
    ///
    /// By default the subdir lives under [`std::env::temp_dir()`]. Pass
    /// a custom `base` (e.g. `/var/cache/myapp/sessions`) and the subdir
    /// becomes `{base}/loopctl-{session_id}/`. Pass the empty path to
    /// opt out of managed temp entirely — the same effect as
    /// [`with_managed_temp_disabled`](Self::with_managed_temp_disabled).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, registry, config)
    ///     .with_temp_dir("/var/cache/myapp/sessions");
    /// ```
    #[must_use]
    pub fn with_temp_dir(mut self, base: impl Into<PathBuf>) -> Self {
        self.debug_assert_idle();
        let base = base.into();
        self.session_temp_dir = if base.as_os_str().is_empty() {
            None
        } else {
            Some(Self::session_temp_subdir(&base, self.session.id))
        };
        self
    }

    /// Opt out of loopctl-managed temp entirely.
    ///
    /// Tool contexts then set `temp_dir` to [`std::env::temp_dir()`]
    /// (the process-wide temp) and dropping the loop removes nothing.
    /// Use this only when the host manages its own temp lifecycle and
    /// wants the old behaviour.
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    #[must_use]
    pub fn with_managed_temp_disabled(mut self) -> Self {
        self.debug_assert_idle();
        self.session_temp_dir = None;
        self
    }

    #[cfg(feature = "streaming")]
    /// Set the [`StreamHandler`] for resilient streaming with retries,
    /// timeouts, and fallback to non-streaming.
    ///
    /// When set, the loop delegates streaming to the handler instead of
    /// using the inline streaming logic. Must be called before
    /// [`run()`](crate::engine::core::Loop::run).
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
    /// let handler = StreamHandler::new().with_timeout_config(
    ///     StreamTimeoutConfig {
    ///         initial_event_timeout: Duration::from_secs(60),
    ///         ..Default::default()
    ///     },
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
    /// tool dispatch, compaction, and run start/end. Hooks can
    /// short-circuit with [`HookAction::Block`](crate::hooks::HookAction::Block).
    /// [`HookAction::Ask`](crate::hooks::HookAction::Ask) is automatically downgraded to `Block` by the
    /// executor in [`crate::hooks::Interactivity::Headless`] mode (the default).
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
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
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
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

    /// Set the agent memory backend.
    ///
    /// When set, the engine stores a trajectory entry after each successful
    /// tool call, retrieves relevant entries as context before each turn,
    /// and consolidates the store at the end of a successful run. Must be
    /// called before [`run()`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::memory::InMemoryStore;
    /// use std::sync::Arc;
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_memory(Arc::new(InMemoryStore::new()));
    /// ```
    pub fn set_memory(&mut self, memory: Arc<dyn crate::memory::LoopMemory>) {
        self.debug_assert_idle();
        self.managers.set_memory(memory);
    }

    /// Set the demotion sink for compaction passes.
    ///
    /// When a compaction pass removes messages, the loop hands them to
    /// this sink before the compacted history replaces the old one —
    /// eviction becomes a handoff instead of a discard. Without a sink,
    /// the no-op default discards removed messages. Must be called before
    /// [`run()`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::compact::demote::MemoryDemotionSink;
    /// use loopctl::memory::InMemoryStore;
    /// use std::sync::Arc;
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// let memory = Arc::new(InMemoryStore::new());
    /// agent.set_demotion_sink(Arc::new(MemoryDemotionSink::new(memory)));
    /// ```
    pub fn set_demotion_sink(&mut self, sink: Arc<dyn crate::compact::demote::DemotionSink>) {
        self.debug_assert_idle();
        self.managers.set_demotion_sink(sink);
    }

    /// Set the middleware pipeline for tool dispatch.
    ///
    /// Replaces the default (no pipeline) with a caller-supplied
    /// [`ToolPipeline`](crate::middleware::ToolPipeline). When set, tool calls flow through the
    /// pipeline's middleware chain before reaching the registry.
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started.
    ///
    /// Build the pipeline using [`ToolPipeline::builder()`](crate::middleware::ToolPipeline::builder), adding middleware
    /// layers **without** calling `.with_core()` — the registry is injected
    /// automatically from `self.tools` so that schema generation and dispatch
    /// always share the same underlying registry:
    ///
    /// The pipeline is also the injection point for per-dispatch host state:
    /// a middleware may augment `ctx.tool_context` — set extensions, `cwd`,
    /// `is_non_interactive` — before the tool runs. See
    /// [`ToolContext`](crate::tool::ToolContext) ("Passing host state to
    /// tools") and the `host-state` example for the full pattern.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "testing")]
    /// # fn example() {
    /// # use std::sync::Arc;
    /// # use loopctl::config::SessionConfig;
    /// # use loopctl::engine::BareLoop;
    /// # use loopctl::testing::MockApiClient;
    /// # use loopctl::tool::ToolRegistry;
    /// use loopctl::middleware::{ToolPipeline, TimeoutMiddleware};
    ///
    /// let builder = ToolPipeline::builder()
    ///     .with_middleware(TimeoutMiddleware::from_secs(30));
    ///
    /// # let client = MockApiClient::new("demo");
    /// # let registry = ToolRegistry::new();
    /// # let config = SessionConfig::default();
    /// let mut agent = BareLoop::new(Arc::new(client), registry, config);
    /// agent.set_pipeline(builder).expect("static composition is valid");
    /// # }
    /// # fn main() {
    /// # #[cfg(feature = "testing")]
    /// # example();
    /// # }
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
    /// Must be called before [`run()`](crate::engine::core::Loop::run).
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

    #[cfg(feature = "streaming")]
    /// Set a real-time text streaming callback.
    ///
    /// The callback is invoked for each text delta token as it arrives
    /// from the API during [`run`](crate::engine::core::Loop::run) under the
    /// streaming turn mode. This enables real-time display of the model's
    /// output without waiting for the full turn to complete. Requires the
    /// `streaming` feature; no-op under the non-streaming path.
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
    pub fn set_text_streamer(&mut self, f: TextStreamer) {
        self.debug_assert_idle();
        self.text_streamer = Some(f);
    }

    /// Register a [`ContextContributor`] consulted at the top of every turn.
    ///
    /// Contributors are consulted in registration order after
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) and
    /// before the model call. Each contributor that returns [`Some`] message
    /// has that message re-emitted into the outbound request for the current
    /// turn only — the message is never recorded into the conversation
    /// history, so it does not accumulate across turns and vanishes the
    /// moment the contributor stops returning it. Registering a contributor
    /// that always returns the same message is the cheap way to keep a goal
    /// reminder in front of the model every turn without growing the
    /// history.
    ///
    /// With no contributors registered, the loop behaves identically to a loop
    /// built without any — the turn-top consultation is a single cheap branch.
    ///
    /// Must be called before
    /// [`run`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., once the machine has advanced past [`MachineState::Start`](crate::engine::core::MachineState::Start)).
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

    /// Remove every registered turn-boundary contributor.
    ///
    /// The opt-out seam for contributor wiring: [`FrontierProfile`](crate::presets::FrontierProfile)
    /// calls this to strip default construction's goal reminder, and a
    /// host can use it to reset contributor state before installing its
    /// own. Same pre-first-`run` rule as
    /// [`add_contributor`](Self::add_contributor): a no-op on an empty
    /// list, a debug-asserted boundary violation once the session has
    /// started.
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., once the machine has advanced past [`MachineState::Start`](crate::engine::core::MachineState::Start)).
    pub fn clear_contributors(&mut self) {
        self.debug_assert_idle();
        self.contributors.clear();
    }

    /// Apply a [`Profile`](crate::presets::Profile) to this loop, chainably.
    ///
    /// The convenience spelling of the profile system: delegates to the
    /// trait's [`apply`](crate::presets::Profile::apply) and returns the
    /// loop on success, so a host writes
    /// `BareLoop::new(..)?.with_profile(&FrontierProfile)?` — or applies
    /// the same profile mutably with
    /// [`FrontierProfile::apply`](crate::presets::FrontierProfile::apply).
    /// Must be called before the first
    /// [`run`](crate::engine::core::Loop::run).
    ///
    /// # Example
    ///
    /// ```rust
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use std::pin::Pin;
    /// use std::sync::Arc;
    /// use loopctl::api::{ApiClient, StreamRequest};
    /// use loopctl::engine::{BareLoop, Loop};
    /// use loopctl::presets::FrontierProfile;
    /// use loopctl::tool::ToolRegistry;
    ///
    /// struct DocClient;
    ///
    /// impl ApiClient for DocClient {
    ///     fn model(&self) -> String {
    ///         "doc".to_string()
    ///     }
    ///     fn stream_messages(
    ///         &self,
    ///         _request: &StreamRequest,
    ///     ) -> Pin<Box<dyn futures::Stream<Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>> + Send + 'static>> {
    ///         let events = vec![
    ///             Ok(loopctl::stream::StreamEvent::MessageStart(
    ///                 loopctl::stream::MessageStart {
    ///                     message: loopctl::stream::MessageMetadata {
    ///                         id: "doc".to_string(),
    ///                         role: "assistant".to_string(),
    ///                         model: "doc".to_string(),
    ///                     },
    ///                 },
    ///             )),
    ///             Ok(loopctl::stream::StreamEvent::PartStart(loopctl::stream::PartStart {
    ///                 index: 0,
    ///                 part: Some(loopctl::message::MessagePart::text("hi")),
    ///             })),
    ///             Ok(loopctl::stream::StreamEvent::PartStop { index: None }),
    ///             Ok(loopctl::stream::StreamEvent::MessageDelta(
    ///                 loopctl::stream::MessageDelta {
    ///                     delta: loopctl::stream::MessageDeltaPayload {
    ///                         stop_reason: Some("end_turn".to_string()),
    ///                     },
    ///                     usage: None,
    ///                 },
    ///             )),
    ///             Ok(loopctl::stream::StreamEvent::MessageStop),
    ///         ];
    ///         Box::pin(futures::stream::iter(events))
    ///     }
    ///     fn create_message(
    ///         &self,
    ///         _request: &StreamRequest,
    ///     ) -> Pin<Box<dyn std::future::Future<Output = Result<loopctl::api::NonStreamingResponse, loopctl::api::error::ApiError>> + Send + '_>> {
    ///         Box::pin(async {
    ///             Ok(loopctl::api::NonStreamingResponse {
    ///                 message: loopctl::message::Message::assistant("hi"),
    ///                 stop_reason: loopctl::stream::StreamStopReason::EndTurn,
    ///                 usage: None,
    ///             })
    ///         })
    ///     }
    /// }
    ///
    /// let mut agent = BareLoop::new(
    ///     Arc::new(DocClient),
    ///     ToolRegistry::new(),
    ///     Default::default(),
    /// )
    /// .with_profile(&FrontierProfile)
    /// .expect("the frontier profile applies");
    /// let run = agent.run("hello", &Default::default()).await;
    /// assert!(run.is_ok());
    /// # });
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates the profile's own error — the constrained arm installs
    /// a middleware pipeline and its validation can reject the
    /// composition; the frontier arm cannot fail.
    pub fn with_profile(
        mut self,
        profile: &impl crate::presets::Profile,
    ) -> Result<Self, crate::error::LoopError> {
        profile.apply(&mut self)?;
        Ok(self)
    }

    /// Wire the small-model pieces default construction carries.
    ///
    /// The P8 default, applied by every public constructor: the
    /// builtin-verified pipeline (output cap, verify-on-write, memoize
    /// with path-aware invalidation), the goal reminder re-injecting the
    /// original request past the cadence, and — only when the client's
    /// [`supports_tool_constraints`](crate::api::ApiClient::supports_tool_constraints)
    /// probe agrees — the strict tool-call constraint. Every seam is the
    /// public one a host would call, so a later host call replaces or
    /// composes on top exactly as before; the explicit opt-out is
    /// [`FrontierProfile`](crate::presets::FrontierProfile). The
    /// pipeline install fills only the vacant seam: a pipeline the
    /// managers bundle already carries is kept untouched, so
    /// [`LoopManagers::with_pipeline`] composes with the default
    /// instead of being clobbered by it. A resumed
    /// machine past [`Start`](crate::engine::core::MachineState::Start)
    /// is left untouched — the checkpoint's loop was configured before
    /// serialization, and re-wiring it would change the resumed run's
    /// shape.
    pub(crate) fn wire_default_profile(&mut self) {
        if !matches!(
            self.machine.state(),
            crate::engine::core::MachineState::Start
        ) {
            return;
        }
        if self.managers.pipeline().is_none()
            && let Err(error) = self.set_pipeline(
                crate::presets::ConstrainedProfile::pipeline_builder_with_builtin_verification(),
            )
        {
            tracing::warn!(
                error = %error,
                "default profile pipeline rejected; the loop runs bare"
            );
        }
        self.add_contributor(Box::new(crate::presets::GoalReminder::new(
            crate::presets::DEFAULT_GOAL_REMINDER_EVERY_N_TURNS,
        )));
        if self.client.supports_tool_constraints() {
            self.set_request_options(crate::presets::ConstrainedProfile::request_options());
        }
    }

    /// Set the per-turn [`RequestOptions`] applied to every provider call.
    ///
    /// Carries [`tool_constraint`](crate::structured::ToolConstraint) — set to
    /// [`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)
    /// for strict tool-call decoding (small-model reliability), or leave at the
    /// default ([`RequestOptions::default`]) for unconstrained behavior.
    ///
    /// Must be called before
    /// [`run`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., once the machine has advanced past [`MachineState::Start`](crate::engine::core::MachineState::Start)).
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
    ///
    /// Like every `set_*` configurator, this asserts the loop has not
    /// started running only in debug builds; a release build accepts a
    /// call between runs — after `run()` returns, even once the session
    /// has started — and the new options apply from the next provider
    /// turn (options are read per turn). Configure before the first
    /// `run()`; reconfiguring a live session is not a supported path.
    pub fn set_request_options(&mut self, options: RequestOptions) {
        self.debug_assert_idle();
        self.request_options = options;
    }

    /// Return the active [`TurnMode`].
    ///
    /// Reflects what was set via [`set_turn_mode`](Self::set_turn_mode) or
    /// the constructor default (`TurnMode::Streaming` when `streaming` is
    /// compiled in, [`TurnMode::NonStreaming`] otherwise).
    #[must_use]
    pub fn turn_mode(&self) -> TurnMode {
        self.turn_mode
    }

    /// Select how the engine fulfils each LLM turn.
    ///
    /// Pass [`TurnMode::NonStreaming`] to drive turns through
    /// [`ApiClient::create_message`] with no streaming machinery; pass
    /// `TurnMode::Streaming` (requires the `streaming` feature) to drive
    /// them through `StreamHandler` with per-delta observer callbacks.
    ///
    /// Must be called before
    /// [`run`](crate::engine::core::Loop::run).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if called after the session has started
    /// (i.e., once the machine has advanced past [`MachineState::Start`](crate::engine::core::MachineState::Start)).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::engine::{BareLoop, TurnMode};
    ///
    /// let mut agent = BareLoop::new(client, registry, config);
    /// agent.set_turn_mode(TurnMode::NonStreaming);
    /// ```
    pub fn set_turn_mode(&mut self, mode: TurnMode) {
        self.debug_assert_idle();
        self.turn_mode = mode;
    }

    /// Select the turn mode, consuming `self`. Fluent mirror of
    /// [`set_turn_mode`](BareLoop::set_turn_mode).
    #[must_use]
    pub fn with_turn_mode(mut self, mode: TurnMode) -> Self {
        self.set_turn_mode(mode);
        self
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
    #[cfg(feature = "streaming")]
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

    /// Set the agent memory backend, consuming `self`. Fluent mirror of
    /// [`set_memory`](BareLoop::set_memory).
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn crate::memory::LoopMemory>) -> Self {
        self.set_memory(memory);
        self
    }

    /// Set the demotion sink, consuming `self`. Fluent mirror of
    /// [`set_demotion_sink`](BareLoop::set_demotion_sink).
    #[must_use]
    pub fn with_demotion_sink(
        mut self,
        sink: Arc<dyn crate::compact::demote::DemotionSink>,
    ) -> Self {
        self.set_demotion_sink(sink);
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
    #[cfg(feature = "streaming")]
    #[must_use]
    pub fn with_text_streamer(mut self, f: TextStreamer) -> Self {
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
}
