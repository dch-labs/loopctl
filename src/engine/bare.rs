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
use crate::core::{AgentConfig, AgentError, AgentObserver, SessionResult};
use crate::loop_control::bundle::ManagerBundle;
use crate::message::{Message, MessagePart, Role, ToolContent};
use crate::stream::{StreamAccumulator, StreamEvent, StreamStopReason, Usage};
use crate::tool::{ToolContext, ToolRegistry, ToolSchema};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
///       → stream_turn() ─→ dispatch_tools() ─→ stream_turn()
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
    tools: ToolRegistry,

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
    ///
    /// Currently marked `dead_code` until those features are wired in.
    #[allow(dead_code)]
    managers: ManagerBundle,

    /// Shared cancellation signal.
    ///
    /// Set via [`cancel()`](BareLoop::cancel). Checked at the top of every
    /// loop iteration (before streaming) and between tool dispatches.
    /// Streaming is also cancel-aware via `tokio::select!` — the loop
    /// will wake up mid-stream when cancelled.
    cancelled: Arc<CancelSignal>,
}

impl<C: ApiClient> BareLoop<C> {
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
            tools,
            config,
            conversation: Vec::new(),
            observers: Vec::new(),
            managers: ManagerBundle::new(),
            cancelled: Arc::new(CancelSignal::new()),
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
            tools,
            config,
            conversation: Vec::new(),
            observers,
            managers: ManagerBundle::new(),
            cancelled: Arc::new(CancelSignal::new()),
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
            tools,
            config,
            conversation: Vec::new(),
            observers,
            managers,
            cancelled: Arc::new(CancelSignal::new()),
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
            tools,
            config,
            conversation: Vec::new(),
            observers,
            managers,
            cancelled: Arc::new(CancelSignal::new()),
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
    #[allow(clippy::arithmetic_side_effects, clippy::cast_lossless)]
    pub async fn run(mut self, user_input: &str) -> Result<SessionResult, AgentError> {
        let session_id = self.config.session_id;
        let max_turns = self.config.max_turns;
        let start = Instant::now();
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut total_tool_calls: usize = 0;
        let mut turn_count: usize = 0;

        self.notify_session_start();
        self.conversation.push(Message::user(user_input));

        // Main agent loop
        loop {
            if self.is_cancelled() {
                self.notify_session_end(false, Some("Cancelled"));
                return Err(AgentError::Cancelled);
            }
            if turn_count >= max_turns {
                self.notify_session_end(false, Some("Max turns exceeded"));
                return Err(AgentError::MaxTurnsExceeded { max: max_turns });
            }

            self.notify_turn_start(user_input);
            let _turn_start = Instant::now();
            let stream_result = self.stream_turn().await;

            match stream_result {
                Ok((assistant_msg, usage, stop_reason)) => {
                    if let Some(u) = usage {
                        input_tokens += u.input_tokens as u64;
                        output_tokens += u.output_tokens as u64;
                    }
                    let text = Self::extract_text(&assistant_msg);
                    let tool_calls = Self::extract_tool_calls(&assistant_msg);
                    let has_tool_calls = !tool_calls.is_empty();

                    self.conversation.push(assistant_msg);
                    turn_count += 1;

                    if has_tool_calls {
                        let tool_result = self.dispatch_tools(&tool_calls).await;
                        match tool_result {
                            Ok(results) => {
                                total_tool_calls += results.len();
                                let tool_result_msg = Self::build_tool_result_message(results);
                                self.conversation.push(tool_result_msg);
                                self.notify_turn_end(true, None);
                            }
                            Err(AgentError::Cancelled) => {
                                self.notify_turn_end(false, Some("Cancelled"));
                                self.notify_session_end(false, Some("Cancelled"));
                                return Err(AgentError::Cancelled);
                            }
                            Err(e) => {
                                self.notify_turn_end(false, Some(&e.to_string()));
                                self.notify_session_end(false, Some(&e.to_string()));
                                return Err(e);
                            }
                        }
                    } else {
                        // No tool calls — session is done
                        let success = stop_reason == StreamStopReason::EndTurn;
                        let error = if success {
                            None
                        } else {
                            Some(format!("Stream stopped with reason: {stop_reason:?}"))
                        };
                        self.notify_turn_end(success, error.as_deref());
                        self.notify_session_end(success, error.as_deref());
                        let duration = start.elapsed();
                        return Ok(SessionResult {
                            session_id,
                            total_turns: turn_count,
                            input_tokens,
                            output_tokens,
                            total_duration: duration,
                            tool_calls: total_tool_calls,
                            success,
                            final_output: Some(text),
                            error,
                        });
                    }
                }
                Err(e) => {
                    self.notify_turn_end(false, Some(&e.to_string()));
                    self.notify_session_end(false, Some(&e.to_string()));
                    return Err(e);
                }
            }
        }
    }

    // ==================================================
    // Streaming
    // ==================================================

    /// Stream one turn from the API, accumulating the response.
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

    // ==================================================
    // Tool dispatch
    // ==================================================

    /// Execute tool calls and return results.
    ///
    /// Iterates over each [`ToolCallInfo`] extracted from the assistant
    /// message, looks up the corresponding tool in the [`ToolRegistry`],
    /// and invokes it. Each result is wrapped in a [`ToolCallResult`].
    ///
    /// Tool execution is **sequential** so that cancellation can be
    /// checked between invocations. A tool that is not found in the
    /// registry produces a soft error result (not a hard [`AgentError`]),
    /// allowing the model to recover.
    ///
    /// Observers are notified before and after each tool invocation via
    /// [`on_tool_call`](AgentObserver::on_tool_call) and
    /// [`on_tool_complete`](AgentObserver::on_tool_complete).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancellation flag is set
    /// between tool invocations.
    #[allow(clippy::single_match_else)]
    async fn dispatch_tools(
        &self,
        tool_calls: &[ToolCallInfo],
    ) -> Result<Vec<ToolCallResult>, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        let tool_context = self.build_tool_context();
        for tc in tool_calls {
            if self.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            self.notify_tool_call(&tc.name, &tc.input.to_string());
            let start = Instant::now();
            let tool_result = match self.tools.get(&tc.name) {
                Some(tool) => {
                    let cancel = Arc::clone(&self.cancelled);
                    let call_result = tokio::select! {
                        r = tool.call(tc.input.clone(), &tool_context) => r,
                        () = cancel.notified() => {
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
                            ToolCallResult {
                                tool_call_id: tc.id.clone(),
                                output: result.payload,
                                is_error: result.is_error,
                                duration,
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
                            ToolCallResult {
                                tool_call_id: tc.id.clone(),
                                output: ToolContent::Text(error_msg),
                                is_error: true,
                                duration,
                            }
                        }
                    }
                }
                None => {
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
                    ToolCallResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(error_msg),
                        is_error: true,
                        duration: Duration::ZERO,
                    }
                }
            };

            results.push(tool_result);
        }

        Ok(results)
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
    /// the model can distinguish success from failure.
    ///
    /// # Parameters
    ///
    /// - `results` — The [`ToolCallResult`]s produced by
    ///   [`dispatch_tools()`](BareLoop::dispatch_tools).
    ///
    /// # Returns
    ///
    /// A [`Message`] with [`Role::User`] and one `tool_result`
    /// [`MessagePart`] per result.
    fn build_tool_result_message(results: Vec<ToolCallResult>) -> Message {
        let parts: Vec<MessagePart> = results
            .into_iter()
            .map(|r| MessagePart::tool_result(r.tool_call_id, r.output, r.is_error))
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
    }

    /// Notify all observers that the session has ended.
    ///
    /// Called once when [`run()`](BareLoop::run) returns — whether
    /// successfully, due to an error, or because of cancellation.
    /// The `success` flag and optional `error` message let observers
    /// distinguish normal termination from failures.
    fn notify_session_end(&self, success: bool, error: Option<&str>) {
        for obs in &self.observers {
            obs.on_session_end(success, error);
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
    /// The `#[allow(clippy::too_many_arguments)]` annotation suppresses
    /// the lint for this notification method because all parameters are
    /// required by the [`AgentObserver::on_tool_complete`] trait method.
    #[allow(clippy::too_many_arguments)]
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
    /// tool call. Copied into [`ToolCallResult::tool_call_id`] after
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
// ToolCallResult (re-export)
// ==================================================

/// Result of executing a single tool call.
///
/// This is the internal version used during dispatch. Each executed tool
/// produces one `ToolCallResult`, which is then converted into a
/// `tool_result` [`MessagePart`] (with the output wrapped as
/// [`ToolContent`](ToolContent)) by
/// [`build_tool_result_message()`](BareLoop::build_tool_result_message)
/// and appended to the conversation.
///
/// This type is distinct from the framework-level
/// [`ToolCallResult`](crate::ToolCallResult) that appears in the final
/// [`TurnResult`]. The internal version carries execution metadata
/// (`duration`, `is_error`) needed for observer notifications.
///
/// # Fields
///
/// - [`tool_call_id`](ToolCallResult::tool_call_id) — Correlates back to
///   the original [`ToolCallInfo::id`].
/// - [`output`](ToolCallResult::output) — The tool's text output or
///   error message.
/// - [`is_error`](ToolCallResult::is_error) — `true` when the tool
///   returned an error or was not found.
/// - [`duration`](ToolCallResult::duration) — Wall-clock time spent
///   executing the tool.
#[derive(Debug, Clone)]
struct ToolCallResult {
    /// The tool call ID this result is for.
    ///
    /// Copied from [`ToolCallInfo::id`] to ensure the API can match
    /// the result to the original request.
    tool_call_id: String,

    /// The tool's output content or error message.
    ///
    /// On success this holds the full [`ToolOutput::payload`] returned by
    /// [`Tool::call()`](crate::tool::Tool::call) — preserving multipart and
    /// image content instead of flattening to text.
    /// On failure it contains an error message wrapped in
    /// [`ToolContent::Text`](ToolContent).
    /// This content is passed directly to [`MessagePart::tool_result`] when
    /// building the `tool_result` message part.
    output: ToolContent,

    /// Whether the execution resulted in an error.
    ///
    /// When `true`, the model receives this result with the `is_error`
    /// flag set, allowing it to decide whether to retry, use a
    /// different tool, or inform the user.
    is_error: bool,

    /// Wall-clock duration of the tool execution.
    ///
    /// Measured with [`Instant::now()`] around the
    /// [`Tool::call()`](crate::tool::Tool::call) invocation. Currently
    /// marked `dead_code` because it is not yet surfaced in the public
    /// [`SessionResult`], but it is passed to observer callbacks.
    #[allow(dead_code)]
    duration: Duration,
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
    #[allow(unused_imports)]
    use futures::stream;
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
        #[allow(dead_code)]
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

    /// Verify that a single-turn conversation (no tools) completes
    /// successfully and returns the model's text as `final_output`.
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

    /// Verify that a two-turn conversation (tool_call then end_turn)
    /// records one tool call and two total turns.
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

    /// Verify that exceeding `max_turns` produces
    /// [`AgentError::MaxTurnsExceeded`].
    ///
    /// The mock returns only `tool_call` responses so the loop never
    /// receives an `end_turn`. After 3 turns the loop should abort
    /// with the correct error variant.
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

    /// Verify that cancelling the loop before it starts returns a
    /// failed [`SessionResult`] (not an error).
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

    /// Verify that an API error (no mock responses) propagates as
    /// [`AgentError::Api`].
    #[tokio::test]
    async fn test_bare_loop_api_error() {
        let client = MockClient::new("test-model");
        // Don't add any responses — the mock will return an error

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

    /// Verify that requesting a nonexistent tool produces a soft error
    /// result (the loop continues), not a hard error.
    ///
    /// The model asks for a tool that isn't in the registry. The loop
    /// should create an error tool result, feed it back, and then the
    /// model's second response (end_turn) should complete the session
    /// successfully.
    #[tokio::test]
    async fn test_tool_not_found_returns_error_result() {
        let client = MockClient::new("test-model");
        client.add_tool_then_text("tool_1", "nonexistent", json!({}), "I see the tool failed.");

        let config = make_config();
        // Empty registry — tool won't be found
        let agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), config);

        let result = agent.run("Use nonexistent tool").await.unwrap();

        assert!(result.success);
        // The tool-not-found should be returned as an error result in the conversation,
        // not as a hard error. The loop should continue and eventually get the end_turn.
        assert_eq!(result.total_turns, 2);
    }

    /// Verify that a tool that returns an execution error is handled
    /// as a soft error result, allowing the session to continue.
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

    /// Verify that a single-turn session fires the expected observer
    /// callbacks: one session start, one session end, one turn start,
    /// and one turn end.
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

    /// Verify that a tool-using session fires tool call/complete
    /// callbacks in addition to the turn callbacks.
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
        let results = vec![super::ToolCallResult {
            tool_call_id: "tool_123".to_string(),
            output: ToolContent::Text("Echo: hello".to_string()),
            is_error: false,
            duration: Duration::from_millis(100),
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

    /// Verify that multiple tool_call parts in a single assistant
    /// message are all dispatched and counted.
    ///
    /// The mock emits two `tool_call` parts in one response, followed
    /// by an `end_turn` response. The session should report 2 turns
    /// and 2 tool calls.
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
}
