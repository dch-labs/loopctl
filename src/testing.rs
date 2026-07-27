//! Testing utilities — mock components and fixture factories for loopctl tests.
//!
//! Reusable mocks and fixture factories for testing
//! code that depends on loopctl traits. Instead of wiring up real API
//! clients and tools in tests, these stubs can be used to exercise
//! agent logic in isolation, assert on streaming events, and verify tool
//! dispatch — all without network calls or external dependencies.
//!
//! The module is only compiled when the `testing` feature is enabled, so
//! it has zero impact on production builds.
//!
//! # Architecture
//!
//! The testing module follows a standard mock-object pattern: each mock
//! implements a production trait ([`ApiClient`], [`Tool`]) but returns
//! preconfigured data instead of calling external services. Builder methods
//! let you configure what the mock returns before passing it to the code
//! under test.
//!
//! [`MockApiClient`] implements [`ApiClient`] and serves canned
//! [`MockResponse`]s as `StreamEvent`s. Each response can carry text deltas,
//! tool-call requests, or errors. [`MockTool`] implements [`Tool`] and returns
//! a fixed result or error when invoked.
//!
//! # Available Mocks
//!
//! - [`MockApiClient`] — A canned [`ApiClient`] that returns preconfigured
//!   streaming responses, tool calls, or errors.
//! - [`MockTool`] — A stub [`Tool`] that returns a fixed result or error
//!   when invoked.
//!
//! # Supporting Types
//!
//! - [`MockResponse`] — A single canned response used by [`MockApiClient`].
//! - [`MockToolCall`] — A single canned tool call embedded in a [`MockResponse`].
//!
//! # Fixture Factories
//!
//! - [`test_message`] — Create a test user [`Message`].
//! - [`test_assistant_message`] — Create a test assistant [`Message`].
//! - [`test_tool_use_message`] — Create an assistant [`Message`] with
//!   tool-call content blocks.
//! - [`test_config`] — Create a test [`SessionConfig`] with sensible defaults.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::testing::{MockApiClient, MockTool, test_config};
//! use loopctl::tool::ToolRegistry;
//!
//! // Build a mock client that streams "Hello, world!" and stops.
//! let client = MockApiClient::new("test-model")
//!     .with_text_response("Hello, world!")
//!     .with_stop_reason("end_turn");
//!
//! // Build a mock tool and register it.
//! let tool = MockTool::new("echo", "Echoes input")
//!     .with_result("Echo: hello");
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(tool);
//!
//! // Grab a test agent config.
//! let config = test_config();
//! ```
//!
//! # Multi-turn Example
//!
//! ```
//! use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
//! use serde_json::json;
//!
//! let client = MockApiClient::new("test-model").with_responses(vec![
//!     MockResponse {
//!         text: "Let me look that up.".into(),
//!         tool_call: Some(MockToolCall {
//!             id: "call_1".into(),
//!             name: "search".into(),
//!             input: json!({"query": "rust"}),
//!         }),
//!         stop_reason: "tool_use".into(),
//!     },
//!     MockResponse {
//!         text: "Here is what I found.".into(),
//!         tool_call: None,
//!         stop_reason: "end_turn".into(),
//!     },
//! ]);

use crate::api::ApiClient;
use crate::api::error::ApiError;
use crate::config::SessionConfig;
use crate::message::{Message, MessagePart, Role};
use crate::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamEvent, Usage,
};
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};
use futures::Stream;
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use std::sync::Mutex;

/// A mock [`ApiClient`] that returns preconfigured streaming responses.
///
/// Use this in unit tests to exercise agent logic without making real
/// API calls. Configure what the client returns using the builder-style
/// methods:
///
/// - [`with_text_response`](MockApiClient::with_text_response) — set the
///   text the model "says".
/// - [`with_tool_call`](MockApiClient::with_tool_call) — make the model
///   request a tool invocation.
/// - [`with_stop_reason`](MockApiClient::with_stop_reason) — override the
///   stop reason (e.g. `"end_turn"`, `"tool_use"`).
/// - [`with_responses`](MockApiClient::with_responses) — queue multiple
///   responses for multi-turn tests.
/// - [`with_error`](MockApiClient::with_error) — simulate an API error on
///   every call.
///
/// For multi-turn scenarios, chain multiple [`MockResponse`] values via
/// [`with_responses`](MockApiClient::with_responses). Each call to
/// [`stream_messages`](ApiClient::stream_messages) pops the next response;
/// once exhausted the last response is repeated.
///
/// # Construction
///
/// ```rust
/// use loopctl::testing::MockApiClient;
///
/// let client = MockApiClient::new("test-model")
///     .with_text_response("I'm here to help!");
/// ```
///
/// # Multi-turn
///
/// ```rust
/// use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
/// use serde_json::json;
///
/// let client = MockApiClient::new("test-model").with_responses(vec![
///     MockResponse {
///         text: "Let me look that up.".into(),
///         tool_call: Some(MockToolCall {
///             id: "call_1".into(),
///             name: "search".into(),
///             input: json!({"query": "rust"}),
///         }),
///         stop_reason: "tool_use".into(),
///     },
///     MockResponse {
///         text: "Here is what I found.".into(),
///         tool_call: None,
///         stop_reason: "end_turn".into(),
///     },
/// ]);
/// ```
#[derive(Clone)]
pub struct MockApiClient {
    /// The model name returned by [`model`](ApiClient::model).
    ///
    /// Set via [`MockApiClient::new`]. Typically something like
    /// `"test-model"` — it is not validated against any real model
    /// registry.
    ///
    /// The value is returned verbatim by the [`ApiClient::model`]
    /// implementation. It appears in log messages and
    /// [`MessageMetadata`] fields.
    model_name: Arc<std::sync::Mutex<String>>,

    /// The queue of canned responses.
    ///
    /// Each call to [`stream_messages`](ApiClient::stream_messages) pops
    /// the front entry. When only one entry remains it is cloned and
    /// reused so the client never runs dry.
    ///
    /// Shared via `Arc<Mutex<...>>` so the cloned handle inside the
    /// `impl ApiClient` methods can mutate the queue.
    ///
    /// Use [`MockApiClient::with_responses`] to replace the entire
    /// queue or individual builder methods to mutate the front entry.
    responses: Arc<Mutex<Vec<MockResponse>>>,

    /// When set, every call returns an [`ApiError`] instead of a normal
    /// response.
    ///
    /// Set via [`MockApiClient::with_error`]. Takes precedence over all
    /// other configuration — useful for testing error-handling paths.
    error: Option<String>,
}

/// A single canned response produced by [`MockApiClient`].
///
/// Each [`MockResponse`] describes exactly one assistant reply: the text
/// the model "says", an optional tool call, and a stop reason. The mock
/// client translates these fields into the appropriate
/// [`StreamEvent`] sequence when [`stream_messages`](ApiClient::stream_messages)
/// is called.
///
/// # Construction
///
/// Build directly or via [`MockApiClient::with_responses`]:
///
/// ```rust
/// use loopctl::testing::MockResponse;
///
/// let response = MockResponse {
///     text: "Done.".to_string(),
///     tool_call: None,
///     stop_reason: "end_turn".to_string(),
/// };
/// ```
#[derive(Clone, Default)]
pub struct MockResponse {
    /// The text content the assistant should produce.
    ///
    /// Emitted as a [`IndexedDelta`] event containing a
    /// [`DeltaPart::Text`] variant. Defaults to `"Hello!"` when
    /// constructed via [`MockApiClient::new`].
    ///
    /// The text is sent as a single delta — the mock does not break it
    /// into chunks. If your test needs to verify incremental streaming,
    /// construct the [`StreamEvent`] sequence manually.
    pub text: String,

    /// An optional tool call the assistant should make.
    ///
    /// When present, the mock emits a second content block of type
    /// [`MessagePart::ToolCall`] and sets the stop reason to
    /// `"tool_use"`. See [`MockToolCall`] for the fields involved.
    ///
    /// Set to `None` for a plain text response. Set to `Some(...)` when
    /// testing tool dispatch or multi-turn tool-call scenarios.
    pub tool_call: Option<MockToolCall>,

    /// The stop reason for this response.
    ///
    /// Common values are `"end_turn"` (normal completion) and
    /// `"tool_use"` (agent should execute a tool). Defaults to
    /// `"end_turn"`.
    ///
    /// The agent loop uses this to decide whether to continue
    /// processing or to finalize the session. Setting it to
    /// `"max_tokens"` is useful for testing truncation handling.
    pub stop_reason: String,
}

/// A single canned tool call embedded in a [`MockResponse`].
///
/// When [`MockResponse::tool_call`] is `Some`, the mock client emits a
/// `tool_use` content block with these fields, allowing tests to verify
/// that the agent correctly dispatches tool invocations.
///
/// # Example
///
/// ```rust
/// use loopctl::testing::MockToolCall;
/// use serde_json::json;
///
/// let call = MockToolCall {
///     id: "call_1".to_string(),
///     name: "search".to_string(),
///     input: json!({"query": "hello"}),
/// };
/// ```
#[derive(Clone)]
pub struct MockToolCall {
    /// The tool call identifier.
    ///
    /// Matches the `id` field on the [`MessagePart::ToolCall`] variant
    /// produced by the mock. Agents use this ID to correlate the tool
    /// result back to the originating request.
    ///
    /// Typically a string like `"call_1"`, `"call_2"`, etc. Must be
    /// unique within a single assistant turn to avoid ambiguity.
    pub id: String,

    /// The name of the tool to invoke.
    ///
    /// Must correspond to a tool registered in the agent's
    /// [`ToolRegistry`](crate::tool::ToolRegistry). The mock does not
    /// validate this — passes the name through directly.
    ///
    /// During test execution, if the agent loop cannot find a tool
    /// with this name in the registry, it will return an error.
    pub name: String,

    /// The tool input as a JSON value.
    ///
    /// Serialized as the `input` field on the [`MessagePart::ToolCall`]
    /// variant. Use [`serde_json::json!`] to construct it ergonomically.
    ///
    /// The value is passed as-is to the tool's [`Tool::call`] method
    /// (in a real scenario; the mock itself ignores it).
    pub input: Value,
}

/// Construction and builder methods for [`MockApiClient`].
///
/// The builder pattern lets you configure mock responses fluently.
/// All builder methods consume and return `Self`, so you can chain
/// them directly after [`MockApiClient::new`].
///
/// # Response lifecycle
///
/// [`new`](MockApiClient::new) preloads the client with a single default
/// response (text `"Hello!"`, stop reason `"end_turn"`). The builder methods
/// then mutate or replace that queue:
///
/// - [`with_text_response`](MockApiClient::with_text_response) mutates the text
///   on the front response.
/// - [`with_tool_call`](MockApiClient::with_tool_call) adds a tool call and sets
///   the stop reason to `"tool_use"`.
/// - [`with_stop_reason`](MockApiClient::with_stop_reason) overrides the stop
///   reason.
/// - [`with_responses`](MockApiClient::with_responses) replaces the entire
///   response queue.
/// - [`with_error`](MockApiClient::with_error) forces an error on every call.
impl MockApiClient {
    /// Create a new mock client with the given model name.
    ///
    /// Returns a client preloaded with a single default response:
    /// text `"Hello!"` with stop reason `"end_turn"` and no tool call.
    /// Use the builder methods to customize before passing the client
    /// to the code under test.
    ///
    /// The default response is simple so that most tests
    /// only need to call [`with_text_response`](MockApiClient::with_text_response)
    /// to get started.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::ApiClient;
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model");
    /// assert_eq!(client.model(), "test-model");
    /// ```
    #[must_use]
    pub fn new(model: &str) -> Self {
        // Build the default single-response queue.
        //
        // This response is intentionally minimal — just text `"Hello!"`
        // with `"end_turn"` — so that tests that don't care about the
        // response content can use the mock without any configuration.
        let default_response = MockResponse {
            text: "Hello!".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        };
        Self {
            model_name: Arc::new(std::sync::Mutex::new(model.to_string())),
            responses: Arc::new(Mutex::new(vec![default_response])),
            error: None,
        }
    }

    /// Set the text response for the first (or only) turn.
    ///
    /// Overwrites the `text` field on the initial [`MockResponse`]
    /// created by [`new`](MockApiClient::new). Simplest way
    /// to configure a single-turn mock — the model will "say" the given
    /// text and stop.
    ///
    /// If you need to set text for multiple turns, use
    /// [`with_responses`](MockApiClient::with_responses) instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model")
    ///     .with_text_response("I am a test assistant.");
    /// ```
    #[must_use]
    pub fn with_text_response(self, text: &str) -> Self {
        if let Some(r) = crate::error::recover_guard(self.responses.lock()).first_mut() {
            r.text = text.to_string();
        }
        self
    }

    /// Add a tool call to the first (or only) response.
    ///
    /// The mock will emit a `tool_use` content block and set the stop
    /// reason to `"tool_use"` so the agent loop knows to execute the
    /// tool. The `id`, `name`, and `input` parameters map directly to
    /// the fields on [`MockToolCall`].
    ///
    /// The tool `name` should match a tool registered in the agent's
    /// [`ToolRegistry`](crate::tool::ToolRegistry) — otherwise the
    /// agent loop will fail when it tries to dispatch the call.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockApiClient;
    /// use serde_json::json;
    ///
    /// let client = MockApiClient::new("test-model")
    ///     .with_tool_call("call_1", "bash", json!({"command": "ls"}));
    /// ```
    #[must_use]
    pub fn with_tool_call(self, id: &str, name: &str, input: Value) -> Self {
        let mut responses = crate::error::recover_guard(self.responses.lock());
        if let Some(r) = responses.first_mut() {
            r.tool_call = Some(MockToolCall {
                id: id.to_string(),
                name: name.to_string(),
                input,
            });
            r.stop_reason = "tool_use".to_string();
        }
        drop(responses);
        self
    }

    /// Override the stop reason on the first (or only) response.
    ///
    /// Common values are `"end_turn"` (default) and `"tool_use"`.
    ///
    /// Note that [`with_tool_call`](MockApiClient::with_tool_call)
    /// automatically sets the stop reason to `"tool_use"`, so this method
    /// should be used when you want a different value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model")
    ///     .with_stop_reason("max_tokens");
    /// ```
    #[must_use]
    pub fn with_stop_reason(self, reason: &str) -> Self {
        if let Some(r) = crate::error::recover_guard(self.responses.lock()).first_mut() {
            r.stop_reason = reason.to_string();
        }
        self
    }

    /// Set the full response queue for multi-turn behaviour.
    ///
    /// Each call to [`stream_messages`](ApiClient::stream_messages)
    /// pops the front entry. When only one entry remains it is cloned
    /// and reused, so the mock never panics on an empty queue.
    ///
    /// If `responses` is empty the call is a no-op (the default response
    /// is retained). Recommended way to set up complex
    /// multi-turn scenarios where the model needs to reply differently
    /// across successive turns.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::{MockApiClient, MockResponse};
    ///
    /// let client = MockApiClient::new("test-model").with_responses(vec![
    ///     MockResponse {
    ///         text: "First reply".into(),
    ///         tool_call: None,
    ///         stop_reason: "end_turn".into(),
    ///     },
    ///     MockResponse {
    ///         text: "Second reply".into(),
    ///         tool_call: None,
    ///         stop_reason: "end_turn".into(),
    ///     },
    /// ]);
    /// ```
    #[must_use]
    pub fn with_responses(self, responses: Vec<MockResponse>) -> Self {
        if !responses.is_empty() {
            *crate::error::recover_guard(self.responses.lock()) = responses;
        }
        self
    }

    /// Simulate an API error on every call.
    ///
    /// Once set, both [`stream_messages`](ApiClient::stream_messages)
    /// and [`create_message`](ApiClient::create_message) will return
    /// an [`ApiError`] immediately. This overrides any response
    /// configuration.
    ///
    /// Useful for exercising agent error-handling and retry paths.
    /// The error message is passed through to the [`ApiError`] so
    /// tests can assert on the specific error string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model")
    ///     .with_error("rate limit exceeded");
    /// ```
    #[must_use]
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }

    /// Pop the next [`MockResponse`] from the internal queue.
    ///
    /// If more than one response is queued the front entry is removed
    /// and returned. If only one remains it is cloned in place so the
    /// queue never empties. This ensures repeated calls to
    /// [`stream_messages`](ApiClient::stream_messages) always succeed.
    ///
    /// Helper used by both
    /// [`stream_messages`](ApiClient::stream_messages) and
    /// [`create_message`](ApiClient::create_message). The method
    /// acquires the `Mutex` guard, so callers do not need
    /// to handle locking.
    ///
    /// # Queue exhaustion strategy
    ///
    /// ```text
    /// [R1, R2, R3]  → pop → R1, queue becomes [R2, R3]
    /// [R2, R3]      → pop → R2, queue becomes [R3]
    /// [R3]          → pop → R3, queue stays   [R3] (cloned)
    /// ```
    fn pop_response(&self) -> MockResponse {
        let mut guard = crate::error::recover_guard(self.responses.lock());
        if guard.len() > 1 {
            guard.remove(0)
        } else {
            guard.first().cloned().unwrap_or_default()
        }
    }
}

/// Trait implementation that turns canned [`MockResponse`] values into
/// real [`StreamEvent`] sequences and JSON payloads.
///
/// Both [`stream_messages`](ApiClient::stream_messages) and
/// [`create_message`](ApiClient::create_message) consult the internal
/// response queue (or the error override) and produce output that is
/// indistinguishable from a live API at the protocol level.
///
/// The translation layer converts each [`MockResponse`] into a
/// well-formed stream of [`StreamEvent`] variants — `MessageStart`,
/// `PartStart`, `IndexedDelta`, `MessagePartStop`,
/// `MessageDelta`, and `MessageStop` — so that any consumer expecting
/// the real API event protocol works without modification.
///
/// # Error path
///
/// When [`MockApiClient::with_error`] has been called, both methods
/// short-circuit and return an [`ApiError`] immediately, bypassing the
/// response queue entirely. This allows tests to verify error-handling
/// and retry logic without needing a flaky network.
///
/// # Ignored parameters
///
/// The `_messages`, `_system`, and `_tools` parameters are accepted for
/// trait compatibility but ignored — the mock always
/// returns its preconfigured response regardless of the input.
impl ApiClient for MockApiClient {
    /// Return the model name this mock was created with.
    ///
    /// Called by the framework to identify which model is being used
    /// throughout the session. Always returns the string passed to
    /// [`MockApiClient::new`] — the value is not validated against any
    /// real model registry, so any string is acceptable for testing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::ApiClient;
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("my-test-model");
    /// assert_eq!(client.model(), "my-test-model");
    /// ```
    fn model(&self) -> String {
        crate::error::recover_guard(self.model_name.lock()).clone()
    }

    /// Hot-swap the mock's model name at runtime.
    ///
    /// Unlike the trait default (which returns `false`), the mock stores
    /// its model behind a mutex and updates it in place so tests can
    /// exercise [`BareLoop::switch_model`](crate::engine::BareLoop::switch_model)
    /// and verify the new name is observed by subsequent [`model`](Self::model)
    /// calls. Returns `false` (no-op) when `model` is empty or whitespace,
    /// matching the trait contract that an empty model is not a valid switch.
    fn set_model(&self, model: &str) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
        true
    }

    /// Stream a canned sequence of [`StreamEvent`]s for the next response.
    ///
    /// Called by the agent loop to obtain the model's reply. The mock
    /// translates the current [`MockResponse`] into the standard event
    /// sequence:
    ///
    /// ```text
    /// MessageStart → PartStart(text) → IndexedDelta(text) → MessagePartStop
    ///              → [PartStart(tool_use) → MessagePartStop]   (if tool_call is set)
    ///              → MessageDelta(stop_reason, usage) → MessageStop
    /// ```
    ///
    /// If [`with_error`](MockApiClient::with_error) was called the stream
    /// contains a single [`ApiError`] event instead.
    ///
    /// The `_messages`, `_system`, and `_tools` parameters are accepted for
    /// trait compatibility but ignored — the mock always
    /// returns the preconfigured response.
    ///
    /// # Usage tokens
    ///
    /// The mock always reports 50 input tokens and 25 output tokens in
    /// the [`MessageDelta`] event. This lets tests assert on usage data
    /// without needing a real model response.
    ///
    /// # Example
    ///
    /// ```rust
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use loopctl::api::{ApiClient, StreamRequest};
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model").with_text_response("Hi!");
    /// let stream = client.stream_messages(StreamRequest::new(vec![]));
    /// let events: Vec<_> = futures::StreamExt::collect(stream).await;
    /// assert!(events.len() >= 4);
    /// # });
    /// ```
    fn stream_messages(
        &self,
        _request: crate::api::StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        if let Some(ref err) = self.error {
            let err = err.clone();
            return Box::pin(futures::stream::once(
                async move { Err(ApiError::api(&err)) },
            ));
        }

        let response = self.pop_response();
        let model = crate::error::recover_guard(self.model_name.lock()).clone();
        let mut events: Vec<Result<StreamEvent, ApiError>> =
            vec![Ok(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_test".to_string(),
                    role: "assistant".to_string(),
                    model,
                },
            }))];

        // Text content block
        let text = response.text.clone();
        events.push(Ok(StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("")),
        })));

        // Text delta
        events.push(Ok(StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text { text },
        })));

        events.push(Ok(StreamEvent::PartStop));

        // Tool call content block (if any)
        if let Some(tc) = &response.tool_call {
            events.push(Ok(StreamEvent::PartStart(PartStart {
                index: 1,
                part: Some(MessagePart::tool_call(&tc.id, &tc.name, tc.input.clone())),
            })));
            events.push(Ok(StreamEvent::PartStop));
        }

        // Message delta with stop reason
        events.push(Ok(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(response.stop_reason),
            },
            usage: Some(Usage::new(50, 25)),
        })));

        events.push(Ok(StreamEvent::MessageStop));

        Box::pin(futures::stream::iter(events))
    }

    /// Return a canned non-streaming JSON response.
    ///
    /// Called by code paths that use the non-streaming API. The mock
    /// returns a JSON object with a `content` array containing a single
    /// text block drawn from the current [`MockResponse`].
    ///
    /// If [`with_error`](MockApiClient::with_error) was called the
    /// future resolves to an [`ApiError`] instead, bypassing the
    /// response queue entirely.
    ///
    /// The `_request` parameter is accepted for trait compatibility but ignored.
    ///
    /// # Example
    ///
    /// ```rust
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use loopctl::api::{ApiClient, StreamRequest};
    /// use loopctl::testing::MockApiClient;
    ///
    /// let client = MockApiClient::new("test-model").with_text_response("Hi!");
    /// let result = client.create_message(StreamRequest::new(vec![])).await;
    /// assert_eq!(result.unwrap()["content"][0]["text"], "Hi!");
    /// # });
    /// ```
    fn create_message(
        &self,
        _request: crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        if let Some(ref err) = self.error {
            let err = err.clone();
            return Box::pin(async move { Err(ApiError::api(&err)) });
        }

        let response = self.pop_response();
        Box::pin(async move {
            Ok(json!({
                "content": [{"type": "text", "text": response.text}]
            }))
        })
    }
}

/// A mock [`Tool`] that returns a fixed result or error when called.
///
/// Useful for testing tool dispatch, registries, and agent-loop tool
/// execution without implementing a real tool. Configure the behaviour
/// via the builder-style methods:
///
/// - [`with_result`](MockTool::with_result) — set the text result
///   returned on success.
/// - [`with_error`](MockTool::with_error) — make the tool return a
///   [`ToolError::Execution`] instead.
/// - [`with_concurrency_safe`](MockTool::with_concurrency_safe) — set
///   the concurrency-safety flag.
/// - [`with_read_only`](MockTool::with_read_only) — set the read-only
///   flag.
/// - [`with_schema`](MockTool::with_schema) — override the input JSON
///   schema.
/// - [`with_system_prompt`](MockTool::with_system_prompt) — attach a
///   system prompt that the framework injects when the tool is available.
///
/// # Construction
///
/// ```rust
/// use loopctl::testing::MockTool;
///
/// let tool = MockTool::new("echo", "Echoes input")
///     .with_result("Echo: hello")
///     .with_concurrency_safe(true);
/// ```
///
/// # Example — registering in a tool registry
///
/// ```rust
/// use loopctl::testing::MockTool;
/// use loopctl::tool::ToolRegistry;
///
/// let tool = MockTool::new("echo", "Echoes input")
///     .with_result("Echo: hello");
///
/// let mut registry = ToolRegistry::new();
/// registry.register(tool);
///
/// assert!(registry.contains("echo"));
/// ```
pub struct MockTool {
    /// The tool name, returned by [`Tool::name`].
    ///
    /// Set via [`MockTool::new`]. Must be unique within a
    /// [`ToolRegistry`](crate::tool::ToolRegistry).
    ///
    /// The framework uses this to match tool-call requests from the
    /// model to the correct [`Tool`] implementation.
    name: String,

    /// The human-readable description, returned by [`Tool::description`].
    ///
    /// Set via [`MockTool::new`]. Shown to the model as part of the
    /// tool schema so it can decide which tool to invoke.
    ///
    /// The description should be concise yet informative — it is
    /// included verbatim in the [`ToolSchema`] sent to the API.
    description: String,

    /// The JSON schema for tool input, returned by [`Tool::schema`].
    ///
    /// Defaults to a trivial `{"type": "object", "properties": {"input": {"type": "string"}}}`.
    /// Override with [`MockTool::with_schema`] when the code under test
    /// validates the schema.
    ///
    /// The schema is included in the [`ToolSchema`] struct that the
    /// framework sends to the model API when listing available tools.
    input_schema: Value,

    /// The text result returned on success.
    ///
    /// Wrapped in [`ToolOutput::text`] by the [`Tool::call`]
    /// implementation. Set via [`MockTool::with_result`]. Defaults to
    /// `"mock result"`.
    ///
    /// If [`MockTool::with_error`] is called, this string is used as
    /// the error message in the [`ToolError::Execution`] variant instead.
    result: String,

    /// Whether [`Tool::call`] should return a [`ToolError::Execution`].
    ///
    /// When `true`, the `result` field is used as the error message.
    /// Set via [`MockTool::with_error`].
    ///
    /// Defaults to `false`. When testing error paths, set this to `true`
    /// after configuring the result message.
    is_error: bool,

    /// Whether this tool is safe to run concurrently with others.
    ///
    /// Returned by [`Tool::is_concurrency_safe`]. Defaults to `false`.
    /// Set via [`MockTool::with_concurrency_safe`].
    ///
    /// When `true`, the tool executor may schedule this tool in parallel
    /// with other concurrency-safe tools, reducing overall latency.
    is_concurrency_safe: bool,

    /// Whether this tool only reads data (no side effects).
    ///
    /// Returned by [`Tool::is_read_only`]. Defaults to `true` because
    /// most test tools don't need to simulate writes.
    ///
    /// The framework may use this hint to optimise scheduling — for
    /// example, running multiple read-only tools in parallel while
    /// serialising write operations.
    is_read_only: bool,

    /// Artificial delay injected into [`Tool::call`] before it resolves.
    ///
    /// Defaults to zero (instant resolution). Set via
    /// [`MockTool::with_delay`] when testing timing-sensitive behaviour
    /// such as parallel dispatch overlap or cancellation.
    delay: std::time::Duration,

    /// An optional system prompt injected when the tool is available.
    ///
    /// Returned by [`Tool::system_prompt`]. Defaults to `None`. Set via
    /// [`MockTool::with_system_prompt`] when testing system-prompt
    /// assembly logic.
    ///
    /// The framework concatenates all tool system prompts and appends
    /// them to the agent's system message before sending to the model.
    system_prompt: Option<String>,
}

/// Construction and builder methods for [`MockTool`].
///
/// The builder pattern lets you configure the mock's behaviour fluently.
/// All builder methods consume and return `Self`, so you can chain them
/// directly after [`MockTool::new`].
///
/// # Configuration matrix
///
/// | Method                                | Affects                  |
/// |---------------------------------------|--------------------------|
/// | [`MockTool::with_result`]             | Text returned on success |
/// | [`MockTool::with_error`]              | Switches to error path   |
/// | [`MockTool::with_concurrency_safe`]   | Concurrency flag         |
/// | [`MockTool::with_read_only`]          | Read-only flag           |
/// | [`MockTool::with_schema`]             | JSON input schema        |
/// | [`MockTool::with_system_prompt`]      | System prompt text       |
impl MockTool {
    /// Create a new mock tool with the given name and description.
    ///
    /// Returns a tool with sensible defaults that can be registered in a
    /// [`ToolRegistry`](crate::tool::ToolRegistry) immediately. Use the
    /// builder methods to customise behaviour before registration.
    ///
    /// Defaults:
    ///
    /// | Property              | Default                                                            |
    /// |-----------------------|--------------------------------------------------------------------|
    /// | `result`              | `"mock result"`                                                    |
    /// | `is_error`            | `false`                                                            |
    /// | `is_concurrency_safe` | `false`                                                            |
    /// | `is_read_only`        | `true`                                                             |
    /// | `input_schema`        | `{"type":"object","properties":{"input":{"type":"string"}}}`       |
    /// | `system_prompt`       | `None`                                                             |
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("calculator", "Performs arithmetic");
    /// ```
    #[must_use]
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "input": { "type": "string" } }
            }),
            result: "mock result".to_string(),
            is_error: false,
            is_concurrency_safe: false,
            is_read_only: true,
            delay: std::time::Duration::ZERO,
            system_prompt: None,
        }
    }

    /// Set the text result this tool returns on success.
    ///
    /// The value is wrapped in [`ToolOutput::text`] when
    /// [`Tool::call`] is invoked. If [`MockTool::with_error`] is also
    /// called, this string is used as the error message instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("echo", "Echoes input")
    ///     .with_result("Echo: hello");
    /// ```
    #[must_use]
    pub fn with_result(mut self, result: &str) -> Self {
        self.result = result.to_string();
        self
    }

    /// Make this tool return a [`ToolError::Execution`] instead of a
    /// successful result.
    ///
    /// The `result` value (set via [`MockTool::with_result`]) is used as the error
    /// message string. Useful for testing agent error-handling and
    /// retry logic. Call this *after* [`MockTool::with_result`] to
    /// ensure the error message is set correctly.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("fail", "Always fails")
    ///     .with_result("something went wrong")
    ///     .with_error();
    /// ```
    #[must_use]
    pub fn with_error(mut self) -> Self {
        self.is_error = true;
        self
    }

    /// Set the concurrency-safety flag.
    ///
    /// When `true`, the framework may invoke this tool concurrently
    /// with other concurrency-safe tools. Returned by
    /// [`Tool::is_concurrency_safe`].
    ///
    /// Defaults to `false` — most test tools don't need concurrency.
    /// Set to `true` when testing the framework's parallel tool
    /// execution logic.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("read", "Reads data")
    ///     .with_concurrency_safe(true);
    /// ```
    #[must_use]
    pub fn with_concurrency_safe(mut self, safe: bool) -> Self {
        self.is_concurrency_safe = safe;
        self
    }

    /// Set the read-only flag.
    ///
    /// When `true` (the default), the tool is considered side-effect
    /// free. Returned by [`Tool::is_read_only`].
    ///
    /// Set to `false` when testing that the framework serialises
    /// write operations correctly — e.g. two `write` tools should
    /// not execute concurrently.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("write", "Writes data")
    ///     .with_read_only(false);
    /// ```
    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.is_read_only = read_only;
        self
    }

    /// Inject an artificial delay into [`Tool::call`] before it resolves.
    ///
    /// Defaults to zero (instant resolution). Use this when testing
    /// timing-sensitive behaviour such as parallel-dispatch overlap,
    /// cancellation-during-execution, or per-event timeouts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::time::Duration;
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("slow", "A slow tool")
    ///     .with_delay(Duration::from_millis(50));
    /// ```
    #[must_use]
    pub fn with_delay(mut self, delay: std::time::Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Override the input JSON schema.
    ///
    /// The default schema is a trivial object with a single `input`
    /// string property. Use this when the code under test validates
    /// tool schemas, generates documentation from them, or when the
    /// model needs a richer schema to produce correct tool calls.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    /// use serde_json::json;
    ///
    /// let tool = MockTool::new("search", "Searches the web")
    ///     .with_schema(json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "query": { "type": "string" },
    ///             "limit": { "type": "integer" }
    ///         },
    ///         "required": ["query"]
    ///     }));
    /// ```
    #[must_use]
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.input_schema = schema;
        self
    }

    /// Attach a system prompt that the framework injects when this tool
    /// is available.
    ///
    /// Returned by [`Tool::system_prompt`]. Useful for testing that the
    /// agent correctly assembles system prompts from tool metadata.
    ///
    /// When the tool is registered, the framework concatenates all tool
    /// system prompts into the system message sent to the model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::testing::MockTool;
    ///
    /// let tool = MockTool::new("bash", "Runs shell commands")
    ///     .with_system_prompt("Prefer simple commands over pipelines.");
    /// ```
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }
}

/// Trait implementation that returns canned tool metadata and results.
///
/// Every method delegates to the fields configured via the builder
/// methods on [`MockTool`]. The [`call`](Tool::call) implementation
/// ignores its `_input` and `_context` parameters entirely, returning
/// either [`ToolOutput::text`] or [`ToolError::Execution`] depending
/// on whether [`MockTool::with_error`] was called.
///
/// # Metadata methods
///
/// The [`name`](Tool::name), [`description`](Tool::description), and
/// [`schema`](Tool::schema) methods return the values set at
/// construction time via [`MockTool::new`]. The
/// [`is_concurrency_safe`](Tool::is_concurrency_safe),
/// [`is_read_only`](Tool::is_read_only), and
/// [`system_prompt`](Tool::system_prompt) methods reflect the flags
/// configured through their respective builder methods.
///
/// # Execution semantics
///
/// The [`call`](Tool::call) future resolves immediately — there is no
/// artificial delay. If your test needs to verify timeout or
/// cancellation behaviour, wrap the mock in a layer that adds delays.
impl Tool for MockTool {
    /// Return the tool name.
    ///
    /// Always returns the string passed to [`MockTool::new`]. The
    /// framework uses this to look up tools in the
    /// [`ToolRegistry`](crate::tool::ToolRegistry) and to correlate
    /// tool-call requests from the model with the right implementation.
    fn name(&self) -> &str {
        &self.name
    }

    /// Return the tool description.
    ///
    /// Always returns the string passed to [`MockTool::new`]. The
    /// description is included in the [`ToolSchema`] sent to the model
    /// so it can decide which tool to invoke.
    fn description(&self) -> &str {
        &self.description
    }

    /// Build the [`ToolSchema`] for this mock tool.
    ///
    /// Combines the `name`, `description`, and `input_schema` fields
    /// into the schema struct the framework sends to the model. The
    /// schema is also used by the [`ToolRegistry`](crate::tool::ToolRegistry)
    /// to describe available tools when calling the API.
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    /// Execute the mock tool, returning the canned result or error.
    ///
    /// - If [`with_error`](MockTool::with_error) was called, returns
    ///   [`ToolError::Execution`] with the result string as the message.
    /// - Otherwise returns [`ToolOutput::text`] containing the result
    ///   string.
    ///
    /// The `_input` and `_context` parameters are ignored — the mock
    /// always returns the preconfigured value. This means you cannot
    /// test input validation through the mock; if you need that, write
    /// a real tool implementation.
    ///
    /// The future resolves immediately (zero delay), making tests fast
    /// and deterministic.
    ///
    /// # Example
    ///
    /// ```rust
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// use loopctl::testing::MockTool;
    /// use loopctl::tool::{Tool, ToolContext};
    /// use serde_json::json;
    ///
    /// let tool = MockTool::new("echo", "Echoes").with_result("pong");
    /// let ctx = ToolContext::default();
    /// let result = tool.call(json!({"msg": "ping"}), &ctx).await;
    /// assert_eq!(result.unwrap().text_content(), "pong");
    /// # });
    /// ```
    fn call(
        &self,
        _input: Value,
        _context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let result = self.result.clone();
        let is_error = self.is_error;
        let delay = self.delay;
        Box::pin(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            if is_error {
                Err(ToolError::Execution(result))
            } else {
                Ok(ToolOutput::text(result))
            }
        })
    }

    /// Return whether this tool is safe to run concurrently.
    ///
    /// Set via [`MockTool::with_concurrency_safe`]. Defaults to `false`.
    /// When `true`, the framework's tool executor may invoke this tool
    /// in parallel with other concurrency-safe tools, improving
    /// throughput for read-only or independent operations.
    fn is_concurrency_safe(&self) -> bool {
        self.is_concurrency_safe
    }

    /// Return whether this tool is read-only (no side effects).
    ///
    /// Set via [`MockTool::with_read_only`]. Defaults to `true` because
    /// most test tools don't need to simulate writes.
    fn is_read_only(&self) -> bool {
        self.is_read_only
    }

    /// Return the optional system prompt for this tool.
    ///
    /// Set via [`MockTool::with_system_prompt`]. Defaults to `None`.
    /// When present, the framework appends this prompt to the agent's
    /// system message, giving the model contextual guidance on how to
    /// use the tool effectively.
    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }
}

/// Create a test user [`Message`] with the given text.
///
/// Shorthand for `Message::user(text)`. Useful when building message
/// histories for tests — avoids importing [`Message`] constructors
/// directly. The returned message has [`Role::User`] and a single
/// [`MessagePart::Text`] variant containing the provided string.
///
/// Most common fixture for constructing the "user says"
/// part of a conversation history.
///
/// # Example
///
/// ```rust
/// use loopctl::testing::test_message;
/// use loopctl::message::Role;
///
/// let msg = test_message("What is 2 + 2?");
/// assert_eq!(msg.role, Role::User);
/// ```
#[must_use]
pub fn test_message(text: &str) -> Message {
    Message::user(text)
}

/// Create a test assistant [`Message`] with the given text.
///
/// Shorthand for `Message::assistant(text)`. Useful for constructing
/// conversation histories where the assistant has already replied.
/// The returned message has [`Role::Assistant`] and a single
/// [`MessagePart::Text`] variant.
///
/// Use this when simulating a multi-turn conversation where the
/// assistant's previous reply is part of the context sent to the model.
///
/// # Example
///
/// ```rust
/// use loopctl::testing::test_assistant_message;
/// use loopctl::message::Role;
///
/// let msg = test_assistant_message("The answer is 4.");
/// assert_eq!(msg.role, Role::Assistant);
/// ```
#[must_use]
pub fn test_assistant_message(text: &str) -> Message {
    Message::assistant(text)
}

/// Create a test assistant [`Message`] containing tool-call content blocks.
///
/// Each tuple in `calls` is `(tool_use_id, tool_name, input_json)`.
/// If `tool_use_id` is an empty string a unique ID of the form
/// `"call_{i}"` is generated automatically (where `i` is the index).
///
/// Message the agent loop produces when the model requests
/// tool execution — use it to simulate the "assistant asked for a tool"
/// step in multi-turn tests. Each tuple becomes a [`MessagePart::ToolCall`]
/// variant in the message's `content` vector.
///
/// # Example
///
/// ```rust
/// use loopctl::testing::test_tool_use_message;
/// use loopctl::message::Role;
/// use serde_json::json;
///
/// let msg = test_tool_use_message(&[
///     ("call_1", "bash", json!({"command": "ls"})),
///     ("", "search", json!({"query": "hello"})),  // auto-ID: "call_1"
/// ]);
/// assert_eq!(msg.role, Role::Assistant);
/// assert_eq!(msg.parts.len(), 2);
/// ```
#[must_use]
pub fn test_tool_use_message(calls: &[(&str, &str, Value)]) -> Message {
    let blocks: Vec<MessagePart> = calls
        .iter()
        .enumerate()
        .map(|(i, (id, name, input))| {
            MessagePart::tool_call(
                if id.is_empty() {
                    format!("call_{i}")
                } else {
                    id.to_string()
                },
                *name,
                input.clone(),
            )
        })
        .collect();
    Message::new(Role::Assistant, blocks)
}

/// Create a test [`SessionConfig`] with sensible defaults.
///
/// The returned config has:
///
/// - `system_prompt` set to `"You are a test assistant."`.
/// - All other fields at their [`Default`] values.
///
/// Useful as a starting point when you don't care about specific
/// configuration values.
///
/// # Example
///
/// ```rust
/// use loopctl::testing::test_config;
///
/// let config = test_config();
/// assert_eq!(config.system_prompt.as_deref(), Some("You are a test assistant."));
/// ```
#[must_use]
pub fn test_config() -> SessionConfig {
    SessionConfig {
        system_prompt: Some("You are a test assistant.".to_string()),
        ..SessionConfig::default()
    }
}

/// Unit tests for the testing module itself.
///
/// These tests verify that the mock implementations and fixture
/// factories behave correctly. They are not part of the public API
/// but serve as a safety net for future changes to this module.
///
/// # Test categories
///
/// - **MockApiClient** — model name, default response, custom text,
///   tool calls, error simulation, multi-turn behaviour, non-streaming.
/// - **MockTool** — success path, error path, registry integration.
/// - **Fixtures** — [`test_message`], [`test_assistant_message`],
///   [`test_tool_use_message`], [`test_config`].
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolRegistry;
    use futures::StreamExt;

    #[test]
    fn test_mock_client_model() {
        let client = MockApiClient::new("test-model");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_mock_client_default_response() {
        let client = MockApiClient::new("test-model");
        let stream = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events: Vec<_> = stream.collect().await;
        assert!(events.len() >= 4);
        assert!(events[0].is_ok());
    }

    #[tokio::test]
    async fn test_mock_client_custom_text() {
        let client = MockApiClient::new("test-model").with_text_response("Custom response");
        let stream = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events: Vec<_> = stream.collect().await;
        let has_text = events.iter().any(|e| {
            if let Ok(StreamEvent::IndexedDelta(delta)) = e
                && let DeltaPart::Text { text } = &delta.delta
            {
                return text == "Custom response";
            }
            false
        });
        assert!(has_text);
    }

    #[tokio::test]
    async fn test_mock_client_tool_call() {
        let client = MockApiClient::new("test-model").with_tool_call(
            "call_1",
            "echo",
            json!({"message": "hi"}),
        );
        let stream = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events: Vec<_> = stream.collect().await;

        let has_tool_use = events.iter().any(|e| {
            if let Ok(StreamEvent::PartStart(start)) = e
                && let Some(MessagePart::ToolCall { name, .. }) = &start.part
            {
                return name == "echo";
            }
            false
        });
        assert!(has_tool_use);

        let has_tool_stop = events.iter().any(|e| {
            if let Ok(StreamEvent::MessageDelta(delta)) = e {
                delta.delta.stop_reason.as_deref() == Some("tool_use")
            } else {
                false
            }
        });
        assert!(has_tool_stop);
    }

    #[tokio::test]
    async fn test_mock_client_error() {
        let client = MockApiClient::new("test-model").with_error("API error");
        let stream = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
    }

    #[tokio::test]
    async fn test_mock_client_multi_turn() {
        let client = MockApiClient::new("test-model").with_responses(vec![
            MockResponse {
                text: "First".to_string(),
                tool_call: None,
                stop_reason: "end_turn".to_string(),
            },
            MockResponse {
                text: "Second".to_string(),
                tool_call: None,
                stop_reason: "end_turn".to_string(),
            },
        ]);

        let stream1 = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events1: Vec<_> = stream1.collect().await;
        let has_first = events1.iter().any(|e| {
            if let Ok(StreamEvent::IndexedDelta(delta)) = e
                && let DeltaPart::Text { text } = &delta.delta
            {
                return text == "First";
            }
            false
        });
        assert!(has_first);

        let stream2 = client.stream_messages(crate::api::StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });
        let events2: Vec<_> = stream2.collect().await;
        let has_second = events2.iter().any(|e| {
            if let Ok(StreamEvent::IndexedDelta(delta)) = e
                && let DeltaPart::Text { text } = &delta.delta
            {
                return text == "Second";
            }
            false
        });
        assert!(has_second);
    }

    #[tokio::test]
    async fn test_mock_client_create_message() {
        let client = MockApiClient::new("test-model").with_text_response("Hello!");
        let result = client
            .create_message(crate::api::StreamRequest {
                messages: vec![Message::user("Hi")],
                system: None,
                tools: None,
            })
            .await;
        assert!(result.is_ok());
        let json = result.unwrap();
        assert_eq!(json["content"][0]["text"], "Hello!");
    }

    #[tokio::test]
    async fn test_mock_tool() {
        let tool = MockTool::new("echo", "Echoes input").with_result("Echo: hello");
        let ctx = ToolContext::default();
        let result = tool.call(json!({"input": "hello"}), &ctx).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text_content(), "Echo: hello");
    }

    #[tokio::test]
    async fn test_mock_tool_error() {
        let tool = MockTool::new("fail", "Always fails")
            .with_result("Something went wrong")
            .with_error();
        let ctx = ToolContext::default();
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_tool_registry() {
        let tool = MockTool::new("echo", "Echoes input").with_concurrency_safe(true);
        let mut registry = ToolRegistry::new();
        registry.register(tool);

        assert!(registry.contains("echo"));
        assert_eq!(registry.len(), 1);

        let t = registry.get("echo").unwrap();
        assert_eq!(t.name(), "echo");
        assert!(t.is_concurrency_safe());
        assert!(t.is_read_only());
    }

    #[test]
    fn test_fixture_test_message() {
        let msg = test_message("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.parts.len(), 1);
    }

    #[test]
    fn test_fixture_test_assistant_message() {
        let msg = test_assistant_message("Hi there");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn test_fixture_test_tool_use_message() {
        let msg = test_tool_use_message(&[("call_1", "bash", json!({"command": "ls"}))]);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.parts.len(), 1);
        assert!(msg.parts[0].is_tool_call());
    }

    #[test]
    fn test_fixture_test_config() {
        let config = test_config();
        assert_eq!(
            config.system_prompt.as_deref(),
            Some("You are a test assistant.")
        );
    }

    #[test]
    fn mock_api_client_set_model() {
        let client = MockApiClient::new("model-a");
        assert_eq!(client.model(), "model-a");

        assert!(client.set_model("model-b"));
        assert_eq!(client.model(), "model-b");
    }

    #[test]
    fn mock_api_client_set_model_rejects_empty() {
        let client = MockApiClient::new("model-a");
        assert!(!client.set_model(""));
        assert!(!client.set_model("   "));
        assert_eq!(client.model(), "model-a");
    }
}
