//! API client and error types — interface for LLM provider communication.
//!
//! - **`ApiClient`** — Trait that all LLM provider implementations must satisfy.
//! - **`error`** — `ApiError` and `ErrorCode` for all API/infrastructure errors.
//!
//! See the sub-modules for detailed documentation.

pub mod error;
use crate::api::error::ApiError;
use crate::message::Message;
use crate::stream::StreamEvent;
use crate::tool::ToolSchema;
use futures::Stream;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Interface for API clients that communicate with LLM providers.
///
/// Defines the contract for both streaming and non-streaming
/// message requests. Implementations handle provider-specific details such
/// as authentication headers, request body formatting, SSE event parsing,
/// and error code mapping.
///
/// The trait is **object-safe**, so clients can be used as
/// [`BoxedApiClient`] (`Box<dyn ApiClient>`) or [`SharedApiClient`]
/// (`Arc<dyn ApiClient>`) without issues.
///
/// # Streaming Contract
///
/// [`stream_messages`](ApiClient::stream_messages) returns a `'static`
/// [`Stream`] of [`StreamEvent`]s. Concrete implementations **must** clone
/// all necessary data from `&self` into the returned future — no borrows
/// of `&self` are captured. This allows the stream to outlive the client
/// reference.
///
/// # Implementors
///
/// - Downstream crates provide implementations for concrete LLM providers.
/// - Mock implementations for testing can be found in test utilities.
///
/// # Example
///
/// ```rust,ignore
/// struct MyProviderClient {
///     api_key: String,
///     model: String,
/// }
///
/// impl ApiClient for MyProviderClient {
///     fn model(&self) -> String {
///         self.model.clone()
///     }
///
///     fn stream_messages(
///         &self,
///         messages: Vec<Message>,
///         system: Option<String>,
///         tools: Option<Vec<ToolSchema>>,
///     ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
///         // Clone data from &self, then build and return a stream
///         let model = self.model.clone();
///         let api_key = self.api_key.clone();
///         // ... provider-specific streaming logic
///         todo!()
///     }
///
///     fn create_message(
///         &self,
///         messages: Vec<Message>,
///         system: Option<String>,
///         tools: Option<Vec<ToolSchema>>,
///     ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>> {
///         // Non-streaming fallback
///         todo!()
///     }
/// }
/// ```
///
/// # See Also
///
/// - [`BoxedApiClient`] — owned, single-threaded type alias.
/// - [`SharedApiClient`] — shared, multi-threaded type alias.
/// - [`ApiError`] — error type for API failures.
/// - [`StreamEvent`] — streaming event types returned by [`ApiClient::stream_messages`].
pub trait ApiClient: Send + Sync {
    /// Get the model identifier for this client.
    ///
    /// Returns the provider-specific model string (e.g.,
    /// `"llm-1"`, `"llm-2"`, `"llm-3"`). Used by the
    /// framework for logging, token estimation, and fallback routing via
    /// [`FallbackManager`](crate::fallback::FallbackManager).
    ///
    /// Called by the framework during initialization and on each turn for
    /// observability purposes.
    fn model(&self) -> String;

    /// Attempt to switch the model at runtime.
    ///
    /// Returns `true` if the client supports hot-swapping and the model
    /// was updated successfully. Returns `false` by default (not supported).
    /// Provider implementations that store their model behind interior
    /// mutability override this to enable
    /// [`BareLoop::switch_model`](crate::engine::BareLoop::switch_model).
    fn set_model(&self, _model: &str) -> bool {
        false
    }

    /// The provider's base URL, used as the per-provider rate-limit bucket key.
    ///
    /// Provider implementations override this to return their configured
    /// endpoint so that [`RateLimiter`](crate::stream::rate_limit::RateLimiter)
    /// can keep an independent budget per distinct endpoint. The default empty
    /// string suits test clients that have no notion of a URL.
    fn base_url(&self) -> String {
        String::new()
    }

    /// Stream messages from the LLM provider.
    ///
    /// Sends the conversation history (`messages`), an optional
    /// `system` prompt, and optional [`ToolSchema`] definitions to the
    /// LLM, returning a `'static` [`Stream`] of [`StreamEvent`]s.
    ///
    /// Called by the agent's turn-processing loop for every LLM
    /// interaction. The stream produces events such as
    /// [`MessageStart`](StreamEvent::MessageStart),
    /// [`PartStart`](StreamEvent::PartStart),
    /// [`IndexedDelta`](StreamEvent::IndexedDelta),
    /// [`MessageDelta`](StreamEvent::MessageDelta), and
    /// [`MessageStop`](StreamEvent::MessageStop).
    ///
    /// # Streaming Lifetime
    ///
    /// The returned stream is `'static` — implementations must clone all
    /// required data from `&self` before constructing the stream. No
    /// references to `&self` may be captured.
    ///
    /// # Parameters
    ///
    /// - `messages` — The conversation history as a [`Vec<Message>`].
    ///   Takes ownership because the request body must be built from owned
    ///   data for the returned `+ '_` future; callers (e.g.
    ///   [`BareLoop`](crate::engine::BareLoop)) clone the full history each
    ///   turn — O(n) in the number of messages.
    /// - `system` — An optional system prompt to prepend.
    /// - `tools` — Optional tool definitions the model may invoke.
    ///
    /// # Returns
    ///
    /// A pinned, boxed stream of [`Result<StreamEvent, ApiError>`].
    fn stream_messages(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>;

    /// Non-streaming message request (fallback).
    ///
    /// Sends the same parameters as [`stream_messages`](ApiClient::stream_messages)
    /// but returns a single [`serde_json::Value`] instead of a stream. Useful
    /// for simple one-shot queries where streaming overhead isn't needed,
    /// or as a fallback when the provider does not support streaming.
    ///
    /// Called by utility code that needs a complete response in one shot,
    /// such as token estimation probes or health checks.
    ///
    /// # Parameters
    ///
    /// - `messages` — The conversation history as a [`Vec<Message>`].
    ///   Takes ownership because the request body must be built from owned
    ///   data for the returned `+ '_` future; callers (e.g.
    ///   [`BareLoop`](crate::engine::BareLoop)) clone the full history each
    ///   turn — O(n) in the number of messages.
    /// - `system` — An optional system prompt to prepend.
    /// - `tools` — Optional tool definitions the model may invoke.
    ///
    /// # Returns
    ///
    /// A pinned, boxed future resolving to the raw JSON response value
    /// from the provider, or an [`ApiError`] if the request fails.
    fn create_message(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>>;

    /// Streaming variant that honors [`RequestOptions`](crate::structured::RequestOptions).
    ///
    /// Default implementation ignores `options` and delegates to
    /// [`stream_messages`](ApiClient::stream_messages) — so a provider that
    /// does not support structured output still compiles and behaves as
    /// before. `OpenAiClient`, `AnthropicClient`, and `GeminiClient` override
    /// this to inject the schema.
    fn stream_messages_with_options(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let _ = options;
        self.stream_messages(messages, system, tools)
    }

    /// Non-streaming variant that honors [`RequestOptions`](crate::structured::RequestOptions).
    ///
    /// Same default-delegates story as
    /// [`stream_messages_with_options`](ApiClient::stream_messages_with_options).
    /// This is the primary path for structured output — a complete JSON
    /// document must be present before deserialization, so callers that need
    /// a typed `T` should use `create_message_with_options` and then
    /// [`StructuredOutput::from_value`](crate::structured::StructuredOutput::from_value)
    /// on the extracted payload.
    fn create_message_with_options(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>> {
        let _ = options;
        self.create_message(messages, system, tools)
    }

    /// Extract the structured-output payload from a provider response.
    ///
    /// Each provider knows its own response envelope shape. This method pulls
    /// the inner JSON value that should be fed to
    /// [`StructuredOutput::from_value`](crate::structured::StructuredOutput::from_value).
    /// The default implementation returns the raw value as-is (for mock
    /// clients and custom providers that already return the structured value
    /// without an envelope). `OpenAiClient`, `AnthropicClient`, and
    /// `GeminiClient` override this to navigate their respective envelopes.
    fn extract_structured(&self, raw: &serde_json::Value) -> serde_json::Value {
        raw.clone()
    }
}

/// Owned, single-threaded API client handle.
///
/// Use when you need owned, single-owner access to a provider client.
/// The underlying client remains `Send + Sync` (required by the trait),
/// so the box can be moved across thread boundaries — but it cannot be
/// cloned or shared.
///
/// Ideal for test fixtures and single-agent runners.
///
/// For shared ownership, see [`SharedApiClient`].
///
/// # Example
///
/// ```rust,ignore
/// let client: BoxedApiClient = Box::new(MyProviderClient::new(api_key));
/// assert_eq!(client.model(), "llm-1");
/// ```
pub type BoxedApiClient = Box<dyn ApiClient>;

/// Shared, reference-counted API client handle.
///
/// Use when multiple tasks or threads need concurrent access to the
/// same provider client — for example, in a multi-agent system or when
/// sharing a client between an agent and a background metrics collector.
///
/// For single-owner usage, see [`BoxedApiClient`].
///
/// # Example
///
/// ```rust,ignore
/// let client: SharedApiClient = Arc::new(MyProviderClient::new(api_key));
/// let agent_client = client.clone();
/// let metrics_client = client.clone();
/// ```
pub type SharedApiClient = Arc<dyn ApiClient>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Usage;
    use futures::StreamExt;

    struct MockClient {
        model_name: String,
    }

    impl MockClient {
        fn new(model: &str) -> Self {
            Self {
                model_name: model.to_string(),
            }
        }
    }

    impl ApiClient for MockClient {
        fn model(&self) -> String {
            self.model_name.clone()
        }

        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            // Return a simple stream with one event
            let events: Vec<Result<StreamEvent, ApiError>> = vec![
                Ok(StreamEvent::MessageStart(crate::stream::MessageStart {
                    message: crate::stream::MessageMetadata {
                        id: "msg_test".to_string(),
                        role: "assistant".to_string(),
                        model: self.model_name.clone(),
                    },
                })),
                Ok(StreamEvent::PartStart(crate::stream::PartStart {
                    index: 0,
                    part: Some(crate::message::MessagePart::text("Hello!")),
                })),
                Ok(StreamEvent::MessageDelta(crate::stream::MessageDelta {
                    delta: crate::stream::MessageDeltaPayload {
                        stop_reason: Some("end_turn".to_string()),
                    },
                    usage: Some(Usage::new(10, 5)),
                })),
                Ok(StreamEvent::MessageStop),
            ];
            Box::pin(futures::stream::iter(events))
        }

        fn create_message(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(serde_json::json!({
                    "content": [{"type": "text", "text": "Hello!"}]
                }))
            })
        }
    }

    #[test]
    fn test_mock_client_model() {
        let client = MockClient::new("test-model");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_mock_client_stream() {
        let client = MockClient::new("test-model");
        let stream = client.stream_messages(vec![Message::user("Hi")], None, None);

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 4);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::MessageStart(_)
        ));
        assert!(matches!(
            events[1].as_ref().unwrap(),
            StreamEvent::PartStart(_)
        ));
        assert!(matches!(
            events[2].as_ref().unwrap(),
            StreamEvent::MessageDelta(_)
        ));
        assert!(matches!(
            events[3].as_ref().unwrap(),
            StreamEvent::MessageStop
        ));
    }

    #[tokio::test]
    async fn test_mock_client_create_message() {
        let client = MockClient::new("test-model");
        let result = client
            .create_message(vec![Message::user("Hi")], None, None)
            .await;
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.get("content").is_some());
    }

    #[test]
    fn test_boxed_client() {
        let client: BoxedApiClient = Box::new(MockClient::new("boxed"));
        assert_eq!(client.model(), "boxed");
    }

    #[test]
    fn test_shared_client() {
        let client: SharedApiClient = Arc::new(MockClient::new("shared"));
        assert_eq!(client.model(), "shared");
    }

    #[test]
    fn default_set_model_returns_false() {
        // MockClient does not override set_model, so the default impl
        // should return false (unsupported).
        let client = MockClient::new("test-model");
        assert!(!client.set_model("other-model"));
        assert_eq!(client.model(), "test-model");
    }
}
