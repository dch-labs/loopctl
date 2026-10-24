//! API client and error types — interface for LLM provider communication.
//!
//! - **`ApiClient`** — Trait that all LLM provider implementations must satisfy.
//! - **`error`** — `ApiError` and `ErrorCode` for all API/infrastructure errors.
//!
//! See the sub-modules for detailed documentation.

pub mod error;
pub mod probe;
use crate::message::Message;
use crate::stream::{StreamEvent, StreamStopReason, Usage};
use crate::tool::ToolSchema;
use error::ApiError;
use futures::Stream;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// The refusal a client answers when it has no model-listing endpoint.
///
/// One spelling for two sites: the [`ApiClient::list_models`] default
/// and the testing mock's unscripted state, so a consumer's error path
/// — and the [`Unsupported`](probe::ProbeClass::Unsupported)
/// classification hanging off it — reads identically against mock and
/// reality. Both sites pin the rendered message by exact text, so
/// rewording this const moves the two pins together or CI says so.
pub(crate) const UNSUPPORTED_LISTING_MESSAGE: &str =
    "this client does not expose a model listing (list_models)";

/// A request to an LLM provider — the conversation, system prompt, and tools.
///
/// Bundles the three fields that every [`ApiClient`] method takes together,
/// so method signatures stay short (`&self, request, ...`) instead of
/// repeating `messages, system, tools` at every call site.
///
/// Construct via `StreamRequest::new(messages)` and chain
/// `.with_system(...)` / `.with_tools(...)` as needed, or build with
/// struct-literal syntax.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::api::StreamRequest;
///
/// let req = StreamRequest::new(vec![Message::user("hi")])
///     .with_system(Some("be brief".to_string()))
///     .with_tools(Some(vec![search_tool]));
/// let stream = client.stream_messages(req);
/// ```
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// The conversation history sent to the model.
    ///
    /// Owned because the request body is built from owned data — callers clone
    /// the full history each turn (O(n) in messages). This is the only required
    /// field; an empty `Vec` is permitted (a cold-start prompt with no prior
    /// turns).
    pub messages: Vec<Message>,

    /// An optional system prompt prepended above `messages`.
    ///
    /// When `Some`, providers emit it in their native system-prompt slot
    /// (top-level `system` on Anthropic/Gemini, a leading `system` role message
    /// on OpenAI). When `None`, the provider receives no system prompt for the
    /// turn. The value is forwarded verbatim from
    /// [`SessionConfig::system_prompt`](crate::config::SessionConfig::system_prompt);
    /// set it there to drive this field.
    pub system: Option<String>,

    /// Optional tool definitions the model may invoke this turn.
    ///
    /// When `Some`, providers advertise these schemas to the model (OpenAI
    /// `tools`, Anthropic `tools`, Gemini `functionDeclarations`). When `None`,
    /// no tools are advertised and the model cannot issue tool calls. Note the
    /// framework silently suppresses `tools` when a
    /// [`response_format`](crate::structured::RequestOptions) constraint is set
    /// — see the structured-output docs.
    pub tools: Option<Vec<ToolSchema>>,
}

impl StreamRequest {
    /// Create a request carrying `messages` and no system prompt or tools.
    ///
    /// The minimal constructor: only the required conversation history
    /// is set. Chain
    /// [`with_system`](Self::with_system) and
    /// [`with_tools`](Self::with_tools) to populate the optional fields,
    /// or use struct-literal syntax to set every field at once.
    #[must_use]
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            system: None,
            tools: None,
        }
    }

    /// Set the system prompt.
    ///
    /// The field is optional: pass `Some(prompt)` to set it or `None` to leave
    /// it unset (the default). This mirrors the field type directly, so a
    /// caller that already holds an `Option<String>` (e.g. forwarded from
    /// another config field) needs no conversion.
    #[must_use]
    pub fn with_system(mut self, system: Option<String>) -> Self {
        self.system = system;
        self
    }

    /// Set the tool definitions.
    ///
    /// The field is optional: pass `Some(tools)` to attach tool schemas or
    /// `None` to send no tools (the default). Mirrors the field type so an
    /// existing `Option<Vec<ToolSchema>>` maps without conversion.
    #[must_use]
    pub fn with_tools(mut self, tools: Option<Vec<ToolSchema>>) -> Self {
        self.tools = tools;
        self
    }
}

/// A completed, non-streaming LLM response.
///
/// The typed counterpart to a streamed response: instead of a sequence of
/// [`StreamEvent`]s, the complete assistant [`Message`], the reason
/// generation stopped, and the token [`Usage`] are delivered in one shot.
/// Each provider builds this from its own native JSON envelope, so callers
/// never see provider-specific shapes — exactly mirroring how the streaming
/// path emits typed events.
///
/// Produced by [`create_message`](ApiClient::create_message) and
/// [`create_message_with_options`](ApiClient::create_message_with_options).
#[derive(Debug, Clone)]
pub struct NonStreamingResponse {
    /// The fully assembled assistant message.
    ///
    /// Contains the same [`MessagePart`](crate::message::MessagePart) sequence
    /// a stream would accumulate — text blocks, tool calls, etc. Built by the
    /// provider from its native response shape.
    pub message: Message,

    /// Why the model stopped generating.
    ///
    /// Mapped by the provider from its native finish/stop field. Drives the
    /// agent loop's decision to continue to tool execution or end the turn.
    pub stop_reason: StreamStopReason,

    /// Token counts for the request, as reported by the provider.
    ///
    /// Extracted from the provider's native usage field (`usage` on OpenAI and
    /// Anthropic, `usageMetadata` on Gemini). `None` when the provider omits
    /// usage from the response — symmetric with the `Option<Usage>` carried by
    /// the final [`MessageDelta`](StreamEvent::MessageDelta) event on the
    /// streaming path.
    pub usage: Option<Usage>,
}

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
///         request: &StreamRequest,
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
///         request: &StreamRequest,
///     ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
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

    /// Whether this client can honor a tool-call constraint.
    ///
    /// The capability probe behind default construction's constrained
    /// decoding: a loop built with defaults attaches
    /// [`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)
    /// request options only when this returns `true`, so a client that
    /// cannot forward the constraint keeps today's unconstrained request
    /// shape instead of failing every request loudly. Returns `false` by
    /// default; the shipped provider clients that forward constraints
    /// (`openai`, `anthropic`, `gemini`) override it to `true`, the
    /// testing mock ties it to its
    /// [`with_tool_constraint_support`](crate::testing::MockApiClient::with_tool_constraint_support)
    /// flag, and clients that do not encode the constraint on the wire
    /// (Bedrock) keep the honest `false`.
    fn supports_tool_constraints(&self) -> bool {
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

    /// Probe the model's context window in tokens, when the provider
    /// exposes a metadata endpoint.
    ///
    /// `None` when the provider offers nothing to probe, when the
    /// probe fails (timeout, HTTP error, unparseable body), or when
    /// the deployment reports no window — a probe failure is soft: the
    /// caller keeps whatever window it declared and nothing surfaces
    /// as a turn error. The shipped OpenAI-compatible client overrides
    /// this to ask the llama.cpp family's `/props` and Ollama's
    /// `/api/show` documents; the hosted APIs expose no metadata
    /// endpoint and keep the default. Implementations cache the answer
    /// and re-probe only when the model changes, so a caller may
    /// consult this once per run without paying a request per call.
    fn model_context_window(&self) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + '_>> {
        Box::pin(async { None })
    }

    /// List the models this endpoint actually serves.
    ///
    /// The discovery half of the provider-truth surface: what a
    /// health check or a model picker asks before spending a
    /// completion. Each entry carries
    /// [`ModelInfo::metadata`](probe::ModelInfo::metadata) with its
    /// provenance marked — provider-disclosed facts (Gemini's listing
    /// `inputTokenLimit`) win over the crate's
    /// [`curated_model_metadata`](probe::curated_model_metadata)
    /// opinions, which fill only the gaps.
    ///
    /// The default answers [`ApiError::Config`] — "this client kind has no
    /// listing endpoint" — classified as
    /// [`ProbeClass::Unsupported`](probe::ProbeClass::Unsupported), never
    /// an empty list: a client that cannot answer says so instead of
    /// pretending the endpoint had no models. Overrides:
    ///
    /// - `openai`: `GET {base}/models`; on a genuine `404` (no OpenAI-style
    ///   route) falls back to Ollama's `GET /api/tags`, so one client
    ///   covers both spellings of the self-hosted wire. Failures keep the
    ///   providers' own classification ([`ProbeClass`](probe::ProbeClass)
    ///   reads them); if the fallback also fails, the original error
    ///   surfaces with its forensics intact. The `ping` verb shares this
    ///   route and its fallback, so a tags-only server pings as live.
    /// - `anthropic`: `GET {base}/v1/models` with the credential headers;
    ///   `has_more` continuation is not followed — page one is the bound.
    /// - `gemini`: `GET {base}/models`, entries carrying their disclosed
    ///   input-token limit; `nextPageToken` continuation is not followed,
    ///   the same first-page bound.
    /// - `bedrock`: `GET` the SigV4-signed control-plane
    ///   `/foundation-models` — catalog truth, not account entitlement;
    ///   `nextToken` continuation is not followed, the same first-page
    ///   bound.
    ///
    /// The two outcomes stay exact: either a classified failure, or a
    /// listing stating exactly what the endpoint answered within this
    /// surface's documented bound — **the endpoint's own first page**.
    /// Paginated catalogs (Anthropic's `has_more`, Gemini's
    /// `nextPageToken`, Bedrock's `nextToken`) answer page one;
    /// continuation chaining is deliberately unshipped and the bound
    /// carries it, with no flag on the result. What a caller can always
    /// rely on: a success is complete within the bound, and a failure is
    /// never a short list wearing a success's face.
    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<probe::ModelInfo>, ApiError>> + Send + '_>> {
        Box::pin(async { Err(ApiError::config(UNSUPPORTED_LISTING_MESSAGE)) })
    }

    /// Liveness-probe the endpoint without spending a completion.
    ///
    /// The health half of the provider-truth surface: a cheap request to
    /// a metadata route (listing endpoints are the universal choice —
    /// every client that overrides this has one, and each takes its
    /// listing's full route set, fallbacks included) returning the
    /// [`probe::PingReport`] evidence a caller reports instead of a bare
    /// boolean: latency, the rate-limit headers the answer carried, the
    /// request id a support desk could look up.
    ///
    /// The default sends one minimal non-streaming request
    /// ([`create_message`](Self::create_message) with a single "ping"
    /// user message) so every client is probeable even without a
    /// metadata route; callers on a metered hosted API should prefer the
    /// clients whose override uses the listing route instead of tokens.
    /// A probe that failed returns the classified [`ApiError`] — the
    /// auth / throttle / network distinction a health check exists to
    /// make survives the probe path ([`ProbeClass::from_error`]
    /// reads it), and a default-path failure carries the full response
    /// forensics ([`provider_meta`](ApiError::provider_meta)) exactly
    /// like a completion failure.
    ///
    /// [`ProbeClass::from_error`]: probe::ProbeClass::from_error
    fn ping(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<probe::PingReport, ApiError>> + Send + '_>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let request = StreamRequest::new(vec![Message::user("ping")]);
            self.create_message(&request)
                .await
                .map(|_| probe::PingReport::new(started.elapsed()))
        })
    }

    /// What the client knows about one model id — window, modalities,
    /// capabilities, and who said so.
    ///
    /// The metadata half of the discovery surface and the single funnel
    /// a context-window planner or capability-gated request shaping
    /// asks through. The default consults only the crate's
    /// [`curated_model_metadata`](probe::curated_model_metadata) table:
    /// a known family answers with the published values marked
    /// [`MetadataSource::Curated`](probe::MetadataSource::Curated), an
    /// unknown id answers fully unreported — never a plausible zero.
    ///
    /// The `openai` client additionally folds in the wire-probed window
    /// (the same cached probe behind
    /// [`model_context_window`](Self::model_context_window)) when asked
    /// about *its own* model, merged per
    /// [`ModelMetadata::merged`](probe::ModelMetadata::merged) so the
    /// disclosed number wins and the provenance says so. It performs no
    /// request for any other model id, keeping the metadata path
    /// probe-budget-neutral for callers that enumerate listings.
    fn model_metadata(
        &self,
        model: &str,
    ) -> Pin<Box<dyn Future<Output = probe::ModelMetadata> + Send + '_>> {
        let curated = probe::curated_model_metadata(model);
        Box::pin(async move { curated.unwrap_or_else(probe::ModelMetadata::unreported) })
    }

    /// Stream messages from the LLM provider.
    ///
    /// Sends the [`StreamRequest`] (conversation history, optional system
    /// prompt, optional tool definitions) to the LLM, returning a `'static`
    /// [`Stream`] of [`StreamEvent`]s.
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
    /// # Returns
    ///
    /// A pinned, boxed stream of [`Result<StreamEvent, ApiError>`].
    fn stream_messages(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>;

    /// Non-streaming message request (fallback).
    ///
    /// Sends the same [`StreamRequest`] as
    /// [`stream_messages`](ApiClient::stream_messages) but returns a fully
    /// assembled [`NonStreamingResponse`] instead of a stream. Useful for
    /// simple one-shot queries where streaming overhead isn't needed, or as a
    /// fallback when the provider does not support streaming. Each provider
    /// builds the typed [`Message`] from its own native JSON, so no
    /// provider-specific parsing is required at the call site.
    ///
    /// Called by utility code that needs a complete response in one shot,
    /// such as token estimation probes or health checks.
    ///
    /// # Returns
    ///
    /// A pinned, boxed future resolving to the typed
    /// [`NonStreamingResponse`], or an [`ApiError`] if the request fails.
    fn create_message(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>;

    /// Streaming variant that honors [`RequestOptions`](crate::structured::RequestOptions).
    ///
    /// When `options` is empty (the default), delegates to
    /// [`stream_messages`](ApiClient::stream_messages). When `options`
    /// contains fields the client does not support (`response_format`,
    /// `tool_constraint`, a per-request `model` override, or a
    /// thinking-effort level, on a client that has not overridden this
    /// method), yields an
    /// [`ApiError::config`] error as the first stream item — a field the
    /// client cannot forward must fail loudly rather than be silently
    /// dropped: a dropped `model` override would serve a different model
    /// than the one the fallback machinery routed and reported.
    /// `OpenAiClient`, `AnthropicClient`, and `GeminiClient` override
    /// this to honor all four fields.
    fn stream_messages_with_options(
        &self,
        request: &StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        if let Some(err) = unsupported_options_error(&options) {
            return Box::pin(futures::stream::once(async move { Err(err) }));
        }
        self.stream_messages(request)
    }

    /// Non-streaming variant that honors [`RequestOptions`](crate::structured::RequestOptions).
    ///
    /// When `options` is empty (the default), delegates to
    /// [`create_message`](ApiClient::create_message). When `options`
    /// contains fields the client does not support (see
    /// [`stream_messages_with_options`](ApiClient::stream_messages_with_options)
    /// for the field list and the fail-loudly rationale), returns an
    /// [`ApiError::config`] error. This is the primary path for structured
    /// output — a complete JSON document must be present before
    /// deserialization, so callers that need a typed `T` should use this
    /// method and then
    /// [`StructuredOutput::from_value`](crate::structured::StructuredOutput::from_value)
    /// on the extracted payload.
    fn create_message_with_options(
        &self,
        request: &StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        if let Some(err) = unsupported_options_error(&options) {
            return Box::pin(async move { Err(err) });
        }
        self.create_message(request)
    }

    /// Extract the structured-output payload from an assistant message.
    ///
    /// Returns the JSON value that should be fed to
    /// [`StructuredOutput::from_value`](crate::structured::StructuredOutput::from_value).
    /// The default implementation derives this purely from the typed
    /// [`Message`], so it works for every provider without overrides: the
    /// first [`ToolCall`](crate::message::MessagePart::ToolCall) part's `input`
    /// when present, otherwise the joined
    /// [`text_content`](Message::text_content) lenient-parsed as JSON. When the
    /// text is not valid JSON — plain prose, or an empty message — the raw text
    /// string is returned as-is, which downstream deserialization will reject.
    fn extract_structured(&self, message: &Message) -> serde_json::Value {
        if let Some((_, _, input)) = message.tool_call_parts().into_iter().next() {
            return input.clone();
        }
        let text = message.text_content();
        crate::structured::parse_json_lenient(&text).unwrap_or(serde_json::Value::String(text))
    }
}

/// The config error for [`RequestOptions`](crate::structured::RequestOptions)
/// fields a client cannot forward, or `None` when every field is at its
/// default.
///
/// Used by the trait's default `*_with_options` implementations. An
/// unsupported field must fail loudly: silently dropping `model` would
/// serve a different model than the one the fallback machinery routed and
/// reported, silently dropping `tool_constraint` or `response_format`
/// would degrade a constrained request to an unconstrained one with no
/// signal to the caller, and silently dropping `effort` would serve a
/// turn the caller asked to think harder on at the provider's default.
pub(crate) fn unsupported_options_error(
    options: &crate::structured::RequestOptions,
) -> Option<ApiError> {
    if let Some(err) = unsupported_structured_output_error(options) {
        return Some(err);
    }
    if options.model.is_some() {
        return Some(ApiError::config(
            "this client does not support per-request model overrides (model)",
        ));
    }
    if options.effort.is_some() {
        return Some(ApiError::config(
            "this client does not support thinking-effort requests (effort)",
        ));
    }
    None
}

/// The config error for the structured-output option fields a client
/// cannot forward (`response_format`, `tool_constraint`), or `None`.
///
/// The shared core of [`unsupported_options_error`] — the default
/// `*_with_options` implementations' guard.
pub(crate) fn unsupported_structured_output_error(
    options: &crate::structured::RequestOptions,
) -> Option<ApiError> {
    if options.response_format.is_some() {
        return Some(ApiError::config(
            "this client does not support structured output (response_format)",
        ));
    }
    if !matches!(
        options.tool_constraint,
        crate::structured::ToolConstraint::None
    ) {
        return Some(ApiError::config(
            "this client does not support tool-call constraints (tool_constraint)",
        ));
    }
    None
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
            _request: &StreamRequest,
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
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(NonStreamingResponse {
                    message: Message::assistant("Hello!"),
                    stop_reason: StreamStopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
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
        let stream = client.stream_messages(&StreamRequest {
            messages: vec![Message::user("Hi")],
            system: None,
            tools: None,
        });

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
            .create_message(&StreamRequest {
                messages: vec![Message::user("Hi")],
                system: None,
                tools: None,
            })
            .await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.message.parts.is_empty());
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[tokio::test]
    async fn default_with_options_rejects_every_unsupported_field() {
        let client = MockClient::new("primary-model");
        let req = StreamRequest::new(vec![]);

        let model_override =
            crate::structured::RequestOptions::default().with_model("fallback-model");
        let mut stream = client.stream_messages_with_options(&req, model_override.clone());
        let first = stream.next().await;
        drop(stream);
        assert!(
            matches!(&first, Some(Err(err)) if err.to_string().contains("model")),
            "a model override the default impl cannot forward must fail loudly, got: {first:?}"
        );
        let result = client
            .create_message_with_options(&req, model_override)
            .await;
        assert!(
            result.is_err(),
            "the non-streaming default must reject the same override"
        );

        let strict = crate::structured::RequestOptions::default()
            .with_tool_constraint(crate::structured::ToolConstraint::Strict);
        let mut stream = client.stream_messages_with_options(&req, strict.clone());
        let first = stream.next().await;
        drop(stream);
        assert!(
            matches!(&first, Some(Err(err)) if err.to_string().contains("tool_constraint")),
            "a tool constraint the default impl cannot forward must fail loudly, got: {first:?}"
        );
        let result = client.create_message_with_options(&req, strict).await;
        assert!(
            result.is_err(),
            "the non-streaming default must reject the same constraint"
        );

        let format = crate::structured::RequestOptions::default().with_response_format(
            crate::structured::ResponseFormat::new("action", serde_json::json!({})),
        );
        let result = client.create_message_with_options(&req, format).await;
        assert!(
            result.is_err(),
            "the non-streaming default must reject an unsupported response format"
        );

        let effort = crate::structured::RequestOptions::default()
            .with_effort(crate::structured::ThinkingEffort::High);
        let mut stream = client.stream_messages_with_options(&req, effort.clone());
        let first = stream.next().await;
        drop(stream);
        assert!(
            matches!(&first, Some(Err(err)) if err.to_string().contains("effort")),
            "a thinking effort the default impl cannot forward must fail loudly, got: {first:?}"
        );
        let result = client.create_message_with_options(&req, effort).await;
        assert!(
            result.is_err(),
            "the non-streaming default must reject the same effort level"
        );

        let plain = crate::structured::RequestOptions::default();
        let mut stream = client.stream_messages_with_options(&req, plain);
        let first = stream.next().await;
        drop(stream);
        let delegated = first.is_some_and(|item| item.is_ok());
        assert!(
            delegated,
            "default options must still delegate to the plain method"
        );
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn mock_client_serves_the_turn_under_the_routed_model() {
        let mock = crate::testing::MockApiClient::new("primary").with_text_response("hi");
        let opts = crate::structured::RequestOptions::default().with_model("fallback");
        let mut stream =
            mock.stream_messages_with_options(&StreamRequest::new(vec![Message::user("q")]), opts);
        let first = futures::StreamExt::next(&mut stream).await;
        match first {
            Some(Ok(StreamEvent::MessageStart(start))) => assert_eq!(
                start.message.model, "fallback",
                "the mock — like the real providers — must serve the turn under the routed model"
            ),
            other => panic!("expected a clean MessageStart under the override, got {other:?}"),
        }
    }

    #[test]
    fn stream_request_new_defaults() {
        let req = StreamRequest::new(vec![Message::user("hi")]);
        assert_eq!(req.messages.len(), 1);
        assert!(req.system.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn stream_request_system_builder() {
        let req = StreamRequest::new(vec![]).with_system(Some("be brief".to_string()));
        assert_eq!(req.system.as_deref(), Some("be brief"));
        let req = StreamRequest::new(vec![]).with_system(None);
        assert!(req.system.is_none());
    }

    #[test]
    fn stream_request_tools_builder() {
        let tools = vec![crate::tool::ToolSchema {
            tool: "search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let req = StreamRequest::new(vec![]).with_tools(Some(tools));
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        let req = StreamRequest::new(vec![]).with_tools(None);
        assert!(req.tools.is_none());
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

    #[test]
    fn extract_structured_default_returns_tool_call_input() {
        let client = MockClient::new("m");
        let message = Message::new(
            crate::message::Role::Assistant,
            vec![crate::message::MessagePart::tool_call(
                "tc_1",
                "search",
                serde_json::json!({"q": "rust"}),
            )],
        );
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!({"q": "rust"}));
    }

    #[test]
    fn extract_structured_default_returns_first_tool_call_input() {
        let client = MockClient::new("m");
        let message = Message::new(
            crate::message::Role::Assistant,
            vec![
                crate::message::MessagePart::tool_call(
                    "tc_1",
                    "first",
                    serde_json::json!({"order": 1}),
                ),
                crate::message::MessagePart::tool_call(
                    "tc_2",
                    "second",
                    serde_json::json!({"order": 2}),
                ),
            ],
        );
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!({"order": 1}));
    }

    #[test]
    fn extract_structured_default_parses_text_as_json() {
        let client = MockClient::new("m");
        let message = Message::assistant(r#"{"tool": "write", "args": {}}"#);
        let value = client.extract_structured(&message);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_default_lenient_parses_embedded_json() {
        let client = MockClient::new("m");
        let message = Message::assistant(r#"Here is the result: {"answer": 42}"#);
        let value = client.extract_structured(&message);
        assert_eq!(value["answer"], 42);
    }

    #[test]
    fn extract_structured_default_prose_falls_back_to_string() {
        let client = MockClient::new("m");
        let message = Message::assistant("just prose, no json here");
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!("just prose, no json here"));
    }

    #[test]
    fn extract_structured_default_empty_message_falls_back_to_empty_string() {
        let client = MockClient::new("m");
        let message = Message::assistant("");
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!(""));
    }

    #[test]
    fn non_streaming_response_fields_are_accessible() {
        let response = NonStreamingResponse {
            message: Message::assistant("hello"),
            stop_reason: StreamStopReason::EndTurn,
            usage: Some(Usage::new(100, 50)),
        };
        assert_eq!(response.message.text_content(), "hello");
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
        let usage = response.usage.expect("usage present");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens(), 150);
        assert!(!response.message.parts.is_empty());
    }

    #[test]
    fn non_streaming_response_usage_can_be_none() {
        let response = NonStreamingResponse {
            message: Message::assistant("hello"),
            stop_reason: StreamStopReason::EndTurn,
            usage: None,
        };
        assert!(response.usage.is_none());
    }

    #[tokio::test]
    async fn default_with_options_rejects_unsupported_tool_constraint() {
        struct PlainClient;

        impl ApiClient for PlainClient {
            fn model(&self) -> String {
                "plain".to_string()
            }
            fn stream_messages(
                &self,
                _request: &StreamRequest,
            ) -> Pin<
                Box<
                    dyn futures::Stream<Item = Result<StreamEvent, crate::api::error::ApiError>>
                        + Send,
                >,
            > {
                Box::pin(futures::stream::empty())
            }
            fn create_message(
                &self,
                _request: &StreamRequest,
            ) -> Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<NonStreamingResponse, crate::api::error::ApiError>,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    Ok(NonStreamingResponse {
                        message: crate::message::Message::assistant(""),
                        stop_reason: StreamStopReason::EndTurn,
                        usage: None,
                    })
                })
            }
        }

        let options = crate::structured::RequestOptions::new()
            .with_tool_constraint(crate::structured::ToolConstraint::Strict);
        let mut stream =
            PlainClient.stream_messages_with_options(&StreamRequest::new(vec![]), options);
        let first = stream.next().await;
        assert!(
            matches!(&first, Some(Err(crate::api::error::ApiError::Config(_)))),
            "doc: unsupported option fields yield an ApiError::config error as the first item; got {first:?}"
        );
    }

    #[tokio::test]
    async fn a_client_without_a_listing_refuses_loudly_and_metadata_stays_honest() {
        use crate::api::probe::{MetadataSource, ModelMetadata, ProbeClass};

        let client = MockClient::new("test-model");

        let err = client
            .list_models()
            .await
            .expect_err("the trait default must refuse, never fabricate an empty catalog");
        assert_eq!(
            ProbeClass::from_error(&err),
            ProbeClass::Unsupported,
            "a client kind with no listing answers as a capability statement: {err}"
        );
        assert_eq!(
            err.to_string(),
            "Configuration error: this client does not expose a model listing (list_models)",
            "the refusal is the shared message rendered once — the mock's \
             unscripted refusal must read byte-identical to this"
        );

        let known = client.model_metadata("gpt-4o").await;
        assert_eq!(known.source, MetadataSource::Curated);
        assert_eq!(known.context_window, Some(128_000));

        let unknown = client.model_metadata("no-such-family").await;
        assert_eq!(
            unknown,
            ModelMetadata::unreported(),
            "an id outside the curated families answers unreported, never plausible"
        );
    }

    #[tokio::test]
    async fn the_default_ping_answers_through_create_message() {
        let client = MockClient::new("test-model");
        let report = client
            .ping()
            .await
            .expect("the default ping rides create_message, which every client implements");
        assert_eq!(
            report.rate_limit, None,
            "a mock answer carries no forensics; the report must not invent any"
        );
        assert_eq!(report.request_id, None);
        assert!(
            report.latency.as_secs() < 5,
            "the latency is the measured round trip, not a fabricated constant"
        );
    }

    #[tokio::test]
    async fn the_default_ping_forwards_the_wire_failure_untouched() {
        use crate::api::probe::ProbeClass;

        struct FailingClient;

        impl ApiClient for FailingClient {
            fn model(&self) -> String {
                "failing".to_string()
            }
            fn stream_messages(
                &self,
                _request: &StreamRequest,
            ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send>> {
                Box::pin(futures::stream::empty())
            }
            fn create_message(
                &self,
                _request: &StreamRequest,
            ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
            {
                Box::pin(async { Err(ApiError::auth_invalid_key("listed for the probe")) })
            }
        }

        let err = FailingClient
            .ping()
            .await
            .expect_err("a failing completion path must fail the default ping too");
        assert_eq!(
            ProbeClass::from_error(&err),
            ProbeClass::Auth,
            "the auth-vs-throttle-vs-network distinction a health check exists to \
             make must survive the default probe path: {err}"
        );
    }
}
