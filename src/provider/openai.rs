//! OpenAI-compatible API client.
//!
//! Works with any provider that implements the OpenAI Chat Completions
//! API with streaming: OpenAI itself, `DeepSeek`, `Grok`, Ollama (via
//! the `ollama()` constructor in the parent module), vLLM, LM Studio, etc.
//!
//! # Construction
//!
//! ```rust,ignore
//! use loopctl::provider::OpenAiClient;
//!
//! // From environment (OPENAI_API_KEY, optional OPENAI_BASE_URL):
//! let client = OpenAiClient::from_env()?;
//!
//! // Explicit:
//! let client = OpenAiClient::builder()
//!     .with_api_key("sk-...")
//!     .with_base_url("https://api.deepseek.com/v1")
//!     .with_model("deepseek-chat")
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::stream::Stream;
use serde::Deserialize;
use serde_json::Value;

use crate::api::ApiClient;
use crate::api::error::ApiError;
use crate::message::{Message, MessagePart, Role};
use crate::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamEvent, StreamStopReason, Usage,
};
use crate::structured::ToolConstraint;
use crate::structured::tighten_json_schema;
use crate::tool::ToolSchema;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o";
const SSE_DONE: &str = "[DONE]";
const TEXT_PART_INDEX: usize = 0;
const THINKING_PART_INDEX: usize = 1;

/// An OpenAI-compatible chat completions client with streaming support.
///
/// Implements [`ApiClient`] by translating between the framework's
/// [`StreamEvent`] protocol and the OpenAI Chat Completions SSE format.
///
/// Works with any OpenAI-compatible endpoint. Use a custom `base_url`
/// to target `DeepSeek`, `Grok`, Ollama, `vLLM`, or other compatible APIs.
pub struct OpenAiClient {
    /// The underlying HTTP client (connection-pooled `reqwest::Client`).
    ///
    /// Created once at build time with the configured timeouts; reused
    /// across all requests for connection pooling.
    http: reqwest::Client,

    /// The API key used for authentication.
    ///
    /// Sent as the `Authorization: Bearer <key>` header on every request.
    /// Set via [`OpenAiClientBuilder::api_key`].
    api_key: String,

    /// The base URL for API requests.
    ///
    /// The chat-completions endpoint is `{base_url}/chat/completions`.
    /// Defaults to `https://api.openai.com/v1`; override for
    /// `DeepSeek`, `Grok`, `Ollama`, `vLLM`, or other compatible endpoints.
    base_url: String,

    /// The current model identifier, stored behind a mutex for runtime
    /// hot-swapping.
    ///
    /// Changed via [`ApiClient::set_model`] when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips to a
    /// fallback model.
    model: std::sync::Mutex<String>,

    /// Whether to request `stream_options.include_usage` on streaming requests.
    ///
    /// Defaults to `true` (real OpenAI supports it). Disabled for
    /// OpenAI-compatible servers that reject the parameter via
    /// [`OpenAiClientBuilder::with_stream_usage`].
    stream_usage: bool,
}

impl OpenAiClient {
    /// Create a builder for configuring an [`OpenAiClient`].
    ///
    /// Returns an [`OpenAiClientBuilder`] with sensible defaults. The only
    /// required field is `api_key`; everything else has a production-ready
    /// default. Call `.with_api_key(...).build()` to finish, or chain additional
    /// setters for custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::provider::OpenAiClient;
    ///
    /// let client = OpenAiClient::builder()
    ///     .with_api_key("sk-...")
    ///     .with_model("gpt-4o")
    ///     .build()
    /// .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> OpenAiClientBuilder {
        OpenAiClientBuilder::default()
    }

    /// Create a client from environment variables.
    ///
    /// Reads the following variables:
    ///
    /// - `OPENAI_API_KEY` (or `API_KEY`) — **required**. The API key for
    ///   authentication.
    /// - `OPENAI_BASE_URL` (or `BASE_URL`) — optional. Defaults to
    ///   `https://api.openai.com/v1`. Override for OpenAI-compatible endpoints.
    /// - `OPENAI_MODEL` (or `MODEL`) — optional. Defaults to `gpt-4o`.
    ///
    /// This is a convenience constructor that delegates to
    /// [`builder`](Self::builder) with the env vars as setter arguments.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key is found.
    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .map_err(|_| ApiError::auth_invalid_key("OPENAI_API_KEY not set"))?;

        let base_url = std::env::var("OPENAI_BASE_URL")
            .or_else(|_| std::env::var("BASE_URL"))
            .unwrap_or_else(|_| DEFAULT_BASE_URL.into());

        let model = std::env::var("OPENAI_MODEL")
            .or_else(|_| std::env::var("MODEL"))
            .unwrap_or_else(|_| DEFAULT_MODEL.into());

        Self::builder()
            .with_api_key(api_key)
            .with_base_url(base_url)
            .with_model(model)
            .build()
    }

    /// Build the full URL for the OpenAI chat-completions endpoint.
    ///
    /// Appends `/chat/completions` to the client's `base_url`. All four
    /// `ApiClient` methods (`stream_messages`, `create_message`, and their
    /// `*_with_options` variants) POST to this URL.
    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Build a typed [`NonStreamingResponse`] from OpenAI's native JSON.
    ///
    /// Reads `choices[0].message` into [`MessagePart`]s: the `content`
    /// string becomes a [`MessagePart::Text`] part (skipped when `null`),
    /// and each entry in `tool_calls` becomes a [`MessagePart::ToolCall`]
    /// with its `function.arguments` JSON-string parsed into a [`Value`].
    /// Maps `choices[0].finish_reason` to a [`StreamStopReason`] using the
    /// same mapping the streaming emitter applies (`"tool_calls"` →
    /// `ToolCall`, `"length"` → `MaxTokens`, anything else via
    /// [`StreamStopReason::from_api_str`], defaulting to `EndTurn`). Reads
    /// `usage.prompt_tokens` / `usage.completion_tokens` into [`Usage`],
    /// returning `None` when the object is absent or all-zero. Missing or
    /// empty `function.arguments` default to `{}`; non-empty arguments that
    /// fail to parse as JSON return an error.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if a tool call's `function.arguments` is present,
    /// non-empty, and not valid JSON.
    fn build_response(raw: &Value) -> Result<crate::api::NonStreamingResponse, ApiError> {
        let choice = raw.get("choices").and_then(|c| c.get(0));
        let message = choice.and_then(|c| c.get("message"));
        let mut parts: Vec<MessagePart> = Vec::new();
        if let Some(msg) = message {
            if let Some(text) = msg.get("content").and_then(|t| t.as_str()) {
                parts.push(MessagePart::text(text));
            }
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tool_calls {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let function = tc.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let input = match function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                    {
                        None | Some("") => serde_json::json!({}),
                        Some(s) => serde_json::from_str::<Value>(s).map_err(|e| {
                            ApiError::http(format!("tool_call arguments is not valid JSON: {e}"))
                        })?,
                    };
                    parts.push(MessagePart::tool_call(id, name, input));
                }
            }
        }
        let reason = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(|r| r.as_str())
            .unwrap_or("stop");
        let stop_reason = match reason {
            "tool_calls" => StreamStopReason::ToolCall,
            "length" => StreamStopReason::MaxTokens,
            other => StreamStopReason::from_api_str(other).unwrap_or(StreamStopReason::EndTurn),
        };
        let usage = raw
            .get("usage")
            .and_then(|u| OpenAiUsage::deserialize(u).ok())
            .map(|u| Usage::from(&u))
            .filter(|u| u.input_tokens > 0 || u.output_tokens > 0);
        Ok(crate::api::NonStreamingResponse {
            message: Message::new(Role::Assistant, parts),
            stop_reason,
            usage,
        })
    }

    /// Send a POST request to the chat-completions endpoint.
    ///
    /// Shared by both [`ApiClient::stream_messages`] and
    /// [`ApiClient::create_message`]. Delegates to
    /// [`post_json_checked`](super::post_json_checked), which classifies
    /// non-success responses (auth rejections, rate limits with their
    /// server-advised delay) into structured [`ApiError`] variants.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the request fails or the server
    /// responds with a non-success status code.
    async fn post_completions(
        http: &reqwest::Client,
        url: &str,
        api_key: &str,
        body: &Value,
    ) -> Result<reqwest::Response, ApiError> {
        let mut bearer = reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| ApiError::auth_invalid_key(format!("invalid bearer token: {e}")))?;
        bearer.set_sensitive(true);
        super::post_json_checked(http, url, &[(reqwest::header::AUTHORIZATION, bearer)], body).await
    }
}

impl ApiClient for OpenAiClient {
    fn model(&self) -> String {
        crate::error::recover_guard(self.model.lock()).clone()
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn set_model(&self, model: &str) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        *crate::error::recover_guard(self.model.lock()) = model.to_string();
        true
    }

    fn stream_messages(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let body = RequestBody::build(
            &model,
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
        )
        .with_stream_usage(self.stream_usage);
        let url = self.completions_url();
        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_completions(&http, &url, &api_key, &body.to_json(true)).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_openai_data().await? {
                let Some(chunk) = OpenAiChunk::parse(&data) else {
                    continue;
                };
                emitter.process_chunk(&chunk);
                for ev in emitter.drain() {
                    yield ev;
                }
            }

            for ev in emitter.finish() {
                yield ev;
            }
        })
    }

    fn create_message(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let body = RequestBody::build(
            &model,
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
        );
        let url = self.completions_url();

        Box::pin(async move {
            let resp =
                Self::post_completions(&self.http, &url, &self.api_key, &body.to_json(false))
                    .await?;
            let resp = super::read_bounded_body(resp).await?;
            let raw = serde_json::from_slice::<Value>(&resp)
                .map_err(|e| ApiError::http(e.to_string()))?;
            Self::build_response(&raw)
        })
    }

    fn stream_messages_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let rf = options.response_format.as_ref();
        let body = RequestBody::build(
            &model,
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            rf,
            &options.tool_constraint,
        )
        .with_stream_usage(self.stream_usage);
        let url = self.completions_url();
        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_completions(&http, &url, &api_key, &body.to_json(true)).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_openai_data().await? {
                let Some(chunk) = OpenAiChunk::parse(&data) else {
                    continue;
                };
                emitter.process_chunk(&chunk);
                for ev in emitter.drain() {
                    yield ev;
                }
            }

            for ev in emitter.finish() {
                yield ev;
            }
        })
    }

    fn create_message_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let rf = options.response_format.as_ref();
        let body = RequestBody::build(
            &model,
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            rf,
            &options.tool_constraint,
        );
        let url = self.completions_url();

        Box::pin(async move {
            let resp =
                Self::post_completions(&self.http, &url, &self.api_key, &body.to_json(false))
                    .await?;
            let resp = super::read_bounded_body(resp).await?;
            let raw = serde_json::from_slice::<Value>(&resp)
                .map_err(|e| ApiError::http(e.to_string()))?;
            Self::build_response(&raw)
        })
    }
}

/// Builder for [`OpenAiClient`].
///
/// Created via [`OpenAiClientBuilder::default`] or
/// [`OpenAiClient::builder`]. All fields have sensible defaults except
/// `api_key`, which must be set before [`build`](Self::build).
pub struct OpenAiClientBuilder {
    /// The API key for authentication (required).
    ///
    /// Must be set before building. Sent as the `Authorization: Bearer`
    /// header on every request.
    api_key: Option<String>,

    /// The base URL for API requests.
    ///
    /// Defaults to `https://api.openai.com/v1`. Override for
    /// `DeepSeek`, `Grok`, `Ollama`, `vLLM`, or other OpenAI-compatible endpoints.
    base_url: String,

    /// The default model identifier.
    ///
    /// Can be changed at runtime via [`OpenAiClient::set_model`].
    model: String,

    /// Shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// Holds the timeout, connection-pool, and TCP knobs that apply to the
    /// internally-built `reqwest::Client`, or an externally-supplied client
    /// injected via [`with_http_client`](Self::with_http_client).
    http: super::HttpClientConfig,

    /// Whether to request `stream_options.include_usage` on streaming requests.
    ///
    /// Defaults to `true`. Disable for OpenAI-compatible servers that reject
    /// the parameter (older Ollama, some self-hosted deployments). Read by
    /// [`build`](Self::build) and stored on [`OpenAiClient`].
    stream_usage: bool,
}

impl Default for OpenAiClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            http: super::HttpClientConfig::default(),
            stream_usage: true,
        }
    }
}

impl OpenAiClientBuilder {
    /// Set the API key for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is not set.
    /// The key is sent as the `Authorization: Bearer <key>` header on every
    /// request.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL for API requests.
    ///
    /// Defaults to `https://api.openai.com/v1`. Override when targeting an
    /// OpenAI-compatible endpoint (e.g. `https://api.deepseek.com/v1`,
    /// Trailing `/` separators are trimmed, so joined request paths never
    /// contain `//` — a `…/v1/` base behaves identically to `…/v1`.
    /// `http://localhost:11434/v1` for Ollama, or a vLLM server).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Set the default model identifier.
    ///
    /// The model string is sent as the `model` field on every request. Can be
    /// changed at runtime via [`OpenAiClient::set_model`] (e.g. when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the total request timeout (connect + response + body).
    ///
    /// Defaults to 120 seconds. Ignored when a client was supplied via
    /// [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http = self.http.with_timeout(timeout);
        self
    }

    /// Set the TCP connection establishment timeout.
    ///
    /// Defaults to 10 seconds. Ignored when a client was supplied via
    /// [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.http = self.http.with_connect_timeout(timeout);
        self
    }

    /// Inject a pre-built, shared `reqwest::Client`.
    ///
    /// When set, the client's connection pool is shared with every other
    /// provider built from the same handle, and the pool/TCP knobs are
    /// ignored. Configure timeouts on the injected client, not here.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = self.http.with_http_client(client);
        self
    }

    /// Control whether streaming requests include `stream_options.include_usage`.
    ///
    /// Defaults to `true` — real OpenAI, `DeepSeek`, and Grok support it and
    /// send a final usage chunk. Pass `false` for OpenAI-compatible servers
    /// that reject the parameter with a validation error (older Ollama, some
    /// self-hosted vLLM/LM Studio deployments). When disabled, streamed turns
    /// report `usage: None` instead of real token counts.
    ///
    /// Ignored on non-streaming requests.
    #[must_use]
    pub fn with_stream_usage(mut self, enabled: bool) -> Self {
        self.stream_usage = enabled;
        self
    }

    /// Set the maximum idle connections kept alive per host.
    ///
    /// Defaults to reqwest's built-in default (unlimited). Ignored when a
    /// client was supplied via [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_pool_max_idle_per_host(mut self, n: usize) -> Self {
        self.http = self.http.with_pool_max_idle_per_host(n);
        self
    }

    /// Set how long an idle connection stays in the pool before being closed.
    ///
    /// Defaults to reqwest's built-in default (90s). Ignored when a client
    /// was supplied via [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_pool_idle_timeout(mut self, d: Duration) -> Self {
        self.http = self.http.with_pool_idle_timeout(d);
        self
    }

    /// Set the OS-level TCP keepalive interval.
    ///
    /// Defaults to disabled (reqwest default). Ignored when a client was
    /// supplied via [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_tcp_keepalive(mut self, d: Duration) -> Self {
        self.http = self.http.with_tcp_keepalive(d);
        self
    }

    /// Control whether `TCP_NODELAY` is set on connections.
    ///
    /// Defaults to `true` — SSE streaming benefits from disabling Nagle's
    /// algorithm. Pass `false` to re-enable it. Ignored when a client was
    /// supplied via [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub fn with_tcp_nodelay(mut self, enabled: bool) -> Self {
        self.http = self.http.with_tcp_nodelay(enabled);
        self
    }

    /// Build the client.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key was set.
    pub fn build(self) -> Result<OpenAiClient, ApiError> {
        let api_key = self
            .api_key
            .ok_or_else(|| ApiError::auth_invalid_key("API key not provided"))?;
        let http = self.http.build()?;

        Ok(OpenAiClient {
            http,
            api_key,
            base_url: self.base_url,
            model: std::sync::Mutex::new(self.model),
            stream_usage: self.stream_usage,
        })
    }
}

/// A built OpenAI Chat Completions request body.
///
/// Separating construction from serialization lets us reuse the same
/// body for both streaming and non-streaming requests, toggling only
/// the `stream` flag via [`to_json`](Self::to_json).
struct RequestBody {
    /// The model identifier sent as the `model` field on every request.
    ///
    /// Copied from [`OpenAiClient`]'s model field (which is mutable at
    /// runtime via [`ApiClient::set_model`]). Each request carries it so
    /// the provider knows which model to invoke.
    model: String,

    /// The conversation messages converted to OpenAI's JSON format.
    ///
    /// Each message is an object with `role` (`"system"`, `"user"`, or
    /// `"assistant"`) and `content` (text string, tool-call array, or tool
    /// result). Built by [`RequestBody::build`] from the framework's
    /// [`Message`] list via [`convert_message`].
    messages: Vec<Value>,

    /// The registered tools in OpenAI function-calling format, or `None`
    /// when no tools are registered or when `response_format` is set
    /// (mutual exclusion).
    tools: Option<Vec<Value>>,

    /// The structured-output `response_format` JSON object, or `None`
    /// when structured output is not requested. When `Some`, the field is
    /// emitted as `response_format: { type: "json_schema", ... }` and
    /// `tools` is suppressed.
    response_format: Option<Value>,

    /// Grammar to pass through as `guided_json` for vLLM-style grammar-aware
    /// samplers. `None` unless the caller set
    /// [`ToolConstraint::Grammar`](crate::structured::ToolConstraint::Grammar)
    /// and no `response_format` was set. Stored as a string so the body is
    /// serializable without re-borrowing the trait object.
    guided_json: Option<String>,

    /// Whether to request `stream_options.include_usage` when streaming.
    ///
    /// Defaults to `true` (real OpenAI supports it). Disabled for providers
    /// that reject the parameter (older Ollama, some self-hosted servers)
    /// via [`with_stream_usage`](Self::with_stream_usage). Ignored on
    /// non-streaming requests.
    stream_usage: bool,
}

impl RequestBody {
    /// Translate the framework's [`Message`] list into the OpenAI Chat
    /// Completions request shape.
    ///
    /// Converts messages to OpenAI's `role`/`content` JSON format, wraps
    /// tool schemas in the `function` envelope, and applies the
    /// tool-call constraint:
    /// - When `response_format` is set, suppresses `tools` (OpenAI's
    ///   structured output and free-form tool-calling are mutually
    ///   exclusive) and emits the `response_format: json_schema` object.
    ///   Any `tool_constraint` is ignored in this case.
    /// - Otherwise, when `tool_constraint` is `Strict`, wraps each tool
    ///   via [`convert_tools_strict`] (tightens the schema and sets
    ///   `strict: true`).
    /// - When `tool_constraint` is `Grammar(g)`, captures the grammar
    ///   string for `guided_json` emission in [`to_json`](Self::to_json).
    fn build(
        model: &str,
        messages: &[Message],
        system: Option<&str>,
        tools: Option<&[ToolSchema]>,
        response_format: Option<&crate::structured::ResponseFormat>,
        tool_constraint: &ToolConstraint,
    ) -> Self {
        let tools = tools.filter(|t| !t.is_empty());
        let mut msgs = Vec::with_capacity(messages.len().saturating_add(1));

        if let Some(sys) = system {
            msgs.push(serde_json::json!({ "role": "system", "content": sys }));
        }

        for m in messages {
            msgs.extend(convert_message(m));
        }

        let (tools, guided_json) = if response_format.is_some() {
            (None, None)
        } else {
            match tool_constraint {
                ToolConstraint::None => (tools.map(convert_tools), None),
                ToolConstraint::Strict => (tools.map(convert_tools_strict), None),
                #[cfg(feature = "grammar")]
                ToolConstraint::Grammar(provider) => {
                    let has_tools = tools.is_some_and(|t| !t.is_empty());
                    (
                        tools.map(convert_tools),
                        has_tools.then(|| provider.grammar().to_string()),
                    )
                }
            }
        };

        let rf = response_format.map(|rf| {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": rf.name,
                    "schema": rf.schema,
                    "strict": rf.strict
                }
            })
        });

        Self {
            model: model.into(),
            messages: msgs,
            tools,
            response_format: rf,
            guided_json,
            stream_usage: true,
        }
    }

    /// Control whether streaming requests include `stream_options.include_usage`.
    ///
    /// Defaults to `true` after [`build`](Self::build). Pass `false` for
    /// OpenAI-compatible servers that reject the parameter (older Ollama, some
    /// self-hosted deployments). The flag is read by [`to_json`](Self::to_json)
    /// and ignored on non-streaming requests.
    #[must_use]
    fn with_stream_usage(mut self, enabled: bool) -> Self {
        self.stream_usage = enabled;
        self
    }

    /// Serialize to a [`serde_json::Value`] for the HTTP request body.
    ///
    /// Emits `model`, `messages`, `stream` (toggled by the parameter),
    /// and `tools`. When streaming and [`stream_usage`](Self::stream_usage)
    /// is enabled, sets `stream_options.include_usage` so the server appends a
    /// final usage chunk. When `response_format` is set, appends the
    /// `response_format` key; otherwise omits it entirely (not `null`). When a
    /// grammar was captured, appends `guided_json`.
    fn to_json(&self, stream: bool) -> Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
        });
        if let Some(obj) = body.as_object_mut() {
            if stream && self.stream_usage {
                obj.insert(
                    "stream_options".to_string(),
                    serde_json::json!({"include_usage": true}),
                );
            }
            if let Some(tools) = &self.tools {
                obj.insert("tools".to_string(), Value::Array(tools.clone()));
            }
            if let Some(rf) = &self.response_format {
                obj.insert("response_format".to_string(), rf.clone());
            }
            if let Some(grammar) = &self.guided_json {
                obj.insert("guided_json".to_string(), Value::String(grammar.clone()));
            }
        }
        body
    }
}

/// Convert a single framework [`Message`] into the OpenAI JSON shape.
///
/// OpenAI expects assistant messages with `tool_calls` to carry them in
/// a dedicated array, tool results to use the `tool` role, and plain
/// text to use a simple `{role, content}` pair. A single loopctl message
/// with multiple tool-result parts expands to one OpenAI `tool` message per
/// result, so the return is a vector; text parts alongside tool results are
/// preserved as a trailing `user` message after the `tool` messages.
fn convert_message(m: &Message) -> Vec<Value> {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    };
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();

    for p in &m.parts {
        match p {
            MessagePart::Text { text } => text_parts.push(text.as_str()),
            MessagePart::ToolCall { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input_to_string(input),
                    }
                }));
            }
            MessagePart::ToolResult {
                call_id, output, ..
            } => {
                tool_results.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output.to_string(),
                }));
            }
            MessagePart::Image { .. } => {} // not supported in this path
        }
    }

    if !tool_calls.is_empty() {
        vec![build_assistant_message(role, &tool_calls, &text_parts)]
    } else if !tool_results.is_empty() {
        if !text_parts.is_empty() {
            tool_results.push(serde_json::json!({
                "role": "user",
                "content": text_parts.join(""),
            }));
        }
        tool_results
    } else {
        vec![serde_json::json!({ "role": role, "content": text_parts.join("") })]
    }
}

/// Build an assistant message JSON object that includes `tool_calls`.
///
/// Constructs the OpenAI-shaped `{ role, content, tool_calls }` object from
/// the accumulated text parts and tool-call entries. When there is no text
/// (pure tool-call turn), `content` is set to `null` — OpenAI's convention
/// for tool-call-only assistant messages. The `tool_calls` array carries the
/// converted tool-call entries produced by [`convert_message`].
fn build_assistant_message(role: &str, tool_calls: &[Value], text_parts: &[&str]) -> Value {
    let text = text_parts.join("");
    let content = if text.is_empty() {
        Value::Null
    } else {
        Value::String(text)
    };
    serde_json::json!({
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
    })
}

/// Convert framework tool schemas into the OpenAI `tools` array shape.
///
/// Each [`ToolSchema`] becomes a JSON object with `type: "function"` and a
/// nested `function` object carrying `name`, `description`, and `parameters`
/// (the framework's `input_schema`). When structured output is active
/// (`response_format` set), this function is not called — `tools` is
/// suppressed entirely.
fn convert_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.tool,
                    "description": &t.description,
                    "parameters": t.input_schema.clone(),
                }
            })
        })
        .collect()
}

/// Convert framework tool schemas into the OpenAI `tools` array shape with
/// strict mode enabled.
///
/// Like [`convert_tools`], but tightens each tool's `parameters`
/// (recursive `additionalProperties: false` and full `required`) and sets
/// `strict: true` on each `function` entry. This is the
/// [`ToolConstraint::Strict`] path: OpenAI rejects a strict tool whose
/// schema isn't already tightened, so the tightening is done here rather
/// than left to the tool author.
fn convert_tools_strict(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let parameters = tighten_json_schema(&t.input_schema);
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.tool,
                    "description": &t.description,
                    "parameters": parameters,
                    "strict": true,
                }
            })
        })
        .collect()
}

/// Serialize a JSON value to a compact string for the OpenAI `arguments` field.
///
/// OpenAI expects `arguments` to be a string containing JSON, so a raw
/// [`Value`] must be stringified. If the value is already a string we
/// pass it through unchanged.
fn input_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

use super::sse::SseReader;

impl SseReader {
    /// Extract the next SSE `data:` payload, blocking until one is
    /// available or the stream ends.
    ///
    /// Returns `Ok(None)` at end-of-stream (including `[DONE]`).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails.
    async fn next_openai_data(&mut self) -> Result<Option<String>, ApiError> {
        loop {
            while let Some(line) = self.take_line()? {
                let Some(data) = super::sse_data_payload(&line) else {
                    continue;
                };
                if data == SSE_DONE {
                    return Ok(None);
                }
                return Ok(Some(data.into()));
            }
            if self.next_chunk().await?.is_none() {
                return Ok(None);
            }
        }
    }
}

/// A single SSE chunk from the OpenAI streaming API.
///
/// Each `data:` line in a streamed Chat Completions response deserializes
/// into one of these. The chunk carries the message identity, the model
/// that produced it, a list of [`OpenAiChoice`] deltas that the
/// [`StreamEmitter`] assembles into [`StreamEvent`]s, and — on the final
/// chunk when `stream_options.include_usage` is set — the cumulative
/// [`OpenAiUsage`] for the entire request.
#[derive(Deserialize)]
struct OpenAiChunk {
    /// Server-assigned identifier for the overall completion.
    ///
    /// Stable across every chunk in a single streamed response (for
    /// example `chatcmpl-abc123`); emitted once as the message id in
    /// the initial [`MessageStart`].
    id: String,

    /// Name of the model that produced this chunk.
    ///
    /// Echoed back by the server on the first chunk; forwarded in the
    /// initial [`MessageStart`] so downstream consumers know which model
    /// answered, even after a fallback switch.
    model: String,

    /// One entry per alternative the model is generating.
    ///
    /// In practice OpenAI streams a single choice (`n=1`), so this
    /// vector usually holds exactly one [`OpenAiChoice`] carrying the
    /// incremental [`OpenAiDelta`] for this chunk. When
    /// `stream_options.include_usage` is set, the final chunk carries an
    /// empty `choices` array and the cumulative [`OpenAiUsage`] in
    /// [`usage`](Self::usage).
    choices: Vec<OpenAiChoice>,

    /// Cumulative token usage, present only on the final chunk.
    ///
    /// Populated when the request sets `stream_options.include_usage`;
    /// `None` on every preceding chunk and on all chunks when the option
    /// is not set. The [`StreamEmitter`] stores this and includes it in
    /// the [`MessageDelta`](StreamEvent::MessageDelta) event.
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

impl OpenAiChunk {
    /// Parse a raw SSE data payload into an [`OpenAiChunk`].
    ///
    /// Returns `None` for malformed payloads so the caller can skip
    /// them without interrupting the stream.
    fn parse(data: &str) -> Option<Self> {
        match serde_json::from_str(data) {
            Ok(chunk) => Some(chunk),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    data_len = data.len(),
                    "failed to parse OpenAI SSE chunk, skipping"
                );
                None
            }
        }
    }
}

/// A single alternative within an [`OpenAiChunk`].
///
/// Carries the incremental content for this turn (in `delta`) and, on
/// the final chunk for the choice, the reason the model stopped (in
/// `finish_reason`).
#[derive(Deserialize)]
struct OpenAiChoice {
    /// Incremental content for this chunk, or `None` on the terminal
    /// chunk that carries only a `finish_reason`.
    delta: Option<OpenAiDelta>,

    /// Why the model stopped generating, present only on the last chunk.
    ///
    /// Common values are `"stop"`, `"tool_calls"`, and `"length"`; the
    /// [`StreamEmitter`] maps it to a [`StreamEvent`] stop reason.
    finish_reason: Option<String>,
}

/// Token usage object carried by OpenAI's final streaming chunk.
///
/// Mirrors the `usage` field that appears on the last chunk when the request
/// sets `stream_options.include_usage`. Deserialized into a [`Usage`] so the
/// emitter can include it in the terminal [`MessageDelta`](StreamEvent::MessageDelta).
/// Both fields default to zero via `#[serde(default)]` so a partial usage
/// object from a non-conforming provider (e.g. one that omits
/// `completion_tokens`) does not fail deserialization and silently drop the
/// entire chunk.
#[derive(Deserialize)]
struct OpenAiUsage {
    /// Number of tokens in the input prompt.
    #[serde(default)]
    prompt_tokens: u64,

    /// Number of tokens in the output completion.
    #[serde(default)]
    completion_tokens: u64,
}

impl From<&OpenAiUsage> for Usage {
    fn from(u: &OpenAiUsage) -> Self {
        Usage::new(
            u32::try_from(u.prompt_tokens).unwrap_or(0),
            u32::try_from(u.completion_tokens).unwrap_or(0),
        )
    }
}

/// Incremental content delivered by one chunk.
///
/// Mirrors the `delta` object in OpenAI's streaming protocol. Every
/// field is optional because a single chunk typically populates only
/// the field it is extending (text content, reasoning, or a tool call).
#[derive(Deserialize)]
struct OpenAiDelta {
    /// Incremental assistant text for this chunk.
    ///
    /// Concatenated across chunks to reconstruct the full message body.
    content: Option<String>,

    /// Incremental chain-of-thought / reasoning text.
    ///
    /// Some models (e.g. o1-style reasoning models) emit their private
    /// reasoning here under `reasoning_content`; the `reasoning` alias
    /// covers providers that use the shorter key.
    #[serde(alias = "reasoning")]
    reasoning_content: Option<String>,

    /// Incremental tool-call fragments for this chunk.
    ///
    /// Tool calls arrive across multiple chunks keyed by `index`; the
    /// [`StreamEmitter`] accumulates them per index until each call's
    /// arguments are complete.
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

/// Incremental fragment of a single tool call within a chunk.
///
/// OpenAI streams tool calls in pieces: the first chunk for a given
/// `index` carries the call `id` and function `name`, subsequent chunks
/// append to `arguments`. The [`StreamEmitter`] reassembles these per
/// index.
#[derive(Deserialize)]
struct OpenAiToolCallDelta {
    /// Server-assigned identifier for the tool call.
    ///
    /// Present only on the first chunk for this `index`; continuation
    /// chunks omit it. The emitter latches it on the first chunk and
    /// forwards it as the call id so the host can match the result
    /// later. Defaults to `None` when the server omits the field.
    #[serde(default)]
    id: Option<String>,

    /// Position of this tool call in the request's tool list.
    ///
    /// Used to correlate fragments across chunks — chunks with the same
    /// `index` belong to the same tool call.
    index: usize,

    /// Function name and accumulated arguments for this tool call.
    ///
    /// `None` on chunks that carry no function update (for example a
    /// chunk that only extends a different tool call's `arguments`).
    /// When present, the inner [`OpenAiToolCallFunction`] holds either
    /// the function `name` (first chunk for this `index`) or a fragment
    /// of the JSON `arguments` (subsequent chunks) — the
    /// [`StreamEmitter`] reassembles both per index.
    #[serde(default)]
    function: Option<OpenAiToolCallFunction>,
}

/// Name and arguments of a tool call, as carried by an [`OpenAiToolCallDelta`].
///
/// The `name` arrives on the first chunk for a tool call; `arguments`
/// is a JSON string that may itself arrive in fragments across several
/// chunks and must be concatenated before parsing.
#[derive(Deserialize, Default)]
struct OpenAiToolCallFunction {
    /// Accumulated JSON arguments for the tool call.
    ///
    /// A partial JSON string that grows across chunks; the emitter
    /// buffers it per `index` and hands the complete string to the
    /// caller once the part stops.
    #[serde(default)]
    arguments: String,

    /// Fully-qualified name of the tool to invoke.
    ///
    /// Matches the `name` the tool was registered under in the request's
    /// `tools` array. Arrives on the first chunk for a given `index`
    /// only; continuation chunks for the same call omit it, so the
    /// emitter latches this value on the first chunk and ignores it on
    /// later ones. Defaults to `None` when the server omits the field.
    #[serde(default)]
    name: Option<String>,
}

/// Whether a content lane (text or thinking) currently has an open part.
///
/// [`StreamEmitter`] separates assistant text and model reasoning into two
/// independent lanes, each opened with a [`PartStart`](StreamEvent::PartStart)
/// on its first non-empty fragment and closed with a
/// [`PartStop`](StreamEvent::PartStop) when the lane switches or the stream
/// finishes. Only one lane may be open at a time — the emitter emits a
/// `PartStop` for the active lane before opening the other — so this enum
/// tracks the open/closed state of each lane without a bare `bool`.
#[derive(Default)]
enum PartLane {
    /// No part is open for this lane.
    ///
    /// The default state before the stream delivers any content for the
    /// lane, and the state it returns to once a part has been closed.
    #[default]
    Closed,

    /// A part is open and accumulating deltas.
    ///
    /// Set when the first non-empty fragment opens the lane; cleared when
    /// [`process_finish`](StreamEmitter::process_finish) emits the matching
    /// [`PartStop`](StreamEvent::PartStop).
    Open,
}

/// Stateful translator that converts a sequence of [`OpenAiChunk`]s
/// into [`StreamEvent`]s.
///
/// This encapsulates all the protocol-level bookkeeping:
/// - Emitting [`MessageStart`] once.
/// - Emitting [`PartStart`] / [`IndexedDelta`] for text and tool-call content.
/// - Emitting [`PartStop`] when parts finish.
/// - Emitting the final [`MessageDelta`] with a stop reason.
///
/// Splitting this out from the `try_stream!` macro body makes the
/// translation logic testable without a live network connection.
#[derive(Default)]
struct StreamEmitter {
    /// Whether [`StreamEvent::MessageStart`] has been emitted for the
    /// current stream.
    ///
    /// The first chunk carries the message `id` and `model`; the emitter
    /// forwards these once as a `MessageStart` and never again, so chunks
    /// arriving later in the stream are treated as content only.
    started: bool,

    /// Whether the text content part is currently open.
    ///
    /// OpenAI streams assistant text as a sequence of `delta.content`
    /// fragments on the first choice. The emitter opens a text part with
    /// [`StreamEvent::PartStart`] on the first non-empty fragment and
    /// tracks the open state so [`process_finish`](Self::process_finish)
    /// emits exactly one [`StreamEvent::PartStop`] to close it.
    text: PartLane,

    /// Whether the reasoning (thinking) content part is currently open.
    ///
    /// Reasoning models stream reasoning via `delta.reasoning_content` (or its
    /// alias `delta.reasoning`). The emitter opens a thinking part on the
    /// first non-empty fragment and closes it in `process_finish`, symmetric
    /// to the text lane.
    thinking: PartLane,

    /// Tool-call indices that have already had their
    /// [`StreamEvent::PartStart`] emitted.
    ///
    /// OpenAI streams a single tool call across many chunks, all sharing
    /// the same `index`: the first carries the call `id` and function
    /// `name`, and every later chunk carries only an `arguments`
    /// fragment (still under `function`). Gating `PartStart` on
    /// `function.is_some()` would re-emit it on every fragment and wipe
    /// the accumulator's buffered arguments. Tracking seen indices lets
    /// the emitter open each part exactly once.
    seen_tool_indices: Vec<usize>,

    /// Number of tool-call parts currently open.
    ///
    /// Each distinct tool `index` opens one tool part via
    /// [`StreamEvent::PartStart`]. The counter drives the matching batch
    /// of `PartStop` emissions on finish (one per open tool) so callers
    /// see balanced part lifecycles.
    open_tool_count: usize,

    /// Whether the terminal stop signal has been processed.
    ///
    /// Set by [`process_finish`](Self::process_finish) when a
    /// `finish_reason` arrives. Guards against emitting a second
    /// [`StreamEvent::MessageDelta`] if the stream delivers a duplicate
    /// finish chunk, and against `finish` appending a spurious
    /// [`StreamEvent::MessageStop`] after the stream already terminated.
    finished: bool,

    /// The stop reason captured by [`process_finish`](Self::process_finish),
    /// deferred until the usage chunk arrives or [`finish`](Self::finish)
    /// flushes it.
    ///
    /// OpenAI streams usage on a separate final chunk *after* the
    /// `finish_reason` chunk (when `stream_options.include_usage` is set).
    /// Rather than emit a `MessageDelta` immediately on `finish_reason` and
    /// lose the usage, the emitter stores the stop reason here and emits the
    /// `MessageDelta` once usage is known — either from the usage chunk or
    /// when [`finish`](Self::finish) flushes pending state at stream end.
    pending_stop_reason: Option<StreamStopReason>,

    /// Token usage captured from the final usage chunk, if the request
    /// set `stream_options.include_usage`.
    ///
    /// `None` until the usage chunk arrives. When `finish` flushes the
    /// deferred `MessageDelta`, this is converted to `Some(Usage)` (or left
    /// as `None` if the provider never sent usage).
    pending_usage: Option<Usage>,

    /// Buffered [`StreamEvent`]s waiting to be yielded to the consumer.
    ///
    /// All event-producing methods push onto this queue via
    /// [`push`](Self::push); the stream loop reads them back through
    /// [`drain`](Self::drain) after each chunk so events are yielded
    /// promptly rather than buffered until stream end.
    pending: Vec<StreamEvent>,
}

impl StreamEmitter {
    /// Process a single parsed OpenAI SSE chunk into stream events.
    ///
    /// On the first call, emits [`MessageStart`](StreamEvent::MessageStart)
    /// with the chunk's message ID and model. Then delegates to
    /// [`process_delta`](Self::process_delta) for text/tool-call deltas and
    /// [`process_finish`](Self::process_finish) for the terminal finish
    /// reason. Events accumulate in the internal queue until
    /// [`drain`](Self::drain) is called.
    fn process_chunk(&mut self, chunk: &OpenAiChunk) {
        if !self.started {
            self.started = true;
            self.push(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: chunk.id.clone(),
                    role: "assistant".into(),
                    model: chunk.model.clone(),
                },
            }));
        }

        if let Some(usage) = &chunk.usage {
            let typed = Usage::from(usage);
            if typed.input_tokens > 0 || typed.output_tokens > 0 {
                self.pending_usage = Some(typed);
            }
        }

        if let Some(choice) = chunk.choices.first() {
            if let Some(delta) = &choice.delta {
                self.process_delta(delta);
            }
            if let Some(reason) = &choice.finish_reason {
                self.process_finish(reason);
            }
        }

        if chunk.usage.is_some() {
            self.flush_message_delta();
        }
    }

    /// Translate a single delta object into text and/or reasoning events.
    ///
    /// If the delta carries non-empty `content`, emits a `PartStart` (on the
    /// first text delta) followed by `IndexedDelta(Text)` events. If it
    /// carries non-empty `reasoning_content`, the same shape is emitted for
    /// the reasoning lane. Switching lanes emits a `PartStop` for the previous
    /// lane first, keeping at most one content lane open — a lane left open
    /// behind a switch could later share its part index with a tool call and
    /// make closing ambiguous. If it carries `tool_calls`, delegates each to
    /// [`process_tool_call`](Self::process_tool_call).
    fn process_delta(&mut self, delta: &OpenAiDelta) {
        if let Some(text) = &delta.content
            && !text.is_empty()
        {
            if matches!(self.thinking, PartLane::Open) {
                self.thinking = PartLane::Closed;
                self.push(StreamEvent::PartStop {
                    index: Some(THINKING_PART_INDEX),
                });
            }
            if matches!(self.text, PartLane::Closed) {
                self.text = PartLane::Open;
                self.push(StreamEvent::PartStart(PartStart {
                    index: TEXT_PART_INDEX,
                    part: Some(MessagePart::text("")),
                }));
            }
            self.push(StreamEvent::IndexedDelta(IndexedDelta {
                index: TEXT_PART_INDEX,
                delta: DeltaPart::Text { text: text.clone() },
            }));
        }

        if let Some(reasoning) = &delta.reasoning_content
            && !reasoning.is_empty()
        {
            if matches!(self.text, PartLane::Open) {
                self.text = PartLane::Closed;
                self.push(StreamEvent::PartStop {
                    index: Some(TEXT_PART_INDEX),
                });
            }
            if matches!(self.thinking, PartLane::Closed) {
                self.thinking = PartLane::Open;
                self.push(StreamEvent::PartStart(PartStart {
                    index: THINKING_PART_INDEX,
                    part: None,
                }));
            }
            self.push(StreamEvent::IndexedDelta(IndexedDelta {
                index: THINKING_PART_INDEX,
                delta: DeltaPart::Thinking {
                    text: reasoning.clone(),
                },
            }));
        }

        if let Some(tool_calls) = &delta.tool_calls {
            self.close_content_lanes();
            for tc in tool_calls {
                self.process_tool_call(tc);
            }
        }
    }

    /// Close the open text and thinking lanes, if any.
    ///
    /// Called before the first tool [`PartStart`](StreamEvent::PartStart) of
    /// a chunk and at finish: wire tool-call indices and the content-lane
    /// indices overlap (a text lane at part index 0 versus a first tool call
    /// at wire index 0), so an open content lane at tool time would share its
    /// index with a tool lane. Each stop names the lane it closes.
    fn close_content_lanes(&mut self) {
        if matches!(self.text, PartLane::Open) {
            self.text = PartLane::Closed;
            self.push(StreamEvent::PartStop {
                index: Some(TEXT_PART_INDEX),
            });
        }
        if matches!(self.thinking, PartLane::Open) {
            self.thinking = PartLane::Closed;
            self.push(StreamEvent::PartStop {
                index: Some(THINKING_PART_INDEX),
            });
        }
    }

    /// Handle a single tool-call delta from the stream.
    ///
    /// OpenAI streams one tool call across many chunks that share an
    /// `index`: the first carries the call `id` and function `name`
    /// under `function`; every later chunk carries only an `arguments`
    /// fragment (still under `function`). The emitter opens the part
    /// with [`StreamEvent::PartStart`] exactly once per `index` (on the
    /// first chunk it sees for that index), then forwards every
    /// non-empty `arguments` fragment as an
    /// [`InputJson`](crate::stream::DeltaPart::InputJson) delta so the
    /// caller can concatenate them into the full JSON input.
    fn process_tool_call(&mut self, tc: &OpenAiToolCallDelta) {
        if tc.function.is_some() && !self.seen_tool_indices.contains(&tc.index) {
            self.seen_tool_indices.push(tc.index);
            self.push(StreamEvent::PartStart(PartStart {
                index: tc.index,
                part: Some(MessagePart::ToolCall {
                    id: tc.id.clone().unwrap_or_default(),
                    name: tc
                        .function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default(),
                    input: Value::Null,
                }),
            }));
            self.open_tool_count = self.open_tool_count.saturating_add(1);
        }

        if let Some(func) = &tc.function
            && !func.arguments.is_empty()
        {
            self.push(StreamEvent::IndexedDelta(IndexedDelta {
                index: tc.index,
                delta: DeltaPart::InputJson {
                    partial_json: func.arguments.clone(),
                },
            }));
        }
    }

    /// Handle a finish reason, closing open parts and deferring the
    /// `MessageDelta`.
    ///
    /// Closes any open text parts and tool-call parts with
    /// [`PartStop`](StreamEvent::PartStop), then stores the mapped
    /// [`StreamStopReason`] as pending. The [`MessageDelta`] is not emitted
    /// here — it is deferred until the usage chunk arrives (when
    /// `stream_options.include_usage` is set) or flushed by
    /// [`finish`](Self::finish) at stream end, so the `MessageDelta` carries
    /// both the stop reason and the usage in one event. Maps `"tool_calls"`
    /// → [`ToolCall`](StreamStopReason::ToolCall), `"length"` →
    /// [`MaxTokens`](StreamStopReason::MaxTokens), and anything else via
    /// [`StreamStopReason::from_api_str`]. No-ops if already finished.
    fn process_finish(&mut self, reason: &str) {
        if self.finished {
            return;
        }
        self.finished = true;

        self.close_content_lanes();
        let open_tool_indices: Vec<usize> = self
            .seen_tool_indices
            .iter()
            .take(self.open_tool_count)
            .copied()
            .collect();
        for index in open_tool_indices {
            self.push(StreamEvent::PartStop { index: Some(index) });
        }

        let stop_reason = match reason {
            "tool_calls" => StreamStopReason::ToolCall,
            "length" => StreamStopReason::MaxTokens,
            other => StreamStopReason::from_api_str(other).unwrap_or(StreamStopReason::EndTurn),
        };
        self.pending_stop_reason = Some(stop_reason);
    }

    /// Emit the deferred [`MessageDelta`](StreamEvent::MessageDelta), if one
    /// is pending.
    ///
    /// Called by [`process_chunk`](Self::process_chunk) when the usage chunk
    /// arrives, and by [`finish`](Self::finish) as a last resort. Emits the
    /// `MessageDelta` carrying the pending stop reason and whatever usage has
    /// been captured so far, then clears the pending state so it fires at
    /// most once.
    fn flush_message_delta(&mut self) {
        if let Some(stop_reason) = self.pending_stop_reason.take() {
            self.push(StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some(stop_reason.to_api_str().into()),
                },
                usage: self.pending_usage,
            }));
        }
    }

    /// Finalize the stream, emitting the terminal
    /// [`MessageStop`](StreamEvent::MessageStop) if one was started.
    ///
    /// Flushes any deferred [`MessageDelta`](StreamEvent::MessageDelta)
    /// (when no usage chunk arrived), drains remaining pending events, and
    /// appends the stop event. Called exactly once at the end of the SSE
    /// stream.
    fn finish(&mut self) -> Vec<StreamEvent> {
        self.flush_message_delta();
        let mut out = self.drain();
        if self.started {
            out.push(StreamEvent::MessageStop);
        }
        out
    }

    /// Drain all pending events from the internal queue.
    ///
    /// Returns the accumulated [`StreamEvent`]s and clears the queue.
    /// Called by the stream loop after each chunk is processed so events
    /// are yielded promptly rather than buffered until stream end.
    fn drain(&mut self) -> Vec<StreamEvent> {
        std::mem::take(&mut self.pending)
    }

    /// Push an event onto the internal pending queue.
    ///
    /// The single write point — all methods (`process_delta`,
    /// `process_tool_call`, `process_finish`, `process_chunk`) funnel
    /// through here. Events are held until [`drain`](Self::drain) is called.
    fn push(&mut self, ev: StreamEvent) {
        self.pending.push(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessagePart, Role, ToolContent};
    use crate::tool::ToolSchema;

    #[test]
    fn openai_emitter_part_lane_default_closed() {
        let em = StreamEmitter::default();
        assert!(
            matches!(em.text, PartLane::Closed) && matches!(em.thinking, PartLane::Closed),
            "both content lanes must start closed"
        );
    }

    #[test]
    fn request_body_includes_system_message_first() {
        let msgs = vec![Message::user("hello")];
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            Some("be brief"),
            None,
            None,
            &ToolConstraint::None,
        );
        let json = body.to_json(true);

        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn request_body_without_system() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);
        let json = body.to_json(false);

        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn request_body_stream_flag_toggles() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);

        assert_eq!(body.to_json(true)["stream"], true);
        assert_eq!(body.to_json(false)["stream"], false);
    }

    #[test]
    fn request_body_streaming_includes_usage_option() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);

        assert_eq!(body.to_json(true)["stream_options"]["include_usage"], true);
    }

    #[test]
    fn request_body_non_streaming_omits_usage_option() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);

        assert!(
            body.to_json(false).get("stream_options").is_none(),
            "stream_options should only be present when streaming"
        );
    }

    #[test]
    fn request_body_stream_usage_disabled_omits_stream_options() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None)
            .with_stream_usage(false);

        assert!(
            body.to_json(true).get("stream_options").is_none(),
            "stream_options should be absent when stream_usage is disabled"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn ollama_constructor_disables_stream_usage() {
        let client = crate::provider::ollama("test-model").unwrap();
        assert!(
            !client.stream_usage,
            "ollama() should disable stream_usage for compatibility"
        );
    }

    #[test]
    fn default_builder_enables_stream_usage() {
        let client = OpenAiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        assert!(
            client.stream_usage,
            "default builder should enable stream_usage"
        );
    }

    #[test]
    fn emitter_usage_chunk_after_finish_carries_usage_in_delta() {
        let mut em = StreamEmitter::default();

        let text = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&text);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();

        let usage_chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
        )
        .unwrap();
        em.process_chunk(&usage_chunk);
        let events = em.drain();

        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta from usage chunk");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("end_turn"));
        let usage = delta.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn emitter_finish_without_usage_chunk_emits_delta_with_none_usage() {
        let mut em = StreamEmitter::default();

        let text = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&text);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();

        let events = em.finish();
        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta from finish");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("end_turn"));
        assert!(delta.usage.is_none());
    }

    #[test]
    fn request_body_model_and_tools() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = RequestBody::build(
            "my-model",
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::None,
        );
        let json = body.to_json(true);

        assert_eq!(json["model"], "my-model");
        let tools_arr = json["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["type"], "function");
        assert_eq!(tools_arr[0]["function"]["name"], "echo");
    }

    #[test]
    fn request_body_tools_absent_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);
        let json = body.to_json(false);
        assert!(
            json.get("tools").is_none(),
            "tools key should be absent when no tools are set"
        );
    }

    #[test]
    fn convert_message_user_text() {
        let m = Message::user("hello world");
        let v = convert_message(&m).remove(0);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello world");
    }

    #[test]
    fn convert_message_assistant_text() {
        let m = Message::new(Role::Assistant, vec![MessagePart::text("hi there")]);
        let v = convert_message(&m).remove(0);
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "hi there");
    }

    #[test]
    fn convert_message_assistant_tool_calls() {
        let m = Message::new(
            Role::Assistant,
            vec![MessagePart::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: serde_json::json!({"message": "hi"}),
            }],
        );
        let v = convert_message(&m).remove(0);
        assert_eq!(v["role"], "assistant");
        assert!(v["content"].is_null());
        let calls = v["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "echo");
        assert_eq!(
            calls[0]["function"]["arguments"].as_str().unwrap(),
            r#"{"message":"hi"}"#
        );
    }

    #[test]
    fn convert_message_tool_result() {
        let m = Message::new(
            Role::User,
            vec![MessagePart::ToolResult {
                call_id: "call_1".into(),
                name: "echo".into(),
                output: ToolContent::from_string("result text"),
                is_error: None,
            }],
        );
        let v = convert_message(&m).remove(0);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_1");
        assert!(v["content"].is_string());
    }

    #[test]
    fn convert_message_multiple_tool_results_expand() {
        let m = Message::new(
            Role::User,
            vec![
                MessagePart::ToolResult {
                    call_id: "call_1".into(),
                    name: "echo".into(),
                    output: ToolContent::from_string("a"),
                    is_error: None,
                },
                MessagePart::ToolResult {
                    call_id: "call_2".into(),
                    name: "echo".into(),
                    output: ToolContent::from_string("b"),
                    is_error: None,
                },
            ],
        );
        let vs = convert_message(&m);
        assert_eq!(vs.len(), 2, "two tool results expand to two messages");
        assert_eq!(vs[0]["role"], "tool");
        assert_eq!(vs[0]["tool_call_id"], "call_1");
        assert_eq!(vs[1]["role"], "tool");
        assert_eq!(vs[1]["tool_call_id"], "call_2");
    }

    #[test]
    fn convert_tools_shape() {
        let tools = vec![
            ToolSchema {
                tool: "search".into(),
                description: "Search the web".into(),
                input_schema: serde_json::json!({"type": "object"}),
            },
            ToolSchema {
                tool: "calc".into(),
                description: "Calculate".into(),
                input_schema: serde_json::json!({"type": "object"}),
            },
        ];
        let out = convert_tools(&tools);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["function"]["name"], "search");
        assert_eq!(out[1]["function"]["name"], "calc");
    }

    #[test]
    fn input_to_string_passes_through_strings() {
        assert_eq!(input_to_string(&Value::String("raw".into())), "raw");
    }

    #[test]
    fn input_to_string_serializes_objects() {
        let v = serde_json::json!({"a": 1});
        let s = input_to_string(&v);
        assert_eq!(s, r#"{"a":1}"#);
    }

    #[test]
    fn input_to_string_serializes_numbers() {
        let s = input_to_string(&Value::from(42));
        assert_eq!(s, "42");
    }

    #[test]
    fn parse_valid_chunk() {
        let data = r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let chunk = OpenAiChunk::parse(data).unwrap();
        assert_eq!(chunk.id, "chatcmpl-1");
        assert_eq!(chunk.model, "gpt-4o");
        assert_eq!(chunk.choices.len(), 1);
    }

    #[test]
    fn parse_malformed_returns_none() {
        assert!(OpenAiChunk::parse("not json").is_none());
        assert!(OpenAiChunk::parse("").is_none());
    }

    #[test]
    fn parse_malformed_partial_json_returns_none() {
        // Truncated JSON should also fail gracefully with a warning log.
        assert!(OpenAiChunk::parse(r#"{"id":"chatcmpl-1","choices":[{"delta":{"con"#).is_none());
    }

    #[test]
    fn parse_valid_chunk_with_all_fields() {
        let data = r#"{"id":"abc","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}],"usage":null}"#;
        let chunk = OpenAiChunk::parse(data).unwrap();
        assert_eq!(chunk.id, "abc");
        assert_eq!(chunk.model, "gpt-4o");
        assert_eq!(chunk.choices.len(), 1);
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn parse_chunk_missing_usage_defaults_to_none() {
        let data = r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let chunk = OpenAiChunk::parse(data).unwrap();
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn parse_final_chunk_with_partial_usage_defaults_missing_fields() {
        let data = r#"{"id":"c1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":15}}"#;
        let chunk = OpenAiChunk::parse(data).unwrap();
        let usage = chunk.usage.as_ref().expect("usage should parse");
        assert_eq!(usage.prompt_tokens, 15);
        assert_eq!(usage.completion_tokens, 0);
    }

    #[test]
    fn parse_final_chunk_with_usage() {
        let data = r#"{"id":"c1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}}"#;
        let chunk = OpenAiChunk::parse(data).unwrap();
        assert!(chunk.choices.is_empty());
        let usage = chunk.usage.as_ref().expect("usage");
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 7);
        let typed: Usage = usage.into();
        assert_eq!(typed.input_tokens, 42);
        assert_eq!(typed.output_tokens, 7);
    }

    #[test]
    fn emitter_usage_and_finish_in_same_chunk() {
        let mut em = StreamEmitter::default();

        let text = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&text);
        em.drain();

        let combined = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3}}"#,
        )
        .unwrap();
        em.process_chunk(&combined);
        let events = em.drain();

        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta from combined chunk");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("end_turn"));
        let usage = delta.usage.expect("usage");
        assert_eq!(usage.input_tokens, 8);
        assert_eq!(usage.output_tokens, 3);
    }

    #[test]
    fn emitter_tool_call_stream_with_usage_chunk() {
        let mut em = StreamEmitter::default();

        let open = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&open);
        em.drain();

        let args = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"rust\"}"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&args);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();

        let usage_chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":20}}"#,
        )
        .unwrap();
        em.process_chunk(&usage_chunk);
        let events = em.drain();

        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta after usage chunk");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("tool_call"));
        let usage = delta.usage.expect("usage");
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.output_tokens, 20);
    }

    #[test]
    fn emitter_emits_message_start_on_first_chunk() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":""},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStart(_)))
        );
    }

    #[test]
    fn emitter_text_delta_starts_part_then_deltas() {
        let mut em = StreamEmitter::default();

        // First chunk with text — should emit MessageStart + PartStart + IndexedDelta.
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);

        let events = em.drain();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[1],
            StreamEvent::PartStart(ref p) if p.index == TEXT_PART_INDEX
        ));
        assert!(matches!(
            events[2],
            StreamEvent::IndexedDelta(ref d) if d.index == TEXT_PART_INDEX
        ));

        // Second text chunk — should only emit a delta (part already open).
        let chunk2 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk2);
        let events2 = em.drain();
        assert_eq!(events2.len(), 1);
        assert!(matches!(events2[0], StreamEvent::IndexedDelta(_)));
    }

    #[test]
    fn emitter_tool_call_emits_part_start_and_delta() {
        let mut em = StreamEmitter::default();

        // Start message first.
        let chunk0 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk0);
        em.drain();

        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            StreamEvent::PartStart(ref p) if p.index == 1
        ));
        assert!(matches!(
            events[1],
            StreamEvent::IndexedDelta(ref d) if d.index == 1
        ));
    }

    #[test]
    fn emitter_multi_chunk_tool_call_emits_part_start_once() {
        let mut em = StreamEmitter::default();
        let chunk0 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk0);
        em.drain();

        let header = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&header);
        em.drain();

        let fragment = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&fragment);
        let events = em.drain();
        let deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::IndexedDelta(_)))
            .collect();
        assert_eq!(
            deltas.len(),
            1,
            "follow-up chunk must emit only an argument delta, not a PartStart"
        );
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, StreamEvent::PartStart(_)))
        );
        assert_eq!(em.open_tool_count, 1);
    }

    #[test]
    fn emitter_multi_chunk_tool_call_accumulates_through_accumulator() {
        use crate::stream::StreamAccumulator;
        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).unwrap();
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).unwrap();
            }
        }
        for ev in em.finish() {
            acc.process(&ev).unwrap();
        }

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        match &msg.parts[0] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(input, &serde_json::json!({"msg": "hi"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn emitter_two_interleaved_multi_chunk_tool_calls_accumulate() {
        use crate::stream::StreamAccumulator;

        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"search","arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"b\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).unwrap();
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).unwrap();
            }
        }
        for ev in em.finish() {
            acc.process(&ev).unwrap();
        }

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 2, "two tool calls expected");
        match &msg.parts[0] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(input, &serde_json::json!({"msg": "a"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &msg.parts[1] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "search");
                assert_eq!(input, &serde_json::json!({"q": "b"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn text_then_tool_call_stream_preserves_tool_arguments() {
        use crate::stream::StreamAccumulator;

        // The text lane opens at part index 0 and the wire tool index is
        // also 0 — without addressed lane closing and kind-aware routing
        // every InputJson fragment lands in the text slot and the tool
        // executes with `{}`.
        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"Let me check that."},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).unwrap();
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).unwrap();
            }
        }
        for ev in em.finish() {
            acc.process(&ev).unwrap();
        }

        let msg = acc.build();
        assert_eq!(
            msg.parts.len(),
            2,
            "the text part and the tool part both flush"
        );
        match &msg.parts[0] {
            MessagePart::Text { text } => assert_eq!(text, "Let me check that."),
            other => panic!("expected Text, got {other:?}"),
        }
        match &msg.parts[1] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(
                    input,
                    &serde_json::json!({"msg": "hi"}),
                    "the tool arguments must survive the text-lane index collision"
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn tool_call_then_text_reopen_preserves_both() {
        use crate::stream::StreamAccumulator;

        // Wire-legal interleaving: content resumes after tool fragments, so
        // the reopened text lane shares part index 0 with the tool lane.
        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"Done, here is what I found."},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).unwrap();
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).unwrap();
            }
        }
        for ev in em.finish() {
            acc.process(&ev).unwrap();
        }

        let msg = acc.build();
        let tool = msg
            .parts
            .iter()
            .find_map(|p| match p {
                MessagePart::ToolCall { name, input, .. } => Some((name.clone(), input.clone())),
                _ => None,
            })
            .expect("the tool call must survive the reopened text lane");
        assert_eq!(tool.0, "echo");
        assert_eq!(tool.1, serde_json::json!({"msg": "hi"}));
        let text = msg
            .parts
            .iter()
            .find_map(|p| match p {
                MessagePart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .expect("the trailing text must survive the reopened lane");
        assert_eq!(text, "Done, here is what I found.");
    }

    #[test]
    fn thinking_then_two_tool_calls_preserve_arguments() {
        use crate::stream::StreamAccumulator;

        // The reasoning lane sits at part index 1 and the second tool
        // call's wire index is also 1 — the collision shape.
        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"reasoning_content":"thinking hard"},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"echo","arguments":"{\"a\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"search","arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"rust\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).unwrap();
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).unwrap();
            }
        }
        for ev in em.finish() {
            acc.process(&ev).unwrap();
        }

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 2, "both tool calls flush");
        match &msg.parts[0] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(input, &serde_json::json!({"a": 1}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &msg.parts[1] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "search");
                assert_eq!(
                    input,
                    &serde_json::json!({"q": "rust"}),
                    "the index-1 call must survive the thinking-lane collision"
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn emitter_real_continuation_chunks_omit_id_and_name() {
        use crate::stream::StreamAccumulator;

        let mut em = StreamEmitter::default();
        let mut acc = StreamAccumulator::new();
        let chunks = [
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        ];
        for raw in chunks {
            let chunk = OpenAiChunk::parse(raw).expect("real chunk shape must deserialize");
            em.process_chunk(&chunk);
            for ev in em.drain() {
                acc.process(&ev).expect("accumulator accepts events");
            }
        }
        for ev in em.finish() {
            acc.process(&ev).expect("accumulator accepts finish events");
        }

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1, "one tool call expected");
        match &msg.parts[0] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(
                    input,
                    &serde_json::json!({"msg": "hi"}),
                    "continuation fragment must accumulate, not be dropped"
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn tool_open_closes_text_lane_with_addressed_stop() {
        let mut em = StreamEmitter::default();
        let text = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"let me look"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&text);
        em.drain();

        let tool = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{}"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&tool);
        let events = em.drain();

        assert!(
            matches!(
                events.first(),
                Some(StreamEvent::PartStop { index: Some(0) })
            ),
            "the open text lane must close with an addressed stop before the tool part opens: {events:?}"
        );
        assert!(
            matches!(events.get(1), Some(StreamEvent::PartStart(ps))
                if matches!(ps.part, Some(MessagePart::ToolCall { .. }))),
            "the tool PartStart follows the lane close: {events:?}"
        );
    }

    #[test]
    fn tool_open_closes_thinking_lane_with_addressed_stop() {
        let mut em = StreamEmitter::default();
        let reasoning = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"reasoning_content":"deliberating"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&reasoning);
        em.drain();

        let tool = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"search","arguments":"{}"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&tool);
        let events = em.drain();

        assert!(
            matches!(
                events.first(),
                Some(StreamEvent::PartStop { index: Some(1) })
            ),
            "the open thinking lane must close with an addressed stop naming part index 1 \
             before a tool call at the same wire index opens: {events:?}"
        );
        assert!(
            matches!(events.get(1), Some(StreamEvent::PartStart(ps))
                if matches!(ps.part, Some(MessagePart::ToolCall { .. }))),
            "the tool PartStart follows the lane close: {events:?}"
        );
    }

    #[test]
    fn emitter_finish_emits_part_stops_and_message_delta() {
        let mut em = StreamEmitter::default();

        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        let part_stop_events = em.drain();
        assert_eq!(part_stop_events.len(), 1);
        assert!(matches!(part_stop_events[0], StreamEvent::PartStop { .. }));

        let events = em.finish();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageDelta(_)))
        );
    }

    #[test]
    fn emitter_finish_closes_lanes_so_late_delta_reopens() {
        let mut em = StreamEmitter::default();

        let text_chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&text_chunk);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();
        assert!(
            matches!(em.text, PartLane::Closed),
            "process_finish must close the text lane"
        );

        let late_delta = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"more"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&late_delta);
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PartStart(_))),
            "a delta after finish must re-open the text lane with PartStart"
        );
    }

    #[test]
    fn emitter_finish_with_tool_calls_stop_reason() {
        let mut em = StreamEmitter::default();

        let chunk0 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk0);
        em.drain();

        let tool_chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&tool_chunk);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        let part_stop_events = em.drain();
        assert_eq!(part_stop_events.len(), 1);
        assert!(matches!(part_stop_events[0], StreamEvent::PartStop { .. }));

        let events = em.finish();
        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("tool_call"));
    }

    #[test]
    fn emitter_finish_appends_message_stop() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();

        let final_events = em.finish();
        assert!(matches!(
            final_events.last(),
            Some(StreamEvent::MessageStop)
        ));
    }

    #[test]
    fn emitter_finish_without_start_is_empty() {
        let mut em = StreamEmitter::default();
        let events = em.finish();
        assert!(events.is_empty());
    }

    #[test]
    fn emitter_finish_reason_length_maps_to_max_tokens() {
        let mut em = StreamEmitter::default();
        let chunk0 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"x"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk0);
        em.drain();

        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"length"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        em.drain();

        let events = em.finish();
        let delta = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(md) => Some(md),
            _ => None,
        });
        let delta = delta.expect("MessageDelta");
        assert_eq!(delta.delta.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn emitter_empty_content_does_not_open_text_part() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":""},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        // Only MessageStart, no PartStart for empty content.
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::MessageStart(_)));
    }

    #[test]
    fn emitter_double_finish_ignored() {
        let mut em = StreamEmitter::default();
        let chunk0 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk0);
        em.drain();

        let finish1 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish1);
        em.drain();

        // Second finish should not emit anything extra.
        let finish2 = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish2);
        let events = em.drain();
        assert!(events.is_empty());
    }

    #[test]
    fn sse_reader_take_line_extracts_newline_terminated() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\n".into(),
        };
        let line = reader.take_line().unwrap().unwrap();
        assert_eq!(line, "data: hello");
        assert!(reader.buf.is_empty());
    }

    #[test]
    fn sse_reader_take_line_returns_none_without_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "partial".into(),
        };
        assert!(reader.take_line().unwrap().is_none());
    }

    #[test]
    fn sse_reader_take_line_handles_multiple_lines() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "line1\nline2\n".into(),
        };
        assert_eq!(reader.take_line().unwrap().unwrap(), "line1");
        assert_eq!(reader.take_line().unwrap().unwrap(), "line2");
    }

    #[test]
    fn sse_reader_take_line_trims_cr() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hi\r\n".into(),
        };
        let line = reader.take_line().unwrap().unwrap();
        assert_eq!(line, "data: hi");
    }

    #[test]
    fn builder_timeouts_applied_on_build() {
        let client = OpenAiClient::builder()
            .with_api_key("sk-test")
            .with_timeout(Duration::from_mins(3))
            .with_connect_timeout(Duration::from_secs(15))
            .build();
        assert!(client.is_ok(), "build should succeed with valid timeouts");
    }

    #[tokio::test]
    async fn sse_reader_take_line_splits_on_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\ndata: world\n".to_string().into_bytes(),
        };
        assert_eq!(reader.take_line().unwrap(), Some("data: hello".to_string()));
        assert_eq!(reader.take_line().unwrap(), Some("data: world".to_string()));
        assert_eq!(reader.take_line().unwrap(), None);
    }

    #[tokio::test]
    async fn sse_reader_next_data_extracts_payload() {
        let data = "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[]}\n\n";
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(data.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        let result = reader.next_openai_data().await.unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("c1"));
    }

    #[tokio::test]
    async fn sse_reader_next_data_done_returns_none() {
        let stream = futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
            "data: [DONE]\n\n".into(),
        )]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        let result = reader.next_openai_data().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        let huge = "x".repeat(2 * 1024 * 1024);
        let stream = futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(huge.into())]);
        let mut reader = super::SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        let result = reader.next_openai_data().await;
        assert!(result.is_err(), "should error on buffer overflow");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SSE buffer"),
            "error should mention SSE buffer: {err_msg}"
        );
    }

    #[test]
    fn max_response_body_is_ten_mb() {
        assert_eq!(super::super::MAX_RESPONSE_BODY, 10 * 1024 * 1024);
    }

    #[test]
    fn request_body_response_format_emitted() {
        let msgs = vec![Message::user("hi")];
        let rf =
            crate::structured::ResponseFormat::new("action", serde_json::json!({"type": "object"}));
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            None,
            None,
            Some(&rf),
            &ToolConstraint::None,
        );
        let json = body.to_json(false);

        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["name"], "action");
        assert_eq!(
            json["response_format"]["json_schema"]["schema"],
            serde_json::json!({"type": "object"})
        );
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn request_body_response_format_absent_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &ToolConstraint::None);
        let json = body.to_json(false);
        assert!(
            json.get("response_format").is_none(),
            "response_format should be absent (not null) when not set"
        );
    }

    #[test]
    fn request_body_response_format_suppresses_tools() {
        let msgs = vec![Message::user("hi")];
        let caller_tool = ToolSchema {
            tool: "read".into(),
            description: "Read".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let rf =
            crate::structured::ResponseFormat::new("result", serde_json::json!({"type": "object"}));
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            None,
            Some(&[caller_tool]),
            Some(&rf),
            &ToolConstraint::None,
        );
        let json = body.to_json(false);

        assert!(
            json.get("tools").is_none(),
            "tools key should be absent when response_format is set"
        );
        assert!(json.get("response_format").is_some());
    }

    #[test]
    fn extract_structured_from_text_part() {
        let client = OpenAiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::assistant(r#"{"tool": "write", "args": {}}"#);
        let value = client.extract_structured(&message);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_from_tool_call_part() {
        let client = OpenAiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::new(
            Role::Assistant,
            vec![MessagePart::tool_call(
                "tc_1",
                "action",
                serde_json::json!({"tool": "read", "args": {}}),
            )],
        );
        let value = client.extract_structured(&message);
        assert_eq!(value["tool"], "read");
    }

    #[test]
    fn extract_structured_prose_falls_back_to_string() {
        let client = OpenAiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::assistant("I cannot produce that.");
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!("I cannot produce that."));
    }

    #[test]
    fn build_response_maps_text_and_stop_finish_reason() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "hello"},
                "finish_reason": "stop"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.message.role, Role::Assistant);
        assert_eq!(response.message.text_content(), "hello");
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_maps_tool_calls_finish_reason() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.message.parts.len(), 1);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "search");
                assert_eq!(input, &serde_json::json!({"q": "x"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        assert_eq!(response.stop_reason, StreamStopReason::ToolCall);
    }

    #[test]
    fn build_response_maps_length_to_max_tokens() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "truncated"},
                "finish_reason": "length"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.stop_reason, StreamStopReason::MaxTokens);
    }

    #[test]
    fn build_response_extracts_usage() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7}
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.usage.expect("usage").input_tokens, 42);
        assert_eq!(response.usage.expect("usage").output_tokens, 7);
        assert_eq!(response.usage.expect("usage").total_tokens(), 49);
    }

    #[test]
    fn build_response_missing_usage_is_none() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert!(response.usage.is_none());
    }

    #[test]
    fn build_response_zero_usage_collapses_to_none() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert!(
            response.usage.is_none(),
            "all-zero usage must collapse to None"
        );
    }

    #[test]
    fn build_response_text_and_tool_calls_combined() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Let me search",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.message.parts.len(), 2);
        assert!(response.message.parts[0].is_text());
        assert!(response.message.parts[1].is_tool_call());
        assert_eq!(response.stop_reason, StreamStopReason::ToolCall);
    }

    #[test]
    fn build_response_multiple_tool_calls_preserve_order() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [
                        {"id": "a", "function": {"name": "first", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "second", "arguments": "{\"n\": 2}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.message.parts.len(), 2);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, name, .. } => {
                assert_eq!(id, "a");
                assert_eq!(name, "first");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &response.message.parts[1] {
            MessagePart::ToolCall { id, name, .. } => {
                assert_eq!(id, "b");
                assert_eq!(name, "second");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_malformed_arguments_returns_error() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": "not valid json{"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let result = OpenAiClient::build_response(&raw);
        assert!(
            result.is_err(),
            "malformed non-empty arguments must surface as an error, not silently default to {{}}"
        );
    }

    #[test]
    fn build_response_empty_arguments_defaults_to_empty_object() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": ""}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        match &response.message.parts[0] {
            MessagePart::ToolCall { input, .. } => {
                assert_eq!(input, &serde_json::json!({}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_missing_arguments_defaults_to_empty_object() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        match &response.message.parts[0] {
            MessagePart::ToolCall { input, .. } => {
                assert_eq!(input, &serde_json::json!({}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_missing_choices_yields_empty_message() {
        let raw = serde_json::json!({});
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert!(response.message.parts.is_empty());
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_unrecognized_finish_reason_defaults_to_end_turn() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "content_filter"
            }]
        });
        let response = OpenAiClient::build_response(&raw).unwrap();
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn openai_strict_sets_flag_and_tightens() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"msg": {"type": "string"}}
            }),
        }];
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::Strict,
        );
        let json = body.to_json(false);

        let tools_arr = json["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["function"]["strict"], true);
        // Schema tightened: additionalProperties false, required enumerated.
        let params = &tools_arr[0]["function"]["parameters"];
        assert_eq!(params["additionalProperties"], false);
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "msg");
    }

    #[test]
    fn openai_none_constraint_unchanged_shape() {
        // Default None must produce a plain body: no `strict` field, no
        // tightening, no guided_json.
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::None,
        );
        let json = body.to_json(false);

        let tools_arr = json["tools"].as_array().unwrap();
        // No `strict` field on the function entry under None.
        assert!(
            tools_arr[0]["function"].get("strict").is_none(),
            "strict must not appear under ToolConstraint::None"
        );
        assert!(
            json.get("guided_json").is_none(),
            "guided_json must not appear under ToolConstraint::None"
        );
        // input_schema is passed through unchanged (no tightening).
        assert_eq!(
            tools_arr[0]["function"]["parameters"],
            serde_json::json!({"type": "object"})
        );
    }

    #[test]
    fn openai_strict_suppressed_when_response_format_set() {
        // With both set, tools is absent and no strict emission happens.
        let msgs = vec![Message::user("hi")];
        let caller_tool = ToolSchema {
            tool: "read".into(),
            description: "Read".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let rf =
            crate::structured::ResponseFormat::new("result", serde_json::json!({"type": "object"}));
        let body = RequestBody::build(
            "gpt-4o",
            &msgs,
            None,
            Some(&[caller_tool]),
            Some(&rf),
            &ToolConstraint::Strict,
        );
        let json = body.to_json(false);

        assert!(
            json.get("tools").is_none(),
            "tools must be absent when response_format is set"
        );
        assert!(
            json.get("guided_json").is_none(),
            "guided_json must be absent when response_format is set"
        );
        assert!(json.get("response_format").is_some());
    }

    #[cfg(feature = "grammar")]
    #[test]
    fn openai_grammar_injects_guided_json() {
        use crate::provider::grammar::JsonSchemaGrammar;
        use crate::structured::ToolConstraint;

        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let grammar = std::sync::Arc::new(JsonSchemaGrammar::from_schemas(&tools));
        let constraint = ToolConstraint::Grammar(grammar);
        let body = RequestBody::build("gpt-4o", &msgs, None, Some(&tools), None, &constraint);
        let json = body.to_json(false);

        // guided_json carries the compiled grammar string.
        let guided = json["guided_json"].as_str().unwrap();
        assert!(
            guided.contains("echo"),
            "guided_json should reference the tool: {guided}"
        );
        // Tools are still advertised (Grammar is about the sampler, not
        // tool visibility).
        assert!(json.get("tools").is_some());
        // Under Grammar, no `strict: true` is emitted (that's Strict's path).
        let tools_arr = json["tools"].as_array().unwrap();
        assert!(tools_arr[0]["function"].get("strict").is_none());
    }

    #[cfg(feature = "grammar")]
    #[test]
    fn openai_grammar_without_tools_omits_guided_json() {
        use crate::provider::grammar::JsonSchemaGrammar;
        use crate::structured::ToolConstraint;

        let msgs = vec![Message::user("hi")];
        let grammar = std::sync::Arc::new(JsonSchemaGrammar::from_schemas(&[]));
        let constraint = ToolConstraint::Grammar(grammar);

        // No tools registered: guided_json must be absent so the model's
        // free-text output is not forced into the (empty) tool grammar.
        let body = RequestBody::build("gpt-4o", &msgs, None, None, None, &constraint);
        let json = body.to_json(false);
        assert!(
            json.get("guided_json").is_none(),
            "guided_json must be absent when no tools are registered"
        );
        assert!(
            json.get("tools").is_none(),
            "tools must be absent when none were supplied"
        );

        // Empty tool slice: same outcome — no guided_json, no tools.
        let body = RequestBody::build("gpt-4o", &msgs, None, Some(&[]), None, &constraint);
        let json = body.to_json(false);
        assert!(
            json.get("guided_json").is_none(),
            "guided_json must be absent for an empty tool slice"
        );
    }

    #[test]
    fn convert_message_system_role_emitted_inline() {
        // OpenAI accepts an inline {role: "system"} message natively, so a
        // Role::System message is emitted verbatim (not folded into a
        // top-level field).
        let msg = Message::new(Role::System, vec![MessagePart::text("stay on task")]);
        let value = convert_message(&msg).remove(0);
        assert_eq!(value["role"], "system");
        assert_eq!(value["content"], "stay on task");
    }

    #[test]
    fn openai_delta_reasoning_content_emits_thinking() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"reasoning_content":"thinking…"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        // Expect at least one IndexedDelta carrying Thinking.
        let thinking = events.iter().find_map(|e| match e {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { text } => Some(text.clone()),
                _ => None,
            },
            _ => None,
        });
        assert_eq!(thinking.as_deref(), Some("thinking…"));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PartStart(_))),
            "a PartStart should fire for the reasoning lane"
        );
    }

    #[test]
    fn openai_delta_reasoning_alias_works() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"reasoning":"via alias"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        let thinking = events.iter().find_map(|e| match e {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { text } => Some(text.clone()),
                _ => None,
            },
            _ => None,
        });
        assert_eq!(
            thinking.as_deref(),
            Some("via alias"),
            "#[serde(alias = \"reasoning\")] must accept the field"
        );
    }

    #[test]
    fn openai_reasoning_does_not_open_text_part() {
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"reasoning_content":"reasoning only"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        // No Text delta should be emitted from a reasoning-only chunk.
        let has_text = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Text { .. })
            )
        });
        assert!(!has_text, "reasoning-only chunk must not emit Text deltas");
        assert!(
            matches!(em.text, PartLane::Closed),
            "reasoning must not open the text lane"
        );
    }

    #[test]
    fn openai_reasoning_and_text_interleave() {
        let mut em = StreamEmitter::default();

        // Chunk 1: reasoning.
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);

        // Chunk 2: text.
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"content":"answer"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        // Both lanes should have fired — one Thinking, one Text.
        let has_thinking = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Thinking { .. })
            )
        });
        let has_text = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Text { .. })
            )
        });
        assert!(has_thinking, "a Thinking delta should have fired");
        assert!(has_text, "a Text delta should have fired");

        // The lane switch from reasoning to text must emit exactly one
        // PartStop for the reasoning lane before the text PartStart, so at
        // most one content lane is ever open.
        let lane_switch_stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::PartStop { .. }))
            .count();
        assert_eq!(
            lane_switch_stops, 1,
            "lane switch must close the reasoning lane with one PartStop"
        );
    }

    #[test]
    fn openai_combined_content_and_reasoning_in_one_delta() {
        // A single delta object can carry both content and reasoning_content.
        // process_delta handles content first (closing thinking if open), then
        // reasoning (closing text if open). Lock in the expected sequence:
        // PartStart(text) → TextDelta → PartStop → PartStart(thinking) →
        // ThinkingDelta.
        let mut em = StreamEmitter::default();
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"o1","choices":[{"delta":{"content":"answer","reasoning_content":"why"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        let has_text = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Text { ref text } if text == "answer")
            )
        });
        let has_thinking = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Thinking { ref text } if text == "why")
            )
        });
        assert!(has_text, "text delta must fire");
        assert!(has_thinking, "thinking delta must fire");

        // The reasoning lane opens after the text lane, so the text lane must
        // be closed with exactly one PartStop between them.
        let stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::PartStop { .. }))
            .count();
        assert_eq!(stops, 1, "exactly one PartStop for the text lane");

        // Ordering: TextDelta → PartStop → ThinkingDelta.
        let text_idx = events
            .iter()
            .position(|e| {
                matches!(
                    e,
                    StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Text { .. })
                )
            })
            .expect("text delta present");
        let stop_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::PartStop { .. }))
            .expect("PartStop present");
        let thinking_idx = events
            .iter()
            .rposition(|e| {
                matches!(
                    e,
                    StreamEvent::IndexedDelta(d) if matches!(d.delta, DeltaPart::Thinking { .. })
                )
            })
            .expect("thinking delta present");
        assert!(text_idx < stop_idx, "text delta before PartStop");
        assert!(stop_idx < thinking_idx, "PartStop before thinking delta");
    }

    #[test]
    fn request_body_omits_tools_for_empty_slice() {
        let body = RequestBody::build(
            "m",
            &[crate::message::Message::user("hi")],
            None,
            Some(&[]),
            None,
            &ToolConstraint::None,
        )
        .to_json(false);
        assert!(
            body.get("tools").is_none(),
            "an empty tool list must be omitted, not sent as []; got {}",
            body.get("tools").unwrap_or(&serde_json::Value::Null)
        );
    }

    #[test]
    fn completions_url_has_no_double_slash_for_trailing_slash_base() {
        let bare = OpenAiClient::builder()
            .with_api_key("k")
            .with_base_url("https://api.example.com/v1")
            .build()
            .expect("client builds");
        let slashed = OpenAiClient::builder()
            .with_api_key("k")
            .with_base_url("https://api.example.com/v1/")
            .build()
            .expect("client builds");
        assert_eq!(
            slashed.completions_url(),
            bare.completions_url(),
            "a trailing-slash base URL must join to the same request URL as the bare one"
        );
    }

    #[test]
    fn convert_message_keeps_text_alongside_tool_results() {
        let msg = crate::message::Message::new(
            crate::message::Role::User,
            vec![
                crate::message::MessagePart::text("stale results, search again"),
                crate::message::MessagePart::tool_result(
                    "c1",
                    "search",
                    crate::message::ToolContent::from_string("[]"),
                    false,
                ),
            ],
        );
        let json = serde_json::to_string(&convert_message(&msg)).unwrap_or_default();
        assert!(
            json.contains("stale results, search again"),
            "text parts accompanying tool results must reach the model; got {json}"
        );
        let messages = convert_message(&msg);
        assert_eq!(
            messages.len(),
            2,
            "one tool message plus the trailing user text message: {messages:?}"
        );
        assert_eq!(messages[0]["role"], "tool", "the tool result comes first");
        assert_eq!(messages[0]["tool_call_id"], "c1");
        assert_eq!(
            messages[1]["role"], "user",
            "the preserved text rides as a trailing user message"
        );
        assert_eq!(messages[1]["content"], "stale results, search again");
    }

    #[tokio::test]
    async fn sse_data_line_without_space_is_parsed() {
        let data = "data:{\"ok\":true}\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
        };
        let parsed = reader
            .next_openai_data()
            .await
            .expect("reader must not err");
        assert!(
            parsed.is_some(),
            "spec-legal 'data:' line must yield the payload, not be skipped"
        );
    }

    #[tokio::test]
    async fn compact_done_marker_terminates_the_stream() {
        let data = "data:[DONE]\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
        };
        let parsed = reader
            .next_openai_data()
            .await
            .expect("reader must not err");
        assert_eq!(
            parsed, None,
            "the compact [DONE] marker must end the stream exactly like the spaced form"
        );
    }

    #[tokio::test]
    async fn bare_data_line_yields_empty_payload_then_next_chunk_parses() {
        let data = "data:\n\ndata:{\"ok\":1}\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
        };
        let first = reader
            .next_openai_data()
            .await
            .expect("reader must not err");
        assert_eq!(
            first,
            Some(String::new()),
            "a bare data field carries an empty payload, not a skipped line"
        );
        let second = reader
            .next_openai_data()
            .await
            .expect("reader must not err");
        assert_eq!(
            second.as_deref(),
            Some("{\"ok\":1}"),
            "the chunk after a bare data line must still parse"
        );
        let third = reader
            .next_openai_data()
            .await
            .expect("reader must not err");
        assert_eq!(third, None, "the stream must end cleanly after the payload");
    }
}
