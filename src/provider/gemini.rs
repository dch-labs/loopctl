//! Google Gemini API client.
//!
//! Implements [`ApiClient`] by translating between the framework's
//! [`StreamEvent`] protocol and the Gemini Streaming Generate Content
//! API (SSE format).
//!
//! # Construction
//!
//! ```rust,ignore
//! use loopctl::provider::GeminiClient;
//!
//! // From environment (GEMINI_API_KEY or GOOGLE_API_KEY):
//! let client = GeminiClient::from_env()?;
//!
//! // Explicit:
//! let client = GeminiClient::builder()
//!     .with_api_key("AIza...")
//!     .with_model("gemini-2.0-flash")
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;

use futures::stream::Stream;
use serde_json::Value;
use std::time::Duration;

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

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const SSE_DATA_PREFIX: &str = "data: ";
const TEXT_PART_INDEX: usize = 0;
const THINKING_PART_INDEX: usize = 1;
const MAX_ERROR_BODY: usize = 8 * 1024; // 8 Kb

/// A Google Gemini API client with streaming support.
///
/// Implements [`ApiClient`] by translating between the framework's
/// [`StreamEvent`] protocol and the Gemini Streaming Generate Content API.
pub struct GeminiClient {
    /// The underlying HTTP client (connection-pooled `reqwest::Client`).
    ///
    /// Created once at build time with the configured timeouts; reused
    /// across all requests for connection pooling.
    http: reqwest::Client,

    /// The Gemini API key used for authentication.
    ///
    /// Sent as the `x-goog-api-key` header on every request. Set via
    /// [`GeminiClientBuilder::api_key`].
    api_key: String,

    /// The base URL for API requests.
    ///
    /// The streaming endpoint is `{base_url}/models/{model}:streamGenerateContent`
    /// and the non-streaming endpoint is `{base_url}/models/{model}:generateContent`.
    /// Defaults to `https://generativelanguage.googleapis.com/v1beta`.
    base_url: String,

    /// The current model identifier, stored behind a mutex for runtime
    /// hot-swapping.
    ///
    /// Changed via [`ApiClient::set_model`] when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips to a
    /// fallback model.
    model: std::sync::Mutex<String>,

    /// Whether to request thought summaries from reasoning-capable models.
    ///
    /// When `true`, every request body gets
    /// `generationConfig.thinkingConfig.includeThoughts = true`. The Gemini
    /// API rejects `thinkingConfig` with `400 INVALID_ARGUMENT` on models
    /// that don't support thinking, so this is opt-in — the caller must know
    /// their model is reasoning-capable (e.g. Gemini 2.5 Pro/Flash, Gemini 3)
    /// before enabling it. Defaults to `false`. Set via
    /// [`GeminiClientBuilder::include_thoughts`].
    include_thoughts: bool,
}

impl GeminiClient {
    /// Create a builder for configuring a [`GeminiClient`].
    ///
    /// Returns a [`GeminiClientBuilder`] with sensible defaults. The only
    /// required field is `api_key`; everything else has a production-ready
    /// default. Call `.with_api_key(...).build()` to finish, or chain additional
    /// setters for custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::provider::GeminiClient;
    ///
    /// let client = GeminiClient::builder()
    ///     .with_api_key("AI...")
    ///     .with_model("gemini-2.0-flash")
    ///     .build()
    /// .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> GeminiClientBuilder {
        GeminiClientBuilder::default()
    }

    /// Create a client from environment variables.
    ///
    /// Reads the following variables:
    ///
    /// - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — **required**. The API key
    ///   for authentication.
    /// - `GEMINI_BASE_URL` — optional. Defaults to
    ///   `https://generativelanguage.googleapis.com/v1beta`.
    /// - `GEMINI_MODEL` — optional. Defaults to `gemini-2.0-flash`.
    ///
    /// This is a convenience constructor that delegates to
    /// [`builder`](Self::builder) with the env vars as setter arguments.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key is found.
    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| ApiError::auth_invalid_key("GEMINI_API_KEY not set"))?;
        let base_url = std::env::var("GEMINI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());

        Self::builder()
            .with_api_key(api_key)
            .with_base_url(base_url)
            .with_model(model)
            .build()
    }

    /// Build the streaming Generate Content URL.
    ///
    /// Gemini puts the model in the URL path. The API key is sent via
    /// the `x-goog-api-key` header, not as a query parameter.
    fn stream_url(&self) -> String {
        let model = crate::error::recover_guard(self.model.lock()).clone();
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        )
    }

    /// Build the full URL for the Gemini non-streaming Generate Content
    /// endpoint.
    ///
    /// Constructs `{base_url}/models/{model}:generateContent`. The API key
    /// is sent via the `x-goog-api-key` header, not in the URL.
    /// Used by [`ApiClient::create_message`] and its `*_with_options`
    /// variant.
    fn generate_url(&self) -> String {
        let model = crate::error::recover_guard(self.model.lock()).clone();
        format!("{}/models/{}:generateContent", self.base_url, model)
    }

    /// Build a typed [`NonStreamingResponse`] from Gemini's native JSON.
    ///
    /// Reads `candidates[0].content.parts` into [`MessagePart`]s: each `text`
    /// field becomes a [`MessagePart::Text`] part and each `functionCall`
    /// becomes a [`MessagePart::ToolCall`] with its `id` preserved when
    /// present (Gemini 3 assigns a unique id per call; older versions omit
    /// it, in which case the id defaults to an empty string). A single part
    /// may hold both `text` and `functionCall`, in which case it yields two
    /// parts. Maps `candidates[0].finishReason` to a [`StreamStopReason`]
    /// using the same mapping the streaming emitter applies: `"MAX_TOKENS"` →
    /// `MaxTokens`, anything else (including the `"STOP"` default) →
    /// `EndTurn`. Reads `usageMetadata.promptTokenCount` and
    /// `candidatesTokenCount` (plus `thoughtsTokenCount`) into [`Usage`],
    /// returning `None` when the object is absent or all-zero.
    fn build_response(raw: &Value) -> crate::api::NonStreamingResponse {
        let mut parts: Vec<MessagePart> = Vec::new();
        if let Some(content_parts) = raw
            .pointer("/candidates/0/content/parts")
            .and_then(|p| p.as_array())
        {
            for part in content_parts {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    parts.push(MessagePart::text(text));
                }
                if let Some(fc) = part.get("functionCall") {
                    let id = fc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = fc
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));
                    parts.push(MessagePart::tool_call(id, name, input));
                }
            }
        }
        let reason = raw
            .pointer("/candidates/0/finishReason")
            .and_then(|r| r.as_str())
            .unwrap_or("STOP");
        let stop_reason = match reason {
            "MAX_TOKENS" => StreamStopReason::MaxTokens,
            _ => StreamStopReason::EndTurn,
        };
        let usage = extract_usage(raw);
        crate::api::NonStreamingResponse {
            message: Message::new(Role::Assistant, parts),
            stop_reason,
            usage,
        }
    }

    /// Send a POST request and return the raw response.
    ///
    /// Shared by both [`ApiClient::stream_messages`] and
    /// [`ApiClient::create_message`].
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the request fails or the server
    /// responds with a non-success status code.
    async fn post_content(
        http: &reqwest::Client,
        url: &str,
        api_key: &str,
        body: &Value,
    ) -> Result<reqwest::Response, ApiError> {
        let resp = http
            .post(url)
            .header("x-goog-api-key", api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| ApiError::http(e.to_string()))?;
        let status = resp.status();
        if status.is_success() {
            Ok(resp)
        } else {
            // Cap the error body to prevent OOM from oversized error responses.
            let bytes = resp.bytes().await.unwrap_or_default();
            let text = match bytes.get(..MAX_ERROR_BODY) {
                Some(truncated) => String::from_utf8_lossy(truncated).into_owned(),
                None => String::from_utf8_lossy(&bytes).into_owned(),
            };
            Err(ApiError::http_with_status(status.as_u16(), text))
        }
    }
}

/// Extract token [`Usage`] from Gemini's native `usageMetadata` object.
///
/// Reads `usageMetadata.promptTokenCount` for input tokens and sums
/// `candidatesTokenCount` plus `thoughtsTokenCount` for output. Returns `None`
/// when the `usageMetadata` object is absent or when all counts are zero,
/// matching the convention used by the streaming emitter in
/// `extract_finish_reason`.
fn extract_usage(raw: &Value) -> Option<Usage> {
    let usage = raw.pointer("/usageMetadata")?;
    let input = usage
        .get("promptTokenCount")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let output = usage
        .get("candidatesTokenCount")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let thoughts = usage
        .get("thoughtsTokenCount")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let total_output = output.saturating_add(thoughts);
    (input > 0 || total_output > 0).then(|| Usage::new(input, total_output))
}

impl ApiClient for GeminiClient {
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
        let body = build_request_body(
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
            self.include_thoughts,
        );
        let url = self.stream_url();
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_content(&http, &url, &api_key, &body).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_gemini_data().await? {
                emitter.process_chunk(&data);
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
        let body = build_request_body(
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
            self.include_thoughts,
        );
        let url = self.generate_url();

        Box::pin(async move {
            let resp = Self::post_content(&self.http, &url, &self.api_key, &body).await?;
            let resp = super::read_bounded_body(resp).await?;
            let raw = serde_json::from_slice::<Value>(&resp)
                .map_err(|e| ApiError::http(e.to_string()))?;
            Ok(Self::build_response(&raw))
        })
    }

    fn stream_messages_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let system = request.system.clone();
        let tools = request.tools.clone();
        let rf = options.response_format.as_ref();
        let body = build_request_body(
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            rf,
            &options.tool_constraint,
            self.include_thoughts,
        );
        let url = self.stream_url();
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_content(&http, &url, &api_key, &body).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_gemini_data().await? {
                emitter.process_chunk(&data);
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
        let response_format = options.response_format.as_ref();
        let body = build_request_body(
            &request.messages,
            system.as_deref(),
            tools.as_deref(),
            response_format,
            &options.tool_constraint,
            self.include_thoughts,
        );
        let url = self.generate_url();
        Box::pin(async move {
            let resp = Self::post_content(&self.http, &url, &self.api_key, &body).await?;
            let resp = super::read_bounded_body(resp).await?;
            let raw = serde_json::from_slice::<Value>(&resp)
                .map_err(|e| ApiError::http(e.to_string()))?;
            Ok(Self::build_response(&raw))
        })
    }
}

/// Builder for [`GeminiClient`].
///
/// Created via [`GeminiClientBuilder::default`] or
/// [`GeminiClient::builder`]. All fields have sensible defaults except
/// `api_key`, which must be set before [`build`](Self::build).
pub struct GeminiClientBuilder {
    /// The Gemini API key for authentication (required).
    ///
    /// Must be set before building. Sent as the `x-goog-api-key` header on
    /// every request.
    api_key: Option<String>,

    /// The base URL for API requests.
    ///
    /// Defaults to `https://generativelanguage.googleapis.com/v1beta`.
    base_url: String,

    /// The default model identifier.
    ///
    /// Can be changed at runtime via [`GeminiClient::set_model`].
    model: String,

    /// Whether to opt into thought summaries from reasoning-capable models.
    ///
    /// When `true`, every request gets
    /// `generationConfig.thinkingConfig.includeThoughts = true`. The Gemini
    /// API rejects `thinkingConfig` on non-reasoning models with
    /// `400 INVALID_ARGUMENT`, so this is `false` by default and the caller
    /// must opt in once they know their model supports thinking.
    include_thoughts: bool,

    /// Shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// Holds the timeout, connection-pool, and TCP knobs that apply to the
    /// internally-built `reqwest::Client`, or an externally-supplied client
    /// injected via [`with_http_client`](Self::with_http_client).
    http: super::HttpClientConfig,
}

impl Default for GeminiClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            include_thoughts: false,
            http: super::HttpClientConfig::default(),
        }
    }
}

impl GeminiClientBuilder {
    /// Set the API key for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is not set.
    /// The key is sent as the `x-goog-api-key` header on every request.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL for API requests.
    ///
    /// Defaults to `https://generativelanguage.googleapis.com/v1beta`.
    /// Override when targeting a proxy or Google AI-compatible endpoint.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the default model identifier.
    ///
    /// The model string is embedded in the request URL (e.g.
    /// `/models/{model}:generateContent`). Can be changed at runtime via
    /// [`GeminiClient::set_model`] (e.g. when the
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

    /// Opt into thought summaries from reasoning-capable models.
    ///
    /// When enabled, every request body gets
    /// `generationConfig.thinkingConfig.includeThoughts = true`, which asks
    /// Gemini to surface its reasoning alongside the visible answer. Only
    /// reasoning-capable models (e.g. Gemini 2.5 Pro/Flash, Gemini 3) honor
    /// this — non-reasoning models reject `thinkingConfig` with
    /// `400 INVALID_ARGUMENT`, so this defaults to `false`.
    ///
    /// The response-side parser routes `thought: true` parts to
    /// [`DeltaPart::Thinking`] regardless of this flag — it's purely the
    /// request-side opt-in.
    #[must_use]
    pub fn with_include_thoughts(mut self, enabled: bool) -> Self {
        self.include_thoughts = enabled;
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
    pub fn build(self) -> Result<GeminiClient, ApiError> {
        let api_key = self
            .api_key
            .ok_or_else(|| ApiError::auth_invalid_key("API key not provided"))?;
        let http = self.http.build()?;

        Ok(GeminiClient {
            http,
            api_key,
            base_url: self.base_url,
            model: std::sync::Mutex::new(self.model),
            include_thoughts: self.include_thoughts,
        })
    }
}

/// Build the JSON request body for the Gemini Generate Content API.
///
/// Unlike OpenAI/Anthropic, Gemini puts the model in the URL, not the
/// request body. Each [`Message`] is serialized via [`convert_message`].
///
/// `generationConfig` is injected only when it has something to carry:
/// `thinkingConfig.includeThoughts = true` when `include_thoughts` is set,
/// and/or `responseMimeType` + `responseJsonSchema` when `response_format`
/// is set. When neither applies, `generationConfig` is omitted entirely.
///
/// Tool-call constraint:
/// - When `response_format` is set, suppresses `tools`; `tool_constraint`
///   is ignored in that case.
/// - Otherwise, when `tool_constraint` is `Strict`, each
///   `functionDeclaration`'s `parameters` is tightened
///   (`additionalProperties: false`, full `required`) via [`convert_tools`].
///   No `toolConfig.functionCallingConfig` is injected — Strict constrains
///   the call's shape, not its selection.
fn build_request_body(
    messages: &[Message],
    system: Option<&str>,
    tools: Option<&[ToolSchema]>,
    response_format: Option<&crate::structured::ResponseFormat>,
    tool_constraint: &ToolConstraint,
    include_thoughts: bool,
) -> Value {
    let (non_system, effective_system) = super::fold_system_messages(messages, system);
    let contents: Vec<Value> = non_system.iter().map(|m| convert_message(m)).collect();

    let mut body = serde_json::json!({ "contents": contents });
    if let Some(obj) = body.as_object_mut() {
        if let Some(sys) = effective_system {
            obj.insert(
                "systemInstruction".into(),
                serde_json::json!({"parts": [{"text": sys}]}),
            );
        }

        if response_format.is_none()
            && let Some(tool_list) = tools
        {
            let strict = matches!(tool_constraint, ToolConstraint::Strict);
            obj.insert(
                "tools".into(),
                serde_json::json!([{"functionDeclarations": convert_tools(tool_list, strict)}]),
            );
        }

        let mut generation_config = serde_json::Map::new();
        if include_thoughts {
            generation_config.insert(
                "thinkingConfig".into(),
                serde_json::json!({ "includeThoughts": true }),
            );
        }
        if let Some(rf) = response_format {
            generation_config.insert("responseMimeType".into(), "application/json".into());
            generation_config.insert("responseJsonSchema".into(), rf.schema.clone());
        }
        if !generation_config.is_empty() {
            obj.insert("generationConfig".into(), Value::Object(generation_config));
        }
    }

    body
}

/// Convert a single framework [`Message`] into the Gemini JSON shape.
///
/// Gemini uses `role: "user"` / `role: "model"` (not "assistant") and
/// a `parts` array for content blocks.
fn convert_message(m: &Message) -> Value {
    // System messages are folded into the top-level `systemInstruction` field
    // by `build_request_body` before this function is reached, so the `System`
    // pattern below is defensive: if one ever reaches here, route it to `user`
    // so the text renders rather than being dropped silently.
    let role = match m.role {
        Role::User | Role::System => "user",
        Role::Assistant => "model",
    };
    let parts: Vec<Value> = m.parts.iter().filter_map(convert_part).collect();
    serde_json::json!({"role": role, "parts": parts})
}

/// Convert a single [`MessagePart`] into a Gemini part JSON object.
///
/// Returns `None` for image parts (not yet supported for Gemini).
fn convert_part(p: &MessagePart) -> Option<Value> {
    match p {
        MessagePart::Text { text } => Some(serde_json::json!({"text": text})),
        MessagePart::ToolCall { id, name, input } => {
            let mut fc = serde_json::Map::new();
            fc.insert("name".to_string(), serde_json::Value::String(name.clone()));
            fc.insert("args".to_string(), input.clone());
            if !id.is_empty() {
                fc.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            Some(serde_json::json!({"functionCall": fc}))
        }
        MessagePart::ToolResult {
            call_id,
            name,
            output,
            ..
        } => {
            let mut fr = serde_json::Map::new();
            fr.insert("name".to_string(), serde_json::Value::String(name.clone()));
            if !call_id.is_empty() {
                fr.insert("id".to_string(), serde_json::Value::String(call_id.clone()));
            }
            fr.insert(
                "response".to_string(),
                serde_json::json!({"result": output.to_string()}),
            );
            Some(serde_json::json!({"functionResponse": fr}))
        }
        MessagePart::Image { .. } => None,
    }
}

/// Convert framework tool schemas into the Gemini `functionDeclarations`
/// array shape.
///
/// Each [`ToolSchema`] becomes a JSON object with `name`, `description`, and
/// `parameters` — the fields Gemini's function-calling API expects. When
/// `strict` is `true`, each `parameters` is first tightened (recursive
/// `additionalProperties: false` and full `required`) — Gemini has no native
/// per-function strict flag, so the tightening is the structural constraint
/// behind [`ToolConstraint::Strict`].
///
/// When structured output is active (`response_format` set), this function is not
/// called — [`build_request_body`] injects `generationConfig.responseJsonSchema`
/// instead, and `tools` is suppressed.
fn convert_tools(tools: &[ToolSchema], strict: bool) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let parameters = if strict {
                tighten_json_schema(&t.input_schema)
            } else {
                t.input_schema.clone()
            };
            serde_json::json!({
                "name": t.tool,
                "description": &t.description,
                "parameters": parameters,
            })
        })
        .collect()
}

use super::sse::SseReader;

impl SseReader {
    /// Extract the next SSE `data:` payload as parsed JSON.
    ///
    /// Returns `Ok(None)` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails.
    async fn next_gemini_data(&mut self) -> Result<Option<Value>, ApiError> {
        loop {
            while let Some(line) = self.take_line()? {
                if line.is_empty() {
                    continue;
                }
                if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
                    match serde_json::from_str::<Value>(data) {
                        Ok(json) => return Ok(Some(json)),
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                data_len = data.len(),
                                "failed to parse Gemini SSE data, skipping"
                            );
                        }
                    }
                }
            }
            if self.next_chunk().await?.is_none() {
                return Ok(None);
            }
        }
    }
}

/// Whether a content lane (text or thinking) currently has an open part.
///
/// Gemini interleaves regular text parts and reasoning (`thought: true`)
/// parts within a single chunk's `parts[]` array. [`StreamEmitter`] opens a
/// part for each lane with a [`PartStart`](StreamEvent::PartStart) on its
/// first non-empty fragment and closes it with a
/// [`PartStop`](StreamEvent::PartStop) when the lane switches or the stream
/// finishes. The two lanes are mutually exclusive — switching emits a
/// `PartStop` for the active lane first — so this enum tracks each lane's
/// open/closed state without a bare `bool`.
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
    /// [`extract_finish_reason`](StreamEmitter::extract_finish_reason) or a
    /// lane switch emits the matching
    /// [`PartStop`](StreamEvent::PartStop).
    Open,
}

/// How far the stream has progressed through its terminal sequence.
///
/// A Gemini stream has two independent terminal transitions:
/// [`extract_finish_reason`](StreamEmitter::extract_finish_reason) processes a
/// `finishReason` chunk (emitting a [`MessageDelta`](StreamEvent::MessageDelta)
/// and closing any open parts), and [`finish`](StreamEmitter::finish) appends
/// the synthetic [`MessageStop`](StreamEvent::MessageStop). Either may occur
/// first — a `finishReason` chunk may arrive before stream end, and `finish`
/// is always called at stream end — and each must fire exactly once. Because
/// the two are independent, this enum encodes all four combinations rather
/// than a single linear progression, advancing to [`Terminal`](Self::Terminal)
/// only once both are complete.
#[derive(Default)]
enum TerminalStage {
    /// Neither transition has run yet.
    ///
    /// The default state at the start of a stream.
    #[default]
    Pending,

    /// [`finish`](StreamEmitter::finish) has emitted the synthetic
    /// [`MessageStop`](StreamEvent::MessageStop).
    ///
    /// Subsequent `finish` calls are no-ops.
    StopEmitted,

    /// A `finishReason` chunk has been fully processed.
    ///
    /// Subsequent `finishReason` chunks (e.g. from proxies that re-emit)
    /// are no-ops.
    FinishReasonSeen,

    /// Both transitions are complete.
    ///
    /// The only state from which no further terminal work is possible;
    /// reached from either [`StopEmitted`](Self::StopEmitted) or
    /// [`FinishReasonSeen`](Self::FinishReasonSeen) once the other
    /// transition fires.
    Terminal,
}

/// Stateful translator that converts Gemini SSE chunks into
/// [`StreamEvent`]s.
///
/// Gemini's SSE format is simpler than Anthropic's: each `data:` line
/// is a complete JSON object with `candidates[0].content.parts[]` for
/// text/function-call data and `candidates[0].finishReason` for the
/// stop reason. Reasoning models (Gemini 2.5+) interleave thought parts
/// with regular parts in the same `parts[]` array, flagged by a
/// `thought: true` boolean on each thought part.
#[derive(Default)]
struct StreamEmitter {
    /// Whether [`StreamEvent::MessageStart`] has been emitted for the
    /// current stream.
    ///
    /// Gemini does not send a dedicated message-start event; the emitter
    /// synthesizes one on the first chunk (with empty `id` and `model`,
    /// since Gemini's streaming chunks don't carry them) and treats later
    /// chunks as content only.
    started: bool,

    /// Whether the text content part is currently open.
    ///
    /// The emitter opens a text part with [`StreamEvent::PartStart`] on the
    /// first non-empty text fragment and tracks the open state so
    /// [`extract_finish_reason`](Self::extract_finish_reason) emits exactly
    /// one [`StreamEvent::PartStop`] to close it.
    text: PartLane,

    /// Whether the reasoning (thinking) content part is currently open.
    ///
    /// Reasoning models flag thought parts with `thought: true`. The emitter
    /// opens a thinking part on the first non-empty thought fragment and
    /// closes it in [`extract_finish_reason`](Self::extract_finish_reason),
    /// symmetric to the text lane.
    thinking: PartLane,

    /// Next tool-call index for multiple function calls in one response.
    ///
    /// Gemini can emit several `functionCall` parts in a single chunk.
    /// Each gets its own `PartStart` at an incrementing index so the
    /// accumulator can distinguish them.
    next_tool_index: usize,

    /// How far the stream has progressed through its terminal sequence.
    ///
    /// Tracks two independent transitions — `finishReason` processing (by
    /// [`extract_finish_reason`](Self::extract_finish_reason)) and the
    /// synthetic [`StreamEvent::MessageStop`] emission (by
    /// [`finish`](Self::finish)) — as a single field. The two are
    /// independent: a `finishReason` chunk may arrive before or after
    /// `MessageStop`, and each must fire exactly once.
    terminal: TerminalStage,

    /// Buffered [`StreamEvent`]s waiting to be yielded to the consumer.
    ///
    /// All event-producing methods push onto this queue via
    /// [`push`](Self::push); the stream loop reads them back through
    /// [`drain`](Self::drain) after each chunk so events are yielded
    /// promptly rather than buffered until stream end.
    pending: Vec<StreamEvent>,
}

impl StreamEmitter {
    /// Process a single Gemini SSE JSON chunk into stream events.
    ///
    /// On the first call, emits [`MessageStart`](StreamEvent::MessageStart).
    /// Then delegates to the three extractors: [`extract_text`](Self::extract_text)
    /// for text deltas, [`extract_function_call`](Self::extract_function_call)
    /// for tool calls, and [`extract_finish_reason`](Self::extract_finish_reason)
    /// for the terminal stop signal. Events accumulate in the internal queue
    /// until [`drain`](Self::drain) is called.
    fn process_chunk(&mut self, json: &Value) {
        if !self.started {
            self.started = true;
            self.push(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: String::new(),
                    role: "assistant".into(),
                    model: String::new(),
                },
            }));
        }

        self.extract_parts_with_tools(json);
        self.extract_finish_reason(json);
    }

    /// Extract text, thought, and tool-call parts from the chunk, preserving
    /// the original `content.parts` order.
    ///
    /// Gemini can interleave text, thought (`thought: true`), and
    /// `functionCall` parts within a single chunk. This method walks the
    /// parts array in order and routes each to its lane, closing the
    /// previous lane first when switching — the downstream accumulator
    /// keys on a single active index, so failing to close would let
    /// deltas for one lane clobber the other's buffered state.
    fn extract_parts_with_tools(&mut self, json: &Value) {
        let Some(parts) = json
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        else {
            return;
        };

        for part in parts {
            if part.get("functionCall").is_some() {
                self.handle_function_call(part);
                continue;
            }

            let Some(text) = part.get("text").and_then(Value::as_str) else {
                continue;
            };

            if text.is_empty() {
                continue;
            }

            let is_thought = part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            if is_thought {
                if matches!(self.text, PartLane::Open) {
                    self.text = PartLane::Closed;
                    self.push(StreamEvent::PartStop);
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
                        text: text.to_string(),
                    },
                }));
            } else {
                if matches!(self.thinking, PartLane::Open) {
                    self.thinking = PartLane::Closed;
                    self.push(StreamEvent::PartStop);
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
                    delta: DeltaPart::Text {
                        text: text.to_string(),
                    },
                }));
            }
        }
    }

    /// Handle a single `functionCall` part, emitting `PartStart`,
    /// `InputJson`, and `PartStop`.
    ///
    /// Called inline from [`extract_parts_with_tools`] so tool calls appear
    /// at their original position relative to text/thought parts. Each call
    /// gets its own incrementing index.
    fn handle_function_call(&mut self, part: &Value) {
        let Some(func_call) = part.get("functionCall") else {
            return;
        };
        let id = func_call
            .pointer("/id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let name = func_call
            .pointer("/name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let args = func_call.pointer("/args").cloned().unwrap_or(Value::Null);
        let args_str = serde_json::to_string(&args).unwrap_or_default();

        if matches!(self.thinking, PartLane::Open) {
            self.thinking = PartLane::Closed;
            self.push(StreamEvent::PartStop);
        }
        if matches!(self.text, PartLane::Open) {
            self.text = PartLane::Closed;
            self.push(StreamEvent::PartStop);
        }

        let idx = self.next_tool_index;
        self.next_tool_index = self.next_tool_index.saturating_add(1);

        self.push(StreamEvent::PartStart(PartStart {
            index: idx,
            part: Some(MessagePart::ToolCall {
                id,
                name,
                input: args,
            }),
        }));
        self.push(StreamEvent::IndexedDelta(IndexedDelta {
            index: idx,
            delta: DeltaPart::InputJson {
                partial_json: args_str,
            },
        }));
        self.push(StreamEvent::PartStop);
    }

    /// Extract the finish reason from the chunk and emit stop events.
    ///
    /// Reads `candidates[0].finishReason` (e.g. `"STOP"`, `"MAX_TOKENS"`).
    /// When present, closes any open thinking part and text part with
    /// [`PartStop`](StreamEvent::PartStop), then emits a
    /// [`MessageDelta`](StreamEvent::MessageDelta) carrying the mapped
    /// [`StreamStopReason`]. No-op on a second `finishReason` chunk (the
    /// `terminal` stage guards against proxies/gateways that re-emit).
    fn extract_finish_reason(&mut self, json: &Value) {
        if matches!(
            self.terminal,
            TerminalStage::FinishReasonSeen | TerminalStage::Terminal
        ) {
            return;
        }
        let Some(reason) = json
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        else {
            return;
        };
        self.terminal = match self.terminal {
            TerminalStage::StopEmitted => TerminalStage::Terminal,
            _ => TerminalStage::FinishReasonSeen,
        };
        let stop = match reason {
            "MAX_TOKENS" => StreamStopReason::MaxTokens,
            _ => StreamStopReason::EndTurn,
        };

        if matches!(self.thinking, PartLane::Open) {
            self.thinking = PartLane::Closed;
            self.push(StreamEvent::PartStop);
        }

        if matches!(self.text, PartLane::Open) {
            self.text = PartLane::Closed;
            self.push(StreamEvent::PartStop);
        }

        let usage = json.pointer("/usageMetadata").and_then(|u| {
            let input = u
                .get("promptTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let output = u
                .get("candidatesTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let thoughts = u
                .get("thoughtsTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let total_output = output.saturating_add(thoughts);
            if input == 0 && total_output == 0 {
                None
            } else {
                Some(Usage::new(
                    u32::try_from(input).unwrap_or(0),
                    u32::try_from(total_output).unwrap_or(0),
                ))
            }
        });

        self.push(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop.to_api_str().into()),
            },
            usage,
        }));
    }

    /// Drain all pending events from the internal queue.
    ///
    /// Returns the accumulated [`StreamEvent`]s and clears the queue.
    /// Called by the stream loop after each chunk is processed, so events
    /// are yielded to the consumer promptly rather than buffered until the
    /// end of the stream.
    fn drain(&mut self) -> Vec<StreamEvent> {
        std::mem::take(&mut self.pending)
    }

    /// Finalize the stream, emitting the terminal
    /// [`MessageStop`](StreamEvent::MessageStop) if one was started.
    ///
    /// Drains any remaining pending events and appends the stop event.
    /// Safe to call exactly once at the end of the stream; subsequent calls
    /// return an empty vec (the `terminal` stage guards against double-stop).
    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = self.drain();
        let stop_pending = matches!(
            self.terminal,
            TerminalStage::Pending | TerminalStage::FinishReasonSeen
        );
        if self.started && stop_pending {
            self.terminal = match self.terminal {
                TerminalStage::FinishReasonSeen => TerminalStage::Terminal,
                _ => TerminalStage::StopEmitted,
            };
            out.push(StreamEvent::MessageStop);
        }
        out
    }

    /// Push an event onto the internal pending queue.
    ///
    /// Events are held until [`drain`](Self::drain) is called. This is the
    /// single write point — all extractors (`extract_text`,
    /// `extract_function_call`, `extract_finish_reason`) and the lifecycle
    /// methods (`process_chunk`, `finish`) funnel through here.
    fn push(&mut self, ev: StreamEvent) {
        self.pending.push(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessagePart, Role, ToolContent};

    #[test]
    fn gemini_terminal_stage_starts_pending() {
        let em = StreamEmitter::default();
        assert!(
            matches!(em.terminal, TerminalStage::Pending),
            "terminal stage must start pending"
        );
        assert!(
            matches!(em.text, PartLane::Closed) && matches!(em.thinking, PartLane::Closed),
            "both content lanes must start closed"
        );
    }

    #[test]
    fn request_body_user_text() {
        let msgs = vec![Message::user("hello")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn request_body_includes_system_instruction() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &msgs,
            Some("be brief"),
            None,
            None,
            &ToolConstraint::None,
            false,
        );

        let sys = &body["systemInstruction"];
        assert!(sys.is_object());
        assert_eq!(sys["parts"][0]["text"], "be brief");
    }

    #[test]
    fn request_body_no_system_instruction_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);
        assert!(body.get("systemInstruction").is_none());
    }

    #[test]
    fn request_body_assistant_maps_to_model_role() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::text("hello")],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);
        assert_eq!(body["contents"][0]["role"], "model");
    }

    #[test]
    fn request_body_user_role() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);
        assert_eq!(body["contents"][0]["role"], "user");
    }

    #[test]
    fn request_body_assistant_tool_call() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: serde_json::json!({"msg": "hi"}),
            }],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "echo");
        assert_eq!(parts[0]["functionCall"]["id"], "call_1");
        assert_eq!(parts[0]["functionCall"]["args"]["msg"], "hi");
    }

    #[test]
    fn request_body_tool_result() {
        let msgs = vec![Message::new(
            Role::User,
            vec![MessagePart::ToolResult {
                call_id: "call_1".into(),
                name: "echo".into(),
                output: ToolContent::from_string("result text"),
                is_error: None,
            }],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "echo");
        assert_eq!(parts[0]["functionResponse"]["id"], "call_1");
        assert_eq!(
            parts[0]["functionResponse"]["response"]["result"],
            "result text"
        );
    }

    #[test]
    fn request_body_function_response_includes_name_and_id() {
        let msgs = vec![Message::new(
            Role::User,
            vec![MessagePart::ToolResult {
                call_id: "fc_99".into(),
                name: "search".into(),
                output: ToolContent::from_string("results here"),
                is_error: None,
            }],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let fr = &body["contents"][0]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "search");
        assert_eq!(fr["id"], "fc_99");
        assert_eq!(fr["response"]["result"], "results here");
    }

    #[test]
    fn request_body_function_response_omits_id_when_empty() {
        let msgs = vec![Message::new(
            Role::User,
            vec![MessagePart::ToolResult {
                call_id: String::new(),
                name: "search".into(),
                output: ToolContent::from_string("ok"),
                is_error: None,
            }],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let fr = &body["contents"][0]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "search");
        assert!(
            fr.get("id").is_none(),
            "id should be omitted when call_id is empty"
        );
    }

    #[test]
    fn request_body_function_call_omits_id_when_empty() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::tool_call("", "search", serde_json::json!({}))],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let fc = &body["contents"][0]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "search");
        assert!(
            fc.get("id").is_none(),
            "id should be omitted when tool-call id is empty"
        );
    }

    #[test]
    fn request_body_includes_tools() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = build_request_body(
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::None,
            false,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        let decls = tools_arr[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls[0]["name"], "search");
        assert_eq!(decls[0]["description"], "Search the web");
        assert_eq!(decls[0]["parameters"]["type"], "object");
    }

    #[test]
    fn request_body_no_tools_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_multiple_messages() {
        let msgs = vec![
            Message::user("hello"),
            Message::new(Role::Assistant, vec![MessagePart::text("hi")]),
            Message::user("bye"),
        ];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
    }

    #[test]
    fn convert_tools_shape() {
        let tools = vec![ToolSchema {
            tool: "calc".into(),
            description: "Calculate".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let out = convert_tools(&tools, false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["name"], "calc");
        assert_eq!(out[0]["description"], "Calculate");
    }

    #[test]
    fn convert_message_text_only() {
        let m = Message::user("hello");
        let v = convert_message(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["parts"][0]["text"], "hello");
    }

    #[test]
    fn convert_message_assistant_role() {
        let m = Message::new(Role::Assistant, vec![MessagePart::text("hi")]);
        let v = convert_message(&m);
        assert_eq!(v["role"], "model");
    }

    #[test]
    fn convert_message_skips_images() {
        let m = Message::new(
            Role::User,
            vec![
                MessagePart::text("look"),
                MessagePart::Image {
                    source: crate::message::ImageSource {
                        encoding: "base64".into(),
                        media_type: "image/png".into(),
                        data: String::new(),
                    },
                },
            ],
        );
        let v = convert_message(&m);
        let parts = v["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1); // only text, image filtered out
    }

    #[test]
    fn builder_requires_api_key() {
        let result = GeminiClient::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_succeeds_with_key() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(client.model(), DEFAULT_MODEL);
    }

    #[test]
    fn builder_custom_model() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .with_model("gemini-1.5-pro")
            .build()
            .unwrap();
        assert_eq!(client.model(), "gemini-1.5-pro");
    }

    #[test]
    fn builder_custom_base_url() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .with_base_url("https://custom.example.com")
            .with_model("gemini-pro")
            .build()
            .unwrap();
        assert_eq!(client.model(), "gemini-pro");
    }

    #[test]
    fn stream_url_does_not_expose_api_key() {
        let client = GeminiClient::builder()
            .with_api_key("secret-key-123")
            .build()
            .unwrap();
        let url = client.stream_url();
        assert!(
            !url.contains("secret-key-123"),
            "API key must not appear in stream URL: {url}"
        );
        assert!(
            !url.contains("key="),
            "URL must not have key= query param: {url}"
        );
    }

    #[test]
    fn generate_url_does_not_expose_api_key() {
        let client = GeminiClient::builder()
            .with_api_key("secret-key-456")
            .build()
            .unwrap();
        let url = client.generate_url();
        assert!(
            !url.contains("secret-key-456"),
            "API key must not appear in generate URL: {url}"
        );
        assert!(
            !url.contains("key="),
            "URL must not have key= query param: {url}"
        );
    }

    #[test]
    fn emitter_first_chunk_emits_message_start() {
        let mut em = StreamEmitter::default();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
        }));
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStart(_)))
        );
    }

    #[test]
    fn emitter_text_delta() {
        let mut em = StreamEmitter::default();
        em.started = true; // skip MessageStart
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "world"}]}}]
        }));
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::IndexedDelta(_)))
        );
    }

    #[test]
    fn emitter_empty_text_ignored() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": ""}]}}]
        }));
        let events = em.drain();
        // Only MessageStart would be here, but we set started=true so
        // no events at all for empty text.
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, StreamEvent::IndexedDelta(_)))
        );
    }

    #[test]
    fn emitter_function_call() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "rust"}}}]}}]
        }));
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PartStart(_)))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::IndexedDelta(_)))
        );
    }

    #[test]
    fn emitter_function_call_includes_id() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"functionCall": {"id": "fc_7", "name": "search", "args": {"q": "rust"}}}]}}]
        }));
        let events = em.drain();
        let start = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::PartStart(ps) => Some(ps),
                _ => None,
            })
            .expect("PartStart");
        match &start.part {
            Some(MessagePart::ToolCall { id, name, .. }) => {
                assert_eq!(id, "fc_7");
                assert_eq!(name, "search");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn emitter_function_call_without_id_defaults_to_empty() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"functionCall": {"name": "search", "args": {}}}]}}]
        }));
        let events = em.drain();
        let start = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::PartStart(ps) => Some(ps),
                _ => None,
            })
            .expect("PartStart");
        match &start.part {
            Some(MessagePart::ToolCall { id, .. }) => assert_eq!(id, ""),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn emitter_thought_part_routes_to_thinking_variant() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "reasoning here",
                        "thought": true
                    }]
                }
            }]
        }));
        let events = em.drain();
        let delta = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::IndexedDelta(d) => Some(d.clone()),
                _ => None,
            })
            .expect("expected at least one IndexedDelta");
        assert!(
            matches!(delta.delta, DeltaPart::Thinking { .. }),
            "thought:true part must route to Thinking, got {:?}",
            delta.delta
        );
        if let DeltaPart::Thinking { text } = delta.delta {
            assert_eq!(text, "reasoning here");
        }
    }

    #[test]
    fn emitter_thought_part_emits_part_start_at_thinking_index() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "hmm",
                        "thought": true
                    }]
                }
            }]
        }));
        let events = em.drain();
        let start = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::PartStart(p) => Some(p.clone()),
                _ => None,
            })
            .expect("expected a PartStart for the thinking lane");
        assert_eq!(
            start.index, THINKING_PART_INDEX,
            "thought part must open at the thinking index"
        );
    }

    #[test]
    fn emitter_thought_part_does_not_open_text_lane() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "private reasoning",
                        "thought": true
                    }]
                }
            }]
        }));
        let events = em.drain();
        // No DeltaPart::Text event must appear for a thought:true part.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                StreamEvent::IndexedDelta(IndexedDelta {
                    delta: DeltaPart::Text { .. },
                    ..
                })
            )),
            "thought:true part must not produce a Text delta"
        );
    }

    #[test]
    fn emitter_text_part_does_not_open_thinking_lane() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "visible answer"
                    }]
                }
            }]
        }));
        let events = em.drain();
        assert!(
            !events.iter().any(|e| matches!(
                e,
                StreamEvent::IndexedDelta(IndexedDelta {
                    delta: DeltaPart::Thinking { .. },
                    ..
                })
            )),
            "thought field absent must not produce a Thinking delta"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamEvent::IndexedDelta(IndexedDelta {
                    delta: DeltaPart::Text { .. },
                    ..
                })
            )),
            "plain text part must produce a Text delta"
        );
    }

    #[test]
    fn emitter_thought_and_text_parts_interleave() {
        // Gemini interleaves thought and text parts in the same parts[] array.
        // When the lane changes the emitter closes the previous lane with a
        // PartStop before opening the next, so the downstream accumulator
        // (single active index) never sees one lane's deltas clobber the
        // other's buffered state.
        let mut em = StreamEmitter::default();
        em.started = true;

        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "step 1", "thought": true},
                        {"text": "answer", "thought": false}
                    ]
                }
            }]
        }));
        let events = em.drain();
        let deltas: Vec<&IndexedDelta> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::IndexedDelta(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 2);
        assert!(
            matches!(deltas[0].delta, DeltaPart::Thinking { ref text } if text == "step 1"),
            "first delta must be the thinking fragment, got {:?}",
            deltas[0].delta
        );
        assert_eq!(deltas[0].index, THINKING_PART_INDEX);
        assert!(
            matches!(deltas[1].delta, DeltaPart::Text { ref text } if text == "answer"),
            "second delta must be the text fragment, got {:?}",
            deltas[1].delta
        );
        assert_eq!(deltas[1].index, TEXT_PART_INDEX);
        // The lane switch between the two deltas must emit exactly one
        // PartStop for the thinking lane before the text PartStart.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::PartStop))
                .count(),
            1,
            "lane switch must close the thinking lane"
        );
    }

    #[test]
    fn emitter_text_to_thought_lane_switch_emits_part_stop() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "visible", "thought": false}]}
            }]
        }));
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "thinking…", "thought": true}]}
            }]
        }));
        let events = em.drain();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::PartStop))
                .count(),
            1,
            "lane switch must close the text lane"
        );
    }

    #[test]
    fn emitter_finish_closes_thinking_lane() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "thinking…", "thought": true}]
                }
            }]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let events = em.drain();
        // The thinking lane was open; finish must close it with a PartStop.
        assert!(
            events.iter().any(|e| matches!(e, StreamEvent::PartStop)),
            "finish must emit PartStop for the open thinking lane"
        );
    }

    #[test]
    fn request_body_includes_thinking_config_when_opted_in() {
        // Opt-in: includeThoughts injected only when the caller asked for it.
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, true);

        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"], true,
            "includeThoughts must be injected when include_thoughts=true"
        );
    }

    #[test]
    fn request_body_omits_thinking_config_by_default() {
        // Default: includeThoughts NOT injected (non-reasoning models reject
        // it with 400 INVALID_ARGUMENT). generationConfig should be absent
        // entirely when there's nothing else to put in it either.
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        assert!(
            body.get("generationConfig").is_none(),
            "generationConfig must be omitted when include_thoughts=false and no response_format"
        );
    }

    #[test]
    fn request_body_thinking_config_composes_with_response_format() {
        // Both thinkingConfig and responseJsonSchema land under generationConfig.
        let msgs = vec![Message::user("hi")];
        let rf = crate::structured::ResponseFormat::new(
            "result",
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        );
        let body = build_request_body(&msgs, None, None, Some(&rf), &ToolConstraint::None, true);

        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            body["generationConfig"]["responseJsonSchema"],
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}})
        );
    }

    #[test]
    fn emitter_finish_reason_end_turn() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let events = em.drain();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::PartStop)));
        let md = events
            .iter()
            .find(|e| matches!(e, StreamEvent::MessageDelta(_)));
        if let Some(StreamEvent::MessageDelta(d)) = md {
            assert_eq!(d.delta.stop_reason.as_deref(), Some("end_turn"));
        } else {
            panic!("expected MessageDelta");
        }
    }

    #[test]
    fn emitter_finish_reason_max_tokens() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "MAX_TOKENS"}]
        }));
        let events = em.drain();
        let md = events
            .iter()
            .find(|e| matches!(e, StreamEvent::MessageDelta(_)));
        if let Some(StreamEvent::MessageDelta(d)) = md {
            assert_eq!(d.delta.stop_reason.as_deref(), Some("max_tokens"));
        } else {
            panic!("expected MessageDelta");
        }
    }

    #[test]
    fn emitter_duplicate_finish_reason_is_noop() {
        // A second finishReason chunk (e.g. from a proxy/gateway that re-emits)
        // must not produce duplicate PartStop / MessageDelta events.
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let first = em.drain();
        let first_deltas = first
            .iter()
            .filter(|e| matches!(e, StreamEvent::MessageDelta(_)))
            .count();
        let first_stops = first
            .iter()
            .filter(|e| matches!(e, StreamEvent::PartStop))
            .count();

        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let second = em.drain();

        assert_eq!(first_deltas, 1, "first finishReason emits one MessageDelta");
        assert_eq!(first_stops, 1, "first finishReason emits one PartStop");
        assert!(
            second.is_empty(),
            "second finishReason must produce no events, got {second:?}"
        );
    }

    #[test]
    fn emitter_function_call_after_text_closes_text_lane() {
        // When a functionCall arrives after the text lane has been streaming,
        // the emitter must close the text lane with a PartStop before opening
        // the tool part. Both reuse TEXT_PART_INDEX; without the close the
        // accumulator would clobber the text buffer with tool state.
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "calling tool", "thought": false}]}
            }]
        }));
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "search", "args": {"q": "rust"}}}]
                }
            }]
        }));
        let events = em.drain();

        // Exactly one PartStop for the text lane (between the text delta and
        // the tool PartStart).
        let stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::PartStop))
            .count();
        assert_eq!(stops, 2, "text lane closed + tool part closed");

        // Ordering: TextDelta → PartStop → tool PartStart.
        let text_delta_idx = events
            .iter()
            .position(|e| {
                matches!(
                    e,
                    StreamEvent::IndexedDelta(IndexedDelta {
                        delta: DeltaPart::Text { .. },
                        ..
                    })
                )
            })
            .expect("Text delta present");
        let stop_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::PartStop))
            .expect("PartStop present");
        let tool_start_idx = events
            .iter()
            .position(|e| {
                matches!(
                    e,
                    StreamEvent::PartStart(PartStart {
                        part: Some(MessagePart::ToolCall { .. }),
                        ..
                    })
                )
            })
            .expect("Tool PartStart present");
        assert!(text_delta_idx < stop_idx, "text delta before PartStop");
        assert!(stop_idx < tool_start_idx, "PartStop before tool PartStart");
    }

    #[test]
    fn emitter_function_call_at_nonzero_part_index() {
        // A functionCall may appear at parts[1] (or later) when Gemini
        // interleaves a thought part with a tool call in the same chunk.
        // The emitter must scan all parts, not just parts[0].
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "reasoning about the call", "thought": true},
                        {"functionCall": {"name": "search", "args": {"q": "rust"}}}
                    ]
                }
            }]
        }));
        let events = em.drain();
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamEvent::PartStart(PartStart {
                    part: Some(MessagePart::ToolCall { name, .. }),
                    ..
                }) if name == "search"
            )),
            "functionCall at parts[1] must be found and emitted"
        );
    }

    #[test]
    fn emitter_no_function_call_does_nothing() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "just text"},
                        {"text": "more text"}
                    ]
                }
            }]
        }));
        let events = em.drain();
        assert!(
            !events.iter().any(|e| matches!(
                e,
                StreamEvent::PartStart(PartStart {
                    part: Some(MessagePart::ToolCall { .. }),
                    ..
                })
            )),
            "no functionCall in any part must not emit a tool PartStart"
        );
    }

    #[test]
    fn emitter_finish_emits_message_stop_if_needed() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.terminal = TerminalStage::Pending;

        let events = em.finish();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::MessageStop)));
    }

    #[test]
    fn emitter_finish_noop_if_already_stopped() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.terminal = TerminalStage::StopEmitted;

        let events = em.finish();
        assert!(events.is_empty());
    }

    #[test]
    fn emitter_finish_reason_does_not_suppress_message_stop() {
        // Regression: extract_finish_reason advances the terminal stage,
        // but finish() must still emit MessageStop. Previously both used the
        // same `finished` flag, causing finish() to skip MessageStop after
        // a finishReason chunk.
        let mut em = StreamEmitter::default();
        em.started = true;
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        em.drain();
        assert!(matches!(em.terminal, TerminalStage::FinishReasonSeen));
        let events = em.finish();
        assert!(
            events.iter().any(|e| matches!(e, StreamEvent::MessageStop)),
            "MessageStop must be emitted even after finishReason was processed"
        );
    }

    #[test]
    fn emitter_finish_reason_after_finish_advances_to_terminal() {
        // A finishReason arriving after finish() still emits its own
        // MessageDelta, but a *second* such chunk must be a no-op once the
        // terminal stage reaches Terminal.
        let mut em = StreamEmitter::default();
        em.started = true;
        let stop_events = em.finish();
        assert!(
            stop_events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStop)),
            "first finish must emit MessageStop"
        );
        assert!(matches!(em.terminal, TerminalStage::StopEmitted));

        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        assert!(
            matches!(em.terminal, TerminalStage::Terminal),
            "late finishReason after finish must advance to Terminal"
        );
        let late_events = em.drain();
        assert!(
            late_events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageDelta(_))),
            "the first late finishReason still emits its MessageDelta"
        );

        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let duplicate_events = em.drain();
        assert!(
            duplicate_events.is_empty(),
            "a second late finishReason must be a no-op once Terminal is reached"
        );
    }

    #[test]
    fn sse_reader_take_line_extracts_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\n".into(),
        };
        assert_eq!(reader.take_line().unwrap().unwrap(), "data: hello");
        assert!(reader.buf.is_empty());
    }

    #[test]
    fn sse_reader_take_line_none_without_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "partial".into(),
        };
        assert!(reader.take_line().unwrap().is_none());
    }

    #[test]
    fn sse_reader_take_line_multiple() {
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
        assert_eq!(reader.take_line().unwrap().unwrap(), "data: hi");
    }

    #[test]
    fn builder_timeouts_applied_on_build() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .with_timeout(Duration::from_mins(3))
            .with_connect_timeout(Duration::from_secs(15))
            .build();
        assert!(client.is_ok(), "build should succeed with valid timeouts");
    }

    #[test]
    fn builder_include_thoughts_defaults_false() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .build()
            .unwrap();
        assert!(
            !client.include_thoughts,
            "include_thoughts must default to false (non-reasoning models reject thinkingConfig)"
        );
    }

    #[test]
    fn builder_include_thoughts_true_propagates_to_client() {
        let client = GeminiClient::builder()
            .with_api_key("test-key")
            .with_include_thoughts(true)
            .build()
            .unwrap();
        assert!(
            client.include_thoughts,
            "include_thoughts(true) must propagate to the built client"
        );
    }

    #[tokio::test]
    async fn sse_reader_take_line_splits_on_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: {\"candidates\":[]}\n\n".to_string().into_bytes(),
        };
        assert_eq!(
            reader.take_line().unwrap(),
            Some("data: {\"candidates\":[]}".to_string())
        );
        assert_eq!(reader.take_line().unwrap(), Some(String::new()));
        assert_eq!(reader.take_line().unwrap(), None);
    }

    #[tokio::test]
    async fn sse_reader_next_data_extracts_payload() {
        let chunk = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n";
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(chunk.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        let result = reader.next_gemini_data().await.unwrap();
        assert!(result.is_some());
        let json = result.unwrap();
        assert!(json["candidates"].is_array());
    }

    #[tokio::test]
    async fn sse_reader_next_data_malformed_returns_none() {
        let chunk = "data: not valid json\n\ndata: {\"ok\":true}\n\n";
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(chunk.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        // First call should skip malformed and return the valid one.
        let result = reader.next_gemini_data().await.unwrap();
        assert!(result.is_some());
        let json = result.unwrap();
        assert_eq!(json["ok"], true);
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        let huge = "x".repeat(2 * 1024 * 1024);
        let stream = futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(huge.into())]);
        let mut reader = super::SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
        };
        let result = reader.next_gemini_data().await;
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
    fn request_body_response_format_injects_generation_config() {
        let msgs = vec![Message::user("hi")];
        let rf = crate::structured::ResponseFormat::new(
            "result",
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        );
        let body = build_request_body(&msgs, None, None, Some(&rf), &ToolConstraint::None, false);

        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            body["generationConfig"]["responseJsonSchema"],
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}})
        );
    }

    #[test]
    fn request_body_generation_config_omitted_when_nothing_applies() {
        // Without response_format AND include_thoughts=false, generationConfig
        // has nothing to carry, so it must be absent entirely.
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        assert!(
            body.get("generationConfig").is_none(),
            "generationConfig must be omitted when there's nothing to put in it"
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
        let body = build_request_body(
            &msgs,
            None,
            Some(&[caller_tool]),
            Some(&rf),
            &ToolConstraint::None,
            false,
        );

        assert!(
            body.get("tools").is_none(),
            "tools should be suppressed when response_format is set"
        );
        assert!(body.get("generationConfig").is_some());
    }

    #[test]
    fn extract_structured_from_text_part() {
        let client = GeminiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::assistant(r#"{"tool": "write", "args": {}}"#);
        let value = client.extract_structured(&message);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_prose_falls_back_to_string() {
        let client = GeminiClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::assistant("I cannot produce that.");
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!("I cannot produce that."));
    }

    #[test]
    fn build_response_maps_text_part_and_stop() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hello gemini"}]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.message.role, Role::Assistant);
        assert_eq!(response.message.text_content(), "hello gemini");
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_maps_function_call_part() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {"name": "search", "args": {"q": "rust"}}
                    }]
                },
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 1);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, name, input } => {
                assert_eq!(id, "");
                assert_eq!(name, "search");
                assert_eq!(input, &serde_json::json!({"q": "rust"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_parses_function_call_id() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"id": "fc_42", "name": "search", "args": {}}}]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, name, .. } => {
                assert_eq!(id, "fc_42");
                assert_eq!(name, "search");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_function_call_without_id_defaults_to_empty() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "search", "args": {}}}]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, .. } => assert_eq!(id, ""),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_handles_text_and_function_call_in_one_part() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "Let me search",
                        "functionCall": {"name": "search", "args": {}}
                    }]
                },
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(
            response.message.parts.len(),
            2,
            "text + functionCall in one part should yield two MessageParts"
        );
        assert!(response.message.parts[0].is_text());
        assert!(response.message.parts[1].is_tool_call());
    }

    #[test]
    fn build_response_maps_max_tokens_finish_reason() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "truncated"}]},
                "finishReason": "MAX_TOKENS"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::MaxTokens);
    }

    #[test]
    fn build_response_safety_finish_reason_maps_to_end_turn() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "blocked"}]},
                "finishReason": "SAFETY"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_missing_finish_reason_defaults_to_end_turn() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]}
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_missing_candidates_yields_empty_message() {
        let raw = serde_json::json!({});
        let response = GeminiClient::build_response(&raw);
        assert!(response.message.parts.is_empty());
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_extracts_usage() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 25,
                "candidatesTokenCount": 10,
                "thoughtsTokenCount": 5
            }
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.usage.expect("usage").input_tokens, 25);
        assert_eq!(response.usage.expect("usage").output_tokens, 15);
    }

    #[test]
    fn build_response_usage_without_thoughts() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4
            }
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.usage.expect("usage").input_tokens, 8);
        assert_eq!(response.usage.expect("usage").output_tokens, 4);
    }

    #[test]
    fn build_response_missing_usage_is_none() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert!(response.usage.is_none());
    }

    #[test]
    fn build_response_multiple_function_calls_across_parts() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "first", "args": {}}},
                    {"functionCall": {"name": "second", "args": {"n": 2}}}
                ]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 2);
        match &response.message.parts[0] {
            MessagePart::ToolCall { name, .. } => assert_eq!(name, "first"),
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &response.message.parts[1] {
            MessagePart::ToolCall { name, .. } => assert_eq!(name, "second"),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_function_call_missing_args_defaults_to_empty_object() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "search"}}]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        match &response.message.parts[0] {
            MessagePart::ToolCall { input, .. } => {
                assert_eq!(input, &serde_json::json!({}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_multiple_text_parts_preserve_order() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "hello"},
                    {"text": " world"}
                ]},
                "finishReason": "STOP"
            }]
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 2);
        assert_eq!(response.message.text_content(), "hello world");
    }

    #[test]
    fn build_response_partial_usage_with_only_input() {
        let raw = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 99}
        });
        let response = GeminiClient::build_response(&raw);
        assert_eq!(response.usage.expect("usage").input_tokens, 99);
        assert_eq!(response.usage.expect("usage").output_tokens, 0);
    }

    #[test]
    fn gemini_strict_tightens_parameters() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"msg": {"type": "string"}}
            }),
        }];
        let body = build_request_body(
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::Strict,
            false,
        );

        let decls = body["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        let params = &decls[0]["parameters"];
        assert_eq!(params["additionalProperties"], false);
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "msg");
    }

    #[test]
    fn gemini_none_constraint_unchanged_shape() {
        // Default None: tools emitted as before (no tightening).
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = build_request_body(
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::None,
            false,
        );

        let decls = body["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(
            decls[0]["parameters"],
            serde_json::json!({"type": "object"})
        );
        assert!(decls[0]["parameters"].get("additionalProperties").is_none());
    }

    #[test]
    fn gemini_strict_suppressed_when_response_format_set() {
        // With response_format set, the generationConfig path runs, tools
        // are absent, no tightening.
        let msgs = vec![Message::user("hi")];
        let caller_tool = ToolSchema {
            tool: "read".into(),
            description: "Read".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let rf =
            crate::structured::ResponseFormat::new("result", serde_json::json!({"type": "object"}));
        let body = build_request_body(
            &msgs,
            None,
            Some(&[caller_tool]),
            Some(&rf),
            &ToolConstraint::Strict,
            false,
        );

        assert!(
            body.get("tools").is_none(),
            "tools must be suppressed when response_format is set"
        );
        assert!(body.get("generationConfig").is_some());
    }

    #[test]
    fn gemini_strict_does_not_emit_tool_config() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = build_request_body(
            &msgs,
            None,
            Some(&tools),
            None,
            &ToolConstraint::Strict,
            false,
        );

        assert!(
            body.get("toolConfig").is_none(),
            "toolConfig must not appear under Strict"
        );
    }

    fn system_role_msg(text: &str) -> Message {
        Message::new(Role::System, vec![MessagePart::text(text)])
    }

    #[test]
    fn request_body_system_role_folded_into_system_instruction() {
        // An inline Role::System message must NOT appear in `contents`; its
        // text must be folded into the top-level `systemInstruction` field.
        let msgs = vec![
            Message::user("hello"),
            system_role_msg("stay on task"),
            Message::assistant("working"),
        ];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);

        let contents = body["contents"].as_array().expect("contents is an array");
        assert_eq!(contents.len(), 2, "system-role message is filtered out");
        for c in contents {
            assert_ne!(
                c["role"].as_str().unwrap_or(""),
                "system",
                "no inline system-role entry should be emitted"
            );
        }
        let sys_text = body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .expect("systemInstruction.parts[0].text is a string");
        assert_eq!(sys_text, "stay on task");
    }

    #[test]
    fn request_body_system_role_merges_with_caller_system() {
        let msgs = vec![Message::user("hi"), system_role_msg("reminder")];
        let body = build_request_body(
            &msgs,
            Some("be brief"),
            None,
            None,
            &ToolConstraint::None,
            false,
        );

        let sys_text = body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .expect("systemInstruction.parts[0].text is a string");
        assert!(
            sys_text.starts_with("be brief"),
            "caller system prompt comes first: got {sys_text:?}"
        );
        assert!(
            sys_text.contains("reminder"),
            "folded text is appended: got {sys_text:?}"
        );
        assert!(
            sys_text.contains('\n'),
            "caller prompt and folded text are newline-separated: got {sys_text:?}"
        );
    }

    #[test]
    fn request_body_system_role_preserves_message_order() {
        let msgs = vec![
            Message::user("first"),
            system_role_msg("mid reminder"),
            Message::assistant("second"),
            Message::user("third"),
        ];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None, false);
        let contents = body["contents"].as_array().expect("contents is an array");
        let roles: Vec<&str> = contents
            .iter()
            .map(|c| c["role"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(roles, vec!["user", "model", "user"]);
        assert_eq!(contents[0]["parts"][0]["text"], "first");
        assert_eq!(contents[2]["parts"][0]["text"], "third");
    }

    #[test]
    fn emitter_multiple_function_calls_per_chunk() {
        let mut em = StreamEmitter::default();
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "rust"}}},
                        {"functionCall": {"name": "write", "args": {"path": "/tmp"}}}
                    ]
                }
            }]
        }));
        let events = em.drain();

        let tool_starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::PartStart(PartStart {
                    index,
                    part: Some(MessagePart::ToolCall { name, .. }),
                }) => Some((*index, name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_starts.len(),
            2,
            "both function calls must emit PartStart"
        );
        assert_eq!(tool_starts[0].1, "search");
        assert_eq!(tool_starts[1].1, "write");
        assert_ne!(
            tool_starts[0].0, tool_starts[1].0,
            "each tool call must get a distinct index"
        );

        let stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::PartStop))
            .count();
        assert_eq!(stops, 2, "each tool call must emit its own PartStop");
    }

    #[test]
    fn emitter_mixed_text_tool_text_preserves_order() {
        let mut em = StreamEmitter::default();
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "before"},
                        {"functionCall": {"name": "search", "args": {"q": "rust"}}},
                        {"text": "after"}
                    ]
                }
            }]
        }));
        let events = em.drain();

        let event_types: Vec<&str> = events
            .iter()
            .map(|e| match e {
                StreamEvent::PartStart(p) => {
                    if p.part
                        .as_ref()
                        .is_some_and(crate::message::MessagePart::is_tool_call)
                    {
                        "tool_start"
                    } else {
                        "text_start"
                    }
                }
                StreamEvent::IndexedDelta(IndexedDelta {
                    delta: DeltaPart::Text { .. },
                    ..
                }) => "text_delta",
                StreamEvent::IndexedDelta(IndexedDelta {
                    delta: DeltaPart::InputJson { .. },
                    ..
                }) => "json_delta",
                StreamEvent::PartStop => "stop",
                _ => "other",
            })
            .collect();

        let text_delta_pos = event_types
            .iter()
            .position(|t| *t == "text_delta")
            .expect("text delta");
        let tool_start_pos = event_types
            .iter()
            .position(|t| *t == "tool_start")
            .expect("tool start");
        let json_delta_pos = event_types
            .iter()
            .position(|t| *t == "json_delta")
            .expect("json delta");
        let second_text_pos = event_types
            .iter()
            .rposition(|t| *t == "text_delta")
            .expect("second text delta");

        assert!(
            text_delta_pos < tool_start_pos,
            "first text must come before tool call"
        );
        assert!(
            tool_start_pos < json_delta_pos,
            "tool PartStart must come before InputJson"
        );
        assert!(
            json_delta_pos < second_text_pos,
            "tool call must come before second text"
        );
    }

    #[test]
    fn emitter_finish_closes_open_tool_part() {
        let mut em = StreamEmitter::default();
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "search", "args": {"q": "rust"}}}]
                }
            }]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}]
        }));
        let events = em.drain();

        let has_message_delta = events.iter().any(|e| {
            matches!(
                e,
                StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some(_),
                    },
                    ..
                })
            )
        });
        assert!(has_message_delta, "finish must emit a MessageDelta");
    }

    #[test]
    fn emitter_finish_extracts_usage_metadata() {
        let mut em = StreamEmitter::default();
        em.process_chunk(&serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]}
            }]
        }));
        em.drain();
        em.process_chunk(&serde_json::json!({
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 42,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 5
            }
        }));
        let events = em.drain();

        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage: Some(u), .. }) => Some(*u),
            _ => None,
        });
        let usage = usage.expect("finish must include Usage from usageMetadata");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 12);
    }
}
