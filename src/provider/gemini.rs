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
//!     .api_key("AIza...")
//!     .model("gemini-2.0-flash")
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;

use futures::stream::{Stream, StreamExt};
use reqwest::Response;
use serde_json::Value;
use std::time::Duration;

use crate::api::ApiClient;
use crate::api::error::ApiError;
use crate::message::{Message, MessagePart, Role};
use crate::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamEvent, StreamStopReason,
};
use crate::structured::ToolConstraint;
use crate::structured::tighten_json_schema;
use crate::tool::ToolSchema;

// ==================================================
// Constants
// ==================================================

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const SSE_DATA_PREFIX: &str = "data: ";
const TEXT_PART_INDEX: usize = 0;
const THINKING_PART_INDEX: usize = 1;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_mins(2); // connect + response + body
const MAX_RESPONSE_BODY: usize = 10 * 1024 * 1024; // 10 Mb
const SSE_MAX_BUFFER: usize = 1024 * 1024; // 1 Mb
const MAX_ERROR_BODY: usize = 8 * 1024; // 8 Kb
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ==================================================
// Client
// ==================================================

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
}

impl GeminiClient {
    /// Create a builder for configuring a [`GeminiClient`].
    ///
    /// Returns a [`GeminiClientBuilder`] with sensible defaults. The only
    /// required field is `api_key`; everything else has a production-ready
    /// default. Call `.api_key(...).build()` to finish, or chain additional
    /// setters for custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::provider::GeminiClient;
    ///
    /// let client = GeminiClient::builder()
    ///     .api_key("AI...")
    ///     .model("gemini-2.0-flash")
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
            .api_key(api_key)
            .base_url(base_url)
            .model(model)
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
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let body = build_request_body(
            &messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
        );
        let url = self.stream_url();
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_content(&http, &url, &api_key, &body).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_data().await? {
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
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        let body = build_request_body(
            &messages,
            system.as_deref(),
            tools.as_deref(),
            None,
            &ToolConstraint::None,
        );
        let url = self.generate_url();

        Box::pin(async move {
            let resp = Self::post_content(&self.http, &url, &self.api_key, &body).await?;
            let resp = resp
                .bytes()
                .await
                .map_err(|e| ApiError::http(e.to_string()))?;
            if resp.len() > MAX_RESPONSE_BODY {
                return Err(ApiError::http(format!(
                    "response body too large: {} bytes (max {})",
                    resp.len(),
                    MAX_RESPONSE_BODY
                )));
            }
            serde_json::from_slice::<Value>(&resp).map_err(|e| ApiError::http(e.to_string()))
        })
    }

    fn stream_messages_with_options(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let rf = options.response_format.as_ref();
        let body = build_request_body(
            &messages,
            system.as_deref(),
            tools.as_deref(),
            rf,
            &options.tool_constraint,
        );
        let url = self.stream_url();
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_content(&http, &url, &api_key, &body).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_data().await? {
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
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        let response_format = options.response_format.as_ref();
        let body = build_request_body(
            &messages,
            system.as_deref(),
            tools.as_deref(),
            response_format,
            &options.tool_constraint,
        );
        let url = self.generate_url();
        Box::pin(async move {
            let resp = Self::post_content(&self.http, &url, &self.api_key, &body).await?;
            let resp = resp
                .bytes()
                .await
                .map_err(|e| ApiError::http(e.to_string()))?;
            if resp.len() > MAX_RESPONSE_BODY {
                return Err(ApiError::http(format!(
                    "response body too large: {} bytes (max {})",
                    resp.len(),
                    MAX_RESPONSE_BODY
                )));
            }
            serde_json::from_slice::<Value>(&resp).map_err(|e| ApiError::http(e.to_string()))
        })
    }

    fn extract_structured(&self, raw: &Value) -> Value {
        let Some(text) = raw
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(serde_json::Value::as_str)
        else {
            return raw.clone();
        };
        crate::structured::parse_json_lenient(text)
            .unwrap_or_else(|| Value::String(text.to_string()))
    }
}
// ==================================================
// Builder
// ==================================================

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

    /// The total HTTP request timeout (connect + response + body).
    ///
    /// Bounds the entire request lifecycle. Defaults to 120 seconds.
    timeout: Duration,

    /// The TCP connection establishment timeout (including TLS handshake).
    ///
    /// Separate from the total timeout so a slow-connecting server can be
    /// detected faster. Defaults to 10 seconds.
    connect_timeout: Duration,
}

impl Default for GeminiClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl GeminiClientBuilder {
    /// Set the API key for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is not set.
    /// The key is sent as the `x-goog-api-key` header on every request.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL for API requests.
    ///
    /// Defaults to `https://generativelanguage.googleapis.com/v1beta`.
    /// Override when targeting a proxy or Google AI-compatible endpoint.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
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
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the total request timeout (connect + response + body).
    ///
    /// Defaults to 120 seconds. This bounds the entire HTTP request lifecycle —
    /// a hanging server will be aborted after this duration rather than
    /// blocking the agent loop indefinitely.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the TCP connection establishment timeout.
    ///
    /// Defaults to 10 seconds. This is the maximum time to wait for the TCP
    /// connection (including TLS handshake) to be established.
    #[must_use]
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
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
        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .build()
            .map_err(|e| ApiError::http(e.to_string()))?;

        Ok(GeminiClient {
            http,
            api_key,
            base_url: self.base_url,
            model: std::sync::Mutex::new(self.model),
        })
    }
}

// ==================================================
// Request body construction
// ==================================================

/// Build the JSON request body for the Gemini Generate Content API.
///
/// Unlike OpenAI/Anthropic, Gemini puts the model in the URL, not the
/// request body. Each [`Message`] is serialized via [`convert_message`].
///
/// `generationConfig` is always injected with
/// `thinkingConfig.includeThoughts = true` so reasoning-capable models
/// (Gemini 2.5+) surface thought parts in the streamed response;
/// non-reasoning models ignore the flag.
///
/// Tool-call constraint:
/// - When `response_format` is set, injects `responseMimeType` +
///   `responseJsonSchema` into `generationConfig` and suppresses `tools`;
///   `tool_constraint` is ignored in that case.
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
        generation_config.insert(
            "thinkingConfig".into(),
            serde_json::json!({ "includeThoughts": true }),
        );

        if let Some(rf) = response_format {
            generation_config.insert("responseMimeType".into(), "application/json".into());
            generation_config.insert("responseJsonSchema".into(), rf.schema.clone());
        }

        obj.insert("generationConfig".into(), Value::Object(generation_config));
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
        MessagePart::ToolCall { name, input, .. } => Some(serde_json::json!({
            "functionCall": {
                "name": name,
                "args": input,
            }
        })),
        MessagePart::ToolResult {
            call_id, output, ..
        } => Some(serde_json::json!({
            "functionResponse": {
                "name": call_id,
                "response": {"result": output.to_string()},
            }
        })),
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

// ==================================================
// SSE line reader
// ==================================================

/// Minimal SSE line reader over an HTTP byte stream.
///
/// Buffers raw bytes from the response, splits on newlines, and yields
/// the JSON `data` payload of each SSE event. Gemini SSE uses only
/// `data:` lines (no `event:` type headers).
struct SseReader {
    bytes: Pin<Box<dyn Stream<Item = Result<String, ApiError>> + Send>>,
    buf: String,
}

impl SseReader {
    /// Wrap a streaming HTTP response into an SSE reader.
    ///
    /// Takes the response body's byte stream and converts it into a
    /// line-oriented reader that yields SSE event data. Used by
    /// [`stream_messages`](crate::api::ApiClient::stream_messages) and its
    /// `*_with_options` variant to parse Gemini's streaming
    /// `streamGenerateContent` responses.
    fn from_response(resp: Response) -> Self {
        let bytes = resp.bytes_stream().map(|res| {
            res.map(|b| String::from_utf8_lossy(&b).into_owned())
                .map_err(|e| ApiError::http(e.to_string()))
        });

        Self {
            bytes: Box::pin(bytes),
            buf: String::new(),
        }
    }

    /// Extract the next SSE `data:` payload as parsed JSON.
    ///
    /// Returns `Ok(None)` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails.
    async fn next_data(&mut self) -> Result<Option<Value>, ApiError> {
        loop {
            while let Some(line) = self.take_line() {
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

            match self.bytes.next().await {
                Some(Ok(chunk)) => {
                    self.buf.push_str(&chunk);
                    if self.buf.len() > SSE_MAX_BUFFER {
                        return Err(ApiError::http(format!(
                            "SSE buffer exceeded {SSE_MAX_BUFFER} bytes"
                        )));
                    }
                }
                Some(Err(e)) => return Err(e),
                None => return Ok(None),
            }
        }
    }

    /// Pop the first `\n`-terminated line from the internal buffer.
    ///
    /// Returns the line (trimmed) if a newline is present, and removes it
    /// (plus the newline) from the buffer. Returns `None` if the buffer
    /// does not yet contain a complete line — the caller should wait for
    /// more bytes from the HTTP stream.
    fn take_line(&mut self) -> Option<String> {
        let pos = self.buf.find('\n')?;
        let line = self.buf[..pos].trim().to_string();
        let rest_start = pos.saturating_add(1);
        self.buf = self.buf.get(rest_start..).unwrap_or_default().to_string();

        Some(line)
    }
}

// ==================================================
// Stream event emitter
// ==================================================

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
#[allow(clippy::struct_excessive_bools)]
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
    text_part_open: bool,

    /// Whether the reasoning (thinking) content part is currently open.
    ///
    /// Reasoning models flag thought parts with `thought: true`. The emitter
    /// opens a thinking part on the first non-empty thought fragment and
    /// closes it in [`extract_finish_reason`](Self::extract_finish_reason),
    /// symmetric to the text lane.
    thinking_part_open: bool,

    /// Whether the terminal stop signal has been processed.
    ///
    /// Set by [`finish`](Self::finish) when it appends the synthetic
    /// [`StreamEvent::MessageStop`]. Guards against emitting a second
    /// `MessageStop` if `finish` is called again after the stream ends.
    /// (Note: unlike the OpenAI/Anthropic emitters, Gemini's finish
    /// reason arrives inside a regular data chunk and is handled by
    /// [`extract_finish_reason`](Self::extract_finish_reason); this flag
    /// only governs the final `MessageStop` synthesis.)
    finished: bool,

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

        self.extract_parts(json);
        self.extract_function_call(json);
        self.extract_finish_reason(json);
    }

    /// Extract text and thought deltas from the chunk, routing by the
    /// `thought: true` flag on each part.
    ///
    /// Gemini reasoning models (2.5+) interleave thought parts with regular
    /// text parts in `candidates[0].content.parts[]`. Each part carries a
    /// `thought` boolean: `true` for reasoning (routed to
    /// [`DeltaPart::Thinking`]), absent or `false` for visible text (routed
    /// to [`DeltaPart::Text`]). Both lanes open a [`PartStart`] on the first
    /// non-empty fragment and are closed by
    /// [`extract_finish_reason`](Self::extract_finish_reason).
    ///
    /// The `functionCall` part is skipped here — it's handled by
    /// [`extract_function_call`](Self::extract_function_call).
    fn extract_parts(&mut self, json: &Value) {
        let Some(parts) = json
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        else {
            return;
        };

        for part in parts {
            // Function-call parts are handled by extract_function_call.
            if part.get("functionCall").is_some() {
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
                if !self.thinking_part_open {
                    self.thinking_part_open = true;
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
                if !self.text_part_open {
                    self.text_part_open = true;
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

    /// Extract a function (tool) call from the chunk and emit the
    /// corresponding part-start and input-json events.
    ///
    /// Reads `candidates[0].content.parts[0].functionCall` for the tool
    /// `name` and `args`. Emits a [`PartStart`](StreamEvent::PartStart)
    /// with a [`ToolCall`](crate::message::MessagePart::ToolCall) part,
    /// followed by an [`InputJson`](crate::stream::DeltaPart::InputJson)
    /// delta carrying the serialized arguments. Does nothing if no function
    /// call is present in the chunk.
    fn extract_function_call(&mut self, json: &Value) {
        if let Some(func_call) = json.pointer("/candidates/0/content/parts/0/functionCall") {
            let name = func_call
                .pointer("/name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let args = func_call.pointer("/args").cloned().unwrap_or(Value::Null);
            let args_str = serde_json::to_string(&args).unwrap_or_default();

            self.push(StreamEvent::PartStart(PartStart {
                index: TEXT_PART_INDEX,
                part: Some(MessagePart::ToolCall {
                    id: String::new(),
                    name,
                    input: args,
                }),
            }));
            self.push(StreamEvent::IndexedDelta(IndexedDelta {
                index: TEXT_PART_INDEX,
                delta: DeltaPart::InputJson {
                    partial_json: args_str,
                },
            }));
        }
    }

    /// Extract the finish reason from the chunk and emit stop events.
    ///
    /// Reads `candidates[0].finishReason` (e.g. `"STOP"`, `"MAX_TOKENS"`).
    /// When present, closes any open thinking part and text part with
    /// [`PartStop`](StreamEvent::PartStop), then emits a
    /// [`MessageDelta`](StreamEvent::MessageDelta) carrying the mapped
    /// [`StreamStopReason`]. Does nothing if the chunk has no finish reason.
    fn extract_finish_reason(&mut self, json: &Value) {
        let Some(reason) = json
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        else {
            return;
        };
        let stop = match reason {
            "MAX_TOKENS" => StreamStopReason::MaxTokens,
            _ => StreamStopReason::EndTurn,
        };

        if self.thinking_part_open {
            self.push(StreamEvent::PartStop);
        }

        if self.text_part_open {
            self.push(StreamEvent::PartStop);
        }

        self.push(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop.to_api_str().into()),
            },
            usage: None,
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
    /// return an empty vec (the `finished` flag guards against double-stop).
    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = self.drain();
        if self.started && !self.finished {
            self.finished = true;
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

// ==================================================
// Tests
// ==================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessagePart, Role, ToolContent};

    #[test]
    fn request_body_user_text() {
        let msgs = vec![Message::user("hello")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn request_body_includes_system_instruction() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, Some("be brief"), None, None, &ToolConstraint::None);

        let sys = &body["systemInstruction"];
        assert!(sys.is_object());
        assert_eq!(sys["parts"][0]["text"], "be brief");
    }

    #[test]
    fn request_body_no_system_instruction_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
        assert!(body.get("systemInstruction").is_none());
    }

    #[test]
    fn request_body_assistant_maps_to_model_role() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::text("hello")],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
        assert_eq!(body["contents"][0]["role"], "model");
    }

    #[test]
    fn request_body_user_role() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
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
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "echo");
        assert_eq!(parts[0]["functionCall"]["args"]["msg"], "hi");
    }

    #[test]
    fn request_body_tool_result() {
        let msgs = vec![Message::new(
            Role::User,
            vec![MessagePart::ToolResult {
                call_id: "call_1".into(),
                output: ToolContent::from_string("result text"),
                is_error: None,
            }],
        )];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "call_1");
        assert_eq!(
            parts[0]["functionResponse"]["response"]["result"],
            "result text"
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
        let body = build_request_body(&msgs, None, Some(&tools), None, &ToolConstraint::None);

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
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_multiple_messages() {
        let msgs = vec![
            Message::user("hello"),
            Message::new(Role::Assistant, vec![MessagePart::text("hi")]),
            Message::user("bye"),
        ];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

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
        let client = GeminiClient::builder().api_key("test-key").build().unwrap();
        assert_eq!(client.model(), DEFAULT_MODEL);
    }

    #[test]
    fn builder_custom_model() {
        let client = GeminiClient::builder()
            .api_key("test-key")
            .model("gemini-1.5-pro")
            .build()
            .unwrap();
        assert_eq!(client.model(), "gemini-1.5-pro");
    }

    #[test]
    fn builder_custom_base_url() {
        let client = GeminiClient::builder()
            .api_key("test-key")
            .base_url("https://custom.example.com")
            .model("gemini-pro")
            .build()
            .unwrap();
        assert_eq!(client.model(), "gemini-pro");
    }

    #[test]
    fn stream_url_does_not_expose_api_key() {
        let client = GeminiClient::builder()
            .api_key("secret-key-123")
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
            .api_key("secret-key-456")
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
    fn request_body_includes_thinking_config() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"], true,
            "includeThoughts must be injected into generationConfig"
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
        let body = build_request_body(&msgs, None, None, Some(&rf), &ToolConstraint::None);

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
    fn emitter_finish_emits_message_stop_if_needed() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.finished = false;

        let events = em.finish();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::MessageStop)));
    }

    #[test]
    fn emitter_finish_noop_if_already_stopped() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.finished = true;

        let events = em.finish();
        assert!(events.is_empty());
    }

    #[test]
    fn sse_reader_take_line_extracts_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\n".into(),
        };
        assert_eq!(reader.take_line().unwrap(), "data: hello");
        assert!(reader.buf.is_empty());
    }

    #[test]
    fn sse_reader_take_line_none_without_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "partial".into(),
        };
        assert!(reader.take_line().is_none());
    }

    #[test]
    fn sse_reader_take_line_multiple() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "line1\nline2\n".into(),
        };
        assert_eq!(reader.take_line().unwrap(), "line1");
        assert_eq!(reader.take_line().unwrap(), "line2");
    }

    #[test]
    fn sse_reader_take_line_trims_cr() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hi\r\n".into(),
        };
        assert_eq!(reader.take_line().unwrap(), "data: hi");
    }

    #[test]
    fn builder_has_default_timeouts() {
        // The builder should initialize with sensible non-zero defaults
        // so that a hanging server cannot block the agent loop indefinitely.
        let builder = GeminiClientBuilder::default();
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_timeout() {
        let custom = Duration::from_mins(5);
        let builder = GeminiClientBuilder::default().timeout(custom);
        assert_eq!(builder.timeout, custom);
        // connect_timeout should be unchanged.
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_connect_timeout() {
        let custom = Duration::from_secs(45);
        let builder = GeminiClientBuilder::default().connect_timeout(custom);
        assert_eq!(builder.connect_timeout, custom);
        // timeout should be unchanged.
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn builder_custom_both_timeouts() {
        let req_timeout = Duration::from_mins(10);
        let conn_timeout = Duration::from_secs(30);
        let builder = GeminiClientBuilder::default()
            .timeout(req_timeout)
            .connect_timeout(conn_timeout);
        assert_eq!(builder.timeout, req_timeout);
        assert_eq!(builder.connect_timeout, conn_timeout);
    }

    #[test]
    fn builder_timeouts_applied_on_build() {
        let client = GeminiClient::builder()
            .api_key("test-key")
            .timeout(Duration::from_mins(3))
            .connect_timeout(Duration::from_secs(15))
            .build();
        assert!(client.is_ok(), "build should succeed with valid timeouts");
    }

    #[tokio::test]
    async fn sse_reader_take_line_splits_on_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: {\"candidates\":[]}\n\n".to_string(),
        };
        assert_eq!(
            reader.take_line(),
            Some("data: {\"candidates\":[]}".to_string())
        );
        assert_eq!(reader.take_line(), Some(String::new()));
        assert_eq!(reader.take_line(), None);
    }

    #[tokio::test]
    async fn sse_reader_next_data_extracts_payload() {
        let chunk = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(chunk.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_data().await.unwrap();
        assert!(result.is_some());
        let json = result.unwrap();
        assert!(json["candidates"].is_array());
    }

    #[tokio::test]
    async fn sse_reader_next_data_malformed_returns_none() {
        let chunk = "data: not valid json\n\ndata: {\"ok\":true}\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(chunk.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        // First call should skip malformed and return the valid one.
        let result = reader.next_data().await.unwrap();
        assert!(result.is_some());
        let json = result.unwrap();
        assert_eq!(json["ok"], true);
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        let huge = "x".repeat(SSE_MAX_BUFFER + 1);
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(huge)]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_data().await;
        assert!(result.is_err(), "should error on buffer overflow");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SSE buffer"),
            "error should mention SSE buffer: {err_msg}"
        );
    }

    #[test]
    fn max_response_body_is_ten_mb() {
        assert_eq!(MAX_RESPONSE_BODY, 10 * 1024 * 1024);
    }

    #[test]
    fn request_body_response_format_injects_generation_config() {
        let msgs = vec![Message::user("hi")];
        let rf = crate::structured::ResponseFormat::new(
            "result",
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        );
        let body = build_request_body(&msgs, None, None, Some(&rf), &ToolConstraint::None);

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
    fn request_body_response_format_absent_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
        let gc = body
            .get("generationConfig")
            .expect("generationConfig must be present for thinkingConfig");
        assert!(
            gc.get("responseMimeType").is_none(),
            "responseMimeType must be absent without response_format"
        );
        assert!(
            gc.get("responseJsonSchema").is_none(),
            "responseJsonSchema must be absent without response_format"
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
        );

        assert!(
            body.get("tools").is_none(),
            "tools should be suppressed when response_format is set"
        );
        assert!(body.get("generationConfig").is_some());
    }

    #[test]
    fn extract_structured_from_text_field() {
        let client = GeminiClient::builder().api_key("test").build().unwrap();
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": r#"{"tool": "write", "args": {}}"#
                    }]
                }
            }]
        });
        let value = client.extract_structured(&raw);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_prose_falls_back_to_raw() {
        let client = GeminiClient::builder().api_key("test").build().unwrap();
        let raw = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "I cannot produce that."}]
                }
            }]
        });
        let value = client.extract_structured(&raw);
        // Prose text not parseable as JSON → falls back to the string value.
        assert_eq!(value, serde_json::json!("I cannot produce that."));
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
        let body = build_request_body(&msgs, None, Some(&tools), None, &ToolConstraint::Strict);

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
        let body = build_request_body(&msgs, None, Some(&tools), None, &ToolConstraint::None);

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
        let body = build_request_body(&msgs, None, Some(&tools), None, &ToolConstraint::Strict);

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
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);

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
        let body = build_request_body(&msgs, Some("be brief"), None, None, &ToolConstraint::None);

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
        let body = build_request_body(&msgs, None, None, None, &ToolConstraint::None);
        let contents = body["contents"].as_array().expect("contents is an array");
        let roles: Vec<&str> = contents
            .iter()
            .map(|c| c["role"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(roles, vec!["user", "model", "user"]);
        assert_eq!(contents[0]["parts"][0]["text"], "first");
        assert_eq!(contents[2]["parts"][0]["text"], "third");
    }
}
