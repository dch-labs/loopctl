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
use crate::tool::ToolSchema;

// ==================================================
// Constants
// ==================================================

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const SSE_DATA_PREFIX: &str = "data: ";
const TEXT_PART_INDEX: usize = 0;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120); // connect + response + body
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
    model: parking_lot::Mutex<String>,
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
        let model = self.model.lock().clone();
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
        let model = self.model.lock().clone();
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
        self.model.lock().clone()
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn set_model(&self, model: &str) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        *self.model.lock() = model.to_string();
        true
    }

    fn stream_messages(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let body = build_request_body(&messages, system.as_deref(), tools.as_deref(), None);
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
        let body = build_request_body(&messages, system.as_deref(), tools.as_deref(), None);
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
        let body = build_request_body(&messages, system.as_deref(), tools.as_deref(), rf);
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
            model: parking_lot::Mutex::new(self.model),
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
fn build_request_body(
    messages: &[Message],
    system: Option<&str>,
    tools: Option<&[ToolSchema]>,
    response_format: Option<&crate::structured::ResponseFormat>,
) -> Value {
    let contents: Vec<Value> = messages.iter().map(convert_message).collect();
    let mut body = serde_json::json!({ "contents": contents });
    if let Some(obj) = body.as_object_mut() {
        if let Some(sys) = system {
            obj.insert(
                "systemInstruction".into(),
                serde_json::json!({"parts": [{"text": sys}]}),
            );
        }

        if response_format.is_none() {
            if let Some(tool_list) = tools {
                obj.insert(
                    "tools".into(),
                    serde_json::json!([{"functionDeclarations": convert_tools(tool_list)}]),
                );
            }
        }

        if let Some(rf) = response_format {
            obj.insert(
                "generationConfig".into(),
                serde_json::json!({
                    "responseMimeType": "application/json",
                    "responseJsonSchema": rf.schema,
                }),
            );
        }
    }

    body
}

/// Convert a single framework [`Message`] into the Gemini JSON shape.
///
/// Gemini uses `role: "user"` / `role: "model"` (not "assistant") and
/// a `parts` array for content blocks.
fn convert_message(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
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
/// structured output is active (`response_format` set), this function is not
/// called — [`build_request_body`] injects `generationConfig.responseJsonSchema`
/// instead, and `tools` is suppressed.
fn convert_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.tool,
                "description": &t.description,
                "parameters": t.input_schema.clone(),
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
/// stop reason.
#[derive(Default)]
struct StreamEmitter {
    started: bool,
    finished: bool,
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

        self.extract_text(json);
        self.extract_function_call(json);
        self.extract_finish_reason(json);
    }

    /// Extract a text delta from the chunk and emit an `IndexedDelta` event.
    ///
    /// Reads `candidates[0].content.parts[0].text`. If the text is non-empty,
    /// pushes a [`DeltaPart::Text`](crate::stream::DeltaPart::Text) event at
    /// index 0. Does nothing if the path is absent or the text is empty.
    fn extract_text(&mut self, json: &Value) {
        if let Some(text) = json
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str)
        {
            if !text.is_empty() {
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
    /// When present, emits [`PartStop`](StreamEvent::PartStop) followed by a
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

        self.push(StreamEvent::PartStop);
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
        let body = build_request_body(&msgs, None, None, None);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn request_body_includes_system_instruction() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, Some("be brief"), None, None);

        let sys = &body["systemInstruction"];
        assert!(sys.is_object());
        assert_eq!(sys["parts"][0]["text"], "be brief");
    }

    #[test]
    fn request_body_no_system_instruction_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None);
        assert!(body.get("systemInstruction").is_none());
    }

    #[test]
    fn request_body_assistant_maps_to_model_role() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::text("hello")],
        )];
        let body = build_request_body(&msgs, None, None, None);
        assert_eq!(body["contents"][0]["role"], "model");
    }

    #[test]
    fn request_body_user_role() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(&msgs, None, None, None);
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
        let body = build_request_body(&msgs, None, None, None);

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
        let body = build_request_body(&msgs, None, None, None);

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
        let body = build_request_body(&msgs, None, Some(&tools), None);

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
        let body = build_request_body(&msgs, None, None, None);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_multiple_messages() {
        let msgs = vec![
            Message::user("hello"),
            Message::new(Role::Assistant, vec![MessagePart::text("hi")]),
            Message::user("bye"),
        ];
        let body = build_request_body(&msgs, None, None, None);

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
        let out = convert_tools(&tools);
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
    fn emitter_finish_reason_end_turn() {
        let mut em = StreamEmitter::default();
        em.started = true;
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
        let custom = Duration::from_secs(300);
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
        let req_timeout = Duration::from_secs(600);
        let conn_timeout = Duration::from_secs(30);
        let builder = GeminiClientBuilder::default()
            .timeout(req_timeout)
            .connect_timeout(conn_timeout);
        assert_eq!(builder.timeout, req_timeout);
        assert_eq!(builder.connect_timeout, conn_timeout);
    }

    #[test]
    fn builder_timeouts_applied_on_build() {
        // Verify the build succeeds — reqwest validates the configuration
        // internally. If timeout/connect_timeout were somehow invalid,
        // .build() would return an error.
        let client = GeminiClient::builder()
            .api_key("test-key")
            .timeout(Duration::from_secs(180))
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
        // Malformed JSON data should be logged (H4) and the reader
        // continues looking for the next valid data line.
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
        let body = build_request_body(&msgs, None, None, Some(&rf));

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
        let body = build_request_body(&msgs, None, None, None);
        assert!(
            body.get("generationConfig").is_none(),
            "generationConfig should be absent when no response_format"
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
        let body = build_request_body(&msgs, None, Some(&[caller_tool]), Some(&rf));

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
}
