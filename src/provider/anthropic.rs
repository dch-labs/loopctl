//! Anthropic Messages API client.
//!
//! Implements [`ApiClient`] by translating between the framework's
//! [`StreamEvent`] protocol and the Anthropic Messages SSE format.
//!
//! # Construction
//!
//! ```rust,ignore
//! use loopctl::provider::AnthropicClient;
//!
//! // From environment (ANTHROPIC_API_KEY):
//! let client = AnthropicClient::from_env()?;
//!
//! // Explicit:
//! let client = AnthropicClient::builder()
//!     .api_key("sk-ant-...")
//!     .model("claude-sonnet-4-20250514")
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
    PartStart, StreamEvent, StreamStopReason, Usage,
};
use crate::tool::ToolSchema;

// ==================================================
// Constants
// ==================================================
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const SSE_EVENT_PREFIX: &str = "event: ";
const SSE_DATA_PREFIX: &str = "data: ";
const TEXT_PART_INDEX: usize = 0;
const DEFAULT_MAX_TOKENS: u32 = 8192;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120); // connect + response + body
const MAX_RESPONSE_BODY: usize = 10 * 1024 * 1024; // 10 Mb
const SSE_MAX_BUFFER: usize = 1024 * 1024; // 1 Mb
/// Maximum bytes to read from an error response body.  Prevents OOM when a
/// misconfigured or malicious server returns a multi-GB body on a 4xx/5xx.
const MAX_ERROR_BODY: usize = 8 * 1024; // 8 Kb
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ==================================================
// Client
// ==================================================

/// An Anthropic Claude chat client with streaming support.
///
/// Implements [`ApiClient`] by translating between the framework's
/// [`StreamEvent`] protocol and the Anthropic Messages SSE format.
///
/// Also works with Anthropic-compatible endpoints such as `Z.ai`
/// — use a custom `base_url` via [`AnthropicClientBuilder::base_url`].
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: parking_lot::Mutex<String>,
    max_tokens: u32,
}

impl AnthropicClient {
    /// Create a builder for configuring an [`AnthropicClient`].
    #[must_use]
    pub fn builder() -> AnthropicClientBuilder {
        AnthropicClientBuilder::default()
    }

    /// Create from environment variables.
    ///
    /// Reads:
    /// - `ANTHROPIC_API_KEY` — required.
    /// - `ANTHROPIC_BASE_URL` — optional, defaults to `https://api.anthropic.com`.
    /// - `ANTHROPIC_MODEL` — optional, defaults to `claude-sonnet-4-20250514`.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key is found.
    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ApiError::auth_invalid_key("ANTHROPIC_API_KEY not set"))?;
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());

        Self::builder()
            .api_key(api_key)
            .base_url(base_url)
            .model(model)
            .build()
    }

    /// Send a POST request to the Messages endpoint.
    ///
    /// Shared by both [`ApiClient::stream_messages`] and
    /// [`ApiClient::create_message`].
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the request fails or the server
    /// responds with a non-success status code.
    async fn post_messages(
        http: &reqwest::Client,
        url: &str,
        api_key: &str,
        body: &Value,
    ) -> Result<reqwest::Response, ApiError> {
        let resp = http
            .post(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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

    /// Build the Messages API URL for this client.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }
}

impl ApiClient for AnthropicClient {
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
        let model = self.model.lock().clone();
        let body = build_request_body(
            &model,
            &messages,
            system.as_deref(),
            tools.as_deref(),
            true,
            self.max_tokens,
        );
        let url = self.messages_url();
        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_messages(&http, &url, &api_key, &body).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some((event_type, data)) = sse.next_event().await? {
                emitter.process_event(&event_type, data);
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
        let model = self.model.lock().clone();
        let body = build_request_body(
            &model,
            &messages,
            system.as_deref(),
            tools.as_deref(),
            false,
            self.max_tokens,
        );
        let url = self.messages_url();
        Box::pin(async move {
            let resp = Self::post_messages(&self.http, &url, &self.api_key, &body).await?;
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
}

// ==================================================
// Builder
// ==================================================

/// Builder for [`AnthropicClient`].
pub struct AnthropicClientBuilder {
    api_key: Option<String>,
    base_url: String,
    model: String,
    max_tokens: u32,
    timeout: Duration,
    connect_timeout: Duration,
}

impl Default for AnthropicClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl AnthropicClientBuilder {
    /// Set the API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL (e.g. `https://api.z.ai/api/anthropic`).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model name (e.g. `claude-sonnet-4-20250514`).
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the maximum output tokens per response.
    ///
    /// Anthropic requires this field. Defaults to 8192.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
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
    pub fn build(self) -> Result<AnthropicClient, ApiError> {
        let api_key = self
            .api_key
            .ok_or_else(|| ApiError::auth_invalid_key("API key not provided"))?;
        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .build()
            .map_err(|e| ApiError::http(e.to_string()))?;

        Ok(AnthropicClient {
            http,
            api_key,
            base_url: self.base_url,
            model: parking_lot::Mutex::new(self.model),
            max_tokens: self.max_tokens,
        })
    }
}

// ==================================================
// Request body construction
// ==================================================

/// Build the JSON request body for the Anthropic Messages API.
///
/// Each [`Message`] is serialized via [`convert_message`], then assembled
/// with the model, `max_tokens`, system prompt, and optional tools.
fn build_request_body(
    model: &str,
    messages: &[Message],
    system: Option<&str>,
    tools: Option<&[ToolSchema]>,
    stream: bool,
    max_tokens: u32,
) -> Value {
    let msgs: Vec<Value> = messages.iter().map(convert_message).collect();

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": msgs,
        "system": system.unwrap_or(""),
        "stream": stream,
        "tools": tools.map(convert_tools),
    });

    // Remove `tools` if None so we don't send a null field.
    if tools.is_none() {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("tools");
        }
    }

    body
}

/// Convert a single framework [`Message`] into the Anthropic JSON shape.
///
/// - Messages with only a single text part use a plain string for `content`
///   (Anthropic's recommended optimization).
/// - Messages with tool calls or tool results use the full `content` array.
fn convert_message(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    // Bucket parts by Anthropic category.
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();

    for p in &m.parts {
        match p {
            MessagePart::Text { text } => text_parts.push(text.as_str()),
            MessagePart::ToolCall { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                }));
            }
            MessagePart::ToolResult {
                call_id, output, ..
            } => {
                tool_results.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output.to_string(),
                }));
            }
            MessagePart::Image { .. } => {}
        }
    }

    let has_tool_content = !(tool_calls.is_empty() && tool_results.is_empty());

    if !has_tool_content && text_parts.len() == 1 {
        // Single text — Anthropic allows plain string content.
        let text = text_parts.first().copied().unwrap_or_default();
        serde_json::json!({ "role": role, "content": text })
    } else if !has_tool_content {
        // Multiple text parts — array of text blocks.
        let blocks: Vec<Value> = text_parts
            .iter()
            .map(|t| serde_json::json!({"type": "text", "text": t}))
            .collect();
        serde_json::json!({ "role": role, "content": blocks })
    } else {
        // Mixed content — combine text + tool blocks in a single array.
        let mut blocks: Vec<Value> = Vec::new();

        if !text_parts.is_empty() {
            let text = text_parts.join("");
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
        blocks.extend(tool_calls);
        blocks.extend(tool_results);

        serde_json::json!({ "role": role, "content": blocks })
    }
}

/// Convert tool schemas into the Anthropic `tools` array shape.
fn convert_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.tool,
                "description": &t.description,
                "input_schema": t.input_schema.clone(),
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
/// `(event_type, data)` pairs. Anthropic SSE uses separate `event:` and
/// `data:` lines for each event.
struct SseReader {
    bytes: Pin<Box<dyn Stream<Item = Result<String, ApiError>> + Send>>,
    buf: String,
}

impl SseReader {
    /// Wrap a streaming HTTP response.
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

    /// Extract the next SSE event as `(event_type, data_json)`.
    ///
    /// Returns `Ok(None)` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails.
    async fn next_event(&mut self) -> Result<Option<(String, Option<Value>)>, ApiError> {
        let mut event_type = String::new();
        let mut data = String::new();
        let mut have_event = false;

        loop {
            while let Some(line) = self.take_line() {
                if line.is_empty() {
                    // Blank line = event boundary. Emit if we have one.
                    if have_event {
                        let parsed = if data.is_empty() {
                            None
                        } else {
                            match serde_json::from_str(&data) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        event_type = %event_type,
                                        data_len = data.len(),
                                        "failed to parse Anthropic SSE data, skipping"
                                    );
                                    None
                                }
                            }
                        };
                        return Ok(Some((event_type, parsed)));
                    }
                    continue;
                }

                if let Some(ev) = line.strip_prefix(SSE_EVENT_PREFIX) {
                    event_type = ev.into();
                    have_event = true;
                } else if let Some(d) = line.strip_prefix(SSE_DATA_PREFIX) {
                    // Per the SSE specification, multiple consecutive `data:`
                    // lines must be concatenated with `\n` to form a single
                    // event payload.  Using assignment here would silently
                    // discard earlier data lines.
                    if data.is_empty() {
                        data = d.into();
                    } else {
                        data.push('\n');
                        data.push_str(d);
                    }
                    have_event = true;
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
                None => {
                    // End of stream — emit any pending event.
                    if have_event {
                        let parsed = if data.is_empty() {
                            None
                        } else {
                            match serde_json::from_str(&data) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        event_type = %event_type,
                                        data_len = data.len(),
                                        "failed to parse Anthropic SSE data (stream end), skipping"
                                    );
                                    None
                                }
                            }
                        };
                        return Ok(Some((event_type, parsed)));
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// Pop the first `\n`-terminated line from the buffer, if present.
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

/// Stateful translator that converts Anthropic SSE events into
/// [`StreamEvent`]s.
///
/// Encapsulates all protocol-level bookkeeping:
/// - Emitting [`MessageStart`] once.
/// - Emitting [`PartStart`] / [`IndexedDelta`] for text and tool-call content.
/// - Emitting [`PartStop`] when parts finish.
/// - Emitting the final [`MessageDelta`] with stop reason and usage.
#[derive(Default)]
struct StreamEmitter {
    started: bool,
    text_part_open: bool,
    tool_parts_open: usize,
    /// Index of the currently open tool block (from the API's `content_block_start`).
    current_tool_index: Option<usize>,
    finished: bool,
    pending: Vec<StreamEvent>,
}

impl StreamEmitter {
    /// Process a single SSE event, appending events to the internal queue.
    fn process_event(&mut self, event_type: &str, data: Option<Value>) {
        match event_type {
            "message_start" => self.on_message_start(data.as_ref()),
            "content_block_start" => self.on_block_start(data),
            "content_block_delta" => self.on_block_delta(data),
            "content_block_stop" => self.on_block_stop(data),
            "message_delta" => self.on_message_delta(data),
            "message_stop" => self.on_message_stop(),
            _ => {}
        }
    }

    fn on_message_start(&mut self, data: Option<&Value>) {
        if self.started {
            return;
        }
        self.started = true;

        let (id, model) = match data {
            Some(v) => (
                v.pointer("/message/id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                v.pointer("/message/model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            None => (String::new(), String::new()),
        };

        self.push(StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id,
                role: "assistant".into(),
                model,
            },
        }));
    }

    fn on_block_start(&mut self, data: Option<Value>) {
        let Some(v) = data else { return };
        let block_type = v.pointer("/content_block/type").and_then(Value::as_str);
        let index = v
            .pointer("/index")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);

        match block_type {
            Some("tool_use") => {
                let id = v
                    .pointer("/content_block/id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = v
                    .pointer("/content_block/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();

                self.push(StreamEvent::PartStart(PartStart {
                    index,
                    part: Some(MessagePart::ToolCall {
                        id,
                        name,
                        input: Value::Null,
                    }),
                }));
                self.current_tool_index = Some(index);
                self.tool_parts_open = self.tool_parts_open.saturating_add(1);
            }
            Some("text") => {
                self.text_part_open = true;
                self.push(StreamEvent::PartStart(PartStart {
                    index: TEXT_PART_INDEX,
                    part: Some(MessagePart::text("")),
                }));
            }
            _ => {}
        }
    }

    fn on_block_delta(&mut self, data: Option<Value>) {
        let Some(v) = data else { return };
        let delta_type = v.pointer("/delta/type").and_then(Value::as_str);

        match delta_type {
            Some("text_delta") => {
                let text = v
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !text.is_empty() {
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: TEXT_PART_INDEX,
                        delta: DeltaPart::Text { text },
                    }));
                }
            }
            Some("input_json_delta") => {
                let json = v
                    .pointer("/delta/partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !json.is_empty() {
                    // Use the index from the corresponding content_block_start.
                    let tool_index = self.current_tool_index.unwrap_or(TEXT_PART_INDEX);
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: tool_index,
                        delta: DeltaPart::InputJson { partial_json: json },
                    }));
                }
            }
            _ => {}
        }
    }

    fn on_block_stop(&mut self, _data: Option<Value>) {
        if self.text_part_open {
            self.text_part_open = false;
            self.push(StreamEvent::PartStop);
        } else if self.tool_parts_open > 0 {
            self.tool_parts_open = self.tool_parts_open.saturating_sub(1);
            self.current_tool_index = None;
            self.push(StreamEvent::PartStop);
        }
    }

    fn on_message_delta(&mut self, data: Option<Value>) {
        if self.finished {
            return;
        }

        let Some(v) = data else { return };
        let stop_reason = v
            .pointer("/delta/stop_reason")
            .and_then(Value::as_str)
            .map(|s| StreamStopReason::from_api_str(s).unwrap_or(StreamStopReason::EndTurn));
        let in_tok = v
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        let out_tok = v
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);

        let usage = if in_tok > 0 || out_tok > 0 {
            Some(Usage::new(in_tok, out_tok))
        } else {
            None
        };

        self.push(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: stop_reason.map(|r| r.to_api_str().into()),
            },
            usage,
        }));
    }

    fn on_message_stop(&mut self) {
        self.finished = true;
        // Close any remaining open parts.
        if self.text_part_open {
            self.push(StreamEvent::PartStop);
        }
        for _ in 0..self.tool_parts_open {
            self.push(StreamEvent::PartStop);
        }
        self.tool_parts_open = 0;
        self.text_part_open = false;
    }

    /// Emit the terminal [`MessageStop`] if the stream was started.
    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = self.drain();
        if self.started && !self.finished {
            out.push(StreamEvent::MessageStop);
        }
        out
    }

    /// Drain all pending events.
    fn drain(&mut self) -> Vec<StreamEvent> {
        std::mem::take(&mut self.pending)
    }

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
    fn request_body_user_text_single_string() {
        let msgs = vec![Message::user("hello")];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[test]
    fn request_body_includes_system() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            "claude-3",
            &msgs,
            Some("be brief"),
            None,
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["system"], "be brief");
    }

    #[test]
    fn request_body_system_empty_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);
        assert_eq!(body["system"], "");
    }

    #[test]
    fn request_body_model() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            "claude-sonnet-4",
            &msgs,
            None,
            None,
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["model"], "claude-sonnet-4");
    }

    #[test]
    fn request_body_max_tokens() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn request_body_user_role() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn request_body_assistant_role() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::text("hello")],
        )];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);
        assert_eq!(body["messages"][0]["role"], "assistant");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn request_body_assistant_tool_calls() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                input: serde_json::json!({"msg": "hi"}),
            }],
        )];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "call_1");
        assert_eq!(content[0]["name"], "echo");
        assert_eq!(content[0]["input"]["msg"], "hi");
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
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_1");
        assert_eq!(content[0]["content"], "result text");
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
            "claude-3",
            &msgs,
            None,
            Some(&tools),
            false,
            DEFAULT_MAX_TOKENS,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["name"], "search");
        assert_eq!(tools_arr[0]["description"], "Search the web");
        assert_eq!(tools_arr[0]["input_schema"]["type"], "object");
    }

    #[test]
    fn request_body_tools_absent_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_assistant_with_text_and_tool_call() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::text("Let me search."),
                MessagePart::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_use");
    }

    #[test]
    fn request_body_multiple_messages() {
        let msgs = vec![
            Message::user("hello"),
            Message::new(Role::Assistant, vec![MessagePart::text("hi")]),
            Message::user("bye"),
        ];
        let body = build_request_body("claude-3", &msgs, None, None, false, DEFAULT_MAX_TOKENS);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
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
    fn builder_requires_api_key() {
        let result = AnthropicClient::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_succeeds_with_key() {
        let client = AnthropicClient::builder()
            .api_key("sk-test")
            .build()
            .unwrap();
        assert_eq!(client.model(), DEFAULT_MODEL);
    }

    #[test]
    fn builder_custom_base_url_and_model() {
        let client = AnthropicClient::builder()
            .api_key("sk-test")
            .base_url("https://custom.example.com")
            .model("claude-3-haiku")
            .build()
            .unwrap();
        assert_eq!(client.model(), "claude-3-haiku");
    }

    #[test]
    fn emitter_message_start() {
        let mut em = StreamEmitter::default();
        let data = serde_json::json!({
            "message": {"id": "msg_1", "model": "claude-3"}
        });
        em.on_message_start(Some(&data));
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

        // Start a text block.
        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "text"}
        })));
        em.drain();

        // Text delta.
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "text_delta", "text": "hi"}
        })));
        let events = em.drain();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::IndexedDelta(_)));
    }

    #[test]
    fn emitter_tool_use_block() {
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "echo"}
        })));
        let events = em.drain();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PartStart(_)))
        );
        assert_eq!(em.tool_parts_open, 1);

        // Input JSON delta.
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}
        })));
        let events2 = em.drain();
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn emitter_block_stop_closes_text() {
        let mut em = StreamEmitter::default();
        em.text_part_open = true;

        em.on_block_stop(None);
        let events = em.drain();
        assert!(matches!(events[0], StreamEvent::PartStop));
        assert!(!em.text_part_open);
    }

    #[test]
    fn emitter_message_delta_with_usage() {
        let mut em = StreamEmitter::default();

        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })));
        let events = em.drain();

        if let StreamEvent::MessageDelta(md) = &events[0] {
            assert_eq!(md.delta.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(md.usage.as_ref().unwrap().input_tokens, 10);
            assert_eq!(md.usage.as_ref().unwrap().output_tokens, 20);
        } else {
            panic!("expected MessageDelta");
        }
    }

    #[test]
    fn emitter_message_stop_closes_parts() {
        let mut em = StreamEmitter::default();
        em.text_part_open = true;
        em.tool_parts_open = 2;

        em.on_message_stop();
        let events = em.drain();
        // 1 for text + 2 for tools
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| matches!(e, StreamEvent::PartStop)));
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
            buf: "event: ping\n".into(),
        };
        assert_eq!(reader.take_line().unwrap(), "event: ping");
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
        let builder = AnthropicClientBuilder::default();
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_timeout() {
        let custom = Duration::from_secs(300);
        let builder = AnthropicClientBuilder::default().timeout(custom);
        assert_eq!(builder.timeout, custom);
        // connect_timeout should be unchanged.
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_connect_timeout() {
        let custom = Duration::from_secs(45);
        let builder = AnthropicClientBuilder::default().connect_timeout(custom);
        assert_eq!(builder.connect_timeout, custom);
        // timeout should be unchanged.
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn builder_custom_both_timeouts() {
        let req_timeout = Duration::from_secs(600);
        let conn_timeout = Duration::from_secs(30);
        let builder = AnthropicClientBuilder::default()
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
        let client = AnthropicClient::builder()
            .api_key("sk-test")
            .timeout(Duration::from_secs(180))
            .connect_timeout(Duration::from_secs(15))
            .build();
        assert!(client.is_ok(), "build should succeed with valid timeouts");
    }

    #[tokio::test]
    async fn sse_reader_take_line_splits_on_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "event: message_start\ndata: {}\n\n".to_string(),
        };
        assert_eq!(reader.take_line(), Some("event: message_start".to_string()));
        assert_eq!(reader.take_line(), Some("data: {}".to_string()));
        assert_eq!(reader.take_line(), Some(String::new()));
    }

    #[tokio::test]
    async fn sse_reader_next_event_extracts_payload() {
        let chunk = "event: content_block_delta\ndata: {\"type\":\"text_delta\"}\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(chunk.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_event().await.unwrap();
        assert!(result.is_some());
        let (event_type, data) = result.unwrap();
        assert_eq!(event_type, "content_block_delta");
        assert!(data.is_some());
    }

    #[tokio::test]
    async fn sse_reader_next_event_concatenates_multiline_data() {
        let chunk = "event: content_block_delta\ndata: {\"type\":\"text_delta\",\ndata: \"text\":\"hello\"}\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(chunk.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_event().await.unwrap();
        assert!(result.is_some());
        let (event_type, data) = result.unwrap();
        assert_eq!(event_type, "content_block_delta");
        assert!(
            data.is_some(),
            "multi-line data should concatenate into valid JSON"
        );
        let parsed = data.unwrap();
        assert_eq!(parsed["type"], "text_delta");
        assert_eq!(parsed["text"], "hello");
    }

    #[tokio::test]
    async fn sse_reader_next_event_malformed_data_returns_none_value() {
        // Malformed JSON data should be logged and returned as None for the
        // data payload, but the event_type is still captured (H4).
        let chunk = "event: ping\ndata: not valid json\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(chunk.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_event().await.unwrap();
        assert!(result.is_some());
        let (event_type, data) = result.unwrap();
        assert_eq!(event_type, "ping");
        assert!(data.is_none(), "malformed JSON should yield None data");
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        let huge = "x".repeat(SSE_MAX_BUFFER + 1);
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(huge)]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_event().await;
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
}
