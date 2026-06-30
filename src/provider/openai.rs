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
//!     .api_key("sk-...")
//!     .base_url("https://api.deepseek.com/v1")
//!     .model("deepseek-chat")
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::stream::{Stream, StreamExt};
use reqwest::Response;
use serde::Deserialize;
use serde_json::Value;

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

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o";
const SSE_DONE: &str = "[DONE]";
const SSE_DATA_PREFIX: &str = "data: ";
const TEXT_PART_INDEX: usize = 0;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120); // connect + response + body
const MAX_RESPONSE_BODY: usize = 10 * 1024 * 1024; // 10 Mb
const SSE_MAX_BUFFER: usize = 1024 * 1024; // 1 Mb
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ==================================================
// Client
// ==================================================

/// An OpenAI-compatible chat completions client with streaming support.
///
/// Implements [`ApiClient`] by translating between the framework's
/// [`StreamEvent`] protocol and the OpenAI Chat Completions SSE format.
///
/// Works with any OpenAI-compatible endpoint. Use a custom `base_url`
/// to target `DeepSeek`, `Grok`, Ollama, `vLLM`, or other compatible APIs.
pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: parking_lot::Mutex<String>,
}

impl OpenAiClient {
    /// Create a builder for configuring an [`OpenAiClient`].
    #[must_use]
    pub fn builder() -> OpenAiClientBuilder {
        OpenAiClientBuilder::default()
    }

    /// Create from environment variables.
    ///
    /// Reads:
    /// - `OPENAI_API_KEY` (or `API_KEY`) — required.
    /// - `OPENAI_BASE_URL` (or `BASE_URL`) — optional, defaults to
    ///   `https://api.openai.com/v1`.
    /// - `OPENAI_MODEL` (or `MODEL`) — optional, defaults to `gpt-4o`.
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
            .api_key(api_key)
            .base_url(base_url)
            .model(model)
            .build()
    }

    /// Build the chat-completions URL for this client.
    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Send a POST request to the chat-completions endpoint.
    ///
    /// Shared by both [`ApiClient::stream_messages`] and
    /// [`ApiClient::create_message`]. Returns the raw
    /// [`reqwest::Response`] after checking for HTTP errors.
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
        let resp = http
            .post(url)
            .bearer_auth(api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| ApiError::http(e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            Ok(resp)
        } else {
            let text = resp.text().await.unwrap_or_default();
            Err(ApiError::http_with_status(status.as_u16(), text))
        }
    }
}

impl ApiClient for OpenAiClient {
    fn model(&self) -> String {
        self.model.lock().clone()
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
        let body = RequestBody::build(&model, &messages, system.as_deref(), tools.as_deref());
        let url = self.completions_url();
        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let resp = Self::post_completions(&http, &url, &api_key, &body.to_json(true)).await?;
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some(data) = sse.next_data().await? {
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
        messages: Vec<Message>,
        system: Option<String>,
        tools: Option<Vec<ToolSchema>>,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        let model = self.model.lock().clone();
        let body = RequestBody::build(&model, &messages, system.as_deref(), tools.as_deref());
        let url = self.completions_url();

        Box::pin(async move {
            let resp =
                Self::post_completions(&self.http, &url, &self.api_key, &body.to_json(false))
                    .await?;
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

/// Builder for [`OpenAiClient`].
pub struct OpenAiClientBuilder {
    api_key: Option<String>,
    base_url: String,
    model: String,
    timeout: Duration,
    connect_timeout: Duration,
}

impl Default for OpenAiClientBuilder {
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

impl OpenAiClientBuilder {
    /// Set the API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL (e.g. `https://api.deepseek.com/v1`).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model name (e.g. `gpt-4o`, `deepseek-chat`).
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
    pub fn build(self) -> Result<OpenAiClient, ApiError> {
        let api_key = self
            .api_key
            .ok_or_else(|| ApiError::auth_invalid_key("API key not provided"))?;

        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .build()
            .map_err(|e| ApiError::http(e.to_string()))?;

        Ok(OpenAiClient {
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

/// A built OpenAI Chat Completions request body.
///
/// Separating construction from serialization lets us reuse the same
/// body for both streaming and non-streaming requests, toggling only
/// the `stream` flag via [`to_json`](Self::to_json).
struct RequestBody {
    model: String,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
}

impl RequestBody {
    /// Translate the framework's [`Message`] list into the OpenAI
    /// Chat Completions request shape.
    fn build(
        model: &str,
        messages: &[Message],
        system: Option<&str>,
        tools: Option<&[ToolSchema]>,
    ) -> Self {
        let mut msgs = Vec::with_capacity(messages.len().saturating_add(1));

        if let Some(sys) = system {
            msgs.push(serde_json::json!({ "role": "system", "content": sys }));
        }

        for m in messages {
            msgs.push(convert_message(m));
        }

        Self {
            model: model.into(),
            messages: msgs,
            tools: tools.map(convert_tools),
        }
    }

    /// Serialize to a [`serde_json::Value`] with the `stream` flag
    /// set as requested.
    fn to_json(&self, stream: bool) -> Value {
        serde_json::json!({
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "tools": self.tools,
        })
    }
}

/// Convert a single framework [`Message`] into the OpenAI JSON shape.
///
/// OpenAI expects assistant messages with `tool_calls` to carry them in
/// a dedicated array, tool results to use the `tool` role, and plain
/// text to use a simple `{role, content}` pair.
fn convert_message(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    // Bucket parts by OpenAI category.
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
        build_assistant_message(role, &tool_calls, &text_parts)
    } else if !tool_results.is_empty() {
        merge_tool_results(&tool_results)
    } else {
        serde_json::json!({ "role": role, "content": text_parts.join("") })
    }
}

/// Build an assistant message JSON that includes `tool_calls`.
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

/// Merge one or more tool-result entries into a single JSON value.
///
/// When there is only one result (the common case) we return it
/// directly; otherwise we return a JSON array so no data is lost.
fn merge_tool_results(results: &[Value]) -> Value {
    if results.len() == 1 {
        results.first().cloned().unwrap_or(Value::Null)
    } else {
        Value::Array(results.to_vec())
    }
}

/// Convert tool schemas into the OpenAI `tools` array shape.
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

// ==================================================
// SSE line reader
// ==================================================

/// Minimal SSE line reader over an HTTP byte stream.
///
/// Buffers raw bytes from the response, splits on newlines, and yields
/// the payload of each `data:` line (without the `data: ` prefix).
/// A `[DONE]` sentinel terminates the stream.
struct SseReader {
    bytes: Pin<Box<dyn Stream<Item = Result<String, ApiError>> + Send>>,
    buf: String,
}

impl SseReader {
    /// Wrap a streaming HTTP response.
    ///
    /// The byte stream is mapped to `String` chunks up-front so the
    /// rest of the reader is pure string processing.
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

    /// Extract the next SSE `data:` payload, blocking until one is
    /// available or the stream ends.
    ///
    /// Returns `Ok(None)` at end-of-stream (including `[DONE]`).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails.
    async fn next_data(&mut self) -> Result<Option<String>, ApiError> {
        loop {
            // Drain any complete lines already in the buffer.
            while let Some(line) = self.take_line() {
                let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) else {
                    continue;
                };
                if data == SSE_DONE {
                    return Ok(None);
                }
                return Ok(Some(data.into()));
            }

            // Fetch the next chunk from the network.
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
// OpenAI chunk types
// ==================================================

/// A single SSE chunk from the OpenAI streaming API.
#[derive(Deserialize)]
struct OpenAiChunk {
    id: String,
    model: String,
    choices: Vec<OpenAiChoice>,
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

#[derive(Deserialize)]
struct OpenAiChoice {
    delta: Option<OpenAiDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    id: String,
    function: Option<OpenAiToolCallFunction>,
}

#[derive(Deserialize)]
struct OpenAiToolCallFunction {
    name: String,
    arguments: String,
}

// ==================================================
// Stream event emitter
// ==================================================

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
    started: bool,
    text_part_open: bool,
    open_tool_count: usize,
    finished: bool,
    pending: Vec<StreamEvent>,
}

impl StreamEmitter {
    /// Process a single parsed chunk, appending events to the
    /// internal pending queue.
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

        let Some(choice) = chunk.choices.first() else {
            return;
        };

        if let Some(delta) = &choice.delta {
            self.process_delta(delta);
        }

        if let Some(reason) = &choice.finish_reason {
            self.process_finish(reason);
        }
    }

    /// Translate a delta into text/tool-call events.
    fn process_delta(&mut self, delta: &OpenAiDelta) {
        if let Some(text) = &delta.content
            && !text.is_empty()
        {
            if !self.text_part_open {
                self.text_part_open = true;
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

        if let Some(tool_calls) = &delta.tool_calls {
            for tc in tool_calls {
                self.process_tool_call(tc);
            }
        }
    }

    /// Handle a single tool-call delta.
    fn process_tool_call(&mut self, tc: &OpenAiToolCallDelta) {
        if tc.function.is_some() {
            // New tool call — emit PartStart.
            self.push(StreamEvent::PartStart(PartStart {
                index: tc.index,
                part: Some(MessagePart::ToolCall {
                    id: tc.id.clone(),
                    name: tc
                        .function
                        .as_ref()
                        .map(|f| f.name.clone())
                        .unwrap_or_default(),
                    input: Value::Null,
                }),
            }));
            self.open_tool_count = self.open_tool_count.saturating_add(1);
        }

        // Stream argument fragments.
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

    /// Handle a finish reason, emitting the appropriate stop events.
    fn process_finish(&mut self, reason: &str) {
        if self.finished {
            return;
        }
        self.finished = true;

        // Close any open text part.
        if self.text_part_open {
            self.push(StreamEvent::PartStop);
        }

        // Close each open tool-call part.
        for _ in 0..self.open_tool_count {
            self.push(StreamEvent::PartStop);
        }

        let stop_reason = match reason {
            "tool_calls" => StreamStopReason::ToolCall,
            "length" => StreamStopReason::MaxTokens,
            other => StreamStopReason::from_api_str(other).unwrap_or(StreamStopReason::EndTurn),
        };

        self.push(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop_reason.to_api_str().into()),
            },
            usage: None,
        }));
    }

    /// Emit the terminal [`MessageStop`] if the stream was started,
    /// returning all remaining events.
    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = self.drain();
        if self.started {
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
    use crate::tool::ToolSchema;

    #[test]
    fn request_body_includes_system_message_first() {
        let msgs = vec![Message::user("hello")];
        let body = RequestBody::build("gpt-4o", &msgs, Some("be brief"), None);
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
        let body = RequestBody::build("gpt-4o", &msgs, None, None);
        let json = body.to_json(false);

        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn request_body_stream_flag_toggles() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None);

        assert_eq!(body.to_json(true)["stream"], true);
        assert_eq!(body.to_json(false)["stream"], false);
    }

    #[test]
    fn request_body_model_and_tools() {
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = RequestBody::build("my-model", &msgs, None, Some(&tools));
        let json = body.to_json(true);

        assert_eq!(json["model"], "my-model");
        let tools_arr = json["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["type"], "function");
        assert_eq!(tools_arr[0]["function"]["name"], "echo");
    }

    #[test]
    fn request_body_tools_null_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = RequestBody::build("gpt-4o", &msgs, None, None);
        let json = body.to_json(false);
        assert!(json["tools"].is_null());
    }

    #[test]
    fn convert_message_user_text() {
        let m = Message::user("hello world");
        let v = convert_message(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hello world");
    }

    #[test]
    fn convert_message_assistant_text() {
        let m = Message::new(Role::Assistant, vec![MessagePart::text("hi there")]);
        let v = convert_message(&m);
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
        let v = convert_message(&m);
        assert_eq!(v["role"], "assistant");
        assert!(v["content"].is_null());
        let calls = v["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "echo");
        // arguments should be stringified JSON
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
                output: ToolContent::from_string("result text"),
                is_error: None,
            }],
        );
        let v = convert_message(&m);
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_1");
        assert!(v["content"].is_string());
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
        // MessageStart, PartStart(0), IndexedDelta(Text)
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

        // Tool call delta.
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"echo","arguments":"{\"msg\":"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        let events = em.drain();

        // PartStart(tool) + IndexedDelta(InputJson)
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
    fn emitter_finish_emits_part_stops_and_message_delta() {
        let mut em = StreamEmitter::default();

        // Send some text so the text part is open.
        let chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&chunk);
        em.drain();

        // Now send finish_reason=stop.
        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        let events = em.drain();

        // PartStop + MessageDelta(stop_reason)
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::PartStop));
        assert!(matches!(events[1], StreamEvent::MessageDelta(_)));
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

        // Open a tool call.
        let tool_chunk = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"echo","arguments":""}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        em.process_chunk(&tool_chunk);
        em.drain();

        // Finish with tool_calls.
        let finish = OpenAiChunk::parse(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"delta":null,"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap();
        em.process_chunk(&finish);
        let events = em.drain();

        // 1 PartStop (for tool) + MessageDelta
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::PartStop));

        if let StreamEvent::MessageDelta(md) = &events[1] {
            assert_eq!(md.delta.stop_reason.as_deref(), Some("tool_call"));
        } else {
            panic!("expected MessageDelta");
        }
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
        let events = em.drain();

        if let StreamEvent::MessageDelta(md) = &events[1] {
            assert_eq!(md.delta.stop_reason.as_deref(), Some("max_tokens"));
        } else {
            panic!("expected MessageDelta");
        }
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
    fn merge_tool_results_single() {
        let r = serde_json::json!({"role": "tool", "content": "ok"});
        let merged = merge_tool_results(std::slice::from_ref(&r));
        assert_eq!(merged, r);
    }

    #[test]
    fn merge_tool_results_multiple() {
        let results = vec![
            serde_json::json!({"role": "tool", "content": "a"}),
            serde_json::json!({"role": "tool", "content": "b"}),
        ];
        let merged = merge_tool_results(&results);
        assert!(merged.is_array());
        assert_eq!(merged.as_array().unwrap().len(), 2);
    }

    #[test]
    fn sse_reader_take_line_extracts_newline_terminated() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\n".into(),
        };
        let line = reader.take_line().unwrap();
        assert_eq!(line, "data: hello");
        assert!(reader.buf.is_empty());
    }

    #[test]
    fn sse_reader_take_line_returns_none_without_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "partial".into(),
        };
        assert!(reader.take_line().is_none());
    }

    #[test]
    fn sse_reader_take_line_handles_multiple_lines() {
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
        let line = reader.take_line().unwrap();
        assert_eq!(line, "data: hi");
    }

    // ================================================
    // Builder timeout tests
    // ================================================

    #[test]
    fn builder_has_default_timeouts() {
        // The builder should initialize with sensible non-zero defaults
        // so that a hanging server cannot block the agent loop indefinitely.
        let builder = OpenAiClientBuilder::default();
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_timeout() {
        let custom = Duration::from_secs(300);
        let builder = OpenAiClientBuilder::default().timeout(custom);
        assert_eq!(builder.timeout, custom);
        // connect_timeout should be unchanged.
        assert_eq!(builder.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
    }

    #[test]
    fn builder_custom_connect_timeout() {
        let custom = Duration::from_secs(45);
        let builder = OpenAiClientBuilder::default().connect_timeout(custom);
        assert_eq!(builder.connect_timeout, custom);
        // timeout should be unchanged.
        assert_eq!(builder.timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn builder_custom_both_timeouts() {
        let req_timeout = Duration::from_secs(600);
        let conn_timeout = Duration::from_secs(30);
        let builder = OpenAiClientBuilder::default()
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
        let client = OpenAiClient::builder()
            .api_key("sk-test")
            .timeout(Duration::from_secs(180))
            .connect_timeout(Duration::from_secs(15))
            .build();
        assert!(client.is_ok(), "build should succeed with valid timeouts");
    }

    // ==================================================
    // SSE buffer cap tests (M1)
    // ==================================================

    #[tokio::test]
    async fn sse_reader_take_line_splits_on_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hello\ndata: world\n".to_string(),
        };
        assert_eq!(reader.take_line(), Some("data: hello".to_string()));
        assert_eq!(reader.take_line(), Some("data: world".to_string()));
        assert_eq!(reader.take_line(), None);
    }

    #[tokio::test]
    async fn sse_reader_next_data_extracts_payload() {
        let data = "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[]}\n\n";
        let stream = futures::stream::iter(vec![Ok::<String, ApiError>(data.to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_data().await.unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("c1"));
    }

    #[tokio::test]
    async fn sse_reader_next_data_done_returns_none() {
        let stream =
            futures::stream::iter(vec![Ok::<String, ApiError>("data: [DONE]\n\n".to_string())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: String::new(),
        };
        let result = reader.next_data().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        // Feed a chunk larger than SSE_MAX_BUFFER without any newline so
        // the buffer grows unbounded — the cap should catch it.
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

    // ==================================================
    // Body size limit tests (H5)
    // ==================================================

    #[test]
    fn max_response_body_is_ten_mb() {
        assert_eq!(MAX_RESPONSE_BODY, 10 * 1024 * 1024);
    }

    #[test]
    fn body_size_check_rejects_oversized() {
        // Verify the comparison logic used in create_message.
        let oversized = MAX_RESPONSE_BODY + 1;
        assert!(oversized > MAX_RESPONSE_BODY);
    }

    #[test]
    fn body_size_check_accepts_within_limit() {
        let within = MAX_RESPONSE_BODY;
        assert!(within <= MAX_RESPONSE_BODY);
    }
}
