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
//!     .with_api_key("sk-ant-...")
//!     .with_model("claude-sonnet-4-20250514")
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
use crate::structured::ToolConstraint;
use crate::structured::tighten_json_schema;
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
const MAX_RESPONSE_BODY: usize = 10 * 1024 * 1024; // 10 Mb
const SSE_MAX_BUFFER: usize = 1024 * 1024; // 1 Mb
const MAX_ERROR_BODY: usize = 8 * 1024; // 8 Kb

// ==================================================
// Client
// ==================================================

/// An Anthropic Claude chat client with streaming support.
///
/// Implements [`ApiClient`] by translating between the framework's
/// [`StreamEvent`] protocol and the Anthropic Messages SSE format.
///
/// Also works with Anthropic-compatible endpoints such as `Z.ai`
/// — use a custom `base_url` via [`AnthropicClientBuilder::with_base_url`].
pub struct AnthropicClient {
    /// The underlying HTTP client (connection-pooled `reqwest::Client`).
    ///
    /// Created once at build time with the configured timeouts; reused
    /// across all requests for connection pooling.
    http: reqwest::Client,

    /// The Anthropic API key used for authentication.
    ///
    /// Sent as the `x-api-key` header on every request. Set via
    /// [`AnthropicClientBuilder::api_key`].
    api_key: String,

    /// The base URL for API requests.
    ///
    /// The Messages API endpoint is `{base_url}/v1/messages`. Defaults to
    /// `https://api.anthropic.com`; override for proxies or
    /// Anthropic-compatible endpoints.
    base_url: String,

    /// The current model identifier, stored behind a mutex for runtime
    /// hot-swapping.
    ///
    /// Changed via [`ApiClient::set_model`] when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips to a
    /// fallback model.
    model: std::sync::Mutex<String>,

    /// The maximum output tokens per response.
    ///
    /// Anthropic requires this field on every request. Defaults to 8192.
    /// Set via [`AnthropicClientBuilder::max_tokens`].
    max_tokens: u32,
}

impl AnthropicClient {
    /// Create a builder for configuring an [`AnthropicClient`].
    ///
    /// Returns an [`AnthropicClientBuilder`] with sensible defaults. The only
    /// required field is `api_key`; everything else has a production-ready
    /// default. Call `.with_api_key(...).build()` to finish, or chain additional
    /// setters for custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::provider::AnthropicClient;
    ///
    /// let client = AnthropicClient::builder()
    ///     .with_api_key("sk-ant-...")
    ///     .with_model("claude-sonnet-4-20250514")
    ///     .build()
    /// .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> AnthropicClientBuilder {
        AnthropicClientBuilder::default()
    }

    /// Create a client from environment variables.
    ///
    /// Reads the following variables:
    ///
    /// - `ANTHROPIC_API_KEY` — **required**. The API key for authentication.
    /// - `ANTHROPIC_BASE_URL` — optional. Defaults to `https://api.anthropic.com`.
    ///   Override when targeting a proxy or Anthropic-compatible endpoint.
    /// - `ANTHROPIC_MODEL` — optional. Defaults to `claude-sonnet-4-20250514`.
    ///
    /// This is a convenience constructor that delegates to
    /// [`builder`](Self::builder) with the env vars as setter arguments.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if `ANTHROPIC_API_KEY` is not set or is empty.
    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ApiError::auth_invalid_key("ANTHROPIC_API_KEY not set"))?;
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());

        Self::builder()
            .with_api_key(api_key)
            .with_base_url(base_url)
            .with_model(model)
            .build()
    }

    /// Send a POST request to the Anthropic Messages API endpoint.
    ///
    /// Shared by [`stream_messages`](ApiClient::stream_messages),
    /// [`create_message`](ApiClient::create_message),
    /// [`stream_messages_with_options`](ApiClient::stream_messages_with_options),
    /// and [`create_message_with_options`](ApiClient::create_message_with_options).
    /// Sends the JSON body with `x-api-key` and `anthropic-version` headers.
    /// On a non-success status, reads the error body (capped at
    /// `MAX_ERROR_BODY` bytes) and returns it as an [`ApiError`].
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::http`] if the HTTP request fails, or
    /// [`ApiError::http_with_status`] if the server responds with a
    /// non-success status code.
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

    /// Build the full URL for the Anthropic Messages API endpoint.
    ///
    /// Appends `/v1/messages` to the client's `base_url`. All four
    /// `ApiClient` methods (`stream_messages`, `create_message`, and their
    /// `*_with_options` variants) POST to this URL.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }
}

impl ApiClient for AnthropicClient {
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
        request: crate::api::StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.stream_messages_with_options(request, crate::structured::RequestOptions::default())
    }

    fn create_message(
        &self,
        request: crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        self.create_message_with_options(request, crate::structured::RequestOptions::default())
    }

    fn stream_messages_with_options(
        &self,
        request: crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let crate::api::StreamRequest {
            messages,
            system,
            tools,
        } = request;
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let rf = options.response_format.as_ref();
        let body = build_request_body(
            &RequestBodySpec {
                model: &model,
                messages: &messages,
                system: system.as_deref(),
                tools: tools.as_deref(),
                response_format: rf,
                tool_constraint: &options.tool_constraint,
            },
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

    fn create_message_with_options(
        &self,
        request: crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ApiError>> + Send + '_>> {
        let crate::api::StreamRequest {
            messages,
            system,
            tools,
        } = request;
        let model = crate::error::recover_guard(self.model.lock()).clone();
        let rf = options.response_format.as_ref();
        let body = build_request_body(
            &RequestBodySpec {
                model: &model,
                messages: &messages,
                system: system.as_deref(),
                tools: tools.as_deref(),
                response_format: rf,
                tool_constraint: &options.tool_constraint,
            },
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

    fn extract_structured(&self, raw: &Value) -> Value {
        raw.get("content")
            .and_then(|c| c.as_array())
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        block.get("input").cloned()
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| raw.clone())
    }
}

// ==================================================
// Builder
// ==================================================

/// Builder for [`AnthropicClient`].
///
/// Created via [`AnthropicClientBuilder::default`] or
/// [`AnthropicClient::builder`]. All fields have sensible defaults except
/// `api_key`, which must be set before [`build`](Self::build).
pub struct AnthropicClientBuilder {
    /// The Anthropic API key used for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is `None`.
    /// The key is sent as the `x-api-key` header on every request. Set via
    /// [`with_api_key`](Self::with_api_key) on the builder, or read from
    /// `ANTHROPIC_API_KEY` via [`AnthropicClient::from_env`].
    api_key: Option<String>,

    /// The base URL for API requests.
    ///
    /// Defaults to `https://api.anthropic.com`. Override when targeting a
    /// proxy, gateway, or Anthropic-compatible endpoint (e.g. Z.AI).
    base_url: String,

    /// The default model identifier (e.g. `claude-sonnet-4-20250514`).
    ///
    /// Can be changed at runtime via [`AnthropicClient::set_model`].
    model: String,

    /// The maximum number of output tokens per response.
    ///
    /// Anthropic requires this field on every request. Defaults to 8192.
    max_tokens: u32,

    /// Shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// Holds the timeout, connection-pool, and TCP knobs that apply to the
    /// internally-built `reqwest::Client`, or an externally-supplied client
    /// injected via [`with_http_client`](Self::with_http_client).
    http: super::HttpClientConfig,
}

impl Default for AnthropicClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            http: super::HttpClientConfig::default(),
        }
    }
}

impl AnthropicClientBuilder {
    /// Set the API key for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is not set.
    /// The key is sent as the `x-api-key` header on every request.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL for API requests.
    ///
    /// Defaults to `https://api.anthropic.com`. Override when targeting a
    /// proxy, gateway, or Anthropic-compatible endpoint (e.g. Z.AI at
    /// `https://api.z.ai/api/anthropic`).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the default model identifier.
    ///
    /// The model string is sent as the `model` field on every request. Can be
    /// changed at runtime via [`AnthropicClient::set_model`] (e.g. when the
    /// [`FallbackManager`](crate::fallback::FallbackManager) trips).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the maximum number of output tokens per response.
    ///
    /// Anthropic requires this field on every request — unlike OpenAI, which
    /// defaults it server-side. It bounds the length of a single model
    /// response. Defaults to 8192. Increase for long-form generation;
    /// decrease to cap cost on simple queries.
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
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

    /// Construct the [`AnthropicClient`] from the builder's configuration.
    ///
    /// Creates the internal `reqwest::Client` with the configured timeouts,
    /// and validates that an API key was provided.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key was set via
    /// [`with_api_key`](Self::with_api_key).
    pub fn build(self) -> Result<AnthropicClient, ApiError> {
        let api_key = self
            .api_key
            .ok_or_else(|| ApiError::auth_invalid_key("API key not provided"))?;
        let http = self.http.build()?;

        Ok(AnthropicClient {
            http,
            api_key,
            base_url: self.base_url,
            model: std::sync::Mutex::new(self.model),
            max_tokens: self.max_tokens,
        })
    }
}

// ==================================================
// Request body construction
// ==================================================

/// The per-request inputs to [`build_request_body`].
///
/// Carries the request-shape knobs: the model, conversation, tools, and
/// the structured-output / tool-call constraints. `stream` and `max_tokens`
/// are passed separately because they are per-call / per-client rather
/// than per-request-shape.
struct RequestBodySpec<'a> {
    /// The model identifier sent as the top-level `model` field of the
    /// Messages API request body.
    ///
    /// Copied from the client's current model (which may be hot-swapped at
    /// runtime via [`ApiClient::set_model`]). Anthropic uses this to
    /// dispatch to the correct model; it is echoed back on the response.
    ///
    /// [`ApiClient::set_model`]: crate::api::ApiClient::set_model
    model: &'a str,

    /// The conversation history, serialized into the top-level `messages`
    /// array via [`convert_message`].
    ///
    /// Each [`Message`] becomes an object with `role` (`"user"` or
    /// `"assistant"`) and `content` (a plain string for single-text
    /// messages, or an array of `text` / `tool_use` / `tool_result`
    /// blocks for mixed content).
    messages: &'a [Message],

    /// An optional system prompt, emitted as the top-level `system`
    /// string field (not a `system` role message — Anthropic keeps
    /// system context separate from the `messages` array).
    ///
    /// When `None`, the field is set to an empty string rather than
    /// omitted, matching Anthropic's expectation of a present `system`
    /// field.
    system: Option<&'a str>,

    /// The registered tool schemas the model may invoke.
    ///
    /// When `Some`, each [`ToolSchema`] becomes a `tools` array entry
    /// carrying `name`, `description`, and `input_schema`. When
    /// `tool_constraint` is `Strict`, each `input_schema` is tightened
    /// before submission. When `None`, the `tools` field is omitted from
    /// the body entirely (not sent as `null`).
    ///
    /// Suppressed when `response_format` is set; see that field.
    tools: Option<&'a [ToolSchema]>,

    /// If set, requests a schema-conformant *result* rather than free-form
    /// tool use.
    ///
    /// Anthropic has no native `response_format`; the structured-output
    /// path is implemented by synthesizing a single forced tool whose
    /// `input_schema` is `response_format.schema`, and emitting
    /// `tool_choice: { type: "tool", name: <rf.name> }` so the model
    /// must fill in that tool. The model's structured payload then lands
    /// in the assistant `tool_use` block's `input` field.
    ///
    /// When set, `tools` is replaced by this single forced tool — caller-
    /// supplied tools are not sent — and `tool_constraint` has no effect.
    response_format: Option<&'a crate::structured::ResponseFormat>,

    /// How strictly the model's tool-call output must follow the schemas.
    ///
    /// [`ToolConstraint::None`] forwards each tool's `input_schema`
    /// verbatim. [`ToolConstraint::Strict`] tightens each `input_schema`
    /// (recursive `additionalProperties: false` and full `required`)
    /// before submission; no `tool_choice` is emitted, so the model is
    /// free to choose whether to call a tool — only the *shape* of any
    /// call is constrained.
    ///
    /// Has no effect when `response_format` is set.
    ///
    /// [`ToolConstraint::None`]: crate::structured::ToolConstraint::None
    /// [`ToolConstraint::Strict`]: crate::structured::ToolConstraint::Strict
    tool_constraint: &'a ToolConstraint,
}

/// Build the JSON request body for the Anthropic Messages API.
///
/// Each [`Message`] is serialized via [`convert_message`], then assembled
/// with the model, `max_tokens`, system prompt, and optional tools.
///
/// Tool-call constraint:
/// - When `response_format` is set, synthesizes a single forced tool
///   (`tool_choice` forced); `tool_constraint` is ignored in that case.
/// - Otherwise, when `tool_constraint` is `Strict`, each tool's
///   `input_schema` is tightened (`additionalProperties: false` and full
///   `required`) via [`convert_tools`] before submission. No `tool_choice`
///   is emitted — `Strict` constrains the call's shape, not its selection.
fn build_request_body(spec: &RequestBodySpec<'_>, stream: bool, max_tokens: u32) -> Value {
    let RequestBodySpec {
        model,
        messages,
        system,
        tools,
        response_format,
        tool_constraint,
    } = spec;

    let (non_system, effective_system) = super::fold_system_messages(messages, *system);
    let msgs: Vec<Value> = non_system.iter().map(|m| convert_message(m)).collect();
    let effective_system = effective_system.unwrap_or_default();
    let (tools_val, tool_choice) = if let Some(rf) = response_format {
        let forced_tool = serde_json::json!({
            "name": rf.name,
            "description": "Return the result via this tool",
            "input_schema": rf.schema,
        });
        let choice = serde_json::json!({
            "type": "tool",
            "name": rf.name,
        });
        (Some(vec![forced_tool]), Some(choice))
    } else {
        let strict = matches!(tool_constraint, ToolConstraint::Strict);
        (tools.map(|t| convert_tools(t, strict)), None)
    };

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": msgs,
        "system": effective_system,
        "stream": stream,
        "tools": tools_val,
    });

    if tools_val.is_none()
        && let Some(obj) = body.as_object_mut()
    {
        obj.remove("tools");
    }

    if let Some(choice) = tool_choice
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert("tool_choice".to_string(), choice);
    }

    body
}

/// Convert a single framework [`Message`] into the Anthropic JSON shape.
///
/// - Messages with only a single text part use a plain string for `content`
///   (Anthropic's recommended optimization).
/// - Messages with tool calls or tool results use the full `content` array.
fn convert_message(m: &Message) -> Value {
    // System messages are folded into the top-level `system` field by
    // `build_request_body` before this function is reached, so the `System`
    // pattern below is defensive: if one ever reaches here, route it to `user`
    // so the text renders rather than being dropped silently.
    let role = match m.role {
        Role::User | Role::System => "user",
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

/// Convert framework tool schemas into the Anthropic `tools` array shape.
///
/// Each [`ToolSchema`] becomes a JSON object with `name`, `description`, and
/// `input_schema` — the three fields Anthropic's tool-use API expects. The
/// `input_schema` is passed through verbatim from the framework's schema (a
/// JSON Schema Draft 07 object), since Anthropic validates it server-side.
///
/// When `strict` is `true`, each `input_schema` is first tightened
/// (recursive `additionalProperties: false` and full `required`) so the
/// server-side validation enforces the strict shape. This is the
/// [`ToolConstraint::Strict`] path for Anthropic, which has no native
/// per-tool strict flag.
///
/// When structured output is active (`response_format` set), this function is
/// not called — instead, [`build_request_body`] synthesizes a single forced
/// tool whose `input_schema` is the target `ResponseFormat::schema`.
fn convert_tools(tools: &[ToolSchema], strict: bool) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            let input_schema = if strict {
                tighten_json_schema(&t.input_schema)
            } else {
                t.input_schema.clone()
            };
            serde_json::json!({
                "name": t.tool,
                "description": &t.description,
                "input_schema": input_schema,
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
#[allow(clippy::struct_excessive_bools)]
struct StreamEmitter {
    /// Whether [`MessageStart`] has been emitted for the current stream.
    ///
    /// The Anthropic stream opens with one `message_start` event; the
    /// emitter forwards it once as a [`StreamEvent::MessageStart`] and
    /// ignores any subsequent duplicates.
    started: bool,

    /// Whether a text content block is currently open.
    ///
    /// Anthropic signals the start of a text block with
    /// `content_block_start` (`type: "text"`) and its end with
    /// `content_block_stop`. This flag tracks the open state so the
    /// matching `content_block_stop` emits exactly one
    /// [`StreamEvent::PartStop`].
    text_part_open: bool,

    /// Number of tool-use content blocks currently open.
    ///
    /// Anthropic opens a tool block with `content_block_start`
    /// (`type: "tool_use"`) and closes it with `content_block_stop`.
    /// Multiple tool blocks may be open in sequence; this counter drives
    /// the matching `PartStop` emissions (one per open block) on
    /// `message_stop`.
    tool_parts_open: usize,

    /// Index of the tool block most recently opened by
    /// `content_block_start`, used to route subsequent
    /// `input_json_delta` fragments.
    ///
    /// The deltas themselves do not carry an index, so the emitter
    /// remembers the index from the surrounding block-start event and
    /// emits each [`DeltaPart::InputJson`] at that index. Falls back to
    /// the text index when no tool block is open (defensive — should not
    /// happen on a well-formed stream).
    ///
    /// [`DeltaPart::InputJson`]: crate::stream::DeltaPart::InputJson
    current_tool_index: Option<usize>,

    /// Whether a thinking/reasoning content block is currently open.
    ///
    /// Anthropic signals the start of a thinking block with
    /// `content_block_start` (`type: "thinking"` or `"redacted_thinking"`)
    /// and its end with `content_block_stop`. This flag tracks the open state
    /// so the matching `content_block_stop` emits exactly one
    /// [`StreamEvent::PartStop`].
    thinking_part_open: bool,

    /// Index of the thinking block.
    ///
    /// Most recently opened thinking bolock by
    /// `content_block_start`, used to route subsequent `thinking_delta`
    /// fragments. Mirrors [`current_tool_index`](Self::current_tool_index)
    /// for the reasoning lane.
    thinking_index: Option<usize>,

    /// Whether the terminal stop signal has been processed.
    ///
    /// Set by either `message_delta` (which carries the stop reason and
    /// usage) or `message_stop`. Guards against emitting the final
    /// [`StreamEvent::MessageDelta`] twice when both events arrive, and
    /// against `finish` appending a spurious [`StreamEvent::MessageStop`]
    /// after the stream already terminated.
    finished: bool,

    /// Buffered [`StreamEvent`]s waiting to be yielded to the consumer.
    ///
    /// All `on_*` handlers push onto this queue; the stream loop drains it
    /// after each SSE event so events are yielded promptly rather than
    /// held until stream end. Drained by [`drain`](Self::drain).
    pending: Vec<StreamEvent>,
}

impl StreamEmitter {
    /// Dispatch a single SSE event to the matching handler by type.
    ///
    /// `event_type` is the value of the Anthropic `event:` line
    /// (`message_start`, `content_block_start`, `content_block_delta`,
    /// `content_block_stop`, `message_delta`, `message_stop`); `data` is
    /// the parsed JSON payload of the paired `data:` line, or `None` when
    /// the payload was absent or failed to parse. Unknown event types are
    /// ignored. Any events produced are appended to the pending queue.
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

    /// Handle a `message_start` event.
    ///
    /// On the first call, reads the message `id` and `model` from
    /// `/message/id` and `/message/model` and emits a
    /// [`StreamEvent::MessageStart`]. No-ops on subsequent calls (the
    /// `started` flag guards against duplicate emissions). Missing fields
    /// default to empty strings rather than erroring, so a malformed
    /// `message_start` still produces a usable event.
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

    /// Handle a `content_block_start` event.
    ///
    /// Reads the block `type` at `/content_block/type` and the block
    /// `index` at `/index` (defaulting to 0 when absent or non-numeric).
    /// For a `tool_use` block, emits a [`StreamEvent::PartStart`] carrying
    /// a [`MessagePart::ToolCall`] (with `id` and `name` from
    /// `/content_block/id` and `/content_block/name`, `input` left null
    /// until the deltas arrive), records the block's index as the current
    /// tool index, and increments the open-tool counter. For a `text`
    /// block, marks the text part open and emits a `PartStart` at the text
    /// index. Other block types are ignored.
    ///
    /// [`MessagePart::ToolCall`]: crate::message::MessagePart::ToolCall
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
            Some("thinking" | "redacted_thinking") => {
                self.thinking_part_open = true;
                self.thinking_index = Some(index);
                self.push(StreamEvent::PartStart(PartStart { index, part: None }));

                if matches!(block_type, Some("redacted_thinking")) {
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index,
                        delta: DeltaPart::Thinking {
                            text: String::new(),
                        },
                    }));
                }
            }
            _ => {}
        }
    }

    /// Handle a `content_block_delta` event.
    ///
    /// Dispatches on `/delta/type`. A `text_delta` emits a
    /// [`DeltaPart::Text`] at the text index (skipped when the text
    /// fragment is empty). An `input_json_delta` emits a
    /// [`DeltaPart::InputJson`] at the current tool index carrying the
    /// `/delta/partial_json` fragment, so the caller can accumulate the
    /// full tool-call arguments across deltas. Empty fragments are
    /// skipped. Other delta types are ignored.
    ///
    /// [`DeltaPart::Text`]: crate::stream::DeltaPart::Text
    /// [`DeltaPart::InputJson`]: crate::stream::DeltaPart::InputJson
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
            Some("thinking_delta") => {
                let text = v
                    .pointer("/delta/thinking")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !text.is_empty() {
                    // Use the index from the corresponding content_block_start.
                    let thinking_index = self.thinking_index.unwrap_or(TEXT_PART_INDEX);
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: thinking_index,
                        delta: DeltaPart::Thinking { text },
                    }));
                }
            }
            _ => {}
        }
    }

    /// Handle a `content_block_stop` event.
    ///
    /// Closes the currently open content block. If a text block is open,
    /// clears the flag and emits a [`StreamEvent::PartStop`]. Otherwise,
    /// if one or more tool blocks are open, decrements the counter,
    /// clears the current tool index, and emits a single `PartStop` for
    /// the block that just closed. The `data` payload is unused (Anthropic
    /// does not carry useful information on this event beyond the
    /// implicit close).
    fn on_block_stop(&mut self, _data: Option<Value>) {
        if self.text_part_open {
            self.text_part_open = false;
            self.push(StreamEvent::PartStop);
        } else if self.thinking_part_open {
            self.thinking_part_open = false;
            self.thinking_index = None;
            self.push(StreamEvent::PartStop);
        } else if self.tool_parts_open > 0 {
            self.tool_parts_open = self.tool_parts_open.saturating_sub(1);
            self.current_tool_index = None;
            self.push(StreamEvent::PartStop);
        }
    }

    /// Handle a `message_delta` event carrying the terminal stop reason
    /// and usage.
    ///
    /// Reads `/delta/stop_reason` (mapped via
    /// [`StreamStopReason::from_api_str`], defaulting to `EndTurn` on an
    /// unrecognized value) and `/usage/input_tokens` +
    /// `/usage/output_tokens` (defaulting to 0; the usage event is only
    /// attached when at least one is non-zero). Emits a single
    /// [`StreamEvent::MessageDelta`] with both.
    ///
    /// This handler does not mark the stream finished — that is the job of
    /// [`on_message_stop`](Self::on_message_stop), which Anthropic sends
    /// after `message_delta`. The `finished` guard here only defends
    /// against an out-of-order stream where `message_stop` arrived first.
    ///
    /// [`StreamStopReason::from_api_str`]: crate::stream::StreamStopReason::from_api_str
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

    /// Handle a `message_stop` event.
    ///
    /// Marks the stream finished, closes any content blocks still marked
    /// open (one [`StreamEvent::PartStop`] for an open thinking block, one
    /// for an open text block, then one per open tool block, with all
    /// counters reset), and emits the terminal [`StreamEvent::MessageStop`]
    /// that consumers rely on to know the stream is complete. The closing
    /// `PartStop`s are pushed first so the event order matches the
    /// documented protocol (`PartStop* → MessageStop`).
    ///
    /// Setting `finished` here is what suppresses the synthetic
    /// [`StreamEvent::MessageStop`] in [`finish`](Self::finish), so a
    /// stream that delivered `message_stop` does not get a second
    /// terminal event appended after the SSE stream ends.
    fn on_message_stop(&mut self) {
        self.finished = true;
        if self.thinking_part_open {
            self.push(StreamEvent::PartStop);
        }
        if self.text_part_open {
            self.push(StreamEvent::PartStop);
        }
        for _ in 0..self.tool_parts_open {
            self.push(StreamEvent::PartStop);
        }
        self.thinking_part_open = false;
        self.thinking_index = None;
        self.tool_parts_open = 0;
        self.text_part_open = false;
        self.push(StreamEvent::MessageStop);
    }

    /// Finalize the stream and return any remaining events.
    ///
    /// Drains the pending queue, then appends a single
    /// [`StreamEvent::MessageStop`] when the stream was started but no
    /// `message_stop` event has already emitted one (tracked by the
    /// `finished` flag). This covers streams that end without an explicit
    /// terminal event; when `message_stop` was processed, the synthetic
    /// terminal is suppressed so the consumer sees exactly one
    /// `MessageStop`.
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

    /// Append an event to the pending queue.
    ///
    /// Single write point: every `on_*` handler routes through here so the
    /// queue is the only place events accumulate. The stream loop reads
    /// them back via [`drain`](Self::drain).
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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[test]
    fn request_body_includes_system() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: Some("be brief"),
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["system"], "be brief");
    }

    #[test]
    fn request_body_system_empty_when_none() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["system"], "");
    }

    #[test]
    fn request_body_model() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-sonnet-4",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["model"], "claude-sonnet-4");
    }

    #[test]
    fn request_body_max_tokens() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn request_body_user_role() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn request_body_assistant_role() {
        let msgs = vec![Message::new(
            Role::Assistant,
            vec![MessagePart::text("hello")],
        )];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

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
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&tools),
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

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
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

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
        let out = convert_tools(&tools, false);
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
            .with_api_key("sk-test")
            .build()
            .unwrap();
        assert_eq!(client.model(), DEFAULT_MODEL);
    }

    #[test]
    fn builder_custom_base_url_and_model() {
        let client = AnthropicClient::builder()
            .with_api_key("sk-test")
            .with_base_url("https://custom.example.com")
            .with_model("claude-3-haiku")
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
        // 1 PartStop for text + 2 for tools, then the terminal MessageStop.
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], StreamEvent::PartStop));
        assert!(matches!(events[1], StreamEvent::PartStop));
        assert!(matches!(events[2], StreamEvent::PartStop));
        assert!(
            matches!(events.last(), Some(StreamEvent::MessageStop)),
            "message_stop must emit the terminal MessageStop after the PartStops: {events:?}"
        );
    }

    #[test]
    fn emitter_message_stop_then_finish_no_duplicate() {
        let mut em = StreamEmitter::default();
        em.started = true;

        em.on_message_stop();
        let after_stop = em.drain();
        assert!(
            after_stop
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStop))
        );

        let after_finish = em.finish();
        assert!(
            after_finish
                .iter()
                .all(|e| !matches!(e, StreamEvent::MessageStop)),
            "finish() must not emit a second MessageStop after on_message_stop: {after_finish:?}"
        );
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
    fn builder_timeouts_applied_on_build() {
        let client = AnthropicClient::builder()
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

    #[test]
    fn request_body_response_format_forces_tool() {
        let msgs = vec![Message::user("classify this")];
        let rf = crate::structured::ResponseFormat::new(
            "action",
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        );
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: Some(&rf),
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        // Exactly one forced tool with the schema's name + input_schema.
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 1, "should have exactly one forced tool");
        assert_eq!(tools[0]["name"], "action");
        assert_eq!(tools[0]["input_schema"], rf.schema);

        // tool_choice forces the named tool.
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "action");
    }

    #[test]
    fn request_body_response_format_suppresses_caller_tools() {
        let msgs = vec![Message::user("hi")];
        let caller_tool = ToolSchema {
            tool: "read".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let rf =
            crate::structured::ResponseFormat::new("result", serde_json::json!({"type": "object"}));
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&[caller_tool]),
                response_format: Some(&rf),
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        // The forced tool replaces the caller's tools — not appended.
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]["name"], "result",
            "caller tools should be suppressed"
        );
    }

    #[test]
    fn request_body_no_response_format_has_no_tool_choice() {
        let msgs = vec![Message::user("hi")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should only appear with response_format"
        );
    }

    #[test]
    fn extract_structured_from_tool_use_input() {
        let client = AnthropicClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let raw = serde_json::json!({
            "content": [{
                "type": "tool_use",
                "name": "action",
                "input": {"tool": "write", "args": {}}
            }]
        });
        let value = client.extract_structured(&raw);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_text_only_falls_back_to_raw() {
        let client = AnthropicClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let raw = serde_json::json!({
            "id": "msg_1",
            "model": "claude-3",
            "content": [{"type": "text", "text": "I cannot do that."}]
        });
        let value = client.extract_structured(&raw);
        // No tool_use block → returns the raw envelope; T::from_value fails.
        assert_eq!(value["id"], "msg_1");
    }

    #[test]
    fn anthropic_strict_tightens_input_schema() {
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
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&tools),
                response_format: None,
                tool_constraint: &ToolConstraint::Strict,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        let input_schema = &tools_arr[0]["input_schema"];
        assert_eq!(input_schema["additionalProperties"], false);
        let required = input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "msg");
    }

    #[test]
    fn anthropic_none_constraint_unchanged_shape() {
        // Default None: convert_tools path unchanged (no tightening).
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&tools),
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        // input_schema is passed through verbatim (no additionalProperties).
        assert_eq!(
            tools_arr[0]["input_schema"],
            serde_json::json!({"type": "object"})
        );
        assert!(
            tools_arr[0]["input_schema"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn anthropic_strict_does_not_emit_tool_choice() {
        // Strict constrains the call's shape, not its selection. tool_choice
        // must not appear (only response_format forces it).
        let msgs = vec![Message::user("hi")];
        let tools = vec![ToolSchema {
            tool: "echo".into(),
            description: "Echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&tools),
                response_format: None,
                tool_constraint: &ToolConstraint::Strict,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice must not appear under Strict"
        );
    }

    #[test]
    fn anthropic_strict_suppressed_when_response_format_set() {
        // With response_format set, the forced-tool path runs and caller
        // tools are not tightened.
        let msgs = vec![Message::user("hi")];
        let caller_tool = ToolSchema {
            tool: "read".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let rf =
            crate::structured::ResponseFormat::new("result", serde_json::json!({"type": "object"}));
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: Some(&[caller_tool]),
                response_format: Some(&rf),
                tool_constraint: &ToolConstraint::Strict,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        // The forced tool replaces the caller's tools — exactly one tool
        // named "result", tool_choice forced. No tightening applied to
        // caller_tool (it was dropped).
        let tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "result");
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "result");
    }

    fn system_role_msg(text: &str) -> Message {
        Message::new(Role::System, vec![MessagePart::text(text)])
    }

    #[test]
    fn request_body_system_role_folded_into_system_field() {
        // An inline Role::System message must NOT appear in the messages
        // array; its text must be folded into the top-level `system` field.
        let msgs = vec![
            Message::user("hello"),
            system_role_msg("stay on task"),
            Message::assistant("working"),
        ];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );

        let messages = body["messages"].as_array().expect("messages is an array");
        assert_eq!(messages.len(), 2, "system-role message is filtered out");
        for m in messages {
            assert_ne!(
                m["role"].as_str().unwrap_or(""),
                "system",
                "no inline system-role message should be emitted"
            );
        }
        assert_eq!(body["system"], "stay on task");
    }

    #[test]
    fn request_body_system_role_merges_with_caller_system() {
        // When both a caller-supplied system prompt and an inline
        // Role::System message are present, the top-level `system` field
        // carries both (caller prompt first, folded text appended).
        let msgs = vec![Message::user("hi"), system_role_msg("reminder")];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: Some("be brief"),
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        let system = body["system"].as_str().expect("system is a string");
        assert!(
            system.starts_with("be brief"),
            "caller system prompt comes first: got {system:?}"
        );
        assert!(
            system.contains("reminder"),
            "folded text is appended: got {system:?}"
        );
        assert!(
            system.contains('\n'),
            "caller prompt and folded text are newline-separated: got {system:?}"
        );
    }

    #[test]
    fn request_body_system_role_preserves_message_order() {
        // Folding must not reorder the remaining (non-system) messages.
        let msgs = vec![
            Message::user("first"),
            system_role_msg("mid reminder"),
            Message::assistant("second"),
            Message::user("third"),
        ];
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: None,
                tools: None,
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
        );
        let messages = body["messages"].as_array().expect("messages is an array");
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
        // And the user contents arrive in original order.
        assert_eq!(messages[0]["content"], "first");
        assert_eq!(messages[2]["content"], "third");
    }

    #[test]
    fn emitter_thinking_delta_emits_thinking_variant() {
        let mut em = StreamEmitter::default();

        // Start a thinking block at index 0.
        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "thinking"}
        })));
        em.drain();

        // Thinking delta.
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "thinking_delta", "thinking": "reasoning here"}
        })));
        let events = em.drain();

        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { text } => assert_eq!(text, "reasoning here"),
                other => panic!("expected Thinking, got {other:?}"),
            },
            other => panic!("expected IndexedDelta, got {other:?}"),
        }
    }

    #[test]
    fn emitter_signature_delta_is_ignored() {
        let mut em = StreamEmitter::default();

        // Start a thinking block + emit a thinking delta.
        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "thinking"}
        })));
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "thinking_delta", "thinking": "visible reasoning"}
        })));
        em.drain();

        // Now send a signature_delta — must NOT emit any additional event.
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "signature_delta", "signature": "opaque_base64_blob"}
        })));
        let events = em.drain();

        assert!(
            events.is_empty(),
            "signature_delta must not emit any events: got {events:?}"
        );
    }

    #[test]
    fn emitter_redacted_thinking_emits_empty_delta() {
        let mut em = StreamEmitter::default();

        // Start a redacted_thinking block — should emit PartStart + one
        // empty Thinking delta (the placeholder convention).
        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "redacted_thinking"}
        })));
        let events = em.drain();

        // PartStart + one empty Thinking delta.
        assert_eq!(events.len(), 2, "expected PartStart + empty Thinking delta");
        assert!(matches!(events[0], StreamEvent::PartStart(_)));
        match &events[1] {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { text } => {
                    assert!(text.is_empty(), "redacted thinking → empty text");
                }
                other => panic!("expected Thinking, got {other:?}"),
            },
            other => panic!("expected IndexedDelta, got {other:?}"),
        }
        assert!(em.thinking_part_open, "thinking_part_open set");
    }

    #[test]
    fn emitter_thinking_block_stop_closes_part() {
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "thinking"}
        })));
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "thinking_delta", "thinking": "reasoning"}
        })));
        em.drain();

        // Block stop must close the thinking part: emit one PartStop and
        // reset the tracking fields.
        em.on_block_stop(None);
        let events = em.drain();

        assert_eq!(events.len(), 1, "exactly one PartStop");
        assert!(matches!(events[0], StreamEvent::PartStop));
        assert!(!em.thinking_part_open, "thinking_part_open reset");
        assert!(em.thinking_index.is_none(), "thinking_index cleared");
    }

    #[test]
    fn emitter_message_stop_closes_open_thinking_block() {
        // If a thinking block is still open when `message_stop` arrives
        // (e.g. a stream that ended without a final `content_block_stop`),
        // `on_message_stop` must defensively emit a PartStop for it before
        // the terminal MessageStop. Without it the open block leaks and the
        // downstream accumulator never finalizes the part boundary.
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "thinking"}
        })));
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "thinking_delta", "thinking": "reasoning"}
        })));
        em.drain();
        assert!(
            em.thinking_part_open,
            "test precondition: thinking lane open"
        );

        em.on_message_stop();
        let events = em.drain();

        // Expected: one PartStop (thinking) then one MessageStop.
        assert!(
            events.iter().any(|e| matches!(e, StreamEvent::PartStop)),
            "message_stop must emit a PartStop for the open thinking block"
        );
        assert!(
            events.iter().any(|e| matches!(e, StreamEvent::MessageStop)),
            "message_stop must emit the terminal MessageStop"
        );
        assert!(
            !em.thinking_part_open,
            "thinking_part_open must be reset by message_stop"
        );
        assert!(
            em.thinking_index.is_none(),
            "thinking_index must be cleared by message_stop"
        );

        // Ordering: PartStop before MessageStop.
        let stop_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::PartStop))
            .expect("PartStop present");
        let msg_stop_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::MessageStop))
            .expect("MessageStop present");
        assert!(stop_idx < msg_stop_idx, "PartStop must precede MessageStop");
    }
}
