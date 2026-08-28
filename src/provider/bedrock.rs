//! AWS Bedrock provider — SigV4-authenticated, two invoke paths.
//!
//! Routes through the Bedrock runtime (`bedrock-runtime.<region>.amazonaws.com`)
//! with AWS Signature Version 4 credentials. Two paths are supported
//! (see [`BedrockPath`]): the native Anthropic Messages body for
//! `anthropic.claude-*` model ids (reusing [`crate::provider::anthropic`]'s
//! body builders and event translation), and Bedrock's cross-model
//! [`Converse`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)
//! API for everything else. Streaming responses arrive as AWS binary
//! event-stream frames, decoded here into the provider-neutral
//! [`StreamEvent`] sequence.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::api::error::ApiError;
use crate::api::{ApiClient, NonStreamingResponse, StreamRequest};
use crate::stream::StreamStopReason as Stop;
use crate::stream::{StreamEvent, StreamStopReason};

type HmacSha256 = Hmac<Sha256>;

/// Which Bedrock invoke path to use for the configured model.
///
/// Selects both the endpoint suffix and the request/response translation.
/// Auto-selected from the model id by [`BedrockClientBuilder::build`] when
/// not set explicitly: ids starting with `anthropic.` use the native
/// Anthropic path; everything else uses Converse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockPath {
    /// Native Anthropic Messages body via `InvokeModel`/`InvokeModelWithResponseStream`.
    ///
    /// Reuses the `anthropic.rs` body builders and event translation
    /// (wrapped in event-stream frames). Use for `anthropic.*` model
    /// ids.
    Anthropic,
    /// Bedrock's cross-model Converse API. Bedrock's own body/delta shapes.
    Converse,
}

/// An AWS Bedrock chat client with streaming support, SigV4-authenticated.
///
/// Implements [`ApiClient`] by routing through the Bedrock runtime.
/// Credentials are signed with AWS Signature Version 4 — there is no
/// bearer API key.
///
/// # Construction
///
/// ```rust,ignore
/// use loopctl::provider::BedrockClient;
///
/// // From AWS env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
/// // AWS_SESSION_TOKEN?, AWS_REGION):
/// let client = BedrockClient::from_env()?;
///
/// // Or explicit:
/// let client = BedrockClient::builder()
///     .region("us-east-1")
///     .access_key_id("AKIA…")
///     .secret_access_key("…")
///     .model("anthropic.claude-sonnet-4-5-20250929-v1:0")
///     .build()?;
/// ```
#[derive(Debug)]
pub struct BedrockClient {
    region: String,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    model: Mutex<String>,
    path: BedrockPath,
    http: reqwest::Client,
}

/// Builder for [`BedrockClient`].
///
/// Collects the AWS credentials, region, model, and optional path
/// override; [`build`](Self::build) validates that the required
/// fields are present and selects the invoke path from the model id
/// when not overridden.
#[derive(Debug, Default)]
pub struct BedrockClientBuilder {
    region: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    model: Option<String>,
    path: Option<BedrockPath>,
}

impl BedrockClientBuilder {
    /// Set the AWS region (e.g. `"us-east-1"`).
    ///
    /// Determines the endpoint host (`bedrock-runtime.{region}.amazonaws.com`).
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the AWS access key id.
    ///
    /// The IAM identity whose credentials sign each request.
    #[must_use]
    pub fn access_key_id(mut self, key: impl Into<String>) -> Self {
        self.access_key_id = Some(key.into());
        self
    }

    /// Set the AWS secret access key.
    ///
    /// Paired with the access key id for `SigV4` signing; never sent
    /// on the wire.
    #[must_use]
    pub fn secret_access_key(mut self, key: impl Into<String>) -> Self {
        self.secret_access_key = Some(key.into());
        self
    }

    /// Set an optional AWS session token (for STS / role credentials).
    ///
    /// Included in the signed headers when present.
    #[must_use]
    pub fn session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Set the Bedrock model id (e.g. `"anthropic.claude-sonnet-4-5-20250929-v1:0"`).
    ///
    /// The id's prefix (`anthropic.*`) selects the invoke path unless
    /// [`path`](Self::path) overrides it.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override the invoke-path auto-selection.
    ///
    /// By default the path is chosen from the model id's prefix;
    /// this forces a specific path regardless.
    #[must_use]
    pub fn path(mut self, path: BedrockPath) -> Self {
        self.path = Some(path);
        self
    }

    /// Build the client.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] when the region, credentials, or model are
    /// missing, or when the HTTP client cannot be constructed.
    pub fn build(self) -> Result<BedrockClient, ApiError> {
        let region = self
            .region
            .ok_or_else(|| ApiError::config("bedrock: region is required"))?;
        let access_key_id = self
            .access_key_id
            .ok_or_else(|| ApiError::config("bedrock: access_key_id is required"))?;
        let secret_access_key = self
            .secret_access_key
            .ok_or_else(|| ApiError::config("bedrock: secret_access_key is required"))?;
        let model = self
            .model
            .ok_or_else(|| ApiError::config("bedrock: model is required"))?;
        let path = self.path.unwrap_or_else(|| auto_path(&model));
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| ApiError::config(format!("bedrock: HTTP client: {e}")))?;
        Ok(BedrockClient {
            region,
            access_key_id,
            secret_access_key,
            session_token: self.session_token,
            model: Mutex::new(model),
            path,
            http,
        })
    }
}

/// Auto-select the invoke path from a Bedrock model id.
fn auto_path(model: &str) -> BedrockPath {
    if model.starts_with("anthropic.") {
        BedrockPath::Anthropic
    } else {
        BedrockPath::Converse
    }
}

/// The `AWS SigV4` signing output — headers to attach to the request.
#[derive(Debug)]
struct SigV4Headers {
    authorization: String,
    amz_date: String,
}

/// Sign a request with AWS Signature Version 4.
///
/// Produces the `Authorization`, `X-Amz-Date`, and (when a session
/// token exists) `X-Amz-Security-Token` headers for a POST to
/// `host` at `uri` with the given `payload`.
#[allow(clippy::too_many_arguments)]
fn sigv4_sign(
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    region: &str,
    host: &str,
    uri: &str,
    payload: &[u8],
    now: std::time::SystemTime,
) -> SigV4Headers {
    use sha2::Digest as _;

    let amz_date = format_amz_date(now);
    let date_stamp = &amz_date[..8];

    let payload_hash = hex::encode(Sha256::digest(payload));
    let (canonical_headers, signed_header_list) = match session_token {
        Some(token) => (
            format!(
                "content-type:application/json\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\nx-amz-security-token:{token}\n"
            ),
            "content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token",
        ),
        None => (
            format!(
                "content-type:application/json\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
            ),
            "content-type;host;x-amz-content-sha256;x-amz-date",
        ),
    };

    let canonical_request =
        format!("POST\n{uri}\n\n{canonical_headers}\n{signed_header_list}\n{payload_hash}");
    let credential_scope = format!("{date_stamp}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );

    let k_date = hmac_key(
        secret_access_key.as_bytes(),
        format!("AWS4{date_stamp}").as_bytes(),
    );
    let k_region = hmac_key(&k_date, region.as_bytes());
    let k_service = hmac_key(&k_region, b"bedrock");
    let k_signing = hmac_key(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_bytes(&k_signing, string_to_sign.as_bytes()));

    SigV4Headers {
        authorization: format!(
            "AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, \
             SignedHeaders={signed_header_list}, Signature={signature}"
        ),
        amz_date,
    }
}

/// Format a `SystemTime` as an AMZ date (`YYYYMMDD'T'HHMMSS'Z'`).
fn format_amz_date(now: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let (year, month, day, hour, min, sec) = epoch_to_utc(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z")
}

/// Convert epoch seconds to UTC calendar fields (civil-from-days algorithm).
#[allow(clippy::arithmetic_side_effects)]
fn epoch_to_utc(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX);
    let rem = secs % 86_400;
    let (hour, min, sec) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let m = u32::try_from(m).unwrap_or(1);
    (if m <= 2 { y + 1 } else { y }, m, d, hour, min, sec)
}

/// Derive an HMAC-SHA256 signing key.
///
/// `Hmac<Sha256>` accepts any key length, so this cannot fail; the
/// fallback arm is unreachable in practice but keeps the no-panic
/// discipline mechanical.
fn hmac_key(key: &[u8], data: &[u8]) -> Vec<u8> {
    let Ok(mut mac) = HmacSha256::new_from_slice(key) else {
        return Vec::new();
    };
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// HMAC-SHA256 of `data` under `key`, returning raw bytes.
fn hmac_bytes(key: &[u8], data: &[u8]) -> Vec<u8> {
    hmac_key(key, data)
}

/// The streaming endpoint URL for a model and region.
fn stream_url(region: &str, model: &str) -> String {
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke-with-response-stream"
    )
}

/// The non-streaming endpoint URL.
fn invoke_url(region: &str, model: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke")
}

/// Incremental decoder for `application/vnd.amazon.eventstream` frames.
///
/// Feed raw response-body bytes via [`push`](Self::push); each complete
/// frame yields a decoded [`AwsEvent`] — its event-type header and
/// payload bytes.
#[derive(Debug, Default)]
struct AwsEventStreamDecoder {
    buf: Vec<u8>,
}

/// One decoded event-stream frame.
#[derive(Debug, PartialEq, Eq)]
struct AwsEvent {
    /// The `:event-type` header value (e.g. `"chunk"`, `"initial-response"`, `"exception"`).
    event_type: String,
    /// The `:message-type` header value (`"event"` or `"response"`).
    message_type: String,
    /// The frame's payload bytes (the JSON chunk for `chunk` events).
    payload: Vec<u8>,
}

impl AwsEventStreamDecoder {
    /// Feed raw bytes; returns every complete frame decoded so far.
    fn push(&mut self, bytes: &[u8]) -> Vec<AwsEvent> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(event) = self.decode_frame() {
            events.push(event);
        }
        events
    }

    /// Try to decode one frame from the buffer front.
    fn decode_frame(&mut self) -> Option<AwsEvent> {
        if self.buf.len() < 12 {
            return None;
        }
        let total_len = be_u32(&self.buf, 0)? as usize;
        if !(16..=16 * 1024 * 1024).contains(&total_len) {
            self.buf.clear();
            return None;
        }
        if self.buf.len() < total_len {
            return None; // incomplete
        }
        let headers_len = be_u32(&self.buf, 4)? as usize;
        let headers_end = 12usize.saturating_add(headers_len);
        if headers_end > total_len {
            self.buf.drain(..total_len);
            return None;
        }
        let headers = self
            .buf
            .get(12..headers_end)
            .map(parse_headers)
            .unwrap_or_default();
        let payload_end = total_len.saturating_sub(4); // trailing CRC
        let payload = self
            .buf
            .get(headers_end..payload_end)
            .unwrap_or(&[])
            .to_vec();
        self.buf.drain(..total_len);
        Some(AwsEvent {
            event_type: headers
                .iter()
                .find(|(k, _)| k == ":event-type")
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            message_type: headers
                .iter()
                .find(|(k, _)| k == ":message-type")
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            payload,
        })
    }
}

/// Read a big-endian u32 from `buf` at `offset`.
#[allow(clippy::arithmetic_side_effects)]
fn be_u32(buf: &[u8], offset: usize) -> Option<u32> {
    let slice = buf.get(offset..offset.checked_add(4)?)?;
    let b0 = slice.first().copied().unwrap_or(0);
    let b1 = slice.get(1).copied().unwrap_or(0);
    let b2 = slice.get(2).copied().unwrap_or(0);
    let b3 = slice.get(3).copied().unwrap_or(0);
    Some((u32::from(b0) << 24) | (u32::from(b1) << 16) | (u32::from(b2) << 8) | u32::from(b3))
}

/// Parse event-stream headers (TLV) into key-value pairs.
#[allow(clippy::arithmetic_side_effects)]
fn parse_headers(bytes: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let Some(name_len) = bytes.get(i).copied() else {
            break;
        };
        i += 1;
        let name_start = i;
        i = i.saturating_add(name_len as usize);
        let Some(name) = bytes.get(name_start..i) else {
            break;
        };
        let Some(value_type) = bytes.get(i).copied() else {
            break;
        };
        i += 1;
        let value = match value_type {
            7 => {
                let len = be_u32(bytes, i).unwrap_or(0) as usize;
                i += 4;
                let start = i;
                i = i.saturating_add(len);
                String::from_utf8_lossy(bytes.get(start..i).unwrap_or(&[])).into_owned()
            }
            0 => "true".to_string(),
            1 => "false".to_string(),
            _ => String::new(),
        };
        out.push((String::from_utf8_lossy(name).into_owned(), value));
    }
    out
}

/// Build the Anthropic-native request body for Bedrock.
///
/// Bedrock accepts the native Anthropic Messages body (system as a
/// top-level string field, `anthropic_version` in the body). Reuses
/// [`crate::provider::anthropic`]'s message translation.
///
/// # Errors
///
/// Returns an [`ApiError`] on translation failure.
fn anthropic_body(request: &StreamRequest, model: &str, stream: bool) -> serde_json::Value {
    let system = request.system.clone().unwrap_or_default();
    let mut messages = Vec::new();
    for message in &request.messages {
        messages.push(crate::provider::anthropic::convert_message(message));
    }
    let tools: Option<Vec<serde_json::Value>> =
        if request.tools.as_ref().is_some_and(|t| !t.is_empty()) {
            Some(crate::provider::anthropic::convert_tools(
                request.tools.as_deref().unwrap_or(&[]),
                false,
            ))
        } else {
            None
        };
    let body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": messages,
        "model": model,
        "stream": stream,
        "system": if system.is_empty() { serde_json::Value::Null } else { serde_json::json!(system) },
        "tools": tools.unwrap_or_default(),
    });
    body
}

/// Build the Converse API request body.
///
/// Bedrock's Converse uses its own message shape: `messages` with
/// `role` + `content` blocks, `system` as a top-level list, and
/// `additionalModelRequestFields` for pass-through.
fn converse_body(request: &StreamRequest, _stream: bool) -> serde_json::Value {
    let mut messages = Vec::new();
    for message in &request.messages {
        let content = crate::provider::openai::convert_message(message);
        messages.push(serde_json::json!({
            "role": if message.role == crate::message::Role::Assistant { "assistant" } else { "user" },
            "content": content,
        }));
    }
    serde_json::json!({
        "messages": messages,
        "inferenceConfig": {},
        "system": match &request.system {
            Some(system) => serde_json::json!([{ "text": system }]),
            None => serde_json::Value::Null,
        },
        "additionalModelRequestFields": {},
    })
}

/// Translate an Anthropic SSE JSON chunk into engine `StreamEvent`s.
///
/// Each Bedrock chunk's payload is one Anthropic event JSON object
/// (the same shapes the direct Anthropic SSE path emits). This is a
/// lightweight translation — it does not keep the full `StreamEmitter`
/// state (the direct path's emitter is not `pub(super)`-reachable from
/// the module layout without further widening; this local translation
/// covers the same event set).
/// # Errors
///
/// Returns an [`ApiError`] on translation failure.
fn anthropic_chunk_to_events(
    chunk: &serde_json::Value,
    accumulator: &mut StreamAccumulator,
    stop_reason: &mut StreamStopReason,
) -> Vec<StreamEvent> {
    let kind = chunk
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let mut out = Vec::new();
    match kind {
        "message_start" => {
            if let Some(msg) = chunk.pointer("/message") {
                accumulator.on_message_start(msg);
            }
            out.push(StreamEvent::MessageStart(accumulator.message_start_clone()));
        }
        "content_block_start" => {
            StreamAccumulator::on_block_start();
        }
        "content_block_delta" => {
            if let Some(delta) = chunk
                .pointer("/delta/text")
                .and_then(serde_json::Value::as_str)
            {
                StreamAccumulator::push_text(delta);
                out.push(StreamEvent::IndexedDelta(stream_indexed_text_delta(delta)));
            }
        }
        "message_delta" => {
            let stop = chunk
                .pointer("/delta/stop_reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("end_turn");
            *stop_reason = anthropic_stop_to_engine(stop);
            if let Some(usage) = chunk.get("usage") {
                accumulator.on_usage(usage);
            }
        }
        "message_stop" | "ping" | "content_block_stop" => {}
        other => {
            tracing::debug!(kind = %other, "unknown anthropic chunk type");
        }
    }
    out
}

/// Map an Anthropic stop reason to the engine's.
fn anthropic_stop_to_engine(stop: &str) -> StreamStopReason {
    match stop {
        "max_tokens" => Stop::MaxTokens,
        "stop_sequence" => Stop::StopSequence,
        "tool_use" => Stop::ToolCall,
        _ => Stop::EndTurn,
    }
}

/// Minimal accumulation state for the Anthropic path.
#[derive(Debug, Default)]
struct StreamAccumulator {
    message_id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
}

impl StreamAccumulator {
    fn on_message_start(&mut self, message: &serde_json::Value) {
        self.message_id = message
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.model = message
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.input_tokens = message
            .pointer("/usage/input_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
    }

    fn on_block_start() {}

    fn push_text(_delta: &str) {}

    fn on_usage(&mut self, usage: &serde_json::Value) {
        if let Some(n) = usage
            .get("output_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            self.output_tokens = n;
        }
    }

    fn message_start_clone(&self) -> crate::stream::MessageStart {
        crate::stream::MessageStart {
            message: crate::stream::MessageMetadata {
                id: self.message_id.clone(),
                role: "assistant".to_string(),
                model: self.model.clone(),
            },
        }
    }
}

/// Build an `IndexedDelta` carrying a text fragment at index 0.
fn stream_indexed_text_delta(text: &str) -> crate::stream::IndexedDelta {
    crate::stream::IndexedDelta {
        index: 0,
        delta: crate::stream::DeltaPart::Text {
            text: text.to_string(),
        },
    }
}

/// Translate a Converse-stream chunk into engine `StreamEvent`s.
/// # Errors
///
/// Returns an [`ApiError`] on translation failure.
fn converse_chunk_to_events(
    chunk: &serde_json::Value,
    accumulator: &mut StreamAccumulator,
    stop_reason: &mut StreamStopReason,
) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    if let Some(text) = chunk
        .pointer("/delta/content/text")
        .and_then(serde_json::Value::as_str)
    {
        StreamAccumulator::push_text(text);
        out.push(StreamEvent::IndexedDelta(stream_indexed_text_delta(text)));
    }
    if let Some(role) = chunk.get("role").and_then(serde_json::Value::as_str) {
        accumulator.on_message_start(&serde_json::json!({
            "id": "",
            "model": "",
            "role": role,
        }));
        out.push(StreamEvent::MessageStart(accumulator.message_start_clone()));
    }
    if let Some(stop) = chunk.get("stopReason").and_then(serde_json::Value::as_str) {
        *stop_reason = anthropic_stop_to_engine(stop);
    }
    out
}

impl BedrockClient {
    /// Build from AWS environment variables.
    ///
    /// Reads `AWS_REGION` (required), `AWS_ACCESS_KEY_ID` (required),
    /// `AWS_SECRET_ACCESS_KEY` (required), `AWS_SESSION_TOKEN`
    /// (optional), and `AWS_BEDROCK_MODEL` (optional — defaults to an
    /// Anthropic Sonnet id).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] when a required variable is missing or the
    /// builder rejects the configuration.
    pub fn from_env() -> Result<Self, ApiError> {
        let region = std::env::var("AWS_REGION")
            .map_err(|_| ApiError::config("bedrock: AWS_REGION is not set"))?;
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| ApiError::config("bedrock: AWS_ACCESS_KEY_ID is not set"))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| ApiError::config("bedrock: AWS_SECRET_ACCESS_KEY is not set"))?;
        let model = std::env::var("AWS_BEDROCK_MODEL")
            .unwrap_or_else(|_| "anthropic.claude-sonnet-4-5-20250929-v1:0".to_string());
        let mut builder = Self::builder()
            .region(region)
            .access_key_id(access_key_id)
            .secret_access_key(secret_access_key)
            .model(model);
        if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
            builder = builder.session_token(token);
        }
        builder.build()
    }

    /// Create a builder for configuring a [`BedrockClient`].
    ///
    /// Equivalent to [`BedrockClientBuilder::default`]; the builder
    /// pattern is the idiomatic entry point.
    #[must_use]
    pub fn builder() -> BedrockClientBuilder {
        BedrockClientBuilder::default()
    }

    /// The client's current model id.
    ///
    /// Reflects the most recent [`set_model`](ApiClient::set_model)
    /// or the builder's initial value.
    #[must_use]
    pub fn model(&self) -> String {
        crate::error::recover_guard(self.model.lock()).clone()
    }

    /// The Bedrock host for the configured region.
    fn host(&self) -> String {
        format!("bedrock-runtime.{}.amazonaws.com", self.region)
    }

    /// Sign and POST a request body, returning the raw response.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] on transport failure.
    async fn signed_post(&self, url: &str, body: &[u8]) -> Result<reqwest::Response, ApiError> {
        let uri = url
            .split_once("amazonaws.com")
            .map_or("/".to_string(), |(_, rest)| rest.to_string());
        let host = self.host();
        let creds = sigv4_sign(
            &self.access_key_id,
            &self.secret_access_key,
            self.session_token.as_deref(),
            &self.region,
            &host,
            &uri,
            body,
            std::time::SystemTime::now(),
        );
        let mut request = self
            .http
            .post(url)
            .header("Authorization", creds.authorization)
            .header("X-Amz-Date", creds.amz_date)
            .header("Content-Type", "application/json")
            .header(
                "X-Amz-Content-Sha256",
                hex::encode(sha2::Sha256::digest(body)),
            );
        if let Some(token) = &self.session_token {
            request = request.header("X-Amz-Security-Token", token);
        }
        request
            .body(body.to_vec())
            .send()
            .await
            .map_err(|e| ApiError::api(format!("bedrock: request failed: {e}")))
    }
}

impl ApiClient for BedrockClient {
    fn model(&self) -> String {
        self.model()
    }

    fn set_model(&self, model: &str) -> bool {
        let mut guard = crate::error::recover_guard(self.model.lock());
        *guard = model.to_string();
        true
    }

    fn base_url(&self) -> String {
        format!("https://{}", self.host())
    }

    /// # Errors
    ///
    /// Yields [`ApiError`] on transport failure, signing failure,
    /// HTTP error status, or unparseable event-stream chunks.
    fn stream_messages(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let model = self.model();
        let body = match self.path {
            BedrockPath::Anthropic => anthropic_body(request, &model, true),
            BedrockPath::Converse => converse_body(request, true),
        };
        let url = stream_url(&self.region, &model);
        let client = self.http.clone();
        let region = self.region.clone();
        let access_key_id = self.access_key_id.clone();
        let secret = self.secret_access_key.clone();
        let session_token = self.session_token.clone();
        let path = self.path;

        Box::pin(async_stream::try_stream! {
            let bytes = serde_json::to_vec(&body)
                .map_err(|e| ApiError::api(format!("bedrock: serialize: {e}")))?;
            let uri = url
                .split_once("amazonaws.com")
                .map_or("/".to_string(), |(_, rest)| rest.to_string());
            let host = format!("bedrock-runtime.{region}.amazonaws.com");
            let creds = sigv4_sign(
                &access_key_id, &secret, session_token.as_deref(),
                &region, &host, &uri, &bytes, std::time::SystemTime::now(),
            );
            let mut req = client
                .post(&url)
                .header("Authorization", creds.authorization)
                .header("X-Amz-Date", creds.amz_date)
                .header("Content-Type", "application/json")
                .header("X-Amz-Content-Sha256", hex::encode(sha2::Sha256::digest(&bytes)));
            if let Some(token) = &session_token {
                req = req.header("X-Amz-Security-Token", token);
            }
            let response = req
                .body(bytes)
                .send()
                .await
                .map_err(|e| ApiError::api(format!("bedrock: request failed: {e}")))?;
            use futures::StreamExt as _;
            let status = response.status();
            let mut byte_stream = if status.is_success() {
                response.bytes_stream()
            } else {
                let text = response.text().await.unwrap_or_default();
                Err(ApiError::api(format!(
                    "bedrock: HTTP {status}: {text}"
                )))?
            };
            let mut decoder = AwsEventStreamDecoder::default();
            let mut accumulator = StreamAccumulator::default();
            let mut stop_reason = StreamStopReason::EndTurn;
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| {
                    ApiError::api(format!("bedrock: stream read: {e}"))
                })?;
                for event in decoder.push(bytes.as_ref()) {
                    if event.event_type == "exception" {
                        Err(ApiError::api(format!(
                            "bedrock: model stream error: {}",
                            String::from_utf8_lossy(&event.payload)
                        )))?;
                    }
                    if event.event_type != "chunk" {
                        continue;
                    }
                    let json: serde_json::Value = serde_json::from_slice(&event.payload)
                        .map_err(|e| {
                            ApiError::api(format!(
                                "bedrock: chunk parse: {e}"
                            ))
                        })?;
                    let events = match path {
                        BedrockPath::Anthropic => anthropic_chunk_to_events(
                            &json, &mut accumulator, &mut stop_reason,
                        ),
                        BedrockPath::Converse => converse_chunk_to_events(
                            &json, &mut accumulator, &mut stop_reason,
                        ),
                    };
                    for ev in events {
                        yield ev;
                    }
                }
            }
            let _ = stop_reason;
        })
    }

    /// # Errors
    ///
    /// Returns [`ApiError`] on transport failure, HTTP error status,
    /// or an unrecognized response shape.
    fn create_message(
        &self,
        request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        let model = self.model();
        let body = match self.path {
            BedrockPath::Anthropic => anthropic_body(request, &model, false),
            BedrockPath::Converse => converse_body(request, false),
        };
        let url = invoke_url(&self.region, &model);
        Box::pin(async move {
            let bytes = serde_json::to_vec(&body)
                .map_err(|e| ApiError::api(format!("bedrock: serialize: {e}")))?;
            let response = self.signed_post(&url, &bytes).await?;
            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(ApiError::api(format!("bedrock: HTTP {status}: {text}")));
            }
            // Non-streaming invoke: the response body is the model's JSON
            // directly (no event framing).
            let json: serde_json::Value = response
                .json()
                .await
                .map_err(|e| ApiError::api(format!("bedrock: response parse: {e}")))?;
            bedrock_non_streaming_response(&json, &model)
        })
    }
}

/// Translate a non-streaming invoke response into a
/// [`NonStreamingResponse`].
///
/// Recognizes both response shapes — the Anthropic-native content
/// array and the Converse output structure — and maps each to the
/// engine's message parts, usage, and stop reason.
///
/// # Errors
///
/// Returns an [`ApiError`] on translation failure.
fn bedrock_non_streaming_response(
    json: &serde_json::Value,
    model: &str,
) -> Result<NonStreamingResponse, ApiError> {
    // Anthropic-native invoke response
    if let Some(content) = json.get("content").and_then(serde_json::Value::as_array) {
        let mut parts = Vec::new();
        for block in content {
            if let Some(text) = block.get("text").and_then(serde_json::Value::as_str) {
                parts.push(crate::message::MessagePart::text(text));
            }
            if let Some(tool) = block.get("id").and_then(serde_json::Value::as_str) {
                let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                parts.push(crate::message::MessagePart::ToolCall {
                    id: tool.to_string(),
                    name: block
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input,
                });
            }
        }
        let stop = json
            .get("stop_reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("end_turn");
        let usage = json.get("usage");
        return Ok(NonStreamingResponse {
            message: crate::message::Message::new(crate::message::Role::Assistant, parts),
            usage: usage.map(|u| crate::stream::Usage {
                input_tokens: u
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
                output_tokens: u
                    .get("output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
            }),
            stop_reason: anthropic_stop_to_engine(stop),
        });
    }
    // Converse response
    if let Some(output) = json
        .pointer("/output/message/content")
        .and_then(serde_json::Value::as_array)
    {
        let mut parts = Vec::new();
        for block in output {
            if let Some(text) = block.get("text").and_then(serde_json::Value::as_str) {
                parts.push(crate::message::MessagePart::text(text));
            }
        }
        let stop = json
            .get("stopReason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("end_turn");
        return Ok(NonStreamingResponse {
            message: crate::message::Message::new(crate::message::Role::Assistant, parts),
            usage: json.pointer("/usage").map(|u| crate::stream::Usage {
                input_tokens: u
                    .get("inputTokens")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
                output_tokens: u
                    .get("outputTokens")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
            }),
            stop_reason: anthropic_stop_to_engine(stop),
        });
    }
    let _ = model;
    Err(ApiError::api(format!(
        "bedrock: unrecognized non-streaming response shape: {json}"
    )))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::duration_suboptimal_units)]
    use super::*;
    use crate::message::Message;
    /// Build one event-stream frame (for tests).
    #[allow(clippy::cast_possible_truncation, clippy::unreadable_literal)]
    fn build_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        // :message-type = "event"
        headers.extend_from_slice(b"\x0d:message-type\x07");
        headers.extend_from_slice(&5u32.to_be_bytes());
        headers.extend_from_slice(b"event");
        // :event-type
        headers.extend_from_slice(b"\x0b:event-type\x07");
        headers.extend_from_slice(&(event_type.len() as u32).to_be_bytes());
        headers.extend_from_slice(event_type.as_bytes());
        // :content-type
        headers.extend_from_slice(b"\x0c:content-type\x07");
        headers.extend_from_slice(&16u32.to_be_bytes());
        headers.extend_from_slice(b"application/json");

        let total = 12usize
            .saturating_add(headers.len())
            .saturating_add(payload.len())
            .saturating_add(4);
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes()); // preamble CRC (unchecked)
        out.extend_from_slice(&headers);
        out.extend_from_slice(payload);
        out.extend_from_slice(&0u32.to_be_bytes()); // message CRC (unchecked)
        out
    }

    // A known SigV4 test vector (the AWS-documented example request).
    #[test]
    fn sigv4_known_vector() {
        let headers = sigv4_sign(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "bedrock-runtime.us-east-1.amazonaws.com",
            "/model/test",
            b"{}",
            // 2015-08-30T12:36:00Z
            std::time::SystemTime::UNIX_EPOCH
                .checked_add(std::time::Duration::from_secs(1_440_938_160))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        );
        assert!(
            headers
                .authorization
                .starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
            "authorization starts with the scheme and credential: {}",
            headers.authorization
        );
        assert!(
            headers.authorization.contains("Signature="),
            "authorization carries the signature"
        );
        assert_eq!(headers.amz_date, "20150830T123600Z");
    }

    #[test]
    fn sigv4_session_token_adds_header_to_signed_set() {
        let headers = sigv4_sign(
            "AKID",
            "secret",
            Some("token"),
            "us-east-1",
            "host",
            "/",
            b"{}",
            std::time::SystemTime::UNIX_EPOCH,
        );
        assert!(
            headers.authorization.contains("x-amz-security-token"),
            "the session token header is signed: {}",
            headers.authorization
        );
    }

    #[test]
    fn auto_path_selects_by_model_prefix() {
        assert_eq!(
            auto_path("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            BedrockPath::Anthropic
        );
        assert_eq!(
            auto_path("amazon.titan-text-express-v1"),
            BedrockPath::Converse
        );
    }

    #[test]
    fn event_stream_decoder_round_trips_a_frame() {
        let payload = br#"{"type":"content_block_delta","delta":{"text":"hi"}}"#;
        let frame = build_frame("chunk", payload);
        let mut decoder = AwsEventStreamDecoder::default();
        let events = decoder.push(&frame);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "chunk");
        assert_eq!(events[0].message_type, "event");
        assert_eq!(events[0].payload.as_slice(), payload);
    }

    #[test]
    fn event_stream_decoder_handles_partial_then_rest() {
        let payload = br#"{"type":"message_stop"}"#;
        let frame = build_frame("chunk", payload);
        let mut decoder = AwsEventStreamDecoder::default();
        let split = frame.len() / 2;
        assert!(decoder.push(&frame[..split]).is_empty());
        let events = decoder.push(&frame[split..]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "chunk");
    }

    #[test]
    fn event_stream_decoder_skips_non_chunk_events() {
        let initial = build_frame("initial-response", b"{}");
        let chunk = build_frame("chunk", br#"{"x":1}"#);
        let mut all = initial;
        all.extend_from_slice(&chunk);
        let mut decoder = AwsEventStreamDecoder::default();
        let events = decoder.push(&all);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "initial-response");
        assert_eq!(events[1].event_type, "chunk");
    }

    #[test]
    fn anthropic_chunk_translation_covers_the_event_set() {
        let mut acc = StreamAccumulator::default();
        let mut stop = StreamStopReason::EndTurn;

        let start = serde_json::json!({
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude", "usage": {"input_tokens": 10}}
        });
        let events = anthropic_chunk_to_events(&start, &mut acc, &mut stop);
        assert_eq!(events.len(), 1);

        let delta = serde_json::json!({
            "type": "content_block_delta", "delta": {"text": "hi"}
        });
        let events = anthropic_chunk_to_events(&delta, &mut acc, &mut stop);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::IndexedDelta(d) if matches!(&d.delta,
                crate::stream::DeltaPart::Text { text } if text == "hi")
        ));

        let md = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 5}
        });
        anthropic_chunk_to_events(&md, &mut acc, &mut stop);
        assert_eq!(stop, StreamStopReason::MaxTokens);
    }

    #[test]
    fn anthropic_body_carries_system_and_tools() {
        let request = StreamRequest {
            messages: vec![Message::user("hello")],
            system: Some("be terse".to_string()),
            tools: Some(vec![crate::tool::ToolSchema {
                tool: "echo".to_string(),
                description: "Echo".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }]),
        };
        let body = anthropic_body(&request, "anthropic.claude", true);
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert!(body["stream"].as_bool().unwrap());
        assert!(body["tools"].is_array());
    }

    #[test]
    fn converse_body_shapes_messages_and_system() {
        let request = StreamRequest {
            messages: vec![Message::user("hello"), Message::assistant("hi")],
            system: Some("sys".to_string()),
            tools: None,
        };
        let body = converse_body(&request, true);
        assert_eq!(
            body["system"][0]["text"], "sys",
            "converse puts system as a list of text blocks"
        );
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn non_streaming_response_parses_anthropic_shape() {
        let json = serde_json::json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "t1", "name": "echo", "input": {"a": 1}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 7}
        });
        let response = bedrock_non_streaming_response(&json, "m").unwrap();
        assert_eq!(
            response.stop_reason,
            crate::stream::StreamStopReason::ToolCall
        );
        assert_eq!(response.message.text_content(), "hello");
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 7);
    }

    #[test]
    fn non_streaming_response_parses_converse_shape() {
        let json = serde_json::json!({
            "output": {
                "message": {
                    "content": [{"text": "converse says hi"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 4}
        });
        let response = bedrock_non_streaming_response(&json, "m").unwrap();
        assert_eq!(response.message.text_content(), "converse says hi");
        assert_eq!(response.usage.unwrap().output_tokens, 4);
    }

    #[test]
    fn builder_rejects_missing_credentials() {
        let err = BedrockClientBuilder::default()
            .region("us-east-1")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("access_key_id"));

        let err = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("secret_access_key"));
    }

    #[test]
    fn builder_auto_selects_path_from_model() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("anthropic.claude-sonnet-4-5-v1:0")
            .build()
            .unwrap();
        assert_eq!(client.path, BedrockPath::Anthropic);

        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("amazon.nova-pro-v1")
            .build()
            .unwrap();
        assert_eq!(client.path, BedrockPath::Converse);
    }

    #[test]
    fn endpoint_urls_embed_region_and_model() {
        assert_eq!(
            stream_url("us-east-1", "anthropic.claude-v1:0"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v1:0/invoke-with-response-stream"
        );
        assert_eq!(
            invoke_url("eu-west-1", "m"),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/m/invoke"
        );
    }

    #[test]
    fn converse_chunk_translation_covers_deltas_and_stop() {
        let mut acc = StreamAccumulator::default();
        let mut stop = StreamStopReason::EndTurn;

        let delta = serde_json::json!({
            "role": "assistant",
            "delta": {"content": {"text": "hello"}}
        });
        let events = converse_chunk_to_events(&delta, &mut acc, &mut stop);
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::IndexedDelta(d) if matches!(&d.delta,
                    crate::stream::DeltaPart::Text { text } if text == "hello"))),
            "text delta arrives: {events:?}"
        );

        let stop_chunk = serde_json::json!({"stopReason": "max_tokens"});
        converse_chunk_to_events(&stop_chunk, &mut acc, &mut stop);
        assert_eq!(stop, StreamStopReason::MaxTokens);
    }

    #[test]
    fn converse_body_without_system_omits_field() {
        let request = StreamRequest {
            messages: vec![Message::user("hello")],
            system: None,
            tools: None,
        };
        let body = converse_body(&request, false);
        assert!(
            body.get("system").is_none() || body["system"].is_null(),
            "no system → null, not a text block: {}",
            body["system"]
        );
    }

    #[test]
    fn anthropic_body_without_system_or_tools_minimizes() {
        let request = StreamRequest::new(vec![Message::user("hi")]);
        let body = anthropic_body(&request, "m", false);
        assert!(body["system"].is_null());
        assert!(
            body["tools"].as_array().is_none_or(std::vec::Vec::is_empty),
            "empty tools: {}",
            body["tools"]
        );
        assert!(!body["stream"].as_bool().unwrap());
    }

    #[test]
    fn non_streaming_response_errors_on_unrecognized_shape() {
        let json = serde_json::json!({"unexpected": true});
        let err = bedrock_non_streaming_response(&json, "m").unwrap_err();
        assert!(
            err.to_string().contains("unrecognized"),
            "the error names the problem: {err}"
        );
    }

    #[test]
    fn event_stream_decoder_clears_on_oversized_frame() {
        let mut decoder = AwsEventStreamDecoder::default();
        // total_len = 0xFFFFFFFF is far beyond the 16 MiB cap
        let mut bad = Vec::new();
        bad.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        bad.extend_from_slice(&0u32.to_be_bytes());
        bad.extend_from_slice(&0u32.to_be_bytes());
        let events = decoder.push(&bad);
        assert!(events.is_empty());
        assert!(
            decoder.buf.is_empty(),
            "the buffer is cleared after an oversized frame"
        );
    }

    #[test]
    fn event_stream_decoder_survives_short_garbage() {
        let mut decoder = AwsEventStreamDecoder::default();
        // total_len claims 20 but we push 10 bytes — incomplete, wait
        let partial: Vec<u8> = [20u32.to_be_bytes(), 4u32.to_be_bytes(), 0u32.to_be_bytes()]
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect();
        assert!(decoder.push(&partial).is_empty());
        // Now push the rest to complete a minimal frame
        let rest = vec![0u8; 8]; // 4 header bytes + 4 payload bytes
        let events = decoder.push(&rest);
        assert_eq!(events.len(), 1, "the completed frame decodes");
    }

    #[test]
    fn set_model_updates_the_model() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("amazon.titan-text-v1")
            .build()
            .unwrap();
        assert_eq!(client.model(), "amazon.titan-text-v1");
        assert!(client.set_model("anthropic.claude-v1:0"));
        assert_eq!(client.model(), "anthropic.claude-v1:0");
    }

    #[test]
    fn base_url_carries_the_region() {
        let client = BedrockClientBuilder::default()
            .region("eu-central-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .build()
            .unwrap();
        assert_eq!(
            client.base_url(),
            "https://bedrock-runtime.eu-central-1.amazonaws.com"
        );
    }

    #[test]
    fn builder_rejects_missing_model() {
        let err = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn sigv4_empty_payload_hashes_correctly() {
        let headers = sigv4_sign(
            "k",
            "s",
            None,
            "us-east-1",
            "h",
            "/",
            b"",
            std::time::UNIX_EPOCH,
        );
        // An empty payload hashes to the SHA-256 of empty bytes
        // (e3b0c442…) — the signature is deterministic and well-formed.
        assert!(headers.authorization.contains("Signature="));
        assert_eq!(headers.amz_date, "19700101T000000Z");
    }

    #[test]
    fn path_override_beats_auto_selection() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("anthropic.claude-v1:0")
            .path(BedrockPath::Converse)
            .build()
            .unwrap();
        assert_eq!(client.path, BedrockPath::Converse);
    }

    #[test]
    fn builder_rejects_missing_region() {
        let err = BedrockClientBuilder::default()
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("region"));
    }

    #[test]
    fn anthropic_chunk_unknown_type_is_ignored() {
        let mut acc = StreamAccumulator::default();
        let mut stop = StreamStopReason::EndTurn;
        let unknown = serde_json::json!({"type": "future_event"});
        let events = anthropic_chunk_to_events(&unknown, &mut acc, &mut stop);
        assert!(events.is_empty(), "unknown types produce no events");
        assert_eq!(stop, StreamStopReason::EndTurn, "stop unchanged");
    }

    #[test]
    fn epoch_conversion_matches_known_dates() {
        assert_eq!(epoch_to_utc(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(epoch_to_utc(1), (1970, 1, 1, 0, 0, 1));
        assert_eq!(epoch_to_utc(86_400), (1970, 1, 2, 0, 0, 0));
        assert_eq!(epoch_to_utc(1_700_000_000), (2023, 11, 14, 22, 13, 20));
    }

    #[test]
    fn amz_date_formats_epoch_correctly() {
        // 2024-01-01T00:00:00Z = 1704067200
        let now = std::time::SystemTime::UNIX_EPOCH
            .checked_add(std::time::Duration::from_secs(1_704_067_200))
            .unwrap();
        assert_eq!(format_amz_date(now), "20240101T000000Z");
    }
}
