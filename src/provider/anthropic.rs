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

/// Default Messages API endpoint used when a builder sets none.
///
/// The plain HTTPS endpoint; Bedrock's Anthropic-native path uses its
/// own invoke URL instead of this constant.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Model served when a builder or environment names none.
///
/// A dated snapshot so behavior is reproducible across Anthropic
/// alias updates; override with [`with_model`](AnthropicClientBuilder::with_model).
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

/// Wire protocol version sent as the `anthropic-version` header.
///
/// The Messages API date-versioned protocol this client's request and
/// event shapes are written against.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The output-token budget used when a request carries none.
///
/// Shared by the direct Anthropic path and the Bedrock Anthropic-native
/// path (and, as `maxTokens`, by Bedrock Converse) so switching between
/// them never silently changes the output budget.
pub(super) const DEFAULT_MAX_TOKENS: u32 = 8192;

/// The deadline on the Models API thinking-capability probe.
///
/// Deliberately far shorter than the conversational read timeout: an
/// endpoint without a Models API (a gateway, a local Messages-compatible
/// server) is a normal deployment rather than a failure, and a wedged
/// metadata route must not stall a run's first turn waiting on the timeout
/// a completion was given.
const CAPABILITIES_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// How many models one client caches thinking disclosures for.
///
/// A loose bound rather than a tight one: a run consults a handful of models,
/// and an evicted entry costs the next request one probe. What it rules out is
/// a long-lived client handed a fresh model name per request growing the map
/// for the lifetime of the process.
const CAPABILITY_CACHE_MAX_MODELS: usize = 32;

/// How many consecutive inconclusive probes one model gets before the client
/// stops asking it.
///
/// Three, a deliberately small number. One blip is not a verdict — that is the
/// whole reason [`ProbeOutcome::Inconclusive`] exists and is not filed — but a
/// metadata route that refuses or ignores three probes in a row is stating a
/// deployment fact rather than having bad luck: a scoped key that answers
/// `401` on `/v1/models` while happily serving `/v1/messages`, a gateway that
/// `5xx`-es every path it does not know, or a route that accepts the
/// connection and never responds. Untreated, each of those puts a probe in
/// front of *every* request that sets an effort — on the hanging shape, the
/// full [`CAPABILITIES_PROBE_TIMEOUT`] added to every turn of a run, which is
/// exactly how an Anthropic-compatible gateway profile becomes slow.
///
/// The streak resets whenever the endpoint answers, so an outage shorter than
/// the limit still recovers on the next turn. Past the limit the verdict is
/// *served* rather than filed, and [`CAPABILITY_PROBE_REPROBE_AFTER`] decides
/// when the route is asked again.
const CAPABILITY_PROBE_MISS_LIMIT: u8 = 3;

/// Requests served on the give-up verdict before the route is asked again.
///
/// The give-up verdict is this client's inference — three refusals in a row
/// read as "nothing to disclose" — and an inference cannot be allowed to
/// outlive the outage it was drawn from. Filed as though it were an answer, a
/// route that came back an hour later would leave an adaptive-tier model
/// served the budget spelling it refuses, on every turn, for the life of the
/// client: the one failure mode the definitive/inconclusive split exists to
/// avoid, arriving by a different road.
///
/// So the verdict expires instead of standing. The count is what replaces
/// permanence: at most one probe per this many effort-bearing requests, which
/// keeps a permanently dead route to one probe (and at worst one
/// [`CAPABILITIES_PROBE_TIMEOUT`]) per 29 turns after the initial three, while
/// a route that recovered is believed again within a turn or two of ordinary
/// use. A definitive non-answer — a `404`, a body that was never JSON — is not
/// on this path at all: that endpoint *answered*, and is asked once.
///
/// Must stay above [`CAPABILITY_PROBE_MISS_LIMIT`], and
/// `the_capability_cache_renews_its_give_up_verdict_on_a_bound` fails if the
/// two are ever configured the wrong way round.
const CAPABILITY_PROBE_REPROBE_AFTER: u8 = 32;

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
    /// Changed via [`ApiClient::set_model`] (host-initiated swaps) or
    /// overridden per request via
    /// [`RequestOptions::model`](crate::structured::RequestOptions::model)
    /// — the channel the fallback machinery routes through, leaving the
    /// client itself untouched.
    model: std::sync::Mutex<String>,

    /// The maximum output tokens per response.
    ///
    /// Anthropic requires this field on every request. Defaults to 8192.
    /// Set via [`AnthropicClientBuilder::max_tokens`].
    max_tokens: u32,

    /// The client-level thinking-effort default.
    ///
    /// `None` (the default) sends no `thinking` block; when set, every
    /// request carries the level's `budget_tokens` (per
    /// [`ThinkingEffort`](crate::structured::ThinkingEffort)'s table)
    /// unless the request carries its own
    /// [`RequestOptions::effort`](crate::structured::RequestOptions::effort).
    /// A model whose disclosure names a different thinking control gets that
    /// one instead; set via [`AnthropicClientBuilder::thinking_effort`].
    thinking_effort: Option<crate::structured::ThinkingEffort>,

    /// What the serving model disclosed about the thinking it accepts.
    ///
    /// Shared behind an [`std::sync::Arc`] because the streaming path reads
    /// it from inside a `'static` stream, and locked because a probe
    /// resolves asynchronously while later requests are being built. Entries
    /// are keyed by model, so a swap mid-probe cannot have one model read
    /// another's answer and the map is never cleared wholesale — the cached
    /// answer belongs to whichever request names that model, the client's own
    /// or a [`RequestOptions::model`](crate::structured::RequestOptions::model)
    /// override.
    thinking_capabilities: std::sync::Arc<std::sync::Mutex<CapabilityCache>>,

    /// Whether prior-turn reasoning is replayed on outgoing requests.
    ///
    /// Off by default: requests are byte-identical to a client that
    /// never saw reasoning. When on, each history
    /// [`MessagePart::Thinking`] renders as its own capped
    /// `<thinking>` text block ahead of the turn's other content, in
    /// part order — wire-legal on every Messages-compatible endpoint,
    /// unlike an unsigned native thinking block.
    replay_reasoning: bool,

    /// Whether requests carry prompt-cache breakpoints.
    ///
    /// Off by default: the wire is byte-identical to a client that never
    /// asked to cache. When on, every request marks up to four
    /// `cache_control` breakpoints (the last tool, the system block, and
    /// the last two content blocks of the final message) so the server
    /// can serve the stable conversation prefix from cache across an
    /// agentic loop's turns. See
    /// [`AnthropicClientBuilder::prompt_caching`].
    prompt_caching: bool,

    /// The rate-limit snapshot from the most recent response.
    ///
    /// Advisory state refreshed per response by
    /// [`post_json_checked`](super::post_json_checked); read through
    /// [`last_rate_limit`](Self::last_rate_limit). A clone of this slot
    /// rides each streaming task so a stream's response updates the
    /// accessor on the client itself. Anthropic answers the
    /// `rate-limit-*` spelling; both families are read.
    rate_limit: super::RateLimitSlot,
}

impl AnthropicClient {
    /// The rate-limit snapshot from the most recent response, if the
    /// provider sent one.
    ///
    /// Advisory only, refreshed per response — including failed ones, so
    /// a 429's `rate-limit-remaining` is readable right after the error
    /// — and cleared to `None` by the next response that reports no
    /// family. `None` is **unreported**, never a measured zero. See
    /// [`RateLimitSnapshot`](crate::api::error::RateLimitSnapshot).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl::provider::AnthropicClient;
    ///
    /// let client = AnthropicClient::builder()
    ///     .with_api_key("sk-ant-test")
    ///     .build()
    ///     .expect("the builder needs only static fields");
    /// // Before any response: unreported — None, never a zero figure.
    /// assert!(client.last_rate_limit().is_none());
    /// ```
    #[must_use]
    pub fn last_rate_limit(&self) -> Option<crate::api::error::RateLimitSnapshot> {
        self.rate_limit.last()
    }

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
    /// Delegates to [`post_json_checked`](super::post_json_checked), which
    /// classifies non-success responses (auth rejections, rate limits with
    /// their server-advised delay) into structured [`ApiError`] variants.
    ///
    /// Sends the JSON body with `x-api-key` and `anthropic-version`
    /// headers, and reads the forensics headers (the native `request-id`
    /// and the `rate-limit-*` family) from every response into
    /// `rate_limit` when given.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::http`] if the HTTP request fails, or the
    /// classified variant for a non-success status code.
    async fn post_messages(
        http: &reqwest::Client,
        url: &str,
        api_key: &str,
        body: &Value,
        rate_limit: Option<&super::RateLimitSlot>,
    ) -> Result<reqwest::Response, ApiError> {
        let mut key_header = reqwest::header::HeaderValue::from_str(api_key)
            .map_err(|e| ApiError::auth_invalid_key(format!("invalid api key header: {e}")))?;
        key_header.set_sensitive(true);
        super::post_json_checked(
            http,
            url,
            &[
                (
                    reqwest::header::HeaderName::from_static("x-api-key"),
                    key_header,
                ),
                (
                    reqwest::header::HeaderName::from_static("anthropic-version"),
                    reqwest::header::HeaderValue::from_static(ANTHROPIC_VERSION),
                ),
            ],
            body,
            rate_limit,
        )
        .await
    }

    /// Build the full URL for the Anthropic Messages API endpoint.
    ///
    /// Appends `/v1/messages` to the client's `base_url`. All four
    /// `ApiClient` methods (`stream_messages`, `create_message`, and their
    /// `*_with_options` variants) POST to this URL.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// The credential pair every Anthropic wire call carries.
    ///
    /// `x-api-key` (marked sensitive, exactly as the completions path
    /// marks it) and the `anthropic-version` the endpoint requires —
    /// the listing GET refuses without it, so the probe path must send
    /// the same pair a Messages request sends.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::auth_invalid_key`] when the key cannot form
    /// a header value.
    fn credential_headers(
        &self,
    ) -> Result<Vec<(reqwest::header::HeaderName, reqwest::header::HeaderValue)>, ApiError> {
        let mut key_header = reqwest::header::HeaderValue::from_str(&self.api_key)
            .map_err(|e| ApiError::auth_invalid_key(format!("invalid api key header: {e}")))?;
        key_header.set_sensitive(true);
        Ok(vec![
            (
                reqwest::header::HeaderName::from_static("x-api-key"),
                key_header,
            ),
            (
                reqwest::header::HeaderName::from_static("anthropic-version"),
                reqwest::header::HeaderValue::from_static(ANTHROPIC_VERSION),
            ),
        ])
    }

    /// GET the account's model listing under the probe transport.
    ///
    /// `{base}/v1/models` — the credential pair applied, the probe
    /// deadline and full failure classification from
    /// [`get_checked`](super::get_checked), so an auth rejection or a
    /// throttle answers a probe with the same variant and forensics a
    /// Messages failure carries.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::auth_invalid_key`] when the key cannot form
    /// a header value, [`ApiError::http`] on transport failure or probe
    /// deadline, or the classified variant for a non-success status.
    async fn fetch_model_listing(&self) -> Result<reqwest::Response, ApiError> {
        let headers = self.credential_headers()?;
        let url = format!("{}/v1/models", self.base_url);
        super::get_checked(&self.http, &url, &headers, Some(&self.rate_limit)).await
    }

    /// Map a `/v1/models` document to listing entries.
    ///
    /// `data[].id` is the usable id, `display_name` the human name, and
    /// `created_at` the RFC 3339 stamp verbatim. The endpoint discloses
    /// no window or modality facts, so metadata is the curated opinion
    /// where one exists — this is where that rule earns its keep:
    /// Anthropic's own listing would otherwise say nothing about
    /// context windows.
    ///
    /// `has_more` / `next_page` mark a further page; continuation is not
    /// followed — the first page is this surface's documented bound (see
    /// [`ApiClient::list_models`](crate::api::ApiClient::list_models)), so
    /// a completed page-one answer is complete *within that bound*, not
    /// silently truncated.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::api`] when the answered body carries no
    /// `data` array — a document that never spoke of models is a
    /// failure, never an empty catalog.
    fn parse_models(value: &Value) -> Result<Vec<crate::api::probe::ModelInfo>, ApiError> {
        use crate::api::probe::ModelInfo;
        let entries = super::listing_entries(value, &["data"], "model listing")?;
        Ok(entries
            .iter()
            .filter_map(|entry| {
                let id = entry.get("id").and_then(Value::as_str)?;
                let metadata = crate::api::probe::curated_model_metadata(id)
                    .unwrap_or_else(crate::api::probe::ModelMetadata::unreported);
                let mut info = ModelInfo::new(id).with_metadata(metadata);
                if let Some(name) = entry.get("display_name").and_then(Value::as_str) {
                    info = info.with_display_name(name);
                }
                if let Some(stamp) = entry.get("created_at").and_then(Value::as_str) {
                    info = info.with_created(stamp);
                }
                Some(info)
            })
            .collect())
    }

    /// Build a request body under this client's full configuration.
    ///
    /// Wraps [`build_request_body`] with the client-level knobs
    /// (`max_tokens`, `replay_reasoning`) and, when
    /// [`prompt_caching`](Self::prompt_caching) is on, applies
    /// [`apply_prompt_cache_breakpoints`] to the result so every
    /// outgoing request carries the cache markers. Both wire paths —
    /// streaming and non-streaming — go through here, so the two can
    /// never disagree about which requests ask to be cached.
    fn build_body(
        &self,
        spec: &RequestBodySpec<'_>,
        stream: bool,
        effort: Option<crate::structured::ThinkingEffort>,
    ) -> Value {
        let mut body =
            build_request_body(spec, stream, self.max_tokens, effort, self.replay_reasoning);
        if self.prompt_caching {
            apply_prompt_cache_breakpoints(&mut body);
        }
        body
    }

    /// Snapshot what a thinking probe needs to address one model.
    ///
    /// Built per request because a request can name its own model
    /// ([`RequestOptions::model`](crate::structured::RequestOptions::model))
    /// and the client's model can be hot-swapped between requests; the
    /// expensive parts it carries are the pooled handle (an `Arc` clone) and
    /// the shared capability cache.
    fn thinking_probe(&self, model: &str) -> ThinkingProbe {
        ThinkingProbe {
            http: self.http.clone(),
            api_key: self.api_key.clone(),
            url: format!("{}/v1/models/{model}", self.base_url),
            cache: std::sync::Arc::clone(&self.thinking_capabilities),
            model: model.to_string(),
            configured_max_tokens: self.max_tokens,
        }
    }

    /// Build a typed [`NonStreamingResponse`] from Anthropic's native JSON.
    ///
    /// Anthropic's native response already carries a `content` array of typed
    /// blocks, so this reads them directly into [`MessagePart`]s: `text` blocks
    /// become [`MessagePart::Text`] parts, `tool_use` blocks become
    /// [`MessagePart::ToolCall`] parts, and `thinking` / `redacted_thinking`
    /// blocks become [`MessagePart::Thinking`] parts carrying their signature
    /// or opaque payload, preserving their original order — a continuation
    /// under extended thinking must return those blocks verbatim.
    /// Maps `stop_reason` via [`StreamStopReason::from_api_str`] (Anthropic
    /// reports tool invocations as `"tool_use"`, aliased to `ToolCall`),
    /// defaulting to `EndTurn` on an unrecognized or missing value. Reads
    /// `usage.input_tokens` / `usage.output_tokens` into [`Usage`], defaulting
    /// to zero when the `usage` object is absent.
    fn build_response(raw: &Value) -> crate::api::NonStreamingResponse {
        let mut parts: Vec<MessagePart> = Vec::new();
        if let Some(blocks) = raw.get("content").and_then(|c| c.as_array()) {
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            parts.push(MessagePart::text(text));
                        }
                    }
                    Some("thinking") => {
                        let text = block.get("thinking").and_then(|t| t.as_str()).unwrap_or("");
                        let signature = block.get("signature").and_then(|s| s.as_str());
                        if !text.is_empty() || signature.is_some() {
                            parts.push(MessagePart::Thinking {
                                text: text.to_string(),
                                signature: signature.map(str::to_string),
                                redacted: None,
                            });
                        }
                    }
                    Some("redacted_thinking") => {
                        if let Some(data) = block.get("data").and_then(|d| d.as_str()) {
                            parts.push(MessagePart::Thinking {
                                text: String::new(),
                                signature: None,
                                redacted: Some(data.to_string()),
                            });
                        }
                    }
                    Some("tool_use") => {
                        let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let input = block.get("input").cloned().unwrap_or(Value::Null);
                        parts.push(MessagePart::tool_call(id, name, input));
                    }
                    _ => {}
                }
            }
        }
        let stop_reason = raw
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .and_then(StreamStopReason::from_api_str)
            .unwrap_or(StreamStopReason::EndTurn);
        let usage = extract_usage(raw);
        crate::api::NonStreamingResponse {
            message: Message::new(Role::Assistant, parts),
            stop_reason,
            usage,
        }
    }
}

/// Extract token [`Usage`] from Anthropic's native `usage` field.
///
/// Reads `usage.input_tokens` and `usage.output_tokens` via
/// [`extract_usage_object`]. Returns `None` when the `usage` object is absent
/// or when both counts are zero, matching the convention used by the
/// streaming emitter in `on_message_delta`.
fn extract_usage(raw: &Value) -> Option<Usage> {
    extract_usage_object(raw.get("usage")?)
}

/// Extract token [`Usage`] from a single Anthropic `usage` object.
///
/// Reads the object's `input_tokens` and `output_tokens`, defaulting
/// each to zero when absent or non-numeric. The input is normalized:
/// Anthropic's raw `input_tokens` **excludes** cache traffic, so
/// `cache_read_input_tokens` and `cache_creation_input_tokens` are
/// added into the reported input, with the read share kept as
/// [`Usage::cached_input_tokens`] and the creation spend as
/// [`Usage::cache_write_tokens`] — both stay subsets of the input, the
/// documented invariant. Returns `None` when the normalized input and
/// the output are both zero, so an all-zero report is
/// indistinguishable from a missing one — the convention both response
/// paths apply. Shared by the non-streaming `usage` field
/// ([`extract_usage`]) and the streaming `message_start` latch in
/// `on_message_start`.
fn extract_usage_object(usage: &Value) -> Option<Usage> {
    let raw_input = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
    let cached = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
    let creation = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
    let input = raw_input.saturating_add(cached).saturating_add(creation);
    (input > 0 || output > 0).then(|| {
        Usage::new(input, output)
            .with_cached_input(cached)
            .with_cache_write(creation)
    })
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

    fn supports_tool_constraints(&self) -> bool {
        true
    }

    fn list_models(
        &self,
    ) -> Pin<
        Box<dyn Future<Output = Result<Vec<crate::api::probe::ModelInfo>, ApiError>> + Send + '_>,
    > {
        Box::pin(async move {
            let response = self.fetch_model_listing().await?;
            let body = super::read_bounded_body(response).await?;
            let value: Value = serde_json::from_slice(&body)?;
            Self::parse_models(&value)
        })
    }

    fn ping(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::probe::PingReport, ApiError>> + Send + '_>>
    {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let response = self.fetch_model_listing().await?;
            let meta = crate::provider::forensics::capture_provider_meta(response.headers());
            let mut report = crate::api::probe::PingReport::new(started.elapsed());
            if let Some(rate_limit) = meta.rate_limit {
                report = report.with_rate_limit(rate_limit);
            }
            if let Some(request_id) = meta.request_id {
                report = report.with_request_id(request_id);
            }
            Ok(report)
        })
    }

    fn stream_messages(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.stream_messages_with_options(request, crate::structured::RequestOptions::default())
    }

    fn create_message(
        &self,
        request: &crate::api::StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_>>
    {
        self.create_message_with_options(request, crate::structured::RequestOptions::default())
    }

    fn stream_messages_with_options(
        &self,
        request: &crate::api::StreamRequest,
        options: crate::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        #[cfg(feature = "grammar")]
        if matches!(&options.tool_constraint, ToolConstraint::Grammar(_)) {
            return Box::pin(futures::stream::once(async move {
                Err(grammar_unsupported_error())
            }));
        }
        if options.response_format.as_ref().is_some_and(|rf| rf.strict) {
            return Box::pin(futures::stream::once(async move {
                Err(strict_unsupported_error("Anthropic"))
            }));
        }
        let effort = options.effort.or(self.thinking_effort);
        if options.response_format.is_some() && effort.and_then(anthropic_thinking_budget).is_some()
        {
            return Box::pin(futures::stream::once(async move {
                Err(effort_tool_choice_conflict_error())
            }));
        }
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| crate::error::recover_guard(self.model.lock()).clone());
        let rf = options.response_format.as_ref();
        let body = self.build_body(
            &RequestBodySpec {
                model: &model,
                messages: &request.messages,
                system: system.as_deref(),
                tools: tools.as_deref(),
                response_format: rf,
                tool_constraint: &options.tool_constraint,
            },
            true,
            effort,
        );
        let url = self.messages_url();
        let api_key = self.api_key.clone();
        let http = self.http.clone();
        let rate_limit = self.rate_limit.clone();
        let probe = effort.is_some().then(|| self.thinking_probe(&model));

        Box::pin(async_stream::try_stream! {
            let mut body = body;
            if let Some(probe) = &probe {
                let wire = probe.resolve(effort).await?;
                apply_model_thinking(&mut body, wire, probe.configured_max_tokens);
            }
            let resp =
                Self::post_messages(&http, &url, &api_key, &body, Some(&rate_limit)).await?;
            let meta = crate::provider::forensics::capture_provider_meta(resp.headers());
            let mut sse = SseReader::from_response(resp);
            let mut emitter = StreamEmitter::default();

            while let Some((event_type, data)) = sse
                .next_event()
                .await
                .map_err(|e| e.with_provider_meta(meta.clone()))?
            {
                emitter.process_event(&event_type, data);
                if emitter.error_recorded() {
                    break;
                }
                for ev in emitter.drain() {
                    yield ev;
                }
            }

            for ev in emitter
                .finish()
                .map_err(|e| e.with_provider_meta(meta.clone()))?
            {
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
        #[cfg(feature = "grammar")]
        if matches!(&options.tool_constraint, ToolConstraint::Grammar(_)) {
            return Box::pin(async move { Err(grammar_unsupported_error()) });
        }
        if options.response_format.as_ref().is_some_and(|rf| rf.strict) {
            return Box::pin(async move { Err(strict_unsupported_error("Anthropic")) });
        }
        let effort = options.effort.or(self.thinking_effort);
        if options.response_format.is_some() && effort.and_then(anthropic_thinking_budget).is_some()
        {
            return Box::pin(async move { Err(effort_tool_choice_conflict_error()) });
        }
        let system = request.system.clone();
        let tools = request.tools.clone();
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| crate::error::recover_guard(self.model.lock()).clone());
        let rf = options.response_format.as_ref();
        let body = self.build_body(
            &RequestBodySpec {
                model: &model,
                messages: &request.messages,
                system: system.as_deref(),
                tools: tools.as_deref(),
                response_format: rf,
                tool_constraint: &options.tool_constraint,
            },
            false,
            effort,
        );
        let url = self.messages_url();
        let probe = effort.is_some().then(|| self.thinking_probe(&model));
        Box::pin(async move {
            let mut body = body;
            if let Some(probe) = &probe {
                let wire = probe.resolve(effort).await?;
                apply_model_thinking(&mut body, wire, probe.configured_max_tokens);
            }
            let resp = Self::post_messages(
                &self.http,
                &url,
                &self.api_key,
                &body,
                Some(&self.rate_limit),
            )
            .await?;
            let meta = crate::provider::forensics::capture_provider_meta(resp.headers());
            let resp = super::read_bounded_body(resp)
                .await
                .map_err(|e| e.with_provider_meta(meta.clone()))?;
            let raw = serde_json::from_slice::<Value>(&resp)
                .map_err(|e| ApiError::http(e.to_string()).with_provider_meta(meta))?;
            Ok(Self::build_response(&raw))
        })
    }
}

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

    /// The client-level thinking-effort default.
    ///
    /// `None` (the default) sends no `thinking` block. When set, every
    /// request carries the level's `budget_tokens` and a `max_tokens`
    /// raised to fit it (see
    /// [`ThinkingEffort`](crate::structured::ThinkingEffort)'s table).
    /// Read by [`build`](Self::build) and stored on [`AnthropicClient`].
    thinking_effort: Option<crate::structured::ThinkingEffort>,

    /// Whether prior-turn reasoning is replayed on outgoing requests.
    ///
    /// Set via [`AnthropicClientBuilder::replay_reasoning`]; see the
    /// client field's docs for the wire shape.
    replay_reasoning: bool,

    /// Whether requests carry prompt-cache breakpoints.
    ///
    /// Set via [`AnthropicClientBuilder::prompt_caching`]; see the
    /// client field's docs for the breakpoint placement.
    prompt_caching: bool,

    /// Shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// Holds the timeout, connection-pool, and TCP knobs that apply to the
    /// internally-built `reqwest::Client`, or an externally-supplied client
    /// injected via [`with_http_client`](Self::with_http_client).
    http: super::HttpClientConfig,

    /// Error-reporting state for values seeded by the provider-profile
    /// builders.
    ///
    /// Carries the credential variable a seeded key came from; consulted
    /// by [`build`](Self::build).
    seed: super::ProfileSeed,
}

impl Default for AnthropicClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            thinking_effort: None,
            replay_reasoning: false,
            prompt_caching: false,
            http: super::HttpClientConfig::default(),
            seed: super::ProfileSeed::default(),
        }
    }
}

impl AnthropicClientBuilder {
    /// Set the API key for authentication.
    ///
    /// Required — [`build`](Self::build) returns an error if this is not set.
    /// The key is sent as the `x-api-key` header on every request. An
    /// explicit key supersedes a seeded one, including the missing-key
    /// error the seed would otherwise name.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self.seed.key_env_name = None;
        self
    }

    /// Seed the API key from a resolved environment lookup.
    ///
    /// Provider-profile builders call this with the value of the
    /// provider's credential variable (or `None` when unset). An absent
    /// seed leaves the key unset — [`build`](Self::build) then fails
    /// naming `missing_env` — while an explicit
    /// [`with_api_key`](Self::with_api_key) overrides the seed.
    #[cfg(feature = "zai")]
    #[must_use]
    pub(super) fn with_key_seed(mut self, key: Option<String>, missing_env: &'static str) -> Self {
        self.api_key = key;
        self.seed.seed_key(missing_env);
        self
    }

    /// Set the base URL for API requests.
    ///
    /// Defaults to `https://api.anthropic.com`. Override when targeting a
    /// proxy, gateway, or Anthropic-compatible endpoint (e.g. Z.AI at
    /// `https://api.z.ai/api/anthropic`). Trailing `/` separators are
    /// trimmed, so joined request paths never contain `//` — a `…/v1/`
    /// base behaves identically to `…/v1`.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Set the default model identifier.
    ///
    /// The model string is sent as the `model` field on every request,
    /// unless a per-request
    /// [`RequestOptions::model`](crate::structured::RequestOptions::model)
    /// override names another. Can also be swapped wholesale at runtime
    /// via [`AnthropicClient::set_model`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the `max_tokens` sent with every request.
    ///
    /// Anthropic requires this field on every request — unlike OpenAI,
    /// which defaults it server-side — and it must be at least 1:
    /// [`build`](Self::build) rejects a zero value instead of letting
    /// the server answer a guaranteed 400. It bounds the length of a
    /// single model response. Defaults to 8192; increase it for
    /// long-form generation.
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set the client-level thinking-effort default.
    ///
    /// Every request then carries a `thinking` block with the level's
    /// `budget_tokens` (per
    /// [`ThinkingEffort`](crate::structured::ThinkingEffort)'s table),
    /// and a `max_tokens` below `budget_tokens + 1` rises to fit the
    /// budget — except the
    /// [`None`](crate::structured::ThinkingEffort::None) rung, the
    /// client-wide off, which sends `thinking: {"type": "disabled"}` —
    /// omitting the block instead means adaptive thinking on any model
    /// that thinks by default, so the off is stated, not implied. A
    /// per-request
    /// [`RequestOptions::effort`](crate::structured::RequestOptions::effort)
    /// always wins; an unset client default leaves unconfigured requests
    /// byte-identical to prior versions. A
    /// [`response_format`](crate::structured::RequestOptions::response_format)
    /// request under a default that budgets is rejected up front —
    /// extended thinking cannot combine with the forced tool choice a
    /// response format synthesizes — so configure effort per request
    /// (or to the `None` rung) when both are needed. An effort that
    /// budgets also demands the assistant's original signed thinking
    /// blocks on continuations: a history whose tool-use message
    /// carries an unsigned thinking part (recorded before this
    /// version, cross-provider, or hand-built) still sends `tool_use`
    /// alone, which the Messages API rejects — clear the effort for
    /// such histories or re-record them against this version, whose
    /// response path captures signatures. Under the `None` rung,
    /// history reasoning that carries text instead rides as the
    /// capped plain-text replay (see
    /// [`replay_reasoning`](Self::replay_reasoning)); a redacted
    /// part has no text to replay and is dropped.
    ///
    /// A request that resolves an effort first reads what the model
    /// disclosed about thinking from the Models API — once per model, under
    /// its own short deadline — and writes the control that model accepts: a
    /// model taking depth from `output_config.effort` gets the rung's label
    /// there instead of a `budget_tokens` block, and the off rung takes
    /// `{"type":"between_tools"}` on a model that refuses `"disabled"`. An
    /// endpoint that discloses nothing (a gateway, a local
    /// Messages-compatible server) keeps the table's own mapping, and a
    /// request with no effort never probes.
    #[must_use]
    pub fn thinking_effort(mut self, effort: crate::structured::ThinkingEffort) -> Self {
        self.thinking_effort = Some(effort);
        self
    }

    /// Replay prior-turn reasoning on outgoing requests.
    ///
    /// Off by default. When enabled, each history
    /// [`MessagePart::Thinking`] renders as its own capped
    /// `<thinking>` text block ahead of the turn's other content, in
    /// part order, so reasoning-capable models keep their working
    /// memory across turns. The blocks are plain text — every
    /// Messages-compatible endpoint accepts them, including gateways
    /// that reject unsigned native thinking blocks. The `None`
    /// effort rung renders text-carrying history reasoning the same
    /// way with the flag left off: native thinking blocks are
    /// wire-illegal on a request that disables thinking, and dropping
    /// the parts silently would lose the context they carry. A
    /// redacted part has no text to replay and is dropped — only an
    /// effort that budgets returns its payload natively.
    #[must_use]
    pub fn replay_reasoning(mut self, enabled: bool) -> Self {
        self.replay_reasoning = enabled;
        self
    }

    /// Mark prompt-cache breakpoints on every request.
    ///
    /// Off by default. When enabled, each request carries
    /// `cache_control` breakpoints — on the last tool definition, on the
    /// system block, and on the last two content blocks of the final
    /// message — so the provider caches the stable conversation prefix
    /// and later turns in the same loop re-use it instead of
    /// re-processing the whole context. On `api.anthropic.com` this is
    /// a latency and cost win for any multi-turn loop; a read of the
    /// cache shows up in [`Usage::cached_input_tokens`]. Endpoints
    /// that reject the extra field should leave this off.
    ///
    /// [`Usage::cached_input_tokens`]: crate::stream::Usage::cached_input_tokens
    #[must_use]
    pub fn prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Set the HTTP read timeout.
    ///
    /// This bounds *idleness* — how long a gap between bytes on the
    /// connection may last — not the total request lifetime: a stream
    /// that keeps producing bytes runs as long as it keeps producing.
    /// Defaults to 120 seconds. Ignored when a client was supplied via
    /// [`with_http_client`](Self::with_http_client); bound the whole
    /// turn with a [`StreamHandler`](crate::stream::handler::StreamHandler)
    /// total timeout instead.
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

    /// Construct the [`AnthropicClient`] from the builder's configuration.
    ///
    /// Creates the internal `reqwest::Client` with the configured timeouts,
    /// and validates that an API key was provided.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if no API key was set via
    /// [`with_api_key`](Self::with_api_key) — naming the provider's
    /// credential variable when the key came seeded from a
    /// provider-profile builder — or when `max_tokens` is
    /// zero (the Messages API requires at least 1).
    pub fn build(self) -> Result<AnthropicClient, ApiError> {
        let api_key = self.api_key.ok_or_else(|| match self.seed.key_env_name {
            Some(name) => ApiError::auth_invalid_key(format!("{name} not set")),
            None => ApiError::auth_invalid_key("API key not provided"),
        })?;
        let http = self.http.build()?;
        if self.max_tokens == 0 {
            return Err(ApiError::config(
                "max_tokens must be at least 1 — the Messages API rejects 0",
            ));
        }

        Ok(AnthropicClient {
            http,
            api_key,
            base_url: self.base_url,
            model: std::sync::Mutex::new(self.model),
            max_tokens: self.max_tokens,
            thinking_effort: self.thinking_effort,
            thinking_capabilities: std::sync::Arc::new(std::sync::Mutex::new(
                CapabilityCache::default(),
            )),
            replay_reasoning: self.replay_reasoning,
            prompt_caching: self.prompt_caching,
            rate_limit: super::RateLimitSlot::default(),
        })
    }
}

/// The per-request inputs to [`build_request_body`].
///
/// Carries the request-shape knobs: the model, conversation, tools, and
/// the structured-output / tool-call constraints. `stream`, `max_tokens`,
/// and the resolved thinking `effort` are passed separately because they
/// are per-call / per-client rather than per-request-shape.
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

/// The config error for grammar constraints on the Messages API.
///
/// Anthropic has no grammar-constrained decoding, so a requested grammar
/// is rejected loudly instead of silently downgraded to an unconstrained
/// request — a caller asking for structural guarantees must not discover
/// their absence from malformed tool calls downstream.
#[cfg(feature = "grammar")]
fn grammar_unsupported_error() -> ApiError {
    ApiError::config_validation(
        "the Anthropic Messages API has no grammar-constrained decoding; \
         use ToolConstraint::Strict or an OpenAI-compatible endpoint",
    )
}

/// The error for a thinking-effort request the Messages API cannot
/// combine with a forced tool choice.
///
/// Extended thinking requires `tool_choice: "auto"`, and a
/// `response_format` synthesizes a forced tool choice — so a request
/// carrying both would be rejected by the API with a 400 after leaving
/// the client. Rejected up front instead, naming the effective effort
/// (the request's own level or the client-level default it inherits).
fn effort_tool_choice_conflict_error() -> ApiError {
    ApiError::config_validation(
        "the Anthropic Messages API rejects extended thinking combined \
         with a forced tool choice; drop the effort (request-level or \
         client default) or the response format",
    )
}

/// The error for a model the provider says cannot turn thinking off.
///
/// The off rung has no spelling on a model whose disclosure refuses both off
/// shapes, so the rung is refused up front: sent anyway, the request would
/// 400 over `thinking.type` without saying that no accepted value exists,
/// and the two moves that would work — leave the effort unset and accept the
/// model's own thinking, or move to a model with an off level — are not in
/// that message.
fn thinking_unavailable_error(model: &str) -> ApiError {
    ApiError::config_validation(format!(
        "the model {model} does not accept `thinking: {{\"type\":\"disabled\"}}` \
         and disclosed no thinking setting that turns it off, so thinking \
         cannot be turned off on it; drop the effort (request-level or \
         client default) or move to a model with an off level"
    ))
}

/// What a serving model disclosed about the thinking controls it accepts.
///
/// Read from the Models API's `capabilities.thinking.types` object. Every
/// member is an [`Option`] because a disclosure can be partial: `None` means
/// the endpoint never said, which is not the answer `Some(false)` — that the
/// model rejects the spelling — gives. Undisclosed leaves
/// [`anthropic_thinking_budget`]'s own mapping in charge, which is what
/// gateways and Messages-compatible servers without a Models API get.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ThinkingCapabilities {
    /// `capabilities.thinking.types.disabled.supported`.
    ///
    /// Anthropic documents this as `true` also for a model with no thinking
    /// at all, and as possibly-true on a model that still 400s a `disabled`
    /// request for another reason, such as an effort level the model
    /// disallows with thinking off.
    disabled: Option<bool>,

    /// `capabilities.thinking.types.between_tools.supported`.
    ///
    /// Whether the model accepts `thinking: {"type":"between_tools"}`, the
    /// spelling under which it reasons only while composing between tool
    /// calls — on the models that disclose it, the quietest setting they
    /// take, and the closest approach to off that a model refusing
    /// `"disabled"` still offers.
    ///
    /// Read for exactly one decision: the fallback the off rung takes when
    /// [`Self::disabled`] is `Some(false)` (see
    /// [`ThinkingWire::BetweenTools`]). A budgeted rung is never rewritten
    /// to it — that would trade a depth someone asked for for a setting
    /// nobody did. Only `Some(true)` licenses the fallback: `Some(false)`
    /// rules it out, and `None` — the endpoint never said — leaves the off
    /// rung on `"disabled"`, where an endpoint that discloses nothing stays
    /// the arbiter of whether the request runs.
    between_tools: Option<bool>,

    /// `capabilities.thinking.types.enabled.supported`.
    ///
    /// Not a documented member: read opportunistically because it is the
    /// direct answer to "does this model take a `budget_tokens` block", and
    /// treated as undisclosed when absent.
    enabled: Option<bool>,

    /// `capabilities.thinking.types.adaptive.supported`.
    ///
    /// Undocumented, read with [`Self::enabled`]'s caution; together the two
    /// identify a model that takes depth from `output_config.effort` instead
    /// of a budget.
    adaptive: Option<bool>,
}

impl ThinkingCapabilities {
    /// Assemble a disclosure from the tri-states a model answer yields.
    ///
    /// Absent members stay `None` rather than defaulting to `false`, so an
    /// endpoint answering part of the shape is trusted only for the part it
    /// answered. An unrelated body — an error document served under a
    /// success status, a list response — simply discloses nothing.
    const fn from_states(
        disabled: Option<bool>,
        between_tools: Option<bool>,
        enabled: Option<bool>,
        adaptive: Option<bool>,
    ) -> Self {
        Self {
            disabled,
            between_tools,
            enabled,
            adaptive,
        }
    }

    /// Whether the disclosure carries anything this client can act on.
    ///
    /// An all-`None` answer is the same non-answer as a probe that never
    /// resolved, and caches identically so a silent endpoint is not re-asked
    /// per request.
    fn is_empty(self) -> bool {
        self.disabled.is_none()
            && self.between_tools.is_none()
            && self.enabled.is_none()
            && self.adaptive.is_none()
    }
}

/// What one model's probe learned about the thinking it accepts, once the
/// endpoint has actually answered.
///
/// Only a definitive answer is representable here. A probe that got no answer
/// at all — a timeout, a `503` — is [`ProbeOutcome::Inconclusive`] and is
/// never filed, because nothing clears a filed answer later: deciding a
/// model's wire shape from one lost round trip would keep the verdict and the
/// 400s it causes for the rest of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedDisclosure {
    /// The endpoint answered and stated at least one thinking control.
    ///
    /// The tri-states [`thinking_wire`] reads to pick the wire shape; a member
    /// the answer left out keeps [`anthropic_thinking_budget`]'s own mapping
    /// in charge rather than guessing at it.
    Known(ThinkingCapabilities),

    /// The endpoint answered and stated nothing usable about thinking.
    ///
    /// Filed on purpose. A `404` (an endpoint with no Models API), a gateway
    /// answering `200` with HTML, and JSON listing none of the thinking
    /// members are ordinary deployments rather than failures, and an uncached
    /// miss would put a probe under a 5-second deadline in front of every
    /// request that sets an effort. It is also the narrowest verdict a probe
    /// can reach — the ladder's own mapping stands, which is what every
    /// endpoint that never heard of this client already gets. It is also
    /// where a model runs after [`CAPABILITY_PROBE_MISS_LIMIT`] consecutive
    /// inconclusive probes — a route that never answers has said, in effect,
    /// that it has nothing to disclose. Served rather than filed there: that
    /// reading is this client's, so [`CAPABILITY_PROBE_REPROBE_AFTER`] brings
    /// the route back up for a probe when it may have recovered.
    Undisclosed,
}

/// The outcome of one Models API probe.
///
/// The two failures it distinguishes are what a probe is for: an endpoint
/// that *said* something is believed, and an endpoint that was busy, wedged,
/// or mid-credential-rotation is not. A cache that cannot tell them apart
/// turns one blip into a permanent decision — the disclosure a probe never
/// read, held for the life of the client.
enum ProbeOutcome {
    /// The endpoint answered, or definitively declined to. Filed under the
    /// model asked about, and already applied to this request.
    Answered(CachedDisclosure),

    /// Nothing was learned: the request never completed, or the endpoint
    /// refused or failed it in a way that says nothing about thinking. This
    /// request falls back to the ladder's own mapping — the documented answer
    /// for an endpoint that discloses nothing — and the next one asks again,
    /// up to [`CAPABILITY_PROBE_MISS_LIMIT`] times in a row; a route that
    /// cannot answer is then run as one with nothing to disclose rather than
    /// probed ahead of every turn, until [`CAPABILITY_PROBE_REPROBE_AFTER`]
    /// turns of that say enough time has passed to ask once more.
    Inconclusive,
}

/// The client's thinking-capability cache, keyed by model id.
///
/// Keyed rather than one slot because one client serves more than one model:
/// [`RequestOptions::model`](crate::structured::RequestOptions::model) is how
/// the fallback machinery routes a retry, and
/// [`set_model`](ApiClient::set_model) can land between turns. With a single
/// slot, every model but the cached one missed on every request and paid a
/// probe round trip to find that out; a host alternating two models paid one
/// probe per request forever. Keyed, each model pays once.
///
/// Entries are never cleared wholesale — an answer about `claude-opus-5` is
/// still true of `claude-opus-5` whoever asked — and
/// [`CAPABILITY_CACHE_MAX_MODELS`] bounds the map for a host handed a new
/// model name per request.
/// One model's row in the thinking-capability cache.
///
/// The answer and the miss streak live in one row rather than two maps because
/// the streak belongs to the model it was accumulated against: kept apart, a
/// miss counter for a model name nobody is asking about any more would outlive
/// every bound on the answers and keep counting toward a verdict nobody reads.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct CapabilityEntry {
    /// The endpoint's own definitive answer, once it gave one.
    ///
    /// `Some` only for what the endpoint actually said: a disclosure, a `404`
    /// or `405`, a non-JSON body, a JSON answer that named no thinking member.
    /// `None` for everything else — including the [`CachedDisclosure::Undisclosed`]
    /// the give-up streak serves, which lives in [`Self::misses`] instead,
    /// because an inference this client drew is not the endpoint's answer and
    /// must stay open to correction.
    answer: Option<CachedDisclosure>,

    /// Inconclusive probes so far, and once past the limit, requests served on
    /// the give-up verdict.
    ///
    /// One counter carries two thresholds, and [`CapabilityCache::verdict`] is
    /// what reads them: below [`CAPABILITY_PROBE_MISS_LIMIT`] every request
    /// asks, from there to [`CAPABILITY_PROBE_REPROBE_AFTER`] requests run on
    /// the verdict while counting toward its renewal, and at the renewal the
    /// request asks again — where an answer files normally and a fresh miss
    /// rewinds to the limit rather than starting the initial streak over.
    misses: u8,
}

#[derive(Debug, Default)]
struct CapabilityCache {
    /// Model id → that model's row. A missing key means no probe has run for
    /// this model since the client was built; a key with `answer: None` means
    /// probes have run and none learned anything, which is the state that
    /// retries — every request, until the streak gives up, then on the renewal
    /// cadence [`CapabilityCache::verdict`] keeps. A `404` endpoint is filed at
    /// once, so it is asked once rather than per request.
    entries: std::collections::HashMap<String, CapabilityEntry>,
}

impl CapabilityCache {
    /// What this model has established so far, or `None` to go and ask.
    ///
    /// A definitive answer is returned for as long as the row holds one, and
    /// that is the only thing that ends the asking. Short of it, the row's
    /// miss count sets the rule: below [`CAPABILITY_PROBE_MISS_LIMIT`] the
    /// request asks, and from there up to [`CAPABILITY_PROBE_REPROBE_AFTER`]
    /// the request runs on [`CachedDisclosure::Undisclosed`] — the inference
    /// that a route refusing three times in a row has nothing to disclose —
    /// while counting the turn it spent. At the renewal boundary the count
    /// stops short of answering for the endpoint and the request asks again.
    ///
    /// The asymmetry with [`Self::insert`] is the point: an endpoint that
    /// *said* "no Models API" is filed and never re-asked, while this
    /// client's own give-up verdict is re-checked for as long as the client
    /// runs. A row is never created here — a model nobody has probed yet has
    /// nothing to age.
    fn verdict(&mut self, model: &str) -> Option<CachedDisclosure> {
        let entry = self.entries.get_mut(model)?;
        if let Some(answer) = entry.answer {
            return Some(answer);
        }
        if entry.misses < CAPABILITY_PROBE_MISS_LIMIT {
            return None;
        }
        if entry.misses >= CAPABILITY_PROBE_REPROBE_AFTER {
            return None;
        }
        entry.misses = entry.misses.saturating_add(1);
        Some(CachedDisclosure::Undisclosed)
    }

    /// File one model's definitive answer, replacing any earlier one.
    ///
    /// Replacement, not first-write-wins: two requests naming one model can
    /// race their probes, and the second answer is no less the endpoint's than
    /// the first. At capacity an entry for some other model makes room — an
    /// arbitrary one, since the map has no age order to lose, and the cost of
    /// a wrong pick is exactly one re-probe.
    fn insert(&mut self, model: &str, answer: CachedDisclosure) {
        let entry = self.entry(model);
        entry.answer = Some(answer);
        entry.misses = 0;
    }

    /// Note that a probe learned nothing: extend the streak, or rewind it when
    /// what failed was the renewal.
    ///
    /// This files no *answer* — a blip is not a disclosure, and the whole
    /// point of [`ProbeOutcome::Inconclusive`] is that this request runs on
    /// the ladder's mapping and the next one tries again. What it accumulates
    /// is the evidence that starts serving [`CachedDisclosure::Undisclosed`] at
    /// [`CAPABILITY_PROBE_MISS_LIMIT`], which is what keeps a failing metadata
    /// route from costing a probe — or its full timeout — ahead of every turn
    /// of a run. A miss at the renewal boundary rewinds to the limit rather
    /// than parking at the ceiling, where every following request would probe
    /// again and the bound would be the very cost this streak exists to
    /// prevent.
    fn note_inconclusive(&mut self, model: &str) {
        let entry = self.entry(model);
        if entry.answer.is_some() {
            // A model with a disclosure is done being probed, and a blip from
            // a losing race — two requests that both found no entry, one of
            // which got an answer filed first — must not stack onto a verdict
            // the endpoint never implied, let alone reach the limit that would
            // replace a real disclosure with "nothing".
            return;
        }
        entry.misses = if entry.misses >= CAPABILITY_PROBE_REPROBE_AFTER {
            CAPABILITY_PROBE_MISS_LIMIT
        } else {
            entry.misses.saturating_add(1)
        };
    }

    /// The row for one model, evicting an arbitrary other row at capacity.
    ///
    /// Both the answer and the streak go when a row is evicted, so an evicted
    /// streak restarts from zero — at worst one more round of
    /// [`CAPABILITY_PROBE_MISS_LIMIT`] probes for a model this client has
    /// taken up again.
    fn entry(&mut self, model: &str) -> &mut CapabilityEntry {
        if !self.entries.contains_key(model)
            && self.entries.len() >= CAPABILITY_CACHE_MAX_MODELS
            && let Some(evicted) = self.entries.keys().next().cloned()
        {
            self.entries.remove(&evicted);
        }
        self.entries.entry(model.to_string()).or_default()
    }
}

/// The thinking shape a request writes for one rung on one model.
///
/// The ladder's own mapping — `thinking: {"type":"enabled"}` around a
/// `budget_tokens` — is what a model with no disclosure gets. A disclosure
/// can replace it with the control that model actually accepts, so the
/// request runs instead of 400ing over a spelling nobody chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingWire {
    /// No effort resolved: the request carries no `thinking` field at all.
    ///
    /// The shape an effort-less client and an effort-less request both get,
    /// and the one the probe never disturbs: [`ThinkingProbe`] is not even
    /// built for such a request, so the wire stays byte-identical to a client
    /// that never had a ladder and no Models API call is made on its behalf.
    Unset,

    /// The off rung in the API's own off spelling:
    /// `thinking: {"type":"disabled"}`.
    ///
    /// Chosen whenever the disclosure did not refuse `disabled` — accepted
    /// `Some(true)` or never stated. The block replaces whatever budget the
    /// ladder wrote, and no `max_tokens` headroom raise accompanies it, since
    /// the off rung budgets nothing. On an endpoint that disclosed nothing,
    /// this is also the spelling a model with no off level answers with its
    /// own error — the honest refusal the ladder states in place of silently
    /// thinking.
    Off,

    /// The off rung on a model that refuses `"disabled"` and accepts
    /// `thinking: {"type":"between_tools"}`.
    ///
    /// The model reasons only while composing between tool calls, which on
    /// the models offering it (Claude Sonnet 5.5 among them) is the lowest
    /// thinking level the API takes — the nearest thing to off such a model
    /// has, so the off rung is served rather than refused. Reached only
    /// through an affirmative disclosure and only from the off rung: a
    /// disclosure merely omitting the member leaves the rung on
    /// [`Self::Off`], and a disclosure refusing both is refused outright
    /// ([`thinking_wire`]).
    BetweenTools,

    /// Extended thinking: `thinking: {"type":"enabled"}` around the rung's
    /// `budget_tokens`, with the `max_tokens` headroom raise that goes with
    /// it.
    ///
    /// The ladder's own mapping, so this is the one shape
    /// [`apply_model_thinking`] does not rewrite — [`build_request_body`]
    /// already wrote the block and raised `max_tokens` to `budget + 1` when
    /// the caller's value could not hold it. Every model that does not
    /// disclose a different control keeps it, which is what an endpoint
    /// behind a gateway or a Messages-compatible server without a Models API
    /// is asked to serve.
    ///
    /// Holds the [`anthropic_thinking_budget`] the block carries, so a caller
    /// can see the depth the rung asked for rather than re-deriving it.
    Budget(u32),

    /// A model that takes thinking depth from `output_config.effort` rather
    /// than a budget: the `thinking` block is removed, the caller's own
    /// `max_tokens` is restored, and the rung's label rides in
    /// `output_config`.
    ///
    /// Chosen on the disclosure's say-so alone — extended thinking refused and
    /// adaptive thinking accepted, the shape of Anthropic's adaptive tier.
    /// Restoring `max_tokens` is load-bearing, not cosmetic: the raise was
    /// made for a budget this model cannot take, and leaving it would quietly
    /// spend output tokens the caller never configured.
    ///
    /// Holds the label verbatim as [`effort_label`] spells it, refusals
    /// included: `"minimal"` rides to the API and comes back as its error,
    /// because a ladder that clamped the rung to the nearest accepted level
    /// would hide the mismatch it exists to surface.
    Effort(&'static str),
}

/// Choose the wire shape for one rung against one model's disclosure.
///
/// [`None`] is returned for the one case worth refusing rather than sending:
/// a disclosure that says the model accepts neither off spelling. Every
/// alternative is worse than the error — `disabled` would 400 unexplainably,
/// omitting the block means adaptive thinking on a model that thinks by
/// default, and quietly degrading to `between_tools` is the substitution the
/// ladder exists to prevent.
fn thinking_wire(
    effort: crate::structured::ThinkingEffort,
    caps: ThinkingCapabilities,
) -> Option<ThinkingWire> {
    if effort == crate::structured::ThinkingEffort::None {
        return match (caps.disabled, caps.between_tools) {
            (Some(false), Some(true)) => Some(ThinkingWire::BetweenTools),
            (Some(false), _) => None,
            _ => Some(ThinkingWire::Off),
        };
    }
    match (caps.enabled, caps.adaptive) {
        (Some(false), Some(true)) => Some(ThinkingWire::Effort(effort_label(effort))),
        _ => Some(ThinkingWire::Budget(
            anthropic_thinking_budget(effort).unwrap_or_default(),
        )),
    }
}

/// The `output_config.effort` label for a rung: the rung's own name, verbatim.
///
/// Anthropic's effort ladder has no token-check level, so `Minimal` sends
/// `"minimal"` and the API rejects it with its own error — the rule
/// [`ThinkingEffort`](crate::structured::ThinkingEffort) already states for
/// an OpenAI endpoint that tops out below `"xhigh"`: the provider's refusal
/// beats the ladder silently clamping the rung.
fn effort_label(effort: crate::structured::ThinkingEffort) -> &'static str {
    use crate::structured::ThinkingEffort;
    match effort {
        ThinkingEffort::Minimal => "minimal",
        ThinkingEffort::Low | ThinkingEffort::None => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::Xhigh => "xhigh",
        ThinkingEffort::Max => "max",
    }
}

/// Rewrite a built request body onto the model's supported thinking shape.
///
/// The body arrives carrying the ladder's own mapping; this replaces it only
/// where the model disclosed a different control, and restores the caller's
/// configured `max_tokens` when the headroom raise was made for a budget the
/// model does not take. [`ThinkingWire::Unset`] and
/// [`ThinkingWire::Budget`] leave the body untouched, which is what an
/// effort-less request and an undisclosed endpoint both get: the wire stays
/// byte-identical to the ladder's.
fn apply_model_thinking(body: &mut Value, wire: ThinkingWire, configured_max_tokens: u32) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    match wire {
        ThinkingWire::Unset | ThinkingWire::Budget(_) => {}
        ThinkingWire::Off => {
            obj.insert(
                "thinking".to_string(),
                serde_json::json!({ "type": "disabled" }),
            );
        }
        ThinkingWire::BetweenTools => {
            obj.insert(
                "thinking".to_string(),
                serde_json::json!({ "type": "between_tools" }),
            );
        }
        ThinkingWire::Effort(label) => {
            obj.remove("thinking");
            obj.insert(
                "max_tokens".to_string(),
                serde_json::json!(configured_max_tokens),
            );
            obj.insert(
                "output_config".to_string(),
                serde_json::json!({ "effort": label }),
            );
        }
    }
}

/// The Models API probe that resolves what one model accepts for thinking.
///
/// Owns everything a probe needs so both request paths can run it from
/// inside their `'static` async blocks: the client's pooled HTTP handle, its
/// credential, the model's metadata URL, and the cache shared with the
/// client. Built only for a request that resolved a thinking effort, so an
/// effort-less client pays neither the probe nor the clones.
struct ThinkingProbe {
    /// The client's pooled HTTP handle, shared with completion requests.
    ///
    /// Cloned, not rebuilt: a `reqwest::Client` clone shares the connection
    /// pool, so a probe costs a request on a connection the client already
    /// holds rather than a fresh TLS handshake ahead of the turn it serves.
    /// Only the request's own timeout differs —
    /// [`CAPABILITIES_PROBE_TIMEOUT`], never the conversational read timeout.
    http: reqwest::Client,

    /// The credential sent as `x-api-key`, exactly as on a Messages request.
    ///
    /// The same key, because a disclosure is worth only as much as the
    /// account it describes: a model's accepted thinking controls are read
    /// from the endpoint this credential serves, not from an anonymous
    /// metadata route. Owned rather than borrowed because the streaming path
    /// runs the probe inside a `'static` stream.
    api_key: String,

    /// The `{base_url}/v1/models/{model}` URL the disclosure is read from.
    ///
    /// Built per request off the *same* base URL the Messages endpoint uses,
    /// so a deployment that fronts one fronts the other and a `404` here
    /// means what it says: no Models API behind this base URL. It cannot be
    /// cached on the client, because a request can name a model the client
    /// does not run.
    url: String,

    /// The per-model cache, shared with the owning client.
    ///
    /// Shared because the disclosure belongs to a model rather than to a
    /// request: the second turn of a run should not pay for the first turn's
    /// probe. Behind an [`std::sync::Arc`] because the streaming path reaches
    /// it from a `'static` stream, and behind a lock because a probe resolves
    /// asynchronously while later requests are being built.
    /// Entries are keyed by model, so
    /// [`set_model`](ApiClient::set_model) neither invalidates nor evicts
    /// anything: a swapped-in model finds no entry and asks for its own.
    cache: std::sync::Arc<std::sync::Mutex<CapabilityCache>>,

    /// The model this probe addresses.
    ///
    /// Matched against the [`CapabilityCache`] entry's key so a cached
    /// disclosure never crosses models, and named in the refusal
    /// [`resolve`] raises when the model offers no way to turn thinking off —
    /// a host running a fallback chain has to learn which model said no.
    model: String,

    /// The caller's configured `max_tokens`, before any headroom raise.
    ///
    /// Read back out of the request only for [`ThinkingWire::Effort`], whose
    /// model takes no budget and so has no claim on the raise
    /// [`build_request_body`] made to hold one. Carried here rather than
    /// re-derived from the body, which by then carries the raised figure and
    /// cannot say which of its two values the caller chose.
    configured_max_tokens: u32,
}

impl ThinkingProbe {
    /// Resolve one rung against the model's disclosed controls.
    ///
    /// One call per request: the disclosure comes from the shared cache, so
    /// the first request for a model pays the probe and the rest read the
    /// answer.
    ///
    /// An unset effort resolves without probing: the request writes no
    /// thinking field, so there is nothing a disclosure could change.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::config_validation`] naming the model when the
    /// disclosure says thinking cannot be turned off here and the off rung
    /// was asked for; the refusal is raised before the request is sent.
    async fn resolve(
        &self,
        effort: Option<crate::structured::ThinkingEffort>,
    ) -> Result<ThinkingWire, ApiError> {
        let Some(effort) = effort else {
            return Ok(ThinkingWire::Unset);
        };
        thinking_wire(effort, self.capabilities().await)
            .ok_or_else(|| thinking_unavailable_error(&self.model))
    }

    /// Read the cached disclosure for this probe's model, asking once.
    ///
    /// A cached answer is taken under the lock and the lock is dropped before
    /// anything else happens; the probe then files its verdict under the model
    /// it asked about, which is not necessarily the client's current one.
    /// Concurrent probes of one model may each fetch — the second filing
    /// replaces the first, and both requests carried a real answer.
    async fn capabilities(&self) -> ThinkingCapabilities {
        let cached = crate::error::recover_guard(self.cache.lock()).verdict(&self.model);
        if let Some(answer) = cached {
            return Self::apply(answer);
        }
        match self.fetch().await {
            ProbeOutcome::Answered(answer) => {
                crate::error::recover_guard(self.cache.lock()).insert(&self.model, answer);
                Self::apply(answer)
            }
            ProbeOutcome::Inconclusive => {
                crate::error::recover_guard(self.cache.lock()).note_inconclusive(&self.model);
                ThinkingCapabilities::default()
            }
        }
    }

    /// The disclosure a request acts on for one cached verdict.
    ///
    /// An endpoint that answered "nothing", a route this client gave up on, and
    /// one still being asked take the same wire shape — the ladder's own
    /// mapping. They differ only in how soon anyone asks again: never, on the
    /// renewal cadence, and next request, respectively.
    fn apply(answer: CachedDisclosure) -> ThinkingCapabilities {
        match answer {
            CachedDisclosure::Known(caps) => caps,
            CachedDisclosure::Undisclosed => ThinkingCapabilities::default(),
        }
    }

    /// Ask the Models API what this model disclosed about thinking.
    ///
    /// The answer separates what an endpoint *said* from what it merely
    /// failed to deliver. Definitive and cacheable: a success (whether or not
    /// it discloses thinking members), a non-JSON success (a gateway in front
    /// of the API), and a `404` or `405`, which says there is no Models API at
    /// this base URL.
    /// Inconclusive and not filed as an answer (though a streak of them ends
    /// the asking): a transport failure or
    /// timeout, `401`/`403` (a rotated credential says nothing about the
    /// model), `429` and any `5xx` (busy, not silent), and a body that could
    /// not be read within [`MAX_RESPONSE_BODY`].
    ///
    /// The distinction is the difference between a soft fallback and a stuck
    /// one: a filed answer is never revisited, so a probe that filed a blip
    /// would send the ladder's `budget_tokens` to an adaptive-tier model on
    /// every later turn, and each of those turns would 400 on the spelling the
    /// probe never read. The request carries [`CAPABILITIES_PROBE_TIMEOUT`]
    /// rather than the conversational read timeout either way, so a wedged
    /// metadata route cannot stall a run's first turn.
    ///
    /// [`MAX_RESPONSE_BODY`]: super::MAX_RESPONSE_BODY
    async fn fetch(&self) -> ProbeOutcome {
        let Ok(response) = self
            .http
            .get(&self.url)
            .header("x-api-key", &self.api_key)
            .header(
                reqwest::header::HeaderName::from_static("anthropic-version"),
                reqwest::header::HeaderValue::from_static(ANTHROPIC_VERSION),
            )
            .timeout(CAPABILITIES_PROBE_TIMEOUT)
            .send()
            .await
        else {
            return ProbeOutcome::Inconclusive;
        };
        let status = response.status();
        if matches!(
            status,
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
        ) {
            return ProbeOutcome::Answered(CachedDisclosure::Undisclosed);
        }
        if !status.is_success() {
            return ProbeOutcome::Inconclusive;
        }
        let Ok(bytes) = super::read_bounded_body(response).await else {
            return ProbeOutcome::Inconclusive;
        };
        let Ok(model) = serde_json::from_slice::<Value>(&bytes) else {
            return ProbeOutcome::Answered(CachedDisclosure::Undisclosed);
        };
        let types = model.pointer("/capabilities/thinking/types");
        let disclosed = |name: &str| {
            types
                .and_then(|types| types.get(name))
                .and_then(|entry| entry.get("supported"))
                .and_then(Value::as_bool)
        };
        let caps = ThinkingCapabilities::from_states(
            disclosed("disabled"),
            disclosed("between_tools"),
            disclosed("enabled"),
            disclosed("adaptive"),
        );
        ProbeOutcome::Answered(if caps.is_empty() {
            CachedDisclosure::Undisclosed
        } else {
            CachedDisclosure::Known(caps)
        })
    }
}

/// The error for a `strict` request the Anthropic Messages API cannot
/// express.
///
/// It has no strict-mode switch for its forced tool, so the request
/// would be silently served non-strict — rejected loudly instead.
fn strict_unsupported_error(provider: &str) -> ApiError {
    ApiError::config_validation(format!(
        "the {provider} API cannot express response_format.strict; the \
         request would be served non-strict — use an OpenAI-compatible \
         endpoint or drop strict"
    ))
}

/// Build the JSON request body for the Anthropic Messages API.
///
/// Each [`Message`] is serialized via [`convert_message`], then assembled
/// with the model, `max_tokens`, system prompt, and optional tools. A set
/// `effort` emits the `thinking` block — the level's `budget_tokens`, or
/// `{"type": "disabled"}` on the off rung — and a budget raises
/// `max_tokens` for headroom (see [`anthropic_thinking_budget`]).
///
/// Tool-call constraint:
/// - When `response_format` is set, synthesizes a single forced tool
///   (`tool_choice` forced); `tool_constraint` is ignored in that case.
/// - Otherwise, when `tool_constraint` is `Strict`, each tool's
///   `input_schema` is tightened (`additionalProperties: false` and full
///   `required`) via [`convert_tools`] before submission. No `tool_choice`
///   is emitted — `Strict` constrains the call's shape, not its selection.
/// - A `response_format` with `strict: true` never reaches this builder:
///   the `*_with_options` entry points reject it up front with
///   `ApiError::config_validation`.
fn build_request_body(
    spec: &RequestBodySpec<'_>,
    stream: bool,
    max_tokens: u32,
    effort: Option<crate::structured::ThinkingEffort>,
    replay_reasoning: bool,
) -> Value {
    let budget = effort.and_then(anthropic_thinking_budget);
    let max_tokens = match budget {
        Some(budget) => max_tokens.max(budget.saturating_add(1)),
        None => max_tokens,
    };
    let thinking = match (effort, budget) {
        (Some(crate::structured::ThinkingEffort::None), _) => {
            Some(serde_json::json!({ "type": "disabled" }))
        }
        (Some(_), Some(budget)) => Some(serde_json::json!({
            "type": "enabled",
            "budget_tokens": budget,
        })),
        _ => None,
    };
    let RequestBodySpec {
        model,
        messages,
        system,
        tools,
        response_format,
        tool_constraint,
    } = spec;

    let tools = tools.filter(|t| !t.is_empty());
    let (non_system, effective_system) = super::fold_system_messages(messages, *system);
    let owned: Vec<crate::message::Message> = non_system.into_iter().cloned().collect();
    let non_system = crate::message::user_head_before_tool_use_opening(
        crate::message::coalesce_adjacent_assistants(owned),
    );
    let msgs: Vec<Value> = non_system
        .iter()
        .map(|m| convert_message(m, replay_reasoning, effort))
        .collect();
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

    if let Some(thinking) = thinking
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert("thinking".to_string(), thinking);
    }

    body
}

/// Whether a content block is an extended-thinking block.
///
/// Anthropic rejects `cache_control` on `thinking` and
/// `redacted_thinking` blocks — they are server-managed reasoning
/// artifacts, not cacheable content — so the breakpoint placement
/// skips them when selecting the trailing blocks to mark.
fn is_thinking_block(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "thinking" || kind == "redacted_thinking")
}

/// Mark one content block as a cache breakpoint.
///
/// Inserts `cache_control: {type: "ephemeral"}` on the block object;
/// non-object blocks are left alone rather than reshaped.
fn mark_cache_breakpoint(block: &mut Value) {
    if let Some(obj) = block.as_object_mut() {
        obj.insert(
            "cache_control".to_string(),
            serde_json::json!({ "type": "ephemeral" }),
        );
    }
}

/// Place prompt-cache breakpoints on a built Messages request body.
///
/// Mutates the body in place: the last tool definition, the system
/// prompt (rewritten from its plain-string form into a single marked
/// text block), and the last two **eligible** content blocks of the
/// final message each get `cache_control` — thinking and
/// redacted-thinking blocks are skipped (the provider rejects markers
/// on them) but never reordered, so a signed-thinking tail still
/// marks the two text blocks around it. That placement follows the
/// provider's guidance for agentic loops — the tools and system are
/// stable for the session's life, and the trailing pair of message
/// blocks rolls the cached prefix forward one turn at a time, so each
/// request shares the longest prefix with its predecessor. At most
/// four breakpoints are placed, matching the provider's per-request
/// cap. A final message whose content is a plain string is wrapped
/// into a one-block array so it can carry the marker; an empty
/// system string stays a string, since marking nothing-at-all buys
/// no cache.
fn apply_prompt_cache_breakpoints(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    if let Some(Value::Array(tools)) = obj.get_mut("tools")
        && let Some(last) = tools.last_mut()
    {
        mark_cache_breakpoint(last);
    }

    let system_text = obj
        .get("system")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(text) = system_text.filter(|t| !t.is_empty()) {
        obj.insert(
            "system".to_string(),
            serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": { "type": "ephemeral" },
            }]),
        );
    }

    if let Some(Value::Array(messages)) = obj.get_mut("messages")
        && let Some(last_message) = messages.last_mut().and_then(Value::as_object_mut)
        && let Some(content) = last_message.get_mut("content")
    {
        match content {
            Value::String(text) => {
                let text = text.clone();
                *content = serde_json::json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" },
                }]);
            }
            Value::Array(blocks) => {
                let mut marked: usize = 0;
                for block in blocks.iter_mut().rev() {
                    if marked >= 2 {
                        break;
                    }
                    if is_thinking_block(block) {
                        continue;
                    }
                    mark_cache_breakpoint(block);
                    marked = marked.saturating_add(1);
                }
            }
            _ => {}
        }
    }
}

/// The Messages API `thinking.budget_tokens` for one effort level.
///
/// The fixed ladder from [`ThinkingEffort`]'s table: `Minimal` budgets
/// `1024` tokens (the API's floor for an enabled block), `Low` `2048`,
/// `Medium` `8192`, `High` `16384`, `Xhigh` `24576`, and `Max` `32768`.
/// The `None` level budgets nothing — `None` back. The off rung is
/// stated to the API as `thinking: {"type": "disabled"}` rather than by
/// omission, because omitting the block means adaptive thinking on every
/// model that thinks by default. Provider-native numbers, never
/// invented per model.
///
/// The spelling is extended thinking — `thinking: {"type": "enabled"}`
/// around the budget — which is what the Messages API's
/// extended-thinking models accept. Anthropic's adaptive-tier models
/// (Opus 4.7 and later, Sonnet 5 and later, Haiku 5.5, Fable and Mythos)
/// refuse `budget_tokens` and take thinking depth from
/// `output_config.effort` instead, whose own ladder has no token-check
/// rung at all. Which tier a model sits in is deliberately not this
/// function's business — a copy of Anthropic's catalogue would age into a
/// wrong answer about someone else's models — so the tier is read from the
/// endpoint instead: a request that resolves an effort consults the Models
/// API once per model (see [`ThinkingProbe`]) and writes the control that
/// model disclosed, while an endpoint that discloses nothing keeps the
/// mapping below.
fn anthropic_thinking_budget(effort: crate::structured::ThinkingEffort) -> Option<u32> {
    match effort {
        crate::structured::ThinkingEffort::None => None,
        crate::structured::ThinkingEffort::Minimal => Some(1024),
        crate::structured::ThinkingEffort::Low => Some(2048),
        crate::structured::ThinkingEffort::Medium => Some(8192),
        crate::structured::ThinkingEffort::High => Some(16384),
        crate::structured::ThinkingEffort::Xhigh => Some(24576),
        crate::structured::ThinkingEffort::Max => Some(32768),
    }
}

/// The wire form of one thinking part replayed natively.
///
/// Extended thinking signs each reasoning block, and the Messages API
/// demands the original block — signature or redacted payload intact —
/// ahead of `tool_use` on continuation requests, so a signed or
/// redacted part is returned exactly as it arrived rather than
/// rendered as plain text.
fn native_thinking_block(text: &str, signature: Option<&str>, redacted: Option<&str>) -> Value {
    if let Some(data) = redacted {
        serde_json::json!({ "type": "redacted_thinking", "data": data })
    } else {
        serde_json::json!({
            "type": "thinking",
            "thinking": text,
            "signature": signature.unwrap_or_default(),
        })
    }
}

/// Convert a single framework [`Message`] into the Anthropic JSON shape.
///
/// - Messages with only a single text part use a plain string for `content`
///   (Anthropic's recommended optimization).
/// - Messages with tool calls or tool results use the full `content` array.
/// - Thinking parts render as one capped `<thinking>` text block
///   each, in part order, ahead of the other content when
///   `replay_reasoning` is set or the effective effort is the
///   `None` rung (thinking off, where native blocks are
///   wire-illegal) — except that a signed or redacted part rides
///   back natively while the effective effort budgets a thinking
///   block, because Anthropic requires the original block ahead of
///   tool use on continuation requests; image parts are skipped,
///   as this wire carries no images in conversation history.
pub(super) fn convert_message(
    m: &Message,
    replay_reasoning: bool,
    effort: Option<crate::structured::ThinkingEffort>,
) -> Value {
    let thinking_on =
        effort.is_some_and(|effort| effort != crate::structured::ThinkingEffort::None);
    let reasoning_as_text =
        replay_reasoning || matches!(effort, Some(crate::structured::ThinkingEffort::None));
    let role = match m.role {
        Role::User | Role::System => "user",
        Role::Assistant => "assistant",
    };

    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();

    for p in &m.parts {
        match p {
            MessagePart::Text { text } => text_parts.push(text.as_str()),
            MessagePart::Thinking {
                text,
                signature,
                redacted,
            } if thinking_on && (signature.is_some() || redacted.is_some()) => {
                thinking_blocks.push(native_thinking_block(
                    text,
                    signature.as_deref(),
                    redacted.as_deref(),
                ));
            }
            MessagePart::Thinking { text, .. } if reasoning_as_text && !text.is_empty() => {
                let tail = super::replay_reasoning_tail(text);
                thinking_blocks.push(serde_json::json!({
                    "type": "text",
                    "text": format!("<thinking>\n{tail}\n</thinking>"),
                }));
            }
            MessagePart::ToolCall { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                }));
            }
            MessagePart::ToolResult {
                call_id,
                output,
                is_error,
                ..
            } => {
                let mut block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output.to_string(),
                });
                if matches!(is_error, Some(true))
                    && let Some(obj) = block.as_object_mut()
                {
                    obj.insert("is_error".to_string(), Value::Bool(true));
                }
                tool_results.push(block);
            }
            MessagePart::Thinking { .. } | MessagePart::Image { .. } => {}
        }
    }

    let has_tool_content = !(tool_calls.is_empty() && tool_results.is_empty());

    if !has_tool_content && text_parts.len() == 1 && !thinking_blocks.is_empty() {
        let text = text_parts.first().copied().unwrap_or_default();
        let mut blocks = thinking_blocks;
        blocks.push(serde_json::json!({ "type": "text", "text": text }));
        serde_json::json!({ "role": role, "content": blocks })
    } else if !has_tool_content && text_parts.len() == 1 {
        // Single text — Anthropic allows plain string content.
        let text = text_parts.first().copied().unwrap_or_default();
        serde_json::json!({ "role": role, "content": text })
    } else if !has_tool_content {
        // Multiple text parts — array of text blocks.
        let mut blocks = thinking_blocks;
        blocks.extend(
            text_parts
                .iter()
                .map(|t| serde_json::json!({ "type": "text", "text": t })),
        );
        serde_json::json!({ "role": role, "content": blocks })
    } else {
        // Mixed content — combine text + tool blocks in a single array.
        let mut blocks = thinking_blocks;
        if !text_parts.is_empty() {
            let text = text_parts.join("");
            blocks.push(serde_json::json!({ "type": "text", "text": text }));
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
pub(super) fn convert_tools(tools: &[ToolSchema], strict: bool) -> Vec<Value> {
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

use super::sse::SseReader;

impl SseReader {
    /// Extract the next SSE event as `(event_type, data_json)`.
    ///
    /// Returns `Ok(None)` at end-of-stream. A `data:` payload that is
    /// empty or unparseable JSON yields `None` for the `Value` half —
    /// the event type still surfaces, and the emitter's handlers
    /// decide what an absent payload means.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the underlying HTTP stream fails, or if
    /// a completed line is not valid UTF-8 (malformed provider data).
    async fn next_event(&mut self) -> Result<Option<(String, Option<Value>)>, ApiError> {
        let mut event_type = String::new();
        let mut data = String::new();
        let mut have_event = false;

        loop {
            while let Some(line) = self.take_line()? {
                if line.is_empty() {
                    if have_event {
                        let parsed = Self::parse_event_data(&data, &event_type);
                        return Ok(Some((event_type, parsed)));
                    }
                    continue;
                }

                if let Some(ev) = super::sse_event_type(&line) {
                    event_type = ev.into();
                    have_event = true;
                } else if let Some(d) = super::sse_data_payload(&line) {
                    if data.is_empty() {
                        data = d.into();
                    } else {
                        data.push('\n');
                        data.push_str(d);
                    }
                    have_event = true;
                }
            }

            if self.next_chunk().await?.is_none() {
                if have_event {
                    let parsed = Self::parse_event_data(&data, &event_type);
                    return Ok(Some((event_type, parsed)));
                }
                return Ok(None);
            }
        }
    }

    /// Parse the accumulated `data` string into JSON, logging on failure.
    ///
    /// Returns `None` for empty payloads and unparseable JSON, `Some`
    /// otherwise; a parse failure logs the event type and payload size
    /// at `warn`. What an absent payload means — ignored, defaulted, or
    /// fatal — is decided by the handlers that consume
    /// [`next_event`](Self::next_event)'s `Option<Value>`, not here.
    fn parse_event_data(data: &str, event_type: &str) -> Option<Value> {
        if data.is_empty() {
            return None;
        }
        match serde_json::from_str(data) {
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
    }
}

/// Stateful translator that converts Anthropic SSE events into
/// [`StreamEvent`]s.
///
/// Encapsulates all protocol-level bookkeeping:
/// - Emitting [`MessageStart`] once.
/// - Emitting [`PartStart`] / [`IndexedDelta`] for text and tool-call content.
/// - Emitting [`PartStop`] when parts finish.
/// - Emitting the final [`MessageDelta`] with stop reason and usage.
#[derive(Default)]
pub(super) struct StreamEmitter {
    /// Whether [`MessageStart`] has been emitted for the current stream.
    ///
    /// The Anthropic stream opens with one `message_start` event; the
    /// emitter forwards it once as a [`StreamEvent::MessageStart`] and
    /// ignores any subsequent duplicates.
    started: bool,

    /// Index of the text content block currently open, if any.
    ///
    /// Anthropic signals the start of a text block with
    /// `content_block_start` (`type: "text"`) and its end with
    /// `content_block_stop`. This holds the server-supplied block index
    /// while the block is open and `None` when no text block is open, so
    /// the matching `content_block_stop` emits exactly one
    /// [`StreamEvent::PartStop`] and `text_delta` fragments route to the
    /// correct part. Mirrors [`current_tool_index`](Self::current_tool_index)
    /// and [`thinking_index`](Self::thinking_index) for the text lane.
    text_index: Option<usize>,

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
    /// Set by `message_stop` — the one and only writer — and read by
    /// [`process_event`](Self::process_event), which ignores every
    /// event once it is set, and by
    /// [`on_message_delta`](Self::on_message_delta), which no-ops when
    /// `message_stop` arrived first (an out-of-order stream).
    /// [`finish`](Self::finish) does not consult it: it only surfaces
    /// a recorded `error` event and drains pending output.
    finished: bool,

    /// A terminal error delivered as an `event: error` SSE event, if any.
    ///
    /// Anthropic reports mid-stream failures (an `overloaded_error` after
    /// generation began, for example) as error events rather than closing
    /// the HTTP stream with a status. Recorded here so
    /// [`finish`](Self::finish) terminates the stream with the error
    /// instead of completing it — without it, a consumer could not
    /// distinguish an errored stream from a truncated one. First error
    /// wins; later ones are ignored.
    error: Option<ApiError>,

    /// Token usage latched from the `message_start` event.
    ///
    /// Anthropic reports the prompt's real token counts in `message_start`'s
    /// `usage` object; the terminal `message_delta` revises only the output
    /// side (and on server-tool turns may revise both upward). The latch
    /// lets [`on_message_delta`](Self::on_message_delta) report the real
    /// input count on the terminal event instead of zero. Stays zeroed when
    /// the event carries no `usage` object.
    start_usage: Usage,

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
    /// `content_block_stop`, `message_delta`, `message_stop`, `error`);
    /// `data` is the parsed JSON payload of the paired `data:` line, or
    /// `None` when the payload was absent or failed to parse. Unknown event
    /// types are ignored. Any events produced are appended to the pending
    /// queue; an `error` event records the terminal error that
    /// [`finish`](Self::finish) surfaces.
    ///
    /// Also the entry point for the Bedrock Anthropic path, whose
    /// event-stream frame payloads are the same event objects.
    ///
    /// Once [`message_stop`](Self::on_message_stop) has been processed,
    /// every further event is ignored: `message_stop` terminates the
    /// message, and a desynced stream must not append parts to a
    /// message the consumer has already been told is complete.
    pub(super) fn process_event(&mut self, event_type: &str, data: Option<Value>) {
        if self.finished {
            return;
        }
        match event_type {
            "message_start" => self.on_message_start(data.as_ref()),
            "content_block_start" => self.on_block_start(data),
            "content_block_delta" => self.on_block_delta(data),
            "content_block_stop" => self.on_block_stop(data),
            "message_delta" => self.on_message_delta(data),
            "message_stop" => self.on_message_stop(),
            "error" => self.on_error(data.as_ref()),
            _ => {}
        }
    }

    /// Handle a `message_start` event.
    ///
    /// On the first call, reads the message `id` and `model` from
    /// `/message/id` and `/message/model` and emits a
    /// [`StreamEvent::MessageStart`], and latches `/message/usage` into
    /// [`start_usage`](Self::start_usage) — the only place Anthropic reports
    /// input tokens on the streaming path. No-ops on subsequent calls (the
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
        self.start_usage = data
            .and_then(|v| v.pointer("/message/usage"))
            .and_then(extract_usage_object)
            .unwrap_or_default();

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
    /// index. A `thinking` or `redacted_thinking` block opens the
    /// thinking lane with a `PartStart`, and a redacted block's
    /// `/content_block/data` payload rides the lane's first delta. Other
    /// block types are ignored.
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
                self.text_index = Some(index);
                self.push(StreamEvent::PartStart(PartStart {
                    index,
                    part: Some(MessagePart::text("")),
                }));
            }
            Some("thinking" | "redacted_thinking") => {
                self.thinking_part_open = true;
                self.thinking_index = Some(index);
                self.push(StreamEvent::PartStart(PartStart { index, part: None }));

                if matches!(block_type, Some("redacted_thinking")) {
                    let data = v
                        .pointer("/content_block/data")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index,
                        delta: DeltaPart::Thinking {
                            text: String::new(),
                            signature: None,
                            redacted: (!data.is_empty()).then_some(data),
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
    /// skipped. A `thinking_delta` emits a
    /// [`DeltaPart::Thinking`](crate::stream::DeltaPart::Thinking) text
    /// fragment on the thinking lane, a `signature_delta` emits a
    /// thinking delta carrying only the `/delta/signature`, and a
    /// redacted block's payload arrives via its block start. Other delta
    /// types are ignored.
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
                    let text_index = self.text_index.unwrap_or(0);
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: text_index,
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
                    let tool_index = self.current_tool_index.unwrap_or(0);
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
                    let thinking_index = self.thinking_index.unwrap_or(0);
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: thinking_index,
                        delta: DeltaPart::Thinking {
                            text,
                            signature: None,
                            redacted: None,
                        },
                    }));
                }
            }
            Some("signature_delta") => {
                let signature = v
                    .pointer("/delta/signature")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !signature.is_empty() {
                    let thinking_index = self.thinking_index.unwrap_or(0);
                    self.push(StreamEvent::IndexedDelta(IndexedDelta {
                        index: thinking_index,
                        delta: DeltaPart::Thinking {
                            text: String::new(),
                            signature: Some(signature),
                            redacted: None,
                        },
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
        if let Some(index) = self.text_index.take() {
            self.push(StreamEvent::PartStop { index: Some(index) });
        } else if self.thinking_part_open {
            self.thinking_part_open = false;
            let index = self.thinking_index.take();
            self.push(StreamEvent::PartStop { index });
        } else if self.tool_parts_open > 0 {
            self.tool_parts_open = self.tool_parts_open.saturating_sub(1);
            let index = self.current_tool_index.take();
            self.push(StreamEvent::PartStop { index });
        }
    }

    /// Handle a `message_delta` event carrying the terminal stop reason
    /// and usage.
    ///
    /// Reads `/delta/stop_reason` (mapped via
    /// [`StreamStopReason::from_api_str`], defaulting to `EndTurn` on an
    /// unrecognized value) and `/usage/input_tokens` +
    /// `/usage/output_tokens`, each merged with the
    /// [`start_usage`](Self::start_usage) latch by taking the larger of the
    /// two counts — `/usage/cache_read_input_tokens` and
    /// `/usage/cache_creation_input_tokens` merge the same way into the
    /// cached share and the cache-write spend. The max matters because
    /// the two events split the reporting: `message_start` carries the
    /// input count the terminal delta omits, and a server-tools turn's
    /// delta may revise either count upward from its start value. A
    /// delta that revises a cache share upward also folds that
    /// revision into the input total, so the merged input keeps the
    /// family's documented identity — raw + merged read + merged write
    /// ([`Usage::cache_write_tokens`]). A delta that revises the raw
    /// input while omitting the cache fields carries the latched shares
    /// on top of that raw, holding the identity across that shape too.
    /// The merge is max-by-assumption,
    /// not max-by-guarantee: a genuine
    /// downward revision by the terminal delta (cache accounting
    /// adjustments) loses to the latched start value — the cost of never
    /// under-reporting a count the earlier event already claimed. The
    /// usage event is only attached when at least one merged count is
    /// non-zero. Emits a single [`StreamEvent::MessageDelta`] with both,
    /// so streaming and non-streaming turns report identical usage.
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
        let delta_in = v
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        let delta_out = v
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        let delta_cached = v
            .pointer("/usage/cache_read_input_tokens")
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        let delta_creation = v
            .pointer("/usage/cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        let delta_input_total = delta_in
            .saturating_add(delta_cached)
            .saturating_add(delta_creation);
        let cached_tok = delta_cached.max(self.start_usage.cached_input_tokens);
        let write_tok = delta_creation.max(self.start_usage.cache_write_tokens);
        let share_revision = cached_tok
            .saturating_sub(self.start_usage.cached_input_tokens)
            .saturating_add(write_tok.saturating_sub(self.start_usage.cache_write_tokens));
        let latched_shares_on_delta = delta_in
            .saturating_add(cached_tok)
            .saturating_add(write_tok);
        let in_tok = delta_input_total
            .max(self.start_usage.input_tokens.saturating_add(share_revision))
            .max(latched_shares_on_delta);
        let out_tok = delta_out.max(self.start_usage.output_tokens);

        let usage = if in_tok > 0 || out_tok > 0 {
            Some(
                Usage::new(in_tok, out_tok)
                    .with_cached_input(cached_tok)
                    .with_cache_write(write_tok),
            )
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
    /// Setting `finished` here marks the stream complete so later
    /// handlers (a duplicate `message_stop`, an out-of-order
    /// `message_delta`) are no-ops. [`finish`](Self::finish) appends
    /// nothing — this event is the one and only source of the terminal
    /// [`StreamEvent::MessageStop`].
    fn on_message_stop(&mut self) {
        self.finished = true;
        if self.thinking_part_open {
            self.push(StreamEvent::PartStop {
                index: self.thinking_index,
            });
        }
        if self.text_index.is_some() {
            self.push(StreamEvent::PartStop {
                index: self.text_index,
            });
        }
        for _ in 0..self.tool_parts_open {
            self.push(StreamEvent::PartStop { index: None });
        }
        self.thinking_part_open = false;
        self.thinking_index = None;
        self.tool_parts_open = 0;
        self.text_index = None;
        self.push(StreamEvent::MessageStop);
    }

    /// Handle a terminal `error` event.
    ///
    /// Reads `/error/type` and `/error/message` (Anthropic's error payload,
    /// e.g. `overloaded_error` / `"Overloaded"`). Rate-limit family failures
    /// (`rate_limit_error`, the 429 SSE form, and `overloaded_error`, the
    /// 529 SSE form) become [`ApiError::RateLimit`] so the stream handler's
    /// rate-limit ladder — with its backoff and `Retry-After` handling —
    /// owns the response, mirroring the status-based classification of
    /// non-streaming requests. Everything else stays an
    /// [`ApiError::api`] recording. In both cases [`finish`](Self::finish)
    /// surfaces the recording as the stream's terminal error; missing
    /// fields fall back to generic wording so a malformed payload still
    /// terminates with an error rather than being dropped. The first error
    /// wins; later ones are ignored.
    fn on_error(&mut self, data: Option<&Value>) {
        if self.error.is_some() {
            return;
        }
        let (kind, message) = match data {
            Some(v) => (
                v.pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("error")
                    .to_string(),
                v.pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("stream failed")
                    .to_string(),
            ),
            None => ("error".to_string(), "stream failed".to_string()),
        };
        let detail = format!("{kind}: {message}");
        self.error = Some(
            if matches!(kind.as_str(), "rate_limit_error" | "overloaded_error") {
                ApiError::RateLimit {
                    retry_after: None,
                    message: detail,
                    meta: None,
                }
            } else {
                ApiError::api(detail)
            },
        );
    }

    /// Finalize the stream and return any remaining events.
    ///
    /// When an `event: error` was recorded, terminates with that error and
    /// emits nothing further — no synthetic [`StreamEvent::MessageStop`], so
    /// the failure cannot masquerade as a clean stop. Otherwise drains
    /// the pending queue — nothing is appended: `MessageStop` is emitted
    /// exactly once, by the `message_stop` event itself. A stream that
    /// ends without that event is truncated, and the *absence* of the
    /// stop is what tells the handler so; synthesizing one here would
    /// dress a cut connection up as a completed turn.
    ///
    /// # Errors
    ///
    /// Returns the recorded terminal error, if an `event: error` arrived
    /// during the stream.
    pub(super) fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        Ok(self.drain())
    }

    /// Whether a terminal `event: error` has been recorded.
    ///
    /// The stream loop checks this after each SSE event to stop reading —
    /// Anthropic may hold the errored connection open, and waiting for its
    /// EOF would delay the error the emitter already knows about.
    fn error_recorded(&self) -> bool {
        self.error.is_some()
    }

    /// Drain all pending events.
    ///
    /// Takes the queued events produced since the last drain — the
    /// stream loop calls this after every `process_event` (and the
    /// Bedrock path after every frame) so events yield promptly
    /// instead of accumulating until stream end.
    pub(super) fn drain(&mut self) -> Vec<StreamEvent> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_build_injects_a_user_head_before_a_tool_use_opening() {
        // The provider-side safety net for the tool-use opening: a
        // history that begins with an assistant carrying a tool call —
        // the coalesced-seam shape — cannot be the request's first
        // message on strict providers, so the build rides a minimal
        // user message ahead of it.
        let spec = RequestBodySpec {
            model: "glm-4.7",
            messages: &[crate::message::Message::new(
                crate::message::Role::Assistant,
                vec![
                    crate::message::MessagePart::text("SUMMARY: the task so far."),
                    crate::message::MessagePart::ToolCall {
                        id: "call_1".to_string(),
                        name: "Bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                ],
            )],
            system: None,
            tools: Some(&[]),
            response_format: None,
            tool_constraint: &ToolConstraint::None,
        };
        let body = build_request_body(&spec, false, 1024, None, false);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(
            messages.len(),
            2,
            "the opening assistant gains a user head: {messages:?}"
        );
        assert_eq!(
            messages[0]["role"], "user",
            "the injected head is user-role: {messages:?}"
        );
        assert_eq!(
            messages[1]["role"], "assistant",
            "the tool-carrying assistant follows: {messages:?}"
        );
    }

    #[test]
    fn the_request_build_merges_adjacent_assistant_messages() {
        // The provider-side safety net: a future producer of the
        // adjacent-assistant shape cannot kill a run - the Anthropic
        // body build (zai rides it) merges the pair into one
        // assistant message carrying both texts' content blocks.
        let spec = RequestBodySpec {
            model: "glm-4.7",
            messages: &[
                crate::message::Message::user("drive it"),
                crate::message::Message::assistant("SUMMARY: the task so far."),
                crate::message::Message::assistant("the preserved tail turn"),
            ],
            system: None,
            tools: Some(&[]),
            response_format: None,
            tool_constraint: &ToolConstraint::None,
        };
        let body = build_request_body(&spec, false, 1024, None, false);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "the adjacent pair merges: {messages:?}");
        let merged = &messages[1];
        assert_eq!(
            merged["role"], "assistant",
            "the merged message keeps the assistant role: {merged:?}"
        );
        let text = merged["content"].to_string();
        assert!(
            text.contains("SUMMARY: the task so far.") && text.contains("the preserved tail turn"),
            "both texts ride the merged message: {text}"
        );
        assert!(
            text.contains("\\n\\n"),
            "the merged wire text parts the two statements with a separator: {text}"
        );
    }

    #[test]
    fn emitter_ignores_events_after_message_stop() {
        let mut emitter = StreamEmitter::default();
        emitter.process_event(
            "message_stop",
            Some(serde_json::json!({"type": "message_stop"})),
        );
        assert!(!emitter.drain().is_empty(), "message_stop emits the stop");
        emitter.process_event(
            "content_block_delta",
            Some(serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "late"},
            })),
        );
        assert!(
            emitter.drain().is_empty(),
            "a desynced stream must not append parts after the stop"
        );
        assert!(
            emitter.finish().unwrap_or_default().is_empty(),
            "nothing queued behind the guard"
        );
        emitter.process_event(
            "error",
            Some(serde_json::json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "late"}
            })),
        );
        assert!(
            emitter.finish().is_ok(),
            "a late error after a completed message is swallowed — the \
             stream already terminated cleanly, the contradiction is not \
             allowed to fail it"
        );
    }

    use crate::message::{Message, MessagePart, Role, ToolContent};

    #[tokio::test]
    async fn strict_response_format_is_rejected_loudly() {
        use crate::structured::{RequestOptions, ResponseFormat};
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("http://localhost:1".to_string())
            .build()
            .unwrap();
        let rf = ResponseFormat {
            name: "out".to_string(),
            schema: serde_json::json!({"type": "object"}),
            strict: true,
        };
        let err = client
            .create_message_with_options(
                &crate::api::StreamRequest::new(vec![]),
                RequestOptions::default().with_response_format(rf),
            )
            .await
            .expect_err("strict must fail fast, not be silently dropped");
        assert!(
            err.to_string().contains("strict"),
            "the error names the dropped field: {err}"
        );
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigValidationError,
            "a semantic capability rejection classifies as a validation \
             failure, not a parse error"
        );
    }

    #[tokio::test]
    async fn effort_with_response_format_is_rejected_loudly() {
        use crate::structured::{RequestOptions, ResponseFormat};
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("http://localhost:1".to_string())
            .build()
            .unwrap();
        let rf = ResponseFormat {
            name: "out".to_string(),
            schema: serde_json::json!({"type": "object"}),
            strict: false,
        };
        let err = client
            .create_message_with_options(
                &crate::api::StreamRequest::new(vec![]),
                RequestOptions::default()
                    .with_response_format(rf)
                    .with_effort(crate::structured::ThinkingEffort::High),
            )
            .await
            .expect_err("effort plus a forced tool_choice must fail fast, not be sent");
        assert!(
            err.to_string().contains("effort"),
            "the error names the rejected combination: {err}"
        );
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigValidationError,
            "a semantic capability rejection classifies as a validation failure"
        );

        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            RequestOptions::default()
                .with_response_format(ResponseFormat {
                    name: "out".to_string(),
                    schema: serde_json::json!({"type": "object"}),
                    strict: false,
                })
                .with_effort(crate::structured::ThinkingEffort::High),
        );
        let stream_err = futures::StreamExt::next(&mut stream)
            .await
            .expect("the streaming rejection fires before any request is sent")
            .expect_err("the streaming path must reject the same combination");
        assert!(
            stream_err.to_string().contains("effort"),
            "the streaming error names the rejected combination: {stream_err}"
        );
    }

    #[tokio::test]
    async fn a_client_effort_default_under_a_response_format_is_rejected_loudly() {
        use crate::structured::{RequestOptions, ResponseFormat};
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("http://localhost:1".to_string())
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        let err = client
            .create_message_with_options(
                &crate::api::StreamRequest::new(vec![]),
                RequestOptions::default().with_response_format(ResponseFormat {
                    name: "out".to_string(),
                    schema: serde_json::json!({"type": "object"}),
                    strict: false,
                }),
            )
            .await
            .expect_err("a client-level effort default must not hide under a response format");
        assert!(
            err.to_string().contains("effort"),
            "the error names the effective-effort rejection: {err}"
        );
    }

    #[tokio::test]
    async fn a_client_effort_default_under_a_response_format_is_rejected_loudly_on_stream() {
        use crate::structured::{RequestOptions, ResponseFormat};
        use futures::StreamExt;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("http://localhost:1".to_string())
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            RequestOptions::default().with_response_format(ResponseFormat {
                name: "out".to_string(),
                schema: serde_json::json!({"type": "object"}),
                strict: false,
            }),
        );
        let err = stream
            .next()
            .await
            .expect("the streaming rejection fires before any request is sent")
            .expect_err("a client-level effort default must not hide under a response format");
        assert!(
            err.to_string().contains("effort"),
            "the streaming error names the effective-effort rejection: {err}"
        );
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigValidationError,
            "a semantic capability rejection classifies as a validation failure"
        );
    }

    /// An assistant message whose history carries one thinking part per
    /// entry plus a trailing text part.
    fn assistant_with_thinking(traces: &[&str]) -> Message {
        let mut parts: Vec<MessagePart> = traces
            .iter()
            .map(|text| MessagePart::Thinking {
                text: (*text).to_string(),
                signature: None,
                redacted: None,
            })
            .collect();
        parts.push(MessagePart::text("the answer"));
        Message::new(Role::Assistant, parts)
    }

    /// A Models API disclosure for a model that thinks by default and takes
    /// its depth from `output_config.effort`: extended thinking refused,
    /// adaptive thinking accepted, `disabled` still accepted.
    const ADAPTIVE_DISCLOSURE: &str = r#"{"capabilities":{"thinking":{"types":{"disabled":{"supported":true},"between_tools":{"supported":false},"enabled":{"supported":false},"adaptive":{"supported":true}}}}}"#;

    /// A disclosure for a model whose lowest thinking setting is
    /// `between_tools`: Claude Sonnet 5.5, which refuses `"disabled"`.
    const BETWEEN_TOOLS_DISCLOSURE: &str = r#"{"capabilities":{"thinking":{"types":{"disabled":{"supported":false},"between_tools":{"supported":true}}}}}"#;

    /// A disclosure for a model that offers no way to turn thinking off at
    /// all: both off spellings refused. The rung has no wire shape here, so
    /// the client refuses the request before sending it rather than collecting
    /// a `400` over `thinking.type`.
    const NO_OFF_DISCLOSURE: &str = r#"{"capabilities":{"thinking":{"types":{"disabled":{"supported":false},"between_tools":{"supported":false}}}}}"#;

    /// A local endpoint that answers the thinking probe, then captures one request.
    ///
    /// A client with a resolved effort reads the model's thinking disclosure
    /// before it sends a Messages request, so a capture listener has to play
    /// both roles. `disclosure` is the document served for
    /// `GET /v1/models/{model}`; an empty `disclosure` answers `404`, the
    /// shape of an endpoint with no Models API. The Messages request that
    /// follows is captured and failed, so a caller asserts on the body the
    /// client would have sent.
    async fn capture_after_answering_probe(
        disclosure: &str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let disclosure = disclosure.to_string();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return String::new();
                };
                let mut buf = vec![0u8; 8192];
                let n = match sock.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => continue,
                };
                let text = String::from_utf8_lossy(&buf[..n]).into_owned();
                if !text.starts_with("GET /v1/models") {
                    drop(
                        sock.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n",
                        )
                        .await,
                    );
                    drop(sock.shutdown().await);
                    return text;
                }
                let reply = if disclosure.is_empty() {
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_string()
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{disclosure}",
                        disclosure.len()
                    )
                };
                drop(sock.write_all(reply.as_bytes()).await);
                drop(sock.shutdown().await);
            }
        });
        (addr, server)
    }

    /// A local endpoint that records every request line it receives.
    ///
    /// Answers the thinking probe with `disclosure` (empty means `404`) and
    /// fails every other request, so a test can drive a client and still see
    /// what it asked for — how many times it consulted the Models API among
    /// it. The caller aborts the task when done.
    async fn recording_listener(
        disclosure: &'static str,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let status = if disclosure.is_empty() { 404 } else { 200 };
        recording_listener_answering(disclosure, status).await
    }

    /// A recording endpoint that answers the thinking probe with a chosen
    /// status.
    ///
    /// [`recording_listener`] is this with the status an endpoint that has a
    /// Models API would use. A `200` carries `disclosure`; any other status
    /// answers empty, standing in for a metadata route that is down, busy, or
    /// refusing. Messages requests always fail: the caller drives a client
    /// repeatedly and reads back what it asked for, which is how a pin counts
    /// how many times the Models API was consulted.
    async fn recording_listener_answering(
        disclosure: &'static str,
        probe_status: u16,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 8192];
                let n = match sock.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => continue,
                };
                let text = String::from_utf8_lossy(&buf[..n]).into_owned();
                let reply: Vec<u8> = if !text.starts_with("GET /v1/models") {
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if probe_status != 200 {
                    format!(
                        "HTTP/1.1 {probe_status} Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    ).into_bytes()
                } else if disclosure.is_empty() {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_vec()
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{disclosure}",
                        disclosure.len()
                    ).into_bytes()
                };
                recorded.lock().unwrap().push(text);
                drop(sock.write_all(&reply).await);
                drop(sock.shutdown().await);
            }
        });
        (addr, seen, task)
    }

    /// A recording endpoint whose thinking probe fails `failures` times, then answers.
    ///
    /// The first `failures` `GET /v1/models/…` requests get `503` — a metadata
    /// route briefly down — and every later probe gets `disclosure`. Messages
    /// requests fail as elsewhere and every request is recorded, header and
    /// body together, so a caller sees both how often the client asked and what
    /// each turn put on the wire: the point being that a client which filed the
    /// `503` as an answer would never learn that the endpoint had come back.
    /// One failure covers the blip case; a count reaching
    /// [`CAPABILITY_PROBE_MISS_LIMIT`] covers the outage that outlasted the
    /// streak, where the client has already settled into the give-up verdict.
    async fn recording_listener_failing_probe(
        disclosure: &'static str,
        failures: usize,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let probes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 8192];
                let n = match sock.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => continue,
                };
                let text = String::from_utf8_lossy(&buf[..n]).into_owned();
                let reply: Vec<u8> = if !text.starts_with("GET /v1/models") {
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if probes.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < failures {
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{disclosure}",
                        disclosure.len()
                    ).into_bytes()
                };
                recorded.lock().unwrap().push(text);
                drop(sock.write_all(&reply).await);
                drop(sock.shutdown().await);
            }
        });
        (addr, seen, task)
    }

    /// Drive one streaming request on `client` and let it fail.
    ///
    /// These pins care about what the client *sent*, not what came back: the
    /// endpoint answers Messages requests with a failure on purpose.
    async fn drive_request(client: &AnthropicClient, options: crate::structured::RequestOptions) {
        use futures::StreamExt;

        let mut stream =
            client.stream_messages_with_options(&crate::api::StreamRequest::new(vec![]), options);
        let _ = stream.next().await;
    }

    /// Capture the streaming body a client puts on the wire for one history.
    ///
    /// Drives a client against an endpoint that answers the thinking probe
    /// with `404` — the shape of an endpoint with no Models API — so the
    /// captured body shows the ladder's own mapping rather than a
    /// disclosure-driven rewrite. `replay` turns reasoning replay on or off,
    /// `effort` is the client-level thinking default (the level is set on the
    /// builder, not on the request), and `history` is the conversation sent.
    /// The endpoint fails the Messages request on purpose: the caller asserts
    /// on the body the client would have sent.
    async fn captured_replay_body(
        replay: bool,
        effort: Option<crate::structured::ThinkingEffort>,
        history: Vec<Message>,
    ) -> String {
        use futures::StreamExt;

        let (addr, server) = capture_after_answering_probe("").await;

        let mut builder = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_max_tokens(1024);
        if replay {
            builder = builder.replay_reasoning(true);
        }
        if let Some(effort) = effort {
            builder = builder.thinking_effort(effort);
        }
        let client = builder.build().unwrap();
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(history),
            crate::structured::RequestOptions::default(),
        );
        let _ = stream.next().await;
        server.await.unwrap()
    }

    #[tokio::test]
    async fn replayed_reasoning_rides_the_wire_only_when_enabled() {
        let on = captured_replay_body(
            true,
            None,
            vec![assistant_with_thinking(&["first dedup pass"])],
        )
        .await;
        assert!(
            on.contains("<thinking>\\nfirst dedup pass\\n</thinking>"),
            "an enabled client renders history thinking as a text block: {on}"
        );
        let off = captured_replay_body(
            false,
            None,
            vec![assistant_with_thinking(&["first dedup pass"])],
        )
        .await;
        assert!(
            !off.contains("<thinking>"),
            "off by default — requests stay byte-identical to a client that never saw reasoning: {off}"
        );
    }

    #[tokio::test]
    async fn every_history_thinking_part_rides_the_replay_in_order() {
        let body = captured_replay_body(
            true,
            None,
            vec![assistant_with_thinking(&["alpha pass", "beta pass"])],
        )
        .await;
        let first = body.find("<thinking>\\nalpha pass\\n</thinking>");
        let second = body.find("<thinking>\\nbeta pass\\n</thinking>");
        assert!(
            first.is_some_and(|at| second.is_some_and(|other| at < other)),
            "each history thinking part renders as its own block, in part order: {body}"
        );
    }

    #[tokio::test]
    async fn signed_thinking_blocks_ride_continuations_when_effort_is_enabled() {
        let history = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::Thinking {
                    text: "alpha pass".to_string(),
                    signature: Some("sig-blob".to_string()),
                    redacted: None,
                },
                MessagePart::ToolCall {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = captured_replay_body(
            false,
            Some(crate::structured::ThinkingEffort::High),
            history,
        )
        .await;
        assert!(
            body.contains("\"signature\":\"sig-blob\"")
                && body.contains("\"thinking\":\"alpha pass\"")
                && body.contains("\"type\":\"thinking\""),
            "an effective effort returns the signed thinking block natively, ahead of the tool use: {body}"
        );
        assert!(
            !body.contains("<thinking>"),
            "the signed block rides as the native block, not the plain-text replay: {body}"
        );
    }

    #[tokio::test]
    async fn redacted_thinking_blocks_ride_continuations_verbatim() {
        let history = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::Thinking {
                    text: String::new(),
                    signature: None,
                    redacted: Some("opaque-data".to_string()),
                },
                MessagePart::ToolCall {
                    id: "call_2".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = captured_replay_body(
            false,
            Some(crate::structured::ThinkingEffort::High),
            history,
        )
        .await;
        assert!(
            body.contains("\"type\":\"redacted_thinking\"")
                && body.contains("\"data\":\"opaque-data\""),
            "a redacted block returns its opaque payload verbatim on continuations: {body}"
        );
    }

    #[tokio::test]
    async fn the_none_rung_renders_history_thinking_as_replay() {
        // The off rung's twin of the disabled-replay pin: a
        // per-request `None` over a `High` client default disables
        // the thinking block, so signed history parts must render as
        // the plain-text replay - native thinking blocks on a
        // thinking-disabled request are a Messages API rejection.
        let history = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::Thinking {
                    text: "alpha pass".to_string(),
                    signature: Some("sig-blob".to_string()),
                    redacted: None,
                },
                MessagePart::ToolCall {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = captured_replay_body(
            false,
            Some(crate::structured::ThinkingEffort::None),
            history,
        )
        .await;
        assert!(
            body.contains("<thinking>\\nalpha pass\\n</thinking>"),
            "the off rung renders reasoning as the plain-text replay: {body}"
        );
        assert!(
            !body.contains("\"signature\""),
            "no native thinking block rides a request the rung turned off: {body}"
        );
    }

    #[tokio::test]
    async fn the_none_rung_drops_a_redacted_part_with_no_text_to_replay() {
        let history = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::Thinking {
                    text: String::new(),
                    signature: None,
                    redacted: Some("opaque-data".to_string()),
                },
                MessagePart::ToolCall {
                    id: "call_2".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = captured_replay_body(
            false,
            Some(crate::structured::ThinkingEffort::None),
            history,
        )
        .await;
        assert!(
            !body.contains("\"type\":\"redacted_thinking\"") && !body.contains("opaque-data"),
            "a redacted part carries no text to replay, so neither its native block nor its payload rides a request the rung turned off: {body}"
        );
        assert!(
            body.contains("\"type\":\"tool_use\""),
            "the sibling tool call still rides the continuation request: {body}"
        );
    }

    #[tokio::test]
    async fn the_none_rung_lets_a_response_format_request_proceed() {
        // The format/effort conflict guard keys on thinking being
        // enabled: with the off rung there is no thinking block, so a
        // forced-format request is legal and must reach the wire -
        // the guard's rejection is reserved for rungs that budget.
        use futures::StreamExt;

        let (addr, server) = capture_after_answering_probe("").await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_max_tokens(64)
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        let options = crate::structured::RequestOptions::new()
            .with_effort(crate::structured::ThinkingEffort::None)
            .with_response_format(crate::structured::ResponseFormat {
                name: "result".to_string(),
                schema: serde_json::json!({"type": "object"}),
                strict: false,
            });
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![Message::user("hi")]),
            options,
        );
        let _ = stream.next().await;
        drop(stream);
        let request = server.await.unwrap();
        assert!(
            request.contains("\"messages\""),
            "the request reached the wire instead of the conflict guard: {request}"
        );
    }

    #[tokio::test]
    async fn signed_blocks_stay_text_replay_when_thinking_is_disabled() {
        let history = vec![assistant_with_thinking(&["alpha pass"])];
        let body = captured_replay_body(true, None, history).await;
        assert!(
            body.contains("<thinking>\\nalpha pass\\n</thinking>"),
            "without an effective effort the reasoning renders as the plain-text replay: {body}"
        );
        assert!(
            !body.contains("\"signature\""),
            "no native thinking block rides a request that did not enable thinking: {body}"
        );
    }

    #[tokio::test]
    async fn an_unsigned_thinking_part_leaves_tool_use_alone_under_effort() {
        let history = vec![Message::new(
            Role::Assistant,
            vec![
                MessagePart::Thinking {
                    text: "legacy pass".to_string(),
                    signature: None,
                    redacted: None,
                },
                MessagePart::ToolCall {
                    id: "call_3".to_string(),
                    name: "search".to_string(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
        )];
        let body = captured_replay_body(
            false,
            Some(crate::structured::ThinkingEffort::High),
            history,
        )
        .await;
        assert!(
            body.contains("\"type\":\"tool_use\""),
            "the tool call still rides the continuation request: {body}"
        );
        assert!(
            !body.contains("\"type\":\"thinking\"")
                && !body.contains("\"type\":\"redacted_thinking\"")
                && !body.contains("\"signature\""),
            "an unsigned part cannot ride natively — it carries no signature to return: {body}"
        );
        assert!(
            !body.contains("<thinking>"),
            "with replay off an unsigned part renders no text replay either — tool_use rides alone, the rejection the thinking_effort builder doc names: {body}"
        );
    }

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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
        );
        assert_eq!(body["system"], "");
    }

    #[test]
    fn request_body_carries_a_configured_max_tokens() {
        let msgs = vec![Message::user("hi")];
        let spec = RequestBodySpec {
            model: "claude-sonnet-4",
            messages: &msgs,
            system: None,
            tools: None,
            response_format: None,
            tool_constraint: &ToolConstraint::None,
        };
        let body = build_request_body(&spec, false, 1_024, None, false);
        assert_eq!(
            body["max_tokens"], 1_024,
            "a non-default budget rides the wire verbatim"
        );
        let body = build_request_body(&spec, false, DEFAULT_MAX_TOKENS, None, false);
        assert_eq!(
            body["max_tokens"], DEFAULT_MAX_TOKENS,
            "and the default is what the builder sends when nothing is set"
        );
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
        );
        assert_eq!(body["messages"][0]["role"], "assistant");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn request_body_assistant_tool_calls() {
        let msgs = vec![
            Message::user("drive it"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::ToolCall {
                    id: "call_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"msg": "hi"}),
                }],
            ),
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
            None,
            false,
        );

        let msg = &body["messages"][1];
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
                name: "echo".into(),
                output: ToolContent::from_string("result text"),
                is_error: None,
                retention: None,
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
            None,
            false,
        );

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        let content = msg["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_1");
        assert_eq!(content[0]["content"], "result text");
    }

    #[test]
    fn prompt_cache_breakpoints_mark_tools_system_and_tail_blocks() {
        let msgs = vec![
            Message::user("earlier turn"),
            Message::new(
                Role::User,
                vec![
                    MessagePart::ToolResult {
                        call_id: "call_1".into(),
                        name: "echo".into(),
                        output: ToolContent::from_string("result text"),
                        is_error: None,
                        retention: None,
                    },
                    MessagePart::ToolResult {
                        call_id: "call_2".into(),
                        name: "echo".into(),
                        output: ToolContent::from_string("later result"),
                        is_error: None,
                        retention: None,
                    },
                ],
            ),
        ];
        let tools = vec![ToolSchema {
            tool: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let mut body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &msgs,
                system: Some("be brief"),
                tools: Some(&tools),
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            DEFAULT_MAX_TOKENS,
            None,
            false,
        );
        apply_prompt_cache_breakpoints(&mut body);

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(
            tools_arr[0]["cache_control"]["type"], "ephemeral",
            "the single tool definition carries the marker"
        );

        let system_arr = body["system"].as_array().unwrap();
        assert_eq!(system_arr[0]["type"], "text");
        assert_eq!(system_arr[0]["text"], "be brief");
        assert_eq!(system_arr[0]["cache_control"]["type"], "ephemeral");

        let final_blocks = body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_array()
            .unwrap();
        assert_eq!(final_blocks.len(), 2);
        assert_eq!(
            final_blocks[0]["cache_control"]["type"], "ephemeral",
            "second-to-last block keeps the previous prefix cached"
        );
        assert_eq!(
            final_blocks[1]["cache_control"]["type"], "ephemeral",
            "last block rolls the cached prefix forward"
        );

        let earlier_content = &body["messages"].as_array().unwrap()[0]["content"];
        assert_eq!(
            earlier_content, "earlier turn",
            "non-tail messages keep their plain-string shape"
        );
    }

    #[test]
    fn prompt_cache_breakpoints_wrap_string_tail_and_skip_empty_system() {
        let msgs = vec![Message::user("hello")];
        let mut body = build_request_body(
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
            None,
            false,
        );
        apply_prompt_cache_breakpoints(&mut body);

        assert_eq!(
            body["system"], "",
            "an empty system stays a string — marking nothing buys no cache"
        );
        let tail = body["messages"].as_array().unwrap().last().unwrap();
        let blocks = tail["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "hello");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn prompt_cache_breakpoints_skip_thinking_blocks() {
        let thinking_tail = Message::new(
            Role::Assistant,
            vec![
                MessagePart::text("signed reasoning precedes the answer"),
                MessagePart::Thinking {
                    text: "chain of thought".into(),
                    signature: Some("sig".into()),
                    redacted: None,
                },
                MessagePart::text("the answer"),
            ],
        );
        let msgs = vec![Message::user("hello"), thinking_tail];
        let mut body = build_request_body(
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
            Some(crate::structured::ThinkingEffort::Low),
            false,
        );
        apply_prompt_cache_breakpoints(&mut body);

        let blocks = body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_array()
            .unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0]["type"], "thinking",
            "the signed thinking block rides the wire natively"
        );
        assert!(
            blocks[0].get("cache_control").is_none(),
            "cache_control on a thinking block fails Anthropic request validation"
        );
        assert_eq!(
            blocks[1]["cache_control"]["type"], "ephemeral",
            "the last eligible text block rolls the cached prefix forward"
        );
        assert_eq!(
            blocks[2]["type"], "text",
            "the final eligible block is the trailing text"
        );
        assert_eq!(
            blocks[2]["cache_control"]["type"], "ephemeral",
            "the final eligible block carries the marker"
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
            None,
            false,
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
            None,
            false,
        );
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_assistant_with_text_and_tool_call() {
        let msgs = vec![
            Message::user("find it"),
            Message::new(
                Role::Assistant,
                vec![
                    MessagePart::text("Let me search."),
                    MessagePart::ToolCall {
                        id: "call_1".into(),
                        name: "search".into(),
                        input: serde_json::json!({"q": "rust"}),
                    },
                ],
            ),
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
            None,
            false,
        );

        let msg = &body["messages"][1];
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
            None,
            false,
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
    fn emitter_text_after_tool_uses_server_index_not_zero() {
        let mut em = StreamEmitter::default();

        // Tool-use block at server index 0.
        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "echo"}
        })));
        em.drain();

        // Text block at server index 1 — must NOT collide with the tool at 0.
        em.on_block_start(Some(serde_json::json!({
            "index": 1,
            "content_block": {"type": "text"}
        })));
        let starts = em.drain();
        let text_start = starts
            .iter()
            .find(|e| matches!(e, StreamEvent::PartStart(ps) if ps.index == 1))
            .expect("text PartStart must carry the server index 1");

        // Text delta must route to index 1, not the hardcoded 0 that would
        // collide with the tool-use part.
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "text_delta", "text": "after"}
        })));
        let deltas = em.drain();
        match &deltas[0] {
            StreamEvent::IndexedDelta(d) => {
                assert_eq!(d.index, 1, "text delta must use the server index, not 0");
            }
            other => panic!("expected IndexedDelta, got {other:?}"),
        }
        // Confirm the PartStart we matched above really is a text part.
        let StreamEvent::PartStart(ps) = text_start else {
            panic!("matched event must be a PartStart");
        };
        assert!(
            ps.part
                .as_ref()
                .is_some_and(crate::message::MessagePart::is_text)
        );
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
        em.text_index = Some(0);

        em.on_block_stop(None);
        let events = em.drain();
        assert!(matches!(events[0], StreamEvent::PartStop { .. }));
        assert!(em.text_index.is_none());
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
        em.text_index = Some(0);
        em.tool_parts_open = 2;

        em.on_message_stop();
        let events = em.drain();
        // 1 PartStop for text + 2 for tools, then the terminal MessageStop.
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], StreamEvent::PartStop { .. }));
        assert!(matches!(events[1], StreamEvent::PartStop { .. }));
        assert!(matches!(events[2], StreamEvent::PartStop { .. }));
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

        let after_finish = em.finish().expect("no recorded error");
        assert!(
            after_finish
                .iter()
                .all(|e| !matches!(e, StreamEvent::MessageStop)),
            "finish() must not emit a second MessageStop after on_message_stop: {after_finish:?}"
        );
    }

    #[test]
    fn emitter_finish_emits_no_stop_without_message_stop() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.finished = false;

        let events = em.finish().expect("no recorded error");
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, StreamEvent::MessageStop)),
            "a stream that ends without message_stop is truncated — no synthetic stop may dress it up"
        );
    }

    #[test]
    fn emitter_finish_noop_if_already_stopped() {
        let mut em = StreamEmitter::default();
        em.started = true;
        em.finished = true;

        let events = em.finish().expect("no recorded error");
        assert!(events.is_empty());
    }

    #[test]
    fn midstream_error_event_surfaces_as_an_error() {
        let mut em = StreamEmitter::default();
        em.process_event("message_start", Some(serde_json::json!({})));
        em.process_event(
            "content_block_start",
            Some(serde_json::json!({
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })),
        );
        em.process_event(
            "content_block_delta",
            Some(serde_json::json!({
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"}
            })),
        );
        em.process_event(
            "error",
            Some(serde_json::json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"}
            })),
        );
        assert!(
            em.error_recorded(),
            "an event: error mid-stream must be recorded"
        );
        let drained = em.drain();
        assert!(
            drained
                .iter()
                .all(|e| !matches!(e, StreamEvent::MessageStop)),
            "no clean MessageStop may be emitted alongside the failure: {drained:?}"
        );
        let err = em
            .finish()
            .expect_err("finish must surface the recorded error");
        assert!(
            err.to_string().contains("overloaded_error"),
            "the terminal error must name the provider's type: {err}"
        );
        assert!(
            err.to_string().contains("Overloaded"),
            "the terminal error must carry the provider's message: {err}"
        );
    }

    #[test]
    fn builder_rejects_zero_max_tokens() {
        let built = AnthropicClient::builder()
            .with_api_key("test")
            .with_max_tokens(0)
            .build();
        let Err(err) = built else {
            panic!("a zero max_tokens must fail at build time")
        };
        assert!(
            err.to_string().contains("max_tokens"),
            "the error must name the field: {err}"
        );
    }

    #[cfg(feature = "grammar")]
    #[tokio::test]
    async fn grammar_constraint_errors_loudly() {
        let client = AnthropicClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let opts = crate::structured::RequestOptions::default().with_tool_constraint(
            crate::structured::ToolConstraint::Grammar(std::sync::Arc::new(
                crate::provider::grammar::JsonSchemaGrammar::from_schemas(&[])
                    .expect("an empty slice compiles"),
            )),
        );
        let mut stream =
            client.stream_messages_with_options(&crate::api::StreamRequest::new(vec![]), opts);
        let first = futures::StreamExt::next(&mut stream).await;
        assert!(
            matches!(&first, Some(Err(err)) if err.to_string().contains("grammar")),
            "a grammar constraint must be rejected, not silently downgraded: {first:?}"
        );
    }

    #[test]
    fn emitter_downward_delta_revision_loses_to_the_start_latch() {
        let mut em = StreamEmitter::default();
        em.process_event(
            "message_start",
            Some(serde_json::json!({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 25, "output_tokens": 1}}
            })),
        );
        em.drain();
        em.process_event(
            "message_delta",
            Some(serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 12, "output_tokens": 5}
            })),
        );
        let events = em.drain();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
                _ => None,
            })
            .expect("the terminal delta carries usage");
        assert_eq!(
            usage.input_tokens, 25,
            "the merge is max-by-assumption: a downward revision loses to the latched start value"
        );
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn midstream_rate_limit_family_errors_classify_as_rate_limit() {
        for kind in ["rate_limit_error", "overloaded_error"] {
            let mut em = StreamEmitter::default();
            em.process_event(
                "error",
                Some(serde_json::json!({
                    "type": "error",
                    "error": {"type": kind, "message": "slow down"}
                })),
            );
            let err = em
                .finish()
                .expect_err("finish must surface the recorded error");
            assert!(
                matches!(err, ApiError::RateLimit { .. }),
                "{kind} is the SSE form of a rate-limit status and must classify as RateLimit"
            );
        }

        let mut em = StreamEmitter::default();
        em.process_event(
            "error",
            Some(serde_json::json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad shape"}
            })),
        );
        let err = em
            .finish()
            .expect_err("finish must surface the recorded error");
        assert!(
            !matches!(err, ApiError::RateLimit { .. }),
            "non-rate-limit errors stay provider errors: {err}"
        );
    }

    #[test]
    fn sse_reader_take_line_extracts_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "event: ping\n".into(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        assert_eq!(reader.take_line().unwrap().unwrap(), "event: ping");
        assert!(reader.buf.is_empty());
    }

    #[test]
    fn sse_reader_take_line_none_without_newline() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "partial".into(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        assert!(reader.take_line().unwrap().is_none());
    }

    #[test]
    fn sse_reader_take_line_multiple() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "line1\nline2\n".into(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        assert_eq!(reader.take_line().unwrap().unwrap(), "line1");
        assert_eq!(reader.take_line().unwrap().unwrap(), "line2");
    }

    #[test]
    fn sse_reader_take_line_trims_cr() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: "data: hi\r\n".into(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        assert_eq!(reader.take_line().unwrap().unwrap(), "data: hi");
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
            buf: "event: message_start\ndata: {}\n\n"
                .to_string()
                .into_bytes(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        assert_eq!(
            reader.take_line().unwrap(),
            Some("event: message_start".to_string())
        );
        assert_eq!(reader.take_line().unwrap(), Some("data: {}".to_string()));
        assert_eq!(reader.take_line().unwrap(), Some(String::new()));
    }

    #[tokio::test]
    async fn sse_reader_next_event_extracts_payload() {
        let chunk = "event: content_block_delta\ndata: {\"type\":\"text_delta\"}\n\n";
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(chunk.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
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
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(chunk.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
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
        let stream =
            futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(chunk.to_string().into())]);
        let mut reader = SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let result = reader.next_event().await.unwrap();
        assert!(result.is_some());
        let (event_type, data) = result.unwrap();
        assert_eq!(event_type, "ping");
        assert!(data.is_none(), "malformed JSON should yield None data");
    }

    #[tokio::test]
    async fn sse_reader_buffer_overflow_returns_error() {
        let huge = "x".repeat(2 * 1024 * 1024);
        let stream = futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(huge.into())]);
        let mut reader = super::SseReader {
            bytes: Box::pin(stream),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
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
        assert_eq!(super::super::MAX_RESPONSE_BODY, 10 * 1024 * 1024);
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
        );
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should only appear with response_format"
        );
    }

    #[test]
    fn extract_structured_from_tool_call_part() {
        let client = AnthropicClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::new(
            Role::Assistant,
            vec![MessagePart::tool_call(
                "tu_1",
                "action",
                serde_json::json!({"tool": "write", "args": {}}),
            )],
        );
        let value = client.extract_structured(&message);
        assert_eq!(value["tool"], "write");
    }

    #[test]
    fn extract_structured_text_only_falls_back_to_string() {
        let client = AnthropicClient::builder()
            .with_api_key("test")
            .build()
            .unwrap();
        let message = Message::assistant("I cannot do that.");
        let value = client.extract_structured(&message);
        assert_eq!(value, serde_json::json!("I cannot do that."));
    }

    #[test]
    fn build_response_maps_text_block_and_end_turn() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hello there"}],
            "stop_reason": "end_turn"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.message.role, Role::Assistant);
        assert_eq!(response.message.text_content(), "hello there");
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_reports_cached_input_tokens() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 20,
                "cache_read_input_tokens": 1200,
                "cache_creation_input_tokens": 300
            }
        });
        let response = AnthropicClient::build_response(&raw);
        let usage = response.usage.expect("usage");
        assert_eq!(
            usage.input_tokens, 3000,
            "raw input excludes cache traffic — the normalized input adds the \
             cached read and the cache-creation spend (1500 + 1200 + 300)"
        );
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(
            usage.cached_input_tokens, 1200,
            "the cached read share rides the usage report; creation spend is not a read"
        );
    }

    #[test]
    fn build_response_without_cache_fields_reports_zero_cached() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 30, "output_tokens": 12}
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.usage.expect("usage").cached_input_tokens, 0);
    }

    #[test]
    fn build_response_reports_cache_write_tokens() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 20,
                "cache_read_input_tokens": 1200,
                "cache_creation_input_tokens": 300
            }
        });
        let response = AnthropicClient::build_response(&raw);
        let usage = response.usage.expect("usage");
        assert_eq!(
            usage.cache_write_tokens, 300,
            "the cache-creation spend is its own pricing class, distinct from the cached read"
        );
        assert_eq!(
            usage.reasoning_tokens, 0,
            "the Messages usage object reports no reasoning class — unreported, not zero-known"
        );
    }

    #[test]
    fn anthropic_totals_compose_without_double_counting() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 50
            }
        });
        let usage = AnthropicClient::build_response(&raw).usage.expect("usage");
        assert_eq!(
            usage.input_tokens, 1050,
            "raw 100 + read 900 + creation 50 — the normalized input IS the conversation \
             total; summing the detail fields on top of it would double-count"
        );
        assert_eq!(usage.cached_input_tokens, 900);
        assert_eq!(usage.cache_write_tokens, 50);
        assert_eq!(usage.total_tokens(), 1055);
    }

    #[test]
    fn build_response_maps_tool_use_block_and_tool_call() {
        let raw = serde_json::json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search",
                "input": {"q": "rust"}
            }],
            "stop_reason": "tool_use"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 1);
        match &response.message.parts[0] {
            MessagePart::ToolCall { id, name, input } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "search");
                assert_eq!(input, &serde_json::json!({"q": "rust"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        assert_eq!(response.stop_reason, StreamStopReason::ToolCall);
    }

    #[test]
    fn build_response_preserves_block_order() {
        let raw = serde_json::json!({
            "content": [
                {"type": "text", "text": "thinking..."},
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "text", "text": "done"}
            ],
            "stop_reason": "end_turn"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 3);
        assert!(response.message.parts[0].is_text());
        assert!(response.message.parts[1].is_tool_call());
        assert!(response.message.parts[2].is_text());
    }

    #[test]
    fn build_response_captures_thinking_blocks_with_their_signatures() {
        let raw = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "internal reasoning", "signature": "sig-blob"},
                {"type": "redacted_thinking", "data": "opaque-data"},
                {"type": "text", "text": "visible answer"}
            ],
            "stop_reason": "end_turn"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(
            response.message.parts.len(),
            3,
            "thinking and redacted blocks survive as parts in order: {:?}",
            response.message.parts
        );
        assert_eq!(response.message.text_content(), "visible answer");
        assert!(
            matches!(
                response.message.parts.first(),
                Some(MessagePart::Thinking { text, signature, redacted })
                    if text == "internal reasoning"
                        && signature.as_deref() == Some("sig-blob")
                        && redacted.is_none()
            ),
            "a signed thinking block keeps its signature for native replay: {:?}",
            response.message.parts
        );
        assert!(
            matches!(
                response.message.parts.get(1),
                Some(MessagePart::Thinking { text, signature, redacted })
                    if text.is_empty()
                        && signature.is_none()
                        && redacted.as_deref() == Some("opaque-data")
            ),
            "a redacted block keeps its opaque payload verbatim: {:?}",
            response.message.parts
        );
    }

    #[test]
    fn build_response_maps_max_tokens_stop_reason() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::MaxTokens);
    }

    #[test]
    fn build_response_unknown_stop_reason_defaults_to_end_turn() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "something_new"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_missing_stop_reason_defaults_to_end_turn() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}]
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_empty_content_yields_empty_message() {
        let raw = serde_json::json!({"content": [], "stop_reason": "end_turn"});
        let response = AnthropicClient::build_response(&raw);
        assert!(response.message.parts.is_empty());
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_extracts_usage() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 12}
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.usage.expect("usage").input_tokens, 30);
        assert_eq!(response.usage.expect("usage").output_tokens, 12);
    }

    #[test]
    fn build_response_missing_usage_is_none() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn"
        });
        let response = AnthropicClient::build_response(&raw);
        assert!(response.usage.is_none());
    }

    #[test]
    fn build_response_missing_content_yields_empty_message() {
        let raw = serde_json::json!({"stop_reason": "end_turn"});
        let response = AnthropicClient::build_response(&raw);
        assert!(response.message.parts.is_empty());
        assert_eq!(response.stop_reason, StreamStopReason::EndTurn);
    }

    #[test]
    fn build_response_text_block_missing_text_is_skipped() {
        let raw = serde_json::json!({
            "content": [
                {"type": "text"},
                {"type": "text", "text": "valid"}
            ],
            "stop_reason": "end_turn"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.message.parts.len(), 1);
        assert_eq!(response.message.text_content(), "valid");
    }

    #[test]
    fn build_response_tool_use_missing_input_defaults_to_null() {
        let raw = serde_json::json!({
            "content": [{"type": "tool_use", "id": "tu_1", "name": "search"}],
            "stop_reason": "tool_use"
        });
        let response = AnthropicClient::build_response(&raw);
        match &response.message.parts[0] {
            MessagePart::ToolCall { input, .. } => {
                assert_eq!(input, &Value::Null);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn build_response_maps_stop_sequence_reason() {
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "stopped"}],
            "stop_reason": "stop_sequence"
        });
        let response = AnthropicClient::build_response(&raw);
        assert_eq!(response.stop_reason, StreamStopReason::StopSequence);
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
            None,
            false,
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
                DeltaPart::Thinking { text, .. } => assert_eq!(text, "reasoning here"),
                other => panic!("expected Thinking, got {other:?}"),
            },
            other => panic!("expected IndexedDelta, got {other:?}"),
        }
    }

    #[test]
    fn signature_delta_rides_the_thinking_lane_as_its_own_delta() {
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "thinking"}
        })));
        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "thinking_delta", "thinking": "visible reasoning"}
        })));
        em.drain();

        em.on_block_delta(Some(serde_json::json!({
            "delta": {"type": "signature_delta", "signature": "opaque_base64_blob"}
        })));
        let events = em.drain();

        let carried = events.iter().find_map(|e| match e {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { signature, .. } => signature.clone(),
                _ => None,
            },
            _ => None,
        });
        assert_eq!(
            carried.as_deref(),
            Some("opaque_base64_blob"),
            "the block's signature rides a thinking-lane delta so continuations can replay it: {events:?}"
        );
    }

    #[test]
    fn redacted_thinking_data_rides_the_thinking_lane() {
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque_payload"}
        })));
        let events = em.drain();

        let carried = events.iter().find_map(|e| match e {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { redacted, .. } => redacted.clone(),
                _ => None,
            },
            _ => None,
        });
        assert_eq!(
            carried.as_deref(),
            Some("opaque_payload"),
            "the redacted block's payload rides its delta verbatim so continuations can return it: {events:?}"
        );
    }

    #[test]
    fn emitter_redacted_thinking_emits_empty_delta() {
        let mut em = StreamEmitter::default();

        em.on_block_start(Some(serde_json::json!({
            "index": 0,
            "content_block": {"type": "redacted_thinking"}
        })));
        let events = em.drain();

        assert_eq!(
            events.len(),
            2,
            "expected PartStart + one empty-text Thinking delta"
        );
        assert!(matches!(events[0], StreamEvent::PartStart(_)));
        match &events[1] {
            StreamEvent::IndexedDelta(d) => match &d.delta {
                DeltaPart::Thinking { text, .. } => {
                    assert!(
                        text.is_empty(),
                        "redacted thinking arrives with empty text — its payload rides the same delta's redacted field"
                    );
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
        assert!(matches!(events[0], StreamEvent::PartStop { .. }));
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
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PartStop { .. })),
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
            .position(|e| matches!(e, StreamEvent::PartStop { .. }))
            .expect("PartStop present");
        let msg_stop_idx = events
            .iter()
            .position(|e| matches!(e, StreamEvent::MessageStop))
            .expect("MessageStop present");
        assert!(stop_idx < msg_stop_idx, "PartStop must precede MessageStop");
    }

    #[test]
    fn request_body_omits_tools_for_empty_slice() {
        let body = build_request_body(
            &RequestBodySpec {
                model: "claude-3",
                messages: &[crate::message::Message::user("hi")],
                system: None,
                tools: Some(&[]),
                response_format: None,
                tool_constraint: &ToolConstraint::None,
            },
            false,
            1024,
            None,
            false,
        );
        assert!(
            body.get("tools").is_none(),
            "an empty tool list must be omitted, not sent as []; got {}",
            body.get("tools").unwrap_or(&serde_json::Value::Null)
        );
    }

    #[test]
    fn convert_message_marks_error_tool_results() {
        let msg = crate::message::Message::new(
            crate::message::Role::User,
            vec![crate::message::MessagePart::tool_result(
                "c1",
                "bash",
                crate::message::ToolContent::from_string("exit 1"),
                true,
            )],
        );
        let json = convert_message(&msg, false, None);
        assert_eq!(
            json.pointer("/content/0/is_error"),
            Some(&serde_json::json!(true)),
            "Anthropic's tool_result block supports is_error and must receive it; got {json}"
        );
        let ok = crate::message::Message::new(
            crate::message::Role::User,
            vec![crate::message::MessagePart::tool_result(
                "c2",
                "search",
                crate::message::ToolContent::from_string("[]"),
                false,
            )],
        );
        let ok_json = convert_message(&ok, false, None);
        assert!(
            ok_json.pointer("/content/0/is_error").is_none(),
            "successful results carry no is_error (the wire default is false): {ok_json}"
        );
    }

    #[test]
    fn messages_url_has_no_double_slash_for_trailing_slash_base() {
        let bare = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("https://api.example.com")
            .build()
            .expect("client builds");
        let slashed = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url("https://api.example.com/")
            .build()
            .expect("client builds");
        assert_eq!(
            slashed.messages_url(),
            bare.messages_url(),
            "a trailing-slash base URL must join to the same request URL as the bare one"
        );
    }

    #[test]
    fn emitter_streams_input_tokens_from_message_start() {
        let mut em = StreamEmitter::default();
        let start_data = serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        });
        em.on_message_start(Some(&start_data));
        em.drain();
        let delta_data = serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        });
        em.on_message_delta(Some(delta_data));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.input_tokens, 25,
            "input tokens arrive on message_start and must survive to the terminal usage"
        );
        assert_eq!(usage.output_tokens, 15);
    }

    #[test]
    fn emitter_merges_cached_input_tokens_across_start_and_delta() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "output_tokens": 15,
                "cache_read_input_tokens": 1250
            }
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.input_tokens, 2750,
            "raw 1500 + revised read 1250 — the fold identity holds across the merge: \
             the terminal delta's upward read revision rides the input total"
        );
        assert_eq!(
            usage.cached_input_tokens, 1250,
            "the terminal delta's revised cache read wins over the latched start value"
        );
    }

    #[test]
    fn emitter_keeps_latched_cached_tokens_when_delta_omits_them() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(usage.input_tokens, 2700);
        assert_eq!(
            usage.cached_input_tokens, 1200,
            "a delta without cache fields must not erase the start event's cached share"
        );
    }

    #[test]
    fn emitter_merges_cache_write_across_start_and_delta() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200,
                    "cache_creation_input_tokens": 300
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "output_tokens": 15,
                "cache_creation_input_tokens": 350
            }
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.input_tokens, 3050,
            "raw 1500 + read 1200 + revised creation 350 — the start-side normalized total \
             with the delta's upward revision folded in, so the fold identity holds"
        );
        assert_eq!(
            usage.cache_write_tokens, 350,
            "the terminal delta's revised cache write wins over the latched start value"
        );
        assert_eq!(usage.cached_input_tokens, 1200);
    }

    #[test]
    fn anthropic_stream_merge_folds_a_revised_share_into_the_input_total() {
        // One delta revises both cache shares upward; the input total
        // must carry raw + revised-read + revised-write, the same
        // normalized figure the non-streaming reader produces for the
        // identical report.
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200,
                    "cache_creation_input_tokens": 300
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "output_tokens": 15,
                "cache_read_input_tokens": 1300,
                "cache_creation_input_tokens": 400
            }
        })));
        let usage = em.drain().iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.input_tokens, 3200,
            "1500 raw + 1300 revised read + 400 revised write"
        );
        assert_eq!(usage.cached_input_tokens, 1300);
        assert_eq!(usage.cache_write_tokens, 400);

        let non_streamed = AnthropicClient::build_response(&serde_json::json!({
            "id": "msg_1",
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 15,
                "cache_read_input_tokens": 1300,
                "cache_creation_input_tokens": 400
            }
        }))
        .usage
        .expect("non-streaming usage");
        assert_eq!(
            non_streamed.input_tokens, usage.input_tokens,
            "the streaming fold equals the non-streaming normalized input for the same report"
        );
    }

    #[test]
    fn a_generated_counts_sweep_holds_the_anthropic_fold_identity() {
        // Deterministic xorshift (TESTING doctrine) over raw + read +
        // write reports, both present and delta-revised: the merged
        // input must equal the sum of the merged classes, and the
        // classes must stay subsets of it. Terminal deltas also revise
        // the raw input, sometimes without repeating the cache fields
        // — the corner the latched-share carry defends.
        let mut seed: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut revised_cases = 0usize;
        let mut share_less_revisions = 0usize;
        for _case in 0..40 {
            let raw = u32::try_from(next() % 2_000).unwrap();
            let read = u32::try_from(next() % 2_000).unwrap();
            let write = u32::try_from(next() % 2_000).unwrap();
            let read_delta = read.saturating_add(u32::try_from(next() % 500).unwrap());
            let write_delta = write.saturating_add(u32::try_from(next() % 500).unwrap());
            revised_cases += usize::from(read_delta > read || write_delta > write);
            let raw_delta = raw.saturating_add(u32::try_from(next() % 500).unwrap());
            let revise_input = next() % 4 == 0;
            let omit_shares = revise_input && next() % 2 == 0;
            share_less_revisions += usize::from(omit_shares);
            let mut delta_usage = serde_json::json!({ "output_tokens": 15 });
            if revise_input {
                delta_usage["input_tokens"] = serde_json::json!(raw_delta);
            }
            if !omit_shares {
                delta_usage["cache_read_input_tokens"] = serde_json::json!(read_delta);
                delta_usage["cache_creation_input_tokens"] = serde_json::json!(write_delta);
            }

            let mut em = StreamEmitter::default();
            em.on_message_start(Some(&serde_json::json!({
                "message": {
                    "id": "msg_1",
                    "model": "claude-3",
                    "usage": {
                        "input_tokens": raw,
                        "output_tokens": 1,
                        "cache_read_input_tokens": read,
                        "cache_creation_input_tokens": write
                    }
                }
            })));
            em.drain();
            em.on_message_delta(Some(serde_json::json!({
                "delta": {"stop_reason": "end_turn"},
                "usage": delta_usage
            })));
            let usage = em.drain().iter().find_map(|e| match e {
                StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
                _ => None,
            });
            let usage = usage.expect("MessageDelta must carry usage");
            let merged_raw = if revise_input { raw_delta } else { raw };
            let merged_read = if omit_shares { read } else { read_delta };
            let merged_write = if omit_shares { write } else { write_delta };
            assert_eq!(
                usage.cached_input_tokens, merged_read,
                "the read latch takes the larger report: {raw}/{read}/{write}"
            );
            assert_eq!(
                usage.cache_write_tokens, merged_write,
                "the write latch takes the larger report: {raw}/{read}/{write}"
            );
            assert_eq!(
                usage.input_tokens,
                merged_raw
                    .saturating_add(merged_read)
                    .saturating_add(merged_write),
                "the fold identity holds on the merged input: {usage:?} for \
                 raw {raw}->{merged_raw} + read {read}->{merged_read} + \
                 write {write}->{merged_write}"
            );
            assert!(
                usage.cached_input_tokens <= usage.input_tokens
                    && usage.cache_write_tokens <= usage.input_tokens,
                "the cache classes stay subsets of the folded input: {usage:?}"
            );
        }
        assert!(
            revised_cases > 0 && share_less_revisions > 0,
            "anti-vacuity: the sweep really revised a share upward and drew \
             input revisions without the cache fields"
        );
    }

    #[test]
    fn anthropic_stream_delta_revising_input_carries_the_latched_shares_into_the_fold() {
        // The server-tools shape the defensive delta-input path exists
        // for: the terminal delta revises the raw input upward and does
        // not repeat the cache fields. The merged input keeps the
        // family identity — the delta's raw with the latched shares
        // carried on top (raw 4 000 + latched 1 200 + 300 = 5 500).
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200,
                    "cache_creation_input_tokens": 300
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 4000, "output_tokens": 15}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.input_tokens, 5500,
            "the delta's revised raw carries the latched shares: {usage:?}"
        );
        assert_eq!(usage.cached_input_tokens, 1200);
        assert_eq!(usage.cache_write_tokens, 300);
        assert_eq!(usage.output_tokens, 15);
    }

    #[test]
    fn emitter_keeps_latched_cache_write_when_delta_omits_it() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 1200,
                    "cache_creation_input_tokens": 300
                }
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15, "cache_read_input_tokens": 1200}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            usage.cache_write_tokens, 300,
            "a delta carrying only the read share must not erase the latched cache write"
        );
    }

    #[test]
    fn streamed_and_non_streamed_usage_agree() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 25, "output_tokens": 15}
        });
        let non_streamed = AnthropicClient::build_response(&raw);

        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        })));
        let streamed = em.drain().iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });

        assert_eq!(
            streamed, non_streamed.usage,
            "the same exchange served streaming and non-streaming must report identical usage"
        );
    }

    #[tokio::test]
    async fn sse_data_line_without_space_is_parsed() {
        let data = "event: message_start\ndata:{\"message\":{\"id\":\"m1\"}}\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let parsed = reader.next_event().await.expect("reader must not err");
        let (_, payload) = parsed.expect("the event must be delivered");
        assert!(
            payload.is_some(),
            "spec-legal 'data:' line must carry the payload, not be dropped"
        );
    }

    async fn captured_effort_body(
        default: Option<crate::structured::ThinkingEffort>,
        options: crate::structured::RequestOptions,
        max_tokens: u32,
    ) -> String {
        captured_effort_body_serving("", default, options, max_tokens).await
    }

    /// As [`captured_effort_body`], on an endpoint that answers the thinking
    /// probe with `disclosure` (empty for an endpoint with no Models API).
    async fn captured_effort_body_serving(
        disclosure: &str,
        default: Option<crate::structured::ThinkingEffort>,
        options: crate::structured::RequestOptions,
        max_tokens: u32,
    ) -> String {
        use futures::StreamExt;

        let (addr, server) = capture_after_answering_probe(disclosure).await;

        let mut builder = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_max_tokens(max_tokens);
        if let Some(effort) = default {
            builder = builder.thinking_effort(effort);
        }
        let client = builder.build().unwrap();
        let mut stream =
            client.stream_messages_with_options(&crate::api::StreamRequest::new(vec![]), options);
        let _ = stream.next().await;
        server.await.unwrap()
    }

    #[tokio::test]
    async fn a_client_effort_default_is_overridden_per_request() {
        let default_alone = captured_effort_body(
            Some(crate::structured::ThinkingEffort::Medium),
            crate::structured::RequestOptions::new(),
            8192,
        )
        .await;
        assert!(
            default_alone.contains("\"budget_tokens\":8192"),
            "an unset per-request effort leaves the client default budget on the wire: {default_alone}"
        );
        let overridden = captured_effort_body(
            Some(crate::structured::ThinkingEffort::Medium),
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            8192,
        )
        .await;
        assert!(
            overridden.contains("\"budget_tokens\":16384")
                && !overridden.contains("\"budget_tokens\":8192"),
            "the per-request effort wins over the client default budget: {overridden}"
        );
        let no_default = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            8192,
        )
        .await;
        assert!(
            no_default.contains("\"budget_tokens\":16384"),
            "a request effort reaches the wire without any client default: {no_default}"
        );
    }

    #[tokio::test]
    async fn the_wider_rungs_reach_the_anthropic_wire() {
        let minimal = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::Minimal),
            8192,
        )
        .await;
        assert!(
            minimal.contains("\"budget_tokens\":1024"),
            "Minimal budgets the API's floor: {minimal}"
        );
        let xhigh = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::Xhigh),
            8192,
        )
        .await;
        assert!(
            xhigh.contains("\"budget_tokens\":24576"),
            "Xhigh budgets between High and Max: {xhigh}"
        );
        let off = captured_effort_body(
            Some(crate::structured::ThinkingEffort::High),
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::None),
            8192,
        )
        .await;
        assert!(
            off.contains("\"type\":\"disabled\"")
                && !off.contains("budget_tokens")
                && !off.contains("\"type\":\"enabled\""),
            "the None rung overrides a client default with the API's explicit off: {off}"
        );
    }

    #[tokio::test]
    async fn the_off_rung_says_disabled_rather_than_omitting_the_block() {
        let off = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::None),
            8192,
        )
        .await;
        assert!(
            off.contains("\"thinking\":{\"type\":\"disabled\"}"),
            "omitting `thinking` reads as adaptive thinking on a model that thinks by default, so the off rung must state it: {off}"
        );
        let unset =
            captured_effort_body(None, crate::structured::RequestOptions::new(), 8192).await;
        assert!(
            !unset.contains("\"thinking\""),
            "an unset effort still sends no thinking field at all: {unset}"
        );
    }

    #[tokio::test]
    async fn an_adaptive_model_gets_output_config_effort_instead_of_a_budget() {
        let body = captured_effort_body_serving(
            ADAPTIVE_DISCLOSURE,
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            1024,
        )
        .await;
        assert!(
            body.contains("\"output_config\":{\"effort\":\"high\"}")
                && !body.contains("\"thinking\"")
                && !body.contains("budget_tokens"),
            "a model that refuses a thinking budget takes the rung as output_config.effort, thinking left at its default: {body}"
        );
        assert!(
            body.contains("\"max_tokens\":1024"),
            "the headroom raise made for a budget this model cannot take is undone: {body}"
        );
    }

    #[tokio::test]
    async fn the_minimal_rung_names_itself_on_the_effort_ladder() {
        let body = captured_effort_body_serving(
            ADAPTIVE_DISCLOSURE,
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::Minimal),
            8192,
        )
        .await;
        assert!(
            body.contains("\"effort\":\"minimal\""),
            "the rung keeps its own label where Anthropic's ladder has no such level, so the API refuses the request rather than the ladder clamping the rung: {body}"
        );
    }

    #[tokio::test]
    async fn a_model_that_refuses_disabled_is_turned_off_at_its_lowest_setting() {
        let body = captured_effort_body_serving(
            BETWEEN_TOOLS_DISCLOSURE,
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::None),
            8192,
        )
        .await;
        assert!(
            body.contains("\"thinking\":{\"type\":\"between_tools\"}")
                && !body.contains("disabled"),
            "the off rung takes the lowest setting the model discloses, not the spelling it refuses: {body}"
        );
    }

    #[tokio::test]
    async fn a_model_with_no_off_level_refuses_the_off_rung_before_sending() {
        use futures::StreamExt;

        let (addr, seen, task) = recording_listener(NO_OFF_DISCLOSURE).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::None),
        );
        let outcome = stream
            .next()
            .await
            .expect("a model with no off level answers with an error event");
        task.abort();
        let text = outcome
            .expect_err("the refusal arrives as an error, not as a stream of events")
            .to_string();
        assert!(
            text.contains("thinking cannot be turned off")
                && text.contains("claude-sonnet-4-20250514"),
            "the refusal names why and which model, where a raw 400 about thinking.type would say neither: {text}"
        );
        let asked = crate::error::recover_guard(seen.lock()).clone();
        assert!(
            !asked.iter().any(|request| request.starts_with("POST")),
            "a rung the model cannot honour never reaches the Messages API: {asked:?}"
        );
    }

    #[tokio::test]
    async fn a_model_that_refuses_disabled_without_an_off_alternative_refuses() {
        use futures::StreamExt;

        // The disclosure answers one of the two off questions: `disabled` is
        // refused and nothing else is mentioned. Neither `disabled` (a known
        // 400) nor an unmentioned alternative (a guess) is a defensible send,
        // so the rung is refused with the model named.
        let (addr, seen, task) = recording_listener(
            r#"{"capabilities":{"thinking":{"types":{"disabled":{"supported":false}}}}}"#,
        )
        .await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::None),
        );
        let outcome = stream
            .next()
            .await
            .expect("a refusal arrives as an error event");
        task.abort();
        let text = outcome
            .expect_err("the refusal is an error, not an un-thought answer")
            .to_string();
        assert!(
            text.contains("disclosed no thinking setting that turns it off"),
            "the refusal says what the endpoint left unanswered: {text}"
        );
        let asked = crate::error::recover_guard(seen.lock()).clone();
        assert!(
            !asked.iter().any(|request| request.starts_with("POST")),
            "the request is refused before the Messages API sees a spelling it rejects: {asked:?}"
        );
    }

    #[tokio::test]
    async fn an_endpoint_without_a_models_api_keeps_the_budget_ladder() {
        let body = captured_effort_body_serving(
            "",
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            1024,
        )
        .await;
        assert!(
            body.contains("\"budget_tokens\":16384") && !body.contains("output_config"),
            "an endpoint that discloses nothing keeps the ladder's own mapping, so a gateway costs a caller nothing: {body}"
        );
    }

    #[tokio::test]
    async fn the_probe_asks_the_models_api_once_with_the_client_credential() {
        use futures::StreamExt;

        let (addr, seen, task) = recording_listener(ADAPTIVE_DISCLOSURE).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..2 {
            let mut stream = client.stream_messages_with_options(
                &crate::api::StreamRequest::new(vec![]),
                crate::structured::RequestOptions::default(),
            );
            let _ = stream.next().await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probes: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("GET /v1/models/"))
            .collect();
        assert_eq!(
            probes.len(),
            1,
            "the disclosure is read once per model, not once per request: {asked:?}"
        );
        assert!(
            probes[0].contains("/v1/models/claude-sonnet-4-20250514")
                && probes[0].contains("x-api-key: k"),
            "the probe addresses the client's own model with the client's own credential: {}",
            probes[0]
        );
    }

    #[tokio::test]
    async fn a_model_change_rereads_the_disclosure() {
        use futures::StreamExt;

        let (addr, seen, task) = recording_listener(ADAPTIVE_DISCLOSURE).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        let mut first = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            crate::structured::RequestOptions::default(),
        );
        let _ = first.next().await;
        drop(first);
        assert!(
            client.set_model("claude-opus-5"),
            "the hot-swap seam accepts a real model name"
        );
        let mut second = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            crate::structured::RequestOptions::default(),
        );
        let _ = second.next().await;
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let models: Vec<&str> = asked
            .iter()
            .filter_map(|request| request.split_whitespace().nth(1))
            .filter(|path| path.starts_with("/v1/models/"))
            .collect();
        assert_eq!(
            models,
            vec![
                "/v1/models/claude-sonnet-4-20250514",
                "/v1/models/claude-opus-5"
            ],
            "a swapped model reads its own disclosure instead of inheriting the previous one: {asked:?}"
        );
    }

    #[tokio::test]
    async fn an_effort_less_request_never_probes_the_models_api() {
        use futures::StreamExt;

        let (addr, seen, task) = recording_listener(ADAPTIVE_DISCLOSURE).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let mut stream = client.stream_messages_with_options(
            &crate::api::StreamRequest::new(vec![]),
            crate::structured::RequestOptions::default(),
        );
        let _ = stream.next().await;
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        assert!(
            asked.iter().all(|request| !request.starts_with("GET")),
            "with no effort resolved a disclosure could change nothing, so the Models API is not consulted: {asked:?}"
        );
    }

    #[tokio::test]
    async fn a_transient_probe_failure_is_asked_again_on_the_next_request() {
        let (addr, seen, task) = recording_listener_answering(ADAPTIVE_DISCLOSURE, 503).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..2 {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probes: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("GET /v1/models/"))
            .collect();
        assert_eq!(
            probes.len(),
            2,
            "an endpoint that was unavailable said nothing, so it is asked again; a filed silence would keep sending the budget ladder to a model that refuses it: {asked:?}"
        );
    }

    #[tokio::test]
    async fn a_probe_that_fails_once_recovers_on_the_next_turn() {
        let (addr, seen, task) = recording_listener_failing_probe(ADAPTIVE_DISCLOSURE, 1).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_max_tokens(1024)
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..2 {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let bodies: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("POST"))
            .collect();
        assert_eq!(bodies.len(), 2, "both turns reached the wire: {asked:?}");
        assert!(
            bodies[0].contains("\"budget_tokens\":16384"),
            "the turn whose probe failed takes the ladder's mapping, the documented answer for an endpoint that disclosed nothing: {}",
            bodies[0]
        );
        assert!(
            bodies[1].contains("\"output_config\":{\"effort\":\"high\"}")
                && bodies[1].contains("\"max_tokens\":1024"),
            "the next turn reads the disclosure the endpoint did give and takes the effort path with its configured max_tokens back — the 503 was not filed as an answer: {}",
            bodies[1]
        );
    }

    #[tokio::test]
    async fn a_definitive_404_is_filed_so_the_endpoint_is_asked_once() {
        let (addr, seen, task) = recording_listener("").await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..2 {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probes: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("GET /v1/models/"))
            .collect();
        assert_eq!(
            probes.len(),
            1,
            "an endpoint that answered `404` answered definitively — no Models API at this base URL — and is not asked again per request: {asked:?}"
        );
    }

    #[tokio::test]
    async fn a_request_naming_another_model_caches_its_own_disclosure() {
        let (addr, seen, task) = recording_listener(ADAPTIVE_DISCLOSURE).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        let override_model = crate::structured::RequestOptions::new()
            .with_model("claude-opus-5")
            .with_effort(crate::structured::ThinkingEffort::High);
        // Two turns each, alternating: the shape a fallback chain makes, where
        // a single cache slot made every turn after the first pay a probe.
        for _ in 0..2 {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
            drive_request(&client, override_model.clone()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probed: Vec<&str> = asked
            .iter()
            .filter_map(|request| request.split_whitespace().nth(1))
            .filter(|path| path.starts_with("/v1/models/"))
            .collect();
        assert_eq!(
            probed.len(),
            2,
            "each model is asked once, whichever channel named it — the client's own or a per-request override: {asked:?}"
        );
        assert!(
            probed.contains(&"/v1/models/claude-sonnet-4-20250514")
                && probed.contains(&"/v1/models/claude-opus-5"),
            "both models got their own disclosure read: {probed:?}"
        );
    }

    #[tokio::test]
    async fn an_endpoint_that_never_answers_stops_being_probed() {
        let (addr, seen, task) = recording_listener_answering(ADAPTIVE_DISCLOSURE, 503).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..5 {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probes: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("GET /v1/models/"))
            .collect();
        assert_eq!(
            probes.len(),
            usize::from(CAPABILITY_PROBE_MISS_LIMIT),
            "a route that fails every probe three times running has said it has nothing to \
             disclose; asking a fourth time puts another probe — or its full timeout — in front \
             of every turn of the run: {asked:?}"
        );
    }

    #[test]
    fn the_capability_cache_renews_its_give_up_verdict_on_a_bound() {
        // The bound the give-up verdict has to live with. Three misses hand the
        // run to the ladder's mapping, every following turn reads that verdict,
        // and the route is asked again on the renewal cadence: five probes in
        // sixty-four requests, not sixty-four probes and not one probe forever.
        let mut cache = CapabilityCache::default();
        let mut asked = 0_usize;
        for _ in 0..64 {
            if cache.verdict("m").is_none() {
                asked = asked.saturating_add(1);
                cache.note_inconclusive("m");
            }
        }
        assert_eq!(
            asked,
            5,
            "the initial three, then one renewal every {} served turns",
            usize::from(CAPABILITY_PROBE_REPROBE_AFTER)
                .saturating_sub(usize::from(CAPABILITY_PROBE_MISS_LIMIT))
        );
    }

    #[test]
    fn a_recovered_route_takes_over_the_given_up_verdict() {
        // And the recovery itself. Once the renewal slot comes round, whatever
        // the endpoint finally says replaces the inference — filed, this time,
        // so the asking ends there as it always has for a real disclosure.
        let mut cache = CapabilityCache::default();
        for _ in 0..CAPABILITY_PROBE_MISS_LIMIT {
            assert!(
                cache.verdict("m").is_none(),
                "under the limit every request asks"
            );
            cache.note_inconclusive("m");
        }
        assert!(
            matches!(cache.verdict("m"), Some(CachedDisclosure::Undisclosed)),
            "past the limit the run reads the verdict"
        );
        let mut turns = 1_usize;
        loop {
            if cache.verdict("m").is_none() {
                break;
            }
            turns = turns.saturating_add(1);
        }
        assert!(
            turns <= usize::from(CAPABILITY_PROBE_REPROBE_AFTER),
            "the verdict expires on the bound rather than standing: it held for {turns} turns"
        );
        cache.insert(
            "m",
            CachedDisclosure::Known(ThinkingCapabilities {
                disabled: Some(false),
                between_tools: Some(true),
                enabled: Some(false),
                adaptive: Some(true),
            }),
        );
        assert!(
            matches!(
                cache.verdict("m"),
                Some(CachedDisclosure::Known(ThinkingCapabilities {
                    adaptive: Some(true),
                    ..
                }))
            ),
            "what the endpoint said outranks what this client inferred"
        );
    }

    #[tokio::test]
    async fn a_route_that_recovers_after_the_give_up_verdict_reaches_the_wire() {
        // The cost of letting the verdict stand. Three misses settle the client
        // on the ladder's mapping; the metadata route comes back afterwards. A
        // client that filed its own inference would go on sending the budget
        // spelling this model refuses — a 400 on every turn, for the life of the
        // process. The renewal has to reach the wire, not just the cache.
        let (addr, seen, task) = recording_listener_failing_probe(
            ADAPTIVE_DISCLOSURE,
            usize::from(CAPABILITY_PROBE_MISS_LIMIT),
        )
        .await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_max_tokens(1024)
            .thinking_effort(crate::structured::ThinkingEffort::High)
            .build()
            .unwrap();
        for _ in 0..=usize::from(CAPABILITY_PROBE_REPROBE_AFTER) {
            drive_request(&client, crate::structured::RequestOptions::default()).await;
        }
        task.abort();
        let asked = crate::error::recover_guard(seen.lock()).clone();
        let probes = asked
            .iter()
            .filter(|request| request.starts_with("GET /v1/models/"))
            .count();
        assert_eq!(
            probes, 4,
            "three misses, then the verdict serves the turns until the renewal asks once more: {asked:?}"
        );
        let bodies: Vec<&String> = asked
            .iter()
            .filter(|request| request.starts_with("POST"))
            .collect();
        assert!(
            bodies[0].contains("\"budget_tokens\":16384")
                && bodies[2].contains("\"budget_tokens\":16384"),
            "the outage itself runs on the ladder's mapping: {}",
            bodies[0]
        );
        let last = bodies.last().expect("the run reached the wire");
        assert!(
            last.contains("\"output_config\":{\"effort\":\"high\"}")
                && last.contains("\"max_tokens\":1024"),
            "the recovered disclosure takes the effort path with the configured max_tokens back; \
             a filed give-up verdict would still be sending a budget this model refuses: {last}"
        );
    }

    #[test]
    fn a_conclusive_answer_clears_the_miss_streak() {
        let mut cache = CapabilityCache::default();
        cache.note_inconclusive("m");
        cache.note_inconclusive("m");
        cache.insert(
            "m",
            CachedDisclosure::Known(ThinkingCapabilities {
                disabled: Some(true),
                between_tools: None,
                enabled: None,
                adaptive: None,
            }),
        );
        assert_eq!(
            cache.entry("m").misses,
            0,
            "an answer outranks the failures that preceded it"
        );
        cache.note_inconclusive("m");
        cache.note_inconclusive("m");
        assert!(
            matches!(
                cache.verdict("m"),
                Some(CachedDisclosure::Known(ThinkingCapabilities {
                    disabled: Some(true),
                    ..
                }))
            ),
            "two later misses cannot stack onto the pre-answer streak and overwrite what the \
             endpoint actually disclosed"
        );
        cache.note_inconclusive("m");
        assert!(
            matches!(cache.verdict("m"), Some(CachedDisclosure::Known(_))),
            "a model that answered is never demoted to undisclosed by later blips"
        );
    }

    #[tokio::test]
    async fn a_high_effort_enlarges_max_tokens_for_headroom() {
        let raised = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            1024,
        )
        .await;
        assert!(
            raised.contains("\"max_tokens\":16385") && raised.contains("\"budget_tokens\":16384"),
            "a budget above max_tokens raises max_tokens to budget + 1 so the body is never self-rejecting: {raised}"
        );
        let kept = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::High),
            32768,
        )
        .await;
        assert!(
            kept.contains("\"max_tokens\":32768"),
            "a max_tokens already above budget + 1 is left untouched: {kept}"
        );
    }

    #[tokio::test]
    async fn the_ladder_s_outer_rungs_reach_the_anthropic_wire() {
        let low = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::Low),
            8192,
        )
        .await;
        assert!(
            low.contains("\"budget_tokens\":2048"),
            "the Low rung maps to the ladder's 2048 budget: {low}"
        );
        let max = captured_effort_body(
            None,
            crate::structured::RequestOptions::new()
                .with_effort(crate::structured::ThinkingEffort::Max),
            8192,
        )
        .await;
        assert!(
            max.contains("\"budget_tokens\":32768") && max.contains("\"max_tokens\":32769"),
            "Max maps to the ladder's 32768 budget and the headroom rule raises max_tokens to budget + 1: {max}"
        );
    }

    #[tokio::test]
    async fn request_model_override_replaces_the_body_model_on_stream() {
        use futures::StreamExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = sock.read(&mut buf).await.unwrap();
            let head = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
            drop(sock.write_all(head.as_bytes()).await);
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });

        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let options = crate::structured::RequestOptions::new().with_model("override-model");
        let mut stream =
            client.stream_messages_with_options(&crate::api::StreamRequest::new(vec![]), options);
        let _ = stream.next().await;
        let request = server.await.unwrap();
        assert!(
            request.contains("\"model\":\"override-model\""),
            "the streaming path must honor the per-request model override: {request}"
        );
    }

    #[tokio::test]
    async fn request_model_override_replaces_the_body_model() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = sock.read(&mut buf).await.unwrap();
            let head = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
            drop(sock.write_all(head.as_bytes()).await);
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });

        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let options = crate::structured::RequestOptions::new().with_model("override-model");
        drop(
            client
                .create_message_with_options(&crate::api::StreamRequest::new(vec![]), options)
                .await,
        );
        let request = server.await.unwrap();
        assert!(
            request.contains("\"model\":\"override-model\""),
            "the per-request override must replace the body's model field: {request}"
        );
    }

    #[tokio::test]
    async fn sse_multi_line_data_joins_spaced_and_compact_forms() {
        let data = "event: message_start\ndata: {\"a\":\ndata:1}\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let parsed = reader.next_event().await.expect("reader must not err");
        let (_, payload) = parsed.expect("the event must be delivered");
        let payload = payload.expect("the joined payload must parse");
        assert_eq!(
            payload,
            serde_json::json!({"a": 1}),
            "data lines of either spacing form concatenate into one payload"
        );
    }

    #[test]
    fn emitter_delta_usage_revision_wins_over_start_latch() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 40, "output_tokens": 15}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (40, 15),
            "a server-tools turn whose delta revises the counts upward must not be under-reported"
        );
    }

    #[test]
    fn emitter_message_delta_without_usage_still_reports_latched_tokens() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("the latch alone must still produce a usage report");
        assert_eq!((usage.input_tokens, usage.output_tokens), (25, 1));
    }

    #[test]
    fn emitter_duplicate_message_start_keeps_first_usage_latch() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        })));
        em.on_message_start(Some(&serde_json::json!({
            "message": {
                "id": "msg_1",
                "model": "claude-3",
                "usage": {"input_tokens": 99, "output_tokens": 99}
            }
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        let usage = usage.expect("MessageDelta must carry usage");
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (25, 15),
            "only the first message_start latches; a replayed one must not overwrite it"
        );
    }

    #[test]
    fn emitter_message_delta_without_usage_omits_usage() {
        let mut em = StreamEmitter::default();
        em.on_message_start(Some(&serde_json::json!({
            "message": {"id": "msg_1", "model": "claude-3"}
        })));
        em.drain();
        em.on_message_delta(Some(serde_json::json!({
            "delta": {"stop_reason": "end_turn"}
        })));
        let events = em.drain();
        let usage = events.iter().find_map(|e| match e {
            StreamEvent::MessageDelta(MessageDelta { usage, .. }) => *usage,
            _ => None,
        });
        assert_eq!(
            usage, None,
            "with no usage on either event, the delta carries no usage, matching the non-streaming path"
        );
    }

    #[test]
    fn extract_usage_object_defaults_malformed_counts_to_zero() {
        let malformed = serde_json::json!({
            "input_tokens": "many",
            "output_tokens": -5
        });
        assert_eq!(
            extract_usage_object(&malformed),
            None,
            "non-numeric counts default to zero, and an all-zero report is None"
        );
        let partial = serde_json::json!({"input_tokens": 12});
        assert_eq!(
            extract_usage_object(&partial),
            Some(Usage::new(12, 0)),
            "a missing output count defaults to zero without dropping the input report"
        );
    }

    /// Serve one fixed SSE response body on a local TCP listener.
    ///
    /// Binds an ephemeral port, accepts a single connection, and answers
    /// with a `text/event-stream` response carrying `body`. Awaiting the
    /// returned handle confirms the server finished writing.
    async fn serve_sse(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            drop(sock.read(&mut buf).await);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            drop(sock.write_all(head.as_bytes()).await);
            drop(sock.write_all(body.as_bytes()).await);
            drop(sock.flush().await);
        });
        (format!("http://{addr}"), server)
    }

    /// Drain a message stream to `(saw_message_start, terminal_usage)`.
    ///
    /// A helper for the wire tests: collects a stream to the pair the emitter invariants are asserted against.
    async fn collect_stream_telemetry(
        mut stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send>>,
    ) -> (bool, Option<Usage>) {
        use futures::StreamExt;

        let mut usage = None;
        let mut saw_start = false;
        while let Some(event) = stream.next().await {
            match event.expect("stream must not err") {
                StreamEvent::MessageStart(_) => saw_start = true,
                StreamEvent::MessageDelta(MessageDelta { usage: u, .. }) => usage = u,
                _ => {}
            }
        }
        (saw_start, usage)
    }

    #[tokio::test]
    async fn streamed_turn_reports_latched_input_tokens_over_the_wire() {
        let body = concat!(
            "event: message_start\n",
            "data:{\"message\":{\"id\":\"m1\",\"model\":\"claude-3\",",
            "\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            "event: message_delta\n",
            "data:{\"delta\":{\"stop_reason\":\"end_turn\"},",
            "\"usage\":{\"output_tokens\":15}}\n\n",
            "event: message_stop\n",
            "data:{}\n\n",
        );
        let (url, server) = serve_sse(body).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(url)
            .build()
            .unwrap();
        let (saw_start, usage) = collect_stream_telemetry(
            client.stream_messages(&crate::api::StreamRequest::new(vec![Message::user("hi")])),
        )
        .await;
        server.await.unwrap();

        assert!(saw_start, "the stream must deliver its message_start");
        let usage = usage.expect("the terminal MessageDelta must carry usage");
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (25, 15),
            "a compact-data-line wire must yield the latched input tokens and the delta's output tokens"
        );
    }

    #[tokio::test]
    async fn sse_event_line_without_space_dispatches_the_event() {
        let data = "event:message_start\ndata: {\"message\":{\"id\":\"m1\"}}\n\n";
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::iter(vec![Ok::<bytes::Bytes, ApiError>(
                data.to_string().into(),
            )])),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let parsed = reader.next_event().await.expect("reader must not err");
        let (event_type, payload) = parsed.expect("the event must be delivered");
        assert_eq!(
            event_type, "message_start",
            "spec-legal 'event:' line must dispatch under its event type, not as unknown"
        );
        assert!(
            payload.is_some(),
            "the paired data payload must be delivered"
        );
    }

    #[tokio::test]
    async fn streamed_turn_dispatches_compact_event_lines_over_the_wire() {
        let body = concat!(
            "event:message_start\n",
            "data: {\"message\":{\"id\":\"m1\",\"model\":\"claude-3\",",
            "\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            "event:message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"},",
            "\"usage\":{\"output_tokens\":15}}\n\n",
            "event:message_stop\n",
            "data: {}\n\n",
        );
        let (url, server) = serve_sse(body).await;
        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(url)
            .build()
            .unwrap();
        let (saw_start, usage) = collect_stream_telemetry(
            client.stream_messages(&crate::api::StreamRequest::new(vec![Message::user("hi")])),
        )
        .await;
        server.await.unwrap();

        assert!(
            saw_start,
            "compact event lines must dispatch their events, not silence the whole stream"
        );
        let usage = usage.expect("the terminal MessageDelta must carry usage");
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (25, 15),
            "a compact-event-line wire must report the same usage as the spaced form"
        );
    }

    #[test]
    fn listing_entries_carry_display_names_stamps_and_curation() {
        let doc = serde_json::json!({
            "data": [
                {
                    "type": "model",
                    "id": "claude-sonnet-4-5",
                    "display_name": "Claude Sonnet 4.5",
                    "created_at": "2025-09-29T00:00:00Z"
                },
                { "type": "model", "id": "mystery-model" }
            ],
            "has_more": false
        });
        let models = AnthropicClient::parse_models(&doc).expect("listing parses");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "claude-sonnet-4-5");
        assert_eq!(
            models[0].display_name.as_deref(),
            Some("Claude Sonnet 4.5"),
            "the human name is what a picker displays"
        );
        assert_eq!(
            models[0].created.as_deref(),
            Some("2025-09-29T00:00:00Z"),
            "the RFC 3339 stamp rides verbatim"
        );
        assert_eq!(
            models[0].metadata.context_window,
            Some(200_000),
            "the endpoint discloses no windows — the curated family value fills in"
        );
        assert_eq!(
            models[0].metadata.capabilities.structured_output,
            Some(false),
            "and the curated capability claims ride with it"
        );
        assert_eq!(
            models[1].metadata,
            crate::api::probe::ModelMetadata::unreported(),
            "an unknown id answers nobody said, never a plausible zero"
        );
    }

    #[tokio::test]
    async fn the_anthropic_listing_sends_the_version_pair_and_maps_models() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let body = r#"{"data":[{"type":"model","id":"claude-sonnet-4-5","display_name":"Claude Sonnet 4.5","created_at":"2025-09-29T00:00:00Z"}],"has_more":false}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            drop(sock.write_all(head.as_bytes()).await);
            drop(sock.write_all(body.as_bytes()).await);
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });

        let client = AnthropicClient::builder()
            .with_api_key("k-secret")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let models = client
            .list_models()
            .await
            .expect("the listing arrives through the credential pair");
        assert_eq!(models[0].id, "claude-sonnet-4-5");
        assert_eq!(models[0].display_name.as_deref(), Some("Claude Sonnet 4.5"));

        let request = server.await.unwrap();
        assert!(
            request.contains("GET /v1/models"),
            "the models route, not the messages route: {request}"
        );
        assert!(
            request.contains("x-api-key: k-secret"),
            "the credential rides the probe exactly as on a Messages call: {request}"
        );
        assert!(
            request.contains("anthropic-version: "),
            "the endpoint refuses the listing without its version header: {request}"
        );
    }

    #[tokio::test]
    async fn the_anthropic_listing_rejects_an_oversized_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            drop(sock.read(&mut buf).await);
            // The padding pushes a still-valid catalog past the shared
            // ceiling, so refusing it is the only correct answer.
            let oversized = "y".repeat(super::super::MAX_RESPONSE_BODY);
            let body = format!(
                r#"{{"data":[{{"type":"model","id":"oversized","display_name":"{oversized}"}}],"has_more":false}}"#
            );
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            drop(sock.write_all(head.as_bytes()).await);
            drop(sock.write_all(body.as_bytes()).await);
        });

        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let err = client
            .list_models()
            .await
            .expect_err("a catalog past the shared ceiling must fail, not be buffered into memory");
        assert!(
            err.to_string().contains("too large"),
            "the listing must reject through the shared bounded-body reader: {err}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn the_anthropic_listing_answers_page_one_and_stops() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn({
            async move {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _n = sock.read(&mut buf).await.unwrap_or(0);
                let body = r#"{"data":[{"type":"model","id":"claude-sonnet-4-5"},{"type":"model","id":"claude-haiku-4-5"}],"has_more":true,"next_page":"cursor-page-2"}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                drop(sock.write_all(head.as_bytes()).await);
                drop(sock.write_all(body.as_bytes()).await);
                drop(sock);
                // Did a second request (page two) ever arrive?
                tokio::time::timeout(std::time::Duration::from_millis(250), listener.accept())
                    .await
                    .is_ok()
            }
        });

        let client = AnthropicClient::builder()
            .with_api_key("k-secret")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let models = client
            .list_models()
            .await
            .expect("a further page changes nothing about this page's answer");
        assert_eq!(
            models.len(),
            2,
            "the first page is served whole, markers and all"
        );
        assert!(
            !server.await.unwrap(),
            "page-1-only is the documented bound of the listing surface — \
             a `has_more` marker must not silently become a second request"
        );
    }

    #[tokio::test]
    async fn the_anthropic_ping_answers_the_models_route() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for body in [
                r#"{"data":[],"has_more":false}"#,
                r#"{"type":"error","error":{"type":"authentication_error","message":"bad"}}"#,
            ] {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 1024];
                let _n = sock.read(&mut buf).await.unwrap_or(0);
                let status = if body.contains("error") {
                    "403 Forbidden"
                } else {
                    "200 OK"
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                drop(sock.write_all(head.as_bytes()).await);
                drop(sock.write_all(body.as_bytes()).await);
            }
        });

        let client = AnthropicClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .unwrap();
        let report = client
            .ping()
            .await
            .expect("a serving models route answers the ping");
        assert_eq!(report.request_id, None, "this answer sent no headers");

        let err = client
            .ping()
            .await
            .expect_err("the rejection must surface as the classified failure");
        assert_eq!(
            crate::api::probe::ProbeClass::from_error(&err),
            crate::api::probe::ProbeClass::Auth,
            "403 reads as the credential problem it is: {err}"
        );
        server.await.unwrap();
    }
}
