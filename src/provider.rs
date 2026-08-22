//! LLM provider clients.
//!
//! This module provides ready-to-use [`ApiClient`](crate::api::ApiClient)
//! implementations for common LLM providers. Each provider is behind a
//! feature flag so you only compile what you need.
//!
//! # Feature Flags
//!
//! | Provider     | Feature    | API Format              |
//! |--------------|------------|-------------------------|
//! | OpenAI       | `openai`   | OpenAI Chat Completions |
//! | Anthropic    | `anthropic`| Anthropic Messages      |
//! | Gemini       | `gemini`   | Google Gemini           |
//! | Ollama       | `ollama`   | OpenAI-compatible       |
//! | `DeepSeek`   | `deepseek` | OpenAI-compatible       |
//! | `Grok` (xAI) | `grok`     | OpenAI-compatible       |
//! | `Z.ai`       | `zai`      | Anthropic Messages      |
//! | Self-hosted  | any        | OpenAI-compatible       |
//!
//! Any provider that exposes an OpenAI-compatible Chat Completions API
//! can use [`OpenAiClient`] with a custom base URL. The convenience
//! constructors below pre-configure the correct endpoints.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::provider;
//! use loopctl::engine::BareLoop;
//! use loopctl::engine::RunConfig;
//! use loopctl::engine::core::Loop;
//!
//! // OpenAI:
//! let client = provider::OpenAiClient::from_env()?;
//!
//! // DeepSeek:
//! let client = provider::deepseek()?;
//!
//! // Anthropic:
//! let client = provider::AnthropicClient::from_env()?;
//!
//! // Gemini:
//! let client = provider::GeminiClient::from_env()?;
//!
//! // Ollama (local):
//! let client = provider::ollama("llama3")?;
//!
//! // Self-hosted (vLLM, LM Studio, etc.):
//! let client = provider::self_hosted("http://localhost:8080/v1", "my-model")?;
//!
//! let agent = BareLoop::new(
//!     std::sync::Arc::new(client),
//!     tool_registry,
//!     config,
//! );
//! let result = agent.run("Hello!", &RunConfig::default()).await?;
//! ```

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
use crate::api::error::ApiError;
#[cfg(any(feature = "anthropic", feature = "gemini"))]
use crate::message::{MessagePart, Role};
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
use futures::StreamExt;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
use std::time::Duration;

// SSE line-framing shared by every streaming provider. Each provider keeps
// its own event-extraction logic (`next_data` / `next_event`); the struct,
// `from_response`, `take_line`, and the buffer-overflow guard live here.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod sse;

/// Maximum accepted response body size (10 MB).
///
/// Guards against unbounded memory growth from a misbehaving or hostile
/// provider that returns a very large non-streaming response. Enforced
/// *before* the body is fully materialized — see
/// [`read_bounded_body`](crate::provider::read_bounded_body).
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(super) const MAX_RESPONSE_BODY: usize = 10 * 1024 * 1024;

/// Read a response body, rejecting it before peak memory is exceeded.
///
/// Shared guard used by every provider's non-streaming path. Two checks
/// bound memory:
///
/// 1. **Pre-read (`Content-Length`)** — when the header is present and
///    exceeds [`MAX_RESPONSE_BODY`], the body is rejected without reading a
///    single byte. A hostile provider advertising a huge body never allocates
///    it.
/// 2. **Streaming cap** — for responses without `Content-Length` (chunked
///    transfer), the body is read chunk by chunk and the read aborts the
///    moment the running total crosses [`MAX_RESPONSE_BODY`], so peak memory
///    never exceeds the cap by more than one chunk.
///
/// Replaces the old `resp.bytes().await` + post-hoc length check, which
/// materialized the full body before the guard could fire.
///
/// # Errors
///
/// Returns [`ApiError`] when the body exceeds [`MAX_RESPONSE_BODY`] (either
/// via the header pre-check or the streaming cap), or on a transport error
/// reading the body.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(super) async fn read_bounded_body(resp: reqwest::Response) -> Result<bytes::Bytes, ApiError> {
    if let Some(len) = resp.content_length()
        && usize::try_from(len).map_or(true, |n| n > MAX_RESPONSE_BODY)
    {
        return Err(ApiError::http(format!(
            "response body too large: declared {len} bytes (max {MAX_RESPONSE_BODY})"
        )));
    }
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| ApiError::http(format!("error reading response body: {e}")))?;
        buf.extend_from_slice(&chunk);
        if buf.len() > MAX_RESPONSE_BODY {
            return Err(ApiError::http(format!(
                "response body too large: streamed {} bytes (max {MAX_RESPONSE_BODY})",
                buf.len()
            )));
        }
    }
    Ok(buf.into())
}

/// Maximum error-diagnostic body retained from a non-success response (8 `KiB`).
///
/// Bounds both memory and network traffic when a provider answers a failed
/// request with a large body: the read stops as soon as this many bytes are
/// available, so a misbehaving endpoint cannot make the client materialize a
/// multi-gigabyte error page. The retained prefix is what error messages and
/// logs carry.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(super) const MAX_ERROR_BODY: usize = 8 * 1024;

/// Read the diagnostic body of an error response, capped at
/// [`MAX_ERROR_BODY`] bytes.
///
/// Two guards bound the transfer. When `Content-Length` is present and
/// exceeds the cap, the body is refused outright — the response is dropped
/// without reading a single body byte, closing the connection before the
/// server can send the bulk (kernel socket buffers would otherwise let a
/// misbehaving server write far past the cap even after the client stops
/// reading). Otherwise the body is streamed and the read stops as soon as
/// the cap is available, which also bounds chunked responses. Decode errors
/// are replaced lossily so a binary body still yields printable text for
/// logs; an oversized body yields an empty string — its status alone
/// classifies the error.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(super) async fn read_error_body(resp: reqwest::Response) -> String {
    if let Some(len) = resp.content_length()
        && usize::try_from(len).map_or(true, |n| n > MAX_ERROR_BODY)
    {
        return String::new();
    }
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < MAX_ERROR_BODY
        && let Some(chunk) = stream.next().await
    {
        match chunk {
            Ok(bytes) => buf.extend_from_slice(&bytes),
            Err(_) => break,
        }
    }
    let capped = buf.get(..MAX_ERROR_BODY).unwrap_or(&buf);
    String::from_utf8_lossy(capped).into_owned()
}

/// Classify a non-success HTTP response into the matching [`ApiError`]
/// variant.
///
/// The status alone picks the variant so the classification survives without
/// re-parsing message strings: 401 and 403 are authentication failures
/// ([`ApiError::Auth`], permanent), 429/503/529 are rate limits
/// ([`ApiError::RateLimit`], carrying the parsed `Retry-After` when the
/// server sent one), and everything else stays a status-tagged
/// [`ApiError::Http`]. The body text is preserved in the message for
/// diagnostics, prefixed with the status.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
fn classify_error_response(status: u16, body: String, retry_after: Option<Duration>) -> ApiError {
    match status {
        401 => ApiError::auth_invalid_key(format!("HTTP {status}: {body}")),
        403 => ApiError::auth(format!("HTTP {status}: {body}")),
        429 | 503 | 529 => ApiError::rate_limited(format!("HTTP {status}: {body}"), retry_after),
        _ => ApiError::http_with_status(status, body),
    }
}

/// Send a JSON POST and classify non-success responses into structured
/// [`ApiError`] variants.
///
/// The single HTTP-error construction site shared by the provider clients:
/// it sends the request with `headers` applied, and on a non-success status
/// it reads the `Retry-After` header while the response is still in hand,
/// reads the diagnostic body capped at [`MAX_ERROR_BODY`] bytes, and maps the
/// status via [`classify_error_response`]. Callers therefore get the right
/// variant — auth rejections as [`ApiError::Auth`], rate limits with the
/// server-advised delay as [`ApiError::RateLimit`] — without each provider
/// re-implementing (and drifting on) the same branches. Header values are
/// applied verbatim: callers mark credential headers sensitive
/// ([`HeaderValue::set_sensitive`](reqwest::header::HeaderValue::set_sensitive))
/// so they are redacted in debug output and never indexed into HTTP/2's
/// header-compression table.
///
/// # Errors
///
/// Returns [`ApiError::http`] when the request fails at the transport level,
/// or the classified variant for any non-success status.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub(super) async fn post_json_checked(
    client: &reqwest::Client,
    url: &str,
    headers: &[(reqwest::header::HeaderName, reqwest::header::HeaderValue)],
    body: &serde_json::Value,
) -> Result<reqwest::Response, ApiError> {
    let request = headers.iter().fold(client.post(url), |req, (name, value)| {
        req.header(name.clone(), value.clone())
    });
    let resp = request
        .json(body)
        .send()
        .await
        .map_err(|e| ApiError::http(e.to_string()))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(crate::api::error::parse_retry_after);
    let body_text = read_error_body(resp).await;
    Err(classify_error_response(
        status.as_u16(),
        body_text,
        retry_after,
    ))
}

/// Shared HTTP-client configuration embedded by every provider builder.
///
/// Holds the timeout, connection-pool, and TCP knobs that are identical
/// across [`OpenAiClient`](crate::provider::OpenAiClient),
/// [`AnthropicClient`](crate::provider::AnthropicClient), and
/// [`GeminiClient`](crate::provider::GeminiClient). Each provider builder
/// embeds this struct and delegates its HTTP-related setters to it, so the
/// pool/TCP documentation and construction logic lives in one place.
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
#[derive(Clone)]
pub(super) struct HttpClientConfig {
    /// The HTTP read timeout: the maximum gap between bytes of a response.
    ///
    /// Bounds *idleness*, not total duration — a healthy slow stream (a long
    /// SSE generation that keeps emitting events) runs as long as it keeps
    /// producing bytes, while a server that goes silent is aborted after this
    /// gap. This is deliberately not a total request timeout: a total cap at
    /// the HTTP layer would pre-empt the `StreamHandler`'s per-event and
    /// total-stream deadlines (and the engine's turn timeout), killing any
    /// generation longer than the cap. Defaults to 120 seconds.
    ///
    /// Ignored when an external client is supplied via
    /// [`with_http_client`](Self::with_http_client); configure it on that
    /// client instead.
    timeout: Duration,

    /// The TCP connection establishment timeout (including TLS handshake).
    ///
    /// Separate from the total timeout so a slow-connecting server can be
    /// detected faster than a slow-responding one. Defaults to 10 seconds.
    ///
    /// Ignored when an external client is supplied via
    /// [`with_http_client`](Self::with_http_client).
    connect_timeout: Duration,

    /// A pre-built, shared `reqwest::Client`, if injected via
    /// [`with_http_client`](Self::with_http_client).
    ///
    /// When set, the client's connection pool is shared with every other
    /// provider built from the same handle, and the pool/TCP knobs below
    /// (`pool_*`, `tcp_*`) are ignored — they are only applied when the
    /// builder constructs its own client. Configure timeouts on the injected
    /// client, not here.
    http: Option<reqwest::Client>,

    /// Maximum idle connections kept alive per host.
    ///
    /// `None` defers to reqwest's default (unlimited). Set to a small value
    /// (e.g. 1–4) for memory-constrained runners or workloads that make
    /// mostly serial requests to a single host.
    pool_max_idle_per_host: Option<usize>,

    /// How long an idle connection stays in the pool before being closed.
    ///
    /// `None` defers to reqwest's default (90s). Raise for long-idle
    /// interactive workloads to keep TLS sessions warm; lower for tight
    /// batch jobs to free file descriptors sooner.
    pool_idle_timeout: Option<Duration>,

    /// OS-level TCP keepalive interval.
    ///
    /// `None` disables TCP keepalive (reqwest default). Enable (~60s) if
    /// connections are silently dropped after idle periods (e.g. behind
    /// aggressive NATs or load balancers).
    tcp_keepalive: Option<Duration>,

    /// Whether to disable Nagle's algorithm (`TCP_NODELAY`).
    ///
    /// Defaults to `true` — SSE streaming emits many small packets, and
    /// Nagle's algorithm coalesces them, adding latency per delta.
    tcp_nodelay: bool,
}

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_mins(2),
            connect_timeout: Duration::from_secs(10),
            http: None,
            pool_max_idle_per_host: None,
            pool_idle_timeout: None,
            tcp_keepalive: None,
            tcp_nodelay: true,
        }
    }
}

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
impl HttpClientConfig {
    /// Set the read timeout (maximum gap between response bytes).
    ///
    /// Not a total request timeout — long streaming generations are bounded
    /// by the `StreamHandler`'s per-event/total deadlines, not by the HTTP
    /// layer. Ignored when an external client was supplied via
    /// [`with_http_client`](Self::with_http_client).
    #[must_use]
    pub(super) fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the TCP connection establishment timeout.
    ///
    /// Ignored when an external client was supplied.
    #[must_use]
    pub(super) fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Inject a pre-built, shared `reqwest::Client`.
    ///
    /// When set, the client's connection pool is shared with every other
    /// provider built from the same handle, and the pool/TCP knobs are
    /// ignored. Configure timeouts on the injected client, not here.
    #[must_use]
    pub(super) fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Set the maximum idle connections kept alive per host.
    ///
    /// Defaults to reqwest's built-in default (unlimited). Set to a small
    /// value (e.g. 1–4) for memory-constrained runners or workloads that
    /// make mostly serial requests to a single host.
    ///
    /// Ignored when an external client was supplied.
    #[must_use]
    pub(super) fn with_pool_max_idle_per_host(mut self, n: usize) -> Self {
        self.pool_max_idle_per_host = Some(n);
        self
    }

    /// Set how long an idle connection stays in the pool before being closed.
    ///
    /// Defaults to reqwest's built-in default (90s). Raise for long-idle
    /// interactive workloads to keep TLS sessions warm; lower for tight
    /// batch jobs to free file descriptors sooner.
    ///
    /// Ignored when an external client was supplied.
    #[must_use]
    pub(super) fn with_pool_idle_timeout(mut self, d: Duration) -> Self {
        self.pool_idle_timeout = Some(d);
        self
    }

    /// Set the OS-level TCP keepalive interval.
    ///
    /// Defaults to disabled (reqwest default). Enable (~60s) if connections
    /// are silently dropped after idle periods (e.g. behind aggressive NATs
    /// or load balancers).
    ///
    /// Ignored when an external client was supplied.
    #[must_use]
    pub(super) fn with_tcp_keepalive(mut self, d: Duration) -> Self {
        self.tcp_keepalive = Some(d);
        self
    }

    /// Control whether `TCP_NODELAY` is set on connections.
    ///
    /// Defaults to `true` — SSE streaming emits many small packets, and
    /// Nagle's algorithm coalesces them, adding latency per delta. Pass
    /// `false` to re-enable Nagle's algorithm (rarely needed). Ignored when
    /// an external client was supplied.
    #[must_use]
    pub(super) fn with_tcp_nodelay(mut self, enabled: bool) -> Self {
        self.tcp_nodelay = enabled;
        self
    }

    /// Build a `reqwest::Client` from this configuration.
    ///
    /// If an external client was supplied via
    /// [`with_http_client`](Self::with_http_client), it is returned verbatim. Otherwise
    /// a new client is constructed with a connect timeout, a read (idle-gap)
    /// timeout, pool knobs, and `tcp_nodelay(true)`. No total request
    /// timeout is set at this layer: generation-length budgets belong to the
    /// `StreamHandler` and the engine's turn timeout, and a total HTTP cap
    /// would abort healthy long streams.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if `reqwest::Client::builder().build()` fails.
    pub(super) fn build(self) -> Result<reqwest::Client, ApiError> {
        match self.http {
            Some(shared) => Ok(shared),
            None => reqwest::Client::builder()
                .read_timeout(self.timeout)
                .connect_timeout(self.connect_timeout)
                .tcp_nodelay(self.tcp_nodelay)
                .maybe_pool_max_idle_per_host(self.pool_max_idle_per_host)
                .maybe_pool_idle_timeout(self.pool_idle_timeout)
                .maybe_tcp_keepalive(self.tcp_keepalive)
                .build()
                .map_err(|e| ApiError::http(e.to_string())),
        }
    }
}

/// Extension trait for [`reqwest::ClientBuilder`] that accepts `Option<T>`
/// for pool and TCP knobs, no-opping when `None`.
///
/// Used internally by [`HttpClientConfig::build`].
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
trait ClientBuilderExt: Sized {
    fn maybe_pool_max_idle_per_host(self, val: Option<usize>) -> Self;
    fn maybe_pool_idle_timeout(self, val: Option<Duration>) -> Self;
    fn maybe_tcp_keepalive(self, val: Option<Duration>) -> Self;
}

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
impl ClientBuilderExt for reqwest::ClientBuilder {
    fn maybe_pool_max_idle_per_host(self, val: Option<usize>) -> Self {
        match val {
            Some(n) => self.pool_max_idle_per_host(n),
            None => self,
        }
    }

    fn maybe_pool_idle_timeout(self, val: Option<Duration>) -> Self {
        match val {
            Some(d) => self.pool_idle_timeout(Some(d)),
            None => self,
        }
    }

    fn maybe_tcp_keepalive(self, val: Option<Duration>) -> Self {
        match val {
            Some(d) => self.tcp_keepalive(d),
            None => self,
        }
    }
}

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "grammar")]
pub mod grammar;

#[cfg(feature = "openai")]
pub use openai::OpenAiClient;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicClient;

#[cfg(feature = "gemini")]
pub use gemini::GeminiClient;

#[cfg(feature = "grammar")]
pub use grammar::{JsonSchemaGrammar, ToolGrammarProvider};

#[cfg(feature = "ollama")]
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

#[cfg(feature = "deepseek")]
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

#[cfg(feature = "deepseek")]
const DEEPSEEK_DEFAULT_MODEL: &str = "deepseek-chat";

#[cfg(feature = "grok")]
const GROK_BASE_URL: &str = "https://api.x.ai/v1";

#[cfg(feature = "grok")]
const GROK_DEFAULT_MODEL: &str = "grok-beta";

#[cfg(feature = "zai")]
const ZAI_BASE_URL: &str = "https://api.z.ai/api/anthropic";

#[cfg(feature = "zai")]
const ZAI_DEFAULT_MODEL: &str = "glm-4.7";

/// Read an environment variable, falling back to a second name, then a
/// default value.
///
/// Reduces boilerplate in the convenience constructors below where a
/// provider supports multiple env-var aliases (e.g. `XAI_API_KEY` /
/// `GROK_API_KEY`).
/// Read an environment variable or return a default.
#[cfg(any(
    feature = "ollama",
    feature = "deepseek",
    feature = "grok",
    feature = "zai",
    feature = "openai"
))]
fn env_or_default(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// Read a primary env var, falling back to a secondary if unset.
#[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
fn env_or_fallback(primary: &str, fallback: &str) -> Option<String> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
}

/// Look up a required API key from the environment.
///
/// # Errors
///
/// Returns [`ApiError::auth_invalid_key`] if neither environment variable is set.
#[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
fn require_api_key(primary: &str, fallback: Option<&str>) -> Result<String, ApiError> {
    if let Some(fb) = fallback {
        if let Some(val) = env_or_fallback(primary, fb) {
            return Ok(val);
        }
    } else if let Ok(val) = std::env::var(primary) {
        return Ok(val);
    }
    Err(ApiError::auth_invalid_key(format!("{primary} not set")))
}

/// Separate inline `Role::System` messages from the rest of the history and
/// fold their text into a single system string.
///
/// Providers that reject an inline system role accept system content only as a
/// top-level request field. This helper pulls every system message out of
/// `messages`, concatenates their text parts (newline-separated), and merges
/// the result with an optional caller-supplied system prompt.
#[cfg(any(feature = "anthropic", feature = "gemini"))]
fn fold_system_messages<'a>(
    messages: &'a [crate::message::Message],
    system: Option<&str>,
) -> (Vec<&'a crate::message::Message>, Option<String>) {
    let mut folded = String::new();
    let non_system: Vec<&crate::message::Message> = messages
        .iter()
        .filter(|m| {
            if matches!(m.role, Role::System) {
                for part in &m.parts {
                    if let MessagePart::Text { text } = part {
                        if !folded.is_empty() {
                            folded.push('\n');
                        }
                        folded.push_str(text);
                    }
                }
                false
            } else {
                true
            }
        })
        .collect();
    let effective = match (system, folded.is_empty()) {
        (Some(s), false) => Some(format!("{s}\n{folded}")),
        (Some(s), true) => Some(s.to_string()),
        (None, false) => Some(folded),
        (None, true) => None,
    };
    (non_system, effective)
}

/// Ollama client — an [`OpenAiClient`] pointed at an Ollama server.
///
/// Works with both local Ollama (`http://localhost:11434/v1`, no API key
/// needed) and Ollama Cloud (`https://api.ollama.com/v1`, requires
/// `OLLAMA_API_KEY`).
///
/// Reads:
/// - `OLLAMA_API_KEY` — optional for local, required for cloud.
/// - `OLLAMA_BASE_URL` — optional, defaults to `http://localhost:11434/v1`.
///   Set to `https://api.ollama.com/v1` for Ollama Cloud.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::provider;
///
/// // Local:
/// let client = provider::ollama("llama3")?;
///
/// // Cloud (set OLLAMA_API_KEY and OLLAMA_BASE_URL):
/// let client = provider::ollama("llama3")?;
/// ```
///
/// # Errors
///
/// Returns [`ApiError`] if the HTTP client cannot be built.
#[cfg(feature = "ollama")]
pub fn ollama(model: &str) -> Result<OpenAiClient, ApiError> {
    let base = env_or_default("OLLAMA_BASE_URL", OLLAMA_BASE_URL);
    let api_key = env_or_default("OLLAMA_API_KEY", "ollama");

    OpenAiClient::builder()
        .with_api_key(api_key)
        .with_base_url(base)
        .with_model(model)
        .with_stream_usage(false)
        .build()
}

/// `DeepSeek` client — an [`OpenAiClient`] pointed at the `DeepSeek` API.
///
/// Reads `DEEPSEEK_API_KEY` (required) and optionally `DEEPSEEK_MODEL`
/// (defaults to `deepseek-chat`).
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::provider;
///
/// let client = provider::deepseek()?;
/// ```
///
/// # Errors
///
/// Returns [`ApiError`] if no API key is found.
#[cfg(feature = "deepseek")]
pub fn deepseek() -> Result<OpenAiClient, ApiError> {
    let api_key = require_api_key("DEEPSEEK_API_KEY", None)?;
    let model = env_or_default("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL);

    OpenAiClient::builder()
        .with_api_key(api_key)
        .with_base_url(DEEPSEEK_BASE_URL)
        .with_model(model)
        .build()
}

/// `Grok` (xAI) client — an [`OpenAiClient`] pointed at the xAI API.
///
/// Reads `XAI_API_KEY` (or `GROK_API_KEY`) (required) and optionally
/// `GROK_MODEL` (defaults to `grok-beta`).
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::provider;
///
/// let client = provider::grok()?;
/// ```
///
/// # Errors
///
/// Returns [`ApiError`] if no API key is found.
#[cfg(feature = "grok")]
pub fn grok() -> Result<OpenAiClient, ApiError> {
    let api_key = require_api_key("XAI_API_KEY", Some("GROK_API_KEY"))?;
    let model = std::env::var("XAI_MODEL")
        .or_else(|_| std::env::var("GROK_MODEL"))
        .unwrap_or_else(|_| GROK_DEFAULT_MODEL.into());

    OpenAiClient::builder()
        .with_api_key(api_key)
        .with_base_url(GROK_BASE_URL)
        .with_model(model)
        .build()
}

/// `Z.ai` (`ZhipuAI` / `BigModel`) client — an [`AnthropicClient`] pointed
/// at the `Z.ai` Anthropic-compatible API.
///
/// `Z.ai` exposes an Anthropic Messages-compatible API at
/// `https://api.z.ai/api/anthropic`.
///
/// Reads `ZAI_API_KEY` (or `ZHIPUAI_API_KEY`) (required) and optionally
/// `ZAI_MODEL` (defaults to `glm-4.6`).
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::provider;
///
/// let client = provider::zai()?;
/// ```
///
/// # Errors
///
/// Returns [`ApiError`] if no API key is found.
#[cfg(feature = "zai")]
pub fn zai() -> Result<AnthropicClient, ApiError> {
    let api_key = require_api_key("ZAI_API_KEY", Some("ZHIPUAI_API_KEY"))?;
    let model = env_or_default("ZAI_MODEL", ZAI_DEFAULT_MODEL);

    AnthropicClient::builder()
        .with_api_key(api_key)
        .with_base_url(ZAI_BASE_URL)
        .with_model(model)
        .build()
}

/// Self-hosted client — an [`OpenAiClient`] pointed at any custom endpoint.
///
/// Use this for `vLLM`, `LM Studio`, `text-generation-inference`, or any
/// other server that exposes an OpenAI-compatible API.
///
/// For servers that require an API key, set it via the `OPENAI_API_KEY`
/// environment variable or use [`OpenAiClient::builder`] directly.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::provider;
///
/// let client = provider::self_hosted("http://localhost:8080/v1", "my-model")?;
/// ```
///
/// # Errors
///
/// Returns [`ApiError`] if the HTTP client cannot be built.
#[cfg(feature = "openai")]
pub fn self_hosted(base_url: &str, model: &str) -> Result<OpenAiClient, ApiError> {
    let api_key = env_or_default("OPENAI_API_KEY", "self-hosted");

    OpenAiClient::builder()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .with_model(model)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(
        feature = "ollama",
        feature = "deepseek",
        feature = "grok",
        feature = "zai",
        feature = "openai"
    ))]
    macro_rules! env_set {
        ($($arg:tt)*) => {{
            // SAFETY: This is only used in single-threaded test code where
            // no other task is reading or writing environment variables.
            unsafe { std::env::set_var($($arg)*) }
        }};
    }

    #[cfg(any(
        feature = "ollama",
        feature = "deepseek",
        feature = "grok",
        feature = "zai",
        feature = "openai"
    ))]
    macro_rules! env_remove {
        ($($arg:tt)*) => {{
            // SAFETY: This is only used in single-threaded test code where
            // no other task is reading or writing environment variables.
            unsafe { std::env::remove_var($($arg)*) }
        }};
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn env_or_fallback_primary_set() {
        env_set!("LOOPCTL_TEST_PRIMARY", "primary-val");
        env_remove!("LOOPCTL_TEST_FALLBACK");
        assert_eq!(
            env_or_fallback("LOOPCTL_TEST_PRIMARY", "LOOPCTL_TEST_FALLBACK"),
            Some("primary-val".into())
        );
        env_remove!("LOOPCTL_TEST_PRIMARY");
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn env_or_fallback_fallback_used_when_primary_missing() {
        env_remove!("LOOPCTL_TEST_PRIMARY2");
        env_set!("LOOPCTL_TEST_FALLBACK2", "fallback-val");
        assert_eq!(
            env_or_fallback("LOOPCTL_TEST_PRIMARY2", "LOOPCTL_TEST_FALLBACK2"),
            Some("fallback-val".into())
        );
        env_remove!("LOOPCTL_TEST_FALLBACK2");
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn env_or_fallback_none_when_both_missing() {
        env_remove!("LOOPCTL_TEST_NEITHER_A");
        env_remove!("LOOPCTL_TEST_NEITHER_B");
        assert_eq!(
            env_or_fallback("LOOPCTL_TEST_NEITHER_A", "LOOPCTL_TEST_NEITHER_B"),
            None
        );
    }

    #[cfg(any(
        feature = "ollama",
        feature = "deepseek",
        feature = "grok",
        feature = "zai",
        feature = "openai"
    ))]
    #[test]
    fn env_or_default_uses_env_when_set() {
        env_set!("LOOPCTL_TEST_DEFAULT", "from-env");
        assert_eq!(
            env_or_default("LOOPCTL_TEST_DEFAULT", "fallback"),
            "from-env"
        );
        env_remove!("LOOPCTL_TEST_DEFAULT");
    }

    #[cfg(any(
        feature = "ollama",
        feature = "deepseek",
        feature = "grok",
        feature = "zai",
        feature = "openai"
    ))]
    #[test]
    fn env_or_default_uses_default_when_unset() {
        env_remove!("LOOPCTL_TEST_DEFAULT2");
        assert_eq!(
            env_or_default("LOOPCTL_TEST_DEFAULT2", "fallback"),
            "fallback"
        );
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn require_api_key_primary_set() {
        env_set!("LOOPCTL_TEST_KEY_PRIMARY", "secret");
        env_remove!("LOOPCTL_TEST_KEY_FALLBACK");
        let key = require_api_key(
            "LOOPCTL_TEST_KEY_PRIMARY",
            Some("LOOPCTL_TEST_KEY_FALLBACK"),
        )
        .unwrap();
        assert_eq!(key, "secret");
        env_remove!("LOOPCTL_TEST_KEY_PRIMARY");
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn require_api_key_fallback_used() {
        env_remove!("LOOPCTL_TEST_KEY_PRIMARY2");
        env_set!("LOOPCTL_TEST_KEY_FALLBACK2", "fallback-secret");
        let key = require_api_key(
            "LOOPCTL_TEST_KEY_PRIMARY2",
            Some("LOOPCTL_TEST_KEY_FALLBACK2"),
        )
        .unwrap();
        assert_eq!(key, "fallback-secret");
        env_remove!("LOOPCTL_TEST_KEY_FALLBACK2");
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn require_api_key_no_fallback_set() {
        env_set!("LOOPCTL_TEST_KEY_ONLY", "only-val");
        let key = require_api_key("LOOPCTL_TEST_KEY_ONLY", None).unwrap();
        assert_eq!(key, "only-val");
        env_remove!("LOOPCTL_TEST_KEY_ONLY");
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn require_api_key_errors_when_missing() {
        env_remove!("LOOPCTL_TEST_MISSING_KEY");
        let err = require_api_key("LOOPCTL_TEST_MISSING_KEY", None).unwrap_err();
        assert!(err.to_string().contains("LOOPCTL_TEST_MISSING_KEY"));
    }

    #[cfg(any(feature = "deepseek", feature = "grok", feature = "zai"))]
    #[test]
    fn require_api_key_errors_when_both_missing() {
        env_remove!("LOOPCTL_TEST_MISSING_A");
        env_remove!("LOOPCTL_TEST_MISSING_B");
        let err =
            require_api_key("LOOPCTL_TEST_MISSING_A", Some("LOOPCTL_TEST_MISSING_B")).unwrap_err();
        assert!(err.to_string().contains("LOOPCTL_TEST_MISSING_A"));
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_client_builds_with_defaults() {
        use crate::api::ApiClient;
        env_remove!("OLLAMA_BASE_URL");
        let client = ollama("llama3").unwrap();
        assert_eq!(client.model(), "llama3");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_client_respects_base_url_env() {
        use crate::api::ApiClient;
        env_set!("OLLAMA_BASE_URL", "http://my-host:1234/v1");
        let client = ollama("test-model").unwrap();
        assert_eq!(client.model(), "test-model");
        env_remove!("OLLAMA_BASE_URL");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_client_uses_api_key_when_set() {
        use crate::api::ApiClient;
        env_remove!("OLLAMA_BASE_URL");
        env_set!("OLLAMA_API_KEY", "my-cloud-key");
        // Should build successfully with the cloud key — no network call.
        let client = ollama("llama3").unwrap();
        assert_eq!(client.model(), "llama3");
        env_remove!("OLLAMA_API_KEY");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_client_defaults_to_local_without_key() {
        use crate::api::ApiClient;
        env_remove!("OLLAMA_BASE_URL");
        env_remove!("OLLAMA_API_KEY");
        // Should still build — local Ollama doesn't need a real key.
        let client = ollama("llama3").unwrap();
        assert_eq!(client.model(), "llama3");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn self_hosted_client_builds() {
        use crate::api::ApiClient;
        let client = self_hosted("http://localhost:8080/v1", "my-model").unwrap();
        assert_eq!(client.model(), "my-model");
    }

    /// Build a `Role::System` message carrying the given text parts.
    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    fn sys_msg(texts: &[&str]) -> crate::message::Message {
        use crate::message::{MessagePart, Role};
        let parts: Vec<MessagePart> = texts.iter().map(|t| MessagePart::text(*t)).collect();
        crate::message::Message::new(Role::System, parts)
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_no_system_messages_no_caller_returns_none() {
        let msgs = [crate::message::Message::user("hi")];
        let (non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(non_system.len(), 1);
        assert!(system.is_none(), "no system content → None");
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_caller_only_passes_through() {
        let msgs = [crate::message::Message::user("hi")];
        let (non_system, system) = fold_system_messages(&msgs, Some("be brief"));
        assert_eq!(non_system.len(), 1);
        assert_eq!(system.as_deref(), Some("be brief"));
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_single_system_message_removed_and_folded() {
        let msgs = [
            crate::message::Message::user("hello"),
            sys_msg(&["stay on task"]),
            crate::message::Message::assistant("working"),
        ];
        let (non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(non_system.len(), 2, "system message filtered out");
        assert_eq!(system.as_deref(), Some("stay on task"));
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_caller_prompt_prepended_to_folded() {
        let msgs = [crate::message::Message::user("hi"), sys_msg(&["reminder"])];
        let (_non_system, system) = fold_system_messages(&msgs, Some("be brief"));
        let system = system.expect("merged system is Some");
        assert!(
            system.starts_with("be brief"),
            "caller prompt first: got {system:?}"
        );
        assert!(
            system.contains("reminder"),
            "folded text appended: got {system:?}"
        );
        assert!(
            system.contains('\n'),
            "caller and folded are newline-separated: got {system:?}"
        );
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_multiple_system_messages_joined_with_newlines() {
        let msgs = [
            sys_msg(&["first reminder"]),
            crate::message::Message::user("hi"),
            sys_msg(&["second reminder"]),
        ];
        let (non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(non_system.len(), 1, "both system messages filtered");
        let system = system.expect("folded text is Some");
        assert_eq!(system, "first reminder\nsecond reminder");
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_only_text_parts_are_folded() {
        // A System message carrying a tool-call part (unusual, but defensive):
        // only the text parts contribute to the fold.
        use crate::message::{MessagePart, Role};
        let system_msg = crate::message::Message::new(
            Role::System,
            vec![
                MessagePart::text("keep this"),
                MessagePart::tool_call("id", "some_tool", serde_json::json!({})),
            ],
        );
        let msgs = [crate::message::Message::user("hi"), system_msg];
        let (_non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(system.as_deref(), Some("keep this"));
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_preserves_relative_order_of_non_system_messages() {
        let msgs = [
            crate::message::Message::user("first"),
            sys_msg(&["mid reminder"]),
            crate::message::Message::assistant("second"),
            crate::message::Message::user("third"),
        ];
        let (non_system, _system) = fold_system_messages(&msgs, None);
        let texts: Vec<&str> = non_system
            .iter()
            .flat_map(|m| {
                m.parts.iter().filter_map(|p| match p {
                    crate::message::MessagePart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_empty_text_part_contributes_nothing() {
        // A System message whose text is empty: folded string stays empty, so
        // with no caller prompt the result is None.
        let msgs = [sys_msg(&[""])];
        let (non_system, system) = fold_system_messages(&msgs, None);
        assert!(non_system.is_empty(), "system message still filtered");
        assert!(
            system.is_none(),
            "empty folded text and no caller → None (got {system:?})"
        );
    }

    #[cfg(any(feature = "anthropic", feature = "gemini"))]
    #[test]
    fn fold_system_multiple_text_parts_in_one_message_joined() {
        let msgs = [sys_msg(&["part one", "part two"])];
        let (_non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(system.as_deref(), Some("part one\npart two"));
    }

    #[test]
    fn default_builds_clean() {
        assert!(HttpClientConfig::default().build().is_ok());
    }

    #[test]
    fn accepts_injected_http_client() {
        let shared = reqwest::Client::new();
        let config = HttpClientConfig::default().with_http_client(shared);
        assert!(config.build().is_ok());
    }

    #[test]
    fn injected_client_supersedes_timeouts() {
        let shared = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let config = HttpClientConfig::default()
            .with_http_client(shared)
            .with_timeout(Duration::from_secs(99));
        assert!(config.build().is_ok());
    }

    #[test]
    fn pool_knobs_build_clean() {
        let config = HttpClientConfig::default()
            .with_pool_max_idle_per_host(4)
            .with_pool_idle_timeout(Duration::from_secs(30))
            .with_tcp_keepalive(Duration::from_secs(90));
        assert!(config.build().is_ok());
    }

    #[test]
    fn tcp_nodelay_default_is_true() {
        assert!(HttpClientConfig::default().tcp_nodelay);
    }

    #[test]
    fn with_tcp_nodelay_can_disable() {
        let config = HttpClientConfig::default().with_tcp_nodelay(false);
        assert!(!config.tcp_nodelay);
    }

    #[test]
    fn injected_client_ignores_pool_knobs() {
        let shared = reqwest::Client::new();
        let config = HttpClientConfig::default()
            .with_http_client(shared)
            .with_pool_max_idle_per_host(4)
            .with_pool_idle_timeout(Duration::from_secs(30));
        assert!(config.build().is_ok());
    }

    #[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
    async fn serve_once(
        status: u16,
        headers: String,
        body: Vec<u8>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            drop(sock.read(&mut buf).await);
            let extra = if headers.is_empty() {
                String::new()
            } else {
                format!("{headers}\r\n")
            };
            let head = format!(
                "HTTP/1.1 {status} OK\r\nContent-Length: {clen}\r\n{extra}\r\n",
                clen = body.len(),
            );
            drop(sock.write_all(head.as_bytes()).await);
            drop(sock.write_all(&body).await);
            drop(sock.flush().await);
        });
        (format!("http://{addr}"), handle)
    }

    #[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
    async fn get_response(url: &str) -> reqwest::Response {
        reqwest::Client::new()
            .get(url)
            .send()
            .await
            .expect("request to test server must succeed")
    }

    #[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
    #[tokio::test]
    async fn read_bounded_body_accepts_under_limit() {
        let body = b"{\"ok\":true}".to_vec();
        let (url, handle) = serve_once(200, String::new(), body.clone()).await;
        let resp = get_response(&url).await;
        let bytes = read_bounded_body(resp).await.expect("small body must pass");
        assert_eq!(bytes.as_ref(), body.as_slice());
        handle.await.unwrap();
    }

    #[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
    #[tokio::test]
    async fn read_bounded_body_rejects_oversized_content_length() {
        let body = vec![b'x'; MAX_RESPONSE_BODY + 1];
        let (url, handle) = serve_once(200, String::new(), body).await;
        let resp = get_response(&url).await;
        let err = read_bounded_body(resp)
            .await
            .expect_err("oversized body must reject");
        assert!(
            err.to_string().contains("too large"),
            "expected a too-large error, got: {err}"
        );
        handle.await.unwrap();
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn error_body_read_is_bounded_by_the_cap() {
        use crate::api::ApiClient as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let written = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = written.clone();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            drop(sock.read(&mut buf).await);
            let head = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2097152\r\nConnection: close\r\n\r\n";
            drop(sock.write_all(head.as_bytes()).await);
            let chunk = vec![b'x'; 8192];
            for _ in 0..256 {
                // Yield between chunks: on kernels with multi-MiB socket
                // buffers the burst of non-blocking writes below never
                // returns Pending, so without this the server would finish
                // buffering the whole body before the client task ever runs
                // and its early abort could not interrupt the transfer.
                tokio::task::yield_now().await;
                if sock.write_all(&chunk).await.is_err() {
                    break;
                }
                counter.fetch_add(chunk.len(), Ordering::SeqCst);
                drop(sock.flush().await);
            }
        });

        let client = crate::provider::OpenAiClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .build()
            .expect("client builds");
        let result = client
            .create_message(&crate::api::StreamRequest::new(vec![]))
            .await;
        assert!(result.is_err(), "a 500 response must surface an error");
        server.await.unwrap();
        let total = written.load(Ordering::SeqCst);
        assert!(
            total < 2 * 1024 * 1024,
            "the error-body cap (8 KiB) must stop the read short of the declared 2 MiB body; the server wrote {total} bytes"
        );
    }
}
