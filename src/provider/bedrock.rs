//! AWS Bedrock provider — SigV4-authenticated, two invoke paths.
//!
//! Routes through the Bedrock runtime (`bedrock-runtime.<region>.amazonaws.com`)
//! with AWS Signature Version 4 credentials. Two paths are supported
//! (see [`BedrockPath`]): the native Anthropic Messages body for
//! `anthropic.claude-*` model ids (reusing [`crate::provider::anthropic`]'s
//! body builders and stream emitter), and Bedrock's cross-model
//! [`Converse`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
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
/// Derived per request from the model id (ids starting with `anthropic.`
/// use the native Anthropic path; everything else uses Converse) unless
/// pinned by the builder's [`path`](BedrockClientBuilder::path) override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockPath {
    /// Native Anthropic Messages body via `InvokeModel`/`InvokeModelWithResponseStream`.
    ///
    /// Reuses the `anthropic.rs` body builders and stream emitter
    /// (wrapped in event-stream frames). Use for `anthropic.*` model
    /// ids.
    Anthropic,

    /// Bedrock's cross-model Converse API.
    ///
    /// Uses Bedrock's own request body and event shapes (`contentBlockDelta`
    /// etc.) rather than any provider's native format. Use for Titan,
    /// Llama, Mistral, and any non-Anthropic model.
    Converse,
}

/// The `SigV4` signing triple for one credential generation.
///
/// Swapped as a unit (never field-by-field) so every request signs
/// with a key id and secret from the same generation — a mixed pair
/// fails signature verification on every call.
#[derive(Clone)]
struct Credentials {
    /// The IAM access key id.
    ///
    /// Identifies the credentials that sign each request.
    access_key_id: String,

    /// The IAM secret access key.
    ///
    /// Used for `SigV4` signing only; never sent on the wire.
    secret_access_key: String,

    /// The optional STS / role session token.
    ///
    /// Included in the signed headers when present; rotates together
    /// with the pair because temporary credentials bind all three.
    session_token: Option<String>,
}

/// An AWS Bedrock chat client with streaming support, SigV4-authenticated.
///
/// Implements [`ApiClient`] by routing through the Bedrock runtime.
/// Credentials are signed with AWS Signature Version 4 — there is no
/// bearer API key. The model is mutable per request via
/// [`set_model`](ApiClient::set_model), and the credential triple can
/// be swapped atomically on a live client via
/// [`set_credentials`](Self::set_credentials) for short-lived
/// STS/IRSA-style credentials; each request signs with a single
/// consistent snapshot, so an in-flight request never sees a torn
/// pair.
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
pub struct BedrockClient {
    /// The AWS region the client targets.
    ///
    /// Determines the endpoint host (`bedrock-runtime.{region}.amazonaws.com`).
    region: String,

    /// The `SigV4` credential triple, swappable as one unit.
    ///
    /// Held behind a lock so [`set_credentials`](Self::set_credentials)
    /// can rotate short-lived STS/IRSA credentials on a live client:
    /// readers take a snapshot of the whole triple, so key id and
    /// secret are always from the same generation. Category 1 poison
    /// policy (single-struct swap).
    credentials: std::sync::RwLock<Credentials>,

    /// The output-token budget sent with every request.
    ///
    /// Defaults to the engine-wide shared constant; models whose own
    /// cap is lower (Claude 3 Haiku and Sonnet at 4096) reject a budget
    /// above their cap, so set it per client when targeting those.
    max_tokens: u32,

    /// The current Bedrock model id.
    ///
    /// Mutable via [`set_model`](ApiClient::set_model); read per
    /// request. Category 1 poison policy (single `String`).
    model: Mutex<String>,

    /// The invoke-path override, if set.
    ///
    /// When `None` the path is derived per request from the current
    /// model id (see [`effective_path`](Self::effective_path)), so a
    /// mid-run model switch across vendors — a fallback chain, for
    /// example — keeps speaking the right wire format. `Some(…)` pins
    /// the path regardless of model.
    path: Option<BedrockPath>,

    /// The shared HTTP client.
    ///
    /// Reused across requests for connection pooling.
    http: reqwest::Client,
}

/// Builder for [`BedrockClient`].
///
/// Collects the AWS credentials, region, model, optional path override,
/// and optional output-token budget; [`build`](Self::build) validates
/// that the required fields are present and non-empty. The invoke path itself is derived per
/// request from the current model id unless the override pins it (see
/// [`set_model`](ApiClient::set_model), which the path follows).
#[derive(Default)]
pub struct BedrockClientBuilder {
    /// The AWS region, if set.
    ///
    /// Required: [`build`](Self::build) errors without it.
    region: Option<String>,

    /// The IAM access key id, if set.
    ///
    /// Required: [`build`](Self::build) errors without it.
    access_key_id: Option<String>,

    /// The IAM secret access key, if set.
    ///
    /// Required: [`build`](Self::build) errors without it.
    secret_access_key: Option<String>,

    /// An optional STS / role session token.
    ///
    /// Forwarded to [`session_token`](BedrockClientBuilder::session_token)
    /// and signed into every request when present.
    session_token: Option<String>,

    /// The Bedrock model id, if set.
    ///
    /// Required: [`build`](Self::build) errors without it. The id's
    /// prefix derives the invoke path per request unless overridden.
    model: Option<String>,

    /// The invoke-path override, if set.
    ///
    /// When absent, the client derives the path from each request's
    /// model id via [`auto_path`]; when set, the path is pinned
    /// regardless of model.
    path: Option<BedrockPath>,

    /// The output-token budget override, if set.
    ///
    /// When absent, requests carry the engine-wide shared default;
    /// set it below that for models whose own cap is lower.
    max_tokens: Option<u32>,
}

impl std::fmt::Debug for SigV4Headers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4Headers")
            .field("authorization", &"<redacted>")
            .field("amz_date", &self.amz_date)
            .finish()
    }
}

impl std::fmt::Debug for BedrockClientBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockClientBuilder")
            .field("region", &self.region)
            .field("model", &self.model)
            .field("path", &self.path)
            .field("max_tokens", &self.max_tokens)
            .field("access_key_id", &"<redacted>")
            .field("secret_access_key", &"<redacted>")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish_non_exhaustive()
    }
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
    /// The id's prefix (`anthropic.*`) derives the invoke path per
    /// request unless [`path`](Self::path) pins it.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Pin the invoke path, disabling per-request derivation.
    ///
    /// By default the path follows the current model id's prefix;
    /// this forces a specific path regardless of model.
    #[must_use]
    pub fn path(mut self, path: BedrockPath) -> Self {
        self.path = Some(path);
        self
    }

    /// Set the output-token budget sent with every request.
    ///
    /// Defaults to the engine-wide shared constant. Models whose own
    /// cap is lower — Claude 3 Haiku and Sonnet at 4096 — reject a
    /// budget above their cap, so configure it per client when
    /// targeting those. A zero budget is rejected at
    /// [`build`](Self::build).
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Build the client.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] when the region, credentials, or model are
    /// missing or empty, or when the HTTP client cannot be constructed.
    pub fn build(self) -> Result<BedrockClient, ApiError> {
        let region = require_non_empty("region", self.region)?;
        validate_region(&region)?;
        let access_key_id = require_non_empty("access_key_id", self.access_key_id)?;
        let secret_access_key = require_non_empty("secret_access_key", self.secret_access_key)?;
        let model = require_non_empty("model", self.model)?;
        let max_tokens = match self.max_tokens {
            None => crate::provider::anthropic::DEFAULT_MAX_TOKENS,
            Some(0) => {
                return Err(ApiError::config_validation(
                    "bedrock: max_tokens must be greater than zero",
                ));
            }
            Some(max) => max,
        };
        let http = crate::provider::HttpClientConfig::default()
            .build()
            .map_err(|e| ApiError::config(format!("bedrock: HTTP client: {e}")))?;
        Ok(BedrockClient {
            region,
            credentials: std::sync::RwLock::new(Credentials {
                access_key_id,
                secret_access_key,
                session_token: self.session_token,
            }),
            max_tokens,
            model: Mutex::new(model),
            path: self.path,
            http,
        })
    }
}

/// Extract a required, non-empty builder field.
///
/// `None` (unset) and `""` (set but empty) both fail with the same
/// message — an empty region would build a `…..amazonaws.com` host and
/// an empty model a `/model//…` route, and both would only surface as
/// per-request wire failures instead of a build-time error.
///
/// # Errors
///
/// Returns [`ApiError::config`] naming the field when it is unset or
/// empty.
fn require_non_empty(field: &str, value: Option<String>) -> Result<String, ApiError> {
    value
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ApiError::config(format!("bedrock: {field} is missing")))
}

/// Validate the AWS region grammar.
///
/// Region names are lowercase alphanumerics and hyphens (`us-east-1`,
/// `cn-north-1`, `us-gov-west-1`, `us-iso-east-1`). Enforcing this at
/// build time keeps the interpolated host
/// (`bedrock-runtime.{region}.amazonaws.com`) a genuine AWS endpoint —
/// a region containing dots or slashes would change which host the
/// request is sent to, redirecting the (TLS-protected but still
/// delivered) conversation body.
///
/// # Errors
///
/// Returns [`ApiError::config_validation`] for any other shape.
fn validate_region(region: &str) -> Result<(), ApiError> {
    if region
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        Ok(())
    } else {
        Err(ApiError::config_validation(format!(
            "bedrock: region {region:?} must be lowercase alphanumerics and hyphens"
        )))
    }
}

/// Auto-select the invoke path from a Bedrock model id.
///
/// Model ids under the `anthropic.` prefix speak the native Anthropic
/// Messages protocol on Bedrock and take the cheaper, format-preserving
/// [`Anthropic`](BedrockPath::Anthropic) path; every other id (Titan,
/// Nova, Llama, Mistral, inference-profile-prefixed Anthropic ids) goes
/// through [`Converse`](BedrockPath::Converse), the cross-model API.
fn auto_path(model: &str) -> BedrockPath {
    if model.starts_with("anthropic.") {
        BedrockPath::Anthropic
    } else {
        BedrockPath::Converse
    }
}

/// The `AWS SigV4` signing output — headers to attach to a request.
///
/// Produced by [`sigv4_sign`]; carries the two values the caller does
/// not already know: the composed `Authorization` header and the
/// timestamp it was signed at. `Debug` redacts the authorization — it
/// embeds the access key id and a signature derived from the secret —
/// while keeping the timestamp, which is the diagnostically useful
/// half.
struct SigV4Headers {
    /// The full `Authorization` header value.
    ///
    /// `AWS4-HMAC-SHA256 Credential=…/scope, SignedHeaders=…,
    /// Signature=…`, ready to attach verbatim.
    authorization: String,

    /// The `X-Amz-Date` value (`YYYYMMDD'T'HHMMSS'Z'`).
    ///
    /// Must be sent as-is alongside `Authorization`; the signature is
    /// only valid for this exact timestamp.
    amz_date: String,
}

/// Sign a request with AWS Signature Version 4.
///
/// Produces the `Authorization` and `X-Amz-Date` header values for a
/// POST to `host` at `uri` with the given `payload` bytes. A session
/// token, when present, is signed into the canonical headers (the
/// caller attaches the header itself). The signature covers the payload
/// hash, so it must be computed over the exact bytes that go on the
/// wire.
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

    let k_signing = signing_key(secret_access_key, date_stamp, region, "bedrock");
    let signature = hex::encode(hmac_key(&k_signing, string_to_sign.as_bytes()));

    SigV4Headers {
        authorization: format!(
            "AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, \
             SignedHeaders={signed_header_list}, Signature={signature}"
        ),
        amz_date,
    }
}

/// Apply `SigV4` authentication headers to a POST request builder.
///
/// The single signing point for both invoke paths: derives the
/// canonical URI from the URL's `amazonaws.com` path suffix, signs the
/// exact body bytes, and attaches `Authorization`, `X-Amz-Date`,
/// `Content-Type`, `X-Amz-Content-Sha256`, and — when credentials
/// carry one — `X-Amz-Security-Token`. The caller supplies the body
/// separately so the same signed builder serves both the buffered and
/// streaming requests. The `Authorization` and session-token header
/// values are marked sensitive so they never appear in debug output.
///
/// # Errors
///
/// Returns [`ApiError`] when a credential or the signature contains
/// characters HTTP header values reject — control or non-ASCII bytes,
/// which real AWS credentials never contain.
fn apply_sigv4(
    mut request: reqwest::RequestBuilder,
    url: &str,
    region: &str,
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    body: &[u8],
) -> Result<reqwest::RequestBuilder, ApiError> {
    let uri = url
        .split_once("amazonaws.com")
        .map_or("/".to_string(), |(_, rest)| rest.to_string());
    let host = format!("bedrock-runtime.{region}.amazonaws.com");
    let creds = sigv4_sign(
        access_key_id,
        secret_access_key,
        session_token,
        region,
        &host,
        &uri,
        body,
        std::time::SystemTime::now(),
    );
    let mut authorization = reqwest::header::HeaderValue::from_str(&creds.authorization)
        .map_err(|e| ApiError::config(format!("bedrock: authorization header: {e}")))?;
    authorization.set_sensitive(true);
    request = request
        .header("Authorization", authorization)
        .header("X-Amz-Date", creds.amz_date)
        .header("Content-Type", "application/json")
        .header(
            "X-Amz-Content-Sha256",
            hex::encode(sha2::Sha256::digest(body)),
        );
    if let Some(token) = session_token {
        let mut value = reqwest::header::HeaderValue::from_str(token)
            .map_err(|e| ApiError::config(format!("bedrock: session token header: {e}")))?;
        value.set_sensitive(true);
        request = request.header("X-Amz-Security-Token", value);
    }
    Ok(request)
}

/// Format a `SystemTime` as an AMZ date (`YYYYMMDD'T'HHMMSS'Z'`).
///
/// `SigV4` stamps every request with the current UTC time in this
/// exact compact form; `X-Amz-Date` and the credential scope's date
/// stamp are both derived from it.
fn format_amz_date(now: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let (year, month, day, hour, min, sec) = epoch_to_utc(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z")
}

/// Convert epoch seconds to UTC calendar fields.
///
/// Uses Howard Hinnant's civil-from-days algorithm, which is total
/// on `i64` for any representable date.
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

/// Derive the `SigV4` signing key for a service.
///
/// The four-step HMAC chain from the AWS general reference, each step
/// keyed by the previous output:
/// `AWS4<secret>` → date stamp → region → service → `aws4_request`.
/// The chain is pinned against the reference's documented example
/// output for (`20150830`, `us-east-1`, `iam`).
fn signing_key(secret: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_key(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_key(&k_date, region.as_bytes());
    let k_service = hmac_key(&k_region, service.as_bytes());
    hmac_key(&k_service, b"aws4_request")
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

/// Percent-encode one URI path segment.
///
/// Model ids contain `:` (every versioned id, e.g. `…-v1:0`) and ARN ids
/// additionally contain `/`; the AWS SDKs send these percent-encoded, and
/// the `SigV4` canonical URI is the encoded form. Keeps the RFC 3986
/// unreserved set (`A-Za-z0-9-._~`) verbatim and escapes every other
/// byte as `%XX` with uppercase hex, so the signed path and the sent
/// path are always the same string.
fn encode_path_segment(segment: &str) -> String {
    segment
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                char::from(byte).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

/// The streaming endpoint URL for a model and region.
///
/// Both invoke paths keep the model id in the URI path, as the Bedrock
/// runtime routes on it: `…/model/{model}/invoke-with-response-stream`
/// for the native Anthropic path, `…/model/{model}/converse-stream`
/// for Converse. The id is percent-encoded per
/// [`encode_path_segment`] — the same string is signed and sent.
fn stream_url(region: &str, model: &str, path: BedrockPath) -> String {
    let suffix = match path {
        BedrockPath::Anthropic => "invoke-with-response-stream",
        BedrockPath::Converse => "converse-stream",
    };
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{}/{suffix}",
        encode_path_segment(model)
    )
}

/// The non-streaming endpoint URL for a model and region.
///
/// Mirrors [`stream_url`] for the single-shot invoke routes:
/// `…/model/{model}/invoke` (native Anthropic) and
/// `…/model/{model}/converse`.
fn invoke_url(region: &str, model: &str, path: BedrockPath) -> String {
    let suffix = match path {
        BedrockPath::Anthropic => "invoke",
        BedrockPath::Converse => "converse",
    };
    format!(
        "https://bedrock-runtime.{region}.amazonaws.com/model/{}/{suffix}",
        encode_path_segment(model)
    )
}

/// Incremental decoder for `application/vnd.amazon.eventstream` frames.
///
/// Feed raw response-body bytes via [`push`](Self::push); each complete
/// frame yields a decoded [`AwsEvent`] — its event-type header and
/// payload bytes. Frame CRCs are read past but not validated (the
/// transport is TLS, so corruption is unlikely and dropping the stream
/// on a checksum would not recover more than erroring does). A frame
/// that is structurally invalid but whose extent is known (headers
/// overrun the declared length) is skipped individually and decoding
/// continues with the bytes after it; only an out-of-bounds length
/// prefix — which leaves the stream unsynchronized, with no trustworthy
/// extent to skip — clears the buffer.
#[derive(Debug, Default)]
struct AwsEventStreamDecoder {
    /// Bytes received but not yet consumed by a complete frame.
    ///
    /// Grows with every [`push`](Self::push) and is drained from the
    /// front each time a full frame is decoded.
    buf: Vec<u8>,
}

/// One decoded event-stream frame.
///
/// Produced by [`AwsEventStreamDecoder::push`]; carries the framing
/// headers the stream loop dispatches on plus the raw payload bytes.
#[derive(Debug, PartialEq, Eq)]
struct AwsEvent {
    /// The `:event-type` header value.
    ///
    /// `"chunk"` for data frames. Exception frames carry no
    /// `:event-type` at all — they set `:message-type` to
    /// `"exception"` and an `:exception-type` header instead, so an
    /// empty value here with a non-`"event"` [`message_type`](Self::message_type)
    /// is how callers recognize them.
    event_type: String,

    /// The `:message-type` header value.
    ///
    /// `"event"` for data frames, `"response"` for the initial frame,
    /// `"exception"` for exception frames.
    message_type: String,

    /// The `:exception-type` header value.
    ///
    /// Carries the exception's name (e.g. `"throttlingException"`) on
    /// exception frames; empty on every other frame. Surfaced in the
    /// stream error so a failure names its cause — the payload alone is
    /// often empty.
    exception_type: String,

    /// The frame's payload bytes.
    ///
    /// The JSON chunk for `chunk` events; empty for `initial-response`.
    payload: Vec<u8>,
}

/// The outcome of one decoding step over the buffer front.
///
/// Distinguishes the three states [`AwsEventStreamDecoder::push`] loops
/// on: wait for more bytes, discard a malformed frame and keep going,
/// or emit a decoded frame.
enum DecodeStep {
    /// The buffer holds less than one complete frame.
    ///
    /// The front bytes may be a valid frame prefix — nothing is
    /// discarded and decoding stops until more bytes arrive.
    NeedMore,

    /// A malformed frame was discarded.
    ///
    /// Decoding continues from the bytes after it: the frame's declared
    /// extent was trustworthy enough to skip exactly (or, for an
    /// out-of-bounds length, the whole buffer was cleared as
    /// unsynchronized).
    Skipped,

    /// One complete frame decoded.
    ///
    /// Carries the [`AwsEvent`] assembled from the frame's headers and
    /// payload; the frame's bytes are drained from the buffer front.
    Event(AwsEvent),
}

impl AwsEventStreamDecoder {
    /// Feed raw bytes; returns every complete frame decoded so far.
    ///
    /// Bytes are buffered until they form whole frames, so partial
    /// network chunks may return nothing and a later call may return
    /// several frames at once. A skipped malformed frame does not stop
    /// the scan — frames after it in the same buffer still decode.
    fn push(&mut self, bytes: &[u8]) -> Vec<AwsEvent> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            match self.decode_frame() {
                DecodeStep::NeedMore => break,
                DecodeStep::Skipped => {}
                DecodeStep::Event(event) => events.push(event),
            }
        }
        events
    }

    /// Take one decoding step over the buffer front.
    ///
    /// [`NeedMore`](DecodeStep::NeedMore) when the buffer holds less
    /// than a complete frame; [`Skipped`](DecodeStep::Skipped) when a
    /// frame is discarded (structurally invalid frames drain exactly
    /// their declared extent; an out-of-bounds length prefix clears the
    /// buffer, as the stream is unsynchronized); [`Event`](DecodeStep::Event)
    /// for a decoded frame, which drains the buffer front.
    fn decode_frame(&mut self) -> DecodeStep {
        if self.buf.len() < 12 {
            return DecodeStep::NeedMore;
        }
        let Some(total_len) = be_u32(&self.buf, 0)
            .and_then(|len| usize::try_from(len).ok())
            .filter(|len| (16..=16 * 1024 * 1024).contains(len))
        else {
            self.buf.clear();
            return DecodeStep::Skipped;
        };
        if self.buf.len() < total_len {
            return DecodeStep::NeedMore;
        }
        let headers_len = be_u32(&self.buf, 4).map_or(0, |len| len as usize);
        let headers_end = 12usize.saturating_add(headers_len);
        if headers_end > total_len {
            self.buf.drain(..total_len);
            return DecodeStep::Skipped;
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
        DecodeStep::Event(AwsEvent {
            event_type: headers
                .iter()
                .find(|(k, _)| k == ":event-type")
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            exception_type: headers
                .iter()
                .find(|(k, _)| k == ":exception-type")
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

/// Read a big-endian `u16` from `buf` at `offset`.
///
/// Returns `None` when the slice is too short.
fn be_u16(buf: &[u8], offset: usize) -> Option<u16> {
    let slice = buf.get(offset..offset.checked_add(2)?)?;
    let b0 = slice.first().copied().unwrap_or(0);
    let b1 = slice.get(1).copied().unwrap_or(0);
    Some((u16::from(b0) << 8) | u16::from(b1))
}

/// Read a big-endian `u32` from `buf` at `offset`.
///
/// Returns `None` when the slice is too short.
#[allow(clippy::arithmetic_side_effects)]
fn be_u32(buf: &[u8], offset: usize) -> Option<u32> {
    let slice = buf.get(offset..offset.checked_add(4)?)?;
    let b0 = slice.first().copied().unwrap_or(0);
    let b1 = slice.get(1).copied().unwrap_or(0);
    let b2 = slice.get(2).copied().unwrap_or(0);
    let b3 = slice.get(3).copied().unwrap_or(0);
    Some((u32::from(b0) << 24) | (u32::from(b1) << 16) | (u32::from(b2) << 8) | u32::from(b3))
}

/// Parse event-stream TLV headers into key-value pairs.
///
/// Each entry is a 1-byte name length, the name, a 1-byte value type,
/// and the type-encoded value. Handles string (7), boolean true (0),
/// and boolean false (1) value types; other types yield empty strings.
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
                let len = be_u16(bytes, i).unwrap_or(0) as usize;
                i += 2;
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
/// [`crate::provider::anthropic`]'s message translation and system
/// folding. Unused fields (`system`, `tools`) are omitted rather than
/// sent empty — Anthropic's schema rejects an empty `tools` array, so
/// the default no-tools request must not carry one. Messages with no
/// parts are dropped: they would serialize to an empty content array
/// the API rejects. The output budget comes from the client's
/// [`max_tokens`](BedrockClientBuilder::max_tokens) setting — models
/// whose own cap is lower (Claude 3 Haiku and Sonnet at 4096) reject a
/// budget above it, so configure it per client for those.
fn anthropic_body(request: &StreamRequest, max_tokens: u32) -> serde_json::Value {
    let (messages, system) =
        crate::provider::fold_system_messages(&request.messages, request.system.as_deref());
    let converted: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| !m.parts.is_empty())
        .map(|m| crate::provider::anthropic::convert_message(m))
        .collect();
    let mut body = serde_json::json!({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": converted,
    });
    if let Some(obj) = body.as_object_mut() {
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            obj.insert("system".to_string(), serde_json::json!(system));
        }
        if let Some(tools) = request.tools.as_ref().filter(|t| !t.is_empty()) {
            obj.insert(
                "tools".to_string(),
                serde_json::json!(crate::provider::anthropic::convert_tools(
                    tools.as_slice(),
                    false,
                )),
            );
        }
    }
    body
}

/// Build the Converse API request body.
///
/// Translates the engine's [`StreamRequest`] into Bedrock's cross-model
/// shape: each message's content becomes an array of typed blocks
/// (`text`, `toolUse`, `toolResult`), the system prompt a top-level
/// list of text blocks, and tools a `toolConfig` of `toolSpec`
/// entries. Fields the request doesn't exercise (system, `toolConfig`)
/// are omitted rather than sent empty, and messages with no parts are
/// dropped — an empty `content` array fails Converse validation. The
/// output budget comes from the client's
/// [`max_tokens`](BedrockClientBuilder::max_tokens) setting — models
/// whose own cap is lower reject a budget above it, so configure it
/// per client for those.
fn converse_body(request: &StreamRequest, max_tokens: u32) -> serde_json::Value {
    let (messages, system) =
        crate::provider::fold_system_messages(&request.messages, request.system.as_deref());
    let converted: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| !m.parts.is_empty())
        .map(|m| converse_message(m))
        .collect();
    let mut body = serde_json::json!({
        "messages": converted,
        "inferenceConfig": {
            "maxTokens": max_tokens,
        },
    });
    if let Some(obj) = body.as_object_mut() {
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            obj.insert(
                "system".to_string(),
                serde_json::json!([{ "text": system }]),
            );
        }
        if let Some(tools) = request.tools.as_ref().filter(|t| !t.is_empty()) {
            obj.insert(
                "toolConfig".to_string(),
                serde_json::json!({ "tools": convert_converse_tools(tools) }),
            );
        }
    }
    body
}

/// Translate one engine message into a Converse message object.
///
/// Content becomes typed blocks: text parts map to `{"text": …}`,
/// tool-call parts to `{"toolUse": {toolUseId, name, input}}`, and
/// tool results to `{"toolResult": …}` blocks riding a user message —
/// Converse has no separate tool role. Image parts are not supported
/// on this path, matching the sibling providers.
fn converse_message(message: &crate::message::Message) -> serde_json::Value {
    let role = if message.role == crate::message::Role::Assistant {
        "assistant"
    } else {
        "user"
    };
    let mut content = Vec::new();
    for part in &message.parts {
        match part {
            crate::message::MessagePart::Text { text } => {
                content.push(serde_json::json!({ "text": text }));
            }
            crate::message::MessagePart::ToolCall { id, name, input } => {
                content.push(serde_json::json!({
                    "toolUse": {"toolUseId": id, "name": name, "input": input}
                }));
            }
            crate::message::MessagePart::ToolResult {
                call_id,
                output,
                is_error,
                ..
            } => {
                content.push(serde_json::json!({
                    "toolResult": {
                        "toolUseId": call_id,
                        "content": [{"text": output.to_string()}],
                        "status": if matches!(is_error, Some(true)) { "error" } else { "success" },
                    }
                }));
            }
            crate::message::MessagePart::Image { .. } => {}
        }
    }
    serde_json::json!({"role": role, "content": content})
}

/// Convert engine tool schemas into Converse `toolSpec` entries.
///
/// Each [`ToolSchema`](crate::tool::ToolSchema) maps to Bedrock's
/// `toolSpec` triple — `name`, `description`, and `inputSchema`
/// wrapping the JSON schema under a `json` key, the documented member
/// of Bedrock's `ToolInputSchema` shape.
fn convert_converse_tools(tools: &[crate::tool::ToolSchema]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "toolSpec": {
                    "name": t.tool,
                    "description": t.description,
                    "inputSchema": {"json": t.input_schema},
                }
            })
        })
        .collect()
}

/// Map a stop reason to the engine's [`StreamStopReason`].
///
/// Both Bedrock paths share Anthropic's stop vocabulary
/// (`end_turn`, `tool_use`, `max_tokens`, `stop_sequence`); anything
/// else — including Converse-only values such as `content_filtered` —
/// lands on [`EndTurn`](StreamStopReason::EndTurn), the safe default
/// for a completed turn.
fn anthropic_stop_to_engine(stop: &str) -> StreamStopReason {
    match stop {
        "max_tokens" => Stop::MaxTokens,
        "stop_sequence" => Stop::StopSequence,
        "tool_use" => Stop::ToolCall,
        _ => Stop::EndTurn,
    }
}

/// State machine translating a `ConverseStream` chunk sequence into
/// engine [`StreamEvent`]s.
///
/// Each Converse event (`messageStart`,
/// `contentBlockStart`, `contentBlockDelta`, `contentBlockStop`,
/// `messageStop`, `metadata`) arrives as a one-key JSON object. The
/// emitter maps that sequence onto the engine's part vocabulary:
/// `contentBlockStart` opens a lane, deltas append at the block index,
/// and the terminal pair — [`MessageDelta`] carrying the latched stop
/// reason and usage, then [`MessageStop`] — is emitted exactly once,
/// at the `metadata` event that ends a well-formed stream. A stream
/// that ends without `metadata` emits no terminal pair: the absence
/// of [`MessageStop`] is the truncation signal, the same contract the
/// direct Anthropic path's emitter holds.
///
/// [`MessageDelta`]: crate::stream::MessageDelta
/// [`MessageStop`]: crate::stream::MessageStop
#[derive(Debug, Default)]
struct ConverseStreamEmitter {
    /// Whether `messageStart` has been processed.
    ///
    /// Guards against emitting a second [`MessageStart`] if a
    /// duplicate chunk ever arrives.
    started: bool,

    /// Whether the terminal pair has been emitted.
    ///
    /// Set when `metadata` produces the [`MessageDelta`]+[`MessageStop`]
    /// pair so it happens exactly once per stream.
    ///
    /// [`MessageDelta`]: crate::stream::MessageDelta
    /// [`MessageStop`]: crate::stream::MessageStop
    finished: bool,

    /// The stop reason latched from `messageStop`.
    ///
    /// `ConverseStream` delivers the stop reason *before* the
    /// `metadata` event that carries usage; the latch lets the terminal
    /// [`MessageDelta`] report both together. `None` until
    /// `messageStop` arrives, or forever on a truncated stream.
    ///
    /// [`MessageDelta`]: crate::stream::MessageDelta
    stop_reason: Option<String>,

    /// Token usage latched from the `metadata` event.
    ///
    /// `None` until `metadata` arrives; carried by the terminal
    /// [`MessageDelta`].
    ///
    /// [`MessageDelta`]: crate::stream::MessageDelta
    usage: Option<crate::stream::Usage>,
}

impl ConverseStreamEmitter {
    /// Translate one `ConverseStream` chunk into engine events.
    ///
    /// Recognizes the six documented chunk kinds; anything else
    /// (future Converse events, guardrail trace detail) yields no
    /// events. `messageStop` only latches state; `metadata` emits the
    /// terminal [`MessageDelta`]+[`MessageStop`] pair. Once that pair
    /// is out, every further chunk is ignored — a desynced stream must
    /// not append parts to a message the consumer has already been
    /// told is complete.
    ///
    /// [`MessageDelta`]: crate::stream::MessageDelta
    /// [`MessageStop`]: crate::stream::MessageStop
    fn process_chunk(&mut self, chunk: &serde_json::Value) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        if chunk.get("messageStart").is_some() {
            if !self.started {
                self.started = true;
                out.push(StreamEvent::MessageStart(crate::stream::MessageStart {
                    message: crate::stream::MessageMetadata {
                        id: String::new(),
                        role: "assistant".to_string(),
                        model: String::new(),
                    },
                }));
            }
            return out;
        }
        if let Some(start) = chunk.get("contentBlockStart") {
            let index = block_index(start);
            let part = if let Some(tool) = start.pointer("/start/toolUse") {
                Some(crate::message::MessagePart::ToolCall {
                    id: tool
                        .get("toolUseId")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: tool
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input: serde_json::Value::Null,
                })
            } else if start.pointer("/start/reasoningContent").is_some() {
                None
            } else {
                Some(crate::message::MessagePart::text(""))
            };
            out.push(StreamEvent::PartStart(crate::stream::PartStart {
                index,
                part,
            }));
            return out;
        }
        if let Some(delta) = chunk.get("contentBlockDelta") {
            let index = block_index(delta);
            if let Some(text) = delta
                .pointer("/delta/text")
                .and_then(serde_json::Value::as_str)
            {
                out.push(indexed_text(index, text));
            }
            if let Some(fragment) = delta
                .pointer("/delta/toolUse/input")
                .and_then(serde_json::Value::as_str)
            {
                out.push(StreamEvent::IndexedDelta(crate::stream::IndexedDelta {
                    index,
                    delta: crate::stream::DeltaPart::InputJson {
                        partial_json: fragment.to_string(),
                    },
                }));
            }
            if let Some(reasoning) = delta
                .pointer("/delta/reasoningContent/text")
                .and_then(serde_json::Value::as_str)
            {
                out.push(StreamEvent::IndexedDelta(crate::stream::IndexedDelta {
                    index,
                    delta: crate::stream::DeltaPart::Thinking {
                        text: reasoning.to_string(),
                    },
                }));
            }
            return out;
        }
        if let Some(stop) = chunk.get("contentBlockStop") {
            out.push(StreamEvent::PartStop {
                index: Some(block_index(stop)),
            });
            return out;
        }
        if let Some(reason) = chunk
            .pointer("/messageStop/stopReason")
            .and_then(serde_json::Value::as_str)
        {
            self.stop_reason = Some(reason.to_string());
            return out;
        }
        if chunk.get("metadata").is_some() {
            self.usage = chunk.pointer("/metadata/usage").map(converse_usage);
            out.extend(self.terminal());
        }
        out
    }

    /// Build the terminal event pair, once per stream.
    ///
    /// Called at `metadata`; subsequent calls return nothing. The
    /// [`MessageDelta`] carries whatever stop reason and usage were
    /// latched — both fields absent on a stream truncated before
    /// `messageStop`/`metadata` delivered them.
    ///
    /// [`MessageDelta`]: crate::stream::MessageDelta
    fn terminal(&mut self) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        vec![
            StreamEvent::MessageDelta(crate::stream::MessageDelta {
                delta: crate::stream::MessageDeltaPayload {
                    stop_reason: self.stop_reason.take(),
                },
                usage: self.usage.take(),
            }),
            StreamEvent::MessageStop,
        ]
    }
}

/// Read `contentBlockIndex` from a Converse block event.
///
/// Defaults to lane 0 when the field is absent or not a number, so a
/// malformed chunk still routes somewhere instead of dropping content.
fn block_index(event: &serde_json::Value) -> usize {
    event
        .get("contentBlockIndex")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(0)
}

/// Per-stream decoding state for the response body, both invoke paths.
///
/// Owned by the `stream_messages` loop: raw body bytes go in via
/// [`process_bytes`](Self::process_bytes) and engine events come out;
/// [`finish`](Self::finish) flushes the terminal events once the body
/// ends. Holding the decoder and both emitters in one place keeps the
/// frame → event → engine pipeline a single testable unit.
struct BedrockStreamState {
    /// Frame decoder over the raw body bytes.
    ///
    /// Buffers partial frames between `process_bytes` calls; every
    /// complete frame routes by its framing headers.
    decoder: AwsEventStreamDecoder,

    /// Which translation the chunk payloads take.
    ///
    /// Fixed when the state is created — one state per response; it
    /// selects which emitter the chunk payloads feed and which
    /// terminal flush [`finish`](Self::finish) applies.
    path: BedrockPath,

    /// Event translation for the Anthropic-native path.
    ///
    /// Idle when `path` is [`Converse`](BedrockPath::Converse).
    anthropic_emitter: crate::provider::anthropic::StreamEmitter,

    /// Event translation for the Converse path.
    ///
    /// Idle when `path` is [`Anthropic`](BedrockPath::Anthropic).
    converse_emitter: ConverseStreamEmitter,
}

impl BedrockStreamState {
    /// Create the state for one response stream on the given path.
    ///
    /// Both emitters start empty regardless of `path`; only the one
    /// matching the path is ever fed, so the unused one stays inert.
    fn new(path: BedrockPath) -> Self {
        Self {
            decoder: AwsEventStreamDecoder::default(),
            path,
            anthropic_emitter: crate::provider::anthropic::StreamEmitter::default(),
            converse_emitter: ConverseStreamEmitter::default(),
        }
    }

    /// Feed raw response-body bytes; returns the decoded engine events.
    ///
    /// Every decoded frame routes by framing headers: exception frames
    /// surface as an error naming the exception, non-`chunk` frames
    /// (the initial response) are skipped, and `chunk` payloads go to
    /// the path's emitter.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] for exception frames and for chunk
    /// payloads that are not valid JSON.
    fn process_bytes(&mut self, bytes: &[u8]) -> Result<Vec<StreamEvent>, ApiError> {
        let mut out = Vec::new();
        for event in self.decoder.push(bytes) {
            if event.message_type == "exception" || event.event_type == "exception" {
                let detail = String::from_utf8_lossy(&event.payload);
                let name = if event.exception_type.is_empty() {
                    "exception"
                } else {
                    event.exception_type.as_str()
                };
                return Err(ApiError::api(format!(
                    "bedrock: model stream error ({name}): {detail}"
                )));
            }
            if event.event_type != "chunk" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_slice(&event.payload)
                .map_err(|e| ApiError::api(format!("bedrock: chunk parse: {e}")))?;
            match self.path {
                BedrockPath::Anthropic => {
                    let kind = json
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.anthropic_emitter.process_event(&kind, Some(json));
                    out.extend(self.anthropic_emitter.drain());
                }
                BedrockPath::Converse => {
                    out.extend(self.converse_emitter.process_chunk(&json));
                }
            }
        }
        Ok(out)
    }

    /// Flush the terminal events at body end.
    ///
    /// The Anthropic path drains its emitter (surfacing any recorded
    /// `error` event); the Converse path emitted its terminal pair at
    /// the `metadata` event or is truncated, which the missing
    /// [`StreamEvent::MessageStop`] already signals.
    ///
    /// # Errors
    ///
    /// Returns the Anthropic emitter's recorded terminal error, if any.
    fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        match self.path {
            BedrockPath::Anthropic => self.anthropic_emitter.finish(),
            BedrockPath::Converse => Ok(Vec::new()),
        }
    }
}

/// Build an [`IndexedDelta`] text fragment for a block index.
///
/// Converse text deltas arrive whole (one string per chunk), so each
/// becomes one delta event; no re-chunking is needed.
fn indexed_text(index: usize, text: &str) -> StreamEvent {
    StreamEvent::IndexedDelta(crate::stream::IndexedDelta {
        index,
        delta: crate::stream::DeltaPart::Text {
            text: text.to_string(),
        },
    })
}

/// Map a Converse `usage` object to the engine [`Usage`].
///
/// Converse reports camelCase token counts (`inputTokens`,
/// `outputTokens`); missing or oversized values default to 0 rather
/// than erroring mid-stream.
fn converse_usage(usage: &serde_json::Value) -> crate::stream::Usage {
    crate::stream::Usage {
        input_tokens: usage
            .get("inputTokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        output_tokens: usage
            .get("outputTokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
    }
}

impl std::fmt::Debug for BedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockClient")
            .field("region", &self.region)
            .field("model", &self.model)
            .field("path", &self.path)
            .field("max_tokens", &self.max_tokens)
            .field("credentials", &"<redacted>")
            .finish_non_exhaustive()
    }
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

    /// Swap the `SigV4` credential triple on a live client.
    ///
    /// For short-lived credentials (STS assumed roles, EKS IRSA, SSO
    /// sessions) whose lifetime is shorter than the process: the swap
    /// replaces key id, secret, and session token as one unit, so a
    /// request signing concurrently with the swap sees one consistent
    /// generation, never a mixed pair. A rejected swap (empty values)
    /// leaves the previous credentials in place.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] (classified `ConfigMissing`) when the key
    /// id or secret is empty.
    pub fn set_credentials(
        &self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Result<(), ApiError> {
        let access_key_id = require_non_empty("access_key_id", Some(access_key_id.into()))?;
        let secret_access_key =
            require_non_empty("secret_access_key", Some(secret_access_key.into()))?;
        *crate::error::recover_guard(self.credentials.write()) = Credentials {
            access_key_id,
            secret_access_key,
            session_token,
        };
        Ok(())
    }

    /// Snapshot the current credential generation.
    ///
    /// One clone under the read lock; request paths sign with this
    /// snapshot so an entire request belongs to a single generation
    /// even if credentials rotate mid-flight.
    fn credentials(&self) -> Credentials {
        crate::error::recover_guard(self.credentials.read()).clone()
    }

    /// The Bedrock runtime host for the configured region.
    ///
    /// `bedrock-runtime.{region}.amazonaws.com` — the value both the
    /// request URLs and the `SigV4` `host` header are built from.
    fn host(&self) -> String {
        format!("bedrock-runtime.{}.amazonaws.com", self.region)
    }

    /// The invoke path to use for a model id.
    ///
    /// The builder's explicit override (see
    /// [`path`](BedrockClientBuilder::path)) wins; otherwise the id's
    /// prefix selects the path via [`auto_path`]. Evaluated per request
    /// against the current model, so a mid-run model switch across
    /// vendors — a fallback chain, for example — keeps the request body
    /// and endpoint consistent with the model.
    fn effective_path(&self, model: &str) -> BedrockPath {
        self.path.unwrap_or_else(|| auto_path(model))
    }

    /// Sign and POST a request body, returning the raw response.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] on transport failure.
    async fn signed_post(&self, url: &str, body: &[u8]) -> Result<reqwest::Response, ApiError> {
        let credentials = self.credentials();
        let request = apply_sigv4(
            self.http.post(url),
            url,
            &self.region,
            &credentials.access_key_id,
            &credentials.secret_access_key,
            credentials.session_token.as_deref(),
            body,
        )?;
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
        let path = self.effective_path(&model);
        let body = match path {
            BedrockPath::Anthropic => anthropic_body(request, self.max_tokens),
            BedrockPath::Converse => converse_body(request, self.max_tokens),
        };
        let url = stream_url(&self.region, &model, path);
        let client = self.http.clone();
        let region = self.region.clone();
        let credentials = self.credentials();

        Box::pin(async_stream::try_stream! {
            let bytes = serde_json::to_vec(&body)
                .map_err(|e| ApiError::api(format!("bedrock: serialize: {e}")))?;
            let req = apply_sigv4(
                client.post(&url),
                &url,
                &region,
                &credentials.access_key_id,
                &credentials.secret_access_key,
                credentials.session_token.as_deref(),
                &bytes,
            )?;
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
            let mut state = BedrockStreamState::new(path);
            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| {
                    ApiError::api(format!("bedrock: stream read: {e}"))
                })?;
                for ev in state.process_bytes(bytes.as_ref())? {
                    yield ev;
                }
            }
            for ev in state.finish()? {
                yield ev;
            }
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
        let path = self.effective_path(&model);
        let body = match path {
            BedrockPath::Anthropic => anthropic_body(request, self.max_tokens),
            BedrockPath::Converse => converse_body(request, self.max_tokens),
        };
        let url = invoke_url(&self.region, &model, path);
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
/// array and the Converse output structure — and maps each shape's
/// text and tool-use blocks to the engine's message parts, plus
/// usage and stop reason.
///
/// # Errors
///
/// Returns an [`ApiError`] when the response matches neither shape.
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
            if let Some(tool) = block.get("toolUse") {
                parts.push(crate::message::MessagePart::ToolCall {
                    id: tool
                        .get("toolUseId")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: tool
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input: tool.get("input").cloned().unwrap_or(serde_json::json!({})),
                });
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
    use crate::message::{Message, MessagePart};

    /// The default budget body-builder tests exercise.
    const BUDGET: u32 = crate::provider::anthropic::DEFAULT_MAX_TOKENS;
    /// Build one event-stream frame (for tests).
    #[allow(clippy::cast_possible_truncation, clippy::unreadable_literal)]
    fn build_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        // :message-type = "event"
        headers.extend_from_slice(b"\x0d:message-type\x07");
        headers.extend_from_slice(&5u16.to_be_bytes());
        headers.extend_from_slice(b"event");
        // :event-type
        headers.extend_from_slice(b"\x0b:event-type\x07");
        headers.extend_from_slice(&(event_type.len() as u16).to_be_bytes());
        headers.extend_from_slice(event_type.as_bytes());
        // :content-type
        headers.extend_from_slice(b"\x0c:content-type\x07");
        headers.extend_from_slice(&16u16.to_be_bytes());
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

    /// Build one exception event-stream frame (for tests) —
    /// `:message-type: exception` plus `:exception-type`, no
    /// `:event-type`, as real AWS exception frames arrive.
    #[allow(clippy::cast_possible_truncation)]
    fn build_exception_frame(exception_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        headers.extend_from_slice(b"\x0d:message-type\x07");
        headers.extend_from_slice(&9u16.to_be_bytes());
        headers.extend_from_slice(b"exception");
        headers.extend_from_slice(b"\x0f:exception-type\x07");
        headers.extend_from_slice(&(exception_type.len() as u16).to_be_bytes());
        headers.extend_from_slice(exception_type.as_bytes());

        let total = 12usize
            .saturating_add(headers.len())
            .saturating_add(payload.len())
            .saturating_add(4);
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&headers);
        out.extend_from_slice(payload);
        out.extend_from_slice(&0u32.to_be_bytes());
        out
    }

    // A known SigV4 test vector (the AWS-documented example request).
    // The expected authorization below was computed independently (a
    // from-spec Python implementation) for this exact tuple; the
    // signature hex is the pin — any change to the canonical-request
    // layout, header set, or HMAC chain breaks it.
    #[test]
    fn sigv4_known_vector() {
        let headers = sigv4_sign(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "bedrock-runtime.us-east-1.amazonaws.com",
            "/model/test/invoke",
            b"{}",
            // 2015-08-30T12:36:00Z
            std::time::SystemTime::UNIX_EPOCH
                .checked_add(std::time::Duration::from_secs(1_440_938_160))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        );
        assert_eq!(
            headers.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/bedrock/aws4_request, \
             SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, \
             Signature=72851c9c080d2a817323fa59da0a21c9ffe8141b57e30ec956834db9467c39e4"
        );
        assert_eq!(headers.amz_date, "20150830T123600Z");

        let debug = format!("{headers:?}");
        assert!(debug.contains("<redacted>"), "{debug}");
        assert!(
            !debug.contains("AKIDEXAMPLE")
                && !debug
                    .contains("72851c9c080d2a817323fa59da0a21c9ffe8141b57e30ec956834db9467c39e4"),
            "the signing output's Debug carries neither the key id nor the \
             signature: {debug}"
        );
    }

    // The AWS general reference's derive-signing-key example: for
    // (20150830, us-east-1, iam) under the documented example secret,
    // the signing key is the hex constant below. External truth for the
    // HMAC chain itself, independent of our canonical-request layout.
    #[test]
    fn sigv4_signing_key_matches_the_documented_example() {
        let key = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            "iam",
        );
        assert_eq!(
            hex::encode(key),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        );
    }

    // The STS variant: a session token adds a fifth signed header and
    // changes the canonical request. Expected value computed by the
    // same independent from-spec implementation as the no-token
    // vector above.
    #[test]
    fn sigv4_session_token_vector() {
        let headers = sigv4_sign(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("AQoEXAMPLEHROAZ8EXAMPLETOKEN"),
            "us-east-1",
            "bedrock-runtime.us-east-1.amazonaws.com",
            "/model/test/invoke",
            b"{}",
            std::time::SystemTime::UNIX_EPOCH
                .checked_add(std::time::Duration::from_secs(1_440_938_160))
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        );
        assert_eq!(
            headers.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/bedrock/aws4_request, \
             SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token, \
             Signature=9ce96e944e65159a25532face326f1d424a45f9d94e9851c2770e6677f8fbf02"
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
        assert_eq!(
            auto_path(
                "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-v1:0"
            ),
            BedrockPath::Converse,
            "ARN ids do not carry the anthropic. prefix — Converse is the safe route"
        );
        assert_eq!(
            auto_path("us.anthropic.claude-sonnet-4-5-v1:0"),
            BedrockPath::Converse,
            "inference-profile prefixes are not the native anthropic. prefix — \
             they route to Converse, which is also the only API that serves them"
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
        let mut emitter = crate::provider::anthropic::StreamEmitter::default();
        let process = |emitter: &mut crate::provider::anthropic::StreamEmitter,
                       chunk: &serde_json::Value|
         -> Vec<StreamEvent> {
            let kind = chunk
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            emitter.process_event(&kind, Some(chunk.clone()));
            emitter.drain()
        };

        let events = process(
            &mut emitter,
            &serde_json::json!({
                "type": "message_start",
                "message": {"id": "msg_1", "model": "claude", "usage": {"input_tokens": 10}}
            }),
        );
        assert!(matches!(events[0], StreamEvent::MessageStart(_)));

        let events = process(
            &mut emitter,
            &serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        );
        assert!(matches!(
            &events[0],
            StreamEvent::IndexedDelta(d) if matches!(&d.delta,
                crate::stream::DeltaPart::Text { text } if text == "hi")
        ));

        process(
            &mut emitter,
            &serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 5}
            }),
        );
    }

    #[test]
    fn anthropic_stream_sequence_reaches_engine_accumulator() {
        let mut emitter = crate::provider::anthropic::StreamEmitter::default();
        let feed = |emitter: &mut crate::provider::anthropic::StreamEmitter,
                    chunk: &serde_json::Value|
         -> Vec<StreamEvent> {
            let kind = chunk
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            emitter.process_event(&kind, Some(chunk.clone()));
            emitter.drain()
        };

        let mut events: Vec<StreamEvent> = [
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude", "usage": {"input_tokens": 4}}}),
            serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "echo"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1}"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}}),
            serde_json::json!({"type": "message_stop"}),
        ]
        .iter()
        .flat_map(|chunk| feed(&mut emitter, chunk))
        .collect();
        events.extend(emitter.finish().unwrap_or_default());

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let usage = acc.usage().copied();
        let message = acc.build();
        assert_eq!(
            message.tool_call_parts(),
            vec![("t1", "echo", &serde_json::json!({"a": 1}))],
            "the bedrock anthropic path must emit PartStart/IndexedDelta/PartStop, \
             not bare deltas the engine accumulator drops"
        );
        assert_eq!(
            usage,
            Some(crate::stream::Usage {
                input_tokens: 4,
                output_tokens: 6
            })
        );
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
        let body = anthropic_body(&request, BUDGET);
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert!(body["tools"].is_array());
    }

    #[test]
    fn converse_body_builds_native_content_blocks() {
        let request = StreamRequest {
            messages: vec![Message::user("hello"), Message::assistant("hi")],
            system: Some("sys".to_string()),
            tools: None,
        };
        let body = converse_body(&request, BUDGET);
        assert_eq!(
            body["messages"][0],
            serde_json::json!({"role": "user", "content": [{"text": "hello"}]}),
            "user text is one typed text block, not a nested provider message"
        );
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"][0]["text"], "hi");
        assert_eq!(
            body["system"],
            serde_json::json!([{ "text": "sys" }]),
            "converse puts system as a list of text blocks"
        );
    }

    #[test]
    fn converse_body_maps_tool_calls_and_results() {
        let request = StreamRequest {
            messages: vec![
                Message::new(
                    crate::message::Role::Assistant,
                    vec![
                        MessagePart::text("thinking out loud"),
                        MessagePart::tool_call("t1", "echo", serde_json::json!({"a": 1})),
                    ],
                ),
                Message::new(
                    crate::message::Role::User,
                    vec![MessagePart::tool_result("t1", "echo", "ran fine", false)],
                ),
                Message::new(
                    crate::message::Role::User,
                    vec![MessagePart::tool_result("t2", "echo", "boom", true)],
                ),
            ],
            system: None,
            tools: None,
        };
        let body = converse_body(&request, BUDGET);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "mixed parts become ordered blocks");
        assert_eq!(blocks[0]["text"], "thinking out loud");
        assert_eq!(
            blocks[1],
            serde_json::json!({
                "toolUse": {"toolUseId": "t1", "name": "echo", "input": {"a": 1}}
            })
        );
        assert_eq!(
            body["messages"][1]["content"][0],
            serde_json::json!({
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"text": "ran fine"}],
                    "status": "success"
                }
            }),
            "tool results ride a user message as a toolResult block"
        );
        assert_eq!(
            body["messages"][2]["content"][0]["toolResult"]["status"],
            "error"
        );
    }

    #[test]
    fn converse_body_carries_tool_config() {
        let with_tools = StreamRequest {
            messages: vec![Message::user("hi")],
            system: None,
            tools: Some(vec![crate::tool::ToolSchema {
                tool: "echo".to_string(),
                description: "Echo".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }]),
        };
        let body = converse_body(&with_tools, BUDGET);
        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"],
            serde_json::json!({
                "name": "echo",
                "description": "Echo",
                "inputSchema": {"json": {"type": "object"}}
            })
        );

        let without = converse_body(&StreamRequest::new(vec![Message::user("hi")]), BUDGET);
        assert!(
            without.get("toolConfig").is_none(),
            "no tools → no toolConfig: {without}"
        );
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
                    "content": [
                        {"text": "converse says hi"},
                        {"toolUse": {"toolUseId": "t1", "name": "echo", "input": {"a": 1}}}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 2, "outputTokens": 4}
        });
        let response = bedrock_non_streaming_response(&json, "m").unwrap();
        assert_eq!(response.message.text_content(), "converse says hi");
        assert_eq!(
            response.message.tool_call_parts(),
            vec![("t1", "echo", &serde_json::json!({"a": 1}))],
            "toolUse blocks in the Converse output are not dropped"
        );
        assert_eq!(
            response.stop_reason,
            crate::stream::StreamStopReason::ToolCall
        );
        assert_eq!(response.usage.unwrap().output_tokens, 4);
    }

    #[test]
    fn event_stream_decoder_flags_exception_frames() {
        let frame = build_exception_frame("throttling", br#"{"message":"too fast"}"#);
        let mut decoder = AwsEventStreamDecoder::default();
        let events = decoder.push(&frame);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].message_type, "exception");
        assert_eq!(
            events[0].exception_type, "throttling",
            "the exception's name survives decoding for the error message"
        );
        assert_eq!(
            events[0].event_type, "",
            "exception frames have no :event-type"
        );
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
        assert_eq!(
            client.effective_path(&client.model()),
            BedrockPath::Anthropic
        );

        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("amazon.nova-pro-v1")
            .build()
            .unwrap();
        assert_eq!(
            client.effective_path(&client.model()),
            BedrockPath::Converse
        );
    }

    #[test]
    fn path_follows_model_across_set_model() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("anthropic.claude-sonnet-4-5-v1:0")
            .build()
            .unwrap();
        assert!(client.set_model("amazon.nova-pro-v1"));
        assert_eq!(
            client.effective_path(&client.model()),
            BedrockPath::Converse,
            "a fallback-style model switch across vendors must not keep \
             sending the anthropic body to an anthropic-only route"
        );
    }

    #[test]
    fn endpoint_urls_embed_region_and_model() {
        assert_eq!(
            stream_url("us-east-1", "anthropic.claude-v1:0", BedrockPath::Anthropic),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v1%3A0/invoke-with-response-stream",
            "colons in model ids are percent-encoded, as the AWS SDKs send them"
        );
        assert_eq!(
            invoke_url("eu-west-1", "m", BedrockPath::Anthropic),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/m/invoke"
        );
        assert_eq!(
            stream_url("us-east-1", "amazon.nova-pro-v1:0", BedrockPath::Converse),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-pro-v1%3A0/converse-stream",
            "ConverseStream keeps the model id in the URI path"
        );
        assert_eq!(
            invoke_url("ap-south-1", "any", BedrockPath::Converse),
            "https://bedrock-runtime.ap-south-1.amazonaws.com/model/any/converse",
            "Converse keeps the model id in the URI path"
        );
        assert_eq!(
            invoke_url(
                "us-east-1",
                "arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-v1:0",
                BedrockPath::Anthropic
            ),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Afoundation-model%2Fanthropic.claude-v1%3A0/invoke",
            "ARN model ids keep their structure as one encoded segment"
        );
    }

    #[test]
    fn converse_stream_text_sequence_reaches_engine_accumulator() {
        let mut emitter = ConverseStreamEmitter::default();
        let events: Vec<StreamEvent> = [
            serde_json::json!({"messageStart": {"role": "assistant"}}),
            serde_json::json!({"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}),
            serde_json::json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "he"}}}),
            serde_json::json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "llo"}}}),
            serde_json::json!({"contentBlockStop": {"contentBlockIndex": 0}}),
            serde_json::json!({"messageStop": {"stopReason": "end_turn"}}),
            serde_json::json!({"metadata": {"usage": {"inputTokens": 3, "outputTokens": 5}}}),
        ]
        .iter()
        .flat_map(|chunk| emitter.process_chunk(chunk))
        .collect();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStart(_))),
            "messageStart emits a MessageStart: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::IndexedDelta(d) if d.index == 0
                    && matches!(&d.delta, crate::stream::DeltaPart::Text { text } if text == "he"))),
            "text deltas arrive under /contentBlockDelta/delta/text: {events:?}"
        );
        let delta_pos = events
            .iter()
            .position(|e| matches!(e, StreamEvent::MessageDelta(_)))
            .unwrap();
        let stop_pos = events
            .iter()
            .position(|e| matches!(e, StreamEvent::MessageStop))
            .unwrap();
        assert!(delta_pos < stop_pos, "MessageDelta precedes MessageStop");

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let usage = acc.usage().copied();
        let message = acc.build();
        assert_eq!(message.text_content(), "hello");
        assert_eq!(
            usage,
            Some(crate::stream::Usage {
                input_tokens: 3,
                output_tokens: 5
            })
        );
    }

    #[test]
    fn converse_stream_tool_call_sequence_reaches_engine_accumulator() {
        let mut emitter = ConverseStreamEmitter::default();
        let events: Vec<StreamEvent> = [
            serde_json::json!({"messageStart": {"role": "assistant"}}),
            serde_json::json!({"contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "t1", "name": "echo"}}
            }}),
            serde_json::json!({"contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "{\"a\":"}}
            }}),
            serde_json::json!({"contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "1}"}}
            }}),
            serde_json::json!({"contentBlockStop": {"contentBlockIndex": 1}}),
            serde_json::json!({"messageStop": {"stopReason": "tool_use"}}),
            serde_json::json!({"metadata": {"usage": {"inputTokens": 9, "outputTokens": 2}}}),
        ]
        .iter()
        .flat_map(|chunk| emitter.process_chunk(chunk))
        .collect();

        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::PartStart(p) if p.index == 1
                    && matches!(&p.part, Some(MessagePart::ToolCall { id, name, .. })
                        if id == "t1" && name == "echo"))),
            "contentBlockStart/start/toolUse announces id and name: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::IndexedDelta(d) if d.index == 1
                    && matches!(&d.delta, crate::stream::DeltaPart::InputJson { partial_json }
                        if partial_json == "{\"a\":"))),
            "tool input fragments arrive under /contentBlockDelta/delta/toolUse/input: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::MessageDelta(d) if d.delta.stop_reason.as_deref() == Some("tool_use"))),
            "messageStop's stopReason reaches the terminal MessageDelta: {events:?}"
        );

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let message = acc.build();
        assert_eq!(
            message.tool_call_parts(),
            vec![("t1", "echo", &serde_json::json!({"a": 1}))],
            "the engine reassembles the tool call from the event sequence"
        );
    }

    #[test]
    fn converse_stream_metadata_emits_terminal_pair_once() {
        let mut emitter = ConverseStreamEmitter::default();
        for chunk in [
            serde_json::json!({"messageStop": {"stopReason": "end_turn"}}),
            serde_json::json!({"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}),
            serde_json::json!({"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}}),
        ] {
            emitter.process_chunk(&chunk);
        }
        let repeated = emitter.process_chunk(&serde_json::json!({
            "metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}
        }));
        assert!(
            repeated.is_empty(),
            "the terminal pair is emitted exactly once, at metadata"
        );
    }

    #[test]
    fn converse_stream_without_metadata_emits_no_terminal_pair() {
        let mut emitter = ConverseStreamEmitter::default();
        let events = emitter.process_chunk(&serde_json::json!({
            "messageStop": {"stopReason": "end_turn"}
        }));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageDelta(_) | StreamEvent::MessageStop)),
            "a stream cut before metadata signals truncation by the absence of \
             the terminal pair, it does not fake completion"
        );
    }

    #[test]
    fn converse_stream_reasoning_routes_to_the_thinking_lane() {
        let mut emitter = ConverseStreamEmitter::default();
        let events: Vec<StreamEvent> = [
            serde_json::json!({"messageStart": {"role": "assistant"}}),
            serde_json::json!({"contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"reasoningContent": {"reasoningText": {"text": ""}}}
            }}),
            serde_json::json!({"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "pondering"}}
            }}),
            serde_json::json!({"contentBlockStop": {"contentBlockIndex": 0}}),
        ]
        .iter()
        .flat_map(|chunk| emitter.process_chunk(chunk))
        .collect();

        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::PartStart(p) if p.index == 0 && p.part.is_none())),
            "a reasoningContent block opens the thinking lane (part: None): {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::IndexedDelta(d) if matches!(&d.delta,
                    crate::stream::DeltaPart::Thinking { text } if text == "pondering"))),
            "reasoning deltas arrive as Thinking, not dropped: {events:?}"
        );

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        assert!(
            acc.build().text_content().is_empty(),
            "reasoning does not leak into the visible message text"
        );
    }

    #[test]
    fn converse_stream_ignores_chunks_after_metadata() {
        let mut emitter = ConverseStreamEmitter::default();
        emitter.process_chunk(&serde_json::json!({
            "metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}
        }));
        for late in [
            serde_json::json!({"contentBlockStart": {
                "contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "t9", "name": "x"}}
            }}),
            serde_json::json!({"contentBlockDelta": {
                "contentBlockIndex": 0, "delta": {"text": "late"}
            }}),
            serde_json::json!({"messageStart": {"role": "assistant"}}),
        ] {
            assert!(
                emitter.process_chunk(&late).is_empty(),
                "nothing may follow the terminal pair: {late}"
            );
        }
    }

    #[test]
    fn converse_stream_late_message_stop_after_metadata_is_ignored() {
        let mut emitter = ConverseStreamEmitter::default();
        emitter.process_chunk(&serde_json::json!({
            "metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}
        }));
        let trailing = emitter.process_chunk(&serde_json::json!({
            "messageStop": {"stopReason": "end_turn"}
        }));
        assert!(
            trailing.is_empty(),
            "a messageStop arriving after the terminal pair adds nothing"
        );
    }

    #[test]
    fn converse_body_without_system_omits_field() {
        let request = StreamRequest {
            messages: vec![Message::user("hello")],
            system: None,
            tools: None,
        };
        let body = converse_body(&request, BUDGET);
        assert!(
            body.get("system").is_none(),
            "no system → field omitted: {body}"
        );

        let blank = converse_body(
            &StreamRequest {
                messages: vec![Message::user("hello")],
                system: Some(String::new()),
                tools: None,
            },
            BUDGET,
        );
        assert!(
            blank.get("system").is_none(),
            "empty-string system is dropped, not an empty text block: {blank}"
        );
    }

    #[test]
    fn anthropic_body_without_system_or_tools_minimizes() {
        let request = StreamRequest::new(vec![Message::user("hi")]);
        let body = anthropic_body(&request, BUDGET);
        assert!(
            body.get("system").is_none(),
            "no system → field omitted, not null: {body}"
        );
        assert!(
            body.get("tools").is_none(),
            "no tools → field omitted; Anthropic rejects an empty array: {body}"
        );
        assert_eq!(
            body["max_tokens"],
            crate::provider::anthropic::DEFAULT_MAX_TOKENS,
            "the bedrock path shares the direct path's output budget"
        );
    }

    #[test]
    fn bodies_fold_inline_system_messages() {
        let inline_system = Message::new(
            crate::message::Role::System,
            vec![MessagePart::text("be terse")],
        );
        let request = StreamRequest {
            messages: vec![inline_system, Message::user("hi")],
            system: Some("override".to_string()),
            tools: None,
        };
        let anthropic = anthropic_body(&request, BUDGET);
        assert_eq!(
            anthropic["system"], "override\nbe terse",
            "inline system messages join the top-level system, direct-path style"
        );
        assert_eq!(anthropic["messages"].as_array().map(Vec::len), Some(1));

        let converse = converse_body(&request, BUDGET);
        assert_eq!(converse["system"][0]["text"], "override\nbe terse");
        assert_eq!(converse["messages"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn converse_body_uses_the_shared_token_budget() {
        let body = converse_body(&StreamRequest::new(vec![Message::user("hi")]), BUDGET);
        assert_eq!(
            body["inferenceConfig"]["maxTokens"],
            crate::provider::anthropic::DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    fn stop_reason_mapping_covers_both_vocabularies() {
        assert_eq!(anthropic_stop_to_engine("max_tokens"), Stop::MaxTokens);
        assert_eq!(
            anthropic_stop_to_engine("stop_sequence"),
            Stop::StopSequence
        );
        assert_eq!(anthropic_stop_to_engine("tool_use"), Stop::ToolCall);
        assert_eq!(anthropic_stop_to_engine("end_turn"), Stop::EndTurn);
        for converse_only in ["content_filtered", "guardrail_intervened", "refusal"] {
            assert_eq!(
                anthropic_stop_to_engine(converse_only),
                Stop::EndTurn,
                "Converse-only values land on the completed-turn default"
            );
        }
    }

    #[test]
    fn parse_headers_decodes_boolean_and_skips_unknown_value_types() {
        // ":flagA" with value type 0 (true), ":flagB" with type 1
        // (false), ":weird" with an undocumented type — values carry
        // no bytes for non-string types.
        let bytes: Vec<u8> = b"\x06:flagA\x00\x06:flagB\x01\x06:weird\x09".to_vec();
        let headers = parse_headers(&bytes);
        assert_eq!(
            headers,
            vec![
                (":flagA".to_string(), "true".to_string()),
                (":flagB".to_string(), "false".to_string()),
                (":weird".to_string(), String::new()),
            ]
        );
    }

    #[test]
    fn converse_usage_partial_object_defaults_missing_side_to_zero() {
        let mut emitter = ConverseStreamEmitter::default();
        let events = emitter.process_chunk(&serde_json::json!({
            "metadata": {"usage": {"inputTokens": 7}}
        }));
        assert!(
            events.first().is_some_and(|e| matches!(e,
            StreamEvent::MessageDelta(d)
                if d.usage == Some(crate::stream::Usage {
                    input_tokens: 7,
                    output_tokens: 0,
                }))),
            "a usage object reporting only one side zeroes the other: {events:?}"
        );
    }

    #[test]
    fn endpoint_urls_encode_non_ascii_model_bytes() {
        assert_eq!(
            stream_url("us-east-1", "ünïcode", BedrockPath::Converse),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/%C3%BCn%C3%AFcode/converse-stream",
            "multibyte UTF-8 encodes per byte, per RFC 3986"
        );
    }

    #[test]
    fn amz_date_clamps_pre_epoch_clocks_to_1970() {
        let before_epoch = std::time::SystemTime::UNIX_EPOCH
            .checked_sub(std::time::Duration::from_secs(10))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        assert_eq!(format_amz_date(before_epoch), "19700101T000000Z");
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
    fn signed_request_marks_credential_headers_sensitive() {
        let builder = apply_sigv4(
            reqwest::Client::new()
                .post("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            "us-east-1",
            "AKID",
            "secret",
            Some("session-token"),
            b"{}",
        )
        .unwrap();
        let request = builder.build().unwrap();
        let headers = request.headers();
        assert!(
            headers
                .get(reqwest::header::AUTHORIZATION)
                .is_some_and(reqwest::header::HeaderValue::is_sensitive),
            "Authorization must never surface in debug output"
        );
        assert!(
            headers
                .get("x-amz-security-token")
                .is_some_and(reqwest::header::HeaderValue::is_sensitive),
            "the session token must never surface in debug output"
        );
        assert!(
            !headers
                .get("x-amz-date")
                .is_some_and(reqwest::header::HeaderValue::is_sensitive),
            "the timestamp is not a secret"
        );
    }

    #[test]
    fn signed_request_rejects_control_characters_in_credentials() {
        let err = apply_sigv4(
            reqwest::Client::new()
                .post("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            "us-east-1",
            "AKID",
            "secret",
            Some("bad\ntoken"),
            b"{}",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("session token"),
            "the error names the offending header: {err}"
        );
    }

    #[test]
    fn event_stream_decoder_continues_after_skipped_frame() {
        // A structurally invalid frame (headers_len overruns total_len)
        // whose declared extent is trustworthy is skipped alone; the
        // good frame behind it in the same buffer must still decode.
        let mut bad = Vec::new();
        bad.extend_from_slice(&16u32.to_be_bytes());
        bad.extend_from_slice(&100u32.to_be_bytes());
        bad.extend_from_slice(&0u32.to_be_bytes());
        bad.extend_from_slice(&[0u8; 4]);
        let mut wire = bad;
        wire.extend_from_slice(&build_frame("chunk", br#"{"ok":true}"#));

        let mut decoder = AwsEventStreamDecoder::default();
        let events = decoder.push(&wire);
        assert_eq!(
            events.len(),
            1,
            "the good frame after the discarded one still decodes"
        );
        assert_eq!(events[0].event_type, "chunk");
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
    fn stream_state_decodes_a_full_converse_wire() {
        let wire: Vec<u8> = [
            build_frame("initial-response", b"{}"),
            build_frame("chunk", br#"{"messageStart":{"role":"assistant"}}"#),
            build_frame(
                "chunk",
                br#"{"contentBlockStart":{"contentBlockIndex":0,"start":{}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"he"}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"llo"}}}"#,
            ),
            build_frame("chunk", br#"{"contentBlockStop":{"contentBlockIndex":0}}"#),
            build_frame("chunk", br#"{"messageStop":{"stopReason":"end_turn"}}"#),
            build_frame(
                "chunk",
                br#"{"metadata":{"usage":{"inputTokens":3,"outputTokens":5}}}"#,
            ),
        ]
        .concat();

        let mut state = BedrockStreamState::new(BedrockPath::Converse);
        let split = wire.len() / 2;
        let mut events = state.process_bytes(&wire[..split]).unwrap();
        events.extend(state.process_bytes(&wire[split..]).unwrap());
        assert!(
            events
                .first()
                .is_some_and(|e| matches!(e, StreamEvent::MessageStart(_))),
            "the initial-response frame yields nothing; messageStart comes first: {events:?}"
        );
        assert!(
            state.finish().unwrap().is_empty(),
            "the Converse terminal pair came from metadata, not from finish"
        );

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let usage = acc.usage().copied();
        let message = acc.build();
        assert_eq!(message.text_content(), "hello");
        assert_eq!(
            usage,
            Some(crate::stream::Usage {
                input_tokens: 3,
                output_tokens: 5
            })
        );
    }

    #[test]
    fn stream_state_decodes_a_full_anthropic_wire() {
        let wire: Vec<u8> = [
            build_frame(
                "chunk",
                br#"{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":4}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"echo"}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}"#,
            ),
            build_frame("chunk", br#"{"type":"content_block_stop","index":0}"#),
            build_frame(
                "chunk",
                br#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}"#,
            ),
            build_frame("chunk", br#"{"type":"message_stop"}"#),
        ]
        .concat();

        let mut state = BedrockStreamState::new(BedrockPath::Anthropic);
        let mut events = state.process_bytes(&wire).unwrap();
        events.extend(state.finish().unwrap());

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let usage = acc.usage().copied();
        let message = acc.build();
        assert_eq!(
            message.tool_call_parts(),
            vec![("t1", "echo", &serde_json::json!({"a": 1}))],
            "the full frame wire reassembles the tool call"
        );
        assert_eq!(
            usage,
            Some(crate::stream::Usage {
                input_tokens: 4,
                output_tokens: 6
            })
        );
    }

    #[test]
    fn converse_stream_duplicate_message_start_emits_one() {
        let mut emitter = ConverseStreamEmitter::default();
        let first =
            emitter.process_chunk(&serde_json::json!({"messageStart": {"role": "assistant"}}));
        let second =
            emitter.process_chunk(&serde_json::json!({"messageStart": {"role": "assistant"}}));
        assert!(matches!(first.as_slice(), [StreamEvent::MessageStart(_)]));
        assert!(second.is_empty(), "a duplicate messageStart yields nothing");
    }

    #[test]
    fn converse_stream_metadata_without_usage_still_terminates() {
        let mut emitter = ConverseStreamEmitter::default();
        emitter.process_chunk(&serde_json::json!({"messageStop": {"stopReason": "max_tokens"}}));
        let events = emitter.process_chunk(&serde_json::json!({"metadata": {}}));
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::MessageDelta(d), StreamEvent::MessageStop]
                    if d.usage.is_none() && d.delta.stop_reason.as_deref() == Some("max_tokens")
            ),
            "metadata without usage still closes the stream, with no usage claim: {events:?}"
        );
    }

    #[test]
    fn converse_stream_oversized_token_counts_default_to_zero() {
        let mut emitter = ConverseStreamEmitter::default();
        let events = emitter.process_chunk(&serde_json::json!({
            "metadata": {"usage": {
                "inputTokens": 18_446_744_073_709_551_615_u64,
                "outputTokens": 4_294_967_296_u64
            }}
        }));
        assert!(
            events.first().is_some_and(|e| matches!(e,
            StreamEvent::MessageDelta(d)
                if d.usage == Some(crate::stream::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                }))),
            "counts beyond u32 clamp to 0 rather than truncating: {events:?}"
        );
    }

    #[test]
    fn stream_state_decodes_mixed_blocks_wire() {
        let wire: Vec<u8> = [
            build_frame("chunk", br#"{"messageStart":{"role":"assistant"}}"#),
            build_frame(
                "chunk",
                br#"{"contentBlockStart":{"contentBlockIndex":0,"start":{}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"let me check"}}}"#,
            ),
            build_frame("chunk", br#"{"contentBlockStop":{"contentBlockIndex":0}}"#),
            build_frame(
                "chunk",
                br#"{"contentBlockStart":{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"t1","name":"echo"}}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{}"}}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":9,"delta":{"text":"orphan"}}}"#,
            ),
            build_frame("chunk", br#"{"contentBlockStop":{"contentBlockIndex":1}}"#),
            build_frame("chunk", br#"{"messageStop":{"stopReason":"tool_use"}}"#),
            build_frame(
                "chunk",
                br#"{"metadata":{"usage":{"inputTokens":2,"outputTokens":3}}}"#,
            ),
        ]
        .concat();

        let mut state = BedrockStreamState::new(BedrockPath::Converse);
        let events = state.process_bytes(&wire).unwrap();
        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        let message = acc.build();
        assert_eq!(message.text_content(), "let me check");
        assert_eq!(
            message.tool_call_parts(),
            vec![("t1", "echo", &serde_json::json!({}))],
            "text and tool blocks in one stream both reassemble; the orphan \
             delta at an unopened index drops harmlessly"
        );
    }

    #[test]
    fn stream_state_decodes_empty_responses_on_both_paths() {
        // A model may return no content at all: the wire is just the
        // start/terminal events. Both paths must complete cleanly and
        // the engine accumulator must build an empty message.
        let converse_wire: Vec<u8> = [
            build_frame("chunk", br#"{"messageStart":{"role":"assistant"}}"#),
            build_frame("chunk", br#"{"messageStop":{"stopReason":"end_turn"}}"#),
            build_frame(
                "chunk",
                br#"{"metadata":{"usage":{"inputTokens":1,"outputTokens":0}}}"#,
            ),
        ]
        .concat();
        let mut state = BedrockStreamState::new(BedrockPath::Converse);
        let events = state.process_bytes(&converse_wire).unwrap();
        assert!(
            events
                .last()
                .is_some_and(|e| matches!(e, StreamEvent::MessageStop)),
            "an empty response still terminates: {events:?}"
        );

        let anthropic_wire: Vec<u8> = [
            build_frame(
                "chunk",
                br#"{"type":"message_start","message":{"id":"m","model":"claude","usage":{"input_tokens":1}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}"#,
            ),
            build_frame("chunk", br#"{"type":"message_stop"}"#),
        ]
        .concat();
        let mut state = BedrockStreamState::new(BedrockPath::Anthropic);
        let mut events = state.process_bytes(&anthropic_wire).unwrap();
        events.extend(state.finish().unwrap());
        assert!(
            events
                .last()
                .is_some_and(|e| matches!(e, StreamEvent::MessageStop)),
            "the anthropic path terminates via finish: {events:?}"
        );

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        assert!(acc.build().text_content().is_empty());
    }

    #[test]
    fn stream_state_truncated_converse_wire_emits_no_terminal() {
        let wire: Vec<u8> = [
            build_frame("chunk", br#"{"messageStart":{"role":"assistant"}}"#),
            build_frame(
                "chunk",
                br#"{"contentBlockStart":{"contentBlockIndex":0,"start":{}}}"#,
            ),
            build_frame(
                "chunk",
                br#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"partial"}}}"#,
            ),
        ]
        .concat();

        let mut state = BedrockStreamState::new(BedrockPath::Converse);
        let mut events = state.process_bytes(&wire).unwrap();
        events.extend(state.finish().unwrap());
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageDelta(_) | StreamEvent::MessageStop)),
            "a wire cut before messageStop/metadata must not fake completion"
        );

        let mut acc = crate::stream::StreamAccumulator::new();
        for ev in &events {
            acc.process(ev).unwrap();
        }
        assert!(
            acc.build().text_content().is_empty(),
            "content held in a lane the wire never closed is not committed — \
             that discard is the engine's truncation semantics, on top of the \
             missing terminal pair"
        );
    }

    #[test]
    fn stream_state_anthropic_error_frame_fails_at_finish() {
        let wire = build_frame(
            "chunk",
            br#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );
        let mut state = BedrockStreamState::new(BedrockPath::Anthropic);
        assert!(
            state.process_bytes(&wire).unwrap().is_empty(),
            "the error event records; it does not emit mid-stream"
        );
        let err = state.finish().unwrap_err();
        assert!(
            err.to_string().contains("overloaded_error"),
            "finish surfaces the recorded error by name: {err}"
        );
    }

    #[test]
    fn stream_state_surfaces_exception_frames() {
        let mut state = BedrockStreamState::new(BedrockPath::Converse);
        let wire = [
            build_frame("chunk", br#"{"messageStart":{"role":"assistant"}}"#),
            build_exception_frame("throttlingException", br#"{"message":"too fast"}"#),
        ]
        .concat();
        let err = state
            .process_bytes(&wire)
            .expect_err("an exception frame must fail the stream");
        assert!(
            err.to_string().contains("throttlingException"),
            "the error names the exception: {err}"
        );
    }

    #[test]
    fn event_stream_decoder_decodes_across_random_splits() {
        let frames: Vec<Vec<u8>> = ["one", "two", "three"]
            .iter()
            .map(|word| build_frame("chunk", word.as_bytes()))
            .collect();
        let mut wire: Vec<u8> = frames.concat();
        for _ in 0..64 {
            // Deterministic enough for CI: shuffle bytes into random-size
            // chunks, feed the decoder, every frame must come out whole.
            let mut decoder = AwsEventStreamDecoder::default();
            let mut payloads = Vec::new();
            while !wire.is_empty() {
                let cut = (fastrand::usize(0..wire.len()) + 1).min(wire.len());
                let (head, tail) = wire.split_at(cut);
                for event in decoder.push(head) {
                    payloads.push(String::from_utf8_lossy(&event.payload).into_owned());
                }
                wire = tail.to_vec();
            }
            for event in decoder.push(&[]) {
                payloads.push(String::from_utf8_lossy(&event.payload).into_owned());
            }
            assert_eq!(payloads, ["one", "two", "three"]);
            wire = frames.concat();
        }
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
        assert_eq!(
            client.effective_path("anthropic.claude-v1:0"),
            BedrockPath::Converse,
            "the explicit override pins the path regardless of model"
        );
    }

    #[cfg(all(feature = "bedrock", feature = "testing"))]
    #[test]
    fn from_env_reads_aws_variables() {
        use crate::testing::EnvGuard;

        let env = EnvGuard::acquire(&[
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_BEDROCK_MODEL",
        ]);
        env.set("AWS_REGION", "us-east-1");
        env.set("AWS_ACCESS_KEY_ID", "k");
        env.set("AWS_SECRET_ACCESS_KEY", "s");
        env.remove("AWS_SESSION_TOKEN");
        env.remove("AWS_BEDROCK_MODEL");
        let client = BedrockClient::from_env().unwrap();
        assert_eq!(
            client.model(),
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "the documented Sonnet default"
        );

        env.set("AWS_BEDROCK_MODEL", "amazon.nova-pro-v1:0");
        let client = BedrockClient::from_env().unwrap();
        assert_eq!(client.model(), "amazon.nova-pro-v1:0");
        assert_eq!(
            client.effective_path("amazon.nova-pro-v1:0"),
            BedrockPath::Converse
        );

        env.set("AWS_SESSION_TOKEN", "tok");
        BedrockClient::from_env().unwrap();

        env.set("AWS_REGION", "bad region");
        let err = BedrockClient::from_env().unwrap_err();
        assert!(
            err.to_string().contains("region"),
            "a malformed region fails at build, not at the first request: {err}"
        );

        env.remove("AWS_ACCESS_KEY_ID");
        let err = BedrockClient::from_env().unwrap_err();
        assert!(err.to_string().contains("AWS_ACCESS_KEY_ID"), "{err}");
    }

    #[test]
    fn set_credentials_swaps_for_new_requests() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k1")
            .secret_access_key("s1")
            .model("m")
            .build()
            .unwrap();
        let creds = client.credentials();
        assert_eq!(creds.access_key_id, "k1");
        assert_eq!(creds.secret_access_key, "s1");
        assert!(creds.session_token.is_none());

        client
            .set_credentials("k2", "s2", Some("t2".to_string()))
            .unwrap();
        let creds = client.credentials();
        assert_eq!(creds.access_key_id, "k2");
        assert_eq!(creds.secret_access_key, "s2");
        assert_eq!(creds.session_token.as_deref(), Some("t2"));
    }

    #[test]
    fn set_credentials_clears_the_session_token() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k1")
            .secret_access_key("s1")
            .session_token("t1")
            .model("m")
            .build()
            .unwrap();
        assert_eq!(client.credentials().session_token.as_deref(), Some("t1"));

        client.set_credentials("k2", "s2", None).unwrap();
        assert!(
            client.credentials().session_token.is_none(),
            "a rotation without a token clears the old one — temporary \
             credentials bind all three values to one generation"
        );
    }

    #[test]
    fn concurrent_set_credentials_leave_a_matched_pair() {
        use std::sync::Arc;

        let client = Arc::new(
            BedrockClientBuilder::default()
                .region("us-east-1")
                .access_key_id("k0")
                .secret_access_key("s0")
                .model("m")
                .build()
                .unwrap(),
        );
        let mut writers = Vec::new();
        for i in 1..=8 {
            let writer = Arc::clone(&client);
            writers.push(std::thread::spawn(move || {
                writer
                    .set_credentials(format!("k{i}"), format!("s{i}"), None)
                    .unwrap();
            }));
        }
        for writer in writers {
            writer.join().unwrap();
        }
        let creds = client.credentials();
        let generation = creds.access_key_id.strip_prefix('k').unwrap_or_default();
        assert_eq!(
            creds.secret_access_key,
            format!("s{generation}"),
            "racing writers serialize — the final state is one whole generation"
        );
    }

    #[test]
    fn credentials_recover_from_lock_poison() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k0")
            .secret_access_key("s0")
            .model("m")
            .build()
            .unwrap();

        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = client.credentials.write().unwrap();
            panic!("poison the credentials lock");
        }));
        std::panic::set_hook(previous_hook);
        assert!(poisoned.is_err(), "the lock is now poisoned");

        assert_eq!(
            client.credentials().access_key_id,
            "k0",
            "Category 1 policy: snapshots recover from poison"
        );
        client.set_credentials("k9", "s9", None).unwrap();
        assert_eq!(
            client.credentials().access_key_id,
            "k9",
            "so do swaps — a poisoned lock degrades nothing here"
        );
    }

    #[test]
    fn builder_full_configuration_composes() {
        let client = BedrockClientBuilder::default()
            .region("us-gov-west-1")
            .access_key_id("k")
            .secret_access_key("s")
            .session_token("t")
            .model("anthropic.claude-v1:0")
            .path(BedrockPath::Converse)
            .max_tokens(1024)
            .build()
            .unwrap();
        assert_eq!(client.max_tokens, 1024);
        assert_eq!(
            client.effective_path("anthropic.claude-v1:0"),
            BedrockPath::Converse,
            "the override pins the path over the model's own prefix"
        );
        assert_eq!(client.model(), "anthropic.claude-v1:0");
        assert_eq!(
            client.base_url(),
            "https://bedrock-runtime.us-gov-west-1.amazonaws.com"
        );
        assert_eq!(client.credentials().session_token.as_deref(), Some("t"));
    }

    #[test]
    fn set_credentials_validates_non_empty_values() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .build()
            .unwrap();

        let Err(err) = client.set_credentials("", "s", None) else {
            panic!("an empty access key id must fail the swap");
        };
        assert!(err.to_string().contains("access_key_id"), "{err}");
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigMissing,
            "{err}"
        );

        let Err(err) = client.set_credentials("k", "", None) else {
            panic!("an empty secret must fail the swap");
        };
        assert!(err.to_string().contains("secret_access_key"), "{err}");
        // the previous credentials survive a rejected swap
        assert_eq!(client.credentials().access_key_id, "k");
    }

    #[test]
    fn set_credentials_swaps_never_expose_torn_pairs() {
        use std::sync::Arc;

        let client = Arc::new(
            BedrockClientBuilder::default()
                .region("us-east-1")
                .access_key_id("k0")
                .secret_access_key("s0")
                .model("m")
                .build()
                .unwrap(),
        );
        let mut readers = Vec::new();
        for _ in 0..3 {
            let reader = Arc::clone(&client);
            readers.push(std::thread::spawn(move || {
                for _ in 0..3000 {
                    let creds = reader.credentials();
                    assert!(
                        creds.access_key_id.starts_with('k')
                            && creds.secret_access_key.starts_with('s')
                            && creds.access_key_id[1..] == creds.secret_access_key[1..],
                        "torn pair observed: {:?} / {:?}",
                        creds.access_key_id,
                        creds.secret_access_key
                    );
                }
            }));
        }
        for i in 1..=100 {
            client
                .set_credentials(format!("k{i}"), format!("s{i}"), None)
                .unwrap();
        }
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[test]
    fn builder_debug_redacts_credentials() {
        let builder = BedrockClient::builder()
            .region("us-east-1")
            .access_key_id("AKIA_SECRETTOKEN")
            .secret_access_key("wJalrXUtnFEMI_SUPERSECRET")
            .session_token("SESSION_SUPERSECRET")
            .model("m");
        let debug = format!("{builder:?}");
        assert!(debug.contains("<redacted>"), "{debug}");
        for secret in [
            "AKIA_SECRETTOKEN",
            "wJalrXUtnFEMI_SUPERSECRET",
            "SESSION_SUPERSECRET",
        ] {
            assert!(!debug.contains(secret), "{debug}");
        }
        assert!(
            debug.contains("us-east-1"),
            "non-secret configuration stays visible: {debug}"
        );
    }

    #[test]
    fn debug_output_redacts_all_credentials() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("AKIA_SECRETTOKEN")
            .secret_access_key("wJalrXUtnFEMI_SUPERSECRET")
            .session_token("SESSION_SUPERSECRET")
            .model("m")
            .build()
            .unwrap();
        let debug = format!("{client:?}");
        assert!(debug.contains("<redacted>"), "{debug}");
        for secret in [
            "AKIA_SECRETTOKEN",
            "wJalrXUtnFEMI_SUPERSECRET",
            "SESSION_SUPERSECRET",
        ] {
            assert!(!debug.contains(secret), "{debug}");
        }
    }

    #[test]
    fn builder_max_tokens_defaults_to_the_engine_constant() {
        let client = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .build()
            .unwrap();
        assert_eq!(
            client.max_tokens,
            crate::provider::anthropic::DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    fn builder_rejects_zero_max_tokens() {
        let err = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .max_tokens(0)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("max_tokens"), "{err}");
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigValidationError,
            "{err}"
        );
    }

    #[test]
    fn bodies_honor_the_configured_token_budget() {
        let request = StreamRequest::new(vec![Message::user("hi")]);
        assert_eq!(anthropic_body(&request, 512)["max_tokens"], 512);
        assert_eq!(
            converse_body(&request, 512)["inferenceConfig"]["maxTokens"],
            512
        );
    }

    #[test]
    fn builder_rejects_non_aws_region_shapes() {
        for bad in [
            "x.evil.com",
            "us east 1",
            "US-EAST-1",
            "us/east",
            "us\teast",
        ] {
            let err = BedrockClientBuilder::default()
                .region(bad)
                .access_key_id("k")
                .secret_access_key("s")
                .model("m")
                .build()
                .unwrap_err();
            assert!(err.to_string().contains("region"), "{bad:?}: {err}");
            assert_eq!(
                err.code(),
                crate::api::error::ErrorCode::ConfigValidationError,
                "{bad:?}: {err}"
            );
        }

        for good in ["us-east-1", "cn-north-1", "us-gov-west-1", "us-iso-east-1"] {
            BedrockClientBuilder::default()
                .region(good)
                .access_key_id("k")
                .secret_access_key("s")
                .model("anthropic.claude-v1:0")
                .build()
                .unwrap_or_else(|e| panic!("valid region rejected: {good:?}: {e}"));
        }
    }

    #[test]
    fn bodies_drop_empty_messages() {
        let request = StreamRequest {
            messages: vec![
                Message::new(crate::message::Role::Assistant, vec![]),
                Message::user("hi"),
            ],
            system: None,
            tools: None,
        };
        let anthropic = anthropic_body(&request, BUDGET);
        assert_eq!(
            anthropic["messages"].as_array().map(Vec::len),
            Some(1),
            "zero-part messages must not serialize to an empty content array"
        );
        let converse = converse_body(&request, BUDGET);
        assert_eq!(converse["messages"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            converse["messages"][0]["content"].as_array().map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn builder_rejects_empty_strings() {
        let err = BedrockClientBuilder::default()
            .region("")
            .access_key_id("k")
            .secret_access_key("s")
            .model("m")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("region"));
        assert_eq!(
            err.code(),
            crate::api::error::ErrorCode::ConfigMissing,
            "{err}"
        );

        let err = BedrockClientBuilder::default()
            .region("us-east-1")
            .access_key_id("k")
            .secret_access_key("s")
            .model("")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("model"));
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
        let mut emitter = crate::provider::anthropic::StreamEmitter::default();
        emitter.process_event("future_event", Some(serde_json::json!({"x": 1})));
        assert!(
            emitter.drain().is_empty(),
            "unknown types produce no events"
        );
        assert!(
            emitter.finish().unwrap_or_default().is_empty(),
            "an unknown type neither errors nor fakes a stop"
        );
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
