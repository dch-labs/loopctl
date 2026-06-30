//! API and infrastructure error types.
//!
//! Error hierarchy for LLM API interactions, tool execution,
//! configuration handling, and general infrastructure operations. Every
//! failure mode an agent can encounter is captured by [`ApiError`], with
//! a corresponding [`ErrorCode`] for programmatic matching, logging,
//! and metrics.
//!
//! # Error Categories
//!
//! Errors are organized into numeric ranges so that monitoring dashboards
//! and alert rules can filter by category without matching individual variants:
//!
//! | Category | Code Range | Variants                                                |
//! |----------|------------|---------------------------------------------------------|
//! | API      | 1000–1005  | Request, response, rate limit, timeout, stream, context |
//! | Auth     | 1100–1101  | Invalid key, auth failed                                |
//! | HTTP     | 1200–1202  | Connection, request, response                           |
//! | Tool     | 1300–1304  | Not found, execution, permission, input, timeout        |
//! | Config   | 1400–1403  | File not found, parse, validation, missing              |
//! | I/O      | 1500–1502  | File not found, read, write                             |
//! | JSON     | 1600       | Parse                                                   |
//! | Internal | 1700       | Internal error                                          |
//! | Signal   | 1999       | Interrupted                                             |
//!
//! # Provided Types
//!
//! - **[`ApiError`]** — The main error enum with ergonomic constructors
//!   like [`ApiError::api`], [`ApiError::tool`], and [`ApiError::config`].
//! - **[`ErrorCode`]** — A machine-readable code derived from an  [`ApiError`]
//!   via [`ApiError::code`].
//! - **[`Result<T>`]** — A convenience alias for `std::result::Result<T, ApiError>`.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::api::error::{ApiError, ErrorCode};
//!
//! // Construct errors with the ergonomic helpers
//! let err = ApiError::api("request failed");
//! assert_eq!(err.code(), ErrorCode::ApiRequestFailed);
//! assert!(err.is_retryable());
//!
//! // Pattern-match on the error kind
//! match err {
//!     ApiError::Api(msg) => println!("API failure: {msg}"),
//!     _ => unreachable!(),
//! }
//! ```

use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;

/// Machine-readable error codes for programmatic error handling.
///
/// Each [`ErrorCode`] variant maps to a stable numeric value (see the
/// module-level table) designed for logging pipelines, metrics dashboards,
/// and cross-language interoperability. Codes are derived automatically
/// from an [`ApiError`] via [`ApiError::code`] — you rarely construct
/// them directly.
///
/// # Stability
///
/// The numeric values are part of the public API and will not change
/// across semver-compatible releases. New codes may be appended within
/// existing category ranges.
///
/// # Example
///
/// ```rust
/// use loopctl::api::error::{ApiError, ErrorCode};
///
/// let err = ApiError::api_rate_limited();
/// assert_eq!(err.code(), ErrorCode::ApiRateLimited);
/// assert_eq!(err.code() as u32, 1002);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u16)]
#[non_exhaustive]
pub enum ErrorCode {
    // ==================================================
    // API errors (1000-1005)
    // ==================================================
    /// A generic LLM API request failed.
    ///
    /// Returned when the API responds with an unexpected status or the
    /// request could not be completed for an unspecified reason.
    /// Maps to numeric code **1000**.
    ///
    /// Default code for [`ApiError::Api`] variants whose message does
    /// not match a more specific pattern (timeout, rate limit, stream,
    /// or context overflow).
    ///
    /// Use [`ApiError::api`] to construct errors that map to this code.
    ApiRequestFailed = 1000,

    /// The API response could not be parsed.
    ///
    /// Indicates the server returned data in an unexpected format — for
    /// example a changed JSON schema or a truncated body. Maps to **1001**.
    ///
    /// This error is **not** retryable; the response format is unlikely
    /// to change on subsequent attempts. Investigate version
    /// compatibility or schema changes.
    ApiResponseInvalid = 1001,

    /// The API rate limit was exceeded.
    ///
    /// The caller should back off and retry after the period indicated by
    /// the provider's headers. [`ApiError::is_retryable`] returns `true`.
    /// Maps to **1002**.
    ApiRateLimited = 1002,

    /// The API request timed out.
    ///
    /// The request took longer than the configured timeout. Safe to retry
    /// (possibly with a longer timeout). Maps to **1003**.
    ApiTimeout = 1003,

    /// An error occurred while reading the SSE stream.
    ///
    /// The connection was established but the stream was interrupted or
    /// produced malformed events. Maps to **1004**.
    ApiStreamError = 1004,

    /// The request exceeded the model's context window.
    ///
    /// The combined prompt `max_tokens` exceeds the model's limit. The
    /// agent should reduce the context (e.g. via summarisation or
    /// truncation) before retrying. Maps to **1005**.
    ApiContextOverflow = 1005,

    // ==================================================
    // Auth errors (1100-1101)
    // ==================================================
    /// The API key is invalid or missing.
    ///
    /// The key may be expired, revoked, or not yet set. Check
    /// environment variables and configuration files. Maps to **1100**.
    AuthInvalidKey = 1100,

    /// Authentication failed for a non-key-related reason.
    ///
    /// Covers token-expiry, account-suspension, or other auth-layer
    /// rejections that are not specifically about an invalid key.
    /// Maps to **1101**.
    AuthFailed = 1101,

    // ==================================================
    // HTTP errors (1200-1202)
    // ==================================================
    /// A low-level HTTP connection error.
    ///
    /// Typically a DNS failure, TCP timeout, or TLS handshake error.
    /// Maps to **1200**.
    ///
    /// Constructed via [`ApiError::http`] or [`ApiError::from_hyper`].
    /// This code is considered retryable by [`ApiError::is_retryable`]
    /// because connection issues are often transient.
    HttpConnectionError = 1200,

    /// An HTTP request construction or send error.
    ///
    /// The request was malformed or could not be serialised before
    /// reaching the server. Maps to **1201**.
    ///
    /// This code is retryable — a resend may succeed if the root cause
    /// was a transient serialisation glitch.
    HttpRequestError = 1201,

    /// An HTTP response error (non-success status).
    ///
    /// The server responded but with a status code outside the 2xx range.
    /// Maps to **1202**.
    ///
    /// Use [`ApiError::http_with_status`] to embed the status code in
    /// the message for downstream log aggregation.
    HttpResponseError = 1202,

    // ==================================================
    // Tool errors (1300-1304)
    // ==================================================
    /// The requested tool does not exist.
    ///
    /// The agent asked for a tool name that is not registered with the
    /// tool registry. Maps to **1300**.
    ///
    /// Constructed via [`ApiError::tool_not_found`]. Not retryable —
    /// the tool must be registered before use.
    ToolNotFound = 1300,

    /// Tool execution failed.
    ///
    /// The tool was found and invoked but returned an error during
    /// processing. Maps to **1301**.
    ///
    /// Constructed via [`ApiError::tool`] when the message does not
    /// match a more specific tool-error pattern.
    ToolExecutionFailed = 1301,

    /// Permission denied for tool execution.
    ///
    /// The agent is not authorised to run this tool, or the tool's
    /// sandbox policy rejected the operation. Maps to **1302**.
    ///
    /// Constructed via [`ApiError::tool_permission`]. Not retryable —
    /// permissions must be updated before the operation can succeed.
    ToolPermissionDenied = 1302,

    /// Tool input validation failed.
    ///
    /// The arguments provided to the tool did not satisfy its schema.
    /// Maps to **1303**.
    ///
    /// Constructed via [`ApiError::tool_input_invalid`]. Not retryable —
    /// the caller must fix the arguments.
    ToolInputInvalid = 1303,

    /// Tool execution timed out.
    ///
    /// The tool ran longer than its configured time limit. Maps to **1304**.
    ///
    /// May be retryable if the timeout was caused by a transient slowdown,
    /// but the caller should consider increasing the timeout first.
    ToolTimeout = 1304,

    // ==================================================
    // Config errors (1400-1403)
    // ==================================================
    /// The configuration file could not be found.
    ///
    /// The path pointed to by the config setting does not exist or is
    /// not readable. Maps to **1400**.
    ConfigFileNotFound = 1400,

    /// The configuration file could not be parsed.
    ///
    /// The file exists but contains invalid TOML / JSON / YAML. Maps
    /// to **1401**.
    ConfigParseError = 1401,

    /// Configuration values failed semantic validation.
    ///
    /// The file parsed correctly but a value is out of range or
    /// semantically invalid (e.g. negative timeout). Maps to **1402**.
    ConfigValidationError = 1402,

    /// A required configuration value is missing entirely.
    ///
    /// No value was provided and no default is available. Maps to **1403**.
    ConfigMissing = 1403,

    // ==================================================
    // I/O errors (1500-1502)
    // ==================================================
    /// A required file was not found on disk.
    ///
    /// Similar to [`ConfigFileNotFound`] but for general I/O paths.
    /// Maps to **1500**.
    ///
    /// [`ConfigFileNotFound`]: ErrorCode::ConfigFileNotFound
    IoFileNotFound = 1500,

    /// A file read operation failed.
    ///
    /// Permission denied, I/O error, or unexpected EOF. Maps to **1501**.
    IoReadError = 1501,

    /// A file write operation failed.
    ///
    /// Disk full, permission denied, or path invalid. Maps to **1502**.
    IoWriteError = 1502,

    // ==================================================
    // JSON errors (1600)
    // ==================================================
    /// A JSON payload could not be parsed.
    ///
    /// The response body or request payload contained malformed JSON.
    /// Maps to **1600**.
    JsonParseError = 1600,

    // ==================================================
    // Internal (1700)
    // ==================================================
    /// An internal / unexpected error.
    ///
    /// Catch-all for bugs or unclassifiable failures. If you see this
    /// code in production please file a bug report. Maps to **1700**.
    InternalError = 1700,

    // ==================================================
    // Signal (1999)
    // ==================================================
    /// The operation was interrupted by a user signal (e.g. Ctrl-C).
    ///
    /// Not an error per se — the agent loop checks for this code to
    /// perform graceful shutdown. Maps to **1999**.
    Interrupted = 1999,
}

/// Main error type for API and infrastructure operations.
///
/// Covers all failure modes an agent might encounter when
/// interacting with LLM APIs, executing tools, or managing configuration
/// and I/O. Each variant stores a human-readable message (or a source
/// error via `#[from]`) and can be mapped to a stable [`ErrorCode`] via
/// [`ApiError::code`].
///
/// # Construction
///
/// Prefer the ergonomic constructor methods over enum variants directly:
///
/// ```rust
/// use loopctl::api::error::{ApiError, ErrorCode};
/// // Instead of ApiError::Api("...".into())
/// let err = ApiError::api("request failed");
///
/// // Contextual constructors embed structured messages
/// let err = ApiError::tool_not_found("Bash");
/// let err = ApiError::config_not_found("/etc/loopctl.toml");
/// ```
///
/// # Error Classification
///
/// The [`ApiError::code`] method inspects the error message to select the
/// most specific [`ErrorCode`] available. For example, an [`ApiError::Api`]
/// variant whose message contains `"rate limit"` will map to
/// [`ErrorCode::ApiRateLimited`] rather than the generic
/// [`ErrorCode::ApiRequestFailed`].
///
/// # Retry Guidance
///
/// Use [`ApiError::is_retryable`] to decide whether to retry an operation:
///
/// ```rust,ignore
/// if error.is_retryable() {
///     tokio::time::sleep(backoff).await;
///     continue;
/// }
/// ```
///
/// # Variants
///
/// Each variant maps to a category of [`ErrorCode`]. See the individual
/// variant documentation for details on when each is produced and how
/// the message text influences code selection.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ApiError {
    /// An error returned by the LLM API provider.
    ///
    /// Covers request failures, rate limits, timeouts, streaming errors,
    /// and context-overflow conditions. The message is inspected by
    /// [`ApiError::code`] to select the most specific [`ErrorCode`].
    #[error("API error: {0}")]
    Api(String),

    /// An authentication or authorisation error.
    ///
    /// Invalid API keys, expired tokens, or account-level restrictions.
    /// The message text determines whether [`ErrorCode::AuthInvalidKey`]
    /// or [`ErrorCode::AuthFailed`] is returned by [`ApiError::code`].
    #[error("Authentication error: {0}")]
    Auth(String),

    /// An HTTP-level transport error.
    ///
    /// Connection failures, DNS errors, TLS handshake problems, or
    /// non-success status codes. Created via [`ApiError::http`] or
    /// [`ApiError::http_with_status`].
    #[error("HTTP error: {0}")]
    Http(String),

    /// A JSON serialisation / deserialisation error.
    ///
    /// Automatically converted from [`serde_json::Error`] via the
    /// `#[from]` attribute. Maps to [`ErrorCode::JsonParseError`].
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A general I/O error.
    ///
    /// Automatically converted from [`std::io::Error`] via the `#[from]`attribute.
    /// The [`ErrorCode`] is determined by inspecting the [`std::io::ErrorKind`]:
    ///
    /// - [`ErrorKind::NotFound`] → [`ErrorCode::IoFileNotFound`]
    /// - [`ErrorKind::PermissionDenied`] / [`ErrorKind::WriteZero`] → [`ErrorCode::IoWriteError`]
    /// - All other kinds → [`ErrorCode::IoReadError`]
    ///
    /// For explicit construction, use [`ApiError::io_not_found`], [`ApiError::io_read`],
    /// or [`ApiError::io_write`].
    ///
    /// [`ErrorKind::NotFound`]: std::io::ErrorKind::NotFound
    /// [`ErrorKind::PermissionDenied`]: std::io::ErrorKind::PermissionDenied
    /// [`ErrorKind::WriteZero`]: std::io::ErrorKind::WriteZero
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A tool execution error.
    ///
    /// Covers tool-not-found, execution failures, permission denials,
    /// invalid inputs, and timeouts. Prefer the specific constructors
    /// ([`ApiError::tool_not_found`], [`ApiError::tool_permission`],
    /// [`ApiError::tool_input_invalid`]) for richer context.
    #[error("Tool error: {0}")]
    Tool(String),

    /// A configuration error.
    ///
    /// File-not-found, parse failures, validation errors, or missing
    /// required values. Prefer the specific constructors
    /// ([`ApiError::config_not_found`], [`ApiError::config_validation`])
    /// for structured messages.
    #[error("Configuration error: {0}")]
    Config(String),

    /// The operation was interrupted by a user signal.
    ///
    /// Typically triggered by SIGINT. The agent loop checks for
    /// this variant to perform graceful shutdown. Maps to
    /// [`ErrorCode::Interrupted`].
    #[error("Interrupted")]
    Interrupted,

    /// A catch-all error for cases that don't fit another variant.
    ///
    /// Maps to [`ErrorCode::InternalError`]. Prefer a more specific
    /// variant or constructor when possible.
    #[error("Other: {0}")]
    Other(String),
}

impl ApiError {
    // ==================================================
    // Classifiers
    // ==================================================

    /// Derive the machine-readable [`ErrorCode`] for this error.
    ///
    /// Inspects the error variant and, for message-based variants
    /// ([`ApiError::Api`], [`ApiError::Tool`], [`ApiError::Config`]),
    /// parses the message text to select the most specific code
    /// available.
    ///
    /// # Classification Strategy
    ///
    /// ```text
    /// ApiError::Api(msg)
    ///   ├─ "timeout"                          → ApiTimeout
    ///   ├─ "rate limit" / "429"               → ApiRateLimited
    ///   ├─ "stream"                           → ApiStreamError
    ///   ├─ context keywords                   → ApiContextOverflow
    ///   └─ (default)                          → ApiRequestFailed
    ///
    /// ApiError::Auth(msg)
    ///   ├─ "invalid"                          → AuthInvalidKey
    ///   └─ (default)                          → AuthFailed
    ///
    /// ApiError::Tool(msg)
    ///   ├─ "not found"                        → ToolNotFound
    ///   ├─ "permission" / "denied"            → ToolPermissionDenied
    ///   ├─ "timeout"                          → ToolTimeout
    ///   ├─ "invalid"                          → ToolInputInvalid
    ///   └─ (default)                          → ToolExecutionFailed
    ///
    /// ApiError::Config(msg)
    ///   ├─ "not found"                        → ConfigFileNotFound
    ///   ├─ "validation"                       → ConfigValidationError
    ///   ├─ "missing"                          → ConfigMissing
    ///   └─ (default)                          → ConfigParseError
    ///
    /// ApiError::Http(msg)
    ///   ├─ "HTTP 5xx: ..."                    → HttpResponseError
    ///   ├─ "HTTP 4xx: ..."                    → HttpRequestError
    ///   └─ (default)                          → HttpConnectionError
    ///
    /// ApiError::Json(_)                       → JsonParseError
    /// ApiError::Io(_)                         → IoReadError
    /// ApiError::Interrupted                   → Interrupted
    /// ApiError::Other(_)                      → InternalError
    /// ```
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Api(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("timeout") {
                    ErrorCode::ApiTimeout
                } else if msg_lower.contains("rate limit") || msg_lower.contains("429") {
                    ErrorCode::ApiRateLimited
                } else if msg_lower.contains("stream") {
                    ErrorCode::ApiStreamError
                } else if Self::is_context_overflow_internal(msg) {
                    ErrorCode::ApiContextOverflow
                } else {
                    ErrorCode::ApiRequestFailed
                }
            }
            Self::Auth(msg) => {
                if msg.to_lowercase().contains("invalid") {
                    ErrorCode::AuthInvalidKey
                } else {
                    ErrorCode::AuthFailed
                }
            }
            Self::Http(msg) => {
                if let Some(rest) = msg.strip_prefix("HTTP ") {
                    if let Some(colon) = rest.find(':') {
                        if let Ok(status) = rest[..colon].parse::<u16>() {
                            return if status >= 500 {
                                ErrorCode::HttpResponseError
                            } else {
                                ErrorCode::HttpRequestError
                            };
                        }
                    }
                }
                ErrorCode::HttpConnectionError
            }
            Self::Json(_) => ErrorCode::JsonParseError,
            Self::Io(err) => {
                use std::io::ErrorKind;
                match err.kind() {
                    ErrorKind::NotFound => ErrorCode::IoFileNotFound,
                    ErrorKind::PermissionDenied | ErrorKind::WriteZero => ErrorCode::IoWriteError,
                    _ => ErrorCode::IoReadError,
                }
            }
            Self::Tool(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("not found") {
                    ErrorCode::ToolNotFound
                } else if msg_lower.contains("permission") || msg_lower.contains("denied") {
                    ErrorCode::ToolPermissionDenied
                } else if msg_lower.contains("timeout") {
                    ErrorCode::ToolTimeout
                } else if msg_lower.contains("invalid") {
                    ErrorCode::ToolInputInvalid
                } else {
                    ErrorCode::ToolExecutionFailed
                }
            }
            Self::Config(msg) => {
                let msg_lower = msg.to_lowercase();
                if msg_lower.contains("not found") {
                    ErrorCode::ConfigFileNotFound
                } else if msg_lower.contains("validation") {
                    ErrorCode::ConfigValidationError
                } else if msg_lower.contains("missing") {
                    ErrorCode::ConfigMissing
                } else {
                    ErrorCode::ConfigParseError
                }
            }
            Self::Interrupted => ErrorCode::Interrupted,
            Self::Other(_) => ErrorCode::InternalError,
        }
    }

    /// Check whether this error indicates a context-window overflow.
    ///
    /// Returns `true` when the error message contains keywords that
    /// strongly suggest the prompt exceeded the model's maximum context
    /// length — e.g. `"context"`, `"too many tokens"`, or
    /// `"context length"`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api("context length exceeded");
    /// assert!(err.is_context_overflow());
    ///
    /// let err = ApiError::api("server error");
    /// assert!(!err.is_context_overflow());
    /// ```
    #[must_use]
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::Api(msg) | Self::Other(msg) => Self::is_context_overflow_internal(msg),
            _ => false,
        }
    }

    /// Check a message string for context-overflow keywords.
    fn is_context_overflow_internal(msg: &str) -> bool {
        let msg_lower = msg.to_lowercase();
        msg_lower.contains("context")
            || msg_lower.contains("too many tokens")
            || msg_lower.contains("exceeds maximum")
            || msg_lower.contains("max tokens")
    }

    /// Check whether this error is safe to retry.
    ///
    /// Returns `true` for transient failures where a retry (possibly
    /// with back-off) has a reasonable chance of succeeding:
    ///
    /// [`ErrorCode::ApiRequestFailed`], [`ErrorCode::ApiRateLimited`],
    /// [`ErrorCode::ApiTimeout`], [`ErrorCode::HttpConnectionError`],
    /// [`ErrorCode::HttpRequestError`], and [`ErrorCode::HttpResponseError`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api_rate_limited();
    /// assert!(err.is_retryable());
    ///
    /// let err = ApiError::auth("bad key");
    /// assert!(!err.is_retryable());
    /// ```
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.code(),
            ErrorCode::ApiRequestFailed
                | ErrorCode::ApiRateLimited
                | ErrorCode::ApiTimeout
                | ErrorCode::HttpConnectionError
                | ErrorCode::HttpRequestError
                | ErrorCode::HttpResponseError
        )
    }

    /// Check whether this is an authentication error ([`ApiError::Auth`]).
    ///
    /// Useful for deciding whether to prompt the user for a new API key
    /// versus showing a generic error message.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::ApiError;
    /// let error = ApiError::auth("bad key");
    /// if error.is_auth_error() {
    ///     eprintln!("Please check your API key.");
    /// }
    /// ```
    #[must_use]
    pub const fn is_auth_error(&self) -> bool {
        matches!(self, Self::Auth(..))
    }

    /// Check whether this is a configuration error ([`ApiError::Config`]).
    ///
    /// Useful for directing the user to fix their config file rather
    /// than retrying.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::ApiError;
    /// let error = ApiError::config("bad config");
    /// if error.is_config_error() {
    ///     eprintln!("Configuration problem — check loopctl.toml.");
    /// }
    /// ```
    #[must_use]
    pub const fn is_config_error(&self) -> bool {
        matches!(self, Self::Config(..))
    }

    /// Check whether this is an I/O error ([`ApiError::Io`]).
    ///
    /// Returns `true` for file-system failures such as missing files,
    /// permission errors, or disk-full conditions.
    #[must_use]
    pub const fn is_io_error(&self) -> bool {
        matches!(self, Self::Io(..))
    }

    /// Check whether this is a tool execution error ([`ApiError::Tool`]).
    ///
    /// Returns `true` for any tool-related failure — not found,
    /// execution error, permission denied, invalid input, or timeout.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool_not_found("Bash");
    /// assert!(err.is_tool_error());
    /// ```
    #[must_use]
    pub const fn is_tool_error(&self) -> bool {
        matches!(self, Self::Tool(..))
    }

    // ==================================================
    // Constructors
    // ==================================================

    /// Create a generic [`ApiError::Api`] variant.
    ///
    /// Use this for LLM API failures that don't warrant a more specific
    /// constructor. The message will be inspected by [`ApiError::code`]
    /// to select the best [`ErrorCode`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api("unexpected 502 from upstream");
    /// ```
    pub fn api(msg: impl Into<String>) -> Self {
        Self::Api(msg.into())
    }

    /// Create a generic [`ApiError::Auth`] variant.
    ///
    /// The message is inspected by [`ApiError::code`]: if it contains
    /// `"invalid"` the code will be [`ErrorCode::AuthInvalidKey`],
    /// otherwise [`ErrorCode::AuthFailed`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::auth("token expired");
    /// assert_eq!(err.code(), ErrorCode::AuthFailed);
    /// ```
    pub fn auth(msg: impl Into<String>) -> Self {
        Self::Auth(msg.into())
    }

    /// Create an [`ApiError::Auth`] variant with an "Invalid API key" prefix.
    ///
    /// The prefixed message ensures [`ApiError::code`] returns
    /// [`ErrorCode::AuthInvalidKey`]. Use this when the provider
    /// explicitly rejects a key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::auth_invalid_key("key expired");
    /// assert_eq!(err.code(), ErrorCode::AuthInvalidKey);
    /// ```
    pub fn auth_invalid_key(msg: impl Into<String>) -> Self {
        Self::Auth(format!("Invalid API key: {}", msg.into()))
    }

    /// Create a generic [`ApiError::Http`] variant.
    ///
    /// Use for connection-level failures that are not tied to a specific
    /// HTTP status code. For status-specific errors prefer
    /// [`ApiError::http_with_status`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::http("DNS resolution failed");
    /// ```
    pub fn http(msg: impl Into<String>) -> Self {
        Self::Http(msg.into())
    }

    /// Create an [`ApiError::Http`] variant that includes the HTTP status code.
    ///
    /// Formats the message as `"HTTP {status}: {msg}"` so that downstream
    /// log aggregation can extract the status code. [`ApiError::code`]
    /// inspects the embedded status to select the appropriate [`ErrorCode`]:
    ///
    /// - `5xx` → [`ErrorCode::HttpResponseError`] (server error, retryable)
    /// - `4xx` → [`ErrorCode::HttpRequestError`] (client error, retryable)
    /// - Other / unparseable → [`ErrorCode::HttpConnectionError`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::http_with_status(503, "service unavailable");
    /// assert!(err.to_string().contains("HTTP 503"));
    /// assert_eq!(err.code(), ErrorCode::HttpResponseError);
    ///
    /// let err = ApiError::http_with_status(400, "bad request");
    /// assert_eq!(err.code(), ErrorCode::HttpRequestError);
    /// ```
    pub fn http_with_status(status: u16, msg: impl Into<String>) -> Self {
        Self::Http(format!("HTTP {status}: {}", msg.into()))
    }

    /// Create a generic [`ApiError::Tool`] variant.
    ///
    /// The message will be inspected by [`ApiError::code`] to select
    /// the most specific [`ErrorCode`] (not found, permission, timeout,
    /// invalid input, or generic execution failure).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool("execution failed");
    /// assert_eq!(err.code(), ErrorCode::ToolExecutionFailed);
    /// ```
    pub fn tool(msg: impl Into<String>) -> Self {
        Self::Tool(msg.into())
    }

    /// Create a [`ApiError::Tool`] variant prefixed with the tool name.
    ///
    /// Formats the message as `"{tool}: {msg}"` so logs indicate
    /// which tool failed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool_with_name("Read", "file not found");
    /// assert!(err.to_string().contains("Read"));
    /// ```
    pub fn tool_with_name(tool: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::Tool(format!("{}: {}", tool.into(), msg.into()))
    }

    /// Create a [`ApiError::Tool`] variant for a missing tool.
    ///
    /// Formats the message as `"Tool not found: {tool}"` so that
    /// [`ApiError::code`] returns [`ErrorCode::ToolNotFound`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool_not_found("Bash");
    /// assert_eq!(err.code(), ErrorCode::ToolNotFound);
    /// ```
    pub fn tool_not_found(tool: impl Into<String>) -> Self {
        Self::Tool(format!("Tool not found: {}", tool.into()))
    }

    /// Create a [`ApiError::Tool`] variant for a permission denial.
    ///
    /// Formats the message as `"{tool} permission denied: {msg}"` so
    /// that [`ApiError::code`] returns [`ErrorCode::ToolPermissionDenied`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool_permission("Write", "read-only filesystem");
    /// assert_eq!(err.code(), ErrorCode::ToolPermissionDenied);
    /// ```
    pub fn tool_permission(tool: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::Tool(format!("{} permission denied: {}", tool.into(), msg.into()))
    }

    /// Create a [`ApiError::Tool`] variant for invalid tool input.
    ///
    /// Formats the message as `"{tool} invalid input: {msg}"` so that
    /// [`ApiError::code`] returns [`ErrorCode::ToolInputInvalid`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::tool_input_invalid("Read", "path contains null bytes");
    /// assert_eq!(err.code(), ErrorCode::ToolInputInvalid);
    /// ```
    pub fn tool_input_invalid(tool: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::Tool(format!("{} invalid input: {}", tool.into(), msg.into()))
    }

    /// Create a generic [`ApiError::Config`] variant.
    ///
    /// The message will be inspected by [`ApiError::code`] to select
    /// the most specific [`ErrorCode`] (file not found, parse error,
    /// validation error, missing, or generic parse fallback).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::config("invalid TOML syntax at line 42");
    /// assert_eq!(err.code(), ErrorCode::ConfigParseError);
    /// ```
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a [`ApiError::Config`] variant for a missing config file.
    ///
    /// Formats the message as `"Configuration file not found: {path}"`
    /// so that [`ApiError::code`] returns [`ErrorCode::ConfigFileNotFound`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::config_not_found("/etc/loopctl.toml");
    /// assert_eq!(err.code(), ErrorCode::ConfigFileNotFound);
    /// ```
    pub fn config_not_found(path: impl Into<String>) -> Self {
        Self::Config(format!("Configuration file not found: {}", path.into()))
    }

    /// Create a [`ApiError::Config`] variant for a validation failure.
    ///
    /// Formats the message as `"Configuration validation failed: {msg}"`
    /// so that [`ApiError::code`] returns [`ErrorCode::ConfigValidationError`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::config_validation("timeout must be positive");
    /// assert_eq!(err.code(), ErrorCode::ConfigValidationError);
    /// ```
    pub fn config_validation(msg: impl Into<String>) -> Self {
        Self::Config(format!("Configuration validation failed: {}", msg.into()))
    }

    /// Create an [`ApiError::Api`] variant for a request timeout.
    ///
    /// Formats the message as `"Request timeout: {msg}"` so that
    /// [`ApiError::code`] returns [`ErrorCode::ApiTimeout`]. The error
    /// is considered retryable by [`ApiError::is_retryable`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api_timeout("no response after 30s");
    /// assert!(err.is_retryable());
    /// assert_eq!(err.code(), ErrorCode::ApiTimeout);
    /// ```
    pub fn api_timeout(msg: impl Into<String>) -> Self {
        Self::Api(format!("Request timeout: {}", msg.into()))
    }

    /// Create an [`ApiError::Api`] variant for a rate-limit response.
    ///
    /// Returns a fixed message containing `"rate limit"` so that
    /// [`ApiError::code`] returns [`ErrorCode::ApiRateLimited`] and
    /// [`ApiError::is_retryable`] returns `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api_rate_limited();
    /// assert!(err.is_retryable());
    /// assert_eq!(err.code(), ErrorCode::ApiRateLimited);
    /// ```
    #[must_use]
    pub fn api_rate_limited() -> Self {
        Self::Api("Rate limit exceeded, please retry after a moment".into())
    }

    /// Create an [`ApiError::Api`] variant for an SSE stream error.
    ///
    /// Formats the message as `"Stream error: {msg}"` so that
    /// [`ApiError::code`] returns [`ErrorCode::ApiStreamError`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::api_stream("connection reset mid-stream");
    /// assert_eq!(err.code(), ErrorCode::ApiStreamError);
    /// ```
    pub fn api_stream(msg: impl Into<String>) -> Self {
        Self::Api(format!("Stream error: {}", msg.into()))
    }

    /// Create a catch-all [`ApiError::Other`] variant.
    ///
    /// Use only when the error does not fit any other category. Maps to
    /// [`ErrorCode::InternalError`] via [`ApiError::code`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let err = ApiError::other("something unexpected happened");
    /// assert_eq!(err.code(), ErrorCode::InternalError);
    /// ```
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Create an I/O error for a missing file.
    ///
    /// Constructor for file-lookup failures. Maps to
    /// [`ErrorCode::IoFileNotFound`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    ///
    /// let err = ApiError::io_not_found(
    ///     std::io::Error::new(std::io::ErrorKind::NotFound, "config.toml"),
    /// );
    /// assert!(matches!(err, ApiError::Io(_)));
    /// assert_eq!(err.code(), ErrorCode::IoFileNotFound);
    /// ```
    #[must_use]
    pub fn io_not_found(err: std::io::Error) -> Self {
        // Soft-validate: if called with a non-NotFound error, still
        // construct the Io variant rather than panicking.
        Self::Io(err)
    }

    /// Create an I/O error for a read failure.
    ///
    /// Maps to [`ErrorCode::IoReadError`] unless the underlying
    /// [`std::io::ErrorKind`] indicates otherwise (e.g. `NotFound`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    ///
    /// let err = ApiError::io_read(
    ///     std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated"),
    /// );
    /// assert!(matches!(err, ApiError::Io(_)));
    /// assert_eq!(err.code(), ErrorCode::IoReadError);
    /// ```
    #[must_use]
    pub fn io_read(err: std::io::Error) -> Self {
        Self::Io(err)
    }

    /// Create an I/O error for a write failure.
    ///
    /// Maps to [`ErrorCode::IoWriteError`] unless the underlying
    /// [`std::io::ErrorKind`] indicates otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    ///
    /// let err = ApiError::io_write(
    ///     std::io::Error::new(std::io::ErrorKind::WriteZero, "disk full"),
    /// );
    /// assert!(matches!(err, ApiError::Io(_)));
    /// assert_eq!(err.code(), ErrorCode::IoWriteError);
    /// ```
    #[must_use]
    pub fn io_write(err: std::io::Error) -> Self {
        Self::Io(err)
    }

    /// Convert any [`std::error::Error`] into an [`ApiError::Http`] variant.
    ///
    /// Uses the source error's [`Display`](std::fmt::Display) output as
    /// the message string. Useful for collapsing hyper / reqwest errors
    /// into the unified [`ApiError`] type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::{ApiError, ErrorCode};
    /// let io_err = std::io::Error::new(std::io::ErrorKind::Other, "oops");
    /// let api_err = ApiError::from_hyper(io_err);
    /// assert!(matches!(api_err, ApiError::Http(_)));
    /// ```
    pub fn from_hyper<E: std::error::Error>(e: E) -> Self {
        Self::Http(e.to_string())
    }
}

/// Result type for operations that can fail with [`ApiError`].
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::api::error::{ApiError, ErrorCode};
/// fn load_config() -> Result<Config> {
///     let text = std::fs::read_to_string("loopctl.toml")
///         .map_err(|e| ApiError::config_not_found("loopctl.toml"))?;
///     Ok(parse(&text))
/// }
/// ```
pub type Result<T> = std::result::Result<T, ApiError>;

#[cfg(test)]
/// Unit tests for the [`ApiError`] enum and [`ErrorCode`] codes.
///
/// These tests verify that:
///
/// - Each ergonomic constructor produces the correct [`ErrorCode`]
///   via [`ApiError::code`].
/// - Keyword-based classification in [`ApiError::code`] selects the
///   most specific code for each message pattern.
/// - Boolean predicates ([`ApiError::is_retryable`],
///   [`ApiError::is_auth_error`], [`ApiError::is_config_error`],
///   [`ApiError::is_context_overflow`], [`ApiError::is_io_error`],
///   [`ApiError::is_tool_error`]) match the expected variants.
/// - `From` conversions for [`serde_json::Error`] and [`std::io::Error`]
///   produce the correct variant and code.
/// - [`ErrorCode`] serialisation round-trips through JSON.
mod tests {
    use super::*;

    #[test]
    fn test_api_error_code() {
        let error = ApiError::api("API failed");
        assert!(error.to_string().contains("API error"));
        assert!(error.to_string().contains("API failed"));
        assert_eq!(error.code(), ErrorCode::ApiRequestFailed);
    }

    #[test]
    fn test_auth_error() {
        let error = ApiError::auth("Invalid key");
        assert!(error.to_string().contains("Authentication error"));
        assert_eq!(error.code(), ErrorCode::AuthInvalidKey);
        assert!(error.is_auth_error());
    }

    #[test]
    fn test_auth_invalid_key() {
        let error = ApiError::auth_invalid_key("expired");
        assert_eq!(error.code(), ErrorCode::AuthInvalidKey);
    }

    #[test]
    fn test_auth_failed_generic() {
        let error = ApiError::auth("token expired");
        assert_eq!(error.code(), ErrorCode::AuthFailed);
    }

    #[test]
    fn test_http_error() {
        let error = ApiError::http("Connection failed");
        assert!(error.to_string().contains("HTTP error"));
        assert_eq!(error.code(), ErrorCode::HttpConnectionError);
    }

    #[test]
    fn test_http_error_with_status() {
        // 5xx → HttpResponseError
        let error = ApiError::http_with_status(500, "Server error");
        assert!(error.to_string().contains("HTTP 500"));
        assert!(error.to_string().contains("Server error"));
        assert_eq!(error.code(), ErrorCode::HttpResponseError);

        // 4xx → HttpRequestError
        let error = ApiError::http_with_status(429, "Too many requests");
        assert_eq!(error.code(), ErrorCode::HttpRequestError);

        // Transport-level http() → HttpConnectionError
        let error = ApiError::http("Connection refused");
        assert_eq!(error.code(), ErrorCode::HttpConnectionError);

        // Edge: status 399 → still < 500, so HttpRequestError
        let error = ApiError::http_with_status(399, "redirect-ish");
        assert_eq!(error.code(), ErrorCode::HttpRequestError);

        // Edge: manual Http message without "HTTP {n}:" prefix → HttpConnectionError
        let error = ApiError::Http("generic transport failure".into());
        assert_eq!(error.code(), ErrorCode::HttpConnectionError);

        // Edge: malformed status → HttpConnectionError (fallback)
        let error = ApiError::Http("HTTP abc: not a number".into());
        assert_eq!(error.code(), ErrorCode::HttpConnectionError);
    }

    #[test]
    fn test_tool_error() {
        let error = ApiError::tool("Execution failed");
        assert!(error.to_string().contains("Tool error"));
        assert_eq!(error.code(), ErrorCode::ToolExecutionFailed);
        assert!(error.is_tool_error());
    }

    #[test]
    fn test_tool_not_found() {
        let error = ApiError::tool_not_found("Bash");
        assert!(error.to_string().contains("Tool not found"));
        assert_eq!(error.code(), ErrorCode::ToolNotFound);
    }

    #[test]
    fn test_tool_permission() {
        let error = ApiError::tool_permission("Write", "no access");
        assert_eq!(error.code(), ErrorCode::ToolPermissionDenied);
    }

    #[test]
    fn test_tool_input_invalid() {
        let error = ApiError::tool_input_invalid("Read", "bad path");
        assert_eq!(error.code(), ErrorCode::ToolInputInvalid);
    }

    #[test]
    fn test_config_error() {
        let error = ApiError::config("Invalid config");
        assert!(error.to_string().contains("Configuration error"));
        assert_eq!(error.code(), ErrorCode::ConfigParseError);
        assert!(error.is_config_error());
    }

    #[test]
    fn test_config_not_found() {
        let error = ApiError::config_not_found("/etc/config.toml");
        assert_eq!(error.code(), ErrorCode::ConfigFileNotFound);
    }

    #[test]
    fn test_config_validation() {
        let error = ApiError::config_validation("missing field");
        assert_eq!(error.code(), ErrorCode::ConfigValidationError);
    }

    #[test]
    fn test_context_overflow_detection() {
        let error = ApiError::api("Context length exceeded");
        assert!(error.is_context_overflow());
        assert_eq!(error.code(), ErrorCode::ApiContextOverflow);

        let error = ApiError::api("max tokens reached");
        assert!(error.is_context_overflow());

        let error = ApiError::api("too many tokens in request");
        assert!(error.is_context_overflow());

        let error = ApiError::api("normal error");
        assert!(!error.is_context_overflow());
    }

    #[test]
    fn test_retryable_errors() {
        assert!(ApiError::api_timeout("timed out").is_retryable());
        assert!(ApiError::api_rate_limited().is_retryable());
        assert!(ApiError::http("connection failed").is_retryable());
        // HttpResponseError (5xx) and HttpRequestError (4xx) are retryable
        assert!(ApiError::http_with_status(503, "unavailable").is_retryable());
        assert!(ApiError::http_with_status(429, "too many requests").is_retryable());
        assert!(!ApiError::auth("invalid key").is_retryable());
        assert!(!ApiError::tool("not found").is_retryable());
    }

    #[test]
    fn test_error_codes_numeric() {
        assert_eq!(ErrorCode::ApiRequestFailed as u32, 1000);
        assert_eq!(ErrorCode::ApiRateLimited as u32, 1002);
        assert_eq!(ErrorCode::ApiTimeout as u32, 1003);
        assert_eq!(ErrorCode::AuthFailed as u32, 1101);
        assert_eq!(ErrorCode::ToolExecutionFailed as u32, 1301);
        assert_eq!(ErrorCode::Interrupted as u32, 1999);
    }

    #[test]
    fn test_from_json_error() {
        let json_result: std::result::Result<serde_json::Value, _> =
            serde_json::from_str("{invalid}");
        let error: ApiError = json_result.unwrap_err().into();
        assert!(matches!(error, ApiError::Json(_)));
        assert_eq!(error.code(), ErrorCode::JsonParseError);
    }

    #[test]
    fn test_from_io_error() {
        // #[from] conversion routes NotFound → IoFileNotFound
        let error: ApiError =
            std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into();
        assert!(matches!(error, ApiError::Io(_)));
        assert_eq!(error.code(), ErrorCode::IoFileNotFound);
        assert!(error.is_io_error());

        // PermissionDenied → IoWriteError
        let error: ApiError =
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no access").into();
        assert_eq!(error.code(), ErrorCode::IoWriteError);

        // Generic I/O error → IoReadError
        let error: ApiError =
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated").into();
        assert_eq!(error.code(), ErrorCode::IoReadError);

        // WriteZero → IoWriteError
        let error: ApiError =
            std::io::Error::new(std::io::ErrorKind::WriteZero, "disk full").into();
        assert_eq!(error.code(), ErrorCode::IoWriteError);
    }

    #[test]
    fn test_result_type() {
        fn returns_result() -> String {
            "success".to_string()
        }
        let result = returns_result();
        assert_eq!(result, "success");
    }

    #[test]
    fn test_api_stream_error() {
        let error = ApiError::api_stream("connection reset");
        assert_eq!(error.code(), ErrorCode::ApiStreamError);
    }

    #[test]
    fn test_tool_with_name() {
        let error = ApiError::tool_with_name("Read", "file not found");
        assert!(error.to_string().contains("Read"));
        assert!(error.to_string().contains("file not found"));
    }

    #[test]
    fn test_from_hyper() {
        let error = ApiError::from_hyper(std::io::Error::other("oops"));
        assert!(matches!(error, ApiError::Http(_)));
    }

    #[test]
    fn test_other_error() {
        let error = ApiError::other("something went wrong");
        assert!(error.to_string().contains("something went wrong"));
        assert_eq!(error.code(), ErrorCode::InternalError);
    }

    #[test]
    fn test_error_code_serialization() {
        let code = ErrorCode::ApiRequestFailed;
        let json = serde_json::to_string(&code).unwrap();
        assert_eq!(
            json, "1000",
            "ErrorCode should serialize as numeric discriminant"
        );
        let back: ErrorCode = serde_json::from_str(&json).unwrap();
        assert_eq!(code, back);

        // Verify round-trip for a code in the middle of the range
        let code = ErrorCode::ToolPermissionDenied;
        let json = serde_json::to_string(&code).unwrap();
        assert_eq!(json, "1302");
        let back: ErrorCode = serde_json::from_str(&json).unwrap();
        assert_eq!(code, back);

        // Verify round-trip for the largest code
        let code = ErrorCode::Interrupted;
        let json = serde_json::to_string(&code).unwrap();
        assert_eq!(json, "1999");
        let back: ErrorCode = serde_json::from_str(&json).unwrap();
        assert_eq!(code, back);
    }
}
