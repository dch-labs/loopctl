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
//! use loopctl::engine::loop_core::Loop;
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
//! let result = agent.run("Hello!").await?;
//! ```

use crate::{
    api::error::ApiError,
    message::{MessagePart, Role},
};
use std::time::Duration;

/// Extension trait for [`reqwest::ClientBuilder`] that accepts `Option<T>`
/// for pool and TCP knobs, no-opping when `None`.
///
/// Each provider builder stores pool/TCP settings as `Option<T>` so that
/// `None` means "defer to reqwest's default." Without this trait, applying
/// those settings in `build()` requires three separate `if let Some(...)`
/// blocks. With it, the builder chain stays fluent:
///
/// ```rust,ignore
/// reqwest::Client::builder()
///     .timeout(timeout)
///     .connect_timeout(connect_timeout)
///     .tcp_nodelay(true)
///     .maybe_pool_max_idle_per_host(pool_max_idle)
///     .maybe_pool_idle_timeout(pool_idle_timeout)
///     .maybe_tcp_keepalive(tcp_keepalive)
///     .build()
/// ```
///
/// When the value is `None`, the method returns the builder unchanged —
/// reqwest's built-in default is used. When `Some`, the value is forwarded
/// to the corresponding `reqwest::ClientBuilder` method.
pub(super) trait ClientBuilderExt: Sized {
    /// Set the maximum number of idle connections kept alive per host.
    ///
    /// When `Some(n)`, forwards to
    /// [`pool_max_idle_per_host`](reqwest::ClientBuilder::pool_max_idle_per_host).
    /// When `None`, the builder is returned unchanged and reqwest's default
    /// (unlimited) is used.
    ///
    /// Most callers leave this at `None`. Set to a small value (e.g. 1–4)
    /// for memory-constrained runners or workloads that make mostly serial
    /// requests to a single host.
    fn maybe_pool_max_idle_per_host(self, val: Option<usize>) -> Self;

    /// Set how long an idle connection stays in the pool before being closed.
    ///
    /// When `Some(d)`, forwards to
    /// [`pool_idle_timeout`](reqwest::ClientBuilder::pool_idle_timeout).
    /// When `None`, the builder is returned unchanged and reqwest's default
    /// (90 seconds) is used.
    ///
    /// Raise for long-idle interactive workloads to keep TLS sessions warm
    /// between turns. Lower for tight batch jobs to free file descriptors
    /// sooner.
    fn maybe_pool_idle_timeout(self, val: Option<Duration>) -> Self;

    /// Enable OS-level TCP keepalive at the given interval.
    ///
    /// When `Some(d)`, forwards to
    /// [`tcp_keepalive`](reqwest::ClientBuilder::tcp_keepalive). When `None`,
    /// the builder is returned unchanged and TCP keepalive stays disabled
    /// (reqwest default).
    ///
    /// Enable (~60 seconds) if connections are silently dropped after idle
    /// periods — e.g. behind aggressive NATs, firewalls, or load balancers
    /// that reap idle TCP sessions without sending FIN.
    fn maybe_tcp_keepalive(self, val: Option<Duration>) -> Self;
}

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

// =======================================================
// Default endpoints / models for convenience constructors
// =======================================================

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

// ==================================================
// Internal helpers
// ==================================================

/// Read an environment variable, falling back to a second name, then a
/// default value.
///
/// Reduces boilerplate in the convenience constructors below where a
/// provider supports multiple env-var aliases (e.g. `XAI_API_KEY` /
/// `GROK_API_KEY`).
fn env_or_fallback(primary: &str, fallback: &str) -> Option<String> {
    std::env::var(primary)
        .or_else(|_| std::env::var(fallback))
        .ok()
}

/// Read an environment variable or return a default.
fn env_or_default(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// Look up a required API key, returning [`ApiError`] if neither env
/// var is set.
///
/// # Errors
///
/// Returns [`ApiError::auth_invalid_key`] if neither environment
/// variable is set.
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
/// Providers that reject an inline system role (Anthropic, Gemini) accept
/// system content only as a top-level request field. This helper pulls every
/// `Role::System` message out of `messages`, concatenates their text parts
/// (newline-separated), and merges the result with an optional caller-supplied
/// system prompt — caller prompt first, folded text appended.
///
/// Returns the non-system messages (in original order) and the merged system
/// string, or `None` when neither a caller prompt nor any system message is
/// present.
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
        .api_key(api_key)
        .base_url(base)
        .model(model)
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
        .api_key(api_key)
        .base_url(DEEPSEEK_BASE_URL)
        .model(model)
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
    let model = env_or_default("GROK_MODEL", GROK_DEFAULT_MODEL);

    OpenAiClient::builder()
        .api_key(api_key)
        .base_url(GROK_BASE_URL)
        .model(model)
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
        .api_key(api_key)
        .base_url(ZAI_BASE_URL)
        .model(model)
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
        .api_key(api_key)
        .base_url(base_url)
        .model(model)
        .build()
}

// ==================================================
// Tests
// ==================================================

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! env_set {
        ($($arg:tt)*) => {{
            // SAFETY: This is only used in single-threaded test code where
            // no other task is reading or writing environment variables.
            unsafe { std::env::set_var($($arg)*) }
        }};
    }

    macro_rules! env_remove {
        ($($arg:tt)*) => {{
            // SAFETY: This is only used in single-threaded test code where
            // no other task is reading or writing environment variables.
            unsafe { std::env::remove_var($($arg)*) }
        }};
    }

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

    #[test]
    fn env_or_fallback_none_when_both_missing() {
        env_remove!("LOOPCTL_TEST_NEITHER_A");
        env_remove!("LOOPCTL_TEST_NEITHER_B");
        assert_eq!(
            env_or_fallback("LOOPCTL_TEST_NEITHER_A", "LOOPCTL_TEST_NEITHER_B"),
            None
        );
    }

    #[test]
    fn env_or_default_uses_env_when_set() {
        env_set!("LOOPCTL_TEST_DEFAULT", "from-env");
        assert_eq!(
            env_or_default("LOOPCTL_TEST_DEFAULT", "fallback"),
            "from-env"
        );
        env_remove!("LOOPCTL_TEST_DEFAULT");
    }

    #[test]
    fn env_or_default_uses_default_when_unset() {
        env_remove!("LOOPCTL_TEST_DEFAULT2");
        assert_eq!(
            env_or_default("LOOPCTL_TEST_DEFAULT2", "fallback"),
            "fallback"
        );
    }

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

    #[test]
    fn require_api_key_no_fallback_set() {
        env_set!("LOOPCTL_TEST_KEY_ONLY", "only-val");
        let key = require_api_key("LOOPCTL_TEST_KEY_ONLY", None).unwrap();
        assert_eq!(key, "only-val");
        env_remove!("LOOPCTL_TEST_KEY_ONLY");
    }

    #[test]
    fn require_api_key_errors_when_missing() {
        env_remove!("LOOPCTL_TEST_MISSING_KEY");
        let err = require_api_key("LOOPCTL_TEST_MISSING_KEY", None).unwrap_err();
        assert!(err.to_string().contains("LOOPCTL_TEST_MISSING_KEY"));
    }

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
    fn sys_msg(texts: &[&str]) -> crate::message::Message {
        use crate::message::{MessagePart, Role};
        let parts: Vec<MessagePart> = texts.iter().map(|t| MessagePart::text(*t)).collect();
        crate::message::Message::new(Role::System, parts)
    }

    #[test]
    fn fold_system_no_system_messages_no_caller_returns_none() {
        let msgs = [crate::message::Message::user("hi")];
        let (non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(non_system.len(), 1);
        assert!(system.is_none(), "no system content → None");
    }

    #[test]
    fn fold_system_caller_only_passes_through() {
        let msgs = [crate::message::Message::user("hi")];
        let (non_system, system) = fold_system_messages(&msgs, Some("be brief"));
        assert_eq!(non_system.len(), 1);
        assert_eq!(system.as_deref(), Some("be brief"));
    }

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

    #[test]
    fn fold_system_multiple_text_parts_in_one_message_joined() {
        let msgs = [sys_msg(&["part one", "part two"])];
        let (_non_system, system) = fold_system_messages(&msgs, None);
        assert_eq!(system.as_deref(), Some("part one\npart two"));
    }
}
