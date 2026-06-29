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

use crate::api::error::ApiError;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "openai")]
pub use openai::OpenAiClient;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicClient;

#[cfg(feature = "gemini")]
pub use gemini::GeminiClient;

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

// =============================================
// Constructors for OpenAI-compatible providers
// =============================================

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

    /// Helper to safely set an env var in tests (Rust 2024 requires unsafe).
    macro_rules! env_set {
        ($($arg:tt)*) => {{ unsafe { std::env::set_var($($arg)*) } }};
    }

    /// Helper to safely remove an env var in tests.
    macro_rules! env_remove {
        ($($arg:tt)*) => {{ unsafe { std::env::remove_var($($arg)*) } }};
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
}
