//! Live test: provider end-to-end smoke test.
//!
//! Works with any provider. Set the API key for the ones you want to test.
//!
//! Run:
//!   `set -a; source .env; set +a; LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1`
//!
//! The whole file compiles only when at least one provider feature is on;
//! without a provider the helpers have no callers and would trip the
//! `dead_code` lint under `-D warnings`.

#![cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "ollama",
    feature = "deepseek",
    feature = "grok",
    feature = "xai",
    feature = "gemini",
    feature = "zai",
))]
#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use futures::StreamExt;
use loopctl::api::ApiClient;

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn extract_text(events: &[loopctl::stream::StreamEvent]) -> String {
    let mut text = String::new();
    for ev in events {
        if let loopctl::stream::StreamEvent::IndexedDelta(d) = ev
            && let loopctl::stream::DeltaPart::Text { text: delta } = &d.delta
        {
            text.push_str(delta);
        }
    }
    text
}

async fn run_provider_test(client: &dyn ApiClient, name: &str) {
    let model = client.model();
    print!("{GREEN}PASS{RESET} {name} {DIM}({model}){RESET} → ");

    let req = loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(
        "Say hello in exactly 3 words.",
    )]);
    let stream = client.stream_messages(&req);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(ev) => events.push(ev),
            Err(e) => {
                println!("{RED}FAIL{RESET} {name} {DIM}({model}){RESET} → error: {e}");
                panic!("{name} stream error: {e}");
            }
        }
    }

    let text = extract_text(&events);
    let has_stop = events
        .iter()
        .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop));

    println!("{CYAN}\"{text}\"{RESET}");

    assert!(!text.is_empty(), "{name} should produce non-empty text");
    assert!(has_stop, "{name} stream should end with MessageStop");
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn ollama_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") || std::env::var("OLLAMA_MODEL").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Ollama");
        return;
    }
    let model = std::env::var("OLLAMA_MODEL").unwrap();
    let client = loopctl::provider::ollama(&model).unwrap();
    run_provider_test(&client, "Ollama").await;
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("OPENAI_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  OpenAI");
        return;
    }
    let client = loopctl::provider::OpenAiClient::from_env().unwrap();
    run_provider_test(&client, "OpenAI").await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("ANTHROPIC_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Anthropic");
        return;
    }
    let client = loopctl::provider::AnthropicClient::from_env().unwrap();
    run_provider_test(&client, "Anthropic").await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Gemini");
        return;
    }
    let client = loopctl::provider::GeminiClient::from_env().unwrap();
    run_provider_test(&client, "Gemini").await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("XAI_API_KEY").is_err() && std::env::var("GROK_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Grok");
        return;
    }
    let client = loopctl::provider::grok().unwrap();
    run_provider_test(&client, "Grok").await;
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("DEEPSEEK_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  DeepSeek");
        return;
    }
    let client = loopctl::provider::deepseek().unwrap();
    run_provider_test(&client, "DeepSeek").await;
}

#[cfg(feature = "zai")]
#[tokio::test]
async fn zai_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("ZAI_API_KEY").is_err() && std::env::var("ZHIPUAI_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Z.ai");
        return;
    }
    let client = loopctl::provider::zai().unwrap();
    run_provider_test(&client, "Z.ai").await;
}
