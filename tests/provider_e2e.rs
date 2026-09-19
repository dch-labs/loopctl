//! Live test: provider end-to-end smoke + streamed-usage checks.
//!
//! Works with any provider. Set the API key for the ones you want to test.
//! Every cloud provider gets one streamed turn that asserts non-empty text,
//! a terminal `MessageStop`, and non-zero usage on the terminal
//! `MessageDelta`; Ollama keeps a text-only smoke check because its
//! streamed usage support varies by model.
//!
//! Run:
//!   `set -a; source .env; set +a; LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai,azure,moonshot,bedrock --test provider_e2e -- --nocapture --test-threads=1`
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
    feature = "azure",
    feature = "moonshot",
    feature = "bedrock",
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

mod cassette;

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

async fn run_provider_test(
    client: &dyn ApiClient,
    name: &str,
) -> Vec<loopctl::stream::StreamEvent> {
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
    let usage = events.iter().find_map(|e| match e {
        loopctl::stream::StreamEvent::MessageDelta(md) => md.usage,
        _ => None,
    });

    println!("{CYAN}\"{text}\"{RESET}");
    match usage {
        Some(u) => println!(
            "       {DIM}usage: {} in / {} out{RESET}",
            u.input_tokens, u.output_tokens
        ),
        None => println!("       {DIM}usage: not reported on the stream{RESET}"),
    }

    assert!(!text.is_empty(), "{name} should produce non-empty text");
    assert!(has_stop, "{name} stream should end with MessageStop");
    events
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

/// Run one streamed turn and assert the terminal `MessageDelta` carries
/// non-zero usage. For providers whose servers stream usage (every cloud
/// provider below); Ollama stays on the plain smoke check, since its
/// streamed usage support varies by model.
async fn run_streamed_usage_test(client: &dyn ApiClient, name: &str) {
    let events = run_provider_test(client, name).await;
    let usage = events
        .iter()
        .find_map(|e| match e {
            loopctl::stream::StreamEvent::MessageDelta(md) => md.usage,
            _ => None,
        })
        .expect("the terminal MessageDelta must carry usage");
    assert!(
        usage.input_tokens > 0,
        "{name}: streamed input_tokens must be non-zero"
    );
    assert!(
        usage.output_tokens > 0,
        "{name}: streamed output_tokens must be non-zero on a completed turn"
    );
}

/// Live check of streamed usage on OpenAI: the client requests
/// `stream_options.include_usage` by default, so the final chunk's usage
/// must reach the terminal `MessageDelta` with non-zero counts.
#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("OPENAI_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  OpenAI streamed usage");
        return;
    }
    let client = loopctl::provider::OpenAiClient::from_env().unwrap();
    run_streamed_usage_test(&client, "OpenAI streamed usage").await;
}

/// Live check of the streamed usage latch: Anthropic reports input tokens on
/// `message_start`, and the terminal `MessageDelta` must carry them — a
/// regression returns to `input_tokens: 0` on every turn.
#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("ANTHROPIC_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Anthropic streamed usage");
        return;
    }
    let client = loopctl::provider::AnthropicClient::from_env().unwrap();
    run_streamed_usage_test(&client, "Anthropic streamed usage").await;
}

/// Live check of streamed usage on Gemini: the final chunk carries
/// `usageMetadata`, and the terminal `MessageDelta` must report it with
/// non-zero counts.
#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Gemini streamed usage");
        return;
    }
    let client = loopctl::provider::GeminiClient::from_env().unwrap();
    run_streamed_usage_test(&client, "Gemini streamed usage").await;
}

/// Live check of streamed usage on Grok: xAI's OpenAI-compatible server
/// honors `stream_options.include_usage`, so the final chunk's usage must
/// reach the terminal `MessageDelta`.
#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("XAI_API_KEY").is_err() && std::env::var("GROK_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Grok streamed usage");
        return;
    }
    let client = loopctl::provider::grok().unwrap();
    run_streamed_usage_test(&client, "Grok streamed usage").await;
}

/// Live check of streamed usage on DeepSeek: the OpenAI-compatible server
/// honors `stream_options.include_usage`, so the final chunk's usage must
/// reach the terminal `MessageDelta`.
#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("DEEPSEEK_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  DeepSeek streamed usage");
        return;
    }
    let client = loopctl::provider::deepseek().unwrap();
    run_streamed_usage_test(&client, "DeepSeek streamed usage").await;
}

/// Live check of streamed usage on Z.ai: the Anthropic-compatible server
/// reports input tokens on `message_start`, exercising the same usage
/// latch as the Anthropic test against a different endpoint.
#[cfg(feature = "zai")]
#[tokio::test]
async fn zai_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || (std::env::var("ZAI_API_KEY").is_err() && std::env::var("ZHIPUAI_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  Z.ai streamed usage");
        return;
    }
    let client = loopctl::provider::zai().unwrap();
    run_streamed_usage_test(&client, "Z.ai streamed usage").await;
}

/// Live check of streamed usage on Azure OpenAI via the v1 API profile:
/// the resource name comes from `LOOPCTL_AZURE_RESOURCE` (it is an
/// argument, not an env var, in `provider::azure`).
#[cfg(feature = "azure")]
#[tokio::test]
async fn azure_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("AZURE_OPENAI_API_KEY").is_err()
        || std::env::var("AZURE_OPENAI_MODEL").is_err()
        || std::env::var("LOOPCTL_AZURE_RESOURCE").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Azure streamed usage");
        return;
    }
    let resource = std::env::var("LOOPCTL_AZURE_RESOURCE").unwrap();
    let client = loopctl::provider::azure(resource).unwrap();
    run_streamed_usage_test(&client, "Azure streamed usage").await;
}

/// Live check of streamed usage on Moonshot AI (Kimi): the
/// OpenAI-compatible server reports streamed usage like the other
/// profiles.
#[cfg(feature = "moonshot")]
#[tokio::test]
async fn moonshot_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("MOONSHOT_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Moonshot streamed usage");
        return;
    }
    let client = loopctl::provider::moonshot().unwrap();
    run_streamed_usage_test(&client, "Moonshot streamed usage").await;
}

/// Live check of streamed usage on AWS Bedrock: exercises whichever
/// invoke path the configured `AWS_BEDROCK_MODEL` selects (native
/// Anthropic or Converse), the SigV4 chain, and the event-stream
/// decoder against the real runtime.
#[cfg(feature = "bedrock")]
#[tokio::test]
async fn bedrock_streamed_usage_test() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("AWS_REGION").is_err()
        || std::env::var("AWS_ACCESS_KEY_ID").is_err()
        || std::env::var("AWS_SECRET_ACCESS_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  Bedrock streamed usage");
        return;
    }
    let client = loopctl::provider::BedrockClient::from_env().unwrap();
    run_streamed_usage_test(&client, "Bedrock streamed usage").await;
}

/// Cassette recording driver.
///
/// With `LOOPCTL_CASSETTE=record` and `LOOPCTL_E2E=1`, each scenario in
/// the scenario table runs once against the real provider *through the
/// forwarding proxy*, and the scrubbed exchange lands in
/// `tests/cassettes/<provider>/<scenario>.yaml`. Without those
/// variables every driver here skips — recording is a deliberate human
/// act, never a CI step.
fn cassette_recording_enabled() -> bool {
    std::env::var("LOOPCTL_E2E").as_deref() == Ok("1")
        && std::env::var("LOOPCTL_CASSETTE").as_deref() == Ok("record")
}

async fn drive_scenario_and_save(
    client: &dyn ApiClient,
    scenario: &cassette::Scenario,
    session: cassette::CassetteSession<'_>,
) {
    let mut request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    if scenario.tools {
        request = request.with_tools(Some(vec![cassette::get_weather_tool()]));
    }
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => events.push(event),
            Err(e) => panic!("{} recording failed mid-stream: {e}", scenario.name),
        }
    }

    let has_stop = events
        .iter()
        .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop));
    assert!(
        has_stop,
        "{}: the recorded exchange must end with MessageStop, or the cassette is junk",
        scenario.name
    );
    if scenario.stream_usage {
        let usage = events.iter().find_map(|e| match e {
            loopctl::stream::StreamEvent::MessageDelta(md) => md.usage,
            _ => None,
        });
        assert!(
            usage.is_some_and(|u| u.input_tokens > 0),
            "{}: the scenario exists to pin usage-on-final-chunk — the stream must carry it",
            scenario.name
        );
    }

    let path = session.finish().await;
    println!("{GREEN}CASSETTE{RESET} {} → {path:?}", scenario.name);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn record_openai_compat_cassettes() {
    if !cassette_recording_enabled() {
        eprintln!("{DIM}skip{RESET}  cassette recording (openai-compat)");
        return;
    }
    // The proxy forwards to an origin — the request path (/v1/…) is
    // preserved from what the client sends, so no /v1 suffix here.
    let real_base_url = std::env::var("LOOPCTL_CASSETTE_UPSTREAM")
        .unwrap_or_else(|_| "http://localhost:11434".into());
    for scenario in cassette::scenarios()
        .into_iter()
        .filter(|s| s.provider == "openai-compat")
    {
        let server = httpmock::MockServer::start_async().await;
        let session = cassette::CassetteSession::start(
            scenario.provider,
            scenario.name,
            &real_base_url,
            &server,
        )
        .await;
        let client = loopctl::provider::OpenAiClient::builder()
            .with_api_key("ollama")
            .with_base_url(cassette::client_base_url(
                scenario.provider,
                &session.base_url(),
            ))
            .with_model(scenario.model)
            .with_stream_usage(scenario.stream_usage)
            .build()
            .unwrap();
        drive_scenario_and_save(&client, &scenario, session).await;
    }
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn record_anthropic_cassette() {
    if !cassette_recording_enabled() || std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  cassette recording (anthropic)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "anthropic")
        .unwrap();
    let server = httpmock::MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        scenario.provider,
        scenario.name,
        "https://api.anthropic.com",
        &server,
    )
    .await;
    let client = loopctl::provider::AnthropicClient::builder()
        .with_api_key(std::env::var("ANTHROPIC_API_KEY").unwrap())
        .with_base_url(session.base_url())
        .with_model(scenario.model)
        .build()
        .unwrap();
    drive_scenario_and_save(&client, &scenario, session).await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn record_gemini_cassette() {
    if !cassette_recording_enabled()
        || (std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  cassette recording (gemini)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "gemini")
        .unwrap();
    let api_key = std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        .unwrap();
    let server = httpmock::MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        scenario.provider,
        scenario.name,
        "https://generativelanguage.googleapis.com",
        &server,
    )
    .await;
    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key(api_key)
        .with_base_url(session.base_url())
        .with_model(scenario.model)
        .build()
        .unwrap();
    drive_scenario_and_save(&client, &scenario, session).await;
}
