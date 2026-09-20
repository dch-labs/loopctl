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
    for scenario in cassette::scenarios()
        // The rate-limit scenario is a gated attempt, not a driven recording.
        .into_iter()
        .filter(|s| s.provider == "anthropic" && s.name != "rate_limit_error")
    {
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
            .with_base_url(cassette::client_base_url(
                scenario.provider,
                &session.base_url(),
            ))
            .with_model(scenario.model)
            .build()
            .unwrap();
        drive_scenario_and_save(&client, &scenario, session).await;
    }
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
    // The rate-limit scenario is a gated attempt, not a driven recording.
    let scenarios: Vec<_> = cassette::scenarios()
        .into_iter()
        .filter(|s| s.provider == "gemini" && s.name != "rate_limit_error")
        .collect();
    for scenario in scenarios {
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
            .with_api_key(api_key.clone())
            .with_base_url(cassette::client_base_url(
                scenario.provider,
                &session.base_url(),
            ))
            .with_model(scenario.model)
            .build()
            .unwrap();
        drive_scenario_and_save(&client, &scenario, session).await;
    }
}

/// Cloud-OpenAI recordings: the openai-compat corpus's local Ollama
/// truth, backed by the real OpenAI wire — including the fragmented
/// tool-argument streaming Ollama emits as one delta.
#[cfg(feature = "openai")]
#[tokio::test]
async fn record_openai_cassettes() {
    if !cassette_recording_enabled() || std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  cassette recording (openai)");
        return;
        // The rate-limit scenario is a gated attempt, not a driven recording.
    }
    for scenario in cassette::scenarios().into_iter().filter(|s| {
        s.provider == "openai" && s.name != "rate_limit_error" && s.name != "model_override"
    }) {
        let server = httpmock::MockServer::start_async().await;
        let session = cassette::CassetteSession::start(
            scenario.provider,
            scenario.name,
            "https://api.openai.com",
            &server,
        )
        .await;
        let client = loopctl::provider::OpenAiClient::builder()
            .with_api_key(std::env::var("OPENAI_API_KEY").unwrap())
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

/// DeepSeek cloud recording: the shared OpenAI-compat converter
/// against DeepSeek's own response shapes.
///
/// One scenario per run of the driver table's DeepSeek entries; skips
/// without the deliberate-act environment or the credential.
#[cfg(feature = "deepseek")]
#[tokio::test]
async fn record_deepseek_cassettes() {
    if !cassette_recording_enabled() || std::env::var("DEEPSEEK_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  cassette recording (deepseek)");
        return;
    }
    for scenario in cassette::scenarios()
        .into_iter()
        .filter(|s| s.provider == "deepseek")
    {
        let server = httpmock::MockServer::start_async().await;
        let session = cassette::CassetteSession::start(
            scenario.provider,
            scenario.name,
            "https://api.deepseek.com",
            &server,
        )
        .await;
        let client = loopctl::provider::deepseek_builder()
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

/// Grok cloud recording: the shared OpenAI-compat converter against
/// xAI's own response shapes.
///
/// One scenario per run of the driver table's Grok entries; skips
/// without the deliberate-act environment or the credential.
#[cfg(feature = "grok")]
#[tokio::test]
async fn record_grok_cassettes() {
    let has_key = std::env::var("XAI_API_KEY").is_ok() || std::env::var("GROK_API_KEY").is_ok();
    if !cassette_recording_enabled() || !has_key {
        eprintln!("{DIM}skip{RESET}  cassette recording (grok)");
        return;
    }
    for scenario in cassette::scenarios()
        .into_iter()
        .filter(|s| s.provider == "grok")
    {
        let server = httpmock::MockServer::start_async().await;
        let session = cassette::CassetteSession::start(
            scenario.provider,
            scenario.name,
            "https://api.x.ai",
            &server,
        )
        .await;
        let client = loopctl::provider::grok_builder()
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

/// Z.ai cloud recording: the Anthropic-compatible dialect on its own
/// endpoint.
///
/// Runs the driver table's Z.ai entries through the shared
/// Anthropic wire shape; skips without the deliberate-act
/// environment or the credential.
#[cfg(feature = "zai")]
#[tokio::test]
async fn record_zai_cassette() {
    let has_key = std::env::var("ZAI_API_KEY").is_ok() || std::env::var("ZHIPUAI_API_KEY").is_ok();
    if !cassette_recording_enabled() || !has_key {
        eprintln!("{DIM}skip{RESET}  cassette recording (zai)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "zai")
        .unwrap();
    let server = httpmock::MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        scenario.provider,
        scenario.name,
        "https://api.z.ai",
        &server,
    )
    .await;
    let client = loopctl::provider::zai_builder()
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model(scenario.model)
        .build()
        .unwrap();
    drive_scenario_and_save(&client, &scenario, session).await;
}

/// One rate-limit recording attempt: the minimal request against the
/// real provider, with the session's gate deciding what is true.
///
/// A 429 cannot be ordered from the provider, so the attempt runs once
/// — no bursts — and [`finish_rate_limited`](cassette::CassetteSession::finish_rate_limited)
/// writes the cassette only when every recorded response is a 429.
async fn attempt_rate_limit_recording<C: ApiClient>(
    scenario: &cassette::Scenario,
    real_base_url: &str,
    build_client: impl FnOnce(&str) -> C,
) {
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, real_base_url, &server)
            .await;
    let client = build_client(&session.base_url());
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    // The provider's choice, not the driver's: a 429 surfaces as an
    // error event, a success as a served stream. Either way the
    // exchange is on the recording; the gate at the finish decides
    // whether a cassette names itself rate-limited.
    while let Some(result) = stream.next().await {
        if let Err(e) = result {
            eprintln!("rate-limit attempt ({}): {e}", scenario.name);
        }
    }
    match session.finish_rate_limited().await {
        Some(path) => println!("{GREEN}CASSETTE{RESET} {} → {path:?}", scenario.name),
        None => eprintln!(
            "{DIM}no 429{RESET}  {} — cassette not written",
            scenario.name
        ),
    }
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn attempt_anthropic_rate_limit_cassette() {
    if !cassette_recording_enabled() || std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  rate-limit attempt (anthropic)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "anthropic" && s.name == "rate_limit_error")
        .unwrap();
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    attempt_rate_limit_recording(&scenario, "https://api.anthropic.com", |base| {
        loopctl::provider::AnthropicClient::builder()
            .with_api_key(api_key.clone())
            .with_base_url(cassette::client_base_url(scenario.provider, base))
            .with_model(scenario.model)
            .build()
            .unwrap()
    })
    .await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn attempt_gemini_rate_limit_cassette() {
    if !cassette_recording_enabled()
        || (std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err())
    {
        eprintln!("{DIM}skip{RESET}  rate-limit attempt (gemini)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "gemini" && s.name == "rate_limit_error")
        .unwrap();
    let api_key = std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_API_KEY"))
        .unwrap();
    attempt_rate_limit_recording(
        &scenario,
        "https://generativelanguage.googleapis.com",
        |base| {
            loopctl::provider::GeminiClient::builder()
                .with_api_key(api_key.clone())
                .with_base_url(cassette::client_base_url(scenario.provider, base))
                .with_model(scenario.model)
                .build()
                .unwrap()
        },
    )
    .await;
}

/// Record the model-override cassette: the same minimal request, but
/// the loop carries a per-request model override, so the recorded body
/// pins the override on the wire — the one request-shape dimension no
/// other cassette exercises.
#[cfg(feature = "openai")]
#[tokio::test]
async fn record_openai_model_override_cassette() {
    if !cassette_recording_enabled() || std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  cassette recording (openai model_override)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai" && s.name == "model_override")
        .unwrap();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    let server = httpmock::MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        scenario.provider,
        scenario.name,
        "https://api.openai.com",
        &server,
    )
    .await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key(api_key.clone())
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model("gpt-4o-mini")
        .with_stream_usage(scenario.stream_usage)
        .build()
        .unwrap();
    // The client's own model is gpt-4o-mini; the per-request override
    // names the scenario model, so the recorded body must carry the
    // override, not the client default.
    let options = loopctl::structured::RequestOptions::new().with_model(scenario.model);
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let stream = client.stream_messages_with_options(&request, options);
    let mut stream = std::pin::pin!(stream);
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => events.push(event),
            Err(e) => panic!("{} recording failed mid-stream: {e}", scenario.name),
        }
    }

    // The same junk-cassette gate every driver applies: a clean-but-
    // truncated stream must fail here, not freeze as truth.
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
async fn attempt_openai_rate_limit_cassette() {
    if !cassette_recording_enabled() || std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("{DIM}skip{RESET}  rate-limit attempt (openai)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai" && s.name == "rate_limit_error")
        .unwrap();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    attempt_rate_limit_recording(&scenario, "https://api.openai.com", |base| {
        loopctl::provider::OpenAiClient::builder()
            .with_api_key(api_key.clone())
            .with_base_url(cassette::client_base_url(scenario.provider, base))
            .with_model(scenario.model)
            .with_stream_usage(scenario.stream_usage)
            .build()
            .unwrap()
    })
    .await;
}
