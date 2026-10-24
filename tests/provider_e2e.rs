//! Live test: provider end-to-end smoke + streamed-usage checks.
//!
//! Works with any provider. Set the API key for the ones you want to test.
//! Every cloud provider gets one streamed turn that asserts non-empty text,
//! a terminal `MessageStop`, and non-zero usage on the terminal
//! `MessageDelta`; Ollama keeps a text-only smoke check because its
//! streamed usage support varies by model.
//!
//! Run:
//!   `set -a; source .env; set +a; LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai,azure,moonshot,bedrock,vector_memory,testing --test provider_e2e -- --nocapture --test-threads=1`
//!   (the trailing `vector_memory,testing` pair wakes the embedder round-trips
//!   and the local-vs-remote precision measurement; without them the live
//!   embedding entries compile out.)
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
        // model_override needs per-request options and
        // missing_model_error is a gated attempt — neither is a plain
        // driven recording, same reason the cloud loops exclude their
        // rate-limit scenarios.
        .filter(|s| !matches!(s.name, "model_override" | "missing_model_error"))
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

#[cfg(feature = "openai")]
#[tokio::test]
async fn record_openai_compat_model_override_cassette() {
    if !cassette_recording_enabled() {
        eprintln!("{DIM}skip{RESET}  cassette recording (openai-compat model_override)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai-compat" && s.name == "model_override")
        .unwrap();
    let real_base_url = std::env::var("LOOPCTL_CASSETTE_UPSTREAM")
        .unwrap_or_else(|_| "http://localhost:11434".into());
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, &real_base_url, &server)
            .await;
    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key("ollama")
        .with_base_url(cassette::client_base_url(
            scenario.provider,
            &session.base_url(),
        ))
        .with_model("qwen2.5:7b")
        .with_stream_usage(scenario.stream_usage)
        .build()
        .unwrap();
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
    let has_stop = events
        .iter()
        .any(|e| matches!(e, loopctl::stream::StreamEvent::MessageStop));
    assert!(
        has_stop,
        "{}: the recorded exchange must end with MessageStop, or the cassette is junk",
        scenario.name
    );
    let path = session.finish().await;
    println!("{GREEN}CASSETTE{RESET} {} → {path:?}", scenario.name);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn attempt_openai_compat_missing_model_cassette() {
    if !cassette_recording_enabled() {
        eprintln!("{DIM}skip{RESET}  missing-model attempt (openai-compat)");
        return;
    }
    let scenario = cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai-compat" && s.name == "missing_model_error")
        .unwrap();
    let real_base_url = std::env::var("LOOPCTL_CASSETTE_UPSTREAM")
        .unwrap_or_else(|_| "http://localhost:11434".into());
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(scenario.provider, scenario.name, &real_base_url, &server)
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
    let request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    while let Some(result) = stream.next().await {
        if let Err(e) = result {
            eprintln!("missing-model attempt ({}): {e}", scenario.name);
        }
    }
    match session.finish_errored().await {
        Some(path) => println!("{GREEN}CASSETTE{RESET} {} → {path:?}", scenario.name),
        None => eprintln!(
            "{DIM}no error{RESET}  {} — cassette not written",
            scenario.name
        ),
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

#[cfg(all(feature = "openai", feature = "vector_index"))]
#[tokio::test]
async fn openai_embedder_round_trips_live() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("OPENAI_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  openai embedder (live)");
        return;
    }
    use loopctl::memory::vector::EmbeddingProvider as _;
    use loopctl::provider::embeddings::OpenAiEmbedder;

    let embedder = OpenAiEmbedder::from_env().unwrap();
    let embedding = embedder
        .embed("the deploy pipeline reruns on config changes")
        .await
        .unwrap();
    assert_eq!(
        embedding.dim(),
        embedder.dim(),
        "the live answer's dimension matches the configured one"
    );
    assert!(
        embedding.as_slice().iter().copied().all(f32::is_finite),
        "every component of a live embedding is finite"
    );
    eprintln!(
        "{GREEN}ok{RESET}    openai embedder: {} dims, finite",
        embedding.dim()
    );
}

#[cfg(all(feature = "ollama", feature = "vector_index"))]
#[tokio::test]
async fn ollama_embedder_round_trips_live() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("OLLAMA_EMBEDDING_MODEL").is_err()
    {
        eprintln!("{DIM}skip{RESET}  ollama embedder (live)");
        return;
    }
    use loopctl::memory::vector::EmbeddingProvider as _;
    use loopctl::provider::embeddings::OllamaEmbedder;

    let embedder = OllamaEmbedder::from_env().unwrap();
    let embedding = embedder
        .embed("the deploy pipeline reruns on config changes")
        .await
        .unwrap();
    assert_eq!(
        embedding.dim(),
        embedder.dim(),
        "the live answer's dimension matches the configured one"
    );
    assert!(
        embedding.as_slice().iter().copied().all(f32::is_finite),
        "every component of a live embedding is finite"
    );
    eprintln!(
        "{GREEN}ok{RESET}    ollama embedder: {} dims, finite",
        embedding.dim()
    );
}

#[cfg(all(
    feature = "ollama",
    feature = "openai",
    feature = "vector_memory",
    feature = "testing"
))]
#[tokio::test]
async fn ollama_keeps_relative_retrieval_precision_live() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1")
        || std::env::var("OPENAI_API_KEY").is_err()
    {
        eprintln!("{DIM}skip{RESET}  embedder precision pair (live)");
        return;
    }
    use loopctl::memory::LoopMemory as _;
    use loopctl::memory::vector::LinearVectorIndex;
    use loopctl::memory::vector_memory::{GoldenSet, VectorMemoryStore, golden_set};
    use loopctl::provider::embeddings::{OllamaEmbedder, OpenAiEmbedder};

    let set = golden_set();
    let openai_store = VectorMemoryStore::new(
        Box::new(OpenAiEmbedder::from_env().unwrap()),
        Box::new(LinearVectorIndex::new(1536)),
    );
    let mut openai_hits: Vec<bool> = Vec::with_capacity(set.queries.len());
    for fixture_entry in &set.entries {
        openai_store.store(fixture_entry.clone()).await.unwrap();
    }
    for (index, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(index);
        let returned = openai_store.retrieve(query.text, 3).await.unwrap();
        openai_hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let openai_ratio = GoldenSet::precision_ratio(&openai_hits);

    let ollama = match std::env::var("OLLAMA_EMBEDDING_MODEL") {
        Ok(_) => OllamaEmbedder::from_env().unwrap(),
        Err(_) => OllamaEmbedder::builder().build().unwrap(),
    };
    let ollama_store =
        VectorMemoryStore::new(Box::new(ollama), Box::new(LinearVectorIndex::new(768)));
    let mut ollama_hits: Vec<bool> = Vec::with_capacity(set.queries.len());
    for fixture_entry in &set.entries {
        ollama_store.store(fixture_entry.clone()).await.unwrap();
    }
    for (index, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(index);
        let returned = ollama_store.retrieve(query.text, 3).await.unwrap();
        ollama_hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let ollama_ratio = GoldenSet::precision_ratio(&ollama_hits);
    eprintln!(
        "{CYAN}info{RESET}  precision pair: openai {:.2}, ollama {:.2} (relative {:.2})",
        openai_ratio,
        ollama_ratio,
        ollama_ratio / openai_ratio.max(f32::EPSILON)
    );
    assert!(
        ollama_ratio + f32::EPSILON >= 0.9 * openai_ratio,
        "local retrieval keeps at least 90% of the frontier baseline's precision: \
         ollama {ollama_ratio:.2} vs openai {openai_ratio:.2}"
    );
}
