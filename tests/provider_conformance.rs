//! Provider conformance suite: the contract from
//! `tests/common/conformance.rs` run per provider over the cassette
//! corpus. Each provider's presence here is configuration — a profile
//! naming its client builder and dialect — never a re-derivation of
//! which behaviors matter.
//!
//! Hermetic like the replay suites: dummy keys, mock servers serving
//! recorded bytes, no network. Beyond replay's byte guard, the battery
//! asserts the *interpretation*: lifecycle, usage, routing, taxonomy,
//! and options against fixture data as the oracle.
//!
//! `conformance_matrix_artifact` additionally renders the per-provider
//! matrix (contract family × provider) to
//! `target/provider-conformance/matrix.md` — the "which providers are
//! proven" table, generated on every run, gaps visible.

#![cfg(any(feature = "openai", feature = "anthropic", feature = "gemini",))]
#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

mod cassette;

#[path = "common/conformance.rs"]
mod conformance;

use cassette::Scenario;
use conformance::{CellStatus, ClientBundle, ProviderProfile, Violation};
use futures::StreamExt;
use loopctl::api::ApiClient;

/// Corpus default for the local OpenAI-compatible dialect.
///
/// The client seeds it the way every vendor client seeds its own
/// default; the `model_override` fixture must prove the per-request
/// override left it behind on the wire, or the scenario proves
/// nothing and is rejected as evidence.
#[cfg(feature = "openai")]
const DEFAULT_OPENAI_COMPAT: &str = "qwen2.5:7b";

#[cfg(feature = "openai")]
const DEFAULT_OPENAI: &str = "gpt-4o-mini";

#[cfg(feature = "openai")]
fn build_openai_compat(mock_base_url: &str, scenario: &Scenario) -> ClientBundle {
    let model = if scenario.name == "model_override" {
        DEFAULT_OPENAI_COMPAT
    } else {
        scenario.model
    };
    ClientBundle {
        client: Box::new(
            loopctl::provider::OpenAiClient::builder()
                .with_api_key(cassette::CASSETTE_KEY)
                .with_base_url(mock_base_url)
                .with_model(model)
                .with_stream_usage(scenario.stream_usage)
                .build()
                .unwrap(),
        ),
    }
}

#[cfg(feature = "openai")]
fn build_openai(mock_base_url: &str, scenario: &Scenario) -> ClientBundle {
    let model = if scenario.name == "model_override" {
        DEFAULT_OPENAI
    } else {
        scenario.model
    };
    ClientBundle {
        client: Box::new(
            loopctl::provider::OpenAiClient::builder()
                .with_api_key(cassette::CASSETTE_KEY)
                .with_base_url(mock_base_url)
                .with_model(model)
                .with_stream_usage(scenario.stream_usage)
                .build()
                .unwrap(),
        ),
    }
}

#[cfg(feature = "openai")]
fn build_openai_family(mock_base_url: &str, scenario: &Scenario) -> ClientBundle {
    ClientBundle {
        client: Box::new(
            loopctl::provider::OpenAiClient::builder()
                .with_api_key(cassette::CASSETTE_KEY)
                .with_base_url(mock_base_url)
                .with_model(scenario.model)
                .with_stream_usage(scenario.stream_usage)
                .build()
                .unwrap(),
        ),
    }
}

#[cfg(feature = "anthropic")]
fn build_anthropic_family(mock_base_url: &str, scenario: &Scenario) -> ClientBundle {
    ClientBundle {
        client: Box::new(
            loopctl::provider::AnthropicClient::builder()
                .with_api_key(cassette::CASSETTE_KEY)
                .with_base_url(mock_base_url)
                .with_model(scenario.model)
                .build()
                .unwrap(),
        ),
    }
}

#[cfg(feature = "gemini")]
fn build_gemini(mock_base_url: &str, scenario: &Scenario) -> ClientBundle {
    ClientBundle {
        client: Box::new(
            loopctl::provider::GeminiClient::builder()
                .with_api_key(cassette::CASSETTE_KEY)
                .with_base_url(mock_base_url)
                .with_model(scenario.model)
                .build()
                .unwrap(),
        ),
    }
}

/// Every compiled profile, in matrix column order.
///
/// Assembled under the same feature gates as the clients themselves, so
/// a build carries exactly the vendors it can actually drive; the
/// artifact's column order and the completeness check both read this
/// list, which is what makes a missing profile a finding rather than an
/// omission.
pub fn profiles() -> Vec<&'static ProviderProfile> {
    let mut profiles: Vec<&'static ProviderProfile> = Vec::new();
    #[cfg(feature = "openai")]
    {
        static OPENAI_COMPAT: ProviderProfile = ProviderProfile {
            provider: "openai-compat",
            default_model: DEFAULT_OPENAI_COMPAT,
            build: build_openai_compat,
            expects_tool_constraints: true,
        };
        static OPENAI: ProviderProfile = ProviderProfile {
            provider: "openai",
            default_model: DEFAULT_OPENAI,
            build: build_openai,
            expects_tool_constraints: true,
        };
        static DEEPSEEK: ProviderProfile = ProviderProfile {
            provider: "deepseek",
            default_model: "deepseek-v4-flash",
            build: build_openai_family,
            expects_tool_constraints: true,
        };
        static GROK: ProviderProfile = ProviderProfile {
            provider: "grok",
            default_model: "grok-4.5",
            build: build_openai_family,
            expects_tool_constraints: true,
        };
        profiles.extend([&OPENAI_COMPAT, &OPENAI, &DEEPSEEK, &GROK]);
    }
    #[cfg(feature = "anthropic")]
    {
        static ANTHROPIC: ProviderProfile = ProviderProfile {
            provider: "anthropic",
            default_model: "claude-sonnet-4-5",
            build: build_anthropic_family,
            expects_tool_constraints: true,
        };
        static ZAI: ProviderProfile = ProviderProfile {
            provider: "zai",
            default_model: "glm-4.7",
            build: build_anthropic_family,
            expects_tool_constraints: true,
        };
        profiles.extend([&ANTHROPIC, &ZAI]);
    }
    #[cfg(feature = "gemini")]
    {
        static GEMINI: ProviderProfile = ProviderProfile {
            provider: "gemini",
            default_model: "gemini-3.5-flash",
            build: build_gemini,
            expects_tool_constraints: true,
        };
        profiles.push(&GEMINI);
    }
    profiles
}

/// Providers the matrix lists even without a compiled profile.
///
/// Each reason names the gate that unblocks the column — the recording
/// still to take, and what must route through it once taken — so a
/// pending row reads as a declared gap, never a silently absent
/// provider.
fn pending_providers(compiled: &[&'static ProviderProfile]) -> Vec<(String, String)> {
    let mut pending = Vec::new();
    for name in [
        "openai-compat",
        "openai",
        "deepseek",
        "grok",
        "anthropic",
        "zai",
        "gemini",
    ] {
        if !compiled.iter().any(|profile| profile.provider == name) {
            pending.push((
                name.to_string(),
                "feature not compiled in this run — rebuild with it; the conformance \
                 rule applies the moment the profile compiles"
                    .to_string(),
            ));
        }
    }
    pending.push((
        "bedrock".to_string(),
        "Bedrock cassettes pending — its column fills once the hermetic \
         SigV4 recording suite lands and its HTTP failures classify through \
         the shared status table; cassettes alone cannot turn it"
            .to_string(),
    ));
    pending
}

/// Run one profile's battery and fail loudly, naming every violation.
///
/// This is the battery CI runs per vendor: every scenario's findings are
/// collected before the assert, so one run reports every broken rule
/// with its scenario and family instead of stopping at the first.
async fn conformance_run(profile: &'static ProviderProfile) {
    let reports = conformance::run_provider(profile).await;
    let mut violations: Vec<&Violation> = Vec::new();
    for report in &reports {
        match &report.status {
            CellStatus::Fail(violation_list) => {
                violations.extend(violation_list.iter());
            }
            CellStatus::NotRun(reason) => {
                eprintln!("  n/r  {}/{} — {reason}", report.provider, report.scenario);
            }
            CellStatus::Pass => {
                println!("  pass {}/{}", report.provider, report.scenario);
            }
        }
    }
    assert!(
        violations.is_empty(),
        "conformance failed for {}:\n{}",
        profile.provider,
        violations
            .iter()
            .map(|violation| violation.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn conformance_openai_compat() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "openai-compat")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn conformance_openai() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "openai")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn conformance_deepseek() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "deepseek")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn conformance_grok() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "grok")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn conformance_anthropic() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "anthropic")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn conformance_zai() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "zai")
            .unwrap(),
    )
    .await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn conformance_gemini() {
    conformance_run(
        profiles()
            .into_iter()
            .find(|p| p.provider == "gemini")
            .unwrap(),
    )
    .await;
}

#[tokio::test]
async fn conformance_matrix_artifact() {
    let compiled = profiles();
    let mut reports = Vec::new();
    for profile in &compiled {
        reports.extend(conformance::run_provider(profile).await);
    }
    let matrix = conformance::Matrix {
        providers: compiled
            .iter()
            .map(|profile| (*profile.provider).to_string())
            .collect(),
        reports,
        pending: pending_providers(&compiled),
    };

    let markdown = matrix.render_markdown();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("provider-conformance");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("matrix.md"), &markdown).unwrap();
    println!("{markdown}");

    let violations = matrix.violations();
    assert!(
        violations.is_empty(),
        "conformance matrix reports {} violation(s):\n{}",
        violations.len(),
        violations
            .iter()
            .map(|violation| violation.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );

    // The published table is read by people who did not write the checks,
    // so what it renders has to carry its own meaning: a legend for what a
    // `pass` claims, a reason beside every `n/r`, and the families a
    // recording *did* decide staying green beside the one it could not.
    let cell = |row_prefix: &str, family: usize| -> String {
        let row = markdown
            .lines()
            .find(|line| line.starts_with(row_prefix))
            .unwrap_or_else(|| panic!("the table has no row starting {row_prefix:?}"));
        row.split('|')
            .nth(family + 2)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or_else(|| panic!("{row_prefix:?} has no cell for family {family}"))
            .to_string()
    };
    assert!(
        markdown.contains("*family exercised by at least one fixture*"),
        "the legend states what an exercised column means:\n{markdown}"
    );
    #[cfg(feature = "openai")]
    {
        let taxonomy = cell("| openai |", 3);
        assert!(
            taxonomy.starts_with("n/r — ") && taxonomy.contains("no failure to classify"),
            "an unadjudicated family renders with its reason, never a bare `n/r`: {taxonomy}"
        );
        assert_eq!(
            cell("| openai |", 0),
            "pass",
            "the families openai's fixtures did decide stay green beside the one they could not"
        );
        assert_eq!(
            cell("| openai-compat |", 3),
            "pass",
            "openai-compat is the one vendor whose recording decides the taxonomy"
        );
    }
    let bedrock = cell("| bedrock |", 0);
    assert!(
        bedrock.starts_with("n/r — Bedrock cassettes pending") && bedrock.contains("classif"),
        "a provider that has not run yet reads as a gap carrying its reason, \
         and the reason names both gates the column waits on — the recording \
         and the status classification — not cassettes alone: {bedrock}"
    );
}

/// Drive an in-memory (possibly mutated) fixture through the exact
/// battery CI runs.
///
/// The meta-pin harness: it returns findings instead of asserting, so
/// one test can demand both directions — the planted violation
/// surfacing in its right family, and the unmutated fixture staying
/// silent.
#[cfg(feature = "openai")]
async fn drive_battery(
    scenario: &Scenario,
    interactions: Vec<cassette::Interaction>,
) -> conformance::BatteryReport {
    let profile = profiles()
        .into_iter()
        .find(|p| p.provider == scenario.provider)
        .unwrap();
    let server = httpmock::MockServer::start_async().await;
    let mocks = cassette::register_replay_mocks(&server, scenario.provider, &interactions).await;
    let bundle = (profile.build)(
        &cassette::client_base_url(profile.provider, &server.base_url()),
        scenario,
    );
    let client: &dyn ApiClient = bundle.client.as_ref();

    let mut request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    if scenario.tools {
        request = request.with_tools(Some(vec![cassette::get_weather_tool()]));
    }
    let stream = client.stream_messages(&request);
    let mut stream = std::pin::pin!(stream);
    let mut output = conformance::RunOutput {
        events: Vec::new(),
        errors: Vec::new(),
    };
    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => output.events.push(event),
            Err(error) => output.errors.push(error),
        }
    }
    for mock in &mocks {
        mock.assert();
    }
    conformance::battery_findings(client, &output, scenario, profile, &interactions)
}

#[cfg(feature = "openai")]
fn openai_compat_scenario(name: &str) -> Scenario {
    cassette::scenarios()
        .into_iter()
        .find(|s| s.provider == "openai-compat" && s.name == name)
        .unwrap()
}

/// Cut a recorded stream below its provider terminal.
///
/// The terminating chunk and `[DONE]` go, the text and the usage chunk
/// stay — exactly what a re-record cut short looks like on disk, since
/// this dialect reports usage *after* its finish reason.
#[cfg(feature = "openai")]
fn cut_below_terminal(interactions: &mut Vec<cassette::Interaction>) {
    for interaction in interactions {
        if let Some(body) = interaction.then.body.as_mut() {
            let kept: Vec<&str> = body
                .split("\n\n")
                .filter(|block| {
                    !block.contains("[DONE]") && !block.contains("\"finish_reason\":\"stop\"")
                })
                .collect();
            *body = kept.join("\n\n");
        }
    }
}

/// A run that surfaced nothing: no events, no errors.
///
/// What a client that correctly stays silent on a truncated stream
/// produces — the expectation for the pins asserting a cut fixture
/// earns no verdict and no fabricated terminal.
#[cfg(feature = "openai")]
fn silent_run() -> conformance::RunOutput {
    conformance::RunOutput {
        events: Vec::new(),
        errors: Vec::new(),
    }
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn conformance_catches_a_planted_violation() {
    // Green control first: the same harness over the unmutated fixture
    // must be silent, or "catching" the plant means nothing.
    let scenario = openai_compat_scenario("minimal_text");
    let clean = cassette::load_interactions("openai-compat", "minimal_text");
    let control = drive_battery(&scenario, clean.clone()).await;
    assert!(
        control.violations.is_empty() && control.fixture_gaps.is_empty(),
        "the unmutated fixture must pass the battery, got:\n{:?}\n{:?}",
        control.violations,
        control.fixture_gaps
    );
    assert!(
        control
            .not_applicable
            .iter()
            .any(|(family, _)| *family == "error_taxonomy"),
        "a success fixture cannot decide the error taxonomy's status arms and must \
         report that instead of collecting a pass for them: {:?}",
        control.not_applicable
    );

    // Plant A — truncation: cut the stream below its provider terminal,
    // keeping the usage chunk that rides after the finish reason. The
    // pinned client contract answers such a stream with silence (the
    // 0.3.0 no-synthetic-stop rule), so every terminal-shaped rule the
    // cut makes undecidable must come back as a named fixture-integrity
    // gap — never a violation against the conformant client, and never a
    // silent pass either.
    let mut truncated = clean.clone();
    cut_below_terminal(&mut truncated);
    let report = drive_battery(&scenario, truncated).await;
    assert!(
        report.violations.is_empty(),
        "a client that stays silent on a truncated fixture conforms — the \
         suite must not blame it, got: {:?}",
        report.violations
    );
    assert!(
        report.fixture_gaps.iter().any(|(family, detail)| {
            *family == "stream_lifecycle" && detail.contains("ends before the provider's terminal")
        }),
        "the suite must name the truncation as a fixture-integrity gap, got: {:?}",
        report.fixture_gaps
    );
    assert!(
        report
            .fixture_gaps
            .iter()
            .any(|(family, _)| *family == "stop_reason_routing"),
        "the routing family's unjudgeable rules must gap alongside, got: {:?}",
        report.fixture_gaps
    );
    assert!(
        report.fixture_gaps.iter().any(|(family, detail)| {
            *family == "usage_reporting" && detail.contains("cut below the provider's terminal")
        }),
        "usage recorded above the cut is undecidable too: a client that reports \
         nothing on a truncated stream conforms, so the suite must gap rather \
         than blame, got: {:?}",
        report.fixture_gaps
    );

    // Plant B — lost tool identity: rename the tool on the response
    // side only. The call must still route, but the lane no longer
    // carries the requested identity; the suite must say exactly that.
    let tool_scenario = openai_compat_scenario("tool_call_lifecycle");
    let mut renamed = cassette::load_interactions("openai-compat", "tool_call_lifecycle");
    for interaction in &mut renamed {
        if let Some(body) = interaction.then.body.as_mut() {
            *body = body.replace("get_weather", "ghost_weather");
        }
    }
    let control = drive_battery(
        &tool_scenario,
        cassette::load_interactions("openai-compat", "tool_call_lifecycle"),
    )
    .await;
    assert!(
        control.violations.is_empty() && control.fixture_gaps.is_empty(),
        "the unmutated tool fixture must pass the battery, got:\n{:?}\n{:?}",
        control.violations,
        control.fixture_gaps
    );
    assert!(
        control
            .not_applicable
            .iter()
            .any(|(family, _)| *family == "error_taxonomy"),
        "the coverage hole is reported on every success fixture, not just the \
         text one: {:?}",
        control.not_applicable
    );
    let report = drive_battery(&tool_scenario, renamed).await;
    assert!(
        report.violations.iter().any(|(family, detail)| {
            *family == "stop_reason_routing" && detail.contains("get_weather")
        }),
        "the suite must catch the lost tool identity and name it, got: {:?}",
        report.violations
    );
    assert!(
        report.fixture_gaps.is_empty(),
        "the renamed tool fixture still reaches its terminals — no gap \
         expected, got: {:?}",
        report.fixture_gaps
    );
}

#[cfg(feature = "openai")]
#[test]
fn truncation_gaps_unless_the_client_fabricates_a_terminal() {
    let mut truncated = cassette::load_interactions("openai-compat", "minimal_text");
    cut_below_terminal(&mut truncated);
    let silent = conformance::check_stream_lifecycle(&silent_run(), &truncated);
    assert!(
        silent.violations.is_empty(),
        "silence is exactly what a conformant client owes a truncated \
         stream, got: {:?}",
        silent.violations
    );
    assert!(
        silent.fixture_gaps.len() == 1
            && silent.fixture_gaps[0].0 == "stream_lifecycle"
            && silent.fixture_gaps[0]
                .1
                .contains("ends before the provider's terminal"),
        "the gap must be named: {:?}",
        silent.fixture_gaps
    );

    let fabricated = conformance::check_stream_lifecycle(
        &conformance::RunOutput {
            events: vec![loopctl::stream::StreamEvent::MessageStop],
            errors: Vec::new(),
        },
        &truncated,
    );
    assert!(
        fabricated.fixture_gaps.is_empty(),
        "a fabricated terminal is decidable — no gap: {:?}",
        fabricated.fixture_gaps
    );
    assert!(
        fabricated.violations.iter().any(|(family, detail)| {
            *family == "stream_lifecycle" && detail.contains("no synthetic stop")
        }),
        "the 0.3.0 rule must fire on the synthetic stop: {:?}",
        fabricated.violations
    );
}

#[cfg(feature = "openai")]
#[test]
fn a_truncated_usage_fixture_gaps_instead_of_blaming_the_client() {
    let mut truncated = cassette::load_interactions("openai-compat", "minimal_text");
    cut_below_terminal(&mut truncated);
    let gaps = conformance::check_usage_reporting(&silent_run(), &truncated);
    assert!(
        gaps.violations.is_empty(),
        "silence is what a conformant client owes a stream that never reached \
         its terminal, got: {:?}",
        gaps.violations
    );
    assert!(
        gaps.fixture_gaps.iter().any(|(family, detail)| {
            *family == "usage_reporting" && detail.contains("cut below the provider's terminal")
        }),
        "the undecidable rule must be named as a fixture gap: {:?}",
        gaps.fixture_gaps
    );

    // The intact fixture, same silent client: the terminal was recorded,
    // so the rule is decidable and stays a violation.
    let intact = cassette::load_interactions("openai-compat", "minimal_text");
    let blame = conformance::check_usage_reporting(&silent_run(), &intact);
    assert!(
        blame.fixture_gaps.is_empty()
            && blame.violations.iter().any(|(family, detail)| {
                *family == "usage_reporting" && detail.contains("no MessageDelta carried it")
            }),
        "an intact fixture must still blame a client that swallowed its usage: \
         violations {:?} gaps {:?}",
        blame.violations,
        blame.fixture_gaps
    );

    // An error fixture owes no usage block at all — not applicable, and
    // neither a violation nor a re-record demand.
    let errored = cassette::load_interactions("openai-compat", "missing_model_error");
    let report = conformance::check_usage_reporting(&silent_run(), &errored);
    assert!(
        report.violations.is_empty()
            && report.fixture_gaps.is_empty()
            && report
                .not_applicable
                .iter()
                .any(|(family, _)| *family == "usage_reporting"),
        "an error fixture has no usage to carry: violations {:?} gaps {:?} n/a {:?}",
        report.violations,
        report.fixture_gaps,
        report.not_applicable
    );
}

#[cfg(feature = "openai")]
#[test]
fn a_family_no_recording_can_adjudicate_gaps_instead_of_passing() {
    let intact = cassette::load_interactions("openai-compat", "minimal_text");
    let report = conformance::check_error_taxonomy(&silent_run(), &intact);
    assert!(
        report.violations.is_empty(),
        "a success stream that surfaced no error broke nothing: {:?}",
        report.violations
    );
    assert!(
        report.not_applicable.iter().any(|(family, detail)| {
            *family == "error_taxonomy" && detail.contains("no failure to classify")
        }),
        "the status arms are undecidable on a success fixture and must be \
         reported as such: {:?}",
        report.not_applicable
    );

    // The same check on the corpus's error fixture does decide, and says
    // so with a verdict rather than a gap.
    let errored = cassette::load_interactions("openai-compat", "missing_model_error");
    let decided = conformance::check_error_taxonomy(&silent_run(), &errored);
    assert!(
        decided
            .violations
            .iter()
            .any(|(family, _)| *family == "error_taxonomy")
            && decided.not_applicable.is_empty(),
        "a recorded failure is adjudicable: an error fixture with no typed \
         error is a violation, not a gap: violations {:?} n/a {:?}",
        decided.violations,
        decided.not_applicable
    );
}

#[cfg(feature = "openai")]
#[test]
fn the_matrix_counts_a_family_only_where_a_check_chose() {
    let taxonomy = conformance::FAMILIES
        .iter()
        .position(|family| *family == "error_taxonomy")
        .expect("the family table names error_taxonomy");
    let cells = |taxonomy_cell: conformance::CellStatus| {
        conformance::FAMILIES
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index == taxonomy {
                    taxonomy_cell.clone()
                } else {
                    conformance::CellStatus::Pass
                }
            })
            .collect::<Vec<_>>()
    };
    let unproven = conformance::RunReport {
        provider: "success-only".to_string(),
        scenario: "minimal_text".to_string(),
        families: cells(conformance::CellStatus::NotRun(
            "a success fixture carries no failure to classify".to_string(),
        )),
        status: conformance::CellStatus::Pass,
    };
    let proven = conformance::RunReport {
        provider: "has-an-error-fixture".to_string(),
        scenario: "missing_model_error".to_string(),
        families: cells(conformance::CellStatus::Pass),
        status: conformance::CellStatus::Pass,
    };
    let matrix = conformance::Matrix {
        providers: vec![
            "success-only".to_string(),
            "has-an-error-fixture".to_string(),
        ],
        reports: vec![unproven, proven],
        pending: Vec::new(),
    };

    assert_eq!(
        matrix.family_verdict("success-only", taxonomy),
        conformance::FamilyVerdict::Unproven(
            "a success fixture carries no failure to classify".to_string()
        ),
        "nothing adjudicated the taxonomy here, so no cell may read pass"
    );
    assert_eq!(
        matrix.family_verdict("has-an-error-fixture", taxonomy),
        conformance::FamilyVerdict::Exercised,
        "one recorded failure is enough to prove the family"
    );

    let markdown = matrix.render_markdown();
    let row = |provider: &str| -> Vec<String> {
        markdown
            .lines()
            .find(|line| line.starts_with(&format!("| {provider} ")))
            .unwrap_or_else(|| panic!("the table has no row for {provider}:\n{markdown}"))
            .split('|')
            .skip(1)
            .map(|cell| cell.trim().to_string())
            .collect()
    };
    // Column 0 is the provider name; family columns follow in FAMILIES
    // order. An unadjudicated cell renders `n/r` *and* the reason, so the
    // table alone tells a reader what to record.
    assert!(
        row("success-only")[taxonomy + 1].starts_with("n/r"),
        "the vendor with no error fixture must not read green for it:\n{markdown}"
    );
    assert!(
        row("success-only")[taxonomy + 1].contains("no failure to classify"),
        "the cell names why the family went unproven, not merely that it did:\n{markdown}"
    );
    assert_eq!(
        row("success-only")[1],
        "pass",
        "the families the success fixture did decide stay green:\n{markdown}"
    );
    assert_eq!(
        row("has-an-error-fixture")[taxonomy + 1],
        "pass",
        "the vendor that recorded a failure keeps its pass:\n{markdown}"
    );
    assert!(
        markdown.lines().any(
            |line| line.starts_with("- coverage: success-only/error_taxonomy")
                && line.contains("no failure to classify")
        ),
        "the artifact must name the coverage hole and what closes it:\n{markdown}"
    );
    assert!(
        !markdown
            .lines()
            .any(|line| line.starts_with("- coverage: has-an-error-fixture/error_taxonomy")),
        "a family a fixture adjudicated is not a gap:\n{markdown}"
    );
}

#[cfg(feature = "openai")]
#[test]
fn a_fabricated_retry_after_fails_the_taxonomy_equality() {
    let seconds = |value: u64| Some(std::time::Duration::from_secs(value));
    let rate_limit = |delay: Option<std::time::Duration>| conformance::RunOutput {
        events: Vec::new(),
        errors: vec![loopctl::api::error::ApiError::rate_limited(
            "HTTP 429: slow down",
            delay,
        )],
    };
    // The corpus's error fixture read as a throttle: 429, and the
    // header-less arm is the one nobody can order from a provider.
    let mut headerless = cassette::load_interactions("openai-compat", "missing_model_error");
    headerless
        .first_mut()
        .expect("the fixture has an interaction")
        .then
        .status = Some(429);

    let invented = conformance::check_error_taxonomy(&rate_limit(seconds(30)), &headerless);
    assert!(
        invented.violations.iter().any(|(family, detail)| {
            *family == "error_taxonomy" && detail.contains("recorded Retry-After None")
        }),
        "a delay the response never advised is a fabrication, not a pass: {:?}",
        invented.violations
    );
    let agreeing = conformance::check_error_taxonomy(&rate_limit(None), &headerless);
    assert!(
        agreeing.violations.is_empty(),
        "carrying nothing when nothing was advised conforms: {:?}",
        agreeing.violations
    );

    let mut advised = headerless.clone();
    advised
        .first_mut()
        .expect("the fixture has an interaction")
        .then
        .header
        .get_or_insert_with(Vec::new)
        .push(cassette::Pair {
            name: "retry-after".to_string(),
            value: "12".to_string(),
        });
    let dropped = conformance::check_error_taxonomy(&rate_limit(None), &advised);
    assert!(
        dropped.violations.iter().any(|(family, detail)| {
            *family == "error_taxonomy" && detail.contains("the error carries None")
        }),
        "an advised delay the client dropped is the other half of the same rule: {:?}",
        dropped.violations
    );
    let carried = conformance::check_error_taxonomy(&rate_limit(seconds(12)), &advised);
    assert!(
        carried.violations.is_empty(),
        "the advised delay, carried, conforms: {:?}",
        carried.violations
    );
}

#[cfg(feature = "openai")]
#[test]
fn the_retry_after_oracle_reads_both_wire_forms() {
    let fixture_retrying = |value: &str| {
        let mut interactions = cassette::load_interactions("openai-compat", "minimal_text");
        let first = interactions
            .first_mut()
            .expect("the fixture has an interaction");
        first
            .then
            .header
            .get_or_insert_with(Vec::new)
            .push(cassette::Pair {
                name: "retry-after".to_string(),
                value: value.to_string(),
            });
        interactions
    };
    assert_eq!(
        conformance::recorded_retry_after(&fixture_retrying("12")),
        Some(std::time::Duration::from_secs(12)),
        "delta-seconds, as always"
    );
    let dated =
        conformance::recorded_retry_after(&fixture_retrying("Tue, 21 Oct 2098 07:28:00 GMT"));
    assert!(
        dated.is_some_and(|delay| delay.as_secs() > 60 * 60 * 24),
        "an HTTP-date recording must reach the oracle too, got {dated:?}"
    );
    assert_eq!(
        conformance::recorded_retry_after(&fixture_retrying("soon-ish")),
        None,
        "unparseable is no delay — on both sides of the check"
    );
}

#[cfg(feature = "openai")]
#[test]
fn the_retry_after_oracle_reads_the_error_response() {
    let warm_up = cassette::load_interactions("openai-compat", "minimal_text")
        .into_iter()
        .next()
        .expect("the corpus fixture has one interaction");
    let pair = |name: &str, value: &str| cassette::Pair {
        name: name.to_string(),
        value: value.to_string(),
    };

    // Warm-up carries headers but no advice; the 429 records the delay.
    let mut quiet_warm_up = warm_up.clone();
    quiet_warm_up.then.header = Some(vec![pair("x-ratelimit-remaining", "9")]);
    let mut advised_failure = warm_up.clone();
    advised_failure.then.status = Some(429);
    advised_failure.then.header = Some(vec![pair("retry-after", "12")]);
    assert_eq!(
        conformance::recorded_retry_after(&[quiet_warm_up, advised_failure]),
        Some(std::time::Duration::from_secs(12)),
        "the advice recorded on the failed response is the answer — a \
         header-carrying warm-up must not silence it"
    );

    // Mirrored: the warm-up advised a delay, the failure recorded none.
    let mut advising_warm_up = warm_up.clone();
    advising_warm_up.then.status = Some(200);
    advising_warm_up.then.header = Some(vec![pair("retry-after", "5")]);
    let mut bare_failure = warm_up;
    bare_failure.then.status = Some(429);
    bare_failure.then.header = Some(Vec::new());
    assert_eq!(
        conformance::recorded_retry_after(&[advising_warm_up, bare_failure]),
        None,
        "advice recorded on a success is not the failure's advice — reading \
         it as such would blame a conformant client for fabricating"
    );
}
