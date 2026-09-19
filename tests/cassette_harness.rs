//! Pins for the cassette harness itself.
//!
//! The harness is the drift guard, so its own behavior gets the same
//! treatment as any contract: the drift miss, the completeness assert,
//! the scrub policy, the safety scanner, and the deliberate-act gate
//! are each pinned here, hermetically — no cassettes, no network.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

mod cassette;

use cassette::{
    Interaction, Pair, ThenSpec, WhenSpec, register_replay_mocks, scan_for_secrets, scrub,
};
use httpmock::MockServer;

/// One minimal recorded exchange: a JSON POST answered with 200.
///
/// The same shape a real cassette holds, small enough to read in an
/// assertion failure — the pins exercise the harness, not any
/// provider's actual wire.
fn recorded_interaction(path: &str, body: &str) -> Interaction {
    Interaction {
        when: WhenSpec {
            method: Some("POST".to_string()),
            path: Some(path.to_string()),
            query_param: None,
            header: Some(vec![Pair {
                name: "content-type".to_string(),
                value: "application/json".to_string(),
            }]),
            body: Some(body.to_string()),
            body_base64: None,
        },
        then: ThenSpec {
            status: Some(200),
            header: Some(vec![Pair {
                name: "content-type".to_string(),
                value: "application/json".to_string(),
            }]),
            body: Some("{\"ok\":true}".to_string()),
            body_base64: None,
        },
    }
}

/// A byte-exact POST against the mock server over raw TCP.
///
/// Raw sockets instead of a reqwest dependency: the harness pins must
/// run under every feature set, and the request bytes must be under
/// the test's exact control — one flipped character is the experiment.
fn raw_post(server: &MockServer, path: &str, body: &str) -> (u16, String) {
    raw_post_with_headers(server, path, body, &[("Content-Type", "application/json")])
}

/// A byte-exact POST whose header set the caller controls completely.
///
/// The presence pins must be able to omit and add contract headers —
/// the fixed-header [`raw_post`] cannot express either direction.
fn raw_post_with_headers(
    server: &MockServer,
    path: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> (u16, String) {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(server.address()).unwrap();
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: cassette-pin\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("Connection: close\r\n\r\n");
    request.push_str(body);
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_default();
    let body_start = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(text.len());
    (status, text[body_start..].to_string())
}

#[tokio::test]
async fn replay_fails_on_body_drift() {
    let recorded_body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    let interaction = recorded_interaction("/v1/chat/completions", recorded_body);
    let server = MockServer::start_async().await;
    let mocks = register_replay_mocks(&server, "", &[interaction]).await;

    let (ok_status, ok_body) = raw_post(&server, "/v1/chat/completions", recorded_body);
    assert_eq!(ok_status, 200, "the recorded bytes must be served");
    assert_eq!(ok_body, "{\"ok\":true}");

    let drifted_body = r#"{"model":"qwen","messages":[{"role":"user","content":"ho"}]}"#;
    let (drift_status, _) = raw_post(&server, "/v1/chat/completions", drifted_body);
    assert_eq!(
        drift_status, 404,
        "one changed request byte must miss every recorded mock — this miss is the outbound-drift guard"
    );

    for mock in &mocks {
        assert_eq!(
            mock.calls(),
            1,
            "the drifted attempt matched nothing, so the recording saw exactly its one true hit"
        );
    }
}

#[tokio::test]
#[should_panic(expected = "0 of 1 expected requests matched")]
async fn replay_consumes_every_interaction() {
    let first = recorded_interaction("/v1/chat/completions", r#"{"turn":1}"#);
    let second = recorded_interaction("/v1/chat/completions", r#"{"turn":2}"#);
    let server = MockServer::start_async().await;
    let mocks = register_replay_mocks(&server, "", &[first, second]).await;

    let (status, _) = raw_post(&server, "/v1/chat/completions", r#"{"turn":1}"#);
    assert_eq!(status, 200, "the driven interaction is served");

    for mock in &mocks {
        mock.assert();
    }
}

#[tokio::test]
async fn replay_requires_the_provider_contract_headers() {
    let interaction = recorded_interaction("/v1/messages", "{}");
    let server = MockServer::start_async().await;
    let mocks = register_replay_mocks(&server, "anthropic", &[interaction]).await;

    let (missing, _) = raw_post_with_headers(
        &server,
        "/v1/messages",
        "{}",
        &[("Content-Type", "application/json")],
    );
    assert_eq!(
        missing, 404,
        "a request missing the provider's contract headers must miss the mock — a header dropout is drift too"
    );

    let (present, _) = raw_post_with_headers(
        &server,
        "/v1/messages",
        "{}",
        &[
            ("Content-Type", "application/json"),
            ("x-api-key", "any-value"),
            ("anthropic-version", "any-version"),
        ],
    );
    assert_eq!(
        present, 200,
        "presence — never the value — is what the guard asserts"
    );

    for mock in &mocks {
        mock.assert();
    }
}

#[test]
fn scrubber_placeholders_ids() {
    let mut interactions = vec![
        Interaction {
            when: WhenSpec {
                body: Some(r#"{"model":"m"}"#.to_string()),
                ..Default::default()
            },
            then: ThenSpec {
                status: Some(200),
                body: Some(
                    r#"{"id":"chatcmpl-853","tool":{"id":"call_d4kx8uvi"},"parent":"resp_77"}"#
                        .to_string(),
                ),
                ..Default::default()
            },
        },
        Interaction {
            when: WhenSpec {
                path: Some("/v1/responses/resp_77".to_string()),
                query_param: Some(vec![Pair {
                    name: "after".to_string(),
                    value: "call_d4kx8uvi".to_string(),
                }]),
                body: Some(r#"{"tool_call_id":"call_d4kx8uvi","result":"20C"}"#.to_string()),
                ..Default::default()
            },
            then: ThenSpec {
                status: Some(200),
                body: Some(r#"{"id":"msg_9f2"}"#.to_string()),
                ..Default::default()
            },
        },
    ];

    scrub(&mut interactions);

    let first_body = interactions[0].then.body.as_deref().unwrap();
    let second_path = interactions[1].when.path.as_deref().unwrap();
    let second_query = interactions[1].when.query_param.as_ref().unwrap();
    let second_request = interactions[1].when.body.as_deref().unwrap();
    let second_body = interactions[1].then.body.as_deref().unwrap();
    assert_eq!(
        first_body,
        r#"{"id":"chatcmpl-cassette_1","tool":{"id":"call_cassette_2"},"parent":"resp_cassette_3"}"#,
        "every generated id becomes a deterministic placeholder"
    );
    assert_eq!(
        second_path, "/v1/responses/resp_cassette_3",
        "an id echoed into a later request path keeps the same placeholder"
    );
    assert_eq!(
        second_query.first().map(|pair| pair.value.as_str()),
        Some("call_cassette_2"),
        "an id echoed into a later query value keeps the same placeholder"
    );
    assert_eq!(
        second_request, r#"{"tool_call_id":"call_cassette_2","result":"20C"}"#,
        "an id echoed from a response into a later request keeps the same \
        placeholder, so replay's byte matching survives the scrub"
    );
    assert_eq!(
        second_body, r#"{"id":"msg_cassette_4"}"#,
        "the rename counter stays unique across the whole file"
    );
}

#[test]
fn scrubber_handles_multi_byte_text_around_generated_ids() {
    let mut interactions = vec![Interaction {
        when: WhenSpec {
            body: Some(r#"{"q":"café —巴黎"}"#.to_string()),
            ..Default::default()
        },
        then: ThenSpec {
            status: Some(200),
            body: Some(r#"{"text":"réponse — ok","id":"chatcmpl-853"}"#.to_string()),
            ..Default::default()
        },
    }];

    scrub(&mut interactions);

    assert_eq!(
        interactions[0].then.body.as_deref().unwrap(),
        r#"{"text":"réponse — ok","id":"chatcmpl-cassette_1"}"#,
        "ids are renamed through multi-byte text without slicing a \
        character boundary — model output is routinely non-ASCII"
    );
    assert_eq!(
        interactions[0].when.body.as_deref().unwrap(),
        r#"{"q":"café —巴黎"}"#,
        "id-free multi-byte bodies pass through untouched"
    );
}

#[test]
fn scrub_drops_noise_response_headers_and_rewrites_sensitive_queries() {
    let mut interactions = vec![Interaction {
        when: WhenSpec {
            query_param: Some(vec![
                Pair {
                    name: "key".to_string(),
                    value: "AIzaSyA-real-google-key-0123456789".to_string(),
                },
                Pair {
                    name: "alt".to_string(),
                    value: "sse".to_string(),
                },
            ]),
            ..Default::default()
        },
        then: ThenSpec {
            status: Some(200),
            header: Some(vec![
                Pair {
                    name: "content-type".to_string(),
                    value: "text/event-stream".to_string(),
                },
                Pair {
                    name: "retry-after".to_string(),
                    value: "7".to_string(),
                },
                Pair {
                    name: "x-request-id".to_string(),
                    value: "req-abc123".to_string(),
                },
                Pair {
                    name: "x-ratelimit-remaining-requests".to_string(),
                    value: "41".to_string(),
                },
            ]),
            ..Default::default()
        },
    }];

    scrub(&mut interactions);

    let surviving: Vec<(String, String)> = interactions[0]
        .then
        .header
        .as_ref()
        .unwrap()
        .iter()
        .map(|pair| (pair.name.clone(), pair.value.clone()))
        .collect();
    assert_eq!(
        surviving,
        vec![
            ("content-type".to_string(), "text/event-stream".to_string()),
            ("retry-after".to_string(), "7".to_string()),
        ],
        "only the allowlisted response headers survive the scrub — \
        provider request-ids and ratelimit buckets are noise at best \
        and a leak at worst"
    );
    let queries: Vec<(String, String)> = interactions[0]
        .when
        .query_param
        .as_ref()
        .unwrap()
        .iter()
        .map(|pair| (pair.name.clone(), pair.value.clone()))
        .collect();
    assert_eq!(
        queries,
        vec![
            ("key".to_string(), cassette::CASSETTE_KEY.to_string()),
            ("alt".to_string(), "sse".to_string()),
        ],
        "a sensitive query value is rewritten to the fixed dummy — the \
        same value replay clients send, so the bytes still match — while \
        benign pairs pass through untouched"
    );
}

#[test]
fn renames_handle_prefix_overlapping_ids() {
    let mut interactions = vec![Interaction {
        when: WhenSpec {
            body: Some(r#"{"echo":"msg_abcd","note":"msg_abextra"}"#.to_string()),
            ..Default::default()
        },
        then: ThenSpec {
            status: Some(200),
            body: Some(r#"{"a":"msg_ab","b":"msg_abcd"}"#.to_string()),
            ..Default::default()
        },
    }];

    scrub(&mut interactions);

    assert_eq!(
        interactions[0].then.body.as_deref().unwrap(),
        r#"{"a":"msg_cassette_1","b":"msg_cassette_2"}"#,
        "each minted id keeps its own placeholder even when one is a \
        strict prefix of the other — sequential replacement would \
        corrupt the longer one into `msg_cassette_1cd`"
    );
    assert_eq!(
        interactions[0].when.body.as_deref().unwrap(),
        r#"{"echo":"msg_cassette_2","note":"msg_abextra"}"#,
        "the longer id's echo renames whole, and an id-shaped token that \
        was never minted stays untouched — ids rename as tokens, not as \
        substrings"
    );
}

#[test]
fn renames_require_a_leading_token_boundary() {
    let mut interactions = vec![Interaction {
        when: WhenSpec {
            body: Some(r#"{"prompt":"foomsg_ab and msg_ab"}"#.to_string()),
            ..Default::default()
        },
        then: ThenSpec {
            status: Some(200),
            body: Some(r#"{"id":"msg_ab","noise":"foomsg_cd"}"#.to_string()),
            ..Default::default()
        },
    }];

    scrub(&mut interactions);

    assert_eq!(
        interactions[0].when.body.as_deref().unwrap(),
        r#"{"prompt":"foomsg_ab and msg_cassette_1"}"#,
        "an id-shaped suffix inside a client word must survive verbatim — \
        rewriting it would scrub the recorded request into bytes the \
        replay client never sends, and the miss would read as client drift"
    );
    assert_eq!(
        interactions[0].then.body.as_deref().unwrap(),
        r#"{"id":"msg_cassette_1","noise":"foomsg_cd"}"#,
        "the standalone minted id renames while a mid-word suffix never \
        even enters the rename table — the leading boundary narrows both \
        discovery and replacement"
    );
}

#[test]
fn a_key_shape_inside_an_opaque_blob_is_not_flagged() {
    let signature = format!("{}AIza{}", "x".repeat(25), "y".repeat(35));
    let blob = format!("{{\"thoughtSignature\": \"{signature}\"}}");
    assert!(
        scan_for_secrets(&blob).is_empty(),
        "a key shape that begins mid-token, inside a base64 signature \
        blob, is payload coincidence — flagging it would block CI on a \
        coincidence"
    );
    let sk_blob = format!("{}sk-{}", "x".repeat(25), "y".repeat(48));
    assert!(
        scan_for_secrets(&sk_blob).is_empty(),
        "an sk- shape mid-token inside a blob is the same payload class"
    );

    let standalone = format!("key: AIza{}", "z".repeat(35));
    assert!(
        scan_for_secrets(&standalone)
            .iter()
            .any(|v| v.contains("Google-style key")),
        "a standalone key of the same shape must still be flagged"
    );
    for real_length in [
        format!("key: sk-{}", "a".repeat(48)),
        format!("key: sk-proj-{}", "b".repeat(43)),
    ] {
        assert!(
            scan_for_secrets(&real_length)
                .iter()
                .any(|v| v.contains("OpenAI-style key")),
            "a real-length standalone OpenAI key ({real_length:?}) — itself a             run of 51+ word characters — must be flagged: a credential             always arrives delimited, so its needle starts a token, and             length alone must never exempt it"
        );
    }
}

#[test]
fn cassette_safety_flags_a_planted_key() {
    let planted = r#"
- when:
    header:
    - name: authorization
      value: Bearer sk-proj-0123456789abcdef0123
    body: '{"model":"m"}'
  then:
    status: 200
    body: '{"id":"chatcmpl-1"}'
"#;
    let violations = scan_for_secrets(planted);
    assert!(
        violations.iter().any(|v| v.contains("authorization")),
        "the forbidden header must be flagged, got {violations:?}"
    );
    assert!(
        violations.iter().any(|v| v.contains("OpenAI-style key")),
        "the planted key shape must be flagged, got {violations:?}"
    );

    let clean = r#"
- when:
    body: '{"model":"m"}'
  then:
    status: 200
    body: '{"id":"chatcmpl-cassette_1"}'
"#;
    assert!(
        scan_for_secrets(clean).is_empty(),
        "a scrubbed cassette must scan clean"
    );
}

#[tokio::test]
#[should_panic(expected = "binary cassette bodies")]
async fn binary_cassette_bodies_are_refused_at_replay() {
    let interaction = Interaction {
        when: WhenSpec {
            body_base64: Some("aGVsbG8=".to_string()),
            ..Default::default()
        },
        then: ThenSpec {
            status: Some(200),
            ..Default::default()
        },
    };
    let server = MockServer::start_async().await;
    register_replay_mocks(&server, "", &[interaction]).await;
}

#[cfg(feature = "testing")]
#[tokio::test]
#[should_panic(expected = "needs OPENAI_API_KEY")]
async fn record_mode_requires_provider_credentials() {
    let env =
        loopctl::testing::EnvGuard::acquire(&["LOOPCTL_E2E", "LOOPCTL_CASSETTE", "OPENAI_API_KEY"]);
    env.set("LOOPCTL_E2E", "1");
    env.set("LOOPCTL_CASSETTE", "record");
    env.remove("OPENAI_API_KEY");

    let server = MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        "openai-compat",
        "gate-pin",
        "https://api.openai.com/v1",
        &server,
    )
    .await;
    drop(session);
}

#[cfg(feature = "testing")]
#[tokio::test]
#[should_panic(expected = "recording is a deliberate human act")]
async fn record_mode_requires_credentials() {
    let env =
        loopctl::testing::EnvGuard::acquire(&["LOOPCTL_E2E", "LOOPCTL_CASSETTE", "OPENAI_API_KEY"]);
    env.set("LOOPCTL_CASSETTE", "record");
    env.remove("LOOPCTL_E2E");
    env.remove("OPENAI_API_KEY");

    let server = MockServer::start_async().await;
    let session = cassette::CassetteSession::start(
        "openai-compat",
        "gate-pin",
        "https://api.openai.com/v1",
        &server,
    )
    .await;
    drop(session);
}

#[cfg(feature = "testing")]
#[test]
fn unconfigured_environment_defaults_to_replay() {
    use cassette::{CassetteMode, mode_from_env};
    let env = loopctl::testing::EnvGuard::acquire(&["LOOPCTL_CASSETTE"]);
    env.remove("LOOPCTL_CASSETTE");
    assert_eq!(
        mode_from_env(),
        CassetteMode::Replay,
        "an unset or unknown value must never select record — recording \
        cannot happen by accident"
    );
    env.set("LOOPCTL_CASSETTE", "record");
    assert_eq!(
        mode_from_env(),
        CassetteMode::Record,
        "the explicit opt-in selects record"
    );
}
