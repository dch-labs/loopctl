//! Reasoning round-trip: thinking blocks survive the trip back to the wire.
//!
//! The invariant the fidelity work exists for, stated once and pinned per
//! provider: an assistant message's reasoning content, once received with
//! its continuity tokens — the provider signature, the redacted payload —
//! is preserved verbatim when that message is re-sent in a later request,
//! and is provably omitted (never mangled, never half-echoed) where the
//! dialect refuses it.
//!
//! Each provider pin drives the real client against a local scripted
//! server and asserts the *outbound bytes* of the follow-up request, the
//! request a continuation actually rides — converter-level pins in the
//! provider modules guard the same arms in isolation; these are the
//! end-to-end proofs:
//!
//! - **Anthropic direct** — a signed thinking block ahead of a tool call
//!   rides back as the native `thinking` block, signature verbatim, under
//!   an effective thinking effort.
//! - **Gemini** — a `thoughtSignature` latched from a thought part rides
//!   back on the next request, through a tool turn whose dispatch pipeline
//!   contains [`RedactingMiddleware`] — the scrubber eats the token where
//!   it appears in tool output and cannot touch it where it lives on the
//!   lane, the seam boundary the redaction module documents.
//! - **Gemini parity** — the streaming and non-streaming transports yield
//!   identical thinking parts for the same response, the 0.3.0
//!   mock-parity discipline applied to the fidelity fields.
//! - **OpenAI-compatible** — replayed reasoning rides the
//!   `reasoning_content` field and nothing else; the dialect has no
//!   signature lane, and none is invented.
//! - **Cross-cutting** — pre-fidelity serialized histories deserialize
//!   with defaulted fields, and the demotion render summarizes a thinking
//!   lane without ever emitting its signature or redacted payload.
//!
//! Requires the provider features per scenario (`anthropic`, `gemini`,
//! `openai`, `redaction`); the cross-cutting pins run on the core alone.

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

use std::future::Future;
use std::pin::Pin;

use loopctl::message::{Message, MessagePart, Role};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};
use serde_json::{Value, json};

/// A tool whose output echoes a dense signature-shaped token, assembled
/// from pieces so the contiguous form never appears in source.
///
/// The echo is what the redaction interplay pin needs: the same bytes a
/// thinking lane carries as a signature are, in tool output, exactly the
/// high-entropy material the scrubber exists to mask.
fn echo_token() -> String {
    ["qW3nT7", "zX9mK4", "pL2vB8", "cR5jH6", "n4RtY6", "xZ9mL2"].concat()
}

/// The `echo_sig` tool the engine scenarios dispatch.
struct EchoSigTool;

impl Tool for EchoSigTool {
    fn name(&self) -> &str {
        "echo_sig"
    }

    fn description(&self) -> &str {
        "Echoes a dense token for the round-trip scenarios"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo_sig",
            "Echoes a dense token for the round-trip scenarios",
            json!({"type": "object", "properties": {}, "required": []}),
        )
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move { Ok(ToolOutput::text(echo_token())) })
    }
}

/// The Nth textual request body of a recording export (0-indexed).
///
/// The export is a multi-document YAML stream with one interaction per
/// request, bodies in arrival order; panics when fewer than `index` + 1
/// documents carry a textual body.
fn nth_recorded_request_body(export: &str, index: usize) -> String {
    let mut seen = 0usize;
    for document in serde_yaml::Deserializer::from_str(export) {
        let value: serde_yaml::Value =
            serde::Deserialize::deserialize(document).expect("the recording export parses");
        let body = value
            .get("when")
            .and_then(|when| when.get("body"))
            .and_then(|body| body.as_str());
        if let Some(body) = body {
            if seen == index {
                return body.to_string();
            }
            seen = seen.saturating_add(1);
        }
    }
    panic!("the recording holds only {seen} textual bodies; wanted index {index}");
}

/// Parse a recorded body as JSON.
fn parse_body(body: &str) -> Value {
    serde_json::from_str(body).expect("the recorded body parses as JSON")
}

/// A Gemini non-streaming response carrying one thought part signed
/// with `signature` and one function call, as the first turn of a tool
/// loop.
fn gemini_tool_turn(signature: &str) -> Value {
    json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "pondering", "thought": true, "thoughtSignature": signature},
                    {"functionCall": {"name": "echo_sig", "args": {}}}
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
    })
}

#[tokio::test]
#[cfg(all(feature = "gemini", feature = "redaction"))]
async fn a_signed_thought_part_rides_the_next_gemini_request_through_a_redacting_pipeline() {
    use std::sync::Arc;

    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig, TurnMode};
    use loopctl::middleware::{RedactingMiddleware, SecretPatternSet};
    use loopctl::tool::ToolRegistry;

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    // Registered specific-first: the follow-up request alone carries the
    // redacted tool output, and the general matcher must not swallow it.
    let followup = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("[REDACTED:");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "candidates": [{
                            "content": {"parts": [{"text": "done"}]},
                            "finishReason": "STOP"
                        }]
                    })
                    .to_string(),
                );
        })
        .await;
    let opening = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("please echo");
            then.status(200)
                .header("content-type", "application/json")
                .body(gemini_tool_turn(&echo_token()).to_string());
        })
        .await;

    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1beta", server.base_url()))
        .with_model("round-trip-model")
        .build()
        .expect("the gemini client builds");

    let registry = ToolRegistry::new();
    registry.register(EchoSigTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    agent.set_turn_mode(TurnMode::NonStreaming);
    agent
        .set_pipeline(
            loopctl::middleware::ToolPipeline::builder()
                .with_middleware(RedactingMiddleware::new(SecretPatternSet::default_common())),
        )
        .expect("the pipeline installs");

    let result = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(result.is_ok(), "the run completes: {result:?}");

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the engine sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let followup_body = parse_body(&nth_recorded_request_body(&export, 1));

    let parts_of = |body: &Value| -> Vec<Value> {
        body.pointer("/contents")
            .and_then(Value::as_array)
            .expect("the follow-up carries contents")
            .iter()
            .flat_map(|message| {
                message
                    .pointer("/parts")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect()
    };
    let token = echo_token();
    let followup_parts = parts_of(&followup_body);
    let carried = followup_parts.iter().any(|part| {
        part.get("thoughtSignature").and_then(Value::as_str) == Some(token.as_str())
            && part.get("thought") == Some(&json!(true))
    });
    assert!(
        carried,
        "the signed thought part rides the follow-up request with the exact \
         token the tool echoed: {followup_body}"
    );
    let response_part = followup_parts
        .iter()
        .find(|part| part.get("functionResponse").is_some())
        .expect("the follow-up carries the tool result as a functionResponse part");
    let rendered = response_part.to_string();
    assert!(
        rendered.contains("[REDACTED:"),
        "the tool-output echo of the same bytes was scrubbed where it rode the \
         output lane: {rendered}"
    );
    assert!(
        !rendered.contains(token.as_str()),
        "the scrubbed output lane must not carry the raw token — the marker \
         replaces it, byte for byte: {rendered}"
    );

    opening.delete_async().await;
    followup.delete_async().await;
}

#[tokio::test]
#[cfg(all(feature = "gemini", feature = "redaction"))]
async fn a_call_part_signature_rides_the_next_gemini_request() {
    use std::sync::Arc;

    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig, TurnMode};
    use loopctl::middleware::{RedactingMiddleware, SecretPatternSet};
    use loopctl::tool::ToolRegistry;

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let followup = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("[REDACTED:");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "candidates": [{
                            "content": {"parts": [{"text": "done"}]},
                            "finishReason": "STOP"
                        }]
                    })
                    .to_string(),
                );
        })
        .await;
    let opening = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("please echo");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "candidates": [{
                            "content": {
                                "parts": [{
                                    "functionCall": {"name": "echo_sig", "args": {}},
                                    "thoughtSignature": "sig-fc"
                                }]
                            },
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                    })
                    .to_string(),
                );
        })
        .await;

    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1beta", server.base_url()))
        .with_model("round-trip-model")
        .build()
        .expect("the gemini client builds");

    let registry = ToolRegistry::new();
    registry.register(EchoSigTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    agent.set_turn_mode(TurnMode::NonStreaming);
    agent
        .set_pipeline(
            loopctl::middleware::ToolPipeline::builder()
                .with_middleware(RedactingMiddleware::new(SecretPatternSet::default_common())),
        )
        .expect("the pipeline installs");

    let result = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(result.is_ok(), "the run completes: {result:?}");

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the engine sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let followup_body = parse_body(&nth_recorded_request_body(&export, 1));

    let carried = followup_body
        .pointer("/contents")
        .and_then(Value::as_array)
        .expect("the follow-up carries contents")
        .iter()
        .flat_map(|message| {
            message
                .pointer("/parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .any(|part| {
            part.get("functionCall").is_some()
                && part.get("thoughtSignature").and_then(Value::as_str) == Some("sig-fc")
        });
    assert!(
        carried,
        "the signature riding the call part returns beside its function call, \
         the documented carrier for reasoning continuity across a tool turn: {followup_body}"
    );

    opening.delete_async().await;
    followup.delete_async().await;
}

/// The (path, body) of the Nth recorded request, or `None` past the end.
fn nth_recorded_request_quiet(export: &str, index: usize) -> Option<(String, String)> {
    let mut seen = 0usize;
    for document in serde_yaml::Deserializer::from_str(export) {
        let value: serde_yaml::Value =
            serde::Deserialize::deserialize(document).expect("the recording export parses");
        let when = value.get("when");
        let body = when
            .and_then(|when| when.get("body"))
            .and_then(|body| body.as_str());
        if let Some(body) = body {
            if seen == index {
                let path = when
                    .and_then(|when| when.get("path"))
                    .and_then(|path| path.as_str())
                    .unwrap_or_default()
                    .to_string();
                return Some((path, body.to_string()));
            }
            seen = seen.saturating_add(1);
        }
    }
    None
}

/// The (path, body) of the Nth recorded request, in arrival order.
fn nth_recorded_request(export: &str, index: usize) -> (String, String) {
    let mut seen = 0usize;
    for document in serde_yaml::Deserializer::from_str(export) {
        let value: serde_yaml::Value =
            serde::Deserialize::deserialize(document).expect("the recording export parses");
        let when = value.get("when");
        let body = when
            .and_then(|when| when.get("body"))
            .and_then(|body| body.as_str());
        if let Some(body) = body {
            if seen == index {
                let path = when
                    .and_then(|when| when.get("path"))
                    .and_then(|path| path.as_str())
                    .unwrap_or_default()
                    .to_string();
                return (path, body.to_string());
            }
            seen = seen.saturating_add(1);
        }
    }
    panic!("the recording holds only {seen} textual bodies; wanted index {index}");
}

/// An SSE body of `data:` frames, one per chunk.
fn sse(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {}\n\n", chunk));
    }
    body
}

#[tokio::test]
#[cfg(feature = "gemini")]
async fn a_signed_thought_part_rides_the_next_gemini_request_over_streaming() {
    use std::sync::Arc;

    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig, TurnMode};
    use loopctl::tool::ToolRegistry;

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let followup = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/round-trip-model:streamGenerateContent")
                .body_includes(echo_token());
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(sse(&[json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "done"}]},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                })]));
        })
        .await;
    let opening = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/round-trip-model:streamGenerateContent")
                .body_includes("please echo");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(sse(&[
                    json!({
                        "candidates": [{
                            "content": {"parts": [
                                {"text": "pondering", "thought": true, "thoughtSignature": "sig-1"},
                                {"functionCall": {"name": "echo_sig", "args": {}}}
                            ]}
                        }]
                    }),
                    json!({
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                    }),
                ]));
        })
        .await;

    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1beta", server.base_url()))
        .with_model("round-trip-model")
        .build()
        .expect("the gemini client builds");
    let registry = ToolRegistry::new();
    registry.register(EchoSigTool);
    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    agent.set_turn_mode(TurnMode::Streaming);
    let result = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(result.is_ok(), "the run completes: {result:?}");

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the engine sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let (_, followup_body) = nth_recorded_request(&export, 1);
    let carried = parse_body(&followup_body)
        .pointer("/contents")
        .and_then(Value::as_array)
        .expect("the follow-up carries contents")
        .iter()
        .flat_map(|message| {
            message
                .pointer("/parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .any(|part| part.get("thoughtSignature").and_then(Value::as_str) == Some("sig-1"));
    assert!(
        carried,
        "the streaming capture path preserves the signature onto the follow-up request: {followup_body}"
    );

    opening.delete_async().await;
    followup.delete_async().await;
}

#[tokio::test]
#[cfg(all(feature = "gemini", feature = "redaction"))]
async fn a_fallback_routed_follow_up_still_carries_the_signature() {
    use std::sync::Arc;

    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig, TurnMode};
    use loopctl::fallback::{FallbackConfig, FallbackManager};
    use loopctl::managers::LoopManagers;
    use loopctl::middleware::{RedactingMiddleware, SecretPatternSet};
    use loopctl::tool::ToolRegistry;

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let primary = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/primary-model:streamGenerateContent");
            then.status(429).body("rate limited");
        })
        .await;
    let followup = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/fallback-model:streamGenerateContent")
                .body_includes("[REDACTED:");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(sse(&[json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "done"}]},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                })]));
        })
        .await;
    let opening = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/fallback-model:streamGenerateContent")
                .body_includes("please echo");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(sse(&[
                    json!({
                        "candidates": [{
                            "content": {"parts": [
                                {"text": "pondering", "thought": true, "thoughtSignature": "sig-fc"},
                                {"functionCall": {"name": "echo_sig", "args": {}}}
                            ]}
                        }]
                    }),
                    json!({
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                    }),
                ]));
        })
        .await;

    let manager = FallbackManager::new_with_config(FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_secs(3600),
        ..FallbackConfig::default()
    });
    manager
        .set_original_model("primary-model".to_string())
        .expect("the original model sets");
    manager
        .set_fallback_model("fallback-model")
        .expect("the fallback model sets");

    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1beta", server.base_url()))
        .with_model("primary-model")
        .build()
        .expect("the gemini client builds");
    let registry = ToolRegistry::new();
    registry.register(EchoSigTool);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(client),
        registry,
        SessionConfig::default(),
        LoopManagers::new().with_fallback(manager),
    );
    agent.set_turn_mode(TurnMode::Streaming);
    agent
        .set_pipeline(
            loopctl::middleware::ToolPipeline::builder()
                .with_middleware(RedactingMiddleware::new(SecretPatternSet::default_common())),
        )
        .expect("the pipeline installs");

    let first = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(
        first.is_err(),
        "the primary-model run fails on the rate limit"
    );
    let second = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(
        second.is_ok(),
        "the fallback-served run completes: {second:?}"
    );

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the engine sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let fallback_bodies: Vec<(usize, String, String)> = (0..8)
        .filter_map(|index| {
            let (path, body) = nth_recorded_request_quiet(&export, index)?;
            if path.contains("fallback-model") {
                Some((index, path, body))
            } else {
                None
            }
        })
        .collect();
    let carried = fallback_bodies.iter().any(|(_, _, body)| {
        parse_body(body)
            .pointer("/contents")
            .and_then(Value::as_array)
            .is_some_and(|contents| {
                contents.iter().any(|message| {
                    message
                        .pointer("/parts")
                        .and_then(Value::as_array)
                        .is_some_and(|parts| {
                            parts.iter().any(|part| {
                                part.get("thoughtSignature").and_then(Value::as_str)
                                    == Some("sig-fc")
                            })
                        })
                })
            })
    });
    assert!(
        carried,
        "the fallback-routed follow-up still carries the signature verbatim: {fallback_bodies:?}"
    );

    primary.delete_async().await;
    opening.delete_async().await;
    followup.delete_async().await;
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn a_signed_thinking_block_rides_the_next_anthropic_request_under_effort() {
    use std::sync::Arc;

    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig, TurnMode};
    use loopctl::structured::ThinkingEffort;
    use loopctl::tool::ToolRegistry;

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let followup = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("tool_result");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 3}
                    })
                    .to_string(),
                );
        })
        .await;
    let opening = server
        .mock_async(|when, then| {
            when.method("POST").body_includes("please echo");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "content": [
                            {"type": "thinking", "thinking": "hm", "signature": "sig-1"},
                            {"type": "tool_use", "id": "t1", "name": "echo_sig", "input": {}}
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 5, "output_tokens": 7}
                    })
                    .to_string(),
                );
        })
        .await;

    let client = loopctl::provider::AnthropicClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(server.base_url())
        .with_model("round-trip-model")
        .thinking_effort(ThinkingEffort::High)
        .build()
        .expect("the anthropic client builds");

    let registry = ToolRegistry::new();
    registry.register(EchoSigTool);

    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    agent.set_turn_mode(TurnMode::NonStreaming);
    let result = agent
        .run("please echo the token", &RunConfig::default())
        .await;
    assert!(result.is_ok(), "the run completes: {result:?}");

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the engine sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let followup_body = parse_body(&nth_recorded_request_body(&export, 1));

    let blocks = followup_body
        .pointer("/messages")
        .and_then(Value::as_array)
        .expect("the follow-up carries messages")
        .iter()
        .flat_map(|message| {
            message
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let thinking_position = blocks
        .iter()
        .position(|block| block.get("type").and_then(Value::as_str) == Some("thinking"));
    let tool_result_position = blocks
        .iter()
        .position(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"));
    let (thinking, position) = thinking_position
        .map(|position| (&blocks[position], position))
        .expect("the signed thinking block rides the follow-up");
    assert_eq!(thinking.get("thinking").and_then(Value::as_str), Some("hm"));
    assert_eq!(
        thinking.get("signature").and_then(Value::as_str),
        Some("sig-1")
    );
    assert!(
        tool_result_position.is_some_and(|result_position| position < result_position),
        "the thinking block precedes the tool result, the order the Messages API demands: {blocks:?}"
    );

    opening.delete_async().await;
    followup.delete_async().await;
}

#[tokio::test]
#[cfg(feature = "gemini")]
async fn gemini_streaming_and_non_streaming_yield_identical_thinking_parts() {
    use futures::StreamExt;
    use loopctl::api::{ApiClient, StreamRequest};
    use loopctl::stream::StreamAccumulator;

    let server = httpmock::MockServer::start_async().await;
    let stream_mock = server
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1beta/models/m:streamGenerateContent");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(format!(
                    "data: {}\n\ndata: {}\n\ndata: {}\n\n",
                    json!({
                        "candidates": [{
                            "content": {"parts": [{"text": "alpha ", "thought": true}]}
                        }]
                    }),
                    json!({
                        "candidates": [{
                            "content": {"parts": [
                                {"text": "beta", "thought": true, "thoughtSignature": "sig-final"}
                            ]}
                        }]
                    }),
                    json!({
                        "candidates": [{
                            "content": {"parts": [{"text": "done"}]},
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                    })
                ));
        })
        .await;
    let plain_mock = server
        .mock_async(|when, then| {
            when.method("POST").path("/v1beta/models/m:generateContent");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "candidates": [{
                            "content": {"parts": [
                                {"text": "alpha beta", "thought": true, "thoughtSignature": "sig-final"},
                                {"text": "done"}
                            ]},
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}
                    })
                    .to_string(),
                );
        })
        .await;

    let client = loopctl::provider::GeminiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1beta", server.base_url()))
        .with_model("m")
        .build()
        .expect("the gemini client builds");
    let request = StreamRequest::new(vec![Message::user("hi")]);

    let mut stream = client.stream_messages(&request);
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        let event = event.expect("the stream yields an event");
        accumulator
            .process(&event)
            .expect("the accumulator accepts the event");
    }
    let streamed = accumulator.build();

    let plain = client
        .create_message(&request)
        .await
        .expect("the non-streaming call answers");

    let shape = |parts: &[MessagePart]| -> Vec<(&'static str, String, Option<String>)> {
        parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::Thinking {
                    text, signature, ..
                } => Some(("thinking", text.clone(), signature.clone())),
                MessagePart::Text { text } => Some(("text", text.clone(), None)),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        shape(&streamed.parts),
        shape(&plain.message.parts),
        "the two transports yield identical thinking and text parts, kind for kind"
    );
    assert_eq!(
        shape(&plain.message.parts),
        vec![
            (
                "thinking",
                "alpha beta".to_string(),
                Some("sig-final".to_string())
            ),
            ("text", "done".to_string(), None),
        ],
        "a thought streamed as fragment chunks and closed by its signed final \
         delta rebuilds as one complete part carrying the signature computed \
         over the whole summary - a split on every chunk boundary would \
         replay a truncated fragment under that signature"
    );

    stream_mock.delete_async().await;
    plain_mock.delete_async().await;
}

#[tokio::test]
#[cfg(feature = "openai")]
async fn replayed_reasoning_rides_the_openai_wire_as_text_and_only_text() {
    use loopctl::api::{ApiClient, StreamRequest};

    let server = httpmock::MockServer::start_async().await;
    let recording = server
        .record_async(|rule| {
            rule.filter(|when| {
                when.any_request();
            });
        })
        .await;
    let mock = server
        .mock_async(|when, then| {
            when.method("POST");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    json!({
                        "choices": [{
                            "message": {
                                "content": null,
                                "reasoning_content": "step 1",
                                "tool_calls": [{
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "echo_sig", "arguments": "{}"}
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 7}
                    })
                    .to_string(),
                );
        })
        .await;

    let client = loopctl::provider::OpenAiClient::builder()
        .with_api_key("round-trip-key")
        .with_base_url(format!("{}/v1", server.base_url()))
        .with_model("round-trip-model")
        .replay_reasoning(true)
        .build()
        .expect("the openai client builds");

    let first = StreamRequest::new(vec![Message::user("please echo")]);
    let response = client
        .create_message(&first)
        .await
        .expect("the first call answers");
    let assistant = response.message;
    let followup = StreamRequest::new(vec![
        Message::user("please echo"),
        assistant,
        Message::new(
            Role::User,
            vec![loopctl::message::MessagePart::tool_result(
                "t1", "echo_sig", "ran fine", false,
            )],
        ),
    ]);
    client
        .create_message(&followup)
        .await
        .expect("the follow-up call answers");

    let bytes = recording
        .export_async()
        .await
        .expect("the recording exports")
        .expect("the client sent its requests");
    let export = String::from_utf8_lossy(&bytes);
    let followup_body = parse_body(&nth_recorded_request_body(&export, 1));
    let assistant_message = followup_body
        .pointer("/messages/1")
        .expect("the follow-up carries the assistant message");
    assert_eq!(
        assistant_message
            .get("reasoning_content")
            .and_then(Value::as_str),
        Some("step 1"),
        "replay rides the field the dialect owns"
    );
    assert!(
        !export.contains("signature") && !export.contains("redacted"),
        "no signature or redacted payload may be invented for a dialect without one"
    );

    mock.delete_async().await;
}

#[test]
fn pre_fidelity_thinking_parts_deserialize_with_defaulted_fields() {
    let message = Message::new(
        Role::Assistant,
        vec![
            MessagePart::Thinking {
                text: "step 1".to_string(),
                signature: Some("sig-1".to_string()),
                redacted: Some("b3BhcXVl".to_string()),
            },
            MessagePart::tool_call_with_signature(
                "call_1",
                "echo",
                serde_json::json!({"a": 1}),
                Some("sig-fc".to_string()),
            ),
        ],
    );
    let serialized = serde_json::to_value(&message).expect("the message serializes");
    let stripped = strip_fidelity_fields(serialized);
    let deserialized: Message =
        serde_json::from_value(stripped).expect("the pre-fidelity shape deserializes");
    match &deserialized.parts[0] {
        MessagePart::Thinking {
            text,
            signature,
            redacted,
        } => {
            assert_eq!(text, "step 1");
            assert!(signature.is_none(), "an absent signature defaults to None");
            assert!(redacted.is_none(), "an absent payload defaults to None");
        }
        other => panic!("expected a Thinking part, got {other:?}"),
    }
    match &deserialized.parts[1] {
        MessagePart::ToolCall { id, signature, .. } => {
            assert_eq!(id, "call_1");
            assert!(
                signature.is_none(),
                "a call part persisted before the field existed deserializes unsigned"
            );
        }
        other => panic!("expected a ToolCall part, got {other:?}"),
    }

    let round_tripped: Message =
        serde_json::from_value(serde_json::to_value(&message).expect("the message serializes"))
            .expect("the current shape round-trips");
    match round_tripped.parts.first().expect("the part survives") {
        MessagePart::Thinking {
            signature,
            redacted,
            ..
        } => {
            assert_eq!(signature.as_deref(), Some("sig-1"));
            assert_eq!(redacted.as_deref(), Some("b3BhcXVl"));
        }
        other => panic!("expected a Thinking part, got {other:?}"),
    }
    match &round_tripped.parts[1] {
        MessagePart::ToolCall { signature, .. } => {
            assert_eq!(signature.as_deref(), Some("sig-fc"));
        }
        other => panic!("expected a ToolCall part, got {other:?}"),
    }
}

/// Remove `signature` and `redacted` from every part of a serialized
/// message, yielding the pre-fidelity wire shape for both the thinking
/// and the tool-call lanes.
fn strip_fidelity_fields(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value
        && let Some(parts) = map.get_mut("parts").and_then(Value::as_array_mut)
    {
        for part in parts {
            if let Value::Object(part_map) = part {
                part_map.remove("signature");
                part_map.remove("redacted");
            }
        }
    }
    value
}

#[test]
fn the_demotion_render_summarizes_thinking_without_its_fidelity_payloads() {
    let redacted_lane = vec![Message::new(
        Role::Assistant,
        vec![MessagePart::Thinking {
            text: "withheld reasoning".to_string(),
            signature: Some("sig-1".to_string()),
            redacted: Some("b3BhcXVl".to_string()),
        }],
    )];
    let rendered = loopctl::compact::render_evicted(&redacted_lane, 8000);
    assert!(
        rendered.contains("thought: [redacted]"),
        "a redacted lane renders its marker: {rendered}"
    );
    assert!(
        !rendered.contains("withheld reasoning")
            && !rendered.contains("sig-1")
            && !rendered.contains("b3BhcXVl"),
        "neither the withheld text nor either fidelity payload may leak: {rendered}"
    );

    let visible_lane = vec![Message::new(
        Role::Assistant,
        vec![MessagePart::Thinking {
            text: "keep this summary".to_string(),
            signature: Some("sig-1".to_string()),
            redacted: None,
        }],
    )];
    let rendered = loopctl::compact::render_evicted(&visible_lane, 8000);
    assert!(
        rendered.contains("thought: keep this summary"),
        "a visible lane renders its summary: {rendered}"
    );
    assert!(
        !rendered.contains("sig-1"),
        "the signature never renders, even beside its own text: {rendered}"
    );

    let signature_only = vec![Message::new(
        Role::Assistant,
        vec![MessagePart::Thinking {
            text: String::new(),
            signature: Some("sig-1".to_string()),
            redacted: None,
        }],
    )];
    let rendered = loopctl::compact::render_evicted(&signature_only, 8000);
    assert!(
        !rendered.contains("thought:"),
        "a signature-only lane has nothing displayable to summarize and renders no \
         line — the token lives on the part, not in any rendered text: {rendered}"
    );
    assert!(
        !rendered.contains("sig-1"),
        "and the signature itself never renders: {rendered}"
    );
}
