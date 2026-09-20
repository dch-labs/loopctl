//! Engine-level resilience pins for the small-model machinery.
//!
//! Drives [`BareLoop`] against [`MockApiClient`] and a scripted slow
//! stream to pin the contract edges the in-crate engine tests leave
//! implicit: strict constraints through the mock, recovery after a hard
//! failure, mid-stream cancellation committing nothing, and the
//! zero-timeout probe path through the tool-health gate.

#![cfg(feature = "testing")]
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

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use loopctl::api::error::ApiError;
use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::stream::{MessageMetadata, MessageStart, StreamEvent};
use loopctl::structured::{RequestOptions, ToolConstraint};
use loopctl::testing::{MockApiClient, MockResponse};
use loopctl::tool::ToolRegistry;

#[tokio::test]
async fn a_strict_constraint_drives_the_engine_through_the_mock() {
    let client = MockApiClient::new("m")
        .with_tool_constraint_support()
        .with_text_response("done");
    let mut agent = BareLoop::new(
        Arc::new(client.clone()),
        ToolRegistry::new(),
        loopctl::config::SessionConfig::default(),
    );
    agent.set_request_options(RequestOptions::new().with_tool_constraint(ToolConstraint::Strict));

    let run = agent.run("strict", &RunConfig::default()).await;
    assert!(run.is_ok(), "the constrained run completes: {run:?}");
    assert!(
        client.with_options_calls() >= 1,
        "the engine drove the turn through the options-carrying path"
    );
}

#[tokio::test]
async fn a_hard_failed_run_recovers_on_the_next_run() {
    let client = MockApiClient::new("m")
        .with_text_response("recovered")
        .with_errors(vec![Some("hard transport failure".to_string())]);
    let mut agent = BareLoop::new(
        Arc::new(client),
        ToolRegistry::new(),
        loopctl::config::SessionConfig::default(),
    );

    let first = agent.run("q", &RunConfig::default()).await;
    assert!(first.is_err(), "the scripted hard failure fails the run");
    let second = agent.run("q", &RunConfig::default()).await;
    assert!(
        second.is_ok(),
        "a non-cancel hard failure leaves the loop reusable: {second:?}"
    );
    let conversation = agent.conversation();
    assert_eq!(
        conversation.len(),
        2,
        "the failed run committed nothing; the second run's exchange is the \
         whole history: {:?}",
        conversation.iter().map(|m| m.role).collect::<Vec<_>>()
    );
}

/// A streaming client that emits the message start, then waits until
/// cancelled — a stream that never completes on its own.
struct StalledStreamClient;

impl ApiClient for StalledStreamClient {
    fn model(&self) -> String {
        "stalled".to_string()
    }

    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(
            futures::stream::once(async {
                Ok(StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_stall".into(),
                        role: "assistant".into(),
                        model: "stalled".into(),
                    },
                }))
            })
            .chain(futures::stream::pending()),
        )
    }

    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        // The non-streaming fallback stalls too, so the engine's biased
        // cancellation select is what ends the turn — not an error.
        Box::pin(std::future::pending())
    }
}

#[tokio::test]
async fn a_mid_stream_cancel_commits_nothing() {
    let mut agent = BareLoop::new(
        Arc::new(StalledStreamClient),
        ToolRegistry::new(),
        loopctl::config::SessionConfig::default(),
    );
    let signal = agent.cancel_signal();
    let handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        signal.cancel();
    });
    let run = agent.run("q", &RunConfig::default()).await;
    handle.await.unwrap();
    assert!(
        matches!(run, Err(loopctl::error::LoopError::Cancelled)),
        "the mid-stream cancel surfaces as a typed cancellation: {run:?}"
    );
    assert!(
        agent.conversation().is_empty(),
        "a stream that never completed commits nothing — not the prompt, \
         not a partial message"
    );
}

#[tokio::test]
#[cfg(feature = "tool_health")]
async fn a_zero_probe_timeout_still_converges_through_the_gate() {
    use loopctl::tool::health::ToolHealthRegistry;

    struct FailThenSucceed {
        fail_first: Arc<std::sync::atomic::AtomicBool>,
        executions: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl loopctl::tool::Tool for FailThenSucceed {
        fn name(&self) -> &'static str {
            "flaky"
        }
        fn description(&self) -> &'static str {
            "Fails while the flag is set"
        }
        fn schema(&self) -> loopctl::tool::ToolSchema {
            loopctl::tool::ToolSchema::new(
                self.name().to_string(),
                self.description().to_string(),
                serde_json::json!({"type": "object"}),
            )
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &loopctl::tool::ToolContext,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError>>
                    + Send
                    + '_,
            >,
        > {
            let fails = self.fail_first.load(std::sync::atomic::Ordering::SeqCst);
            let executions = Arc::clone(&self.executions);
            Box::pin(async move {
                executions.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if fails {
                    Err(loopctl::tool::ToolError::Execution(
                        "flaky failure".to_string(),
                    ))
                } else {
                    Ok(loopctl::tool::ToolOutput::text("ok"))
                }
            })
        }
    }

    let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let responses = vec![
        MockResponse {
            text: "go".to_string(),
            tool_call: Some(loopctl::testing::MockToolCall {
                id: "c".to_string(),
                name: "flaky".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ];

    let run_once = |fail: bool, health: Arc<ToolHealthRegistry>| {
        let client = MockApiClient::new("m").with_responses(responses.clone());
        let mut registry = ToolRegistry::new();
        registry.register(FailThenSucceed {
            fail_first: Arc::new(std::sync::atomic::AtomicBool::new(fail)),
            executions: Arc::clone(&executions),
        });
        let mut loop_ = BareLoop::new(
            Arc::new(client),
            registry,
            loopctl::config::SessionConfig::default(),
        );
        loop_.set_health_registry(health);
        async move { loop_.run("q", &RunConfig::default()).await }
    };

    let health = Arc::new(ToolHealthRegistry::new().with_config(
        loopctl::tool::health::CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_duration: Duration::from_millis(60),
            probe_timeout: Duration::ZERO,
        },
    ));
    run_once(true, Arc::clone(&health))
        .await
        .expect("run one completes");
    assert_eq!(executions.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Inside the cooldown the gate refuses without consuming a probe.
    run_once(true, Arc::clone(&health))
        .await
        .expect("run two completes");
    assert_eq!(
        executions.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the open breaker refuses dispatch inside the cooldown"
    );

    // A never-expiring probe lease is still released by its own result:
    // the post-cooldown probe executes, its failure re-trips, and the
    // next window's probe succeeds and closes.
    tokio::time::sleep(Duration::from_millis(80)).await;
    run_once(true, Arc::clone(&health))
        .await
        .expect("run three completes");
    assert_eq!(
        executions.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the post-cooldown probe executes and its failure re-trips"
    );
    tokio::time::sleep(Duration::from_millis(80)).await;
    run_once(false, Arc::clone(&health))
        .await
        .expect("run four completes");
    assert_eq!(
        executions.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "the next probe closes the breaker and dispatch resumes"
    );
}

#[test]
fn the_default_dispatch_mode_is_sequential() {
    assert!(
        matches!(
            RunConfig::default().parallel_tool_dispatch.mode,
            loopctl::config::ParallelMode::Sequential
        ),
        "the shipped default dispatches tool calls one at a time — \\
         concurrent dispatch is an opt-in, and this pin makes flipping \\
         that default a seen diff"
    );
}
