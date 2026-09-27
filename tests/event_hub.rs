//! Event-hub contracts over real engine runs.
//!
//! Pins that a scripted run forwards through a registered
//! [`EventHub`] as ordered [`LoopEvent`]s with per-run sequence
//! numbers, and that a consumer which never drains is lagged by the
//! channel instead of ever blocking the engine. The full
//! every-callback mapping is pinned in-crate, in `observer::hub`'s
//! own tests; these contracts drive the engine end to end.
//!
//! Requires the `testing` feature.

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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::observer::{EventHub, LoopEvent, ObservedEvent};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use tokio::sync::broadcast;

/// A tool that echoes a fixed successful output.
struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Echoes a fixed output"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name().to_string(),
            self.description().to_string(),
            serde_json::json!({"type": "object"}),
        )
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Ok(ToolOutput::text("done")) })
    }
}

/// One scripted turn: a text preamble plus the given tool call.
fn turn(text: &str, call: MockToolCall) -> Vec<MockResponse> {
    vec![MockResponse {
        text: text.to_string(),
        tool_call: Some(call),
        stop_reason: "tool_use".to_string(),
    }]
}

/// The scripted terminal response that ends a run with plain text.
fn terminal() -> MockResponse {
    MockResponse {
        text: "finished".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

/// A scripted `echo` tool call.
fn call(id: &str) -> MockToolCall {
    MockToolCall {
        id: id.to_string(),
        name: "echo".to_string(),
        input: serde_json::json!({}),
    }
}

/// A scripted two-turn run: one tool-using turn, then the final answer.
fn scripted_loop(hub: Arc<EventHub>, responses: Vec<MockResponse>) -> BareLoop<MockApiClient> {
    let client = MockApiClient::new("test-model").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    BareLoop::new(Arc::new(client), registry, SessionConfig::default()).with_observer(hub)
}

/// The event's kind name, for ordered-sequence assertions.
fn kind_of(event: &LoopEvent) -> &'static str {
    match event {
        LoopEvent::RunStart(_) => "run_start",
        LoopEvent::RunEnd(_) => "run_end",
        LoopEvent::TurnStart(_) => "turn_start",
        LoopEvent::TurnEnd(_) => "turn_end",
        LoopEvent::StreamSuccess(_) => "stream_success",
        LoopEvent::StreamFailure(_) => "stream_failure",
        LoopEvent::Response(_) => "response",
        LoopEvent::TextDelta(_) => "text_delta",
        LoopEvent::ThinkingDelta(_) => "thinking_delta",
        LoopEvent::ToolCallReceived(_) => "tool_call_received",
        LoopEvent::ToolPre(_) => "tool_pre",
        LoopEvent::ToolPost(_) => "tool_post",
        LoopEvent::PreCompaction(_) => "pre_compaction",
        LoopEvent::Compaction(_) => "compaction",
        LoopEvent::Fallback(_) => "fallback",
        LoopEvent::TransportFallback(_) => "transport_fallback",
        LoopEvent::ModelSwitched(_) => "model_switched",
        LoopEvent::LoopDetected(_) => "loop_detected",
        LoopEvent::ConvergenceDetected(_) => "convergence_detected",
        _ => "unmapped",
    }
}

/// Drain every buffered event without awaiting.
fn drain(receiver: &mut broadcast::Receiver<ObservedEvent>) -> Vec<ObservedEvent> {
    let mut collected = Vec::new();
    while let Ok(observed) = receiver.try_recv() {
        collected.push(observed);
    }
    collected
}

/// The scripted responses for one tool-using turn plus a final answer.
fn tool_then_finish() -> Vec<MockResponse> {
    vec![turn("working", call("call_a")), vec![terminal()]]
        .into_iter()
        .flatten()
        .collect()
}

#[tokio::test]
async fn event_hub_forwards_every_observer_callback() {
    let hub = Arc::new(EventHub::new(256));
    let mut receiver = hub.subscribe();
    let mut loop_ = scripted_loop(Arc::clone(&hub), tool_then_finish());

    loop_
        .run("fix the bug", &RunConfig::default())
        .await
        .expect("the scripted run completes");

    let events = drain(&mut receiver);
    let kinds: Vec<&str> = events.iter().map(|e| kind_of(&e.event)).collect();
    assert_eq!(
        kinds,
        vec![
            "run_start",
            "turn_start",
            "text_delta",
            "stream_success",
            "response",
            "tool_call_received",
            "tool_pre",
            "tool_post",
            "turn_end",
            "turn_start",
            "text_delta",
            "stream_success",
            "response",
            "turn_end",
            "run_end",
        ],
        "the scripted run must forward exactly the callbacks it fires, in order: {kinds:?}"
    );
    let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    let expected_seqs: Vec<u64> =
        (1..=u64::try_from(events.len()).expect("the event count fits a u64")).collect();
    assert_eq!(
        seqs, expected_seqs,
        "the run's sequence numbers must run from 1 in call order"
    );

    let mut second = scripted_loop(Arc::clone(&hub), vec![terminal()]);
    second
        .run("run again", &RunConfig::default())
        .await
        .expect("the second scripted run completes");
    let second_events = drain(&mut receiver);
    assert_eq!(
        second_events.first().map(|e| (e.seq, kind_of(&e.event))),
        Some((1, "run_start")),
        "a second run through the same hub must restart its sequence at 1"
    );
}

#[tokio::test]
async fn slow_subscriber_is_lagged_not_blocking() {
    let hub = Arc::new(EventHub::new(2));
    let mut receiver = hub.subscribe();
    let mut loop_ = scripted_loop(Arc::clone(&hub), tool_then_finish());

    let result = loop_.run("fix the bug", &RunConfig::default()).await;
    assert!(
        result.is_ok(),
        "a consumer that never drains its ring must never block the engine"
    );

    match receiver.recv().await {
        Err(broadcast::error::RecvError::Lagged(dropped)) => assert!(
            dropped >= 1,
            "the never-draining consumer must be told how many events it missed"
        ),
        other => {
            panic!("the never-draining consumer's first receive must report its lag: {other:?}")
        }
    }
    let remaining = drain(&mut receiver);
    assert_eq!(
        remaining.last().map(|e| kind_of(&e.event)),
        Some("run_end"),
        "the newest events must still arrive after the lag report: {:?}",
        remaining
            .iter()
            .map(|e| kind_of(&e.event))
            .collect::<Vec<_>>()
    );
}
