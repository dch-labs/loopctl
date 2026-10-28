//! End-to-end contracts for the [`MetricsObserver`]: the session
//! snapshot filled through real engine runs.
//!
//! The scenarios drive [`BareLoop`] over scripted [`MockApiClient`]
//! responses and read the snapshot at the run boundary. The companion
//! [`MetricsLayer`] end-to-end suite lives in its own integration
//! binary (`tests/metrics_layer_e2e.rs`) because engine emissions
//! interact with `tracing`'s process-wide callsite-interest cache.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::MetricsObserver;
use loopctl::observer::{LoopObserver, RunStartContext};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

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

/// One scripted tool-call turn plus the terminal reply it enables.
fn tool_turn_then_terminal() -> Vec<MockResponse> {
    vec![
        MockResponse {
            text: "calling echo".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "finished".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]
}

fn terminal_only() -> Vec<MockResponse> {
    vec![MockResponse {
        text: "finished".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }]
}

/// A loop wired to the mock client, a registry holding `echo`, and `observer`.
fn scripted_loop(
    observer: Arc<MetricsObserver>,
    responses: Vec<MockResponse>,
) -> BareLoop<MockApiClient> {
    let client = Arc::new(MockApiClient::new("test-model").with_responses(responses));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    BareLoop::new(client, registry, SessionConfig::default()).with_observer(observer)
}

async fn run_once(responses: Vec<MockResponse>) -> Arc<MetricsObserver> {
    let observer = Arc::new(MetricsObserver::default());
    let mut agent = scripted_loop(Arc::clone(&observer), responses);
    agent
        .run("collect me", &RunConfig::default())
        .await
        .expect("scripted run completes");
    observer
}

#[tokio::test]
async fn an_end_to_end_run_fills_the_session_snapshot() {
    let observer = run_once(tool_turn_then_terminal()).await;
    let m = observer.snapshot();

    assert_ne!(m.session_id, uuid::Uuid::default(), "identity landed");
    assert!(m.run_success);
    assert_eq!(m.run_error, None);
    assert_eq!(m.total_turns, 2, "tool turn plus terminal turn");
    assert_eq!(m.failed_turns, 0);
    assert_eq!(m.turn_latency.count(), 2, "both turns sampled latency");
    assert!(m.turn_latency_percentiles().is_some());

    let echo = m.tools.get("echo").expect("echo recorded");
    assert_eq!(echo.calls, 1);
    assert_eq!(echo.errors, 0);
}

#[tokio::test]
async fn a_failed_run_lands_the_failure_outcome() {
    let observer = Arc::new(MetricsObserver::default());
    let client = Arc::new(MockApiClient::new("test-model").with_error("rate limit exceeded"));
    let mut agent = BareLoop::new(client, ToolRegistry::new(), SessionConfig::default())
        .with_observer(Arc::<MetricsObserver>::clone(&observer));
    let outcome = agent.run("fail me", &RunConfig::default()).await;
    assert!(outcome.is_err(), "the forced API error surfaces");

    let m = observer.snapshot();
    assert!(!m.run_success);
    let error = m.run_error.expect("the failure reason is kept");
    assert!(
        error.contains("rate limit"),
        "carries the provider's word: {error}"
    );
}

#[tokio::test]
async fn reset_hands_the_next_run_a_clean_ledger() {
    let observer = Arc::new(MetricsObserver::default());

    let mut first = scripted_loop(Arc::clone(&observer), terminal_only());
    first
        .run("first", &RunConfig::default())
        .await
        .expect("first");
    assert_eq!(observer.snapshot().total_turns, 1);

    // The session boundary is where a sustained observer is reset;
    // the second run must not inherit the first run's turns.
    observer.reset();
    let mut second = scripted_loop(Arc::clone(&observer), tool_turn_then_terminal());
    second
        .run("second", &RunConfig::default())
        .await
        .expect("second");

    let m = observer.snapshot();
    assert_eq!(m.total_turns, 2, "the second run's own turns, only");
    assert_eq!(m.tools.get("echo").map(|s| s.calls), Some(1));
}

#[tokio::test]
async fn an_observer_shared_across_sessions_reports_each_run_alone() {
    let observer = Arc::new(MetricsObserver::default());
    observer.on_run_start(&RunStartContext {
        session_id: uuid::Uuid::new_v4(),
    });

    let mut first = scripted_loop(Arc::clone(&observer), terminal_only());
    first.run("one", &RunConfig::default()).await.expect("one");

    // No reset and no `reset_managers` between runs: the observer still
    // opens a fresh ledger at each run start, so the boundary read is
    // the run that just ended — never that run plus the runs before it.
    let mut second = scripted_loop(Arc::clone(&observer), tool_turn_then_terminal());
    second.run("two", &RunConfig::default()).await.expect("two");

    let m = observer.snapshot();
    assert_eq!(
        m.turn_latency.count(),
        2,
        "the second run's own turns, sampled fresh at its start"
    );
    assert_eq!(
        m.total_turns, 2,
        "the boundary read carries the second run's turns only — a reused \
         observer is per-run without anyone resetting it"
    );
    assert_eq!(
        m.tools.get("echo").map(|s| s.calls),
        Some(1),
        "the first run's tools stay out of the second run's record"
    );
    assert!(m.run_success, "the last run's outcome is the boundary read");
}
