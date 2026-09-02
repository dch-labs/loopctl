//! Trajectory-capture contracts for the engine-to-observer wiring.
//!
//! Pins that a real scripted run assembles into one record with the
//! captured turns, queries (including continuation-turn tool-result text),
//! tool calls, and the three-way outcome classification — plus the JSONL
//! sink's one-line-per-run contract and its best-effort survival behavior.
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

use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::memory::trajectory::{TrajectoryObserver, TrajectoryOutcome, TrajectoryRecord};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use std::future::Future;
use std::pin::Pin;

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
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }
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

fn terminal() -> MockResponse {
    MockResponse {
        text: "finished".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

fn scripted_loop(
    observer: Arc<TrajectoryObserver>,
    responses: Vec<MockResponse>,
) -> BareLoop<MockApiClient> {
    let client = MockApiClient::new("test-model").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(EchoTool);
    BareLoop::new(Arc::new(client), registry, SessionConfig::default()).with_observer(observer)
}

fn call(id: &str) -> MockToolCall {
    MockToolCall {
        id: id.to_string(),
        name: "echo".to_string(),
        input: serde_json::json!({}),
    }
}

#[tokio::test]
async fn happy_run_captures_every_turn_and_call() {
    let observer = Arc::new(TrajectoryObserver::in_memory());
    let responses = vec![
        turn("thinking", call("call_a")),
        turn("using the result", call("call_b")),
        vec![terminal()],
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut loop_ = scripted_loop(Arc::clone(&observer), responses);

    loop_
        .run("fix the bug", &RunConfig::default())
        .await
        .expect("run completes");

    let records = observer.records();
    assert_eq!(records.len(), 1, "one run produces exactly one record");
    let record = &records[0];
    assert_eq!(record.outcome, TrajectoryOutcome::Success);
    assert_eq!(record.total_turns, 3);
    assert_eq!(record.turns.len(), 3, "every turn is captured");
    assert_eq!(record.turns[0].query, "fix the bug");
    assert_eq!(
        record.turns[1].query, "done",
        "a continuation turn's query carries the tool-result text"
    );
    assert_eq!(record.turns[0].response_text, "thinking");
    let ids: Vec<&str> = record.turns[0]
        .tool_calls
        .iter()
        .map(|c| c.tool_call_id.as_str())
        .collect();
    assert_eq!(ids, vec!["call_a"], "one TrajectoryToolCall per dispatch");
    assert!(
        record.turns[1].tool_calls[0].ok,
        "the successful call is ok"
    );
    let input_sum: u64 = record.turns.iter().map(|t| t.input_tokens).sum();
    let output_sum: u64 = record.turns.iter().map(|t| t.output_tokens).sum();
    assert_eq!(
        record.token_summary.input_tokens, input_sum,
        "the summary aggregates every captured turn's input"
    );
    assert_eq!(
        record.token_summary.output_tokens, output_sum,
        "the summary aggregates every captured turn's output"
    );
}

#[tokio::test]
async fn partial_classification() {
    let observer = Arc::new(TrajectoryObserver::in_memory());
    let responses = vec![turn("working", call("call_a"))]
        .into_iter()
        .flatten()
        .collect();
    let mut loop_ = scripted_loop(Arc::clone(&observer), responses);
    let config = RunConfig::default().with_max_turns(1);

    let outcome = loop_.run("fix the bug", &config).await;

    assert!(outcome.is_err(), "the run trips max-turns");
    let record = &observer.records()[0];
    assert_eq!(
        record.outcome,
        TrajectoryOutcome::Partial,
        "a failed run with successful tool work is Partial, not Failure"
    );
}

#[tokio::test]
async fn failure_and_success_classifications() {
    let observer = Arc::new(TrajectoryObserver::in_memory());
    // A call to an unregistered tool fails without producing successful
    // work; the engine keeps turning until max-turns trips.
    let missing = MockToolCall {
        id: "call_missing".to_string(),
        name: "nope".to_string(),
        input: serde_json::json!({}),
    };
    let responses = vec![turn("trying", missing.clone()), turn("retrying", missing)]
        .into_iter()
        .flatten()
        .collect();
    let mut loop_ = scripted_loop(Arc::clone(&observer), responses);
    let config = RunConfig::default().with_max_turns(2);

    let outcome = loop_.run("impossible ask", &config).await;

    assert!(outcome.is_err(), "the run trips max-turns");
    let records = observer.records();
    assert_eq!(
        records[0].outcome,
        TrajectoryOutcome::Failure,
        "a failed run whose tool calls all failed is Failure, not Partial"
    );

    let success_observer = Arc::new(TrajectoryObserver::in_memory());
    let mut loop_ = scripted_loop(
        Arc::clone(&success_observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );
    loop_
        .run("plain question", &RunConfig::default())
        .await
        .expect("run completes");
    assert_eq!(
        success_observer.records()[0].outcome,
        TrajectoryOutcome::Success,
        "a clean run is Success"
    );
}

#[tokio::test]
async fn jsonl_sink_writes_one_line_per_run() {
    let dir = std::env::temp_dir().join(format!("trajectory-test-{}", uuid::Uuid::new_v4()));
    let observer = Arc::new(TrajectoryObserver::writing_to(&dir));
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    loop_
        .run("first", &RunConfig::default())
        .await
        .expect("first run completes");
    loop_
        .run("second", &RunConfig::default())
        .await
        .expect("second run completes");

    let raw = std::fs::read_to_string(dir.join("trajectory.jsonl"))
        .expect("the ledger exists after two runs");
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 2, "one JSONL line per run");
    let parsed: Vec<TrajectoryRecord> = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("each line deserializes into a record"))
        .collect();
    assert_ne!(
        parsed[0].run_id, parsed[1].run_id,
        "two runs never share a run_id"
    );
    assert_eq!(parsed[0].session_id, parsed[1].session_id);
    std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
}

#[tokio::test]
async fn write_failure_warns_and_survives() {
    // A regular file where the ledger directory should be: every
    // `create_dir_all` fails, so the sink is permanently broken.
    let blocker = std::env::temp_dir().join(format!("trajectory-block-{}", uuid::Uuid::new_v4()));
    std::fs::write(&blocker, b"not a directory").expect("blocker file written");
    let observer = Arc::new(TrajectoryObserver::writing_to(&blocker));
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    loop_
        .run("first", &RunConfig::default())
        .await
        .expect("a broken sink must not fail the run");
    loop_
        .run("second", &RunConfig::default())
        .await
        .expect("the observer still records the next run");

    assert_eq!(
        observer.records().len(),
        2,
        "both records survive in memory when the file output is dropped"
    );
    std::fs::remove_file(&blocker).expect("cleanup succeeds");
}

#[tokio::test]
async fn records_in_memory_form_feeds_the_host() {
    let observer = Arc::new(TrajectoryObserver::in_memory());
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    assert!(
        observer.records().is_empty(),
        "no records before any run ends"
    );
    loop_
        .run("only", &RunConfig::default())
        .await
        .expect("run completes");
    loop_
        .run("again", &RunConfig::default())
        .await
        .expect("second run completes");

    let records = observer.records();
    assert_eq!(records.len(), 2, "in-memory form hands over every run");
    assert_ne!(records[0].run_id, records[1].run_id);
    assert_eq!(records[0].session_id, records[1].session_id);
    assert!(!records[1].started_at.is_empty());
}

#[tokio::test]
async fn jsonl_lines_stay_line_delimited_under_hostile_text() {
    let dir = std::env::temp_dir().join(format!("trajectory-hostile-{}", uuid::Uuid::new_v4()));
    let observer = Arc::new(TrajectoryObserver::writing_to(&dir));
    let hostile = MockResponse {
        text: "line one\nline two \"quoted\" {brace}".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    };
    let mut loop_ = scripted_loop(Arc::clone(&observer), vec![hostile]);

    loop_
        .run("hostile text", &RunConfig::default())
        .await
        .expect("run completes");

    let raw = std::fs::read_to_string(dir.join("trajectory.jsonl")).expect("the ledger exists");
    assert_eq!(
        raw.lines().count(),
        1,
        "newlines inside captured text must be escaped, never emitted raw — \
         one record is one line"
    );
    let record: TrajectoryRecord =
        serde_json::from_str(&raw).expect("the single line deserializes into the record");
    assert!(
        record.turns[0].response_text.contains('\n'),
        "the newline survives inside the escaped string"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
}

#[tokio::test]
async fn memory_retention_drops_the_oldest_records() {
    let observer = Arc::new(TrajectoryObserver::in_memory().with_memory_retention(Some(1)));
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    loop_
        .run("first", &RunConfig::default())
        .await
        .expect("first run completes");
    loop_
        .run("second", &RunConfig::default())
        .await
        .expect("second run completes");

    let records = observer.records();
    assert_eq!(
        records.len(),
        1,
        "retention cap bounds how many records memory holds"
    );
    assert_eq!(
        records[0].turns[0].query, "second",
        "the oldest record is dropped, the newest retained"
    );

    // Disk capture is unaffected by the memory cap: the ledger still holds
    // both lines when a sink is configured alongside the cap.
    let dir = std::env::temp_dir().join(format!("trajectory-retention-{}", uuid::Uuid::new_v4()));
    let observer = Arc::new(TrajectoryObserver::writing_to(&dir).with_memory_retention(Some(1)));
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );
    loop_
        .run("first", &RunConfig::default())
        .await
        .expect("first run completes");
    loop_
        .run("second", &RunConfig::default())
        .await
        .expect("second run completes");

    let raw = std::fs::read_to_string(dir.join("trajectory.jsonl")).expect("the ledger exists");
    assert_eq!(
        raw.lines().count(),
        2,
        "the memory cap must not drop ledger lines"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
}

#[tokio::test]
async fn concurrent_observers_sharing_a_ledger_keep_every_line_whole() {
    let dir = std::env::temp_dir().join(format!("trajectory-concurrent-{}", uuid::Uuid::new_v4()));
    let writers: Vec<std::sync::Arc<TrajectoryObserver>> = (0..4)
        .map(|_| std::sync::Arc::new(TrajectoryObserver::writing_to(&dir)))
        .collect();
    let mut loops: Vec<_> = writers
        .iter()
        .map(|observer| {
            scripted_loop(
                std::sync::Arc::clone(observer),
                vec![vec![terminal()]].into_iter().flatten().collect(),
            )
        })
        .collect();

    let mut joins = Vec::new();
    for mut loop_ in loops.drain(..) {
        joins.push(tokio::spawn(async move {
            for _ in 0..5 {
                loop_
                    .run("concurrent", &RunConfig::default())
                    .await
                    .expect("run completes");
            }
        }));
    }
    for join in joins {
        join.await.expect("writer task completes");
    }

    let raw = std::fs::read_to_string(dir.join("trajectory.jsonl")).expect("the ledger exists");
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 20, "four writers x five runs, one line each");
    for line in &lines {
        let parsed: TrajectoryRecord =
            serde_json::from_str(line).expect("every interleaved line stays a whole record");
        assert!(!parsed.run_id.is_empty());
    }
    std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
}

#[tokio::test]
async fn a_zero_retention_cap_retains_nothing() {
    let observer = Arc::new(TrajectoryObserver::in_memory().with_memory_retention(Some(0)));
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    loop_
        .run("gone immediately", &RunConfig::default())
        .await
        .expect("run completes");

    assert!(
        observer.records().is_empty(),
        "a zero retention cap retains nothing, while the run itself is unaffected"
    );
}

#[tokio::test]
async fn the_degenerate_config_combination_stays_consistent() {
    let dir = std::env::temp_dir().join(format!("trajectory-degenerate-{}", uuid::Uuid::new_v4()));
    let observer = Arc::new(
        TrajectoryObserver::writing_to(&dir)
            .with_capture_limit(0)
            .with_memory_retention(Some(0)),
    );
    let mut loop_ = scripted_loop(
        Arc::clone(&observer),
        vec![vec![terminal()]].into_iter().flatten().collect(),
    );

    loop_
        .run("everything minimized", &RunConfig::default())
        .await
        .expect("run completes");

    assert!(
        observer.records().is_empty(),
        "zero retention retains nothing"
    );
    let raw = std::fs::read_to_string(dir.join("trajectory.jsonl")).expect("ledger exists");
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 1, "the ledger keeps the record memory dropped");
    let record: TrajectoryRecord = serde_json::from_str(lines[0]).expect("the line parses");
    assert_eq!(
        record.turns[0].response_text, "",
        "zero capture limit captures no response text"
    );
    assert_eq!(record.outcome, TrajectoryOutcome::Success);
    std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
}
