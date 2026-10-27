//! Integration surface check for the `metrics` feature: drive a
//! run-shaped sequence through `AgentMetrics`, read percentiles off the
//! snapshot, round-trip it through JSON, and validate emissions against
//! the registry catalog.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc
)]

use loopctl::metrics::{
    AgentMetrics, LatencyHistogram, MetricKind, TokenTotals, all, lookup, validate,
};

fn drive_a_run() -> AgentMetrics {
    let mut m = AgentMetrics {
        session_id: uuid::Uuid::parse_str("6f1e0dd2-2c1b-4c7a-9b41-7f4a2f9d3c11").unwrap(),
        turn_latency: LatencyHistogram::default(),
        ..Default::default()
    };
    for (ms, ok) in [(400_u64, true), (600, true), (9_000, false)] {
        m.total_turns += 1;
        if !ok {
            m.failed_turns += 1;
        }
        m.turn_latency.record(ms);
        m.tokens.add(TokenTotals {
            input: 100,
            output: 25,
            cached_input: 40,
            cache_write: 5,
            reasoning: 8,
        });
    }
    m.record_tool("read", 4, false);
    m.record_tool("exec", 3_000, true);
    m.stream_successes = 2;
    m.stream_failures = 1;
    m.compactions = 1;
    m.run_success = false;
    m.run_error = Some("max_turns".to_string());
    m.run_duration_ms = 12_000;
    m
}

#[test]
fn run_snapshot_reports_percentiles_and_totals() {
    let m = drive_a_run();
    let p = m.turn_latency_percentiles().expect("turns ran");
    // Fixed-bucket upper-bound reads: 600 ms lands in the 1000 bucket,
    // the 9000 ms outlier in the 10000 bucket.
    assert_eq!(p.p50.value_ms, 1_000);
    assert_eq!(p.p90.value_ms, 10_000);
    assert_eq!(p.p99.value_ms, 10_000);
    assert_eq!(m.tokens.total(), 375);
    assert_eq!(m.tokens.reasoning, 24);
    assert_eq!(m.total_tool_calls(), 2);
    assert_eq!(m.total_tool_errors(), 1);
    assert_eq!(m.tools.get("exec").map(|s| s.tool.as_str()), Some("exec"));
}

#[test]
fn snapshot_round_trips_through_json() {
    let m = drive_a_run();
    let json = serde_json::to_string(&m).unwrap();
    let back: AgentMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m);
    assert_eq!(
        back.turn_latency.buckets(),
        LatencyHistogram::new().buckets()
    );
}

#[test]
fn catalog_answers_through_the_public_path() {
    assert!(validate("loopctl.tools.calls", &["tool"]).is_some());
    assert!(validate("loopctl.tools.calls", &["tool", "model"]).is_none());
    let d = lookup("loopctl.trajectory.jsonl_bytes").unwrap();
    assert_eq!(d.kind, MetricKind::Gauge);
    assert!(all().len() >= 60);
    assert!(
        lookup("loopctl.reasoning.blocks").is_some(),
        "reasoning surface is declared"
    );
}

#[test]
fn uuid_missing_from_json_defaults_nil() {
    let m: AgentMetrics = serde_json::from_str("{}").unwrap();
    assert_eq!(m.session_id, uuid::Uuid::nil());
    assert!(!m.run_success);
}
