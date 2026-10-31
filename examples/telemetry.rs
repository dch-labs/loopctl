#![allow(clippy::all, clippy::pedantic, clippy::restriction)]
//! Telemetry — one loop run, counted twice.
//!
//! A run's numbers reach a host through two independent collectors, and this
//! example drives **both over the same run** so the difference between them is
//! visible rather than asserted:
//!
//! - [`MetricsLayer`] rides the `tracing` subscriber and turns the engine's own
//!   `DEBUG`-level metric events into cataloged series inside a
//!   [`MetricsRegistry`]. It reports what the crate *emits* — each name in the
//!   first section below is a row of the published catalog in
//!   `src/metrics/registry.rs`.
//! - [`MetricsObserver`] is a [`LoopObserver`](loopctl::observer::LoopObserver)
//!   that accumulates counters from lifecycle callbacks: turns, tokens,
//!   per-tool stats. It reports what this engine *did*, scoped to one run.
//!
//! ```sh
//! cargo run --example telemetry --features metrics,testing
//! ```
//!
//! Nothing reaches the network: the provider is [`MockApiClient`] replaying a
//! scripted tool-call turn, so every counted value printed is deterministic —
//! counters, token totals, and histogram sample counts are fixed by the script.
//! The one figure that is not is the run span's wall-clock, which moves run to
//! run. That also makes the example a smoke test for the catalog. A declared
//! signal that stops being emitted simply stops appearing in the table, and the
//! drift line underneath it names a shape mismatch — a label an event carries
//! that its row does not declare, or a numeric that lands on a counter which
//! never named one.
//!
//! The runtime detail that matters: `tracing` caches per-callsite interest
//! against the default subscriber, so the layer must be installed before the
//! engine's first emission and the run must execute on the thread that
//! installed it. Hence `current_thread` and
//! [`tracing::subscriber::set_default`], not `set_global_default`.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::RunConfig;
use loopctl::engine::core::Loop;
use loopctl::metrics::{
    AgentMetrics, MetricsLayer, MetricsObserver, MetricsRegistry, RegistrySnapshot, SeriesSnapshot,
    SeriesValue,
};
use loopctl::observer::LoopObserver;
use loopctl::stream::Usage;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use tracing_subscriber::layer::SubscriberExt as _;

/// A tool that answers with a fixed string, so dispatch runs for real.
///
/// Written against the [`Tool`] trait rather than the
/// [`FnTool`](loopctl::tool::FnTool) shorthand because the interesting part of
/// this example is what a dispatched call emits; the body is incidental, and a
/// fixed output keeps the printed numbers stable.
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

/// The scripted provider traffic: one tool call, then the terminal reply.
///
/// This pair is the minimum activity that touches both collectors. The tool
/// call makes the engine dispatch, which is where `loopctl.tools.calls` is
/// emitted; the terminal turn is what settles tokens and the run counters the
/// observer reads from callbacks.
fn scripted_turns() -> Vec<MockResponse> {
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

/// One series value as the single figure a dashboard reads off its kind.
///
/// A counter prints its total, a gauge its last observed value, and a
/// histogram the number of samples it holds — the summary's bucket bounds are
/// a store detail this reader does not need.
fn render_value(value: &SeriesValue) -> String {
    match value {
        SeriesValue::Counter { total } => total.to_string(),
        SeriesValue::Gauge { value } => format!("{value}"),
        SeriesValue::Histogram { summary } => format!("n={}", summary.counts.iter().sum::<u64>()),
    }
}

/// A series' labels as `key="value"` pairs, in the store's sorted order.
///
/// Empty is the common case rather than an error: most catalog rows declare no
/// labels at all, and an undeclared one would have been counted as drift
/// instead of reaching here.
fn render_labels(labels: &BTreeMap<&'static str, String>) -> String {
    if labels.is_empty() {
        return String::new();
    }
    labels
        .iter()
        .map(|(key, value)| format!("{key}={value:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Print every series the engine's own events put into the registry.
///
/// Sorted by name so the output is stable enough to diff between runs: a signal
/// that disappears is then a missing line rather than a reshuffled table.
///
/// The counters underneath are the catalog's contract made visible. A label an
/// event carries that its row never declared, or a metric name that is not in
/// the table, would surface there instead of being quietly stored under an
/// invented shape. The one figure a clean run does report is the numeric on a
/// counter: `loopctl.tools.calls` settles a call *and* reports its span, and a
/// counter counts passes — the `duration_ms` rides as drift rather than being
/// guessed at by the store, which is what `tests/metrics_layer_e2e.rs` pins.
fn print_catalog(snap: &RegistrySnapshot) {
    let mut rows: Vec<&SeriesSnapshot> = snap.series.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));

    println!("what the engine emitted, named by the catalog:");
    for row in &rows {
        println!(
            "  {:<46}{:>9}  {}",
            row.name,
            render_value(&row.value),
            render_labels(&row.labels)
        );
    }

    println!();
    println!(
        "shape drift over this run: {} undeclared label(s), {} numeric(s) on a counter, \
         {} ambiguous scalar(s), {} unknown metric name(s)",
        snap.unknown_labels, snap.numeric_on_counter, snap.ambiguous_scalar, snap.unknown_metric
    );
    println!(
        "  of these, a numeric on a counter is expected from a tool call (its span rides \
         the same event); the other three stay zero in a healthy run"
    );
}

/// Print what the observer collected from the same run through callbacks.
///
/// Per-tool rows come from a `BTreeMap`, so they print in name order without
/// any sorting here — the same determinism the export diffing relies on.
fn print_run(metrics: &AgentMetrics) {
    println!();
    println!("what the observer counted from the same run:");
    println!("  succeeded    {}", metrics.run_success);
    println!(
        "  turns        {} ({} failed)",
        metrics.total_turns, metrics.failed_turns
    );
    println!(
        "  tokens       {} in / {} out",
        metrics.tokens.input, metrics.tokens.output
    );
    println!("  run span     {} ms", metrics.run_duration_ms);
    for stats in metrics.tools.values() {
        println!(
            "  tool {:<10} {} call(s), {} error(s)",
            stats.tool, stats.calls, stats.errors
        );
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    let observer = Arc::new(MetricsObserver::default());
    let attached: Arc<dyn LoopObserver> = observer.clone();
    let client = Arc::new(
        MockApiClient::new("mock-model")
            .with_responses(scripted_turns())
            .with_usage(Usage::new(120, 12)),
    );
    let tools = ToolRegistry::new();
    tools.register(EchoTool);
    let mut agent = BareLoop::new(client, tools, SessionConfig::default()).with_observer(attached);

    let ran = agent
        .run("echo the greeting, then stop", &RunConfig::default())
        .await;

    drop(guard);

    if let Err(err) = &ran {
        println!("the scripted run stopped early: {err}");
    }

    print_catalog(&store.snapshot());
    print_run(&observer.snapshot());
}
