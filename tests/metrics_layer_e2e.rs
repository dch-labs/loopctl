//! End-to-end contract for the [`MetricsLayer`]: a real engine run's
//! own metric emissions, landing in the registry as cataloged series.
//!
//! This suite lives in its own integration binary on purpose. The
//! engine's emission sites are `DEBUG`-level `tracing` callsites, and
//! `tracing` caches per-callsite interest against the process-wide
//! default subscriber. Any test that runs the engine without a metrics
//! subscriber installed would poison those callsites as `never` for
//! the whole process, and the layer scenario here would silently
//! capture nothing. Alone in its process, the engine callsites are
//! first exercised under the layer's subscriber, so interest resolves
//! through the dispatcher every time.

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
use loopctl::metrics::{MetricsLayer, MetricsObserver, MetricsRegistry, SeriesValue};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use tracing_subscriber::layer::SubscriberExt as _;

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

#[tokio::test(flavor = "current_thread")]
async fn the_layer_captures_the_engines_own_tool_emissions() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    let observer = Arc::new(MetricsObserver::default());
    let client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut agent = BareLoop::new(client, registry, SessionConfig::default()).with_observer(Arc::<
        MetricsObserver,
    >::clone(
        &observer,
    ));
    agent
        .run("both collectors", &RunConfig::default())
        .await
        .expect("run");
    drop(guard);

    let snap = store.snapshot();
    let calls = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.tools.calls")
        .expect("dispatch emitted its metric");
    assert_eq!(
        calls.labels.get("tool").map(String::as_str),
        Some("echo"),
        "declared label landed"
    );
    match calls.value {
        SeriesValue::Counter { total } => assert_eq!(total, 1, "one call, counted once"),
        _ => panic!("tools.calls is a counter series"),
    }

    // The reconciled contract: the engine's own emission carries exactly
    // the declared labels, so a clean run counts no label drift at all.
    // The `duration_ms` value stays drift — a counter counts passes, and
    // the numeric on the event is counted, never guessed at.
    assert_eq!(
        snap.unknown_labels, 0,
        "a clean engine run must not put its own crate on the drift counter"
    );
    assert!(
        snap.numeric_on_counter >= 1,
        "duration on a counter is drift"
    );

    // The observer saw the same run through callbacks — the two
    // collectors fill different surfaces from the same activity.
    assert_eq!(
        observer.snapshot().tools.get("echo").map(|s| s.calls),
        Some(1)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn every_gauge_and_histogram_surface_accumulates() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    #[cfg(feature = "vector_memory")]
    {
        use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
        use loopctl::memory::vector_memory::VectorMemoryStore;
        use loopctl::memory::{LoopMemory, MemoryCategory, MemoryEntry, SearchMemoriesTool};

        let vector_store = Arc::new(VectorMemoryStore::new(
            Box::new(HashingEmbedder::new(64)),
            Box::new(LinearVectorIndex::new(64)),
        ));
        vector_store
            .store(MemoryEntry::new(
                MemoryCategory::Fact,
                "the launch code is ARC-7",
            ))
            .await
            .expect("the vector store accepts the entry");
        let tool = SearchMemoriesTool::new(Arc::clone(&vector_store) as Arc<dyn LoopMemory>);
        tool.call(
            serde_json::json!({"query": "launch code"}),
            &loopctl::tool::ToolContext::default(),
        )
        .await
        .expect("the vector-backed search succeeds");
        vector_store
            .consolidate()
            .await
            .expect("the vector consolidate pass runs");
    }

    // The calibration histogram rides usage reports: one terminal turn
    // whose provider reported input tokens.
    let client = Arc::new(
        MockApiClient::new("test-model")
            .with_responses(vec![MockResponse {
                text: "done".to_string(),
                tool_call: None,
                stop_reason: "end_turn".to_string(),
            }])
            .with_usage(loopctl::stream::Usage::new(120, 12)),
    );
    let mut agent = BareLoop::new(client, ToolRegistry::new(), SessionConfig::default());
    agent
        .run("calibrate me", &RunConfig::default())
        .await
        .expect("the usage-carrying run completes");
    drop(guard);

    let snap = store.snapshot();
    let histogram_count = |name: &str| -> Option<u64> {
        snap.series
            .iter()
            .find(|s| s.name == name)
            .map(|s| match &s.value {
                SeriesValue::Histogram { summary } => summary.counts.iter().sum(),
                _ => panic!("{name} is a declared histogram"),
            })
    };

    #[cfg(feature = "vector_memory")]
    for name in [
        "loopctl.vector.embed",
        "loopctl.vector.index.search",
        "loopctl.memory.store.retrieve",
        "loopctl.memory.store.consolidate",
    ] {
        assert!(
            histogram_count(name).unwrap_or(0) >= 1,
            "{name} must accumulate through the layer — an explicit value sample \
             rides every emission site"
        );
    }
    assert!(
        histogram_count("loopctl.estimate.calibration").unwrap_or(0) >= 1,
        "the calibration histogram accumulates from a usage-carrying run"
    );

    assert_eq!(
        snap.ambiguous_scalar, 0,
        "every multi-numeric emission names its sample explicitly"
    );
    assert_eq!(
        snap.unknown_labels, 0,
        "the embed family's contract-exempt model field never reaches the drift counter"
    );
    assert_eq!(
        snap.unknown_metric, 0,
        "every emission spoke a declared name"
    );
}
