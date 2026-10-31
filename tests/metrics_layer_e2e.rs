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

use loopctl::compact::CompactReason;
use loopctl::compact::ContextManager;
use loopctl::compact::TruncatingCompactor;
use loopctl::compact::demote::{DemotionContext, DemotionSink, MemoryDemotionSink};
use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::manifest::{EnvMap, ManifestDocument};
use loopctl::memory::{InMemoryStore, LoopMemory};
use loopctl::message::Message;
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
async fn the_demotion_sink_settles_exactly_its_declared_shape() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    let memory = Arc::new(InMemoryStore::new());
    let sink = MemoryDemotionSink::new(Arc::clone(&memory) as Arc<dyn LoopMemory>);
    let evicted = vec![
        Message::user("the plan was to ship on Tuesday"),
        Message::assistant("shipping Wednesday instead"),
    ];
    sink.demote(
        &evicted,
        DemotionContext {
            reason: CompactReason::ThresholdExceeded,
            turn: 3,
            session_id: uuid::Uuid::new_v4(),
        },
    )
    .await
    .expect("the memory-backed sink accepts the delivery");
    drop(guard);

    let snap = store.snapshot();
    let delivered = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.demotion.delivered")
        .expect("a landed delivery counts as delivered");
    assert_eq!(
        delivered.labels.len(),
        1,
        "the row declares only `sink`, so one label may land: {:?}",
        delivered.labels
    );
    assert_eq!(
        delivered.labels.get("sink").map(String::as_str),
        Some("memory"),
        "the declared label lands under its declared name"
    );
    match delivered.value {
        SeriesValue::Counter { total } => assert_eq!(total, 1, "one delivery, counted once"),
        _ => panic!("demotion.delivered is a counter series"),
    }

    // The magnitude the event used to carry is not lost, it simply rides the
    // row that declares it: two messages left the feed, and that is the
    // eviction counter's delta rather than a numeric on the delivery counter.
    let evicted_series = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.compaction.evicted.messages")
        .expect("the eviction is counted where its magnitude is declared");
    match evicted_series.value {
        SeriesValue::Counter { total } => assert_eq!(total, 2, "both messages left the feed"),
        _ => panic!("compaction.evicted.messages is a counter series"),
    }
    assert_eq!(
        evicted_series.labels.len(),
        0,
        "the row declares no labels: {:?}",
        evicted_series.labels
    );

    assert_eq!(
        snap.unknown_labels,
        0,
        "the demotion events must not put the crate on the unknown-label drift counter: {:?}",
        snap.series.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    assert_eq!(
        snap.numeric_on_counter,
        0,
        "a demotion counter carrying a numeric the row does not name is drift: {:?}",
        snap.series.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
}

#[test]
fn headroom_reaches_the_wire_on_the_percent_scale() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    let manager =
        ContextManager::new(Arc::new(TruncatingCompactor::new())).with_context_window(200_000);
    let pre = vec![Message::user("a".repeat(799_980))];
    let post = vec![Message::user("a".repeat(79_980))];
    let telemetry = manager.build_telemetry(
        CompactReason::ThresholdExceeded,
        &pre,
        &post,
        None,
        std::time::Instant::now(),
    );
    drop(guard);

    assert!(
        (telemetry.headroom_pct - 0.9).abs() < f64::EPSILON,
        "the telemetry field keeps reporting the fraction its doc states, got {}",
        telemetry.headroom_pct
    );

    let snap = store.snapshot();
    let gauge = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.compaction.headroom_pct")
        .expect("a pass settles its headroom gauge");
    match gauge.value {
        SeriesValue::Gauge { value } => assert!(
            (value - 90.0).abs() < f64::EPSILON,
            "a name ending `_pct` carries a percent, so ninety percent of the window free \
             must read 90 — an alert written as `headroom_pct < 20` is meaningless against \
             0.9: {value}"
        ),
        _ => panic!("compaction.headroom_pct is a declared gauge series"),
    }
    assert_eq!(
        snap.unknown_labels, 0,
        "the gauge declares no labels, so none may arrive: {:?}",
        gauge.labels
    );
}

#[test]
fn a_profile_reports_only_after_it_has_actually_resolved() {
    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let _guard = tracing::subscriber::set_default(subscriber);

    let sound_yaml = "\
version: 1
budgets:
  turns: 200
metadata:
  name: demo
tools:
  - id: shell
    builtin: shell
";
    let broken_yaml = "\
version: 1
budgets:
  turns: 200
metadata:
  name: ${env:NO_SUCH_VAR_ANYWHERE}
tools:
  - id: shell
    builtin: shell
";

    let broken = ManifestDocument::parse(broken_yaml)
        .expect("the shape parses; it fails on resolution, not syntax");
    assert!(
        broken.resolve(None, &EnvMap::default()).is_err(),
        "a document referencing an unset variable must fail to resolve, or this test \
         pins nothing"
    );
    assert!(
        store
            .snapshot()
            .series
            .iter()
            .all(|s| s.name != "loopctl.profile.active"),
        "a resolution that never completed must not report the profile as active — a \
         dashboard reading that gauge would otherwise claim this process ran a manifest \
         it rejected"
    );

    let sound = ManifestDocument::parse(sound_yaml).expect("the sound fixture parses");
    sound
        .resolve(None, &EnvMap::default())
        .expect("the sound fixture resolves");
    let snap = store.snapshot();
    let active = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.profile.active")
        .expect("a resolved manifest reports its profile");
    assert_eq!(
        active.labels,
        std::collections::BTreeMap::from([("profile", "base".to_string())]),
        "the nameless overlay reports the manifest as authored"
    );
    assert_eq!(snap.unknown_labels, 0, "profile is a declared label");
}

#[tokio::test(flavor = "current_thread")]
async fn a_pinned_clock_freezes_the_memory_retrieve_span() {
    use loopctl::managers::LoopManagers;
    use loopctl::memory::{MemoryCategory, MemoryEntry};
    use loopctl::testing::StepClock;

    let store = Arc::new(MetricsRegistry::default());
    let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(Arc::clone(&store)));
    let guard = tracing::subscriber::set_default(subscriber);

    let memory = Arc::new(InMemoryStore::new());
    memory
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "the launch code is ARC-7",
        ))
        .await
        .expect("the store accepts the entry");

    // Every monotonic read advances a quarter second, so any span measured
    // through the seam is at least that long — while an in-memory retrieve
    // measured on the real clock rounds to zero.
    let managers = LoopManagers::new()
        .with_memory(Arc::clone(&memory) as Arc<dyn LoopMemory>)
        .with_clock(Arc::new(StepClock::new(std::time::Duration::from_millis(
            250,
        ))));

    let client = Arc::new(
        MockApiClient::new("test-model").with_responses(vec![MockResponse {
            text: "done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        }]),
    );
    let mut agent = BareLoop::new_with_managers(
        client,
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    agent
        .run("what is the launch code", &RunConfig::default())
        .await
        .expect("the memory-carrying run completes");
    drop(guard);

    let snap = store.snapshot();
    let series = snap
        .series
        .iter()
        .find(|s| s.name == "loopctl.memory.retrieve.duration_ms")
        .expect("a turn with a memory store reports its retrieve span");
    let (bounds, counts) = match &series.value {
        SeriesValue::Histogram { summary } => (&summary.bounds, &summary.counts),
        _ => panic!("memory.retrieve.duration_ms is a declared histogram"),
    };
    assert_eq!(counts.iter().sum::<u64>(), 1, "one retrieve, one sample");
    let filled = counts
        .iter()
        .position(|c| *c > 0)
        .expect("the sample landed in a bucket");
    let landed_at = bounds
        .get(filled)
        .copied()
        .expect("a quarter second sits inside the bounded range, not the overflow slot");
    assert!(
        landed_at >= 250,
        "the retrieve span must be measured through the pinned clock, whose quarter-second \
         step cannot land in a bucket below 250 ms — the engine read the real clock instead \
         (bucket bound {landed_at}), so a frozen run would not reproduce this duration"
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
