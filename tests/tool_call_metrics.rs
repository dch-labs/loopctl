//! Pins the engine's per-tool-call counter and the origin-labeled
//! retrieve stream through a process-global tracing capture.
//!
//! One test binary, one owning test — a global subscriber cannot be
//! shared across concurrently running tests. Scenarios are separated
//! by snapshotting the capture length and asserting on deltas, so
//! cross-scenario leakage is impossible to miss.
//!
//! Run: `cargo test --all-features --test tool_call_metrics`

#![cfg(feature = "testing")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::missing_panics_doc,
    clippy::redundant_clone
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use loopctl::config::SessionConfig;
use loopctl::engine::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::memory::SearchMemoriesTool;
use loopctl::memory::{InMemoryStore, LoopMemory, MemoryCategory, MemoryEntry};
use loopctl::middleware::{MemoizingMiddleware, PathExtractor, ToolPipeline};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::health::ToolHealthRegistry;
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use serde_json::{Value, json};

/// A subscriber capturing every `loopctl::` event's fields as joined
/// strings, so the dispatch and retrieve paths' metric events are
/// asserted rather than trusted.
struct MetricCapture {
    /// The joined `field=value` render of every captured event, in order.
    ///
    /// One row per `loopctl::`-targeted event the subscriber accepted;
    /// the test snapshots the length between scenarios and asserts on
    /// the delta, so cross-scenario leakage cannot hide.
    events: Mutex<Vec<String>>,
}

impl MetricCapture {
    /// The captured events, oldest first.
    ///
    /// Cloned out under the capture lock; the single owning test reads
    /// this only after each engine run has settled.
    fn events(&self) -> Vec<String> {
        self.events.lock().expect("capture lock").clone()
    }
}

impl tracing::Subscriber for MetricCapture {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target().starts_with("loopctl::")
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::Id {
        tracing::Id::from_u64(1)
    }
    fn record(&self, _id: &tracing::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct FieldVisitor {
            fields: Vec<String>,
        }
        impl tracing::field::Visit for FieldVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.fields.push(format!("{}={:?}", field.name(), value));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.fields.push(format!("{}={}", field.name(), value));
            }
        }
        let mut visitor = FieldVisitor { fields: Vec::new() };
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture lock")
            .push(visitor.fields.join(" "));
    }
    fn enter(&self, _id: &tracing::Id) {}
    fn exit(&self, _id: &tracing::Id) {}
}

/// A tool that always succeeds with a fixed payload.
///
/// The success-side fixture for the per-call counter: its calls must
/// settle exactly one `outcome=ok` event naming it.
struct PlainTool;

impl Tool for PlainTool {
    fn name(&self) -> &'static str {
        "plain"
    }
    fn description(&self) -> &'static str {
        "Returns a payload"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }
    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Ok(ToolOutput::text("the plain tool payload is VICTOR-3")) })
    }
}

/// A tool that always fails with a typed execution error.
///
/// The failure-side fixture for the per-call counter: its calls must
/// settle exactly one `outcome=error` event, never silence.
struct BrokenTool;

impl Tool for BrokenTool {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn description(&self) -> &'static str {
        "Always fails"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }
    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Err(ToolError::Execution("always broken".to_string())) })
    }
}

/// A path extractor contributing no paths, for the memoize middleware.
///
/// The pipeline scenario only needs the middleware installed to route
/// dispatch through `dispatch_via_pipeline`; with no paths extracted
/// the cache never invalidates, which no scenario here depends on.
struct NoopPathExtractor;

impl PathExtractor for NoopPathExtractor {
    fn paths(&self, _tool_name: &str, _input: &Value) -> Vec<String> {
        Vec::new()
    }
}

/// One scripted turn: a text preamble plus the given tool call.
///
/// The response shape the mock client returns for a turn that asks
/// the engine to dispatch `name` with `input` before continuing.
fn turn(id: &str, name: &str, input: Value) -> MockResponse {
    MockResponse {
        text: "go".to_string(),
        tool_call: Some(MockToolCall {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }),
        stop_reason: "tool_use".to_string(),
    }
}

/// The terminal scripted response that ends a run.
///
/// A turn with no tool call closes the scripted conversation, so the
/// engine settles and the capture can be read.
fn terminal() -> MockResponse {
    MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}
/// Drive the direct-registry dispatch path through one ok search, one
/// ok call, and one failing call, and assert the exact event shapes.
///
/// The scenario pins the per-call counter's existence clauses — one
/// event per settled call, the tool name, the outcome split, the
/// duration field — and the exact per-origin retrieve counts: four
/// passive `trigger=turn` events for four scripted turns, one
/// `trigger=tool` event for the single model-issued search, so a
/// double emission at either site fails the count.
async fn direct_path_scenario(capture: &MetricCapture) {
    let store = Arc::new(InMemoryStore::new());
    store
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "the launch code is ARC-7",
        ))
        .await
        .unwrap();
    let responses = vec![
        turn("s1", "search_memories", json!({"query": "launch code"})),
        turn("p1", "plain", json!({})),
        turn("b1", "broken", json!({})),
        terminal(),
    ];
    let mut registry = ToolRegistry::new();
    registry.register(SearchMemoriesTool::new(
        Arc::clone(&store) as Arc<dyn LoopMemory>
    ));
    registry.register(PlainTool);
    registry.register(BrokenTool);
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(responses)),
        registry,
        SessionConfig::default(),
    );
    loop_.set_memory(Arc::clone(&store) as Arc<dyn LoopMemory>);
    loop_.set_health_registry(Arc::new(ToolHealthRegistry::default()));
    loop_
        .run(
            "one ok search, one ok call, one failing call",
            &RunConfig::default(),
        )
        .await
        .expect("the direct-path run completes");

    let events = capture.events();
    let tool_calls: Vec<String> = events
        .iter()
        .filter(|event| event.contains("metric=loopctl.tools.calls"))
        .cloned()
        .collect();
    assert_eq!(
        tool_calls.len(),
        3,
        "one event per settled call on the direct path: {tool_calls:?}"
    );
    assert!(
        tool_calls
            .iter()
            .any(|event| event.contains("tool=search_memories")
                && event.contains("outcome=ok")
                && event.contains("duration_ms=")),
        "the search call's event names its tool, outcome, and seam-sourced duration"
    );
    assert!(
        tool_calls
            .iter()
            .any(|event| event.contains("tool=plain") && event.contains("outcome=ok")),
        "the plain call's event carries its tool and outcome"
    );
    assert!(
        tool_calls
            .iter()
            .any(|event| event.contains("tool=broken") && event.contains("outcome=error")),
        "the failing call's event carries outcome=error, not silence"
    );
    let retrieves: Vec<String> = events
        .iter()
        .filter(|event| event.contains("metric=loopctl.memory.retrieve.results"))
        .cloned()
        .collect();
    assert!(
        retrieves.iter().any(|event| event.contains("trigger=tool")
            && event.contains("outcome=ok")
            && event.contains("k_returned=1")),
        "the tool-driven retrieve carries its origin and result count: {retrieves:?}"
    );
    let turn_labeled = retrieves
        .iter()
        .filter(|event| event.contains("trigger=turn"))
        .count();
    assert_eq!(
        turn_labeled, 4,
        "the four scripted turns each settle exactly one passive retrieve — a double \
         emission at the passive site would show here: {retrieves:?}"
    );
    let tool_labeled = retrieves
        .iter()
        .filter(|event| event.contains("trigger=tool"))
        .count();
    assert_eq!(
        tool_labeled, 1,
        "the one model-issued search settles exactly one labeled retrieve — a double \
         emission at the tool site would show here: {retrieves:?}"
    );
}

/// Drive one call through the middleware pipeline and assert the
/// shared exit counts it exactly once.
///
/// The pipeline path funnels through the same emission exit as the
/// direct path; a double registration there would settle two events
/// for one call, which this scenario's exact count catches.
async fn pipeline_path_scenario(capture: &MetricCapture) {
    let before = capture.events().len();
    let mut registry = ToolRegistry::new();
    registry.register(PlainTool);
    let responses = vec![turn("q1", "plain", json!({})), terminal()];
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(responses)),
        registry,
        SessionConfig::default(),
    );
    loop_
        .set_pipeline(
            ToolPipeline::builder().with_middleware(MemoizingMiddleware::new(
                Vec::new(),
                Vec::new(),
                Arc::new(NoopPathExtractor),
                5,
            )),
        )
        .expect("static pipeline composition is valid");
    loop_
        .run(
            "one call through the middleware pipeline",
            &RunConfig::default(),
        )
        .await
        .expect("the pipeline-path run completes");

    let delta: Vec<String> = capture.events()[before..].to_vec();
    let pipeline_calls: Vec<String> = delta
        .iter()
        .filter(|event| event.contains("metric=loopctl.tools.calls"))
        .cloned()
        .collect();
    assert_eq!(
        pipeline_calls.len(),
        1,
        "the pipeline path settles exactly one event — the shared exit counts it \
         once, never twice: {pipeline_calls:?}"
    );
    assert!(
        pipeline_calls[0].contains("tool=plain") && pipeline_calls[0].contains("outcome=ok"),
        "the pipeline call's event carries its tool and outcome"
    );
}

/// Assert the labeled and store-internal retrieve streams are
/// distinguishable by metric name over `VectorMemoryStore`.
///
/// One tool-driven retrieve must settle exactly one
/// `loopctl.memory.retrieve.results` event carrying `trigger=tool`
/// and exactly one store-internal `loopctl.memory.store.retrieve`
/// event carrying no origin label — the rename's whole purpose.
#[cfg(all(feature = "testing", feature = "vector_memory"))]
fn store_internal_distinguishability_scenario(capture: &MetricCapture) {
    use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
    use loopctl::memory::vector_memory::VectorMemoryStore;

    let before = capture.events().len();
    let store = Arc::new(VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(64)),
        Box::new(LinearVectorIndex::new(64)),
    ));
    futures::executor::block_on(store.store(MemoryEntry::new(
        MemoryCategory::Fact,
        "the launch code is ARC-7",
    )))
    .expect("the vector store accepts the entry");
    let tool = SearchMemoriesTool::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
    let output = futures::executor::block_on(
        tool.call(json!({"query": "launch code"}), &ToolContext::default()),
    )
    .expect("the search over the vector store succeeds");
    assert!(
        output.text_content().contains("ARC-7"),
        "precondition: the retrieve returned the stored entry"
    );

    let delta: Vec<String> = capture.events()[before..].to_vec();
    let labeled: Vec<String> = delta
        .iter()
        .filter(|event| event.contains("metric=loopctl.memory.retrieve.results"))
        .cloned()
        .collect();
    assert_eq!(
        labeled.len(),
        1,
        "one tool-driven retrieve settles exactly one labeled event"
    );
    assert!(
        labeled[0].contains("trigger=tool"),
        "the labeled event carries the tool origin: {labeled:?}"
    );
    let internal: Vec<String> = delta
        .iter()
        .filter(|event| event.contains("metric=loopctl.memory.store.retrieve"))
        .cloned()
        .collect();
    assert_eq!(
        internal.len(),
        1,
        "the same retrieve settles exactly one store-internal event"
    );
    assert!(
        !internal[0].contains("trigger="),
        "the store-internal event carries no origin label — the two streams are \
         distinguishable by metric name alone: {internal:?}"
    );
}

#[cfg(all(feature = "testing", feature = "builtin_tools"))]
async fn deprecated_think_coexistence_scenario(capture: &MetricCapture) {
    use loopctl::tool::builtin::ThinkTool;

    let before = capture.events().len();
    let responses = vec![
        turn(
            "t1",
            "Think",
            json!({"thought": "consider the launch code"}),
        ),
        terminal(),
    ];
    let mut registry = ToolRegistry::new();
    registry.register(ThinkTool::new());
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(responses)),
        registry,
        SessionConfig::default(),
    );
    loop_
        .run("one think call", &RunConfig::default())
        .await
        .expect("the think run completes");

    let delta: Vec<String> = capture.events()[before..].to_vec();
    let engine_counted: Vec<String> = delta
        .iter()
        .filter(|event| {
            event.contains("metric=loopctl.tools.calls")
                && event.contains("tool=Think")
                && event.contains("outcome=ok")
        })
        .cloned()
        .collect();
    assert_eq!(
        engine_counted.len(),
        1,
        "the engine counter counts the Think call"
    );
    let self_reported: Vec<String> = delta
        .iter()
        .filter(|event| event.contains("metric=loopctl.think.calls"))
        .cloned()
        .collect();
    assert_eq!(
        self_reported.len(),
        1,
        "the deprecated self-report still fires — the v0.3.1 surface stays intact \
         until its 0.Y.0 removal, and the two counters coexist"
    );
}

#[tokio::test]
async fn every_call_and_retrieve_settles_one_labeled_event() {
    let capture = Arc::new(MetricCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );
    direct_path_scenario(&capture).await;
    pipeline_path_scenario(&capture).await;
    #[cfg(all(feature = "testing", feature = "vector_memory"))]
    store_internal_distinguishability_scenario(&capture);
    #[cfg(all(feature = "testing", feature = "builtin_tools"))]
    deprecated_think_coexistence_scenario(&capture).await;
}
