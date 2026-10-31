//! Pins that the crate's own live emissions validate against its own
//! catalog: every metric event any driven subsystem settles must carry
//! exactly the label keys its [`MetricDescriptor`] declares.
//!
//! This is the emission-side contract the catalog exists to keep — a
//! declared set that the crate's own emissions fail is drift counted
//! forever on every consumer, so the agreement is asserted here, on
//! real emission sites, not on hand-written field lists. Scenarios drive
//! the engine, both memory stores, the fallback chain, both summarizers,
//! the memoizing middleware, the memory-backed demotion sink, and the
//! embedding client; every event the capture sees on the metrics
//! target must pass `registry::validate`, and every signal the suite is
//! expected to fire must appear at least once.
//!
//! Run: `cargo test --all-features --test emission_catalog_contract`

#![cfg(all(feature = "metrics", feature = "testing"))]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::redundant_clone
)]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use loopctl::api::ApiClient;
use loopctl::api::NonStreamingResponse;
use loopctl::api::StreamRequest;
use loopctl::api::error::ApiError;
use loopctl::compact::demote::{DemotionContext, DemotionSink, MemoryDemotionSink};
use loopctl::compact::qa_summarizer::{QaSummarizer, QaSummarizerConfig};
use loopctl::compact::structured_summarizer::{StructuredSummarizer, StructuredSummaryConfig};
use loopctl::compact::{
    CompactReason, CompactionContext, CompactionOutcome, ContextCompactor, FallbackCompactor,
    HeuristicTokenCounter,
};
use loopctl::config::SessionConfig;
use loopctl::engine::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::memory::consolidate::consolidate_entries;
use loopctl::memory::vector::{HashingEmbedder, LinearVectorIndex};
use loopctl::memory::vector_memory::VectorMemoryStore;
use loopctl::memory::{
    ConsolidationConfig, InMemoryStore, LoopMemory, MemoryCategory, MemoryEntry, SearchMemoriesTool,
};
use loopctl::message::Message;
use loopctl::metrics::registry;
use loopctl::middleware::{MemoizingMiddleware, NoopPathExtractor, ToolPipeline};
use loopctl::stream::{StreamEvent, StreamStopReason, Usage};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use serde_json::json;
use uuid::Uuid;

/// One captured metric event: its catalog name and its label keys.
#[derive(Debug, Clone)]
struct Captured {
    /// The `metric` field's value.
    name: String,
    /// The string-typed field keys, the label set `validate` tests.
    labels: Vec<String>,
}

/// A subscriber capturing `(metric, label keys)` per metrics-target event.
///
/// Field classification mirrors `MetricsLayer`: primary numerics and
/// digit-or-decimal renderings are values; every other string field is
/// a label candidate. [`is_candidate_key`] lifts `metric`, `message`,
/// and [`loopctl::metrics::METRIC_EXEMPT_FIELDS`] — the same constant
/// the layer consults, so capture and layer agree by construction,
/// not by copy.
struct ContractCapture {
    events: Mutex<Vec<Captured>>,
}

impl ContractCapture {
    fn events(&self) -> Vec<Captured> {
        self.events.lock().expect("capture lock").clone()
    }
}

/// Whether a field key can name a label at all.
///
/// `metric` names the series and `message` is the emission's prose —
/// neither is ever a label. Beyond those, the contract's exemption
/// list (`model` and the session/request-id fields) is read from
/// [`loopctl::metrics::METRIC_EXEMPT_FIELDS`], the very slice
/// `MetricsLayer::dispatch` consults — one source, no divergence.
fn is_candidate_key(name: &str) -> bool {
    !matches!(name, "metric" | "message") && !loopctl::metrics::METRIC_EXEMPT_FIELDS.contains(&name)
}

/// Whether a `Debug`/`Display` rendering is a number, not a string.
fn rendered_number(rendered: &str) -> bool {
    let body = rendered.strip_prefix(['-', '+']).unwrap_or(rendered);
    let (integral, fraction) = body.split_once('.').unwrap_or((body, ""));
    let integral_ok = !integral.is_empty() && integral.bytes().all(|byte| byte.is_ascii_digit());
    let fraction_ok = fraction.is_empty() || fraction.bytes().all(|byte| byte.is_ascii_digit());
    integral_ok && fraction_ok
}

impl tracing::Subscriber for ContractCapture {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target() == "loopctl::metrics"
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::Id {
        tracing::Id::from_u64(1)
    }
    fn record(&self, _id: &tracing::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _from: &tracing::Id, _to: &tracing::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct FieldVisitor {
            metric: Option<String>,
            labels: Vec<String>,
        }
        impl tracing::field::Visit for FieldVisitor {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                let name = field.name();
                if name == "metric" {
                    self.metric = Some(value.to_string());
                } else if !is_candidate_key(name) {
                    // prose and contract-exempt fields are not labels
                } else {
                    self.labels.push(name.to_string());
                }
            }
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                let name = field.name();
                if name == "metric" {
                    self.metric = Some(format!("{value:?}").trim_matches('"').to_string());
                    return;
                }
                if !is_candidate_key(name) || rendered_number(&format!("{value:?}")) {
                    return;
                }
                self.labels.push(name.to_string());
            }
        }
        let mut visitor = FieldVisitor {
            metric: None,
            labels: Vec::new(),
        };
        event.record(&mut visitor);
        if let Some(name) = visitor.metric {
            self.events.lock().expect("capture lock").push(Captured {
                name,
                labels: visitor.labels,
            });
        }
    }
    fn enter(&self, _id: &tracing::Id) {}
    fn exit(&self, _id: &tracing::Id) {}
}

/// A tool that always succeeds, for `loopctl.tools.calls` legs.
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
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Ok(ToolOutput::text("payload")) })
    }
}

/// A tool that always fails, for the `outcome=error` leg.
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
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Err(ToolError::Execution("always broken".to_string())) })
    }
}

fn turn(id: &str, name: &str) -> MockResponse {
    MockResponse {
        text: "go".to_string(),
        tool_call: Some(MockToolCall {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({"query": "launch code"}),
        }),
        stop_reason: "tool_use".to_string(),
    }
}

fn terminal() -> MockResponse {
    MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

/// Drive the engine with a memory store and both tool outcomes.
async fn engine_scenario() {
    let store = Arc::new(InMemoryStore::new());
    store
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "the launch code is ARC-7",
        ))
        .await
        .unwrap();
    let registry_tools = ToolRegistry::new();
    registry_tools.register(SearchMemoriesTool::new(
        Arc::clone(&store) as Arc<dyn LoopMemory>
    ));
    registry_tools.register(PlainTool);
    registry_tools.register(BrokenTool);
    let mut engine = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(vec![
            turn("s1", "search_memories"),
            turn("p1", "plain"),
            turn("b1", "broken"),
            terminal(),
        ])),
        registry_tools,
        SessionConfig::default(),
    );
    engine.set_memory(Arc::clone(&store) as Arc<dyn LoopMemory>);
    engine
        .run(
            "emission catalog contract: engine legs",
            &RunConfig::default(),
        )
        .await
        .expect("the scripted run completes");
}

/// Drive `consolidate_entries` and the vector store's whole surface.
#[cfg(feature = "vector_memory")]
async fn stores_scenario() {
    let mut entries = vec![
        MemoryEntry::new(MemoryCategory::Fact, "fact one carries value"),
        MemoryEntry::new(MemoryCategory::Fact, "fact one carries value"),
        MemoryEntry::new(MemoryCategory::Insight, "a preference worth keeping"),
    ];
    for entry in &mut entries {
        entry.access_count = 3;
    }
    consolidate_entries(
        &mut entries,
        &ConsolidationConfig::default(),
        std::time::SystemTime::now(),
    );

    let store = Arc::new(VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(64)),
        Box::new(LinearVectorIndex::new(64)),
    ));
    store
        .store(MemoryEntry::new(
            MemoryCategory::Fact,
            "the launch code is ARC-7",
        ))
        .await
        .expect("the vector store accepts the entry");
    let tool = SearchMemoriesTool::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
    tool.call(json!({"query": "launch code"}), &ToolContext::default())
        .await
        .expect("the vector-backed search succeeds");
    store
        .consolidate()
        .await
        .expect("the vector consolidate pass runs");
}

/// An `ApiClient` that refuses every request, so the fallback chain
/// degrades to its terminal stage.
struct FailingClient;

impl ApiClient for FailingClient {
    fn model(&self) -> String {
        "failing".to_string()
    }
    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }
    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        Box::pin(std::future::ready(Err(ApiError::api("unreachable"))))
    }
}

fn conversation() -> Vec<Message> {
    let mut messages = Vec::new();
    for index in 0..6 {
        messages.push(Message::user(format!(
            "user turn {index} asks about topic-{index}"
        )));
        messages.push(Message::assistant(format!(
            "assistant turn {index} decided fact-{index}"
        )));
    }
    messages
}

fn context_for(messages: &[Message]) -> CompactionContext {
    CompactionContext::new(
        CompactionOutcome::estimate_tokens(messages),
        CompactReason::ThresholdExceeded,
        1_000_000,
        3,
        Arc::new(HeuristicTokenCounter),
    )
}

/// Drive the fallback chain through its declining stages.
async fn fallback_scenario() {
    let chain = FallbackCompactor::default_chain(Arc::new(FailingClient));
    let messages = conversation();
    let outcome = chain
        .compact(messages.clone(), 40_000, context_for(&messages))
        .await;
    assert!(outcome.success, "the terminal stage carries the pass");
}

/// An `ApiClient` double serving scripted responses in order.
struct ScriptedClient {
    script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
}

impl ScriptedClient {
    fn new(script: Vec<Result<NonStreamingResponse, ApiError>>) -> Self {
        Self {
            script: Mutex::new(script.into()),
        }
    }
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        "scripted".to_string()
    }
    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }
    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>> {
        let next = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .expect("every scripted call gets a response");
        Box::pin(std::future::ready(next))
    }
}

fn ok(text: &str) -> NonStreamingResponse {
    NonStreamingResponse {
        message: Message::assistant(text.to_string()),
        stop_reason: StreamStopReason::EndTurn,
        usage: None,
    }
}

fn ok_with_usage(text: &str, input: u32, output: u32) -> NonStreamingResponse {
    NonStreamingResponse {
        message: Message::assistant(text.to_string()),
        stop_reason: StreamStopReason::EndTurn,
        usage: Some(Usage::new(input, output)),
    }
}

/// Drive the QA summarizer's reported and estimated pass shapes.
async fn qa_scenario() {
    let reported = QaSummarizer::new(
        Arc::new(ScriptedClient::new(vec![
            Ok(ok_with_usage("S", 100, 10)),
            Ok(ok_with_usage("[]", 200, 20)),
        ])),
        QaSummarizerConfig::default(),
    );
    let messages = conversation();
    let outcome = reported
        .compact(messages.clone(), 40_000, context_for(&messages))
        .await;
    assert!(outcome.success, "the reported pass completes");

    let mixed = QaSummarizer::new(
        Arc::new(ScriptedClient::new(vec![
            Ok(ok_with_usage("M", 40, 4)),
            Ok(ok("[]")),
        ])),
        QaSummarizerConfig::default(),
    );
    let messages = conversation();
    let outcome = mixed
        .compact(messages.clone(), 40_000, context_for(&messages))
        .await;
    assert!(outcome.success, "the mixed pass completes");
}

/// Drive the structured summarizer's section-state pass.
async fn structured_scenario() {
    let summarizer = StructuredSummarizer::new(
        Arc::new(ScriptedClient::new(vec![Ok(ok(
            "### Key facts\n- src/api.rs uses reqwest\n- the window is 200k",
        ))])),
        StructuredSummaryConfig::default(),
    );
    let messages = conversation();
    let outcome = summarizer
        .compact(messages.clone(), 40_000, context_for(&messages))
        .await;
    assert!(outcome.success, "the structured pass completes");
}

/// Drive one embed batch through a local mock endpoint.
#[cfg(feature = "ollama")]
async fn embed_scenario() {
    use loopctl::provider::embeddings::OllamaEmbedder;

    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200)
                .body("{\"embeddings\":[[0.1,0.2,0.3],[0.4,0.5,0.6]]}");
        })
        .await;
    let embedder = OllamaEmbedder::builder()
        .with_model("contract-embed")
        .with_dim(3)
        .with_base_url(server.base_url())
        .build()
        .expect("the loopback endpoint is a valid base URL");
    let batch = embedder
        .embed_texts(&["first text", "second text"])
        .await
        .expect("the mocked batch succeeds");
    assert_eq!(batch.len(), 2, "both embeddings came back");
}

/// Drive the memoizing middleware through a miss and the hit it enables.
///
/// Two turns call the same tool with the same input, so the second is
/// served from the cache: the pair is what makes the hits/misses ratio the
/// two declared rows exist to carry measurable in both directions.
async fn memoize_scenario() {
    let registry_tools = ToolRegistry::new();
    registry_tools.register(PlainTool);
    let mut engine = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(vec![
            turn("m1", "plain"),
            turn("m2", "plain"),
            terminal(),
        ])),
        registry_tools,
        SessionConfig::default(),
    );
    engine
        .set_pipeline(
            ToolPipeline::builder().with_middleware(MemoizingMiddleware::new(
                vec!["plain".to_string()],
                Vec::new(),
                Arc::new(NoopPathExtractor),
                5,
            )),
        )
        .expect("static pipeline composition is valid");
    engine
        .run(
            "emission catalog contract: memoize legs",
            &RunConfig::default(),
        )
        .await
        .expect("the scripted run completes");
}

/// Hand an evicted slice to the memory-backed demotion sink.
async fn demotion_scenario() {
    let store = Arc::new(InMemoryStore::new());
    let sink = MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
    let evicted = vec![
        Message::user("the launch decision was Tuesday"),
        Message::assistant("shipped on Wednesday instead"),
    ];
    sink.demote(
        &evicted,
        DemotionContext {
            reason: CompactReason::ThresholdExceeded,
            turn: 2,
            session_id: Uuid::new_v4(),
        },
    )
    .await
    .expect("the sink accepts the rendered slice");
}

#[tokio::test]
async fn every_live_emission_validates_against_the_catalog() {
    let capture = Arc::new(ContractCapture {
        events: Mutex::new(Vec::new()),
    });
    assert!(
        tracing::subscriber::set_global_default(Arc::clone(&capture)).is_ok(),
        "this binary has exactly one capture test, so it owns the global subscriber"
    );

    engine_scenario().await;
    #[cfg(feature = "vector_memory")]
    stores_scenario().await;
    fallback_scenario().await;
    qa_scenario().await;
    structured_scenario().await;
    memoize_scenario().await;
    demotion_scenario().await;
    #[cfg(feature = "ollama")]
    embed_scenario().await;

    let events = capture.events();
    for event in &events {
        let keys: Vec<&str> = event.labels.iter().map(String::as_str).collect();
        assert!(
            registry::validate(&event.name, &keys).is_some(),
            "the crate's own emission of {} with label set {keys:?} is refused by its own \
             catalog — declared shape and emission shape have forked",
            event.name
        );
    }

    let mut expected: Vec<&str> = vec![
        "loopctl.tools.calls",
        "loopctl.memory.retrieve.results",
        "loopctl.memory.consolidated",
        "loopctl.compaction.passes",
        "loopctl.compaction.degradations",
        "loopctl.compaction.summarizer.tokens",
        "loopctl.compaction.structured.sections",
        "loopctl.memoize.hits",
        "loopctl.memoize.misses",
        "loopctl.demotion.delivered",
        "loopctl.compaction.evicted.messages",
        "loopctl.memory.stored",
    ];
    #[cfg(feature = "vector_memory")]
    expected.extend([
        "loopctl.memory.store.retrieve",
        "loopctl.memory.store.consolidate",
        "loopctl.vector.embed",
        "loopctl.vector.index.search",
    ]);
    #[cfg(feature = "ollama")]
    expected.push("loopctl.embed.batch.duration_ms");
    for expected in expected {
        assert!(
            events.iter().any(|event| event.name == expected),
            "{expected} was expected to fire in these scenarios; a silent site vacuums \
             this contract: {:?}",
            events.iter().map(|event| &event.name).collect::<Vec<_>>()
        );
    }
}
