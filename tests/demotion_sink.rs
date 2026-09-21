//! Demotion-sink contracts for engine-driven compaction.
//!
//! Pins: content a compaction pass removes is recallable from memory
//! afterwards (the eviction-becomes-handoff headline); a sink that
//! rejects its delivery never fails the compaction that already
//! succeeded — the observer still reports the pass and the run
//! completes; the no-sink default is behavior-identical to the
//! explicit no-op sink, request bytes equal on the mock client; a
//! cancel that fires while a sink write hangs ends the run with the
//! typed cancellation and delivers exactly one hanging write; and the
//! `with_demotion_sink` builder routes deliveries carrying the pass's
//! reason, turn, and session id.
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
use std::sync::{Arc, Mutex};

use loopctl::compact::demote::{DemotionContext, DemotionSink, MemoryDemotionSink};
use loopctl::compact::{CompactReason, ContextManager, TruncatingCompactor};
use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::error::LoopError;
use loopctl::memory::{InMemoryStore, LoopMemory};
use loopctl::message::Message;
use loopctl::observer::{CompactedContext, LoopObserver};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// A tool whose result carries a unique fact, so a later retrieval can
/// prove the evicted tool-result region survived demotion.
struct FactTool;

impl Tool for FactTool {
    fn name(&self) -> &'static str {
        "recall"
    }
    fn description(&self) -> &'static str {
        "Returns a fact"
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
        Box::pin(async {
            Ok(ToolOutput::text(
                "the launch code is ARC-7 ".to_string() + &"supporting detail ".repeat(20),
            ))
        })
    }
}

/// A tool whose result is one large fixed payload — the growth knob for
/// wide-window scenarios.
struct BigResultTool;

impl Tool for BigResultTool {
    fn name(&self) -> &'static str {
        "big"
    }
    fn description(&self) -> &'static str {
        "Returns a large payload"
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
        Box::pin(async { Ok(ToolOutput::text("y".repeat(5_300))) })
    }
}

/// A tool whose result size the scripted input controls.
struct SizedResultTool;

impl Tool for SizedResultTool {
    fn name(&self) -> &'static str {
        "sized"
    }
    fn description(&self) -> &'static str {
        "Returns a payload of the requested size"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name().to_string(),
            self.description().to_string(),
            serde_json::json!({
                "type": "object",
                "properties": { "chars": { "type": "integer" } },
                "required": ["chars"]
            }),
        )
    }
    fn call(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let chars = input
            .get("chars")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        Box::pin(async move { Ok(ToolOutput::text("z".repeat(chars))) })
    }
}

/// One delivery as the recording sink saw it: the evicted texts and
/// the pass metadata.
type Delivery = (Vec<String>, DemotionContext);

/// Records every delivery instead of storing anything.
struct RecordingSink {
    deliveries: Arc<Mutex<Vec<Delivery>>>,
}

impl DemotionSink for RecordingSink {
    fn demote<'a>(
        &'a self,
        evicted: &'a [Message],
        meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        let deliveries = Arc::clone(&self.deliveries);
        let texts = evicted
            .iter()
            .map(Message::text_content)
            .collect::<Vec<_>>();
        Box::pin(async move {
            deliveries
                .lock()
                .map_err(|_| LoopError::LockPoisoned {
                    what: "recording sink".to_string(),
                })?
                .push((texts, meta));
            Ok(())
        })
    }
}

/// Rejects every delivery and counts them.
struct FailingSink {
    calls: Arc<Mutex<usize>>,
}

impl DemotionSink for FailingSink {
    fn demote<'a>(
        &'a self,
        _evicted: &'a [Message],
        _meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        let calls = Arc::clone(&self.calls);
        Box::pin(async move {
            *calls.lock().expect("calls lock") += 1;
            Err(LoopError::ToolExecution {
                tool: "demotion".to_string(),
                message: "store rejected the delivery".to_string(),
            })
        })
    }
}

/// Counts entries and never returns — the in-flight sink write the
/// cancellation has to beat.
struct HangingSink {
    entered: Arc<Mutex<usize>>,
}

impl DemotionSink for HangingSink {
    fn demote<'a>(
        &'a self,
        _evicted: &'a [Message],
        _meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        let entered = Arc::clone(&self.entered);
        Box::pin(async move {
            *entered.lock().expect("entered lock") += 1;
            std::future::pending().await
        })
    }
}

/// Counts `on_compaction` firings — the pass-completed evidence.
struct CountingObserver {
    compactions: Arc<Mutex<usize>>,
}

impl LoopObserver for CountingObserver {
    fn name(&self) -> &str {
        "counting"
    }

    fn on_compaction(&self, _ctx: &CompactedContext) {
        *self.compactions.lock().expect("compaction lock") += 1;
    }
}

/// The response script that grows the history until the tiny window
/// forces compaction, then ends the run.
fn growing_script() -> Vec<MockResponse> {
    let mut responses = Vec::new();
    for i in 0..10 {
        responses.push(MockResponse {
            text: "go".to_string(),
            tool_call: Some(MockToolCall {
                id: format!("c{i}"),
                name: "recall".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        });
    }
    responses.push(MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    });
    responses
}

/// A loop over the growing script with a truncating context manager
/// behind a tiny window — the compaction-forcing shape every test here
/// starts from. The compactor's knobs sit below the conversation size
/// the tiny window compacts at, so every machine-requested pass
/// removes messages (a pass that shaves nothing terminates the run by
/// the machine's no-progress guard).
fn compacting_loop(
    client: MockApiClient,
    sink: Option<Arc<dyn DemotionSink>>,
    observer: Option<Arc<dyn LoopObserver>>,
) -> BareLoop<MockApiClient> {
    let mut registry = ToolRegistry::new();
    registry.register(FactTool);
    let config = SessionConfig::default()
        .with_context_window(400)
        .with_compact_threshold(50);
    let mut loop_ = BareLoop::new(Arc::new(client), registry, config);
    loop_.set_context_manager(Arc::new(
        ContextManager::new(Arc::new(
            TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        ))
        .with_context_window(400),
    ));
    if let Some(sink) = sink {
        loop_.set_demotion_sink(sink);
    }
    if let Some(observer) = observer {
        loop_.register_observer(observer);
    }
    loop_
}

/// Serialize the captured requests for equality comparison — the Debug
/// render covers every field deterministically within one build.
fn request_signatures(client: &MockApiClient) -> Vec<String> {
    client
        .captured_requests()
        .into_iter()
        .map(|request| format!("{request:?}"))
        .collect()
}

#[tokio::test]
async fn demoted_content_is_retrievable() {
    let store = Arc::new(InMemoryStore::new());
    let mut loop_ = compacting_loop(
        MockApiClient::new("m").with_responses(growing_script()),
        Some(Arc::new(MemoryDemotionSink::new(
            Arc::clone(&store) as Arc<dyn LoopMemory>
        ))),
        None,
    );

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let recalled = store
        .retrieve("launch code", 1)
        .await
        .expect("retrieve over the store");
    assert_eq!(recalled.len(), 1, "one entry matches the recall query");
    assert!(
        recalled[0].memory.contains("ARC-7"),
        "the demoted tool-result region is recallable: {}",
        recalled[0].memory
    );
}

#[tokio::test]
async fn sink_failure_does_not_fail_compaction() {
    let calls = Arc::new(Mutex::new(0usize));
    let compactions = Arc::new(Mutex::new(0usize));
    let mut loop_ = compacting_loop(
        MockApiClient::new("m").with_responses(growing_script()),
        Some(Arc::new(FailingSink {
            calls: Arc::clone(&calls),
        })),
        Some(Arc::new(CountingObserver {
            compactions: Arc::clone(&compactions),
        })),
    );

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("a rejecting sink never fails the run");

    assert!(
        *calls.lock().expect("calls lock") > 0,
        "the sink was invoked with evicted content"
    );
    assert!(
        *compactions.lock().expect("compaction lock") > 0,
        "the passes the sink rejected still completed and reported"
    );
}

#[tokio::test]
async fn noop_default_is_byte_identical() {
    let compactions = Arc::new(Mutex::new(0usize));
    let client = MockApiClient::new("m").with_responses(growing_script());
    let handle = client.clone();
    let mut default_loop = compacting_loop(
        client,
        None,
        Some(Arc::new(CountingObserver {
            compactions: Arc::clone(&compactions),
        })),
    );
    default_loop
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("default run completes");
    assert!(
        *compactions.lock().expect("compaction lock") > 0,
        "the comparison runs are compaction-bearing"
    );
    let default_signatures = request_signatures(&handle);

    let client = MockApiClient::new("m").with_responses(growing_script());
    let noop_handle = client.clone();
    let mut noop_loop = compacting_loop(
        client,
        Some(Arc::new(loopctl::compact::demote::NoopDemotionSink)),
        None,
    );
    noop_loop
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("noop run completes");
    let noop_signatures = request_signatures(&noop_handle);

    assert_eq!(
        default_signatures, noop_signatures,
        "the default loop and the explicit no-op sink issue identical requests"
    );
}

#[tokio::test]
async fn a_cancel_during_a_hanging_sink_ends_the_run_typed() {
    let entered = Arc::new(Mutex::new(0usize));
    let mut loop_ = compacting_loop(
        MockApiClient::new("m").with_responses(growing_script()),
        Some(Arc::new(HangingSink {
            entered: Arc::clone(&entered),
        })),
        None,
    );

    let cancel_signal = loop_.cancel_signal();
    let run = tokio::spawn(async move {
        let outcome = loop_.run("grow then demote", &RunConfig::default()).await;
        (outcome, loop_)
    });
    while *entered.lock().expect("entered lock") == 0 {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    cancel_signal.cancel();
    let joined = tokio::time::timeout(std::time::Duration::from_secs(10), run).await;
    let (outcome, _loop_) = joined
        .expect("the cancellation ends the run rather than hanging")
        .expect("spawned run task finished");
    assert!(
        matches!(outcome, Err(LoopError::Cancelled)),
        "the mid-sink cancel surfaces typed: {outcome:?}"
    );
    assert_eq!(
        *entered.lock().expect("entered lock"),
        1,
        "exactly one sink write began"
    );
}

#[tokio::test]
async fn with_demotion_sink_wires_the_manager() {
    let deliveries: Arc<Mutex<Vec<Delivery>>> = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(FactTool);
    let config = SessionConfig::default()
        .with_context_window(400)
        .with_compact_threshold(50);
    // Built through the consuming builder — the test pins the
    // `with_demotion_sink` spelling itself, not just the setter the
    // shared harness uses.
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(growing_script())),
        registry,
        config,
    )
    .with_context_manager(Arc::new(
        ContextManager::new(Arc::new(
            TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        ))
        .with_context_window(400),
    ))
    .with_demotion_sink(Arc::new(RecordingSink {
        deliveries: Arc::clone(&deliveries),
    }));

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let captured = deliveries.lock().expect("delivery lock").clone();
    assert!(
        !captured.is_empty(),
        "the builder routes evicted content to the sink"
    );
    for (texts, meta) in &captured {
        assert!(
            !texts.is_empty(),
            "every delivery carries the messages the pass removed"
        );
        assert_eq!(
            meta.reason,
            CompactReason::ThresholdExceeded,
            "the metadata names the pass's trigger"
        );
    }
    let session_id = loop_.session().id;
    assert!(
        captured
            .iter()
            .all(|(_, meta)| meta.session_id == session_id),
        "the metadata carries the loop's session id"
    );
}

/// A sink and an observer appending to one ordered log — the shared
/// sequence that pins delivery-precedes-reporting.
struct OrderSink {
    log: Arc<Mutex<Vec<String>>>,
}

impl DemotionSink for OrderSink {
    fn demote<'a>(
        &'a self,
        evicted: &'a [Message],
        _meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        let log = Arc::clone(&self.log);
        let count = evicted.len();
        Box::pin(async move {
            log.lock()
                .expect("order log lock")
                .push(format!("demote:{count}"));
            Ok(())
        })
    }
}

/// The observer twin of [`OrderSink`], recording the pass-completed
/// event on the same log.
struct OrderObserver {
    log: Arc<Mutex<Vec<String>>>,
}

impl LoopObserver for OrderObserver {
    fn name(&self) -> &str {
        "order"
    }

    fn on_compaction(&self, _ctx: &CompactedContext) {
        self.log
            .lock()
            .expect("order log lock")
            .push("on_compaction".to_string());
    }
}

#[tokio::test]
async fn delivery_precedes_the_compaction_observer() {
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = compacting_loop(
        MockApiClient::new("m").with_responses(growing_script()),
        Some(Arc::new(OrderSink {
            log: Arc::clone(&log),
        })),
        Some(Arc::new(OrderObserver {
            log: Arc::clone(&log),
        })),
    );

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let events = log.lock().expect("order log lock").clone();
    let first_delivery = events
        .iter()
        .position(|event| event.starts_with("demote:"))
        .expect("the sink delivered at least once");
    let first_report = events
        .iter()
        .position(|event| event == "on_compaction")
        .expect("the observer reported at least once");
    assert!(
        first_delivery < first_report,
        "delivery precedes the pass-completed report: {events:?}"
    );
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn delivery_precedes_the_post_compact_hook() {
    use loopctl::hooks::context::PostCompactContext;
    use loopctl::hooks::{Hook, HookExecutor};

    struct HookOrderSink {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl DemotionSink for HookOrderSink {
        fn demote<'a>(
            &'a self,
            evicted: &'a [Message],
            _meta: DemotionContext,
        ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
            let log = Arc::clone(&self.log);
            let count = evicted.len();
            Box::pin(async move {
                log.lock()
                    .expect("order log lock")
                    .push(format!("demote:{count}"));
                Ok(())
            })
        }
    }

    struct OrderHook {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl Hook for OrderHook {
        fn name(&self) -> &str {
            "order"
        }

        fn on_post_compact(&self, _ctx: &PostCompactContext) {
            self.log
                .lock()
                .expect("order log lock")
                .push("post_compact_hook".to_string());
        }
    }

    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = compacting_loop(
        MockApiClient::new("m").with_responses(growing_script()),
        Some(Arc::new(HookOrderSink {
            log: Arc::clone(&log),
        })),
        None,
    );
    let mut executor = HookExecutor::new();
    executor.register(Arc::new(OrderHook {
        log: Arc::clone(&log),
    }));
    loop_.set_hook_executor(Arc::new(executor));

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let events = log.lock().expect("order log lock").clone();
    let first_delivery = events
        .iter()
        .position(|event| event.starts_with("demote:"))
        .expect("the sink delivered at least once");
    let first_hook = events
        .iter()
        .position(|event| event == "post_compact_hook")
        .expect("the hook fired at least once");
    assert!(
        first_delivery < first_hook,
        "delivery precedes the post-compact hook: {events:?}"
    );
}

#[tokio::test]
async fn memory_and_sink_share_one_store_across_turns() {
    // The feature's two halves over one store: demotion writes entries
    // while the engine's per-turn retrieval reads the same store, with
    // recall on by default. Several turns past a compaction must
    // complete without a compact-trigger loop — the demoted entry
    // returns as a bounded transient, not as unbounded history.
    let store = Arc::new(InMemoryStore::new());
    let mut responses = Vec::new();
    for i in 0..5 {
        responses.push(MockResponse {
            text: "go".to_string(),
            tool_call: Some(MockToolCall {
                id: format!("c{i}"),
                name: "big".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        });
    }
    responses.push(MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    });
    let mut registry = ToolRegistry::new();
    registry.register(BigResultTool);
    let config = SessionConfig::default()
        .with_context_window(8_000)
        .with_compact_threshold(50);
    let client = MockApiClient::new("m").with_responses(responses);
    let handle = client.clone();
    let mut loop_ = BareLoop::new(Arc::new(client), registry, config);
    loop_.set_context_manager(Arc::new(
        ContextManager::new(Arc::new(
            TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        ))
        .with_context_window(8_000)
        .with_threshold(50),
    ));
    loop_.set_memory(Arc::clone(&store) as Arc<dyn LoopMemory>);
    loop_.set_demotion_sink(Arc::new(
        MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>).with_max_chars(400),
    ));

    loop_
        .run(
            "grow past the trigger, then keep going",
            &RunConfig::default(),
        )
        .await
        .expect("the run survives several turns past a compaction");

    let entries = store
        .retrieve("", usize::MAX)
        .await
        .expect("retrieve over the store");
    assert!(
        entries
            .iter()
            .any(|entry| entry.tags.iter().any(|tag| tag == "demoted")),
        "the sink stored demoted content in the shared store"
    );
    let served = handle.captured_requests().len() + handle.captured_stream_requests().len();
    assert!(
        served >= 5,
        "the conversation kept serving turns after the compaction ({served} requests)"
    );
}

#[tokio::test]
async fn an_emergency_pass_delivers_with_its_reason() {
    // A window whose threshold sits at the emergency line (95%) makes
    // every crossing an emergency: the lean turns accumulate freely
    // under the line, and the first check over it delivers with the
    // Emergency reason while the kept slice (first message plus the
    // lean last turn) stays comfortably under the line.
    let deliveries: Arc<Mutex<Vec<Delivery>>> = Arc::new(Mutex::new(Vec::new()));
    let mut responses = Vec::new();
    for i in 0..8 {
        responses.push(MockResponse {
            text: "go".to_string(),
            tool_call: Some(MockToolCall {
                id: format!("c{i}"),
                name: "sized".to_string(),
                input: serde_json::json!({ "chars": 460 }),
            }),
            stop_reason: "tool_use".to_string(),
        });
    }
    responses.push(MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    });
    let mut registry = ToolRegistry::new();
    registry.register(SizedResultTool);
    let config = SessionConfig::default()
        .with_context_window(1_000)
        .with_compact_threshold(95);
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(responses)),
        registry,
        config,
    );
    loop_.set_context_manager(Arc::new(
        ContextManager::new(Arc::new(
            TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        ))
        .with_context_window(1_000)
        .with_threshold(95),
    ));
    loop_.set_demotion_sink(Arc::new(RecordingSink {
        deliveries: Arc::clone(&deliveries),
    }));

    loop_
        .run("grow into the emergency zone", &RunConfig::default())
        .await
        .expect("the emergency pass compacts and the run completes");

    let captured = deliveries.lock().expect("delivery lock").clone();
    assert!(
        captured
            .iter()
            .any(|(_, meta)| meta.reason == CompactReason::Emergency),
        "at least one delivery carries the emergency reason: {captured:?}"
    );
}

#[tokio::test]
async fn default_budget_recall_does_not_spiral_the_compaction_trigger() {
    // The adversarial sizing the stress review named, driven rather
    // than steered around: the default 8 000-char render budget over
    // a window the recalled entry could dwarf. The run survives —
    // the recalled entry rides the turns as a bounded transient and
    // never feeds the machine's history estimate, so the compaction
    // trigger cannot chase it.
    let store = Arc::new(InMemoryStore::new());
    let mut responses = Vec::new();
    for i in 0..5 {
        responses.push(MockResponse {
            text: "go".to_string(),
            tool_call: Some(MockToolCall {
                id: format!("c{i}"),
                name: "big".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        });
    }
    responses.push(MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    });
    let mut registry = ToolRegistry::new();
    registry.register(BigResultTool);
    let config = SessionConfig::default()
        .with_context_window(8_000)
        .with_compact_threshold(50);
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(responses)),
        registry,
        config,
    );
    loop_.set_context_manager(Arc::new(
        ContextManager::new(Arc::new(
            TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        ))
        .with_context_window(8_000)
        .with_threshold(50),
    ));
    loop_.set_memory(Arc::clone(&store) as Arc<dyn LoopMemory>);
    loop_.set_demotion_sink(Arc::new(MemoryDemotionSink::new(
        Arc::clone(&store) as Arc<dyn LoopMemory>
    )));

    let run = loop_
        .run(
            "grow past the trigger with the default budget",
            &RunConfig::default(),
        )
        .await
        .expect("the default-budget recall does not spiral the trigger");

    let entries = store
        .retrieve("", usize::MAX)
        .await
        .expect("retrieve over the store");
    assert!(
        entries
            .iter()
            .any(|entry| entry.tags.iter().any(|tag| tag == "demoted")),
        "demotion stored entries in the shared store"
    );
    assert!(
        run.turns
            .iter()
            .skip(1)
            .any(|turn| turn.input.contains("yyyyyy")),
        "the recalled demoted entry rode a later turn as a transient"
    );
}
