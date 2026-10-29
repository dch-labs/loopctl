//! Engine span hierarchy end-to-end: what a collector sees for one run.
//!
//! One test function holds every scenario on purpose: `tracing` caches
//! per-callsite interest process-wide, so a span-capture suite cannot
//! afford a sibling test that runs the engine without a subscriber (it
//! would poison the engine's DEBUG callsites for the whole process).
//! Scenarios therefore run in one deterministic sequence: capture first
//! (which pins interest as dynamic), then the INFO-filter probe, then
//! the no-subscriber run — in that order every callsite resolves
//! through the thread's dispatcher.
//!
//! The capture layer records each span's identity and fields at open and
//! merges every `record` the engine performs before close, which is how
//! the assertions see post-construction values (outcome, usage,
//! duration) exactly as an exporter would.

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
use std::sync::{Arc, Mutex};
use std::time::Duration;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, Steering, core::Loop};
use loopctl::message::Message;
use loopctl::observer::{LoopObserver, TurnEndContext, TurnStartContext};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl::tracing_spans::fields;
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

/// A tool that echoes a fixed successful output.
struct EchoTool;

/// Counts the run-start and run-end notifications one loop fires, and
/// captures what each callback carried.
///
/// Exists to pin the pairing and its contents: every run a loop starts
/// must also end, the start must carry the loop's own session
/// identity, and the end of a failed run must report the failure —
/// counts alone would pass a start with the wrong session or an end
/// claiming success.
#[derive(Default)]
struct RunLifecycleCounts {
    starts: std::sync::atomic::AtomicUsize,
    ends: std::sync::atomic::AtomicUsize,
    start_sessions: std::sync::Mutex<Vec<String>>,
    end_states: std::sync::Mutex<Vec<(bool, Option<String>, usize)>>,
}

impl loopctl::observer::LoopObserver for RunLifecycleCounts {
    fn name(&self) -> &'static str {
        "run-lifecycle-counts"
    }

    fn on_run_start(&self, ctx: &loopctl::observer::RunStartContext) {
        self.starts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.start_sessions
            .lock()
            .expect("start sessions lock")
            .push(ctx.session_id.to_string());
    }

    fn on_run_end(&self, ctx: &loopctl::observer::RunEndContext) {
        self.ends.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.end_states.lock().expect("end states lock").push((
            ctx.success,
            ctx.error.clone(),
            ctx.total_turns,
        ));
    }
}

/// A tool whose dispatch outlives the run: it sleeps far past the
/// cancel signal, so the dispatcher's own cancellation select is what
/// ends it — the hard-failure path a tool phase can take.
struct StalledTool;

impl Tool for StalledTool {
    fn name(&self) -> &'static str {
        "stalled"
    }
    fn description(&self) -> &'static str {
        "Sleeps past the cancel signal"
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
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(ToolOutput::text("late"))
        })
    }
}

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

/// The guidance text scenario 1d injects mid-run.
const STEERED_INPUT: &str = "steered: answer with the word done";

/// Captures each turn's start query and steers once, at turn 0's end.
///
/// The turn-start record is the observer-side view of "the turn's input
/// text"; the steering push lands between turn 0 and turn 1, exactly
/// the interleaving where a span built from a pre-drain snapshot would
/// disagree with it.
struct SteeringObserver {
    /// `(turn, query)` pairs as the engine computed them at turn start.
    queries: Mutex<Vec<(usize, String)>>,
    /// The loop's steering slot, handed the guidance at turn 0's end.
    steering: Arc<Steering>,
}

impl LoopObserver for SteeringObserver {
    fn name(&self) -> &'static str {
        "steering-span-probe"
    }

    fn on_turn_start(&self, ctx: &TurnStartContext) {
        self.queries
            .lock()
            .expect("queries lock")
            .push((ctx.turn, ctx.query.clone()));
    }

    fn on_turn_end(&self, ctx: &TurnEndContext) {
        if ctx.turn == 0 {
            self.steering.push(Message::user(STEERED_INPUT));
        }
    }
}

/// A loop wired to the mock client and a registry holding `echo`.
fn scripted_loop(responses: Vec<MockResponse>) -> BareLoop<MockApiClient> {
    let client = Arc::new(MockApiClient::new("test-model").with_responses(responses));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    BareLoop::new(client, registry, SessionConfig::default())
}

/// One captured span: identity, parentage, and every field value it
/// carried at open or picked up via `record` before close.
#[derive(Debug, Clone)]
struct Captured {
    /// Span name as constructed.
    name: String,
    /// Span target.
    target: String,
    /// Parent span's name; `None` when the root was the subscriber.
    parent: Option<String>,
    /// Field values by name (debug rendering, quotes stripped).
    values: Vec<(String, String)>,
}

impl Captured {
    /// The value of `key`, if this span carries it.
    fn field(&self, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// One captured event: name, level, target, and field values.
#[derive(Debug, Clone)]
struct CapturedEvent {
    /// Event message — the vocabulary name ("loopctl.compaction", …).
    name: String,
    /// Level the event fired at.
    level: tracing::Level,
    /// Event target.
    target: String,
    /// Field values by name (debug rendering, quotes stripped).
    values: Vec<(String, String)>,
}

impl CapturedEvent {
    /// The value of `key`, if this event carries it.
    fn field(&self, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Storage shared between the capture layer and the assertions.
#[derive(Default)]
struct Captures {
    /// Captured spans in open order.
    spans: Mutex<Vec<Captured>>,
    /// id → index into `spans`, so `record` finds its span.
    ids: Mutex<Vec<(tracing::Id, usize)>>,
    /// Captured events in fire order.
    events: Mutex<Vec<CapturedEvent>>,
}

/// Layer recording span identity at open and merged records until close.
struct CaptureLayer(Arc<Captures>);

/// Visitor turning field values into plain strings.
struct Stringify(Vec<(String, String)>);

impl Visit for Stringify {
    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        let rendered = format!("{value:?}");
        self.0.push((
            field.name().to_string(),
            rendered.trim_matches('"').to_string(),
        ));
    }
}

impl CaptureLayer {
    /// The stored index for a span id, if this layer has seen it.
    fn index_of(&self, id: &tracing::Id) -> Option<usize> {
        self.0
            .ids
            .lock()
            .expect("ids lock")
            .iter()
            .find(|(i, _)| i == id)
            .map(|(_, i)| *i)
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for CaptureLayer {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: Context<'_, S>,
    ) {
        let parent = ctx
            .span(id)
            .and_then(|span| span.parent())
            .map(|parent| parent.metadata().name().to_string());
        let mut visitor = Stringify(Vec::new());
        attrs.record(&mut visitor);
        let meta = attrs.metadata();
        let mut spans = self.0.spans.lock().expect("spans lock");
        let index = spans.len();
        spans.push(Captured {
            name: meta.name().to_string(),
            target: meta.target().to_string(),
            parent,
            values: visitor.0,
        });
        drop(spans);
        self.0
            .ids
            .lock()
            .expect("ids lock")
            .push((id.clone(), index));
    }

    fn on_record(
        &self,
        id: &tracing::Id,
        values: &tracing::span::Record<'_>,
        _ctx: Context<'_, S>,
    ) {
        let Some(index) = self.index_of(id) else {
            return;
        };
        let mut visitor = Stringify(Vec::new());
        values.record(&mut visitor);
        if let Some(span) = self.0.spans.lock().expect("spans lock").get_mut(index) {
            for (key, value) in visitor.0 {
                if let Some(existing) = span.values.iter_mut().find(|(k, _)| *k == key) {
                    existing.1 = value;
                } else {
                    span.values.push((key, value));
                }
            }
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = Stringify(Vec::new());
        event.record(&mut visitor);
        let name = visitor
            .0
            .iter()
            .find(|(key, _)| key == "message")
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        let meta = event.metadata();
        self.0
            .events
            .lock()
            .expect("events lock")
            .push(CapturedEvent {
                name,
                level: *meta.level(),
                target: meta.target().to_string(),
                values: visitor.0,
            });
    }
}

/// Run `responses` under a fresh capture subscriber and return spans.
async fn capture_run(responses: Vec<MockResponse>) -> Vec<Captured> {
    capture_run_with(responses, false).await
}

/// As [`capture_run`], with the mock client also reporting usage
/// detail on every turn when `with_usage` is set.
async fn capture_run_with(responses: Vec<MockResponse>, with_usage: bool) -> Vec<Captured> {
    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let mut client = MockApiClient::new("test-model").with_responses(responses);
    if with_usage {
        client = client.with_usage(
            loopctl::stream::Usage::new(120, 12)
                .with_cached_input(80)
                .with_cache_write(9)
                .with_reasoning(4),
        );
    }
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut agent = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    agent
        .run("span me", &RunConfig::default())
        .await
        .expect("scripted run completes");
    drop(guard);
    let spans = captures.spans.lock().expect("spans lock").clone();
    spans
        .into_iter()
        .filter(|s| s.name.starts_with("agent."))
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn the_engine_emits_the_documented_span_tree() {
    // Scenario 1: full capture at DEBUG. This is also the FIRST exercise
    // of every engine span callsite — interest resolves dynamically from
    // here on, which is what lets the later scenarios work.
    let spans = capture_run(tool_turn_then_terminal()).await;

    let run = spans
        .iter()
        .find(|s| s.name == loopctl::tracing_spans::AGENT_RUN_SPAN)
        .expect("one agent.run");
    assert_eq!(run.target, loopctl::tracing_spans::TARGET);
    assert_eq!(run.parent, None, "the run span is the root");
    assert_eq!(
        run.field(fields::OPERATION),
        Some("invoke_agent"),
        "GenAI operation at emission"
    );
    assert_eq!(run.field(fields::REQUEST_MODEL), Some("test-model"));
    assert!(
        run.field(fields::CONVERSATION_ID)
            .is_some_and(|v| v.len() >= 32),
        "conversation id is the session uuid"
    );
    assert_eq!(run.field(fields::OUTCOME), Some("ok"), "run settled ok");
    assert!(
        run.field(fields::DURATION_MS).is_some(),
        "duration recorded"
    );

    let turns: Vec<&Captured> = spans
        .iter()
        .filter(|s| s.name == loopctl::tracing_spans::TURN_SPAN)
        .collect();
    assert_eq!(turns.len(), 2, "tool turn plus terminal turn");
    assert_eq!(
        turns.first().expect("first turn").field(fields::TURN),
        Some("0")
    );
    assert_eq!(
        turns.first().expect("first turn").field(fields::QUERY),
        Some("span me")
    );
    for turn in &turns {
        assert_eq!(
            turn.parent.as_deref(),
            Some(loopctl::tracing_spans::AGENT_RUN_SPAN),
            "turns nest under the run"
        );
        assert_eq!(turn.field(fields::OUTCOME), Some("ok"));
        assert!(turn.field(fields::USAGE_INPUT_TOKENS).is_some());
        assert!(turn.field(fields::USAGE_OUTPUT_TOKENS).is_some());
        assert!(turn.field(fields::DURATION_MS).is_some());
        assert!(
            turn.field(fields::USAGE_CACHE_READ_INPUT_TOKENS).is_none(),
            "no report, no field — absence means the provider said nothing"
        );
        assert!(turn.field(fields::USAGE_CACHE_WRITE_INPUT_TOKENS).is_none());
        assert!(turn.field(fields::RESPONSE_REASONING_TOKENS).is_none());
    }

    let tool = spans
        .iter()
        .find(|s| s.name == loopctl::tracing_spans::TOOL_SPAN)
        .expect("one agent.tool for the echo call");
    assert_eq!(tool.field(fields::OPERATION), Some("execute_tool"));
    assert_eq!(tool.field(fields::TOOL_NAME), Some("echo"));
    assert_eq!(tool.field(fields::TOOL_CALL_ID), Some("call_1"));
    assert_eq!(tool.field(fields::OUTCOME), Some("ok"));
    assert!(tool.field(fields::DURATION_MS).is_some());
    assert_eq!(
        tool.parent.as_deref(),
        Some(loopctl::tracing_spans::AGENT_RUN_SPAN),
        "the driver dispatches tool spans beside the finished turn, under the run"
    );

    // Scenario 1b: the same tree with reported usage detail — the
    // presence-not-zero fields arrive exactly where a figure was
    // reported.
    let used = capture_run_with(tool_turn_then_terminal(), true).await;
    let turns: Vec<&Captured> = used
        .iter()
        .filter(|s| s.name == loopctl::tracing_spans::TURN_SPAN)
        .collect();
    assert_eq!(turns.len(), 2, "both turns settle with usage");
    for turn in &turns {
        assert_eq!(turn.field(fields::USAGE_INPUT_TOKENS), Some("120"));
        assert_eq!(turn.field(fields::USAGE_OUTPUT_TOKENS), Some("12"));
        assert_eq!(
            turn.field(fields::USAGE_CACHE_READ_INPUT_TOKENS),
            Some("80")
        );
        assert_eq!(
            turn.field(fields::USAGE_CACHE_WRITE_INPUT_TOKENS),
            Some("9")
        );
        assert_eq!(turn.field(fields::RESPONSE_REASONING_TOKENS), Some("4"));
    }

    // Scenario 1c: a failed run closes its run span with the error
    // vocabulary — outcome `error` plus the bounded `error.type` class,
    // the legs none of the success scenarios above exercise. The mock
    // fails the first provider call, which fails the run through the
    // same finalize path every failure takes.
    let failing_captures = Arc::new(Captures::default());
    let failing_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&failing_captures)));
    let failing_guard = tracing::subscriber::set_default(failing_subscriber);
    let failing_client =
        MockApiClient::new("test-model").with_errors(vec![Some("provider exploded".to_string())]);
    let failing_registry = ToolRegistry::new();
    failing_registry.register(EchoTool);
    let mut failing_agent = BareLoop::new(
        Arc::new(failing_client),
        failing_registry,
        SessionConfig::default(),
    );
    let failed = failing_agent.run("doomed run", &RunConfig::default()).await;
    drop(failing_guard);
    assert!(
        failed.is_err(),
        "the scripted provider failure fails the run"
    );
    let failed_spans = failing_captures.spans.lock().expect("spans lock").clone();
    let failed_run = failed_spans
        .iter()
        .find(|s| s.name == loopctl::tracing_spans::AGENT_RUN_SPAN)
        .expect("the failed run still opens its run span");
    assert_eq!(
        failed_run.field(fields::OUTCOME),
        Some("error"),
        "a failed run records outcome=error"
    );
    assert_eq!(
        failed_run.field(fields::ERROR_TYPE),
        Some("api"),
        "the bounded class name lands, never the message text"
    );
    assert!(
        failed_run.field(fields::DURATION_MS).is_some(),
        "the failed run still measures its duration"
    );

    // Scenario 1d: a tool phase that fails hard settles its turn span
    // with the failure vocabulary — the settlement a tool-bearing turn
    // owes after the model phase, which an early `ok` recorded before
    // dispatch would falsify. The scripted failure is a cancellation
    // during dispatch (the dispatcher's own select ends a stalled
    // tool): the hard-failure path every dispatch error takes.
    let cancel_captures = Arc::new(Captures::default());
    let cancel_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&cancel_captures)));
    let cancel_guard = tracing::subscriber::set_default(cancel_subscriber);
    let cancel_client =
        Arc::new(
            MockApiClient::new("test-model").with_responses(vec![MockResponse {
                text: "calling stalled".to_string(),
                tool_call: Some(MockToolCall {
                    id: "call_1".to_string(),
                    name: "stalled".to_string(),
                    input: serde_json::json!({}),
                }),
                stop_reason: "tool_use".to_string(),
            }]),
        );
    let cancel_registry = ToolRegistry::new();
    cancel_registry.register(StalledTool);
    let mut cancel_agent = BareLoop::new(cancel_client, cancel_registry, SessionConfig::default());
    let cancel_signal = cancel_agent.cancel_signal();
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel_signal.cancel();
    });
    let run_outcome = cancel_agent
        .run("tool phase fails hard", &RunConfig::default())
        .await;
    canceller.await.expect("the canceller joins");
    drop(cancel_guard);
    assert!(
        run_outcome.is_err(),
        "the mid-dispatch cancel fails the run: {run_outcome:?}"
    );
    let cancel_spans = cancel_captures.spans.lock().expect("spans lock").clone();
    let cancelled_turn = cancel_spans
        .iter()
        .find(|span| {
            span.name == loopctl::tracing_spans::TURN_SPAN && span.field(fields::TURN) == Some("0")
        })
        .expect("the tool turn's span opened");
    assert_eq!(
        cancelled_turn.field(fields::OUTCOME),
        Some("error"),
        "the dispatch failure settles the turn span — never an ok recorded \
         before the tool phase ran"
    );
    assert!(
        cancelled_turn.field(fields::USAGE_INPUT_TOKENS).is_some(),
        "the failure settlement still carries the attempt's usage"
    );
    let cancel_events = cancel_captures.events.lock().expect("events lock").clone();
    let cancel_signal_event = cancel_events
        .iter()
        .find(|event| event.name == "loopctl.cancelled")
        .expect("the cancellation narrates itself as a signal event");
    assert_eq!(
        cancel_signal_event.field("turn"),
        Some("1"),
        "the cancelled run's event carries its settled turn count — the \
         tool turn settled before the dispatch cancel landed"
    );
    assert_eq!(
        cancel_signal_event.target, "loopctl::engine",
        "signal events ride the engine target the filter contract names"
    );
    assert!(
        cancelled_turn.field(fields::DURATION_MS).is_some(),
        "the failure settlement still carries the attempt's duration"
    );

    // Scenario 1e: the run-span settlement helper writes the failure
    // vocabulary onto a run span — outcome, duration, and the bounded
    // error class — and nothing at all onto a foreign span.
    let helper_captures = Arc::new(Captures::default());
    let helper_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&helper_captures)));
    let helper_guard = tracing::subscriber::set_default(helper_subscriber);
    let run_span = loopctl::tracing_spans::agent_run_span(uuid::Uuid::new_v4(), "test-model");
    run_span.in_scope(|| {
        loopctl::tracing_spans::record_run_end(
            false,
            1_234,
            Some(&loopctl::error::LoopError::Cancelled),
        );
    });
    drop(run_span);
    let turn_span = loopctl::tracing_spans::turn_span(&TurnStartContext {
        turn: 0,
        query: String::new(),
    });
    turn_span.in_scope(|| {
        loopctl::tracing_spans::record_run_end(true, 5, None);
    });
    drop(turn_span);
    drop(helper_guard);
    let helper_spans = helper_captures.spans.lock().expect("spans lock").clone();
    let settled_run = helper_spans
        .iter()
        .find(|span| span.name == loopctl::tracing_spans::AGENT_RUN_SPAN)
        .expect("the run span opened");
    assert_eq!(settled_run.field(fields::OUTCOME), Some("error"));
    assert_eq!(settled_run.field(fields::DURATION_MS), Some("1234"));
    assert_eq!(
        settled_run.field(fields::ERROR_TYPE),
        Some("cancelled"),
        "the bounded class name lands, never the message text"
    );
    let foreign_turn = helper_spans
        .iter()
        .find(|span| span.name == loopctl::tracing_spans::TURN_SPAN)
        .expect("the turn span opened");
    assert!(
        foreign_turn.field(fields::OUTCOME).is_none(),
        "run facts never land on a foreign span"
    );

    // Scenario 1f: mid-run steering — the span's `query` and the
    // turn-start observer's query are the same string, the one computed
    // after the steering drain. A span built from a pre-drain snapshot
    // would show turn 0's tool-result text instead of the guidance.
    let steered_captures = Arc::new(Captures::default());
    let steered_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&steered_captures)));
    let steered_guard = tracing::subscriber::set_default(steered_subscriber);
    let steered_loop = scripted_loop(tool_turn_then_terminal());
    let steering = steered_loop.steering();
    let observer = Arc::new(SteeringObserver {
        queries: Mutex::new(Vec::new()),
        steering: Arc::clone(&steering),
    });
    let mut steered_agent = steered_loop.with_observer(observer.clone());
    steered_agent
        .run("original question", &RunConfig::default())
        .await
        .expect("steered run completes");
    drop(steered_guard);
    let queries = observer.queries.lock().expect("queries lock").clone();
    assert_eq!(
        queries,
        vec![
            (0, "original question".to_string()),
            (1, STEERED_INPUT.to_string()),
        ],
        "the observer sees the steering text as turn 1's input"
    );
    let steered_events = steered_captures.events.lock().expect("events lock").clone();
    let drain = steered_events
        .iter()
        .find(|event| event.name == "loopctl.steering_drain")
        .expect("the steering boundary narrates itself as a signal event");
    assert_eq!(
        drain.field("count"),
        Some("1"),
        "the drain event counts the messages that joined the turn"
    );
    let steered_spans = steered_captures.spans.lock().expect("spans lock").clone();
    let steered_turn = steered_spans
        .iter()
        .find(|span| {
            span.name == loopctl::tracing_spans::TURN_SPAN && span.field(fields::TURN) == Some("1")
        })
        .expect("the steered turn's span opened");
    assert_eq!(
        steered_turn.field(fields::REQUEST_MODEL),
        Some("test-model"),
        "each turn span names the model that served it"
    );
    let span_query = |turn: usize| {
        steered_spans
            .iter()
            .filter(|s| s.name == loopctl::tracing_spans::TURN_SPAN)
            .find(|s| s.field(fields::TURN) == Some(turn.to_string().as_str()))
            .and_then(|s| s.field(fields::QUERY).map(str::to_string))
    };
    assert_eq!(
        span_query(1).as_deref(),
        Some(STEERED_INPUT),
        "the turn span records the post-drain input, never the pre-steer snapshot"
    );
    for (turn, query) in &queries {
        assert_eq!(
            span_query(*turn).as_deref(),
            Some(query.as_str()),
            "span and observer agree on turn {turn}'s input"
        );
    }

    // Scenario 2: an INFO host filter keeps an events-only log — no
    // engine spans materialize at all (the DEBUG contract in practice).
    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry()
        .with(CaptureLayer(Arc::clone(&captures)))
        .with(tracing::level_filters::LevelFilter::INFO);
    let guard = tracing::subscriber::set_default(subscriber);
    let mut agent = scripted_loop(tool_turn_then_terminal());
    agent
        .run("filtered out", &RunConfig::default())
        .await
        .expect("run at INFO completes");
    drop(guard);
    assert!(
        captures.spans.lock().expect("spans lock").is_empty(),
        "an INFO filter sees zero engine spans"
    );

    // Scenario 3: zero subscriber — the spans cost nothing and change
    // nothing; the run behaves exactly as the uninstrumented one.
    let mut agent = scripted_loop(tool_turn_then_terminal());
    let run = agent
        .run("no listener", &RunConfig::default())
        .await
        .expect("run without a subscriber completes");
    assert!(run.turns.len() >= 2, "history is unaffected");

    // Interest re-opens the moment any subscriber accepts DEBUG again —
    // the callsites answer dynamically, which is what lets the no-cost
    // claim above hold without a latch.
    let reopened = Arc::new(Captures::default());
    let open = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&reopened)));
    let open_guard = tracing::subscriber::set_default(open);
    let mut agent = scripted_loop(tool_turn_then_terminal());
    agent
        .run("capture again", &RunConfig::default())
        .await
        .expect("run under a fresh capture subscriber completes");
    drop(open_guard);
    assert!(
        reopened
            .spans
            .lock()
            .expect("spans lock")
            .iter()
            .any(|s| s.name == loopctl::tracing_spans::AGENT_RUN_SPAN),
        "a fresh DEBUG-accepting subscriber sees spans again"
    );

    // A manager reset that finds a poisoned lock fails the run before any
    // turn opens, and that failure still settles the run span: outcome
    // `error`, the error class the run returned on, and a duration. An
    // early return that skipped the settlement would leave the span
    // hanging with no outcome at all.
    let reset_captures = Arc::new(Captures::default());
    let reset_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&reset_captures)));
    let reset_guard = tracing::subscriber::set_default(reset_subscriber);
    let detection = loopctl::detection::DetectionManager::default();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = detection
            .convergence_detector()
            .lock()
            .expect("lock before poison");
        panic!("poison the convergence detector");
    }));
    assert!(poisoned.is_err(), "the poisoning panic is what poisons");
    let lifecycle = std::sync::Arc::new(RunLifecycleCounts::default());
    let managers = loopctl::managers::LoopManagers::new()
        .with_detection(detection)
        .with_observer(lifecycle.clone());
    let reset_client =
        Arc::new(
            MockApiClient::new("test-model").with_responses(vec![MockResponse {
                text: "never reached".to_string(),
                tool_call: None,
                stop_reason: "end_turn".to_string(),
            }]),
        );
    let mut reset_agent = BareLoop::new_with_managers(
        reset_client,
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    let mut reset_config = RunConfig::default();
    reset_config.reset_managers = true;
    let reset_outcome = reset_agent
        .run("the reset poisons first", &reset_config)
        .await;
    drop(reset_guard);
    assert!(
        matches!(
            reset_outcome,
            Err(loopctl::error::LoopError::LockPoisoned { .. })
        ),
        "a poisoned manager reset fails the run with its own error: {reset_outcome:?}"
    );
    let reset_spans = reset_captures.spans.lock().expect("spans lock").clone();
    let failed_run = reset_spans
        .iter()
        .find(|s| s.name == loopctl::tracing_spans::AGENT_RUN_SPAN)
        .expect("the run span opened before the reset was attempted");
    assert_eq!(
        failed_run.field(fields::OUTCOME),
        Some("error"),
        "the reset failure settles the run span as failed"
    );
    assert_eq!(
        failed_run.field(fields::ERROR_TYPE),
        Some("lock_poisoned"),
        "the settlement names the class of error the run returned on"
    );
    assert!(
        failed_run.field(fields::DURATION_MS).is_some(),
        "the settlement carries a duration even though no turn ever ran"
    );
    assert!(
        reset_spans
            .iter()
            .all(|s| s.name != loopctl::tracing_spans::TURN_SPAN),
        "the reset fails the run before any turn opens"
    );
    assert_eq!(
        (
            lifecycle.starts.load(std::sync::atomic::Ordering::SeqCst),
            lifecycle.ends.load(std::sync::atomic::Ordering::SeqCst)
        ),
        (1, 1),
        "a reset-refused run announces a matched start and end — the end \
         carries the reset failure, so no observer is left holding a \
         start with no run to close"
    );
    let session_id = reset_agent.session().id.to_string();
    assert_eq!(
        *lifecycle
            .start_sessions
            .lock()
            .expect("start sessions lock"),
        [session_id],
        "the start carries the loop's own session identity"
    );
    let end_states = lifecycle
        .end_states
        .lock()
        .expect("end states lock")
        .clone();
    assert_eq!(end_states.len(), 1, "exactly one run end fired");
    let (success, error, turns) = end_states.first().expect("the single end state exists");
    assert!(
        !success,
        "the refused run's end reports failure, not success"
    );
    assert!(
        error
            .as_deref()
            .is_some_and(|text| text.to_ascii_lowercase().contains("poison")),
        "the end carries the reset failure's own error class, got {error:?}"
    );
    assert_eq!(
        *turns, 0,
        "the refused run ran no turns, and its end says so"
    );
}

/// The field-name constants are the exporter contract; pin every one.
#[test]
fn field_constants_carry_their_genai_spellings() {
    assert_eq!(fields::OPERATION, "gen_ai.operation.name");
    assert_eq!(fields::CONVERSATION_ID, "gen_ai.conversation.id");
    assert_eq!(fields::REQUEST_MODEL, "gen_ai.request.model");
    assert_eq!(fields::TURN, "turn");
    assert_eq!(fields::QUERY, "query");
    assert_eq!(fields::TOOL_NAME, "gen_ai.tool.name");
    assert_eq!(fields::TOOL_CALL_ID, "gen_ai.tool.call.id");
    assert_eq!(fields::USAGE_INPUT_TOKENS, "gen_ai.usage.input_tokens");
    assert_eq!(fields::USAGE_OUTPUT_TOKENS, "gen_ai.usage.output_tokens");
    assert_eq!(
        fields::USAGE_CACHE_READ_INPUT_TOKENS,
        "gen_ai.usage.cache_read.input_tokens"
    );
    assert_eq!(
        fields::USAGE_CACHE_WRITE_INPUT_TOKENS,
        "gen_ai.usage.cache_creation.input_tokens"
    );
    assert_eq!(
        fields::RESPONSE_REASONING_TOKENS,
        "gen_ai.response.reasoning_tokens"
    );
    assert_eq!(fields::ERROR_TYPE, "error.type");
    assert_eq!(fields::OUTCOME, "outcome");
    assert_eq!(fields::DURATION_MS, "duration_ms");
}

#[test]
fn signal_events_carry_their_documented_names_fields_and_levels() {
    use loopctl::compact::types::CompactReason;
    use loopctl::tracing_spans as signals;

    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);

    signals::event_compaction(
        CompactReason::ThresholdExceeded,
        900,
        400,
        signals::CompactionPassOutcome::Applied,
    );
    signals::event_compaction(
        CompactReason::Manual,
        900,
        900,
        signals::CompactionPassOutcome::Noop,
    );
    signals::event_compaction(
        CompactReason::Emergency,
        700,
        700,
        signals::CompactionPassOutcome::Deferred,
    );
    signals::event_compaction(
        CompactReason::Prune,
        0,
        0,
        signals::CompactionPassOutcome::Failed,
    );
    signals::event_steering_drain(2);
    signals::event_model_switch("model-a", "model-b");
    signals::event_transport_fallback(3);
    signals::event_salvage(signals::SalvageKind::Overflow, 1, 512);
    signals::event_salvage(signals::SalvageKind::LengthCut, 2, 256);
    signals::event_cancelled(0);
    signals::event_hook_followup("hook:extended");
    drop(guard);

    let events = captures.events.lock().expect("events lock").clone();
    for name in signals::SIGNAL_EVENT_NAMES {
        assert!(
            events.iter().any(|event| event.name == *name),
            "the vocabulary fires `{name}` — a helper that drifts from \
             SIGNAL_EVENT_NAMES fails here"
        );
    }
    let named = |name: &str| {
        events
            .iter()
            .find(|event| event.name == name)
            .unwrap_or_else(|| panic!("no {name} event fired, got {events:#?}"))
    };
    for event in &events {
        assert_eq!(
            event.level,
            tracing::Level::DEBUG,
            "signal events are DEBUG — the filter contract keeps them behind the same opt-in as the span tree: {event:?}"
        );
        assert_eq!(
            event.target, "loopctl::engine",
            "signal events ride the engine target: {event:?}"
        );
    }
    let compaction_events: Vec<_> = events
        .iter()
        .filter(|event| event.name == "loopctl.compaction")
        .collect();
    assert_eq!(
        compaction_events.len(),
        4,
        "one compaction event per pass, never per observer — and every \
         reason spelling rides one of them"
    );
    let deferred = compaction_events
        .iter()
        .find(|event| event.field("outcome") == Some("deferred"))
        .expect("the deferred pass is narrated");
    assert!(
        deferred.field("reason").is_some_and(|r| r == "emergency"),
        "the emergency reason keeps its own spelling"
    );
    assert!(
        deferred.field("tokens_before").is_some_and(|t| t == "700")
            && deferred.field("tokens_after").is_some_and(|t| t == "700"),
        "a deferred pass ran no compaction, so after equals before"
    );
    let failed = compaction_events
        .iter()
        .find(|event| event.field("outcome") == Some("failed"))
        .expect("the failed pass is narrated");
    assert!(
        failed.field("reason").is_some_and(|r| r == "prune"),
        "the prune reason keeps its own spelling"
    );
    assert!(
        failed.field("tokens_before").is_some_and(|t| t == "0")
            && failed.field("tokens_after").is_some_and(|t| t == "0"),
        "a failed pass that never measured reports zeros"
    );
    let applied = compaction_events
        .iter()
        .find(|event| event.field("outcome") == Some("applied"))
        .expect("the applied pass is narrated");
    assert!(applied.field("reason").is_some_and(|r| r == "threshold"));
    assert!(applied.field("tokens_before").is_some_and(|t| t == "900"));
    assert!(applied.field("tokens_after").is_some_and(|t| t == "400"));
    let noop = compaction_events
        .iter()
        .find(|event| event.field("outcome") == Some("noop"))
        .expect("the noop pass is narrated");
    assert!(noop.field("reason").is_some_and(|r| r == "manual"));
    let drain = named("loopctl.steering_drain");
    assert!(drain.field("count").is_some_and(|c| c == "2"));
    let switch = named("loopctl.model_switch");
    assert!(switch.field("from").is_some_and(|f| f == "model-a"));
    assert!(switch.field("to").is_some_and(|t| t == "model-b"));
    let transport = named("loopctl.transport_fallback");
    assert!(transport.field("turn").is_some_and(|t| t == "3"));
    let salvage_events: Vec<_> = events
        .iter()
        .filter(|event| event.name == "loopctl.salvage")
        .collect();
    assert_eq!(salvage_events.len(), 2, "one salvage event per retry");
    let overflow = salvage_events
        .iter()
        .find(|event| event.field("kind") == Some("overflow"))
        .expect("the overflow salvage is narrated");
    assert!(overflow.field("turn").is_some_and(|t| t == "1"));
    assert!(overflow.field("target_tokens").is_some_and(|t| t == "512"));
    assert!(
        salvage_events
            .iter()
            .any(|event| event.field("kind") == Some("length_cut")),
        "the length-cut salvage is narrated under its own kind"
    );
    let cancelled = named("loopctl.cancelled");
    assert!(cancelled.field("turn").is_some_and(|t| t == "0"));
    let followup = named("loopctl.hook_followup");
    assert!(
        followup
            .field("origin")
            .is_some_and(|o| o == "hook:extended")
    );

    // The model-switch signal through its real entry point: the public
    // switch API, not the helper — a deleted engine call site cannot
    // pass this arm.
    let switch_captures = Arc::new(Captures::default());
    let switch_subscriber =
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&switch_captures)));
    let switch_guard = tracing::subscriber::set_default(switch_subscriber);
    let mut switch_agent = scripted_loop(vec![MockResponse {
        text: "unused".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }]);
    switch_agent
        .switch_model("model-b")
        .apply()
        .expect("the mock accepts a model switch");
    drop(switch_guard);
    let switch_event = switch_captures
        .events
        .lock()
        .expect("events lock")
        .iter()
        .find(|event| event.name == "loopctl.model_switch")
        .expect("a real switch narrates itself through the engine path")
        .clone();
    assert!(
        switch_event
            .field("from")
            .is_some_and(|f| f == "test-model")
            && switch_event.field("to").is_some_and(|t| t == "model-b"),
        "the switch event names both ends: {switch_event:?}"
    );
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn an_accepted_followup_narrates_itself_as_a_signal_event() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use loopctl::hooks::context::RunEndContext;
    use loopctl::hooks::{Hook, HookExecutor, PostRunAction};
    use loopctl::managers::LoopManagers;

    /// A hook that always offers one follow-up.
    struct ExtendingHook {
        consults: AtomicUsize,
    }

    impl Hook for ExtendingHook {
        fn name(&self) -> &'static str {
            "extending"
        }

        fn on_run_followup(&self, _ctx: &RunEndContext) -> Option<PostRunAction> {
            self.consults.fetch_add(1, Ordering::SeqCst);
            Some(PostRunAction::FollowUp {
                prompt: "and then?".to_string(),
                origin: "hook:extending".to_string(),
            })
        }
    }

    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);

    let client = Arc::new(MockApiClient::new("test-model").with_responses(vec![
        MockResponse {
            text: "clean answer".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
        MockResponse {
            text: "follow-up answer".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]));
    let hook = Arc::new(ExtendingHook {
        consults: AtomicUsize::new(0),
    });
    let executor = HookExecutor::new().with_hook(Arc::clone(&hook) as _);
    let managers = LoopManagers::new().with_hook_executor(Arc::new(executor));
    let mut agent = BareLoop::new_with_managers(
        client,
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    agent
        .run("initial", &RunConfig::default())
        .await
        .expect("the extended run completes");
    drop(guard);

    assert_eq!(
        hook.consults.load(Ordering::SeqCst),
        1,
        "the hook was consulted exactly once"
    );
    let event = captures
        .events
        .lock()
        .expect("events lock")
        .iter()
        .find(|event| event.name == "loopctl.hook_followup")
        .expect("an accepted follow-up narrates itself through the engine path")
        .clone();
    assert_eq!(
        event.field("origin"),
        Some("hook:extending"),
        "the event carries the hook-declared id, never the prompt"
    );
}

#[cfg(feature = "streaming")]
#[tokio::test]
async fn a_fallback_served_turn_span_names_the_routing_model() {
    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
    use loopctl::fallback::FallbackManager;
    use loopctl::managers::LoopManagers;
    use loopctl::stream::{
        DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
        PartStart, StreamEvent,
    };

    enum Step {
        AuthFail,
        Text(String),
    }

    struct RoutingClient {
        requests: Arc<Mutex<Vec<Option<String>>>>,
        script: Mutex<Vec<Step>>,
        model: Mutex<String>,
    }

    impl RoutingClient {
        fn new(script: Vec<Step>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                script: Mutex::new(script),
                model: Mutex::new("primary-model".to_string()),
            }
        }
    }

    impl ApiClient for RoutingClient {
        fn model(&self) -> String {
            self.model.lock().expect("model lock").clone()
        }

        fn set_model(&self, model: &str) -> bool {
            *self.model.lock().expect("model lock") = model.to_string();
            true
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            Box::pin(async { Err(ApiError::api("this test drives the streaming path")) })
        }

        fn stream_messages_with_options(
            &self,
            _request: &StreamRequest,
            options: loopctl::structured::RequestOptions,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            self.requests
                .lock()
                .expect("requests lock")
                .push(options.model);
            let events = match self.script.lock().expect("script lock").remove(0) {
                Step::AuthFail => {
                    return Box::pin(futures::stream::iter(vec![Err(
                        ApiError::auth_invalid_key("scripted permanent failure"),
                    )]));
                }
                Step::Text(text) => text_events(&text),
            };
            Box::pin(futures::stream::iter(events))
        }
    }

    fn text_events(text: &str) -> Vec<Result<StreamEvent, ApiError>> {
        vec![
            Ok(StreamEvent::MessageStart(MessageStart {
                message: MessageMetadata {
                    id: "msg_1".into(),
                    role: "assistant".into(),
                    model: "served".into(),
                },
            })),
            Ok(StreamEvent::PartStart(PartStart {
                index: 0,
                part: Some(loopctl::message::MessagePart::text("")),
            })),
            Ok(StreamEvent::IndexedDelta(IndexedDelta {
                index: 0,
                delta: DeltaPart::Text { text: text.into() },
            })),
            Ok(StreamEvent::PartStop { index: Some(0) }),
            Ok(StreamEvent::MessageDelta(MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                },
                usage: None,
            })),
            Ok(StreamEvent::MessageStop),
        ]
    }

    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .expect("original model set");
    manager
        .set_fallback_model("fallback-model")
        .expect("fallback model set");

    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let client = RoutingClient::new(vec![Step::AuthFail, Step::Text("recovered".to_string())]);
    let requests = Arc::clone(&client.requests);
    let managers = LoopManagers::new().with_fallback(manager);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    let first = agent.run("q", &RunConfig::default()).await;
    assert!(
        first.is_err(),
        "the scripted permanent failure fails the run"
    );
    let second = agent.run("q", &RunConfig::default()).await;
    assert!(
        second.is_ok(),
        "the fallback model serves the next run: {second:?}"
    );
    drop(guard);

    let observed = requests.lock().expect("requests lock").clone();
    assert_eq!(
        observed,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string())
        ],
        "the run after the trip was actually served by the fallback model"
    );
    let spans = captures.spans.lock().expect("spans lock").clone();
    let turn_spans: Vec<&Captured> = spans
        .iter()
        .filter(|s| s.name == loopctl::tracing_spans::TURN_SPAN)
        .collect();
    let fallback_served_turn = turn_spans
        .last()
        .expect("the fallback-served run opened its turn span");
    assert_eq!(
        fallback_served_turn.field(fields::REQUEST_MODEL),
        Some("fallback-model"),
        "the turn span names the model that served the turn — a \
         fallback-served turn is never attributed to the primary"
    );
    let run_spans: Vec<&Captured> = spans
        .iter()
        .filter(|s| s.name == loopctl::tracing_spans::AGENT_RUN_SPAN)
        .collect();
    assert_eq!(
        run_spans
            .last()
            .expect("two runs opened run spans")
            .field(fields::REQUEST_MODEL),
        Some("primary-model"),
        "the run span still names the run's starting model — only the \
         turn span follows the routing"
    );
}

/// A compactor that fails every pass with a fixed cause.
struct FailingCompactor;

impl loopctl::compact::ContextCompactor for FailingCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        _target_tokens: u64,
        _context: loopctl::compact::CompactionContext,
    ) -> Pin<Box<dyn Future<Output = loopctl::compact::CompactionOutcome> + Send + '_>> {
        use loopctl::compact::TokenCounter;
        Box::pin(async move {
            let tokens_after = loopctl::compact::HeuristicTokenCounter.count(&messages);
            loopctl::compact::CompactionOutcome::failed(
                messages,
                tokens_after,
                "summarizer unavailable",
            )
        })
    }
}

/// A compactor that reports success with a result larger than its
/// input — the pass measured, but the window refuses what it produced.
struct OversizedCompactor;

impl loopctl::compact::ContextCompactor for OversizedCompactor {
    fn compact(
        &self,
        mut messages: Vec<Message>,
        _target_tokens: u64,
        _context: loopctl::compact::CompactionContext,
    ) -> Pin<Box<dyn Future<Output = loopctl::compact::CompactionOutcome> + Send + '_>> {
        use loopctl::compact::TokenCounter;
        Box::pin(async move {
            let tokens_before = loopctl::compact::HeuristicTokenCounter.count(&messages);
            messages.push(Message::user("p".repeat(3_000)));
            let tokens_after = loopctl::compact::HeuristicTokenCounter.count(&messages);
            loopctl::compact::CompactionOutcome::compacted(messages, tokens_before, tokens_after)
        })
    }
}

/// A compactor that shrinks to one small summary message, recording the
/// target each pass was aimed at.
struct TargetRecordingShrinker {
    targets: Mutex<Vec<u64>>,
}

impl loopctl::compact::ContextCompactor for TargetRecordingShrinker {
    fn compact(
        &self,
        _messages: Vec<Message>,
        target_tokens: u64,
        _context: loopctl::compact::CompactionContext,
    ) -> Pin<Box<dyn Future<Output = loopctl::compact::CompactionOutcome> + Send + '_>> {
        self.targets
            .lock()
            .expect("targets lock")
            .push(target_tokens);
        Box::pin(async move {
            use loopctl::compact::TokenCounter;
            let summary = vec![Message::user("compacted summary")];
            let tokens = loopctl::compact::HeuristicTokenCounter.count(&summary);
            loopctl::compact::CompactionOutcome::compacted(
                summary,
                tokens.saturating_mul(10),
                tokens,
            )
        })
    }
}

/// An observer capturing the driver-measured `tokens_before` every
/// failed compaction pass reported.
#[derive(Default)]
struct CompactionFailureFigures {
    befores: Mutex<Vec<u64>>,
}

impl LoopObserver for CompactionFailureFigures {
    fn name(&self) -> &'static str {
        "compaction-failure-figures"
    }

    fn on_compaction_failed(&self, ctx: &loopctl::observer::CompactionFailedContext) {
        self.befores
            .lock()
            .expect("befores lock")
            .push(ctx.tokens_before);
    }
}

/// A tiny-window config whose system prompt and a 1 000-char prompt
/// cross the 80% threshold on the first turn.
fn tiny_window_config(window: u64) -> SessionConfig {
    SessionConfig::default()
        .with_context_window(window)
        .with_compact_threshold(80)
        .with_system_prompt("s".repeat(200))
}

/// The events one compaction-forcing run emitted, paired with the
/// observer's driver-measured figures in pass order.
///
/// `usage` seeds the engine's usage anchor and calibration: when set,
/// the first served turn reports it, so every later `context_figure`
/// conversion runs on a non-identity provider-token scale — the two
/// scales the compaction figures could mix are observably different.
async fn compaction_run(
    compactor: Arc<dyn loopctl::compact::ContextCompactor>,
    window: u64,
    usage: Option<loopctl::stream::Usage>,
) -> (
    Vec<CapturedEvent>,
    Vec<u64>,
    Result<loopctl::engine::Run, loopctl::error::LoopError>,
) {
    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let mut client = MockApiClient::new("test-model").with_responses(vec![
        MockResponse {
            text: "calling echo".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_0".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]);
    if let Some(usage) = usage {
        client = client.with_usage(usage);
    }
    let client = Arc::new(client);
    let figures = Arc::new(CompactionFailureFigures::default());
    let managers = loopctl::managers::LoopManagers::new()
        .with_observer(Arc::clone(&figures) as Arc<dyn LoopObserver>)
        .with_context_manager(Arc::new(loopctl::compact::ContextManager::new(compactor)));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut agent =
        BareLoop::new_with_managers(client, registry, tiny_window_config(window), managers);
    let outcome = agent.run(&"x".repeat(1_000), &RunConfig::default()).await;
    drop(guard);
    let events = captures.events.lock().expect("events lock").clone();
    let befores = figures.befores.lock().expect("befores lock").clone();
    (events, befores, outcome)
}

#[tokio::test]
async fn a_failed_compactor_pass_narrates_its_measured_figures() {
    let (events, befores, outcome) = compaction_run(
        Arc::new(FailingCompactor),
        950,
        Some(loopctl::stream::Usage::new(900, 1)),
    )
    .await;
    assert!(
        matches!(
            outcome,
            Err(loopctl::error::LoopError::CompactionFailed { .. })
        ),
        "the scenario is the failed arm: {outcome:?}"
    );
    let failed = events
        .iter()
        .find(|event| {
            event.name == "loopctl.compaction" && event.field("outcome") == Some("failed")
        })
        .expect("the failed pass narrates itself");
    let observer_before = befores
        .first()
        .copied()
        .expect("the failed pass reached the compaction-failed observer");
    let before = failed
        .field("tokens_before")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("numeric before-figure");
    let after = failed
        .field("tokens_after")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("numeric after-figure");
    assert_eq!(
        before, observer_before,
        "the event's before-figure is the driver-measured one the \
         observer context carries, on the provider-token scale every \
         other outcome of this event family uses"
    );
    assert_eq!(
        after, observer_before,
        "a compactor error produced nothing, so the conversation is \
         unchanged — after equals before"
    );
}

#[tokio::test]
async fn a_pass_that_still_does_not_fit_narrates_its_measured_result() {
    let (events, befores, outcome) = compaction_run(
        Arc::new(OversizedCompactor),
        1_000,
        Some(loopctl::stream::Usage::new(900, 1)),
    )
    .await;
    assert!(
        matches!(
            outcome,
            Err(loopctl::error::LoopError::ContextExceeded { .. })
        ),
        "the scenario is the doesn't-fit arm: {outcome:?}"
    );
    let failed = events
        .iter()
        .find(|event| {
            event.name == "loopctl.compaction" && event.field("outcome") == Some("failed")
        })
        .expect("the doesn't-fit pass narrates itself as failed");
    let observer_before = befores
        .first()
        .copied()
        .expect("the doesn't-fit pass reached the compaction-failed observer");
    let before = failed
        .field("tokens_before")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("the event carries a numeric before-figure");
    let after = failed
        .field("tokens_after")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("the event carries a numeric after-figure");
    assert_eq!(
        before, observer_before,
        "the before-figure is the driver's own measurement — a pass that \
         measured never reports zeros"
    );
    assert!(
        after > observer_before,
        "the after-figure is the rejected result's measured size, not the \
         conversation's unchanged size: after {after}, before {observer_before}"
    );
}

#[tokio::test]
async fn a_deferred_threshold_pass_narrates_its_figures_on_one_scale() {
    let (events, befores, outcome) = compaction_run(
        Arc::new(FailingCompactor),
        1_150,
        Some(loopctl::stream::Usage::new(900, 1)),
    )
    .await;
    assert!(
        outcome.is_ok(),
        "the deferring run serves its turn and completes: {outcome:?}"
    );
    let deferred = events
        .iter()
        .find(|event| {
            event.name == "loopctl.compaction" && event.field("outcome") == Some("deferred")
        })
        .expect("the deferred pass narrates itself");
    let observer_before = befores
        .first()
        .copied()
        .expect("the deferred pass reached the compaction-failed observer");
    let before = deferred
        .field("tokens_before")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("numeric before-figure");
    let after = deferred
        .field("tokens_after")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("numeric after-figure");
    assert_eq!(
        before, observer_before,
        "the deferred event's figures ride the provider-token scale — \
         never the error's raw manager-scale pair"
    );
    assert_eq!(
        after, observer_before,
        "no compaction ran, so the conversation is unchanged"
    );
}

#[tokio::test]
async fn a_threshold_pass_narrates_itself_as_a_compaction_signal_event() {
    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let mut script = Vec::new();
    for step in 0..24 {
        script.push(MockResponse {
            text: format!("step {step} {}", "w".repeat(300)),
            tool_call: Some(MockToolCall {
                id: format!("call_{step}"),
                name: "echo".to_string(),
                input: serde_json::json!({ "step": step }),
            }),
            stop_reason: "tool_use".to_string(),
        });
    }
    script.push(MockResponse {
        text: "all done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    });
    let client = Arc::new(
        MockApiClient::new("test-model")
            .without_usage()
            .with_responses(script),
    );
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let config = SessionConfig::default()
        .with_context_window(1_600)
        .with_compact_threshold(80);
    let mut agent = BareLoop::new(client, registry, config);
    agent
        .run("keep echoing", &RunConfig::default())
        .await
        .expect("the compacting run completes");
    drop(guard);

    let events = captures.events.lock().expect("events lock").clone();
    let applied: Vec<&CapturedEvent> = events
        .iter()
        .filter(|event| {
            event.name == "loopctl.compaction"
                && event.field("outcome") == Some("applied")
                && event.field("reason") == Some("threshold")
        })
        .collect();
    assert!(
        !applied.is_empty(),
        "the threshold crossing narrates itself — no applied event fired"
    );
    for event in &applied {
        let before = event
            .field("tokens_before")
            .and_then(|figure| figure.parse::<u64>().ok())
            .expect("numeric before-figure");
        let after = event
            .field("tokens_after")
            .and_then(|figure| figure.parse::<u64>().ok())
            .expect("numeric after-figure");
        assert!(before > 0, "an applied pass measured its before");
        assert!(
            after < before,
            "an applied pass actually shrank the conversation: \
             after {after}, before {before}"
        );
    }
}

#[cfg(feature = "streaming")]
#[tokio::test]
async fn a_transport_fallback_turn_narrates_the_downgrade() {
    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
    use loopctl::managers::LoopManagers;
    use loopctl::stream::handler::{StreamHandler, StreamRetryConfig, StreamTimeoutConfig};
    use loopctl::stream::{StreamEvent, StreamStopReason, Usage};

    struct StreamBreaksClient;

    impl ApiClient for StreamBreaksClient {
        fn model(&self) -> String {
            "test-model".to_string()
        }

        fn set_model(&self, _model: &str) -> bool {
            true
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            Box::pin(futures::stream::once(async {
                Err(ApiError::api("stream transport broke"))
            }))
        }

        fn create_message(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(NonStreamingResponse {
                    message: Message::new(
                        loopctl::message::Role::Assistant,
                        vec![loopctl::message::MessagePart::text("fallback answer")],
                    ),
                    stop_reason: StreamStopReason::EndTurn,
                    usage: Some(Usage::new(4, 6)),
                })
            })
        }
    }

    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let handler = StreamHandler::new()
        .with_retry_config(StreamRetryConfig {
            max_retries: 1,
            base_delay_ms: 1,
            max_delay_ms: 1,
            jitter_factor: 0.0,
        })
        .with_timeout_config(StreamTimeoutConfig {
            fallback_to_non_streaming: true,
            ..StreamTimeoutConfig::default()
        });
    let managers = LoopManagers::new().with_stream_handler(handler);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(StreamBreaksClient),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );
    let result = agent
        .run("Hi", &RunConfig::default())
        .await
        .expect("the fallback serves the turn");
    drop(guard);

    assert_eq!(
        result.transport_fallback_count(),
        1,
        "the turn was actually served by the non-streaming recovery"
    );
    let event = captures
        .events
        .lock()
        .expect("events lock")
        .iter()
        .find(|event| event.name == "loopctl.transport_fallback")
        .expect("the downgrade narrates itself through the engine path")
        .clone();
    assert_eq!(
        event.field("turn"),
        Some("0"),
        "the event names the turn the recovery served"
    );
}

#[cfg(feature = "streaming")]
#[tokio::test]
async fn an_overflow_rejection_arms_a_salvage_signal_with_its_target() {
    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
    use loopctl::compact::{HeuristicTokenCounter, TokenCounter};

    struct HardWindowClient {
        inner: MockApiClient,
        window: u64,
    }

    impl HardWindowClient {
        fn oversized(&self, request: &StreamRequest) -> bool {
            let mut tokens = HeuristicTokenCounter.count(&request.messages);
            if let Some(system) = &request.system {
                tokens = tokens
                    .saturating_add(HeuristicTokenCounter.count(&[Message::user(system.clone())]));
            }
            tokens > self.window
        }
    }

    impl ApiClient for HardWindowClient {
        fn model(&self) -> String {
            self.inner.model()
        }

        fn set_model(&self, _model: &str) -> bool {
            false
        }

        fn stream_messages(
            &self,
            request: &StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<Item = Result<loopctl::stream::StreamEvent, ApiError>>
                    + Send
                    + 'static,
            >,
        > {
            if self.oversized(request) {
                return Box::pin(futures::stream::iter(vec![Err(ApiError::api(
                    "this model's maximum context length is small and the prompt is too long",
                ))]));
            }
            self.inner.stream_messages(request)
        }

        fn create_message(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            Box::pin(async {
                Err(ApiError::api(
                    "this model's maximum context length is small and the prompt is too long",
                ))
            })
        }
    }

    let captures = Arc::new(Captures::default());
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&captures)));
    let guard = tracing::subscriber::set_default(subscriber);
    let inner = MockApiClient::new("test-model")
        .without_usage()
        .with_responses(vec![
            MockResponse {
                text: String::new(),
                tool_call: Some(MockToolCall {
                    id: "c0".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({}),
                }),
                stop_reason: "tool_use".to_string(),
            },
            MockResponse {
                text: "retried turn lands".to_string(),
                tool_call: None,
                stop_reason: "end_turn".to_string(),
            },
        ]);
    let shrinker = Arc::new(TargetRecordingShrinker {
        targets: Mutex::new(Vec::new()),
    });
    let managers = loopctl::managers::LoopManagers::new().with_context_manager(Arc::new(
        loopctl::compact::ContextManager::new(shrinker.clone()),
    ));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let config = SessionConfig::default()
        .with_context_window(4_000)
        .with_compact_threshold(80);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(HardWindowClient { inner, window: 22 }),
        registry,
        config,
        managers,
    );
    let result = agent
        .run("a short opening prompt", &RunConfig::default())
        .await;
    drop(guard);
    assert!(
        result.is_ok(),
        "the rejected turn compacted, retried, and completed: {result:?}"
    );
    let targets = shrinker.targets.lock().expect("targets lock").clone();
    assert!(
        !targets.is_empty(),
        "the salvage forced a compaction pass — the pin had its chance"
    );
    let event = captures
        .events
        .lock()
        .expect("events lock")
        .iter()
        .find(|event| event.name == "loopctl.salvage")
        .expect("the armed retry narrates itself through the engine path")
        .clone();
    assert_eq!(
        event.field("kind"),
        Some("overflow"),
        "the provider-reported rejection is the overflow kind"
    );
    assert_eq!(
        event.field("turn"),
        Some("1"),
        "the event names the turn the rejection landed on"
    );
    let aimed_target = targets
        .first()
        .copied()
        .expect("the pass recorded its target");
    let narrated_target = event
        .field("target_tokens")
        .and_then(|figure| figure.parse::<u64>().ok())
        .expect("numeric target figure");
    assert_eq!(
        narrated_target, aimed_target,
        "the event's target is the one the engine aimed the compaction at"
    );
}
