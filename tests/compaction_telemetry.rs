//! Compaction-telemetry contracts for engine-driven passes.
//!
//! Pins: a compacting pass fires the pre-compaction observer event and
//! the widened pass-completed event exactly once each, with the post
//! event carrying populated telemetry; the pre event carries the pass's
//! reason, turn, pre-pass estimate, window, message count, and session
//! id; the post event's `evicted_messages` matches what the demotion
//! sink was handed on the same run; the reported duration covers the
//! compactor's own call time; a cancelled or no-action pass fires
//! the pre event without the post; a hook-vetoed pass likewise; and
//! the full chain orders pre-observer, demotion, post-observer, and
//! post-compact hook.
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

use loopctl::compact::demote::{DemotionContext, DemotionSink};
use loopctl::compact::types::{CompactionContext, CompactionOutcome};
use loopctl::compact::{CompactReason, ContextCompactor, ContextManager, TruncatingCompactor};
use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::error::LoopError;
use loopctl::message::Message;
use loopctl::observer::{CompactedContext, LoopObserver, PreCompactionContext};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// A tool whose result grows the history — the compaction-forcing knob.
///
/// Each call returns the same mid-sized payload, so a scripted run of
/// tool turns pushes the conversation over the tiny window every few
/// turns while staying small enough for the truncator to recover from.
struct GrowTool;

impl Tool for GrowTool {
    fn name(&self) -> &'static str {
        "grow"
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
        Box::pin(async { Ok(ToolOutput::text("payload ".repeat(40))) })
    }
}

/// Records every pre- and post-compaction event the loop fires.
///
/// The observer half of every pin here: clones each context it is
/// handed into one of two logs, so a test can assert counts, field
/// values, and the start-without-completion shapes after the run.
struct RecordingObserver {
    pre: Arc<Mutex<Vec<PreCompactionContext>>>,
    post: Arc<Mutex<Vec<CompactedContext>>>,
}

impl LoopObserver for RecordingObserver {
    fn name(&self) -> &str {
        "recording"
    }

    fn on_pre_compaction(&self, ctx: &PreCompactionContext) {
        self.pre.lock().expect("pre lock").push(ctx.clone());
    }

    fn on_compaction(&self, ctx: &CompactedContext) {
        self.post.lock().expect("post lock").push(ctx.clone());
    }
}

/// Records the size of every demotion delivery.
///
/// The sink half of the eviction-count pin: the post event's
/// `evicted_messages` must equal one of these sizes, pairing what the
/// window lost with what memory received on the same run.
struct CountingSink {
    sizes: Arc<Mutex<Vec<usize>>>,
}

impl DemotionSink for CountingSink {
    fn demote<'a>(
        &'a self,
        evicted: &'a [Message],
        _meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        let sizes = Arc::clone(&self.sizes);
        let count = evicted.len();
        Box::pin(async move {
            sizes.lock().expect("sizes lock").push(count);
            Ok(())
        })
    }
}

/// Counts entries and never returns — the in-flight pass the
/// cancellation has to beat.
struct HangingCompactor {
    entered: Arc<Mutex<usize>>,
}

impl ContextCompactor for HangingCompactor {
    fn compact(
        &self,
        _messages: Vec<Message>,
        _target_tokens: u64,
        _context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        let entered = Arc::clone(&self.entered);
        Box::pin(async move {
            *entered.lock().expect("entered lock") += 1;
            std::future::pending().await
        })
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
                name: "grow".to_string(),
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
/// starts from.
fn telemetry_loop(
    compactor: Arc<dyn ContextCompactor>,
    sink: Option<Arc<dyn DemotionSink>>,
    observer: Arc<dyn LoopObserver>,
) -> BareLoop<MockApiClient> {
    let mut registry = ToolRegistry::new();
    registry.register(GrowTool);
    let config = SessionConfig::default()
        .with_context_window(400)
        .with_compact_threshold(50);
    let mut loop_ = BareLoop::new(
        Arc::new(MockApiClient::new("m").with_responses(growing_script())),
        registry,
        config,
    );
    loop_.set_context_manager(Arc::new(
        ContextManager::new(compactor).with_context_window(400),
    ));
    if let Some(sink) = sink {
        loop_.set_demotion_sink(sink);
    }
    loop_.register_observer(observer);
    loop_
}

/// The truncating compactor every compacting test uses, with knobs
/// below the conversation size the tiny window compacts at.
fn truncating() -> Arc<dyn ContextCompactor> {
    Arc::new(
        TruncatingCompactor::new()
            .with_min_messages(2)
            .with_preserve_recent(2),
    )
}

/// The observer and its two logs — what every test's `recorder()` hands back.
///
/// The tuple shape every pin unpacks: the observer to register, the
/// pre-compaction log, and the pass-completed log, so tests assert on
/// exactly what the loop fired without each wiring its own pair of
/// mutexes.
type Recorder = (
    Arc<RecordingObserver>,
    Arc<Mutex<Vec<PreCompactionContext>>>,
    Arc<Mutex<Vec<CompactedContext>>>,
);

fn recorder() -> Recorder {
    let pre = Arc::new(Mutex::new(Vec::new()));
    let post = Arc::new(Mutex::new(Vec::new()));
    let observer = Arc::new(RecordingObserver {
        pre: Arc::clone(&pre),
        post: Arc::clone(&post),
    });
    (observer, pre, post)
}

#[tokio::test]
async fn both_events_fire_once_per_compacting_pass() {
    let (observer, pre, post) = recorder();
    let mut loop_ = telemetry_loop(truncating(), None, observer);

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let pre_events = pre.lock().expect("pre lock").clone();
    let post_events = post.lock().expect("post lock").clone();
    assert_eq!(
        pre_events.len(),
        post_events.len(),
        "every compacting pass pairs one pre event with one post event"
    );
    assert!(
        !post_events.is_empty(),
        "the growing script must force at least one compacting pass"
    );
    let report = &post_events[0];
    assert!(
        report.tokens_before > 0,
        "the post event carries a real pre-pass estimate"
    );
    assert!(
        report.telemetry.compression_ratio.is_finite(),
        "the post event's telemetry is populated: {}",
        report.telemetry.compression_ratio
    );
    assert!(
        report.telemetry.duration.as_nanos() > 0,
        "the telemetry's duration is measured outside the hooks feature"
    );
}

#[tokio::test]
async fn the_pre_event_carries_the_pre_state() {
    let (observer, pre, post) = recorder();
    let mut loop_ = telemetry_loop(truncating(), None, observer);

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let pre_events = pre.lock().expect("pre lock").clone();
    let post_events = post.lock().expect("post lock").clone();
    let first = &pre_events[0];
    let report = &post_events[0];
    assert_eq!(
        first.reason,
        CompactReason::ThresholdExceeded,
        "the pre event names the governing reason"
    );
    assert!(
        pre_events
            .iter()
            .all(|event| event.session_id == first.session_id),
        "the pre event carries the session's stable id, identical across passes"
    );
    assert_eq!(
        first.context_window, 400,
        "the pre event carries the manager's window"
    );
    assert_eq!(
        first.tokens_before, report.tokens_before,
        "the pre estimate matches the post event's before figure"
    );
    assert!(
        first.message_count > 0 && first.turn >= 1,
        "the pre event carries the pass's real message count and turn: {} messages, turn {}",
        first.message_count,
        first.turn
    );
}

#[tokio::test]
async fn the_post_event_carries_reason_and_evicted_count() {
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let (observer, _pre, post) = recorder();
    let mut loop_ = telemetry_loop(
        truncating(),
        Some(Arc::new(CountingSink {
            sizes: Arc::clone(&sizes),
        })),
        observer,
    );

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let post_events = post.lock().expect("post lock").clone();
    let deliveries = sizes.lock().expect("sizes lock").clone();
    assert_eq!(
        post_events.len(),
        deliveries.len(),
        "every reported pass made exactly one demotion delivery"
    );
    for (report, delivered) in post_events.iter().zip(&deliveries) {
        assert_eq!(
            report.reason,
            CompactReason::ThresholdExceeded,
            "the post event names the governing reason"
        );
        assert_eq!(
            report.evicted_messages, *delivered,
            "the post event's evicted count is the sink-observed delivery size"
        );
    }
}

/// Wraps the truncating compactor with a fixed delay, so a pass's
/// reported duration can be checked against time the compactor itself
/// spent.
struct SlowCompactor {
    inner: TruncatingCompactor,
}

impl ContextCompactor for SlowCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.inner.compact(messages, target_tokens, context).await
        })
    }
}

#[tokio::test]
async fn the_reported_duration_covers_the_compactors_call_time() {
    let (observer, _pre, post) = recorder();
    let mut loop_ = telemetry_loop(
        Arc::new(SlowCompactor {
            inner: TruncatingCompactor::new()
                .with_min_messages(2)
                .with_preserve_recent(2),
        }),
        None,
        observer,
    );

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let post_events = post.lock().expect("post lock").clone();
    assert!(
        !post_events.is_empty(),
        "the growing script must force at least one compacting pass"
    );
    for report in &post_events {
        assert!(
            report.telemetry.duration >= std::time::Duration::from_millis(50),
            "the reported duration is the engine pass span, so it covers the \
             compactor's own 50 ms call time, got {:?}",
            report.telemetry.duration
        );
    }
}

#[tokio::test]
async fn a_cancelled_pass_fires_pre_without_post() {
    let entered = Arc::new(Mutex::new(0usize));
    let (observer, pre, post) = recorder();
    let mut loop_ = telemetry_loop(
        Arc::new(HangingCompactor {
            entered: Arc::clone(&entered),
        }),
        None,
        observer,
    );

    let cancel_signal = loop_.cancel_signal();
    let run = tokio::spawn(async move { loop_.run("grow then hang", &RunConfig::default()).await });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while *entered.lock().expect("entered lock") == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the run never started its compaction pass — the pre-without-post \
             contract cannot be observed"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    cancel_signal.cancel();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), run)
        .await
        .expect("the cancellation ends the run rather than hanging")
        .expect("spawned run task finished");
    assert!(
        matches!(outcome, Err(LoopError::Cancelled)),
        "the mid-compaction cancel surfaces typed: {outcome:?}"
    );

    let pre_events = pre.lock().expect("pre lock").clone();
    let post_events = post.lock().expect("post lock").clone();
    assert_eq!(
        pre_events.len(),
        1,
        "the cancelled pass fired its pre event before hanging"
    );
    assert!(
        post_events.is_empty(),
        "a cancelled pass never fires the post event"
    );
}

#[tokio::test]
async fn a_no_action_pass_fires_pre_without_post() {
    let (observer, pre, post) = recorder();
    // A truncator whose minimum sits above the whole conversation: the
    // manager classifies every pass as no action, and the run ends by
    // max-turns rather than compaction.
    let stall = Arc::new(TruncatingCompactor::new().with_min_messages(100));
    let mut loop_ = telemetry_loop(stall, None, observer);
    let bounded = RunConfig::default().with_max_turns(3);

    let _outcome = loop_.run("grow but never compact", &bounded).await;

    let pre_events = pre.lock().expect("pre lock").clone();
    let post_events = post.lock().expect("post lock").clone();
    assert!(
        !pre_events.is_empty(),
        "a no-action pass still announces its start"
    );
    assert!(
        post_events.is_empty(),
        "a no-action pass never fires the post event"
    );
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn a_vetoed_pass_fires_pre_without_post() {
    use loopctl::hooks::context::{CompactResult, PreCompactContext as HookPreCompactContext};
    use loopctl::hooks::{Hook, HookExecutor};

    struct VetoHook;
    impl Hook for VetoHook {
        fn name(&self) -> &str {
            "veto"
        }
        fn on_pre_compact(&self, _ctx: &HookPreCompactContext) -> Option<CompactResult> {
            Some(CompactResult::abort("not now"))
        }
    }

    let (observer, pre, post) = recorder();
    let mut loop_ = telemetry_loop(truncating(), None, observer);
    let mut executor = HookExecutor::new();
    executor.register(Arc::new(VetoHook));
    loop_.set_hook_executor(Arc::new(executor));
    let bounded = RunConfig::default().with_max_turns(3);

    let _outcome = loop_.run("grow but vetoed", &bounded).await;

    let pre_events = pre.lock().expect("pre lock").clone();
    let post_events = post.lock().expect("post lock").clone();
    assert!(
        !pre_events.is_empty(),
        "a vetoed pass still announces its start to observers"
    );
    assert!(
        post_events.is_empty(),
        "a vetoed pass never fires the post event"
    );
}

#[cfg(feature = "hooks")]
#[tokio::test]
async fn the_chain_orders_pre_demote_post_and_hook() {
    use loopctl::hooks::context::PostCompactContext;
    use loopctl::hooks::{Hook, HookExecutor};

    struct ChainSink {
        log: Arc<Mutex<Vec<&'static str>>>,
    }
    impl DemotionSink for ChainSink {
        fn demote<'a>(
            &'a self,
            _evicted: &'a [Message],
            _meta: DemotionContext,
        ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
            let log = Arc::clone(&self.log);
            Box::pin(async move {
                log.lock().expect("chain log lock").push("demote");
                Ok(())
            })
        }
    }

    struct ChainObserver {
        log: Arc<Mutex<Vec<&'static str>>>,
    }
    impl LoopObserver for ChainObserver {
        fn name(&self) -> &str {
            "chain"
        }
        fn on_pre_compaction(&self, _ctx: &PreCompactionContext) {
            self.log
                .lock()
                .expect("chain log lock")
                .push("on_pre_compaction");
        }
        fn on_compaction(&self, _ctx: &CompactedContext) {
            self.log
                .lock()
                .expect("chain log lock")
                .push("on_compaction");
        }
    }

    struct ChainHook {
        log: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Hook for ChainHook {
        fn name(&self) -> &str {
            "chain"
        }
        fn on_post_compact(&self, _ctx: &PostCompactContext) {
            self.log
                .lock()
                .expect("chain log lock")
                .push("post_compact_hook");
        }
    }

    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = telemetry_loop(
        truncating(),
        Some(Arc::new(ChainSink {
            log: Arc::clone(&log),
        })),
        Arc::new(ChainObserver {
            log: Arc::clone(&log),
        }),
    );
    let mut executor = HookExecutor::new();
    executor.register(Arc::new(ChainHook {
        log: Arc::clone(&log),
    }));
    loop_.set_hook_executor(Arc::new(executor));

    loop_
        .run("grow until compact", &RunConfig::default())
        .await
        .expect("run completes");

    let events = log.lock().expect("chain log lock").clone();
    let markers = [
        "on_pre_compaction",
        "demote",
        "on_compaction",
        "post_compact_hook",
    ];
    let positions: Vec<usize> =
        markers
            .iter()
            .map(|marker| {
                events.iter().position(|event| event == marker).unwrap_or_else(|| {
                panic!("the pass never fired {marker} — ordering is moot without it: {events:?}")
            })
            })
            .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "one compacting pass orders pre-observer, demotion, post-observer, post-compact hook: {events:?}"
    );
}
