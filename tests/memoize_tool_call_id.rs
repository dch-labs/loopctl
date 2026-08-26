//! `tool_call_id` authority contracts for memoized dispatch.
//!
//! Pins two invariants: a memoize cache hit carries the *requesting*
//! call's id (not the cached first call's), and the engine stamps ids
//! authoritatively after the pipeline (a middleware returning a bogus
//! non-empty id cannot corrupt `tool_use`↔`tool_result` pairing). Recreated
//! 2026-08-26 per the L-74 re-assessment — the original proof-of-gap
//! test was lost with an unmerged branch.
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

use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::message::{Message, MessagePart};
use loopctl::middleware::{
    MemoizingMiddleware, NoopPathExtractor, ToolDispatchContext, ToolDispatchResult,
    ToolMiddleware, ToolPipeline,
};
use loopctl::observer::{LoopObserver, ToolPostContext, ToolPreContext};
use loopctl::reflection::RecoveryAction;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// A tool that counts its executions and echoes the input.
struct CountingTool {
    /// One entry per actual execution (cache hits must not add entries),
    /// shared with the test by handle.
    executions: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl Tool for CountingTool {
    fn name(&self) -> &'static str {
        "count"
    }
    fn description(&self) -> &'static str {
        "Counts executions"
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
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move {
            self.executions
                .lock()
                .expect("execution log lock")
                .push(input);
            Ok(ToolOutput::text("done"))
        })
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

/// The `tool_call_id`s of every tool-result part, in conversation order.
fn result_ids(conversation: &[Message]) -> Vec<String> {
    conversation
        .iter()
        .flat_map(|m| m.parts.iter())
        .filter_map(|p| match p {
            MessagePart::ToolResult { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect()
}

fn memoized_loop(tool: CountingTool) -> BareLoop<MockApiClient> {
    let responses = vec![
        turn(
            "first",
            MockToolCall {
                id: "call_a".to_string(),
                name: "count".to_string(),
                input: serde_json::json!({"n": 1}),
            },
        ),
        turn(
            "second",
            MockToolCall {
                id: "call_b".to_string(),
                name: "count".to_string(),
                input: serde_json::json!({"n": 1}),
            },
        ),
        vec![terminal()],
    ]
    .into_iter()
    .flatten()
    .collect();
    let client = MockApiClient::new("test-model").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(tool);
    let mut loop_ = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    loop_
        .set_pipeline(
            ToolPipeline::builder().with_middleware(MemoizingMiddleware::new(
                vec!["count".to_string()],
                Vec::new(),
                Arc::new(NoopPathExtractor),
                5,
            )),
        )
        .expect("static pipeline composition is valid");
    loop_
}

#[tokio::test]
async fn cache_hit_carries_the_requesting_call_id() {
    let executions = Arc::new(Mutex::new(Vec::new()));
    let tool = CountingTool {
        executions: Arc::clone(&executions),
    };
    let mut loop_ = memoized_loop(tool);

    loop_
        .run("run both turns", &RunConfig::default())
        .await
        .expect("run completes");

    assert_eq!(
        result_ids(&loop_.conversation()),
        vec!["call_a".to_string(), "call_b".to_string()],
        "each tool result must answer the call that requested it — the \
         cache hit must carry call_b, not the cached call_a"
    );
    assert_eq!(
        executions.lock().expect("execution log lock").len(),
        1,
        "the identical second call must be served from the cache"
    );
}

/// A middleware that returns a deliberately wrong non-empty
/// `tool_call_id` for every call — the engine must override it.
struct BogusIdMiddleware;

impl loopctl::middleware::ToolMiddleware for BogusIdMiddleware {
    fn name(&self) -> &'static str {
        "bogus_id"
    }
    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut loopctl::middleware::ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = loopctl::middleware::ToolDispatchResult> + Send + 'a>> {
        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;
            result.tool_call_id = "middleware_made_this_up".to_string();
            result
        })
    }
}

#[tokio::test]
async fn middleware_returned_ids_are_overridden_by_the_engine() {
    let executions = Arc::new(Mutex::new(Vec::new()));
    let tool = CountingTool {
        executions: Arc::clone(&executions),
    };
    let responses = vec![
        turn(
            "go",
            MockToolCall {
                id: "call_real".to_string(),
                name: "count".to_string(),
                input: serde_json::json!({"n": 7}),
            },
        ),
        vec![terminal()],
    ]
    .into_iter()
    .flatten()
    .collect();
    let client = MockApiClient::new("test-model").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(tool);
    let mut loop_ = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    loop_
        .set_pipeline(ToolPipeline::builder().with_middleware(BogusIdMiddleware))
        .expect("static pipeline composition is valid");

    loop_
        .run("one turn", &RunConfig::default())
        .await
        .expect("run completes");

    assert_eq!(
        result_ids(&loop_.conversation()),
        vec!["call_real".to_string()],
        "the engine stamps the model-issued call id after the pipeline — \
         a middleware cannot corrupt tool_use↔tool_result pairing"
    );
}

/// A tool whose first execution fails and whose retry succeeds —
/// drives the recovery loop through one full retry sequence.
struct FailOnceTool {
    /// Execution count.
    calls: Mutex<u32>,
}

impl Tool for FailOnceTool {
    fn name(&self) -> &'static str {
        "flaky"
    }
    fn description(&self) -> &'static str {
        "Fails once"
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
        Box::pin(async move {
            let mut n = self.calls.lock().expect("lock");
            *n += 1;
            if *n == 1 {
                Err(ToolError::Execution("first attempt fails".to_string()))
            } else {
                Ok(ToolOutput::text("recovered"))
            }
        })
    }
}

/// Always retry, no delay.
struct AlwaysRetry;

impl loopctl::reflection::RecoveryStrategy for AlwaysRetry {
    fn decide(
        &self,
        _analysis: &loopctl::reflection::FailureAnalysis,
        _attempt: u32,
        _max_attempts: u32,
    ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
        Box::pin(async {
            RecoveryAction::Retry {
                delay: std::time::Duration::ZERO,
            }
        })
    }
}

/// Records (phase, tool_call_id) in arrival order.
#[derive(Default)]
struct PairingRecorder {
    events: Mutex<Vec<(&'static str, String)>>,
}

impl LoopObserver for PairingRecorder {
    fn name(&self) -> &'static str {
        "pairing_recorder"
    }
    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        self.events
            .lock()
            .expect("lock")
            .push(("pre", ctx.tool_call_id.clone()));
    }
    fn on_tool_post(&self, ctx: &ToolPostContext) {
        self.events
            .lock()
            .expect("lock")
            .push(("post", ctx.tool_call_id.clone()));
    }
}

/// A middleware that rewrites every id to garbage.
struct GarbageIdMiddleware;

impl ToolMiddleware for GarbageIdMiddleware {
    fn name(&self) -> &'static str {
        "garbage_id"
    }
    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;
            result.tool_call_id = format!("garbage-{}", result.tool_call_id);
            result
        })
    }
}

#[tokio::test]
async fn retries_pair_and_ids_survive_stacked_rewriters() {
    let recorder = Arc::new(PairingRecorder::default());
    let responses = vec![
        turn(
            "go",
            MockToolCall {
                id: "call_retry_me".to_string(),
                name: "flaky".to_string(),
                input: serde_json::json!({}),
            },
        ),
        vec![terminal()],
    ]
    .into_iter()
    .flatten()
    .collect();
    let client = MockApiClient::new("m").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(FailOnceTool {
        calls: Mutex::new(0),
    });
    let mut loop_ = BareLoop::new(Arc::new(client), registry, SessionConfig::default())
        .with_observer(Arc::clone(&recorder) as Arc<dyn LoopObserver>);
    loop_.set_recovery_strategy(Arc::new(AlwaysRetry));
    let _ = &recorder;
    loop_
        .set_pipeline(
            ToolPipeline::builder()
                .with_middleware(GarbageIdMiddleware)
                .with_middleware(MemoizingMiddleware::new(
                    vec!["flaky".to_string()],
                    Vec::new(),
                    Arc::new(NoopPathExtractor),
                    5,
                )),
        )
        .expect("pipeline");

    loop_
        .run("q", &RunConfig::default())
        .await
        .expect("run completes");

    assert_eq!(
        result_ids(&loop_.conversation()),
        vec!["call_retry_me".to_string()],
        "a retried call yields exactly one tool result, under the \
         model-issued id"
    );

    let events = recorder.events.lock().expect("lock").clone();
    assert_eq!(
        events,
        vec![
            ("pre", "call_retry_me".to_string()),
            ("post", "call_retry_me".to_string()),
            ("pre", "call_retry_me".to_string()),
            ("post", "call_retry_me".to_string()),
        ],
        "retry attempts strictly interleave pre/post under one id — \
         FIFO pairing is exact"
    );
}
