//! Detection false positives: progressing runs must survive the detectors.
//!
//! Run: `cargo test --all-features --test detection_false_positives -- --nocapture`

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use futures::Stream;
use loopctl::api::error::ApiError;
use loopctl::api::{ApiClient, StreamRequest};
use loopctl::config::SessionConfig;
use loopctl::detection::{ConvergenceAction, DetectionConfig, DetectionManager};
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::managers::LoopManagers;
use loopctl::message::{MessagePart, ToolContent, ToolContentPart};
use loopctl::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamEvent, Usage,
};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use serde_json::Value;

/// One scripted model turn.
enum Step {
    /// A tool-call turn with the given preamble text (may be empty).
    Preamble(String),
    /// A terminal text-only turn.
    Text(String),
}

/// A client replaying one [`Step`] per request and recording turn count.
struct ScriptedClient {
    script: Mutex<Vec<Step>>,
    turns: Mutex<usize>,
}

impl ScriptedClient {
    fn new(script: Vec<Step>) -> Self {
        Self {
            script: Mutex::new(script),
            turns: Mutex::new(0),
        }
    }

    fn turns(&self) -> usize {
        *self.turns.lock().unwrap()
    }
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        "test-model".to_string()
    }

    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }

    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<loopctl::api::NonStreamingResponse, ApiError>> + Send + '_>,
    > {
        Box::pin(async { Err(ApiError::api("these tests drive the streaming path")) })
    }

    fn stream_messages_with_options(
        &self,
        _request: &StreamRequest,
        _options: loopctl::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        let step = self.script.lock().unwrap().remove(0);
        *self.turns.lock().unwrap() += 1;
        let events = match step {
            Step::Preamble(text) => {
                let mut events: Vec<Result<StreamEvent, ApiError>> =
                    vec![Ok(StreamEvent::MessageStart(MessageStart {
                        message: MessageMetadata {
                            id: "msg_1".into(),
                            role: "assistant".into(),
                            model: "test-model".into(),
                        },
                    }))];
                if !text.is_empty() {
                    events.push(Ok(StreamEvent::PartStart(PartStart {
                        index: 0,
                        part: Some(MessagePart::text("")),
                    })));
                    events.push(Ok(StreamEvent::IndexedDelta(IndexedDelta {
                        index: 0,
                        delta: DeltaPart::Text { text },
                    })));
                    events.push(Ok(StreamEvent::PartStop { index: Some(0) }));
                }
                events.push(Ok(StreamEvent::PartStart(PartStart {
                    index: 1,
                    part: Some(MessagePart::tool_call("call_1", "search", Value::Null)),
                })));
                events.push(Ok(StreamEvent::IndexedDelta(IndexedDelta {
                    index: 1,
                    delta: DeltaPart::InputJson {
                        partial_json: "{}".into(),
                    },
                })));
                events.push(Ok(StreamEvent::PartStop { index: Some(1) }));
                events.push(Ok(StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("tool_call".into()),
                    },
                    usage: Some(Usage::new(1, 1)),
                })));
                events.push(Ok(StreamEvent::MessageStop));
                events
            }
            Step::Text(text) => vec![
                Ok(StreamEvent::MessageStart(MessageStart {
                    message: MessageMetadata {
                        id: "msg_1".into(),
                        role: "assistant".into(),
                        model: "test-model".into(),
                    },
                })),
                Ok(StreamEvent::PartStart(PartStart {
                    index: 0,
                    part: Some(MessagePart::text("")),
                })),
                Ok(StreamEvent::IndexedDelta(IndexedDelta {
                    index: 0,
                    delta: DeltaPart::Text { text },
                })),
                Ok(StreamEvent::PartStop { index: Some(0) }),
                Ok(StreamEvent::MessageDelta(MessageDelta {
                    delta: MessageDeltaPayload {
                        stop_reason: Some("end_turn".into()),
                    },
                    usage: Some(Usage::new(1, 1)),
                })),
                Ok(StreamEvent::MessageStop),
            ],
        };
        Box::pin(futures::stream::iter(events))
    }
}

/// A tool returning a multipart result whose text changes per call.
struct ChangingMultipartTool {
    calls: Mutex<usize>,
}

impl Tool for ChangingMultipartTool {
    fn name(&self) -> &'static str {
        "search"
    }

    fn description(&self) -> &'static str {
        "Returns a changing multipart result"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "search".into(),
            description: "Returns a changing multipart result".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let calls = &self.calls;
        Box::pin(async move {
            let n = {
                let mut guard = calls.lock().unwrap();
                *guard += 1;
                *guard
            };
            Ok(ToolOutput::success(ToolContent::Multipart(vec![
                ToolContentPart::Text {
                    text: format!("result batch {n}"),
                },
            ])))
        })
    }
}

/// A tool returning an identical multipart result every call.
struct StuckMultipartTool;

impl Tool for StuckMultipartTool {
    fn name(&self) -> &'static str {
        "search"
    }

    fn description(&self) -> &'static str {
        "Returns the same multipart result every call"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "search".into(),
            description: "Returns the same multipart result every call".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move {
            Ok(ToolOutput::success(ToolContent::Multipart(vec![
                ToolContentPart::Text {
                    text: "same result every time".into(),
                },
            ])))
        })
    }
}

/// A tool returning a changing plain-text result.
struct ChangingTextTool {
    calls: Mutex<usize>,
}

impl Tool for ChangingTextTool {
    fn name(&self) -> &'static str {
        "search"
    }

    fn description(&self) -> &'static str {
        "Returns a changing text result"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "search".into(),
            description: "Returns a changing text result".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let calls = &self.calls;
        Box::pin(async move {
            let n = {
                let mut guard = calls.lock().unwrap();
                *guard += 1;
                *guard
            };
            Ok(ToolOutput::text(format!("result {n}")))
        })
    }
}

fn detection_manager() -> DetectionManager {
    DetectionManager::new_with_config(DetectionConfig {
        loop_threshold: 3,
        stop_threshold: 10,
        ..DetectionConfig::default()
    })
    .unwrap()
}

fn make_agent(
    client: ScriptedClient,
    manager: DetectionManager,
    registry: ToolRegistry,
) -> (BareLoop<ScriptedClient>, std::sync::Arc<ScriptedClient>) {
    let client = std::sync::Arc::new(client);
    let managers = LoopManagers::new().with_detection(manager);
    let agent = BareLoop::new_with_managers(
        std::sync::Arc::clone(&client),
        registry,
        SessionConfig::default(),
        managers,
    );
    (agent, client)
}

fn registry_with<T: Tool + 'static>(tool: T) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(tool);
    registry
}

fn tool_script(turns: usize) -> Vec<Step> {
    let mut script = Vec::new();
    for _ in 0..turns {
        script.push(Step::Preamble("Let me search for that.".into()));
    }
    script.push(Step::Text("done".into()));
    script
}

#[tokio::test]
async fn multipart_progress_never_trips_the_loop_detector() {
    let client = ScriptedClient::new(tool_script(9));
    let (mut agent, client) = make_agent(
        client,
        detection_manager(),
        registry_with(ChangingMultipartTool {
            calls: Mutex::new(0),
        }),
    );

    let result = agent.run("q", &RunConfig::default()).await;
    assert!(
        result.is_ok(),
        "changing multipart output is progress — 9 calls under a stop threshold of 10 must complete: {result:?}"
    );
    assert!(
        client.turns() >= 10,
        "all 9 tool turns plus the final answer must run, got {}",
        client.turns()
    );
}

#[tokio::test]
async fn progressing_tool_work_is_never_convergence_killed() {
    let client = ScriptedClient::new(tool_script(5));
    let (mut agent, client) = make_agent(
        client,
        detection_manager(),
        registry_with(ChangingTextTool {
            calls: Mutex::new(0),
        }),
    );

    let result = agent.run("q", &RunConfig::default()).await;
    assert!(
        result.is_ok(),
        "identical preambles over progressing tool work must not converge-kill the run: {result:?}"
    );
    assert!(
        client.turns() >= 6,
        "all 5 tool turns plus the final answer must run, got {}",
        client.turns()
    );
}

#[tokio::test]
async fn changing_results_survive_the_loop_detector() {
    let client = ScriptedClient::new(tool_script(9));
    let (mut agent, _client) = make_agent(
        client,
        detection_manager(),
        registry_with(ChangingTextTool {
            calls: Mutex::new(0),
        }),
    );

    let result = agent.run("q", &RunConfig::default()).await;
    assert!(
        result.is_ok(),
        "changing text results are progress and must survive the loop detector: {result:?}"
    );
}

#[tokio::test]
async fn identical_multipart_repeats_still_stop() {
    // Single-record counting means the stop fires on the call after the
    // threshold is reached, not halfway to it.
    let client = ScriptedClient::new(tool_script(11));
    let (mut agent, _client) = make_agent(
        client,
        detection_manager(),
        registry_with(StuckMultipartTool),
    );

    let result = agent.run("q", &RunConfig::default()).await;
    assert!(
        result.is_err(),
        "a genuinely stuck multipart tool (identical input and output) must still be stopped"
    );
}

#[tokio::test]
async fn default_convergence_action_is_warn() {
    // A terminal response ends its run, so convergence sees repeated final
    // answers across runs: three identical terminal runs satisfy the
    // default window (3 consecutive similar at 0.95). The default action
    // warns — every run completes.
    let script = (0..3)
        .map(|_| Step::Text("the answer is forty two".into()))
        .collect::<Vec<_>>();
    let client = ScriptedClient::new(script);
    let (mut agent, _client) = make_agent(client, detection_manager(), ToolRegistry::new());

    for run in 0..3 {
        let result = agent.run("q", &RunConfig::default()).await;
        assert!(
            result.is_ok(),
            "run {run}: default convergence warns but must not end the run: {result:?}"
        );
    }
}

#[tokio::test]
async fn stop_opt_in_still_ends_the_run() {
    // Opting back into Stop restores the old behavior: the third identical
    // terminal run errs instead of completing once the window is satisfied.
    let script = (0..3)
        .map(|_| Step::Text("the answer is forty two".into()))
        .collect::<Vec<_>>();
    let client = ScriptedClient::new(script);
    let manager = DetectionManager::new_with_config(DetectionConfig {
        loop_threshold: 3,
        stop_threshold: 10,
        on_converge: ConvergenceAction::Stop,
        ..DetectionConfig::default()
    })
    .unwrap();
    let (mut agent, _client) = make_agent(client, manager, ToolRegistry::new());

    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    let third = agent.run("q", &RunConfig::default()).await;
    assert!(
        third.is_err(),
        "an explicit Stop opt-in must end the third converged run: {third:?}"
    );
}
