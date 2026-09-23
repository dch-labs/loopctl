//! Default-construction profile pins: the machinery a default-built
//! loop carries and the profiles that own it.
//!
//! Pins: a default-built loop verifies deny-class writes (the builtin
//! `CommandVerifier`), memoizes repeat reads, caps oversized tool output,
//! and re-injects the goal from turn five; a client that declares
//! tool-constraint support gets `Strict` request options by default while
//! a plain client keeps `None`; the explicit profiles own their seams
//! (the constrained apply replaces the reminder, the frontier opt-out
//! removes it, a bundle-carried pipeline survives).
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

use loopctl::api::ApiClient;
use loopctl::engine::{BareLoop, Loop, RunConfig};
use loopctl::message::{Message, MessagePart, Role, ToolContent};
use loopctl::structured::RequestOptions;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// A `Read`-named tool that counts executions and returns a fixed payload.
///
/// The counter is the memoize oracle: a cache hit serves the second call without touching it.
struct CountingRead {
    executions: Arc<Mutex<usize>>,
    payload: String,
}

impl Tool for CountingRead {
    fn name(&self) -> &'static str {
        "Read"
    }
    fn description(&self) -> &'static str {
        "Returns the fixed payload"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name().to_string(),
            self.description().to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        )
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let executions = Arc::clone(&self.executions);
        let payload = self.payload.clone();
        Box::pin(async move {
            *executions.lock().expect("read lock") += 1;
            Ok(ToolOutput::text(payload))
        })
    }
}

/// A `Grep`-named tool that counts executions and returns a small
/// payload — the memoize arm without the cap's truncation marker in
/// the way.
struct CountingGrep {
    executions: Arc<Mutex<usize>>,
}

impl Tool for CountingGrep {
    fn name(&self) -> &'static str {
        "Grep"
    }
    fn description(&self) -> &'static str {
        "Returns a small payload"
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
        let executions = Arc::clone(&self.executions);
        Box::pin(async move {
            *executions.lock().expect("grep lock") += 1;
            Ok(ToolOutput::text("found it"))
        })
    }
}

/// A `Bash`-named tool that always succeeds — the verifier's deny set is
/// the behavior under test, not the tool.
struct OkBash;

impl Tool for OkBash {
    fn name(&self) -> &'static str {
        "Bash"
    }
    fn description(&self) -> &'static str {
        "Runs a command"
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
        Box::pin(async { Ok(ToolOutput::text("ran")) })
    }
}

/// One scripted model turn that calls a tool, by call id and input.
///
/// The response carries the call with a `tool_use` stop reason, so the
/// engine dispatches the named tool on the next turn of the script.
fn tool_call_response(id: &str, name: &str, input: serde_json::Value) -> MockResponse {
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

/// The scripted terminal turn that ends a run.
///
/// Plain text with an `end_turn` stop reason — the engine completes the
/// run after serving it, so scripts finish deterministically.
fn done_response() -> MockResponse {
    MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

/// The standard registry for these pins: both counting fixtures plus
/// the shell stand-in.
///
/// Every tool-dispatching test registers through this so the fixtures'
/// counters and the verifier's shell-class matching see one consistent
/// tool surface.
fn registry_with(read: CountingRead, grep: CountingGrep) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(read);
    registry.register(grep);
    registry.register(OkBash);
    registry
}

/// Flatten every tool-result message's text parts, oldest first.
///
/// The dispatch-path oracle: verify blocks, cache markers, and
/// truncation all ride tool results, so the joined texts are what the
/// middleware-contract assertions inspect.
fn tool_result_texts(history: &[Message]) -> Vec<String> {
    history
        .iter()
        .filter(|m| {
            m.role == Role::User
                && m.parts
                    .iter()
                    .any(|p| matches!(p, MessagePart::ToolResult { .. }))
        })
        .map(|m| {
            m.parts
                .iter()
                .filter_map(|p| match p {
                    MessagePart::ToolResult {
                        output: ToolContent::Text(text),
                        ..
                    } => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>()
        })
        .collect()
}

#[tokio::test]
async fn default_construction_wires_the_small_model_profile() {
    let read_executions = Arc::new(Mutex::new(0usize));
    let grep_executions = Arc::new(Mutex::new(0usize));
    let read = CountingRead {
        executions: Arc::clone(&read_executions),
        payload: "x".repeat(20_000),
    };
    let grep = CountingGrep {
        executions: Arc::clone(&grep_executions),
    };
    let client = MockApiClient::new("m").with_responses(vec![
        tool_call_response("c1", "Bash", serde_json::json!({"command": "rm -rf /"})),
        tool_call_response("c2", "Read", serde_json::json!({"path": "/tmp/a.txt"})),
        tool_call_response("c3", "Grep", serde_json::json!({"query": "needle"})),
        tool_call_response("c4", "Grep", serde_json::json!({"query": "needle"})),
        done_response(),
    ]);
    let mut agent = BareLoop::new(
        Arc::new(client),
        registry_with(read, grep),
        Default::default(),
    );

    agent
        .run("do the work", &RunConfig::default())
        .await
        .expect("run completes");

    let history = agent.conversation();
    let texts = tool_result_texts(&history);
    assert_eq!(texts.len(), 4, "four tool results ride the conversation");
    assert!(
        texts[0].contains("[verify] failed:"),
        "the default pipeline verifies: a deny-class command fails softly, got {:?}",
        texts[0]
    );
    assert!(
        texts[1].chars().count() <= 16_384 + "\n[truncated]".chars().count(),
        "the default pipeline caps oversized output, got {} chars",
        texts[1].chars().count()
    );
    assert!(
        texts[3].ends_with("\n[cached]"),
        "the default pipeline memoizes: the repeat grep is served from cache, got {:?}",
        texts[3]
    );
    assert_eq!(
        *read_executions.lock().expect("read lock"),
        1,
        "the read ran exactly once"
    );
    assert_eq!(
        *grep_executions.lock().expect("grep lock"),
        1,
        "the repeat grep never re-ran the tool"
    );
}

#[tokio::test]
async fn default_construction_reinjects_the_goal_from_turn_five() {
    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "y".repeat(64),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let mut responses = Vec::new();
    for i in 0..5 {
        responses.push(tool_call_response(
            &format!("c{i}"),
            "Read",
            serde_json::json!({"path": format!("/tmp/{i}.txt")}),
        ));
    }
    responses.push(done_response());
    let client = MockApiClient::new("m").with_responses(responses);
    let mut agent = BareLoop::new(
        Arc::new(client.clone()),
        registry_with(read, grep),
        Default::default(),
    );

    agent
        .run("remember the goal text", &RunConfig::default())
        .await
        .expect("run completes");

    let requests = client.captured_stream_requests();
    assert!(
        requests.iter().any(|request| {
            request
                .messages
                .iter()
                .any(|m| m.role == Role::System && m.text_content() == "remember the goal text")
        }),
        "the goal reminder re-injects the original request as a system message \
         on a turn at or past the cadence"
    );
}

/// The default `ApiClient` probe: a minimal client that forwards nothing.
///
/// A struct with the three required methods and nothing else, so the probe test observes the trait default in isolation.
struct PlainClient;

impl ApiClient for PlainClient {
    fn model(&self) -> String {
        "plain".to_string()
    }
    fn stream_messages(
        &self,
        _request: &loopctl::api::StreamRequest,
    ) -> Pin<
        Box<
            dyn futures::Stream<
                    Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>,
                > + Send
                + 'static,
        >,
    > {
        Box::pin(futures::stream::empty())
    }
    fn create_message(
        &self,
        _request: &loopctl::api::StreamRequest,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        loopctl::api::NonStreamingResponse,
                        loopctl::api::error::ApiError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async {
            Ok(loopctl::api::NonStreamingResponse {
                message: Message::assistant(""),
                stop_reason: loopctl::stream::StreamStopReason::EndTurn,
                usage: None,
            })
        })
    }
}

#[test]
fn supports_tool_constraints_defaults_false() {
    assert!(
        !PlainClient.supports_tool_constraints(),
        "the trait default declares no constraint support"
    );
    assert!(
        !MockApiClient::new("m").supports_tool_constraints(),
        "the mock without the opt-in flag declares no support"
    );
}

/// Render the captured requests as comparable signatures.
///
/// The Debug render covers every field deterministically within one
/// build, so two loops' request streams compare for equality without a
/// byte-level harness.
fn request_signatures(client: &MockApiClient) -> Vec<String> {
    client
        .captured_stream_requests()
        .into_iter()
        .map(|request| format!("{request:?}"))
        .collect()
}

/// A `Write`-named tool that always succeeds — the host verifier's
/// judgment is the behavior under test.
struct HostWrite;

impl Tool for HostWrite {
    fn name(&self) -> &'static str {
        "Write"
    }
    fn description(&self) -> &'static str {
        "Writes a file"
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
        Box::pin(async { Ok(ToolOutput::text("wrote")) })
    }
}

#[tokio::test]
async fn frontier_profile_restores_the_bare_loop() {
    let script = |path: &str| {
        vec![
            tool_call_response("c1", "Read", serde_json::json!({"path": path})),
            tool_call_response("c2", "Bash", serde_json::json!({"command": "echo hi"})),
            tool_call_response("c3", "Read", serde_json::json!({"path": path})),
            done_response(),
        ]
    };

    let via_builder_read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let via_builder_grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let builder_client = MockApiClient::new("m").with_responses(script("/tmp/a.txt"));
    let mut via_builder = BareLoop::new(
        Arc::new(builder_client.clone()),
        registry_with(via_builder_read, via_builder_grep),
        Default::default(),
    )
    .with_profile(&loopctl::presets::FrontierProfile)
    .expect("the frontier profile applies");
    via_builder
        .run("bare please", &RunConfig::default())
        .await
        .expect("run completes");
    drop(via_builder);

    let via_apply_read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let via_apply_grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let apply_client = MockApiClient::new("m").with_responses(script("/tmp/a.txt"));
    let mut via_apply = BareLoop::new(
        Arc::new(apply_client.clone()),
        registry_with(via_apply_read, via_apply_grep),
        Default::default(),
    );
    loopctl::presets::FrontierProfile::apply(&mut via_apply).expect("the apply spelling works");
    via_apply
        .run("bare please", &RunConfig::default())
        .await
        .expect("run completes");

    let builder_signatures = request_signatures(&builder_client);
    let apply_signatures = request_signatures(&apply_client);
    assert_eq!(
        builder_signatures, apply_signatures,
        "both opt-out spellings produce identical request streams"
    );
    assert_eq!(builder_signatures.len(), 4);
    for signature in &builder_signatures {
        assert!(
            !signature.contains("[cached]"),
            "no cache markers ride the bare loop: {signature}"
        );
    }
}

#[tokio::test]
async fn opt_out_composes_with_explicit_middleware() {
    use loopctl::middleware::ToolPipeline;
    use loopctl::middleware::verify::{NoopVerifier, VerifyMiddleware};

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![
        tool_call_response("c1", "Write", serde_json::json!({"path": "/tmp/a.txt"})),
        done_response(),
    ]);
    let mut all_tools = registry_with(read, grep);
    all_tools.register(HostWrite);
    let mut agent = BareLoop::new(Arc::new(client), all_tools, Default::default());
    loopctl::presets::FrontierProfile::apply(&mut agent).expect("opt out first");
    let mut tools = ToolRegistry::new();
    tools.register(OkBash);
    tools.register(HostWrite);
    let pipeline = ToolPipeline::builder()
        .with_middleware(VerifyMiddleware::new(
            Arc::new(NoopVerifier),
            vec!["Write".to_string()],
        ))
        .with_core(Arc::new(tools));
    agent
        .set_pipeline(pipeline)
        .expect("host pipeline installs");

    agent
        .run("compose please", &RunConfig::default())
        .await
        .expect("run completes");

    let history = agent.conversation();
    let texts = tool_result_texts(&history);
    assert_eq!(texts.len(), 1, "one tool result rides the conversation");
    assert!(
        texts[0].contains("[verify]"),
        "the host's own middleware runs on the opted-out loop, got {:?}",
        texts[0]
    );
}

#[tokio::test]
async fn with_profile_chains_and_propagates() {
    struct RejectingProfile;
    impl loopctl::presets::Profile for RejectingProfile {
        fn apply<C: ApiClient>(
            &self,
            _loop_: &mut BareLoop<C>,
        ) -> Result<(), loopctl::error::LoopError> {
            Err(loopctl::error::LoopError::InvalidInput(
                "profiles may refuse".to_string(),
            ))
        }
    }

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![done_response()]);
    let rejected = BareLoop::new(
        Arc::new(client.clone()),
        registry_with(read, grep),
        Default::default(),
    )
    .with_profile(&RejectingProfile);
    assert!(
        rejected.is_err(),
        "the convenience builder propagates a profile's refusal"
    );

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let mut chained = BareLoop::new(
        Arc::new(client),
        registry_with(read, grep),
        Default::default(),
    )
    .with_profile(&loopctl::presets::FrontierProfile)
    .expect("the frontier profile applies");
    chained
        .run("chain please", &RunConfig::default())
        .await
        .expect("the chained loop runs");
}

#[tokio::test]
async fn a_bundle_supplied_pipeline_survives_default_construction() {
    use loopctl::managers::LoopManagers;
    use loopctl::middleware::ToolPipeline;
    use loopctl::middleware::verify::{NoopVerifier, VerifyMiddleware};

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "payload".to_string(),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![
        tool_call_response("c1", "Write", serde_json::json!({"command": "rm -rf /"})),
        done_response(),
    ]);
    let mut host_tools = ToolRegistry::new();
    host_tools.register(OkBash);
    host_tools.register(HostWrite);
    let host_pipeline = ToolPipeline::builder()
        .with_middleware(VerifyMiddleware::new(
            Arc::new(NoopVerifier),
            vec!["Write".to_string()],
        ))
        .with_core(Arc::new(host_tools))
        .build()
        .expect("the host pipeline builds");
    let managers = LoopManagers::new().with_pipeline(host_pipeline);
    let mut loop_tools = registry_with(read, grep);
    loop_tools.register(HostWrite);
    let mut agent =
        BareLoop::new_with_managers(Arc::new(client), loop_tools, Default::default(), managers);

    agent
        .run("host stack please", &RunConfig::default())
        .await
        .expect("run completes");

    let history = agent.conversation();
    let texts = tool_result_texts(&history);
    assert_eq!(texts.len(), 1, "one tool result rides the conversation");
    assert!(
        texts[0].contains("[verify] passed:"),
        "the bundle-carried host middleware survives default construction — \
         the noop verifier's pass, not the default stack's verdict, rides \
         the result: {:?}",
        texts[0]
    );
}

#[cfg(feature = "bedrock")]
#[test]
fn bedrock_declares_no_constraint_support_it_cannot_forward() {
    let client = loopctl::provider::bedrock::BedrockClient::builder()
        .region("us-east-1")
        .access_key_id("k")
        .secret_access_key("s")
        .model("m")
        .build()
        .expect("the fixture client builds");
    assert!(
        !client.supports_tool_constraints(),
        "the probe must not declare support the client cannot forward — \
         a true here fails every default-built Bedrock turn at the \
         options gate"
    );
}

/// A contributor that always emits one fixed system message — the
/// marker `clear_contributors` must remove.
struct TagContributor;

impl loopctl::contributor::ContextContributor for TagContributor {
    fn contribute(&self, _context: &loopctl::contributor::ContributorContext) -> Option<Message> {
        let mut message = Message::assistant("tag-contributor marker");
        message.role = Role::System;
        Some(message)
    }
}

fn goal_message_count(request: &loopctl::api::StreamRequest, goal: &str) -> usize {
    request
        .messages
        .iter()
        .filter(|m| m.role == Role::System && m.text_content() == goal)
        .count()
}

fn six_turn_read_script() -> Vec<MockResponse> {
    let mut responses = Vec::new();
    for i in 0..5 {
        responses.push(tool_call_response(
            &format!("c{i}"),
            "Read",
            serde_json::json!({"path": format!("/tmp/{i}.txt")}),
        ));
    }
    responses.push(done_response());
    responses
}

#[tokio::test]
async fn explicit_apply_keeps_the_default_pipeline_verifying() {
    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "r".repeat(32),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![
        tool_call_response("c1", "Bash", serde_json::json!({"command": "rm -rf /"})),
        tool_call_response("c2", "Grep", serde_json::json!({"query": "n"})),
        tool_call_response("c3", "Grep", serde_json::json!({"query": "n"})),
        done_response(),
    ]);
    let mut agent = BareLoop::new(
        Arc::new(client),
        registry_with(read, grep),
        Default::default(),
    );
    loopctl::presets::ConstrainedProfile::apply(&mut agent).expect("the profile applies");
    let _run = agent.run("keep verifying", &RunConfig::default()).await;

    let texts = tool_result_texts(&agent.conversation());
    assert_eq!(texts.len(), 3, "three tool results ride the conversation");
    assert!(
        texts[0].contains("[verify] failed:"),
        "an explicit apply over a default-built loop must keep the builtin \
         verifier — a deny-class command still fails softly, got {:?}",
        texts[0]
    );
    assert!(
        texts[2].ends_with("\n[cached]"),
        "an explicit apply over a default-built loop must keep memoization — \
         the repeat grep serves from cache, got {:?}",
        texts[2]
    );
}

#[tokio::test]
async fn explicit_apply_keeps_a_host_supplied_pipeline() {
    use loopctl::managers::LoopManagers;
    use loopctl::middleware::ToolPipeline;
    use loopctl::middleware::verify::{NoopVerifier, VerifyMiddleware};

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "r".repeat(32),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![
        tool_call_response("c1", "Write", serde_json::json!({"command": "rm -rf /"})),
        done_response(),
    ]);
    let mut host_tools = registry_with(read, grep);
    host_tools.register(HostWrite);
    let mut loop_tools = registry_with(
        CountingRead {
            executions: Arc::new(Mutex::new(0usize)),
            payload: "r".repeat(32),
        },
        CountingGrep {
            executions: Arc::new(Mutex::new(0usize)),
        },
    );
    loop_tools.register(HostWrite);
    loop_tools.register(OkBash);
    let host_pipeline = ToolPipeline::builder()
        .with_middleware(VerifyMiddleware::new(
            Arc::new(NoopVerifier),
            vec!["Write".to_string()],
        ))
        .with_core(Arc::new(host_tools))
        .build()
        .expect("the host pipeline builds");
    let managers = LoopManagers::new().with_pipeline(host_pipeline);
    let mut agent =
        BareLoop::new_with_managers(Arc::new(client), loop_tools, Default::default(), managers);
    loopctl::presets::ConstrainedProfile::apply(&mut agent).expect("the profile applies");
    let _run = agent
        .run("host stack survives", &RunConfig::default())
        .await;

    let texts = tool_result_texts(&agent.conversation());
    assert_eq!(texts.len(), 1, "one tool result rides the conversation");
    assert!(
        texts[0].contains("[verify] passed:"),
        "an explicit apply must not replace a host-supplied pipeline — the \
         noop verifier's pass, not the builtin stack's verdict, rides the \
         result: {:?}",
        texts[0]
    );
}

#[tokio::test]
async fn explicit_profile_apply_owns_the_goal_reminder_seam() {
    let goal = "the one goal text";

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "y".repeat(64),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(six_turn_read_script());
    let mut agent = BareLoop::new(
        Arc::new(client.clone()),
        registry_with(read, grep),
        Default::default(),
    );
    loopctl::presets::ConstrainedProfile::apply(&mut agent).expect("the profile applies");
    let _run = agent.run(goal, &RunConfig::default()).await;

    let doubled = client
        .captured_stream_requests()
        .iter()
        .map(|request| goal_message_count(request, goal))
        .max()
        .unwrap_or(0);
    assert_eq!(
        doubled, 1,
        "a default-built loop that then applies the constrained profile sends \
         the original request exactly once per reminder turn, got {doubled}"
    );

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "y".repeat(64),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(six_turn_read_script());
    let mut agent = BareLoop::new(
        Arc::new(client.clone()),
        registry_with(read, grep),
        Default::default(),
    );
    loopctl::presets::FrontierProfile::apply(&mut agent).expect("the opt-out applies");
    let _run = agent.run(goal, &RunConfig::default()).await;

    let frontier_max = client
        .captured_stream_requests()
        .iter()
        .map(|request| goal_message_count(request, goal))
        .max()
        .unwrap_or(0);
    assert_eq!(
        frontier_max, 0,
        "the frontier opt-out sends no goal reminder at all, got {frontier_max}"
    );

    let read = CountingRead {
        executions: Arc::new(Mutex::new(0usize)),
        payload: "y".repeat(64),
    };
    let grep = CountingGrep {
        executions: Arc::new(Mutex::new(0usize)),
    };
    let client = MockApiClient::new("m").with_responses(vec![done_response()]);
    let mut agent = BareLoop::new(
        Arc::new(client.clone()),
        registry_with(read, grep),
        Default::default(),
    );
    agent.add_contributor(Box::new(TagContributor));
    agent.clear_contributors();
    let _run = agent.run("clear the seam", &RunConfig::default()).await;

    assert!(
        client
            .captured_stream_requests()
            .iter()
            .all(|request| !request.messages.iter().any(|m| {
                m.role == Role::System && m.text_content().contains("tag-contributor marker")
            })),
        "clear_contributors removes every registered contributor — the tag \
         marker must never ride a request"
    );
}
struct RecordingConstraintClient {
    declares_support: bool,
    seen_constraints: Arc<Mutex<Vec<loopctl::structured::ToolConstraint>>>,
}

impl ApiClient for RecordingConstraintClient {
    fn model(&self) -> String {
        "recording".to_string()
    }
    fn stream_messages(
        &self,
        _request: &loopctl::api::StreamRequest,
    ) -> Pin<
        Box<
            dyn futures::Stream<
                    Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>,
                > + Send
                + 'static,
        >,
    > {
        Box::pin(futures::stream::empty())
    }
    fn create_message(
        &self,
        _request: &loopctl::api::StreamRequest,
    ) -> Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        loopctl::api::NonStreamingResponse,
                        loopctl::api::error::ApiError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async {
            Ok(loopctl::api::NonStreamingResponse {
                message: Message::assistant(""),
                stop_reason: loopctl::stream::StreamStopReason::EndTurn,
                usage: None,
            })
        })
    }
    fn supports_tool_constraints(&self) -> bool {
        self.declares_support
    }
    fn stream_messages_with_options(
        &self,
        _request: &loopctl::api::StreamRequest,
        options: RequestOptions,
    ) -> Pin<
        Box<
            dyn futures::Stream<
                    Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>,
                > + Send
                + 'static,
        >,
    > {
        self.seen_constraints
            .lock()
            .expect("constraints lock")
            .push(options.tool_constraint.clone());
        Box::pin(futures::stream::empty())
    }
}

#[tokio::test]
async fn default_construction_attaches_strict_options_only_when_the_client_declares_support() {
    for (declares_support, expect_strict) in [(true, true), (false, false)] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let client = RecordingConstraintClient {
            declares_support,
            seen_constraints: Arc::clone(&seen),
        };
        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), Default::default());
        let _run = agent.run("probe arm", &RunConfig::default()).await;

        let recorded = seen.lock().expect("constraints lock").clone();
        assert!(
            !recorded.is_empty(),
            "the run must issue at least one model call"
        );
        assert!(
            recorded.iter().all(|constraint| {
                matches!(constraint, loopctl::structured::ToolConstraint::Strict) == expect_strict
            }),
            "a client declaring {declares_support} must get {} on every \
             request, got {recorded:?}",
            if expect_strict { "Strict" } else { "None" }
        );
    }
}
