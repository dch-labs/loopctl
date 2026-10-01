//! External-approver pins — the parked `ask` channel end to end.
//!
//! One contract: a hook's `ask` parks the run at the pre-execution
//! boundary until the installed
//! [`AskResolver`](loopctl::ask::AskResolver) answers (or the ask
//! deadline denies), every resolution and expiry emits a
//! `GateDecision` record, and an approval is bound to the args digest
//! of the call that minted it — a changed call re-asks.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone,
    clippy::clone_on_ref_ptr
)]

#[cfg(all(feature = "hooks", feature = "testing"))]
mod engine {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::Duration;

    use loopctl::ask::{ApprovalChannel, AskResolution, AskResolver, PendingAsk};
    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::hooks::context::PreToolUseContext;
    use loopctl::hooks::{Hook, HookAction, HookExecutor, Interactivity};
    use loopctl::managers::LoopManagers;
    use loopctl::message::{Message, MessagePart};
    use loopctl::observer::{
        GateDecisionContext, LoopObserver, RunStartContext, ToolPostContext, ToolPreContext,
    };
    use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
    use loopctl::tool::permission::GateDecision;
    use loopctl::tool::{GateRuleSource, GateVerdict, ToolContext, ToolOutput, ToolRegistry};

    /// A hook that asks for external approval on every deploy call.
    struct AskOnDeploy;

    impl Hook for AskOnDeploy {
        fn name(&self) -> &'static str {
            "ask-on-deploy"
        }

        fn on_pre_tool_use(&self, ctx: &PreToolUseContext) -> Option<HookAction> {
            (ctx.tool_name == "deploy").then(|| HookAction::ask("approve the deploy?"))
        }
    }

    /// A resolver that never answers — the expiry pin's approver.
    struct NeverResolves;

    impl AskResolver for NeverResolves {
        fn resolve<'a>(
            &'a self,
            _pending: &'a PendingAsk,
        ) -> Pin<Box<dyn Future<Output = AskResolution> + Send + 'a>> {
            Box::pin(std::future::pending())
        }
    }

    /// An observer collecting gate decisions and the dispatch lifecycle.
    struct GateCollector {
        decisions: std::sync::Mutex<Vec<(usize, String, GateDecision)>>,
        order: std::sync::Mutex<Vec<(&'static str, String, bool)>>,
    }

    impl GateCollector {
        fn new() -> Self {
            Self {
                decisions: std::sync::Mutex::new(Vec::new()),
                order: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<(usize, String, GateDecision)> {
            self.decisions.lock().expect("decision lock").clone()
        }

        fn order(&self) -> Vec<(&'static str, String, bool)> {
            self.order.lock().expect("order lock").clone()
        }
    }

    impl LoopObserver for GateCollector {
        fn name(&self) -> &'static str {
            "gate-collector"
        }

        fn on_run_start(&self, _ctx: &RunStartContext) {}

        fn on_gate_decision(&self, ctx: &GateDecisionContext) {
            self.order
                .lock()
                .expect("order lock")
                .push(("gate", ctx.call_id.clone(), true));
            self.decisions.lock().expect("decision lock").push((
                ctx.turn,
                ctx.call_id.clone(),
                ctx.decision.clone(),
            ));
        }

        fn on_tool_pre(&self, ctx: &ToolPreContext) {
            self.order.lock().expect("order lock").push((
                "tool_pre",
                ctx.tool_call_id.clone(),
                true,
            ));
        }

        fn on_tool_post(&self, ctx: &ToolPostContext) {
            self.order.lock().expect("order lock").push((
                "tool_post",
                ctx.tool_call_id.clone(),
                ctx.is_error,
            ));
        }
    }

    /// The deploy tool, counting its own executions.
    struct CountingDeploy {
        executions: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl loopctl::tool::Tool for CountingDeploy {
        fn name(&self) -> &'static str {
            "deploy"
        }

        fn description(&self) -> &'static str {
            "Deploy the service"
        }

        fn schema(&self) -> loopctl::tool::ToolSchema {
            loopctl::tool::ToolSchema::new(
                "deploy",
                "Deploy the service",
                serde_json::json!({
                    "type": "object",
                    "properties": {"target": {"type": "string"}}
                }),
            )
        }

        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send + '_>>
        {
            let executions = Arc::clone(&self.executions);
            Box::pin(async move {
                executions.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(ToolOutput::text("deployed"))
            })
        }
    }

    /// A registry holding the counting deploy tool.
    fn deploy_registry(executions: Arc<std::sync::atomic::AtomicUsize>) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(CountingDeploy { executions });
        registry
    }

    /// The scripted model responses for one deploy call and its wrap-up.
    fn deploy_then_done(first_target: &str, second_target: Option<&str>) -> Vec<MockResponse> {
        let mut responses = vec![MockResponse {
            text: "deploying".into(),
            tool_call: Some(MockToolCall {
                id: "call_ask_1".into(),
                name: "deploy".into(),
                input: serde_json::json!({ "target": first_target }),
            }),
            stop_reason: "tool_use".into(),
        }];
        if let Some(target) = second_target {
            responses.push(MockResponse {
                text: "deploying again".into(),
                tool_call: Some(MockToolCall {
                    id: "call_ask_2".into(),
                    name: "deploy".into(),
                    input: serde_json::json!({ "target": target }),
                }),
                stop_reason: "tool_use".into(),
            });
        }
        responses.push(MockResponse {
            text: "done".into(),
            tool_call: None,
            stop_reason: "end_turn".into(),
        });
        responses
    }

    fn asking_executor() -> Arc<HookExecutor> {
        Arc::new(
            HookExecutor::new()
                .with_hook(Arc::new(AskOnDeploy))
                .with_interactivity(Interactivity::Interactive),
        )
    }

    fn fixed_wall() -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(1_700_000_000))
            .expect("a representable wall time")
    }

    #[tokio::test]
    async fn parked_run_resumes_on_external_approval() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let registry = deploy_registry(Arc::clone(&executions));
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            registry,
            SessionConfig::default(),
            managers,
        );
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert_eq!(pending.tool, "deploy", "the ask names the parked tool");
        assert_eq!(pending.call_id, "call_ask_1", "the ask binds its call id");
        assert_eq!(pending.turn, 0, "the ask binds its turn");
        assert_eq!(
            pending.args_digest,
            GateDecision::args_digest(&serde_json::json!({ "target": "/prod" })),
            "the ask carries the gate digest of the model's arguments"
        );
        assert_eq!(
            pending.prompt, "approve the deploy?",
            "the hook's prompt reaches the approver verbatim"
        );
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval reaches the still-parked run"
        );

        let run = run_task
            .await
            .expect("the run task completes")
            .expect("the resumed run completes");
        assert_eq!(run.output.as_deref(), Some("done"), "the run finished");
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the approved call executed exactly once"
        );

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            1,
            "one decision for one parked ask — the resolution, not the step"
        );
        let (turn, call_id, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(turn, 0, "the record carries the parked dispatch's turn");
        assert_eq!(call_id, "call_ask_1", "the record pairs with its call");
        assert_eq!(decision.verdict, GateVerdict::AskAllowed);
        assert_eq!(decision.rule_id, "ask");
        assert_eq!(decision.rule_source, GateRuleSource::AskResolver);
        assert_eq!(
            decision.args_digest,
            GateDecision::args_digest(&serde_json::json!({ "target": "/prod" })),
            "the record digests the call the approval unblocked"
        );
        assert_eq!(
            decision.ts, 1_700_000_000_000,
            "the engine stamped the record from its clock seam"
        );

        let order = collector.order();
        let kinds: Vec<&'static str> = order.iter().map(|(kind, _, _)| *kind).collect();
        assert_eq!(
            kinds,
            vec!["tool_pre", "gate", "tool_post"],
            "the decision fires after the pipeline parks and before the post event"
        );
        assert!(
            !order.last().expect("the post entry exists").2,
            "the approved call's result is not an error"
        );
    }

    #[tokio::test]
    async fn ask_timeout_defaults_to_deny_and_audits_the_expiry() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let registry = deploy_registry(Arc::clone(&executions));
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(Arc::new(NeverResolves))
            .with_ask_timeout(Duration::from_millis(50))
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            registry,
            SessionConfig::default(),
            managers,
        );

        let run = agent
            .run("deploy", &RunConfig::default())
            .await
            .expect("an expired ask denies softly — the run still completes");
        assert_eq!(
            run.output.as_deref(),
            Some("done"),
            "the model saw the denial and wrapped up"
        );
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the expired call never executed"
        );

        let recorded = collector.recorded();
        assert_eq!(recorded.len(), 1, "exactly one expiry record");
        let (_, call_id, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(call_id, "call_ask_1");
        assert_eq!(
            decision.verdict,
            GateVerdict::AskExpired,
            "the verdict names the expiry — not a refusal and not a headless ask"
        );
        assert_eq!(
            decision.rule_source,
            GateRuleSource::Engine,
            "the deadline policy is the engine's, not the approver's"
        );
        assert!(
            decision
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("expired")),
            "the reason names the expiry: {:?}",
            decision.reason
        );
        let order = collector.order();
        assert!(
            order.last().expect("the post entry exists").2,
            "the model saw the expiry as an errored tool result"
        );
    }

    #[tokio::test]
    async fn approval_is_bound_to_the_args_digest() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let seen: Arc<std::sync::Mutex<Vec<PendingAsk>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let approver_task = {
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                for _ in 0..2 {
                    let Some(pending) = approver.next_pending().await else {
                        break;
                    };
                    seen.lock().expect("seen lock").push(pending.clone());
                    assert!(
                        approver.resolve(pending.id, AskResolution::Approve),
                        "the auto-approver's answer reaches the still-parked ask"
                    );
                }
            })
        };
        let registry = deploy_registry(Arc::clone(&executions));
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/a", Some("/b")));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            registry,
            SessionConfig::default(),
            managers,
        );

        let run = agent
            .run("deploy twice", &RunConfig::default())
            .await
            .expect("both approved calls complete");
        assert_eq!(run.output.as_deref(), Some("done"));
        drop(agent);
        approver_task.await.expect("the approver task ends");

        let seen = seen.lock().expect("seen lock").clone();
        assert_eq!(
            seen.len(),
            2,
            "a changed call re-asks — the first approval never carries across"
        );
        assert_ne!(
            seen[0].args_digest, seen[1].args_digest,
            "the two asks digest their own calls' arguments"
        );
        assert_eq!(
            seen[0].args_digest,
            GateDecision::args_digest(&serde_json::json!({ "target": "/a" })),
            "the first ask digests the model's first call"
        );
        assert_eq!(
            seen[1].args_digest,
            GateDecision::args_digest(&serde_json::json!({ "target": "/b" })),
            "the second ask digests the changed call"
        );
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "both approvals unblocked their own calls"
        );
        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            2,
            "one resolution record per ask — two approvals, two records"
        );
        assert!(
            recorded
                .iter()
                .all(|(_, _, decision)| decision.verdict == GateVerdict::AskAllowed),
            "both records name their approvals"
        );
    }

    #[tokio::test]
    async fn a_hook_ask_without_a_resolver_denies_headlessly_and_records_it() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let registry = deploy_registry(Arc::clone(&executions));
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            registry,
            SessionConfig::default(),
            managers,
        );

        let run = agent
            .run("deploy", &RunConfig::default())
            .await
            .expect("a headless ask denies softly — the run completes");
        assert_eq!(run.output.as_deref(), Some("done"));
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "nothing executes without an approver"
        );

        let headless_text = agent
            .conversation()
            .iter()
            .rev()
            .find_map(|message: &Message| {
                message.parts.iter().find_map(|part| match part {
                    MessagePart::ToolResult { output, .. } => Some(output.to_string()),
                    _ => None,
                })
            })
            .expect("the denial landed in the conversation as a tool result");
        assert_eq!(
            headless_text, "approve the deploy?",
            "the headless denial carries the hook's message byte-for-byte, exactly as before"
        );

        let recorded = collector.recorded();
        assert_eq!(recorded.len(), 1, "one record for the unanswered ask");
        let (_, _, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(decision.verdict, GateVerdict::AskUnresolved);
        assert_eq!(decision.rule_id, "hook");
        assert_eq!(decision.rule_source, GateRuleSource::Engine);
    }

    #[tokio::test]
    async fn an_approver_refusal_denies_softly_and_records_it() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            deploy_registry(Arc::clone(&executions)),
            SessionConfig::default(),
            managers,
        );
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert!(
            approver.resolve(pending.id, AskResolution::Deny),
            "the refusal reaches the still-parked run"
        );

        let run = run_task
            .await
            .expect("the run task completes")
            .expect("a refused ask denies softly — the run still completes");
        assert_eq!(
            run.output.as_deref(),
            Some("done"),
            "the model saw the refusal and wrapped up"
        );
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the refused call never executed"
        );

        let recorded = collector.recorded();
        assert_eq!(recorded.len(), 1, "one decision for the refused ask");
        let (_, call_id, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(call_id, "call_ask_1");
        assert_eq!(decision.verdict, GateVerdict::AskDenied);
        assert_eq!(decision.rule_id, "ask");
        assert_eq!(decision.rule_source, GateRuleSource::AskResolver);
        assert_eq!(
            decision.reason.as_deref(),
            Some("denied by approver"),
            "the refusal's reason names the approver"
        );
        assert!(
            collector.order().last().expect("the post entry exists").2,
            "the model saw the refusal as an errored tool result"
        );
    }

    #[tokio::test]
    async fn a_cancelled_park_ends_the_run_typed_cancelled_and_records_it() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            deploy_registry(Arc::clone(&executions)),
            SessionConfig::default(),
            managers,
        );
        let cancel = agent.cancel_signal();
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask before the cancel");
        cancel.cancel();
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval physically lands (no yield between cancel and resolve, so \
             the engine has not dropped its receiver) — and the biased select still \
             ends the wait Cancelled"
        );

        let outcome = run_task.await.expect("the run task completes");
        assert!(
            matches!(outcome, Err(loopctl::error::LoopError::Cancelled)),
            "a cancelled park ends the run typed Cancelled, got {outcome:?}"
        );
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "nothing executes on a cancelled park"
        );

        let recorded = collector.recorded();
        assert_eq!(recorded.len(), 1, "one record for the cancelled park");
        let (_, _, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(decision.verdict, GateVerdict::Cancelled);
        assert_eq!(decision.rule_id, "hook");
        assert_eq!(decision.rule_source, GateRuleSource::Engine);
        assert_eq!(
            decision.reason.as_deref(),
            Some("cancelled while awaiting approval"),
            "the reason names the cancellation, not a refusal — the biased select \
             makes the cancel dominate even an approval that landed first"
        );

        let order = collector.order();
        let kinds: Vec<&'static str> = order.iter().map(|(kind, _, _)| *kind).collect();
        assert_eq!(
            kinds,
            vec!["tool_pre", "gate"],
            "a cancelled park fires pre and the record but no post — the run ends \
             instead; the unpaired pre is paired against the run-end event"
        );
    }

    #[tokio::test]
    async fn a_downstream_pipeline_denial_displaces_the_park_approval_record() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let mut pipeline_core = ToolRegistry::new();
        pipeline_core.register(CountingDeploy {
            executions: Arc::clone(&executions),
        });
        let pipeline = loopctl::middleware::ToolPipeline::builder()
            .with_middleware(loopctl::middleware::PermissionMiddleware::deny_all())
            .with_core(Arc::new(pipeline_core))
            .build()
            .expect("the gated pipeline assembles");
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_pipeline(pipeline)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            deploy_registry(Arc::new(std::sync::atomic::AtomicUsize::new(0))),
            SessionConfig::default(),
            managers,
        );
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval reaches the still-parked run"
        );

        let run = run_task
            .await
            .expect("the run task completes")
            .expect("the downstream denial is soft — the run still completes");
        assert_eq!(run.output.as_deref(), Some("done"));
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the pipeline denied the approved call — nothing executed"
        );

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            1,
            "one decision record per dispatch — the approval rode the precedence \
             lattice and the downstream denial displaced it: {recorded:?}"
        );
        let (_, call_id, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(call_id, "call_ask_1");
        assert_eq!(
            decision.verdict,
            GateVerdict::Deny,
            "the surviving record is the deciding one — the pipeline's denial"
        );
        assert_eq!(
            decision.rule_source,
            GateRuleSource::Middleware,
            "the denial is the pipeline middleware's, not the approver's"
        );
        assert_eq!(decision.rule_id, "middleware");
        assert!(
            recorded
                .iter()
                .all(|(_, _, record)| record.verdict != GateVerdict::AskAllowed),
            "no approval record is emitted for a call that did not proceed"
        );
        assert!(
            collector.order().last().expect("the post entry exists").2,
            "the model saw the pipeline denial as an errored tool result"
        );
    }

    /// A deploy tool that counts its start and never resolves — the
    /// cancelled-dispatch pin's tool.
    ///
    /// The count is the pin's arm-identity guard: one start proves the
    /// cancel landed with the tool in flight (the mid-execution arm),
    /// not before dispatch began.
    struct CountingHangingDeploy {
        starts: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl loopctl::tool::Tool for CountingHangingDeploy {
        fn name(&self) -> &'static str {
            "deploy"
        }

        fn description(&self) -> &'static str {
            "Counts its start, never resolves"
        }

        fn schema(&self) -> loopctl::tool::ToolSchema {
            loopctl::tool::ToolSchema::new(
                "deploy",
                "Counts its start, never resolves",
                serde_json::json!({"type": "object"}),
            )
        }

        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send + '_>>
        {
            let starts = Arc::clone(&self.starts);
            Box::pin(async move {
                starts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::pending::<Result<ToolOutput, loopctl::tool::ToolError>>().await
            })
        }
    }

    #[tokio::test]
    async fn an_approved_ask_records_its_decision_when_dispatch_is_cancelled() {
        let collector = Arc::new(GateCollector::new());
        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let mut registry = ToolRegistry::new();
        registry.register(CountingHangingDeploy {
            starts: Arc::clone(&starts),
        });
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            registry,
            SessionConfig::default(),
            managers,
        );
        let cancel = agent.cancel_signal();
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval unblocks the park — the call proceeds to dispatch"
        );
        tokio::task::yield_now().await;
        cancel.cancel();

        let outcome = run_task.await.expect("the run task completes");
        assert!(
            matches!(outcome, Err(loopctl::error::LoopError::Cancelled)),
            "the cancel ends the run while the approved call hangs in dispatch: {outcome:?}"
        );
        assert_eq!(
            starts.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the cancelled dispatch had started the tool — the cancel landed \
             mid-execution, not before dispatch"
        );

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            1,
            "the approval is recorded even though the run stopped before the dispatch \
             returned: {recorded:?}"
        );
        let (_, _, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(decision.verdict, GateVerdict::AskAllowed);
        assert_eq!(decision.rule_source, GateRuleSource::AskResolver);
    }

    #[tokio::test]
    #[cfg(feature = "tool_health")]
    async fn an_approved_ask_records_its_decision_when_the_breaker_refuses() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let health = Arc::new(loopctl::tool::health::ToolHealthRegistry::new());
        for _ in 0..3 {
            health.record_failure("deploy", Duration::ZERO);
        }
        assert!(
            !health.allow_request("deploy"),
            "precondition: the breaker is open for deploy"
        );
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_health_registry(Arc::clone(&health))
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            deploy_registry(Arc::clone(&executions)),
            SessionConfig::default(),
            managers,
        );
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval unblocks the park"
        );

        let run = run_task
            .await
            .expect("the run task completes")
            .expect("the breaker refusal is soft — the run still completes");
        assert_eq!(run.output.as_deref(), Some("done"));
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the open breaker refused the approved call"
        );

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            1,
            "the approval is recorded even though the breaker refused before execution: \
             {recorded:?}"
        );
        let (_, _, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(decision.verdict, GateVerdict::AskAllowed);
        assert_eq!(decision.rule_source, GateRuleSource::AskResolver);
    }

    #[tokio::test]
    async fn an_approved_ask_records_its_decision_when_pre_detection_refuses() {
        let collector = Arc::new(GateCollector::new());
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let detection = loopctl::detection::DetectionManager::new_with_config(
            loopctl::detection::DetectionConfig {
                loop_threshold: 2,
                stop_threshold: 2,
                ..Default::default()
            },
        )
        .expect("valid detection config");
        let repeated = loopctl::detection::loop_detector::Operation {
            tool: "deploy".to_string(),
            primary_param: "{\"target\":\"/prod\"}".to_string(),
            result_hash: Some(7),
        };
        detection
            .record_operation(repeated.clone())
            .expect("records");
        detection
            .record_operation(repeated)
            .expect("records — two identical operations trip the stop threshold");
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_hook_executor(asking_executor())
            .with_ask_resolver(resolver)
            .with_detection(detection)
            .with_clock(Arc::new(
                loopctl::determinism::FixedClock::new(fixed_wall()),
            ));
        let client = MockApiClient::new("gated").with_responses(deploy_then_done("/prod", None));
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            deploy_registry(Arc::clone(&executions)),
            SessionConfig::default(),
            managers,
        );
        let run_task =
            tokio::spawn(async move { agent.run("deploy", &RunConfig::default()).await });

        let pending = approver
            .next_pending()
            .await
            .expect("the parked run delivers its ask");
        assert!(
            approver.resolve(pending.id, AskResolution::Approve),
            "the approval unblocks the park"
        );

        let outcome = run_task.await.expect("the run task completes");
        assert!(
            matches!(outcome, Err(loopctl::error::LoopError::LoopDetected { .. })),
            "the pre-seeded repetition hard-stops the run before execution: {outcome:?}"
        );
        assert_eq!(
            executions.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the refused call never executed"
        );

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            1,
            "the approval is recorded even though pre-detection refused the dispatch: \
             {recorded:?}"
        );
        let (_, _, decision) = recorded.first().cloned().expect("the record exists");
        assert_eq!(decision.verdict, GateVerdict::AskAllowed);
        assert_eq!(decision.rule_source, GateRuleSource::AskResolver);
    }
}
