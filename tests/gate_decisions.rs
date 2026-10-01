//! Gate-decision pins — the permission gate's verdicts as data.
//!
//! Two contracts: every decision the permission middleware makes
//! rides the dispatch result out to the engine, which emits it as an
//! [`on_gate_decision`](loopctl::observer::LoopObserver::on_gate_decision)
//! record with rule provenance and a stable argument digest; and
//! [`PermissionMiddleware::evaluate`] answers what the gate *would*
//! decide for a call without dispatching it — the real policy stack,
//! never the resolver.

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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use loopctl::cancel::CancelSignal;
use loopctl::middleware::permission::AskResolverFn;
use loopctl::middleware::{
    GateRuleSource, GateVerdict, MemoizingMiddleware, NoopPathExtractor, PermissionMiddleware,
    ToolDispatchContext, ToolPipeline,
};
use loopctl::tool::permission::GateDecision;
use loopctl::tool::registry::{FnTool, ToolFn};
use loopctl::tool::{PermissionCheck, ToolContext, ToolOutput, ToolRegistry};

/// The probe tool's implementation: echoes one fixed word.
fn probe_tool(
    _input: serde_json::Value,
    _ctx: &ToolContext,
) -> Pin<Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send>> {
    Box::pin(async { Ok(ToolOutput::text("probed")) })
}

/// A registry holding the one probe tool every verdict targets.
fn probe_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(
        FnTool::new(
            "probe".into(),
            "Echo the probe word back".into(),
            serde_json::json!({
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"]
            }),
            probe_tool as ToolFn,
        )
        .read_only(),
    );
    registry
}

/// A pipeline over the probe registry behind one permission gate.
fn gated_pipeline(permission: PermissionMiddleware) -> Arc<ToolPipeline> {
    Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .with_middleware(permission)
            .build()
            .expect("a core plus the permission middleware assembles"),
    )
}

/// One dispatch context aimed at the probe tool.
fn probe_ctx(word: &str) -> ToolDispatchContext {
    ToolDispatchContext {
        tool_name: "probe".into(),
        input: serde_json::json!({ "word": word }),
        call_id: "call_gate".into(),
        turn_number: 0,
        cancel: Arc::new(CancelSignal::new()),
        permission: PermissionCheck::allow(),
        tool_context: ToolContext::default(),
    }
}

/// The call every evaluate pin asks about.
fn probe_args(word: &str) -> serde_json::Value {
    serde_json::json!({ "word": word })
}

#[tokio::test]
async fn evaluate_matches_enforcement_for_the_same_call() {
    let allow = PermissionMiddleware::from_context().with_check(|_| PermissionCheck::Allow);
    let evaluated = allow.evaluate("probe", &probe_args("hello"));
    assert_eq!(
        evaluated.verdict,
        GateVerdict::Allow,
        "evaluate answers allow for a call the gate passes"
    );
    let result = gated_pipeline(allow).invoke(probe_ctx("hello")).await;
    assert!(!result.is_error, "the same call executes for real");
    assert_eq!(
        result
            .gate
            .as_ref()
            .expect("the allow arm attaches a record")
            .verdict,
        GateVerdict::Allow,
        "the enforcement record agrees with the evaluation"
    );

    let deny = PermissionMiddleware::from_context()
        .with_check(|_| PermissionCheck::deny("not on the allowlist"));
    let evaluated = deny.evaluate("probe", &probe_args("hello"));
    assert_eq!(
        evaluated.verdict,
        GateVerdict::Deny,
        "evaluate answers deny for a call the gate refuses"
    );
    assert_eq!(
        evaluated.reason.as_deref(),
        Some("not on the allowlist"),
        "the refusal reason rides the dry-run answer"
    );
    let result = gated_pipeline(deny).invoke(probe_ctx("hello")).await;
    assert!(result.is_error, "the same call is refused for real");
    assert_eq!(
        result
            .gate
            .as_ref()
            .expect("the deny arm attaches a record")
            .verdict,
        GateVerdict::Deny,
        "the enforcement record agrees with the evaluation"
    );

    let modify = PermissionMiddleware::from_context()
        .with_check(|_| PermissionCheck::modify(serde_json::json!({ "word": "sanitized" })));
    let evaluated = modify.evaluate("probe", &probe_args("hello"));
    assert_eq!(
        evaluated.verdict,
        GateVerdict::AllowModified,
        "evaluate answers allow-modified for a call the gate rewrites"
    );

    let consultations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let spy: AskResolverFn = {
        let consultations = Arc::clone(&consultations);
        Arc::new(move |_prompt: &str, _tool: &str| {
            consultations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { true })
        })
    };
    let asking = PermissionMiddleware::from_context()
        .with_check(|_| PermissionCheck::ask("approve the probe?"))
        .with_ask_resolver(spy);
    let evaluated = asking.evaluate("probe", &probe_args("hello"));
    assert_eq!(
        evaluated.verdict,
        GateVerdict::Ask,
        "evaluate reports the ask without resolving it — a dry-run never prompts"
    );
    assert_eq!(
        consultations.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the resolver was never consulted by the evaluation"
    );
    let result = gated_pipeline(asking).invoke(probe_ctx("hello")).await;
    assert!(!result.is_error, "the resolver approves the real dispatch");
    assert_eq!(
        result
            .gate
            .as_ref()
            .expect("the resolved ask attaches a record")
            .verdict,
        GateVerdict::AskAllowed,
        "enforcement records the user's approval — evaluate's Ask is the pre-resolution answer"
    );
    assert_eq!(
        result
            .gate
            .as_ref()
            .expect("the record carries provenance")
            .rule_source,
        GateRuleSource::AskResolver,
        "the approval's source is the resolver, not the rule that asked"
    );
    assert_eq!(
        consultations.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "exactly the real dispatch consulted the resolver"
    );
}

#[tokio::test]
async fn a_result_without_a_gate_carries_none() {
    let plain = Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .build()
            .expect("a bare core assembles"),
    );
    let result = plain.invoke(probe_ctx("hello")).await;
    assert!(
        result.gate.is_none(),
        "a dispatch no gate consulted carries no record: {result:?}"
    );
}

#[tokio::test]
async fn a_memoized_hit_replays_no_gate_record() {
    let memoize =
        MemoizingMiddleware::new(vec!["probe".into()], vec![], Arc::new(NoopPathExtractor), 5);
    let permission = PermissionMiddleware::from_context()
        .with_named_check("allow-reads", |_| PermissionCheck::Allow);
    let pipeline = Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .with_middleware(memoize)
            .with_middleware(permission)
            .build()
            .expect("the memoize-over-gate pipeline assembles"),
    );

    let first = pipeline.invoke(probe_ctx("hello")).await;
    let first_gate = first
        .gate
        .as_ref()
        .expect("the gate ran for the first, uncached dispatch");
    assert_eq!(
        first_gate.verdict,
        GateVerdict::Allow,
        "the first dispatch's decision rides its result"
    );
    assert!(
        first.output.to_string().contains("probed"),
        "the first dispatch executed the tool: {}",
        first.output
    );

    let second = pipeline.invoke(probe_ctx("hello")).await;
    assert!(
        second.output.to_string().contains("[cached]"),
        "the second dispatch was served from the cache: {}",
        second.output
    );
    assert!(
        second.gate.is_none(),
        "a cache hit replays the stored output, never a gate record — no gate \
         ran for the replaying dispatch, so the engine must emit no decision: \
         {:?}",
        second.gate
    );
}

#[tokio::test]
async fn an_outer_allow_cannot_erase_an_inner_gate_s_deny() {
    let pipeline = Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .with_middleware(PermissionMiddleware::allow_all())
            .with_middleware(PermissionMiddleware::deny_all())
            .build()
            .expect("the stacked-gate pipeline assembles"),
    );

    let result = pipeline.invoke(probe_ctx("hello")).await;
    assert!(
        result.is_error,
        "the inner deny governs the outcome the engine acted on"
    );
    let gate = result
        .gate
        .as_ref()
        .expect("the deciding record survives the outer allow's pass-through");
    assert_eq!(
        gate.verdict,
        GateVerdict::Deny,
        "the emitted verdict must match the outcome — an outer allow record \
         here would self-contradict the errored result beside it"
    );
    assert_eq!(
        gate.reason.as_deref(),
        Some("blocked by policy"),
        "the surviving record is the inner deny's, not the outer allow's"
    );
}

#[tokio::test]
async fn an_inner_pass_through_cannot_mask_an_outer_rewrite() {
    let sanitize = PermissionMiddleware::from_context().with_named_check("sanitize", |_| {
        PermissionCheck::modify(serde_json::json!({ "word": "sanitized" }))
    });
    let authorize = PermissionMiddleware::from_context()
        .with_named_check("authorize-reads", |_| PermissionCheck::Allow);
    let pipeline = Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .with_middleware(sanitize)
            .with_middleware(authorize)
            .build()
            .expect("the rewrite-over-gate pipeline assembles"),
    );

    let model_args = probe_args("hello");
    let result = pipeline.invoke(probe_ctx("hello")).await;
    assert!(!result.is_error, "the rewritten call proceeds");
    let gate = result
        .gate
        .as_ref()
        .expect("the rewriting gate's record rides the result");
    assert_eq!(
        gate.verdict,
        GateVerdict::AllowModified,
        "the surviving verdict must name the rewrite — a plain Allow claims the \
         input ran exactly as the model sent it, which is false here"
    );
    assert_eq!(
        gate.rule_id, "sanitize",
        "the surviving record is the rewriting gate's, not the inner \
         pass-through's"
    );
    assert_eq!(
        gate.args_digest,
        GateDecision::args_digest(&model_args),
        "the surviving digest must match the model's original call — the \
         matchability contract a call-digest join depends on"
    );
}

#[tokio::test]
async fn a_plain_allow_does_not_displace_an_inner_rewrite() {
    let authorize = PermissionMiddleware::from_context()
        .with_named_check("authorize-reads", |_| PermissionCheck::Allow);
    let sanitize = PermissionMiddleware::from_context().with_named_check("sanitize", |_| {
        PermissionCheck::modify(serde_json::json!({ "word": "sanitized" }))
    });
    let pipeline = Arc::new(
        ToolPipeline::builder()
            .with_core(Arc::new(probe_registry()))
            .with_middleware(authorize)
            .with_middleware(sanitize)
            .build()
            .expect("the gate-over-rewrite pipeline assembles"),
    );

    let result = pipeline.invoke(probe_ctx("hello")).await;
    assert!(!result.is_error, "the rewritten call proceeds");
    let gate = result
        .gate
        .as_ref()
        .expect("the rewriting gate's record rides the result");
    assert_eq!(
        gate.verdict,
        GateVerdict::AllowModified,
        "a plain pass-through never outranks a rewrite — precedence moves one \
         way, or an outer allow could mask the rewrite by displacing upward"
    );
    assert_eq!(
        gate.rule_id, "sanitize",
        "the inner rewrite's record survives the outer pass-through"
    );
}

#[test]
fn args_digests_are_canonical_and_stable() {
    let a = GateDecision::args_digest(&serde_json::json!({
        "path": "/tmp/x",
        "limit": 10
    }));
    let b = GateDecision::args_digest(&serde_json::json!({
        "limit": 10,
        "path": "/tmp/x"
    }));
    assert_eq!(
        a, b,
        "object key order never moves the digest — canonical JSON is the input"
    );
    let c = GateDecision::args_digest(&serde_json::json!({
        "path": "/tmp/y",
        "limit": 10
    }));
    assert_ne!(
        a, c,
        "different arguments produce different digests — matchability is preserved"
    );
    assert_eq!(
        a.chars().count(),
        16,
        "the digest renders as 16 hex characters (FNV-1a 64)"
    );
    assert_eq!(
        a, "f1ed61cd8ca5e26b",
        "a known fixture pins the algorithm: FNV-1a 64 over the canonical JSON \
         `{}` — an algorithm or rendering swap breaks the pin, not the callers",
        r#"{"limit":10,"path":"/tmp/x"}"#
    );

    let nested_a = GateDecision::args_digest(&serde_json::json!({
        "b": {"y": 1, "x": 2},
        "a": 0
    }));
    let nested_b = GateDecision::args_digest(&serde_json::json!({
        "a": 0,
        "b": {"x": 2, "y": 1}
    }));
    assert_eq!(
        nested_a, nested_b,
        "canonicalization is recursive — nested key order never moves the digest"
    );
    let array_a = GateDecision::args_digest(&serde_json::json!([{"x": 1}, {"y": 2}]));
    let array_b = GateDecision::args_digest(&serde_json::json!([{"y": 2}, {"x": 1}]));
    assert_ne!(
        array_a, array_b,
        "array order is semantic and must move the digest"
    );
}

#[tokio::test]
async fn a_named_check_carries_its_rule_id() {
    let permission = PermissionMiddleware::from_context()
        .with_named_check("deny-write-etc", |_| {
            PermissionCheck::deny("etc is off limits")
        });
    let result = gated_pipeline(permission).invoke(probe_ctx("hello")).await;
    let gate = result
        .gate
        .as_ref()
        .expect("the deny arm attaches a record");
    assert_eq!(
        gate.rule_id, "deny-write-etc",
        "the manifest-referenceable rule id rides the record verbatim"
    );
    assert_eq!(
        gate.rule_source,
        GateRuleSource::Middleware,
        "the middleware's own check is the rule's source"
    );
}

#[tokio::test]
async fn an_unanswered_ask_records_the_headless_denial() {
    let headless = PermissionMiddleware::from_context()
        .with_check(|_| PermissionCheck::ask("approve the probe?"));
    let result = gated_pipeline(headless).invoke(probe_ctx("hello")).await;
    assert!(result.is_error, "an ask without a resolver denies the call");
    let gate = result
        .gate
        .as_ref()
        .expect("the headless arm attaches a record");
    assert_eq!(
        gate.verdict,
        GateVerdict::AskUnresolved,
        "the verdict names the headless path — the ask was never answered"
    );
    assert_eq!(
        gate.rule_source,
        GateRuleSource::Middleware,
        "the unanswered ask came from the middleware's rule"
    );
}

#[tokio::test]
async fn a_cancelled_prompt_records_the_cancellation() {
    let resolver: AskResolverFn = Arc::new(|_prompt: &str, _tool: &str| {
        Box::pin(std::future::pending::<bool>()) as Pin<Box<dyn Future<Output = bool> + Send>>
    });
    let permission = PermissionMiddleware::from_context()
        .with_check(|_| PermissionCheck::ask("approve the probe?"))
        .with_ask_resolver(resolver);
    let pipeline = gated_pipeline(permission);
    let ctx = probe_ctx("hello");
    ctx.cancel.cancel();
    let result = tokio::time::timeout(Duration::from_secs(1), pipeline.invoke(ctx))
        .await
        .expect("the fired cancel resolves the pending prompt immediately");
    let gate = result
        .gate
        .as_ref()
        .expect("the cancelled arm attaches a record");
    assert_eq!(
        gate.verdict,
        GateVerdict::Cancelled,
        "a prompt the cancel signal cut short records cancellation, not refusal"
    );
}

#[cfg(feature = "testing")]
mod engine_emission {
    use super::*;
    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::managers::LoopManagers;
    use loopctl::observer::{
        GateDecisionContext, LoopObserver, RunStartContext, ToolPostContext, ToolPreContext,
    };
    use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};

    /// An observer collecting every gate decision the engine emits,
    /// plus the dispatch-lifecycle order around each one.
    struct GateCollector {
        decisions: std::sync::Mutex<Vec<(usize, String, GateDecision)>>,
        order: std::sync::Mutex<Vec<(&'static str, String)>>,
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

        fn order(&self) -> Vec<(&'static str, String)> {
            self.order.lock().expect("order lock").clone()
        }
    }

    impl LoopObserver for GateCollector {
        fn name(&self) -> &str {
            "gate-collector"
        }

        fn on_run_start(&self, _ctx: &RunStartContext) {}

        fn on_gate_decision(&self, ctx: &GateDecisionContext) {
            self.order
                .lock()
                .expect("order lock")
                .push(("gate", ctx.call_id.clone()));
            self.decisions.lock().expect("decision lock").push((
                ctx.turn,
                ctx.call_id.clone(),
                ctx.decision.clone(),
            ));
        }

        fn on_tool_pre(&self, ctx: &ToolPreContext) {
            self.order
                .lock()
                .expect("order lock")
                .push(("tool_pre", ctx.tool_call_id.clone()));
        }

        fn on_tool_post(&self, ctx: &ToolPostContext) {
            self.order
                .lock()
                .expect("order lock")
                .push(("tool_post", ctx.tool_call_id.clone()));
        }
    }

    #[tokio::test]
    async fn every_hook_decision_emits_a_record_with_rule_provenance() {
        let collector = Arc::new(GateCollector::new());
        let mut registry = ToolRegistry::new();
        registry.register(
            FnTool::new(
                "deploy".into(),
                "Deploy the service".into(),
                serde_json::json!({"type": "object", "properties": {"target": {"type": "string"}}}),
                |input: serde_json::Value,
                 _ctx: &ToolContext|
                 -> Pin<
                    Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send>,
                > {
                    let _ = input;
                    Box::pin(async { Ok(ToolOutput::text("deployed")) })
                },
            )
            .read_only(),
        );

        let resolver: AskResolverFn = Arc::new(|prompt: &str, _tool: &str| {
            let approved = prompt.contains("prod");
            Box::pin(async move { approved })
        });
        let permission = PermissionMiddleware::from_context()
            .with_check(|ctx| {
                let target = ctx
                    .input
                    .get("target")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                match target {
                    "/etc" => PermissionCheck::deny("etc is off limits"),
                    "/prod" | "/root" => {
                        PermissionCheck::ask(format!("approve the deploy to {target}?"))
                    }
                    _ => PermissionCheck::Allow,
                }
            })
            .with_ask_resolver(resolver);
        let pipeline = ToolPipeline::builder()
            .with_core(Arc::new(registry))
            .with_middleware(permission)
            .build()
            .expect("the gated pipeline assembles");
        let wall = std::time::SystemTime::UNIX_EPOCH
            .checked_add(std::time::Duration::from_secs(1_700_000_000))
            .expect("a representable wall time");
        let managers = LoopManagers::new()
            .with_observer(collector.clone() as Arc<dyn LoopObserver>)
            .with_pipeline(pipeline)
            .with_clock(Arc::new(loopctl::determinism::FixedClock::new(wall)));
        let client = MockApiClient::new("gated-model").with_responses(vec![
            MockResponse {
                text: "deploying to etc".into(),
                tool_call: Some(MockToolCall {
                    id: "call_gate_1".into(),
                    name: "deploy".into(),
                    input: serde_json::json!({ "target": "/etc" }),
                }),
                stop_reason: "tool_use".into(),
            },
            MockResponse {
                text: "deploying to srv".into(),
                tool_call: Some(MockToolCall {
                    id: "call_gate_2".into(),
                    name: "deploy".into(),
                    input: serde_json::json!({ "target": "/srv" }),
                }),
                stop_reason: "tool_use".into(),
            },
            MockResponse {
                text: "deploying to prod".into(),
                tool_call: Some(MockToolCall {
                    id: "call_gate_3".into(),
                    name: "deploy".into(),
                    input: serde_json::json!({ "target": "/prod" }),
                }),
                stop_reason: "tool_use".into(),
            },
            MockResponse {
                text: "deploying to root".into(),
                tool_call: Some(MockToolCall {
                    id: "call_gate_4".into(),
                    name: "deploy".into(),
                    input: serde_json::json!({ "target": "/root" }),
                }),
                stop_reason: "tool_use".into(),
            },
            MockResponse {
                text: "done".into(),
                tool_call: None,
                stop_reason: "end_turn".into(),
            },
        ]);
        let mut advertised = ToolRegistry::new();
        advertised.register(
            FnTool::new(
                "deploy".into(),
                "Deploy the service".into(),
                serde_json::json!({"type": "object"}),
                |_input: serde_json::Value,
                 _ctx: &ToolContext|
                 -> Pin<
                    Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send>,
                > { Box::pin(async { Ok(ToolOutput::text("deployed")) }) },
            )
            .read_only(),
        );
        let mut agent = BareLoop::new_with_managers(
            Arc::new(client),
            advertised,
            SessionConfig::default(),
            managers,
        );
        let run = agent
            .run("deploy to etc, srv, prod, and root", &RunConfig::default())
            .await;
        assert!(run.is_ok(), "the gated run completes: {run:?}");

        let recorded = collector.recorded();
        assert_eq!(
            recorded.len(),
            4,
            "one record per gated dispatch — deny, allow, ask approved, ask refused"
        );
        let (turn, call_id, denied) = recorded.first().cloned().expect("the denial is first");
        assert_eq!(turn, 0, "the denial's turn matches its dispatch");
        assert_eq!(
            call_id, "call_gate_1",
            "the record pairs with its tool call"
        );
        assert_eq!(denied.tool, "deploy");
        assert_eq!(denied.verdict, GateVerdict::Deny);
        assert_eq!(denied.rule_id, "middleware");
        assert_eq!(denied.rule_source, GateRuleSource::Middleware);
        assert_eq!(
            denied.reason.as_deref(),
            Some("etc is off limits"),
            "the refusal reason rides the emitted record"
        );
        assert_eq!(
            denied.args_digest,
            GateDecision::args_digest(&serde_json::json!({ "target": "/etc" })),
            "the digest is reproducible from the call's arguments"
        );
        assert_eq!(
            denied.ts, 1_700_000_000_000,
            "the engine stamped the record from its clock seam"
        );

        let (turn, call_id, allowed) = recorded.get(1).cloned().expect("the allowance is second");
        assert_eq!(turn, 1);
        assert_eq!(call_id, "call_gate_2");
        assert_eq!(allowed.verdict, GateVerdict::Allow);
        assert_eq!(allowed.rule_source, GateRuleSource::Middleware);
        assert_eq!(allowed.reason, None, "a plain allow needs no reason");

        let (turn, call_id, approved) =
            recorded.get(2).cloned().expect("the approved ask is third");
        assert_eq!(turn, 2);
        assert_eq!(call_id, "call_gate_3");
        assert_eq!(
            approved.verdict,
            GateVerdict::AskAllowed,
            "the resolver's approval is the decision the engine acted on — one record, not two"
        );
        assert_eq!(
            approved.rule_source,
            GateRuleSource::AskResolver,
            "the user's answer resolved the ask; the record says so"
        );
        assert_eq!(approved.rule_id, "ask");

        let (turn, call_id, refused) = recorded.get(3).cloned().expect("the refused ask is fourth");
        assert_eq!(turn, 3);
        assert_eq!(call_id, "call_gate_4");
        assert_eq!(refused.verdict, GateVerdict::AskDenied);
        assert_eq!(refused.rule_source, GateRuleSource::AskResolver);
        assert_eq!(refused.rule_id, "ask");
        assert_eq!(
            refused.reason.as_deref(),
            Some("denied by user"),
            "the refusal reason names the user's answer"
        );
        assert_eq!(
            (allowed.ts, approved.ts, refused.ts),
            (denied.ts, denied.ts, denied.ts),
            "the frozen clock stamps every record identically"
        );

        let order = collector.order();
        let kinds: Vec<&str> = order.iter().map(|(kind, _)| *kind).collect();
        assert_eq!(
            kinds,
            ["tool_pre", "gate", "tool_post"].repeat(4),
            "the gate decision fires after the pipeline returns and before the post event, \
             per dispatch"
        );
        let expected_ids = ["call_gate_1", "call_gate_2", "call_gate_3", "call_gate_4"];
        for (index, (_, call_id)) in order.iter().enumerate() {
            let expected_id = expected_ids[index / 3];
            assert_eq!(
                *call_id, expected_id,
                "the lifecycle events of one dispatch share its call id"
            );
        }
    }
}
