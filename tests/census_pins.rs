//! Census coverage pins: deliberate behaviors the round-3 census
//! (CODE-REVIEW-ROUND3.md §2–3) flagged as tested-nowhere, locked as
//! rows 1–5 of the L-88 test plan. Each test's name is the behavior
//! sentence it pins.
//!
//! Requires `testing`, `tool_health`, and `redaction`.

#![cfg(all(feature = "testing", feature = "tool_health", feature = "redaction"))]
#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::message::MessagePart;
use loopctl::middleware::redaction::{RedactingMiddleware, SecretPattern, SecretPatternSet};
use loopctl::middleware::verify::{Verifier, VerifyMiddleware, VerifyResult};
use loopctl::middleware::{
    MemoizingMiddleware, NoopPathExtractor, PermissionMiddleware, ToolDispatchContext,
    ToolDispatchResult, ToolMiddleware, ToolPipeline,
};
use loopctl::presets::ConstrainedProfile;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::health::ToolHealthRegistry;
use loopctl::tool::{Tool, ToolContext, ToolOutput, ToolRegistry};

fn scripted(calls: &[(&str, &str, serde_json::Value)]) -> Vec<MockResponse> {
    let mut responses = Vec::new();
    for (id, tool, input) in calls {
        responses.push(MockResponse {
            text: "go".to_string(),
            tool_call: Some(MockToolCall {
                id: id.to_string(),
                name: tool.to_string(),
                input: input.clone(),
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

fn result_texts(loop_: &BareLoop<MockApiClient>) -> Vec<String> {
    loop_
        .conversation()
        .iter()
        .flat_map(|m| m.parts.iter())
        .filter_map(|p| match p {
            MessagePart::ToolResult { output, .. } => Some(output.to_string()),
            _ => None,
        })
        .collect()
}

/// A tool with a fixed name and output, counting its executions.
struct FixedTool {
    /// The registered name.
    name: &'static str,
    /// The output every call returns.
    output: &'static str,
    /// One entry per execution.
    executions: Arc<Mutex<Vec<()>>>,
    /// Artificial work time per execution.
    delay: Duration,
}

impl Tool for FixedTool {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> &'static str {
        "test fixture"
    }
    fn schema(&self) -> loopctl::tool::ToolSchema {
        loopctl::tool::ToolSchema {
            tool: self.name.to_string(),
            description: "test fixture".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send + '_>>
    {
        let executions = Arc::clone(&self.executions);
        let output = self.output;
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            executions.lock().expect("exec lock").push(());
            Ok(ToolOutput::text(output))
        })
    }
}

/// Rewrites every call's tool name before inner layers see it.
struct RenamingMiddleware {
    /// The name every call is rewritten to.
    new_name: String,
}

impl ToolMiddleware for RenamingMiddleware {
    fn name(&self) -> &'static str {
        "renaming"
    }
    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        ctx.tool_name = self.new_name.clone();
        Box::pin(async move { next.dispatch(ctx).await })
    }
}

/// A verifier whose diagnostics embed a fixed secret, standing in for
/// a compiler echoing the written content.
struct SecretEchoVerifier {
    /// The string embedded in every diagnostics block.
    secret: &'static str,
}

impl Verifier for SecretEchoVerifier {
    fn verify<'a>(
        &'a self,
        _ctx: &'a ToolContext,
        _tool_name: &str,
    ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>> {
        let secret = self.secret;
        Box::pin(async move {
            VerifyResult {
                passed: true,
                diagnostics: format!("compiled with {secret}"),
            }
        })
    }
}

/// Build a loop with the given pipeline and mock script.
fn loop_with_pipeline(
    registry: ToolRegistry,
    pipeline: loopctl::middleware::ToolPipelineBuilder,
    calls: &[(&str, &str, serde_json::Value)],
) -> BareLoop<MockApiClient> {
    let client = MockApiClient::new("m").with_responses(scripted(calls));
    let mut loop_ = BareLoop::new(Arc::new(client), registry, SessionConfig::default());
    loop_
        .set_pipeline(pipeline)
        .expect("static pipeline composition is valid");
    loop_
}

#[tokio::test]
async fn permission_denials_count_toward_the_breaker() {
    let executions = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(FixedTool {
        name: "Bash",
        output: "ran",
        executions: Arc::clone(&executions),
        delay: Duration::ZERO,
    });
    let calls: Vec<(&str, &str, serde_json::Value)> = (0..3)
        .map(|i| ("c", "Bash", serde_json::json!({"cmd": i})))
        .collect();
    let mut loop_ = loop_with_pipeline(
        registry,
        loopctl::middleware::ToolPipeline::builder()
            .with_middleware(PermissionMiddleware::deny_all()),
        &calls,
    );
    let health = Arc::new(ToolHealthRegistry::new().with_config(
        loopctl::tool::health::CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_duration: Duration::from_secs(60),
            probe_timeout: Duration::from_secs(60),
        },
    ));
    loop_.set_health_registry(Arc::clone(&health));

    loop_
        .run("try anyway", &RunConfig::default())
        .await
        .expect("run completes");

    assert!(
        executions.lock().expect("exec lock").is_empty(),
        "the policy denies every call before execution"
    );
    let texts = result_texts(&loop_);
    let denials = texts.iter().filter(|t| t.contains("Permission")).count();
    let breaker = texts
        .iter()
        .filter(|t| t.contains("temporarily unavailable"))
        .count();
    assert_eq!(
        (denials, breaker),
        (2, 1),
        "two policy denials trip the breaker (threshold 2), the third call is \\
         refused by the gate — one counting rule for all policy refusals, \\
         symmetric with shield blocks: {texts:?}"
    );
}

#[tokio::test]
async fn preset_order_cap_verify_memoize_replays_cached_without_verify() {
    let write_execs = Arc::new(Mutex::new(Vec::new()));
    let read_execs = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(FixedTool {
        name: "Write",
        output: "wrote",
        executions: Arc::clone(&write_execs),
        delay: Duration::ZERO,
    });
    registry.register(FixedTool {
        name: "Read",
        output: "file contents",
        executions: Arc::clone(&read_execs),
        delay: Duration::ZERO,
    });

    let calls = [
        ("w1", "Write", serde_json::json!({"path": "a.txt"})),
        ("r1", "Read", serde_json::json!({"path": "a.txt"})),
        ("r2", "Read", serde_json::json!({"path": "a.txt"})),
    ];
    let client = MockApiClient::new("m").with_responses(scripted(&calls));
    let mut loop_ = BareLoop::new(
        Arc::new(client),
        registry,
        ConstrainedProfile::session_config(),
    );
    // request_options() carries a tool-call constraint the mock client
    // cannot honor; apply() alone wires the pipeline under test.
    ConstrainedProfile::apply(&mut loop_).expect("profile applies");

    loop_
        .run("write then read twice", &RunConfig::default())
        .await
        .expect("run completes");

    let texts = result_texts(&loop_);
    assert!(
        texts.iter().any(|t| t.contains("[verify]")),
        "the write is verified — verify sits inside the composition and \\
         fires on live write calls: {texts:?}"
    );
    assert_eq!(
        read_execs.lock().expect("exec lock").len(),
        1,
        "the second read is served from the memoization cache"
    );
    let replay = texts.iter().filter(|t| t.contains("[cached]")).count();
    assert_eq!(replay, 1, "the replayed read carries the [cached] marker");
    let cached_verify = texts
        .iter()
        .filter(|t| t.contains("[cached]") && t.contains("[verify]"))
        .count();
    assert_eq!(
        cached_verify, 0,
        "memoize sits innermost — it cached the pre-verify output, so the \\
         replay never sees a verify block"
    );
}

#[tokio::test]
async fn redaction_composes_outside_verify_and_memoize() {
    const SECRET: &str = "hunter2-supersecret";
    let write_execs = Arc::new(Mutex::new(Vec::new()));
    let read_execs = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(FixedTool {
        name: "Write",
        output: "wrote",
        executions: Arc::clone(&write_execs),
        delay: Duration::ZERO,
    });
    registry.register(FixedTool {
        name: "Read",
        output: SECRET,
        executions: Arc::clone(&read_execs),
        delay: Duration::ZERO,
    });

    let patterns = SecretPatternSet::default_common().with_pattern(SecretPattern {
        kind: "test_secret",
        pattern: regex::Regex::new(SECRET).expect("literal pattern compiles"),
    });
    let calls = [
        ("w1", "Write", serde_json::json!({"path": "a"})),
        ("r1", "Read", serde_json::json!({"path": "a"})),
        ("r2", "Read", serde_json::json!({"path": "a"})),
    ];
    let mut loop_ = loop_with_pipeline(
        registry,
        loopctl::middleware::ToolPipeline::builder()
            .with_middleware(RedactingMiddleware::new(patterns))
            .with_middleware(VerifyMiddleware::new(
                Arc::new(SecretEchoVerifier { secret: SECRET }),
                vec!["Write".to_string()],
            ))
            .with_middleware(MemoizingMiddleware::new(
                vec!["Read".to_string()],
                Vec::new(),
                Arc::new(NoopPathExtractor),
                5,
            )),
        &calls,
    );

    loop_
        .run("write then read twice", &RunConfig::default())
        .await
        .expect("run completes");

    let texts = result_texts(&loop_);
    assert!(
        texts.iter().any(|t| t.contains("[REDACTED:test_secret]")),
        "redaction is the outermost layer and scrubs what inner layers emit"
    );
    for (i, text) in texts.iter().enumerate() {
        assert!(
            !text.contains(SECRET),
            "the secret never reaches the conversation (result {i}): {text}"
        );
    }
    assert!(
        texts
            .iter()
            .any(|t| t.contains("[cached]") && t.contains("[REDACTED:test_secret]")),
        "the memoized replay hit is redacted again on the way out — the \\
         cache stores the raw inner output, redaction re-applies per dispatch"
    );
}

#[tokio::test]
async fn memoize_hits_grow_health_with_the_original_duration() {
    let read_execs = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(FixedTool {
        name: "Read",
        output: "payload",
        executions: Arc::clone(&read_execs),
        delay: Duration::from_millis(30),
    });

    let calls = [
        ("r1", "Read", serde_json::json!({"path": "a"})),
        ("r2", "Read", serde_json::json!({"path": "a"})),
    ];
    let mut loop_ = loop_with_pipeline(
        registry,
        loopctl::middleware::ToolPipeline::builder().with_middleware(MemoizingMiddleware::new(
            vec!["Read".to_string()],
            Vec::new(),
            Arc::new(NoopPathExtractor),
            5,
        )),
        &calls,
    );
    let health = Arc::new(ToolHealthRegistry::new());
    loop_.set_health_registry(Arc::clone(&health));

    loop_
        .run("read twice", &RunConfig::default())
        .await
        .expect("run completes");

    let summary = health.health_summary();
    let (status, rate) = summary
        .get("Read")
        .unwrap_or_else(|| panic!("Read recorded: {summary:?}"));
    assert_eq!(
        *rate, 1.0,
        "the cache hit is a dispatch and records as a success — a failed \\
         or skipped recording would drop the rate"
    );
    assert_eq!(
        *status,
        loopctl::tool::health::HealthStatus::Healthy,
        "two successes (one live, one cached-with-original-duration) keep \\
         the tool healthy"
    );
    // The original-duration half of the row is pinned at the unit level:
    // cache_hit_preserves_the_original_call_duration (memoize.rs).
}

#[tokio::test]
async fn verify_matches_write_tools_by_resolved_name() {
    let execs = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    // The called name must pass the availability gate, so the alias is
    // registered too; the inner renamer rewrites it to the canonical
    // write tool before the registry dispatch, and the executed name
    // rides back on the result.
    registry.register(FixedTool {
        name: "my-write",
        output: "unreachable — the rewrite diverts execution",
        executions: Arc::new(Mutex::new(Vec::new())),
        delay: Duration::ZERO,
    });
    registry.register(FixedTool {
        name: "Write",
        output: "done",
        executions: Arc::clone(&execs),
        delay: Duration::ZERO,
    });

    // The model calls the alias; verify (outer) sees the alias name —
    // not a write tool — and must fall back to the result's resolved
    // name, which carries the canonical "Write" the inner renamer
    // executed.
    let calls = vec![("w1", "my-write", serde_json::json!({"path": "a"}))];
    let mut loop_ = loop_with_pipeline(
        registry,
        loopctl::middleware::ToolPipeline::builder()
            .with_middleware(VerifyMiddleware::new(
                Arc::new(loopctl::middleware::verify::NoopVerifier),
                vec!["Write".to_string()],
            ))
            .with_middleware(RenamingMiddleware {
                new_name: "Write".to_string(),
            }),
        &calls,
    );

    loop_
        .run("write via alias", &RunConfig::default())
        .await
        .expect("run completes");

    let texts = result_texts(&loop_);
    assert!(
        texts.iter().any(|t| t.contains("[verify]")),
        "verify recognizes the write through the call's resolved name even \\
         when an outer middleware rewrote the dispatch name: {texts:?}"
    );
}
