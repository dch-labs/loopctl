//! Pins for the shipped verifier and path extractor.
//!
//! [`CommandVerifier`] judges write-class calls statically — deny
//! patterns, shell allowlist, parse validity, path writability — and
//! [`WritePathExtractor`] feeds memoize invalidation with the paths a
//! call touches. Every pin here drives the public traits, hermetically
//! (canned tools, virtual paths, no execution of any verified
//! command).

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
use std::sync::Arc;

use loopctl::message::ToolContent;
use loopctl::middleware::{
    CommandVerifier, MemoizingMiddleware, PathExtractor, ToolDispatchContext, ToolDispatchResult,
    ToolMiddleware, ToolPipeline, Verifier, VerifyMiddleware, VerifyResult, WritePathExtractor,
};
use loopctl::tool::{
    PermissionCheck, Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema,
};
use serde_json::{Value, json};

/// A canned tool that counts its executions by name.
struct CountingTool {
    /// The registered tool name.
    name: &'static str,
    /// Shared execution counter for this tool.
    executions: Arc<std::sync::Mutex<usize>>,
}

impl Tool for CountingTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        "Counts its own executions"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name.to_string(),
            description: self.description().to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }
    }

    fn call(
        &self,
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let executions = Arc::clone(&self.executions);
        Box::pin(async move {
            *executions.lock().expect("execution counter lock") += 1;
            Ok(ToolOutput::text(format!("{} ran", self.name)))
        })
    }
}

/// One dispatch context for `tool` carrying `input`, under `cwd`.
fn ctx_for(tool: &str, input: Value, cwd: &str) -> ToolDispatchContext {
    ToolDispatchContext {
        tool_name: tool.to_string(),
        input,
        call_id: "c1".to_string(),
        turn_number: 0,
        cancel: Arc::new(loopctl::cancel::CancelSignal::new()),
        permission: PermissionCheck::Allow,
        tool_context: ToolContext {
            cwd: cwd.to_string(),
            ..ToolContext::default()
        },
    }
}

/// Run the verifier against one call and return its result.
async fn verify(input: Value) -> VerifyResult {
    let verifier = CommandVerifier::new();
    let ctx = ToolContext::default();
    verifier.verify_call(&ctx, "Write", &input).await
}

/// Run the verifier with a custom cwd against one call.
async fn verify_in(cwd: &str, input: Value) -> VerifyResult {
    let verifier = CommandVerifier::new();
    let ctx = ToolContext {
        cwd: cwd.to_string(),
        ..ToolContext::default()
    };
    verifier.verify_call(&ctx, "Write", &input).await
}

#[tokio::test]
async fn deny_globs_block_the_destructive_class() {
    for (command, expected) in [
        ("rm -rf /", "deny rule `rm -rf /`"),
        ("rm -fr /", "deny rule `rm -fr /`"),
        ("rm -rf /*", "deny rule `rm -rf /*`"),
        (":(){:|:&};:", "deny rule `:(){:|:&};:`"),
    ] {
        let result = verify(json!({"command": command})).await;
        assert!(
            !result.passed,
            "{command}: the destructive class must fail verification"
        );
        assert!(
            result.diagnostics.contains(expected),
            "{command}: the diagnostic names the rule and is actionable: {}",
            result.diagnostics
        );
    }

    // A wrapped payload is judged too — the deny check runs against
    // what the shell would run, not the wrapper.
    let result = verify(json!({"command": "sh -c 'rm -rf /'"})).await;
    assert!(!result.passed, "a wrapped root delete must fail");
    assert!(
        result.diagnostics.contains("deny rule"),
        "the wrapped denial names its rule: {}",
        result.diagnostics
    );

    // A wrapped command naming a non-allowed interpreter fails on the
    // allowlist, with the allowed set stated.
    let result = verify(json!({"command": "zsh -c 'echo hi'"})).await;
    assert!(!result.passed, "a non-allowed shell must fail");
    assert!(
        result.diagnostics.contains("zsh") && result.diagnostics.contains("bash"),
        "the shell diagnostic names the offender and the allowed set: {}",
        result.diagnostics
    );

    // Padding inside a wrapped payload cannot hide a rule: the shell
    // re-splits on whitespace runs, so the matcher normalizes the
    // command exactly as it normalizes the pattern.
    for padded in [
        "sh -c 'rm -rf  /'",
        "sh -c 'rm  -rf /'",
        "bash -c \"rm -rf\t/\"",
    ] {
        let result = verify(json!({"command": padded})).await;
        assert!(
            !result.passed,
            "{padded}: extra whitespace inside the payload must not bypass the deny rule"
        );
    }

    // Separator-adjacent and flag-split spellings of the same
    // destructive command deny identically: shell terminators and
    // combinators are token edges, and the split flag forms are listed
    // beside the compact ones.
    for evading in [
        "rm -rf /; echo done",
        "rm -rf /&&x",
        "(rm -rf /)",
        "rm -r -f /",
        "rm -f -r /",
        "rm --recursive --force /",
        "rm -r -f /*",
    ] {
        let result = verify(json!({"command": evading})).await;
        assert!(
            !result.passed,
            "{evading}: a separator-adjacent or flag-split spelling must not bypass the deny rule"
        );
    }
}

#[tokio::test]
async fn benign_writes_pass() {
    // The token-boundary proof: a deep recursive delete under /home is
    // a legitimate write, not the root-delete deny rule.
    let result = verify(json!({"command": "rm -rf /home/user/build"})).await;
    assert!(
        result.passed && result.diagnostics.is_empty(),
        "a scoped recursive delete passes with nothing to say: {result:?}"
    );

    let result = verify(json!({"command": "echo done > build/out.log"})).await;
    assert!(result.passed, "an ordinary redirected write passes");

    // An edit/write input whose parent exists (the temp dir) passes.
    let parent = std::env::temp_dir().join(format!("builtin-verifier-{}", std::process::id()));
    std::fs::create_dir_all(&parent).expect("the pin's parent directory is creatable");
    let result = verify_in(parent.to_str().unwrap_or("."), json!({"path": "notes.txt"})).await;
    assert!(
        result.passed && result.diagnostics.is_empty(),
        "a write into an existing parent passes: {result:?}"
    );

    // No recognizable call shape: nothing to judge.
    let result = verify(json!({"pattern": "x"})).await;
    assert!(result.passed, "a read-class call passes unconditionally");
}

#[tokio::test]
async fn unparseable_command_fails_soft_not_silent() {
    let result = verify(json!({"command": "echo \"unclosed"})).await;
    assert!(
        !result.passed,
        "an unparseable command must not silently pass"
    );
    assert!(
        result.diagnostics.contains("unparseable"),
        "the diagnostic names the class so the model can act: {}",
        result.diagnostics
    );
}

#[test]
fn extractor_reads_write_paths() {
    let extractor = WritePathExtractor;

    let paths = extractor.paths_with_cwd("Write", &json!({"path": "src/lib.rs"}), "/repo");
    assert_eq!(
        paths,
        vec!["/repo/src/lib.rs".to_string()],
        "a relative path resolves against the cwd"
    );

    let paths = extractor.paths_with_cwd(
        "Write",
        &json!({"file_path": "/etc/absolute.conf"}),
        "/repo",
    );
    assert_eq!(
        paths,
        vec!["/etc/absolute.conf".to_string()],
        "an absolute path rides through normalization"
    );

    let paths = extractor.paths_with_cwd(
        "Write",
        &json!({"filename": "../../shared/notes.md"}),
        "/repo/a/b",
    );
    assert_eq!(
        paths,
        vec!["/repo/shared/notes.md".to_string()],
        ".. segments resolve lexically without touching the filesystem"
    );

    let paths = extractor.paths_with_cwd("Grep", &json!({"pattern": "x"}), "/repo");
    assert!(
        paths.is_empty(),
        "a call with no path field and no command yields no paths"
    );
}

#[test]
fn shell_redirection_extraction() {
    let extractor = WritePathExtractor;

    let paths = extractor.paths_with_cwd("Bash", &json!({"command": "cmd > out.txt"}), "/repo");
    assert_eq!(
        paths,
        vec!["/repo/out.txt".to_string()],
        "a > redirection yields its target"
    );

    let paths = extractor.paths_with_cwd("Bash", &json!({"command": "cmd >> app.log"}), "/repo");
    assert_eq!(
        paths,
        vec!["/repo/app.log".to_string()],
        "a >> redirection yields its target"
    );

    let paths = extractor.paths_with_cwd("Bash", &json!({"command": "cmd 2>&1"}), "/repo");
    assert!(
        paths.is_empty(),
        "a descriptor duplication is not a file write"
    );

    let paths = extractor.paths_with_cwd("Bash", &json!({"command": "echo \"a > b\""}), "/repo");
    assert!(
        paths.is_empty(),
        "a quoted redirection character is not a redirection"
    );

    // Compact and fd-numbered forms carry the target on the operator
    // token itself.
    for (command, expected) in [
        ("echo hi >out.txt", "/repo/out.txt"),
        ("cmd 2>err.log", "/repo/err.log"),
        ("cmd &>all.log", "/repo/all.log"),
        ("cmd >&dup.log", "/repo/dup.log"),
        ("cmd 2>>append.log", "/repo/append.log"),
        ("cmd >>append.log", "/repo/append.log"),
        ("cmd 1>stdout.log", "/repo/stdout.log"),
        ("cmd 1>>stdout.log", "/repo/stdout.log"),
    ] {
        let paths = extractor.paths_with_cwd("Bash", &json!({"command": command}), "/repo");
        assert_eq!(
            paths,
            vec![expected.to_string()],
            "{command}: the compact/fd form extracts its inline target"
        );
    }

    // A redirection outside the wrapper evicts too: the wrapper's own
    // output is as much a write as the payload's.
    let paths = extractor.paths_with_cwd(
        "Bash",
        &json!({"command": "sh -c 'echo hi' > out.txt"}),
        "/repo",
    );
    assert_eq!(
        paths,
        vec!["/repo/out.txt".to_string()],
        "an outer redirection survives the wrapper"
    );

    // Both levels at once: outer and inner targets are extracted
    // together.
    let paths = extractor.paths_with_cwd(
        "Bash",
        &json!({"command": "sh -c 'x > inner.txt' > outer.txt"}),
        "/repo",
    );
    let mut sorted = paths;
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["/repo/inner.txt".to_string(), "/repo/outer.txt".to_string()],
        "outer and inner redirections are one write set"
    );

    // A wrapped payload's redirections are seen through the wrapper.
    for (command, expected) in [
        ("sh -c 'echo hi > out.txt'", "/repo/out.txt"),
        ("bash -c 'x >> log.txt'", "/repo/log.txt"),
    ] {
        let paths = extractor.paths_with_cwd("Bash", &json!({"command": command}), "/repo");
        assert_eq!(
            paths,
            vec![expected.to_string()],
            "{command}: one level of shell wrapping is unwrapped"
        );
    }

    // Descriptor duplications name no file, in either direction.
    for command in ["cmd 2>&1", "cmd >&2", "cmd 2> &1"] {
        let paths = extractor.paths_with_cwd("Bash", &json!({"command": command}), "/repo");
        assert!(
            paths.is_empty(),
            "{command}: a descriptor duplication is not a file write"
        );
    }
}

#[tokio::test]
async fn write_invalidates_cached_read() {
    let read_executions = Arc::new(std::sync::Mutex::new(0usize));
    let write_executions = Arc::new(std::sync::Mutex::new(0usize));
    let mut registry = ToolRegistry::new();
    registry.register(CountingTool {
        name: "Read",
        executions: Arc::clone(&read_executions),
    });
    registry.register(CountingTool {
        name: "Write",
        executions: Arc::clone(&write_executions),
    });
    let pipeline = ToolPipeline::builder()
        .with_middleware(MemoizingMiddleware::new(
            vec!["Read".to_string()],
            vec!["Write".to_string()],
            Arc::new(WritePathExtractor),
            5,
        ))
        .with_core(Arc::new(registry))
        .build()
        .expect("the pin's pipeline builds");

    // The same file under two spellings: the read arrives as a
    // relative path under one cwd, the write as a different relative
    // spelling under another — they match only through lexical
    // normalization against each call's working directory.
    let root = format!("/pin/{}", std::process::id());
    let read_root = format!("{root}/a");
    let write_cwd = format!("{root}/b");
    let read_input = json!({"path": "memo.txt"});

    let mut ctx = ctx_for("Read", read_input.clone(), &read_root);
    pipeline.dispatch(&mut ctx).await;
    let mut ctx = ctx_for("Read", read_input.clone(), &read_root);
    pipeline.dispatch(&mut ctx).await;
    assert_eq!(
        *read_executions.lock().unwrap(),
        1,
        "the second read is served from the cache"
    );

    let write_input = json!({"path": "../a/memo.txt", "content": "new"});
    let mut ctx = ctx_for("Write", write_input, &write_cwd);
    pipeline.dispatch(&mut ctx).await;
    let mut ctx = ctx_for("Read", read_input, &read_root);
    pipeline.dispatch(&mut ctx).await;
    assert_eq!(
        *read_executions.lock().unwrap(),
        2,
        "a write to the cached path invalidates it: the next read executes"
    );
    assert_eq!(
        *write_executions.lock().unwrap(),
        1,
        "the write itself executed exactly once"
    );
}

#[tokio::test]
async fn profile_wiring_is_explicit_this_release() {
    let mut registry = ToolRegistry::new();
    registry.register(CountingTool {
        name: "Write",
        executions: Arc::new(std::sync::Mutex::new(0usize)),
    });

    // The default construction still wires the no-op verifier: a
    // destructive command rides through as a pass. The flip is a
    // later, deliberately separate change — this pin exists so that
    // flip shows up as a seen diff.
    let default_pipeline = loopctl::presets::ConstrainedProfile::pipeline_builder()
        .with_core(Arc::new(registry))
        .build()
        .expect("the default profile pipeline builds");
    let mut ctx = ctx_for("Write", json!({"command": "rm -rf /"}), ".");
    let result = default_pipeline.dispatch(&mut ctx).await;
    assert!(
        !result.is_error,
        "verification failure is a soft verdict, never an error result"
    );
    assert!(
        !result.output_text().contains("[verify] failed"),
        "the 0.3.x default profile does not verify commands: {}",
        result.output_text()
    );

    // The explicit builtin-verified variant fails the same call, with
    // an actionable diagnostic riding the output.
    let mut registry = ToolRegistry::new();
    registry.register(CountingTool {
        name: "Write",
        executions: Arc::new(std::sync::Mutex::new(0usize)),
    });
    let verified_pipeline =
        loopctl::presets::ConstrainedProfile::pipeline_builder_with_builtin_verification()
            .with_core(Arc::new(registry))
            .build()
            .expect("the verified profile pipeline builds");
    let mut ctx = ctx_for("Write", json!({"command": "rm -rf /"}), ".");
    let result = verified_pipeline.dispatch(&mut ctx).await;
    assert!(
        result.output_text().contains("[verify] failed")
            && result.output_text().contains("deny rule"),
        "the verified variant fails the destructive command with its rule named: {}",
        result.output_text()
    );
}

/// Render a dispatch result's text for assertion messages and checks.
trait OutputText {
    /// The result's text content, parts joined.
    fn output_text(&self) -> String;
}

impl OutputText for loopctl::middleware::ToolDispatchResult {
    fn output_text(&self) -> String {
        match &self.output {
            ToolContent::Text(text) => text.clone(),
            ToolContent::Multipart(parts) => parts
                .iter()
                .map(|part| match part {
                    loopctl::message::ToolContentPart::Text { text } => text.clone(),
                    _ => String::new(),
                })
                .collect(),
        }
    }
}

#[tokio::test]
async fn shell_redirection_invalidates_cached_read_through_the_preset() {
    let read_executions = Arc::new(std::sync::Mutex::new(0usize));
    let bash_executions = Arc::new(std::sync::Mutex::new(0usize));
    let mut registry = ToolRegistry::new();
    registry.register(CountingTool {
        name: "Read",
        executions: Arc::clone(&read_executions),
    });
    registry.register(CountingTool {
        name: "Bash",
        executions: Arc::clone(&bash_executions),
    });
    let pipeline =
        loopctl::presets::ConstrainedProfile::pipeline_builder_with_builtin_verification()
            .with_core(Arc::new(registry))
            .build()
            .expect("the preset pipeline builds");

    let root = format!("/pin/{}", std::process::id());
    let read_input = json!({"path": "memo.txt"});

    let mut ctx = ctx_for("Read", read_input.clone(), &root);
    pipeline.dispatch(&mut ctx).await;
    let mut ctx = ctx_for("Read", read_input.clone(), &root);
    pipeline.dispatch(&mut ctx).await;
    assert_eq!(
        *read_executions.lock().unwrap(),
        1,
        "the second read is served from the cache"
    );

    let bash_input = json!({"command": format!("echo new-content > {root}/memo.txt")});
    let mut ctx = ctx_for("Bash", bash_input, &root);
    let result = pipeline.dispatch(&mut ctx).await;
    assert!(
        result.output_text().contains("[verify] failed")
            || result.output_text().contains("[verify] passed"),
        "the Bash write carries a verify block: {}",
        result.output_text()
    );

    let mut ctx = ctx_for("Read", read_input, &root);
    pipeline.dispatch(&mut ctx).await;
    assert_eq!(
        *read_executions.lock().unwrap(),
        2,
        "a shell redirection to the cached path invalidates it through the preset wiring"
    );
}

/// A middleware that rewrites the dispatch input before the tool runs.
struct InputRewriter;

impl ToolMiddleware for InputRewriter {
    fn name(&self) -> &'static str {
        "input_rewriter"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        Box::pin(async move {
            ctx.input = json!({"command": "echo rewritten"});
            next.dispatch(ctx).await
        })
    }
}

#[tokio::test]
async fn verification_judges_the_input_as_sent_not_as_rewritten() {
    let mut registry = ToolRegistry::new();
    registry.register(CountingTool {
        name: "Bash",
        executions: Arc::new(std::sync::Mutex::new(0usize)),
    });
    let pipeline = ToolPipeline::builder()
        .with_middleware(VerifyMiddleware::new(
            Arc::new(CommandVerifier::new()),
            vec!["Bash".to_string()],
        ))
        .with_middleware(InputRewriter)
        .with_core(Arc::new(registry))
        .build()
        .expect("the pin's pipeline builds");

    let mut ctx = ctx_for("Bash", json!({"command": "rm -rf /"}), ".");
    let result = pipeline.dispatch(&mut ctx).await;
    assert!(
        result.output_text().contains("[verify] failed")
            && result.output_text().contains("deny rule"),
        "the verifier judges the input the model sent, not an inner rewrite: {}",
        result.output_text()
    );
}

#[tokio::test]
async fn the_default_cwd_checks_the_real_parent() {
    // ToolContext::default() carries cwd "."; with the root-anchoring
    // fixed, a relative target resolves against the process's working
    // directory (the crate root under cargo test), so an existing file
    // passes and a genuinely missing parent fails — not /.
    let verifier = CommandVerifier::new();
    let ctx = ToolContext::default();
    let result = verifier
        .verify_call(&ctx, "Write", &json!({"path": "Cargo.toml"}))
        .await;
    assert!(
        result.passed,
        "the crate manifest's parent exists — the default cwd checks the \
         real directory: {result:?}"
    );
    let result = verifier
        .verify_call(&ctx, "Write", &json!({"path": "no-such-dir/x.txt"}))
        .await;
    assert!(
        !result.passed && result.diagnostics.contains("no-such-dir"),
        "a missing parent under the real cwd fails with its name: {result:?}"
    );
}

#[tokio::test]
async fn root_target_gets_a_coherent_diagnostic() {
    // "/" is the root target; ".." is not — under a resolved cwd it
    // names the parent of the working directory, a real directory.
    for path in ["/"] {
        let result = verify(json!({"path": path})).await;
        assert!(
            !result.passed,
            "{path}: the filesystem root is never a writable file target"
        );
        assert!(
            result.diagnostics.contains("filesystem root"),
            "{path}: the root case is named, not an empty parent path: {}",
            result.diagnostics
        );
    }
}

#[tokio::test]
async fn builder_options_replace_their_sets() {
    // with_deny_globs replaces: the custom rule denies, and the
    // default root-delete rule no longer does.
    let verifier = CommandVerifier::new().with_deny_globs(vec!["deploy prod".to_string()]);
    let result = verifier
        .verify_call(
            &ToolContext::default(),
            "Bash",
            &json!({"command": "deploy prod"}),
        )
        .await;
    assert!(!result.passed, "the custom deny rule fires");

    let result = verifier
        .verify_call(
            &ToolContext::default(),
            "Bash",
            &json!({"command": "rm -rf /"}),
        )
        .await;
    assert!(
        result.passed,
        "replacement semantics: the defaults are gone once the set is replaced"
    );

    // with_allowed_shells replaces: zsh passes, the default bash now
    // fails the allowlist.
    let verifier = CommandVerifier::new().with_allowed_shells(vec!["zsh".to_string()]);
    let result = verifier
        .verify_call(
            &ToolContext::default(),
            "Bash",
            &json!({"command": "zsh -c 'echo hi'"}),
        )
        .await;
    assert!(result.passed, "the custom allowlist admits its shell");

    let result = verifier
        .verify_call(
            &ToolContext::default(),
            "Bash",
            &json!({"command": "bash -c 'echo hi'"}),
        )
        .await;
    assert!(
        !result.passed,
        "replacement semantics: the default shells are gone once the list is replaced"
    );
}
