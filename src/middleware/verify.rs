//! Post-execution verification middleware.
//!
//! [`VerifyMiddleware`] runs a caller-supplied [`Verifier`] after
//! configured write-class tools and appends the pass/fail result +
//! diagnostics to the [`ToolOutput`](crate::tool::ToolOutput) so the
//! next turn sees it. This is the structural mechanism that makes
//! "verify-on-write" automatic instead of relying on the model to decide
//! to check its own work — the right default for small models, which
//! ship edits that don't compile far more often than frontier models.
//!
//! The *mechanism* (run a verifier after a write tool, append result to
//! output) lives here. The *verifier impl* (`cargo check`, `tsc`) is
//! domain-specific and supplied by the consumer.
//!
//! # Registration
//!
//! Register `VerifyMiddleware` before `.core(registry)` in the pipeline
//! so it wraps the result on the way out. To bound diagnostics size,
//! pair it with [`OutputLimitMiddleware`](super::OutputLimitMiddleware)
//! registered *after* `VerifyMiddleware`, so the limit wraps the
//! verify-appended output too.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::message::{ToolContent, ToolContentPart};
use crate::tool::ToolContext;

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};

// ===================================================
// Verifier trait + VerifyResult
// ===================================================

/// A pluggable verifier run after write-class tools.
///
/// Implement this trait to supply the verification logic for
/// [`VerifyMiddleware`] — typically a build / lint / typecheck command
/// run against the project state the write tool just modified.
/// Domain-specific impls (e.g. `cargo check`, `tsc`) live in the
/// consumer crate; loopctl ships only the trait and the middleware.
///
/// # Side effects
///
/// Impl authors should assume the verifier runs after *every*
/// configured write tool call. A verifier like `cargo check` writes to
/// `target/` as a side effect of running; that's expected, but it means
/// the verifier is not free and should be chosen with care.
///
/// # Diagnostics on failure
///
/// Impl authors should populate
/// [`diagnostics`](VerifyResult::diagnostics) with something useful even
/// when verification fails — the middleware appends whatever it receives
/// verbatim to the tool output, and an empty diagnostics string on a
/// `passed: false` result gives the model nothing to act on.
///
/// # Async
///
/// The `verify` method is async-boxed (`Pin<Box<dyn Future + Send>>`),
/// mirroring [`Tool::call`](crate::tool::Tool::call). This keeps the
/// runtime non-blocking for subprocess-based verifiers (`cargo check`,
/// `tsc`) that take seconds, without forcing every impl to hand-roll
/// `spawn_blocking`.
pub trait Verifier: Send + Sync {
    /// Run the verifier against the project state after a write tool.
    ///
    /// `ctx` is the same [`ToolContext`] the tool itself received
    /// (carrying `cwd`, `session_id`, `temp_dir`, extensions);
    /// `tool_name` is the name of the write tool that just ran, in case
    /// the verifier wants to vary its behavior by tool.
    ///
    /// # Returns
    ///
    /// A [`VerifyResult`] carrying `passed` + `diagnostics`. The
    /// middleware appends this to the tool output; it does not interpret
    /// the diagnostics or gate further dispatch on `passed`.
    fn verify<'a>(
        &'a self,
        ctx: &'a ToolContext,
        tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>>;
}

/// The outcome of a verification run.
///
/// Produced by [`Verifier::verify`] and appended to the write tool's
/// [`ToolOutput`](crate::tool::ToolOutput) by [`VerifyMiddleware`].
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Whether the verifier considers the project state valid.
    ///
    /// `true` for "checks pass, the write is safe to build on"; `false`
    /// for "the project is in a broken state the model should address."
    /// The middleware does not gate dispatch on this — it appends the
    /// result and lets the next turn decide.
    pub passed: bool,

    /// Human-readable diagnostics from the verifier.
    ///
    /// Typically the compiler/linter output that explains *why* a
    /// `passed: false` result occurred. Surfaced verbatim in the tool
    /// output so the model can read it on the next turn. Keep this
    /// bounded — pair [`VerifyMiddleware`] with
    /// [`OutputLimitMiddleware`](super::OutputLimitMiddleware) to cap
    /// total output size.
    pub diagnostics: String,
}

/// A [`Verifier`] that always passes with empty diagnostics.
///
/// Useful as the default verifier when wiring [`VerifyMiddleware`] into a
/// pipeline that has no real build/lint step to run — the middleware still
/// registers and appends a `[verify] passed: ` block, but no actual
/// check happens. Swap in a real verifier (`cargo check`, `tsc`) when one is
/// available.
///
/// Zero-sized; cheap to construct and `Arc`-share.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use loopctl::middleware::{NoopVerifier, Verifier, VerifyMiddleware};
///
/// let pipeline = ToolPipeline::builder()
///     .with(VerifyMiddleware::new(Arc::new(NoopVerifier), vec!["Write".into()]))
///     .core(registry)
///     .build()?;
/// ```
pub struct NoopVerifier;

impl Verifier for NoopVerifier {
    fn verify<'a>(
        &'a self,
        _ctx: &'a ToolContext,
        _tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>> {
        Box::pin(async move {
            VerifyResult {
                passed: true,
                diagnostics: String::new(),
            }
        })
    }
}

// ===================================================
// VerifyMiddleware
// ===================================================

/// Post-execution middleware that runs a [`Verifier`] after configured
/// write-class tools.
///
/// Holds a shared verifier handle and a list of write-class tool names.
/// After the inner tool runs, if the tool is in the write list **and**
/// the tool did not itself error, the middleware calls the verifier and
/// appends a `[verify] passed: …` or `[verify] failed: …` block to the
/// tool's output. The next turn sees the block and can act on it.
///
/// The middleware never marks a successful write as an error when
/// verification fails — the model sees the diagnostics in plain text
/// and decides whether to retry the write, undo it, or proceed.
///
/// # Registration
///
/// Register *before* `.core(registry)` so the middleware wraps the
/// result on the way out:
///
/// ```rust,ignore
/// use loopctl::middleware::{VerifyMiddleware, ToolPipeline};
/// use std::sync::Arc;
///
/// let verifier: Arc<dyn loopctl::middleware::Verifier> = /* … */;
/// let pipeline = ToolPipeline::builder()
///     .with(VerifyMiddleware::new(verifier, vec!["Write".into(), "Edit".into()]))
///     .core(registry)
///     .build()?;
/// ```
///
/// To bound diagnostics size, also register
/// [`OutputLimitMiddleware`](super::OutputLimitMiddleware) *after*
/// `VerifyMiddleware`.
pub struct VerifyMiddleware {
    /// The verifier this middleware invokes after write-class tool calls.
    ///
    /// Held as a shared `Arc<dyn Verifier>` handle so a single
    /// verifier instance can back multiple middleware layers if a future
    /// composition wants that — e.g. one `cargo check` verifier reused
    /// across parallel pipelines. The handle is cloned cheaply per
    /// dispatch (the middleware itself is shared via
    /// `Arc<dyn ToolMiddleware>` in the pipeline, so this inner `Arc`
    /// only bumps a refcount).
    verifier: Arc<dyn Verifier>,

    /// Exact-match tool names that trigger verification.
    ///
    /// A tool not in this list bypasses the verifier entirely. List
    /// members are compared verbatim against the tool name the call
    /// *resolved to* after dispatch
    /// (`ToolDispatchResult::resolved_tool_name`, falling back to the
    /// originally requested name when the result was built without one),
    /// so a routing middleware that redirects a write-class call keeps
    /// verification working as long as the redirected name is listed.
    /// There is deliberately no glob or regex
    /// support — callers list tool names explicitly.
    write_tools: Vec<String>,
}

impl VerifyMiddleware {
    /// Construct a verify middleware.
    ///
    /// `verifier` is held as a shared handle ([`Arc<dyn Verifier>`]),
    /// so one verifier can back multiple middleware instances if a
    /// future composition wants that. `write_tools` is the
    /// exact-match list of tool names that trigger verification — tools
    /// not in the list pass through unchanged.
    #[must_use]
    pub fn new(verifier: Arc<dyn Verifier>, write_tools: Vec<String>) -> Self {
        Self {
            verifier,
            write_tools,
        }
    }
}

impl std::fmt::Debug for VerifyMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyMiddleware")
            .field("verifier", &"<dyn Verifier>")
            .field("write_tools", &self.write_tools)
            .finish()
    }
}

impl ToolMiddleware for VerifyMiddleware {
    fn name(&self) -> &'static str {
        "verify"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let verifier = &self.verifier;
        let write_tools = &self.write_tools;
        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;
            let resolved = if result.resolved_tool_name.is_empty() {
                &ctx.tool_name
            } else {
                &result.resolved_tool_name
            };

            let is_write = write_tools.iter().any(|t| t == resolved);
            if !is_write || result.is_error {
                return result;
            }

            let verify = verifier.verify(&ctx.tool_context, resolved).await;
            append_verify_result(&mut result.output, &verify);
            result
        })
    }
}

/// Append a `[verify]` block to a [`ToolContent`].
///
/// For [`ToolContent::Text`] the block is appended to the existing
/// string. For [`ToolContent::Multipart`] a new
/// [`ToolContentPart::Text`] carrying the block is pushed; existing
/// parts are left untouched.
fn append_verify_result(output: &mut ToolContent, verify: &VerifyResult) {
    let status = if verify.passed { "passed" } else { "failed" };
    let block = format!("\n\n[verify] {status}: {}", verify.diagnostics);
    match output {
        ToolContent::Text(s) => s.push_str(&block),
        ToolContent::Multipart(parts) => {
            parts.push(ToolContentPart::Text {
                text: block.trim_start().to_string(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ToolContent;
    use crate::middleware::{ToolDispatchContext, ToolPipeline};
    use crate::tool::{PermissionCheck, ToolContext, ToolRegistry};
    use std::sync::Arc;

    /// A `Verifier` impl that returns a fixed `VerifyResult`, recording
    /// whether it was called.
    struct CannedVerifier {
        result: VerifyResult,
        called: Arc<std::sync::Mutex<bool>>,
    }

    impl Verifier for CannedVerifier {
        fn verify<'a>(
            &'a self,
            _ctx: &'a ToolContext,
            _tool_name: &'a str,
        ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>> {
            let result = self.result.clone();
            let called = self.called.clone();
            Box::pin(async move {
                *called.lock().unwrap() = true;
                result
            })
        }
    }

    fn make_verifier(
        passed: bool,
        diagnostics: &str,
    ) -> (Arc<CannedVerifier>, Arc<std::sync::Mutex<bool>>) {
        let called = Arc::new(std::sync::Mutex::new(false));
        let v = Arc::new(CannedVerifier {
            result: VerifyResult {
                passed,
                diagnostics: diagnostics.to_string(),
            },
            called: called.clone(),
        });
        (v, called)
    }

    /// A middleware that short-circuits with a fixed result.
    struct FixedOutputMiddleware {
        output: ToolContent,
        is_error: bool,
    }

    impl ToolMiddleware for FixedOutputMiddleware {
        fn name(&self) -> &'static str {
            "fixed_output"
        }
        fn dispatch<'a>(
            &'a self,
            _ctx: &'a mut ToolDispatchContext,
            _next: &'a ToolPipeline,
        ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
            let output = self.output.clone();
            let is_error = self.is_error;
            Box::pin(async move {
                ToolDispatchResult {
                    output,
                    is_error,
                    resolved_tool_name: String::new(),
                    tool_call_id: String::new(),
                    duration: std::time::Duration::ZERO,
                }
            })
        }
    }

    fn pipeline_with(
        verify: VerifyMiddleware,
        output: ToolContent,
        is_error: bool,
    ) -> ToolPipeline {
        let registry = Arc::new(ToolRegistry::new());
        ToolPipeline::builder()
            .with(verify)
            .with(FixedOutputMiddleware { output, is_error })
            .core(registry)
            .build()
            .expect("pipeline builds")
    }

    fn ctx_for(tool_name: &str) -> ToolDispatchContext {
        ToolDispatchContext {
            tool_name: tool_name.to_string(),
            input: serde_json::json!({}),
            call_id: "c1".to_string(),
            turn_number: 0,
            cancel: Arc::new(crate::cancel::CancelSignal::new()),
            permission: PermissionCheck::Allow,
            tool_context: ToolContext::default(),
        }
    }

    fn write_tools() -> Vec<String> {
        vec!["Write".to_string(), "Edit".to_string()]
    }

    #[tokio::test]
    async fn verify_runs_after_write_tool() {
        let (v, called) = make_verifier(true, "ok");
        let mw = VerifyMiddleware::new(v, write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("wrote 42 bytes"), false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        assert!(
            *called.lock().unwrap(),
            "verifier should be called for write tools"
        );
        assert!(
            result.output.to_string().contains("[verify]"),
            "output should contain verify block: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn verify_skipped_for_read_tool() {
        let (v, called) = make_verifier(true, "ok");
        let mw = VerifyMiddleware::new(v, write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("file contents"), false);

        let mut ctx = ctx_for("Read");
        let result = pipeline.dispatch(&mut ctx).await;

        assert!(
            !*called.lock().unwrap(),
            "verifier must not be called for non-write tools"
        );
        assert!(
            !result.output.to_string().contains("[verify]"),
            "output should not contain verify block for read tools"
        );
    }

    #[tokio::test]
    async fn verify_result_appended_to_text_output() {
        let (v, _) = make_verifier(true, "all checks pass");
        let mw = VerifyMiddleware::new(v, write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("wrote 1 file"), false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        let s = match result.output {
            ToolContent::Text(s) => s,
            ToolContent::Multipart(parts) => {
                panic!("expected Text, got Multipart with {} parts", parts.len())
            }
        };
        assert!(
            s.contains("wrote 1 file"),
            "original output should be preserved: {s}"
        );
        assert!(
            s.contains("[verify] passed: all checks pass"),
            "verify block should be appended: {s}"
        );
    }

    #[tokio::test]
    async fn verify_result_appended_to_multipart_output() {
        let (v, _) = make_verifier(false, "1 error: expected `;`");
        let mw = VerifyMiddleware::new(v, write_tools());
        // Multipart with one existing text part.
        let existing = ToolContent::from_multipart(vec![ToolContentPart::Text {
            text: "wrote 2 files".to_string(),
        }]);
        let pipeline = pipeline_with(mw, existing, false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        let parts = match result.output {
            ToolContent::Multipart(parts) => parts,
            ToolContent::Text(t) => panic!("expected Multipart, got Text: {t}"),
        };
        // Original part preserved.
        assert_eq!(parts.len(), 2, "verify block should be a new part");
        match &parts[0] {
            ToolContentPart::Text { text } => {
                assert!(
                    text.contains("wrote 2 files"),
                    "first part unchanged: {text}"
                );
                assert!(
                    !text.contains("[verify]"),
                    "verify block must not leak into the first part: {text}"
                );
            }
            ToolContentPart::Image { .. } => panic!("expected first part to be Text, got Image"),
        }
        // New verify part.
        match &parts[1] {
            ToolContentPart::Text { text } => {
                assert!(
                    text.contains("[verify] failed: 1 error: expected `;`"),
                    "verify block in second part: {text}"
                );
            }
            ToolContentPart::Image { .. } => panic!("expected second part to be Text, got Image"),
        }
    }

    #[tokio::test]
    async fn verify_pass_does_not_block() {
        let (v, _) = make_verifier(true, "clean");
        let mw = VerifyMiddleware::new(v, write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("ok"), false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        assert!(
            !result.is_error,
            "verify pass must not mark the result as error"
        );
        assert!(
            result.output.to_string().contains("[verify] passed"),
            "diagnostics should be present: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn verify_fail_appended_not_raised() {
        let (v, _) = make_verifier(false, "compile error in main.rs");
        let mw = VerifyMiddleware::new(v, write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("wrote main.rs"), false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        // is_error stays as the tool returned it (false) — the model
        // sees the failure in the output text and decides.
        assert!(
            !result.is_error,
            "verify failure must not mark the result as error"
        );
        assert!(
            result
                .output
                .to_string()
                .contains("[verify] failed: compile error"),
            "failed verify block should be appended: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn verify_skipped_when_tool_errored() {
        let (v, called) = make_verifier(true, "ok");
        let mw = VerifyMiddleware::new(v, write_tools());
        // The write tool itself errored.
        let pipeline = pipeline_with(mw, ToolContent::from_string("permission denied"), true);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        assert!(
            !*called.lock().unwrap(),
            "verifier must not be called when the tool errored"
        );
        assert!(
            !result.output.to_string().contains("[verify]"),
            "no verify block expected on a tool error: {}",
            result.output
        );
        assert!(
            result.is_error,
            "is_error from the tool should be preserved"
        );
    }

    #[tokio::test]
    async fn verify_block_format() {
        // Pin the exact appended format so downstream consumers parsing
        // the [verify] marker don't break silently.
        let (pass_v, _) = make_verifier(true, "diag-pass");
        let mw_pass = VerifyMiddleware::new(pass_v, write_tools());
        let pipeline = pipeline_with(mw_pass, ToolContent::from_string("x"), false);
        let mut ctx = ctx_for("Write");
        let pass_out = pipeline.dispatch(&mut ctx).await.output.to_string();
        assert!(
            pass_out.contains("[verify] passed: diag-pass"),
            "pass format: {pass_out}"
        );

        let (fail_v, _) = make_verifier(false, "diag-fail");
        let mw_fail = VerifyMiddleware::new(fail_v, write_tools());
        let pipeline = pipeline_with(mw_fail, ToolContent::from_string("x"), false);
        let mut ctx = ctx_for("Write");
        let fail_out = pipeline.dispatch(&mut ctx).await.output.to_string();
        assert!(
            fail_out.contains("[verify] failed: diag-fail"),
            "fail format: {fail_out}"
        );
    }

    #[test]
    fn verify_middleware_name() {
        let (v, _) = make_verifier(true, "");
        let mw = VerifyMiddleware::new(v, write_tools());
        assert_eq!(mw.name(), "verify");
    }

    #[test]
    fn verify_middleware_debug() {
        let (v, _) = make_verifier(true, "");
        let mw = VerifyMiddleware::new(v, write_tools());
        let debug = format!("{mw:?}");
        assert!(debug.contains("VerifyMiddleware"));
        assert!(debug.contains("write_tools"));
    }

    #[tokio::test]
    async fn noop_verifier_always_passes() {
        let verifier = NoopVerifier;
        let ctx = ToolContext::default();
        let result = verifier.verify(&ctx, "Write").await;
        assert!(result.passed, "NoopVerifier must always pass");
        assert!(
            result.diagnostics.is_empty(),
            "NoopVerifier must produce empty diagnostics"
        );
    }

    #[tokio::test]
    async fn noop_verifier_in_verify_middleware_appends_passed_block() {
        let mw = VerifyMiddleware::new(Arc::new(NoopVerifier), write_tools());
        let pipeline = pipeline_with(mw, ToolContent::from_string("wrote 42 bytes"), false);

        let mut ctx = ctx_for("Write");
        let result = pipeline.dispatch(&mut ctx).await;

        let rendered = result.output.to_string();
        assert!(
            rendered.contains("[verify]"),
            "NoopVerifier still triggers the middleware's append: got {rendered:?}"
        );
        assert!(
            rendered.contains("[verify] passed:"),
            "appended block reports passed status: got {rendered:?}"
        );
    }
}
