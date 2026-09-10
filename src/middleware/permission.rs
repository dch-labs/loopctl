//! Middleware that checks tool permissions before execution.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::tool::PermissionCheck;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Permission check function type.
///
/// Shared behind an `Arc` so several middleware instances carry one
/// policy closure without copying it; changing the policy means
/// building a new middleware, not mutating a shared one.
pub type PermissionCheckFn = Arc<dyn Fn(&ToolDispatchContext) -> PermissionCheck + Send + Sync>;

/// Async resolver for [`PermissionCheck::Ask`].
///
/// Receives the prompt string and the tool name, returns `true` to allow
/// the tool call or `false` to deny it. Called by [`PermissionMiddleware`]
/// when the permission check resolves to [`PermissionCheck::Ask`].
///
/// The returned future may be dropped before it resolves: when the
/// dispatch's [`CancelSignal`](crate::cancel::CancelSignal) fires while
/// the answer is still pending, the middleware stops awaiting and denies
/// with `cancelled while awaiting approval`, and no completion signal
/// reaches the resolver side. Implementations should be
/// cancellation-safe — detect a dropped answer channel and tear the
/// prompt down — rather than assuming every prompt is answered.
pub type AskResolverFn =
    Arc<dyn Fn(&str, &str) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// Middleware that checks tool permissions before execution.
///
/// Inspects the [`PermissionCheck`] in the dispatch context:
///
/// - `Allow` — passes through to the next layer.
/// - `Deny` — short-circuits with an error result.
/// - `Modify` — replaces `ctx.input` with the modified input, then proceeds.
/// - `Ask` — if an [`AskResolverFn`] is configured, calls it to prompt the
///   user; the tool proceeds on `true` or is denied on `false`. Without a
///   resolver, `Ask` is denied (headless mode). The prompt races the
///   dispatch's [`CancelSignal`](crate::cancel::CancelSignal), and a
///   cancellation that already fired when the answer lands wins
///   deterministically — a cancelled run never executes the tool,
///   however the approval resolves.
///
/// # Example
///
/// ```rust,ignore
/// // Deny all by default
/// let mw = PermissionMiddleware::deny_all();
///
/// // Custom logic
/// let mw = PermissionMiddleware::with_check(|ctx| {
///     if ctx.tool_name == "safe_read" {
///         PermissionCheck::Allow
///     } else {
///         PermissionCheck::Deny { reason: "not on allowlist".into() }
///     }
/// });
/// ```
pub struct PermissionMiddleware {
    /// When `Some`, overrides [`ToolDispatchContext::permission`].
    ///
    /// Lets one pipeline impose a uniform policy regardless of what
    /// each dispatch claims.
    check_fn: Option<PermissionCheckFn>,
    /// When `Some`, called to resolve [`PermissionCheck::Ask`] interactively.
    ///
    /// Absent, an `Ask` degrades to a denial naming the prompt, so a
    /// headless run never blocks on an unanswered question.
    ask_resolver: Option<AskResolverFn>,
}

impl PermissionMiddleware {
    /// Create a permission middleware that denies all calls.
    ///
    /// Every tool call will be short-circuited with a permission-denied
    /// error. Useful as a safety default in restricted environments.
    #[must_use]
    pub fn deny_all() -> Self {
        Self {
            check_fn: Some(Arc::new(|_| PermissionCheck::Deny {
                reason: "blocked by policy".into(),
            })),
            ask_resolver: None,
        }
    }

    /// Create a permission middleware that allows all calls.
    ///
    /// No permission checks are performed — every tool call passes
    /// through to the next layer. Equivalent to having no permission
    /// middleware, but can be used for logging or metrics in permissive
    /// environments.
    #[must_use]
    pub fn allow_all() -> Self {
        Self {
            check_fn: Some(Arc::new(|_| PermissionCheck::Allow)),
            ask_resolver: None,
        }
    }

    /// Set a custom permission check function.
    ///
    /// The function receives a reference to the dispatch context and
    /// returns the appropriate [`PermissionCheck`] for that call.
    #[must_use]
    pub fn with_check(
        mut self,
        f: impl Fn(&ToolDispatchContext) -> PermissionCheck + Send + Sync + 'static,
    ) -> Self {
        self.check_fn = Some(Arc::new(f));
        self
    }

    /// Create a permission middleware that reads from the context.
    ///
    /// The middleware reads `ctx.permission` directly, without
    /// applying any override. Use when the permission is set by the
    /// framework or a prior middleware.
    #[must_use]
    pub fn from_context() -> Self {
        Self {
            check_fn: None,
            ask_resolver: None,
        }
    }

    /// Attach an async resolver for [`PermissionCheck::Ask`].
    ///
    /// When the permission check returns `Ask`, the resolver is called with
    /// the prompt and tool name. The tool call proceeds if the resolver
    /// returns `true`, and is denied if it returns `false`.
    ///
    /// Without a resolver, `Ask` is denied (headless mode).
    #[must_use]
    pub fn with_ask_resolver(mut self, resolver: AskResolverFn) -> Self {
        self.ask_resolver = Some(resolver);
        self
    }

    /// Decide the permission for one dispatch: the middleware's own
    /// check function when configured, otherwise the context's
    /// per-call verdict.
    ///
    /// Centralizing the fallback keeps the configured and default
    /// paths on one resolution rule.
    fn resolve_permission(&self, ctx: &ToolDispatchContext) -> PermissionCheck {
        match &self.check_fn {
            Some(f) => f(ctx),
            None => ctx.permission.clone(),
        }
    }
}

impl ToolMiddleware for PermissionMiddleware {
    fn name(&self) -> &'static str {
        "permission"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let permission = self.resolve_permission(ctx);
        match permission {
            PermissionCheck::Allow => next.dispatch(ctx),
            PermissionCheck::Modify { modified_input } => {
                ctx.input = modified_input;
                next.dispatch(ctx)
            }
            PermissionCheck::Deny { reason } => Self::deny(ctx, &reason),
            PermissionCheck::Ask { prompt } => {
                if let Some(resolver) = &self.ask_resolver {
                    let resolver = Arc::clone(resolver);
                    Box::pin(async move {
                        let tool_name = ctx.tool_name.clone();
                        let cancel = Arc::clone(&ctx.cancel);
                        let approved = resolver(&prompt, &tool_name);
                        let approved = tokio::select! {
                            approved = approved => approved,
                            () = cancel.notified() => {
                                return Self::deny(
                                    ctx,
                                    "cancelled while awaiting approval",
                                )
                                .await;
                            }
                        };
                        if approved && !cancel.is_cancelled() {
                            next.dispatch(ctx).await
                        } else if approved {
                            Self::deny(ctx, "cancelled while awaiting approval").await
                        } else {
                            Self::deny(ctx, "denied by user").await
                        }
                    })
                } else {
                    tracing::warn!(
                        tool = %ctx.tool_name,
                        prompt = %prompt,
                        "permission Ask denied: no resolver configured"
                    );
                    Self::deny(ctx, &format!("permission required: {prompt}"))
                }
            }
        }
    }
}

impl PermissionMiddleware {
    /// Build a denied result with tracing.
    ///
    /// One shape for every denial site — an explicit `Deny`, an
    /// unanswered `Ask`, a user refusal, and a pipeline cancelled
    /// mid-prompt — so logs and results stay in sync wherever the
    /// verdict originated.
    fn deny<'a>(
        ctx: &'a mut ToolDispatchContext,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let tool_name = ctx.tool_name.clone();
        let reason = reason.to_string();
        tracing::warn!(
            tool = %tool_name,
            permission = %reason,
            "tool call blocked by permission middleware"
        );
        Box::pin(std::future::ready(ToolDispatchResult::err(
            &tool_name,
            format!("Permission {reason} for tool '{tool_name}'"),
            Duration::ZERO,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::CancelSignal;
    use crate::middleware::ToolPipeline;
    use crate::tool::{PermissionCheck, ToolContext, ToolRegistry};

    #[tokio::test]
    async fn a_cancelled_pipeline_aborts_a_pending_approval_prompt() {
        let resolver: AskResolverFn = Arc::new(|_prompt: &str, _tool: &str| {
            Box::pin(std::future::pending::<bool>()) as Pin<Box<dyn Future<Output = bool> + Send>>
        });
        let permission = PermissionMiddleware::from_context()
            .with_check(|_| PermissionCheck::Ask {
                prompt: "approve the deploy?".into(),
            })
            .with_ask_resolver(resolver);
        let pipeline = Arc::new(
            ToolPipeline::builder()
                .with_core(Arc::new(ToolRegistry::new()))
                .with_middleware(permission)
                .build()
                .expect("a core plus one middleware assembles"),
        );
        let cancel = Arc::new(CancelSignal::new());
        let ctx = ToolDispatchContext {
            tool_name: "deploy".into(),
            input: serde_json::Value::Null,
            call_id: "call_1".into(),
            turn_number: 0,
            cancel: Arc::clone(&cancel),
            permission: PermissionCheck::allow(),
            tool_context: ToolContext::default(),
        };

        let invoked = tokio::spawn({
            let pipeline = Arc::clone(&pipeline);
            async move { pipeline.invoke(ctx).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(1), invoked)
            .await
            .expect("cancellation ends the prompt well inside the timeout")
            .expect("the spawned invoke survives to completion");
        assert!(result.is_error, "a cancelled prompt denies the call");
        match &result.output {
            crate::message::ToolContent::Text(text) => assert!(
                text.contains("cancelled while awaiting approval"),
                "the denial names the cancelled prompt: {text}"
            ),
            crate::message::ToolContent::Multipart(parts) => {
                panic!("expected a text denial, got {} parts", parts.len())
            }
        }
    }

    #[tokio::test]
    async fn an_approval_landing_after_cancellation_still_denies() {
        let resolver: AskResolverFn = Arc::new(|_prompt: &str, _tool: &str| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                true
            }) as Pin<Box<dyn Future<Output = bool> + Send>>
        });
        let permission = PermissionMiddleware::from_context()
            .with_check(|_| PermissionCheck::Ask {
                prompt: "approve the deploy?".into(),
            })
            .with_ask_resolver(resolver);
        let pipeline = Arc::new(
            ToolPipeline::builder()
                .with_core(Arc::new(ToolRegistry::new()))
                .with_middleware(permission)
                .build()
                .expect("a core plus one middleware assembles"),
        );
        let cancel = Arc::new(CancelSignal::new());
        cancel.cancel();
        let ctx = ToolDispatchContext {
            tool_name: "deploy".into(),
            input: serde_json::Value::Null,
            call_id: "call_1".into(),
            turn_number: 0,
            cancel: Arc::clone(&cancel),
            permission: PermissionCheck::allow(),
            tool_context: ToolContext::default(),
        };

        let result = tokio::time::timeout(Duration::from_secs(1), pipeline.invoke(ctx))
            .await
            .expect("the already-cancelled signal resolves the select immediately");
        assert!(result.is_error, "a cancelled run must not execute the tool");
        match &result.output {
            crate::message::ToolContent::Text(text) => assert!(
                text.contains("cancelled while awaiting approval"),
                "cancellation wins the both-ready race deterministically: {text}"
            ),
            crate::message::ToolContent::Multipart(parts) => {
                panic!("expected a text denial, got {} parts", parts.len())
            }
        }
    }
}
