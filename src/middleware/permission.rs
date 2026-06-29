//! Middleware that checks tool permissions before execution.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::tool::PermissionCheck;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Permission check function type.
pub type PermissionCheckFn = Arc<dyn Fn(&ToolDispatchContext) -> PermissionCheck + Send + Sync>;

/// Async resolver for [`PermissionCheck::Ask`].
///
/// Receives the prompt string and the tool name, returns `true` to allow
/// the tool call or `false` to deny it. Called by [`PermissionMiddleware`]
/// when the permission check resolves to [`PermissionCheck::Ask`].
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
///   resolver, `Ask` is denied (headless mode).
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
    check_fn: Option<PermissionCheckFn>,
    /// When `Some`, called to resolve [`PermissionCheck::Ask`] interactively.
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
            PermissionCheck::Ask { prompt } => match &self.ask_resolver {
                Some(resolver) => {
                    let resolver = Arc::clone(resolver);
                    Box::pin(async move {
                        let tool_name = ctx.tool_name.clone();
                        let approved = resolver(&prompt, &tool_name);
                        if approved.await {
                            next.dispatch(ctx).await
                        } else {
                            ToolDispatchResult::err(
                                &tool_name,
                                format!("Permission denied by user for tool '{tool_name}'"),
                                Duration::ZERO,
                            )
                        }
                    })
                }
                None => Self::deny(ctx, &format!("permission required: {prompt}")),
            },
        }
    }
}

impl PermissionMiddleware {
    /// Build a denied result with tracing.
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
