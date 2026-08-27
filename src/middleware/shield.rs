//! Safety-shield enforcement for tool dispatch.
//!
//! [`SafetyShieldMiddleware`] wraps the dispatch pipeline and consults a
//! [`ToolSafetyShield`] before every watched
//! call (the shield's `watched_tools` names the tools it inspects): a
//! `Block` decision skips execution and surfaces as a soft error
//! carrying the decision's reason (the model sees the refusal and can
//! adapt; the run continues), while `Warn` proceeds and is emitted as
//! a tracing warning carrying the decision's reason and category
//! (advisory — the call still runs, consistent with the shield's
//! scoring contract) and `Allow` proceeds. After the call,
//! [`record_invocation`](crate::tool::shield::ToolSafetyShield::record_invocation)
//! feeds the shield's internal history so multi-call combination rules
//! (a `curl` followed by `| sh`) can fire across turns.
//!
//! Opt-in (P5): nothing changes for a loop that does not install it. A
//! default-constructed loop does not.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::message::ToolContent;
use crate::tool::shield::{SafetyAction, ShieldContext, ToolSafetyShield};

/// Middleware enforcing a [`ToolSafetyShield`]'s decisions.
///
/// Installed over a pipeline, it evaluates every watched call
/// pre-dispatch: a `Block` decision becomes a soft error, and a
/// `Warn` decision proceeds with its reason and category emitted
/// as a tracing warning. The shield is
/// shared behind an `Arc` so the same instance can serve a host's other
/// evaluations; its internal history is updated by this middleware via
/// `record_invocation`, so install it exactly once per session — two
/// installed instances (or one plus a host calling `record_invocation`
/// on the shared handle) evaluate every call twice and split the
/// `recent_calls` window between them.
///
/// The shield evaluates the tool name as it reaches this layer.
/// Middleware may rewrite [`ctx.tool_name`](ToolDispatchContext::tool_name)
/// to redirect a call; install this middleware *after* (inside) any
/// name-rewriting layer so aliases are evaluated under the name the
/// registry will execute. A rewriter placed inside the shield redirects
/// after evaluation, and the aliased call runs unshielded. For the
/// same reason, install it *before* (outside)
/// [`MemoizingMiddleware`](super::MemoizingMiddleware): a repeat
/// served from an inner cache is still evaluated and recorded — the
/// repetition dimension scores the model's behavior, not the tool's
/// execution. The post-call record (`record_invocation`,
/// `recent_calls`) describes the call as this middleware evaluated
/// it: a name or input rewritten by an inner layer does not leak into
/// the shield's history.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::middleware::{SafetyShieldMiddleware, ToolPipeline};
/// use loopctl::tool::shield::UnixShield;
/// use std::sync::Arc;
///
/// let pipeline = ToolPipeline::builder()
///     .with_middleware(SafetyShieldMiddleware::new(Arc::new(UnixShield::default())))
///     .with_core(Arc::new(tools))
///     .build()?;
/// ```
pub struct SafetyShieldMiddleware {
    /// The shield consulted before every dispatch.
    ///
    /// Held behind an `Arc` because the trait is object-safe and the
    /// same shield instance may be referenced by host-side evaluations
    /// alongside this middleware.
    shield: Arc<dyn ToolSafetyShield>,

    /// `(tool_name, turn)` for the most recent 20 *watched* calls —
    /// calls outside the watched set are neither evaluated nor recorded
    /// here, so a shield reading this snapshot is blind to them.
    ///
    /// The snapshot source for
    /// [`ShieldContext::recent_calls`](crate::tool::shield::ShieldContext::recent_calls):
    /// shields that prefer not to keep their own history read the
    /// context field; shields that do (like `UnixShield`) ignore it.
    /// A recency window, not a session archive — the per-evaluation
    /// snapshot stays constant-cost. Blocked attempts never happened,
    /// so they never enter the record; with an inner memoizer, a
    /// cache-served repeat still enters — the call was made.
    recent: std::sync::Mutex<Vec<(String, usize)>>,
}

impl SafetyShieldMiddleware {
    /// Create a shield-enforcing middleware.
    ///
    /// `name()` is `"safety_shield"`.
    #[must_use]
    pub fn new(shield: Arc<dyn ToolSafetyShield>) -> Self {
        Self {
            shield,
            recent: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Build the per-call [`ShieldContext`] from a dispatch context.
    fn shield_ctx(&self, ctx: &ToolDispatchContext) -> ShieldContext {
        ShieldContext {
            tool_name: ctx.tool_name.clone(),
            input: ctx.input.clone(),
            turn: ctx.turn_number,
            recent_calls: self
                .recent
                .lock()
                .map(|log| log.clone())
                .unwrap_or_default(),
        }
    }
}

impl ToolMiddleware for SafetyShieldMiddleware {
    fn name(&self) -> &'static str {
        "safety_shield"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let watched = self.shield.watched_tools();
        let watched = (!watched.is_empty()).then(|| watched.contains(&ctx.tool_name));
        let decision = match watched {
            Some(true) => Some(self.shield.evaluate(&self.shield_ctx(ctx))),
            _ => None,
        };
        if let Some(decision) = &decision
            && decision.action == SafetyAction::Warn
        {
            tracing::warn!(
                tool = %ctx.tool_name,
                reason = decision.reason.as_deref().unwrap_or(""),
                category = decision.category.as_deref().unwrap_or(""),
                "safety shield warning"
            );
        }
        Box::pin(async move {
            if let Some(decision) = decision
                && decision.action == SafetyAction::Block
            {
                let reason = decision.reason;
                return ToolDispatchResult {
                    tool_call_id: ctx.call_id.clone(),
                    output: ToolContent::Text(format!(
                        "blocked by safety shield: {}",
                        reason.as_deref().unwrap_or("high risk")
                    )),
                    is_error: true,
                    resolved_tool_name: String::new(),
                    duration: std::time::Duration::ZERO,
                    display_hint: None,
                };
            }
            let (tool_name, input) = (ctx.tool_name.clone(), ctx.input.clone());
            let result = next.dispatch(ctx).await;
            if watched == Some(true) {
                if let Ok(mut log) = self.recent.lock() {
                    log.push((tool_name.clone(), ctx.turn_number));
                    if log.len() > 20 {
                        let drain_until = log.len().saturating_sub(20);
                        log.drain(..drain_until);
                    }
                }
                self.shield
                    .record_invocation(&tool_name, &input, !result.is_error);
            }
            result
        })
    }
}
