//! Safety-shield enforcement for tool dispatch.
//!
//! [`SafetyShieldMiddleware`] wraps the dispatch pipeline and consults a
//! [`ToolSafetyShield`] before every
//! call: a `Block` decision skips execution and surfaces as a soft error
//! carrying the decision's reason (the model sees the refusal and can
//! adapt; the run continues), while `Warn` and `Allow` proceed — advisory
//! only, consistent with the shield's scoring contract. After the call,
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
/// Installed over a pipeline, it evaluates every call pre-dispatch and
/// blocks (as a soft error) when the shield says `Block`. The shield is
/// shared behind an `Arc` so the same instance can serve a host's other
/// evaluations; its internal history is updated by this middleware via
/// `record_invocation`, so install it exactly once per session.
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

    /// `(tool_name, turn)` for every executed call this middleware
    /// dispatched.
    ///
    /// The snapshot source for
    /// [`ShieldContext::recent_calls`](crate::tool::shield::ShieldContext::recent_calls):
    /// shields that prefer not to keep their own history read the
    /// context field; shields that do (like `UnixShield`) ignore it.
    /// Appended only for executed calls — blocked attempts never
    /// happened, so they never enter the record.
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
        Box::pin(async move {
            if let Some(decision) = decision
                && decision.action == SafetyAction::Block
            {
                let reason = decision.reason;
                return ToolDispatchResult {
                    tool_call_id: String::new(),
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
            let result = next.dispatch(ctx).await;
            if watched == Some(true) {
                if let Ok(mut log) = self.recent.lock() {
                    log.push((ctx.tool_name.clone(), ctx.turn_number));
                }
                self.shield
                    .record_invocation(&ctx.tool_name, &ctx.input, !result.is_error);
            }
            result
        })
    }
}
