//! The innermost middleware that performs the actual tool invocation.

use super::{ToolDispatchContext, ToolDispatchResult};
use crate::error::LoopError;
use crate::tool::ToolRegistry;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

/// The innermost middleware that performs the actual tool invocation.
///
/// Looks up the tool by name in the [`ToolRegistry`], calls
/// [`crate::tool::Tool::call()`], and converts the result into a
/// [`ToolDispatchResult`]. If the tool is not found, produces a
/// soft error result (not a hard error) so the model can recover.
///
/// This middleware is automatically created by the pipeline builder
/// and always occupies the innermost position in the chain.
pub struct ToolCallMiddleware {
    pub(super) registry: Arc<ToolRegistry>,
}

impl ToolCallMiddleware {
    pub(super) const NAME: &str = "tool_call";

    /// Create a new core dispatch wrapping the given registry.
    #[must_use]
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }

    /// Execute the tool call — the terminal dispatch.
    ///
    /// Looks up the tool by name in the registry, calls `Tool::call()`,
    /// and converts the result. There is no `next` parameter because
    /// there is nothing to chain to.
    pub(super) fn dispatch(
        &self,
        ctx: &mut ToolDispatchContext,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + '_>> {
        let tool_name = ctx.tool_name.clone();
        let input = ctx.input.clone();
        let tool_ctx = ctx.tool_context.clone();
        let registry = Arc::clone(&self.registry);
        let cancel = Arc::clone(&ctx.cancel);
        let call_id = ctx.call_id.clone();

        Box::pin(async move {
            let start = Instant::now();
            let Some(tool) = registry.get(&tool_name) else {
                let available: Vec<String> = registry.tool_names();
                let available_refs: Vec<&str> = available.iter().map(String::as_str).collect();
                let error = LoopError::tool_not_found(&tool_name, &available_refs);
                return ToolDispatchResult::err(&tool_name, error.to_string(), start.elapsed())
                    .with_call_id(&call_id);
            };

            let call_result = tokio::select! {
                r = tool.call(input, &tool_ctx) => r,
                () = cancel.notified() => {
                    return ToolDispatchResult::err(
                        &tool_name,
                        format!("Tool '{tool_name}' cancelled"),
                        start.elapsed(),
                    )
                    .with_call_id(&call_id);
                }
            };

            let duration = start.elapsed();
            ToolDispatchResult::from_result(&tool_name, call_result, duration)
                .with_call_id(&call_id)
        })
    }
}
