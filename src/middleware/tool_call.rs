//! The innermost middleware that performs the actual tool invocation.

use super::{ToolDispatchContext, ToolDispatchResult};
use crate::error::LoopError;
use crate::tool::ToolRegistry;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::FutureExt;

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

            // Wrap the tool call in `catch_unwind` so a panicking tool
            // implementation produces an error result instead of unwinding
            // through and aborting the entire agent loop.  Tools are
            // user-supplied `dyn Tool` implementations and the framework
            // cannot trust them to be panic-free.
            let call_result = tokio::select! {
                r = AssertUnwindSafe(tool.call(input, &tool_ctx)).catch_unwind() => r,
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

            // Convert a panic payload into a tool-error result.
            let call_result = match call_result {
                Ok(inner) => inner,
                Err(panic_payload) => {
                    let msg = panic_payload
                        .downcast_ref::<&'static str>()
                        .map(std::string::ToString::to_string)
                        .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| {
                            format!("Tool '{tool_name}' panicked (unknown payload)")
                        });
                    tracing::error!(tool = %tool_name, panic_message = %msg, "tool panicked during execution");
                    return ToolDispatchResult::err(
                        &tool_name,
                        format!("Tool '{tool_name}' panicked: {msg}"),
                        duration,
                    )
                    .with_call_id(&call_id);
                }
            };

            ToolDispatchResult::from_result(&tool_name, call_result, duration)
                .with_call_id(&call_id)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unnecessary_literal_bound)]
mod tests {
    use super::*;
    use crate::cancel::CancelSignal;
    use crate::message::ToolContent;
    use crate::middleware::ToolDispatchContext;
    use crate::tool::PermissionCheck;
    use crate::tool::{
        FnTool, Tool, ToolContext, ToolError, ToolOutput, ToolSchema, registry::ToolRegistry,
    };
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;

    struct PanickingTool;

    fn echo_fn(
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>> {
        Box::pin(async { Ok(ToolOutput::text("ok")) })
    }

    impl Tool for PanickingTool {
        fn name(&self) -> &str {
            "panic_tool"
        }
        fn description(&self) -> &str {
            "A tool that panics"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "panic_tool".into(),
                description: "A tool that panics".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async {
                panic!("boom from panicking tool");
            })
        }
    }

    fn make_ctx(tool_name: &str, cancel: Arc<CancelSignal>) -> ToolDispatchContext {
        ToolDispatchContext {
            tool_name: tool_name.into(),
            input: serde_json::Value::Null,
            call_id: "call_1".into(),
            turn_number: 0,
            cancel,
            permission: PermissionCheck::allow(),
            tool_context: ToolContext::default(),
        }
    }

    #[tokio::test]
    async fn panicking_tool_produces_error_result_not_abort() {
        let mut registry = ToolRegistry::new();
        registry.register(PanickingTool);

        let middleware = ToolCallMiddleware::new(Arc::new(registry));
        let cancel = Arc::new(CancelSignal::new());
        let mut ctx = make_ctx("panic_tool", cancel);

        let result = middleware.dispatch(&mut ctx).await;

        assert!(result.is_error, "panic should become an error result");
        match &result.output {
            ToolContent::Text(text) => {
                assert!(
                    text.contains("panicked"),
                    "error message should mention panic: {text}"
                );
            }
            ToolContent::Multipart(_) => panic!("expected Text output"),
        }
    }

    #[tokio::test]
    async fn normal_tool_still_works() {
        let mut registry = ToolRegistry::new();
        registry.register(FnTool::new(
            "echo".into(),
            "Echoes input".into(),
            serde_json::json!({"type": "object"}),
            echo_fn,
        ));

        let middleware = ToolCallMiddleware::new(Arc::new(registry));
        let cancel = Arc::new(CancelSignal::new());
        let mut ctx = make_ctx("echo", cancel);

        let result = middleware.dispatch(&mut ctx).await;
        assert!(!result.is_error);
        match &result.output {
            ToolContent::Text(text) => assert_eq!(text, "ok"),
            ToolContent::Multipart(_) => panic!("expected Text output"),
        }
    }
}
