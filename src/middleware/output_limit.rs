//! Middleware that truncates tool output to a maximum character count.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::message::ToolContent;
use std::future::Future;
use std::pin::Pin;

/// Middleware that truncates tool output to a maximum character count.
///
/// If the tool's text output exceeds the limit, it is truncated and
/// suffixed with a `[truncated]` marker. Non-text outputs ([`ToolContent::Multipart`])
/// are passed through unchanged.
///
/// This prevents runaway tools from flooding the conversation with
/// excessive output that would blow the context window.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::middleware::OutputLimitMiddleware;
///
/// let pipeline = ToolPipeline::builder()
///     .with(OutputLimitMiddleware::new(10_000))
///     .core(registry)
///     .build()?;
/// ```
pub struct OutputLimitMiddleware {
    max_chars: usize,
}

impl OutputLimitMiddleware {
    /// Create a new output-limiting middleware.
    ///
    /// `max_chars` is the maximum number of characters in the text
    /// output. Outputs at or below this limit pass through unchanged.
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl ToolMiddleware for OutputLimitMiddleware {
    fn name(&self) -> &'static str {
        "output_limit"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let max_chars = self.max_chars;
        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;

            if let ToolContent::Text(ref text) = result.output {
                let char_count = text.chars().count();
                if char_count > max_chars {
                    let truncated: String = text.chars().take(max_chars).collect();
                    result.output = ToolContent::Text(format!("{truncated}\n[truncated]"));
                }
            }

            result
        })
    }
}
