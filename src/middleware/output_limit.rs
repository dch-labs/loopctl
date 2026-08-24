//! Middleware that truncates tool output to a maximum character count.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::message::{ToolContent, ToolContentPart};
use std::future::Future;
use std::pin::Pin;

/// Middleware that truncates tool output to a maximum character count.
///
/// The limit is a **whole-output budget over the text content**: a text
/// part that does not fit is cut so that the kept text *plus* its
/// `[truncated]` marker stays within the remaining budget, and once the
/// budget is exhausted every later text part is emptied (a marker per
/// part would overflow the cap it enforces; when an earlier part was
/// cut, its marker already signals the truncation). A remaining budget
/// smaller than the marker itself still yields the bare marker on a
/// cut part — truncation must stay visible, so the cap floors at one
/// marker. Image parts are left unchanged.
///
/// A `max_chars` of `0` disables the middleware entirely (the crate-wide
/// zero-disables sentinel) — output passes through untouched.
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
///     .with_middleware(OutputLimitMiddleware::new(10_000))
///     .with_core(registry)
///     .build()?;
/// ```
pub struct OutputLimitMiddleware {
    max_chars: usize,
}

/// The truncation marker appended to a cut text part.
///
/// Twelve characters including the leading newline; its length is
/// reserved inside the remaining budget by [`truncate_marked`] so a cut
/// part lands exactly on the cap.
const TRUNCATION_MARKER: &str = "\n[truncated]";

/// Truncate `text` so the kept text plus the marker fits `budget`.
///
/// When `budget` is smaller than the marker itself the result is the
/// bare marker — truncation must stay visible even at tiny budgets, so
/// the cap floors at one marker.
fn truncate_marked(text: &str, budget: usize) -> String {
    let marker_len = TRUNCATION_MARKER.chars().count();
    let kept = budget.saturating_sub(marker_len);
    let truncated: String = text.chars().take(kept).collect();
    format!("{truncated}{TRUNCATION_MARKER}")
}

impl OutputLimitMiddleware {
    /// Create a new output-limiting middleware.
    ///
    /// `max_chars` is the maximum number of characters of text output
    /// (markers included once a part is cut). Outputs at or below this
    /// limit pass through unchanged. A cap smaller than the
    /// `[truncated]` marker still leaves the bare marker on a cut
    /// part, so the effective floor is one marker. `0` disables the
    /// middleware.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::middleware::{OutputLimitMiddleware, ToolMiddleware};
    ///
    /// let limiter = OutputLimitMiddleware::new(10_000);
    /// assert_eq!(limiter.name(), "output_limit");
    /// ```
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
            if max_chars == 0 {
                return result;
            }

            match result.output {
                ToolContent::Text(ref text) => {
                    let char_count = text.chars().count();
                    if char_count > max_chars {
                        result.output = ToolContent::Text(truncate_marked(text, max_chars));
                    }
                }
                ToolContent::Multipart(ref mut parts) => {
                    let mut remaining = max_chars;
                    for part in parts.iter_mut() {
                        if let ToolContentPart::Text { text } = part {
                            let char_count = text.chars().count();
                            if char_count > remaining {
                                if remaining == 0 {
                                    text.clear();
                                } else {
                                    *text = truncate_marked(text, remaining);
                                }
                                remaining = 0;
                            } else {
                                remaining = remaining.saturating_sub(char_count);
                            }
                        }
                    }
                }
            }

            result
        })
    }
}
