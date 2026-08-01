//! Message construction and extraction.
//!
//! Pure functions that build or extract data from [`Message`] instances —
//! assembling tool-result messages, extracting text or tool calls from an
//! assistant response, and computing token counts.

use super::{ApiClient, BareLoop, MessagePart, ToolContext, ToolDispatchResult, ToolSchema};

impl<C: ApiClient> BareLoop<C> {
    /// Build the tool-result parts from executed tool results.
    ///
    /// Each dispatch result becomes one `tool_result` [`MessagePart`]:
    /// the `tool_call_id` paired with the tool's output (wrapped in a
    /// [`ToolContent`](ToolContent)) and an `is_error` flag so the model
    /// can distinguish successes from failures. The caller is responsible
    /// for assembling these parts — alongside any preresolved results —
    /// into the single user [`Message`] that represents the turn.
    ///
    /// # Parameters
    ///
    /// - `results` — The [`ToolDispatchResult`]s produced by
    ///   [`dispatch_tools()`](BareLoop::dispatch_tools).
    pub(super) fn build_tool_result_parts(results: Vec<ToolDispatchResult>) -> Vec<MessagePart> {
        results
            .into_iter()
            .map(|r| {
                MessagePart::tool_result(r.tool_call_id, r.resolved_tool_name, r.output, r.is_error)
            })
            .collect()
    }

    /// Build tool schemas for the API request.
    ///
    /// Collects all tool schemas from the [`ToolRegistry`] and returns
    /// them as `Some(Vec<ToolSchema>)`, or `None` if the registry is
    /// empty (i.e. the agent has no tools). The API uses these schemas
    /// to inform the model what tools are available and their expected
    /// input shapes.
    pub(super) fn build_tool_schemas(&self) -> Option<Vec<ToolSchema>> {
        let schemas = self.tools.all_schemas();
        if schemas.is_empty() {
            None
        } else {
            Some(schemas)
        }
    }

    /// Build a tool context for tool invocations.
    ///
    /// Creates a [`ToolContext`] pre-populated with the current session
    /// ID. Tools can use the context to correlate their work with the
    /// enclosing session (e.g. for logging, tracing, or storage).
    pub(super) fn build_tool_context(&self) -> ToolContext {
        ToolContext {
            session_id: self.session.id,
            ..ToolContext::default()
        }
    }
}
