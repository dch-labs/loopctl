//! Message construction and extraction.
//!
//! Pure functions that build or extract data from [`Message`] instances —
//! assembling tool-result messages, extracting text or tool calls from an
//! assistant response, and computing token counts.

use super::{
    ApiClient, BareLoop, Message, MessagePart, Role, ToolContext, ToolDispatchResult, ToolSchema,
};

impl<C: ApiClient> BareLoop<C> {
    /// Build the tool result message from executed tool results.
    ///
    /// Per API convention, tool results are sent as a **user** message
    /// containing `tool_result` content parts. Each part pairs the
    /// `tool_call_id` with the tool's output (wrapped in a
    /// [`ToolContent`](ToolContent)) and an `is_error`
    /// flag so the model can distinguish successes from failures.
    ///
    /// # Parameters
    ///
    /// - `results` — The [`ToolDispatchResult`]s produced by
    ///   [`dispatch_tools()`](BareLoop::dispatch_tools).
    ///
    /// # Returns
    ///
    /// A [`Message`] with [`Role::User`] and one `tool_result`
    /// [`MessagePart`] per result.
    pub(super) fn build_tool_result_message(results: Vec<ToolDispatchResult>) -> Message {
        let parts: Vec<MessagePart> = results
            .into_iter()
            .map(|r| MessagePart::tool_result(r.tool_call_id, r.output, r.is_error))
            .collect();
        Message::new(Role::User, parts)
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
