//! Streaming phase — send conversation to the LLM API and accumulate the response.
//!
//! Extracted from [`BareLoop`] to isolate the streaming concern. When a
//! [`StreamHandler`] is configured, delegates to it for resilient streaming
//! (retry, timeout, fallback). Otherwise, uses basic inline logic.

use super::{
    AgentError, ApiClient, BareLoop, Message, StreamAccumulator, StreamEvent, StreamStopReason,
    Usage,
};
use crate::stream::handler::{StreamHandler, StreamHandlerError};
use futures::StreamExt;

impl<C: ApiClient> BareLoop<C> {
    /// Stream one turn from the API, accumulating the response.
    ///
    /// When a [`StreamHandler`] is configured (via
    /// [`set_stream_handler()`](BareLoop::set_stream_handler)), delegates
    /// to the handler for resilient streaming with retry, timeout, and
    /// fallback capabilities. Otherwise, uses the basic inline logic
    /// with no retries.
    ///
    /// Sends the current conversation history to the LLM API via
    /// [`ApiClient::stream_messages`] and uses a [`StreamAccumulator`]
    /// to collect the events into a single [`Message`].
    ///
    /// Also captures the stop reason (e.g. `end_turn`, `tool_call`) and
    /// token [`Usage`] from the stream's final `MessageDelta` event.
    ///
    /// # Returns
    ///
    /// A tuple of `(Message, Option<Usage>, StreamStopReason)`:
    ///
    /// - **[`Message`]** — the fully accumulated assistant message, including
    ///   any text and `tool_call` content parts.
    /// - **Option<[`Usage`]>** — token counts for this turn, if reported.
    /// - **[`StreamStopReason`]** — why the model stopped generating.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Api`] if any stream event is an error.
    /// Returns [`AgentError::Cancelled`] if the cancellation signal fires mid-stream.
    pub(super) async fn stream_turn(
        &self,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), AgentError> {
        // Delegate to StreamHandler if configured.
        if let Some(ref handler) = self.stream_handler {
            return self.stream_turn_via_handler(handler).await;
        }

        // Inline streaming (no handler).
        let system = self.config.system_prompt.clone();
        let tool_schemas = self.build_tool_schemas();
        let mut stream =
            self.client
                .stream_messages(self.conversation.clone(), system, tool_schemas);
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;
        loop {
            let event_result = tokio::select! {
                event = stream.next() => event,
                () = self.cancelled.notified() => {
                    return Err(AgentError::Cancelled);
                }
            };

            match event_result {
                Some(Ok(event)) => {
                    if let StreamEvent::MessageDelta(delta) = &event {
                        if let Some(ref reason_str) = delta.delta.stop_reason {
                            stop_reason =
                                StreamStopReason::from_api_str(reason_str).unwrap_or(stop_reason);
                        }
                    }
                    drop(accumulator.process(&event));
                }
                Some(Err(api_error)) => {
                    return Err(AgentError::Api(api_error.to_string()));
                }
                None => break,
            }
        }

        let usage = accumulator.usage().copied();
        let message = accumulator.build();

        Ok((message, usage, stop_reason))
    }

    /// Stream one turn via the [`StreamHandler`].
    ///
    /// Delegates streaming to the handler, which manages retries,
    /// timeouts, and fallback to non-streaming. Maps the handler's
    /// result/error types back to the `(Message, Option<Usage>,
    /// StreamStopReason)` tuple expected by the run loop.
    ///
    /// # Errors
    ///
    /// Maps [`StreamHandlerError`] variants to the appropriate
    /// [`AgentError`] variants:
    /// - [`Cancelled`](StreamHandlerError::Cancelled) → [`AgentError::Cancelled`]
    /// - [`InitFailed`](StreamHandlerError::InitFailed) → [`AgentError::Api`]
    /// - [`StreamFailed`](StreamHandlerError::StreamFailed) → [`AgentError::Api`]
    /// - [`FallbackFailed`](StreamHandlerError::FallbackFailed) → [`AgentError::Api`]
    async fn stream_turn_via_handler(
        &self,
        handler: &StreamHandler,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), AgentError> {
        let system = self.config.system_prompt.clone();
        let tool_schemas = self.build_tool_schemas();
        let result = handler
            .stream_turn(
                &*self.client,
                self.conversation.clone(),
                system,
                tool_schemas,
                &self.cancelled,
            )
            .await
            .map_err(Self::map_handler_error)?;
        Ok((result.message, result.usage, result.stop_reason))
    }

    /// Map a [`StreamHandlerError`] to an [`AgentError`].
    ///
    /// Preserves cancellation semantics —
    /// [`StreamHandlerError::Cancelled`] maps to [`AgentError::Cancelled`].
    /// All other variants map to [`AgentError::Api`] with a descriptive
    /// message.
    fn map_handler_error(error: StreamHandlerError) -> AgentError {
        match error {
            StreamHandlerError::Cancelled => AgentError::Cancelled,
            StreamHandlerError::InitFailed(outcome) => {
                AgentError::Api(format!("stream init failed: {outcome}"))
            }
            StreamHandlerError::StreamFailed(outcome) => {
                AgentError::Api(format!("stream failed: {outcome}"))
            }
            StreamHandlerError::FallbackFailed {
                stream_outcome,
                fallback_error,
            } => AgentError::Api(format!(
                "stream ({stream_outcome}) and fallback failed: {fallback_error}"
            )),
        }
    }
}
