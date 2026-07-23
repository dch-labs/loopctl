//! Streaming — send the conversation to the LLM API and accumulate the response.
//!
//! Streaming always routes through a [`StreamHandler`](crate::stream::handler::StreamHandler).
//! When no handler is configured, the engine uses
//! [`StreamHandler::passthrough_default`](crate::stream::handler::StreamHandler::passthrough_default)
//! — a no-resilience handler that yields the raw provider stream with no retries,
//! timeouts, or fallback (equivalent to the pre-redesign inline path). Configuring
//! a handler via [`set_stream_handler()`](BareLoop::set_stream_handler) opts into
//! retry, timeout, fallback, and rate-limit handling.

use super::{
    ApiClient, BareLoop, LoopError, Message, StreamAccumulator, StreamEvent, StreamStopReason,
    Usage,
};
use crate::capabilities::StreamCapable;
use crate::observer::{TextDeltaContext, ThinkingDeltaContext};
use crate::stream::handler::{HandlerEvent, StreamHandlerError};
use futures::StreamExt;

impl<C: ApiClient> BareLoop<C> {
    /// Stream one turn from the API, accumulating the response.
    ///
    /// Always routes through a [`StreamHandler`] — when none is configured,
    /// [`passthrough_default`](crate::stream::handler::StreamHandler::passthrough_default)
    /// is used (no retries, no timeouts, no fallback). Configure a handler via
    /// [`set_stream_handler()`](BareLoop::set_stream_handler) to opt into
    /// resilient streaming.
    ///
    /// Sends the current conversation history to the LLM API via
    /// [`ApiClient::stream_messages_with_options`] and uses a
    /// [`StreamAccumulator`] to collect the events into a single [`Message`].
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
    /// Returns [`LoopError::Api`] if any stream event is an error. May also
    /// return [`LoopError::Cancelled`] if the handler's cancel-aware `select!`
    /// fires mid-stream.
    pub(super) async fn stream_turn(
        &self,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        let handler = self.managers.stream_handler();
        let mut stream = handler.stream_turn(
            &*self.client,
            crate::api::StreamRequest::new(self.conversation.clone())
                .with_system(self.config.system_prompt.clone())
                .with_tools(self.build_tool_schemas()),
            self.request_options.clone(),
            &self.cancelled,
        );

        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;

        while let Some(result) = stream.next().await {
            match result.map_err(Self::map_handler_error)? {
                HandlerEvent::Stream(ev) => {
                    self.dispatch_stream_event(&ev, &mut accumulator, &mut stop_reason)?;
                }
                HandlerEvent::AttemptReset => {
                    accumulator = StreamAccumulator::new();
                    stop_reason = StreamStopReason::EndTurn;
                }
                HandlerEvent::Fallback {
                    message,
                    stop_reason: fallback_stop_reason,
                } => {
                    return Ok((message, None, fallback_stop_reason));
                }
            }
        }

        let usage = accumulator.usage().copied();
        Ok((accumulator.build(), usage, stop_reason))
    }

    /// Dispatch one stream event: fire observer callbacks
    /// (`text_streamer` + `on_text_delta` for [`DeltaPart::Text`],
    /// `on_thinking_delta` for [`DeltaPart::Thinking`]), extract the stop
    /// reason from [`MessageDelta`] events, then fold the event into the
    /// accumulator.
    ///
    /// Shared by all `HandlerEvent::Stream` events regardless of whether the
    /// source is the passthrough handler or a configured resilient handler.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Api`] if the event cannot be accumulated (e.g.
    /// malformed tool-call JSON in a `PartStop` boundary).
    ///
    /// [`DeltaPart::Text`]: crate::stream::DeltaPart::Text
    /// [`DeltaPart::Thinking`]: crate::stream::DeltaPart::Thinking
    fn dispatch_stream_event(
        &self,
        event: &StreamEvent,
        accumulator: &mut StreamAccumulator,
        stop_reason: &mut StreamStopReason,
    ) -> Result<(), LoopError> {
        if let StreamEvent::IndexedDelta(d) = event
            && let crate::stream::DeltaPart::Text { text } = &d.delta
        {
            if let Some(streamer) = &self.text_streamer {
                streamer(text);
            }
            self.managers.observers().on_text_delta(&TextDeltaContext {
                turn: self.budget.total_turns,
                delta: text.clone(),
            });
        }
        if let StreamEvent::IndexedDelta(d) = event
            && let crate::stream::DeltaPart::Thinking { text } = &d.delta
        {
            self.managers
                .observers()
                .on_thinking_delta(&ThinkingDeltaContext {
                    turn: self.budget.total_turns,
                    delta: text.clone(),
                });
        }
        if let StreamEvent::MessageDelta(d) = event
            && let Some(reason_str) = &d.delta.stop_reason
        {
            *stop_reason = StreamStopReason::from_api_str(reason_str).unwrap_or(*stop_reason);
        }
        accumulator
            .process(event)
            .map_err(|e| LoopError::Api(format!("stream accumulation error: {e}")))
    }

    /// Map a [`StreamHandlerError`] to an [`LoopError`].
    ///
    /// Preserves cancellation semantics —
    /// [`StreamHandlerError::Cancelled`] maps to [`LoopError::Cancelled`].
    /// All other variants map to [`LoopError::Api`] with a descriptive
    /// message.
    fn map_handler_error(error: StreamHandlerError) -> LoopError {
        match error {
            StreamHandlerError::Cancelled => LoopError::Cancelled,
            StreamHandlerError::InitFailed(outcome) => {
                LoopError::Api(format!("stream init failed: {outcome}"))
            }
            StreamHandlerError::StreamFailed(outcome) => {
                LoopError::Api(format!("stream failed: {outcome}"))
            }
            StreamHandlerError::FallbackFailed {
                stream_outcome,
                fallback_error,
            } => LoopError::Api(format!(
                "stream ({stream_outcome}) and fallback failed: {fallback_error}"
            )),
            StreamHandlerError::RateLimitEscalation {
                attempts,
                retry_after,
            } => LoopError::RateLimitEscalation {
                attempts,
                retry_after,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::ApiError;

    // Minimal ApiClient so the `BareLoop<C>` associated fn is callable.
    struct StubClient;
    impl ApiClient for StubClient {
        fn model(&self) -> String {
            "stub".to_string()
        }
        fn stream_messages(
            &self,
            _request: crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _request: crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>,
        > {
            Box::pin(async { Ok(serde_json::json!({})) })
        }
    }

    #[test]
    fn map_handler_error_escalation() {
        let mapped =
            BareLoop::<StubClient>::map_handler_error(StreamHandlerError::RateLimitEscalation {
                attempts: 3,
                retry_after: Some(std::time::Duration::from_secs(12)),
            });
        match mapped {
            LoopError::RateLimitEscalation {
                attempts,
                retry_after,
            } => {
                assert_eq!(attempts, 3);
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(12)));
            }
            other => panic!("expected RateLimitEscalation, got {other:?}"),
        }
    }
}
