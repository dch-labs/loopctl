//! Streaming — send the conversation to the LLM API and accumulate the response.
//!
//! When a [`StreamHandler`](crate::stream::handler::StreamHandler) is configured,
//! delegates to it for resilient streaming (retry, timeout, fallback). Otherwise,
//! uses basic inline logic.

use super::{
    ApiClient, BareLoop, LoopError, Message, StreamAccumulator, StreamEvent, StreamStopReason,
    Usage,
};
use crate::capabilities::StreamCapable;
use crate::observer::{TextDeltaContext, ThinkingDeltaContext};
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
    /// Returns [`LoopError::Api`] if any stream event is an error. When a
    /// [`StreamHandler`](crate::stream::handler::StreamHandler) is configured,
    /// may also return [`LoopError::Cancelled`] if the handler's cancel-aware
    /// `select!` fires mid-stream. The inline path does not check cancellation
    /// itself — that is handled by the `select!` in `process_turn`, which drops
    /// this future if cancelled.
    pub(super) async fn stream_turn(
        &self,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        // Delegate to StreamHandler if configured.
        if let Some(handler) = self.managers.stream_handler() {
            return self.stream_turn_via_handler(handler).await;
        }

        // Inline streaming (no handler).
        let system = self.config.system_prompt.clone();
        let tool_schemas = self.build_tool_schemas();
        // Clone the conversation history for the API request. The `ApiClient`
        // trait requires `'static` streams (it takes ownership of the
        // messages), so a clone is unavoidable here. The in-memory clone is
        // O(n) in the number of messages but is typically dwarfed by the
        // cost of serialising the messages into an HTTP request body. For
        // very long sessions (>200 turns with large tool outputs), consider
        // enabling auto-compaction to bound the history size.
        let mut stream = self.client.stream_messages_with_options(
            self.conversation.clone(),
            system,
            tool_schemas,
            self.request_options.clone(),
        );
        let mut accumulator = StreamAccumulator::new();
        let mut stop_reason = StreamStopReason::EndTurn;
        loop {
            let event_result = stream.next().await;

            match event_result {
                Some(Ok(event)) => {
                    if let StreamEvent::IndexedDelta(indexed_delta) = &event
                        && let crate::stream::DeltaPart::Text { text } = &indexed_delta.delta
                    {
                        if let Some(ref streamer) = self.text_streamer {
                            streamer(text);
                        }
                        self.managers.observers().on_text_delta(&TextDeltaContext {
                            turn: self.budget.total_turns,
                            delta: text.clone(),
                        });
                    }

                    if let StreamEvent::IndexedDelta(indexed_delta) = &event
                        && let crate::stream::DeltaPart::Thinking { text } = &indexed_delta.delta
                    {
                        self.managers
                            .observers()
                            .on_thinking_delta(&ThinkingDeltaContext {
                                turn: self.budget.total_turns,
                                delta: text.clone(),
                            });
                    }

                    if let StreamEvent::MessageDelta(delta) = &event
                        && let Some(ref reason_str) = delta.delta.stop_reason
                    {
                        stop_reason =
                            StreamStopReason::from_api_str(reason_str).unwrap_or(stop_reason);
                    }
                    accumulator
                        .process(&event)
                        .map_err(|e| LoopError::Api(format!("stream accumulation error: {e}")))?;
                }
                Some(Err(api_error)) => {
                    return Err(LoopError::Api(api_error.to_string()));
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
    /// [`LoopError`] variants:
    /// - [`Cancelled`](StreamHandlerError::Cancelled) → [`LoopError::Cancelled`]
    /// - [`InitFailed`](StreamHandlerError::InitFailed) → [`LoopError::Api`]
    /// - [`StreamFailed`](StreamHandlerError::StreamFailed) → [`LoopError::Api`]
    /// - [`FallbackFailed`](StreamHandlerError::FallbackFailed) → [`LoopError::Api`]
    /// - [`RateLimitEscalation`](StreamHandlerError::RateLimitEscalation) →
    ///   [`LoopError::RateLimitEscalation`]
    async fn stream_turn_via_handler(
        &self,
        handler: &StreamHandler,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
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
                prior: _,
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
    use crate::stream::handler::{DetectedRateLimit, RateLimitKind, StreamOutcome};

    // Minimal ApiClient so the `BareLoop<C>` associated fn is callable.
    struct StubClient;
    impl ApiClient for StubClient {
        fn model(&self) -> String {
            "stub".to_string()
        }
        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>,
        > {
            Box::pin(async { Ok(serde_json::json!({})) })
        }
    }

    #[test]
    fn map_handler_error_escalation() {
        let prior = StreamOutcome::RateLimited {
            detail: DetectedRateLimit {
                kind: RateLimitKind::RateLimited,
                retry_after: Some(std::time::Duration::from_secs(12)),
                message: "slow down".to_string(),
            },
            has_partial_data: false,
            events_processed: 0,
        };
        let mapped =
            BareLoop::<StubClient>::map_handler_error(StreamHandlerError::RateLimitEscalation {
                attempts: 3,
                retry_after: Some(std::time::Duration::from_secs(12)),
                prior,
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
