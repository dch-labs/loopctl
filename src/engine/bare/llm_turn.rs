//! The LLM-turn driver arm — both streaming and non-streaming turn paths.
//!
//! This module owns everything that happens when the [`LoopMachine`] requests
//! a `CallLLM` step: building the request (history + contributors + system
//! prompt + tool schemas), driving the provider (streaming via
//! [`StreamHandler`](crate::stream::handler::StreamHandler), or non-streaming
//! via [`ApiClient::create_message_with_options`]), and recording the turn
//! outcome with the fallback manager + observers (via the `record_*` helpers
//! in the `emission` submodule).
//!
//! Both paths share [`build_turn_request`](BareLoop::build_turn_request) so the
//! request shape is defined exactly once.
//!
//! [`LoopMachine`]: crate::engine::core::LoopMachine

#[cfg(feature = "streaming")]
use super::Run;
use super::{ApiClient, BareLoop, LoopError, Message};
use crate::api::StreamRequest;
#[cfg(feature = "streaming")]
use crate::capabilities::StreamCapable;
use crate::capabilities::{Detectable, FallbackCapable};
use crate::detection::{ConvergenceAction, DetectedPattern};
#[cfg(feature = "streaming")]
use crate::observer::{TextDeltaContext, ThinkingDeltaContext};
#[cfg(feature = "streaming")]
use crate::stream::handler::{HandlerEvent, StreamHandlerError};
#[cfg(feature = "streaming")]
use crate::stream::{StreamAccumulator, StreamEvent};
use crate::stream::{StreamStopReason, Usage};
#[cfg(feature = "streaming")]
use futures::StreamExt;

impl<C: ApiClient> BareLoop<C> {
    /// Build tool schemas for the API request.
    ///
    /// Collects all tool schemas from the [`ToolRegistry`] and returns
    /// them as `Some(Vec<ToolSchema>)`, or `None` if the registry is empty.
    pub(super) fn build_tool_schemas(&self) -> Option<Vec<crate::tool::ToolSchema>> {
        let schemas = self.tools.all_schemas();
        if schemas.is_empty() {
            None
        } else {
            Some(schemas)
        }
    }

    /// Build the per-turn [`StreamRequest`] shared by both turn paths.
    ///
    /// Merges the transient contributor messages with the machine's full
    /// history, attaches the session system prompt and the current tool
    /// schemas. Defined once here so the streaming and non-streaming paths
    /// cannot drift on request shape.
    pub(super) fn build_turn_request(&self, messages: Vec<Message>) -> StreamRequest {
        let mut messages = messages;
        messages.extend(self.machine.full_history());
        StreamRequest::new(messages)
            .with_system(self.session.config.system_prompt.clone())
            .with_tools(self.build_tool_schemas())
    }

    /// The single model the fallback manager resolves this turn's
    /// request to.
    ///
    /// While the breaker is closed this is the manager's primary
    /// (`None` only when the manager is unconfigured, which leaves the
    /// request on the client's own model or the host's override,
    /// byte-identical to a loop without fallback). While tripped this
    /// is the active fallback model; during recovery, the primary. With
    /// a configured manager the resolution is exclusive: outbound
    /// requests, breaker bookkeeping, recovery probes, and observer
    /// contexts all derive from this one value, overriding any
    /// host-provided per-request model.
    pub(super) fn routed_model(&self) -> Option<String> {
        let manager = self.managers.fallback();
        match manager.state() {
            crate::fallback::FallbackState::Primary => manager.original_model(),
            _ => manager.active_model(),
        }
    }

    /// The request options for this turn with the fallback routing applied.
    ///
    /// Applies [`routed_model`](Self::routed_model) as the per-request
    /// model override; anything the host set on `request_options` passes
    /// through untouched when no manager is configured.
    fn turn_request_options(&self) -> crate::structured::RequestOptions {
        let mut options = self.request_options.clone();
        if let Some(model) = self.routed_model() {
            options.model = Some(model);
        }
        options
    }

    /// Fire [`on_model_switched`] when the routed model changed since the
    /// last request, and remember the current one.
    ///
    /// One mechanical signal for every cause of a model change — trip,
    /// chain advance, recovery — instead of observers piecing it together
    /// from breaker callbacks. No-op while no manager routes models.
    pub(super) fn note_routed_model(&mut self) {
        let served = self.routed_or_client_model();
        if self
            .last_routed_model
            .as_ref()
            .is_some_and(|m| *m != served)
        {
            let from = self.last_routed_model.clone().unwrap_or(served.clone());
            self.managers
                .observers()
                .on_model_switched(&crate::observer::ModelSwitchedContext {
                    from,
                    to: served.clone(),
                });
        }
        self.last_routed_model = Some(served);
    }

    /// Dispatch one LLM turn according to [`turn_mode`](BareLoop::turn_mode).
    ///
    /// Single entry point for the run loop's `CallLLM` arm. Guards the
    /// already-cancelled case once here so neither turn path polls its
    /// provider future on a dead run.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if the run is already cancelled;
    /// otherwise propagates the selected turn path's error.
    pub(super) async fn do_turn(
        &mut self,
        turn: usize,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        if self.cancelled.is_cancelled() {
            return Err(LoopError::Cancelled);
        }
        match self.turn_mode {
            #[cfg(feature = "streaming")]
            super::TurnMode::Streaming => self.do_stream(turn, messages).await,
            super::TurnMode::NonStreaming => self.do_create_message(turn, messages).await,
        }
    }

    /// Request one assistant response via the non-streaming API.
    ///
    /// Builds the request via [`build_turn_request`](Self::build_turn_request),
    /// then calls [`ApiClient::create_message_with_options`] and races it
    /// against both [`CancelSignal::notified`](crate::cancel::CancelSignal::notified)
    /// and the configured total-stream timeout. The timeout reuses
    /// [`StreamTimeoutConfig::total_stream_timeout`] so both turn paths share one
    /// wall-clock budget; the streaming path enforces it inside `StreamHandler`,
    /// this path enforces it here. Records success/failure via the shared
    /// `record_*` helpers.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if cancellation wins the `select!`;
    /// [`LoopError::Api`] with a timeout message if the deadline elapses;
    /// otherwise the provider error mapped to [`LoopError::Api`].
    async fn do_create_message(
        &mut self,
        turn: usize,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        let request = self.build_turn_request(messages);
        let cancel = std::sync::Arc::clone(&self.cancelled);
        let client = &self.client;
        let options = self.turn_request_options();
        let timeout = self.turn_timeout();
        let result = tokio::select! {
            biased;
            () = cancel.notified() => Err(LoopError::Cancelled),
            () = async {
                if timeout == std::time::Duration::MAX {
                    std::future::pending::<()>().await;
                } else {
                    tokio::time::sleep(timeout).await;
                }
            } => Err(LoopError::Api(format!("request timed out after {timeout:?}"))),
            res = client.create_message_with_options(&request, options) => {
                res.map_err(|e| LoopError::Api(e.to_string()))
            }
        };
        match result {
            Ok(response) => {
                self.record_turn_success(turn, response.usage.as_ref());
                Ok((response.message, response.usage, response.stop_reason))
            }
            Err(e) => Err(self.record_turn_failure(turn, e)),
        }
    }

    /// Stream one assistant response via the [`StreamHandler`] and apply
    /// post-stream bookkeeping.
    ///
    /// Delegates the actual streaming to [`stream_turn`](Self::stream_turn),
    /// then records success/failure via the shared `record_*` helpers.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`stream_turn`](Self::stream_turn) returns.
    #[cfg(feature = "streaming")]
    async fn do_stream(
        &mut self,
        turn: usize,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        match self.stream_turn(messages).await {
            Ok((msg, usage, stop)) => {
                self.record_turn_success(turn, usage.as_ref());
                Ok((msg, usage, stop))
            }
            Err(e) => Err(self.record_turn_failure(turn, e)),
        }
    }

    /// Stream one turn from the API, accumulating the response.
    ///
    /// Always routes through a [`StreamHandler`](crate::stream::handler::StreamHandler)
    /// — when none is configured, [`passthrough_default`](crate::stream::handler::StreamHandler::passthrough_default)
    /// is used (no retries, no timeouts, no fallback).
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Api`] if any stream event is an error, or
    /// [`LoopError::Cancelled`] if the handler's cancel-aware path fires.
    #[cfg(feature = "streaming")]
    pub(super) async fn stream_turn(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<Usage>, StreamStopReason), LoopError> {
        let handler = self.managers.stream_handler();
        let request = self.build_turn_request(messages);
        let mut stream = handler.stream_turn(
            &*self.client,
            &request,
            self.turn_request_options(),
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
                    usage: fallback_usage,
                } => {
                    return Ok((message, fallback_usage, fallback_stop_reason));
                }
            }
        }

        let usage = accumulator.usage().copied();
        Ok((accumulator.build(), usage, stop_reason))
    }

    /// Dispatch one stream event: fire per-delta observer callbacks
    /// (`on_text_delta`, `on_thinking_delta`) and the `text_streamer`, extract
    /// the stop reason, then fold the event into the accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Api`] if the event cannot be accumulated.
    #[cfg(feature = "streaming")]
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
                streamer(text.as_str());
            }
            self.managers.observers().on_text_delta(&TextDeltaContext {
                turn: self.session.current_run().map_or(0, Run::turn_count),
                delta: text.clone(),
            });
        }
        if let StreamEvent::IndexedDelta(d) = event
            && let crate::stream::DeltaPart::Thinking { text } = &d.delta
        {
            self.managers
                .observers()
                .on_thinking_delta(&ThinkingDeltaContext {
                    turn: self.session.current_run().map_or(0, Run::turn_count),
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
    /// Consult the detection manager and, if a pattern forced a hard stop,
    /// return the error for the driver loop to act on.
    ///
    /// Returns `None` when no pattern fired (the driver continues with tool
    /// extraction and dispatch), or `Some(err)` when detection aborted the
    /// session. Does **not** set the terminal state — the caller's single
    /// `set_error_state` call in the `run()` error path does that, matching
    /// every other handler error.
    pub(super) fn apply_loop_detection(
        &self,
        current_turn: usize,
        pattern: &DetectedPattern,
    ) -> Option<LoopError> {
        self.managers.notify_detected_pattern(pattern, current_turn);
        self.decide_detected_pattern(pattern)
    }

    /// Decide whether a detected pattern warrants aborting the loop.
    ///
    /// Reads the detection config (`stop_threshold`, `on_converge`) to
    /// determine if the pattern is severe enough to halt. Returns
    /// `Some(LoopError)` to abort, `None` to continue. A loop hard-stop
    /// consumes the pattern: the detector's window is cleared on the
    /// way out, so the run's failure does not block the next run's
    /// first dispatch with stale repetitions (a `stop_threshold` of 0
    /// disables hard stops entirely, mirroring the detector's own
    /// rule).
    pub(super) fn decide_detected_pattern(&self, pattern: &DetectedPattern) -> Option<LoopError> {
        let config = self.managers.detection().config();
        match pattern {
            DetectedPattern::NoPattern => None,
            DetectedPattern::LoopDetected {
                repetitions,
                pattern_description,
            } => {
                if config.stop_threshold > 0 && *repetitions >= config.stop_threshold {
                    tracing::error!(
                        repetitions,
                        pattern = %pattern_description,
                        "stopping agent: loop threshold exceeded"
                    );
                    self.managers.detection().loop_detector().clear();
                    Some(LoopError::LoopDetected {
                        message: format!("{pattern_description} repeated {repetitions} times"),
                    })
                } else {
                    None
                }
            }
            DetectedPattern::ConvergenceDetected { .. } => match config.on_converge {
                ConvergenceAction::Stop => Some(LoopError::LoopDetected {
                    message: "agent stopped: convergence detected".into(),
                }),
                ConvergenceAction::AskUser => Some(LoopError::UserInputRequired {
                    message: "convergence detected, user input needed".into(),
                }),
                ConvergenceAction::Warn
                | ConvergenceAction::Compact
                | ConvergenceAction::SwitchPhase => None,
            },
        }
    }

    /// Map a [`StreamHandlerError`] to a [`LoopError`].
    ///
    /// Preserves cancellation semantics —
    /// [`StreamHandlerError::Cancelled`] maps to [`LoopError::Cancelled`].
    /// All other variants map to [`LoopError::Api`] with a descriptive
    /// message.
    #[cfg(feature = "streaming")]
    pub(super) fn map_handler_error(error: StreamHandlerError) -> LoopError {
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

#[cfg(all(test, feature = "streaming"))]
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
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::api::NonStreamingResponse, ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
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
