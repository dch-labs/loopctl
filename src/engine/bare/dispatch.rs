//! Tool dispatch — execute tool calls requested by the model.
//!
//! Sequential tool execution with reflection and recovery on errors, hook
//! interception, health recording, and middleware pipeline support.

#[cfg(feature = "hooks")]
use super::HookAction;
use super::{
    ApiClient, Arc, BareLoop, Duration, Instant, LoopError, PermissionCheck, RecoveryAction,
    ReflectionContext, ToolCall, ToolContent, ToolContext, ToolDispatchContext, ToolDispatchResult,
    ToolPipeline,
};
#[cfg(feature = "hooks")]
use super::{PostToolUseContext, PreToolUseContext};
#[cfg(feature = "tool_health")]
use crate::capabilities::HealthTrackable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
use crate::capabilities::PipelineAware;
use crate::detection::loop_detector::{self, Operation};
use crate::observer::{ToolPostContext, ToolPreContext};
use crate::reflection::{Correction, CorrectionResult};

use futures::FutureExt;
use std::panic::AssertUnwindSafe;

/// Result of deciding what to do after a tool error during recovery.
///
/// Distinguishes between returning a soft-error result (the tool failed,
/// but the session should continue) and a hard cancellation (the user
/// cancelled during the backoff sleep).
enum RecoveryOutcome {
    /// Return this soft-error result to the caller.
    SoftError(ToolDispatchResult),
}

impl<C: ApiClient> BareLoop<C> {
    /// Execute tool calls and return results.
    ///
    /// Dispatch a batch of tool calls sequentially.
    ///
    /// Each [`ToolCall`] is executed one at a time via
    /// [`dispatch_tool_with_recovery`](Self::dispatch_tool_with_recovery).
    /// A tool that is not found in the registry produces a soft error result
    /// (not a hard [`LoopError`]), allowing the model to recover.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if any tool dispatch fails with a hard error
    /// (e.g. loop detection forces an abort).
    pub(super) async fn dispatch_tools(
        &self,
        tool_calls: &[ToolCall],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, LoopError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for tc in tool_calls {
            let result = self.dispatch_tool_with_recovery(tc, turn_idx).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// Dispatch a single tool call with reflection and recovery on errors.
    ///
    /// If the tool succeeds, returns immediately. On failure, consults the
    /// [`Reflector`](crate::reflection::Reflector) and
    /// [`RecoveryStrategy`](crate::reflection::RecoveryStrategy) to decide
    /// whether to retry, skip, ask the user, or fail.
    ///
    /// Each attempt fires [`on_tool_pre`](crate::observer::LoopObserver::on_tool_pre)
    /// before dispatch and [`on_tool_post`](crate::observer::LoopObserver::on_tool_post)
    /// after, so observers always see a complete lifecycle pair.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if the detection manager signals a hard stop
    /// (e.g. [`LoopError::LoopDetected`]).
    async fn dispatch_tool_with_recovery(
        &self,
        tc: &ToolCall,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, LoopError> {
        let tool_context = self.build_tool_context();
        let mut attempt: u32 = 0;
        let mut tc = tc.clone();

        loop {
            self.fire_tool_pre(turn_idx, &tc);

            if let Some(blocked) = self.check_pre_tool_use_hooks(&tc, turn_idx) {
                self.fire_tool_post(turn_idx, &tc, &blocked);
                return Ok(blocked);
            }

            if let Some(blocked) = self.pre_detection(&tc, turn_idx)? {
                self.fire_tool_post(turn_idx, &tc, &blocked);
                return Ok(blocked);
            }

            let start = Instant::now();
            let tool_result = self
                .dispatch_tool(&tc, &tool_context, start, turn_idx)
                .await?;

            self.post_detection(&tc, &tool_result);
            self.fire_tool_post(turn_idx, &tc, &tool_result);
            self.notify_post_tool_use_hooks(&tc, &tool_result, turn_idx);
            self.record_tool_health(tc.tool.as_str(), &tool_result);

            if !tool_result.is_error {
                return Ok(tool_result);
            }

            match self
                .recovery_wait_or_return(&tc, &tool_result, attempt)
                .await
            {
                Ok((next_attempt, correction)) => {
                    attempt = next_attempt;
                    Self::apply_correction_if_present(&mut tc, correction, &tool_result);
                }
                Err(RecoveryOutcome::SoftError(returned_result)) => return Ok(returned_result),
            }
        }
    }

    /// Fire [`on_tool_pre`](crate::observer::LoopObserver::on_tool_pre) for
    /// the tool call about to be dispatched.
    fn fire_tool_pre(&self, turn_idx: usize, tc: &ToolCall) {
        self.managers.observers().on_tool_pre(&ToolPreContext {
            turn: turn_idx,
            tool: tc.tool.clone(),
            tool_call_id: tc.id.clone(),
        });
    }

    /// Fire [`on_tool_post`](crate::observer::LoopObserver::on_tool_post).
    ///
    /// Called for both successful dispatches and blocked paths (hooks,
    /// detection) so observers always see a complete `pre` / `post` pair.
    fn fire_tool_post(&self, turn_idx: usize, tc: &ToolCall, result: &ToolDispatchResult) {
        self.managers.observers().on_tool_post(&ToolPostContext {
            turn: turn_idx,
            tool: tc.tool.clone(),
            result_hash: loop_detector::hash_result(&result.output.to_string()),
            is_error: result.is_error,
            duration: result.duration,
        });
    }

    /// Apply a correction produced by the recovery strategy, if any.
    ///
    /// The correction modifies the tool call's input before the next retry
    /// attempt. If the correction cannot be applied, logs a warning and
    /// proceeds with the original input.
    fn apply_correction_if_present(
        tc: &mut ToolCall,
        correction: Option<Correction>,
        tool_result: &ToolDispatchResult,
    ) {
        let Some(correction) = correction else { return };
        if let CorrectionResult::Failed(msg) = tc.apply_correction(&correction, tool_result) {
            tracing::warn!(
                tool = %tc.tool,
                error = %msg,
                "correction failed to produce a usable retry"
            );
        }
    }

    /// Record the tool call's input signature and check for loop patterns.
    ///
    /// Returns `Some(blocked_result)` if loop detection blocks the call,
    /// or `None` if dispatch should proceed.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::LoopDetected`] if the detection manager signals
    /// a hard stop.
    fn pre_detection(
        &self,
        tc: &ToolCall,
        turn_idx: usize,
    ) -> Result<Option<ToolDispatchResult>, LoopError> {
        let operation = Operation::from_input_with_signature(
            &tc.tool,
            &tc.input,
            self.managers.detection.signature(),
        );
        let pattern = self.managers.detection.record_operation(operation);

        // Inline loop/convergence detection. Some(error) forces a hard stop —
        // propagate it so the agent loop terminates. None means no detection
        // fired and dispatch continues.
        match self.managers.handle_detected_pattern(&pattern, turn_idx) {
            Some(e) => Err(e),
            None => Ok(None),
        }
    }

    /// Record the tool result's output hash for loop detection.
    ///
    /// Lets the detector distinguish "same input, same output" (stuck) from
    /// "same input, different output" (progress).
    fn post_detection(&self, tc: &ToolCall, tool_result: &ToolDispatchResult) {
        let result_hash = match &tool_result.output {
            ToolContent::Text(t) => loop_detector::hash_result(t),
            ToolContent::Multipart(_) => None,
        };
        let operation = Operation::from_input_with_result_and_signature(
            &tc.tool,
            &tc.input,
            result_hash,
            self.managers.detection.signature(),
        );
        self.managers.detection.record_operation(operation);
    }

    /// Execute a single tool call.
    ///
    /// Tries the middleware pipeline first (if configured), then falls back
    /// to a direct registry lookup. Tool panics are caught and converted to
    /// error results. A tool not in the registry produces a soft error.
    ///
    /// Observer notifications are handled by the caller
    /// ([`dispatch_tool_with_recovery`](Self::dispatch_tool_with_recovery)).
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if loop detection forces a hard stop.
    async fn dispatch_tool(
        &self,
        tc: &ToolCall,
        tool_context: &ToolContext,
        start: Instant,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, LoopError> {
        if let Some(pipeline) = self.managers.pipeline() {
            return self
                .dispatch_via_pipeline(pipeline, tc, tool_context, turn_idx)
                .await;
        }

        let tool_result = if let Some(tool) = self.tools.get(&tc.tool) {
            let call_result = AssertUnwindSafe(tool.call(tc.input.clone(), tool_context))
                .catch_unwind()
                .await;
            match call_result {
                Ok(Ok(result)) => {
                    let duration = start.elapsed();
                    ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: result.payload,
                        is_error: result.is_error,
                        duration,
                        resolved_tool_name: tc.tool.clone(),
                    }
                }
                Ok(Err(e)) => {
                    let duration = start.elapsed();
                    let error_msg = e.to_string();
                    ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(error_msg),
                        is_error: true,
                        duration,
                        resolved_tool_name: tc.tool.clone(),
                    }
                }
                Err(panic_payload) => {
                    let duration = start.elapsed();
                    let msg = panic_payload
                        .downcast_ref::<&'static str>()
                        .map(std::string::ToString::to_string)
                        .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| {
                            format!("Tool '{}' panicked (unknown payload)", tc.tool)
                        });
                    tracing::error!(
                        tool = %tc.tool,
                        panic_message = %msg,
                        "tool panicked during execution"
                    );
                    ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(format!("Tool '{}' panicked: {msg}", tc.tool)),
                        is_error: true,
                        duration,
                        resolved_tool_name: tc.tool.clone(),
                    }
                }
            }
        } else {
            self.tool_not_found(tc)
        };

        Ok(tool_result)
    }

    /// Build a soft-error result for a tool that is not in the registry.
    ///
    /// The error message lists available tool names to help the model recover.
    fn tool_not_found(&self, tc: &ToolCall) -> ToolDispatchResult {
        let available: Vec<String> = self.tools.tool_names();
        let available_refs: Vec<&str> = available.iter().map(String::as_str).collect();
        let error = LoopError::tool_not_found(&tc.tool, &available_refs);
        let error_msg = error.to_string();
        ToolDispatchResult {
            tool_call_id: tc.id.clone(),
            output: ToolContent::Text(error_msg),
            is_error: true,
            duration: Duration::ZERO,
            resolved_tool_name: tc.tool.clone(),
        }
    }

    /// Decide whether to retry a failed tool or return the error as a soft result.
    ///
    /// Consults the [`Reflector`](crate::reflection::Reflector) and
    /// [`RecoveryStrategy`](crate::reflection::RecoveryStrategy). On
    /// [`Retry`](RecoveryAction::Retry), sleeps for the prescribed delay and
    /// returns the updated attempt count and optional [`Correction`]. On all
    /// other actions (`Skip`, `Fail`, `AskUser`), returns the original error
    /// result as a soft error.
    ///
    /// # Errors
    ///
    /// Returns `Err(RecoveryOutcome::SoftError)` when the recovery strategy
    /// decides not to retry.
    async fn recovery_wait_or_return(
        &self,
        tc: &ToolCall,
        tool_result: &ToolDispatchResult,
        attempt: u32,
    ) -> Result<(u32, Option<Correction>), RecoveryOutcome> {
        let (recovery_action, correction) = self.recover_tool_error(tc, tool_result, attempt).await;
        match recovery_action {
            RecoveryAction::Retry { delay } => {
                let next_attempt = attempt.saturating_add(1);
                tokio::time::sleep(delay).await;
                Ok((next_attempt, correction))
            }
            RecoveryAction::Skip(_) | RecoveryAction::AskUser(_) | RecoveryAction::Fail(_) => {
                Err(RecoveryOutcome::SoftError(tool_result.clone()))
            }
        }
    }

    /// Check pre-tool-use hooks and return a blocked result if any hook
    /// blocks or asks.
    ///
    /// Returns `Some(ToolDispatchResult)` with an error result if a hook
    /// blocked the call, or `None` if the call should proceed.
    #[cfg(feature = "hooks")]
    fn check_pre_tool_use_hooks(
        &self,
        tc: &ToolCall,
        turn_idx: usize,
    ) -> Option<ToolDispatchResult> {
        let executor = self.managers.hook_executor()?;
        let ctx = PreToolUseContext {
            tool_name: tc.tool.clone(),
            input: tc.input.clone(),
            session_id: self.config.session_id,
            turn_number: turn_idx,
        };
        match executor.check_pre_tool_use(&ctx) {
            HookAction::Allow => None,
            HookAction::Block { reason } => Some(ToolDispatchResult {
                tool_call_id: tc.id.clone(),
                output: ToolContent::Text(reason),
                is_error: true,
                duration: Duration::ZERO,
                resolved_tool_name: tc.tool.clone(),
            }),
            HookAction::Ask { message } => Some(ToolDispatchResult {
                tool_call_id: tc.id.clone(),
                output: ToolContent::Text(message),
                is_error: true,
                duration: Duration::ZERO,
                resolved_tool_name: tc.tool.clone(),
            }),
        }
    }

    #[cfg(not(feature = "hooks"))]
    fn check_pre_tool_use_hooks(
        &self,
        _tc: &ToolCall,
        _turn_idx: usize,
    ) -> Option<ToolDispatchResult> {
        None
    }

    /// Notify post-tool-use hooks with the execution result.
    #[cfg(feature = "hooks")]
    fn notify_post_tool_use_hooks(
        &self,
        tc: &ToolCall,
        tool_result: &ToolDispatchResult,
        turn_idx: usize,
    ) {
        let Some(executor) = self.managers.hook_executor() else {
            return;
        };
        let output_text = tool_result.output.to_string();
        let ctx = PostToolUseContext {
            tool_name: tc.tool.clone(),
            input: tc.input.clone(),
            output: output_text,
            is_error: tool_result.is_error,
            duration_ms: tool_result
                .duration
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            session_id: self.config.session_id,
            turn_number: turn_idx,
        };
        executor.notify_post_tool_use(&ctx);
    }

    #[cfg(not(feature = "hooks"))]
    fn notify_post_tool_use_hooks(
        &self,
        _tc: &ToolCall,
        _tool_result: &ToolDispatchResult,
        _turn_idx: usize,
    ) {
    }

    /// Record tool health (success or failure) in the health registry.
    #[cfg(feature = "tool_health")]
    fn record_tool_health(&self, tool_name: &str, tool_result: &ToolDispatchResult) {
        let Some(health) = self.managers.health_registry() else {
            return;
        };
        if tool_result.is_error {
            health.record_failure(tool_name, tool_result.duration);
        } else {
            health.record_success(tool_name, tool_result.duration);
        }
    }

    #[cfg(not(feature = "tool_health"))]
    fn record_tool_health(&self, _tool_name: &str, _tool_result: &ToolDispatchResult) {}

    /// Dispatch a tool call through the middleware pipeline.
    ///
    /// Builds a [`ToolDispatchContext`] and delegates to the pipeline's
    /// middleware chain (timeout, permissions, output limits, etc.).
    ///
    /// Observer notifications are handled by the caller
    /// ([`dispatch_tool_with_recovery`](Self::dispatch_tool_with_recovery)).
    ///
    /// # Errors
    ///
    /// Never returns an error — pipeline dispatch always produces a result
    /// (soft errors are returned as `Ok` with `is_error: true`).
    async fn dispatch_via_pipeline(
        &self,
        pipeline: &ToolPipeline,
        tc: &ToolCall,
        tool_context: &ToolContext,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, LoopError> {
        let ctx = ToolDispatchContext {
            tool_name: tc.tool.clone(),
            input: tc.input.clone(),
            call_id: tc.id.clone(),
            turn_number: turn_idx,
            cancel: Arc::clone(&self.cancelled),
            permission: PermissionCheck::Allow,
            tool_context: tool_context.clone(),
        };
        let dispatch_result = pipeline.invoke(ctx).await;
        Ok(ToolDispatchResult {
            tool_call_id: if dispatch_result.tool_call_id.is_empty() {
                tc.id.clone()
            } else {
                dispatch_result.tool_call_id
            },
            output: dispatch_result.output,
            is_error: dispatch_result.is_error,
            duration: dispatch_result.duration,
            resolved_tool_name: dispatch_result.resolved_tool_name,
        })
    }

    /// Analyse a tool error and decide on a recovery action.
    ///
    /// Calls [`Reflector::analyze`](crate::reflection::Reflector::analyze) to
    /// classify the failure, then
    /// [`RecoveryStrategy::decide`](crate::reflection::RecoveryStrategy::decide)
    /// to choose the action. If the reflector itself fails, conservatively
    /// returns [`RecoveryAction::Fail`].
    ///
    /// Returns the [`RecoveryAction`] and an optional [`Correction`] that the
    /// retry loop applies to the tool input before re-dispatching.
    async fn recover_tool_error(
        &self,
        tc: &ToolCall,
        result: &ToolDispatchResult,
        attempt: u32,
    ) -> (RecoveryAction, Option<Correction>) {
        let error_msg = match &result.output {
            ToolContent::Text(msg) => msg.clone(),
            ToolContent::Multipart(_) => result.output.to_string(),
        };
        let context = ReflectionContext {
            task: String::new(),
            attempt,
            max_attempts: Self::MAX_RECOVERY_ATTEMPTS,
        };

        let Ok(analysis) = self
            .reflector
            .analyze(&error_msg, &tc.tool, &tc.input, &context)
            .await
        else {
            // Reflector failed — conservatively fail.
            return (RecoveryAction::Fail(error_msg), None);
        };

        let correction = analysis.correction.clone();
        let action = self
            .recovery
            .decide(&analysis, attempt, Self::MAX_RECOVERY_ATTEMPTS)
            .await;
        (action, correction)
    }
}

#[cfg(test)]
#[allow(clippy::unnecessary_literal_bound)]
mod tests {
    use crate::api::error::ApiError;
    use crate::config::LoopConfig;
    use crate::engine::loop_core::ToolCall;
    use crate::message::ToolContent;
    use crate::tool::{
        Tool, ToolContext, ToolError, ToolOutput, ToolSchema, registry::ToolRegistry,
    };
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::Instant;

    use parking_lot::Mutex;

    use super::*;

    struct MockClient {
        model_name: Arc<Mutex<String>>,
    }

    impl MockClient {
        fn new(model: &str) -> Self {
            Self {
                model_name: Arc::new(Mutex::new(model.to_string())),
            }
        }
    }

    impl ApiClient for MockClient {
        fn model(&self) -> String {
            self.model_name.lock().clone()
        }
        fn set_model(&self, model: &str) -> bool {
            if model.trim().is_empty() {
                return false;
            }
            *self.model_name.lock() = model.to_string();
            true
        }
        fn stream_messages(
            &self,
            _history: Vec<crate::message::Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<
            Box<
                dyn futures::Stream<Item = Result<crate::stream::StreamEvent, ApiError>>
                    + Send
                    + 'static,
            >,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _history: Vec<crate::message::Message>,
            _system: Option<String>,
            _tools: Option<Vec<ToolSchema>>,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ApiError>> + Send + '_>>
        {
            Box::pin(async { Err(ApiError::http("not implemented")) })
        }
    }

    struct PanicTool;

    impl Tool for PanicTool {
        fn name(&self) -> &str {
            "panic_tool"
        }
        fn description(&self) -> &str {
            "Panics on call"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "panic_tool".into(),
                description: "Panics on call".into(),
                input_schema: Value::Object(serde_json::Map::new()),
            }
        }
        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { panic!("dispatch.rs panic tool") })
        }
    }

    fn echo_fn(
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>> {
        Box::pin(async { Ok(ToolOutput::text("ok")) })
    }

    fn make_loop(tools: ToolRegistry) -> BareLoop<MockClient> {
        let config = LoopConfig::default();
        let client = Arc::new(MockClient::new("test"));
        BareLoop::new(client, tools, config)
    }

    #[tokio::test]
    async fn dispatch_tool_catches_panic() {
        let mut registry = ToolRegistry::new();
        registry.register(PanicTool);
        let bare = make_loop(registry);

        let tc = ToolCall {
            id: "tc1".into(),
            tool: "panic_tool".into(),
            input: Value::Null,
        };
        let tool_context = ToolContext::default();
        let start = Instant::now();

        let result = bare.dispatch_tool(&tc, &tool_context, start, 0).await;

        assert!(result.is_ok(), "panic should be caught, not propagated");
        let dispatch_result = result.unwrap();
        assert!(dispatch_result.is_error);
        match &dispatch_result.output {
            ToolContent::Text(text) => {
                assert!(text.contains("panicked"), "expected panic message: {text}");
            }
            ToolContent::Multipart(_) => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn dispatch_tool_normal_tool_works() {
        let mut registry = ToolRegistry::new();
        registry.register(crate::tool::FnTool::new(
            "echo".into(),
            "echo".into(),
            Value::Object(serde_json::Map::new()),
            echo_fn,
        ));
        let bare = make_loop(registry);

        let tc = ToolCall {
            id: "tc1".into(),
            tool: "echo".into(),
            input: Value::Null,
        };
        let tool_context = ToolContext::default();
        let start = Instant::now();

        let result = bare.dispatch_tool(&tc, &tool_context, start, 0).await;

        assert!(result.is_ok());
        let dispatch_result = result.unwrap();
        assert!(!dispatch_result.is_error);
        match &dispatch_result.output {
            ToolContent::Text(text) => assert_eq!(text, "ok"),
            ToolContent::Multipart(_) => panic!("expected Text"),
        }
    }
}
