//! Tool dispatch phase — execute tool calls requested by the model.
//!
//! Extracted from [`BareLoop`] to isolate the tool dispatch concern.
//! Handles sequential tool execution, reflection/recovery on errors,
//! hook interception, health recording, and middleware pipeline dispatch.

#[cfg(feature = "hooks")]
use super::HookAction;
use super::{
    AgentError, ApiClient, Arc, BareLoop, Duration, Instant, PermissionCheck, RecoveryAction,
    ReflectionContext, ToolCallInfo, ToolContent, ToolContentPart, ToolContext,
    ToolDispatchContext, ToolDispatchResult, ToolPipeline,
};
#[cfg(feature = "hooks")]
use super::{PostToolUseContext, PreToolUseContext};
use crate::loop_control::loop_detector;

impl<C: ApiClient> BareLoop<C> {
    /// Execute tool calls and return results.
    ///
    /// Iterates over each [`ToolCallInfo`] extracted from the assistant
    /// message, looks up the corresponding tool in the [`ToolRegistry`],
    /// and invokes it. Each result is wrapped in a [`ToolDispatchResult`].
    ///
    /// Tool execution is **sequential** so that cancellation can be
    /// checked between invocations. A tool that is not found in the
    /// registry produces a soft error result (not a hard [`AgentError`]),
    /// allowing the model to recover.
    ///
    /// When a tool returns an error (execution failure or not-found),
    /// the framework consults the [`Reflector`] and [`RecoveryStrategy`]
    /// to decide whether to retry, skip, ask user, or fail. Retry
    /// attempts use the delay specified by the [`RecoveryAction`].
    ///
    /// Observers are notified before and after each tool invocation via
    /// [`on_tool_call`](AgentObserver::on_tool_call) and
    /// [`on_tool_complete`](AgentObserver::on_tool_complete).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancellation flag is set
    /// between tool invocations.
    pub(super) async fn dispatch_tools(
        &self,
        tool_calls: &[ToolCallInfo],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, AgentError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for tc in tool_calls {
            if self.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            let result = self.dispatch_tool_with_recovery(tc, turn_idx).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// Dispatch a single tool call, using reflector + recovery on errors.
    ///
    /// If the tool call succeeds, returns the result immediately. If it
    /// fails, calls [`Reflector::analyze()`] and [`RecoveryStrategy::decide()`]
    /// to determine the next action:
    ///
    /// - [`Retry`](RecoveryAction::Retry) — re-dispatch the tool after the
    ///   specified delay, up to the recovery strategy's retry limit.
    /// - [`Skip`](RecoveryAction::Skip) — produce a soft error result and
    ///   continue to the next tool.
    /// - [`Fail`](RecoveryAction::Fail) — produce a soft error result (the
    ///   model sees the failure and can decide how to respond).
    /// - [`AskUser`](RecoveryAction::AskUser) — treated as `Skip` (interactive
    ///   recovery not yet supported in `BareLoop`).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancellation signal fires
    /// during tool execution or between retry attempts.
    async fn dispatch_tool_with_recovery(
        &self,
        tc: &ToolCallInfo,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, AgentError> {
        let tool_context = self.build_tool_context();
        let mut attempt: u32 = 0;

        loop {
            if self.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            if let Some(blocked) = self.check_pre_tool_use_hooks(tc, turn_idx) {
                return Ok(blocked);
            }

            self.notify_tool_call(&tc.name, &tc.input.to_string());
            self.emit_tool_start(&tc.name, &tc.input.to_string());

            if let Some(blocked) = self.pre_detection(tc, turn_idx) {
                return Ok(blocked);
            }

            let start = Instant::now();
            let tool_result = self
                .dispatch_tool(tc, &tool_context, start, turn_idx)
                .await?;

            self.post_detection(tc, &tool_result);
            self.notify_post_tool_use_hooks(tc, &tool_result, turn_idx);
            self.record_tool_health(tc.name.as_str(), &tool_result);

            if !tool_result.is_error {
                return Ok(tool_result);
            }

            match self
                .recovery_wait_or_return(tc, &tool_result, attempt)
                .await
            {
                Ok(next_attempt) => attempt = next_attempt,
                Err(returned_result) => return Ok(returned_result),
            }
        }
    }

    /// Check for a loop pattern before executing the tool.
    ///
    /// Hashes the tool input, records the call with the detection manager,
    /// and returns a soft-error result if the same (tool, input-hash) pair
    /// has exceeded the loop threshold. Returns `None` when dispatch should
    /// proceed normally.
    fn pre_detection(&self, tc: &ToolCallInfo, turn_idx: usize) -> Option<ToolDispatchResult> {
        let input_hash = Self::hash_tool_input(&tc.input);
        let pattern = self
            .managers
            .detection
            .record_tool_call(&tc.name, input_hash);
        self.handle_detected_pattern(&pattern, turn_idx)
            .map(|_result| ToolDispatchResult {
                tool_call_id: tc.id.clone(),
                output: ToolContent::Text("loop detected: aborting tool dispatch".into()),
                is_error: true,
                duration: Duration::ZERO,
                resolved_tool_name: tc.name.clone(),
            })
    }

    /// Record the tool result with the detection manager (post-execution).
    ///
    /// Hashes the tool's text output and feeds the (tool, input-hash,
    /// result-hash) triple back to the detection manager. This lets the
    /// detector distinguish "same input, same output" (stuck) from
    /// "same input, different output" (progress).
    fn post_detection(&self, tc: &ToolCallInfo, tool_result: &ToolDispatchResult) {
        let input_hash = Self::hash_tool_input(&tc.input);
        let result_hash = match &tool_result.output {
            ToolContent::Text(t) => loop_detector::hash_result(t),
            ToolContent::Multipart(_) => None,
        };
        if let Some(rh) = result_hash {
            let _ = self.managers.detection.record_tool_call_with_result(
                &tc.name,
                input_hash,
                Some(rh),
            );
        }
    }

    /// Execute a single tool call through the pipeline or registry.
    ///
    /// Tries the middleware pipeline first, then a direct registry lookup,
    /// then produces a not-found error result. Handles cancellation during
    /// execution and emits observer/sink events.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancel signal fires
    /// during tool execution.
    async fn dispatch_tool(
        &self,
        tc: &ToolCallInfo,
        tool_context: &ToolContext,
        start: Instant,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, AgentError> {
        if let Some(ref pipeline) = self.pipeline {
            return self
                .dispatch_via_pipeline(pipeline, tc, tool_context, start, turn_idx)
                .await;
        }

        let tool_result = if let Some(tool) = self.tools.get(&tc.name) {
            let cancel = Arc::clone(&self.cancelled);
            let call_result = tokio::select! {
                r = tool.call(tc.input.clone(), tool_context) => r,
                () = cancel.notified() => {
                    let dur = start.elapsed();
                    self.notify_tool_complete(
                        &tc.name,
                        &tc.input.to_string(),
                        "",
                        dur,
                        false,
                        Some("cancelled"),
                    );
                    self.emit_tool_complete(&tc.name, "", true, dur);
                    return Err(AgentError::Cancelled);
                }
            };
            match call_result {
                Ok(result) => {
                    let duration = start.elapsed();
                    let output_text = result.text_content();
                    let success = !result.is_error;
                    self.notify_tool_complete(
                        &tc.name,
                        &tc.input.to_string(),
                        &output_text,
                        duration,
                        success,
                        None,
                    );
                    self.emit_tool_complete(&tc.name, &output_text, !success, duration);
                    ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: result.payload,
                        is_error: result.is_error,
                        duration,
                        resolved_tool_name: tc.name.clone(),
                    }
                }
                Err(e) => {
                    let duration = start.elapsed();
                    let error_msg = e.to_string();
                    self.notify_tool_complete(
                        &tc.name,
                        &tc.input.to_string(),
                        &error_msg,
                        duration,
                        false,
                        Some(&error_msg),
                    );
                    self.emit_tool_complete(&tc.name, &error_msg, true, duration);
                    ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(error_msg),
                        is_error: true,
                        duration,
                        resolved_tool_name: tc.name.clone(),
                    }
                }
            }
        } else {
            self.tool_not_found(tc)
        };

        Ok(tool_result)
    }

    /// Build a soft-error result for a tool that isn't in the registry.
    ///
    /// Notifies observers and emits a sink event with the error message
    /// that lists available tool names to help the model recover.
    fn tool_not_found(&self, tc: &ToolCallInfo) -> ToolDispatchResult {
        let available: Vec<String> = self.tools.tool_names();
        let available_refs: Vec<&str> = available.iter().map(String::as_str).collect();
        let error = AgentError::tool_not_found(&tc.name, &available_refs);
        let error_msg = error.to_string();
        self.notify_tool_complete(
            &tc.name,
            &tc.input.to_string(),
            &error_msg,
            Duration::ZERO,
            false,
            Some(&error_msg),
        );
        self.emit_tool_complete(&tc.name, &error_msg, true, Duration::ZERO);
        ToolDispatchResult {
            tool_call_id: tc.id.clone(),
            output: ToolContent::Text(error_msg),
            is_error: true,
            duration: Duration::ZERO,
            resolved_tool_name: tc.name.clone(),
        }
    }

    /// Decide whether to retry a failed tool or return the error result.
    ///
    /// Consults the reflector and recovery strategy. On `Retry`, sleeps for
    /// the prescribed delay (cancellation-aware) and returns the updated
    /// attempt count via `Ok`. On all other recovery actions, returns the
    /// original error result via `Err` (which ends the retry loop).
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolDispatchResult)` when the recovery strategy decides
    /// not to retry — the caller should return this as a soft error.
    async fn recovery_wait_or_return(
        &self,
        tc: &ToolCallInfo,
        tool_result: &ToolDispatchResult,
        attempt: u32,
    ) -> Result<u32, ToolDispatchResult> {
        let recovery_action = self.recover_tool_error(tc, tool_result, attempt).await;
        match recovery_action {
            RecoveryAction::Retry { delay } => {
                let next_attempt = attempt.saturating_add(1);
                if next_attempt >= Self::MAX_RECOVERY_ATTEMPTS {
                    return Err(tool_result.clone());
                }
                tokio::select! {
                    () = tokio::time::sleep(delay) => {},
                    () = self.cancelled.notified() => {
                        // Return a placeholder — the caller checks
                        // cancellation at the top of the retry loop.
                        return Err(tool_result.clone());
                    }
                }
                Ok(next_attempt)
            }
            RecoveryAction::Skip(_) | RecoveryAction::AskUser(_) | RecoveryAction::Fail(_) => {
                Err(tool_result.clone())
            }
        }
    }

    /// Check pre-tool-use hooks and return a blocked result if any hook
    /// blocks or asks.
    ///
    /// Returns `Some(ToolDispatchResult)` with an error result if a hook
    /// blocked the call, or `None` if the call should proceed.
    ///
    /// *Requires `hooks` feature; returns `None` otherwise.*
    fn check_pre_tool_use_hooks(
        &self,
        tc: &ToolCallInfo,
        turn_idx: usize,
    ) -> Option<ToolDispatchResult> {
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let ctx = PreToolUseContext {
                tool_name: tc.name.clone(),
                input: tc.input.clone(),
                session_id: self.config.session_id,
                turn_number: turn_idx,
            };
            match executor.check_pre_tool_use(&ctx) {
                HookAction::Allow => None,
                HookAction::Block { reason } => {
                    self.emit_tool_complete(&tc.name, &reason, true, Duration::ZERO);
                    Some(ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(reason),
                        is_error: true,
                        duration: Duration::ZERO,
                        resolved_tool_name: tc.name.clone(),
                    })
                }
                HookAction::Ask { message } => {
                    // In Headless mode (the default) the executor already
                    // downgrades Ask → Block. If we reach this arm the
                    // executor is Interactive, but BareLoop has no UI to
                    // show a prompt, so we still treat it as Block.
                    self.emit_tool_complete(&tc.name, &message, true, Duration::ZERO);
                    Some(ToolDispatchResult {
                        tool_call_id: tc.id.clone(),
                        output: ToolContent::Text(message),
                        is_error: true,
                        duration: Duration::ZERO,
                        resolved_tool_name: tc.name.clone(),
                    })
                }
            }
        } else {
            None
        }
        #[cfg(not(feature = "hooks"))]
        {
            let _ = (tc, turn_idx);
            None
        }
    }

    /// Notify post-tool-use hooks with the execution result.
    ///
    /// *Requires `hooks` feature; no-op otherwise.*
    fn notify_post_tool_use_hooks(
        &self,
        tc: &ToolCallInfo,
        tool_result: &ToolDispatchResult,
        turn_idx: usize,
    ) {
        #[cfg(feature = "hooks")]
        if let Some(ref executor) = self.hook_executor {
            let output_text = tool_result.output.to_string();
            let ctx = PostToolUseContext {
                tool_name: tc.name.clone(),
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
        {
            let _ = (tc, tool_result, turn_idx);
        }
    }

    /// Record tool health (success or failure) in the health registry.
    ///
    /// *Requires `tool_health` feature; no-op otherwise.*
    fn record_tool_health(&self, tool_name: &str, tool_result: &ToolDispatchResult) {
        #[cfg(feature = "tool_health")]
        if let Some(ref health) = self.health_registry {
            if tool_result.is_error {
                health.record_failure(tool_name, tool_result.duration);
            } else {
                health.record_success(tool_name, tool_result.duration);
            }
        }
        #[cfg(not(feature = "tool_health"))]
        {
            let _ = (tool_name, tool_result);
        }
    }

    /// Dispatch a tool call through the middleware pipeline.
    ///
    /// Builds a [`ToolDispatchContext`] from the tool call info, delegates
    /// to the pipeline's middleware chain, and converts the
    /// [`ToolDispatchResult`] back to a [`ToolDispatchResult`] with proper
    /// event emission. Handles cancellation via `tokio::select!`.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Cancelled`] if the cancel signal fires
    /// during pipeline dispatch.
    async fn dispatch_via_pipeline(
        &self,
        pipeline: &ToolPipeline,
        tc: &ToolCallInfo,
        tool_context: &ToolContext,
        start: Instant,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, AgentError> {
        let ctx = ToolDispatchContext {
            tool_name: tc.name.clone(),
            input: tc.input.clone(),
            call_id: tc.id.clone(),
            turn_number: turn_idx,
            cancel: Arc::clone(&self.cancelled),
            permission: PermissionCheck::Allow,
            tool_context: tool_context.clone(),
        };
        let cancel = Arc::clone(&self.cancelled);
        let dispatch_result = tokio::select! {
            r = pipeline.invoke(ctx) => r,
            () = cancel.notified() => {
                let dur = start.elapsed();
                self.notify_tool_complete(
                    &tc.name,
                    &tc.input.to_string(),
                    "",
                    dur,
                    false,
                    Some("cancelled"),
                );
                self.emit_tool_complete(&tc.name, "", true, dur);
                return Err(AgentError::Cancelled);
            }
        };
        let output_text = match &dispatch_result.output {
            ToolContent::Text(t) => t.clone(),
            ToolContent::Multipart(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ToolContentPart::Text { text } => Some(text.as_str()),
                    ToolContentPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };
        self.notify_tool_complete(
            &tc.name,
            &tc.input.to_string(),
            &output_text,
            dispatch_result.duration,
            !dispatch_result.is_error,
            if dispatch_result.is_error {
                Some(&output_text)
            } else {
                None
            },
        );
        self.emit_tool_complete(
            &tc.name,
            &output_text,
            dispatch_result.is_error,
            dispatch_result.duration,
        );
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
    /// Calls [`Reflector::analyze()`] and then [`RecoveryStrategy::decide()`].
    /// If the reflector itself fails, logs the error and returns
    /// [`RecoveryAction::Fail`] (conservative default).
    async fn recover_tool_error(
        &self,
        tc: &ToolCallInfo,
        result: &ToolDispatchResult,
        attempt: u32,
    ) -> RecoveryAction {
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
            .analyze(&error_msg, &tc.name, &tc.input, &context)
            .await
        else {
            // Reflector failed — conservatively fail.
            return RecoveryAction::Fail(error_msg);
        };

        self.recovery
            .decide(&analysis, attempt, Self::MAX_RECOVERY_ATTEMPTS)
            .await
    }
}
