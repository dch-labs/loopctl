//! Tool dispatch — execute tool calls requested by the model.
//!
//! Sequential and parallel tool execution with reflection and recovery on
//! errors, hook interception, health recording, and middleware pipeline
//! support.

/// Truncate a string to `max_len` chars, appending `…` when truncated.
fn truncate_to(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        return s.to_string();
    }
    let mut cut = s.char_indices().take(max_len).last().map_or(0, |(i, _)| i);
    cut = cut.saturating_add(s[cut..].chars().next().map_or(0, char::len_utf8));
    format!("{}…", &s[..cut])
}

#[cfg(feature = "hooks")]
use super::HookAction;
use super::{
    ApiClient, Arc, BareLoop, Duration, Instant, LoopError, PermissionCheck, RecoveryAction,
    ReflectionContext, ToolCall, ToolContent, ToolContext, ToolDispatchContext, ToolDispatchResult,
    ToolPipeline,
};
#[cfg(feature = "hooks")]
use super::{PostToolUseContext, PreToolUseContext};
use crate::capabilities::Detectable;
#[cfg(feature = "tool_health")]
use crate::capabilities::HealthTrackable;
#[cfg(feature = "hooks")]
use crate::capabilities::Hookable;
use crate::capabilities::PipelineAware;
use crate::detection::loop_detector::{self, Operation};

use crate::observer::{ToolPostContext, ToolPreContext};
use crate::reflection::{Correction, CorrectionResult};
use crate::tool::ToolRegistry;

use futures::FutureExt;
use std::collections::HashSet;
use std::panic::AssertUnwindSafe;

/// What the recovery loop decided to do after a tool error.
///
/// Produced by [`recovery_wait_or_return`](BareLoop::recovery_wait_or_return)
/// after it consults the configured
/// [`RecoveryStrategy`](crate::reflection::RecoveryStrategy) (and, on a `Retry`
/// decision, races the backoff sleep against the cancel signal). Matched
/// exhaustively at the retry-loop call site in
/// [`execute_tool_call`](BareLoop::execute_tool_call), where each variant maps
/// to one control-flow branch: continue the loop, return a soft result, or
/// propagate cancellation.
///
/// This is a driver-internal control-flow type — it never escapes the dispatch
/// module. None of the variants is an "error"; the loop's `Result` is reserved
/// for actual failures. A `Soft` result carries `is_error: true` *inside* its
/// [`ToolDispatchResult`], but at this layer it's a value being returned, not
/// an error being raised.
enum RecoveryDecision {
    /// Retry the call after sleeping for the strategy's backoff delay.
    ///
    /// Produced only when the strategy returned
    /// [`RecoveryAction::Retry`](crate::reflection::RecoveryAction::Retry) *and*
    /// the backoff sleep completed without cancellation. The driver applies
    /// any carried correction to the [`ToolCall`] before re-entering the
    /// dispatch loop, then bumps its attempt counter to `next_attempt`. If
    /// that counter crosses `MAX_RECOVERY_ATTEMPTS`, the loop gives up and
    /// surfaces [`LoopError::ToolRecoveryExhausted`] rather than retrying
    /// again — so receiving this variant does not guarantee another attempt
    /// will actually run.
    Retry {
        /// The next attempt number, 1-indexed within the retry sequence.
        ///
        /// Pre-incremented by [`recovery_wait_or_return`] so the call site
        /// just assigns `attempt = next_attempt` — there is exactly one
        /// `saturating_add(1)` and it lives in the producer, not the
        /// consumer. The original call is attempt `0`; the first retry is
        /// `1`; the ceiling check `attempt > MAX_RECOVERY_ATTEMPTS` (default
        /// 5) fires at `6`.
        ///
        /// [`recovery_wait_or_return`]: BareLoop::recovery_wait_or_return
        next_attempt: u32,

        /// An optional correction produced by the
        /// [`Reflector`](crate::reflection::Reflector) to apply before the
        /// retry.
        ///
        /// `None` when the strategy chose to retry without consulting the
        /// reflector, or when the reflector had no suggestion. When `Some`,
        /// the driver routes it through
        /// [`ToolCall::apply_correction`] before the next attempt, which may
        /// rewrite the input JSON or swap the tool name. A correction that
        /// fails validation is logged and dropped — the retry still runs
        /// with the uncorrected call.
        ///
        /// [`ToolCall::apply_correction`]: crate::engine::ToolCall::apply_correction
        correction: Option<Correction>,
    },

    /// Stop retrying and return this [`ToolDispatchResult`] to the model as a
    /// soft error.
    ///
    /// Produced when the strategy chose anything other than `Retry` —
    /// specifically [`Skip`](crate::reflection::RecoveryAction::Skip),
    /// [`AskUser`](crate::reflection::RecoveryAction::AskUser), or
    /// [`Fail`](crate::reflection::RecoveryAction::Fail). The carried result
    /// is the *original* failing `ToolDispatchResult` (with `is_error: true`),
    /// cloned verbatim — no new execution happens, the model simply sees the
    /// failure and gets to decide how to recover on its next turn.
    ///
    /// Soft errors do not terminate the run. They flow back through the
    /// normal tool-result path, the model responds, and the loop continues —
    /// the model may retry the tool itself, try a different tool, or give up
    /// and produce a final answer acknowledging the failure.
    Soft(ToolDispatchResult),

    /// The cancel signal fired during the recovery backoff sleep.
    ///
    /// Produced only on the `Retry` path, when
    /// [`CancelSignal::notified`](crate::cancel::CancelSignal::notified) wins
    /// the `select!` against `tokio::time::sleep(delay)`. Distinct from a
    /// cancellation observed during tool *execution* (which surfaces as
    /// [`LoopError::Cancelled`] directly from the dispatch `select!`): this
    /// variant specifically means the user cancelled in the gap between
    /// deciding-to-retry and starting-the-retry. The call site maps it to
    /// `Err(LoopError::Cancelled)`, which the driver's error path records as
    /// [`MachineOutcome::Cancelled`](crate::engine::core::MachineOutcome::Cancelled)
    /// — a clean stop, not a failure.
    Cancelled,
}

/// Analysis of a batch of tool calls, classifying each as parallelizable and
/// grouping independent calls into waves.
///
/// Pure: takes the calls + a `&ToolRegistry` (for concurrency-safety queries)
/// and produces a [`DispatchPlan`]. No I/O, no async, no side-effects — fully
/// testable in isolation.
struct ToolDependencyGraph {
    /// One node per input call, in input order.
    ///
    /// Each entry records the call's index, its concurrency-safety verdict,
    /// and its declared resource key. The plan is derived by walking this vec
    /// in order.
    nodes: Vec<GraphNode>,
}

/// Per-call analysis node produced by [`ToolDependencyGraph::from_calls`].
///
/// Records whether the call may run in parallel and, if so, which resource it
/// declares (for conflict detection).
struct GraphNode {
    /// Index into the original `&[ToolCall]` slice.
    ///
    /// Preserved so the plan can refer back to the original call ordering
    /// even after waves partition the indices into concurrent groups.
    idx: usize,

    /// Whether the call's tool reported
    /// [`is_safe_for_concurrent_execution`](crate::tool::Tool::is_safe_for_concurrent_execution)
    /// as `true` for this input.
    ///
    /// When `false`, the call is serialized into its own singleton wave
    /// regardless of its resource key — it never runs alongside any other
    /// call.
    parallelizable: bool,

    /// The resource key the tool declared for this input, if any.
    ///
    /// Two parallelizable calls with equal `Some(_)` keys conflict and are
    /// placed in separate waves. `None` when the tool returned `None` or the
    /// call is not parallelizable.
    resource_key: Option<String>,
}

/// A run plan produced by [`ToolDependencyGraph::plan`].
///
/// Each wave is a set of original call indices that may run concurrently.
/// Waves execute sequentially; within a wave, calls are independent (no shared
/// resource keys, all parallelizable).
struct DispatchPlan {
    /// The set of waves, each holding original call indices that may run
    /// concurrently.
    ///
    /// `waves[w]` is a vec of indices into the original `&[ToolCall]` slice.
    /// Waves execute sequentially (wave 0 first); within a wave, calls have
    /// disjoint resource keys and are all parallelizable.
    waves: Vec<Vec<usize>>,
}

impl ToolDependencyGraph {
    /// Build the graph from a batch of calls and a registry.
    ///
    /// Looks up each tool in the registry and records its
    /// [`is_safe_for_concurrent_execution`](crate::tool::Tool::is_safe_for_concurrent_execution)
    /// verdict and [`resource_key`](crate::tool::Tool::resource_key). Calls whose
    /// tool is not in the registry are marked non-parallelizable (they produce
    /// a not-found result in their wave, in order).
    fn from_calls(calls: &[ToolCall], registry: &ToolRegistry) -> Self {
        let nodes = calls
            .iter()
            .enumerate()
            .map(|(idx, call)| {
                let (parallelizable, resource_key) = match registry.get(&call.tool) {
                    Some(tool) => {
                        let safe = tool.is_safe_for_concurrent_execution(&call.input);
                        let key = if safe {
                            tool.resource_key(&call.input)
                        } else {
                            None
                        };
                        (safe, key)
                    }
                    None => (false, None),
                };
                GraphNode {
                    idx,
                    parallelizable,
                    resource_key,
                }
            })
            .collect();
        Self { nodes }
    }

    /// Partition into waves.
    ///
    /// Rule: two calls conflict if both are parallelizable AND their
    /// `resource_key`s are equal `Some(_)`, or either is non-parallelizable.
    /// Non-parallelizable calls each get their own singleton wave. Greedy:
    /// assign each parallelizable call to the earliest wave whose existing
    /// members do not share its resource key.
    fn plan(&self) -> DispatchPlan {
        let mut waves: Vec<Vec<usize>> = Vec::new();
        let mut wave_keys: Vec<HashSet<String>> = Vec::new();
        let mut wave_open: Vec<bool> = Vec::new();

        for node in &self.nodes {
            if !node.parallelizable {
                waves.push(vec![node.idx]);
                wave_keys.push(HashSet::new());
                wave_open.push(false);
                continue;
            }
            // Find the earliest open wave that doesn't already hold this key.
            // A call with no resource key (None) can join any open wave.
            let target_wave_idx = match &node.resource_key {
                None => wave_open.iter().position(|open| *open),
                Some(key) => wave_open
                    .iter()
                    .enumerate()
                    .find(|(wave_idx, open)| {
                        **open
                            && !wave_keys
                                .get(*wave_idx)
                                .is_some_and(|keys| keys.contains(key))
                    })
                    .map(|(wave_idx, _)| wave_idx),
            };
            let wave_idx = if let Some(wave_idx) = target_wave_idx {
                wave_idx
            } else {
                waves.push(Vec::new());
                wave_keys.push(HashSet::new());
                wave_open.push(true);
                waves.len().saturating_sub(1)
            };
            if let Some(wave) = waves.get_mut(wave_idx) {
                wave.push(node.idx);
            }
            if let Some(key) = &node.resource_key
                && let Some(keys) = wave_keys.get_mut(wave_idx)
            {
                keys.insert(key.clone());
            }
        }
        DispatchPlan { waves }
    }
}

impl<C: ApiClient> BareLoop<C> {
    /// Build a tool context for tool invocations.
    ///
    /// Creates a [`ToolContext`] pre-populated with the current session ID.
    pub(super) fn build_tool_context(&self) -> ToolContext {
        ToolContext {
            session_id: self.session.id,
            ..ToolContext::default()
        }
    }

    /// Execute a batch of tool calls and return results in input order.
    ///
    /// Routes to the sequential or parallel path based on
    /// [`parallel_tool_dispatch`](crate::engine::RunConfig::parallel_tool_dispatch).
    /// Both paths honour mid-batch cancellation: if the cancel signal fires
    /// between (or during) calls, the method returns
    /// [`LoopError::Cancelled`] promptly.
    ///
    /// A tool that is not found in the registry produces a soft error result
    /// (not a hard [`LoopError`]), allowing the model to recover.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if the cancel signal fires mid-batch,
    /// [`LoopError::LoopDetected`] if the detection manager signals a hard
    /// stop, or any hard error propagated from an individual tool dispatch.
    pub(super) async fn dispatch_tools(
        &self,
        tool_calls: &[ToolCall],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, LoopError> {
        match self.dispatch_mode().mode {
            crate::config::ParallelMode::Parallel => {
                self.dispatch_tools_parallel(tool_calls, turn_idx).await
            }
            crate::config::ParallelMode::Sequential => {
                self.dispatch_tools_sequential(tool_calls, turn_idx).await
            }
        }
    }

    /// Dispatch tool calls one at a time.
    ///
    /// Each [`ToolCall`] runs to completion via
    /// [`execute_tool_call`](Self::execute_tool_call) before the next
    /// begins. The cancel signal is checked between calls so a Ctrl-C
    /// mid-batch aborts the remaining calls rather than running them all.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if the cancel signal fires between
    /// calls, or any hard error from
    /// [`execute_tool_call`](Self::execute_tool_call).
    async fn dispatch_tools_sequential(
        &self,
        tool_calls: &[ToolCall],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, LoopError> {
        let mut results = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            if self.is_cancelled() {
                return Err(LoopError::Cancelled);
            }
            let result = self.execute_tool_call(call.clone(), turn_idx).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// Dispatch independent tool calls concurrently.
    ///
    /// Builds a [`DispatchPlan`] from the registry's
    /// [`is_safe_for_concurrent_execution`](crate::tool::Tool::is_safe_for_concurrent_execution)
    /// and [`resource_key`](crate::tool::Tool::resource_key) metadata, then
    /// runs each wave of calls concurrently under a semaphore capped at
    /// [`max_concurrency`](crate::config::ParallelDispatchConfig::max_concurrency).
    /// Each call runs through [`execute_tool_call`](Self::execute_tool_call)
    /// — the same end-to-end path as sequential — so observers,
    /// detection, hooks, and health all fire identically regardless of
    /// dispatch mode.
    ///
    /// Falls back to the sequential path when there are fewer than 2 calls.
    ///
    /// # Errors
    ///
    /// [`LoopError::Cancelled`] if the cancel signal fires during dispatch.
    /// [`LoopError::LoopDetected`] on a hard stop from detection. Any hard
    /// error from an individual [`execute_tool_call`](Self::execute_tool_call).
    ///
    /// # Hard-error semantics
    ///
    /// A hard error from any call in a wave (cancellation, loop detection, or
    /// recovery exhaustion) aborts the entire batch immediately. Results from
    /// sibling calls in the same wave — including ones that already resolved
    /// successfully — are **discarded**; only the error propagates. This
    /// matches the sequential path's "first hard error wins" semantics: no
    /// partial results are returned. Soft errors (`is_error: true`) do *not*
    /// trigger this — they are collected alongside successful results so the
    /// model can see all of a turn's outcomes. Pinned by
    /// `parallel_hard_error_discards_sibling_results`.
    async fn dispatch_tools_parallel(
        &self,
        tool_calls: &[ToolCall],
        turn_idx: usize,
    ) -> Result<Vec<ToolDispatchResult>, LoopError> {
        if tool_calls.len() < 2 {
            return self.dispatch_tools_sequential(tool_calls, turn_idx).await;
        }

        let plan = ToolDependencyGraph::from_calls(tool_calls, &self.tools).plan();
        let max_concurrency = self
            .dispatch_mode()
            .max_concurrency
            .clamp(1, tool_calls.len());
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
        let mut results: Vec<Option<ToolDispatchResult>> =
            (0..tool_calls.len()).map(|_| None).collect();

        for wave in &plan.waves {
            if self.is_cancelled() {
                return Err(LoopError::Cancelled);
            }

            let mut tasks = Vec::with_capacity(wave.len());
            for &idx in wave {
                let tc = tool_calls.get(idx).cloned();
                let sem = Arc::clone(&semaphore);
                let turn = turn_idx;
                tasks.push(async move {
                    let _permit = sem.acquire_owned().await.ok()?;
                    Some(self.execute_tool_call(tc?, turn).await)
                });
            }

            let outcomes = futures::future::join_all(tasks).await;
            for (outcome, &idx) in outcomes.into_iter().zip(wave) {
                match outcome {
                    None => return Err(LoopError::Cancelled),
                    Some(Ok(result)) => {
                        if let Some(slot) = results.get_mut(idx) {
                            *slot = Some(result);
                        }
                    }
                    Some(Err(e)) => return Err(e),
                }
            }
        }

        Ok(results
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                // Defensive: the planner invariant guarantees every slot is
                // filled by the wave loop above. If that invariant ever breaks,
                // this produces a soft error rather than a panic.
                r.unwrap_or_else(|| Self::missing_result(tool_calls.get(idx)))
            })
            .collect())
    }

    /// Build the defensive soft-error result for a parallel-dispatch slot that
    /// the wave loop did not fill.
    ///
    /// Reachable only if the planner invariant ("every slot is filled") breaks.
    /// Produces a soft error (`is_error: true`, zero duration) so the model can
    /// react, rather than panicking.
    fn missing_result(tc: Option<&ToolCall>) -> ToolDispatchResult {
        ToolDispatchResult {
            tool_call_id: tc.map(|c| c.id.clone()).unwrap_or_default(),
            output: ToolContent::Text("dispatch produced no result".to_string()),
            is_error: true,
            duration: Duration::ZERO,
            resolved_tool_name: tc.map(|c| c.tool.clone()).unwrap_or_default(),
            display_hint: None,
        }
    }

    /// Build a [`ToolDispatchResult`] for a dispatched call.
    ///
    /// The `tool_call_id` and `resolved_tool_name` come from the call; the
    /// caller supplies the elapsed `duration`, the `output`, the `is_error`
    /// flag, and any `display_hint`. Used by the three dispatch-outcome arms
    /// (success, error, panic) so they share one construction shape.
    fn result_for_call(
        tc: &ToolCall,
        duration: Duration,
        output: ToolContent,
        is_error: bool,
        display_hint: Option<crate::tool::DisplayHint>,
    ) -> ToolDispatchResult {
        ToolDispatchResult {
            tool_call_id: tc.id.clone(),
            output,
            is_error,
            duration,
            resolved_tool_name: tc.tool.clone(),
            display_hint,
        }
    }

    /// Execute a single tool call end-to-end.
    ///
    /// The single function that owns the full lifecycle of one tool call:
    /// PRE (observer pre-notification, pre-hooks, pre-detection) → dispatch
    /// → POST (post-detection, observer post-notification, post-hooks,
    /// health recording) → recovery decision. On failure, consults
    /// [`recovery_wait_or_return`](Self::recovery_wait_or_return) and loops
    /// — re-firing PRE and POST on every retry attempt — until the tool
    /// succeeds, the strategy gives up (returns a soft error), or the user
    /// cancels.
    ///
    /// Safe to run concurrently: every side-effect target (`LoopObserver`,
    /// `DetectionManager`, `HookExecutor`, `HealthRegistry`) is `Send +
    /// Sync`. Sequential and parallel dispatch both call this function, so
    /// there is exactly one definition of what "execute a tool call"
    /// means — no divergence in side-effect granularity, observer event
    /// counts, or recovery behaviour between modes.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] when the recovery strategy aborts
    /// on cancellation, [`LoopError::LoopDetected`] on a hard detection
    /// stop, or any hard error from
    /// [`dispatch_tool`](Self::dispatch_tool).
    async fn execute_tool_call(
        &self,
        mut tc: ToolCall,
        turn_idx: usize,
    ) -> Result<ToolDispatchResult, LoopError> {
        let tool_context = self.build_tool_context();
        let original_tool = tc.tool.clone();
        let mut attempt: u32 = 0;

        loop {
            self.notify_tool_pre(turn_idx, &tc);

            #[cfg(feature = "hooks")]
            if let Some(blocked) = self.check_pre_tool_use_hooks(&tc, turn_idx) {
                self.notify_tool_post(turn_idx, &tc, &blocked);
                return Ok(blocked);
            }

            if let Err(e) = self.pre_detection(turn_idx) {
                let blocked = Self::result_for_call(
                    &tc,
                    Duration::ZERO,
                    ToolContent::Text(format!("dispatch refused before execution: {e}")),
                    true,
                    None,
                );
                self.notify_tool_post(turn_idx, &tc, &blocked);
                return Err(e);
            }

            let start = Instant::now();
            let tool_result = tokio::select! {
                biased;
                () = self.cancelled.notified() => return Err(LoopError::Cancelled),
                r = self.dispatch_tool(&tc, &tool_context, start, turn_idx) => r,
            };
            self.post_detection(&tc, &tool_result);
            self.notify_tool_post(turn_idx, &tc, &tool_result);
            #[cfg(feature = "hooks")]
            self.notify_post_tool_use_hooks(&tc, &tool_result, turn_idx);
            #[cfg(feature = "tool_health")]
            self.record_tool_health(&Self::health_key(&tc, &tool_result), &tool_result);
            self.record_tool_memory(&tc, &tool_result).await;

            if !tool_result.is_error {
                return Ok(tool_result);
            }

            match self
                .recovery_wait_or_return(&tc, &tool_result, attempt)
                .await
            {
                RecoveryDecision::Retry {
                    next_attempt,
                    correction,
                } => {
                    attempt = next_attempt;
                    if attempt > Self::MAX_RECOVERY_ATTEMPTS {
                        return Err(LoopError::ToolRecoveryExhausted {
                            tool: original_tool,
                            attempts: attempt,
                        });
                    }
                    Self::apply_correction_if_present(&mut tc, correction);
                }
                RecoveryDecision::Soft(returned_result) => return Ok(returned_result),
                RecoveryDecision::Cancelled => return Err(LoopError::Cancelled),
            }
        }
    }

    /// Notify observers that a tool call is about to be dispatched.
    ///
    /// Fires [`on_tool_pre`](crate::observer::LoopObserver::on_tool_pre) with
    /// the turn index, tool name, and tool-call ID. Called once per call before
    /// any hook checks, detection, or execution — observers always see this
    /// first, regardless of whether the call is later blocked or retried.
    fn notify_tool_pre(&self, turn_idx: usize, tc: &ToolCall) {
        self.managers.observers().on_tool_pre(&ToolPreContext {
            turn: turn_idx,
            tool: tc.tool.clone(),
            tool_call_id: tc.id.clone(),
        });
    }

    /// Notify observers that a tool call has completed (or been blocked).
    ///
    /// Fires [`on_tool_post`](crate::observer::LoopObserver::on_tool_post) with
    /// a result hash, error flag, and timing. Called for every outcome —
    /// successful execution, hook block, detection block, or soft error — so
    /// that every `on_tool_pre` has a matching `on_tool_post`, regardless of
    /// the path taken. Observers can pair the two by `tool_call_id`.
    fn notify_tool_post(&self, turn_idx: usize, tc: &ToolCall, result: &ToolDispatchResult) {
        self.managers.observers().on_tool_post(&ToolPostContext {
            turn: turn_idx,
            tool_call_id: result.tool_call_id.clone(),
            tool: tc.tool.clone(),
            result_hash: loop_detector::hash_result(&result.output.to_string()),
            is_error: result.is_error,
            duration: result.duration,
            display_hint: result.display_hint.clone(),
        });
    }

    /// Apply a correction produced by the recovery strategy, if any.
    ///
    /// The correction modifies the tool call's input before the next retry
    /// attempt. If the correction cannot be applied, logs a warning and
    /// proceeds with the original input.
    fn apply_correction_if_present(tc: &mut ToolCall, correction: Option<Correction>) {
        let Some(correction) = correction else { return };
        if let CorrectionResult::Failed(msg) = tc.apply_correction(&correction) {
            tracing::warn!(
                tool = %tc.tool,
                error = %msg,
                "correction failed to produce a usable retry"
            );
        }
    }

    /// Query the loop detector before dispatch, without recording.
    ///
    /// A pure read of the sliding window: the single record point per
    /// invocation is [`post_detection`](Self::post_detection), so a
    /// re-derived or re-driven step cannot double-count. A pattern at or
    /// over the stop threshold aborts the call; anything less severe
    /// (no pattern, a warning-band pattern) lets dispatch proceed.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::LoopDetected`] if the loop window holds a
    /// pattern at or over the stop threshold. Convergence patterns are
    /// decided on the response path, not here.
    fn pre_detection(&self, turn_idx: usize) -> Result<(), LoopError> {
        let pattern = self.managers.detection().check_loop_pattern();

        self.managers.notify_detected_pattern(&pattern, turn_idx);
        match self.decide_detected_pattern(&pattern) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Record the completed invocation for loop detection.
    ///
    /// The single write point per invocation: hashes the result — multipart
    /// by its rendered text — so the detector distinguishes "same input,
    /// same output" (stuck) from "same input, different output" (progress).
    /// The primary parameter is the configured signature's extraction, with
    /// a canonical-JSON rendering of the input as the fallback when the
    /// signature yields nothing (see [`detection_primary_param`]).
    fn post_detection(&self, tc: &ToolCall, tool_result: &ToolDispatchResult) {
        let result_hash = match &tool_result.output {
            ToolContent::Text(t) => loop_detector::hash_result(t),
            ToolContent::Multipart(_) => {
                loop_detector::hash_result(&tool_result.output.to_string())
            }
        };
        let operation = Operation {
            tool: tc.tool.clone(),
            primary_param: self.detection_primary_param(&tc.tool, &tc.input),
            result_hash,
        };
        self.managers.detection().record_operation(operation);
    }

    /// The primary parameter one dispatch is recorded under.
    ///
    /// The configured [`ToolSignature`]'s extraction, falling back to a
    /// canonical-JSON rendering of the whole input when the extraction
    /// yields the empty string — the default
    /// [`NoOpToolSignature`](crate::detection::NoOpToolSignature)'s blind
    /// spot. Without the fallback, two calls of one tool with different
    /// inputs but byte-identical outputs (the same file read from two
    /// paths, `git status` twice on an unchanged tree) collapse into one
    /// operation and count as repetition; with it, only *identical
    /// inputs* repeating is flagged. A signature that does extract keeps
    /// full control of the key.
    fn detection_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
        let extracted = self
            .managers
            .detection()
            .signature()
            .extract_primary_param(tool, input);
        if extracted.is_empty() {
            serde_json::to_string(input).unwrap_or_default()
        } else {
            extracted
        }
    }

    /// Execute a single tool call.
    ///
    /// Tries the middleware pipeline first (if configured), then falls back
    /// to a direct registry lookup. Tool panics are caught and converted to
    /// error results. A tool not in the registry produces a soft error.
    ///
    /// Always returns a [`ToolDispatchResult`] — hard stops (cancellation,
    /// loop detection, recovery exhaustion) are handled by the caller
    /// [`execute_tool_call`](Self::execute_tool_call), which wraps this call.
    async fn dispatch_tool(
        &self,
        tc: &ToolCall,
        tool_context: &ToolContext,
        start: Instant,
        turn_idx: usize,
    ) -> ToolDispatchResult {
        if let Some(pipeline) = self.managers.pipeline() {
            return self
                .dispatch_via_pipeline(pipeline, tc, tool_context, turn_idx)
                .await;
        }

        if let Some(tool) = self.tools.get(&tc.tool) {
            let call_result = AssertUnwindSafe(tool.call(tc.input.clone(), tool_context))
                .catch_unwind()
                .await;
            match call_result {
                Ok(Ok(result)) => Self::result_for_call(
                    tc,
                    start.elapsed(),
                    result.payload,
                    result.is_error,
                    result.display_hint,
                ),
                Ok(Err(e)) => Self::result_for_call(
                    tc,
                    start.elapsed(),
                    ToolContent::Text(e.to_string()),
                    true,
                    None,
                ),
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
                    Self::result_for_call(
                        tc,
                        duration,
                        ToolContent::Text(format!("Tool '{}' panicked: {msg}", tc.tool)),
                        true,
                        None,
                    )
                }
            }
        } else {
            self.tool_not_found(tc)
        }
    }

    /// Build a soft-error result for a tool whose name is not in the registry.
    ///
    /// The result carries `is_error: true` and a human-readable message that
    /// lists the available tool names, helping the model correct itself on the
    /// next turn. The duration is zero (no execution occurred). This is a soft
    /// error — the batch continues and the model sees the result.
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
            display_hint: None,
        }
    }

    /// Decide whether to retry a failed tool or return the error as a soft result.
    ///
    /// Consults the [`Reflector`](crate::reflection::Reflector) and
    /// [`RecoveryStrategy`](crate::reflection::RecoveryStrategy). On
    /// [`Retry`](RecoveryAction::Retry), sleeps for the prescribed delay and
    /// returns [`RecoveryDecision::Retry`] with the updated attempt count and
    /// optional [`Correction`]. On all other actions (`Skip`, `Fail`,
    /// `AskUser`), returns [`RecoveryDecision::Soft`] with the original error
    /// result. The backoff sleep is cancel-aware: if the cancel signal fires
    /// during the wait, returns [`RecoveryDecision::Cancelled`].
    async fn recovery_wait_or_return(
        &self,
        tc: &ToolCall,
        tool_result: &ToolDispatchResult,
        attempt: u32,
    ) -> RecoveryDecision {
        let (recovery_action, correction) = self.recover_tool_error(tc, tool_result, attempt).await;
        match recovery_action {
            RecoveryAction::Retry { delay } => {
                let next_attempt = attempt.saturating_add(1);
                tokio::select! {
                    biased;
                    () = self.cancelled.notified() => RecoveryDecision::Cancelled,
                    () = tokio::time::sleep(delay) => RecoveryDecision::Retry { next_attempt, correction },
                }
            }
            RecoveryAction::Skip(_) | RecoveryAction::AskUser(_) | RecoveryAction::Fail(_) => {
                RecoveryDecision::Soft(tool_result.clone())
            }
        }
    }

    /// Check pre-tool-use hooks before a call executes.
    ///
    /// Consults the session's [`HookExecutor`](crate::hooks::HookExecutor) (if
    /// configured) with the tool name, input, and turn number. Returns:
    ///
    /// - `None` when no hook is configured or all hooks return `Allow` — the
    ///   call should proceed to execution.
    /// - `Some(result)` when a hook returns `Block` or `Ask` — the result is a
    ///   soft error (`is_error: true`, zero duration) carrying the hook's
    ///   reason/message. The call is **not** executed; the caller returns this
    ///   result to the model.
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
            session_id: self.session.id,
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
                display_hint: None,
            }),
            HookAction::Ask { message } => Some(ToolDispatchResult {
                tool_call_id: tc.id.clone(),
                output: ToolContent::Text(message),
                is_error: true,
                duration: Duration::ZERO,
                resolved_tool_name: tc.tool.clone(),
                display_hint: None,
            }),
        }
    }

    /// Notify post-tool-use hooks with the execution result.
    ///
    /// If a [`HookExecutor`](crate::hooks::HookExecutor) is configured, builds
    /// a [`PostToolUseContext`] from the tool call and its result (output text,
    /// error flag, duration) and passes it to `notify_post_tool_use`. This lets
    /// hooks observe or react to completed executions — e.g. logging, auditing,
    /// or triggering side-effects based on the result.
    ///
    /// Called after every successful or errored dispatch, but not for calls
    /// blocked in PRE (hooks already saw those).
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
            session_id: self.session.id,
            turn_number: turn_idx,
        };
        executor.notify_post_tool_use(&ctx);
    }

    /// Record tool execution health (success or failure) in the health registry.
    ///
    /// If a [`ToolHealthRegistry`](crate::tool::health::ToolHealthRegistry) is
    /// configured, records the outcome and duration so the health system can
    /// track per-tool success rates, latency, and degraded-state transitions.
    /// Failures call `record_failure`; successes call `record_success`. Safe to
    /// call concurrently — the registry uses interior mutability.
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

    /// The tool name health statistics are keyed on for one dispatch.
    ///
    /// The executed (resolved) tool name when a routing middleware
    /// renamed it, else the requested name — the same key space
    /// [`VerifyMiddleware`](crate::middleware::verify::VerifyMiddleware)
    /// matches on, so health, verification, and telemetry all describe
    /// one dispatch by one name.
    #[cfg(feature = "tool_health")]
    fn health_key(tc: &ToolCall, tool_result: &ToolDispatchResult) -> String {
        if tool_result.resolved_tool_name.is_empty() {
            tc.tool.clone()
        } else {
            tool_result.resolved_tool_name.clone()
        }
    }

    /// Store a successful tool-execution trajectory into the memory backend.
    ///
    /// Called after each tool dispatch that did not error. Guards on
    /// [`RememberCapable`] — when no memory store is configured this is a
    /// no-op. Builds a [`MemoryEntry`](crate::memory::MemoryEntry) tagged
    /// [`Trajectory`](crate::memory::MemoryCategory::Trajectory) carrying the
    /// tool name, input, and result, then stores it. Errors are logged and
    /// swallowed — a memory-store failure must never crash the turn.
    async fn record_tool_memory(&self, tc: &ToolCall, tool_result: &ToolDispatchResult) {
        const MAX_FIELD_LEN: usize = 500;
        let Some(memory) = self.managers.memory() else {
            return;
        };
        if tool_result.is_error {
            return;
        }
        let input = truncate_to(&tc.input.to_string(), MAX_FIELD_LEN);
        let result = truncate_to(&tool_result.output.to_string(), MAX_FIELD_LEN);
        let entry = crate::memory::MemoryEntry::new(
            crate::memory::MemoryCategory::Trajectory,
            format!("tool={}; input={input}; result={result}", tc.tool),
        );
        if let Err(e) = memory.store(entry).await {
            tracing::warn!(error = %e, tool = %tc.tool, "memory store failed");
        }
    }

    /// Dispatch a tool call through the middleware pipeline.
    ///
    /// Builds a [`ToolDispatchContext`] and delegates to the pipeline's
    /// middleware chain (timeout, permissions, output limits, etc.).
    ///
    /// Always returns a [`ToolDispatchResult`] — soft errors are carried as
    /// `is_error: true`. Observer notifications are handled by the caller
    /// ([`execute_tool_call`](Self::execute_tool_call)).
    async fn dispatch_via_pipeline(
        &self,
        pipeline: &ToolPipeline,
        tc: &ToolCall,
        tool_context: &ToolContext,
        turn_idx: usize,
    ) -> ToolDispatchResult {
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
        ToolDispatchResult {
            tool_call_id: if dispatch_result.tool_call_id.is_empty() {
                tc.id.clone()
            } else {
                dispatch_result.tool_call_id
            },
            output: dispatch_result.output,
            is_error: dispatch_result.is_error,
            duration: dispatch_result.duration,
            resolved_tool_name: dispatch_result.resolved_tool_name,
            display_hint: dispatch_result.display_hint,
        }
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

        // Resolve the schema under the name a routing middleware may have
        // redirected the call to, falling back to the requested name when
        // the resolved name is empty or unknown to the registry.
        let resolved_tool = if result.resolved_tool_name.is_empty() {
            &tc.tool
        } else {
            &result.resolved_tool_name
        };
        let tool_schema = self
            .tools
            .get(resolved_tool)
            .or_else(|| self.tools.get(&tc.tool))
            .map(crate::tool::Tool::schema);
        let Ok(analysis) = self
            .reflector
            .analyze(
                &error_msg,
                &tc.tool,
                &tc.input,
                tool_schema.as_ref(),
                &context,
            )
            .await
        else {
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

#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::api::error::ApiError;
    use crate::config::SessionConfig;
    use crate::engine::core::ToolCall;
    use crate::engine::{Run, RunConfig};
    use crate::message::ToolContent;
    use crate::reflection::{FailureAnalysis, FailureSeverity};
    use crate::tool::{
        Tool, ToolContext, ToolError, ToolOutput, ToolSchema, registry::ToolRegistry,
    };
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::Instant;

    use std::sync::Mutex;

    use super::*;

    /// Reflector that marks every failure recoverable, used by the recovery tests.
    struct AlwaysRecoverable;
    impl crate::reflection::Reflector for AlwaysRecoverable {
        fn analyze(
            &self,
            error: &str,
            tool_name: &str,
            _tool_input: &Value,
            _tool_schema: Option<&crate::tool::ToolSchema>,
            _context: &crate::reflection::ReflectionContext,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<FailureAnalysis, crate::reflection::ReflectionError>>
                    + Send
                    + '_,
            >,
        > {
            let error = error.to_string();
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                Ok(FailureAnalysis {
                    is_recoverable: true,
                    root_cause: error,
                    severity: FailureSeverity::Medium,
                    correction: None,
                    context: format!("tool: {tool_name}"),
                })
            })
        }
    }

    #[test]
    fn truncate_to_short_string_unchanged() {
        assert_eq!(truncate_to("hello", 10), "hello");
    }

    #[test]
    fn truncate_to_exact_length_unchanged() {
        assert_eq!(truncate_to("hello", 5), "hello");
    }

    #[test]
    fn truncate_to_longer_string_appends_ellipsis() {
        assert_eq!(truncate_to("hello world", 5), "hello…");
    }

    #[test]
    fn truncate_to_multibyte_chars_counts_characters_not_bytes() {
        assert_eq!(truncate_to("héllo", 3), "hél…");
        assert_eq!(truncate_to("日本語テスト", 3), "日本語…");
    }

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
            crate::error::recover_guard(self.model_name.lock()).clone()
        }
        fn set_model(&self, model: &str) -> bool {
            if model.trim().is_empty() {
                return false;
            }
            *crate::error::recover_guard(self.model_name.lock()) = model.to_string();
            true
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
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
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            Box::pin(async { Err(ApiError::http("not implemented")) })
        }
    }

    struct PanicTool;

    impl Tool for PanicTool {
        fn name(&self) -> &'static str {
            "panic_tool"
        }
        fn description(&self) -> &'static str {
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
        let config = SessionConfig::default();
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

        let dispatch_result = bare.dispatch_tool(&tc, &tool_context, start, 0).await;

        assert!(
            dispatch_result.is_error,
            "panic should be caught as a soft error"
        );
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

        let dispatch_result = bare.dispatch_tool(&tc, &tool_context, start, 0).await;

        assert!(!dispatch_result.is_error);
        match &dispatch_result.output {
            ToolContent::Text(text) => assert_eq!(text, "ok"),
            ToolContent::Multipart(_) => panic!("expected Text"),
        }
    }

    fn make_call(id: &str, tool: &str, input: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            tool: tool.into(),
            input,
        }
    }

    fn safe_tool(name: &str) -> crate::tool::FnTool {
        crate::tool::FnTool::new(
            name.into(),
            name.into(),
            Value::Object(serde_json::Map::new()),
            echo_fn,
        )
        .concurrency_safe()
    }

    fn unsafe_tool(name: &str) -> crate::tool::FnTool {
        crate::tool::FnTool::new(
            name.into(),
            name.into(),
            Value::Object(serde_json::Map::new()),
            echo_fn,
        )
    }

    fn path_key(input: &Value) -> Option<String> {
        input.get("path").and_then(|p| p.as_str()).map(String::from)
    }

    fn safe_tool_with_key(name: &str) -> crate::tool::FnTool {
        safe_tool(name).with_resource_key(path_key)
    }

    #[test]
    fn graph_all_parallelizable_one_wave() {
        let mut registry = ToolRegistry::new();
        registry.register(safe_tool("a"));
        registry.register(safe_tool("b"));
        let calls = vec![
            make_call("1", "a", Value::Null),
            make_call("2", "b", Value::Null),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        assert_eq!(plan.waves.len(), 1, "all safe, no resources → 1 wave");
        assert_eq!(plan.waves[0].len(), 2);
    }

    #[test]
    fn graph_non_parallelizable_singleton_wave() {
        let mut registry = ToolRegistry::new();
        registry.register(unsafe_tool("unsafe"));
        registry.register(safe_tool("safe"));
        let calls = vec![
            make_call("1", "unsafe", Value::Null),
            make_call("2", "safe", Value::Null),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        // Non-parallelizable call gets its own singleton wave.
        assert!(
            plan.waves.iter().any(|w| w == &[0]),
            "unsafe call should be alone in a wave"
        );
        assert!(
            plan.waves.iter().any(|w| w == &[1]),
            "safe call should be in its own wave"
        );
    }

    #[test]
    fn graph_same_resource_separate_waves() {
        let mut registry = ToolRegistry::new();
        registry.register(safe_tool_with_key("file"));
        let calls = vec![
            make_call("1", "file", serde_json::json!({"path": "/a"})),
            make_call("2", "file", serde_json::json!({"path": "/a"})),
            make_call("3", "file", serde_json::json!({"path": "/b"})),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        // Calls 0 and 1 share "/a" → separate waves. Call 2 ("/b") can share
        // a wave with either.
        let wave_of = |idx: usize| {
            plan.waves
                .iter()
                .position(|w| w.contains(&idx))
                .expect("call must be in a wave")
        };
        assert_ne!(wave_of(0), wave_of(1), "same-resource calls must differ");
        // Call 2 shares a wave with one of them (disjoint key).
        assert!(
            wave_of(2) == wave_of(0) || wave_of(2) == wave_of(1),
            "disjoint-key call should share a wave"
        );
    }

    #[test]
    fn graph_resource_chain() {
        let mut registry = ToolRegistry::new();
        registry.register(safe_tool_with_key("file"));
        // A("/x"), B("/x"), C("/y") → waves [A,C] then [B] (or [C,A] then [B]).
        let calls = vec![
            make_call("a", "file", serde_json::json!({"path": "/x"})),
            make_call("b", "file", serde_json::json!({"path": "/x"})),
            make_call("c", "file", serde_json::json!({"path": "/y"})),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        let wave_of = |idx: usize| {
            plan.waves
                .iter()
                .position(|w| w.contains(&idx))
                .expect("call must be in a wave")
        };
        assert_ne!(wave_of(0), wave_of(1), "A and B share /x");
        assert_eq!(
            wave_of(0),
            wave_of(2),
            "A and C share a wave (disjoint keys)"
        );
    }

    #[test]
    fn graph_unknown_tool_non_parallelizable() {
        let mut registry = ToolRegistry::new();
        registry.register(safe_tool("a"));
        // "ghost" is not in the registry.
        let calls = vec![
            make_call("1", "a", Value::Null),
            make_call("2", "ghost", Value::Null),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        // Unknown tool → non-parallelizable → singleton wave. No panic.
        assert!(
            plan.waves.iter().any(|w| w == &[1]),
            "unknown tool should be a singleton wave"
        );
    }

    #[test]
    fn graph_empty_input() {
        let registry = ToolRegistry::new();
        let graph = ToolDependencyGraph::from_calls(&[], &registry);
        let plan = graph.plan();
        assert!(plan.waves.is_empty(), "no calls → no waves");
    }

    #[test]
    fn graph_order_preservation() {
        let mut registry = ToolRegistry::new();
        registry.register(safe_tool("a"));
        registry.register(safe_tool("b"));
        registry.register(safe_tool("c"));
        registry.register(safe_tool("d"));
        let calls = vec![
            make_call("1", "a", Value::Null),
            make_call("2", "b", Value::Null),
            make_call("3", "c", Value::Null),
            make_call("4", "d", Value::Null),
        ];
        let graph = ToolDependencyGraph::from_calls(&calls, &registry);
        let plan = graph.plan();
        // All safe, no resources → one wave with indices in order.
        assert_eq!(plan.waves.len(), 1);
        assert_eq!(plan.waves[0], vec![0, 1, 2, 3]);
    }

    fn make_parallel_loop(tools: ToolRegistry) -> BareLoop<MockClient> {
        let client = Arc::new(MockClient::new("test"));
        let run_config = RunConfig {
            parallel_tool_dispatch: crate::config::ParallelDispatchConfig {
                mode: crate::config::ParallelMode::Parallel,
                ..Default::default()
            },
            ..RunConfig::default()
        };
        let mut bare = BareLoop::new(client, tools, SessionConfig::default());
        bare.session.runs.push(Run::new("", &run_config));
        bare
    }

    #[tokio::test]
    async fn parallel_latency_independent_calls_overlap() {
        use crate::testing::MockTool;
        let mut registry = ToolRegistry::new();
        registry.register(
            MockTool::new("slow", "slow")
                .with_concurrency_safe(true)
                .with_delay(std::time::Duration::from_millis(100)),
        );
        let bare = make_parallel_loop(registry);

        let calls = vec![
            make_call("1", "slow", Value::Null),
            make_call("2", "slow", Value::Null),
            make_call("3", "slow", Value::Null),
        ];
        let start = Instant::now();
        let results = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect("should succeed");
        let elapsed = start.elapsed();

        // Sequential would be ~300ms; parallel should be ~100ms. Assert <290ms
        // (proves overlap with CI scheduling headroom) and all 3 results present.
        assert!(
            elapsed < std::time::Duration::from_millis(290),
            "parallel should overlap 3×100ms calls; elapsed {elapsed:?}"
        );
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn parallel_result_order_matches_input() {
        use crate::testing::MockTool;
        let mut registry = ToolRegistry::new();
        registry.register(
            MockTool::new("a", "tool a")
                .with_concurrency_safe(true)
                .with_result("result_a"),
        );
        registry.register(
            MockTool::new("b", "tool b")
                .with_concurrency_safe(true)
                .with_result("result_b")
                .with_delay(std::time::Duration::from_millis(20)),
        );
        registry.register(
            MockTool::new("c", "tool c")
                .with_concurrency_safe(true)
                .with_result("result_c")
                .with_delay(std::time::Duration::from_millis(40)),
        );
        let bare = make_parallel_loop(registry);

        // Call order: c (slowest), a (fastest), b. Results must come back in
        // input order [c, a, b], not completion order [a, b, c].
        let calls = vec![
            make_call("1", "c", Value::Null),
            make_call("2", "a", Value::Null),
            make_call("3", "b", Value::Null),
        ];
        let results = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect("should succeed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].tool_call_id, "1");
        assert_eq!(results[1].tool_call_id, "2");
        assert_eq!(results[2].tool_call_id, "3");
    }

    #[tokio::test]
    async fn parallel_soft_errors_collected() {
        use crate::testing::MockTool;
        let mut registry = ToolRegistry::new();
        registry.register(
            MockTool::new("ok", "succeeds")
                .with_concurrency_safe(true)
                .with_result("fine"),
        );
        registry.register(
            MockTool::new("bad", "fails")
                .with_concurrency_safe(true)
                .with_error(),
        );
        let bare = make_parallel_loop(registry);

        let calls = vec![
            make_call("1", "ok", Value::Null),
            make_call("2", "bad", Value::Null),
            make_call("3", "ok", Value::Null),
        ];
        let results = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect("soft errors should not fail the batch");
        assert_eq!(results.len(), 3);
        assert!(!results[0].is_error, "call 1 should succeed");
        assert!(results[1].is_error, "call 2 should be a soft error");
        assert!(!results[2].is_error, "call 3 should succeed");
    }

    #[tokio::test]
    async fn parallel_hard_error_discards_sibling_results() {
        use crate::testing::MockTool;
        let mut registry = ToolRegistry::new();
        registry.register(
            MockTool::new("fast", "completes immediately")
                .with_concurrency_safe(true)
                .with_result("done"),
        );
        registry.register(
            MockTool::new("slow", "blocks until cancelled")
                .with_concurrency_safe(true)
                .with_delay(std::time::Duration::from_secs(10)),
        );

        let run_config = RunConfig {
            parallel_tool_dispatch: crate::config::ParallelDispatchConfig {
                mode: crate::config::ParallelMode::Parallel,
                max_concurrency: 1,
            },
            ..RunConfig::default()
        };
        let mut bare = BareLoop::new(
            Arc::new(MockClient::new("test")),
            registry,
            SessionConfig::default(),
        );
        bare.session.runs.push(Run::new("", &run_config));

        let cancel_signal = bare.cancel_signal();
        let calls = vec![
            make_call("1", "fast", Value::Null),
            make_call("2", "slow", Value::Null),
        ];

        // Fire cancel shortly after call #1 completes — call #2 (slow, 10s)
        // will observe it in its `select!` and return Err(Cancelled).
        let sig = Arc::clone(&cancel_signal);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            sig.cancel();
        });

        let err = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect_err("hard error should abort the batch");
        assert!(
            matches!(err, LoopError::Cancelled),
            "expected Cancelled, got {err:?}"
        );
    }

    #[tokio::test]
    async fn parallel_sequential_fallback() {
        use crate::testing::MockTool;
        let config = SessionConfig::default();
        // Default dispatch mode is Sequential — no change needed.
        let mut registry = ToolRegistry::new();
        registry.register(
            MockTool::new("a", "a")
                .with_concurrency_safe(true)
                .with_result("ok_a"),
        );
        let client = Arc::new(MockClient::new("test"));
        let bare = BareLoop::new(client, registry, config);

        let calls = vec![
            make_call("1", "a", Value::Null),
            make_call("2", "a", Value::Null),
        ];
        let results = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect("should succeed");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id, "1");
        assert_eq!(results[1].tool_call_id, "2");
    }

    #[tokio::test]
    async fn recovery_backoff_cancelled_promptly() {
        use crate::reflection::{FailureAnalysis, RecoveryAction, RecoveryStrategy};

        struct SlowRetry;
        impl RecoveryStrategy for SlowRetry {
            fn decide(
                &self,
                _analysis: &FailureAnalysis,
                _attempt: u32,
                _max_attempts: u32,
            ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
                Box::pin(async {
                    RecoveryAction::Retry {
                        delay: std::time::Duration::from_secs(10),
                    }
                })
            }
        }

        let error_tool = crate::tool::FnTool::new(
            "error_tool".into(),
            "Always errors".into(),
            Value::Object(serde_json::Map::new()),
            |_, _| Box::pin(async { Err(ToolError::Execution("boom".to_string())) }),
        );
        let mut registry = ToolRegistry::new();
        registry.register(error_tool);

        let mut bare = make_loop(registry);
        bare.set_reflector(Arc::new(AlwaysRecoverable));
        bare.set_recovery_strategy(Arc::new(SlowRetry));

        let cancelled = Arc::clone(&bare.cancelled);

        let calls = vec![make_call("1", "error_tool", Value::Null)];
        let call_handle = tokio::spawn(async move { bare.dispatch_tools(&calls, 0).await });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        cancelled.cancel();

        let start = Instant::now();
        let result = call_handle.await.expect("task should complete");
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(LoopError::Cancelled)),
            "expected Cancelled, got {result:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "cancel should interrupt the 10s backoff promptly; elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn parallel_retried_call_fires_side_effects_per_attempt() {
        // Pins the documented contract (config.rs `ParallelMode`): detection,
        // observer, hook, and health side-effects fire on EVERY retry attempt
        // in BOTH modes. A retried parallel call must therefore emit multiple
        // observer PRE+POST pairs, not one. Guards against a future change
        // re-introducing per-mode gating that the contract explicitly disclaims
        // (all side-effect targets are Send + Sync).
        use crate::observer::{LoopObserver, ToolPostContext, ToolPreContext};
        use crate::reflection::{FailureAnalysis, RecoveryAction, RecoveryStrategy};
        use std::sync::atomic::{AtomicU32, Ordering};

        // Retry the first two attempts, then give up with a soft error so the
        // call terminates. Each attempt is a full dispatch with PRE+POST.
        struct RetryTwice;
        impl RecoveryStrategy for RetryTwice {
            fn decide(
                &self,
                _analysis: &FailureAnalysis,
                attempt: u32,
                _max_attempts: u32,
            ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
                Box::pin(async move {
                    if attempt < 2 {
                        RecoveryAction::Retry {
                            delay: std::time::Duration::ZERO,
                        }
                    } else {
                        RecoveryAction::Skip("giving up".into())
                    }
                })
            }
        }

        struct CountingObserver {
            pre: Arc<AtomicU32>,
            post: Arc<AtomicU32>,
        }
        impl LoopObserver for CountingObserver {
            fn name(&self) -> &'static str {
                "counting"
            }
            fn on_tool_pre(&self, _ctx: &ToolPreContext) {
                self.pre.fetch_add(1, Ordering::Relaxed);
            }
            fn on_tool_post(&self, _ctx: &ToolPostContext) {
                self.post.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Tool that always errors; recovery drives 3 attempts (2 retries + 1 skip).
        let error_tool = crate::tool::FnTool::new(
            "error_tool".into(),
            "Always errors".into(),
            Value::Object(serde_json::Map::new()),
            |_, _| Box::pin(async { Err(ToolError::Execution("boom".to_string())) }),
        );

        let pre_count = Arc::new(AtomicU32::new(0));
        let post_count = Arc::new(AtomicU32::new(0));

        let mut registry = ToolRegistry::new();
        registry.register(error_tool);
        let mut bare = make_parallel_loop(registry);
        bare.set_reflector(Arc::new(AlwaysRecoverable));
        bare.set_recovery_strategy(Arc::new(RetryTwice));
        bare.register_observer(Arc::new(CountingObserver {
            pre: Arc::clone(&pre_count),
            post: Arc::clone(&post_count),
        }));

        let calls = vec![make_call("1", "error_tool", Value::Null)];
        let _ = bare
            .dispatch_tools(&calls, 0)
            .await
            .expect("dispatch should not hard-error");

        let pres = pre_count.load(Ordering::Relaxed);
        let posts = post_count.load(Ordering::Relaxed);
        assert_eq!(
            pres, 3,
            "RetryTwice does 2 retries + 1 final skip = 3 attempts = 3 PRE events; got {pres}"
        );
        assert_eq!(
            posts, 3,
            "matching 3 POST events for the 3 attempts; got {posts}"
        );
        assert_eq!(
            pres, posts,
            "every PRE must have a matching POST (pairing invariant)"
        );
    }

    #[tokio::test]
    async fn execute_tool_call_runs_recovery_on_failure() {
        use crate::reflection::{FailureAnalysis, RecoveryAction, RecoveryStrategy};
        use std::sync::atomic::{AtomicU32, Ordering};

        struct CountingRetry {
            calls: Arc<AtomicU32>,
        }
        impl RecoveryStrategy for CountingRetry {
            fn decide(
                &self,
                _analysis: &FailureAnalysis,
                _attempt: u32,
                _max_attempts: u32,
            ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                Box::pin(async { RecoveryAction::Skip("counted".into()) })
            }
        }

        let decide_calls = Arc::new(AtomicU32::new(0));
        let error_tool = crate::tool::FnTool::new(
            "error_tool".into(),
            "Always errors".into(),
            Value::Object(serde_json::Map::new()),
            |_, _| Box::pin(async { Err(ToolError::Execution("boom".to_string())) }),
        );

        let mut registry = ToolRegistry::new();
        registry.register(error_tool);
        let mut bare = make_loop(registry);
        bare.set_reflector(Arc::new(AlwaysRecoverable));
        bare.set_recovery_strategy(Arc::new(CountingRetry {
            calls: Arc::clone(&decide_calls),
        }));

        let tc = make_call("1", "error_tool", Value::Null);
        let _ = bare.execute_tool_call(tc, 0).await.ok();

        assert!(
            decide_calls.load(Ordering::Relaxed) > 0,
            "execute_tool_call must run recovery on failure"
        );
    }

    #[tokio::test]
    async fn recovery_ceiling_stops_retry_forever_strategy() {
        use crate::reflection::{FailureAnalysis, RecoveryAction, RecoveryStrategy};

        struct RetryForever;
        impl RecoveryStrategy for RetryForever {
            fn decide(
                &self,
                _analysis: &FailureAnalysis,
                _attempt: u32,
                _max_attempts: u32,
            ) -> Pin<Box<dyn Future<Output = RecoveryAction> + Send + '_>> {
                Box::pin(async {
                    RecoveryAction::Retry {
                        delay: std::time::Duration::ZERO,
                    }
                })
            }
        }

        let error_tool = crate::tool::FnTool::new(
            "error_tool".into(),
            "Always errors".into(),
            Value::Object(serde_json::Map::new()),
            |_, _| Box::pin(async { Err(ToolError::Execution("boom".to_string())) }),
        );

        let mut registry = ToolRegistry::new();
        registry.register(error_tool);
        let mut bare = make_loop(registry);
        bare.set_reflector(Arc::new(AlwaysRecoverable));
        bare.set_recovery_strategy(Arc::new(RetryForever));

        let tc = make_call("1", "error_tool", Value::Null);
        let err = bare
            .execute_tool_call(tc, 0)
            .await
            .expect_err("retry-forever must hit the ceiling, not loop");

        match err {
            LoopError::ToolRecoveryExhausted { tool, attempts } => {
                assert_eq!(tool, "error_tool");
                assert_eq!(
                    attempts, 6,
                    "5 retries after the original call = attempt 6 trips the > 5 ceiling"
                );
            }
            other => panic!("expected ToolRecoveryExhausted, got {other:?}"),
        }
    }
}
