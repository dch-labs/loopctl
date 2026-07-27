//! Hook executor — manages an ordered list of hooks and dispatches events.
//!
//! The executor has two modes:
//!
//! - **Check** (for pre-hooks) — returns [`HookAction`] or [`CompactResult`],
//!   short-circuits on first non-default result.
//! - **Notify** (for post-hooks) — fire-and-forget, all hooks run.
//!
//! # Pre-hook Execution Model
//!
//! For `on_pre_*` hooks, the executor iterates through hooks and returns
//! the **first non-Allow result**. Explicit `Allow` results are skipped so
//! that later safety-critical hooks are still evaluated. This ensures that
//! safety-critical hooks (registered first or last) take precedence.
//!
//! ```text
//! check_pre_tool_use(ctx)
//!   → hook_1.on_pre_tool_use(ctx) → Some(Allow) ← skip, continue
//!   → hook_2.on_pre_tool_use(ctx) → Some(Block{...}) ← short-circuit!
//!   → return Block{...}
//!   // hook_3 is never called
//! ```
//!
//! # Post-hook Execution Model
//!
//! For `on_post_*` and session hooks, the executor calls **all** hooks.
//! There's no short-circuit because post-hooks are notification-only.
//!
//! # Thread Safety
//!
//! [`HookExecutor`] is `Send + Sync` because hooks are `Arc<dyn Hook>`.
//! The executor itself has no mutable state after construction.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::hooks::Hook;
use crate::hooks::HookAction;
use crate::hooks::Interactivity;
use crate::hooks::context::{
    CompactResult, PostCompactContext, PostToolUseContext, PreCompactContext, PreToolUseContext,
    SessionEndContext, SessionStartContext,
};

/// Executes hooks in registration order with short-circuit semantics.
///
/// # Interactivity
///
/// The executor carries an [`Interactivity`] mode that controls how
/// [`HookAction::Ask`] is handled:
///
/// - [`Interactivity::Headless`] (the default) — `Ask` is automatically
///   downgraded to `Block`, because there is no user to interact with.
/// - [`Interactivity::Interactive`] — `Ask` passes through unchanged,
///   allowing the agent to present a prompt to the user.
pub struct HookExecutor {
    /// Registered hooks in the order they will be invoked.
    ///
    /// Stored as `Arc<dyn Hook>` so a single hook instance can be
    /// shared across executors cheaply, and so the executor itself is
    /// `Send + Sync`. The vector is append-only after construction and
    /// never reordered.
    hooks: Vec<Arc<dyn Hook>>,

    /// How [`HookAction::Ask`] results are handled.
    ///
    /// Set at construction (default
    /// [`Interactivity::Headless`]) and applied by
    /// [`apply_interactivity`](Self::apply_interactivity) to every
    /// pre-tool-use result. Headless downgrades `Ask` to `Block`;
    /// interactive passes it through unchanged.
    interactivity: Interactivity,
}

impl Default for HookExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HookExecutor {
    /// Create an executor in [`Interactivity::Headless`] mode with no hooks.
    ///
    /// Use [`with_hook`](Self::with_hook) to add hooks via builder pattern,
    /// or [`register`](Self::register) for mutable registration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            interactivity: Interactivity::Headless,
        }
    }

    /// Set the interactivity mode (builder style).
    ///
    /// Overrides the default [`Interactivity::Headless`] set by
    /// [`new`](Self::new). Switch to [`Interactivity::Interactive`] when
    /// a human is available to confirm [`HookAction::Ask`] prompts, so
    /// they pass through unchanged instead of being downgraded to
    /// [`HookAction::Block`].
    #[must_use]
    pub fn with_interactivity(mut self, interactivity: Interactivity) -> Self {
        self.interactivity = interactivity;
        self
    }

    /// Register a hook via the builder pattern.
    ///
    /// Appends `hook` to the end of the execution list, so hooks fire
    /// in registration order. Returns `self` for chaining — for example
    /// `HookExecutor::new().with_hook(a).with_hook(b)`. Use
    /// [`register`](Self::register) instead when you need to mutate an
    /// existing executor.
    #[must_use]
    pub fn with_hook(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Register a hook by mutating the executor in place.
    ///
    /// Appends `hook` to the end of the execution list, mirroring
    /// [`with_hook`](Self::with_hook) but taking `&mut self` instead of
    /// consuming the executor. Useful when hooks are registered
    /// conditionally after construction.
    pub fn register(&mut self, hook: Arc<dyn Hook>) {
        self.hooks.push(hook);
    }

    /// Downgrade [`HookAction::Ask`] to [`HookAction::Block`] in headless mode.
    ///
    /// In [`Interactivity::Headless`] mode, `Ask` is converted to `Block`
    /// with a reason that includes the original message. All other actions
    /// (including `Allow` and `Block`) pass through unchanged.
    fn apply_interactivity(&self, action: HookAction) -> HookAction {
        match (&self.interactivity, action) {
            (Interactivity::Headless, HookAction::Ask { message }) => HookAction::block(format!(
                "Hook requested confirmation ({message}) but the session is not interactive"
            )),
            (_, action) => action,
        }
    }

    /// Number of hooks currently registered.
    ///
    /// Returns `0` for a freshly constructed executor. Cheap (`O(1)`)
    /// since it reads the vector length directly; safe to call from
    /// hot paths.
    #[must_use]
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    /// Check pre-tool-use hooks.
    ///
    /// Returns the first non-Allow action (e.g., `Block`, `Ask`), continuing
    /// past explicit `Allow` results so that later safety-critical hooks are
    /// still evaluated. Returns [`HookAction::Allow`] if no hook produced a
    /// non-Allow action.
    ///
    /// In [`Interactivity::Headless`] mode, [`HookAction::Ask`] is
    /// automatically downgraded to [`HookAction::Block`].
    #[must_use]
    pub fn check_pre_tool_use(&self, ctx: &PreToolUseContext) -> HookAction {
        for hook in &self.hooks {
            if let Some(action) = hook.on_pre_tool_use(ctx) {
                match action {
                    HookAction::Allow => {}
                    other => return self.apply_interactivity(other),
                }
            }
        }
        HookAction::Allow
    }

    /// Async wrapper for [`check_pre_tool_use`](Self::check_pre_tool_use).
    ///
    /// Runs the synchronous check and wraps the result in a `Pin<Box<Future>>`
    /// for use in async contexts without spawning a separate task.
    #[must_use]
    pub fn check_pre_tool_use_async(
        &self,
        ctx: &PreToolUseContext,
    ) -> Pin<Box<dyn Future<Output = HookAction> + Send + '_>> {
        let action = self.check_pre_tool_use(ctx);
        Box::pin(async move { action })
    }

    /// Check pre-compact hooks, merging every hook's [`CompactResult`].
    ///
    /// Unlike [`check_pre_tool_use`](Self::check_pre_tool_use), there is
    /// no short-circuit on the first result — all hooks run, and their
    /// results are combined: if any hook returns
    /// [`abort: true`](CompactResult::abort), the merged result is that
    /// abort (the first one wins, returned immediately); otherwise the
    /// last hook's `new_instructions` overrides earlier ones, and every
    /// hook's `additional_context` entries accumulate in registration
    /// order.
    #[must_use]
    pub fn check_pre_compact(&self, ctx: &PreCompactContext) -> CompactResult {
        let mut result = CompactResult::allow();
        for hook in &self.hooks {
            if let Some(hook_result) = hook.on_pre_compact(ctx) {
                if hook_result.abort {
                    return hook_result;
                }
                if let Some(instr) = hook_result.new_instructions {
                    result.new_instructions = Some(instr);
                }
                result
                    .additional_context
                    .extend(hook_result.additional_context);
            }
        }
        result
    }

    /// Async wrapper for [`check_pre_compact`](Self::check_pre_compact).
    ///
    /// Runs the synchronous check and wraps the result in a `Pin<Box<Future>>`
    /// for use in async contexts without spawning a separate task.
    #[must_use]
    pub fn check_pre_compact_async(
        &self,
        ctx: &PreCompactContext,
    ) -> Pin<Box<dyn Future<Output = CompactResult> + Send + '_>> {
        let result = self.check_pre_compact(ctx);
        Box::pin(async move { result })
    }

    /// Notify every registered post-tool-use hook.
    ///
    /// Fires [`Hook::on_post_tool_use`] on each hook in registration
    /// order. There is no short-circuit and no return value —
    /// post-hooks are notification-only, so every hook always runs.
    /// Use this for logging, metrics, and side-effects like file
    /// tracking.
    pub fn notify_post_tool_use(&self, ctx: &PostToolUseContext) {
        for hook in &self.hooks {
            hook.on_post_tool_use(ctx);
        }
    }

    /// Async wrapper for [`notify_post_tool_use`](Self::notify_post_tool_use).
    ///
    /// Runs all post-tool-use hooks synchronously and wraps completion
    /// in a `Pin<Box<Future>>` for async compatibility.
    #[must_use]
    pub fn notify_post_tool_use_async(
        &self,
        ctx: &PostToolUseContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.notify_post_tool_use(ctx);
        Box::pin(async {})
    }

    /// Notify every registered post-compact hook.
    ///
    /// Fires [`Hook::on_post_compact`] on each hook in registration
    /// order with the compaction outcome (messages removed, tokens
    /// saved, duration). Notification-only — every hook always runs,
    /// regardless of any prior hook's behaviour. Use it for budget
    /// tracking and post-compaction logging.
    pub fn notify_post_compact(&self, ctx: &PostCompactContext) {
        for hook in &self.hooks {
            hook.on_post_compact(ctx);
        }
    }

    /// Notify every registered session-start hook.
    ///
    /// Fires [`Hook::on_session_start`] on each hook in registration
    /// order, once, at the beginning of a session. Notification-only —
    /// every hook always runs. Use it to initialize per-session state
    /// (open resources, reset counters, emit a start log line).
    pub fn notify_session_start(&self, ctx: &SessionStartContext) {
        for hook in &self.hooks {
            hook.on_session_start(ctx);
        }
    }

    /// Notify every registered session-end hook.
    ///
    /// Fires [`Hook::on_session_end`] on each hook in registration
    /// order, once, after the loop has terminated. Notification-only —
    /// every hook always runs. Use it to flush resources, finalize
    /// tracking, and emit a summary log line keyed off
    /// [`SessionEndContext::reason`].
    pub fn notify_session_end(&self, ctx: &SessionEndContext) {
        for hook in &self.hooks {
            hook.on_session_end(ctx);
        }
    }

    /// Async wrapper for [`notify_post_compact`](Self::notify_post_compact).
    ///
    /// Runs all post-compact hooks synchronously and wraps completion
    /// in a `Pin<Box<Future>>` for async compatibility.
    #[must_use]
    pub fn notify_post_compact_async(
        &self,
        ctx: &PostCompactContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.notify_post_compact(ctx);
        Box::pin(async {})
    }

    /// Async wrapper for [`notify_session_start`](Self::notify_session_start).
    ///
    /// Runs all session-start hooks synchronously and wraps completion
    /// in a `Pin<Box<Future>>` for async compatibility.
    #[must_use]
    pub fn notify_session_start_async(
        &self,
        ctx: &SessionStartContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.notify_session_start(ctx);
        Box::pin(async {})
    }

    /// Async wrapper for [`notify_session_end`](Self::notify_session_end).
    ///
    /// Runs all session-end hooks synchronously and wraps completion
    /// in a `Pin<Box<Future>>` for async compatibility.
    #[must_use]
    pub fn notify_session_end_async(
        &self,
        ctx: &SessionEndContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.notify_session_end(ctx);
        Box::pin(async {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::context::{CompactTrigger, SessionEndReason};
    use serde_json::json;

    struct AllowHook;
    impl Hook for AllowHook {
        fn name(&self) -> &'static str {
            "allow"
        }
        fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
            None
        }
    }

    struct BlockHook {
        reason: String,
    }
    impl Hook for BlockHook {
        fn name(&self) -> &'static str {
            "block"
        }
        fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
            Some(HookAction::block(&self.reason))
        }
    }

    struct PostRecorder {
        called: std::sync::atomic::AtomicBool,
    }
    impl PostRecorder {
        fn new() -> Self {
            Self {
                called: std::sync::atomic::AtomicBool::new(false),
            }
        }
        fn was_called(&self) -> bool {
            self.called.load(std::sync::atomic::Ordering::Relaxed)
        }
    }
    impl Hook for PostRecorder {
        fn name(&self) -> &'static str {
            "post_recorder"
        }
        fn on_post_tool_use(&self, _ctx: &PostToolUseContext) {
            self.called
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn dummy_pre_ctx() -> PreToolUseContext {
        PreToolUseContext {
            tool_name: "test_tool".to_string(),
            input: json!({}),
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        }
    }

    #[test]
    fn empty_executor_allows() {
        let executor = HookExecutor::new();
        let ctx = dummy_pre_ctx();
        assert!(matches!(
            executor.check_pre_tool_use(&ctx),
            HookAction::Allow
        ));
    }

    #[test]
    fn allow_hook_passes_through() {
        let executor = HookExecutor::new().with_hook(Arc::new(AllowHook));
        let ctx = dummy_pre_ctx();
        assert!(matches!(
            executor.check_pre_tool_use(&ctx),
            HookAction::Allow
        ));
    }

    #[test]
    fn block_hook_short_circuits() {
        let executor = HookExecutor::new().with_hook(Arc::new(BlockHook {
            reason: "blocked".to_string(),
        }));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        match action {
            HookAction::Block { reason } => assert_eq!(reason, "blocked"),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[test]
    fn first_block_wins() {
        let executor = HookExecutor::new()
            .with_hook(Arc::new(BlockHook {
                reason: "first".to_string(),
            }))
            .with_hook(Arc::new(BlockHook {
                reason: "second".to_string(),
            }));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        match action {
            HookAction::Block { reason } => assert_eq!(reason, "first"),
            other => panic!("expected first Block, got {other:?}"),
        }
    }

    #[test]
    fn post_hooks_all_run() {
        let recorder = Arc::new(PostRecorder::new());
        let executor = HookExecutor::new()
            .with_hook(recorder.clone())
            .with_hook(recorder.clone());
        let ctx = PostToolUseContext {
            tool_name: "test".to_string(),
            input: json!({}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        executor.notify_post_tool_use(&ctx);
        assert!(recorder.was_called());
    }

    #[test]
    fn hook_count_tracks_registrations() {
        let executor = HookExecutor::new();
        assert_eq!(executor.hook_count(), 0);
        let executor = executor.with_hook(Arc::new(AllowHook));
        assert_eq!(executor.hook_count(), 1);
    }

    #[test]
    fn check_pre_compact_merges_instructions() {
        struct InstructionHook;
        impl Hook for InstructionHook {
            fn name(&self) -> &'static str {
                "instruction"
            }
            fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
                Some(
                    CompactResult::allow()
                        .with_instructions("focus on tests")
                        .with_context("keep test suite"),
                )
            }
        }
        let executor = HookExecutor::new().with_hook(Arc::new(InstructionHook));
        let ctx = PreCompactContext {
            trigger: CompactTrigger::Auto,
            custom_instructions: None,
            message_count: 10,
            tokens_before: 5000,
            context_window: 200_000,
            session_id: uuid::Uuid::nil(),
        };
        let result = executor.check_pre_compact(&ctx);
        assert!(!result.abort);
        assert_eq!(result.new_instructions.as_deref(), Some("focus on tests"));
        assert_eq!(result.additional_context.len(), 1);
    }

    #[test]
    fn check_pre_compact_abort_takes_priority() {
        struct AbortHook;
        impl Hook for AbortHook {
            fn name(&self) -> &'static str {
                "abort"
            }
            fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
                Some(CompactResult::abort("too risky"))
            }
        }
        struct InstructionHook;
        impl Hook for InstructionHook {
            fn name(&self) -> &'static str {
                "instruction"
            }
            fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
                Some(CompactResult::allow().with_instructions("override"))
            }
        }
        let executor = HookExecutor::new()
            .with_hook(Arc::new(AbortHook))
            .with_hook(Arc::new(InstructionHook));
        let ctx = PreCompactContext {
            trigger: CompactTrigger::Auto,
            custom_instructions: None,
            message_count: 10,
            tokens_before: 5000,
            context_window: 200_000,
            session_id: uuid::Uuid::nil(),
        };
        let result = executor.check_pre_compact(&ctx);
        assert!(result.abort);
        assert_eq!(result.abort_reason.as_deref(), Some("too risky"));
        assert!(result.new_instructions.is_none());
    }

    #[test]
    fn session_start_end_notify_all() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CounterHook {
            starts: AtomicUsize,
            ends: AtomicUsize,
        }
        impl Hook for CounterHook {
            fn name(&self) -> &'static str {
                "counter"
            }
            fn on_session_start(&self, _ctx: &SessionStartContext) {
                self.starts.fetch_add(1, Ordering::Relaxed);
            }
            fn on_session_end(&self, _ctx: &SessionEndContext) {
                self.ends.fetch_add(1, Ordering::Relaxed);
            }
        }
        let counter = Arc::new(CounterHook {
            starts: AtomicUsize::new(0),
            ends: AtomicUsize::new(0),
        });
        let executor = HookExecutor::new()
            .with_hook(counter.clone())
            .with_hook(counter.clone());
        executor.notify_session_start(&SessionStartContext {
            session_id: uuid::Uuid::nil(),
            model: "test".to_string(),
            working_directory: "/tmp".to_string(),
        });
        assert_eq!(counter.starts.load(Ordering::Relaxed), 2);
        executor.notify_session_end(&SessionEndContext {
            session_id: uuid::Uuid::nil(),
            reason: SessionEndReason::Complete,
            total_turns: 5,
            total_tokens: 1000,
            duration_secs: 30,
        });
        assert_eq!(counter.ends.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn register_adds_hook() {
        let mut executor = HookExecutor::new();
        assert_eq!(executor.hook_count(), 0);
        executor.register(Arc::new(AllowHook));
        assert_eq!(executor.hook_count(), 1);
        executor.register(Arc::new(BlockHook {
            reason: "x".to_string(),
        }));
        assert_eq!(executor.hook_count(), 2);
    }

    #[test]
    fn notify_post_tool_use_empty_executor() {
        let executor = HookExecutor::new();
        let ctx = PostToolUseContext {
            tool_name: "test".to_string(),
            input: json!({}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        executor.notify_post_tool_use(&ctx);
    }

    #[test]
    fn notify_post_compact_runs_all_hooks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CompactRecorder {
            count: AtomicUsize,
        }
        impl Hook for CompactRecorder {
            fn name(&self) -> &'static str {
                "compact_recorder"
            }
            fn on_post_compact(&self, _ctx: &PostCompactContext) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
        }
        let recorder = Arc::new(CompactRecorder {
            count: AtomicUsize::new(0),
        });
        let executor = HookExecutor::new()
            .with_hook(recorder.clone())
            .with_hook(recorder.clone());
        let ctx = PostCompactContext {
            trigger: CompactTrigger::Auto,
            messages_compacted: 5,
            tokens_saved: 3000,
            tokens_after: 2000,
            duration_ms: 100,
            session_id: uuid::Uuid::nil(),
        };
        executor.notify_post_compact(&ctx);
        assert_eq!(recorder.count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn notify_session_start_empty_executor() {
        let executor = HookExecutor::new();
        executor.notify_session_start(&SessionStartContext {
            session_id: uuid::Uuid::nil(),
            model: "test".to_string(),
            working_directory: "/tmp".to_string(),
        });
    }

    #[test]
    fn notify_session_end_empty_executor() {
        let executor = HookExecutor::new();
        executor.notify_session_end(&SessionEndContext {
            session_id: uuid::Uuid::nil(),
            reason: SessionEndReason::Complete,
            total_turns: 0,
            total_tokens: 0,
            duration_secs: 0,
        });
    }

    #[test]
    fn check_pre_tool_use_headless_downgrades_ask_to_block() {
        struct AskHook;
        impl Hook for AskHook {
            fn name(&self) -> &'static str {
                "ask"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::ask("confirm?"))
            }
        }

        // Default (Headless) executor downgrades Ask → Block.
        let executor = HookExecutor::new().with_hook(Arc::new(AskHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        assert!(
            action.is_block(),
            "expected Block in Headless mode, got {action:?}"
        );
    }

    #[test]
    fn check_pre_tool_use_interactive_passes_ask_through() {
        struct AskHook;
        impl Hook for AskHook {
            fn name(&self) -> &'static str {
                "ask"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::ask("confirm?"))
            }
        }

        // Interactive executor passes Ask through unchanged.
        let executor = HookExecutor::new()
            .with_interactivity(Interactivity::Interactive)
            .with_hook(Arc::new(AskHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        match action {
            HookAction::Ask { message } => assert_eq!(message, "confirm?"),
            other => panic!("expected Ask, got {other:?}"),
        }
    }

    #[test]
    fn check_pre_tool_use_interactivity_builder() {
        struct AskHook;
        impl Hook for AskHook {
            fn name(&self) -> &'static str {
                "ask"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::ask("ok?"))
            }
        }

        // Builder style: start Headless, switch to Interactive.
        let executor = HookExecutor::new()
            .with_interactivity(Interactivity::Interactive)
            .with_hook(Arc::new(AskHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        assert!(
            action.is_ask(),
            "expected Ask in Interactive mode, got {action:?}"
        );
    }

    #[test]
    fn default_is_same_as_new() {
        let default_exec = HookExecutor::default();
        let new_exec = HookExecutor::new();
        assert_eq!(default_exec.hook_count(), 0);
        assert_eq!(new_exec.hook_count(), 0);
        assert_eq!(default_exec.hook_count(), new_exec.hook_count());
    }

    #[test]
    fn headless_block_passes_through_unchanged() {
        struct BlockOnlyHook;
        impl Hook for BlockOnlyHook {
            fn name(&self) -> &'static str {
                "block_only"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::block("forbidden"))
            }
        }

        // Headless mode does NOT alter Block actions.
        let executor = HookExecutor::new().with_hook(Arc::new(BlockOnlyHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        assert!(action.is_block());
        assert_eq!(action.block_reason(), Some("forbidden"));
    }

    #[test]
    fn headless_downgrade_preserves_original_message() {
        struct AskHook;
        impl Hook for AskHook {
            fn name(&self) -> &'static str {
                "ask"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::ask("please confirm deployment"))
            }
        }

        let executor = HookExecutor::new().with_hook(Arc::new(AskHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        let reason = action
            .block_reason()
            .expect("downgraded action should be a Block with a reason");
        assert!(
            reason.contains("please confirm deployment"),
            "Block reason should contain the original Ask message, got: {reason}"
        );
        assert!(
            reason.contains("not interactive"),
            "Block reason should explain the downgrade, got: {reason}"
        );
    }

    #[test]
    fn with_interactivity_constructor_sets_mode() {
        struct AskHook;
        impl Hook for AskHook {
            fn name(&self) -> &'static str {
                "ask"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::ask("ok?"))
            }
        }

        // with_interactivity(Headless) should downgrade.
        let headless = HookExecutor::new()
            .with_interactivity(Interactivity::Headless)
            .with_hook(Arc::new(AskHook));
        assert!(
            headless.check_pre_tool_use(&dummy_pre_ctx()).is_block(),
            "Headless via with_interactivity should downgrade Ask"
        );
    }

    #[test]
    fn interactive_block_passes_through_unchanged() {
        struct BlockOnlyHook;
        impl Hook for BlockOnlyHook {
            fn name(&self) -> &'static str {
                "block_only"
            }
            fn on_pre_tool_use(&self, _ctx: &PreToolUseContext) -> Option<HookAction> {
                Some(HookAction::block("forbidden"))
            }
        }

        // Interactive mode does NOT alter Block actions.
        let executor = HookExecutor::new()
            .with_interactivity(Interactivity::Interactive)
            .with_hook(Arc::new(BlockOnlyHook));
        let ctx = dummy_pre_ctx();
        let action = executor.check_pre_tool_use(&ctx);
        assert!(action.is_block());
        assert_eq!(action.block_reason(), Some("forbidden"));
    }

    #[test]
    fn no_hooks_returns_allow_in_headless() {
        let executor = HookExecutor::new();
        let ctx = dummy_pre_ctx();
        assert!(executor.check_pre_tool_use(&ctx).is_allow());
    }

    #[test]
    fn no_hooks_returns_allow_in_interactive() {
        let executor = HookExecutor::new().with_interactivity(Interactivity::Interactive);
        let ctx = dummy_pre_ctx();
        assert!(executor.check_pre_tool_use(&ctx).is_allow());
    }
}
