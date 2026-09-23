//! Named runtime profiles for `BareLoop`.
//!
//! A [`ConstrainedProfile`] bundles the small-model reliability machinery —
//! verify-on-write, tool-call memoization, output truncation, goal
//! re-injection, and strict tool-call decoding — behind one type, so a
//! consumer gets "small-model-ready by default" without per-feature wiring.
//! [`FrontierProfile`] is the named opt-out: v0.1.0-style defaults with none
//! of the small-model machinery installed.
//!
//! Apply a profile with [`ConstrainedProfile::apply`] (context manager +
//! pipeline + contributor) or compose the individual pieces
//! ([`ConstrainedProfile::session_config`],
//! [`ConstrainedProfile::run_config`],
//! [`ConstrainedProfile::pipeline_builder`],
//! [`ConstrainedProfile::request_options`]) by hand.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::engine::BareLoop;
//! use loopctl::presets::ConstrainedProfile;
//! use loopctl::tool::ToolRegistry;
//! use std::sync::Arc;
//!
//! // `client` is any ApiClient impl; `registry` is your tool set.
//! let mut agent = BareLoop::new(client, registry, ConstrainedProfile::session_config());
//! agent.set_request_options(ConstrainedProfile::request_options());
//! ConstrainedProfile::apply(&mut agent).unwrap();
//! ```

use std::sync::Arc;

use crate::config::SessionConfig;
use crate::contributor::{ContextContributor, ContributorContext};
use crate::engine::BareLoop;
use crate::engine::RunConfig;
use crate::error::LoopError;
use crate::message::{Message, MessagePart, Role};
use crate::middleware::{
    CommandVerifier, MemoizingMiddleware, NoopPathExtractor, NoopVerifier, OutputLimitMiddleware,
    ToolPipeline, VerifyMiddleware, WritePathExtractor,
};
use crate::structured::{RequestOptions, ToolConstraint};

/// Default per-tool-output cap (characters) applied by `OutputLimitMiddleware`.
///
/// Keeps one chatty tool from flooding a small model's context.
const OUTPUT_CAP_CHARS: usize = 16_384;

/// Default cache TTL (turns) for the preset's `MemoizingMiddleware`.
///
/// Long enough to save repeats within a task, short enough to track
/// edits within a session.
const MEMOIZE_TTL_TURNS: u32 = 5;

/// Default reminder cadence (turns) for [`GoalReminder`].
///
/// Re-injects the goal often enough to arrest drift without spending
/// a turn's budget on repetition.
const GOAL_REMINDER_EVERY_N_TURNS: usize = 5;

/// The default goal-reminder cadence the engine's default wiring uses.
///
/// Re-exported for [`BareLoop`'s default profile
/// wiring](crate::engine::BareLoop); the public tuning surface remains
/// [`GoalReminder::new`](GoalReminder::new) with an explicit cadence.
pub(crate) const DEFAULT_GOAL_REMINDER_EVERY_N_TURNS: usize = GOAL_REMINDER_EVERY_N_TURNS;

/// Default write-class tool names the preset wires into its middleware. Advisory.
///
/// The names cover the common coding-agent set; hosts with
/// differently-named write tools build the middleware stack directly
/// rather than through the preset.
const WRITE_TOOLS: &[&str] = &["Write", "Edit", "MultiEdit"];

/// Default memoized tool names the preset wires into its middleware. Advisory.
///
/// Read-class, side-effect-free calls — the shapes whose repeats are
/// pure waste.
const MEMOIZED_TOOLS: &[&str] = &["Read", "Glob", "Grep", "LS"];

/// Shell-class tool names the builtin-verified variant also verifies.
///
/// The default write list is edit-class only; a shell tool can write
/// too, so the verified pipeline routes these names through the
/// [`CommandVerifier`] as well.
const SHELL_TOOLS: &[&str] = &["Bash", "bash", "shell", "run_command", "execute"];

/// The small-model-tuned runtime profile.
///
/// Bundles context-budget machinery (a context manager that compacts the
/// conversation at the session's configured threshold), a smaller window and
/// fewer turns via [`session_config`](Self::session_config) /
/// [`run_config`](Self::run_config), verify-on-write
/// ([`VerifyMiddleware`] with [`NoopVerifier`]), tool-call memoization
/// ([`MemoizingMiddleware`] with [`NoopPathExtractor`]), output truncation
/// ([`OutputLimitMiddleware`]), goal re-injection ([`GoalReminder`]), and
/// strict tool-call decoding ([`ToolConstraint::Strict`]) into one coherent
/// profile.
///
/// This is the harder-problem profile: it assumes the model drifts off-goal,
/// repeats tool calls, ships broken edits, and emits malformed tool
/// arguments, and installs machinery that catches each. Frontier models
/// tolerate the same defaults fine; use [`FrontierProfile`] to opt out.
///
/// Two verification levels ship. The default [`pipeline_builder`](Self::pipeline_builder)
/// wires the no-op verifier and path extractor: the middleware registers
/// and the cache works by TTL, but nothing verifies and no path-based
/// invalidation happens — bring your own `cargo check` / `tsc` verifier
/// for project-state checks. The builtin-verified variant
/// ([`pipeline_builder_with_builtin_verification`](Self::pipeline_builder_with_builtin_verification))
/// wires loopctl's own [`CommandVerifier`] (static, no execution)
/// and [`WritePathExtractor`] (path-aware cache invalidation) with
/// zero host code.
///
/// Small-model recipe: register the `think` scratchpad tool
/// (`tool::builtin::ThinkTool`, behind the `builtin_tools` feature)
/// alongside this profile — a place to plan before acting is one of the
/// cheapest accuracy levers for local models.
pub struct ConstrainedProfile;

impl ConstrainedProfile {
    /// A [`SessionConfig`] tuned for a small model.
    ///
    /// Sets a smaller context window than the default — small models degrade
    /// faster as context fills, so the window is tightened. Compaction
    /// threshold is unchanged (tightness comes from the window, not from
    /// compacting earlier).
    ///
    /// Value: `context_window = 32_768`.
    #[must_use]
    pub fn session_config() -> SessionConfig {
        SessionConfig::default().with_context_window(32_768)
    }

    /// A [`RunConfig`] tuned for a small model.
    ///
    /// Fewer max turns than the default, since small models are more prone to
    /// non-converging tool loops. All other run-scoped knobs stay at their
    /// defaults.
    ///
    /// Value: `max_turns = 100`.
    #[must_use]
    pub fn run_config() -> RunConfig {
        RunConfig {
            max_turns: 100,
            ..RunConfig::default()
        }
    }

    /// A [`ToolPipeline`] builder pre-loaded with the small-model middleware stack.
    ///
    /// Middleware registration order (first-registered outermost):
    /// output-limit → verify → memoize. Memoize (innermost) caches the
    /// raw tool result before verify appends its diagnostics — the cache
    /// never holds a verify block, and every successful write-class
    /// call is verified anew — while the output cap (outermost,
    /// post-processing last) truncates the combined output, so
    /// verify-appended diagnostics cannot escape the cap.
    ///
    /// No `.with_core()` is set — pass the result to
    /// [`BareLoop::set_pipeline`], which attaches the tool registry.
    /// Middleware a host chains onto this builder registers *inside* the
    /// preset's memoize layer — for anything that must see the original
    /// call (a redaction scrub, say) that is the wrong slot: chain it on
    /// the pipeline after the preset's layers instead.
    #[must_use]
    pub fn pipeline_builder() -> crate::middleware::ToolPipelineBuilder {
        ToolPipeline::builder()
            .with_middleware(OutputLimitMiddleware::new(OUTPUT_CAP_CHARS))
            .with_middleware(VerifyMiddleware::new(
                Arc::new(NoopVerifier),
                WRITE_TOOLS.iter().map(|s| (*s).to_string()).collect(),
            ))
            .with_middleware(MemoizingMiddleware::new(
                MEMOIZED_TOOLS.iter().map(|s| (*s).to_string()).collect(),
                WRITE_TOOLS.iter().map(|s| (*s).to_string()).collect(),
                Arc::new(NoopPathExtractor),
                MEMOIZE_TTL_TURNS,
            ))
    }

    /// The builtin-verified [`ToolPipeline`]: the small-model stack
    /// with the shipped verifier and path extractor wired in place of
    /// the no-ops.
    ///
    /// Same stack and ordering as [`pipeline_builder`](Self::pipeline_builder),
    /// with two differences: [`VerifyMiddleware`] runs
    /// [`CommandVerifier`] (static command parse with a deny set;
    /// parent-existence and writability checks for edit/write inputs)
    /// over the write list extended with the shell-class tool names,
    /// and [`MemoizingMiddleware`] extracts paths with
    /// [`WritePathExtractor`], so a write to a path evicts cached
    /// results for that path instead of waiting out the TTL.
    ///
    /// This is the stack default construction wires; the plain
    /// [`pipeline_builder`](Self::pipeline_builder)
    /// remains the bring-your-own-verifier recipe for hosts composing
    /// their own pipeline.
    #[must_use]
    pub fn pipeline_builder_with_builtin_verification() -> crate::middleware::ToolPipelineBuilder {
        let verified: Vec<String> = WRITE_TOOLS
            .iter()
            .chain(SHELL_TOOLS.iter())
            .map(|tool| (*tool).to_string())
            .collect();
        ToolPipeline::builder()
            .with_middleware(OutputLimitMiddleware::new(OUTPUT_CAP_CHARS))
            .with_middleware(VerifyMiddleware::new(
                Arc::new(CommandVerifier::new()),
                verified.clone(),
            ))
            .with_middleware(MemoizingMiddleware::new(
                MEMOIZED_TOOLS.iter().map(|s| (*s).to_string()).collect(),
                verified,
                Arc::new(WritePathExtractor),
                MEMOIZE_TTL_TURNS,
            ))
    }

    /// [`RequestOptions`] requesting strict tool-call decoding.
    ///
    /// Apply via [`BareLoop::set_request_options`] so the constraint reaches
    /// the provider on every turn.
    #[must_use]
    pub fn request_options() -> RequestOptions {
        RequestOptions::new().with_tool_constraint(ToolConstraint::Strict)
    }

    /// Apply the profile's compaction machinery, pipeline, and goal-reminder
    /// contributor to a [`BareLoop`].
    ///
    /// Installs a [`ContextManager`](crate::compact::ContextManager) around a
    /// [`TruncatingCompactor`](crate::compact::TruncatingCompactor) with its
    /// window and threshold synced from the loop's session config, replacing
    /// whatever the constructor seeded, so the profile's context budgeting is
    /// enforced by machinery rather than left to the caller. Also sets the
    /// small-model middleware stack (via [`Self::pipeline_builder`]) and
    /// registers a [`GoalReminder`] firing every 5 turns. The apply owns
    /// the contributor seam: contributors registered before the call are
    /// cleared, so exactly one reminder rides each reminder turn (the
    /// same seam-owning semantics
    /// [`FrontierProfile::apply`](FrontierProfile::apply) carries). Does
    /// **not** set the loop's config or request options — those are set
    /// separately at construction (`BareLoop::new`) and via
    /// [`BareLoop::set_request_options`].
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if [`BareLoop::set_pipeline`] fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::engine::BareLoop;
    /// use loopctl::presets::ConstrainedProfile;
    /// use loopctl::tool::ToolRegistry;
    /// use std::sync::Arc;
    ///
    /// // `client` is any ApiClient impl.
    /// let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), ConstrainedProfile::session_config());
    /// ConstrainedProfile::apply(&mut agent).unwrap();
    /// ```
    pub fn apply<C: crate::api::ApiClient>(loop_: &mut BareLoop<C>) -> Result<(), LoopError> {
        let manager = crate::compact::ContextManager::new(Arc::new(
            crate::compact::TruncatingCompactor::default(),
        ))
        .with_context_window(loop_.session_config().context_window)
        .with_threshold(loop_.session_config().compact_threshold);
        loop_.set_context_manager(Arc::new(manager));
        loop_.set_pipeline(Self::pipeline_builder())?;
        loop_.clear_contributors();
        loop_.add_contributor(Box::new(GoalReminder::new(GOAL_REMINDER_EVERY_N_TURNS)));
        Ok(())
    }
}

/// The frontier-opt-out profile.
///
/// Restores the bare, pre-profile loop: default session/run
/// configuration, no small-model middleware, no tool-call constraint,
/// no goal re-injection. This is the explicit opt-out from the machinery
/// default construction installs — apply it with
/// [`apply`](FrontierProfile::apply), the chainable
/// [`with_profile`](crate::engine::BareLoop::with_profile), or the
/// [`Profile`] trait.
pub struct FrontierProfile;

impl FrontierProfile {
    /// A default [`SessionConfig`] — no small-model tightening.
    ///
    /// Literally [`SessionConfig::default`]: the full context budget
    /// (200k window). The opt-out counterpart to
    /// [`ConstrainedProfile::session_config`](ConstrainedProfile::session_config);
    /// the two are swap-in replacements at the call site.
    #[must_use]
    pub fn session_config() -> SessionConfig {
        SessionConfig::default()
    }

    /// A default [`RunConfig`] — no small-model tightening.
    ///
    /// Literally [`RunConfig::default`]: the full turn/token budget
    /// (200 turns). The opt-out counterpart to
    /// [`ConstrainedProfile::run_config`](ConstrainedProfile::run_config).
    #[must_use]
    pub fn run_config() -> RunConfig {
        RunConfig::default()
    }

    /// An empty middleware pipeline — no small-model middleware installed.
    ///
    /// Returns a bare [`ToolPipeline::builder()`] with nothing wired; pass it
    /// to [`BareLoop::set_pipeline`] to get the plain v0.1.0 tool-dispatch
    /// path (no verify-on-write, no memoization, no output cap). The opt-out
    /// counterpart to
    /// [`ConstrainedProfile::pipeline_builder`](ConstrainedProfile::pipeline_builder).
    #[must_use]
    pub fn pipeline_builder() -> crate::middleware::ToolPipelineBuilder {
        ToolPipeline::builder()
    }

    /// Default [`RequestOptions`] — no tool-call constraint.
    ///
    /// Literally [`RequestOptions::default`] (`tool_constraint: None`), so
    /// tool calls are unconstrained. The opt-out counterpart to
    /// [`ConstrainedProfile::request_options`](ConstrainedProfile::request_options)
    /// (which sets `Strict`); apply via [`BareLoop::set_request_options`].
    #[must_use]
    pub fn request_options() -> RequestOptions {
        RequestOptions::default()
    }

    /// Restore the bare, pre-profile loop — the default wiring, undone.
    ///
    /// The opt-out's one-call form: clears every turn-boundary contributor
    /// (default construction's goal reminder), installs the empty
    /// pipeline (no verify-on-write, no memoization, no output cap), and
    /// resets the per-turn request options to unconstrained — the exact
    /// tool-dispatch path and request shape a loop built before the
    /// default profile existed produced. Call before the first
    /// [`run`](crate::engine::core::Loop::run); the chainable spelling is
    /// [`BareLoop::with_profile`](crate::engine::BareLoop::with_profile).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut agent = BareLoop::new(client, tools, config);
    /// FrontierProfile::apply(&mut agent).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates [`set_pipeline`](crate::engine::BareLoop::set_pipeline)'s
    /// validation error; the empty builder it installs cannot fail, so in
    /// practice this returns `Ok(())`.
    pub fn apply<C: crate::api::ApiClient>(loop_: &mut BareLoop<C>) -> Result<(), LoopError> {
        loop_.clear_contributors();
        loop_.set_pipeline(Self::pipeline_builder())?;
        loop_.set_request_options(Self::request_options());
        Ok(())
    }
}

/// A named bundle of loop wiring.
///
/// The profile system's seam: a profile is anything that can wire a
/// [`BareLoop`] — install middleware, contributors, request options —
/// behind one call. The shipped implementations are
/// [`ConstrainedProfile`] (the small-model bundle) and
/// [`FrontierProfile`] (the bare-loop opt-out); default construction
/// applies the constrained machinery itself, so the trait's main job for
/// hosts is the explicit opt-out via
/// [`BareLoop::with_profile`](crate::engine::BareLoop::with_profile).
/// Call before the first [`run`](crate::engine::core::Loop::run).
pub trait Profile: Send + Sync {
    /// Wire the loop with this profile's bundle.
    ///
    /// # Errors
    ///
    /// Propagates the wiring's own failure — a profile that installs a
    /// middleware pipeline can have its composition rejected by
    /// validation.
    fn apply<C: crate::api::ApiClient>(&self, loop_: &mut BareLoop<C>) -> Result<(), LoopError>;
}

impl Profile for ConstrainedProfile {
    fn apply<C: crate::api::ApiClient>(&self, loop_: &mut BareLoop<C>) -> Result<(), LoopError> {
        ConstrainedProfile::apply(loop_)
    }
}

impl Profile for FrontierProfile {
    fn apply<C: crate::api::ApiClient>(&self, loop_: &mut BareLoop<C>) -> Result<(), LoopError> {
        FrontierProfile::apply(loop_)
    }
}

/// A [`ContextContributor`] that re-injects the first user message as a
/// [`Role::System`] reminder every `n` turns.
///
/// Small models drift off-goal after several turns of tool calling.
/// Re-emitting the original request periodically keeps the model anchored to
/// what the user actually asked for. The reminder text is the **first**
/// [`Role::User`] message in the conversation, taken verbatim — the first
/// thing the user asked is a reasonable goal proxy and avoids parsing user
/// prose.
///
/// Construct with [`GoalReminder::new`], or get a default-cadence one via
/// [`ConstrainedProfile::apply`].
pub struct GoalReminder {
    /// Reminder cadence in turns.
    ///
    /// The reminder fires when `turn > 0 && turn % every_n_turns == 0`,
    /// so the first turn of a run stays clean.
    every_n_turns: usize,
}

impl GoalReminder {
    /// Create a goal reminder that fires every `every_n_turns` turns.
    ///
    /// `every_n_turns` of 0 or 1 disables the cadence guard; with 0 the
    /// reminder never fires (turn 0 is always skipped), with 1 it fires on
    /// every turn after the first. Sensible values are 3–10.
    #[must_use]
    pub fn new(every_n_turns: usize) -> Self {
        Self { every_n_turns }
    }
}

impl ContextContributor for GoalReminder {
    fn contribute(&self, ctx: &ContributorContext<'_>) -> Option<Message> {
        // Skip turn 0 — the user's message is fresh context, no reminder yet.
        if ctx.turn == 0 || self.every_n_turns == 0 {
            return None;
        }

        if !ctx.turn.is_multiple_of(self.every_n_turns) {
            return None;
        }

        // Find the first user message and re-emit its text verbatim.
        let first_user_text = ctx.conversation.iter().find_map(|m| {
            if !matches!(m.role, Role::User) {
                return None;
            }
            let texts: Vec<&str> = m
                .parts
                .iter()
                .filter_map(|p| match p {
                    MessagePart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        })?;
        Some(Message::new(
            Role::System,
            vec![MessagePart::text(first_user_text)],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrained_config_values() {
        let session = ConstrainedProfile::session_config();
        assert_eq!(session.context_window, 32_768);

        let run = ConstrainedProfile::run_config();
        assert_eq!(run.max_turns, 100);
    }

    #[test]
    fn test_constrained_request_options_strict() {
        let opts = ConstrainedProfile::request_options();
        assert!(matches!(opts.tool_constraint, ToolConstraint::Strict));
        assert!(opts.response_format.is_none());
    }

    #[test]
    fn test_frontier_config_matches_default() {
        let session = FrontierProfile::session_config();
        assert_eq!(
            session.context_window,
            SessionConfig::default().context_window
        );

        let run = FrontierProfile::run_config();
        let default = RunConfig::default();
        assert_eq!(run.max_turns, default.max_turns);
    }

    #[test]
    fn test_frontier_request_options_default() {
        let opts = FrontierProfile::request_options();
        assert!(matches!(opts.tool_constraint, ToolConstraint::None));
        assert!(opts.response_format.is_none());
    }

    fn ctx_at(turn: usize, conv: &[Message]) -> ContributorContext<'_> {
        ContributorContext::new(turn, conv)
    }

    #[test]
    fn test_goal_reminder_skips_turn_zero() {
        let reminder = GoalReminder::new(5);
        let conv = [Message::user("ship the demo")];
        let ctx = ctx_at(0, &conv);
        assert!(
            reminder.contribute(&ctx).is_none(),
            "turn 0 must be skipped"
        );
    }

    #[test]
    fn test_goal_reminder_never_fires_when_n_is_zero() {
        let reminder = GoalReminder::new(0);
        let conv = [Message::user("ship the demo")];
        // Even on a multiple-of-0-ish turn, n=0 disables firing.
        let ctx = ctx_at(5, &conv);
        assert!(
            reminder.contribute(&ctx).is_none(),
            "n=0 disables the reminder"
        );
    }

    #[test]
    fn test_goal_reminder_fires_every_n_turns() {
        let reminder = GoalReminder::new(5);
        let conv = [Message::user("goal")];
        for turn in [1, 2, 3, 4, 6, 7, 8, 9, 11] {
            let ctx = ctx_at(turn, &conv);
            assert!(
                reminder.contribute(&ctx).is_none(),
                "turn {turn} must not fire (cadence 5)"
            );
        }
        for turn in [5, 10, 15, 20] {
            let ctx = ctx_at(turn, &conv);
            assert!(
                reminder.contribute(&ctx).is_some(),
                "turn {turn} must fire (cadence 5)"
            );
        }
    }

    #[test]
    fn test_goal_reminder_injects_first_user_message_verbatim() {
        let reminder = GoalReminder::new(5);
        let conv = [
            Message::assistant("hi"),                   // not a user message
            Message::user("ship the demo"),             // first user message
            Message::user("a later unrelated request"), // must NOT be picked
        ];
        let ctx = ctx_at(5, &conv);
        let msg = reminder.contribute(&ctx).expect("turn 5 fires");
        assert_eq!(msg.role, Role::System);
        // The reminder text is the FIRST user message, verbatim.
        match &msg.parts[0] {
            MessagePart::Text { text } => assert_eq!(text, "ship the demo"),
            other => panic!("expected Text part, got {other:?}"),
        }
    }

    #[test]
    fn test_goal_reminder_returns_none_with_no_user_message() {
        let reminder = GoalReminder::new(5);
        let conv = [Message::assistant("no user here")];
        let ctx = ctx_at(5, &conv);
        assert!(
            reminder.contribute(&ctx).is_none(),
            "no user message → no reminder, even on a firing turn"
        );
    }

    #[test]
    fn test_goal_reminder_handles_multi_part_first_user_message() {
        // A first user message with multiple text parts: they are joined
        // with newlines into a single reminder.
        let reminder = GoalReminder::new(1);
        let conv = [Message::new(
            Role::User,
            vec![MessagePart::text("line one"), MessagePart::text("line two")],
        )];
        let ctx = ctx_at(1, &conv);
        let msg = reminder.contribute(&ctx).expect("turn 1 fires (cadence 1)");
        match &msg.parts[0] {
            MessagePart::Text { text } => {
                assert_eq!(text, "line one\nline two");
            }
            other => panic!("expected Text part, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn constrained_profile_output_cap_bounds_final_tool_output() {
        use crate::message::{ToolContent, ToolContentPart};
        use crate::middleware::ToolDispatchContext;
        use crate::tool::{
            PermissionCheck, Tool, ToolContext, ToolOutput, ToolRegistry, ToolSchema,
        };
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::Arc;

        struct CapSizedWriteTool;

        impl Tool for CapSizedWriteTool {
            fn name(&self) -> &'static str {
                "Write"
            }
            fn description(&self) -> &'static str {
                "Returns exactly the profile cap in characters"
            }
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    tool: self.name().to_string(),
                    description: self.description().to_string(),
                    input_schema: serde_json::json!({"type": "object", "properties": {}}),
                }
            }
            fn call(
                &self,
                _input: serde_json::Value,
                _ctx: &ToolContext,
            ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, crate::tool::ToolError>> + Send + '_>>
            {
                Box::pin(async move {
                    Ok(ToolOutput {
                        payload: ToolContent::Text("w".repeat(OUTPUT_CAP_CHARS)),
                        is_error: false,
                        display_hint: None,
                    })
                })
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(CapSizedWriteTool);
        let pipeline = ConstrainedProfile::pipeline_builder()
            .with_core(Arc::new(registry))
            .build()
            .expect("pipeline builds");

        let ctx = ToolDispatchContext {
            tool_name: "Write".to_string(),
            input: serde_json::json!({}),
            call_id: "c1".to_string(),
            turn_number: 0,
            cancel: Arc::new(crate::cancel::CancelSignal::new()),
            permission: PermissionCheck::Allow,
            tool_context: ToolContext::default(),
        };
        let result = pipeline.invoke(ctx).await;
        let len = match &result.output {
            ToolContent::Text(text) => text.chars().count(),
            ToolContent::Multipart(parts) => parts
                .iter()
                .map(|p| match p {
                    ToolContentPart::Text { text } => text.chars().count(),
                    ToolContentPart::Image { .. } => 0,
                })
                .sum(),
        };
        let expected = OUTPUT_CAP_CHARS;
        assert_eq!(
            len, expected,
            "doc: verify's appended diagnostics flow through the output cap — the combined \
             output must truncate to exactly the cap — the marker is inside the budget"
        );
    }
}
