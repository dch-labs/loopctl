//! Turn-boundary context injection.
//!
//! A [`ContextContributor`] produces an optional [`Message`] that the loop
//! appends to the conversation before the next model call. Register one on
//! [`BareLoop`](crate::engine::BareLoop) via
//! [`add_contributor`](crate::engine::BareLoop::add_contributor); the loop
//! consults every registered contributor at the top of each turn, after
//! [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) and before
//! the model is called. A typical use is re-emitting the agent's goal or
//! current plan every few turns so a small model stays on-task.

use crate::message::Message;

/// Produces an optional message to inject before the next turn's model call.
///
/// Register an implementor on
/// [`BareLoop`](crate::engine::BareLoop) via
/// [`add_contributor`](crate::engine::BareLoop::add_contributor). The loop
/// consults every registered contributor at the top of each turn, after
/// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) and before
/// the model is called. Returning [`None`] injects nothing for that turn.
///
/// The *mechanism* (when and how to inject) is the framework's; the *policy*
/// (what to inject) is the implementor's. A typical implementor re-emits the
/// agent's goal or current plan every N turns to keep a small model on-task.
///
/// The returned message, if any, is pushed onto the conversation history (so
/// it is visible to compaction and subsequent turns) and reaches the model as
/// part of the next request. Returning a message with
/// [`Role::System`](crate::message::Role::System) lets providers route the
/// content correctly.
///
/// # Examples
///
/// ```rust,ignore
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use loopctl::contributor::{ContextContributor, ContributorContext};
/// use loopctl::message::{Message, MessagePart, Role};
///
/// // Re-emit a reminder every 5 turns.
/// struct GoalReminder { goal: String, calls: AtomicUsize }
///
/// impl ContextContributor for GoalReminder {
///     fn contribute(&self, ctx: &ContributorContext<'_>) -> Option<Message> {
///         let n = self.calls.fetch_add(1, Ordering::Relaxed);
///         if n > 0 && n % 5 == 0 {
///             Some(Message::new(Role::System, vec![
///                 MessagePart::text(format!("Reminder: {}", self.goal)),
///             ]))
///         } else {
///             None
///         }
///     }
/// }
/// ```
pub trait ContextContributor: Send + Sync {
    /// Inspect the current turn count and conversation snapshot, returning a
    /// message to append to the conversation before the model call.
    ///
    /// The returned message is prepended to that turn's outbound request in
    /// registration order, so it reaches the model this turn; it is **not**
    /// persisted into history — it reappears in later turns only if the
    /// contributor returns it again. When a turn defers to compaction after
    /// the consultation, the retried turn consults the contributors again
    /// against the compacted history.
    fn contribute(&self, ctx: &ContributorContext<'_>) -> Option<Message>;
}

/// Read-only view passed to a [`ContextContributor`] at the turn boundary.
///
/// Borrows the conversation slice for the duration of the consultation only;
/// the loop constructs and drops this value synchronously at the top of each
/// turn, so it is not [`Clone`].
#[derive(Debug)]
pub struct ContributorContext<'a> {
    /// The 0-indexed turn about to execute.
    ///
    /// Matches the `turn` field reported by
    /// [`on_turn_start`](crate::observer::LoopObserver::on_turn_start) for the
    /// same turn.
    pub turn: usize,

    /// A read-only snapshot of the conversation so far.
    ///
    /// Includes the system prompt only when it was carried inline; the loop
    /// passes the same history the model will see on this turn.
    pub conversation: &'a [Message],
}

impl<'a> ContributorContext<'a> {
    /// Construct a contributor context from a turn number and a conversation
    /// slice.
    ///
    /// `conversation` is borrowed for the lifetime of the resulting context;
    /// the loop is the only intended caller and drops the context before the
    /// borrowed history is mutably borrowed again.
    #[must_use]
    pub fn new(turn: usize, conversation: &'a [Message]) -> Self {
        Self { turn, conversation }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessagePart, Role};

    #[test]
    fn test_contributor_context_new_round_trip() {
        let msgs = vec![Message::user("hello"), Message::assistant("hi")];
        let ctx = ContributorContext::new(3, &msgs);
        assert_eq!(ctx.turn, 3);
        assert_eq!(ctx.conversation.len(), 2);
        // Borrowed slice identity: same pointer + length as the input.
        assert_eq!(ctx.conversation.as_ptr(), msgs.as_ptr());
        assert_eq!(ctx.conversation.len(), msgs.len());
    }

    #[test]
    fn test_contributor_context_debug_formats() {
        let msgs = vec![Message::user("hello")];
        let ctx = ContributorContext::new(7, &msgs);
        let s = format!("{ctx:?}");
        assert!(s.contains("turn"), "Debug output should mention turn: {s}");
        assert!(
            s.contains('7'),
            "Debug output should mention turn number: {s}"
        );
    }

    #[test]
    fn test_contributor_context_empty_conversation() {
        let msgs: Vec<Message> = Vec::new();
        let ctx = ContributorContext::new(0, &msgs);
        assert_eq!(ctx.turn, 0);
        assert!(ctx.conversation.is_empty());
    }

    #[test]
    fn test_contributor_trait_object_safe() {
        struct Always;
        impl ContextContributor for Always {
            fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
                Some(Message::new(
                    Role::System,
                    vec![MessagePart::text("stay on task")],
                ))
            }
        }

        let boxed: Box<dyn ContextContributor> = Box::new(Always);
        let msgs = vec![Message::user("go")];
        let ctx = ContributorContext::new(1, &msgs);
        let out = boxed.contribute(&ctx).expect("Always returns Some");
        assert_eq!(out.role, Role::System);
        assert_eq!(out.parts.len(), 1);
    }
}
