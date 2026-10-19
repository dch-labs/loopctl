//! The loop-owned steering slot.
//!
//! Guidance a host hands a running loop between turn boundaries: a
//! user watching the transcript knows something the agent does not —
//! the file it is about to read is generated, the service it is
//! restarting is the one they are logged into. The handle is
//! cloneable and shared the way
//! [`cancel_signal`](super::BareLoop::cancel_signal) is shared,
//! because the task driving a session lives inside
//! [`run`](super::BareLoop::run)'s mutable borrow — a steering seam
//! has to be *called from outside the loop*, never on it.
//!
//! The loop itself drains the slot at each turn boundary through
//! [`LoopMachine::inject`](crate::engine::core::LoopMachine::inject),
//! so guidance enters the pending set the machine already commits
//! and the conversation the user reads is the one the model
//! continues — the property the
//! [`ContextContributor`](crate::contributor::ContextContributor)
//! path deliberately cannot offer (its messages ride one request
//! and never persist).

use std::sync::Mutex;

use crate::message::Message;

/// One slot of steering guidance, shared by every handle.
///
/// The single guarded state every method reads and writes under one
/// lock, so acceptance, draining, and sealing never interleave.
#[derive(Default)]
struct Slot {
    /// Whether a run is currently in flight and may still drain.
    ///
    /// Flipped by the loop — `open_run` ahead of a run's first turn,
    /// `close_run` at its finalize tail — and read by
    /// [`push`](Steering::push) as the acceptance gate.
    running: bool,

    /// Guidance pushed during the run in flight, in arrival order.
    ///
    /// Drained by the loop at the next turn boundary; anything still
    /// here when the run ends becomes a leftover the host owns.
    waiting: Vec<Message>,

    /// Guidance from runs that ended before a boundary could drain
    /// it, in arrival order.
    ///
    /// Sealed by the next run's opening and never delivered by a
    /// later one — between runs the host may have rewound or
    /// cancelled, and advice about a turn that no longer stands is
    /// not advice. The host collects it with
    /// [`take_leftovers`](Steering::take_leftovers) or drops it with
    /// [`clear`](Steering::clear).
    leftovers: Vec<Message>,
}

/// Guidance waiting for the loop's next turn boundary.
///
/// Cloned handles reach the same slot; the loop hands the handle out
/// through [`steering`](super::BareLoop::steering). The timing law is
/// the machine's own pinned one: guidance lands at the next turn
/// boundary, never inside an in-flight request or a dispatched tool.
pub struct Steering {
    /// The slot state, guarded so "is a run in flight" and "hand the
    /// message over" are one atomic act.
    inner: Mutex<Slot>,
}

impl Steering {
    /// Build an empty, idle slot.
    ///
    /// The loop constructs exactly one per
    /// [`BareLoop`](super::BareLoop) and hands clones of the handle
    /// out; hosts never construct their own.
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(Slot::default()),
        }
    }

    /// Offer `message` to the run in flight.
    ///
    /// `false` when no run is in flight — the message stays the
    /// caller's to deliver another way. Acceptance is atomic with
    /// the in-flight check, so a push landing the instant a run ends
    /// is either accepted (and later drained or collectable as a
    /// leftover) or refused and returned, never both and never
    /// neither.
    pub fn push(&self, message: Message) -> bool {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        if !slot.running {
            return false;
        }
        slot.waiting.push(message);
        true
    }

    /// Whether nothing is waiting.
    ///
    /// `false` while either this run's undrained guidance or a
    /// previous run's leftovers sit in the slot.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let slot = crate::error::recover_guard(self.inner.lock());
        slot.waiting.is_empty() && slot.leftovers.is_empty()
    }

    /// Take the sealed leftovers, in the order they arrived.
    ///
    /// The sealed pocket alone: guidance from runs that ended before
    /// a boundary could drain it. Reading mid-run is safe — the live
    /// queue is **not** touched, so a peek cannot steal guidance the
    /// next boundary would deliver. The pocket is empty afterwards.
    pub fn take_leftovers(&self) -> Vec<Message> {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        std::mem::take(&mut slot.leftovers)
    }

    /// Take everything the slot holds — sealed and waiting alike.
    ///
    /// Destructive, and meant for shutdown: it gathers the sealed
    /// pocket and the live run's still-waiting queue together, oldest
    /// first, leaving the slot empty. Calling it mid-run steals
    /// guidance the next boundary would have delivered — a host that
    /// is quitting anyway may want exactly that; a host that is not
    /// should read [`take_leftovers`](Self::take_leftovers) instead.
    /// The slot is empty afterwards.
    pub fn take_all(&self) -> Vec<Message> {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        let mut taken = std::mem::take(&mut slot.leftovers);
        taken.append(&mut slot.waiting);
        taken
    }

    /// Drop everything waiting.
    ///
    /// The host that discards on cancel or quit; a cancelled run
    /// leaves the slot untouched on its own — cancellation ends a
    /// turn, it does not discard the user's words.
    pub fn clear(&self) {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        slot.waiting.clear();
        slot.leftovers.clear();
    }

    /// Open the slot for a run: seal anything waiting as a leftover,
    /// then accept pushes.
    ///
    /// Guidance left over from a previous run must not ride turn 0
    /// of this one — between runs the host may have rewound or
    /// cancelled, and advice about a turn that no longer stands is
    /// not advice. Sealing and the in-flight flip share the lock, so
    /// no push can slip between them.
    pub(crate) fn open_run(&self) {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        let waiting = std::mem::take(&mut slot.waiting);
        slot.leftovers.extend(waiting);
        slot.running = true;
    }

    /// Close the slot at the end of a run.
    ///
    /// Undrained guidance stays where it is — observable through
    /// [`is_empty`](Self::is_empty) and
    /// [`take_leftovers`](Self::take_leftovers), and collectable
    /// until a later run seals it.
    pub(crate) fn close_run(&self) {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        slot.running = false;
    }

    /// Take what the current run's next boundary should deliver.
    ///
    /// The loop's own drain: everything pushed since the last
    /// boundary, in arrival order. Leftovers never travel through
    /// here.
    pub(crate) fn drain_waiting(&self) -> Vec<Message> {
        let mut slot = crate::error::recover_guard(self.inner.lock());
        std::mem::take(&mut slot.waiting)
    }
}

impl Default for Steering {
    /// An empty, idle slot.
    ///
    /// Mirrors the loop's own construction; see
    /// [`steering`](super::BareLoop::steering) for the handle a host
    /// actually holds.
    fn default() -> Self {
        Self::new()
    }
}
