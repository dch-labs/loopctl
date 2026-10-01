//! External approval for `ask` decisions — the parked-ask channel.
//!
//! [`AskResolver`] is the async surface an engine consults when a gate
//! parks a run waiting for an external decision: the resolver receives
//! one [`PendingAsk`] — the serialized decision, with the arguments
//! reduced to the gate records' stable digest — and eventually answers
//! [`Approve`](AskResolution::Approve) or
//! [`Deny`](AskResolution::Deny). Two implementations ship: a closure
//! adapter ([`SyncAskResolver`]) for answerers that know the answer
//! immediately, and [`ApprovalChannel`], the reference channel delivery
//! — pendings stream out to a host-side [`Approver`], resolutions come
//! back by ask id, and each ask accepts exactly one answer.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::ask::{ApprovalChannel, AskResolution};
//! use loopctl::managers::LoopManagers;
//!
//! let (resolver, mut approver) = ApprovalChannel::pair(16);
//! let managers = LoopManagers::new()
//!     .with_ask_resolver(resolver)
//!     .with_ask_timeout(std::time::Duration::from_secs(300));
//!
//! // On the approver side, from any task:
//! // let pending = approver.next_pending().await;
//! // approver.resolve(pending.id, AskResolution::Approve);
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

/// One parked decision awaiting an external answer.
///
/// The serialized form of an `ask` at the moment a run parks: what the
/// approver reads to decide, and what an audit joins the eventual
/// [`GateDecision`](crate::tool::permission::GateDecision) against. The
/// arguments ride as [`args_digest`](Self::args_digest) — the same
/// canonical-JSON FNV-1a digest the gate records carry — never as
/// content. The identity fields (`session_id`, `run_id`, `turn`,
/// `call_id`) bind the ask to the one dispatch that minted it; the
/// approval that unblocks this ask cannot unblock any other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PendingAsk {
    /// The identifier of this ask, minted when the run parked.
    ///
    /// Unique within the engine that minted it; the approver quotes it
    /// back in [`Approver::resolve`] to answer exactly this ask.
    pub id: Uuid,

    /// The session whose run parked.
    ///
    /// Binds the ask to the session's identity, so an approval cannot
    /// wander across sessions a host runs concurrently.
    pub session_id: Uuid,

    /// The run that parked, by its own record id.
    ///
    /// Together with `turn` and `call_id`, the positional binding an
    /// audit uses to pair the ask with the dispatch that raised it.
    pub run_id: Uuid,

    /// The 0-indexed turn whose dispatch raised the ask.
    ///
    /// Matches the turn on the surrounding `tool.call` / `tool.result`
    /// ledger lines and observer events.
    pub turn: usize,

    /// The model-assigned call id the ask is about.
    ///
    /// The same value the dispatch's `tool.call` / `tool.result` records
    /// carry, so a ledger join lands on the right call.
    pub call_id: String,

    /// The tool the parked dispatch would execute.
    ///
    /// The name as the model called it — the approver's primary signal
    /// for what approving would let happen.
    pub tool: String,

    /// The stable digest of the call's arguments.
    ///
    /// [`GateDecision::args_digest`](crate::tool::permission::GateDecision::args_digest)
    /// of the call's input — key-order independent, content-free. An
    /// approval is bound to this digest: a later call with different
    /// arguments raises a fresh ask.
    pub args_digest: String,

    /// The prompt the asking gate produced for the approver.
    ///
    /// The hook's or gate's message — what a human approver would be
    /// shown. Only the headless denial carries it onward onto the
    /// decision record's `reason`; a refusal's and an expiry's reasons
    /// name their own causes.
    pub prompt: String,
}

/// The external answer to a parked [`PendingAsk`].
///
/// `Approve` lets the parked dispatch proceed exactly as the model
/// issued it; `Deny` returns a soft denial to the model, which may
/// adjust and try again. Expiry is not an answer — the engine's own
/// deadline policy produces it, and no resolver can produce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AskResolution {
    /// Let the parked call proceed.
    ///
    /// The dispatch resumes through the ordinary path; the approval is
    /// recorded as the decision the engine acted on.
    Approve,

    /// Refuse the parked call.
    ///
    /// The model receives a soft denial naming the refusal and may
    /// react on its next turn.
    Deny,
}

/// The async surface an engine consults to resolve a parked ask.
///
/// Implementations deliver the ask to whatever makes the decision — a
/// channel to a human operator, a polled approval store, a policy
/// service — and answer through the returned future. The future may be
/// dropped before it resolves (the engine's deadline or the cancel
/// signal ended the wait); implementations must therefore be
/// cancellation-safe and treat a dropped answer channel as an ask that
/// was never answered.
pub trait AskResolver: Send + Sync {
    /// Answer one parked ask.
    ///
    /// Receives the serialized [`PendingAsk`] and eventually returns its
    /// [`AskResolution`]. The engine races this future against its
    /// cancellation signal and, when configured, the ask deadline.
    fn resolve<'a>(
        &'a self,
        pending: &'a PendingAsk,
    ) -> Pin<Box<dyn Future<Output = AskResolution> + Send + 'a>>;
}

/// An [`AskResolver`] that answers synchronously from a closure.
///
/// The adapter for answerers that know the answer when they see the
/// ask — an allowlist check, a test double, a policy function. The
/// closure returns `true` to approve and `false` to deny; the future it
/// rides in resolves immediately.
///
/// # Example
///
/// ```
/// use loopctl::ask::SyncAskResolver;
///
/// let resolver = SyncAskResolver::new(|pending| pending.tool == "read_file");
/// ```
pub struct SyncAskResolver {
    answer: Arc<dyn Fn(&PendingAsk) -> bool + Send + Sync>,
}

impl SyncAskResolver {
    /// Build a resolver that answers from `answer`.
    ///
    /// The closure runs once per parked ask, at resolve time, on the
    /// engine's executor; `true` approves the ask and `false` denies it.
    #[must_use]
    pub fn new(answer: impl Fn(&PendingAsk) -> bool + Send + Sync + 'static) -> Self {
        Self {
            answer: Arc::new(answer),
        }
    }
}

impl AskResolver for SyncAskResolver {
    fn resolve<'a>(
        &'a self,
        pending: &'a PendingAsk,
    ) -> Pin<Box<dyn Future<Output = AskResolution> + Send + 'a>> {
        Box::pin(std::future::ready(if (self.answer)(pending) {
            AskResolution::Approve
        } else {
            AskResolution::Deny
        }))
    }
}

/// The shared state between a channel resolver and its approver.
///
/// One map entry per ask awaiting an answer: the ask id to the one-shot
/// sender that completes the engine's parked future. An entry exists
/// exactly while the engine is parked on that ask — the await slot's
/// `Drop` removes it — so a resolution arriving for a missing entry is
/// a late or duplicate answer for an ask that already ended, and
/// completes nothing.
struct ChannelState {
    /// Asks delivered to the approver side, in park order.
    ///
    /// Under parallel dispatch several calls can park at once, so the
    /// queue holds one pending per concurrently parked ask — size the
    /// channel to the run's `max_concurrency` when parallel dispatch
    /// can ask, or a later park waits for the approver to drain an
    /// earlier pending (it waits, it is never lost).
    outgoing: tokio::sync::mpsc::Sender<PendingAsk>,

    /// Per-ask completion slots, present while the engine is parked.
    ///
    /// Keyed by ask id: the approver quotes the id back in
    /// [`resolve`](Approver::resolve), and a missing entry means the
    /// wait already ended — the answer is late and completes nothing.
    awaiting: Mutex<HashMap<Uuid, tokio::sync::oneshot::Sender<AskResolution>>>,
}

/// Removes its ask's completion slot when the park ends.
///
/// The engine's deadline or cancel signal can drop the parked future at
/// any await point; this guard closes the map entry on the way out so
/// the channel never holds a slot whose receiver is gone.
struct AwaitSlot {
    state: Arc<ChannelState>,
    id: Uuid,
}

impl Drop for AwaitSlot {
    fn drop(&mut self) {
        if let Ok(mut awaiting) = self.state.awaiting.lock() {
            awaiting.remove(&self.id);
        }
    }
}

/// The reference [`AskResolver`] for channel-delivered approvals.
///
/// Built as one half of [`ApprovalChannel::pair`]: the engine-side half
/// delivers every parked ask to the approver's receiver and awaits that
/// ask's one answer. An approver that is gone (the receiver dropped)
/// makes every ask deny — a channel with no one listening fails closed,
/// the same posture a headless run has.
#[derive(Clone)]
pub struct ApprovalChannel {
    state: Arc<ChannelState>,
}

impl ApprovalChannel {
    /// Create the resolver half paired with a fresh [`Approver`].
    ///
    /// `capacity` bounds how many pendings may sit undelivered before
    /// the engine's delivery blocks. Parallel dispatch can park one ask
    /// per concurrently running call, so size the capacity to the run's
    /// `max_concurrency` when parallel dispatch can ask — a smaller
    /// capacity still delivers every ask, but a later park waits for
    /// the approver to drain an earlier pending. Install the returned
    /// resolver on [`LoopManagers`](crate::managers::LoopManagers) with
    /// [`with_ask_resolver`](crate::managers::LoopManagers::with_ask_resolver)
    /// and hold the [`Approver`] on the side that answers.
    #[must_use]
    pub fn pair(capacity: usize) -> (Arc<Self>, Approver) {
        let (outgoing, incoming) = tokio::sync::mpsc::channel(capacity.max(1));
        let state = Arc::new(ChannelState {
            outgoing,
            awaiting: Mutex::new(HashMap::new()),
        });
        (
            Arc::new(Self {
                state: Arc::clone(&state),
            }),
            Approver { state, incoming },
        )
    }
}

impl AskResolver for ApprovalChannel {
    fn resolve<'a>(
        &'a self,
        pending: &'a PendingAsk,
    ) -> Pin<Box<dyn Future<Output = AskResolution> + Send + 'a>> {
        let state = Arc::clone(&self.state);
        let pending = pending.clone();
        Box::pin(async move {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            if let Ok(mut awaiting) = state.awaiting.lock() {
                awaiting.insert(pending.id, sender);
            }
            let _slot = AwaitSlot {
                state: Arc::clone(&state),
                id: pending.id,
            };
            if state.outgoing.send(pending).await.is_err() {
                return AskResolution::Deny;
            }
            match receiver.await {
                Ok(resolution) => resolution,
                Err(_) => AskResolution::Deny,
            }
        })
    }
}

/// The answering half of an [`ApprovalChannel`].
///
/// Held by whatever makes the decision: receive each parked ask with
/// [`next_pending`](Self::next_pending), decide, and answer with
/// [`resolve`](Self::resolve) quoting the ask's id. Each ask accepts
/// exactly one answer — a second `resolve` for an id returns `false`
/// and completes nothing, so a late or duplicated answer can never
/// reach a run that already moved on. Dropping the approver denies
/// every ask still awaiting an answer — no run parks forever on an
/// answerer that is gone.
pub struct Approver {
    state: Arc<ChannelState>,
    incoming: tokio::sync::mpsc::Receiver<PendingAsk>,
}

impl Approver {
    /// Receive the next parked ask, in park order.
    ///
    /// Resolves to `None` once the resolver half and the engine holding
    /// it are gone — the channel is closed and no further ask can
    /// arrive.
    pub async fn next_pending(&mut self) -> Option<PendingAsk> {
        self.incoming.recv().await
    }

    /// Answer one parked ask.
    ///
    /// Returns `true` when the answer reached a run still parked on the
    /// ask, and `false` when no such ask exists — the id is unknown,
    /// already answered, or its wait already ended by deadline or
    /// cancellation. A `false` return leaves nothing pending: the
    /// answer is dropped, never queued.
    #[must_use]
    pub fn resolve(&self, id: Uuid, resolution: AskResolution) -> bool {
        let Ok(mut awaiting) = self.state.awaiting.lock() else {
            return false;
        };
        match awaiting.remove(&id) {
            Some(sender) => sender.send(resolution).is_ok(),
            None => false,
        }
    }
}

impl Drop for Approver {
    /// Deny every ask still awaiting an answer.
    ///
    /// The delivered ask's completion slot lives in the shared
    /// channel state, so an approver that goes away without answering
    /// would otherwise leave its parked waits suspended forever — the
    /// default no-deadline park would hang the run. Draining the
    /// slots here fails every outstanding ask closed; a send racing an
    /// already-ended wait returns an error that is deliberately
    /// ignored, exactly like a late `resolve`.
    fn drop(&mut self) {
        let Ok(mut awaiting) = self.state.awaiting.lock() else {
            return;
        };
        let senders: Vec<tokio::sync::oneshot::Sender<AskResolution>> =
            awaiting.drain().map(|(_, sender)| sender).collect();
        let denied = senders
            .into_iter()
            .filter_map(|sender| sender.send(AskResolution::Deny).ok())
            .count();
        tracing::debug!(denied, "approver dropped: outstanding asks denied");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ask(id: Uuid, tool: &str) -> PendingAsk {
        PendingAsk {
            id,
            session_id: Uuid::nil(),
            run_id: Uuid::nil(),
            turn: 0,
            call_id: "call_a".to_string(),
            tool: tool.to_string(),
            args_digest: "0123456789abcdef".to_string(),
            prompt: format!("approve {tool}?"),
        }
    }

    #[test]
    fn sync_resolver_answers_from_the_closure() {
        let approving = SyncAskResolver::new(|ask| ask.tool == "read_file");
        let refusing = SyncAskResolver::new(|_| false);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime builds");
        let ask = sample_ask(Uuid::new_v4(), "read_file");
        let approved = runtime.block_on(approving.resolve(&ask));
        let refused = runtime.block_on(refusing.resolve(&ask));
        assert_eq!(
            approved,
            AskResolution::Approve,
            "the closure's true approves the ask it saw"
        );
        assert_eq!(
            refused,
            AskResolution::Deny,
            "the closure's false denies the ask"
        );
    }

    #[tokio::test]
    async fn approval_channel_resolutions_are_single_use() {
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let parked = sample_ask(Uuid::new_v4(), "deploy");
        let id = parked.id;
        let parked_clone = parked.clone();
        let resolution_task = tokio::spawn(async move { resolver.resolve(&parked_clone).await });

        let delivered = approver.next_pending().await;
        assert_eq!(
            delivered,
            Some(parked.clone()),
            "the parked ask reaches the approver in park order"
        );

        assert!(
            approver.resolve(id, AskResolution::Approve),
            "the first answer reaches the still-parked ask"
        );
        assert_eq!(
            resolution_task
                .await
                .expect("the resolution task completes"),
            AskResolution::Approve,
            "the parked future completes with the one answer it accepted"
        );
        assert!(
            !approver.resolve(id, AskResolution::Deny),
            "a second answer for the same ask completes nothing — single use"
        );
    }

    #[tokio::test]
    async fn approval_channel_fails_closed_when_no_approver_listens() {
        let (resolver, approver) = ApprovalChannel::pair(1);
        drop(approver);
        let ask = sample_ask(Uuid::new_v4(), "deploy");
        let answer =
            tokio::time::timeout(std::time::Duration::from_secs(1), resolver.resolve(&ask))
                .await
                .expect("a closed channel denies immediately, without hanging");
        assert_eq!(
            answer,
            AskResolution::Deny,
            "an ask with no one listening denies — the headless posture"
        );
    }

    #[tokio::test]
    async fn a_late_answer_completes_nothing() {
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let parked = sample_ask(Uuid::new_v4(), "deploy");
        let id = parked.id;
        let expired = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            resolver.resolve(&parked),
        )
        .await;
        assert!(
            expired.is_err(),
            "nobody answers — the engine's deadline, not an answer, ends the wait"
        );
        assert!(
            approver.next_pending().await.is_some(),
            "the ask was still delivered before the wait ended"
        );
        assert!(
            !approver.resolve(id, AskResolution::Approve),
            "an answer after the wait ended reaches nothing — the slot is gone"
        );
    }

    #[tokio::test]
    async fn dropping_the_approver_after_delivery_denies_the_parked_ask() {
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let parked = sample_ask(Uuid::new_v4(), "deploy");
        let parked_clone = parked.clone();
        let resolution_task = tokio::spawn(async move { resolver.resolve(&parked_clone).await });
        assert!(
            approver.next_pending().await.is_some(),
            "the ask is delivered while the approver still lives"
        );
        drop(approver);
        let answer = tokio::time::timeout(std::time::Duration::from_secs(1), resolution_task)
            .await
            .expect("dropping the approver denies the parked ask — no hang")
            .expect("the resolution task completes");
        assert_eq!(
            answer,
            AskResolution::Deny,
            "the drop-denial fails the delivered ask closed"
        );
    }

    #[tokio::test]
    async fn a_delivered_denial_reaches_the_parked_ask() {
        let (resolver, mut approver) = ApprovalChannel::pair(4);
        let parked = sample_ask(Uuid::new_v4(), "deploy");
        let id = parked.id;
        let parked_clone = parked.clone();
        let resolution_task = tokio::spawn(async move { resolver.resolve(&parked_clone).await });
        assert!(
            approver.next_pending().await.is_some(),
            "the parked ask is delivered before it is answered"
        );
        assert!(
            approver.resolve(id, AskResolution::Deny),
            "the delivered refusal reaches the still-parked ask"
        );
        assert_eq!(
            resolution_task
                .await
                .expect("the resolution task completes"),
            AskResolution::Deny,
            "the parked future completes with the refusal it was delivered"
        );
    }

    #[tokio::test]
    async fn concurrent_parks_both_deliver() {
        let (resolver, mut approver) = ApprovalChannel::pair(2);
        let first = sample_ask(Uuid::new_v4(), "deploy");
        let second = sample_ask(Uuid::new_v4(), "deploy");
        let ids = [first.id, second.id];
        let first_resolver = Arc::clone(&resolver);
        let first_task = tokio::spawn(async move { first_resolver.resolve(&first).await });
        let second_task = tokio::spawn(async move { resolver.resolve(&second).await });

        let delivered: Vec<Uuid> = [approver.next_pending().await, approver.next_pending().await]
            .into_iter()
            .flatten()
            .map(|pending| pending.id)
            .collect();
        assert_eq!(
            delivered.len(),
            2,
            "two concurrently parked asks both deliver — parallel dispatch parks \
             one ask per concurrently running call"
        );
        assert!(
            delivered.iter().all(|id| ids.contains(id)),
            "each delivered pending is one of the two parked asks"
        );
        for id in delivered {
            assert!(
                approver.resolve(id, AskResolution::Approve),
                "each concurrently parked ask accepts its own answer"
            );
        }
        assert_eq!(
            first_task.await.expect("the first task completes"),
            AskResolution::Approve
        );
        assert_eq!(
            second_task.await.expect("the second task completes"),
            AskResolution::Approve
        );
    }
}
