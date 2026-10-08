//! Rolling compaction: age turns incrementally instead of one giant pass.
//!
//! [`RollingCompactor`] implements [`ContextCompactor`] against a
//! caller-supplied [`ApiClient`](crate::api::ApiClient) by summarizing
//! each *aged* turn group with its own bounded call, instead of
//! presenting a whole history to one summarizer. Pass cost scales with
//! one turn, not the conversation: a small-model summarizer sees a
//! small input every time (sharp extraction, no monolithic-input
//! amnesia), and no single call stalls the run for minutes.
//!
//! The composition shape is a chain-leading stage:
//!
//! ```rust,ignore
//! use loopctl::compact::rolling::RollingCompactor;
//! use loopctl::compact::rolling::RollingConfig;
//! use loopctl::compact::FallbackCompactor;
//!
//! let rolling = RollingCompactor::new(client.clone(), RollingConfig::default());
//! let chain = FallbackCompactor::builder()
//!     .stage("RollingCompactor", std::sync::Arc::new(rolling))
//!     // …the big-bang stages follow, the terminal truncate last…
//!     ;
//! ```
//!
//! Aged content becomes micro-summaries — one assistant message per
//! rolled group, accumulated across passes — and the micro-summaries
//! themselves coalesce (see [`RollingConfig`]) so a marathon session's
//! summary head stays bounded. The prior big-bang ledger — the summary
//! message an earlier pass left at the head — rides verbatim, never
//! re-summarized. Burst growth the aging cannot keep up with lands
//! over the compaction target and the chain declines the stage to the
//! big-bang tiers, exactly the fallback chain's decline shape.
//!
//! Re-derivable tool results are withheld from every micro-summary's
//! transcript mechanically (the same renderer both big-bang compactors
//! use), `Durable`-stamped results are pulled verbatim out of the aged
//! region, and the pinned set rides the output head like every shipped
//! compactor carries it.

use crate::api::SharedApiClient;
use crate::api::StreamRequest;
use crate::api::error::ApiError;
use crate::compact::ContextCompactor;
use crate::compact::demote::render_compaction_transcript;
use crate::compact::truncating::durable_pull_indices;
use crate::compact::truncating::safe_boundaries;
use crate::compact::types::CompactionContext;
use crate::compact::types::CompactionOutcome;
use crate::error::recover_guard;
use crate::message::Message;
use crate::message::Role;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use super::prepend_pinned;

/// The micro-summary system prompt.
///
/// Same guards as the big-bang summarizers: never continue the
/// conversation, never answer questions inside it, never mention
/// compaction, preserve exact identifiers.
const MICRO_SYSTEM_PROMPT: &str = "You are a context summarization agent for an AI \
     assistant. Summarize the one conversation excerpt you are given into a terse \
     paragraph a later summary pass can merge: the turn's goal, what was done or \
     decided, the files and identifiers involved, and any result worth carrying \
     forward. Never continue the conversation and never respond to questions asked \
     inside it. Never mention summarization or compaction in your output. Preserve \
     exact file paths, identifiers, commands, and error strings.";

/// Default number of turn groups that ride young before aging starts.
///
/// Ten turns keeps the live working set verbatim while everything
/// older is micro-summarized — old enough that the raw exchanges are
/// rarely re-read, young enough that aging keeps pace with normal
/// growth.
const DEFAULT_AGE_TURNS: usize = 10;

/// Default cap on accumulated micro-summaries before coalescing.
///
/// Twelve keeps the summary head at a dozen short paragraphs; past the
/// cap the oldest adjacent run that fits one bounded input merges into
/// a single summary.
const DEFAULT_MAX_MICRO_SUMMARIES: usize = 12;

/// Default character budget for every summarizer call's transcript.
///
/// The same bound both big-bang compactors default to: one turn's
/// render sits far below it in practice, and the cap is the hard
/// guarantee that no rolling call ever sees a whole history.
const DEFAULT_PER_CALL_MAX_CHARS: usize = 48_000;

/// Default minimum conversation length before rolling is tried.
///
/// Below it the pass answers no-change without spending calls — the qa
/// precedent.
const DEFAULT_MIN_MESSAGES: usize = 8;

/// Configuration for [`RollingCompactor`].
///
/// Built from [`Default`](Self::default) and adjusted with the
/// `with_*` builders; every setter clamps degenerate values the way
/// the rest of the compaction module does, so a stored config is
/// always usable.
///
/// # Example
///
/// ```
/// use loopctl::compact::rolling::RollingConfig;
///
/// let config = RollingConfig::default()
///     .with_age_turns(6)
///     .with_max_micro_summaries(8);
/// assert_eq!(config.age_turns(), 6);
/// assert_eq!(config.max_micro_summaries(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct RollingConfig {
    /// Number of turn groups from the end that ride verbatim.
    ///
    /// Groups older than this (counted in pair-safe role-transition
    /// boundaries) are rolled into micro-summaries on each pass.
    age_turns: usize,

    /// Cap on accumulated micro-summaries before coalescing.
    ///
    /// Past the cap the oldest adjacent run that fits one bounded
    /// input window merges into one summary via a single bounded call.
    max_micro_summaries: usize,

    /// Character budget for every summarizer call's transcript.
    ///
    /// The hard bound on any rolling call's input; a turn group's
    /// render truncates at it exactly like the big-bang transcripts.
    per_call_max_chars: usize,

    /// Minimum conversation length before rolling is tried.
    ///
    /// Below it the pass answers no-change without spending calls; the
    /// qa compactor's precedent for keeping short chats free.
    min_messages: usize,
}

impl RollingConfig {
    /// The validated configuration with every field at its default.
    ///
    /// The single construction site for defaults; [`Default`] delegates
    /// here.
    fn fresh() -> Self {
        Self {
            age_turns: DEFAULT_AGE_TURNS,
            max_micro_summaries: DEFAULT_MAX_MICRO_SUMMARIES,
            per_call_max_chars: DEFAULT_PER_CALL_MAX_CHARS,
            min_messages: DEFAULT_MIN_MESSAGES,
        }
    }

    /// Set the number of young turn groups that ride verbatim.
    ///
    /// Groups older than this roll on the next pass; values below `1`
    /// clamp to `1` (a zero age would summarize the live turn).
    #[must_use]
    pub fn with_age_turns(mut self, turns: usize) -> Self {
        self.age_turns = turns.max(1);
        self
    }

    /// Set the cap on accumulated micro-summaries.
    ///
    /// Past the cap the oldest adjacent run that fits one bounded
    /// input window coalesces; values below `2` clamp to `2` (coalescing
    /// needs at least a pair to merge).
    #[must_use]
    pub fn with_max_micro_summaries(mut self, cap: usize) -> Self {
        self.max_micro_summaries = cap.max(2);
        self
    }

    /// Set the character budget for every summarizer call's transcript.
    ///
    /// Values below `1` clamp to `1`.
    #[must_use]
    pub fn with_per_call_max_chars(mut self, max_chars: usize) -> Self {
        self.per_call_max_chars = max_chars.max(1);
        self
    }

    /// Set the minimum conversation length before rolling is tried.
    ///
    /// Below it `compact` returns a no-change outcome without spending
    /// any calls; values below `2` clamp to `2`.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// The number of young turn groups that ride verbatim.
    ///
    /// The stored value after the `max(1)` clamp.
    #[must_use]
    pub fn age_turns(&self) -> usize {
        self.age_turns
    }

    /// The cap on accumulated micro-summaries.
    ///
    /// The stored value after the `max(2)` clamp.
    #[must_use]
    pub fn max_micro_summaries(&self) -> usize {
        self.max_micro_summaries
    }

    /// The character budget for every summarizer call's transcript.
    ///
    /// The stored value after the `max(1)` clamp.
    #[must_use]
    pub fn per_call_max_chars(&self) -> usize {
        self.per_call_max_chars
    }

    /// The minimum conversation length before rolling is tried.
    ///
    /// The stored value after the `max(2)` clamp.
    #[must_use]
    pub fn min_messages(&self) -> usize {
        self.min_messages
    }
}

impl Default for RollingConfig {
    fn default() -> Self {
        Self::fresh()
    }
}

/// The accumulated rolling state across passes within a session.
///
/// Written only at the end of a successful pass, read once at pass
/// entry — the [`QaSummarizer`](crate::compact::QaSummarizer) prior
/// discipline: no lock is held across an LLM call, and a pass dropped
/// mid-flight (the trait's cancellation-safety contract) leaves the
/// state at its last committed value.
#[derive(Debug, Clone, Default)]
struct RollingState {
    /// The text of the big-bang ledger message this compactor carries.
    ///
    /// Recognized by exact text equality at the next pass's head and
    /// excluded from aging — the ledger rides verbatim, never
    /// re-summarized.
    ledger: Option<String>,

    /// The accumulated micro-summary texts, oldest first.
    ///
    /// One entry per rolled group, coalesced at the cap; committed only
    /// at a successful pass's end.
    micro_summaries: Vec<String>,
}

/// The incremental compactor: one bounded call per aged turn group.
///
/// Construct with any [`ApiClient`](crate::api::ApiClient) — the same
/// caller-supplied-client shape as the big-bang summarizers, and the
/// same recommendation: a dedicated cheaper client for the
/// summarization calls. Compose as a chain-leading stage (see the
/// [module docs](self)) so burst growth the aging cannot absorb
/// declines to the big-bang tiers.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::compact::rolling::{RollingCompactor, RollingConfig};
///
/// let rolling = RollingCompactor::new(client, RollingConfig::default().with_age_turns(8));
/// ```
pub struct RollingCompactor {
    /// The client the micro-summary calls ride.
    ///
    /// Shared, not owned: the same `Arc` the loop holds, or a dedicated
    /// cheaper client — the calls are plain `create_message`.
    client: SharedApiClient,

    /// The clamped, always-usable configuration.
    ///
    /// Every setter normalized on write, so a stored config can never
    /// hold a degenerate value.
    config: RollingConfig,

    /// The accumulated state across passes, committed on success only.
    ///
    /// Read once at pass entry, written once at exit; never held across
    /// an LLM call.
    state: Mutex<RollingState>,
}

impl RollingCompactor {
    /// Create a rolling compactor bound to the given API client.
    ///
    /// Hosts reusing one instance across sessions call
    /// [`reset`](Self::reset) at the boundary.
    #[must_use]
    pub fn new(client: SharedApiClient, config: RollingConfig) -> Self {
        Self {
            client,
            config,
            state: Mutex::new(RollingState::default()),
        }
    }

    /// The carried ledger text, if a pass has adopted one.
    ///
    /// The big-bang summary message whose survival this compactor
    /// guarantees verbatim; `None` until a pass found a ledger at the
    /// head. Hosts persisting session state can serialize this value to
    /// checkpoint where the rolling accumulation stands.
    #[must_use]
    pub fn ledger_summary(&self) -> Option<String> {
        recover_guard(self.state.lock()).ledger.clone()
    }

    /// Restore a carried ledger text, e.g. from persisted session state.
    ///
    /// The persistence half of the ledger contract: a host resuming a
    /// session rebuilds its compactors fresh, and this seam hands back
    /// the ledger text a previous instance reported through
    /// [`ledger_summary`](Self::ledger_summary) so the next pass
    /// recognizes and carries the summary verbatim instead of aging it
    /// into a micro-summary. `None` clears it.
    pub fn restore_ledger(&self, text: Option<String>) {
        recover_guard(self.state.lock()).ledger = text;
    }

    /// Clear the accumulated state.
    ///
    /// Call at session start when one compactor serves more than one
    /// session: the next pass adopts whatever ledger it finds fresh and
    /// starts a new micro-summary accumulation.
    pub fn reset(&self) {
        *recover_guard(self.state.lock()) = RollingState::default();
    }

    /// Run one micro-summary call over one aged group.
    ///
    /// The only call shape this compactor makes: a system prompt, a
    /// user prompt carrying the group's transcript render under the
    /// per-call cap — the same renderer the big-bang compactors use,
    /// so re-derivable results are withheld mechanically — and a
    /// one-paragraph instruction. A blank response fails the pass.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] when the call fails or
    /// the response carries no usable text.
    async fn micro_summarize(&self, group: &[Message]) -> Result<String, ApiError> {
        let transcript = render_compaction_transcript(group, self.config.per_call_max_chars);
        let prompt = format!(
            "Summarize the conversation excerpt in <conversation> as one terse \
             paragraph. Emit only the paragraph.\n\n\
             <conversation>\n{transcript}\n</conversation>"
        );
        let request = StreamRequest::new(vec![Message::user(prompt)])
            .with_system(Some(MICRO_SYSTEM_PROMPT.to_string()));
        let response = self.client.create_message(&request).await?;
        let text = response.message.text_content();
        if text.trim().is_empty() {
            return Err(ApiError::api("the model returned an empty micro-summary"));
        }
        Ok(text)
    }

    /// Coalesce the oldest adjacent micro-summaries into one.
    ///
    /// The bounded-merge half of the count cap: takes the longest run
    /// from the oldest end whose combined render fits one per-call
    /// window (at least two — if even the oldest pair overflows, the
    /// pair merges under the renderer's own truncation), and replaces
    /// the run with the merged text. Returns the merged summaries so
    /// the caller can recount; an empty slice is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates the provider's [`ApiError`] from the merge call.
    async fn coalesce_oldest(
        &self,
        summaries: &[String],
    ) -> Result<Option<(String, usize)>, ApiError> {
        if summaries.len() < 2 {
            return Ok(None);
        }
        let mut run_end = 0usize;
        let mut run_chars = 0usize;
        for (index, summary) in summaries.iter().enumerate() {
            let chars = summary.chars().count();
            if index < 2 || run_chars.saturating_add(chars) <= self.config.per_call_max_chars {
                run_chars = run_chars.saturating_add(chars);
                run_end = index.saturating_add(1);
            } else {
                break;
            }
        }
        let run: Vec<Message> = summaries
            .get(..run_end)
            .unwrap_or_default()
            .iter()
            .map(|text| Message::assistant(text.clone()))
            .collect();
        let merged = self.micro_summarize(&run).await?;
        Ok(Some((merged, run_end)))
    }

    /// Assemble the pass's output from its surviving pieces.
    ///
    /// The head the history opened with — a leading system message,
    /// then the ledger message when one is carried or found — rides
    /// verbatim; the micro-summaries follow as one assistant message
    /// each, oldest first; then the young region — every index from
    /// the age boundary plus the durable-stamped indices pulled out of
    /// the aged slice — in conversation order; the pinned set rides at
    /// the very head through the shared carriage helper.
    fn assemble(
        messages: &[Message],
        head_indices: &[usize],
        summaries: &[String],
        young_from: usize,
        durable: &[usize],
    ) -> Vec<Message> {
        let mut out = Vec::with_capacity(
            head_indices
                .len()
                .saturating_add(summaries.len())
                .saturating_add(messages.len().saturating_sub(young_from))
                .saturating_add(durable.len()),
        );
        for index in head_indices {
            if let Some(message) = messages.get(*index) {
                out.push(message.clone());
            }
        }
        for summary in summaries {
            out.push(Message::assistant(summary.clone()));
        }
        let mut kept: Vec<usize> = (young_from..messages.len()).collect();
        for index in durable {
            if !kept.contains(index) {
                kept.push(*index);
            }
        }
        kept.sort_unstable();
        kept.dedup();
        for index in kept {
            if let Some(message) = messages.get(index) {
                out.push(message.clone());
            }
        }
        out
    }
}

impl ContextCompactor for RollingCompactor {
    fn compact(
        &self,
        messages: Vec<Message>,
        _target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let original = messages.clone();
            let total = messages.len();
            if total <= self.config.min_messages {
                return CompactionOutcome::no_change(messages);
            }

            let boundaries = safe_boundaries(&messages);
            let Some(&age_split) = boundaries
                .iter()
                .rev()
                .nth(self.config.age_turns.saturating_sub(1))
            else {
                return CompactionOutcome::no_change(messages);
            };
            if age_split == 0 {
                return CompactionOutcome::no_change(messages);
            }

            let (carried_ledger, prior) = {
                let state = recover_guard(self.state.lock());
                (state.ledger.clone(), state.micro_summaries.clone())
            };
            let ledger_index = carried_ledger.as_ref().and_then(|ledger| {
                messages
                    .iter()
                    .take(2)
                    .position(|message| message.text_content() == *ledger)
            });
            let mut head_indices: Vec<usize> = Vec::new();
            if messages
                .first()
                .is_some_and(|first| first.role == Role::System)
            {
                head_indices.push(0);
            }
            if let Some(index) = ledger_index
                && !head_indices.contains(&index)
            {
                head_indices.push(index);
            }

            let durable = durable_pull_indices(&messages, age_split);
            let carried_summaries: Vec<usize> = (0..age_split)
                .filter(|index| {
                    prior.iter().any(|summary| {
                        messages
                            .get(*index)
                            .is_some_and(|message| message.text_content() == *summary)
                    })
                })
                .collect();
            let survivor: Vec<usize> = head_indices
                .iter()
                .copied()
                .chain(durable.iter().copied())
                .chain(carried_summaries.iter().copied())
                .collect();
            let aged: Vec<usize> = (0..age_split)
                .filter(|index| !survivor.contains(index))
                .collect();
            if aged.is_empty() {
                return CompactionOutcome::no_change(messages);
            }
            let aged_groups: Vec<Vec<Message>> = {
                let mut groups = Vec::new();
                let mut start: Option<usize> = None;
                for index in aged.iter().copied().chain([age_split]) {
                    let at_boundary = index == age_split || boundaries.contains(&index);
                    if at_boundary {
                        if let Some(group_start) = start.take() {
                            let group: Vec<&Message> = aged
                                .iter()
                                .filter(|aged_index| {
                                    **aged_index >= group_start && **aged_index < index
                                })
                                .filter_map(|aged_index| messages.get(*aged_index))
                                .collect();
                            if !group.is_empty() {
                                let owned: Vec<Message> = group.into_iter().cloned().collect();
                                groups.push(owned);
                            }
                        }
                        if index != age_split {
                            start = Some(index);
                        }
                    } else if start.is_none() {
                        start = Some(index);
                    }
                }
                groups
            };

            let mut summaries = prior;
            for group in &aged_groups {
                match self.micro_summarize(group).await {
                    Ok(text) => summaries.push(text),
                    Err(error) => {
                        tracing::warn!(
                            target: "loopctl::compact",
                            error = %error,
                            "rolling compactor micro-summary failed; returning the original \
                             messages"
                        );
                        return CompactionOutcome::failed(
                            original,
                            context.tokens_before,
                            format!("rolling compactor micro-summary failed: {error}"),
                        );
                    }
                }
            }
            while summaries.len() > self.config.max_micro_summaries {
                match self.coalesce_oldest(&summaries).await {
                    Ok(Some((merged, run_end))) => {
                        summaries.drain(..run_end);
                        summaries.insert(0, merged);
                    }
                    Ok(None) => break,
                    Err(error) => {
                        return CompactionOutcome::failed(
                            original,
                            context.tokens_before,
                            format!("rolling compactor coalesce failed: {error}"),
                        );
                    }
                }
            }

            let mut out = Self::assemble(&messages, &head_indices, &summaries, age_split, &durable);
            prepend_pinned(&mut out, &context.pinned);
            let tokens_after = context.counter.count(&out);
            if tokens_after >= context.tokens_before {
                return CompactionOutcome::no_change(original);
            }
            let evicted: Vec<Message> = aged
                .iter()
                .filter_map(|index| messages.get(*index).cloned())
                .collect();
            *recover_guard(self.state.lock()) = RollingState {
                ledger: carried_ledger.or_else(|| {
                    ledger_index.map(|index| {
                        messages
                            .get(index)
                            .map(Message::text_content)
                            .unwrap_or_default()
                    })
                }),
                micro_summaries: summaries,
            };
            CompactionOutcome::compacted(out, context.tokens_before, tokens_after)
                .with_evicted(evicted)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ApiClient;
    use crate::api::NonStreamingResponse;
    use crate::api::StreamRequest;
    use crate::api::error::ApiError;
    use crate::compact::HeuristicTokenCounter;
    use crate::compact::TokenCounter;
    use crate::compact::types::CompactReason;
    use crate::compact::{FallbackCompactor, TruncatingCompactor};
    use crate::stream::StreamEvent;
    use crate::stream::StreamStopReason;
    use futures::Stream;
    use std::collections::VecDeque;
    use std::sync::Arc;

    fn ok(text: &str) -> Result<NonStreamingResponse, ApiError> {
        Ok(NonStreamingResponse {
            message: Message::assistant(text),
            stop_reason: StreamStopReason::EndTurn,
            usage: None,
        })
    }

    /// An `ApiClient` double recording every call's prompt and serving
    /// the scripted responses in order.
    struct RecordingClient {
        prompts: Mutex<Vec<String>>,
        script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
    }

    impl RecordingClient {
        fn new(script: Vec<Result<NonStreamingResponse, ApiError>>) -> Arc<Self> {
            Arc::new(Self {
                prompts: Mutex::new(Vec::new()),
                script: Mutex::new(script.into()),
            })
        }

        fn prompts(&self) -> Vec<String> {
            recover_guard(self.prompts.lock()).clone()
        }
    }

    impl ApiClient for RecordingClient {
        fn model(&self) -> String {
            "rolling-test".to_string()
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            recover_guard(self.prompts.lock())
                .push(request.messages.iter().map(Message::text_content).collect());
            let next = recover_guard(self.script.lock())
                .pop_front()
                .expect("every scripted call gets a response");
            Box::pin(std::future::ready(next))
        }
    }

    /// Turn groups: a small system head is not used here — plain
    /// user/assistant pairs with probe-able unique text per turn.
    fn turn_history(turns: usize, chars: usize) -> Vec<Message> {
        let mut messages = Vec::new();
        for turn in 0..turns {
            messages.push(Message::user(format!(
                "request-{turn} {}",
                "u".repeat(chars)
            )));
            messages.push(Message::assistant(format!(
                "response-{turn} {}",
                "a".repeat(chars)
            )));
        }
        messages
    }

    fn context_for(messages: &[Message]) -> CompactionContext {
        CompactionContext {
            tokens_before: HeuristicTokenCounter.count(messages),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000,
            turn: 4,
            counter: Arc::new(HeuristicTokenCounter),
            instructions: None,
            additional_context: Vec::new(),
            pinned: Vec::new(),
        }
    }

    #[tokio::test]
    async fn every_rolling_call_sees_one_turn_of_input() {
        let client = RecordingClient::new(vec![ok("summary of turn A"), ok("summary of turn B")]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(2)
                .with_min_messages(4),
        );
        let messages = turn_history(4, 40);
        let outcome = rolling
            .compact(messages, 40_000, context_for(&turn_history(4, 40)))
            .await;
        assert!(
            outcome.success,
            "the rolling pass runs: {}",
            outcome.error.as_deref().unwrap_or_default()
        );
        let prompts = client.prompts();
        assert_eq!(prompts.len(), 2, "one call per aged turn group");
        assert!(
            prompts[0].contains("request-0") && prompts[0].contains("response-0"),
            "the first call's transcript is exactly the first aged group"
        );
        assert!(
            !prompts[0].contains("request-1"),
            "no call ever sees a second turn's content: {}",
            prompts[0]
        );
        assert!(
            prompts[1].contains("request-1") && !prompts[1].contains("request-2"),
            "the second call's transcript is exactly the second aged group: {}",
            prompts[1]
        );
    }

    #[tokio::test]
    async fn aged_turns_roll_into_micro_summaries_while_young_groups_ride_verbatim() {
        let client = RecordingClient::new(vec![ok("micro-A"), ok("micro-B")]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(2)
                .with_min_messages(4),
        );
        let messages = turn_history(4, 40);
        let outcome = rolling
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let texts: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        assert!(
            texts.contains(&"micro-A".to_string()) && texts.contains(&"micro-B".to_string()),
            "each aged group becomes one micro-summary message"
        );
        assert!(
            texts.contains(&format!("request-2 {}", "u".repeat(40))),
            "the young groups ride verbatim"
        );
        assert_eq!(
            outcome.evicted.len(),
            4,
            "exactly the aged raw messages were evicted"
        );
    }

    #[tokio::test]
    async fn micro_summaries_coalesce_when_the_count_cap_is_exceeded() {
        let client = RecordingClient::new(vec![
            ok("micro-1"),
            ok("micro-2"),
            ok("micro-3"),
            ok("merged old run"),
        ]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(1)
                .with_min_messages(4)
                .with_max_micro_summaries(2),
        );
        let first = turn_history(2, 40);
        let outcome_one = rolling
            .compact(first, 40_000, context_for(&turn_history(2, 40)))
            .await;
        assert!(outcome_one.success, "the first pass rolls one group");
        let mut second = outcome_one.messages.clone();
        second.extend(turn_history(2, 40));
        let outcome_two = rolling
            .compact(second, 40_000, context_for(&turn_history(2, 40)))
            .await;
        assert!(outcome_two.success, "the second pass rolls");
        let prompts = client.prompts();
        assert_eq!(
            prompts.len(),
            4,
            "one aging call in pass one, two in pass two, plus one bounded coalesce call"
        );
        assert!(
            prompts[3].contains("micro-1") && prompts[3].contains("micro-2"),
            "the coalesce call merges the oldest run: {}",
            prompts[3]
        );
        let texts: Vec<String> = outcome_two
            .messages
            .iter()
            .map(Message::text_content)
            .collect();
        assert!(
            texts.contains(&"merged old run".to_string()),
            "the merged run replaces the oldest micro-summaries"
        );
        assert!(
            !texts.contains(&"micro-1".to_string()),
            "the merged run's inputs do not survive alone"
        );
    }

    #[tokio::test]
    async fn rolling_declines_cleanly_into_the_chain() {
        let client = RecordingClient::new(vec![]);
        let rolling = Arc::new(RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default(),
        ));
        let chain = FallbackCompactor::builder()
            .stage("RollingCompactor", rolling as Arc<dyn ContextCompactor>)
            .terminal(&TruncatingCompactor::new())
            .build();
        let short = turn_history(2, 40);
        let outcome = chain.compact(short.clone(), 40, context_for(&short)).await;
        assert!(
            outcome.success,
            "a below-min rolling stage declines to the terminal without failing"
        );
    }

    #[tokio::test]
    async fn the_ledger_rides_a_rolling_pass_and_a_big_bang_pass_identically() {
        let ledger_text = "## Conversation summary (compacted)\n- fact one";
        let client = RecordingClient::new(vec![ok("micro-A"), ok("micro-B")]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(2)
                .with_min_messages(4),
        );
        rolling.restore_ledger(Some(ledger_text.to_string()));
        let mut messages = vec![Message::assistant(ledger_text)];
        messages.extend(turn_history(4, 40));
        let outcome = rolling
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        assert!(
            outcome
                .messages
                .iter()
                .any(|m| m.role == Role::Assistant && m.text_content() == ledger_text),
            "the ledger message survives the rolling pass verbatim"
        );
        let ledger_evicted = outcome
            .evicted
            .iter()
            .any(|m| m.text_content() == ledger_text);
        assert!(
            !ledger_evicted,
            "the carried ledger is never handed to the sink"
        );
        let second = rolling
            .compact(outcome.messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            second
                .messages
                .iter()
                .any(|m| m.text_content() == ledger_text),
            "the ledger rides the second pass identically"
        );
    }

    #[tokio::test]
    async fn a_burst_the_aging_cannot_absorb_declines_to_the_chain_terminal() {
        let client = RecordingClient::new(vec![
            ok(&"verbose ".repeat(2_000)),
            ok(&"verbose ".repeat(2_000)),
        ]);
        let rolling = Arc::new(RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(1)
                .with_min_messages(4),
        ));
        let chain = FallbackCompactor::builder()
            .stage("RollingCompactor", rolling as Arc<dyn ContextCompactor>)
            .terminal(
                &TruncatingCompactor::new()
                    .with_min_messages(2)
                    .with_preserve_recent(2),
            )
            .build();
        let burst = turn_history(3, 400);
        let outcome = chain.compact(burst.clone(), 560, context_for(&burst)).await;
        assert!(outcome.success);
        assert!(
            HeuristicTokenCounter.count(&outcome.messages) <= 560,
            "the chain's landing is verified regardless of which stage carried it"
        );
        let report = chain.last_report().expect("a report is stored per run");
        assert_eq!(
            report.winning_stage,
            Some(1),
            "an over-target rolling landing declines to the terminal under the \
             verified-landing rule"
        );
    }

    #[tokio::test]
    async fn growth_faster_than_aging_still_lands_under_the_window() {
        let client = RecordingClient::new(vec![
            ok("micro-1"),
            ok("micro-2"),
            ok("micro-3"),
            ok("micro-4"),
            ok("micro-5"),
            ok("micro-6"),
        ]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default()
                .with_age_turns(1)
                .with_min_messages(4),
        );
        let mut messages = turn_history(2, 40);
        for round in 0..3 {
            let snapshot = messages.clone();
            let outcome = rolling
                .compact(snapshot.clone(), 40_000, context_for(&snapshot))
                .await;
            assert!(outcome.success, "round {round} rolls");
            messages = outcome.messages;
            messages.extend(turn_history(1, 40));
        }
        let total = HeuristicTokenCounter.count(&messages);
        let raw = HeuristicTokenCounter.count(&turn_history(5, 40));
        assert!(
            total < raw,
            "three rolling rounds leave the history ({total}) well under its raw size ({raw})"
        );
        assert!(
            !client.prompts().is_empty(),
            "the rounds spent their aging calls"
        );
    }

    #[test]
    fn rolling_config_clamps_degenerate_values() {
        let config = RollingConfig::default()
            .with_age_turns(0)
            .with_max_micro_summaries(1)
            .with_per_call_max_chars(0)
            .with_min_messages(1);
        assert_eq!(config.age_turns(), 1, "a zero age would roll the live turn");
        assert_eq!(config.max_micro_summaries(), 2, "coalescing needs a pair");
        assert_eq!(config.per_call_max_chars(), 1);
        assert_eq!(config.min_messages(), 2);
        assert_eq!(RollingConfig::default().age_turns(), 10);
        assert_eq!(RollingConfig::default().max_micro_summaries(), 12);
        assert_eq!(RollingConfig::default().per_call_max_chars(), 48_000);
    }

    #[test]
    fn the_chain_builder_accepts_a_rolling_leading_stage() {
        let client = RecordingClient::new(vec![]);
        let rolling = RollingCompactor::new(
            Arc::clone(&client) as SharedApiClient,
            RollingConfig::default(),
        );
        let chain: FallbackCompactor = FallbackCompactor::builder()
            .stage(
                "RollingCompactor",
                Arc::new(rolling) as Arc<dyn ContextCompactor>,
            )
            .terminal(&TruncatingCompactor::new())
            .build();
        assert_eq!(
            chain.stage_names(),
            vec!["RollingCompactor", "TruncatingCompactor"]
        );
    }
}
