//! Demotion sinks — where evicted conversation content goes.
//!
//! A compaction pass removes messages from the feed; a [`DemotionSink`]
//! receives what was removed so eviction is a handoff rather than a
//! discard. The engine awaits the configured sink inside the compaction
//! pass, before the compacted history replaces the old one; the default
//! sink is a no-op, which preserves the discard behavior that preceded
//! demotion.

use crate::compact::types::CompactReason;
use crate::error::LoopError;
use crate::memory::{LoopMemory, MemoryCategory, MemoryEntry};
use crate::message::{Message, MessagePart, Role};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Render budget for one stored demotion entry, in characters.
///
/// The store-side cost cap: one trajectory entry per compaction pass,
/// truncated from the full render so a large eviction cannot flood the
/// memory store. Retrieval-side cost is bounded separately by the
/// run's `memory_top_k`.
const DEFAULT_MAX_CHARS: usize = 8_000;

/// Characters of a serialized tool-call input or result kept in a render.
///
/// Per-part budget for the renderer's summaries — serialized tool-call
/// inputs and tool-result outputs truncate to this many characters, so a
/// single huge argument cannot crowd the rest of the render out of the
/// entry.
const PART_CHARS: usize = 200;

/// Receives messages a compaction pass removed from the feed.
///
/// Called once per compacting pass that removed messages, with every
/// message the pass dropped, in conversation order, before the compacted
/// history replaces the old one. Implementations must tolerate
/// redelivery — a retried pass may deliver the same slice twice — by
/// being idempotent in effect or by keying on content identity (the
/// rendered text hash) so a duplicate delivery stays detectable.
///
/// Cancellation safety: the engine awaits `demote` inside the compaction
/// pass it races against the loop's cancel signal, so a cancel can drop
/// the returned future mid-write. Implementations must be drop-safe
/// futures with no required cleanup that only runs to completion; a
/// partially applied write must not corrupt the backing store.
///
/// # Example
///
/// ```rust
/// use std::future::Future;
/// use std::pin::Pin;
///
/// use loopctl::compact::CompactReason;
/// use loopctl::compact::demote::{DemotionContext, DemotionSink};
/// use loopctl::error::LoopError;
/// use loopctl::message::Message;
///
/// struct LogSink;
///
/// impl DemotionSink for LogSink {
///     fn demote<'a>(
///         &'a self,
///         evicted: &'a [Message],
///         meta: DemotionContext,
///     ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
///         Box::pin(async move {
///             tracing::debug!(
///                 count = evicted.len(),
///                 turn = meta.turn,
///                 reason = %meta.reason,
///                 "demoted evicted turns"
///             );
///             Ok(())
///         })
///     }
/// }
/// ```
pub trait DemotionSink: Send + Sync {
    /// Take delivery of the messages a compaction pass removed.
    ///
    /// `evicted` holds every message the pass dropped, in conversation
    /// order; `meta` describes the pass itself. A returned `Err` never
    /// fails the compaction — the engine logs it once and continues with
    /// the compacted history.
    ///
    /// # Errors
    ///
    /// Returns an error when the sink rejected the delivery (a full
    /// backing store, a failed write). The engine treats the content as
    /// lost to demotion and carries on; compaction has already succeeded.
    fn demote<'a>(
        &'a self,
        evicted: &'a [Message],
        meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>>;
}

/// What the sink needs to know about the pass that evicted.
///
/// The trigger, position, and identity of the compaction pass that removed
/// the delivered messages — everything a sink needs to label, group, or
/// weight its record of the evicted content, and nothing about the content
/// itself (that arrives as the `evicted` slice).
#[derive(Debug, Clone, Copy)]
pub struct DemotionContext {
    /// Why compaction ran.
    ///
    /// The [`CompactReason`] that triggered the pass — a sink may weight
    /// emergency evictions differently from routine threshold ones, or
    /// record the tier alongside the content.
    pub reason: CompactReason,

    /// The run's turn index at eviction.
    ///
    /// Zero-indexed within the run, for staleness weighting: entries
    /// demoted at later turns describe fresher content.
    pub turn: usize,

    /// The session id.
    ///
    /// Identifies the conversation the evicted content belongs to, so a
    /// memory store can group entries by session.
    pub session_id: uuid::Uuid,
}

/// The default sink: eviction is discard.
///
/// `demote` accepts every delivery and does nothing — the observable
/// behavior of a loop with no sink configured is byte-identical to a
/// loop running before demotion existed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopDemotionSink;

impl DemotionSink for NoopDemotionSink {
    fn demote<'a>(
        &'a self,
        _evicted: &'a [Message],
        _meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

/// Renders evicted turns to text and stores them as trajectory memories.
///
/// The memory-backed sink: one [`MemoryEntry`] per compaction pass, its
/// content the rendered evicted slice and its tags the demotion origin,
/// a content hash for duplicate detection, and the session id for
/// grouping. A redelivered slice stores again — duplicates stay
/// detectable through the shared hash tag and fold when the same store
/// also backs the engine's memory wiring and a run completes
/// successfully (consolidation runs only at the finalize of successful
/// runs). Works over any [`LoopMemory`] implementation, from the
/// in-memory reference store to a vector-backed store — the sink only
/// calls `store`, so the backend decides what recall means.
#[derive(Clone)]
pub struct MemoryDemotionSink {
    /// The store receiving rendered trajectories.
    ///
    /// Shared by handle, so the same backing store can also serve the
    /// engine's per-turn retrieval — demoted content becomes recallable
    /// through the same `retrieve` the loop already consults.
    memory: Arc<dyn LoopMemory>,

    /// Maximum rendered characters per store call.
    ///
    /// Saturating truncation budget for one entry's content; see
    /// `DEFAULT_MAX_CHARS`.
    max_chars: usize,
}

impl fmt::Debug for MemoryDemotionSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryDemotionSink")
            .field("max_chars", &self.max_chars)
            .finish_non_exhaustive()
    }
}

impl MemoryDemotionSink {
    /// Create a memory-backed sink with the default render budget.
    ///
    /// Stores one trajectory entry per demotion call, rendered to at most
    /// `DEFAULT_MAX_CHARS` characters.
    #[must_use]
    pub fn new(memory: Arc<dyn LoopMemory>) -> Self {
        Self {
            memory,
            max_chars: DEFAULT_MAX_CHARS,
        }
    }

    /// Set the render budget, consuming `self`.
    ///
    /// The entry *body* for each demotion call truncates to this many
    /// characters (on a character boundary); the saturation marker —
    /// and, in the exact-fit edge, the ellipsis — appends past the
    /// budget, so the stored string may run slightly over it. The
    /// budget bounds content, not metadata. Values below the length of
    /// one rendered message truncate mid-conversation; zero keeps only
    /// the saturation marker.
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }
}

impl DemotionSink for MemoryDemotionSink {
    fn demote<'a>(
        &'a self,
        evicted: &'a [Message],
        meta: DemotionContext,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + 'a>> {
        Box::pin(async move {
            if evicted.is_empty() {
                return Ok(());
            }
            let rendered = render_evicted(evicted, self.max_chars);
            let hash = fnv1a64(rendered.as_bytes());
            let entry = MemoryEntry::new(MemoryCategory::Trajectory, rendered)
                .with_tag("demoted")
                .with_tag(format!("demoted-hash:{hash:016x}"))
                .with_tag(format!("session:{}", meta.session_id));
            self.memory.store(entry).await
        })
    }
}

/// Render evicted messages to compact natural text.
///
/// One line per message: the role (`User:`, `Assistant:`, `System:`),
/// the text content, then per-part summaries — tool calls as
/// `calls: name(input)`, tool results as `result[name, ok|err]: output`,
/// image parts as `[image]`. Serialized inputs and outputs truncate to
/// `PART_CHARS` characters. The render body saturates at `max_chars`
/// characters (never splitting a character); the trailing
/// `…[evicted {n} more messages]` marker — counting the messages not
/// fully rendered, the partially-rendered one included — appends past
/// the budget, so the returned string may exceed it by the marker's
/// length. Hosts may call this directly to build custom sinks with the
/// same shape as the memory-backed one.
///
/// # Example
///
/// ```rust
/// use loopctl::compact::demote::render_evicted;
/// use loopctl::message::Message;
///
/// let messages = vec![Message::user("hello"), Message::assistant("hi")];
/// let rendered = render_evicted(&messages, 8_000);
/// assert!(rendered.starts_with("User: hello\n"));
/// assert!(rendered.contains("Assistant: hi"));
/// ```
#[must_use]
pub fn render_evicted(messages: &[Message], max_chars: usize) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut used = 0usize;
    for (index, msg) in messages.iter().enumerate() {
        let block = render_message(msg);
        let block_len = block.chars().count();
        if used.saturating_add(block_len) <= max_chars {
            out.push_str(&block);
            used = used.saturating_add(block_len);
            continue;
        }
        let budget = max_chars.saturating_sub(used);
        let keep = budget.saturating_sub(1);
        out.extend(block.chars().take(keep));
        out.push('…');
        let unrendered = messages.len().saturating_sub(index);
        let _ignored = write!(out, "[evicted {unrendered} more messages]");
        return out;
    }
    out
}

/// Render one message as a single summary line.
///
/// The per-message unit of [`render_evicted`]: role prefix, text content,
/// then per-part summaries in part order, closed by a newline.
fn render_message(msg: &Message) -> String {
    use std::fmt::Write as _;
    let role = match msg.role {
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::System => "System",
    };
    let mut line = format!("{role}: ");
    let text = msg.text_content();
    if !text.is_empty() {
        line.push_str(&text);
    }
    for part in &msg.parts {
        match part {
            MessagePart::ToolCall { name, input, .. } => {
                let rendered_input: String = input.to_string().chars().take(PART_CHARS).collect();
                line.push_str(" calls: ");
                line.push_str(name);
                line.push('(');
                line.push_str(&rendered_input);
                line.push(')');
            }
            MessagePart::ToolResult {
                name,
                output,
                is_error,
                ..
            } => {
                let status = if is_error.unwrap_or(false) {
                    "err"
                } else {
                    "ok"
                };
                let rendered_output: String = output.to_string().chars().take(PART_CHARS).collect();
                let _ignored = write!(line, " result[{name}, {status}]: {rendered_output}");
            }
            MessagePart::Image { .. } => line.push_str(" [image]"),
            MessagePart::Text { .. } => {}
        }
    }
    line.push('\n');
    line
}

/// FNV-1a 64-bit over `bytes`.
///
/// The standard offset basis and prime. The value is fixed by the
/// algorithm, not the toolchain — unlike `DefaultHasher`, whose output
/// Rust does not guarantee across releases — so tags derived from it
/// stay comparable in stores that persist across upgrades.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::message::{ImageSource, ToolContent};
    use serde_json::json;

    async fn demote_once(sink: &MemoryDemotionSink, evicted: &[Message]) -> DemotionContext {
        let meta = DemotionContext {
            reason: CompactReason::ThresholdExceeded,
            turn: 7,
            session_id: uuid::Uuid::new_v4(),
        };
        sink.demote(evicted, meta)
            .await
            .expect("store accepts the entry");
        meta
    }

    async fn stored(store: &InMemoryStore) -> Vec<MemoryEntry> {
        store
            .retrieve("", usize::MAX)
            .await
            .expect("retrieve over the store")
    }

    #[test]
    fn render_covers_roles_calls_results_and_images() {
        let messages = vec![
            Message::user("find the weather"),
            Message::new(
                Role::Assistant,
                vec![MessagePart::tool_call(
                    "call_1",
                    "get_weather",
                    json!({"city": "Tromsø"}),
                )],
            ),
            Message::new(
                Role::User,
                vec![
                    MessagePart::tool_result(
                        "call_1",
                        "get_weather",
                        ToolContent::from_string("3C, rain"),
                        false,
                    ),
                    MessagePart::tool_result(
                        "call_2",
                        "get_weather",
                        ToolContent::from_string("boom"),
                        true,
                    ),
                ],
            ),
            Message::new(
                Role::User,
                vec![MessagePart::Image {
                    source: ImageSource::new_base64("image/png", "opaque-bytes"),
                }],
            ),
        ];
        let rendered = render_evicted(&messages, DEFAULT_MAX_CHARS);
        assert!(
            rendered.starts_with("User: find the weather\n"),
            "the first line carries the user role and text: {rendered:?}"
        );
        assert!(
            rendered.contains("Assistant:  calls: get_weather("),
            "call summaries follow the role line: {rendered:?}"
        );
        assert!(
            rendered.contains("\"city\":\"Tromsø\""),
            "the serialized input rides along: {rendered:?}"
        );
        assert!(
            rendered.contains("result[get_weather, ok]: 3C, rain"),
            "success results carry the ok status: {rendered:?}"
        );
        assert!(
            rendered.contains("result[get_weather, err]: boom"),
            "error results carry the err status: {rendered:?}"
        );
        assert!(
            rendered.contains("[image]"),
            "image parts render as a marker: {rendered:?}"
        );
    }

    #[test]
    fn render_saturates_at_max_chars_on_a_char_boundary() {
        let multi_byte = "å".repeat(500);
        let messages: Vec<Message> = (0..40)
            .map(|i| Message::user(format!("{i}{multi_byte}")))
            .collect();
        let rendered = render_evicted(&messages, 300);
        let content = rendered
            .strip_suffix("[evicted 40 more messages]")
            .expect("the saturation marker closes the render");
        assert!(
            content.ends_with('…'),
            "the truncation point is the ellipsis: {content:?}"
        );
        let body = content.strip_suffix('…').expect("ellipsis present");
        assert_eq!(
            body.chars().count(),
            299,
            "the body fills the budget exactly"
        );
        assert!(
            body.ends_with('å'),
            "the cut lands between whole characters: {:?}",
            body.chars().rev().take(3).collect::<String>()
        );
    }

    #[tokio::test]
    async fn fnv_hash_tags_are_deterministic() {
        let a = fnv1a64(b"same render");
        let b = fnv1a64(b"same render");
        let c = fnv1a64(b"same rendex");
        assert_eq!(a, b, "identical renders hash identically");
        assert_ne!(a, c, "a one-byte difference changes the hash");

        let store = Arc::new(InMemoryStore::new());
        let sink = MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
        let evicted = vec![Message::user("deterministic content")];
        demote_once(&sink, &evicted).await;
        demote_once(&sink, &evicted).await;
        let entries = stored(&store).await;
        assert_eq!(entries.len(), 2, "a duplicate delivery stores twice");
        let hashes: Vec<&String> = entries
            .iter()
            .filter_map(|entry| {
                entry
                    .tags
                    .iter()
                    .find(|tag| tag.starts_with("demoted-hash:"))
            })
            .collect();
        assert_eq!(hashes.len(), 2, "each entry carries the hash tag");
        assert_eq!(
            hashes[0], hashes[1],
            "duplicate deliveries are detectable by the shared tag"
        );
    }

    #[tokio::test]
    async fn memory_sink_stores_rendered_trajectory() {
        let store = Arc::new(InMemoryStore::new());
        let sink = MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
        let meta = DemotionContext {
            reason: CompactReason::Emergency,
            turn: 3,
            session_id: uuid::Uuid::new_v4(),
        };
        let evicted = vec![Message::user("the launch code is ARC-7")];
        sink.demote(&evicted, meta)
            .await
            .expect("store accepts the entry");
        let entries = stored(&store).await;
        assert_eq!(entries.len(), 1, "one entry per pass");
        let entry = &entries[0];
        assert_eq!(entry.category, MemoryCategory::Trajectory);
        assert!(
            entry.memory.contains("the launch code is ARC-7"),
            "the rendered content carries the evicted marker text"
        );
        assert!(entry.tags.contains(&"demoted".to_string()));
        assert!(
            entry
                .tags
                .iter()
                .any(|tag| tag.starts_with("demoted-hash:")),
            "the content hash tag rides along: {:?}",
            entry.tags
        );
        assert!(
            entry.tags.contains(&format!("session:{}", meta.session_id)),
            "the session tag groups the entry: {:?}",
            entry.tags
        );
    }

    #[tokio::test]
    async fn noop_sink_demote_returns_ok() {
        let sink = NoopDemotionSink;
        let meta = DemotionContext {
            reason: CompactReason::Manual,
            turn: 0,
            session_id: uuid::Uuid::new_v4(),
        };
        sink.demote(&[Message::user("gone")], meta)
            .await
            .expect("the noop sink accepts every delivery");
    }

    #[tokio::test]
    async fn empty_delivery_stores_nothing() {
        let store = Arc::new(InMemoryStore::new());
        let sink = MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
        let meta = DemotionContext {
            reason: CompactReason::Manual,
            turn: 1,
            session_id: uuid::Uuid::new_v4(),
        };
        sink.demote(&[], meta).await.expect("empty is a no-op");
        assert_eq!(stored(&store).await.len(), 0, "nothing was demoted");
    }

    #[tokio::test]
    async fn an_explicit_budget_bounds_the_stored_body() {
        let store = Arc::new(InMemoryStore::new());
        let sink =
            MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>).with_max_chars(120);
        let evicted: Vec<Message> = (0..6)
            .map(|i| {
                Message::user(format!(
                    "message-{i} carries enough padding to fill the budget"
                ))
            })
            .collect();
        demote_once(&sink, &evicted).await;
        let entries = stored(&store).await;
        assert_eq!(entries.len(), 1, "one entry per delivery");
        let content = &entries[0].memory;
        let marker_start = content
            .find("[evicted ")
            .expect("the saturation marker rides past the budget");
        let body = &content[..marker_start];
        assert!(
            body.chars().count() <= 121,
            "the body plus its ellipsis stays inside the budget: {} chars",
            body.chars().count()
        );
    }

    #[tokio::test]
    async fn duplicate_demotions_fold_under_consolidation() {
        let store = Arc::new(InMemoryStore::new());
        let sink = MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>);
        let evicted = vec![
            Message::user("the same evicted slice"),
            Message::assistant("delivered twice by a retried pass"),
        ];
        demote_once(&sink, &evicted).await;
        demote_once(&sink, &evicted).await;
        assert_eq!(
            stored(&store).await.len(),
            2,
            "precondition: the redelivery stored twice, detectably"
        );
        store
            .consolidate()
            .await
            .expect("consolidation runs over the store");
        assert_eq!(
            stored(&store).await.len(),
            1,
            "near-duplicate merging folds the redelivery into one entry"
        );
    }

    #[tokio::test]
    async fn a_zero_budget_keeps_only_the_saturation_marker() {
        let store = Arc::new(InMemoryStore::new());
        let sink =
            MemoryDemotionSink::new(Arc::clone(&store) as Arc<dyn LoopMemory>).with_max_chars(0);
        let evicted = vec![
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ];
        demote_once(&sink, &evicted).await;
        let entries = stored(&store).await;
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].memory, "…[evicted 3 more messages]",
            "a zero budget keeps only the marker"
        );
    }
}
