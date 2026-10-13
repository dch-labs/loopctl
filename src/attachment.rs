//! Compaction-immune attachments: data the agent pins so it survives any
//! compactor with zero loss.
//!
//! An attachment never enters the machine-owned conversation at all. It
//! renders into one consolidated [`User`](crate::message::Role::User)
//! message that rides every outbound request as per-turn transient
//! context — the same path context contributors and memory transients
//! ride — so no compactor (truncating, summarizing, or chained) can
//! even see it, let alone drop or distort it. Zero loss holds by
//! construction, not by a preservation pass running everywhere.
//!
//! Because the rendered parts re-ride each request fresh from the
//! [`AttachmentStore`], an attachment updated between turns reflects its
//! latest state on the next request, and a deterministic
//! [`render`](Attachment::render) keeps the request prefix byte-stable
//! across turns — the property provider-side prompt caching depends on.
//! The token cost of pinned content is counted in the per-turn payload
//! estimate like any transient: an attachment that pushes the payload
//! over the line defers the turn to compaction exactly like a
//! contributor. One larger than the provider will serve at all is
//! beyond any compactor's help — attachments never enter the history a
//! pass shrinks — so the outcome is provider-dependent: a permissive
//! provider keeps serving the deferred turns, while a rejecting
//! provider drives the turn through the salvage retry into the run's
//! [`ContextExceeded`](crate::error::LoopError::ContextExceeded) death,
//! the same death any oversized payload dies. `FileAttachment`'s
//! 8 `KiB` ceiling is the default guard against pinning the whole
//! window (P8: for a small-model host, pinning the todo list is the
//! cheap alternative to re-reading history — reason about the budget
//! with the render forms below).

use crate::message::{Message, MessagePart, Role};

/// The ceiling on a [`FileAttachment`]'s inline contents in its rendered
/// form, in bytes.
///
/// Bounds the window cost of a pinned file: contents past the ceiling
/// render truncated with a byte-count footer, so a host that pins a huge
/// file cannot silently pin the whole window. The ceiling is the default
/// — `truncate = false` renders verbatim at the caller's risk.
pub const FILE_CONTENTS_LIMIT_BYTES: usize = 8 * 1024;

/// What kind of attachment this is.
///
/// Drives the within-message ordering of
/// [`AttachmentStore::render_all`] (the most durable context leads and
/// the most volatile trails) and lets telemetry describe the pinned set
/// without inspecting content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AttachmentKind {
    /// A file's contents or path pinned by the user or agent.
    ///
    /// The most volatile of the durable kinds: the agent or host
    /// re-pins it as the session's file focus moves.
    File,

    /// A fact pulled from memory that must stay visible.
    ///
    /// Leads the consolidated message — the most durable context
    /// reads first.
    Memory,

    /// The live task list.
    ///
    /// Trails the consolidated message — the most volatile kind
    /// reads last.
    Todo,
}

/// The re-injection precedence of a kind: memories lead, files follow,
/// the live todo list trails.
///
/// [`render_all`](AttachmentStore::render_all) sorts by this rank with
/// a stable sort, so pins made earlier keep their lead within a kind.
fn kind_precedence(kind: AttachmentKind) -> u8 {
    match kind {
        AttachmentKind::Memory => 0,
        AttachmentKind::File => 1,
        AttachmentKind::Todo => 2,
    }
}

/// A value that renders into a [`MessagePart`] and is immune to
/// compaction.
///
/// Implementations must be cheap to clone (they live in an
/// [`AttachmentStore`] the engine reads at every request build) and
/// deterministic in their rendered form. The trait is object-safe for
/// use as `Arc<dyn Attachment>`.
///
/// # Contract
///
/// - [`render`](Attachment::render) must produce a stable part for a
///   given attachment state (same state ⇒ same part) — this is what
///   keeps the request prefix byte-stable and provider-side prompt
///   caching hitting.
/// - [`id`](Attachment::id) must be unique within a store; pinning an
///   attachment whose `id` matches one already stored replaces it in
///   place, which is the update path for a mutable attachment such as
///   the todo list.
/// - [`render`](Attachment::render) must return a
///   [`Text`](crate::message::MessagePart::Text) part. The
///   consolidated ride packs every part into one all-text user message
///   ahead of the prompt — the exact shape the request builder's
///   adjacent-user-text coalescing folds for providers that document
///   alternating roles — so a non-text part would ship as a separate
///   adjacent user message instead.
/// - Implementations must not mutate during `render` (it takes `&self`).
pub trait Attachment: Send + Sync {
    /// Stable unique identifier for deduplication in the store.
    ///
    /// Two attachments with equal `id`s are the same attachment; the
    /// store keeps the newest. Convention: `"{kind}:{slug}"` — e.g.
    /// `"file:src/main.rs"`, `"memory:auth-design"`, `"todo:session"`.
    fn id(&self) -> String;

    /// Which kind of attachment this is.
    ///
    /// Drives the within-message ordering of the consolidated render
    /// and lets hosts describe the pinned set without its content.
    fn kind(&self) -> AttachmentKind;

    /// Render into the [`MessagePart`] that represents this attachment
    /// on every request. Called fresh each turn, so it must be
    /// deterministic.
    fn render(&self) -> MessagePart;
}

/// A file pinned to the context so its contents or path survive
/// compaction.
///
/// Renders as a text part with a header naming the path, so the model
/// always knows which file this is:
///
/// ```text
/// [Attached file: src/main.rs]
/// <file contents, or a note that only the path is pinned>
/// ```
#[derive(Debug, Clone)]
pub struct FileAttachment {
    /// Path as given by the caller; need not exist on disk for a
    /// path-only pin.
    pub path: String,

    /// Optional inline contents. `None` pins the path only — the agent
    /// knows the file matters and can re-read it; the *path* is what
    /// must survive.
    pub contents: Option<String>,

    /// Whether to truncate contents past
    /// [`FILE_CONTENTS_LIMIT_BYTES`] in the rendered form. Default
    /// `true`; `false` renders verbatim at the caller's risk.
    pub truncate: bool,
}

impl FileAttachment {
    /// Pin a file's path without inlining its contents.
    ///
    /// The rendered form carries the header and a path-only note; the
    /// agent re-reads the file when it needs the contents.
    pub fn path_only(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            contents: None,
            truncate: true,
        }
    }

    /// Pin a file's path with its contents inlined.
    ///
    /// The rendered form carries the header and the contents, truncated
    /// past [`FILE_CONTENTS_LIMIT_BYTES`] with a byte-count footer
    /// unless `truncate` is turned off after construction.
    pub fn inline(path: impl Into<String>, contents: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            contents: Some(contents.into()),
            truncate: true,
        }
    }
}

impl Attachment for FileAttachment {
    fn id(&self) -> String {
        format!("file:{}", self.path)
    }

    fn kind(&self) -> AttachmentKind {
        AttachmentKind::File
    }

    fn render(&self) -> MessagePart {
        let mut text = format!("[Attached file: {}]", self.path);
        match &self.contents {
            None => {
                text.push_str(
                    "\n\n(path pinned; contents not inlined — \
                               read the file when its contents matter)",
                );
            }
            Some(contents) => {
                text.push_str("\n\n");
                if self.truncate && contents.len() > FILE_CONTENTS_LIMIT_BYTES {
                    text.push_str(&truncated_contents(contents));
                } else {
                    text.push_str(contents);
                }
            }
        }
        MessagePart::text(text)
    }
}

/// Render the inlined-contents body of a file over the truncation
/// ceiling.
///
/// Takes the largest `char`-boundary prefix at or under
/// [`FILE_CONTENTS_LIMIT_BYTES`] — a cut mid-codepoint would corrupt
/// the rendered text — and appends the footer naming the full byte
/// count, so the model knows what it is not seeing.
fn truncated_contents(contents: &str) -> String {
    format!(
        "{}\n\n… (truncated, {} bytes total)",
        &contents[..contents.floor_char_boundary(FILE_CONTENTS_LIMIT_BYTES)],
        contents.len()
    )
}

/// A memory-derived fact that must remain visible across compaction.
///
/// Typically produced from a memory-retrieval hit the agent deems
/// load-bearing, then pinned so a later compaction cannot summarise it
/// away. Renders as a text part:
///
/// ```text
/// [Pinned memory: auth-design]
/// Use Argon2id with the parameters from RFC 9106; do not use bcrypt.
/// ```
#[derive(Debug, Clone)]
pub struct MemoryAttachment {
    /// Slug identifying the memory; becomes part of the
    /// [`id`](Attachment::id).
    ///
    /// Two pins under one slug are one attachment — the newer replaces
    /// the older in place.
    pub slug: String,

    /// The memory text, verbatim.
    ///
    /// Rendered unchanged under the slug header; no summarisation ever
    /// touches it.
    pub memory: String,
}

impl MemoryAttachment {
    /// Pin a memory fact under its slug.
    ///
    /// The memory text rides every request verbatim under the slug
    /// header.
    pub fn new(slug: impl Into<String>, memory: impl Into<String>) -> Self {
        Self {
            slug: slug.into(),
            memory: memory.into(),
        }
    }
}

impl Attachment for MemoryAttachment {
    fn id(&self) -> String {
        format!("memory:{}", self.slug)
    }

    fn kind(&self) -> AttachmentKind {
        AttachmentKind::Memory
    }

    fn render(&self) -> MessagePart {
        MessagePart::text(format!("[Pinned memory: {}]\n\n{}", self.slug, self.memory))
    }
}

/// The live task list.
///
/// Renders as a text part with a checklist so the model re-enters the
/// session seeing exactly what is done and what is pending:
///
/// ```text
/// [Todo list]
/// - [x] read compact.rs
/// - [ ] wire the attachments system
/// ```
#[derive(Debug, Clone)]
pub struct TodoAttachment {
    /// Ordered items; the `completed` flags render as `[x]` / `[ ]`.
    ///
    /// The order is the render order — the model reads the list top
    /// to bottom.
    pub items: Vec<TodoItem>,
}

impl TodoAttachment {
    /// Pin the task list with its items in order.
    ///
    /// One live list per session: the fixed `todo:session` id means a
    /// later pin replaces this one in place.
    #[must_use]
    pub fn new(items: Vec<TodoItem>) -> Self {
        Self { items }
    }
}

impl Attachment for TodoAttachment {
    fn id(&self) -> String {
        "todo:session".to_string()
    }

    fn kind(&self) -> AttachmentKind {
        AttachmentKind::Todo
    }

    fn render(&self) -> MessagePart {
        let mut text = String::from("[Todo list]");
        for item in &self.items {
            text.push_str(if item.completed {
                "\n- [x] "
            } else {
                "\n- [ ] "
            });
            text.push_str(&item.text);
        }
        MessagePart::text(text)
    }
}

/// One entry in a [`TodoAttachment`].
///
/// Pure data: the text and its done flag, rendered in list order.
#[derive(Debug, Clone)]
pub struct TodoItem {
    /// The item's text as shown in the checklist.
    ///
    /// Rendered verbatim after the checkbox marker.
    pub text: String,

    /// Whether the item is done; renders as `[x]` versus `[ ]`.
    ///
    /// The one field a mid-session update flips.
    pub completed: bool,
}

impl TodoItem {
    /// One checklist entry.
    ///
    /// Shorthand for the struct literal with the same field order.
    pub fn new(text: impl Into<String>, completed: bool) -> Self {
        Self {
            text: text.into(),
            completed,
        }
    }
}

/// Holds the set of active attachments and renders them for the
/// per-turn request.
///
/// Owned by the engine alongside the machine-owned conversation;
/// insertion order is preserved so the rendered form is deterministic.
#[derive(Clone, Default)]
pub struct AttachmentStore {
    /// Insertion-ordered attachments, deduped by
    /// [`id`](Attachment::id).
    attachments: Vec<std::sync::Arc<dyn Attachment>>,
}

impl AttachmentStore {
    /// An empty store.
    ///
    /// A fresh loop starts here; an empty store builds no message, so
    /// requests stay byte-identical to an attachments-unaware loop.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of pinned attachments.
    ///
    /// After dedupe by [`id`](Attachment::id) — a re-pin does not
    /// grow the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.attachments.len()
    }

    /// Whether no attachments are pinned.
    ///
    /// The consolidated message is built only when this is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attachments.is_empty()
    }

    /// Pin an attachment, replacing the one already stored under the
    /// same [`id`](Attachment::id) in place.
    ///
    /// Replacement preserves the original insertion position, so the
    /// rendered order of an updated attachment does not move — the
    /// update path for a mutable attachment such as the todo list.
    pub fn insert(&mut self, attachment: std::sync::Arc<dyn Attachment>) {
        let id = attachment.id();
        if let Some(slot) = self.attachments.iter_mut().find(|stored| stored.id() == id) {
            *slot = attachment;
        } else {
            self.attachments.push(attachment);
        }
    }

    /// Unpin the attachment with this [`id`](Attachment::id).
    ///
    /// Returns whether an attachment was removed; an unknown `id` is a
    /// no-op returning `false`.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.attachments.len();
        self.attachments.retain(|stored| stored.id() != id);
        before != self.attachments.len()
    }

    /// Whether an attachment with this [`id`](Attachment::id) is
    /// pinned.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.attachments.iter().any(|stored| stored.id() == id)
    }

    /// The pinned attachments in insertion order.
    ///
    /// The raw store view; [`render_all`](Self::render_all) is the
    /// ordered render.
    #[must_use]
    pub fn attachments(&self) -> &[std::sync::Arc<dyn Attachment>] {
        &self.attachments
    }

    /// Render every attachment to its [`MessagePart`], most durable
    /// kind first.
    ///
    /// Ordering is kind precedence — [`Memory`](AttachmentKind::Memory),
    /// then [`File`](AttachmentKind::File), then
    /// [`Todo`](AttachmentKind::Todo) — with insertion order preserved
    /// within a kind, so the pinned context the model reads first is
    /// the least volatile.
    #[must_use]
    pub fn render_all(&self) -> Vec<MessagePart> {
        let mut ordered = self.attachments.clone();
        ordered.sort_by_key(|stored| kind_precedence(stored.kind()));
        ordered.iter().map(|stored| stored.render()).collect()
    }

    /// The consolidated message every outbound request carries ahead of
    /// the conversation, or `None` when nothing is pinned.
    ///
    /// One [`User`](Role::User) message holding every attachment's
    /// rendered part in [`render_all`](Self::render_all) order; a loop
    /// with no attachments builds no message at all, so its requests
    /// stay byte-identical to a loop that has never heard of
    /// attachments.
    #[must_use]
    pub fn consolidated_message(&self) -> Option<Message> {
        if self.attachments.is_empty() {
            return None;
        }
        Some(Message::new(Role::User, self.render_all()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn path_only_file_renders_the_header_and_a_note() {
        let part = FileAttachment::path_only("src/main.rs").render();
        let text = match &part {
            MessagePart::Text { text } => text.as_str(),
            _ => panic!("a file attachment renders a text part"),
        };
        assert!(
            text.starts_with("[Attached file: src/main.rs]\n\n"),
            "the header names the path: {text}"
        );
        assert!(
            text.contains("path pinned"),
            "the path-only body states that only the path is pinned: {text}"
        );
    }

    #[test]
    fn inline_file_contents_render_verbatim_under_the_ceiling() {
        let attachment = FileAttachment::inline("notes.txt", "keep this");
        let text = rendered_text(&attachment);
        assert_eq!(
            text, "[Attached file: notes.txt]\n\nkeep this",
            "inline contents render header plus body verbatim"
        );
    }

    #[test]
    fn oversized_contents_truncate_at_the_ceiling_with_a_byte_footer() {
        let body = "a".repeat(FILE_CONTENTS_LIMIT_BYTES + 40);
        let attachment = FileAttachment::inline("big.log", body);
        let text = rendered_text(&attachment);
        assert!(
            text.ends_with(&format!(
                "\n\n… (truncated, {} bytes total)",
                FILE_CONTENTS_LIMIT_BYTES + 40
            )),
            "the footer names the full byte count: …{}",
            text.chars().rev().take(60).collect::<String>()
        );
        let rendered_body = text
            .strip_prefix("[Attached file: big.log]\n\n")
            .unwrap_or_default();
        assert_eq!(
            rendered_body.len(),
            FILE_CONTENTS_LIMIT_BYTES + "\n\n… (truncated, 8232 bytes total)".len(),
            "the body is the ceiling-sized cut plus the footer — nothing more"
        );
    }

    #[test]
    fn a_truncation_cut_lands_on_a_char_boundary() {
        let mut body = String::from("x");
        body.push_str(&"é".repeat(4_100));
        let attachment = FileAttachment::inline("accents.txt", body);
        let text = rendered_text(&attachment);
        let cut = text
            .strip_prefix("[Attached file: accents.txt]\n\n")
            .unwrap_or_default();
        let footer_at = cut.find("\n\n… (truncated,").unwrap_or_default();
        let carried = &cut[..footer_at];
        assert_eq!(
            carried.len(),
            FILE_CONTENTS_LIMIT_BYTES - 1,
            "byte {} falls inside an é, so the largest boundary at or under \
             the ceiling is {} — the cut length itself is the oracle",
            FILE_CONTENTS_LIMIT_BYTES,
            FILE_CONTENTS_LIMIT_BYTES - 1
        );
        assert!(
            cut.is_char_boundary(footer_at)
                && cut.ends_with(&format!("\n\n… (truncated, {} bytes total)", 1 + 2 * 4_100)),
            "the footer names the full byte count: {cut}"
        );
    }

    #[test]
    fn contents_exactly_at_the_ceiling_render_verbatim() {
        let body = "y".repeat(FILE_CONTENTS_LIMIT_BYTES);
        let attachment = FileAttachment::inline("exact.log", body.clone());
        let text = rendered_text(&attachment);
        assert_eq!(
            text,
            format!("[Attached file: exact.log]\n\n{body}"),
            "a body exactly at the ceiling is not truncated"
        );
    }

    #[test]
    fn truncate_off_renders_verbatim_over_the_ceiling() {
        let body = "z".repeat(FILE_CONTENTS_LIMIT_BYTES + 40);
        let attachment = FileAttachment {
            truncate: false,
            ..FileAttachment::inline("raw.log", body.clone())
        };
        let text = rendered_text(&attachment);
        assert_eq!(
            text,
            format!("[Attached file: raw.log]\n\n{body}"),
            "truncate = false renders the whole body at the caller's risk"
        );
    }

    #[test]
    fn memory_renders_its_slug_header_and_text_verbatim() {
        let attachment = MemoryAttachment::new("auth-design", "Use Argon2id; do not use bcrypt.");
        let text = rendered_text(&attachment);
        assert_eq!(
            text, "[Pinned memory: auth-design]\n\nUse Argon2id; do not use bcrypt.",
            "the memory body rides verbatim under the slug header"
        );
    }

    #[test]
    fn todo_items_render_as_an_ordered_checklist() {
        let attachment = TodoAttachment::new(vec![
            TodoItem::new("read compact.rs", true),
            TodoItem::new("wire the attachments system", false),
        ]);
        let text = rendered_text(&attachment);
        assert_eq!(
            text, "[Todo list]\n- [x] read compact.rs\n- [ ] wire the attachments system",
            "completed and pending items render in order"
        );
    }

    #[test]
    fn an_empty_todo_list_renders_its_header_alone() {
        let attachment = TodoAttachment::new(Vec::new());
        let text = rendered_text(&attachment);
        assert_eq!(text, "[Todo list]", "the header survives an empty list");
    }

    #[test]
    fn ids_follow_the_kind_colon_slug_convention() {
        assert_eq!(
            FileAttachment::path_only("src/main.rs").id(),
            "file:src/main.rs",
            "the file id names its path"
        );
        assert_eq!(
            MemoryAttachment::new("auth-design", "").id(),
            "memory:auth-design",
            "the memory id names its slug"
        );
        assert_eq!(
            TodoAttachment::new(Vec::new()).id(),
            "todo:session",
            "one live todo list per session"
        );
        assert_ne!(
            FileAttachment::path_only("a.rs").id(),
            FileAttachment::path_only("b.rs").id(),
            "different paths are different attachments"
        );
    }

    #[test]
    fn render_is_deterministic_for_an_unchanged_attachment() {
        let attachment = FileAttachment::inline("stable.txt", "stable body");
        assert_eq!(
            rendered_text(&attachment),
            rendered_text(&FileAttachment::inline("stable.txt", "stable body")),
            "same state renders the same part — the prefix-cache invariant"
        );
        let todo = TodoAttachment::new(vec![TodoItem::new("one", true)]);
        assert_eq!(
            rendered_text(&todo),
            rendered_text(&TodoAttachment::new(vec![TodoItem::new("one", true)])),
            "same state renders the same part — the prefix-cache invariant"
        );
    }

    #[test]
    fn insert_replaces_in_place_preserving_order() {
        let mut store = AttachmentStore::new();
        let first = MemoryAttachment::new("first", "one");
        let second = MemoryAttachment::new("second", "two");
        store.insert(Arc::new(first));
        store.insert(Arc::new(second));
        store.insert(Arc::new(MemoryAttachment::new("first", "updated")));
        assert_eq!(store.len(), 2, "a re-pin under the same id replaces");
        assert_eq!(
            rendered_text(store.attachments()[0].as_ref()),
            rendered_text(&MemoryAttachment::new("first", "updated")),
            "the replacement keeps the original position"
        );
        assert_eq!(
            rendered_text(store.attachments()[1].as_ref()),
            rendered_text(&MemoryAttachment::new("second", "two")),
            "the untouched attachment keeps its position"
        );
    }

    #[test]
    fn remove_flips_contains_and_unknown_ids_are_no_ops() {
        let mut store = AttachmentStore::new();
        store.insert(Arc::new(FileAttachment::path_only("gone.rs")));
        assert!(store.contains("file:gone.rs"), "the pin is stored");
        assert!(
            store.remove("file:gone.rs"),
            "removing a stored id reports true"
        );
        assert!(!store.contains("file:gone.rs"), "the pin is gone");
        assert!(
            !store.remove("file:never-was.rs"),
            "removing an unknown id reports false"
        );
    }

    #[test]
    fn render_all_orders_by_kind_precedence_then_insertion() {
        let mut store = AttachmentStore::new();
        store.insert(Arc::new(TodoAttachment::new(Vec::new())));
        store.insert(Arc::new(FileAttachment::path_only("b.rs")));
        store.insert(Arc::new(MemoryAttachment::new("fact", "keep")));
        store.insert(Arc::new(FileAttachment::path_only("a.rs")));
        let parts = store.render_all();
        let texts = parts
            .iter()
            .map(|part| match part {
                MessagePart::Text { text } => text.as_str(),
                _ => panic!("attachments render text parts"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            vec![
                "[Pinned memory: fact]\n\nkeep",
                "[Attached file: b.rs]\n\n(path pinned; contents not inlined — read the file when its contents matter)",
                "[Attached file: a.rs]\n\n(path pinned; contents not inlined — read the file when its contents matter)",
                "[Todo list]",
            ],
            "memory leads, files follow in pin order, the todo list trails"
        );
    }

    #[test]
    fn every_builtin_renders_a_text_part() {
        let parts = [
            FileAttachment::path_only("contract.rs").render(),
            MemoryAttachment::new("contract", "keep").render(),
            TodoAttachment::new(vec![TodoItem::new("pin the contract", false)]).render(),
        ];
        assert!(
            parts
                .iter()
                .all(|part| matches!(part, MessagePart::Text { .. })),
            "the consolidated ride is one all-text user message — the render contract"
        );
    }

    #[test]
    fn the_consolidated_message_is_none_for_an_empty_store() {
        let store = AttachmentStore::new();
        assert!(
            store.consolidated_message().is_none(),
            "an empty store builds no message — requests stay byte-identical"
        );
    }

    fn rendered_text(attachment: &dyn Attachment) -> String {
        match attachment.render() {
            MessagePart::Text { text } => text,
            _ => panic!("attachments render text parts"),
        }
    }
}
