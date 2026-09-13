//! Shared retrieval scoring for flat-list [`LoopMemory`](super::LoopMemory)
//! backends.
//!
//! One composite formula — `relevance × 0.5 + word-overlap × 0.4 + tag
//! bonus + baseline` — used by every store the crate ships, so
//! [`InMemoryStore`](super::builtin::InMemoryStore),
//! [`FileMemoryStore`](super::file::FileMemoryStore), and external
//! companions such as `loopctl-sqlite` rank identical entry sets
//! identically for the same query. A pure function: no locks, no I/O, no
//! allocation beyond the lowercased entry text.

use crate::memory::entry::MemoryEntry;

/// Score one entry against a query, returning its composite rank and
/// whether the query actually matched.
///
/// The composite score is `relevance × 0.5 + word-overlap × 0.4 + 0.3 tag
/// bonus + 0.1 baseline`, computed exactly as the reference
/// [`InMemoryStore`](super::builtin::InMemoryStore) always has. A
/// [`relevance`](MemoryEntry::relevance) outside `0.0..=1.0` — including
/// non-finite values — scores as zero, so a hand-poisoned entry cannot
/// disorder the ranking it is returned in. Callers pass the lowercased,
/// trimmed query string in `query_trimmed` (whole-query tag matching) and
/// its lowercased words in `query_words` (word-overlap matching); an
/// empty query yields the baseline ranking for every entry.
///
/// The returned `matched` flag is `true` only when a query word overlapped
/// the memory text or the whole query hit a tag. Access tracking keys on
/// it — baseline-only returns are delivered to the caller but never
/// stamped, so irrelevant queries cannot inflate
/// [`access_count`](MemoryEntry::access_count) or
/// [`last_accessed`](MemoryEntry::last_accessed) and shield entries from
/// decay.
#[must_use]
pub fn score_entry(entry: &MemoryEntry, query_trimmed: &str, query_words: &[&str]) -> (f32, bool) {
    let memory_lower = entry.memory.to_lowercase();
    let tag_match = !query_trimmed.is_empty()
        && entry
            .tags
            .iter()
            .any(|t| t.to_lowercase().contains(query_trimmed));
    let word_matches = query_words
        .iter()
        .filter(|w| memory_lower.contains(*w))
        .count();
    let base_score = if (0.0..=1.0).contains(&entry.relevance) {
        entry.relevance
    } else {
        0.0
    };
    let denom = query_words.len().max(1);
    let query_bonus = if word_matches > 0 {
        crate::numeric::unit_ratio(word_matches, denom)
    } else {
        0.0
    };
    let tag_bonus = if tag_match { 0.3 } else { 0.0 };
    let matched = word_matches > 0 || tag_match;
    (
        base_score * 0.5 + query_bonus * 0.4 + tag_bonus + 0.1,
        matched,
    )
}
