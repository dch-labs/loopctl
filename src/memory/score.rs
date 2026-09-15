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

#[cfg(test)]
mod tests {
    use super::score_entry;
    use crate::memory::entry::{MemoryCategory, MemoryEntry};

    fn entry(text: &str, relevance: f32) -> MemoryEntry {
        let mut entry = MemoryEntry::new(MemoryCategory::Fact, text);
        entry.relevance = relevance;
        entry
    }

    #[test]
    fn an_unmatched_entry_scores_half_its_relevance_plus_the_baseline() {
        let (score, matched) = score_entry(&entry("unrelated text", 0.8), "deploy", &["deploy"]);
        assert!(
            (score - 0.5).abs() < 1e-6,
            "0.8 relevance with no match: {score}"
        );
        assert!(!matched, "no word overlap and no tag hit means unmatched");
    }

    #[test]
    fn word_overlap_scales_the_query_bonus() {
        let (score, matched) =
            score_entry(&entry("rust traits", 1.0), "rust async", &["rust", "async"]);
        assert!(
            (score - 0.8).abs() < 1e-6,
            "half of one matched word out of two: {score}"
        );
        assert!(matched);
    }

    #[test]
    fn a_tag_hit_adds_the_bonus_and_marks_the_entry_matched() {
        let mut tagged = entry("scripts live in ops", 0.9);
        tagged.tags.push("deploy".to_string());
        let (score, matched) = score_entry(&tagged, "deploy", &["deploy"]);
        assert!(
            (score - (0.9 * 0.5 + 0.3 + 0.1)).abs() < 1e-6,
            "relevance half plus tag bonus plus baseline: {score}"
        );
        assert!(matched, "a tag hit matches even without word overlap");
    }

    #[test]
    fn an_empty_query_ranks_every_entry_by_baseline_only() {
        let (score, matched) = score_entry(&entry("anything", 0.6), "", &[]);
        assert!(
            (score - 0.4).abs() < 1e-6,
            "0.6 relevance baseline: {score}"
        );
        assert!(!matched, "an empty query can never match");
    }

    #[test]
    fn out_of_range_relevance_scores_as_zero() {
        for poisoned in [1.5, -0.1, f32::NAN, f32::INFINITY] {
            let (score, _) = score_entry(&entry("rust traits", poisoned), "rust", &["rust"]);
            assert!(
                (score - (0.4 + 0.1)).abs() < 1e-6,
                "poisoned relevance {poisoned} contributes nothing: {score}"
            );
        }
    }
}
