//! Reusable memory-consolidation primitives.
//!
//! Strategy-agnostic building blocks any [`LoopMemory`] implementation
//! composes to give its [`consolidate`](crate::memory::LoopMemory::consolidate) method real
//! behaviour: composite quality scoring, category-weighted exponential
//! decay, near-duplicate clustering with corroboration merges, and a
//! one-call [`consolidate_entries`] pass tying them together. The in-memory
//! store uses them today; file- and SQLite-backed stores reuse the same
//! primitives under their own write locks.
//!
//! [`LoopMemory`]: crate::memory::LoopMemory
//! [`LoopMemory::consolidate`]: crate::memory::LoopMemory::consolidate

use std::collections::HashSet;
use std::time::{Duration, SystemTime};

use crate::memory::entry::{ConsolidationStats, MemoryCategory, MemoryEntry};

/// Weight of the entry's own relevance in [`quality_score`].
///
/// Relevance dominates the composite because it carries both the
/// extractor's confidence and the accumulated decay signal — the two
/// inputs consolidation most needs to respect. The weight keeps its
/// contribution just under half of the total score.
const RELEVANCE_WEIGHT: f32 = 0.45;

/// Weight of the (log-scaled) access count in [`quality_score`].
///
/// The access factor is log-scaled before weighting, so the first few
/// retrievals move the score much more than the fiftieth. The weight
/// rewards memories that prove useful repeatedly without letting a
/// single hot entry dominate on popularity alone.
const ACCESS_WEIGHT: f32 = 0.30;

/// Weight of the recency factor in [`quality_score`].
///
/// The recency factor decays exponentially from the entry's last access
/// or creation. The modest weight keeps fresh-but-irrelevant memories
/// from outranking proven ones merely for being new.
const RECENCY_WEIGHT: f32 = 0.15;

/// Weight of the validated flag in [`quality_score`].
///
/// Validation is a fixed trust bonus, applied as a full-weight step
/// rather than a gradient. It is set explicitly through the builder and
/// by extraction when a mined memory's confidence clears the
/// auto-validate threshold.
const VALIDATED_WEIGHT: f32 = 0.10;

/// Access count at which the log-scaled access factor saturates at 1.0.
///
/// The point sits far above realistic per-retrieval counts for a single
/// entry. Merges sum whole clusters' access counts, so a merged hot
/// entry can pass the saturation point — the clamp then pins its access
/// factor at the ceiling rather than distorting the score. Raising the
/// constant flattens the access curve further; lowering it makes
/// popularity matter sooner.
const ACCESS_SATURATION: f32 = 50.0;

/// Recency time scale for the composite score, in seconds.
///
/// An entry accessed now scores 1.0 on the recency factor; one left
/// untouched for a month approaches zero.
const RECENCY_SCALE_SECS: f32 = 60.0 * 60.0 * 24.0 * 30.0;

/// The composite quality of a memory entry, independent of any query.
///
/// Used by consolidation to decide what survives: a memory that is
/// high-relevance but never accessed and very old should not outrank a
/// mid-relevance memory that is retrieved often. Composed as
/// `relevance × 0.45 + access × 0.30 + recency × 0.15 + validated × 0.10`,
/// with every component normalized to `0.0..=1.0` so the result stays in
/// range — relevance is clamped into the range, so a finite
/// hand-constructed or deserialized out-of-range value cannot push the
/// composite past `1.0` (a NaN input propagates, staying visibly broken
/// for the sanitizing pass — the same policy `merge_cluster`'s cap
/// follows).
#[must_use]
pub fn quality_score(entry: &MemoryEntry, now: SystemTime) -> f32 {
    let access_count = f32::from(u16::try_from(entry.access_count).unwrap_or(u16::MAX));
    let access_factor =
        ((1.0 + access_count).ln() / (1.0 + ACCESS_SATURATION).ln()).clamp(0.0, 1.0);
    let recency_factor = now
        .duration_since(entry.last_accessed.unwrap_or(entry.created_at))
        .ok()
        .map_or(1.0, |age| (-age.as_secs_f32() / RECENCY_SCALE_SECS).exp());
    let validated_bonus = if entry.validated { 1.0 } else { 0.0 };
    entry.relevance.clamp(0.0, 1.0) * RELEVANCE_WEIGHT
        + access_factor * ACCESS_WEIGHT
        + recency_factor * RECENCY_WEIGHT
        + validated_bonus * VALIDATED_WEIGHT
}

/// Per-category decay multipliers applied to elapsed half-lives.
///
/// Facts and Strategies are durable knowledge and decay at half rate;
/// Working memory is ephemeral and decays at double rate; `ErrorPattern`s
/// and Insights sit in between — the fix itself is durable, but how often
/// it matters fades. Trajectories decay at the neutral rate.
#[must_use]
pub fn category_decay_weight(category: MemoryCategory) -> f32 {
    match category {
        MemoryCategory::Fact | MemoryCategory::Strategy => 0.5,
        MemoryCategory::Insight | MemoryCategory::ErrorPattern => 0.75,
        MemoryCategory::Trajectory => 1.0,
        MemoryCategory::Working => 2.0,
    }
}

/// Exponential time-based decay of one entry's relevance.
///
/// Multiplies `entry.relevance` by `0.5 ^ elapsed_half_lives`, where the
/// elapsed time is measured from the later of the entry's last decay pass
/// and its last access (recently retrieved memories resist decay), or from
/// its creation when neither is set, scaled by [`category_decay_weight`].
/// An entry untouched for *n* half-lives loses roughly `1 − 0.5ⁿ` of its
/// relevance.
///
/// The pass is **idempotent over cadence**: each call decays only the time
/// since the previous call and then advances the entry's
/// [`last_decayed`](MemoryEntry::last_decayed) stamp, so consolidating
/// hourly, daily, or once produces the same relevance as a single pass
/// over the same total age. The divisor floors the half-life at one
/// second: a direct call with a sub-second or zero `half_life` decays
/// per second rather than erroring — the [`ConsolidationConfig`] path substitutes its documented default instead. Retrieval between passes pauses decay — the
/// stamp jumps to the access time, which shields the interval before it.
/// The stamp never moves backwards: a pass whose clock sits before an
/// earlier pass — an NTP step, a resumed VM — leaves the stamp alone, so
/// time that has already been decayed is never decayed a second time.
pub fn decay_relevance(entry: &mut MemoryEntry, now: SystemTime, half_life: Duration) {
    let effective_stamp = entry
        .last_decayed
        .unwrap_or(entry.created_at)
        .max(entry.last_accessed.unwrap_or(entry.created_at));
    let advance = |entry: &mut MemoryEntry| {
        entry.last_decayed = Some(entry.last_decayed.map_or(now, |prev| prev.max(now)));
    };
    let Some(age) = now.duration_since(effective_stamp).ok() else {
        advance(entry);
        return;
    };
    let half_lives = age.as_secs_f32() / half_life.as_secs_f32().max(1.0)
        * category_decay_weight(entry.category);
    let factor = 0.5_f32.powf(half_lives.max(0.0));
    entry.relevance = (entry.relevance * factor).clamp(0.0, 1.0);
    advance(entry);
}

/// A group of entries deemed near-duplicates of one another.
///
/// Built by [`cluster_duplicates`]. The canonical member is the highest
/// [`quality_score`] entry of the group; the rest merge into it via
/// [`merge_cluster`], which keeps the canonical's identity and text while
/// absorbing corroboration.
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    /// The entry the cluster merges into — its `id` and text are kept.
    ///
    /// Merging never rewrites the canonical's text, so the store keeps one
    /// stable identity per lesson. [`consolidate_entries`] promotes the
    /// highest-`quality_score` member into this slot before merging, in case
    /// clustering seeded the group with a weaker first-seen entry.
    pub canonical: MemoryEntry,

    /// Entries judged near-duplicates of the canonical member.
    ///
    /// Each duplicate corroborates the canonical on merge: its tags union
    /// in, its access count adds, and its presence raises the canonical's
    /// relevance by a fixed step. The duplicate texts themselves are not
    /// concatenated — merged entries stay concise.
    pub duplicates: Vec<MemoryEntry>,
}

/// Partition a slice of entries into duplicate clusters.
///
/// Two entries cluster when they share a [`MemoryCategory`] and the Jaccard
/// similarity of their normalized token sets reaches `similarity_threshold`
/// against the cluster's canonical member. The threshold defends itself
/// like the pass does: a value that is non-finite or outside
/// `(0.0..=1.0]` falls back to the default 0.6 — a threshold of exactly
/// zero would satisfy `jaccard >= threshold` for every same-category
/// pair, disjoint sets included, and collapse the category into one
/// entry. An entry whose normalized token set is empty — empty,
/// punctuation-only, or emoji-only text — never clusters: corroboration
/// requires shared tokens, so degenerate entries prune on their own
/// relevance instead of promoting each other. Greedy
/// and `O(n²)` in the worst case — acceptable because consolidation is
/// periodic, not per-turn. Every input entry lands in exactly one
/// cluster; singletons come back with an empty `duplicates` vector so
/// callers treat both shapes alike.
#[must_use]
pub fn cluster_duplicates(
    entries: &[MemoryEntry],
    similarity_threshold: f32,
) -> Vec<MemoryCluster> {
    let similarity_threshold = if similarity_threshold.is_finite()
        && (0.0..=1.0).contains(&similarity_threshold)
        && similarity_threshold > 0.0
    {
        similarity_threshold
    } else {
        ConsolidationConfig::default().merge_threshold
    };
    let mut clusters: Vec<(MemoryCluster, HashSet<String>)> = Vec::new();
    for entry in entries {
        let tokens = normalized_tokens(&entry.memory);
        let mut placed = false;
        if !tokens.is_empty() {
            for (cluster, canonical_tokens) in &mut clusters {
                let same_category = std::mem::discriminant(&cluster.canonical.category)
                    == std::mem::discriminant(&entry.category);
                if same_category && jaccard(&tokens, canonical_tokens) >= similarity_threshold {
                    cluster.duplicates.push(entry.clone());
                    placed = true;
                    break;
                }
            }
        }
        if !placed {
            clusters.push((
                MemoryCluster {
                    canonical: entry.clone(),
                    duplicates: Vec::new(),
                },
                tokens,
            ));
        }
    }
    clusters.into_iter().map(|(cluster, _)| cluster).collect()
}

/// Merge a cluster into one [`MemoryEntry`].
///
/// Keeps the canonical entry's `id` and memory text; unions `tags`
/// (deduplicated); raises `relevance` by `0.1` per duplicate as
/// corroboration, clamped into `0.0..=1.0` (a NaN stays NaN — visibly
/// broken for the next sanitizing pass — instead of `f32::min`'s
/// ignore-NaN turning it into exactly `1.0`); sums `access_count`; sets
/// `validated` when any member was validated; keeps the earliest
/// `created_at` (the cluster's provenance) and the freshest
/// `last_accessed`. Adopting the earlier provenance stamps
/// `last_decayed` with the canonical's pre-merge decay baseline (its
/// own `last_decayed`, else its original `created_at`), so the first
/// post-merge pass decays only the window the canonical's relevance
/// actually lived through — the rewind cannot charge it the duplicate's
/// age.
#[must_use]
pub fn merge_cluster(cluster: MemoryCluster) -> MemoryEntry {
    let mut merged = cluster.canonical;
    let canonical_created_at = merged.created_at;
    let earliest = cluster
        .duplicates
        .iter()
        .fold(merged.created_at, |acc, dup| acc.min(dup.created_at));
    merged.created_at = earliest;
    merged.last_decayed = Some(merged.last_decayed.unwrap_or(canonical_created_at));
    let latest = cluster
        .duplicates
        .iter()
        .filter_map(|dup| dup.last_accessed)
        .fold(merged.last_accessed, |acc, stamp| acc.max(Some(stamp)));
    merged.last_accessed = latest;
    for dup in &cluster.duplicates {
        merged.relevance = (merged.relevance + 0.1).clamp(0.0, 1.0);
        merged.access_count = merged.access_count.saturating_add(dup.access_count);
        merged.validated = merged.validated || dup.validated;
        for tag in &dup.tags {
            if !merged.tags.contains(tag) {
                merged.tags.push(tag.clone());
            }
        }
    }
    merged
}

/// Which consolidation behaviours to run, and with which parameters.
///
/// The struct is publicly constructible and `Clone`, so
/// [`consolidate_entries`] never trusts it: thresholds are normalized once
/// per pass — values outside `0.0..=1.0` or non-finite fall back to the
/// documented defaults — so a misconfigured store degrades to
/// conservative behaviour instead of merging every same-category pair or
/// pruning everything.
///
/// The default runs the full pass — decay with a 14-day half-life, merge at
/// a 0.6 similarity threshold, and a 0.05 quality floor matching the
/// store's historical relevance floor — so out-of-the-box behaviour is
/// comparable to the plain pruner it replaces, only smarter.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Run time-based decay first.
    ///
    /// Dedup and pruning then see post-decay relevance, so a pass
    /// judges entries by the value they have now.
    pub decay: bool,

    /// Half-life for decay. Default: 14 days.
    ///
    /// One half of an entry's relevance is lost for each half-life of
    /// unaccessed age, scaled by [`category_decay_weight`]. Fourteen days
    /// lets a store consolidated daily keep working memories for weeks while
    /// genuinely stale knowledge fades within a couple of months.
    ///
    /// Normalized per pass: a zero or sub-second half-life falls back to
    /// the default instead of decaying everything to dust in a second.
    pub half_life: Duration,

    /// Run near-duplicate detection and corroboration merges.
    ///
    /// Merging runs after decay, so clusters form on post-decay relevance
    /// and the survivors enter pruning already deduplicated. Disabling it
    /// leaves a pass that only decays and prunes — the right shape for hosts
    /// that curate duplicates themselves.
    pub merge: bool,

    /// Jaccard similarity threshold for duplicate clustering. Default: 0.6.
    ///
    /// Two entries cluster when the Jaccard similarity of their normalized
    /// token sets reaches this value within one category. Lower values merge
    /// more aggressively; 0.6 folds re-worded repeats while keeping genuine
    /// paraphrases — which carry real information differences — separate.
    ///
    /// Normalized per pass: values outside `(0.0..=1.0]` — including zero,
    /// since disjoint sets score exactly `0.0` and would then merge every
    /// same-category entry — or non-finite, fall back to the default.
    pub merge_threshold: f32,

    /// Prune entries whose post-decay relevance **or** composite
    /// [`quality_score`] falls below this floor. Default: 0.05.
    ///
    /// The relevance half preserves the historical pruning contract — an
    /// entry stored below the floor is dropped on the first pass however
    /// fresh it is — while the quality half ages out stale, never-accessed
    /// knowledge that decay has hollowed out. Normalized per pass:
    /// values outside `0.0..=1.0`, or non-finite, fall back to the
    /// default.
    pub prune_floor: f32,
}

impl Default for ConsolidationConfig {
    /// Returns the full-pass defaults: decay on a 14-day half-life,
    /// merge at a 0.6 similarity threshold, prune at a 0.05 floor.
    ///
    /// These match what the in-memory store's historical plain pruner
    /// produced, so adopting the full pass changes nothing for a store
    /// that never tuned it.
    fn default() -> Self {
        Self {
            decay: true,
            half_life: Duration::from_hours(336),
            merge: true,
            merge_threshold: 0.6,
            prune_floor: 0.05,
        }
    }
}

/// Run a full consolidation pass over a batch of entries.
///
/// The sequence — decay, cluster, merge, prune — mutates `entries` in place
/// and returns the resulting [`ConsolidationStats`]. `bytes_saved`
/// estimates the reclaimed text as the summed `memory` length of every
/// pruned and merged-away entry. This is the pass a store runs under its
/// write lock; it takes no locks of its own.
///
/// An entry whose relevance is not finite or sits outside `0.0..=1.0` —
/// reachable only through hand-constructed values, since every internal
/// path clamps — is degraded to zero and **excluded from clustering**,
/// so it cannot corroborate a survivor into unprunable relevance or
/// donate a garbage `validated`/`access_count`; with any prune floor
/// above zero (the default is 0.05) it then prunes instead of poisoning
/// decay, clustering, and merges. The exclusion is a property of the
/// pass that sees the garbage: a zero prune floor — valid, since nothing
/// sits below it — retains the degraded entry at relevance `0.0`, where
/// the next pass is entitled to treat it like any legitimately decayed
/// entry. Hosts that must expel poisoned input unconditionally should
/// keep the floor above zero.
pub fn consolidate_entries(
    entries: &mut Vec<MemoryEntry>,
    config: &ConsolidationConfig,
    now: SystemTime,
) -> ConsolidationStats {
    let config = config.normalized();
    let entries_before = entries.len();
    let mut degraded = Vec::new();
    let mut healthy = Vec::new();
    for entry in std::mem::take(entries) {
        if (0.0..=1.0).contains(&entry.relevance) {
            healthy.push(entry);
        } else {
            degraded.push(entry);
        }
    }
    for entry in &mut degraded {
        entry.relevance = 0.0;
    }
    if config.decay {
        for entry in &mut healthy {
            decay_relevance(entry, now, config.half_life);
        }
    }
    let mut merged = 0usize;
    let mut merged_away_text = 0usize;
    if config.merge {
        let clusters = cluster_duplicates(&healthy, config.merge_threshold);
        let mut consolidated = Vec::with_capacity(clusters.len());
        for mut cluster in clusters {
            promote_highest_quality(&mut cluster, now);
            merged_away_text = merged_away_text.saturating_add(
                cluster
                    .duplicates
                    .iter()
                    .map(|dup| dup.memory.len())
                    .sum::<usize>(),
            );
            merged = merged.saturating_add(cluster.duplicates.len());
            consolidated.push(merge_cluster(cluster));
        }
        healthy = consolidated;
    }
    healthy.append(&mut degraded);
    *entries = healthy;
    let before_prune = entries.len();
    let prunable = |entry: &MemoryEntry| {
        entry.relevance < config.prune_floor || quality_score(entry, now) < config.prune_floor
    };
    let mut kept: Vec<MemoryEntry> = Vec::with_capacity(entries.len());
    let mut pruned_text = 0usize;
    for entry in std::mem::take(entries) {
        if prunable(&entry) {
            pruned_text = pruned_text.saturating_add(entry.memory.len());
        } else {
            kept.push(entry);
        }
    }
    let pruned = before_prune.saturating_sub(kept.len());
    *entries = kept;
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.memory.consolidated",
        removed = pruned,
        kept = entries.len(),
        merged
    );
    ConsolidationStats {
        entries_before,
        entries_after: entries.len(),
        pruned,
        merged,
        bytes_saved: pruned_text.saturating_add(merged_away_text),
    }
}

impl ConsolidationConfig {
    /// Return a pass-safe copy: thresholds outside their valid range — or
    /// non-finite — are replaced by the documented defaults.
    ///
    /// The config is `Clone` and publicly constructible, so validation at
    /// construction cannot hold. Falling back to the defaults (rather than
    /// clamping) is deliberate: a clamped-to-zero merge threshold would
    /// merge every same-category entry, and a clamped-to-one prune floor
    /// would prune every entry — the exact destruction the normalization
    /// exists to prevent. The default fallback can only narrow behaviour.
    ///
    /// `merge_threshold` is additionally valid only *above* zero: disjoint
    /// token sets have Jaccard exactly `0.0`, so a threshold of `0.0`
    /// satisfies `jaccard >= threshold` for every same-category pair and
    /// collapses the whole category into one entry — in range and finite,
    /// but destructive all the same. `prune_floor` keeps `0.0` (nothing
    /// sits below a zero floor).
    fn normalized(&self) -> Self {
        let normalize = |value: f32, default: f32, exclusive_zero: bool| {
            let in_range = value.is_finite()
                && if exclusive_zero {
                    (0.0..=1.0).contains(&value) && value > 0.0
                } else {
                    (0.0..=1.0).contains(&value)
                };
            if in_range { value } else { default }
        };
        let half_life = if self.half_life < Duration::from_secs(1) {
            Self::default().half_life
        } else {
            self.half_life
        };
        Self {
            decay: self.decay,
            half_life,
            merge: self.merge,
            merge_threshold: normalize(self.merge_threshold, Self::default().merge_threshold, true),
            prune_floor: normalize(self.prune_floor, Self::default().prune_floor, false),
        }
    }
}

/// Swap the cluster's highest-quality member into the canonical slot.
///
/// Clustering seeds each cluster with the first entry seen; the canonical
/// member is defined as the best-scoring one, so merging keeps the
/// strongest entry's identity and text.
fn promote_highest_quality(cluster: &mut MemoryCluster, now: SystemTime) {
    let mut best_index = None;
    let mut best_score = quality_score(&cluster.canonical, now);
    for (index, duplicate) in cluster.duplicates.iter().enumerate() {
        let score = quality_score(duplicate, now);
        if score > best_score {
            best_score = score;
            best_index = Some(index);
        }
    }
    if let Some(index) = best_index {
        let duplicate = cluster.duplicates.remove(index);
        let demoted = std::mem::replace(&mut cluster.canonical, duplicate);
        cluster.duplicates.push(demoted);
    }
}

/// Lowercased, punctuation-trimmed whitespace tokens of a text.
///
/// The comparison unit for Jaccard similarity: punctuation edges do
/// not distinguish lessons, so they are stripped before tokenizing.
fn normalized_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

/// Jaccard similarity of two token sets: shared tokens over the union.
///
/// Two empty sets are identical (1.0); a non-empty set against an empty
/// one shares nothing — a zero intersection over a non-empty union,
/// which divides to `0.0` through the ordinary
/// [`unit_ratio`](crate::numeric::unit_ratio) path.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    crate::numeric::unit_ratio(a.intersection(b).count(), a.union(b).count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::entry::{MemoryCategory, MemoryEntry};
    use std::time::Duration;

    fn entry(category: MemoryCategory, text: &str) -> MemoryEntry {
        MemoryEntry::new(category, text)
    }

    #[test]
    fn quality_score_weights_relevance_access_recency_and_validation() {
        let now = SystemTime::now();
        let mut relevance_heavy = entry(MemoryCategory::Insight, "high relevance, never touched");
        relevance_heavy.relevance = 1.0;
        let mut access_heavy = entry(MemoryCategory::Insight, "low relevance, often retrieved");
        access_heavy.relevance = 0.2;
        access_heavy.access_count = 50;
        assert!(
            quality_score(&relevance_heavy, now) > quality_score(&access_heavy, now),
            "relevance dominates the composite: {} vs {}",
            quality_score(&relevance_heavy, now),
            quality_score(&access_heavy, now)
        );
        let mut plain = entry(MemoryCategory::Insight, "baseline");
        plain.relevance = 0.5;
        let mut out_of_range = entry(MemoryCategory::Insight, "deserialized above the range");
        out_of_range.relevance = 1.5;
        assert!(
            quality_score(&out_of_range, now) <= 1.0,
            "the relevance term is clamped, so an out-of-range value cannot push the \
            composite past its documented ceiling"
        );
        let mut validated = plain.clone();
        validated.validated = true;
        assert!(
            quality_score(&validated, now) > quality_score(&plain, now),
            "validation is a strict bonus"
        );
    }

    #[test]
    fn decay_relevance_halves_per_half_life_and_weights_categories() {
        let now = SystemTime::now();
        let half_life = Duration::from_secs(3600);
        let mut one_half = entry(MemoryCategory::Trajectory, "aged one half-life");
        one_half.relevance = 0.8;
        one_half.created_at = now - half_life;
        decay_relevance(&mut one_half, now, half_life);
        assert!(
            (one_half.relevance - 0.4).abs() < 1e-4,
            "one half-life should halve relevance, got {}",
            one_half.relevance
        );
        let mut two_halves = entry(MemoryCategory::Trajectory, "aged two half-lives");
        two_halves.relevance = 0.8;
        two_halves.created_at = now - half_life * 2;
        decay_relevance(&mut two_halves, now, half_life);
        assert!(
            (two_halves.relevance - 0.2).abs() < 1e-4,
            "two half-lives should quarter relevance, got {}",
            two_halves.relevance
        );
        let mut fact = entry(MemoryCategory::Fact, "durable");
        fact.relevance = 0.8;
        fact.created_at = now - half_life;
        let mut working = entry(MemoryCategory::Working, "ephemeral");
        working.relevance = 0.8;
        working.created_at = now - half_life;
        decay_relevance(&mut fact, now, half_life);
        decay_relevance(&mut working, now, half_life);
        assert!(
            fact.relevance > working.relevance,
            "facts decay slower than working memory at the same age"
        );
    }

    #[test]
    fn decay_relevance_is_idempotent_across_consolidation_cadences() {
        let start = SystemTime::now();
        let half_life = Duration::from_secs(3600);
        let total_age = Duration::from_hours(10);
        let mut single = entry(MemoryCategory::Trajectory, "decayed once");
        single.relevance = 1.0;
        single.created_at = start;
        decay_relevance(&mut single, start + total_age, half_life);
        let mut incremental = entry(MemoryCategory::Trajectory, "decayed in ten passes");
        incremental.relevance = 1.0;
        incremental.created_at = start;
        for step in 1..=10 {
            decay_relevance(
                &mut incremental,
                start + Duration::from_hours(step),
                half_life,
            );
        }
        assert!(
            (single.relevance - incremental.relevance).abs() < 1e-5,
            "ten hourly passes must equal one pass over the same total age: \
            {} vs {}",
            single.relevance,
            incremental.relevance
        );
        assert!(
            (incremental.relevance - 0.5_f32.powf(10.0)).abs() < 1e-5,
            "ten half-lives at neutral weight decay to 2^-10 exactly — no more, no \
            less: {}",
            incremental.relevance
        );
        let mut durable = entry(MemoryCategory::Fact, "decayed as a durable fact");
        durable.relevance = 1.0;
        durable.created_at = start;
        for step in 1..=10 {
            decay_relevance(&mut durable, start + Duration::from_hours(step), half_life);
        }
        assert!(
            durable.relevance > incremental.relevance,
            "the category weight applies per elapsed time, not per pass — a fact \
            outlives a trajectory at the same age"
        );
    }

    #[test]
    fn decay_relevance_never_redecays_time_across_a_backwards_clock() {
        let start = SystemTime::now();
        let half_life = Duration::from_secs(3600);
        let mut stepped = entry(MemoryCategory::Trajectory, "passed across a clock step");
        stepped.relevance = 1.0;
        stepped.created_at = start;
        decay_relevance(&mut stepped, start + Duration::from_secs(3600), half_life);
        let after_first_pass = stepped.relevance;
        decay_relevance(&mut stepped, start + Duration::from_secs(3599), half_life);
        decay_relevance(&mut stepped, start + Duration::from_secs(3600), half_life);
        assert!(
            (stepped.relevance - after_first_pass).abs() < 1e-6,
            "a pass behind the stamp must neither decay nor rewind it — otherwise the \
            next pass re-decays already-decayed time: {} vs {}",
            stepped.relevance,
            after_first_pass
        );
        assert_eq!(
            stepped.last_decayed,
            Some(start + Duration::from_secs(3600)),
            "the stamp stays at the latest pass, never steps back"
        );
    }

    #[test]
    fn decay_resists_for_recently_accessed_entries() {
        let now = SystemTime::now();
        let half_life = Duration::from_secs(3600);
        let old = now - Duration::from_hours(24);
        let mut untouched = entry(MemoryCategory::Trajectory, "old and untouched");
        untouched.relevance = 0.8;
        untouched.created_at = old;
        let mut touched = entry(MemoryCategory::Trajectory, "old but recently retrieved");
        touched.relevance = 0.8;
        touched.created_at = old;
        touched.last_accessed = Some(now);
        decay_relevance(&mut untouched, now, half_life);
        decay_relevance(&mut touched, now, half_life);
        assert!(
            touched.relevance > untouched.relevance,
            "a recent access must shield an entry from decay"
        );
    }

    #[test]
    fn cluster_duplicates_threshold_and_category_partition() {
        let near = "run cargo check after multi-file edits before declaring done";
        let first = entry(MemoryCategory::Strategy, near);
        let mut second = entry(MemoryCategory::Strategy, near);
        second.tags.push("alpha".to_string());
        let paraphrase = entry(
            MemoryCategory::Strategy,
            "verify the build once several files have changed",
        );
        let other_category = entry(MemoryCategory::Insight, near);
        let clusters = cluster_duplicates(&[first, second, paraphrase, other_category], 0.6);
        assert_eq!(
            clusters.len(),
            3,
            "identical wording in another category never joins the strategy pair"
        );
        let clustered = clusters
            .iter()
            .find(|cluster| !cluster.duplicates.is_empty())
            .expect("the identical strategy pair must cluster");
        assert_eq!(
            clustered.duplicates.len(),
            1,
            "the paraphrase stays a singleton, not a duplicate"
        );
        assert_eq!(
            clustered.canonical.category,
            MemoryCategory::Strategy,
            "the cluster is the strategy pair"
        );
    }

    #[test]
    fn merge_cluster_unions_tags_sums_access_and_caps_relevance() {
        let now = SystemTime::now();
        let long_ago = now - Duration::from_hours(720);
        let mut canonical = entry(MemoryCategory::Insight, "canonical text");
        canonical.relevance = 0.9;
        canonical.access_count = 3;
        canonical.tags.push("core".to_string());
        canonical.created_at = now;
        let mut first = entry(MemoryCategory::Insight, "canonical text variant");
        first.relevance = 0.7;
        first.access_count = 2;
        first.tags.push("core".to_string());
        first.tags.push("extra".to_string());
        first.validated = true;
        first.created_at = long_ago;
        let mut second = entry(MemoryCategory::Insight, "canonical text variant two");
        second.relevance = 0.7;
        second.access_count = 4;
        second.created_at = long_ago;
        let accessed_late = now - Duration::from_hours(1);
        second.last_accessed = Some(accessed_late);
        let merged = merge_cluster(MemoryCluster {
            canonical,
            duplicates: vec![first, second],
        });
        assert_eq!(
            merged.last_accessed,
            Some(accessed_late),
            "the merge keeps the freshest access stamp"
        );
        assert_eq!(
            merged.tags,
            vec!["core", "extra"],
            "tags union without duplicates"
        );
        assert_eq!(
            merged.access_count, 9,
            "access counts sum across the cluster"
        );
        assert!(merged.validated, "any validated member validates the merge");
        assert!(
            (merged.relevance - 1.0).abs() < f32::EPSILON,
            "corroboration raises relevance but caps at 1.0"
        );
        assert_eq!(
            merged.created_at, long_ago,
            "provenance keeps the earliest creation time"
        );
    }

    #[test]
    fn consolidate_entries_prunes_merges_and_keeps_the_valuable() {
        let now = SystemTime::now();
        let stale_age = now - Duration::from_hours(8760);
        let mut stale = entry(MemoryCategory::Working, "stale and hollow");
        stale.relevance = 0.3;
        stale.created_at = stale_age;
        let mut duplicate_a = entry(
            MemoryCategory::Strategy,
            "run cargo check after multi-file edits before declaring done",
        );
        duplicate_a.relevance = 0.6;
        let mut duplicate_b = entry(
            MemoryCategory::Strategy,
            "run cargo check after multi-file edits before declaring done",
        );
        duplicate_b.relevance = 0.6;
        let mut valuable = entry(MemoryCategory::Fact, "durable, accessed, validated");
        valuable.relevance = 0.8;
        valuable.validated = true;
        valuable.access_count = 40;
        valuable.last_accessed = Some(now);
        let mut entries = vec![stale, duplicate_a, duplicate_b, valuable];
        let config = ConsolidationConfig::default();
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(stats.entries_before, 4, "four entries entered the pass");
        assert_eq!(stats.merged, 1, "the near-duplicate pair folds into one");
        assert_eq!(stats.pruned, 1, "the stale hollow entry ages out");
        assert_eq!(
            stats.entries_after, 2,
            "the merged strategy and the valuable fact survive"
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.category == MemoryCategory::Strategy),
            "the merged strategy survives"
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.memory == "durable, accessed, validated"),
            "the valuable fact survives"
        );
    }

    #[test]
    fn garbage_thresholds_degrade_to_conservative_behaviour() {
        let now = SystemTime::now();
        let mut alpha = entry(MemoryCategory::Strategy, "run the build before committing");
        alpha.relevance = 0.9;
        let mut beta = entry(
            MemoryCategory::Strategy,
            "totally different advice about tests",
        );
        beta.relevance = 0.9;
        let mut keeper = entry(MemoryCategory::Fact, "durable and validated");
        keeper.relevance = 0.9;
        keeper.validated = true;

        let mut entries = vec![alpha.clone(), beta.clone(), keeper.clone()];
        let mut config = ConsolidationConfig::default();
        config.merge_threshold = -1.0;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.merged, 0,
            "a negative threshold falls back to the default, so distinct same-category \
            texts stay separate"
        );
        assert_eq!(entries.len(), 3, "nothing merged, nothing pruned");

        let mut entries = vec![alpha.clone(), beta.clone(), keeper.clone()];
        let mut config = ConsolidationConfig::default();
        config.merge_threshold = 0.0;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.merged, 0,
            "a zero threshold also falls back: disjoint sets score Jaccard 0.0, so \
            keeping it would collapse every same-category entry into one"
        );
        assert_eq!(entries.len(), 3, "nothing merged, nothing pruned");

        let mut entries = vec![alpha.clone(), beta.clone(), keeper.clone()];
        let mut config = ConsolidationConfig::default();
        config.prune_floor = 0.0;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.pruned, 0,
            "a zero prune floor is valid, not garbage — nothing sits below it"
        );
        assert_eq!(entries.len(), 3, "a zero floor prunes nothing");

        let mut entries = vec![alpha.clone(), beta, keeper.clone()];
        let mut config = ConsolidationConfig::default();
        config.prune_floor = 2.0;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.entries_after,
            entries.len(),
            "stats describe the same store the mutation produced"
        );
        assert_eq!(
            entries.len(),
            3,
            "a floor above 1.0 falls back to the default and must not prune anything"
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.memory == "durable and validated"),
            "the validated high-relevance entry survives the clamped floor"
        );

        let mut entries = vec![alpha, keeper];
        let mut config = ConsolidationConfig::default();
        config.merge_threshold = f32::NAN;
        config.prune_floor = f32::NAN;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.pruned, 0,
            "NaN thresholds fall back to the defaults, and healthy entries are above \
            the default floor"
        );
        assert_eq!(entries.len(), 2, "NaN thresholds behave like the defaults");
    }

    #[test]
    fn a_non_finite_relevance_prunes_instead_of_poisoning_the_pass() {
        let now = SystemTime::now();
        let mut healthy = entry(
            MemoryCategory::Strategy,
            "commit only after the build passes",
        );
        healthy.relevance = 0.9;
        let mut nan = entry(MemoryCategory::Strategy, "poisoned by a NaN relevance");
        nan.relevance = f32::NAN;
        let mut infinite = entry(MemoryCategory::Strategy, "poisoned by an infinite one");
        infinite.relevance = f32::INFINITY;

        let mut entries = vec![healthy, nan, infinite];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(
            stats.pruned, 2,
            "a hand-constructed non-finite relevance degrades to zero and prunes, \
            instead of failing every comparison and surviving"
        );
        assert_eq!(entries.len(), 1, "only the healthy entry remains");
        assert!(
            (entries[0].relevance - 0.9).abs() < 1e-6,
            "the healthy entry's relevance is untouched by its neighbours' sanitization"
        );
    }

    #[test]
    fn a_zero_half_life_falls_back_to_the_default_instead_of_total_decay() {
        let now = SystemTime::now();
        let mut entry = entry(MemoryCategory::Trajectory, "fresh knowledge");
        entry.relevance = 1.0;
        entry.created_at = now - Duration::from_secs(5);
        let mut entries = vec![entry];
        let mut config = ConsolidationConfig::default();
        config.half_life = Duration::ZERO;
        consolidate_entries(&mut entries, &config, now);
        assert!(
            entries[0].relevance > 0.99,
            "a zero half-life must fall back to the 14-day default, so an entry \
            aged five seconds barely moves — not decay to ~zero: {}",
            entries[0].relevance
        );
    }

    #[test]
    fn an_out_of_range_relevance_degrades_like_a_non_finite_one() {
        let now = SystemTime::now();
        let mut healthy = entry(MemoryCategory::Fact, "in range and durable");
        healthy.relevance = 0.9;
        healthy.validated = true;
        let mut above = entry(
            MemoryCategory::Strategy,
            "relevance above the documented range",
        );
        above.relevance = 3.0;
        let mut below = entry(
            MemoryCategory::Working,
            "relevance below the documented range",
        );
        below.relevance = -0.5;

        let mut entries = vec![healthy, above, below];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(
            stats.pruned, 2,
            "a finite but out-of-range relevance degrades to zero and prunes, keeping \
            quality_score inside its documented 0.0..=1.0 composite range"
        );
        assert_eq!(entries.len(), 1, "only the healthy entry remains");
        assert!(
            (entries[0].relevance - 0.9).abs() < 1e-6,
            "the healthy entry's relevance is untouched"
        );
    }

    #[test]
    fn a_zero_threshold_falls_back_instead_of_collapsing_categories() {
        let first = entry(MemoryCategory::Strategy, "run the build before committing");
        let second = entry(
            MemoryCategory::Strategy,
            "totally different advice about tests",
        );
        let clusters = cluster_duplicates(&[first, second], 0.0);
        assert_eq!(
            clusters.len(),
            2,
            "a zero threshold falls back to the default — disjoint same-category \
            entries must stay separate even when a caller bypasses the config path"
        );
    }

    #[test]
    fn tokenless_entries_never_corroborate_each_other() {
        let now = SystemTime::now();
        let mut empty_text = entry(MemoryCategory::Insight, "");
        empty_text.relevance = 0.04;
        let mut emoji_only = entry(MemoryCategory::Insight, "😅 … !");
        emoji_only.relevance = 0.04;

        let mut entries = vec![empty_text, emoji_only];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(
            stats.merged, 0,
            "entries sharing zero tokens must not cluster — corroboration manufactured \
            from nothing would lift them over the prune floor"
        );
        assert_eq!(
            stats.pruned, 2,
            "degenerate entries prune on their own weak relevance"
        );
        assert!(entries.is_empty());
    }

    #[test]
    fn degraded_entries_never_corroborate_a_survivor() {
        let now = SystemTime::now();
        let mut healthy_entry = entry(MemoryCategory::Insight, "run the build before committing");
        healthy_entry.relevance = 0.9;
        let mut corrupted = entry(MemoryCategory::Insight, "run the build before committing");
        corrupted.relevance = f32::NAN;
        corrupted.validated = true;
        corrupted.access_count = 900_000;
        let mut entries = vec![healthy_entry, corrupted];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(
            stats.merged, 0,
            "a degraded entry must not corroborate the healthy one — no cluster forms"
        );
        assert_eq!(entries.len(), 1, "the corrupted entry prunes");
        assert!(
            !entries[0].validated && entries[0].access_count == 0,
            "the garbage validated flag and access count die with it"
        );
    }

    #[test]
    fn degraded_entries_do_not_cluster_into_an_unprunable_survivor() {
        let now = SystemTime::now();
        let mut first = entry(MemoryCategory::Fact, "identical corrupted text");
        first.relevance = f32::INFINITY;
        let mut second = entry(MemoryCategory::Fact, "identical corrupted text");
        second.relevance = -2.0;
        let mut entries = vec![first, second];
        consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert!(
            entries.is_empty(),
            "two degraded near-duplicates must not merge into a corroborated \
            survivor that clears the prune floor"
        );
    }

    #[test]
    fn a_zero_prune_floor_retains_a_degraded_entry_at_zero_relevance() {
        let now = SystemTime::now();
        let mut healthy_entry = entry(MemoryCategory::Insight, "cache the build outputs");
        healthy_entry.relevance = 0.9;
        let mut corrupted = entry(MemoryCategory::Insight, "cache the build outputs");
        corrupted.relevance = f32::NAN;
        let mut entries = vec![healthy_entry, corrupted];
        let mut config = ConsolidationConfig::default();
        config.prune_floor = 0.0;
        let stats = consolidate_entries(&mut entries, &config, now);
        assert_eq!(
            stats.merged, 0,
            "the degraded entry stays out of clustering on the pass that sees it"
        );
        let corrupted = entries
            .iter()
            .find(|entry| entry.relevance == 0.0)
            .expect("the zero floor retains the degraded entry");
        assert_eq!(
            corrupted.memory, "cache the build outputs",
            "retained at degraded relevance, exactly as the floor-0.0 doc promises"
        );
        assert_eq!(entries.len(), 2, "nothing prunes under a zero floor");
    }

    #[test]
    fn consolidate_entries_promotes_the_strongest_member_into_canonical() {
        let now = SystemTime::now();
        let mut weak = entry(MemoryCategory::Insight, "run the build before committing");
        weak.relevance = 0.3;
        weak.created_at = now - Duration::from_hours(24 * 30);
        let mut strong = entry(MemoryCategory::Insight, "run the build before committing");
        strong.relevance = 0.9;
        strong.validated = true;
        strong.access_count = 20;
        strong.last_accessed = Some(now);
        let strong_id = strong.id;
        let mut entries = vec![weak, strong];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(
            stats.merged, 1,
            "the weak canonical counts as the one merged-away duplicate"
        );
        assert_eq!(entries.len(), 1, "the cluster folds into one survivor");
        assert_eq!(
            entries[0].id, strong_id,
            "promotion swaps the strongest member into the canonical slot — \
            the survivor carries its identity, not the first-seen canonical's"
        );
        assert!(
            entries[0].validated,
            "the promoted member's validated flag survives the merge"
        );
    }

    #[test]
    fn a_merged_survivor_decays_only_the_canonicals_window() {
        let now = SystemTime::now();
        let mut canonical = entry(MemoryCategory::Trajectory, "fresh canonical");
        canonical.relevance = 0.9;
        canonical.created_at = now - Duration::from_hours(24);
        let mut stale = entry(MemoryCategory::Trajectory, "fresh canonical");
        stale.relevance = 0.1;
        stale.created_at = now - Duration::from_hours(24 * 60);
        stale.last_decayed = Some(now - Duration::from_hours(24 * 29));
        let mut merged = merge_cluster(MemoryCluster {
            canonical,
            duplicates: vec![stale],
        });
        assert_eq!(
            merged.created_at,
            now - Duration::from_hours(24 * 60),
            "provenance still adopts the cluster's earliest creation"
        );
        decay_relevance(&mut merged, now, Duration::from_hours(336));
        assert!(
            merged.relevance > 0.9 && merged.relevance < 1.0,
            "the first post-merge pass decays one day — the canonical's unaccounted \
            window — not the adopted 60-day provenance: {}",
            merged.relevance
        );
    }

    #[test]
    fn a_nan_relevance_survives_a_merge_visibly_broken() {
        let mut canonical = entry(MemoryCategory::Insight, "poisoned canonical");
        canonical.relevance = f32::NAN;
        let duplicate = entry(MemoryCategory::Insight, "poisoned canonical");
        let merged = merge_cluster(MemoryCluster {
            canonical,
            duplicates: vec![duplicate],
        });
        assert!(
            merged.relevance.is_nan(),
            "the clamp propagates NaN instead of f32::min's ignore-NaN cementing it \
            at exactly 1.0 — the next sanitizing pass prunes it"
        );
    }

    #[test]
    fn bytes_saved_is_nonzero_when_entries_are_removed() {
        let now = SystemTime::now();
        let mut below_floor = entry(MemoryCategory::Working, "drop me");
        below_floor.relevance = 0.01;
        let mut keeper = entry(MemoryCategory::Fact, "keep me around");
        keeper.relevance = 0.9;
        keeper.validated = true;
        let mut entries = vec![below_floor, keeper];
        let stats = consolidate_entries(&mut entries, &ConsolidationConfig::default(), now);
        assert_eq!(stats.pruned, 1, "the below-floor entry is dropped");
        assert!(
            stats.bytes_saved > 0,
            "pruning reclaims the entry's text bytes"
        );
    }
}
