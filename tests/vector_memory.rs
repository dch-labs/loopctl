//! Integration tests for
//! [`VectorMemoryStore`](loopctl::memory::vector_memory::VectorMemoryStore)
//! — the locked contract cases from the store's design record.
//!
//! Everything here is deterministic and network-free: the
//! [`HashingEmbedder`](loopctl::memory::vector::HashingEmbedder) +
//! [`LinearVectorIndex`](loopctl::memory::vector::LinearVectorIndex)
//! reference pair backs every store, and the golden-set fixture pins the
//! semantic-precision gate.

#![cfg(feature = "vector_memory")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::error::LoopError;
use loopctl::memory::vector::{Embedding, HashingEmbedder, LinearVectorIndex, VectorIndex};
use loopctl::memory::vector_memory::{
    GoldenSet, VectorMemoryConfig, VectorMemoryStore, golden_set,
};
use loopctl::memory::{InMemoryStore, LoopMemory, MemoryCategory, MemoryEntry};
use uuid::Uuid;

const DIM: usize = 128;

fn store() -> VectorMemoryStore {
    VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(LinearVectorIndex::new(DIM)),
    )
}

/// An entry with a caller-chosen id, so id tie-breaks are deterministic.
///
/// Retrieval sorts descending by blend and breaks ties by ascending id;
/// pinning ids lets a test decide which of two equal-scoring twins wins
/// the tie.
fn entry(id: Uuid, text: &str) -> MemoryEntry {
    let mut built = MemoryEntry::new(MemoryCategory::Fact, text);
    built.id = id;
    built
}

/// A delegating [`VectorIndex`] whose inner index stays observable after
/// the store takes ownership — the eviction pin must see the index
/// directly, because `retrieve` deliberately filters candidates missing
/// from the payload map and would otherwise mask a stale vector.
///
/// The wrapper clones cheaply (every field is an `Arc`), so one index
/// serves both the store under test and the test's assertion handle.
#[derive(Clone)]
struct SharedIndex(Arc<LinearVectorIndex>);

impl SharedIndex {
    /// Build the store-side wrapper and the test-side handle over one
    /// index.
    ///
    /// Both halves view the same underlying `LinearVectorIndex`, so an
    /// eviction performed through the store is visible through the
    /// handle immediately.
    fn shared(dim: usize) -> (SharedIndex, SharedIndex) {
        let inner = Arc::new(LinearVectorIndex::new(dim));
        (SharedIndex(Arc::clone(&inner)), SharedIndex(inner))
    }
}

impl loopctl::memory::vector::VectorIndex for SharedIndex {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn add(
        &self,
        id: Uuid,
        vector: Embedding,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        self.0.add(id, vector)
    }

    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<loopctl::memory::vector::VectorMatch>, LoopError>>
                + Send
                + '_,
        >,
    > {
        self.0.search(query, k)
    }

    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        self.0.remove(id)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// A set-once flag a spawned task can await, for deterministic
/// interleavings on the current-thread test runtime.
///
/// The flag-plus-`Notify` shape makes the choreography tests
/// order-dependent by construction rather than by timing: a task parks
/// until the test explicitly releases it.
#[derive(Default)]
struct Event {
    set: std::sync::atomic::AtomicBool,
    notify: tokio::sync::Notify,
}

impl Event {
    /// Mark the event as set and wake every waiter.
    ///
    /// `notify_waiters` wakes only currently registered listeners, which
    /// is why [`wait`](Self::wait) re-checks the flag around the listener
    /// registration instead of trusting a single check.
    fn set(&self) {
        self.set.store(true, std::sync::atomic::Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Whether the event has been marked.
    ///
    /// The non-blocking probe the double-check in [`wait`](Self::wait)
    /// and the interleaving tests' assertions use.
    fn is_set(&self) -> bool {
        self.set.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Wait until the event is marked.
    ///
    /// The double check brackets the listener registration, per the
    /// `Notify` docs, so a `set` between the check and the await cannot
    /// be missed.
    async fn wait(&self) {
        loop {
            if self.is_set() {
                return;
            }
            let notified = self.notify.notified();
            tokio::pin!(notified);
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

/// A delegating index whose `remove` parks on a gate, so a test can hold
/// a consolidation pass mid-sweep and probe what a racing `store` does.
///
/// Every other operation delegates straight through, so the parked pass
/// exercises the production sweep path rather than a test double's
/// semantics.
#[derive(Clone)]
struct GatedIndex {
    inner: Arc<LinearVectorIndex>,
    remove_started: Arc<Event>,
    remove_proceed: Arc<Event>,
}

impl loopctl::memory::vector::VectorIndex for GatedIndex {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn add(
        &self,
        id: Uuid,
        vector: Embedding,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        self.inner.add(id, vector)
    }

    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<loopctl::memory::vector::VectorMatch>, LoopError>>
                + Send
                + '_,
        >,
    > {
        self.inner.search(query, k)
    }

    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let remove_started = Arc::clone(&self.remove_started);
        let remove_proceed = Arc::clone(&self.remove_proceed);
        Box::pin(async move {
            remove_started.set();
            remove_proceed.wait().await;
            inner.remove(id).await
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// While a consolidation pass is parked mid-sweep, a racing `store` must
/// not be able to complete — the pass-sequencing lock is what keeps the
/// snapshot-bound removals sound, and what prevents both reviewed
/// interleavings (a swallowed `Ok` store and a mapped-but-unindexed
/// zombie).
///
/// After the pass drains, the replacement must survive intact:
/// retrieved, counted, never swallowed.
#[tokio::test]
async fn a_store_cannot_complete_while_a_consolidation_pass_is_mid_sweep() {
    let gated = GatedIndex {
        inner: Arc::new(LinearVectorIndex::new(DIM)),
        remove_started: Arc::new(Event::default()),
        remove_proceed: Arc::new(Event::default()),
    };
    let store = Arc::new(VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(gated.clone()),
    ));

    let mut stale = entry(
        Uuid::from_u128(1),
        "a stale fact racing its own replacement",
    );
    stale.relevance = 0.05;
    store.store(stale).await.unwrap();

    let pass_store = Arc::clone(&store);
    let pass = tokio::spawn(async move { pass_store.consolidate().await });
    gated.remove_started.wait().await;

    let racing_store = Arc::clone(&store);
    let racing = tokio::spawn(async move {
        let mut fresh = entry(
            Uuid::from_u128(1),
            "a stale fact racing its own replacement",
        );
        fresh.relevance = 0.9;
        racing_store.store(fresh).await
    });

    for _ in 0..64 {
        assert!(
            !racing.is_finished(),
            "a store must not complete while a consolidation pass is \
             parked mid-sweep — the pass lock must serialize them"
        );
        tokio::task::yield_now().await;
    }

    gated.remove_proceed.set();
    let stats = pass.await.unwrap().unwrap();
    assert_eq!(stats.pruned, 1, "the pass prunes the stale snapshot entry");
    racing.await.unwrap().unwrap();
    let found = store
        .retrieve("a stale fact racing its own replacement", 5)
        .await
        .unwrap();
    assert!(
        found
            .iter()
            .any(|e| e.id == Uuid::from_u128(1) && (e.relevance - 0.9).abs() < 1e-6),
        "the racing store's replacement must survive the pass — neither \
         swallowed nor left a mapped-but-unindexed zombie: {found:?}"
    );
    assert_eq!(store.len(), 1, "exactly the replacement remains");
}

/// While a consolidation pass is parked mid-sweep, a `retrieve` that stamps
/// a planned victim must race that victim into survival.
///
/// The pass's snapshot-bound removal re-check sees the access stamp, keeps
/// the entry, and re-links its index vector — and the victim's tag fold is
/// suppressed, because folds only apply to confirmed removals. This is the
/// raced branch's live trigger: `retrieve` takes no part in the pass lock,
/// so it is the one operation that can change a payload between the
/// snapshot and the apply.
#[tokio::test]
async fn a_retrieve_stamp_between_snapshot_and_apply_races_the_victim_into_survival() {
    let gated = GatedIndex {
        inner: Arc::new(LinearVectorIndex::new(DIM)),
        remove_started: Arc::new(Event::default()),
        remove_proceed: Arc::new(Event::default()),
    };
    let store = Arc::new(VectorMemoryStore::new(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(gated.clone()),
    ));

    let shared_text = "the deploy pipeline reruns on config changes";
    let mut survivor = entry(Uuid::from_u128(1), shared_text);
    survivor.relevance = 0.9;
    let victim_id = Uuid::from_u128(2);
    let mut victim = entry(victim_id, shared_text);
    victim.relevance = 0.5;
    victim.tags.push("folds-on-confirm".to_string());
    store.store(survivor.clone()).await.unwrap();
    store.store(victim).await.unwrap();

    let pass_store = Arc::clone(&store);
    let pass = tokio::spawn(async move { pass_store.consolidate().await });
    gated.remove_started.wait().await;

    let stamped = store.retrieve(shared_text, 2).await.unwrap();
    assert_eq!(
        stamped.len(),
        2,
        "the parked pass holds no payload lock, so the retrieve completes \
         and stamps both near-duplicates mid-pass"
    );

    gated.remove_proceed.set();
    let stats = pass.await.unwrap().unwrap();
    assert_eq!(
        stats.merged, 0,
        "the retrieve stamp raced the victim out of its confirmed removal"
    );
    assert_eq!(stats.pruned, 0, "nothing fell below the prune floor");
    assert_eq!(store.len(), 2, "the raced victim survives the pass");
    let found = store.retrieve(shared_text, 2).await.unwrap();
    assert!(
        found.iter().any(|stored| stored.id == victim_id),
        "the raced victim is re-linked into the index and still retrieves: \
         {found:?}"
    );
    for stored in found.iter().filter(|stored| stored.id == survivor.id) {
        assert!(
            !stored.tags.contains(&"folds-on-confirm".to_string()),
            "a fold suppressed for an unconfirmed victim must never reach \
             the survivor: {stored:?}"
        );
    }
}

/// The golden fixture spans exactly the four documented categories.
///
/// The `GoldenSet` doc names `Strategy`, `Fact`, `Insight`, and
/// `ErrorPattern` — a characterization pin, so widening the fixture to
/// `Trajectory` or `Working` breaks here and forces the doc along with it.
#[test]
fn the_golden_fixture_spans_four_of_the_six_categories() {
    let set = golden_set();
    assert_eq!(
        set.entries.len(),
        50,
        "the fixture holds fifty entries — the documented size"
    );
    let mut categories: Vec<String> = Vec::new();
    for fixture_entry in &set.entries {
        let name = format!("{:?}", fixture_entry.category);
        if !categories.contains(&name) {
            categories.push(name);
        }
    }
    categories.sort_unstable();
    assert_eq!(
        categories,
        vec![
            "ErrorPattern".to_string(),
            "Fact".to_string(),
            "Insight".to_string(),
            "Strategy".to_string(),
        ],
        "the fixture spans exactly the four documented categories — widen \
         the fixture and the GoldenSet doc together"
    );
}

#[tokio::test]
async fn golden_set_queries_hit_a_relevant_memory_eighty_percent_of_the_time() {
    let set = golden_set();
    let store = store();
    for fixture_entry in &set.entries {
        store.store(fixture_entry.clone()).await.unwrap();
    }
    assert_eq!(store.len(), set.entries.len(), "every fixture entry stored");
    let mut hits: Vec<bool> = Vec::with_capacity(set.queries.len());
    for (index, query) in set.queries.iter().enumerate() {
        let relevant = set.relevant_ids(index);
        let returned = store.retrieve(query.text, 3).await.unwrap();
        hits.push(returned.iter().any(|entry| relevant.contains(&entry.id)));
    }
    let precision = GoldenSet::precision_ratio(&hits);
    assert!(
        precision >= 0.8,
        "semantic precision {precision:.2} is below the 80% acceptance gate; \
         missed queries: {:?}",
        hits.iter()
            .zip(set.queries.iter())
            .filter(|(hit, _)| !**hit)
            .map(|(_, query)| query.text)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn vector_store_satisfies_the_loop_memory_contract_like_in_memory_store() {
    let fixture: Vec<MemoryEntry> = [
        "the deploy pipeline stages every artifact",
        "the retry ladder backs off on 429s",
        "hash tokens into buckets for embeddings",
        "compaction summarizes the oldest turns",
        "cosine scores the angle between vectors",
        "sqlite journals every write durably",
    ]
    .iter()
    .enumerate()
    .map(|(i, text)| entry(Uuid::from_u128(i as u128), text))
    .collect();
    let in_memory: Arc<dyn LoopMemory> = Arc::new(InMemoryStore::new());
    let vector: Arc<dyn LoopMemory> = Arc::new(store());
    for subject in [&in_memory, &vector] {
        for fixture_entry in &fixture {
            subject.store(fixture_entry.clone()).await.unwrap();
        }
        let returned = subject.retrieve("the", 3).await.unwrap();
        assert_eq!(
            returned.len(),
            3,
            "retrieve returns up to the limit for both stores"
        );
        assert_eq!(subject.len(), fixture.len(), "len counts stored entries");
        assert!(!subject.is_empty(), "a populated store is not empty");
        let stats = subject.consolidate().await.unwrap();
        assert_eq!(
            stats.entries_before,
            fixture.len(),
            "stats describe the pre-consolidation size"
        );
        assert_eq!(subject.len(), fixture.len(), "healthy entries survive");
    }
}

#[tokio::test]
async fn store_embeds_and_indexes_one_entry() {
    let store = store();
    let stored = entry(Uuid::nil(), "prefer glob over manual file search");
    store.store(stored.clone()).await.unwrap();
    assert_eq!(store.len(), 1, "one entry is stored");
    let returned = store
        .retrieve("prefer glob over manual file search", 1)
        .await
        .unwrap();
    assert_eq!(
        returned.first().map(|entry| entry.id),
        Some(stored.id),
        "the entry's own text retrieves it at rank 0 — it was embedded and indexed"
    );
}

#[tokio::test]
async fn retrieve_respects_the_limit_including_zero() {
    let store = store();
    for i in 0..10_u128 {
        store
            .store(entry(
                Uuid::from_u128(i),
                &format!("fact {i} about caching"),
            ))
            .await
            .unwrap();
    }
    let three = store.retrieve("caching", 3).await.unwrap();
    assert_eq!(three.len(), 3, "limit 3 returns exactly 3");
    let none = store.retrieve("caching", 0).await.unwrap();
    assert!(none.is_empty(), "limit 0 returns an empty vec");
}

#[tokio::test]
async fn lexical_exactness_promotes_an_entry_and_the_weight_knob_flips_it() {
    let query = "alpha beta";
    let substring_only = entry(Uuid::from_u128(2), "alphas betas");
    let unrelated = entry(Uuid::nil(), "unrelated filler words entirely");
    let default_store = store();
    default_store.store(substring_only.clone()).await.unwrap();
    default_store.store(unrelated.clone()).await.unwrap();
    let default_hits = default_store.retrieve(query, 2).await.unwrap();
    assert_eq!(
        default_hits.first().map(|entry| entry.id),
        Some(substring_only.id),
        "with the default blend the lexically active entry ranks first"
    );
    let lexical_off = VectorMemoryConfig {
        lexical_weight: 0.0,
        ..VectorMemoryConfig::default()
    };
    let flipped_store = VectorMemoryStore::with_config(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(LinearVectorIndex::new(DIM)),
        lexical_off,
    );
    flipped_store.store(substring_only).await.unwrap();
    flipped_store.store(unrelated.clone()).await.unwrap();
    let flipped_hits = flipped_store.retrieve(query, 2).await.unwrap();
    assert_eq!(
        flipped_hits.first().map(|entry| entry.id),
        Some(unrelated.id),
        "zeroing the lexical weight removes its channel; the tie resolves \
         by ascending id, flipping the order"
    );
}

#[tokio::test]
async fn a_tag_hit_promotes_an_entry_until_the_tag_weight_is_zeroed() {
    let mut tagged = entry(Uuid::from_u128(9), "deploy the service");
    tagged.tags.push("deploy".to_string());
    let untagged = entry(Uuid::nil(), "deploy the service");
    let default_store = store();
    default_store.store(tagged.clone()).await.unwrap();
    default_store.store(untagged.clone()).await.unwrap();
    let default_hits = default_store.retrieve("deploy", 2).await.unwrap();
    assert_eq!(
        default_hits.first().map(|entry| entry.id),
        Some(tagged.id),
        "the tagged entry outranks its otherwise-identical twin"
    );
    let tags_off = VectorMemoryConfig {
        tag_weight: 0.0,
        ..VectorMemoryConfig::default()
    };
    let flipped_store = VectorMemoryStore::with_config(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(LinearVectorIndex::new(DIM)),
        tags_off,
    );
    flipped_store.store(tagged).await.unwrap();
    flipped_store.store(untagged.clone()).await.unwrap();
    let flipped_hits = flipped_store.retrieve("deploy", 2).await.unwrap();
    assert_eq!(
        flipped_hits.first().map(|entry| entry.id),
        Some(untagged.id),
        "zeroing the tag weight ties the twins; the id tiebreak flips the order"
    );
}

#[tokio::test]
async fn retrieval_increments_access_count_on_returned_entries_only() {
    let store = store();
    let hot = entry(Uuid::from_u128(1), "the deploy runs through the pipeline");
    let cold = entry(Uuid::from_u128(2), "an unrelated note about gardening");
    store.store(hot.clone()).await.unwrap();
    store.store(cold.clone()).await.unwrap();
    for _ in 0..3 {
        let returned = store.retrieve("deploy pipeline", 1).await.unwrap();
        assert_eq!(returned.len(), 1, "only the matching entry is returned");
    }
    let refreshed = store.retrieve("deploy pipeline", 5).await.unwrap();
    let hot_after = refreshed
        .iter()
        .find(|entry| entry.id == hot.id)
        .expect("the hot entry is returned");
    assert_eq!(
        hot_after.access_count, 3,
        "three retrievals stamp exactly three accesses on the returned entry"
    );
    let cold_after = refreshed
        .iter()
        .find(|entry| entry.id == cold.id)
        .expect("the wide window also returns the unrelated entry");
    assert_eq!(
        cold_after.access_count, 0,
        "entries that were never returned keep their zero access count"
    );
}

#[tokio::test]
async fn a_fresh_store_behaves_like_the_in_memory_reference_on_empty() {
    let vector = store();
    let reference = InMemoryStore::new();
    assert_eq!(vector.len(), 0, "fresh vector store is empty");
    assert!(vector.is_empty(), "is_empty agrees with len");
    assert!(vector.retrieve("anything", 3).await.unwrap().is_empty());
    let vector_stats = vector.consolidate().await.unwrap();
    let reference_stats = reference.consolidate().await.unwrap();
    assert_eq!(
        vector_stats
            .entries_before
            .saturating_add(vector_stats.entries_after),
        0,
        "consolidation on empty is a no-op with default stats"
    );
    assert_eq!(
        vector_stats
            .pruned
            .saturating_add(vector_stats.merged)
            .saturating_add(vector_stats.bytes_saved),
        0,
        "no work is reported for an empty pass"
    );
    assert_eq!(reference_stats.entries_after, 0, "the reference agrees");
}

#[tokio::test]
async fn consolidate_prunes_low_relevance_and_evicts_its_vector() {
    let (store_side, handle) = SharedIndex::shared(DIM);
    let store = VectorMemoryStore::new(Box::new(HashingEmbedder::new(DIM)), Box::new(store_side));
    let mut healthy = entry(Uuid::from_u128(1), "a healthy highly relevant fact");
    healthy.relevance = 0.9;
    let mut stale = entry(Uuid::from_u128(2), "a stale fact past its floor");
    stale.relevance = 0.05;
    store.store(healthy).await.unwrap();
    store.store(stale.clone()).await.unwrap();
    assert_eq!(handle.len(), 2, "the index holds both vectors pre-pass");
    let stats = store.consolidate().await.unwrap();
    assert_eq!(stats.pruned, 1, "the below-threshold entry is pruned");
    assert_eq!(stats.merged, 0, "nothing merged");
    assert_eq!(stats.entries_before, 2, "the pass started from two entries");
    assert_eq!(stats.entries_after, 1, "one entry survives");
    assert!(
        stats.bytes_saved > 0,
        "the evicted vector is reported as reclaimed bytes"
    );
    let stale_embedding = HashingEmbedder::new(DIM).embed_sync("a stale fact past its floor");
    let indexed = handle.search(&stale_embedding, 5).await.unwrap();
    assert!(
        indexed.iter().all(|m| m.id != stale.id),
        "the pruned entry's vector was evicted from the index, not just \
         from the payload map: {:?}",
        indexed.iter().map(|m| m.id).collect::<Vec<_>>()
    );
    let survivors = store
        .retrieve("a stale fact past its floor", 5)
        .await
        .unwrap();
    assert!(
        survivors.iter().all(|entry| entry.id != stale.id),
        "the pruned entry no longer retrieves"
    );
}

#[tokio::test]
async fn consolidate_never_prunes_a_validated_entry() {
    let store = store();
    let mut kept = entry(Uuid::from_u128(1), "a validated but stale note");
    kept.relevance = 0.05;
    kept = kept.validated();
    let mut dropped = entry(Uuid::from_u128(2), "a validated but stale note");
    dropped.relevance = 0.05;
    store.store(kept).await.unwrap();
    store.store(dropped).await.unwrap();
    let stats = store.consolidate().await.unwrap();
    assert_eq!(stats.pruned, 1, "only the unvalidated twin is pruned");
    assert_eq!(store.len(), 1, "the validated entry survives the floor");
    let again = store.consolidate().await.unwrap();
    assert_eq!(again.entries_after, 1, "the survivor stays through passes");
}

#[tokio::test]
async fn consolidate_merges_near_duplicates_and_folds_their_tags() {
    let store = store();
    let mut strong = entry(Uuid::from_u128(1), "the cache invalidates on every write");
    strong.relevance = 0.9;
    let mut weak = entry(Uuid::from_u128(2), "the cache invalidates on every write");
    weak.relevance = 0.5;
    weak.tags.push("cache".to_string());
    weak.tags
        .push(loopctl::memory::entry::PROVIDER_DERIVED_TAG.to_string());
    store.store(strong.clone()).await.unwrap();
    store.store(weak).await.unwrap();
    let stats = store.consolidate().await.unwrap();
    assert_eq!(stats.merged, 1, "the identical pair merges");
    assert_eq!(stats.pruned, 0, "merging is not pruning");
    assert_eq!(store.len(), 1, "the survivor replaces the pair");
    let merged = store
        .retrieve("the cache invalidates on every write", 1)
        .await
        .unwrap();
    let survivor = merged.first().expect("the survivor retrieves");
    assert_eq!(
        survivor.id, strong.id,
        "the higher-relevance entry survives"
    );
    assert!(
        survivor.tags.contains(&"cache".to_string()),
        "the victim's topic tags fold into the survivor"
    );
    assert!(
        !survivor
            .tags
            .contains(&loopctl::memory::entry::PROVIDER_DERIVED_TAG.to_string()),
        "the provenance tag never folds — it belongs to the entry that earned it"
    );
    let dedup_disabled = VectorMemoryConfig {
        dedup_threshold: 1.0,
        ..VectorMemoryConfig::default()
    };
    let disabled_store = VectorMemoryStore::with_config(
        Box::new(HashingEmbedder::new(DIM)),
        Box::new(LinearVectorIndex::new(DIM)),
        dedup_disabled,
    );
    disabled_store
        .store(entry(Uuid::from_u128(3), "an identical twin text"))
        .await
        .unwrap();
    disabled_store
        .store(entry(Uuid::from_u128(4), "an identical twin text"))
        .await
        .unwrap();
    let stats = disabled_store.consolidate().await.unwrap();
    assert_eq!(stats.merged, 0, "a threshold of 1.0 disables merging");
    assert_eq!(disabled_store.len(), 2, "both twins survive");
}

#[tokio::test]
async fn concurrent_store_and_retrieve_complete_without_deadlock() {
    let store = Arc::new(store());
    for i in 0..200_u128 {
        store
            .store(entry(
                Uuid::from_u128(i),
                &format!("entry number {i} about topic {}", i % 7),
            ))
            .await
            .unwrap();
    }
    let reader_store = Arc::clone(&store);
    let writer_store = Arc::clone(&store);
    let reader = tokio::spawn(async move {
        for _ in 0..50 {
            reader_store.retrieve("entry number", 5).await.unwrap();
        }
    });
    let writer = tokio::spawn(async move {
        for i in 200..250_u128 {
            writer_store
                .store(entry(
                    Uuid::from_u128(i),
                    &format!("entry number {i} about topic {}", i % 7),
                ))
                .await
                .unwrap();
        }
    });
    let joined = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        tokio::try_join!(reader, writer)
    })
    .await;
    assert!(
        joined.is_ok(),
        "a retrieve racing a store must complete — no lock may be held \
         across the embed await"
    );
    assert_eq!(store.len(), 250, "every concurrent write landed");
}

/// An embedder that always fails, for the error-propagation pin.
///
/// Both paths a caller can hit — storing an entry and querying — funnel
/// through `embed`, so one failing implementation exercises the
/// propagation contract on each side of the store.
struct FailingEmbedder;

impl loopctl::memory::vector::EmbeddingProvider for FailingEmbedder {
    fn dim(&self) -> usize {
        8
    }

    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Embedding, LoopError>> + Send + 'a>> {
        Box::pin(async { Err(LoopError::Memory("embedder offline".to_string())) })
    }
}

#[tokio::test]
async fn an_embedder_failure_propagates_as_a_loop_error() {
    let store = VectorMemoryStore::new(
        Box::new(FailingEmbedder),
        Box::new(LinearVectorIndex::new(8)),
    );
    let stored = store
        .store(entry(Uuid::nil(), "anything at all"))
        .await
        .unwrap_err();
    assert!(
        matches!(stored, LoopError::Memory(_)),
        "a store-side embedder failure surfaces as LoopError, not a panic: {stored:?}"
    );
    let retrieved = store.retrieve("anything", 3).await.unwrap_err();
    assert!(
        matches!(retrieved, LoopError::Memory(_)),
        "a query-side embedder failure surfaces as LoopError: {retrieved:?}"
    );
    assert_eq!(store.len(), 0, "nothing was stored by the failed pass");
}
