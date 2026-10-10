//! Recall, determinism, and tombstone contracts for
//! [`HnswIndex`](loopctl_hnsw::HnswIndex) — the approximate index's
//! backend contracts.
//!
//! The quality gate is the recall gate: HNSW's top-10 must overlap exact
//! brute-force top-10 by at least 90% over 100 queries on 1 000 seeded
//! random unit vectors. Everything is deterministic — fixed seeds
//! everywhere — and network-free.
//!
//! Run: `cargo test -p loopctl-hnsw`

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::HashSet;

use loopctl::memory::vector::{Embedding, LinearVectorIndex, VectorIndex};
use loopctl_hnsw::{HnswIndex, HnswParams};
use uuid::Uuid;

const DIM: usize = 64;
const CORPUS: usize = 1_000;
const QUERIES: usize = 100;
const TOP_K: usize = 10;

/// A seeded pseudo-random unit vector.
///
/// Drawn from the caller's RNG (never entropy) and L2-normalized, so the
/// recall corpus is identical on every run of every machine.
fn unit_vector(rng: &mut fastrand::Rng) -> Embedding {
    let mut vector: Vec<f32> = (0..DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    Embedding::new(vector)
}

/// Build a deterministic corpus and query set from one seeded stream, so
/// every run sees the same data.
///
/// `corpus_size` vectors are drawn first, then `queries` query vectors,
/// all from one `fastrand` generator seeded with `seed` and normalized to
/// unit length — sharing a single stream is what makes the recall,
/// determinism, and rebuild pins reproducible on every machine.
fn seeded_data(corpus_size: usize, queries: usize, seed: u64) -> (Vec<Embedding>, Vec<Embedding>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let corpus = (0..corpus_size).map(|_| unit_vector(&mut rng)).collect();
    let queries = (0..queries).map(|_| unit_vector(&mut rng)).collect();
    (corpus, queries)
}

#[tokio::test]
async fn hnsw_recall_at_ten_tracks_brute_force_on_random_unit_vectors() {
    let (corpus, queries) = seeded_data(CORPUS, QUERIES, 0x5EED_1A18);
    let linear = LinearVectorIndex::new(DIM);
    let hnsw = HnswIndex::new(DIM);
    for (index, vector) in corpus.iter().enumerate() {
        let id = Uuid::from_u128(index as u128);
        linear.add(id, vector.clone()).await.unwrap();
        hnsw.add(id, vector.clone()).await.unwrap();
    }
    assert_eq!(hnsw.len(), CORPUS, "the graph holds every vector");
    let mut recall_sum = 0.0_f32;
    for query in &queries {
        let exact: HashSet<Uuid> = linear
            .search(query, TOP_K)
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        let approximate = hnsw.search(query, TOP_K).await.unwrap();
        let overlap = approximate.iter().filter(|m| exact.contains(&m.id)).count();
        recall_sum += overlap as f32 / TOP_K as f32;
    }
    let recall = recall_sum / QUERIES as f32;
    assert!(
        recall >= 0.9,
        "recall@10 {recall:.3} is below the 90% quality gate against brute force"
    );
}

#[tokio::test]
async fn added_ids_round_trip_through_search() {
    let hnsw = HnswIndex::new(8);
    let mut ids = Vec::new();
    for i in 0..10_u128 {
        let id = Uuid::from_u128(i);
        hnsw.add(
            id,
            Embedding::from_slice(&[i as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .await
        .unwrap();
        ids.push(id);
    }
    let query = Embedding::from_slice(&[9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = hnsw.search(&query, 10).await.unwrap();
    assert_eq!(hits.len(), 10, "every added id is reachable");
    assert!(
        hits.iter().all(|m| ids.contains(&m.id)),
        "the ids that come back are exactly the ids that went in"
    );
    assert_eq!(
        hits.first().map(|m| m.id),
        Some(ids[9]),
        "the query's own direction retrieves rank 0"
    );
}

#[tokio::test]
async fn removed_ids_are_tombstoned_until_rebuild_compacts() {
    let hnsw = HnswIndex::new(8);
    for i in 0..10_u128 {
        hnsw.add(
            Uuid::from_u128(i),
            Embedding::from_slice(&[i as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .await
        .unwrap();
    }
    let removed_ids: [u128; 3] = [1, 4, 7];
    for removed in removed_ids {
        hnsw.remove(Uuid::from_u128(removed)).await.unwrap();
    }
    assert_eq!(hnsw.len(), 7, "len counts live ids only");
    let query = Embedding::from_slice(&[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = hnsw.search(&query, 10).await.unwrap();
    assert!(
        !hits.iter().any(|m| removed_ids.contains(&m.id.as_u128())),
        "the removed ids are absent from results: {:?}",
        hits.iter().map(|m| m.id).collect::<Vec<_>>()
    );
    hnsw.rebuild().unwrap();
    assert_eq!(hnsw.len(), 7, "rebuild keeps the live count");
    let after = hnsw.search(&query, 10).await.unwrap();
    assert_eq!(after.len(), 7, "only live vectors remain after compaction");
    assert!(
        !after.iter().any(|m| removed_ids.contains(&m.id.as_u128())),
        "recall is unaffected by the rebuild"
    );
}

#[tokio::test]
async fn removals_past_the_ratio_compact_tombstones_without_a_rebuild() {
    let hnsw = HnswIndex::new(8);
    for i in 0..4_u128 {
        hnsw.add(
            Uuid::from_u128(i),
            Embedding::from_slice(&[i as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .await
        .unwrap();
    }
    for removed in 0..3_u128 {
        hnsw.remove(Uuid::from_u128(removed)).await.unwrap();
    }
    assert_eq!(
        hnsw.tombstone_count(),
        0,
        "three removals past the default ratio must compact the tombstones away"
    );
    assert_eq!(hnsw.len(), 1, "the live count survives compaction");
    let query = Embedding::from_slice(&[3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = hnsw.search(&query, 10).await.unwrap();
    assert_eq!(
        hits.iter().map(|m| m.id).collect::<Vec<_>>(),
        vec![Uuid::from_u128(3)],
        "the surviving vector still retrieves after the automatic compaction"
    );
}

#[tokio::test]
async fn upserts_past_the_ratio_compact_tombstones_without_a_rebuild() {
    let hnsw = HnswIndex::new(8);
    let id = Uuid::from_u128(7);
    for component in 1..=3_u128 {
        hnsw.add(
            id,
            Embedding::from_slice(&[component as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .await
        .unwrap();
    }
    assert_eq!(
        hnsw.tombstone_count(),
        0,
        "the third same-id upsert must compact its superseded slots away"
    );
    assert_eq!(hnsw.len(), 1, "exactly the freshest vector is live");
    let query = Embedding::from_slice(&[3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = hnsw.search(&query, 1).await.unwrap();
    assert_eq!(
        hits.first().map(|m| m.id),
        Some(id),
        "the upserted id retrieves with its freshest vector after compaction"
    );
}

#[tokio::test]
async fn sub_ratio_tombstones_stay_until_an_explicit_rebuild() {
    let hnsw = HnswIndex::new(8);
    for i in 0..10_u128 {
        hnsw.add(
            Uuid::from_u128(i),
            Embedding::from_slice(&[i as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .await
        .unwrap();
    }
    for removed in [1_u128, 4, 7] {
        hnsw.remove(Uuid::from_u128(removed)).await.unwrap();
    }
    assert_eq!(
        hnsw.tombstone_count(),
        3,
        "three tombstones below the ratio must stay — compaction is \
         threshold-gated, not eager"
    );
    hnsw.rebuild().unwrap();
    assert_eq!(
        hnsw.tombstone_count(),
        0,
        "the explicit rebuild still reclaims them"
    );
}

#[tokio::test]
async fn a_configured_ratio_below_one_compacts_sooner() {
    let vector = Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let tuned = HnswIndex::with_params(
        8,
        HnswParams {
            max_tombstone_ratio: 0.25,
            ..HnswParams::default()
        },
    );
    for i in 0..2_u128 {
        tuned.add(Uuid::from_u128(i), vector.clone()).await.unwrap();
    }
    tuned.remove(Uuid::from_u128(0)).await.unwrap();
    assert_eq!(
        tuned.tombstone_count(),
        0,
        "one tombstone behind two live slots exceeds a 0.25 ratio — the \
         compaction must fire"
    );
    assert_eq!(tuned.len(), 1, "the live vector survives the compaction");

    let default_ratio = HnswIndex::new(8);
    for i in 0..2_u128 {
        default_ratio
            .add(Uuid::from_u128(i), vector.clone())
            .await
            .unwrap();
    }
    default_ratio.remove(Uuid::from_u128(0)).await.unwrap();
    assert_eq!(
        default_ratio.tombstone_count(),
        1,
        "the same debt stays under the default ratio of one tombstone per \
         live vector"
    );
}

#[tokio::test]
async fn same_seed_and_insertion_sequence_build_identical_indexes() {
    let (corpus, queries) = seeded_data(200, 5, 42);
    let build = || async {
        let hnsw = HnswIndex::with_params(DIM, HnswParams::default());
        for (index, vector) in corpus.iter().enumerate() {
            hnsw.add(Uuid::from_u128(index as u128), vector.clone())
                .await
                .unwrap();
        }
        hnsw
    };
    let first = build().await;
    let second = build().await;
    for query in &queries {
        let left = first.search(query, 10).await.unwrap();
        let right = second.search(query, 10).await.unwrap();
        assert_eq!(
            left, right,
            "same seed plus same insert sequence must search identically"
        );
    }
}

#[tokio::test]
async fn a_rebuild_replays_the_same_live_set_identically() {
    let (corpus, queries) = seeded_data(1_000, 20, 7);
    let hnsw = HnswIndex::with_params(
        DIM,
        HnswParams {
            ef_search: 40,
            ..HnswParams::default()
        },
    );
    for (index, vector) in corpus.iter().enumerate() {
        hnsw.add(Uuid::from_u128(index as u128), vector.clone())
            .await
            .unwrap();
    }
    for victim in (0..1_000usize).step_by(5) {
        hnsw.remove(Uuid::from_u128(victim as u128)).await.unwrap();
    }
    assert_eq!(hnsw.len(), 800, "200 tombstones leave 800 live vectors");

    hnsw.rebuild().unwrap();
    let mut first: Vec<Vec<_>> = Vec::with_capacity(queries.len());
    for query in &queries {
        first.push(hnsw.search(query, 10).await.unwrap());
    }

    hnsw.rebuild().unwrap();
    let mut second: Vec<Vec<_>> = Vec::with_capacity(queries.len());
    for query in &queries {
        second.push(hnsw.search(query, 10).await.unwrap());
    }

    assert_eq!(
        first, second,
        "two rebuilds over the same live set must answer identically — \
         the replay is seeded from the configured value, not left to the \
         level RNG's accumulated state"
    );
}

#[tokio::test]
async fn dimension_mismatch_is_rejected_like_the_linear_index() {
    let hnsw = HnswIndex::new(4);
    let wrong_add = hnsw
        .add(Uuid::new_v4(), Embedding::from_slice(&[1.0, 2.0]))
        .await
        .unwrap_err();
    assert!(
        matches!(wrong_add, loopctl::error::LoopError::Memory(_)),
        "add rejects a wrong-dimensional vector: {wrong_add:?}"
    );
    let wrong_search = hnsw
        .search(&Embedding::from_slice(&[1.0, 2.0]), 3)
        .await
        .unwrap_err();
    assert!(
        matches!(wrong_search, loopctl::error::LoopError::Memory(_)),
        "search rejects a wrong-dimensional query: {wrong_search:?}"
    );
}

#[tokio::test]
async fn an_upsert_replaces_the_old_row_without_duplicating() {
    let hnsw = HnswIndex::new(2);
    let id = Uuid::from_u128(1);
    hnsw.add(id, Embedding::from_slice(&[1.0, 0.0]))
        .await
        .unwrap();
    hnsw.add(id, Embedding::from_slice(&[0.0, 1.0]))
        .await
        .unwrap();
    assert_eq!(hnsw.len(), 1, "the upsert leaves one live id");
    let hits = hnsw
        .search(&Embedding::from_slice(&[0.0, 1.0]), 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "the stale slot cannot resurface");
    assert_eq!(
        hits.first().map(|m| (m.id, m.score)),
        Some((id, 1.0)),
        "the replacement vector is the one that answers"
    );
}

#[tokio::test]
async fn search_on_an_empty_index_returns_no_matches() {
    let hnsw = HnswIndex::new(4);
    let hits = hnsw
        .search(&Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0]), 5)
        .await
        .unwrap();
    assert!(hits.is_empty(), "an empty index has nothing to return");
    assert!(hnsw.is_empty(), "is_empty agrees with len");
}

/// A vector almost aligned with the first axis, distinct per `offset`.
///
/// Cosine to the first-axis query stays above `0.999` whatever the offset,
/// so every vector this builds outranks the far-from-query fixture — the
/// shape of the tombstones that must be traversed past, not returned.
fn near_query(offset: f32) -> Embedding {
    let mut vector = vec![0.0_f32; 8];
    vector[0] = 1.0;
    vector[1] = offset;
    Embedding::from_slice(&vector)
}

/// A vector almost anti-aligned with the first axis, distinct per `offset`.
///
/// Cosine to the first-axis query stays below `-0.89` whatever the offset,
/// and the distinct third components keep these live vectors separately
/// scoreable so all ten remain individually returnable.
fn far_from_query(offset: f32) -> Embedding {
    let mut vector = vec![0.0_f32; 8];
    vector[0] = -1.0;
    vector[2] = offset;
    Embedding::from_slice(&vector)
}

#[tokio::test]
async fn tombstones_cannot_crowd_live_results_out_of_the_top_k() {
    let hnsw = HnswIndex::new(8);
    let query = Embedding::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut dead_ids = Vec::new();
    for index in 0..8_u128 {
        let id = Uuid::from_u128(100 + index);
        hnsw.add(id, near_query(0.02 * index as f32)).await.unwrap();
        dead_ids.push(id);
    }
    let mut live_ids = Vec::new();
    for index in 0..10_u128 {
        let id = Uuid::from_u128(200 + index);
        hnsw.add(id, far_from_query(0.05 * index as f32))
            .await
            .unwrap();
        live_ids.push(id);
    }
    for id in &dead_ids {
        hnsw.remove(*id).await.unwrap();
    }
    assert_eq!(
        hnsw.len(),
        10,
        "eight tombstones leave the ten live vectors counted"
    );
    let hits = hnsw.search(&query, 10).await.unwrap();
    let returned: HashSet<Uuid> = hits.iter().map(|m| m.id).collect();
    assert_eq!(
        returned,
        live_ids.iter().copied().collect::<HashSet<_>>(),
        "a deleted neighbour cannot crowd a live one out of the top k — \
         the removal model's promise behind eight nearer tombstones"
    );
}

#[tokio::test]
async fn equal_scores_tiebreak_by_id_like_the_linear_index() {
    let twin = Embedding::from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let later_id = Uuid::from_u128(2_000);
    let earlier_id = Uuid::from_u128(1_000);
    let linear = LinearVectorIndex::new(8);
    let hnsw = HnswIndex::new(8);
    for id in [later_id, earlier_id] {
        linear.add(id, twin.clone()).await.unwrap();
        hnsw.add(id, twin.clone()).await.unwrap();
    }
    let query = Embedding::from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let expected = linear.search(&query, 2).await.unwrap();
    let approximate = hnsw.search(&query, 2).await.unwrap();
    assert_eq!(
        approximate, expected,
        "equal scores must order exactly like the linear index — \
         descending score, then ascending id"
    );
    assert_eq!(
        approximate.first().map(|m| m.id),
        Some(earlier_id),
        "the smaller id wins the tie, not the earlier-inserted slot"
    );
}
