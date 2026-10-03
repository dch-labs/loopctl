//! The layered HNSW graph — construction, greedy descent, and
//! best-first layer search.
//!
//! The graph is a dumb adjacency structure keyed by the [`IdMap`]'s `u32`
//! slots; every distance is computed on demand against the map's vectors
//! with the same [`cosine_similarity`] the reference linear index uses, so
//! an `HnswIndex` and a `LinearVectorIndex` score identical pairs
//! identically and recall comparisons measure *search*, not metrics.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use loopctl::memory::vector::cosine_similarity;

use super::map::IdMap;

/// The adjacency structure: per slot, per level, a neighbour list.
///
/// `neighbors[slot]` has one list per level the slot participates in
/// (`0..=level`), so a slot's top level is
/// `neighbors[slot].len() - 1`. Slot numbers are allocated by the [`IdMap`]
/// and never reused, so the outer vector only grows.
pub(crate) struct Graph {
    /// The `(slot, level)` of the current entry point — the highest node
    /// inserted so far.
    ///
    /// Every search and every insert's greedy descent starts from it; it
    /// moves only when a new node draws a deeper level.
    entry: Option<(u32, usize)>,

    /// Per-slot, per-level neighbour lists.
    ///
    /// `neighbors[slot]` holds one list per level the slot participates
    /// in, so a slot's top level is its list count minus one; the lists
    /// only ever grow until a rebuild replaces the whole graph.
    neighbors: Vec<Vec<Vec<u32>>>,
}

impl Graph {
    /// Create an empty graph with no entry point.
    ///
    /// The state every fresh index and every rebuild starts from: the
    /// first insert becomes the entry point without searching.
    pub(crate) fn new() -> Self {
        Self {
            entry: None,
            neighbors: Vec::new(),
        }
    }

    /// The current entry point, if any node has been inserted.
    ///
    /// `None` exactly when the graph is empty, which search treats as an
    /// empty result rather than an error.
    pub(crate) fn entry(&self) -> Option<(u32, usize)> {
        self.entry
    }

    /// Install `slot` at `level` as the graph's entry point.
    ///
    /// Happens on the very first insert and whenever a new node draws a
    /// level above the current entry point's.
    pub(crate) fn set_entry(&mut self, slot: u32, level: usize) {
        self.entry = Some((slot, level));
    }

    /// Register a new slot carrying `level` levels of empty neighbour
    /// lists.
    ///
    /// Called on insert before any linking, so later writes can target the
    /// slot's levels through [`set_neighbors`](Graph::set_neighbors).
    pub(crate) fn add_slot(&mut self, level: usize) {
        self.neighbors
            .push(vec![Vec::new(); level.saturating_add(1)]);
    }

    /// Replace the neighbour list of `slot` at `layer`.
    ///
    /// A missing slot or layer writes nothing — the insert algorithm only
    /// ever addresses levels it allocated, so the fallback is purely
    /// defensive.
    pub(crate) fn set_neighbors(&mut self, slot: u32, layer: usize, links: Vec<u32>) {
        if let Some(levels) = self.neighbors.get_mut(slot as usize)
            && let Some(list) = levels.get_mut(layer)
        {
            *list = links;
        }
    }

    /// The neighbour list of `slot` at `layer`, or an empty slice.
    ///
    /// Upper-layer lookups on nodes that do not participate in that layer
    /// resolve here, which is what makes the descent code branch-free.
    pub(crate) fn neighbors_of(&self, slot: u32, layer: usize) -> &[u32] {
        self.neighbors
            .get(slot as usize)
            .and_then(|levels| levels.get(layer))
            .map_or(&[], Vec::as_slice)
    }

    /// Append `neighbor` to `slot`'s list at `layer`.
    ///
    /// The back-link half of insertion: every selected neighbour links
    /// back to the new node before any capacity pruning runs.
    pub(crate) fn append_neighbor(&mut self, slot: u32, layer: usize, neighbor: u32) {
        if let Some(levels) = self.neighbors.get_mut(slot as usize)
            && let Some(list) = levels.get_mut(layer)
        {
            list.push(neighbor);
        }
    }

    /// How many neighbours `slot` holds at `layer`.
    ///
    /// The capacity check that triggers re-selection after a back-link;
    /// layer 0 tolerates twice the upper-layer capacity.
    pub(crate) fn link_count(&self, slot: u32, layer: usize) -> usize {
        self.neighbors_of(slot, layer).len()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A distance carrying a total order for the heaps.
///
/// Cosine distance is always finite (`1.0 - clamped cosine`), so the
/// `partial_cmp` fallback is unreachable; the wrapper exists because
/// `f32` itself is not `Ord` and the search needs ordered heaps.
#[derive(Clone, Copy, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Cosine distance from `query` to the vector stored in `slot`.
///
/// A missing slot scores as maximally distant rather than panicking — the
/// graph only ever names slots the map allocated, so the fallback is
/// purely defensive.
pub(crate) fn distance_to(map: &IdMap, query: &[f32], slot: u32) -> f32 {
    map.vector(slot)
        .map_or(2.0, |stored| 1.0 - cosine_similarity(query, stored))
}

/// Draw a fresh node's level: a geometric sample advancing one level per
/// draw with probability `1/m` — the distribution the paper's
/// `floor(-ln(u) * mL)` rule induces, without the float-to-integer
/// conversion the crate's clippy contract denies.
///
/// The RNG is the index's seeded generator, so the same insert sequence
/// with the same seed draws the same levels, which is what keeps the
/// pinned result-level contract: identical search results for the same
/// insert sequence.
pub(crate) fn random_level(rng: &mut fastrand::Rng, m: usize) -> usize {
    let links = u16::try_from(m.max(2)).unwrap_or(u16::MAX);
    let advance_below = 1.0 / f32::from(links);
    let mut level = 0usize;
    while level < MAX_DRAWN_LEVEL && rng.f32() < advance_below {
        level = level.saturating_add(1);
    }
    level
}

/// Deepest level the draw loop may reach.
///
/// A truncation guard, not a distributional parameter: the geometric draw
/// reaches it with probability `m^-64` — never in practice — so the loop
/// terminates unconditionally without a float-to-integer conversion.
const MAX_DRAWN_LEVEL: usize = 64;

/// Walk the graph greedily from `start`, always taking the closer
/// neighbour, until no neighbour improves.
///
/// This is the ef=1 search used on the upper layers: cheap, and only
/// required to land in the target's neighbourhood.
pub(crate) fn greedy(map: &IdMap, graph: &Graph, query: &[f32], start: u32, layer: usize) -> u32 {
    let mut current = start;
    let mut current_dist = distance_to(map, query, current);
    loop {
        let neighbors = graph.neighbors_of(current, layer).to_vec();
        let mut best = None;
        for neighbor in neighbors {
            let candidate_dist = distance_to(map, query, neighbor);
            if candidate_dist < current_dist {
                current_dist = candidate_dist;
                best = Some(neighbor);
            }
        }
        match best {
            Some(neighbor) => current = neighbor,
            None => return current,
        }
    }
}

/// Best-first search on one layer, returning up to `ef` nearest candidates
/// sorted by ascending distance (slot tiebreak).
///
/// Standard HNSW layer search: a min-heap of frontier candidates, a
/// max-heap of the `ef` best results found, expansion stopped once the
/// closest unexplored candidate is farther than the worst retained result.
pub(crate) fn search_layer(
    map: &IdMap,
    graph: &Graph,
    query: &[f32],
    entry_points: &[(f32, u32)],
    ef: usize,
    layer: usize,
) -> Vec<(f32, u32)> {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<std::cmp::Reverse<(Dist, u32)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(Dist, u32)> = BinaryHeap::new();
    for (distance, slot) in entry_points {
        if visited.insert(*slot) {
            candidates.push(std::cmp::Reverse((Dist(*distance), *slot)));
            results.push((Dist(*distance), *slot));
        }
    }
    while results.len() > ef {
        results.pop();
    }
    while let Some(std::cmp::Reverse((candidate_dist, slot))) = candidates.pop() {
        let worst = results.peek().map_or(f32::INFINITY, |(dist, _)| dist.0);
        if candidate_dist.0 > worst && results.len() >= ef {
            break;
        }
        let neighbors = graph.neighbors_of(slot, layer).to_vec();
        for neighbor in neighbors {
            if !visited.insert(neighbor) {
                continue;
            }
            let dist = distance_to(map, query, neighbor);
            if results.len() < ef || dist < worst {
                candidates.push(std::cmp::Reverse((Dist(dist), neighbor)));
                results.push((Dist(dist), neighbor));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut found: Vec<(f32, u32)> = results
        .into_iter()
        .map(|(dist, slot)| (dist.0, slot))
        .collect();
    found.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    found
}

/// Select up to `m` neighbours from candidates using the paper's
/// heuristic, keeping the closest pruned candidates to fill any quota the
/// heuristic leaves unused.
///
/// A candidate survives the heuristic pass when it is closer to the query
/// than to every already-selected neighbour — the diversity rule that
/// keeps links from clustering around a single hub. On random vectors the
/// heuristic can under-fill; the keep-pruned tail keeps the graph
/// connected and the recall gate honest.
fn select_neighbors(map: &IdMap, mut candidates: Vec<(f32, u32)>, m: usize) -> Vec<u32> {
    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let mut selected: Vec<(f32, u32)> = Vec::new();
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    for (distance, slot) in candidates {
        if selected.len() >= m {
            break;
        }
        let dominated = selected.iter().any(|(_, kept)| {
            map.vector(*kept)
                .is_some_and(|kept_vector| distance_to(map, kept_vector, slot) < distance)
        });
        if dominated {
            pruned.push((distance, slot));
        } else {
            selected.push((distance, slot));
        }
    }
    for (distance, slot) in pruned {
        if selected.len() >= m {
            break;
        }
        selected.push((distance, slot));
    }
    selected.into_iter().map(|(_, slot)| slot).collect()
}

/// Insert `slot` (holding `vector`) at `level` into the graph.
///
/// The first insertion becomes the entry point. Otherwise the walk
/// descends greedily through the layers above the new node, then builds
/// the node's links level by level with [`select_neighbors`],
/// back-linking every neighbour and re-selecting any list that exceeds
/// its capacity (`M` above layer 0, `2·M` at layer 0). A level higher
/// than the current entry point's takes over as entry.
pub(crate) fn insert(
    map: &IdMap,
    graph: &mut Graph,
    params: super::HnswParams,
    slot: u32,
    vector: &[f32],
    level: usize,
) {
    graph.add_slot(level);
    let Some((entry_slot, top_level)) = graph.entry() else {
        graph.set_entry(slot, level);
        return;
    };
    let mut entry_point = entry_slot;
    for layer in (level.saturating_add(1)..=top_level).rev() {
        entry_point = greedy(map, graph, vector, entry_point, layer);
    }
    let mut entry_points: Vec<(f32, u32)> =
        vec![(distance_to(map, vector, entry_point), entry_point)];
    for layer in (0..=level.min(top_level)).rev() {
        let found = search_layer(
            map,
            graph,
            vector,
            &entry_points,
            params.ef_construction,
            layer,
        );
        let selected = select_neighbors(map, found.clone(), params.m);
        graph.set_neighbors(slot, layer, selected.clone());
        for neighbor in &selected {
            graph.append_neighbor(*neighbor, layer, slot);
            let capacity = if layer == 0 {
                params.m.saturating_mul(2)
            } else {
                params.m
            };
            if graph.link_count(*neighbor, layer) > capacity {
                let Some(neighbor_vector) = map.vector(*neighbor) else {
                    continue;
                };
                let links = graph.neighbors_of(*neighbor, layer).to_vec();
                let candidates: Vec<(f32, u32)> = links
                    .into_iter()
                    .map(|link| (distance_to(map, neighbor_vector, link), link))
                    .collect();
                let kept = select_neighbors(map, candidates, capacity);
                graph.set_neighbors(*neighbor, layer, kept);
            }
        }
        entry_points = found;
    }
    if level > top_level {
        graph.set_entry(slot, level);
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_DRAWN_LEVEL, random_level};
    use crate::DEFAULT_SEED;

    #[test]
    fn same_seed_draws_the_same_level_sequence() {
        let mut first = fastrand::Rng::with_seed(7);
        let mut second = fastrand::Rng::with_seed(7);
        for _ in 0..512 {
            assert_eq!(
                random_level(&mut first, 16),
                random_level(&mut second, 16),
                "the same seed must replay the same level sequence"
            );
        }
    }

    #[test]
    fn drawn_levels_stay_within_the_bounded_maximum() {
        let mut rng = fastrand::Rng::with_seed(DEFAULT_SEED);
        for _ in 0..4_096 {
            let level = random_level(&mut rng, 16);
            assert!(
                level <= MAX_DRAWN_LEVEL,
                "the draw loop must stay within its truncation guard: {level}"
            );
        }
    }

    #[test]
    fn the_draw_is_biased_toward_level_zero() {
        let mut rng = fastrand::Rng::with_seed(DEFAULT_SEED);
        let zero_level = (0..2_048)
            .filter(|_| random_level(&mut rng, 16) == 0)
            .count();
        assert!(
            zero_level >= 1_024,
            "at m = 16 at least half of the draws must land on level 0, \
             got {zero_level} of 2_048"
        );
    }
}
