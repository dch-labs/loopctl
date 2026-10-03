//! The `Uuid ↔ u32` slot map behind
//! [`HnswIndex`](super::HnswIndex) — the translation layer that lets the
//! integer-keyed graph be addressed by
//! [`MemoryEntry`](loopctl::memory::MemoryEntry) ids.

use std::collections::{HashMap, HashSet};

use loopctl::error::LoopError;
use uuid::Uuid;

/// One slot in the map: its live id (if any) and its vector.
///
/// A tombstoned slot keeps its vector — graph traversal still scores it
/// while filtering it from results — but its id is cleared, which is what
/// makes `forward` and [`len`](IdMap::len) agree on the live count.
pub(crate) struct Slot {
    /// The live id occupying this slot, or [`None`] once tombstoned.
    ///
    /// Clearing the id is what tombstoning means at this layer: the
    /// forward map drops the key in the same step, so an id can never
    /// resolve to a dead slot.
    pub(crate) id: Option<Uuid>,

    /// The stored vector components.
    ///
    /// Kept after tombstoning because the graph's neighbour expansion
    /// still scores this slot while walking; a rebuild drops it together
    /// with the slot.
    pub(crate) vector: Vec<f32>,
}

/// The append-only `Uuid → u32` map with tombstoned removal.
///
/// Slots are never reused: `add` allocates the next free `u32`, `remove`
/// tombstones the old slot, and an upsert allocates a fresh slot after
/// tombstoning the old one — so slot numbers are stable for the graph's
/// lifetime and `rebuild` is free to renumber.
pub(crate) struct IdMap {
    /// The slots, in allocation order; indexed by the `u32` key.
    ///
    /// Append-only growth keeps every slot number meaningful for the
    /// graph's adjacency lists; only a rebuild compacts, and it rebuilds
    /// the graph along with the map.
    slots: Vec<Slot>,

    /// Forward lookup: live id to its slot.
    ///
    /// Exactly the live ids — tombstoning removes the key — so this map's
    /// size is the count [`len`](IdMap::len) reports.
    forward: HashMap<Uuid, u32>,

    /// Slots removed since the last rebuild, still traversable in the
    /// graph but excluded from results and `len`.
    ///
    /// The set is what search over-fetches against: a deleted neighbour
    /// must be traversed past without crowding a live one out of the top
    /// `k`.
    tombstones: HashSet<u32>,
}

/// Translate a slot-vector length into the next slot key, refusing to wrap.
///
/// The single decision behind both [`IdMap::insert`]'s bound and the
/// capacity pre-check [`ensure_capacity`](IdMap::ensure_capacity)
/// performs: a length above `u32::MAX` has no slot key, no in-process
/// index reaches it, and the map refuses to wrap the key space silently.
///
/// # Errors
///
/// [`LoopError::Memory`] when `slots_len` exceeds `u32::MAX` — the
/// refusal every caller above translates into its own guarantee.
pub(crate) fn slot_key(slots_len: usize) -> Result<u32, LoopError> {
    u32::try_from(slots_len)
        .map_err(|_| LoopError::Memory("hnsw id space exhausted (over u32::MAX slots)".to_string()))
}

impl IdMap {
    /// Create an empty map.
    ///
    /// No slots, no forward entries, no tombstones — the state a fresh
    /// [`HnswIndex`](super::HnswIndex) and every `rebuild` start from.
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            forward: HashMap::new(),
            tombstones: HashSet::new(),
        }
    }

    /// Refuse one more insert before any tombstoning happens.
    ///
    /// [`add`](super::HnswIndex) calls this ahead of an upsert's
    /// `remove`+`insert` pair so the overflow error cannot destroy the
    /// caller's previous entry; the check and the insert share one write
    /// lock, so nothing can consume the last slot between them.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when one more slot would overflow the `u32`
    /// key space — surfaced by `add` before the upsert tombstones
    /// anything.
    pub(crate) fn ensure_capacity(&self) -> Result<(), LoopError> {
        slot_key(self.slots.len()).map(|_| ())
    }

    /// Append one live slot for `id`, returning its `u32` key.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] if the id space overflows `u32` — a bound no
    /// in-process index reaches, but one the map refuses to wrap silently.
    pub(crate) fn insert(&mut self, id: Uuid, vector: Vec<f32>) -> Result<u32, LoopError> {
        let slot = slot_key(self.slots.len())?;
        self.slots.push(Slot {
            id: Some(id),
            vector,
        });
        self.forward.insert(id, slot);
        Ok(slot)
    }

    /// Tombstone the slot holding `id`, returning the slot number.
    ///
    /// A missing id is a `None` — removal stays idempotent for the same
    /// reason [`VectorIndex::remove`](loopctl::memory::vector::VectorIndex::remove)
    /// is. The slot's id is cleared so the forward map and the live count
    /// stay in agreement.
    pub(crate) fn remove(&mut self, id: Uuid) -> Option<u32> {
        let slot = self.forward.remove(&id)?;
        self.tombstones.insert(slot);
        if let Some(entry) = self.slots.get_mut(slot as usize) {
            entry.id = None;
        }
        Some(slot)
    }

    /// The vector stored in `slot` (live or tombstoned — graph traversal
    /// still scores tombstoned slots; result filtering excludes them).
    ///
    /// Returning tombstoned vectors is deliberate: neighbour expansion
    /// mid-walk must keep scoring dead slots for graph stability, while
    /// the result filter — not this accessor — excludes them.
    pub(crate) fn vector(&self, slot: u32) -> Option<&[f32]> {
        self.slots
            .get(slot as usize)
            .map(|entry| entry.vector.as_slice())
    }

    /// The live id stored in `slot`, if the slot has not been tombstoned.
    ///
    /// This is the results-side check: a slot whose id survived is live,
    /// so search can resolve and filter in one lookup.
    pub(crate) fn id(&self, slot: u32) -> Option<Uuid> {
        self.slots.get(slot as usize).and_then(|entry| entry.id)
    }

    /// Whether `slot` has been tombstoned since the last rebuild.
    ///
    /// Search consults this while filtering candidates; the graph itself
    /// never does — dead slots stay traversable by design.
    pub(crate) fn is_tombstoned(&self, slot: u32) -> bool {
        self.tombstones.contains(&slot)
    }

    /// The number of tombstoned slots still traversable in the graph.
    ///
    /// Feeds search's over-fetch: the candidate list must be sized to
    /// skip past dead slots and still surface `k` live ones.
    pub(crate) fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// The number of live ids — the count
    /// [`len`](loopctl::memory::vector::VectorIndex::len) reports.
    ///
    /// Tombstoned slots are excluded because the trait's `len` contract
    /// counts distinct live ids; a rebuild shrinks the slot vector back
    /// to this count.
    pub(crate) fn len(&self) -> usize {
        self.forward.len()
    }

    /// Whether the map holds no live ids.
    ///
    /// Tombstones alone do not count — an index holding only dead slots
    /// is empty for every caller's purposes, and `search` short-circuits
    /// on this.
    pub(crate) fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Every live `(slot, id, vector)` in allocation order.
    ///
    /// This is the input a rebuild replays: same order, same seed, same
    /// graph.
    pub(crate) fn live(&self) -> Vec<(u32, Uuid, &[f32])> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let id = entry.id?;
                Some((
                    match u32::try_from(index) {
                        Ok(slot) => slot,
                        Err(_) => return None,
                    },
                    id,
                    entry.vector.as_slice(),
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// slot_key maps every representable length onto itself.
    ///
    /// Zero is the first slot a fresh map allocates and `u32::MAX` is the
    /// last key the space can hold, so both ends translate without error.
    #[test]
    fn slot_key_maps_every_representable_length_onto_itself() {
        assert_eq!(slot_key(0).unwrap(), 0, "the first slot key is zero");
        assert_eq!(
            slot_key(u32::MAX as usize).unwrap(),
            u32::MAX,
            "the last representable length still has a slot key"
        );
    }

    /// slot_key refuses lengths the `u32` key space cannot hold.
    ///
    /// The refusal is the seam `add` consults before tombstoning an
    /// upsert's previous slot, so a refusal never destroys an entry.
    #[test]
    fn slot_key_refuses_lengths_past_the_u32_bound() {
        assert!(
            matches!(slot_key(usize::MAX), Err(LoopError::Memory(_))),
            "a length above u32::MAX must error, never wrap"
        );
    }
}
