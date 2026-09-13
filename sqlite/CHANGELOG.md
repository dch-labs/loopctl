# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- **`SqliteMemoryStore`** — a persistent [`LoopMemory`](https://docs.rs/loopctl/latest/loopctl/memory/trait.LoopMemory.html) backend over a local SQLite database (WAL journaling, `NORMAL` sync, five-second busy timeout; `rusqlite` with the `bundled` feature so no system SQLite is required). Every [`MemoryEntry`](https://docs.rs/loopctl/latest/loopctl/memory/struct.MemoryEntry.html) field round-trips — `SystemTime` stamps at millisecond resolution (sub-millisecond precision is truncated), tags as a JSON text column, `relevance` on the `REAL` column — and memory survives a process restart by reopening the same file. Retrieval loads every entry and ranks it in Rust with loopctl's shared `score_entry` — the same entries match, in the same order with the same tie-breaking, as `InMemoryStore`/`FileMemoryStore`; an FTS5 (`porter unicode61`) index is maintained on every write but not consulted by retrieval — ranking every entry is what guarantees the parity contract. Retrieval stamps land in a durable `access_stamps` table (upsert, newest-wins) rather than per-instance memory, so they are shared across every store instance on the same file and survive a restart; `consolidate()` consumes every pending stamp inside its transaction (fold, then decay, merge, prune via loopctl's shared pass) and rewrites the table and the full-text index in that same transaction — a stamp is either folded by a pass or survives as a row for the next one, never lost to a retrieve racing a consolidate. Multi-process concurrency: open several stores against the same file — WAL coordinates them. Version-pinned to `loopctl` (`=0.3.x`) until the trait surface stabilizes. Pinned by `opening_an_existing_database_is_idempotent`, `entries_survive_a_restart_with_every_field_intact`, `retrieval_ranks_full_text_matches_first`, `retrieval_respects_the_limit`, `consolidation_prunes_and_the_delete_is_durable`, `len_counts_rows_and_respects_deletes`, `two_stores_on_one_file_coordinate_through_wal`, and `an_in_memory_database_round_trips_without_the_filesystem`.
