# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- **`SqliteMemoryStore`** — a persistent [`LoopMemory`](https://docs.rs/loopctl/latest/loopctl/memory/trait.LoopMemory.html) backend over a local SQLite database (WAL journaling, `NORMAL` sync, five-second busy timeout; `rusqlite` with the `bundled` feature so no system SQLite is required). Every [`MemoryEntry`](https://docs.rs/loopctl/latest/loopctl/memory/struct.MemoryEntry.html) field round-trips losslessly — `SystemTime` stamps as milliseconds since the Unix epoch, tags as a JSON text column, `relevance` on the `REAL` column — and memory survives a process restart by reopening the same file. Retrieval goes through an FTS5 full-text index (`porter unicode61`) with a `LIKE` fallback that makes odd queries over-match rather than error, is re-ranked in Rust with loopctl's shared `score_entry` — matched candidates order by the same formula as `InMemoryStore`/`FileMemoryStore`, though recall is token-level (stemmed whole words; a substring-only hit surfaces through baseline fill-up) — and tops the candidate set up from the highest-relevance rows when fewer than the limit matched, mirroring the flat backends' baseline delivery. `consolidate()` runs loopctl's shared consolidation pass (access-log fold, decay, merge, prune) and rewrites the table and the full-text index in one transaction. Multi-process concurrency: open several stores against the same file — WAL coordinates them. Version-pinned to `loopctl` (`=0.3.x`) until the trait surface stabilizes. Pinned by `opening_an_existing_database_is_idempotent`, `entries_survive_a_restart_with_every_field_intact`, `retrieval_ranks_full_text_matches_first`, `retrieval_respects_the_limit`, `consolidation_prunes_and_the_delete_is_durable`, `len_counts_rows_and_respects_deletes`, `two_stores_on_one_file_coordinate_through_wal`, and `an_in_memory_database_round_trips_without_the_filesystem`.
