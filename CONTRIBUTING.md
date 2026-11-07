# Contributing to loopctl

Thanks for considering a contribution. loopctl is a library in shape —
traits users compose, never an imposed architecture — with every
integration behind an opt-in feature flag and no panics in non-test
code. Contributions that keep those properties are welcome.

## Read this first

- [`AGENTS.md`](AGENTS.md) — the contribution rules this project actually
  enforces: documentation style, code style, the no-panic policy, feature
  flags, semver, testing, and the changelog. Despite the name, it is the
  rulebook for every contributor, human and AI alike. CI rejects code
  that violates them; this file is the friendly summary, not the
  contract.
- [`TESTING.md`](TESTING.md) — the testing doctrine. The one-line version:
  a green test is only evidence if you have seen it red for the right
  reason.

## The short version

1. **One logical change per PR.** Features and refactors do not ride in
   the same diff.
2. **`make ci` green before you push.** From the crate root; it runs
   every build, lint, and test leg CI runs on a Linux runner. CI's
   `vector-e2e` job (real Qdrant/pgvector service containers) and
   `check-darwin` job (a macOS runner compiling the test tree under
   `--features fs_tools`) have no `make ci` legs — they need hardware
   or services a contributor's machine cannot assume. One conditional
   gate sits outside it: changes touching provider wire code should
   also pass `make conformance` (the cassette-backed provider
   contract). The live `make e2e` gates need real credentials and
   models — they are deliberate owner-side checks, never expected of a
   contributor (full gate map: AGENTS.md §8).
3. **Changing public API?** Update `[Unreleased]` in `CHANGELOG.md` in
   the same PR — what changed, why, and the migration if it breaks.
   **Breaking changes additionally need a maintainer's go-ahead before
   the PR** (open an issue first): releases are cut from `master`, so a
   merged break makes the *next* release a minor — it can never ride a
   patch, and it holds every urgent fix behind it until that minor
   ships. Breaking changes are welcome; surprise release-scheduling is
   not.
4. **Adding a feature flag?** `default = []` stays empty; the flag gates
   its module and its optional dependencies; the README feature table
   gains a row in the same PR.
5. **No panics in non-test code.** No `unwrap`, `expect`, `panic!`,
   indexing, or `todo!` — clippy denies them; return typed errors
   instead. Test code is exempt.
6. **Document everything.** Multi-line `///` doc comments on every item,
   including private ones and every field/variant; `# Errors` on public
   functions returning `Result`.
7. **Tests prove behavior.** Sentence names, assertions that state the
   violated contract, deterministic timing, no network access in CI.

## Reporting issues

Open an issue with: what you did, what you expected, what happened, the
crate version and feature set, and a minimal reproduction. For provider
wire problems, include the exact request/response shape if you can — the
cassette corpus pins wire contracts, and drift reports are how it grows.

## Questions

Read the docs first (`docs.rs/loopctl`, [loopctl.rs](https://loopctl.rs));
most answers live there. A well-researched issue is the best question.
