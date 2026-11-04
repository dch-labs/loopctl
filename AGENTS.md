# AGENTS.md — loopctl

> ← [`README.md`](README.md) (project entry point) ·
> [`TESTING.md`](TESTING.md) (the testing doctrine)
>
> **Purpose:** the authoritative rules for any agent (human or AI)
> contributing to loopctl — how the crate is styled, tested, versioned,
> and released. Read this before writing code, opening a PR, or cutting
> a release. A rule marked **enforced** has a mechanical check behind
> it, and the text names it — a `make ci` leg, a lint level, a test. A
> rule with no check behind it is a review expectation, and says so.
>
> **Source of truth:** `Cargo.toml` (features, version, MSRV, lint
> levels), `.clippy.toml` (thresholds), the `Makefile` (the CI leg list),
> `CHANGELOG.md` (release log). Where this file quotes a value one of
> those defines, that file wins — read it rather than trusting prose here.

---

## The contract

1. A framework in genre, a **library in shape** — traits users compose; loopctl never owns your main loop or imposes an architecture.
2. **`default = []`** — `cargo add loopctl` pulls zero optional dependencies; every integration is opt-in.
3. **No panics in non-test code** — `unwrap`, `expect`, `panic!`, indexing, `todo!` are clippy-denied.
4. **Everything is documented** — multi-line `///` on every item, field, and variant, public or private; `# Errors` on public fallible functions.
5. **Semver is honest** — breaking changes ship in minors (pre-1.0), never hidden in patches, never avoided when the API is wrong.
6. **The engine is a sans-IO state machine + an IO driver** — the machine decides, the driver performs, `finalize()` owns every exit.
7. **Tests are evidence** — a green test counts only if you have seen it red for the right reason.
8. **`make ci` is the gate** — the Makefile target is the contract; `make -n ci` prints the live leg list.
9. **The changelog records changes, not history** — what differs for a reader of the version, never the planning story.

---

## 1. Project identity

The thesis — built for the small/local model regime, the compensation
layer on by default since 0.4.0 — and the shape claim ("a framework in
genre, a library in shape: loopctl never owns your main loop") are
stated once, publicly, in [`README.md`](README.md)'s *Why loopctl*
section. What contributors and agents additionally hold to are the
engineering principles behind that story:

| Principle | In practice |
|-----------|-------------|
| **Library in shape** | "Library, not framework" in the classic phrasing: never require our builders, macros, or architecture. Provide traits; let users compose. |
| **Zero-cost abstractions** | Generics (static dispatch) on the hot path; `dyn` only where ergonomics demand it (`Arc<dyn ContextCompactor>`). |
| **Feature-flagged** | `default = []`. An integration that pulls into `default` is a blocking review issue. |
| **No panics in production** | Clippy-denied, not advised (see §3). |
| **Documented everything** | `#![warn(missing_docs)]` plus the full doc canon (see §4). |
| **Semver-honest** | The bump reflects the true nature of the change (see §6). |

## 2. Code style

### Formatting — rustfmt defaults, enforced by `make fmt`

The tree ships **no rustfmt configuration** — no `rustfmt.toml` anywhere
— so rustfmt's stable defaults *are* the rules and `cargo fmt --all --
--check` is the whole enforcement. Notable effective values: `max_width`
100, `tab_spaces` 4, `fn_call_width` 60, and
`use_field_init_shorthand` **false** (`S { alpha: alpha }` survives
formatting as written). `make lint` auto-formats.

### Naming

Types `UpperCamelCase`; functions/methods `snake_case`; constants and
statics `SCREAMING_SNAKE_CASE`; modules and feature flags `snake_case`;
generic parameters — one uppercase letter for the primary param (`C` for
`ApiClient`), descriptive names for the rest (`Ctx`, `Req`). Clippy
disallows `foo`/`bar`/`baz`/`quux` even in tests.

### Modules

- No `mod.rs` — the `foo.rs` + `foo/` pattern.
- One concept per module; no line budget. Split on responsibility count,
  never on length — a 1,500-line file holding one concept is fine; a
  200-line file mixing three is not. Doc comments never count as bloat.
- Feature-gated modules use positive `#[cfg(feature = "...")]` guards at
  the declaration site.

### Function bodies

- Short enough to read without scrolling (~30–40 lines) as a soft
  target. Split when a function does more than one thing, no longer
  fits, or you feel the urge to add a structural comment — that urge is
  the function asking to be broken up. Nesting ≥3 levels says the same.
  Helpers are free: five 20-line functions beat one 100-line function.
- **No inline comments in bodies.** Explanation belongs in the item's
  `///` doc; names and small helpers carry the rest. The rare exception
  is one short comment for a genuinely non-obvious decision (an upstream
  provider workaround, a seemingly-redundant clone's reason) — more than
  one per function means split it.
- No separator lines (`// ---`, `// ===`) — structure comes from
  `mod`/`impl`/`fn` boundaries and blank lines.

## 3. The no-panic policy

The most strictly enforced rule. `Cargo.toml [lints.clippy]` denies
`unwrap_used`, `expect_used`, `panic` (which covers `todo!` and
`unreachable!`), `indexing_slicing`, `arithmetic_side_effects`, and more
— the full deny set is the manifest, and CI's `-D warnings` makes even
`pedantic` warnings fatal. In practice nothing clippy says is advisory.

| Instead of | Use |
|------------|-----|
| `result.unwrap()` | `result.map_err(|e| LoopError::…)?` |
| `option.unwrap()` | `option.ok_or(LoopError::…)?` |
| `vec[i]` | `vec.get(i).ok_or(LoopError::…)?` |
| `panic!("…")` / `expect("never")` | `Err(LoopError::…)`; `unreachable!()` is banned too |
| Bare `+ - * / %` | `checked_*` / `saturating_*`; for floats `f64::mul_add(x, m, 0.0)` |

**Test code is exempt** (`cfg_attr(test, allow(...))` in `lib.rs`
covers `#[cfg(test)]` modules, the `tests/` tree, and the `testing`
feature). For genuinely unknown-failure external code, the established
pattern is `catch_unwind` + downcast at the tool-dispatch boundary —
follow that shape.

## 4. The documentation canon

### Required everywhere

- Every public item has `///` docs (`missing_docs` is active); private
  items get them when non-obvious.
- **Every field, variant, and trait method — public or private — carries
  its own doc.** No bare fields, no bare variants.
- Every module starts with a `//!` block: what it does, key types, a
  quick-start example if user-facing.
- `# Errors` on public fallible functions; `# Panics` only where a panic
  is possible (it should not be); `# Safety` on `unsafe`.

### Multi-line shape (required)

Summary sentence ending in a period, blank `///` line, 2–4 sentences of
elaboration (origin, default, invariant, neighbors via `[Links]`). A
bare single-line doc is acceptable only for re-exports and trivial
redirects — constants, methods, and private items get the full
treatment. A trivial field still states its purpose **and unit**.

**Run the audit before claiming conformance** — authors and reviewers
have both verified presence while missing shape (28 single-line blocks
once passed a style review). Over every changed `.rs` file:

```
awk 'BEGIN{prev=""} {if ($0 ~ /^[[:space:]]*\/\/\//) {cur="doc"} else {cur="other"}; if (cur=="doc" && prev!="doc") {startline=FNR; starttext=$0; blocklen=1} else if (cur=="doc" && prev=="doc") {blocklen++}; if (cur=="other" && prev=="doc") {if (blocklen==1) print FILENAME":"startline": "starttext; blocklen=0}; prev=cur}' <files>
```

Every hit is a violation: keep the summary, add the blank separator,
write the elaboration — never fix a hit by deleting the doc. Scope is
production code; the test tree is exempt (see below).

### Doctests

- Every public function should have a runnable doctest; the live count
  and the honest-minimum census are `cargo test --doc --all-features` and
  `make doc-census` — never a number pinned in prose.
- `rust,ignore` is the honest-survivor hatch, not the default: only for
  blocks that genuinely cannot run in-process, with the reason as the
  **first body line** (`// ignore — <why>`; the reason lives inside
  because rustdoc drops fences whose info string carries prose). The
  census test fails any unreasoned ignore.
- Feature-gated blocks use a hidden `# #[cfg(feature = "…")]` line — CI
  runs `--all-features`, so gated blocks still execute.
- A hidden example function must be **called** — rustdoc wraps bodies in
  its own `fn main`, so an uncalled example compiles, "passes", and
  asserts nothing; the census fails it.
- `rust,no_run` compiles without executing; `rust,ignore` does neither.

### Test documentation

Tests read like executable specifications. The name carries the intent;
comments only for what code cannot say (a replayed byte sequence's shape,
why a magic number is the threshold, which wire quirk a fixture
reproduces). No separator comments; no `///` on `#[test]` fns; helpers
get a one-line doc only where the name doesn't already say it.

### No planning metadata in any file

Doc comments and every artifact (manifests, Makefile, CI YAML, scripts)
never contain task ids, roadmap references, planning-doc citations, or
process narrative. The test: would a maintainer who has never seen the
planning docs find this useful? Code and artifacts describe what they do
— history lives in the commit log, planning in the planning repo.

## 5. Feature flags

- **`default = []` and it stays that way.** The intent is concrete: a
  user who brings their own `ApiClient` must not pay for an HTTP stack
  they did not ask for. When the rule and a correct, simple API conflict,
  the API wins and the rule is amended honestly in the CHANGELOG — never
  preserve a wrong shape to keep the default literal.
- With no features, the engine runs non-streaming (`TurnMode::
  NonStreaming` via `create_message`); the streaming path sits behind
  `streaming`, implied by every HTTP provider feature.
- One concept per flag; flags map to modules; dependencies are explicit
  (`bar = ["foo"]`); never `#[cfg(not(feature))]` for required behavior.
- **The feature list lives in `Cargo.toml` and nowhere else**;
  `README.md` carries its user-facing copy and must be updated in the
  same PR that touches a flag. A list kept in prose drifts — this repo's
  own history proves it.

## 6. Architecture

- **Trait-first** for every capability: define the trait → reference
  impl → integration point → feature-gate → document + example.
- **Object safety** where `dyn` is needed: `Pin<Box<dyn Future + Send>>`
  returns, stored as `Arc<dyn Trait>` — the established shape for async
  capability traits.
- **Dispatch:** generics on the hot path (per-turn, per-tool, per-event);
  `Arc<dyn Trait>` on the cold path; never `dyn` where the concrete type
  is known.
- **Observer vs hook:** observers notify (`()`, always available); hooks
  control (allow/block/ask, `hooks` feature). Control flow never goes into
  observer.
- **The engine's state machine is explicit and legal transitions only**
  — custom `Loop` implementations respect the same graph.
- **Errors are typed, never `String`:** `LoopError` (framework),
  `ApiError` (provider), `ToolError` (tools), all `thiserror`. Soft
  errors (unknown tool, tool failure) record and continue; hard errors
  (API failure, max turns) terminate the run.

## 7. Versioning, semver, MSRV

| Change | Bump | Breaking |
|--------|------|----------|
| Bug fix, internal refactor, post-release hardening | Patch `0.x.Y` | No |
| New feature, new module, defaulted trait method | Minor `0.Y.0` | No |
| Removed/changed/renamed API (pre-1.0) | Minor `0.Y.0` | Yes |
| Breaking change (post-1.0) | Major `Y.0.0` | Yes |

**Architecture outranks compatibility.** When the right design requires
a break, make it; a correct, simple API beats preserving a wrong call
site. The only obligations: an honest bump (never a break in a patch)
and a CHANGELOG migration note. `#[deprecated]` is not a gate and not a
ritual.

**MSRV is 1.98** (`Cargo.toml` `rust-version`; `.clippy.toml` mirrors
it). MSRV rises only in minors. CI runs on the **pinned** toolchain, not
floating stable — a toolchain roll must not flip a lint verdict mid-PR.
Widening the matrix is a deliberate act: add the job, then the rule.

## 8. Testing and CI

`make ci` **is** the pipeline — the target and its prerequisites in the
Makefile are the contract; print the live list with `make -n ci` and
trust no count kept in prose. The legs cover: fmt; `package-check` (the
tarball ships no crate-root document beyond `CHANGELOG.md`,
`README.md`, `TESTING.md` — untracked-but-not-ignored files *do* ship
unless excluded); check on all/default/no-default feature sets (the
stripped build is its own contract: a helper that serves only a
feature-gated call site is dead code under no-features); clippy across
targets, features, and every companion crate; tests (default,
all-features, doctests, companions); docs under `-D warnings`; every
example binary; the minimal-feature probes (`redaction-minimal`,
`search_tools-minimal`, `digest-canonical`, `derive-consumer`) that see
feature-graph shapes all-features runs cannot; the darwin/windows
cross-compile legs; and the vector/client companion checks.

Deliberately **not** in `ci`: `make conformance` (provider contract over
recorded cassettes), `make doc-census`, and the live `e2e` gates.

Conventions: one canonical `#[cfg(test)] mod tests` per file — no
provenance modules; mocks from the `testing` feature; no network or real
keys, ever (real integration tests only on explicit owner request);
names like `test_<what>_<condition>`. Complexity thresholds
(cognitive 30, body 150 lines, 7 args, etc.) live in `.clippy.toml` —
that file is the list.

## 9. Changelog

Keep a Changelog; an `[Unreleased]` section always exists at the top.
Every public-API PR updates it; breaks carry a migration note. **An
entry states changes between versions and nothing else**: what was
added/changed/fixed/removed, migrations, API names, wire shapes, pinning
test names — never task ids, roadmap tags, review-round provenance, or
process narrative. Enforced by the source-hygiene ratchet
(`tests/source_hygiene.rs`), which fails the build on planning metadata
inside `CHANGELOG.md`. Development history belongs to the commit log.

## 10. Releases

Checklist: `make ci` green → rename `[Unreleased]` to `[VERSION] -
DATE` and open a fresh `[Unreleased]` → bump the version **and every
companion crate together with their exact `=X.Y.Z` core pins** (the pins
are exact; skipping them leaves companions unresolvable) → README table
if flags changed → docs/examples for new public API → migration notes →
MSRV reviewed → cassettes re-recorded (every minor at minimum) →
`cargo package --list` holds nothing internal-only. Cadence: 3–4
releases a year; a red CI holds the release; patches ship for critical
fixes.

## 11. Dependencies

`default-features = false` on everything optional-capable; feature-gated
code uses `optional = true` deps enabled by the flag (`providers =
["dep:reqwest", …]`). Core (always compiled): `serde`, `serde_json`,
`thiserror`, `tokio`, `tracing`, `uuid`, `futures`, `fastrand`,
`tokio-util`, `serde_repr`. Caret requirements; tokio pinned to a
specific minimum; `*` never. Proc macros live in `loopctl-derive`, not
core. Core with no features compiles on core deps alone.

## 12. Git

**Agents never commit or push.** No command that writes to the commit
history or a remote, no exceptions — not a fix-up commit, not "CI needs
it", not "trivial". An agent's change ends in the working tree: edit,
verify, report, stop. If a task appears to require a commit to proceed,
surface that and wait for the human.

Contributors: branches `feat/…`, `fix/…`, `release/v0.X.0`; Conventional
Commits (`feat`/`fix`/`docs`/`refactor`/`test`/`chore`, `!` for
breaking); one logical change per PR; public-API PRs update
`[Unreleased]`; flag PRs update the README table; CI green before merge.

## 13. Forbidden patterns

**CI rejects** (manifest-level denies; `pedantic` is warn and promoted
to error): `unwrap`/`expect`/`panic!`/`todo!`/`unreachable!`/indexing
in non-test code; bare arithmetic (`checked_*`/`saturating_*`/
`mul_add`); `match_same_arms` (merge with or-patterns);
`redundant_clone` (move it); `clone_on_ref_ptr` (move or borrow);
`let_underscore_must_use` (bind, handle, or `?` it); `default = [...]`
in the manifest; non-optional deps in feature-gated code; undocumented
public items, fields, or variants; inline body comments; separator
lines; planning metadata in the changelog; disallowed placeholder names.

**Review pushes back on:** `Arc<Mutex<…>>` on the hot path;
`Box::pin(async move)` without need; `String` errors;
`unwrap_or_default()` where `None` matters; >7 parameters; direct
`tokio::spawn` in framework code (use `CancelSignal` + structured
concurrency).

## 14. Telemetry

Metric names are a published contract — once a name reaches a dashboard
it cannot be renamed or rescaled without a migration note. The contract
is `src/metrics/registry.rs`: one row per signal with kind and a
**closed** label set.

- **Every declared row has an emission site** (`tests/
  registry_emission_audit.rs` fails otherwise; the audit strips comments
  so prose cannot bless a dead row); **every emission is declared**
  (`tests/emission_catalog_contract.rs`). Declare a metric *with* its
  `tracing::debug!`, never before — the gap was once 35 of 65 rows wide.
- Units ride the **name** (`…duration_ms`), never the event field; a
  counter's numeric field matches the name's last segment; no label
  outside the declared set (`model`, `session_id`, `request_id` are the
  three always-present exemptions); `_pct` means percent 0–100, with the
  fraction→percent conversion at the emission site, asserted on both
  scales.
- Durations flow through `managers.clock()` (a frozen test clock must
  freeze an emitted duration); where no clock handle reaches, the basis
  is stated where the metric is described. Record on the path the metric
  describes — reporting "happened" before it happens is a false
  statement about a run. `ContextManager::build_telemetry` derives; it
  never emits.

## 15. Quick reference

README's Development section carries the public summary; these are the
working commands contributors and agents actually run:

```bash
make ci              # the full gate — the target is the contract
make -n ci           # print the live leg list; trust no prose count
make test            # default, all-features, doctests, companions
make lint            # auto-format (does not check)
make doc-census      # doctest census: runnable fences + reasoned ignores
make conformance     # provider contract over cassettes (not in ci)
make e2e             # live gates (opt-in, never in ci)

cargo test --features hooks,testing        # fast iteration
cargo test --doc --all-features            # doctests only
cargo clippy --features hooks -- -D warnings
```

*This document is the single source of truth for contribution rules.
Where it quotes something another file defines — a version, an MSRV, a
lint level, a feature list, the CI leg list — **that file wins**: this
one is prose, and prose drifts. It is committed to the repository and
kept out of the published tarball via `Cargo.toml`'s `exclude`.*