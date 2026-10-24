# Testing

loopctl's test system has three hermetic layers plus a deliberate-act
recording mode. Nothing in CI ever touches the network.

## 1. Principles

Tests are executable specifications: names carry intent, fixtures carry
the oracle, and a green run must mean something provable. A rule the
suite cannot test is a rule the suite cannot protect — when behavior
gets a doc promise, it gets a pin in the same change.

## 2. Layers

| Layer | What it pins | Where |
|---|---|---|
| Unit + integration tests | Behavior on our own seams (mock clients, direct emitter calls) | `src/**/tests`, `tests/*.rs` |
| Golden request bytes | Outbound wire shape (canonical JSON of built requests) | `tests/golden_requests.rs` |
| Cassette replay | Every byte a provider accepted, served back byte-exact | `tests/cassette*.rs`, `tests/providers/`, `tests/cassettes/` |
| **Conformance battery** | The behavioral contract every `ApiClient` must uphold | `tests/common/conformance.rs`, `tests/provider_conformance.rs` |

## 3. Cassettes: record/replay (`tests/cassettes/`)

A cassette is a YAML recording of real HTTP interactions — the exact
request bytes a provider accepted and the exact response it returned.
Replay registers each as a byte-exact mock: a client that builds one
byte differently misses every mock and fails. That miss is the
outbound-drift guard.

## 4. Recording is a deliberate act

Recording never happens in CI and never happens by accident: set
`LOOPCTL_CASSETTE=record` **and** `LOOPCTL_E2E=1` (plus real provider
credentials — except local endpoints like Ollama, which need only the
two flags) and run the drivers in `tests/provider_e2e.rs`. Everything
is scrubbed before disk: credentials never enter a file; generated ids
become deterministic placeholders; response headers are allowlisted.
`tests/cassette_safety.rs` scans the committed corpus in CI.

## 5. Error fixtures and the response gate

A rate limit cannot be ordered from a provider: the 429 attempt driver
runs the request once and the cassette is written only when the
provider actually answered 429. Error-shaped scenarios follow the same
gate in reverse (`finish_errored` writes only on a real ≥ 400). An
unanswered attempt writes nothing rather than freeze a fiction into
the corpus.

## 6. Provider conformance (`make conformance`)

The suite in `tests/common/conformance.rs` asserts one behavioral
contract per provider over the recorded corpus — five families, each
protecting a documented behavior fix. The fixture is the oracle:
expected numbers, reasons, and retry values are parsed from the
recorded bytes, never hardcoded. Per-provider instantiation is
configuration (a profile names a client builder and dialect); the
battery never changes.

## 7. The five families

- **stream_lifecycle** — exactly one terminal; nothing after it; no
  synthetic stop on a truncated stream (the 0.3.0 rule); balanced
  lanes.
- **usage_reporting** — recorded numbers reach the terminal
  `MessageDelta`; repeated reports merge field-wise by max (the 0.3.x
  latch); unreported never becomes a fabricated number.
- **stop_reason_routing** — tool vs text route identically; tool lanes
  carry the requested identity (Gemini's STOP-with-functionCall wire
  truth rides the named-lane arm).
- **error_taxonomy** — 401/403 → `Auth` (one attempt); 429/503/529 →
  `RateLimit` carrying the recorded `Retry-After`; else typed `Http`.
- **options_honoring** — model override reaches the wire bytes;
  `include_usage` agrees with the scenario; the tool-constraint
  capability probe tells the truth.

## 8. The rule

**A new provider ships with a green conformance run.** Joining means
adding a profile in `tests/provider_conformance.rs` and cassettes for
each applicable family — or an explicit `n/r` reason in the matrix.
Gaps are visible, never silently green: `n/r` cells — no fixture yet, a
fixture that ends before its provider terminal, or a fixture whose shape
cannot decide the family at all (a success recording says nothing about
the 401/403 or 429/503/529 arms) — appear in the matrix and in its gap
list, and any recorded violation fails CI. A family reads `pass` only
where one of the provider's fixtures actually adjudicated it; a check
that had nothing to judge reports the gap instead of a pass, so silence
is never green.

## 9. The matrix

`conformance_matrix_artifact` regenerates
`target/provider-conformance/matrix.md` (contract family × provider)
on every run and the CI test job publishes it as the
`provider-conformance-matrix` artifact. It is generated — never
hand-edit it. Every cell carries its meaning: a legend row states what a
column claims, an unproven family renders `n/r` beside the reason, and a
coverage list names each unadjudicated (provider, family) pair with the
fixture that would close it. Bedrock's column fills in as soon as its
cassettes are recorded.

## 10. Meta-pins

`conformance_catches_a_planted_violation` proves the suite catches and
*names* what it finds: it edits a copy of a recorded fixture in memory
(drops the provider terminal; renames the tool identity), drives the
exact battery CI runs, and asserts the suite reports the right family
with the right detail — with the unmutated fixture verified silent
first. A fixture truncated below its provider terminal is reported as
a fixture-integrity `n/r`, never as a violation: a conformant client
ends a truncated stream in silence (the 0.3.0 rule) — including the
usage a Messages-family stream reports on its opening event, which a
client that never reaches the terminal correctly never surfaces — and
blaming it would be the suite lying in the other direction. Committed
cassettes are never mutated by tests.

## 11. Feature-gated runs

Provider suites compile only with their feature; CI's
`cargo test --all-features` runs every one of them. Locally:

```
make conformance                                   # openai,anthropic,gemini profiles
cargo test --features openai,anthropic,gemini,zai,deepseek,grok --test provider_conformance
cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test providers
```

## 12. Live smoke tests

Live smoke (`LOOPCTL_E2E=1 make e2e`) stays a human act against real
providers, exactly like recording.

## 13. Cassette currency (release discipline)

Provider cassettes are re-recorded at every minor release at minimum
(the release checklist points here). Providers drift silently — wire
shapes change without any API version moving — and a stale corpus
proves stale conformance. Provider drift therefore re-enters the
system only through this deliberate act: re-record, re-run the replay
suites and the conformance battery, and regenerate the matrix. A
release never ships on a corpus older than the release before it.
