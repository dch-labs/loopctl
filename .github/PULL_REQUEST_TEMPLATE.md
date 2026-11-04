Rules: [`CONTRIBUTING.md`](CONTRIBUTING.md) (summary) · [`AGENTS.md`](AGENTS.md) (canon)

## What changes

<!-- One logical change. If the diff does two things, it should be two
PRs. -->

## Why

<!-- The problem, the doc claim, or the issue this closes. -->

## Checklist

- [ ] `make ci` green from the crate root
- [ ] Provider/wire changes: `make conformance` green (cassette-backed;
      the live e2e gates are owner-side, not contributor-blocking). A red
      golden is a finding: request construction changed — regenerating
      (`UPDATE_GOLDENS=1`) is deliberate and the diff is reviewed like
      code, never a silent fix
- [ ] Public API changed → `CHANGELOG.md` `[Unreleased]` updated (what,
      why, migration if breaking); breaking changes have a maintainer's
      go-ahead from an open issue (see CONTRIBUTING §3)
- [ ] Feature flag added/changed → `default = []` intact, README feature
      table updated in this PR
- [ ] New/changed behavior pinned by a test that states the violated
      contract in its assertion messages
- [ ] Docs: multi-line `///` on new items and fields; `# Errors` on
      public fallible functions
- [ ] No panics in non-test code (`unwrap`/`expect`/indexing/`todo!`)
