# Security Policy

## Supported versions

loopctl is pre-1.0: **the newest release receives fixes**; no backport
branches are maintained for older minor lines. Breaking changes ship in
minor bumps and are recorded in [`CHANGELOG.md`](CHANGELOG.md) with
migration notes.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** on this repository
(Security tab → "Report a vulnerability"). Please do not open a public
issue for anything exploitable.

Include what you found, how it can be triggered, and — if you have it —
a minimal reproduction. A fix and a public advisory (GitHub Security
Advisory, which feeds `cargo audit`/RustSec visibility for the crate)
follow depending on severity.

## Scope notes

- loopctl is a **library**: it never reads credentials from disk beyond
  the documented environment variables each provider feature uses, and
  it makes no network calls unless you construct a provider client.
- Tool implementations run model-requested code by design — hosts are
  expected to gate execution (see the permission middleware and the
  tool-safety shield). Reports about a tool *doing what it was configured
  to do* are working-as-intended unless a containment promise is broken.
- The redaction middleware is defense-in-depth for tool output, not a
  guarantee that no secret can ever reach a provider prompt.
