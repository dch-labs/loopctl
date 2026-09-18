//! The committed-cassette safety meta-test.
//!
//! Every cassette under `tests/cassettes/` is scanned for anything
//! that must never be committed: credentials, forbidden headers, real
//! user content shapes. The scan is fail-closed — a corpus that
//! shrinks to zero cassettes fails too, because an empty corpus makes
//! this guard vacuous.

#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

mod cassette;

use std::path::Path;

use cassette::scan_for_secrets;

#[test]
fn committed_cassettes_carry_no_secrets() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes");
    let mut scanned = 0;
    let mut violations = Vec::new();
    let mut stack = vec![corpus];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) => panic!("cannot read cassette dir {dir:?}: {e}"),
        };
        for entry in entries {
            let path = entry.expect("a readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "yaml") {
                scanned += 1;
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read cassette {path:?}: {e}"));
                for violation in scan_for_secrets(&text) {
                    violations.push(format!("{path:?}: {violation}"));
                }
            }
        }
    }

    assert!(
        scanned > 0,
        "the corpus must hold at least one cassette — an empty corpus \
        makes this safety scan vacuous"
    );
    assert!(
        violations.is_empty(),
        "cassette secrets must never be committed: {violations:?}"
    );
}
