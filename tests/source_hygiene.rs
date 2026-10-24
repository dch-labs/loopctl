//! Source-hygiene ratchet: the repo's shipped artifacts carry no
//! planning metadata, and `.rs` files find structure in code
//! boundaries, never in comment dividers.
//!
//! Task identifiers, roadmap references and future-shipping prose are
//! banned from every file by the contribution rules; the corpus predates
//! the strictest reading, so this ratchet fails on any *new* leak: the
//! scan covers all production and test sources, build and CI artifacts,
//! and the testing policy — and the patterns it hunts are assembled from
//! pieces at runtime so this file stays free of the tokens it hunts.

#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

/// The prefix letters whose dash-digit suffix marks a planning or
/// review-round id.
const ID_PREFIXES: [char; 3] = ['L', 'T', 'R'];

/// Files whose id-shaped strings are data under test (redaction
/// inputs), not leaked metadata.
const TASK_ID_ALLOWLIST: &[&str] = &["tests/redacting_middleware.rs"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Collect every `.rs` file under `dir`, recursively.
fn collect_rust(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_rust(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `true` when `line` carries a planning identifier: one of the
/// reserved prefix letters, a dash, and a digit, at a word boundary.
fn has_task_id(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    chars.iter().enumerate().any(|(index, c)| {
        ID_PREFIXES.contains(c)
            && (index == 0 || !chars[index - 1].is_alphanumeric())
            && index + 2 < chars.len()
            && chars[index + 1] == '-'
            && chars[index + 2].is_ascii_digit()
    })
}

/// `true` when `line` is a comment divider — structure comes from
/// module/impl/function boundaries and blank lines, never from a
/// hand-drawn rule.
fn is_separator(line: &str) -> bool {
    let Some(rest) = line.trim().strip_prefix("//").map(str::trim_start) else {
        return false;
    };
    // A comment whose opening run of decoration makes it a divider.
    let leading_run =
        |chars: &[char], min: usize| rest.chars().take_while(|c| chars.contains(c)).count() >= min;
    leading_run(&['-', '=', '*', '#', '+', '~'], 3)
        || leading_run(&['\u{2014}', '\u{2013}', '\u{2500}'], 2)
}

/// The release-shaped phrases that announce future shipping.
///
/// Assembled at runtime from pieces so this file stays free of the
/// tokens it hunts.
fn future_shipping_phrases() -> &'static [String] {
    static PHRASES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PHRASES.get_or_init(|| {
        let when = [("lat", "er"), ("fut", "ure"), ("ne", "xt")];
        let width = ["", "minor ", "major "];
        let unit = ["release", "version"];
        let mut phrases = Vec::new();
        for (prefix, suffix) in when {
            for qualifier in width {
                for tail in unit {
                    phrases.push(format!("{prefix}{suffix} {qualifier}{tail}"));
                }
            }
        }
        phrases
    })
}

/// `true` when `line` promises that something ships in a not-yet-cut
/// release or version. Shipped text states what ships now; a promise
/// about a cut nobody has scheduled is nobody's contract, and no test
/// can hold it.
fn names_future_shipping(line: &str) -> bool {
    let lower = line.to_lowercase();
    future_shipping_phrases()
        .iter()
        .any(|phrase| lower.contains(phrase))
}

fn rust_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in ["src", "tests", "examples"] {
        collect_rust(&root().join(dir), &mut files);
    }
    files.sort();
    files
}

fn relative(path: &Path) -> String {
    path.strip_prefix(root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn scan(path: &Path, hits: &mut Vec<String>, probe: fn(&str) -> bool) {
    let content = std::fs::read_to_string(path).unwrap();
    for (number, line) in content.lines().enumerate() {
        if probe(line) {
            hits.push(format!(
                "{}:{}: {}",
                relative(path),
                number + 1,
                line.trim()
            ));
        }
    }
}

#[test]
fn no_planning_ids_ship_in_rust_sources() {
    let mut leaks = Vec::new();
    for file in rust_files() {
        if TASK_ID_ALLOWLIST.contains(&relative(&file).as_str()) {
            continue;
        }
        scan(&file, &mut leaks, has_task_id);
    }
    assert!(
        leaks.is_empty(),
        "planning identifiers shipped into the sources — a reader of any \
         single file must not need the project's task numbering:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn no_comment_dividers_in_rust_sources() {
    let mut dividers = Vec::new();
    for file in rust_files() {
        scan(&file, &mut dividers, is_separator);
    }
    assert!(
        dividers.is_empty(),
        "comment divider lines in `.rs` files — structure comes from \
         module/impl/function boundaries and blank lines:\n{}",
        dividers.join("\n")
    );
}

#[test]
fn no_future_shipping_prose_in_rust_sources() {
    // Sensitivity first: a sentence assembled from the same pieces must
    // trip the probe, or the silent scan below proves nothing.
    let planted = format!(
        "/// the real transports arrive in a {}{} {}.",
        "lat", "er", "release"
    );
    assert!(
        names_future_shipping(&planted),
        "the probe must catch the phrase family it exists to hunt: {planted}"
    );
    assert!(
        !names_future_shipping("/// in-process is the only constructor shipped."),
        "stating what ships now is not roadmap narration"
    );

    let mut narration = Vec::new();
    for file in rust_files() {
        scan(&file, &mut narration, names_future_shipping);
    }
    assert!(
        narration.is_empty(),
        "roadmap narration shipped into the sources — shipped text states \
         what ships now, not which unreleased cut is meant to add it:\n{}",
        narration.join("\n")
    );
}

#[test]
fn build_and_ci_artifacts_ship_no_planning_ids() {
    let mut leaks = Vec::new();
    for name in ["Makefile", "Cargo.toml"] {
        let path = root().join(name);
        if path.exists() {
            scan(&path, &mut leaks, has_task_id);
        }
    }
    let workflows = root().join(".github").join("workflows");
    if let Ok(entries) = std::fs::read_dir(workflows) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "yml" || ext == "yaml")
            {
                scan(&path, &mut leaks, has_task_id);
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "planning identifiers shipped into build/CI artifacts:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn testing_md_is_clean() {
    let testing = std::fs::read_to_string(root().join("TESTING.md")).unwrap();
    let mut leaks = Vec::new();
    for (number, line) in testing.lines().enumerate() {
        if has_task_id(line) {
            leaks.push(format!("TESTING.md:{}: {}", number + 1, line.trim()));
        }
    }
    assert!(
        leaks.is_empty(),
        "planning identifiers shipped into the testing policy:\n{}",
        leaks.join("\n")
    );
}
