//! Doctest census — the `rust,ignore` retirement gate.
//!
//! Two sweeps over every doc-comment fence in `src/`:
//!
//! 1. **Reasoned ignores.** Every `rust,ignore` fence must open with a
//!    reason comment as its first body line: `// ignore — <why>`. An
//!    ignored doctest is documentation that never compiles; the reason
//!    is what keeps that honest — and the marker must *start* the line,
//!    so a string or trailing comment carrying the text cannot impersonate
//!    a reason.
//! 2. **Live examples.** Every non-ignored fence that defines a hidden
//!    `fn example` must also *drive* it (a `fn main` caller or a
//!    `block_on`). Rustdoc wraps a doctest body in its own `fn main`, so
//!    a nested `example()` nothing calls is dead code: the block
//!    compiles, "passes", and asserts nothing — an example that lies.
//!
//! Both gates are one-directional: they fail what is broken, not what is
//! merely old. Driving the ignore count down is conversion work
//! (`make doc-census` prints the live totals); these tests guarantee
//! every survivor says why it survives and every "runnable" example
//! actually runs. The tree walk fails loud on an unreadable path —
//! a census that silently skips a subtree reports a clean tree it never
//! read (the source-hygiene ratchet's traversal policy, mirrored).

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

/// The exact fence attribute this census polices.
const IGNORED_FENCE: &str = "rust,ignore";

/// The reason comment a reasoned block opens with — at the START of the
/// line, after doc markers are stripped, so code cannot impersonate it.
const REASON_PREFIX: &str = "// ignore — ";

/// The least prose a reason must carry to count as an explanation.
const MIN_REASON_PROSE: usize = 8;

/// One doc-comment code fence: where it is and what it holds.
struct Fence {
    /// The file holding the fence.
    path: PathBuf,
    /// The fence's opening line, 1-indexed.
    line: usize,
    /// The info string after the opening backticks, trimmed.
    attrs: String,
    /// The body lines with doc markers stripped (what rustdoc feeds the
    /// compiler, comment-visible text only).
    body: Vec<String>,
}

/// A fence is an ignore when its attrs are exactly `rust,ignore`.
///
/// A fence carrying trailing prose is not a doctest at all (rustdoc drops
/// it), and the reason convention forbids that shape — so it is not read
/// as a reasoned ignore either.
fn is_ignored_fence(attrs: &str) -> bool {
    attrs.trim() == IGNORED_FENCE
}

/// A fence is meant to run when its attrs say nothing about ignoring.
fn is_executable_fence(attrs: &str) -> bool {
    !attrs.contains("ignore")
}

/// The reason a first body line states, or `None`.
///
/// The line is a fence-body line (doc markers already stripped by
/// [`doc_line`]): it must *begin* with [`REASON_PREFIX`] — a marker
/// inside a string literal or a trailing comment is code, not a stated
/// reason — and carry at least [`MIN_REASON_PROSE`] characters of prose
/// after it.
fn ignore_reason(first_body_line: &str) -> Option<&str> {
    let reason = first_body_line
        .trim_start()
        .strip_prefix(REASON_PREFIX)?
        .trim();
    (reason.len() >= MIN_REASON_PROSE).then_some(reason)
}

/// Whether a fence body defines a hidden example nothing drives.
///
/// Detection reads code lines only (comment prose may name anything),
/// after stripping the hidden-line `#` marker. A body drives its example
/// with a `fn main` caller or a `block_on`; a nested `fn example` with no
/// driver is dropped by rustdoc's own `fn main` wrapper and never runs.
fn example_wrapper_goes_uncalled(body: &[String]) -> bool {
    let defines_example = body
        .iter()
        .any(|line| code_line(line).is_some_and(|code| code.contains("fn example(")));
    let drives_it = body.iter().any(|line| {
        code_line(line).is_some_and(|code| code.contains("fn main(") || code.contains("block_on"))
    });
    defines_example && !drives_it
}

/// A body line's code text, or `None` for comment lines and empty lines.
fn code_line(line: &str) -> Option<&str> {
    let stripped = line.trim_start().strip_prefix('#').unwrap_or(line).trim();
    if stripped.is_empty() || stripped.starts_with("//") {
        None
    } else {
        Some(stripped)
    }
}

/// The doc-comment content of a line (`/// x` → `x`), or `None` for
/// non-doc lines — doctests live in doc comments and nowhere else.
fn doc_line(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let content = trimmed
        .strip_prefix("//!")
        .or_else(|| trimmed.strip_prefix("///"))?;
    Some(content.strip_prefix(' ').unwrap_or(content))
}

/// Collect every doc-comment fence in `path`.
fn collect_file_fences(path: &PathBuf) -> Vec<Fence> {
    let Ok(text) = std::fs::read_to_string(path) else {
        panic!(
            "cannot scan {}: the census must read every file it walks",
            path.display()
        );
    };
    let mut fences = Vec::new();
    let mut open: Option<(usize, String, Vec<String>)> = None;
    for (index, raw) in text.lines().enumerate() {
        let Some(content) = doc_line(raw) else {
            continue;
        };
        let Some(attrs) = content.strip_prefix("```") else {
            if let Some((_, _, body)) = open.as_mut() {
                body.push(content.to_string());
            }
            continue;
        };
        match open.take() {
            None => open = Some((index + 1, attrs.trim().to_string(), Vec::new())),
            Some((line, attrs_opened, body)) => fences.push(Fence {
                path: path.clone(),
                line,
                attrs: attrs_opened,
                body,
            }),
        }
    }
    fences
}

/// Walk `dir` for `.rs` files, loudly. A missing root is the documented
/// optional case and scans empty; anything else that cannot be read is
/// an error naming its path — unreasoned fences hide behind silent skips.
fn collect_rust(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => panic!("cannot scan {}: {err}", dir.display()),
    };
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("cannot read an entry of {}: {err}", dir.display()));
        let file_type = entry
            .file_type()
            .unwrap_or_else(|err| panic!("cannot classify {}: {err}", entry.path().display()));
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            collect_rust(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Every fence in the crate's sources.
fn all_fences() -> Vec<Fence> {
    let mut files = Vec::new();
    collect_rust(Path::new("src"), &mut files);
    files.sort();
    files.iter().flat_map(collect_file_fences).collect()
}

/// A census scratch tree under the temp dir, unique per process.
fn scratch_tree(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "loopctl-doctest-census-{name}-{}",
        std::process::id()
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn every_ignored_doctest_states_its_reason() {
    let unreasoned: Vec<String> = all_fences()
        .iter()
        .filter(|fence| is_ignored_fence(&fence.attrs))
        .filter(|fence| {
            !fence
                .body
                .first()
                .is_some_and(|first| ignore_reason(first).is_some())
        })
        .map(|fence| format!("{}:{}", fence.path.display(), fence.line))
        .collect();
    assert!(
        unreasoned.is_empty(),
        "rust,ignore fences without a stated reason — open the block with \
         `// ignore — <why>`, convert it, or delete it: {unreasoned:?}"
    );
}

#[test]
fn no_executable_doctest_defines_an_uncalled_example() {
    let dead: Vec<String> = all_fences()
        .iter()
        .filter(|fence| is_executable_fence(&fence.attrs))
        .filter(|fence| example_wrapper_goes_uncalled(&fence.body))
        .map(|fence| format!("{}:{}", fence.path.display(), fence.line))
        .collect();
    assert!(
        dead.is_empty(),
        "doctests that define `fn example` but never call it — rustdoc \
         wraps the body in its own `fn main`, so an undriven example is \
         dead code and the block asserts nothing while looking runnable; \
         end the body with `# fn main() {{ … example(); … }}` (or drive \
         it through `block_on`), or mark the fence `ignore` with a reason: \
         {dead:?}"
    );
}

#[test]
fn the_census_reads_each_reason_shape_correctly() {
    let cases: [(&str, bool); 6] = [
        ("// ignore — needs a live provider key", true),
        ("   // ignore — needs a live provider key", true),
        ("// ignore — why", false),
        (r#"let reason = "// ignore — placeholder";"#, false),
        ("let x = 1; // ignore — trailing comment", false),
        ("// ignoring everything today", false),
    ];
    for (line, expected_reasoned) in cases {
        assert_eq!(
            ignore_reason(line).is_some(),
            expected_reasoned,
            "first body line `{line}` must read as {}",
            if expected_reasoned {
                "reasoned"
            } else {
                "unreasoned"
            }
        );
    }
    let raw = doc_line("//! // ignore — needs a live provider key");
    assert_eq!(
        raw.and_then(ignore_reason),
        Some("needs a live provider key"),
        "a module-doc reason reaches the parser through the doc-marker strip"
    );
    assert!(is_ignored_fence("rust,ignore"));
    assert!(
        !is_ignored_fence("rust,ignore — needs a key"),
        "prose on the fence line is not a doctest at all — rustdoc drops \
         such fences, so the census must not read them as reasoned ignores"
    );
    assert!(!is_executable_fence("rust,ignore"));
    assert!(is_executable_fence(""));
    assert!(is_executable_fence("rust,no_run"));
}

#[test]
fn the_census_reads_each_wrapper_shape_correctly() {
    let dead = vec![
        "# #[cfg(feature = \"testing\")]".to_string(),
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
    ];
    let wired = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# fn main() {".to_string(),
        "# example();".to_string(),
        "# }".to_string(),
    ];
    let async_driven = vec![
        "# async fn example() {".to_string(),
        "run().await;".to_string(),
        "# }".to_string(),
        "# fn main() {".to_string(),
        "# tokio::runtime::Runtime::new().unwrap().block_on(example());".to_string(),
        "# }".to_string(),
    ];
    let no_example = vec!["let x = 1;".to_string()];
    let comment_only = vec![
        "// call fn example() from main".to_string(),
        "let x = 1;".to_string(),
    ];
    assert!(example_wrapper_goes_uncalled(&dead));
    assert!(!example_wrapper_goes_uncalled(&wired));
    assert!(!example_wrapper_goes_uncalled(&async_driven));
    assert!(!example_wrapper_goes_uncalled(&no_example));
    assert!(
        !example_wrapper_goes_uncalled(&comment_only),
        "prose inside a comment names no wrapper — only code lines count"
    );
}

#[test]
fn missing_optional_root_scans_empty() {
    let base = scratch_tree("missing");
    let mut files = Vec::new();
    collect_rust(&base.join("never-created"), &mut files);
    assert!(
        files.is_empty(),
        "a root that is simply not there is the documented optional case: \
         an empty scan, not a failure: {files:?}"
    );
    std::fs::remove_dir_all(&base).unwrap();
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "cannot scan")]
fn traversal_errors_are_not_silently_skipped() {
    let base = scratch_tree("loud");
    let file = base.join("not-a-directory.rs");
    std::fs::write(&file, "// present\n").unwrap();
    let mut files = Vec::new();
    collect_rust(&file, &mut files);
    std::fs::remove_dir_all(&base).unwrap();
}
