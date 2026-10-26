//! Doctest census — the `rust,ignore` retirement gate.
//!
//! Two sweeps over every doc-comment fence in `src/`:
//!
//! 1. **Reasoned ignores.** Every `rust,ignore` fence — the `ignore`
//!    token read comma-by-comma, so modifiers like
//!    `rust,ignore,edition2021` cannot escape — must open with a reason
//!    comment as its first body line: `// ignore — <why>`. An ignored
//!    doctest is documentation that never compiles; the reason is what
//!    keeps that honest — and the marker must *start* the line, so a
//!    string or trailing comment carrying the text cannot impersonate a
//!    reason. Prose on the fence line itself (`rust,ignore — why`) voids
//!    the attribute the way rustdoc parses it and silently deletes the
//!    doctest, so that shape is reported too.
//! 2. **Live examples.** Every non-ignored fence that defines a hidden
//!    `fn example` must also *drive* it — a line that actually calls
//!    `example()`, bare or inside `block_on`; a definition-shaped empty
//!    `fn main` and an unrelated `block_on` drive nothing. Rustdoc wraps
//!    a doctest body in its own `fn main`, so a nested `example()`
//!    nothing calls is dead code: the block compiles, "passes", and
//!    asserts nothing — an example that lies.
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

/// Info-string tokens that keep a fence Rust in rustdoc's attribute
/// parser: bare attributes (```` ```no_run ````, ```` ```ignore ````) are
/// Rust code with an attribute, not another language.
const RUST_KEEP_LANGUAGE_ATTRS: [&str; 8] = [
    "no_run",
    "compile_fail",
    "should_panic",
    "ignore",
    "edition2015",
    "edition2018",
    "edition2021",
    "edition2024",
];

/// The first comma-separated token of an info string.
fn first_attr(attrs: &str) -> &str {
    attrs
        .split(',')
        .next()
        .unwrap_or_default()
        .trim_start_matches(' ')
        .trim_end()
}

/// A fence is a Rust doctest candidate when its attributes are empty
/// (plain ```` ``` ````, which rustdoc treats as Rust) or its first token
/// names Rust or a rustdoc attribute — the comma-token reading of an
/// info string, matching rustdoc's own attribute parse.
fn is_rust_fence(attrs: &str) -> bool {
    let trimmed = attrs.trim();
    trimmed.is_empty()
        || first_attr(trimmed) == "rust"
        || RUST_KEEP_LANGUAGE_ATTRS.contains(&first_attr(trimmed))
}

/// A Rust fence is an ignore when its comma-separated attributes carry
/// the exact `ignore` token — `rust,ignore`, `rust,ignore,edition2021`,
/// `rust,no_run,ignore`, bare `ignore`. Token-by-token matching is what
/// closes the modifiers hole: an exact-string comparison let
/// `rust,ignore,edition2021` skip the reason check entirely.
fn is_ignored_fence(attrs: &str) -> bool {
    is_rust_fence(attrs) && attrs.split(',').map(str::trim).any(|attr| attr == "ignore")
}

/// `true` when a Rust fence's `ignore` token carries trailing prose
/// (`rust,ignore — needs credentials`). rustdoc parses comma-separated
/// attributes and stops at the first whitespace, so prose on the fence
/// line voids the `ignore` — and the fence vanishes from the suite
/// without a build error. The shape the reason convention forbids is
/// exactly the shape that silently deletes a doctest, so the census
/// reports it instead of skipping it.
fn has_malformed_ignore_prose(attrs: &str) -> bool {
    is_rust_fence(attrs)
        && attrs.split(',').map(str::trim).any(|attr| {
            attr.strip_prefix("ignore")
                .is_some_and(|rest| rest.starts_with(char::is_whitespace))
        })
}

/// A fence is meant to run when it is a Rust fence that neither the
/// ignore reading nor the malformed reading claims. Non-Rust fences
/// (`text`, `sh`, …) are doctests in no language the census speaks and
/// stay outside both sweeps; the malformed shape is neither executable
/// nor a reasoned ignore — its own gate in the reason test owns it.
fn is_executable_fence(attrs: &str) -> bool {
    is_rust_fence(attrs) && !is_ignored_fence(attrs) && !has_malformed_ignore_prose(attrs)
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

/// Whether `code` shows the example being *called*: the name driven by
/// call parens, at a clean token boundary, and not the `fn example(`
/// definition line (nor a same-named method on a receiver, which is
/// someone else's `example`).
fn calls_example(code: &str) -> bool {
    const NAME: &str = "example";
    code.match_indices(NAME).any(|(index, _)| {
        if !code[index + NAME.len()..].trim_start().starts_with('(') {
            return false;
        }
        let before = &code[..index];
        let word_char = |c: char| c.is_alphanumeric() || c == '_';
        let clean_boundary = before
            .chars()
            .next_back()
            .is_none_or(|c| !word_char(c) && c != '.');
        let definition = before
            .trim_end()
            .strip_suffix("fn")
            .is_some_and(|head| head.chars().next_back().is_none_or(|c| !word_char(c)));
        clean_boundary && !definition
    })
}

/// Whether a fence body defines a hidden example nothing drives.
///
/// Detection reads code lines only (comment prose may name anything),
/// after stripping the hidden-line `#` marker. Driving means *calling*
/// `example()` — bare, or inside `block_on`; a definition-shaped `fn main`
/// with an empty body and an unrelated `block_on(something_else)` prove
/// nothing, and a nested `fn example` no line calls is dropped by
/// rustdoc's own `fn main` wrapper and never runs.
fn example_wrapper_goes_uncalled(body: &[String]) -> bool {
    let defines_example = body
        .iter()
        .any(|line| code_line(line).is_some_and(|code| code.contains("fn example(")));
    let drives_it = body
        .iter()
        .any(|line| code_line(line).is_some_and(calls_example));
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
    let fences = all_fences();
    let unreasoned: Vec<String> = fences
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
    let malformed: Vec<String> = fences
        .iter()
        .filter(|fence| has_malformed_ignore_prose(&fence.attrs))
        .map(|fence| format!("{}:{}", fence.path.display(), fence.line))
        .collect();
    assert!(
        unreasoned.is_empty(),
        "rust,ignore fences without a stated reason — open the block with \
         `// ignore — <why>`, convert it, or delete it: {unreasoned:?}"
    );
    assert!(
        malformed.is_empty(),
        "ignore fences with prose on the fence line — rustdoc parses \
         comma-separated attributes and stops at whitespace, so the prose \
         voids the ignore and the doctest silently leaves the suite; put \
         the reason inside the block as `// ignore — <why>` instead: \
         {malformed:?}"
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
        is_ignored_fence("rust,ignore,edition2021"),
        "modifiers after the ignore token must not read the fence back as executable"
    );
    assert!(is_ignored_fence("rust,no_run,ignore"));
    assert!(
        !is_ignored_fence("rust,edition2021"),
        "a plain modifier is not an ignore"
    );
    assert!(
        !is_ignored_fence("rust,ignore — needs a key"),
        "prose on the fence line voids the ignore token — its own malformed \
         gate owns that shape"
    );
    assert!(
        is_ignored_fence("ignore"),
        "a bare `ignore` attribute is still rustdoc-Rust — an ignored
         doctest in the language the census polices"
    );
    assert!(!is_ignored_fence("text,ignore"), "non-Rust is outside");
    assert!(has_malformed_ignore_prose("rust,ignore — needs a key"));
    assert!(
        has_malformed_ignore_prose("rust,no_run,ignore — needs a key"),
        "the prose arm is found wherever the ignore token sits"
    );
    assert!(!has_malformed_ignore_prose("rust,ignore"));
    assert!(
        !has_malformed_ignore_prose("rust,ignore2021"),
        "a fused suffix is a malformed token, not ignore-with-prose"
    );
    assert!(is_executable_fence(""), "a plain fence is Rust by default");
    assert!(is_executable_fence("rust,no_run"));
    assert!(
        is_executable_fence("no_run"),
        "a bare rustdoc attribute is Rust — dropping it would silently
         shrink the sweep"
    );
    assert!(!is_executable_fence("rust,ignore"), "an ignore never runs");
    assert!(
        !is_executable_fence("rust,ignore,edition2021"),
        "modifiers cannot smuggle an ignore into the runnable count"
    );
    assert!(
        !is_executable_fence("rust,ignore — needs a key"),
        "the malformed shape runs nothing — rustdoc drops the fence"
    );
    assert!(
        !is_executable_fence("sh"),
        "a shell fence is not a Rust doctest"
    );
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
    let empty_main = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# fn main() {}".to_string(),
    ];
    let unrelated_block_on = vec![
        "# async fn example() {".to_string(),
        "work().await;".to_string(),
        "# }".to_string(),
        "# fn main() {".to_string(),
        "# block_on(some_other_future());".to_string(),
        "# }".to_string(),
    ];
    let method_call = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# let done = thing.example();".to_string(),
    ];
    let comment_only = vec![
        "// call fn example() from main".to_string(),
        "let x = 1;".to_string(),
    ];
    assert!(example_wrapper_goes_uncalled(&dead));
    assert!(!example_wrapper_goes_uncalled(&wired));
    assert!(!example_wrapper_goes_uncalled(&async_driven));
    assert!(!example_wrapper_goes_uncalled(&no_example));
    assert!(
        example_wrapper_goes_uncalled(&empty_main),
        "a definition-shaped `fn main` that never calls the example \
         leaves it dead — the driver must be a call"
    );
    assert!(
        example_wrapper_goes_uncalled(&unrelated_block_on),
        "an unrelated `block_on` drives a different future, not the example"
    );
    assert!(
        example_wrapper_goes_uncalled(&method_call),
        "a `.example()` on a receiver is someone else's method — the \
         hidden wrapper still goes uncalled"
    );
    assert!(
        !example_wrapper_goes_uncalled(&comment_only),
        "prose inside a comment names no wrapper — only code lines count"
    );
}

#[test]
fn the_census_reports_live_totals() {
    // `make doc-census` runs this test with --nocapture and prints the
    // line it emits: totals from the census parser — plain `` ``` ``
    // fences counted (rustdoc's Rust default), non-Rust and malformed
    // fences excluded — where a grep over ```` ```rust ```` lines counted
    // neither the plain fences nor anything the classifier rejects.
    let planted = [
        ("empty attrs run", "", true, false),
        ("explicit rust runs", "rust", true, false),
        ("no_run compiles", "rust,no_run", true, false),
        ("ignored counts separately", "rust,ignore", false, true),
        (
            "modifiers cannot escape the ignore",
            "rust,ignore,edition2021",
            false,
            true,
        ),
        (
            "prose fences run nothing",
            "rust,ignore — why",
            false,
            false,
        ),
        ("shell is not Rust", "sh", false, false),
    ];
    let counts = |attrs: &str| {
        (
            usize::from(is_executable_fence(attrs)),
            usize::from(is_ignored_fence(attrs)),
        )
    };
    for (what, attrs, expect_runnable, expect_ignored) in planted {
        let (runnable, ignored) = counts(attrs);
        assert_eq!(
            (runnable, ignored),
            (usize::from(expect_runnable), usize::from(expect_ignored)),
            "the totals classifier mis-reads {what}: {attrs:?}"
        );
    }
    let fences = all_fences();
    let runnable: usize = fences
        .iter()
        .filter(|fence| is_executable_fence(&fence.attrs))
        .count();
    let ignored: usize = fences
        .iter()
        .filter(|fence| is_ignored_fence(&fence.attrs))
        .count();
    let malformed: usize = fences
        .iter()
        .filter(|fence| has_malformed_ignore_prose(&fence.attrs))
        .count();
    assert_eq!(
        malformed, 0,
        "malformed fences must fail the reason gate, never reach the totals"
    );
    println!(
        "doc blocks: {} | runnable fences: {} | rust,ignore: {}",
        runnable + ignored,
        runnable,
        ignored
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
