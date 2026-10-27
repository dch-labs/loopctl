//! Doctest census — the `rust,ignore` retirement gate.
//!
//! Two sweeps over every doc-comment fence in `src/`:
//!
//! 1. **Reasoned ignores.** Every ignored Rust fence — the `ignore`
//!    word read one at a time, comma- or space-separated, so modifiers
//!    like `rust,ignore,edition2021` and `rust ignore` cannot escape —
//!    must open with a reason comment as its first body line:
//!    `// ignore — <why>`. An ignored doctest is documentation that
//!    never compiles; the reason is what keeps that honest — and the
//!    marker must *start* the line, so a string or trailing comment
//!    carrying the text cannot impersonate a reason. A fence rustdoc
//!    extracts nothing from — any non-ASCII character in the info
//!    string, like the em dash of `rust,ignore — why` — silently loses
//!    the doctest instead, and its own gate reports that shape.
//! 2. **Live examples.** Every non-ignored fence that defines a hidden
//!    `fn example` must also *drive* it — a line of real code that
//!    actually calls `example()` (a string or comment naming the call
//!    drives nothing), bare or inside `block_on`; a definition-shaped empty
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

/// The info string's attribute words, split the way rustdoc splits an
/// info string: on commas *and* whitespace. `rust,ignore` and
/// `rust ignore` carry the same two words (probed against rustc 1.98:
/// both are doctests, and both of them ignored).
fn fence_words(attrs: &str) -> impl Iterator<Item = &str> + '_ {
    attrs
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|word| !word.is_empty())
}

/// The first attribute word of an info string — the language slot.
fn first_attr(attrs: &str) -> &str {
    fence_words(attrs).next().unwrap_or("")
}

/// A fence is a Rust doctest candidate when its attributes are empty
/// (plain ```` ``` ````, which rustdoc treats as Rust) or the language
/// slot names Rust or a rustdoc attribute — bare ```` ```no_run ```` is
/// Rust with an attribute, not another language.
fn is_rust_fence(attrs: &str) -> bool {
    let first = first_attr(attrs);
    first.is_empty() || first == "rust" || RUST_KEEP_LANGUAGE_ATTRS.contains(&first)
}

/// A Rust fence is an ignore when its attribute words carry the exact
/// `ignore` word — `rust,ignore`, `rust ignore`, bare `ignore`,
/// `rust,no_run,ignore`. Word-by-word matching is what closes both
/// holes an exact-string comparison left: modifiers cannot escape
/// (`rust,ignore,edition2021` is still an ignore), and a space between
/// words cannot hide one (`rust ignore` is still an ignore). The word
/// must be exact: rustdoc 1.98 runs a fence whose only spare word is
/// `ignore-foo`, so the census counts it as running too.
fn is_ignored_fence(attrs: &str) -> bool {
    is_rust_fence(attrs)
        && !is_invisible_to_rustdoc(attrs)
        && fence_words(attrs).any(|word| word == "ignore")
}

/// A Rust-intent fence rustdoc silently drops: ANY non-ASCII character
/// in the info string — the em dash of ```` ```rust,ignore — needs
/// credentials ````, a stray accent — leaves rustdoc extracting no
/// doctest at all. The example is not ignored and not failing; it is
/// simply absent from the suite, and any reason it carried is compiled
/// nowhere. Probed against rustdoc 1.98: ASCII junk words and
/// punctuation (`unknownjunk`, `a:b`, `(x)`, `1bad`) are inert and the
/// doctest survives, but every non-ASCII character, wherever it sits,
/// erases the fence. A foreign-language fence with a strange info
/// string was never a doctest, so only Rust-intent fences are checked.
fn is_invisible_to_rustdoc(attrs: &str) -> bool {
    is_rust_fence(attrs) && !attrs.is_ascii()
}

/// A fence is meant to run when it is a Rust fence rustdoc can see that
/// is not ignored. Non-Rust fences (`text`, `sh`, …) are doctests in no
/// language the census speaks and stay outside both sweeps; an invisible
/// fence is neither executable nor a reasoned ignore — its own gate in
/// the reason test owns it.
fn is_executable_fence(attrs: &str) -> bool {
    is_rust_fence(attrs) && !is_invisible_to_rustdoc(attrs) && !is_ignored_fence(attrs)
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

/// Whether `code` (string- and comment-free text, from [`code_only`])
/// shows the example being *called*: the name driven by call parens, at
/// a clean token boundary, and not the `fn example(` definition line
/// (nor a same-named method on a receiver, which is someone else's
/// `example`).
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

/// The code only from one doctest body line: the tail is cut at a `//`
/// comment, and double-quoted string literals are dropped, so text that
/// merely *mentions* `example()` — inside prose or a `println!` — is not
/// a call. Line-local and ordinary-case: raw strings, multi-line
/// strings, and `/* */` blocks are past this scanner's appetite; no
/// doctest body in this repo uses one, and an `example(` hidden inside
/// those is the same text-match fiction this cut exists to refuse.
fn code_only(line: &str) -> String {
    let mut code = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => loop {
                match chars.next() {
                    None | Some('"') => break,
                    Some('\\') => {
                        chars.next();
                    }
                    Some(_) => {}
                }
            },
            '/' if chars.peek() == Some(&'/') => break,
            other => code.push(other),
        }
    }
    code
}

/// Whether a fence body defines a hidden example nothing drives.
///
/// Detection reads code lines only (comment prose may name anything),
/// after stripping the hidden-line `#` marker and cutting string
/// literals and comments. Driving means *calling* `example()` in code —
/// bare, or inside `block_on`; a definition-shaped `fn main` with an
/// empty body, an unrelated `block_on`, and text merely naming the call
/// prove nothing, and a nested `fn example` no line calls is dropped by
/// rustdoc's own `fn main` wrapper and never runs.
fn example_wrapper_goes_uncalled(body: &[String]) -> bool {
    let defines_example = body
        .iter()
        .any(|line| code_line(line).is_some_and(|code| code.contains("fn example(")));
    let drives_it = body
        .iter()
        .any(|line| code_line(line).is_some_and(|code| calls_example(&code_only(code))));
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

/// The length of a line's leading backtick run where it can open or
/// close a fence: three or more, or `None`.
fn fence_marker_len(line: &str) -> Option<usize> {
    let run = line.chars().take_while(|&c| c == '`').count();
    (run >= 3).then_some(run)
}

/// Collect every doc-comment fence in `path`, under the CommonMark
/// rules rustdoc's markdown follows. A fence opens at three or more
/// backticks and closes only on a run at least the opener's length,
/// indented at most three spaces and followed by whitespace alone — a
/// ``` line is content inside a ```` fence, and a marker with trailing
/// text never closes anything. A doc block ends at the first non-doc
/// line and an unterminated fence ends with it: rustdoc extracts each
/// doc block as its own markdown, where an open fence auto-closes at
/// the end. Carrying `open` across the boundary lets the NEXT block's
/// opener read as a closer and inverts marker parity for the rest of
/// the file — the live `src/testing.rs` shape, where a module-doc fence
/// left open before `pub mod determinism;` swallowed the next item's
/// example whole — so a pending fence is flushed at the boundary.
fn collect_file_fences(path: &PathBuf) -> Vec<Fence> {
    let Ok(text) = std::fs::read_to_string(path) else {
        panic!(
            "cannot scan {}: the census must read every file it walks",
            path.display()
        );
    };
    let mut fences = Vec::new();
    let mut open: Option<(usize, usize, String, Vec<String>)> = None;
    for (index, raw) in text.lines().enumerate() {
        let Some(content) = doc_line(raw) else {
            if let Some((line, _run, attrs, body)) = open.take() {
                fences.push(Fence {
                    path: path.clone(),
                    line,
                    attrs,
                    body,
                });
            }
            continue;
        };
        let indented = content.trim_start();
        let too_far_in = content.len() - indented.len() > 3;
        let marker = (!too_far_in).then(|| fence_marker_len(indented)).flatten();
        if let Some(run) = marker {
            let closes = open.as_ref().is_some_and(|(_line, opened, _a, _b)| {
                run >= *opened && indented[run..].trim().is_empty()
            });
            if closes {
                if let Some((line, _opened, attrs, body)) = open.take() {
                    fences.push(Fence {
                        path: path.clone(),
                        line,
                        attrs,
                        body,
                    });
                }
            } else if let Some((_line, _run, _attrs, body)) = open.as_mut() {
                body.push(content.to_string());
            } else {
                open = Some((
                    index + 1,
                    run,
                    indented[run..].trim().to_string(),
                    Vec::new(),
                ));
            }
        } else if let Some((_line, _run, _attrs, body)) = open.as_mut() {
            body.push(content.to_string());
        }
    }
    if let Some((line, _run, attrs, body)) = open.take() {
        fences.push(Fence {
            path: path.clone(),
            line,
            attrs,
            body,
        });
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
        .filter(|fence| is_invisible_to_rustdoc(&fence.attrs))
        .map(|fence| format!("{}:{}", fence.path.display(), fence.line))
        .collect();
    assert!(
        unreasoned.is_empty(),
        "ignored fences without a stated reason — open the block with \
         `// ignore — <why>`, convert it, or delete it: {unreasoned:?}"
    );
    assert!(
        malformed.is_empty(),
        "fences rustdoc extracts nothing from — any non-ASCII character \
         in the info string (an em-dash reason on the fence line) deletes \
         the doctest without a compile error; keep the fence line plain \
         and move the reason inside, as `// ignore — <why>`: {malformed:?}"
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
        is_ignored_fence("rust ignore"),
        "rustdoc splits info strings on whitespace as well as commas — a \
         space-separated ignore is still an ignored doctest"
    );
    assert!(
        is_ignored_fence("no_run ignore"),
        "bare attributes compose with spaces too"
    );
    assert!(
        !is_ignored_fence("rust,edition2021"),
        "a plain modifier is not an ignore"
    );
    assert!(
        !is_ignored_fence("rust,ignore — needs a key"),
        "the whole fence is invisible to rustdoc — that gate owns the shape"
    );
    assert!(
        is_ignored_fence("ignore"),
        "a bare `ignore` attribute is still rustdoc-Rust — an ignored
         doctest in the language the census polices"
    );
    assert!(!is_ignored_fence("text,ignore"), "non-Rust is outside");
    assert!(is_invisible_to_rustdoc("rust,ignore — needs a key"));
    assert!(
        is_invisible_to_rustdoc("rust ignore — needs a key"),
        "space-separated attributes do not escape the invisible rule"
    );
    assert!(
        is_invisible_to_rustdoc("rust é"),
        "any non-ASCII character voids the fence — no ignore token required"
    );
    assert!(
        is_ignored_fence("rust,ignore - needs a key"),
        "an ASCII hyphen is inert punctuation: rustdoc 1.98 still extracts \
         this as an ignored doctest, so the reason gate owns it"
    );
    assert!(!is_invisible_to_rustdoc("rust,ignore"));
    assert!(!is_invisible_to_rustdoc("rust,ignore2021"));
    assert!(
        !is_invisible_to_rustdoc("sh — whatever"),
        "a foreign-language fence never entered the suite; there is nothing \
         it could silently lose"
    );
    assert!(
        is_executable_fence("rust,ignore2021"),
        "a fused suffix is no attribute rustdoc knows — it runs as plain Rust"
    );
    assert!(is_executable_fence(""), "a plain fence is Rust by default");
    assert!(is_executable_fence("rust,no_run"));
    assert!(
        is_executable_fence("rust no_run"),
        "a space-separated modifier still runs — rustdoc reads both \
         separators as attribute words"
    );
    assert!(
        is_executable_fence("no_run"),
        "a bare rustdoc attribute is Rust — dropping it would silently \
         shrink the sweep"
    );
    assert!(
        is_executable_fence("rust unknownjunk"),
        "ASCII junk is inert: rustdoc 1.98 doctests the fence as plain Rust"
    );
    assert!(
        is_executable_fence("rust ignore-foo"),
        "rustdoc 1.98 runs a fence whose only spare word is `ignore-foo` — \
         the census counts what the suite runs"
    );
    assert!(!is_executable_fence("rust,ignore"), "an ignore never runs");
    assert!(
        !is_executable_fence("rust ignore"),
        "a space cannot hide an ignore from the reason gate"
    );
    assert!(
        !is_executable_fence("rust,ignore,edition2021"),
        "modifiers cannot smuggle an ignore into the runnable count"
    );
    assert!(
        !is_executable_fence("rust,ignore — needs a key"),
        "an invisible fence runs nothing — rustdoc extracts no doctest at all"
    );
    assert!(
        !is_executable_fence("rust é"),
        "the invisible rule holds without an ignore token"
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
    let string_mention = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# println!(\"remember to call example() someday\");".to_string(),
    ];
    let comment_mention = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# let ready = true; // later: example()".to_string(),
    ];
    let driver_after_a_string = vec![
        "# fn example() {".to_string(),
        "assert_eq!(1 + 1, 2);".to_string(),
        "# }".to_string(),
        "# println!(\"starting\"); example();".to_string(),
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
    assert!(
        example_wrapper_goes_uncalled(&string_mention),
        "a string literal naming the call is prose with quotes, not a driver"
    );
    assert!(
        example_wrapper_goes_uncalled(&comment_mention),
        "a trailing comment naming the call is prose, not a driver"
    );
    assert!(
        !example_wrapper_goes_uncalled(&driver_after_a_string),
        "a real call after a string literal still drives the example"
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
            "fences rustdoc cannot see run nothing",
            "rust,ignore — why",
            false,
            false,
        ),
        (
            "space-separated attributes are still attributes",
            "rust ignore",
            false,
            true,
        ),
        (
            "space-separated modifiers still run",
            "rust no_run",
            true,
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
        .filter(|fence| is_invisible_to_rustdoc(&fence.attrs))
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

/// The census's own view of one synthesized doc file.
fn synthesized_fences(name: &str, content: &str) -> Vec<Fence> {
    let base = scratch_tree(name);
    let file = base.join("synth.rs");
    std::fs::write(&file, content).unwrap();
    let fences = collect_file_fences(&file);
    std::fs::remove_dir_all(&base).unwrap();
    fences
}

#[test]
fn fence_boundaries_follow_the_rules_rustdoc_extracts_by() {
    // A shorter marker is content inside a longer fence, and a marker
    // with trailing text never closes — the CommonMark rules rustdoc's
    // markdown follows.
    let nested = [
        "//! ````",
        "//! # let a = 1;",
        "//! ```",
        "//! # let b = 2;",
        "//! ````",
        "/// ```rust",
        "/// # fn example() {}",
        "/// ```",
    ]
    .join("\n");
    let fences = synthesized_fences("nested", &nested);
    assert_eq!(
        fences.len(),
        2,
        "the 3-backtick line is content of the 4-backtick fence, not a closer"
    );
    assert_eq!(
        fences[0].body.len(),
        3,
        "the short marker rides inside the long fence's body"
    );
    assert_eq!(fences[1].body, vec!["# fn example() {}".to_string()]);
    let glued = [
        "/// ```rust",
        "/// # fn example() {}",
        "/// ``` glued text is not a closer",
        "/// # fn main() { example(); }",
        "/// ```",
    ]
    .join("\n");
    let fences = synthesized_fences("glued", &glued);
    assert_eq!(
        fences.len(),
        1,
        "one fence — a marker with trailing text closes nothing"
    );
    assert!(
        !example_wrapper_goes_uncalled(&fences[0].body),
        "the driver below the false marker is inside the same fence"
    );
    // A doc block ends at the first non-doc line; rustdoc extracts each
    // block as its own markdown, where an open fence auto-closes at the
    // end. Carrying `open` past the boundary lets the next block's
    // opener read as a closer and inverts parity for the rest of the
    // file — the live src/testing.rs shape.
    let split = [
        "//! ```",
        "//! # let open = 1;",
        "",
        "pub mod later;",
        "",
        "/// ```rust",
        "/// # fn example() {}",
        "/// ```",
    ]
    .join("\n");
    let fences = synthesized_fences("split", &split);
    assert_eq!(
        fences.len(),
        2,
        "the unterminated fence flushes at the doc boundary and the next \
         block parses on its own parity"
    );
    assert_eq!(fences[0].attrs, "");
    assert_eq!(fences[0].body, vec!["# let open = 1;".to_string()]);
    assert_eq!(fences[1].attrs, "rust");
    assert_eq!(fences[1].body, vec!["# fn example() {}".to_string()]);
    let at_eof = ["/// ```rust", "/// # let ends = 1;"].join("\n");
    let fences = synthesized_fences("eof", &at_eof);
    assert_eq!(
        fences.len(),
        1,
        "rustdoc auto-closes at the end of the doc block; so does the census"
    );
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
