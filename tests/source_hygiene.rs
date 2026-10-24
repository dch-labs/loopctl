//! Source-hygiene ratchet: the repo's shipped artifacts carry no
//! planning metadata, and `.rs` files find structure in code
//! boundaries, never in comment dividers.
//!
//! Task identifiers, roadmap references and future-shipping prose are
//! banned from every file by the contribution rules; the corpus predates
//! the strictest reading, so this ratchet fails on any *new* leak: the
//! scan covers the core crate's sources and tests, every workspace
//! companion crate, build and CI artifacts, and the testing policy —
//! loudly, so a directory the scan cannot open fails the ratchet rather
//! than vanishing from it — and the patterns it hunts are assembled from
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
///
/// Trackers mint ids shaped like this and nothing else does, so the
/// prefix-letter-dash-digit triple at a word boundary is the shape
/// shipped text must never carry; the letters live as individual
/// `char`s so this file itself never spells the tokens it hunts.
const ID_PREFIXES: [char; 3] = ['L', 'T', 'R'];

/// Files whose id-shaped strings *in code* are data under test
/// (redaction inputs), not leaked metadata.
///
/// The exemption is per line kind, never per file: a comment in an
/// allowlisted file is metadata like anywhere else and stays under the
/// rule — the gating lives in [`scanned_for_ids`].
const TASK_ID_ALLOWLIST: &[&str] = &["tests/redacting_middleware.rs"];

/// The source roots the `.rs` scans cover: the core crate plus every
/// workspace member.
///
/// A root that is absent scans empty — the documented optional case —
/// while a present root that will not open fails the ratchet. The
/// companion crate under `crates/vector` is excluded from the workspace
/// and from every CI feature set, so its text ships nowhere scanned.
const SCAN_ROOTS: &[&str] = &[
    "src",
    "tests",
    "examples",
    "crates/derive",
    "crates/sqlite",
    "crates/hnsw",
    "crates/client",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Collect every `.rs` file under `dir`, recursively.
///
/// Traversal is loud: a present directory that cannot be opened, or an
/// entry the OS refuses to name, fails the scan — silently skipping
/// either is how forbidden text hides. A missing directory is the one
/// quiet case (an optional root scans empty). Symlinks are never
/// descended and never collected (`Path::is_dir` follows them, so an
/// ancestor-pointing link would re-traverse the same tree), and a
/// directory named `target` is build output, not shipped source.
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
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            collect_rust(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `true` when `line` carries a planning identifier.
///
/// The shape is a reserved prefix letter, a dash, and a digit, with the
/// letter starting a word; the boundary leg keeps an ordinary word
/// containing one of the letters (or a URL path segment) from reading
/// as an id.
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

/// `true` when `line` is a comment divider.
///
/// Structure in this codebase comes from module, `impl`, and function
/// boundaries and from blank lines — never from a hand-drawn rule — so
/// a comment whose opening run of decoration characters is long enough
/// to read as one is a divider and is flagged.
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
/// release or version.
///
/// Shipped text states what ships now; a promise about a cut nobody
/// has scheduled is nobody's contract and no test can hold it, so the
/// phrase table is matched case-insensitively across every scanned
/// file — prose and docs alike.
fn names_future_shipping(line: &str) -> bool {
    let lower = line.to_lowercase();
    future_shipping_phrases()
        .iter()
        .any(|phrase| lower.contains(phrase))
}

/// `true` when `line` is a comment — `//`, `///`, or `//!` spellings.
///
/// Deliberately syntactic and line-leading: the rule is "trimmed line
/// starts with a double slash", with no attempt to spot slash pairs
/// inside string literals — a data line cannot claim the comment lane's
/// exemptions by containing a URL.
fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// `true` when `line` is under the planning-id rule, given that its
/// file is (or is not) on the data-under-test allowlist.
///
/// The allowlist covers id-shaped *strings in code* — redaction inputs
/// the middleware's tests feed through scrubbing — and nothing wider:
/// a comment in such a file is metadata like anywhere else and stays
/// scanned, which is the exact hole the file-wide exemption used to be.
fn scanned_for_ids(allowlisted: bool, line: &str) -> bool {
    !allowlisted || is_comment(line)
}

fn rust_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in SCAN_ROOTS {
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

fn scan(path: &Path, hits: &mut Vec<String>, probe: &impl Fn(&str) -> bool) {
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
        let allowlisted = TASK_ID_ALLOWLIST.contains(&relative(&file).as_str());
        scan(&file, &mut leaks, &|line| {
            scanned_for_ids(allowlisted, line) && has_task_id(line)
        });
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
        scan(&file, &mut dividers, &is_separator);
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
        scan(&file, &mut narration, &names_future_shipping);
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
            scan(&path, &mut leaks, &has_task_id);
        }
    }
    // A missing workflows directory stays an empty scan; a present one
    // the OS will not open fails the ratchet instead of hiding files.
    let workflows = root().join(".github").join("workflows");
    if workflows.exists() {
        let entries = std::fs::read_dir(&workflows)
            .unwrap_or_else(|err| panic!("cannot scan {}: {err}", workflows.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|err| {
                panic!("cannot read an entry of {}: {err}", workflows.display())
            });
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "yml" || ext == "yaml")
            {
                scan(&path, &mut leaks, &has_task_id);
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

/// A fresh scratch tree named for `name`, guarded by the process pid.
///
/// Recreated empty on every call — a prior aborted run's residue
/// cannot leak into the next assertion — and named per test so the
/// binary's threads running different pins never share (or trash) one
/// tree.
fn scratch_tree(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "loopctl-source-hygiene-{name}-{}",
        std::process::id()
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The part of `path` under `base`, slash-joined.
///
/// The normalization lets an assertion spell its expected list as a
/// literal that reads identically on every platform.
#[cfg(unix)]
fn under(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/")
}

#[test]
fn missing_optional_dir_scans_empty() {
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
    // A present path that cannot be opened as a directory is an error,
    // not an absent root: forbidden text hides behind silent skips.
    collect_rust(&file, &mut files);
    std::fs::remove_dir_all(&base).unwrap();
}

#[cfg(unix)]
#[test]
fn symlinked_dirs_and_files_are_not_followed() {
    let base = scratch_tree("links");
    let real = base.join("real");
    std::fs::create_dir_all(&real).unwrap();
    std::fs::write(base.join("root.rs"), "// root\n").unwrap();
    std::fs::write(real.join("inner.rs"), "// inner\n").unwrap();
    std::os::unix::fs::symlink(&real, base.join("linked-dir")).unwrap();
    std::os::unix::fs::symlink(base.join("root.rs"), base.join("linked.rs")).unwrap();

    let mut files = Vec::new();
    collect_rust(&base, &mut files);
    let mut names: Vec<String> = files.iter().map(|path| under(&base, path)).collect();
    names.sort();
    assert_eq!(
        names,
        vec!["real/inner.rs", "root.rs"],
        "links must neither duplicate the tree (a sibling link re-collects \
         it; an ancestor link would re-traverse forever) nor masquerade as \
         sources: {names:?}"
    );
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn target_directories_are_not_scanned() {
    let base = scratch_tree("target");
    std::fs::create_dir_all(base.join("src")).unwrap();
    std::fs::create_dir_all(base.join("tests/consumer/target/debug/out")).unwrap();
    std::fs::write(base.join("src/live.rs"), "// live\n").unwrap();
    std::fs::write(
        base.join("tests/consumer/target/debug/out/generated.rs"),
        "// generated\n",
    )
    .unwrap();

    let mut files = Vec::new();
    collect_rust(&base, &mut files);
    assert_eq!(
        files,
        vec![base.join("src/live.rs")],
        "build output is not shipped source — scanning generated files \
         spends the ratchet on text no author wrote"
    );
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn allowlisted_files_still_scan_comment_lines() {
    // Samples are assembled at runtime so this file stays free of the
    // token it demonstrates.
    let id = format!("{}-7", ID_PREFIXES[0]);
    let data_line = format!("let row = \"{id} ships\";");
    let comment = format!("// {id} ships — a leaked note");
    let ordinary = "let row = \"an ordinary line\";";
    assert!(
        has_task_id(&data_line) && has_task_id(&comment),
        "both samples carry an id-shaped token by construction"
    );

    // Outside an allowlist every line is under the rule…
    assert!(
        scanned_for_ids(false, &data_line) && scanned_for_ids(false, &comment),
        "unallowlisted files scan every line"
    );
    // …inside one, id-shaped *data* stays exempt (it is the input under
    // test) and comments never were.
    assert!(
        !scanned_for_ids(true, &data_line),
        "id-shaped redaction input in an allowlisted file stays data"
    );
    assert!(
        scanned_for_ids(true, &comment) && has_task_id(&comment),
        "an id in a comment of an allowlisted file is metadata — the \
         file-wide exemption was broader than its documented purpose"
    );
    assert!(
        !scanned_for_ids(true, ordinary),
        "ordinary non-comment lines of allowlisted files keep their exemption"
    );
}
