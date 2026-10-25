//! Source-hygiene ratchet: the repo's shipped artifacts carry no
//! planning metadata, and `.rs` files find structure in code
//! boundaries, never in comment dividers.
//!
//! Task identifiers, roadmap references and future-shipping prose are
//! banned from every file by the contribution rules; the corpus predates
//! the strictest reading, so this ratchet fails on any *new* leak: the
//! scan covers the core crate's sources and tests, every workspace
//! companion crate, build and CI artifacts, the testing policy, and the
//! changelog —
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

/// Reserved multi-letter id prefixes: an external issue id from the
/// private tracker (this two-letter prefix, a dash, a digit) is
/// planning metadata exactly like the single-letter ids, but the
/// word-boundary leg of [`has_task_id`] cannot see it — the prefix's
/// own letters continue the word, so the dash never lands after a fresh
/// word start. Spelled as the bare prefix, never the id shape: this
/// file hunts its own tokens and must not carry one.
const EXTENDED_ID_PREFIXES: [&str; 1] = ["TX"];

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
    }) || EXTENDED_ID_PREFIXES.iter().any(|prefix| {
        line.match_indices(*prefix).any(|(index, _)| {
            let after = &line[index + prefix.len()..];
            after.starts_with('-')
                && after.chars().nth(1).is_some_and(|c| c.is_ascii_digit())
                && (index == 0
                    || !line[..index]
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_alphanumeric()))
        })
    })
}

/// `true` when `line` is a comment divider.
///
/// Structure in this codebase comes from module, `impl`, and function
/// boundaries and from blank lines — never from a hand-drawn rule — so
/// a comment whose opening run of decoration characters is long enough
/// to read as one is a divider and is flagged. The full comment marker
/// comes off first: docs draw rules too, and leaving the `///` or `//!`
/// marker's third character in front of the decoration would let
/// exactly those spellings — where a real divider hides most naturally —
/// walk past the check reading as ordinary text.
fn is_separator(line: &str) -> bool {
    let Some(markerless) = line.trim().strip_prefix("//") else {
        return false;
    };
    let rest = markerless
        .strip_prefix(['/', '!'])
        .unwrap_or(markerless)
        .trim_start();
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

/// Phrases that betray development-process provenance in a changelog.
///
/// A changelog entry states what changed between versions; which review
/// surfaced a change or which planning artifact decided it is metadata
/// that rides the commit record, never the release log. Joined from
/// pieces at runtime, like the id prefixes, so this file stays free of
/// the phrases it hunts.
fn banned_changelog_phrases() -> &'static [String] {
    static PHRASES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PHRASES.get_or_init(|| {
        [
            ("Road", "map"),
            ("planning", " docs"),
            ("review", "-round"),
            ("pr", " round"),
            ("pr", " review"),
            ("hostile", " review"),
            ("hostile", "-review"),
            ("review", " pins"),
            ("review", "'s"),
            ("follow-up", " review"),
            ("external", " review"),
            ("DCH-", "DESIGN"),
            ("LOOPCTL-", "DESIGN"),
            ("review", " round"),
            ("sk", "etch"),
            ("task", "-named"),
            ("task", " locked"),
            ("locked", " names"),
            ("task", " later"),
        ]
        .iter()
        .map(|(head, tail)| format!("{head}{tail}").to_lowercase())
        .collect()
    })
}

/// `true` when `line` narrates planning provenance, not a change.
///
/// The changelog's characteristic relapses beyond the id shape
/// [`has_task_id`] already hunts: roadmap references, planning-doc
/// inventories, review-round attribution, task-or-sketch process prose,
/// and internal consumer references, matched case-insensitively. The
/// numbered markers (gap analysis, development round, external issue)
/// count only in their exact shapes, and the consumer-product token
/// only as a standalone word — so real shipped spellings, underscored
/// test names, and the repository URL itself stay clean.
fn names_planning_prose(line: &str) -> bool {
    let lower = line.to_lowercase();
    let bytes = lower.as_bytes();
    let numbered = |prefix: &str, offset: usize| {
        lower.match_indices(prefix).any(|(index, _)| {
            bytes[index + offset..]
                .first()
                .is_some_and(u8::is_ascii_digit)
        })
    };
    let consumer_ref = lower.match_indices("dch").any(|(index, _)| {
        let before_free = index
            .checked_sub(1)
            .is_none_or(|i| !(bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_'));
        let after_free = bytes
            .get(index + 3)
            .is_none_or(|byte| !(byte.is_ascii_alphanumeric() || *byte == b'_'));
        before_free && after_free && !lower[index..].starts_with("dch-labs")
    });
    let issue_id = lower.match_indices("tx-").any(|(index, _)| {
        let word_start = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
        word_start
            && bytes
                .get(index + 3)
                .is_some_and(|byte| byte.is_ascii_digit())
    });
    numbered("gap-", 4)
        || numbered("round-", 6)
        || consumer_ref
        || issue_id
        || banned_changelog_phrases()
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

#[test]
fn changelog_ships_changes_not_planning_metadata() {
    // Sensitivity first: lines assembled from the same pieces must trip
    // the probes, or the silent scan below proves nothing.
    let id = format!("{}-9", ID_PREFIXES[0]);
    let roadmap_tail = format!("the wire moved. ({}{} {id} rider, v0.5.0.)", "Road", "map:");
    let provenance = format!("{} {} named it; decided at the sketch.", "Review", "round");
    let gap_marker = format!("the {}-1 disease, on the wire", "Gap");
    assert!(
        has_task_id(&roadmap_tail) && names_planning_prose(&roadmap_tail),
        "a tagged tail trips both probes by construction: {roadmap_tail}"
    );
    let external_id = format!("ships after {}{}-9 lands", "T", "X");
    assert!(
        has_task_id(&external_id),
        "a multi-letter tracker id trips the id probe: {external_id}"
    );
    assert!(
        !has_task_id("dates follow ISO-8601 spellings on the wire"),
        "an ordinary standard reference is not a planning id"
    );
    assert!(
        names_planning_prose(&provenance),
        "review-and-sketch provenance trips the prose probe: {provenance}"
    );
    assert!(
        names_planning_prose(&gap_marker),
        "the numbered gap marker trips the prose probe: {gap_marker}"
    );
    let doc_inventory = format!(
        "Reconciled the {} ({}{}).",
        concat!("planning", " docs"),
        "DCH-",
        "DESIGN"
    );
    assert!(
        names_planning_prose(&doc_inventory),
        "a planning-doc inventory trips the prose probe: {doc_inventory}"
    );
    let consumer_ref = format!("no {}-side change, per {}-42.", "dch", "TX");
    assert!(
        names_planning_prose(&consumer_ref),
        "consumer-product references and external issue ids trip the probe: {consumer_ref}"
    );
    let round_pin = format!("{}-5 pins: `a_contract_holds`", "Round");
    assert!(
        names_planning_prose(&round_pin),
        "development-round numbering trips the probe: {round_pin}"
    );
    let lowercase_issue = format!("issue {}-7 tracked the fix.", "tx");
    assert!(
        names_planning_prose(&lowercase_issue),
        "a lowercase issue id trips the probe as surely as the spelled one: {lowercase_issue}"
    );
    assert!(
        !names_planning_prose("the `postx-9` fixture id names a fixture, not an issue"),
        "a lowercase tx- inside a word keeps the boundary exemption"
    );
    assert!(
        !names_planning_prose(
            "`echo_round_trip_returns_text`, `.dchignore`, `like_dch_v1`, github.com/dch-labs/loopctl"
        ),
        "shipped spellings, real identifiers, and the repository URL are not provenance"
    );
    assert!(
        !names_planning_prose(
            "- **Edit results stop echoing files** — no result echoes a file now; \
             vocabulary gaps close in the descriptions."
        ),
        "a change stated as a change must not read as provenance"
    );

    let mut leaks = Vec::new();
    scan(&root().join("CHANGELOG.md"), &mut leaks, &|line| {
        has_task_id(line) || names_planning_prose(line)
    });
    assert!(
        leaks.is_empty(),
        "planning metadata shipped into the changelog — it records the changes \
         between versions, never the task, review, or planning artifact behind one:\n{}",
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

#[test]
fn doc_comment_dividers_are_detected() {
    // A divider drawn inside a doc comment is still a divider; a probe
    // that stripped only `//` left the marker's third character ahead
    // of the decoration, and the run check read those spellings as
    // ordinary text.
    assert!(
        is_separator("// ---- plain rule"),
        "the plain spelling was never the hole, but it must stay caught"
    );
    assert!(
        is_separator("/// ==== section ===="),
        "a divider drawn in a doc comment is still a divider"
    );
    assert!(
        is_separator("//! ###### module rule ######"),
        "a divider drawn in a module doc is still a divider"
    );
    assert!(
        is_separator("    /// --- indented rule"),
        "indentation is no hiding place either"
    );
    assert!(
        is_separator("/// ——— unicode rule"),
        "the em-dash arm reads doc comments too"
    );
    // Doc prose is not a rule: bullets, slashes, and short runs stay clean.
    assert!(
        !is_separator("/// - a single dash is a list marker"),
        "one dash is prose, not decoration"
    );
    assert!(
        !is_separator("// -- short run below the divider bound"),
        "a decoration run below the minimum is not a divider"
    );
    assert!(
        !is_separator("/// reads as ordinary doc prose"),
        "doc text without a decoration run stays unflagged"
    );
}
