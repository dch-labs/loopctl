//! Every declared registry row must have an emission site.
//!
//! The catalog in `src/metrics/registry.rs` is the published promise of
//! which signals loopctl emits — the list a host writes a dashboard
//! against. The companion contract test
//! (`tests/emission_catalog_contract.rs`) proves the other direction:
//! everything emitted is declared. This file closes the loop, so a row
//! added ahead of its emitter (or an emitter deleted and its row left
//! behind) fails the build instead of quietly shipping a metric that
//! never carries a sample.
//!
//! Two properties make that promise worth having, and both are pinned
//! here rather than assumed:
//!
//! - **The site is code.** Emission sites are located by the literal
//!   `metric = "<name>"` field the crate's emission idiom requires, and
//!   only in code — a name inside a doc example, a module narrative, or a
//!   comment is prose that can never emit anything, so it cannot satisfy
//!   the gate.
//! - **Every row is read.** A row is found by its opener, which always
//!   sits at a line's start, and its name is the first quoted string from
//!   there on — which is what makes a row rustfmt wrapped across lines
//!   (`c(` on one line, the name on the next, the shape any long metric
//!   name takes once it passes the width bound) as visible as a
//!   single-line row. An opener with no name anywhere fails loud, and
//!   [`the_gate_sees_every_row_the_crate_publishes`] holds the parsed
//!   table against the catalog the crate actually compiles, so a parser
//!   blind spot cannot shrink the audit in silence.
//!
//! Run: `cargo test --all-features --test registry_emission_audit`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    // The walk fails loud on an unreadable path: a scan that skipped a file
    // it could not open would report its rows as unemitted — the same
    // traversal policy the doctest census states and allows for.
    clippy::panic
)]

use std::fs;
use std::path::Path;
use std::path::PathBuf;

/// Reads a file, naming the path on failure.
///
/// A scan that silently skips an unreadable file would report its rows as
/// unemitted (or its emissions as absent) — the failure mode that hides
/// exactly what this audit exists to catch.
fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e} (an unreadable file is not a clean scan)",
            path.display()
        )
    })
}

/// The first quoted string in `line`, if it carries one.
///
/// A row's name is always its first quoted string, and a row's label list
/// follows it, so the first pair on a line is the name wherever that line
/// sits in the row.
fn first_quoted(line: &str) -> Option<&str> {
    let after_name_quote = line.split_once('"')?.1;
    after_name_quote.split_once('"').map(|(name, _)| name)
}

/// Whether `trimmed` opens a row of the `REGISTRY` table.
fn opens_row(trimmed: &str) -> bool {
    trimmed.starts_with("c(") || trimmed.starts_with("g(") || trimmed.starts_with("h(")
}

/// Collects the declared metric names from the `REGISTRY` table only.
///
/// Parsing starts at the table's opening line and stops at its closing
/// bracket, so metric names that appear later in the same file — inline-test
/// fixtures, doctest examples — are not mistaken for declarations. A row is
/// recognized by its `c(`/`g(`/`h(` opener at the line's start; its name is
/// the first quoted string at or after that opener, whether the row is
/// written on one line or wrapped by rustfmt. An opener that yields no name
/// panics: a row the parser cannot read is a row the gate would silently
/// stop auditing, which is the failure this file exists to prevent.
fn declared_names(registry_src: &str) -> Vec<String> {
    let lines: Vec<&str> = registry_src.lines().collect();
    let mut names = Vec::new();
    let mut in_table = false;
    let mut index = 0;
    while let Some(raw) = lines.get(index) {
        let trimmed = raw.trim();
        if trimmed.starts_with("pub const REGISTRY") {
            in_table = true;
            index += 1;
            continue;
        }
        if !in_table {
            index += 1;
            continue;
        }
        if trimmed.starts_with("];") {
            break;
        }
        if !opens_row(trimmed) {
            index += 1;
            continue;
        }
        let mut found = None;
        let mut cursor = index;
        for (offset, continuation) in lines.iter().enumerate().skip(index) {
            cursor = offset;
            if offset > index {
                let next = continuation.trim();
                if opens_row(next) || next.starts_with("];") {
                    break;
                }
            }
            if let Some(name) = first_quoted(continuation) {
                found = Some(name.to_string());
                break;
            }
        }
        names.push(found.unwrap_or_else(|| {
            panic!(
                "REGISTRY row opening at line {} (`{trimmed}`) carries no quoted name \
                 before the next row or the table's end — the gate cannot audit a row it \
                 cannot read",
                index + 1
            )
        }));
        index = cursor.max(index) + 1;
    }
    names
}

/// `source` with every comment line removed.
///
/// An emission is a call, so only code can hold an emission site: a name
/// inside a `///` example, a `//!` module narrative, or a `//` note is prose
/// describing a signal that may well never fire. Leaving prose in the
/// haystack lets a doc example satisfy the gate for a dead row — the exact
/// false positive this audit was written to prevent, and the reason the
/// search runs over code alone.
fn code_only(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Walks `dir` collecting the text of every `.rs` file.
///
/// Symlinked directories are never followed and `target` is pruned, so the
/// scan reads the crate's own source rather than generated output.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot open {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("cannot read entry in {}: {e}", dir.display()));
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" {
            continue;
        }
        let file_type = entry
            .file_type()
            .unwrap_or_else(|e| panic!("cannot stat {}: {e}", path.display()));
        if file_type.is_dir() {
            collect_rs(&path, out);
        } else if name.to_string_lossy().ends_with(".rs") {
            out.push(path);
        }
    }
}

/// Every `.rs` file under `src/` and `crates/`, code-only.
fn emission_haystack() -> String {
    let files = {
        let mut files = Vec::new();
        collect_rs(Path::new("src"), &mut files);
        collect_rs(Path::new("crates"), &mut files);
        files
    };
    let mut haystack = String::new();
    for file in &files {
        haystack.push_str(&code_only(&read(file)));
        haystack.push('\n');
    }
    haystack
}

#[test]
fn every_declared_metric_row_has_an_emission_site() {
    let haystack = emission_haystack();
    let declared = declared_names(&read(Path::new("src/metrics/registry.rs")));
    assert!(
        declared.len() > 30,
        "the registry table must parse into its rows, got {}",
        declared.len()
    );

    let unemitted: Vec<&String> = declared
        .iter()
        .filter(|name| !haystack.contains(&format!("metric = \"{name}\"")))
        .collect();

    assert!(
        unemitted.is_empty(),
        "{} declared metric row(s) have no emission site (`metric = \"<name>\"` appears \
         nowhere in the code under src/ or crates/), so the catalog advertises a signal \
         nothing can ever emit. Either wire the emission or remove the row:\n{}",
        unemitted.len(),
        unemitted
            .iter()
            .map(|n| format!("  - {n}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_parser_reads_rows_written_in_either_shape() {
    let table = "pub const REGISTRY: &[MetricDescriptor] = &[\n    \
                 c(\"loopctl.one.line\", &[\"label\"]),\n    \
                 g(\"loopctl.also.one.line\", &[]),\n    \
                 c(\n        \"loopctl.wrapped.row\",\n        &[\"a\", \"b\"],\n    ),\n    \
                 h(\n        \"loopctl.wrapped.h\",\n        &[],\n    ),\n];\n";
    assert_eq!(
        declared_names(table),
        vec![
            "loopctl.one.line",
            "loopctl.also.one.line",
            "loopctl.wrapped.row",
            "loopctl.wrapped.h",
        ],
        "a row rustfmt wrapped across lines must yield the same name as a single-line one"
    );

    let real = declared_names(&read(Path::new("src/metrics/registry.rs")));
    for wrapped in [
        "loopctl.compaction.summarizer.tokens",
        "loopctl.compaction.structured.sections",
    ] {
        assert!(
            real.iter().any(|name| name == wrapped),
            "{wrapped} is declared with its name on the line after the opener, and the \
             gate read no such row — a wrapped row would drop out of the audit: {real:?}"
        );
    }
}

#[test]
#[should_panic(expected = "carries no quoted name")]
fn an_opener_without_a_name_fails_loud() {
    declared_names("pub const REGISTRY: &[MetricDescriptor] = &[\n    c(\n];\n");
}

#[test]
fn a_name_that_lives_only_in_prose_is_not_an_emission_site() {
    let module_narrative = "//! metric = \"loopctl.dead.row\"\n";
    let doc_example = "/// ```\n/// metric = \"loopctl.dead.row\"\n/// ```\n";
    let note = "// metric = \"loopctl.dead.row\"\n";
    for prose in [module_narrative, doc_example, note] {
        assert!(
            !code_only(prose).contains("metric = \"loopctl.dead.row\""),
            "prose must not satisfy the gate: {prose:?}"
        );
    }

    let code =
        "fn emit() {\n    tracing::debug!(\n        metric = \"loopctl.dead.row\",\n    );\n}\n";
    assert!(
        code_only(code).contains("metric = \"loopctl.dead.row\""),
        "a real call site must survive the strip: {code:?}"
    );
}

#[cfg(feature = "metrics")]
#[test]
fn the_gate_sees_every_row_the_crate_publishes() {
    let declared = declared_names(&read(Path::new("src/metrics/registry.rs")));
    let published = loopctl::metrics::registry::all();
    assert_eq!(
        declared.len(),
        published.len(),
        "the gate read {} rows but the crate publishes {} — the audit is silently \
         narrower than the catalog",
        declared.len(),
        published.len()
    );
    for descriptor in published {
        assert!(
            declared.iter().any(|name| name == descriptor.name),
            "{} is published by `registry::all` but the gate never parsed it, so nothing \
             can ever fail its way to green",
            descriptor.name
        );
        assert!(
            loopctl::metrics::registry::lookup(descriptor.name).is_some(),
            "the parsed name {} does not resolve in the catalog it came from",
            descriptor.name
        );
    }
}
