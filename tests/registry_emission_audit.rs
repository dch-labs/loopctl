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
//!   comment *line* is prose that can never emit anything, so it cannot
//!   satisfy the gate. The strip works line by line, so it is a line that
//!   opens with a comment marker that is excluded; an emission call with a
//!   trailing note is still the code it looks like.
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
use std::ops::Range;
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
    let mut out = String::with_capacity(source.len());
    let mut in_block = false;
    for line in source.lines() {
        let mut kept = String::new();
        let mut in_str = false;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if in_block {
                if c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    in_block = false;
                }
                continue;
            }
            if c == '"' {
                in_str = !in_str;
                kept.push(c);
                continue;
            }
            if c == '/' && !in_str {
                // A `//` ends the code on this line; a `/*` ends it until the
                // closing `*/`, which may sit lines away. Either way, a
                // `metric = "…"` inside a comment is prose about a metric, not
                // a site that emits it.
                if chars.peek() == Some(&'/') {
                    break;
                }
                if chars.peek() == Some(&'*') {
                    chars.next();
                    in_block = true;
                    continue;
                }
            }
            kept.push(c);
        }
        out.push_str(&kept);
        out.push('\n');
    }
    out
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
        // `target` is generated; `tests` and `examples` are prose about
        // metrics rather than emitters of them. A companion crate's test that
        // spells `metric = "…"` must not stand in for library code that never
        // emits the row.
        if name == "target" || name == "tests" || name == "examples" {
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

/// The code-only text of every `.rs` file under `src/` and `crates/`.
///
/// Kept one entry per file rather than one joined blob: an emission site is
/// decided by the call it sits in, and a scan that runs across a file
/// boundary could let an opener in one file enclose a name in another.
fn emission_sources() -> Vec<String> {
    let files = {
        let mut files = Vec::new();
        collect_rs(Path::new("src"), &mut files);
        collect_rs(Path::new("crates"), &mut files);
        files
    };
    files.iter().map(|file| code_only(&read(file))).collect()
}

/// Whether `name` has an emission site in `sources`.
///
/// A site is `metric = "name"` appearing as a *field* of a tracing event aimed
/// at the metrics target, which is the only shape an emission takes
/// (`tracing::debug!` with `target: "loopctl::metrics"`). Two shapes are refused:
/// text inside a string literal — prose *about* a row, and prose has never
/// written a sample — and a field of anything that is not a macro call.
///
/// Spacing around `=` is not load-bearing. `cargo fmt` cannot normalise it
/// inside a macro arm it cannot parse as expressions (`target:` is one), so an
/// emitter written `metric="name"` survives formatting exactly as its author
/// typed it, and a gate demanding the spaced spelling would report a working
/// emitter as having no site.
fn has_emission_site(sources: &[String], name: &str) -> bool {
    sources.iter().any(|src| emission_site_in(src, name))
}

/// One source, one row name.
fn emission_site_in(src: &str, name: &str) -> bool {
    let mut from = 0usize;
    while let Some(tail) = src.get(from..) {
        let Some(offset) = find_metric_field(tail, name) else {
            return false;
        };
        let at = from.saturating_add(offset);
        if is_macro_argument(src, at)
            && is_tracing_event(src, at)
            && aimed_at_metrics(
                enclosing_macro(src, at)
                    .and_then(|(_, args)| src.get(args).map(str::to_owned))
                    .as_deref()
                    .unwrap_or_default(),
            )
        {
            return true;
        }
        from = at.saturating_add(1);
    }
    false
}

/// Locate `metric` `=` `"name"`, at any spacing around the `=`.
fn find_metric_field(hay: &str, name: &str) -> Option<usize> {
    let bytes = hay.as_bytes();
    let mut quoted = String::from("\"");
    quoted.push_str(name);
    quoted.push('"');
    let skip_spaces = |mut at: usize| {
        while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            at = at.saturating_add(1);
        }
        at
    };
    let mut from = 0usize;
    while let Some(relative) = hay.get(from..).and_then(|tail| tail.find("metric")) {
        let start = from.saturating_add(relative);
        let after_word = start.saturating_add("metric".len());
        let after_spaces = skip_spaces(after_word);
        if bytes.get(after_spaces) == Some(&b'=') {
            let at = skip_spaces(after_spaces.saturating_add(1));
            if hay
                .get(at..)
                .is_some_and(|rest| rest.starts_with(quoted.as_str()))
            {
                return Some(start);
            }
        }
        from = after_word;
    }
    None
}

/// Whether the `metric` token at `at` opens an argument of a macro call.
///
/// Two structural conditions. What introduces the token, past whitespace, is
/// `(` or `,` — so the match begins an argument rather than sitting inside a
/// string or in the middle of an expression. And the bracket it belongs to is
/// opened by a `!`, so it is a macro and not a struct literal or a plain
/// function call, neither of which reaches a subscriber.
fn is_macro_argument(src: &str, at: usize) -> bool {
    let bytes = src.as_bytes();
    let mut before = at;
    while matches!(
        bytes.get(before.saturating_sub(1)),
        Some(b' ' | b'\t' | b'\n' | b'\r')
    ) {
        before = before.saturating_sub(1);
    }
    if !matches!(bytes.get(before.saturating_sub(1)), Some(b'(' | b',')) {
        return false;
    }
    enclosing_macro(src, at).is_some()
}

/// The macro call enclosing `at`: the name before its `!`, and its arguments.
///
/// Bracket matching treats every bracket as one nesting class, the way the
/// backwards walk always has — the question is which call a field belongs to,
/// not how the file parses.
fn enclosing_macro(src: &str, at: usize) -> Option<(&str, Range<usize>)> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (seen, byte) in bytes.iter().take(at).rev().enumerate() {
        match *byte {
            b')' | b']' | b'}' => depth = depth.saturating_add(1),
            b'(' | b'[' | b'{' => {
                if depth == 0 {
                    let open = at.saturating_sub(1).saturating_sub(seen);
                    let mut before = open;
                    while matches!(
                        bytes.get(before.saturating_sub(1)),
                        Some(b' ' | b'\t' | b'\n' | b'\r')
                    ) {
                        before = before.saturating_sub(1);
                    }
                    if bytes.get(before.saturating_sub(1)) != Some(&b'!') {
                        return None;
                    }
                    let name_end = before.saturating_sub(1);
                    let mut name_start = name_end;
                    while name_start > 0
                        && matches!(
                            bytes.get(name_start.saturating_sub(1)),
                            Some(b) if b.is_ascii_alphanumeric() || matches!(b, b'_' | b':' | b'.')
                        )
                    {
                        name_start = name_start.saturating_sub(1);
                    }
                    let close = matching_close(src, open)?;
                    return Some((
                        src.get(name_start..name_end).unwrap_or_default(),
                        open.saturating_add(1)..close,
                    ));
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }
    None
}

/// The index of the bracket closing the one at `open`.
fn matching_close(src: &str, open: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(open) {
        match byte {
            b'(' | b'[' | b'{' => depth = depth.saturating_add(1),
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether the field at `at` belongs to a tracing event macro.
///
/// A `metric` field handed to any other macro reaches no subscriber at all,
/// whatever its fields say.
fn is_tracing_event(src: &str, at: usize) -> bool {
    let Some((name, _)) = enclosing_macro(src, at) else {
        return false;
    };
    matches!(
        name.rsplit(':').next().unwrap_or_default(),
        "trace" | "debug" | "info" | "warn" | "error" | "event"
    )
}

/// Whether an argument list names the target the metrics layer keeps.
///
/// Both spellings count: the literal every shipped emitter writes, and
/// `MetricsLayer::TARGET`, which the crate's own test emitters use. The layer
/// drops an event whose target differs from its own, so a field aimed anywhere
/// else never reaches a row however faithfully it names one.
fn aimed_at_metrics(args: &str) -> bool {
    let Some(relative) = args.find("target:") else {
        return false;
    };
    let value = args
        .get(relative.saturating_add("target:".len())..)
        .unwrap_or_default()
        .trim_start();
    value.starts_with("\"loopctl::metrics\"") || value.starts_with("MetricsLayer::TARGET")
}

#[test]
fn a_quoted_metric_name_is_not_an_emission_site() {
    // The gate asks whether a row is *spoken*, not whether its name is spelled.
    // A real call site counts; the same text inside a string literal, or handed
    // to a function that never reaches a subscriber, is prose about a row.
    const Q: &str = "loopctl.dead.row";
    let call =
        r#"tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row", "spoken");"#;
    assert!(
        has_emission_site(&[call.to_string()], Q),
        "a `tracing::debug!` site must count, or the gate fails the crate's own emitters"
    );
    let quoted = r#"const GUIDE: &str = "spell it as metric = "loopctl.dead.row" in the docs";"#;
    assert!(
        !has_emission_site(&[quoted.to_string()], Q),
        "a string literal quoting the shape is prose, not an emission site"
    );
    let plain_call = r#"record("metric = "loopctl.dead.row".to_string());"#;
    assert!(
        !has_emission_site(&[plain_call.to_string()], Q),
        "a function call reaches no subscriber, so it books nothing"
    );
    let across = [
        r#"tracing::debug!("unclosed".to_string(), "#.to_string(),
        r#"metric = "loopctl.dead.row";"#.to_string(),
    ];
    assert!(
        !has_emission_site(&across, Q),
        "an unterminated opener in an earlier file cannot make a later name an emission"
    );
}

#[test]
fn a_compactly_spaced_field_counts_and_a_strings_quoted_one_does_not() {
    // `cargo fmt` cannot rewrite `metric="name"` inside a macro arm it cannot
    // parse as expressions — `target:` is one — so the compact spelling is a
    // real emitter and this gate may not report it as having no site.
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", metric="loopctl.dead.row", "spoken");"#,
            "loopctl.dead.row"
        ),
        "a compactly spaced field is an emission: fmt leaves it exactly as written"
    );
    // The same text inside a macro's *string* is prose the macro prints. Built
    // rather than written as one literal: the snippet contains a raw-string
    // terminator, and nesting them costs hash counts.
    let quoted_in_macro = format!(
        "println!(r#\"{inner}\"#);",
        inner = r#"metric = "loopctl.dead.row""#
    );
    assert!(
        !emission_site_in(&quoted_in_macro, "loopctl.dead.row"),
        "a raw string inside a macro call quotes the shape and writes no sample"
    );
    assert!(
        !emission_site_in(
            r#"println!("metric = \"loopctl.dead.row\"");"#,
            "loopctl.dead.row"
        ),
        "an escaped copy of the shape inside a string writes no sample"
    );
    // And a field named `metric` on something that is not a macro reaches no
    // subscriber either.
    assert!(
        !emission_site_in(r#"Row { metric = "loopctl.dead.row" }"#, "loopctl.dead.row"),
        "a struct field named `metric` books nothing"
    );
}

/// The gate must read the macro and its target, not just its field.
///
/// A row is fed only by an event the metrics layer accepts: the layer drops
/// every event whose target is not its own, so a `metric` field spoken from
/// another target, or from a macro that is not a tracing event at all, is a
/// declared row with no emitter wearing the costume of one. Both spellings of
/// the target count, because the crate's own test emitters write the constant.
#[test]
fn a_site_is_a_tracing_event_aimed_at_the_metrics_target() {
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row", "spoken");"#,
            "loopctl.dead.row"
        ),
        "the spelling every shipped emitter uses must count"
    );
    assert!(
        emission_site_in(
            "tracing::debug!(target: MetricsLayer::TARGET, metric = \"loopctl.dead.row\");",
            "loopctl.dead.row"
        ),
        "the constant is the same target, and the crate's test emitters use it"
    );
    assert!(
        !emission_site_in(
            r#"tracing::debug!(target: "loopctl::engine", metric = "loopctl.dead.row", "spoken");"#,
            "loopctl.dead.row"
        ),
        "an event on another target is dropped by the layer, so it books nothing"
    );
    assert!(
        !emission_site_in(
            r#"tracing::debug!(metric = "loopctl.dead.row", "spoken");"#,
            "loopctl.dead.row"
        ),
        "an event with no target lands on the module path and is dropped too"
    );
    assert!(
        !emission_site_in(
            r#"println!("counted {}", metric = "loopctl.dead.row");"#,
            "loopctl.dead.row"
        ),
        "a macro that is not a tracing event reaches no subscriber"
    );
}

#[test]
fn a_commented_metric_is_not_an_emission_site() {
    // The gate reads code. A row named only in prose — a trailing comment, or
    // a block comment spanning lines — has no emitter and must not pass.
    let trailing = r#"let x = 1; // metric = "loopctl.invented.row""#;
    assert!(
        !code_only(trailing).contains("loopctl.invented.row"),
        "a metric named only after code on a line is a comment"
    );
    let block = "/*\nmetric = \"loopctl.invented.row\"\n*/\nfn real() {}\n";
    assert!(
        !code_only(block).contains("loopctl.invented.row"),
        "a metric named inside a block comment is prose"
    );
    assert!(
        code_only(block).contains("fn real() {}"),
        "the code around a comment is still code: {:?}",
        code_only(block)
    );
    let real = "metric = \"loopctl.invented.row\",";
    assert!(
        code_only(real).contains("loopctl.invented.row"),
        "stripping comments must never strip a string literal"
    );
}

#[test]
fn the_scan_skips_test_and_example_sources() {
    let mut files = Vec::new();
    collect_rs(std::path::Path::new("crates"), &mut files);
    assert!(
        !files.is_empty(),
        "the companion crates are scanned at all, or this pin verifies nothing"
    );
    let strays: Vec<String> = files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .filter(|s| s.contains("/tests/") || s.contains("/examples/"))
        .collect();
    assert!(
        strays.is_empty(),
        "a test or example spelling `metric = \"…\"` is prose about a row, not an \
         emitter of it: {strays:?}"
    );
}

#[test]
fn every_declared_metric_row_has_an_emission_site() {
    let sources = emission_sources();
    let declared = declared_names(&read(Path::new("src/metrics/registry.rs")));
    assert!(
        declared.len() > 30,
        "the registry table must parse into its rows, got {}",
        declared.len()
    );

    let unemitted: Vec<&String> = declared
        .iter()
        .filter(|name| !has_emission_site(&sources, name))
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
