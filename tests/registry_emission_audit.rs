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
    files
        .iter()
        .map(|file| strip_cfg_test(&code_only(&read(file))))
        .collect()
}

/// Blank the items annotated `#[cfg(test)]`, leaving every other byte alone.
///
/// An emitter that only compiles for the test build ships nothing, so it cannot
/// be the site that keeps a declared row — the same reason the scan skips the
/// `tests` directory, applied inside a file that is otherwise production code.
/// Blanking rather than deleting keeps every remaining byte at its offset, so
/// the bracket walks still see whole calls. `#[cfg(not(test))]` is the shipped
/// half and stays standing; a compound `#[cfg(all(test, feature = "x"))]` item
/// is excluded along with the plain one.
fn strip_cfg_test(source: &str) -> String {
    let bytes = source.as_bytes();
    let code = code_mask(source);
    let mut keep = vec![true; bytes.len()];
    let mut from = 0usize;
    while let Some(relative) = source.get(from..).and_then(|tail| tail.find("#[cfg(")) {
        let attr = from.saturating_add(relative);
        if !code.get(attr).copied().unwrap_or(true) {
            from = attr.saturating_add(1);
            continue;
        }
        let Some(closer) = source
            .get(attr..)
            .and_then(|tail| tail.find(")]"))
            .map(|offset| attr.saturating_add(offset))
        else {
            break;
        };
        from = closer.saturating_add(")]".len());
        let inner = source
            .get(attr.saturating_add("#[cfg(".len())..closer)
            .unwrap_or_default();
        if !inner.contains("test") || inner.contains("not(test") {
            continue;
        }
        let Some(end) = item_end(source, &code, from) else {
            break;
        };
        for slot in keep.iter_mut().skip(attr).take(end.saturating_sub(attr)) {
            *slot = false;
        }
    }
    let blanked = bytes
        .iter()
        .enumerate()
        .map(|(index, byte)| {
            if *byte == b'\n' || keep.get(index).copied().unwrap_or(true) {
                *byte
            } else {
                b' '
            }
        })
        .collect::<Vec<u8>>();
    // Excluded ranges always cover whole literals, so replacing their bytes
    // with spaces leaves valid text behind.
    String::from_utf8(blanked).unwrap_or_default()
}

/// Where the item following an attribute ends: the `;` of a declaration, or the
/// bracket closing its block.
///
/// Delimiters inside a string are skipped, so a message containing `{` cannot
/// start a block that has to be matched.
fn item_end(source: &str, code: &[bool], from: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut index = from;
    while let Some(byte) = bytes.get(index).copied() {
        if code.get(index).copied().unwrap_or(true) {
            match byte {
                b';' => return Some(index.saturating_add(1)),
                b'{' => return matching_close(source, index).map(|close| close.saturating_add(1)),
                _ => {}
            }
        }
        index = index.saturating_add(1);
    }
    None
}

/// Whether `name` has an emission site in `sources`.
///
/// A site is `metric = "name"` appearing as a *field* of a tracing event aimed
/// at the metrics target — `tracing::debug!` with `target: "loopctl::metrics"`,
/// the shape every shipped emitter uses. Two shapes are refused: text inside a
/// string literal — prose *about* a row, and prose has never written a sample —
/// and a field of anything that is not a macro call.
///
/// The target is recognised by its spelling rather than by what it evaluates
/// to, which is a real limit of this gate; see [`aimed_at_metrics`] for what it
/// costs and why it is drawn there.
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
///
/// A match inside a literal is prose, so the literal mask is consulted before
/// anything else: a message that quotes an emitter, even in the shape
/// `debug!(metric = "…")`, describes a row without writing a sample.
fn emission_site_in(src: &str, name: &str) -> bool {
    let code = code_mask(src);
    let mut from = 0usize;
    while let Some(tail) = src.get(from..) {
        let Some(offset) = find_metric_field(tail, name) else {
            return false;
        };
        let at = from.saturating_add(offset);
        if !code.get(at).copied().unwrap_or(true) {
            from = at.saturating_add(1);
            continue;
        }
        let aimed = enclosing_macro(src, at).and_then(|(_, span)| {
            if at >= span.end {
                return None;
            }
            let text = src.get(span.clone())?;
            Some((text.to_owned(), at.saturating_sub(span.start)))
        });
        let aimed_at_the_target = match &aimed {
            Some((text, metric_at)) => aimed_at_metrics(text, *metric_at),
            None => false,
        };
        if is_macro_argument(src, at)
            && !inside_token_consumer(src, at, &code)
            && is_tracing_event(src, at)
            && aimed_at_the_target
            && !dormant_macro_body(src, &code, at)
        {
            return true;
        }
        from = at.saturating_add(1);
    }
    false
}

/// A macro that reads the tokens inside it instead of running them.
///
/// What goes in comes out as text or as a value, never as executed code.
const TOKEN_CONSUMERS: [&str; 8] = [
    "stringify",
    "concat",
    "include",
    "include_str",
    "include_bytes",
    "env",
    "option_env",
    "cfg",
];

/// Whether `at` sits inside the argument list of such a macro.
///
/// A call handed to `stringify!` is text: no subscriber ever sees that event, so
/// a field written inside it books no sample however faithfully it is spelled.
/// A `macro_rules!` body is deliberately absent from the list -- its tokens run
/// at every call, so a row fed only from a macro definition is a site this scan
/// cannot follow, and the row fails loud rather than being written off as dead.
/// The check costs a search per consumer per candidate, so it runs last of the
/// conditions guarding a site.
fn inside_token_consumer(src: &str, at: usize, code: &[bool]) -> bool {
    TOKEN_CONSUMERS.iter().any(|name| {
        let needle = format!("{name}!");
        let mut from = 0usize;
        loop {
            let Some(relative) = src.get(from..).and_then(|tail| tail.find(needle.as_str())) else {
                return false;
            };
            let head = from.saturating_add(relative);
            from = head.saturating_add(needle.len());
            let mut open = from;
            while matches!(src.as_bytes().get(open), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                open = open.saturating_add(1);
            }
            // `my_stringify!` and `crate::stringify!` are not this macro, and a
            let is_the_word = !matches!(
                src.as_bytes().get(head.saturating_sub(1)),
                Some(b) if b.is_ascii_alphanumeric() || *b == b'_' || *b == b':'
            );
            if !is_the_word
                || !matches!(src.as_bytes().get(open), Some(b'(' | b'[' | b'{'))
                || !code.get(head).copied().unwrap_or(true)
            {
                continue;
            }
            if !matching_close(src, open).is_some_and(|close| at > open && at < close) {
                continue;
            }
            if src.contains("tracing::")
                && imported_from_tracing(src, name, head)
                    .is_some_and(|(origin, _, _)| is_tracing_event_macro(origin))
            {
                continue;
            }
            return true;
        }
    })
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
/// not how the file parses. Delimiters written inside a string or a comment are
/// skipped: an unmatched `[` in a log message is prose, and counting it would
/// report the call as unlocatable.
fn enclosing_macro(src: &str, at: usize) -> Option<(&str, Range<usize>)> {
    let bytes = src.as_bytes();
    let code = code_mask(src);
    let mut depth = 0usize;
    for (seen, byte) in bytes.iter().take(at).rev().enumerate() {
        let here = at.saturating_sub(1).saturating_sub(seen);
        if !code.get(here).copied().unwrap_or(true) {
            continue;
        }
        match *byte {
            b')' | b']' | b'}' => depth = depth.saturating_add(1),
            b'(' | b'[' | b'{' => {
                if depth == 0 {
                    let open = here;
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
///
/// Delimiters inside a string or a comment do not count, or a `)` written in a
/// message would close the call early and the arguments handed to the target
/// check would stop where the prose happened to end.
fn matching_close(src: &str, open: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let code = code_mask(src);
    let mut depth = 0usize;
    for (index, byte) in bytes.iter().enumerate().skip(open) {
        if !code.get(index).copied().unwrap_or(true) {
            continue;
        }
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
/// whatever its fields say, so the path is checked and not only the last
/// segment: `log::info!` takes a `metric` field just as readily and books
/// nothing. A qualified name must name `tracing`, in either the relative or the
/// absolute spelling.
///
/// A bare `debug!` is settled by the file, which is as far as this resolves a
/// path: the name is whichever macro the imports in scope of the call bind it
/// to, so an import line naming it decides what it is and from where, and the
/// block nearest the call that carries such a line wins. A rename is settled
/// there too --
/// `use tracing::debug as dbg` makes `dbg!` an event and leaves `debug!` out of
/// scope -- because a binding is no clue to the macro behind it. A macro
/// re-exported behind another path is therefore unseen, and the row it serves
/// fails here rather than passing on an event no subscriber reads. Two further
/// refusals guard the name itself: a local `macro_rules!` of the event's name
/// shadows the import for every call written below its definition, and an
/// `as tracing` rename of another crate spells every relative `tracing::…`
/// path — bare or qualified — that other crate's name. An absolute `::tracing`
/// anchors to the extern prelude, where no rename reaches: it, and any import
/// that arrived by one, rides out the rename untouched.
fn is_tracing_event(src: &str, at: usize) -> bool {
    let Some((name, _)) = enclosing_macro(src, at) else {
        return false;
    };
    let absolute = name.starts_with("::");
    let spelled = name.trim_start_matches(':');
    if let Some((path, last)) = spelled.rsplit_once("::") {
        // Qualified: the path must name tracing, the last segment an event,
        // and a relative spelling must survive the rename check.
        return path == "tracing"
            && is_tracing_event_macro(last)
            && (absolute || !renamed_veto(src, at));
    }
    // Bare: the import carries the six-name rule, judged where the import was
    // written rather than where the call sits, since under a rename the
    // name at the call site says nothing about which macro it came from —
    // unless it arrived by an absolute path, which no rename can reach —
    // and under a same-named `macro_rules!` written above the call the
    // import says nothing either, because the local definition wins.
    match imported_from_tracing(src, spelled, at) {
        Some((origin, from_absolute, use_start)) if is_tracing_event_macro(origin) => {
            (from_absolute || !renamed_veto(src, use_start))
                && (!src.contains("macro_rules!")
                    || !shadowed_by_macro_rules(src, &code_mask(src), spelled, at))
        }
        _ => false,
    }
}

/// Whether an `as tracing` rename reaches `at`: the scope machinery decides,
/// with no textual fast path — a rename can be spaced however its writer
/// pleased, so only the parse itself can rule it out.
fn renamed_veto(src: &str, at: usize) -> bool {
    tracing_renamed_in_scope(src, &code_mask(src), at)
}

/// The macros tracing offers for writing an event.
const TRACING_EVENT_MACROS: [&str; 6] = ["trace", "debug", "info", "warn", "error", "event"];

/// Whether `name` is one of them. `span!` is a tracing macro too and writes a
/// span, which the metrics layer never sees as an event.
fn is_tracing_event_macro(name: &str) -> bool {
    TRACING_EVENT_MACROS.contains(&name)
}

/// When this file binds `name` from `tracing`, the tracing-side name behind the
/// binding — decided at the call site `at` and nowhere else: a `use` binds the
/// block it is declared in and every block nested inside it, never a sibling,
/// and where two bind the same spelling at one spot the nearer block wins, so
/// an inner `use log::debug;` shadows an outer `use tracing::debug;`. The crate
/// root's glob (`use tracing::*;`) binds the event macros under their own
/// names; a glob over a submodule binds what that submodule exports, which a
/// textual check cannot see, so it binds nothing — and a glob never outranks
/// a named import written beside it in the same block.
///
/// Covers the plain, braced and `as`-renamed spellings: a rename binds the name
/// on its right and arrives from the one on its left, and a braced list binds
/// each entry the same way. The answer carries whether the winning `use` was
/// written as an absolute path — such an import resolves through the extern
/// prelude, immune to any `as tracing` rename, and the caller needs to know.
/// A file importing through a re-export is outside what
/// a textual check can settle, and the same rule applies to it as to any unseen
/// path: the row fails loud.
///
/// Only code counts. A string reciting an import line -- a codegen template, a
/// doc example -- binds nothing where it is written.
/// What a `use` item bound: the tracing-side origin name and whether the
/// item was written as an absolute path.
type Binding<'a> = Option<(&'a str, bool)>;

/// A binding ranked for scope resolution: the block it sits in, whether it
/// arrived by glob, and what it bound.
type RankedBinding<'a> = (usize, bool, Binding<'a>);

fn imported_from_tracing<'a>(
    src: &'a str,
    name: &'a str,
    at: usize,
) -> Option<(&'a str, bool, usize)> {
    let code = code_mask(src);
    let mut best: Option<(usize, bool, usize, Binding<'a>)> = None;
    let mut from = 0usize;
    while let Some(relative) = src.get(from..).and_then(|tail| tail.find("use ")) {
        let start = from.saturating_add(relative);
        from = start.saturating_add(4);
        let inside_a_word = start > 0
            && matches!(
                src.as_bytes().get(start.saturating_sub(1)),
                Some(b) if b.is_ascii_alphanumeric() || *b == b'_' || *b == b':'
            );
        if inside_a_word || !code.get(start).copied().unwrap_or(true) {
            continue;
        }
        let end = src
            .get(start..)
            .and_then(|tail| tail.find(';'))
            .map_or(src.len(), |offset| start.saturating_add(offset));
        let item = src
            .get(start.saturating_add(4)..end)
            .unwrap_or_default()
            .trim();
        let Some((scope, is_glob, origin)) = binds_name(src, &code, start, at, item, name) else {
            continue;
        };
        if best.is_none_or(|(farther, incumbent_glob, _, _)| {
            scope > farther || (scope == farther && incumbent_glob && !is_glob)
        }) {
            best = Some((scope, is_glob, start, origin));
        }
    }
    best.and_then(|(_, _, use_start, binding)| {
        binding.map(|(origin, absolute)| (origin, absolute, use_start))
    })
}

/// What the `use` item written at `start` binds `name` to, if it binds it at
/// all: `Some(Some((origin, absolute)))` when the binding arrives from the
/// `tracing` root — absolute stating whether the item was written as a `::`
/// path — `Some(None)` when it arrives from anywhere else, since a binding
/// from another crate is still a binding and it shadows, and `None` when the
/// item never carries this name into scope. The item counts only when its
/// block reaches the call site `at`; `scope` is the block's rank, `0` for the
/// file root, so the caller can let the innermost binder of the name win.
fn binds_name<'a>(
    src: &'a str,
    code: &[bool],
    start: usize,
    at: usize,
    item: &'a str,
    name: &'a str,
) -> Option<RankedBinding<'a>> {
    let mut bound: Option<Option<&'a str>> = None;
    let is_glob = !item.contains('{') && item.ends_with("::*");
    if let Some(brace) = item.find('{') {
        let (path, opened) = item.split_at(brace);
        let inner = opened
            .get(1..opened.rfind('}').unwrap_or(opened.len()))
            .unwrap_or_default();
        for entry in inner.split(',') {
            let entry = entry.trim();
            if entry.is_empty() || entry.ends_with("::*") {
                continue;
            }
            let (source, binding) = match split_as(entry) {
                Some((source, binding)) => (source, binding),
                None => (entry, entry),
            };
            if binding == name {
                bound = Some(
                    under_tracing_root(path)
                        .then_some(source)
                        .filter(|source| !source.contains("::")),
                );
                break;
            }
        }
    } else {
        let (left, renamed) = match split_as(item) {
            Some((left, right)) => (left, Some(right)),
            None => (item, None),
        };
        if let Some(stem) = left.strip_suffix("::*") {
            if under_tracing_root(stem) {
                bound = Some(Some(name));
            }
        } else if left != "*" {
            let last = left.rsplit("::").next().unwrap_or_default();
            if renamed.unwrap_or(last) == name {
                bound = Some(
                    under_tracing_root(
                        left.get(..left.len().saturating_sub(last.len()))
                            .unwrap_or_default(),
                    )
                    .then_some(last),
                );
            }
        }
    }
    let origin = bound?;
    let scope = match enclosing_block(src, code, start) {
        None => 0,
        Some(open) if matching_close(src, open).is_some_and(|close| at >= open && at < close) => {
            open
        }
        Some(_) => return None,
    };
    Some((
        scope,
        is_glob,
        origin.map(|origin| (origin, item.starts_with("::"))),
    ))
}

/// Whether the text before an entry, a name, or a `::*` names the `tracing`
/// crate root itself — with or without the leading `::` an absolute path
/// carries, and with or without the trailing `::` a braced list leaves behind.
fn under_tracing_root(path: &str) -> bool {
    path.trim()
        .trim_start_matches("::")
        .trim()
        .trim_end_matches("::")
        == "tracing"
}

/// Split a `use` item at the word `as`, tolerating any whitespace around it:
/// `left as right` with spaces, tabs or newlines on either side of the word.
fn split_as(item: &str) -> Option<(&str, &str)> {
    let bytes = item.as_bytes();
    let ws = |at: usize| matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r'));
    for i in 1..bytes.len() {
        if i + 2 >= bytes.len() {
            break;
        }
        if matches!((bytes.get(i), bytes.get(i + 1)), (Some(b'a'), Some(b's')))
            && ws(i - 1)
            && ws(i + 2)
        {
            return Some((item[..i].trim(), item[i + 2..].trim()));
        }
    }
    None
}

/// Whether the candidate at `at` sits inside the body of a `macro_rules!`
/// definition that nothing in this file ever invokes. A macro body is prose
/// until a call expands it, so a metric written there feeds no row while the
/// definition sleeps; one invocation anywhere in the file wakes it.
fn dormant_macro_body(src: &str, code: &[bool], at: usize) -> bool {
    let marker = "macro_rules!";
    let mut from = 0usize;
    while let Some(relative) = src.get(from..).and_then(|tail| tail.find(marker)) {
        let head = from.saturating_add(relative);
        from = head.saturating_add(marker.len());
        if !code.get(head).copied().unwrap_or(true) {
            continue;
        }
        let mut name_at = from;
        while matches!(
            src.as_bytes().get(name_at),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            name_at = name_at.saturating_add(1);
        }
        let name_start = name_at;
        while matches!(
            src.as_bytes().get(name_at),
            Some(b) if b.is_ascii_alphanumeric() || *b == b'_' || *b >= 0x80
        ) {
            name_at = name_at.saturating_add(1);
        }
        let Some(name) = src.get(name_start..name_at) else {
            continue;
        };
        let mut open = name_at;
        while matches!(src.as_bytes().get(open), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            open = open.saturating_add(1);
        }
        if src.as_bytes().get(open) != Some(&b'{') {
            continue;
        }
        if let Some(close) = matching_close(src, open)
            && at > open
            && at < close
        {
            return !has_invocation(src, code, name);
        }
    }
    false
}

/// Whether `name!` appears in code anywhere in the file, outside its own
/// definition — a definition carries no bang after its name, so it never
/// matches its own needle.
fn has_invocation(src: &str, code: &[bool], name: &str) -> bool {
    let needle = format!("{name}!");
    let mut from = 0usize;
    while let Some(relative) = src.get(from..).and_then(|tail| tail.find(needle.as_str())) {
        let head = from.saturating_add(relative);
        from = head.saturating_add(needle.len());
        let is_the_word = !matches!(
            src.as_bytes().get(head.saturating_sub(1)),
            Some(b) if b.is_ascii_alphanumeric() || *b == b'_' || *b == b':' || *b >= 0x80
        );
        if is_the_word && code.get(head).copied().unwrap_or(true) {
            return true;
        }
    }
    false
}

/// The index of the innermost `{` whose `}` has not yet closed at `at`, seen
/// through comments and strings; `None` at the file root. Blocks are braces
/// only: an item binds within the braces it sits in and every block nested
/// inside them, which is exactly how a `use` reaches the names it shadows.
fn enclosing_block(src: &str, code: &[bool], at: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut i = at;
    while i > 0 {
        i = i.saturating_sub(1);
        if !code.get(i).copied().unwrap_or(true) {
            continue;
        }
        match bytes.get(i) {
            Some(b'}') => depth = depth.saturating_add(1),
            Some(b'{') if depth == 0 => return Some(i),
            Some(b'{') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
}

/// Whether a `macro_rules!` definition of this very name reaches the call site
/// `at`. A macro definition binds textually — from the definition to the end
/// of the block it sits in and into the blocks nested there — and a local
/// definition outranks any import of the same spelling, so the call expands
/// code this scan cannot read. Unfollowable is unfollowable: the caller
/// refuses the site rather than crediting an expansion that may run no event.
fn shadowed_by_macro_rules(src: &str, code: &[bool], name: &str, at: usize) -> bool {
    let marker = "macro_rules!";
    let mut from = 0usize;
    while let Some(relative) = src.get(from..).and_then(|tail| tail.find(marker)) {
        let head = from.saturating_add(relative);
        from = head.saturating_add(marker.len());
        if !code.get(head).copied().unwrap_or(true) {
            continue;
        }
        let mut after = from;
        while matches!(
            src.as_bytes().get(after),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            after = after.saturating_add(1);
        }
        let word = after;
        while matches!(
            src.as_bytes().get(after),
            Some(b) if b.is_ascii_alphanumeric() || *b == b'_'
        ) {
            after = after.saturating_add(1);
        }
        if src.get(word..after) != Some(name) {
            continue;
        }
        // Textual order: a definition written below the call shadows nothing.
        if head >= at {
            continue;
        }
        match enclosing_block(src, code, head) {
            None => return true,
            Some(open) if matching_close(src, open).is_some_and(|close| at < close) => {
                return true;
            }
            Some(_) => {}
        }
    }
    false
}

/// Whether the name `tracing` itself has been rebound by the time of the call
/// site `at`, so paths spelled `tracing::…` spell something else. Renames bind
/// and rank exactly as the imports `binds_name` reads — a braced list renames
/// through any of its entries — and where two bind the name at one spot the
/// deeper block wins: a `use ::tracing as tracing;` written nearer than an
/// outer rename anchors the name back to the crate and lifts the veto, while
/// any other nearer binding keeps it. Only absolute paths restore: a relative
/// one resolves through the rename it stands under.
fn tracing_renamed_in_scope(src: &str, code: &[bool], at: usize) -> bool {
    let mut best: Option<(usize, bool)> = None;
    let mut from = 0usize;
    while let Some(relative) = src.get(from..).and_then(|tail| tail.find("use ")) {
        let start = from.saturating_add(relative);
        from = start.saturating_add(4);
        let inside_a_word = start > 0
            && matches!(
                src.as_bytes().get(start.saturating_sub(1)),
                Some(b) if b.is_ascii_alphanumeric() || *b == b'_' || *b == b':'
            );
        if inside_a_word || !code.get(start).copied().unwrap_or(true) {
            continue;
        }
        let end = src
            .get(start..)
            .and_then(|tail| tail.find(';'))
            .map_or(src.len(), |offset| start.saturating_add(offset));
        let item = src
            .get(start.saturating_add(4)..end)
            .unwrap_or_default()
            .trim();
        let mut candidate: Option<bool> = None;
        if let Some(brace) = item.find('{') {
            let (path, opened) = item.split_at(brace);
            let inner = opened
                .get(1..opened.rfind('}').unwrap_or(opened.len()))
                .unwrap_or_default();
            for entry in inner.split(',') {
                let entry = entry.trim();
                let (source, binding) = match split_as(entry) {
                    Some((source, binding)) => (source, binding),
                    None => (entry, entry),
                };
                if binding != "tracing" {
                    continue;
                }
                // `use ::tracing::{self as tracing};` restores the crate; any
                // other root — and any relative spelling — renames it.
                candidate = Some(
                    source == "self" && path.trim().starts_with("::") && under_tracing_root(path),
                );
                break;
            }
        } else if let Some((left, binding)) = split_as(item)
            && binding == "tracing"
        {
            if left == "tracing" {
                // A relative self-import binds nothing new; an outer rename
                // still speaks through its own binding.
                continue;
            }
            candidate = Some(left == "::tracing");
        }
        let Some(restore) = candidate else {
            continue;
        };
        let scope = match enclosing_block(src, code, start) {
            None => 0,
            Some(open)
                if matching_close(src, open).is_some_and(|close| at >= open && at < close) =>
            {
                open
            }
            Some(_) => continue,
        };
        if best.is_none_or(|(farther, _)| scope > farther) {
            best = Some((scope, restore));
        }
    }
    // The deepest binding of the name reaching the call decides: only a
    // restore — never a rename — clears the veto.
    best.is_some_and(|(_, restore)| !restore)
}

/// Whether an argument list names the target the metrics layer keeps.
///
/// Two spellings are recognised: the literal most shipped emitters write, and
/// `MetricsLayer::TARGET`, which the rest write and the crate's test emitters
/// use. The layer drops an event whose target differs from its own, so a field
/// aimed anywhere else never reaches a row however faithfully it names one.
///
/// The match is textual, deliberately. An alias holding the same string —
/// `Self::TARGET`, or a local `const` — is a working emitter this gate cannot
/// see, because deciding that would mean following `use` and `const`
/// declarations across files. Such an emitter reads as no site at all, which is
/// the failure this audit can hear about: the row fails loud rather than passing
/// on a target the layer would have dropped. So the rule the crate holds to is
/// that an emitter writes one of the two spellings above.
///
/// Only an argument of the call itself can set an event's target, so `target:`
/// is read at the start of a top-level argument — never inside a string literal
/// or a nested group. An event whose message merely quotes the field still
/// lands on the module path, and a `target:` handed to something nested belongs
/// to that call rather than to the event.
fn aimed_at_metrics(args: &str, metric_at: usize) -> bool {
    top_level_arguments(args).into_iter().any(|span| {
        if span.end > metric_at {
            return false;
        }
        let argument = args.get(span).unwrap_or_default().trim_start();
        let Some(after) = argument.strip_prefix("target:") else {
            return false;
        };
        let value = after.trim_start();
        value.starts_with("\"loopctl::metrics\"") || value.starts_with("MetricsLayer::TARGET")
    })
}

/// The spans of `args`'s top-level arguments, split at the commas between them.
///
/// A comma inside a string, a char literal or a nested bracket group belongs to
/// something else. Splitting there would invent an argument starting wherever a
/// message happened to say `target:`, which is the false site this replaces.
fn top_level_arguments(args: &str) -> Vec<Range<usize>> {
    let bytes = args.as_bytes();
    let mut spans = Vec::new();
    let (mut start, mut depth, mut index) = (0usize, 0i32, 0usize);
    while let Some(byte) = bytes.get(index).copied() {
        match byte {
            b'/' if bytes.get(index.saturating_add(1)) == Some(&b'/') => {
                index = line_comment_end(bytes, index);
                continue;
            }
            b'r' if bytes.get(index.saturating_add(1)) == Some(&b'#') => {
                // `r#type` is a raw identifier, not the head of a string: only a
                // terminator makes it one, and reading an identifier as an
                // unterminated string would hide the rest of the file.
                if let Some(past) = raw_string_end(bytes, index) {
                    index = past;
                    continue;
                }
            }
            b'"' => {
                index = string_end(bytes, index).unwrap_or(bytes.len());
                continue;
            }
            b'\'' => {
                index = char_or_lifetime_end(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth = depth.saturating_add(1),
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                spans.push(start..index);
                start = index.saturating_add(1);
            }
            _ => {}
        }
        index = index.saturating_add(1);
    }
    spans.push(start..bytes.len());
    spans
}

/// One past the `//` comment starting at `at`.
fn line_comment_end(bytes: &[u8], at: usize) -> usize {
    (at..bytes.len())
        .find(|&index| bytes.get(index) == Some(&b'\n'))
        .unwrap_or(bytes.len())
}

/// One past the raw string whose `r` is at `at`, any hash count. `None` when
/// the `r#` opens a raw identifier instead, which has no terminator.
fn raw_string_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut hashes = 0usize;
    while bytes.get(at.saturating_add(1).saturating_add(hashes)) == Some(&b'#') {
        hashes = hashes.saturating_add(1);
    }
    let quote = at.saturating_add(1).saturating_add(hashes);
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }
    let content = quote.saturating_add(1);
    let terminator = std::iter::once(b'"')
        .chain(std::iter::repeat_n(b'#', hashes))
        .collect::<Vec<_>>();
    let closed = (content..bytes.len()).find(|&index| {
        bytes
            .get(index..)
            .is_some_and(|tail| tail.starts_with(terminator.as_slice()))
    })?;
    Some(closed.saturating_add(terminator.len()))
}

/// One past the string literal opened at `at`, honouring backslash escapes.
fn string_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut index = at.saturating_add(1);
    while let Some(byte) = bytes.get(index).copied() {
        match byte {
            b'\\' => index = index.saturating_add(2),
            b'"' => return Some(index.saturating_add(1)),
            _ => index = index.saturating_add(1),
        }
    }
    None
}

/// One past a char literal, or past the apostrophe of a lifetime.
///
/// Lifetimes appear in argument types (`&'a str`), and treating one as the open
/// of a literal would swallow the rest of the argument list.
fn char_or_lifetime_end(bytes: &[u8], at: usize) -> usize {
    if bytes.get(at.saturating_add(1)) == Some(&b'\\') {
        return (at.saturating_add(2)..bytes.len())
            .find(|&index| bytes.get(index) == Some(&b'\''))
            .map_or(bytes.len(), |index| index.saturating_add(1));
    }
    if bytes.get(at.saturating_add(2)) == Some(&b'\'') {
        return at.saturating_add(3);
    }
    at.saturating_add(1)
}

/// Which bytes of `src` are code rather than literal text or a comment.
///
/// A bracket typed inside a string is punctuation in a message, not nesting, so
/// both walks consult this instead of counting every byte. Delimiters outside a
/// literal are unaffected: real nesting still reads exactly as it did.
fn code_mask(src: &str) -> Vec<bool> {
    let bytes = src.as_bytes();
    let mut mask = vec![true; bytes.len()];
    let mut index = 0usize;
    while index < bytes.len() {
        let literal_end = match bytes.get(index).copied() {
            Some(b'/') if bytes.get(index.saturating_add(1)) == Some(&b'/') => {
                Some(line_comment_end(bytes, index))
            }
            Some(b'r') if bytes.get(index.saturating_add(1)) == Some(&b'#') => {
                // A raw identifier masks nothing; see `top_level_arguments`.
                raw_string_end(bytes, index)
            }
            Some(b'"') => Some(string_end(bytes, index).unwrap_or(bytes.len())),
            Some(b'\'') => Some(char_or_lifetime_end(bytes, index)),
            _ => None,
        };
        match literal_end {
            Some(past) => {
                for slot in mask.iter_mut().skip(index).take(past.saturating_sub(index)) {
                    *slot = false;
                }
                index = past;
            }
            None => index = index.saturating_add(1),
        }
    }
    mask
}

#[test]
fn a_raw_identifier_masks_nothing_after_it() {
    // `r#type` is an identifier, not the head of a raw string. Read as one with
    // no terminator in sight, it swallows the rest of the file and the audit
    // loses every emitter written below it — generated protobuf shapes are the
    // common case, since a proto field named `type` arrives as `r#type`.
    let source = concat!(
        "pub struct Field {\n    pub r#type: i32,\n}\n",
        "pub fn emit() {\n",
        "    tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.real.row\");\n",
        "}\n",
        "pub fn quoted() -> &'static str { r#\"metric = \"loopctl.dead.row\"\"# }\n",
    );
    assert!(
        emission_site_in(source, "loopctl.real.row"),
        "a raw identifier must not blind the scan to the emitter below it"
    );
    assert!(
        !emission_site_in(source, "loopctl.dead.row"),
        "a real raw string further down is still prose"
    );
}

#[test]
fn a_call_handed_to_a_token_consuming_macro_emits_nothing() {
    // `stringify!` turns the tokens inside it into text. The `tracing::debug!`
    // handed to it never runs, so it feeds no row however real it looks, and
    // the same holds for every macro that reads its arguments rather than
    // running them.
    assert!(
        !emission_site_in(
            r#"pub fn snippet() -> &'static str { stringify!(tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row")) }"#,
            "loopctl.dead.row"
        ),
        "a stringified call is text about an emission, not one"
    );
    assert!(
        !emission_site_in(
            r#"const DOC: &str = concat!(env!("CARGO_PKG_VERSION"), tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row"));"#,
            "loopctl.dead.row"
        ),
        "another token consumer swallows the call the same way"
    );
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row", "spoken");"#,
            "loopctl.dead.row"
        ),
        "the plain call stays a site"
    );
}

#[test]
fn an_import_quoted_in_a_string_binds_nothing() {
    // A string holding `use tracing::debug;` is text — a codegen template, a
    // doc example — and binds no name where it is written. The file's own
    // import is the one that decides what a bare `debug!` is.
    let source = concat!(
        "pub const SNIPPET: &str = \"use tracing::debug;\";\n",
        "use log::debug;\n",
        "fn emit() { debug!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\"); }\n",
    );
    assert!(
        !emission_site_in(source, "loopctl.dead.row"),
        "an import quoted inside a literal binds nothing"
    );
}

#[test]
fn a_renamed_import_binds_the_name_it_arrives_as() {
    // `use tracing::debug as dbg;` puts `dbg` in scope and leaves `debug`
    // outside: the binding is what a call site spells.
    let renamed = concat!(
        "use tracing::debug as dbg;\n",
        "fn emit() { dbg!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\"); }\n",
    );
    assert!(
        emission_site_in(renamed, "loopctl.dead.row"),
        "the bound name is tracing's macro"
    );
    let stale = concat!(
        "use tracing::debug as dbg;\n",
        "fn emit() { debug!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\"); }\n",
    );
    assert!(
        !emission_site_in(stale, "loopctl.dead.row"),
        "the renamed-away name is not in this file's scope"
    );
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
    // The unrecognised spelling is deliberate, not an oversight: see
    // `aimed_at_metrics`. A row emitted only through an alias fails here rather
    // than passing on something the layer would drop.
    assert!(
        !emission_site_in(
            "tracing::debug!(target: Self::TARGET, metric = \"loopctl.dead.row\");",
            "loopctl.dead.row"
        ),
        "an alias to the target is out of scope for a textual match"
    );
}

#[test]
fn a_target_inside_a_string_or_a_nested_argument_is_not_a_field() {
    // Only a top-level argument sets an event's target. The same text inside
    // the message, or inside a nested call, leaves the event on the module
    // path — which the layer drops — so neither may be credited as a site.
    let quoted_in_message = concat!(
        "tracing::debug!(r#\"say target: \"loopctl::metrics\" aloud\"#",
        ", metric = \"loopctl.dead.row\");"
    );
    assert!(
        !emission_site_in(quoted_in_message, "loopctl.dead.row"),
        "a message that quotes the field sets no target"
    );
    // A field handed to a nested macro is that macro's input, not a field of
    // the event: the event records `note`, and the name inside never reaches a
    // field visitor.
    let nested = concat!(
        "tracing::debug!(target: \"loopctl::metrics\", ",
        "note = other!(metric = \"loopctl.dead.row\"));"
    );
    assert!(
        !emission_site_in(nested, "loopctl.dead.row"),
        "a field inside a nested macro is not a field of the event"
    );
    // A `target:` after a field is not tracing syntax at all: the option is
    // parsed ahead of the fields, and `debug!(metric = "…", target: "…")` is a
    // compile error, so nothing that cannot compile may be a site.
    assert!(
        !emission_site_in(
            r#"tracing::debug!(metric = "loopctl.dead.row", target: "loopctl::metrics");"#,
            "loopctl.dead.row"
        ),
        "a target that follows a field names no event"
    );
}

#[test]
fn an_emitter_that_only_builds_for_tests_is_not_a_production_site() {
    // A declared row is a promise about the shipped library. An emitter inside
    // `#[cfg(test)]` never compiles into that library, so it cannot be the site
    // that keeps the row, while a `#[cfg(not(test))]` emitter does.
    let source = concat!(
        "pub fn emit() {\n",
        "    tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.real.row\");\n",
        "}\n",
        "#[cfg(test)]\n",
        "mod checks {\n",
        "    fn only_in_tests() {\n",
        "        tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.test.only.row\");\n",
        "    }\n",
        "}\n",
        "#[cfg(not(test))]\n",
        "pub fn still_ships() {\n",
        "    tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.still.real.row\");\n",
        "}\n",
    );
    let stripped = strip_cfg_test(source);
    let sources = [stripped.clone()];
    assert!(
        has_emission_site(&sources, "loopctl.real.row"),
        "a production emitter must survive the exclusion"
    );
    assert!(
        has_emission_site(&sources, "loopctl.still.real.row"),
        "cfg(not(test)) is the shipped half, so its emitter counts"
    );
    assert!(
        !has_emission_site(&sources, "loopctl.test.only.row"),
        "an emitter that exists only under cfg(test) ships nothing"
    );
    assert_eq!(
        stripped.len(),
        source.len(),
        "blanking preserves offsets, so the walks still see whole calls"
    );
}

#[test]
fn a_cfg_attribute_quoted_in_a_string_blanks_nothing() {
    // `strip_cfg_test` looks for an attribute, and a string can hold the
    // characters of one. A match read inside a literal opens a blanked range
    // that runs to the end of the next item, swallowing a real emitter with it.
    let quoted = concat!(
        "pub fn note() -> &'static str { \"annotate with #[cfg(test)]\" }\n",
        "pub fn emit() {\n",
        "    tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.real.row\");\n",
        "}\n",
    );
    let sources = [strip_cfg_test(quoted)];
    assert!(
        has_emission_site(&sources, "loopctl.real.row"),
        "an attribute quoted inside a string must not blank the item after it"
    );
    // A real attribute still blanks its item, and only its item.
    let real = concat!(
        "pub fn emit() {\n",
        "    tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.real.row\");\n",
        "}\n",
        "#[cfg(test)]\n",
        "mod checks {\n",
        "    fn t() {\n",
        "        tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.test.row\");\n",
        "    }\n",
        "}\n",
    );
    let real_sources = [strip_cfg_test(real)];
    assert!(
        has_emission_site(&real_sources, "loopctl.real.row"),
        "the production emitter outside the attribute still counts"
    );
    assert!(
        !has_emission_site(&real_sources, "loopctl.test.row"),
        "the item the attribute governs is still excluded"
    );
}

#[test]
fn a_metric_event_must_come_from_tracing() {
    // The final name segment is only half the question. A macro named `info!`
    // from another crate records its fields somewhere else entirely, and a
    // subscriber that never sees the event books no row.
    assert!(
        !emission_site_in(
            r#"log::info!(target: "loopctl::metrics", metric = "loopctl.dead.row", "m");"#,
            "loopctl.dead.row"
        ),
        "an event macro from another crate reaches no metrics subscriber"
    );
    assert!(
        !emission_site_in(
            r#"crate::telemetry::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row");"#,
            "loopctl.dead.row"
        ),
        "a project macro borrowing the name is not tracing"
    );
    assert!(
        emission_site_in(
            "::tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\");",
            "loopctl.dead.row"
        ),
        "the absolute path names the same macro"
    );
    // Unqualified is still decidable, because the file says which macro it is.
    let from_tracing = concat!(
        "use tracing::debug;\n",
        "fn emit() { debug!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\"); }\n",
    );
    assert!(
        emission_site_in(from_tracing, "loopctl.dead.row"),
        "a `use tracing::debug` import makes the bare name tracing's own"
    );
    assert!(
        !emission_site_in(
            "use log::debug;\nfn emit() { debug!(target: \"loopctl::metrics\", metric = \"loopctl.dead.row\"); }\n",
            "loopctl.dead.row"
        ),
        "the same bare name imported from log is not an emission"
    );
}

#[test]
fn a_metric_field_quoted_in_an_argument_is_not_the_field_of_the_event() {
    // A genuine metrics event can carry a string that quotes the field — a
    // hint, a catalog dump. The quoted name reaches no field visitor, so it
    // must not stand in as the emitter of the row it spells.
    let call = format!(
        "tracing::debug!(target: \"loopctl::metrics\", metric = \"loopctl.real.row\", r#\"{hint}\"#);",
        hint = r#"example: debug!(metric = "loopctl.dead.row")"#
    );
    assert!(
        emission_site_in(&call, "loopctl.real.row"),
        "the event's own field is a site"
    );
    assert!(
        !emission_site_in(&call, "loopctl.dead.row"),
        "a row named only inside the message is not emitted by it"
    );
}

#[test]
fn a_delimiter_inside_a_string_is_not_nesting() {
    // Locating the call a field belongs to asks which brackets wrap it. A
    // bracket written inside a message is punctuation in prose, not nesting,
    // and counting it reads as the enclosing bracket, which makes the real call
    // unlocatable.
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", note = "retry [attempt 1", metric = "loopctl.dead.row");"#,
            "loopctl.dead.row"
        ),
        "a bracket inside a value before the field must not hide the call"
    );
    // A delimiter inside a *field value* ahead of the metric field: read as
    // nesting, it closes the call early and the arguments handed to the target
    // check stop short of the field.
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", note = "saw ) here", metric = "loopctl.dead.row");"#,
            "loopctl.dead.row"
        ),
        "a bracket inside a field value must not end the argument list early"
    );
    // And the shipped shape: options and fields first, message last, with an
    // unmatched bracket inside that message.
    assert!(
        emission_site_in(
            r#"tracing::debug!(target: "loopctl::metrics", metric = "loopctl.dead.row", "closing ) paren");"#,
            "loopctl.dead.row"
        ),
        "a bracket inside the trailing message must not drop the field"
    );
    // And the walk must still respect real nesting: a field handed to a nested
    // macro belongs to that macro, whose arguments have nothing to say about
    // the outer event's target.
    assert!(
        !emission_site_in(
            r#"tracing::debug!(target: "loopctl::engine", wrap(metric = "loopctl.dead.row"));"#,
            "loopctl.dead.row"
        ),
        "nesting outside a string still changes which call the field belongs to"
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

#[test]
fn an_aliased_tracing_import_reclaims_the_spelling_it_binds() {
    let aliased = r#"
use tracing::debug as stringify;
fn emit() {
    stringify!(target: "loopctl::metrics", metric = "loopctl.alias.count", n = 1);
}
"#;
    assert!(
        emission_site_in(aliased, "loopctl.alias.count"),
        "stringify! imported from tracing emits: the event runs, the macro does not"
    );
    let plain = r#"
fn emit() {
    stringify!(target: "loopctl::metrics", metric = "loopctl.plain.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(plain, "loopctl.plain.count"),
        "the same spelling with no tracing import still eats its tokens"
    );
}

#[test]
fn a_braced_or_bracketed_consumer_still_swallows_its_tokens() {
    let braced = r#"
fn emit() {
    stringify! { tracing::debug!(target: "loopctl::metrics", metric = "loopctl.brace.count", n = 1) };
    stringify![tracing::debug!(target: "loopctl::metrics", metric = "loopctl.brack.count", n = 1)];
    stringify! (tracing::debug!(target: "loopctl::metrics", metric = "loopctl.space.count", n = 1));
}
"#;
    for name in [
        "loopctl.brace.count",
        "loopctl.brack.count",
        "loopctl.space.count",
    ] {
        assert!(
            !emission_site_in(braced, name),
            "tokens handed to a consumer emit nothing regardless of delimiter: {name}"
        );
    }
}

#[test]
fn a_sibling_modules_import_binds_nothing_at_the_call_site() {
    let siblings = r#"
mod quiet {
    use tracing::debug;
    fn hush() {}
}
mod loud {
    fn emit() {
        debug!(target: "loopctl::metrics", metric = "loopctl.sibling.count", n = 1);
    }
}
"#;
    assert!(
        !emission_site_in(siblings, "loopctl.sibling.count"),
        "quiet's import does not reach loud's call site"
    );
}

#[test]
fn a_module_level_import_reaches_calls_in_nested_bodies() {
    let nested = r#"
use tracing::debug;
fn outer() {
    fn emit() {
        debug!(target: "loopctl::metrics", metric = "loopctl.nested.count", n = 1);
    }
}
"#;
    assert!(
        emission_site_in(nested, "loopctl.nested.count"),
        "a module-level import covers calls in the bodies nested inside it"
    );
}

#[test]
fn a_crate_root_glob_binds_the_event_macros_under_their_own_names() {
    let rooted = r#"
use tracing::*;
fn emit() {
    debug!(target: "loopctl::metrics", metric = "loopctl.rootglob.count", n = 1);
}
"#;
    assert!(
        emission_site_in(rooted, "loopctl.rootglob.count"),
        "a glob over the tracing root imports debug, so the bare call is an event"
    );
}

#[test]
fn a_submodule_glob_binds_no_event_macros() {
    let sub_glob = r#"
use tracing::field::*;
fn emit() {
    debug!(target: "loopctl::metrics", metric = "loopctl.subglob.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(sub_glob, "loopctl.subglob.count"),
        "a glob over a submodule imports no debug; the bare call binds nothing here"
    );
}

#[test]
fn a_nearer_import_shadows_the_outer_tracing_one() {
    let shadowed = r#"
use tracing::debug;
mod noisy {
    use log::debug;
    fn emit() {
        debug!(target: "loopctl::metrics", metric = "loopctl.shadow.count", n = 1);
    }
}
"#;
    assert!(
        !emission_site_in(shadowed, "loopctl.shadow.count"),
        "the nearer log::debug import shadows the outer tracing one"
    );
}

#[test]
fn a_local_macro_named_like_the_event_is_no_import() {
    let shadowed = r#"
use tracing::debug;
fn main() {
    macro_rules! debug {
        ($($t:tt)*) => {};
    }
    debug!(target: "loopctl::metrics", metric = "loopctl.local.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(shadowed, "loopctl.local.count"),
        "the file's own `debug` shadows the import below its definition: the call \
         expands code this scan cannot follow, so it credits nothing"
    );
}

#[test]
fn a_crate_renamed_to_tracing_is_never_tracing() {
    let qualified = r#"
use log as tracing;
fn emit() {
    tracing::debug!(target: "loopctl::metrics", metric = "loopctl.qualified.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(qualified, "loopctl.qualified.count"),
        "a path spelled tracing:: under an `as tracing` rename spells the other crate"
    );
    let bare = r#"
use log as tracing;
use tracing::debug;
fn emit() {
    debug!(target: "loopctl::metrics", metric = "loopctl.bare.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(bare, "loopctl.bare.count"),
        "an import written under the rename arrives from the other crate too"
    );
}

#[test]
fn a_braced_as_tracing_rename_still_vetoes() {
    let braced = r#"
use other::{inner as tracing};
fn emit() {
    tracing::debug!(target: "loopctl::metrics", metric = "loopctl.braced.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(braced, "loopctl.braced.count"),
        "a braced rename lands the name `tracing` just as a bare one does"
    );
}

#[test]
fn an_absolute_tracing_path_ignores_the_rename() {
    let absolute_import = r#"
use log as tracing;
use ::tracing::debug as emit_metric;
fn emit() {
    emit_metric!(target: "loopctl::metrics", metric = "loopctl.absolute_import.count", n = 1);
}
"#;
    assert!(
        emission_site_in(absolute_import, "loopctl.absolute_import.count"),
        "an import anchored to the extern prelude resolves through no local rename"
    );
    let absolute_call = r#"
use log as tracing;
fn emit() {
    ::tracing::debug!(target: "loopctl::metrics", metric = "loopctl.absolute_call.count", n = 1);
}
"#;
    assert!(
        emission_site_in(absolute_call, "loopctl.absolute_call.count"),
        "a call spelled ::tracing:: rides out the rename the same way"
    );
}

#[test]
fn a_self_restore_under_a_rename_restores_the_name() {
    let restored = r#"
use log as tracing;
fn outer() {
    use ::tracing as tracing;
    tracing::debug!(target: "loopctl::metrics", metric = "loopctl.restore.count", n = 1);
}
"#;
    assert!(
        emission_site_in(restored, "loopctl.restore.count"),
        "the nearer absolute self-restore wins the name, so the call is tracing's"
    );
    let rebound = r#"
use ::tracing as tracing;
fn outer() {
    use log as tracing;
    tracing::debug!(target: "loopctl::metrics", metric = "loopctl.rebound.count", n = 1);
}
"#;
    assert!(
        !emission_site_in(rebound, "loopctl.rebound.count"),
        "and the nearer rename wins the other way, vetting the outer restore"
    );
}

#[test]
fn a_shadow_definition_spares_earlier_and_outside_calls() {
    let before = r#"
use tracing::debug;
fn main() {
    debug!(target: "loopctl::metrics", metric = "loopctl.before.count", n = 1);
    macro_rules! debug {
        ($($t:tt)*) => {};
    }
}
"#;
    assert!(
        emission_site_in(before, "loopctl.before.count"),
        "a definition written below the call shadows nothing above it"
    );
    let outside = r#"
use tracing::debug;
fn main() {
    {
        macro_rules! debug {
            ($($t:tt)*) => {};
        }
    }
    debug!(target: "loopctl::metrics", metric = "loopctl.outside.count", n = 1);
}
"#;
    assert!(
        emission_site_in(outside, "loopctl.outside.count"),
        "a definition in a block that closed before the call reaches no further"
    );
}

/// A rename is spaced however its writer pleased: the veto reads the item as
/// a parser does, not as a single-space literal, so a newline, a tab, or a
/// stray double space between `as` and `tracing` changes nothing.
#[test]
fn a_rename_spaced_any_way_still_vetoes() {
    for spaced in [
        "use other as\n    tracing;",
        "use other as\ttracing;",
        "use other  as tracing;",
        "use other as tracing ;",
    ] {
        let source = format!(
            "{spaced}\nfn emit() {{ tracing::debug!(target: \"loopctl::metrics\", \
             metric = \"loopctl.spaced.count\", n = 1); }}"
        );
        assert!(
            !emission_site_in(&source, "loopctl.spaced.count"),
            "the rename however spaced still vetoes: {spaced:?}"
        );
    }
}

/// Rust ranks a named import above a glob written beside it in the same
/// block, whatever the textual order, so the audit must let the named one
/// decide what the bare call is.
#[test]
fn a_named_import_outranks_a_same_scope_glob() {
    let others_named = r#"
use tracing::*;
use other::debug;
fn emit() { debug!(target: "loopctl::metrics", metric = "loopctl.globorder.count", n = 1); }
"#;
    assert!(
        !emission_site_in(others_named, "loopctl.globorder.count"),
        "the named other import wins however late it is written"
    );
    let tracings_named = r#"
use tracing::debug;
use other::*;
fn emit() { debug!(target: "loopctl::metrics", metric = "loopctl.globorder.count", n = 1); }
"#;
    assert!(
        emission_site_in(tracings_named, "loopctl.globorder.count"),
        "the named tracing import wins however early the glob is written"
    );
}

/// Only the `::tracing` root restores the name: a braced `self as tracing`
/// arriving from any other crate is a rename like any other.
#[test]
fn a_wrong_root_self_restore_is_a_rename() {
    let wrong_root = r#"
use ::other::{self as tracing};
fn emit() { tracing::debug!(target: "loopctl::metrics", metric = "loopctl.root.count", n = 1); }
"#;
    assert!(
        !emission_site_in(wrong_root, "loopctl.root.count"),
        "::other::{{self as tracing}} lands the name on other, not tracing"
    );
}

/// `use tracing as tracing;` binds nothing new, so it neither vetoes nor
/// restores: the crate name is exactly what it already was.
#[test]
fn a_relative_self_import_binds_nothing() {
    let self_import = r#"
use tracing as tracing;
fn emit() { tracing::debug!(target: "loopctl::metrics", metric = "loopctl.self.count", n = 1); }
"#;
    assert!(
        emission_site_in(self_import, "loopctl.self.count"),
        "a relative self-import is a no-op, and the call stays tracing's"
    );
}

/// A bare import resolves where its `use` line sits, so a rename written in a
/// deeper block cannot rebind it retroactively: the call it serves still
/// emits.
#[test]
fn a_deeper_rename_does_not_rebind_an_earlier_import() {
    let deeper = r#"
use tracing::debug as emit_metric;
fn outer() {
    use other as tracing;
    emit_metric!(target: "loopctl::metrics", metric = "loopctl.deeper.count", n = 1);
}
"#;
    assert!(
        emission_site_in(deeper, "loopctl.deeper.count"),
        "the import resolved before the inner rename existed"
    );
}

/// `pub use` binds locally like any `use`, in both directions: a public
/// rename vetoes and a public import credits.
#[test]
fn a_pub_use_binds_like_a_use() {
    let pub_rename = r#"
pub use other as tracing;
fn emit() { tracing::debug!(target: "loopctl::metrics", metric = "loopctl.pub.count", n = 1); }
"#;
    assert!(
        !emission_site_in(pub_rename, "loopctl.pub.count"),
        "a public rename reaches the call site like a private one"
    );
    let pub_import = r#"
pub use tracing::debug;
fn emit() { debug!(target: "loopctl::metrics", metric = "loopctl.pub.count", n = 1); }
"#;
    assert!(
        emission_site_in(pub_import, "loopctl.pub.count"),
        "a public import binds the name locally like any use"
    );
}

/// A metric field written inside a macro body is prose until something
/// expands it: a definition nothing calls emits nothing, while one invocation
/// anywhere in the file makes the body a live site.
#[test]
fn a_metric_inside_a_macro_body_counts_only_when_the_macro_runs() {
    let dormant = r#"
use tracing::debug;
macro_rules! never_called {
    () => { debug!(target: "loopctl::metrics", metric = "loopctl.dormant.count", n = 1) };
}
"#;
    assert!(
        !emission_site_in(dormant, "loopctl.dormant.count"),
        "a definition nothing invokes never runs, so its body emits nothing"
    );
    let invoked = r#"
use tracing::debug;
macro_rules! emit_it {
    () => { debug!(target: "loopctl::metrics", metric = "loopctl.invoked.count", n = 1) };
}
fn f() { emit_it!(); }
"#;
    assert!(
        emission_site_in(invoked, "loopctl.invoked.count"),
        "one invocation anywhere expands the body into a live site"
    );
    let unicode_name = concat!(
        "use tracing::debug;\n",
        "macro_rules! café {\n",
        "    () => { debug!(target: \"loopctl::metrics\", metric = \"loopctl.unicode.macro\", n = 1) };\n",
        "}\n",
    );
    assert!(
        !emission_site_in(unicode_name, "loopctl.unicode.macro"),
        "a never-invoked macro keeps sleeping whatever alphabet names it"
    );
}

/// Identifiers in Rust span Unicode, and a `use` item may carry one: the
/// rename matcher reads bytes at those offsets and must survive them.
#[test]
fn a_unicode_identifier_survives_the_scan() {
    let plain = r#"
use crate::café;
use tracing::debug;
fn f() { debug!(target: "loopctl::metrics", metric = "loopctl.unicode.count", n = 1); }
"#;
    assert!(
        emission_site_in(plain, "loopctl.unicode.count"),
        "a café import is legal rust and the scan must not panic on it"
    );
    let renamed = r#"
use crate::café as tracing;
fn f() { tracing::debug!(target: "loopctl::metrics", metric = "loopctl.unicode.count", n = 1); }
"#;
    assert!(
        !emission_site_in(renamed, "loopctl.unicode.count"),
        "the rename matcher reads bytes too: café as tracing vetoes"
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
