//! Miss repair for `Edit` and `MultiEdit`: classify the miss, echo
//! the file's actual lines, and apply unique whitespace-normalized
//! matches.
//!
//! Session-history mining (2026-10-18, 70 failed Edit/MultiEdit calls
//! across 22 sessions) shows the misses cluster into three root
//! causes the tools can distinguish: a stale read (the file changed
//! since the model last saw it), a whitespace-only mismatch (the
//! text is right but for indentation width, trailing whitespace, or
//! line endings), and no similar passage (the model paraphrased or
//! hallucinated text that was never there). Every miss costs a full
//! round-trip — usually a `Read` repeating what the error could have
//! carried — so the miss path here classifies the cause and echoes
//! the file's actual lines around the best candidate window, and a
//! **unique** normalized match applies outright, disclosed in the
//! receipt.

/// Normalize one line for whitespace-tolerant comparison.
///
/// Strips the line ending and all leading and trailing whitespace,
/// so a line matches its twin regardless of indentation width,
/// trailing spaces, or CRLF versus LF. Empty (or all-whitespace)
/// lines normalize to the empty string and match each other.
fn normalized_line(line: &str) -> &str {
    line.trim()
}

/// A run of aligned line indices in the file's content.
///
/// Line `i` of the content paired with line `i` of the old text;
/// the window is the candidate region a miss is compared against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LineWindow {
    /// The first content line index of the window, 0-based.
    ///
    /// Paired with `end` for range math; splicing converts through
    /// [`window_range`](self).
    start: usize,

    /// One past the last content line index of the window.
    ///
    /// Exclusive, so `end - start` is the window's line count.
    end: usize,
}

/// How many lines a rendered echo may carry at most.
///
/// The echo replaces a re-read, not the whole file; the bound keeps
/// a miss against a huge file from flooding the context (the
/// everything-the-model-reads-is-bounded rule).
const ECHO_MAX_LINES: usize = 16;

/// How many bytes a rendered echo may carry at most.
///
/// The char-boundary-respecting companion of [`ECHO_MAX_LINES`].
const ECHO_MAX_BYTES: usize = 2048;

/// The classification of an `old_text` miss, with the evidence the
/// repair needs.
///
/// Built by [`classify_miss`] against the file's actual content; the
/// error message renders from it, and the whitespace-tolerant apply
/// path consumes its unique windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MissKind {
    /// The text matches the file modulo whitespace, in exactly one
    /// place.
    ///
    /// The carried window is the file's line range that matches
    /// after normalization — the apply path splices it with the
    /// file's own bytes.
    WhitespaceUnique(LineWindow),

    /// The text matches the file modulo whitespace, in `count`
    /// places.
    ///
    /// Ambiguity never applies: the miss keeps rejecting, with the
    /// first window echoed.
    WhitespaceAmbiguous {
        /// How many distinct regions match after normalization.
        ///
        /// Two or more — one is
        /// [`WhitespaceUnique`](MissKind::WhitespaceUnique).
        count: usize,

        /// The first matching region, for the echo.
        ///
        /// The earliest by line index; later regions are count-only.
        first: LineWindow,
    },

    /// The file resembles the text near the best window but differs
    /// in content — the stale-read shape.
    ///
    /// The first differing aligned line rides with it, both sides
    /// quoted.
    Stale {
        /// The best-scoring window in the file.
        ///
        /// Highest normalized-overlap score; ties keep the earliest.
        window: LineWindow,

        /// The 1-based line number of the first differing line.
        ///
        /// 1-based to match the echo's rendered numbers.
        line_no: usize,

        /// The file's actual line at the differing position.
        ///
        /// Quoted with `{:?}` so whitespace differences stay visible.
        file_line: String,

        /// The old text's line at the differing position.
        ///
        /// Quoted with `{:?}` beside the file's, so the first
        /// difference reads at a glance.
        old_line: String,
    },

    /// Nothing in the file resembles the text — the paraphrase or
    /// hallucination shape.
    ///
    /// No echo exists for this kind; the classification itself says
    /// the repair is a re-read, not a re-match.
    NoPassage,
}

/// Split content into lines with their byte offsets.
///
/// Line start offsets are computed by a single scan, so a window's
/// byte range (for splicing with the file's actual bytes) is exact
/// under both LF and CRLF endings.
fn line_table(content: &str) -> Vec<(usize, &str)> {
    let mut table = Vec::new();
    let mut offset: usize = 0;
    for line in content.split('\n') {
        let stripped = line.strip_suffix('\r').unwrap_or(line);
        table.push((offset, stripped));
        let advance = line.len().saturating_add(1);
        offset = offset.saturating_add(advance);
    }
    if let Some(&(last_offset, last)) = table.last()
        && last.is_empty()
    {
        table.pop();
        let _ = last_offset;
    }
    table
}

/// Whether two normalized lines resemble each other.
///
/// Word-overlap at the half mark (computed without floats): the file
/// line `fn two() { todo!() }` resembles the old text's
/// `fn two() {}` — the stale shape — while a paraphrase shares no
/// half of its words with anything in the file.
/// A line prepared for window comparison: its normalized form and
/// its token set, computed once per line instead of once per compared
/// pair.
struct PreparedLine<'a> {
    /// The whitespace-normalized text of the line.
    ///
    /// The comparison form: line ending stripped, edges trimmed, so
    /// indentation width and trailing whitespace never count as
    /// content differences.
    normalized: &'a str,

    /// The whitespace-split token set of the normalized line.
    ///
    /// The resemblance comparison's building block — word-overlap at
    /// the half mark between two lines' sets is the stale shape's
    /// signal.
    tokens: std::collections::HashSet<&'a str>,
}

/// Prepare one line for comparison.
///
/// Normalizes once and splits once: the window loop compares
/// O(file x old) pairs and must never rebuild either per pair —
/// a ten-`MiB` file against a fifty-line old text would otherwise
/// allocate millions of token sets on the executor thread.
fn prepare_line(line: &str) -> PreparedLine<'_> {
    let normalized = normalized_line(line);
    let tokens = normalized.split_whitespace().collect();
    PreparedLine { normalized, tokens }
}

/// Whether two prepared lines resemble each other.
///
/// Word-overlap at the half mark (computed without floats): the file
/// line `fn two() { todo!() }` resembles the old text's
/// `fn two() {}` — the stale shape — while a paraphrase shares no
/// half of its words with anything in the file.
fn lines_resemble(a: &PreparedLine<'_>, b: &PreparedLine<'_>) -> bool {
    if a.normalized == b.normalized {
        return true;
    }
    if a.tokens.is_empty() && b.tokens.is_empty() {
        return true;
    }
    let intersection = a.tokens.intersection(&b.tokens).count();
    let union = a.tokens.union(&b.tokens).count();
    intersection.saturating_mul(2) >= union
}

/// Classify an `old_text` miss against the file's actual content.
///
/// Exact matches never reach here. The classifier first looks for
/// windows whose lines all match after normalization (the
/// whitespace-only shape — unique windows are the tolerant-apply
/// candidates, several are ambiguous); failing that, it scores every
/// window by normalized line overlap and reports the best as stale
/// when the resemblance clears the half-of-lines bar **and** the
/// window is anchored — a one-line old text anchors on its similar
/// line, a longer one needs a first or last line that matches
/// normalized *and* is not generic (a lone `}` or blank), so one
/// generic line in an unrelated region cannot masquerade as the
/// neighborhood; otherwise no similar passage.
pub(crate) fn classify_miss(content: &str, old_text: &str) -> MissKind {
    let content_lines = line_table(content);
    let old_lines: Vec<&str> = old_text.lines().collect();
    let old_len = old_lines.len();
    if old_len == 0 || content_lines.len() < old_len {
        return MissKind::NoPassage;
    }
    let content_prepared: Vec<PreparedLine<'_>> = content_lines
        .iter()
        .map(|(_, line)| prepare_line(line))
        .collect();
    let old_prepared: Vec<PreparedLine<'_>> =
        old_lines.iter().map(|line| prepare_line(line)).collect();

    let mut whitespace_windows: Vec<LineWindow> = Vec::new();
    let mut best = None;
    let mut best_score: usize = 0;
    let last_start = content_lines.len().saturating_sub(old_len);
    for start in 0..=last_start {
        let mut score: usize = 0;
        let mut all_match = true;
        let mut anchored = false;
        let mut first_diff: Option<(usize, &str, &str)> = None;
        for (index, old_prepared_line) in old_prepared.iter().enumerate() {
            let Some(file_prepared) = content_prepared.get(start.saturating_add(index)) else {
                continue;
            };
            let equal = file_prepared.normalized == old_prepared_line.normalized;
            if equal {
                score = score.saturating_add(1);
                if (index == 0 || index.saturating_add(1) == old_len)
                    && old_prepared_line.normalized.chars().count() > 1
                {
                    anchored = true;
                }
            } else {
                all_match = false;
                if lines_resemble(file_prepared, old_prepared_line) {
                    score = score.saturating_add(1);
                }
                if first_diff.is_none() {
                    first_diff = Some((
                        start.saturating_add(index),
                        file_prepared.normalized,
                        old_prepared_line.normalized,
                    ));
                }
            }
        }
        if all_match {
            whitespace_windows.push(LineWindow {
                start,
                end: start.saturating_add(old_len),
            });
        }
        if old_len == 1 && score == 1 {
            anchored = true;
        }
        if score > best_score {
            best_score = score;
            best = Some((start, first_diff, anchored));
        }
    }

    match whitespace_windows.len() {
        0 => {}
        1 => {
            return MissKind::WhitespaceUnique(
                *whitespace_windows
                    .first()
                    .unwrap_or(&LineWindow { start: 0, end: 0 }),
            );
        }
        count => {
            let first = *whitespace_windows
                .first()
                .unwrap_or(&LineWindow { start: 0, end: 0 });
            return MissKind::WhitespaceAmbiguous { count, first };
        }
    }

    let Some((best_start, first_diff, anchored)) = best else {
        return MissKind::NoPassage;
    };
    let stale_threshold = old_len.div_ceil(2);
    if best_score < stale_threshold || !anchored {
        return MissKind::NoPassage;
    }
    let Some((diff_index, file_line, old_line)) = first_diff else {
        return MissKind::NoPassage;
    };
    MissKind::Stale {
        window: LineWindow {
            start: best_start,
            end: best_start.saturating_add(old_len),
        },
        line_no: diff_index.saturating_add(1),
        file_line: file_line.to_string(),
        old_line: old_line.to_string(),
    }
}

/// The byte range of a line window's *core*, the file's own edge
/// whitespace preserved.
///
/// [`window_range`](self) spans whole lines; this shrinks it past the
/// first line's leading whitespace and the last line's trailing
/// whitespace, so a tolerant apply splices the replacement between
/// the file's own indentation and line-ending region — the file's
/// actual bytes on both edges, not the model's spelling of them.
pub(crate) fn window_core_range(content: &str, window: LineWindow) -> std::ops::Range<usize> {
    let whole = window_range(content, window);
    let table = line_table(content);
    let lead = table.get(window.start).map_or(0, |(_, line)| {
        line.len().saturating_sub(line.trim_start().len())
    });
    let trail = table
        .get(window.end.saturating_sub(1))
        .map_or(0, |(_, line)| {
            line.len().saturating_sub(line.trim_end().len())
        });
    whole.start.saturating_add(lead)..whole.end.saturating_sub(trail)
}

/// The byte range of a line window in the content it came from.
///
/// The range spans from the window's first line start through the
/// end of its last line, exclusive of the trailing line ending — the
/// splice keeps the file's own bytes on both edges.
pub(crate) fn window_range(content: &str, window: LineWindow) -> std::ops::Range<usize> {
    let table = line_table(content);
    let start = table.get(window.start).map_or(0, |(offset, _)| *offset);
    let end = table
        .get(window.end.saturating_sub(1))
        .map_or(content.len(), |(offset, line)| {
            offset.saturating_add(line.len())
        });
    start..end
}

/// Render the file's actual lines around a window, with line
/// numbers, bounded.
///
/// The echo is the re-read the old hints used to demand: the window
/// plus one line of context on each side, `line | text` per line,
/// 1-based numbers, cut to [`ECHO_MAX_LINES`] lines and
/// [`ECHO_MAX_BYTES`] bytes on a char boundary.
pub(crate) fn render_echo(content: &str, window: LineWindow) -> String {
    let table = line_table(content);
    if table.is_empty() {
        return "(the file is empty)".to_string();
    }
    let start = window.start.saturating_sub(1);
    let end = (window.end.saturating_add(1)).min(table.len());
    let shown_start = start;
    let shown_end = end.min(start.saturating_add(ECHO_MAX_LINES));
    let mut rendered = String::new();
    let mut bytes: usize = 0;
    for (index, (_, line)) in table
        .iter()
        .enumerate()
        .skip(shown_start)
        .take(shown_end.saturating_sub(shown_start))
    {
        let entry = format!("{} | {}\n", index.saturating_add(1), line);
        if bytes.saturating_add(entry.len()) > ECHO_MAX_BYTES {
            rendered.push_str("… (echo truncated)\n");
            break;
        }
        bytes = bytes.saturating_add(entry.len());
        rendered.push_str(&entry);
    }
    if end > shown_end {
        rendered.push_str("… (echo truncated)\n");
    }
    rendered
}

use std::fmt::Write as _;

/// Render the classification and the bounded echo for a miss.
///
/// The paragraph pair every miss error carries: a `Classification:`
/// sentence naming the root cause (with the stale quote or the
/// ambiguity count inline), and the file's actual lines around the
/// best candidate window with 1-based numbers. The caller owns the
/// first line and any hints that follow.
pub(crate) fn render_miss(kind: &MissKind, content: &str) -> String {
    let mut rendered = String::new();
    match kind {
        MissKind::WhitespaceUnique(_) => rendered.push_str(
            "Classification: whitespace-only mismatch - but the normalized \
             match is not unique, so it does not apply.\n",
        ),
        MissKind::WhitespaceAmbiguous { count, first } => {
            writeln!(
                rendered,
                "Classification: whitespace-only mismatch - the text matches \
                 {count} regions of the file after normalizing indentation, \
                 line endings, and trailing whitespace; add surrounding \
                 context to disambiguate."
            )
            .ok();
            rendered.push_str("\nThe file's actual lines around the first match:\n");
            rendered.push_str(&render_echo(content, *first));
        }
        MissKind::Stale {
            window,
            line_no,
            file_line,
            old_line,
        } => {
            writeln!(
                rendered,
                "Classification: stale - the file differs from the text near \
                 its closest passage; first differing line {line_no}: the file \
                 has {file_line:?}, old_text has {old_line:?}."
            )
            .ok();
            rendered.push_str("\nThe file's actual lines around the closest match:\n");
            rendered.push_str(&render_echo(content, *window));
        }
        MissKind::NoPassage => rendered.push_str(
            "Classification: no similar passage - the file contains nothing \
             resembling old_text; re-read the file with Read before editing.\n",
        ),
    }
    rendered
}

/// Trim a replacement's edge whitespace, symmetric to the core's
/// shrink.
///
/// [`window_core_range`](self) keeps the file's own first-line
/// leading and last-line trailing whitespace outside the splice — so
/// the replacement must drop its *own* edge whitespace, or the two
/// concatenate: a 4-space file repaired with a 6-space model text
/// lands as a ten-space first line. Interior lines ride verbatim.
pub(crate) fn edge_trimmed_replacement(new_text: &str) -> &str {
    new_text.trim()
}

/// The leading-whitespace width of a line, in characters.
///
/// The model-side measure for the relative-depth mapping: how deep
/// the model spelled the line, counted in the model's own
/// characters.
fn leading_width(line: &str) -> usize {
    line.len().saturating_sub(line.trim_start().len())
}

/// The window's per-line leading-whitespace strings, in the file's
/// own characters.
///
/// Each replacement line aligned to a window line takes that line's
/// leading whitespace verbatim — characters *and* width — so a
/// tab-indented file never receives the model's spaces back, and a
/// delta-zero pair whose characters differ (a Makefile recipe tab
/// versus the model's spaces) is rewritten in the file's spelling
/// all the same.
fn file_indents(content: &str, window: LineWindow) -> Vec<&str> {
    let table = line_table(content);
    let mut indents = Vec::new();
    for (index, (offset, line)) in table.iter().enumerate() {
        if index >= window.start && index < window.end {
            let _ = offset;
            let width = line.len().saturating_sub(line.trim_start().len());
            if let Some(leading) = line.get(..width) {
                indents.push(leading);
            }
        }
    }
    indents
}

/// The line ending the window's own interior breaks use.
///
/// Derived from [`window_range`](self)'s text, which contains every
/// internal break but not the last line's terminator: a CRLF window
/// joins replacement lines with CRLF, an LF window with LF — a
/// repaired region never mixes endings inside one file. A single-
/// line window reads the terminator immediately after it, so a
/// multi-line replacement into a single-line match joins in the
/// file's ending all the same.
fn window_eol(content: &str, window: LineWindow) -> &'static str {
    let range = window_range(content, window);
    if content
        .get(range.clone())
        .is_some_and(|text| text.contains("\r\n"))
    {
        return "\r\n";
    }
    let after = range.end;
    if content
        .get(after..after.saturating_add(2))
        .is_some_and(|terminator| terminator == "\r\n")
    {
        return "\r\n";
    }
    "\n"
}

/// Rewrite each replacement line's leading whitespace in the file's
/// own characters, mapped through `old_text` by content.
///
/// Edits change structure, so position alone cannot name an indent.
/// Each `old_text` line's normalized content is keyed to its
/// window line's actual indent (the pair the match already proved
/// equivalent); a new line whose normalized content matches an old
/// line takes that old line's file indent — a surviving line keeps
/// its own block membership however the edit re-arranges it. A line
/// with no old counterpart (genuinely new) inherits the nearest
/// preceding mapped line's file indent shifted by the model's
/// relative depth to that neighbor's old spelling — the model's
/// nesting intent survives, expressed in the file's characters.
/// Duplicate contents resolve to the first unused candidate at or
/// after the matching cursor — positionally nearest, deterministic.
/// Pure-whitespace lines stay untouched, and the join between lines
/// is the file's own line ending.
fn aligned_replacement(new_text: &str, old_text: &str, indents: &[&str], eol: &str) -> String {
    let old_raw: Vec<&str> = old_text.lines().collect();
    let old_norm: Vec<&str> = old_raw.iter().map(|line| normalized_line(line)).collect();
    let old_widths: Vec<usize> = old_raw.iter().map(|line| leading_width(line)).collect();
    let file_widths: Vec<usize> = indents
        .iter()
        .map(|indent| indent.chars().count())
        .collect();
    let mut model_step = usize::MAX;
    let mut file_step = 1;
    for from in 0..old_widths.len() {
        for to in 0..old_widths.len() {
            if let (Some(&to_width), Some(&from_width)) = (old_widths.get(to), old_widths.get(from))
                && to_width > from_width
            {
                let difference = to_width.saturating_sub(from_width);
                if difference < model_step {
                    model_step = difference;
                    let file_cost = file_widths
                        .get(to)
                        .copied()
                        .unwrap_or(1)
                        .saturating_sub(file_widths.get(from).copied().unwrap_or(1))
                        .max(1);
                    file_step = file_cost;
                }
            }
        }
    }

    let mut used = vec![false; old_norm.len()];
    let mut cursor = 0;
    let mut last_mapped: Option<(usize, &str)> = None;
    let mut rendered = String::with_capacity(new_text.len().saturating_add(16));
    for (index, line) in new_text.lines().enumerate() {
        if index > 0 {
            rendered.push_str(eol);
        }
        if line.trim().is_empty() {
            rendered.push_str(line);
            continue;
        }
        let content = normalized_line(line);
        let mut search = (cursor..old_norm.len()).chain(0..cursor.min(old_norm.len()));
        let hit = search.find(|old_index| {
            !used.get(*old_index).copied().unwrap_or(false)
                && old_norm.get(*old_index) == Some(&content)
        });
        if let Some(old_index) = hit {
            if let Some(flag) = used.get_mut(old_index) {
                *flag = true;
            }
            cursor = old_index.saturating_add(1);
            let indent = indents.get(old_index).copied().unwrap_or("");
            if let Some(width) = old_widths.get(old_index) {
                last_mapped = Some((*width, indent));
            }
            rendered.push_str(indent);
            let content_start = line.len().saturating_sub(line.trim_start().len());
            if let Some(rest) = line.get(content_start..) {
                rendered.push_str(rest);
            }
            continue;
        }
        let model_width = leading_width(line);
        let Some((neighbor_width, neighbor_indent)) = last_mapped else {
            let indent = indents.last().copied().unwrap_or("");
            rendered.push_str(indent);
            let content_start = line.len().saturating_sub(line.trim_start().len());
            if let Some(rest) = line.get(content_start..) {
                rendered.push_str(rest);
            }
            continue;
        };
        let neighbor_file_width = neighbor_indent.chars().count();
        let target_width = if model_step == usize::MAX {
            neighbor_file_width.saturating_add(model_width.saturating_sub(neighbor_width))
        } else {
            let levels = model_width
                .saturating_sub(neighbor_width)
                .checked_div(model_step)
                .unwrap_or(0);
            neighbor_file_width.saturating_add(levels.saturating_mul(file_step))
        };
        let unit = neighbor_indent
            .chars()
            .last()
            .or_else(|| {
                indents
                    .iter()
                    .find(|indent| !indent.is_empty())
                    .and_then(|indent| indent.chars().next())
            })
            .unwrap_or(' ');
        for _ in 0..target_width {
            rendered.push(unit);
        }
        let content_start = line.len().saturating_sub(line.trim_start().len());
        if let Some(rest) = line.get(content_start..) {
            rendered.push_str(rest);
        }
    }
    rendered
}

/// The line ending the window's own interior breaks use.
///
/// Drop only the first line's leading whitespace.
///
/// After the aligned rewrite, the first line carries exactly the
/// file's own indentation — which the core range already keeps
/// outside the splice. Dropping it here keeps the two from stacking;
/// interior lines keep their rewritten indent.
fn first_line_unindented(text: &str) -> String {
    match text.find('\n') {
        Some(cut) => format!(
            "{}{}",
            text.get(..cut).map_or("", str::trim_start),
            text.get(cut..).unwrap_or_default()
        ),
        None => text.trim_start().to_string(),
    }
}

pub(crate) fn normalized_splice(
    content: &str,
    window: LineWindow,
    old_text: &str,
    new_text: &str,
) -> String {
    let indents = file_indents(content, window);
    let eol = window_eol(content, window);
    let trimmed = edge_trimmed_replacement(new_text);
    let aligned = aligned_replacement(trimmed, old_text, &indents, eol);
    let replacement = first_line_unindented(&aligned);
    super::edit::splice(content, window_core_range(content, window), &replacement)
}

/// The disclosure line a whitespace-tolerant apply adds to the
/// receipt.
pub(crate) const NORMALIZED_MATCH_NOTE: &str = "matched with normalized indentation";

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: usize, end: usize) -> LineWindow {
        LineWindow { start, end }
    }

    #[test]
    fn a_whitespace_mismatch_classifies_as_unique() {
        // Indentation width differs, trailing space rides, and the
        // line ending is CRLF — one place in the file.
        let content = "fn one() {}\r\n    fn two() {} \r\nfn three() {}\n";
        let kind = classify_miss(content, "fn two() {}\n");
        assert_eq!(
            kind,
            MissKind::WhitespaceUnique(window(1, 2)),
            "the normalized match is unique: {kind:?}"
        );
    }

    #[test]
    fn several_normalized_matches_classify_as_ambiguous() {
        let content = "  alpha\nbeta\n    alpha\ngamma\n";
        let kind = classify_miss(content, "alpha\n");
        assert_eq!(
            kind,
            MissKind::WhitespaceAmbiguous {
                count: 2,
                first: window(0, 1)
            },
            "two regions match after normalization: {kind:?}"
        );
    }

    #[test]
    fn a_content_difference_classifies_as_stale_with_the_first_diff() {
        let content = "fn one() {}\nfn two() { todo!() }\nfn three() {}\n";
        let kind = classify_miss(content, "fn one() {}\nfn two() {}\nfn three() {}\n");
        let MissKind::Stale {
            line_no,
            file_line,
            old_line,
            ..
        } = kind
        else {
            panic!("a close passage with a content difference is stale: {kind:?}");
        };
        assert_eq!(line_no, 2, "the first differing line is 1-based");
        assert_eq!(file_line, "fn two() { todo!() }");
        assert_eq!(old_line, "fn two() {}", "normalized for comparison");
    }

    #[test]
    fn an_unanchored_single_generic_line_is_no_passage() {
        // The coincidence guard: a short old_text whose one generic
        // matching line (`}`) sits in an unrelated region must not
        // name that region stale - the window must be anchored by a
        // matching first or last line.
        let content = "fn one() {}\nlet x = 1;\n}\nfn three() {}\n";
        let kind = classify_miss(content, "fn two() {}\n}");
        assert_eq!(
            kind,
            MissKind::NoPassage,
            "one generic line in an unrelated region is coincidence: {kind:?}"
        );
    }

    #[test]
    fn a_paraphrase_classifies_as_no_passage() {
        let content = "fn one() {}\nfn two() {}\n";
        let kind = classify_miss(content, "the function that makes tea\nand pours it\n");
        assert_eq!(kind, MissKind::NoPassage, "nothing resembles it");
    }

    #[test]
    fn an_old_text_longer_than_the_file_is_no_passage() {
        let kind = classify_miss("one line\n", "one line\nplus more\nand more\n");
        assert_eq!(kind, MissKind::NoPassage);
    }

    #[test]
    fn the_replacement_takes_the_file_s_indent_characters() {
        // Characters and width both come from the aligned file lines,
        // mapped through old_text by content - the position of a line
        // in the replacement never names its indent.
        assert_eq!(
            aligned_replacement(
                "def f():\n    return 2",
                "def f():\n    return 1",
                &["\t", "\t\t"],
                "\n"
            ),
            "\tdef f():\n\t\treturn 2",
            "content-matched lines take the aligned file indent"
        );
        assert_eq!(
            aligned_replacement("all:\n    echo a", "all:\n    echo a", &["", "\t"], "\n"),
            "all:\n\techo a",
            "a surviving line keeps its own block membership"
        );
        assert_eq!(
            aligned_replacement("        return 2", "        return 1", &["\t"], "\n"),
            "\treturn 2",
            "the over-indent collapses to the file's width and character"
        );
        assert_eq!(
            aligned_replacement(
                "code\n   \ncode",
                "code\n   \ncode",
                &["  ", "  ", "  "],
                "\n"
            ),
            "  code\n   \n  code",
            "pure-whitespace lines stay untouched"
        );
        assert_eq!(
            aligned_replacement("a\nb\nc\nd", "a", &["\t"], "\r\n"),
            "\ta\r\n\tb\r\n\tc\r\n\td",
            "lines join with the file's own ending, new lines inherit and step"
        );
    }

    #[test]
    fn the_window_eol_reads_the_interior_breaks() {
        let lf = "one\n    two\nthree\n";
        let crlf = "one\r\n    two\r\nthree\r\n";
        assert_eq!(window_eol(lf, window(0, 3)), "\n");
        assert_eq!(window_eol(crlf, window(0, 3)), "\r\n");
    }

    #[test]
    fn the_replacement_loses_its_own_edge_whitespace() {
        assert_eq!(
            edge_trimmed_replacement("      fn two() { todo!() }"),
            "fn two() { todo!() }",
            "the model's own indent never doubles the file's"
        );
        assert_eq!(
            edge_trimmed_replacement("  a\n  b  "),
            "a\n  b",
            "interior lines stay verbatim, both edges trimmed"
        );
    }

    #[test]
    fn the_window_range_spans_the_files_actual_bytes() {
        let content = "fn one() {}\r\n    fn two() {}\r\nfn three() {}\r\n";
        let range = window_range(content, window(1, 2));
        assert_eq!(
            content.get(range.clone()),
            Some("    fn two() {}"),
            "the splice works on the file's own bytes: {range:?}"
        );
    }

    #[test]
    fn the_core_range_preserves_the_file_s_edge_whitespace() {
        let content = "fn one() {}\r\n    fn two() {}  \r\nfn three() {}\r\n";
        let range = window_core_range(content, window(1, 2));
        assert_eq!(
            content.get(range.clone()),
            Some("fn two() {}"),
            "the file's indent and trailing spaces stay outside the splice: {range:?}"
        );
    }

    #[test]
    fn the_echo_carries_line_numbers_and_context() {
        let content = "a\nb\nc\nd\ne\n";
        let rendered = render_echo(content, window(2, 3));
        assert!(rendered.contains("2 | b"), "one line before: {rendered}");
        assert!(rendered.contains("3 | c"), "the window itself");
        assert!(rendered.contains("4 | d"), "one line after");
    }

    #[test]
    fn the_echo_is_bounded_on_huge_files() {
        let content: String = "filler line\n".repeat(200);
        let rendered = render_echo(&content, window(50, 51));
        let line_count = rendered.lines().count();
        assert!(
            line_count <= ECHO_MAX_LINES.saturating_add(1),
            "the echo stays small on a huge file: {line_count} lines"
        );
    }

    #[test]
    fn an_empty_file_echoes_its_emptiness() {
        assert_eq!(render_echo("", window(0, 0)), "(the file is empty)");
    }
}
