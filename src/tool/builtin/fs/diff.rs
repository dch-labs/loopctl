//! Plain-text diff renderer for file-write/edit success messages.

use std::fmt::Write;

/// Max old×new line-count product before the LCS diff falls back to a preview.
///
/// The DP table is that product of cells, so past this bound the full
/// diff would dominate the write's own output; the preview keeps
/// the message proportional to the change instead.
const MAX_LCS_PRODUCT: usize = 1_000_000;

/// Lines shown per side in the large-file fallback preview (past `MAX_LCS_PRODUCT`).
///
/// Large enough that a reviewer sees the shape of each side, small
/// enough that even a giant rename renders as two bounded blocks.
const LARGE_DIFF_PREVIEW_LINES: usize = 1000;

/// One line in an LCS diff, produced by `compute_lcs_diff`.
///
/// Each variant carries the line's raw text content (no prefix). The diff
/// algorithm walks the old and new line lists and classifies each line:
/// present in both → [`Unchanged`](Self::Unchanged), only in old →
/// [`Deleted`](Self::Deleted), only in new → [`Inserted`](Self::Inserted).
/// The rendered `+`/`-` prefixes are added later by
/// `format_diff_with_context`, not stored here.
#[derive(Debug)]
enum LineDiff {
    /// A line present in both old and new content.
    ///
    /// The payload is the line's text without a prefix character. At render
    /// time, `format_diff_with_context` emits it with a leading two-space
    /// indent (`  `) to mark it as unchanged context surrounding a change
    /// region.
    Unchanged(String),

    /// A line only in old content — removed by the change.
    ///
    /// The payload is the raw line text without prefix. At render time,
    /// `format_diff_with_context` emits it with a `- ` prefix so the reader
    /// can see what was taken away.
    Deleted(String),

    /// A line only in new content — added by the change.
    ///
    /// The payload is the raw line text without prefix. At render time,
    /// `format_diff_with_context` emits it with a `+ ` prefix so the reader
    /// can see what was introduced.
    Inserted(String),
}

/// What a success message renders against: the previous state of the
/// written target.
///
/// Composed by the writing tools when they report a completed write.
/// The variants exist so that a file with no prior entry, a file with
/// readable prior text, and a file that existed but did not decode
/// each render distinctly — conflating the last with either of the
/// others would mislabel an overwrite as a creation or silently drop
/// the fact that the replaced bytes were never readable text.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OldContent<'a> {
    /// The target had no existing filesystem entry.
    ///
    /// The message renders the `Created:` block: the new content's
    /// first lines as an addition preview.
    Absent,

    /// The target's previous content, decoded as text.
    ///
    /// The message renders the `Changed:` block: a line diff (or the
    /// large-file preview) computed against this text.
    Text(&'a str),

    /// The target existed, but its bytes were not valid UTF-8.
    ///
    /// Nothing can be diffed line-wise against undecodable content,
    /// so the message renders the `Changed:` header labeled as
    /// modified with the previous content not UTF-8, and no diff
    /// body. The write itself is not a creation and must not be
    /// reported as one.
    NotUtf8,
}

/// Format a file change for the tool's success message.
///
/// The output differs by what [`OldContent`] says the target held:
///
/// - **New file** ([`OldContent::Absent`]): prints a `Created:` header
///   followed by the first 10 lines, each prefixed with `+ `. If the file
///   exceeds 10 lines, a `... +N more lines` summary is appended.
/// - **Modified, readable text** ([`OldContent::Text`]): prints a
///   `Changed:` header followed by an LCS-based line diff with 3 lines
///   of context around each change region. Unchanged context lines are
///   prefixed with two spaces; inserted lines with `+ `; deleted lines
///   with `- `.
/// - **Modified, undecodable bytes** ([`OldContent::NotUtf8`]): prints
///   the `Changed:` header labeled `previous content not UTF-8` with no
///   diff body — there is no text to diff against, and an overwrite
///   must not be reported as a creation.
///
/// The diff format is plain text (no ANSI color codes) so it renders cleanly
/// in any consumer.
#[must_use]
pub(crate) fn format_file_change(
    file_path: &str,
    old_content: OldContent<'_>,
    new_content: &str,
) -> String {
    match old_content {
        OldContent::Absent => {
            let lines: Vec<&str> = new_content.lines().collect();
            let preview_count = lines.len().min(10);
            let mut result = format!("Created: {file_path} (new file)\n");
            for line in lines.iter().take(preview_count) {
                writeln!(result, "+ {line}").ok();
            }
            if lines.len() > preview_count {
                let more = lines.len().saturating_sub(preview_count);
                writeln!(result, "... +{more} more lines").ok();
            }
            result
        }
        OldContent::NotUtf8 => {
            format!("Changed: {file_path} (modified, previous content not UTF-8)\n")
        }
        OldContent::Text(old) => {
            let old_lines: Vec<&str> = old.lines().collect();
            let new_lines: Vec<&str> = new_content.lines().collect();

            if old_lines.len().saturating_mul(new_lines.len()) > MAX_LCS_PRODUCT {
                return format_large_diff(file_path, &old_lines, &new_lines);
            }

            let diff = compute_lcs_diff(&old_lines, &new_lines);
            let mut result = format!("Changed: {file_path} (modified)\n");
            if let Some(summary) = change_summary(&diff) {
                result.push_str(&summary);
            }
            if let Some(note) = eof_change_note(old, new_content) {
                result.push_str(&note);
            }
            result.push_str(&format_diff_with_context(&diff, 3));
            result
        }
    }
}

/// Build a one-line `N lines removed, M added` summary, prefixed with the
/// diff gutter character, for a line diff.
///
/// Returns `None` when the diff has no removed and no inserted lines (a
/// no-op edit), so an unchanged file renders header-only with no spurious
/// summary. Counts are pluralized: `1 line` vs `2 lines`.
fn change_summary(diff: &[LineDiff]) -> Option<String> {
    let removed = diff
        .iter()
        .filter(|d| matches!(d, LineDiff::Deleted(_)))
        .count();
    let added = diff
        .iter()
        .filter(|d| matches!(d, LineDiff::Inserted(_)))
        .count();
    if removed == 0 && added == 0 {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if removed > 0 {
        parts.push(format!("{removed} line{} removed", plural(removed)));
    }
    if added > 0 {
        parts.push(format!("{added} line{} added", plural(added)));
    }
    Some(format!("│ {}\n", parts.join(", ")))
}

/// Pluralization suffix for a count: empty string for one, `"s"` otherwise.
///
/// Concatenated after `line` in the change summary so counts read
/// naturally in both numbers.
fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

/// Describe a trailing-newline-at-EOF change, when that is the *only* change.
///
/// `str::lines()` drops a file's final `\n`, so two contents that differ only
/// in their trailing newline (e.g. `"a\n"` → `"a"`) produce identical line
/// vectors and an empty LCS diff. Without this note such a write would render
/// as a bare header with no indication anything changed.
///
/// Returns `None` when the line content actually differs (the LCS already
/// covers it) or when the trailing-newline state is unchanged.
fn eof_change_note(old: &str, new: &str) -> Option<String> {
    if old.lines().ne(new.lines()) {
        return None;
    }
    let had = old.ends_with('\n');
    let has = new.ends_with('\n');
    let note = match (had, has) {
        (true, false) => "No newline at end of file",
        (false, true) => "Newline added at end of file",
        _ => return None,
    };
    Some(format!("│ {note}\n"))
}

/// Format a diff for files too large for the LCS algorithm (product exceeds
/// `MAX_LCS_PRODUCT`).
///
/// Shows a truncated before/after preview instead of computing the full
/// diff: up to `LARGE_DIFF_PREVIEW_LINES` lines of old content (prefixed
/// `- `) and the same number of new content (prefixed `+ `), with
/// `... N more lines` summaries for each side.
fn format_large_diff(file_path: &str, old_lines: &[&str], new_lines: &[&str]) -> String {
    let mut result = format!("Changed: {file_path} (modified, large file)\n");
    let old_preview = old_lines.len().min(LARGE_DIFF_PREVIEW_LINES);

    result.push_str("Before:\n");

    for line in old_lines.iter().take(old_preview) {
        writeln!(result, "- {line}").ok();
    }

    if old_lines.len() > old_preview {
        let more = old_lines.len().saturating_sub(old_preview);
        writeln!(result, "... {more} more lines").ok();
    }

    let new_preview = new_lines.len().min(LARGE_DIFF_PREVIEW_LINES);
    result.push_str("After:\n");

    for line in new_lines.iter().take(new_preview) {
        writeln!(result, "+ {line}").ok();
    }

    if new_lines.len() > new_preview {
        let more = new_lines.len().saturating_sub(new_preview);
        writeln!(result, "... {more} more lines").ok();
    }

    result
}

/// Read a cell from the flat DP table at row `i`, column `j`.
///
/// The table is stored as a single `Vec<usize>` with row-major layout: cell
/// `(i, j)` lives at index `i * stride + j`, where `stride` is the number of
/// columns. Returns `0` if the computed index is out of bounds (which cannot
/// happen for indices derived from valid line counts, but the safe `.get()`
/// guards against arithmetic edge cases).
fn dp_get(dp: &[usize], stride: usize, i: usize, j: usize) -> usize {
    dp.get(i.saturating_mul(stride).saturating_add(j))
        .copied()
        .unwrap_or(0)
}

/// Write a cell in the flat DP table at row `i`, column `j`.
///
/// Counterpart to `dp_get`. If the computed index is out of bounds the write
/// is silently skipped — again, this cannot happen for indices derived from
/// valid line counts.
fn dp_set(dp: &mut [usize], stride: usize, i: usize, j: usize, val: usize) {
    if let Some(slot) = dp.get_mut(i.saturating_mul(stride).saturating_add(j)) {
        *slot = val;
    }
}

/// Compute an LCS-based diff between two line lists.
///
/// Uses the classic dynamic-programming longest-common-subsequence
/// algorithm to classify each line as unchanged, deleted, or inserted. The
/// result is in document order (top to bottom).
///
/// Empty inputs are fast-pathed: if `old_lines` is empty, every new line is
/// [`Inserted`](LineDiff::Inserted); if `new_lines` is empty, every old line
/// is [`Deleted`](LineDiff::Deleted).
fn compute_lcs_diff(old_lines: &[&str], new_lines: &[&str]) -> Vec<LineDiff> {
    let m = old_lines.len();
    let n = new_lines.len();
    if m == 0 {
        return new_lines
            .iter()
            .map(|l| LineDiff::Inserted(l.to_string()))
            .collect();
    }
    if n == 0 {
        return old_lines
            .iter()
            .map(|l| LineDiff::Deleted(l.to_string()))
            .collect();
    }

    let stride = n.saturating_add(1);
    let dp = build_lcs_dp(old_lines, new_lines, stride);
    let mut result = backtrack_lcs(&dp, stride, old_lines, new_lines);
    result.reverse();
    result
}

/// Fill the Longest Common Subsequence dynamic-programming table.
///
/// Returns a flat `Vec<usize>` where cell `(i, j)` is at index
/// `i * stride + j`, holding the LCS length of `old_lines[..i]` and
/// `new_lines[..j]`.
fn build_lcs_dp(old_lines: &[&str], new_lines: &[&str], stride: usize) -> Vec<usize> {
    let dp_len = old_lines.len().saturating_add(1).saturating_mul(stride);
    let mut dp = vec![0usize; dp_len];
    for (i, old_line) in old_lines.iter().enumerate() {
        let next_i = i.saturating_add(1);
        for (j, new_line) in new_lines.iter().enumerate() {
            let next_j = j.saturating_add(1);
            let val = if old_line == new_line {
                dp_get(&dp, stride, i, j).saturating_add(1)
            } else {
                dp_get(&dp, stride, i, next_j).max(dp_get(&dp, stride, next_i, j))
            };
            dp_set(&mut dp, stride, next_i, next_j, val);
        }
    }
    dp
}

/// Walk the DP table backwards from `(m, n)` to reconstruct the diff.
///
/// Starting at the bottom-right cell, moves up and left one step at a time.
/// At each position `(i, j)`:
///
/// - If `old_lines[i-1] == new_lines[j-1]`, the line is unchanged: emit
///   [`Unchanged`](LineDiff::Unchanged) and move diagonally to `(i-1, j-1)`.
/// - Otherwise, compare the two neighbours — `dp[i][j-1]` (skip new line) vs
///   `dp[i-1][j]` (skip old line) — and follow the higher value:
///   - Skip a new line → emit [`Inserted`](LineDiff::Inserted), move left.
///   - Skip an old line → emit [`Deleted`](LineDiff::Deleted), move up.
///
/// The loop terminates when both `i` and `j` reach zero. Because the walk
/// runs from the end, lines are collected in reverse document order; the
/// caller reverses before use.
fn backtrack_lcs(
    dp: &[usize],
    stride: usize,
    old_lines: &[&str],
    new_lines: &[&str],
) -> Vec<LineDiff> {
    let mut result = Vec::new();
    let mut i = old_lines.len();
    let mut j = new_lines.len();

    while i > 0 || j > 0 {
        let old_line = i.checked_sub(1).and_then(|idx| old_lines.get(idx));
        let new_line = j.checked_sub(1).and_then(|idx| new_lines.get(idx));

        if let (Some(old), Some(new)) = (old_line, new_line)
            && old == new
        {
            result.push(LineDiff::Unchanged(old.to_string()));
            i = i.saturating_sub(1);
            j = j.saturating_sub(1);
            continue;
        }

        let is_inserted = is_new_line_inserted(dp, stride, i, j);
        if is_inserted {
            if let Some(line) = new_line {
                result.push(LineDiff::Inserted(line.to_string()));
            }
            j = j.saturating_sub(1);
        } else {
            if let Some(line) = old_line {
                result.push(LineDiff::Deleted(line.to_string()));
            }
            i = i.saturating_sub(1);
        }
    }
    result
}

/// Determine whether the new line at position `j-1` is an insertion.
///
/// Returns `true` if the LCS value to the left `dp[i][j-1]` is >= the value
/// above `dp[i-1][j]`, meaning the new line has no match in old.
fn is_new_line_inserted(dp: &[usize], stride: usize, i: usize, j: usize) -> bool {
    if j == 0 {
        return false;
    }
    if i == 0 {
        return true;
    }
    let left = dp_get(dp, stride, i, j.saturating_sub(1));
    let up = dp_get(dp, stride, i.saturating_sub(1), j);
    left >= up
}

/// Format a diff with `context` lines of surrounding context around each
/// change region.
///
/// Walks the [`LineDiff`] sequence and renders it as plain text: unchanged
/// context lines are prefixed with two spaces (`  `); inserted lines with
/// `+ `; deleted lines with `- `. To keep the output readable for large
/// changes, only `context` unchanged lines are shown before and after each
/// run of insertions/deletions. Runs of unchanged lines longer than
/// `context` are elided — including at end of input, where only an open
/// change region's pending context is flushed (its trailing context), so
/// unchanged tail lines distant from the last change never render as if
/// contiguous with it.
fn format_diff_with_context(line_diff: &[LineDiff], context: usize) -> String {
    let mut result = String::new();
    let mut pending_context: Vec<String> = Vec::new();
    let mut in_change = false;
    let mut context_after = 0usize;

    for diff in line_diff {
        match diff {
            LineDiff::Unchanged(line) => {
                if in_change {
                    pending_context.push(line.clone());
                    context_after = context_after.saturating_add(1);
                    if context_after >= context {
                        for ctx_line in &pending_context {
                            writeln!(result, "  {ctx_line}").ok();
                        }
                        pending_context.clear();
                        in_change = false;
                        context_after = 0;
                    }
                } else {
                    pending_context.push(line.clone());
                    if pending_context.len() > context {
                        pending_context.remove(0);
                    }
                }
            }
            LineDiff::Deleted(line) => {
                for ctx_line in &pending_context {
                    writeln!(result, "  {ctx_line}").ok();
                }
                pending_context.clear();
                context_after = 0;
                writeln!(result, "- {line}").ok();
                in_change = true;
            }
            LineDiff::Inserted(line) => {
                for ctx_line in &pending_context {
                    writeln!(result, "  {ctx_line}").ok();
                }
                pending_context.clear();
                context_after = 0;
                writeln!(result, "+ {line}").ok();
                in_change = true;
            }
        }
    }
    if in_change {
        for ctx_line in &pending_context {
            writeln!(result, "  {ctx_line}").ok();
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::missing_panics_doc, clippy::format_collect)]
mod tests {
    use super::*;

    #[test]
    fn new_file_preview() {
        let content = "line 1\nline 2\nline 3\n";
        let result = format_file_change("test.rs", OldContent::Absent, content);
        assert!(result.contains("Created: test.rs (new file)"));
        assert!(result.contains("+ line 1"));
        assert!(result.contains("+ line 2"));
        assert!(result.contains("+ line 3"));
    }

    #[test]
    fn new_file_truncates_long_preview() {
        let content: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        let result = format_file_change("test.rs", OldContent::Absent, &content);
        assert!(result.contains("... +10 more lines"));
        assert!(!result.contains("+ line 11"));
    }

    #[test]
    fn edit_shows_inserted_lines() {
        let old = "a\nb\nc\n";
        let new = "a\nb\nNEW\nc\n";
        let result = format_file_change("test.rs", OldContent::Text(old), new);
        assert!(result.contains("Changed: test.rs (modified)"));
        assert!(result.contains("+ NEW"));
        assert!(result.contains("  a"));
        assert!(result.contains("  b"));
    }

    #[test]
    fn edit_shows_deleted_lines() {
        let old = "a\nOLD\nb\nc\n";
        let new = "a\nb\nc\n";
        let result = format_file_change("test.rs", OldContent::Text(old), new);
        assert!(result.contains("- OLD"));
    }

    #[test]
    fn edit_context_limit() {
        let old: String = (1..=10)
            .map(|i| format!("keep {i}\n"))
            .chain(std::iter::once("OLD\n".to_string()))
            .chain((1..=10).map(|i| format!("tail {i}\n")))
            .collect();
        let new: String = (1..=10)
            .map(|i| format!("keep {i}\n"))
            .chain(std::iter::once("NEW\n".to_string()))
            .chain((1..=10).map(|i| format!("tail {i}\n")))
            .collect();
        let result = format_file_change("test.rs", OldContent::Text(&old), &new);
        assert!(result.contains("- OLD"));
        assert!(result.contains("+ NEW"));
        assert!(
            result.contains("keep 8"),
            "context before the change is the last 3 lines: {result}"
        );
        assert!(result.contains("keep 10"), "{result}");
        assert!(
            !result.contains("keep 7"),
            "elided middle must not render: {result}"
        );
        assert!(
            result.contains("tail 1"),
            "context after the change is the first 3 lines: {result}"
        );
        assert!(result.contains("tail 3"), "{result}");
        assert!(!result.contains("tail 4"), "{result}");
        assert!(
            !result.contains("tail 10"),
            "distant trailing lines after a closed region must not render: {result}"
        );
    }

    #[test]
    fn non_utf8_previous_content_renders_a_labeled_header_only() {
        let result = format_file_change("f.bin", OldContent::NotUtf8, "new\n");
        assert_eq!(
            result, "Changed: f.bin (modified, previous content not UTF-8)\n",
            "no summary, no diff body — there is nothing to diff against"
        );
    }

    #[test]
    fn large_files_fall_back_to_a_preview() {
        let old: String = (1..=2000).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=2000)
            .map(|i| format!("line {i}\n"))
            .chain(std::iter::once("APPENDED\n".to_string()))
            .collect();
        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        assert!(
            result.contains("Changed: big.rs (modified, large file)"),
            "{result}"
        );
        assert!(result.contains("Before:"));
        assert!(result.contains("After:"));
        assert!(result.contains("more lines"));
    }

    #[test]
    fn moderate_files_still_get_a_real_diff() {
        let old: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=99)
            .map(|i| format!("line {i}\n"))
            .chain(std::iter::once("CHANGED\n".to_string()))
            .collect();
        let result = format_file_change("small.rs", OldContent::Text(&old), &new);
        assert!(result.contains("Changed: small.rs (modified)"));
        assert!(!result.contains("large file"));
        assert!(result.contains("- line 100"));
        assert!(result.contains("+ CHANGED"));
    }

    #[test]
    fn summary_line_present_for_mixed_edit() {
        let old = "a\nb\nc\n";
        let new = "a\nX\nY\nc\n";
        let result = format_file_change("f.rs", OldContent::Text(old), new);
        assert!(result.contains("1 line removed"), "{result}");
        assert!(result.contains("2 lines added"), "{result}");
        assert!(result.contains("- b"));
        assert!(result.contains("+ X"));
    }

    #[test]
    fn summary_line_absent_for_noop_edit() {
        let content = "a\nb\nc\n";
        let result = format_file_change("f.rs", OldContent::Text(content), content);
        assert!(result.starts_with("Changed: f.rs (modified)\n"));
        assert!(!result.contains("line removed"));
        assert!(!result.contains("line added"));
        assert!(!result.contains("\n+ "));
        assert!(!result.contains("\n- "));
    }

    #[test]
    fn summary_line_pure_deletion() {
        let old = "a\nb\nc\n";
        let new = "a\nc\n";
        let result = format_file_change("f.rs", OldContent::Text(old), new);
        assert!(result.contains("1 line removed"), "{result}");
        assert!(!result.contains("added"));
    }

    #[test]
    fn summary_line_empty_old_content_render_without_panic() {
        let result = format_file_change("f.rs", OldContent::Text(""), "x\n");
        assert!(result.contains("Changed: f.rs (modified)"));
        assert!(result.contains("1 line added"));
        assert!(result.contains("+ x"));
    }

    #[test]
    fn trailing_newline_removed_is_visible() {
        let result = format_file_change("f.txt", OldContent::Text("a\n"), "a");
        assert!(result.contains("Changed: f.txt (modified)"));
        assert!(result.contains("No newline at end of file"), "{result}");
        assert!(!result.contains("\n+ "));
        assert!(!result.contains("\n- "));
    }

    #[test]
    fn trailing_newline_added_is_visible() {
        let result = format_file_change("f.txt", OldContent::Text("a"), "a\n");
        assert!(result.contains("Newline added at end of file"), "{result}");
    }

    #[test]
    fn unchanged_trailing_newline_emits_no_eof_note() {
        let result = format_file_change("f.txt", OldContent::Text("a\n"), "a\n");
        assert!(!result.contains("end of file"), "{result}");
    }

    #[test]
    fn real_line_change_does_not_trigger_eof_note() {
        let result = format_file_change("f.txt", OldContent::Text("a\n"), "b\n");
        assert!(result.contains("- a"));
        assert!(result.contains("+ b"));
        assert!(!result.contains("end of file"), "{result}");
    }
}
