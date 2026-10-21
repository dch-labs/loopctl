//! Plain-text diff renderer for file-write/edit success messages.
//!
//! The renderer addresses every line it prints by its position in the file,
//! and its cost is bounded by the *change* rather than by the file: the
//! unchanged bulk surrounding a change is measured by linear scans and
//! rendered as counted elisions, never carried through the quadratic
//! matching core. A receipt therefore stays hunk-sized however long the file
//! it describes is.
//!
//! The shape it emits — a `Changed:`/`Created:` header, a `│`-prefixed
//! change summary, and `@@ -old,count +new,count @@` hunks with three lines
//! of context — is a read contract for any consumer that renders these
//! receipts as diffs, so the hunked form is what every diffable change must
//! reach, and the hunkless before/after preview survives only where no
//! diffable region could be bounded at all.

use std::collections::HashMap;
use std::fmt::Write;

/// Max old×new line-count product one change region may carry into the LCS
/// matcher, past which the region is split before it is matched.
///
/// The DP table is that product of cells, so the bound is what keeps a
/// single region's matching proportional to the change: it caps the
/// quadratic core, not the file it describes. Regions the bound cannot
/// cover are aligned on lines unique to each side and matched in parts;
/// only a region with nothing to align on falls back to the preview.
const MAX_LCS_PRODUCT: usize = 1_000_000;

/// Unchanged lines printed on each side of a change region.
///
/// The width a reader needs to see where a change sits: three lines before
/// and after, and the same width of elision margin — an unchanged run
/// longer than twice this is skipped rather than printed, its line count
/// carried into the next hunk header instead.
const CONTEXT_LINES: usize = 3;

/// Lines a bounded receipt block may print per side.
///
/// Two blocks are bounded by this: the large-file fallback preview (past
/// [`MAX_LCS_PRODUCT`] with nothing to align on), which prints this many
/// lines of each side, and a new file's additions, which print this many
/// `+ ` lines before the `... +N more lines` tail. The bound keeps a
/// receipt proportional to what was written even when the answer is "this
/// whole file", where no smaller honest form exists.
const LARGE_DIFF_PREVIEW_LINES: usize = 1000;

/// One line in a diff sequence, produced by `build_diff_lines`.
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

    /// A run of unchanged lines the sequence leaves out, sized but not spelled.
    ///
    /// An unchanged run wider than the context margin is not printed — but
    /// the lines it holds still count, because the `@@` headers address
    /// lines by position. This entry carries the count on each side (always
    /// equal for unchanged content, stored twice so the counts stay honest
    /// however the sequence was assembled) and is never rendered itself: it
    /// ends the hunk it follows and advances both line counters, which is
    /// what lets a change at line 9 000 of a large file print its real
    /// numbers without printing 9 000 lines to reach it.
    Skipped {
        /// Old-side lines the entry stands in for.
        ///
        /// Counted into every old-side line number that follows it, so a
        /// hunk printed past the elision still addresses the line it names.
        old: usize,

        /// New-side lines the entry stands in for.
        ///
        /// Equal to [`Skipped::old`](Self::old) for the unchanged runs this
        /// renderer builds; carried separately so no caller can quietly make
        /// the two sides agree by accident.
        new: usize,
    },
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

    /// The target existed, but its content was not read.
    ///
    /// The pre-read's guards classified the target without buffering
    /// its bytes — the size exceeds the family read cap, or the
    /// target is not a regular file — so nothing can be diffed
    /// line-wise. The message renders the `Changed:` header labeled
    /// as modified with the previous content not read, and no diff
    /// body: the write replaced prior bytes it never buffered, which
    /// is neither a creation nor a diffable modification.
    Unreadable,
}

/// Format a file change for the tool's success message.
///
/// The output differs by what [`OldContent`] says the target held:
///
/// - **New file** ([`OldContent::Absent`]): prints a `Created:` header
///   followed by the file's additions, each prefixed with `+ `, up to
///   [`LARGE_DIFF_PREVIEW_LINES`] lines. Past that a `... +N more lines`
///   summary closes the block, so a generated file stays bounded instead of
///   echoing itself into the reader's context.
/// - **Modified, readable text** ([`OldContent::Text`]): prints a
///   `Changed:` header followed by a line diff with [`CONTEXT_LINES`] lines
///   of context around each change region. Unchanged context lines are
///   prefixed with two spaces; inserted lines with `+ `; deleted lines
///   with `- `. Every region opens with an `@@` header carrying its real
///   line numbers whatever the file's length, so an edit to a
///   ten-thousand-line file prints its change and its margin, not its file.
/// - **Modified, undecodable bytes** ([`OldContent::NotUtf8`]): prints
///   the `Changed:` header labeled `previous content not UTF-8` with no
///   diff body — there is no text to diff against, and an overwrite
///   must not be reported as a creation.
/// - **Modified, unread content** ([`OldContent::Unreadable`]): prints
///   the `Changed:` header labeled `previous content not read` with no
///   diff body — the pre-read's guards classified the target without
///   buffering its bytes, and the overwrite still is not a creation.
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
            let preview_count = lines.len().min(LARGE_DIFF_PREVIEW_LINES);
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
        OldContent::Unreadable => {
            format!("Changed: {file_path} (modified, previous content not read)\n")
        }
        OldContent::Text(old) => {
            let old_lines: Vec<&str> = old.lines().collect();
            let new_lines: Vec<&str> = new_content.lines().collect();
            let Some(diff) = build_diff_lines(&old_lines, &new_lines) else {
                return format_large_diff(file_path, &old_lines, &new_lines);
            };

            let mut result = format!("Changed: {file_path} (modified)\n");
            if let Some(summary) = change_summary(&diff) {
                result.push_str(&summary);
            }
            if let Some(note) = eof_change_note(old, new_content) {
                result.push_str(&note);
            }
            result.push_str(&format_diff_with_context(&diff, CONTEXT_LINES));
            result
        }
    }
}

/// A change region located in the file: the changed lines and where they sit.
///
/// Produced by [`split_segment`] and stitched into the diff sequence by
/// [`build_diff_lines`]. The ranges are absolute, half-open line positions in
/// the old and new content and hold only the region's own changes — context
/// is not stored here but drawn from the file around the range when the
/// sequence is assembled, which is what keeps the stored work proportional to
/// the change rather than to the file.
struct ChangeWindow {
    /// Half-open line range this region replaces in the old content.
    ///
    /// Absolute in the old file, and empty for a pure insertion — the pair of
    /// ranges is what tells the stitcher how wide the unchanged stretch
    /// around the region is, without carrying that stretch anywhere.
    old: std::ops::Range<usize>,

    /// Half-open line range this region writes into the new content.
    ///
    /// Absolute in the new file, and empty for a pure deletion. It usually
    /// differs in length from [`ChangeWindow::old`], and by exactly the
    /// region's net change — the stitcher's arithmetic depends on that.
    new: std::ops::Range<usize>,

    /// The region's own diff: removed and inserted lines, in order.
    ///
    /// Never holds [`LineDiff::Unchanged`] entries: context belongs to the
    /// gaps between regions, which is where it can be shared, elided, and
    /// counted without being materialised once per region.
    entries: Vec<LineDiff>,
}

/// A stretch of the two files still to be aligned, in absolute line indices.
///
/// The unit of [`split_segment`]'s worklist: a segment is either resolved
/// into a [`ChangeWindow`] outright or cut at anchor lines into smaller
/// segments, so a file's whole diff is driven by an explicit queue rather than
/// by call depth — a relocated block in a huge file cannot overflow the stack
/// by being one recursion level deep per anchor.
struct Segment {
    /// Lines still to be matched on the old side, absolute in that file.
    ///
    /// Shrink-only: every part derived from a segment excludes the anchor
    /// lines that split it, which is what makes the worklist terminate.
    old: std::ops::Range<usize>,

    /// Lines still to be matched on the new side, absolute in that file.
    ///
    /// Paired with [`Segment::old`] by the trimming and anchoring above it, so
    /// the two ranges describe the same stretch of content on each side.
    new: std::ops::Range<usize>,
}

/// Build the diff sequence between two files' lines at a cost the change sets.
///
/// The unchanged bulk around a change is the expensive part of diffing a
/// large file and the least interesting part of reading one, so it is
/// measured rather than matched: [`common_edges`] trims what the two sides
/// share at their heads and tails by scanning, [`split_segment`] cuts anything
/// still too wide to match at lines unique to each side, and the unchanged
/// runs between the resulting regions are rendered by [`push_gap`] as context
/// lines plus a counted [`LineDiff::Skipped`]. What comes out is the sequence
/// a whole-file matcher would have produced — unchanged, deleted, and inserted
/// lines in document order, with counted elisions standing in for runs wider
/// than the context margin — so `format_diff_with_context` numbers every hunk
/// from real file positions.
///
/// Returns `None` when a region too wide to match has no line unique to each
/// side to align on: a wholesale rewrite, for which the caller renders the
/// bounded before/after preview rather than a diff that would either cost the
/// file's square or silently drop the lines it skipped.
fn build_diff_lines(old_lines: &[&str], new_lines: &[&str]) -> Option<Vec<LineDiff>> {
    let mut windows: Vec<ChangeWindow> = Vec::new();
    let mut work: Vec<Segment> = vec![Segment {
        old: 0..old_lines.len(),
        new: 0..new_lines.len(),
    }];
    // The worklist is a stack and `split_segment` pushes the parts it derives
    // in reverse, so regions come off it in document order.
    while let Some(segment) = work.pop() {
        split_segment(old_lines, new_lines, &segment, &mut windows, &mut work)?;
    }
    if windows.is_empty() {
        return Some(Vec::new());
    }

    let mut sequence: Vec<LineDiff> = Vec::new();
    let mut old_at = 0_usize;
    let mut new_at = 0_usize;
    for window in windows {
        push_gap(
            &mut sequence,
            old_lines,
            old_at,
            window.old.start,
            new_at,
            window.new.start,
        );
        old_at = window.old.end;
        new_at = window.new.end;
        sequence.extend(window.entries);
    }
    push_gap(
        &mut sequence,
        old_lines,
        old_at,
        old_lines.len(),
        new_at,
        new_lines.len(),
    );
    Some(sequence)
}

/// Resolve one segment into a change region, or cut it into smaller ones.
///
/// Three outcomes, tried in order:
///
/// 1. The segment's differing middle fits inside [`MAX_LCS_PRODUCT`], so
///    [`compute_lcs_diff`] matches it there and the middle becomes a
///    [`ChangeWindow`]. A middle empty on one side is a pure insertion or
///    deletion and needs no matching at all.
/// 2. The middle is wider, but [`unique_anchors`] finds lines occurring once
///    on each side: the segment is cut at them and the stretches between
///    consecutive anchors go back on the worklist, each one strictly smaller
///    than the segment they came from.
/// 3. The middle is wider and has nothing to align on — a rewrite with no
///    surviving line. `None` sends the caller to the preview: no bound can be
///    honoured, and eliding the middle would print numbers about lines the
///    diff never saw.
fn split_segment(
    old_lines: &[&str],
    new_lines: &[&str],
    segment: &Segment,
    windows: &mut Vec<ChangeWindow>,
    work: &mut Vec<Segment>,
) -> Option<()> {
    let old_mid = old_lines.get(segment.old.clone())?;
    let new_mid = new_lines.get(segment.new.clone())?;
    let (prefix, suffix) = common_edges(old_mid, new_mid);
    let old_start = segment.old.start.saturating_add(prefix);
    let old_end = segment.old.end.saturating_sub(suffix);
    let new_start = segment.new.start.saturating_add(prefix);
    let new_end = segment.new.end.saturating_sub(suffix);
    let old_count = old_end.saturating_sub(old_start);
    let new_count = new_end.saturating_sub(new_start);

    if old_count == 0 && new_count == 0 {
        return Some(());
    }

    let addressable = old_count == 0 || new_count == 0;
    if addressable || old_count.saturating_mul(new_count) <= MAX_LCS_PRODUCT {
        let entries = compute_lcs_diff(
            old_lines.get(old_start..old_end).unwrap_or_default(),
            new_lines.get(new_start..new_end).unwrap_or_default(),
        );
        if !entries.is_empty() {
            windows.push(ChangeWindow {
                old: old_start..old_end,
                new: new_start..new_end,
                entries,
            });
        }
        return Some(());
    }

    let anchors: Vec<(usize, usize)> = unique_anchors(
        old_lines.get(old_start..old_end).unwrap_or_default(),
        new_lines.get(new_start..new_end).unwrap_or_default(),
    )
    .into_iter()
    .map(|(old_at, new_at)| {
        (
            old_start.saturating_add(old_at),
            new_start.saturating_add(new_at),
        )
    })
    .collect();
    if anchors.is_empty() {
        return None;
    }

    let mut parts: Vec<Segment> = Vec::with_capacity(anchors.len().saturating_add(1));
    let mut cursor = (old_start, new_start);
    for &(anchor_old, anchor_new) in anchors.iter().chain(std::iter::once(&(old_end, new_end))) {
        if anchor_old > cursor.0 || anchor_new > cursor.1 {
            parts.push(Segment {
                old: cursor.0..anchor_old,
                new: cursor.1..anchor_new,
            });
        }
        cursor = (anchor_old.saturating_add(1), anchor_new.saturating_add(1));
    }
    work.extend(parts.into_iter().rev());
    Some(())
}

/// Count the lines the two sides share at their head and at their tail.
///
/// The cheap measurement that keeps the quadratic core off unchanged
/// material: a prefix scan and a suffix scan, the second bounded by what the
/// prefix left so the two runs cannot overlap. Everything between the counts
/// is what actually differs — for a one-line edit anywhere in a large file,
/// one line per side.
fn common_edges(old: &[&str], new: &[&str]) -> (usize, usize) {
    let prefix = old
        .iter()
        .zip(new.iter())
        .take_while(|(before, after)| before == after)
        .count();
    let room = old
        .len()
        .saturating_sub(prefix)
        .min(new.len().saturating_sub(prefix));
    let suffix = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take_while(|(before, after)| before == after)
        .take(room)
        .count();
    (prefix, suffix)
}

/// The line pairs a too-wide region can be cut at, in ascending file order.
///
/// A line occurring exactly once on each side of a region is no coincidence:
/// pairing its two occurrences is sound alignment however far apart they sit,
/// which is what lets a region too wide for [`MAX_LCS_PRODUCT`] be cut into
/// pieces small enough to match. Candidate pairs are the lines unique on both
/// sides, taken in old-side order; the longest strictly increasing run of
/// their new-side positions is returned, because an anchor sequence that
/// crossed itself would align nothing. Content that genuinely moved — a
/// relocated block, a file of repeated filler — yields few or no pairs, and
/// [`split_segment`] then reports the region unalignable rather than guessing.
fn unique_anchors(old: &[&str], new: &[&str]) -> Vec<(usize, usize)> {
    let old_once = single_occurrences(old);
    let new_once = single_occurrences(new);
    let pairs: Vec<(usize, usize)> = old
        .iter()
        .enumerate()
        .filter_map(|(old_at, line)| match old_once.get(line) {
            Some(Slot::Once(first)) if *first == old_at => {
                new_once.get(line).and_then(|slot| match slot {
                    Slot::Once(new_at) => Some((old_at, *new_at)),
                    Slot::Many => None,
                })
            }
            _ => None,
        })
        .collect();
    longest_anchor_run(&pairs)
}

/// How often a line occurs in a stretch: once, at which index, or more.
///
/// The histogram behind [`unique_anchors`], kept deliberately two-valued:
/// an anchor only needs to know that a line is singular and where it sits,
/// and a count would let a repeated line look almost unique.
enum Slot {
    /// The line occurred exactly once, at this index.
    ///
    /// The index is the pairing candidate — sound because there is no other
    /// occurrence on this side to confuse it with.
    Once(usize),

    /// The line occurred more than once, so it cannot anchor an alignment.
    ///
    /// Marked on the second sighting and never revisited: which copy a
    /// repeated line should pair with is not something the matcher can
    /// decide from line identity alone.
    Many,
}

/// Index every line that occurs exactly once in `lines`, by first occurrence.
///
/// Singleness is what makes an anchor trustworthy: a line repeated on either
/// side could be paired with the wrong copy of itself. Lines seen twice or
/// more are marked [`Slot::Many`] and never offered as a pair.
fn single_occurrences<'a>(lines: &[&'a str]) -> HashMap<&'a str, Slot> {
    let mut seen: HashMap<&str, Slot> = HashMap::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        match seen.entry(line) {
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(Slot::Once(index));
            }
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                occupied.insert(Slot::Many);
            }
        }
    }
    seen
}

/// The longest strictly increasing run of new-side anchor positions.
///
/// `pairs` arrives ordered by old-side position, so an anchor sequence is
/// valid exactly when its new-side positions rise. The classic patience tails
/// array finds one in `O(k log k)`: `tails[len]` holds the index of the
/// smallest new-side position that can end a run of `len + 1` anchors, and
/// each pair records its predecessor so the winning run can be walked back
/// out. Equal positions cannot occur — every pair's new-side index is unique
/// by construction — so the run is strict without special-casing ties.
fn longest_anchor_run(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut tails: Vec<usize> = Vec::new();
    let mut predecessors: Vec<Option<usize>> = vec![None; pairs.len()];

    for (index, pair) in pairs.iter().enumerate() {
        let position = tails
            .partition_point(|tail| pairs.get(*tail).is_some_and(|earlier| earlier.1 < pair.1));
        if position > 0 {
            let previous = tails.get(position.saturating_sub(1)).copied();
            if let Some(slot) = predecessors.get_mut(index)
                && let Some(previous) = previous
            {
                *slot = Some(previous);
            }
        }
        if position == tails.len() {
            tails.push(index);
        } else if let Some(slot) = tails.get_mut(position) {
            *slot = index;
        }
    }

    let mut run: Vec<(usize, usize)> = Vec::with_capacity(tails.len());
    let mut cursor = tails.last().copied();
    while let Some(index) = cursor {
        if let Some(pair) = pairs.get(index) {
            run.push(*pair);
        }
        cursor = predecessors.get(index).copied().flatten();
    }
    run.reverse();
    run
}

/// Append one shared line to the sequence, when the range still holds it.
///
/// The only writer of [`LineDiff::Unchanged`] in the stitcher, and the bounds
/// check is the whole reason it exists: a context margin at either end of a
/// large file is measured from a position that may not have [`CONTEXT_LINES`]
/// lines left before or after it.
fn push_shared(out: &mut Vec<LineDiff>, lines: &[&str], at: usize) {
    if let Some(line) = lines.get(at) {
        out.push(LineDiff::Unchanged((*line).to_string()));
    }
}

/// Render the unchanged stretch between two change regions as context.
///
/// A short run — no wider than the context margin on both sides together — is
/// printed whole, which is what keeps two nearby changes inside one hunk
/// rather than splitting them across two. A wider run prints the margin on
/// each side and a [`LineDiff::Skipped`] carrying the lines left out, so the
/// next hunk's header still addresses real file positions.
///
/// The printed text comes from the old side because the run between two
/// regions is shared content — trimming and anchoring only ever cut at lines
/// both sides hold — so the counts are taken from both ranges and the printed
/// lines from one. A caller whose ranges disagreed would print the margin it
/// measured on the old side and count the difference into the elision, keeping
/// the following numbers honest about lines it did not print.
fn push_gap(
    out: &mut Vec<LineDiff>,
    old_lines: &[&str],
    old_from: usize,
    old_to: usize,
    new_from: usize,
    new_to: usize,
) {
    let old_gap = old_to.saturating_sub(old_from);
    let new_gap = new_to.saturating_sub(new_from);
    if old_gap == 0 && new_gap == 0 {
        return;
    }
    let shared = old_gap.min(new_gap);
    let margin = CONTEXT_LINES.saturating_mul(2);

    if shared > margin {
        for offset in 0..CONTEXT_LINES {
            push_shared(out, old_lines, old_from.saturating_add(offset));
        }
        out.push(LineDiff::Skipped {
            old: shared.saturating_sub(margin),
            new: shared.saturating_sub(margin),
        });
        for offset in shared.saturating_sub(CONTEXT_LINES)..shared {
            push_shared(out, old_lines, old_from.saturating_add(offset));
        }
    } else {
        for offset in 0..shared {
            push_shared(out, old_lines, old_from.saturating_add(offset));
        }
    }

    let old_extra = old_gap.saturating_sub(shared);
    let new_extra = new_gap.saturating_sub(shared);
    if old_extra > 0 || new_extra > 0 {
        out.push(LineDiff::Skipped {
            old: old_extra,
            new: new_extra,
        });
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
/// Walks the [`LineDiff`] sequence and renders it as plain text in the
/// unified-diff shape: each change region opens with an `@@` header
/// carrying the real 1-based line numbers and counts of both sides —
/// `-old_start,old_count +new_start,new_count` — followed by the
/// region's lines: unchanged context prefixed with two spaces (`  `),
/// inserted lines with `+ `, deleted lines with `- `. To keep the
/// output readable for large changes, only `context` unchanged lines
/// are shown before and after each run of insertions/deletions, and
/// runs of unchanged lines longer than `context` are elided into the
/// next region's header. The numbers let a reader — model or display —
/// address the change where it sits in the file, not only in the
/// preview.
///
/// A [`LineDiff::Skipped`] is never printed. It closes the run it follows and
/// advances both counters by the lines it stands for, so the numbers on the
/// far side of an elision are still the file's own: what a reader sees is a
/// change's real address even when the diff reached it across unchanged lines
/// it did not print.
fn format_diff_with_context(line_diff: &[LineDiff], context: usize) -> String {
    let mut result = String::new();
    let mut old_at = 0_usize;
    let mut new_at = 0_usize;
    let mut cursor = 0_usize;
    loop {
        let run_end = line_diff
            .get(cursor..)
            .and_then(|rest| {
                rest.iter()
                    .position(|entry| matches!(entry, LineDiff::Skipped { .. }))
            })
            .map_or(line_diff.len(), |offset| cursor.saturating_add(offset));
        let run = line_diff.get(cursor..run_end).unwrap_or_default();
        if !run.is_empty() {
            result.push_str(&format_numbered_run(run, context, old_at, new_at));
            for entry in run {
                match entry {
                    LineDiff::Unchanged(_) => {
                        old_at = old_at.saturating_add(1);
                        new_at = new_at.saturating_add(1);
                    }
                    LineDiff::Deleted(_) => old_at = old_at.saturating_add(1),
                    LineDiff::Inserted(_) => new_at = new_at.saturating_add(1),
                    LineDiff::Skipped { .. } => {}
                }
            }
        }
        let Some(LineDiff::Skipped { old, new }) = line_diff.get(run_end) else {
            break;
        };
        old_at = old_at.saturating_add(*old);
        new_at = new_at.saturating_add(*new);
        cursor = run_end.saturating_add(1);
    }
    result
}

/// Render one numbering run — a stretch of diff entries with no elision in it.
///
/// `old_at` and `new_at` are the lines each side has already accounted for
/// before this run, printed or elided, so numbering continues where the file
/// stands instead of restarting at one. A run with no line on one side is an
/// empty side and is addressed at the line *before* the change: an insertion
/// below line 400 opens `-400,0`, and the head of a file opens at zero.
fn format_numbered_run(
    line_diff: &[LineDiff],
    context: usize,
    old_at: usize,
    new_at: usize,
) -> String {
    /// One numbered diff entry: the kind, its text, and its 1-based
    /// line number on whichever side it occupies.
    struct Numbered<'a> {
        /// Set for a line only the old content holds — printed with `- `.
        deleted: bool,
        /// Set for a line only the new content holds — printed with `+ `.
        inserted: bool,
        /// The line's text, without the gutter prefix.
        text: &'a str,
        /// Its 1-based position in the old file, or `None` if it has none.
        old_no: Option<usize>,
        /// Its 1-based position in the new file, or `None` if it has none.
        new_no: Option<usize>,
    }

    let mut numbered: Vec<Numbered> = Vec::with_capacity(line_diff.len());
    let mut old_no = old_at;
    let mut new_no = new_at;
    for diff in line_diff {
        match diff {
            LineDiff::Unchanged(text) => {
                old_no = old_no.saturating_add(1);
                new_no = new_no.saturating_add(1);
                numbered.push(Numbered {
                    deleted: false,
                    inserted: false,
                    text,
                    old_no: Some(old_no),
                    new_no: Some(new_no),
                });
            }
            LineDiff::Deleted(text) => {
                old_no = old_no.saturating_add(1);
                numbered.push(Numbered {
                    deleted: true,
                    inserted: false,
                    text,
                    old_no: Some(old_no),
                    new_no: None,
                });
            }
            LineDiff::Inserted(text) => {
                new_no = new_no.saturating_add(1);
                numbered.push(Numbered {
                    deleted: false,
                    inserted: true,
                    text,
                    old_no: None,
                    new_no: Some(new_no),
                });
            }
            LineDiff::Skipped { old, new } => {
                old_no = old_no.saturating_add(*old);
                new_no = new_no.saturating_add(*new);
            }
        }
    }

    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut index = 0_usize;
    while let Some(change_at) = numbered.get(index..).and_then(|rest| {
        rest.iter()
            .position(|entry| entry.deleted || entry.inserted)
    }) {
        let change_at = index.saturating_add(change_at);
        let region_start = change_at.saturating_sub(context);
        let mut region_end = change_at.saturating_add(1);
        index = region_end;
        while let Some(next_change) = numbered.get(index..).and_then(|rest| {
            rest.iter()
                .position(|entry| entry.deleted || entry.inserted)
        }) {
            let next_change = index.saturating_add(next_change);
            let next_start = next_change.saturating_sub(context);
            if next_start > region_end {
                break;
            }
            region_end = next_change.saturating_add(1);
            index = region_end;
        }
        region_end = region_end.saturating_add(context).min(numbered.len());
        if let Some((_, last_end)) = regions.last_mut()
            && region_start <= *last_end
        {
            *last_end = (*last_end).max(region_end);
            continue;
        }
        regions.push((region_start, region_end));
    }

    let mut result = String::new();
    for (start, end) in regions {
        let slice = numbered.get(start..end).unwrap_or_default();
        let old_start = slice
            .iter()
            .find_map(|entry| entry.old_no)
            .unwrap_or(old_at);
        let new_start = slice
            .iter()
            .find_map(|entry| entry.new_no)
            .unwrap_or(new_at);
        let old_count = slice.iter().filter(|entry| entry.old_no.is_some()).count();
        let new_count = slice.iter().filter(|entry| entry.new_no.is_some()).count();
        writeln!(
            result,
            "@@ -{old_start},{old_count} +{new_start},{new_count} @@"
        )
        .ok();
        for entry in slice {
            if entry.deleted {
                writeln!(result, "- {}", entry.text).ok();
            } else if entry.inserted {
                writeln!(result, "+ {}", entry.text).ok();
            } else {
                writeln!(result, "  {}", entry.text).ok();
            }
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
    fn created_file_renders_full_additions() {
        // Every line of a new file is an addition, yet the block printed ten
        // of them and a count: a 30-line write showed 10 lines and told the
        // reader about the other 20. The additions are the diff here, so they
        // print.
        let content: String = (1..=30).map(|i| format!("line {i}\n")).collect();
        let result = format_file_change("test.rs", OldContent::Absent, &content);
        assert!(result.contains("Created: test.rs (new file)"), "{result}");
        assert!(result.contains("+ line 30"), "{result}");
        assert!(!result.contains("more lines"), "{result}");
        let added = result.lines().filter(|line| line.starts_with("+ ")).count();
        assert_eq!(added, 30, "every addition is on the receipt: {result}");
    }

    #[test]
    fn a_new_file_past_the_preview_bound_keeps_its_tail_note() {
        // The bound is what stops a generated file from echoing itself into
        // the reader's context, and the tail is what keeps the bound honest
        // about how much it left out.
        let content: String = (1..=1200).map(|i| format!("line {i}\n")).collect();
        let result = format_file_change("test.rs", OldContent::Absent, &content);
        let added = result.lines().filter(|line| line.starts_with("+ ")).count();
        assert_eq!(
            added, LARGE_DIFF_PREVIEW_LINES,
            "the additions block stays bounded"
        );
        assert!(
            result.contains("... +200 more lines"),
            "the tail names what it hid"
        );
        assert!(
            !result.contains("+ line 1001"),
            "nothing prints past the bound"
        );
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
    fn an_all_insertion_header_addresses_the_empty_side_as_zero() {
        let result = format_file_change("test.rs", OldContent::Text(""), "first\nsecond\n");
        let header = result.lines().find(|line| line.starts_with("@@"));
        assert_eq!(
            header,
            Some("@@ -0,0 +1,2 @@"),
            "unified-diff convention addresses the line before the change — \
             an empty side starts at zero, never at one"
        );
    }

    #[test]
    fn an_all_deletion_header_addresses_the_empty_side_as_zero() {
        let result = format_file_change("test.rs", OldContent::Text("first\nsecond\n"), "");
        let header = result.lines().find(|line| line.starts_with("@@"));
        assert_eq!(
            header,
            Some("@@ -1,2 +0,0 @@"),
            "the emptied new side is addressed at zero, never at one"
        );
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
    fn hunk_headers_carry_real_line_numbers() {
        // A change at old line 6 with three context lines renders as a
        // single region spanning old lines 3..=9 — seven old lines and
        // seven new — and the header names those addresses exactly, so
        // a display can show true line numbers beside the preview.
        let old = "1\n2\n3\n4\n5\nOLD\n7\n8\n9\n10\n11\n12\n";
        let new = "1\n2\n3\n4\n5\nNEW\n7\n8\n9\n10\n11\n12\n";
        let result = format_file_change("test.rs", OldContent::Text(old), new);
        assert!(
            result.contains("@@ -3,7 +3,7 @@"),
            "the header names the region's real old/new spans: {result}"
        );

        // A change near the top of the file clamps its leading context
        // at line 1 — the header stays truthful about what follows.
        let old_head = "OLD\na\nb\nc\nd\ne\nf\ng\n";
        let new_head = "NEW\na\nb\nc\nd\ne\nf\ng\n";
        let head = format_file_change("test.rs", OldContent::Text(old_head), new_head);
        assert!(
            head.contains("@@ -1,4 +1,4 @@"),
            "a region starting at line 1 says so: {head}"
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
    fn large_file_edit_emits_hunks() {
        // Two thousand lines against two thousand lines is four million pairs,
        // past the budget the matcher used to check the whole file against —
        // which is where every real source file landed: a `Changed:` header,
        // no `@@` at all, and up to a thousand untouched lines echoed per
        // side. The change is one line wide, so the receipt is now too.
        let old: String = (1..=2000).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=2000)
            .map(|i| {
                if i == 1500 {
                    "CHANGED\n".to_string()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();
        let result = format_file_change("big.rs", OldContent::Text(&old), &new);

        assert!(result.contains("Changed: big.rs (modified)"), "{result}");
        assert!(!result.contains("large file"), "{result}");
        assert!(!result.contains("Before:"), "{result}");
        assert!(result.contains("@@ -1497,7 +1497,7 @@"), "{result}");
        assert!(result.contains("- line 1500"), "{result}");
        assert!(result.contains("+ CHANGED"), "{result}");
        assert_eq!(
            result.lines().count(),
            11,
            "a header, a summary, one hunk header, and the change with three \
             context lines per side — the file itself is never echoed: {result}"
        );
    }

    #[test]
    fn a_line_appended_to_a_large_file_reports_the_end_of_the_file() {
        let old: String = (1..=2000).map(|i| format!("line {i}\n")).collect();
        let new: String = old
            .lines()
            .chain(std::iter::once("APPENDED"))
            .map(|line| format!("{line}\n"))
            .collect();
        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        assert!(result.contains("Changed: big.rs (modified)"), "{result}");
        assert!(!result.contains("large file"), "{result}");
        assert!(result.contains("@@ -1998,3 +1998,4 @@"), "{result}");
        assert!(result.contains("+ APPENDED"), "{result}");
    }

    #[test]
    fn hunk_line_numbers_match_the_file() {
        // Hand-computed fixture: 2 000 distinct lines, the change at line
        // 1 001, so the region addresses lines 998..=1004 on each side.
        // Numbers taken from the change alone would say 1..=7; an elision
        // that lost count of the lines it stepped over would say anything
        // else, which is the failure this fixture exists to catch.
        let old: String = (1..=2000).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=2000)
            .map(|i| {
                if i == 1001 {
                    "CHANGED\n".to_string()
                } else {
                    format!("line {i}\n")
                }
            })
            .collect();
        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        assert!(result.contains("@@ -998,7 +998,7 @@"), "{result}");
        let body: Vec<&str> = result.lines().skip(3).collect();
        assert_eq!(
            body,
            vec![
                "  line 998",
                "  line 999",
                "  line 1000",
                "- line 1001",
                "+ CHANGED",
                "  line 1002",
                "  line 1003",
                "  line 1004",
            ],
            "the region prints exactly the file's own lines around the change: {result}"
        );
    }

    #[test]
    fn two_far_apart_edits_in_a_large_file_report_their_own_hunks() {
        // The stretch between the two changes is ~1 400 lines per side — over
        // the matcher's budget on its own — so the region is aligned on lines
        // each side holds once, and each change keeps its real address.
        let lines: Vec<String> = (1..=2000).map(|i| format!("line {i}")).collect();
        let mut changed = lines.clone();
        changed[9] = "FIRST".to_string();
        changed[1499] = "SECOND".to_string();
        let old = lines.join("\n");
        let new = changed.join("\n");

        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        let hunks: Vec<&str> = result
            .lines()
            .filter(|line| line.starts_with("@@"))
            .collect();
        assert_eq!(
            hunks,
            vec!["@@ -7,7 +7,7 @@", "@@ -1497,7 +1497,7 @@"],
            "both changes report where they sit: {result}"
        );
        assert!(
            result.contains("2 lines removed, 2 lines added"),
            "{result}"
        );
        assert!(
            !result.contains("line 700"),
            "the stretch between them is not echoed: {result}"
        );
        assert!(
            result.lines().count() <= 20,
            "two regions cost two hunks, not the file between them: {result}"
        );
    }

    #[test]
    fn two_edits_within_the_context_margin_share_one_hunk() {
        // The merge the elision depends on: changes close enough that their
        // context windows touch print as one region, exactly as two changes in
        // a small file do.
        let lines: Vec<String> = (1..=2000).map(|i| format!("line {i}")).collect();
        let mut changed = lines.clone();
        changed[1499] = "FIRST".to_string();
        changed[1502] = "SECOND".to_string();
        let old = lines.join("\n");
        let new = changed.join("\n");

        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        let hunks: Vec<&str> = result
            .lines()
            .filter(|line| line.starts_with("@@"))
            .collect();
        assert_eq!(hunks, vec!["@@ -1497,10 +1497,10 @@"], "{result}");
    }

    #[test]
    fn a_line_deleted_from_a_large_file_reports_both_sides_counts() {
        // A removal is asymmetric, and the header has to say so: seven old
        // lines addressed, six on the new side, because one line is gone.
        let old: String = (1..=2000).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=2000)
            .filter(|i| *i != 1000)
            .map(|i| format!("line {i}\n"))
            .collect();
        let result = format_file_change("big.rs", OldContent::Text(&old), &new);
        assert!(result.contains("@@ -997,7 +997,6 @@"), "{result}");
        assert!(result.contains("- line 1000"), "{result}");
        assert!(result.contains("1 line removed"), "{result}");
        assert!(!result.contains("added"), "{result}");
    }

    #[test]
    fn an_unalignable_large_rewrite_keeps_the_preview() {
        // The fallback keeps its own pin. Interleaved filler shares no line
        // that occurs once on each side, so there is nothing to align the
        // region on; past the budget the honest answer is the bounded
        // before/after block, labelled as such and readable as prose — never a
        // hunk whose numbers would describe lines the diff never reached.
        let old: String = (0..1200).map(|_| "a\nb\n").collect();
        let new: String = (0..1200).map(|_| "b\na\n").collect();
        let result = format_file_change("data.txt", OldContent::Text(&old), &new);
        assert!(
            result.contains("Changed: data.txt (modified, large file)"),
            "{result}"
        );
        assert!(result.contains("Before:"), "{result}");
        assert!(result.contains("After:"), "{result}");
        assert!(
            !result.contains("@@"),
            "no hunk may claim addresses the matcher did not reach: {result}"
        );
        assert_eq!(
            result.lines().count(),
            LARGE_DIFF_PREVIEW_LINES.saturating_mul(2).saturating_add(5),
            "two bounded sides, two tails, one header: {result}"
        );
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
