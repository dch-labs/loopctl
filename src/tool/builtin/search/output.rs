//! Large-output shaping for the search tools.
//!
//! When a tool's result is small enough (at or under
//! [`MAX_INLINE_OUTPUT_BYTES`]) it is returned inline as plain text.
//! When it exceeds the limit, the result is written to a fresh file
//! under the tool context's temp directory and the tool returns a
//! short message naming the path plus a preview and a pointer telling
//! the caller how to read the full content — one oversized search
//! cannot blow out a model's context window. Bodies whose lines
//! outgrow what one numbered Read line can return are wrapped at
//! character boundaries before spilling, so every physical line of a
//! spilled file stays within the Read tool's emission.
//!
//! Any failure along the spill path (the temp directory cannot be
//! created, the file cannot be written) degrades gracefully to inline
//! truncation: the caller always gets a usable result, never an error
//! from this module.
//!
//! Spill lifecycle: files land under the temp directory the caller's
//! `ToolContext` names — under the engine that is the per-session
//! subdir removed when the loop drops. A direct context caller (or a
//! host whose managed session directory could not be created and fell
//! back to process-wide temp) owns cleaning what it passed; degraded
//! attempts remove every file they created, and no pointer is
//! returned unless every chunk reached disk.

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::tool::ToolOutput;

/// Default inline-output limit (50 `KiB`); results over this spill to a file.
///
/// The threshold a tool result must stay at or under to return
/// inline; anything larger is written under the context's temp
/// directory with a preview, so one oversized search cannot blow
/// out a model's context window.
pub const MAX_INLINE_OUTPUT_BYTES: usize = 50 * 1024;

/// The line width spilled bodies wrap to, for Read retrievability.
///
/// Read's default output cap bounds the joined numbered view and
/// always cuts at complete lines, so a physical line wider than
/// that cap can never be emitted — at any offset. Spilled bodies
/// wrap lines past this width at character boundaries, continuation
/// pieces carrying the wrap marker; a quarter of the cap leaves
/// the `cat -n` number, its tab, and any framing three times over
/// while keeping a wrapped piece a readable, page-sized row.
pub const MAX_SPILL_LINE_BYTES: usize = crate::tool::builtin::read::DEFAULT_MAX_BYTES / 4;

/// The prefix marking a wrapped line's continuation pieces.
///
/// Makes wrap points self-describing inside the spilled file: a
/// reader paging through the Read tool concatenates a marked line
/// with the line above it, marker stripped, to rebuild the original
/// line.
const WRAP_CONTINUATION_MARKER: &str = "↪ ";

/// Preview size when output spills or truncates (~10 `KiB`), sliced on a
/// character boundary.
const PREVIEW_BYTES: usize = 10 * 1024;

/// The spill-file sequence, keeping names unique within one process.
///
/// Paired with the process id in the file name, this makes every spill
/// file distinct without pulling a temp-file dependency into the
/// feature — the counter is monotonic for the process lifetime.
static SPILL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Return `content` as a tool result, spilling to files when oversized.
///
/// If `content.len()` is at or under `threshold`, it is returned
/// inline. If it exceeds `threshold`, it is written under `temp_dir`
/// and the returned text carries a preview of the first ~10 `KiB`
/// and points the caller at the Read tool for the full content. A
/// body over the Read tool's own refuse-before-read ceiling (the
/// crate-internal `DEFAULT_MAX_SIZE_BYTES` the read path enforces)
/// is split on line boundaries into several files each at or under
/// that ceiling, every path named — the message must never point at
/// a file the named tool would refuse to open. Before that split,
/// physical lines past [`MAX_SPILL_LINE_BYTES`] wrap at character
/// boundaries with marked continuations — Read's output cap cuts at
/// complete lines, so one over-long line could never be returned at
/// any offset — and the pointer discloses the wrap convention
/// whenever wrapping happened.
///
/// The returned flag says which arm ran: `true` only when the body
/// came back verbatim (it fit `threshold`); both the spill pointer
/// and a degraded inline truncation report `false`. Callers that tag
/// the body's shape — Grep's `Json` display hint — key on this flag
/// so the tag can never desync from the decision made here.
///
/// `threshold` is a parameter (not a const read inside) so tests can
/// drive the spill path with a tiny fixture instead of generating
/// 50 `KiB` of content; production callers pass
/// [`MAX_INLINE_OUTPUT_BYTES`].
///
/// Every failure mode (the temp directory cannot be created, a file
/// cannot be written, the disk fills) degrades to inline truncation
/// with a note — the function never returns an error and never
/// panics, and every spill file the attempt created, including the
/// one whose write failed, is removed so a degraded result never
/// leaves orphaned files.
#[must_use]
pub fn truncate_or_spill(
    content: String,
    tool_name: &str,
    temp_dir: &Path,
    threshold: usize,
) -> (ToolOutput, bool) {
    if content.len() <= threshold {
        return (ToolOutput::text(content), true);
    }

    if let Err(error) = std::fs::create_dir_all(temp_dir) {
        tracing::warn!(
            target: "loopctl::metrics",
            path = %temp_dir.display(),
            error = %error,
            "failed to create spill directory; falling back to inline truncation"
        );
        return (ToolOutput::text(truncate_inline(&content)), false);
    }

    let budget =
        usize::try_from(crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES).unwrap_or(usize::MAX);
    let wrapped = wrap_overlong_lines(&content, MAX_SPILL_LINE_BYTES);
    let chunks = split_into_chunks(&wrapped, budget);
    let Some(written) = spill_chunks(temp_dir, tool_name, &chunks, |file, chunk| {
        file.write_all(chunk.as_bytes())
    }) else {
        return (ToolOutput::text(truncate_inline(&content)), false);
    };

    let preview = preview_slice(&content);
    let wrap_note = if wrapped.len() == content.len() {
        String::new()
    } else {
        format!(
            "lines over {MAX_SPILL_LINE_BYTES} bytes are wrapped at character boundaries; a line starting with `{WRAP_CONTINUATION_MARKER}` continues the line above\n\n"
        )
    };
    let pointer = if let [only] = written.as_slice() {
        format!(
            "result too large to return inline; full output written to: {}\n\n",
            only.display()
        )
    } else {
        let listed = written
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "result too large to return inline; full output written to {} files:\n{}\n\n",
            written.len(),
            listed
        )
    };
    (
        ToolOutput::text(format!(
            "{pointer}{wrap_note}preview (first ~10 `KiB`):\n{preview}\n\nuse the Read tool on the path above to view the full content"
        )),
        false,
    )
}

/// Wrap physical lines past `width`, at character boundaries.
///
/// Bodies whose lines already fit come back byte-equal. An overlong
/// line becomes its leading `width` bytes on one physical line with
/// the remainder in bounded pieces, each continuation prefixed with
/// [`WRAP_CONTINUATION_MARKER`] so the wrap points stay
/// self-describing; stripping the markers and joining the pieces
/// rebuilds the original line exactly. Splits land on `str`
/// boundaries — never inside a multi-byte character.
fn wrap_overlong_lines(content: &str, width: usize) -> String {
    if width == 0 {
        return content.to_string();
    }
    let mut wrapped = String::with_capacity(content.len());
    let mut separator_pending = false;
    for line in content.split('\n') {
        if separator_pending {
            wrapped.push('\n');
        }
        separator_pending = true;
        if line.len() <= width {
            wrapped.push_str(line);
            continue;
        }
        let mut rest = line;
        let mut continuation = false;
        while rest.len() > piece_limit(width, continuation) {
            if continuation {
                wrapped.push('\n');
                wrapped.push_str(WRAP_CONTINUATION_MARKER);
            }
            let cut = char_boundary_prefix(rest, piece_limit(width, continuation));
            wrapped.push_str(&rest[..cut]);
            rest = &rest[cut..];
            continuation = true;
        }
        if !rest.is_empty() {
            if continuation {
                wrapped.push('\n');
                wrapped.push_str(WRAP_CONTINUATION_MARKER);
            }
            wrapped.push_str(rest);
        }
    }
    wrapped
}

/// The content-byte limit for one wrapped piece of a line.
///
/// Continuation pieces must leave room for their marker inside
/// `width`; the first piece takes the whole width.
fn piece_limit(width: usize, continuation: bool) -> usize {
    if continuation {
        width.saturating_sub(WRAP_CONTINUATION_MARKER.len())
    } else {
        width
    }
}

/// The largest character boundary at or below `limit`.
///
/// Never zero for non-empty text: when `limit` falls inside the
/// first character, that character's own length is the boundary, so
/// wrapping always makes progress.
fn char_boundary_prefix(text: &str, limit: usize) -> usize {
    let mut cut = text.floor_char_boundary(limit);
    if cut == 0 && !text.is_empty() {
        cut = text.chars().next().map_or(0, char::len_utf8);
    }
    cut
}

/// Write every chunk to its own spill file, or `None` on any failure.
///
/// One chunk per file, each path from [`spill_file_path`], each file
/// from [`create_spill_file`]. A file joins the ownership list the
/// moment it is created — before its write — so a failure anywhere
/// (this chunk's write, a later chunk's creation) cleans every file
/// the attempt made, including a partially written current chunk.
/// The write step is a parameter so that failure path is pinnable
/// without conjuring a full disk; production passes a `write_all`
/// closure over the chunk's bytes.
fn spill_chunks<F>(
    temp_dir: &Path,
    tool_name: &str,
    chunks: &[&str],
    mut write: F,
) -> Option<Vec<PathBuf>>
where
    F: FnMut(&mut std::fs::File, &str) -> std::io::Result<()>,
{
    let mut written: Vec<PathBuf> = Vec::new();
    for chunk in chunks {
        let spill_path = spill_file_path(temp_dir, tool_name);
        let Some(mut file) = create_spill_file(&spill_path) else {
            remove_orphans(&written);
            return None;
        };
        written.push(spill_path);
        if write(&mut file, chunk).is_err() {
            drop(file);
            remove_orphans(&written);
            return None;
        }
    }
    Some(written)
}

/// Best-effort removal of spill files a degraded result must disown.
///
/// When a later chunk fails after earlier ones reached disk, the
/// inline-truncation fallback names no files — so the half-written
/// set (the failing chunk's partial file included) must not linger
/// behind a pointer nobody gave. A removal failure is logged, never
/// propagated: the caller is already on its degrade path.
fn remove_orphans(written: &[PathBuf]) {
    for orphan in written {
        remove_spill_file(orphan);
    }
}

/// Remove one spill file best-effort, logging a failure.
///
/// Shared by the orphan sweep and the partial-file cleanup on a
/// failed chunk write, so every removal path logs the same way.
fn remove_spill_file(path: &Path) {
    if let Err(error) = std::fs::remove_file(path) {
        tracing::warn!(
            target: "loopctl::metrics",
            path = %path.display(),
            error = %error,
            "failed to remove a partial spill file"
        );
    }
}

/// Split `content` into line-boundary chunks of at most `budget` bytes.
///
/// The read tool refuses files over its own size ceiling, so a spill
/// body larger than that ceiling must reach disk as several files the
/// named tool can actually open. Chunks break on newlines — every
/// tool body is line-oriented (pretty JSON rows, rendered listings) —
/// and reassemble to the original bytes exactly; a body that fits the
/// budget comes back as one chunk. A final suffix with no trailing
/// newline gets the same budget check at the last line boundary, so
/// unterminated output (`CodeSearch` renders without a final newline)
/// cannot ride one over-budget chunk. A single line longer than the
/// budget lands alone in its own chunk rather than being split
/// mid-line (no in-family body produces one: matching lines are
/// bounded by the per-file read cap).
fn split_into_chunks(content: &str, budget: usize) -> Vec<&str> {
    if budget == 0 {
        return vec![content];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut boundary = 0;
    for (offset, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            let after_line = offset.saturating_add(1);
            if after_line.saturating_sub(start) > budget && boundary > start {
                chunks.push(&content[start..boundary]);
                start = boundary;
            }
            boundary = after_line;
            if boundary.saturating_sub(start) >= budget {
                chunks.push(&content[start..boundary]);
                start = boundary;
            }
        }
    }
    if start < content.len() {
        if content.len().saturating_sub(start) > budget && boundary > start {
            chunks.push(&content[start..boundary]);
            start = boundary;
        }
        chunks.push(&content[start..]);
    }
    chunks
}

/// Create the spill file at `path`, exclusively and owner-only.
///
/// `create_new` refuses a path that already exists, so a local user
/// who pre-creates the predictable pid-and-counter name in a shared
/// temp directory can neither redirect the write into their file nor
/// read the results through it; on Unix the mode is `0o600` for the
/// same reason. On platforms without a mode knob the exclusive
/// creation still applies. Every failure — including the collision
/// itself — reads as `None` and the caller degrades to inline
/// truncation, never an error.
fn create_spill_file(path: &Path) -> Option<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .ok()
    }
    #[cfg(not(unix))]
    {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .ok()
    }
}

/// The unique spill-file path for one oversized result.
///
/// Process id plus a monotonic counter plus the tool label keeps every
/// spill distinct within and across sessions sharing the directory,
/// without a temp-file dependency and without consulting a clock (a
/// wall-clock name would change across runs for identical content).
fn spill_file_path(temp_dir: &Path, tool_name: &str) -> PathBuf {
    let sequence = SPILL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    temp_dir.join(format!(
        "loopctl-{tool_name}-{}-{sequence}.txt",
        std::process::id()
    ))
}

/// The inline truncation fallback when spilling is impossible.
///
/// Names the total size and line count so a reader can tell "the
/// result was big and we spilled it" (the spill message names a file)
/// apart from "the result was big and we could not spill it" (this
/// preview is all there is).
fn truncate_inline(content: &str) -> String {
    let total_lines = content.lines().count();
    let total_size = content.len();
    let preview = preview_slice(content);
    format!(
        "Result truncated: {total_size} bytes, {total_lines} lines.\n\n\
         Preview:\n\
         {preview}\n\n\
         [Result was truncated because output exceeded limit and the spill file write failed]"
    )
}

/// Take the first `PREVIEW_BYTES` bytes of `content`, sliced on a
/// character boundary.
///
/// The boundary slice ensures the preview never ends mid-code-point.
fn preview_slice(content: &str) -> &str {
    let cutoff = PREVIEW_BYTES.min(content.len());
    &content[..content.floor_char_boundary(cutoff)]
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn inline_when_under_threshold() {
        let (out, _) = truncate_or_spill("hello".to_string(), "grep", Path::new("/tmp"), 64);
        assert!(!out.is_error);
        assert_eq!(out.text_content(), "hello");
    }

    #[test]
    fn inline_when_exactly_at_threshold() {
        let body = "a".repeat(64);
        let (out, _) = truncate_or_spill(body.clone(), "grep", Path::new("/tmp"), 64);
        assert_eq!(out.text_content(), body);
    }

    #[test]
    fn the_verbatim_flag_marks_the_delivered_inline_arm_and_only_it() {
        let at_threshold = "a".repeat(64);
        let (out, verbatim) =
            truncate_or_spill(at_threshold.clone(), "grep", Path::new("/tmp"), 64);
        assert!(
            verbatim,
            "a body exactly at the threshold is delivered verbatim"
        );
        assert_eq!(out.text_content(), at_threshold);

        let tmp = tempfile::TempDir::new().unwrap();
        let (out, verbatim) = truncate_or_spill("a".repeat(65), "grep", tmp.path(), 64);
        assert!(
            !verbatim,
            "a spilled body is pointer prose, not the verbatim body"
        );
        assert!(out.text_content().contains("result too large"));

        let (out, verbatim) = truncate_or_spill(
            "a".repeat(65),
            "grep",
            Path::new("/proc/dch_should_not_exist"),
            64,
        );
        assert!(
            !verbatim,
            "a degraded inline truncation is not the verbatim body either"
        );
        assert!(out.text_content().contains("Result truncated"));
    }

    #[test]
    fn spills_when_over_threshold() {
        let tmp = tempfile::TempDir::new().unwrap();
        let body = "match line\n".repeat(20); // 220 bytes
        let (out, _) = truncate_or_spill(body, "grep", tmp.path(), 64);
        let text = out.text_content();
        assert!(!out.is_error, "{text}");
        assert!(text.contains("result too large"), "{text}");
        assert!(text.contains("full output written to:"), "{text}");
        assert!(text.contains("use the Read tool"), "{text}");
        let start = text
            .find("written to: ")
            .map(|index| index + "written to: ".len())
            .unwrap();
        let end = text[start..]
            .find('\n')
            .map(|offset| start + offset)
            .unwrap();
        let path_str = text[start..end].trim();
        assert!(
            Path::new(path_str).is_file(),
            "spilled file should exist at {path_str}"
        );
        let spilled = std::fs::read_to_string(path_str).unwrap();
        assert_eq!(spilled.len(), 220, "spill carries the whole body");
    }

    #[test]
    fn spill_names_are_unique_across_calls() {
        let tmp = tempfile::TempDir::new().unwrap();
        let body = "x".repeat(200);
        let (first, _) = truncate_or_spill(body.clone(), "grep", tmp.path(), 64);
        let (second, _) = truncate_or_spill(body, "grep", tmp.path(), 64);
        let path_of = |out: &ToolOutput| -> String {
            let text = out.text_content();
            let start = text.find("written to: ").unwrap() + "written to: ".len();
            let end = text[start..].find('\n').unwrap() + start;
            text[start..end].trim().to_string()
        };
        assert_ne!(path_of(&first), path_of(&second), "distinct spill files");
    }

    #[test]
    fn spill_files_refuse_a_pre_created_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let contested = tmp.path().join("loopctl-grep-0-0.txt");
        std::fs::write(&contested, b"attacker content").unwrap();
        assert!(
            create_spill_file(&contested).is_none(),
            "an existing path must not be truncated or followed into"
        );
        assert_eq!(
            std::fs::read_to_string(&contested).unwrap(),
            "attacker content",
            "the pre-created file must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn spill_files_are_owner_only() {
        use std::os::unix::fs::MetadataExt as _;
        let tmp = tempfile::TempDir::new().unwrap();
        let body = "x".repeat(200);
        let (out, _) = truncate_or_spill(body, "grep", tmp.path(), 64);
        let text = out.text_content();
        let start = text.find("written to: ").unwrap() + "written to: ".len();
        let end = text[start..].find('\n').unwrap() + start;
        let spilled = std::path::Path::new(text[start..end].trim());
        let mode = std::fs::metadata(spilled).unwrap().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "spill files must not be group- or world-readable"
        );
    }

    #[test]
    fn chunk_split_respects_the_budget_and_round_trips() {
        let body = "alpha\nbravo\ncharlie\ndelta\n";
        let chunks = split_into_chunks(body, 13);
        assert!(
            chunks.len() >= 2,
            "a body over the budget must split: {chunks:?}"
        );
        for chunk in &chunks {
            assert!(
                chunk.len() <= 13,
                "every chunk must fit the budget: {chunk:?}"
            );
        }
        assert_eq!(chunks.concat(), body, "chunks must reassemble the body");
        for chunk in &chunks {
            assert!(
                chunk.ends_with('\n') || *chunk == body,
                "chunks break on line boundaries: {chunk:?}"
            );
        }
    }

    #[test]
    fn chunk_split_keeps_a_fitting_body_whole() {
        let body = "one\ntwo\n";
        let chunks = split_into_chunks(body, 1024);
        assert_eq!(chunks, vec!["one\ntwo\n"], "no split under the budget");
    }

    #[test]
    fn a_final_suffix_with_no_trailing_newline_cannot_escape_the_budget() {
        let chunks = split_into_chunks("aaaaa\nbbbbbbbb", 10);
        assert_eq!(
            chunks,
            vec!["aaaaa\n", "bbbbbbbb"],
            "the unterminated tail must split at the last line boundary: {chunks:?}"
        );
        assert_eq!(
            chunks.concat(),
            "aaaaa\nbbbbbbbb",
            "chunks must reassemble the body exactly"
        );
    }

    #[test]
    fn a_failed_chunk_write_removes_every_file_including_the_current_one() {
        use std::io::Write as _;
        let tmp = tempfile::TempDir::new().unwrap();
        let chunks = vec!["first\n", "second\n"];
        let mut attempts = 0;
        let outcome = spill_chunks(tmp.path(), "grep", &chunks, |file, chunk| {
            attempts += 1;
            if attempts == 2 {
                return Err(std::io::Error::other("injected write failure"));
            }
            file.write_all(chunk.as_bytes())
        });
        assert!(
            outcome.is_none(),
            "an injected write failure must degrade the whole spill"
        );
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().collect();
        assert!(
            leftovers.is_empty(),
            "a degraded spill must leave nothing behind, not even the partially written chunk: {leftovers:?}"
        );
    }

    #[test]
    fn oversized_spills_chunk_under_the_read_ceiling_and_name_every_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        // ~11.5 `MiB` of lines: two chunks under the read tool's own
        // 10 `MiB` refuse-before-read ceiling, not one 11.5 `MiB` file
        // the named tool would refuse.
        let line = format!("{}\n", "needle".repeat(64));
        let body = line.repeat(48_000);
        assert!(
            body.len()
                > usize::try_from(crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES)
                    .unwrap_or(usize::MAX)
        );
        let (out, _) = truncate_or_spill(body.clone(), "grep", tmp.path(), 64);
        let text = out.text_content();
        assert!(text.contains("full output written to 2 files"), "{text}");
        assert!(text.contains("use the Read tool"), "{text}");
        let prefix = tmp.path().to_string_lossy().into_owned();
        let mut paths = Vec::new();
        let mut rest = text.as_str();
        while let Some(at) = rest.find(&prefix) {
            let after = &rest[at..];
            let end = after.find('\n').unwrap_or(after.len());
            paths.push(after[..end].trim().to_string());
            rest = &rest[at + end..];
        }
        assert_eq!(paths.len(), 2, "both chunk paths must be named: {text}");
        let mut reassembled = String::new();
        for path in &paths {
            let spilled = std::path::Path::new(path);
            let len = std::fs::metadata(spilled).unwrap().len();
            assert!(
                len <= crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES,
                "each chunk must stay at or under the read ceiling: {len}"
            );
            reassembled.push_str(&std::fs::read_to_string(spilled).unwrap());
        }
        assert_eq!(reassembled, body, "the chunks must carry the whole body");
    }

    #[test]
    fn oversized_unterminated_spills_chunk_under_the_read_ceiling() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ceiling = usize::try_from(crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES)
            .unwrap_or(usize::MAX);
        let line = format!("{}\n", "needle".repeat(64));
        let head_target = ceiling.saturating_sub(100);
        let mut head = line.repeat(head_target / line.len());
        let shortfall = head_target - head.len();
        if shortfall > 0 {
            head.push_str(&"p".repeat(shortfall - 1));
            head.push('\n');
        }
        let body = format!("{head}{}", "t".repeat(200));
        assert!(body.len() > ceiling, "the fixture must exceed the ceiling");
        let (out, _) = truncate_or_spill(body.clone(), "code_search", tmp.path(), 64);
        let text = out.text_content();
        assert!(text.contains("full output written to 2 files"), "{text}");
        let prefix = tmp.path().to_string_lossy().into_owned();
        let mut paths = Vec::new();
        let mut rest = text.as_str();
        while let Some(at) = rest.find(&prefix) {
            let after = &rest[at..];
            let end = after.find('\n').unwrap_or(after.len());
            paths.push(after[..end].trim().to_string());
            rest = &rest[at + end..];
        }
        assert_eq!(paths.len(), 2, "both chunk paths must be named: {text}");
        let mut reassembled = String::new();
        for path in &paths {
            let spilled = std::path::Path::new(path);
            let len = std::fs::metadata(spilled).unwrap().len();
            assert!(
                len <= crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES,
                "each chunk must stay at or under the read ceiling: {len}"
            );
            reassembled.push_str(&std::fs::read_to_string(spilled).unwrap());
        }
        assert_eq!(reassembled, body, "the chunks must carry the whole body");
    }

    #[test]
    fn overlong_lines_wrap_at_character_boundaries_for_read_retrieval() {
        let tmp = tempfile::TempDir::new().unwrap();
        let giant = format!("z{}", "é-x".repeat(3000));
        let body = format!("start\n{giant}\nend\n");
        let (out, _) = truncate_or_spill(body.clone(), "code_search", tmp.path(), 64);
        let text = out.text_content();
        let start = text.find("written to: ").expect("spill pointer") + "written to: ".len();
        let end = text[start..].find('\n').expect("line end") + start;
        let spilled = std::fs::read_to_string(text[start..end].trim()).expect("spill body");
        let mut logical = String::new();
        let mut saw_marker = false;
        for line in spilled.split('\n') {
            assert!(
                line.len() <= MAX_SPILL_LINE_BYTES,
                "every physical line must fit one numbered Read line: {}",
                line.len()
            );
            if let Some(rest) = line.strip_prefix(WRAP_CONTINUATION_MARKER) {
                saw_marker = true;
                logical.push_str(rest);
            } else {
                if !logical.is_empty() {
                    logical.push('\n');
                }
                logical.push_str(line);
            }
        }
        assert!(saw_marker, "the overlong line must actually wrap");
        assert_eq!(
            logical, body,
            "marker-stripped lines must reassemble the exact body"
        );
        assert!(
            text.contains("wrapped at character boundaries"),
            "the pointer must disclose the wrap convention: {text}"
        );
    }

    #[test]
    fn write_failure_degrades_to_inline_truncation() {
        let body = "x".repeat(200);
        let (out, _) = truncate_or_spill(body, "grep", Path::new("/proc/dch_should_not_exist"), 64);
        let text = out.text_content();
        assert!(!out.is_error, "degraded output is still a success: {text}");
        assert!(text.contains("Result truncated"), "{text}");
        assert!(text.contains("spill file write failed"), "{text}");
    }

    #[test]
    fn preview_slice_respects_char_boundary() {
        let s = "abcdefghij";
        assert_eq!(preview_slice(s), "abcdefghij");
        let ascii_prefix = "a".repeat(PREVIEW_BYTES - 1);
        let content = format!("{ascii_prefix}ééé");
        assert!(content.len() > PREVIEW_BYTES);
        let preview = preview_slice(&content);
        assert_eq!(preview, ascii_prefix);
    }
}
