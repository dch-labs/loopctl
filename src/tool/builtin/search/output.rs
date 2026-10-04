//! Large-output shaping for the search tools.
//!
//! When a tool's result is small enough (at or under
//! [`MAX_INLINE_OUTPUT_BYTES`]) it is returned inline as plain text.
//! When it exceeds the limit, the result is written to a fresh file
//! under the tool context's temp directory and the tool returns a
//! short message naming the path plus a preview and a pointer telling
//! the caller how to read the full content — one oversized search
//! cannot blow out a model's context window.
//!
//! Any failure along the spill path (the temp directory cannot be
//! created, the file cannot be written) degrades gracefully to inline
//! truncation: the caller always gets a usable result, never an error
//! from this module.

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

/// Preview size when output spills or truncates (~10 `KiB`), sliced on a
/// character boundary.
const PREVIEW_BYTES: usize = 10 * 1024;

/// The spill-file sequence, keeping names unique within one process.
///
/// Paired with the process id in the file name, this makes every spill
/// file distinct without pulling a temp-file dependency into the
/// feature — the counter is monotonic for the process lifetime.
static SPILL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Return `content` as a tool result, spilling to a file when oversized.
///
/// If `content.len()` is at or under `threshold`, it is returned
/// inline. If it exceeds `threshold`, it is written to a fresh file
/// under `temp_dir` and the returned text names the path, carries a
/// preview of the first ~10 `KiB`, and points the caller at the Read
/// tool for the full content.
///
/// `threshold` is a parameter (not a const read inside) so tests can
/// drive the spill path with a tiny fixture instead of generating
/// 50 `KiB` of content; production callers pass
/// [`MAX_INLINE_OUTPUT_BYTES`].
///
/// Every failure mode (the temp directory cannot be created, the file
/// cannot be written, the disk fills) degrades to inline truncation
/// with a note — the function never returns an error and never panics.
#[must_use]
pub fn truncate_or_spill(
    content: String,
    tool_name: &str,
    temp_dir: &Path,
    threshold: usize,
) -> ToolOutput {
    if content.len() <= threshold {
        return ToolOutput::text(content);
    }

    if let Err(error) = std::fs::create_dir_all(temp_dir) {
        tracing::warn!(
            target: "loopctl::metrics",
            path = %temp_dir.display(),
            error = %error,
            "failed to create spill directory; falling back to inline truncation"
        );
        return ToolOutput::text(truncate_inline(&content));
    }

    let spill_path = spill_file_path(temp_dir, tool_name);
    let Some(mut file) = create_spill_file(&spill_path) else {
        return ToolOutput::text(truncate_inline(&content));
    };
    if file.write_all(content.as_bytes()).is_err() {
        return ToolOutput::text(truncate_inline(&content));
    }
    drop(file);

    let preview = preview_slice(&content);
    ToolOutput::text(format!(
        "result too large to return inline; full output written to: {}\n\n\
         preview (first ~10 `KiB`):\n{}\n\n\
         use the Read tool on the path above to view the full content",
        spill_path.display(),
        preview
    ))
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
        let out = truncate_or_spill("hello".to_string(), "grep", Path::new("/tmp"), 64);
        assert!(!out.is_error);
        assert_eq!(out.text_content(), "hello");
    }

    #[test]
    fn inline_when_exactly_at_threshold() {
        let body = "a".repeat(64);
        let out = truncate_or_spill(body.clone(), "grep", Path::new("/tmp"), 64);
        assert_eq!(out.text_content(), body);
    }

    #[test]
    fn spills_when_over_threshold() {
        let tmp = tempfile::TempDir::new().unwrap();
        let body = "match line\n".repeat(20); // 220 bytes
        let out = truncate_or_spill(body, "grep", tmp.path(), 64);
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
        let first = truncate_or_spill(body.clone(), "grep", tmp.path(), 64);
        let second = truncate_or_spill(body, "grep", tmp.path(), 64);
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
        let out = truncate_or_spill(body, "grep", tmp.path(), 64);
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
    fn write_failure_degrades_to_inline_truncation() {
        let body = "x".repeat(200);
        let out = truncate_or_spill(body, "grep", Path::new("/proc/dch_should_not_exist"), 64);
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
