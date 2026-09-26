//! The `FileViewer` tool — paginated, token-efficient file viewing.
//!
//! The complement to the shared `read` tool: where `read` caps at ~200
//! lines for quick lookups, `FileViewer` navigates large files in
//! chunks via `page`/`page_size` (sequential) or `offset`/`limit`
//! (direct seek), with a header naming the current window and a
//! `[Navigate: …]` hint. Only regular files are viewable — a missing
//! or non-regular target (a directory, a FIFO, a device) is a soft
//! error, so a special file can never hang the viewer on an open that
//! never completes or a stream that never ends. Windows are read with
//! a single-pass buffered scan that never holds more than the window
//! and the per-line byte cap in memory, so a huge file costs its
//! window, not its size — and a pathological single-line file costs at
//! most one capped line.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncSeekExt;
use tokio::io::SeekFrom;

use crate::tool::DisplayHint;
use crate::tool::Tool;
use crate::tool::ToolContext;
use crate::tool::ToolError;
use crate::tool::ToolOutput;
use crate::tool::ToolSchema;

use super::FileSession;
use super::require_session;
use super::resolve;
use super::resolve::ResolvePolicy;

/// Default number of lines returned per page.
///
/// Balances token cost against navigation steps: small enough that a
/// page fits comfortably in context, large enough that paging through a
/// typical source file takes a handful of calls.
const DEFAULT_PAGE_SIZE: usize = 100;

/// Maximum page size to prevent excessive token usage.
///
/// Explicit `page_size` and `limit` values clamp here, so a model asking
/// for ten thousand lines gets five hundred and a header that says what
/// it received.
const MAX_PAGE_SIZE: usize = 500;

/// Per-line byte cap the window scan retains for display.
///
/// A line at or under the cap is kept verbatim; a longer line is cut
/// back to the last character boundary at or under the cap and ends
/// with a `… [+N bytes truncated]` marker naming the omitted byte
/// count. The cap keeps the documented window bound honest on files
/// whose lines are pathological — a minified bundle or a one-line
/// JSON dump — where an uncapped line would otherwise cost its whole
/// length in memory and output.
const MAX_LINE_BYTES: usize = 16 * 1024;

/// The paginated file viewer over the session's workspace.
///
/// Navigation is via `page`/`page_size` for sequential paging or
/// `offset`/`limit` for direct seeks, with `offset`+`limit` winning
/// when both are given. The model-facing name is `FileViewer`.
#[derive(Debug, Clone, Default)]
pub struct FileViewerTool;

impl Tool for FileViewerTool {
    fn name(&self) -> &'static str {
        "FileViewer"
    }

    fn description(&self) -> &'static str {
        "View a file with pagination. Shows 100 lines per page by default. \
         Use page parameter for sequential access or offset/limit for direct \
         line access."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to view.  May be relative, in which case it is resolved against the runner's cwd, not the process's. URLs are rejected; a missing file is a soft error, not an invalid-input error."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number to view, 1-indexed (default: 1).  Sequential navigation mode together with `page_size`: page `n` shows lines `(n - 1) * page_size + 1` through `n * page_size`. Ignored when `offset` is supplied; zero is rejected as invalid input."
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of lines per page (default: 100, clamped to 500).  Also serves as the default window size in offset mode when `limit` is omitted. Large requests are lowered to the cap to bound token usage."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line of the window, 1-indexed — alternative to `page`.  Direct-seek navigation mode: when present it takes precedence over `page`/`page_size`, and the window size comes from `limit`. Zero is rejected as invalid input; a start beyond the file's last line is a soft over-seek error."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return — alternative to `page_size`.  Applies only in offset mode; clamped to 500 together with `page_size`. When omitted, the offset-mode window uses `page_size`."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'plain' (default), 'colored', or 'markdown'.  Matching is case-insensitive and accepts aliases ('md' for markdown, 'color'/'ansi' for colored). 'colored' degrades to plain output; any unrecognized value falls back to plain."
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    fn call(
        &self,
        input: Value,
        ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let session = ctx.get_extension::<FileSession>().cloned();
        Box::pin(async move {
            let session = require_session(session)?;
            view_inner(input, &session).await
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// Body of [`Tool::call`].
///
/// Orchestrates parse → resolve → regular-file gate → count → bounds →
/// window → render. Recoverable conditions (missing file, non-regular
/// target, offset/page beyond EOF) are soft [`ToolOutput`] errors; bad
/// args and OS faults become [`ToolError`]. A target that swaps to a
/// non-regular file between the pre-open gate and the opened handle's
/// own stat fails closed as a hard error rather than scan. The line
/// count and the window each come from a streaming scan that retains
/// no more than the window's capped lines.
///
/// # Errors
///
/// Returns `ToolError::InvalidInput` for a missing `file_path`, a
/// URL, a zero `page`/`offset`, or malformed numeric fields, and
/// `ToolError::Execution` on a genuine I/O fault, on a target that
/// opened as a non-regular file, or when the contained handle check
/// fails.
async fn view_inner(input: Value, session: &FileSession) -> Result<ToolOutput, ToolError> {
    let parsed = parse_input(&input)?;
    let full_path =
        resolve::resolve_path(parsed.file_path, session.cwd(), session.resolve_policy())?;

    match tokio::fs::metadata(&full_path).await {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ToolOutput::error_text(format!(
                "File not found: {}",
                parsed.file_path
            )));
        }
        Err(e) => return Err(ToolError::Execution(e.to_string())),
        Ok(meta) if !meta.is_file() => {
            return Ok(ToolOutput::error_text(format!(
                "{} is not a regular file.",
                parsed.file_path
            )));
        }
        Ok(_) => {}
    }
    let mut file = open_verified(session, &full_path).await?;
    let handle_meta = file
        .metadata()
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    if !handle_meta.is_file() {
        return Err(ToolError::Execution(format!(
            "cannot view {}: not a regular file",
            parsed.file_path
        )));
    }
    let total_lines = count_lines(&mut file).await?;
    let bounds = calculate_bounds(&input, total_lines)?;

    if total_lines == 0 {
        return Ok(ToolOutput::text(bounds.format_header(parsed.file_path)));
    }

    if bounds.start > total_lines {
        return Ok(ToolOutput::error_text(format!(
            "File: {}\nOffset {} is beyond file length ({})",
            parsed.file_path, bounds.start, total_lines
        )));
    }

    let window_end = bounds.end.min(total_lines);
    let window_len = window_end.saturating_sub(bounds.start).saturating_add(1);
    let view_lines = read_window(&mut file, bounds.start, window_len).await?;

    let output = render_output(parsed.file_path, &bounds, &view_lines, parsed.output_format);
    Ok(ToolOutput::text(output).with_hint(DisplayHint::Code {
        language: detect_language(parsed.file_path).to_string(),
    }))
}

/// Open the view target, verifying the handle under containment.
///
/// The open handle is what both streaming scans read through, so the
/// contained policy's post-open verification covers every byte they
/// see: a symlink swapped onto a path component after validation
/// cannot serve the view from outside the pinned workspace.
///
/// # Errors
///
/// Returns `ToolError::Execution` when the file cannot be opened or
/// the contained handle check fails.
async fn open_verified(
    session: &FileSession,
    full_path: &Path,
) -> Result<tokio::fs::File, ToolError> {
    let file = tokio::fs::File::open(full_path)
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    if session.resolve_policy() == ResolvePolicy::Contained {
        resolve::verify_handle_inside(&file, full_path, session.cwd(), Some(session.anchor()))?;
    }
    Ok(file)
}

/// Count the file's lines in one streaming pass, holding nothing.
///
/// Counts the way `str::lines()` splits: every `\n` ends a line, and a
/// final unterminated line counts as one. Rewinds the handle first, so
/// the caller may reuse the same open file for the window pass. The
/// cost is one buffered read of the whole file — memory stays at the
/// buffer size, which is what bounds the viewer's footprint on huge
/// files.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] on any seek or read fault.
async fn count_lines(file: &mut tokio::fs::File) -> Result<usize, ToolError> {
    file.seek(SeekFrom::Start(0))
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    let mut reader = tokio::io::BufReader::new(file);
    let mut count = 0usize;
    let mut last = b'\n';
    loop {
        let chunk = reader.fill_buf().await.map_err(|e| read_fault(&e))?;
        if chunk.is_empty() {
            break;
        }
        let mut newlines = 0usize;
        for byte in chunk {
            if *byte == b'\n' {
                newlines = newlines.saturating_add(1);
            }
        }
        count = count.saturating_add(newlines);
        last = *chunk.get(chunk.len().saturating_sub(1)).unwrap_or(&last);
        let len = chunk.len();
        reader.consume(len);
    }
    if last != b'\n' {
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// Collect lines `start..start+count` (1-indexed, inclusive) in one streaming
/// pass, holding only the window.
///
/// Rewinds first, skips the lines before the window byte-wise — the
/// skipped span is counted, never decoded or retained — and collects
/// exactly the window, each line capped at [`MAX_LINE_BYTES`] with the
/// truncation marker naming any omitted bytes. Nothing before or after
/// the window is retained, so the memory cost is the window's capped
/// lines, never the file's size or a single line's full length.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] on any seek fault, or any read
/// fault, including window content that is not UTF-8.
async fn read_window(
    file: &mut tokio::fs::File,
    start: usize,
    count: usize,
) -> Result<Vec<String>, ToolError> {
    file.seek(SeekFrom::Start(0))
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    let mut reader = tokio::io::BufReader::new(file);
    if !skip_lines(&mut reader, start.saturating_sub(1)).await? {
        return Ok(Vec::new());
    }
    let mut window = Vec::new();
    for _ in 0..count {
        match read_capped_line(&mut reader).await? {
            Some(line) => window.push(line),
            None => break,
        }
    }
    Ok(window)
}

/// Advance the reader past `count` complete lines, decoding nothing.
///
/// Consumes bytes through the `count`-th newline without retaining
/// them, the same buffered scan [`count_lines`](fn@count_lines) uses,
/// so a window deep in a huge file costs buffer reads, not skipped
/// text. Returns `false` when EOF arrives before the last newline to
/// skip — the caller's window is empty then — and `true` with the
/// reader positioned at the first byte of line `count + 1`.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] on any read fault.
async fn skip_lines(
    reader: &mut tokio::io::BufReader<&mut tokio::fs::File>,
    count: usize,
) -> Result<bool, ToolError> {
    let mut remaining = count;
    loop {
        if remaining == 0 {
            return Ok(true);
        }
        let chunk = reader.fill_buf().await.map_err(|e| read_fault(&e))?;
        if chunk.is_empty() {
            return Ok(false);
        }
        let len = chunk.len();
        let mut stop_at = None;
        for (offset, byte) in chunk.iter().enumerate() {
            if *byte == b'\n' {
                remaining = remaining.saturating_sub(1);
                if remaining == 0 {
                    stop_at = Some(offset.saturating_add(1));
                    break;
                }
            }
        }
        match stop_at {
            Some(consumed) => reader.consume(consumed),
            None => reader.consume(len),
        }
    }
}

/// Read one display line, capped at [`MAX_LINE_BYTES`].
///
/// Reads through the line's terminating newline (or EOF) in buffered
/// chunks, so an over-long line's tail is consumed without being
/// stored; the retained prefix is cut at the last character boundary
/// at or under the cap and, when anything was omitted, suffixed with
/// the `… [+N bytes truncated]` marker naming the omitted byte count.
/// A carriage return before the terminating newline is stripped, the
/// same `str::lines`-compatible split the line count uses. `None`
/// marks EOF before the line's first byte, matching `str::lines`
/// end-of-input semantics.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] on any read fault, including
/// retained content that is not UTF-8.
async fn read_capped_line(
    reader: &mut tokio::io::BufReader<&mut tokio::fs::File>,
) -> Result<Option<String>, ToolError> {
    let mut retained: Vec<u8> = Vec::new();
    let mut omitted = 0usize;
    let mut saw_bytes = false;
    let mut complete = false;
    while !complete {
        let chunk = reader.fill_buf().await.map_err(|e| read_fault(&e))?;
        if chunk.is_empty() {
            break;
        }
        saw_bytes = true;
        if let Some(index) = chunk.iter().position(|b| *b == b'\n') {
            if let Some(part) = chunk.get(..index) {
                retain_within_cap(part, &mut retained, &mut omitted);
            }
            reader.consume(index.saturating_add(1));
            complete = true;
        } else {
            retain_within_cap(chunk, &mut retained, &mut omitted);
            let len = chunk.len();
            reader.consume(len);
        }
    }
    if !saw_bytes {
        return Ok(None);
    }
    let mut line = if omitted == 0 {
        String::from_utf8(retained).map_err(|_| utf8_fault())?
    } else {
        let boundary = last_char_boundary(&retained);
        let text = String::from_utf8(retained.get(..boundary).unwrap_or(&[]).to_vec())
            .map_err(|_| utf8_fault())?;
        format!("{text} … [+{omitted} bytes truncated]")
    };
    if complete && line.ends_with('\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// Append `part` to `retained` up to [`MAX_LINE_BYTES`], counting the
/// rest as omitted.
///
/// Pure bookkeeping for [`read_capped_line`]: fills the retained buffer
/// to the cap exactly and accumulates every byte past it into `omitted`
/// instead, so the caller's marker can name the true remainder.
fn retain_within_cap(part: &[u8], retained: &mut Vec<u8>, omitted: &mut usize) {
    let room = MAX_LINE_BYTES.saturating_sub(retained.len());
    let kept = part.len().min(room);
    if let Some(slice) = part.get(..kept) {
        retained.extend_from_slice(slice);
    }
    *omitted = omitted.saturating_add(part.len().saturating_sub(kept));
}

/// The largest prefix length of `bytes` that ends on a character boundary.
///
/// Backs off continuation bytes (`10xxxxxx`) from the cut point, so the
/// per-line cap never splits a multi-byte character; the answer is never
/// past the input's end.
fn last_char_boundary(bytes: &[u8]) -> usize {
    let mut boundary = bytes.len();
    while boundary > 0 && bytes.get(boundary).is_some_and(|b| b & 0xC0 == 0x80) {
        boundary = boundary.saturating_sub(1);
    }
    boundary
}

/// The read fault for window content that does not decode.
///
/// Reproduces the message the line-oriented scan raised for undecodable
/// content, so the fault shape is unchanged by the capped reader.
fn utf8_fault() -> ToolError {
    read_fault(&std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "stream did not contain valid UTF-8",
    ))
}

/// Map a streaming read fault, naming the operation.
///
/// Both scans funnel their I/O failures through this one mapper so fault
/// messages keep a consistent shape.
fn read_fault(error: &std::io::Error) -> ToolError {
    ToolError::Execution(format!("Failed to read file: {error}"))
}

/// Parsed and validated `FileViewer` input.
///
/// Carries the path exactly as the model spelled it (for headers and
/// errors) plus the parsed rendering mode; window math happens later,
/// against the counted line total.
struct ParsedInput<'a> {
    /// The file path exactly as supplied by the caller, before cwd resolution.
    ///
    /// Borrowed from the input JSON. Kept in its raw form so headers and
    /// error messages show the path the model named, not the resolved
    /// absolute path.
    file_path: &'a str,

    /// The requested rendering mode, parsed from the `output_format` parameter.
    ///
    /// Defaults to [`OutputFormat::Plain`] when the parameter is absent or
    /// unrecognized. Determines whether the body is rendered as plain
    /// numbered lines, a single markdown fenced block, or (degraded) plain.
    output_format: OutputFormat,
}

/// Extract the file path and output format from the input.
///
/// The path must be present and not a URL; the format is optional and
/// defaults to plain.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a missing `file_path` or a URL.
fn parse_input(input: &Value) -> Result<ParsedInput<'_>, ToolError> {
    let file_path = input
        .get("file_path")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing file_path".to_string()))?;

    resolve::reject_path_url("FileViewer", file_path)?;

    let output_format = input
        .get("output_format")
        .and_then(Value::as_str)
        .map(OutputFormat::from_str)
        .unwrap_or_default();

    Ok(ParsedInput {
        file_path,
        output_format,
    })
}

/// Which rendering mode the caller asked for.
///
/// Selected by the `output_format` parameter in the tool's input schema.
/// `Plain` and `Markdown` are fully wired; `Colored` degrades to plain
/// (ANSI escapes are never emitted in tool text output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OutputFormat {
    /// Plain text with line numbers.
    ///
    /// The default when `output_format` is omitted or set to `"plain"`.
    /// Each line is rendered as `{line_num:>6} │ {content}` with no
    /// decoration or ANSI escapes. Syntax highlighting belongs to the
    /// consumer, not to tool text output.
    #[default]
    Plain,

    /// ANSI-colored output.
    ///
    /// Accepted when the caller passes `"colored"`, `"color"`, or
    /// `"ansi"`, but **degrades to plain** — no ANSI escape bytes are
    /// emitted, since escapes waste model tokens. The variant exists so
    /// callers requesting it get a usable answer rather than an error.
    Colored,

    /// Markdown-fenced output.
    ///
    /// Wraps the *entire* view window in a single fenced code block with
    /// a language tag from `detect_language`.
    Markdown,
}

impl OutputFormat {
    /// Parse the `output_format` parameter, case-insensitively.
    ///
    /// `"md"` is markdown; `"color"`/`"ansi"` are colored (which
    /// degrades to plain); anything unrecognized falls back to plain.
    fn from_str(raw: &str) -> Self {
        match raw.to_lowercase().as_str() {
            "md" | "markdown" => Self::Markdown,
            "color" | "colored" | "ansi" => Self::Colored,
            _ => Self::Plain,
        }
    }
}

/// Render the header, numbered body lines, and navigation hint as one string.
///
/// Builds the full tool output in three parts: the
/// [`ViewBounds::format_header`] two-line header, the numbered body
/// lines (`{:>6} │ {content}`), and the trailing
/// [`ViewBounds::format_hint`] navigation hint (omitted when empty).
fn render_output(
    file_path: &str,
    bounds: &ViewBounds,
    view_lines: &[String],
    output_format: OutputFormat,
) -> String {
    let header = bounds.format_header(file_path);
    let hint = bounds.format_hint();

    let body: Vec<String> = view_lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let line_num = bounds.start.saturating_add(i);
            format!("{line_num:>6} │ {line}")
        })
        .collect();

    let mut output = Vec::new();
    output.push(header);
    output.push(String::new());

    match output_format {
        OutputFormat::Markdown => {
            let lang = detect_language(file_path);
            output.push(format!("```{lang}"));
            output.extend(body);
            output.push("```".to_string());
        }
        OutputFormat::Colored | OutputFormat::Plain => {
            output.extend(body);
        }
    }

    if !hint.is_empty() {
        output.push(hint);
    }

    output.join("\n")
}

/// A computed read window into a file.
///
/// Produced by [`calculate_bounds`] from the caller's `page`/`page_size`
/// or `offset`/`limit` input. All line numbers are 1-indexed and
/// inclusive. The `page`/`total_pages` fields are `Some` only in
/// page-based mode; in offset-based mode they are `None` and the
/// header omits the `(Page n/N)` annotation.
struct ViewBounds {
    /// First line number in the window (1-indexed, inclusive).
    ///
    /// Computed as `(page - 1) * page_size + 1` in page mode, or the raw
    /// `offset` value in offset mode. When `start > total`, the view is
    /// beyond EOF and the caller returns a soft over-seek error.
    start: usize,

    /// Last line number in the window (1-indexed, inclusive).
    ///
    /// Computed as `start + window_size - 1`. The actual rendered end may
    /// be clamped to the file's line count (minimum 1) if the window
    /// extends past the file's last line.
    end: usize,

    /// Total number of lines in the file.
    ///
    /// Computed once from the streaming count. Used for the header's
    /// `of T` annotation, the over-seek check, and the navigation hint's
    /// "is this the last window?" decision.
    total: usize,

    /// Page number when in page-based mode.
    ///
    /// `Some(page)` when the caller navigated via `page`/`page_size`;
    /// `None` when in offset-based mode. Drives the `(Page n/N)` header
    /// annotation and the `page=N+1` / `page=N-1` navigation hints.
    page: Option<usize>,

    /// Total page count when in page-based mode.
    ///
    /// `Some(total_pages)` alongside [`page`](Self::page); computed as
    /// `total_lines.div_ceil(page_size)`. Always paired with `page` —
    /// both are `Some` or both are `None`.
    total_pages: Option<usize>,
}

impl ViewBounds {
    /// Format the two-line header for the view window.
    ///
    /// In page-based mode: `File: <path> (Page n/N)\nLines a-b of T`.
    /// In offset-based mode: `File: <path>\nLines a-b of T`.
    ///
    /// The exact header wording is a stable contract: the model has
    /// learned to read this format, and downstream consumers may parse
    /// it. Do not change the wording without coordinating across all
    /// readers.
    fn format_header(&self, file_path: &str) -> String {
        let page_info = if let (Some(page), Some(total_pages)) = (self.page, self.total_pages) {
            format!(" (Page {page}/{total_pages})")
        } else {
            String::new()
        };
        format!(
            "File: {file_path}{page_info}\nLines {}-{} of {}",
            self.start, self.end, self.total
        )
    }

    /// Format the trailing `[Navigate: …]` hint for the model.
    ///
    /// The hint tells the model which parameter values retrieve the next
    /// or previous window. In page-based mode, the hint offers
    /// `page=N+1` (if not the last page) and `page=N-1` (if not the
    /// first). In both modes, `offset=1` is offered when the window
    /// doesn't start at line 1, and `offset=end+1` when it doesn't reach
    /// the last line — so the model can always jump to either end of the
    /// file regardless of which navigation mode it's in.
    ///
    /// Returns an empty string when the window spans the entire file (no
    /// navigation possible), so the caller can omit the hint entirely.
    fn format_hint(&self) -> String {
        let mut hints = Vec::new();

        if let (Some(page), Some(total_pages)) = (self.page, self.total_pages) {
            if page < total_pages {
                hints.push(format!("page={}", page.saturating_add(1)));
            }
            if page > 1 {
                hints.push(format!("page={}", page.saturating_sub(1)));
            }
        }

        if self.start > 1 {
            hints.push("offset=1".to_string());
        }

        if self.end < self.total {
            let next_offset = self.end.saturating_add(1);
            hints.push(format!("offset={next_offset}"));
        }

        if hints.is_empty() {
            String::new()
        } else {
            format!("\n[Navigate: {}]", hints.join(" | "))
        }
    }
}

/// Compute the view window from the input parameters.
///
/// When `offset` is present, offset-mode is used (ignoring `page`/
/// `page_size`). Otherwise page-mode is used with `page` defaulting to
/// 1 and `page_size` to [`DEFAULT_PAGE_SIZE`]. Both `page_size` and
/// `limit` are clamped to [`MAX_PAGE_SIZE`].
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when `page == 0`, `offset == 0`,
/// or any explicitly supplied numeric field is non-integer, negative,
/// or out of range.
fn calculate_bounds(input: &Value, total_lines: usize) -> Result<ViewBounds, ToolError> {
    let page_size = json_usize_strict(input, "page_size")?
        .unwrap_or(DEFAULT_PAGE_SIZE)
        .min(MAX_PAGE_SIZE);

    if input.get("offset").is_some() {
        bounds_from_offset(input, total_lines, page_size)
    } else {
        bounds_from_page(input, total_lines, page_size)
    }
}

/// Extract an optional `usize` integer field from JSON, rejecting malformed
/// values.
///
/// Returns `Ok(None)` when the key is absent (caller applies a default).
/// Returns `Ok(Some(n))` when the key is present and is a valid
/// non-negative integer that fits in `usize`. Returns
/// `Err(InvalidInput)` when the key is present but is not an integer,
/// is negative, or exceeds the platform's `usize` range — so malformed
/// input is caught rather than silently defaulted.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the key is present but the
/// value is not a valid non-negative integer.
fn json_usize_strict(input: &Value, key: &str) -> Result<Option<usize>, ToolError> {
    match input.get(key) {
        None => Ok(None),
        Some(Value::Number(n)) => n
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| {
                ToolError::InvalidInput(format!("'{key}' must be a non-negative integer"))
            }),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "'{key}' must be a non-negative integer"
        ))),
    }
}

/// Compute bounds in offset-based mode (`offset`+`limit`).
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when `offset == 0` or any supplied
/// field is malformed.
fn bounds_from_offset(
    input: &Value,
    total_lines: usize,
    page_size: usize,
) -> Result<ViewBounds, ToolError> {
    let offset = json_usize_strict(input, "offset")?
        .ok_or_else(|| ToolError::InvalidInput("Missing offset value".to_string()))?;
    if offset == 0 {
        return Err(ToolError::InvalidInput(
            "Offset must be at least 1".to_string(),
        ));
    }

    let limit = json_usize_strict(input, "limit")?
        .unwrap_or(page_size)
        .min(MAX_PAGE_SIZE);

    Ok(ViewBounds {
        start: offset,
        end: offset
            .saturating_add(limit)
            .saturating_sub(1)
            .min(total_lines.max(1)),
        total: total_lines,
        page: None,
        total_pages: None,
    })
}

/// Compute bounds in page-based mode (`page`+`page_size`).
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when `page == 0` or any supplied
/// field is malformed.
fn bounds_from_page(
    input: &Value,
    total_lines: usize,
    page_size: usize,
) -> Result<ViewBounds, ToolError> {
    let page = json_usize_strict(input, "page")?.unwrap_or(1);

    if page == 0 {
        return Err(ToolError::InvalidInput(
            "Page must be at least 1".to_string(),
        ));
    }

    let start = page
        .saturating_sub(1)
        .saturating_mul(page_size)
        .saturating_add(1);
    let end = start
        .saturating_add(page_size)
        .saturating_sub(1)
        .min(total_lines.max(1));
    let total_pages = if total_lines == 0 {
        1
    } else {
        total_lines.div_ceil(page_size.max(1))
    };

    Ok(ViewBounds {
        start,
        end,
        total: total_lines,
        page: Some(page),
        total_pages: Some(total_pages),
    })
}

/// Detect a language tag from the file's extension.
///
/// Used by [`OutputFormat::Markdown`] to tag the fenced code block and
/// by the output hint so non-terminal consumers can apply their own
/// highlighting. The mapping is a small inline extension match — not a
/// full language-detection heuristic — and is purely for output
/// tagging. Extensions with no known mapping return an empty string,
/// which renders as a bare fence with no language hint.
fn detect_language(file_path: &str) -> &'static str {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext.to_lowercase().as_str() {
        "rs" => "rust",
        "py" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" => "typescript",
        "go" => "go",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "java" => "java",
        "rb" => "ruby",
        "sh" | "bash" => "bash",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "html" => "html",
        "css" => "css",
        "sql" => "sql",
        "md" => "markdown",
        _ => "",
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing,
    clippy::format_collect
)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        super::super::FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
        ctx
    }

    #[test]
    fn bounds_page_one() {
        let input = json!({"page": 1, "page_size": 100});
        let b = calculate_bounds(&input, 1000).unwrap();
        assert_eq!(b.start, 1);
        assert_eq!(b.end, 100);
        assert_eq!(b.page, Some(1));
        assert_eq!(b.total_pages, Some(10));
    }

    #[test]
    fn bounds_page_two() {
        let input = json!({"page": 2, "page_size": 100});
        let b = calculate_bounds(&input, 1000).unwrap();
        assert_eq!(b.start, 101);
        assert_eq!(b.end, 200);
        assert_eq!(b.page, Some(2));
    }

    #[test]
    fn bounds_offset_mode() {
        let input = json!({"offset": 500, "limit": 50});
        let b = calculate_bounds(&input, 1000).unwrap();
        assert_eq!(b.start, 500);
        assert_eq!(b.end, 549);
        assert!(b.page.is_none());
    }

    #[test]
    fn bounds_default_empty_input() {
        let input = json!({});
        let b = calculate_bounds(&input, 1000).unwrap();
        assert_eq!(b.start, 1);
        assert_eq!(b.end, 100);
        assert_eq!(b.page, Some(1));
    }

    #[test]
    fn bounds_invalid_page_zero() {
        assert!(calculate_bounds(&json!({"page": 0}), 1000).is_err());
    }

    #[test]
    fn bounds_invalid_offset_zero() {
        assert!(calculate_bounds(&json!({"offset": 0}), 1000).is_err());
    }

    #[test]
    fn bounds_page_size_clamped() {
        let input = json!({"page": 1, "page_size": 1000});
        let b = calculate_bounds(&input, 10000).unwrap();
        assert_eq!(b.end - b.start + 1, MAX_PAGE_SIZE);
    }

    #[test]
    fn bounds_end_clamped_to_total_in_page_mode() {
        let input = json!({"page": 1, "page_size": 100});
        let b = calculate_bounds(&input, 5).unwrap();
        assert_eq!(b.start, 1);
        assert_eq!(b.end, 5, "end must clamp to total_lines");
    }

    #[test]
    fn bounds_end_clamped_to_total_in_offset_mode() {
        let input = json!({"offset": 8, "limit": 100});
        let b = calculate_bounds(&input, 10).unwrap();
        assert_eq!(b.start, 8);
        assert_eq!(b.end, 10, "end must clamp to total_lines");
    }

    #[test]
    fn bounds_end_clamped_on_partial_final_page() {
        let input = json!({"page": 2, "page_size": 100});
        let b = calculate_bounds(&input, 150).unwrap();
        assert_eq!(b.start, 101);
        assert_eq!(b.end, 150, "partial final page must clamp to total");
    }

    #[test]
    fn bounds_end_clamped_on_empty_file() {
        let input = json!({});
        let b = calculate_bounds(&input, 0).unwrap();
        assert_eq!(b.end, 1, "empty file end must be 1 (max(1))");
        assert_eq!(b.total_pages, Some(1), "empty file must show 1 page");
    }

    #[test]
    fn bounds_offset_precedence_over_page() {
        let input = json!({"page": 2, "page_size": 100, "offset": 500, "limit": 50});
        let b = calculate_bounds(&input, 1000).unwrap();
        assert_eq!(b.start, 500);
        assert_eq!(b.end, 549);
        assert!(b.page.is_none());
    }

    #[test]
    fn header_contains_page_and_lines() {
        let b = ViewBounds {
            start: 1,
            end: 100,
            total: 1000,
            page: Some(1),
            total_pages: Some(10),
        };
        let h = b.format_header("test.rs");
        assert!(h.contains("test.rs"), "{h}");
        assert!(h.contains("Page 1/10"), "{h}");
        assert!(h.contains("Lines 1-100 of 1000"), "{h}");
    }

    #[test]
    fn hint_first_page_offers_next() {
        let b = ViewBounds {
            start: 1,
            end: 100,
            total: 1000,
            page: Some(1),
            total_pages: Some(10),
        };
        let h = b.format_hint();
        assert!(h.contains("page=2"), "{h}");
        assert!(!h.contains("page=0"), "{h}");
    }

    #[test]
    fn hint_last_page_offers_previous_and_no_next() {
        let b = ViewBounds {
            start: 901,
            end: 1000,
            total: 1000,
            page: Some(10),
            total_pages: Some(10),
        };
        let h = b.format_hint();
        assert!(h.contains("page=9"), "{h}");
        assert!(!h.contains("page=11"), "{h}");
        assert!(h.contains("offset=1"), "{h}");
    }

    #[test]
    fn hint_middle_page_offers_both() {
        let b = ViewBounds {
            start: 101,
            end: 200,
            total: 1000,
            page: Some(2),
            total_pages: Some(10),
        };
        let h = b.format_hint();
        assert!(h.contains("page=3"), "{h}");
        assert!(h.contains("page=1"), "{h}");
        assert!(h.contains("offset=1"), "{h}");
        assert!(h.contains("offset=201"), "{h}");
    }

    #[test]
    fn hint_whole_file_is_empty() {
        let b = ViewBounds {
            start: 1,
            end: 100,
            total: 100,
            page: Some(1),
            total_pages: Some(1),
        };
        assert_eq!(b.format_hint(), "");
    }

    #[test]
    fn offset_mode_hint_offers_both_offsets() {
        let b = ViewBounds {
            start: 500,
            end: 549,
            total: 1000,
            page: None,
            total_pages: None,
        };
        let h = b.format_hint();
        assert!(h.contains("offset=1"), "{h}");
        assert!(h.contains("offset=550"), "{h}");
    }

    #[test]
    fn malformed_numeric_fields_rejected() {
        assert!(calculate_bounds(&json!({"page": "one"}), 1000).is_err());
        assert!(calculate_bounds(&json!({"page_size": -1}), 1000).is_err());
        assert!(calculate_bounds(&json!({"offset": 5, "limit": "x"}), 1000).is_err());
    }

    #[tokio::test]
    async fn first_page_shows_numbered_lines_and_navigation() {
        let tmp = tempfile::TempDir::new().unwrap();
        let content: String = (1..=250).map(|i| format!("line {i}\n")).collect();
        std::fs::write(tmp.path().join("big.txt"), content).unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "big.txt"}), &ctx)
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("File: big.txt (Page 1/3)"), "{text}");
        assert!(text.contains("Lines 1-100 of 250"), "{text}");
        assert!(text.contains("     1 │ line 1"), "{text}");
        assert!(text.contains("   100 │ line 100"), "{text}");
        assert!(!text.contains("line 101"), "{text}");
        assert!(text.contains("[Navigate: page=2 | offset=101]"), "{text}");
    }

    #[tokio::test]
    async fn offset_window_reads_the_requested_lines() {
        let tmp = tempfile::TempDir::new().unwrap();
        let content: String = (1..=250).map(|i| format!("line {i}\n")).collect();
        std::fs::write(tmp.path().join("big.txt"), content).unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "big.txt", "offset": 150, "limit": 10}),
                &ctx,
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("Lines 150-159 of 250"), "{text}");
        assert!(text.contains("line 150"), "{text}");
        assert!(text.contains("line 159"), "{text}");
        assert!(!text.contains("line 149"), "{text}");
        assert!(!text.contains("line 160"), "{text}");
    }

    #[tokio::test]
    async fn markdown_format_fences_the_window() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("code.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "code.rs", "output_format": "markdown"}),
                &ctx,
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("```rust"), "{text}");
        assert!(text.contains("```"), "{text}");
        assert!(text.contains("fn a() {}"), "{text}");
    }

    #[tokio::test]
    async fn colored_degrades_to_plain() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("x.txt"), "one\ntwo\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "x.txt", "output_format": "colored"}),
                &ctx,
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(!text.contains("\u{1b}["), "no ANSI escapes: {text:?}");
        assert!(text.contains("one"), "{text}");
    }

    #[tokio::test]
    async fn missing_file_is_a_soft_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "absent.txt"}), &ctx)
            .await
            .unwrap();
        assert!(out.is_error, "a missing file is recoverable, not a fault");
        assert!(out.text_content().contains("File not found: absent.txt"));
    }

    #[tokio::test]
    async fn offset_beyond_eof_is_a_soft_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("tiny.txt"), "a\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "tiny.txt", "offset": 99}), &ctx)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content()
                .contains("Offset 99 is beyond file length (1)"),
            "{}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn url_file_path_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let err = tool
            .call(json!({"file_path": "https://example.com/x"}), &ctx)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("filesystem path")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn short_file_header_shows_clamped_end() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("short.txt"), "a\nb\nc\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "short.txt"}), &ctx)
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("Lines 1-3 of 3"), "{text}");
        assert!(!text.contains("Lines 1-100"), "{text}");
    }

    #[tokio::test]
    async fn empty_file_header_shows_page_one_of_one() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("empty.txt"), "").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "empty.txt"}), &ctx)
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("Page 1/1"), "{text}");
        assert!(!text.contains("Page 1/0"), "{text}");
    }

    #[tokio::test]
    async fn partial_final_page_header_correct() {
        let tmp = tempfile::TempDir::new().unwrap();
        let content: String = (1..=150).map(|i| format!("line {i}\n")).collect();
        std::fs::write(tmp.path().join("f.txt"), content).unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "f.txt", "page": 2}), &ctx)
            .await
            .unwrap();
        let text = out.text_content();
        assert!(text.contains("Lines 101-150 of 150"), "{text}");
        assert!(!text.contains("101-200"), "{text}");
    }

    #[tokio::test]
    async fn window_lines_match_str_lines_semantics() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("mixed.txt"), "a\nb\nc").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = FileViewerTool;
        let ctx = ctx_in(cwd);
        let out = tool
            .call(json!({"file_path": "mixed.txt"}), &ctx)
            .await
            .unwrap();
        let text = out.text_content();
        assert!(
            text.contains("Lines 1-3 of 3"),
            "an unterminated final line counts as a line: {text}"
        );
    }

    #[tokio::test]
    async fn count_lines_agrees_with_str_lines() {
        for (content, expected) in [
            ("", 0usize),
            ("\n", 1),
            ("a", 1),
            ("a\n", 1),
            ("a\nb", 2),
            ("a\nb\n", 2),
        ] {
            let tmp = tempfile::TempDir::new().unwrap();
            std::fs::write(tmp.path().join("f.txt"), content).unwrap();
            let mut file = tokio::fs::File::open(tmp.path().join("f.txt"))
                .await
                .unwrap();
            let counted = count_lines(&mut file).await.unwrap();
            assert_eq!(
                counted, expected,
                "content {content:?} must count {expected}"
            );
        }
    }

    #[tokio::test]
    async fn read_window_returns_exactly_the_requested_lines() {
        let tmp = tempfile::TempDir::new().unwrap();
        let content: String = (1..=10).map(|i| format!("L{i}\n")).collect();
        std::fs::write(tmp.path().join("f.txt"), content).unwrap();
        let mut file = tokio::fs::File::open(tmp.path().join("f.txt"))
            .await
            .unwrap();
        let window = read_window(&mut file, 4, 3).await.unwrap();
        assert_eq!(
            window,
            vec!["L4".to_string(), "L5".to_string(), "L6".to_string()]
        );
    }

    #[tokio::test]
    async fn a_directory_target_is_a_soft_refusal() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        let out = FileViewerTool
            .call(
                json!({"file_path": "sub"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            )
            .await
            .unwrap();
        assert!(
            out.text_content().contains("sub is not a regular file."),
            "{}",
            out.text_content()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_fifo_target_is_a_soft_refusal_without_hanging() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let tmp = tempfile::TempDir::new().unwrap();
        let fifo = tmp.path().join("pipe");
        let spelled = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: `spelled` names a fresh path in a temp dir this test owns; mkfifo writes no memory.
        let created = unsafe { libc::mkfifo(spelled.as_ptr(), 0o600) };
        assert_eq!(
            created,
            0,
            "mkfifo must succeed: {}",
            std::io::Error::last_os_error()
        );

        let out = FileViewerTool
            .call(
                json!({"file_path": "pipe"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            )
            .await
            .unwrap();
        assert!(
            out.text_content().contains("pipe is not a regular file."),
            "{}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn an_over_long_line_is_capped_with_a_truncation_marker() {
        let tmp = tempfile::TempDir::new().unwrap();
        let long_line = "a".repeat(100 * 1024);
        let content = format!("short one\n{long_line}\nshort three\n");
        std::fs::write(tmp.path().join("bundle.txt"), content).unwrap();
        let out = FileViewerTool
            .call(
                json!({"file_path": "bundle.txt"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(
            text.contains("Lines 1-3 of 3"),
            "an over-long line still counts as one line: {text}"
        );
        assert!(
            text.contains(" … [+86016 bytes truncated]"),
            "the omitted byte count must be named: {text}"
        );
        assert!(
            text.contains(&"a".repeat(16 * 1024)),
            "the retained prefix fills the cap exactly: {text}"
        );
        assert!(
            !text.contains(&"a".repeat(16 * 1024 + 1)),
            "nothing beyond the cap may be retained: {text}"
        );
    }

    #[tokio::test]
    async fn missing_session_is_a_hard_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("f.txt"), "x\n").unwrap();
        let ctx = ToolContext::default();
        let err = FileViewerTool
            .call(
                json!({"file_path": tmp.path().join("f.txt").to_str().unwrap()}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("FileSession"),
            "the error must name the missing wiring: {err}"
        );
    }

    #[test]
    fn flags_advertise_read_only_and_concurrency_safe() {
        assert!(FileViewerTool.is_read_only());
        assert!(FileViewerTool.is_concurrency_safe());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn contained_view_pins_the_workspace_anchor_against_a_later_swap() {
        use std::os::unix::fs::symlink;

        let real = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let workspace = tempfile::TempDir::new().unwrap();
        let anchor_link = workspace.path().join("project");
        symlink(real.path(), &anchor_link).unwrap();
        std::fs::write(real.path().join("data.txt"), "PINNED\n").unwrap();
        std::fs::write(outside.path().join("data.txt"), "OUTSIDE\n").unwrap();

        let ctx = ctx_in(anchor_link.to_str().unwrap());
        std::fs::remove_file(&anchor_link).unwrap();
        symlink(outside.path(), &anchor_link).unwrap();

        let input = json!({"file_path": anchor_link.join("data.txt").to_str().unwrap()});
        let err = FileViewerTool.call(input, &ctx).await.unwrap_err();
        assert!(
            err.to_string().contains("escaped"),
            "the post-swap location must be rejected: {err}"
        );
    }
}
