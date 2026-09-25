//! The `read` tool and the [`ContentSource`] seam it is parameterized by.
//!
//! A read that silently crops content is worse than no read at all: the
//! model cannot tell a cut file from a complete one. This module is
//! loopctl's answer, policy-free — loopctl knows addresses and text,
//! nothing else. The source decides what an address means (a filesystem
//! path, a blob at a pinned revision, a document id) and whether content
//! arrives as decoded text or raw bytes; the tool owns windowing and
//! kind: explicit `offset`/`limit` beat `line_range`, zero is rejected,
//! every output line is numbered `cat -n` style, and a truncated view
//! always says so — naming the returned lines and where reading
//! continues — so a partial read can never masquerade as a complete
//! one. Raw bytes get kind detection: an image extension returns a
//! native multipart image part plus a summary, refused when the encoded
//! payload exceeds a `5 MiB` default; anything else is refused as binary
//! by name and size; and a source that can report sizes enables a
//! refuse-before-read guard at a `10 MiB` default.
//!
//! Registering [`ReadTool`] is the only way it enters a session; the
//! `builtin_tools` feature is the only way it compiles.

use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;

use base64::Engine as _;
use serde::Deserialize;
use serde_json::Value;

use crate::message::ImageSource;
use crate::message::ToolContent;
use crate::message::ToolContentPart;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Default line ceiling, sized so whole-file reads stay the default.
///
/// Pagination-by-default would multiply turns and bet on the model's
/// diligence; larger views page via `offset`/`limit`, each page
/// self-describing.
const DEFAULT_MAX_LINES: usize = 200;

/// Default output ceiling in bytes, guarding long-line content.
///
/// A single minified line can exceed any line ceiling, so the joined
/// numbered output is also capped in bytes: the cut lands on the last
/// complete line and the marker names the returned range, the next
/// offset, and the remedy.
const DEFAULT_MAX_BYTES: usize = 400_000;

/// Default `limit` when `offset` is given but `limit` is not.
///
/// Reads that start mid-content get the same window size as a read from
/// the top: the model is navigating, not skimming, and a stable page size
/// keeps successive windows predictable.
const DEFAULT_OFFSET_LIMIT: usize = 200;

/// Default outright-refusal threshold, in bytes.
///
/// Past this size the tool refuses before reading when the source reports
/// a size — the windowing ceilings trim views, but an unbounded read into
/// memory is a failure no marker can frame.
const DEFAULT_MAX_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Default ceiling on the encoded image payload, in bytes.
///
/// Base64 output is four thirds the input, and providers cap images per
/// request (Anthropic at 5 MB) — an oversized image would fail on the
/// provider request, outside the tool, poisoning every later turn of the
/// session. The tool refuses first, in its own output, where the model
/// can read why.
const DEFAULT_MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// The content a [`ContentSource`] can return for one address.
///
/// Text flows into windowing and numbering; bytes flow into kind
/// detection — an image extension turns them into a multipart image,
/// anything else is refused as binary. Sources decide where the split is:
/// a filesystem source decodes UTF-8 itself and returns
/// [`SourceContent::Text`] only for content it knows is text.
#[derive(Debug, Clone)]
pub enum SourceContent {
    /// Fully decoded UTF-8 text, windowed and numbered by the tool.
    ///
    /// Sources that decode themselves hand this over; the tool never
    /// re-decodes or lossily converts.
    Text(String),

    /// Raw bytes whose interpretation the tool decides by address kind.
    ///
    /// Image extensions become multipart image parts; everything else is
    /// refused as binary.
    Bytes(Vec<u8>),
}

/// A policy-free content source: loopctl knows addresses and content.
///
/// Implementations decide what an address means — a file on disk, a blob
/// at a pinned revision, a virtual document. The tool never inspects the
/// address, never caches, and never retries: one call, one honest result,
/// errors surface verbatim as [`ToolError`].
///
/// Async by boxed future so the trait stays object-safe, matching the
/// house pattern used by [`LoopMemory`](crate::memory::LoopMemory).
pub trait ContentSource: Send + Sync {
    /// Read the full content addressed by `path`.
    ///
    /// Returns the entire content; windowing is the tool's job, so the
    /// source cannot accidentally pre-crop what the tool would then frame
    /// as complete. Sources that decode text return
    /// [`SourceContent::Text`]; sources holding raw bytes return
    /// [`SourceContent::Bytes`] and let the tool decide the kind.
    fn read<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>>;

    /// Report the byte size of the content at `path`, when cheaply known.
    ///
    /// The default of `None` means "unknown — read and see": sources that
    /// cannot probe skip the guard and behave exactly as before, while
    /// sources that can (a filesystem's metadata, a gateway's content
    /// length) enable the tool's refuse-before-read size guard.
    fn size<'a>(
        &'a self,
        _path: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + 'a>> {
        Box::pin(async move { None })
    }
}

/// Line-aware read tool over any [`ContentSource`].
///
/// The windowing contract: `offset` (1-indexed) and `limit` (line count)
/// are the explicit form and win over `line_range` when both are given;
/// `line_range` accepts `'1-100'`, `'50:'`, `':100'`, or `'50'`; zero
/// values are invalid input, never silently defaulted; nothing given
/// means the whole content up to the line ceiling. Every output line is
/// numbered `cat -n` style, a view starting past line 1 says so up front,
/// and a cut view ends with a marker naming the returned range and the
/// next offset — or, when no complete line fits the byte ceiling,
/// pointing at where reading can resume — so a truncated read never
/// looks complete.
///
/// [`is_read_only`](Tool::is_read_only) and
/// [`is_concurrency_safe`](Tool::is_concurrency_safe) are true: reading
/// has no effect to guard.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::future::Future;
/// use std::pin::Pin;
/// use loopctl::tool::ToolRegistry;
/// use loopctl::tool::builtin::ReadTool;
/// use loopctl::tool::builtin::read::ContentSource;
/// use loopctl::tool::builtin::read::SourceContent;
/// use loopctl::tool::ToolError;
///
/// struct MapSource(HashMap<String, String>);
///
/// impl ContentSource for MapSource {
///     fn read<'a>(&'a self, path: &'a str)
///         -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>>
///     {
///         Box::pin(async move {
///             self.0.get(path).cloned().map(SourceContent::Text).ok_or_else(|| {
///                 ToolError::Execution(format!("no such content: {path}"))
///             })
///         })
///     }
/// }
///
/// let mut registry = ToolRegistry::new();
/// registry.register(ReadTool::new(MapSource(HashMap::new())));
/// assert!(registry.contains("read"));
/// ```
#[derive(Debug, Clone)]
pub struct ReadTool<S: ContentSource> {
    /// The address-resolving backing the tool reads through.
    ///
    /// Owned by value: generic, statically dispatched, and cheap to move —
    /// sources are handles (maps, clients, gateways), not payloads.
    source: S,

    /// Ceiling on returned lines per call.
    ///
    /// Both the default window and the clamp for explicit `limit` and
    /// `line_range` values; overridable via [`ReadTool::with_max_lines`].
    max_lines: usize,

    /// Ceiling on the joined numbered output, in bytes.
    ///
    /// Guards long-line content a line ceiling cannot; overridable via
    /// [`ReadTool::with_max_bytes`].
    max_bytes: usize,

    /// `limit` applied when `offset` is given without one.
    ///
    /// Overridable via [`ReadTool::with_default_offset_limit`].
    default_offset_limit: usize,

    /// Outright-refusal threshold in bytes, checked against the source's
    /// size probe before reading.
    ///
    /// Overridable via [`ReadTool::with_max_size_bytes`].
    max_size_bytes: u64,

    /// Ceiling on the encoded image payload, in bytes.
    ///
    /// Checked against the base64 length before encoding leaves the tool,
    /// regardless of the size probe — a provider rejecting an oversized
    /// image fails the whole turn, outside the tool's honest-marker
    /// machinery. Overridable via [`ReadTool::with_max_image_bytes`].
    max_image_bytes: usize,
}

impl<S: ContentSource> ReadTool<S> {
    /// Build a read tool over `source` with the default ceilings.
    ///
    /// 200 lines, 400 000 bytes of joined output, a 200-line default
    /// window for offset-only reads, a `10 MiB` outright-refusal
    /// threshold when the source reports sizes, and a `5 MiB` ceiling on
    /// encoded image payloads.
    #[must_use]
    pub fn new(source: S) -> Self {
        Self {
            source,
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
            default_offset_limit: DEFAULT_OFFSET_LIMIT,
            max_size_bytes: DEFAULT_MAX_SIZE_BYTES,
            max_image_bytes: DEFAULT_MAX_IMAGE_BYTES,
        }
    }

    /// Override the line ceiling.
    ///
    /// Applies to the default window and to the clamp on explicit
    /// `limit` and `line_range` values alike. Values below 1 clamp to 1
    /// — a zero ceiling would frame a self-contradictory empty window.
    #[must_use]
    pub fn with_max_lines(mut self, lines: usize) -> Self {
        self.max_lines = lines.max(1);
        self
    }

    /// Override the joined-output byte ceiling.
    ///
    /// Long-line content is the target: a single minified line can exceed
    /// any line ceiling, and this ceiling is what cuts it honestly.
    /// Values below 1 clamp to 1 — no line is zero bytes wide once
    /// numbered.
    #[must_use]
    pub fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes.max(1);
        self
    }

    /// Override the default window for offset-only reads.
    ///
    /// A stable page size keeps successive windows predictable for a model
    /// navigating a large document. Values below 1 clamp to 1.
    #[must_use]
    pub fn with_default_offset_limit(mut self, lines: usize) -> Self {
        self.default_offset_limit = lines.max(1);
        self
    }

    /// Override the outright-refusal size threshold, in bytes.
    ///
    /// Only consulted when the source's [`size`](ContentSource::size)
    /// probe reports; sources that cannot probe are never refused on
    /// size. Values below 1 clamp to 1.
    #[must_use]
    pub fn with_max_size_bytes(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes.max(1);
        self
    }

    /// Override the encoded-image payload ceiling, in bytes.
    ///
    /// Compared against the base64 length (four thirds the raw size)
    /// before an image part is emitted, regardless of the size probe.
    /// Values below 1 clamp to 1.
    #[must_use]
    pub fn with_max_image_bytes(mut self, bytes: usize) -> Self {
        self.max_image_bytes = bytes.max(1);
        self
    }
}

/// The tool's input, parsed strictly.
///
/// `offset` and `limit` are `usize`: negative values fail deserialization
/// and surface as invalid input, never a silent default.
#[derive(Debug, Default, Deserialize)]
struct ReadInput {
    /// The address the configured source resolves.
    ///
    /// Opaque to the tool: whatever the source's addresses mean, the tool
    /// passes this through verbatim.
    path: String,

    /// Starting line, 1-indexed; at least 1.
    ///
    /// Zero is invalid input, never a silent default to 1.
    #[serde(default)]
    offset: Option<usize>,

    /// Maximum lines to return from the offset.
    ///
    /// Clamped to the configured line ceiling; zero is invalid input.
    #[serde(default)]
    limit: Option<usize>,

    /// Range alternative: `a-b`, `a:`, `:b`, or a single line.
    ///
    /// Ignored whenever `offset` or `limit` is present — the explicit
    /// fields win.
    #[serde(default)]
    line_range: Option<String>,
}

/// Resolve `(offset, limit)` from the input.
///
/// Explicit fields win over `line_range`; zero is invalid input; limits
/// are clamped to the configured ceiling; no range at all reads from the
/// top up to the ceiling.
///
/// # Errors
///
/// [`ToolError::InvalidInput`] for zero `offset`/`limit` and for malformed
/// `line_range` values.
fn resolve_range(
    input: &ReadInput,
    max_lines: usize,
    default_limit: usize,
) -> Result<(usize, usize), ToolError> {
    if input.offset.is_some() || input.limit.is_some() {
        let offset = match input.offset {
            Some(0) => {
                return Err(ToolError::InvalidInput(
                    "offset must be at least 1, got 0".to_string(),
                ));
            }
            Some(n) => n,
            None => 1,
        };
        let limit = match input.limit {
            Some(0) => {
                return Err(ToolError::InvalidInput(
                    "limit must be at least 1, got 0".to_string(),
                ));
            }
            Some(n) => n,
            None => default_limit,
        };
        return Ok((offset, limit.min(max_lines)));
    }
    if let Some(range) = input.line_range.as_deref() {
        let (offset, limit) = parse_line_range(range).map_err(ToolError::InvalidInput)?;
        return Ok((offset, limit.min(max_lines)));
    }
    Ok((1, max_lines))
}

/// Parse a `line_range` string into `(offset, limit)`.
///
/// Supported: `"1-100"` (lines 1 to 100), `"50:"` (50 to end), `":100"`
/// (first 100), `"100"` (line 100 only).
///
/// # Errors
///
/// A descriptive message for empty input, zero values, inverted ranges,
/// or non-numeric parts.
fn parse_line_range(range: &str) -> Result<(usize, usize), String> {
    let range = range.trim();
    if range.is_empty() {
        return Err("line_range must not be empty".to_string());
    }
    if let Some((left, right)) = range.split_once('-') {
        return parse_dash_range((left, right));
    }
    if let Some((left, right)) = range.split_once(':') {
        return parse_colon_range((left, right));
    }
    parse_single_line(range)
}

/// Parse `"a-b"` into `(a, b - a + 1)`.
///
/// # Errors
///
/// Zero, non-numeric, or inverted halves.
fn parse_dash_range((left, right): (&str, &str)) -> Result<(usize, usize), String> {
    let start = parse_line_number(left.trim(), "line_range start")?;
    let end = parse_line_number(right.trim(), "line_range end")?;
    if end < start {
        return Err(format!("line_range end {end} is before start {start}"));
    }
    let limit = end.saturating_sub(start).saturating_add(1);
    Ok((start, limit))
}

/// Parse `"a:"` / `":b"` into an offset and limit.
///
/// # Errors
///
/// Zero or non-numeric halves; both sides empty.
fn parse_colon_range((left, right): (&str, &str)) -> Result<(usize, usize), String> {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() && right.is_empty() {
        return Err("line_range needs at least one side of the colon".to_string());
    }
    if left.is_empty() {
        let end = parse_line_number(right, "line_range end")?;
        return Ok((1, end));
    }
    let start = parse_line_number(left, "line_range start")?;
    if right.is_empty() {
        return Ok((start, usize::MAX));
    }
    let end = parse_line_number(right, "line_range end")?;
    if end < start {
        return Err(format!("line_range end {end} is before start {start}"));
    }
    Ok((start, end.saturating_sub(start).saturating_add(1)))
}

/// Parse a single line number into `(n, 1)`.
///
/// # Errors
///
/// Zero or non-numeric input.
fn parse_single_line(range: &str) -> Result<(usize, usize), String> {
    let line = parse_line_number(range, "line_range")?;
    Ok((line, 1))
}

/// Parse one strictly positive line number.
///
/// # Errors
///
/// A descriptive message for zero or non-numeric text.
fn parse_line_number(text: &str, field: &str) -> Result<usize, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    match trimmed.parse::<usize>() {
        Ok(0) => Err(format!("{field} must be at least 1, got 0")),
        Ok(n) => Ok(n),
        Err(_) => Err(format!(
            "{field} must be a positive whole number, got {trimmed:?}"
        )),
    }
}

/// Render the windowed, numbered view of `content`.
///
/// Numbering is `cat -n` style: right-aligned to the widest shown line
/// number, then a tab. A view starting past line 1 opens with an
/// omission header; a view cut by the line ceiling ends with a
/// truncation footer naming the returned range and the next offset; a
/// partial view that reaches the end closes with a plain range line;
/// the whole content under both ceilings carries no markers at all.
/// The byte ceiling guards long-line content the same way: the joined
/// numbered output is cut back to the last complete line and closed
/// with a footer naming the returned range, the next offset, and the
/// remedy — or, when not even one line fits, a pointer to where
/// reading can resume.
fn format_window(content: &str, offset: usize, limit: usize, max_bytes: usize) -> String {
    let all_lines: Vec<&str> = content.lines().collect();
    let total_lines = all_lines.len();

    if total_lines == 0 {
        return "Content is empty (0 lines)".to_string();
    }

    if offset > total_lines {
        return format!("Offset {offset} is beyond content length ({total_lines})");
    }

    let start_idx = offset.saturating_sub(1);
    let effective_end = offset
        .saturating_add(limit)
        .saturating_sub(1)
        .min(total_lines);
    let view_lines = all_lines.get(start_idx..effective_end).unwrap_or_default();

    let width = effective_end.to_string().len();
    let mut numbered = String::new();
    for (idx, line) in view_lines.iter().enumerate() {
        let number = offset.saturating_add(idx);
        writeln!(numbered, "{number:>width$}\t{line}").ok();
    }
    let numbered = numbered.trim_end_matches('\n').to_string();

    if numbered.len() > max_bytes {
        return byte_capped_view(&numbered, offset, total_lines, max_bytes);
    }

    if offset == 1 && effective_end >= total_lines {
        return numbered;
    }

    let mut output = String::new();
    if offset > 1 {
        write!(
            output,
            "[Lines before offset {offset} omitted — use offset=1 to read from the start]\n\n"
        )
        .ok();
    }
    output.push_str(&numbered);
    if effective_end < total_lines {
        let remaining = total_lines.saturating_sub(effective_end);
        let next_offset = effective_end.saturating_add(1);
        write!(
            output,
            "\n\n[FILE TRUNCATED: Showing lines {offset}-{effective_end} of {total_lines}. \
             Use offset={next_offset} to see the remaining {remaining} lines.]"
        )
        .ok();
    } else {
        write!(
            output,
            "\n\n[Showing lines {offset}-{effective_end} of {total_lines}]"
        )
        .ok();
    }
    output
}

/// Cut the joined numbered output back to complete lines and frame it.
///
/// A partial line is indistinguishable from corrupt content once shown,
/// so the cut lands on the last complete numbered line and the footer
/// names the returned range, the next offset, and the remedy — a
/// smaller window drops bytes along with lines. When not even one
/// complete line fits, the message points at the first window line
/// verified to fit, or just past the window when none does — the advice
/// always advances, and the line past the window is only verified by
/// the read it advises.
fn byte_capped_view(numbered: &str, offset: usize, total_lines: usize, max_bytes: usize) -> String {
    let mut cut = max_bytes.min(numbered.len());
    while !numbered.is_char_boundary(cut) && cut > 0 {
        cut = cut.saturating_sub(1);
    }
    let candidate = numbered.get(..cut).unwrap_or("");
    let complete = match candidate.rfind('\n') {
        Some(idx) => &candidate[..idx],
        None => "",
    };
    if complete.is_empty() {
        return no_complete_line_view(numbered, offset, max_bytes);
    }
    let shown_lines = complete.matches('\n').count().saturating_add(1);
    let last_shown = offset.saturating_add(shown_lines).saturating_sub(1);
    let next_offset = last_shown.saturating_add(1);
    let remaining = total_lines.saturating_sub(last_shown);
    let mut output = String::new();
    if offset > 1 {
        write!(
            output,
            "[Lines before offset {offset} omitted — use offset=1 to read from the start]\n\n"
        )
        .ok();
    }
    output.push_str(complete);
    write!(
        output,
        "\n\n[FILE TRUNCATED: Showing lines {offset}-{last_shown} of {total_lines}, cut at the \
         {max_bytes}-byte output limit — the content has very long lines. Use offset={next_offset} \
         with a smaller limit or line_range to page through the remaining {remaining} lines.]"
    )
    .ok();
    output
}

/// Frame the view when not even one complete line fits the byte cap.
///
/// The first line of the window alone is at or over the limit, but later
/// lines may fit — an absolute "nothing is viewable" claim would lose the
/// readable tail. The numbered lines are scanned for the first one that
/// fits and the message points at its offset — always past the current
/// one, because a window line that fit would have been framed instead.
/// When no line of the window fits, the message points past the window.
/// Every outcome advances, so the advice can never loop. A view starting
/// past line 1 opens with the same omission header every other view
/// carries.
fn no_complete_line_view(numbered: &str, offset: usize, max_bytes: usize) -> String {
    let segments: Vec<&str> = numbered.split('\n').collect();
    let window_end = offset.saturating_add(segments.len()).saturating_sub(1);
    let mut message = if let Some(idx) = segments.iter().position(|line| line.len() < max_bytes) {
        let resume = offset.saturating_add(idx);
        format!(
            "[FILE TRUNCATED: line {offset} alone does not fit the {max_bytes}-byte output \
             limit — the content has very long lines. Use offset={resume} to read past it.]"
        )
    } else {
        let past_window = window_end.saturating_add(1);
        format!(
            "[FILE TRUNCATED: no line from {offset} to {window_end} fits the {max_bytes}-byte \
             output limit — the content has very long lines. Use offset={past_window} to \
             read past this window.]"
        )
    };
    if offset > 1 {
        let mut framed = String::new();
        write!(
            framed,
            "[Lines before offset {offset} omitted — use offset=1 to read from the start]\n\n"
        )
        .ok();
        framed.push_str(&message);
        message = framed;
    }
    message
}

/// The image MIME an address extension selects, if any.
///
/// The set png, jpeg (both spellings), gif, webp — everything a
/// policy-free tool can select without content sniffing, which would
/// drag format knowledge in.
fn image_mime(path: &str) -> Option<&'static str> {
    let extension = path.rsplit_once('.')?.1.to_ascii_lowercase();
    match extension.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Frame a size-guard refusal.
///
/// A soft, successful-shaped refusal (not a [`ToolError`]): the model
/// addressed real content that is simply too large to serve. The
/// message names both numbers and states that no windowed view can
/// bypass the limit — the guard runs before every read, ranged or not,
/// so advising `offset`/`limit` here would send the model into a retry
/// loop.
fn too_large(path: &str, size: u64, cap: u64) -> ToolOutput {
    ToolOutput::text(format!(
        "Content at {path} is {size} bytes, over the {cap}-byte read limit. \
         The limit applies to the whole content, not the requested window — \
         no offset or limit view of it is available through this tool. Read \
         the content another way."
    ))
}

/// Interpret raw bytes by address kind.
///
/// An image extension yields native multipart output — the image part
/// (base64 through [`ImageSource`], the same standard table the message
/// format defines) plus a one-line text summary naming the address and
/// byte count — unless the encoded payload exceeds the configured
/// ceiling, in which case a soft error names the address, both sizes,
/// and the limit: the refusal lands in the tool output, not on a later
/// provider request. Anything else is refused as binary with a soft
/// error: the model gets a readable explanation, not an opaque failure.
fn bytes_output(path: &str, bytes: &[u8], max_image_bytes: usize) -> ToolOutput {
    if let Some(mime) = image_mime(path) {
        let encoded_len = bytes.len().div_ceil(3).saturating_mul(4);
        if encoded_len > max_image_bytes {
            return ToolOutput::error(format!(
                "Image at {path} is {} bytes ({} bytes encoded), over the \
                 {max_image_bytes}-byte image limit.",
                bytes.len(),
                encoded_len
            ));
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let summary = format!("Image at {path} ({} bytes, {mime})", bytes.len());
        return ToolOutput::success(ToolContent::from_multipart(vec![
            ToolContentPart::Image {
                source: ImageSource::new_base64(mime, encoded),
            },
            ToolContentPart::Text { text: summary },
        ]));
    }
    ToolOutput::error(format!(
        "Content at {path} is binary ({} bytes) and cannot be shown as text.",
        bytes.len()
    ))
}

impl<S: ContentSource> Tool for ReadTool<S> {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read content through the configured source, numbered line by line. \
         Whole content when it fits; a cut view always says so, naming the \
         returned lines and where reading continues."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The address the configured source resolves."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Starting line, 1-indexed."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum lines to return from the offset."
                    },
                    "line_range": {
                        "type": "string",
                        "description": "Range alternative: '1-100', '50:', ':100', or '50'. \
                                        Ignored when offset or limit is given."
                    }
                },
                "required": ["path"]
            }),
        }
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }

    fn call(
        &self,
        input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move {
            let parsed: ReadInput = serde_json::from_value(input).map_err(|error| {
                ToolError::InvalidInput(format!("read input must match the schema: {error}"))
            })?;
            let (offset, limit) =
                resolve_range(&parsed, self.max_lines, self.default_offset_limit)?;
            if let Some(size) = self.source.size(&parsed.path).await
                && size > self.max_size_bytes
            {
                return Ok(too_large(&parsed.path, size, self.max_size_bytes));
            }
            match self.source.read(&parsed.path).await? {
                SourceContent::Text(text) => Ok(ToolOutput::text(format_window(
                    &text,
                    offset,
                    limit,
                    self.max_bytes,
                ))),
                SourceContent::Bytes(bytes) => {
                    Ok(bytes_output(&parsed.path, &bytes, self.max_image_bytes))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A fake source: text and byte entries, optional reported sizes, an
    /// optional failing address, and a `read` call counter for pins that
    /// must prove the tool never fetched.
    ///
    /// No filesystem, no network — the whole suite is hermetic.
    struct FakeSource {
        entries: HashMap<String, SourceContent>,
        sizes: HashMap<String, u64>,
        failing: Option<String>,
        reads: std::sync::atomic::AtomicU32,
    }

    impl FakeSource {
        /// Build a source holding the given text entries.
        ///
        /// Addresses map verbatim to decoded text; no filesystem is
        /// involved.
        fn with(entries: &[(&str, &str)]) -> Self {
            Self {
                entries: entries
                    .iter()
                    .map(|(path, content)| {
                        (
                            (*path).to_string(),
                            SourceContent::Text((*content).to_string()),
                        )
                    })
                    .collect(),
                sizes: HashMap::new(),
                failing: None,
                reads: std::sync::atomic::AtomicU32::new(0),
            }
        }

        /// Add a raw-bytes entry, as an image-bearing source would.
        ///
        /// Byte entries exercise the tool's kind detection: image
        /// extensions become multipart output, everything else is refused.
        fn with_bytes(mut self, path: &str, bytes: &[u8]) -> Self {
            self.entries
                .insert(path.to_string(), SourceContent::Bytes(bytes.to_vec()));
            self
        }

        /// Report a size for `path` from the probe.
        ///
        /// Sizes the default `None` probe cannot know; drives the
        /// refuse-before-read guard pins.
        fn reporting_size(mut self, path: &str, size: u64) -> Self {
            self.sizes.insert(path.to_string(), size);
            self
        }

        /// Mark `path` as failing with a fixed error.
        ///
        /// Used to pin that source errors surface verbatim through the tool.
        fn failing_on(mut self, path: &str) -> Self {
            self.failing = Some(path.to_string());
            self
        }

        /// How many times `read` has fired.
        ///
        /// Backs the pins proving the size guard refuses before fetching.
        fn read_count(&self) -> u32 {
            self.reads.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl ContentSource for FakeSource {
        fn read<'a>(
            &'a self,
            path: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if self.failing.as_deref() == Some(path) {
                    return Err(ToolError::Execution(format!("source failed for {path}")));
                }
                self.entries
                    .get(path)
                    .cloned()
                    .ok_or_else(|| ToolError::Execution(format!("no such content: {path}")))
            })
        }

        fn size<'a>(
            &'a self,
            path: &'a str,
        ) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + 'a>> {
            Box::pin(async move { self.sizes.get(path).copied() })
        }
    }

    /// A ten-line fixture, one word per line.
    ///
    /// Words double as line identifiers, so window assertions can name the
    /// lines they expect without off-by-one ambiguity.
    const TEN_LINES: &str = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten";

    /// Run the tool over the fake source and return the output text.
    ///
    /// Every test drives the tool through the public `Tool::call` seam, so
    /// the pins exercise parse, resolve, fetch, and format together.
    async fn read(tool: &ReadTool<FakeSource>, input: Value) -> Result<String, ToolError> {
        tool.call(input, &ToolContext::default())
            .await
            .map(|output| output.payload.to_string())
    }

    /// A base input for `path` with no range fields.
    ///
    /// The minimal valid call: the whole-content default window.
    fn input(path: &str) -> Value {
        serde_json::json!({ "path": path })
    }

    /// A base input with extra fields merged in.
    ///
    /// Keeps each call site to the fields it varies.
    fn with_range(path: &str, fields: &Value) -> Value {
        let mut value = input(path);
        if let (Some(map), Some(fields)) = (value.as_object_mut(), fields.as_object()) {
            for (key, field) in fields {
                map.insert(key.clone(), field.clone());
            }
        }
        value
    }

    #[tokio::test]
    async fn explicit_offset_and_limit_win_over_line_range() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let output = read(
            &tool,
            with_range(
                "doc",
                &serde_json::json!({ "offset": 3, "limit": 2, "line_range": "5-9" }),
            ),
        )
        .await
        .expect("the explicit fields resolve");
        assert!(
            output.contains("three") && output.contains("four"),
            "lines 3-4 are shown, not the 5-9 range: {output}"
        );
        assert!(
            !output.contains("five"),
            "the ignored line_range must not leak into the view: {output}"
        );
        assert!(
            output.contains("[Lines before offset 3 omitted"),
            "a mid-content view opens with the omission header: {output}"
        );
    }

    #[tokio::test]
    async fn the_full_file_is_returned_when_no_range_is_given() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let output = read(&tool, input("doc"))
            .await
            .expect("a small document resolves");
        for word in ["one", "five", "ten"] {
            assert!(
                output.contains(word),
                "the whole document is present, including {word}: {output}"
            );
        }
        assert!(
            output.lines().count() == 10,
            "exactly the ten content lines, no markers: {output}"
        );
        assert!(
            output.starts_with(" 1\tone"),
            "numbering starts at line 1, padded to the widest line number: {output}"
        );
    }

    #[tokio::test]
    async fn a_zero_limit_or_offset_is_rejected_as_invalid_input() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        for bad in [
            serde_json::json!({ "offset": 0 }),
            serde_json::json!({ "limit": 0 }),
        ] {
            let error = read(&tool, with_range("doc", &bad))
                .await
                .expect_err("zero must be rejected, never silently defaulted");
            assert!(
                matches!(error, ToolError::InvalidInput(ref message) if message.contains("at least 1")),
                "the error names the violated minimum, got: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_cut_output_names_the_next_offset_and_the_total_range() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)])).with_max_lines(4);
        let output = read(&tool, input("doc"))
            .await
            .expect("the ceiling applies without erroring");
        assert!(
            output.contains("Showing lines 1-4 of 10"),
            "the marker states the returned range and the total: {output}"
        );
        assert!(
            output.contains("Use offset=5 to see the remaining 6 lines"),
            "the marker names the next offset and what remains: {output}"
        );
        assert!(
            !output.contains("five"),
            "only the first four lines are in the view: {output}"
        );
    }

    #[tokio::test]
    async fn an_uncut_output_carries_no_truncation_marker() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let whole = read(&tool, input("doc"))
            .await
            .expect("a small document resolves");
        assert!(
            !whole.contains("TRUNCATED") && !whole.contains("Showing lines"),
            "a complete view carries no markers at all: {whole}"
        );
        let tail = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 7 })),
        )
        .await
        .expect("the tail resolves");
        assert!(
            tail.contains("[Showing lines 7-10 of 10]"),
            "a partial view that reaches the end closes with the plain range line: {tail}"
        );
        assert!(
            !tail.contains("TRUNCATED"),
            "reaching the end is not truncation: {tail}"
        );
    }

    #[tokio::test]
    async fn a_source_error_surfaces_as_execution_unchanged() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]).failing_on("doc"));
        let error = read(&tool, input("doc"))
            .await
            .expect_err("source failures must not be swallowed");
        assert!(
            matches!(error, ToolError::Execution(ref message) if message == "source failed for doc"),
            "the tool adds no wrapping, retry, or translation: {error:?}"
        );
    }

    #[tokio::test]
    async fn line_range_formats_resolve_to_their_windows() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let dash = read(
            &tool,
            with_range("doc", &serde_json::json!({ "line_range": "2-3" })),
        )
        .await
        .expect("the dash range resolves");
        assert!(
            dash.contains("two") && dash.contains("three") && !dash.contains("four"),
            "2-3 shows exactly lines two and three: {dash}"
        );
        let open_end = read(
            &tool,
            with_range("doc", &serde_json::json!({ "line_range": "9:" })),
        )
        .await
        .expect("the open-end range resolves");
        assert!(
            open_end.contains("nine") && open_end.contains("ten") && !open_end.contains("eight"),
            "9: shows the tail from line nine: {open_end}"
        );
        let open_start = read(
            &tool,
            with_range("doc", &serde_json::json!({ "line_range": ":2" })),
        )
        .await
        .expect("the open-start range resolves");
        assert!(
            open_start.contains("one")
                && open_start.contains("two")
                && !open_start.contains("three"),
            ":2 shows the first two lines: {open_start}"
        );
        let single = read(
            &tool,
            with_range("doc", &serde_json::json!({ "line_range": "4" })),
        )
        .await
        .expect("the single-line range resolves");
        assert!(
            single.contains("four") && !single.contains("three") && !single.contains("five"),
            "4 shows exactly line four: {single}"
        );
    }

    #[tokio::test]
    async fn malformed_line_ranges_are_invalid_input() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        for bad in ["", "5-2", "0-3", "3-0", ":", "abc", "1-x"] {
            let error = read(
                &tool,
                with_range("doc", &serde_json::json!({ "line_range": bad })),
            )
            .await
            .expect_err("a malformed range must be rejected");
            assert!(
                matches!(error, ToolError::InvalidInput(_)),
                "every malformed form is invalid input, {bad:?} got: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn negative_offset_and_limit_are_rejected_not_silently_defaulted() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        for bad in [
            serde_json::json!({ "offset": -1 }),
            serde_json::json!({ "limit": -5 }),
        ] {
            let error = read(&tool, with_range("doc", &bad))
                .await
                .expect_err("negatives must never become usize defaults");
            assert!(
                matches!(error, ToolError::InvalidInput(_)),
                "the parse failure surfaces as invalid input, got: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_limit_above_the_ceiling_is_clamped() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)])).with_max_lines(3);
        let output = read(
            &tool,
            with_range("doc", &serde_json::json!({ "limit": 100 })),
        )
        .await
        .expect("an over-ceiling limit resolves, clamped");
        assert!(
            output.contains("Showing lines 1-3 of 10"),
            "the ceiling wins over the asked limit: {output}"
        );
    }

    #[tokio::test]
    async fn the_byte_cap_cuts_long_lines_with_a_marker_naming_the_remedy() {
        let long_lines = (0..10)
            .map(|_| "x".repeat(50))
            .collect::<Vec<_>>()
            .join("\n");
        let tool = ReadTool::new(FakeSource::with(&[("doc", &long_lines)])).with_max_bytes(100);
        let output = read(&tool, input("doc"))
            .await
            .expect("the byte-capped view resolves");
        assert!(
            output.contains("Showing lines 1-1 of 10")
                && output.contains("Use offset=2")
                && output.contains("smaller limit"),
            "the byte cut names the returned line, the next offset, and the remedy: {output}"
        );
        assert!(
            output.len() < 300,
            "the returned view respects the byte ceiling plus its marker: {}",
            output.len()
        );
        let mid = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 2 })),
        )
        .await
        .expect("a byte-capped mid-content view resolves");
        assert!(
            mid.starts_with("[Lines before offset 2 omitted"),
            "a byte cut mid-content still opens with the omission header: {mid}"
        );
    }

    #[tokio::test]
    async fn a_first_line_over_the_byte_cap_points_past_it() {
        let content = format!("{}\nsecond", "x".repeat(1000));
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)])).with_max_bytes(100);
        let output = read(&tool, input("doc"))
            .await
            .expect("the over-long first line resolves to a framed message");
        assert!(
            output.contains("line 1 alone does not fit the 100-byte output limit")
                && output.contains("Use offset=2 to read past it"),
            "only line 1 is over-long, so the message points at the readable tail: {output}"
        );
        assert!(
            !output.contains("Use offset=1 to read past"),
            "the advice must never name the call that produced it: {output}"
        );
        let tail = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 2 })),
        )
        .await
        .expect("the advised offset resolves to a real view");
        assert!(
            tail.contains("second") && tail.contains("[Showing lines 2-2 of 2]"),
            "the advice leads somewhere: {tail}"
        );
    }

    #[tokio::test]
    async fn a_boundary_exact_first_line_still_advances() {
        // Two lines total, so line numbers render one digit wide: the
        // first numbered line is "1\t" + 98 = 100 bytes — exactly the
        // cap, its newline at the cut point and therefore excluded.
        let content = format!("{}\nsecond", "x".repeat(98));
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)])).with_max_bytes(100);
        let output = read(&tool, input("doc"))
            .await
            .expect("a first line exactly filling the cap resolves to a framed message");
        assert!(
            output.contains("line 1 alone does not fit the 100-byte output limit")
                && output.contains("Use offset=2 to read past it"),
            "a line equal to the cap leaves no room to frame it: {output}"
        );
        assert!(
            !output.contains("Use offset=1 to read past"),
            "exact-equality must not match the first line and echo the current offset: {output}"
        );
    }

    #[tokio::test]
    async fn a_window_where_no_line_fits_points_past_the_window() {
        let content = (0..3)
            .map(|_| "x".repeat(1000))
            .chain(std::iter::once("tail".to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)]))
            .with_max_lines(2)
            .with_max_bytes(100);
        let output = read(&tool, input("doc"))
            .await
            .expect("an all-over-long window resolves to a framed message");
        assert!(
            output.contains("no line from 1 to 2 fits the 100-byte output limit")
                && output.contains("Use offset=3 to read past this window"),
            "the claim is scoped to the window and points past it: {output}"
        );
        let past = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 4 })),
        )
        .await
        .expect("the line past the window resolves");
        assert!(
            past.contains("tail") && past.contains("[Showing lines 4-4 of 4]"),
            "later content lines are readable, so the window-scoped wording is honest: {past}"
        );
        let all_long = (0..3)
            .map(|_| "x".repeat(1000))
            .collect::<Vec<_>>()
            .join("\n");
        let terminal = ReadTool::new(FakeSource::with(&[("doc", &all_long)])).with_max_bytes(100);
        let output = read(&terminal, input("doc"))
            .await
            .expect("an all-over-long document resolves to a framed message");
        assert!(
            output.contains("no line from 1 to 3 fits")
                && output.contains("Use offset=4 to read past this window"),
            "when the window spans the content, the advice points one past the end: {output}"
        );
        let end = read(
            &terminal,
            with_range("doc", &serde_json::json!({ "offset": 4 })),
        )
        .await
        .expect("following the advice resolves");
        assert_eq!(
            end, "Offset 4 is beyond content length (3)",
            "the terminal chain ends at the honest beyond-content message: {end}"
        );
    }

    #[tokio::test]
    async fn an_offset_beyond_the_content_yields_the_one_line_message() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let output = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 11 })),
        )
        .await
        .expect("an out-of-range offset resolves to a message, not an error");
        assert_eq!(
            output, "Offset 11 is beyond content length (10)",
            "the message names the offset and the true length: {output}"
        );
    }

    #[tokio::test]
    async fn numbering_is_right_aligned_to_the_widest_shown_line() {
        let content = (1..=10)
            .map(|n| format!("line{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)]));
        let output = read(&tool, input("doc"))
            .await
            .expect("the fixture resolves");
        let first = output.lines().next().unwrap_or_default();
        assert_eq!(
            first, " 1\tline1",
            "single-digit numbers pad to the width of the widest (10): {first:?}"
        );
        let last = output.lines().last().unwrap_or_default();
        assert_eq!(
            last, "10\tline10",
            "the widest number takes the full width: {last:?}"
        );
    }

    #[tokio::test]
    async fn an_image_address_returns_a_multipart_image_and_a_summary() {
        let tool = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("shot.png", &[1, 2, 3, 4]),
        );
        let output = tool
            .call(input("shot.png"), &ToolContext::default())
            .await
            .expect("an image entry resolves");
        assert!(
            !output.is_error,
            "an image read is a success, not a failure: {output:?}"
        );
        let crate::message::ToolContent::Multipart(parts) = &output.payload else {
            panic!("image output is multipart, got: {:?}", output.payload)
        };
        assert_eq!(
            parts.len(),
            2,
            "one image part plus one text summary: {parts:?}"
        );
        assert!(
            matches!(&parts[0], crate::message::ToolContentPart::Image { source }
                if source.media_type == "image/png"
                    && source.encoding == "base64"
                    && source.data == "AQIDBA=="),
            "the image part carries the extension-selected MIME and the standard-base64 \
             payload of the four input bytes: {:?}",
            parts[0]
        );
        assert!(
            matches!(&parts[1], crate::message::ToolContentPart::Text { text }
                if text.contains("shot.png") && text.contains("4 bytes") && text.contains("image/png")),
            "the summary names the address, size, and kind: {:?}",
            parts[1]
        );
    }

    #[tokio::test]
    async fn an_oversized_image_is_refused_naming_both_sizes() {
        let tool = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("shot.png", &[1, 2, 3, 4]),
        )
        .with_max_image_bytes(4);
        let output = tool
            .call(input("shot.png"), &ToolContext::default())
            .await
            .expect("an over-ceiling image resolves to a refusal, not an error");
        assert!(
            output.is_error,
            "the image ceiling refuses in the tool output, not on a later provider request: {output:?}"
        );
        let text = output.payload.to_string();
        assert!(
            text.contains("shot.png")
                && text.contains("4 bytes")
                && text.contains("8 bytes encoded")
                && text.contains("4-byte image limit"),
            "the refusal names the address, the raw size, the encoded size, and the limit: {text}"
        );
        let clamped = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("shot.png", &[1, 2, 3, 4]),
        )
        .with_max_image_bytes(0);
        let output = clamped
            .call(input("shot.png"), &ToolContext::default())
            .await
            .expect("a zero image ceiling clamps to one byte");
        assert!(
            output.payload.to_string().contains("1-byte image limit"),
            "the clamped ceiling refuses by its configured value: {:?}",
            output.payload
        );
    }

    #[tokio::test]
    async fn binary_content_is_refused_with_a_soft_error_naming_the_address() {
        let tool = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("blob.bin", &[0, 255, 128]),
        );
        let output = tool
            .call(input("blob.bin"), &ToolContext::default())
            .await
            .expect("a binary entry resolves to a refusal, not an error");
        assert!(
            output.is_error,
            "the refusal is a soft error the model can read: {output:?}"
        );
        let text = output.payload.to_string();
        assert!(
            text.contains("blob.bin") && text.contains("binary") && text.contains("3 bytes"),
            "the refusal names the address, the kind, and the size: {text}"
        );
    }

    #[tokio::test]
    async fn a_no_fit_view_mid_content_still_opens_with_the_omission_header() {
        let content = format!("short\n{}\ntail", "x".repeat(1000));
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)])).with_max_bytes(100);
        let output = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 2 })),
        )
        .await
        .expect("a mid-content no-fit view resolves");
        assert!(
            output.starts_with("[Lines before offset 2 omitted"),
            "every view starting past line 1 opens with the omission header: {output}"
        );
        assert!(
            output.contains("line 2 alone does not fit the 100-byte output limit")
                && output.contains("Use offset=3 to read past it"),
            "the header carries the same advice the top-of-file view would: {output}"
        );
    }

    #[tokio::test]
    async fn an_oversized_address_is_refused_before_reading_when_size_is_known() {
        let source = FakeSource::with(&[("huge.txt", TEN_LINES)])
            .reporting_size("huge.txt", 11 * 1024 * 1024);
        let tool = ReadTool::new(source);
        let output = tool
            .call(input("huge.txt"), &ToolContext::default())
            .await
            .expect("an oversized address resolves to a refusal, not an error");
        assert_eq!(
            tool.source.read_count(),
            0,
            "the size probe must refuse before any read fires"
        );
        let text = output.payload.to_string();
        assert!(
            text.contains("huge.txt")
                && text.contains("over the")
                && text.contains("-byte read limit"),
            "the refusal names the address and both numbers: {text}"
        );
        assert!(
            text.contains("The limit applies to the whole content, not the requested window"),
            "the refusal is terminal: no ranged-view advice that would re-refuse: {text}"
        );
    }

    #[tokio::test]
    async fn an_unknown_size_reads_as_before() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]));
        let output = read(&tool, input("doc"))
            .await
            .expect("a source with no size probe reads as before");
        assert!(
            output.contains("one") && output.contains("ten"),
            "the default size probe never refuses: {output}"
        );
    }

    #[tokio::test]
    async fn with_max_size_bytes_overrides_the_refusal_threshold() {
        let source = FakeSource::with(&[("doc", TEN_LINES)]).reporting_size("doc", 500);
        let tool = ReadTool::new(source).with_max_size_bytes(100);
        let output = read(&tool, input("doc"))
            .await
            .expect("a lowered threshold refuses the same content it would admit by default");
        assert_eq!(
            tool.source.read_count(),
            0,
            "the overridden threshold still refuses before any read fires"
        );
        assert!(
            output.contains("100-byte read limit"),
            "the refusal names the configured threshold, not the default: {output}"
        );
    }

    #[tokio::test]
    async fn builder_overrides_take_effect() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]))
            .with_max_lines(2)
            .with_default_offset_limit(1);
        let default_view = read(&tool, input("doc"))
            .await
            .expect("the default view resolves");
        assert!(
            default_view.contains("Showing lines 1-2 of 10"),
            "with_max_lines drives the default window: {default_view}"
        );
        let offset_only = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 3 })),
        )
        .await
        .expect("the offset-only view resolves");
        assert!(
            offset_only.contains("Showing lines 3-3 of 10"),
            "with_default_offset_limit drives the offset-only window: {offset_only}"
        );
        assert!(
            offset_only.contains("Use offset=4 to see the remaining 7 lines"),
            "a cut tail names the continuation offset: {offset_only}"
        );
    }

    #[test]
    fn schema_advertises_the_documented_fields() {
        let schema = ReadTool::new(FakeSource::with(&[])).schema();
        assert_eq!(schema.tool, "read", "the registered name is lowercase read");
        let properties = schema
            .input_schema
            .get("properties")
            .and_then(Value::as_object)
            .expect("the schema carries properties");
        for field in ["path", "offset", "limit", "line_range"] {
            assert!(
                properties.contains_key(field),
                "the schema advertises {field}: {properties:?}"
            );
        }
        let required = schema
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .expect("the schema carries required fields");
        assert!(
            required.len() == 1 && required[0] == "path",
            "only path is required: {required:?}"
        );
    }

    #[tokio::test]
    async fn the_default_ceilings_are_the_documented_ones() {
        let content = (1..=300)
            .map(|n| format!("line{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tool = ReadTool::new(FakeSource::with(&[("doc", &content)]));
        let head = read(&tool, input("doc"))
            .await
            .expect("the default window resolves");
        assert!(
            head.contains("Showing lines 1-200 of 300"),
            "the default line ceiling is 200: {head}"
        );
        let page = read(
            &tool,
            with_range("doc", &serde_json::json!({ "offset": 6 })),
        )
        .await
        .expect("the default offset-only window resolves");
        assert!(
            page.contains("Showing lines 6-205 of 300"),
            "the default offset-only window is 200 lines: {page}"
        );
        let wide_line = "x".repeat(3000);
        let wide = (0..300)
            .map(|_| wide_line.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let wide_tool = ReadTool::new(FakeSource::with(&[("wide", &wide)]));
        let output = read(&wide_tool, input("wide"))
            .await
            .expect("the default byte cap resolves");
        assert!(
            output.contains("400000-byte output limit"),
            "the default byte ceiling is 400 000: {output}"
        );
    }

    #[tokio::test]
    async fn the_default_image_ceiling_is_five_mebibytes_encoded() {
        let exact = "a".repeat(3_932_160);
        let tool = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("ok.png", exact.as_bytes()),
        );
        let output = tool
            .call(input("ok.png"), &ToolContext::default())
            .await
            .expect("an image encoding to exactly 5 MiB resolves");
        assert!(
            !output.is_error,
            "the ceiling refuses only beyond the limit, not at it: {output:?}"
        );
        let over = "a".repeat(3_932_161);
        let tool = ReadTool::new(
            FakeSource::with(&[("doc", "irrelevant")]).with_bytes("big.png", over.as_bytes()),
        );
        let output = tool
            .call(input("big.png"), &ToolContext::default())
            .await
            .expect("one raw byte over resolves to a refusal");
        let text = output.payload.to_string();
        assert!(
            output.is_error
                && text.contains("5242884 bytes encoded")
                && text.contains("5242880-byte image limit"),
            "the default image ceiling is 5 MiB encoded, named exactly: {text}"
        );
    }

    #[tokio::test]
    async fn image_kind_selection_covers_the_documented_extension_set() {
        for (name, mime) in [
            ("shot.png", "image/png"),
            ("shot.jpg", "image/jpeg"),
            ("shot.jpeg", "image/jpeg"),
            ("shot.gif", "image/gif"),
            ("shot.webp", "image/webp"),
            ("shot.PNG", "image/png"),
        ] {
            let tool = ReadTool::new(
                FakeSource::with(&[("doc", "irrelevant")]).with_bytes(name, &[9, 9, 9]),
            );
            let output = tool
                .call(input(name), &ToolContext::default())
                .await
                .unwrap_or_else(|error| panic!("{name} resolves, got: {error}"));
            let crate::message::ToolContent::Multipart(parts) = &output.payload else {
                panic!("{name} is served as multipart, got: {:?}", output.payload)
            };
            assert!(
                matches!(&parts[0], crate::message::ToolContentPart::Image { source }
                    if source.media_type == mime),
                "{name} selects {mime}: {:?}",
                parts[0]
            );
        }
    }

    #[tokio::test]
    async fn a_byte_cap_cut_inside_a_multibyte_character_is_honest_not_panicking() {
        let tool = ReadTool::new(FakeSource::with(&[("doc", "αααα\nββββ")])).with_max_bytes(4);
        let output = read(&tool, input("doc"))
            .await
            .expect("the multibyte cut resolves without panicking");
        assert!(
            output.contains("no line from 1 to 2 fits the 4-byte output limit"),
            "the cut walks back to a char boundary before any complete line: {output}"
        );
    }

    #[test]
    fn flags_are_true() {
        let tool = ReadTool::new(FakeSource::with(&[]));
        assert!(tool.is_read_only(), "reading has no effect to guard");
        assert!(
            tool.is_concurrency_safe(),
            "reads share no mutable state through the tool"
        );
    }

    #[tokio::test]
    async fn empty_text_content_reports_empty_rather_than_a_beyond_length_offset() {
        let tool = ReadTool::new(FakeSource::with(&[("empty", "")]));
        let output = read(&tool, input("empty"))
            .await
            .expect("an empty document resolves");
        assert_eq!(
            output, "Content is empty (0 lines)",
            "a rangeless read of empty content is empty, not beyond-length: {output}"
        );
    }

    #[tokio::test]
    async fn zero_ceiling_overrides_are_clamped_to_one() {
        let lines_tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)])).with_max_lines(0);
        let output = read(&lines_tool, input("doc"))
            .await
            .expect("a zero line ceiling clamps to a one-line window");
        assert!(
            output.contains("Showing lines 1-1 of 10") && output.contains("Use offset=2"),
            "the clamped ceiling frames one line honestly: {output}"
        );
        let offset_tool =
            ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)])).with_default_offset_limit(0);
        let output = read(
            &offset_tool,
            with_range("doc", &serde_json::json!({ "offset": 3 })),
        )
        .await
        .expect("a zero offset-only window clamps to one line");
        assert!(
            output.contains("Showing lines 3-3 of 10"),
            "the clamped page size frames one line honestly: {output}"
        );
        let bytes_tool = ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)])).with_max_bytes(0);
        let output = read(&bytes_tool, input("doc"))
            .await
            .expect("a zero byte ceiling clamps to one byte, which holds no line");
        assert!(
            output.contains("no line from 1 to 10 fits"),
            "the clamped byte ceiling refuses rather than framing an empty view: {output}"
        );
        let size_tool =
            ReadTool::new(FakeSource::with(&[("doc", TEN_LINES)]).reporting_size("doc", 500))
                .with_max_size_bytes(0);
        let output = read(&size_tool, input("doc"))
            .await
            .expect("a zero size threshold clamps to one byte");
        assert!(
            output.contains("1-byte read limit"),
            "the clamped threshold refuses by its configured value: {output}"
        );
        assert_eq!(
            size_tool.source.read_count(),
            0,
            "the clamped threshold still refuses before any read fires"
        );
    }
}
