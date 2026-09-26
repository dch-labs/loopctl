//! The `Edit` tool — replace a unique occurrence of text in a file.
//!
//! Edit reads a file, locates `old_text`, and requires it to appear
//! exactly once (non-overlapping). The replacement is run through the
//! validation gate and a staleness check before writing, and the result
//! is returned as a line diff preview.

use std::fmt;
use std::future::Future;
use std::ops::Range;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::AsyncReadExt;

use crate::tool::DisplayHint;
use crate::tool::Tool;
use crate::tool::ToolContext;
use crate::tool::ToolError;
use crate::tool::ToolOutput;
use crate::tool::ToolSchema;

use super::ContentValidator;
use super::FileSession;
use super::atomic;
use super::conflict;
use super::conflict::CheckFailure;
use super::diff::format_file_change;
use super::require_session;
use super::resolve;
use super::resolve::ResolvePolicy;
use super::state;
use super::write::format_validation_failure;

/// The single-occurrence edit tool over the session's workspace.
///
/// Replaces a unique occurrence of text in a file, runs the installed
/// [`ContentValidator`] on the result before writing, and returns a
/// line diff preview of the change. The model-facing name is `Edit`.
#[derive(Clone)]
pub struct EditTool {
    /// The installed syntax gate, if any.
    ///
    /// `None` — the default — performs no validation; see
    /// [`WriteTool`](super::WriteTool) for the gate's contract.
    validator: Option<Arc<dyn ContentValidator>>,
}

impl EditTool {
    /// Build an edit tool with no validation gate.
    ///
    /// The tool writes whatever content the edit produces; install a gate
    /// with `with_validator` when a host wants syntax checking before the
    /// write.
    #[must_use]
    pub fn new() -> Self {
        Self { validator: None }
    }

    /// Install a validation gate the tool consults before writing.
    ///
    /// Same contract as [`WriteTool::with_validator`](super::WriteTool::with_validator).
    #[must_use]
    pub fn with_validator(mut self, validator: Arc<dyn ContentValidator>) -> Self {
        self.validator = Some(validator);
        self
    }
}

impl fmt::Debug for EditTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EditTool")
            .field("validator", &self.validator.is_some())
            .finish()
    }
}

impl Default for EditTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "Edit"
    }

    fn description(&self) -> &'static str {
        "Edit a file by replacing text. Syntax validation is automatically \
         performed for supported file types."
    }

    fn system_prompt(&self) -> Option<String> {
        Some(
            "old_text must be unique in the file. For multiple changes, use \
             MultiEdit. Both run the linter after applying."
                .to_string(),
        )
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
                        "description": "The path to the file to edit.  May be absolute or relative; relative paths are resolved against the runner's working directory. The file must already exist — creating new files is the Write tool's job — and URLs are rejected."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The text to replace.  Must appear exactly once in the file: uniqueness is enforced, and a non-unique match returns a soft error asking the model to add surrounding context (or use `MultiEdit`) to disambiguate. Must be non-empty; include enough context to make the match unique."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text.  Written in place of `old_text` once the unique match is located. May be empty (a pure deletion) or longer than `old_text` (an insertion); the result is checked by the linter gate before the file is written."
                    },
                    "skip_linter": {
                        "type": "boolean",
                        "description": "Skip syntax validation (not recommended).  When `true`, the linter gate is bypassed and the file is written even if the resulting content has syntax errors. Defaults to `false`; the gate exists to prevent file corruption from malformed edits."
                    }
                },
                "required": ["file_path", "old_text", "new_text"]
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
            edit_inner(self, input, &session).await
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn is_concurrency_safe(&self) -> bool {
        false
    }
}

/// Body of [`Tool::call`].
///
/// Orchestrates parse → read → apply → gate → staleness check → write.
/// Recoverable conditions (text not found, ambiguous match, a
/// validator rejection, a file changed on disk since this call read
/// it) are surfaced as soft [`ToolOutput`] errors; hard failures (bad
/// args, missing file, I/O fault) become [`ToolError`]. A successful
/// write refreshes the path's recorded baseline, so the model's own
/// edit never registers as a later external change.
///
/// # Errors
///
/// Returns `ToolError::InvalidInput` for a missing field, an empty
/// `old_text`, or a URL `file_path`; `ToolError::FileNotFound` when
/// the target does not exist; and `ToolError::Execution` on a
/// genuine I/O fault.
async fn edit_inner(
    tool: &EditTool,
    input: Value,
    session: &FileSession,
) -> Result<ToolOutput, ToolError> {
    let policy = session.resolve_policy();
    let cwd = session.cwd().to_path_buf();
    let parsed = parse_input(&input)?;
    let mut full_path = resolve::resolve_path(parsed.file_path, &cwd, policy)?;
    if policy == ResolvePolicy::Unrestricted {
        full_path = resolve::canonicalize_existing(&full_path)?;
    }
    let old_content = read_existing(
        &full_path,
        parsed.file_path,
        &cwd,
        policy,
        Some(session.anchor()),
    )
    .await?;
    let new_content = match apply_edit(&old_content, parsed.old_text, parsed.new_text) {
        Ok(content) => content,
        Err(reason) => return Ok(reason.into_output()),
    };

    if !parsed.skip_linter
        && let Some(validator) = tool.validator.as_ref()
    {
        let diagnostics = validator.validate(&full_path, &new_content).await;
        if !diagnostics.is_empty() {
            return Ok(ToolOutput::error_text(format_validation_failure(
                &full_path,
                &diagnostics,
            )));
        }
    }

    let expected = match conflict::check_content_unchanged(&old_content, &full_path).await {
        Ok(identity) => Some(identity),
        Err(failure) => {
            return match failure {
                CheckFailure::Changed => Ok(EditError::Conflict.into_output()),
                CheckFailure::Fault(e) => Err(e),
            };
        }
    };

    atomic::atomic_write(
        &full_path,
        &new_content,
        &cwd,
        policy,
        expected.as_ref(),
        Some(session.anchor()),
    )?;

    session.record_baseline(&full_path, state::observe_bytes(new_content.as_bytes()));

    let message = format_file_change(parsed.file_path, Some(&old_content), &new_content);
    Ok(ToolOutput::text(message).with_hint(DisplayHint::Diff))
}

/// Parsed and validated Edit input.
///
/// Produced by [`parse_input`] from the raw JSON the model sends. All
/// string fields are borrowed from the input (`'a`) — no cloning until
/// the async body needs owned copies for file I/O. Validated up front:
/// `old_text` must be non-empty and `file_path` must not be a URL, both
/// checked before this struct is constructed.
#[derive(Debug)]
struct EditArgs<'a> {
    /// The file path exactly as supplied by the caller, before cwd resolution.
    ///
    /// Borrowed from the input JSON. Kept in its raw (pre-resolution) form so
    /// error messages and the diff preview show the path the model named, not
    /// the canonicalized absolute path.
    file_path: &'a str,

    /// The text to find in the file.
    ///
    /// Must be non-empty (rejected by [`parse_input`] before this struct is
    /// built) and must appear exactly once in the target file (enforced later
    /// by [`locate_unique`]).
    old_text: &'a str,

    /// The text to replace `old_text` with.
    ///
    /// May be empty (a pure deletion) or longer than `old_text` (an
    /// insertion). Spliced into the file content by [`splice`] after the
    /// unique match is located.
    new_text: &'a str,

    /// Whether to skip the validation gate on the edited result.
    ///
    /// When `true`, the gate is bypassed entirely — the file is written even
    /// if the installed validator rejects the resulting content. Defaults
    /// to `false`; the gate exists to prevent file corruption from malformed
    /// edits.
    skip_linter: bool,
}

/// A recoverable reason an edit was not applied.
///
/// These are *soft* failure modes: the caller surfaces them to the loop as
/// a [`ToolOutput::error_text`] so the model can correct its arguments and
/// retry, rather than as a hard [`ToolError`] (which signals an
/// unrecoverable fault like a missing file or an I/O error).
/// [`EditError::into_output`] is the single place the structured reason is
/// formatted into a human-readable message for the loop.
#[derive(Debug, PartialEq, Eq)]
enum EditError {
    /// `old_text` does not appear anywhere in the file.
    ///
    /// Produced by [`apply_edit`] when [`locate_unique`] returns
    /// [`FindResult::NotFound`]. The formatted message includes hints
    /// (re-read the file, check for whitespace/Unicode differences) so the
    /// model has a clear recovery path.
    NotFound,

    /// `old_text` appears more than once in the file.
    ///
    /// Edit requires `old_text` to be unique so it can splice without
    /// ambiguity. Produced by [`apply_edit`] when [`locate_unique`] returns
    /// [`FindResult::Ambiguous`]. The formatted message names the count and
    /// suggests adding surrounding context or using `MultiEdit` (which also
    /// enforces uniqueness per edit but lets the model batch disambiguated
    /// edits).
    Ambiguous {
        /// The non-overlapping occurrence count.
        ///
        /// Always greater than 1 (zero occurrences would be
        /// [`NotFound`](Self::NotFound) instead).
        count: usize,
    },

    /// The file changed on disk between this call's read and its write.
    ///
    /// Detected by the detect-on-write check immediately before the write:
    /// an external writer modified the file after this call read it.
    /// Writing would clobber the newer content, so the edit is refused and
    /// the model re-reads and retries.
    Conflict,
}

impl EditError {
    /// Format this reason as the soft [`ToolOutput`] returned to the loop.
    ///
    /// Each variant produces a human-readable error message with
    /// actionable hints so the model can correct its input and retry.
    /// This is the single formatting site — the structured enum is carried
    /// through the pipeline and only stringified here, at the boundary.
    fn into_output(self) -> ToolOutput {
        match self {
            EditError::NotFound => ToolOutput::error_text(
                "Old text not found in file.\n\n\
                 Hints:\n  \
                 - The file may have changed since you last read it — try re-reading with `Read`\n  \
                 - Check for whitespace or Unicode differences\n  \
                 - Use `Grep` to search for the text you want to replace",
            ),
            EditError::Ambiguous { count } => ToolOutput::error_text(format!(
                "old_text appears {count} times in the file; it must be unique. \
                 Add surrounding context to disambiguate, or use MultiEdit."
            )),
            EditError::Conflict => ToolOutput::error_text(
                "File changed on disk since this call read it; not writing to \
                 avoid clobbering the newer content.\n\n\
                 Hints:\n  \
                 - Re-read the file with `Read`\n  \
                 - Re-issue the edit against the current content",
            ),
        }
    }
}

/// Extract the Edit arguments from the JSON `input` and validate them.
///
/// `file_path`, `old_text`, and `new_text` must be present strings;
/// `old_text` must be non-empty; `file_path` must not be a URL.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a missing field, an empty
/// `old_text`, or a URL `file_path`.
fn parse_input(input: &Value) -> Result<EditArgs<'_>, ToolError> {
    let file_path = input
        .get("file_path")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing file_path".to_string()))?;
    let old_text = input
        .get("old_text")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing old_text".to_string()))?;
    let new_text = input
        .get("new_text")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing new_text".to_string()))?;
    let skip_linter = input
        .get("skip_linter")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if old_text.is_empty() {
        return Err(ToolError::InvalidInput(
            "old_text must not be empty".to_string(),
        ));
    }

    resolve::reject_path_url("Edit", file_path)?;
    Ok(EditArgs {
        file_path,
        old_text,
        new_text,
        skip_linter,
    })
}

/// Read an existing file's full contents as UTF-8.
///
/// Distinguishes a missing file ([`ToolError::FileNotFound`]) from a
/// genuine I/O fault ([`ToolError::Execution`]). The `display_path` is
/// used verbatim in the not-found error so the caller sees the path it
/// supplied. Under the contained policy the opened handle is verified
/// against the session's pinned workspace anchor, so a symlink swapped
/// onto the workspace spelling after the session was constructed cannot
/// serve an edit's old content from outside the pinned workspace.
///
/// # Errors
///
/// Returns [`ToolError::FileNotFound`] when the file does not exist,
/// [`ToolError::Execution`] on any other I/O error (including non-UTF-8
/// reads), and when the contained handle check fails.
async fn read_existing(
    full_path: &Path,
    display_path: &str,
    workspace: &Path,
    policy: ResolvePolicy,
    anchor: Option<&super::atomic::WorkspaceAnchor>,
) -> Result<String, ToolError> {
    if !tokio::fs::try_exists(full_path)
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?
    {
        return Err(ToolError::FileNotFound(display_path.to_string()));
    }
    let mut file = tokio::fs::File::open(full_path)
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    if policy == ResolvePolicy::Contained {
        resolve::verify_handle_inside(&file, workspace, anchor)?;
    }
    let mut content = String::new();
    file.read_to_string(&mut content)
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    Ok(content)
}

/// Locate `old_text` in `content` and splice `new_text` into its place.
///
/// On success returns the new content. The result is structured; the
/// caller formats the error for the loop via [`EditError::into_output`].
///
/// # Errors
///
/// Returns `Err(EditError::NotFound)` when `old_text` is absent and
/// `Err(EditError::Ambiguous)` when it appears more than once.
fn apply_edit(content: &str, old_text: &str, new_text: &str) -> Result<String, EditError> {
    match locate_unique(content, old_text) {
        FindResult::NotFound => Err(EditError::NotFound),
        FindResult::Ambiguous { count } => Err(EditError::Ambiguous { count }),
        FindResult::Unique(range) => Ok(splice(content, range, new_text)),
    }
}

/// Outcome of locating `old_text` within the file content.
///
/// Produced by [`locate_unique`]. The three variants classify the search
/// result into the cases the `Edit` and `MultiEdit` pipelines need to
/// distinguish: exactly one match (safe to splice), zero matches
/// (recoverable: the model should re-read), or more than one
/// (recoverable: the model should add disambiguating context).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FindResult {
    /// Exactly one non-overlapping occurrence of `old_text` in `content`.
    ///
    /// The payload is the byte range (`start..end`) of that single match,
    /// suitable for passing directly to [`splice`]. The range is guaranteed
    /// to fall on valid UTF-8 boundaries because it was derived from
    /// `str::find`.
    Unique(Range<usize>),

    /// `old_text` does not appear anywhere in `content`.
    ///
    /// The model likely has stale knowledge of the file (it changed since
    /// the last read) or supplied the wrong text. The recovery path is to
    /// re-read the file and retry. Maps to [`EditError::NotFound`].
    NotFound,

    /// `old_text` appears more than once in `content`.
    ///
    /// The model needs to supply a longer, more specific `old_text` that
    /// matches only the intended site, or issue several `MultiEdit` edits,
    /// each with its own unique `old_text`. Maps to
    /// [`EditError::Ambiguous`].
    Ambiguous {
        /// The non-overlapping occurrence count.
        ///
        /// Always greater than 1 (zero occurrences yields
        /// [`NotFound`](Self::NotFound) instead). Included in the error
        /// message so the model knows how many sites it's dealing with.
        count: usize,
    },
}

/// Classify how many non-overlapping times `old_text` occurs in `content`.
///
/// Uses `str::matches` (non-overlapping count) and `str::find` (first
/// position). Returns [`FindResult::Unique`] only for exactly one
/// occurrence, carrying the byte range to splice into.
pub(crate) fn locate_unique(content: &str, old_text: &str) -> FindResult {
    let Some(start) = content.find(old_text) else {
        return FindResult::NotFound;
    };
    let after_first = start.saturating_add(old_text.len());
    if content
        .get(after_first..)
        .is_some_and(|rest| rest.contains(old_text))
    {
        let count = content.matches(old_text).count();
        return FindResult::Ambiguous { count };
    }
    let end = start.saturating_add(old_text.len());
    FindResult::Unique(start..end)
}

/// Splice `replacement` into `content`, replacing the byte `range`.
///
/// The `range` comes from [`locate_unique`], whose `str::find`-derived
/// offsets are char boundaries, so the slicing cannot split a code point;
/// the `get` fallbacks cover only unreachable non-boundary input.
pub(crate) fn splice(content: &str, range: Range<usize>, replacement: &str) -> String {
    let prefix = content.get(..range.start).unwrap_or("");
    let suffix = content.get(range.end..).unwrap_or("");
    let cap = prefix
        .len()
        .saturating_add(replacement.len())
        .saturating_add(suffix.len());
    let mut result = String::with_capacity(cap);
    result.push_str(prefix);
    result.push_str(replacement);
    result.push_str(suffix);
    result
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default
)]
mod tests {
    use super::super::ValidationDiagnostic;
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        super::super::FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
        ctx
    }

    struct RejectBad;

    impl ContentValidator for RejectBad {
        fn validate<'a>(
            &'a self,
            _path: &'a Path,
            content: &'a str,
        ) -> Pin<Box<dyn Future<Output = Vec<ValidationDiagnostic>> + Send + 'a>> {
            Box::pin(async move {
                if content.contains("BAD") {
                    vec![ValidationDiagnostic {
                        line: Some(2),
                        message: "content contains BAD".to_string(),
                    }]
                } else {
                    Vec::new()
                }
            })
        }
    }

    #[tokio::test]
    async fn replaces_a_unique_occurrence_with_a_diff_preview() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "fn one() {}\nfn two() {}\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "fn two() {}", "new_text": "fn two() { todo!() }"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let text = out.text_content();
        assert!(text.contains("Changed: a.rs (modified)"), "{text}");
        assert!(text.contains("- fn two() {}"), "{text}");
        assert!(text.contains("+ fn two() { todo!() }"), "{text}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "fn one() {}\nfn two() { todo!() }\n"
        );
    }

    #[tokio::test]
    async fn absent_text_is_a_soft_error_with_hints() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "content\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "absent", "new_text": "x"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content().contains("Old text not found"),
            "{}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "content\n",
            "a refused edit must not write"
        );
    }

    #[tokio::test]
    async fn ambiguous_text_names_the_count() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "x\nx\nx\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "x", "new_text": "y"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content().contains("appears 3 times"),
            "{}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn empty_old_text_is_invalid_input() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "x\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let err = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "", "new_text": "y"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn missing_file_is_file_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let err = EditTool::new()
            .call(
                json!({"file_path": "absent.rs", "old_text": "x", "new_text": "y"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::FileNotFound(_)));
    }

    #[tokio::test]
    async fn url_file_path_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let err = EditTool::new()
            .call(
                json!({"file_path": "https://example.com/x", "old_text": "x", "new_text": "y"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn a_failing_validator_blocks_the_edit() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "ok\nBAD\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = EditTool::new().with_validator(Arc::new(RejectBad));
        let out = tool
            .call(
                json!({"file_path": "a.rs", "old_text": "ok", "new_text": "ok BAD"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content().contains("Syntax validation failed"),
            "{}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "ok\nBAD\n",
            "a blocked edit must not write"
        );
    }

    #[tokio::test]
    async fn skip_linter_bypasses_the_gate() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "ok\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = EditTool::new().with_validator(Arc::new(RejectBad));
        let out = tool
            .call(
                json!({"file_path": "a.rs", "old_text": "ok", "new_text": "BAD", "skip_linter": true}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "BAD\n"
        );
    }

    #[tokio::test]
    async fn an_external_change_during_the_call_refuses_the_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "v1\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = FileSession::new(tmp.path().to_path_buf());
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&tmp.path().join("a.rs"), state::observe_bytes(b"v1\n"));
        std::fs::write(tmp.path().join("a.rs"), "EXTERNAL\n").unwrap();

        let out = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "v1", "new_text": "v2"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            out.is_error,
            "the staleness compare sees the newer bytes: {}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "EXTERNAL\n"
        );
    }

    #[tokio::test]
    async fn the_models_own_edit_never_conflicts() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "a\nb\nc\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let ctx = ctx_in(cwd);
        let first = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "a", "new_text": "A"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!first.is_error);
        let second = EditTool::new()
            .call(
                json!({"file_path": "a.rs", "old_text": "b", "new_text": "B"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!second.is_error);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "A\nB\nc\n"
        );
    }

    #[test]
    fn locate_unique_classifies_the_three_outcomes() {
        assert_eq!(locate_unique("abc", "b"), FindResult::Unique(1..2));
        assert_eq!(locate_unique("abc", "z"), FindResult::NotFound);
        assert_eq!(
            locate_unique("xbx", "x"),
            FindResult::Ambiguous { count: 2 }
        );
    }

    #[test]
    fn splice_replaces_only_the_matched_range() {
        assert_eq!(splice("hello world", 6..11, "there"), "hello there");
        assert_eq!(splice("abc", 1..2, ""), "ac");
    }

    #[tokio::test]
    async fn flags_advertise_write_semantics() {
        let tool = EditTool::new();
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }
}
