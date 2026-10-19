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

use crate::tool::DisplayHint;
use crate::tool::Retention;
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
use super::diff::OldContent;
use super::diff::format_file_change;
use super::existing;
use super::existing::ExistingText;
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
/// args, missing file, I/O fault) become [`ToolError`]. A baseline
/// whose only observation is resume-armed holds the edit — the same
/// refusal Write applies — until a live read re-records the bytes. A
/// successful write refreshes the path's recorded baseline, so the
/// model's own edit never registers as a later external change.
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
    let mut full_path = session.resolve(parsed.file_path)?;
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
    if session
        .baseline_for(&full_path)
        .is_some_and(|baseline| baseline.resumed)
    {
        return Ok(ToolOutput::error_text(conflict::resumed_baseline_message(
            &full_path,
        )));
    }
    let mut normalized_note = false;
    let new_content = match apply_edit(&old_content, parsed.old_text, parsed.new_text) {
        Ok(content) => content,
        Err(EditError::NotFound) => {
            match super::match_repair::classify_miss(&old_content, parsed.old_text) {
                super::match_repair::MissKind::WhitespaceUnique(window) => {
                    normalized_note = true;
                    super::match_repair::normalized_splice(
                        &old_content,
                        window,
                        parsed.old_text,
                        parsed.new_text,
                    )
                }
                kind => {
                    return Ok(ToolOutput::error_text(format!(
                        "Old text not found in file.\n\n{}\n{}",
                        super::match_repair::render_miss(&kind, &old_content),
                        "Hints:\n  - The file may have changed since you last read it — try re-reading with `Read`\n  - Check for whitespace or Unicode differences\n  - Use `Grep` to search for the text you want to replace"
                    )));
                }
            }
        }
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

    let mut message = format_file_change(
        parsed.file_path,
        OldContent::Text(&old_content),
        &new_content,
    );
    if normalized_note {
        insert_normalized_note(&mut message);
    }
    Ok(ToolOutput::text(message)
        .with_hint(DisplayHint::Diff)
        .with_retention(Retention::Durable))
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
    let file_path =
        super::resolve_path_key(input).ok_or_else(|| super::missing_path_error("Edit", input))?;
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
/// Returns [`ToolError::FileNotFound`] when the file does not exist and
/// [`ToolError::Execution`] when the content is not valid UTF-8, the
/// target is not a regular file (a directory included), its size
/// exceeds the family read cap, the contained handle check fails, or
/// any other I/O error occurs.
async fn read_existing(
    full_path: &Path,
    display_path: &str,
    workspace: &Path,
    policy: ResolvePolicy,
    anchor: Option<&super::atomic::WorkspaceAnchor>,
) -> Result<String, ToolError> {
    match existing::read_existing_text(full_path, workspace, policy, anchor).await? {
        ExistingText::Text(content) => Ok(content),
        ExistingText::NotUtf8 => Err(ToolError::Execution(format!(
            "{display_path} is not valid UTF-8"
        ))),
        ExistingText::Absent => Err(ToolError::FileNotFound(display_path.to_string())),
        ExistingText::TooLarge => Err(ToolError::Execution(format!(
            "{display_path} exceeds the {}-byte size guard for reading existing content",
            existing::MAX_EXISTING_READ_BYTES
        ))),
        ExistingText::NonRegular | ExistingText::Directory => Err(ToolError::Execution(format!(
            "{display_path} is not a regular file; refusing to read it for editing"
        ))),
    }
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
/// Insert the normalized-match disclosure after the summary line.
///
/// The receipt's `│`-note channel sits between the `Changed:` header
/// and the `@@` hunks; the disclosure rides exactly there, so the
/// model sees *why* the edit applied without hunting the diff.
fn insert_normalized_note(receipt: &mut String) {
    let note = format!("│ {}\n", super::match_repair::NORMALIZED_MATCH_NOTE);
    if let Some(position) = receipt.find("\n@@") {
        receipt.insert_str(position.saturating_add(1), &note);
    } else {
        if !receipt.ends_with('\n') {
            receipt.push('\n');
        }
        receipt.push_str(&note);
    }
}

/// Apply one edit by exact-match locate-and-splice.
///
/// # Errors
///
/// [`EditError::NotFound`] when `old_text` is absent and
/// [`EditError::Ambiguous`] when it appears more than once.
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
    async fn a_whitespace_mismatched_old_text_applies_and_discloses() {
        // The corpus's whitespace root cause: the text is right but
        // for indentation width, trailing spaces, or line endings.
        // A unique normalized match applies - splicing with the
        // file's actual bytes - and the receipt says so.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.rs"),
            "fn one() {}\r\n    fn two() {}  \r\nfn three() {}\r\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "      fn two() {}",
            "new_text": "fn two() { todo!() }",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        assert!(
            out.text_content()
                .contains("matched with normalized indentation"),
            "the receipt discloses the normalization: {}",
            out.text_content()
        );
        let on_disk = std::fs::read_to_string(dir.path().join("a.rs")).expect("read back");
        assert!(on_disk.contains("fn two() { todo!() }"), "the edit landed");
        assert!(
            on_disk.starts_with("fn one() {}\r\n    "),
            "the file's own indentation and line endings survive the splice: {on_disk:?}"
        );
    }

    #[tokio::test]
    async fn the_tolerant_apply_gives_the_file_s_indentation_the_first_line() {
        // The corpus shape itself: the model that mismatched the
        // indent spells BOTH texts with its own (wrong) indent. The
        // splice keeps the file's edge bytes and trims the
        // replacement's own edge whitespace symmetric to the core's
        // shrink - never the concatenated ten-space first line.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.rs"),
            "fn one() {}\n    fn two() {}\nfn three() {}\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "      fn two() {}",
            "new_text": "      fn two() { todo!() }",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        assert!(
            out.text_content()
                .contains("matched with normalized indentation"),
            "the disclosure rides: {}",
            out.text_content()
        );
        let on_disk = std::fs::read_to_string(dir.path().join("a.rs")).expect("read back");
        assert!(
            on_disk
                .lines()
                .any(|line| line == "    fn two() { todo!() }"),
            "the file's 4-space indent owns the first line, no doubling: {on_disk:?}"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_re_indents_the_whole_replacement() {
        // The review's Python shape: the model matched with shallower
        // indentation, and the replacement's interior lines keep that
        // shallow spelling - applied verbatim they would produce an
        // IndentationError. The whole block shifts by the window's
        // indentation delta; the file's absolute indent wins.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.py"),
            "def outer():\n    def f(self):\n        return 1\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.py",
            "old_text": "def f(self):\n    return 1",
            "new_text": "def f(self):\n    return 2",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        assert!(
            out.text_content()
                .contains("matched with normalized indentation"),
            "the disclosure rides: {}",
            out.text_content()
        );
        let on_disk = std::fs::read_to_string(dir.path().join("a.py")).expect("read back");
        assert!(
            on_disk.lines().any(|line| line == "    def f(self):")
                && on_disk.lines().any(|line| line == "        return 2"),
            "every replacement line carries the file's indent - no IndentationError: {on_disk:?}"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_dedents_when_the_model_over_indented() {
        // The negative-delta twin: the model's spelling is deeper
        // than the file's; the block shifts back by the same delta.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("a.rs"), "mod m {\nfn inner() {}\n}\n")
            .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "        fn inner() {}",
            "new_text": "        fn inner() { todo!() }",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        let _ = out;
        let on_disk = std::fs::read_to_string(dir.path().join("a.rs")).expect("read back");
        assert!(
            on_disk.lines().any(|line| line == "fn inner() { todo!() }"),
            "the over-indent shifts back to the file's level: {on_disk:?}"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_preserves_the_file_s_tab_indentation() {
        // The file's indent characters win: a tab-indented window
        // with space-indented model texts must not splice the model's
        // spaces back in - the written bytes match the file's own
        // tabs exactly.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.py"),
            "def outer():\n\tdef f():\n\t\treturn 1\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.py",
            "old_text": "def f():\n    return 1",
            "new_text": "def f():\n    return 2",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        assert!(
            out.text_content()
                .contains("matched with normalized indentation"),
            "the disclosure rides: {}",
            out.text_content()
        );
        let on_disk = std::fs::read_to_string(dir.path().join("a.py")).expect("read back");
        assert_eq!(
            on_disk, "def outer():\n\tdef f():\n\t\treturn 2\n",
            "the file's tab characters and widths own every line"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_preserves_makefile_recipe_tabs() {
        // The delta-zero class: the first lines match at width zero
        // while the characters differ - only a per-line rewrite in
        // the file's character keeps the recipe tab (make: missing
        // separator otherwise).
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("Makefile"), "all:\n\techo a\n").expect("write fixture");
        let input = json!({
            "file_path": "Makefile",
            "old_text": "all:\n    echo a",
            "new_text": "all:\n    echo b",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        let _ = out;
        let on_disk = std::fs::read_to_string(dir.path().join("Makefile")).expect("read back");
        assert_eq!(
            on_disk, "all:\n\techo b\n",
            "the recipe line keeps the file's tab at delta zero"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_dedents_in_the_file_s_characters() {
        // The dedent twin in the file's own characters: the model's
        // deeper space indent shifts back to the file's shallower
        // tabs.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("a.py"), "def outer():\n\treturn 1\n")
            .expect("write fixture");
        let input = json!({
            "file_path": "a.py",
            "old_text": "        return 1",
            "new_text": "        return 2",
        });
        let _ = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        let on_disk = std::fs::read_to_string(dir.path().join("a.py")).expect("read back");
        assert_eq!(
            on_disk, "def outer():\n\treturn 2\n",
            "the over-indent shifts back in the file's tab"
        );
    }

    #[tokio::test]
    async fn a_tolerant_apply_joins_lines_with_the_file_s_endings() {
        // A multi-line replacement in a CRLF file joins with the
        // file's CRLF, never LF-only interior breaks.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.rs"),
            "fn one() {}\r\n    fn two() {}\r\nfn three() {}\r\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "      fn two() {}",
            "new_text": "      fn two() { todo!() }\n      // replaced",
        });
        let _ = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        let on_disk = std::fs::read_to_string(dir.path().join("a.rs")).expect("read back");
        assert_eq!(
            on_disk,
            "fn one() {}\r\n    fn two() { todo!() }\r\n    // replaced\r\nfn three() {}\r\n",
            "every interior join carries the file's CRLF"
        );
    }

    #[tokio::test]
    async fn an_ambiguous_normalized_match_rejects_with_the_echo() {
        // Tolerant matching never applies an ambiguous match: both
        // regions match after normalization, so the miss keeps
        // rejecting - now carrying the classification and the file's
        // actual lines with numbers.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.rs"),
            "  alpha \nbeta \n\talpha \nbeta \n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "alpha\nbeta",
            "new_text": "omega",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("soft error");
        let text = out.text_content();
        assert!(
            text.contains("Old text not found"),
            "the miss stays a miss: {text}"
        );
        assert!(
            text.contains("whitespace-only mismatch"),
            "the classification names the kind: {text}"
        );
        assert!(
            text.contains("2 regions"),
            "the ambiguity is explicit: {text}"
        );
        assert!(
            text.contains("1 |   alpha"),
            "the echo carries the file's actual lines, whitespace intact: {text}"
        );
        let on_disk = std::fs::read_to_string(dir.path().join("a.rs")).expect("read back");
        assert_eq!(
            on_disk, "  alpha \nbeta \n\talpha \nbeta \n",
            "nothing was written"
        );
    }

    #[tokio::test]
    async fn a_stale_miss_cites_the_first_differing_line() {
        // The stale root cause: the file changed since the model last
        // saw it. The classification names it and quotes both sides
        // of the first differing line.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("a.rs"),
            "fn one() {}\nfn two() { todo!() }\nfn three() {}\n",
        )
        .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "fn one() {}\nfn two() {}\nfn three() {}",
            "new_text": "replaced",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("soft error");
        let text = out.text_content();
        assert!(
            text.contains("stale"),
            "the classification names stale: {text}"
        );
        assert!(
            text.contains("line 2"),
            "the first differing line is cited: {text}"
        );
        assert!(
            text.contains("\"fn two() { todo!() }\"") && text.contains("\"fn two() {}\""),
            "both sides are quoted, whitespace visible: {text}"
        );
    }

    #[tokio::test]
    async fn a_miss_with_no_similar_passage_says_so() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("a.rs"), "fn one() {}\nfn two() {}\n")
            .expect("write fixture");
        let input = json!({
            "file_path": "a.rs",
            "old_text": "the function that makes tea",
            "new_text": "replaced",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("soft error");
        assert!(
            out.text_content().contains("no similar passage"),
            "the paraphrase shape is named: {}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn the_disclosure_survives_the_large_diff_path() {
        // Past MAX_LCS_PRODUCT the receipt carries no @@ hunk marker;
        // the disclosure must ride anyway - it is the only signal
        // that the model's text was not applied as sent.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let mut content = "filler line\n".repeat(1100);
        content.push_str("unique anchor line\n");
        content.push_str(&"filler line\n".repeat(100));
        std::fs::write(dir.path().join("big.rs"), &content).expect("write fixture");
        let input = json!({
            "file_path": "big.rs",
            "old_text": "      unique anchor line",
            "new_text": "      replaced anchor",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("the normalized match applies");
        assert!(
            out.text_content()
                .contains("matched with normalized indentation"),
            "the disclosure survives every receipt path: {}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn a_miss_on_a_huge_file_echoes_bounded() {
        // The echo replaces a re-read, not the file: a miss against a
        // two-hundred-line file carries a bounded window.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let filler = "filler line\n".repeat(200);
        std::fs::write(dir.path().join("big.rs"), &filler).expect("write fixture");
        let input = json!({
            "file_path": "big.rs",
            "old_text": "filler line\nand its successor",
            "new_text": "replaced",
        });
        let out = edit_inner(&EditTool::default(), input, &session)
            .await
            .expect("soft error");
        let text = out.text_content();
        let echoed = text.lines().filter(|line| line.contains(" | ")).count();
        assert!(
            echoed <= 17,
            "the echo stays bounded on a huge file: {echoed} lines"
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
    async fn a_resumed_baseline_holds_the_edit_until_a_live_read() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("held.rs");
        std::fs::write(&target, "v1\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = super::super::FileSession::new(PathBuf::from(cwd));
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&target, state::observe_resumed_bytes(b"v1\n"));

        let held = EditTool::new()
            .call(
                json!({"file_path": "held.rs", "old_text": "v1", "new_text": "v2"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(held.is_error, "a resume-armed baseline must hold the edit");
        assert!(
            held.text_content().contains("previous session"),
            "{}",
            held.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "v1\n",
            "nothing may be written while held"
        );

        session.record_baseline(&target, state::observe_bytes(b"v1\n"));
        let after = EditTool::new()
            .call(
                json!({"file_path": "held.rs", "old_text": "v1", "new_text": "v2"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            !after.is_error,
            "a live read must release the guard: {}",
            after.text_content()
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "v2\n");
    }

    #[tokio::test]
    async fn flags_advertise_write_semantics() {
        let tool = EditTool::new();
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }

    #[tokio::test]
    async fn an_edit_receipt_is_stamped_durable() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "x\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = EditTool::new();
        let out = tool
            .call(
                json!({"file_path": "a.rs", "old_text": "x", "new_text": "y"}),
                &ctx_in(cwd),
            )
            .await
            .expect("the edit succeeds");
        assert_eq!(
            out.retention,
            Some(crate::tool::Retention::Durable),
            "an edit receipt is a side-effect record — it must survive compaction verbatim"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_fifo_edit_target_refuses_without_hanging() {
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

        let err = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            EditTool::new().call(
                json!({"file_path": "pipe", "old_text": "a", "new_text": "b"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            ),
        )
        .await
        .expect("the gated read resolves instead of parking on the fifo")
        .unwrap_err();
        match err {
            ToolError::Execution(message) => assert!(
                message.contains("pipe is not a regular file"),
                "the refusal names the target and its kind: {message}"
            ),
            other => panic!("expected a refusal naming the fifo, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_oversized_edit_target_refuses_naming_the_guard() {
        let cap = crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES;
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("big.log");
        let oversized = vec![b'x'; usize::try_from(cap).unwrap() + 1];
        std::fs::write(&target, oversized).unwrap();

        let err = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            EditTool::new().call(
                json!({"file_path": "big.log", "old_text": "a", "new_text": "b"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            ),
        )
        .await
        .expect("the capped read resolves")
        .unwrap_err();
        match err {
            ToolError::Execution(message) => assert!(
                message.contains("big.log") && message.contains("size guard"),
                "the refusal names the target and the guard: {message}"
            ),
            other => panic!("expected a size-guard refusal, got {other:?}"),
        }
    }
    #[test]
    fn edit_refuses_a_mistyped_canonical_key() {
        let input = serde_json::json!({
            "file_path": 123,
            "path": "alias.txt",
            "old_text": "a",
            "new_text": "b",
        });
        let error = parse_input(&input).expect_err("the mistyped canonical refuses");
        assert!(
            error.to_string().contains("file_path must be a string"),
            "the type error rides: {error}"
        );
    }
    #[test]
    fn edit_accepts_path_where_file_path_was_expected() {
        let input = serde_json::json!({
            "path": "src/lib.rs",
            "old_text": "a",
            "new_text": "b",
        });
        let parsed = parse_input(&input).expect("the alias spelling parses");
        assert_eq!(parsed.file_path, "src/lib.rs");
    }

    #[test]
    fn edit_both_present_resolves_to_the_canonical_file_path() {
        let input = serde_json::json!({
            "path": "alias.txt",
            "file_path": "canonical.txt",
            "old_text": "a",
            "new_text": "b",
        });
        let parsed = parse_input(&input).expect("parses");
        assert_eq!(parsed.file_path, "canonical.txt");
    }
}
