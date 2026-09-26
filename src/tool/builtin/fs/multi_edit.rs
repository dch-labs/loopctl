//! The `MultiEdit` tool — apply a batch of text edits across one or more
//! files.
//!
//! Every edit in the batch is validated before any file is written. One
//! invalid edit (missing file, `old_text` not found or not unique, the
//! validator rejects the merged content, two edits overlap in the same
//! file) aborts the entire batch — no file is touched — as does a
//! symbolic-link target under the contained policy. Under the
//! unrestricted policy links are honored and writes land on the
//! referent. Under both policies, a batch that addresses one physical
//! file through multiple alias spellings — hard links included — is
//! refused: each alias would merge independently against the same
//! original and the writes would clobber one another. `dry_run: true`
//! previews diffs without writing.
//!
//! The atomicity guarantee covers *validation*: if any edit is invalid,
//! nothing is written. It does **not** cover crashes mid-write-batch —
//! see the "Atomicity scope" docs on the call body.

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use serde_json::json;
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
use super::conflict::CheckFailure;
use super::conflict::changed_message;
use super::conflict::check_content_unchanged;
use super::conflict::resumed_baseline_message;
use super::diff::OldContent;
use super::diff::format_file_change;
use super::edit::FindResult;
use super::edit::locate_unique;
use super::edit::splice;
use super::require_session;
use super::resolve;
use super::resolve::ResolvePolicy;
use super::state;
use super::write::format_validation_failure;

/// Maximum number of edits permitted in a single call.
///
/// Bounds the preview size, the validation pass, and the write loop; a
/// larger batch should be split so a failure surfaces before the whole
/// plan is built.
const MAX_EDITS: usize = 50;

/// The batch-edit tool over the session's workspace.
///
/// All edits are validated before any writes; use `dry_run=true` to
/// preview changes without writing. Not concurrency-safe and not
/// read-only: it mutates files, and two concurrent batches touching
/// overlapping paths would race. The model-facing name is `MultiEdit`.
#[derive(Clone)]
pub struct MultiEditTool {
    /// The installed syntax gate, if any.
    ///
    /// `None` — the default — performs no validation; see
    /// [`WriteTool`](super::WriteTool) for the gate's contract.
    validator: Option<Arc<dyn ContentValidator>>,
}

impl MultiEditTool {
    /// Build a batch-edit tool with no validation gate.
    ///
    /// Batches apply unvalidated; install a gate with `with_validator` when
    /// a host wants merged content checked before any file is written.
    #[must_use]
    pub fn new() -> Self {
        Self { validator: None }
    }

    /// Install a validation gate the tool consults before writing.
    ///
    /// Same contract as
    /// [`WriteTool::with_validator`](super::WriteTool::with_validator).
    #[must_use]
    pub fn with_validator(mut self, validator: Arc<dyn ContentValidator>) -> Self {
        self.validator = Some(validator);
        self
    }
}

impl fmt::Debug for MultiEditTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiEditTool")
            .field("validator", &self.validator.is_some())
            .finish()
    }
}

impl Default for MultiEditTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MultiEditTool {
    fn name(&self) -> &'static str {
        "MultiEdit"
    }

    fn description(&self) -> &'static str {
        "Edit multiple files atomically. All edits are validated before any \
         writes. Use dry_run=true to preview changes without writing."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "Array of edit operations to perform",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string", "description": "The path to the file to edit" },
                                "old_text": { "type": "string", "description": "The text to replace" },
                                "new_text": { "type": "string", "description": "The replacement text" }
                            },
                            "required": ["file_path", "old_text", "new_text"]
                        },
                        "minItems": 1,
                        "maxItems": 50
                    },
                    "dry_run": { "type": "boolean", "description": "Preview changes without writing files", "default": false },
                    "skip_linter": { "type": "boolean", "description": "Skip syntax validation", "default": false }
                },
                "required": ["edits"]
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
            multi_edit_inner(self, input, &session).await
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
/// Orchestrates the pipeline: parse → dup-path → alias → read → symlink →
/// resumed-hold → overlap-detect → locate+merge → validate → preview, then
/// a per-file staleness check immediately before each write. Recoverable
/// conditions (text not found, ambiguous match, overlap, symlink target,
/// a validator rejection, a file changed on disk since the batch read it,
/// a target whose only recorded observation is resume-armed) are surfaced
/// as soft [`ToolOutput`] errors; hard failures (bad args, missing file,
/// I/O fault) become [`ToolError`]. Each successful write refreshes the
/// path's recorded baseline.
///
/// # Atomicity scope
///
/// The all-or-nothing guarantee covers *validation*: by the time any
/// write happens, every edit has been validated and every file's merged
/// content has passed the gate. Writes are individual atomic file
/// replacements (temp-then-rename via
/// [`atomic_write`](super::atomic::atomic_write)), but the batch of
/// writes is **not** a single filesystem transaction — a process crash
/// mid-batch could leave some files written and others not. The partial
/// result is at worst *incomplete*, never syntactically corrupt,
/// because every file's content already passed the gate.
///
/// # Errors
///
/// Returns `ToolError::InvalidInput` for a missing `edits` array, an
/// empty array, more than `MAX_EDITS` edits, a missing field, an empty
/// `old_text`, or a URL `file_path`. Returns `ToolError::FileNotFound`
/// when a target does not exist. Returns `ToolError::Execution` on a
/// genuine I/O fault.
async fn multi_edit_inner(
    tool: &MultiEditTool,
    input: Value,
    session: &FileSession,
) -> Result<ToolOutput, ToolError> {
    let policy = session.resolve_policy();
    let cwd = session.cwd().to_path_buf();

    let parsed = parse_input(&input)?;
    let mut operations = build_operations(parsed.edits, &cwd, policy)?;

    if let Some(reason) = dup_path_check(&operations) {
        return Ok(reason.into_output());
    }
    if let Some(reason) = physical_alias_check(&operations) {
        return Ok(reason.into_output());
    }
    if policy == ResolvePolicy::Unrestricted {
        honor_symlink_targets(&mut operations)?;
    }
    let originals = read_files(&operations, session).await?;
    if let Some(held) = resumed_target(&operations, session) {
        return Ok(ToolOutput::error_text(resumed_baseline_message(held)));
    }
    if policy == ResolvePolicy::Contained
        && let Some(reason) = symlink_check(&operations, &cwd)
    {
        return Ok(reason.into_output());
    }
    if let Some(reason) = overlap_check(&operations, &originals) {
        return Ok(reason.into_output());
    }

    let finals = match merge_per_file(&operations, &originals) {
        Ok(f) => f,
        Err(reason) => return Ok(reason.into_output()),
    };
    if !parsed.skip_linter
        && let Some(validator) = tool.validator.as_ref()
        && let Some(reason) = validate_all(validator, &operations, &finals).await
    {
        return Ok(reason.into_output());
    }

    let summary = build_preview(&operations, &originals, &finals, parsed.dry_run);
    if parsed.dry_run {
        return Ok(ToolOutput::text(summary).with_hint(DisplayHint::Diff));
    }
    if let Some(conflict) = write_finals(&operations, &originals, &finals, session).await? {
        let mut message = changed_message(Path::new(&conflict.path));
        if !conflict.applied.is_empty() {
            message.push_str("\n\nAlready written by this batch: ");
            message.push_str(&conflict.applied.join(", "));
            message.push('.');
        }
        return Ok(ToolOutput::error_text(message));
    }

    let applied: Vec<&str> = finals.keys().map(String::as_str).collect();
    let message = apply_summary(&summary, &applied, &operations);
    Ok(ToolOutput::text(message).with_hint(DisplayHint::Diff))
}

/// Why and where a batch write stopped.
///
/// `applied` lists the caller-supplied paths successfully written before
/// the conflict, in write order — their new content is already on disk.
/// `path` names the file that changed, in the caller-supplied spelling.
#[derive(Debug)]
struct BatchConflict {
    /// Files written before the abort.
    ///
    /// Caller-supplied spellings, in write order — their new content is
    /// already on disk.
    applied: Vec<String>,

    /// The conflicted file.
    ///
    /// The caller-supplied spelling of the path whose staleness check
    /// failed.
    path: String,
}

/// Write each file's final content, staleness-checked per file.
///
/// Before each write, the file's current bytes are compared against the
/// content this batch read in phase 1 (`originals`); a file changed by
/// an external writer aborts the batch at that file — files already
/// written in earlier iterations stay written, and are reported in the
/// returned [`BatchConflict`]. Each successful write refreshes the
/// path's recorded baseline, so the model's own batch never registers
/// as a later external change.
///
/// Returns `Ok(Some(conflict))` when the batch aborted; `Ok(None)` when
/// every file was written.
///
/// # Errors
///
/// Returns [`ToolError`] when an atomic temp-then-rename write fails or
/// the conflict check hits a genuine I/O fault. A fault after earlier
/// writes have landed names them in the error message, so a partial
/// batch always reports what reached the disk.
async fn write_finals(
    operations: &[EditOperation],
    originals: &BTreeMap<String, String>,
    finals: &BTreeMap<String, String>,
    session: &FileSession,
) -> Result<Option<BatchConflict>, ToolError> {
    let workspace = session.cwd();
    let policy = session.resolve_policy();
    let anchor = session.anchor();
    let mut written: std::collections::HashSet<&Path> = std::collections::HashSet::new();
    let mut applied: Vec<String> = Vec::new();
    for op in operations {
        if !written.insert(&op.full_path) {
            continue;
        }
        if let Some(final_content) = finals.get(&op.file_path) {
            let mut expected = None;
            if let Some(baseline) = originals.get(&op.file_path).map(String::as_str) {
                match check_content_unchanged(baseline, &op.full_path).await {
                    Ok(identity) => expected = Some(identity),
                    Err(CheckFailure::Changed) => {
                        return Ok(Some(BatchConflict {
                            applied,
                            path: op.file_path.clone(),
                        }));
                    }
                    Err(CheckFailure::Fault(e)) => return Err(fault_with_applied(e, &applied)),
                }
            }
            atomic::atomic_write(
                &op.full_path,
                final_content,
                workspace,
                policy,
                expected.as_ref(),
                Some(anchor),
            )
            .map_err(|e| fault_with_applied(e, &applied))?;
            applied.push(op.file_path.clone());
            session.record_baseline(
                &op.full_path,
                state::observe_bytes(final_content.as_bytes()),
            );
        }
    }
    Ok(None)
}

/// Fold the files already written into a mid-batch fault's message.
///
/// Earlier writes of a batch stay on disk when a later file faults, so
/// the model must learn which files landed. When nothing has been
/// written yet the error is returned untouched.
fn fault_with_applied(error: ToolError, applied: &[String]) -> ToolError {
    if applied.is_empty() {
        return error;
    }
    ToolError::Execution(format!(
        "{error}\n\nAlready written by this batch: {}.",
        applied.join(", ")
    ))
}

/// Parsed top-level `MultiEdit` input.
///
/// Produced by [`parse_input`]. Carries the validated edits array plus
/// the two option flags, consumed by the rest of the pipeline.
#[derive(Debug)]
struct ParsedInput<'a> {
    /// The edits array, borrowed from the caller's input.
    ///
    /// Validated non-empty and within `MAX_EDITS` by [`parse_input`]
    /// before this struct is built; item-level field validation happens
    /// later in [`build_operations`].
    edits: &'a [Value],

    /// Whether to preview without writing.
    ///
    /// When `true`, the pipeline stops after building the diff preview
    /// and no file is touched; when `false` (the default), the writes
    /// run after validation.
    dry_run: bool,

    /// Whether to skip the validation gate on the merged content.
    ///
    /// Defaults to `false`. When `true`, [`validate_all`] is not
    /// consulted and the batch proceeds to write (or preview) without
    /// syntax validation.
    skip_linter: bool,
}

/// Extract the top-level `MultiEdit` arguments and bounds-check the edits
/// array.
///
/// `edits` must be a non-empty array of length ≤ `MAX_EDITS`; `dry_run`
/// and `skip_linter` default to `false`.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a missing `edits` array, an
/// empty array, or more than `MAX_EDITS` edits.
fn parse_input(input: &Value) -> Result<ParsedInput<'_>, ToolError> {
    let edits = input
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| ToolError::InvalidInput("Missing edits array".to_string()))?;
    if edits.is_empty() {
        return Err(ToolError::InvalidInput(
            "Edits array cannot be empty".to_string(),
        ));
    }
    if edits.len() > MAX_EDITS {
        return Err(ToolError::InvalidInput(format!(
            "Too many edits: maximum {MAX_EDITS} allowed, got {}",
            edits.len()
        )));
    }
    let dry_run = input
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let skip_linter = input
        .get("skip_linter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(ParsedInput {
        edits,
        dry_run,
        skip_linter,
    })
}

/// One parsed edit operation, with the resolved absolute path.
///
/// Built by [`build_operations`] from each item in the caller's `edits`
/// array. The `file_path` is kept verbatim (pre-resolution) for error
/// and preview messages; `full_path` is what every read/write actually
/// targets.
#[derive(Debug, Clone)]
struct EditOperation {
    /// The caller-supplied path (pre-resolution).
    ///
    /// Used in messages so the model sees the path it named, not the
    /// canonicalized form.
    file_path: String,

    /// The path resolved against the session's working directory.
    ///
    /// This is what every read/write and the duplicate-path / symlink
    /// checks actually target.
    full_path: PathBuf,

    /// The text to find in the file.
    ///
    /// Must be non-empty and, at merge time, appear exactly once in the
    /// running content of the target file (see [`merge_per_file`]); an
    /// absent or ambiguous match aborts the whole batch.
    old_text: String,

    /// The text to replace `old_text` with.
    ///
    /// Spliced over the matched range by [`splice`]; may be empty, which
    /// deletes the matched text.
    new_text: String,
}

/// Parse each edit item into an [`EditOperation`].
///
/// Validates fields, rejecting empty `old_text` and URL `file_path`;
/// relative paths are resolved against `cwd` under `policy`; absolute
/// paths are accepted only when they stay inside `cwd` (contained
/// resolution).
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a missing `file_path`/
/// `old_text`/`new_text`, an empty `old_text`, a URL `file_path`, or —
/// under contained resolution — a path that escapes `cwd`.
fn build_operations(
    edits: &[Value],
    cwd: &Path,
    policy: ResolvePolicy,
) -> Result<Vec<EditOperation>, ToolError> {
    let mut operations = Vec::with_capacity(edits.len());
    for edit_value in edits {
        let file_path = edit_value
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("Missing file_path in edit".to_string()))?;
        let old_text = edit_value
            .get("old_text")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("Missing old_text in edit".to_string()))?;
        let new_text = edit_value
            .get("new_text")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("Missing new_text in edit".to_string()))?;

        if old_text.is_empty() {
            return Err(ToolError::InvalidInput(
                "old_text must not be empty".to_string(),
            ));
        }
        resolve::reject_path_url("MultiEdit", file_path)?;

        let full_path = resolve::resolve_path(file_path, cwd, policy)?;
        operations.push(EditOperation {
            file_path: file_path.to_string(),
            full_path,
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
        });
    }
    Ok(operations)
}

/// Read each distinct target file once into a map keyed by `file_path`.
///
/// Returns the original contents keyed by the caller-supplied path (so
/// the preview can address files the way the model named them). Under
/// the contained policy each opened handle is verified against the
/// session's pinned workspace anchor, so a symlink swapped onto the
/// workspace spelling after the session was constructed cannot feed a
/// batch from outside the pinned workspace.
///
/// # Errors
///
/// Returns `ToolError::FileNotFound` when a target does not exist,
/// `ToolError::Execution` on any other read fault (including
/// non-UTF-8), and when a contained handle check fails.
async fn read_files(
    operations: &[EditOperation],
    session: &FileSession,
) -> Result<BTreeMap<String, String>, ToolError> {
    let workspace = session.cwd();
    let policy = session.resolve_policy();
    let mut originals = BTreeMap::new();
    for op in operations {
        if originals.contains_key(&op.file_path) {
            continue;
        }
        if !tokio::fs::try_exists(&op.full_path)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?
        {
            return Err(ToolError::FileNotFound(op.file_path.clone()));
        }
        let mut file = tokio::fs::File::open(&op.full_path)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        if policy == ResolvePolicy::Contained {
            resolve::verify_handle_inside(&file, workspace, Some(session.anchor()))?;
        }
        let mut content = String::new();
        file.read_to_string(&mut content)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        originals.insert(op.file_path.clone(), content);
    }
    Ok(originals)
}

/// A recoverable reason a batch was aborted before any write.
///
/// Each variant carries a pre-formatted message and is converted to a
/// soft [`ToolOutput`] via [`AbortReason::into_output`]. Distinct from a
/// hard [`ToolError`], which is reserved for bad arguments, missing
/// files, and I/O faults that the model cannot simply retry around.
#[derive(Debug)]
enum AbortReason {
    /// A target, or an in-workspace parent directory, is a symbolic link
    /// (named in the message).
    ///
    /// Produced by [`symlink_check`] under the contained policy, before
    /// any file is written: resolve the link and pass the real path.
    Symlink(String),

    /// Two edits' caller-supplied paths are physically the same file.
    ///
    /// Both spellings are named in the message, plus the shared stat
    /// identity where the platform prints one. Produced by
    /// [`physical_alias_check`] under both policies: each alias would
    /// merge independently against the same original, and the second
    /// write would silently clobber the first.
    Alias(String),

    /// Two edits resolve to the same physical file under different path
    /// aliases.
    ///
    /// The aliases are named in the message; the result would be
    /// ambiguous. Produced by [`dup_path_check`].
    DupPath(String),

    /// One edit's `old_text` is absent or not unique in the running
    /// content.
    ///
    /// The file is named in the message. Produced by [`merge_per_file`].
    Locate(String),

    /// Two edits' byte-ranges overlap in the same file (named in the
    /// message).
    ///
    /// Produced by [`overlap_check`], which runs in both `dry_run` and
    /// apply modes so the preview shows exactly what an apply would
    /// catch.
    Overlap(String),

    /// The installed validator rejected one file's merged content (named
    /// in the message).
    ///
    /// Produced by [`validate_all`] (skipped when `skip_linter` is set).
    Lint(String),
}

impl AbortReason {
    /// Format this reason as the soft [`ToolOutput`] returned to the loop.
    ///
    /// Every variant carries a pre-formatted human-readable message; this
    /// just wraps it as an error output so the model can read the reason
    /// and retry.
    fn into_output(self) -> ToolOutput {
        match self {
            AbortReason::Symlink(msg)
            | AbortReason::Alias(msg)
            | AbortReason::DupPath(msg)
            | AbortReason::Locate(msg)
            | AbortReason::Overlap(msg)
            | AbortReason::Lint(msg) => ToolOutput::error_text(msg),
        }
    }
}

/// Reject the batch if two edits resolve to the same physical file under
/// different path aliases.
///
/// Aliases like `a.rs` and `./a.rs` are the trigger — the lexically
/// visible case. Maps each resolved path to the first caller-supplied
/// path seen for it and rejects on any later alias. Each edit would
/// otherwise be merged independently against the same original, and the
/// second write would silently clobber the first — losing one edit-set.
/// Refusing is safer than picking a winner. Multiple edits sharing both
/// `file_path` and `full_path` (the normal multi-edit-to-one-file case)
/// are allowed. Lexically distinct but physically identical targets —
/// the workspace's resolved spelling beside the operator's, or hard
/// links — are caught by [`physical_alias_check`], which runs after this
/// check.
fn dup_path_check(operations: &[EditOperation]) -> Option<AbortReason> {
    let mut owner: std::collections::HashMap<&Path, &str> = std::collections::HashMap::new();
    for op in operations {
        match owner.get(op.full_path.as_path()) {
            Some(&existing) if existing != op.file_path => {
                return Some(AbortReason::DupPath(format!(
                    "Edits target the same file via two different paths: '{}' and '{}' both \
                     resolve to '{}'. Combine them into one set of edits.",
                    existing,
                    op.file_path,
                    op.full_path.display()
                )));
            }
            None => {
                owner.insert(op.full_path.as_path(), &op.file_path);
            }
            _ => {}
        }
    }
    None
}

/// The first batch target whose only recorded observation is resume-armed,
/// if any.
///
/// Write holds such a target for a live read; the batch applies the same
/// rule to every distinct target before anything is written, so an edit
/// can never land on bytes the model never saw in this session. Distinct
/// physical paths are each checked once.
fn resumed_target<'a>(operations: &'a [EditOperation], session: &FileSession) -> Option<&'a Path> {
    let mut seen = std::collections::HashSet::new();
    operations
        .iter()
        .map(|op| op.full_path.as_path())
        .filter(|path| seen.insert(*path))
        .find(|path| {
            session
                .baseline_for(path)
                .is_some_and(|baseline| baseline.resumed)
        })
}

/// Reject the batch if any target — or any of its in-workspace ancestor
/// directories — is a symbolic link.
///
/// The write path's own symlink guard fires at write time and checks
/// only the final component, too late for the atomic contract (file #1
/// could already be written before file #2's symlink errors). This
/// pre-check walks every ancestor of each target with `symlink_metadata`
/// (no follow) during the read pass, before any write, so a symlinked
/// parent directory is caught too. Ancestors at or above `workspace` are
/// skipped: the anchor's own spelling is the operator's choice and may
/// cross symlinks by design — the same below-anchor judgment the pinned
/// write's walk applies — so only in-workspace components are judged.
fn symlink_check(operations: &[EditOperation], workspace: &Path) -> Option<AbortReason> {
    let mut seen = std::collections::HashSet::new();
    for op in operations {
        for ancestor in op.full_path.ancestors() {
            if workspace.starts_with(ancestor) {
                continue;
            }
            if !seen.insert(ancestor) {
                continue;
            }
            if std::fs::symlink_metadata(ancestor).is_ok_and(|m| m.file_type().is_symlink()) {
                return Some(AbortReason::Symlink(format!(
                    "Refusing to write: {} crosses a symbolic link ({}). \
                     Resolve it and pass the real path.",
                    op.file_path,
                    ancestor.display()
                )));
            }
        }
    }
    None
}

/// Rewrite each target to its physical path (Unrestricted).
///
/// Unrestricted writes honor symbolic links, so each existing target is
/// canonicalized and the write lands on the real file with the link left
/// intact; targets that do not exist yet (new files) keep their
/// submitted path. An unresolvable link — dangling or looping — refuses
/// the batch with the same resolution error Write and Edit surface,
/// rather than a generic miss later in the read phase. Physically
/// identical targets are refused earlier by [`physical_alias_check`],
/// which also covers the hard-link aliases canonicalization cannot see.
///
/// # Errors
///
/// Returns [`ToolError`] when a target cannot be canonicalized.
fn honor_symlink_targets(operations: &mut [EditOperation]) -> Result<(), ToolError> {
    for op in operations {
        op.full_path = resolve::canonicalize_existing(&op.full_path)?;
    }
    Ok(())
}

/// Reject the batch when two lexically distinct targets are physically the
/// same file.
///
/// Path spellings can alias without any symlink: the workspace's
/// resolved spelling and the operator's spelling both pass containment
/// and name the same directories, and hard links alias any two entries.
/// Each alias would merge independently against the same original and
/// the writes would clobber one another — and the staleness re-read
/// would then blame an external change the batch itself produced. Every
/// distinct target is stated once and compared by physical identity;
/// entries that cannot be stated do not exist yet and are left to the
/// read phase to refuse. Runs under both policies: it protects the
/// batch's own merge semantics, it is not containment.
///
/// Identity is per-platform (see [`physical_identity`]). Degradation: on
/// platforms that expose no identity the check cannot fire — under
/// Contained those platforms refuse contained writes at the write
/// anyway, while unrestricted multi-edit there keeps the alias hole.
/// Two residuals are accepted and shared with the rest of the guard
/// stack: separate (non-batch) calls to two hard-link spellings can
/// still split a shared file — each call is internally consistent and no
/// batch contract is at stake — and the universal stat-to-rename swap
/// window applies here as everywhere else.
fn physical_alias_check(operations: &[EditOperation]) -> Option<AbortReason> {
    let mut seen = std::collections::HashSet::new();
    let mut identities: std::collections::HashMap<PhysicalIdentity, String> =
        std::collections::HashMap::new();
    for op in operations {
        if !seen.insert(op.full_path.as_path()) {
            continue;
        }
        let Some(identity) = physical_identity(&op.full_path) else {
            continue;
        };
        if let Some(existing) = identities.get(&identity) {
            if *existing != op.file_path.as_str() {
                return Some(AbortReason::Alias(alias_refusal(
                    existing,
                    &op.file_path,
                    identity,
                )));
            }
            continue;
        }
        identities.insert(identity, op.file_path.clone());
    }
    None
}

/// The physical identity of one filesystem entry, for alias comparison.
///
/// Unix keys by the stat pair (device, inode). A platform without such
/// an identity has nothing to key on and the check degrades to no
/// detection there.
#[cfg(unix)]
type PhysicalIdentity = (u64, u64);

/// The physical identity of one filesystem entry, for alias comparison.
///
/// Platforms with no stat identity use the unit type as a placeholder
/// that never resolves, so [`physical_alias_check`] degrades to no
/// detection there.
#[cfg(not(unix))]
type PhysicalIdentity = ();

/// The identity of the file `path` reaches, or `None` when it cannot be
/// determined.
///
/// `None` covers a not-yet-existing target — the read phase refuses that
/// instead — and an entry that exposes no identity.
#[cfg(unix)]
fn physical_identity(path: &Path) -> Option<PhysicalIdentity> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path)
        .ok()
        .map(|meta| (meta.dev(), meta.ino()))
}

/// The identity of the file `path` reaches, or `None` when it cannot be
/// determined.
///
/// Platform without an identity mechanism: always `None`.
#[cfg(not(unix))]
fn physical_identity(_path: &Path) -> Option<PhysicalIdentity> {
    None
}

/// Format the refusal for two spellings of one physical file.
///
/// Unix names the shared stat pair so the caller can see the aliases
/// collided at the inode level; platforms without a printable identity
/// name the fact alone.
#[cfg(unix)]
fn alias_refusal(first: &str, second: &str, identity: PhysicalIdentity) -> String {
    format!(
        "Refusing to write: '{first}' and '{second}' are the same file (same \
         inode: {}:{}). Combine them into one set of edits.",
        identity.0, identity.1
    )
}

/// Format the refusal for two spellings of one physical file.
///
/// Platform without an identity mechanism: the branch is unreachable
/// (no identity, no collision), but the compilation unit needs the
/// binding.
#[cfg(not(unix))]
fn alias_refusal(first: &str, second: &str, _identity: PhysicalIdentity) -> String {
    format!("Refusing to write: '{first}' and '{second}' are the same file.")
}

/// A pair of edits whose `old_text` byte-ranges overlap in the same file.
///
/// Produced by [`detect_edit_conflicts`]. Each field names the two edits
/// (by their 0-indexed position in the batch and a truncated snippet of
/// their `old_text`) so the abort message can point the caller at both.
#[derive(Debug, Clone)]
struct EditConflict {
    /// The file both edits target.
    ///
    /// Stored as the caller-supplied path (pre-resolution), not the
    /// normalized form, so the abort message addresses the file the way
    /// the model named it.
    file_path: String,

    /// 0-indexed position of the first edit in the batch's `edits` array.
    ///
    /// The raw value is used for the dedup-by-sorted-pair step in
    /// [`detect_edit_conflicts`]; the abort message renders it 1-indexed
    /// (`edit_index_a + 1`) for human readability.
    edit_index_a: usize,

    /// 0-indexed position of the second edit in the batch's `edits` array.
    ///
    /// Paired with [`edit_index_a`](Self::edit_index_a) to identify the
    /// conflicting pair; rendered 1-indexed in the abort message.
    edit_index_b: usize,

    /// Truncated `old_text` of the first edit, for the abort message.
    ///
    /// Produced by [`truncate_str`] so a long needle doesn't flood the
    /// output; the caller sees enough to recognize the edit without the
    /// full text.
    snippet_a: String,

    /// Truncated `old_text` of the second edit, for the abort message.
    ///
    /// Produced by [`truncate_str`]; paired with
    /// [`snippet_a`](Self::snippet_a) so the message shows both halves of
    /// the overlapping pair.
    snippet_b: String,
}

/// Detect pairs of edits whose `old_text` ranges overlap in the same file.
///
/// Two edits to one file conflict when one's matched byte-range
/// intersects the other's (one contains the other, or they share text).
/// Applying either first would invalidate the other's match. Different
/// files never conflict, even with identical `old_text`; a file needs at
/// least two edits before a pair can exist. Each edit contributes at
/// most one range — the first occurrence's, since full uniqueness is
/// only enforced later by [`locate_unique`] at merge time. Each
/// conflicting pair is reported once, deduplicated by its sorted index
/// pair.
fn detect_edit_conflicts(
    operations: &[EditOperation],
    file_contents: &BTreeMap<String, String>,
) -> Vec<EditConflict> {
    let mut conflicts = Vec::new();

    let mut file_edits: BTreeMap<String, Vec<(usize, &str)>> = BTreeMap::new();
    for (i, op) in operations.iter().enumerate() {
        file_edits
            .entry(op.file_path.clone())
            .or_default()
            .push((i, &op.old_text));
    }

    for (file_path, edits) in &file_edits {
        if edits.len() < 2 {
            continue;
        }
        let Some(content) = file_contents.get(file_path) else {
            continue;
        };

        let mut ranges: Vec<(usize, usize, usize, &str)> = Vec::new();
        for (edit_idx, old_text) in edits {
            if let Some(start) = content.find(old_text) {
                let end = start.saturating_add(old_text.len());
                let snippet = truncate_str(old_text, 60);
                ranges.push((start, end, *edit_idx, snippet));
            }
        }

        ranges.sort_by_key(|r| r.0);

        for i in 0..ranges.len() {
            let Some(&(_start_a, end_a, idx_a, snippet_a)) = ranges.get(i) else {
                continue;
            };
            for &entry in ranges.iter().skip(i.saturating_add(1)) {
                let (start_b, _end_b, idx_b, snippet_b) = entry;
                if idx_a == idx_b {
                    continue;
                }
                if start_b < end_a {
                    conflicts.push(EditConflict {
                        file_path: file_path.clone(),
                        edit_index_a: idx_a,
                        edit_index_b: idx_b,
                        snippet_a: snippet_a.to_string(),
                        snippet_b: snippet_b.to_string(),
                    });
                }
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    conflicts.retain(|c| {
        let key = (
            c.edit_index_a.min(c.edit_index_b),
            c.edit_index_a.max(c.edit_index_b),
        );
        seen.insert(key)
    });

    conflicts
}

/// Truncate to at most `max_len` bytes, landing on a UTF-8 char boundary.
///
/// Used to keep `old_text` snippets short in conflict messages. If
/// `max_len` falls inside a multibyte char, the cut backs up to the
/// preceding boundary so the result is always valid UTF-8.
fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while !s.is_char_boundary(end) && end > 0 {
            end = end.saturating_sub(1);
        }
        s.get(..end).unwrap_or(s)
    }
}

/// Reject the batch if any two edits' byte-ranges overlap in the same file.
///
/// Delegates the detection to [`detect_edit_conflicts`] and, on the
/// first conflict, formats a message naming both edits (by 1-indexed
/// position and a truncated `old_text` snippet) with a hint to split the
/// batch or dry-run. Runs in both `dry_run` and apply modes so the
/// preview shows exactly what an apply would catch.
fn overlap_check(
    operations: &[EditOperation],
    originals: &BTreeMap<String, String>,
) -> Option<AbortReason> {
    let conflicts = detect_edit_conflicts(operations, originals);
    if conflicts.is_empty() {
        return None;
    }
    let mut msg =
        "Edit conflict detected — the following edits overlap in the same file:\n".to_string();
    for conflict in &conflicts {
        writeln!(
            msg,
            "  - File '{}': edit #{} and edit #{} target overlapping text regions",
            conflict.file_path,
            conflict.edit_index_a.saturating_add(1),
            conflict.edit_index_b.saturating_add(1)
        )
        .ok();
        writeln!(
            msg,
            "    Edit #{} old_text: {:?}...",
            conflict.edit_index_a.saturating_add(1),
            conflict.snippet_a
        )
        .ok();
        writeln!(
            msg,
            "    Edit #{} old_text: {:?}...",
            conflict.edit_index_b.saturating_add(1),
            conflict.snippet_b
        )
        .ok();
    }
    msg.push_str(
        "\nResolve by: (1) splitting into separate calls, or (2) use dry_run=true to preview first.",
    );
    Some(AbortReason::Overlap(msg))
}

/// Merge each file's edits sequentially into a final content map,
/// validating uniqueness as it goes.
///
/// Edits run in array order, each seeing the prior's output; each edit's
/// `old_text` must be unique in the *running* content at that point. A
/// later edit to the same file may target text that only exists after
/// an earlier edit runs — so the locate check must be against the
/// accumulated content, not the original.
///
/// # Errors
///
/// Returns `Err(AbortReason::Locate)` on the first edit whose `old_text`
/// is absent or not unique in the running content at that point in
/// the sequence.
fn merge_per_file(
    operations: &[EditOperation],
    originals: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, AbortReason> {
    let mut finals: BTreeMap<String, String> = BTreeMap::new();
    for op in operations {
        let entry = finals
            .entry(op.file_path.clone())
            .or_insert_with(|| originals.get(&op.file_path).cloned().unwrap_or_default());
        match locate_unique(entry, &op.old_text) {
            FindResult::NotFound => {
                return Err(AbortReason::Locate(format!(
                    "Old text not found in file: {}",
                    op.file_path
                )));
            }
            FindResult::Ambiguous { count } => {
                return Err(AbortReason::Locate(format!(
                    "old_text appears {count} times in file {}; it must be unique. \
                     Add surrounding context to disambiguate, or use Edit.",
                    op.file_path
                )));
            }
            FindResult::Unique(range) => {
                *entry = splice(entry, range, &op.new_text);
            }
        }
    }
    Ok(finals)
}

/// Validate each distinct file's final merged content; return the first
/// failure.
///
/// Iterates the operations but validates each physical file only once
/// (deduped by `full_path`). A failure produces [`AbortReason::Lint`]
/// carrying the formatted diagnostics plus the "No files were modified"
/// trailer.
async fn validate_all(
    validator: &dyn ContentValidator,
    operations: &[EditOperation],
    finals: &BTreeMap<String, String>,
) -> Option<AbortReason> {
    let mut seen = std::collections::HashSet::new();
    for op in operations {
        if !seen.insert(&op.full_path) {
            continue;
        }
        let Some(final_content) = finals.get(&op.file_path) else {
            continue;
        };
        let diagnostics = validator.validate(&op.full_path, final_content).await;
        if !diagnostics.is_empty() {
            return Some(AbortReason::Lint(format!(
                "{}\n\nNo files were modified.",
                format_validation_failure(&op.full_path, &diagnostics).trim_end()
            )));
        }
    }
    None
}

/// Build the per-file diff preview block, with a header chosen by `dry_run`.
///
/// Lists each distinct file once (in first-seen order across the batch),
/// then the indented output of
/// [`format_file_change`](super::diff::format_file_change) comparing that
/// file's *original* content against its *fully merged* final content
/// from `finals` — so multiple edits to one file render cumulatively,
/// not as isolated fragments. The dry-run path appends a footer telling
/// the caller how to apply; the apply path reuses this block as the
/// summary header.
fn build_preview(
    operations: &[EditOperation],
    originals: &BTreeMap<String, String>,
    finals: &BTreeMap<String, String>,
    dry_run: bool,
) -> String {
    let mut lines = Vec::new();
    if dry_run {
        lines.push("Dry Run Preview — No files will be modified".to_string());
    } else {
        lines.push("Multi-File Edit Summary".to_string());
    }
    lines.push(String::new());

    let mut seen = std::collections::HashSet::new();
    let mut index = 1usize;
    for op in operations {
        if !seen.insert(&op.file_path) {
            continue;
        }
        lines.push(format!("File {index}: {}", op.file_path));
        let original = originals.get(&op.file_path).map_or("", String::as_str);
        let final_content = finals.get(&op.file_path).map_or("", String::as_str);
        let diff = format_file_change(&op.file_path, OldContent::Text(original), final_content);
        for line in diff.lines() {
            lines.push(format!("  {line}"));
        }
        lines.push(String::new());
        index = index.saturating_add(1);
    }

    if dry_run {
        lines.push("Use dry_run=false to apply these changes.".to_string());
    }
    lines.join("\n")
}

/// Append the applied-files summary to the preview block.
///
/// Called only on the apply path (not dry-run). Lists each written file
/// once, with its edit count when more than one edit targeted it.
fn apply_summary(preview: &str, applied: &[&str], operations: &[EditOperation]) -> String {
    let mut result = preview.to_string();
    result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    writeln!(result, "Applied: {} file(s)", applied.len()).ok();
    for path in applied {
        let count = operations
            .iter()
            .filter(|o| &o.file_path.as_str() == path)
            .count();
        if count > 1 {
            writeln!(result, "  + {path} ({count} edits)").ok();
        } else {
            writeln!(result, "  + {path}").ok();
        }
    }
    result
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::tool::builtin::fs::ValidationDiagnostic;
    use serde_json::json;
    use std::path::PathBuf;

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        super::super::FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
        ctx
    }

    fn ctx_with_policy(cwd: &std::path::Path, policy: ResolvePolicy) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string_lossy().into_owned();
        super::super::FileSession::new(cwd.to_path_buf())
            .with_resolve_policy(policy)
            .attach(&mut ctx);
        ctx
    }

    fn edit(file_path: &str, old_text: &str, new_text: &str) -> Value {
        json!({"file_path": file_path, "old_text": old_text, "new_text": new_text})
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
                        line: None,
                        message: "content contains BAD".to_string(),
                    }]
                } else {
                    Vec::new()
                }
            })
        }
    }

    #[tokio::test]
    async fn applies_edits_across_files_and_refreshes_baselines() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "alpha\n").unwrap();
        std::fs::write(tmp.path().join("b.rs"), "beta\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let ctx = ctx_in(cwd);
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("a.rs", "alpha", "ALPHA"),
                    edit("b.rs", "beta", "BETA")
                ]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "ALPHA\n"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("b.rs")).unwrap(),
            "BETA\n"
        );
        let text = out.text_content();
        assert!(text.contains("Applied: 2 file(s)"), "{text}");
        assert!(text.contains("File 1: a.rs"), "{text}");
        assert!(text.contains("File 2: b.rs"), "{text}");
    }

    #[tokio::test]
    async fn multiple_edits_to_one_file_merge_cumulatively() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "one\ntwo\nthree\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("a.rs", "one", "ONE"),
                    edit("a.rs", "three", "THREE")
                ]}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "ONE\ntwo\nTHREE\n"
        );
        assert!(
            out.text_content().contains("a.rs (2 edits)"),
            "{}",
            out.text_content()
        );
    }

    #[tokio::test]
    async fn one_invalid_edit_aborts_the_whole_batch() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "alpha\n").unwrap();
        std::fs::write(tmp.path().join("b.rs"), "beta\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("a.rs", "alpha", "ALPHA"),
                    edit("b.rs", "absent", "X")
                ]}),
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
            "alpha\n",
            "an aborted batch must leave every file untouched"
        );
    }

    #[tokio::test]
    async fn overlapping_edits_abort_with_both_named() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "hello world\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("a.rs", "hello wo", "A"),
                    edit("a.rs", "world", "B")
                ]}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        let text = out.text_content();
        assert!(text.contains("overlap"), "{text}");
        assert!(text.contains("edit #1 and edit #2"), "{text}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "hello world\n"
        );
    }

    #[tokio::test]
    async fn duplicate_paths_abort_with_both_spellings_named() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "x\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("a.rs", "x", "y"),
                    edit("./a.rs", "x", "z")
                ]}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content().contains("two different paths"),
            "{}",
            out.text_content()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hard_link_aliases_abort_the_batch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let real = tmp.path().join("real.rs");
        std::fs::write(&real, "x\n").unwrap();
        let hard = tmp.path().join("hard.rs");
        std::fs::hard_link(&real, &hard).unwrap();
        let ctx = ctx_with_policy(tmp.path(), ResolvePolicy::Unrestricted);
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [
                    edit("real.rs", "x", "y"),
                    edit("hard.rs", "x", "z")
                ]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.text_content().contains("are the same file"),
            "{}",
            out.text_content()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_target_aborts_the_batch_under_containment() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("real.rs"), "x\n").unwrap();
        std::fs::write(outside.path().join("real.rs"), "x\n").unwrap();
        symlink(tmp.path().join("real.rs"), tmp.path().join("link.rs")).unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(json!({"edits": [edit("link.rs", "x", "y")]}), &ctx_in(cwd))
            .await
            .unwrap();
        assert!(
            out.is_error,
            "a link target must abort the batch before any write"
        );
        assert!(
            out.text_content().contains("crosses a symbolic link"),
            "the refusal names the link: {}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("real.rs")).unwrap(),
            "x\n",
            "an aborted batch must not write through the link"
        );
    }

    #[tokio::test]
    async fn dry_run_previews_without_writing() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "alpha\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let out = MultiEditTool::new()
            .call(
                json!({"edits": [edit("a.rs", "alpha", "ALPHA")], "dry_run": true}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let text = out.text_content();
        assert!(text.contains("Dry Run Preview"), "{text}");
        assert!(text.contains("Use dry_run=false to apply"), "{text}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "alpha\n",
            "a dry run must not write"
        );
    }

    #[tokio::test]
    async fn an_external_change_mid_batch_reports_the_applied_files() {
        struct MutateB {
            b_path: std::path::PathBuf,
        }

        impl ContentValidator for MutateB {
            fn validate<'a>(
                &'a self,
                path: &'a Path,
                _content: &'a str,
            ) -> Pin<Box<dyn Future<Output = Vec<ValidationDiagnostic>> + Send + 'a>> {
                Box::pin(async move {
                    if path == self.b_path.as_path() {
                        std::fs::write(path, "EXTERNAL\n").unwrap();
                    }
                    Vec::new()
                })
            }
        }

        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "alpha\n").unwrap();
        std::fs::write(tmp.path().join("b.rs"), "beta\n").unwrap();
        let ctx = ctx_in(tmp.path().to_str().unwrap());
        let tool = MultiEditTool::new().with_validator(Arc::new(MutateB {
            b_path: tmp.path().join("b.rs"),
        }));
        let out = tool
            .call(
                json!({"edits": [
                    edit("a.rs", "alpha", "ALPHA"),
                    edit("b.rs", "beta", "BETA")
                ]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            out.is_error,
            "the externally changed file must abort the batch"
        );
        let text = out.text_content();
        assert!(text.contains("changed on disk"), "{text}");
        assert!(
            text.contains("Already written by this batch: a.rs"),
            "the earlier writes must be reported: {text}"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "ALPHA\n",
            "writes before the conflict stay on disk"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("b.rs")).unwrap(),
            "EXTERNAL\n",
            "the externally changed file must be untouched"
        );
    }

    #[tokio::test]
    async fn too_many_edits_is_invalid_input() {
        let edits: Vec<Value> = (0..51).map(|_| edit("a.rs", "x", "y")).collect();
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let err = MultiEditTool::new()
            .call(json!({"edits": edits}), &ctx_in(cwd))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("Too many edits")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_failing_validator_aborts_with_no_files_modified() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "x\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = MultiEditTool::new().with_validator(Arc::new(RejectBad));
        let out = tool
            .call(json!({"edits": [edit("a.rs", "x", "BAD")]}), &ctx_in(cwd))
            .await
            .unwrap();
        assert!(out.is_error);
        let text = out.text_content();
        assert!(text.contains("Syntax validation failed"), "{text}");
        assert!(text.contains("No files were modified"), "{text}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a.rs")).unwrap(),
            "x\n"
        );
    }

    #[tokio::test]
    async fn missing_edits_array_is_invalid_input() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let err = MultiEditTool::new()
            .call(json!({"dry_run": true}), &ctx_in(cwd))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn schema_pins_the_nested_edits_array() {
        let schema = MultiEditTool::new().schema();
        let str = serde_json::to_string(&schema.input_schema).unwrap();
        assert!(str.contains("minItems"), "{str}");
        assert!(str.contains("maxItems"), "{str}");
        assert!(str.contains("old_text"), "{str}");
    }

    #[tokio::test]
    async fn a_resumed_baseline_holds_the_batch_until_a_live_read() {
        let tmp = tempfile::TempDir::new().unwrap();
        let held_file = tmp.path().join("held.rs");
        let free_file = tmp.path().join("free.rs");
        std::fs::write(&held_file, "alpha\n").unwrap();
        std::fs::write(&free_file, "beta\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = super::super::FileSession::new(PathBuf::from(cwd));
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&held_file, state::observe_resumed_bytes(b"alpha\n"));

        let batch = json!({"edits": [
            {"file_path": "held.rs", "old_text": "alpha", "new_text": "ALPHA"},
            {"file_path": "free.rs", "old_text": "beta", "new_text": "BETA"}
        ]});
        let held = MultiEditTool::new()
            .call(batch.clone(), &ctx)
            .await
            .unwrap();
        assert!(held.is_error, "a resume-armed baseline must hold the batch");
        assert!(
            held.text_content().contains("previous session"),
            "{}",
            held.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(&held_file).unwrap(),
            "alpha\n",
            "nothing may be written while held"
        );
        assert_eq!(
            std::fs::read_to_string(&free_file).unwrap(),
            "beta\n",
            "the whole batch is refused, not just the held file"
        );

        session.record_baseline(&held_file, state::observe_bytes(b"alpha\n"));
        let after = MultiEditTool::new().call(batch, &ctx).await.unwrap();
        assert!(
            !after.is_error,
            "a live read must release the guard: {}",
            after.text_content()
        );
        assert_eq!(std::fs::read_to_string(&held_file).unwrap(), "ALPHA\n");
        assert_eq!(std::fs::read_to_string(&free_file).unwrap(), "BETA\n");
    }

    #[tokio::test]
    async fn flags_advertise_write_semantics() {
        let tool = MultiEditTool::new();
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }
}
