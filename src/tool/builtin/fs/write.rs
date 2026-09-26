//! The `Write` tool — writes content to a file, after the validation gate
//! and a staleness check against the path's last-recorded read.

use std::fmt;
use std::future::Future;
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
use super::ValidationDiagnostic;
use super::atomic;
use super::conflict;
use super::conflict::CheckFailure;
use super::diff::format_file_change;
use super::require_session;
use super::resolve;
use super::resolve::ResolvePolicy;
use super::state;

/// The file-writing tool over the session's workspace.
///
/// Writes content to a file, creating parent directories as needed and
/// running the installed [`ContentValidator`] on the candidate content
/// before the atomic write. The model-facing name is `Write`.
#[derive(Clone)]
pub struct WriteTool {
    /// The installed syntax gate, if any.
    ///
    /// `None` — the default — performs no validation: what counts as
    /// valid content is a host policy, and the library stays neutral.
    validator: Option<Arc<dyn ContentValidator>>,
}

impl WriteTool {
    /// Build a write tool with no validation gate.
    ///
    /// The tool writes whatever content it is given; install a gate with
    /// `with_validator` when a host wants syntax checking before the write.
    #[must_use]
    pub fn new() -> Self {
        Self { validator: None }
    }

    /// Install a validation gate the tool consults before writing.
    ///
    /// The validator sees the candidate content and the resolved path;
    /// any diagnostic blocks the write with the family's shared refusal
    /// text, and the model-facing `skip_linter` flag bypasses the gate
    /// for that one call.
    #[must_use]
    pub fn with_validator(mut self, validator: Arc<dyn ContentValidator>) -> Self {
        self.validator = Some(validator);
        self
    }
}

impl fmt::Debug for WriteTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteTool")
            .field("validator", &self.validator.is_some())
            .finish()
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "Write"
    }

    fn description(&self) -> &'static str {
        "Write content to a file. Syntax validation is automatically performed \
         for supported file types (.rs, .json, .py, .js, .ts, etc.)"
    }

    fn system_prompt(&self) -> Option<String> {
        Some(
            "Use Write for new files or full rewrites; prefer Edit for \
             targeted changes. The linter runs automatically on supported \
             types — fix reported errors."
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
                        "description": "The path to the file to write.  May be absolute or relative; relative paths are resolved against the runner's working directory. Parent directories are created as needed, and an existing file is overwritten in full — prefer Edit for targeted changes to an existing file. URLs are rejected."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.  The complete new contents of the file, not a patch or fragment. The linter gate validates it for supported file types before the write happens, so malformed syntax is blocked rather than written to disk."
                    },
                    "skip_linter": {
                        "type": "boolean",
                        "description": "Skip syntax validation (not recommended).  When `true`, the linter gate is bypassed and the file is written even if the content has syntax errors. Defaults to `false`; the gate exists to prevent file corruption from malformed output."
                    }
                },
                "required": ["file_path", "content"]
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
            write_inner(self, input, &session).await
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
/// Orchestrates validate → gate → staleness check → write. When the
/// path has a recorded baseline (a prior read this session), content
/// that differs from the recorded hash refuses the write as a soft
/// conflict; a baseline whose only observation is resume-armed refuses
/// the write outright — the model never saw the file's bytes in this
/// session, so there is nothing honest to compare against — until a
/// live read re-records them. A successful write refreshes the recorded
/// baseline, so the model's own write never registers as a later
/// external change.
///
/// # Errors
///
/// Returns [`ToolError`] for a missing `FileSession`, a missing
/// `file_path`, a missing `content`, a URL `file_path` or a path
/// escaping the working directory, a target whose only recorded
/// baseline is resume-armed, a target that changed while the write was
/// being prepared, or a file-system error during parent creation or the
/// atomic write.
async fn write_inner(
    tool: &WriteTool,
    input: Value,
    session: &FileSession,
) -> Result<ToolOutput, ToolError> {
    let policy = session.resolve_policy();
    let cwd = session.cwd().to_path_buf();
    let file_path = input
        .get("file_path")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing file_path".to_string()))?;
    resolve::reject_path_url("Write", file_path)?;
    let content = input
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing content".to_string()))?;
    let skip_linter = input
        .get("skip_linter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut full_path = resolve::resolve_path(file_path, &cwd, policy)?;
    if policy == ResolvePolicy::Unrestricted {
        full_path = resolve::canonicalize_existing(&full_path)?;
    }

    if !skip_linter && let Some(validator) = tool.validator.as_ref() {
        let diagnostics = validator.validate(&full_path, content).await;
        if !diagnostics.is_empty() {
            return Ok(ToolOutput::error_text(format_validation_failure(
                &full_path,
                &diagnostics,
            )));
        }
    }

    let old_content = match tokio::fs::File::open(&full_path).await.ok() {
        Some(mut file) => {
            if policy == ResolvePolicy::Contained {
                resolve::verify_handle_inside(&file, &cwd, Some(session.anchor()))?;
            }
            let mut buffer = String::new();
            match file.read_to_string(&mut buffer).await {
                Ok(_) => Some(buffer),
                Err(_) => None,
            }
        }
        None => None,
    };

    let mut expected = None;
    if let Some(baseline) = session.baseline_for(&full_path) {
        if baseline.resumed {
            return Ok(ToolOutput::error_text(resumed_baseline_message(&full_path)));
        }
        match conflict::check_content_hash_unchanged(baseline.hash, &full_path).await {
            Ok(identity) => expected = Some(identity),
            Err(failure) => {
                return match failure {
                    CheckFailure::Changed => Ok(ToolOutput::error_text(conflict::changed_message(
                        &full_path,
                    ))),
                    CheckFailure::Fault(e) => Err(e),
                };
            }
        }
    }

    if policy == ResolvePolicy::Unrestricted
        && let Some(parent) = full_path.parent()
    {
        tokio::fs::create_dir_all(parent).await?;
    }

    atomic::atomic_write(
        &full_path,
        content,
        &cwd,
        policy,
        expected.as_ref(),
        Some(session.anchor()),
    )?;

    session.record_baseline(&full_path, state::observe_bytes(content.as_bytes()));

    let message = format_file_change(file_path, old_content.as_deref(), content);

    Ok(ToolOutput::text(message).with_hint(DisplayHint::Diff))
}

/// Format the soft-error message for a write held against a resume-armed
/// baseline.
///
/// The path's only recorded observation came from re-arming on resume —
/// the model never saw the file's bytes in this session — so a hash
/// compare cannot honestly clear the write: changes made while the
/// session was inactive would pass it unseen. The text states that and
/// directs the model to the same recovery path as a staleness refusal.
fn resumed_baseline_message(path: &Path) -> String {
    format!(
        "{path} was last read in a previous session, so its current bytes \
         are not part of this session's context; not writing until it has \
         been read here.\n\nRead the file with Read, then re-issue the write \
         against the content it returns.",
        path = path.display()
    )
}

/// Format a validator's findings as the message that blocks the write.
///
/// The message is structured so the model can read the finding list and
/// correct its output: the header names the file, each finding is
/// indented on its own line prefixed with `line N:` when the line is
/// known, and the trailing two lines explain why the write did not
/// happen and how to bypass the check if the caller explicitly accepts
/// the risk. Shared by Write and Edit.
pub(crate) fn format_validation_failure(
    path: &Path,
    diagnostics: &[ValidationDiagnostic],
) -> String {
    use std::fmt::Write;
    let mut msg = format!("Syntax validation failed for {}:\n", path.display());
    for diagnostic in diagnostics {
        match diagnostic.line {
            Some(line) => writeln!(msg, "  line {line}: {}", diagnostic.message).ok(),
            None => writeln!(msg, "  {}", diagnostic.message).ok(),
        };
    }
    msg.push_str("Blocked to prevent file corruption.\n");
    msg.push_str("To bypass this check, use skip_linter: true (not recommended).");
    msg
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
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        super::super::FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
        ctx
    }

    /// A validator that rejects any content containing `BAD`.
    ///
    /// Test double for the gate contract: one deterministic finding, so the
    /// blocked-write and bypass paths assert against known text.
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
                        line: Some(1),
                        message: "content contains BAD".to_string(),
                    }]
                } else {
                    Vec::new()
                }
            })
        }
    }

    #[tokio::test]
    async fn writes_a_new_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "notes.txt", "content": "just some text\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(tmp.path().join("notes.txt").exists());
    }

    #[tokio::test]
    async fn no_temp_file_left_on_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        tool.call(
            json!({"file_path": "clean.rs", "content": "fn main() {}\n"}),
            &ctx,
        )
        .await
        .unwrap();
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries, vec!["clean.rs"]);
    }

    #[tokio::test]
    async fn creates_missing_parent_directories() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "sub/dir/new.rs", "content": "fn main() {}\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(tmp.path().join("sub/dir/new.rs").exists());
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("existing.rs");
        std::fs::write(&target, "old content\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "existing.rs", "content": "fn main() {}\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let written = std::fs::read_to_string(&target).unwrap();
        assert_eq!(written, "fn main() {}\n");
        assert!(out.text_content().contains("Changed: existing.rs"));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn write_preserves_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("script.sh");
        std::fs::write(&target, "#!/bin/bash\necho old\n").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();

        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        tool.call(
            json!({"file_path": "script.sh", "content": "#!/bin/bash\necho new\n"}),
            &ctx,
        )
        .await
        .unwrap();

        let mode = std::fs::metadata(&target).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o755,
            "permissions should be preserved as 0o755, got 0o{:o}",
            mode & 0o777
        );
    }

    #[tokio::test]
    async fn missing_file_path_errors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let err = tool.call(json!({"content": "x"}), &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn missing_content_errors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let err = tool
            .call(json!({"file_path": "x.rs"}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn url_file_path_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let err = tool
            .call(
                json!({"file_path": "https://example.com/page", "content": "x"}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("filesystem path")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn absolute_path_honored() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("abs.rs");
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": target.to_str().unwrap(), "content": "fn main() {}\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(target.exists());
    }

    #[tokio::test]
    async fn a_failing_validator_blocks_the_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new().with_validator(Arc::new(RejectBad));
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "bad.rs", "content": "fn main() { BAD }"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.is_error);
        let text = out.text_content();
        assert!(text.contains("Syntax validation failed"), "{text}");
        assert!(text.contains("line 1: content contains BAD"), "{text}");
        assert!(text.contains("use skip_linter: true"), "{text}");
        assert!(!tmp.path().join("bad.rs").exists());
    }

    #[tokio::test]
    async fn skip_linter_bypasses_the_gate() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new().with_validator(Arc::new(RejectBad));
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "bad.rs", "content": "fn main() { BAD }", "skip_linter": true}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(tmp.path().join("bad.rs").exists());
    }

    #[tokio::test]
    async fn a_passing_validator_writes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new().with_validator(Arc::new(RejectBad));
        let ctx = ctx_in(cwd);
        let out = tool
            .call(
                json!({"file_path": "good.rs", "content": "fn main() {}\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(tmp.path().join("good.rs").exists());
    }

    #[tokio::test]
    async fn an_external_change_after_the_read_refuses_the_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("watched.rs");
        std::fs::write(&target, "v1\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = FileSession::new(tmp.path().to_path_buf());
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&target, state::observe_bytes(b"v1\n"));
        std::fs::write(&target, "EXTERNAL\n").unwrap();

        let out = WriteTool::new()
            .call(
                json!({"file_path": "watched.rs", "content": "ours\n"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            out.is_error,
            "a write against a stale baseline must refuse: {}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "EXTERNAL\n",
            "the external content must be untouched"
        );
    }

    #[tokio::test]
    async fn the_models_own_write_never_conflicts() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("twice.rs");
        std::fs::write(&target, "v1\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = FileSession::new(tmp.path().to_path_buf());
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&target, state::observe_bytes(b"v1\n"));

        let first = WriteTool::new()
            .call(json!({"file_path": "twice.rs", "content": "v2\n"}), &ctx)
            .await
            .unwrap();
        assert!(!first.is_error);
        let second = WriteTool::new()
            .call(json!({"file_path": "twice.rs", "content": "v3\n"}), &ctx)
            .await
            .unwrap();
        assert!(
            !second.is_error,
            "the refreshed baseline must clear the model's own follow-up: {}",
            second.text_content()
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "v3\n");
    }

    #[tokio::test]
    async fn a_resumed_baseline_holds_the_write_until_a_live_read() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("resumed.rs");
        std::fs::write(&target, "v1\n").unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let session = FileSession::new(tmp.path().to_path_buf());
        let mut ctx = ToolContext::default();
        ctx.cwd = cwd.to_string();
        session.attach(&mut ctx);
        session.record_baseline(&target, state::observe_resumed_bytes(b"v1\n"));

        let held = WriteTool::new()
            .call(json!({"file_path": "resumed.rs", "content": "v2\n"}), &ctx)
            .await
            .unwrap();
        assert!(held.is_error, "a resume-armed baseline must hold the write");
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
        let after = WriteTool::new()
            .call(json!({"file_path": "resumed.rs", "content": "v2\n"}), &ctx)
            .await
            .unwrap();
        assert!(!after.is_error, "a live read must release the guard");
    }

    #[tokio::test]
    async fn flags_advertise_write_semantics() {
        let tool = WriteTool::new();
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }

    #[tokio::test]
    async fn missing_session_is_a_hard_error() {
        let ctx = ToolContext::default();
        let err = WriteTool::new()
            .call(json!({"file_path": "x.txt", "content": "x"}), &ctx)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("FileSession"),
            "the error must name the missing wiring: {err}"
        );
    }
}
