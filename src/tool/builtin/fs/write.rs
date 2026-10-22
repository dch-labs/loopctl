//! The `Write` tool — writes content to a file, after the validation gate
//! and a staleness check against the path's last-recorded read.

use std::fmt;
use std::future::Future;
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
use super::ValidationDiagnostic;
use super::atomic;
use super::conflict;
use super::conflict::CheckFailure;
use super::diff::OldContent;
use super::diff::format_file_change;
use super::existing;
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
/// escaping the working directory, an existing target that is a
/// directory, an existing target that cannot be opened or read, a
/// target whose only recorded baseline is resume-armed, a target that
/// changed while the write was being prepared, or a file-system error
/// during parent creation or the atomic write.
async fn write_inner(
    tool: &WriteTool,
    input: Value,
    session: &FileSession,
) -> Result<ToolOutput, ToolError> {
    let policy = session.resolve_policy();
    let cwd = session.cwd().to_path_buf();
    let file_path = super::resolve_path_key(&input)
        .ok_or_else(|| super::missing_path_error("Write", &input))?;
    resolve::reject_path_url("Write", file_path)?;
    let content = input
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput("Missing content".to_string()))?;
    let skip_linter = input
        .get("skip_linter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut full_path = session.resolve(file_path)?;
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

    let old_buffer;
    let loaded =
        existing::read_existing_text(&full_path, &cwd, policy, Some(session.anchor())).await?;
    let old_content = match loaded {
        existing::ExistingText::Text(text) => {
            old_buffer = text;
            OldContent::Text(&old_buffer)
        }
        existing::ExistingText::NotUtf8 => OldContent::NotUtf8,
        existing::ExistingText::Absent => OldContent::Absent,
        existing::ExistingText::TooLarge | existing::ExistingText::NonRegular => {
            OldContent::Unreadable
        }
        existing::ExistingText::Directory => {
            return Err(ToolError::Execution(format!(
                "{file_path} is a directory; refusing to write"
            )));
        }
    };

    let mut expected = None;
    if let Some(baseline) = session.baseline_for(&full_path) {
        if baseline.resumed {
            return Ok(ToolOutput::error_text(conflict::resumed_baseline_message(
                &full_path,
            )));
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

    let message = format_file_change(file_path, old_content, content);

    Ok(ToolOutput::text(message)
        .with_hint(DisplayHint::Diff)
        .with_retention(Retention::Durable))
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
    async fn overwriting_a_non_utf8_file_is_labeled_not_created() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("blob.bin");
        std::fs::write(&target, [0xFF, 0xFE]).unwrap();
        let ctx = ctx_in(tmp.path().to_str().unwrap());
        let out = WriteTool::new()
            .call(json!({"file_path": "blob.bin", "content": "text\n"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(
            out.text_content().starts_with("Changed: blob.bin"),
            "an existing file must never render as created: {}",
            out.text_content()
        );
        assert!(
            out.text_content().contains("previous content not UTF-8"),
            "{}",
            out.text_content()
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "text\n",
            "the new content must still land"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_existing_target_refuses_the_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("blocked.d")).unwrap();
        let ctx = ctx_in(tmp.path().to_str().unwrap());
        let err = WriteTool::new()
            .call(json!({"file_path": "blocked.d", "content": "x\n"}), &ctx)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ToolError::Execution(msg)
                if msg.contains("blocked.d is a directory")),
            "a directory target refuses the write naming the file: {err}"
        );
        let entries: Vec<_> = std::fs::read_dir(tmp.path().join("blocked.d"))
            .unwrap()
            .collect();
        assert!(
            entries.is_empty(),
            "the unreadable target must survive untouched, with no temp file inside"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unopenable_existing_target_refuses_the_write() {
        use std::os::unix::fs::PermissionsExt;
        // SAFETY: `geteuid` reads process state and touches no memory.
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("secret.txt");
        std::fs::write(&target, "secrets\n").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o000)).unwrap();
        let ctx = ctx_in(tmp.path().to_str().unwrap());
        let err = WriteTool::new()
            .call(json!({"file_path": "secret.txt", "content": "x\n"}), &ctx)
            .await
            .unwrap_err();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(
            matches!(&err, ToolError::Execution(msg)
                if msg.starts_with("cannot open existing target") && msg.contains("secret.txt")),
            "an open fault must refuse the write naming the file: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "secrets\n",
            "the unopenable file's bytes must be untouched"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_swapped_workspace_spelling_cannot_redirect_contained_writes() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::TempDir::new().unwrap();
        let ws = parent.path().join("ws");
        std::fs::create_dir(&ws).unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "OUTSIDE\n").unwrap();
        let ctx = ctx_in(ws.to_str().unwrap());

        std::fs::remove_dir(&ws).unwrap();
        symlink(outside.path(), &ws).unwrap();

        let err = WriteTool::new()
            .call(
                json!({"file_path": "secret.txt", "content": "ours\n"}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("escaped"),
            "the write must judge against the pinned root, not the swapped spelling: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
            "OUTSIDE\n",
            "nothing may land in the swapped-in tree"
        );
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

    #[tokio::test]
    async fn a_write_receipt_is_stamped_durable() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let tool = WriteTool::new();
        let out = tool
            .call(
                json!({"file_path": "new.rs", "content": "fn main() {}"}),
                &ctx_in(cwd),
            )
            .await
            .expect("the write succeeds");
        assert_eq!(
            out.retention,
            Some(crate::tool::Retention::Durable),
            "a write receipt is a side-effect record — it must survive compaction verbatim"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_fifo_write_target_is_replaced_without_hanging() {
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

        let out = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            WriteTool::new().call(
                json!({"file_path": "pipe", "content": "replacement"}),
                &ctx_in(tmp.path().to_str().unwrap()),
            ),
        )
        .await
        .expect("the gated pre-read resolves instead of parking on the fifo")
        .unwrap();
        assert!(
            out.text_content().contains("Changed:"),
            "the replacement reports as a change: {}",
            out.text_content()
        );
        assert!(
            fifo.is_file(),
            "the atomic write replaced the fifo with a regular file"
        );
        assert_eq!(
            std::fs::read_to_string(&fifo).unwrap(),
            "replacement",
            "the written bytes are the tool's content"
        );
    }

    #[tokio::test]
    async fn an_oversized_target_writes_with_the_unreadable_header_and_no_diff() {
        let cap = crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES;
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("big.log");
        let oversized = vec![b'x'; usize::try_from(cap).unwrap() + 1];
        std::fs::write(&target, oversized).unwrap();
        let cwd = tmp.path().to_str().unwrap();

        let out = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            WriteTool::new().call(
                json!({"file_path": "big.log", "content": "fresh start"}),
                &ctx_in(cwd),
            ),
        )
        .await
        .expect("the capped pre-read resolves")
        .unwrap();
        let text = out.text_content();
        assert!(
            text.contains("Changed: big.log") && text.contains("previous content not read"),
            "the oversized target reports the unreadable header: {text}"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "fresh start",
            "the write itself lands"
        );
    }

    #[tokio::test]
    async fn a_wholesale_overwrite_of_a_large_file_shows_no_echo() {
        // Two interleaved halves of filler share no line that occurs once on
        // each side, so nothing anchors an alignment and no hunk could honestly
        // address the middle. The old answer was the Before/After echo — up to
        // 1 000 lines a side of a file the write had already replaced — and the
        // reason it existed (inventing line numbers is worse than showing
        // nothing) still holds. What changed is that showing nothing is now what
        // the receipt does, in two lines, instead of echoing the file.
        let tmp = tempfile::TempDir::new().unwrap();
        let before: String = (0..1200).map(|_| "a\nb\n").collect();
        std::fs::write(tmp.path().join("data.txt"), &before).unwrap();
        let after: String = (0..1200).map(|_| "b\na\n").collect();
        let cwd = tmp.path().to_str().unwrap();

        let out = WriteTool::new()
            .call(
                json!({"file_path": "data.txt", "content": after}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(
            text.contains("Changed: data.txt (modified, change not shown)"),
            "{text}"
        );
        assert!(
            !text.contains("Before:") && !text.contains("After:"),
            "a write result is a receipt, not a copy of the file it wrote: {text}"
        );
        assert_eq!(
            text.lines().count(),
            2,
            "header and the reason there are no hunks: {text}"
        );
        assert!(
            !text.contains("@@"),
            "no hunk may address lines the diff never reached: {text}"
        );
    }

    #[tokio::test]
    async fn an_at_cap_target_still_diffs_normally() {
        let cap = crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES;
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("edge.log");
        std::fs::write(&target, vec![b'x'; usize::try_from(cap).unwrap()]).unwrap();
        let cwd = tmp.path().to_str().unwrap();

        let out = WriteTool::new()
            .call(
                json!({"file_path": "edge.log", "content": "edge written"}),
                &ctx_in(cwd),
            )
            .await
            .unwrap();
        let text = out.text_content();
        assert!(
            text.contains("Changed: edge.log") && !text.contains("previous content not read"),
            "a target at exactly the cap takes the ordinary diff path: {text}"
        );
    }

    #[tokio::test]
    async fn write_accepts_path_where_file_path_was_expected() {
        // The corpus drift: models send `path` to Write (68 of 69
        // recorded failures exactly this shape).
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let input = serde_json::json!({
            "path": "out.txt",
            "content": "written through the alias",
        });
        let output = write_inner(&WriteTool::default(), input, &session)
            .await
            .expect("the alias spelling writes");
        assert!(
            output.text_content().contains("out.txt"),
            "the write landed: {}",
            output.text_content()
        );
        let written = std::fs::read_to_string(dir.path().join("out.txt")).expect("file");
        assert_eq!(written, "written through the alias");
    }

    #[tokio::test]
    async fn write_both_present_resolves_to_the_canonical_file_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let input = serde_json::json!({
            "path": "alias-target.txt",
            "file_path": "canonical-target.txt",
            "content": "body",
        });
        let _ = write_inner(&WriteTool::default(), input, &session)
            .await
            .expect("parses");
        assert!(
            dir.path().join("canonical-target.txt").exists(),
            "the canonical file_path wins when both ride — never last-one-in"
        );
        assert!(
            !dir.path().join("alias-target.txt").exists(),
            "the alias spelling did not also write"
        );
    }

    #[tokio::test]
    async fn a_mistyped_canonical_key_refuses_instead_of_serving_the_alias() {
        // The canonical key was present but mistyped: teaching the
        // type error beats silently serving the alias — Read already
        // refuses this shape, the family follows.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let input = serde_json::json!({
            "file_path": 123,
            "path": "alias-would-write.txt",
            "content": "body",
        });
        let error = write_inner(&WriteTool::default(), input, &session)
            .await
            .expect_err("a present-but-mistyped canonical refuses");
        let message = error.to_string();
        assert!(
            message.contains("file_path must be a string"),
            "the type error names the canonical key: {message}"
        );
        assert!(
            !dir.path().join("alias-would-write.txt").exists(),
            "the alias never served under the mistyped canonical"
        );
    }

    #[tokio::test]
    async fn the_received_keys_echo_is_bounded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let flood_key = "k".repeat(8_192);
        let input = serde_json::json!({
            flood_key.clone(): "v",
            "content": "body",
        });
        let error = write_inner(&WriteTool::default(), input, &session)
            .await
            .expect_err("no path rides");
        let message = error.to_string();
        assert!(
            message.len() < 2 * 1024,
            "a pathological key cannot flood the teaching error: {} bytes",
            message.len()
        );
    }

    #[tokio::test]
    async fn a_mistyped_path_alias_reports_its_type_not_a_miss() {
        // The alias leg mirrors the canonical's teaching: a present
        // non-string `path` is a type error, not "missing file_path" —
        // the model must not be told to resend what it already sent.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let input = serde_json::json!({
            "path": 123,
            "content": "body",
        });
        let error = write_inner(&WriteTool::default(), input, &session)
            .await
            .expect_err("a mistyped alias refuses");
        let message = error.to_string();
        assert!(
            message.contains("path must be a string"),
            "the type error names the alias key: {message}"
        );
        assert!(
            message.contains("number"),
            "the received type rides: {message}"
        );
    }

    #[tokio::test]
    async fn a_sibling_workspace_escape_names_the_boundary_and_class() {
        // The corpus shape: a neighbor repo's context, reached for by
        // absolute path. The rejection keeps its first line and gains
        // the boundary, the class, and the copy-in remedy.
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("member-repo");
        std::fs::create_dir_all(&repo).expect("repo dir");
        let sibling = dir.path().join("AGENTS.md");
        std::fs::write(&sibling, "neighbor context").expect("sibling");
        let session = FileSession::new(repo.clone());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": sibling.to_string_lossy(), "content": "x"}),
            &session,
        )
        .await
        .expect_err("the sibling path escapes");
        let message = error.to_string();
        assert!(
            message.contains("Path escapes the working directory"),
            "the first line is byte-identical: {message}"
        );
        assert!(
            message.contains(&format!(
                "contained to the working directory: {}",
                repo.display()
            )),
            "the boundary the policy holds to is named: {message}"
        );
        assert!(
            message.contains("sibling workspace"),
            "the target class is named: {message}"
        );
        assert!(
            message.contains("copy what you need into the working directory"),
            "the sibling remedy rides: {message}"
        );
    }

    #[tokio::test]
    async fn a_temp_shaped_escape_names_the_scratch_root_when_configured() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf())
            .with_scratch_root(dir.path().join(".dch/tmp"));
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/read_gguf_kv.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path escapes");
        let message = error.to_string();
        assert!(
            message.contains("temp scratch"),
            "the target class is named: {message}"
        );
        assert!(
            message.contains(&format!(
                "scratch root: {}",
                dir.path().join(".dch/tmp").display()
            )),
            "the configured scratch root is the named remedy: {message}"
        );
    }

    #[tokio::test]
    async fn a_temp_shaped_escape_without_a_scratch_root_stays_generic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/read_gguf_kv.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path escapes");
        let message = error.to_string();
        assert!(
            !message.contains("scratch root:"),
            "no root is advertised the host did not configure: {message}"
        );
        assert!(
            message.contains("scratch files belong inside the working directory"),
            "the unconfigured remedy still teaches: {message}"
        );
    }

    #[tokio::test]
    async fn the_unrestricted_remedy_appears_only_when_declared() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/usr/share/doc/libc/notes.txt", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the system path escapes");
        assert!(
            !error.to_string().contains("unrestricted"),
            "an undeclared mode is never advertised: {error}"
        );

        let session = FileSession::new(dir.path().to_path_buf()).with_unrestricted_available();
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/usr/share/doc/libc/notes.txt", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the system path escapes");
        let message = error.to_string();
        assert!(
            message.contains("system or dependency path"),
            "the class is named: {message}"
        );
        assert!(
            message.contains("unrestricted"),
            "the declared mode is the named road: {message}"
        );
    }

    #[tokio::test]
    async fn a_relative_scratch_root_joins_the_working_directory() {
        // The doc's own `.dch/tmp` shape is relative; the builder
        // joins it to the working directory before validation, the
        // same joining `resolve` applies to tool inputs.
        let dir = tempfile::tempdir().expect("tempdir");
        let session =
            FileSession::new(dir.path().to_path_buf()).with_scratch_root(".dch/tmp".to_string());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/probe.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path still escapes");
        let message = error.to_string();
        assert!(
            message.contains(&dir.path().join(".dch/tmp").display().to_string()),
            "the relative root joined and is the named remedy: {message}"
        );
    }

    #[tokio::test]
    async fn a_scratch_root_target_writes_without_ceremony() {
        // The root sits under the working directory, so containment
        // already admits it: a Write into it needs nothing else, and
        // the model can follow its own error's advice.
        let dir = tempfile::tempdir().expect("tempdir");
        let scratch = dir.path().join(".dch/tmp");
        let session = FileSession::new(dir.path().to_path_buf()).with_scratch_root(scratch.clone());
        let out = write_inner(
            &WriteTool::default(),
            serde_json::json!({
                "file_path": scratch.join("probe.py").to_string_lossy(),
                "content": "print('ok')\n"
            }),
            &session,
        )
        .await
        .expect("the scratch root is sanctioned");
        assert!(!out.is_error, "the write lands: {}", out.text_content());
        assert!(
            scratch.join("probe.py").is_file(),
            "the file exists under the scratch root"
        );
    }

    #[tokio::test]
    async fn a_scratch_root_escape_spelling_is_refused() {
        // `cwd/../outside` passes a naive starts_with - the builder's
        // contract honors only a root that stays inside after
        // lexical normalization.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = FileSession::new(dir.path().to_path_buf())
            .with_scratch_root(dir.path().join("../elsewhere"));
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/probe.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path still escapes");
        assert!(
            !error.to_string().contains("scratch root:"),
            "the escape spelling is never named: {error}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_scratch_root_through_a_link_and_dotdot_creates_inside() {
        // The validate/create TOCTOU: `link/../scratch` normalizes
        // inside and passes validation, but creating the *original*
        // spelling walks the outbound symlink. The root that is
        // created and stored is the one that was validated - the
        // normalized destination, inside the working directory.
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        let link = dir.path().join("out-link");
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");
        let session = FileSession::new(dir.path().to_path_buf())
            .with_scratch_root(link.join("..").join("scratch"));
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/probe.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path still escapes");
        let message = error.to_string();
        assert!(
            message.contains(&dir.path().join("scratch").display().to_string()),
            "the named root is the validated inside destination: {message}"
        );
        assert!(
            !outside.path().join("scratch").exists()
                && !outside
                    .path()
                    .parent()
                    .is_some_and(|parent| parent.join("scratch").exists()),
            "nothing is created outside the working directory — not under the \
             link's destination, and not beside it (symlink traversal escapes \
             to the destination's parent, not the destination)"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_scratch_root_through_a_symlink_is_refused() {
        // A symlink below cwd pointing out would let create_dir_all
        // build outside the workspace - the same containment the
        // resolver grades, checked before creating or storing.
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        let link = dir.path().join("out-link");
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");
        let session =
            FileSession::new(dir.path().to_path_buf()).with_scratch_root(link.join("scratch"));
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/probe.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path still escapes");
        assert!(
            !error.to_string().contains("scratch root:"),
            "a root through an outbound symlink is never named: {error}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_escape_classifies_as_a_link_pointing_out() {
        // A spelling inside the workspace whose target is a link out
        // is not traversal: the class names the link and the remedy
        // names a real target inside.
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("secret.txt"), "x").expect("outside file");
        let link = dir.path().join("leak.txt");
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), &link).expect("symlink");
        let session = FileSession::new(dir.path().to_path_buf());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({
                "file_path": link.to_string_lossy(),
                "content": "x"
            }),
            &session,
        )
        .await
        .expect_err("the symlink target escapes");
        let message = error.to_string();
        assert!(
            message.contains("via symlink"),
            "the first line is the pinned symlink arm: {message}"
        );
        assert!(
            message.contains("a symlink inside the working directory points outside it"),
            "the class names the link: {message}"
        );
        assert!(
            message.contains("pass the real path"),
            "the remedy names a real target: {message}"
        );
    }

    #[tokio::test]
    async fn a_scratch_root_outside_the_working_directory_is_ignored() {
        // The gate: only a root at or under the working directory is
        // honored — anything else would advertise a directory the
        // policy refuses.
        let dir = tempfile::tempdir().expect("tempdir");
        let elsewhere = tempfile::tempdir().expect("elsewhere");
        let session = FileSession::new(dir.path().to_path_buf())
            .with_scratch_root(elsewhere.path().to_path_buf());
        let error = write_inner(
            &WriteTool::default(),
            serde_json::json!({"file_path": "/tmp/probe.py", "content": "x"}),
            &session,
        )
        .await
        .expect_err("the temp path still escapes");
        assert!(
            !error.to_string().contains("scratch root:"),
            "the ignored root is never named: {error}"
        );
    }
}
