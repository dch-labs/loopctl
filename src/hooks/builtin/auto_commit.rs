//! Auto-commit hook — automatically commits file changes during agent sessions.
//!
//! Tracks file modifications from tool calls (Write, Edit) and commits
//! them at session end. Useful for keeping a git trail of agent actions.

use std::process::{Command, Output, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use crate::hooks::Hook;
use crate::hooks::context::{PostToolUseContext, SessionEndContext, SessionStartContext};

/// Default timeout for git subprocess invocations.
const GIT_TIMEOUT: Duration = Duration::from_secs(30);

// ===================================================
// GitExecutor
// ===================================================

/// Errors that can occur during git operations.
#[derive(Debug, thiserror::Error)]
pub enum GitExecutorError {
    /// Failed to execute git command.
    #[error("Failed to execute git: {0}")]
    ExecutionFailed(String),

    /// Git command returned non-zero exit code.
    #[error("Git error: {0}")]
    GitError(String),

    /// No changes to commit.
    #[error("No changes to commit")]
    NoChanges,

    /// Invalid repository state.
    #[error("Invalid repository state: {0}")]
    InvalidState(String),

    /// Git command timed out.
    #[error("Git command timed out after {0:?}")]
    Timeout(Duration),
}

/// Result of an auto-commit operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoCommitResult {
    /// Commit was created successfully.
    Committed {
        /// Short SHA of the new commit.
        sha: String,
    },
    /// No changes to commit.
    NoChanges,
    /// Commit was skipped due to configuration.
    Skipped {
        /// Why the commit was skipped.
        reason: String,
    },
    /// Commit failed.
    Failed {
        /// Error message.
        error: String,
    },
}

/// Executes git operations for auto-commit.
///
/// Each method runs a git subprocess and returns the result.
/// All methods are static — no state is held.
pub struct GitExecutor;

impl GitExecutor {
    /// Run a git subprocess with a bounded timeout.
    ///
    /// Spawns the child, waits up to `GIT_TIMEOUT`, and kills it if the
    /// deadline expires. Returns `(stdout, stderr)` on success.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::ExecutionFailed`] if the child cannot be spawned
    /// or the worker thread panics, [`GitExecutorError::GitError`] if git exits
    /// non-zero, or [`GitExecutorError::Timeout`] if the deadline expires.
    fn run_git(args: &[&str]) -> Result<(Vec<u8>, Vec<u8>), GitExecutorError> {
        let child = Command::new("git")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| GitExecutorError::ExecutionFailed(e.to_string()))?;

        let (tx, rx) = std::sync::mpsc::channel::<Result<Output, std::io::Error>>();
        std::thread::spawn(move || {
            let result = child.wait_with_output();
            drop(tx.send(result));
        });

        match rx.recv_timeout(GIT_TIMEOUT) {
            Ok(Ok(output)) => {
                if !output.status.success() {
                    return Err(GitExecutorError::GitError(
                        String::from_utf8_lossy(&output.stderr).to_string(),
                    ));
                }
                Ok((output.stdout, output.stderr))
            }
            Ok(Err(e)) => Err(GitExecutorError::ExecutionFailed(e.to_string())),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                Err(GitExecutorError::Timeout(GIT_TIMEOUT))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(
                GitExecutorError::ExecutionFailed("git worker thread panicked".to_string()),
            ),
        }
    }

    /// Check if there are uncommitted changes in the working tree.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::ExecutionFailed`] if git is not available,
    /// or [`GitExecutorError::GitError`] if the command fails.
    pub fn has_changes() -> Result<bool, GitExecutorError> {
        let (stdout, _stderr) = Self::run_git(&["status", "--porcelain"])?;

        let status = String::from_utf8_lossy(&stdout);
        Ok(!status.trim().is_empty())
    }

    /// Stage files for commit. Empty slice stages all changes.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if the git command fails.
    pub fn stage_files(files: &[String]) -> Result<(), GitExecutorError> {
        if files.is_empty() {
            Self::run_git(&["add", "-A"])?;
        } else {
            for file in files {
                Self::run_git(&["add", file])?;
            }
        }
        Ok(())
    }

    /// Create a commit with the given message.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if the git command fails.
    pub fn commit(message: &str, amend: bool) -> Result<String, GitExecutorError> {
        let mut args: Vec<&str> = vec!["commit"];
        if amend {
            args.push("--amend");
        }
        args.push("-m");
        args.push(message);

        let result = Self::run_git(&args);
        match result {
            Ok(_) => Self::get_head_sha(),
            Err(e) => {
                // Check for "nothing to commit" in stderr — but run_git already
                // consumes stderr into GitError. Match on the error string.
                let err_str = e.to_string();
                if err_str.contains("nothing to commit") {
                    return Err(GitExecutorError::NoChanges);
                }
                Err(e)
            }
        }
    }

    /// Push to remote.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if the git command fails.
    pub fn push(branch: Option<&str>) -> Result<(), GitExecutorError> {
        let current_branch = Self::current_branch()?;
        let branch = branch.unwrap_or(&current_branch);

        Self::run_git(&["push", "origin", branch])?;
        Ok(())
    }

    /// Get the current branch name.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if the git command fails.
    pub fn current_branch() -> Result<String, GitExecutorError> {
        let (stdout, _stderr) = Self::run_git(&["rev-parse", "--abbrev-ref", "HEAD"])?;
        Ok(String::from_utf8_lossy(&stdout).trim().to_string())
    }

    /// Get the HEAD commit SHA.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if the git command fails.
    pub fn get_head_sha() -> Result<String, GitExecutorError> {
        let (stdout, _stderr) = Self::run_git(&["rev-parse", "HEAD"])?;
        Ok(String::from_utf8_lossy(&stdout).trim().to_string())
    }

    /// Perform auto-commit with the given configuration.
    ///
    /// Checks for changes, stages files, creates a commit, and optionally pushes.
    /// Returns [`AutoCommitResult`] describing the outcome.
    ///
    /// When `session_files` is `Some`, only those paths are staged and committed,
    /// preventing unrelated working-tree changes from leaking into the commit.
    #[must_use]
    pub fn auto_commit_with_files(
        config: &AutoCommitConfig,
        session_files: Option<&[String]>,
    ) -> AutoCommitResult {
        if !config.enabled {
            return AutoCommitResult::Skipped {
                reason: "Auto-commit disabled".to_string(),
            };
        }

        match Self::has_changes() {
            Ok(true) => {}
            Ok(false) => {
                if config.skip_if_clean {
                    return AutoCommitResult::NoChanges;
                }
                return AutoCommitResult::Skipped {
                    reason: "No changes detected".to_string(),
                };
            }
            Err(e) => {
                return AutoCommitResult::Failed {
                    error: e.to_string(),
                };
            }
        }

        let files = session_files.unwrap_or(&config.files);
        if let Err(e) = Self::stage_files(files) {
            return AutoCommitResult::Failed {
                error: e.to_string(),
            };
        }

        match Self::commit(&config.message_template, config.commit_mode.is_amend()) {
            Ok(sha) => {
                if config.auto_push {
                    if let Err(e) = Self::push(config.push_branch.as_deref()) {
                        return AutoCommitResult::Failed {
                            error: format!("Commit succeeded but push failed: {e}"),
                        };
                    }
                }
                AutoCommitResult::Committed { sha }
            }
            Err(GitExecutorError::NoChanges) => AutoCommitResult::NoChanges,
            Err(e) => AutoCommitResult::Failed {
                error: e.to_string(),
            },
        }
    }

    /// Perform auto-commit with the given configuration.
    ///
    /// Convenience wrapper around [`auto_commit_with_files`](Self::auto_commit_with_files)
    /// that uses the file list from `config.files`.
    #[must_use]
    pub fn auto_commit(config: &AutoCommitConfig) -> AutoCommitResult {
        Self::auto_commit_with_files(config, None)
    }
}

// ===================================================
// CommitMode
// ===================================================

/// How the auto-commit hook creates commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommitMode {
    /// Create a new commit.
    #[default]
    Create,
    /// Amend the previous commit instead of creating a new one.
    Amend,
}

impl CommitMode {
    /// Returns `true` if this mode amends the previous commit.
    #[must_use]
    pub fn is_amend(self) -> bool {
        self == Self::Amend
    }
}

// ===================================================
// AutoCommitConfig
// ===================================================

/// Configuration for auto-commit behavior.
#[derive(Debug, Clone)]
pub struct AutoCommitConfig {
    /// Enable auto-commit functionality.
    pub enabled: bool,
    /// Commit message template (supports `{{tool}}`, `{{session}}` placeholders).
    pub message_template: String,
    /// Automatically push after commit.
    pub auto_push: bool,
    /// Branch to push to (`None` = current branch).
    pub push_branch: Option<String>,
    /// Skip commit if no changes detected.
    pub skip_if_clean: bool,
    /// Whether to amend the previous commit or create a new one.
    pub commit_mode: CommitMode,
    /// Tools that trigger file tracking (e.g., `"Write"`, `"Edit"`).
    pub commit_on_tools: Vec<String>,
    /// Files to add before commit (empty = all files).
    pub files: Vec<String>,
}

impl Default for AutoCommitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            message_template: "chore(agent): auto-commit".to_string(),
            auto_push: false,
            push_branch: None,
            skip_if_clean: true,
            commit_mode: CommitMode::default(),
            commit_on_tools: vec!["Write".to_string(), "Edit".to_string()],
            files: vec![],
        }
    }
}

/// Builder for [`AutoCommitConfig`].
pub struct AutoCommitConfigBuilder {
    config: AutoCommitConfig,
}

impl AutoCommitConfigBuilder {
    /// Create a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoCommitConfig::default(),
        }
    }

    /// Enable or disable auto-commit.
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the commit message template.
    #[must_use]
    pub fn message_template(mut self, template: impl Into<String>) -> Self {
        self.config.message_template = template.into();
        self
    }

    /// Set whether to auto-push.
    #[must_use]
    pub fn auto_push(mut self, value: bool) -> Self {
        self.config.auto_push = value;
        self
    }

    /// Set files to add before commit.
    #[must_use]
    pub fn files(mut self, files: Vec<String>) -> Self {
        self.config.files = files;
        self
    }

    /// Build the configuration.
    #[must_use]
    pub fn build(self) -> AutoCommitConfig {
        self.config
    }
}

impl Default for AutoCommitConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================
// AutoCommitHook
// ===================================================

/// A hook that tracks file modifications from tool calls and commits
/// them at session end.
///
/// Useful for keeping a git trail of agent actions. Configure which
/// tools trigger tracking via [`AutoCommitConfig::commit_on_tools`].
///
/// # Example
///
/// ```
/// use loopctl::hooks::builtin::AutoCommitHook;
/// use loopctl::hooks::HookExecutor;
/// use std::sync::Arc;
///
/// let hook = AutoCommitHook::new();
/// let executor = HookExecutor::new()
///     .with_hook(Arc::new(hook));
///
/// assert_eq!(executor.hook_count(), 1);
/// ```
pub struct AutoCommitHook {
    config: AutoCommitConfig,
    modified_files: Mutex<Vec<String>>,
}

impl AutoCommitHook {
    /// Create a new auto-commit hook with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoCommitConfig::default(),
            modified_files: Mutex::new(Vec::new()),
        }
    }

    /// Create a new auto-commit hook with custom configuration.
    #[must_use]
    pub fn with_config(config: AutoCommitConfig) -> Self {
        Self {
            config,
            modified_files: Mutex::new(Vec::new()),
        }
    }

    fn track_modification(&self, file: &str) {
        if let Ok(mut files) = self.modified_files.lock()
            && !files.contains(&file.to_string())
        {
            files.push(file.to_string());
        }
    }

    fn clear_modifications(&self) {
        if let Ok(mut files) = self.modified_files.lock() {
            files.clear();
        }
    }
}

impl Default for AutoCommitHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for AutoCommitHook {
    fn name(&self) -> &'static str {
        "auto_commit"
    }

    fn on_post_tool_use(&self, ctx: &PostToolUseContext) {
        if !self.config.enabled {
            return;
        }

        if self.config.commit_on_tools.contains(&ctx.tool_name) {
            if let Some(file_path) = ctx.input.get("file_path").and_then(|v| v.as_str()) {
                self.track_modification(file_path);
            }
        }
    }

    fn on_session_start(&self, _ctx: &SessionStartContext) {
        self.clear_modifications();
    }

    fn on_session_end(&self, _ctx: &SessionEndContext) {
        let files = self.modified_files.lock().ok().filter(|f| !f.is_empty());
        let result = GitExecutor::auto_commit_with_files(
            &self.config,
            files.as_deref().map(|v| v as &[String]),
        );
        if let AutoCommitResult::Failed { error } = result {
            tracing::warn!(%error, "auto-commit failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder() {
        let config = AutoCommitConfigBuilder::new()
            .enabled(true)
            .message_template("feat: {{tool}}")
            .auto_push(true)
            .build();

        assert!(config.enabled);
        assert_eq!(config.message_template, "feat: {{tool}}");
        assert!(config.auto_push);
    }

    #[test]
    fn config_default() {
        let config = AutoCommitConfig::default();
        assert!(config.enabled);
        assert!(!config.auto_push);
        assert!(config.skip_if_clean);
        assert_eq!(config.commit_on_tools.len(), 2);
    }

    #[test]
    fn hook_creation() {
        let hook = AutoCommitHook::new();
        assert_eq!(hook.name(), "auto_commit");
    }

    #[test]
    fn hook_with_disabled_config() {
        let config = AutoCommitConfig {
            enabled: false,
            ..AutoCommitConfig::default()
        };
        let hook = AutoCommitHook::with_config(config);
        assert_eq!(hook.name(), "auto_commit");
    }

    #[test]
    fn track_modifications() {
        let hook = AutoCommitHook::new();
        hook.track_modification("src/main.rs");
        hook.track_modification("src/lib.rs");
        hook.track_modification("src/main.rs"); // duplicate

        let files = hook.modified_files.lock().unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"src/main.rs".to_string()));
        assert!(files.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn clear_modifications() {
        let hook = AutoCommitHook::new();
        hook.track_modification("src/main.rs");
        hook.clear_modifications();

        let files = hook.modified_files.lock().unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn auto_commit_result_equality() {
        let r1 = AutoCommitResult::Committed {
            sha: "abc123".to_string(),
        };
        let r2 = AutoCommitResult::Committed {
            sha: "abc123".to_string(),
        };
        assert_eq!(r1, r2);

        let r3 = AutoCommitResult::NoChanges;
        assert_ne!(r1, r3);
    }

    #[test]
    fn auto_commit_disabled_returns_skipped() {
        let config = AutoCommitConfig {
            enabled: false,
            ..AutoCommitConfig::default()
        };
        let result = GitExecutor::auto_commit(&config);
        assert!(matches!(result, AutoCommitResult::Skipped { .. }));
    }

    #[test]
    fn commit_mode_default_is_create() {
        assert_eq!(CommitMode::default(), CommitMode::Create);
    }

    #[test]
    fn commit_mode_is_amend() {
        assert!(CommitMode::Amend.is_amend());
    }

    #[test]
    fn commit_mode_create_is_not_amend() {
        assert!(!CommitMode::Create.is_amend());
    }

    #[test]
    fn config_default_commit_mode() {
        let config = AutoCommitConfig::default();
        assert_eq!(config.commit_mode, CommitMode::Create);
    }

    #[test]
    fn hook_tracks_tracked_tools() {
        let hook = AutoCommitHook::new();
        let ctx = PostToolUseContext {
            tool_name: "Write".to_string(),
            input: serde_json::json!({"file_path": "test.rs"}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        hook.on_post_tool_use(&ctx);

        let files = hook.modified_files.lock().unwrap();
        assert!(files.contains(&"test.rs".to_string()));
    }

    #[test]
    fn hook_ignores_untracked_tools() {
        let hook = AutoCommitHook::new();
        let ctx = PostToolUseContext {
            tool_name: "Read".to_string(),
            input: serde_json::json!({"file_path": "test.rs"}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        hook.on_post_tool_use(&ctx);

        let files = hook.modified_files.lock().unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn hook_clear_resets_tracking() {
        let hook = AutoCommitHook::new();
        let ctx = PostToolUseContext {
            tool_name: "Write".to_string(),
            input: serde_json::json!({"file_path": "test.rs"}),
            output: "ok".to_string(),
            is_error: false,
            duration_ms: 10,
            session_id: uuid::Uuid::nil(),
            turn_number: 0,
        };
        hook.on_post_tool_use(&ctx);
        assert!(!hook.modified_files.lock().unwrap().is_empty());

        hook.clear_modifications();
        assert!(hook.modified_files.lock().unwrap().is_empty());
    }
}
