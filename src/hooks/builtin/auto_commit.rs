//! Auto-commit hook — automatically commits file changes during agent sessions.
//!
//! Tracks file modifications from tool calls (Write, Edit) and commits
//! them at run end. Useful for keeping a git trail of agent actions.
//!
//! # How it fits together
//!
//! 1. [`AutoCommitHook`] observes post-tool-use events, recording each
//!    `file_path` touched by the configured tracking tools.
//! 2. At run end the hook hands the recorded paths (plus the
//!    [`AutoCommitConfig`]) to [`GitExecutor`], which shells out to
//!    `git` to stage, commit, and optionally push.
//! 3. [`AutoCommitResult`] describes the outcome so callers can log
//!    failures without inspecting git's stderr.

use std::process::{Command, Output, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use crate::hooks::Hook;
use crate::hooks::context::{PostToolUseContext, RunEndContext, RunStartContext};

/// Default timeout for git subprocess invocations.
///
/// Applied to every `git` call made by [`GitExecutor`] so a stuck git
/// process (for example a hung GPG-signing prompt or an unreachable
/// remote during a fetch-adjacent operation) cannot block the agent
/// loop indefinitely. After this elapses the child is killed and the
/// call surfaces as [`GitExecutorError::Timeout`].
const GIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Errors that can occur during git operations.
///
/// Every `git` invocation made by [`GitExecutor`] returns `Result<_,
/// GitExecutorError>`. The variants distinguish the failure modes that
/// callers are likely to want to react to differently — a missing
/// repository, an empty commit, or a hung process — while folding the
/// remainder into [`GitExecutorError::GitError`] with git's own stderr
/// attached for diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum GitExecutorError {
    /// The `git` binary could not be spawned or the worker thread panicked.
    ///
    /// Typically means `git` is not on `PATH`, the working directory is
    /// inaccessible, or the OS refused to create the subprocess. The
    /// carried string is the underlying `std::io::Error` rendering.
    #[error("Failed to execute git: {0}")]
    ExecutionFailed(String),

    /// Git exited with a non-zero status code.
    ///
    /// The carried string is git's own `stderr` output, which usually
    /// identifies the failure (merge conflict, lock file held, bad
    /// ref). Inspect it before deciding whether a retry is worthwhile.
    #[error("Git error: {0}")]
    GitError(String),

    /// There is nothing to commit.
    ///
    /// Returned when `git commit` refuses to create a commit because
    /// the index is empty or the staged changes are identical to
    /// `HEAD`. [`AutoCommitConfig::skip_if_clean`] governs whether this
    /// is treated as a silent no-op or surfaced to the caller.
    #[error("No changes to commit")]
    NoChanges,

    /// The repository is in a state that prevents the operation.
    ///
    /// For example a rebase in progress, a detached `HEAD`, or a
    /// missing upstream. The carried string describes the specific
    /// condition git reported.
    #[error("Invalid repository state: {0}")]
    InvalidState(String),

    /// The git subprocess exceeded the configured timeout deadline.
    ///
    /// The carried [`Duration`] echoes the elapsed timeout so callers
    /// can correlate it with their own deadline tracking. The child
    /// process has been killed by the time this is returned.
    #[error("Git command timed out after {0:?}")]
    Timeout(Duration),
}

/// Result of an auto-commit operation.
///
/// Returned by [`GitExecutor::auto_commit`] and
/// [`GitExecutor::auto_commit_with_files`] to describe the outcome
/// without surfacing raw git output. Callers typically match on this
/// enum to decide whether to log the run as informational (a clean
/// working tree), as a silent success, or as a warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoCommitResult {
    /// A commit was created successfully.
    ///
    /// The carried SHA lets the caller record the commit (for example
    /// to attribute it to the session) without re-running `git`.
    Committed {
        /// SHA of the newly created commit.
        ///
        /// The full `HEAD` object name as returned by
        /// `git rev-parse HEAD`, trimmed of surrounding whitespace.
        /// Suitable for linking from session logs or PR descriptions.
        sha: String,
    },

    /// There was nothing to commit.
    ///
    /// The working tree was clean (or the staged set matched `HEAD`).
    /// This is a successful no-op, not an error —
    /// [`AutoCommitConfig::skip_if_clean`] decides whether the clean
    /// state was a deliberate skip or an explicit no-op.
    NoChanges,

    /// The commit was skipped due to configuration.
    ///
    /// Emitted when auto-commit is disabled
    /// ([`AutoCommitConfig::enabled`] is `false`) or when
    /// `skip_if_clean` is `false` and no changes were found. The
    /// carried reason explains which condition applied.
    Skipped {
        /// Human-readable reason the commit was skipped.
        ///
        /// Short diagnostic string such as `"Auto-commit disabled"` or
        /// `"No changes detected"`. Intended for log output rather than
        /// programmatic matching.
        reason: String,
    },

    /// The commit could not be created.
    ///
    /// One of the underlying git operations failed; the carried
    /// message is the [`GitExecutorError`] rendering. The working tree
    /// is left in whatever partial state git reached (staged but not
    /// committed, committed but not pushed, etc.).
    Failed {
        /// Description of the failure.
        ///
        /// The full [`GitExecutorError`] message — typically git's own
        /// stderr or an IO failure description. Use it for logging and
        /// user-facing diagnostics.
        error: String,
    },
}

/// Executes git operations for auto-commit.
///
/// A zero-sized type whose methods each shell out to `git` and return
/// the result. No state is held between calls, so every invocation
/// starts a fresh subprocess; this keeps the executor trivially
/// thread-safe at the cost of one `git` process per operation.
pub struct GitExecutor;

impl GitExecutor {
    /// Run a git subprocess with a bounded timeout.
    ///
    /// Spawns the child, waits up to the configured timeout, and kills it if
    /// the deadline expires. Returns `(stdout, stderr)` on success.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::ExecutionFailed`] if the child
    /// cannot be spawned or the worker thread panics,
    /// [`GitExecutorError::GitError`] if git exits non-zero, or
    /// [`GitExecutorError::Timeout`] if the deadline expires.
    fn run_git(args: &[&str]) -> Result<(Vec<u8>, Vec<u8>), GitExecutorError> {
        let child = Command::new("git")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| GitExecutorError::ExecutionFailed(e.to_string()))?;

        let pid = child.id();

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
                let _ = Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .ok();
                let _ = rx.recv().ok();
                Err(GitExecutorError::Timeout(GIT_TIMEOUT))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(
                GitExecutorError::ExecutionFailed("git worker thread panicked".to_string()),
            ),
        }
    }

    /// Check whether the working tree has uncommitted changes.
    ///
    /// Runs `git status --porcelain` and returns `true` when any output
    /// is produced. Use this to short-circuit a commit attempt before
    /// staging, avoiding the overhead of a `git commit` invocation that
    /// would refuse to run anyway.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::ExecutionFailed`] if git is not
    /// available, or [`GitExecutorError::GitError`] if the command
    /// fails (for example, outside a repository).
    pub fn has_changes() -> Result<bool, GitExecutorError> {
        let (stdout, _stderr) = Self::run_git(&["status", "--porcelain"])?;
        let status = String::from_utf8_lossy(&stdout);
        Ok(!status.trim().is_empty())
    }

    /// Stage files for commit.
    ///
    /// Each path in `files` is passed to `git add -- <path>` individually,
    /// scoping staging to exactly the listed paths. An empty slice is
    /// refused with an error rather than falling through to `git add -A`:
    /// the auto-commit hook is meant to commit only the agent's own
    /// recorded modifications, so staging the entire working tree would
    /// silently sweep up unrelated edits and secret files (`.env`,
    /// credentials, scratch buffers). Misconfiguration must fail loudly,
    /// not broaden scope.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::GitError`] if `files` is empty —
    /// populate `AutoCommitConfig::files` or record modifications through
    /// the hook. Returns [`GitExecutorError`] if any individual `git add`
    /// invocation fails; earlier files in the list may already be staged.
    pub fn stage_files(files: &[String]) -> Result<(), GitExecutorError> {
        if files.is_empty() {
            return Err(GitExecutorError::GitError(
                "refusing to stage with empty file list (would run `git add -A` \
                 and commit unrelated changes); populate `AutoCommitConfig::files` \
                 or record modifications via the hook"
                    .to_string(),
            ));
        }
        for file in files {
            Self::run_git(&["add", "--", file])?;
        }
        Ok(())
    }

    /// Create a commit with the given message.
    ///
    /// Runs `git commit` (optionally with `--amend` when `amend` is
    /// `true`) and returns the SHA of the resulting commit. The
    /// "nothing to commit" case is mapped to
    /// [`GitExecutorError::NoChanges`] so callers can distinguish it
    /// from a genuine failure without parsing git's stderr.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError::NoChanges`] if there is nothing to
    /// commit, or any other [`GitExecutorError`] variant on failure.
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

    /// Push the current branch to the configured remote.
    ///
    /// Resolves the branch from `branch` when given, otherwise from the
    /// currently checked-out branch via [`current_branch`](Self::current_branch).
    /// The push targets `origin` and is non-forceful — it will fail if
    /// the remote has diverged.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if branch resolution or the push
    /// itself fails (for example, no upstream configured or
    /// non-fast-forward).
    pub fn push(branch: Option<&str>) -> Result<(), GitExecutorError> {
        let current_branch = Self::current_branch()?;
        let branch = branch.unwrap_or(&current_branch);

        Self::run_git(&["push", "origin", branch])?;
        Ok(())
    }

    /// Get the currently checked-out branch name.
    ///
    /// Runs `git rev-parse --abbrev-ref HEAD` and trims whitespace.
    /// Returns `"HEAD"` when in a detached-HEAD state (git's own
    /// convention).
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if git is unavailable or not inside
    /// a repository.
    pub fn current_branch() -> Result<String, GitExecutorError> {
        let (stdout, _stderr) = Self::run_git(&["rev-parse", "--abbrev-ref", "HEAD"])?;
        Ok(String::from_utf8_lossy(&stdout).trim().to_string())
    }

    /// Get the full SHA of the current `HEAD` commit.
    ///
    /// Runs `git rev-parse HEAD` and trims whitespace. Returns the full
    /// object name (40 characters for SHA-1) rather than the short form.
    ///
    /// # Errors
    ///
    /// Returns [`GitExecutorError`] if git is unavailable or not inside
    /// a repository.
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
                if config.auto_push
                    && let Err(e) = Self::push(config.push_branch.as_deref())
                {
                    return AutoCommitResult::Failed {
                        error: format!("Commit succeeded but push failed: {e}"),
                    };
                }
                AutoCommitResult::Committed { sha }
            }
            Err(GitExecutorError::NoChanges) => AutoCommitResult::NoChanges,
            Err(e) => AutoCommitResult::Failed {
                error: e.to_string(),
            },
        }
    }

    /// Perform auto-commit using the file list from the config.
    ///
    /// Convenience wrapper around
    /// [`auto_commit_with_files`](Self::auto_commit_with_files) that
    /// passes `None` for the session-files override, so
    /// [`AutoCommitConfig::files`] controls what gets staged.
    #[must_use]
    pub fn auto_commit(config: &AutoCommitConfig) -> AutoCommitResult {
        Self::auto_commit_with_files(config, None)
    }
}

/// How the auto-commit hook creates commits.
///
/// Selected via [`AutoCommitConfig::commit_mode`] to control whether a
/// session's edits produce a fresh commit or fold into the previous
/// one. The default is [`CommitMode::Create`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommitMode {
    /// Create a new commit on top of `HEAD`.
    ///
    /// The standard git behavior: each auto-commit produces a distinct
    /// history entry. Choose this when you want a per-session audit
    /// trail in the commit log.
    #[default]
    Create,

    /// Amend the previous commit instead of creating a new one.
    ///
    /// Runs `git commit --amend`, folding the new changes into `HEAD`.
    /// Useful for squashing a session's worth of edits into a single
    /// history entry, but rewrites the previous commit's SHA — avoid
    /// on branches that may already be shared.
    Amend,
}

impl CommitMode {
    /// Returns `true` if this mode amends the previous commit.
    ///
    /// Convenience predicate used by [`GitExecutor::commit`] to decide
    /// whether to pass `--amend` to `git commit`. Equivalent to
    /// comparing against [`CommitMode::Amend`].
    #[must_use]
    pub fn is_amend(self) -> bool {
        self == Self::Amend
    }
}

/// Configuration for auto-commit behavior.
///
/// Bundles every tunable that [`GitExecutor`] and [`AutoCommitHook`]
/// consult: whether to run at all, which tools to track, what to stage,
/// how to phrase the message, and whether to push afterwards. Build
/// with [`AutoCommitConfigBuilder`] or construct directly —
/// [`Default`] supplies sensible values for a local-only workflow.
#[derive(Debug, Clone)]
pub struct AutoCommitConfig {
    /// Enable auto-commit functionality.
    ///
    /// Master switch for the whole hook. When `false`, no file tracking or
    /// git operations occur regardless of the other fields.
    pub enabled: bool,

    /// Commit message template (supports `{{tool}}`, `{{session}}` placeholders).
    ///
    /// Template string used to build the commit message; `{{tool}}` and
    /// `{{session}}` are expanded at run end into the triggering tool
    /// name and the session identifier respectively.
    pub message_template: String,

    /// Automatically push after commit.
    ///
    /// When `true`, a successful commit is immediately followed by a `git
    /// push` to the configured (or current) branch. When `false` commits
    /// stay local.
    pub auto_push: bool,

    /// Branch to push to (`None` = current branch).
    ///
    /// Explicit remote branch name passed to `git push`. `None` pushes to the
    /// currently checked-out branch, which is usually what you want.
    pub push_branch: Option<String>,

    /// Skip commit if no changes detected.
    ///
    /// When `true` a clean working tree is treated as a no-op
    /// ([`AutoCommitResult::NoChanges`]) rather than a skipped run, so the
    /// hook stays quiet during sessions that made no file changes.
    pub skip_if_clean: bool,

    /// Whether to amend the previous commit or create a new one.
    ///
    /// [`CommitMode::Amend`] folds new changes into the previous commit
    /// instead of producing a separate one, useful for squashing a session's
    /// edits into a single history entry.
    pub commit_mode: CommitMode,

    /// Tools that trigger file tracking (e.g., `"Write"`, `"Edit"`).
    ///
    /// Tool names whose output should be watched for a `file_path`; when one
    /// of these tools runs, the touched file is recorded and staged at
    /// run end.
    pub commit_on_tools: Vec<String>,

    /// Files to add before commit.
    ///
    /// Explicit allow-list of paths passed to `git add`. When the hook
    /// records no per-session modifications, this list is what gets
    /// staged. An empty list is **not** a "commit everything" wildcard:
    /// [`stage_files`](GitExecutor::stage_files) refuses it with an error
    /// rather than running `git add -A`, so the default configuration
    /// commits nothing instead of sweeping up unrelated or secret files.
    /// Populate this list, or rely on the hook's per-session modification
    /// tracking, to stage exactly the agent's own changes.
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
///
/// Provides a fluent API for constructing a config without naming every
/// field. Each `with_*` method overrides one setting; call
/// [`build`](Self::build) to finish. Omitted fields keep their
/// [`AutoCommitConfig::default`] values.
pub struct AutoCommitConfigBuilder {
    /// The in-progress configuration, mutated by each `with_*` call.
    config: AutoCommitConfig,
}

impl AutoCommitConfigBuilder {
    /// Create a new builder seeded with default configuration.
    ///
    /// Equivalent to [`AutoCommitConfigBuilder::default`]; every field
    /// starts at its [`AutoCommitConfig::default`] value.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoCommitConfig::default(),
        }
    }

    /// Enable or disable auto-commit.
    ///
    /// Mirrors [`AutoCommitConfig::enabled`]; `false` suppresses all
    /// file tracking and git operations.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the commit message template.
    ///
    /// Accepts any `Into<String>` so literals work directly. The
    /// template may include `{{tool}}` and `{{session}}` placeholders
    /// expanded at run end.
    #[must_use]
    pub fn with_message_template(mut self, template: impl Into<String>) -> Self {
        self.config.message_template = template.into();
        self
    }

    /// Set whether commits are followed by a `git push`.
    ///
    /// Mirrors [`AutoCommitConfig::auto_push`]; `true` enables
    /// automatic pushing after each successful commit.
    #[must_use]
    pub fn with_auto_push(mut self, value: bool) -> Self {
        self.config.auto_push = value;
        self
    }

    /// Set the explicit file allow-list to stage before commit.
    ///
    /// Mirrors [`AutoCommitConfig::files`]; an empty vector stages every
    /// change in the working tree.
    #[must_use]
    pub fn with_files(mut self, files: Vec<String>) -> Self {
        self.config.files = files;
        self
    }

    /// Finalize and return the built configuration.
    ///
    /// Consumes the builder. The returned [`AutoCommitConfig`] is ready
    /// to pass to [`GitExecutor`] or [`AutoCommitHook::with_config`].
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

/// A hook that tracks file modifications from tool calls and commits
/// them at run end.
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
    /// Configuration consulted at each lifecycle callback.
    ///
    /// Set via [`new`](Self::new) (defaults) or
    /// [`with_config`](Self::with_config) and held immutably for the
    /// hook's lifetime. Controls which tools trigger tracking, the
    /// commit message, and whether to push.
    config: AutoCommitConfig,

    /// File paths recorded by tracked tools this session.
    ///
    /// A deduplicated, insertion-ordered list updated by
    /// [`track_modification`](Self::track_modification) and cleared on
    /// [`clear_modifications`](Self::clear_modifications). Guarded by a
    /// [`Mutex`] so the hook can be shared across threads via
    /// [`HookExecutor`](crate::hooks::HookExecutor).
    modified_files: Mutex<Vec<String>>,
}

impl AutoCommitHook {
    /// Create a new auto-commit hook with default configuration.
    ///
    /// Equivalent to [`AutoCommitHook::default`]; the hook starts with
    /// an empty modification list and [`AutoCommitConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoCommitConfig::default(),
            modified_files: Mutex::new(Vec::new()),
        }
    }

    /// Replace this hook's configuration.
    ///
    /// Builder-style setter intended for use during hook assembly. The
    /// existing modification list is preserved so reconfiguring
    /// mid-session does not lose already-tracked files.
    #[must_use]
    pub fn with_config(mut self, config: AutoCommitConfig) -> Self {
        self.config = config;
        self
    }

    /// Record a file path touched by a tracked tool.
    ///
    /// Deduplicates against the existing list — calling twice with the
    /// same path records it only once. Lock failures are silently
    /// ignored so a poisoned mutex cannot block the agent loop.
    fn track_modification(&self, file: &str) {
        if let Ok(mut files) = self.modified_files.lock()
            && !files.contains(&file.to_string())
        {
            files.push(file.to_string());
        }
    }

    /// Clear all recorded file modifications.
    ///
    /// Called at run start so a fresh run does not inherit
    /// modifications from the previous one. Lock failures are silently
    /// ignored.
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

        if self.config.commit_on_tools.contains(&ctx.tool_name)
            && let Some(file_path) = ctx.input.get("file_path").and_then(|v| v.as_str())
        {
            self.track_modification(file_path);
        }
    }

    fn on_run_start(&self, _ctx: &RunStartContext) {
        self.clear_modifications();
    }

    fn on_run_end(&self, _ctx: &RunEndContext) {
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
            .with_enabled(true)
            .with_message_template("feat: {{tool}}")
            .with_auto_push(true)
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
        let hook = AutoCommitHook::new().with_config(config);
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

    /// An empty file list must be refused, not staged as `git add -A`.
    ///
    /// Reproduces the footgun: a real git repo with a tracked file that
    /// has an uncommitted modification, then `stage_files(&[])`. The
    /// fixed code returns `Err` and leaves the file unstaged; the buggy
    /// code ran `git add -A` and staged the unrelated change.
    struct RepoGuard(std::path::PathBuf);
    impl Drop for RepoGuard {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn stage_files_empty_list_does_not_stage_unrelated_changes() {
        use std::fs;
        use std::process::Command;

        if Command::new("git")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("skipping: git not on PATH");
            return;
        }

        let mut repo_dir = std::env::temp_dir();
        repo_dir.push(format!(
            "loopctl-stage-files-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        fs::create_dir_all(&repo_dir).expect("created temp repo dir");
        let guard = RepoGuard(repo_dir.clone());
        let prev_dir = std::env::current_dir().expect("cwd readable");

        for (label, args) in [
            ("init", vec!["init"]),
            ("config user.name", vec!["config", "user.name", "test"]),
            ("config user.email", vec!["config", "user.email", "t@t"]),
        ] {
            let status = Command::new("git")
                .args(&args)
                .current_dir(&repo_dir)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap_or_else(|e| panic!("git {label} spawn failed: {e}"));
            assert!(status.success(), "git {label} failed");
        }

        let sentinel = repo_dir.join("unrelated.txt");
        fs::write(&sentinel, "v1\n").expect("wrote sentinel v1");
        for (label, args) in [
            ("add", vec!["add", "unrelated.txt"]),
            ("commit", vec!["commit", "-m", "init"]),
        ] {
            let status = Command::new("git")
                .args(&args)
                .current_dir(&repo_dir)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap_or_else(|e| panic!("git {label} spawn failed: {e}"));
            assert!(status.success(), "git {label} failed");
        }

        fs::write(&sentinel, "v2-uncommitted\n").expect("wrote sentinel v2");

        std::env::set_current_dir(&repo_dir).expect("cd into repo");
        let result = GitExecutor::stage_files(&[]);
        std::env::set_current_dir(&prev_dir).expect("restored cwd");

        assert!(
            result.is_err(),
            "stage_files(&[]) must refuse to run, got {result:?}"
        );

        let output = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&repo_dir)
            .output()
            .unwrap_or_else(|e| panic!("git status spawn failed: {e}"));
        let status_text = String::from_utf8_lossy(&output.stdout);
        assert!(
            status_text.contains(" M unrelated.txt"),
            "unrelated.txt must remain unstaged after stage_files(&[]), \
             but git status was:\n{status_text}\n\
             (this means stage_files ran `git add -A` — the empty-list \
             footgun is still present)"
        );

        drop(guard);
    }
}
