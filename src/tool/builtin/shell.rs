//! The shell tool family — command execution over a pluggable backend.
//!
//! Two tools plus the machinery they share: a backend seam
//! ([`ShellBackend`]) owning process spawn, capped stream capture,
//! and timeout/kill mechanics; a shared background-job store
//! ([`JobStore`]) fed by the `Shell` tool's background spawns and
//! read by the `Jobs` tool. The model-facing name is `Shell`, not
//! `bash`, so gate policies stay OS-portable — the Unix backend
//! (`bash -c` with process-group kill) ships behind `#[cfg(unix)]`,
//! and the seam keeps schema and gate rules backend-neutral, so a
//! second backend changes neither.

pub mod jobs;
#[cfg(unix)]
pub mod unix;

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use serde_json::json;

use crate::tool::builtin::shell::jobs::JobStore;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Default command timeout in seconds.
///
/// Used when the model's input omits `timeout_s`; the same default
/// bounds background jobs spawned without one.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Hard ceiling on a command timeout, in seconds.
///
/// Larger input values are clamped rather than rejected, so an
/// over-ambitious request still runs with the maximum allowed wait.
const MAX_TIMEOUT_SECS: u64 = 600;

/// Per-stream cap on captured stdout or stderr, in bytes.
///
/// Each stream is read independently under this cap and drained past
/// it, so a command's total retained output is bounded by twice this
/// plus the join and metadata line. Shared with the job store, whose
/// stored payloads and renderings answer to the same bound.
pub(crate) const MAX_OUTPUT_BYTES: usize = 1_000_000;

/// The marker appended wherever captured output was capped.
///
/// One wording across every path — the live command's streams, a
/// render-time over-cap join, a stored job payload's middle cut — so
/// a consumer keying on the marker finds all of them.
pub(crate) const TRUNCATION_MARKER: &str = "...[output truncated]";

/// The result of one backend run, before the tool shapes it.
///
/// Carries everything the tool's rendering needs: the two captured
/// streams with their cut flags, the exit code (`-1` when a signal
/// took the process, matching `std::process::ExitStatus::code`'s
/// `None`), the wall-clock duration, and whether the timeout fired.
/// Backends that cannot run at all answer `Err` at the seam instead.
#[derive(Debug, Clone)]
pub struct ShellOutcome {
    /// Captured standard output, capped at the family's per-stream ceiling.
    ///
    /// Lossy UTF-8: a command may emit arbitrary bytes, and the
    /// replacement characters keep the body renderable. The cap is
    /// the crate-internal `MAX_OUTPUT_BYTES`, one million bytes per stream.
    pub stdout: String,

    /// Captured standard error, capped independently.
    ///
    /// Joined after stdout in the rendered body, so a failing
    /// command's diagnostics land below whatever it printed first.
    pub stderr: String,

    /// Whether stdout retention stopped at the cap.
    ///
    /// The stream is still drained to EOF past the cap so the child
    /// cannot block on a full pipe; this flag only reports that the
    /// retained text was cut.
    pub stdout_cut: bool,

    /// Whether stderr retention stopped at the cap.
    ///
    /// Independent of stdout's flag — one noisy stream must not
    /// erase the other's tail.
    pub stderr_cut: bool,

    /// The process exit code, or `-1` when a signal ended it.
    ///
    /// Signals reach this as `None` from `ExitStatus::code`; the
    /// `-1` spelling keeps the metadata line a single integer.
    pub exit_code: i64,

    /// Wall-clock duration of the run, in milliseconds.
    ///
    /// Measured by the backend from spawn to reap, so the metadata
    /// line reports what the process cost, not the tool's shaping.
    pub duration_ms: u128,

    /// Whether the timeout fired and the run was killed for it.
    ///
    /// A timed-out run carries no partial output — the kill races
    /// the capture — so the tool renders the fixed timeout message
    /// instead of the streams.
    pub timed_out: bool,
}

/// One platform's command-execution mechanics.
///
/// The seam the `Shell` tool runs commands through: the backend owns
/// spawning, stream capture under the family cap, and timeout
/// enforcement with the platform's kill idiom (process-group kill on
/// Unix; job objects on Windows). The tool owns
/// parsing, schema, output shaping, and the job table, so a new
/// backend changes no schema or gate rule. Object-safe and boxed,
/// the [`ContentSource`](crate::tool::builtin::read::ContentSource)
/// shape.
pub trait ShellBackend: Send + Sync {
    /// Whether this backend can execute commands on this platform.
    ///
    /// Backends that cannot run answer `false` here **and** `Err`
    /// from [`run`](Self::run); the tool consults the probe before
    /// acknowledging a background spawn, so a platform without a
    /// backend refuses synchronously instead of recording a job
    /// whose payload only later reveals the refusal. Defaults to
    /// `true` — an implementable backend is available by definition.
    fn available(&self) -> bool {
        true
    }

    /// Run `command` in `cwd` under `timeout`, capturing its streams.
    ///
    /// The returned future resolves once the process is reaped or
    /// the timeout killed it; a backend that cannot run commands on
    /// this platform answers `Err` immediately.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] when the process fails to
    /// spawn or wait, or when no backend exists on the platform.
    fn run<'a>(
        &'a self,
        command: &'a str,
        cwd: &'a Path,
        timeout: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<ShellOutcome, ToolError>> + Send + 'a>>;
}

/// Execute shell commands through a [`ShellBackend`].
///
/// Foreground runs capture both streams under the family cap and
/// return them with an `[exit {code}, {duration_ms}ms]` metadata
/// line; `background: true` hands the command to the shared
/// [`JobStore`] and returns a job id immediately — the `Jobs` tool
/// is the other half of that contract. Not read-only; concurrency
/// is decided per input by an allowlist of read-only command
/// prefixes (read-only commands dispatch in parallel with other
/// reads, everything else runs serialized). Registering the tool
/// grants the model the host account's execution authority: the
/// working directory is a starting point, not a sandbox, so the
/// host decides what registering this tool permits.
pub struct ShellTool<B: ShellBackend + 'static> {
    /// The backend commands run through.
    ///
    /// Shared by `Arc` so the detached background-completion tasks
    /// can hold a clone for the job's lifetime.
    backend: Arc<B>,

    /// The background-job store this tool feeds.
    ///
    /// Shared with the `Jobs` tool the host builds from the same
    /// `Arc`, so spawns and polls see one table.
    jobs: Arc<JobStore>,
}

impl<B: ShellBackend + 'static> ShellTool<B> {
    /// Build a shell tool over `backend`, feeding `jobs`.
    ///
    /// The store is host-created and shared: build one [`JobStore`],
    /// hand clones to this constructor and to
    /// [`JobsTool::new`](crate::tool::builtin::shell::jobs::JobsTool::new),
    /// and the two tools share one table without any process global.
    #[must_use]
    pub fn new(backend: B, jobs: Arc<JobStore>) -> Self {
        Self {
            backend: Arc::new(backend),
            jobs,
        }
    }
}

impl<B: ShellBackend + 'static> Tool for ShellTool<B> {
    fn name(&self) -> &'static str {
        "Shell"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command. Supports background jobs, timeout enforcement, \
         and a dynamic concurrency check (read-only commands are safe to run \
         concurrently). Prefer the Read tool for file contents — a whole-file \
         dump here lands in context at its full size; bounded peeks \
         (`sed -n '10,20p'`, `head`, `tail`) are the shell way to look."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name(),
            self.description(),
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 120, max: 600)",
                        "default": 120,
                        "minimum": 1,
                        "maximum": 600
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run in the background and return a job ID immediately",
                        "default": false
                    }
                },
                "required": ["command"]
            }),
        )
    }

    fn call(
        &self,
        input: Value,
        context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let cwd = PathBuf::from(&context.cwd);
        Box::pin(async move { self.call_inner(input, cwd).await })
    }

    fn is_safe_for_concurrent_execution(&self, input: &Value) -> bool {
        is_read_only_command(input)
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn system_prompt(&self) -> Option<String> {
        Some(
            "Run commands via the Shell tool. Prefer specific commands over \
              scripts. For reading files prefer the Read tool — a whole-file \
              cat here lands in context at its full size. \
              Never run destructive commands (`rm -rf /`, force-push) \
              without stating intent first. Background jobs you expect to \
              outlast ten minutes cannot finish — the timeout bounds them at \
              600 seconds; split the work or checkpoint and resume. Manage \
              and poll background jobs through the Jobs tool."
                .to_string(),
        )
    }
}

impl<B: ShellBackend + 'static> ShellTool<B> {
    /// Body of [`Tool::call`].
    ///
    /// The foreground path awaits the backend and shapes the outcome
    /// (joined streams, truncation marker, exit-metadata line); the
    /// background path records a running job, detaches the
    /// completion task, and answers with the job id. The detached
    /// task is the one sanctioned escape from the structured-
    /// concurrency rule: a background job's lifetime intentionally
    /// outlasts the call, bounded by its timeout and the store's
    /// terminal-job cap.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::InvalidInput`] for a missing `command`
    /// or a zero `timeout_s`, and [`ToolError::Execution`] when the
    /// backend fails to spawn or wait the process.
    async fn call_inner(&self, input: Value, cwd: PathBuf) -> Result<ToolOutput, ToolError> {
        let command = match input.get("command") {
            None => {
                return Err(ToolError::InvalidInput("Missing command".to_string()));
            }
            Some(Value::String(command)) => command.clone(),
            Some(_) => {
                return Err(ToolError::InvalidInput(
                    "'command' must be a string".to_string(),
                ));
            }
        };
        let timeout_secs = get_timeout_secs(&input)?.min(MAX_TIMEOUT_SECS);

        if get_background(&input)? {
            if !self.backend.available() {
                return Err(ToolError::Execution(
                    "no shell backend on this platform".to_string(),
                ));
            }
            let id = self.jobs.spawn(
                Arc::clone(&self.backend),
                command.clone(),
                cwd,
                timeout_secs,
            );
            let summary = jobs::command_summary(&command);
            return Ok(ToolOutput::text(format!(
                "Started background job {id}: {summary}"
            )));
        }

        let outcome = self
            .backend
            .run(&command, &cwd, Duration::from_secs(timeout_secs))
            .await?;
        Ok(render_outcome(&outcome, timeout_secs))
    }
}

/// Render a backend outcome as the tool's output.
///
/// Joins the streams (stdout first), appends the truncation marker
/// when either was cut, and closes with the `[exit …]` metadata
/// line. A timeout renders the fixed message and an error-shaped
/// output; a non-zero exit renders the captured body as an error
/// text so the model sees the failure without an exception. Shared
/// by the foreground path and the background completion task, so a
/// polled job's payload is exactly what the foreground call would
/// have returned.
pub(crate) fn render_outcome(outcome: &ShellOutcome, timeout_secs: u64) -> ToolOutput {
    if outcome.timed_out {
        return ToolOutput::error_text(format!("Command timed out after {timeout_secs} seconds"));
    }
    let mut body = if outcome.stderr.is_empty() {
        outcome.stdout.clone()
    } else {
        format!("{}\n{}", outcome.stdout, outcome.stderr)
    };
    if outcome.stdout_cut || outcome.stderr_cut {
        body.push('\n');
        body.push_str(TRUNCATION_MARKER);
    }
    let output_text = format!(
        "{body}\n[exit {}, {}ms]",
        outcome.exit_code, outcome.duration_ms
    );
    if outcome.exit_code == 0 {
        ToolOutput::text(output_text)
    } else {
        ToolOutput::error_text(output_text)
    }
}

/// Commands that are safe to run concurrently (read-only).
///
/// Matched as boundary-aware prefixes of the trimmed command —
/// `cargo check` qualifies, `cargo checkout` does not. A command
/// matching here, with no shell operator or unsafe substring,
/// dispatches in parallel with other reads.
const READ_ONLY_PREFIXES: &[&str] = &[
    "cat",
    "ls",
    "ll",
    "grep",
    "find",
    "head",
    "tail",
    "wc",
    "echo",
    "pwd",
    "which",
    "file",
    "stat",
    "git status",
    "git diff",
    "git log",
    "git show",
    "cargo check",
    "cargo test --no-run",
    "cargo clippy --no-deps",
    "make -n",
];

/// Shell operators that indicate a compound command (always unsafe).
///
/// Any one of these disqualifies a command from concurrent execution
/// regardless of the words around it — a pipeline or redirection can
/// have side effects even when every word looks read-only, and a bare
/// `&` backgrounds the command, whose lifetime then escapes the call.
/// A `$` or `{` disqualifies for a different reason: the shell expands
/// parameters (`${IFS}`, `$HOME`), command substitutions (`$(…)`),
/// ANSI-C quoting (`$'…'`), and brace expansions (`{a,b}`) into words
/// *after* the command text is matched, so any command carrying one
/// executes words the checks never saw. Refusing is conservative — a
/// benign `echo $HOME` runs serialized — which is the safe direction.
const SHELL_OPERATORS: &[&str] = &["&&", "||", ";", "|", "`", "$", "{", ">", ">>", "<", "&"];

/// Substrings that make an otherwise-allowlisted command unsafe.
///
/// Guard against mutating flags and subcommands hiding inside an
/// allowlisted prefix, such as `find -delete`, find's file-writing
/// `-fls`/`-fprint*` flags, `git diff --output` writing its output
/// to a file in either the `=`-joined or space-separated form, or
/// `git branch -D`.
const UNSAFE_SUBSTRINGS: &[&str] = &[
    " -delete",
    " -exec",
    " -fls",
    " -fprint",
    " --output",
    "git branch -D",
    "git branch -d",
    "git branch --delete",
    "git remote add",
    "git remote remove",
    "git remote rm",
    "git remote set-url",
    "git remote rename",
];

/// Read the `timeout_s` field: absent → the default, integer → kept.
///
/// The family-wide numeric-input rule, matching the shared search
/// parse: absent is the default, a non-negative integer is taken,
/// anything else is a correction prompt naming the field and its
/// expected type — a model that sends `"timeout_s": "1"` must be
/// told, not silently handed a 120-second bound it did not ask for.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a non-negative integer.
fn get_timeout_secs(input: &Value) -> Result<u64, ToolError> {
    match input.get("timeout_s") {
        None => Ok(DEFAULT_TIMEOUT_SECS),
        Some(value) if value.is_u64() => {
            let secs = value.as_u64().unwrap_or(DEFAULT_TIMEOUT_SECS);
            if secs == 0 {
                return Err(ToolError::InvalidInput(
                    "'timeout_s' must be at least 1, got 0".to_string(),
                ));
            }
            Ok(secs)
        }
        Some(_) => Err(ToolError::InvalidInput(
            "'timeout_s' must be a positive integer".to_string(),
        )),
    }
}

/// Read the `background` flag: absent → `false`, boolean → kept.
///
/// The same present-but-wrong-typed rule as [`get_timeout_secs`]: a
/// string `"true"` must be corrected, not silently run foreground.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a boolean.
fn get_background(input: &Value) -> Result<bool, ToolError> {
    match input.get("background") {
        None => Ok(false),
        Some(value) if value.is_boolean() => Ok(value.as_bool().unwrap_or(false)),
        Some(_) => Err(ToolError::InvalidInput(
            "'background' must be a boolean".to_string(),
        )),
    }
}

/// Check whether a command is read-only (safe to run concurrently).
///
/// Compound commands (containing shell operators), shell
/// redirections, and destructive subcommands are always unsafe.
/// Otherwise the command is checked against the read-only prefix
/// allowlist with boundary-aware matching so `cargo check` matches
/// but `cargo checkout` does not. A command carrying a newline or
/// any other control character besides tab is unsafe outright: the
/// normalizer collapses those away, but the shell that runs the
/// command honors a newline as a statement separator, so the
/// matched text is not the whole story of what would execute.
fn is_read_only_command(input: &Value) -> bool {
    let Some(command) = input.get("command").and_then(Value::as_str) else {
        return false;
    };
    if has_control_separator(command) {
        return false;
    }
    let normalized = shell_normalized(command);
    if normalized.is_empty() {
        return false;
    }
    if SHELL_OPERATORS.iter().any(|op| normalized.contains(op)) {
        return false;
    }
    if UNSAFE_SUBSTRINGS.iter().any(|sub| normalized.contains(sub)) {
        return false;
    }
    READ_ONLY_PREFIXES.iter().any(|prefix| {
        if normalized.len() == prefix.len() {
            return normalized == *prefix;
        }
        if normalized.len() > prefix.len() {
            return normalized.starts_with(prefix)
                && normalized[prefix.len()..].starts_with(char::is_whitespace);
        }
        false
    })
}

/// Collapse a command the way the shell would tokenize it for matching.
///
/// Whitespace runs become single spaces and quote characters and
/// backslash escapes are dropped, so a tab-separated, quoted, or
/// escaped argument cannot slip an unsafe flag past checks that
/// match space-separated text — a tab-separated `find . -delete`, `find .
/// "-delete"`, and `git branch \-D` normalize to the same literal
/// forms the denylist guards. The collapse is purely lexical: it
/// sees only the literal text, never the words run-time expansion
/// produces, so expansion-bearing commands are refused outright by
/// the operator check rather than normalized.
fn shell_normalized(command: &str) -> String {
    command
        .replace(['"', '\'', '\\'], "")
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Whether the raw command carries a control character other than tab.
///
/// A newline is a statement separator in the shell that runs the
/// command, and the other control characters have no honest business
/// in one — but the normalizer erases them all the same, so any
/// command bearing one must be treated as not matching the text it
/// normalizes to. Tab is exempt: it is an ordinary word separator the
/// normalizer folds like a space.
fn has_control_separator(command: &str) -> bool {
    command
        .chars()
        .any(|character| character != '\t' && character.is_control())
}

/// The placeholder backend for platforms without a shell implementation.
///
/// Answers every run with the error that names the gap, so the
/// `Shell` tool's schema stays present and gate rules stay portable
/// cross-OS — a Windows host enabling `shell_tools` gets the tool
/// and an honest refusal, not a missing symbol. The trait accepts a
/// Windows backend (command execution through `pwsh -Command` with
/// job-object kill) without touching the tools.
#[cfg(not(unix))]
#[derive(Debug, Default)]
pub struct UnsupportedShellBackend;

#[cfg(not(unix))]
impl ShellBackend for UnsupportedShellBackend {
    fn available(&self) -> bool {
        false
    }

    fn run<'a>(
        &'a self,
        _command: &'a str,
        _cwd: &'a Path,
        _timeout: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<ShellOutcome, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            Err(ToolError::Execution(
                "no shell backend on this platform".to_string(),
            ))
        })
    }
}

#[cfg(all(test, unix))]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::tool::builtin::shell::jobs::JobStore;
    use crate::tool::builtin::shell::unix::UnixShellBackend;
    use serde_json::json;
    use std::time::Instant;

    fn tool() -> ShellTool<UnixShellBackend> {
        ShellTool::new(UnixShellBackend, Arc::new(JobStore::new()))
    }

    fn ctx_in(cwd: &std::path::Path) -> ToolContext {
        ToolContext {
            cwd: cwd.to_string_lossy().into_owned(),
            ..ToolContext::default()
        }
    }

    #[test]
    fn schema_name_is_shell_not_bash() {
        let tool = tool();
        assert_eq!(tool.name(), "Shell");
        assert_eq!(tool.schema().tool, "Shell");
        assert!(
            !tool.name().eq_ignore_ascii_case("bash"),
            "the wire name is 'Shell', never 'bash' — gate policies key on it"
        );
    }

    #[test]
    fn concurrency_check_allowlist_hits() {
        for command in [
            "cat file",
            "ls -la",
            "grep -rn foo .",
            "git status",
            "git diff --stat",
            "git log -5",
            "cargo check",
            "cargo test --no-run",
            "echo hi",
        ] {
            assert!(
                is_read_only_command(&json!({ "command": command })),
                "'{command}' should be read-only"
            );
        }
    }

    #[test]
    fn concurrency_check_write_commands() {
        for command in [
            "rm -rf /",
            "touch x",
            "mkdir d",
            "cargo build",
            "cargo test",
            "git commit -m x",
            "git push",
            "npm install",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "'{command}' mutates and must run serialized"
            );
        }
    }

    #[test]
    fn concurrency_check_boundary_correctness() {
        // Prefix matching is boundary-aware: `cargo check` qualifies,
        // `cargo checkout` does not.
        assert!(is_read_only_command(&json!({ "command": "cargo check" })));
        assert!(!is_read_only_command(
            &json!({ "command": "cargo checkout" })
        ));
        assert!(is_read_only_command(&json!({ "command": "git log" })));
        assert!(!is_read_only_command(&json!({ "command": "git logs" })));
    }

    #[test]
    fn concurrency_check_compound_commands_unsafe() {
        for command in [
            "cat a && cat b",
            "ls || ls",
            "echo a; echo b",
            "cat a | wc -l",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "a compound command is never concurrently safe: {command}"
            );
        }
    }

    #[test]
    fn concurrency_check_control_bearing_commands_unsafe() {
        assert!(!is_read_only_command(&json!({
            "command": "echo a\necho b"
        })));
        assert!(is_read_only_command(&json!({ "command": "ls\t-la" })));
    }

    #[test]
    fn concurrency_check_redirections_unsafe() {
        for command in ["ls > out", "cat a >> b", "wc < a", "sleep 5 &"] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "a redirection or background is never concurrently safe: {command}"
            );
        }
    }

    #[test]
    fn concurrency_check_find_mutating_unsafe() {
        for command in [
            "find . -delete",
            "find . -exec rm {} \";",
            "find . -fls log",
            "find . -fprint out",
            "find . ${IFS}-delete",
            "find . $'-delete'",
            "find . {-delete,}",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "find's mutating flags must not ride the read-only prefix, spelled literally \
                 or through shell expansion: {command}"
            );
        }
    }

    #[test]
    fn concurrency_check_shell_expansion_spellings_unsafe() {
        for command in [
            "find . ${IFS}-delete",
            "find . $'-delete'",
            "find . {-delete,}",
            "find . $IFS-delete",
            "find . \"${IFS}-delete\"",
            "cat $(echo hi)",
            "ls ${PWD}",
            "echo $HOME",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "the shell expands '$' and '{{' into words the checked text never shows, so \
                 the command must run serialized: {command}"
            );
        }
    }

    #[test]
    fn concurrency_check_whitespace_and_quotes_normalized() {
        // Tab-separated, quoted, and escaped forms normalize to the
        // same literal the denylist guards.
        assert!(!is_read_only_command(&json!({
            "command": "find\t.\t-delete"
        })));
        assert!(!is_read_only_command(&json!({
            "command": "find . \"-delete\""
        })));
        assert!(!is_read_only_command(&json!({
            "command": "git branch \\-D x"
        })));
        assert!(is_read_only_command(&json!({ "command": "ls\t-la" })));
    }

    #[test]
    fn concurrency_check_git_output_flag_unsafe() {
        assert!(!is_read_only_command(&json!({
            "command": "git diff --output=/tmp/x"
        })));
        assert!(!is_read_only_command(&json!({
            "command": "git diff --output /tmp/x"
        })));
    }

    #[test]
    fn concurrency_check_git_mutating_subcommands_unsafe() {
        for command in [
            "git branch -D feature",
            "git branch -d old",
            "git branch --delete stale",
            "git remote add origin url",
            "git remote remove upstream",
            "git remote set-url origin url",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "'{command}' mutates git and must run serialized"
            );
        }
    }

    #[test]
    fn concurrency_check_remote_forms_run_sequential() {
        for command in [
            "git remote",
            "git remote -v",
            "git remote prune origin",
            "git remote update",
            "git remote set-head origin master",
        ] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "'{command}' writes or could write refs and must run sequential"
            );
        }
    }

    #[test]
    fn concurrency_check_branch_forms_run_sequential() {
        for command in ["git branch", "git branch --list", "git branch feature-x"] {
            assert!(
                !is_read_only_command(&json!({ "command": command })),
                "'{command}' creates or could create a ref and must run sequential"
            );
        }
    }

    #[test]
    fn concurrency_check_missing_command() {
        assert!(!is_read_only_command(&json!({})));
        assert!(!is_read_only_command(&json!({ "command": "" })));
    }

    #[test]
    fn schema_has_v1_properties() {
        let tool = tool();
        let schema = tool.schema();
        let props = schema
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("command"));
        assert!(props.contains_key("background"));
        assert!(props.contains_key("timeout_s"));
        assert!(
            !props.contains_key("timeout"),
            "the schema spells the neutral 'timeout_s', not the dch 'timeout'"
        );
        // Job management belongs to the Jobs tool — shell never asks
        // for a field it would ignore.
        assert!(!props.contains_key("operation"));
        assert!(!props.contains_key("job_id"));

        let required = schema
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        // `command` is the single required field — local models and
        // schema-to-grammar conversions do not honor `anyOf`.
        assert_eq!(
            required.len(),
            1,
            "command must be the single required field: {required:?}"
        );
        assert_eq!(required[0], "command");
        assert!(
            schema.input_schema.get("anyOf").is_none(),
            "anyOf alternatives are not understood by local models; keep it out"
        );
    }

    #[test]
    fn constants_match_spec() {
        assert_eq!(DEFAULT_TIMEOUT_SECS, 120);
        assert_eq!(MAX_TIMEOUT_SECS, 600);
        assert_eq!(MAX_OUTPUT_BYTES, 1_000_000);
    }

    #[test]
    fn no_default_registry_enables_shell_tools() {
        // The sandbox story must exist before defaults ever include
        // shell execution; the preset builders are the crate's default
        // registries, and none of them may construct the family.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let presets = std::fs::read_to_string(root.join("src/presets.rs"))
            .expect("presets source must be readable");
        assert!(
            !presets.contains("ShellTool") && !presets.contains("JobsTool"),
            "a default preset constructs the shell family — ship it off by default"
        );
    }

    #[test]
    fn system_prompt_present() {
        let prompt = tool().system_prompt().expect("the tool ships guidance");
        assert!(prompt.contains("Shell tool"));
        assert!(prompt.contains("Jobs tool"));
    }

    #[tokio::test]
    async fn echo_returns_stdout() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tool = tool();
        let output = tool
            .call(json!({ "command": "echo hello" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        assert!(!output.is_error);
        assert!(output.text_content().contains("hello"));
        assert!(output.text_content().contains("[exit 0,"));
    }

    #[tokio::test]
    async fn failing_command_includes_exit_code() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tool = tool();
        let output = tool
            .call(json!({ "command": "exit 3" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        assert!(output.is_error);
        assert!(output.text_content().contains("[exit 3,"));
    }

    #[tokio::test]
    async fn stdout_and_stderr_combined() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tool = tool();
        let output = tool
            .call(
                json!({ "command": "echo out; echo err 1>&2" }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        let text = output.text_content();
        assert!(text.contains("out"));
        assert!(text.contains("err"));
    }

    #[tokio::test]
    async fn missing_command_is_invalid_input() {
        let tmp = tempfile::TempDir::new().unwrap();
        let error = tool()
            .call(json!({}), &ctx_in(tmp.path()))
            .await
            .expect_err("a missing command is a correction prompt");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_wrong_typed_command_is_rejected_as_typed_not_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        for wrong in [json!(42), json!(2.5), json!(null), json!(["ls", "-la"])] {
            let error = tool()
                .call(json!({ "command": wrong }), &ctx_in(tmp.path()))
                .await
                .expect_err("a present-but-wrong-typed command is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => assert_eq!(
                    message, "'command' must be a string",
                    "the model sent the field; it must be told the type, not that it is missing"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn a_missing_command_still_says_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let error = tool()
            .call(json!({}), &ctx_in(tmp.path()))
            .await
            .expect_err("an absent command is a missing-field prompt");
        match error {
            ToolError::InvalidInput(message) => {
                assert_eq!(message, "Missing command");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_wrong_typed_timeout_is_rejected_not_defaulted() {
        let tmp = tempfile::TempDir::new().unwrap();
        for wrong in [json!("1"), json!(-5), json!(12.5), json!(null), json!(true)] {
            let error = tool()
                .call(
                    json!({ "command": "echo hi", "timeout_s": wrong }),
                    &ctx_in(tmp.path()),
                )
                .await
                .expect_err("a present-but-wrong-typed timeout is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => assert!(
                    message.contains("'timeout_s'"),
                    "the message must name the field: {message}"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn a_wrong_typed_background_flag_is_rejected_not_defaulted() {
        let tmp = tempfile::TempDir::new().unwrap();
        let error = tool()
            .call(
                json!({ "command": "echo hi", "background": "true" }),
                &ctx_in(tmp.path()),
            )
            .await
            .expect_err("a present-but-wrong-typed flag is a correction prompt");
        match error {
            ToolError::InvalidInput(message) => assert!(
                message.contains("'background'"),
                "the message must name the field: {message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_absent_timeout_still_takes_the_default() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(json!({ "command": "echo hi" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        assert!(!output.is_error, "absent means the default, never an error");
    }

    #[tokio::test]
    async fn zero_timeout_is_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let error = tool()
            .call(
                json!({ "command": "echo hi", "timeout_s": 0 }),
                &ctx_in(tmp.path()),
            )
            .await
            .expect_err("a zero timeout is a correction prompt");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn oversized_timeout_is_clamped_not_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "echo hi", "timeout_s": 99_999 }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        assert!(!output.is_error, "clamped, not rejected");
    }

    /// The task-named pin: the timeout must kill the whole process
    /// group, not just the direct child.
    ///
    /// The command backgrounds a marker-writing subshell (which
    /// holds the pipe, keeping the run alive) and sleeps. On
    /// timeout the still-armed guard SIGKILLs the group; if only the
    /// direct `bash` child died, the subshell would survive to write
    /// its marker — the marker staying absent is the discrimination.
    #[tokio::test]
    async fn timeout_kills_the_process_group_unix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join("grandchild-marker");
        let command = format!("(sleep 1 && touch {}) & sleep 30", marker.display());
        let start = Instant::now();
        let output = tool()
            .call(
                json!({ "command": command, "timeout_s": 1 }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        assert!(output.is_error);
        assert!(
            output.text_content().contains("timed out"),
            "{}",
            output.text_content()
        );
        assert!(
            start.elapsed().as_secs() < 10,
            "the run must end at the timeout, not the sleep: {:?}",
            start.elapsed()
        );
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        assert!(
            !marker.exists(),
            "the backgrounded subshell died with the group — only a \
             process-group kill reaches it"
        );
    }

    #[tokio::test]
    async fn timeout_kills_pipeline() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "sleep 30 | cat", "timeout_s": 1 }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        assert!(output.is_error);
        assert!(output.text_content().contains("timed out"));
    }

    #[tokio::test]
    async fn output_truncation_bounds_the_body() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "yes y | head -c 2000000" }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        let text = output.text_content();
        assert!(
            text.len() <= MAX_OUTPUT_BYTES.saturating_add(512),
            "retained {} bytes",
            text.len()
        );
        assert!(
            text.contains(TRUNCATION_MARKER),
            "a capped stream must say so"
        );
    }

    #[tokio::test]
    async fn bounded_read_does_not_exhaust_memory() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "yes y | head -c 10000000" }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        let text = output.text_content();
        assert!(
            text.len() <= MAX_OUTPUT_BYTES.saturating_add(512),
            "retained {} bytes",
            text.len()
        );
    }

    #[tokio::test]
    async fn capped_output_carries_a_truncation_marker() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "yes y | head -c 2000000" }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        assert!(
            output.text_content().contains(TRUNCATION_MARKER),
            "a capped stream must say so"
        );
    }

    #[tokio::test]
    async fn stderr_is_independently_capped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tool()
            .call(
                json!({ "command": "echo ok && dd if=/dev/zero bs=2000 count=1000 1>&2" }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        assert!(
            output.text_content().len() <= MAX_OUTPUT_BYTES.saturating_add(512),
            "one noisy stream must not erase the other's bound"
        );
    }

    /// A backend that reports itself unavailable must refuse a
    /// background spawn synchronously — never acknowledge a job whose
    /// payload would only later reveal the refusal — and leave the
    /// store empty. Runs on Unix through a fake backend, so the
    /// non-Unix arm is exercised where tests actually run.
    #[tokio::test]
    async fn an_unavailable_backend_refuses_a_background_spawn_synchronously() {
        struct UnavailableBackend;
        impl ShellBackend for UnavailableBackend {
            fn available(&self) -> bool {
                false
            }
            fn run<'a>(
                &'a self,
                _command: &'a str,
                _cwd: &'a std::path::Path,
                _timeout: Duration,
            ) -> Pin<Box<dyn Future<Output = Result<ShellOutcome, ToolError>> + Send + 'a>>
            {
                Box::pin(async {
                    Err(ToolError::Execution(
                        "no shell backend on this platform".to_string(),
                    ))
                })
            }
        }
        let jobs = Arc::new(JobStore::new());
        let tool = ShellTool::new(UnavailableBackend, Arc::clone(&jobs));
        let tmp = tempfile::TempDir::new().unwrap();
        let error = tool
            .call(
                json!({ "command": "echo hi", "background": true }),
                &ctx_in(tmp.path()),
            )
            .await
            .expect_err("the refusal must be synchronous, not a recorded job");
        match error {
            ToolError::Execution(message) => assert_eq!(
                message, "no shell backend on this platform",
                "the background refusal matches the foreground one"
            ),
            other => panic!("expected Execution, got {other:?}"),
        }
        assert!(
            jobs.list().is_empty(),
            "no job entry may be recorded for a refused spawn"
        );
    }

    #[tokio::test]
    async fn a_detached_helper_survives_a_successful_command() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join("late-marker");
        let command = format!("(sleep 0.4 && touch {}) & echo started", marker.display());
        let output = tool()
            .call(json!({ "command": command }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        assert!(!output.is_error);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !marker.exists() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            marker.exists(),
            "a helper detached before the tool returned must not be group-killed"
        );
    }
}
