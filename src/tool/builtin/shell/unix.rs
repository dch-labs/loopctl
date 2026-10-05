//! The Unix [`ShellBackend`] — `bash -c` with process-group kill.
//!
//! The Unix mechanics: the command runs in
//! its own process group with `kill_on_drop`, both pipes are read
//! concurrently under the family cap, and the timeout kills the whole
//! group — sub-shells, pipelines, and `sleep` grandchildren die too,
//! not just the direct `bash` child.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::tool::ToolError;
use crate::tool::builtin::shell::MAX_OUTPUT_BYTES;
use crate::tool::builtin::shell::ShellBackend;
use crate::tool::builtin::shell::ShellOutcome;

/// The Unix backend: `bash -c`, own process group, group kill on timeout.
///
/// The reference [`ShellBackend`]: commands run through `bash -c`
/// inside their own process group, streams are captured under the
/// family cap, and a timeout SIGKILLs the whole group so pipelines
/// and grandchildren cannot outlive the deadline.
#[derive(Debug, Default)]
pub struct UnixShellBackend;

impl ShellBackend for UnixShellBackend {
    fn run<'a>(
        &'a self,
        command: &'a str,
        cwd: &'a Path,
        timeout: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<ShellOutcome, ToolError>> + Send + 'a>> {
        Box::pin(execute(command, cwd, timeout))
    }
}

/// Run one command and return its captured outcome.
///
/// Spawns `bash -c command` in `cwd` inside its own process group,
/// reads both pipes concurrently under [`MAX_OUTPUT_BYTES`], and
/// enforces `timeout`: on expiry the still-armed [`ChildGuard`] kills
/// the whole group as the dropped future unwinds, and the outcome is
/// marked `timed_out` with no partial output.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the process fails to spawn
/// or the wait fails.
async fn execute(command: &str, cwd: &Path, timeout: Duration) -> Result<ShellOutcome, ToolError> {
    let start = Instant::now();
    let mut cmd = Command::new("bash");
    cmd.arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd.process_group(0);

    let mut child = cmd
        .spawn()
        .map_err(|error| ToolError::Execution(format!("Failed to spawn command: {error}")))?;

    let mut guard = ChildGuard {
        pgid: child.id().and_then(|id| i32::try_from(id).ok()),
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let exec = async {
        let ((stdout, stdout_cut), (stderr, stderr_cut)) = tokio::join!(
            async {
                match stdout {
                    Some(mut stream) => read_bounded(&mut stream, MAX_OUTPUT_BYTES).await,
                    None => (String::new(), false),
                }
            },
            async {
                match stderr {
                    Some(mut stream) => read_bounded(&mut stream, MAX_OUTPUT_BYTES).await,
                    None => (String::new(), false),
                }
            },
        );
        let status = child.wait().await.map_err(|error| {
            ToolError::Execution(format!("Failed to wait for command: {error}"))
        })?;
        Ok::<_, ToolError>((stdout, stdout_cut, stderr, stderr_cut, status))
    };

    match tokio::time::timeout(timeout, Box::pin(exec)).await {
        Ok(Ok((stdout, stdout_cut, stderr, stderr_cut, status))) => {
            guard.disarm();
            Ok(ShellOutcome {
                stdout,
                stderr,
                stdout_cut,
                stderr_cut,
                exit_code: i64::from(status.code().unwrap_or(-1)),
                duration_ms: start.elapsed().as_millis(),
                timed_out: false,
            })
        }
        Ok(Err(error)) => Err(error),
        Err(_) => Ok(ShellOutcome {
            stdout: String::new(),
            stderr: String::new(),
            stdout_cut: false,
            stderr_cut: false,
            exit_code: -1,
            duration_ms: timeout.as_millis(),
            timed_out: true,
        }),
    }
}

/// RAII guard that kills a child's process group on drop.
///
/// On timeout cancellation, `tokio::time::timeout` drops the future,
/// dropping the `Child` (SIGKILL to the direct child) and this guard
/// (SIGKILL to the whole process group). This ensures sub-shells,
/// pipelines, and `sleep` grandchildren die too — not just the
/// `bash` child. A reaped child disarms the guard, so a helper the
/// command backgrounded **with its output redirected away from the
/// pipes** survives the tool's return. A bare `cmd &` does not: the
/// helper inherits the pipe write ends, so the tool blocks on EOF
/// until the helper exits — and on timeout the still-armed guard
/// kills it with the group.
struct ChildGuard {
    /// Process-group ID of the child, when it spawned into its own group.
    ///
    /// `None` when no live group exists to signal — the guard then
    /// has nothing to do on drop. The guard stores the PGID rather
    /// than the PID because killing the negated PGID reaches the
    /// whole group (sub-shells, pipelines, and grandchildren), not
    /// just the direct `bash` child.
    pgid: Option<i32>,
}

impl ChildGuard {
    /// Disarm the drop-kill.
    ///
    /// Called once the child has been reaped. From that point the
    /// process group belongs to whatever the command left running in
    /// it, so the guard must not signal it — neither to kill a
    /// deliberately backgrounded helper nor a group whose ID a new
    /// process may already have recycled.
    fn disarm(&mut self) {
        self.pgid = None;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(pgid) = self.pgid {
            // SAFETY: a negative pid makes libc::kill signal the entire
            // process group (the standard Unix pgroup-kill idiom); SIGKILL
            // delivery cannot fail on a live group in a way the caller could
            // recover from, so the return value is deliberately ignored.
            unsafe {
                libc::kill(pgid.wrapping_neg(), libc::SIGKILL);
            }
        }
    }
}

/// Read a child pipe into a `String`, capping retained data at `max_bytes`.
///
/// Once the cap is reached, the remaining output is drained to EOF (so
/// the pipe doesn't block the child) but not stored, preventing
/// unbounded memory growth from commands that produce gigabytes of
/// output. The lossy UTF-8 conversion can itself overshoot the cap —
/// invalid bytes expand up to three-to-one under replacement — so the
/// converted text is re-cut to the cap here. The returned flag reports
/// either overflow, so the caller can mark the truncation for the
/// model instead of cutting silently.
async fn read_bounded<R>(stream: &mut R, max_bytes: usize) -> (String, bool)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = Vec::with_capacity(8192);
    let mut truncated = false;
    let mut tmp = [0u8; 8192];
    loop {
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if buf.len() < max_bytes {
                    let room = max_bytes.saturating_sub(buf.len());
                    if let Some(chunk) = tmp.get(..n.min(room)) {
                        buf.extend_from_slice(chunk);
                    }
                    if n > room {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
        }
    }
    let mut text = String::from_utf8_lossy(&buf).into_owned();
    if text.len() > max_bytes {
        truncated = true;
        let mut cut = max_bytes;
        while !text.is_char_boundary(cut) && cut > 0 {
            cut = cut.saturating_sub(1);
        }
        text.truncate(cut);
    }
    (text, truncated)
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_bounded_grows_past_initial_capacity() {
        use std::io::Cursor;
        let data = "x".repeat(50_000);
        let mut cursor = Cursor::new(data.clone().into_bytes());
        let (result, truncated) = read_bounded(&mut cursor, MAX_OUTPUT_BYTES).await;
        assert_eq!(result, data, "all data should be retained");
        assert!(!truncated, "under-cap data must not report overflow");
    }

    #[tokio::test]
    async fn read_bounded_caps_at_max_bytes() {
        use std::io::Cursor;
        let data = "y".repeat(100_000);
        let mut cursor = Cursor::new(data.into_bytes());
        let (result, truncated) = read_bounded(&mut cursor, 10_000).await;
        assert!(
            result.len() <= 10_000,
            "retained {} bytes, should be <= 10000",
            result.len()
        );
        assert!(truncated, "an over-cap stream must report the cut");
    }

    #[tokio::test]
    async fn read_bounded_small_data_preserved() {
        use std::io::Cursor;
        let data = "tiny";
        let mut cursor = Cursor::new(data.as_bytes().to_vec());
        let (result, truncated) = read_bounded(&mut cursor, 8192).await;
        assert_eq!(result, "tiny");
        assert!(!truncated);
    }

    #[tokio::test]
    async fn read_bounded_caps_the_lossy_expansion_of_invalid_utf8() {
        use std::io::Cursor;
        let data = vec![0xFFu8; 20_000];
        let mut cursor = Cursor::new(data);
        let (result, truncated) = read_bounded(&mut cursor, 10_000).await;
        assert!(
            result.len() <= 10_000,
            "the lossy conversion's expansion must be re-cut: {}",
            result.len()
        );
        assert!(truncated);
    }

    #[tokio::test]
    async fn read_bounded_drains_after_cap() {
        use std::io::Cursor;
        let data = "z".repeat(50_000);
        let mut cursor = Cursor::new(data.into_bytes());
        let (result, truncated) = read_bounded(&mut cursor, 10_000).await;
        assert_eq!(result.len(), 10_000);
        assert!(truncated, "the drain past the cap still reports the cut");
    }

    #[tokio::test]
    async fn run_reports_a_missing_command_as_exit_127() {
        let cwd = std::env::temp_dir();
        let outcome = UnixShellBackend
            .run(
                "command-does-not-exist-nonexistent",
                &cwd,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        // `bash -c 'missing'` resolves through PATH and exits 127 —
        // an ordinary nonzero exit, not a spawn error or a timeout.
        // The spawn-error arm itself needs a host without `bash` in
        // PATH and is untestable here; its shape is the `Err` return
        // the trait documents.
        assert_eq!(outcome.exit_code, 127);
        assert!(!outcome.timed_out);
        assert!(!outcome.stdout_cut);
        assert!(!outcome.stderr_cut);
    }

    #[tokio::test]
    async fn run_captures_both_streams_and_the_exit_code() {
        let tmp = tempfile::TempDir::new().unwrap();
        let outcome = UnixShellBackend
            .run(
                "echo out; echo err 1>&2; exit 3",
                tmp.path(),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(outcome.exit_code, 3);
        assert!(outcome.stdout.contains("out"), "{:?}", outcome.stdout);
        assert!(outcome.stderr.contains("err"), "{:?}", outcome.stderr);
        assert!(!outcome.timed_out);
    }

    #[tokio::test]
    async fn run_marks_the_timeout_without_partial_output() {
        let tmp = tempfile::TempDir::new().unwrap();
        let outcome = UnixShellBackend
            .run("sleep 30", tmp.path(), Duration::from_secs(1))
            .await
            .unwrap();
        assert!(outcome.timed_out);
        assert_eq!(outcome.exit_code, -1);
        assert!(outcome.stdout.is_empty());
    }
}
