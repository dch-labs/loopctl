//! The background-job store and the `Jobs` tool that reads it.
//!
//! The `Shell` tool starts jobs with `background: true` and hands
//! back an ID; [`JobsTool`] is the other half of that contract —
//! listing what is running, polling one job's status and captured
//! output by ID, and clearing finished jobs out of the table. The
//! store is host-created and shared by `Arc` between the two tools;
//! there is no process-global table, so every consumer (and every
//! test) gets an isolated slate by construction.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::Value;
use serde_json::json;

use crate::tool::builtin::shell::MAX_OUTPUT_BYTES;
use crate::tool::builtin::shell::ShellBackend;
use crate::tool::builtin::shell::TRUNCATION_MARKER;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Display bound for a command rendered in job output, in bytes.
///
/// The `jobs` listing and the `job_status` header interpolate the
/// command verbatim, and nothing bounds the model's input — a huge
/// command would re-open the unbounded listing hole the payload work
/// closed, and would outgrow the header room [`JobStore`]'s payload
/// cap reserves. Renderings show at most this many bytes of it.
const MAX_COMMAND_SUMMARY_BYTES: usize = 256;

/// How many terminal jobs the table retains before the oldest are evicted.
///
/// Completed payloads are the table's memory cost; a session that
/// never calls `cleanup_jobs` would otherwise retain every job
/// forever.
const MAX_TERMINAL_JOBS: usize = 20;

/// Status of a background job.
///
/// Stored inside one job entry. Transitions are one-way: a job
/// starts [`Running`](JobStatus::Running), then moves to either
/// [`Completed`](JobStatus::Completed) or [`Failed`](JobStatus::Failed)
/// when the process exits or the timeout fires. Once terminal, the
/// status never changes again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JobStatus {
    /// Still running.
    ///
    /// No output is available yet — the process has not exited. The
    /// payload is absent because stdout/stderr are collected only
    /// when the job finishes. Polled by `job_status` until it
    /// transitions to a terminal variant.
    Running,

    /// Finished successfully.
    ///
    /// The payload is the job's final rendered output: stdout
    /// followed by stderr, plus the `[exit …, …ms]` metadata line.
    /// Returned to the model when it polls a completed job.
    Completed(String),

    /// Failed or timed out.
    ///
    /// The payload is the job's final rendered output: a failed-
    /// but-completed job's captured streams plus the metadata line,
    /// the fixed timeout message when the deadline fired — no
    /// partial pre-kill output is captured, the cancellation
    /// discards the streams — or the spawn error text when the
    /// command never started.
    Failed(String),
}

/// One tracked background job.
///
/// Stored in the [`JobStore`] table keyed by `id`. Created by
/// [`JobStore::spawn`] when the `Shell` tool runs with `background:
/// true`; updated by the detached completion task when the job exits
/// or times out.
#[derive(Debug, Clone)]
pub(crate) struct BackgroundJob {
    /// Monotonic job identifier.
    ///
    /// Allocated from the store's counter at spawn time and never
    /// reused, so a stale ID from a finished, cleaned-up, or evicted
    /// job can never resolve to a different job later.
    pub(crate) id: u64,

    /// The command string.
    ///
    /// Stored verbatim (exactly as the model supplied it) for
    /// display in the `jobs` listing. Not used for execution —
    /// that happens at spawn time.
    pub(crate) command: String,

    /// Current status of the job.
    ///
    /// Polled by `job_status` on each request. Updated in place
    /// when the process exits (success or failure) or when the
    /// timeout fires, under the store's mutex.
    pub(crate) status: JobStatus,
}

/// The shared background-job table.
///
/// Hosts create one store and hand `Arc` clones to the `Shell` tool
/// (which spawns into it) and the `Jobs` tool (which reads it) — the
/// two halves of the background-job contract share one table
/// without any process global. A poisoned lock is tolerated:
/// accessors return empty rather than failing the tool call.
pub struct JobStore {
    /// Monotonic counter for job IDs.
    ///
    /// Relaxed increments at spawn time hand out IDs that are never
    /// reused within the store's lifetime.
    next_id: AtomicU64,

    /// The job table, keyed by ID in insertion-eviction order.
    ///
    /// Detached completion tasks update their entries in place
    /// under this mutex; the `Jobs` tool's operations read clones.
    table: Mutex<BTreeMap<u64, BackgroundJob>>,
}

impl JobStore {
    /// Build an empty job store.
    ///
    /// IDs start at 1; the table grows per spawn and prunes terminal
    /// jobs past the retention cap (twenty finished jobs kept).
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            table: Mutex::new(BTreeMap::new()),
        }
    }

    /// Spawn `command` as a background job and return its ID immediately.
    ///
    /// Allocates a fresh monotonic ID, inserts a [`Running`]
    /// (`JobStatus::Running`) entry, and detaches a completion task
    /// that runs the command through `backend` capped at
    /// `timeout_secs`. When the task finishes (success, failure, or
    /// timeout) it updates the entry in place to `Completed` or
    /// `Failed`; the caller never blocks on that transition — it
    /// polls later through the `Jobs` tool. The detached task is the
    /// sanctioned escape from the structured-concurrency rule: a
    /// background job's lifetime intentionally outlasts the call,
    /// bounded by its timeout and this store's terminal cap.
    ///
    /// A poisoned table lock is tolerated: the spawn still returns
    /// the ID even if the entry could not be recorded, matching the
    /// rest of the accessors' handling.
    pub(crate) fn spawn<B: ShellBackend + 'static>(
        self: &Arc<Self>,
        backend: Arc<B>,
        command: String,
        cwd: PathBuf,
        timeout_secs: u64,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let job = BackgroundJob {
            id,
            command: command.clone(),
            status: JobStatus::Running,
        };

        if let Ok(mut table) = self.table.lock() {
            table.insert(id, job);
        }

        let store = Arc::clone(self);
        let timeout = Duration::from_secs(timeout_secs);
        tokio::spawn(async move {
            let run = backend.run(&command, &cwd, timeout);
            let (mut text, is_error) = match tokio::time::timeout(timeout, run).await {
                Ok(Ok(outcome)) => {
                    let rendered = crate::tool::builtin::shell::render_outcome(
                        &outcome,
                        timeout_secs,
                        &command,
                    );
                    (rendered.text_content(), rendered.is_error)
                }
                Ok(Err(error)) => (error.to_string(), true),
                Err(_) => (
                    format!("Command timed out after {timeout_secs} seconds"),
                    true,
                ),
            };
            cap_job_payload(&mut text);
            let status = if is_error {
                JobStatus::Failed(text)
            } else {
                JobStatus::Completed(text)
            };

            if let Ok(mut table) = store.table.lock() {
                if let Some(job) = table.get_mut(&id) {
                    job.status = status;
                }
                prune_terminal_jobs(&mut table);
            }
        });

        id
    }

    /// Retrieve a single background job by its ID.
    ///
    /// Returns a clone of the entry's current state, or `None` when
    /// the ID doesn't exist (never spawned, evicted by the terminal
    /// cap, or already cleaned up).
    pub(crate) fn get(&self, id: u64) -> Option<BackgroundJob> {
        self.table.lock().ok()?.get(&id).cloned()
    }

    /// List all tracked background jobs.
    ///
    /// A clone of every entry, in ascending job-ID order. Empty when
    /// the table is empty or the lock is poisoned.
    pub(crate) fn list(&self) -> Vec<BackgroundJob> {
        self.table
            .lock()
            .map(|table| table.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Remove every terminal job from the table.
    ///
    /// The `cleanup_jobs` operation: running jobs stay (their
    /// completion tasks still hold their IDs), finished jobs and
    /// their payloads go.
    pub(crate) fn cleanup(&self) {
        if let Ok(mut table) = self.table.lock() {
            table.retain(|_, job| job.status == JobStatus::Running);
        }
    }
}

impl Default for JobStore {
    /// Same as [`JobStore::new`].
    ///
    /// An empty table and a counter starting at 1; explicit since
    /// the type is public.
    fn default() -> Self {
        Self::new()
    }
}

/// Evict the oldest terminal jobs past the retention cap.
///
/// Running jobs never count against the cap and are never evicted —
/// only completed payloads are the table's memory cost.
fn prune_terminal_jobs(table: &mut BTreeMap<u64, BackgroundJob>) {
    let mut terminal_ids: Vec<u64> = table
        .iter()
        .filter(|(_, job)| job.status != JobStatus::Running)
        .map(|(id, _)| *id)
        .collect();
    while terminal_ids.len() > MAX_TERMINAL_JOBS {
        if let Some(oldest) = terminal_ids.first() {
            table.remove(oldest);
        }
        terminal_ids.remove(0);
    }
}

/// A display-bounded rendering of a job's command.
///
/// The stored command stays verbatim; only renderings pass through
/// here, so a pathological command cannot outgrow the job output's
/// caps. An elided command carries [`TRUNCATION_MARKER`] like any
/// other cut.
pub(crate) fn command_summary(command: &str) -> String {
    let mut summary = command.to_string();
    truncate_string(&mut summary, MAX_COMMAND_SUMMARY_BYTES);
    summary
}

/// Truncate `s` in place to at most `max_bytes`, landing on a UTF-8 char boundary.
///
/// If `s` already fits, it is left untouched. Otherwise the cut
/// point walks back from `max_bytes` to the preceding char boundary
/// so the result stays valid UTF-8, the tail is dropped, and a
/// [`TRUNCATION_MARKER`] is appended so the model can see the output
/// was capped. Used on renderings that combine independently capped
/// streams — a job payload joining capped stdout and stderr can
/// exceed the cap by construction; the live command path marks its
/// own truncation at the source instead, and a stored job payload
/// is capped before it ever reaches here.
fn truncate_string(text: &mut String, max_bytes: usize) {
    if text.len() <= max_bytes {
        return;
    }
    let mut cut = max_bytes;
    while !text.is_char_boundary(cut) && cut > 0 {
        cut = cut.saturating_sub(1);
    }
    text.truncate(cut);
    text.push_str(TRUNCATION_MARKER);
}

/// A payload-free status rendering for the `jobs` listing.
///
/// The listing concatenates one line per job; inlining each terminal
/// payload would multiply the per-job cap into an unbounded listing,
/// so the listing reports sizes and leaves payloads to `job_status`.
fn job_summary(status: &JobStatus) -> String {
    match status {
        JobStatus::Running => "Running".to_string(),
        JobStatus::Completed(payload) => format!("Completed ({} bytes)", payload.len()),
        JobStatus::Failed(payload) => format!("Failed ({} bytes)", payload.len()),
    }
}

/// Cap a stored job payload, cutting from the middle so both ends survive.
///
/// A completed job's text joins two independently capped streams, so
/// it can reach twice the render cap; a head-only cut would drop the
/// stderr tail and the `[exit …]` line — the parts a failed
/// high-output job is polled for. The middle cut keeps the head of
/// the stdout and the tail (the stderr end, any stream markers, the
/// exit metadata) with [`TRUNCATION_MARKER`] naming the elided span.
/// The halves are sized to leave room for the header `job_status`
/// prepends at render time — the id, the [`command_summary`]-bounded
/// command, and the status prefix stay under the reservation by
/// construction — so the render-time cut is a last-resort guard, not
/// the plan. The threshold carries the reservation too: a payload
/// that lands just under the cap but leaves no header room is cut
/// like any other.
fn cap_job_payload(text: &mut String) {
    let reserved = 512usize
        .saturating_add(TRUNCATION_MARKER.len())
        .saturating_add(2);
    if text.len().saturating_add(reserved) <= MAX_OUTPUT_BYTES {
        return;
    }
    let keep = MAX_OUTPUT_BYTES.saturating_sub(reserved) / 2;
    let mut head_end = keep;
    while !text.is_char_boundary(head_end) && head_end > 0 {
        head_end = head_end.saturating_sub(1);
    }
    let mut tail_start = text.len().saturating_sub(keep);
    while !text.is_char_boundary(tail_start) && tail_start < text.len() {
        tail_start = tail_start.saturating_add(1);
    }
    let tail = text.split_off(tail_start);
    text.truncate(head_end);
    text.push('\n');
    text.push_str(TRUNCATION_MARKER);
    text.push('\n');
    text.push_str(&tail);
}

/// Manage the background jobs the `Shell` tool starts.
///
/// The `Shell` tool starts jobs with `background: true` and hands back
/// an ID; this tool is the other half of that contract — listing
/// what is running, polling one job's status and captured output by
/// ID, and clearing finished jobs out of the table. Splitting job
/// management into its own tool keeps every schema single-shape:
/// `operation` is this tool's one required field and `command` is
/// the `Shell` tool's, so no input needs a field it does not use.
pub struct JobsTool {
    /// The store this tool reads.
    ///
    /// The same `Arc` the host handed the `Shell` tool, so spawns and
    /// polls share one table.
    jobs: Arc<JobStore>,
}

impl JobsTool {
    /// Build the job-management tool over `jobs`.
    ///
    /// Backend-free by construction: every operation reads or
    /// mutates the shared table, so the tool works on every
    /// platform (a platform without a shell backend simply never
    /// has jobs to show).
    #[must_use]
    pub fn new(jobs: Arc<JobStore>) -> Self {
        Self { jobs }
    }
}

impl Tool for JobsTool {
    fn name(&self) -> &'static str {
        "Jobs"
    }

    fn description(&self) -> &'static str {
        "Manage background jobs started by the Shell tool: list them, poll one \
         job's status and captured output by id, or remove finished ones."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name(),
            self.description(),
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The job-management action to perform.",
                        "enum": ["jobs", "job_status", "cleanup_jobs"]
                    },
                    "job_id": {
                        "type": "integer",
                        "description": "Job ID to query (required for operation=job_status)"
                    }
                },
                "required": ["operation"]
            }),
        )
    }

    fn call(
        &self,
        input: Value,
        _context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let jobs = Arc::clone(&self.jobs);
        Box::pin(async move { call_inner(&jobs, &input) })
    }

    fn system_prompt(&self) -> Option<String> {
        Some(
            "Background jobs return an ID immediately; poll their status and \
              captured output with the Jobs tool (operation=job_status, job_id) \
              instead of re-running the command. List running jobs with \
              operation=jobs, and clear finished ones with operation=cleanup_jobs."
                .to_string(),
        )
    }
}

/// Body of [`JobsTool`]'s [`Tool::call`], synchronous by design.
///
/// Every operation reads or mutates the shared job table, so there
/// is nothing to await.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a missing `operation`, an
/// unknown `operation`, or a `job_status` called without a `job_id`.
fn call_inner(jobs: &JobStore, input: &Value) -> Result<ToolOutput, ToolError> {
    let operation = match input.get("operation") {
        None => {
            return Err(ToolError::InvalidInput("Missing operation".to_string()));
        }
        Some(Value::String(operation)) => operation.as_str(),
        Some(_) => {
            return Err(ToolError::InvalidInput(
                "'operation' must be a string".to_string(),
            ));
        }
    };
    match operation {
        "jobs" => {
            let listed = jobs.list();
            let text = if listed.is_empty() {
                "No background jobs.".to_string()
            } else {
                listed
                    .iter()
                    .map(|job| {
                        format!(
                            "  [{}] {} — {}",
                            job.id,
                            command_summary(&job.command),
                            job_summary(&job.status)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            Ok(ToolOutput::text(text))
        }
        "job_status" => {
            let id = match input.get("job_id") {
                None => {
                    return Err(ToolError::InvalidInput(
                        "job_status requires job_id".to_string(),
                    ));
                }
                Some(value) if value.is_u64() => value.as_u64().unwrap_or(0),
                Some(_) => {
                    return Err(ToolError::InvalidInput(
                        "'job_id' must be a positive integer".to_string(),
                    ));
                }
            };
            match jobs.get(id) {
                Some(job) => {
                    let mut text = format!("[{}] {}: ", job.id, command_summary(&job.command));
                    match &job.status {
                        JobStatus::Running => text.push_str("Running"),
                        JobStatus::Completed(payload) | JobStatus::Failed(payload) => {
                            text.push_str(&payload.clone());
                        }
                    }
                    truncate_string(&mut text, MAX_OUTPUT_BYTES);
                    Ok(ToolOutput::text(text))
                }
                None => Ok(ToolOutput::error_text(format!("No such job: {id}"))),
            }
        }
        "cleanup_jobs" => {
            jobs.cleanup();
            Ok(ToolOutput::text("Cleaned up finished jobs.".to_string()))
        }
        other => Err(ToolError::InvalidInput(format!(
            "Unknown operation: {other}"
        ))),
    }
}

#[cfg(all(test, unix))]
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
    use crate::tool::builtin::shell::unix::UnixShellBackend;
    use serde_json::json;

    fn tools() -> (
        crate::tool::builtin::shell::ShellTool<UnixShellBackend>,
        JobsTool,
    ) {
        let store = Arc::new(JobStore::new());
        (
            crate::tool::builtin::shell::ShellTool::new(UnixShellBackend, Arc::clone(&store)),
            JobsTool::new(store),
        )
    }

    fn ctx_in(cwd: &std::path::Path) -> ToolContext {
        ToolContext {
            cwd: cwd.to_string_lossy().into_owned(),
            ..ToolContext::default()
        }
    }

    #[test]
    fn schema_pins_the_operation_only_shape() {
        let (_shell, jobs) = tools();
        let schema = jobs.schema();
        let props = schema
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("operation"));
        assert!(props.contains_key("job_id"));
        assert!(
            !props.contains_key("command"),
            "job management never needs a command — the shell/Jobs split \
             exists so no call carries a field it does not use"
        );

        let required = schema
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        // `operation` is the single required field — the exact
        // grammar-safe shape the split bought: plain required lists,
        // no `anyOf`, so schema-to-grammar servers can emit a valid
        // operation-only request without a filler `command`.
        assert_eq!(
            required.len(),
            1,
            "operation must be the single required field: {required:?}"
        );
        assert_eq!(required[0], "operation");
        assert!(
            schema.input_schema.get("anyOf").is_none(),
            "anyOf alternatives are not understood by local models; keep it out"
        );
    }

    #[test]
    fn constants_match_the_render_caps() {
        assert_eq!(MAX_COMMAND_SUMMARY_BYTES, 256);
        assert_eq!(MAX_TERMINAL_JOBS, 20);
    }

    #[test]
    fn terminal_eviction_keeps_the_newest_bound_and_never_a_running_job() {
        let mut table = BTreeMap::new();
        let running_ids = [1u64, 2];
        for id in running_ids {
            table.insert(
                id,
                BackgroundJob {
                    id,
                    command: format!("long-running-{id}"),
                    status: JobStatus::Running,
                },
            );
        }
        for id in 3u64..=27 {
            let status = if id % 2 == 0 {
                JobStatus::Completed(format!("done-{id}"))
            } else {
                JobStatus::Failed(format!("failed-{id}"))
            };
            table.insert(
                id,
                BackgroundJob {
                    id,
                    command: format!("finished-{id}"),
                    status,
                },
            );
        }

        prune_terminal_jobs(&mut table);

        let terminal = table
            .values()
            .filter(|job| job.status != JobStatus::Running)
            .count();
        assert_eq!(
            terminal, MAX_TERMINAL_JOBS,
            "eviction must bound the terminal count at the retention cap"
        );
        assert_eq!(
            table.len(),
            MAX_TERMINAL_JOBS + running_ids.len(),
            "running jobs never count against the terminal cap"
        );
        for id in 3u64..=7 {
            assert!(
                !table.contains_key(&id),
                "eviction drops the oldest terminal jobs first: id {id} must be gone"
            );
        }
        for id in running_ids {
            let job = table
                .get(&id)
                .expect("a running job is never eviction material");
            assert_eq!(
                job.status,
                JobStatus::Running,
                "eviction must leave running jobs untouched"
            );
        }
        for id in 8u64..=27 {
            assert!(
                table.contains_key(&id),
                "the newest terminal jobs survive eviction: id {id} must remain"
            );
        }
    }

    #[test]
    fn command_summary_bounds_a_pathological_command() {
        let long = "x".repeat(10_000);
        let summary = command_summary(&long);
        assert!(
            summary.len() <= MAX_COMMAND_SUMMARY_BYTES + TRUNCATION_MARKER.len(),
            "an elided command stays within the display bound plus marker"
        );
        assert!(summary.contains(TRUNCATION_MARKER));
        assert_eq!(command_summary("echo hi"), "echo hi");
    }

    #[test]
    fn cap_job_payload_cuts_payloads_that_leave_no_header_room() {
        let mut text = "x".repeat(MAX_OUTPUT_BYTES - 100);
        cap_job_payload(&mut text);
        assert!(
            text.len() < MAX_OUTPUT_BYTES - 300,
            "the render header must fit after the cut, retained {}",
            text.len()
        );
        assert!(text.contains(TRUNCATION_MARKER));
    }

    #[test]
    fn cap_job_payload_leaves_small_payloads_untouched() {
        let mut text = "out\n[exit 0, 12ms]".to_string();
        cap_job_payload(&mut text);
        assert_eq!(text, "out\n[exit 0, 12ms]");
    }

    #[test]
    fn truncate_string_cuts_at_a_char_boundary() {
        let mut text = "\u{20ac}".repeat(400_000);
        truncate_string(&mut text, MAX_OUTPUT_BYTES);
        assert!(
            text.len() <= MAX_OUTPUT_BYTES + TRUNCATION_MARKER.len(),
            "the cut must land on a char boundary: {}",
            text.len()
        );
        assert!(text.contains(TRUNCATION_MARKER));
    }

    #[tokio::test]
    async fn listing_an_empty_store_is_an_honest_line() {
        let (_shell, jobs) = tools();
        let output = jobs
            .call(
                json!({ "operation": "jobs" }),
                &ctx_in(std::path::Path::new("/")),
            )
            .await
            .unwrap();
        assert_eq!(output.text_content(), "No background jobs.");
    }

    #[tokio::test]
    async fn a_wrong_typed_operation_is_rejected_as_typed_not_missing() {
        let (_shell, jobs) = tools();
        for wrong in [json!(42), json!(["jobs"])] {
            let error = jobs
                .call(
                    json!({ "operation": wrong }),
                    &ctx_in(std::path::Path::new("/")),
                )
                .await
                .expect_err("a present-but-wrong-typed operation is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => assert_eq!(
                    message, "'operation' must be a string",
                    "the model sent the field; it must be told the type, not that it is missing"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn a_wrong_typed_job_id_is_rejected_as_typed() {
        let (_shell, jobs) = tools();
        let error = jobs
            .call(
                json!({ "operation": "job_status", "job_id": "1" }),
                &ctx_in(std::path::Path::new("/")),
            )
            .await
            .expect_err("a present-but-wrong-typed id is a correction prompt");
        match error {
            ToolError::InvalidInput(message) => assert_eq!(
                message, "'job_id' must be a positive integer",
                "the model sent the id; it must be told the type"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_operation_is_invalid_input() {
        let (_shell, jobs) = tools();
        let error = jobs
            .call(
                json!({ "operation": "nope" }),
                &ctx_in(std::path::Path::new("/")),
            )
            .await
            .expect_err("an unknown operation is a correction prompt");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn job_status_without_an_id_is_invalid_input() {
        let (_shell, jobs) = tools();
        let error = jobs
            .call(
                json!({ "operation": "job_status" }),
                &ctx_in(std::path::Path::new("/")),
            )
            .await
            .expect_err("job_status requires an id");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn polling_an_unknown_job_is_an_error_text_success() {
        let (_shell, jobs) = tools();
        let output = jobs
            .call(
                json!({ "operation": "job_status", "job_id": 99 }),
                &ctx_in(std::path::Path::new("/")),
            )
            .await
            .unwrap();
        assert!(output.is_error);
        assert!(output.text_content().contains("No such job: 99"));
    }

    /// The task-named pin: a background job's lifecycle reports
    /// through polling — running first, then the terminal payload
    /// with the captured output and exit metadata.
    #[tokio::test]
    async fn background_jobs_report_on_poll() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (shell, jobs) = tools();
        let started = shell
            .call(
                json!({
                    "command": "sleep 0.3 && echo job-done",
                    "background": true
                }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        let started_text = started.text_content();
        assert!(
            started_text.starts_with("Started background job 1:"),
            "{started_text}"
        );

        let listed = jobs
            .call(json!({ "operation": "jobs" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        assert!(
            listed
                .text_content()
                .contains("[1] sleep 0.3 && echo job-done — Running"),
            "a fresh job lists as running: {}",
            listed.text_content()
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut polled = String::new();
        while std::time::Instant::now() < deadline {
            let status = jobs
                .call(
                    json!({ "operation": "job_status", "job_id": 1 }),
                    &ctx_in(tmp.path()),
                )
                .await
                .unwrap();
            polled = status.text_content();
            if polled.contains("[exit 0,") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            polled.contains("job-done"),
            "the terminal payload carries the captured output: {polled}"
        );
        assert!(
            polled.contains("[exit 0,"),
            "the terminal payload carries the exit metadata: {polled}"
        );
    }

    #[tokio::test]
    async fn a_failed_background_job_polls_as_failed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (shell, jobs) = tools();
        shell
            .call(
                json!({ "command": "exit 7", "background": true }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut polled = String::new();
        while std::time::Instant::now() < deadline {
            let status = jobs
                .call(
                    json!({ "operation": "job_status", "job_id": 1 }),
                    &ctx_in(tmp.path()),
                )
                .await
                .unwrap();
            polled = status.text_content();
            if polled.contains("[exit 7,") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            polled.contains("[exit 7,"),
            "a failed job's payload carries its exit code: {polled}"
        );
    }

    #[tokio::test]
    async fn cleanup_removes_terminal_jobs_and_keeps_running_ones() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (shell, jobs) = tools();
        shell
            .call(
                json!({ "command": "echo quick", "background": true }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        shell
            .call(
                json!({ "command": "sleep 5", "background": true }),
                &ctx_in(tmp.path()),
            )
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        jobs.call(json!({ "operation": "cleanup_jobs" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        let listed = jobs
            .call(json!({ "operation": "jobs" }), &ctx_in(tmp.path()))
            .await
            .unwrap();
        let text = listed.text_content();
        assert!(
            !text.contains("echo quick"),
            "the finished job is gone: {text}"
        );
        assert!(
            text.contains("sleep 5"),
            "the running job survives cleanup: {text}"
        );
    }
}
