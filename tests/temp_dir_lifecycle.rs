//! Per-session temp-directory lifecycle for `BareLoop` tool contexts.
//!
//! Pins the [`ToolContext::temp_dir`](loopctl::tool::ToolContext::temp_dir)
//! contract: a loop-built context names a per-session subdir under the
//! configured base, the subdir is materialised lazily on first dispatch,
//! dropping the loop removes it (best-effort, never a panic) — including
//! via the husk dropped by [`into_machine`](loopctl::engine::BareLoop::into_machine)
//! — and the host can override the base or opt out entirely. A
//! randomized sequence test holds the leak-freedom property across
//! dispatch, checkpoint-resume, and loop-replacement legs.
//!
//! Requires the `testing` feature.

#![cfg(feature = "testing")]
#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};

/// A tool that records every `ToolContext` it was invoked with.
struct CaptureCtxTool {
    /// The captured `(temp_dir, session_id)` pairs, one per dispatch.
    seen: Arc<Mutex<Vec<(String, String)>>>,
}

impl Tool for CaptureCtxTool {
    fn name(&self) -> &'static str {
        "capture_ctx"
    }
    fn description(&self) -> &'static str {
        "Records its tool context"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }
    }
    fn call(
        &self,
        _input: serde_json::Value,
        ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let seen = Arc::clone(&self.seen);
        let captured = (ctx.temp_dir.clone(), ctx.session_id.to_string());
        Box::pin(async move {
            seen.lock().expect("context log lock").push(captured);
            Ok(ToolOutput::text("recorded"))
        })
    }
}

/// Removes the scratch base directory when the test ends, however it ends.
struct ScratchGuard(PathBuf);

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// A fresh scratch base directory path, unique per call.
///
/// The name combines the pid, a per-process counter, and the wall
/// clock. The counter is the load-bearing part: on hosts whose clock
/// granularity is coarse (VM runners), a tight loop can hand two
/// calls the same nanosecond, and parallel tests sharing a base would
/// delete each other's scratch — the counter keeps bases distinct
/// regardless of the clock.
fn unique_base() -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "loopctl-temp-dir-test-{}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ))
}

#[test]
fn tight_loop_bases_stay_distinct() {
    // The collision the counter exists for: a coarse wall clock hands
    // a tight loop identical nanosecond stamps, and two tests drawing
    // the same base would delete each other's scratch mid-flight.
    let bases: Vec<PathBuf> = (0..256).map(|_| unique_base()).collect();
    let distinct: std::collections::HashSet<&Path> = bases.iter().map(|b| b.as_path()).collect();
    assert_eq!(
        bases.len(),
        distinct.len(),
        "every base must be distinct even when timestamps repeat"
    );
}

/// One scripted round: a response dispatching the capture tool, then a
/// terminal reply.
fn scripted_pair() -> Vec<MockResponse> {
    vec![
        MockResponse {
            text: "let me check".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_1".to_string(),
                name: "capture_ctx".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]
}

/// Build a loop that dispatches the capture tool once per run for the
/// next `dispatches` runs, finishing each run afterwards.
fn loop_with_dispatches(
    seen: Arc<Mutex<Vec<(String, String)>>>,
    dispatches: usize,
) -> BareLoop<MockApiClient> {
    let responses = (0..dispatches).flat_map(|_| scripted_pair()).collect();
    let client = MockApiClient::new("test-model").with_responses(responses);
    let mut registry = ToolRegistry::new();
    registry.register(CaptureCtxTool { seen });
    BareLoop::new(Arc::new(client), registry, SessionConfig::default())
}

/// Run one dispatch on the caller's loop and return the context the
/// tool saw.
async fn dispatch_and_capture(
    loop_: &mut BareLoop<MockApiClient>,
    seen: Arc<Mutex<Vec<(String, String)>>>,
) -> (String, String) {
    loop_
        .run("hello", &RunConfig::default())
        .await
        .expect("run completes");
    seen.lock()
        .expect("context log lock")
        .last()
        .cloned()
        .expect("the capture tool was dispatched")
}

#[tokio::test]
async fn dispatch_sets_a_per_session_path_under_the_override_base() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, session_id) = dispatch_and_capture(&mut loop_, seen).await;

    let path = Path::new(&temp_dir);
    assert!(
        path.starts_with(&base),
        "the session temp dir must live under the override base: {temp_dir}"
    );
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    assert!(
        file_name.starts_with("loopctl-") && file_name.contains(&session_id),
        "the subdir is named after the session: got {file_name}, session {session_id}"
    );
}

#[tokio::test]
async fn two_sessions_get_distinct_non_overlapping_dirs() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());

    let seen_a = Arc::new(Mutex::new(Vec::new()));
    let mut loop_a = loop_with_dispatches(Arc::clone(&seen_a), 1).with_temp_dir(base.clone());
    let (dir_a, _) = dispatch_and_capture(&mut loop_a, seen_a).await;
    let seen_b = Arc::new(Mutex::new(Vec::new()));
    let mut loop_b = loop_with_dispatches(Arc::clone(&seen_b), 1).with_temp_dir(base.clone());
    let (dir_b, _) = dispatch_and_capture(&mut loop_b, seen_b).await;

    assert_ne!(dir_a, dir_b, "two sessions must never share a temp dir");
    assert!(
        !Path::new(&dir_a).starts_with(&dir_b) && !Path::new(&dir_b).starts_with(&dir_a),
        "neither dir may nest inside the other: {dir_a} vs {dir_b}"
    );
}

#[tokio::test]
async fn subdir_exists_after_dispatch() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;

    assert!(
        std::fs::metadata(&temp_dir).is_ok(),
        "the lazy materialisation must create the subdir on first dispatch: {temp_dir}"
    );
}

#[tokio::test]
async fn drop_removes_the_materialised_subdir() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;
    assert!(
        std::fs::metadata(&temp_dir).is_ok(),
        "precondition: the subdir exists after dispatch"
    );

    drop(loop_);
    assert!(
        std::fs::metadata(&temp_dir).is_err(),
        "dropping the loop must remove its per-session subdir: {temp_dir}"
    );
}

#[tokio::test]
async fn into_machine_cleans_the_session_temp_via_the_dropped_husk() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;
    assert!(
        std::fs::metadata(&temp_dir).is_ok(),
        "precondition: the subdir exists after dispatch"
    );

    let _machine = loop_.into_machine();
    assert!(
        std::fs::metadata(&temp_dir).is_err(),
        "consuming the loop for its machine drops the husk, whose drop \
         removes the session temp — the checkpoint path leaks nothing: \
         {temp_dir}"
    );
}

#[tokio::test]
async fn drop_without_dispatch_creates_nothing() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    drop(loop_);

    let empty = std::fs::read_dir(&base)
        .map(|mut entries| entries.next().is_none())
        .unwrap_or(true);
    assert!(
        empty,
        "a loop that never dispatches must not touch the filesystem under the base"
    );
}

#[tokio::test]
async fn default_base_is_the_process_temp() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1);
    let (temp_dir, session_id) = dispatch_and_capture(&mut loop_, seen).await;

    let os_temp = std::env::temp_dir();
    let path = Path::new(&temp_dir);
    assert!(
        path.starts_with(&os_temp),
        "without an override the subdir lives under the OS temp: {temp_dir}"
    );
    assert!(
        temp_dir.contains(&session_id),
        "the default path is still per-session: {temp_dir}"
    );

    drop(loop_);
    assert!(
        std::fs::metadata(&temp_dir).is_err(),
        "dropping the loop must remove its per-session subdir: {temp_dir}"
    );
}

#[tokio::test]
async fn opt_out_uses_process_temp_and_drop_removes_nothing() {
    let sentinel =
        std::env::temp_dir().join(format!("loopctl-opt-out-sentinel-{}", std::process::id()));
    std::fs::write(&sentinel, "keep me").expect("wrote sentinel");

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_managed_temp_disabled();
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;
    assert_eq!(
        temp_dir,
        std::env::temp_dir().to_string_lossy(),
        "opted-out loops hand tools the process-wide temp"
    );

    drop(loop_);
    assert!(
        std::fs::metadata(&sentinel).is_ok(),
        "an opted-out drop must remove nothing from the process temp"
    );
    std::fs::remove_file(&sentinel).expect("cleaned sentinel");
}

#[tokio::test]
async fn empty_base_path_opts_out_like_the_disabled_builder() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir("");
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;
    assert_eq!(
        temp_dir,
        std::env::temp_dir().to_string_lossy(),
        "an empty base path disables managed temp, it must not become a relative dir"
    );
}

#[cfg(debug_assertions)]
#[tokio::test]
#[should_panic(expected = "configuration setters must be called before run()")]
async fn with_temp_dir_panics_after_the_session_started() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1);
    let _ = dispatch_and_capture(&mut loop_, seen).await;
    drop(loop_.with_temp_dir(unique_base()));
}

#[cfg(debug_assertions)]
#[tokio::test]
#[should_panic(expected = "configuration setters must be called before run()")]
async fn with_managed_temp_disabled_panics_after_the_session_started() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1);
    let _ = dispatch_and_capture(&mut loop_, seen).await;
    drop(loop_.with_managed_temp_disabled());
}

#[tokio::test]
async fn materialisation_failure_falls_back_to_process_temp() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 2).with_temp_dir(base.clone());

    let _ = dispatch_and_capture(&mut loop_, Arc::clone(&seen)).await;
    let (subdir, _) = seen
        .lock()
        .expect("context log lock")
        .first()
        .cloned()
        .expect("first dispatch captured");
    assert!(
        Path::new(&subdir).starts_with(&base),
        "precondition: the first dispatch materialised the per-session subdir: {subdir}"
    );

    // Block materialisation: replace the subdir with a plain file, so
    // the second dispatch's create_dir_all fails.
    std::fs::remove_dir_all(&subdir).expect("removed the subdir");
    std::fs::write(&subdir, "now a file").expect("wrote blocker file");

    let (fallback, _) = dispatch_and_capture(&mut loop_, seen).await;
    assert_eq!(
        fallback,
        std::env::temp_dir().to_string_lossy(),
        "a tool must always receive a writable path: creation failure degrades to the process temp"
    );
}

#[tokio::test]
async fn drop_is_silent_when_the_dir_is_already_gone() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;

    // Simulate a host that pre-cleaned the subdir.
    std::fs::remove_dir_all(&temp_dir).expect("pre-removed the subdir");
    drop(loop_);

    // Completing without panicking is the contract — a NotFound during
    // cleanup is swallowed, not surfaced — and the pre-removed path
    // stays gone.
    assert!(
        std::fs::metadata(&temp_dir).is_err(),
        "the pre-removed subdir must stay gone after a silent drop: {temp_dir}"
    );
}

#[tokio::test]
async fn drop_swallows_io_errors_from_cleanup() {
    let base = unique_base();
    let _guard = ScratchGuard(base.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
    let (temp_dir, _) = dispatch_and_capture(&mut loop_, seen).await;

    // Replace the subdir with a plain file: remove_dir_all then fails
    // with a non-NotFound error, exercising the warn-and-swallow path
    // without platform-specific permission games.
    std::fs::remove_dir_all(&temp_dir).expect("removed the subdir");
    std::fs::write(&temp_dir, "now a file").expect("wrote blocker file");

    drop(loop_);

    assert!(
        std::fs::metadata(&temp_dir).is_ok(),
        "the blocked removal must leave the path untouched, and drop must not panic"
    );
}

/// A deterministic linear-congruential generator.
///
/// A fixed seed keeps the sequence reproducible by iteration number.
struct Lcg(u64);

impl Lcg {
    /// Advance the generator and return the new high bits.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    /// Draw a value below `n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[tokio::test]
async fn random_lifecycle_sequences_never_leak_session_dirs() {
    let mut rng = Lcg(0xABCD_1234);
    for iter in 0..300 {
        let base = unique_base();
        std::fs::create_dir_all(&base).expect("created scratch base");
        let seen = Arc::new(Mutex::new(Vec::new()));

        let mut loop_ = loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
        let mut dispatches: u32 = 0;
        let rounds = 1 + rng.below(4);
        for _ in 0..rounds {
            match rng.below(3) {
                0 | 1 => {
                    loop_.run("q", &RunConfig::default()).await.expect("run");
                    dispatches = dispatches.saturating_add(1);
                }
                _ => {
                    // Replace the loop mid-life: via into_machine while
                    // the machine is still Start (never run), otherwise
                    // via a plain drop + fresh construction. Both paths
                    // must clean the outgoing loop's session dir.
                    if dispatches == 0 {
                        let machine = loop_.into_machine();
                        let client =
                            MockApiClient::new("test-model").with_responses(scripted_pair());
                        let mut registry = ToolRegistry::new();
                        registry.register(CaptureCtxTool {
                            seen: Arc::clone(&seen),
                        });
                        loop_ = BareLoop::from_machine(
                            machine,
                            SessionConfig::default(),
                            Arc::new(client),
                            registry,
                        )
                        .with_temp_dir(base.clone());
                    } else {
                        drop(loop_);
                        loop_ =
                            loop_with_dispatches(Arc::clone(&seen), 1).with_temp_dir(base.clone());
                    }
                }
            }
        }
        drop(loop_);

        let leftovers: Vec<String> = std::fs::read_dir(&base)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "iter {iter}: session dirs outlived their loops: {leftovers:?} \
             ({dispatches} dispatches)"
        );
        std::fs::remove_dir_all(&base).ok();
    }
}
