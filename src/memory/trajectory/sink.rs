//! Bounded asynchronous sink for the trajectory JSONL ledger.
//!
//! [`LedgerWriter`] owns a single worker thread that performs all ledger
//! filesystem work — directory creation, file opening, appends, and the
//! truncate-back repair after a failed write — off the observer's
//! callbacks. Records are handed over as pre-serialized lines through a
//! bounded queue; when the queue is full the oldest queued line is
//! dropped with a warning, preserving the capture contract that a slow
//! or failing sink never blocks or fails a run.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::error::recover_guard;

/// File name of the JSON Lines ledger inside the sink directory.
const LEDGER_FILE: &str = "trajectory.jsonl";

/// Default bound on records waiting to be written.
///
/// Sized so that a burst of run completions queues without loss while
/// bounding memory retained on behalf of a stalled sink.
const DEFAULT_QUEUE_CAPACITY: usize = 128;

/// Serializes ledger appends process-wide.
///
/// Each writer's failed-write repair truncates back to the pre-write
/// length; the lock keeps one writer's repair from clipping a record
/// another writer appended in between. Cross-process interleaving
/// remains the documented host responsibility.
static LEDGER_APPEND_LOCK: Mutex<()> = Mutex::new(());

/// The queue and its lifecycle state, shared with the worker thread.
///
/// One instance per writer: the enqueueing threads and the worker both
/// reach it through `Arc`, with `state` guarding every mutation and
/// `idle` coordinating the three waiters — the worker waiting for
/// work, `flush` waiting for drain, and drop waiting to close.
struct WriterShared {
    /// Directory the ledger is appended to.
    ///
    /// Fixed when the writer is constructed; the ledger file inside it
    /// is [`LEDGER_FILE`], created on first write.
    dir: PathBuf,

    /// Maximum records kept waiting to be written.
    ///
    /// Clamped to at least one at construction. When the queue reaches
    /// this bound, enqueuing drops the oldest record with a warning
    /// rather than blocking or failing the run.
    capacity: usize,

    /// Queue state guarded by its mutex.
    ///
    /// Holds the pending lines, the worker's in-flight count, and the
    /// closed flag; never held across filesystem I/O, so enqueues
    /// never wait on the worker's writes.
    state: Mutex<QueueState>,

    /// Condvar pairing enqueues, closes, and batch completions with
    /// the worker loop and `flush`.
    ///
    /// The associated mutex is always `state`. Wakeup safety rests on
    /// one invariant: the worker sleeps only after observing an empty
    /// queue with no batch in flight, while `flush` waits only when
    /// the queue is non-empty or a batch is in flight — so the two are
    /// never waiting at the same time and a single `notify_one` from
    /// enqueue or close cannot be misdirected away from the worker.
    /// Preserve that invariant when changing either wait condition.
    idle: Condvar,
}

/// The mutable half of [`WriterShared`], guarded by its mutex.
///
/// Split out so the lock's protected region is explicit: everything
/// the worker and enqueuers contend on lives here, while the immutable
/// configuration stays lock-free on `WriterShared` itself.
#[derive(Default)]
struct QueueState {
    /// Serialized records waiting to be written, oldest first.
    ///
    /// Bounded by [`WriterShared::capacity`]; the front element is the
    /// overflow victim when the queue is full.
    lines: VecDeque<String>,

    /// Lines currently held by the worker between drain and write
    /// completion.
    ///
    /// `flush` waits until both this and `lines` reach zero, so it
    /// returns only after in-flight writes land in the file.
    writing: usize,

    /// Whether the writer was dropped and the worker should exit.
    ///
    /// Enqueues after closing are refused with a warning; the worker
    /// drains what remains and then returns.
    closed: bool,
}

/// Appends one serialized record to a ledger directory.
///
/// Performs the open-write-close pass as a single append; a failed write
/// is repaired by truncating back to the pre-write length so a partial
/// line never corrupts the one-record-per-line contract. Best-effort:
/// any failure emits exactly one warning.
fn append_line(dir: &Path, line: &str) {
    if std::fs::create_dir_all(dir).is_err() {
        tracing::warn!(
            target: "loopctl::trajectory",
            records_lost = 1,
            dir = %dir.display(),
            "trajectory ledger directory could not be created; file output dropped"
        );
        return;
    }
    let path = dir.join(LEDGER_FILE);
    let _append_guard = recover_guard(LEDGER_APPEND_LOCK.lock());
    let write = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| {
            use std::io::Write as _;
            let start = file.metadata().map_or(0, |m| m.len());
            let mut out = String::with_capacity(line.len().saturating_add(1));
            out.push_str(line);
            out.push('\n');
            match file.write_all(out.as_bytes()) {
                Ok(()) => Ok(()),
                Err(error) => {
                    if file.set_len(start).is_err() {
                        tracing::debug!(
                            target: "loopctl::trajectory",
                            "ledger truncation after a failed write did not succeed"
                        );
                    }
                    Err(error)
                }
            }
        });
    match write {
        Ok(()) => {
            tracing::debug!(
                target: "loopctl::metrics",
                metric = "loopctl.trajectory.jsonl_bytes",
                value = line.len().saturating_add(1) as u64,
                "trajectory ledger flushed"
            );
        }
        Err(error) => {
            tracing::warn!(
                target: "loopctl::trajectory",
                error = %error,
                records_lost = 1,
                path = %path.display(),
                "trajectory ledger write failed; file output dropped"
            );
        }
    }
}

impl WriterShared {
    /// The worker loop: drain queued lines and append them.
    ///
    /// Exits once closed and fully drained; every batch is written
    /// outside the queue lock so enqueues never wait on filesystem I/O.
    fn work(self: &Arc<Self>) {
        loop {
            let mut state = recover_guard(self.state.lock());
            while state.lines.is_empty() && !state.closed {
                state = recover_guard(self.idle.wait(state));
            }
            if state.lines.is_empty() && state.closed {
                return;
            }
            let batch = std::mem::take(&mut state.lines);
            state.writing = batch.len();
            drop(state);

            for line in &batch {
                append_line(&self.dir, line);
            }

            let mut state = recover_guard(self.state.lock());
            state.writing = state.writing.saturating_sub(batch.len());
            let written_all = state.writing == 0 && state.lines.is_empty();
            drop(state);
            if written_all {
                self.idle.notify_all();
            }
        }
    }
}

/// Handle to a ledger's background writer.
///
/// Cloning is not supported; the observer holds it behind an `Arc`. When
/// the last reference is dropped the worker drains its queue and exits,
/// so records accepted before shutdown are still written.
pub(crate) struct LedgerWriter {
    /// State shared with the worker: directory, capacity, queue, condvar.
    ///
    /// Held by the `Arc` clones pairing the handle with the thread, so
    /// the worker keeps running until the last handle drops.
    inner: Arc<WriterShared>,

    /// The worker thread's join handle, when one was spawned.
    ///
    /// Joined on drop after closing the queue, guaranteeing accepted
    /// records are written before the writer goes away.
    handle: Option<JoinHandle<()>>,

    /// Whether enqueues must write inline after a failed thread spawn.
    ///
    /// Set only when spawning was attempted and failed; a deliberately
    /// unspawned writer (tests) still queues normally.
    inline: bool,
}

impl LedgerWriter {
    /// A writer with the default queue capacity for `dir`.
    ///
    /// The form the observer constructs; the bound is
    /// [`DEFAULT_QUEUE_CAPACITY`], sized to absorb bursts of run
    /// completions without loss.
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self::with_capacity(dir, DEFAULT_QUEUE_CAPACITY, true)
    }

    /// A writer with an explicit queue capacity.
    ///
    /// `spawn` starts the worker thread; when spawning fails the writer
    /// degrades to inline writes on the calling thread so capture still
    /// functions, at the cost of the responsiveness the worker provides.
    fn with_capacity(dir: PathBuf, capacity: usize, spawn: bool) -> Self {
        let inner = Arc::new(WriterShared {
            dir,
            capacity: capacity.max(1),
            state: Mutex::new(QueueState::default()),
            idle: Condvar::new(),
        });
        let handle = if spawn {
            let worker = Arc::clone(&inner);
            std::thread::Builder::new()
                .name("loopctl-trajectory-ledger".to_string())
                .spawn(move || worker.work())
                .ok()
        } else {
            None
        };
        let inline = spawn && handle.is_none();
        if inline {
            tracing::warn!(
                target: "loopctl::trajectory",
                "trajectory ledger worker could not be spawned; writes run inline"
            );
        }
        Self {
            inner,
            handle,
            inline,
        }
    }

    /// The directory this writer appends to.
    ///
    /// Exposed for diagnostics and the observer's redacted `Debug`
    /// rendering; the ledger file name inside it is fixed.
    pub(crate) fn dir(&self) -> &Path {
        &self.inner.dir
    }

    /// Queue a serialized record for writing.
    ///
    /// Never blocks on filesystem I/O. When the queue is at capacity the
    /// oldest queued record is dropped with a warning; when the worker
    /// could not be spawned, the write happens inline instead.
    pub(crate) fn enqueue(&self, line: String) {
        if self.inline {
            append_line(&self.inner.dir, &line);
            return;
        }
        let mut state = recover_guard(self.inner.state.lock());
        if state.closed {
            tracing::warn!(
                target: "loopctl::trajectory",
                records_lost = 1,
                "trajectory ledger writer already shut down; record dropped"
            );
            return;
        }
        if state.lines.len() >= self.inner.capacity {
            let dropped = state.lines.pop_front();
            if dropped.is_some() {
                tracing::warn!(
                    target: "loopctl::trajectory",
                    records_lost = 1,
                    "trajectory ledger queue full; dropped the oldest queued record"
                );
            }
        }
        state.lines.push_back(line);
        drop(state);
        self.inner.idle.notify_one();
    }

    /// Block until every record accepted before this call has been written.
    ///
    /// Intended for tests and orderly shutdown: returns once the queue
    /// is empty and no batch is mid-write. Records enqueued
    /// concurrently with this call may still be in flight when it
    /// returns; each such record gets its own later flush.
    pub(crate) fn flush(&self) {
        let mut state = recover_guard(self.inner.state.lock());
        while !(state.lines.is_empty() && state.writing == 0) {
            state = recover_guard(self.inner.idle.wait(state));
        }
    }
}

impl Drop for LedgerWriter {
    fn drop(&mut self) {
        let mut state = recover_guard(self.inner.state.lock());
        state.closed = true;
        drop(state);
        self.inner.idle.notify_one();
        if let Some(handle) = self.handle.take() {
            let _joined = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temporary ledger directory for one test.
    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("trajectory-sink-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir created");
        dir
    }

    #[test]
    fn enqueued_lines_land_in_the_ledger_in_order() {
        let dir = temp_dir("ordered");
        let writer = LedgerWriter::new(dir.clone());
        for index in 0..5 {
            writer.enqueue(format!("{{\"run\":\"{index}\"}}"));
        }
        writer.flush();

        let raw = std::fs::read_to_string(dir.join(LEDGER_FILE)).expect("ledger written");
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 5, "one line per accepted record");
        assert_eq!(lines[0], "{\"run\":\"0\"}", "queue order is write order");
        drop(writer);
        std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn a_full_queue_drops_the_oldest_record() {
        let dir = temp_dir("overflow");
        let writer = LedgerWriter::with_capacity(dir.clone(), 1, false);
        writer.enqueue("first".to_string());
        writer.enqueue("second".to_string());
        writer.enqueue("third".to_string());

        let mut state = recover_guard(writer.inner.state.lock());
        assert_eq!(
            state.lines.make_contiguous(),
            &["third".to_string()],
            "only the newest record stays queued past the cap"
        );
        drop(state);
        drop(writer);
        std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn a_zero_capacity_is_clamped_to_keep_the_newest_record() {
        let dir = temp_dir("zero-cap");
        let writer = LedgerWriter::with_capacity(dir.clone(), 0, false);
        writer.enqueue("first".to_string());
        writer.enqueue("second".to_string());

        let mut state = recover_guard(writer.inner.state.lock());
        assert_eq!(
            state.lines.make_contiguous(),
            &["second".to_string()],
            "a zero capacity clamps to one, keeping the newest record"
        );
        drop(state);
        drop(writer);
        std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn concurrent_enqueue_and_flush_never_deadlocks() {
        let dir = temp_dir("soak");
        // Capacity above the full burst keeps the documented drop-oldest
        // policy out of the picture; this pins no-loss, no-duplication,
        // and no deadlock under concurrent enqueue and flush. The burst
        // behavior *with* the default cap is the overflow test above —
        // the soak's first draft lost 74 of 800 records to that policy.
        let writer = LedgerWriter::with_capacity(dir.clone(), 1024, true);
        let writer = std::sync::Arc::new(writer);
        let mut handles = Vec::new();
        for thread_index in 0..4 {
            let writer = std::sync::Arc::clone(&writer);
            handles.push(std::thread::spawn(move || {
                for record in 0..200 {
                    writer.enqueue(format!("{{\"t\":{thread_index},\"r\":{record}}}"));
                    if record % 50 == 49 {
                        writer.flush();
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("soak thread completes");
        }
        writer.flush();
        let raw = std::fs::read_to_string(dir.join(LEDGER_FILE)).expect("ledger written");
        let count = raw.lines().count();
        assert_eq!(count, 800, "every soaked record lands exactly once");
        drop(writer);
        std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn dropping_the_writer_drains_the_queue() {
        let dir = temp_dir("drain");
        let mut lines_expected = Vec::new();
        {
            let writer = LedgerWriter::new(dir.clone());
            for index in 0..3 {
                let line = format!("{{\"run\":\"{index}\"}}");
                lines_expected.push(line.clone());
                writer.enqueue(line);
            }
        }
        let raw = std::fs::read_to_string(dir.join(LEDGER_FILE)).expect("drained on drop");
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 3, "drop joins after draining the queue");
        std::fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }
}
