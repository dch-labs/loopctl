//! Trajectory → memory extraction.
//!
//! Turns a recorded [`TrajectoryRecord`] (as written by
//! [`TrajectoryObserver`](crate::memory::trajectory::TrajectoryObserver))
//! into reusable memories: which strategies worked, which errors were hit
//! and how they were recovered, and what repetition cost turns. Mined
//! items are returned as [`ExtractedMemory`] values or written straight
//! into a [`LoopMemory`] store via [`extract_into`]; the optional
//! [`ExtractionObserver`] automates the pass at run end.
//!
//! The default [`Heuristic`](ExtractionStrategy::Heuristic) strategy is
//! deterministic and offline — no provider, no keys, no token spend. The
//! [`Llm`](ExtractionStrategy::Llm) and
//! [`Hybrid`](ExtractionStrategy::Hybrid) strategies take a
//! caller-supplied [`ApiClient`], so this module adds no dependencies.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::api::{ApiClient, StreamRequest};
use crate::error::LoopError;
use crate::memory::LoopMemory;
use crate::memory::entry::{MemoryCategory, MemoryEntry};
use crate::memory::trajectory::{TrajectoryOutcome, TrajectoryRecord, outcome_label};
use crate::message::Message;
use crate::observer::LoopObserver;
use crate::observer::context::RunEndContext;

/// One item mined from a trajectory, before it is written to memory.
///
/// Carries everything the writer needs to build a [`MemoryEntry`]: a
/// category, the learned text, optional tags, and the extractor's
/// confidence that the memory is worth keeping.
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    /// Which kind of knowledge this is — drives `MemoryEntry::category`.
    ///
    /// The category shapes what consolidation does with the memory later:
    /// how fast it decays and how it survives pruning. Mining assigns
    /// `Strategy`, `ErrorPattern`, or `Insight`, the last carrying the
    /// `optimization` tag.
    pub category: MemoryCategory,

    /// The learned statement, phrased as reusable advice.
    ///
    /// Advice shape — not a log line — is the point: the memory is retrieved
    /// into a later session's context, where an imperative lesson changes
    /// behavior and a bare event record does not. Both the heuristic
    /// templates and the LLM prompt are written to produce this shape.
    pub content: String,

    /// Free-form tags copied through to `MemoryEntry::tags`.
    ///
    /// Tags give retrieval an exact-match path on top of word overlap and
    /// let hosts filter by lesson kind. The miners tag by shape —
    /// `recovery`, `chain`, `optimization` — so a host can, for example,
    /// inject only recovery knowledge.
    pub tags: Vec<String>,

    /// Confidence the memory is worth keeping (`0.0..=1.0`).
    ///
    /// Becomes the initial `MemoryEntry::relevance` when the memory is
    /// stored; the miners cap their self-assessment below the
    /// auto-validate threshold.
    pub quality: f32,
}

/// How memories are mined from a trajectory.
///
/// The strategy is a configuration choice, not a compile-time split — all
/// three ship unconditionally and switching costs nothing but a struct
/// field.
#[derive(Debug, Clone, Default)]
pub enum ExtractionStrategy {
    /// Pattern-match the trajectory for known shapes and synthesize
    /// memories from rule-based templates.
    ///
    /// The shapes are error→recovery adjacency, successful tool chains,
    /// and repeated same-turn calls. Zero LLM cost, deterministic, and
    /// fully offline.
    #[default]
    Heuristic,

    /// Ask an [`ApiClient`] to read a summarized trajectory and emit a
    /// JSON array of memories.
    ///
    /// Higher-quality lessons than the templates produce, at the cost of
    /// tokens and a configured provider.
    Llm,

    /// Heuristic first, then an optional LLM pass over the candidates.
    ///
    /// The cheap templates always run; the provider refines, merges, and
    /// generalizes their output, and a provider failure degrades to the
    /// heuristic result.
    Hybrid,
}

/// Configuration for one extraction pass.
///
/// Defaults describe a conservative offline pass: at most 10 memories from
/// successful runs of at least 3 turns, heuristic strategy, no provider.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Which strategy to use.
    ///
    /// The default `Heuristic` works offline with no provider and is fully
    /// deterministic — what tests and zero-cost setups need. `Llm` and
    /// `Hybrid` take a client at the call site and trade tokens for
    /// generality; see [`ExtractionStrategy`] for the trade-offs.
    pub strategy: ExtractionStrategy,

    /// Ceiling on memories one trajectory may yield.
    ///
    /// Caps LLM cost and keeps a single run from flooding the store;
    /// candidates beyond the cap are dropped after sorting by quality.
    pub max_memories: usize,

    /// Skip extraction entirely below this many turns.
    ///
    /// A one-shot trajectory has nothing to generalize from, so short
    /// runs produce no memories rather than noise.
    pub min_turns: usize,

    /// Mine from non-failed runs by default.
    ///
    /// A `Partial` run's successful tool work is exactly what its grade
    /// certifies, so it counts; outright failures often contain the most
    /// instructive recovery pairs, so opt in to learn from them.
    pub include_failures: bool,

    /// Cap on the bytes of trajectory context fed to the LLM strategies.
    ///
    /// Bounds the combined prompt: the trajectory summary (newest turns
    /// first, so truncation drops the run's beginning and keeps its
    /// outcome) plus, for the `Hybrid` strategy, the heuristic candidate
    /// lines share this budget. The provider call therefore stays bounded
    /// whatever the run's size; oversized runs lose their oldest turns
    /// first — recent work, the most instructive part, survives.
    pub llm_context_budget: usize,
}

impl Default for ExtractionConfig {
    /// Returns the conservative offline defaults: heuristic strategy, at
    /// most 10 memories, runs of at least 3 turns, no outright failures,
    /// an 8,000-byte LLM context budget.
    ///
    /// Every default is overridable on the public struct; only the
    /// heuristic strategy avoids needing a provider entirely.
    fn default() -> Self {
        Self {
            strategy: ExtractionStrategy::Heuristic,
            max_memories: 10,
            min_turns: 3,
            include_failures: false,
            llm_context_budget: 8000,
        }
    }
}

/// Read a trajectory from disk and mine memories from it.
///
/// The file may hold a single serialized [`TrajectoryRecord`] or a
/// multi-run JSONL ledger; the newest parseable line is mined — torn or
/// corrupt lines are skipped wherever they sit, the same lenient read
/// the [`ExtractionObserver`] uses, so a ledger that has accumulated
/// runs still extracts. The mined items are returned without being
/// written; use [`extract_into`] for the store-writing convenience.
///
/// # Errors
///
/// - [`LoopError::Memory`] when the file cannot be read, or holds no
///   parseable record at all.
/// - [`LoopError::Memory`] when no tokio runtime is in context — the
///   ledger read runs on tokio's blocking pool, so driving this future on
///   another executor returns an error instead of panicking (the
///   [`ExtractionObserver`] skips the same condition with a warning).
/// - [`LoopError::Api`] when the `Llm` strategy's provider
///   call fails or returns unparseable output. The `Hybrid` strategy never
///   fails this way — an LLM failure falls back to the heuristic result.
///
/// # Example
///
/// ```no_run
/// use loopctl::memory::extractor::{ExtractionConfig, extract};
///
/// # async fn demo() -> Result<(), loopctl::error::LoopError> {
/// let mined = extract(
///     std::path::Path::new("trajectories/run.jsonl"),
///     &ExtractionConfig::default(),
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn extract(
    trajectory_path: &Path,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
) -> Result<Vec<ExtractedMemory>, LoopError> {
    if tokio::runtime::Handle::try_current().is_err() {
        return Err(LoopError::Memory(
            "extraction requires a tokio runtime: the ledger read runs on the blocking \
             pool, so driving this future on another executor is not supported"
                .into(),
        ));
    }
    let Some(record) = last_complete_record(trajectory_path).await? else {
        return Err(LoopError::Memory(format!(
            "no complete trajectory record in {} — the file is empty or every line \
             fails to parse",
            trajectory_path.display()
        )));
    };
    extract_from_record(&record, config, client).await
}

/// Mine memories from an in-memory record.
///
/// Shared by [`extract`] and the [`ExtractionObserver`]; applies the
/// `min_turns` and `include_failures` gates before dispatching to the
/// configured strategy.
///
/// # Errors
///
/// - [`LoopError::Api`] when the `Llm` strategy's provider
///   call fails or returns unparseable output (the `Hybrid` strategy falls
///   back to its heuristic result instead of erroring).
#[tracing::instrument(name = "memory.extract", skip_all, level = "debug")]
pub async fn extract_from_record(
    record: &TrajectoryRecord,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
) -> Result<Vec<ExtractedMemory>, LoopError> {
    if record.turns.len() < config.min_turns {
        return Ok(Vec::new());
    }
    let outright_failed = matches!(record.outcome, TrajectoryOutcome::Failure);
    if outright_failed && !config.include_failures {
        return Ok(Vec::new());
    }
    let mut fallback_outcome: Option<&'static str> = None;
    let mut memories = match config.strategy {
        ExtractionStrategy::Heuristic => heuristic_memories(record),
        ExtractionStrategy::Llm => {
            let Some(client) = client else {
                emit_attempt_metric(NO_CLIENT);
                return Err(LoopError::Api(
                    "the Llm extraction strategy requires an ApiClient".into(),
                ));
            };
            match llm_memories(record, config, client, None).await {
                Ok(mined) => mined,
                Err(failure) => {
                    emit_attempt_metric(failure.outcome);
                    return Err(failure.error);
                }
            }
        }
        ExtractionStrategy::Hybrid => {
            let candidates = heuristic_memories(record);
            match client {
                None => {
                    fallback_outcome = Some(NO_CLIENT);
                    candidates
                }
                Some(client) => {
                    match llm_memories(record, config, client, Some(&candidates)).await {
                        Ok(refined) => refined,
                        Err(failure) => {
                            fallback_outcome = Some(failure.outcome);
                            tracing::warn!(
                                target: "loopctl::memory",
                                error = %failure.error,
                                "hybrid extraction fell back to heuristic candidates"
                            );
                            candidates
                        }
                    }
                }
            }
        }
    };
    memories.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dedupe_candidates(&mut memories);
    memories.truncate(config.max_memories);
    let outcome = settle_outcome(fallback_outcome);
    emit_extraction_metrics(&memories, outcome);
    Ok(memories)
}

/// Drop candidates whose `(category, content)` pair already appears.
///
/// Mining is pattern-shaped, so one run can observe the same lesson many
/// times (twelve failed-then-recovered calls to one tool are one lesson,
/// not twelve); deduplicating before the `max_memories` truncation keeps
/// the cap from being spent on repeats that crowd out other candidates.
fn dedupe_candidates(memories: &mut Vec<ExtractedMemory>) {
    let mut seen: std::collections::HashSet<(&'static str, String)> =
        std::collections::HashSet::new();
    memories
        .retain(|memory| seen.insert((category_label(memory.category), memory.content.clone())));
}

/// Build the store-ready [`MemoryEntry`] for one mined memory.
///
/// Shared by [`extract_into`] and the spawned observer path so the two
/// cannot drift: `relevance = quality` (clamped), tags copied, and the
/// [`validated`](MemoryEntry::validated) flag set at `quality >= 0.9`.
fn entry_for(memory: &ExtractedMemory) -> MemoryEntry {
    let mut entry = MemoryEntry::new(memory.category, memory.content.clone());
    entry.relevance = memory.quality.clamp(0.0, 1.0);
    for tag in &memory.tags {
        entry = entry.with_tag(tag.clone());
    }
    if memory.quality >= 0.9 {
        entry = entry.validated();
    }
    entry
}

/// Mine memories and write them straight into a [`LoopMemory`] store.
///
/// Each [`ExtractedMemory`] becomes a [`MemoryEntry`] with
/// `relevance = quality`, the mined tags attached, and the
/// [`validated`](MemoryEntry::validated) flag set for high-confidence
/// items (`quality >= 0.9`). Returns the number written alongside the raw
/// candidates so callers can log or inspect them.
///
/// # Errors
///
/// Propagates [`extract`]'s errors plus any store failure. A mid-loop
/// failure that already carries [`LoopError::Memory`] is annotated with
/// the number of memories written so far, so callers can tell a partial
/// pass from an empty one; every other variant passes through untouched —
/// flattening a recoverable variant such as [`LoopError::Api`] into
/// `Memory` would make a retryable store failure terminal to callers.
pub async fn extract_into(
    trajectory_path: &Path,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
    store: &dyn LoopMemory,
) -> Result<(usize, Vec<ExtractedMemory>), LoopError> {
    let candidates = extract(trajectory_path, config, client).await?;
    let mut written = 0usize;
    for candidate in &candidates {
        store
            .store(entry_for(candidate))
            .await
            .map_err(|err| match err {
                LoopError::Memory(message) => {
                    LoopError::Memory(format!("{message} (after {written} memories written)"))
                }
                other => other,
            })?;
        written = written.saturating_add(1);
    }
    Ok((written, candidates))
}

/// Observer that runs extraction when a run ends.
///
/// Register it alongside a
/// [`TrajectoryObserver`](crate::memory::trajectory::TrajectoryObserver)
/// pointing at the same trajectory ledger: at `on_run_end` it reads the
/// last complete record from the ledger, mines it with the configured
/// strategy, and writes the results into the configured store on a
/// spawned task — session teardown never waits on extraction. Because the
/// trajectory writer queues records asynchronously, the just-finished
/// record may not be flushed yet when this observer fires; the pass is
/// best-effort and may then silently mine the previous run's record
/// (consolidation folds any duplicate lesson away) — nothing in the logs
/// distinguishes that case, since the observer has no run id to compare
/// against. Register this observer after the trajectory observer to
/// minimize the window. The same best-effort contract covers runtime
/// shutdown: an extraction task still queued when the host drops the
/// tokio runtime is lost without executing and without a log — hosts
/// that require every run's extraction should call [`extract_into`]
/// directly instead of relying on the observer.
pub struct ExtractionObserver {
    /// The extraction configuration applied to each mined record.
    ///
    /// Fixed at construction; a different strategy or cap means building a
    /// new observer. The observer never mutates it — observers are shared
    /// behind an `Arc`.
    config: ExtractionConfig,

    /// The trajectory ledger mined at each run end.
    ///
    /// The observer reads the ledger's newest parseable line, so this must
    /// be the ledger the run's trajectory observer writes to. Register this
    /// observer after that one to minimize the window where the finished
    /// record is still queued.
    trajectory_path: PathBuf,

    /// Optional client enabling the LLM refinement strategies.
    ///
    /// Absent, the observer stays strictly heuristic and offline regardless
    /// of configuration. Present, it is used only when the configured
    /// strategy asks for a provider — the heuristic pass itself never spends
    /// tokens.
    client: Option<Arc<dyn ApiClient>>,

    /// Where mined memories are written.
    ///
    /// Held behind an `Arc` so several loops can learn into one shared pool.
    /// Writes happen on the spawned extraction task, never on the thread
    /// firing the observer.
    store: Arc<dyn LoopMemory>,
}

impl ExtractionObserver {
    /// Build an observer mining `trajectory_path` into `store`.
    ///
    /// Uses the [`Heuristic`](ExtractionStrategy::Heuristic) strategy; add
    /// a client with [`with_client`](Self::with_client) to enable the LLM
    /// strategies.
    pub fn new(trajectory_path: impl Into<PathBuf>, store: Arc<dyn LoopMemory>) -> Self {
        Self {
            config: ExtractionConfig::default(),
            trajectory_path: trajectory_path.into(),
            client: None,
            store,
        }
    }

    /// Supply a client for the `Llm`/`Hybrid` strategies.
    ///
    /// Without a client those configurations cannot run — `Llm` errors and
    /// `Hybrid` falls back to its heuristic half — so builders that want
    /// provider-backed learning pass one here. The observer stores it behind
    /// an `Arc` and never constructs a provider itself.
    #[must_use]
    pub fn with_client(mut self, client: Arc<dyn ApiClient>) -> Self {
        self.client = Some(client);
        self
    }

    /// Override the extraction configuration.
    ///
    /// Replaces the conservative default — useful for raising
    /// `max_memories`, including failed runs, or switching strategy. Every
    /// other piece of builder state is preserved.
    #[must_use]
    pub fn with_config(mut self, config: ExtractionConfig) -> Self {
        self.config = config;
        self
    }
}

impl LoopObserver for ExtractionObserver {
    /// Returns `"memory-extractor"`, the label observers report to hosts.
    ///
    /// Constant per implementation, so hosts can key observer routing on
    /// it.
    fn name(&self) -> &'static str {
        "memory-extractor"
    }

    /// Mine the ledger's newest record into the store, off-thread.
    ///
    /// The pass runs on a spawned task so run teardown never waits on
    /// extraction; without a tokio runtime in context it skips with a
    /// warning instead of panicking. The whole contract is best-effort —
    /// see the struct docs for the flush race and shutdown-loss caveats.
    fn on_run_end(&self, _ctx: &RunEndContext) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                target: "loopctl::memory",
                "memory extraction skipped: no tokio runtime is running"
            );
            return;
        };
        let config = self.config.clone();
        let path = self.trajectory_path.clone();
        let client = self.client.clone();
        let store = Arc::clone(&self.store);
        let spawned = handle.spawn(async move {
            match last_complete_record(&path).await {
                Ok(Some(record)) => {
                    if let Err(err) =
                        extract_from_record_owned(record, &config, client.as_deref(), &*store).await
                    {
                        tracing::warn!(
                            target: "loopctl::memory",
                            error = %err,
                            "background memory extraction failed"
                        );
                    }
                }
                Ok(None) => tracing::warn!(
                    target: "loopctl::memory",
                    "no complete trajectory record available for extraction"
                ),
                Err(err) => tracing::warn!(
                    target: "loopctl::memory",
                    error = %err,
                    "background memory extraction could not read the trajectory"
                ),
            }
        });
        drop(spawned);
    }
}

/// Spawned-task variant of [`extract_from_record`] that also writes each
/// mined memory into the store, mirroring [`extract_into`]'s entry shape.
///
/// # Errors
///
/// Propagates extraction and store failures unchanged — the observer
/// logs and drops whatever this returns.
async fn extract_from_record_owned(
    record: TrajectoryRecord,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
    store: &dyn LoopMemory,
) -> Result<Vec<ExtractedMemory>, LoopError> {
    let memories = extract_from_record(&record, config, client).await?;
    for memory in &memories {
        store.store(entry_for(memory)).await?;
    }
    Ok(memories)
}

/// Ceiling on one provider answer's bytes before bracket-span parsing.
///
/// The lenient parser tries every `]` after every `[`, which is
/// quadratic on bracket-dense input; the cap bounds that worst case to
/// a size no legitimate memory array approaches (a memory is itself
/// capped at [`MAX_WIRE_CONTENT_BYTES`], so hundreds fit well below
/// this), and the parse runs on the blocking pool so even the bounded
/// worst case cannot stall a tokio worker.
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Ceiling on one ledger line's buffered bytes.
///
/// Bounds the reader's memory even when a region of the file has no
/// newline — a torn write that lost its terminator, or a blob spliced
/// into the ledger. Sized far above any record a capture-limited
/// trajectory observer produces (each turn's response is capped), so a
/// legitimate line never trips it; an over-long line is skipped like any
/// other unparseable line.
const MAX_LEDGER_LINE: usize = 16 * 1024 * 1024;

/// Read the last complete JSONL line of a trajectory ledger.
///
/// Streams the file forward line by line, keeping only the newest line
/// that parses, so memory stays flat no matter how long a session's
/// ledger grows — including newline-less regions, which are consumed in
/// bounded chunks and discarded past [`MAX_LEDGER_LINE`] instead of
/// buffering the whole tail. Lines are read as bytes and decoded per
/// line, so a torn or corrupt line — invalid UTF-8 from a split
/// multibyte write, truncated JSON, or a missing terminator — fails to
/// parse and is simply skipped, wherever it sits; only genuine I/O
/// errors abort the scan. [`None`] means nothing in the ledger parses —
/// an empty or all-torn file — so a best-effort observer can simply
/// skip.
///
/// The file read runs on the blocking thread pool, so a large ledger
/// never stalls a tokio worker.
///
/// # Errors
///
/// [`LoopError::Memory`] when the ledger cannot be read at all.
async fn last_complete_record(path: &Path) -> Result<Option<TrajectoryRecord>, LoopError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(&path)
            .map_err(|err| LoopError::Memory(format!("cannot read trajectory ledger: {err}")))?;
        let mut reader = std::io::BufReader::new(file);
        let mut last = None;
        let mut line = Vec::new();
        loop {
            let end_of_input = read_capped_line(&mut reader, &mut line, MAX_LEDGER_LINE)
                .map_err(|err| LoopError::Memory(format!("cannot read ledger line: {err}")))?;
            if end_of_input {
                break;
            }
            if let Ok(record) = serde_json::from_slice::<TrajectoryRecord>(&line) {
                last = Some(record);
            }
        }
        Ok(last)
    })
    .await
    .map_err(|err| LoopError::Memory(format!("trajectory reader task failed: {err}")))?
}

/// Read one line into `buf`, never buffering past `cap`.
///
/// Returns `true` at end of input. A line longer than the cap completes
/// as a truncated buffer — the excess is consumed and discarded — so the
/// caller's parse simply fails and the line is skipped, while the
/// reader's memory stays bounded however long the newline-less region
/// grows. `buf` always ends at the line's newline when one terminated
/// it, matching plain [`read_until`](std::io::BufRead::read_until)
/// output for in-cap lines.
///
/// # Errors
///
/// Propagates I/O errors from the underlying reader unchanged.
fn read_capped_line(
    reader: &mut impl std::io::BufRead,
    buf: &mut Vec<u8>,
    cap: usize,
) -> Result<bool, std::io::Error> {
    buf.clear();
    let mut over_cap = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(buf.is_empty());
        }
        if let Some(index) = available.iter().position(|&byte| byte == b'\n') {
            let terminator = index.saturating_add(1);
            if buf.len().saturating_add(terminator) > cap {
                over_cap = true;
            }
            if !over_cap && let Some(through_newline) = available.get(..terminator) {
                buf.extend_from_slice(through_newline);
            }
            reader.consume(terminator);
            return Ok(false);
        }
        let len = available.len();
        if buf.len().saturating_add(len) > cap {
            over_cap = true;
        }
        if !over_cap {
            buf.extend_from_slice(available);
        }
        reader.consume(len);
    }
}

/// Run all three heuristic miners over one record.
///
/// Each miner is independent — recovery pairs, successful chains, and
/// same-turn repetition — and the combined candidate list is deduplicated
/// and capped later in [`extract_from_record`].
fn heuristic_memories(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();
    memories.append(&mut mine_recoveries(record));
    memories.append(&mut mine_strategies(record));
    memories.append(&mut mine_repetition(record));
    memories
}

/// Mine error-then-recovery pairs as `ErrorPattern` memories.
///
/// A failed call whose tool succeeds in a later call — the same turn
/// or a later turn — means the model found the fix; the memory pairs
/// the failing tool with the retry advice so the next run starts from
/// the fix instead of the failure. Turns are walked once, newest
/// first, and each turn's calls with them, accumulating the tools
/// already recovered further down — one linear pass instead of a
/// rescan per failed call.
fn mine_recoveries(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();
    let mut recovered_tools: Vec<&str> = Vec::new();
    for turn in record.turns.iter().rev() {
        for call in turn.tool_calls.iter().rev() {
            if call.ok {
                if !recovered_tools.contains(&call.tool.as_str()) {
                    recovered_tools.push(call.tool.as_str());
                }
                continue;
            }
            let recovered = recovered_tools.contains(&call.tool.as_str());
            if recovered {
                memories.push(ExtractedMemory {
                    category: MemoryCategory::ErrorPattern,
                    content: format!(
                        "a {tool} call failed and a later {tool} call succeeded — retry {tool} \
                         after adjusting the input before giving up",
                        tool = call.tool
                    ),
                    tags: vec!["recovery".into()],
                    quality: 0.7,
                });
            }
        }
    }
    memories
}

/// Ceiling on one provider-supplied lesson's content, in bytes.
///
/// A degenerate or runaway answer must not persist megabyte-scale
/// lessons into the store — a stored memory is re-paid on every future
/// retrieval, so one oversized lesson is a recurring cost with no knob
/// to unwind it. Oversized items drop like unknown categories do.
const MAX_WIRE_CONTENT_BYTES: usize = 2_000;

/// Ceiling on one provider-supplied tag's length.
///
/// Tags are short filter labels; a longer one is model noise dropped
/// whole before it can bloat the stored entry — a cut label would
/// filter differently than the model intended.
const MAX_WIRE_TAG_CHARS: usize = 64;

/// Ceiling on the tag count one provider-supplied lesson may carry.
///
/// A lesson needs a handful of labels to be selectable by kind and
/// topic; beyond that the list is padding that grows every retrieved
/// copy of the memory for no filtering value. One slot is reserved for
/// the `provider-derived` provenance tag every LLM-mined memory
/// carries, so model-supplied tags alone can never fill the cap.
const MAX_WIRE_TAGS: usize = 8;

/// Cap on a tool name interpolated into heuristic mined content.
///
/// Tool names on the wire come from the model's tool calls, so a
/// hallucinated or attacker-steered name can be arbitrarily long; the
/// repetition miner is the one heuristic sink where such a name is
/// stored without first matching a successful dispatch. The cap keeps
/// the persisted lesson bounded whatever the ledger carried.
const MAX_MINED_TOOL_NAME_CHARS: usize = 64;

/// How many leading tool names a strategy memory lists before collapsing
/// the rest into a count.
///
/// A run with hundreds of successful calls still yields one bounded
/// memory: the first names carry the shape of the ordering, the "and K
/// more" tail carries its length, and the stored content never grows
/// with run length.
const CHAIN_LISTING_CAP: usize = 8;

/// Mine successful tool chains as `Strategy` memories.
///
/// A run that did not outright fail and issued three or more successful
/// calls carries a repeatable ordering; the memory abstracts the
/// observed sequence into follow-this-order advice, listing at most
/// [`CHAIN_LISTING_CAP`] names so the content stays bounded however long
/// the run was. Quality caps just below the auto-validate threshold —
/// a rule-based mining pass is not confirmation, so even long chains
/// store unvalidated.
fn mine_strategies(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let outright_failed = matches!(record.outcome, TrajectoryOutcome::Failure);
    if outright_failed {
        return Vec::new();
    }
    let mut chain: Vec<&str> = Vec::new();
    for turn in &record.turns {
        for call in &turn.tool_calls {
            if call.ok {
                chain.push(call.tool.as_str());
            }
        }
    }
    if chain.len() < 3 {
        return Vec::new();
    }
    let listed = chain
        .iter()
        .take(CHAIN_LISTING_CAP)
        .map(|tool| (*tool).to_string())
        .collect::<Vec<_>>()
        .join(" → ");
    let listing = if chain.len() > CHAIN_LISTING_CAP {
        format!(
            "{listed} → … and {} more",
            chain.len().saturating_sub(CHAIN_LISTING_CAP)
        )
    } else {
        listed
    };
    vec![ExtractedMemory {
        category: MemoryCategory::Strategy,
        content: format!(
            "a chain of {count} successful tool calls ({listing}) completed this task — \
             for similar tasks, follow this order",
            count = chain.len(),
        ),
        tags: vec!["chain".into()],
        quality: (0.5 + 0.05 * f32::from(u16::try_from(chain.len()).unwrap_or(u16::MAX))).min(0.85),
    }]
}

/// Mine same-turn repetition — the loop smell worth reporting.
///
/// Counting tool *names* across a whole run misfires (four sequential
/// `Edit` calls are ordinary work, not waste), so the heuristic reports
/// only a tool invoked three or more times **within one turn** — the
/// shape that means the model re-issued the same call instead of batching
/// or caching it. Same-run name counts stay unmined until trajectories
/// carry call fingerprints to compare.
fn mine_repetition(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();
    for turn in &record.turns {
        let mut counted: Vec<(String, usize)> = Vec::new();
        for call in &turn.tool_calls {
            if let Some(entry) = counted.iter_mut().find(|(tool, _)| *tool == call.tool) {
                entry.1 = entry.1.saturating_add(1);
            } else {
                counted.push((call.tool.clone(), 1));
            }
        }
        for (tool, count) in counted {
            if count >= 3 {
                let tool = truncate(&tool, MAX_MINED_TOOL_NAME_CHARS);
                memories.push(ExtractedMemory {
                    category: MemoryCategory::Insight,
                    content: format!(
                        "{tool} was called {count} times within one turn — batch or \
                         cache the repeated {tool} calls to save turns"
                    ),
                    tags: vec!["optimization".into(), "performance".into()],
                    quality: (0.4 + 0.1 * f32::from(u16::try_from(count).unwrap_or(u16::MAX)))
                        .min(0.8),
                });
            }
        }
    }
    memories
}

/// One element of the JSON array the LLM strategy asks the provider for.
///
/// The wire shape mirrors the prompt's requested object exactly, so serde
/// rejects a model answering in the wrong shape instead of silently
/// mis-mapping it. The optional fields tolerate models that skip tags or
/// self-assessed quality.
#[derive(Debug, Deserialize)]
struct LlmMemoryWire {
    /// Category name as the prompt defines them.
    ///
    /// Exactly `strategy`, `error_pattern`, `insight`, or
    /// `optimization`; any other string drops the whole element during
    /// mapping.
    category: String,

    /// The learned lesson, phrased as reusable advice.
    ///
    /// Taken verbatim from the model's answer; extraction does not rewrite
    /// provider text, so prompt quality is what keeps these advice-shaped.
    content: String,

    /// Free-form tags; absent means none.
    ///
    /// Copied through when present. The `optimization` and `error_pattern`
    /// categories also gain their selecting tags (`optimization`,
    /// `recovery`) if the model omitted them, keeping the filter-by-kind
    /// contract consistent across strategies.
    tags: Option<Vec<String>>,

    /// The model's self-reported confidence; absent defaults to 0.5.
    ///
    /// Clamped into `0.0..=1.0`, then capped just below the 0.9
    /// auto-validate threshold — self-assessed confidence reaches the
    /// relevance scale but never the validated flag, mirroring the
    /// heuristic miners' rule that a mining pass is not confirmation.
    /// The mid-scale default means un-self-assessed memories neither
    /// dominate the store nor vanish at the prune floor.
    quality: Option<f32>,
}

/// Render a compact, budget-capped transcript for the LLM strategies.
///
/// Turns are rendered newest-first — when a long run must be truncated,
/// the dropped part is the old beginning and the surviving summary ends
/// where the run ended, which is where the outcome lives. Apart from the
/// fixed outcome header — which always rides along, however small the
/// budget — the returned summary never exceeds `budget` bytes: the
/// truncation marker is reserved before any turn line is appended. A
/// budget smaller than the marker itself is pathological configuration;
/// there the marker still rides on top of the header, since an
/// unmarked silent truncation would lie worse. Keeps
/// what generalizes — outcome, turn index, per-turn tool names with
/// ok/error flags, and trimmed query and response text.
fn summarize_trajectory(record: &TrajectoryRecord, budget: usize) -> String {
    let header = format!(
        "outcome: {}\ntotal turns: {}\n",
        outcome_label(&record.outcome),
        record.total_turns
    );
    let marker = "…\n";
    let mut lines: Vec<String> = Vec::new();
    let mut used = header.len();
    for turn in record.turns.iter().rev() {
        let tools = turn
            .tool_calls
            .iter()
            .map(|call| {
                if call.ok {
                    format!("{}(ok)", call.tool)
                } else {
                    format!("{}(error)", call.tool)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let line = format!(
            "turn: {} | tools: {} | query: {} | response: {}\n",
            turn.turn,
            tools,
            truncate(&turn.query, 200),
            truncate(&turn.response_text, 160),
        );
        if used.saturating_add(line.len()) > budget.saturating_sub(marker.len()) {
            lines.push(marker.to_string());
            break;
        }
        used = used.saturating_add(line.len());
        lines.push(line);
    }
    let mut summary = header;
    for line in lines {
        summary.push_str(&line);
    }
    summary
}

/// Cut `text` to at most `limit` bytes on a char boundary.
///
/// The cut is marked with an ellipsis so summaries stay honest about
/// what they dropped.
fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    let mut cut = text.get(..end).unwrap_or_default().to_string();
    cut.push('…');
    cut
}

/// Run the provider-backed extraction pass.
///
/// When `heuristic_candidates` is present the prompt asks the model to
/// refine them instead of mining from scratch (the hybrid shape).
///
/// # Errors
///
/// [`LoopError::Api`] when the provider call fails or the
/// response contains no parseable JSON array.
async fn llm_memories(
    record: &TrajectoryRecord,
    config: &ExtractionConfig,
    client: &dyn ApiClient,
    heuristic_candidates: Option<&[ExtractedMemory]>,
) -> Result<Vec<ExtractedMemory>, LlmFailure> {
    /// The hybrid prompt's candidate preamble.
    ///
    /// Appended to the system instruction only when heuristic candidates
    /// ride along, telling the model to refine rather than mine from
    /// scratch; its length is reserved from the context budget before
    /// the trajectory summary is rendered.
    const PREAMBLE: &str = "\nCandidate lessons already mined from this trajectory follow; \
refine, merge, generalize, and drop the weak ones:\n";
    let mut system = String::from(
        "You are a memory extractor. Read the agent trajectory and emit JSON: an array of \
         {\"category\",\"content\",\"tags\",\"quality\"} objects. category is one of \
         \"strategy\", \"error_pattern\", \"insight\", or \"optimization\". Only emit \
         genuinely reusable lessons, phrased as advice.",
    );
    let preamble_len = if heuristic_candidates.is_some() {
        PREAMBLE.len()
    } else {
        0
    };
    let summary_budget = config
        .llm_context_budget
        .saturating_sub(system.len().saturating_add(preamble_len));
    let summary = summarize_trajectory(record, summary_budget);
    if let Some(candidates) = heuristic_candidates {
        system.push_str(PREAMBLE);
        let candidate_budget = config
            .llm_context_budget
            .saturating_sub(system.len())
            .saturating_sub(summary.len());
        append_candidate_lines(&mut system, candidates, candidate_budget);
    }
    let request = StreamRequest {
        messages: vec![Message::user(summary)],
        system: Some(system),
        tools: None,
    };
    let response = client
        .create_message(&request)
        .await
        .map_err(|err| LlmFailure {
            error: LoopError::Api(format!("extraction provider call failed: {err}")),
            outcome: "api_error",
        })?;
    let text = response.message.text_content();
    let parsed = match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle
            .spawn_blocking(move || parse_llm_memories(&text))
            .await
            .map_err(|err| LlmFailure {
                error: LoopError::Api(format!("extraction parser task failed: {err}")),
                outcome: "api_error",
            }),
        Err(_) => Ok(parse_llm_memories(&text)),
    }?;
    parsed.map_err(|error| LlmFailure {
        error,
        outcome: "parse_error",
    })
}

/// A failed provider pass carrying the telemetry outcome label.
///
/// The label is chosen at the failure site — `api_error` on the
/// transport legs, `parse_error` when the answer arrived but carried no
/// parseable array — so the `loopctl.memory.extract.attempts` split of
/// model quality from transport health is carried in the error's shape
/// rather than recovered by matching rendered message text a provider
/// body could forge.
#[derive(Debug)]
struct LlmFailure {
    /// The error the caller propagates.
    ///
    /// Always a [`LoopError::Api`] today; kept untyped so callers
    /// propagate it unchanged wherever the pass fails.
    error: LoopError,

    /// The attempt-metric outcome for the failed pass.
    ///
    /// One of `api_error` or `parse_error`, fixed at the site that
    /// produced the error.
    outcome: &'static str,
}

/// Append the hybrid candidate lines to `system`, bounded by `budget`.
///
/// A line that fits the remaining budget exactly is kept — the boundary
/// matches the summary loop's, so both budget checks treat the cap as
/// inclusive — and the first line that does not fit ends the listing.
fn append_candidate_lines(system: &mut String, candidates: &[ExtractedMemory], budget: usize) {
    let mut remaining = budget;
    for candidate in candidates {
        let line = format!(
            "- [{}] {}\n",
            category_label(candidate.category),
            candidate.content
        );
        if line.len() > remaining {
            break;
        }
        remaining = remaining.saturating_sub(line.len());
        system.push_str(&line);
    }
}

/// Parse the provider's answer: the outermost JSON array in the text,
/// mapped to [`ExtractedMemory`] values with unknown categories dropped.
///
/// Bracket pairs are matched once with a single stack pass — brackets
/// inside JSON strings do not count — and each `[` is tried only
/// against its own matching `]`, earliest start first, until one span
/// deserializes. Prose containing an earlier bracket pair ("[3
/// total]") or trailing commentary with a bracket ("(done [2 items])")
/// still does not reject a response whose array parses fine, while the
/// scan stays linear in the input instead of trying every start
/// against every later `]` — a bracket-heavy answer under the size cap
/// cannot make the pass stall. Text above [`MAX_RESPONSE_BYTES`] is
/// rejected outright before any scanning.
///
/// # Errors
///
/// [`LoopError::Api`] when no array is present or no
/// candidate span deserializes.
fn parse_llm_memories(text: &str) -> Result<Vec<ExtractedMemory>, LoopError> {
    if text.len() > MAX_RESPONSE_BYTES {
        return Err(unparseable_response());
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '[' if !in_string => open.push(index),
            ']' if !in_string => {
                if let Some(start) = open.pop() {
                    pairs.push((start, index));
                }
            }
            _ => {}
        }
    }
    pairs.sort_unstable_by_key(|&(start, _)| start);
    for (start, end) in pairs {
        let Some(span) = text.get(start..=end) else {
            continue;
        };
        if let Ok(wire) = serde_json::from_str::<Vec<LlmMemoryWire>>(span) {
            return Ok(wire.into_iter().filter_map(wire_to_extracted).collect());
        }
    }
    Err(unparseable_response())
}

/// The error for a provider answer that transports fine but carries no
/// parseable memory array.
///
/// The failure site labels its telemetry outcome `parse_error`
/// directly, so the message states the cause and carries no marker.
fn unparseable_response() -> LoopError {
    LoopError::Api("extraction response contained no parseable JSON array".into())
}

/// Map one provider-reported memory onto [`ExtractedMemory`].
///
/// Unknown category names are dropped (`None`) rather than guessed — the
/// accepted names are exactly the four the prompt defines, so the wire
/// vocabulary the parser accepts cannot drift from the one the prompt
/// requests. The `optimization` name maps to `Insight` and gains the
/// `optimization` tag it is selected by; `error_pattern` likewise gains
/// `recovery` — each kind backfills the selecting tag a host filters by,
/// keeping filter-by-kind parity with the heuristic miners. Every
/// mapped memory also gains the `provider-derived` tag: provider text
/// is untrusted — trajectory content an attacker shaped can steer the
/// answer, and the answer persists verbatim — so the tag lets hosts
/// filter, frame, or strip provider-sourced lessons at retrieval.
fn wire_to_extracted(wire: LlmMemoryWire) -> Option<ExtractedMemory> {
    let name = wire.category.to_lowercase();
    let category = match name.as_str() {
        "strategy" => MemoryCategory::Strategy,
        "error_pattern" => MemoryCategory::ErrorPattern,
        "insight" | "optimization" => MemoryCategory::Insight,
        _ => return None,
    };
    if wire.content.trim().is_empty() || wire.content.len() > MAX_WIRE_CONTENT_BYTES {
        return None;
    }
    let selecting_tag = match name.as_str() {
        "optimization" => Some("optimization"),
        "error_pattern" => Some("recovery"),
        _ => None,
    };
    let tag_budget =
        MAX_WIRE_TAGS.saturating_sub(usize::from(selecting_tag.is_some()).saturating_add(1));
    let mut tags: Vec<String> = wire
        .tags
        .unwrap_or_default()
        .into_iter()
        .filter(|tag| tag.len() <= MAX_WIRE_TAG_CHARS)
        .take(tag_budget)
        .collect();
    if let Some(selecting) = selecting_tag
        && !tags.iter().any(|t| t == selecting)
    {
        tags.push(selecting.into());
    }
    tags.push("provider-derived".into());
    Some(ExtractedMemory {
        category,
        content: wire.content,
        tags,
        quality: wire
            .quality
            .filter(|q| q.is_finite())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0)
            .min(0.85),
    })
}

/// Emit the settled-pass counters: the attempts outcome plus one
/// per-category `loopctl.memory.extracted` count per mined memory.
///
/// Category labels are the stable `snake_case` names (matching the
/// trajectory module's `outcome_label` convention), never Debug-formatted
/// variant paths.
fn emit_extraction_metrics(memories: &[ExtractedMemory], outcome: &'static str) {
    emit_attempt_metric(outcome);
    for memory in memories {
        tracing::debug!(
            target: "loopctl::metrics",
            metric = "loopctl.memory.extracted",
            category = category_label(memory.category)
        );
    }
}

/// Emit the `loopctl.memory.extract.attempts` counter for one settled
/// pass.
///
/// The outcome label carries how the pass settled, so a dashboard can
/// split provider quality from configuration mistakes.
fn emit_attempt_metric(outcome: &'static str) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.memory.extract.attempts",
        outcome
    );
}

/// The attempt outcome for a pass that never reached its provider: the
/// `Llm` strategy is configured but no client was supplied (the pass
/// errors), or the `Hybrid` strategy is in the same state (the pass
/// settles on its heuristic half).
///
/// Counted distinctly from `ok` so a host that configured a
/// provider-backed strategy without wiring a client sees the
/// misconfiguration in telemetry instead of a forever-`ok` counter —
/// or, for `Llm`, no counter at all.
const NO_CLIENT: &str = "no_client";

/// The attempt outcome a finished pass reports: the fallback reason when a
/// hybrid pass settled for its heuristic candidates — the LLM failure
/// outcome, or [`NO_CLIENT`] when no client was supplied at all — and
/// `ok` otherwise.
///
/// Exactly one attempt event is emitted per extraction pass, so a hybrid
/// fallback reports its settle reason here instead of pairing an early
/// failure event with a spurious `ok` at the settle point — the counter
/// keeps answering "is the model good enough at extraction" without
/// double-counting the pass.
fn settle_outcome(fallback: Option<&'static str>) -> &'static str {
    fallback.unwrap_or("ok")
}

/// The stable `snake_case` telemetry label for a memory category.
///
/// The labels are the contract for the `loopctl.memory.extracted`
/// counter, matching the trajectory module's outcome-label convention so
/// downstream dashboards see one vocabulary. Debug-formatted variant
/// names are deliberately not used — they are unstable presentation.
fn category_label(category: MemoryCategory) -> &'static str {
    match category {
        MemoryCategory::Strategy => "strategy",
        MemoryCategory::ErrorPattern => "error_pattern",
        MemoryCategory::Insight => "insight",
        MemoryCategory::Fact => "fact",
        MemoryCategory::Working => "working",
        MemoryCategory::Trajectory => "trajectory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::LoopMemory;
    use crate::memory::builtin::InMemoryStore;
    use crate::memory::trajectory::{TokenSummary, TrajectoryToolCall, TrajectoryTurn};

    #[test]
    fn summary_boundary_budget_never_exceeds_the_cap() {
        let mut boundary_turns: Vec<TrajectoryTurn> = Vec::new();
        for index in 0..60 {
            let mut entry_turn = turn(index, "filler", vec![call("Read", true)]);
            entry_turn.query = format!("query {index}");
            boundary_turns.push(entry_turn);
        }
        let boundary_record = record(TrajectoryOutcome::Success, boundary_turns);
        for budget in (200..=1_500).step_by(7) {
            let summary = summarize_trajectory(&boundary_record, budget);
            assert!(
                summary.len() <= budget,
                "budget {budget}: summary is {} bytes — the marker must be reserved, \
                never appended on top",
                summary.len()
            );
        }
    }

    #[cfg(feature = "testing")]
    use crate::testing::MockApiClient;

    use std::future::Future;
    use std::pin::Pin;

    /// Store double whose `store` fails after a chosen number of writes
    /// with a chosen error shape, so `extract_into`'s error mapping can be
    /// exercised without a real backend.
    struct FailingStore {
        fail_after: usize,
        memory_variant: bool,
        writes: std::sync::Mutex<usize>,
    }

    impl FailingStore {
        fn failing_on_write(fail_after: usize, memory_variant: bool) -> Self {
            Self {
                fail_after,
                memory_variant,
                writes: std::sync::Mutex::new(0),
            }
        }
    }

    impl LoopMemory for FailingStore {
        fn store(
            &self,
            _entry: MemoryEntry,
        ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
            Box::pin(async move {
                let mut writes = crate::error::recover_guard(self.writes.lock());
                *writes = writes.saturating_add(1);
                if *writes > self.fail_after {
                    return if self.memory_variant {
                        Err(LoopError::Memory("disk full".into()))
                    } else {
                        Err(LoopError::Api("provider unreachable".into()))
                    };
                }
                Ok(())
            })
        }

        fn retrieve<'a>(
            &'a self,
            _query: &'a str,
            _limit: usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, LoopError>> + Send + 'a>>
        {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn consolidate(
            &self,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::memory::ConsolidationStats, LoopError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::memory::ConsolidationStats {
                    entries_before: 0,
                    entries_after: 0,
                    pruned: 0,
                    merged: 0,
                    bytes_saved: 0,
                })
            })
        }

        fn len(&self) -> usize {
            0
        }
    }

    fn call(tool: &str, ok: bool) -> TrajectoryToolCall {
        TrajectoryToolCall {
            tool_call_id: format!("{tool}-{}", if ok { "ok" } else { "err" }),
            tool: tool.to_string(),
            ok,
            duration_ms: 10,
        }
    }

    fn turn(index: usize, query: &str, calls: Vec<TrajectoryToolCall>) -> TrajectoryTurn {
        TrajectoryTurn {
            turn: index,
            query: query.to_string(),
            response_text: format!("response {index}"),
            tool_calls: calls,
            duration_ms: 20,
            input_tokens: 5,
            output_tokens: 5,
        }
    }

    fn record(outcome: TrajectoryOutcome, turns: Vec<TrajectoryTurn>) -> TrajectoryRecord {
        TrajectoryRecord {
            session_id: "session".into(),
            run_id: "run".into(),
            outcome,
            started_at: "2026-09-06T00:00:00Z".into(),
            duration_ms: 100,
            total_turns: turns.len(),
            token_summary: TokenSummary::default(),
            turns,
        }
    }

    fn write_record(record: &TrajectoryRecord) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "loopctl-extractor-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let json = serde_json::to_string(record).expect("record serializes");
        std::fs::write(&path, json).expect("record file writes");
        path
    }

    #[tokio::test]
    async fn short_trajectory_yields_no_memories() {
        let record = record(
            TrajectoryOutcome::Success,
            vec![turn(0, "one shot", vec![call("Bash", true)])],
        );
        let config = ExtractionConfig::default();
        let mined = extract_from_record(&record, &config, None)
            .await
            .expect("short trajectories are not an error");
        assert!(
            mined.is_empty(),
            "a one-turn run has nothing worth generalizing"
        );
    }

    #[tokio::test]
    async fn failed_trajectory_mines_only_when_failures_are_included() {
        let mut failing = record(
            TrajectoryOutcome::Failure,
            vec![
                turn(0, "start", vec![call("Bash", false)]),
                turn(1, "fix it", vec![call("Bash", true)]),
                turn(2, "again", vec![call("Edit", true)]),
            ],
        );
        failing.outcome = TrajectoryOutcome::Failure;
        let strict = ExtractionConfig::default();
        let mined = extract_from_record(&failing, &strict, None)
            .await
            .expect("ok");
        assert!(mined.is_empty(), "failures are skipped by default");
        let mut including = ExtractionConfig::default();
        including.include_failures = true;
        let mined = extract_from_record(&failing, &including, None)
            .await
            .expect("ok");
        assert!(
            mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::ErrorPattern),
            "an error-then-recovery pair must yield a recovery memory"
        );
    }

    #[tokio::test]
    async fn heuristic_mines_recovery_from_error_then_success() {
        let recovering = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "run the build", vec![call("Bash", false)]),
                turn(1, "adjust and retry", vec![call("Bash", true)]),
                turn(2, "finish", vec![call("Edit", true)]),
            ],
        );
        let mined = extract_from_record(&recovering, &ExtractionConfig::default(), None)
            .await
            .expect("heuristic extraction never needs a provider");
        let recovery = mined
            .iter()
            .find(|memory| memory.category == MemoryCategory::ErrorPattern)
            .expect("the error-then-retry shape must yield an ErrorPattern");
        assert!(
            recovery.content.contains("Bash"),
            "the memory names the failing tool"
        );
        assert!(recovery.quality > 0.0, "a recovered error is worth keeping");
    }

    #[tokio::test]
    async fn a_same_turn_success_before_the_failure_is_not_a_recovery() {
        let backwards = record(
            TrajectoryOutcome::Success,
            vec![
                turn(
                    0,
                    "succeed then break",
                    vec![call("Bash", true), call("Bash", false)],
                ),
                turn(1, "carry on", vec![call("Edit", true)]),
                turn(2, "finish", vec![call("Read", true)]),
            ],
        );
        let mined = extract_from_record(&backwards, &ExtractionConfig::default(), None)
            .await
            .expect("heuristic extraction never needs a provider");
        assert!(
            !mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::ErrorPattern),
            "a success that preceded the failure did not recover from it"
        );
    }

    #[tokio::test]
    async fn a_same_turn_retry_after_the_failure_is_a_recovery() {
        let retried = record(
            TrajectoryOutcome::Success,
            vec![
                turn(
                    0,
                    "fail then adjust",
                    vec![call("Bash", false), call("Bash", true)],
                ),
                turn(1, "carry on", vec![call("Edit", true)]),
                turn(2, "finish", vec![call("Read", true)]),
            ],
        );
        let mined = extract_from_record(&retried, &ExtractionConfig::default(), None)
            .await
            .expect("heuristic extraction never needs a provider");
        assert!(
            mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::ErrorPattern),
            "a retry that succeeded after the failure is a recovery"
        );
    }

    #[tokio::test]
    async fn repeated_recovery_pairs_dedupe_to_one_memory() {
        let flaky = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "try", vec![call("Bash", false), call("Bash", false)]),
                turn(1, "retry", vec![call("Bash", true)]),
                turn(2, "more", vec![call("Bash", false), call("Bash", false)]),
                turn(3, "recover", vec![call("Bash", true), call("Edit", true)]),
            ],
        );
        let mut config = ExtractionConfig::default();
        config.max_memories = 10;
        let mined = extract_from_record(&flaky, &config, None)
            .await
            .expect("ok");
        let recoveries = mined
            .iter()
            .filter(|memory| memory.category == MemoryCategory::ErrorPattern)
            .count();
        assert_eq!(
            recoveries, 1,
            "four failed-then-recovered Bash calls are one lesson, not four"
        );
    }

    #[tokio::test]
    async fn heuristic_mines_strategy_from_successful_chains() {
        let successful = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "edit", vec![call("Edit", true), call("Edit", true)]),
                turn(2, "check", vec![call("Bash", true)]),
            ],
        );
        let mined = extract_from_record(&successful, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        assert!(
            mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::Strategy),
            "a successful multi-tool chain must yield a Strategy memory"
        );
    }

    #[tokio::test]
    async fn partial_runs_mine_their_successful_chains_by_default() {
        let partial = record(
            TrajectoryOutcome::Partial,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "edit", vec![call("Edit", true), call("Edit", true)]),
                turn(2, "verify", vec![call("Bash", true)]),
            ],
        );
        let mined = extract_from_record(&partial, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        assert!(
            mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::Strategy),
            "a partial run's successful tool chain is exactly what its grade certifies — \
            it must mine without opting into outright failures"
        );
    }

    #[tokio::test]
    async fn long_chains_cap_their_listing_and_keep_the_count() {
        let calls: Vec<TrajectoryToolCall> = (0..30)
            .map(|index| call(&format!("Tool{index}"), true))
            .collect();
        let long_chain = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "start", calls),
                turn(1, "more", vec![call("Read", true)]),
                turn(2, "end", vec![call("Read", true)]),
            ],
        );
        let mined = extract_from_record(&long_chain, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        let strategy = mined
            .iter()
            .find(|memory| memory.category == MemoryCategory::Strategy)
            .expect("the chain yields a strategy");
        assert!(
            strategy
                .content
                .contains("a chain of 32 successful tool calls"),
            "the memory names the true chain length: {}",
            strategy.content
        );
        assert!(
            strategy.content.contains("… and 24 more"),
            "the listing collapses past the cap instead of embedding every call: {}",
            strategy.content
        );
        assert!(
            !strategy.content.contains("Tool8"),
            "names past the cap are dropped from the stored content"
        );
    }

    #[test]
    fn extract_without_a_tokio_runtime_errors_instead_of_panicking() {
        let path = std::env::temp_dir().join("loopctl-extractor-no-runtime.jsonl");
        let result = std::thread::spawn(move || {
            futures::executor::block_on(extract(&path, &ExtractionConfig::default(), None))
        })
        .join()
        .expect("the extraction future must not panic off-runtime");
        match result {
            Err(LoopError::Memory(message)) => assert!(
                message.contains("tokio runtime"),
                "the error names the missing runtime: {message}"
            ),
            other => panic!("expected a Memory error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_reads_the_newest_record_from_an_accumulating_ledger() {
        let unminable = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "plain", vec![call("Read", true)]),
                turn(1, "work", vec![call("Read", true)]),
                turn(2, "done", vec![call("Read", true)]),
            ],
        );
        let minable = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let path = std::env::temp_dir().join(format!(
            "loopctl-extractor-ledger-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let first = serde_json::to_string(&unminable).expect("serializes");
        let second = serde_json::to_string(&minable).expect("serializes");
        std::fs::write(&path, format!("{first}\n{second}\n")).expect("ledger writes");
        let mined = extract(&path, &ExtractionConfig::default(), None)
            .await
            .expect("an accumulating ledger is a valid extraction source");
        std::fs::remove_file(&path).ok();
        assert!(
            mined
                .iter()
                .any(|memory| memory.category == MemoryCategory::ErrorPattern),
            "the newest record is mined — the reliable path must read the same \
            lenient ledger the observer does, not fail on trailing characters"
        );
    }

    #[test]
    fn an_overlong_ledger_line_is_skipped_without_buffering_it() {
        let mut overlong = vec![b'A'; 512];
        overlong.push(b'\n');
        overlong.extend_from_slice(br#"{"category":"insight"}"#);
        overlong.push(b'\n');
        let mut reader = std::io::Cursor::new(overlong);
        let mut line = Vec::new();
        let end_of_input = read_capped_line(&mut reader, &mut line, 64).expect("read");
        assert!(
            !end_of_input && line.len() <= 65,
            "the over-long line completes truncated at the cap — the reader never \
            buffers the whole newline-less region: {} bytes",
            line.len()
        );
        assert!(
            serde_json::from_slice::<serde_json::Value>(&line).is_err(),
            "a truncated line fails to parse and is skipped like any corrupt line"
        );
        read_capped_line(&mut reader, &mut line, 64).expect("read");
        assert!(
            serde_json::from_slice::<serde_json::Value>(&line).is_ok(),
            "the following in-cap line still parses"
        );
    }

    #[tokio::test]
    async fn a_final_line_without_a_newline_still_parses() {
        let complete = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let json = serde_json::to_string(&complete).expect("serializes");
        let path = std::env::temp_dir().join(format!(
            "loopctl-extractor-nonewline-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, json).expect("ledger writes");
        let found = last_complete_record(&path)
            .await
            .expect("read")
            .expect("the final line parses even without its newline");
        std::fs::remove_file(&path).ok();
        assert_eq!(found.run_id, "run");
    }

    #[test]
    fn oversized_provider_content_drops_and_runaway_tags_drop() {
        let oversized = LlmMemoryWire {
            category: "insight".into(),
            content: "x".repeat(MAX_WIRE_CONTENT_BYTES + 1),
            tags: None,
            quality: Some(0.5),
        };
        assert!(
            wire_to_extracted(oversized).is_none(),
            "a runaway lesson must not persist into the store"
        );
        let tagged = LlmMemoryWire {
            category: "insight".into(),
            content: "a real lesson".into(),
            tags: Some((0..20).map(|i| format!("tag-{i}")).collect()),
            quality: Some(0.5),
        };
        let mined = wire_to_extracted(tagged).expect("parses");
        assert_eq!(
            mined.tags.len(),
            MAX_WIRE_TAGS,
            "the tag list trims to the cap: {:?}",
            mined.tags
        );
    }

    #[test]
    fn an_oversized_provider_answer_is_a_parse_error_before_scanning() {
        let valid = r#"[{"category":"insight","content":"the lesson","quality":0.5}]"#;
        let huge = format!("{}{}", "[0]".repeat(MAX_RESPONSE_BYTES / 3 + 1), valid);
        let err = parse_llm_memories(&huge).expect_err("oversized answers reject outright");
        assert!(
            err.to_string().contains("no parseable JSON array"),
            "the rejection is the parse failure: {err}"
        );
    }

    #[test]
    fn a_self_assessed_quality_never_reaches_the_validated_tier() {
        let confident = LlmMemoryWire {
            category: "insight".into(),
            content: "a lesson the model graded itself highly".into(),
            tags: None,
            quality: Some(0.99),
        };
        let mined = wire_to_extracted(confident).expect("parses");
        assert!(
            mined.quality <= 0.85,
            "self-assessed confidence reaches the relevance scale but never \
            the 0.9 auto-validate threshold — mining is not confirmation: {}",
            mined.quality
        );
        let entry = entry_for(&mined);
        assert!(
            !entry.validated,
            "a wire-derived memory must not store pre-validated"
        );
    }

    #[test]
    fn trailing_prose_with_a_bracket_does_not_reject_the_array() {
        let text = "[{\"category\":\"insight\",\"content\":\"a real lesson\",\"quality\":0.5}]\n(done [2 items])";
        let mined = parse_llm_memories(text).expect("the array behind the trailing prose parses");
        assert_eq!(mined.len(), 1);
        assert_eq!(mined[0].content, "a real lesson");
    }

    #[test]
    fn prose_with_an_earlier_bracket_pair_still_parses() {
        let text = "Here are the lessons [3 total]:\n[{\"category\":\"insight\",\
                    \"content\":\"a real lesson\",\"quality\":0.5}]";
        let mined = parse_llm_memories(text).expect("the array behind the prose parses");
        assert_eq!(
            mined.len(),
            1,
            "the bracketed prose does not reject the response"
        );
        assert_eq!(mined[0].content, "a real lesson");
    }

    #[test]
    fn a_bracket_flood_under_the_cap_parses_the_real_array() {
        let flood = format!(
            "{}{}",
            "[0]".repeat(10_000),
            "[{\"category\":\"insight\",\"content\":\"the survivor\",\"quality\":0.5}]",
        );
        assert!(flood.len() < MAX_RESPONSE_BYTES);
        let mined = parse_llm_memories(&flood)
            .expect("the flood degrades to fast parse failures; the real array parses");
        assert_eq!(mined.len(), 1);
        assert_eq!(mined[0].content, "the survivor");
    }

    #[test]
    fn nested_arrays_fall_through_to_the_inner_memory_array() {
        let text = "[[{\"category\":\"insight\",\"content\":\"nested lesson\",\"quality\":0.5}]]";
        let mined = parse_llm_memories(text).expect("the inner array carries the memories");
        assert_eq!(mined.len(), 1);
        assert_eq!(mined[0].content, "nested lesson");
    }

    #[test]
    fn candidate_lines_keep_an_exact_fit_and_stop_at_the_cap() {
        let candidates: Vec<ExtractedMemory> = (0..3)
            .map(|_| ExtractedMemory {
                category: MemoryCategory::Insight,
                content: "abc".to_string(),
                tags: Vec::new(),
                quality: 0.5,
            })
            .collect();
        let line = "- [insight] abc\n";
        let mut exact = String::new();
        append_candidate_lines(&mut exact, &candidates, line.len() * 2);
        assert_eq!(
            exact.len(),
            line.len() * 2,
            "two exactly-fitting lines are kept — the cap is inclusive"
        );
        let mut overflow = String::new();
        append_candidate_lines(&mut overflow, &candidates, line.len() * 2 - 1);
        assert_eq!(
            overflow.len(),
            line.len(),
            "the third line one byte over budget stops the listing"
        );
    }

    #[tokio::test]
    async fn heuristic_mines_repetition_as_optimization_insight() {
        let same_turn = record(
            TrajectoryOutcome::Success,
            vec![
                turn(
                    0,
                    "search",
                    vec![call("Grep", true), call("Grep", true), call("Grep", true)],
                ),
                turn(1, "edit", vec![call("Edit", true)]),
                turn(2, "verify", vec![call("Bash", true)]),
            ],
        );
        let mined = extract_from_record(&same_turn, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        let insight = mined
            .iter()
            .find(|memory| memory.tags.iter().any(|tag| tag == "optimization"))
            .expect("three same-turn repeats must yield a tagged optimization insight");
        assert_eq!(
            insight.category,
            MemoryCategory::Insight,
            "optimization learnings reuse the Insight category, tagged"
        );
        assert!(
            insight.content.contains("within one turn"),
            "the memory describes the same-turn shape it mined"
        );
        let cross_turn = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "look", vec![call("Edit", true)]),
                turn(1, "look again", vec![call("Edit", true)]),
                turn(2, "look once more", vec![call("Edit", true)]),
            ],
        );
        let mined = extract_from_record(&cross_turn, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        assert!(
            !mined
                .iter()
                .any(|memory| memory.tags.iter().any(|tag| tag == "optimization")),
            "sequential same-tool calls across turns are ordinary work, not waste"
        );
    }

    #[tokio::test]
    async fn a_runaway_tool_name_is_bounded_in_the_mined_repetition_lesson() {
        let long_name = "x".repeat(500);
        let runaway = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "look", vec![call("Read", true)]),
                turn(1, "think", vec![call("Read", true)]),
                turn(
                    2,
                    "flail",
                    vec![
                        call(&long_name, false),
                        call(&long_name, false),
                        call(&long_name, false),
                    ],
                ),
            ],
        );
        let mined = extract_from_record(&runaway, &ExtractionConfig::default(), None)
            .await
            .expect("ok");
        let insight = mined
            .iter()
            .find(|memory| memory.tags.iter().any(|tag| tag == "optimization"))
            .expect("three repeats of one name still mine the lesson");
        assert!(
            insight.content.len() < 300,
            "a hallucinated half-kilobyte tool name must not persist into the \
            store unbounded: {} bytes",
            insight.content.len()
        );
        assert!(
            insight.content.contains('…'),
            "the cut name is marked rather than silently shortened"
        );
    }

    #[tokio::test]
    async fn max_memories_caps_the_candidate_list() {
        let busy = record(
            TrajectoryOutcome::Success,
            vec![
                turn(
                    0,
                    "explore",
                    vec![call("Grep", true), call("Grep", true), call("Grep", true)],
                ),
                turn(1, "fail then fix", vec![call("Bash", false)]),
                turn(2, "retry", vec![call("Bash", true)]),
                turn(
                    3,
                    "wrap up",
                    vec![call("Edit", true), call("Write", true), call("Read", true)],
                ),
            ],
        );
        let mut config = ExtractionConfig::default();
        config.max_memories = 1;
        let mined = extract_from_record(&busy, &config, None).await.expect("ok");
        assert_eq!(
            mined.len(),
            1,
            "the cap keeps only the highest-quality candidate"
        );
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn llm_strategy_parses_the_provider_array() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "one", vec![call("Read", true)]),
                turn(1, "two", vec![call("Edit", true)]),
                turn(2, "three", vec![call("Bash", true)]),
            ],
        );
        let client = MockApiClient::new("test-model").with_text_response(
            "[{\"category\":\"strategy\",\"content\":\"write tests first\",\"tags\":[\"tdd\"],\"quality\":0.9},\
              {\"category\":\"optimization\",\"content\":\"batch grep calls\",\"quality\":0.6}]",
        );
        let mut config = ExtractionConfig::default();
        config.strategy = ExtractionStrategy::Llm;
        let mined = extract_from_record(&trajectory, &config, Some(&client))
            .await
            .expect("a valid array parses");
        assert_eq!(mined.len(), 2, "both provider memories survive mapping");
        assert_eq!(mined[0].category, MemoryCategory::Strategy);
        assert!(
            mined[1].tags.iter().any(|tag| tag == "optimization"),
            "the optimization category maps to Insight with the optimization tag"
        );
        let broken = MockApiClient::new("test-model").with_text_response("no array here");
        let err = extract_from_record(&trajectory, &config, Some(&broken))
            .await
            .expect_err("a response without an array must be an Api error");
        assert!(
            matches!(err, LoopError::Api(_)),
            "malformed provider output surfaces as LoopError::Api"
        );
        assert!(
            err.to_string().contains("no parseable JSON array"),
            "a response that transports fine but carries no array fails as a \
            parse failure: {err}"
        );
        let transport = MockApiClient::new("test-model").with_errors(vec![Some("boom".into())]);
        let err = extract_from_record(&trajectory, &config, Some(&transport))
            .await
            .expect_err("a provider transport failure is an error");
        assert!(
            err.to_string().contains("provider call failed"),
            "a transport failure carries the transport message: {err}"
        );
    }

    #[test]
    fn provider_slips_are_dropped_or_sanitized_during_wire_mapping() {
        let tagged = LlmMemoryWire {
            category: "insight".into(),
            content: "a real lesson".into(),
            tags: Some((0..20).map(|i| format!("tag-{i}")).collect()),
            quality: Some(0.5),
        };
        let mined = wire_to_extracted(tagged).expect("parses");
        assert!(
            mined.tags.iter().any(|tag| tag == "provider-derived"),
            "every LLM-mined memory carries the provenance tag even when \
            the model filled the tag budget: {:?}",
            mined.tags
        );
        assert_eq!(
            mined.tags.len(),
            MAX_WIRE_TAGS,
            "the provenance slot keeps the total at the cap"
        );

        let backfilled = LlmMemoryWire {
            category: "error_pattern".into(),
            content: "retry after adjusting the input".into(),
            tags: Some((0..20).map(|i| format!("tag-{i}")).collect()),
            quality: Some(0.5),
        };
        let mined = wire_to_extracted(backfilled).expect("parses");
        assert_eq!(
            mined.tags.len(),
            MAX_WIRE_TAGS,
            "a backfilled selecting tag and the provenance tag both fit within the \
            cap: {:?}",
            mined.tags
        );
        assert!(
            mined.tags.iter().any(|tag| tag == "recovery")
                && mined.tags.iter().any(|tag| tag == "provider-derived"),
            "the selecting tag is backfilled and the provenance tag survives the squeeze"
        );

        let empty = LlmMemoryWire {
            category: "insight".into(),
            content: "   ".into(),
            tags: None,
            quality: Some(0.9),
        };
        assert!(
            wire_to_extracted(empty).is_none(),
            "an empty or whitespace-only lesson is not a memory — it must drop like an \
            unknown category, not store as validated text"
        );

        let untagged_recovery = LlmMemoryWire {
            category: "error_pattern".into(),
            content: "retry the failed call after fixing the input".into(),
            tags: None,
            quality: Some(0.7),
        };
        let mined = wire_to_extracted(untagged_recovery)
            .expect("an error-pattern lesson with no tags still mines");
        assert!(
            mined.tags.iter().any(|tag| tag == "recovery"),
            "the error_pattern category backfills the recovery tag it is selected by, \
            keeping filter-by-kind parity with the heuristic miner: {:?}",
            mined.tags
        );

        let out_of_vocabulary = LlmMemoryWire {
            category: "recovery".into(),
            content: "a real lesson".into(),
            tags: None,
            quality: Some(0.8),
        };
        assert!(
            wire_to_extracted(out_of_vocabulary).is_none(),
            "a category the prompt never defined is dropped, not silently re-mapped — \
            `recovery` is the heuristic tag vocabulary, not a prompt category"
        );

        for poisoned in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let wire = LlmMemoryWire {
                category: "insight".into(),
                content: "a real lesson".into(),
                tags: None,
                quality: Some(poisoned),
            };
            let mined =
                wire_to_extracted(wire).expect("a real lesson with a poisoned quality still mines");
            assert!(
                mined.quality.is_finite(),
                "a non-finite quality must never reach the store's relevance"
            );
            assert!(
                (mined.quality - 0.5).abs() < f32::EPSILON,
                "a non-finite quality falls back to the mid-scale default, not a clamp \
                that passes NaN through: {}",
                mined.quality
            );
        }
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn hybrid_falls_back_to_heuristic_on_provider_failure() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let failing = MockApiClient::new("test-model").with_errors(vec![Some("boom".into())]);
        let mut config = ExtractionConfig::default();
        config.strategy = ExtractionStrategy::Hybrid;
        let mined = extract_from_record(&trajectory, &config, Some(&failing))
            .await
            .expect("hybrid never errors on provider failure");
        assert!(
            !mined.is_empty(),
            "the heuristic candidates survive the provider failure"
        );
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn hybrid_candidate_lines_label_categories_in_the_wire_vocabulary() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let mock = MockApiClient::new("test-model").with_text_response(
            r#"[{"category":"strategy","content":"refined lesson","tags":[],"quality":0.8}]"#,
        );
        let mut config = ExtractionConfig::default();
        config.strategy = ExtractionStrategy::Hybrid;
        let mined = extract_from_record(&trajectory, &config, Some(&mock))
            .await
            .expect("the mock returns a parseable array");
        let requests = mock.captured_requests();
        assert_eq!(requests.len(), 1, "one provider call was made");
        let system = requests[0].system.as_deref().unwrap_or_default();
        assert!(
            system.contains("[error_pattern]"),
            "candidate labels must use the snake_case names the parser accepts: {system}"
        );
        assert!(
            !system.contains("[Strategy]") && !system.contains("[ErrorPattern]"),
            "Debug names a model might echo back are silently dropped by the parser: \
            {system}"
        );
        assert!(
            mined
                .iter()
                .any(|memory| memory.content == "refined lesson"),
            "the refined candidate survives parsing"
        );
    }

    #[tokio::test]
    async fn extract_into_writes_mined_memories_to_the_store() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let path = write_record(&trajectory);
        let store = InMemoryStore::new();
        let (written, candidates) = extract_into(&path, &ExtractionConfig::default(), None, &store)
            .await
            .expect("mining a freshly written record succeeds");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            written,
            candidates.len(),
            "every candidate is written to the store"
        );
        assert_eq!(
            store.len(),
            candidates.len(),
            "the store grew by the candidate count"
        );
        assert!(
            candidates
                .iter()
                .any(|memory| memory.category == MemoryCategory::ErrorPattern),
            "the recovery pair was mined from disk"
        );
    }

    #[tokio::test]
    async fn extract_into_annotates_memory_errors_and_passes_other_variants_through() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Edit", true), call("Bash", true)]),
            ],
        );
        let path = write_record(&trajectory);

        let recoverable = FailingStore::failing_on_write(1, false);
        let err = extract_into(&path, &ExtractionConfig::default(), None, &recoverable)
            .await
            .expect_err("the second store call fails");
        assert!(
            matches!(err, LoopError::Api(_)),
            "a recoverable store error must round-trip as its own variant, not flatten \
            to the terminal Memory variant: {err}"
        );

        let annotated = FailingStore::failing_on_write(1, true);
        let err = extract_into(&path, &ExtractionConfig::default(), None, &annotated)
            .await
            .expect_err("the second store call fails");
        match err {
            LoopError::Memory(message) => assert!(
                message.contains("after 1 memories written"),
                "the Memory variant carries the partial-written count: {message}"
            ),
            other => panic!("expected the Memory variant, got: {other}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn a_torn_trailing_ledger_line_still_yields_the_complete_record() {
        let complete = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let json = serde_json::to_string(&complete).expect("serializes");
        let path = std::env::temp_dir().join(format!(
            "loopctl-extractor-test-torn-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, format!("{json}\n{{\"session_id\":\"x\",\"run_i"))
            .expect("ledger writes");
        let found = last_complete_record(&path)
            .await
            .expect("a torn tail is not an error")
            .expect("the complete line is found");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            found.run_id, "run",
            "the complete record is extracted despite the torn trailing line"
        );
    }

    #[tokio::test]
    async fn a_corrupt_line_does_not_abort_the_ledger_scan() {
        let path = std::env::temp_dir().join(format!(
            "loopctl-extractor-corrupt-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let mut bytes = serde_json::to_vec(&record(
            TrajectoryOutcome::Success,
            vec![turn(0, "read", vec![call("Read", true)])],
        ))
        .expect("record serializes");
        bytes.push(b'\n');
        bytes.extend_from_slice(b"\xff\xfe not utf-8 at all\n");
        bytes.extend_from_slice(
            &serde_json::to_vec(&record(
                TrajectoryOutcome::Success,
                vec![
                    turn(0, "check", vec![call("Bash", true)]),
                    turn(1, "fix", vec![call("Edit", true)]),
                    turn(2, "done", vec![call("Bash", true)]),
                ],
            ))
            .expect("record serializes"),
        );
        std::fs::write(&path, &bytes).expect("ledger writes");

        let found = last_complete_record(&path)
            .await
            .expect("a corrupt line is skipped, not fatal");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            found.expect("the newest valid record is found").total_turns,
            3,
            "the scan continues past the corrupt line to the newest valid record"
        );
    }

    #[test]
    fn a_hybrid_fallback_settles_on_the_failure_outcome() {
        assert_eq!(settle_outcome(None), "ok");
        assert_eq!(
            settle_outcome(Some("api_error")),
            "api_error",
            "the fallback pass reports the LLM failure, not a second ok"
        );
        assert_eq!(settle_outcome(Some("parse_error")), "parse_error");
        assert_eq!(
            settle_outcome(Some(NO_CLIENT)),
            NO_CLIENT,
            "a configured-for-hybrid pass with no client reports no_client, not ok"
        );
    }

    #[tokio::test]
    async fn a_clientless_hybrid_pass_mines_heuristically_and_settles_on_no_client() {
        let trajectory = record(
            TrajectoryOutcome::Success,
            vec![
                turn(0, "read", vec![call("Read", true)]),
                turn(1, "fail", vec![call("Bash", false)]),
                turn(2, "recover", vec![call("Bash", true)]),
            ],
        );
        let mut config = ExtractionConfig::default();
        config.strategy = ExtractionStrategy::Hybrid;
        let mined = extract_from_record(&trajectory, &config, None)
            .await
            .expect("hybrid stays heuristic without a client");
        assert!(
            !mined.is_empty(),
            "the heuristic half still mines without a provider"
        );
    }

    #[tokio::test]
    async fn extraction_observer_never_panics_on_a_missing_ledger() {
        let store: std::sync::Arc<dyn LoopMemory> = std::sync::Arc::new(InMemoryStore::new());
        let observer = ExtractionObserver::new("/nonexistent/trajectory.jsonl", store);
        observer.on_run_end(&RunEndContext {
            success: true,
            error: None,
            total_turns: 3,
            duration_ms: 100,
        });
        tokio::task::yield_now().await;
    }

    #[test]
    fn summary_labels_turns_and_responses_newest_first_within_budget() {
        let turns = vec![
            turn(0, "first query alpha", vec![call("Read", true)]),
            turn(1, "second query beta", vec![call("Edit", true)]),
            turn(2, "third query gamma", vec![call("Bash", true)]),
        ];
        let mut record = record(TrajectoryOutcome::Success, turns);
        record.turns[0].response_text = "first response".to_string();
        record.turns[2].response_text = "third response".to_string();
        let summary = summarize_trajectory(&record, 8_000);
        assert!(
            summary.starts_with("outcome: success\n"),
            "the header renders the outcome in the stable snake_case vocabulary, not a \
            Debug variant name a model might echo back"
        );
        assert!(
            summary.contains("turn: 0")
                && summary.contains("turn: 1")
                && summary.contains("turn: 2"),
            "every turn line carries its index"
        );
        let third = summary.find("turn: 2").expect("turn 2 present");
        let first = summary.find("turn: 0").expect("turn 0 present");
        assert!(
            third < first,
            "newest turns render first so truncation drops the run's beginning"
        );
        let line_end = third + summary[third..].find('\n').unwrap_or(0);
        let line = &summary[third..line_end];
        let response_at = line.find("response:").expect("response field labeled");
        let value_at = line.find("third response").expect("response text present");
        assert!(
            value_at > response_at,
            "the response text lands after the response label, not after turn:"
        );
        let tiny = summarize_trajectory(&record, 120);
        assert!(
            tiny.len() <= 120 + "outcome: Success\n".len() + "total turns: 3\n".len(),
            "apart from the fixed header, the marker is reserved and the cap holds: \
            {} bytes",
            tiny.len()
        );
        assert!(tiny.ends_with("…\n"), "a truncated summary is marked");
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn hybrid_candidates_share_the_context_budget() {
        let turns: Vec<TrajectoryTurn> = (0..40)
            .map(|index| {
                turn(
                    index,
                    &format!("query number {index} with padding to fill the summary"),
                    vec![call("Read", true)],
                )
            })
            .collect();
        let trajectory = record(TrajectoryOutcome::Success, turns);
        let candidates: Vec<ExtractedMemory> = (0..200)
            .map(|index| ExtractedMemory {
                category: MemoryCategory::Strategy,
                content: format!("lesson number {index} with several words to fill the budget"),
                tags: Vec::new(),
                quality: 0.6,
            })
            .collect();
        let mut config = ExtractionConfig::default();
        config.strategy = ExtractionStrategy::Llm;
        config.llm_context_budget = 900;
        let client = MockApiClient::new("test-model").with_text_response("[]");
        llm_memories(&trajectory, &config, &client, Some(&candidates))
            .await
            .expect("an empty array is a valid answer");
        let request = &client.captured_requests()[0];
        let total = request.system.as_ref().map_or(0, String::len)
            + request
                .messages
                .iter()
                .map(|message| message.text_content().len())
                .sum::<usize>();
        assert!(
            total <= config.llm_context_budget,
            "system instruction, candidates, and summary together stay within the \
            configured budget (got {total} bytes)"
        );
        let user_text = request.messages[0].text_content();
        assert!(
            user_text.contains("turn: 39") && !user_text.contains("turn: 0 "),
            "truncation keeps the newest turns and drops the run's beginning"
        );
    }
}
