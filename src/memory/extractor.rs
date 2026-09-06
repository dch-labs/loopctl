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
use crate::memory::trajectory::{TrajectoryOutcome, TrajectoryRecord};
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

    /// Confidence the memory is worth keeping (`0.0..=1.0`); becomes the
    /// initial `MemoryEntry::relevance`.
    pub quality: f32,
}

/// How memories are mined from a trajectory.
///
/// The strategy is a configuration choice, not a compile-time split — all
/// three ship unconditionally and switching costs nothing but a struct
/// field.
#[derive(Debug, Clone, Default)]
pub enum ExtractionStrategy {
    /// Pattern-match the trajectory for known shapes (error→recovery
    /// adjacency, successful tool chains, repeated calls) and synthesize
    /// memories from rule-based templates. Zero LLM cost, deterministic,
    /// offline.
    #[default]
    Heuristic,

    /// Ask an [`ApiClient`] to read a summarized trajectory and emit a
    /// JSON array of memories. Higher quality; costs tokens and needs a
    /// provider.
    Llm,

    /// Heuristic first (cheap, always runs), then an optional LLM pass to
    /// refine, merge, and generalize the candidates. Degrades to the
    /// heuristic result when the provider call fails.
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

    /// Don't extract more than this many memories per trajectory — caps LLM
    /// cost and keeps one run from flooding the store.
    pub max_memories: usize,

    /// Skip extraction entirely for trajectories with fewer turns — a
    /// one-shot has nothing to generalize from.
    pub min_turns: usize,

    /// Mine only from successful runs by default; failures often contain
    /// the most instructive recovery pairs, so opt in to learn from them.
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
/// The file must contain one serialized [`TrajectoryRecord`] — a single
/// JSONL line from a trajectory ledger qualifies. The mined items are
/// returned without being written; use [`extract_into`] for the
/// store-writing convenience.
///
/// # Errors
///
/// - [`LoopError::Memory`] when the file cannot be read
///   or parsed.
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
    let path = trajectory_path.to_path_buf();
    let record: TrajectoryRecord = tokio::task::spawn_blocking(move || {
        let text = std::fs::read_to_string(&path)
            .map_err(|err| LoopError::Memory(format!("cannot read trajectory: {err}")))?;
        serde_json::from_str(text.trim())
            .map_err(|err| LoopError::Memory(format!("cannot parse trajectory: {err}")))
    })
    .await
    .map_err(|err| LoopError::Memory(format!("trajectory reader task failed: {err}")))??;
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
pub async fn extract_from_record(
    record: &TrajectoryRecord,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
) -> Result<Vec<ExtractedMemory>, LoopError> {
    let span = tracing::debug_span!("memory.extract");
    if record.turns.len() < config.min_turns {
        return Ok(Vec::new());
    }
    let successful = matches!(record.outcome, TrajectoryOutcome::Success);
    if !successful && !config.include_failures {
        return Ok(Vec::new());
    }
    let mut memories = match config.strategy {
        ExtractionStrategy::Heuristic => heuristic_memories(record),
        ExtractionStrategy::Llm => {
            let Some(client) = client else {
                return Err(LoopError::Api(
                    "the Llm extraction strategy requires an ApiClient".into(),
                ));
            };
            match llm_memories(record, config, client, None).await {
                Ok(mined) => mined,
                Err(err) => {
                    emit_attempt_metric(err_outcome(&err));
                    return Err(err);
                }
            }
        }
        ExtractionStrategy::Hybrid => {
            let candidates = heuristic_memories(record);
            match client {
                None => candidates,
                Some(client) => {
                    match llm_memories(record, config, client, Some(&candidates)).await {
                        Ok(refined) => refined,
                        Err(err) => {
                            emit_attempt_metric(err_outcome(&err));
                            tracing::warn!(
                                target: "loopctl::memory",
                                error = %err,
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
    {
        let _entered = span.enter();
        emit_extraction_metrics(&memories, "ok");
    }
    Ok(memories)
}

/// Drop candidates whose `(category, content)` pair already appears.
///
/// Mining is pattern-shaped, so one run can observe the same lesson many
/// times (twelve failed-then-recovered calls to one tool are one lesson,
/// not twelve); deduplicating before the `max_memories` truncation keeps
/// the cap from being spent on repeats that crowd out other candidates.
fn dedupe_candidates(memories: &mut Vec<ExtractedMemory>) {
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    memories
        .retain(|memory| seen.insert((format!("{:?}", memory.category), memory.content.clone())));
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
/// Propagates [`extract`]'s errors plus any store failure as
/// [`LoopError::Memory`].
pub async fn extract_into(
    trajectory_path: &Path,
    config: &ExtractionConfig,
    client: Option<&dyn ApiClient>,
    store: &dyn LoopMemory,
) -> Result<(usize, Vec<ExtractedMemory>), LoopError> {
    let candidates = extract(trajectory_path, config, client).await?;
    let mut written = 0usize;
    for candidate in &candidates {
        store.store(entry_for(candidate)).await?;
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
/// best-effort, may then mine the previous run's record (consolidation
/// folds any duplicate lesson away), and simply warns in that case.
/// Register this observer after the trajectory observer to minimize the
/// window. The same best-effort contract covers runtime shutdown: an
/// extraction task still queued when the host drops the tokio runtime is
/// lost with a log — hosts that require every run's extraction should
/// call [`extract_into`] directly instead of relying on the observer.
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
    fn name(&self) -> &'static str {
        "memory-extractor"
    }

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
/// Propagates extraction failures and any store failure as
/// [`LoopError::Memory`]; the observer logs and drops
/// whatever this returns.
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

/// Read the last complete JSONL line of a trajectory ledger.
///
/// Walks backwards past a torn trailing line (the writer's queue not yet
/// flushed, or a crash mid-append) and returns the newest line that
/// parses. [`None`] means nothing in the ledger parses — an empty or
/// all-torn file — so a best-effort observer can simply skip.
///
/// The file read runs on the blocking thread pool, so a large ledger
/// never stalls a tokio worker.
///
/// # Errors
///
/// [`LoopError::Memory`] when the ledger cannot be read at all.
async fn last_complete_record(path: &Path) -> Result<Option<TrajectoryRecord>, LoopError> {
    let path = path.to_path_buf();
    let text = tokio::task::spawn_blocking(move || {
        std::fs::read_to_string(&path)
            .map_err(|err| LoopError::Memory(format!("cannot read trajectory ledger: {err}")))
    })
    .await
    .map_err(|err| LoopError::Memory(format!("trajectory reader task failed: {err}")))??;
    let parsed = text
        .lines()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| serde_json::from_str(line).ok());
    Ok(parsed)
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
/// A failed call whose tool succeeds in a later turn means the model found
/// the fix; the memory pairs the failing tool with the retry advice so the
/// next run starts from the fix instead of the failure.
fn mine_recoveries(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();
    for (turn_index, turn) in record.turns.iter().enumerate() {
        for call in &turn.tool_calls {
            if call.ok {
                continue;
            }
            let recovered = record
                .turns
                .iter()
                .skip(turn_index.saturating_add(1))
                .any(|later| {
                    later
                        .tool_calls
                        .iter()
                        .any(|later_call| later_call.ok && later_call.tool == call.tool)
                });
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

/// Mine successful tool chains as `Strategy` memories.
///
/// A run that succeeded and issued three or more successful calls carries
/// a repeatable ordering; the memory abstracts the observed sequence into
/// follow-this-order advice.
fn mine_strategies(record: &TrajectoryRecord) -> Vec<ExtractedMemory> {
    let successful = matches!(record.outcome, TrajectoryOutcome::Success);
    if !successful {
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
    let names = chain
        .iter()
        .map(|tool| (*tool).to_string())
        .collect::<Vec<_>>()
        .join(" → ");
    vec![ExtractedMemory {
        category: MemoryCategory::Strategy,
        content: format!(
            "a chain of {count} successful tool calls ({names}) completed this task — \
             for similar tasks, follow this order",
            count = chain.len(),
        ),
        tags: vec!["chain".into()],
        quality: (0.5 + 0.05 * f32::from(u16::try_from(chain.len()).unwrap_or(u16::MAX))).min(0.9),
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
    /// Category name as the prompt defines them: `strategy`,
    /// `error_pattern`, `insight`, or `optimization`.
    category: String,

    /// The learned lesson, phrased as reusable advice.
    ///
    /// Taken verbatim from the model's answer; extraction does not rewrite
    /// provider text, so prompt quality is what keeps these advice-shaped.
    content: String,

    /// Free-form tags; absent means none.
    ///
    /// Copied through when present. The `optimization` category also gains
    /// its selecting tag if the model omitted it, keeping the
    /// filter-by-kind contract consistent across strategies.
    tags: Option<Vec<String>>,

    /// The model's self-reported confidence; absent defaults to 0.5.
    ///
    /// Clamped into `0.0..=1.0` and used as the mined memory's relevance.
    /// The mid-scale default means un-self-assessed memories neither dominate
    /// the store nor vanish at the prune floor.
    quality: Option<f32>,
}

/// Render a compact, budget-capped transcript for the LLM strategies.
///
/// Turns are rendered newest-first — when a long run must be truncated,
/// the dropped part is the old beginning and the surviving summary ends
/// where the run ended, which is where the outcome lives. Apart from the
/// fixed outcome header — which always rides along, however small the
/// budget — the returned summary never exceeds `budget` bytes: the
/// truncation marker is reserved before any turn line is appended. Keeps
/// what generalizes — outcome, turn index, per-turn tool names with
/// ok/error flags, and trimmed query and response text.
fn summarize_trajectory(record: &TrajectoryRecord, budget: usize) -> String {
    let header = format!(
        "outcome: {:?}\ntotal turns: {}\n",
        record.outcome, record.total_turns
    );
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
        if used.saturating_add(line.len()) > budget.saturating_sub(2) {
            lines.push("…\n".to_string());
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

/// Cut `text` to at most `limit` bytes on a char boundary, marking the
/// cut with an ellipsis so summaries stay honest about what they dropped.
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
) -> Result<Vec<ExtractedMemory>, LoopError> {
    let mut system = String::from(
        "You are a memory extractor. Read the agent trajectory and emit JSON: an array of \
         {\"category\",\"content\",\"tags\",\"quality\"} objects. category is one of \
         \"strategy\", \"error_pattern\", \"insight\", or \"optimization\". Only emit \
         genuinely reusable lessons, phrased as advice.",
    );
    if let Some(candidates) = heuristic_candidates {
        system.push_str(
            "\nCandidate lessons already mined from this trajectory follow; \
                         refine, merge, generalize, and drop the weak ones:\n",
        );
        let mut candidate_budget = config
            .llm_context_budget
            .saturating_sub(summarize_trajectory(record, config.llm_context_budget).len())
            .min(config.llm_context_budget);
        for candidate in candidates {
            let line = format!("- [{:?}] {}\n", candidate.category, candidate.content);
            if candidate_budget.saturating_sub(line.len()) == 0 {
                break;
            }
            candidate_budget = candidate_budget.saturating_sub(line.len());
            system.push_str(&line);
        }
    }
    let request = StreamRequest {
        messages: vec![Message::user(summarize_trajectory(
            record,
            config.llm_context_budget,
        ))],
        system: Some(system),
        tools: None,
    };
    let response = client
        .create_message(&request)
        .await
        .map_err(|err| LoopError::Api(format!("extraction provider call failed: {err}")))?;
    parse_llm_memories(&response.message.text_content())
}

/// Parse the provider's answer: the outermost JSON array in the text,
/// mapped to [`ExtractedMemory`] values with unknown categories dropped.
///
/// # Errors
///
/// [`LoopError::Api`] when no array is present or the
/// slice does not deserialize.
fn parse_llm_memories(text: &str) -> Result<Vec<ExtractedMemory>, LoopError> {
    let Some(start) = text.find('[') else {
        return Err(unparseable_response());
    };
    let Some(end) = text.rfind(']') else {
        return Err(unparseable_response());
    };
    let Some(slice) = text.get(start..=end) else {
        return Err(unparseable_response());
    };
    let wire: Vec<LlmMemoryWire> = serde_json::from_str(slice).map_err(|err| {
        LoopError::Api(format!(
            "extraction response was not parseable: {err} [parse]"
        ))
    })?;
    Ok(wire.into_iter().filter_map(wire_to_extracted).collect())
}

/// The error for a provider answer that is transport-fine but not a
/// parseable memory array — labelled `[parse]` so telemetry can
/// distinguish "the model cannot do this job" from "the transport
/// failed".
fn unparseable_response() -> LoopError {
    LoopError::Api("extraction response contained no parseable JSON array [parse]".into())
}

/// Map one provider-reported memory onto [`ExtractedMemory`].
///
/// Unknown category names are dropped (`None`) rather than guessed; the
/// `optimization` name maps to `Insight` and gains the `optimization` tag
/// it is selected by.
fn wire_to_extracted(wire: LlmMemoryWire) -> Option<ExtractedMemory> {
    let category = match wire.category.to_lowercase().as_str() {
        "strategy" => MemoryCategory::Strategy,
        "error_pattern" | "recovery" => MemoryCategory::ErrorPattern,
        "insight" | "optimization" => MemoryCategory::Insight,
        _ => return None,
    };
    let mut tags = wire.tags.unwrap_or_default();
    if wire.category.to_lowercase() == "optimization" && !tags.iter().any(|t| t == "optimization") {
        tags.push("optimization".into());
    }
    Some(ExtractedMemory {
        category,
        content: wire.content,
        tags,
        quality: wire.quality.unwrap_or(0.5).clamp(0.0, 1.0),
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
/// pass, labelled by how it settled.
fn emit_attempt_metric(outcome: &'static str) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.memory.extract.attempts",
        outcome
    );
}

/// Map an extraction error to its telemetry outcome label — a parse
/// failure answers "is the model good enough at this" differently than a
/// transport failure. Parse failures carry the `[parse]` marker set by
/// [`unparseable_response`]; everything else is a transport-level
/// `api_error`.
fn err_outcome(err: &LoopError) -> &'static str {
    match err {
        LoopError::Api(message) if message.contains("[parse]") => "parse_error",
        _ => "api_error",
    }
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
    #[cfg(feature = "testing")]
    use crate::testing::MockApiClient;

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
        assert_eq!(
            err_outcome(&err),
            "parse_error",
            "a response that transports fine but carries no array is a parse \
            failure, not a transport failure"
        );
        let transport = MockApiClient::new("test-model").with_errors(vec![Some("boom".into())]);
        let err = extract_from_record(&trajectory, &config, Some(&transport))
            .await
            .expect_err("a provider transport failure is an error");
        assert_eq!(
            err_outcome(&err),
            "api_error",
            "a transport failure keeps the api_error label"
        );
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
            total <= config.llm_context_budget + 400,
            "summary plus candidates stay bounded by the configured budget \
            (got {total} bytes)"
        );
        let user_text = request.messages[0].text_content();
        assert!(
            user_text.contains("turn: 39") && !user_text.contains("turn: 0 "),
            "truncation keeps the newest turns and drops the run's beginning"
        );
    }
}
