//! LLM-driven structured compaction: a sectioned summary rendered from a
//! fixed template.
//!
//! [`StructuredSummarizer`] implements [`ContextCompactor`] against a
//! caller-supplied [`ApiClient`](crate::api::ApiClient), trading a little of
//! the free-form [`QaSummarizer`](crate::compact::qa_summarizer::QaSummarizer)'s
//! retention for a contractually stable shape: the summary always carries the
//! same headings — key facts, decisions, pending tasks, open questions — each
//! as a bullet list, so a consumer (telemetry, an attachment layer, a UI) can
//! locate "pending tasks" without parsing prose. One LLM call per pass (plus
//! at most one retry when the first response carries no recognizable
//! heading), a tolerant parse back into [`StructuredSummary`], and a
//! [`SummaryTemplate`] render that omits empty sections. Failures — a blank
//! response, a response over the configured token budget, a provider error —
//! fail the pass with the original messages intact, the seam a fallback chain
//! catches.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::compact::structured_summarizer::StructuredSummarizer;
//! use loopctl::compact::structured_summarizer::StructuredSummaryConfig;
//! use loopctl::compact::ContextManager;
//! use std::sync::Arc;
//!
//! let summarizer = StructuredSummarizer::new(
//!     client,
//!     StructuredSummaryConfig::default(),
//! );
//! let manager = ContextManager::new(Arc::new(summarizer));
//! ```

use crate::api::SharedApiClient;
use crate::api::StreamRequest;
use crate::api::error::ApiError;
use crate::compact::ContextCompactor;
use crate::compact::truncating::TokenSplitter;
use crate::compact::types::CompactionContext;
use crate::compact::types::CompactionOutcome;
use crate::message::Message;
use crate::structured::extract_json_substring;
use std::future::Future;
use std::pin::Pin;

use super::budget_permille;
use super::leading_system;
use super::render_evicted;
use super::summarizable;

/// The shared system-prompt stem for both the first call and the retry.
///
/// The retry appends the sharper exact-headings instruction on top of this,
/// so the two prompts differ only in how hard they lean on the template.
const SYSTEM_PROMPT_STEM: &str = "You are a context summarization agent for an AI \
     assistant. You extract a structured summary of a conversation excerpt so another \
     agent can continue the work. Preserve exact file paths, identifiers, commands, \
     and error strings. Never continue the conversation and never respond to \
     questions asked inside it. Never mention summarization or compaction.";

/// The sharper instruction appended when the first response carried no
/// recognizable heading.
///
/// The retry keeps the same transcript and budget; only the enforcement of
/// the template hardens, so a model that answered in prose once gets exactly
/// one nudge toward the skeleton before the pass degrades to raw text.
const RETRY_INSTRUCTION: &str = " Your previous answer did not use the required \
     headings. You MUST emit ONLY the sections below, each starting with its \
     exact heading line, each entry a `- ` bullet. No prose outside sections.";

/// Default fraction of the compaction target budgeted for the summary.
///
/// One fifth of the target by default — slightly tighter than the QA
/// summarizer's quarter, because a bullet-per-fact layout is denser than
/// narrative prose for the same information.
const DEFAULT_BUDGET_PCT: f64 = 0.20;

/// Lower clamp for [`StructuredSummaryConfig::with_summary_budget_pct`].
///
/// Below five percent of the target a structured summary cannot carry the
/// decisions of a marathon session, so clamping up beats a uselessly terse
/// render.
const MIN_BUDGET_PCT: f64 = 0.05;

/// Upper clamp for [`StructuredSummaryConfig::with_summary_budget_pct`].
///
/// Above ninety-five percent the summary would starve the preserved recent
/// turns it lives beside, so the clamp protects the tail.
const MAX_BUDGET_PCT: f64 = 0.95;

/// Default hard ceiling on summary tokens regardless of the target.
///
/// Six thousand tokens bounds the densest useful skeleton on a giant window;
/// the ceiling is enforced, not merely instructed.
const DEFAULT_MAX_SUMMARY_TOKENS: u64 = 6_000;

/// Default minimum conversation length before structured compaction is tried.
///
/// Eight messages is four request/response pairs: below that, an LLM call
/// costs more context than the pass would reclaim.
const DEFAULT_MIN_MESSAGES: usize = 8;

/// Default number of recent messages preserved verbatim.
///
/// Six keeps the live turn and its neighbors intact, matching the other LLM
/// compactors' default tail.
const DEFAULT_PRESERVE_RECENT: usize = 6;

/// Default character budget for the dropped-context transcript in a prompt.
///
/// Bounds the summarizer's *input* independently of the output budget — the
/// small-model knob the shared renderer honors.
const DEFAULT_TRANSCRIPT_MAX_CHARS: usize = 24_000;

/// The header line a default template renders above the sections.
///
/// Marks the block as a summary of compacted context without narrating the
/// compaction process to the continuing model.
const DEFAULT_HEADER: &str = "## Conversation summary (compacted)";

/// The line a fully-empty summary renders as.
///
/// A dropped context with nothing salient is a valid outcome; the stub keeps
/// a visible marker where the context was instead of an empty message.
const EMPTY_SUMMARY_STUB: &str = "No salient content in the compacted portion.";

/// How many items one section may hold after parsing.
///
/// A runaway model cannot balloon one section past this cap, keeping the
/// rendered summary within a scannable size even before the token budget
/// check has its say.
const MAX_ITEMS_PER_SECTION: usize = 50;

/// One canonical section of a structured summary.
///
/// `#[non_exhaustive]` so future sections (a files list, an errors lane) can
/// arrive in minor releases — downstream matches need a `_` wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SummarySection {
    /// Hard facts established by the dropped context.
    ///
    /// Paths, identifiers, configuration values — the things a future turn
    /// would re-derive expensively if lost.
    KeyFacts,

    /// Choices the user or agent committed to.
    ///
    /// Decisions are authoritative; losing them makes the model re-litigate
    /// settled questions.
    Decisions,

    /// Work in flight or explicitly outstanding.
    ///
    /// These become the model's implicit todo list on re-entry.
    PendingTasks,

    /// Unknowns the dropped context raised but did not answer.
    ///
    /// Distinct from facts (known) and tasks (actions): these are the
    /// questions the model should know it does not know.
    OpenQuestions,
}

impl SummarySection {
    /// Every canonical section, in default-template order.
    ///
    /// The order is the render order of
    /// [`default_template`](SummaryTemplate::default_template): facts before
    /// the decisions built on them, tasks before the questions that may
    /// reshape both.
    pub const ALL: [Self; 4] = [
        Self::KeyFacts,
        Self::Decisions,
        Self::PendingTasks,
        Self::OpenQuestions,
    ];

    /// The section's stable heading title.
    ///
    /// The exact string the prompt requires, the renderer emits after
    /// `### `, and the parser matches case-insensitively — the contract
    /// string of the whole module.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::KeyFacts => "Key facts",
            Self::Decisions => "Decisions",
            Self::PendingTasks => "Pending tasks",
            Self::OpenQuestions => "Open questions",
        }
    }

    /// The section's telemetry slug.
    ///
    /// The compact lowercase label the sections-counter events carry
    /// (`facts`, `decisions`, `todos`, `questions`).
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::KeyFacts => "facts",
            Self::Decisions => "decisions",
            Self::PendingTasks => "todos",
            Self::OpenQuestions => "questions",
        }
    }

    /// The section's JSON-object key.
    ///
    /// The `snake_case` name the tolerant parser accepts when a provider
    /// answers in JSON mode, so such a provider works without changes.
    #[must_use]
    pub const fn json_key(self) -> &'static str {
        match self {
            Self::KeyFacts => "key_facts",
            Self::Decisions => "decisions",
            Self::PendingTasks => "pending_tasks",
            Self::OpenQuestions => "open_questions",
        }
    }

    /// This section's items in a summary.
    ///
    /// The one accessor the renderer, the telemetry, and the template
    /// filter share, so a new section variant lands in exactly one match.
    #[must_use]
    pub fn items(self, summary: &StructuredSummary) -> &[String] {
        match self {
            Self::KeyFacts => &summary.key_facts,
            Self::Decisions => &summary.decisions,
            Self::PendingTasks => &summary.pending_tasks,
            Self::OpenQuestions => &summary.open_questions,
        }
    }
}

/// Which sections a structured summary requests and renders, and in what
/// order.
///
/// Both the prompt builder and the renderer walk `sections` in order, so a
/// template change reshapes the request and the output in one place. Build
/// from [`default_template`](Self::default_template) or
/// [`action_only`](Self::action_only) and mutate the public fields — the
/// type is `#[non_exhaustive]`, so struct literals compile only inside the
/// crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SummaryTemplate {
    /// The sections to request and render, in order.
    ///
    /// Duplicates are honored literally (a section listed twice renders
    /// twice); the presets never list one twice.
    pub sections: Vec<SummarySection>,

    /// The header line rendered above the first section.
    ///
    /// A markdown `##` heading by convention; the default is
    /// `"## Conversation summary (compacted)"`.
    pub header: String,
}

impl SummaryTemplate {
    /// The default four-section template in canonical order.
    ///
    /// Key facts, decisions, pending tasks, open questions — the task-oriented
    /// skeleton the module exists to guarantee.
    #[must_use]
    pub fn default_template() -> Self {
        Self {
            sections: SummarySection::ALL.to_vec(),
            header: DEFAULT_HEADER.to_string(),
        }
    }

    /// A terse two-section template: decisions and pending tasks.
    ///
    /// The action-oriented preset for short windows or runs that only need
    /// what is settled and what is left; key facts and open questions are
    /// neither requested nor rendered.
    #[must_use]
    pub fn action_only() -> Self {
        Self {
            sections: vec![SummarySection::Decisions, SummarySection::PendingTasks],
            header: DEFAULT_HEADER.to_string(),
        }
    }
}

/// A parsed structured summary: the four canonical section lists.
///
/// Produced by the tolerant parser and consumed by
/// [`render`]; empty sections are valid and are omitted from
/// the rendered text. Construct via [`new`](Self::new) (the type is
/// `#[non_exhaustive]`) and read or mutate the public fields freely.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StructuredSummary {
    /// Hard facts established by the dropped context.
    ///
    /// One entry per fact, phrased to stand alone; the renderer emits each
    /// as a `- ` bullet under `### Key facts`.
    pub key_facts: Vec<String>,

    /// Choices the user or agent committed to.
    ///
    /// Each entry states the decision and, where the conversation recorded
    /// it, the reason — losing the why is how re-litigation starts.
    pub decisions: Vec<String>,

    /// Work in flight or explicitly outstanding.
    ///
    /// The model's implicit todo list on re-entry; the attachment layer can
    /// lift this section wholesale.
    pub pending_tasks: Vec<String>,

    /// Unknowns the dropped context raised but did not answer.
    ///
    /// Awareness of not-knowing, distinct from facts and from actions.
    pub open_questions: Vec<String>,
}

impl StructuredSummary {
    /// A summary with all four sections populated from the given lists.
    ///
    /// The constructor for external builders and tests; in-crate code may
    /// also literal-construct. Entries render in list order.
    #[must_use]
    pub fn new(
        key_facts: Vec<String>,
        decisions: Vec<String>,
        pending_tasks: Vec<String>,
        open_questions: Vec<String>,
    ) -> Self {
        Self {
            key_facts,
            decisions,
            pending_tasks,
            open_questions,
        }
    }

    /// Whether every section is empty.
    ///
    /// A fully-empty summary is a valid parse of a trivial dropped context;
    /// the renderer answers it with the stub line rather than nothing, so
    /// the feed still marks where the context was.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.key_facts.is_empty()
            && self.decisions.is_empty()
            && self.pending_tasks.is_empty()
            && self.open_questions.is_empty()
    }

    /// How many of the given sections hold at least one entry.
    ///
    /// The populated count the pass-complete telemetry reports; sections not
    /// listed are not counted.
    #[must_use]
    pub fn populated_count(&self, sections: &[SummarySection]) -> usize {
        sections
            .iter()
            .filter(|section| !section.items(self).is_empty())
            .count()
    }
}

/// Render a structured summary to text via a template.
///
/// The free-function counterpart of the compactor: emits the template's
/// header, then each non-empty section in template order as `### <Title>`
/// followed by one `- ` bullet per entry. Empty sections are omitted rather
/// than rendered as bare headings; a fully-empty summary renders the header
/// and the stub line. A template whose `sections` list is empty cannot carry
/// content, so a content-bearing summary falls back to the canonical
/// [`SummarySection::ALL`] order rather than rendering a bare header over
/// dropped context. The heading strings are the module's stable contract —
/// the same strings the prompt requires and the parser matches.
///
/// # Example
///
/// ```
/// use loopctl::compact::structured_summarizer::{
///     StructuredSummary, SummaryTemplate, render,
/// };
///
/// let summary = StructuredSummary::new(
///     vec!["src/api.rs uses reqwest".to_string()],
///     Vec::new(),
///     Vec::new(),
///     Vec::new(),
/// );
/// let text = render(&summary, &SummaryTemplate::default_template());
/// assert!(text.contains("### Key facts"));
/// assert!(text.contains("- src/api.rs uses reqwest"));
/// assert!(!text.contains("### Decisions"), "empty sections are omitted");
/// ```
#[must_use]
pub fn render(summary: &StructuredSummary, template: &SummaryTemplate) -> String {
    if summary.is_empty() {
        return format!("{}\n\n{EMPTY_SUMMARY_STUB}", template.header);
    }
    let mut out = template.header.clone();
    for section in rendered_sections(template) {
        let items = section.items(summary);
        if items.is_empty() {
            continue;
        }
        out.push_str("\n\n### ");
        out.push_str(section.title());
        for item in items {
            out.push_str("\n- ");
            out.push_str(item);
        }
    }
    out
}

/// The sections a render walks, never empty for a content-bearing summary.
///
/// The render-side half of the empty-template guard: a hand-mutated template
/// with no sections would otherwise collapse a populated summary to a bare
/// header, silently discarding the content the pass just paid for.
fn rendered_sections(template: &SummaryTemplate) -> &[SummarySection] {
    if template.sections.is_empty() {
        return &SummarySection::ALL;
    }
    &template.sections
}

/// Configuration for [`StructuredSummarizer`].
///
/// Built from [`Default`](Self::default) and adjusted with the `with_*`
/// builders; every numeric setter clamps or rejects degenerate values the
/// way the other compactor configs do, so a stored config is always usable.
///
/// # Example
///
/// ```
/// use loopctl::compact::structured_summarizer::StructuredSummaryConfig;
///
/// let config = StructuredSummaryConfig::default()
///     .with_summary_budget_pct(0.25)
///     .with_max_summary_tokens(4_000)
///     .with_preserve_recent(8);
/// assert_eq!(config.max_summary_tokens(), 4_000);
/// assert_eq!(config.preserve_recent(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct StructuredSummaryConfig {
    /// Fraction of the compaction target budgeted for the summary.
    ///
    /// The stored float is the getter's answer; the pass budget itself is
    /// computed from the exact [`Self::budget_permille`] twin so the
    /// arithmetic never rides a float cast.
    budget_pct: f64,

    /// The budget fraction as an exact per-mille integer.
    ///
    /// Derived from `budget_pct`'s decimal rendering through the shared
    /// rational conversion, so the budget arithmetic stays in exact integers
    /// and never runs through a float cast.
    budget_permille: u64,

    /// Hard ceiling on summary tokens regardless of the target.
    ///
    /// A giant window's 0.20 share could buy a whole essay; the ceiling keeps
    /// the skeleton scannable and leaves the window for the preserved tail.
    max_summary_tokens: u64,

    /// Minimum conversation length before structured compaction is tried.
    ///
    /// Below this length a pass would spend an LLM call to reclaim less
    /// context than the call costs, so `compact` answers no-change instead.
    min_messages: usize,

    /// Number of recent messages preserved verbatim.
    ///
    /// Forwarded to the internal [`TokenSplitter`], so the boundary is the
    /// same pair-safe one the other LLM compactors draw.
    preserve_recent: usize,

    /// Character budget for the dropped-context transcript in a prompt.
    ///
    /// Bounds the summarizer's *input* independently of the output budget —
    /// the small-model knob the shared renderer honors.
    transcript_max_chars: usize,

    /// Which sections to request and render.
    ///
    /// The prompt enumerates and the renderer emits exactly these sections
    /// in this order, so one field reshapes both sides of the contract.
    template: SummaryTemplate,

    /// Whether the model's output is parsed back into a
    /// [`StructuredSummary`].
    parse_output: bool,
}

impl StructuredSummaryConfig {
    /// The validated configuration with every field at its default.
    ///
    /// The single construction site for defaults; [`Default`] delegates
    /// here and every builder starts from a `fresh` value.
    fn fresh() -> Self {
        Self {
            budget_pct: DEFAULT_BUDGET_PCT,
            budget_permille: budget_permille(DEFAULT_BUDGET_PCT),
            max_summary_tokens: DEFAULT_MAX_SUMMARY_TOKENS,
            min_messages: DEFAULT_MIN_MESSAGES,
            preserve_recent: DEFAULT_PRESERVE_RECENT,
            transcript_max_chars: DEFAULT_TRANSCRIPT_MAX_CHARS,
            template: SummaryTemplate::default_template(),
            parse_output: true,
        }
    }

    /// Clamp a budget fraction and derive its exact per-mille form.
    ///
    /// A non-finite or non-positive fraction is rejected with a warning and
    /// this module's default takes its place — a stored NaN would poison
    /// the budget — then the value is clamped into `[0.05, 0.95]` so the
    /// summary can neither vanish nor starve the preserved tail.
    fn validated_pct(pct: f64) -> (f64, u64) {
        let usable = pct.is_finite() && pct > 0.0;
        if !usable {
            tracing::warn!(
                target: "loopctl::compact",
                pct,
                "summary budget fraction not finite or not positive; using the 0.20 default"
            );
        }
        let clamped = if usable {
            pct.clamp(MIN_BUDGET_PCT, MAX_BUDGET_PCT)
        } else {
            DEFAULT_BUDGET_PCT
        };
        (clamped, budget_permille(clamped))
    }

    /// Set the fraction of the compaction target budgeted for the summary.
    ///
    /// Clamped to `[0.05, 0.95]`; NaN and infinities are rejected with a
    /// warning and fall back to the default `0.20`. The ceiling this
    /// fraction feeds is enforced on the model's response, not merely
    /// instructed in the prompt.
    #[must_use]
    pub fn with_summary_budget_pct(mut self, pct: f64) -> Self {
        (self.budget_pct, self.budget_permille) = Self::validated_pct(pct);
        self
    }

    /// Set the hard ceiling on summary tokens.
    ///
    /// The smaller of this ceiling and the fraction's share of the target
    /// is the pass budget; a response whose estimate exceeds it fails the
    /// pass. Values below `1` clamp to `1`.
    #[must_use]
    pub fn with_max_summary_tokens(mut self, tokens: u64) -> Self {
        self.max_summary_tokens = tokens.max(1);
        self
    }

    /// Set the minimum conversation length before compaction is tried.
    ///
    /// Below this length `compact` returns a no-change outcome without
    /// spending any LLM calls; values below `2` clamp to `2`, matching the
    /// splitter's own floor.
    #[must_use]
    pub fn with_min_messages(mut self, count: usize) -> Self {
        self.min_messages = count.max(2);
        self
    }

    /// Set the number of recent messages preserved verbatim.
    ///
    /// Passed straight to the internal [`TokenSplitter`], so the boundary is
    /// the same pair-safe one the other compactors draw; values below `1`
    /// clamp to `1`.
    #[must_use]
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count.max(1);
        self
    }

    /// Set the character budget for the dropped-context transcript.
    ///
    /// Bounds the summarizer's *input* prompt independently of the output
    /// budget, which matters on small local models. Values below `1` clamp
    /// to `1`.
    #[must_use]
    pub fn with_transcript_max_chars(mut self, max_chars: usize) -> Self {
        self.transcript_max_chars = max_chars.max(1);
        self
    }

    /// Set the template carrying which sections are requested and rendered.
    ///
    /// The prompt enumerates and the renderer emits exactly the template's
    /// sections in its order; the presets are
    /// [`default_template`](SummaryTemplate::default_template) and
    /// [`action_only`](SummaryTemplate::action_only). A template carrying no
    /// sections is rejected with a warning and the default template takes
    /// its place — an empty skeleton cannot carry content, and a pass that
    /// demoted its dropped slice with nothing replacing it is the outcome
    /// the module's blank-response guard exists to prevent.
    #[must_use]
    pub fn with_template(mut self, template: SummaryTemplate) -> Self {
        if template.sections.is_empty() {
            tracing::warn!(
                target: "loopctl::compact",
                "summary template carries no sections; using the default template"
            );
            self.template = SummaryTemplate::default_template();
            return self;
        }
        self.template = template;
        self
    }

    /// Set whether the model's output is parsed back into sections.
    ///
    /// `true` (the default) parses tolerantly and falls back to the raw
    /// text on failure; `false` skips parsing and carries the raw text as
    /// the summary body unchanged.
    #[must_use]
    pub fn with_parse_output(mut self, parse_output: bool) -> Self {
        self.parse_output = parse_output;
        self
    }

    /// The fraction of the compaction target budgeted for the summary.
    ///
    /// The stored value after clamping — `0.20` unless overridden.
    #[must_use]
    pub fn summary_budget_pct(&self) -> f64 {
        self.budget_pct
    }

    /// The hard ceiling on summary tokens.
    ///
    /// The stored value after the `max(1)` clamp — a zero ceiling would
    /// fail every response at its own budget.
    #[must_use]
    pub fn max_summary_tokens(&self) -> u64 {
        self.max_summary_tokens
    }

    /// The minimum conversation length before compaction is tried.
    ///
    /// The stored value after the `max(2)` clamp, matching the splitter's
    /// own floor.
    #[must_use]
    pub fn min_messages(&self) -> usize {
        self.min_messages
    }

    /// The number of recent messages preserved verbatim.
    ///
    /// The stored value after the `max(1)` clamp; zero would preserve
    /// nothing and summarize the live turn out from under the model.
    #[must_use]
    pub fn preserve_recent(&self) -> usize {
        self.preserve_recent
    }

    /// The character budget for the dropped-context transcript.
    ///
    /// The stored value after the `max(1)` clamp, applied to every prompt
    /// the transcript is rendered for.
    #[must_use]
    pub fn transcript_max_chars(&self) -> usize {
        self.transcript_max_chars
    }

    /// The template carrying which sections are requested and rendered.
    ///
    /// A clone of the stored value, ready to mutate; the compactor itself
    /// is unaffected by the caller's copy.
    #[must_use]
    pub fn template(&self) -> SummaryTemplate {
        self.template.clone()
    }

    /// Whether the model's output is parsed back into sections.
    ///
    /// Read this before deciding whether a pass's summary message is
    /// section-shaped or raw text.
    #[must_use]
    pub fn parse_output(&self) -> bool {
        self.parse_output
    }

    /// The summary token budget for a compaction pass at `target_tokens`.
    ///
    /// Reserve-then-cap: the smaller of the hard ceiling and the fraction's
    /// exact share of the target. The ceiling is enforced — a response
    /// whose heuristic estimate exceeds this figure fails the pass.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::compact::structured_summarizer::StructuredSummaryConfig;
    ///
    /// let config = StructuredSummaryConfig::default();
    /// assert_eq!(config.summary_budget(20_000), 4_000);
    /// assert_eq!(config.summary_budget(40_000), 6_000);
    /// ```
    #[must_use]
    pub fn summary_budget(&self, target_tokens: u64) -> u64 {
        let share = target_tokens.saturating_mul(self.budget_permille) / 1_000;
        share.min(self.max_summary_tokens)
    }
}

impl Default for StructuredSummaryConfig {
    fn default() -> Self {
        Self::fresh()
    }
}

/// LLM-driven structured compactor.
///
/// Construct with any [`ApiClient`](crate::api::ApiClient) and a
/// [`StructuredSummaryConfig`], then install on a
/// [`ContextManager`](crate::compact::ContextManager) as
/// `Arc<dyn ContextCompactor>`. The client may be the loop's own or a
/// dedicated cheaper one — the recommended production shape, documented in
/// the QA summarizer's docs — because the single call rides plain
/// [`create_message`](crate::api::ApiClient::create_message) with no request
/// options.
///
/// Each pass makes one LLM call, retried at most once when the first
/// response carries no recognizable section heading; a second malformed
/// response degrades to its raw text as the summary body rather than
/// failing. Blank responses and responses over the configured budget fail
/// the pass with the original messages intact — the seam a fallback chain
/// catches. The compactor is stateless: no summary accumulates across
/// passes, each pass summarizes the current transcript through the
/// template, and there is no lock to hold or poison.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::compact::structured_summarizer::{
///     StructuredSummarizer, StructuredSummaryConfig,
/// };
/// use loopctl::compact::ContextManager;
/// use std::sync::Arc;
///
/// let summarizer = Arc::new(StructuredSummarizer::new(
///     client,
///     StructuredSummaryConfig::default(),
/// ));
/// let manager = ContextManager::new(summarizer);
/// ```
pub struct StructuredSummarizer {
    /// The client the summarization call (and its retry) rides.
    ///
    /// Shared, not owned: the same `Arc` the loop holds, or a dedicated
    /// client pointed at a cheaper model.
    client: SharedApiClient,

    /// The clamped, always-usable configuration.
    ///
    /// Every setter normalizes on write, so a stored config can never hold
    /// a degenerate value the budget math would trip over.
    config: StructuredSummaryConfig,
}

/// The outcome of one summarization attempt.
///
/// Distinguishes a parsed (or degraded) success from the typed failures the
/// caller maps onto a failed outcome, and carries whether the retry path
/// ran so the telemetry can name it.
enum SummarizeOutcome {
    /// A parsed summary, or raw text degraded into `key_facts`.
    ///
    /// Both shapes are successes: the parse path fills the configured
    /// sections, while a second unparsable response rides whole as the
    /// single key-facts entry so no data is lost to the shape contract.
    Parsed {
        /// The summary to render.
        ///
        /// Section-populated when the parse recognized headings or JSON,
        /// raw otherwise; the renderer handles either.
        summary: StructuredSummary,

        /// Whether the sharper retry prompt produced it.
        ///
        /// The pass-complete telemetry reports it, so a climbing retry rate
        /// is visible before the template quietly degrades further.
        retried: bool,
    },

    /// The response carried no usable text at all.
    ///
    /// A thinking-only or content-filtered response would otherwise commit
    /// an empty summary over a demoted slice, so the pass fails instead.
    Blank,

    /// The response's token estimate exceeded the pass budget.
    ///
    /// The ceiling is enforced, not merely instructed; a model that ignores
    /// the prompt's budget line fails the pass instead of shipping through.
    OverBudget {
        /// The budget the estimate exceeded, for the error message.
        ///
        /// Rendered into the failure reason so the caller's log names the
        /// figure the response crossed.
        budget: u64,
    },

    /// The provider call itself failed.
    ///
    /// Carried as the typed [`ApiError`] so the failure reason names the
    /// provider's own diagnosis verbatim.
    Failed(ApiError),
}

impl StructuredSummarizer {
    /// Create a structured summarizer bound to the given API client.
    ///
    /// The client is kept as-is for the instance's whole life; hosts
    /// wanting a cheaper summarizer model build a dedicated client and pass
    /// it here.
    #[must_use]
    pub fn new(client: SharedApiClient, config: StructuredSummaryConfig) -> Self {
        Self { client, config }
    }

    /// The splitter whose boundary this compactor delegates to.
    ///
    /// Rebuilt from the config each pass — a plain value, so the
    /// construction cost is nothing next to the LLM call that follows.
    fn splitter(&self) -> TokenSplitter {
        TokenSplitter::new()
            .with_preserve_recent(self.config.preserve_recent)
            .with_min_messages(self.config.min_messages)
    }

    /// The system prompt enumerating the configured sections.
    ///
    /// Names each section's exact heading and one-line description, forbids
    /// prose outside the sections, and states the token budget — the
    /// enforcement instruction the budget check backs up. The retry variant
    /// appends the sharper exact-headings instruction.
    fn system_prompt(&self, budget: u64, retry: bool) -> String {
        use std::fmt::Write as _;
        let mut prompt = String::from(SYSTEM_PROMPT_STEM);
        let _ignored = write!(
            prompt,
            "\n\nEmit ONLY the sections below, each starting with its exact heading \
             line, each entry a `- ` bullet. Omit a section entirely if it has no \
             entries. No prose outside sections. Keep the total under roughly \
             {budget} tokens."
        );
        for section in &self.config.template.sections {
            let _ignored = write!(
                prompt,
                "\n### {}\n{}",
                section.title(),
                section_purpose(*section)
            );
        }
        if retry {
            prompt.push_str(RETRY_INSTRUCTION);
        }
        prompt
    }

    /// Run the summarization call (plus its retry) over the dropped slice.
    ///
    /// One call; a response with no recognizable heading triggers exactly
    /// one retry with the harder template instruction; a second malformed
    /// response degrades to its raw text as the summary body. Blank and
    /// over-budget responses fail the pass regardless of which call delivers
    /// them — the retry exists for shape problems, not content problems.
    async fn summarize(&self, dropped: &[Message], budget: u64) -> SummarizeOutcome {
        let transcript = render_evicted(dropped, self.config.transcript_max_chars);
        let user_prompt = format!(
            "Summarize the conversation in <conversation> into the sections the system \
             message specifies, so another agent can continue the work without \
             re-reading it.\n\n<conversation>\n{transcript}\n</conversation>"
        );
        let mut retry = false;
        loop {
            let system = self.system_prompt(budget, retry);
            let request = StreamRequest::new(vec![Message::user(user_prompt.clone())])
                .with_system(Some(system));
            let response = match self.client.create_message(&request).await {
                Ok(response) => response,
                Err(error) => return SummarizeOutcome::Failed(error),
            };
            let text = response.message.text_content();
            if text.trim().is_empty() {
                return SummarizeOutcome::Blank;
            }
            if estimate_text_tokens(&text) > budget {
                return SummarizeOutcome::OverBudget { budget };
            }
            if !self.config.parse_output {
                return SummarizeOutcome::Parsed {
                    summary: raw_text_summary(text),
                    retried: retry,
                };
            }
            if let Some(summary) = Self::parse_summary(&text) {
                return SummarizeOutcome::Parsed {
                    summary,
                    retried: retry,
                };
            }
            if retry {
                return SummarizeOutcome::Parsed {
                    summary: raw_text_summary(text),
                    retried: true,
                };
            }
            retry = true;
        }
    }

    /// Parse a response into a structured summary, tolerantly.
    ///
    /// Prefers a JSON object keyed by the sections' `snake_case` names (the
    /// shape a JSON-mode provider emits); falls back to scanning for the
    /// canonical headings case-insensitively under any `#` depth, with `- `,
    /// `* `, or numbered bullets. Unknown headings and non-bullet lines are
    /// ignored; each section caps at [`MAX_ITEMS_PER_SECTION`] entries.
    /// Returns `None` when no recognizable section appears at all — the
    /// caller's retry signal.
    fn parse_summary(text: &str) -> Option<StructuredSummary> {
        if let Some(summary) = Self::parse_json_summary(text) {
            return Some(summary);
        }
        Self::parse_heading_summary(text)
    }

    /// The JSON-object parse path.
    ///
    /// Recognizes an object carrying at least one known section key whose
    /// array holds a string entry, or whose array is empty — the model
    /// stating a section has nothing is a recognized answer. An array whose
    /// entries are all non-strings (numbers, nested objects) carries content
    /// the parse cannot represent, so it leaves the object unrecognized and
    /// the caller's retry/degrade path keeps that content. Unknown keys are
    /// ignored; any other JSON shape falls through to the heading scan.
    fn parse_json_summary(text: &str) -> Option<StructuredSummary> {
        let value = extract_json_substring(text.trim())?;
        let serde_json::Value::Object(map) = &value else {
            return None;
        };
        let mut summary = StructuredSummary::default();
        let mut recognized = false;
        for section in SummarySection::ALL {
            let Some(serde_json::Value::Array(items)) = map.get(section.json_key()) else {
                continue;
            };
            let entries = items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .filter(|entry| !entry.trim().is_empty())
                .take(MAX_ITEMS_PER_SECTION)
                .collect::<Vec<String>>();
            if !entries.is_empty() {
                recognized = true;
                *section_mut(&mut summary, section) = entries;
            } else if items.is_empty() {
                recognized = true;
            }
        }
        recognized.then_some(summary)
    }

    /// The heading-scan parse path.
    ///
    /// Walks the lines: a heading line (any `#` depth, case-insensitive,
    /// optional trailing colon or bold markers) names the current section;
    /// bullet lines under it append their text. A fenced code block's
    /// markers are skipped so a fenced whole-document answer still parses.
    fn parse_heading_summary(text: &str) -> Option<StructuredSummary> {
        let mut summary = StructuredSummary::default();
        let mut current: Option<SummarySection> = None;
        let mut recognized = false;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                continue;
            }
            if let Some(section) = heading_section(trimmed) {
                recognized = true;
                current = Some(section);
                continue;
            }
            let Some(section) = current else {
                continue;
            };
            if let Some(bullet) = bullet_item(trimmed) {
                let entries = section_mut(&mut summary, section);
                if entries.len() < MAX_ITEMS_PER_SECTION {
                    entries.push(bullet.to_string());
                }
            }
        }
        recognized.then_some(summary)
    }

    /// The failure outcome for a typed summarization failure.
    ///
    /// The original messages return intact and the reason names the cause —
    /// the seam a fallback chain catches.
    fn fail(
        original: Vec<Message>,
        context: &CompactionContext,
        reason: String,
    ) -> CompactionOutcome {
        tracing::warn!(
            target: "loopctl::compact",
            reason = %reason,
            "structured summarizer pass failed; returning the original messages"
        );
        CompactionOutcome::failed(original, context.tokens_before, reason)
    }

    /// Emit the pass's telemetry.
    ///
    /// One sections-counter event per configured section (populated or
    /// empty — a climbing empty rate is the earliest visible sign the
    /// template has quietly degraded) and one pass-complete debug line.
    /// No duration: the engine's clock-seamed telemetry already reports the
    /// whole pass.
    fn emit_telemetry(&self, summary: &StructuredSummary, budget: u64, retried: bool) {
        for section in &self.config.template.sections {
            let state = if section.items(summary).is_empty() {
                "empty"
            } else {
                "populated"
            };
            tracing::debug!(
                target: "loopctl::metrics",
                metric = "loopctl.compaction.structured.sections",
                section = section.slug(),
                state,
                "one structured-summary section state for one compaction pass"
            );
        }
        tracing::debug!(
            target: "loopctl::compact",
            budget,
            populated = summary.populated_count(&self.config.template.sections),
            retried,
            "structured summarizer pass complete"
        );
    }
}

impl ContextCompactor for StructuredSummarizer {
    fn compact(
        &self,
        messages: Vec<Message>,
        target_tokens: u64,
        context: CompactionContext,
    ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
        Box::pin(async move {
            let original = messages.clone();
            if messages.len() <= self.config.min_messages {
                return CompactionOutcome::no_change(messages);
            }
            let split = self.splitter().split(&messages);
            let dropped = summarizable(&split.to_compact);
            if dropped.is_empty() {
                return CompactionOutcome::no_change(messages);
            }

            let budget = self.config.summary_budget(target_tokens);
            let outcome = self.summarize(dropped, budget).await;
            let summary = match outcome {
                SummarizeOutcome::Parsed { summary, retried } => (summary, retried),
                SummarizeOutcome::Blank => {
                    return Self::fail(
                        original,
                        &context,
                        "structured summarizer step failed: the model returned an empty summary"
                            .to_string(),
                    );
                }
                SummarizeOutcome::OverBudget { budget } => {
                    return Self::fail(
                        original,
                        &context,
                        format!(
                            "structured summarizer step failed: the model returned a summary \
                             over the {budget}-token budget"
                        ),
                    );
                }
                SummarizeOutcome::Failed(error) => {
                    return Self::fail(
                        original,
                        &context,
                        format!("structured summarizer step failed: {error}"),
                    );
                }
            };

            let text = render(&summary.0, &self.config.template);
            let mut out = Vec::with_capacity(split.preserved.len().saturating_add(2));
            if leading_system(&messages)
                && let Some(first) = messages.first()
            {
                out.push(first.clone());
            }
            out.push(Message::assistant(text));
            out.extend_from_slice(&split.preserved);
            let tokens_after = context.counter.count(&out);
            self.emit_telemetry(&summary.0, budget, summary.1);
            CompactionOutcome::compacted(out, context.tokens_before, tokens_after)
                .with_evicted(dropped.to_vec())
        })
    }
}

/// The one-line purpose a section's prompt entry carries.
///
/// Tells the model what belongs in the section in the same terms the type's
/// docs use, so the prompt and the docs cannot drift apart silently.
const fn section_purpose(section: SummarySection) -> &'static str {
    match section {
        SummarySection::KeyFacts => {
            "Hard facts: file paths, identifiers, configuration values, \
             environment details a future turn would re-derive expensively."
        }
        SummarySection::Decisions => {
            "Choices committed to, each with its reason where the \
             conversation recorded one."
        }
        SummarySection::PendingTasks => {
            "Work in flight or explicitly outstanding; the continuation's \
             implicit todo list."
        }
        SummarySection::OpenQuestions => {
            "Unknowns raised but not answered; things the continuation \
             should know it does not know."
        }
    }
}

/// The summary a raw (unparsed) response degrades to.
///
/// The whole text rides as the single key-facts entry so the render still
/// produces a section-shaped message; the fallback never loses data.
fn raw_text_summary(text: String) -> StructuredSummary {
    StructuredSummary::new(vec![text], Vec::new(), Vec::new(), Vec::new())
}

/// A mutable borrow of one section's entries.
///
/// The parser's single append point, so a new section variant lands in one
/// match alongside its reader.
fn section_mut(summary: &mut StructuredSummary, section: SummarySection) -> &mut Vec<String> {
    match section {
        SummarySection::KeyFacts => &mut summary.key_facts,
        SummarySection::Decisions => &mut summary.decisions,
        SummarySection::PendingTasks => &mut summary.pending_tasks,
        SummarySection::OpenQuestions => &mut summary.open_questions,
    }
}

/// Which canonical section a heading line names, if any.
///
/// Accepts any `#` depth, any casing, an optional trailing colon, and
/// surrounding bold markers — the tolerant half of the heading contract.
fn heading_section(line: &str) -> Option<SummarySection> {
    let stripped = line.trim_start_matches('#').trim();
    let stripped = stripped.trim_matches('*').trim();
    let stripped = stripped.strip_suffix(':').unwrap_or(stripped);
    let lowered = stripped.to_lowercase();
    SummarySection::ALL
        .into_iter()
        .find(|section| section.title().to_lowercase() == lowered)
}

/// The bullet text a list line carries, if it is one.
///
/// Recognizes `- `, `* `, and numbered `N. ` / `N) ` markers, mirroring the
/// QA summarizer's question-list tolerance; any other line is not a bullet.
fn bullet_item(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return (!rest.trim().is_empty()).then_some(rest.trim());
    }
    numbered_bullet(line)
}

/// Strip a numbered marker, requiring digits before it.
///
/// Requiring at least one digit keeps prose like `3.14 radians` from
/// parsing as bullet three.
fn numbered_bullet(line: &str) -> Option<&str> {
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let after_digits = line.get(digits..)?;
    let rest = after_digits
        .strip_prefix(". ")
        .or_else(|| after_digits.strip_prefix(") "))?;
    (!rest.trim().is_empty()).then_some(rest.trim())
}

/// The heuristic token estimate for one summary-sized text.
///
/// The same message-wrapped heuristic every compactor self-reports with, so
/// the budget check and the outcome's counter speak one unit.
fn estimate_text_tokens(text: &str) -> u64 {
    CompactionOutcome::estimate_tokens(std::slice::from_ref(&Message::assistant(text.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ApiClient;
    use crate::api::NonStreamingResponse;
    use crate::api::error::ApiError;
    use crate::compact::HeuristicTokenCounter;
    use crate::compact::TokenCounter;
    use crate::compact::types::CompactReason;
    use crate::compact::{ContextManager, EnsureContextResult};
    use crate::error::recover_guard;
    use crate::message::Role;
    use crate::stream::{StreamEvent, StreamStopReason};
    use futures::Stream;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::Mutex;

    fn ok(text: &str) -> Result<NonStreamingResponse, ApiError> {
        Ok(NonStreamingResponse {
            message: Message::assistant(text),
            stop_reason: StreamStopReason::EndTurn,
            usage: None,
        })
    }

    /// An `ApiClient` double recording every call's user and system prompts
    /// and serving the scripted responses in order.
    struct RecordingClient {
        calls: Mutex<Vec<(String, String)>>,
        script: Mutex<VecDeque<Result<NonStreamingResponse, ApiError>>>,
    }

    impl RecordingClient {
        fn new(script: Vec<Result<NonStreamingResponse, ApiError>>) -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
                script: Mutex::new(script.into()),
            })
        }

        fn calls(&self) -> Vec<(String, String)> {
            recover_guard(self.calls.lock()).clone()
        }

        fn user_prompts(&self) -> Vec<String> {
            self.calls().into_iter().map(|(user, _)| user).collect()
        }

        fn systems(&self) -> Vec<String> {
            self.calls().into_iter().map(|(_, system)| system).collect()
        }
    }

    impl ApiClient for RecordingClient {
        fn model(&self) -> String {
            "structured-test".to_string()
        }

        fn stream_messages(
            &self,
            _request: &StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            Box::pin(futures::stream::empty())
        }

        fn create_message(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            let user = request
                .messages
                .first()
                .map_or(String::new(), Message::text_content);
            let system = request.system.clone().unwrap_or_default();
            recover_guard(self.calls.lock()).push((user, system));
            let next = recover_guard(self.script.lock())
                .pop_front()
                .expect("every scripted call gets a response");
            Box::pin(std::future::ready(next))
        }
    }

    fn scripted(
        script: Vec<Result<NonStreamingResponse, ApiError>>,
    ) -> (Arc<RecordingClient>, StructuredSummarizer) {
        scripted_with(script, StructuredSummaryConfig::default())
    }

    /// The config-parameterized construction the non-default configs need.
    ///
    /// `scripted` covers the default-config majority; this twin accepts the
    /// config so the over-budget, parse-off, and transcript-budget pins can
    /// build their exact fixtures.
    fn scripted_with(
        script: Vec<Result<NonStreamingResponse, ApiError>>,
        config: StructuredSummaryConfig,
    ) -> (Arc<RecordingClient>, StructuredSummarizer) {
        let client = RecordingClient::new(script);
        let summarizer = StructuredSummarizer::new(Arc::clone(&client) as SharedApiClient, config);
        (client, summarizer)
    }

    /// Six user/assistant pairs with unique, probe-able text per message.
    ///
    /// Each message carries its turn number twice, so prompt-presence and
    /// transcript-budget assertions can probe exact substrings without
    /// colliding across messages.
    fn conversation() -> Vec<Message> {
        let mut messages = Vec::new();
        for turn in 0..6 {
            messages.push(Message::user(format!(
                "user turn {turn} asks about topic-{turn}"
            )));
            messages.push(Message::assistant(format!(
                "assistant turn {turn} decided fact-{turn}"
            )));
        }
        messages
    }

    fn context_for(messages: &[Message]) -> CompactionContext {
        CompactionContext {
            tokens_before: CompactionOutcome::estimate_tokens(messages),
            reason: CompactReason::ThresholdExceeded,
            context_window: 1_000_000,
            turn: 3,
            counter: Arc::new(HeuristicTokenCounter),
            instructions: None,
            additional_context: Vec::new(),
        }
    }

    /// A canonical structured response with a known section payload.
    ///
    /// The exact heading-and-bullet shape the parser's contract names; the
    /// round-trip pin asserts this text back byte-for-byte under the header.
    fn canonical_response() -> String {
        "### Key facts\n- src/api.rs uses reqwest\n- the context window is 200k\n\n\
         ### Decisions\n- compaction is not feature-gated\n\n\
         ### Pending tasks\n- finish the tests\n\n\
         ### Open questions\n- should the ceiling scale?"
            .to_string()
    }

    #[tokio::test]
    async fn a_short_conversation_returns_no_change_with_zero_calls() {
        let (client, summarizer) = scripted(vec![]);
        let short: Vec<Message> = conversation().into_iter().take(8).collect();
        let outcome = summarizer
            .compact(short.clone(), 40_000, context_for(&short))
            .await;
        assert!(outcome.success, "a short conversation is a no-op success");
        assert_eq!(outcome.messages.len(), 8, "every message returns unchanged");
        assert!(
            client.calls().is_empty(),
            "no LLM call is spent below min_messages"
        );
    }

    #[tokio::test]
    async fn the_split_delegates_to_the_token_splitter() {
        let messages = conversation();
        let expected = TokenSplitter::new()
            .with_preserve_recent(6)
            .with_min_messages(8)
            .split(&messages);
        let (_client, summarizer) = scripted(vec![ok("### Key facts\n- a fact")]);
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let preserved = &outcome.messages[1..];
        assert_eq!(
            preserved.len(),
            expected.preserved.len(),
            "the preserved tail is the splitter's, not a reinvented boundary"
        );
        for (got, want) in preserved.iter().zip(&expected.preserved) {
            assert_eq!(got.text_content(), want.text_content());
        }
    }

    #[test]
    fn the_budget_takes_the_smaller_of_the_cap_and_the_share() {
        let config = StructuredSummaryConfig::default();
        assert_eq!(
            config.summary_budget(20_000),
            4_000,
            "the 0.20 share wins at 20k"
        );
        assert_eq!(
            config.summary_budget(40_000),
            6_000,
            "the 6 000 cap wins at 40k"
        );
        let capped = StructuredSummaryConfig::default().with_max_summary_tokens(2_000);
        assert_eq!(capped.summary_budget(40_000), 2_000);
        let low = StructuredSummaryConfig::default().with_summary_budget_pct(0.01);
        assert!((low.summary_budget_pct() - 0.05).abs() < f64::EPSILON);
        let high = StructuredSummaryConfig::default().with_summary_budget_pct(2.0);
        assert!((high.summary_budget_pct() - 0.95).abs() < f64::EPSILON);
        let rejected = StructuredSummaryConfig::default()
            .with_summary_budget_pct(f64::NAN)
            .with_max_summary_tokens(100_000);
        assert!((rejected.summary_budget_pct() - 0.20).abs() < f64::EPSILON);
        let computed = StructuredSummaryConfig::default().with_summary_budget_pct(0.1 + 0.2);
        assert_eq!(
            computed.summary_budget(20_000),
            6_000,
            "a computed fraction budgets at its true per-mille, not a saturated one"
        );
    }

    #[tokio::test]
    async fn the_prompt_names_exactly_the_configured_sections_and_headings() {
        let (client, summarizer) = scripted(vec![
            ok("### Key facts\n- a fact"),
            ok(&canonical_response()),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let (user, system) = client
            .calls()
            .first()
            .cloned()
            .expect("the pass made its call");
        assert!(
            system.contains("### Key facts")
                && system.contains("### Decisions")
                && system.contains("### Pending tasks")
                && system.contains("### Open questions"),
            "the system prompt names every configured heading exactly: {system}"
        );
        assert!(
            system.contains("No prose outside sections"),
            "the prompt forbids prose outside the sections"
        );
        assert!(
            system.contains("Keep the total under roughly 6000 tokens"),
            "the prompt states the enforced budget figure — the instructed counterpart of the ceiling: {system}"
        );
        assert!(
            system.contains("Hard facts: file paths, identifiers"),
            "each section entry carries its purpose line: {system}"
        );
        assert!(
            user.contains("<conversation>") && user.contains("user turn 0 asks about topic-0"),
            "the user prompt carries the dropped transcript: {user}"
        );
        let action = StructuredSummarizer::new(
            Arc::clone(&client) as SharedApiClient,
            StructuredSummaryConfig::default().with_template(SummaryTemplate::action_only()),
        );
        let messages = conversation();
        let outcome = action
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let system = client.systems()[1].clone();
        assert!(
            system.contains("### Decisions") && system.contains("### Pending tasks"),
            "the action template's sections are requested"
        );
        assert!(
            !system.contains("### Key facts") && !system.contains("### Open questions"),
            "sections outside the template are never requested: {system}"
        );
    }

    #[test]
    fn render_produces_the_header_and_each_section_in_template_order() {
        let summary = StructuredSummary::new(
            vec!["fact one".to_string()],
            vec!["decision one".to_string()],
            vec!["task one".to_string()],
            vec!["question one".to_string()],
        );
        let text = render(&summary, &SummaryTemplate::default_template());
        let expected = "## Conversation summary (compacted)\n\n\
             ### Key facts\n- fact one\n\n\
             ### Decisions\n- decision one\n\n\
             ### Pending tasks\n- task one\n\n\
             ### Open questions\n- question one";
        assert_eq!(text, expected, "the render is the canonical shape");
    }

    #[test]
    fn empty_sections_are_omitted_from_the_render() {
        let summary = StructuredSummary::new(
            vec!["fact one".to_string()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let text = render(&summary, &SummaryTemplate::default_template());
        assert!(text.contains("### Key facts"), "populated sections render");
        for absent in ["### Decisions", "### Pending tasks", "### Open questions"] {
            assert!(
                !text.contains(absent),
                "an empty section renders no bare heading: {text}"
            );
        }
    }

    #[test]
    fn a_fully_empty_summary_renders_the_stub_line() {
        let text = render(
            &StructuredSummary::default(),
            &SummaryTemplate::default_template(),
        );
        assert_eq!(
            text,
            format!("## Conversation summary (compacted)\n\n{EMPTY_SUMMARY_STUB}"),
            "the stub marks where the context was"
        );
    }

    #[test]
    fn the_action_only_template_renders_only_its_two_sections() {
        let summary = StructuredSummary::new(
            vec!["fact one".to_string()],
            vec!["decision one".to_string()],
            vec!["task one".to_string()],
            vec!["question one".to_string()],
        );
        let text = render(&summary, &SummaryTemplate::action_only());
        assert!(
            text.contains("### Decisions") && text.contains("### Pending tasks"),
            "the template's sections render"
        );
        assert!(
            !text.contains("### Key facts") && !text.contains("### Open questions"),
            "the template filters even populated sections: {text}"
        );
    }

    #[test]
    fn the_parser_tolerates_heading_case_markers_bullets_and_fences() {
        let canonical = StructuredSummarizer::parse_summary(&canonical_response())
            .expect("canonical headings parse");
        assert_eq!(
            canonical.key_facts,
            vec![
                "src/api.rs uses reqwest".to_string(),
                "the context window is 200k".to_string()
            ],
            "bullets collect under their heading"
        );
        assert_eq!(
            canonical.decisions,
            vec!["compaction is not feature-gated".to_string()]
        );

        let variant = StructuredSummarizer::parse_summary(
            "```markdown\n## key facts:\n* first fact\n1) second fact\n\n## KEY FACTS\n- replaced heading also parses\n```",
        )
        .expect("case, depth, colons, fences, and bullet markers are tolerated");
        assert_eq!(
            variant.key_facts,
            vec![
                "first fact".to_string(),
                "second fact".to_string(),
                "replaced heading also parses".to_string(),
            ],
            "every tolerated marker shape collects the same way"
        );
        assert!(
            variant.pending_tasks.is_empty() && variant.open_questions.is_empty(),
            "absent headings yield empty sections"
        );
    }

    #[test]
    fn the_parser_also_accepts_the_json_object_shape() {
        let json = r#"{"key_facts": ["path is /etc/app.conf"], "pending_tasks": ["run the migration"], "ignored_key": 7}"#;
        let summary = StructuredSummarizer::parse_summary(json)
            .expect("a JSON object with known keys parses");
        assert_eq!(summary.key_facts, vec!["path is /etc/app.conf".to_string()]);
        assert_eq!(summary.pending_tasks, vec!["run the migration".to_string()]);
        assert!(
            summary.decisions.is_empty(),
            "unknown keys and missing sections are simply absent"
        );
    }

    #[test]
    fn parse_then_render_round_trips_stably() {
        let parsed = StructuredSummarizer::parse_summary(&canonical_response())
            .expect("the canonical shape parses");
        let rendered = render(&parsed, &SummaryTemplate::default_template());
        assert_eq!(
            rendered,
            format!(
                "## Conversation summary (compacted)\n\n{}",
                canonical_response()
            ),
            "parse then render reproduces the canonical text under the header exactly"
        );
        let reparsed =
            StructuredSummarizer::parse_summary(&rendered).expect("the render parses back");
        assert_eq!(reparsed, parsed, "the round trip is stable");
    }

    #[test]
    fn the_parser_caps_each_section_at_fifty_items() {
        let mut runaway = String::from("### Key facts\n");
        for index in 0..80 {
            runaway.push_str("- fact ");
            runaway.push_str(&index.to_string());
            runaway.push('\n');
        }
        let summary =
            StructuredSummarizer::parse_summary(&runaway).expect("the heading is recognized");
        assert_eq!(
            summary.key_facts.len(),
            MAX_ITEMS_PER_SECTION,
            "a runaway section stops at the cap"
        );
    }

    #[tokio::test]
    async fn the_summary_message_is_assistant_role_and_precedes_the_preserved_tail() {
        let (_client, summarizer) = scripted(vec![ok(&canonical_response())]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        let first = outcome.messages.first().expect("the output is non-empty");
        assert_eq!(first.role, Role::Assistant, "the summary is assistant-role");
        assert!(first.text_content().starts_with("## Conversation summary"));
        assert_eq!(
            outcome.messages.len(),
            7,
            "one summary message plus six preserved"
        );
    }

    #[tokio::test]
    async fn a_leading_system_message_survives_and_never_rides_the_transcript() {
        let (client, summarizer) = scripted(vec![ok(&canonical_response())]);
        let mut messages = conversation();
        messages.insert(
            0,
            Message::new(
                Role::System,
                vec![crate::message::MessagePart::text("standing instructions")],
            ),
        );
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let (user, _system) = client
            .calls()
            .first()
            .cloned()
            .expect("the pass made its call");
        assert!(
            !user.contains("standing instructions"),
            "the system message survives at the head, so it must not ride the transcript: {user}"
        );
        let head = outcome.messages.first().expect("the output is non-empty");
        assert_eq!(
            (head.role, head.text_content()),
            (Role::System, "standing instructions".to_string()),
            "the survival contract is unchanged — the message still heads the output"
        );
        assert_eq!(
            outcome.messages.get(1).map(|message| message.role),
            Some(Role::Assistant),
            "the summary follows the system message"
        );
        assert!(
            outcome
                .evicted
                .iter()
                .all(|message| message.text_content() != "standing instructions"),
            "the pulled-back system message is never demoted"
        );
    }

    #[tokio::test]
    async fn an_api_error_fails_the_outcome_and_keeps_the_original_messages() {
        let (_client, summarizer) = scripted(vec![Err(ApiError::api("provider exploded"))]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(!outcome.success, "a provider failure fails the pass");
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains("structured summarizer") && error.contains("provider exploded"),
            "the error names the step and the cause: {error}"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(outcome.evicted.len(), 0, "nothing left the feed");
    }

    #[tokio::test]
    async fn a_blank_response_fails_the_pass() {
        let (client, summarizer) = scripted(vec![ok(""), ok("   \n\t")]);
        for expectation in ["empty", "whitespace"] {
            let messages = conversation();
            let outcome = summarizer
                .compact(messages.clone(), 40_000, context_for(&messages))
                .await;
            assert!(
                !outcome.success,
                "a {expectation} response must fail, not commit an empty summary"
            );
            assert!(
                outcome
                    .error
                    .as_deref()
                    .is_some_and(|error| error.contains("empty summary")),
                "the failure names the cause: {:?}",
                outcome.error
            );
            assert_eq!(
                outcome.messages.len(),
                messages.len(),
                "the original messages return intact"
            );
            assert_eq!(outcome.evicted.len(), 0, "nothing is demoted");
        }
        assert_eq!(
            client.calls().len(),
            2,
            "each blank shape costs exactly one call — no retry for content problems"
        );
    }

    #[tokio::test]
    async fn an_over_budget_summary_fails_the_pass() {
        let scripted_text = "### Key facts\n- a summary text of a known length";
        let as_message = Message::assistant(scripted_text.to_string());
        let at_budget = CompactionOutcome::estimate_tokens(std::slice::from_ref(&as_message));
        let one_under = at_budget.saturating_sub(1);

        let (client, summarizer) = scripted_with(
            vec![ok(scripted_text)],
            StructuredSummaryConfig::default().with_max_summary_tokens(one_under),
        );
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "a summary over the hard ceiling must fail the pass, not ship through it"
        );
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains(&format!("{one_under}-token budget")),
            "the error names the budget figure: {error}"
        );
        assert_eq!(
            client.calls().len(),
            1,
            "an over-budget response does not retry"
        );

        let (boundary_client, boundary) = scripted_with(
            vec![ok(scripted_text)],
            StructuredSummaryConfig::default().with_max_summary_tokens(at_budget),
        );
        let messages = conversation();
        let outcome = boundary
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            outcome.success,
            "a summary estimating exactly at the budget passes — the ceiling rejects strictly"
        );
        assert_eq!(
            boundary_client.calls().len(),
            1,
            "a parseable at-budget response needs no retry"
        );
        assert_eq!(
            outcome.messages.first().map(Message::text_content),
            Some(
                "## Conversation summary (compacted)\n\n### Key facts\n- a summary text of a known length"
                    .to_string()
            ),
            "the parsed at-budget summary renders inside the summary message"
        );
    }

    #[tokio::test]
    async fn a_malformed_first_response_retries_once_then_succeeds() {
        let (client, summarizer) = scripted(vec![
            ok("Sure! Here is a summary of the conversation in flowing prose."),
            ok(&canonical_response()),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success, "the retry recovers the shape");
        let systems = client.systems();
        assert_eq!(systems.len(), 2, "exactly one retry ran");
        assert!(
            systems[1].contains("MUST emit ONLY the sections"),
            "the retry prompt carries the sharper exact-headings instruction"
        );
        assert!(
            !systems[0].contains("MUST emit ONLY"),
            "the first prompt is the ordinary one"
        );
        let summary_text = outcome
            .messages
            .first()
            .map(Message::text_content)
            .unwrap_or_default();
        assert!(
            summary_text.contains("### Key facts")
                && summary_text.contains("src/api.rs uses reqwest"),
            "the parsed summary from the retry renders sectioned: {summary_text}"
        );
    }

    #[tokio::test]
    async fn two_malformed_responses_degrade_to_raw_text_and_succeed() {
        let (client, summarizer) = scripted(vec![
            ok("First prose answer with no headings at all."),
            ok("Second prose answer, still no headings."),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            outcome.success,
            "a second malformed response degrades to raw text, never errors: {outcome:?}"
        );
        assert_eq!(client.calls().len(), 2, "the retry cap held");
        let summary_text = outcome
            .messages
            .first()
            .map(Message::text_content)
            .unwrap_or_default();
        assert!(
            summary_text.contains("Second prose answer, still no headings."),
            "the raw text of the final response rides the summary message: {summary_text}"
        );
    }

    #[tokio::test]
    async fn parse_output_false_carries_the_raw_text() {
        let (client, summarizer) = scripted_with(
            vec![ok(&canonical_response())],
            StructuredSummaryConfig::default().with_parse_output(false),
        );
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        assert_eq!(client.calls().len(), 1, "no parse means no retry path");
        let summary_text = outcome
            .messages
            .first()
            .map(Message::text_content)
            .unwrap_or_default();
        assert!(
            summary_text.contains("### Key facts\n- ### Key facts"),
            "the unparsed whole text rides as the single key-facts entry: {summary_text}"
        );
    }

    #[tokio::test]
    async fn evicted_carries_the_dropped_slice_for_the_sink() {
        let (_client, summarizer) = scripted(vec![ok(&canonical_response())]);
        let messages = conversation();
        let expected = TokenSplitter::new()
            .with_preserve_recent(6)
            .with_min_messages(8)
            .split(&messages);
        let outcome = summarizer
            .compact(messages, 40_000, context_for(&expected.preserved))
            .await;
        assert_eq!(
            outcome.evicted.len(),
            expected.to_compact.len(),
            "every dropped message rides evicted in conversation order"
        );
        for (evicted, dropped) in outcome.evicted.iter().zip(&expected.to_compact) {
            assert_eq!(evicted.text_content(), dropped.text_content());
        }
        let kept: Vec<String> = outcome.messages.iter().map(Message::text_content).collect();
        assert!(
            outcome
                .evicted
                .iter()
                .all(|message| !kept.contains(&message.text_content())),
            "nothing listed as evicted survives in the output"
        );
    }

    /// A counter answering a constant per message — a figure the heuristic
    /// estimate cannot produce on these fixtures, so the pin distinguishes
    /// the context counter from the static estimate.
    struct ConstantPerMessageCounter(u64);

    impl TokenCounter for ConstantPerMessageCounter {
        fn count(&self, messages: &[Message]) -> u64 {
            self.0.saturating_mul(messages.len() as u64)
        }
    }

    #[tokio::test]
    async fn the_outcome_counts_tokens_with_the_context_counter() {
        let counter = Arc::new(ConstantPerMessageCounter(7));
        let (_client, summarizer) = scripted(vec![ok(&canonical_response())]);
        let messages = conversation();
        let mut context = context_for(&messages);
        context.counter = Arc::clone(&counter) as Arc<dyn TokenCounter>;
        let tokens_before = counter.count(&messages);
        context.tokens_before = tokens_before;
        let outcome = summarizer.compact(messages, 40_000, context).await;
        assert_eq!(
            outcome.messages.len(),
            7,
            "one summary message plus the six preserved"
        );
        assert_eq!(
            outcome.tokens_after,
            7 * 7,
            "tokens_after is the context counter's figure, not the static heuristic estimate"
        );
        assert_eq!(
            outcome.tokens_saved,
            tokens_before.saturating_sub(outcome.tokens_after),
            "tokens_saved is the before/after difference"
        );
    }

    #[tokio::test]
    async fn a_transcript_budget_bounds_the_prompt() {
        let (client, summarizer) = scripted_with(
            vec![ok(&canonical_response())],
            StructuredSummaryConfig::default().with_transcript_max_chars(60),
        );
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success);
        let user = client.user_prompts()[0].clone();
        assert!(
            user.contains("user turn 0 asks about topic-0"),
            "the head of the dropped slice still renders: {user}"
        );
        assert!(
            user.contains("[evicted "),
            "a saturated render closes with the truncation marker: {user}"
        );
        assert!(
            !user.contains("topic-3"),
            "messages past the transcript budget never reach the prompt: {user}"
        );
    }

    #[tokio::test]
    async fn the_summarizer_compacts_through_the_manager() {
        let (_client, summarizer) = scripted(vec![ok(&canonical_response())]);
        let manager = ContextManager::new(Arc::new(summarizer)).with_context_window(1_000_000);
        let messages = conversation();
        let result = manager.compact_manual(messages, 1).await;
        match result {
            Ok(EnsureContextResult::Compacted(outcome)) => {
                let first = outcome.messages.first().expect("the output is non-empty");
                assert!(
                    first.text_content().contains("### Key facts"),
                    "the sectioned summary leads the manager's compacted history"
                );
            }
            _ => panic!("a scripted pass through the manager classifies Compacted"),
        }
    }

    #[tokio::test]
    async fn a_cleared_sections_template_falls_back_to_the_default_and_never_discards() {
        let mut cleared = SummaryTemplate::default_template();
        cleared.sections.clear();
        let config = StructuredSummaryConfig::default().with_template(cleared);
        assert_eq!(
            config.template().sections.len(),
            SummarySection::ALL.len(),
            "an emptied sections list is rejected with the default template taking its place"
        );
        let (client, summarizer) = scripted_with(vec![ok(&canonical_response())], config);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(outcome.success, "the clamped config still compacts");
        let system = client.systems()[0].clone();
        assert!(
            system.contains("### Key facts") && system.contains("### Open questions"),
            "the fallback template's sections are requested, never a section-less prompt: {system}"
        );
        let summary_text = outcome
            .messages
            .first()
            .map(Message::text_content)
            .unwrap_or_default();
        assert!(
            summary_text.contains("### Key facts")
                && summary_text.contains("src/api.rs uses reqwest"),
            "the dropped context's content survives the render, never a bare header: {summary_text}"
        );
        assert_eq!(
            outcome.messages.len(),
            7,
            "one summary message plus the six preserved"
        );
    }

    #[test]
    fn render_falls_back_to_the_canonical_sections_when_the_template_has_none() {
        let mut cleared = SummaryTemplate::default_template();
        cleared.sections.clear();
        let summary = StructuredSummary::new(
            vec!["fact one".to_string()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let text = render(&summary, &cleared);
        assert!(
            text.contains("### Key facts") && text.contains("- fact one"),
            "a content-bearing summary never renders to a bare header: {text}"
        );
    }

    #[test]
    fn a_duplicate_section_renders_twice_by_design() {
        let duplicate = SummaryTemplate {
            sections: vec![SummarySection::Decisions, SummarySection::Decisions],
            header: DEFAULT_HEADER.to_string(),
        };
        let summary = StructuredSummary::new(
            Vec::new(),
            vec!["decision one".to_string()],
            Vec::new(),
            Vec::new(),
        );
        let text = render(&summary, &duplicate);
        assert_eq!(
            text.matches("### Decisions").count(),
            2,
            "a section listed twice renders twice — duplicates are honored literally: {text}"
        );
    }

    #[tokio::test]
    async fn a_numeric_only_json_response_degrades_to_raw_text() {
        let numeric = r#"{ "key_facts": [1, 2, 3] }"#;
        let (client, summarizer) = scripted(vec![ok(numeric), ok(numeric)]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            outcome.success,
            "unparsable content degrades to raw text, never errors: {outcome:?}"
        );
        assert_eq!(client.calls().len(), 2, "the retry cap held");
        let summary_text = outcome
            .messages
            .first()
            .map(Message::text_content)
            .unwrap_or_default();
        assert!(
            summary_text.contains("[1, 2, 3]"),
            "the model's non-string content rides the summary message instead of vanishing \
             behind the stub: {summary_text}"
        );
        assert!(
            !summary_text.contains(EMPTY_SUMMARY_STUB),
            "a content-bearing response never renders the nothing-salient stub: {summary_text}"
        );
    }

    #[test]
    fn an_all_empty_json_object_is_a_deliberate_empty_parse() {
        let summary = StructuredSummarizer::parse_summary(r#"{"key_facts": [], "decisions": []}"#)
            .expect("known keys with empty arrays are recognized");
        assert!(
            summary.is_empty(),
            "the model stating no sections is a valid empty parse, not a retry signal"
        );
    }

    #[tokio::test]
    async fn a_blank_retry_fails_the_pass() {
        let (client, summarizer) = scripted(vec![
            ok("Prose answer with no headings at all."),
            ok("   \n\t"),
        ]);
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "a blank response fails the pass regardless of which call delivers it"
        );
        assert!(
            outcome
                .error
                .as_deref()
                .is_some_and(|error| error.contains("empty summary")),
            "the failure names the cause: {:?}",
            outcome.error
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(client.calls().len(), 2, "the blank arrived on the retry");
    }

    #[tokio::test]
    async fn an_over_budget_retry_fails_the_pass() {
        let scripted_text = "### Key facts\n- a retry summary text of a known length";
        let as_message = Message::assistant(scripted_text.to_string());
        let at_budget = CompactionOutcome::estimate_tokens(std::slice::from_ref(&as_message));
        let one_under = at_budget.saturating_sub(1);
        let (client, summarizer) = scripted_with(
            vec![
                ok("Prose answer with no headings at all."),
                ok(scripted_text),
            ],
            StructuredSummaryConfig::default().with_max_summary_tokens(one_under),
        );
        let messages = conversation();
        let outcome = summarizer
            .compact(messages.clone(), 40_000, context_for(&messages))
            .await;
        assert!(
            !outcome.success,
            "an over-budget response fails the pass regardless of which call delivers it"
        );
        let error = outcome.error.expect("the failure carries a reason");
        assert!(
            error.contains(&format!("{one_under}-token budget")),
            "the error names the budget figure: {error}"
        );
        assert_eq!(
            outcome.messages.len(),
            messages.len(),
            "the original messages return intact"
        );
        assert_eq!(
            client.calls().len(),
            2,
            "the over-budget arrived on the retry"
        );
    }
}
