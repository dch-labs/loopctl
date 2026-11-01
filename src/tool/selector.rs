//! Per-turn tool advertisement: the dynamic tool surface.
//!
//! Decouples the *registered* surface (everything the
//! [`ToolRegistry`](crate::tool::ToolRegistry) knows) from the
//! *advertised* surface (the subset riding this turn's request).
//! Frontier models handle a wide tool list; small models pick the
//! wrong tool, conflate similar ones, or call tools that do not exist
//! — and every schema spends window tokens on every request. A
//! [`ToolSelector`] narrows what the model sees each turn (the
//! cheap alternative to a bigger model is less confusion).
//!
//! The engine consults the configured selector at each request build
//! and threads the chosen schemas into both the request and the
//! window estimate — the figure counts what the wire carries, never
//! the whole registry. A turn whose request is rebuilt after a
//! compaction pass re-selects against the compacted conversation, so
//! the choice always describes the request it rides. The default is no selector: every registered tool
//! is advertised, byte-identical to a loop that has never heard of
//! selection. [`ToolConstraint::Strict`](crate::structured::ToolConstraint)
//! composes automatically — it constrains the model's calls to
//! whatever is advertised this turn.

use std::future::Future;
use std::pin::Pin;

use crate::message::Message;
use crate::tool::ToolSchema;

/// What the selector sees when narrowing this turn's advertisement.
///
/// Built by the engine at the turn's request build: the run's turn
/// number, the text of the prompt that opened this run, and the
/// conversation the request carries. Selectors must treat the context
/// as read-only advice — the engine threads their output into the
/// request and the window estimate as one selection.
#[derive(Debug, Clone, Copy)]
pub struct SelectionContext<'a> {
    /// The current turn number within the run.
    ///
    /// Selectors that adapt to depth — fewer tools early, more once
    /// the task's shape is clear — read it here.
    pub turn: usize,

    /// The text of the prompt that opened this run, if any input was
    /// given.
    ///
    /// The strongest selection signal a turn carries; `None` on a
    /// promptless turn, where a fail-open selector advertises
    /// everything.
    pub latest_user_message: Option<&'a str>,

    /// The conversation the request carries, in order.
    ///
    /// The engine's request-build view: transients lead, committed
    /// history follows — the same messages the model sees.
    pub conversation: &'a [Message],
}

/// Chooses which registered tools to advertise to the model this turn.
///
/// Implementations must be deterministic for a given context — the
/// engine threads the result into both the request and its window
/// estimate, so a selector whose output varies between equal contexts
/// would make the estimate unexplainable. The trait is object-safe
/// for use as `Arc<dyn ToolSelector>`.
///
/// # Contract
///
/// - The returned schemas must be members of the `all` slice the
///   engine passed in — the engine advertises the selection verbatim,
///   so a fabricated schema would advertise a tool the registry cannot
///   dispatch.
/// - An empty return advertises no tools this turn (the request omits
///   the tools field); a selector that never wants that outcome owns
///   its fail-open policy (see [`KeywordSelector`]).
/// - [`select`](ToolSelector::select) must not mutate the selector
///   (it takes `&self`).
pub trait ToolSelector: Send + Sync {
    /// The subset of `all` to advertise this turn.
    ///
    /// Returned schemas must be members of `all` — the engine
    /// advertises the selection verbatim.
    fn select(&self, ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema>;

    /// The selector's stable name, as it appears on the `selector`
    /// label of tool-selection telemetry.
    ///
    /// A host-written selector keeps the default `"custom"`, so a
    /// dashboard never attributes a hand-rolled selection to one of the
    /// crate's own strategies; the built-ins override it with their own
    /// spelling. The value is a label, not an identity — two selectors
    /// of the same kind share one name by design.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// The async companion to [`ToolSelector`], for selections that need
/// a call to decide.
///
/// Consulted exactly once per turn at the same engine seam as the
/// sync trait — the request-build step ahead of the payload estimate —
/// so the estimate and the wire share one selection, whichever trait
/// produced it. Implementations own their failure policy and must
/// fail open: an embed error, a rewrite timeout, or unparseable
/// output advertises the full surface (the returned `Vec` carries no
/// error; log what happened and return `all` unchanged). Selection is
/// **not** required to be deterministic — the sync trait's
/// determinism contract is deliberately relaxed here — because the
/// value of an async selector is exactly the judgment a call buys;
/// the engine's once-per-turn discipline is what keeps the estimate
/// explainable, not replayability. One distinction the engine cannot
/// make and the implementation must: an **intentional** empty
/// selection is a selector's right (`with_top_k(0)`-shaped choices),
/// but an **inability to rank** — a failed call, an unrankable
/// vector — must fail open to the full surface; the engine
/// advertises whatever comes back, silence included.
///
/// The engine takes a sync selector or an async one, never both:
/// configuring one clears the other.
pub trait AsyncToolSelector: Send + Sync {
    /// The subset of `all` to advertise this turn.
    ///
    /// Same contract as [`ToolSelector::select`]: returned schemas
    /// must be members of `all`, an empty return advertises nothing,
    /// and fail-open is the implementation's own responsibility.
    fn select<'a>(
        &'a self,
        ctx: &'a SelectionContext<'a>,
        all: &'a [ToolSchema],
    ) -> Pin<Box<dyn Future<Output = Vec<ToolSchema>> + Send + 'a>>;

    /// The selector's stable name, as it appears on the `selector`
    /// label of tool-selection telemetry.
    ///
    /// Same contract as [`ToolSelector::name`]: host implementations keep
    /// the `"custom"` default, built-ins name themselves.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// A semantic selector: embeds each tool once per registry generation
/// and the turn's query per turn, advertising the top-k nearest by
/// cosine similarity.
///
/// For registries where keyword recall is structurally insufficient
/// (MCP-mounted surfaces, fifty-plus tools): "look up rust news"
/// lands near "search the web" because the embeddings say so, not the
/// vocabulary. The tool corpus is embedded once and cached against
/// the ordered name list — a registry mutation (the runtime
/// surface) changes the list and re-embeds on the next turn — and the
/// query embeds fresh every turn. Any embed failure fails open to the
/// full surface with a warning: selection must never strand the
/// model. Deterministic for a fixed provider, but the sync trait's
/// determinism contract is relaxed (see [`AsyncToolSelector`]).
///
/// # Cost
///
/// One embedding call per turn for the query plus one call per
/// registry generation for the corpus — cheap relative to a rewrite
/// call, and nothing on a miss.
#[cfg(feature = "vector_index")]
pub struct EmbeddingSelector<P> {
    /// The embedding source both lanes consult.
    ///
    /// Corpus and query vectors come from the same provider, so one
    /// dimensionality — and one quality profile — underpins every
    /// ranking.
    provider: P,

    /// How many of the nearest tools a selection advertises.
    ///
    /// Clamped to the registry size at selection time; tuned through
    /// [`with_top_k`](Self::with_top_k), defaulting to 8.
    top_k: usize,

    /// The per-generation corpus cache.
    ///
    /// Holds the embedded tool set behind an interior lock that is
    /// released across the embed await (an await under a std lock is
    /// forbidden by the crate's lints); see [`EmbeddingCache`] for
    /// the invalidation rule.
    cache: std::sync::Mutex<EmbeddingCache>,
}

#[cfg(feature = "vector_index")]
struct EmbeddingCache {
    /// The embedded texts the corpus embeddings describe.
    ///
    /// A registry generation's identity: each text is the
    /// `"{tool} {description}"` the selection built, so a membership
    /// change **and** a same-name description overwrite (the
    /// `changed` surface) both invalidate the corpus on the next
    /// selection — the semantic position always follows the tool's
    /// current text.
    texts: Vec<String>,

    /// Each tool's embedding, aligned with `texts`.
    ///
    /// Built by one batch embed per registry generation; indexed
    /// beside `texts` so a ranking maps straight back to its schema.
    embeddings: Vec<crate::memory::vector::Embedding>,
}

#[cfg(feature = "vector_index")]
impl<P: crate::memory::vector::EmbeddingProvider> EmbeddingSelector<P> {
    /// Build over an embedding provider, advertising the top 8.
    ///
    /// The corpus embeds lazily on the first selection.
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            top_k: 8,
            cache: std::sync::Mutex::new(EmbeddingCache {
                texts: Vec::new(),
                embeddings: Vec::new(),
            }),
        }
    }

    /// Advertise the k nearest tools instead of 8.
    ///
    /// `k` is clamped to the registry size at selection time, so an
    /// over-large k simply advertises everything.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// The corpus embeddings for `all`, re-embedding when the ordered
    /// name list moved.
    ///
    /// The cache read happens under the lock; the embed call happens
    /// with it released (an await under a std lock is forbidden by
    /// the crate's lints). The read-side key check is the guard: a
    /// concurrent same-generation caller either finds the cache
    /// already populated or re-embeds and stores — last writer wins
    /// with the same key, so the corpus stays consistent.
    /// # Errors
    ///
    /// The provider's embed failure for the corpus batch — the
    /// caller fails open.
    async fn corpus(
        &self,
        texts: &[String],
    ) -> Result<Vec<crate::memory::vector::Embedding>, crate::error::LoopError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        {
            let cache = crate::error::recover_guard(self.cache.lock());
            if *cache.texts == *texts {
                return Ok(cache.embeddings.clone());
            }
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embedded = self.provider.embed_batch(&refs).await?;
        let mut cache = crate::error::recover_guard(self.cache.lock());
        cache.texts = texts.to_vec();
        cache.embeddings = embedded;
        Ok(cache.embeddings.clone())
    }
}

#[cfg(feature = "vector_index")]
impl<P: crate::memory::vector::EmbeddingProvider> AsyncToolSelector for EmbeddingSelector<P> {
    fn name(&self) -> &'static str {
        "embedding"
    }

    fn select<'a>(
        &'a self,
        ctx: &'a SelectionContext<'a>,
        all: &'a [ToolSchema],
    ) -> Pin<Box<dyn Future<Output = Vec<ToolSchema>> + Send + 'a>> {
        Box::pin(async move {
            if all.is_empty() {
                return all.to_vec();
            }
            let Some(query) = ctx.latest_user_message else {
                return all.to_vec();
            };
            let texts: Vec<String> = all
                .iter()
                .map(|schema| format!("{} {}", schema.tool, schema.description))
                .collect();
            let corpus = match self.corpus(&texts).await {
                Ok(corpus) => corpus,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "embedding the tool corpus failed; advertising the full surface"
                    );
                    return all.to_vec();
                }
            };
            let embedded_query = match self.provider.embed(query).await {
                Ok(embedded) => embedded,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "embedding the query failed; advertising the full surface"
                    );
                    return all.to_vec();
                }
            };
            let mut ranked: Vec<(usize, f32)> = corpus
                .iter()
                .zip(&texts)
                .enumerate()
                .filter_map(|(index, (embedding, _))| {
                    let score = cosine(embedded_query.as_slice(), embedding.as_slice());
                    (score.is_finite()).then_some((index, score))
                })
                .collect();
            if ranked.is_empty() {
                tracing::warn!(
                    "no tool scored a finite similarity — an unrankable query \
                     advertises the full surface rather than an empty selection"
                );
                return all.to_vec();
            }
            ranked.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = self.top_k.min(all.len());
            let chosen: Vec<usize> = ranked
                .into_iter()
                .take(take)
                .map(|(index, _)| index)
                .collect();
            all.iter()
                .enumerate()
                .filter(|(index, _)| chosen.contains(index))
                .map(|(_, schema)| schema.clone())
                .collect()
        })
    }
}

/// Cosine similarity between two vectors.
///
/// Zero-norm vectors score `None` (excluded upstream) rather than
/// NaN; equal-length is assumed from a single provider's fixed
/// dimensionality.
#[cfg(feature = "vector_index")]
fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (l, r) in left.iter().zip(right) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return f32::NAN;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

/// A query-rewriting selector: a small-model call that sees the tool
/// names and one-line summaries — never the schemas — plus the run
/// prompt and recent context, and answers with this turn's tool list.
///
/// The strongest recall layer below "advertise everything": judgment
/// instead of vocabulary, at the cost of one small non-streaming call
/// per turn. Names and one-line summaries only, because a rewrite
/// that sees full schemas spends the tokens selection exists to save.
/// Failure fails open everywhere — a timeout, a transport error, a
/// budget refusal, or output naming no known tool advertises the full
/// surface with a warning; output naming *some* tools advertises
/// exactly those, in registry order. Determinism is relaxed and
/// documented (see [`AsyncToolSelector`]); the engine's one-selection
/// -per-turn seam keeps estimate and wire on the same answer.
///
/// # Cost
///
/// One non-streaming model call per turn, bounded by
/// [`with_timeout`](Self::with_timeout) (default 10 seconds) and by
/// the optional [`with_budget`](Self::with_budget) gate, with each
/// conversation excerpt capped at 400 bytes so the prompt grows with
/// the tool list, never with the history's bodies — the trade-off
/// stated plainly: this selector spends tokens to save window.
pub struct RewritingSelector<C> {
    /// The model the rewrite calls ride on.
    ///
    /// Shared behind an [`Arc`](std::sync::Arc) with whatever else
    /// uses the same handle — a small model on a cheap route is the
    /// intended wiring.
    client: std::sync::Arc<C>,

    /// How long one rewrite call may take.
    ///
    /// Set through [`with_timeout`](Self::with_timeout), defaulting
    /// to 10 seconds; a call that outlives it fails open instead of
    /// stalling the turn's request build.
    timeout: std::time::Duration,

    /// The optional spend gate over the rewrite calls.
    ///
    /// Built by [`with_budget`](Self::with_budget); fed every call's
    /// reported usage and consulted before each new call, refusing on
    /// a hard crossing.
    budget: Option<crate::budget::BudgetGate>,

    /// Whether the gate's spend window has been latched open.
    ///
    /// Flipped exactly once, at the first budgeted selection, so
    /// `begin` resets the spend state a single time for the
    /// selector's lifetime.
    budget_begun: std::sync::atomic::AtomicBool,

    /// The instant the spend window started.
    ///
    /// Latched alongside `budget_begun`; the elapsed since it is what
    /// the gate's wall-clock dimension measures — the clock the
    /// caller-side check needs, since the gate records no start of
    /// its own.
    budget_started_at: std::sync::Mutex<Option<std::time::Instant>>,
}

impl<C: crate::api::ApiClient> RewritingSelector<C> {
    /// Build over a client, one call per turn bounded to 10 seconds.
    ///
    /// The client is the same handle the loop uses — a small model on
    /// a cheap route is the intended wiring.
    #[must_use]
    pub fn new(client: std::sync::Arc<C>) -> Self {
        Self {
            client,
            timeout: std::time::Duration::from_secs(10),
            budget: None,
            budget_begun: std::sync::atomic::AtomicBool::new(false),
            budget_started_at: std::sync::Mutex::new(None),
        }
    }

    /// Bound each rewrite call to `timeout` instead of 10 seconds.
    ///
    /// A rewrite that cannot answer in time fails open rather than
    /// stalling the turn's request build.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Gate the rewrite calls on their own budget.
    ///
    /// The selector owns a [`BudgetGate`](crate::budget::BudgetGate)
    /// built from `limits`, feeds it every rewrite call's reported
    /// usage, and measures wall-clock spend from its first selection —
    /// the existing budget machinery, applied to selection spend. A
    /// hard crossing (tokens or duration) refuses further rewrite
    /// calls (each failing open to the full surface) for the
    /// selector's lifetime.
    #[must_use]
    pub fn with_budget(mut self, limits: crate::budget::BudgetLimits) -> Self {
        self.budget = Some(crate::budget::BudgetGate::new(limits));
        self
    }

    /// The one-line summary of a tool for the rewrite prompt.
    ///
    /// The first sentence-ish line of the description, trimmed —
    /// names plus summaries, never schemas.
    fn summary(schema: &ToolSchema) -> &str {
        schema
            .description
            .split('.')
            .next()
            .map_or(schema.description.as_str(), str::trim)
    }

    /// The bounded excerpt of a conversation message.
    ///
    /// Tool-name judgment needs a message's shape, not its body: each
    /// excerpt keeps the first `MAX_EXCERPT_BYTES` on a character
    /// boundary with an explicit marker, so a hundred-kilobyte tool
    /// result cannot ride the rewrite prompt — or its response —
    /// verbatim.
    fn bound_excerpt(text: &str) -> String {
        const MAX_EXCERPT_BYTES: usize = 400;
        if text.len() <= MAX_EXCERPT_BYTES {
            return text.to_string();
        }
        let cut = text.floor_char_boundary(MAX_EXCERPT_BYTES);
        format!("{}…", &text[..cut])
    }

    /// Render the rewrite prompt.
    ///
    /// Names and one-line summaries, the recent conversation, and the
    /// request — the shapes a small model needs to judge, and nothing
    /// it does not.
    fn prompt(ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> String {
        let mut prompt = String::from(
            "You select tools. Reply with a comma-separated list of tool \
             names from the list below, nothing else.\n\n",
        );
        prompt.push_str("Tools:\n");
        for schema in all {
            prompt.push_str(schema.tool.as_str());
            prompt.push_str(": ");
            prompt.push_str(Self::summary(schema));
            prompt.push('\n');
        }
        prompt.push_str("\nRecent conversation:\n");
        for message in ctx.conversation.iter().rev().take(6).rev() {
            let text = message.text_content();
            if !text.is_empty() {
                prompt.push_str(&message.role.to_string().to_lowercase());
                prompt.push_str(": ");
                prompt.push_str(&Self::bound_excerpt(&text));
                prompt.push('\n');
            }
        }
        prompt.push_str("\nThe request: ");
        prompt.push_str(ctx.latest_user_message.unwrap_or_default());
        prompt.push_str("\n\nTool names to advertise this turn:");
        prompt
    }

    /// Parse the model's answer into known tool names.
    ///
    /// Split on separators, trimmed, matched case-insensitively
    /// against the advertised names; unknown words drop, order
    /// follows the registry at the caller.
    fn parse_answer(answer: &str, all: &[ToolSchema]) -> Vec<String> {
        let lowered = answer.to_ascii_lowercase();
        lowered
            .split([',', '\n', ';'])
            .map(str::trim)
            .filter(|word| !word.is_empty())
            .filter_map(|word| {
                let trimmed = word.trim_start_matches("- ");
                all.iter()
                    .find(|schema| schema.tool.to_ascii_lowercase() == trimmed)
                    .map(|schema| schema.tool.clone())
            })
            .collect()
    }
}

impl<C: crate::api::ApiClient> AsyncToolSelector for RewritingSelector<C> {
    fn name(&self) -> &'static str {
        "rewriting"
    }

    fn select<'a>(
        &'a self,
        ctx: &'a SelectionContext<'a>,
        all: &'a [ToolSchema],
    ) -> Pin<Box<dyn Future<Output = Vec<ToolSchema>> + Send + 'a>> {
        Box::pin(async move {
            if all.is_empty() {
                return all.to_vec();
            }
            if let Some(gate) = &self.budget {
                use std::sync::atomic::Ordering::Relaxed;
                let started = *crate::error::recover_guard(self.budget_started_at.lock())
                    .get_or_insert_with(std::time::Instant::now);
                if !self.budget_begun.swap(true, Relaxed) {
                    gate.begin();
                }
                if gate.pre_request(started.elapsed()).1.is_some() {
                    tracing::warn!("rewrite budget exhausted; advertising the full surface");
                    return all.to_vec();
                }
            }
            let request =
                crate::api::StreamRequest::new(vec![Message::user(Self::prompt(ctx, all))]);
            let rewritten = tokio::time::timeout(self.timeout, async {
                match self.client.create_message(&request).await {
                    Ok(response) => {
                        let usage = response.usage;
                        Some((response.message.text_content(), usage))
                    }
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            "rewrite call failed; advertising the full surface"
                        );
                        None
                    }
                }
            })
            .await;
            let (answer, usage) = match rewritten {
                Ok(Some((text, usage))) => (text, usage),
                Ok(None) => return all.to_vec(),
                Err(_) => {
                    tracing::warn!(
                        timeout = ?self.timeout,
                        "rewrite call timed out; advertising the full surface"
                    );
                    return all.to_vec();
                }
            };
            if let (Some(gate), Some(usage)) = (&self.budget, usage) {
                gate.observe_usage(
                    u64::from(usage.input_tokens),
                    u64::from(usage.output_tokens),
                );
            }
            let chosen = Self::parse_answer(&answer, all);
            if chosen.is_empty() {
                tracing::warn!(
                    answer = %answer,
                    "rewrite named no known tool; advertising the full surface"
                );
                return all.to_vec();
            }
            all.iter()
                .filter(|schema| chosen.contains(&schema.tool))
                .cloned()
                .collect()
        })
    }
}

/// Advertise every registered tool — the loop's default behavior as an
/// explicit knob.
///
/// Selection-free loops get this for free; the type exists for hosts
/// that want the surface configurable without special-casing `None`,
/// and as the fail-open fallback other selectors document against.
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticSelector;

impl ToolSelector for StaticSelector {
    fn name(&self) -> &'static str {
        "static"
    }

    fn select(&self, _ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema> {
        all.to_vec()
    }
}

/// Advertise the tools whose name or description shares a word with
/// the turn's prompt.
///
/// Tokenizes the prompt and each tool's name + description into
/// lowercase alphanumeric words of length 3 or more (dropping most
/// stopwords and filler), and selects every tool whose tokens overlap
/// the prompt's — after the query's words pass through the synonym
/// table (see `DEFAULT_SYNONYM_PAIRS`): a user saying "verify",
/// "check", "find", or "look up" matches a tool described with
/// "search" or "read" without the description carrying every phrasing.
/// A [`with_core_tools`](Self::with_core_tools) set rides every
/// non-empty selection regardless of phrasing — the union of the
/// keyword matches and the core set — so a partial wrong match cannot
/// evict a tool every turn may need, while a query that matches
/// nothing still fails all the way open. A prompt whose expanded tokens share
/// nothing with any tool fails open to the full registered surface —
/// zero advertised tools would strand a model that must act, so the
/// failure mode is today's behavior, not silence. Deterministic and
/// cheap: allocation bounded by the registry size and the prompt
/// length.
#[derive(Debug, Clone, Default)]
pub struct KeywordSelector {
    /// Host-added synonym pairs, matched like the default table's.
    ///
    /// Accumulated by [`with_synonyms`](Self::with_synonyms);
    /// bidirectional and query-side, exactly like the curated
    /// defaults.
    extra_synonyms: Vec<(String, String)>,

    /// Tool names included in every selection regardless of the query.
    ///
    /// Accumulated by
    /// [`with_core_tools`](Self::with_core_tools); the union with the
    /// keyword matches is what a turn advertises.
    core_tools: Vec<String>,
}

impl KeywordSelector {
    /// Add host synonym pairs to the default table.
    ///
    /// Each pair is bidirectional and query-side, exactly like the
    /// curated defaults (see `DEFAULT_SYNONYM_PAIRS`): a query word
    /// matching either side of a pair also matches the other side
    /// against the tool text. Sides are lowercased on store, matching
    /// the tokenizer exactly, so capitalization never defeats a pair.
    /// Each side must be **one ASCII-alphanumeric token of at least 3
    /// characters that is not a stopword** — the shape the tokenizer
    /// keeps — or the pair can never fire; multiword phrasings belong
    /// to the phrase table, not the pair table. Repeated additions
    /// accumulate.
    #[must_use]
    pub fn with_synonyms(mut self, pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        self.extra_synonyms.extend(
            pairs
                .into_iter()
                .map(|(left, right)| (left.to_ascii_lowercase(), right.to_ascii_lowercase())),
        );
        self
    }

    /// Always include these tool names in every selection.
    ///
    /// The hybrid core set: the selection becomes the union of the
    /// keyword matches and these names, in registry order, so a
    /// partial wrong match cannot evict a tool every turn may need.
    /// The union composes with fail-open rather than replacing it — a
    /// query that matches nothing still fails all the way open to the
    /// full surface (core included), because a vocabulary miss must
    /// not strand the tool the prompt actually needs. Names not
    /// present in the registry are ignored.
    #[must_use]
    pub fn with_core_tools(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.core_tools.extend(names);
        self
    }

    /// The query's tokens expanded through the synonym tables.
    ///
    /// The default pairs, the host's additions, and the phrase table
    /// (checked against the raw query, since tokenizing drops the
    /// short words phrases are made of) each contribute their
    /// other-side word when a query token matches one side.
    /// Deduplicated; the original tokens always ride along.
    fn expand_query(&self, query: &str, tokens: &[String]) -> Vec<String> {
        let mut expanded = tokens.to_vec();
        let push_new = |word: &str, expanded: &mut Vec<String>| {
            if !expanded.iter().any(|seen| seen == word) {
                expanded.push(word.to_string());
            }
        };
        let defaults: Vec<(&str, &str)> = DEFAULT_SYNONYM_PAIRS.to_vec();
        let hosts: Vec<(&str, &str)> = self
            .extra_synonyms
            .iter()
            .map(|(left, right)| (left.as_str(), right.as_str()))
            .collect();
        for (left, right) in defaults.iter().chain(hosts.iter()) {
            if tokens.iter().any(|token| token == left) {
                push_new(right, &mut expanded);
            }
            if tokens.iter().any(|token| token == right) {
                push_new(left, &mut expanded);
            }
        }
        let lowered = query.to_ascii_lowercase();
        for (phrase, target) in DEFAULT_PHRASE_SYNONYMS {
            if contains_phrase(&lowered, phrase) {
                push_new(target, &mut expanded);
            }
        }
        expanded
    }
}

/// The curated synonym pairs behind keyword recall.
///
/// Bidirectional and query-side: a query containing either word also
/// matches tool text carrying the other. The pairs map the vocabulary
/// users actually use onto the vocabulary tools are described with —
/// a small, conservative set, because every pair widens recall for
/// both of its words. Hosts extend it through
/// [`with_synonyms`](KeywordSelector::with_synonyms); the tool-side
/// lever for vocabulary gaps is description wording, not expansion.
const DEFAULT_SYNONYM_PAIRS: &[(&str, &str)] = &[
    ("check", "verify"),
    ("check", "inspect"),
    ("check", "search"),
    ("find", "search"),
    ("find", "locate"),
    ("open", "read"),
    ("show", "read"),
    ("view", "read"),
    ("fix", "edit"),
    ("change", "edit"),
    ("modify", "edit"),
    ("create", "write"),
    ("make", "write"),
    ("start", "run"),
    ("launch", "run"),
];

/// The curated phrase pairs behind keyword recall.
///
/// Two-word phrases whose short words tokenizing would drop, mapped to
/// their tool-side single word; checked against the raw query with
/// word boundaries, so "look upon" does not fire the "look up" pair.
/// The phrase table is the discriminating layer for shared first
/// words: "look at" reads, "look up" and "look for" search — which is
/// exactly why no single-word `look` pair exists in
/// `DEFAULT_SYNONYM_PAIRS`.
const DEFAULT_PHRASE_SYNONYMS: &[(&str, &str)] = &[
    ("look up", "search"),
    ("look at", "read"),
    ("look for", "search"),
    ("figure out", "plan"),
];

/// Whether `lowered` contains `phrase` on word boundaries.
///
/// Every word of the phrase must appear consecutively with
/// non-alphanumeric neighbors on both sides, so a phrase never fires
/// from inside a longer word.
fn contains_phrase(lowered: &str, phrase: &str) -> bool {
    let mut haystack = lowered;
    let mut offset: usize = 0;
    while let Some(found) = haystack.find(phrase) {
        let start = offset.saturating_add(found);
        let end = start.saturating_add(phrase.len());
        let before_ok = start == 0
            || !lowered[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric);
        let after_ok = !lowered[end..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric);
        if before_ok && after_ok {
            return true;
        }
        offset = end;
        haystack = &lowered[offset..];
    }
    false
}

impl ToolSelector for KeywordSelector {
    fn name(&self) -> &'static str {
        "keyword"
    }

    fn select(&self, ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema> {
        let Some(query) = ctx.latest_user_message else {
            return all.to_vec();
        };
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return all.to_vec();
        }
        let expanded = self.expand_query(query, &query_tokens);
        let matched = all
            .iter()
            .filter(|schema| {
                let mut tool_text = schema.tool.clone();
                tool_text.push(' ');
                tool_text.push_str(&schema.description);
                tokenize(&tool_text)
                    .iter()
                    .any(|token| expanded.contains(token))
            })
            .cloned()
            .collect::<Vec<_>>();
        if matched.is_empty() {
            return all.to_vec();
        }
        let matched_names: Vec<&str> = matched.iter().map(|schema| schema.tool.as_str()).collect();
        all.iter()
            .filter(|schema| {
                matched_names.contains(&schema.tool.as_str())
                    || self.core_tools.contains(&schema.tool)
            })
            .cloned()
            .collect()
    }
}

/// Lowercase alphanumeric runs of length 3 or more that are not
/// stopwords, in order of first appearance.
///
/// The shared tokenizer behind [`KeywordSelector`]: punctuation and
/// casing vanish, two- and one-letter fragments (filler like "to",
/// "a", "of") drop, and so do the common English glue words that are
/// long enough to count but match everything — "the", "for", "with" —
/// a prompt's "the" must not advertise every tool whose description
/// says "the". The same word appearing twice in the query yields one
/// token — overlap is a set question, not a count.
fn tokenize(text: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "that", "this", "these", "those", "you", "your",
        "are", "was", "were", "has", "have", "had", "its", "can", "could", "will", "would",
        "should", "use", "using", "get", "gets", "all", "how", "what", "when", "where", "who",
        "why", "into", "out", "not", "but", "please",
    ];
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars().chain([' ']) {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() >= 3
                && !STOPWORDS.contains(&current.as_str())
                && !tokens.contains(&current)
            {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(name: &str, description: &str) -> ToolSchema {
        ToolSchema::new(name, description, serde_json::json!({"type": "object"}))
    }

    fn ctx(prompt: &str) -> SelectionContext<'_> {
        SelectionContext {
            turn: 0,
            latest_user_message: Some(prompt),
            conversation: &[],
        }
    }

    fn names(schemas: &[ToolSchema]) -> Vec<&str> {
        schemas.iter().map(|s| s.tool.as_str()).collect()
    }

    #[test]
    fn the_static_selector_advertises_every_registered_tool() {
        let all = vec![
            schema("read", "read a file"),
            schema("write", "write a file"),
        ];
        let selected = StaticSelector.select(&ctx("anything at all"), &all);
        assert_eq!(
            names(&selected),
            vec!["read", "write"],
            "static = the full surface"
        );
    }

    #[test]
    fn a_keyword_query_advertises_the_overlapping_tools_only() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected =
            KeywordSelector::default().select(&ctx("please search the web for rust news"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "only the overlapping tool rides"
        );
    }

    #[test]
    fn matching_is_case_insensitive_across_name_and_description() {
        let all = vec![
            schema("weather", "Report the forecast for a city"),
            schema("calc", "Evaluate an arithmetic expression"),
        ];
        let selected =
            KeywordSelector::default().select(&ctx("what is the WEATHER forecast"), &all);
        assert_eq!(
            names(&selected),
            vec!["weather"],
            "case and position never matter"
        );
    }

    #[test]
    fn a_keyword_miss_fails_open_to_the_full_surface() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
        ];
        let selected =
            KeywordSelector::default().select(&ctx("completely unrelated zebra topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "write_file"],
            "a miss is today's behavior, not silence"
        );
    }

    #[test]
    fn a_promptless_turn_fails_open_to_the_full_surface() {
        let all = vec![schema("read", "read a file")];
        let no_prompt = SelectionContext {
            turn: 3,
            latest_user_message: None,
            conversation: &[],
        };
        assert_eq!(
            names(&KeywordSelector::default().select(&no_prompt, &all)),
            vec!["read"],
            "nothing to match against advertises everything"
        );
    }

    #[test]
    fn short_fragments_never_match() {
        let all = vec![schema("ab", "A tool whose name is a short fragment")];
        let selected =
            KeywordSelector::default().select(&ctx("match me by the ab token please"), &all);
        assert_eq!(
            names(&selected),
            vec!["ab"],
            "the fail-open path serves a query of only-short tokens"
        );
    }

    #[test]
    fn selection_is_deterministic_for_an_unchanged_context() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let context = ctx("search the web");
        assert_eq!(
            names(&KeywordSelector::default().select(&context, &all)),
            names(&KeywordSelector::default().select(&context, &all)),
            "equal contexts select equal surfaces — the estimate must be explainable"
        );
    }

    #[test]
    fn the_selectors_are_object_safe() {
        let all = vec![schema("read", "read a file")];
        let selector: std::sync::Arc<dyn ToolSelector> =
            std::sync::Arc::new(KeywordSelector::default());
        assert_eq!(
            names(&selector.select(&ctx("read something"), &all)),
            vec!["read"],
            "Arc<dyn ToolSelector> dispatches"
        );
    }

    #[test]
    fn a_synonym_only_match_selects_the_tool() {
        let all = vec![
            schema("audit_build", "Check the build for regressions"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selected =
            KeywordSelector::default().select(&ctx("verify the deployment passed"), &all);
        assert_eq!(
            names(&selected),
            vec!["audit_build"],
            "a query word matching only through the synonym table still selects — \
             fail-open would advertise everything instead"
        );
    }

    #[test]
    fn check_the_weather_selects_the_search_tool_through_synonyms() {
        let all = vec![
            schema("web_search", "Search the web for pages"),
            schema("read_file", "Read a file from disk"),
        ];
        let selected = KeywordSelector::default().select(&ctx("check the weather"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "check -> search closes the vocabulary miss without the description \
             carrying the word itself"
        );
    }

    #[test]
    fn look_up_selects_via_the_phrase_table() {
        let all = vec![
            schema("web_search", "Search the web for pages"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look up rust news"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "the two-word phrase maps to its tool-side single word"
        );
    }

    #[test]
    fn host_synonyms_extend_the_default_table() {
        let all = vec![
            schema("food_tool", "Prepare a meal"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector =
            KeywordSelector::default().with_synonyms([("ruoka".to_string(), "meal".to_string())]);
        let selected = selector.select(&ctx("ruoka time"), &all);
        assert_eq!(
            names(&selected),
            vec!["food_tool"],
            "a host pair matches exactly like a default one"
        );
    }

    #[test]
    fn unknown_synonym_words_change_nothing() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
        ];
        let selector = KeywordSelector::default()
            .with_synonyms([("zebra".to_string(), "platypus".to_string())]);
        let selected = selector.select(&ctx("completely unrelated topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "write_file"],
            "an unmatched expansion is a no-op and the miss still fails open"
        );
    }

    #[test]
    fn core_tools_ride_on_every_prompt_including_misses() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selector = KeywordSelector::default().with_core_tools(["read_file".to_string()]);
        let selected = selector.select(&ctx("completely unrelated zebra topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "web_search"],
            "a total keyword miss fails open to the full surface — the core set \
             composes with fail-open, never replaces it: a vocabulary miss must \
             not strand the tool the prompt actually needs"
        );
    }

    #[test]
    fn the_widest_synonym_pair_does_not_pull_clearly_unrelated_tools() {
        let all = vec![
            schema("weather", "Report the weather forecast for a city"),
            schema("list_files", "List files in a directory"),
        ];
        let selected = KeywordSelector::default().select(&ctx("show me the weather"), &all);
        assert_eq!(
            names(&selected),
            vec!["weather"],
            "show widens to read but never to list — the precision guard every \
             future pair addition must argue against"
        );
    }

    #[test]
    fn a_partial_wrong_match_cannot_evict_a_core_tool() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector = KeywordSelector::default().with_core_tools(["read_file".to_string()]);
        let selected = selector.select(&ctx("what is the weather in Rome"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "weather"],
            "the keyword match and the core set union — a strong wrong-side match \
             cannot evict the tool every turn may need"
        );
    }

    #[test]
    fn host_synonyms_match_case_insensitively() {
        let all = vec![
            schema("query_tool", "Query the dataset"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector =
            KeywordSelector::default().with_synonyms([("SQL".to_string(), "query".to_string())]);
        let selected = selector.select(&ctx("sql time"), &all);
        assert_eq!(
            names(&selected),
            vec!["query_tool"],
            "pair sides normalize on store — a host pair spelled in caps still \
             expands the tokenizer's lowercase query token"
        );
    }

    #[test]
    fn look_at_selects_read_tools_without_pulling_search() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look at the config"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file"],
            "the phrase table discriminates: look at is a read intent, and the \
             word-level look pair must not drag search tools into it"
        );
    }

    #[test]
    fn look_for_keeps_the_search_intent() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look for the bug"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "look for is a search intent, carried by the phrase table after the \
             word pair's removal"
        );
    }

    #[test]
    fn every_default_pair_side_survives_tokenization() {
        for (left, right) in DEFAULT_SYNONYM_PAIRS {
            for side in [left, right] {
                let tokens = tokenize(side);
                assert_eq!(
                    tokens,
                    vec![side.to_string()],
                    "every curated pair side must be a single token the tokenizer \
                     keeps (>= 3 chars, non-stopword) or the pair can never fire: \
                     '{side}'"
                );
                assert!(
                    side.chars().all(|ch| ch.is_ascii_alphanumeric()),
                    "pair sides stay ASCII alphanumeric: '{side}'"
                );
            }
        }
    }

    #[test]
    fn every_default_phrase_target_survives_tokenization() {
        for (phrase, target) in DEFAULT_PHRASE_SYNONYMS {
            assert!(
                phrase.contains(' '),
                "phrase keys are multi-word — a single word belongs in the pair \
                 table: '{phrase}'"
            );
            assert!(
                phrase
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch == ' '),
                "phrase keys are lowercase, matching the lowered query: '{phrase}'"
            );
            let tokens = tokenize(target);
            assert_eq!(
                tokens,
                vec![target.to_string()],
                "every phrase target must survive tokenization — it joins the \
                 expanded set: '{target}'"
            );
        }
    }

    #[test]
    fn expansion_keeps_selection_deterministic() {
        let all = vec![
            schema("verify_build", "Check the build for regressions"),
            schema("web_search", "Search the web for pages"),
        ];
        let context = ctx("check the weather and verify the build");
        assert_eq!(
            names(&KeywordSelector::default().select(&context, &all)),
            names(&KeywordSelector::default().select(&context, &all)),
            "expansion is a pure function of the query — equal contexts select \
             equal surfaces"
        );
    }

    #[test]
    fn the_tokenizer_drops_filler_and_dedupes() {
        assert_eq!(
            tokenize("Read a FILE, read it now!"),
            vec!["read", "file", "now"],
            "lowercase runs of 3+ chars, first appearance only"
        );
    }

    #[test]
    fn glue_words_never_match() {
        assert!(
            tokenize("search the web for rust news with the api")
                .iter()
                .all(|token| !matches!(token.as_str(), "the" | "for" | "with")),
            "a prompt's glue words must not advertise every tool whose \
             description says 'the'"
        );
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    #[cfg(feature = "testing")]
    use std::sync::Arc;
    #[cfg(feature = "vector_index")]
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn schema(name: &str, description: &str) -> ToolSchema {
        ToolSchema::new(name, description, serde_json::json!({"type": "object"}))
    }

    /// A selection-recording async double advertising a fixed subset.
    struct FixedAsyncSelector {
        chosen: Vec<String>,
        turns: std::sync::Mutex<Vec<usize>>,
    }

    impl AsyncToolSelector for FixedAsyncSelector {
        fn select<'a>(
            &'a self,
            ctx: &'a SelectionContext<'a>,
            all: &'a [ToolSchema],
        ) -> Pin<Box<dyn Future<Output = Vec<ToolSchema>> + Send + 'a>> {
            Box::pin(async move {
                let mut turns = self
                    .turns
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                turns.push(ctx.turn);
                drop(turns);
                all.iter()
                    .filter(|schema| self.chosen.contains(&schema.tool))
                    .cloned()
                    .collect()
            })
        }
    }

    #[tokio::test]
    async fn an_async_double_selects_through_the_companion_trait() {
        let all = vec![
            schema("web_search", "Search the web for pages"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector = FixedAsyncSelector {
            chosen: vec!["weather".to_string()],
            turns: std::sync::Mutex::new(Vec::new()),
        };
        let selected = selector
            .select(
                &SelectionContext {
                    turn: 0,
                    latest_user_message: Some("anything"),
                    conversation: &[],
                },
                &all,
            )
            .await;
        assert_eq!(
            selected.iter().map(|s| s.tool.as_str()).collect::<Vec<_>>(),
            vec!["weather"],
            "the companion trait carries selection through a future"
        );
        let turns = selector
            .turns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(*turns, vec![0], "the context the selector saw");
    }

    #[cfg(feature = "vector_index")]
    mod embedding {
        use super::*;
        use crate::memory::vector::{Embedding, EmbeddingProvider};

        /// A table-driven provider: exact-text vectors, a call
        /// counter, and an optional poison word that fails any text
        /// containing it.
        struct TableProvider {
            table: Vec<(String, Vec<f32>)>,
            calls: AtomicUsize,
            poison: Option<&'static str>,
        }

        impl EmbeddingProvider for TableProvider {
            fn dim(&self) -> usize {
                2
            }

            fn embed<'a>(
                &'a self,
                text: &'a str,
            ) -> Pin<Box<dyn Future<Output = Result<Embedding, crate::error::LoopError>> + Send + 'a>>
            {
                let _ = self.calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if self.poison.is_some_and(|word| text.contains(word)) {
                        return Err(crate::error::LoopError::Internal("poisoned".into()));
                    }
                    let vector = self
                        .table
                        .iter()
                        .find(|(known, _)| known == text)
                        .map(|(_, vector)| vector.clone())
                        .unwrap_or_default();
                    Ok(Embedding::new(vector))
                })
            }
        }

        fn fixture() -> Vec<ToolSchema> {
            vec![
                schema("weather", "Report the weather forecast for a city"),
                schema("web_search", "Search the web for pages"),
            ]
        }

        fn table() -> Vec<(String, Vec<f32>)> {
            vec![
                (
                    "weather Report the weather forecast for a city".to_string(),
                    vec![1.0, 0.0],
                ),
                (
                    "web_search Search the web for pages".to_string(),
                    vec![0.0, 1.0],
                ),
                ("find today's forecast".to_string(), vec![0.95, 0.05]),
            ]
        }

        async fn select_with(
            provider: &TableProvider,
            query: &str,
            all: &[ToolSchema],
        ) -> Vec<ToolSchema> {
            EmbeddingSelector::new(TableProvider {
                table: provider.table.clone(),
                calls: AtomicUsize::new(0),
                poison: provider.poison,
            })
            .with_top_k(1)
            .select(
                &SelectionContext {
                    turn: 0,
                    latest_user_message: Some(query),
                    conversation: &[],
                },
                all,
            )
            .await
        }

        #[tokio::test]
        async fn semantic_proximity_selects_the_nearest_tool() {
            let provider = TableProvider {
                table: table(),
                calls: AtomicUsize::new(0),
                poison: None,
            };
            let selected = select_with(&provider, "find today's forecast", &fixture()).await;
            assert_eq!(
                selected.iter().map(|s| s.tool.as_str()).collect::<Vec<_>>(),
                vec!["weather"],
                "the query lands near the forecast tool with no shared vocabulary"
            );
        }

        #[tokio::test]
        async fn a_zero_vector_query_fails_open_to_the_full_surface() {
            let mut entries = table();
            entries.push(("empty headed query".to_string(), vec![0.0, 0.0]));
            let provider = TableProvider {
                table: entries,
                calls: AtomicUsize::new(0),
                poison: None,
            };
            let all = fixture();
            let selected = select_with(&provider, "empty headed query", &all).await;
            assert_eq!(
                selected.len(),
                all.len(),
                "an unrankable query (every cosine NaN) advertises everything — \
                 an inability to rank is not an intentional empty selection"
            );
        }

        #[tokio::test]
        async fn a_query_embed_failure_fails_open() {
            let provider = TableProvider {
                table: table(),
                calls: AtomicUsize::new(0),
                poison: Some("forecast"),
            };
            let all = fixture();
            let selected = select_with(&provider, "find today's forecast", &all).await;
            assert_eq!(
                selected.len(),
                all.len(),
                "a failed embed advertises everything"
            );
        }

        #[tokio::test]
        async fn an_overwritten_description_re_embeds_the_corpus() {
            let all = fixture();
            let selector = EmbeddingSelector::new(TableProvider {
                table: table(),
                calls: AtomicUsize::new(0),
                poison: None,
            });
            let ctx = SelectionContext {
                turn: 0,
                latest_user_message: Some("find today's forecast"),
                conversation: &[],
            };
            selector.select(&ctx, &all).await;
            let overwritten = vec![
                schema("weather", "Report the precipitation outlook for a city"),
                schema("web_search", "Search the web for pages"),
            ];
            selector.select(&ctx, &overwritten).await;
            assert_eq!(
                selector.provider.calls.load(Ordering::SeqCst),
                6,
                "a same-name/different-description overwrite is a new generation — \
                 the corpus re-embeds (3 + 3: two corpus batches, one query each) \
                 so the semantic position follows the tool's current text"
            );
        }

        #[tokio::test]
        async fn the_corpus_embeds_once_per_registry_generation() {
            let all = fixture();
            let selector = EmbeddingSelector::new(TableProvider {
                table: table(),
                calls: AtomicUsize::new(0),
                poison: None,
            });
            let ctx = SelectionContext {
                turn: 0,
                latest_user_message: Some("find today's forecast"),
                conversation: &[],
            };
            selector.select(&ctx, &all).await;
            selector.select(&ctx, &all).await;
            assert_eq!(
                selector.provider.calls.load(Ordering::SeqCst),
                4,
                "two corpus texts once + one query per selection — the corpus \
                 served from cache on the second turn"
            );
            let mutated = vec![schema("weather", "Report the weather forecast for a city")];
            selector.select(&ctx, &mutated).await;
            assert_eq!(
                selector.provider.calls.load(Ordering::SeqCst),
                6,
                "a membership change re-embeds the corpus"
            );
        }
    }

    #[cfg(feature = "testing")]
    mod rewriting {
        use super::*;

        /// A client that never answers: every non-streaming call
        /// outlives any test timeout.
        struct NeverAnsweringClient;

        impl crate::api::ApiClient for NeverAnsweringClient {
            fn model(&self) -> String {
                "small-model".to_string()
            }

            fn set_model(&self, _model: &str) -> bool {
                false
            }

            fn stream_messages(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>,
                        > + Send
                        + 'static,
                >,
            > {
                Box::pin(futures::stream::empty())
            }

            fn create_message(
                &self,
                _request: &crate::api::StreamRequest,
            ) -> Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                crate::api::NonStreamingResponse,
                                crate::api::error::ApiError,
                            >,
                        > + Send
                        + '_,
                >,
            > {
                Box::pin(async {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    Err(crate::api::error::ApiError::api("the double never answers"))
                })
            }
        }

        #[tokio::test]
        async fn the_rewrite_prompt_carries_names_never_schemas() {
            let client = Arc::new(crate::testing::MockApiClient::new("small-model"));
            drop((*client).clone().with_text_response("web_search"));
            let all = vec![
                schema(
                    "web_search",
                    "Search the web for pages. Results arrive as links.",
                ),
                schema("weather", "Report the weather forecast for a city"),
            ];
            let selector = RewritingSelector::new(Arc::clone(&client));
            let selected = selector
                .select(
                    &SelectionContext {
                        turn: 0,
                        latest_user_message: Some("look up rust news"),
                        conversation: &[],
                    },
                    &all,
                )
                .await;
            assert_eq!(
                selected.iter().map(|s| s.tool.as_str()).collect::<Vec<_>>(),
                vec!["web_search"],
                "the rewrite's answer names the selection"
            );
            let captured = client.captured_requests();
            assert_eq!(captured.len(), 1, "exactly one rewrite call");
            let prompt = captured
                .first()
                .and_then(|request| request.messages.first())
                .and_then(|message| message.parts.first())
                .and_then(|part| match part {
                    crate::message::MessagePart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .expect("the prompt rides as text");
            assert!(prompt.contains("web_search"), "names ride: {prompt}");
            assert!(
                prompt.contains("Search the web for pages"),
                "one-line summaries ride: {prompt}"
            );
            assert!(
                !prompt.contains("input_schema") && !prompt.contains("properties"),
                "never the schemas — that would spend the tokens selection saves: {prompt}"
            );
            assert!(prompt.contains("look up rust news"), "the request rides");
        }

        #[tokio::test]
        async fn an_answer_naming_no_known_tool_fails_open() {
            let client = Arc::new(crate::testing::MockApiClient::new("small-model"));
            drop((*client).clone().with_text_response("banana"));
            let all = fixture_like();
            let selector = RewritingSelector::new(client);
            let selected = selector
                .select(
                    &SelectionContext {
                        turn: 0,
                        latest_user_message: Some("anything"),
                        conversation: &[],
                    },
                    &all,
                )
                .await;
            assert_eq!(
                selected.len(),
                all.len(),
                "garbage output advertises the full surface"
            );
        }

        #[tokio::test]
        async fn an_exhausted_budget_refuses_the_call() {
            let client = Arc::new(crate::testing::MockApiClient::new("small-model"));
            drop((*client).clone().with_text_response("web_search"));
            let all = fixture_like();
            let selector = RewritingSelector::new(Arc::clone(&client)).with_budget(
                crate::budget::BudgetLimits {
                    tokens: Some(0),
                    turns: None,
                    duration: None,
                    usd: None,
                },
            );
            let selected = selector
                .select(
                    &SelectionContext {
                        turn: 0,
                        latest_user_message: Some("anything"),
                        conversation: &[],
                    },
                    &all,
                )
                .await;
            assert_eq!(selected.len(), all.len(), "a refused rewrite fails open");
            assert_eq!(
                client.create_message_calls(),
                0,
                "the budget gate refused before the call"
            );
        }

        #[tokio::test]
        async fn a_duration_budget_refuses_once_the_clock_runs() {
            let client = Arc::new(crate::testing::MockApiClient::new("small-model"));
            drop((*client).clone().with_text_response("web_search"));
            let all = fixture_like();
            let selector = RewritingSelector::new(Arc::clone(&client)).with_budget(
                crate::budget::BudgetLimits {
                    tokens: None,
                    turns: None,
                    duration: Some(std::time::Duration::from_millis(1)),
                    usd: None,
                },
            );
            let ctx = SelectionContext {
                turn: 0,
                latest_user_message: Some("anything"),
                conversation: &[],
            };
            let first = selector.select(&ctx, &all).await;
            assert_eq!(
                first.iter().map(|s| s.tool.as_str()).collect::<Vec<_>>(),
                vec!["web_search"],
                "the first call sits at zero elapsed and proceeds"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let second = selector.select(&ctx, &all).await;
            assert_eq!(
                second.len(),
                all.len(),
                "the duration line refuses the second call and fails open"
            );
            assert_eq!(
                client.create_message_calls(),
                1,
                "exactly one rewrite call — the clock refused the rest"
            );
        }

        #[tokio::test]
        async fn the_rewrite_prompt_bounds_each_conversation_excerpt() {
            let client = Arc::new(crate::testing::MockApiClient::new("small-model"));
            drop((*client).clone().with_text_response("web_search"));
            let flood = format!("{}\n", "enormous tool result body ".repeat(600));
            let message = crate::message::Message::user(flood.clone());
            let conversation = vec![message];
            let all = fixture_like();
            let selector = RewritingSelector::new(Arc::clone(&client));
            selector
                .select(
                    &SelectionContext {
                        turn: 0,
                        latest_user_message: Some("look up rust news"),
                        conversation: &conversation,
                    },
                    &all,
                )
                .await;
            let prompt = client
                .captured_requests()
                .first()
                .and_then(|request| request.messages.first())
                .and_then(|message| message.parts.first())
                .and_then(|part| match part {
                    crate::message::MessagePart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .expect("the prompt rides as text");
            assert!(
                prompt.contains(&flood[..64]),
                "the excerpt keeps the message's head"
            );
            assert!(
                prompt.len() < 4 * 1024,
                "a pathological message cannot flood the rewrite prompt: {} bytes",
                prompt.len()
            );
        }

        fn fixture_like() -> Vec<ToolSchema> {
            vec![
                schema("web_search", "Search the web for pages"),
                schema("weather", "Report the weather forecast for a city"),
            ]
        }

        #[tokio::test]
        async fn a_timed_out_rewrite_fails_open() {
            let all = fixture_like();
            let selector = RewritingSelector::new(Arc::new(NeverAnsweringClient))
                .with_timeout(std::time::Duration::from_millis(5));
            let selected = selector
                .select(
                    &SelectionContext {
                        turn: 0,
                        latest_user_message: Some("anything"),
                        conversation: &[],
                    },
                    &all,
                )
                .await;
            assert_eq!(
                selected.len(),
                all.len(),
                "a rewrite that cannot answer in time fails open"
            );
        }
    }
}
