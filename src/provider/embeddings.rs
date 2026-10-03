//! Real [`EmbeddingProvider`](crate::memory::vector::EmbeddingProvider) backends: Ollama's native embedding endpoint
//! and OpenAI's embeddings API.
//!
//! Two embedders live here, both implementing the
//! [`EmbeddingProvider`](crate::memory::vector::EmbeddingProvider) trait
//! unchanged so they drop straight into
//! `VectorMemoryStore`:
//!
//! - [`OllamaEmbedder`] — local embeddings through Ollama's **native**
//!   `/api/embed` endpoint (the OpenAI-compatible surface Ollama exposes
//!   does not serve embeddings, so this is its own wire shape, not another
//!   `OpenAiClient` profile). Default model `nomic-embed-text`, 768
//!   dimensions, no API key, no network beyond the local server — the
//!   zero-cost, offline, private path.
//! - [`OpenAiEmbedder`] — the `/v1/embeddings` endpoint
//!   (`text-embedding-3-small`, 1536 dimensions by default, Matryoshka
//!   truncation configurable down to 256), the frontier-grade path and the
//!   default for live validation.
//!
//! Both share the provider family's HTTP discipline: validated builders,
//! `from_env()` profiles, a sensitive `Authorization` header (never in
//! `Debug` output), bounded response bodies, and typed [`ApiError`]
//! failures. Batches split at 256 inputs per request, and every batch
//! settles one `embed.batch` metric event.
//!
//! # Embedding models are not interchangeable
//!
//! A store built with one embedder cannot be queried with another: the
//! dimensions differ *and* the vector spaces are unrelated — the same text
//! maps to unrelated geometry under a different model. Keep one embedder
//! (and one model) for a store's lifetime; the embedders enforce the
//! dimension half of that rule by failing loudly when a server returns a
//! vector whose length differs from the configured `dim()`.
//!
//! # Example
//!
//! ```rust,no_run
//! # async fn demo() -> Result<(), loopctl::error::LoopError> {
//! use loopctl::memory::vector::EmbeddingProvider;
//! use loopctl::provider::embeddings::OpenAiEmbedder;
//!
//! let embedder = OpenAiEmbedder::from_env()
//!     .map_err(|error| loopctl::error::LoopError::Api(error.to_string()))?;
//! let embedding = embedder.embed("prefer glob over manual search").await?;
//! assert_eq!(embedding.dim(), 1536);
//! # Ok(())
//! # }
//! ```

use std::time::Instant;

use crate::api::error::ApiError;
use crate::memory::vector::Embedding;

/// The default OpenAI embedding model.
///
/// `text-embedding-3-small`: 1536 dimensions, the cheap general-purpose
/// member of the 3-series, and the model whose geometry the default
/// `dim()` assumes.
const OPENAI_DEFAULT_MODEL: &str = "text-embedding-3-small";

/// The dimension `text-embedding-3-small` returns by default.
///
/// Builders that override the model without overriding `dimensions` keep
/// this `dim()` and therefore fail the first response whose length differs
/// — the loud failure is the honest behavior, not a limitation.
const OPENAI_DEFAULT_DIM: usize = 1536;

/// The smallest Matryoshka truncation the 3-series accepts.
///
/// OpenAI's `dimensions` request parameter truncates the model's output
/// to any length from this floor up to the model's own maximum — 1536 on
/// `text-embedding-3-small`, 3072 on `text-embedding-3-large`; shorter
/// vectors lose recall but shrink index memory proportionally.
const OPENAI_MIN_DIMENSIONS: usize = 256;

/// The largest Matryoshka truncation the 3-series accepts.
///
/// The parameter's series-wide bounds are enforced here at build time;
/// each model's own maximum still applies server-side, where exceeding
/// it is the server's loud 400 — never a silent resize. Pairing
/// `text-embedding-3-large` with `with_dimensions(3072)` matches its
/// full-length output exactly.
const OPENAI_MAX_DIMENSIONS: usize = 3072;

/// OpenAI's per-input token ceiling.
///
/// The API rejects any single input over 8191 tokens; the embedder guards
/// client-side so the failure names the offending input index instead of
/// surfacing as an opaque server 400 after the request was already paid
/// for.
const OPENAI_MAX_INPUT_TOKENS: usize = 8191;

/// The chars-per-token estimate the input guard uses.
///
/// A guard, not a bill: four characters per token is the classic English
/// heuristic, accurate enough to catch pathological inputs while letting
/// borderline ones through to the server's exact counter.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// The character budget one input must fit within.
///
/// `8191 tokens × 4 chars/token`; an input longer than this is rejected
/// before any HTTP traffic, naming the input's index.
const OPENAI_MAX_INPUT_CHARS: usize = OPENAI_MAX_INPUT_TOKENS * CHARS_PER_TOKEN_ESTIMATE;

/// Inputs per embedding request, shared by both providers.
///
/// Sized by response-body arithmetic: 256 inputs × 1536 components × ~11
/// bytes per JSON float ≈ 4.3 MB worst case, which fits the
/// dimension-derived response budget with headroom while keeping
/// per-request latency
/// bounded. OpenAI's own per-request cap is 2048 inputs — the tighter
/// client-side split also keeps a batch's response inside the bounded-body
/// budget where one 2048-input request would need ~34 MB.
const MAX_BATCH_INPUTS: usize = 256;

/// The per-float byte estimate the response budget is derived from.
///
/// A JSON float renders at roughly 11–12 bytes (digits, sign, point,
/// separator); 16 is the conservative ceiling, so a derived budget never
/// under-sizes a legitimate answer.
const RESPONSE_BYTES_PER_FLOAT: usize = 16;

/// The headroom multiplier over the worst-case legitimate batch.
///
/// Doubles the arithmetic estimate: a full batch of maximal floats must
/// clear the ceiling with room, while hostile streams stay bounded.
const RESPONSE_BUDGET_HEADROOM: usize = 2;

/// The floor of a derived response budget.
///
/// Degenerate dimensions (a one-component test model) would otherwise
/// derive a budget tighter than the JSON envelope around even a modest
/// batch; one mebibyte keeps every legitimate answer inside.
const MIN_RESPONSE_BUDGET_BYTES: usize = 1024 * 1024;

/// The ceiling of a derived response budget.
///
/// Binds hostile streams even when a caller configures an absurd
/// dimension; no legitimate batch needs more.
const MAX_RESPONSE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// The response-body budget for one embedding request at `dim` components.
///
/// Computed as `dim × MAX_BATCH_INPUTS × RESPONSE_BYTES_PER_FLOAT ×
/// RESPONSE_BUDGET_HEADROOM` with saturating arithmetic, then clamped to
/// [`MIN_RESPONSE_BUDGET_BYTES`]`..=`[`MAX_RESPONSE_BUDGET_BYTES`]: a
/// legitimate batch cannot brush the ceiling in any regime — a 4096-dim
/// model's full 256-input batch (~12 MB on the wire) sits well inside its
/// ~33.5 MB budget — while a hostile or misbehaving server cannot stream
/// an unbounded body into memory whatever the configured geometry.
fn response_budget(dim: usize) -> usize {
    dim.saturating_mul(MAX_BATCH_INPUTS)
        .saturating_mul(RESPONSE_BYTES_PER_FLOAT)
        .saturating_mul(RESPONSE_BUDGET_HEADROOM)
        .clamp(MIN_RESPONSE_BUDGET_BYTES, MAX_RESPONSE_BUDGET_BYTES)
}

/// The default Ollama server root.
///
/// The native embedding route hangs off the server root
/// (`{base}/api/embed`), unlike the chat profile's `/v1`-prefixed surface
/// — which is why the two profiles keep different defaults for the same
/// `OLLAMA_BASE_URL` variable.
#[cfg(feature = "ollama")]
const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// The default Ollama embedding model.
///
/// `nomic-embed-text` is Ollama's canonical text-embedding model: 768
/// dimensions, small enough to run anywhere, and the model whose geometry
/// the default `dim()` assumes.
#[cfg(feature = "ollama")]
const OLLAMA_DEFAULT_MODEL: &str = "nomic-embed-text";
/// The dimension `nomic-embed-text` returns.
///
/// Ollama models have fixed output dimensions, so the builder carries this
/// as a plain number; a server returning a different length fails the
/// dimension check with both numbers named.
#[cfg(feature = "ollama")]
const OLLAMA_DEFAULT_DIM: usize = 768;

/// The default OpenAI API root.
///
/// Matches the chat client's default so one `OPENAI_BASE_URL` override
/// moves both clients to the same compatible server.
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// One usage report OpenAI attaches to an embeddings response.
///
/// Kept as plain numbers and re-emitted on the batch metric event; absent
/// on Ollama, whose native endpoint reports no usage.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct OpenAiEmbedUsage {
    /// The billed input tokens of the request.
    ///
    /// Embedding requests bill input only, so this is the cost-bearing
    /// figure of the pair.
    prompt_tokens: u64,

    /// The API's total-token figure, equal to input for embeddings.
    ///
    /// Reported because the wire carries it and telemetry consumers
    /// expect the pair together.
    total_tokens: u64,
}

/// Read a successful response body under the [`MAX_RESPONSE_BYTES`] budget.
///
/// The content-length header is consulted first when present (an early,
/// cheap refusal), then the stream is accumulated chunk by chunk so a
/// lying or absent header cannot bypass the cap. The budget is the
/// caller's dimension-derived [`response_budget`] — see its doc for the
/// sizing rule.
///
/// # Errors
///
/// Returns [`ApiError::api`] when the body exceeds the budget, and
/// [`ApiError::http`] when the stream itself fails mid-read.
async fn read_body_bounded(
    response: reqwest::Response,
    budget: usize,
) -> Result<Vec<u8>, ApiError> {
    let declared_budget = u64::try_from(budget).unwrap_or(u64::MAX);
    if let Some(declared) = response.content_length()
        && declared > declared_budget
    {
        return Err(ApiError::api(format!(
            "embedding response declares {declared} bytes, over the {budget}-byte \
             bounded-body budget"
        )));
    }
    let mut response = response;
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| ApiError::http(error.to_string()))?
    {
        if body.len().saturating_add(chunk.len()) > budget {
            return Err(ApiError::api(format!(
                "embedding response exceeded the {budget}-byte bounded-body budget \
                 mid-stream"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// L2-normalize a vector in place, leaving a zero vector untouched.
///
/// Neither API normalizes by default;
/// [`cosine_similarity`](crate::memory::vector::cosine_similarity) handles
/// unnormalized inputs, but normalizing once at the source makes every
/// downstream dot product a cosine. The accumulator is scaled by the
/// largest absolute component first — mirroring
/// [`cosine_similarity`](crate::memory::vector::cosine_similarity)'s
/// documented robustness — so huge-but-finite components normalize
/// exactly instead of overflowing the squared sum to a silent all-zero
/// vector. A zero-norm vector (empty input, or an all-zero server
/// response) passes through unchanged rather than dividing by zero.
///
/// # Errors
///
/// Returns [`ApiError::api`] naming the first non-finite component's
/// position — JSON cannot carry such a value onto this path, so seeing
/// one means the vector did not come from a parsed response, and the
/// failure is loud rather than a quiet zero vector.
fn l2_normalize_in_place(vector: &mut [f32]) -> Result<(), ApiError> {
    let mut scale = 0.0_f32;
    for (position, component) in vector.iter().enumerate() {
        if !component.is_finite() {
            return Err(ApiError::api(format!(
                "embedding vector carries a non-finite component at position {position} \
                 — the answer is not usable"
            )));
        }
        scale = scale.max(component.abs());
    }
    if scale == 0.0 {
        return Ok(());
    }
    let mut squared = 0.0_f32;
    for component in vector.iter() {
        let scaled = component / scale;
        squared += scaled * scaled;
    }
    let root = squared.sqrt();
    for value in vector.iter_mut() {
        *value = (*value / scale) / root;
    }
    Ok(())
}

/// Verify a returned vector's length against the configured dimension.
///
/// The dimension check is the honest half of "models are not
/// interchangeable": a server whose model was swapped under the index
/// returns a different length, and a silent resize here would corrupt
/// every index built on the provider. The mismatch emits a `WARN` metric
/// event (it is the near-miss worth alerting on) before failing with both
/// numbers named.
///
/// # Errors
///
/// Returns [`ApiError::config_validation`] naming the expected and
/// received lengths.
fn enforce_dimension(
    provider: &str,
    model: &str,
    expected: usize,
    received: usize,
) -> Result<(), ApiError> {
    if expected == received {
        return Ok(());
    }
    tracing::warn!(
        target: "loopctl::metrics",
        provider,
        model,
        expected,
        received,
        "embedding dimension mismatch — the server's model may have been swapped under the index"
    );
    Err(ApiError::config_validation(format!(
        "{provider} embedding dimension mismatch: model {model} returned {received} \
         components, the embedder is configured for {expected}"
    )))
}

/// Emit the `embed.batch` metric event for one settled batch.
///
/// One event per HTTP batch on both the success and failure paths, with
/// the usage pair recorded when the provider reports one — the intake
/// rate, the local-vs-remote latency split, and the reliability signal in
/// a single stream.
fn emit_batch_metric(
    provider: &str,
    model: &str,
    dim: usize,
    inputs: usize,
    outcome: &str,
    usage: Option<OpenAiEmbedUsage>,
    started: Instant,
) {
    tracing::debug!(
        target: "loopctl::metrics",
        span = "embed.batch",
        provider,
        model,
        dim,
        inputs,
        outcome,
        prompt_tokens = usage.map(|report| report.prompt_tokens),
        total_tokens = usage.map(|report| report.total_tokens),
        duration_ms = %started.elapsed().as_millis(),
        "embedding batch settled"
    );
}

/// Normalize a chat-style `OLLAMA_BASE_URL` value to the server root.
///
/// The chat profile seeds the same variable with a `/v1`-suffixed default
/// and appends OpenAI-style paths to it; the native embedding route hangs
/// off the root instead, so a trailing `/v1` (and any trailing slashes) is
/// trimmed before the endpoint is formed. A root-style value passes
/// through untouched.
#[cfg(feature = "ollama")]
fn ollama_root_base(raw: &str) -> String {
    let trimmed = raw.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

/// The request body of one `/api/embed` call.
///
/// Borrowed fields keep the request allocation-free until serialization;
/// Ollama answers with `embeddings` in input order, so no index field
/// exists to model.
#[cfg(feature = "ollama")]
#[derive(serde::Serialize)]
struct OllamaEmbedRequest<'a> {
    /// The model whose embedding geometry the inputs map into.
    ///
    /// Names the server-side model verbatim so the answer's vectors come
    /// from exactly the geometry the caller pinned.
    model: &'a str,

    /// The batch of texts to embed, preserved verbatim and in order.
    ///
    /// Ollama answers with one vector per entry of this array, in array
    /// order — the contract the response parsing relies on.
    input: &'a [&'a str],
}

/// The response body of one `/api/embed` call.
///
/// Ollama guarantees `embeddings` parallels the request's `input` array;
/// unknown fields — including the model echo — are ignored, because the
/// API adds them without notice.
#[cfg(feature = "ollama")]
#[derive(serde::Deserialize)]
struct OllamaEmbedResponse {
    /// One vector per request input, in request order.
    ///
    /// The length must equal the request's input count and every vector's
    /// length must equal the configured dimension — both are verified
    /// before the vectors are returned.
    embeddings: Vec<Vec<f32>>,
}

/// Local embeddings through Ollama's native `/api/embed` endpoint.
///
/// The zero-cost path: a local server, no API key, and
/// `nomic-embed-text`'s 768-dimensional geometry by default. The endpoint
/// is the *native* one — the OpenAI-compatible surface Ollama also exposes
/// does not serve embeddings — and the legacy `/api/embeddings` (singular
/// input) is deliberately not used: it batches nothing and upstream
/// deprecates it.
///
/// Batches split at `MAX_BATCH_INPUTS` inputs per request; a transport
/// failure carries a "is `ollama serve` running?" hint because the
/// overwhelmingly common cause is a server that is not up.
///
/// # Example
///
/// ```rust,no_run
/// use loopctl::provider::embeddings::OllamaEmbedder;
///
/// let embedder = OllamaEmbedder::builder().build()?;
/// # let _ = embedder;
/// # Ok::<(), loopctl::api::error::ApiError>(())
/// ```
#[cfg(feature = "ollama")]
#[derive(Debug, Clone)]
pub struct OllamaEmbedder {
    /// The connection-pooled HTTP client, built once with the configured
    /// timeouts.
    ///
    /// Reused across batches so a multi-batch call keeps its connections
    /// warm.
    http: reqwest::Client,

    /// The server root the `/api/embed` endpoint hangs off.
    ///
    /// Kept as the root (no `/v1`); `ollama_root_base` normalizes
    /// chat-style values so the two profiles can share `OLLAMA_BASE_URL`.
    base_url: String,

    /// The embedding model every request names.
    ///
    /// Fixed at build time: swapping models mid-store corrupts every
    /// comparison, so a swap means a new embedder and a new store.
    model: String,

    /// The dimension this embedder reports and every returned vector must
    /// carry.
    ///
    /// Ollama fixes dimensions per model; a server answer of any other
    /// length fails the `enforce_dimension` check with both numbers
    /// named rather than silently resizing.
    dim: usize,

    /// Whether returned vectors are L2-normalized in place.
    ///
    /// Off by default (the cosine helper normalizes either way); on, it
    /// makes every downstream dot product a cosine at the source.
    normalized: bool,
}

#[cfg(feature = "ollama")]
impl OllamaEmbedder {
    /// The dimension every embedding this embedder returns carries.
    ///
    /// Fixed at build time to the model's geometry; inherent so callers
    /// and the trait impl read one source of truth.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Create a builder with production-ready defaults.
    ///
    /// Local server root, `nomic-embed-text`, 768 dimensions,
    /// normalization off — the one decision left to the caller is whether
    /// to override any of it.
    #[must_use]
    pub fn builder() -> OllamaEmbedderBuilder {
        OllamaEmbedderBuilder::default()
    }

    /// Build from the Ollama environment.
    ///
    /// Reads `OLLAMA_EMBEDDING_MODEL` (**required** — deliberately not
    /// `OLLAMA_MODEL`, which names the chat model; conflating them
    /// silently embeds with a chat model) and `OLLAMA_BASE_URL`
    /// (optional, default `http://localhost:11434`; a chat-style `/v1`
    /// suffix is trimmed). The dimension defaults to `nomic-embed-text`'s
    /// 768 — pair a different-geometry model with the builder's
    /// [`with_dim`](OllamaEmbedderBuilder::with_dim).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::config`] when `OLLAMA_EMBEDDING_MODEL` is not
    /// set, and the builder's own validation otherwise.
    pub fn from_env() -> Result<Self, ApiError> {
        let model = std::env::var("OLLAMA_EMBEDDING_MODEL").map_err(|_| {
            ApiError::config(
                "OLLAMA_EMBEDDING_MODEL is not set — the embedder deliberately refuses \
                 OLLAMA_MODEL, which names the chat model; embedding with a chat model \
                 silently degrades retrieval",
            )
        })?;
        let base_url = std::env::var("OLLAMA_BASE_URL").map_or_else(
            |_| OLLAMA_DEFAULT_BASE_URL.to_string(),
            |raw| ollama_root_base(&raw),
        );
        Self::builder()
            .with_model(model)
            .with_base_url(base_url)
            .build()
    }

    /// The full `/api/embed` endpoint URL.
    ///
    /// Formed once per batch from the stored root; the builder validated
    /// its shape at construction.
    fn endpoint(&self) -> String {
        format!("{}/api/embed", self.base_url)
    }

    /// Embed `texts`, splitting into bounded batches and concatenating
    /// the results in input order.
    ///
    /// An empty slice returns an empty vector without touching the wire.
    /// Each batch settles one `embed.batch` metric event, success or
    /// failure; a failed batch aborts the call with its error (partial
    /// results are never returned — a caller cannot mistake them for a
    /// complete embedding of the input set).
    ///
    /// # Errors
    ///
    /// Returns the classified [`ApiError`] of the first failing batch —
    /// transport failures carry the "is `ollama serve` running?" hint.
    pub async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Embedding>, ApiError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH_INPUTS) {
            let started = Instant::now();
            match self.embed_batch_once(batch).await {
                Ok(mut embedded) => {
                    emit_batch_metric(
                        "ollama",
                        &self.model,
                        self.dim,
                        batch.len(),
                        "ok",
                        None,
                        started,
                    );
                    embeddings.append(&mut embedded);
                }
                Err(error) => {
                    emit_batch_metric(
                        "ollama",
                        &self.model,
                        self.dim,
                        batch.len(),
                        "error",
                        None,
                        started,
                    );
                    return Err(error);
                }
            }
        }
        Ok(embeddings)
    }

    /// Run exactly one `/api/embed` request and validate its answer.
    ///
    /// The wire call, the bounded read, the count check, and the
    /// per-vector dimension check all live here so the batching layer
    /// above stays pure orchestration.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::http`] with the "is `ollama serve` running?"
    /// hint appended on transport failure, the classified status error
    /// otherwise, and [`ApiError::api`] when the response's embedding
    /// count does not match the input count.
    async fn embed_batch_once(&self, batch: &[&str]) -> Result<Vec<Embedding>, ApiError> {
        let body = OllamaEmbedRequest {
            model: &self.model,
            input: batch,
        };
        let response = super::post_json_checked(
            &self.http,
            &self.endpoint(),
            &[],
            &serde_json::to_value(body)?,
        )
        .await
        .map_err(|error| match error {
            ApiError::Http(message) => ApiError::http(format!(
                "{message} — no embedding answer from {}; is `ollama serve` running?",
                self.endpoint()
            )),
            other => other,
        })?;
        let bytes = read_body_bounded(response, response_budget(self.dim)).await?;
        let parsed: OllamaEmbedResponse = serde_json::from_slice(&bytes)?;
        if parsed.embeddings.len() != batch.len() {
            return Err(ApiError::api(format!(
                "ollama returned {} embeddings for {} inputs — the /api/embed batch \
                 contract is one vector per input in order",
                parsed.embeddings.len(),
                batch.len()
            )));
        }
        let mut embeddings = Vec::with_capacity(parsed.embeddings.len());
        for mut vector in parsed.embeddings {
            enforce_dimension("ollama", &self.model, self.dim, vector.len())?;
            if self.normalized {
                l2_normalize_in_place(&mut vector)?;
            }
            embeddings.push(Embedding::new(vector));
        }
        Ok(embeddings)
    }
}

/// The builder for [`OllamaEmbedder`].
///
/// Everything has the local-profile default; the builder exists to pin a
/// non-`nomic-embed-text` model with its true
/// [`with_dim`](Self::with_dim), point at a non-local server, and carry
/// the family's shared HTTP knobs.
#[cfg(feature = "ollama")]
#[derive(Default)]
pub struct OllamaEmbedderBuilder {
    /// The shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// The same embedded struct every provider builder carries, so pool
    /// and timeout semantics cannot drift between the chat and embedding
    /// clients.
    http: super::HttpClientConfig,

    /// The server root, if explicitly set.
    ///
    /// `None` means `OLLAMA_DEFAULT_BASE_URL` at build time.
    base_url: Option<String>,

    /// The embedding model, if explicitly set.
    ///
    /// `None` builds with `OLLAMA_DEFAULT_MODEL` — the local profile's
    /// identity *is* the model choice, and `nomic-embed-text` is it.
    model: Option<String>,

    /// The dimension every returned vector must carry.
    ///
    /// Defaults to `OLLAMA_DEFAULT_DIM`; override when the chosen model
    /// embeds into a different geometry.
    dim: Option<usize>,

    /// Whether returned vectors are L2-normalized in place.
    ///
    /// Set through [`normalized`](Self::normalized); the stored flag is
    /// read once per batch.
    normalized: bool,
}

#[cfg(feature = "ollama")]
impl OllamaEmbedderBuilder {
    /// Point the embedder at `base_url` (the server root, no `/v1`).
    ///
    /// The value is used verbatim after the shared chat-style `/v1`
    /// trimming rule applies to environment reads — explicit builder
    /// values are trusted as the root.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Pin the embedding model.
    ///
    /// The model choice is the store's identity: every vector the store
    /// ever holds comes from this geometry.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the dimension the embedder reports and enforces.
    ///
    /// Must match the model's true output length; a mismatch surfaces on
    /// the first embed as a loud error naming both numbers.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = Some(dim);
        self
    }

    /// L2-normalize every returned vector in place.
    ///
    /// Off by default; on, it costs one pass per vector and makes every
    /// downstream dot product a cosine.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.normalized = true;
        self
    }

    /// Set the HTTP read timeout.
    ///
    /// Delegates to the shared `HttpClientConfig`
    /// every provider builder embeds.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http = self.http.with_timeout(timeout);
        self
    }

    /// Set the TCP connection establishment timeout.
    ///
    /// Delegates to the shared `HttpClientConfig`.
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http = self.http.with_connect_timeout(timeout);
        self
    }

    /// Inject a pre-built, shared HTTP client.
    ///
    /// Pool and timeout knobs are then the injected client's — the same
    /// sharing contract the chat clients offer.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = self.http.with_http_client(client);
        self
    }

    /// Validate and build the embedder.
    ///
    /// Checks the model is present and non-blank, the dimension is at
    /// least 1, and that the base URL forms a parseable `/api/embed`
    /// endpoint — all as [`ApiError::config_validation`] failures, so a
    /// misconfiguration dies at build time rather than mid-run.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::config_validation`] for each invalid field.
    pub fn build(self) -> Result<OllamaEmbedder, ApiError> {
        let model = self
            .model
            .unwrap_or_else(|| OLLAMA_DEFAULT_MODEL.to_string());
        if model.trim().is_empty() {
            return Err(ApiError::config_validation(
                "ollama embedder: model must not be empty",
            ));
        }
        let dim = self.dim.unwrap_or(OLLAMA_DEFAULT_DIM);
        if dim == 0 {
            return Err(ApiError::config_validation(
                "ollama embedder: dim must be at least 1",
            ));
        }
        let base_url = self
            .base_url
            .unwrap_or_else(|| OLLAMA_DEFAULT_BASE_URL.to_string());
        let endpoint = format!("{base_url}/api/embed");
        reqwest::Url::parse(&endpoint).map_err(|error| {
            ApiError::config_validation(format!(
                "ollama embedder: base URL {base_url} does not form a valid /api/embed \
                 endpoint: {error}"
            ))
        })?;
        Ok(OllamaEmbedder {
            http: self.http.build()?,
            base_url,
            model,
            dim,
            normalized: self.normalized,
        })
    }
}

/// The request body of one `/v1/embeddings` call.
///
/// `dimensions` rides only when set — the 3-series' Matryoshka
/// truncation — and is omitted entirely otherwise so non-3-series models
/// never see a parameter they would reject.
#[derive(serde::Serialize)]
struct OpenAiEmbedRequest<'a> {
    /// The model whose embedding geometry the inputs map into.
    ///
    /// Names the server-side model verbatim so the answer's vectors come
    /// from exactly the geometry the caller pinned.
    model: &'a str,

    /// The batch of texts to embed, preserved verbatim and in order.
    ///
    /// The response carries one datum per entry, keyed by the input
    /// position it answers.
    input: &'a [&'a str],

    /// The Matryoshka truncation length, when requested.
    ///
    /// Serialized only when set, so models without the parameter never
    /// see a field they would reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// One entry of the `/v1/embeddings` response's `data` array.
///
/// `index` names the input position the embedding answers; the array's
/// order is not guaranteed, so reassembly keys on this field.
#[derive(serde::Deserialize)]
struct OpenAiEmbedDatum {
    /// The input position this embedding answers.
    ///
    /// The wire does not guarantee array order, so this field — not
    /// position — drives reassembly.
    index: usize,

    /// The embedding vector for that input.
    ///
    /// Its length must equal the configured dimension; a mismatch is
    /// the loud model-swap failure, never a silent resize.
    embedding: Vec<f32>,
}

/// The response body of one `/v1/embeddings` call.
///
/// `data` is reassembled by [`OpenAiEmbedDatum::index`]; the model echo
/// and any other unknown fields are ignored.
#[derive(serde::Deserialize)]
struct OpenAiEmbedResponse {
    /// One datum per request input, in unspecified order.
    ///
    /// Sorted by each datum's index and checked for count and
    /// permutation before the vectors are returned.
    data: Vec<OpenAiEmbedDatum>,

    /// The request's usage report, when the server sends one.
    ///
    /// Re-emitted on the batch metric event so cost telemetry survives
    /// the trip without a second request.
    #[serde(default)]
    usage: Option<OpenAiEmbedUsage>,
}

/// The bearer `Authorization` header for `api_key`, marked sensitive.
///
/// The sensitivity marker keeps the credential out of debug output and
/// HTTP/2 header indexing, matching the family contract of
/// [`post_json_checked`](super::post_json_checked) callers.
///
/// # Errors
///
/// Returns [`ApiError::auth_invalid_key`] when the key cannot form a
/// header value (non-ASCII or control characters).
fn bearer_header(api_key: &str) -> Result<reqwest::header::HeaderValue, ApiError> {
    let mut bearer = reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
        .map_err(|error| ApiError::auth_invalid_key(format!("invalid bearer token: {error}")))?;
    bearer.set_sensitive(true);
    Ok(bearer)
}

/// Remote embeddings through OpenAI's `/v1/embeddings` endpoint.
///
/// The frontier-grade path and the default for live validation:
/// `text-embedding-3-small` (1536 dimensions) whose quality is a
/// constant, so retrieval-quality measurements built on it measure the
/// *store*, not the embedder. The 3-series' Matryoshka truncation is
/// exposed as [`with_dimensions`](OpenAiEmbedderBuilder::with_dimensions)
/// (`256..=1536`), and any OpenAI-compatible embeddings server can be
/// targeted through the base-URL override with honest naming.
///
/// Batches split at `MAX_BATCH_INPUTS` inputs per request, every input
/// is guarded against OpenAI's per-input token ceiling before any wire
/// traffic, and the response's `data` array is reassembled by its `index`
/// field — never by position, because the wire does not guarantee order.
///
/// # Example
///
/// ```rust,no_run
/// use loopctl::provider::embeddings::OpenAiEmbedder;
///
/// let embedder = OpenAiEmbedder::from_env()?;
/// # let _ = embedder;
/// # Ok::<(), loopctl::api::error::ApiError>(())
/// ```
#[derive(Clone)]
pub struct OpenAiEmbedder {
    /// The connection-pooled HTTP client, built once with the configured
    /// timeouts.
    ///
    /// Reused across batches so a multi-batch call keeps its connections
    /// warm.
    http: reqwest::Client,

    /// The API key sent as the sensitive `Authorization: Bearer` header.
    ///
    /// Never rendered by the manual [`Debug`](std::fmt::Debug) impl —
    /// the same redaction discipline the Bedrock client applies to its
    /// credentials.
    api_key: String,

    /// The API root the `/embeddings` endpoint hangs off.
    ///
    /// Defaults to `https://api.openai.com/v1`; an override targets any
    /// OpenAI-compatible embeddings server.
    base_url: String,

    /// The embedding model every request names.
    ///
    /// Fixed at build time: swapping models mid-store corrupts every
    /// comparison, so a swap means a new embedder and a new store.
    model: String,

    /// The dimension this embedder reports and every returned vector must
    /// carry.
    ///
    /// Equals the Matryoshka truncation when one is set, and the model's
    /// native length otherwise.
    dim: usize,

    /// The Matryoshka truncation length sent as the `dimensions` request
    /// parameter, when set.
    ///
    /// `None` sends no parameter — non-3-series models reject it.
    dimensions: Option<usize>,

    /// Whether returned vectors are L2-normalized in place.
    ///
    /// Off by default; on, it makes every downstream dot product a
    /// cosine at the source.
    normalized: bool,
}

impl std::fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("dimensions", &self.dimensions)
            .field("normalized", &self.normalized)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl OpenAiEmbedder {
    /// The dimension every embedding this embedder returns carries.
    ///
    /// The Matryoshka truncation when one is set, the model's native
    /// length otherwise; inherent so callers and the trait impl read one
    /// source of truth.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Create a builder with production-ready defaults.
    ///
    /// `text-embedding-3-small`, 1536 dimensions, the public API root,
    /// normalization off — the caller's one required decision is the API
    /// key.
    #[must_use]
    pub fn builder() -> OpenAiEmbedderBuilder {
        OpenAiEmbedderBuilder::default()
    }

    /// Build from the OpenAI environment.
    ///
    /// Reads `OPENAI_API_KEY` (**required**), `OPENAI_BASE_URL`
    /// (optional, default `https://api.openai.com/v1`), and
    /// `OPENAI_EMBEDDING_MODEL` (optional, default
    /// `text-embedding-3-small`). The dimension follows the model default
    /// — `text-embedding-3-large` and other geometries pair with the
    /// builder's
    /// [`with_dimensions`](OpenAiEmbedderBuilder::with_dimensions),
    /// anywhere in its enforced `256..=3072` range (3072 matching
    /// 3-large's full-length output).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::auth_invalid_key`] when `OPENAI_API_KEY` is
    /// not set, and the builder's own validation otherwise.
    pub fn from_env() -> Result<Self, ApiError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ApiError::auth_invalid_key("OPENAI_API_KEY not set"))?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| OPENAI_DEFAULT_BASE_URL.to_string());
        let model = std::env::var("OPENAI_EMBEDDING_MODEL")
            .unwrap_or_else(|_| OPENAI_DEFAULT_MODEL.to_string());
        Self::builder()
            .with_api_key(api_key)
            .with_base_url(base_url)
            .with_model(model)
            .build()
    }

    /// The full `/embeddings` endpoint URL.
    ///
    /// Formed once per batch from the stored root; the builder validated
    /// its shape at construction.
    fn endpoint(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }

    /// Reject any input over OpenAI's per-input token ceiling.
    ///
    /// The chars/4 estimate is a guard, not a bill — it catches
    /// pathological inputs before the wire so the failure names the
    /// offending input index instead of surfacing as an opaque server 400
    /// the request was already sent for.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::config_validation`] naming the first
    /// over-budget input's index and its character count.
    fn guard_input_sizes(texts: &[&str]) -> Result<(), ApiError> {
        for (index, text) in texts.iter().enumerate() {
            let chars = text.chars().count();
            if chars > OPENAI_MAX_INPUT_CHARS {
                let estimated_tokens = chars / CHARS_PER_TOKEN_ESTIMATE;
                return Err(ApiError::config_validation(format!(
                    "openai embedder: input {index} is {chars} chars \
                     (≈{estimated_tokens} tokens), over the {OPENAI_MAX_INPUT_TOKENS}-token \
                     per-input ceiling — split or truncate the input before embedding"
                )));
            }
        }
        Ok(())
    }

    /// Embed `texts`, splitting into bounded batches and concatenating
    /// the results in input order.
    ///
    /// An empty slice returns an empty vector without touching the wire;
    /// every input passes the token guard before any HTTP traffic. Each
    /// batch settles one `embed.batch` metric event — carrying the
    /// request's usage pair when the server reports one — and a failed
    /// batch aborts the call (partial results are never returned).
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::config_validation`] for an over-budget input,
    /// otherwise the classified [`ApiError`] of the first failing batch.
    pub async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Embedding>, ApiError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        Self::guard_input_sizes(texts)?;
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH_INPUTS) {
            let started = Instant::now();
            match self.embed_batch_once(batch).await {
                Ok((mut embedded, usage)) => {
                    emit_batch_metric(
                        "openai",
                        &self.model,
                        self.dim,
                        batch.len(),
                        "ok",
                        usage,
                        started,
                    );
                    embeddings.append(&mut embedded);
                }
                Err(error) => {
                    emit_batch_metric(
                        "openai",
                        &self.model,
                        self.dim,
                        batch.len(),
                        "error",
                        None,
                        started,
                    );
                    return Err(error);
                }
            }
        }
        Ok(embeddings)
    }

    /// Run exactly one `/v1/embeddings` request and reassemble its answer
    /// by `index`.
    ///
    /// The wire call, the bounded read, the index-permutation check, the
    /// per-vector dimension check, and the usage capture live here so the
    /// batching layer above stays pure orchestration.
    ///
    /// # Errors
    ///
    /// Returns the classified [`ApiError`] of the wire call or parse, and
    /// [`ApiError::api`] when the `data` indices are not a permutation of
    /// the input positions.
    async fn embed_batch_once(
        &self,
        batch: &[&str],
    ) -> Result<(Vec<Embedding>, Option<OpenAiEmbedUsage>), ApiError> {
        let body = OpenAiEmbedRequest {
            model: &self.model,
            input: batch,
            dimensions: self.dimensions,
        };
        let authorization = bearer_header(&self.api_key)?;
        let headers = [(reqwest::header::AUTHORIZATION, authorization)];
        let response = super::post_json_checked(
            &self.http,
            &self.endpoint(),
            &headers,
            &serde_json::to_value(body)?,
        )
        .await?;
        let bytes = read_body_bounded(response, response_budget(self.dim)).await?;
        let parsed: OpenAiEmbedResponse = serde_json::from_slice(&bytes)?;
        let usage = parsed.usage;
        let mut data = parsed.data;
        data.sort_unstable_by_key(|datum| datum.index);
        if data.len() != batch.len()
            || data
                .iter()
                .enumerate()
                .any(|(position, datum)| datum.index != position)
        {
            let indices: Vec<usize> = data.iter().map(|datum| datum.index).collect();
            return Err(ApiError::api(format!(
                "openai embedder: response carries {} data rows with indices {indices:?} \
                 for {} inputs — reassembly by index refuses to guess",
                data.len(),
                batch.len()
            )));
        }
        let mut embeddings = Vec::with_capacity(data.len());
        for mut datum in data {
            enforce_dimension("openai", &self.model, self.dim, datum.embedding.len())?;
            if self.normalized {
                l2_normalize_in_place(&mut datum.embedding)?;
            }
            embeddings.push(Embedding::new(datum.embedding));
        }
        Ok((embeddings, usage))
    }
}

/// The builder for [`OpenAiEmbedder`].
///
/// The API key is the one required field; the model, dimension, and
/// endpoint default to `text-embedding-3-small`'s facts, and the shared
/// HTTP knobs ride the same embedded configuration the chat builders
/// carry.
pub struct OpenAiEmbedderBuilder {
    /// The API key, required before building.
    ///
    /// Sent as the sensitive `Authorization: Bearer` header on every
    /// request and never rendered by `Debug`.
    api_key: Option<String>,

    /// The API root, if explicitly set.
    ///
    /// `None` means `OPENAI_DEFAULT_BASE_URL` at build time.
    base_url: Option<String>,

    /// The embedding model, if explicitly set.
    ///
    /// `None` builds with `OPENAI_DEFAULT_MODEL`.
    model: Option<String>,

    /// The Matryoshka truncation length, when requested.
    ///
    /// Validated into `256..=1536` at build time; also fixes the
    /// embedder's reported dimension.
    dimensions: Option<usize>,

    /// Whether returned vectors are L2-normalized in place.
    ///
    /// Set through [`normalized`](OpenAiEmbedderBuilder::normalized);
    /// the stored flag is read once per batch.
    normalized: bool,

    /// The shared HTTP client configuration (timeouts, pool, TCP).
    ///
    /// The same embedded struct every provider builder carries.
    http: super::HttpClientConfig,
}

impl Default for OpenAiEmbedderBuilder {
    /// Returns the frontier-profile defaults.
    ///
    /// Everything empty except the HTTP configuration's production-ready
    /// timeouts — the caller fills the API key and optionally pins the
    /// rest.
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            model: None,
            dimensions: None,
            normalized: false,
            http: super::HttpClientConfig::default(),
        }
    }
}

impl OpenAiEmbedderBuilder {
    /// Set the API key.
    ///
    /// Sent as the sensitive bearer header; required before
    /// [`build`](Self::build).
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Point the embedder at `base_url` (the API root, `/v1` included).
    ///
    /// The value is used verbatim, exactly like the chat client's
    /// override — one `OPENAI_BASE_URL` moves both.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Pin the embedding model.
    ///
    /// The model choice is the store's identity: every vector the store
    /// ever holds comes from this geometry.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Request Matryoshka truncation to `dimensions` components.
    ///
    /// Validated into `256..=3072` (the 3-series parameter's series-wide
    /// bounds) and sent as the `dimensions` request field; the embedder's
    /// reported dimension becomes this value. Each model's own maximum
    /// still applies server-side — `text-embedding-3-small` caps at 1536,
    /// and exceeding a model's cap is the server's loud 400, never a
    /// silent resize. Non-3-series models reject the parameter — pair it
    /// only with models that support it.
    #[must_use]
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// L2-normalize every returned vector in place.
    ///
    /// Off by default; on, it costs one pass per vector and makes every
    /// downstream dot product a cosine.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.normalized = true;
        self
    }

    /// Set the HTTP read timeout.
    ///
    /// Delegates to the shared `HttpClientConfig`
    /// every provider builder embeds.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http = self.http.with_timeout(timeout);
        self
    }

    /// Set the TCP connection establishment timeout.
    ///
    /// Delegates to the shared `HttpClientConfig`.
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http = self.http.with_connect_timeout(timeout);
        self
    }

    /// Inject a pre-built, shared HTTP client.
    ///
    /// Pool and timeout knobs are then the injected client's — the same
    /// sharing contract the chat clients offer.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = self.http.with_http_client(client);
        self
    }

    /// Validate and build the embedder.
    ///
    /// Checks the API key is present, the model is non-blank, the
    /// truncation (when set) is inside `256..=3072`, and that the base
    /// URL forms a parseable `/embeddings` endpoint — all as
    /// [`ApiError`] failures, so a misconfiguration dies at build time
    /// rather than mid-run.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::auth_invalid_key`] for a missing key and
    /// [`ApiError::config_validation`] for each other invalid field.
    pub fn build(self) -> Result<OpenAiEmbedder, ApiError> {
        let api_key = self.api_key.ok_or_else(|| {
            ApiError::auth_invalid_key(
                "openai embedder: api_key is not set — set it with with_api_key or \
                 OPENAI_API_KEY",
            )
        })?;
        let model = self
            .model
            .unwrap_or_else(|| OPENAI_DEFAULT_MODEL.to_string());
        if model.trim().is_empty() {
            return Err(ApiError::config_validation(
                "openai embedder: model must not be empty",
            ));
        }
        if let Some(dimensions) = self.dimensions
            && !(OPENAI_MIN_DIMENSIONS..=OPENAI_MAX_DIMENSIONS).contains(&dimensions)
        {
            return Err(ApiError::config_validation(format!(
                "openai embedder: dimensions {dimensions} is outside the supported \
                 {OPENAI_MIN_DIMENSIONS}..={OPENAI_MAX_DIMENSIONS} truncation range"
            )));
        }
        let base_url = self
            .base_url
            .unwrap_or_else(|| OPENAI_DEFAULT_BASE_URL.to_string());
        let endpoint = format!("{base_url}/embeddings");
        reqwest::Url::parse(&endpoint).map_err(|error| {
            ApiError::config_validation(format!(
                "openai embedder: base URL {base_url} does not form a valid /embeddings \
                 endpoint: {error}"
            ))
        })?;
        let dim = self.dimensions.unwrap_or(OPENAI_DEFAULT_DIM);
        Ok(OpenAiEmbedder {
            http: self.http.build()?,
            api_key,
            base_url,
            model,
            dim,
            dimensions: self.dimensions,
            normalized: self.normalized,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::arithmetic_side_effects,
        clippy::float_cmp
    )]

    use super::{
        OPENAI_DEFAULT_MODEL, OPENAI_MAX_INPUT_CHARS, OpenAiEmbedResponse, OpenAiEmbedder,
        bearer_header, l2_normalize_in_place,
    };

    #[test]
    #[cfg(feature = "ollama")]
    fn chat_style_base_urls_normalize_to_the_server_root() {
        assert_eq!(
            super::ollama_root_base("http://localhost:11434/v1"),
            "http://localhost:11434",
            "a chat-style /v1 suffix must be trimmed to the server root"
        );
        assert_eq!(
            super::ollama_root_base("http://gpu-box:11434/v1/"),
            "http://gpu-box:11434",
            "a trailing slash before the suffix is trimmed with it"
        );
        assert_eq!(
            super::ollama_root_base("http://localhost:11434"),
            "http://localhost:11434",
            "an already-root value passes through untouched"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn the_ollama_response_parser_ignores_fields_the_wire_may_add() {
        let body = r#"{"model":"nomic-embed-text","embeddings":[[0.1,0.2],[0.3,0.4]],
                       "future_field":{"nested":true}}"#;
        let parsed: super::OllamaEmbedResponse = serde_json::from_str(body).unwrap();
        assert_eq!(
            parsed.embeddings.len(),
            2,
            "one vector per input survives an unknown sibling field"
        );
        assert_eq!(
            parsed.embeddings.first().unwrap().len(),
            2,
            "component values parse as f32 without loss of shape"
        );
    }

    #[test]
    fn the_openai_response_parser_accepts_a_missing_usage_and_model() {
        let body = r#"{"data":[{"index":1,"embedding":[0.5]},{"index":0,"embedding":[0.25]}]}"#;
        let parsed: OpenAiEmbedResponse = serde_json::from_str(body).unwrap();
        assert_eq!(
            parsed.data.len(),
            2,
            "a minimal response with neither echo nor usage still parses"
        );
        assert!(
            parsed.usage.is_none(),
            "usage is optional on the wire and None when absent"
        );
    }

    #[test]
    fn the_openai_builder_rejects_out_of_range_dimensions() {
        for dimensions in [255_usize, 3073] {
            let error = OpenAiEmbedder::builder()
                .with_api_key("sk-test")
                .with_dimensions(dimensions)
                .build()
                .unwrap_err();
            assert!(
                matches!(error, crate::api::error::ApiError::Config(_)),
                "dimensions {dimensions} must fail config validation: {error:?}"
            );
        }
        for dimensions in [256_usize, 1536, 1537, 2048, 3072] {
            let embedder = OpenAiEmbedder::builder()
                .with_api_key("sk-test")
                .with_dimensions(dimensions)
                .build()
                .unwrap_or_else(|error| panic!("dimensions {dimensions} must build: {error}"));
            assert_eq!(
                embedder.dim, dimensions,
                "a legal truncation reports itself as the dimension"
            );
        }
    }

    #[test]
    fn the_response_budget_scales_with_dimension_and_clamps() {
        assert_eq!(
            super::response_budget(1536),
            1536 * 256 * 16 * 2,
            "the default geometry derives twelve mebibytes — double its worst-case batch"
        );
        assert_eq!(
            super::response_budget(4096),
            4096 * 256 * 16 * 2,
            "a 4096-dim model derives ~33.5 MB — its ~12 MB full batch fits with room"
        );
        assert_eq!(
            super::response_budget(1),
            super::MIN_RESPONSE_BUDGET_BYTES,
            "a degenerate dimension clamps up to the floor"
        );
        assert_eq!(
            super::response_budget(10_000_000),
            super::MAX_RESPONSE_BUDGET_BYTES,
            "an absurd dimension clamps down to the ceiling — hostile streams stay bounded"
        );
    }

    #[test]
    fn the_openai_builder_defaults_match_the_frontier_profile() {
        let embedder = OpenAiEmbedder::builder()
            .with_api_key("sk-test")
            .build()
            .unwrap();
        assert_eq!(embedder.model, OPENAI_DEFAULT_MODEL, "model defaults");
        assert_eq!(embedder.dim, 1536, "dimension defaults");
        assert_eq!(
            embedder.base_url, "https://api.openai.com/v1",
            "root defaults"
        );
        let truncated = OpenAiEmbedder::builder()
            .with_api_key("sk-test")
            .with_dimensions(512)
            .build()
            .unwrap();
        assert_eq!(
            truncated.dim, 512,
            "a truncation fixes the reported dimension"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn the_ollama_builder_defaults_match_the_local_profile() {
        let embedder = super::OllamaEmbedder::builder().build().unwrap();
        assert_eq!(
            embedder.model,
            super::OLLAMA_DEFAULT_MODEL,
            "model defaults"
        );
        assert_eq!(embedder.dim, 768, "dimension defaults");
        assert_eq!(
            embedder.base_url, "http://localhost:11434",
            "the server root defaults without a /v1 suffix"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn the_ollama_builder_rejects_invalid_fields() {
        let empty_model = super::OllamaEmbedder::builder()
            .with_model("   ")
            .build()
            .unwrap_err();
        assert!(
            matches!(empty_model, crate::api::error::ApiError::Config(_)),
            "a blank model must fail validation: {empty_model:?}"
        );
        let zero_dim = super::OllamaEmbedder::builder()
            .with_dim(0)
            .build()
            .unwrap_err();
        assert!(
            matches!(zero_dim, crate::api::error::ApiError::Config(_)),
            "a zero dimension must fail validation: {zero_dim:?}"
        );
    }

    #[test]
    fn debug_never_shows_the_api_key() {
        let openai = OpenAiEmbedder::builder()
            .with_api_key("sk-live-do-not-leak")
            .build()
            .unwrap();
        let rendered = format!("{openai:?}");
        assert!(
            !rendered.contains("sk-live-do-not-leak"),
            "the OpenAI embedder's Debug must never contain the key: {rendered}"
        );
        assert!(
            rendered.contains("<redacted>"),
            "the key's slot renders as an explicit redaction marker: {rendered}"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn the_local_embedder_debug_carries_no_credential() {
        let ollama = format!("{:?}", super::OllamaEmbedder::builder().build().unwrap());
        assert!(
            !ollama.contains("sk-"),
            "the local embedder carries no credential at all: {ollama}"
        );
    }

    #[test]
    fn the_bearer_header_is_marked_sensitive() {
        let header = bearer_header("sk-test").unwrap();
        assert!(
            header.is_sensitive(),
            "the Authorization value must carry the sensitivity marker"
        );
        assert_eq!(
            header.to_str().unwrap(),
            "Bearer sk-test",
            "the header renders the bearer scheme with the key"
        );
    }

    #[test]
    fn oversized_inputs_fail_the_guard_with_their_index_named() {
        let oversized = "x".repeat(OPENAI_MAX_INPUT_CHARS.saturating_add(1));
        let inputs = ["fine", oversized.as_str()];
        let error = OpenAiEmbedder::guard_input_sizes(&inputs).unwrap_err();
        assert!(
            matches!(error, crate::api::error::ApiError::Config(_)),
            "an over-budget input is a config-validation failure: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("input 1"),
            "the failure names the offending input's index: {message}"
        );
    }

    #[test]
    fn the_token_guard_counts_characters_not_bytes() {
        let wide = "é".repeat(OPENAI_MAX_INPUT_CHARS.saturating_sub(1));
        assert!(
            wide.len() > OPENAI_MAX_INPUT_CHARS,
            "precondition: the fixture's byte length exceeds its char count"
        );
        OpenAiEmbedder::guard_input_sizes(&[wide.as_str()])
            .expect("a char-legal input passes the guard regardless of byte length");
    }

    #[test]
    fn normalization_produces_unit_vectors_and_passes_zero_through() {
        let mut vector = vec![3.0_f32, 4.0];
        l2_normalize_in_place(&mut vector).unwrap();
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "a 3-4 vector normalizes to unit length, got norm {norm}"
        );
        let mut zero = vec![0.0_f32, 0.0];
        l2_normalize_in_place(&mut zero).unwrap();
        assert_eq!(
            zero,
            vec![0.0, 0.0],
            "a zero vector passes through unchanged rather than dividing by zero"
        );
    }

    #[test]
    fn normalization_rescales_huge_but_finite_components_exactly() {
        let mut vector = [3.0_f32, 4.0]
            .iter()
            .map(|v| v * 1.0e20)
            .collect::<Vec<_>>();
        l2_normalize_in_place(&mut vector).unwrap();
        assert!(
            (vector[0] - 0.6).abs() < 1e-6 && (vector[1] - 0.8).abs() < 1e-6,
            "max-abs rescaling must keep the direction of components whose naive \
             squared sum would overflow, got {vector:?}"
        );
    }

    #[test]
    fn a_non_finite_component_fails_normalization_loudly() {
        for garbage in [f32::NAN, f32::INFINITY] {
            let mut vector = vec![1.0_f32, garbage];
            let error = l2_normalize_in_place(&mut vector).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("non-finite") && message.contains("position 1"),
                "the failure must name the problem and the component's position: {message}"
            );
        }
    }

    #[cfg(all(feature = "ollama", feature = "testing"))]
    #[test]
    fn ollama_from_env_requires_the_embedding_model_var() {
        let env = crate::testing::EnvGuard::acquire(&["OLLAMA_EMBEDDING_MODEL", "OLLAMA_BASE_URL"]);
        env.remove("OLLAMA_EMBEDDING_MODEL");
        env.remove("OLLAMA_BASE_URL");
        let error = super::OllamaEmbedder::from_env().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("OLLAMA_EMBEDDING_MODEL is not set"),
            "the failure names the required variable: {error}"
        );
        env.set("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large");
        env.set("OLLAMA_BASE_URL", "http://gpu-box:11434/v1");
        let embedder = super::OllamaEmbedder::from_env().unwrap();
        assert_eq!(
            embedder.model, "mxbai-embed-large",
            "the embedding model var pins the model"
        );
        assert_eq!(
            embedder.base_url, "http://gpu-box:11434",
            "a chat-style base URL is normalized to the server root"
        );
    }

    #[cfg(feature = "testing")]
    #[test]
    fn openai_from_env_requires_the_key_and_defaults_the_rest() {
        let env = crate::testing::EnvGuard::acquire(&[
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_EMBEDDING_MODEL",
        ]);
        env.remove("OPENAI_API_KEY");
        env.remove("OPENAI_BASE_URL");
        env.remove("OPENAI_EMBEDDING_MODEL");
        let error = OpenAiEmbedder::from_env().unwrap_err();
        assert!(
            error.to_string().contains("OPENAI_API_KEY"),
            "the failure names the required variable: {error}"
        );
        env.set("OPENAI_API_KEY", "sk-test");
        let embedder = OpenAiEmbedder::from_env().unwrap();
        assert_eq!(
            embedder.model, OPENAI_DEFAULT_MODEL,
            "the model defaults without the env var"
        );
        assert_eq!(
            embedder.base_url, "https://api.openai.com/v1",
            "the API root defaults without the env var"
        );
        env.set("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large");
        env.set("OPENAI_BASE_URL", "https://proxy.example/v1");
        let overridden = OpenAiEmbedder::from_env().unwrap();
        assert_eq!(
            overridden.model, "text-embedding-3-large",
            "the env var overrides the model default"
        );
        assert_eq!(
            overridden.base_url, "https://proxy.example/v1",
            "the env var overrides the API root"
        );
    }
}
