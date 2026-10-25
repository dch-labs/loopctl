//! Model discovery and liveness probes over the provider wire.
//!
//! Two questions a caller has about a configured provider before it
//! spends a completion on it: *which models are there* and *is it
//! alive*. This module owns the value types the [`ApiClient`] surface
//! answers them with — [`ModelInfo`] listings, [`ModelMetadata`]
//! capability records, [`PingReport`] liveness evidence — and the one
//! failure vocabulary, [`ProbeClass`], that every probe shares so a
//! health-check consumer can branch on *why* a probe failed instead of
//! string-matching error text.
//!
//! The trait-level defaults and per-provider overrides of
//! [`list_models`](crate::api::ApiClient::list_models),
//! [`ping`](crate::api::ApiClient::ping) and
//! [`model_metadata`](crate::api::ApiClient::model_metadata) live on [`ApiClient`]
//! itself; this module is deliberately transport-free so the types (and
//! the curated metadata table) stay usable — and testable — without a
//! client or a socket.
//!
//! [`ApiClient`]: crate::api::ApiClient

use serde::{Deserialize, Serialize};

use crate::api::error::{ApiError, ErrorCode, RateLimitSnapshot, http_status_is_overload};

/// A content type a model can accept as input or emit as output.
///
/// Providers spell modalities in their own wire dialects (`"TEXT"`,
/// `"IMAGE"`, …); this enum is the crate's normalized spelling. A
/// [`ModelMetadata`] field holding a modality list is only populated
/// when the provider (or the curated table) actually stated one —
/// `None` means unreported, not "text only".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Modality {
    /// Plain text.
    ///
    /// The universal modality: every chat model in the crate's table
    /// accepts text input and emits text output. Wire spellings reach it
    /// through [`Modality::from_wire_token`] case-insensitively.
    Text,

    /// Still images.
    ///
    /// Covers image input (photo understanding) and image output where a
    /// provider lists it; a model that only *emits* images would carry
    /// `Image` in its output list alone, which is why input and output
    /// are separate [`ModelMetadata`] fields.
    Image,

    /// Audio.
    ///
    /// Speech input or output as the provider discloses it. Mostly a
    /// Gemini-family inhabitant of listings today; other vendors spell
    /// audio capability in prose the parsers do not read.
    Audio,

    /// Video.
    ///
    /// Video input, disclosed almost exclusively by the Gemini family's
    /// listing payload. There is no video output anywhere in the curated
    /// table; the variant exists because the wire does.
    Video,

    /// Embeddings output.
    ///
    /// Marks a model that emits vectors rather than completions. A
    /// listing entry carrying this in its output modalities is a signal
    /// to route the id to an embeddings call, not a chat turn — sending
    /// an embedding model a chat request fails with an unhelpful 4xx on
    /// every provider that ships both kinds.
    Embedding,
}

impl Modality {
    /// Parse a provider's wire spelling of a modality.
    ///
    /// Case-insensitive, trimmed; unknown or absent spellings return
    /// [`None`] rather than guessing, so a parser written against a new
    /// provider payload degrades to "unreported" instead of a wrong
    /// modality.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::probe::Modality;
    /// assert_eq!(Modality::from_wire_token("TEXT"), Some(Modality::Text));
    /// assert_eq!(Modality::from_wire_token(" image "), Some(Modality::Image));
    /// assert_eq!(Modality::from_wire_token("smell"), None);
    /// ```
    #[must_use]
    pub fn from_wire_token(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "text" => Some(Self::Text),
            "image" => Some(Self::Image),
            "audio" => Some(Self::Audio),
            "video" => Some(Self::Video),
            "embedding" | "embeddings" | "embeddingvector" => Some(Self::Embedding),
            _ => None,
        }
    }
}

/// Where the facts in a [`ModelMetadata`] came from.
///
/// The honesty marker of the metadata surface: a number the provider
/// itself served is a different kind of fact from a curated opinion,
/// and a consumer shipping on one (billing headroom against a
/// disclosed window, for instance) needs to know which it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MetadataSource {
    /// Nothing was reported by anyone; every field is `None`.
    ///
    /// The type's default, and the state a record holds until some part
    /// of it is actually stated. Every construction path that could
    /// contradict the marker closes it: the [`with_*` builders](ModelMetadata)
    /// promote a record out of this variant the moment they attach a
    /// fact, and deserialization applies the same promotion to a document
    /// that states a fact while omitting `source` — `Unreported` beside a
    /// populated field is a state no public path constructs.
    #[default]
    Unreported,

    /// Only the crate's curated table contributed.
    ///
    /// The model was not disclosed by the wire; every populated field
    /// came from a published, documented row in
    /// [`curated_model_metadata`]. A consumer that wants wire facts only
    /// treats this source as no answer at all — the marker exists to
    /// make that reading one field access, not a table lookup.
    Curated,

    /// Only the provider's own payload contributed.
    ///
    /// A disclosed context window, a listing entry's
    /// `inputTokenLimit`, Bedrock's `inputModalities` — facts the wire
    /// itself carried. Attaching a fact through a
    /// [`ModelMetadata`] builder lands here too: the builders are the
    /// "the caller is stating this from the endpoint's answer" surface,
    /// and they never claim curation.
    Disclosed,

    /// The provider disclosed some facts and the curated table filled
    /// the rest.
    ///
    /// Produced by [`ModelMetadata::merged`] when curation genuinely
    /// filled a gap beside a disclosure; a record that needed nothing
    /// from the table keeps [`Disclosed`](Self::Disclosed) so the
    /// marker never over-credits the table.
    DisclosedAndCurated,
}

/// The capability half of [`ModelMetadata`]: what a model can be *asked
/// to do*, as reported by provider documentation or its listing payload.
///
/// Every field is an `Option`: providers do not publish machine-readable
/// capability matrices, and a half-answer ("supports tools, structured
/// output unknown") is the honest shape. A `Some(false)` is a real
/// claim — the model was checked or documented not to support it — as
/// distinct from `None` ("nobody said").
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelCapabilities {
    /// Native structured output: a real JSON-Schema-constrained
    /// response mode (OpenAI `json_schema`-style, Gemini
    /// `responseSchema`).
    ///
    /// `Some(false)` deliberately covers the models that offer only a
    /// schema-less JSON mode (`json_object` spelling only): to a caller
    /// deciding whether a schema will be *honoured*, "schema-less JSON
    /// only" answers "no native schema mode", and the corrective-retry
    /// path treats it exactly like no support at all.
    pub structured_output: Option<bool>,

    /// Native tool / function calling on this client's wire.
    ///
    /// `Some(false)` is reserved for models documented not to accept
    /// tool definitions at all (a reasoning endpoint that ignores the
    /// `tools` field, say) — not for "unknown".
    pub tool_calling: Option<bool>,
}

impl ModelCapabilities {
    /// Every capability unreported.
    ///
    /// The starting point for a record nobody has made a claim about,
    /// and the identity of [`ModelMetadata::merged`] — attaching it
    /// changes nothing. Pairs with the builder methods below for callers
    /// assembling a stated set in one expression.
    #[must_use]
    pub const fn unreported() -> Self {
        Self {
            structured_output: None,
            tool_calling: None,
        }
    }

    /// Set the structured-output claim, returning the modified value.
    ///
    /// A stated `true`/`false` outranks curation through
    /// [`ModelMetadata::merged`]; what it means about provenance is the
    /// enclosing [`ModelMetadata::source`]'s business, not this
    /// method's — capabilities carry no source of their own.
    #[must_use]
    pub const fn with_structured_output(mut self, supported: bool) -> Self {
        self.structured_output = Some(supported);
        self
    }

    /// Set the tool-calling claim, returning the modified value.
    ///
    /// Companion of [`with_structured_output`](Self::with_structured_output)
    /// for the other lane; the same rule about what a claim does to
    /// merges and provenance applies.
    #[must_use]
    pub const fn with_tool_calling(mut self, supported: bool) -> Self {
        self.tool_calling = Some(supported);
        self
    }
}

/// What is known about one model: size, modalities, capabilities, and
/// who said so.
///
/// The metadata half of the discovery surface — the facts a
/// context-window planner needs before sizing a request, and the facts
/// a readiness report surfaces instead of a string match. Built three
/// ways: from a provider's listing payload (a **source** of
/// [`Disclosed`](MetadataSource::Disclosed)), from the curated table
/// ([`curated_model_metadata`]), or from both merged
/// ([`ModelMetadata::merged`]).
///
/// Every field is `Option` and the default is fully unreported — an
/// unknown model answers "nobody said", never a plausible zero.
///
/// # Example
///
/// ```rust
/// use loopctl::api::probe::{MetadataSource, ModelMetadata};
/// let disclosed = ModelMetadata::unreported().with_context_window(32_768);
/// assert_eq!(disclosed.context_window, Some(32_768));
/// assert_eq!(
///     disclosed.source,
///     MetadataSource::Disclosed,
///     "attaching a fact is a stating event — the record is no longer unreported"
/// );
/// assert_eq!(ModelMetadata::default().context_window, None);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "RawModelMetadata")]
#[non_exhaustive]
pub struct ModelMetadata {
    /// Total context window in tokens, when anyone said.
    ///
    /// Units are tokens as the provider spelled them; note that
    /// providers disagree on whether this counts output too, which is
    /// why consumers compare it against token *inputs* rather than
    /// trusting it as a hard budget.
    pub context_window: Option<u64>,

    /// Content types the model accepts, when stated.
    ///
    /// `Some(vec![])` is a stated empty list (the endpoint said "no
    /// extra modalities"), distinct from `None` ("nobody said");
    /// parsers only populate it from an explicit modality field.
    pub input_modalities: Option<Vec<Modality>>,

    /// Content types the model emits, when stated.
    ///
    /// The output twin of [`input_modalities`](Self::input_modalities);
    /// kept separate because text-only-output with multimodal input is
    /// the common shape, and collapsing the two would erase it.
    pub output_modalities: Option<Vec<Modality>>,

    /// Structured-output and tool-calling claims.
    ///
    /// The one field that is a struct rather than an `Option`: each
    /// claim inside it carries its own `None`, so "one lane stated, one
    /// lane unknown" is representable — which is exactly how capability
    /// documentation usually arrives.
    pub capabilities: ModelCapabilities,

    /// Who contributed the populated fields — the provenance marker
    /// callers are expected to read alongside the values.
    ///
    /// Kept honest by construction: [`unreported`](Self::unreported)
    /// states nothing, the [`with_*` builders](Self::with_context_window)
    /// promote only to [`Disclosed`](MetadataSource::Disclosed), the
    /// `Deserialize` impl applies that same promotion to documents that
    /// state a fact without a marker, and
    /// [`Curated`](MetadataSource::Curated) is reachable only through
    /// [`merged`](Self::merged) or the curated table itself.
    pub source: MetadataSource,
}

impl ModelMetadata {
    /// A fully unreported record.
    ///
    /// Every field `None`, source [`Unreported`](MetadataSource::Unreported)
    /// — the answer "nobody said" as a value. Equal to
    /// [`Default::default`]; named anyway because the honesty discipline
    /// deserves a spelling, not a type-level default the reader has to
    /// decode.
    #[must_use]
    pub const fn unreported() -> Self {
        Self {
            context_window: None,
            input_modalities: None,
            output_modalities: None,
            capabilities: ModelCapabilities::unreported(),
            source: MetadataSource::Unreported,
        }
    }

    /// A record with only a disclosed context window.
    ///
    /// The shape a `/props`-style probe or a listing's
    /// `inputTokenLimit` produces on its own; see
    /// [`Disclosed`](MetadataSource::Disclosed) for why the source is
    /// set at construction. Records with more than the window are built
    /// through the [`with_*`](Self::with_context_window) chain over
    /// [`unreported`](Self::unreported), which marks the same source on
    /// every attach.
    #[must_use]
    pub const fn disclosed_context_window(tokens: u64) -> Self {
        Self {
            context_window: Some(tokens),
            input_modalities: None,
            output_modalities: None,
            capabilities: ModelCapabilities::unreported(),
            source: MetadataSource::Disclosed,
        }
    }

    /// Attach a stated context window, returning the modified value.
    ///
    /// Attaching a fact is a stating event: a record still marked
    /// [`Unreported`](MetadataSource::Unreported) is promoted to
    /// [`Disclosed`](MetadataSource::Disclosed) — the builders are the
    /// "the endpoint (or this caller, from the endpoint's answer) said
    /// it" surface, so a builder-attached field can never read
    /// [`Curated`](MetadataSource::Curated) through a later
    /// [`merged`](Self::merged). A base already marked `Curated` or
    /// `Disclosed` keeps its marker.
    #[must_use]
    pub const fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        if matches!(self.source, MetadataSource::Unreported) {
            self.source = MetadataSource::Disclosed;
        }
        self
    }

    /// Attach stated input modalities, returning the modified value.
    ///
    /// Source promotion exactly as [`with_context_window`](Self::with_context_window)
    /// documents; an explicitly empty list counts as stated (it says
    /// "nothing but the base"), which is why the parameter is a `Vec`
    /// and not an `Option`.
    #[must_use]
    pub fn with_input_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.input_modalities = Some(modalities);
        if matches!(self.source, MetadataSource::Unreported) {
            self.source = MetadataSource::Disclosed;
        }
        self
    }

    /// Attach stated output modalities, returning the modified value.
    ///
    /// The output twin of [`with_input_modalities`](Self::with_input_modalities),
    /// same promotion rule, same stated-empty-list reading.
    #[must_use]
    pub fn with_output_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.output_modalities = Some(modalities);
        if matches!(self.source, MetadataSource::Unreported) {
            self.source = MetadataSource::Disclosed;
        }
        self
    }

    /// Attach capability claims, returning the modified value.
    ///
    /// Promotion to [`Disclosed`](MetadataSource::Disclosed) happens
    /// only when the attached record states something — at least one
    /// `Some` claim. Attaching an all-[`None`](ModelCapabilities::unreported)
    /// set states nothing, so an unreported base stays unreported: the
    /// invariant the [`Unreported`](MetadataSource::Unreported) marker
    /// advertises is "every fact silent", and an empty claim struct
    /// keeps that promise.
    #[must_use]
    pub const fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        let stated =
            capabilities.structured_output.is_some() || capabilities.tool_calling.is_some();
        self.capabilities = capabilities;
        if stated && matches!(self.source, MetadataSource::Unreported) {
            self.source = MetadataSource::Disclosed;
        }
        self
    }

    /// Overlay two partial records, recording the provenance honestly.
    ///
    /// `disclosed` wins field by field; `curated` only fills fields the
    /// disclosed record leaves `None`. The resulting
    /// [`source`](ModelMetadata::source) says which of the inputs
    /// actually contributed: a disclosed record extended by curated
    /// facts becomes [`DisclosedAndCurated`](MetadataSource::DisclosedAndCurated),
    /// one that needed nothing keeps [`Disclosed`](MetadataSource::Disclosed),
    /// and an unreported base filled entirely by curation reports
    /// [`Curated`](MetadataSource::Curated).
    ///
    /// Credit follows contribution, never naming: a base assembled
    /// through the [`with_*`](Self::with_context_window) builders is
    /// already marked [`Disclosed`](MetadataSource::Disclosed) by the
    /// attach, so nothing a builder contributed can end up attributed
    /// to the curated table — `Curated` is only produced when the base
    /// stated nothing at all.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::probe::{curated_model_metadata, MetadataSource, ModelMetadata};
    /// let disclosed = ModelMetadata::disclosed_context_window(32_768);
    /// let curated = curated_model_metadata("gpt-4o").expect("the gpt-4o family has a row");
    /// let merged = ModelMetadata::merged(Some(&disclosed), Some(&curated));
    /// assert_eq!(merged.context_window, Some(32_768), "the disclosure wins the window");
    /// assert_eq!(
    ///     merged.capabilities.tool_calling,
    ///     Some(true),
    ///     "curation fills what the disclosure left open"
    /// );
    /// assert_eq!(merged.source, MetadataSource::DisclosedAndCurated);
    /// ```
    #[must_use]
    pub fn merged(disclosed: Option<&Self>, curated: Option<&Self>) -> Self {
        let Some(curated) = curated else {
            return disclosed.cloned().unwrap_or_default();
        };
        let Some(disclosed) = disclosed else {
            return curated.clone();
        };

        let mut merged = disclosed.clone();
        let mut filled = false;
        if merged.context_window.is_none() && curated.context_window.is_some() {
            merged.context_window = curated.context_window;
            filled = true;
        }
        if merged.input_modalities.is_none() && curated.input_modalities.is_some() {
            merged
                .input_modalities
                .clone_from(&curated.input_modalities);
            filled = true;
        }
        if merged.output_modalities.is_none() && curated.output_modalities.is_some() {
            merged
                .output_modalities
                .clone_from(&curated.output_modalities);
            filled = true;
        }
        if merged.capabilities.structured_output.is_none()
            && curated.capabilities.structured_output.is_some()
        {
            merged.capabilities.structured_output = curated.capabilities.structured_output;
            filled = true;
        }
        if merged.capabilities.tool_calling.is_none() && curated.capabilities.tool_calling.is_some()
        {
            merged.capabilities.tool_calling = curated.capabilities.tool_calling;
            filled = true;
        }
        merged.source = match (disclosed.source, filled) {
            (MetadataSource::Unreported, true) => MetadataSource::Curated,
            (MetadataSource::Disclosed, true) => MetadataSource::DisclosedAndCurated,
            (source, _) => source,
        };
        merged
    }
}

/// The wire shape a [`ModelMetadata`] arrives from.
///
/// Deserialization is construction, so the wire shape is held to the
/// same honesty rule the builders enforce: a document that states a
/// fact while omitting `source` lands [`Disclosed`], never
/// [`Unreported`] beside a populated field. A document that does state
/// a marker is taken verbatim — including `"Unreported"` beside nulls.
///
/// [`Disclosed`]: MetadataSource::Disclosed
/// [`Unreported`]: MetadataSource::Unreported
#[derive(Deserialize)]
struct RawModelMetadata {
    /// Context window exactly as the document spelled it.
    ///
    /// Mirrors [`ModelMetadata::context_window`]; an absent key is the
    /// same unreported `None` the builders start from.
    #[serde(default)]
    context_window: Option<u64>,

    /// Input modalities as the document spelled them.
    ///
    /// Mirrors [`ModelMetadata::input_modalities`], stated-empty and
    /// absent alike.
    #[serde(default)]
    input_modalities: Option<Vec<Modality>>,

    /// Output modalities as the document spelled them.
    ///
    /// Mirrors [`ModelMetadata::output_modalities`]; the same
    /// stated-empty versus absent distinction survives untouched.
    #[serde(default)]
    output_modalities: Option<Vec<Modality>>,

    /// Capability claims as the document spelled them.
    ///
    /// Mirrors [`ModelMetadata::capabilities`]; each claim lane inside
    /// defaults to unreported independently.
    #[serde(default)]
    capabilities: ModelCapabilities,

    /// The provenance marker as written, or unreported when omitted.
    ///
    /// Mirrors [`ModelMetadata::source`]; the promotion the record
    /// possibly needs is decided by [`From`], which sees all fields at
    /// once — a field-level default could not state a cross-field rule.
    #[serde(default)]
    source: MetadataSource,
}

impl From<RawModelMetadata> for ModelMetadata {
    /// Fold a wire record into the model, promoting a fact stated
    /// without a marker exactly as the
    /// [`with_*` builders](ModelMetadata) do.
    ///
    /// Only the `Unreported` marker is ever rewritten: a document that
    /// claims `"Curated"` keeps the claim — provenance the wire asserts
    /// is the wire's to assert, and `merged` remains the only path into
    /// `Curated` taken by this crate's own code.
    fn from(raw: RawModelMetadata) -> Self {
        let stated = raw.context_window.is_some()
            || raw.input_modalities.is_some()
            || raw.output_modalities.is_some()
            || raw.capabilities.structured_output.is_some()
            || raw.capabilities.tool_calling.is_some();
        let source = if stated && raw.source == MetadataSource::Unreported {
            MetadataSource::Disclosed
        } else {
            raw.source
        };
        Self {
            context_window: raw.context_window,
            input_modalities: raw.input_modalities,
            output_modalities: raw.output_modalities,
            capabilities: raw.capabilities,
            source,
        }
    }
}

/// One entry of a provider's model listing.
///
/// The wire payloads differ wildly in how much they say — an OpenAI
/// `/models` entry is an `id` and a creation stamp, a Bedrock entry
/// carries display names and modality lists, Ollama reports a size and
/// a modification time — and this type holds exactly what the entry
/// carried, no invented fill. The [`metadata`](ModelInfo::metadata)
/// field is where [`model_metadata`](crate::api::ApiClient::model_metadata)
/// semantics attach: provider-disclosed facts merged over curated ones,
/// provenance marked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelInfo {
    /// The model id to pass to the provider.
    ///
    /// What a completion request must carry for this entry to be usable
    /// — the raw endpoint id, not a display name or an alias. Bedrock
    /// entries keep their full prefixed id (`meta.llama…`) here even
    /// though curated lookups strip the prefix internally.
    pub id: String,

    /// Human-facing name, when the provider served one.
    ///
    /// Anthropic's `display_name`, Gemini's `displayName`, Bedrock's
    /// `modelName`; the OpenAI shape carries none, so `None` here is
    /// ordinary, not a parse failure. Never substituted into [`id`](Self::id).
    #[serde(default)]
    pub display_name: Option<String>,

    /// Creation stamp, verbatim as the provider spelled it (an ISO date,
    /// an epoch, a `modified_at` timestamp).
    ///
    /// Deliberately a string: the providers do not agree on a unit or
    /// format, and a lossy parse that guesses a unit is worse for
    /// consumers than the raw text they can parse with their own rules.
    #[serde(default)]
    pub created: Option<String>,

    /// The organisation that operates the model, as the provider named
    /// it (`"openai"`, `"anthropic"`, Bedrock's `providerName`).
    ///
    /// Free text owned by the endpoint — casing and spelling are its
    /// own, and comparisons should be case-insensitive at the consumer.
    #[serde(default)]
    pub owned_by: Option<String>,

    /// Local weight size in bytes, when the endpoint reported one
    /// (Ollama's `/api/tags` `size` field is the only known source
    /// today).
    ///
    /// A disk/VRAM figure, not a capability statement; hosted listings
    /// never carry it, which is the normal answer for a cloud endpoint.
    #[serde(default)]
    pub size_bytes: Option<u64>,

    /// Context window, modalities, and capability claims with their
    /// provenance marker; defaults to fully unreported.
    ///
    /// Listing parsers fill this per the funnel rule — wire disclosures
    /// over the curated table — so the [`source`](ModelMetadata::source)
    /// inside says whether a hosted disclosure, pure curation, or
    /// neither produced the values.
    #[serde(default)]
    pub metadata: ModelMetadata,
}

impl ModelInfo {
    /// A listing entry with only its usable id.
    ///
    /// Everything else — name, stamps, operator, size, metadata —
    /// unreported; the constructors' baseline for a payload that only
    /// says the id, and the honest floor for a minimal endpoint.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::probe::{ModelInfo, ModelMetadata};
    /// let entry = ModelInfo::new("gpt-4o-2024-08-06")
    ///     .with_owned_by("openai")
    ///     .with_created("1721172744");
    /// assert_eq!(entry.id, "gpt-4o-2024-08-06");
    /// assert_eq!(entry.metadata, ModelMetadata::unreported());
    /// ```
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            created: None,
            owned_by: None,
            size_bytes: None,
            metadata: ModelMetadata::unreported(),
        }
    }

    /// Attach the human-facing name, returning the modified value.
    ///
    /// Fills [`display_name`](Self::display_name) verbatim; the entry's
    /// usable [`id`](Self::id) is never touched, so display and request
    /// text cannot drift into each other.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Attach the provider's creation stamp verbatim, returning the
    /// modified value.
    ///
    /// Any text at all is accepted — an epoch integer rendered as a
    /// string, an RFC 3339 date, an Ollama `modified_at` — because the
    /// field's contract is the provider's own spelling, not a format.
    #[must_use]
    pub fn with_created(mut self, created: impl Into<String>) -> Self {
        self.created = Some(created.into());
        self
    }

    /// Attach the operating organisation, returning the modified value.
    ///
    /// Fills [`owned_by`](Self::owned_by) with the provider's own
    /// naming; the crate never normalizes `"OpenAI"` to `"openai"` or
    /// guesses an operator an entry did not state.
    #[must_use]
    pub fn with_owned_by(mut self, owned_by: impl Into<String>) -> Self {
        self.owned_by = Some(owned_by.into());
        self
    }

    /// Attach a reported weight size, returning the modified value.
    ///
    /// Bytes, as the endpoint spelled [`size_bytes`](Self::size_bytes);
    /// unit conversion is left to the consumer, since Ollama's `size`
    /// field is already a byte count and inventing a different reading
    /// would be guesswork.
    #[must_use]
    pub const fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Attach a metadata record, returning the modified value.
    ///
    /// The attach point listing parsers use for the funnel result
    /// (disclosures merged over curation); it replaces whatever record
    /// was there and carries its provenance in the value itself, so no
    /// separate source marker can drift out of sync.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Why a model-listing or liveness probe failed, in four usable
/// categories plus the honest leftovers.
///
/// The distinction a health-check consumer actually branches on —
/// *"is my credential bad, am I throttled, is the endpoint down, or did
/// it answer something unexpected?"* — computed mechanically from the
/// [`ApiError`] variant and (for [`Http`](ApiError::Http)) the embedded
/// HTTP status, mirroring the provider's own status→variant mapping:
/// the same failure reads the same here as through
/// [`ApiError::code`], and [`ProbeClass::from_error`] agrees with
/// the telemetry vocabulary's `wire_failure_class` on every wire
/// shape (pinned by test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProbeClass {
    /// The credential was rejected.
    ///
    /// HTTP 401/403, an [`ApiError::Auth`] failure. Fix the key, not the
    /// URL: no URL or model choice turns a refused credential into an
    /// accepted one. The class tracks the typed *variant*: a 401/403
    /// riding a plain [`Http`](ApiError::Http) is an unclassified refusal
    /// and reads [`Answered`](Self::Answered) — every wire path builds
    /// `Auth` from a real rejection, so the bare shape arrives only
    /// hand-built.
    Auth,

    /// The endpoint answered and refused on load grounds.
    ///
    /// An [`ApiError::RateLimit`] (its `retry_after`, when carried, is
    /// the provider's own advice) or a throttle status (429/503/529)
    /// carried by a plain [`Http`](ApiError::Http) — the throttle is the
    /// status's meaning, not the variant's, so a hand-built or
    /// legacy-shaped `Http("HTTP 429: …")` classifies here exactly as
    /// [`ApiError::is_rate_limited`](ApiError::is_rate_limited) already
    /// reads it.
    RateLimited,

    /// The provider itself failed: a 5xx that is not a throttle.
    ///
    /// The endpoint is present; it is having a bad day. The load
    /// refusals 503/529 belong to [`RateLimited`](Self::RateLimited)
    /// instead — a *retry later* is a different remedy from *the server
    /// is broken*, and the crate's own `is_rate_limited` family has
    /// always read those statuses as throttles.
    Server,

    /// The endpoint answered with a non-success status that is none of
    /// the above.
    ///
    /// 404 "no such route" is the canonical case. Something is there and
    /// it understood the request enough to refuse it — usually a wrong
    /// URL or model id, never a dead credential, a quota, or a down
    /// server.
    Answered,

    /// No response ever arrived.
    ///
    /// Connection refused, DNS failure, TLS problem, or the probe
    /// deadline elapsed before an answer. The classic false "auth" — an
    /// offline endpoint must never read as a credential problem, which
    /// is why this class is derived from the absence of a status, never
    /// from error text.
    Network,

    /// This client kind cannot answer the probe at all.
    ///
    /// The trait default's [`ApiError::Config`] on a client with no
    /// listing endpoint. A capability statement about the *client*, not
    /// a verdict about the provider — the provider may be perfectly
    /// healthy and simply never expose a catalog route to this client.
    Unsupported,

    /// A failure the probe vocabulary has no category for.
    ///
    /// A malformed listing body, an interrupted wait, a wire shape the
    /// status tables do not reach. The honest leftover: a probe is not
    /// allowed to guess a category just to have an answer.
    Other,
}

impl ProbeClass {
    /// Classify a probe failure.
    ///
    /// Total: every [`ApiError`] lands in exactly one class, and the
    /// auth / throttle / network split — the three a health check must
    /// never confuse — is mechanical from the variant and its embedded
    /// status, never from message text. A throttle status carried by a
    /// plain [`Http`](ApiError::Http) classifies as
    /// [`RateLimited`](Self::RateLimited) before the status-code split,
    /// because `429`/`503`/`529` spell "retry later" whoever carries
    /// them; that agreement is pinned against `wire_failure_class` over
    /// the whole 400–529 sweep.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::error::ApiError;
    /// use loopctl::api::probe::ProbeClass;
    /// assert_eq!(ProbeClass::from_error(&ApiError::auth("bad key")), ProbeClass::Auth);
    /// assert_eq!(
    ///     ProbeClass::from_error(&ApiError::rate_limited("slow down", None)),
    ///     ProbeClass::RateLimited
    /// );
    /// assert_eq!(
    ///     ProbeClass::from_error(&ApiError::http_with_status(429, "slow down")),
    ///     ProbeClass::RateLimited,
    ///     "a throttle that arrived as a plain HTTP status is still a throttle"
    /// );
    /// assert_eq!(
    ///     ProbeClass::from_error(&ApiError::http("connection refused")),
    ///     ProbeClass::Network,
    ///     "an endpoint that never answered is never a credential problem"
    /// );
    /// ```
    #[must_use]
    pub fn from_error(error: &ApiError) -> Self {
        match error {
            ApiError::Auth(..) => Self::Auth,
            ApiError::RateLimit { .. } => Self::RateLimited,
            ApiError::Http(msg, ..) => {
                if http_status_is_overload(msg) {
                    Self::RateLimited
                } else {
                    match error.code() {
                        ErrorCode::HttpConnectionError => Self::Network,
                        ErrorCode::HttpRequestError => Self::Answered,
                        ErrorCode::HttpResponseError => Self::Server,
                        _ => Self::Other,
                    }
                }
            }
            ApiError::Config(..) => Self::Unsupported,
            _ => Self::Other,
        }
    }
}

/// What a liveness probe observed.
///
/// A probe's value is the *evidence*, not a boolean: how long the
/// endpoint took, what quota state its answer carried, which request id
/// a support desk could look up. Every field past [`latency`](Self::latency)
/// is `None` until the provider actually served the header — silence
/// from a spartan endpoint is normal and carries no meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PingReport {
    /// Time from request start to the endpoint's answer.
    ///
    /// Wall-clock latency to a classified success — a probe that
    /// failed has no report at all (the failure itself is the
    /// observation, classified as a [`ProbeClass`]), so this duration
    /// always measures a completed round trip, not an abandoned one.
    pub latency: std::time::Duration,

    /// The `x-ratelimit-*` / `rate-limit-*` family from the answer,
    /// verbatim.
    ///
    /// Captured through the same plumbing a completion failure uses:
    /// the strings are the provider's own spelling (units differ per
    /// endpoint and are never normalized here).
    pub rate_limit: Option<RateLimitSnapshot>,

    /// The provider's request identifier from the answer
    /// (`x-request-id`, `request-id`, `x-amzn-requestid`, …).
    ///
    /// The handle a support desk asks for; `None` for endpoints that
    /// answer without echoing one, which is normal for local servers.
    pub request_id: Option<String>,
}

impl PingReport {
    /// A report carrying only the measured latency.
    ///
    /// The floor of an honest probe result: one number the caller
    /// measured, no headers invented beside it. Consumers that need the
    /// evidence family extend it with the builders below as the answer
    /// actually supplies each piece.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::api::probe::PingReport;
    /// use std::time::Duration;
    /// let report = PingReport::new(Duration::from_millis(12));
    /// assert_eq!(report.latency, Duration::from_millis(12));
    /// assert_eq!(report.request_id, None, "silence from the endpoint is not data");
    /// ```
    #[must_use]
    pub fn new(latency: std::time::Duration) -> Self {
        Self {
            latency,
            rate_limit: None,
            request_id: None,
        }
    }

    /// Attach the rate-limit headers observed, returning the modified
    /// value.
    ///
    /// Fills [`rate_limit`](Self::rate_limit) with the verbatim snapshot
    /// the response plumbing captured; callers never hand-assemble
    /// header text here — the snapshot type bounds it at construction.
    #[must_use]
    pub fn with_rate_limit(mut self, rate_limit: RateLimitSnapshot) -> Self {
        self.rate_limit = Some(rate_limit);
        self
    }

    /// Attach the provider's request id, returning the modified value.
    ///
    /// Fills [`request_id`](Self::request_id) exactly as the header
    /// spelled it — the value's whole purpose is to be quoted back to
    /// the provider's support desk unchanged.
    #[must_use]
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }
}

/// One row of the crate's curated metadata table.
///
/// An internal shape; every field mirrors its [`ModelMetadata`] twin.
/// `None` means "the crate takes no position", never "false". Rows are
/// `const` data read by [`curated_model_metadata`] on every lookup —
/// there is no builder because there is no runtime editing story.
struct CuratedRow {
    /// Lowercase model-id prefix the row covers.
    ///
    /// Longest match wins, so `"gpt-4o-2024-08-06"` inherits the
    /// `"gpt-4o"` row; prefixes ending in a separator (`llama3.` vs
    /// `llama3-`) are deliberate spellings, pinning which id shapes the
    /// table claims an opinion about.
    prefix: &'static str,

    /// The published total context window, or `None` for no position.
    ///
    /// Tokens, as the family's documentation spells the total; a family
    /// with disputed or deployment-dependent totals keeps `None` here
    /// rather than shipping a number the table cannot stand behind.
    context_window: Option<u64>,

    /// Structured-output claim, or `None` for no position.
    ///
    /// `Some(false)` is a documented refusal (a family with no native
    /// JSON-schema response mode); an unverified family is `None`, which
    /// is the table's default answer, not an oversight.
    structured_output: Option<bool>,

    /// Tool-calling claim, or `None` for no position.
    ///
    /// `Some(false)` is reserved for documented refusals (a reasoning
    /// endpoint that ignores `tools`); "nobody verified" is `None`.
    tool_calling: Option<bool>,

    /// Input modality claim, or `None` for no position.
    ///
    /// A static slice to keep rows const-constructible and lookups
    /// allocation-free until the record is materialized.
    input_modalities: Option<&'static [Modality]>,

    /// Output modality claim, or `None` for no position.
    ///
    /// The output twin of [`input_modalities`](Self::input_modalities);
    /// every row today emits text, but the field stays `Option` because
    /// future rows (image or embedding emitters) will not.
    output_modalities: Option<&'static [Modality]>,
}

/// Text-only input; the majority of open-weight models.
///
/// The curated table's baseline modality list — a family gets this when
/// its documentation says nothing beyond "send it text". Static so rows
/// are const-constructible; consumers receive an owned `Vec` only when
/// a record materializes.
const TEXT_IN: &[Modality] = &[Modality::Text];

/// Text-and-image input.
///
/// The hosted frontier families' usual disclosure: text plus still
/// images, audio left out. Spelled separately from the single-modality
/// lists so rows state exactly what their family documented, never the
/// maximal set.
const TEXT_IMAGE_IN: &[Modality] = &[Modality::Text, Modality::Image];

/// The full multimodal input set the Gemini family documents.
///
/// Text, image, audio, and video — the one list in the table that
/// reaches four modalities, matching Gemini's published listing rather
/// than any broader claim; the wire remains the authority when it says
/// more.
const MULTIMODAL_IN: &[Modality] = &[
    Modality::Text,
    Modality::Image,
    Modality::Audio,
    Modality::Video,
];

/// Text output; every model in the table today emits text.
///
/// Kept as a named constant (not inlined into rows) so a future row
/// that emits something else visibly *contradicts* this default instead
/// of the constant quietly rotting out of use.
const TEXT_OUT: &[Modality] = &[Modality::Text];

/// The crate's curated opinions, keyed by model-id prefix.
///
/// Deliberately small and deliberately conservative: only published,
/// stable, well-known values, and `None` wherever the record would
/// otherwise be a guess (the structured-output column is `None` for
/// every family whose schema support the crate has not verified —
/// Qwen, Llama, Grok-3, Kimi, GLM). A provider that discloses its own
/// numbers on the wire always wins; this table only ever fills gaps,
/// and [`MetadataSource::Curated`] keeps the provenance visible so a
/// consumer can disregard it entirely.
const CURATED: &[CuratedRow] = &[
    CuratedRow {
        prefix: "gpt-4o",
        context_window: Some(128_000),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "gpt-4.1",
        context_window: Some(1_048_576),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "o3",
        context_window: Some(200_000),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "o4-mini",
        context_window: Some(200_000),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "claude-3",
        context_window: Some(200_000),
        structured_output: Some(false),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "claude-opus-4",
        context_window: Some(200_000),
        structured_output: Some(false),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "claude-sonnet-4",
        context_window: Some(200_000),
        structured_output: Some(false),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "claude-haiku",
        context_window: Some(200_000),
        structured_output: Some(false),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "gemini-1.5",
        context_window: Some(1_048_576),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(MULTIMODAL_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "gemini-2.0",
        context_window: Some(1_048_576),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(MULTIMODAL_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "gemini-2.5",
        context_window: Some(1_048_576),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(MULTIMODAL_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "deepseek-chat",
        context_window: Some(64_000),
        structured_output: Some(false),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "deepseek-reasoner",
        context_window: Some(64_000),
        structured_output: Some(false),
        tool_calling: Some(false),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "llama3.",
        context_window: Some(128_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "llama3-",
        context_window: Some(128_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "llama-3.",
        context_window: Some(128_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "llama-3-",
        context_window: Some(128_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "grok-3",
        context_window: Some(131_072),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "grok-4",
        context_window: Some(256_000),
        structured_output: Some(true),
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IMAGE_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "kimi-k2",
        context_window: Some(262_144),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "glm-4.5",
        context_window: Some(131_072),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "glm-4.6",
        context_window: Some(200_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "qwen2.5",
        context_window: Some(131_072),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "qwen3",
        context_window: Some(131_072),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
    CuratedRow {
        prefix: "mistral-large",
        context_window: Some(128_000),
        structured_output: None,
        tool_calling: Some(true),
        input_modalities: Some(TEXT_IN),
        output_modalities: Some(TEXT_OUT),
    },
];

/// The crate's curated metadata opinion for a model id, if it has one.
///
/// Prefix lookup over the published model families the crate documents
/// positions on; case-insensitive, longest matching prefix wins (so
/// dated and snapshot ids inherit their family's row). An unknown id —
/// a fine-tune, a private deployment, a next month's release — answers
/// [`None`], which every consumer path degrades to "unreported": a
/// missing opinion is always the honest default for a model this table
/// predates.
///
/// These are *opinions*: they carry
/// [`MetadataSource::Curated`] so a caller can tell them apart from a
/// provider's own disclosure and disregard them if it wants only wire
/// facts.
///
/// # Example
///
/// ```rust
/// use loopctl::api::probe::{curated_model_metadata, MetadataSource};
/// let gpt4o = curated_model_metadata("gpt-4o-2024-08-06").expect("the gpt-4o family has a row");
/// assert_eq!(gpt4o.context_window, Some(128_000));
/// assert_eq!(gpt4o.source, MetadataSource::Curated);
/// assert!(curated_model_metadata("my-fine-tune-9000").is_none());
/// ```
#[must_use]
pub fn curated_model_metadata(model: &str) -> Option<ModelMetadata> {
    let id = model.trim().to_ascii_lowercase();
    let row = CURATED
        .iter()
        .filter(|row| id.starts_with(row.prefix))
        .max_by_key(|row| row.prefix.len())?;
    Some(ModelMetadata {
        context_window: row.context_window,
        input_modalities: row.input_modalities.map(<[Modality]>::to_vec),
        output_modalities: row.output_modalities.map(<[Modality]>::to_vec),
        capabilities: ModelCapabilities {
            structured_output: row.structured_output,
            tool_calling: row.tool_calling,
        },
        source: MetadataSource::Curated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_class_reads_each_wire_failure_shape() {
        use std::time::Duration;

        let cases: &[(&str, ApiError, ProbeClass)] = &[
            (
                "401",
                ApiError::auth_invalid_key("bad key"),
                ProbeClass::Auth,
            ),
            ("403", ApiError::auth("forbidden"), ProbeClass::Auth),
            (
                "429 with advice",
                ApiError::rate_limited("slow down", Some(Duration::from_secs(7))),
                ProbeClass::RateLimited,
            ),
            (
                "500",
                ApiError::http_with_status(500, "internal"),
                ProbeClass::Server,
            ),
            (
                "429 riding a plain Http",
                ApiError::http_with_status(429, "slow down"),
                ProbeClass::RateLimited,
            ),
            (
                "503 riding a plain Http",
                ApiError::http_with_status(503, "unavailable"),
                ProbeClass::RateLimited,
            ),
            (
                "529 riding a plain Http",
                ApiError::http_with_status(529, "overloaded"),
                ProbeClass::RateLimited,
            ),
            (
                "404 unclassified",
                ApiError::http_with_status(404, "no route"),
                ProbeClass::Answered,
            ),
            (
                "connection refused",
                ApiError::http("connection refused"),
                ProbeClass::Network,
            ),
            (
                "tls handshake",
                ApiError::http("invalid peer certificate"),
                ProbeClass::Network,
            ),
            (
                "no listing endpoint",
                ApiError::config("this client does not expose a model listing"),
                ProbeClass::Unsupported,
            ),
            ("api failure", ApiError::api("boom"), ProbeClass::Other),
            ("tool failure", ApiError::tool("boom"), ProbeClass::Other),
        ];
        for (label, error, expected) in cases {
            assert_eq!(
                ProbeClass::from_error(error),
                *expected,
                "{label} must classify as {expected:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn curated_table_matches_exact_then_longest_prefix() {
        let exact = curated_model_metadata("gpt-4o")
            .expect("gpt-4o is a family the crate holds an opinion about");
        assert_eq!(exact.context_window, Some(128_000));

        let dated = curated_model_metadata("gpt-4o-2024-08-06")
            .expect("dated snapshots inherit their family row");
        assert_eq!(dated, exact);

        assert!(
            curated_model_metadata("gpt-4").is_none(),
            "a bare 'gpt-4' must not silently inherit the gpt-4o row — the \
             table holds no gpt-4 opinion, and prefix matching must not invent one"
        );

        let upper = curated_model_metadata("GPT-4O-MINI").expect("lookup is case-insensitive");
        assert_eq!(upper.context_window, Some(128_000));

        let tag = curated_model_metadata("llama3.1:latest")
            .expect("ollama tags carry their family prefix");
        assert_eq!(tag.context_window, Some(128_000));
        assert_eq!(tag.capabilities.structured_output, None);

        let bedrock_dashed = curated_model_metadata("llama3-2-11b-instruct-v1:0")
            .expect("Bedrock's dash-spelled llama ids (past the `meta.` prefix) hit the llama row");
        assert_eq!(bedrock_dashed.context_window, Some(128_000));
        assert_eq!(bedrock_dashed.capabilities.tool_calling, Some(true));

        let dashed = curated_model_metadata("llama-3-1-70b")
            .expect("the dashed llama-3- spelling has a row too");
        assert_eq!(dashed.context_window, Some(128_000));

        assert!(curated_model_metadata("not-a-real-model-9000").is_none());
        assert!(curated_model_metadata("").is_none());
    }

    #[test]
    fn curated_table_holds_back_where_the_record_is_thin() {
        // The honesty rule: `None` wherever the crate takes no position.
        let claude = curated_model_metadata("claude-sonnet-4-5")
            .expect("the claude sonnet 4 family has a row");
        assert_eq!(
            claude.capabilities.structured_output,
            Some(false),
            "the Messages API has no native JSON-schema response mode"
        );
        assert_eq!(claude.capabilities.tool_calling, Some(true));

        let reasoner =
            curated_model_metadata("deepseek-reasoner").expect("deepseek-reasoner has a row");
        assert_eq!(
            reasoner.capabilities.tool_calling,
            Some(false),
            "the reasoning endpoint is documented not to accept tool definitions"
        );

        let qwen = curated_model_metadata("qwen2.5-coder").expect("qwen2.5 has a row");
        assert_eq!(
            qwen.capabilities.structured_output, None,
            "the crate has no verified schema-support claim for Qwen"
        );
        assert_eq!(qwen.capabilities.tool_calling, Some(true));

        let mistral = curated_model_metadata("mistral-large-latest")
            .expect("the mistral-large family has a row");
        assert_eq!(
            mistral.context_window,
            Some(128_000),
            "the family's published window, pinned"
        );
        assert_eq!(
            mistral.capabilities.tool_calling,
            Some(true),
            "function calling is the family's documented surface"
        );
        assert_eq!(
            mistral.capabilities.structured_output, None,
            "the table holds no schema-mode claim beside the published record"
        );
    }

    #[test]
    fn metadata_merge_marks_the_provenance() {
        use MetadataSource::*;

        let disclosed = ModelMetadata::disclosed_context_window(32_768);
        let curated = curated_model_metadata("gpt-4o").expect("gpt-4o is a curated family");

        let both = ModelMetadata::merged(Some(&disclosed), Some(&curated));
        assert_eq!(both.context_window, Some(32_768), "disclosed wins");
        assert_eq!(
            both.capabilities.tool_calling, curated.capabilities.tool_calling,
            "curated fills what the disclosure left open"
        );
        assert_eq!(both.source, DisclosedAndCurated);

        let full_disclosure = disclosed
            .clone()
            .with_capabilities(
                ModelCapabilities::unreported()
                    .with_structured_output(true)
                    .with_tool_calling(true),
            ) // every field now stated
            .with_input_modalities(vec![Modality::Text])
            .with_output_modalities(vec![Modality::Text]);
        let nothing_to_fill = ModelMetadata::merged(Some(&full_disclosure), Some(&curated));
        assert_eq!(
            nothing_to_fill.source, Disclosed,
            "curated contributed nothing, so the record must not claim it did"
        );

        let curated_only = ModelMetadata::merged(None, Some(&curated));
        assert_eq!(curated_only.source, Curated);
        assert_eq!(curated_only.context_window, Some(128_000));

        let disclosed_only = ModelMetadata::merged(Some(&disclosed), None);
        assert_eq!(disclosed_only.source, Disclosed);

        let neither = ModelMetadata::merged(None, None);
        assert_eq!(neither, ModelMetadata::unreported());
        assert_eq!(neither.source, Unreported);

        let empty_base = ModelMetadata::unreported();
        let filled_from_nothing = ModelMetadata::merged(Some(&empty_base), Some(&curated));
        assert_eq!(
            filled_from_nothing.source, Curated,
            "an unreported base filled entirely by curation reports Curated, \
             not Unreported"
        );
    }

    #[test]
    fn builders_promote_provenance_and_curation_earns_curated_alone() {
        use MetadataSource::*;

        let attached = ModelMetadata::unreported()
            .with_context_window(10)
            .with_capabilities(ModelCapabilities::unreported().with_tool_calling(true));
        assert_eq!(
            attached.source, Disclosed,
            "attaching a fact is a stating event — an Unreported marker beside \
             a populated field is the contradiction this promotion closes"
        );

        let curated = curated_model_metadata("gpt-4o").expect("gpt-4o is a curated family");
        let credited = ModelMetadata::merged(Some(&attached), Some(&curated));
        assert_eq!(
            credited.context_window,
            Some(10),
            "the base's own number survives the merge"
        );
        assert_eq!(
            credited.source, DisclosedAndCurated,
            "a builder-attached field must never read Curated — the table \
             did not contribute the window it would be credited with"
        );
        assert_ne!(credited.source, Curated);

        let quiet_attach =
            ModelMetadata::unreported().with_capabilities(ModelCapabilities::unreported());
        assert_eq!(
            quiet_attach.source, Unreported,
            "attaching a claim set that states nothing leaves an unreported \
             record unreported"
        );

        let from_curated = curated.with_context_window(99);
        assert_eq!(
            from_curated.source, Curated,
            "the promotion is Unreported -> Disclosed only; a curated base \
             keeps its marker"
        );
    }

    #[test]
    fn every_metadata_field_starts_unreported() {
        let unreported = ModelMetadata::unreported();
        assert_eq!(unreported, ModelMetadata::default());
        assert_eq!(unreported.context_window, None);
        assert_eq!(unreported.input_modalities, None);
        assert_eq!(unreported.output_modalities, None);
        assert_eq!(unreported.capabilities, ModelCapabilities::unreported());
        assert_eq!(unreported.source, MetadataSource::Unreported);
    }

    #[test]
    fn deserialization_promotes_a_fact_stated_without_a_source() {
        // Deserializing is construction, so the wire shape obeys the
        // builders' rule. A hand-written or foreign JSON document stating
        // a fact while omitting `source` must not land in the
        // `Unreported`-beside-a-populated-field state, and must not let
        // `merged` credit the curated table with numbers it never
        // contributed.
        let disclosed: ModelMetadata =
            serde_json::from_str(r#"{"context_window":32768}"#).expect("the plain shape parses");
        assert_eq!(disclosed.source, MetadataSource::Disclosed);
        assert_eq!(disclosed.context_window, Some(32_768));

        let claimed: ModelMetadata =
            serde_json::from_str(r#"{"capabilities":{"tool_calling":true}}"#)
                .expect("a capability claim parses");
        assert_eq!(
            claimed.source,
            MetadataSource::Disclosed,
            "a claim inside the capabilities struct counts as a stated fact"
        );

        let bare: ModelMetadata = serde_json::from_str("{}").expect("the empty document parses");
        assert_eq!(
            bare.source,
            MetadataSource::Unreported,
            "no facts, no promotion"
        );

        let marked: ModelMetadata =
            serde_json::from_str(r#"{"context_window":1,"source":"Curated"}"#)
                .expect("an explicit marker parses");
        assert_eq!(
            marked.source,
            MetadataSource::Curated,
            "an explicit marker is respected, never rewritten"
        );

        let credited =
            ModelMetadata::merged(Some(&disclosed), curated_model_metadata("gpt-4o").as_ref());
        assert_eq!(
            credited.source,
            MetadataSource::DisclosedAndCurated,
            "the table is never credited with a number it never contributed"
        );
        assert_eq!(
            credited.context_window,
            Some(32_768),
            "the host's stated number still wins"
        );

        let entry: ModelInfo =
            serde_json::from_str(r#"{"id":"x","metadata":{"context_window":8}}"#)
                .expect("a listing entry parses");
        assert_eq!(
            entry.metadata.source,
            MetadataSource::Disclosed,
            "the promotion reaches metadata nested in a listing entry too"
        );

        let round_tripped: ModelMetadata = serde_json::from_str(
            &serde_json::to_string(&ModelMetadata::unreported()).expect("metadata serializes"),
        )
        .expect("the crate's own serialization re-parses");
        assert_eq!(
            round_tripped.source,
            MetadataSource::Unreported,
            "the crate always writes the marker, so its own documents need no promotion"
        );
    }
}
