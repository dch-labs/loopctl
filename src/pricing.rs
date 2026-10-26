//! Price tables and cost math — estimated spend from reported usage.
//!
//! The cost plane's data half: a vendored, dated table of per-model
//! prices in US dollars per million tokens, override files that replace
//! or add rows without a release, and the cost math that turns one
//! [`Usage`](crate::stream::Usage) report into a per-class dollar
//! estimate. Everything here is
//! an *estimate for budgeting* — "warn me at five dollars" tolerates a
//! stale table; it does not tolerate a silent guess, so an unknown model
//! answers [`None`] and every estimate carries the table version and
//! pricing date it was computed under. Historical records stamped with a
//! table version never rewrite when prices change.
//!
//! Money is carried as **micro-dollars** (millionths of a US dollar,
//! `u64`) end to end: exact integer arithmetic with saturating bounds,
//! no floating-point drift between the math and the stamp. The one
//! decimal boundary is the table's own parse, where published
//! dollars-per-million rates scale once into micro-dollars per million
//! tokens.
//!
//! # Refreshing the vendored table
//!
//! The curated table is vendored data refreshed by a deliberate act:
//! fetch an [`OpenRouter`](https://openrouter.ai) models listing,
//! convert it, review the diff. One command pair, no hand-editing:
//!
//! ```text
//! curl -s https://openrouter.ai/api/v1/models \
//!   | cargo run --example import_openrouter_prices -- --date 2026-10-25 \
//!   > src/pricing/curated_prices.json
//! ```
//!
//! [`from_openrouter`](PriceTable::from_openrouter) does the conversion
//! (offline, pure — the fetch is the caller's deliberate act); the
//! resulting file is still pinned, dated data, so replayed stamps stay
//! comparable across the refresh.
//!
//! # Runtime freshness for hosts
//!
//! The vendored table is the zero-config floor, not the ceiling: a
//! long-lived binary (a CLI, a daemon) that wants day-fresh prices
//! without a rebuild owns the fetch and caches the conversion. The
//! ladder — a runtime fetch, the persisted cache, the baked table:
//!
//! ```text
//! fetch https://openrouter.ai/api/v1/models   (host policy: timeout, offline)
//!   -> PriceTable::from_openrouter(body, today_iso())
//!   -> install into the gate/ledger; persist with to_json() for offline boots
//! boot with no network: PriceTable::curated_with_cache(cache_dir)
//!   -> cache files over the baked table, a corrupt file skipped with a
//!      warning, a missing cache just the baked table — the ladder never
//!      fails a boot
//! ```
//!
//! One effective table per session — the gate and the ledger observer
//! price through the same one, so a run's stamps name a single
//! vintage.
//!
//! # Example
//!
//! Two same-priced runs produce identical estimates, and an unpriced
//! model answers [`None`] rather than a number:
//!
//! ```
//! use loopctl::pricing::PriceTable;
//! use loopctl::stream::Usage;
//!
//! let table = PriceTable::from_json(
//!     r#"{ "table_version": "demo", "pricing_updated": "2026-01-01",
//!         "models": { "example-family": { "input_per_mtok": 2.0,
//!                                         "output_per_mtok": 4.0 } } }"#,
//! )
//! .expect("a valid table");
//! let usage = Usage::new(50, 25);
//!
//! let priced = table.cost("example-family", usage)
//!     .expect("the family carries a row");
//! assert_eq!(priced.model, "example-family");
//! assert!(!priced.table_version.is_empty());
//!
//! assert!(table.cost("my-fine-tune-9000", usage).is_none());
//! ```

use std::path::Path;
use std::sync::LazyLock;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// How many days past `pricing_updated` a table counts as stale.
///
/// The threshold the cost plane warns at: a table older than this still
/// prices (its stamps stay comparable), but consumers surface the age
/// beside every rollup so nobody mistakes last quarter's rates for this
/// quarter's.
pub const STALE_AFTER_DAYS: u64 = 90;

/// Dollars, as the decimal a user or manifest carries, into micro-dollars.
///
/// The one boundary between user-facing decimal dollars and the
/// pipeline's integer micro-dollars: a finite, positive amount scales
/// by a million exactly (well inside `f64`'s integer-exact range) and
/// parses back as `u64`. Infinity reads as `u64::MAX` — an unbounded
/// budget is no ceiling, not an instant refusal — while negative and
/// NaN read as zero, and anything past `9.0e15` micro-dollars (the
/// edge of `f64`'s exactly-representable integers, far past any real
/// budget) saturates at `u64::MAX`. The budget gate's USD line and
/// any burn display render through it.
#[must_use]
pub fn micro_usd(dollars: f64) -> u64 {
    if dollars == f64::INFINITY {
        return u64::MAX;
    }
    if !dollars.is_finite() || dollars <= 0.0 {
        return 0;
    }
    let micro = dollars * 1_000_000.0;
    if micro >= 9.0e15 {
        return u64::MAX;
    }
    format!("{micro:.0}").parse().unwrap_or(u64::MAX)
}

/// Everything that can go wrong loading a price table or its overrides.
///
/// Construction-time only — pricing itself never fails, it answers
/// [`None`] — so the variants name the file and the reason, ready for a
/// host's `prices` verb to surface verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PriceTableError {
    /// A table file could not be read.
    ///
    /// Carries the path and the OS reason; a missing override directory
    /// lands here like any other unreadable path.
    #[error("reading price table `{path}`: {reason}")]
    Io {
        /// The path that failed to read.
        ///
        /// Rendered into the error verbatim, so a host's `prices` verb
        /// can name the exact file that failed.
        path: String,

        /// The underlying failure, rendered.
        ///
        /// The OS's own text; carried as a string so the error stays
        /// `Clone + Eq` for tests and comparisons.
        reason: String,
    },

    /// A table file is not valid JSON.
    ///
    /// Carries the path and the parser's message.
    #[error("parsing price table `{path}`: {reason}")]
    Json {
        /// The path that failed to parse.
        ///
        /// The offending file's path, verbatim.
        path: String,

        /// The parser's message.
        ///
        /// Where in the text the JSON broke, as the parser reported it.
        reason: String,
    },

    /// A table parsed but violates the format's invariants.
    ///
    /// An empty version, a malformed `pricing_updated` date, or a rate
    /// that is negative, non-finite, or too large to scale into
    /// micro-dollars.
    #[error("invalid price table `{path}`: {reason}")]
    Invalid {
        /// The path whose content is invalid.
        ///
        /// The file that parsed but broke a table invariant.
        path: String,

        /// Which invariant broke, and the offending value.
        ///
        /// Names the field and value that broke it, so fixing a
        /// hand-edited override file is mechanical.
        reason: String,
    },
}

/// One model family's rates, in micro-dollars per million tokens.
///
/// The integer-scaled form of the published dollars-per-million figures:
/// `2_500_000` is $2.50 per million input tokens. Scaled once at parse
/// so every later computation is exact saturating integer math. The two
/// cache columns are [`Option`]s because providers price caching
/// asymmetrically — the Anthropic family discounts reads to a tenth and
/// charges a write premium, the OpenAI family halves reads and has no
/// write class — and a family whose pricing has no cache position
/// simply leaves the column out.
///
/// A missing column is never a discount: reported cache traffic against
/// a row without the matching column prices at the base input rate, so
/// the estimate bends toward overcharging, never under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPrice {
    /// Uncached input tokens, per million.
    ///
    /// The base rate every input-class fallback lands on.
    pub input_micro_per_mtok: u64,

    /// Output tokens, per million.
    ///
    /// The full output rate; a reasoning share is inside the output
    /// count and prices here too.
    pub output_micro_per_mtok: u64,

    /// Input tokens served from the provider's cache, per million.
    ///
    /// `None` when the family publishes no cache-read price; reported
    /// cached tokens then price at the base input rate.
    pub cache_read_micro_per_mtok: Option<u64>,

    /// Tokens written to the provider's cache, per million.
    ///
    /// `None` when the family charges no write premium; reported write
    /// traffic then prices at the base input rate.
    pub cache_write_micro_per_mtok: Option<u64>,
}

/// What one usage report cost under the table that priced it.
///
/// Every figure is micro-dollars, every field carries its class, and
/// the stamp fields — the priced model, the table version, the pricing
/// date — are what a historical rollup groups by so last month's spend
/// keeps last month's prices.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CostBreakdown {
    /// The table row's family key that priced the report.
    ///
    /// The normalized family id (`gpt-4o` for a dated snapshot id), so
    /// rollups group families rather than snapshot spellings.
    pub model: String,

    /// The version of the table that produced this estimate.
    ///
    /// Composed from the base table and any overrides; two stamps with
    /// the same version were priced by identical data.
    pub table_version: String,

    /// The pricing date the table carries.
    ///
    /// ISO `YYYY-MM-DD`; the date the table's rates were believed
    /// current, which staleness is measured against.
    pub pricing_updated: String,

    /// Uncached input tokens' cost.
    ///
    /// The cache-aware input total less the read and write shares,
    /// priced at the base input rate.
    pub input_usd_micro: u64,

    /// Output tokens' cost, reasoning share included.
    ///
    /// The whole output count at the output rate; a reasoning share
    /// is inside the count, never priced twice.
    pub output_usd_micro: u64,

    /// The cached-read share's cost.
    ///
    /// The reported cached tokens at the family's read rate — zero
    /// when the report carried no cached share, and at the base rate
    /// when the family's row carries no read column.
    pub cache_read_usd_micro: u64,

    /// The cache-write share's cost.
    ///
    /// The reported write tokens at the family's write premium — zero
    /// when the report carried no write share, and at the base rate
    /// when the family's row carries no write column.
    pub cache_write_usd_micro: u64,

    /// The four classes summed, saturating.
    ///
    /// The one number a burn display renders (divided by a million);
    /// the per-class fields keep the breakdown honest beside it.
    pub total_usd_micro: u64,
}

/// A dated, versioned set of per-family prices.
///
/// Built from the vendored curated table, a host's override files, or
/// both merged — and the one source every cost estimate and USD budget
/// line reads from. Lookup is prefix-based and case-insensitive, so
/// dated snapshot ids inherit their family's row and Bedrock's
/// region-prefixed spellings resolve to the same family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriceTable {
    /// The identity of this data, composed across merges.
    ///
    /// `loopctl-<date>` for the curated table; `base+override` after a
    /// non-empty merge, so a stamp names exactly which data priced it.
    version: String,

    /// The date this table's rates were believed current, ISO.
    ///
    /// A merge takes the lexically greater date, so staleness reflects
    /// the freshest data actually in use, never an older base under a
    /// newer override.
    pricing_updated: String,

    /// The rows, keyed by family prefix.
    ///
    /// Scanned longest-prefix-first at lookup; order-independent, so
    /// JSON object order never affects pricing.
    rows: Vec<(String, ModelPrice)>,
}

impl PriceTable {
    /// The crate's vendored table.
    ///
    /// Parsed once from the JSON compiled into the crate. The vendored
    /// seed is a conversion of [`OpenRouter`](https://openrouter.ai)'s
    /// public models listing on `pricing_updated`'s date — hundreds of
    /// current models across every provider the listing carries
    /// (`from_openrouter` documents the conversion's key, alias, and
    /// skip rules) — so a fresh install prices what people actually
    /// run without configuring anything. The staleness check reports
    /// the table's age from that date until a refresh swaps in newer
    /// data: the release-cut ritual, or the drift canary opening a
    /// refresh PR. A parse failure — impossible for a shipped table,
    /// possible for a hand-edited build — degrades to an error-logged
    /// empty table dated at the Unix epoch, which prices nothing and
    /// reads as maximally stale. Degrading loudly beats panicking in
    /// a library.
    #[must_use]
    pub fn curated() -> &'static Self {
        &CURATED
    }

    /// The baked table with a host's price cache layered over it.
    ///
    /// The read half of the runtime-freshness ladder: every `*.json`
    /// in `dir`, in filename order, merges over
    /// [`curated()`](Self::curated) exactly as
    /// [`with_overrides`](Self::with_overrides) prescribes. The ladder
    /// never fails a boot — a missing directory is the first-run norm
    /// (the baked table answers alone, silently), an unreadable or
    /// corrupt file is skipped with a warning naming it while the rest
    /// of the cache still applies, and only a directory that exists
    /// and cannot be listed warns whole. Hosts persist a fetched
    /// table's [`to_json`](Self::to_json) output into this directory
    /// to serve the next offline boot at the fetched vintage.
    ///
    /// # Example
    ///
    /// ```
    /// use std::fs;
    /// use loopctl::pricing::PriceTable;
    ///
    /// let dir = std::env::temp_dir()
    ///     .join(format!("loopctl-pricing-doctest-{}", std::process::id()));
    /// fs::create_dir_all(&dir).expect("cache dir");
    /// fs::write(
    ///     dir.join("00-refresh.json"),
    ///     r#"{ "table_version": "refresh", "pricing_updated": "2999-01-01",
    ///         "models": { "private-deploy": { "input_per_mtok": 0.5,
    ///                                         "output_per_mtok": 0.5 } } }"#,
    /// )
    /// .expect("persist");
    ///
    /// let table = PriceTable::curated_with_cache(&dir);
    /// assert!(
    ///     table.price("private-deploy").is_some(),
    ///     "the cache layered a family the baked table never carried"
    /// );
    /// assert!(table.version().starts_with("openrouter-"), "the version composes over the baked identity");
    /// fs::remove_dir_all(&dir).ok();
    /// ```
    #[must_use]
    pub fn curated_with_cache(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Self::curated().clone();
            }
            Err(error) => {
                tracing::warn!(
                    target: "loopctl::pricing",
                    dir = %dir.display(),
                    reason = %error,
                    "the price cache directory could not be listed; the baked table answers"
                );
                return Self::curated().clone();
            }
        };
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for entry in entries {
            match entry {
                Ok(entry) => {
                    if entry.path().extension().is_some_and(|ext| ext == "json") {
                        files.push(entry.path());
                    }
                }
                Err(error) => tracing::warn!(
                    target: "loopctl::pricing",
                    dir = %dir.display(),
                    reason = %error,
                    "a cache directory entry could not be read; the rest still applies"
                ),
            }
        }
        files.sort();
        let mut merged = Self::curated().clone();
        for file in files {
            match Self::from_file(&file) {
                Ok(table) => merged = merged.with_overrides(&table),
                Err(error) => tracing::warn!(
                    target: "loopctl::pricing",
                    file = %file.display(),
                    error = %error,
                    "skipping an unreadable price cache file; the rest of the cache still applies"
                ),
            }
        }
        merged
    }

    /// Parse a table from its JSON text.
    ///
    /// The interchange format for the curated table and override files
    /// alike: `{ "table_version", "pricing_updated", "models": { family:
    /// { "input_per_mtok", "output_per_mtok", "cache_read_per_mtok"?,
    /// "cache_write_per_mtok"? } } }`, rates in US dollars per million
    /// tokens. Unknown keys within this shape are ignored, so a host's
    /// extra columns ride along harmlessly; a foreign cost-map layout
    /// does not parse — host numbers go in this shape, which the
    /// [`OpenRouter`](https://openrouter.ai) importer renders.
    ///
    /// # Errors
    ///
    /// [`PriceTableError::Json`] when the text is not valid JSON, and
    /// [`PriceTableError::Invalid`] when it parses but breaks the
    /// format's invariants: an empty version, a `pricing_updated` that
    /// is not an ISO `YYYY-MM-DD` date, or a rate that is negative,
    /// non-finite, or too large to scale into micro-dollars.
    pub fn from_json(text: &str) -> Result<Self, PriceTableError> {
        Self::parse(text, "<inline>")
    }

    /// Load a table from one JSON file on disk.
    ///
    /// The single-file form of [`load_overrides_dir`](Self::load_overrides_dir);
    /// errors carry the path.
    ///
    /// # Errors
    ///
    /// [`PriceTableError::Io`] when the file cannot be read;
    /// [`Json`](PriceTableError::Json) and [`Invalid`](PriceTableError::Invalid)
    /// as in [`from_json`](Self::from_json), both naming the path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PriceTableError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|error| PriceTableError::Io {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Self::parse(&text, &path.display().to_string())
    }

    /// Build a table from an [`OpenRouter`](https://openrouter.ai) models listing.
    ///
    /// The refresh path for the vendored table: [`OpenRouter`](https://openrouter.ai)'s public
    /// `GET /api/v1/models` carries, for hundreds of models across
    /// dozens of providers, per-token USD prices in its `pricing`
    /// object — `prompt`, `completion`, and the `input_cache_read` /
    /// `input_cache_write` columns where the provider prices caching.
    /// This converter turns that listing into a table offline and
    /// deterministically: prices scale once from per-token to
    /// per-million, every kept model lands under **both** its bare id
    /// (`gpt-6.1-sol`, what a first-party client sends) and its
    /// vendor-prefixed spelling (`openai/gpt-6.1-sol`, what an
    /// [`OpenRouter`](https://openrouter.ai) client sends), and dotted spellings additionally
    /// carry their dashed aliases — [`OpenRouter`](https://openrouter.ai) writes
    /// `anthropic/claude-opus-5.5` where Anthropic's own API ids read
    /// `claude-opus-5-5-…`, so both worlds resolve to the same row.
    /// Rows price the **base tier** — a
    /// provider's tiered long-context overrides
    /// (`overrides[].min_prompt_tokens`) are not representable in a
    /// flat row, so prompts past a tier boundary underprice against
    /// that provider's own ladder; hosts on the boundary override the
    /// row. Skipped, deliberately: `:free` and `:batch` variants
    /// (different products, not price estimates), `~`-prefixed aliases,
    /// and entries with no positive prompt-and-completion price. Where
    /// two providers expose the same bare id at the same listing, the
    /// later row wins — collisions are none in practice today and the
    /// rule is deterministic either way. The table's version is
    /// `openrouter-<pricing_date>` and its date is `pricing_date`, so
    /// every stamp names the source and vintage it was priced under.
    ///
    /// # Errors
    ///
    /// [`PriceTableError::Json`] when the text is not valid JSON, and
    /// [`PriceTableError::Invalid`] when `pricing_date` is not an ISO
    /// `YYYY-MM-DD` date.
    pub fn from_openrouter(models_json: &str, pricing_date: &str) -> Result<Self, PriceTableError> {
        if iso_to_epoch_days(pricing_date).is_none() {
            return Err(PriceTableError::Invalid {
                path: "<openrouter>".to_string(),
                reason: format!("pricing date `{pricing_date}` is not an ISO YYYY-MM-DD date"),
            });
        }
        let listing: OpenRouterListing =
            serde_json::from_str(models_json).map_err(|error| PriceTableError::Json {
                path: "<openrouter>".to_string(),
                reason: error.to_string(),
            })?;
        let mut rows: Vec<(String, ModelPrice)> = Vec::new();
        for model in listing.data {
            let Some(price) = openrouter_price(&model) else {
                continue;
            };
            let id = model.id.trim().to_ascii_lowercase();
            let Some((vendor, bare)) = id.split_once('/') else {
                continue;
            };
            let dashed = bare.replace('.', "-");
            let mut keys = vec![bare.to_string(), id.clone()];
            if dashed != bare {
                keys.push(format!("{vendor}/{dashed}"));
                keys.push(dashed);
            }
            for key in keys {
                match rows.iter_mut().find(|(row_key, _)| *row_key == key) {
                    Some(existing) => existing.1 = price,
                    None => rows.push((key, price)),
                }
            }
        }
        Ok(Self {
            version: format!("openrouter-{pricing_date}"),
            pricing_updated: pricing_date.to_string(),
            rows,
        })
    }

    /// Render the table as its interchange-format JSON.
    ///
    /// The inverse of [`from_json`](Self::from_json): the same shape
    /// the curated table and override files use, model keys sorted, so
    /// a converted table (`from_openrouter` … `to_json`) diffs cleanly
    /// against the vendored file at refresh time and round-trips
    /// losslessly.
    ///
    /// # Errors
    ///
    /// Never in practice — the table's own fields are always
    /// serializable; the `Result` shape serves the JSON boundary's
    /// convention.
    pub fn to_json(&self) -> Result<String, PriceTableError> {
        let mut models = std::collections::BTreeMap::new();
        for (family, price) in &self.rows {
            let mut row = serde_json::Map::new();
            row.insert(
                "input_per_mtok".to_string(),
                micro_to_decimal(price.input_micro_per_mtok),
            );
            row.insert(
                "output_per_mtok".to_string(),
                micro_to_decimal(price.output_micro_per_mtok),
            );
            if let Some(rate) = price.cache_read_micro_per_mtok {
                row.insert("cache_read_per_mtok".to_string(), micro_to_decimal(rate));
            }
            if let Some(rate) = price.cache_write_micro_per_mtok {
                row.insert("cache_write_per_mtok".to_string(), micro_to_decimal(rate));
            }
            models.insert(family.clone(), Value::Object(row));
        }
        let table = serde_json::json!({
            "table_version": self.version,
            "pricing_updated": self.pricing_updated,
            "models": models,
        });
        serde_json::to_string_pretty(&table).map_err(|error| PriceTableError::Json {
            path: "<render>".to_string(),
            reason: error.to_string(),
        })
    }

    /// Load and merge every `*.json` file in a directory, by name.
    ///
    /// The `prices.d/` shape: each file is a table in the same format,
    /// applied over the previous in filename order (deterministic on
    /// every filesystem), later files winning key collisions. The
    /// result is the overrides table — empty when the directory holds
    /// no JSON — ready to merge over the curated table with
    /// [`with_overrides`](Self::with_overrides).
    ///
    /// # Errors
    ///
    /// [`PriceTableError::Io`] when the directory cannot be read or a
    /// directory entry cannot be enumerated (both name the directory);
    /// [`Json`](PriceTableError::Json) and [`Invalid`](PriceTableError::Invalid)
    /// naming the offending file.
    pub fn load_overrides_dir(path: impl AsRef<Path>) -> Result<Self, PriceTableError> {
        let path = path.as_ref();
        let entries = std::fs::read_dir(path).map_err(|error| PriceTableError::Io {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| PriceTableError::Io {
                path: path.display().to_string(),
                reason: format!("reading a directory entry: {error}"),
            })?;
            if entry.path().extension().is_some_and(|ext| ext == "json") {
                files.push(entry.path());
            }
        }
        files.sort();
        let mut merged = Self::empty();
        for file in files {
            let table = Self::from_file(&file)?;
            merged = merged.with_overrides(&table);
        }
        Ok(merged)
    }

    /// This table with `overrides` applied over it.
    ///
    /// Rows replace by exact key and new keys add — an override for a
    /// dated id (`gpt-4o-2024-11-20`) is simply a more specific prefix,
    /// so exact keys and prefix lookup compose. The merged version is
    /// `base+override` when the overrides carry rows (the curated
    /// identity unchanged otherwise), and `pricing_updated` takes the
    /// lexically greater date so staleness tracks the freshest data in
    /// use.
    #[must_use]
    pub fn with_overrides(&self, overrides: &Self) -> Self {
        if overrides.rows.is_empty() {
            return self.clone();
        }
        let mut rows = self.rows.clone();
        for (key, price) in &overrides.rows {
            if let Some(existing) = rows.iter_mut().find(|(row_key, _)| row_key == key) {
                existing.1 = *price;
            } else {
                rows.push((key.clone(), *price));
            }
        }
        Self {
            version: format!("{}+{}", self.version, overrides.version),
            pricing_updated: if overrides.pricing_updated > self.pricing_updated {
                overrides.pricing_updated.clone()
            } else {
                self.pricing_updated.clone()
            },
            rows,
        }
    }

    /// Whether two tables carry identical rows, ignoring vintage.
    ///
    /// The drift check's comparator: `true` when every family key and
    /// price matches regardless of `table_version` or
    /// `pricing_updated`, compared order-insensitively (the vendored
    /// file and a fresh conversion order rows differently by
    /// construction). A nightly consumer converts the current listing,
    /// compares against the checked-in table, and a date-only
    /// difference reads as unchanged — no refresh PR without a real
    /// price or family change.
    #[must_use]
    pub fn same_prices(&self, other: &Self) -> bool {
        if self.rows.len() != other.rows.len() {
            return false;
        }
        let mut left = self.rows.clone();
        let mut right = other.rows.clone();
        left.sort_by(|a, b| a.0.cmp(&b.0));
        right.sort_by(|a, b| a.0.cmp(&b.0));
        left == right
    }

    /// How many family rows the table carries, every spelling counted.
    ///
    /// The counted basis of the refresh fence: the importer refuses a
    /// conversion that lands zero rows or drops most of the rows the
    /// checked table carries, so the figure counts spellings the way
    /// the fence reads them — a dual-keyed family is two rows.
    /// [`same_prices`](Self::same_prices) answers equality; this
    /// answers scale.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The empty table: no rows, epoch-dated.
    ///
    /// The additive identity of [`with_overrides`](Self::with_overrides)
    /// and the curated loader's degradation target — prices nothing,
    /// reads as maximally stale.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            version: "empty".to_string(),
            pricing_updated: "1970-01-01".to_string(),
            rows: Vec::new(),
        }
    }

    /// This table's version identity.
    ///
    /// Stamped into every estimate and USD refusal; two stamps sharing
    /// a version were priced by identical data.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The date this table's rates were believed current, ISO.
    ///
    /// Staleness is measured from here; a merge keeps the freshest
    /// contributor's date.
    #[must_use]
    pub fn pricing_updated(&self) -> &str {
        &self.pricing_updated
    }

    /// The price row a model id resolves to, if any.
    ///
    /// Case-insensitive, longest-prefix-wins over the row keys, after
    /// trimming Bedrock's region and provider prefixes
    /// (`us.anthropic.`, `meta.`, …), so a dated or regionally spelled
    /// id inherits its family's rates. An id no row matches — a
    /// fine-tune, a private deployment, a model this table predates —
    /// answers [`None`]: the honest default, never a nearest guess.
    #[must_use]
    pub fn price(&self, model: &str) -> Option<&ModelPrice> {
        self.row_for(model).map(|(_, price)| price)
    }

    /// The row a model id resolves to, with its family key.
    ///
    /// The shared resolver behind [`price`](Self::price) and
    /// [`cost`](Self::cost), so a stamp and its lookup can never disagree
    /// on which row priced the report.
    fn row_for(&self, model: &str) -> Option<(&str, &ModelPrice)> {
        let id = normalize_model_id(model);
        self.rows
            .iter()
            .filter(|(prefix, _)| id.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(prefix, price)| (prefix.as_str(), price))
    }

    /// Whole days between `pricing_updated` and `as_of`, when both parse.
    ///
    /// `Some(0)` covers an `as_of` at or before the pricing date (a
    /// future-dated table is fresh, not negative); [`None`] when either
    /// date is not a valid ISO `YYYY-MM-DD`, leaving the caller's
    /// staleness judgment to its own default.
    #[must_use]
    pub fn days_stale(&self, as_of: &str) -> Option<u64> {
        let updated = iso_to_epoch_days(&self.pricing_updated)?;
        let as_of_days = iso_to_epoch_days(as_of)?;
        u64::try_from(as_of_days.saturating_sub(updated).max(0)).ok()
    }

    /// Whether the table is older than [`STALE_AFTER_DAYS`] as of a date.
    ///
    /// Stale tables still price — their stamps stay internally
    /// comparable — the flag exists so consumers surface the age; an
    /// unparseable `as_of` answers `false` rather than guessing stale.
    #[must_use]
    pub fn is_stale(&self, as_of: &str) -> bool {
        self.days_stale(as_of)
            .is_some_and(|days| days > STALE_AFTER_DAYS)
    }

    /// Estimate what one usage report cost under this table.
    ///
    /// Splits the reported usage superset into its priced classes: the
    /// uncached remainder of the cache-aware input total at the base
    /// rate, the cached-read share at the read rate, the cache-write
    /// share at the write rate, and output at the output rate — a
    /// missing cache column prices its class at the base input rate, so
    /// the error bends toward overcharging. An unpriced model answers
    /// [`None`]; that is the whole no-guess contract.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::pricing::PriceTable;
    /// use loopctl::stream::Usage;
    ///
    /// let table = PriceTable::from_json(
    ///     r#"{ "table_version": "demo", "pricing_updated": "2026-01-01",
    ///         "models": { "example-family": { "input_per_mtok": 2.5,
    ///                                         "output_per_mtok": 10.0 } } }"#,
    /// )
    /// .expect("a valid table");
    /// let priced = table
    ///     .cost("example-family", Usage::new(1_000_000, 0))
    ///     .expect("the family carries a row");
    /// assert_eq!(priced.input_usd_micro, 2_500_000);
    /// ```
    #[must_use]
    pub fn cost(&self, model: &str, usage: crate::stream::Usage) -> Option<CostBreakdown> {
        let (family, price) = self.row_for(model)?;
        let input = u64::from(usage.input_tokens);
        let output = u64::from(usage.output_tokens);
        let cached_read = u64::from(usage.cached_input_tokens);
        let cache_write = u64::from(usage.cache_write_tokens);
        let cached_read = cached_read.min(input);
        let cache_write = cache_write.min(input.saturating_sub(cached_read));
        let uncached = input
            .saturating_sub(cached_read)
            .saturating_sub(cache_write);
        let input_usd_micro = rate_apply(uncached, price.input_micro_per_mtok);
        let cache_read_usd_micro = rate_apply(
            cached_read,
            price
                .cache_read_micro_per_mtok
                .unwrap_or(price.input_micro_per_mtok),
        );
        let cache_write_usd_micro = rate_apply(
            cache_write,
            price
                .cache_write_micro_per_mtok
                .unwrap_or(price.input_micro_per_mtok),
        );
        let output_usd_micro = rate_apply(output, price.output_micro_per_mtok);
        let total_usd_micro = input_usd_micro
            .saturating_add(cache_read_usd_micro)
            .saturating_add(cache_write_usd_micro)
            .saturating_add(output_usd_micro);
        Some(CostBreakdown {
            model: family.to_string(),
            table_version: self.version.clone(),
            pricing_updated: self.pricing_updated.clone(),
            input_usd_micro,
            output_usd_micro,
            cache_read_usd_micro,
            cache_write_usd_micro,
            total_usd_micro,
        })
    }

    /// Parse and validate table JSON against a named source.
    ///
    /// The single construction path for every file-backed table; the
    /// name rides into errors so a bad override file identifies itself.
    ///
    /// # Errors
    ///
    /// [`PriceTableError::Json`] when the text does not parse, and
    /// [`PriceTableError::Invalid`] for every broken invariant, both
    /// naming `source`.
    fn parse(text: &str, source: &str) -> Result<Self, PriceTableError> {
        let raw: RawTable = serde_json::from_str(text).map_err(|error| PriceTableError::Json {
            path: source.to_string(),
            reason: error.to_string(),
        })?;
        if raw.table_version.trim().is_empty() {
            return Err(PriceTableError::Invalid {
                path: source.to_string(),
                reason: "table_version is empty".to_string(),
            });
        }
        if iso_to_epoch_days(&raw.pricing_updated).is_none() {
            return Err(PriceTableError::Invalid {
                path: source.to_string(),
                reason: format!(
                    "pricing_updated `{}` is not an ISO YYYY-MM-DD date",
                    raw.pricing_updated
                ),
            });
        }
        let mut rows = Vec::new();
        for (family, row) in raw.models {
            let key = normalize_model_id(&family);
            if key.is_empty() {
                return Err(PriceTableError::Invalid {
                    path: source.to_string(),
                    reason: format!("model key `{family}` normalizes to nothing"),
                });
            }
            let price = ModelPrice {
                input_micro_per_mtok: scale_rate(row.input_per_mtok, source, "input_per_mtok")?,
                output_micro_per_mtok: scale_rate(row.output_per_mtok, source, "output_per_mtok")?,
                cache_read_micro_per_mtok: row
                    .cache_read_per_mtok
                    .map(|rate| scale_rate(rate, source, "cache_read_per_mtok"))
                    .transpose()?,
                cache_write_micro_per_mtok: row
                    .cache_write_per_mtok
                    .map(|rate| scale_rate(rate, source, "cache_write_per_mtok"))
                    .transpose()?,
            };
            if let Some(existing) = rows.iter_mut().find(|(row_key, _)| *row_key == key) {
                existing.1 = price;
            } else {
                rows.push((key, price));
            }
        }
        Ok(Self {
            version: raw.table_version,
            pricing_updated: raw.pricing_updated,
            rows,
        })
    }
}

static CURATED: LazyLock<PriceTable> = LazyLock::new(|| {
    PriceTable::from_json(include_str!("pricing/curated_prices.json")).unwrap_or_else(|error| {
        tracing::error!(
            target: "loopctl::pricing",
            error = %error,
            "the vendored price table failed to parse; every model answers unpriced until \
             a corrected table ships"
        );
        PriceTable::empty()
    })
});

/// The wire shape of one table file.
///
/// Rates arrive as the published dollars-per-million decimals and scale
/// into micro-dollars at parse; unknown keys within this shape are
/// ignored, so a host's extra columns ride along harmlessly.
#[derive(Deserialize)]
struct RawTable {
    /// The file's version identity.
    ///
    /// Rides into the parsed table's composed version.
    table_version: String,

    /// The date the file's rates were believed current.
    ///
    /// ISO `YYYY-MM-DD`, validated at parse; staleness measures from
    /// it.
    pricing_updated: String,

    /// Family-keyed rows.
    ///
    /// A map so duplicate keys in one file collapse last-wins at
    /// parse rather than arriving twice.
    models: std::collections::BTreeMap<String, RawPrice>,
}

/// The wire shape of one family's rates.
///
/// Dollars per million tokens, exactly as published; parsing scales
/// into micro-dollars once at the boundary.
#[derive(Deserialize)]
struct RawPrice {
    /// Dollars per million uncached input tokens.
    ///
    /// Required — a row without a base input rate is not a price.
    input_per_mtok: f64,

    /// Dollars per million output tokens.
    ///
    /// Required, same reasoning as the input rate.
    output_per_mtok: f64,

    /// Dollars per million cached-read tokens, when the family prices one.
    ///
    /// `None` leaves the parsed row without a read column.
    cache_read_per_mtok: Option<f64>,

    /// Dollars per million cache-write tokens, when the family prices one.
    ///
    /// `None` leaves the parsed row without a write column.
    cache_write_per_mtok: Option<f64>,
}

/// The wire shape of [`OpenRouter`](https://openrouter.ai)'s models listing.
///
/// Only the fields the converter reads; everything else in the listing
/// is ignored by omission, exactly as unknown keys are in a table file.
#[derive(Deserialize)]
struct OpenRouterListing {
    /// The listing's models.
    ///
    /// Order preserved as delivered; the converter's later-row-wins
    /// collision rule is deterministic over it.
    data: Vec<OpenRouterModel>,
}

/// One model entry of [`OpenRouter`](https://openrouter.ai)'s models listing.
///
/// Only the id and pricing columns survive into the table; the
/// listing's capability and context metadata stays where it belongs
/// — metadata, not price.
#[derive(Deserialize)]
struct OpenRouterModel {
    /// The vendor-prefixed id (`openai/gpt-6.1-sol`).
    ///
    /// The one spelling the listing guarantees; every other key the
    /// table carries is derived from it at convert time.
    id: String,

    /// The per-token USD price columns.
    ///
    /// `None` on entries the listing carries without pricing — they
    /// become no row, never a zero-priced one.
    pricing: Option<OpenRouterPricing>,
}

/// The `pricing` object of one listing entry.
///
/// Every column arrives as a per-token USD string (numbers tolerated),
/// which is why each is an untyped [`serde_json::Value`] until
/// [`openrouter_price`] reads or skips it.
#[derive(Deserialize)]
struct OpenRouterPricing {
    /// Uncached input, per token.
    ///
    /// The string form the API documents; numbers tolerated.
    prompt: Option<Value>,

    /// Output, per token.
    ///
    /// Same string-or-number shape as the prompt column.
    completion: Option<Value>,

    /// Cached-read input, per token, where the provider prices one.
    ///
    /// Absent on providers with no cache-read price; the row then
    /// carries no read column and reported cache prices at base.
    #[serde(default)]
    input_cache_read: Option<Value>,

    /// Cache-write spend, per token, where the provider prices one.
    ///
    /// The short-tier write where the provider tiers them; see
    /// [`from_openrouter`](PriceTable::from_openrouter) for the tier caveat.
    #[serde(default)]
    input_cache_write: Option<Value>,
}

/// One listing entry's usable price, or why it has none.
///
/// The skip rules live here: variants (`:free`, `:batch`), `~`
/// aliases, and entries without a positive prompt-and-completion pair
/// are not price estimates and never become rows.
fn openrouter_price(model: &OpenRouterModel) -> Option<ModelPrice> {
    let id = model.id.trim();
    if id.contains(':') || id.starts_with('~') {
        return None;
    }
    let pricing = model.pricing.as_ref()?;
    let prompt = per_mtok(pricing.prompt.as_ref())?;
    let completion = per_mtok(pricing.completion.as_ref())?;
    if prompt <= 0.0 || completion <= 0.0 {
        return None;
    }
    Some(ModelPrice {
        input_micro_per_mtok: scale_rate(prompt, "<openrouter>", "prompt").ok()?,
        output_micro_per_mtok: scale_rate(completion, "<openrouter>", "completion").ok()?,
        cache_read_micro_per_mtok: per_mtok(pricing.input_cache_read.as_ref())
            .filter(|rate| *rate > 0.0)
            .and_then(|rate| scale_rate(rate, "<openrouter>", "input_cache_read").ok()),
        cache_write_micro_per_mtok: per_mtok(pricing.input_cache_write.as_ref())
            .filter(|rate| *rate > 0.0)
            .and_then(|rate| scale_rate(rate, "<openrouter>", "input_cache_write").ok()),
    })
}

/// A listing price column as per-million dollars, when it states one.
///
/// The column may arrive as a string (the documented shape) or a bare
/// number; empty strings and unparseable text read as absent.
fn per_mtok(column: Option<&Value>) -> Option<f64> {
    let text = match column? {
        Value::String(text) => text.trim().to_string(),
        Value::Number(number) => number.to_string(),
        _ => return None,
    };
    if text.is_empty() {
        return None;
    }
    let per_token: f64 = text.parse().ok()?;
    Some(per_token * 1_000_000.0)
}

/// Micro-dollars as the interchange format's decimal dollars.
///
/// Integer division builds the decimal string (whole and fractional
/// parts, trailing zeros trimmed), and only the finished string meets
/// a float — so the value round-trips exactly through
/// [`to_json`](PriceTable::to_json) and [`from_json`](PriceTable::from_json)
/// for every micro figure a rate can produce.
fn micro_to_decimal(micro: u64) -> Value {
    let whole = micro / 1_000_000;
    let fraction = micro % 1_000_000;
    if fraction == 0 {
        return Value::from(whole);
    }
    let mut fraction_text = format!("{fraction:06}");
    while fraction_text.ends_with('0') {
        fraction_text.pop();
    }
    let decimal: f64 = format!("{whole}.{fraction_text}").parse().unwrap_or(0.0);
    Value::from(decimal)
}

/// Scale one published rate into micro-dollars per million tokens.
///
/// The pipeline's single decimal boundary: a finite, non-negative rate
/// at most `1e9` scales by a million and renders as a whole number —
/// exact, because the result stays far inside `f64`'s integer-exact
/// range — then parses back as `u64`. The string round-trip keeps the
/// conversion free of float-to-int casts while behaving exactly like a
/// checked `round()`.
///
/// # Errors
///
/// [`PriceTableError::Invalid`] naming `source` and `field` when the
/// rate is negative, non-finite, or too large to scale.
fn scale_rate(rate: f64, source: &str, field: &str) -> Result<u64, PriceTableError> {
    if !rate.is_finite() || rate < 0.0 {
        return Err(PriceTableError::Invalid {
            path: source.to_string(),
            reason: format!("{field} rate {rate} is negative or not a number"),
        });
    }
    if rate > 1e9 {
        return Err(PriceTableError::Invalid {
            path: source.to_string(),
            reason: format!("{field} rate {rate} is too large to scale into micro-dollars"),
        });
    }
    let micro = rate * 1_000_000.0;
    Ok(format!("{micro:.0}").parse().unwrap_or(u64::MAX))
}

/// Apply a micro-per-million rate to a token count, truncating each
/// class to whole micro-dollars.
///
/// Tokens × rate ÷ a million, in the order that cannot overflow before
/// the division lands: the multiply saturates at `u64`'s top and the
/// divide brings it back, so the only loss is the sub-micro-dollar
/// remainder of a single class — never a panic, never a wrap.
fn rate_apply(tokens: u64, micro_per_mtok: u64) -> u64 {
    let scaled = tokens.saturating_mul(micro_per_mtok);
    scaled / 1_000_000
}

/// Normalize a model id for family lookup.
///
/// Lowercase, trimmed, and stripped of Bedrock's regional and provider
/// prefixes — `us.anthropic.claude-3-5-sonnet-v2:0` and
/// `claude-3-5-sonnet-20241022` both look up as the bare family — so
/// one spelling of a family prices like every other.
fn normalize_model_id(model: &str) -> String {
    let mut id = model.trim().to_ascii_lowercase();
    loop {
        let stripped = [
            "us.",
            "eu.",
            "apac.",
            "global.",
            "anthropic.",
            "meta.",
            "amazon.",
        ]
        .into_iter()
        .find_map(|prefix| id.strip_prefix(prefix));
        match stripped {
            Some(rest) => id = rest.to_string(),
            None => break,
        }
    }
    id
}

/// Today's UTC date as an ISO `YYYY-MM-DD` string.
///
/// The stamp date for a runtime-fetched listing: a host that fetched
/// [`OpenRouter`](https://openrouter.ai)'s models just now converts with
/// `from_openrouter(listing, today_iso())`, and the table prices — and
/// stamps — under the day it was actually fetched. Reads the system
/// clock once per call; a host with its own clock source passes its
/// own date instead.
#[must_use]
pub fn today_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = i64::try_from(secs).unwrap_or(i64::MAX).div_euclid(86_400);
    epoch_days_to_iso(days).unwrap_or_else(|| "1970-01-01".to_string())
}

/// Days since the Unix epoch as an ISO date.
///
/// Howard Hinnant's `civil_from_days` — the inverse of
/// [`iso_to_epoch_days`] — over checked arithmetic; [`None`] only
/// where an intermediate overflows, which realistic clock values
/// cannot reach.
fn epoch_days_to_iso(days: i64) -> Option<String> {
    let z = days.checked_add(719_468)?;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = doe
        .checked_sub(doe.div_euclid(1_460))?
        .checked_add(doe.div_euclid(36_524))?
        .checked_sub(doe.div_euclid(146_096))?
        .div_euclid(365);
    let y = yoe.checked_add(era.checked_mul(400)?)?;
    let doy = doe.checked_sub(
        yoe.checked_mul(365)?
            .checked_add(yoe.div_euclid(4))?
            .checked_sub(yoe.div_euclid(100))?,
    )?;
    let mp = doy.checked_mul(5)?.checked_add(2)?.div_euclid(153);
    let day = doy
        .checked_sub(mp.checked_mul(153)?.checked_add(2)?.div_euclid(5))?
        .checked_add(1)?;
    let (month, year) = if mp < 10 {
        (mp.checked_add(3)?, y)
    } else {
        (mp.checked_sub(9)?, y.checked_add(1)?)
    };
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

/// Parse an ISO `YYYY-MM-DD` date into days since the Unix epoch.
///
/// Howard Hinnant's `days_from_civil` over checked arithmetic, the
/// inverse of [`epoch_days_to_iso`]; [`None`] when the text is not a
/// well-formed calendar date — right shape *and* a day the month
/// actually has, so `2026-02-31` rejects rather than rolling into
/// March — which is the caller's signal to treat the input as
/// unparseable rather than guessing.
fn iso_to_epoch_days(date: &str) -> Option<i64> {
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = civil_parts_to_epoch_days(year, month, day)?;
    (epoch_days_to_iso(days)? == date).then_some(days)
}

/// The civil-from-days arithmetic behind [`iso_to_epoch_days`].
///
/// Split out so the validator can compare the caller's text against
/// the date the arithmetic actually lands on — the month-length check
/// that needs no calendar table of its own.
fn civil_parts_to_epoch_days(year: i64, month: i64, day: i64) -> Option<i64> {
    let y = if month <= 2 {
        year.checked_sub(1)?
    } else {
        year
    };
    let era = if y >= 0 { y } else { y.checked_sub(399)? }.div_euclid(400);
    let yoe = y.checked_sub(era.checked_mul(400)?)?;
    let mp = month.checked_add(9)? % 12;
    let doy = mp
        .checked_mul(153)?
        .checked_add(2)?
        .div_euclid(5)
        .checked_add(day)?
        .checked_sub(1)?;
    yoe.checked_mul(365)?
        .checked_add(yoe.div_euclid(4))?
        .checked_sub(yoe.div_euclid(100))?
        .checked_add(doy)?
        .checked_add(era.checked_mul(146_097)?)?
        .checked_sub(719_468)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Usage;

    #[test]
    fn invalid_calendar_dates_reject_instead_of_rolling_forward() {
        for date in [
            "2026-02-31",
            "2026-04-31",
            "2026-06-31",
            "2026-09-31",
            "2026-11-31",
            "2025-02-29",
            "2026-13-01",
            "2026-00-10",
            "2026-01-00",
            "2026-01-32",
            "2026-1-1",
            "2026-01-1x",
            "20260101",
            "2026-01-01-01",
        ] {
            assert_eq!(
                iso_to_epoch_days(date),
                None,
                "{date} is not a day any month has — reject, never roll forward"
            );
        }
        for date in [
            "2024-02-29",
            "2000-02-29",
            "1600-02-29",
            "2026-02-28",
            "2026-10-25",
        ] {
            assert!(
                iso_to_epoch_days(date).is_some(),
                "{date} is a real calendar day and parses"
            );
        }
    }

    #[test]
    fn micro_usd_scales_exactly_and_saturates() {
        assert_eq!(
            micro_usd(0.000_375),
            375,
            "a turn's estimate in exact micro-dollars"
        );
        assert_eq!(micro_usd(5.0), 5_000_000, "a five-dollar budget line");
        assert_eq!(micro_usd(0.0), 0, "zero stays zero");
        assert_eq!(micro_usd(-1.0), 0, "negative reads as zero, never wraps");
        assert_eq!(micro_usd(f64::NAN), 0, "NaN reads as zero");
        assert_eq!(
            micro_usd(f64::INFINITY),
            u64::MAX,
            "an unbounded budget is no ceiling, not an instant refusal"
        );
        assert_eq!(micro_usd(1.0e30), u64::MAX, "past u64's top saturates");
    }

    /// A tiny [`OpenRouter`](https://openrouter.ai)-shaped listing exercising every skip rule.
    fn openrouter_fixture() -> String {
        r#"{
            "data": [
                { "id": "openai/fixture-sol",
                  "pricing": { "prompt": "0.000002", "completion": "0.00001",
                               "input_cache_read": "0.0000001",
                               "input_cache_write": "0.0000025" } },
                { "id": "vendor/flat-model",
                  "pricing": { "prompt": "0.000001", "completion": "0.000003" } },
                { "id": "anthropic/claude-opus-9.9",
                  "pricing": { "prompt": "0.000004", "completion": "0.00002" } },
                { "id": "vendor/flat-model:batch",
                  "pricing": { "prompt": "0.0000005", "completion": "0.0000015" } },
                { "id": "vendor/giveaway:free",
                  "pricing": { "prompt": "0", "completion": "0" } },
                { "id": "~vendor/alias-latest",
                  "pricing": { "prompt": "0.000001", "completion": "0.000002" } },
                { "id": "vendor/unpriced", "pricing": {} },
                { "id": "vendor/no-pricing" }
            ]
        }"#
        .to_string()
    }

    #[test]
    fn openrouter_import_skips_variants_and_dual_keys_each_model() {
        let table =
            PriceTable::from_openrouter(&openrouter_fixture(), "2026-01-01").expect("converts");
        assert_eq!(table.version(), "openrouter-2026-01-01");
        assert_eq!(table.pricing_updated(), "2026-01-01");
        let bare = table.price("fixture-sol").expect("the bare id resolves");
        let prefixed = table
            .price("openai/fixture-sol")
            .expect("the vendor-prefixed id resolves for OpenRouter clients");
        assert_eq!(bare, prefixed, "one model, one price, two spellings");
        assert_eq!(
            bare.input_micro_per_mtok, 2_000_000,
            "per-token scaled to per-million"
        );
        assert_eq!(bare.output_micro_per_mtok, 10_000_000);
        assert_eq!(bare.cache_read_micro_per_mtok, Some(100_000));
        assert_eq!(bare.cache_write_micro_per_mtok, Some(2_500_000));
        let flat = table.price("vendor/flat-model").expect("kept");
        assert_eq!(
            flat.input_micro_per_mtok, 1_000_000,
            "the base row kept its full price"
        );
        assert_eq!(
            flat.cache_read_micro_per_mtok, None,
            "no cache column stated"
        );
        assert!(
            table.price("alias-latest").is_none(),
            "a ~ alias is not a price estimate"
        );
        assert!(
            table.price("unpriced").is_none(),
            "no positive prompt price means no row, never a zero guess"
        );
        assert!(
            table.price("no-pricing").is_none(),
            "a missing pricing object means no row"
        );
        let dotted = table
            .price("claude-opus-9.9")
            .expect("the [`OpenRouter`](https://openrouter.ai) dot spelling resolves");
        let dashed_alias = table
            .price("claude-opus-9-9-20261101")
            .expect("the first-party dash spelling resolves through the alias");
        assert_eq!(
            dotted, dashed_alias,
            "dot and dash spellings of one model are one price"
        );
        let batch_lookup = table
            .price("flat-model:batch")
            .expect("a variant lookup inherits its base model's row by prefix");
        assert_eq!(
            batch_lookup, flat,
            "the variant never became a row — the base price answers, an \
             over-estimate for batch usage, never the half-price under-estimate"
        );
    }

    #[test]
    fn openrouter_round_trip_through_the_interchange_format() {
        let table =
            PriceTable::from_openrouter(&openrouter_fixture(), "2026-01-01").expect("converts");
        let rendered = table.to_json().expect("renders");
        let reparsed = PriceTable::from_json(&rendered).expect("reparses");
        assert_eq!(
            table.to_json().unwrap(),
            reparsed.to_json().unwrap(),
            "convert-render-parse is lossless: the canonical renderings agree"
        );
    }

    #[test]
    fn same_prices_ignores_vintage_and_detects_real_drift() {
        let base_json = r#"{ "table_version": "a", "pricing_updated": "2026-01-01",
            "models": { "m": { "input_per_mtok": 2.0, "output_per_mtok": 4.0 } } }"#;
        let base = PriceTable::from_json(base_json).expect("base");
        let newer_date = r#"{ "table_version": "b", "pricing_updated": "2026-02-01",
            "models": { "m": { "input_per_mtok": 2.0, "output_per_mtok": 4.0 } } }"#;
        assert!(
            base.same_prices(&PriceTable::from_json(newer_date).expect("same rows")),
            "a date-only difference is unchanged — no refresh PR without real drift"
        );
        let repriced = r#"{ "table_version": "c", "pricing_updated": "2026-02-01",
            "models": { "m": { "input_per_mtok": 3.0, "output_per_mtok": 4.0 } } }"#;
        assert!(
            !base.same_prices(&PriceTable::from_json(repriced).expect("repriced")),
            "a moved price is drift"
        );
        let grown = r#"{ "table_version": "d", "pricing_updated": "2026-02-01",
            "models": { "m": { "input_per_mtok": 2.0, "output_per_mtok": 4.0 },
                        "n": { "input_per_mtok": 1.0, "output_per_mtok": 1.0 } } }"#;
        assert!(
            !base.same_prices(&PriceTable::from_json(grown).expect("grown")),
            "a new family is drift"
        );
        let fixture =
            PriceTable::from_openrouter(&openrouter_fixture(), "2026-01-01").expect("converts");
        let reconverted =
            PriceTable::from_openrouter(&openrouter_fixture(), "2026-06-01").expect("converts");
        assert!(
            fixture.same_prices(&reconverted),
            "row order from the listing is order-insensitive in the comparison"
        );
    }

    #[test]
    fn row_count_counts_every_spelling() {
        assert_eq!(
            PriceTable::empty().row_count(),
            0,
            "the empty table carries no rows"
        );
        let table =
            PriceTable::from_openrouter(&openrouter_fixture(), "2026-01-01").expect("converts");
        assert_eq!(
            table.row_count(),
            8,
            "two dual-keyed models plus the dotted spelling's four keys — \
             the figure the refresh fence counts"
        );
    }

    #[test]
    fn curated_with_cache_layers_a_cache_over_the_baked_table() {
        let dir =
            std::env::temp_dir().join(format!("loopctl-pricing-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("cache dir");
        std::fs::write(
            dir.join("b-second.json"),
            r#"{ "table_version": "second", "pricing_updated": "2999-01-01",
                "models": { "gpt-6.1-sol": { "input_per_mtok": 4.0,
                                            "output_per_mtok": 4.0 } } }"#,
        )
        .expect("write");
        std::fs::write(
            dir.join("a-first.json"),
            r#"{ "table_version": "first", "pricing_updated": "2026-01-01",
                "models": { "private-deploy": { "input_per_mtok": 0.5,
                                                "output_per_mtok": 0.5 } } }"#,
        )
        .expect("write");

        let table = PriceTable::curated_with_cache(&dir);
        assert_eq!(
            table
                .price("gpt-6.1-sol")
                .expect("the family still resolves")
                .input_micro_per_mtok,
            4_000_000,
            "the cache repriced a baked family — name order applies b over a"
        );
        assert!(
            table.price("private-deploy").is_some(),
            "the cache added a family the baked table never carried"
        );
        assert!(
            table.version().starts_with("openrouter-"),
            "the merged version composes over the baked identity: {}",
            table.version()
        );

        std::fs::write(dir.join("c-corrupt.json"), "{ not json").expect("write");
        let tolerant = PriceTable::curated_with_cache(&dir);
        assert_eq!(
            tolerant
                .price("gpt-6.1-sol")
                .map(|price| price.input_micro_per_mtok),
            Some(4_000_000),
            "a corrupt cache file is skipped, the readable cache still applies"
        );
        assert_eq!(
            tolerant
                .price("claude-opus-5-5")
                .map(|price| price.input_micro_per_mtok),
            PriceTable::curated()
                .price("claude-opus-5-5")
                .map(|price| price.input_micro_per_mtok),
            "the baked rows the cache never touched survive beneath it"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn curated_with_cache_never_fails_a_boot() {
        let missing =
            std::env::temp_dir().join(format!("loopctl-pricing-absent-{}", std::process::id()));
        let table = PriceTable::curated_with_cache(&missing);
        assert_eq!(
            table.version(),
            PriceTable::curated().version(),
            "a missing cache directory is the first-run norm — the baked table answers"
        );
    }

    #[test]
    fn today_iso_round_trips_through_the_date_parser() {
        let today = today_iso();
        assert_eq!(today.len(), 10, "YYYY-MM-DD shape: {today}");
        assert!(
            iso_to_epoch_days(&today).is_some(),
            "today parses as a well-formed ISO date: {today}"
        );
        assert_eq!(
            epoch_days_to_iso(0).as_deref(),
            Some("1970-01-01"),
            "the epoch renders as the epoch"
        );
        for date in ["2026-10-25", "2000-02-29", "1999-12-31", "2024-02-29"] {
            let days = iso_to_epoch_days(date).expect("parses");
            assert_eq!(
                epoch_days_to_iso(days).as_deref(),
                Some(date),
                "days-from-civil is the exact inverse of civil-from-days: {date}"
            );
        }
    }

    #[test]
    fn the_curated_seed_holds_a_well_formed_openrouter_table() {
        let table = PriceTable::curated();
        assert!(
            table
                .version()
                .strip_prefix("openrouter-")
                .is_some_and(|date| iso_to_epoch_days(date).is_some()),
            "the version names the source and a parseable date: {}",
            table.version()
        );
        assert!(
            iso_to_epoch_days(table.pricing_updated()).is_some(),
            "the pricing date is a well-formed ISO date: {}",
            table.pricing_updated()
        );
        assert!(
            !table.rows.is_empty(),
            "the seed carries rows — an empty seed would price nothing"
        );
        for (key, _) in &table.rows {
            assert!(
                !key.contains(':') && !key.starts_with('~'),
                "no variant or alias id survives the import: {key}"
            );
        }
        for (key, _) in &table.rows {
            if let Some((_vendor, bare)) = key.split_once('/') {
                assert!(
                    table.rows.iter().any(|(other, _)| other == bare),
                    "every vendor-prefixed key carries its bare twin: {key}"
                );
            }
        }
    }

    #[test]
    fn the_seed_elapses_zero_days_on_its_own_data_date() {
        let table = PriceTable::curated();
        assert_eq!(
            table.days_stale(table.pricing_updated()),
            Some(0),
            "a table is fresh on the day of the listing it was converted from"
        );
    }

    #[test]
    fn the_seed_stays_within_the_staleness_ratchet() {
        let table = PriceTable::curated();
        assert!(
            !table.is_stale(&today_iso()),
            "the vendored table is within {STALE_AFTER_DAYS} days of a refresh — \
             this pin is the ratchet: a master older than the threshold fails \
             here until the nightly drift job (or a release cut) lands a fresh \
             table"
        );
    }

    #[test]
    fn unknown_model_costs_none_never_a_guess() {
        let table = PriceTable::curated();
        let usage = Usage::new(50, 25);
        assert!(
            table.price("my-fine-tune-9000").is_none(),
            "an id the table predates has no row"
        );
        assert!(
            table.cost("my-fine-tune-9000", usage).is_none(),
            "cost answers None rather than a nearest-neighbor guess"
        );
    }

    #[test]
    fn cost_math_splits_the_usage_superset() {
        let anthropic = r#"{
            "table_version": "t1", "pricing_updated": "2026-01-01",
            "models": { "claude-x": {
                "input_per_mtok": 3.0, "output_per_mtok": 15.0,
                "cache_read_per_mtok": 0.3, "cache_write_per_mtok": 3.75 } }
        }"#;
        let table = PriceTable::from_json(anthropic).expect("valid table");
        let usage = Usage::new(1040, 1_000_000)
            .with_cached_input(900)
            .with_cache_write(40);
        let cost = table.cost("claude-x", usage).expect("priced");
        assert_eq!(
            cost.input_usd_micro, 300,
            "the uncached remainder at the base rate"
        );
        assert_eq!(
            cost.cache_read_usd_micro, 270,
            "the read share at the read rate"
        );
        assert_eq!(
            cost.cache_write_usd_micro, 150,
            "the write share at the write premium"
        );
        assert_eq!(cost.output_usd_micro, 15_000_000, "output per million");
        assert_eq!(
            cost.total_usd_micro,
            cost.input_usd_micro
                + cost.cache_read_usd_micro
                + cost.cache_write_usd_micro
                + cost.output_usd_micro,
            "the total is exactly the four classes summed"
        );

        let openai = r#"{
            "table_version": "t2", "pricing_updated": "2026-01-01",
            "models": { "gpt-x": {
                "input_per_mtok": 2.0, "output_per_mtok": 8.0,
                "cache_read_per_mtok": 1.0 } }
        }"#;
        let table = PriceTable::from_json(openai).expect("valid table");
        let usage = Usage::new(1_000_000, 0).with_cached_input(500_000);
        let cost = table.cost("gpt-x", usage).expect("priced");
        assert_eq!(
            cost.cache_read_usd_micro, 500_000,
            "the cached half at the read rate"
        );
        assert_eq!(
            cost.input_usd_micro,
            500_000 * 2,
            "the uncached half at the base rate"
        );
        assert_eq!(cost.cache_write_usd_micro, 0, "no write class reported");

        let bare = r#"{
            "table_version": "t3", "pricing_updated": "2026-01-01",
            "models": { "flat-x": { "input_per_mtok": 2.0, "output_per_mtok": 8.0 } }
        }"#;
        let table = PriceTable::from_json(bare).expect("valid table");
        let usage = Usage::new(1_000_000, 0).with_cached_input(1_000_000);
        let cost = table.cost("flat-x", usage).expect("priced");
        assert_eq!(
            cost.cache_read_usd_micro, 2_000_000,
            "a missing read column prices reported cache at the base rate — over, never under"
        );
    }

    #[test]
    fn prefix_resolution_strips_bedrock_spellings() {
        let table = PriceTable::curated();
        let regional = table
            .price("us.anthropic.claude-sonnet-4-5-20250929:0")
            .expect("the regional Bedrock spelling resolves to the family");
        let bare = table
            .price("claude-sonnet-4-5-20251014")
            .expect("the bare spelling resolves too");
        assert_eq!(regional, bare, "one family, one price");
        let cost = table
            .cost(
                "us.anthropic.claude-sonnet-4-5-20250929:0",
                Usage::new(10, 10),
            )
            .expect("priced through the same resolution");
        assert_eq!(
            cost.model, "claude-sonnet-4-5",
            "the stamp carries the family key — the dashed 4.5 alias, the \
             longest matching prefix, not the older claude-sonnet-4 row"
        );
    }

    #[test]
    fn staleness_is_days_between_iso_dates() {
        let table_json = r#"{ "table_version": "t", "pricing_updated": "2026-01-01",
            "models": { "m": { "input_per_mtok": 1.0, "output_per_mtok": 1.0 } } }"#;
        let table = PriceTable::from_json(table_json).expect("valid table");
        assert_eq!(table.days_stale("2026-03-31"), Some(89), "89 days in");
        assert!(!table.is_stale("2026-03-31"), "89 days is fresh");
        assert_eq!(table.days_stale("2026-04-01"), Some(90), "the boundary day");
        assert!(
            !table.is_stale("2026-04-01"),
            "90 days is still fresh — stale is strictly past it"
        );
        assert!(table.is_stale("2026-04-02"), "91 days is stale");
        assert_eq!(
            table.days_stale("2025-12-31"),
            Some(0),
            "an as-of before the pricing date is zero, never negative"
        );
        assert_eq!(table.days_stale("not-a-date"), None, "unparseable as-of");
        assert!(
            !table.is_stale("not-a-date"),
            "unparseable as-of does not guess stale"
        );
    }

    #[test]
    fn overrides_replace_and_add_rows_and_redate() {
        let base_json = r#"{ "table_version": "base-1", "pricing_updated": "2026-01-01",
            "models": { "gpt-4o": { "input_per_mtok": 2.5, "output_per_mtok": 10.0 } } }"#;
        let base = PriceTable::from_json(base_json).expect("valid base");
        let override_json = r#"{ "table_version": "fix-1", "pricing_updated": "2026-03-01",
            "models": {
                "gpt-4o": { "input_per_mtok": 2.0, "output_per_mtok": 8.0 },
                "private-deploy": { "input_per_mtok": 0.5, "output_per_mtok": 0.5 } } }"#;
        let layered = base
            .with_overrides(&PriceTable::from_json(override_json).expect("valid override"))
            .with_overrides(&PriceTable::empty());
        assert_eq!(
            layered.version(),
            "base-1+fix-1",
            "the version composes, identity preserved through an empty merge"
        );
        assert_eq!(
            layered.pricing_updated(),
            "2026-03-01",
            "the fresher date wins"
        );
        let repriced = layered
            .price("gpt-4o")
            .expect("the row survived the replace");
        assert_eq!(
            repriced.input_micro_per_mtok, 2_000_000,
            "the override's rate replaced the base's"
        );
        assert!(layered.price("private-deploy").is_some(), "a new key adds");
        assert!(
            layered.price("claude-3").is_none(),
            "the base's other families are untouched — this base had only gpt-4o"
        );
    }

    #[test]
    fn override_dir_loads_deterministically_and_names_bad_files() {
        let dir = std::env::temp_dir().join(format!("loopctl-pricing-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(
            dir.join("b-second.json"),
            r#"{ "table_version": "second", "pricing_updated": "2026-02-01",
                "models": { "m": { "input_per_mtok": 4.0, "output_per_mtok": 4.0 } } }"#,
        )
        .expect("write override");
        std::fs::write(
            dir.join("a-first.json"),
            r#"{ "table_version": "first", "pricing_updated": "2026-01-01",
                "models": { "m": { "input_per_mtok": 2.0, "output_per_mtok": 2.0 } } }"#,
        )
        .expect("write override");
        let loaded = PriceTable::load_overrides_dir(&dir).expect("both files load");
        let price = loaded.price("m").expect("the row is there");
        assert_eq!(
            price.input_micro_per_mtok, 4_000_000,
            "name order applies b over a — deterministic on every filesystem"
        );
        assert_eq!(
            loaded.version(),
            "empty+first+second",
            "merges compose in file order"
        );

        let bad = dir.join("c-bad.json");
        std::fs::write(&bad, "{ not json").expect("write bad override");
        let error = PriceTable::load_overrides_dir(&dir).expect_err("the bad file fails the load");
        assert!(
            error.to_string().contains("c-bad.json"),
            "the error names the offending file: {error}"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
