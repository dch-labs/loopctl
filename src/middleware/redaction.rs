//! Secret scrubbing for tool output.
//!
//! [`RedactingMiddleware`] wraps the dispatch pipeline and rewrites every
//! text part of the returned [`ToolOutput`](crate::tool::ToolOutput),
//! replacing anything matching a [`SecretPatternSet`] with a
//! `[REDACTED:<kind>]` placeholder. Tools that capture external content
//! — shell stdout, fetched URL bodies — can emit credentials; without
//! scrubbing those flow back into the model's context and into whatever
//! the host persists. The mechanism is generic; the policy (which
//! patterns, how strict) is the host's via
//! [`SecretPatternSet::default_common`] plus [`SecretPatternSet::with_pattern`].
//!
//! The rewrite is post-tool and pre-result-re-entry: loop semantics
//! (compaction, loop-detection hashing, turn counting) are unaffected,
//! and a redacted output is still a successful tool result.
//!
//! The entropy heuristic measures maximal runs of token characters;
//! path-shaped runs are measured per segment, so word-like structure
//! passes while a dense segment still masks —
//! [`SecretPatternSet::scrub`] states the exact rule, and
//! [`SecretPatternSet::with_entropy_heuristic`] toggles the lane.
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::middleware::{RedactingMiddleware, SecretPatternSet, ToolPipeline};
//! use loopctl::tool::ToolRegistry;
//! use std::sync::Arc;
//!
//! let pipeline = ToolPipeline::builder()
//!     .with_middleware(RedactingMiddleware::new(SecretPatternSet::default_common()))
//!     .with_core(Arc::new(ToolRegistry::new()))
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::message::{ToolContent, ToolContentPart};

/// Matches `Authorization: Bearer …` header values.
///
/// Case-insensitive scheme and header name, RFC 3986 unreserved
/// characters plus separators in the credential — the shape HTTP
/// clients echo in verbose logs and fetched-error bodies.
const BEARER: &str = r#"(?i)authorization:\s*bearer\s+[A-Za-z0-9\-._~+/=]+"#;

/// Matches `api_key=` / `token:` / `secret=` style key-value tokens.
///
/// Covers `.env` dumps and config prints: the key with `_`/`-`
/// separators, either `=` or `:` as the separator, and an optionally
/// quoted value of at least 16 alphanumerics, captured as the `value`
/// group. A leading `decl` group (see [`is_code_declaration`]) marks
/// variable declarations — `let token = "…"`, `const secret: Type =
/// …`, and compound names (`let entropy_token = …`, `let client_secret
/// = …`) where the key sits anywhere inside the declared identifier.
/// The declaration keywords cover Rust (`let`/`const`/`static`/`fn`),
/// `JavaScript` and `TypeScript` (`let`/`const`/`var`/`function`),
/// Python (`def`), and `final` in its adjacent-name shapes;
/// type-prefixed declarations (Java, C, Go) are not covered. The `[^=\n]{0,48}`
/// tolerance spans the optional type annotation — a declaration whose
/// annotation plus spacing exceeds 48 characters matches neither arm,
/// and its value then leans on the entropy heuristic alone.
///
/// A declaration passes through verbatim when its initializer is
/// unquoted (a constructor call, or the annotation itself standing in
/// as the matched value) or its quoted value carries a placeholder
/// marker ([`PLACEHOLDER_MARKERS`]); any other quoted value masks in
/// place through the `value` group, keeping the statement's syntax
/// intact. `export` is deliberately not a declaration keyword:
/// `export TOKEN="…"` is the env-dump shape and must keep firing, and
/// the plain arm keeps matching keys inside compound dump names
/// (`OPENAI_API_KEY=…`) — the underscore env convention — because no
/// declaration keyword precedes them.
const API_KEY_KV: &str = r#"(?i)(?:(?P<decl>\b(?:let|const|var|static|final|fn|def|function)(?:\s+mut)?\s+[A-Za-z0-9_]*(?:api[_-]?key|token|secret)[A-Za-z0-9_]*\b[^=\n]{0,48})|(?:api[_-]?key|token|secret))\s*[=:]\s*(?P<value>["']?[A-Za-z0-9]{16,}["']?)"#;

/// Marker words that mark a quoted declaration value as documentation.
///
/// A quoted value containing any of these substrings (compared
/// case-insensitively) passes through verbatim inside a declaration —
/// the shape fixtures and examples spell (`placeholder0123456789`,
/// `yourexamplekey…`) — while a dense literal (`J8kL2mN4pQ…`) carries
/// none of them and masks in place.
const PLACEHOLDER_MARKERS: [&str; 8] = [
    "placeholder",
    "example",
    "sample",
    "dummy",
    "changeme",
    "your",
    "xxxx",
    "redacted",
];

/// Matches AWS access-key IDs (`AKIA`, `ASIA`, `AGPA` prefixes).
///
/// The four-letter prefix plus 16 uppercase alphanumerics is the
/// documented access-key-id shape; the paired secret key is left to
/// the entropy heuristic (it is a contextless 40-char base64 string).
const AWS_ACCESS_KEY: &str = r"A(?:KIA|SIA|GPA)[0-9A-Z]{16}";

/// Matches whole PEM private-key blocks, header through footer.
///
/// Any key type (`RSA`, `EC`, `OPENSSH`, …) between the `BEGIN`/`END`
/// markers; the non-greedy body keeps two blocks in one output
/// separate, and `scrub` collapses each to a single placeholder.
const PEM_PRIVATE_KEY: &str =
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----";

/// Matches the `GitHub` PAT prefix family (`ghp_`, `gho_`, …).
///
/// The fine-grained (`gh[pousr]_`) prefixes followed by at least 36
/// alphanumerics — the shape `gh auth token` and CI masks print.
const GITHUB_PAT: &str = r"gh[pousr]_[A-Za-z0-9]{36,}";

/// Matches `GitLab` PATs (`glpat-…`).
///
/// The `glpat-` prefix plus 20 token characters, the personal access
/// token shape `GitLab`'s UI creates by default.
const GITLAB_PAT: &str = r"glpat-[A-Za-z0-9_\-]{20}";

/// Minimum token length considered by the high-entropy heuristic.
///
/// Shorter tokens are ignored even when their byte distribution looks
/// random — most false positives (IDs, short hashes) fall below this.
const MIN_ENTROPY_TOKEN_LEN: usize = 32;

/// Shannon-entropy threshold, in bits per byte, for the heuristic.
///
/// A hex string tops out at 4.0 (16 symbols), so commit SHAs and hex
/// hashes stay visible; base64-family tokens sit near 6.0 and are
/// redacted. The value is the one truffleHog/gitleaks-style tools
/// converged on.
const ENTROPY_THRESHOLD: f64 = 4.5;

/// One secret-detection rule: a compiled pattern and the label that
/// replaces its matches.
///
/// `kind` is the short string substituted into the output as
/// `[REDACTED:<kind>]` (e.g. `"aws_access_key"`, `"github_pat"`,
/// `"bearer"`). It is also the key a host uses when reasoning about its
/// own extensions, so pick names that are stable and grep-able.
#[derive(Debug)]
pub struct SecretPattern {
    /// The label used in the `[REDACTED:<kind>]` placeholder.
    ///
    /// Short, lowercase, underscore-separated — the shapes above use
    /// `"bearer"`, `"aws_access_key"`, and peers; a host's custom
    /// patterns should follow the same convention.
    pub kind: &'static str,

    /// The compiled matcher for this secret shape.
    ///
    /// Constructed by the host (via [`regex::Regex::new`] — fallible,
    /// so the host handles invalid patterns at its own boundary) and
    /// moved into the set with
    /// [`SecretPatternSet::with_pattern`](SecretPatternSet::with_pattern).
    pub pattern: regex::Regex,
}

/// A collection of secret-detection rules applied to tool output.
///
/// Construct with [`SecretPatternSet::default_common`] for the curated
/// set (Authorization headers, key-value tokens, AWS keys, PEM
/// private-key blocks, GitHub/GitLab PATs, and a high-entropy heuristic
/// for unknown formats), then extend with
/// [`SecretPatternSet::with_pattern`] for host-specific shapes. The set
/// is `Send + Sync` (each [`regex::Regex`] is), so it can live behind
/// an `Arc` in a shared pipeline.
#[derive(Debug)]
pub struct SecretPatternSet {
    /// The explicit rules, applied in insertion order.
    ///
    /// Curated shapes first (from [`default_common`]), host additions
    /// after; every rule runs on every text part, so order matters only
    /// when two patterns can overlap.
    patterns: Vec<SecretPattern>,

    /// Whether the Shannon-entropy heuristic runs on tokens no explicit
    /// pattern matched.
    ///
    /// Default `true`; a host turns it off via
    /// [`with_entropy_heuristic`](Self::with_entropy_heuristic) when
    /// false positives are noisy for its workload.
    entropy_heuristic: bool,
}

impl SecretPatternSet {
    /// The curated default: shapes that recur across providers and hosts.
    ///
    /// Covers `Authorization: Bearer …` headers, `api_key=`-style
    /// key-value tokens, AWS access-key IDs, PEM private-key blocks,
    /// and the `GitHub` (`gh[pousr]_…`) and `GitLab` (`glpat-…`) PAT
    /// families. Each match becomes `[REDACTED:<kind>]`; the whole PEM
    /// block collapses to one placeholder. The high-entropy heuristic is
    /// on.
    #[must_use]
    pub fn default_common() -> Self {
        Self {
            patterns: [
                curated("bearer", BEARER),
                curated("api_key_kv", API_KEY_KV),
                curated("aws_access_key", AWS_ACCESS_KEY),
                curated("pem_private_key", PEM_PRIVATE_KEY),
                curated("github_pat", GITHUB_PAT),
                curated("gitlab_pat", GITLAB_PAT),
            ]
            .into_iter()
            .flatten()
            .collect(),
            entropy_heuristic: true,
        }
    }

    /// Add a host-supplied rule. Returns `self` for chaining.
    ///
    /// The pattern arrives already compiled (the host owns the
    /// invalid-regex error), so this cannot fail.
    #[must_use]
    pub fn with_pattern(mut self, pattern: SecretPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Toggle the high-entropy heuristic. Returns `self` for chaining.
    ///
    /// With the heuristic off, only the explicit patterns (curated plus
    /// host-added) scrub — zero false positives from novel-run
    /// detection, at the cost of missing formats no literal covers.
    /// With it on, dense runs of at least 32 token characters are
    /// masked, except inside path-shaped runs (two or more `/`
    /// separators carrying at least two word-like segments), where
    /// each segment is measured on its own and only the dense
    /// segments mask.
    #[must_use]
    pub fn with_entropy_heuristic(mut self, enabled: bool) -> Self {
        self.entropy_heuristic = enabled;
        self
    }

    /// Rewrite `text` in place, replacing every secret match with its
    /// `[REDACTED:<kind>]` placeholder.
    ///
    /// Returns the count of redactions made, for observability (a
    /// host can log it). Explicit patterns run first; a match whose
    /// pattern carries a participating `decl` capture group is a
    /// variable declaration, which shares the key-value shape with a
    /// credential dump but is code: it passes through verbatim when
    /// its initializer is unquoted or its quoted value names itself a
    /// placeholder, and otherwise masks only the value, leaving the
    /// statement's syntax intact. When the entropy heuristic is
    /// enabled, any remaining run of at least 32 token characters
    /// whose byte entropy reaches 4.5 bits per byte becomes
    /// `[REDACTED:high_entropy]` — except that a path-shaped run (two
    /// or more `/` separators carrying at least two word-like
    /// lowercase-letter segments) is measured per segment: each
    /// `/`-segment that alone clears both gates masks, while the
    /// separators and word-like segments survive, so a dense segment
    /// cannot hide behind the path exemption.
    pub fn scrub(&self, text: &mut String) -> usize {
        let mut rewritten = std::mem::take(text);
        let mut count = 0usize;
        for pattern in &self.patterns {
            let hits = pattern
                .pattern
                .captures_iter(&rewritten)
                .filter(|caps| {
                    !matches!(declaration_rewrite(caps), DeclarationRewrite::PassThrough)
                })
                .count();
            if hits > 0 {
                let placeholder = format!("[REDACTED:{}]", pattern.kind);
                rewritten = pattern
                    .pattern
                    .replace_all(&rewritten, |caps: &regex::Captures<'_>| {
                        rewrite_match(caps, &placeholder)
                    })
                    .into_owned();
            }
            count = count.saturating_add(hits);
        }
        if self.entropy_heuristic {
            count = count.saturating_add(scrub_high_entropy(&mut rewritten));
        }
        *text = rewritten;
        count
    }
}

/// How one pattern match is rewritten by [`SecretPatternSet::scrub`].
///
/// A credential dump replaces the whole match; a declaration either
/// passes through or masks only its value substring, per
/// [`declaration_rewrite`].
enum DeclarationRewrite {
    /// The whole match is a dump — replace it with the placeholder.
    ///
    /// Nothing of the match survives, not even the key: a dump is
    /// configuration or environment text, not code a reader needs.
    Dump,

    /// The match is a declaration with a benign value — keep it
    /// verbatim.
    PassThrough,

    /// The match is a declaration with a secret-shaped value — mask
    /// the value substring, keeping the declaration syntax.
    MaskValue,
}

/// Decide one match's disposition under [`SecretPatternSet::scrub`].
///
/// A match without a participating `decl` group is a dump; a
/// declaration masks its value only when the value is quoted and
/// carries no placeholder marker.
fn declaration_rewrite(caps: &regex::Captures<'_>) -> DeclarationRewrite {
    if !is_code_declaration(caps) {
        return DeclarationRewrite::Dump;
    }
    if declaration_value_is_masked(caps) {
        DeclarationRewrite::MaskValue
    } else {
        DeclarationRewrite::PassThrough
    }
}

/// Render one match's replacement under its pattern's `placeholder`.
///
/// Consults [`declaration_rewrite`] for the disposition and renders
/// it — the placeholder for a dump, the untouched match for a benign
/// declaration, the value-masked splice otherwise.
fn rewrite_match(caps: &regex::Captures<'_>, placeholder: &str) -> String {
    match declaration_rewrite(caps) {
        DeclarationRewrite::Dump => placeholder.to_string(),
        DeclarationRewrite::PassThrough => caps
            .get(0)
            .map_or_else(String::new, |whole| whole.as_str().to_owned()),
        DeclarationRewrite::MaskValue => masked_replacement(caps, placeholder),
    }
}

/// Whether a declaration match's value must be masked in place.
///
/// Only quoted literals mask: a bare initializer (a constructor call,
/// or a type annotation standing in as the matched value) is code,
/// not a credential literal. A quoted value masks unless it names
/// itself a placeholder ([`PLACEHOLDER_MARKERS`]).
fn declaration_value_is_masked(caps: &regex::Captures<'_>) -> bool {
    caps.name("value").is_some_and(|value| {
        let text = value.as_str();
        let bare = text.trim_matches(['"', '\'']);
        (text.starts_with('"') || text.starts_with('\'')) && !looks_like_placeholder(bare)
    })
}

/// Whether `value` names itself a placeholder rather than a secret.
///
/// Case-insensitive, so `PlaceHolder0123` qualifies exactly like its
/// lowercase spelling.
fn looks_like_placeholder(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    PLACEHOLDER_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// The whole match with only its `value` group masked.
///
/// The value's own quote characters survive, so the statement still
/// parses; only the secret literal is gone.
fn masked_replacement(caps: &regex::Captures<'_>, placeholder: &str) -> String {
    let Some((whole, value)) = caps.get(0).zip(caps.name("value")) else {
        return String::new();
    };
    let start = value.start().saturating_sub(whole.start());
    let end = value.end().saturating_sub(whole.start());
    let source = whole.as_str();
    [
        source.get(..start).map_or_else(String::new, str::to_string),
        quote_preserving_mask(value.as_str(), placeholder),
        source.get(end..).map_or_else(String::new, str::to_string),
    ]
    .concat()
}

/// The value span's replacement: the placeholder wrapped in whichever
/// quote characters the value itself carried.
///
/// An opening quote without a closing one (or the reverse) keeps
/// exactly the quotes that were there — masking never invents syntax.
fn quote_preserving_mask(value: &str, placeholder: &str) -> String {
    let opening = value.chars().next().filter(|ch| matches!(ch, '"' | '\''));
    let closing = value.chars().last().filter(|ch| matches!(ch, '"' | '\''));
    let mut masked = String::with_capacity(placeholder.len().saturating_add(2));
    if let Some(ch) = opening {
        masked.push(ch);
    }
    masked.push_str(placeholder);
    if let Some(ch) = closing {
        masked.push(ch);
    }
    masked
}

/// Whether `caps` marks a code declaration rather than a credential
/// dump.
///
/// A pattern opts in by naming its declaration arm `decl`; for every
/// pattern without such a group — all host patterns and every other
/// curated shape — the group never participates and this returns
/// `false`, so the declaration pass-through is inert for them. Within
/// a declaration, a `value` group is consulted for in-place masking
/// (see [`declaration_value_is_masked`]); a declaration match without
/// one passes through verbatim.
fn is_code_declaration(caps: &regex::Captures<'_>) -> bool {
    caps.name("decl").is_some()
}

/// Compile one curated literal, returning `None` if it fails to compile.
///
/// The shipped literals are known-good, so `None` is unreachable in
/// practice; the `debug_assert` turns a broken literal into a test-time
/// failure instead of a silent gap.
fn curated(kind: &'static str, literal: &'static str) -> Option<SecretPattern> {
    let pattern = regex::Regex::new(literal).ok();
    debug_assert!(pattern.is_some(), "curated literal must compile: {literal}");
    pattern.map(|pattern| SecretPattern { kind, pattern })
}

/// Redact high-entropy tokens no explicit pattern matched.
///
/// Splits on whitespace — spaces, tabs, carriage returns, newlines —
/// preserving every separator, and within each piece measures the
/// maximal runs of token characters, replacing any run of
/// [`MIN_ENTROPY_TOKEN_LEN`] or more characters whose Shannon entropy
/// reaches [`ENTROPY_THRESHOLD`], unless the run is path-shaped
/// ([`run_is_path_shaped`]). Whitespace bounding is the contract:
/// a credential never spans a newline, while space-free multi-line
/// tool output (a piped `ls` listing, CRLF logs) must not merge into
/// one giant candidate. Run bounding is the same contract one level
/// down: a credential never carries dots, colons, brackets, or
/// variable interpolation, while shell lines, paths, URLs, and JSON
/// literals are built from exactly those. Returns the number of runs
/// redacted.
fn scrub_high_entropy(text: &mut String) -> usize {
    let mut count = 0usize;
    let mut out = String::new();
    for (piece, redacted) in std::mem::take(text)
        .split_inclusive(char::is_whitespace)
        .map(redact_piece_if_high_entropy)
    {
        count = count.saturating_add(redacted);
        out.push_str(&piece);
    }
    *text = out;
    count
}

/// Redact the qualifying token runs of one whitespace-delimited piece.
///
/// A piece is one or more maximal runs of token characters separated
/// by non-token punctuation; each run is measured on its own, because
/// a credential is a dense run of credential-alphabet characters —
/// never a dotted, bracketed, or interpolated structure. Returns the
/// (possibly rewritten) piece and how many of its runs were
/// substituted — the count, not marker sniffing, is what tallies, so
/// an echoed placeholder's own runs (`REDACTED`, `high_entropy`) are
/// both far below the length gate and never counted again.
fn redact_piece_if_high_entropy(piece: &str) -> (String, usize) {
    let mut out = String::new();
    let mut run = String::new();
    let mut count = 0usize;
    for ch in piece.chars() {
        if is_token_char(ch) {
            run.push(ch);
        } else {
            flush_run(&mut run, &mut out, &mut count);
            out.push(ch);
        }
    }
    flush_run(&mut run, &mut out, &mut count);
    (out, count)
}

/// Append `run`'s disposition to `out` and reset it.
///
/// A qualifying run becomes the placeholder; a path-shaped run is
/// measured per segment (see [`redact_dense_path_segments`]) so a
/// dense segment cannot hide behind the path exemption; any other
/// run is appended verbatim. `count` advances only on a substitution.
fn flush_run(run: &mut String, out: &mut String, count: &mut usize) {
    if run.is_empty() {
        return;
    }
    if run_is_path_shaped(run) {
        redact_dense_path_segments(run.as_str(), out, count);
    } else if run_qualifies(run) {
        out.push_str("[REDACTED:high_entropy]");
        *count = count.saturating_add(1);
    } else {
        out.push_str(run);
    }
    run.clear();
}

/// Redact the dense segments of one path-shaped run into `out`.
///
/// Each `/`-segment is measured exactly as a slash-free run would be
/// ([`run_qualifies`] — its path guard is inert on a single segment):
/// a segment that alone clears the length and entropy gates masks to
/// the placeholder while the separators and the word-like segments
/// survive, so a secret riding at the end of a path-shaped run
/// (`…/tokens/<secret>`) no longer passes unmeasured. Pinned paths
/// keep passing because their segments stay under the gates —
/// `target/debug/deps/…`'s trailing build-artifact segment is 30
/// characters, below the 32-character length gate.
fn redact_dense_path_segments(run: &str, out: &mut String, count: &mut usize) {
    for piece in run.split_inclusive('/') {
        let segment = piece.strip_suffix('/').unwrap_or(piece);
        if run_qualifies(segment) {
            out.push_str("[REDACTED:high_entropy]");
            if piece.len() > segment.len() {
                out.push('/');
            }
            *count = count.saturating_add(1);
        } else {
            out.push_str(piece);
        }
    }
}

/// Whether `run` is measured as a credential candidate.
///
/// The length and entropy gates plus the path guard — a dense enough
/// run that is not slash-separated word structure. The gates run
/// cheapest first: length, then the path guard's allocation, then the
/// entropy scan.
fn run_qualifies(run: &str) -> bool {
    run.len() >= MIN_ENTROPY_TOKEN_LEN
        && !run_is_path_shaped(run)
        && shannon_entropy(run) >= ENTROPY_THRESHOLD
}

/// Whether `run` is a slash-separated path, not a credential.
///
/// Two or more `/` separators with at least two word segments (see
/// [`is_dictionary_segment`]) — `target/debug/deps/…` — is how build
/// systems, repositories, and shells spell locations. A random base64
/// run may carry slashes, but its slash-segments are mixed-case or
/// digit-bearing noise rather than words, so a slash-bearing secret
/// stays measured.
fn run_is_path_shaped(run: &str) -> bool {
    let segments: Vec<&str> = run.split('/').collect();
    segments.len() >= 3
        && segments
            .iter()
            .filter(|segment| is_dictionary_segment(segment))
            .count()
            >= 2
}

/// Whether `segment` reads as a word, not random material.
///
/// Four or more ASCII lowercase letters — digits disqualify the
/// segment — the shape dictionary words, crate names, and directory
/// names take, and a uniform-random credential segment almost never
/// does.
fn is_dictionary_segment(segment: &str) -> bool {
    segment.len() >= 4 && segment.chars().all(|ch| ch.is_ascii_lowercase())
}

/// Whether `c` appears in the token alphabet the heuristic scans.
///
/// Alphanumerics plus the base64 and common credential separators
/// (`+ / = - _`); everything else — quotes, brackets, colons — bounds a
/// token.
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '-' | '_')
}

/// Shannon entropy of `token`'s bytes, in bits per byte.
///
/// A uniform sample over `n` distinct symbols scores `log2(n)`: hex
/// tops out at 4.0, base64 near 6.0 — the spread
/// [`ENTROPY_THRESHOLD`] sits between.
fn shannon_entropy(token: &str) -> f64 {
    let mut freq = [0u64; 256];
    for byte in token.bytes() {
        if let Some(slot) = freq.get_mut(usize::from(byte)) {
            *slot = slot.saturating_add(1);
        }
    }
    let total = u64::try_from(token.len()).unwrap_or(u64::MAX);
    let mut entropy = 0.0;
    for count in freq {
        if count == 0 {
            continue;
        }
        let p = crate::numeric::unit_ratio(count, total);
        entropy -= p * p.log2();
    }
    entropy
}

/// Middleware that scrubs secrets from tool output after execution.
///
/// Wraps the dispatch pipeline and rewrites each text part of the
/// returned [`ToolOutput`](crate::tool::ToolOutput) using a
/// [`SecretPatternSet`], replacing matches with `[REDACTED:<kind>]`.
/// Image and other non-text multipart parts are left unchanged. The
/// rewrite is post-tool, pre-result-re-entry — it does not affect loop
/// semantics (compaction, loop-detection hashing, turn counting), never
/// sets `is_error`, and preserves any `DisplayHint`.
///
/// Default off: register it explicitly in the pipeline. A host that
/// does not register it sees today's behaviour (no scrubbing).
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::middleware::{RedactingMiddleware, SecretPatternSet, ToolPipeline};
/// use loopctl::tool::ToolRegistry;
/// use std::sync::Arc;
///
/// let pipeline = ToolPipeline::builder()
///     .with_middleware(RedactingMiddleware::new(SecretPatternSet::default_common()))
///     .with_core(Arc::new(ToolRegistry::new()))
///     .build()?;
/// ```
pub struct RedactingMiddleware {
    /// The rules this middleware applies to every text part.
    ///
    /// Held by value (constructed once, moved in) — the same ownership
    /// shape as `OutputLimitMiddleware`'s `max_chars` and
    /// `VerifyMiddleware`'s verifier.
    patterns: SecretPatternSet,
}

impl RedactingMiddleware {
    /// Create a redacting middleware with the given pattern set.
    ///
    /// The set is moved in and shared by nothing else; build one
    /// middleware per pipeline. `name()` is `"redaction"`.
    #[must_use]
    pub fn new(patterns: SecretPatternSet) -> Self {
        Self { patterns }
    }
}

impl ToolMiddleware for RedactingMiddleware {
    fn name(&self) -> &'static str {
        "redaction"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let patterns = &self.patterns;
        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;
            rewrite_text_parts(&mut result, patterns);
            result
        })
    }
}

/// Rewrite every text part of `result.output` through `patterns`.
///
/// `ToolContent::Text` scrubs the single string;
/// `ToolContent::Multipart` scrubs each `ToolContentPart::Text` in
/// place and leaves image and other parts untouched. Substitutions are
/// applied silently — the `[REDACTED:<kind>]` placeholder is the
/// model-visible signal.
fn rewrite_text_parts(result: &mut ToolDispatchResult, patterns: &SecretPatternSet) {
    match result.output {
        ToolContent::Text(ref mut text) => {
            patterns.scrub(text);
        }
        ToolContent::Multipart(ref mut parts) => {
            for part in parts.iter_mut() {
                if let ToolContentPart::Text { text } = part {
                    patterns.scrub(text);
                }
            }
        }
    }
}
