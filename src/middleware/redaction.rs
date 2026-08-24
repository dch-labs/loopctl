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
/// quoted value of at least 16 alphanumerics.
const API_KEY_KV: &str = r#"(?i)(?:api[_-]?key|token|secret)\s*[=:]\s*["']?[A-Za-z0-9]{16,}["']?"#;

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
    /// host-added) scrub — zero false positives from novel-token
    /// detection, at the cost of missing formats no literal covers.
    #[must_use]
    pub fn with_entropy_heuristic(mut self, enabled: bool) -> Self {
        self.entropy_heuristic = enabled;
        self
    }

    /// Rewrite `text` in place, replacing every secret match with its
    /// `[REDACTED:<kind>]` placeholder.
    ///
    /// Returns the count of redactions made, for observability (a
    /// host can log it). Explicit patterns run first; when the entropy
    /// heuristic is enabled, any remaining token of at least 32
    /// characters whose byte entropy reaches 4.5 bits per byte becomes
    /// `[REDACTED:high_entropy]`.
    pub fn scrub(&self, text: &mut String) -> usize {
        let mut rewritten = std::mem::take(text);
        let mut count = 0usize;
        for pattern in &self.patterns {
            let hits = pattern.pattern.find_iter(&rewritten).count();
            if hits > 0 {
                let placeholder = format!("[REDACTED:{}]", pattern.kind);
                rewritten = pattern
                    .pattern
                    .replace_all(&rewritten, placeholder.as_str())
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
/// Splits on spaces (preserving all other structure), trims
/// non-token characters from each piece's edges, and replaces any
/// remaining core of [`MIN_ENTROPY_TOKEN_LEN`] or more characters whose
/// Shannon entropy reaches [`ENTROPY_THRESHOLD`]. Returns the number of
/// tokens redacted.
fn scrub_high_entropy(text: &mut String) -> usize {
    let pieces: Vec<String> = std::mem::take(text)
        .split(' ')
        .map(redact_piece_if_high_entropy)
        .collect();
    let count = pieces
        .iter()
        .filter(|p| p.contains("[REDACTED:high_entropy]"))
        .count();
    *text = pieces.join(" ");
    count
}

/// Redact the token core of one space-delimited piece, if it qualifies.
fn redact_piece_if_high_entropy(piece: &str) -> String {
    let core = piece.trim_matches(|c: char| !is_token_char(c));
    if core.len() < MIN_ENTROPY_TOKEN_LEN || shannon_entropy(core) < ENTROPY_THRESHOLD {
        return piece.to_string();
    }
    piece.replace(core, "[REDACTED:high_entropy]")
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
