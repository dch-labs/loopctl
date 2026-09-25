//! The v1 `loop.yaml` manifest model.
//!
//! The declarative center of a loopctl-driven agent: strict serde types for
//! every section, named profile overlays with documented merge semantics,
//! compose-style environment interpolation, span-bearing validation errors,
//! and the JSON Schema export that keeps editor tooling and validators in
//! sync with the parser.
//!
//! The entry point is [`ManifestDocument::parse`], which runs the strict
//! typed pass (unknown keys fail with file position and a did-you-mean
//! suggestion) and keeps the raw tree for profile merging.
//! [`ManifestDocument::resolve`] applies an optional profile, validates the
//! merged document, interpolates environment references, and returns a
//! [`ResolvedManifest`] whose [`config_hash`](ResolvedManifest::config_hash)
//! pins the exact resolved configuration for runs and cassettes.
//!
//! # Example
//!
//! ```rust,ignore
//! let document = ManifestDocument::parse(&yaml_text)?;
//! let resolved = document.resolve(Some("cheap"), &EnvVars)?;
//! println!("pinned config hash: {}", resolved.config_hash());
//! ```
//!
//! Secrets are references, never values: `${secret:NAME}` resolves through
//! the [`EnvResolver`] (env-only by default; richer sources such as
//! keyring or files are the consumer's addition), and a literal key-shaped
//! string anywhere in the document fails validation.

mod error;
mod interpolate;
mod merge;
mod schema;
mod types;
mod validate;

pub use error::{ManifestError, Span};
pub use schema::manifest_json_schema;
pub use types::{
    AgentSection, Budgets, CassettesSection, CompactionSettings, ContextSection, Manifest,
    McpServer, McpServerOverlay, MemorySection, Metadata, MissedFire, ModelEntry,
    ModelEntryOverlay, ModelsOverlay, ModelsSection, Overlap, PermissionMode, PermissionRules,
    Permissions, Profile, RecordMode, ReplayMode, SandboxSection, ScheduleSection, ToolEntry,
    TriggersSection,
};

use std::collections::BTreeMap;

use serde_yaml_ng::Value;
/// The FNV-1a 64 offset basis.
///
/// The standard 64-bit FNV-1a starting state.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The FNV-1a 64 prime.
///
/// The standard 64-bit FNV-1a multiplier.
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Where interpolation references resolve from.
///
/// Injected so resolution is deterministic under test and so the CLI can
/// layer richer sources (keyring, files) ahead of the env-only default.
pub trait EnvResolver {
    /// Look up `name`, returning its value when present.
    ///
    /// Returning `None` is the miss signal; empty values are legitimate
    /// resolutions and are not treated as misses.
    fn resolve(&self, name: &str) -> Option<String>;
}

/// The default [`EnvResolver`]: the process environment.
///
/// Resolves `${VAR}` and `${secret:NAME}` references from
/// [`std::env::var`], skipping variables that are not valid Unicode the
/// same way a missing variable is skipped.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvVars;

impl EnvResolver for EnvVars {
    fn resolve(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

/// A fixed [`EnvResolver`] backed by a map.
///
/// Deterministic by construction — sorted iteration, no environment access
/// — which is why both the crate's tests and downstream consumers prefer
/// it over mutating the process environment.
#[derive(Debug, Clone, Default)]
pub struct EnvMap(BTreeMap<String, String>);

impl EnvMap {
    /// Build a resolver from an iterator of `(name, value)` pairs.
    ///
    /// Later pairs win on duplicate names, matching [`BTreeMap::extend`]
    /// semantics.
    #[must_use]
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, S)>,
        S: Into<String>,
    {
        Self(
            pairs
                .into_iter()
                .map(|(name, value)| (name.into(), value.into()))
                .collect(),
        )
    }
}

impl EnvResolver for EnvMap {
    fn resolve(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }
}

impl EnvResolver for BTreeMap<String, String> {
    fn resolve(&self, name: &str) -> Option<String> {
        self.get(name).cloned()
    }
}

/// A parsed manifest plus the raw tree profile merging needs.
///
/// Construction runs the strict typed pass — unknown keys, wrong types, and
/// a missing `version` all fail here with source positions — and keeps the
/// untyped tree because `!append` tags are only visible at that level.
#[derive(Debug, Clone)]
pub struct ManifestDocument {
    /// The typed, pre-merge document.
    ///
    /// Validated to the extent the strict parse can (unknown keys, types,
    /// the version gate); profile application happens on demand.
    typed: Manifest,

    /// The raw value tree, source of the profile merge.
    ///
    /// Kept alongside the typed view because `!append` tags are only
    /// visible at this level — the typed pass unwraps them away.
    raw: Value,
}

impl ManifestDocument {
    /// Parse manifest text.
    ///
    /// Rejects unknown fields (with position and suggestion), malformed
    /// values, a missing `version`, and multi-document YAML — a manifest is
    /// exactly one document.
    ///
    /// # Errors
    ///
    /// [`ManifestError`] describing the first failure, with a
    /// [`Span`]-derived position whenever the strict pass can attribute one.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let typed: Manifest =
            serde_yaml_ng::from_str(text).map_err(|error| ManifestError::from_yaml(&error))?;
        validate::check_version(&typed)?;
        let raw: Value =
            serde_yaml_ng::from_str(text).map_err(|error| ManifestError::from_yaml(&error))?;
        Ok(Self { typed, raw })
    }

    /// The typed document before any profile is applied.
    ///
    /// Useful for tooling that lists profiles or diffs overlays; resolution
    /// consumers want [`Self::resolve`].
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.typed
    }

    /// Apply an optional profile, validate, interpolate, and hash.
    ///
    /// The pipeline: overlay guard (the profile exists; it touches neither
    /// `version` nor `profiles`), tag-aware deep-merge over the raw tree, a
    /// defense-in-depth typed re-parse of the merged tree, structural and
    /// cross-reference validation, then interpolation of every effective
    /// string except `token_key` values. The returned
    /// [`ResolvedManifest`] carries the interpolated document for wiring
    /// plus its canonical-JSON config hash; both documents drop the profile
    /// declarations, and the pin alone pins secret-consuming strings as
    /// written — any string whose written form carries a `${secret:…}`
    /// reference, plus MCP server `env` maps wholesale — so resolved
    /// secrets never reach a recorded pin.
    ///
    /// # Errors
    ///
    /// [`ManifestError::UnknownProfile`] when `profile` names nothing;
    /// [`ManifestError::OverlayRule`] for overlay-rule violations;
    /// [`ManifestError::MissingEnvVar`] / [`ManifestError::InvalidReference`]
    /// from interpolation; and any validation error on the merged document.
    pub fn resolve(
        &self,
        profile: Option<&str>,
        env: &dyn EnvResolver,
    ) -> Result<ResolvedManifest, ManifestError> {
        let merged_value = self.merged_value(profile)?;
        let merged: Manifest = serde_yaml_ng::from_value(merged_value)
            .map_err(|error| ManifestError::from_merged(&error))?;
        validate::validate_pre_expansion(&merged)?;
        let source = serde_json::to_value(&merged).map_err(|error| projection_failure(&error))?;
        let mut expanded = source.clone();
        interpolate::expand_document(&mut expanded, env)?;
        let mut runtime = Self::typed_manifest(expanded.clone())?;
        validate::validate_post_expansion(&runtime)?;
        runtime.profiles.clear();
        let mut pinned =
            Self::typed_manifest(interpolate::restore_secret_references(expanded, &source))?;
        pinned.profiles.clear();
        restore_env_as_written(&mut pinned, &merged);
        let canonical =
            serde_json::to_value(&pinned).map_err(|error| projection_failure(&error))?;
        let rendered = canonical.to_string();
        Ok(ResolvedManifest {
            manifest: runtime,
            canonical_json: canonical,
            config_hash: fnv1a64(rendered.as_bytes()),
        })
    }

    /// Re-parse a JSON projection into the typed manifest.
    ///
    /// Used for both the runtime document and the pinning view; the
    /// projections are always structurally valid (one is an expansion of a
    /// serialized `Manifest`, the other a restore pass over it), so a
    /// re-parse failure is a bug report, not an author error.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Type`] if the projection ever fails to re-parse.
    fn typed_manifest(projection: serde_json::Value) -> Result<Manifest, ManifestError> {
        serde_json::from_value(projection).map_err(|error| ManifestError::Type {
            message: format!("the expanded document re-parsed badly: {error}"),
            line: None,
            column: None,
        })
    }

    /// The merged raw tree for the selected profile.
    ///
    /// # Errors
    ///
    /// [`ManifestError::UnknownProfile`] for an undeclared profile name;
    /// [`ManifestError::OverlayRule`] for a broken overlay; merge errors
    /// propagate unchanged.
    fn merged_value(&self, profile: Option<&str>) -> Result<Value, ManifestError> {
        match profile {
            Some(name) => {
                let overlay = self.overlay_for(name)?;
                guard_overlay(&overlay, name)?;
                merge::merged(&self.raw, &overlay)
            }
            None => Ok(self.raw.clone()),
        }
    }

    /// The raw overlay tree for `name`, or the unknown-profile error.
    ///
    /// # Errors
    ///
    /// [`ManifestError::UnknownProfile`] when no profile of that name is
    /// declared, listing what is.
    fn overlay_for(&self, name: &str) -> Result<Value, ManifestError> {
        self.typed
            .profiles
            .get(name)
            .map(|_| overlay_value(&self.raw, name))
            .ok_or_else(|| ManifestError::UnknownProfile {
                name: name.to_string(),
                available: self.typed.profiles.keys().cloned().collect(),
            })
    }
}

/// Extract the named profile's raw overlay tree from the document.
///
/// The typed [`Manifest::profiles`] map answers whether the profile exists;
/// the merge needs the raw tree (tags), so this digs the same node out of
/// the untyped document. An absent node yields an empty mapping, which the
/// overlay guard and merge treat as a no-op overlay.
fn overlay_value(raw: &Value, name: &str) -> Value {
    raw.as_mapping()
        .and_then(|mapping| mapping.get("profiles"))
        .and_then(Value::as_mapping)
        .and_then(|profiles| profiles.get(name))
        .cloned()
        .unwrap_or_else(|| Value::Mapping(serde_yaml_ng::Mapping::new()))
}

/// Restore every MCP server `env` map to its written form.
///
/// The pin answers "is this the same effective configuration as that run?"
/// — an `env` map's resolved values are runtime injection data, not
/// configuration identity, so the whole map pins as written (secret-bearing
/// entries are already restored by the reference pass; this covers the
/// rest). Everything else is the runtime document verbatim; both documents
/// have already dropped the profile declarations.
fn restore_env_as_written(pinned: &mut Manifest, merged: &Manifest) {
    for (name, server) in &mut pinned.mcp {
        if let Some(declared) = merged.mcp.get(name) {
            server.env = declared.env.clone();
        }
    }
}

/// Map a projection serialization failure to the typed error.
///
/// Unreachable with the current all-`String`-keyed types, but a silent
/// fallback here would skip the pinning (or the secret scan) — failing
/// loudly is the only safe direction.
fn projection_failure(error: &serde_json::Error) -> ManifestError {
    ManifestError::Type {
        message: format!("the document failed to serialize for resolution: {error}"),
        line: None,
        column: None,
    }
}

/// Reject overlays that break the no-touch rules.
///
/// `version` is fixed for the document and `profiles` cannot nest; both are
/// authoring mistakes with dedicated errors rather than merge surprises.
///
/// # Errors
///
/// [`ManifestError::OverlayRule`] naming the forbidden key.
fn guard_overlay(overlay: &Value, name: &str) -> Result<(), ManifestError> {
    let Some(mapping) = overlay.as_mapping() else {
        return Err(ManifestError::OverlayRule {
            path: format!("profiles.{name}"),
            detail: "a profile overlay must be a mapping".to_string(),
        });
    };
    for forbidden in ["version", "profiles"] {
        if mapping.contains_key(Value::String(forbidden.to_string())) {
            return Err(ManifestError::OverlayRule {
                path: format!("profiles.{name}.{forbidden}"),
                detail: format!("profiles may not touch `{forbidden}`"),
            });
        }
    }
    Ok(())
}

/// A validated, interpolated manifest plus its pinning hash.
///
/// Produced by [`ManifestDocument::resolve`]. Two documents live side by
/// side: the runtime manifest (every effective string interpolated, MCP
/// server `env` expanded for process wiring) and the pinning projection.
/// The pin drops the profile declarations and pins secret-consuming strings
/// as written — any string whose written form carries a `${secret:…}`
/// reference, plus MCP `env` maps wholesale — so the config hash is never a
/// function of the environment an unused profile names or of a secret's
/// value, whatever field the secret reference lives in.
#[derive(Debug, Clone)]
pub struct ResolvedManifest {
    /// The runtime manifest — every effective `${…}` reference resolved
    /// except the exempt `token_key` names.
    ///
    /// This is the wiring view: strings that consumed secret references
    /// (including MCP `env` values) are expanded, ready to hand to whatever
    /// process needs them, and must never be logged or recorded. Unapplied
    /// profiles are absent (an empty map) — this document is the effective
    /// configuration, not the declarations.
    manifest: Manifest,

    /// The canonical JSON projection the hash was computed over.
    ///
    /// Object keys are sorted (the types use `BTreeMap`s exclusively), so
    /// the same effective configuration always serializes identically
    /// across processes. Diff this — not the hash — to explain a pin
    /// mismatch; secret-consuming strings (including MCP `env`) appear here
    /// in their written reference form, never the resolved values.
    canonical_json: serde_json::Value,

    /// The FNV-1a 64-bit hash of the canonical JSON.
    ///
    /// An identity pin, not a security primitive — it answers "is this the
    /// same resolved config as that run?" and nothing else.
    config_hash: u64,
}

impl ResolvedManifest {
    /// The runtime manifest.
    ///
    /// Ready for the host to wire into engine configuration; every
    /// effective reference has been replaced or resolution failed. Strings
    /// that consumed secret references (including MCP `env` values) are
    /// expanded here — never log or record them; the pinning form lives in
    /// [`Self::canonical_json`].
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The canonical JSON projection the hash was computed over.
    ///
    /// Exposed so callers can diff two configurations structurally
    /// instead of byte-comparing hashes and guessing at the difference.
    #[must_use]
    pub fn canonical_json(&self) -> &serde_json::Value {
        &self.canonical_json
    }

    /// The resolved-config hash.
    ///
    /// Equal hashes mean identical resolved configurations (bar hash
    /// collisions, which FNV-1a 64 makes vanishingly unlikely for
    /// human-scale documents).
    #[must_use]
    pub fn config_hash(&self) -> u64 {
        self.config_hash
    }
}

/// FNV-1a 64-bit over `bytes`.
///
/// The same construction the demotion tags use (see `compact/demote.rs`),
/// kept module-private because it is an implementation detail of the pin.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "version: 1\nmetadata:\n  name: demo\nmodels:\n  entries:\n    primary:\n      provider: anthropic\n      model: claude-sonnet-4-6\n      token_key: ANTHROPIC_API_KEY\n  fallbacks: [primary]\n";

    #[test]
    fn a_clean_document_resolves_and_hashes_deterministically() {
        let document = ManifestDocument::parse(BASE).expect("the fixture parses");
        let first = document
            .resolve(None, &EnvMap::default())
            .expect("the fixture resolves");
        let second = document
            .resolve(None, &EnvMap::default())
            .expect("the fixture resolves again");
        assert_eq!(
            first.config_hash(),
            second.config_hash(),
            "the same resolved configuration must hash identically"
        );
        assert!(
            first.canonical_json().get("version").is_some(),
            "the canonical projection carries the document"
        );
    }

    #[test]
    fn a_changed_document_hashes_differently() {
        let first = ManifestDocument::parse(BASE)
            .expect("the fixture parses")
            .resolve(None, &EnvMap::default())
            .expect("the fixture resolves");
        let amended = ManifestDocument::parse(&format!("{BASE}budgets:\n  turns: 5\n"))
            .expect("the amended fixture parses")
            .resolve(None, &EnvMap::default())
            .expect("the amended fixture resolves");
        assert_ne!(
            first.config_hash(),
            amended.config_hash(),
            "a one-section difference must move the hash"
        );
    }

    #[test]
    fn a_newer_version_hard_errors_and_a_missing_version_names_the_field() {
        let newer = ManifestDocument::parse("version: 2\n").expect_err("version 2 is unreadable");
        assert!(
            matches!(
                newer,
                ManifestError::UnsupportedVersion {
                    found: 2,
                    supported: 1
                }
            ),
            "newer manifests hard-error, got: {newer:?}"
        );
        let missing =
            ManifestDocument::parse("metadata:\n  name: x\n").expect_err("version is required");
        assert!(
            matches!(missing, ManifestError::MissingVersion),
            "the missing version is named, got: {missing:?}"
        );
    }

    #[test]
    fn multi_document_yaml_is_refused() {
        let error = ManifestDocument::parse("version: 1\n---\nversion: 1\n")
            .expect_err("a manifest is exactly one document");
        assert!(
            matches!(error, ManifestError::Type { .. }),
            "the underlying multi-document rejection surfaces, got: {error:?}"
        );
    }

    #[test]
    fn an_unknown_profile_lists_what_exists() {
        let document = ManifestDocument::parse(BASE).expect("the fixture parses");
        let error = document
            .resolve(Some("absent"), &EnvMap::default())
            .expect_err("an undeclared profile cannot resolve");
        assert!(
            matches!(error, ManifestError::UnknownProfile { ref name, .. } if name == "absent"),
            "the missing profile is named, got: {error:?}"
        );
    }

    #[test]
    fn a_profile_touching_version_fails_at_parse_with_a_span() {
        let text = "version: 1\nprofiles:\n  tricky:\n    version: 2\n";
        let error = ManifestDocument::parse(text).expect_err("a profile may not declare version");
        let ManifestError::UnknownField {
            path, field, line, ..
        } = error
        else {
            panic!("the strict parse rejects the overlay key with a span, got: {error:?}");
        };
        assert_eq!(
            path, "profiles.tricky",
            "the error names the overlay holding the forbidden key"
        );
        assert_eq!(field, "version", "the forbidden key itself is named");
        assert!(
            line > 0,
            "the error carries the source position of the forbidden key"
        );
    }

    #[test]
    fn the_raw_overlay_guard_rejects_forbidden_keys_directly() {
        let overlay = serde_yaml_ng::from_str("version: 2\n").expect("the fixture parses");
        let guarded = guard_overlay(&overlay, "tricky");
        assert!(
            matches!(guarded, Err(ManifestError::OverlayRule { ref path, .. })
                if path == "profiles.tricky.version"),
            "the merge-path guard rejects version even when reached programmatically"
        );
        let nesting = serde_yaml_ng::from_str("profiles: {}\n").expect("the fixture parses");
        assert!(
            matches!(
                guard_overlay(&nesting, "tricky"),
                Err(ManifestError::OverlayRule { .. })
            ),
            "the merge-path guard rejects nested profiles too"
        );
    }
}
