//! Error types for the manifest model.
//!
//! [`ManifestError`] carries the position of the offending YAML node wherever
//! the strict string-level parse can supply one ([`Span`]); errors that can
//! only arise after the profile merge report the field path without a span
//! because the merged document never existed as source text.

use serde_yaml_ng::Error as YamlError;
use serde_yaml_ng::Location;
use std::fmt;

/// A one-based position in the manifest source text.
///
/// Populated from the strict parse pass, where the underlying YAML deserializer
/// reports the line and column of the node that failed. Errors raised on the
/// merged document (which never existed as text) carry no span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// One-based line of the offending node.
    ///
    /// Points at the key of an unknown field, the value of a type mismatch, or
    /// wherever the YAML deserializer surfaced the error.
    pub line: usize,

    /// One-based column of the offending node.
    ///
    /// Counts characters from the start of the line, matching the underlying
    /// deserializer's reporting.
    pub column: usize,
}

impl Span {
    /// Build a span from an optional [`Location`].
    ///
    /// Returns `None` when the deserializer could not attribute a position —
    /// the caller then treats the error as span-less rather than inventing a
    /// position.
    #[must_use]
    fn from_location(location: Option<Location>) -> Option<Self> {
        location.map(|loc| Self {
            line: loc.line(),
            column: loc.column(),
        })
    }
}

/// Everything that can go wrong loading, merging, or validating a manifest.
///
/// The enum is `#[non_exhaustive]`; match with a wildcard arm so new variants
/// can be added additively. Display is hand-rolled (not thiserror) because
/// several variants render optional context — spans and suggestions — that
/// format-string interpolation cannot express cleanly.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ManifestError {
    /// The document contains a key no v1 type declares.
    ///
    /// Carries the dotted field path, the offending key, its source span, and
    /// the closest expected name when one is within edit distance.
    UnknownField {
        /// Dotted path to the mapping that holds the unknown key.
        ///
        /// Built from the serde message prefix, so it names where the key
        /// lives rather than where the parser stopped.
        path: String,

        /// The offending key exactly as written.
        ///
        /// Unmodified from the source document — suggestions live in their
        /// own field, this one is evidence.
        field: String,

        /// The closest documented field name, when one is close enough.
        ///
        /// `None` when nothing is within edit distance; an honest blank
        /// beats a nonsense guess.
        suggestion: Option<String>,

        /// One-based line of the offending key, when the strict pass saw it.
        ///
        /// Zero when the error came from the merged document, which never
        /// existed as text.
        line: usize,

        /// One-based column of the offending key, when the strict pass saw it.
        ///
        /// Points at the key itself, not the mapping around it.
        column: usize,
    },

    /// A value could not be deserialized into its declared type, or failed a
    /// value-shape rule that has no dedicated variant.
    ///
    /// The span is present when the failure came from the strict parse pass
    /// and absent when it came from the merged document.
    Type {
        /// Human-readable description of the mismatch.
        ///
        /// Carries the underlying deserializer wording so no diagnostic
        /// detail is lost in translation.
        message: String,

        /// Source line of the failure, when known.
        ///
        /// Present only for strict-pass failures; merged-document failures
        /// have no position to report.
        line: Option<usize>,

        /// Source column of the failure, when known.
        ///
        /// Pairs with `line`; both are set or both are absent.
        column: Option<usize>,
    },

    /// The document declares a manifest version this crate cannot read.
    ///
    /// Newer manifests hard-error rather than best-effort parse; older ones
    /// would migrate, but v1 is the first version so nothing older exists.
    UnsupportedVersion {
        /// The version integer found in the document.
        ///
        /// Reported verbatim so the operator sees the gap between their
        /// document and this build.
        found: u32,

        /// The version this crate implements.
        ///
        /// Currently always `1`; the field exists so the message stays
        /// truthful when v2 lands.
        supported: u32,
    },

    /// The document omits the required `version` key.
    ///
    /// Detected during the strict parse; every other check presumes a
    /// version was agreed on.
    MissingVersion,

    /// An interpolation reference could not be resolved.
    ///
    /// Names the variable so the operator can see exactly which environment
    /// entry is absent.
    MissingEnvVar {
        /// The variable name inside the unresolved reference.
        ///
        /// Exactly the identifier the author wrote, which is the env entry
        /// to add or the typo to fix.
        variable: String,

        /// Dotted path to the string holding the reference.
        ///
        /// Locates the field, not the character offset — spans do not
        /// survive the merge.
        path: String,

        /// The author-supplied `${VAR:?message}` text, when present.
        ///
        /// The message is the author talking to their future self about
        /// why this variable matters.
        detail: Option<String>,
    },

    /// An interpolation reference is malformed.
    ///
    /// For example an unterminated `${`, or a variable name that does not
    /// match the identifier grammar.
    InvalidReference {
        /// Dotted path to the string holding the reference.
        ///
        /// Locates the field, not the character offset — spans do not
        /// survive the merge.
        path: String,

        /// The offending reference text.
        ///
        /// The malformed body without braces, so the error shows what the
        /// parser actually saw.
        reference: String,
    },

    /// A string in the document looks like a literal API key.
    ///
    /// Manifests carry references to secrets, never secret values; the shape
    /// names the matched key family.
    LiteralSecret {
        /// Dotted path to the offending string.
        ///
        /// Names the field carrying the key-shaped value so the author can
        /// jump straight to it.
        path: String,

        /// The matched key family, such as `sk-` or `AKIA`.
        ///
        /// Names the family rather than echoing the key — the value itself
        /// must never appear in logs or errors.
        shape: String,
    },

    /// A name reference points at nothing in the document.
    ///
    /// Model fallback chains, tool ids, and MCP server names are all
    /// cross-checked; this error names the kind, the missing name, and where
    /// the dangling reference lives.
    UnresolvedReference {
        /// What kind of name was dangling: model, tool, or MCP server.
        ///
        /// Human-readable so error text reads naturally without a variant
        /// mapping table.
        kind: String,

        /// The name that resolved to nothing.
        ///
        /// Verbatim from the document; paired with the path it appeared at.
        name: String,

        /// Dotted path to the reference site.
        ///
        /// Names the field holding the dangling reference.
        path: String,
    },

    /// The requested profile does not exist in the document.
    ///
    /// Lists the available names so the operator can see the typo.
    UnknownProfile {
        /// The requested profile name.
        ///
        /// What the caller passed to resolve, not something the document
        /// declared.
        name: String,

        /// The profiles the document actually declares, sorted.
        ///
        /// Sorted because it exists to be shown; a stable order makes the
        /// error diffable.
        available: Vec<String>,
    },

    /// A profile overlay breaks an overlay rule.
    ///
    /// Overlays may not touch `version`, may not nest `profiles`, and `!append`
    /// is only meaningful on a list the base document already declares.
    OverlayRule {
        /// Dotted path to the offending overlay node.
        ///
        /// Points inside `profiles.<name>` at the rule-breaking key.
        path: String,

        /// What rule was broken.
        ///
        /// One sentence naming the violated overlay rule.
        detail: String,
    },

    /// A tool entry is neither a builtin nor an MCP reference, or is both.
    ///
    /// Every entry needs exactly one source of implementation.
    InvalidToolEntry {
        /// The `id` of the offending entry.
        ///
        /// The author-declared handle, not a list index — ids are what the
        /// rest of the document references.
        id: String,

        /// What rule was broken.
        ///
        /// One sentence naming the violated rule.
        detail: String,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownField {
                path,
                field,
                suggestion,
                line,
                column,
            } => {
                write!(f, "unknown field `{field}` at {path} ({line}:{column})")?;
                match suggestion {
                    Some(name) => write!(f, "; did you mean `{name}`?"),
                    None => Ok(()),
                }
            }
            Self::Type {
                message,
                line,
                column,
            } => {
                write!(f, "{message}")?;
                match (line, column) {
                    (Some(line), Some(column)) => write!(f, " at line {line} column {column}"),
                    _ => Ok(()),
                }
            }
            Self::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "unsupported manifest version {found}; this crate reads version {supported}"
                )
            }
            Self::MissingVersion => {
                write!(f, "missing required field `version`")
            }
            Self::MissingEnvVar {
                variable,
                path,
                detail,
            } => {
                write!(
                    f,
                    "unresolved environment reference `${{{variable}}}` at {path}"
                )?;
                match detail {
                    Some(message) => write!(f, ": {message}"),
                    None => Ok(()),
                }
            }
            Self::InvalidReference { path, reference } => {
                write!(
                    f,
                    "malformed interpolation reference at {path}: {reference}"
                )
            }
            Self::LiteralSecret { path, shape } => {
                write!(
                    f,
                    "literal secret-shaped value at {path} (matched {shape}); manifests reference secrets, they never embed them"
                )
            }
            Self::UnresolvedReference { kind, name, path } => {
                write!(f, "unresolved {kind} reference `{name}` at {path}")
            }
            Self::UnknownProfile { name, available } => {
                write!(
                    f,
                    "unknown profile `{name}`; available: {}",
                    available.join(", ")
                )
            }
            Self::OverlayRule { path, detail } => {
                write!(f, "profile overlay rule violated at {path}: {detail}")
            }
            Self::InvalidToolEntry { id, detail } => {
                write!(f, "invalid tool entry `{id}`: {detail}")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

impl ManifestError {
    /// Convert a strict-pass deserialization failure.
    ///
    /// Extracts the unknown field name, its path, and the expected-field list
    /// from serde's stable `` unknown field `x` `` message shape so a
    /// Levenshtein suggestion can be computed; every other failure becomes
    /// [`ManifestError::Type`] with the original message and span intact.
    pub(crate) fn from_yaml(error: &YamlError) -> Self {
        let message = error.to_string();
        let span = Span::from_location(error.location());
        let (line, column) = match span {
            Some(found) => (Some(found.line), Some(found.column)),
            None => (None, None),
        };
        Self::from_message(&message, line, column)
    }

    /// Convert a merged-document deserialization failure.
    ///
    /// The merged document never existed as text, so no span can be attached;
    /// unknown-field failures keep their name, path, and suggestion.
    pub(crate) fn from_merged(error: &YamlError) -> Self {
        Self::from_message(&error.to_string(), None, None)
    }

    /// Shared tail of the two deserialization-failure conversions.
    ///
    /// Both callers hand over the raw message plus whatever position they
    /// could attribute; this fn owns the message-shape decisions once.
    fn from_message(message: &str, line: Option<usize>, column: Option<usize>) -> Self {
        if let Some(parsed) = parse_unknown_field(message) {
            let suggestion = closest_field(&parsed.field, &parsed.expected);
            return Self::UnknownField {
                path: parsed.path,
                field: parsed.field,
                suggestion,
                line: line.unwrap_or(0),
                column: column.unwrap_or(0),
            };
        }
        if message.starts_with("missing field `version`") {
            return Self::MissingVersion;
        }
        Self::Type {
            message: strip_position(message),
            line,
            column,
        }
    }
}

/// The pieces of serde's unknown-field message this module cares about.
///
/// A tiny struct rather than a tuple so the three parts cannot be
/// transposed at the call site.
struct UnknownFieldMessage {
    /// Dotted path prefix, empty for a top-level key.
    ///
    /// `"<root>"` when the message carries no path — never an empty
    /// string, so error rendering never special-cases it.
    path: String,

    /// The offending field name.
    ///
    /// The bare name after any path prefix is split off.
    field: String,

    /// The expected field names enumerated by serde.
    ///
    /// Both spellings serde emits — `one of` lists and `or` pairs — land
    /// here as one flat list.
    expected: Vec<String>,
}

/// Parse serde's unknown-field message into its parts.
///
/// Handles both shapes `serde` emits — ``path.to.map: unknown field `x`,
/// expected one of `a`, `b` `` for nested keys and ``unknown field `x`,
/// expected `a` or `b` `` at the root. Returns `None` for any other message
/// shape; the caller then falls back to the generic [`ManifestError::Type`]
/// mapping so no information is lost.
fn parse_unknown_field(message: &str) -> Option<UnknownFieldMessage> {
    let body = strip_position(message);
    let (path, rest) = located_field(&body)?;
    let (field, expected_part) = rest.split_once("`, expected")?;
    let names = expected_part.trim_start_matches(" one of ");
    let expected = names
        .split(" or ")
        .flat_map(|disjunct| disjunct.split(", "))
        .map(|name| name.trim_matches(['`', ' ']).to_string())
        .filter(|name| !name.is_empty())
        .collect();
    Some(UnknownFieldMessage {
        path,
        field: field.to_string(),
        expected,
    })
}

/// Split the located field name out of either message shape.
///
/// Nested keys carry their path before the colon; root-level keys have none,
/// and anything else is not an unknown-field message.
fn located_field(body: &str) -> Option<(String, &str)> {
    if let Some((path, rest)) = body.split_once(": unknown field `") {
        return Some((path.to_string(), rest));
    }
    let rest = body.strip_prefix("unknown field `")?;
    Some((String::from("<root>"), rest))
}

/// Drop the trailing ` at line L column C` position `serde_yaml` appends.
///
/// The position is carried separately by [`Span`]; keeping it inside the
/// message would duplicate it in the rendered error.
fn strip_position(message: &str) -> String {
    match message.find(" at line ") {
        Some(idx) => String::from(&message[..idx]),
        None => String::from(message),
    }
}

/// The closest expected name within a small edit distance.
///
/// Returns `None` when nothing is close enough to be a plausible typo — an
/// honest "no idea" beats a nonsense suggestion.
fn closest_field(field: &str, expected: &[String]) -> Option<String> {
    expected
        .iter()
        .map(|name| (name.as_str(), levenshtein(field, name)))
        .min_by_key(|(_, distance)| *distance)
        .and_then(|(name, distance)| {
            let budget = (field.chars().count().saturating_add(2) / 3).max(2);
            (distance <= budget).then(|| name.to_string())
        })
}

/// Classic edit distance between two strings, by Unicode scalar.
///
/// Case-sensitive and symmetric; used only for suggestions, so the specific
/// distance values never surface in behavior contracts.
fn levenshtein(a: &str, b: &str) -> usize {
    let target: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = Vec::with_capacity(target.len().saturating_add(1));
    previous.extend(0..=target.len());
    for (row, source_char) in a.chars().enumerate() {
        let mut current = Vec::with_capacity(target.len().saturating_add(1));
        current.push(row.saturating_add(1));
        for (col, target_char) in target.iter().enumerate() {
            let substitution_cost = usize::from(source_char != *target_char);
            let from_delete = previous
                .get(col)
                .map_or(usize::MAX, |value| value.saturating_add(1));
            let from_insert = current
                .get(col)
                .map_or(usize::MAX, |value| value.saturating_add(1));
            let from_substitute = previous
                .get(col)
                .zip(previous.get(col.saturating_add(1)))
                .map_or(usize::MAX, |(diagonal, above)| {
                    diagonal
                        .saturating_add(substitution_cost)
                        .min(above.saturating_add(1))
                });
            current.push(from_delete.min(from_insert).min(from_substitute));
        }
        previous = current;
    }
    previous.last().copied().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_unknown_field_message_yields_path_field_and_suggestion() {
        let parsed = parse_unknown_field(
            "models.entries.primary: unknown field `modelx`, expected one of `provider`, `model`, `max_tokens` at line 5 column 5",
        )
        .expect("serde's nested unknown-field message shape is parseable");
        assert_eq!(
            parsed.path, "models.entries.primary",
            "the path prefix before the colon survives"
        );
        assert_eq!(parsed.field, "modelx", "the bare field name is extracted");
        assert_eq!(
            parsed.expected,
            ["provider", "model", "max_tokens"],
            "every disjunct becomes a candidate without serde's connective words"
        );
        assert_eq!(
            closest_field("modelx", &parsed.expected),
            Some("model".to_string()),
            "the one-edit neighbor wins"
        );
    }

    #[test]
    fn a_root_level_unknown_field_message_parses_with_root_path() {
        let parsed = parse_unknown_field(
            "unknown field `versoin`, expected `version` or `models` at line 2 column 1",
        )
        .expect("serde's root unknown-field message shape is parseable");
        assert_eq!(parsed.path, "<root>", "a keyless message reports the root");
        assert_eq!(
            parsed.expected,
            ["version", "models"],
            "the two-name `or` shape splits into candidates"
        );
        assert_eq!(
            closest_field("versoin", &parsed.expected),
            Some("version".to_string()),
            "the transposed typo suggests its neighbor"
        );
    }

    #[test]
    fn other_messages_parse_to_none_and_fall_back() {
        assert!(
            parse_unknown_field("version: invalid type: string \"one\", expected u64").is_none(),
            "type mismatches are not unknown-field messages"
        );
    }

    #[test]
    fn edit_distance_counts_substitutions_deletions_and_insertions() {
        assert_eq!(levenshtein("model", "model"), 0);
        assert_eq!(levenshtein("modelx", "model"), 1);
        assert_eq!(levenshtein("mode", "model"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn distant_names_get_no_suggestion() {
        assert!(
            closest_field("unwatched", &["models".to_string()]).is_none(),
            "an unrelated name must not produce a nonsense suggestion"
        );
    }
}
