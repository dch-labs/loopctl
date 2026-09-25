//! Structural and semantic validation of the merged manifest.
//!
//! Validation runs on the merged, pre-interpolation document: version gate,
//! name cross-references, tool-entry shape, rule-string shape, and the
//! literal-secret scan. Interpolation reference checks happen in the resolve
//! pipeline through the same walk the expansion uses.

use serde_json::Value as Json;

use super::error::ManifestError;
use super::types::Manifest;

/// The manifest version this crate implements.
///
/// A named constant rather than a literal so the version gate, the
/// schema export, and the docs cite one number.
pub(crate) const SUPPORTED_VERSION: u32 = 1;

/// Literal key shapes rejected by the secret scan.
///
/// Each entry is a well-known key family prefix and the minimum total length
/// a string carrying it must reach before it is treated as a literal key —
/// short strings that merely start like a key stay legal. Deliberately
/// deterministic prefixes only: no entropy heuristic, so legitimate
/// high-entropy config values (sha256 pins, base64 blobs) never trip it.
const SECRET_SHAPES: &[(&str, usize)] = &[
    ("sk-", 20),
    ("AKIA", 16),
    ("eyJ", 20),
    ("ghp_", 20),
    ("github_pat_", 30),
    ("xoxb-", 20),
];

/// Validate the merged manifest before interpolation.
///
/// These checks read the document as written: the version gate (no
/// reference can appear in a number), the tool-entry exclusive-or
/// (defaulting a reference cannot change `Some`-ness), finite numbers, and
/// the literal-secret scan — which must see the written form, including
/// keys hiding in `${VAR:-sk-…}` defaults that expansion would carry into
/// the runtime document verbatim.
///
/// # Errors
///
/// [`ManifestError::UnsupportedVersion`] for any version other than the
/// supported one; [`ManifestError::InvalidToolEntry`] for tool entries
/// without exactly one implementation source; [`ManifestError::Type`] for
/// non-finite numbers; [`ManifestError::LiteralSecret`] for key-shaped
/// literals.
pub(crate) fn validate_pre_expansion(manifest: &Manifest) -> Result<(), ManifestError> {
    check_version(manifest)?;
    check_tool_entries(manifest)?;
    check_finite_numbers(manifest)?;
    check_literal_secrets(manifest)
}

/// Validate the expanded manifest after interpolation.
///
/// These checks read what the run will actually use: names resolve (or
/// fail) against their expanded values, so a defaulted
/// `tools[].mcp: ${SERVER:-github}` checks `github`, not the reference
/// syntax — and a rule that only had its shape before expansion (`${BAD:-x}`
/// expanding to a shapeless string) is rejected here, where the gate
/// engine would otherwise receive it malformed.
///
/// # Errors
///
/// [`ManifestError::UnresolvedReference`] for dangling model, tool, or
/// MCP-server names; [`ManifestError::Type`] for malformed rule strings.
pub(crate) fn validate_post_expansion(manifest: &Manifest) -> Result<(), ManifestError> {
    check_cross_references(manifest)?;
    check_rule_shapes(manifest)
}

/// Enforce the version gate.
///
/// Runs at parse time (fail on the unreadable document before anything else
/// happens) and again in [`validate`] on the merged document — the merge
/// cannot change `version` because overlays may not touch it, so the second
/// check is defense in depth.
///
/// # Errors
///
/// [`ManifestError::UnsupportedVersion`] for any version other than the
/// supported one.
pub(crate) fn check_version(manifest: &Manifest) -> Result<(), ManifestError> {
    if manifest.version == SUPPORTED_VERSION {
        Ok(())
    } else {
        Err(ManifestError::UnsupportedVersion {
            found: manifest.version,
            supported: SUPPORTED_VERSION,
        })
    }
}

/// Enforce the builtin-XOR-mcp rule on every tool entry.
///
/// # Errors
///
/// [`ManifestError::InvalidToolEntry`] for the first entry that sets both
/// or neither of `builtin` and `mcp`.
fn check_tool_entries(manifest: &Manifest) -> Result<(), ManifestError> {
    for (idx, entry) in manifest.tools.iter().enumerate() {
        let exclusive = entry.builtin.is_some() ^ entry.mcp.is_some();
        if !exclusive {
            let detail = "exactly one of `builtin` or `mcp` must be set".to_string();
            return Err(ManifestError::InvalidToolEntry {
                id: entry.id.clone(),
                detail: format!("{detail} at tools[{idx}]"),
            });
        }
    }
    Ok(())
}

/// Resolve every name cross-reference.
///
/// Fallback chains point at named models, agent tool ids point at tool
/// entries, and MCP-backed tool entries point at declared servers — a
/// manifest that references a name it never declares cannot run, so it does
/// not validate.
///
/// # Errors
///
/// [`ManifestError::UnresolvedReference`] for the first dangling name.
fn check_cross_references(manifest: &Manifest) -> Result<(), ManifestError> {
    for (idx, name) in manifest.models.fallbacks.iter().enumerate() {
        if !manifest.models.entries.contains_key(name) {
            return Err(ManifestError::UnresolvedReference {
                kind: "model".to_string(),
                name: name.clone(),
                path: format!("models.fallbacks[{idx}]"),
            });
        }
    }
    for (idx, name) in manifest.agent.tools.iter().enumerate() {
        if !manifest.tools.iter().any(|entry| &entry.id == name) {
            return Err(ManifestError::UnresolvedReference {
                kind: "tool".to_string(),
                name: name.clone(),
                path: format!("agent.tools[{idx}]"),
            });
        }
    }
    for (idx, entry) in manifest.tools.iter().enumerate() {
        if let Some(server) = &entry.mcp
            && !manifest.mcp.contains_key(server)
        {
            return Err(ManifestError::UnresolvedReference {
                kind: "MCP server".to_string(),
                name: server.clone(),
                path: format!("tools[{idx}].mcp"),
            });
        }
    }
    Ok(())
}

/// Enforce the `tool:pattern` shape on every rule string.
///
/// # Errors
///
/// [`ManifestError::Type`] for the first rule without a non-empty tool name
/// and pattern on either side of one colon.
fn check_rule_shapes(manifest: &Manifest) -> Result<(), ManifestError> {
    let mut rules: Vec<(&str, String)> = Vec::new();
    for (idx, rule) in manifest.permissions.rules.deny.iter().enumerate() {
        rules.push((rule.as_str(), format!("permissions.rules.deny[{idx}]")));
    }
    for (idx, rule) in manifest.permissions.rules.allow.iter().enumerate() {
        rules.push((rule.as_str(), format!("permissions.rules.allow[{idx}]")));
    }
    for (entry_idx, entry) in manifest.tools.iter().enumerate() {
        for (when_idx, rule) in entry.when.iter().enumerate() {
            rules.push((
                rule.as_str(),
                format!("tools[{entry_idx}].when[{when_idx}]"),
            ));
        }
    }
    for (rule, path) in rules {
        let shaped = rule
            .split_once(':')
            .is_some_and(|(tool, pattern)| !tool.is_empty() && !pattern.is_empty());
        if !shaped {
            return Err(ManifestError::Type {
                message: format!("rule `{rule}` at {path} must have the `tool:pattern` shape"),
                line: None,
                column: None,
            });
        }
    }
    Ok(())
}

/// Reject non-finite numbers in numeric fields.
///
/// YAML spells them `.nan`/`.inf`, and a declared number that silently
/// becomes *no value* is the worst possible reading of a typo — the error
/// names the field so the author sees exactly what to fix. Every `f64`
/// field in the model is covered: `budgets.cost_usd` and
/// `context.compaction.trigger`.
///
/// # Errors
///
/// [`ManifestError::Type`] naming the first non-finite field found.
fn check_finite_numbers(manifest: &Manifest) -> Result<(), ManifestError> {
    let mut offender: Option<&str> = None;
    if let Some(cost) = manifest.budgets.cost_usd
        && !cost.is_finite()
    {
        offender = Some("budgets.cost_usd");
    }
    if offender.is_none()
        && let Some(trigger) = manifest.context.compaction.trigger
        && !trigger.is_finite()
    {
        offender = Some("context.compaction.trigger");
    }
    match offender {
        Some(field) => Err(ManifestError::Type {
            message: format!("{field} must be a finite number"),
            line: None,
            column: None,
        }),
        None => Ok(()),
    }
}

/// Scan every string in the document for literal key shapes.
///
/// Runs on the merged, pre-interpolation document over the JSON projection,
/// so every present and future string field is scanned by one code path —
/// `token_key` included, which is exactly where a pasted key would land.
///
/// # Errors
///
/// [`ManifestError::LiteralSecret`] for the first key-shaped string found;
/// [`ManifestError::Type`] if the projection cannot be serialized — a
/// skipped scan is never an acceptable failure mode.
fn check_literal_secrets(manifest: &Manifest) -> Result<(), ManifestError> {
    let projection = serde_json::to_value(manifest).map_err(|error| ManifestError::Type {
        message: format!("the document failed to serialize for the secret scan: {error}"),
        line: None,
        column: None,
    })?;
    let mut offender: Option<ManifestError> = None;
    for_each_string(&projection, "<root>", &mut |text, path| {
        if offender.is_none()
            && let Some(shape) = matched_secret_shape(text)
        {
            offender = Some(ManifestError::LiteralSecret {
                path: path.to_string(),
                shape: shape.to_string(),
            });
        }
    });
    match offender {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The key family a string matches, if any.
///
/// Keys are matched per token — a run of identifier-ish characters — so a
/// literal key is caught after any separator (`Bearer sk-…`, `?key=AKIA…`),
/// and the token is matched with any leading dashes trimmed, which is what
/// a reference default's `:-` operator leaves in front of the value
/// (`${VAR:-sk-…}`). First match wins; the families are mutually exclusive
/// by prefix so order among them does not matter.
fn matched_secret_shape(text: &str) -> Option<&'static str> {
    text.split(|character: char| {
        !(character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
    })
    .find_map(|token| {
        let candidate = token.trim_start_matches('-');
        SECRET_SHAPES
            .iter()
            .find(|(prefix, minimum)| candidate.starts_with(prefix) && candidate.len() >= *minimum)
            .map(|(prefix, _)| *prefix)
    })
}

/// Visit every string in the JSON projection with its dotted path.
///
/// Shared walker for value-shape checks that must see all string data; the
/// interpolation module walks with its own recursion because it mutates and
/// skips exempt keys.
fn for_each_string(value: &Json, path: &str, visit: &mut dyn FnMut(&str, &str)) {
    match value {
        Json::String(text) => visit(text, path),
        Json::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                for_each_string(item, &format!("{path}[{idx}]"), visit);
            }
        }
        Json::Object(fields) => {
            for (key, item) in fields {
                let child = if path == "<root>" {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                for_each_string(item, &child, visit);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_families_match_only_at_plausible_lengths() {
        assert_eq!(
            matched_secret_shape("sk-proj-abcdefghijklmnop"),
            Some("sk-"),
            "a full-length OpenAI-style key matches"
        );
        assert_eq!(
            matched_secret_shape("sk-"),
            None,
            "a bare prefix is not yet a key"
        );
        assert_eq!(
            matched_secret_shape("AKIAIOSFODNN7EXAMPLE"),
            Some("AKIA"),
            "an AWS access key id matches"
        );
        assert_eq!(
            matched_secret_shape("sha256:9f2a7c1e5b8d3a6f0c4e2d1b7a9f3e5c8d2b1a4f6e0c3d5b7a9f1e3c"),
            None,
            "legitimate high-entropy config values never match"
        );
    }

    #[test]
    fn keys_after_a_separator_match_their_family() {
        assert_eq!(
            matched_secret_shape("Authorization: Bearer sk-proj-abcdefghijklmnop"),
            Some("sk-"),
            "a key after a separator is still a key"
        );
        assert_eq!(
            matched_secret_shape("postgres://user:ghp_abcdefghijklmnopqrstuvwxyz@host/db"),
            Some("ghp_"),
            "a key inside a connection string is still a key"
        );
        assert_eq!(
            matched_secret_shape("${OPENAI_KEY:-sk-proj-abcdefghijklmnop}"),
            Some("sk-"),
            "a literal key hiding in a reference default is still a key"
        );
    }
}
