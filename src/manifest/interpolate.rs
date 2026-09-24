//! Compose-style interpolation over the merged document.
//!
//! The expansion runs on the JSON projection of the typed manifest rather
//! than a hand-written per-field walk, so every present and future string
//! field is interpolated by one code path. Two exemptions exist:
//! `token_key` values (an environment variable NAME, never a value) and
//! the top-level `profiles` section (declarations the run did not select —
//! their references resolve when that profile is chosen, not before).

use serde_json::Value as Json;

use super::EnvResolver;
use super::error::ManifestError;

/// The key whose values are exempt from interpolation.
///
/// `token_key` holds the NAME of the environment variable that stores a key;
/// expanding it would at best mangle it and at worst inline a secret.
const EXEMPT_KEY: &str = "token_key";

/// The root-level section never interpolated.
///
/// Profiles are declarations: an unapplied overlay's `${…}` references name
/// the environment that profile is *for*, so expanding them against the
/// current environment would both fail resolves that should succeed and
/// make the resolved document depend on overlays the run never selected.
const EXEMPT_SECTION: &str = "profiles";

/// The marker a written string carries when it consumes a secret.
///
/// Any string containing this marker pins as written — the pinning pass
/// restores it from the pre-expansion document so a resolved secret never
/// reaches a recorded pin, whatever field it lives in.
const SECRET_MARKER: &str = "${secret:";

/// Restore every secret-consuming string of `expanded` to its written form.
///
/// Walks the expanded projection and its pre-expansion source in lockstep
/// (expansion replaces string values in place, so the two trees always have
/// the same shape); wherever the written string contains a `${secret:…}`
/// reference, the written form wins. References in strings that carry no
/// secret stay expanded; a string carrying any secret reference pins
/// wholly as written, plain references included.
pub(crate) fn restore_secret_references(expanded: Json, source: &Json) -> Json {
    restore_node(expanded, source)
}

/// Restore one node pair, the recursive core of [`restore_secret_references`].
///
/// Shape-mismatched pairs (impossible today, since expansion only replaces
/// string values in place) fall through unchanged rather than panicking —
/// the pin degrades to the runtime value, never to a crash. Arrays guard
/// their length for exactly this reason: `zip` alone would truncate the
/// longer side, silently dropping entries from the pin.
fn restore_node(expanded: Json, source: &Json) -> Json {
    match (expanded, source) {
        (Json::String(_), Json::String(written)) if written.contains(SECRET_MARKER) => {
            Json::String(written.clone())
        }
        (Json::Array(items), Json::Array(written_items)) if items.len() == written_items.len() => {
            Json::Array(
                items
                    .into_iter()
                    .zip(written_items)
                    .map(|(item, written)| restore_node(item, written))
                    .collect(),
            )
        }
        (Json::Object(fields), Json::Object(written_fields)) => Json::Object(
            fields
                .into_iter()
                .map(|(key, value)| {
                    let restored = match written_fields.get(&key) {
                        Some(written) => restore_node(value, written),
                        None => value,
                    };
                    (key, restored)
                })
                .collect(),
        ),
        (expanded, _) => expanded,
    }
}

/// Expand every reference in every effective string of the merged document.
///
/// Walks the JSON projection in place; values under a `token_key` key and
/// the root-level `profiles` section are skipped, everything else expands.
/// Strings without a `$` are untouched.
///
/// # Errors
///
/// [`ManifestError::MissingEnvVar`] for an unresolvable reference,
/// [`ManifestError::InvalidReference`] for malformed reference syntax. The
/// document is left partially expanded on error — the caller discards it.
pub(crate) fn expand_document(
    document: &mut Json,
    resolver: &dyn EnvResolver,
) -> Result<(), ManifestError> {
    expand_node(document, resolver, "<root>")
}

/// Expand one node at `path`, the recursive core of [`expand_document`].
///
/// # Errors
///
/// Propagates the first reference failure found anywhere below this node.
fn expand_node(
    node: &mut Json,
    resolver: &dyn EnvResolver,
    path: &str,
) -> Result<(), ManifestError> {
    match node {
        Json::String(text) => {
            if text.contains('$') {
                *text = expand_string(text, resolver, path)?;
            }
            Ok(())
        }
        Json::Array(items) => {
            for (idx, item) in items.iter_mut().enumerate() {
                expand_node(item, resolver, &array_path(path, idx))?;
            }
            Ok(())
        }
        Json::Object(fields) => {
            for (key, item) in fields.iter_mut() {
                if key == EXEMPT_KEY || (path == "<root>" && key == EXEMPT_SECTION) {
                    continue;
                }
                expand_node(item, resolver, &child_path(path, key))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Expand the references inside one string.
///
/// Grammar: `${VAR}`, `${VAR:-default}`, `${VAR:?message}`, `${secret:NAME}`,
/// with `$${` escaping to a literal `${` and `$$` to a literal `$`. Any other
/// `$` passes through untouched.
///
/// # Errors
///
/// [`ManifestError::MissingEnvVar`] when a reference without a usable default
/// does not resolve; [`ManifestError::InvalidReference`] for malformed
/// reference syntax.
pub(crate) fn expand_string(
    input: &str,
    resolver: &dyn EnvResolver,
    path: &str,
) -> Result<String, ManifestError> {
    let mut rest = String::with_capacity(input.len());
    let mut expanded = input;
    while let Some(dollar) = expanded.find('$') {
        rest.push_str(&expanded[..dollar]);
        let after_dollar = &expanded[dollar.saturating_add(1)..];
        if let Some(next) = after_dollar.strip_prefix('$') {
            rest.push('$');
            expanded = next;
        } else if let Some(remaining) = after_dollar.strip_prefix('{') {
            let Some((body, next)) = split_reference(remaining) else {
                return Err(ManifestError::InvalidReference {
                    path: path.to_string(),
                    reference: remaining.to_string(),
                });
            };
            rest.push_str(&resolve_reference(body, resolver, path)?);
            expanded = next;
        } else {
            rest.push('$');
            expanded = after_dollar;
        }
    }
    rest.push_str(expanded);
    Ok(rest)
}

/// Split one `${…}` reference into its body and the text after it.
///
/// Returns `Some((body, remainder))` with the body stripped of braces and
/// the remainder starting after the closing `}`; `None` when no closing
/// brace exists — an unterminated reference is malformed, never a lookup.
fn split_reference(rest: &str) -> Option<(&str, &str)> {
    let end = rest.find('}')?;
    Some((&rest[..end], &rest[end.saturating_add(1)..]))
}

/// How a reference behaves when the variable does not resolve.
///
/// Distinguishes the three postures a reference can take toward a miss:
/// hard-fail, substitute a default, or fail with the author's message.
enum ReferenceFallback<'a> {
    /// No fallback: the missing variable is a hard error.
    ///
    /// The `${VAR}` shape — a missing variable means the configuration is
    /// incomplete, not merely different.
    None,

    /// `${VAR:-value}` substitutes `value`.
    ///
    /// The default is taken verbatim; it is not itself scanned for
    /// nested references.
    Default(&'a str),

    /// `${VAR:?message}` fails with the author's message.
    ///
    /// The message rides along into the error so the operator sees why
    /// the variable was required, not just that it was.
    Required(&'a str),
}

/// Resolve one reference body against the resolver.
///
/// # Errors
///
/// [`ManifestError::InvalidReference`] when the body does not match the
/// reference grammar; [`ManifestError::MissingEnvVar`] when it matches but
/// nothing resolves and no default applies.
fn resolve_reference(
    body: &str,
    resolver: &dyn EnvResolver,
    path: &str,
) -> Result<String, ManifestError> {
    if let Some(name) = body.strip_prefix("secret:") {
        let name = valid_name(name, path, body)?;
        return resolver
            .resolve(name)
            .ok_or_else(|| missing_var(name, path, None));
    }
    let (name, fallback) = reference_parts(body);
    let name = valid_name(name, path, body)?;
    match (resolver.resolve(name), fallback) {
        (Some(value), _) => Ok(value),
        (None, ReferenceFallback::Default(value)) => Ok(value.to_string()),
        (None, ReferenceFallback::Required(message)) => Err(missing_var(name, path, Some(message))),
        (None, ReferenceFallback::None) => Err(missing_var(name, path, None)),
    }
}

/// Split a reference body into its name and fallback behavior.
///
/// `:-` wins over `:?` when both appear — the first separator found
/// reading left to right defines the shape, matching compose's behavior.
fn reference_parts(body: &str) -> (&str, ReferenceFallback<'_>) {
    if let Some((name, value)) = body.split_once(":-") {
        (name, ReferenceFallback::Default(value))
    } else if let Some((name, message)) = body.split_once(":?") {
        (name, ReferenceFallback::Required(message))
    } else {
        (body, ReferenceFallback::None)
    }
}

/// Validate a variable name, reporting the offending body on failure.
///
/// Names are ASCII identifiers: an alphabetic or `_` first character, then
/// alphanumeric or `_`. Anything else (including an empty name) is an
/// invalid-reference error naming the body it came from.
///
/// # Errors
///
/// [`ManifestError::InvalidReference`] when the name is not a valid
/// identifier.
fn valid_name<'a>(name: &'a str, path: &str, body: &'a str) -> Result<&'a str, ManifestError> {
    let valid = !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_');
    if valid {
        Ok(name)
    } else {
        Err(ManifestError::InvalidReference {
            path: path.to_string(),
            reference: body.to_string(),
        })
    }
}

/// Build the missing-variable error shared by plain and secret references.
///
/// One constructor so both reference kinds fail with the same variant
/// and the same field semantics.
fn missing_var(name: &str, path: &str, detail: Option<&str>) -> ManifestError {
    ManifestError::MissingEnvVar {
        variable: name.to_string(),
        path: path.to_string(),
        detail: detail.map(str::to_string),
    }
}

/// Build the dotted child path of `key` under `path`.
///
/// The root marker is replaced by the first real key so paths read
/// `models.entries`, not `<root>.models.entries`.
fn child_path(path: &str, key: &str) -> String {
    if path == "<root>" {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

/// Build the indexed child path of `idx` under `path`.
///
/// Bracketed so list members are visually distinct from map keys in
/// error paths.
fn array_path(path: &str, idx: usize) -> String {
    format!("{path}[{idx}]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn resolver_with(values: &[(&str, &str)]) -> BTreeMap<String, String> {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn expand(input: &str, vars: &BTreeMap<String, String>) -> Result<String, ManifestError> {
        expand_string(input, vars, "test.path")
    }

    #[test]
    fn plain_references_substitute_and_missing_ones_name_the_variable() {
        let vars = resolver_with(&[("ROOT", "/data")]);
        assert_eq!(
            expand("root is ${ROOT}", &vars).expect("a set variable resolves"),
            "root is /data"
        );
        let error = expand("needs ${ABSENT}", &vars).expect_err("an unset variable must fail");
        assert!(
            matches!(error, ManifestError::MissingEnvVar { ref variable, ref path, .. }
                if variable == "ABSENT" && path == "test.path"),
            "the error names the variable and the path, got: {error:?}"
        );
    }

    #[test]
    fn defaults_apply_and_required_messages_ride_along() {
        let empty = resolver_with(&[]);
        assert_eq!(
            expand("${HOST:-localhost}", &empty).expect("a default covers a missing variable"),
            "localhost"
        );
        let error = expand("${KEY:?set KEY first}", &empty)
            .expect_err("a required variable without a value must fail");
        assert!(
            matches!(error, ManifestError::MissingEnvVar { ref variable, ref detail, .. }
                if variable == "KEY" && detail.as_deref() == Some("set KEY first")),
            "the author's message survives into the error, got: {error:?}"
        );
    }

    #[test]
    fn secret_references_resolve_and_unresolved_ones_fail() {
        let vars = resolver_with(&[("GITHUB_TOKEN", "gh-value")]);
        assert_eq!(
            expand("token=${secret:GITHUB_TOKEN}", &vars).expect("a set secret resolves"),
            "token=gh-value"
        );
        let error = expand("${secret:MISSING}", &vars).expect_err("a missing secret must fail");
        assert!(
            matches!(error, ManifestError::MissingEnvVar { ref variable, .. } if variable == "MISSING"),
            "secret references name their variable, got: {error:?}"
        );
    }

    #[test]
    fn dollar_escapes_pass_through_literally() {
        let vars = resolver_with(&[]);
        assert_eq!(
            expand("cost is 5$ and brace is $${LITERAL}", &vars).expect("escapes never resolve"),
            "cost is 5$ and brace is ${LITERAL}"
        );
    }

    #[test]
    fn malformed_references_are_rejected_with_their_body() {
        let vars = resolver_with(&[]);
        let error = expand("${not a name}", &vars).expect_err("spaces are not valid in a name");
        assert!(
            matches!(error, ManifestError::InvalidReference { ref reference, .. }
                if reference == "not a name"),
            "the offending body is reported, got: {error:?}"
        );
        let unterminated =
            expand("trailing ${OPEN", &vars).expect_err("an unterminated reference is malformed");
        assert!(
            matches!(unterminated, ManifestError::InvalidReference { .. }),
            "unterminated references fail loudly, got: {unterminated:?}"
        );
    }

    #[test]
    fn the_restore_pass_keeps_plain_expansions_and_rewrites_secret_bearers() {
        let source = serde_json::json!({
            "plain": "${HOST:-localhost}",
            "mixed": "db://${secret:PW}@${HOST}",
            "untouched": "no references"
        });
        let mut expanded = source.clone();
        let vars = resolver_with(&[("HOST", "h1"), ("PW", "pw-value")]);
        expand_document(&mut expanded, &vars).expect("both references resolve");
        let restored = restore_secret_references(expanded, &source);
        assert_eq!(
            restored.get("plain").and_then(Json::as_str),
            Some("h1"),
            "plain references stay expanded — the pin tracks effective configuration"
        );
        assert_eq!(
            restored.get("mixed").and_then(Json::as_str),
            Some("db://${secret:PW}@${HOST}"),
            "a string containing any secret reference pins wholly as written, plain refs included"
        );
        assert_eq!(
            restored.get("untouched").and_then(Json::as_str),
            Some("no references"),
            "reference-free strings pass through untouched"
        );
    }

    #[test]
    fn a_length_mismatched_array_falls_through_unchanged() {
        let expanded = serde_json::json!(["a", "${secret:PW}", "tail"]);
        let shorter_source = serde_json::json!(["a"]);
        let restored = restore_secret_references(expanded.clone(), &shorter_source);
        assert_eq!(
            restored, expanded,
            "a shape-mismatched pair keeps the runtime value whole — never a truncated pin"
        );
    }

    #[test]
    fn the_document_walk_exempts_token_key_values() {
        let vars = resolver_with(&[("NAMED_KEY", "resolved")]);
        let mut document = serde_json::json!({
            "models": { "entries": { "primary": {
                "token_key": "ANTHROPIC_API_KEY",
                "base_url": "${NAMED_KEY}/v1"
            }}}
        });
        expand_document(&mut document, &vars).expect("both fields are walked");
        let primary = document
            .pointer("/models/entries/primary")
            .expect("the fixture shape holds");
        assert_eq!(
            primary.get("token_key").and_then(Json::as_str),
            Some("ANTHROPIC_API_KEY"),
            "token_key is a name and must never be expanded"
        );
        assert_eq!(
            primary.get("base_url").and_then(Json::as_str),
            Some("resolved/v1"),
            "every other string expands"
        );
    }
}
