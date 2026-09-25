//! Profile overlay merging over the raw YAML value tree.
//!
//! The merge deliberately runs on [`Value`] rather than the typed model:
//! `!append` tags are only visible at the value level, and merge semantics
//! (scalars replace, maps merge, lists replace unless `!append`) are a
//! property of the document shape, not of any single type.

use serde_yaml_ng::value::Tag;
use serde_yaml_ng::value::TaggedValue;
use serde_yaml_ng::{Mapping, Value};

use super::error::ManifestError;

/// The tag that switches a list overlay from replace to append.
///
/// Written `!append` in YAML; the leading `!` is not significant to the tag
/// comparison, matching the underlying library's semantics.
const APPEND_TAG: &str = "append";

/// Merge `overlay` over `base`, returning the merged tree.
///
/// Scalars and sequences in the overlay replace the base value; mappings
/// merge recursively; an `!append`-tagged sequence concatenates onto the
/// base sequence. The `profiles` and `version` keys are rejected by the
/// overlay guard before this runs, so the merge never sees them.
///
/// # Errors
///
/// [`ManifestError::OverlayRule`] when `!append` targets a missing or
/// non-sequence base node, or when an unrecognized tag appears.
pub(crate) fn merged(base: &Value, overlay: &Value) -> Result<Value, ManifestError> {
    merge_node(base, overlay, "<root>")
}

/// Merge one node pair at `path`, the recursive core of [`merged`].
///
/// # Errors
///
/// Propagates whatever the mapping merge or the tag handler reports.
fn merge_node(base: &Value, overlay: &Value, path: &str) -> Result<Value, ManifestError> {
    match overlay {
        Value::Mapping(overlay_map) => match base {
            Value::Mapping(base_map) => merge_mappings(base_map, overlay_map, path),
            _ => merge_mappings(&Mapping::new(), overlay_map, path),
        },
        Value::Tagged(tagged) => append_tagged(base, tagged, path),
        _ => Ok(overlay.clone()),
    }
}

/// Merge two mappings key by key.
///
/// Keys present only in the overlay are inserted as-is (minus any `!append`
/// tag — appending to a key the base never declared is an error, caught by
/// [`append_tagged`]); keys present in both recurse.
///
/// # Errors
///
/// [`ManifestError::OverlayRule`] for a tag rule broken at any child key.
fn merge_mappings(base: &Mapping, overlay: &Mapping, path: &str) -> Result<Value, ManifestError> {
    let mut merged = base.clone();
    for (key, overlay_value) in overlay {
        let Some(key_name) = key.as_str() else {
            continue;
        };
        let child_path = child_path(path, key_name);
        let next = match merged.get(key) {
            Some(base_value) => merge_node(base_value, overlay_value, &child_path)?,
            None => insert_new_key(overlay_value, &child_path)?,
        };
        merged.insert(key.clone(), next);
    }
    Ok(Value::Mapping(merged))
}

/// Resolve an overlay value for a key the base does not declare.
///
/// An `!append` here has nothing to append to, so it fails as an overlay
/// rule; anything else inserts verbatim.
///
/// # Errors
///
/// [`ManifestError::OverlayRule`] when the new key's value is tagged.
fn insert_new_key(overlay_value: &Value, child_path: &str) -> Result<Value, ManifestError> {
    match overlay_value {
        Value::Tagged(tagged) => append_tagged(&Value::Null, tagged, child_path),
        Value::Mapping(map) => merge_mappings(&Mapping::new(), map, child_path),
        _ => Ok(overlay_value.clone()),
    }
}

/// Apply an `!append`-tagged overlay onto its base node.
///
/// The base must already hold a sequence; appending to a missing or
/// non-sequence node is an overlay-rule error because the operation the
/// author asked for cannot be performed. Any tag other than `append` is
/// rejected the same way — unknown tags are authoring mistakes, not data.
///
/// # Errors
///
/// [`ManifestError::OverlayRule`] for an unknown tag or a missing or
/// non-sequence append target.
fn append_tagged(base: &Value, tagged: &TaggedValue, path: &str) -> Result<Value, ManifestError> {
    if tagged.tag != Tag::new(APPEND_TAG) {
        return Err(ManifestError::OverlayRule {
            path: path.to_string(),
            detail: format!(
                "unknown tag `{}`; only !append is meaningful in an overlay",
                tagged.tag
            ),
        });
    }
    match (base, &tagged.value) {
        (Value::Sequence(base_items), Value::Sequence(overlay_items)) => {
            let mut combined = base_items.clone();
            combined.extend(overlay_items.iter().cloned());
            Ok(Value::Sequence(combined))
        }
        _ => Err(ManifestError::OverlayRule {
            path: path.to_string(),
            detail: "!append needs a list on both the base and the overlay".to_string(),
        }),
    }
}

/// Build the dotted child path of `key` under `path`.
///
/// The root marker is replaced by the first real key so overlay-rule
/// errors name real paths from the first key down.
fn child_path(path: &str, key: &str) -> String {
    if path == "<root>" {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(text: &str) -> Value {
        serde_yaml_ng::from_str(text).expect("test fixture parses")
    }

    #[test]
    fn scalars_replace_and_maps_merge_recursively() {
        let base = yaml("a: 1\nnested:\n  x: 1\n  y: 2\n");
        let overlay = yaml("a: 2\nnested:\n  y: 3\n");
        let merged = merged(&base, &overlay).expect("the merge succeeds");
        let mapping = merged.as_mapping().expect("the root stays a mapping");
        let nested = mapping
            .get("nested")
            .and_then(Value::as_mapping)
            .expect("nested stays a mapping");
        assert_eq!(
            mapping.get("a"),
            Some(&Value::Number(2.into())),
            "overlay scalars replace base scalars"
        );
        assert_eq!(
            nested.get("x"),
            Some(&Value::Number(1.into())),
            "map keys the overlay omits survive"
        );
        assert_eq!(
            nested.get("y"),
            Some(&Value::Number(3.into())),
            "map keys the overlay names are replaced"
        );
    }

    #[test]
    fn lists_replace_by_default_and_append_on_tag() {
        let base = yaml("plain: [a, b]\ntagged: [a, b]\n");
        let overlay = yaml("plain: [c]\ntagged: !append [c]\n");
        let merged = merged(&base, &overlay).expect("the merge succeeds");
        let mapping = merged.as_mapping().expect("the root stays a mapping");
        assert_eq!(
            mapping.get("plain"),
            Some(&yaml("[c]")),
            "an untagged list overlay replaces the base list wholesale"
        );
        assert_eq!(
            mapping.get("tagged"),
            Some(&yaml("[a, b, c]")),
            "an !append-tagged list overlay concatenates onto the base list"
        );
    }

    #[test]
    fn append_to_a_missing_key_is_an_overlay_rule_error() {
        let base = yaml("other: 1\n");
        let overlay = yaml("fresh: !append [x]\n");
        let error = merged(&base, &overlay).expect_err("appending to nothing cannot succeed");
        assert!(
            matches!(error, ManifestError::OverlayRule { ref path, .. } if path == "fresh"),
            "the error names the offending key, got: {error:?}"
        );
    }

    #[test]
    fn tags_below_a_new_section_are_checked() {
        let base = yaml("version: 1\n");
        let overlay = yaml("permissions:\n  rules:\n    deny: !append [x]\n");
        let error = merged(&base, &overlay)
            .expect_err("appending below a section the base never declared is still an error");
        assert!(
            matches!(error, ManifestError::OverlayRule { ref path, .. }
                if path == "permissions.rules.deny"),
            "the error names the nested offending key, got: {error:?}"
        );
    }

    #[test]
    fn tags_below_a_non_mapping_base_are_checked() {
        let base = yaml("permissions: 5\n");
        let overlay = yaml("permissions:\n  rules:\n    deny: !merge [x]\n");
        let error = merged(&base, &overlay)
            .expect_err("an unknown tag below a replaced scalar is still an error");
        assert!(
            matches!(error, ManifestError::OverlayRule { .. }),
            "unknown tags are rejected at every depth, got: {error:?}"
        );
    }

    #[test]
    fn unknown_tags_are_rejected() {
        let base = yaml("key: [a]\n");
        let overlay = yaml("key: !merge [b]\n");
        let error = merged(&base, &overlay).expect_err("unknown tags are authoring mistakes");
        assert!(
            matches!(error, ManifestError::OverlayRule { .. }),
            "unknown tags fail as overlay-rule violations, got: {error:?}"
        );
    }
}
