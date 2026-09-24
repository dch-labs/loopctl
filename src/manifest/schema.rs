//! The JSON Schema export for the v1 manifest.
//!
//! One function, generated from the same serde types that parse the
//! document, so the schema can never drift from the parser. The
//! `loopctl schema` verb that prints it ships with the CLI.

use super::error::ManifestError;
use super::types::Manifest;

/// The JSON Schema for the v1 manifest, as canonical JSON.
///
/// Generated via schemars from [`Manifest`] and its section types; every
/// object type emits `additionalProperties: false` because the structs are
/// `deny_unknown_fields` (closed string enums are strict through their
/// `enum` list instead), which is exactly the strictness the parser
/// enforces.
///
/// # Errors
///
/// Fails only if schema serialization fails, which the schemars data model
/// does not permit in practice; the `Result` keeps this function honest
/// under the crate's no-panic policy.
pub fn manifest_json_schema() -> Result<serde_json::Value, ManifestError> {
    let schema = schemars::schema_for!(Manifest);
    serde_json::to_value(&schema).map_err(|error| ManifestError::Type {
        message: format!("the generated schema failed to serialize: {error}"),
        line: None,
        column: None,
    })
}
