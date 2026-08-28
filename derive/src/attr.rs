//! `#[tool(...)]` attribute parsing for the derive.

use syn::{Attribute, Expr, Lit, LitStr, Meta};

/// Container-level (`#[tool(...)]` on the struct) attributes.
///
/// Collected from every `#[tool(...)]` attribute on the derived
/// struct; unset members keep their `Default` (absent/false), and the
/// codegen in [`crate::expand`] reads them to decide what the
/// generated `impl Tool` contains.
#[derive(Default)]
pub(crate) struct ContainerAttrs {
    /// Override for the derived tool name.
    ///
    /// When `None`, the codegen falls back to the `snake_cased` struct
    /// identifier (`EchoInput` → `echo_input`). Non-empty and stable
    /// for the session, per the trait's contract.
    pub name: Option<String>,

    /// Override for the description.
    ///
    /// When `None`, the struct's `///` doc comment is used; when
    /// neither is present the derive errors — the trait requires a
    /// non-empty description.
    pub description: Option<String>,

    /// Emit `is_read_only() -> true`.
    ///
    /// The method override is generated only when the flag is set;
    /// otherwise the trait's default (`false`) applies untouched.
    pub read_only: bool,

    /// Emit `is_concurrency_safe() -> true`.
    ///
    /// Same conditional-generation rule as
    /// [`read_only`](Self::read_only): absent means the trait default.
    pub concurrency_safe: bool,

    /// Emit `system_prompt() -> Some(...)`.
    ///
    /// The tool's extra LLM hint, surfaced to the model verbatim.
    /// `None` leaves the trait's default (`None`) in place.
    pub system_prompt: Option<String>,

    /// Name of the handler `call` dispatches to (default: `run`).
    ///
    /// The generated `call` resolves the handler by this name as an
    /// inherent method on the struct; a wrong name surfaces as a
    /// normal "no method named …" compiler error at the call site.
    pub handler: Option<String>,

    /// Omit `additionalProperties: false` from the schema.
    ///
    /// The schema closes the world by default (strict-mode
    /// friendly); tools that accept open-ended input set this to
    /// advertise the absence of the flag instead.
    pub allow_extra: bool,
}

/// Field-level (`#[tool(...)]` on a field) attributes.
///
/// One instance per named field of the derived struct, collected the
/// same way as [`ContainerAttrs`]; the schema generator consults
/// them per field.
#[derive(Default)]
pub(crate) struct FieldAttrs {
    /// JSON property name override.
    ///
    /// Mirrors serde's `#[serde(rename)]` for the schema side only —
    /// the two must agree for deserialization to match the schema.
    pub name: Option<String>,

    /// Property description override.
    ///
    /// Falls back to the field's `///` doc comment; when neither is
    /// present the property simply carries no `description` key.
    pub description: Option<String>,

    /// Exclude the field from the schema and the required array.
    ///
    /// Valid only on fields that deserialize without model input —
    /// `Option<T>` or `#[serde(default)]`; the derive enforces this
    /// with a spanned error.
    pub skip: bool,

    /// Keep the property but omit it from the required array.
    ///
    /// Distinct from [`skip`](Self::skip): the property stays
    /// advertised, the call just succeeds without it. Pairs with
    /// `#[serde(default)]` on the Rust side.
    pub default: bool,
}

const CONTAINER_KEYS: &str =
    "name, description, read_only, concurrency_safe, system_prompt, handler, allow_extra";
const FIELD_KEYS: &str = "name, description, skip, default";

/// Parse the container-level `#[tool(...)]` attributes.
///
/// Walks every `tool` attribute on the struct with syn's nested-meta
/// parser, so diagnostics carry the offending attribute's span. Later
/// attributes win for repeated keys.
///
/// # Errors
///
/// Returns a spanned error for malformed or unknown attributes.
pub(crate) fn parse_container(attrs: &[Attribute]) -> syn::Result<ContainerAttrs> {
    let mut out = ContainerAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("tool")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                out.name = Some(string_value(&meta)?);
            } else if meta.path.is_ident("description") {
                out.description = Some(string_value(&meta)?);
            } else if meta.path.is_ident("system_prompt") {
                out.system_prompt = Some(string_value(&meta)?);
            } else if meta.path.is_ident("handler") {
                out.handler = Some(string_value(&meta)?);
            } else if meta.path.is_ident("read_only") {
                out.read_only = true;
            } else if meta.path.is_ident("concurrency_safe") {
                out.concurrency_safe = true;
            } else if meta.path.is_ident("allow_extra") {
                out.allow_extra = true;
            } else {
                return Err(meta.error(format!(
                    "unknown `tool` attribute; expected one of: {CONTAINER_KEYS}"
                )));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Parse the field-level `#[tool(...)]` attributes.
///
/// Same nested-meta walk as
/// [`parse_container`](fn@parse_container), over one field's
/// attributes.
///
/// # Errors
///
/// Returns a spanned error for malformed or unknown attributes.
pub(crate) fn parse_field(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("tool")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                out.name = Some(string_value(&meta)?);
            } else if meta.path.is_ident("description") {
                out.description = Some(string_value(&meta)?);
            } else if meta.path.is_ident("skip") {
                out.skip = true;
            } else if meta.path.is_ident("default") {
                out.default = true;
            } else {
                return Err(meta.error(format!(
                    "unknown `tool` attribute; expected one of: {FIELD_KEYS}"
                )));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Read a `key = "value"` string from a nested meta item.
///
/// Used by both parsers for every value-shaped attribute; rejects
/// non-string-literal values with the value's span.
///
/// # Errors
///
/// Returns a spanned error when the value is not a string literal.
fn string_value(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<String> {
    let value = meta.value()?;
    let lit: LitStr = value.parse()?;
    Ok(lit.value())
}

/// The `///` doc comment text of an item, joined across lines, if any.
///
/// Each line is trimmed and the lines are joined with single spaces,
/// so a multi-line `///` paragraph reads as one sentence chain in the
/// generated description.
pub(crate) fn doc_string(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(expr) = &nv.value
            && let Lit::Str(s) = &expr.lit
        {
            lines.push(s.value().trim().to_string());
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" "))
    }
}

/// Whether the field carries a `#[serde(default)]`-shaped attribute.
///
/// The schema-side condition for `#[tool(skip)]` validity and for
/// omitting a field from `required`: a field the runtime accepts
/// without is not truly required. Matches the `default` key whether
/// it is a bare flag or `default = "path"`.
pub(crate) fn has_serde_default(attrs: &[Attribute]) -> bool {
    let mut found = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("default") {
                found = true;
            }
            Ok(())
        });
    }
    found
}
