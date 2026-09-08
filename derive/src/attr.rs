//! `#[tool(...)]` attribute parsing for the derive.
//!
//! Parses the container-level attributes (name, description,
//! `read_only`, `concurrency_safe`, `system_prompt`, `handler`,
//! `allow_extra`)
//! and the field-level attributes (`name`, `description`, `skip`,
//! `default`),
//! plus the serde attributes the schema mirrors (`rename`,
//! `rename_all`, `default`). All parsers use syn's nested-meta walker
//! so diagnostics carry the offending attribute's span.

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

/// Accepted keys on the container-level `#[tool(...)]` attribute.
///
/// Unknown keys are rejected with this list in the error so a typo
/// fails at compile time, not as a silently ignored attribute.
const CONTAINER_KEYS: &str =
    "name, description, read_only, concurrency_safe, system_prompt, handler, allow_extra";
/// Accepted keys on a field-level `#[tool(...)]` attribute.
///
/// Unknown keys are rejected with this list in the error.
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
                flag_without_value(&meta, "read_only")?;
                out.read_only = true;
            } else if meta.path.is_ident("concurrency_safe") {
                flag_without_value(&meta, "concurrency_safe")?;
                out.concurrency_safe = true;
            } else if meta.path.is_ident("allow_extra") {
                flag_without_value(&meta, "allow_extra")?;
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
                flag_without_value(&meta, "skip")?;
                out.skip = true;
            } else if meta.path.is_ident("default") {
                flag_without_value(&meta, "default")?;
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

/// Reject a value on a flag-shaped `#[tool(...)]` key.
///
/// `read_only`, `concurrency_safe`, `allow_extra`, `skip`, and
/// `default` are presence flags; a stray `= value` would otherwise
/// surface as a bare syntax error (or, worse, silently invert the
/// flag's meaning), so misuse fails with the key's own span and name.
///
/// # Errors
///
/// Returns a spanned error when the key carries a value.
fn flag_without_value(meta: &syn::meta::ParseNestedMeta<'_>, key: &str) -> syn::Result<()> {
    if meta.value().is_ok() {
        Err(meta.error(format!("`{key}` takes no value")))
    } else {
        Ok(())
    }
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

/// The `Meta` items inside every `#[serde(…)]` attribute, in order.
///
/// A malformed serde attribute is skipped wholesale, so every serde-key
/// check built on this walks the same arguments under the same policy —
/// an unparseable key a host carries never aborts the walk for the
/// well-formed ones.
pub(crate) fn serde_metas(attrs: &[Attribute]) -> Vec<Meta> {
    let mut metas = Vec::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        if let Ok(list) = attr
            .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            metas.extend(list);
        }
    }
    metas
}

/// Whether the attrs carry a `#[serde(default)]`-shaped attribute.
///
/// The schema-side condition for `#[tool(skip)]` validity and for
/// omitting a field from `required`: a field the runtime accepts
/// without is not truly required. Works on a field's attrs and on the
/// container's — a struct-level `#[serde(default)]` fills every missing
/// field the same way. Matches the `default` key whether it is a bare
/// flag or `default = "path"`.
pub(crate) fn has_serde_default(attrs: &[Attribute]) -> bool {
    serde_metas(attrs).iter().any(|meta| match meta {
        Meta::Path(path) => path.is_ident("default"),
        Meta::NameValue(nv) => nv.path.is_ident("default"),
        Meta::List(_) => false,
    })
}

/// The `#[serde(rename = "…")]` value on a field, if any.
///
/// The schema mirrors serde's rename for deserialization consistency —
/// a mismatch between the schema's property name and the key serde
/// looks for would make the schema lie about what the model should
/// send.
pub(crate) fn serde_rename(attrs: &[Attribute]) -> Option<String> {
    let mut plain = None;
    let mut deserialize = None;
    for meta in serde_metas(attrs) {
        match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                if let Expr::Lit(expr) = &nv.value
                    && let Lit::Str(lit) = &expr.lit
                {
                    plain = Some(lit.value());
                }
            }
            Meta::List(ml) if ml.path.is_ident("rename") => {
                let Ok(inner) = syn::parse::Parser::parse2(
                    &syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                    ml.tokens.clone(),
                ) else {
                    continue;
                };
                for meta in &inner {
                    if let Meta::NameValue(nv) = meta
                        && nv.path.is_ident("deserialize")
                        && let Expr::Lit(expr) = &nv.value
                        && let Lit::Str(lit) = &expr.lit
                    {
                        deserialize = Some(lit.value());
                    }
                }
            }
            _ => {}
        }
    }
    deserialize.or(plain)
}

/// The `#[serde(rename_all = "…")]` strategy on the struct, if any.
///
/// Applied to each field's Rust name to derive its JSON property name,
/// exactly as serde deserializes it.
pub(crate) fn serde_rename_all(attrs: &[Attribute]) -> Option<RenameAll> {
    let mut out = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename_all")
                && let Ok(lit) = meta.value()?.parse::<LitStr>()
            {
                out = RenameAll::from_str(&lit.value());
            }
            Ok(())
        });
    }
    out
}

/// The `#[serde(rename_all = "…")]` casing strategies.
///
/// Mirrored from serde so the derive honours the same casing names a
/// `Deserialize` input struct already declares.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RenameAll {
    /// The `lowercase` strategy — field names lowercased, ASCII-only.
    ///
    /// Like serde's field arm, only ASCII case changes: no separators
    /// are inserted or removed, and non-ASCII characters keep their
    /// shape.
    Lower,

    /// The `UPPERCASE` strategy — field names uppercased.
    ///
    /// Words keep their positions; only case changes.
    Upper,

    /// The `PascalCase` strategy — each word capitalized, no separators.
    ///
    ///
    /// Word boundaries come from the field name's existing separators.
    Pascal,

    /// The `camelCase` strategy — first word lowercase, rest capitalized.
    ///
    ///
    /// The conventional local-variable and JSON-field shape.
    Camel,

    /// The `snake_case` strategy — underscore-separated lowercase words.
    ///
    ///
    /// The conventional Rust field shape, and the derive's default.
    Snake,

    /// The `SCREAMING_SNAKE_CASE` strategy — underscore-separated uppercase.
    ///
    ///
    /// The conventional Rust constant shape.
    ScreamingSnake,

    /// The `kebab-case` strategy — hyphen-separated lowercase words.
    ///
    ///
    /// The conventional CLI-flag and HTML-attribute shape.
    Kebab,

    /// The `SCREAMING-KEBAB-CASE` strategy — hyphen-separated uppercase.
    ///
    ///
    /// Rare in practice; accepted for serde parity.
    ScreamingKebab,
}

impl RenameAll {
    /// Parse the serde casing name into the strategy.
    ///
    /// Returns `None` for unrecognized names (serde itself errors in
    /// that case; the derive then ignores the attribute).
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "lowercase" => Some(Self::Lower),
            "UPPERCASE" => Some(Self::Upper),
            "PascalCase" => Some(Self::Pascal),
            "camelCase" => Some(Self::Camel),
            "snake_case" => Some(Self::Snake),
            "SCREAMING_SNAKE_CASE" => Some(Self::ScreamingSnake),
            "kebab-case" => Some(Self::Kebab),
            "SCREAMING-KEBAB-CASE" => Some(Self::ScreamingKebab),
            _ => None,
        }
    }

    /// Apply the strategy to a field name, exactly as serde's
    /// `rename_all` does for struct fields.
    ///
    /// The input is the Rust field identifier; the output is the JSON
    /// property name serde will look for during deserialization. The
    /// separator-inserting rules apply to *variants* in serde — for
    /// fields, `snake_case` is the identity and `lowercase` lowercases
    /// ASCII-only, `PascalCase` capitalizes after each underscore, and
    /// the screaming/kebab rules are plain case and separator
    /// conversions; mirroring those exactly is what keeps the schema
    /// from lying about the wire name.
    pub(crate) fn apply(self, name: &str) -> String {
        match self {
            Self::Lower => name.to_ascii_lowercase(),
            Self::Snake => name.to_owned(),
            Self::Upper | Self::ScreamingSnake => name.to_ascii_uppercase(),
            Self::Pascal => to_pascal_case(name),
            Self::Camel => {
                let pascal = to_pascal_case(name);
                let mut chars = pascal.chars();
                match chars.next() {
                    Some(first) => String::from(first.to_ascii_lowercase()) + chars.as_str(),
                    None => String::new(),
                }
            }
            Self::Kebab => name.replace('_', "-"),
            Self::ScreamingKebab => name.to_ascii_uppercase().replace('_', "-"),
        }
    }
}

/// `PascalCase` the way serde's field rule builds it.
///
/// Underscores disappear and the character after each one is
/// capitalized; everything else passes through unchanged, matching
/// serde's own loop character for character.
fn to_pascal_case(name: &str) -> String {
    let mut pascal = String::new();
    let mut capitalize = true;
    for ch in name.chars() {
        if ch == '_' {
            capitalize = true;
        } else if capitalize {
            pascal.extend(ch.to_uppercase());
            capitalize = false;
        } else {
            pascal.push(ch);
        }
    }
    pascal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_all_covers_the_full_serde_strategy_set() {
        let cases: Vec<(&str, &str, &str)> = vec![
            ("lowercase", "FileName", "filename"),
            ("UPPERCASE", "FileName", "FILENAME"),
            ("PascalCase", "file_name", "FileName"),
            ("camelCase", "file_name", "fileName"),
            ("snake_case", "FileName", "FileName"),
            ("SCREAMING_SNAKE_CASE", "FileName", "FILENAME"),
            ("kebab-case", "file_name", "file-name"),
            ("SCREAMING-KEBAB-CASE", "file_name", "FILE-NAME"),
        ];
        for (name, input, expected) in cases {
            let strategy =
                RenameAll::from_str(name).unwrap_or_else(|| panic!("unknown strategy: {name}"));
            assert_eq!(
                strategy.apply(input),
                expected,
                "strategy {name:?} on {input:?}"
            );
        }
    }

    #[test]
    fn separator_rules_are_identities_on_fields_matching_serde() {
        assert_eq!(
            RenameAll::from_str("snake_case").unwrap().apply("userID"),
            "userID",
            "serde's field arm for snake_case is the identity — the \
            separator-inserting logic applies to variants, not fields"
        );
        assert_eq!(
            RenameAll::from_str("lowercase").unwrap().apply("FileName"),
            "filename",
            "serde's field arm for lowercase lowercases — only snake_case \
            is the identity on fields, so the schema must advertise the \
            lowercased key serde will actually look for"
        );
        assert_eq!(
            RenameAll::from_str("camelCase").unwrap().apply("user_ID"),
            "userID",
            "camelCase is pascal-case-after-underscore with the first \
            character lowered, character for character like serde"
        );
    }

    #[test]
    fn screaming_kebab_accepts_only_serdes_hyphenated_name() {
        assert!(RenameAll::from_str("SCREAMING-KEBAB-CASE").is_some());
        assert!(
            RenameAll::from_str("SCREAMING_KEBAB_CASE").is_none(),
            "serde rejects the underscore spelling; the derive must not \
            accept a name serde itself would refuse"
        );
    }

    #[test]
    fn has_serde_default_survives_value_bearing_keys_before_it() {
        use syn::parse_quote;
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(with = "humantime_serde", default)]
        };
        assert!(
            has_serde_default(&attrs),
            "a value-bearing key before `default` must not abort the walk"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(skip_serializing_if = "Option::is_none", default)]
        };
        assert!(has_serde_default(&attrs));
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(default = "default_path")]
        };
        assert!(has_serde_default(&attrs));
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(rename = "other")]
        };
        assert!(!has_serde_default(&attrs));
    }

    #[test]
    fn flag_keys_reject_values_with_a_named_error() {
        use syn::parse_quote;
        let attrs: Vec<Attribute> = parse_quote! {
            #[tool(read_only = false)]
        };
        let err = parse_container(&attrs)
            .err()
            .expect("a flag with a value must fail");
        assert!(
            err.to_string().contains("`read_only` takes no value"),
            "the error names the key: {err}"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[tool(skip = false)]
        };
        let err = parse_field(&attrs)
            .err()
            .expect("a flag with a value must fail");
        assert!(
            err.to_string().contains("`skip` takes no value"),
            "the error names the key: {err}"
        );
    }

    #[test]
    fn serde_rename_deserialize_form_is_preferred() {
        use syn::parse_quote;
        let attr: Attribute = parse_quote! {
            #[serde(rename(deserialize = "from_wire", serialize = "to_wire"))]
        };
        assert_eq!(
            serde_rename(&[attr]),
            Some("from_wire".to_string()),
            "the deserialize half wins over the serialize half"
        );
        let attr: Attribute = parse_quote! {
            #[serde(rename = "simple")]
        };
        assert_eq!(serde_rename(&[attr]), Some("simple".to_string()));
    }

    #[test]
    fn rename_all_from_str_rejects_unknown_names() {
        assert!(RenameAll::from_str("NonsenseCase").is_none());
        assert!(RenameAll::from_str("").is_none());
    }
}
