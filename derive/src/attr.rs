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

/// Reject anything trailing a flag-shaped `#[tool(...)]` key.
///
/// `read_only`, `concurrency_safe`, `allow_extra`, `skip`, and
/// `default` are presence flags; a stray `= value` or the
/// parenthesized `flag(false)` form would otherwise surface as a bare
/// syntax error (or, worse, silently set the flag the user was trying
/// to negate), so misuse fails with the key's own span and name.
///
/// # Errors
///
/// Returns a spanned error when the key carries a value or a
/// parenthesized argument list.
fn flag_without_value(meta: &syn::meta::ParseNestedMeta<'_>, key: &str) -> syn::Result<()> {
    let takes_value = meta.value().is_ok() || meta.input.peek(syn::token::Paren);
    if takes_value {
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

/// Reject serde attributes that change what serde reads on the wire in
/// ways the generated schema cannot mirror.
///
/// The derive's contract is that the schema never disagrees with serde
/// on the wire. `rename`, `rename_all`, and `default` are mirrored;
/// these keys are not, and a schema generated alongside them would
/// advertise keys serde never looks for (or omit ones it requires) —
/// so they fail at derive time with the manual-impl escape hatch
/// instead of silently lying to the model.
///
/// Field-level read-side skip keys (`skip`, `skip_deserializing`) are
/// allowed when `#[tool(skip)]` also removes the field from the schema
/// — that combination is consistent; alone they advertise a field
/// serde ignores. Write-side keys (`skip_serializing`,
/// `skip_serializing_if`, `serialize_with`) change only what serde
/// emits, so a schema advertising the field stays truthful and they
/// pass. `with` and `deserialize_with` rewrite the wire format itself
/// and are rejected on the field side. The same split governs the
/// container: `into`/`try_into` convert only what serde emits and
/// pass, while `from`/`try_from` convert the deserialization input
/// and are rejected.
///
/// # Errors
///
/// Returns a spanned error naming the unsupported key.
pub(crate) fn check_unmirrorable_serde(
    attrs: &[Attribute],
    container: bool,
    tool_skip: bool,
) -> syn::Result<()> {
    const CONTAINER_KEYS: &[&str] = &[
        "from",
        "try_from",
        "remote",
        "transparent",
        "rename_all_fields",
    ];
    const SKIP_KEYS: &[&str] = &["skip", "skip_deserializing"];
    const WITH_KEYS: &[&str] = &["with", "deserialize_with"];
    for meta in serde_metas(attrs) {
        let name = match &meta {
            Meta::Path(path) => path.get_ident().map(std::string::ToString::to_string),
            Meta::NameValue(nv) => nv.path.get_ident().map(std::string::ToString::to_string),
            Meta::List(ml) => ml.path.get_ident().map(std::string::ToString::to_string),
        };
        let Some(name) = name else {
            continue;
        };
        let unmirrorable = if container {
            CONTAINER_KEYS.contains(&name.as_str())
        } else {
            name == "flatten"
                || (SKIP_KEYS.contains(&name.as_str()) && !tool_skip)
                || WITH_KEYS.contains(&name.as_str())
        };
        if unmirrorable {
            if !container && SKIP_KEYS.contains(&name.as_str()) {
                return Err(syn::Error::new_spanned(
                    &meta,
                    format!(
                        "`#[serde({name})]` leaves the schema advertising a field serde \
                         ignores — add `#[tool(skip)]` to remove it from the schema, or \
                         write a manual `Tool` impl"
                    ),
                ));
            }
            return Err(syn::Error::new_spanned(
                &meta,
                format!(
                    "`#[serde({name})]` changes what serde reads on the wire in ways \
                     `#[derive(Tool)]` cannot mirror — write a manual `Tool` impl for \
                     this input"
                ),
            ));
        }
    }
    Ok(())
}

/// Whether the attrs carry a read-side `#[serde(skip)]`-shaped key.
///
/// `skip` and `skip_deserializing` fill the field from
/// `Default::default()` without reading the wire, so they satisfy the
/// same fillability precondition as `#[serde(default)]` — the validity
/// check for `#[tool(skip)]` credits them here. Write-side
/// `skip_serializing` does not fill anything and is deliberately not
/// matched.
pub(crate) fn has_serde_skip(attrs: &[Attribute]) -> bool {
    serde_metas(attrs).iter().any(|meta| match meta {
        Meta::Path(path) => path.is_ident("skip") || path.is_ident("skip_deserializing"),
        Meta::NameValue(nv) => nv.path.is_ident("skip") || nv.path.is_ident("skip_deserializing"),
        Meta::List(ml) => ml.path.is_ident("skip") || ml.path.is_ident("skip_deserializing"),
    })
}

/// Whether the attrs carry `#[serde(deny_unknown_fields)]`.
///
/// Alone it matches the schema's default closed world
/// (`additionalProperties: false`), so it is not an unmirrorable key;
/// the caller rejects only its contradiction with
/// `#[tool(allow_extra)]`.
pub(crate) fn has_serde_deny_unknown(attrs: &[Attribute]) -> bool {
    serde_metas(attrs).iter().any(|meta| match meta {
        Meta::Path(path) => path.is_ident("deny_unknown_fields"),
        Meta::NameValue(nv) => nv.path.is_ident("deny_unknown_fields"),
        Meta::List(ml) => ml.path.is_ident("deny_unknown_fields"),
    })
}

/// The `#[serde(rename_all = "…")]` strategy on the struct, if any.
///
/// Applied to each field's Rust name to derive its JSON property name,
/// exactly as serde deserializes it. A strategy serde itself would
/// reject fails here too — falling back to raw field names would
/// advertise keys serde never looks for.
///
/// # Errors
///
/// Returns a spanned error when `rename_all` is present with a
/// strategy name the serde rule set does not contain.
pub(crate) fn serde_rename_all(attrs: &[Attribute]) -> syn::Result<Option<RenameAll>> {
    let mut out = None;
    for meta in serde_metas(attrs) {
        if let Meta::NameValue(nv) = &meta
            && nv.path.is_ident("rename_all")
            && let Expr::Lit(expr) = &nv.value
            && let Lit::Str(lit) = &expr.lit
        {
            match RenameAll::from_str(&lit.value()) {
                Some(strategy) => out = Some(strategy),
                None => {
                    return Err(syn::Error::new_spanned(
                        nv,
                        format!(
                            "unknown serde `rename_all` strategy `{}` — serde rejects this \
                             spelling, and the derive will not fall back to raw field names",
                            lit.value()
                        ),
                    ));
                }
            }
        }
    }
    Ok(out)
}

/// The `#[serde(rename_all = "…")]` casing strategies.
///
/// Mirrored from serde so the derive honours the same casing names a
/// `Deserialize` input struct already declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenameAll {
    /// The `lowercase` strategy — the identity on fields, like serde's.
    ///
    /// Serde's field arm treats `lowercase` and `snake_case` the same:
    /// the name passes through unchanged, because separator handling
    /// belongs to variant renaming.
    Lower,

    /// The `UPPERCASE` strategy — field names uppercased, ASCII-only.
    ///
    /// Words keep their positions; non-ASCII characters pass through
    /// untouched, exactly as in serde's field arm.
    Upper,

    /// The `PascalCase` strategy — each word capitalized, no separators.
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
    /// property name serde will look for during deserialization.
    /// Serde's field arm is deliberately narrow: `lowercase` and
    /// `snake_case` are identities (separator handling belongs to
    /// variant renaming), the case-changing rules convert ASCII only,
    /// and the separator rules are plain `_` replacements — mirroring
    /// those exactly is what keeps the schema from lying about the
    /// wire name.
    pub(crate) fn apply(self, name: &str) -> String {
        match self {
            Self::Lower | Self::Snake => name.to_owned(),
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
            pascal.push(ch.to_ascii_uppercase());
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
            ("lowercase", "FileName", "FileName"),
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
    fn non_ascii_identifiers_follow_serdes_ascii_only_field_arms() {
        let cases: Vec<(&str, &str, &str)> = vec![
            ("lowercase", "straße_x", "straße_x"),
            ("UPPERCASE", "straße_x", "STRAßE_X"),
            ("PascalCase", "straße_x", "StraßeX"),
            ("camelCase", "straße_x", "straßeX"),
            ("SCREAMING_SNAKE_CASE", "straße_x", "STRAßE_X"),
        ];
        for (name, input, expected) in cases {
            let strategy =
                RenameAll::from_str(name).unwrap_or_else(|| panic!("unknown strategy: {name}"));
            assert_eq!(
                strategy.apply(input),
                expected,
                "strategy {name:?} on {input:?} must match serde's ASCII-only field arm"
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
            "FileName",
            "serde's field arm groups lowercase with snake_case as the \
            identity — mirroring that is what keeps the advertised key \
            the one serde actually looks for"
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
    fn serde_rename_all_survives_value_bearing_keys_beside_it() {
        use syn::parse_quote;
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(bound = "T: Clone", crate = "serde", rename_all = "kebab-case")]
        };
        assert_eq!(
            serde_rename_all(&attrs).expect("other keys are tolerated"),
            Some(RenameAll::Kebab),
            "a value-bearing serde key that is not rename_all must not abort the walk"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(rename = "container")]
        };
        assert_eq!(
            serde_rename_all(&attrs).expect("container rename is tolerated"),
            None
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(rename_all = "screaming")]
        };
        assert!(
            serde_rename_all(&attrs).is_err(),
            "an unknown strategy still fails instead of falling back to raw names"
        );
    }

    #[test]
    fn read_side_skip_keys_are_the_only_skip_shaped_rejections() {
        use syn::parse_quote;
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(skip_serializing)]
        };
        assert!(
            check_unmirrorable_serde(&attrs, false, false).is_ok(),
            "skip_serializing changes only what serde emits — advertising the field \
            stays truthful"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(skip)]
        };
        assert!(check_unmirrorable_serde(&attrs, false, false).is_err());
        assert!(
            check_unmirrorable_serde(&attrs, false, true).is_ok(),
            "the tool(skip) combination is the consistent escape"
        );
        assert!(has_serde_skip(&attrs));
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(skip_deserializing)]
        };
        assert!(has_serde_skip(&attrs));
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(skip_serializing)]
        };
        assert!(
            !has_serde_skip(&attrs),
            "the write-side key fills nothing and must not count as fillable"
        );
    }

    #[test]
    fn wire_format_rewriters_and_transparent_are_rejected() {
        use syn::parse_quote;
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(with = "base64_bytes")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, false, false).is_err(),
            "`with` rewrites the wire format the schema describes"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(deserialize_with = "parse_duration")]
        };
        assert!(check_unmirrorable_serde(&attrs, false, false).is_err());
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(serialize_with = "emit_compact")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, false, false).is_ok(),
            "serialize_with is write-side only, like skip_serializing"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(transparent)]
        };
        assert!(
            check_unmirrorable_serde(&attrs, true, false).is_err(),
            "transparent makes the wire the inner value, not the advertised object"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(into = "String")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, true, false).is_ok(),
            "into converts only what serde emits — the read side the schema describes is untouched"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(try_into = "Vec<u8>")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, true, false).is_ok(),
            "try_into is write-side like into"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(from = "Raw")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, true, false).is_err(),
            "from converts the deserialization input itself"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[serde(try_from = "Raw")]
        };
        assert!(
            check_unmirrorable_serde(&attrs, true, false).is_err(),
            "try_from is read-side like from"
        );
    }

    #[test]
    fn rename_all_matches_serdes_serialized_wire_names() {
        use serde::Serialize;

        macro_rules! renamed {
            ($name:ident, $strategy:literal) => {
                #[derive(Serialize)]
                #[serde(rename_all = $strategy)]
                #[allow(non_snake_case)]
                struct $name {
                    fileName: &'static str,
                    straße_x: &'static str,
                }
            };
        }
        renamed!(RenamedLower, "lowercase");
        renamed!(RenamedUpper, "UPPERCASE");
        renamed!(RenamedPascal, "PascalCase");
        renamed!(RenamedCamel, "camelCase");
        renamed!(RenamedSnake, "snake_case");
        renamed!(RenamedScreamingSnake, "SCREAMING_SNAKE_CASE");
        renamed!(RenamedKebab, "kebab-case");
        renamed!(RenamedScreamingKebab, "SCREAMING-KEBAB-CASE");

        let lower = serde_json::to_value(RenamedLower {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let upper = serde_json::to_value(RenamedUpper {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let pascal = serde_json::to_value(RenamedPascal {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let camel = serde_json::to_value(RenamedCamel {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let snake = serde_json::to_value(RenamedSnake {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let screaming_snake = serde_json::to_value(RenamedScreamingSnake {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let kebab = serde_json::to_value(RenamedKebab {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");
        let screaming_kebab = serde_json::to_value(RenamedScreamingKebab {
            fileName: "",
            straße_x: "",
        })
        .expect("serializes");

        let cases: [(&str, serde_json::Value); 8] = [
            ("lowercase", lower),
            ("UPPERCASE", upper),
            ("PascalCase", pascal),
            ("camelCase", camel),
            ("snake_case", snake),
            ("SCREAMING_SNAKE_CASE", screaming_snake),
            ("kebab-case", kebab),
            ("SCREAMING-KEBAB-CASE", screaming_kebab),
        ];
        for (strategy, serialized) in cases {
            let keys = serialized
                .as_object()
                .expect("the renamed struct serializes as an object")
                .keys()
                .cloned()
                .collect::<Vec<String>>();
            let rule = RenameAll::from_str(strategy)
                .unwrap_or_else(|| panic!("unknown strategy: {strategy}"));
            assert_eq!(
                keys,
                vec![rule.apply("fileName"), rule.apply("straße_x")],
                "serde's own serialized keys for {strategy} are the oracle — the \
                derive must advertise exactly them"
            );
        }
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
        let attrs: Vec<Attribute> = parse_quote! {
            #[tool(allow_extra(false))]
        };
        let err = parse_container(&attrs)
            .err()
            .expect("the parenthesized form must not slip past the check and set the flag");
        assert!(
            err.to_string().contains("`allow_extra` takes no value"),
            "the error names the key: {err}"
        );
        let attrs: Vec<Attribute> = parse_quote! {
            #[tool(default(false))]
        };
        let err = parse_field(&attrs)
            .err()
            .expect("the parenthesized form must not slip past the check and set the flag");
        assert!(
            err.to_string().contains("`default` takes no value"),
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
