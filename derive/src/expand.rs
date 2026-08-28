//! The codegen core of `#[derive(Tool)]`.
//!
//! Walks the input struct's fields, reads the container and field
//! attributes (via [`crate::attr`]), and emits the `impl Tool` block:
//! `name`, `description`, a statically built JSON Schema from the
//! field types, and a `call` that deserializes and dispatches to the
//! handler. Errors surface as spanned `syn::Error`s — never panics.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Type};

use crate::attr::{self, ContainerAttrs};

/// Expand a `#[derive(Tool)]` input into an `impl Tool` block.
///
/// # Errors
///
/// Returns a spanned error for non-struct inputs, missing
/// descriptions, unmappable field types, and invalid attribute use;
/// errors flatten to `compile_error!` tokens at the entry point.
pub(crate) fn expand_derive_tool(input: DeriveInput) -> TokenStream2 {
    let result = expand(&input);
    drop(input);
    match result {
        Ok(tokens) => tokens,
        Err(error) => error.into_compile_error(),
    }
}

/// The codegen core.
///
/// # Errors
///
/// Returns a spanned error for non-struct inputs, missing descriptions,
/// unmappable field types, and invalid attribute use.
fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let ident = &input.ident;
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new(
                    ident.span(),
                    "`Tool` can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                ident.span(),
                "`Tool` can only be derived on structs",
            ));
        }
    };

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            ident.span(),
            "`Tool` cannot be derived for generic structs; implement it manually",
        ));
    }
    let container = attr::parse_container(&input.attrs)?;
    let description = container
        .description
        .clone()
        .or_else(|| attr::doc_string(&input.attrs))
        .ok_or_else(|| {
            syn::Error::new(
                ident.span(),
                format!(
                    "`{ident}` has no description: set \
                     `#[tool(description = \"…\")]` or add a `///` doc comment"
                ),
            )
        })?;
    let tool_name = container
        .name
        .clone()
        .unwrap_or_else(|| to_snake_case(&ident.to_string()));

    let rename_all = attr::serde_rename_all(&input.attrs);
    let mut properties = Vec::<TokenStream2>::new();
    let mut required = Vec::<String>::new();
    for field in fields {
        let Some(field_ident) = &field.ident else {
            continue;
        };
        let field_attrs = attr::parse_field(&field.attrs)?;
        if field_attrs.skip {
            check_skip_validity(field_ident, &field.ty, &field.attrs)?;
            continue;
        }
        let rust_name = field_ident.to_string();
        let json_name = field_attrs
            .name
            .clone()
            .or_else(|| attr::serde_rename(&field.attrs))
            .unwrap_or_else(|| match rename_all {
                Some(strategy) => strategy.apply(&rust_name),
                None => rust_name.clone(),
            });
        let field_doc = field_attrs
            .description
            .clone()
            .or_else(|| attr::doc_string(&field.attrs));
        let (schema_value, optional) = type_schema(&field.ty)?;
        let description_part = match field_doc {
            Some(doc) => quote! { , "description": #doc },
            None => quote! {},
        };
        properties.push(quote! {
            #json_name: { #schema_value #description_part }
        });
        if !optional && !field_attrs.default && !attr::has_serde_default(&field.attrs) {
            required.push(json_name);
        }
    }

    let additional = if container.allow_extra {
        quote! {}
    } else {
        quote! { "additionalProperties": false, }
    };
    let required_tokens: Vec<&str> = required.iter().map(String::as_str).collect();

    let handler = format_ident!("{}", container.handler.as_deref().unwrap_or("run"));
    let overrides = provided_overrides(&container);

    Ok(quote! {
        #[automatically_derived]
        impl loopctl::tool::Tool for #ident {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn schema(&self) -> loopctl::tool::ToolSchema {
                loopctl::tool::ToolSchema {
                    tool: #tool_name.to_string(),
                    description: #description.to_string(),
                    input_schema: loopctl::__private::serde_json::json!({
                        "type": "object",
                        #additional
                        "properties": { #(#properties),* },
                        "required": [#(#required_tokens),*]
                    }),
                }
            }

            fn call(
                &self,
                input: loopctl::__private::serde_json::Value,
                ctx: &loopctl::tool::ToolContext,
            ) -> std::pin::Pin<
                std::boxed::Box<
                    dyn std::future::Future<
                        Output = Result<
                            loopctl::tool::ToolOutput,
                            loopctl::tool::ToolError,
                        >,
                    > + std::marker::Send
                    + '_,
                >,
            > {
                // Own the context: the boxed future is bounded by
                // `self`'s lifetime alone, so it cannot borrow `ctx`.
                let ctx = std::clone::Clone::clone(ctx);
                std::boxed::Box::pin(async move {
                    let parsed: Self = match loopctl::__private::serde_json::from_value(input) {
                        Ok(value) => value,
                        Err(err) => {
                            return Err(
                                loopctl::tool::ToolError::InvalidInput(
                                    err.to_string(),
                                )
                            );
                        }
                    };
                    Self::#handler(self, parsed, &ctx).await
                })
            }

            #overrides
        }

    })
}

/// The provided-method overrides, emitted only for present attributes.
///
/// A method is generated exactly when its container attribute is set;
/// absent attributes leave the trait's defaults in place — the user
/// can still hand-write an override because the macro never emits a
/// method they did not ask for.
fn provided_overrides(container: &ContainerAttrs) -> TokenStream2 {
    let mut methods = Vec::<TokenStream2>::new();
    if container.read_only {
        methods.push(quote! {
            fn is_read_only(&self) -> bool { true }
        });
    }
    if container.concurrency_safe {
        methods.push(quote! {
            fn is_concurrency_safe(&self) -> bool { true }
        });
    }
    if let Some(prompt) = &container.system_prompt {
        methods.push(quote! {
            fn system_prompt(&self) -> Option<String> {
                Some(#prompt.to_string())
            }
        });
    }
    quote! { #(#methods)* }
}

/// `#[tool(skip)]` requires the field to deserialize without LLM input.
/// # Errors
///
/// Enforce the deserialization precondition of `#[tool(skip)]`.
///
/// A skipped field never reaches the model, so serde must be able to
/// produce it without input — `Option<T>` or `#[serde(default)]`.
///
/// # Errors
///
/// Returns a spanned error naming the precondition when neither holds.
fn check_skip_validity(
    field_ident: &Ident,
    ty: &Type,
    attrs: &[syn::Attribute],
) -> syn::Result<()> {
    if is_option(ty).is_some() || attr::has_serde_default(attrs) {
        return Ok(());
    }
    Err(syn::Error::new(
        field_ident.span(),
        "`#[tool(skip)]` requires the field to be `Option<T>` or carry \
         `#[serde(default)]` so deserialization succeeds when the \
         property is absent",
    ))
}

/// Map a Rust type to its JSON Schema value tokens and an `optional`
/// flag (`true` for `Option<T>`).
/// # Errors
///
/// Returns a spanned error describing the unmappable or invalid input.
fn type_schema(ty: &Type) -> syn::Result<(TokenStream2, bool)> {
    if let Some(inner) = is_option(ty) {
        let (schema, _) = type_schema(inner)?;
        return Ok((schema, true));
    }
    let last = last_path_segment(ty);
    let Some(seg) = last else {
        return Err(unmappable(ty));
    };
    let ident = seg.ident.to_string();
    match ident.as_str() {
        "String" | "str" => Ok((quote! { "type": "string" }, false)),
        "Cow" => {
            let inner = generic_arg(seg)?;
            let Some(inner_seg) = last_path_segment(inner) else {
                return Err(unmappable(ty));
            };
            if matches!(inner_seg.ident.to_string().as_str(), "String" | "str") {
                Ok((quote! { "type": "string" }, false))
            } else {
                Err(unmappable(ty))
            }
        }
        "bool" => Ok((quote! { "type": "boolean" }, false)),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
        | "usize" => Ok((quote! { "type": "integer" }, false)),
        "f32" | "f64" => Ok((quote! { "type": "number" }, false)),
        "Vec" => {
            let inner = generic_arg(seg)?;
            let (schema, _) = type_schema(inner)?;
            Ok((quote! { "type": "array", "items": { #schema } }, false))
        }
        "HashMap" | "BTreeMap" => {
            let inner = second_generic_arg(seg)?;
            let (schema, _) = type_schema(inner)?;
            Ok((
                quote! { "type": "object", "additionalProperties": { #schema } },
                false,
            ))
        }
        _ => Err(unmappable(ty)),
    }
}

/// The error for a field type the type map cannot express.
///
/// Names the offending type and offers the two escapes: a manual
/// `impl Tool`, or `#[tool(skip)]` to keep the derive for the rest
/// of the struct.
fn unmappable(ty: &Type) -> syn::Error {
    use quote::ToTokens;
    let text = ty.to_token_stream().to_string();
    syn::Error::new(
        proc_macro2::Span::call_site(),
        format!(
            "the derive cannot map `{text}` to a JSON Schema type; \
             implement `Tool` manually or use `#[tool(skip)]`"
        ),
    )
}

/// The last segment of a (possibly referenced) path type.
///
/// Peels `&` references so `&str` maps like `str`; returns `None`
/// for non-path types (the caller turns that into an unmappable
/// error).
fn last_path_segment(ty: &Type) -> Option<&syn::PathSegment> {
    match ty {
        Type::Path(path) => path.path.segments.last(),
        Type::Reference(reference) => last_path_segment(&reference.elem),
        _ => None,
    }
}

/// The `T` inside an `Option<T>`, if `ty` is one.
///
/// The caller unwraps the schema (an optional field advertises its
/// inner type) and removes the field from `required`.
fn is_option(ty: &Type) -> Option<&Type> {
    let segment = last_path_segment(ty)?;
    if segment.ident != "Option" {
        return None;
    }
    match &segment.arguments {
        syn::PathArguments::AngleBracketed(args) => args.args.first().and_then(|arg| {
            if let syn::GenericArgument::Type(inner) = arg {
                Some(inner)
            } else {
                None
            }
        }),
        _ => None,
    }
}

/// # Errors
///
/// The first type argument of a path segment (`Vec<T>` → `T`).
///
/// # Errors
///
/// Returns a spanned error when the argument list is absent or its
/// first entry is not a type.
fn generic_arg(segment: &syn::PathSegment) -> syn::Result<&Type> {
    match &segment.arguments {
        syn::PathArguments::AngleBracketed(args) => {
            let inner = args.args.iter().find_map(|arg| {
                if let syn::GenericArgument::Type(ty) = arg {
                    Some(ty)
                } else {
                    None
                }
            });
            inner.ok_or_else(|| unmappable_of(segment))
        }
        _ => Err(unmappable_of(segment)),
    }
}

/// # Errors
///
/// The second type argument of a path segment (`HashMap<K, V>` → `V`).
///
/// The key parameter is assumed string-keyed per the type table; only
/// the value parameter feeds the schema.
///
/// # Errors
///
/// Returns a spanned error when the argument list is too short or the
/// second entry is not a type.
fn second_generic_arg(segment: &syn::PathSegment) -> syn::Result<&Type> {
    match &segment.arguments {
        syn::PathArguments::AngleBracketed(args) => {
            let inner = args
                .args
                .iter()
                .filter_map(|arg| {
                    if let syn::GenericArgument::Type(ty) = arg {
                        Some(ty)
                    } else {
                        None
                    }
                })
                .nth(1);
            inner.ok_or_else(|| unmappable_of(segment))
        }
        _ => Err(unmappable_of(segment)),
    }
}

/// The error for a generic type whose arguments are missing or not
/// types (e.g. a bare `Vec` without a parameter).
fn unmappable_of(segment: &syn::PathSegment) -> syn::Error {
    syn::Error::new(
        segment.ident.span(),
        "the derive cannot map this type's generic arguments",
    )
}

/// Convert a type identifier to its default tool name.
///
/// `EchoInput` becomes `echo_input` — each uppercase boundary starts
/// a new underscore-separated word. Explicit `#[tool(name = "…")]`
/// sidesteps this entirely.
fn to_snake_case(name: &str) -> String {
    let mut out = String::new();
    for (index, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if index != 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}
