//! Procedural-macro companion to `loopctl`'s [`Tool`] trait.
//!
//! [`Tool`] here is a derive macro: apply it to a
//! `Deserialize` struct describing the tool's input and it generates the
//! `impl loopctl::tool::Tool` block — the name, description, JSON-Schema
//! built statically from the struct's fields, and a `call` that
//! deserializes the incoming `Value` into the struct and dispatches to
//! an inherent `async fn run` handler you supply.
//!
//! # Attribute Grammar
//!
//! On the struct (container attributes):
//!
//! ```text
//! #[tool(
//!     name = "snake_case_name",       // default: snake_cased struct ident
//!     description = "one-liner",       // default: struct doc comment
//!     read_only,                       // override is_read_only() -> true
//!     concurrency_safe,                // override is_concurrency_safe() -> true
//!     system_prompt = "extra hint",    // override system_prompt()
//!     handler = "my_run",              // default: "run"
//!     allow_extra                      // omit additionalProperties: false
//! )]
//! ```
//!
//! On each field:
//!
//! ```text
//! #[tool(
//!     name = "json_key",               // default: Rust field name (or serde's rename)
//!     description = "field help",      // default: field doc comment
//!     skip,                            // exclude from schema + required
//!     default                          // omit from required, keep in properties
//! )]
//! ```
//!
//! `#[serde(rename = "…")]` and `#[serde(rename_all = "…")]` are
//! honored for schema naming, mirroring what serde deserializes with;
//! `#[tool(name = "…")]` takes precedence over both.
//!
//! Re-exported from the `loopctl` crate behind its `derive` feature, so
//! the usual spelling is:
//!
//! ```rust,ignore
//! use loopctl::Tool; // brings both the trait and the derive into scope
//! ```
//!
//! See the `Tool` trait docs in `loopctl` for the attribute grammar.
//!
//! [`Tool`]: https://docs.rs/loopctl/latest/loopctl/tool/trait.Tool.html

mod attr;
mod expand;

use proc_macro::TokenStream;

/// Derive `loopctl::tool::Tool` for a struct.
///
/// Generates all four required methods from the struct's shape: the
/// name (container override or the `snake_cased` identifier), the
/// description (container override or the struct's doc comment — one
/// of the two is required), the JSON Schema built statically from the
/// fields, and a `call` that deserializes the incoming `Value` and
/// dispatches to a handler. The struct must implement
/// `serde::Deserialize` (add `#[derive(serde::Deserialize)]`
/// alongside) and provide an inherent
/// `async fn run(&self, input: Self, ctx: &ToolContext)` returning
/// `Result<ToolOutput, ToolError>`; rename the handler with
/// `#[tool(handler = "…")]`. Tools with dynamic schemas or
/// unmappable field types fall back to a manual `impl`.
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    expand::expand_derive_tool(input).into()
}
