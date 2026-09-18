# loopctl-derive

`#[derive(Tool)]` generates the `impl loopctl::tool::Tool` block for a
`Deserialize` input struct: the name, the description, a statically built
JSON Schema, and a `call` that deserializes the incoming value and
dispatches to an inherent `async fn run` handler. Re-exported from the
`loopctl` crate behind its `derive` feature — the usual spelling is
`use loopctl::Tool;`.

## Supported / rejected / warned

Every rejected row is pinned by a [trybuild](tests/ui.rs) case with a
golden diagnostic (`tests/ui/<case>.rs` + `.stderr`); a changed
diagnostic fails CI until the golden is deliberately updated. The
accepted rows are pass-pinned — they must compile, with no golden.
Nothing is warned today: every mode the derive cannot mirror exactly is
a hard error, so there is no "warned" table.

### Rejected at derive time

| Case | Golden | Diagnostic in one line |
|---|---|---|
| No doc comment and no `#[tool(description)]` | `missing_description` | names both ways to fix it |
| A doc comment with no text at all | `empty_doc_comment` | counts as absent — the `missing_description` error fires |
| Empty `#[tool(name = "")]` / `#[tool(description = "")]` | `empty_name`, `empty_description` | fails at the attribute's span |
| Unknown `#[tool(...)]` key | `unknown_attr` | lists the accepted keys |
| `#[tool(skip)]` on a field serde cannot fill | `bad_skip` | names the fillability precondition |
| `#[tool(default)]` without serde fill | `default_without_serde` | schema-optional vs serde-required mismatch |
| `#[tool(name)]` disagreeing with serde's key | `name_disagreement` | quotes both names and the fix |
| Handler attribute not a valid identifier | `handler_not_ident`, `bad_handler` | names the offending value |
| Non-string `#[tool(name = 123)]` | `non_string_value` | `expected string literal` |
| Value on a flag key (`skip = true`) | `flag_with_value` | `` `skip` takes no value `` |
| Handler resolving to a non-`async fn run` shape | `handler_wrong_shape` | rustc's not-a-future error at the generated dispatch |
| Enum input | `enum_input` | structs only |
| Tuple / unit struct | `tuple_struct`, `unit_struct` | named fields only |
| Generic struct (type or lifetime parameters) | `generic_struct`, `lifetime_struct` | implement manually |
| Unmappable field type | `unsupported_type`, `cow_bytes` | implement manually |
| `#[serde(rename_all)]` strategy serde itself rejects | `unknown_rename_all` | serde's spelling, no silent fallback |
| Container `#[serde(from/try_from/remote/transparent/rename_all_fields)]` | `serde_container_from`, `serde_rename_all_fields`, `serde_transparent` | wire rewrites the schema cannot mirror |
| Field `#[serde(flatten)]` | `serde_flatten` | same class |
| Field `#[serde(with/deserialize_with)]` | `serde_deserialize_with` | same class |
| Read-side `#[serde(skip)]` without `#[tool(skip)]` | `serde_skip_without_tool_skip` | schema would advertise a field serde ignores |
| `#[tool(allow_extra)]` + `#[serde(deny_unknown_fields)]` | `allow_extra_conflict` | the schema would invite payloads serde refuses |

### Accepted (pinned as passing)

| Case | Golden | Why it is fine |
|---|---|---|
| Private fields in a `pub` struct | `private_field_pass` | the expansion names no fields — serde does the construction |
| `#[serde(deny_unknown_fields)]` alone | `deny_unknown_alone_pass` | agrees with the schema's `additionalProperties: false` |
| The full happy-path surface | `pass` | names, defaults, schema, dispatch |

## Consumer hygiene

`tests/consumer/` is a standalone downstream crate that consumes the
derive exactly as a user would — its own manifest, the `loopctl`
dependency **renamed** (`lc = { package = "loopctl" }`), bare `use`
paths — and builds under `RUSTFLAGS="-D warnings"`. The expansion holds
no coupling to the crate's own name: generated paths resolve through the
name the consumer's manifest binds, so a dependency rename compiles and
an expansion that would trip downstream lints fails the build. Run it
with `make derive-consumer` (part of `make ci`).
