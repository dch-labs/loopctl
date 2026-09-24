//! Integration pins for the v1 manifest model (L-122).
//!
//! Each test exercises the public pipeline end to end — parse, profile
//! merge, validation, interpolation, schema export — against small in-file
//! YAML fixtures. No filesystem, no environment: the env resolver is an
//! injected map.

#![cfg(feature = "manifest")]
// Integration tests are a separate crate and do not inherit `lib.rs`'s
// `cfg_attr(test, allow(...))`. Apply the same test-code relaxations the lib
// uses: assertions legitimately `unwrap`/`expect`/`panic`/index for clarity.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use loopctl::manifest::{EnvMap, ManifestDocument, ManifestError, manifest_json_schema};

#[test]
fn unknown_field_error_carries_span_and_suggestion() {
    let text = "version: 1\nmodels:\n  entries:\n    primary:\n      provider: anthropic\n      modelx: claude\n";
    let error =
        ManifestDocument::parse(text).expect_err("an unknown key must fail the strict parse");
    let ManifestError::UnknownField {
        path,
        field,
        suggestion,
        line,
        column,
    } = error
    else {
        panic!("the failure must surface as UnknownField, got: {error:?}");
    };
    assert_eq!(field, "modelx", "the offending key is named exactly");
    assert_eq!(
        path, "models.entries.primary",
        "the dotted path points at the holding mapping"
    );
    assert_eq!(
        suggestion.as_deref(),
        Some("model"),
        "the did-you-mean suggestion is the one-edit neighbor"
    );
    assert_eq!(line, 6, "the span is the key's own line");
    assert_eq!(column, 7, "the span is the key's own column");
}

#[test]
fn profiles_merge_scalars_maps_lists_with_documented_semantics() {
    let text = "\
version: 1
budgets:
  turns: 200
  cost_usd: 25.0
metadata:
  name: demo
mcp:
  github:
    command: [npx]
    env:
      GITHUB_TOKEN: x
tools:
  - id: shell
    builtin: shell
profiles:
  cheap:
    budgets:
      turns: 5
      cost_usd: 0.5
    metadata:
      version: 0.2.0
    tools: !append
      - id: peek
        builtin: read
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let resolved = document
        .resolve(Some("cheap"), &EnvMap::default())
        .expect("the profile resolves");
    let manifest = resolved.manifest();
    assert_eq!(
        manifest.budgets.turns,
        Some(5),
        "overlay scalars replace base scalars"
    );
    assert_eq!(
        manifest.budgets.cost_usd,
        Some(0.5),
        "a second scalar in the same overlay map also replaces"
    );
    assert_eq!(
        manifest.metadata.name.as_deref(),
        Some("demo"),
        "map keys the overlay omits survive the deep merge"
    );
    assert_eq!(
        manifest.metadata.version.as_deref(),
        Some("0.2.0"),
        "map keys the overlay names are replaced"
    );
    let tool_ids: Vec<&str> = manifest.tools.iter().map(|tool| tool.id.as_str()).collect();
    assert_eq!(
        tool_ids,
        ["shell", "peek"],
        "lists replace by default and !append concatenates — pinned in one fixture"
    );
    assert_eq!(
        manifest.mcp.len(),
        1,
        "sections the overlay never mentions pass through untouched"
    );
}

#[test]
fn missing_required_env_var_fails_validation_with_variable_name() {
    let text = "\
version: 1
models:
  entries:
    primary:
      provider: anthropic
      model: claude-sonnet-4-6
      base_url: ${MISSING_BASE_URL:?set MISSING_BASE_URL first}
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let error = document
        .resolve(None, &EnvMap::default())
        .expect_err("an unresolved required reference must fail resolution");
    let ManifestError::MissingEnvVar {
        variable,
        path,
        detail,
    } = error
    else {
        panic!("the failure must surface as MissingEnvVar, got: {error:?}");
    };
    assert_eq!(
        variable, "MISSING_BASE_URL",
        "the error names the missing variable"
    );
    assert_eq!(
        path, "models.entries.primary.base_url",
        "the error names where the reference lives"
    );
    assert_eq!(
        detail.as_deref(),
        Some("set MISSING_BASE_URL first"),
        "the author's required-message rides along"
    );
}

#[test]
fn literal_api_key_in_manifest_fails_validation() {
    let text = "\
version: 1
models:
  entries:
    primary:
      provider: openai
      model: gpt-5
      token_key: sk-proj-abcdefghijklmnopqrstuvwxyz
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let error = document
        .resolve(None, &EnvMap::default())
        .expect_err("a pasted key must never validate");
    let ManifestError::LiteralSecret { path, shape } = error else {
        panic!("the failure must surface as LiteralSecret, got: {error:?}");
    };
    assert_eq!(
        path, "models.entries.primary.token_key",
        "the error points at the exact field carrying the key"
    );
    assert_eq!(shape, "sk-", "the matched key family is reported");
}

#[test]
fn unused_profile_references_neither_fail_resolution_nor_move_the_hash() {
    let text = "\
version: 1
budgets:
  turns: 200
profiles:
  cheap:
    budgets:
      turns: 5
  prod:
    budgets:
      duration: ${PROD_DURATION}
    metadata:
      name: ${PROD_NAME:-prod-default}
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let resolved = document
        .resolve(Some("cheap"), &EnvMap::default())
        .expect("an unused profile's references must never fail resolution");
    assert_eq!(
        resolved.manifest().budgets.turns,
        Some(5),
        "the selected profile still applies"
    );
    assert!(
        resolved.manifest().profiles.is_empty(),
        "the resolved manifest is the effective configuration — declarations do not ride along"
    );
    let profile_node = resolved
        .canonical_json()
        .get("profiles")
        .and_then(|profiles| profiles.as_object())
        .expect("the pin carries the (empty) profiles field");
    assert!(
        profile_node.is_empty(),
        "unapplied overlays must never enter the pin"
    );
    let other_env = EnvMap::from_pairs([("PROD_NAME", "beta"), ("PROD_DURATION", "4h")]);
    let re_resolved = document
        .resolve(Some("cheap"), &other_env)
        .expect("the unused profile still does not fail under different env values");
    assert_eq!(
        resolved.config_hash(),
        re_resolved.config_hash(),
        "environment changes that only touch unused profiles must not move the identity pin"
    );
}

#[test]
fn mcp_env_pins_as_written_and_never_leaks_resolved_secrets() {
    let text = "\
version: 1
mcp:
  srv:
    command: [npx]
    env:
      TOKEN: ${secret:GITHUB_TOKEN}
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let first = document
        .resolve(
            None,
            &EnvMap::from_pairs([("GITHUB_TOKEN", "gh-supersecret-value-1")]),
        )
        .expect("a set secret resolves");
    let second = document
        .resolve(
            None,
            &EnvMap::from_pairs([("GITHUB_TOKEN", "gh-entirely-different-2")]),
        )
        .expect("the other secret resolves too");
    assert_eq!(
        first
            .canonical_json()
            .pointer("/mcp/srv/env/TOKEN")
            .and_then(serde_json::Value::as_str),
        Some("${secret:GITHUB_TOKEN}"),
        "the pin records the reference form, never the resolved secret"
    );
    assert_eq!(
        first.config_hash(),
        second.config_hash(),
        "the identity pin must not be a function of a secret value"
    );
    assert_eq!(
        first
            .manifest()
            .mcp
            .get("srv")
            .and_then(|server| server.env.get("TOKEN"))
            .map(String::as_str),
        Some("gh-supersecret-value-1"),
        "the runtime manifest still carries the expanded value for process wiring"
    );
}

#[test]
fn non_finite_budget_values_fail_validation() {
    let text = "version: 1\nbudgets:\n  cost_usd: .nan\n";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let error = document
        .resolve(None, &EnvMap::default())
        .expect_err("a declared budget must never silently become no budget");
    let rendered = error.to_string();
    assert!(
        rendered.contains("cost_usd") && rendered.contains("finite"),
        "the error names the field and the problem, got: {rendered}"
    );
}

#[test]
fn secret_references_outside_mcp_env_pin_as_written() {
    let text = "\
version: 1
memory:
  store: postgres://user:${secret:DB_PASSWORD}@host/db
";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let first = document
        .resolve(None, &EnvMap::from_pairs([("DB_PASSWORD", "hunter2")]))
        .expect("a set secret resolves");
    let second = document
        .resolve(
            None,
            &EnvMap::from_pairs([("DB_PASSWORD", "correct-horse-battery")]),
        )
        .expect("the other secret resolves too");
    assert_eq!(
        first
            .canonical_json()
            .pointer("/memory/store")
            .and_then(serde_json::Value::as_str),
        Some("postgres://user:${secret:DB_PASSWORD}@host/db"),
        "a secret reference anywhere — not just MCP env — pins as written"
    );
    assert_eq!(
        first.config_hash(),
        second.config_hash(),
        "the identity pin must never be a function of a secret value, in any field"
    );
    assert_eq!(
        first.manifest().memory.store.as_deref(),
        Some("postgres://user:hunter2@host/db"),
        "the runtime manifest still carries the expanded value for wiring"
    );
}

#[test]
fn non_finite_compaction_triggers_fail_validation() {
    let text = "version: 1\ncontext:\n  compaction:\n    trigger: .nan\n";
    let document = ManifestDocument::parse(text).expect("the fixture parses");
    let error = document
        .resolve(None, &EnvMap::default())
        .expect_err("a declared trigger must never silently become no trigger");
    let rendered = error.to_string();
    assert!(
        rendered.contains("context.compaction.trigger") && rendered.contains("finite"),
        "the error names the field and the problem, got: {rendered}"
    );
}

#[test]
fn schema_export_round_trips_every_type() {
    let schema = manifest_json_schema().expect("the schema export never fails");
    let reparsed: serde_json::Value =
        serde_json::from_value(schema).expect("the exported schema is valid canonical JSON");
    let definitions = reparsed
        .get("$defs")
        .and_then(|defs| defs.as_object())
        .expect("every named type lands in $defs");
    let expected = [
        "Metadata",
        "AgentSection",
        "ModelsSection",
        "ModelEntry",
        "ToolEntry",
        "McpServer",
        "Permissions",
        "PermissionRules",
        "PermissionMode",
        "Budgets",
        "ContextSection",
        "CompactionSettings",
        "MemorySection",
        "ScheduleSection",
        "Overlap",
        "MissedFire",
        "TriggersSection",
        "CassettesSection",
        "RecordMode",
        "ReplayMode",
        "SandboxSection",
        "Profile",
    ];
    for name in expected {
        assert!(
            definitions.contains_key(name),
            "the schema must export the `{name}` type — a missing $defs entry means the round trip dropped it"
        );
    }
    assert!(
        reparsed.get("title").and_then(|t| t.as_str()) == Some("Manifest"),
        "the root type itself is described by the root schema, not a $defs entry"
    );
    let root = reparsed
        .get("properties")
        .and_then(|props| props.as_object())
        .expect("the root schema lists its section properties");
    for section in [
        "version",
        "metadata",
        "agent",
        "models",
        "tools",
        "mcp",
        "permissions",
        "budgets",
        "context",
        "memory",
        "schedule",
        "triggers",
        "cassettes",
        "sandbox",
        "profiles",
    ] {
        assert!(
            root.contains_key(section),
            "the root schema must advertise the `{section}` section"
        );
    }
    assert_eq!(
        reparsed.get("$schema").and_then(|s| s.as_str()),
        Some("https://json-schema.org/draft/2020-12/schema"),
        "the schema declares its draft so validators know which rules apply"
    );
    for (name, definition) in definitions {
        let is_object_schema = definition.get("properties").is_some();
        if !is_object_schema {
            continue;
        }
        let strict = definition
            .get("additionalProperties")
            .and_then(serde_json::Value::as_bool);
        assert_eq!(
            strict,
            Some(false),
            "the `{name}` object type must be strict (`additionalProperties: false`) — \
             a lax entry would let the schema drift from the deny-unknown-fields parser \
             (closed string enums are exempt: their `enum` list is the strictness)"
        );
    }
    let strict_root = reparsed
        .get("additionalProperties")
        .and_then(serde_json::Value::as_bool);
    assert_eq!(
        strict_root,
        Some(false),
        "the root Manifest schema must be strict like every section type"
    );
    let document =
        ManifestDocument::parse("version: 1\nmetadata:\n  name: demo\nbudgets:\n  turns: 5\n")
            .expect("the round-trip fixture parses");
    let serialized = serde_json::to_value(document.manifest()).expect("the manifest serializes");
    let round_tripped: loopctl::manifest::Manifest =
        serde_json::from_value(serialized).expect("the manifest re-parses from its own value");
    assert_eq!(
        round_tripped,
        *document.manifest(),
        "a manifest survives to_value → from_value unchanged — the export and the parser agree"
    );
}
