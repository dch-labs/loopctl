//! Pins for the hardened public surface: every `#[non_exhaustive]`
//! type keeps an externally-legitimate construction path, the marker
//! changes nothing on the wire, and the derive's generated schema
//! construction rides the constructor.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone,
    clippy::wildcard_in_or_patterns
)]

use loopctl::compact::CompactionOutcome;
use loopctl::engine::Turn;
use loopctl::memory::{MemoryCategory, MemoryEntry};
use loopctl::message::MessagePart;
use loopctl::stream::Usage;
use loopctl::tool::ToolSchema;

#[test]
fn construction_still_possible_via_constructors() {
    let usage = Usage::new(11, 7);
    assert_eq!(
        (usage.input_tokens, usage.output_tokens),
        (11, 7),
        "Usage constructs through its constructor"
    );

    let schema = ToolSchema::new("t", "d", serde_json::json!({"type": "object"}));
    assert_eq!(schema.tool, "t");
    assert_eq!(schema.description, "d");
    assert_eq!(schema.input_schema["type"], "object");

    let failed = CompactionOutcome::failed(Vec::new(), 4, "boom");
    assert!(!failed.success, "the failure outcome reports failure");
    assert_eq!(failed.tokens_saved, 0, "a failed pass saves nothing");
    let kept = CompactionOutcome::compacted(Vec::new(), 9, 5);
    assert_eq!(kept.tokens_saved, 4, "savings derive from before/after");
    assert!(
        CompactionOutcome::no_change(Vec::new()).success,
        "no-change is a successful outcome"
    );

    let entry = MemoryEntry::new(MemoryCategory::Insight, "fact").with_tag("tagged");
    assert_eq!(entry.memory, "fact");
    assert_eq!(entry.tags, vec!["tagged".to_string()]);

    let restored = MemoryEntry::new(MemoryCategory::Insight, "fact".to_string())
        .with_id(entry.id)
        .with_tags(vec!["tagged".to_string()])
        .with_created_at(entry.created_at)
        .with_relevance(entry.relevance)
        .with_access_count(entry.access_count)
        .with_last_accessed(entry.last_accessed)
        .with_last_decayed(entry.last_decayed)
        .validated();
    assert_eq!(
        restored.id, entry.id,
        "the restore builders preserve identity"
    );
    assert!(restored.validated);

    let run_end = loopctl::observer::RunEndContext::new(true, None, 2, 250);
    assert_eq!(run_end.total_turns, 2);

    assert!(
        matches!(MessagePart::text("body"), MessagePart::Text { .. }),
        "the single-field variants construct through their constructors"
    );

    let built = loopctl::message::Message::new(
        loopctl::message::Role::Assistant,
        vec![
            MessagePart::text("thinking"),
            MessagePart::tool_call("c1", "echo", serde_json::json!({"x": 1})),
        ],
    );
    assert_eq!(built.parts.len(), 2, "Message::new carries its parts");
    assert_eq!(
        loopctl::message::Message::user("hi").parts.len(),
        1,
        "the role helper constructs too"
    );
    let result = MessagePart::tool_result(
        "c1",
        "echo",
        loopctl::message::ToolContent::from_string("out"),
        false,
    );
    assert!(
        matches!(result, MessagePart::ToolResult { .. }),
        "the multi-field variant constructs through its constructor"
    );
}

#[test]
fn provider_forensics_surface_builds_externally() {
    // The wire-failure forensics types are `#[non_exhaustive]` —
    // the external-legitimate construction paths are the constructors.
    use loopctl::api::error::{ApiError, ProviderMeta, RateLimitSnapshot};

    let err = ApiError::rate_limited("slow down", Some(std::time::Duration::from_secs(2)));
    // `RateLimit` is a marked variant: patterns bind through `..`, and
    // meta defaults to None through the constructor.
    match &err {
        ApiError::RateLimit {
            retry_after,
            message,
            ..
        } => {
            assert_eq!(*retry_after, Some(std::time::Duration::from_secs(2)));
            assert_eq!(message, "slow down");
        }
        other => panic!("constructed RateLimit, got {other:?}"),
    }
    assert!(err.provider_meta().is_empty(), "no wire, no forensics");

    let snapshot = RateLimitSnapshot::new(
        Some("500".to_string()),
        Some("42".to_string()),
        None::<String>,
    );
    assert_eq!(snapshot.remaining.as_deref(), Some("42"));
    let meta = ProviderMeta::with_rate_limit(snapshot.clone());
    assert_eq!(meta.rate_limit.as_ref(), Some(&snapshot));
    let ided = ProviderMeta::with_request_id("req_ext_1");
    assert_eq!(ided.request_id.as_deref(), Some("req_ext_1"));

    // Attaching an external meta to a wire failure is the documented
    // bridge for replay tooling.
    let attached = ApiError::http("HTTP 500: boom")
        .with_provider_meta(ProviderMeta::with_request_id("req_ext_2"));
    assert!(
        attached.to_string().ends_with(" [req req_ext_2]"),
        "the suffix carries an externally attached id: {attached}"
    );
    // Tuple wire variants bind through wildcard fields.
    assert!(matches!(
        attached,
        ApiError::Http(_, _) | ApiError::Api(_, _) | ApiError::Auth(_, _)
    ));
}

#[test]
fn old_serialized_data_deserializes() {
    let usage: Usage =
        serde_json::from_str(r#"{"input_tokens":3,"output_tokens":5}"#).expect("usage json");
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 5));

    // Pre-hardening shape: neither defaulted field present.
    let turn: Turn = serde_json::from_str(
        r#"{"turn":1,"input":"q","output":"a","tool_calls":[],"input_tokens":10,"output_tokens":20}"#,
    )
    .expect("turn json");
    assert_eq!(turn.input, "q");
    assert!(matches!(
        turn.stop_reason,
        loopctl::stream::StreamStopReason::EndTurn
    ));
    assert!(!turn.transport_fallback);

    // ToolCall rides inside a turn; the frozen literal pins its field
    // names against renames.
    let turn_with_call: Turn = serde_json::from_str(
        r#"{"turn":0,"input":"q","output":"a","tool_calls":[{"id":"c1","tool":"echo","input":{"a":1}}],"input_tokens":1,"output_tokens":2}"#,
    )
    .expect("turn-with-call json");
    let call = &turn_with_call.tool_calls[0];
    assert_eq!(call.id, "c1");
    assert_eq!(call.tool, "echo");
    assert_eq!(call.input["a"], 1);

    let schema: ToolSchema =
        serde_json::from_str(r#"{"tool":"t","description":"d","input_schema":{"type":"object"}}"#)
            .expect("tool schema json");
    assert_eq!(schema.tool, "t");

    let message: loopctl::message::Message =
        serde_json::from_str(r#"{"role":"user","parts":[{"type":"text","text":"hi"}]}"#)
            .expect("message json");
    assert!(
        matches!(&message.parts[0], MessagePart::Text { text } if text == "hi"),
        "the tagged part shape is pinned"
    );

    let run: loopctl::engine::Run = serde_json::from_str(
        r#"{"id":"7edc6088-1dbb-4534-afc4-06b1cece86d8","turns":[],"input":"q","output":null,"config":{"max_turns":200,"parallel_tool_dispatch":{"mode":"sequential","max_concurrency":8},"reset_managers":false,"memory_top_k":3,"memory_include_provider_derived":false}}"#,
    )
    .expect("run json");
    assert_eq!(run.input, "q");
    assert_eq!(run.id.to_string(), "7edc6088-1dbb-4534-afc4-06b1cece86d8");

    let session: loopctl::engine::Session = serde_json::from_str(
        r#"{"id":"41669eb0-1c8a-4eed-8b9d-2040fcabac34","config":{"system_prompt":null,"context_window":200000,"compact_threshold":80,"auto_compact":true},"runs":[]}"#,
    )
    .expect("session json");
    assert!(session.runs.is_empty());
    assert_eq!(session.config.context_window, 200_000);

    let reason: loopctl::compact::CompactReason =
        serde_json::from_str("\"ThresholdExceeded\"").expect("compact reason json");
    assert_eq!(reason, loopctl::compact::CompactReason::ThresholdExceeded);

    // Hand-written, not a same-version round-trip: the frozen literal
    // fails on a renamed field or a dropped default.
    let entry: MemoryEntry = serde_json::from_str(
        r#"{"id":"29435f89-7213-4eaf-8fc1-ff9ca8697da6","category":"insight","memory":"fact","tags":["kept"],"created_at":{"secs_since_epoch":0,"nanos_since_epoch":0},"relevance":0.5,"access_count":2,"validated":true,"last_accessed":null,"last_decayed":null}"#,
    )
    .expect("entry json");
    assert_eq!(entry.access_count, 2);
    assert!(entry.validated);
    assert_eq!(entry.tags, vec!["kept".to_string()]);
}

/// Gated on `derive` because the macro is an opt-in feature — the
/// full-feature CI run is where this pin executes; a default-feature
/// run skips it by design, not by rot.
#[cfg(feature = "derive")]
#[test]
fn derive_tool_still_compiles() {
    use loopctl::Tool;

    #[derive(Tool, serde::Deserialize)]
    #[tool(name = "derive_pinned", description = "Pins the generated schema")]
    struct DerivedInput {
        value: String,
    }

    impl DerivedInput {
        async fn run(
            &self,
            _input: DerivedInput,
            _ctx: &loopctl::tool::ToolContext,
        ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
            Ok(loopctl::tool::ToolOutput::text("ok"))
        }
    }

    let tool = DerivedInput {
        value: "v".to_string(),
    };
    let schema = tool.schema();
    assert_eq!(schema.tool, "derive_pinned");
    assert_eq!(schema.description, "Pins the generated schema");
    assert_eq!(schema.input_schema["type"], "object");
}

#[test]
fn old_turn_json_deserializes_with_zero_detail() {
    let turn: Turn = serde_json::from_str(
        r#"{"turn":2,"input":"q","output":"a","tool_calls":[],"input_tokens":10,"output_tokens":20}"#,
    )
    .expect("turn json");
    assert_eq!((turn.input_tokens, turn.output_tokens), (10, 20));
    assert_eq!(
        (
            turn.cached_input_tokens,
            turn.cache_write_tokens,
            turn.reasoning_tokens
        ),
        (0, 0, 0),
        "a pre-detail turn record deserializes with unreported usage detail"
    );
}

#[test]
fn probe_types_build_through_their_constructors() {
    use loopctl::api::error::ApiError;
    use loopctl::api::probe::{
        MetadataSource, Modality, ModelCapabilities, ModelInfo, ModelMetadata, PingReport,
        ProbeClass, curated_model_metadata,
    };

    let info = ModelInfo::new("gpt-4o")
        .with_owned_by("openai")
        .with_created("1721172744")
        .with_size_bytes(1)
        .with_metadata(ModelMetadata::disclosed_context_window(128_000));
    assert_eq!(info.id, "gpt-4o");
    assert_eq!(info.created.as_deref(), Some("1721172744"));

    let caps = ModelCapabilities::unreported()
        .with_tool_calling(true)
        .with_structured_output(false);
    assert_eq!(caps.tool_calling, Some(true));
    assert_eq!(caps.structured_output, Some(false));

    let curated = curated_model_metadata("gpt-4o").expect("the gpt-4o family is curated");
    let merged = ModelMetadata::merged(
        Some(&ModelMetadata::disclosed_context_window(9)),
        Some(&curated),
    );
    assert_eq!(merged.context_window, Some(9), "disclosure wins");
    assert_eq!(
        merged.capabilities.tool_calling, curated.capabilities.tool_calling,
        "curation fills the gap"
    );
    assert_eq!(merged.source, MetadataSource::DisclosedAndCurated);

    assert_eq!(Modality::from_wire_token("AUDIO"), Some(Modality::Audio));
    assert_eq!(
        ProbeClass::from_error(&ApiError::auth("x")),
        ProbeClass::Auth
    );
    let report = PingReport::new(std::time::Duration::from_millis(3));
    assert_eq!(report.request_id, None);
    assert_eq!(report.rate_limit, None);
}

#[test]
fn probe_enums_carry_their_growth_marker() {
    // Compile pin: every new probe enum is `#[non_exhaustive]`, so an
    // outside matcher needs a wildcard arm. Enumerating every variant
    // *and* a wildcard compiles quietly only while the marker holds:
    // drop it, the wildcard arm becomes unreachable, and
    // `unreachable_patterns` fails the build under CI's `-D warnings`.
    use loopctl::api::probe::{MetadataSource, Modality, ProbeClass};

    let modality = match Modality::Text {
        Modality::Text
        | Modality::Image
        | Modality::Audio
        | Modality::Video
        | Modality::Embedding
        | _ => true,
    };
    let source = match MetadataSource::Unreported {
        MetadataSource::Unreported
        | MetadataSource::Disclosed
        | MetadataSource::Curated
        | MetadataSource::DisclosedAndCurated
        | _ => true,
    };
    let class = match ProbeClass::Auth {
        ProbeClass::Auth
        | ProbeClass::RateLimited
        | ProbeClass::Server
        | ProbeClass::Answered
        | ProbeClass::Network
        | ProbeClass::Unsupported
        | ProbeClass::Other
        | _ => true,
    };
    assert!(
        modality && source && class,
        "the marker sweep is a compile pin, not a behavior sweep"
    );
}
