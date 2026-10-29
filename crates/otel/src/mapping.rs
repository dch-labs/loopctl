//! Name/kind/provider mapping — pass-through where loopctl is already
//! spec-shaped.
//!
//! loopctl's emitters carry spec-shaped attribute names, so this module
//! is deliberately *not* a translation layer: [`otel_attribute`] exists
//! to prove the identity contract, while span naming, span kinds, and
//! the coarse `gen_ai.system` derivation are the only real work here.

use std::borrow::Cow;

use opentelemetry::trace::SpanKind;

/// The attributes a span name depends on, extracted before renaming.
///
/// Renaming happens on the exported `OTel` span data, where the tracing name
/// has already lost per-span context; this struct carries the remainder
/// of what the name needs. Today only [`SpanFields::tool`] qualifies —
/// [`otel_span_name`] derives everything else from the tracing name.
#[derive(Debug, Default, Clone)]
pub struct SpanFields {
    /// `gen_ai.tool.name`, when the span describes a tool execution.
    ///
    /// Read from the exported attributes before the rename; spans that
    /// carry no such key fall back to the bare `execute_tool` name.
    pub tool: Option<String>,
}

/// Which tracing-level attribute keys are exported to `OTel`, unchanged.
///
/// Every key loopctl emits is already spec-shaped (`gen_ai.*`,
/// `error.type`, and the documented extras), so the bridge copies: the
/// input is returned verbatim for every key. The function exists as the
/// identity contract's test point — the one place a key would be
/// renamed or dropped, and the one place a test would catch it.
#[must_use]
pub fn otel_attribute(key: &str) -> &str {
    key
}

/// `OTel` span name per the `GenAI` conventions.
///
/// `agent.run` → `invoke_agent`; `agent.tool` → `execute_tool <tool>`
/// (the qualifier the tracing name alone lacks); `agent.turn` passes
/// through — the spec has no turn span, so loopctl's name stands.
#[must_use]
pub fn otel_span_name(tracing_name: &str, fields: &SpanFields) -> String {
    match tracing_name {
        "agent.run" => "invoke_agent".to_string(),
        "agent.tool" => match &fields.tool {
            Some(tool) => format!("execute_tool {tool}"),
            None => "execute_tool".to_string(),
        },
        other => other.to_string(),
    }
}

/// `OTel` span kind for a tracing span name.
///
/// `INTERNAL` for every name: the loopctl spans describe in-process
/// work, and the bridge assigns no other kind — a span that names a
/// remote call would carry `CLIENT`, which none of loopctl's emitted
/// vocabulary does (provider-call detail is recorded as attributes on
/// the enclosing span).
#[must_use]
pub fn otel_span_kind(_tracing_name: &str) -> SpanKind {
    SpanKind::Internal
}

/// Coarse provider family for `gen_ai.system`, derived from a model id.
///
/// Prefix table, not a rename: the OpenAI families (`gpt-*`,
/// `chatgpt-*`, `o1`/`o3`/`o4`, `text-embedding*`) → `openai`,
/// `claude*` → `anthropic`, `gemini*` → `google.genai`, `deepseek*` →
/// `deepseek`, `qwen*` → `qwen`, `llama*` → `llama`, `mistral*` and
/// `magistral*` → `mistral`; everything unrecognized → `unknown`.
#[must_use]
pub fn map_system(model: &str) -> &'static str {
    let lower = model.to_ascii_lowercase();
    let head = lower
        .split(['-', '/', ':'])
        .next()
        .unwrap_or(lower.as_str());
    if lower.starts_with("gpt-")
        || lower.starts_with("chatgpt-")
        || matches!(head, "o1" | "o3" | "o4")
        || head.starts_with("text-embedding")
    {
        "openai"
    } else if lower.starts_with("claude") {
        "anthropic"
    } else if lower.starts_with("gemini") {
        "google.genai"
    } else if lower.starts_with("deepseek") {
        "deepseek"
    } else if lower.starts_with("qwen") {
        "qwen"
    } else if lower.starts_with("llama") {
        "llama"
    } else if lower.starts_with("mistral") || lower.starts_with("magistral") {
        "mistral"
    } else {
        "unknown"
    }
}

/// Whether a tracing span name is one loopctl's bridge recognizes.
///
/// The transform leaves everything outside this vocabulary exactly as
/// the stock layer produced it, so third-party spans sharing the same
/// subscriber keep their own names, kinds, and statuses.
#[must_use]
pub fn is_loopctl_span(name: &str) -> bool {
    matches!(name, "agent.run" | "agent.turn" | "agent.tool")
}

/// Attribute key the bridge sets on the run span when stamping systems.
///
/// Derived by [`map_system`] from the requested model, and only when
/// the engine recorded no `gen_ai.system` of its own.
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";

/// Attribute carrying the tool name on `agent.tool` spans.
///
/// Emitted verbatim by the engine's tool span; [`otel_span_name`] is
/// its consumer, qualifying the `execute_tool <tool>` span name.
pub const TOOL_NAME_KEY: &str = "gen_ai.tool.name";

/// Attribute carrying the run/turn/tool outcome (`"ok"` / `"error"`).
///
/// The span transform reads it to set `OTel` status; the metrics
/// catalog carries the same vocabulary on the engine's own events.
pub const OUTCOME_KEY: &str = "outcome";

/// Attribute carrying the classified error type on failed spans.
///
/// Becomes the `OTel` error status message; a span that records an
/// error `outcome` without this key still marks, with a plain
/// `"error"` message.
pub const ERROR_TYPE_KEY: &str = "error.type";

/// Attribute carrying the requested model id.
///
/// Doubles as the input to [`map_system`] whenever no explicit
/// `gen_ai.system` was recorded on the run span.
pub const REQUEST_MODEL_KEY: &str = "gen_ai.request.model";

/// The tracing name of the run span.
///
/// The transform's run-stamping branch keys on this name; it is the
/// same span core opens in its `tracing_spans` vocabulary.
pub const AGENT_RUN: &str = "agent.run";

/// Re-export of [`Cow`] keeps callers' import lists honest in tests.
///
/// The `OTel` SDK names spans with `Cow<'static, str>`; the alias saves
/// test and example imports — production paths use the SDK type
/// directly.
pub type SpanName = Cow<'static, str>;

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc)]

    use super::*;

    #[test]
    fn attribute_keys_pass_through_unchanged() {
        for key in ["gen_ai.tool.name", "error.type", "duration_ms", "x"] {
            assert_eq!(otel_attribute(key), key);
        }
    }

    #[test]
    fn span_names_follow_the_genai_conventions() {
        let none = SpanFields::default();
        assert_eq!(otel_span_name("agent.run", &none), "invoke_agent");
        assert_eq!(otel_span_name("agent.turn", &none), "agent.turn");
        let tool = SpanFields {
            tool: Some("bash".to_string()),
        };
        assert_eq!(otel_span_name("agent.tool", &tool), "execute_tool bash");
        assert_eq!(otel_span_name("agent.tool", &none), "execute_tool");
    }

    #[test]
    fn systems_derive_from_model_prefixes() {
        assert_eq!(map_system("gpt-4o-mini"), "openai");
        assert_eq!(map_system("o3-mini-2025"), "openai");
        assert_eq!(map_system("claude-sonnet-4-5"), "anthropic");
        assert_eq!(map_system("gemini-2.5-pro"), "google.genai");
        assert_eq!(map_system("deepseek-chat"), "deepseek");
        assert_eq!(map_system("huggingface/some-model"), "unknown");
    }
}
