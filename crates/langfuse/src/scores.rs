//! Score payloads on traces.
//!
//! The score *definitions* live with the rest of the mapping surface
//! ([`metrics_to_scores`]); this module
//! owns only the wire shape — one ingestion envelope per score, keyed to a
//! trace id, numeric or categorical.

use serde_json::{Value, json};

use crate::ingest;
use crate::mapping::{LangfuseScore, metrics_to_scores};

/// Envelopes for one trace's standard score set.
///
/// Maps every score [`metrics_to_scores`] derives onto its own
/// envelope: called on run close when the layer carries a metrics
/// source, and by hosts exporting manually through the guard.
#[must_use]
pub fn score_events(trace_id: &str, metrics: &loopctl::metrics::AgentMetrics) -> Vec<Value> {
    metrics_to_scores(metrics)
        .into_iter()
        .map(|score| score_event(trace_id, &score))
        .collect()
}

/// One `score-create` envelope.
///
/// Numeric scores carry `value` directly; categorical ones anchor
/// `value` at zero and add `stringValue` plus
/// `dataType: "CATEGORICAL"`, which is what the cloud API requires
/// for categorical scores.
#[must_use]
pub fn score_event(trace_id: &str, score: &LangfuseScore) -> Value {
    let mut body = json!({
        "traceId": trace_id,
        "name": score.name,
        "source": "API",
    });
    if let Some(value) = score.value {
        ingest::set(&mut body, "value", json!(value));
    }
    if let Some(text) = &score.string_value {
        ingest::set(&mut body, "value", json!(0.0));
        ingest::set(&mut body, "stringValue", json!(text));
        ingest::set(&mut body, "dataType", json!("CATEGORICAL"));
    }
    ingest::envelope("score-create", &body)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc
    )]

    use loopctl::metrics::AgentMetrics;

    use super::*;

    #[test]
    fn score_event_shapes_numeric_and_categorical() {
        let mut metrics = AgentMetrics {
            run_success: true,
            ..AgentMetrics::default()
        };
        metrics.tokens.input = 10;
        let events = score_events("trace-1", &metrics);
        let numeric = events
            .iter()
            .find(|e| e["body"]["name"] == "input_tokens")
            .expect("input_tokens");
        assert_eq!(numeric["type"], "score-create");
        assert_eq!(numeric["body"]["traceId"], "trace-1");
        assert_eq!(numeric["body"]["value"], 10.0);
        let categorical = events
            .iter()
            .find(|e| e["body"]["name"] == "outcome")
            .expect("outcome");
        assert_eq!(categorical["body"]["stringValue"], "success");
        assert_eq!(categorical["body"]["dataType"], "CATEGORICAL");
    }
}
