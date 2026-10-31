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
/// Numeric scores carry `value` directly. A categorical score carries its
/// **label** in `value` under `dataType: "CATEGORICAL"`: the server's schema
/// discriminates on `dataType`, and a numeric `value` under that discriminator
/// is refused with a per-entry 400 (`expected string` at `body.value`) rather
/// than stored. The server derives `stringValue` from the label, so only
/// `value` is sent.
#[must_use]
pub fn score_event(trace_id: &str, score: &LangfuseScore) -> Value {
    let mut body = json!({
        "id": uuid::Uuid::new_v4().to_string(),
        "traceId": trace_id,
        "name": score.name,
        "source": "API",
    });
    if let Some(value) = score.value {
        ingest::set(&mut body, "value", json!(value));
    }
    if let Some(text) = &score.string_value {
        ingest::set(&mut body, "value", json!(text));
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
        assert!(
            numeric["body"]["id"].is_string(),
            "a score body without its own id is accepted by the server and never stored"
        );
        assert_eq!(numeric["body"]["traceId"], "trace-1");
        assert_eq!(numeric["body"]["value"], 10.0);
        let categorical = events
            .iter()
            .find(|e| e["body"]["name"] == "outcome")
            .expect("outcome");
        assert_eq!(categorical["body"]["value"], "success");
        assert_eq!(categorical["body"]["dataType"], "CATEGORICAL");
        assert!(
            categorical["body"]["value"].is_string(),
            "a numeric value under dataType CATEGORICAL is refused by the server"
        );
    }
}
