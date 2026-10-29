//! Pins [`TELEMETRY.md`] to the telemetry source.
//!
//! The reference document is the inventory of record: every metric row
//! must exist in the registry catalog with the same kind and labels,
//! every catalog metric must have a row, and every span and signal-event
//! name must appear. A new emission without a row — or a row without an
//! emission — fails here, so the page cannot drift from the code it
//! describes.
//!
//! Feature-gated with the surface it inventories: the registry exists
//! only under `metrics`, and a build without it has no telemetry to
//! pin.

#![cfg(feature = "metrics")]
#![allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::panic
)]

use loopctl::metrics::registry::{self, MetricKind};

/// The rendered reference document.
fn reference() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/TELEMETRY.md"))
        .expect("TELEMETRY.md sits at the crate root")
}

/// The metric rows of the document: a (name, kind, labels) triple from
/// every metric table row.
fn metric_rows(document: &str) -> Vec<(String, &'static str, Vec<String>)> {
    let mut rows = Vec::new();
    for line in document.lines() {
        let Some(rest) = line.strip_prefix("| `loopctl.") else {
            continue;
        };
        // Only metric table rows carry the kind column; diagnostics and
        // rollup prose lines start the same way, so require the row to
        // parse into name / kind / labels cells.
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let Some(name) = cells[1]
            .trim_matches('`')
            .strip_prefix("loopctl.")
            .map(str::to_string)
        else {
            continue;
        };
        let _ = rest;
        let Some(kind) = ["counter", "gauge", "histogram"]
            .iter()
            .find(|kind| cells[2] == **kind)
        else {
            continue;
        };
        let labels = cells[3]
            .split(',')
            .map(|label| label.trim().trim_matches('`'))
            .filter(|label| !label.is_empty())
            .map(str::to_string)
            .collect();
        rows.push((name, *kind, labels));
    }
    rows
}

#[test]
fn every_catalog_metric_has_a_matching_row() {
    let document = reference();
    let rows = metric_rows(&document);
    for descriptor in registry::REGISTRY {
        let name = descriptor
            .name
            .strip_prefix("loopctl.")
            .unwrap_or(descriptor.name);
        let found = rows.iter().find(|(row_name, _, _)| row_name == name);
        let Some((_, kind, labels)) = found else {
            panic!(
                "TELEMETRY.md has no row for cataloged metric `{}` — add \
                 it or retire the metric",
                descriptor.name
            );
        };
        assert_eq!(
            *kind,
            match descriptor.kind {
                MetricKind::Counter => "counter",
                MetricKind::Gauge => "gauge",
                MetricKind::Histogram => "histogram",
            },
            "the row for `{}` names the wrong kind",
            descriptor.name
        );
        assert_eq!(
            labels, &descriptor.labels,
            "the row for `{}` lists different labels than the catalog",
            descriptor.name
        );
    }
}

#[test]
fn every_metric_row_exists_in_the_catalog() {
    let document = reference();
    for (name, _, _) in metric_rows(&document) {
        assert!(
            registry::lookup(&format!("loopctl.{name}")).is_some(),
            "TELEMETRY.md rows a metric `loopctl.{name}` the catalog does \
             not declare — the document may not invent signals"
        );
    }
}

#[test]
fn every_signal_event_name_is_pinned_and_documented() {
    let document = reference();
    for name in loopctl::tracing_spans::SIGNAL_EVENT_NAMES {
        assert!(
            document.contains(name),
            "TELEMETRY.md never names `{name}` — the inventory is incomplete"
        );
    }
    let section = document
        .split("## 2. Signal events")
        .nth(1)
        .and_then(|rest| rest.split("## 3.").next())
        .expect("the document carries its signal-events section");
    let mut documented: Vec<String> = section
        .lines()
        .filter_map(|line| {
            line.strip_prefix("| `loopctl.")
                .and_then(|rest| rest.split('`').next())
                .map(|suffix| format!("loopctl.{suffix}"))
        })
        .collect();
    documented.sort();
    let mut pinned: Vec<String> = loopctl::tracing_spans::SIGNAL_EVENT_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    pinned.sort();
    assert_eq!(
        documented, pinned,
        "the signal-event table and the pinned vocabulary must name \
         exactly the same events — a row without a signal or a signal \
         without a row fails here"
    );
    for name in [
        loopctl::tracing_spans::AGENT_RUN_SPAN,
        loopctl::tracing_spans::TURN_SPAN,
        loopctl::tracing_spans::TOOL_SPAN,
        "structured.extract",
        "trajectory.run",
    ] {
        assert!(
            document.contains(name),
            "TELEMETRY.md never names `{name}` — the inventory is incomplete"
        );
    }
}

/// The run-span `error.type` row documents exactly the classes the
/// classifier emits — a dashboard grouping on a documented spelling
/// must never miss the emitted one.
#[test]
fn the_error_type_row_lists_classes_the_classifier_emits() {
    use loopctl::error::LoopError;
    use loopctl::tracing_spans::error_type;

    let document = reference();
    let row = document
        .lines()
        .find(|line| line.starts_with("| `error.type` |") && line.contains("bounded class name"))
        .expect("the run-span table carries its bounded error.type row");
    let cell = row
        .split('|')
        .map(str::trim)
        .nth(2)
        .expect("the row carries its value cell");
    let classes: Vec<&str> = cell
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|class| !class.is_empty())
        .collect();
    assert!(
        !classes.is_empty(),
        "the row names its classes in backticks — an unparseable row is a \
         broken pin"
    );
    for class in classes {
        let err = match class {
            "cancelled" => LoopError::Cancelled,
            "api" => LoopError::Api("boom".to_string()),
            "tool_not_found" => LoopError::ToolNotFound {
                tool: "nope".to_string(),
                available: "echo".to_string(),
            },
            "tool_execution" => LoopError::ToolExecution {
                tool: "echo".to_string(),
                message: "kaput".to_string(),
            },
            "invalid_input" => LoopError::InvalidInput("bad".to_string()),
            "max_turns_exceeded" => LoopError::MaxTurnsExceeded { max: 3 },
            "context_exceeded" => LoopError::ContextExceeded { used: 10, limit: 8 },
            "budget_exhausted" => LoopError::BudgetExhausted {
                dimension: loopctl::budget::BudgetDimension::Tokens,
                spent: 10,
                limit: 8,
            },
            "loop_detected" => LoopError::LoopDetected {
                message: "spinning".to_string(),
            },
            "lock_poisoned" => LoopError::LockPoisoned {
                what: "store".to_string(),
            },
            "stream_error" => LoopError::StreamError("reset".to_string()),
            "config" => LoopError::Config("nope".to_string()),
            "memory" => LoopError::Memory("gone".to_string()),
            other => panic!(
                "no LoopError mapping for documented class `{other}` — the row \
                 lists a class the pin does not know; extend the mapping"
            ),
        };
        assert_eq!(
            error_type(&err),
            class,
            "the documented class must be exactly what the classifier emits"
        );
    }
}
