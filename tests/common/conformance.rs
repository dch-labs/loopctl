//! The provider conformance contract.
//!
//! One behavioral contract every `ApiClient` must uphold, written once
//! and run per provider over recorded cassette fixtures. Each check
//! returns named [`Violation`]s instead of panicking, so the suite can
//! report *which* rule broke and the meta-pin can prove it caught a
//! deliberate planting. The fixture is the oracle: expected usage
//! numbers, stop reasons, retry-after values, and wire models are all
//! read from the recorded bytes, never hardcoded here.
//!
//! The five families and the behavior each one protects — this is the
//! executable index of the wire-behavior fixes:
//!
//! | Family | Rule (provenance) |
//! |---|---|
//! | `stream_lifecycle` | A success stream ends in exactly one `MessageStop`, as the terminal event, with nothing after it; a stream whose provider stream never reached its terminal must not receive a synthetic one (the 0.3.0 no-synthetic-stop rule) — and a fixture that itself ends before the provider's terminal is a fixture-integrity gap (`n/r`), never a verdict on the client, which correctly ends a truncated stream in silence. Lanes open with `PartStart` and close with `PartStop` on the same index. |
//! | `usage_reporting` | Token counts reach the terminal `MessageDelta` exactly as the provider reported them, merged field-wise by max across repeats (the 0.3.x usage-latch rule); a provider that reports nothing must see nothing fabricated (the unreported ≠ zero rule). On a fixture cut below its terminal the carrying rule is a fixture-integrity gap, and on an error fixture the family is not applicable — nothing is owed a usage block there. |
//! | `stop_reason_routing` | `tool_use`-class stops route through `StreamStopReason::ToolCall` and a named, closed tool lane; text stops never route to the tool class; every terminal reports a reason the table can parse. Not applicable to a stream that ended in a typed error: no stop reason was ever reported. |
//! | `error_taxonomy` | 401/403 → `Auth`; 429/503/529 → `RateLimit` carrying the recorded `Retry-After` — exactly, so a delay the response never advised is as much a violation as one it advised and the client dropped; anything else → typed `Http` with status. One recorded interaction means one attempt — replay completeness is the no-retry-after-auth-guard. A success fixture decides only "no error on success"; the status arms need a recorded failure, and without one the family is a reported gap, never a silent pass. |
//! | `options_honoring` | The per-request model override is what the wire carried; the recorded `include_usage` flag matches what the scenario asked; `supports_tool_constraints()` tells the truth about the client. Every fixture carries the probe and the usage-flag check, so every fixture is a verdict. |
//!
//! Two kinds of not-green and their difference matters: a
//! **fixture-integrity gap** blames the recording (it ends below its
//! provider terminal, so terminal-shaped rules cannot be judged) and is
//! fixed by re-recording; a **coverage gap** says the corpus holds no
//! fixture that could test the rule at all and is fixed by recording one.
//! Neither is ever reported as a client violation, and neither is
//! silently counted green.

use futures::StreamExt;
use loopctl::api::ApiClient;
use loopctl::api::error::ApiError;
use loopctl::stream::StreamEvent;

use crate::cassette;

/// The contract families, in report order.
///
/// Column order of the rendered [`Matrix`] and the index basis of
/// [`RunReport::families`]; a family is added by extending this table
/// and writing its check, never by reordering the existing names.
pub const FAMILIES: [&str; 5] = [
    "stream_lifecycle",
    "usage_reporting",
    "stop_reason_routing",
    "error_taxonomy",
    "options_honoring",
];

/// One broken contract rule, named precisely enough to act on.
///
/// Produced by a family check instead of panicking, so one run reports
/// every rule it broke, and the meta-pin's deliberately planted fixture
/// edits surface as exactly this shape — named, attributed, actionable.
#[derive(Debug, Clone)]
pub struct Violation {
    /// The profile that ran.
    ///
    /// The matrix column the violation lands in; it names the wire
    /// dialect the broken rule was adjudicated under.
    pub provider: String,

    /// The cassette scenario that was replayed.
    ///
    /// Names the recording that proves the violation, so the repair
    /// starts at a known fixture rather than at a stack frame.
    pub scenario: String,

    /// The contract family that caught it (a [`FAMILIES`] name).
    ///
    /// The family tags *which* rule broke, keeping the two failure kinds
    /// separable: a `Fail` can never be confused with a fixture gap or
    /// a coverage hole, which ride [`CellStatus::NotRun`] instead.
    pub family: &'static str,

    /// What the contract promised and what the run did instead.
    ///
    /// Written to read on its own: the matrix cell quotes this text, so
    /// the reader never has to cross-reference the scenario roll-up to
    /// know what to fix.
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "conformance violation: {}/{} [{}]: {}",
            self.provider, self.scenario, self.family, self.detail
        )
    }
}

/// One client built for one provider profile.
///
/// The bundle a [`ProviderProfile::build`] hands back after pointing a
/// real client at the cassette server; the battery holds only the trait
/// object, so every profile instantiates through one shape.
pub struct ClientBundle {
    /// The client under conformance.
    ///
    /// Built fresh per scenario from dummy credentials and the mock
    /// base URL supplied by the runner — never from the environment, so
    /// a run cannot leak an ambient key onto the replay endpoint.
    pub client: Box<dyn ApiClient>,
}

/// One provider's conformance configuration — instantiation, not code.
///
/// A new provider joins by adding a profile here (behind its feature)
/// plus, to green every applicable family, cassettes; the battery is
/// the shared checks below and never changes.
pub struct ProviderProfile {
    /// The cassette folder / dialect name.
    ///
    /// Keys fixture discovery (`cassettes/<provider>/…`) and doubles as
    /// the provider's display name in every report and matrix column.
    pub provider: &'static str,

    /// The model clients are configured with when no per-request
    /// override is in play.
    ///
    /// The `model_override` scenario is driven against this default, so
    /// the recorded wire model must prove the request left this string
    /// behind — a fixture that records the default proves nothing about
    /// the override path and is rejected as evidence.
    pub default_model: &'static str,

    /// A client builder for one scenario run.
    ///
    /// Hands back a [`ClientBundle`] wired to dummy credentials and the
    /// mock server's base URL, configured with the profile's model; the
    /// override scenario reaches a different name only through the
    /// per-request options, which is the path under test.
    pub build: fn(mock_base_url: &str, scenario: &cassette::Scenario) -> ClientBundle,

    /// Whether the client truthfully claims tool-constraint support.
    ///
    /// Cross-checked against `supports_tool_constraints()` by
    /// [`check_options_honoring`]: a probe that lies in either direction
    /// is a violation, because the engine decides constraint emission
    /// on the answer the probe gives.
    pub expects_tool_constraints: bool,
}

/// Everything one scenario run produced.
///
/// The interpreted evidence both halves of a verdict read from: the
/// family checks assert against the events, the error taxonomy against
/// the typed errors — a success run carries an empty `errors`, an error
/// fixture reaches no terminal events.
pub struct RunOutput {
    /// The interpreted events, in order.
    ///
    /// Exactly what the client streamed for the one request the cassette
    /// replays — the sequence [`check_stream_lifecycle`] counts terminals
    /// and lane pairs in.
    pub events: Vec<StreamEvent>,

    /// The typed errors, in order (success streams yield none).
    ///
    /// Errors the client surfaced to its caller, not transport noise
    /// swallowed on the way; [`check_error_taxonomy`] classifies each
    /// against the recorded status.
    pub errors: Vec<ApiError>,
}

/// A provider's `Retry-After` as recorded on the fixture, parsed — the
/// oracle side of the rate-limit arm.
///
/// Reads through the crate's own parser so a delta-seconds **and** an
/// HTTP-date recording mean the same thing to oracle and client. The
/// advice is sourced from the fixture's failed response (see
/// [`error_interaction`]): a warm-up success that carried headers must
/// neither hide nor impersonate the 429's own advice.
pub fn recorded_retry_after(interactions: &[cassette::Interaction]) -> Option<std::time::Duration> {
    error_interaction(interactions)
        .and_then(|interaction| interaction.then.header.as_ref())
        .and_then(|headers| {
            headers
                .iter()
                .find(|pair| pair.name.eq_ignore_ascii_case("retry-after"))
                .and_then(|pair| loopctl::api::error::parse_retry_after(&pair.value))
        })
}

/// The interaction holding the fixture's failed response, when present.
///
/// The **last** interaction with an error status is the one the client
/// errored on — the response a `Retry-After` rule must be read from.
/// When no interaction records a status at all (the legacy single-response
/// fixture shape), the last one carrying headers stands in: an error
/// rides at the end of a recording, never the front.
fn error_interaction(interactions: &[cassette::Interaction]) -> Option<&cassette::Interaction> {
    interactions
        .iter()
        .rev()
        .find(|interaction| interaction.then.status.is_some_and(|status| status >= 400))
        .or_else(|| {
            interactions
                .iter()
                .rev()
                .find(|interaction| interaction.then.header.is_some())
        })
}

/// The first recorded response status, when the fixture carries one.
///
/// The shape discriminator for the fixture-driven arms: a status at or
/// above 400 marks the recording as an error fixture, and a legacy
/// single-response fixture without a status field reads as the success
/// shape it was recorded before status fields existed.
fn fixture_status(interactions: &[cassette::Interaction]) -> Option<u16> {
    interactions.first().and_then(|i| i.then.status)
}

/// One provider-reported usage pair; per-field `None` = unreported.
///
/// The oracle unit of the usage latch: each recorded report payload
/// yields one of these, and [`UsageReport`] folds them with the same
/// field-wise max the interpretation under test must apply.
#[derive(Debug, Clone, Copy, Default)]
pub struct UsagePair {
    /// Tokens in, when reported.
    ///
    /// Read from the dialect's input key (`prompt_tokens`,
    /// `input_tokens`, `promptTokenCount`); `None` is absence, not zero.
    pub input: Option<u64>,

    /// Tokens out, when reported.
    ///
    /// The dialect's matching output key; `None` is absence — a
    /// provider that reported 0 output did report, which is a different
    /// fact this type keeps addressable.
    pub output: Option<u64>,
}

/// The fixture's usage oracle: every recorded report, field-wise
/// max-merged.
///
/// Applies the identical latch rule the client interpretation must
/// apply, but to the recorded bytes rather than the event stream — so
/// the check compares two honest readings of one fixture instead of a
/// reading against a hardcoded expectation.
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageReport {
    /// Any provider reported input tokens.
    ///
    /// The field-wise max over every recorded report's input — the
    /// latch shape the usage-merge contract pins, arrived at
    /// independently of the client under test.
    pub input: Option<u64>,

    /// Any provider reported output tokens.
    ///
    /// Max-merged the same way; together with [`UsageReport::present`]
    /// it separates "reported nothing" from "reported zero".
    pub output: Option<u64>,

    /// Whether any usage appeared at all.
    ///
    /// Presence is its own fact, distinct from the counts: a provider
    /// that reported zeroes did report, and a stream that sent no usage
    /// block must not be scored against `input`/`output` at all.
    pub present: bool,
}

impl UsageReport {
    /// Latch one report with the field-wise-max rule.
    ///
    /// Applies to the recorded bytes the identical merge the
    /// interpretation under test must apply, so the comparison is two
    /// honest readings of one fixture, never a number written by the
    /// test author.
    pub fn absorb(&mut self, pair: UsagePair) {
        self.present = true;
        self.input = max_opt(self.input, pair.input);
        self.output = max_opt(self.output, pair.output);
    }
}

/// Field-wise max of two optional counts, treating `None` as no report.
///
/// Two reports take the larger; a report beside an absence keeps the
/// report; two absences stay absent — never folded to zero, which the
/// oracle's unreported-is-not-zero rule depends on.
fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, y) => x.or(y),
    }
}

/// The fixture's usage oracle: parse every recorded response payload
/// and collect what the provider reported.
///
/// Dialect key sets all recognized here (a fixture either carries usage
/// keys or it does not): OpenAI-family
/// `usage.{prompt_tokens,completion_tokens}`, Anthropic-family
/// `usage.{input_tokens,output_tokens}`, and Gemini
/// `usageMetadata.{promptTokenCount,candidatesTokenCount}`. Absent
/// numbers stay `None` — zero is a report, absence is not.
fn fixture_usage(interactions: &[cassette::Interaction]) -> UsageReport {
    let mut report = UsageReport::default();
    for interaction in interactions {
        let Some(body) = interaction.then.body.as_deref() else {
            continue;
        };
        for payload in wire_payloads(body) {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&payload) else {
                continue;
            };
            if let Some(pair) = usage_from_value(&value) {
                report.absorb(pair);
            }
        }
    }
    report
}

/// Split a recorded response body into JSON payloads: SSE `data:`
/// lines on streaming bodies, the whole body otherwise.
///
/// Streaming recordings wrap each chunk in an SSE frame and end on the
/// `[DONE]` sentinel — only the framed payloads are JSON, and the
/// sentinel is transport punctuation, not a payload. A body carrying
/// no frames *is* one payload: the non-streaming shape, pretty-printed
/// or not, lands in that branch.
fn wire_payloads(body: &str) -> Vec<String> {
    let mut payloads = Vec::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            let payload = rest.trim();
            if !payload.is_empty() && payload != "[DONE]" {
                payloads.push(payload.to_string());
            }
        }
    }
    if payloads.is_empty() {
        payloads.push(body.to_string());
    }
    payloads
}

/// Pull one JSON value's usage pair, trying every dialect's key set.
///
/// The three spellings and their keys are enumerated on
/// [`fixture_usage`]; Anthropic additionally nests usage inside
/// `message_start` / `message_delta` event objects, so a value that
/// carries none of the keys here descends into its objects and arrays
/// looking for them.
fn usage_from_value(value: &serde_json::Value) -> Option<UsagePair> {
    let num = |v: Option<&serde_json::Value>| v.and_then(serde_json::Value::as_u64);
    if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
        let input = num(usage.get("prompt_tokens")).or_else(|| num(usage.get("input_tokens")));
        let output =
            num(usage.get("completion_tokens")).or_else(|| num(usage.get("output_tokens")));
        if input.is_some() || output.is_some() {
            return Some(UsagePair { input, output });
        }
    }
    if let Some(meta) = value.get("usageMetadata").filter(|m| !m.is_null()) {
        let input = num(meta.get("promptTokenCount"));
        // The crate's documented Gemini reading folds thinking tokens
        // into output (pinned in the provider unit tests): the oracle
        // folds the same way so conformance measures interpretation,
        // not a second opinion on billing semantics.
        let output = num(meta.get("candidatesTokenCount"))
            .map(|c| c + num(meta.get("thoughtsTokenCount")).unwrap_or(0));
        if input.is_some() || output.is_some() {
            return Some(UsagePair { input, output });
        }
    }
    match value {
        serde_json::Value::Object(map) => map.values().find_map(usage_from_value),
        serde_json::Value::Array(items) => items.iter().find_map(usage_from_value),
        _ => None,
    }
}

/// The terminal `MessageDelta`'s usage, if the stream produced one.
///
/// Read from the back: the contract places the terminal last, and usage
/// that reached it is the number the oracle compares against — earlier
/// reports without a terminal are the truncation case, screened out by
/// the fixture-gap arms before this reads anything.
fn terminal_usage(events: &[StreamEvent]) -> Option<loopctl::stream::Usage> {
    events.iter().rev().find_map(|event| match event {
        StreamEvent::MessageDelta(md) => md.usage,
        _ => None,
    })
}

/// The stop reason the terminal `MessageDelta` reported.
///
/// Read from the back for the same reason as [`terminal_usage`]; the
/// absence of a parseable reason at a terminal is precisely what the
/// routing family reports.
fn reported_stop_reason(events: &[StreamEvent]) -> Option<String> {
    events.iter().rev().find_map(|event| match event {
        StreamEvent::MessageDelta(md) => md.delta.stop_reason.clone(),
        _ => None,
    })
}

/// Whether the recorded body shows the provider reaching its own
/// terminal.
///
/// Either an SSE `[DONE]` or a JSON payload carrying a string-valued
/// finish/stop marker — a truncated fixture (the meta-pin's plant)
/// lacks both. Detection parses the payloads rather than matching
/// substrings: providers pretty-print their JSON, and a space after
/// the colon must not hide a terminal.
fn provider_reached_terminal(interactions: &[cassette::Interaction]) -> bool {
    interactions.iter().any(|i| {
        i.then.body.as_deref().is_some_and(|body| {
            body.contains("[DONE]")
                || wire_payloads(body)
                    .iter()
                    .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
                    .any(|value| value_has_terminal(&value))
        })
    })
}

/// One payload (or its nesting) carries a string-valued terminal field.
///
/// The JSON-semantic leg of [`provider_reached_terminal`]: it descends
/// objects and arrays looking for the dialects' finish/stop fields
/// holding string values, which substring matching lost on
/// pretty-printed recordings.
fn value_has_terminal(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            for (key, inner) in map {
                if matches!(
                    key.as_str(),
                    "finish_reason" | "stop_reason" | "finishReason"
                ) && inner.is_string()
                {
                    return true;
                }
                if key == "type" && inner.as_str() == Some("message_stop") {
                    return true;
                }
                if value_has_terminal(inner) {
                    return true;
                }
            }
            false
        }
        serde_json::Value::Array(items) => items.iter().any(value_has_terminal),
        _ => false,
    }
}

/// Tool lanes the stream surfaced: lane index plus the name
/// `PartStart` announced, in order.
///
/// Identity is read at the lane's open, where the event contract puts
/// it: the routing rule checks the *requested* name against this list,
/// and a lane that opened without one stays listed with a `None` name
/// rather than being skipped — an unnamed lane is a finding, not an
/// absence.
fn tool_lanes(events: &[StreamEvent]) -> Vec<(usize, Option<String>)> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::PartStart(start) => match &start.part {
                Some(loopctl::message::MessagePart::ToolCall { name, .. }) => {
                    Some((start.index, Some(name.clone())))
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Accumulated tool-argument JSON across a dialect's two delta shapes.
///
/// Concatenates the argument fragments the interpretation surfaced, in
/// stream order, so the routing check parses one JSON document per lane
/// instead of judging mid-flight fragments that cannot be valid yet.
fn tool_argument_text(events: &[StreamEvent]) -> String {
    let mut text = String::new();
    for event in events {
        if let StreamEvent::IndexedDelta(delta) = event {
            match &delta.delta {
                loopctl::stream::DeltaPart::ToolCall { partial_json } => {
                    text.push_str(&partial_json.to_string());
                }
                loopctl::stream::DeltaPart::InputJson { partial_json } => {
                    text.push_str(partial_json);
                }
                _ => {}
            }
        }
    }
    text
}

/// Lanes opened by `PartStart` and never closed by a matching
/// `PartStop`.
///
/// An indexless `PartStop` closes the oldest open lane — the legacy
/// rule the event contract documents — so a balanced stream reads empty
/// here whatever spelling its closes used; a lane still open at the
/// terminal is the lifecycle violation this channel reports.
fn unclosed_lanes(events: &[StreamEvent]) -> Vec<usize> {
    let mut open: Vec<usize> = Vec::new();
    for event in events {
        match event {
            StreamEvent::PartStart(start) => open.push(start.index),
            StreamEvent::PartStop { index } => match index {
                Some(index) => {
                    if let Some(position) = open.iter().position(|lane| lane == index) {
                        open.remove(position);
                    }
                }
                None => {
                    if !open.is_empty() {
                        open.remove(0);
                    }
                }
            },
            _ => {}
        }
    }
    open
}

/// A violation before its provider and scenario are known: the family
/// tag plus the detail.
///
/// The shape the family checks report in; [`run_scenario`] and the
/// matrix builders dress each into a full [`Violation`] once the run
/// context is available.
type Finding = (&'static str, String);

/// One battery run: named rule violations, the rules the fixture's shape
/// could not adjudicate, and the families no fixture of this shape can
/// test.
///
/// Neither gap is a pass or a failure: a fixture that ends before the
/// provider's terminal proves nothing about the terminal-shaped rules,
/// and a success fixture proves nothing about the status arms of the
/// error taxonomy. Both show up in the matrix as `n/r` with their reason
/// instead of blaming a conformant client or silently counting green —
/// which is why the checks report them and the aggregator cannot invent
/// them afterwards.
#[derive(Debug, Default)]
pub struct BatteryReport {
    /// Broken contract rules, named precisely enough to act on.
    ///
    /// The verdict channel: a non-empty list fails the run, and the
    /// first entry's detail is what the matrix cell quotes for a `FAIL`.
    pub violations: Vec<Finding>,

    /// Rules the fixture's shape made unjudgeable, each with its reason.
    ///
    /// The fixture-integrity channel — it blames the recording, not the
    /// client: this run ended below its provider terminal, so the
    /// terminal-shaped rules in the list cannot be scored against it.
    pub fixture_gaps: Vec<Finding>,

    /// The fixture's shape cannot exercise this family at all, whatever
    /// the client did.
    ///
    /// The coverage channel: a hole in the corpus, not a verdict, naming
    /// the fixture whose recording would close it.
    pub not_applicable: Vec<Finding>,
}

impl BatteryReport {
    /// Fold another family's outcome into this one, channel by channel.
    ///
    /// Channels never cross: a violation stays a violation no matter how
    /// many gaps the sibling report contributes, which is what keeps
    /// "not decided" from diluting — or hiding — "decided against".
    pub fn merge(&mut self, other: Self) {
        self.violations.extend(other.violations);
        self.fixture_gaps.extend(other.fixture_gaps);
        self.not_applicable.extend(other.not_applicable);
    }
}

/// A success-shaped fixture whose recorded provider stream never reaches
/// its terminal.
///
/// The shape that makes every terminal-shaped rule unjudgeable against
/// it: the recording stopped short, not the client, so checks report
/// fixture gaps on this shape and the matrix renders the cell `n/r`
/// rather than scoring silence either way.
pub fn fixture_truncated(interactions: &[cassette::Interaction]) -> bool {
    !fixture_status(interactions).is_some_and(|s| s >= 400)
        && !provider_reached_terminal(interactions)
}

/// `stream_lifecycle`: exactly one terminal, nothing after it, balanced
/// lanes.
///
/// On a fixture that ends before the provider's terminal the family
/// reports a gap rather than a verdict: the pinned client contract for a
/// truncated stream is silence (the 0.3.0 no-synthetic-stop rule), so
/// the run can only decide what it can actually see — a fabricated
/// terminal is decidable, silence is not.
pub fn check_stream_lifecycle(
    output: &RunOutput,
    interactions: &[cassette::Interaction],
) -> BatteryReport {
    let mut report = BatteryReport::default();
    let status = fixture_status(interactions);
    let errored = !output.errors.is_empty();
    let stops = output
        .events
        .iter()
        .filter(|event| matches!(event, StreamEvent::MessageStop))
        .count();

    if errored {
        if stops > 0 {
            report.violations.push((
                "stream_lifecycle",
                "a stream that surfaced a typed error also emitted MessageStop — \
                 errors end the stream; a stop after an error is a second terminal"
                    .to_string(),
            ));
        }
        return report;
    }

    if status.is_some_and(|s| s >= 400) {
        report.violations.push((
            "stream_lifecycle",
            "an error fixture yielded no typed error at all — the failure surfaced as \
             events (or as nothing) instead"
                .to_string(),
        ));
        return report;
    }

    // Success-shaped fixture.
    if !provider_reached_terminal(interactions) {
        // The pinned client contract for a truncated stream is
        // events-then-silence: no MessageStop and no error — the
        // engine, not the client, reads the missing terminal as
        // truncation (pinned in provider unit tests as
        // finish_without_done_marker_emits_no_message_stop). Silence
        // here is therefore conformance, not a violation, and no
        // terminal-shaped rule is decidable: fixture-integrity gap.
        if stops > 0 {
            report.violations.push((
                "stream_lifecycle",
                "MessageStop emitted although the provider stream never reached its \
                 terminal — the 0.3.0 rule: a truncated stream gets no synthetic stop"
                    .to_string(),
            ));
        } else {
            report.fixture_gaps.push((
                "stream_lifecycle",
                "the fixture ends before the provider's terminal — no terminal-shaped \
                 rule is decidable against it; re-record the scenario whole"
                    .to_string(),
            ));
        }
        return report;
    }

    if stops == 0 {
        report.violations.push((
            "stream_lifecycle",
            "the provider stream reached its terminal but interpretation never \
             emitted MessageStop — a consumer waiting for the turn to end waits \
             forever"
                .to_string(),
        ));
        return report;
    }
    if stops > 1 {
        report.violations.push((
            "stream_lifecycle",
            format!("MessageStop emitted {stops} times — exactly one terminal per turn"),
        ));
        return report;
    }

    if output
        .events
        .last()
        .is_some_and(|last| !matches!(last, StreamEvent::MessageStop))
    {
        report.violations.push((
            "stream_lifecycle",
            "events followed MessageStop — the terminal is terminal, nothing may \
             arrive after it"
                .to_string(),
        ));
    }
    if !matches!(output.events.first(), Some(StreamEvent::MessageStart(_))) {
        report.violations.push((
            "stream_lifecycle",
            "the first event of a success stream is not MessageStart".to_string(),
        ));
    }
    let unclosed = unclosed_lanes(&output.events);
    if !unclosed.is_empty() {
        report.violations.push((
            "stream_lifecycle",
            format!(
                "lanes left open at the terminal: {unclosed:?} — every PartStart \
                 closes with a PartStop on the same index"
            ),
        ));
    }
    report
}

/// `usage_reporting`: the fixture's reported numbers reach the terminal
/// `MessageDelta`, field-wise max-merged; unreported stays unfabricated.
///
/// A fixture that is not a success response owes no usage block, so the
/// family is not applicable there. On a success fixture cut below its
/// provider terminal the numbers still merge, but the rule that the
/// terminal *carried* them is undecidable: the pinned client contract
/// answers a truncated stream with silence (the 0.3.0 rule), so the run
/// reports a fixture-integrity gap rather than blaming a client that did
/// what the contract requires.
pub fn check_usage_reporting(
    output: &RunOutput,
    interactions: &[cassette::Interaction],
) -> BatteryReport {
    let mut report = BatteryReport::default();
    if !fixture_status(interactions).is_some_and(|s| s == 200) {
        report.not_applicable.push((
            "usage_reporting",
            "the fixture is not a success response — no provider reports a usage \
             block on it, so there is nothing to compare"
                .to_string(),
        ));
        return report;
    }
    let oracle = fixture_usage(interactions);
    let seen = terminal_usage(&output.events);

    if !oracle.present {
        if let Some(usage) = seen.filter(|u| u.input_tokens > 0 || u.output_tokens > 0) {
            report.violations.push((
                "usage_reporting",
                format!(
                    "the provider reported no usage on this stream, but \
                     interpretation surfaced ({}, {}) — unreported must never \
                     become a fabricated number",
                    usage.input_tokens, usage.output_tokens
                ),
            ));
        }
        return report;
    }

    let Some(usage) = seen else {
        if fixture_truncated(interactions) {
            report.fixture_gaps.push((
                "usage_reporting",
                "the fixture is cut below the provider's terminal while usage was \
                 recorded — silence is what a conformant client owes a truncated \
                 stream, so whether the numbers reach the terminal is undecidable; \
                 re-record the scenario whole"
                    .to_string(),
            ));
        } else {
            report.violations.push((
                "usage_reporting",
                format!("the provider reported usage {oracle:?} but no MessageDelta carried it"),
            ));
        }
        return report;
    };
    if let Some(want) = oracle.input
        && u64::from(usage.input_tokens) != want
    {
        report.violations.push((
            "usage_reporting",
            format!(
                "input tokens {} != recorded {want} (recorded reports merge \
                 field-wise by max — the 0.3.x latch rule)",
                usage.input_tokens
            ),
        ));
    }
    if let Some(want) = oracle.output
        && u64::from(usage.output_tokens) != want
    {
        report.violations.push((
            "usage_reporting",
            format!(
                "output tokens {} != recorded {want} (field-wise max latch)",
                usage.output_tokens
            ),
        ));
    }
    report
}

/// `stop_reason_routing`: tool vs text route to the same decision, and
/// the tool lane carries the requested identity.
///
/// Two fixture shapes settle before any routing rule is read: a stream
/// that ended in a typed error never reported a stop reason (not
/// applicable), and a success fixture cut below its provider terminal
/// has no terminal to classify (fixture-integrity gap) — the client is
/// correct to have surfaced nothing, and a fabricated terminal is
/// `stream_lifecycle`'s to catch.
pub fn check_stop_reason_routing(
    output: &RunOutput,
    scenario: &cassette::Scenario,
    interactions: &[cassette::Interaction],
) -> BatteryReport {
    let mut report = BatteryReport::default();
    if !output.errors.is_empty() {
        report.not_applicable.push((
            "stop_reason_routing",
            "the stream ended in a typed error — no stop reason was ever reported, \
             so the routing table has nothing to classify"
                .to_string(),
        ));
        return report;
    }
    if fixture_truncated(interactions) {
        report.fixture_gaps.push((
            "stop_reason_routing",
            "the fixture ends before the provider's terminal — the routing table has \
             nothing to classify against it; re-record the scenario whole"
                .to_string(),
        ));
        return report;
    }
    let reason = reported_stop_reason(&output.events);
    let lanes = tool_lanes(&output.events);
    let mapped = reason
        .as_deref()
        .map(loopctl::stream::StreamStopReason::from_api_str);

    if scenario.tools {
        if lanes.is_empty() {
            report.violations.push((
                "stop_reason_routing",
                "a tool-carrying turn surfaced no tool lane at all — the call is \
                 invisible to the engine's routing"
                    .to_string(),
            ));
        } else if !lanes
            .iter()
            .any(|(_, name)| name.as_deref() == Some("get_weather"))
        {
            report.violations.push((
                "stop_reason_routing",
                format!(
                    "the tool lane never carried the requested tool identity \
                     'get_weather' — lanes carry: {lanes:?}"
                ),
            ));
        }
        let arguments = tool_argument_text(&output.events);
        if !arguments.trim().is_empty()
            && serde_json::from_str::<serde_json::Value>(&arguments).is_err()
        {
            report.violations.push((
                "stop_reason_routing",
                format!("accumulated tool arguments do not parse as JSON: {arguments:?}"),
            ));
        }
        if !matches!(
            mapped,
            Some(Some(loopctl::stream::StreamStopReason::ToolCall))
        ) {
            // Provider truth is not a violation: Gemini's wire reports
            // STOP even alongside functionCall parts, and the tool
            // identity then rides entirely on the lane. Accept that arm
            // only with a named lane and a parseable terminal; a tool
            // turn whose decision is nowhere is still a violation.
            let decision_on_lane = !lanes.is_empty()
                && lanes
                    .iter()
                    .any(|(_, name)| name.as_deref() == Some("get_weather"))
                && matches!(mapped, Some(Some(_)));
            if !decision_on_lane {
                report.violations.push((
                    "stop_reason_routing",
                    format!(
                        "a tool turn's terminal stop reason {reason:?} does not map to \
                         StreamStopReason::ToolCall and no named tool lane carries the \
                         decision — routing lost"
                    ),
                ));
            }
        }
    } else {
        if !lanes.is_empty() {
            report.violations.push((
                "stop_reason_routing",
                format!(
                    "a text-only turn opened tool lanes {lanes:?} — text must never \
                     route to the tool class"
                ),
            ));
        }
        match mapped {
            None => report.violations.push((
                "stop_reason_routing",
                "a success stream's terminal MessageDelta reported no stop reason — \
                 every provider's terminal must land in the routing table"
                    .to_string(),
            )),
            Some(Some(loopctl::stream::StreamStopReason::ToolCall)) => report.violations.push((
                "stop_reason_routing",
                format!("a text-only turn's stop reason {reason:?} routed to ToolCall"),
            )),
            Some(None) => report.violations.push((
                "stop_reason_routing",
                format!(
                    "stop reason {reason:?} is not in the routing table (from_api_str \
                     maps it to None) — the engine cannot decide from it"
                ),
            )),
            Some(_) => {}
        }
    }
    report
}

/// `error_taxonomy`: the recorded status must classify per the table —
/// 401/403 Auth; 429/503/529 RateLimit carrying the recorded
/// Retry-After; everything else typed Http with the status. Success
/// fixtures must produce no errors at all.
///
/// A success fixture cannot decide the status arms no matter how the
/// client behaves, so it reports the family as not applicable alongside
/// whatever it did decide: a vendor with no recorded failure reads `n/r`
/// here instead of collecting a pass from every 200 in the corpus.
pub fn check_error_taxonomy(
    output: &RunOutput,
    interactions: &[cassette::Interaction],
) -> BatteryReport {
    let mut report = BatteryReport::default();
    let Some(status) = fixture_status(interactions) else {
        report.not_applicable.push((
            "error_taxonomy",
            "the fixture records no response status — there is nothing to classify".to_string(),
        ));
        return report;
    };
    if status < 400 {
        if let Some(first) = output.errors.first() {
            report.violations.push((
                "error_taxonomy",
                format!("a 200 fixture yielded the typed error {first:?}"),
            ));
        }
        report.not_applicable.push((
            "error_taxonomy",
            "a success fixture carries no failure to classify — the 401/403 \u{2192} Auth and \
             429/503/529 \u{2192} RateLimit arms need a recorded error fixture for this vendor"
                .to_string(),
        ));
        return report;
    }
    let Some(error) = output.errors.first() else {
        report.violations.push((
            "error_taxonomy",
            format!(
                "a {status} fixture yielded no typed error — the failure is \
                 unclassifiable downstream"
            ),
        ));
        return report;
    };
    match status {
        401 | 403 => {
            if !matches!(error, ApiError::Auth(..)) {
                report.violations.push((
                    "error_taxonomy",
                    format!("{status} must classify as Auth, got {error:?}"),
                ));
            }
        }
        429 | 503 | 529 => match error {
            ApiError::RateLimit { retry_after, .. } => {
                let recorded = recorded_retry_after(interactions);
                if *retry_after != recorded {
                    report.violations.push((
                        "error_taxonomy",
                        format!(
                            "recorded Retry-After {recorded:?} but the error carries \
                             {retry_after:?} — the 0.3.0 rule carries exactly what the \
                             response advised: dropping an advised delay and inventing \
                             one an endpoint never sent are the same lie in opposite \
                             directions"
                        ),
                    ));
                }
            }
            other => report.violations.push((
                "error_taxonomy",
                format!("{status} must classify as RateLimit, got {other:?}"),
            )),
        },
        other => {
            if !matches!(error, ApiError::Http(..) | ApiError::Api(..)) {
                report.violations.push((
                    "error_taxonomy",
                    format!("{other} must classify as typed Http/Api, got {error:?}"),
                ));
            }
        }
    }
    report
}

/// `options_honoring`: the override is on the recorded bytes, the usage
/// flag the scenario asked for is the flag the wire carried, and the
/// capability flag tells the truth.
///
/// Every fixture carries the capability probe and (where the dialect
/// spells the usage ask on the wire) the flag cross-check, so this family
/// renders a verdict on every scenario rather than reporting a gap.
pub fn check_options_honoring(
    client: &dyn ApiClient,
    interactions: &[cassette::Interaction],
    scenario: &cassette::Scenario,
    profile: &ProviderProfile,
) -> BatteryReport {
    let mut violations = Vec::new();

    if scenario.name == "model_override" {
        let recorded = interactions
            .first()
            .and_then(|i| i.when.body.as_ref())
            .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
            .and_then(|body| {
                body.get("model")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            });
        match recorded.as_deref() {
            None => violations.push((
                "options_honoring",
                "the override fixture carries no parseable model in the request body".to_string(),
            )),
            Some(model) => {
                if model != scenario.model {
                    violations.push((
                        "options_honoring",
                        format!(
                            "the recorded wire model {model:?} is not the override \
                             {:?} — the override never reached the bytes",
                            scenario.model
                        ),
                    ));
                }
                if model == profile.default_model {
                    violations.push((
                        "options_honoring",
                        format!(
                            "the override scenario records the client default {model:?} \
                             — such a fixture proves nothing"
                        ),
                    ));
                }
            }
        }
    }

    if let Some(body) = interactions
        .first()
        .and_then(|i| i.when.body.as_ref())
        .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
    {
        // The OpenAI-family dialects express the streamed-usage ask on
        // the wire as stream_options.include_usage; the recorded flag
        // must agree with the scenario table (byte matching enforces it
        // at replay — this check names the mismatch when a fixture and
        // the table were edited out of sync).
        if let Some(include) = body
            .get("stream_options")
            .and_then(|options| options.get("include_usage"))
            .and_then(serde_json::Value::as_bool)
            && include != scenario.stream_usage
        {
            violations.push((
                "options_honoring",
                format!(
                    "the recorded request asked include_usage={include} but the \
                     scenario table says {}",
                    scenario.stream_usage
                ),
            ));
        }
    }

    if client.supports_tool_constraints() != profile.expects_tool_constraints {
        violations.push((
            "options_honoring",
            format!(
                "supports_tool_constraints() says {} but the profile (matching the \
                 wire encoder) expects {} — a dishonest capability probe lets the \
                 engine attach constraints the encoder silently drops",
                client.supports_tool_constraints(),
                profile.expects_tool_constraints
            ),
        ));
    }
    BatteryReport {
        violations,
        fixture_gaps: Vec::new(),
        not_applicable: Vec::new(),
    }
}

/// One cell of the matrix: the verdict one (scenario, family) pair
/// reached, or the reason none was reachable.
///
/// The states are deliberately asymmetric: only [`CellStatus::Pass`]
/// required a check to actually adjudicate a fixture, so the rendered
/// table cannot go green by silence — the failure mode every gap channel
/// in this module exists to prevent.
#[derive(Debug, Clone)]
pub enum CellStatus {
    /// A check adjudicated this fixture against the family's rules and
    /// found nothing broken.
    ///
    /// The only state that renders `pass` in the matrix — reachable
    /// exclusively when a check chose against a fixture it could judge.
    Pass,

    /// Violations, rendered.
    ///
    /// Carries every [`Violation`] the family named for this scenario;
    /// the cell quotes the first, and [`Matrix::violations`] collects
    /// the full list for the CI gate.
    Fail(Vec<Violation>),

    /// Nothing was decided — no cassette, a fixture cut below its
    /// terminal, or a fixture whose shape cannot test this family.
    ///
    /// The reason string says which of the three it was and what would
    /// close it. A visible gap, never a silent green.
    NotRun(String),
}

/// One (provider, scenario) run's outcome with per-family detail.
///
/// The grain the [`Matrix`] aggregates twice over: family cells roll up
/// into the scenario-level [`RunReport::status`], and every scenario a
/// provider ran rolls up into that provider's column verdict.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// The profile that ran.
    ///
    /// The matrix column this report contributes to.
    pub provider: String,

    /// The scenario replayed.
    ///
    /// Names the cassette; a gap reason quotes it, so the recording to
    /// take is identifiable from the report alone.
    pub scenario: String,

    /// Per-family outcomes, in [`FAMILIES`] order.
    ///
    /// A family the fixture's shape could not test is `NotRun` here even
    /// when the scenario as a whole passed — the per-family grain is
    /// where coverage gaps live and stay visible.
    pub families: Vec<CellStatus>,

    /// The scenario-level roll-up of the family cells.
    ///
    /// `Fail` on any violation, `NotRun` on a fixture-integrity gap or a
    /// missing cassette, `Pass` otherwise. Coverage gaps stay at family
    /// level — a scenario that broke nothing does not become `n/r`
    /// because one family had nothing to chew on.
    pub status: CellStatus,
}

/// Run one scenario of one profile: replay, drive, assert the battery.
///
/// `Err` when the cassette is not on disk (a gap, never silent).
/// Replay's completeness assert (`session.finish`) is part of the run:
/// unconsumed or double-hit interactions panic — the outbound-drift
/// guard doing its job.
pub async fn run_scenario(
    profile: &ProviderProfile,
    scenario: &cassette::Scenario,
) -> Result<RunReport, String> {
    if !cassette::cassette_path(profile.provider, scenario.name).exists() {
        return Err("no recorded cassette for this scenario yet".to_string());
    }
    let server = httpmock::MockServer::start_async().await;
    let session =
        cassette::CassetteSession::start(profile.provider, scenario.name, "", &server).await;
    let bundle = (profile.build)(
        &cassette::client_base_url(profile.provider, &session.base_url()),
        scenario,
    );
    let client: &dyn ApiClient = bundle.client.as_ref();

    let mut request =
        loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(scenario.prompt)]);
    if scenario.tools {
        request = request.with_tools(Some(vec![cassette::get_weather_tool()]));
    }
    let interactions = cassette::load_interactions(profile.provider, scenario.name);

    let stream = if scenario.name == "model_override" {
        let options = loopctl::structured::RequestOptions::new().with_model(scenario.model);
        client.stream_messages_with_options(&request, options)
    } else {
        client.stream_messages(&request)
    };
    let mut stream = std::pin::pin!(stream);
    let mut output = RunOutput {
        events: Vec::new(),
        errors: Vec::new(),
    };
    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => output.events.push(event),
            Err(error) => output.errors.push(error),
        }
    }
    session.finish().await;

    let outcome = battery_findings(client, &output, scenario, profile, &interactions);
    let violations: Vec<Violation> = outcome
        .violations
        .into_iter()
        .map(|(family, detail)| Violation {
            provider: profile.provider.to_string(),
            scenario: scenario.name.to_string(),
            family,
            detail,
        })
        .collect();
    let gaps: Vec<Violation> = outcome
        .fixture_gaps
        .into_iter()
        .map(|(family, detail)| Violation {
            provider: profile.provider.to_string(),
            scenario: scenario.name.to_string(),
            family,
            detail,
        })
        .collect();
    let inapplicable: Vec<Violation> = outcome
        .not_applicable
        .into_iter()
        .map(|(family, detail)| Violation {
            provider: profile.provider.to_string(),
            scenario: scenario.name.to_string(),
            family,
            detail,
        })
        .collect();
    let joined = |list: &Vec<Violation>| {
        list.iter()
            .map(|violation| violation.detail.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    };
    let families = FAMILIES
        .iter()
        .map(|family| {
            let family_violations: Vec<Violation> = violations
                .iter()
                .filter(|violation| violation.family == *family)
                .cloned()
                .collect();
            let family_gaps: Vec<Violation> = gaps
                .iter()
                .filter(|gap| gap.family == *family)
                .cloned()
                .collect();
            let family_na: Vec<Violation> = inapplicable
                .iter()
                .filter(|gap| gap.family == *family)
                .cloned()
                .collect();
            if !family_violations.is_empty() {
                CellStatus::Fail(family_violations)
            } else if !family_gaps.is_empty() {
                CellStatus::NotRun(joined(&family_gaps))
            } else if !family_na.is_empty() {
                CellStatus::NotRun(joined(&family_na))
            } else {
                CellStatus::Pass
            }
        })
        .collect();
    let status = if !violations.is_empty() {
        CellStatus::Fail(violations)
    } else if !gaps.is_empty() {
        CellStatus::NotRun(joined(&gaps))
    } else {
        CellStatus::Pass
    };
    Ok(RunReport {
        provider: profile.provider.to_string(),
        scenario: scenario.name.to_string(),
        families,
        status,
    })
}

/// The whole battery, one call: every family check over one run.
///
/// Shared by the disk-backed `run_scenario` and the meta-pin's
/// mutation runs, so the planted-violation proof exercises exactly the
/// checks CI runs — nothing weaker, nothing extra. Each check owns its
/// own not-decidable arms; this function only folds the outcomes
/// together, so no family can be silenced by the roll-up.
pub fn battery_findings(
    client: &dyn ApiClient,
    output: &RunOutput,
    scenario: &cassette::Scenario,
    profile: &ProviderProfile,
    interactions: &[cassette::Interaction],
) -> BatteryReport {
    let mut report = check_stream_lifecycle(output, interactions);
    report.merge(check_usage_reporting(output, interactions));
    report.merge(check_stop_reason_routing(output, scenario, interactions));
    report.merge(check_error_taxonomy(output, interactions));
    report.merge(check_options_honoring(
        client,
        interactions,
        scenario,
        profile,
    ));
    report
}

/// Drive a whole profile: every scenario in the registry for it.
///
/// A scenario whose cassette is not on disk still lands in the output —
/// as an all-`NotRun` report carrying the missing-cassette reason — so
/// the column shows what is absent instead of going quiet about it.
pub async fn run_provider(profile: &ProviderProfile) -> Vec<RunReport> {
    let mut reports = Vec::new();
    for scenario in cassette::scenarios()
        .into_iter()
        .filter(|s| s.provider == profile.provider)
    {
        match run_scenario(profile, &scenario).await {
            Ok(report) => reports.push(report),
            Err(reason) => reports.push(RunReport {
                provider: profile.provider.to_string(),
                scenario: scenario.name.to_string(),
                families: FAMILIES
                    .iter()
                    .map(|_| CellStatus::NotRun(reason.clone()))
                    .collect(),
                status: CellStatus::NotRun(reason),
            }),
        }
    }
    reports
}

/// The matrix: providers × contract families, generated from runs.
///
/// Assembled fresh on every artifact run, so the published table is
/// derived from what the battery decided this second — the drift a
/// hand-maintained coverage table invites is structurally unavailable.
pub struct Matrix {
    /// Column headers, provider display order.
    ///
    /// The providers that actually ran; providers still awaiting their
    /// cassettes ride [`Matrix::pending`] so their gaps render too.
    pub providers: Vec<String>,

    /// One report per (provider, scenario) run, in column order.
    ///
    /// The raw grain: [`Matrix::family_verdict`] rolls these up per
    /// column and [`Matrix::violations`] flattens their failures.
    pub reports: Vec<RunReport>,

    /// Providers listed without any run (e.g. awaiting their cassettes),
    /// each with the reason a cell is not yet runnable.
    ///
    /// Rendered as full gap rows: a pending provider is a visible hole
    /// in every column, not a missing row.
    pub pending: Vec<(String, String)>,
}

/// The first violation a provider's column recorded for one family.
///
/// The detail that makes a rendered `FAIL` actionable on its own line,
/// scenario and promise included, without the reader cross-referencing
/// the scenario roll-up below the table.
fn failure_reason(provider: &str, family: usize, reports: &[RunReport]) -> String {
    for report in reports.iter().filter(|report| report.provider == provider) {
        let Some(CellStatus::Fail(violations)) = report.families.get(family) else {
            continue;
        };
        let Some(violation) = violations.first() else {
            continue;
        };
        return format!(
            "{} in {}: {}",
            violation.family, report.scenario, violation.detail
        );
    }
    "a scenario named a violation for this family".to_string()
}

/// How one provider's whole corpus settled a single contract family.
///
/// The column-level roll-up the matrix renders: the three states keep a
/// failure, an adjudicated pass, and an honest "nothing could decide"
/// distinct, so the table never blurs what broke with what was never
/// tested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FamilyVerdict {
    /// Some scenario named a violation.
    ///
    /// The verdict that outranks all evidence: one adjudicated fixture
    /// that broke a rule beats any number of fixtures that had nothing
    /// to say about the family.
    Failed,

    /// Some check adjudicated a fixture and found nothing broken — the
    /// only state that earns a `pass` in the table.
    ///
    /// "Adjudicated" is the load-bearing word: a check that ran over
    /// fixtures whose shape it could not judge contributes nothing here,
    /// which is what keeps a lucky silence from reading as green.
    Exercised,

    /// No check could adjudicate any of this provider's fixtures: no
    /// cassette, a recording cut below its terminal, or a corpus with
    /// nothing of the shape the family tests.
    ///
    /// The carried reason is the cell's text, naming what to record to
    /// close the gap — the rendered table is a to-do list as much as a
    /// scoreboard.
    Unproven(String),
}

impl Matrix {
    /// The verdict across every scenario this provider ran for one
    /// [`FAMILIES`] column.
    ///
    /// Failed outranks everything (one violation is a finding), then
    /// Exercised: a family counts as proven only where a check actually
    /// adjudicated a fixture, never because some other fixture happened
    /// to produce nothing to complain about.
    pub fn family_verdict(&self, provider: &str, family: usize) -> FamilyVerdict {
        let mut exercised = false;
        let mut reason: Option<String> = None;
        for report in self
            .reports
            .iter()
            .filter(|report| report.provider == provider)
        {
            match report.families.get(family) {
                Some(CellStatus::Fail(_)) => return FamilyVerdict::Failed,
                Some(CellStatus::Pass) => exercised = true,
                Some(CellStatus::NotRun(detail)) => {
                    reason = reason.or_else(|| Some(detail.clone()));
                }
                None => {}
            }
        }
        if exercised {
            return FamilyVerdict::Exercised;
        }
        FamilyVerdict::Unproven(
            reason.unwrap_or_else(|| "no scenario of this provider ran this family".to_string()),
        )
    }

    /// Render the matrix as markdown — the CI-visible table of which
    /// providers are proven.
    ///
    /// Generated, never hand-edited: every cell comes from
    /// [`Matrix::family_verdict`] over the runs and every gap line from
    /// the reports that carried it, so the published artifact cannot
    /// claim more — or less — than the battery actually decided.
    pub fn render_markdown(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("# Provider conformance matrix".to_string());
        lines.push(String::new());
        lines.push(
            "Generated by `cargo test --features … --test provider_conformance \
             conformance_matrix_artifact -- --exact --nocapture` — do not edit by hand."
                .to_string(),
        );
        lines.push(String::new());
        lines.push(format!("| provider | {} |", FAMILIES.join(" | ")));
        lines.push(format!(
            "|---|{}|",
            FAMILIES.iter().map(|_| "---").collect::<Vec<_>>().join("|")
        ));
        // A column is only meaningful next to its definition: `pass`
        // means "a fixture of this provider adjudicated this family",
        // which is exactly what a silent check used to fake.
        let mut legend = String::from("| | *family exercised by at least one fixture* |");
        for _ in 1..FAMILIES.len() {
            legend.push_str(" |");
        }
        lines.push(legend);
        for provider in &self.providers {
            let mut row = format!("| {provider} ");
            for (index, _family) in FAMILIES.iter().enumerate() {
                let (marker, reason) = match self.family_verdict(provider, index) {
                    FamilyVerdict::Failed => {
                        ("FAIL", Some(failure_reason(provider, index, &self.reports)))
                    }
                    FamilyVerdict::Exercised => ("pass", None),
                    FamilyVerdict::Unproven(text) => ("n/r", Some(text)),
                };
                row.push_str(&match reason {
                    Some(detail) => format!("| {marker} — {detail} "),
                    None => format!("| {marker} "),
                });
            }
            row.push('|');
            lines.push(row);
        }
        for (provider, reason) in &self.pending {
            let cell = format!("n/r — {reason}");
            lines.push(format!(
                "| {provider} | {} |",
                vec![cell.as_str(); FAMILIES.len()].join(" | ")
            ));
        }
        lines.push(String::new());
        let mut gaps = 0usize;
        for report in &self.reports {
            if let CellStatus::NotRun(reason) = &report.status {
                lines.push(format!(
                    "- gap: {}/{} — {reason}",
                    report.provider, report.scenario
                ));
                gaps += 1;
            }
        }
        for (provider, reason) in &self.pending {
            lines.push(format!("- gap: {provider} — {reason}"));
            gaps += 1;
        }
        // A family the whole corpus could not adjudicate is a coverage
        // hole: named per provider and family, so what is unproven is
        // readable from the artifact instead of inferred from a `pass`
        // that only ever saw one shape of fixture.
        for provider in &self.providers {
            for (index, family) in FAMILIES.iter().enumerate() {
                if let FamilyVerdict::Unproven(reason) = self.family_verdict(provider, index) {
                    lines.push(format!("- coverage: {provider}/{family} — {reason}"));
                    gaps += 1;
                }
            }
        }
        if gaps > 0 {
            lines.push(String::new());
            lines.push(format!(
                "{gaps} cells await a fixture — recorded, they turn to \
                 pass/fail automatically; gaps are visible, never silently green."
            ));
        }
        let mut out = lines.join("\n");
        out.push('\n');
        out
    }

    /// Every violation from any run — CI fails on non-empty.
    ///
    /// Read from the scenario-level [`RunReport::status`] roll-up, so one
    /// violation counts once however many families caught the same slip.
    /// The table may legitimately show `n/r` cells while the corpus
    /// grows; it may never show a rule broken.
    pub fn violations(&self) -> Vec<&Violation> {
        self.reports
            .iter()
            .filter_map(|report| match &report.status {
                CellStatus::Fail(violations) => Some(violations.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }
}
