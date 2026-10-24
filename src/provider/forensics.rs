//! Provider response-header forensics — request ids and rate-limit state,
//! read once per response at HTTP time.
//!
//! The two things a provider-support investigation asks for are the
//! **request id** and the **rate-limit state** ("what was the remaining
//! quota at that moment?"), and both ride the very headers the status
//! line arrives with. The rules this module holds to:
//!
//! - **Capture at HTTP time, in one place.** [`capture_provider_meta`]
//!   reads a [`HeaderMap`] once; every provider inherits the capture by
//!   going through [`post_json_checked`](super::post_json_checked). The
//!   data is never reconstructed from an error string afterwards.
//! - **Verbatim values, bounded.** Ids and quota strings are stored as
//!   the provider spelled them (units differ per provider — see
//!   [`RateLimitSnapshot`]), each cut at a documented character bound on
//!   a `char` boundary so a hostile header cannot inflate an error text.
//! - **Advisory success-path state.** The most recent response's snapshot
//!   lives in a per-client [`RateLimitSlot`] behind a Category-1 lock:
//!   poison recovers, the last snapshot survives. `None` means
//!   "unreported", never "zero" — the
//!   [`Usage`](crate::stream::Usage) discipline.
//! - **Pure observability.** Nothing here feeds
//!   [`is_retryable`](crate::api::error::ApiError::is_retryable), the
//!   retry ladder, or classification. Failures behave exactly as if the
//!   headers had never been read.

use std::sync::{Arc, Mutex};

use reqwest::header::{HeaderMap, HeaderName};

use crate::api::error::{
    MAX_RATE_LIMIT_VALUE_CHARS, MAX_REQUEST_ID_CHARS, ProviderMeta, RATE_LIMIT_LIMIT_HEADERS,
    RATE_LIMIT_REMAINING_HEADERS, RATE_LIMIT_RESET_HEADERS, REQUEST_ID_HEADERS, RateLimitSnapshot,
    bounded_header_value,
};

/// The request id from a response's headers, or `None`.
///
/// First present, valid-`String`, non-empty header of
/// [`REQUEST_ID_HEADERS`] wins, in that order. `to_str()` failing (a
/// non-UTF-8 id from a broken proxy) skips that candidate rather than
/// losing the whole lookup — the next header is still a real id.
pub(super) fn capture_request_id(headers: &HeaderMap) -> Option<String> {
    first_header(headers, REQUEST_ID_HEADERS, MAX_REQUEST_ID_CHARS)
}

/// The `x-ratelimit-*` / `rate-limit-*` family from a response's headers,
/// captured verbatim.
///
/// `None` when no family member is present — capture never stores an
/// all-`None` snapshot, so `Some(_)` always carries at least one string.
pub(super) fn capture_rate_limit(headers: &HeaderMap) -> Option<RateLimitSnapshot> {
    let snapshot = RateLimitSnapshot {
        limit: first_header(
            headers,
            RATE_LIMIT_LIMIT_HEADERS,
            MAX_RATE_LIMIT_VALUE_CHARS,
        ),
        remaining: first_header(
            headers,
            RATE_LIMIT_REMAINING_HEADERS,
            MAX_RATE_LIMIT_VALUE_CHARS,
        ),
        reset: first_header(
            headers,
            RATE_LIMIT_RESET_HEADERS,
            MAX_RATE_LIMIT_VALUE_CHARS,
        ),
    };
    (!snapshot.is_empty()).then_some(snapshot)
}

/// Both captures in one pass over the header names.
///
/// Returns an all-`None` [`ProviderMeta`] when the response carried
/// neither an id nor the rate-limit family — callers that only store
/// non-empty meta can gate on [`ProviderMeta::is_empty`].
///
/// Streaming call sites capture on the open response **before** handing it
/// to the body reader: the reader consumes the response object, and the
/// open response is the last point its headers are readable — that
/// ordering is the whole forensics contract for mid-stream failures.
pub(super) fn capture_provider_meta(headers: &HeaderMap) -> ProviderMeta {
    ProviderMeta {
        request_id: capture_request_id(headers),
        rate_limit: capture_rate_limit(headers),
    }
}

/// First present, valid, non-empty value among `names`, bounded to `max`.
///
/// A candidate whose header name fails to parse, whose value is not
/// visible ASCII, or which trims to empty is skipped rather than fatal —
/// the next name in the list still gets its chance at the lookup.
fn first_header(headers: &HeaderMap, names: &[&str], max: usize) -> Option<String> {
    names.iter().find_map(|name| {
        HeaderName::from_bytes(name.as_bytes())
            .ok()
            .as_ref()
            .and_then(|name| headers.get(name))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| bounded_header_value(value, max))
    })
}

/// The failure class a wire-failure telemetry event carries.
///
/// The telemetry classes: `auth` (401/403),
/// `rate_limit` (429/503/529 — the throttle-shaped family this crate
/// already classifies as rate-limited), `server` (any other 5xx),
/// `network` (transport failure — no response, hence no status), and
/// `other` (everything else). Pure label; the classification that drives
/// retries is the error variant itself, untouched.
pub(super) fn wire_failure_class(status: u16) -> &'static str {
    match status {
        401 | 403 => "auth",
        429 | 503 | 529 => "rate_limit",
        500..=599 => "server",
        _ => "other",
    }
}

/// One log line for one classified wire failure, fields attached.
///
/// Permanent failures (`auth`, `other`) log at `WARN` — nobody retries
/// those, the line is the whole report. Retryable shapes (rate-limit,
/// server) log at `DEBUG`: the stream handler narrates its own retries
/// at its level, and the provider layer must not spam a ladder run that
/// recovers. The request id and the verbatim quota family ride the
/// event when the provider sent them — the family on a failure is
/// "what was my remaining quota at that moment" answering itself.
pub(super) fn log_wire_failure(status: u16, meta: &ProviderMeta) {
    let class = wire_failure_class(status);
    let empty = RateLimitSnapshot::default();
    let quota = meta.rate_limit.as_ref().unwrap_or(&empty);
    match class {
        "auth" | "other" => {
            tracing::warn!(
                target: "loopctl::provider::forensics",
                status,
                class,
                request_id = meta.request_id.as_deref().unwrap_or(""),
                rate_limit_limit = quota.limit.as_deref().unwrap_or(""),
                rate_limit_remaining = quota.remaining.as_deref().unwrap_or(""),
                rate_limit_reset = quota.reset.as_deref().unwrap_or(""),
                "provider wire failure"
            );
        }
        _ => {
            tracing::debug!(
                target: "loopctl::provider::forensics",
                status,
                class,
                request_id = meta.request_id.as_deref().unwrap_or(""),
                rate_limit_limit = quota.limit.as_deref().unwrap_or(""),
                rate_limit_remaining = quota.remaining.as_deref().unwrap_or(""),
                rate_limit_reset = quota.reset.as_deref().unwrap_or(""),
                "provider wire failure"
            );
        }
    }
}

/// One line for one transport failure (no response, so no id).
///
/// The `network` class of the failure taxonomy: there is no status line
/// and no header to carry, only the transport's own message. `DEBUG` — a
/// connect failure is usually transient and the retry narrates it at the
/// caller's level.
pub(super) fn log_transport_failure(error: &str) {
    tracing::debug!(
        target: "loopctl::provider::forensics",
        class = "network",
        error,
        "provider transport failure"
    );
}

/// The numeric arms of the low-water test: `(remaining, limit)` when both
/// the snapshot's `remaining` and `limit` parse as `u64`, `limit > 0`, and
/// `remaining` is strictly below one tenth of `limit`.
///
/// This is the **only** place the verbatim strings are interpreted, and
/// the interpretation feeds nothing but the event below — a snapshot that
/// never parses still preserves every header exactly.
fn below_low_water(snapshot: &RateLimitSnapshot) -> Option<(u64, u64)> {
    let limit = snapshot.limit.as_deref()?.parse::<u64>().ok()?;
    let remaining = snapshot.remaining.as_deref()?.parse::<u64>().ok()?;
    if limit == 0 {
        return None;
    }
    (remaining.saturating_mul(10) < limit).then_some((remaining, limit))
}

/// The per-client advisory rate-limit state (Category-1 lock).
///
/// Holds the snapshot from the most recent response and one latch bit
/// for the low-water event. Cloning the slot shares the state — a client
/// hands a clone into each streaming task so a stream's first response
/// updates the accessor the host reads on the client itself. Concurrent
/// writers are therefore the designed case, not an accident of a
/// multi-stream host.
///
/// Lock discipline (Category 1, advisory read, last writer wins): one
/// acquisition swaps one response's snapshot whole, so streams sharing a
/// client serialize per response and every reader sees the most recently
/// recorded one — never one response's `remaining` beside another's
/// `limit`. Every access goes through
/// [`recover_guard`](crate::error::recover_guard), so a panic while
/// holding the lock poisons it without ever taking the snapshot down —
/// the accessor keeps reporting the last value.
#[derive(Clone, Default)]
pub(crate) struct RateLimitSlot {
    inner: Arc<Mutex<SlotState>>,
}

/// The state behind a [`RateLimitSlot`].
///
/// The snapshot and the low-water latch bit move together under the
/// slot's single lock acquisition — the check-and-set that keeps the
/// event to one per window cannot interleave with the snapshot update.
#[derive(Default)]
struct SlotState {
    /// The most recent response's snapshot; `None` = unreported.
    ///
    /// Replaced wholesale per response (a silent response clears it), so
    /// the accessor answers "what did the last response say", never
    /// "what was said the last time anything spoke".
    snapshot: Option<RateLimitSnapshot>,

    /// The low-water event has been emitted for the current window and
    /// has not yet been rearmed (a later response above the line, or
    /// with no parseable numbers, rearms it).
    warned: bool,
}

impl RateLimitSlot {
    /// Record one response: replace the snapshot (headers absent ⇒ `None`
    /// — the accessor is per-response, not stickiest-ever), and emit the
    /// bounded low-water event when the numbers say the window is nearly
    /// gone.
    ///
    /// Called for **every** response — success and failure alike: a 429's
    /// `x-ratelimit-remaining: 0` is exactly the state the accessor and
    /// the event exist to preserve, and the 200-with-3-remaining is the
    /// interesting neighbor of the 429. Every response that carries the
    /// family therefore narrates its state once at `DEBUG`, and one below
    /// the low-water line additionally **warns** at `WARN` — the visible
    /// level of the documented "warns once per window" promise, hosts
    /// filtering at `WARN` or above included; the latch bounds it, so it
    /// cannot flood. A disabled-interest subscriber pays a callsite
    /// check, nothing more.
    pub(crate) fn record(&self, headers: &HeaderMap) {
        let snapshot = capture_rate_limit(headers);
        if let Some(seen) = snapshot.as_ref() {
            tracing::debug!(
                target: "loopctl::provider::forensics",
                rate_limit_limit = seen.limit.as_deref().unwrap_or(""),
                rate_limit_remaining = seen.remaining.as_deref().unwrap_or(""),
                rate_limit_reset = seen.reset.as_deref().unwrap_or(""),
                "provider rate-limit state"
            );
        }
        let mut state = crate::error::recover_guard(self.inner.lock());
        match snapshot.as_ref().and_then(below_low_water) {
            Some((remaining, limit)) => {
                if !state.warned {
                    tracing::warn!(
                        target: "loopctl::provider::forensics",
                        remaining,
                        limit,
                        reset = snapshot
                            .as_ref()
                            .and_then(|s| s.reset.as_deref())
                            .unwrap_or(""),
                        "provider rate-limit window below 10% remaining"
                    );
                    state.warned = true;
                }
            }
            None => state.warned = false,
        }
        state.snapshot = snapshot;
    }

    /// The most recent response's snapshot, if it reported one.
    ///
    /// `None` is "unreported" — never a measured zero (the `Usage`
    /// discipline). Poisoned-lock-safe: the last value survives.
    pub(crate) fn last(&self) -> Option<RateLimitSnapshot> {
        crate::error::recover_guard(self.inner.lock())
            .snapshot
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headers from raw name/value pairs (one-shot; invalid names drop).
    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
                value.parse().expect("valid header value"),
            );
        }
        map
    }

    #[test]
    fn request_id_follows_the_documented_priority() {
        // Every candidate present → the first in the list wins.
        let all = headers(&[
            ("cf-ray", "cf.1"),
            ("request-id", "anthropic.1"),
            ("x-ms-request-id", "azure.1"),
            ("apiproxy-id", "proxy.1"),
            ("openai-request-id", "openai.1"),
            ("x-request-id", "family.1"),
            ("x-amzn-requestid", "aws.1"),
        ]);
        assert_eq!(
            capture_request_id(&all).as_deref(),
            Some("aws.1"),
            "x-amzn-requestid leads the documented order"
        );
    }

    #[test]
    fn each_request_id_header_works_alone() {
        for name in REQUEST_ID_HEADERS {
            let h = headers(&[(name, "the-id")]);
            assert_eq!(
                capture_request_id(&h).as_deref(),
                Some("the-id"),
                "{name} must be readable on its own"
            );
        }
    }

    #[test]
    fn header_names_match_case_insensitively() {
        // HTTP headers are case-insensitive and reqwest lowercases its
        // map — a provider spelling `X-Amzn-RequestId` still lands.
        let h = headers(&[("X-Amzn-RequestId", "Mixed.1")]);
        assert_eq!(capture_request_id(&h).as_deref(), Some("Mixed.1"));
    }

    #[test]
    fn an_unusable_header_falls_through_to_the_next() {
        // Empty / whitespace-only first candidates are no report; the
        // next present header wins rather than the lookup failing.
        let h = headers(&[("x-request-id", "   "), ("openai-request-id", "later.1")]);
        assert_eq!(capture_request_id(&h).as_deref(), Some("later.1"));
    }

    #[test]
    fn a_non_ascii_id_is_skipped_not_garbled() {
        // HeaderValue::to_str fails on non-visible-ASCII; the candidate
        // drops and the family rule (never invent) leaves the next id.
        let mut map = HeaderMap::new();
        map.append(
            HeaderName::from_static("x-request-id"),
            reqwest::header::HeaderValue::from_bytes(b"\xff\xfe").expect("raw bytes accepted"),
        );
        map.append(
            HeaderName::from_static("cf-ray"),
            reqwest::header::HeaderValue::from_static("cf.2"),
        );
        assert_eq!(capture_request_id(&map).as_deref(), Some("cf.2"));
    }

    #[test]
    fn rate_limit_family_captures_verbatim_from_both_families() {
        let openai = headers(&[
            ("x-ratelimit-limit", "500"),
            ("x-ratelimit-remaining", "42"),
            ("x-ratelimit-reset", "1780000000"),
        ]);
        let snap = capture_rate_limit(&openai).expect("family present");
        assert_eq!(snap.limit.as_deref(), Some("500"));
        assert_eq!(snap.remaining.as_deref(), Some("42"));
        assert_eq!(snap.reset.as_deref(), Some("1780000000"));

        let anthropic = headers(&[
            ("rate-limit-limit", "50"),
            ("rate-limit-remaining", "3"),
            ("rate-limit-reset", "2026-08-30T12:00:00Z"),
        ]);
        let snap = capture_rate_limit(&anthropic).expect("anthropic family present");
        assert_eq!(snap.remaining.as_deref(), Some("3"));
        assert_eq!(snap.reset.as_deref(), Some("2026-08-30T12:00:00Z"));

        // Verbatim: an ISO date, a float, and a word are all preserved
        // exactly — nothing interprets them.
        let odd = headers(&[("x-ratelimit-reset", "about a minute")]);
        assert_eq!(
            capture_rate_limit(&odd).expect("member").reset.as_deref(),
            Some("about a minute")
        );
    }

    #[test]
    fn no_family_is_none_not_an_empty_snapshot() {
        let none = headers(&[("content-type", "application/json")]);
        assert!(capture_rate_limit(&none).is_none());
        let meta = capture_provider_meta(&none);
        assert!(meta.is_empty());
    }

    #[test]
    fn oversized_header_values_are_cut_not_stored() {
        let h = headers(&[("x-request-id", &"x".repeat(4000))]);
        let id = capture_request_id(&h).expect("present");
        assert_eq!(id.chars().count(), MAX_REQUEST_ID_CHARS);
        let h = headers(&[("x-ratelimit-limit", &"9".repeat(400))]);
        assert_eq!(
            capture_rate_limit(&h)
                .expect("present")
                .limit
                .expect("kept")
                .chars()
                .count(),
            MAX_RATE_LIMIT_VALUE_CHARS
        );
    }

    #[test]
    fn wire_failure_classes_match_the_documented_table() {
        assert_eq!(wire_failure_class(401), "auth");
        assert_eq!(wire_failure_class(403), "auth");
        assert_eq!(wire_failure_class(429), "rate_limit");
        assert_eq!(wire_failure_class(503), "rate_limit");
        assert_eq!(wire_failure_class(529), "rate_limit");
        assert_eq!(wire_failure_class(500), "server");
        assert_eq!(wire_failure_class(502), "server");
        assert_eq!(wire_failure_class(400), "other");
        assert_eq!(wire_failure_class(404), "other");
    }

    #[test]
    fn low_water_arms() {
        let snap = |limit: &str, remaining: &str| RateLimitSnapshot {
            limit: Some(limit.into()),
            remaining: Some(remaining.into()),
            reset: None,
        };
        // Strictly below one tenth triggers; exactly one tenth does not.
        assert_eq!(below_low_water(&snap("100", "9")), Some((9, 100)));
        assert_eq!(below_low_water(&snap("100", "10")), None);
        assert_eq!(below_low_water(&snap("100", "50")), None);
        // Zero limit is no event (and no division); unparseable or
        // missing members are simply "cannot tell".
        assert_eq!(below_low_water(&snap("0", "0")), None);
        assert_eq!(below_low_water(&snap("many", "1")), None);
        assert_eq!(
            below_low_water(&RateLimitSnapshot {
                remaining: Some("1".into()),
                ..Default::default()
            }),
            None
        );
        // Huge values saturate instead of overflowing.
        assert_eq!(
            below_low_water(&snap("18446744073709551615", "1")),
            Some((1, u64::MAX))
        );
    }

    #[test]
    fn slot_is_per_response_not_sticky() {
        let slot = RateLimitSlot::default();
        assert_eq!(slot.last(), None);
        slot.record(&headers(&[("x-ratelimit-remaining", "42")]));
        assert_eq!(
            slot.last().expect("reported").remaining.as_deref(),
            Some("42")
        );
        // A response without the family clears the snapshot: the accessor
        // is "most recent response", not "most recent response that spoke".
        slot.record(&headers(&[("content-type", "application/json")]));
        assert_eq!(slot.last(), None);
    }

    #[test]
    fn slot_records_from_failed_responses_too() {
        // The 429 case: throttle headers are the reason the accessor
        // exists, and they ride a failure.
        let slot = RateLimitSlot::default();
        slot.record(&headers(&[
            ("retry-after", "7"),
            ("x-ratelimit-remaining", "0"),
        ]));
        assert_eq!(
            slot.last()
                .expect("kept despite the failure")
                .remaining
                .as_deref(),
            Some("0")
        );
    }

    #[test]
    fn clones_share_one_slot_state() {
        let slot = RateLimitSlot::default();
        let clone = slot.clone();
        clone.record(&headers(&[("x-ratelimit-limit", "5")]));
        assert_eq!(slot.last().expect("shared").limit.as_deref(), Some("5"));
    }

    /// The multi-writer reality the type doc states: clones handed to
    /// concurrent streams contend, and a snapshot always arrives whole —
    /// a reader never sees one response's `remaining` under another's
    /// `limit`.
    #[test]
    fn concurrent_writers_leave_one_whole_snapshot() {
        let first = ("100", "97", "1780000000");
        let second = ("250", "240", "1780000900");
        let headers_for = |spec: (&str, &str, &str)| {
            headers(&[
                ("x-ratelimit-limit", spec.0),
                ("x-ratelimit-remaining", spec.1),
                ("x-ratelimit-reset", spec.2),
            ])
        };
        let slot = RateLimitSlot::default();
        // Both writers wait at one gate so they race into the lock rather
        // than running one after the other.
        let gate = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (spec, writer) in [(first, slot.clone()), (second, slot.clone())] {
                let ready = &gate;
                handles.push(scope.spawn(move || {
                    ready.wait();
                    writer.record(&headers_for(spec));
                }));
            }
            for handle in handles {
                handle.join().expect("a writer never panics");
            }
        });

        let snapshot = slot.last().expect("a writer recorded");
        let is_whole = |spec: (&str, &str, &str)| {
            snapshot.limit.as_deref() == Some(spec.0)
                && snapshot.remaining.as_deref() == Some(spec.1)
                && snapshot.reset.as_deref() == Some(spec.2)
        };
        assert!(
            is_whole(first) || is_whole(second),
            "one response's header family is swapped in one acquisition, so a \
             reader sees one response or the other — never a mixture: {snapshot:?}"
        );
    }

    #[test]
    fn a_panicked_holder_keeps_the_last_snapshot() {
        // Category-1 discipline: the holder panics while locked, the
        // mutex poisons, and the accessor still reports the last value.
        let slot = RateLimitSlot::default();
        slot.record(&headers(&[("x-ratelimit-remaining", "42")]));
        let doomed = slot.clone();
        let handle = std::thread::spawn(move || {
            let _guard = doomed.inner.lock().expect("fresh slot locks clean");
            panic!("holder panicked while holding the forensics lock");
        });
        let rejoined = handle.join();
        assert!(
            rejoined.is_err(),
            "the fixture thread must actually panic for the poison to exist"
        );
        assert_eq!(
            slot.last()
                .expect("survives the poison")
                .remaining
                .as_deref(),
            Some("42"),
            "a panicked holder must never take the snapshot down"
        );
    }

    #[test]
    fn low_water_event_latches_once_per_window() {
        // The event itself is pinned end-to-end by the capture-harness
        // binary `low_water_events.rs` (drives below-10% responses
        // through a real client and counts emitted events); these pins
        // own the latch state: crossing below latches, recovering
        // rearms, and a non-numeric response rearms too.
        let slot = RateLimitSlot::default();
        slot.record(&headers(&[
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-remaining", "5"),
        ]));
        assert!(
            slot.inner.lock().expect("clean").warned,
            "first crossing latches"
        );
        slot.record(&headers(&[
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-remaining", "4"),
        ]));
        // Staying below must never spuriously reset the latch; the
        // no-double-warn half of that promise is counted end to end by
        // the same integration pin.
        assert!(slot.inner.lock().expect("clean").warned);
        slot.record(&headers(&[
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-remaining", "80"),
        ]));
        assert!(!slot.inner.lock().expect("clean").warned, "recovery rearms");
        slot.record(&headers(&[]));
        assert!(!slot.inner.lock().expect("clean").warned, "silence rearms");
    }

    #[test]
    fn provider_meta_combines_both_captures() {
        let h = headers(&[("x-request-id", "r.1"), ("rate-limit-remaining", "7")]);
        let meta = capture_provider_meta(&h);
        assert_eq!(meta.request_id.as_deref(), Some("r.1"));
        assert_eq!(
            meta.rate_limit
                .as_ref()
                .expect("family")
                .remaining
                .as_deref(),
            Some("7")
        );
        assert!(!meta.is_empty());
    }
}
