//! Cassette record/replay harness for provider wire contracts.
//!
//! A cassette is a YAML file under `tests/cassettes/<provider>/` holding
//! real HTTP interactions: the exact request bytes a provider accepted
//! and the exact response it returned, recorded through a forwarding
//! proxy during a deliberate live session. Replay registers each
//! recorded interaction as a mock that matches method, path, query,
//! allowlisted headers, and **body bytes** — a request built
//! differently than recorded misses every mock and fails the test.
//! That miss is the outbound-drift guard: the entire point of the
//! harness.
//!
//! Recording is a deliberate human act: set `LOOPCTL_CASSETTE=record`
//! plus `LOOPCTL_E2E=1` (and real credentials for non-local providers)
//! and run the record driver in `tests/provider_e2e.rs`. Everything is
//! scrubbed before it touches disk — credentials never enter a file,
//! and generated ids under the known prefixes (`msg_`, `call_`,
//! `cmpl-`, `chatcmpl-`, `resp_`) become deterministic placeholders so
//! replay stays stable across recordings; provider values outside
//! those prefixes ride verbatim — responses are served, never
//! matched.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use std::path::{Path, PathBuf};

use httpmock::{Mock, MockServer, Recording};

/// The api-key value replay builds clients with.
///
/// Sensitive query parameters are scrubbed to this exact string, so a
/// replay client configured with it produces the recorded request
/// bytes. Request headers are never recorded or matched, so header
/// auth accepts any dummy value.
pub const CASSETTE_KEY: &str = "cassette-key";

/// Request header names that must never appear in a cassette.
///
/// These carry credentials verbatim, so the recording allowlist never
/// captures them and the scrubber removes any that slip through — the
/// safety scan then flags the file if one somehow survived both.
const FORBIDDEN_HEADERS: [&str; 4] = ["authorization", "x-api-key", "api-key", "cookie"];

/// Response headers preserved in a cassette.
///
/// `content-type` is load-bearing for replay (the SSE parser keys off
/// it) and `retry-after` feeds the retry machinery — 429 cassettes
/// must carry it. Every other response header is noise at best and a
/// leak at worst, so the scrubber drops it.
const RESPONSE_HEADER_ALLOWLIST: [&str; 2] = ["content-type", "retry-after"];

/// Query parameter names whose values are scrubbed to [`CASSETTE_KEY`].
///
/// Gemini-style APIs put the credential in the query string, so unlike
/// headers these are recorded (replay matches them) — the value is
/// rewritten to the fixed dummy, which replay clients also use.
const SENSITIVE_QUERY_PARAMS: [&str; 3] = ["key", "api_key", "token"];

/// Prefixes of provider-generated ids replaced with deterministic
/// placeholders, so replay matching stays stable across recordings.
const GENERATED_ID_PREFIXES: [&str; 5] = ["msg_", "call_", "cmpl-", "chatcmpl-", "resp_"];

/// Whether this session records or replays.
///
/// Selected by `LOOPCTL_CASSETTE` (`record` to record, anything else —
/// including unset — replays). Replay is the default because it is the
/// hermetic, credential-free mode CI runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CassetteMode {
    /// Serve recorded interactions; miss on any byte drift.
    ///
    /// The hermetic default CI runs: no network, dummy credentials,
    /// and every registered interaction must be consumed exactly once.
    Replay,

    /// Forward to the real provider and capture the exchange.
    ///
    /// Entered only through the deliberate-act environment; the
    /// captured interactions are scrubbed before anything is written.
    Record,
}

/// The recorded mode selected from the environment.
///
/// Unset or unknown values mean replay — recording cannot happen by
/// accident.
pub fn mode_from_env() -> CassetteMode {
    match std::env::var("LOOPCTL_CASSETTE").as_deref() {
        Ok("record") => CassetteMode::Record,
        _ => CassetteMode::Replay,
    }
}

/// One name/value pair as it appears in a cassette (headers, query).
///
/// The recording format serializes header and query entries as
/// `{name, value}` maps; this mirrors that shape so a cassette
/// round-trips through the harness unchanged.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pair {
    /// The pair's name, e.g. `content-type` or `key`.
    ///
    /// Serialized as a map key inside the cassette's YAML, matching
    /// the recording format's `{name, value}` shape.
    pub name: String,

    /// The pair's value exactly as recorded (post-scrub).
    ///
    /// Replay matches this verbatim, so any rewrite (the sensitive
    /// query scrub, an id rename) must be mirrored by the replay-side
    /// client configuration.
    pub value: String,
}

/// The recorded request side of one interaction.
///
/// Only the fields the harness records and matches; unknown fields in
/// a file are ignored on load and dropped on rewrite.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WhenSpec {
    /// The HTTP method, e.g. `POST`.
    ///
    /// Recorded verbatim from the accepted request and registered as
    /// an exact matcher on replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    /// The request path, e.g. `/v1/chat/completions`.
    ///
    /// Includes the provider's full path prefix — the forwarding proxy
    /// preserves it, so the cassette and the replay request agree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Recorded query parameters, matched exactly.
    ///
    /// Sensitive names keep their slot with the scrubbed dummy value,
    /// because replay clients send that same dummy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_param: Option<Vec<Pair>>,

    /// Recorded request headers — the match allowlist only.
    ///
    /// `content-type` and `accept` as captured by the recording rule;
    /// credential-bearing headers never enter this list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<Vec<Pair>>,

    /// The exact request body bytes, matched verbatim.
    ///
    /// This field is the outbound-drift guard: a client that builds
    /// one byte differently than the recorded request misses the mock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,

    /// Binary request bodies.
    ///
    /// Kept on round-trip but refused at replay: none of the supported
    /// providers send binary requests, and a binary matcher cannot be
    /// silently skipped without losing the drift guard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_base64: Option<String>,
}

/// The recorded response side of one interaction.
///
/// Everything here is served verbatim on replay after the scrub pass;
/// nothing in it participates in request matching.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThenSpec {
    /// The response status code.
    ///
    /// Served as-is on replay — including error statuses a scenario
    /// deliberately recorded (a 429 cassette must fail the request).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,

    /// The response headers after the allowlist pass.
    ///
    /// Only [`RESPONSE_HEADER_ALLOWLIST`] names survive the scrub, so
    /// nothing provider-specific leaks into the committed file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<Vec<Pair>>,

    /// The response body verbatim (ids already placeholdered).
    ///
    /// Not matched on replay — responses are served, requests are
    /// matched — so nondeterministic model text between recordings is
    /// fine; only the ids are renamed, for stable diffs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,

    /// Binary response bodies.
    ///
    /// Same refusal as [`WhenSpec::body_base64`]: every body these
    /// providers stream is UTF-8 JSON or SSE, and replay panics rather
    /// than serving nothing for a binary payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_base64: Option<String>,
}

/// One recorded request/response exchange.
///
/// The cassette file is a YAML sequence of these, one per HTTP
/// interaction, in the order the scenario drove them.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interaction {
    /// The request the provider accepted.
    ///
    /// Everything replay matches on lives here.
    pub when: WhenSpec,

    /// The response the provider returned.
    ///
    /// Everything replay serves lives here.
    pub then: ThenSpec,
}

/// The base URL a provider client should point at the mock server.
///
/// Each client appends its own endpoint path to its base URL — the
/// OpenAI-compat convention carries `/v1` in the base and Gemini
/// carries `/v1beta`, while Anthropic keeps the bare origin — so the
/// mock's root has to be decorated per provider to reproduce the
/// exact recorded path.
pub fn client_base_url(provider: &str, mock_base_url: &str) -> String {
    match provider {
        "openai-compat" | "openai" | "deepseek" | "grok" => format!("{mock_base_url}/v1"),
        "gemini" => format!("{mock_base_url}/v1beta"),
        "zai" => format!("{mock_base_url}/api/anthropic"),
        _ => mock_base_url.to_string(),
    }
}

/// The on-disk location of one provider scenario's cassette.
///
/// Rooted at this crate's manifest so the path is correct no matter
/// which working directory a test or the record driver runs from.
pub fn cassette_path(provider: &str, scenario: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cassettes")
        .join(provider)
        .join(format!("{scenario}.yaml"))
}

/// Read and parse a committed cassette.
///
/// # Panics
///
/// Panics when the cassette is missing or unparseable — a scenario
/// wired into CI without its recording is a broken suite, not a skip.
pub fn load_interactions(provider: &str, scenario: &str) -> Vec<Interaction> {
    let path = cassette_path(provider, scenario);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing cassette {path:?} for {provider}/{scenario} — record it first (LOOPCTL_CASSETTE=record LOOPCTL_E2E=1): {e}"));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("cassette {path:?} does not parse: {e}"))
}

/// Redact real data, never invent data: apply the scrub policy to a
/// recorded exchange set.
///
/// Three passes, in order: response headers are filtered to the
/// allowlist (request headers were never recorded beyond it); forbidden
/// request-header names and sensitive query values are removed or
/// rewritten; provider-generated ids are renamed to deterministic
/// placeholders **consistently across the whole file** — an id the
/// engine copies from one response into a later request keeps matching
/// after the rename.
pub fn scrub(interactions: &mut [Interaction]) {
    filter_response_headers(interactions);
    scrub_request_pairs(interactions);
    let renames = collect_id_renames(interactions);
    apply_id_renames(interactions, &renames);
}

/// One active cassette, replaying or recording against a caller-owned
/// [`MockServer`].
///
/// httpmock's mock and recording handles borrow the server they were
/// created on, so the session borrows it too: create the server, hand
/// it to [`CassetteSession::start`], drive your scenario against
/// [`CassetteSession::base_url`], then [`CassetteSession::finish`] —
/// replay asserts every interaction was consumed exactly once, record
/// scrubs and writes the cassette.
pub struct CassetteSession<'s> {
    server: &'s MockServer,
    mode: CassetteMode,
    provider: String,
    scenario: String,
    mocks: Vec<Mock<'s>>,
    recording: Option<Recording<'s>>,
}

impl<'s> CassetteSession<'s> {
    /// Start the session for one provider scenario.
    ///
    /// Replay (the default) registers every recorded interaction as a
    /// byte-exact mock. Record forwards to `real_base_url` and captures
    /// the exchange — and panics unless the deliberate-act environment
    /// is set (see [`require_deliberate_recording`]).
    ///
    /// # Panics
    ///
    /// Panics in record mode without `LOOPCTL_E2E=1` and the
    /// provider's credentials, and in replay mode when the cassette
    /// cannot be loaded or an interaction cannot be registered.
    pub async fn start(
        provider: &str,
        scenario: &str,
        real_base_url: &str,
        server: &'s MockServer,
    ) -> CassetteSession<'s> {
        match mode_from_env() {
            CassetteMode::Replay => {
                let interactions = load_interactions(provider, scenario);
                let mocks = register_replay_mocks(server, &interactions).await;
                CassetteSession {
                    server,
                    mode: CassetteMode::Replay,
                    provider: provider.to_string(),
                    scenario: scenario.to_string(),
                    mocks,
                    recording: None,
                }
            }
            CassetteMode::Record => {
                require_deliberate_recording(provider, real_base_url);
                server
                    .forward_to_async(real_base_url.to_string(), |rule| {
                        rule.filter(|when| {
                            when.any_request();
                        });
                    })
                    .await;
                let recording = server
                    .record_async(|rule| {
                        rule.record_request_headers(vec!["content-type", "accept"])
                            .filter(|when| {
                                when.any_request();
                            });
                    })
                    .await;
                CassetteSession {
                    server,
                    mode: CassetteMode::Record,
                    provider: provider.to_string(),
                    scenario: scenario.to_string(),
                    mocks: Vec::new(),
                    recording: Some(recording),
                }
            }
        }
    }

    /// The URL scenarios direct provider clients at.
    ///
    /// The mock server's root; per-provider path conventions are
    /// applied by [`client_base_url`] before a client sees it.
    pub fn base_url(&self) -> String {
        self.server.base_url()
    }

    /// Close the session.
    ///
    /// Replay asserts every registered interaction was consumed
    /// exactly once — none skipped, none served twice. Record exports
    /// the captured exchange, scrubs it, and writes the cassette,
    /// returning its path.
    ///
    /// # Panics
    ///
    /// Panics when a replayed interaction was not hit exactly once, or
    /// when the recording cannot be exported or written.
    pub async fn finish(self) -> PathBuf {
        match self.mode {
            CassetteMode::Replay => {
                for mock in &self.mocks {
                    mock.assert();
                }
                cassette_path(&self.provider, &self.scenario)
            }
            CassetteMode::Record => {
                let recording = self.recording.expect("record session holds its recording");
                let bytes = recording
                    .export_async()
                    .await
                    .expect("the recording exports")
                    .unwrap_or_else(|| {
                        panic!(
                            "nothing was recorded for {}/{}",
                            self.provider, self.scenario
                        )
                    });
                let text = String::from_utf8_lossy(&bytes).into_owned();
                let mut interactions = Vec::new();
                for document in serde_yaml::Deserializer::from_str(&text) {
                    let value = serde::Deserialize::deserialize(document)
                        .unwrap_or_else(|e| panic!("recorded exchange does not parse: {e}"));
                    let interaction: Interaction = serde_yaml::from_value(value)
                        .unwrap_or_else(|e| panic!("recorded interaction does not parse: {e}"));
                    interactions.push(interaction);
                }
                scrub(&mut interactions);
                let path = cassette_path(&self.provider, &self.scenario);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .unwrap_or_else(|e| panic!("cannot create {parent:?}: {e}"));
                }
                let yaml = serde_yaml::to_string(&interactions)
                    .unwrap_or_else(|e| panic!("scrubbed cassette does not serialize: {e}"));
                std::fs::write(&path, yaml)
                    .unwrap_or_else(|e| panic!("cannot write {path:?}: {e}"));
                path
            }
        }
    }
}

/// Register every interaction as a byte-exact mock on the server.
///
/// Matching covers method, path, query, the recorded header allowlist,
/// and the full request body — anything else the client sends is
/// ignored, and anything recorded that the client builds differently
/// misses.
///
/// # Panics
///
/// Panics on any interaction carrying a `body_base64` field: a binary
/// body has no registered matcher or served payload, so such an
/// interaction would silently lose the drift guard — refusing it is
/// the only honest replay.
pub async fn register_replay_mocks<'s>(
    server: &'s MockServer,
    interactions: &[Interaction],
) -> Vec<Mock<'s>> {
    let mut mocks = Vec::new();
    for interaction in interactions {
        if interaction.when.body_base64.is_some() || interaction.then.body_base64.is_some() {
            panic!(
                "binary cassette bodies (`body_base64`) cannot be replayed — refusing rather than silently dropping the body matcher"
            );
        }
        let when = interaction.when.clone();
        let then = interaction.then.clone();
        mocks.push(
            server
                .mock_async(move |when_spec, then_spec| {
                    apply_when(when_spec, &when);
                    apply_then(then_spec, &then);
                })
                .await,
        );
    }
    mocks
}

/// Fill one mock's request matcher from a recorded `when` spec.
///
/// Present fields become exact matchers; absent ones are left unset,
/// so the mock matches exactly what was recorded and nothing more.
fn apply_when(mut spec: httpmock::When, when: &WhenSpec) -> httpmock::When {
    if let Some(method) = &when.method {
        spec = spec.method(method.as_str());
    }
    if let Some(path) = &when.path {
        spec = spec.path(path.clone());
    }
    if let Some(pairs) = &when.query_param {
        for pair in pairs {
            spec = spec.query_param(pair.name.clone(), pair.value.clone());
        }
    }
    if let Some(pairs) = &when.header {
        for pair in pairs {
            spec = spec.header(pair.name.clone(), pair.value.clone());
        }
    }
    if let Some(body) = &when.body {
        spec = spec.body(body.clone());
    }
    spec
}

/// Fill one mock's response from a recorded `then` spec.
///
/// Status, allowlisted headers, and body are served exactly as they
/// were scrubbed into the cassette.
fn apply_then(mut spec: httpmock::Then, then: &ThenSpec) -> httpmock::Then {
    if let Some(status) = then.status {
        spec = spec.status(status);
    }
    if let Some(pairs) = &then.header {
        for pair in pairs {
            spec = spec.header(pair.name.clone(), pair.value.clone());
        }
    }
    if let Some(body) = &then.body {
        spec = spec.body(body.clone());
    }
    spec
}

/// Refuse to record unless a human deliberately asked for it.
///
/// Recording needs `LOOPCTL_E2E=1` — the same gate as every live test
/// — plus the provider's real credentials when the upstream is not a
/// local address: a cassette must be produced by the real server, and
/// hitting it with real credentials is not something a test run does
/// by accident.
///
/// The credential crosses the client-to-proxy hop in cleartext, on
/// loopback only, and that is the accepted posture: anyone positioned
/// to sniff loopback traffic is equally positioned to read the key
/// from this process's environment, so the hop adds no exposure the
/// environment variable does not already have, and encrypting it
/// would cost an all-or-nothing TLS mockserver every plain-TCP pin
/// depends on being without.
///
/// # Panics
///
/// Panics when either half of the deliberate-act contract is missing.
pub fn require_deliberate_recording(provider: &str, real_base_url: &str) {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        panic!(
            "recording is a deliberate human act: set LOOPCTL_E2E=1 (plus the provider credentials) to record {provider}"
        );
    }
    let is_local = ["localhost", "127.0.0.1", "[::1]", "::1"]
        .iter()
        .any(|host| real_base_url.contains(host));
    if is_local {
        return;
    }
    let accepted = match provider {
        "openai-compat" | "openai" => vec!["OPENAI_API_KEY"],
        "anthropic" => vec!["ANTHROPIC_API_KEY"],
        // Each entry accepts the same spellings its record driver and
        // the crate's provider constructors read.
        "gemini" => vec!["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "deepseek" => vec!["DEEPSEEK_API_KEY"],
        "grok" => vec!["XAI_API_KEY", "GROK_API_KEY"],
        "zai" => vec!["ZAI_API_KEY", "ZHIPUAI_API_KEY"],
        other => panic!("unknown cassette provider {other:?}"),
    };
    if !accepted.iter().any(|name| std::env::var(name).is_ok()) {
        panic!(
            "recording {provider} needs {} in the environment — real cassettes come from real servers",
            accepted.join(" or ")
        );
    }
}

/// Drop every response header outside the allowlist.
///
/// `content-type` and `retry-after` are load-bearing for replay (the
/// SSE parser keys off the first; the retry machinery parses the
/// second, so 429 cassettes must carry it). Everything else a provider
/// sends — request ids, ratelimit buckets, server banners — is noise
/// at best and a leak at worst.
fn filter_response_headers(interactions: &mut [Interaction]) {
    for interaction in interactions {
        if let Some(headers) = interaction.then.header.take() {
            interaction.then.header = Some(
                headers
                    .into_iter()
                    .filter(|pair| {
                        RESPONSE_HEADER_ALLOWLIST.contains(&pair.name.to_ascii_lowercase().as_str())
                    })
                    .collect(),
            );
        }
    }
}

/// Remove forbidden request headers and rewrite sensitive query values.
///
/// The recording rule only ever captured `content-type` and `accept`
/// request headers, so this pass is the belt to that braces: a
/// forbidden name found here means the allowlist leaked, and the
/// safety meta-test will flag the file as well.
fn scrub_request_pairs(interactions: &mut [Interaction]) {
    for interaction in interactions {
        if let Some(headers) = interaction.when.header.take() {
            interaction.when.header = Some(
                headers
                    .into_iter()
                    .filter(|pair| {
                        let name = pair.name.to_ascii_lowercase();
                        !FORBIDDEN_HEADERS.iter().any(|forbidden| name == *forbidden)
                            && !name.starts_with("x-amz-")
                    })
                    .collect(),
            );
        }
        if let Some(queries) = interaction.when.query_param.take() {
            interaction.when.query_param = Some(
                queries
                    .into_iter()
                    .map(|mut pair| {
                        if SENSITIVE_QUERY_PARAMS.contains(&pair.name.to_ascii_lowercase().as_str())
                        {
                            pair.value = CASSETTE_KEY.to_string();
                        }
                        pair
                    })
                    .collect(),
            );
        }
    }
}

/// Build the file-wide rename table for provider-generated ids.
///
/// Ids are discovered in first-appearance order across all response
/// bodies (where the provider mints them) and mapped to
/// `<prefix>cassette_<n>` — deterministic across recordings of the
/// same scenario, unique within the file.
fn collect_id_renames(interactions: &[Interaction]) -> Vec<(String, String)> {
    let mut renames = Vec::new();
    for interaction in interactions {
        if let Some(body) = &interaction.then.body {
            for id in scan_generated_ids(body) {
                if !renames.iter().any(|(from, _)| *from == id) {
                    let prefix = GENERATED_ID_PREFIXES
                        .iter()
                        .find(|prefix| id.starts_with(**prefix))
                        .unwrap_or(&"");
                    let to = format!("{prefix}cassette_{}", renames.len() + 1);
                    renames.push((id, to));
                }
            }
        }
    }
    renames
}

/// Apply the rename table everywhere an id can travel in the file.
///
/// Bodies on both sides (the response that minted the id, the request
/// that echoed it back), the request path, and every recorded pair
/// value — a provider that puts a generated id in a URL path or query
/// keeps the whole exchange consistent under the rename, so replay's
/// byte matching survives the scrub. Each text is rewritten in a
/// single pass ([`rename_ids_in_text`]), so placeholders can never be
/// rescanned and one-to-one mappings stay one-to-one.
fn apply_id_renames(interactions: &mut [Interaction], renames: &[(String, String)]) {
    for interaction in interactions {
        let mut texts = vec![
            interaction.when.body.as_mut(),
            interaction.then.body.as_mut(),
            interaction.when.path.as_mut(),
        ];
        for pairs in [
            interaction.when.header.as_mut(),
            interaction.when.query_param.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            for pair in pairs {
                texts.push(Some(&mut pair.value));
            }
        }
        for text in texts.into_iter().flatten() {
            *text = rename_ids_in_text(text, renames);
        }
    }
}

/// The byte class an id is made of — the single definition shared by
/// discovery and replacement, so the two can never disagree about what
/// a token is.
///
/// ASCII word characters only: a preceding multi-byte character's
/// continuation byte falls outside the class, which is exactly the
/// boundary semantics the id grammar needs.
fn is_id_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

/// Rewrite one text by replacing whole id tokens, longest match first.
///
/// A single left-to-right pass over the original text: at each
/// character boundary, the longest source id that starts *and* ends at
/// a token boundary wins; every other byte is copied verbatim. Three
/// properties sequential `replace` calls cannot give: a placeholder
/// written earlier in the pass is never itself rescanned, a shorter id
/// that prefixes a longer one (`msg_ab` inside `msg_abcd`) never
/// corrupts the longer token, and an id-shaped suffix inside a longer
/// client word (`msg_ab` inside `foomsg_ab`) is never touched — the
/// scrubbed request must stay byte-identical to what the replay client
/// sends.
fn rename_ids_in_text(text: &str, renames: &[(String, String)]) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let matched = renames
            .iter()
            .filter(|(from, _)| {
                text[i..].starts_with(from.as_str())
                    && (i == 0 || !bytes.get(i - 1).is_some_and(|b| is_id_byte(*b)))
                    && !bytes.get(i + from.len()).is_some_and(|b| is_id_byte(*b))
            })
            .max_by_key(|(from, _)| from.len());
        if let Some((from, to)) = matched {
            out.push_str(to);
            i += from.len();
        } else {
            let step = text[i..].chars().next().map_or(1, char::len_utf8);
            let end = (i + step).min(bytes.len());
            out.push_str(&text[i..end]);
            i = end;
        }
    }
    out
}

/// Extract provider-generated ids from one body, in appearance order.
///
/// A hand-rolled scan rather than a regex: the id grammar is a known
/// prefix followed by word characters, and the policy (which prefixes,
/// what replacement) belongs in named constants a reader can find. An
/// id is only discovered as a whole token — the same leading and
/// trailing boundaries [`rename_ids_in_text`] enforces — so an
/// id-shaped suffix inside a longer word never enters the rename
/// table and the placeholder numbering stays independent of prose.
fn scan_generated_ids(body: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let remaining = &body[i..];
        if (i == 0 || !bytes.get(i - 1).is_some_and(|b| is_id_byte(*b)))
            && let Some(prefix) = GENERATED_ID_PREFIXES
                .iter()
                .find(|prefix| remaining.starts_with(**prefix))
        {
            let id_start = i;
            let mut end = i + prefix.len();
            while end < bytes.len() && is_id_byte(bytes[end]) {
                end += 1;
            }
            if end > i + prefix.len() {
                let id = body[id_start..end].to_string();
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            i = end;
        } else {
            i += body[i..].chars().next().map_or(1, char::len_utf8);
        }
    }
    ids
}

/// One scripted provider scenario, shared verbatim by the record
/// driver and the replay suites.
///
/// Replay matches request bytes, so everything that shapes them —
/// model, prompt, tool schema, stream options — is fixed here and used
/// by both sides. Re-recording on a machine without a scenario's model
/// means editing this table and re-recording that provider's corpus.
pub struct Scenario {
    /// The cassette folder: `tests/cassettes/<provider>/`.
    ///
    /// Names the wire dialect, not the vendor — `openai-compat` covers
    /// every OpenAI-compatible endpoint including local Ollama.
    pub provider: &'static str,

    /// The scenario name: the cassette file stem.
    ///
    /// Unique within its provider; the record driver and the replay
    /// suite agree on it through this table.
    pub name: &'static str,

    /// The exact model string sent on the wire.
    ///
    /// Part of the recorded request body, so replay must send the
    /// same string or miss; re-recording with a different model is a
    /// table edit plus a fresh recording.
    pub model: &'static str,

    /// The exact user prompt sent on the wire.
    ///
    /// Fixed for the same byte-stability reason as the model — the
    /// scenario's determinism is what makes replay matching possible.
    pub prompt: &'static str,

    /// Whether the request carries the fixed [`get_weather_tool`].
    ///
    /// Tool-carrying scenarios pin the full tool-call streaming
    /// lifecycle, not just text.
    pub tools: bool,

    /// Whether the request asks the server for streamed usage
    /// (`stream_options.include_usage` on the OpenAI-compat wire).
    ///
    /// A request-body difference like any other: the replay client
    /// sets the same flag the recording did.
    pub stream_usage: bool,
}

/// The fixed tool every tool-carrying scenario sends.
///
/// One function, one required string property — small enough to keep
/// the recorded bodies readable, real enough to drive the full
/// tool-call streaming lifecycle.
pub fn get_weather_tool() -> loopctl::tool::ToolSchema {
    loopctl::tool::ToolSchema {
        tool: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    }
}

/// The scenario set the record driver and replay suites share.
///
/// The `openai-compat` scenarios are recorded against a local Ollama
/// (`http://localhost:11434/v1`) — local wire shapes drift the most
/// across server versions, which is exactly what the guard exists for.
/// The `anthropic` and `gemini` scenarios record against the real
/// clouds and need credentials.
pub fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            provider: "openai-compat",
            name: "minimal_text",
            model: "qwen2.5:7b",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: true,
        },
        Scenario {
            provider: "openai-compat",
            name: "tool_call_lifecycle",
            model: "qwen3.5:4b",
            prompt: "Use the get_weather tool to check the weather in Paris.",
            tools: true,
            stream_usage: true,
        },
        Scenario {
            provider: "anthropic",
            name: "minimal_text",
            model: "claude-sonnet-4-5",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: false,
        },
        Scenario {
            provider: "gemini",
            name: "minimal_text",
            model: "gemini-3.5-flash",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: false,
        },
        Scenario {
            provider: "openai",
            name: "minimal_text",
            model: "gpt-4.1-mini",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: true,
        },
        Scenario {
            provider: "openai",
            name: "tool_call_lifecycle",
            model: "gpt-4.1-mini",
            prompt: "Use the get_weather tool to check the weather in Paris.",
            tools: true,
            stream_usage: true,
        },
        Scenario {
            provider: "anthropic",
            name: "tool_call_lifecycle",
            model: "claude-sonnet-4-5",
            prompt: "Use the get_weather tool to check the weather in Paris.",
            tools: true,
            stream_usage: false,
        },
        Scenario {
            provider: "gemini",
            name: "tool_call_lifecycle",
            model: "gemini-3.5-flash",
            prompt: "Use the get_weather tool to check the weather in Paris.",
            tools: true,
            stream_usage: false,
        },
        Scenario {
            provider: "deepseek",
            name: "minimal_text",
            model: "deepseek-v4-flash",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: true,
        },
        Scenario {
            provider: "grok",
            name: "minimal_text",
            model: "grok-4.5",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: true,
        },
        Scenario {
            provider: "zai",
            name: "minimal_text",
            model: "glm-4.7",
            prompt: "Say hello in exactly 3 words.",
            tools: false,
            stream_usage: false,
        },
    ]
}

/// Scan cassette text for anything that must never be committed.
///
/// Returns one violation message per finding — forbidden header names,
/// bearer tokens, and the well-known shapes of OpenAI, Google, and AWS
/// keys. A key shape embedded in a long opaque run (the multi-KiB
/// base64 blobs thinking models attach) is not a finding: those runs
/// are signatures and payloads, not credentials, and flagging them
/// would block CI on a coincidence. The committed-corpus meta-test
/// fails on any non-empty result.
pub fn scan_for_secrets(text: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let lowercase = text.to_ascii_lowercase();
    for forbidden in FORBIDDEN_HEADERS
        .iter()
        .copied()
        .chain(std::iter::once("x-amz-security-token"))
    {
        let needle = format!("name: {forbidden}");
        if lowercase.contains(&needle) {
            violations.push(format!("forbidden header {forbidden:?} present"));
        }
    }
    if lowercase.contains("bearer ") {
        violations.push("bearer token present".to_string());
    }
    for (needle, what) in [
        ("sk-", "OpenAI-style key"),
        ("AIza", "Google-style key"),
        ("AKIA", "AWS-style key"),
    ] {
        let mut from = 0;
        while let Some(found) = text[from..].find(needle) {
            let start = from + found + needle.len();
            let run: String = text[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if run.len() >= 16 && enclosing_run_len(text, from + found) < OPAQUE_RUN_FLOOR {
                violations.push(format!("{what} present ({needle}{run})"));
                break;
            }
            from = start;
        }
    }
    violations
}

/// The length of the unbroken alphanumeric run a byte offset sits in.
///
/// Used by [`scan_for_secrets`] to tell a standalone credential from a
/// key shape that happens to occur inside a base64 signature or
/// payload blob.
fn enclosing_run_len(text: &str, at: usize) -> usize {
    let bytes = text.as_bytes();
    let is_word = |b: u8| -> bool { b.is_ascii_alphanumeric() || b == b'_' || b == b'-' };
    let mut start = at;
    while start > 0 && bytes.get(start - 1).is_some_and(|b| is_word(*b)) {
        start -= 1;
    }
    let mut end = at;
    while bytes.get(end).is_some_and(|b| is_word(*b)) {
        end += 1;
    }
    end - start
}

/// Runs of word characters at least this long are treated as opaque
/// blobs, not credentials.
///
/// A real key of every flagged shape is shorter than this floor on its
/// own; only coincidence inside signature payloads reaches past it.
const OPAQUE_RUN_FLOOR: usize = 40;
