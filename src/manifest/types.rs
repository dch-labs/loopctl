//! Serde types for the v1 `loop.yaml` manifest.
//!
//! Every type is strict (`deny_unknown_fields`), schemars-exportable, and
//! `#[non_exhaustive]` so future field additions stay additive per the
//! v0.4.0 semver-hardening guarantee. Maps are [`BTreeMap`]s so the canonical
//! JSON projection — and with it the resolved-config hash — is deterministic
//! across processes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The root document type of a v1 `loop.yaml`.
///
/// All sections except `version` are optional; validation runs on the merged
/// document (base plus the selected profile overlay), and interpolation runs
/// after that, so the values seen here are post-merge but pre-interpolation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Manifest {
    /// The manifest format version; v1 is the only version this crate reads.
    ///
    /// A newer version hard-errors
    /// ([`UnsupportedVersion`][crate::manifest::ManifestError::UnsupportedVersion]);
    /// there is no older version to migrate. Enforced at parse time so an
    /// unreadable document never reaches profile merging, and exported as
    /// a `"const": 1` constraint so schema validators reject other values
    /// exactly like the parser does.
    #[schemars(extend("const" = 1))]
    pub version: u32,

    /// Distribution metadata for the agent-as-artifact flow.
    ///
    /// Purely identity data; nothing in resolution or validation depends on
    /// it, so tooling may strip or regenerate it without changing the
    /// resolved configuration.
    #[serde(default)]
    pub metadata: Metadata,

    /// The agent's identity: instruction text and advertised tool ids.
    ///
    /// Both fields are optional — a manifest can exist purely to declare
    /// models and budgets for a host that supplies its own instruction.
    #[serde(default)]
    pub agent: AgentSection,

    /// Named model entries plus the fallback chain over those names.
    ///
    /// The chain projects onto the engine's fallback manager at run time;
    /// validation checks every name in it resolves before that happens.
    #[serde(default)]
    pub models: ModelsSection,

    /// The tool surface: builtin tools and MCP-sourced tools.
    ///
    /// Order is author-declared and preserved — the surface the model sees
    /// follows the document, not map iteration.
    #[serde(default)]
    pub tools: Vec<ToolEntry>,

    /// Named MCP server declarations referenced by [`ToolEntry`] stanzas.
    ///
    /// Declaring a server here is inert until a tool entry points at it;
    /// unreferenced servers validate fine and simply never launch.
    #[serde(default)]
    pub mcp: BTreeMap<String, McpServer>,

    /// The permission mode and declarative rule lists.
    ///
    /// The whole stanza is data for the gate engine; this model validates
    /// shape, never intent.
    ///
    /// The mode sets the default posture; the rules are carried as data —
    /// their matching semantics belong to the gate engine.
    #[serde(default)]
    pub permissions: Permissions,

    /// Run budgets; enforcement is the budget gate middleware's concern.
    ///
    /// Carried as declared intent so a resolved manifest can be diffed
    /// against what a live run was actually configured with.
    #[serde(default)]
    pub budgets: Budgets,

    /// Context-management settings (compaction trigger and strategy).
    ///
    /// Maps onto the context manager's thresholds when the host wires one;
    /// absent means the engine defaults apply.
    #[serde(default)]
    pub context: ContextSection,

    /// Persistent-memory settings (store path and retention).
    ///
    /// The store path is resolved like any other string — interpolation
    /// applies — but no filesystem access happens during validation.
    #[serde(default)]
    pub memory: MemorySection,

    /// The schedule stanza; carrying it as data is this crate's whole job —
    /// execution belongs to the scheduler.
    ///
    /// A manifest declaring a schedule is valid on a machine that never
    /// fires it; the stanza is a declaration, not a running thing.
    #[serde(default)]
    pub schedule: ScheduleSection,

    /// File-watch trigger settings.
    ///
    /// The v1 trigger surface: globs plus a debounce window.
    ///
    /// The watch globs and debounce window are declared here so trigger
    /// behavior is diffable alongside everything else.
    #[serde(default)]
    pub triggers: TriggersSection,

    /// Cassette record/replay policy.
    ///
    /// Two independent knobs — recording and replaying are separate
    /// concerns that happen to share a corpus.
    ///
    /// Feeds the cassette harness's mode selection; the policy travels with
    /// the manifest so a recorded suite replays under the rules it was
    /// recorded with.
    #[serde(default)]
    pub cassettes: CassettesSection,

    /// Sandbox declaration; enforcement belongs to the sandbox tier.
    ///
    /// Declaring a tier the runtime cannot enforce is a run-time failure,
    /// not a validation one — this model does not know the host's tiers.
    #[serde(default)]
    pub sandbox: SandboxSection,

    /// Named in-file overlays merged over the base document on selection.
    ///
    /// Profiles live in the same file on purpose: the delta is reviewable
    /// next to the base it modifies, and one document pins everything.
    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
}

/// OCI-style distribution metadata.
///
/// Purely declarative identity for the agent-as-artifact flow; nothing in the
/// runtime reads it beyond display.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Metadata {
    /// The manifest's human-facing name.
    ///
    /// Optional so a profile overlay can patch `version` or `description`
    /// without repeating it; a nameless base document is legal but unusual.
    #[serde(default)]
    pub name: Option<String>,

    /// The manifest's own version string, independent of `Manifest::version`.
    ///
    /// Follows the artifact's release line, not the format version — bumping
    /// this never trips the format's version gate.
    #[serde(default)]
    pub version: Option<String>,

    /// A free-form description of what this agent does.
    ///
    /// Display-only; nothing parses it, so it may be any length and any
    /// language.
    #[serde(default)]
    pub description: Option<String>,
}

/// The agent stanza: instruction and the advertised tool ids.
///
/// Everything that names *what the agent is* rather than what it is
/// allowed to use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct AgentSection {
    /// The system-prompt text, or a path to a file containing it.
    ///
    /// Prose lives in files by convention; deciding whether this string is a
    /// path (and reading it) is the consumer's job, kept out of the model so
    /// validation never touches the filesystem.
    #[serde(default)]
    pub instruction: Option<String>,

    /// The ids of [`ToolEntry`] stanzas this agent advertises.
    ///
    /// Every id must resolve during validation.
    #[serde(default)]
    pub tools: Vec<String>,
}

/// Named model entries and the fallback chain over their names.
///
/// The chain references entries by name, so renaming a model is a
/// two-place edit the cross-reference check keeps honest.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelsSection {
    /// The named models addressable by [`ModelsSection::fallbacks`] and by
    /// host tooling.
    ///
    /// The map key is the model's declared name; entry contents carry the
    /// provider addressing. Keys are sorted in the canonical projection, so
    /// name order never affects the config hash.
    #[serde(default)]
    pub entries: BTreeMap<String, ModelEntry>,

    /// The fallback chain as model names, outermost first.
    ///
    /// Every name must exist in [`ModelsSection::entries`]; an empty chain
    /// means no fallback — the primary model alone serves the run.
    #[serde(default)]
    pub fallbacks: Vec<String>,
}

/// One named model: provider addressing plus auth reference.
///
/// The auth reference is a key NAME by design — see `token_key`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelEntry {
    /// The provider id, e.g. `anthropic` or `ollama`.
    ///
    /// Matches the provider feature names the library builds clients with;
    /// validation does not check the id exists — a manifest for a provider
    /// this build lacks fails when the host tries to construct the client.
    pub provider: String,

    /// The provider-native model string, verbatim.
    ///
    /// Never canonicalized here: whatever the provider's API expects goes
    /// on the wire exactly as written, aliasing left to host tooling.
    pub model: String,

    /// An optional base-URL override, interpolated at resolve time.
    ///
    /// The natural home of `${OLLAMA_URL:-http://localhost:11434}`-style
    /// references; the resolved value replaces this string wholesale.
    #[serde(default)]
    pub base_url: Option<String>,

    /// The per-request generation budget, when the provider takes one.
    ///
    /// Mapped onto the provider's max-tokens parameter by the host; `None`
    /// leaves the provider default in place.
    #[serde(default)]
    pub max_tokens: Option<u64>,

    /// The NAME of the environment variable holding the API key.
    ///
    /// Never interpolated and never a key value — validation rejects literal
    /// key-shaped strings here, which is precisely the failure this field
    /// exists to prevent.
    #[serde(default)]
    pub token_key: Option<String>,
}

/// One tool on the advertised surface: a builtin or an MCP-backed tool.
///
/// The entry carries the manifest-local id plus exactly one
/// implementation source; permission and argument constraints ride along.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ToolEntry {
    /// The id the agent references this tool by.
    ///
    /// Distinct from the builtin name or the MCP tool name — the id is the
    /// manifest-local handle `agent.tools` and permission rules match on.
    pub id: String,

    /// The builtin tool name this entry instantiates, when it is a builtin.
    ///
    /// Exactly one of `builtin` and `mcp` must be set; validation enforces
    /// the exclusive-or.
    #[serde(default)]
    pub builtin: Option<String>,

    /// The MCP server name (a key of [`Manifest::mcp`]) backing this entry.
    ///
    /// The server must be declared in the same document — a dangling server
    /// name is an [`UnresolvedReference`][crate::manifest::ManifestError::UnresolvedReference]
    /// at validation time.
    #[serde(default)]
    pub mcp: Option<String>,

    /// The per-tool permission override for this entry.
    ///
    /// When absent the document-level [`Permissions::mode`] governs; when
    /// present it wins for this tool only.
    #[serde(default)]
    pub permission: Option<PermissionMode>,

    /// Argument constraints in the `tool:pattern` rule shape.
    ///
    /// Only the shape is validated here; matching semantics belong to the
    /// gate engine.
    #[serde(default)]
    pub when: Vec<String>,
}

/// One named MCP server declaration.
///
/// Name-keyed so tool entries reference servers symbolically and a
/// retarget is one edit in one place.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct McpServer {
    /// The server launch command as an argv array.
    ///
    /// An array, never a shell string, so no word-splitting rules are
    /// implied anywhere in the pipeline.
    pub command: Vec<String>,

    /// Environment entries passed to the server, interpolated at resolve
    /// time — `${secret:NAME}` references land here.
    ///
    /// Values resolve through the [`EnvResolver`][crate::manifest::EnvResolver];
    /// the resolved secrets go to the server process, never into the
    /// canonical projection or the config hash.
    #[serde(default)]
    pub env: BTreeMap<String, String>,

    /// The advertised-tool filter; `None` advertises everything the server
    /// lists.
    ///
    /// Names are the server's own tool names, applied at discovery time by
    /// the host — the model never sees filtered-out tools.
    #[serde(default)]
    pub tools: Option<Vec<String>>,

    /// The supply-chain pin for this server, e.g. `sha256:…`.
    ///
    /// A content hash, so the literal-secret scan deliberately does not
    /// treat it as key-shaped.
    #[serde(default)]
    pub pin: Option<String>,
}

/// The permission mode and declarative rule lists.
///
/// The whole stanza is data for the gate engine; this model validates
/// shape, never intent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Permissions {
    /// The default mode when no rule matches.
    ///
    /// Gate is the safe default; the mode is data the gate engine reads,
    /// not behavior this crate performs.
    #[serde(default)]
    pub mode: PermissionMode,

    /// The deny/allow rule lists, deny-first.
    ///
    /// Order inside each list is author-declared; precedence between the
    /// lists is the gate engine's to define (deny wins).
    #[serde(default)]
    pub rules: PermissionRules,
}

/// The default permission posture.
///
/// Kebab-case on the wire: `gate`, `collaborative`, `unattended-sandboxed`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum PermissionMode {
    /// Ask before side effects (the default).
    ///
    /// The safe posture: anything unreviewed pauses for a human.
    #[default]
    Gate,

    /// Run with prompts only for destructive operations.
    ///
    /// Reads flow freely; writes and side effects still ask.
    Collaborative,

    /// Refuse to start without a sandbox tier and audit everything.
    ///
    /// Strictly stronger than the older bypass-style modes: no sandbox,
    /// no start.
    UnattendedSandboxed,
}

/// The declarative deny/allow rule lists.
///
/// Two lists rather than one weighted list because deny-first
/// precedence is the whole security story.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PermissionRules {
    /// Deny rules; matched first and unoverridable.
    ///
    /// Each entry carries the `tool:pattern` shape; shape is validated
    /// here, matching happens in the gate engine.
    #[serde(default)]
    pub deny: Vec<String>,

    /// Allow rules consulted after deny.
    ///
    /// An allow can never rescue a call a deny matched — the deny-first
    /// precedence is the point of having two lists.
    #[serde(default)]
    pub allow: Vec<String>,
}

/// Run budgets.
///
/// All fields are advisory data here; the budget gate middleware owns
/// enforcement semantics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Budgets {
    /// The maximum number of turns per run.
    ///
    /// Mapped onto the run config's turn cap by the host; `None` means the
    /// engine default applies.
    #[serde(default)]
    pub turns: Option<u64>,

    /// The maximum total tokens per run.
    ///
    /// Counted from provider-reported usage, not estimates; the budget gate
    /// decides what happens at the line.
    #[serde(default)]
    pub tokens: Option<u64>,

    /// The maximum spend per run, in US dollars.
    ///
    /// Requires a price table to be meaningful; without one the gate treats
    /// the budget as unenforceable rather than guessing a cost.
    #[serde(default)]
    pub cost_usd: Option<f64>,

    /// The wall-clock budget as a human duration string, e.g. `4h`.
    ///
    /// Kept as a string — parsing durations is the consumer's concern, so
    /// the model imposes no duration grammar beyond what the host accepts.
    #[serde(default)]
    pub duration: Option<String>,
}

/// The context-management stanza.
///
/// Deliberately thin: compaction is the only context policy the v1
/// manifest carries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ContextSection {
    /// Compaction settings.
    ///
    /// Absent entirely means the engine's own compaction defaults govern;
    /// this stanza only overrides what it names.
    #[serde(default)]
    pub compaction: CompactionSettings,
}

/// Compaction trigger and strategy.
///
/// Both fields optional so a manifest can tune either independently.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CompactionSettings {
    /// The context-occupancy fraction that triggers compaction, e.g. `0.8`.
    ///
    /// A plain fraction, not a percentage — `0.8` means eighty percent of
    /// the window. Out-of-range values are the host's to reject.
    #[serde(default)]
    pub trigger: Option<f64>,

    /// The named compaction strategy.
    ///
    /// Strategy names resolve against the host's registered compactors;
    /// unknown names fail at wiring time, not validation, because the
    /// model cannot know which strategies a host carries.
    #[serde(default)]
    pub strategy: Option<String>,
}

/// Persistent-memory settings.
///
/// The model stores where memory lives and how long it lasts; the store
/// implementation is the memory module's concern.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct MemorySection {
    /// The memory store location (e.g. a file path).
    ///
    /// Interpolated like every other string, so `${DATA_DIR}/memory.db`
    /// works; the store is opened by the host, never by validation.
    #[serde(default)]
    pub store: Option<String>,

    /// The retention window for memory entries, in days.
    ///
    /// Enforced by the store's consolidation pass; `None` keeps entries
    /// indefinitely.
    #[serde(default)]
    pub ttl_days: Option<u64>,
}

/// The schedule stanza, carried as data.
///
/// Execution (cron math, overlap locks, catch-up) is the scheduler's job;
/// this type only declares intent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ScheduleSection {
    /// A five-field cron expression.
    ///
    /// Mutually exclusive in spirit with `every` — declaring both is the
    /// host's ambiguity to reject; the model carries both verbatim.
    #[serde(default)]
    pub cron: Option<String>,

    /// An interval expression such as `30m`, as an alternative to cron.
    ///
    /// Interval firing needs no calendar math, which is why local-first
    /// setups prefer it; parsing the expression is the scheduler's job.
    #[serde(default)]
    pub every: Option<String>,

    /// The IANA time-zone name; `None` means local time.
    ///
    /// Only meaningful for cron schedules; intervals are zone-free by
    /// construction.
    #[serde(default)]
    pub time_zone: Option<String>,

    /// Whether a still-running instance blocks the next fire.
    ///
    /// `None` means the scheduler's own default (forbid) applies.
    #[serde(default)]
    pub overlap: Option<Overlap>,

    /// What happens to a fire missed while the scheduler was down.
    ///
    /// `None` means the scheduler's own default (skip) applies.
    #[serde(default)]
    pub missed_fire: Option<MissedFire>,

    /// The catch-up window in seconds within which one coalesced fire is
    /// allowed.
    ///
    /// Fires older than the window are skipped and recorded, never
    /// backlogged — the k8s `startingDeadlineSeconds` lesson.
    #[serde(default)]
    pub catch_up_window_secs: Option<u64>,

    /// Whether the schedule is paused.
    ///
    /// A paused schedule still validates and still hashes — pausing is a
    /// state change the scheduler honors, not a document change.
    #[serde(default)]
    pub suspend: Option<bool>,

    /// A one-shot timestamp that auto-disables after firing.
    ///
    /// Carried as a string so the host's timestamp grammar (offsets, date
    /// formats) is not frozen into the model.
    #[serde(default)]
    pub run_at: Option<String>,
}

/// The overlap policy for a schedule.
///
/// One knob with exactly two v1 values; `replace` stays deferred
/// because killing a mid-turn loop is destructive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Overlap {
    /// A running instance forbids the next fire (the default posture).
    ///
    /// Loops are rarely safe to run concurrently with themselves; the
    /// skipped fire is recorded, not silently dropped.
    Forbid,

    /// Fires proceed concurrently with a running instance.
    ///
    /// Opt-in for loops that are genuinely re-entrant — the author is
    /// asserting the safety the default refuses to assume.
    Allow,
}

/// The missed-fire policy for a schedule.
///
/// What a scheduler does about the fires it slept through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MissedFire {
    /// Skip fires missed while the scheduler was down (the default posture).
    ///
    /// No backlog replay — a loop that missed its window waits for the next
    /// one.
    Skip,

    /// Fire once on startup for a missed window.
    ///
    /// Coalesces any number of missed fires into one — the Quartz
    /// `FIRE_ONCE_NOW` semantics.
    FireOnce,
}

/// File-watch trigger settings.
///
/// The v1 trigger surface: globs plus a debounce window, both carried
/// verbatim for the watcher to interpret.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TriggersSection {
    /// The glob paths watched for changes.
    ///
    /// The scheduler's file-watch layer interprets the globs (gitignore
    /// semantics included); the model only carries them.
    #[serde(default)]
    pub watch: Vec<String>,

    /// The debounce window as a human duration string, e.g. `100ms`.
    ///
    /// Coalesces editor save-storms into one fire; parsed by the watcher,
    /// not by this model.
    #[serde(default)]
    pub debounce: Option<String>,
}

/// Cassette record/replay policy.
///
/// Two independent knobs — recording and replaying are separate
/// concerns that happen to share a corpus.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CassettesSection {
    /// When runs are recorded.
    ///
    /// `None` defers to the harness's own default; the modes themselves are
    /// the cassette harness's vocabulary.
    #[serde(default)]
    pub record: Option<RecordMode>,

    /// When recorded cassettes are replayed.
    ///
    /// Independent of `record` — a suite can replay without recording and
    /// vice versa.
    #[serde(default)]
    pub replay: Option<ReplayMode>,

    /// The redaction preset name.
    ///
    /// Names the preset, never the patterns — the redaction module owns
    /// which patterns a preset compiles to.
    #[serde(default)]
    pub redact: Option<String>,
}

/// When cassette recording happens.
///
/// The record half of the cassette policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum RecordMode {
    /// Record every run.
    ///
    /// The always-on posture for suites whose value is the accumulating
    /// corpus.
    Always,

    /// Never record.
    ///
    /// Replaying a fixed corpus without growing it.
    Never,
}

/// When cassette replay happens.
///
/// The replay half of the cassette policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReplayMode {
    /// Replay only when explicitly requested.
    ///
    /// The default posture — ordinary runs hit the wire, tests opt in.
    OnDemand,

    /// Replay every matching request when a cassette exists.
    ///
    /// Hermetic mode; a request with no matching cassette fails rather than
    /// reaching the network.
    Always,

    /// Never replay.
    ///
    /// Recording posture that keeps the corpus cold while runs stay live.
    Never,
}

/// The sandbox declaration.
///
/// Intent only — the tier taxonomy and enforcement belong to the
/// sandbox tier, not to this model.
///
/// The tier string is free-form because the tier taxonomy belongs to the
/// sandbox tier, not to this model; this stanza only carries the author's
/// intent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SandboxSection {
    /// The requested sandbox tier name.
    ///
    /// Free-form because the tier taxonomy is not this model's to define;
    /// the host maps the tier name onto whatever tiers it enforces.
    #[serde(default)]
    pub tier: Option<String>,

    /// The paths writable inside the sandbox.
    ///
    /// An allowlist — everything not listed is read-only at best; empty
    /// means the tier's own default write policy.
    #[serde(default)]
    pub fs_write: Vec<String>,

    /// The hosts reachable from inside the sandbox.
    ///
    /// Also an allowlist; empty means the tier's default network posture,
    /// which for strict tiers is no egress at all.
    #[serde(default)]
    pub network_allow: Vec<String>,
}

/// Overlay counterpart of [`McpServer`] for profile stanzas.
///
/// Every field is optional so a profile can retarget one aspect of a
/// server (a pin, an env entry) without re-declaring the rest; the merged
/// document deserializes as [`McpServer`], where `command` is required —
/// so a profile patching an entry the base never declared fails the merge
/// with the missing-field error, exactly as a base document would.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct McpServerOverlay {
    /// The server launch command as an argv array, when the overlay
    /// retargets it.
    ///
    /// Replaces the base command wholesale; an array, never a shell
    /// string, for the same reason as [`McpServer::command`].
    #[serde(default)]
    pub command: Option<Vec<String>>,

    /// Environment entries merged over the base map by key.
    ///
    /// Interpolation and the pinning rules are identical to
    /// [`McpServer::env`].
    #[serde(default)]
    pub env: BTreeMap<String, String>,

    /// The advertised-tool filter, when the overlay narrows or widens it.
    ///
    /// Replaces the base filter wholesale; `None` leaves the base filter
    /// (or its absence) untouched.
    #[serde(default)]
    pub tools: Option<Vec<String>>,

    /// The supply-chain pin, when the overlay pins or re-pins the server.
    ///
    /// A content hash like [`McpServer::pin`]; the literal-secret scan
    /// leaves hashes alone for the same reason.
    #[serde(default)]
    pub pin: Option<String>,
}

/// Overlay counterpart of [`ModelEntry`] for profile stanzas.
///
/// Every field is optional so a profile can retune one knob of one model
/// (the generation budget, a base URL) without re-declaring the entry; the
/// merged document deserializes as [`ModelEntry`], where `provider` and
/// `model` are required — so a profile patching an entry the base never
/// declared fails the merge with the missing-field error.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelEntryOverlay {
    /// The provider id, when the overlay switches providers.
    ///
    /// Switching providers usually pairs with switching `model` and
    /// `token_key`; nothing enforces the pairing, the provider's own
    /// client construction will.
    #[serde(default)]
    pub provider: Option<String>,

    /// The provider-native model string, when the overlay switches models.
    ///
    /// Verbatim like [`ModelEntry::model`] — the overlay carries the
    /// provider's own spelling, not an alias.
    #[serde(default)]
    pub model: Option<String>,

    /// A base-URL override, when the overlay retargets the endpoint.
    ///
    /// The natural place for a profile to point a named model at a local
    /// gateway; interpolation and pinning behave exactly as in
    /// [`ModelEntry::base_url`].
    #[serde(default)]
    pub base_url: Option<String>,

    /// The per-request generation budget, when the overlay retunes it.
    ///
    /// The one-field patch this overlay type exists for: a profile that
    /// trades answer length for cost sets exactly this field and nothing
    /// else.
    #[serde(default)]
    pub max_tokens: Option<u64>,

    /// The API-key environment-variable name, when the overlay switches
    /// auth.
    ///
    /// Still a name, never a value; the literal-secret scan applies inside
    /// overlays exactly as it does in the base document.
    #[serde(default)]
    pub token_key: Option<String>,
}

/// Overlay counterpart of [`ModelsSection`] for profile stanzas.
///
/// Named entries deep-merge field by field via [`ModelEntryOverlay`];
/// `fallbacks` follows the list rules (replace unless `!append`ed).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelsOverlay {
    /// The named-model patches, keyed like [`ModelsSection::entries`].
    ///
    /// A key the base never declared is a new entry, and the merged
    /// document then enforces the base-required fields on it.
    #[serde(default)]
    pub entries: BTreeMap<String, ModelEntryOverlay>,

    /// The fallback chain overlay as model names, outermost first.
    ///
    /// Replaces the base chain unless tagged `!append`.
    #[serde(default)]
    pub fallbacks: Vec<String>,
}

/// A named in-file overlay merged over the base document.
///
/// Shaped exactly like [`Manifest`] minus `version` (fixed) and `profiles`
/// (no nesting) — one shape means one set of merge rules to document, and
/// the overlay guard rejects both forbidden keys before the merge runs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Profile {
    /// Distribution metadata overlay.
    ///
    /// Merges by the map rules — keys the overlay names replace, keys it
    /// omits survive.
    #[serde(default)]
    pub metadata: Metadata,

    /// Agent stanza overlay.
    ///
    /// The instruction string is a scalar: an overlay that names it
    /// replaces it whole.
    #[serde(default)]
    pub agent: AgentSection,

    /// Models overlay; named entries deep-merge field by field through
    /// [`ModelEntryOverlay`], `fallbacks` replaces unless `!append`ed.
    ///
    /// A profile can retune one knob of one model — the generation budget,
    /// a base URL — without re-declaring the entry; required fields are
    /// enforced on the merged document, not the overlay.
    #[serde(default)]
    pub models: ModelsOverlay,

    /// Tool-surface overlay; the list replaces unless `!append`ed.
    ///
    /// Replacement keeps profile tool surfaces explicit; `!append` is the
    /// escape hatch for additive profiles.
    #[serde(default)]
    pub tools: Vec<ToolEntry>,

    /// MCP server declarations overlay; named servers deep-merge field by
    /// field through [`McpServerOverlay`].
    ///
    /// A profile can retarget a server's command or add a pin without
    /// re-declaring its whole stanza.
    #[serde(default)]
    pub mcp: BTreeMap<String, McpServerOverlay>,

    /// Permissions overlay.
    ///
    /// Mode is a scalar (replace); the rule lists follow the list rules
    /// (replace unless `!append`ed).
    #[serde(default)]
    pub permissions: Permissions,

    /// Budgets overlay.
    ///
    /// Each budget field is a scalar — a profile tightening `cost_usd`
    /// replaces exactly that number.
    #[serde(default)]
    pub budgets: Budgets,

    /// Context-management overlay.
    ///
    /// The compaction sub-map merges by key, so a profile can move the
    /// trigger without naming a strategy.
    #[serde(default)]
    pub context: ContextSection,

    /// Memory overlay.
    ///
    /// Scalar fields under the map rules.
    #[serde(default)]
    pub memory: MemorySection,

    /// Schedule overlay.
    ///
    /// Scalar fields under the map rules.
    #[serde(default)]
    pub schedule: ScheduleSection,

    /// Triggers overlay.
    ///
    /// The watch list replaces unless `!append`ed.
    #[serde(default)]
    pub triggers: TriggersSection,

    /// Cassettes overlay.
    ///
    /// Scalar fields under the map rules.
    #[serde(default)]
    pub cassettes: CassettesSection,

    /// Sandbox overlay.
    ///
    /// Scalar tier plus the two list fields under the list rules.
    #[serde(default)]
    pub sandbox: SandboxSection,
}
