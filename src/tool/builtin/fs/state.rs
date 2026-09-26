//! Per-session record of the model's latest known content per touched file.
//!
//! The write tools' detect-on-write conflict check compares the
//! target's current content hash against the hash recorded when the
//! path was last touched; this module supplies that record.
//! [`FileSource`](super::FileSource) records what it observes; a
//! successful Write/Edit/MultiEdit records the post-write content, so
//! the model's own writes never register as external changes. The map
//! holds one entry per path, and the entry is always the *newest*
//! observation: each touch carries a sequence number stamped when its
//! bytes were in hand, and an older observation never supersedes a
//! newer one, whichever insert lands last.
//!
//! An observation made by re-arming on resume carries a marker (see
//! [`FileBaseline::resumed`]): the bytes were read at resume time, not
//! by the model in this session, so the write guard holds such a file
//! for a fresh live read before the first write.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// Monotonic observation counter, ordering concurrent touches of one path.
///
/// Stamped by `observe_bytes` at the moment the bytes are in hand;
/// the process-local counter never repeats a value within a session,
/// so the stamp order matches the observation order exactly.
static OBSERVATION_SEQ: AtomicU64 = AtomicU64::new(0);

/// The model's latest known content hash per touched file.
///
/// Keyed by each tool's resolution of the submitted path: the
/// [`resolve_path`](super::resolve::resolve_path) output under the
/// contained policy, and the canonicalized physical file on top of it
/// under the unrestricted policy. Under Unrestricted, then, equivalent
/// spellings of the same file — including spellings that pass through
/// a symbolic link — share one baseline and a staleness check cannot
/// be dodged by re-spelling a path. Under Contained the keys are the
/// lexical resolution output, so alias spellings are deliberately
/// distinct keys; there the protection comes from the write layer
/// refusing symbolic-link spellings rather than from key merging. The
/// rule is applied by the owning session —
/// [`FileSession`](super::FileSession)'s `record_baseline` and
/// `baseline_for` normalize keys on the way in and out, so no call
/// site can record or query under a divergent spelling.
///
/// The hash is a process-local `DefaultHasher` fingerprint of the
/// file's bytes — deliberately not a stable or cryptographic digest:
/// baselines live and die with the session, and the hash only ever
/// answers "did the bytes change since the model last touched this
/// file".
pub(crate) type FileBaselines = BTreeMap<PathBuf, FileBaseline>;

/// One observation of a file's content, at a known point in the session's
/// observation order.
///
/// Produced by `observe_bytes` at the moment the bytes were in
/// hand; a map entry always holds the newest one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileBaseline {
    /// Where this observation sits in the session's touch order.
    ///
    /// Stamped when the bytes were in hand; `record` uses it so
    /// concurrent touches of one path resolve to the newest
    /// observation regardless of the order their inserts land.
    pub observed: u64,

    /// Content hash of the observed bytes.
    ///
    /// Process-local fingerprint (see `content_hash`); compared
    /// against the target's current content at the next write.
    pub hash: u64,

    /// Whether this observation comes from re-arming on resume rather
    /// than a live read in this session.
    ///
    /// A resumed session re-reads the files its transcript shows being
    /// read, but the model never saw those bytes live — the transcript
    /// carries previews only — so the write guard cannot treat the
    /// re-read as the model's knowledge: an edit made while the
    /// session was inactive would otherwise pass a hash compare it was
    /// never entitled to. A marked baseline makes the guard refuse the
    /// first write to the file and direct the model to read it first;
    /// any live observation (a read, a successful write) records an
    /// unmarked baseline and supersedes the marked one.
    pub resumed: bool,
}

/// The process-local content hash of `bytes`.
///
/// Deliberately not a stable or cryptographic digest (see
/// [`FileBaselines`]): the hash only ever answers, within one
/// session, "did the bytes change since the model last touched this
/// file".
pub(crate) fn content_hash(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Observe `bytes`: stamp the session's touch order and hash the content.
///
/// Call this at the moment the bytes are in hand — the stamp is what
/// makes the baseline the *newest* observation rather than merely the
/// last insert. The observation is unmarked: a live read or a
/// successful write made it, so the write guard trusts it as the
/// model's knowledge.
pub(crate) fn observe_bytes(bytes: &[u8]) -> FileBaseline {
    FileBaseline {
        observed: OBSERVATION_SEQ.fetch_add(1, Ordering::Relaxed),
        hash: content_hash(bytes),
        resumed: false,
    }
}

/// Observe `bytes` the way the resume re-arm path does: stamped and hashed
/// like a live observation, but marked as resume-armed.
///
/// The resume path holds the file's current bytes, not anything the
/// model saw in this session, so the baseline it records carries
/// [`FileBaseline::resumed`] — the write guard refuses the first write
/// to such a file until a live read records an unmarked baseline.
pub(crate) fn observe_resumed_bytes(bytes: &[u8]) -> FileBaseline {
    FileBaseline {
        observed: OBSERVATION_SEQ.fetch_add(1, Ordering::Relaxed),
        hash: content_hash(bytes),
        resumed: true,
    }
}

/// Record an observation as the model's latest known state of `path`.
///
/// An observation with a lower sequence than the recorded one arrived
/// out of order (concurrent touches of one path) and is discarded, so
/// the entry always reflects the newest observation.
pub(crate) fn record(baselines: &mut FileBaselines, path: &Path, baseline: FileBaseline) {
    match baselines.get(path) {
        Some(existing) if existing.observed > baseline.observed => return,
        _ => {}
    }
    baselines.insert(path.to_path_buf(), baseline);
}

/// The observation recorded for `path`, if the path was touched.
///
/// The stamp and hash together, for callers that consult both indexes
/// and must hold whichever observation is newer.
pub(crate) fn entry(baselines: &FileBaselines, path: &Path) -> Option<FileBaseline> {
    baselines.get(path).copied()
}

/// The model's latest known content per live file identity (unix device and
/// inode).
///
/// The unrestricted policy's second index over the same observations:
/// keys by path cannot unify aliases that canonicalization cannot see
/// — two hard links to one file are two equally canonical spellings —
/// so the recorded file's stat identity is kept alongside the path
/// key, and a lookup stats the target to find the baseline whichever
/// spelling arrives. Kept in step with [`FileBaselines`] by the same
/// record calls. A lookup consults both indexes and holds whichever
/// of the two entries carries the newer observation stamp: the path
/// entry of one alias spelling can be older than the identity entry,
/// and letting it win would judge the file against content the model
/// has since superseded.
///
/// Residual, mirroring the identity-gate field docs on the write
/// path: an externally deleted-and-recreated file that reclaims the
/// recorded device-inode pair reads as an identity match, arming a
/// staleness guard for a file the model never touched. The direction
/// is safe — a false *refusal*, recovered by re-reading, never a
/// false pass, because the guard passes only when the current bytes
/// hash-match the newest observation the model actually made of this
/// file.
pub(crate) type FileIdentities = BTreeMap<(u64, u64), FileBaseline>;

/// Record an observation as the model's latest known state of the file
/// `identity` names.
///
/// Mirrors [`record`]'s newest-observation-wins semantics with one
/// entry per live identity: a later observation of the same file
/// through any spelling supersedes the earlier one, so the entry
/// always reflects the newest content the model is known to have
/// produced for that file, however it is spelled. Called alongside
/// [`record`] by the unrestricted policy's record path.
pub(crate) fn record_identity(
    identities: &mut FileIdentities,
    identity: (u64, u64),
    baseline: FileBaseline,
) {
    match identities.get(&identity) {
        Some(existing) if existing.observed > baseline.observed => return,
        _ => {}
    }
    identities.insert(identity, baseline);
}

/// The observation recorded for the file `identity` names, if any.
///
/// The identity-side counterpart of [`entry`]: carries the stamp, so
/// a caller consulting both indexes can hold the newer of the two
/// observations instead of letting a stale path entry shadow it.
pub(crate) fn entry_identity(
    identities: &FileIdentities,
    identity: (u64, u64),
) -> Option<FileBaseline> {
    identities.get(&identity).copied()
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]
mod tests {
    use super::*;

    #[test]
    fn baseline_follows_the_latest_recorded_touch() {
        let mut baselines = FileBaselines::default();
        let a = Path::new("a.rs");
        let b = Path::new("b.rs");
        record(&mut baselines, a, observe_bytes(b"one"));
        record(&mut baselines, b, observe_bytes(b"two"));
        record(&mut baselines, a, observe_bytes(b"three"));

        assert_eq!(
            entry(&baselines, a).map(|obs| obs.hash),
            Some(content_hash(b"three")),
            "the newest observation of a path must win, whichever order the inserts land"
        );
    }

    #[test]
    fn an_out_of_order_touch_never_supersedes_a_newer_observation() {
        let mut baselines = FileBaselines::default();
        let path = Path::new("a.rs");
        let newer = observe_bytes(b"newer");
        let older = FileBaseline {
            observed: newer.observed.saturating_sub(1),
            hash: content_hash(b"older"),
            resumed: false,
        };
        record(&mut baselines, path, newer);
        record(&mut baselines, path, older);

        assert_eq!(
            entry(&baselines, path).map(|obs| obs.hash),
            Some(newer.hash),
            "a stamped-older observation landing later must be discarded"
        );
    }

    #[test]
    fn content_hash_is_stable_per_content_and_distinct_across_contents() {
        assert_eq!(
            content_hash(b"same"),
            content_hash(b"same"),
            "identical bytes must hash identically within a session"
        );
        assert_ne!(
            content_hash(b"one"),
            content_hash(b"two"),
            "different bytes must produce different fingerprints"
        );
    }

    #[test]
    fn resumed_observations_are_marked_and_live_ones_are_not() {
        assert!(
            observe_resumed_bytes(b"x").resumed,
            "the resume re-arm path must mark its observations"
        );
        assert!(
            !observe_bytes(b"x").resumed,
            "a live observation must carry no resume marker"
        );
    }

    #[test]
    fn identity_entries_hold_the_newest_observation() {
        let mut identities = FileIdentities::default();
        let key = (1, 2);
        let newer = observe_bytes(b"newer");
        let older = FileBaseline {
            observed: newer.observed.saturating_sub(1),
            hash: content_hash(b"older"),
            resumed: false,
        };
        record_identity(&mut identities, key, newer);
        record_identity(&mut identities, key, older);

        assert_eq!(
            entry_identity(&identities, key).map(|obs| obs.hash),
            Some(newer.hash),
            "the identity index must follow the same newest-wins rule as paths"
        );
    }
}
