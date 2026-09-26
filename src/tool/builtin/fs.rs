//! The filesystem tool family: shared session state and the validation seam.
//!
//! The family — `Write`, `Edit`, `MultiEdit`, `FileViewer`, and the
//! filesystem [`FileSource`] the shared `read`
//! tool reads through — is wired together by one [`FileSession`]:
//! the working directory every path resolves against, the containment
//! policy, the pinned workspace anchor, and the map of the model's
//! latest known content per touched file that powers the write tools'
//! detect-on-write staleness guard. A host constructs one session,
//! [`attaches`](FileSession::attach) it to every
//! [`ToolContext`], and hands the same clone
//! to [`FileSource`]; clones share their
//! mutable state, so a read through one member is knowledge every
//! write against the same file consults.
//!
//! Syntax validation before a write is a host policy, not a library
//! one: the write tools accept an optional
//! [`ContentValidator`] whose diagnostics block the write with the
//! family's shared refusal text, and the model-facing `skip_linter`
//! flag bypasses whatever validator is installed.
//!
//! # Example
//!
//! ```
//! use loopctl::tool::ToolRegistry;
//! use loopctl::tool::builtin::ReadTool;
//! use loopctl::tool::builtin::fs::FileSession;
//! use loopctl::tool::builtin::fs::FileSource;
//! use loopctl::tool::builtin::fs::WriteTool;
//!
//! let session = FileSession::new(".".into());
//! let mut registry = ToolRegistry::new();
//! registry.register(WriteTool::new());
//! registry.register(ReadTool::new(FileSource::new(session)));
//! assert!(registry.contains("Write"));
//! assert!(registry.contains("read"));
//! ```

mod atomic;
mod conflict;
mod diff;
mod edit;
mod file_source;
mod file_viewer;
mod multi_edit;
mod resolve;
mod state;
mod write;

pub use edit::EditTool;
pub use file_source::FileSource;
pub use file_viewer::FileViewerTool;
pub use multi_edit::MultiEditTool;
pub use resolve::ResolvePolicy;
pub use state::FileBaseline;
pub use write::WriteTool;

use std::fmt;
use std::future::Future;
use std::io::Read as _;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use crate::tool::ToolContext;
use crate::tool::ToolError;

use atomic::WorkspaceAnchor;
use resolve::ResolvePolicy as Policy;
use state::FileBaselines;
use state::FileIdentities;

/// One finding a [`ContentValidator`] reports against candidate content.
///
/// The line number is optional because not every validator can attribute
/// a finding to a line; the write tools render whatever is present in the
/// family's shared refusal text, so a validator that locates errors gives
/// the model something actionable to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationDiagnostic {
    /// The 1-indexed line the finding belongs to, when known.
    ///
    /// Rendered as a `line N:` prefix on the diagnostic's row; omitted
    /// cleanly when `None`.
    pub line: Option<usize>,

    /// The finding's human-readable message.
    ///
    /// One row per diagnostic in the refusal text, indented under the
    /// header naming the file.
    pub message: String,
}

/// A host-supplied syntax gate the write tools consult before persisting.
///
/// The family ships no validators: what counts as valid content is a
/// host policy (a linter, a schema check, a compile probe), and the
/// library stays neutral. A validator is installed per tool through
/// `with_validator`; an empty diagnostics vector passes the content,
/// any finding blocks the write before any byte moves, and the
/// model-facing `skip_linter` flag bypasses the gate for that one
/// call. Async by boxed future so the trait stays object-safe,
/// matching the house pattern used by
/// [`LoopMemory`](crate::memory::LoopMemory).
pub trait ContentValidator: Send + Sync {
    /// Validate `content` as the proposed new content of `path`.
    ///
    /// Returns every finding; an empty vector approves the write.
    fn validate<'a>(
        &'a self,
        path: &'a Path,
        content: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<ValidationDiagnostic>> + Send + 'a>>;
}

impl<T: ContentValidator + ?Sized> ContentValidator for Arc<T> {
    fn validate<'a>(
        &'a self,
        path: &'a Path,
        content: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<ValidationDiagnostic>> + Send + 'a>> {
        (**self).validate(path, content)
    }
}

/// The shared filesystem state every tool in the family operates through.
///
/// Carries what a tool invocation needs that is specific to this
/// session: the working directory every relative path resolves against,
/// the path-containment policy, a retained descriptor for the
/// workspace's resolved root, and the model's file-baseline map backing
/// the write tools' staleness check. Stored as a typed extension on the
/// [`ToolContext`] via [`attach`](Self::attach)
/// and retrieved by the tools through [`fs_session`].
///
/// Cloning is cheap — the baseline maps and the workspace anchor are
/// behind `Arc`s, so clones share the same mutable state rather than
/// copying it. This is how the write tools see what a prior read
/// recorded: one session, shared by every tool and by the
/// [`FileSource`] the read tool reads through.
#[derive(Clone)]
pub struct FileSession {
    /// The working directory the family operates within.
    ///
    /// Every tool that touches the filesystem resolves relative paths
    /// against this directory. Set once at construction; a running
    /// agent cannot move it.
    cwd: PathBuf,

    /// The containment policy file tools resolve under.
    ///
    /// Contained is the default; an explicit opt-out at construction
    /// produces unrestricted resolution. Fixed for the session's
    /// lifetime — a running agent cannot widen it mid-run.
    resolve_policy: Policy,

    /// A retained descriptor for the workspace's resolved root, opened
    /// when the session was constructed.
    ///
    /// Contained walks start from a duplicate of this descriptor rather
    /// than reopening the anchor path, so a symlink swapped onto the
    /// workspace spelling after construction cannot redirect them. When
    /// the workspace could not be opened at construction time the
    /// anchor retains no descriptor and contained operations fail
    /// closed. Cloning shares the same descriptor.
    workspace_anchor: Arc<WorkspaceAnchor>,

    /// The model's latest known content hash per touched file.
    ///
    /// The write tools' detect-on-write conflict check compares the
    /// target's current content hash against this record before
    /// overwriting. Concurrent touches of one path resolve to the newest
    /// observation. Cloning shares the same map. Keys are normalized by
    /// [`record_baseline`](Self::record_baseline) and
    /// [`baseline_for`](Self::baseline_for) — never by the tool call
    /// sites — so every spelling of a file meets at one key.
    baselines: Arc<Mutex<FileBaselines>>,

    /// The model's latest known content per live file identity (unix
    /// device and inode).
    ///
    /// The unrestricted policy's second baseline index (see
    /// [`FileIdentities`](state::FileIdentities)): path keys cannot
    /// unify aliases that canonicalization cannot see — two hard links
    /// to one file are two equally canonical spellings — so records and
    /// lookups carry the file's stat identity alongside the path, and
    /// the staleness guard holds whichever spelling of a file arrives.
    /// Only populated and consulted under the unrestricted policy;
    /// contained keys are the lexical resolution output by design.
    /// Cloning shares the same map.
    identities: Arc<Mutex<FileIdentities>>,
}

impl FileSession {
    /// Create a session for `cwd` with no recorded baselines and contained
    /// path resolution.
    ///
    /// A relative `cwd` is anchored to the process's current directory.
    /// Containment decides by comparing lexical prefixes, and a bare `.`
    /// normalizes to nothing — left un-anchored, it would reject every
    /// relative target. `.` and `..` are collapsed lexically — a leftover
    /// `..` would break the pinned write's prefix matching against this
    /// path — while symlinks are not resolved, matching the lexical
    /// philosophy applied to targets. On the rare failure of the
    /// current-directory probe, `cwd` is stored as given.
    #[must_use]
    pub fn new(cwd: PathBuf) -> Self {
        let cwd = std::path::absolute(&cwd).unwrap_or(cwd);
        let cwd = resolve::normalize_lexical(&cwd);
        let workspace_anchor = Arc::new(WorkspaceAnchor::pin(&cwd));
        Self {
            cwd,
            resolve_policy: Policy::default(),
            workspace_anchor,
            baselines: Arc::new(Mutex::new(FileBaselines::default())),
            identities: Arc::new(Mutex::new(FileIdentities::default())),
        }
    }

    /// Set the path-containment policy the file tools resolve under.
    ///
    /// Builder-style companion to [`new`](Self::new), used by the host
    /// wiring to lift a configured unsafe-paths switch onto the session.
    /// Contained resolution is the default; only an explicit opt-out
    /// produces [`ResolvePolicy::Unrestricted`].
    #[must_use]
    pub fn with_resolve_policy(mut self, resolve_policy: ResolvePolicy) -> Self {
        self.resolve_policy = resolve_policy;
        self
    }

    /// Install this session on `ctx` as the family's extension.
    ///
    /// The stored value is a clone sharing this session's maps and
    /// anchor, so a context constructed once per dispatch still observes
    /// everything every tool in the family records. Replaces any session
    /// previously installed on the context.
    pub fn attach(&self, ctx: &mut ToolContext) {
        ctx.set_extension(self.clone());
    }

    /// The working directory every relative path resolves against.
    ///
    /// Anchored to an absolute spelling at construction; a relative root
    /// would make containment judgments depend on the process's current
    /// directory.
    #[must_use]
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// The containment policy this session resolves under.
    ///
    /// Contained by default; unrestricted requires the explicit builder
    /// opt-out, mirroring the host-level switch this maps from.
    #[must_use]
    pub fn resolve_policy(&self) -> ResolvePolicy {
        self.resolve_policy
    }

    /// The retained workspace anchor shared by the family.
    ///
    /// Clones share one anchor, so every contained operation of a session is
    /// judged against the same pinned root.
    #[must_use]
    pub(crate) fn anchor(&self) -> &WorkspaceAnchor {
        &self.workspace_anchor
    }

    /// Record an observation as the model's latest known state of `path`.
    ///
    /// `path` is normalized to the map's key form before storing (see
    /// [`baselines`](Self::baselines)), so a record made through a
    /// symlinked-directory spelling of a just-created file lands on the
    /// same key a later lookup through the physical spelling produces.
    /// Thin locking wrapper over [`record`](state::record), which owns
    /// the ordering semantics (newest observation wins; an older one
    /// arriving out of order is discarded).
    pub(crate) fn record_baseline(&self, path: &Path, baseline: FileBaseline) {
        let key = self.baseline_map_key(path);
        if self.resolve_policy == Policy::Unrestricted
            && let Some(identity) = identity_of(path)
        {
            let mut identities = self
                .identities
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state::record_identity(&mut identities, identity, baseline);
        }
        let mut baselines = self
            .baselines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state::record(&mut baselines, &key, baseline);
    }

    /// The baseline recorded for `path`, if the path was touched.
    ///
    /// `path` is normalized with the same rule
    /// [`record_baseline`](Self::record_baseline) applies, so a lookup
    /// through any spelling of a file finds the baseline regardless of
    /// which spelling recorded it. Under [`ResolvePolicy::Unrestricted`]
    /// the lookup consults both indexes — the path key and the file's
    /// stat identity, since hard-link aliases are distinct path keys
    /// over one physical file — and holds whichever entry carries the
    /// newer observation, mirroring the record rule. The entry carries
    /// its [`resumed`](FileBaseline::resumed) marker, which the Write
    /// tool's guard consults alongside the hash.
    pub(crate) fn baseline_for(&self, path: &Path) -> Option<FileBaseline> {
        let key = self.baseline_map_key(path);
        let by_path = {
            let baselines = self
                .baselines
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state::entry(&baselines, &key)
        };
        if self.resolve_policy != Policy::Unrestricted {
            return by_path;
        }
        let by_identity = identity_of(path).and_then(|identity| {
            let identities = self
                .identities
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state::entry_identity(&identities, identity)
        });
        match (by_path, by_identity) {
            (Some(path_entry), Some(identity_entry)) => Some({
                if identity_entry.observed > path_entry.observed {
                    identity_entry
                } else {
                    path_entry
                }
            }),
            (only, None) | (None, only) => only,
        }
    }

    /// Record what a re-read of `file_path`'s current bytes observes,
    /// re-arming the read-before-write guard a resumed session would
    /// otherwise leave disarmed.
    ///
    /// A session restored from a transcript carries its prior reads as
    /// previews only — no bytes, so no baselines — and without this the
    /// guard would treat every previously-read file as never-read. The
    /// path resolves under the session's policy and the file's current
    /// bytes are read — bounded the way the live read path bounds its
    /// reads: regular files at or under the size cap only, at most the
    /// cap's worth of bytes, the opened handle verified against the
    /// pinned workspace, and on the blocking pool so the async executor
    /// is never held. The observation lands marked
    /// [`resumed`](FileBaseline::resumed): the model never saw these
    /// bytes in this session, so the Write tool's guard holds the file
    /// for a fresh live read before its first write. Returns whether the
    /// path resolved and the file could be read and recorded; a missing,
    /// moved, unreadable, out-of-reach, irregular, or over-cap file
    /// records nothing and the guard simply stays disarmed for it.
    pub async fn record_resumed_read(&self, file_path: &str) -> bool {
        if resolve::is_url(file_path) {
            return false;
        }
        let cwd = self.cwd.clone();
        let policy = self.resolve_policy;
        let anchor = Arc::clone(&self.workspace_anchor);
        let spelling = file_path.to_string();
        let read = tokio::task::spawn_blocking(move || {
            read_resumable(&spelling, &cwd, policy, Some(&anchor))
        })
        .await
        .ok()
        .flatten();
        match read {
            Some((full_path, bytes)) => {
                let baseline = state::observe_resumed_bytes(&bytes);
                self.record_baseline(&full_path, baseline);
                true
            }
            None => false,
        }
    }

    /// The baseline map key for `path`, per the session's resolve policy.
    ///
    /// Under [`ResolvePolicy::Contained`] the key is the path as given —
    /// the tools' contained resolution output — so records and lookups
    /// stay in sync without filesystem probes. Under
    /// [`ResolvePolicy::Unrestricted`] the key is the canonicalized
    /// physical file, falling back to the path as given when the probe
    /// fails (a file removed again between its write and this call);
    /// this is what lets a file first created through a symlinked
    /// directory be re-keyed to its now-resolvable referent.
    fn baseline_map_key(&self, path: &Path) -> PathBuf {
        match self.resolve_policy {
            Policy::Contained => path.to_path_buf(),
            Policy::Unrestricted => {
                std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
            }
        }
    }
}

impl fmt::Debug for FileSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let baselines = self
            .baselines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        let identities = self
            .identities
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        f.debug_struct("FileSession")
            .field("cwd", &self.cwd)
            .field("resolve_policy", &self.resolve_policy)
            .field("workspace_anchor", &self.workspace_anchor)
            .field("baselines", &baselines)
            .field("identities", &identities)
            .finish()
    }
}

/// Retrieve the [`FileSession`] the family's tools operate through.
///
/// The session is installed on the context by the host via
/// [`FileSession::attach`]; its absence means a family tool ran outside
/// a host that wired the family, which is a wiring error rather than a
/// model error, so it surfaces as a hard error naming the fix.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] explaining that a `FileSession`
/// must be attached to the context before family tools can run.
pub fn fs_session(ctx: &ToolContext) -> Result<FileSession, ToolError> {
    ctx.get_extension::<FileSession>().cloned().ok_or_else(|| {
        ToolError::Execution(
            "FileSession extension is not installed on the ToolContext; construct one with \
             FileSession::new and attach it before running filesystem tools"
                .to_string(),
        )
    })
}

/// Turn an optional session into a session, naming the missing wiring.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] explaining that a `FileSession`
/// must be attached to the context before family tools can run.
pub(crate) fn require_session(session: Option<FileSession>) -> Result<FileSession, ToolError> {
    session.ok_or_else(|| {
        ToolError::Execution(
            "FileSession extension is not installed on the ToolContext; construct one with \
             FileSession::new and attach it before running filesystem tools"
                .to_string(),
        )
    })
}

/// Resolve `file_path` and read its current bytes when it is a regular
/// file within the live read path's size cap.
///
/// The resume re-arm's bounded read, run on the blocking pool. Metadata
/// comes first: anything other than a regular file (a directory, a
/// device, a FIFO) or a size over the cap reads as nothing, so neither
/// a special file's unbounded stream nor an oversized file's full bytes
/// can enter memory. The open and the read that follow are capped one
/// byte past the limit, so growth between the metadata check and the
/// read cannot turn into an unbounded allocation either. Returns the
/// resolved path with its bytes, or `None` for every refusal case.
/// This function does not return `Result`; every refusal case reads
/// as `None` (see the docs above for the refusal conditions).
fn read_resumable(
    file_path: &str,
    cwd: &Path,
    policy: Policy,
    anchor: Option<&WorkspaceAnchor>,
) -> Option<(PathBuf, Vec<u8>)> {
    let full_path = resolve::resolve_path(file_path, cwd, policy).ok()?;
    let full_path = if policy == Policy::Unrestricted {
        resolve::canonicalize_existing(&full_path).ok()?
    } else {
        full_path
    };
    let metadata = std::fs::metadata(&full_path).ok()?;
    if !metadata.is_file() || metadata.len() > super::read::DEFAULT_MAX_SIZE_BYTES {
        return None;
    }
    let file = std::fs::File::open(&full_path).ok()?;
    if policy == Policy::Contained {
        resolve::verify_handle_inside(&file, cwd, anchor).ok()?;
    }
    let cap = super::read::DEFAULT_MAX_SIZE_BYTES.saturating_add(1);
    let mut bytes = Vec::new();
    file.take(cap).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 > super::read::DEFAULT_MAX_SIZE_BYTES {
        return None;
    }
    Some((full_path, bytes))
}

/// The stat identity (device, inode) of the file `path` reaches, if the
/// platform exposes one and the file exists.
///
/// Metadata follows links, so the identity is the referent's — hard-link
/// aliases, which canonicalization cannot unify, stat to the same pair
/// and share one baseline. `None` covers the two cases where no identity
/// can be recorded and callers fall back to path keys alone: the path
/// does not resolve (nothing exists to guard yet), and the record side
/// simply skips the identity index while the lookup side reports a miss.
#[cfg(unix)]
fn identity_of(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.dev(), meta.ino()))
}

/// The identity of the file `path` reaches, on a platform without a
/// stable stat identity.
///
/// Always `None`: there is nothing to key the identity index with, so
/// records and lookups degrade to path keys alone and the hard-link
/// alias guard is absent — consistent with this platform's other
/// degraded checks, which fail closed where containment is at stake and
/// merely narrow protection where it is not.
#[cfg(not(unix))]
fn identity_of(path: &Path) -> Option<(u64, u64)> {
    let _ = path;
    None
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

    fn ctx_in(cwd: &Path) -> ToolContext {
        let mut ctx = ToolContext::default();
        FileSession::new(cwd.to_path_buf()).attach(&mut ctx);
        ctx
    }

    #[test]
    fn fs_session_returns_the_attached_session() {
        let ctx = ctx_in(Path::new("."));
        assert!(
            fs_session(&ctx).is_ok(),
            "an attached session must be retrievable"
        );
    }

    #[test]
    fn fs_session_without_an_extension_errors_with_an_install_hint() {
        let ctx = ToolContext::default();
        let err = fs_session(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("FileSession"),
            "the error must name what to install: {err}"
        );
        assert!(
            err.to_string().contains("attach"),
            "the error must name the recovery path: {err}"
        );
    }

    #[test]
    fn clones_share_one_baseline_map() {
        let session = FileSession::new(Path::new(".").to_path_buf());
        let other = session.clone();
        session.record_baseline(Path::new("shared.txt"), state::observe_bytes(b"one"));
        assert!(
            other
                .baseline_for(Path::new("shared.txt"))
                .is_some_and(|baseline| baseline.hash == state::content_hash(b"one")),
            "a clone must observe what its original recorded"
        );
    }

    #[test]
    fn an_attached_context_sees_what_the_session_records() {
        let ctx = ctx_in(Path::new("."));
        let session = fs_session(&ctx).unwrap();
        session.record_baseline(Path::new("seen.txt"), state::observe_bytes(b"x"));
        assert!(
            session.baseline_for(Path::new("seen.txt")).is_some(),
            "the extension must share the session's map, not copy it"
        );
    }

    #[tokio::test]
    async fn a_resume_re_arm_is_marked_and_a_live_read_supersedes_it() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("note.txt"), b"v2").unwrap();

        let session = FileSession::new(tmp.path().to_path_buf());
        assert!(session.record_resumed_read("note.txt").await);
        assert!(
            session
                .baseline_for(&tmp.path().join("note.txt"))
                .is_some_and(|baseline| baseline.resumed),
            "a resume re-arm must not pose as the model's live knowledge"
        );

        session.record_baseline(&tmp.path().join("note.txt"), state::observe_bytes(b"v2"));
        assert!(
            session
                .baseline_for(&tmp.path().join("note.txt"))
                .is_some_and(|baseline| !baseline.resumed),
            "a live observation supersedes the marker"
        );
    }

    #[tokio::test]
    async fn record_resumed_read_skips_files_over_the_live_read_cap() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("huge.bin"),
            vec![
                b'x';
                usize::try_from(super::super::read::DEFAULT_MAX_SIZE_BYTES)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1)
            ],
        )
        .unwrap();

        let session = FileSession::new(tmp.path().to_path_buf());
        assert!(
            !session.record_resumed_read("huge.bin").await,
            "an over-cap file records nothing"
        );
        assert!(
            session.baseline_for(&tmp.path().join("huge.bin")).is_none(),
            "no baseline, no guard arming"
        );
    }

    #[tokio::test]
    async fn record_resumed_read_skips_directories() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("adir")).unwrap();

        let session = FileSession::new(tmp.path().to_path_buf());
        assert!(
            !session.record_resumed_read("adir").await,
            "a directory is not a readable baseline"
        );
    }

    #[cfg(unix)]
    #[test]
    fn baseline_lookup_takes_the_newest_across_aliases_in_both_insert_orders() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real.txt");
        let hard = tmp.path().join("hard.txt");
        std::fs::write(&real, b"v1").unwrap();
        std::fs::hard_link(&real, &hard).unwrap();

        let older = state::observe_bytes(b"v1");
        let newer = state::observe_bytes(b"EXT");

        let session = FileSession::new(tmp.path().to_path_buf())
            .with_resolve_policy(ResolvePolicy::Unrestricted);
        session.record_baseline(&hard, older);
        session.record_baseline(&real, newer);
        assert_eq!(
            session.baseline_for(&hard).map(|baseline| baseline.hash),
            Some(newer.hash),
            "newer through the physical spelling must win on the alias"
        );

        let session = FileSession::new(tmp.path().to_path_buf())
            .with_resolve_policy(ResolvePolicy::Unrestricted);
        session.record_baseline(&real, older);
        session.record_baseline(&hard, newer);
        assert_eq!(
            session.baseline_for(&real).map(|baseline| baseline.hash),
            Some(newer.hash),
            "newer through the alias must win on the physical spelling"
        );
    }
}
