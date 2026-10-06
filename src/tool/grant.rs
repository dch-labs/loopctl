//! Persisted interactive grants — the approval ledger behind "always allow".
//!
//! The store half of the TCC model: when a user answers an approval
//! ask with a persist option, the narrowest rule covering what was
//! approved lands here as a [`Grant`], project-scoped and revocable;
//! a later run consults the store before asking again. The store
//! persists denies exactly as it persists allows, pins the
//! [`descriptor_digest`](crate::tool::Tool::descriptor_digest) the
//! user actually approved (a mismatch is the re-ask signal — the
//! tool's contract moved), and returns the affected grant from
//! [`GrantStore::save`] and [`GrantStore::revoke`] so the host mints
//! the matching gate-decision audit record.
//!
//! Matching is mechanical and owned by the data — [`Grant::covers`].
//! Suggesting the narrowest rule for an approval, and deny-over-allow
//! precedence, are consumer policy.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::compact::demote::fnv1a64;
use crate::error::LoopError;
use crate::tool::canonical_json;

/// The narrowness ladder a persisted grant covers.
///
/// The approver picks the rung; the store only carries it. Matching
/// is [`covers`](Self::covers) — mechanical, canonical, and shared by
/// every consumer so one grant means one rule everywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantScope {
    /// Every call to the grant's tool.
    ///
    /// The widest rung worth persisting: any argument set dispatches
    /// under this grant, so it claims exactly as much as its name
    /// says — the tool, whole.
    ThisTool,

    /// One exact argument set.
    ///
    /// The narrowest rung: a call is covered only when its arguments
    /// are canonically equal to `args` — same value, any key order.
    /// The approver's "just this once, persisted" answer.
    ThisCall {
        /// The argument set the grant covers.
        ///
        /// Compared by canonical-JSON equality, so key order in the
        /// stored value and in the live call never matters.
        args: Value,
    },
}

impl GrantScope {
    /// Whether a call's arguments fall under this scope.
    ///
    /// [`ThisTool`](Self::ThisTool) covers any arguments;
    /// [`ThisCall`](Self::ThisCall) covers exactly the argument sets
    /// that are canonically equal to the stored one. The tool name is
    /// matched by [`Grant::covers`](crate::tool::Grant::covers) —
    /// this method answers for the arguments alone.
    #[must_use]
    pub fn covers(&self, args: &Value) -> bool {
        match self {
            Self::ThisTool => true,
            Self::ThisCall { args: stored } => canonical_json(stored) == canonical_json(args),
        }
    }
}

/// What a persisted grant decides.
///
/// The store keeps both verdicts with identical mechanics — a
/// persisted "never ask again, just refuse" is the same ledger entry
/// with the opposite meaning, and both are revocable the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantVerdict {
    /// Calls under the scope are pre-approved.
    ///
    /// The persisted "always allow": dispatch under this grant skips
    /// the ask.
    Allow,

    /// Calls under the scope are pre-refused.
    ///
    /// The persisted "never ask again, just refuse" — the same
    /// ledger mechanics with the opposite meaning.
    Deny,
}

/// One persisted interactive grant.
///
/// Construct through [`new`](Self::new) (which mints the
/// content-stable `id`) and the `with_*` builders; the identity is
/// the `{tool, verdict, scope}` triple, so re-granting the same rule
/// — including re-approving after a descriptor change, which only
/// moves [`pinned_descriptor`](Self::pinned_descriptor) — upserts
/// onto the same id instead of accumulating duplicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grant {
    /// The grant's content-stable identity.
    ///
    /// Sixteen hex characters: the FNV-1a 64-bit hash of the
    /// canonical JSON over `{tool, verdict, scope}` — the digest
    /// family every persisted pin in this crate uses. Stable across
    /// constructions and releases, so a re-grant finds its own entry.
    pub id: String,

    /// The tool the grant names.
    ///
    /// The model-facing tool name, matched exactly against the call.
    pub tool: String,

    /// What the grant decides.
    ///
    /// See [`GrantVerdict`]; the verdict changes the identity, so an
    /// allow and a deny for the same scope coexist as two entries.
    pub verdict: GrantVerdict,

    /// How much the grant covers.
    ///
    /// See [`GrantScope`]; the scope is part of the identity.
    pub scope: GrantScope,

    /// The descriptor digest pinned at approval.
    ///
    /// [`Tool::descriptor_digest`](crate::tool::Tool::descriptor_digest)
    /// at the moment the user approved; a later mismatch means the
    /// tool's contract moved and the approval no longer speaks for
    /// it — the re-ask signal. `None` when the grant predates
    /// pinning or the approver chose not to pin.
    pub pinned_descriptor: Option<String>,

    /// When the grant was persisted, in epoch milliseconds.
    ///
    /// Caller-supplied — the store reads no clock; hosts pass their
    /// clock seam's reading so replayed sessions stamp consistently.
    pub created_at_ms: u64,

    /// Free-form provenance for the `show` view.
    ///
    /// Whatever the approving surface wants remembered — the session,
    /// the ask's reason, the origin of a deny.
    pub note: Option<String>,
}

impl Grant {
    /// Mint a grant over the identity triple.
    ///
    /// Computes [`id`](Self::id) from `{tool, verdict, scope}` and
    /// leaves the metadata builders at their defaults; every field
    /// stays public for serialization, but construction through here
    /// keeps the id honest.
    #[must_use]
    pub fn new(tool: impl Into<String>, verdict: GrantVerdict, scope: GrantScope) -> Self {
        let tool = tool.into();
        Self {
            id: Self::id_of(&tool, verdict, &scope),
            tool,
            verdict,
            scope,
            pinned_descriptor: None,
            created_at_ms: 0,
            note: None,
        }
    }

    /// The identity digest for a `{tool, verdict, scope}` triple.
    ///
    /// Canonical JSON over the triple, hashed with the crate's
    /// FNV-1a 64 — the fixed algorithm every persisted pin uses, so
    /// ids never churn across releases.
    fn id_of(tool: &str, verdict: GrantVerdict, scope: &GrantScope) -> String {
        let identity = serde_json::json!({
            "tool": tool,
            "verdict": verdict,
            "scope": scope,
        });
        format!("{:016x}", fnv1a64(canonical_json(&identity).as_bytes()))
    }

    /// Pin the descriptor digest the approval covers.
    ///
    /// Metadata, not identity: re-approving after a descriptor
    /// change moves the pin onto the same grant's entry.
    #[must_use]
    pub fn with_pinned_descriptor(mut self, digest: impl Into<String>) -> Self {
        self.pinned_descriptor = Some(digest.into());
        self
    }

    /// Stamp the persistence time.
    ///
    /// Epoch milliseconds from the caller's clock seam — the store
    /// itself reads no clock.
    #[must_use]
    pub fn with_created_at_ms(mut self, created_at_ms: u64) -> Self {
        self.created_at_ms = created_at_ms;
        self
    }

    /// Attach the `show`-view provenance note.
    ///
    /// Free-form text the approving surface wants remembered
    /// alongside the grant.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Whether this grant covers the named call.
    ///
    /// The tool name must match exactly and the arguments must fall
    /// under the scope — verdict-agnostic, so the same question
    /// serves allow-lookups and deny-lookups; precedence between
    /// covering grants stays consumer policy.
    #[must_use]
    pub fn covers(&self, tool: &str, args: &Value) -> bool {
        self.tool == tool && self.scope.covers(args)
    }
}

/// The persisted-grant seam: load, save, revoke, clear.
///
/// Sync by design — grants are tiny records read at startup and
/// written at approval, pure filesystem work behind whichever impl a
/// host installs (the shipped one is [`FileGrantStore`], behind the
/// `file_grants` feature). Object-safe so hosts hold one
/// `Arc<dyn GrantStore>` for every consumer; the store stores and
/// [`Grant::covers`] matches, never the reverse.
pub trait GrantStore: Send + Sync {
    /// Every persisted grant, in file order.
    ///
    /// An empty store is an empty list, not an error.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the backing storage
    /// cannot be read or its contents cannot be parsed as grants.
    fn load(&self) -> Result<Vec<Grant>, LoopError>;

    /// Persist (or update) one grant.
    ///
    /// An upsert by [`id`](Grant::id): re-granting the same
    /// `{tool, verdict, scope}` replaces the entry — the pinned
    /// descriptor and note move, the identity does not. Returns the
    /// stored grant so the host mints the grant audit record.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the backing storage
    /// cannot be read or written.
    fn save(&self, grant: &Grant) -> Result<Grant, LoopError>;

    /// Remove one grant by id.
    ///
    /// Returns the removed grant for the revoke audit record, or
    /// `None` when no such id exists (a no-op, not an error).
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the backing storage
    /// cannot be read or written.
    fn revoke(&self, id: &str) -> Result<Option<Grant>, LoopError>;

    /// Remove every grant, returning how many there were.
    ///
    /// The `reset` verb: the whole project ledger goes at once.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the backing storage
    /// cannot be read or removed.
    fn clear(&self) -> Result<usize, LoopError>;
}

/// The file-backed [`GrantStore`], one project per directory.
///
/// Construct with a root directory and the project path the grants
/// bind to: grants live in `<root>/projects/<identity>/grants.json`,
/// where `<identity>` is the 16-hex digest of the project path's
/// stable spelling (see [`new`](Self::new) — the binding survives the
/// project directory's creation in the common cases) — one project's
/// grants are invisible to every other project over the same root,
/// and the layout the CLI documents as its reference
/// (`~/.local/state/<app>/grants/`) is the same shape. The ledger is
/// created owner-only (`0600`, unix; best-effort, like the ledger
/// writers) and replaced atomically on every write, staged under a
/// per-process name beside the ledger so two processes over one
/// project never collide on the staging file itself. Concurrency is
/// single-writer: `save`, `revoke`, and `clear` are read-modify-
/// writes under a one-process-per-project contract — concurrent
/// processes over one project are unsupported, and the last replace
/// silently wins, which can lose a grant or a revocation.
///
/// Requires the `file_grants` feature.
#[cfg(feature = "file_grants")]
#[derive(Debug, Clone)]
pub struct FileGrantStore {
    /// The grants file this instance reads and writes.
    ///
    /// Fixed at construction by the root + project binding; every
    /// method touches exactly this path.
    path: std::path::PathBuf,
}

#[cfg(feature = "file_grants")]
impl FileGrantStore {
    /// Bind a store to one project under one root.
    ///
    /// The project's identity-bearing spelling is resolved
    /// best-effort, most-canonical first: the canonicalized project
    /// when it resolves; else its canonicalized parent with the final
    /// component joined back — the path the project will carry once
    /// created; else the lexically absolute path (anchored at the
    /// working directory, `..` not normalized); else the path as
    /// given. A project bound before its directory exists therefore
    /// selects the same ledger after creation in the common cases,
    /// and once the directory exists the first rung always wins, so
    /// the identity never moves.
    #[must_use]
    pub fn new(root: impl AsRef<std::path::Path>, project: impl AsRef<std::path::Path>) -> Self {
        let stable = stable_project_identity(project.as_ref());
        let identity = format!("{:016x}", fnv1a64(stable.to_string_lossy().as_bytes()));
        Self {
            path: root
                .as_ref()
                .join("projects")
                .join(identity)
                .join("grants.json"),
        }
    }

    /// The per-process staging path for the atomic replace.
    ///
    /// `grants.json.<process id>.staged` beside the ledger: unique
    /// across processes over one project, stable within this process,
    /// so a crashed write leaves at most one orphan and the next
    /// write truncates and reuses it rather than accumulating.
    fn staged_path(&self) -> std::path::PathBuf {
        self.path
            .with_extension(format!("json.{}.staged", std::process::id()))
    }

    /// Read the ledger, treating a missing file as empty.
    ///
    /// A fresh project has no file yet — absence is the empty
    /// ledger, not an error; only an unreadable or unparseable file
    /// fails.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the ledger cannot be read
    /// or its contents do not parse as a grant list.
    fn read(&self) -> Result<Vec<Grant>, LoopError> {
        let contents = match std::fs::read_to_string(&self.path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Vec::new());
            }
            Err(error) => {
                return Err(LoopError::Internal(format!(
                    "cannot read the grant ledger {}: {error}",
                    self.path.display()
                )));
            }
        };
        serde_json::from_str(&contents).map_err(|error| {
            LoopError::Internal(format!(
                "the grant ledger {} is not parseable: {error}",
                self.path.display()
            ))
        })
    }

    /// Replace the ledger, creating it owner-only and atomically.
    ///
    /// The write stages under this process's name beside the ledger
    /// and renames over the original, so a reader never sees a
    /// half-written file and a crash leaves either the old or the new
    /// ledger whole.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Internal`] when the directory cannot be
    /// created, the ledger cannot be serialized, or either file
    /// operation fails.
    fn write(&self, grants: &[Grant]) -> Result<(), LoopError> {
        let parent = self
            .path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        std::fs::create_dir_all(parent).map_err(|error| {
            LoopError::Internal(format!(
                "cannot create the grant directory {}: {error}",
                parent.display()
            ))
        })?;
        let staged = self.staged_path();
        let contents = serde_json::to_string(grants).map_err(|error| {
            LoopError::Internal(format!("cannot serialize the grant ledger: {error}"))
        })?;
        write_owner_only(&staged, &contents)?;
        std::fs::rename(&staged, &self.path).map_err(|error| {
            if let Err(cleanup) = std::fs::remove_file(&staged) {
                tracing::debug!(
                    error = %cleanup,
                    "staged grant ledger left behind after a failed replace"
                );
            }
            LoopError::Internal(format!(
                "cannot replace the grant ledger {}: {error}",
                self.path.display()
            ))
        })
    }
}

/// Write `contents` to `path` owner-only, truncating.
///
/// The staged-write half of the atomic replace: the file appears at
/// `0600` on unix — including a staged file left behind by an earlier
/// crashed write, whose mode is reset on open — best-effort in both
/// directions (a permissions failure falls back to a plain write
/// rather than losing the ledger), and with platform defaults
/// elsewhere.
///
/// # Errors
///
/// Returns [`LoopError::Internal`] when the file cannot be created
/// or written.
#[cfg(feature = "file_grants")]
fn write_owner_only(path: &std::path::Path, contents: &str) -> Result<(), LoopError> {
    #[cfg(unix)]
    let result = {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .or_else(|_| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)
            })
            .map_err(|error| {
                LoopError::Internal(format!(
                    "cannot create the staged grant ledger {}: {error}",
                    path.display()
                ))
            })?;
        reset_owner_only(&file);
        file.write_all(contents.as_bytes())
    };
    #[cfg(not(unix))]
    let result = std::fs::write(path, contents);
    result.map_err(|error| {
        LoopError::Internal(format!(
            "cannot write the staged grant ledger {}: {error}",
            path.display()
        ))
    })
}

/// Reset an opened staged file to owner-only, best-effort.
///
/// The open's `0600` mode applies only at creation; a staged file
/// left behind by a crashed write keeps the mode it died with, and
/// the rename would carry that mode onto the ledger. The reset closes
/// that hole; a failure is logged and the write proceeds — the
/// best-effort contract [`write_owner_only`] documents.
#[cfg(all(feature = "file_grants", unix))]
fn reset_owner_only(file: &std::fs::File) {
    use std::os::unix::fs::PermissionsExt;
    let permissions = std::fs::Permissions::from_mode(0o600);
    if let Err(error) = file.set_permissions(permissions) {
        tracing::debug!(
            error = %error,
            "could not reset the staged grant ledger to owner-only"
        );
    }
}

/// The identity-bearing spelling of a project path.
///
/// The ladder, most-canonical first: the canonicalized project when
/// it resolves; else the canonicalized parent with the final
/// component joined back — the spelling the project carries once
/// created; else the lexically absolute path; else the path as given.
/// Best-effort by design: the rung chosen for a not-yet-existing
/// project matches its canonical form after creation in the common
/// cases, and an existing project always takes the first rung, so a
/// bound identity never moves.
#[cfg(feature = "file_grants")]
fn stable_project_identity(project: &std::path::Path) -> std::path::PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(project) {
        return canonical;
    }
    let joined = project
        .parent()
        .and_then(|parent| std::fs::canonicalize(parent).ok())
        .zip(project.file_name())
        .map(|(parent, name)| parent.join(name));
    if let Some(joined) = joined {
        return joined;
    }
    std::path::absolute(project).unwrap_or_else(|_| project.to_path_buf())
}

#[cfg(feature = "file_grants")]
impl GrantStore for FileGrantStore {
    fn load(&self) -> Result<Vec<Grant>, LoopError> {
        self.read()
    }

    fn save(&self, grant: &Grant) -> Result<Grant, LoopError> {
        let mut grants = self.read()?;
        match grants.iter_mut().find(|entry| entry.id == grant.id) {
            Some(entry) => *entry = grant.clone(),
            None => grants.push(grant.clone()),
        }
        self.write(&grants)?;
        Ok(grant.clone())
    }

    fn revoke(&self, id: &str) -> Result<Option<Grant>, LoopError> {
        let grants = self.read()?;
        let (removed, remaining): (Vec<Grant>, Vec<Grant>) =
            grants.into_iter().partition(|entry| entry.id == id);
        if removed.is_empty() {
            return Ok(None);
        }
        self.write(&remaining)?;
        Ok(removed.into_iter().next())
    }

    fn clear(&self) -> Result<usize, LoopError> {
        let count = self.read()?.len();
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(count),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(LoopError::Internal(format!(
                "cannot remove the grant ledger {}: {error}",
                self.path.display()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn grant_scope_matching_is_canonical_not_key_order() {
        let one_call = Grant::new(
            "deploy",
            GrantVerdict::Allow,
            GrantScope::ThisCall {
                args: json!({ "b": 1, "a": 2 }),
            },
        );
        assert!(
            one_call.covers("deploy", &json!({ "a": 2, "b": 1 })),
            "key order never decides coverage — canonical equality does"
        );
        assert!(
            !one_call.covers("deploy", &json!({ "a": 2 })),
            "a smaller argument set is a different call"
        );
        assert!(
            !one_call.covers("deploy", &json!({ "a": 2, "b": 3 })),
            "a changed value is a different call"
        );
        assert!(
            !one_call.covers("other", &json!({ "a": 2, "b": 1 })),
            "the tool name must match exactly"
        );
        let whole_tool = Grant::new("deploy", GrantVerdict::Allow, GrantScope::ThisTool);
        assert!(
            whole_tool.covers("deploy", &json!({ "anything": true })),
            "ThisTool covers every argument set of its tool"
        );
        assert!(
            !whole_tool.covers("other", &json!({ "anything": true })),
            "ThisTool claims its own tool only"
        );
    }

    #[cfg(feature = "file_grants")]
    mod file_tests {
        use super::*;

        #[test]
        fn grants_are_project_scoped_and_revocable() {
            let root = tempfile::tempdir().expect("temp grant root");
            let alpha = FileGrantStore::new(root.path(), "/projects/alpha");
            let beta = FileGrantStore::new(root.path(), "/projects/beta");
            let allow = Grant::new("shell", GrantVerdict::Allow, GrantScope::ThisTool)
                .with_created_at_ms(1_000);
            let deny = Grant::new(
                "shell",
                GrantVerdict::Deny,
                GrantScope::ThisCall {
                    args: json!({ "command": "rm -rf /" }),
                },
            )
            .with_note("never this one");

            alpha.save(&allow).expect("save the allow");
            let stored_deny = alpha.save(&deny).expect("save the deny");
            assert_eq!(
                stored_deny.verdict,
                GrantVerdict::Deny,
                "the store returns the grant it persisted"
            );
            assert!(
                beta.load().expect("beta loads").is_empty(),
                "a sibling project sees none of alpha's grants"
            );

            let revoked = alpha
                .revoke(&allow.id)
                .expect("revoke reads and writes")
                .expect("the allow was persisted");
            assert_eq!(
                revoked.verdict,
                GrantVerdict::Allow,
                "revoke returns the removed grant for the audit record"
            );
            let remaining = alpha.load().expect("alpha reloads");
            assert_eq!(
                remaining.len(),
                1,
                "only the deny survives the revoke: {remaining:?}"
            );
            assert_eq!(
                remaining[0].verdict,
                GrantVerdict::Deny,
                "the persisted deny round-trips"
            );
            assert!(
                alpha.revoke("no-such-id").expect("absent revoke").is_none(),
                "revoking an absent id is a no-op, not an error"
            );
        }

        #[test]
        fn a_regrant_of_the_same_rule_upserts_idempotently() {
            let root = tempfile::tempdir().expect("temp grant root");
            let store = FileGrantStore::new(root.path(), "/projects/alpha");
            let first = Grant::new("shell", GrantVerdict::Allow, GrantScope::ThisTool)
                .with_pinned_descriptor("aaaaaaaaaaaaaaaa")
                .with_created_at_ms(1);
            let second = Grant::new("shell", GrantVerdict::Allow, GrantScope::ThisTool)
                .with_pinned_descriptor("bbbbbbbbbbbbbbbb")
                .with_created_at_ms(2);
            assert_eq!(
                first.id, second.id,
                "identity is the tool-verdict-scope triple, not the metadata"
            );
            store.save(&first).expect("first save");
            store.save(&second).expect("second save");
            let grants = store.load().expect("reload");
            assert_eq!(
                grants.len(),
                1,
                "the re-grant replaced its own entry: {grants:?}"
            );
            assert_eq!(
                grants[0].pinned_descriptor.as_deref(),
                Some("bbbbbbbbbbbbbbbb"),
                "the newer pin wins the upsert"
            );
            assert_eq!(grants[0].created_at_ms, 2);
        }

        #[test]
        #[cfg(unix)]
        fn a_created_grant_file_is_owner_only() {
            use std::os::unix::fs::PermissionsExt;
            let root = tempfile::tempdir().expect("temp grant root");
            let store = FileGrantStore::new(root.path(), "/projects/alpha");
            store
                .save(&Grant::new(
                    "shell",
                    GrantVerdict::Allow,
                    GrantScope::ThisTool,
                ))
                .expect("save creates the ledger");
            let mode = std::fs::metadata(&store.path)
                .expect("the ledger exists after a save")
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "the grant ledger is created owner-only, like the ledgers"
            );
        }

        #[test]
        fn grant_store_survives_reload() {
            let root = tempfile::tempdir().expect("temp grant root");
            let grant = Grant::new(
                "read",
                GrantVerdict::Allow,
                GrantScope::ThisCall {
                    args: json!({ "path": "a.txt" }),
                },
            )
            .with_pinned_descriptor("feedfacefeedface");
            FileGrantStore::new(root.path(), "/projects/alpha")
                .save(&grant)
                .expect("save");
            let reloaded = FileGrantStore::new(root.path(), "/projects/alpha")
                .load()
                .expect("reload");
            assert_eq!(
                reloaded,
                vec![grant],
                "a fresh store over the same root and project reloads the grant"
            );
        }

        #[test]
        fn clear_removes_every_grant_and_counts_them() {
            let root = tempfile::tempdir().expect("temp grant root");
            let store = FileGrantStore::new(root.path(), "/projects/alpha");
            store
                .save(&Grant::new(
                    "shell",
                    GrantVerdict::Allow,
                    GrantScope::ThisTool,
                ))
                .expect("save one");
            store
                .save(&Grant::new(
                    "read",
                    GrantVerdict::Deny,
                    GrantScope::ThisTool,
                ))
                .expect("save two");
            assert_eq!(
                store.clear().expect("clear"),
                2,
                "clear reports how many grants it removed"
            );
            assert!(
                store.load().expect("reload").is_empty(),
                "the ledger reads empty after clear"
            );
            assert_eq!(
                store.clear().expect("clear again"),
                0,
                "clearing an absent ledger is Ok(0), not an error"
            );
        }

        #[test]
        #[cfg(unix)]
        fn grants_saved_through_a_symlinked_parent_survive_the_directorys_creation() {
            let root = tempfile::tempdir().expect("temp grant root");
            let real_parent = root.path().join("real");
            std::fs::create_dir(&real_parent).expect("create the real parent");
            let linked_parent = root.path().join("link");
            std::os::unix::fs::symlink(&real_parent, &linked_parent).expect("symlink the parent");
            let project = linked_parent.join("alpha");
            let grant = Grant::new("shell", GrantVerdict::Allow, GrantScope::ThisTool)
                .with_pinned_descriptor("feedfacefeedface");

            FileGrantStore::new(root.path(), &project)
                .save(&grant)
                .expect("save before the project directory exists");
            std::fs::create_dir(real_parent.join("alpha"))
                .expect("create the project directory at its real location");

            let reloaded = FileGrantStore::new(root.path(), &project)
                .load()
                .expect("load after the project directory exists");
            assert_eq!(
                reloaded,
                vec![grant],
                "one project is one identity whether bound before or after \
                 its directory exists — the ledger must not move"
            );
        }

        #[test]
        #[cfg(unix)]
        fn a_save_over_a_preexisting_wide_staged_file_lands_the_ledger_owner_only() {
            use std::os::unix::fs::PermissionsExt;
            let root = tempfile::tempdir().expect("temp grant root");
            let store = FileGrantStore::new(root.path(), "/projects/alpha");
            let staged = store.staged_path();
            std::fs::create_dir_all(
                store
                    .path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .expect("create the project directory");
            std::fs::write(&staged, b"stale staged debris").expect("seed the staged file");
            std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o644))
                .expect("widen the staged file");

            store
                .save(&Grant::new(
                    "shell",
                    GrantVerdict::Allow,
                    GrantScope::ThisTool,
                ))
                .expect("save over the preexisting staged file");

            let mode = std::fs::metadata(&store.path)
                .expect("the ledger exists after the save")
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "a staged file left wide by an earlier write must not carry \
                 its mode onto the ledger through the rename"
            );
        }

        #[test]
        fn a_save_ignores_staged_debris_beside_the_ledger() {
            let root = tempfile::tempdir().expect("temp grant root");
            let store = FileGrantStore::new(root.path(), "/projects/alpha");
            let debris = root
                .path()
                .join("projects")
                .join(format!("{:016x}", fnv1a64(b"/projects/alpha")))
                .join("grants.json.staged");
            std::fs::create_dir_all(
                store
                    .path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
            .expect("create the project directory");
            std::fs::write(&debris, "not json at all").expect("seed foreign staged debris");

            let grant = Grant::new("read", GrantVerdict::Allow, GrantScope::ThisTool);
            store.save(&grant).expect("save beside the debris");

            assert_eq!(
                store.load().expect("reload"),
                vec![grant],
                "staging never depends on prior staged state — foreign \
                 debris beside the ledger changes nothing"
            );
            assert_eq!(
                std::fs::read_to_string(&debris).expect("the debris survives"),
                "not json at all",
                "the debris is neither read nor reused"
            );
        }
    }
}
