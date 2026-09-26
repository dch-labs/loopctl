//! Atomic, policy-aware file persistence for the write tools.
//!
//! Every write lands through one entry point, [`atomic_write`], which
//! dispatches on the session's resolve policy: the unrestricted
//! policy takes the plain path-based write, the contained policy
//! takes the descriptor-pinned walk on unix — and refuses contained
//! writes outright on platforms without descriptor-relative
//! operations, deliberately offering no pathname-based fallback. Both
//! performing arms create the new content in a co-located temporary
//! file and rename it into place, so the switch is a single
//! filesystem operation with no torn-write window, and both re-check
//! a conflict-check identity on the target's entry immediately
//! before the rename.

use std::fmt;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
#[cfg(unix)]
use std::sync::Mutex;

use crate::tool::ToolError;

use super::conflict::TargetIdentity;
use super::resolve::ResolvePolicy;

/// Write `content` to `target` atomically under `policy`.
///
/// The temp file is co-located with the target and renamed into place, so
/// the switch is a single filesystem operation with no torn-write window.
/// The two policies differ in how much of the filesystem is trusted:
///
/// Under [`ResolvePolicy::Contained`] the write is *pinned* on unix —
/// Linux and macOS alike: the parent chain is walked from the
/// workspace anchor one component at a time with `O_NOFOLLOW` —
/// creating missing directories through that same link-free chain —
/// the temp file is created inside the pinned directory with
/// `openat`, and the persist is a `renameat` within that one
/// descriptor. Nothing after validation re-resolves a path
/// component, so a concurrent swap cannot relocate the write:
/// placement outside the workspace is impossible by construction. A
/// symbolic link anywhere in the parent chain, or as the final entry,
/// is refused — resolve it and pass the real path. On platforms
/// without descriptor-relative operations contained writes are
/// refused outright: a pathname-based fallback could not hold this
/// guarantee, so none is offered.
///
/// Under [`ResolvePolicy::Unrestricted`] the write is the plain
/// path-based counterpart: temp file in the target's directory,
/// permissions preserved from the existing entry, rename onto the
/// path as given.
///
/// When `expected` carries the [`TargetIdentity`] a conflict check
/// captured, the target's entry is re-checked against it immediately
/// before the rename — by `fstatat` on the pinned descriptor under
/// Contained on Linux, by a path stat otherwise — and a swapped or
/// removed target aborts the write instead of silently replacing a
/// file that was never compared. `None` skips the re-check: new-file
/// writes and platforms without a stable identity. The residual
/// either way is the universal rename-semantics window: an entry
/// swapped in between that stat and the rename (two adjacent
/// syscalls) is replaced by the rename.
///
/// The walk starts from `anchor` when the caller can supply one: the
/// session's retained descriptor for the workspace's resolved root,
/// so a symlink swapped onto the anchor after the session was
/// constructed cannot redirect the descent. `None` (direct callers,
/// no session) opens the anchor spelling per call, as before.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when a contained target or
/// parent component is a symbolic link. Returns
/// [`ToolError::Execution`] when the checked identity no longer
/// matches the target's entry, when the walk cannot reach or create
/// the parent directory, and on any failure creating, writing, or
/// persisting the temp file.
pub(crate) fn atomic_write(
    target: &Path,
    content: &str,
    workspace: &Path,
    policy: ResolvePolicy,
    expected: Option<&TargetIdentity>,
    anchor: Option<&WorkspaceAnchor>,
) -> Result<(), ToolError> {
    match policy {
        ResolvePolicy::Unrestricted => path_write(target, content, expected),
        ResolvePolicy::Contained => {
            #[cfg(unix)]
            {
                pinned_write(target, content, workspace, expected, anchor)
            }
            #[cfg(not(unix))]
            {
                let _ = (target, content, workspace, expected, anchor);
                Err(ToolError::Execution(
                    "contained writes require descriptor-relative filesystem \
                     operations, which this platform does not provide; refuse \
                     the write or reconstruct the session with an unrestricted \
                     policy"
                        .to_string(),
                ))
            }
        }
    }
}

/// The path-based write behind the Unrestricted policy.
///
/// Unrestricted is the documented no-probing mode: the temp file is
/// created in the target's directory by path, permissions are copied
/// from the existing entry by path, and the rename lands on the path
/// as given. The only check beyond I/O is the identity gate, which
/// stats the path immediately before the rename when a conflict
/// check armed it.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the checked identity no
/// longer matches the path's entry, and on any failure creating,
/// writing, or persisting the temp file.
fn path_write(
    target: &Path,
    content: &str,
    expected: Option<&TargetIdentity>,
) -> Result<(), ToolError> {
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| ToolError::Execution(format!("Failed to create temp file: {e}")))?;
    if let Ok(meta) = std::fs::metadata(target) {
        let perms = meta.permissions();
        tmp.as_file()
            .set_permissions(perms)
            .map_err(|e| ToolError::Execution(format!("Failed to set permissions: {e}")))?;
    }

    tmp.write_all(content.as_bytes())
        .map_err(|e| ToolError::Execution(format!("Failed to write temp file: {e}")))?;
    tmp.flush()
        .map_err(|e| ToolError::Execution(format!("Failed to flush temp file: {e}")))?;
    if let Some(identity) = expected {
        let current = match std::fs::metadata(target) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(swap_abort(target));
            }
            Err(err) => {
                return Err(ToolError::Execution(format!(
                    "cannot re-check the write target {}: {err}",
                    target.display()
                )));
            }
        };
        if !identity.matches(&current) {
            return Err(swap_abort(target));
        }
    }
    tmp.persist(target)
        .map_err(|e| ToolError::Execution(format!("Failed to persist file: {e}")))?;

    Ok(())
}

/// The abort for a target that changed between the conflict check and the
/// rename.
///
/// The swap is a fault rather than a plain content conflict, so the error
/// stays on the hard [`ToolError`] channel the write's plumbing already
/// carries. The message still points at the soft path's recovery: the
/// model re-reads the file and re-issues the write against the current
/// content.
fn swap_abort(target: &Path) -> ToolError {
    ToolError::Execution(format!(
        "{} changed while the write was being prepared; re-read it and re-issue \
         the write against the current content.",
        target.display()
    ))
}

/// A retained pin of the workspace's resolved root.
///
/// Captured once when the session is constructed — following the
/// anchor spelling's links at that moment — and held until the session
/// is dropped. On unix the pin is an open descriptor: contained writes
/// start from a duplicate of it rather than reopening the anchor
/// path, descending to the target without following links; on Linux,
/// contained reads also verify opened handles against that
/// descriptor's true location. On unix platforms other than Linux the
/// pin additionally carries the root's canonical path — the
/// comparison root for the name-based read check — and on platforms
/// without descriptors the canonical path is the pin itself. Either
/// way a symlink swapped onto the anchor after construction cannot
/// redirect a contained operation: the root a path is judged against,
/// and the directory a walk starts from, are the ones the operator's
/// spelling resolved to at construction time.
pub(crate) struct WorkspaceAnchor {
    /// The pinned root descriptor, once the workspace root could be
    /// opened.
    ///
    /// `None` means the open failed at construction time. The anchor
    /// never re-opens the spelling lazily — a swap during the gap
    /// would pin whatever it then resolved to, a directory the
    /// operator's configuration never validated — so contained
    /// operations fail closed until the session is reconstructed with
    /// an openable workspace.
    #[cfg(unix)]
    resolved: Mutex<Option<AnchorFd>>,

    /// The workspace root's canonical path, captured at construction.
    ///
    /// The comparison root for the name-based contained-read check on
    /// platforms without `/proc/self/fd`, and the only pin on
    /// platforms without descriptors at all. `None` means the root
    /// could not be resolved at construction time. The anchor never
    /// re-resolves the spelling lazily — a swap during the gap would
    /// capture whatever it then resolved to, a directory the
    /// operator's configuration never validated — so the reads that
    /// judge against it fail closed until the session is
    /// reconstructed with a resolvable workspace.
    #[cfg(not(target_os = "linux"))]
    pinned_root: Option<PathBuf>,
}

impl WorkspaceAnchor {
    /// Pin the workspace's resolved root, if that can be done now.
    ///
    /// Deliberately un-failing: a session must stay constructible even
    /// when the root cannot be pinned — construction is public API and
    /// cannot fail. When the pin fails the anchor retains nothing,
    /// and contained operations fail closed rather than pin whatever
    /// the workspace spelling resolves to later; reconstructing the
    /// session once the workspace can be pinned is the recovery.
    pub(crate) fn pin(workspace: &Path) -> Self {
        Self {
            #[cfg(unix)]
            resolved: Mutex::new(AnchorFd::open(workspace).ok()),
            #[cfg(not(target_os = "linux"))]
            pinned_root: std::fs::canonicalize(workspace).ok(),
        }
    }

    /// A fresh duplicate of the pinned root descriptor.
    ///
    /// The duplicate is owned by the caller's walk and closed with it;
    /// the retained master descriptor stays open for the session's
    /// lifetime. When the construction-time pin failed there is
    /// nothing to duplicate: contained operations fail closed rather
    /// than pin whatever the workspace spelling resolves to once it
    /// exists again — a swap during the gap would otherwise redirect
    /// writes to a directory the operator's configuration never
    /// validated.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] when no descriptor was pinned
    /// at construction time, or when the duplicate cannot be created.
    #[cfg(unix)]
    pub(crate) fn dup_fd(&self, workspace: &Path) -> Result<i32, ToolError> {
        let slot = self
            .resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(master) = slot.as_ref() else {
            return Err(ToolError::Execution(format!(
                "the workspace {} could not be opened when this session \
                 started; contained writes are refused until the session is \
                 reconstructed",
                workspace.display()
            )));
        };
        // SAFETY: `master` is a valid open descriptor for the call; the duplicate is owned by the caller.
        let duplicate = unsafe { libc::dup(master.raw()) };
        if duplicate < 0 {
            return Err(ToolError::Execution(format!(
                "cannot duplicate the workspace anchor: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(duplicate)
    }

    /// The true filesystem location of the pinned root, if one is retained.
    ///
    /// Read from the descriptor itself (`/proc/self/fd`), so the answer
    /// is where the workspace resolved at pin time — not wherever the
    /// spelling points now. `None` covers the un-pinned anchor (fail
    /// closed at the caller) and the unreadable descriptor, so a
    /// caller can never mistake an unverifiable root for a verified
    /// one.
    #[cfg(target_os = "linux")]
    pub(crate) fn pinned_location(&self) -> Option<PathBuf> {
        let slot = self
            .resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let master = slot.as_ref()?;
        std::fs::read_link(format!("/proc/self/fd/{}", master.raw())).ok()
    }

    /// The true filesystem location of the pinned root, if one is retained.
    ///
    /// The captured canonical path itself — where the workspace
    /// resolved at pin time, not wherever the spelling points now.
    /// `None` covers the un-pinned anchor (fail closed at the caller),
    /// so a caller can never mistake an unverifiable root for a
    /// verified one.
    #[cfg(not(target_os = "linux"))]
    pub(crate) fn pinned_location(&self) -> Option<PathBuf> {
        self.pinned_root.clone()
    }
}

/// The root a portable containment check judges paths against.
///
/// With an anchor, the root is the one pinned at session construction
/// — an un-pinnable anchor fails closed with the family's
/// reconstruction message rather than fall back to a check-time
/// resolution, which a mid-session swap would subvert. Without an
/// anchor (direct callers, tests) the workspace is canonicalized at
/// check time, the best such a caller can do.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the anchor retained no
/// pinned root, or when an anchor-less workspace cannot be
/// canonicalized.
#[cfg(any(not(target_os = "linux"), test))]
pub(crate) fn pinned_containment_root(
    workspace: &Path,
    anchor: Option<&WorkspaceAnchor>,
) -> Result<PathBuf, ToolError> {
    match anchor {
        Some(anchor) => anchor.pinned_location().ok_or_else(|| {
            ToolError::Execution(format!(
                "cannot verify path containment: the workspace {} could not \
                 be resolved when this session started; contained operations \
                 are refused until the session is reconstructed",
                workspace.display()
            ))
        }),
        None => std::fs::canonicalize(workspace).map_err(|error| {
            ToolError::Execution(format!("cannot verify path containment: {error}"))
        }),
    }
}

impl fmt::Debug for WorkspaceAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WorkspaceAnchor")
    }
}

/// An owned directory descriptor, closed on drop.
///
/// Wraps a raw descriptor so the walk can hold the anchor root without
/// reopening it; the drop closes it exactly once on every path.
#[cfg(unix)]
struct AnchorFd(i32);

#[cfg(unix)]
impl AnchorFd {
    /// Open `workspace` as a directory descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] when the path cannot be opened as
    /// a directory, carrying the OS error.
    fn open(workspace: &Path) -> Result<Self, ToolError> {
        Ok(Self(open_dir_fd(
            workspace,
            libc::O_RDONLY | libc::O_DIRECTORY,
        )?))
    }

    fn raw(&self) -> i32 {
        self.0
    }
}

#[cfg(unix)]
impl Drop for AnchorFd {
    fn drop(&mut self) {
        // SAFETY: the descriptor is closed exactly once, here.
        let _ = unsafe { libc::close(self.0) };
    }
}

/// Unique-name counter for contained temp files, alongside the creating
/// process's id in the name.
#[cfg(unix)]
static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A directory reached through the no-follow contained walk.
///
/// Owns every descriptor the walk opened — the workspace anchor and one
/// per descended component — and closes them all exactly once, on every
/// path, through `Drop`. The walk's rollback bookkeeping lives here too:
/// [`created`](Self::created) records the directories the walk itself
/// made as the index of their parent's descriptor in
/// [`walked`](Self::walked) plus the entry name, so a failed walk can
/// unlink them through still-open descriptors before anything is closed.
#[cfg(unix)]
#[derive(Debug)]
struct PinnedDir {
    /// The final walked directory — the parent the write targets.
    ///
    /// Set once the descent reaches the target's parent; `-1` while the
    /// walk is still running so a failure can never mistake an unset value
    /// for a valid descriptor.
    dir_fd: i32,

    /// Every descriptor the walk opened, workspace anchor first, `dir_fd`
    /// last.
    walked: Vec<i32>,

    /// Directories the walk created: the index into
    /// [`walked`](Self::walked) of each parent descriptor (pushed before
    /// descent, so always valid while the walk owns it) plus the entry
    /// name. Recorded immediately after `mkdirat` succeeds, before
    /// anything else can fail, so a created directory is never untracked.
    created: Vec<(usize, std::ffi::CString)>,
}

#[cfg(unix)]
impl PinnedDir {
    /// Remove the directories this walk created, deepest first.
    ///
    /// Best-effort and empty-directory-only: an entry that gained content
    /// concurrently is left standing rather than force-deleted. Called on
    /// the walk's failure paths, while every descriptor is still open.
    fn remove_created(&mut self) {
        for (parent, name) in self.created.iter().rev() {
            let Some(dir_fd) = self.walked.get(*parent) else {
                continue;
            };
            // SAFETY: `dir_fd` is a live walk descriptor and `name` outlives the call; empty-dir removal only.
            let _ = unsafe { libc::unlinkat(*dir_fd, name.as_ptr(), libc::AT_REMOVEDIR) };
        }
        self.created.clear();
    }
}

#[cfg(unix)]
impl Drop for PinnedDir {
    fn drop(&mut self) {
        for fd in self.walked.drain(..) {
            // SAFETY: each descriptor was pushed exactly once and is closed only here, on every path.
            let _ = unsafe { libc::close(fd) };
        }
    }
}

/// Pin `parent` for a contained write: walk to it without following
/// symlinks, creating missing directories when `create_missing` is set.
///
/// The walk descends from the `workspace` anchor one component at a time,
/// opening each with `O_NOFOLLOW`, so a component swapped for a symbolic
/// link after validation can never be traversed — descent happens only
/// through a descriptor chain that stayed on the named, link-free path. A
/// symbolic link found anywhere in the chain is refused (matching the
/// batch tools' stricter pre-write posture): resolve it and pass the real
/// path. This is deliberately stricter than reads, which may traverse
/// in-workspace links — only operations that leave new filesystem entries
/// behind are no-follow. The workspace anchor itself is opened following
/// links: it is the operator-supplied root, while the descent — where a
/// concurrent swap would land — is strictly no-follow. `parent` may be
/// spelled through the anchor or through the workspace's resolved form
/// (both are accepted for containment); the walk anchors at whichever
/// spelling matched — physically the same directory.
///
/// With `create_missing`, directories that do not exist are created
/// through the pinned chain (`mkdirat` on the current descriptor) and
/// recorded so a later failure in the same walk removes them again —
/// empty-directory removal only, so a concurrently filled directory is
/// left standing rather than force-deleted.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when `parent` is not inside
/// `workspace`, when the anchor cannot be opened, when a component is a
/// symbolic link or cannot be opened or created, and — without
/// `create_missing` — when a component does not exist. On any error the
/// walk's own creations are rolled back and every descriptor is closed
/// before the error is returned.
#[cfg(unix)]
fn open_contained_dir(
    parent: &Path,
    workspace: &Path,
    create_missing: bool,
    anchor: Option<&WorkspaceAnchor>,
) -> Result<PinnedDir, ToolError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let canonical_workspace =
        std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let (anchor_spelling, relative) = match parent.strip_prefix(workspace) {
        Ok(relative) => (workspace, relative),
        Err(_) => match parent.strip_prefix(&canonical_workspace) {
            Ok(relative) => (canonical_workspace.as_path(), relative),
            Err(_) => {
                return Err(ToolError::Execution(format!(
                    "cannot create directories safely: {} is not inside {}",
                    parent.display(),
                    workspace.display()
                )));
            }
        },
    };
    let root = match anchor {
        Some(anchor) => anchor.dup_fd(anchor_spelling)?,
        None => open_dir_fd(anchor_spelling, libc::O_RDONLY | libc::O_DIRECTORY)?,
    };

    let mut pinned = PinnedDir {
        dir_fd: -1,
        walked: Vec::new(),
        created: Vec::new(),
    };
    let failure = (|| {
        pinned.walked.push(root);
        let mut current = root;
        for component in relative.components() {
            let name = CString::new(component.as_os_str().as_bytes())
                .map_err(|_| ToolError::Execution("path contains a NUL byte".to_string()))?;
            match openat_dir(current, &name) {
                Ok(fd) => {
                    pinned.walked.push(fd);
                    current = fd;
                }
                Err(error) => {
                    if is_symlink_entry(current, &name) {
                        return Err(ToolError::Execution(format!(
                            "Refusing to write: {} is a symbolic link. \
                             Resolve it and pass the real path.",
                            component.as_os_str().to_string_lossy()
                        )));
                    }
                    if error.kind() != std::io::ErrorKind::NotFound || !create_missing {
                        return Err(ToolError::Execution(format!(
                            "cannot descend into {}: {error}",
                            component.as_os_str().to_string_lossy()
                        )));
                    }
                    // SAFETY: `current` is a valid open directory descriptor; `name` outlives the call; umask applies.
                    if unsafe { libc::mkdirat(current, name.as_ptr(), 0o777) } != 0 {
                        return Err(ToolError::Execution(format!(
                            "cannot create directory {}: {}",
                            component.as_os_str().to_string_lossy(),
                            std::io::Error::last_os_error()
                        )));
                    }
                    pinned
                        .created
                        .push((pinned.walked.len().saturating_sub(1), name.clone()));
                    let fd = openat_dir(current, &name).map_err(|error| {
                        ToolError::Execution(format!(
                            "cannot open the created directory {}: {error}",
                            component.as_os_str().to_string_lossy()
                        ))
                    })?;
                    pinned.walked.push(fd);
                    current = fd;
                }
            }
        }
        pinned.dir_fd = current;
        Ok(())
    })();

    if failure.is_err() {
        pinned.remove_created();
    }
    failure.map(|()| pinned)
}

/// Widen a stat mode to the `u32` the permissions API takes.
///
/// libc's `mode_t` is `u32` on some unices and `u16` on others; the
/// widening is an identity where the widths already match, so the walk
/// compiles against whichever width the platform's libc chose.
#[cfg(unix)]
fn stat_mode_u32(mode: libc::mode_t) -> u32 {
    #[cfg(target_os = "linux")]
    {
        mode
    }
    #[cfg(not(target_os = "linux"))]
    {
        u32::from(mode)
    }
}

/// Widen a stat device id to the `u64` identity comparisons use.
///
/// libc's `dev_t` is `u64` on some unices and a narrower signed type
/// on others; the widening is an identity where the widths already
/// match, and an impossible id when a platform could produce a
/// negative one, so no real identity ever false-matches.
#[cfg(unix)]
fn stat_dev_u64(dev: libc::dev_t) -> u64 {
    #[cfg(target_os = "linux")]
    {
        dev
    }
    #[cfg(not(target_os = "linux"))]
    {
        u64::try_from(dev).unwrap_or(u64::MAX)
    }
}

/// The contained write on unix: everything happens through the pinned
/// directory.
///
/// The parent chain is walked no-follow (creating missing directories),
/// the final entry is refused if it is a symbolic link, permissions are
/// copied from the existing entry by descriptor, the temp file is created
/// in the pinned directory with `openat(O_CREAT | O_EXCL)` under a unique
/// name, and the persist is a `renameat` within that one descriptor. No
/// step after the walk resolves a path component, so the placement the
/// walk proved cannot be changed by a concurrent swap.
///
/// # Errors
///
/// Propagates `open_contained_dir`'s errors; returns
/// [`ToolError::InvalidInput`] for a symbolic-link final entry,
/// [`swap_abort`] when an armed identity no longer matches the entry
/// (missing included), and [`ToolError::Execution`] for temp-file,
/// write, or rename failures. A temp file that cannot be written is
/// unlinked before returning.
#[cfg(unix)]
fn pinned_write(
    target: &Path,
    content: &str,
    workspace: &Path,
    expected: Option<&TargetIdentity>,
    anchor: Option<&WorkspaceAnchor>,
) -> Result<(), ToolError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let parent = target.parent().ok_or_else(|| {
        ToolError::Execution(format!(
            "cannot write to {}: no parent directory",
            target.display()
        ))
    })?;
    let name = target.file_name().ok_or_else(|| {
        ToolError::Execution(format!(
            "cannot write to {}: no file name",
            target.display()
        ))
    })?;
    let name = CString::new(name.as_bytes())
        .map_err(|_| ToolError::Execution("path contains a NUL byte".to_string()))?;

    let pinned = open_contained_dir(parent, workspace, true, anchor)?;
    let existing = fstatat_entry(pinned.dir_fd, &name, libc::AT_SYMLINK_NOFOLLOW).ok();
    if existing
        .as_ref()
        .is_some_and(|entry| (entry.st_mode & libc::S_IFMT) == libc::S_IFLNK)
    {
        return Err(ToolError::InvalidInput(format!(
            "Refusing to write: {} is a symbolic link. Resolve it and pass the real path.",
            target.display()
        )));
    }
    let (tmp_name, mut tmp_file) = create_temp_in(pinned.dir_fd)?;
    if let Some(existing) = &existing {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(stat_mode_u32(existing.st_mode) & 0o7777);
        if let Err(e) = tmp_file.set_permissions(perms) {
            drop(tmp_file);
            pinned_discard(pinned.dir_fd, &tmp_name);
            return Err(ToolError::Execution(format!(
                "Failed to set permissions: {e}"
            )));
        }
    }
    if let Err(error) = tmp_file
        .write_all(content.as_bytes())
        .and_then(|()| tmp_file.flush())
    {
        drop(tmp_file);
        // SAFETY: `dir_fd` is the pinned directory and `tmp_name` names a temp file created there.
        let _ = unsafe { libc::unlinkat(pinned.dir_fd, tmp_name.as_ptr(), 0) };
        return Err(ToolError::Execution(format!(
            "Failed to write temp file: {error}"
        )));
    }

    if let Some(identity) = expected {
        match fstatat_entry(pinned.dir_fd, &name, libc::AT_SYMLINK_NOFOLLOW) {
            Ok(entry) if identity.matches_parts(stat_dev_u64(entry.st_dev), entry.st_ino) => {}
            Ok(_) => {
                pinned_discard(pinned.dir_fd, &tmp_name);
                return Err(swap_abort(target));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                pinned_discard(pinned.dir_fd, &tmp_name);
                return Err(swap_abort(target));
            }
            Err(error) => {
                pinned_discard(pinned.dir_fd, &tmp_name);
                return Err(ToolError::Execution(format!(
                    "cannot re-check the write target {}: {error}",
                    target.display()
                )));
            }
        }
    }

    // SAFETY: every argument refers to the pinned directory or names this function created there.
    if unsafe {
        libc::renameat(
            pinned.dir_fd,
            tmp_name.as_ptr(),
            pinned.dir_fd,
            name.as_ptr(),
        )
    } != 0
    {
        let error = std::io::Error::last_os_error();
        pinned_discard(pinned.dir_fd, &tmp_name);
        return Err(ToolError::Execution(format!(
            "Failed to persist file: {error}"
        )));
    }
    Ok(())
}

/// Unlink a failed write's temp file from the pinned directory.
///
/// Best-effort by contract: the result is discarded and the write's error
/// is returned regardless, so a cleanup failure leaves a temp file behind
/// rather than a wrongly deleted target.
#[cfg(unix)]
fn pinned_discard(dir_fd: i32, tmp_name: &std::ffi::CString) {
    // SAFETY: `dir_fd` is the pinned directory and `tmp_name` names a temp file created there.
    let _ = unsafe { libc::unlinkat(dir_fd, tmp_name.as_ptr(), 0) };
}

/// Create a uniquely named temp file inside the pinned directory.
///
/// The name is derived from the process id and a process-local counter,
/// created with `O_CREAT | O_EXCL` and mode `0o600` (subject to the
/// umask), with a bounded retry on the vanishingly unlikely collision.
/// The returned [`std::fs::File`] owns the descriptor and closes it on
/// drop.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when a temp file cannot be created
/// for any reason other than a name collision, or when the collision
/// retry budget is exhausted.
#[cfg(unix)]
fn create_temp_in(dir_fd: i32) -> Result<(std::ffi::CString, std::fs::File), ToolError> {
    use std::ffi::CString;
    use std::os::unix::io::FromRawFd;
    use std::sync::atomic::Ordering;

    for _ in 0..100 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp_name = CString::new(format!(".tmp-{}-{sequence}", std::process::id()))
            .map_err(|_| ToolError::Execution("path contains a NUL byte".to_string()))?;
        // SAFETY: `dir_fd` is a valid open directory descriptor; `tmp_name` outlives the call; umask applies.
        let fd = unsafe {
            libc::openat(
                dir_fd,
                tmp_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd >= 0 {
            // SAFETY: `fd` is the descriptor `openat` just created for this temp file, owned nowhere else.
            return Ok((tmp_name, unsafe { std::fs::File::from_raw_fd(fd) }));
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(ToolError::Execution(format!(
                "Failed to create temp file: {error}"
            )));
        }
    }
    Err(ToolError::Execution(
        "Failed to create temp file: exhausted unique names".to_string(),
    ))
}

/// Stat `name` under the open directory `dir`, by descriptor.
///
/// Never resolves a path: the stat is relative to `dir`, so the answer
/// belongs to the entry in the pinned directory rather than to whatever a
/// re-resolved path would reach.
///
/// # Errors
///
/// Returns the OS error when the entry cannot be stated — including
/// `ENOENT` for a missing entry.
#[cfg(unix)]
fn fstatat_entry(dir: i32, name: &std::ffi::CString, flags: i32) -> std::io::Result<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dir` is a valid open descriptor, `name` outlives the call, and the stat buffer is writable.
    let filled = unsafe { libc::fstatat(dir, name.as_ptr(), stat.as_mut_ptr(), flags) };
    if filled != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fstatat` fully initialized the buffer on success.
    Ok(unsafe { stat.assume_init() })
}

/// Whether the entry `name` under the open directory `dir` is a symlink.
///
/// `O_NOFOLLOW` reports a symbolic-link component as a generic
/// not-a-directory or loop error, indistinguishable from a component that
/// is genuinely not a directory. Statting the entry without following —
/// `fstatat` with `AT_SYMLINK_NOFOLLOW`, relative to the same descriptor
/// the failed open used — separates the two, so the walk can refuse links
/// with a precise message. An entry that cannot itself be stated is
/// reported as not-a-link and lands in the caller's generic error path.
#[cfg(unix)]
fn is_symlink_entry(dir: i32, name: &std::ffi::CString) -> bool {
    fstatat_entry(dir, name, libc::AT_SYMLINK_NOFOLLOW)
        .is_ok_and(|stat| (stat.st_mode & libc::S_IFMT) == libc::S_IFLNK)
}

/// Open `path` as a directory descriptor with `flags`.
///
/// The workspace anchor is opened through this helper so every
/// descriptor the walk holds carries close-on-exec. The `flags`
/// argument is the caller's containment statement: the anchor itself
/// is opened following links (it is the operator-supplied root), while
/// the walk's own `openat_dir` calls add `O_NOFOLLOW` to their flags.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the path cannot be opened as a
/// directory, carrying the OS error.
#[cfg(unix)]
fn open_dir_fd(path: &Path, flags: i32) -> Result<i32, ToolError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| ToolError::Execution("path contains a NUL byte".to_string()))?;
    // SAFETY: `path` outlives the call.
    let fd = unsafe { libc::open(path.as_ptr(), flags | libc::O_CLOEXEC) };
    if fd < 0 {
        Err(ToolError::Execution(format!(
            "cannot open {}: {}",
            path.to_string_lossy(),
            std::io::Error::last_os_error()
        )))
    } else {
        Ok(fd)
    }
}

/// Open `name` under the open directory `dir`, refusing symbolic links.
///
/// One component per call is what makes the bounded walk sound: with
/// `O_NOFOLLOW`, `O_DIRECTORY`, and `O_CLOEXEC` set, a component can only
/// resolve to a real directory descriptor, and a symbolic link fails
/// instead of being traversed. The failure kind is not distinguishable by
/// [`std::io::ErrorKind`] alone (a loop error versus not-a-directory),
/// which is why the caller confirms the link case through
/// `is_symlink_entry`.
///
/// # Errors
///
/// Returns the OS error when the entry cannot be opened as a directory,
/// including the failure a symbolic-link component produces under
/// `O_NOFOLLOW`.
#[cfg(unix)]
fn openat_dir(dir: i32, name: &std::ffi::CString) -> std::io::Result<i32> {
    // SAFETY: `dir` is a valid open descriptor and `name` outlives the call; `O_NOFOLLOW` refuses links.
    let fd = unsafe {
        libc::openat(
            dir,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fd)
    }
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
    use super::super::conflict::check_content_unchanged;
    use super::*;

    #[test]
    fn atomic_write_replaces_existing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("out.txt");
        std::fs::write(&target, "old\n").unwrap();
        atomic_write(
            &target,
            "new\n",
            tmp.path(),
            ResolvePolicy::Unrestricted,
            None,
            None,
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new\n");
    }

    #[test]
    fn atomic_write_no_temp_left() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("clean.rs");
        atomic_write(
            &target,
            "fn main() {}\n",
            tmp.path(),
            ResolvePolicy::Unrestricted,
            None,
            None,
        )
        .unwrap();
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["clean.rs"]);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("script.sh");
        std::fs::write(&target, "#!/bin/bash\necho old\n").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();
        atomic_write(
            &target,
            "#!/bin/bash\necho new\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            None,
        )
        .unwrap();
        let mode = std::fs::metadata(&target).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o755,
            "permissions should be preserved as 0o755, got 0o{:o}",
            mode & 0o777
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_creates_missing_parents_under_containment() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("deep/nested/dir/new.rs");
        atomic_write(
            &target,
            "fn main() {}\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "fn main() {}\n",
            "the pinned walk must create the missing chain and land the write"
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_rejects_symlink_target_without_clobbering() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let real = tmp.path().join("real.txt");
        std::fs::write(&real, "original\n").unwrap();
        let link = tmp.path().join("link.txt");
        symlink(&real, &link).unwrap();

        let err = atomic_write(
            &link,
            "new content\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("symbolic link")),
            "{err:?}"
        );

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "link should still be a symlink"
        );
        assert_eq!(std::fs::read_to_string(&real).unwrap(), "original\n");
    }

    #[tokio::test]
    async fn atomic_write_aborts_when_the_target_changed_since_the_check() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("watched.txt");
        std::fs::write(&target, "checked\n").unwrap();
        let identity = check_content_unchanged("checked\n", &target).await.unwrap();
        let newcomer = tmp.path().join("swapped-in.txt");
        std::fs::write(&newcomer, "swapped in\n").unwrap();
        std::fs::rename(&newcomer, &target).unwrap();

        let err = atomic_write(
            &target,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Unrestricted,
            Some(&identity),
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("changed while the write was being prepared"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "swapped in\n",
            "the swapped-in file must be untouched"
        );
    }

    #[tokio::test]
    async fn atomic_write_proceeds_when_the_identity_still_matches() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("stable.txt");
        std::fs::write(&target, "old\n").unwrap();
        let identity = check_content_unchanged("old\n", &target).await.unwrap();

        atomic_write(
            &target,
            "new\n",
            tmp.path(),
            ResolvePolicy::Unrestricted,
            Some(&identity),
            None,
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new\n");
    }

    #[tokio::test]
    async fn atomic_write_aborts_when_the_target_vanished_since_the_check() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = tmp.path().join("gone.txt");
        std::fs::write(&target, "checked\n").unwrap();
        let identity = check_content_unchanged("checked\n", &target).await.unwrap();
        std::fs::remove_file(&target).unwrap();

        let err = atomic_write(
            &target,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Unrestricted,
            Some(&identity),
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("changed while the write was being prepared"),
            "{err}"
        );
        assert!(
            !target.exists(),
            "nothing may be recreated by an aborted write"
        );
    }

    #[cfg(not(unix))]
    #[test]
    fn contained_writes_refuse_without_descriptor_operations() {
        let err = atomic_write(
            Path::new("x.txt"),
            "x\n",
            Path::new("."),
            ResolvePolicy::Contained,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(&err, ToolError::Execution(msg) if msg.contains("descriptor-relative")),
            "contained writes must fail closed naming the platform limitation: {err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_creates_and_overwrites() {
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let target = tmp.path().join("note.txt");
        atomic_write(
            &target,
            "v1\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        atomic_write(
            &target,
            "v2\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "v2\n",
            "the pinned walk must create and then overwrite through one anchor"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_creates_missing_parents() {
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let target = tmp.path().join("a/b/c/new.txt");
        atomic_write(
            &target,
            "x\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "x\n",
            "the pinned walk must create the missing chain and land the write"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_leaves_no_temp_residue() {
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let target = tmp.path().join("clean.txt");
        atomic_write(
            &target,
            "x\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["clean.txt"]);
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let target = tmp.path().join("script.sh");
        std::fs::write(&target, "#!/bin/sh\necho old\n").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o750)).unwrap();
        atomic_write(
            &target,
            "#!/bin/sh\necho new\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        let mode = std::fs::metadata(&target).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o750, "got 0o{:o}", mode & 0o777);
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_accepts_a_workspace_spelled_through_an_alias() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let real_ws = tmp.path().join("real_ws");
        std::fs::create_dir(&real_ws).unwrap();
        let alias = tmp.path().join("alias_ws");
        symlink(&real_ws, &alias).unwrap();
        let anchor = WorkspaceAnchor::pin(&alias);

        let target = alias.join("note.txt");
        atomic_write(
            &target,
            "ours\n",
            &alias,
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(real_ws.join("note.txt")).unwrap(),
            "ours\n",
            "a write through the workspace's own alias spelling must land in the workspace"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_refuses_a_symlink_below_the_workspace() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let outside = tempfile::TempDir::new().unwrap();
        let link = tmp.path().join("escape");
        symlink(outside.path(), &link).unwrap();

        let target = link.join("file.txt");
        let err = atomic_write(
            &target,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("symbolic link")),
            "{err:?}"
        );
        assert!(
            !outside.path().join("file.txt").exists(),
            "nothing may land outside the workspace through a link below it"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_refuses_a_symlinked_final_entry_without_clobbering() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let real = tmp.path().join("real.txt");
        std::fs::write(&real, "original\n").unwrap();
        let link = tmp.path().join("link.txt");
        symlink(&real, &link).unwrap();

        let err = atomic_write(
            &link,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("symbolic link")),
            "{err:?}"
        );
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link must survive as a link"
        );
        assert_eq!(
            std::fs::read_to_string(&real).unwrap(),
            "original\n",
            "the referent must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_refuses_a_symlinked_parent_component() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let outside = tempfile::TempDir::new().unwrap();
        let link_dir = tmp.path().join("linked");
        symlink(outside.path(), &link_dir).unwrap();

        let target = link_dir.join("escape.txt");
        let err = atomic_write(
            &target,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("symbolic link")),
            "{err:?}"
        );
        assert!(
            !outside.path().join("escape.txt").exists(),
            "nothing may land outside the workspace through a link"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn contained_write_aborts_on_a_swapped_identity() {
        let tmp = tempfile::TempDir::new().unwrap();
        let anchor = WorkspaceAnchor::pin(tmp.path());
        let target = tmp.path().join("watched.txt");
        std::fs::write(&target, "checked\n").unwrap();
        let identity = check_content_unchanged("checked\n", &target).await.unwrap();
        let newcomer = tmp.path().join("swapped-in.txt");
        std::fs::write(&newcomer, "swapped in\n").unwrap();
        std::fs::rename(&newcomer, &target).unwrap();

        let err = atomic_write(
            &target,
            "ours\n",
            tmp.path(),
            ResolvePolicy::Contained,
            Some(&identity),
            Some(&anchor),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("changed while the write was being prepared"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "swapped in\n",
            "the swapped-in file must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn contained_write_refuses_a_swapped_workspace_spelling() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::TempDir::new().unwrap();
        let ws = parent.path().join("ws");
        std::fs::create_dir(&ws).unwrap();
        let anchor = WorkspaceAnchor::pin(&ws);
        let outside = tempfile::TempDir::new().unwrap();

        std::fs::remove_dir(&ws).unwrap();
        symlink(outside.path(), &ws).unwrap();

        let target = ws.join("escape.txt");
        let err = atomic_write(
            &target,
            "ours\n",
            &ws,
            ResolvePolicy::Contained,
            None,
            Some(&anchor),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(_)),
            "the walk must refuse to persist through a swapped spelling: {err:?}"
        );
        assert!(
            !outside.path().join("escape.txt").exists(),
            "nothing may land in the swapped-in tree"
        );
    }
}
