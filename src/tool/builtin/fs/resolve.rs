//! Path resolution and containment for the file tools.
//!
//! Every tool that touches the filesystem resolves the model's path
//! through [`resolve_path`] under a [`ResolvePolicy`], so no two tools
//! can drift apart on what a relative path means or on which escapes
//! are refused. Containment is checked in two layers — lexical
//! normalization against the workspace, then a walk of the existing
//! symlink prefix — and the write path closes the remaining
//! check-to-open race with descriptor-pinned operations on Linux.

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use crate::tool::ToolError;

/// Whether a string looks like an HTTP(S) or `file://` URL.
///
/// Used by [`reject_path_url`] to refuse URLs where a filesystem path
/// is required, so a model that sends a `Read` against `https://…`
/// gets a clear redirect instead of a confusing filesystem error.
/// Case-insensitive (`HTTP://`, `Https://`, `FILE://` also match).
/// Returns `false` for `ftp://`, bare paths, and empty strings.
#[must_use]
pub(crate) fn is_url(path: &str) -> bool {
    path.get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("http://"))
        || path
            .get(..8)
            .is_some_and(|p| p.eq_ignore_ascii_case("https://"))
        || path
            .get(..7)
            .is_some_and(|p| p.eq_ignore_ascii_case("file://"))
}

/// Whether a string looks like a `file://` URL.
///
/// The URL spelling of a local file, case-insensitive (`FILE://` also
/// matches). Used by [`reject_path_url`] to give `file://` inputs
/// their own redirect — pass a filesystem path — since they are local
/// files in the wrong spelling rather than remote addresses.
#[must_use]
pub(crate) fn is_file_url(path: &str) -> bool {
    path.get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("file://"))
}

/// Reject a URL where a filesystem path is required.
///
/// Every file-touching tool guards its path arguments with this
/// check, so a model that sends `https://…` receives a consistent
/// error naming the tool it called instead of a confusing filesystem
/// error. `file://` URIs get their own message — they are local files
/// in the wrong spelling, so the redirect asks for a filesystem path.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] naming `tool` and asking for a
/// filesystem path, for both remote URLs (see [`is_url`]) and
/// `file://` URIs (see [`is_file_url`]).
pub(crate) fn reject_path_url(tool: &str, path: &str) -> Result<(), ToolError> {
    if is_file_url(path) {
        return Err(ToolError::InvalidInput(format!(
            "file:// URLs are not supported by the {tool} tool. Pass a filesystem path instead."
        )));
    }
    if is_url(path) {
        return Err(ToolError::InvalidInput(format!(
            "URLs are not supported by the {tool} tool. Pass a filesystem path instead."
        )));
    }
    Ok(())
}
/// Whether path resolution confines results to the working directory.
///
/// File tools thread the policy carried on the session's
/// [`FileSession`](super::FileSession) through `resolve_path`. The
/// host-facing surface is a boolean (an "unsafe paths" switch), which
/// maps onto this enum at the boundary — the enum is the internal,
/// call-site-readable form. Enablement is operator-only: the policy
/// is fixed when the session is constructed, and a running agent
/// cannot widen it mid-run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolvePolicy {
    /// Reject paths that escape the working directory, both lexically
    /// and through symlinks.
    ///
    /// The default, so every caller that does not name a policy gets
    /// containment. Post-open verification runs on every platform:
    /// Linux checks the opened handle's real location through its
    /// descriptor, other platforms check the opened path by name
    /// against the root pinned at session construction (the weaker,
    /// documented residual the portable arms carry), and contained
    /// writes take the pinned walk on Linux and judge placement
    /// against that same pinned root elsewhere.
    #[default]
    Contained,

    /// Lexical normalization only: any path the OS permits resolves,
    /// with no containment check and no filesystem probing.
    ///
    /// Restores the reach workflows like reading `/etc/nginx` or
    /// `/home/you/.ssh` need. There is no tilde expansion anywhere in
    /// resolution: a leading `~` is an ordinary path component.
    Unrestricted,
}

/// Resolve a possibly-relative `file_path` against `cwd` under `policy`.
///
/// The result is lexically normalized under both policies (`.`/`..`
/// collapsed without touching the filesystem, so not-yet-existing
/// write targets work). Relative paths are joined to `cwd`; absolute
/// paths are taken as given.
///
/// Under [`ResolvePolicy::Contained`] the result must additionally
/// stay inside the workspace, checked in two layers:
///
/// 1. **Lexical containment** — the normalized result must start with
///    `cwd`'s components or with the workspace's resolved form: both
///    spellings name the same workspace, and the resolved one is what
///    handle-verification and containment error messages print, so a
///    model echoing it back must not be rejected. Rejects `..`
///    traversal and unrelated absolute paths.
/// 2. **Symlink containment** — the workspace's *resolved* form is
///    taken as the anchor (the operator-supplied spelling may itself
///    cross symlinks; that choice is not judged), and each *existing*
///    component below it is probed with `symlink_metadata`. A
///    symlink's target is resolved (absolute, or relative to the
///    link's directory) and recursively checked against the resolved
///    workspace boundary. A symlink whose chain leaves the workspace
///    is rejected. Non-existent components stop the walk, so writing
///    a new file still works — only the existing prefix is verified.
///
/// Under [`ResolvePolicy::Unrestricted`] neither check runs —
/// resolution makes no filesystem calls at all. Tilde (`~`) is never
/// expanded; a leading `~` is an ordinary path component under either
/// policy.
///
/// This is the shared path-resolution primitive used by every
/// file-touching tool in the family, so they cannot drift apart. A
/// TOCTOU window remains between this check and the caller's open: on
/// Linux, file tools close it by verifying the opened handle against
/// the session's pinned workspace anchor and by writing through a
/// descriptor-pinned, no-follow walk; on other platforms the handle
/// check is name-based after the open, judged against the root pinned
/// at session construction (see [`verify_handle_inside`]), the same
/// weaker residual the portable write arm documents.
///
/// # Errors
///
/// Under [`ResolvePolicy::Contained`], returns
/// [`ToolError::InvalidInput`] when the resolved path does not stay
/// within `cwd`, either lexically or via a symlink, and
/// [`ToolError::Execution`] when a symlink cannot be read or when a
/// symlink chain exceeds the depth cap. [`ResolvePolicy::Unrestricted`]
/// never fails on policy grounds.
pub(crate) fn resolve_path(
    file_path: &str,
    cwd: &Path,
    policy: ResolvePolicy,
) -> Result<PathBuf, ToolError> {
    let path = Path::new(file_path);
    let joined = if path.is_relative() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    let normalized = normalize_lexical(&joined);
    match policy {
        ResolvePolicy::Unrestricted => return Ok(normalized),
        ResolvePolicy::Contained => {}
    }
    let base = normalize_lexical(cwd);
    let base_canonical = std::fs::canonicalize(cwd).unwrap_or_else(|_| base.clone());
    if !lexically_inside(&normalized, &base) && !lexically_inside(&normalized, &base_canonical) {
        return Err(ToolError::InvalidInput(format!(
            "Path escapes the working directory: {file_path}"
        )));
    }
    let mut visited = Vec::new();
    verify_symlinks_inside(&normalized, &base, &base_canonical, 0, &mut visited)?;
    Ok(normalized)
}

/// Canonicalize `path` when it resolves, refusing a dangling link.
///
/// Write-path callers use this under [`ResolvePolicy::Unrestricted`]
/// to honor symbolic links: an existing target is rewritten to its
/// physical file so the write lands on the referent and the link
/// stays intact. A path that does not exist yet is returned as given,
/// so new-file writes keep working. A final component that is a
/// symbolic link to a nonexistent file is refused: the caller's
/// rename would replace the link entry with a regular file instead of
/// creating the referent through it, silently severing the link. Any
/// other resolution failure — an unreadable ancestor or a symlink
/// loop — is propagated too: the physical target cannot be determined.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when canonicalization fails for
/// any reason other than a missing path, and when the final component
/// is a dangling symbolic link.
pub(crate) fn canonicalize_existing(path: &Path) -> Result<PathBuf, ToolError> {
    match std::fs::canonicalize(path) {
        Ok(real) => Ok(real),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
                Err(dangling_link_error(path))
            } else {
                Ok(path.to_path_buf())
            }
        }
        Err(error) => Err(ToolError::Execution(format!(
            "cannot resolve {}: {error}",
            path.display()
        ))),
    }
}

/// Build the refusal for a final component that is a link to nothing.
///
/// The message names the link's target so the caller can address the
/// referent directly; when the target itself cannot be read, the read
/// failure carries the reason instead.
fn dangling_link_error(path: &Path) -> ToolError {
    match std::fs::read_link(path) {
        Ok(target) => ToolError::Execution(format!(
            "cannot resolve {}: it is a symbolic link to {}, which does not \
             exist; write to the target path to create the referent",
            path.display(),
            target.display()
        )),
        Err(error) => ToolError::Execution(format!("cannot resolve {}: {error}", path.display())),
    }
}

/// Verify an opened handle actually resolved inside the workspace
/// (Linux).
///
/// Closes the check-to-use race the symlink walk cannot: a concurrent
/// process may swap a path component between validation and the open,
/// and the open follows the swap. The kernel pins whatever the swap
/// produced into the handle, and `/proc/self/fd` reveals its true
/// location — a handle resolving outside the workspace is rejected
/// before any bytes move. Unrestricted dispatches skip this check:
/// outside paths are permitted there by policy. The check is
/// descriptor-based; platforms without `/proc/self/fd` verify the
/// opened path by name instead, with the weaker residual that arm
/// documents.
///
/// The comparison anchors at the caller's retained
/// [`WorkspaceAnchor`](super::atomic::WorkspaceAnchor) when one is
/// supplied: the check reads the pinned root's true location from the
/// descriptor itself, so a symlink swapped onto the workspace
/// spelling after the session was constructed cannot make the
/// post-swap location pass — the read is judged against the directory
/// the operator's spelling resolved to at construction time. Without
/// an anchor the workspace path is canonicalized at check time.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the handle's real location
/// is outside the workspace, when the anchor retained no pinned root,
/// or when the check cannot be performed (fail closed).
#[cfg(target_os = "linux")]
pub(crate) fn verify_handle_inside<F: std::os::unix::io::AsRawFd>(
    handle: &F,
    _path: &Path,
    workspace: &Path,
    anchor: Option<&super::atomic::WorkspaceAnchor>,
) -> Result<(), ToolError> {
    verify_handle_pinned(handle, workspace, anchor)
}

/// Verify an opened handle actually resolved inside the workspace
/// (non-Linux).
///
/// The name-based twin of the Linux descriptor check: the comparison
/// root is resolved once — the root pinned at session construction
/// when an anchor is supplied, fail-closed when the anchor retained
/// none — and the opened file's path is canonicalized post-open and
/// judged against it, so the check grades where the path actually
/// landed against the session's root, not the spelling that produced
/// it nor wherever the workspace spelling points now. Unrestricted
/// dispatches skip this check: outside paths are permitted there by
/// policy. The residual is the one the portable write arm documents:
/// the answer is derived from the path, not pinned into the handle,
/// so a component swapped between the open and this check is not
/// caught — though a swapped path still has to land inside the pinned
/// root to pass.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the opened location is
/// outside the pinned root, when the anchor retained no pinned root,
/// or when the canonicalizations fault (fail closed on a genuine
/// fault, not by policy).
#[cfg(not(target_os = "linux"))]
pub(crate) fn verify_handle_inside<F>(
    _handle: &F,
    path: &Path,
    workspace: &Path,
    anchor: Option<&super::atomic::WorkspaceAnchor>,
) -> Result<(), ToolError> {
    let root = super::atomic::pinned_containment_root(workspace, anchor)?;
    verify_handle_portable(path, &root)
}

/// The descriptor-based containment check behind
/// [`verify_handle_inside`] on Linux.
///
/// The comparison anchors at the caller's retained
/// [`WorkspaceAnchor`](super::atomic::WorkspaceAnchor) when one is
/// supplied: the check reads the pinned root's true location from the
/// descriptor itself, so a symlink swapped onto the workspace
/// spelling after the session was constructed cannot make the
/// post-swap location pass — the read is judged against the directory
/// the operator's spelling resolved to at construction time. Without
/// an anchor the workspace path is canonicalized at check time.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the handle's real location
/// is outside the workspace, when the anchor retained no pinned root,
/// or when the check cannot be performed (fail closed).
#[cfg(target_os = "linux")]
fn verify_handle_pinned<F: std::os::unix::io::AsRawFd>(
    handle: &F,
    workspace: &Path,
    anchor: Option<&super::atomic::WorkspaceAnchor>,
) -> Result<(), ToolError> {
    let actual =
        std::fs::read_link(format!("/proc/self/fd/{}", handle.as_raw_fd())).map_err(|error| {
            ToolError::Execution(format!("cannot verify path containment: {error}"))
        })?;
    let workspace = match anchor {
        Some(anchor) => anchor.pinned_location().ok_or_else(|| {
            ToolError::Execution(format!(
                "cannot verify path containment: the workspace {} could not \
                 be opened when this session started; contained reads are \
                 refused until the session is reconstructed",
                workspace.display()
            ))
        })?,
        None => std::fs::canonicalize(workspace).map_err(|error| {
            ToolError::Execution(format!("cannot verify path containment: {error}"))
        })?,
    };
    if actual.starts_with(&workspace) {
        Ok(())
    } else {
        Err(ToolError::Execution(format!(
            "Path escaped the working directory: {}",
            actual.display()
        )))
    }
}

/// The name-based containment check behind [`verify_handle_inside`]
/// on platforms without `/proc/self/fd`.
///
/// Canonicalizes the opened file's path and judges it against the
/// caller-resolved `root` — the root pinned at session construction
/// when the dispatcher supplied an anchor — accepting when the former
/// falls under the latter. Canonicalization resolves symlinks, so a
/// link spelled inside the workspace that lands outside is refused by
/// where it lands, and a workspace spelling swapped mid-session cannot
/// relocate the root the file is judged against. The residual is the
/// one the portable write arm documents: the answer is derived from
/// the path, not pinned into the handle, so a component swapped
/// between the open and this check is not caught. A canonicalization
/// failure is a genuine fault, not a policy refusal, and fails
/// closed.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the opened location is
/// outside `root`, or when the canonicalization faults.
#[cfg(any(not(target_os = "linux"), test))]
fn verify_handle_portable(path: &Path, root: &Path) -> Result<(), ToolError> {
    let actual = std::fs::canonicalize(path).map_err(|error| {
        ToolError::Execution(format!("cannot verify path containment: {error}"))
    })?;
    if actual.starts_with(root) {
        Ok(())
    } else {
        Err(ToolError::Execution(format!(
            "Path escaped the working directory: {}",
            actual.display()
        )))
    }
}

/// Lexically normalize `path`, collapsing `.` and `..` without touching the
/// filesystem.
///
/// A leading `..` that would escape above the root is dropped,
/// matching the behavior of the shared normalization helper.
pub(crate) fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True when `path` starts with all of `base`'s components.
///
/// Both arguments are assumed already normalized (see
/// [`normalize_lexical`]). The root base (`/`) contains everything;
/// an empty `base` contains only itself.
fn lexically_inside(path: &Path, base: &Path) -> bool {
    if base.as_os_str().is_empty() {
        return path.as_os_str().is_empty();
    }
    let mut path_iter = path.components();
    for base_comp in base.components() {
        match path_iter.next() {
            Some(path_comp) if path_comp == base_comp => {}
            _ => return false,
        }
    }
    true
}

/// Deepest symbolic-link chain the containment walk follows before
/// refusing.
///
/// Matches the kernel's own per-resolution link limit, so the error a
/// hostile chain produces here is the shape an over-long chain would
/// produce at `open` anyway — a clean failure instead of stack
/// exhaustion.
const MAX_SYMLINK_DEPTH: usize = 40;

/// Walk the *existing* prefix of `resolved` and reject any symlink whose
/// target chain leaves the workspace.
///
/// `resolved` is assumed already lexically contained in `base_lex` or
/// in `base_canonical` (the caller's responsibility), and
/// `base_canonical` is the workspace's resolved form — `base_lex` may
/// itself spell the workspace through symlinks, which is the
/// operator's anchor choice and is not traversed or judged. Only the
/// components *below* the anchor are probed, by path —
/// `symlink_metadata` on the accumulated spelling under
/// `base_canonical` — and a symlink there is followed to its target,
/// which must stay (lexically, after normalization) inside
/// `base_canonical`, and its own existing prefix is walked the same
/// way (recursively, for chains). The walk itself is not
/// descriptor-pinned, so a component swapped mid-walk can skew a
/// rejection; that residual is closed downstream by the pinned write
/// and the handle verification. The walk stops at the first
/// non-existent component, so a not-yet-existing write leaf is never
/// rejected — only the existing prefix is checked. Cycles are bounded
/// by `visited`, a set of lexically normalized paths already examined,
/// and chains by `depth` against [`MAX_SYMLINK_DEPTH`] — a linear
/// chain of distinct links terminates with an error instead of
/// exhausting the stack.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when an existing symlink
/// component resolves (directly or via a chain) to a path outside
/// `base_canonical`. Returns [`ToolError::Execution`] when a symlink
/// cannot be read or when the chain exceeds [`MAX_SYMLINK_DEPTH`].
fn verify_symlinks_inside(
    resolved: &Path,
    base_lex: &Path,
    base_canonical: &Path,
    depth: usize,
    visited: &mut Vec<PathBuf>,
) -> Result<(), ToolError> {
    let relative = resolved
        .strip_prefix(base_lex)
        .or_else(|_| resolved.strip_prefix(base_canonical))
        .unwrap_or(resolved);
    let mut acc = base_canonical.to_path_buf();
    for comp in relative.components() {
        acc.push(comp.as_os_str());
        let Some(meta) = std::fs::symlink_metadata(&acc).ok() else {
            return Ok(());
        };
        if !meta.file_type().is_symlink() {
            continue;
        }
        if depth >= MAX_SYMLINK_DEPTH {
            return Err(ToolError::Execution(format!(
                "cannot resolve {}: too many levels of symbolic links",
                acc.display()
            )));
        }
        let target = std::fs::read_link(&acc).map_err(|e| {
            ToolError::Execution(format!("Failed to read symlink {}: {e}", acc.display()))
        })?;
        let resolved_target = if target.is_absolute() {
            target
        } else {
            acc.parent().unwrap_or_else(|| Path::new(".")).join(target)
        };
        let normalized_target = normalize_lexical(&resolved_target);
        if !lexically_inside(&normalized_target, base_canonical) {
            return Err(ToolError::InvalidInput(format!(
                "Path escapes the working directory via symlink: {}",
                acc.display()
            )));
        }
        let canonical = normalize_lexical(&normalized_target);
        if visited.iter().any(|seen| seen == &canonical) {
            continue;
        }
        visited.push(canonical);
        verify_symlinks_inside(
            &normalized_target,
            base_canonical,
            base_canonical,
            depth.saturating_add(1),
            visited,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn is_url_detects_http_https_and_file_urls() {
        assert!(is_url("http://example.com"));
        assert!(is_url("https://example.com/page"));
        assert!(is_url("file:///tmp/x"));
        assert!(is_url("HTTP://example.com"));
        assert!(is_url("FILE://x"));
    }

    #[test]
    fn is_url_rejects_non_urls() {
        assert!(!is_url("src/main.rs"));
        assert!(!is_url("ftp://example.com"));
        assert!(!is_url(""));
    }

    #[test]
    fn is_url_boundary_lengths() {
        assert!(is_url("http://"));
        assert!(is_url("https://"));
        assert!(!is_url("htt"));
        assert!(!is_url("http:/"));
    }

    #[test]
    fn is_url_non_ascii_does_not_panic() {
        assert!(!is_url("éttp://"));
        assert!(!is_url("éxample"));
    }

    #[test]
    fn reject_path_url_message_names_the_calling_tool_byte_for_byte() {
        let ToolError::InvalidInput(msg) =
            reject_path_url("Read", "https://example.com/x").unwrap_err()
        else {
            panic!("expected InvalidInput");
        };
        assert_eq!(
            msg,
            "URLs are not supported by the Read tool. Pass a filesystem path instead."
        );
    }

    #[test]
    fn reject_path_url_accepts_plain_paths() {
        assert!(reject_path_url("Read", "src/main.rs").is_ok());
        assert!(reject_path_url("Write", "/abs/path.txt").is_ok());
        assert!(reject_path_url("FileViewer", ".").is_ok());
    }

    #[test]
    fn reject_path_url_refuses_file_urls() {
        let ToolError::InvalidInput(msg) = reject_path_url("Grep", "file:///tmp/x").unwrap_err()
        else {
            panic!("expected InvalidInput");
        };
        assert_eq!(
            msg,
            "file:// URLs are not supported by the Grep tool. Pass a filesystem path instead."
        );
        assert!(reject_path_url("Read", "FILE://x").is_err());
    }

    #[test]
    fn resolve_path_relative_joins_cwd_and_normalizes() {
        let cwd = Path::new("/work");
        assert_eq!(
            resolve_path("sub/a.rs", cwd, ResolvePolicy::Contained).unwrap(),
            PathBuf::from("/work/sub/a.rs")
        );
    }

    #[test]
    fn resolve_path_rejects_unrelated_absolute() {
        let cwd = Path::new("/work");
        let err = resolve_path("/abs/a.rs", cwd, ResolvePolicy::Contained).unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref msg) if msg.contains("escapes")),
            "{err:?}"
        );
    }

    #[test]
    fn resolve_path_rejects_traversal_escape() {
        let cwd = Path::new("/work");
        assert!(resolve_path("../escape/a.rs", cwd, ResolvePolicy::Contained).is_err());
        assert!(resolve_path("sub/../../escape/a.rs", cwd, ResolvePolicy::Contained).is_err());
        assert!(resolve_path("../../..", cwd, ResolvePolicy::Contained).is_err());
    }

    #[test]
    fn resolve_path_allows_in_workspace_dots() {
        let cwd = Path::new("/work");
        assert_eq!(
            resolve_path("sub/../a.rs", cwd, ResolvePolicy::Contained).unwrap(),
            PathBuf::from("/work/a.rs")
        );
        assert_eq!(
            resolve_path("a/./b/../c.rs", cwd, ResolvePolicy::Contained).unwrap(),
            PathBuf::from("/work/a/c.rs")
        );
        assert_eq!(
            resolve_path("sub/..", cwd, ResolvePolicy::Contained).unwrap(),
            PathBuf::from("/work")
        );
    }

    #[test]
    fn resolve_path_accepts_absolute_inside_workspace() {
        let cwd = Path::new("/work");
        assert_eq!(
            resolve_path("/work/src/a.rs", cwd, ResolvePolicy::Contained).unwrap(),
            PathBuf::from("/work/src/a.rs")
        );
    }

    #[test]
    fn resolve_path_rejects_prefix_collision_directory() {
        let cwd = Path::new("/work");
        assert!(resolve_path("/workspace/a.rs", cwd, ResolvePolicy::Contained).is_err());
    }

    #[test]
    fn resolve_path_unrestricted_allows_traversal_escape() {
        assert_eq!(
            resolve_path(
                "../etc/passwd",
                Path::new("/work"),
                ResolvePolicy::Unrestricted
            )
            .unwrap(),
            PathBuf::from("/etc/passwd")
        );
    }

    #[test]
    fn resolve_path_unrestricted_allows_unrelated_absolute() {
        assert_eq!(
            resolve_path("/abs/a.rs", Path::new("/work"), ResolvePolicy::Unrestricted).unwrap(),
            PathBuf::from("/abs/a.rs")
        );
    }

    #[test]
    fn resolve_path_treats_a_leading_tilde_as_a_literal_component() {
        for policy in [ResolvePolicy::Contained, ResolvePolicy::Unrestricted] {
            assert_eq!(
                resolve_path("~/.ssh/config", Path::new("/work"), policy).unwrap(),
                PathBuf::from("/work/~/.ssh/config")
            );
        }
    }

    #[test]
    fn resolve_path_unrestricted_still_normalizes_dots() {
        assert_eq!(
            resolve_path(
                "a/./b/../c.rs",
                Path::new("/work"),
                ResolvePolicy::Unrestricted
            )
            .unwrap(),
            PathBuf::from("/work/a/c.rs")
        );
    }

    #[test]
    fn normalize_lexical_collapses_dots_and_dotdot() {
        assert_eq!(
            normalize_lexical(Path::new("./a.rs")),
            PathBuf::from("a.rs")
        );
        assert_eq!(
            normalize_lexical(Path::new("src/../a.rs")),
            PathBuf::from("a.rs")
        );
        assert_eq!(
            normalize_lexical(Path::new("/work/./b/../a.rs")),
            PathBuf::from("/work/a.rs")
        );
    }

    #[test]
    fn normalize_lexical_consecutive_dotdot_pop_each() {
        assert_eq!(
            normalize_lexical(Path::new("/work/a/b/../../c.rs")),
            PathBuf::from("/work/c.rs")
        );
    }

    #[test]
    fn normalize_lexical_drops_leading_dotdot_at_root() {
        assert_eq!(normalize_lexical(Path::new("/..")), PathBuf::from("/"));
        assert_eq!(normalize_lexical(Path::new("/a/../..")), PathBuf::from("/"));
    }

    #[test]
    fn normalize_lexical_is_idempotent() {
        for raw in ["./a/../b.rs", "/work/./x/../../y", "a/b/../..", "/"] {
            let once = normalize_lexical(Path::new(raw));
            assert_eq!(normalize_lexical(&once), once, "{raw}");
        }
    }

    #[test]
    fn lexically_inside_accepts_self_and_descendant() {
        assert!(lexically_inside(Path::new("/work"), Path::new("/work")));
        assert!(lexically_inside(
            Path::new("/work/src/a.rs"),
            Path::new("/work")
        ));
        assert!(lexically_inside(Path::new("/work/a"), Path::new("/work/a")));
    }

    #[test]
    fn lexically_inside_rejects_sibling_and_unrelated() {
        assert!(!lexically_inside(Path::new("/wor"), Path::new("/work")));
        assert!(!lexically_inside(
            Path::new("/workspace/a.rs"),
            Path::new("/work")
        ));
        assert!(!lexically_inside(
            Path::new("/other/a.rs"),
            Path::new("/work")
        ));
    }

    #[test]
    fn lexically_inside_root_base_contains_everything() {
        assert!(lexically_inside(Path::new("/anything"), Path::new("/")));
        assert!(lexically_inside(Path::new("/"), Path::new("/")));
    }

    #[test]
    fn lexically_inside_empty_base_only_contains_empty() {
        assert!(lexically_inside(Path::new(""), Path::new("")));
        assert!(!lexically_inside(Path::new("a"), Path::new("")));
    }

    #[test]
    fn canonicalize_existing_returns_the_given_path_for_a_missing_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let missing = tmp.path().join("not-there.rs");
        assert_eq!(
            canonicalize_existing(&missing).unwrap(),
            missing,
            "a new-file target must come back as given"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_accepts_symlink_pointing_inside_workspace() {
        use std::os::unix::fs::symlink;
        let work = tempfile::TempDir::new().unwrap();
        let real = work.path().join("real.txt");
        std::fs::write(&real, "x").unwrap();
        let link = work.path().join("link.txt");
        symlink(&real, &link).unwrap();
        assert_eq!(
            resolve_path("link.txt", work.path(), ResolvePolicy::Contained).unwrap(),
            work.path().join("link.txt")
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_accepts_a_symlinked_workspace_spelling() {
        use std::os::unix::fs::symlink;

        let real = tempfile::TempDir::new().unwrap();
        let workspace = tempfile::TempDir::new().unwrap();
        let link = workspace.path().join("project");
        symlink(real.path(), &link).unwrap();
        std::fs::create_dir_all(real.path().join("src")).unwrap();
        std::fs::write(real.path().join("src").join("a.rs"), "x").unwrap();

        let existing = resolve_path("src/a.rs", &link, ResolvePolicy::Contained).unwrap();
        assert_eq!(existing, link.join("src").join("a.rs"));
        let new_file = resolve_path("src/new.rs", &link, ResolvePolicy::Contained).unwrap();
        assert_eq!(new_file, link.join("src").join("new.rs"));

        let physical = real.path().join("src").join("a.rs");
        assert_eq!(
            resolve_path(physical.to_str().unwrap(), &link, ResolvePolicy::Contained).unwrap(),
            physical
        );
        assert_eq!(
            resolve_path(
                real.path().join("src").join("new.rs").to_str().unwrap(),
                &link,
                ResolvePolicy::Contained
            )
            .unwrap(),
            real.path().join("src").join("new.rs")
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_unrestricted_ignores_an_escaping_symlink() {
        use std::os::unix::fs::symlink;
        let work = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, "x").unwrap();
        let link = work.path().join("link.txt");
        symlink(&secret, &link).unwrap();
        assert!(
            resolve_path("link.txt", work.path(), ResolvePolicy::Unrestricted).is_ok(),
            "the symlink walk exists solely to enforce containment; unrestricted skips it"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_rejects_symlink_pointing_outside_workspace() {
        use std::os::unix::fs::symlink;
        let work = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, "x").unwrap();
        let link = work.path().join("link.txt");
        symlink(&secret, &link).unwrap();
        let err = resolve_path("link.txt", work.path(), ResolvePolicy::Contained).unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("symlink")),
            "{err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_rejects_symlink_chain_escaping_workspace() {
        use std::os::unix::fs::symlink;
        let work = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, "x").unwrap();
        let first = work.path().join("first");
        std::fs::create_dir_all(&first).unwrap();
        let hop = first.join("hop");
        symlink(&secret, &hop).unwrap();
        let link = work.path().join("link.txt");
        symlink(&hop, &link).unwrap();
        let err = resolve_path("link.txt", work.path(), ResolvePolicy::Contained).unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref s) if s.contains("symlink")),
            "{err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_caps_a_linear_symlink_chain() {
        use std::os::unix::fs::symlink;

        let work = tempfile::TempDir::new().unwrap();
        let mut link = work.path().join("deep.txt");
        std::fs::write(&link, "x").unwrap();
        for i in 0..45 {
            let next = work.path().join(format!("l{i}"));
            symlink(&link, &next).unwrap();
            link = next;
        }

        let err = resolve_path(
            link.to_str().unwrap(),
            work.path(),
            ResolvePolicy::Contained,
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("too many levels")),
            "{err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_path_allows_write_to_new_file_under_existing_dir() {
        let work = tempfile::TempDir::new().unwrap();
        let nested = work.path().join("src");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(
            resolve_path("src/new.rs", work.path(), ResolvePolicy::Contained).unwrap(),
            work.path().join("src/new.rs")
        );
    }

    #[cfg(unix)]
    #[test]
    fn canonicalize_existing_propagates_an_unresolvable_link() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::TempDir::new().unwrap();
        let looped = tmp.path().join("loop.rs");
        symlink(&looped, &looped).unwrap();
        let err = canonicalize_existing(&looped).unwrap_err();
        assert!(err.to_string().contains("cannot resolve"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn canonicalize_existing_refuses_a_dangling_symlink() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::TempDir::new().unwrap();
        let link = tmp.path().join("link.rs");
        symlink(tmp.path().join("absent.rs"), &link).unwrap();
        let err = canonicalize_existing(&link).unwrap_err();
        assert!(err.to_string().contains("cannot resolve"), "{err}");
        assert!(
            err.to_string().contains("absent.rs"),
            "the refusal must name the broken target: {err}"
        );
    }
}

#[cfg(all(test, target_os = "linux"))]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]
mod verify_tests {
    use super::*;

    #[test]
    fn verify_accepts_a_handle_inside_the_workspace() {
        let workspace = tempfile::TempDir::new().unwrap();
        let inside = workspace.path().join("inside.txt");
        std::fs::write(&inside, "x").unwrap();
        let handle = std::fs::File::open(&inside).unwrap();
        assert!(verify_handle_inside(&handle, &inside, workspace.path(), None).is_ok());
    }

    #[test]
    fn verify_rejects_a_handle_outside_the_workspace() {
        let workspace = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let file = outside.path().join("outside.txt");
        std::fs::write(&file, "x").unwrap();
        let handle = std::fs::File::open(&file).unwrap();
        let err = verify_handle_inside(&handle, &file, workspace.path(), None).unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("escaped")),
            "{err:?}"
        );
    }

    #[test]
    fn verify_detects_a_swap_that_happened_before_the_open() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "x").unwrap();
        let link = workspace.path().join("link.txt");
        symlink(&secret, &link).unwrap();

        let handle = std::fs::File::open(&link).unwrap();
        let err = verify_handle_inside(&handle, &link, workspace.path(), None).unwrap_err();
        assert!(err.to_string().contains("escaped"), "{err}");
    }

    #[test]
    fn portable_verify_accepts_a_path_inside_the_workspace() {
        let workspace = tempfile::TempDir::new().unwrap();
        let inside = workspace.path().join("inside.txt");
        std::fs::write(&inside, "x").unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        assert!(
            verify_handle_portable(&inside, &root).is_ok(),
            "a file inside the workspace must verify"
        );
    }

    #[test]
    fn portable_verify_rejects_a_path_outside_the_workspace() {
        let workspace = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let file = outside.path().join("outside.txt");
        std::fs::write(&file, "x").unwrap();
        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let err = verify_handle_portable(&file, &root).unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("escaped")),
            "{err:?}"
        );
    }

    #[test]
    fn portable_verify_sees_through_a_symlink_pointing_outside() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "x").unwrap();
        let link = workspace.path().join("link.txt");
        symlink(&secret, &link).unwrap();

        let root = std::fs::canonicalize(workspace.path()).unwrap();
        let err = verify_handle_portable(&link, &root).unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("escaped")),
            "the check must judge where the link lands, not its spelling: {err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn portable_verify_refuses_a_swapped_workspace_spelling() {
        use super::super::atomic::WorkspaceAnchor;
        use std::os::unix::fs::symlink;

        let parent = tempfile::TempDir::new().unwrap();
        let ws = parent.path().join("ws");
        std::fs::create_dir(&ws).unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "s").unwrap();
        let anchor = WorkspaceAnchor::pin(&ws);

        std::fs::remove_dir(&ws).unwrap();
        symlink(outside.path(), &ws).unwrap();

        let secret_via_ws = ws.join("secret.txt");
        let root = super::super::atomic::pinned_containment_root(&ws, Some(&anchor)).unwrap();
        let err = verify_handle_portable(&secret_via_ws, &root).unwrap_err();
        assert!(
            matches!(err, ToolError::Execution(ref s) if s.contains("escaped")),
            "the check must judge against the pinned root, not the swapped spelling: {err:?}"
        );
    }
}
