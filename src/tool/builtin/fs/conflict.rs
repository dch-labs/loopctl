//! Detect-on-write conflict checks for the file-writing tools.
//!
//! Each writing tool re-verifies that its target's bytes have not changed
//! since the tool captured its baseline, immediately before the write. This
//! narrows the read→write race window from the whole call to the gap between
//! the check and the write; that residual gap is deliberate (both checks
//! document it). The tools hold baselines in two shapes, and each has a
//! matching check: `Edit` and `MultiEdit` hold the bytes they read and
//! compare them exactly; `Write` holds the content hash the read recorded
//! for the path and compares hashes.
//!
//! The check returns the *identity* of the file the compared bytes came
//! from, captured from the open handle rather than a path lookup. The write
//! re-checks that identity on the path's entry immediately before its
//! rename, so a target swapped between the check and the rename aborts
//! instead of silently replacing a file that was never compared.

use std::path::Path;

use tokio::io::AsyncReadExt;

use crate::tool::ToolError;

use super::state::content_hash;

/// Why a conflict check did not approve the write.
///
/// The two outcomes demand different handling: `Changed` is recoverable and
/// surfaces to the model as a soft error, while `Fault` propagates as a hard
/// [`ToolError`].
#[derive(Debug)]
pub(crate) enum CheckFailure {
    /// The target's bytes differ from the baseline.
    ///
    /// A recoverable conflict: the caller refuses the write and surfaces
    /// [`changed_message`] as a soft error.
    Changed,

    /// A genuine I/O fault while re-reading or statting the target.
    ///
    /// A missing file is a [`CheckFailure::Changed`], not a fault.
    Fault(ToolError),
}

/// The on-disk identity of the file whose bytes a conflict check compared.
///
/// Captured from the open read handle's own stat, so it pins the inode the
/// bytes actually came from even if the path's directory entry is swapped
/// concurrently. The write re-checks the path's entry against this identity
/// immediately before its rename; a mismatch — or a missing entry — aborts
/// the write, closing the swap window between the byte comparison and the
/// rename that a byte or hash comparison alone cannot see.
///
/// Platforms without a stable file identity carry no distinguishing
/// fields, and their identity matches any entry, degrading to the
/// pre-check-only behavior documented on the race windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TargetIdentity {
    /// The file's device id, from the read-time handle stat.
    ///
    /// Paired with [`ino`](Self::ino) to name one live file uniquely across
    /// the system; the re-check passes only when both fields match the
    /// entry now at the path. A replacement file allocated while the
    /// checked file still existed — the rename-swap case this gate targets
    /// — always carries a different pairing.
    #[cfg(unix)]
    dev: u64,

    /// The file's inode number, from the read-time handle stat.
    ///
    /// Stable for the file's lifetime within its filesystem and never
    /// reused while the file exists. One case stays beyond the gate by
    /// construction: a delete-then-recreate whose replacement reclaims the
    /// freed inode on the same device reads as an identity match — a
    /// filesystem-dependent residual shared with every inode-based guard,
    /// not the concurrent swap this gate exists for.
    #[cfg(unix)]
    ino: u64,
}

impl TargetIdentity {
    /// Extract the identity from a stat.
    ///
    /// The unix fields come from the platform's stat extensions, so the
    /// identity belongs to the inode that was stated — never to whatever a
    /// fresh path lookup would resolve. Platforms without a stable file
    /// identity produce an empty identity whose
    /// [`matches`](Self::matches) always succeeds.
    fn from_metadata(meta: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                dev: meta.dev(),
                ino: meta.ino(),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = meta;
            Self {}
        }
    }

    /// Whether the path entry `meta` describes is still the checked file.
    ///
    /// The comparison is exact on platforms with a stable file identity: a
    /// swapped-in entry fails because its device-inode pairing differs
    /// (see the field docs for the delete-and-reclaim residual). Only ever
    /// false there — on platforms without one, every entry matches so the
    /// write proceeds with the pre-check-only behavior documented on the
    /// race windows.
    pub(crate) fn matches(self, meta: &std::fs::Metadata) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            self.matches_parts(meta.dev(), meta.ino())
        }
        #[cfg(not(unix))]
        {
            let _ = (self, meta);
            true
        }
    }

    /// Whether raw stat fields still identify the checked file.
    ///
    /// The descriptor-relative counterpart of [`matches`](Self::matches):
    /// the contained write stats its target through the pinned directory's
    /// descriptor, which yields raw fields rather than a
    /// [`std::fs::Metadata`], and compares them here. Exact on platforms
    /// with a stable file identity; always true without one.
    pub(crate) fn matches_parts(self, dev: u64, ino: u64) -> bool {
        #[cfg(unix)]
        {
            self.dev == dev && self.ino == ino
        }
        #[cfg(not(unix))]
        {
            let _ = (self, dev, ino);
            true
        }
    }
}

/// Format the soft-error message for a refused write of `path`.
///
/// The text states what happened and directs the model to the recovery
/// path. Both check flavors produce the same shape — only the refused file
/// differs — so a consumer can render either uniformly.
pub(crate) fn changed_message(path: &Path) -> String {
    format!(
        "{path} changed on disk since it was read; not writing to avoid \
         clobbering the newer content.\n\nRe-read the file with Read, then \
         re-issue the write against the current content.",
        path = path.display()
    )
}

/// Re-read `path` and compare its bytes against `baseline`.
///
/// Returns the identity of the file the compared bytes came from; the
/// caller passes it to the write so the swap window between this check and
/// the rename can be closed (see [`TargetIdentity`]). A differing file, or
/// a file that no longer exists (a deletion between read and re-read is
/// "the file changed"), is `Err(CheckFailure::Changed)`; the caller refuses
/// the write and surfaces a soft error. An external writer replacing the
/// text file with non-UTF-8 content is a content change, not a fault.
///
/// # Race window
///
/// This check narrows the read→write race window from "the whole tool call"
/// to "the gap between this compare and the caller's write"; the returned
/// identity is what lets the caller close even that gap at rename time.
///
/// # Errors
///
/// Returns `Err(CheckFailure::Changed)` when the bytes differ or the
/// file is gone, and `Err(CheckFailure::Fault)` only on a genuine I/O
/// fault that is not "file missing".
pub(crate) async fn check_content_unchanged(
    baseline: &str,
    path: &Path,
) -> Result<TargetIdentity, CheckFailure> {
    check_bytes_unchanged(baseline.as_bytes(), path).await
}

/// Re-read `path` and compare its content hash against `baseline_hash`.
///
/// Used by Write, whose baseline is the content hash the read recorded for
/// the path — Write holds no prior bytes to compare. Hash comparison inherits
/// the exactness of the underlying byte comparison and has no timestamp
/// blind spot: any content change is caught regardless of filesystem
/// timestamp granularity. Missing-at-reread is a conflict. On success the
/// identity of the compared file is returned for the caller's pre-rename
/// re-check (see [`TargetIdentity`]).
///
/// # Race window
///
/// As with [`check_content_unchanged`], the check narrows the race window
/// to the gap between this compare and the caller's write; the returned
/// identity closes that gap at rename time.
///
/// # Errors
///
/// Returns `Err(CheckFailure::Changed)` when the bytes differ or the
/// file is gone, and `Err(CheckFailure::Fault)` only on a genuine I/O
/// fault that is not "file missing".
pub(crate) async fn check_content_hash_unchanged(
    baseline_hash: u64,
    path: &Path,
) -> Result<TargetIdentity, CheckFailure> {
    let mut file = open_target(path).await?;
    let identity = TargetIdentity::from_metadata(&stat_target(&file, path).await?);
    let mut current = Vec::new();
    file.read_to_end(&mut current)
        .await
        .map_err(|e| read_fault(path, &e))?;
    if content_hash(&current) == baseline_hash {
        Ok(identity)
    } else {
        Err(CheckFailure::Changed)
    }
}

/// Shared body of the byte-exact check: open, stat the handle, compare.
///
/// The identity comes from the handle's own stat, so it belongs to the
/// inode the bytes were read from even under a concurrent swap of the
/// path's entry.
///
/// # Errors
///
/// Returns `Err(CheckFailure::Changed)` when the bytes differ or the
/// file is gone, and `Err(CheckFailure::Fault)` only on a genuine I/O
/// fault that is not "file missing".
async fn check_bytes_unchanged(
    baseline: &[u8],
    path: &Path,
) -> Result<TargetIdentity, CheckFailure> {
    let mut file = open_target(path).await?;
    let identity = TargetIdentity::from_metadata(&stat_target(&file, path).await?);
    let mut current = Vec::new();
    file.read_to_end(&mut current)
        .await
        .map_err(|e| read_fault(path, &e))?;
    if current == baseline {
        Ok(identity)
    } else {
        Err(CheckFailure::Changed)
    }
}

/// Open the check target for reading.
///
/// The check opens the target itself rather than reading through a helper
/// so the same handle can be stated for the returned identity — a separate
/// path lookup afterwards could name a different file under a concurrent
/// swap. A missing file is a change, not a fault, matching the checks'
/// deletion-between-reads doctrine.
///
/// # Errors
///
/// Returns `Err(CheckFailure::Changed)` for a missing file and
/// `Err(CheckFailure::Fault)` for any other open failure.
async fn open_target(path: &Path) -> Result<tokio::fs::File, CheckFailure> {
    match tokio::fs::File::open(path).await {
        Ok(file) => Ok(file),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Err(CheckFailure::Changed),
        Err(err) => Err(read_fault(path, &err)),
    }
}

/// Stat the open handle — the inode the bytes come from, not a path lookup.
///
/// The stat rides the open descriptor, so the identity reflects the file
/// the check actually reads bytes from even if the path's directory entry
/// is replaced while the read is in flight.
///
/// # Errors
///
/// Returns `Err(CheckFailure::Fault)` when the open handle cannot be
/// stated.
async fn stat_target(
    file: &tokio::fs::File,
    path: &Path,
) -> Result<std::fs::Metadata, CheckFailure> {
    file.metadata().await.map_err(|e| read_fault(path, &e))
}

/// Map a check-side I/O error to the fault variant, naming the target.
///
/// Every I/O failure in the check's open-stat-read sequence funnels through
/// this one mapper, so fault messages keep a consistent shape and always
/// say which path was being verified when the check gave up.
fn read_fault(path: &Path, err: &std::io::Error) -> CheckFailure {
    CheckFailure::Fault(ToolError::Execution(format!(
        "conflict check for {}: {err}",
        path.display()
    )))
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

    fn temp_file(dir: &Path, name: &str, contents: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).expect("write fixture");
        path
    }

    #[tokio::test]
    async fn content_check_passes_when_the_file_is_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        assert!(check_content_unchanged("A", &path).await.is_ok());
    }

    #[tokio::test]
    async fn content_check_fails_when_an_external_writer_changed_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        std::fs::write(&path, "EXTERNAL").unwrap();
        assert!(matches!(
            check_content_unchanged("A", &path).await,
            Err(CheckFailure::Changed)
        ));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "EXTERNAL",
            "the check must never write"
        );
    }

    #[tokio::test]
    async fn content_check_treats_a_deleted_file_as_a_conflict() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            check_content_unchanged("A", &path).await,
            Err(CheckFailure::Changed)
        ));
    }

    #[tokio::test]
    async fn content_check_maps_a_genuine_io_fault_to_a_fault() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(
            check_content_unchanged("A", tmp.path()).await,
            Err(CheckFailure::Fault(ToolError::Execution(_)))
        ));
    }

    #[tokio::test]
    async fn content_check_detects_non_utf8_external_content_as_a_conflict() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.bin", "text");
        std::fs::write(&path, [0xFF, 0xFE, 0x00]).unwrap();
        assert!(
            matches!(
                check_content_unchanged("text", &path).await,
                Err(CheckFailure::Changed)
            ),
            "a binary swap is a content change, not an io fault"
        );
    }

    #[tokio::test]
    async fn hash_check_passes_when_the_file_is_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        let baseline = content_hash(b"A");
        assert!(check_content_hash_unchanged(baseline, &path).await.is_ok());
    }

    #[tokio::test]
    async fn hash_check_catches_a_same_mtime_content_change() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        let baseline_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let baseline_hash = content_hash(b"A");
        std::fs::write(&path, "EXTERNAL").unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(baseline_mtime).unwrap();

        assert!(matches!(
            check_content_hash_unchanged(baseline_hash, &path).await,
            Err(CheckFailure::Changed)
        ));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "EXTERNAL",
            "the check must never write"
        );
    }

    #[tokio::test]
    async fn hash_check_treats_a_deleted_file_as_a_conflict() {
        let tmp = tempfile::tempdir().unwrap();
        let path = temp_file(tmp.path(), "a.txt", "A");
        let baseline = content_hash(b"A");
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            check_content_hash_unchanged(baseline, &path).await,
            Err(CheckFailure::Changed)
        ));
    }

    #[test]
    fn changed_message_names_the_file_and_the_recovery_path() {
        let message = changed_message(Path::new("src/a.rs"));
        assert!(message.contains("src/a.rs"), "{message}");
        assert!(message.contains("changed"), "{message}");
        assert!(message.contains("Read"), "{message}");
    }
}
