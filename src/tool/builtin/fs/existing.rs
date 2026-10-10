//! The write family's guarded pre-read of an existing target.
//!
//! Every tool that inspects an existing file before writing —
//! [`WriteTool`](super::write::WriteTool),
//! [`EditTool`](super::edit::EditTool),
//! [`MultiEditTool`](super::multi_edit::MultiEditTool), and the
//! staleness checks in [`conflict`](super::conflict) — reads the old
//! content through one type-gated, size-capped reader: a non-regular
//! target (a FIFO, a device, a socket) is classified without ever
//! being opened, so a writer-less FIFO cannot park the dispatch, and
//! a target over the family cap is classified without its bytes ever
//! being buffered, so a huge or endless file cannot balloon memory.
//! The cap matches the read tool's whole-content guard
//! ([`DEFAULT_MAX_SIZE_BYTES`](crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES)):
//! one figure governs what the family reads and what it writes over.

use super::atomic::WorkspaceAnchor;
use super::conflict::TargetIdentity;
use super::resolve::{self, ResolvePolicy};
use crate::tool::ToolError;
use std::path::Path;
use tokio::io::AsyncReadExt as _;

/// The most bytes of an existing target the write family reads.
///
/// Shared with the read tool's whole-content guard so one figure
/// governs both directions of the family's file access: a target over
/// this cap is reported as unreadable rather than buffered. Targets
/// reach the write path with baselines recorded under the read tool's
/// own cap, so an existing target now over this figure has grown
/// since it was last observed.
pub(crate) const MAX_EXISTING_READ_BYTES: u64 = crate::tool::builtin::read::DEFAULT_MAX_SIZE_BYTES;

/// What a guarded pre-read found at an existing target, for text
/// consumers.
///
/// The classification a writing tool needs before it proceeds: the
/// decoded previous content when it is readable text, and the reason
/// it is not otherwise. [`TooLarge`](ExistingText::TooLarge) and
/// [`NonRegular`](ExistingText::NonRegular) are classified without
/// reading — the guards exist so neither a huge nor a special file
/// can stall or balloon the dispatch that inspects it.
#[derive(Debug)]
pub(crate) enum ExistingText {
    /// The target's previous content, decoded as text.
    ///
    /// Bounded by [`MAX_EXISTING_READ_BYTES`]; the ordinary arm every
    /// diff, match, and preview consumes.
    Text(String),

    /// The target existed, but its bytes were not valid UTF-8.
    ///
    /// The bytes were read and bounded; only the decode refused.
    NotUtf8,

    /// The target had no existing filesystem entry.
    ///
    /// Classified from metadata without an open, and again from the
    /// open if the entry vanishes between the two.
    Absent,

    /// The target's size exceeds the family read cap.
    ///
    /// Classified from metadata (or from a bounded take that saw more
    /// than the cap) without buffering the content.
    TooLarge,

    /// The target is not a regular file.
    ///
    /// A FIFO, device, or socket: classified from metadata before the
    /// open, so a writer-less FIFO cannot park it, and re-checked on
    /// the opened handle for the swap window between the two.
    NonRegular,

    /// The target is a directory.
    ///
    /// Distinct from [`NonRegular`](ExistingText::NonRegular) because
    /// an atomic replace can never land on one: consumers that
    /// replace special files legitimately still refuse a directory
    /// outright.
    Directory,
}

/// What a guarded pre-read found at an existing target, for the hash
/// consumers.
///
/// The staleness checks hash raw bytes rather than decoding text, so
/// the guarded read hands them the opened handle's identity beside
/// the bounded bytes; the classifications match
/// [`ExistingText`](ExistingText) otherwise.
#[derive(Debug)]
pub(crate) enum ExistingBytes {
    /// The target's bytes and the identity of the handle they came
    /// from.
    ///
    /// The identity is taken from the opened descriptor, so it names
    /// the inode the bytes were read from even if the path's
    /// directory entry is replaced mid-read.
    Present {
        /// The opened handle's identity (dev/ino), for the
        /// caller's baseline records.
        identity: TargetIdentity,

        /// The target's content, bounded by
        /// [`MAX_EXISTING_READ_BYTES`].
        bytes: Vec<u8>,
    },

    /// The target had no existing filesystem entry.
    ///
    /// Classified from metadata without an open; the callers map it
    /// onto their own missing-file outcomes.
    Absent,

    /// The target's size exceeds the family read cap.
    ///
    /// Classified without buffering the content, so an oversized
    /// target costs one stat, not its bytes in memory.
    TooLarge,

    /// The target is not a regular file.
    ///
    /// A FIFO, device, or socket — an atomic replace can displace
    /// these, so the write arm degrades over them while the editing
    /// arms refuse.
    NonRegular,

    /// The target is a directory.
    ///
    /// Distinct from [`NonRegular`](ExistingBytes::NonRegular) for
    /// parity with [`ExistingText`](ExistingText::Directory): an
    /// atomic replace can never land on one.
    Directory,
}

/// Classify an existing target for a text consumer.
///
/// The metadata gate runs before the open — a non-regular target is
/// never opened, which is what keeps a writer-less FIFO from parking
/// the read — and the opened handle is re-checked for the swap window
/// between the two; the residual of that window (a target replaced by
/// a FIFO after the metadata check and before the open) is the same
/// check-then-act residual the family's other pre-open gates carry.
/// Under the contained policy the handle is verified inside the
/// workspace exactly as the callers' own reads did.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the metadata probe, the
/// open, or the bounded read fails for an I/O reason other than the
/// classified outcomes.
pub(crate) async fn read_existing_text(
    full_path: &Path,
    workspace: &Path,
    policy: ResolvePolicy,
    anchor: Option<&WorkspaceAnchor>,
) -> Result<ExistingText, ToolError> {
    let file = match open_gated(full_path).await? {
        GatedOpen::Absent => return Ok(ExistingText::Absent),
        GatedOpen::NonRegular => return Ok(ExistingText::NonRegular),
        GatedOpen::Directory => return Ok(ExistingText::Directory),
        GatedOpen::TooLarge => return Ok(ExistingText::TooLarge),
        GatedOpen::Handle(file) => file,
    };
    if policy == ResolvePolicy::Contained {
        resolve::verify_handle_inside(&file, full_path, workspace, anchor)?;
    }
    let handle_meta = file.metadata().await.map_err(|err| {
        ToolError::Execution(format!(
            "cannot stat existing target {}: {err}",
            full_path.display()
        ))
    })?;
    if !handle_meta.is_file() {
        return Ok(match classify_special(&handle_meta) {
            GatedOpen::Directory => ExistingText::Directory,
            _ => ExistingText::NonRegular,
        });
    }
    let bytes = read_bounded(file, full_path).await?;
    match bytes {
        GatedBytes::TooLarge => Ok(ExistingText::TooLarge),
        GatedBytes::Bytes(bytes) => match String::from_utf8(bytes) {
            Ok(text) => Ok(ExistingText::Text(text)),
            Err(_) => Ok(ExistingText::NotUtf8),
        },
    }
}

/// Classify an existing target for a hash consumer.
///
/// The same gates as [`read_existing_text`](read_existing_text),
/// answering raw bytes beside the opened handle's identity instead of
/// decoded text; containment verification stays with the callers, who
/// hold the workspace context the check needs.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the metadata probe, the
/// open, or the bounded read fails for an I/O reason other than the
/// classified outcomes.
pub(crate) async fn read_existing_bytes(full_path: &Path) -> Result<ExistingBytes, ToolError> {
    let file = match open_gated(full_path).await? {
        GatedOpen::Absent => return Ok(ExistingBytes::Absent),
        GatedOpen::NonRegular => return Ok(ExistingBytes::NonRegular),
        GatedOpen::Directory => return Ok(ExistingBytes::Directory),
        GatedOpen::TooLarge => return Ok(ExistingBytes::TooLarge),
        GatedOpen::Handle(file) => file,
    };
    let handle_meta = file.metadata().await.map_err(|err| {
        ToolError::Execution(format!(
            "cannot stat existing target {}: {err}",
            full_path.display()
        ))
    })?;
    if !handle_meta.is_file() {
        return Ok(match classify_special(&handle_meta) {
            GatedOpen::Directory => ExistingBytes::Directory,
            _ => ExistingBytes::NonRegular,
        });
    }
    let identity = TargetIdentity::from_metadata(&handle_meta);
    let bytes = read_bounded(file, full_path).await?;
    match bytes {
        GatedBytes::TooLarge => Ok(ExistingBytes::TooLarge),
        GatedBytes::Bytes(bytes) => Ok(ExistingBytes::Present { identity, bytes }),
    }
}

/// The gated open's outcome.
///
/// One answer per way the pre-read's gates can dispose of a target: a
/// usable handle, or the classification that explains why none was
/// opened.
enum GatedOpen {
    /// A regular-file handle whose metadata size sat within the cap.
    ///
    /// The only arm that opened the target; everything the bounded
    /// read needs flows from this handle.
    Handle(tokio::fs::File),

    /// No filesystem entry at the target.
    ///
    /// From the metadata gate, or from the open when the entry
    /// vanishes between the two.
    Absent,

    /// A non-regular entry at the target.
    ///
    /// Classified before the open — the anti-hang gate's whole point.
    NonRegular,

    /// A directory at the target.
    ///
    /// Kept apart from [`NonRegular`](GatedOpen::NonRegular) because
    /// no atomic replace can land on one.
    Directory,

    /// An entry whose size exceeds the cap.
    ///
    /// Classified from metadata before any byte is buffered.
    TooLarge,
}

/// Open an existing target through the metadata gates.
///
/// The pre-open metadata gate classifies absence, non-regular kinds,
/// and over-cap sizes without ever opening — the anti-hang and
/// anti-balloon gates — and the open maps its own absence race back to
/// [`GatedOpen::Absent`].
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the metadata probe or the
/// open fails for an I/O reason other than the classified outcomes.
async fn open_gated(full_path: &Path) -> Result<GatedOpen, ToolError> {
    match tokio::fs::metadata(full_path).await {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(GatedOpen::Absent),
        Err(err) => Err(ToolError::Execution(format!(
            "cannot inspect existing target {}: {err}",
            full_path.display()
        ))),
        Ok(meta) if !meta.is_file() => Ok(classify_special(&meta)),
        Ok(meta) if meta.len() > MAX_EXISTING_READ_BYTES => Ok(GatedOpen::TooLarge),
        Ok(_) => match tokio::fs::File::open(full_path).await {
            Ok(file) => Ok(GatedOpen::Handle(file)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(GatedOpen::Absent),
            Err(err) => Err(ToolError::Execution(format!(
                "cannot open existing target {}: {err}",
                full_path.display()
            ))),
        },
    }
}

/// The bounded read's outcome.
///
/// The take reads at most the cap plus one byte, and which side of the
/// cap the take landed on is the whole answer.
enum GatedBytes {
    /// The target's bytes, at most the cap.
    ///
    /// Ready for the consumers' decode or hash without further
    /// bounding.
    Bytes(Vec<u8>),

    /// The take saw more than the cap.
    ///
    /// The plus-one probe fired: the target is over the cap however
    /// its metadata sized it.
    TooLarge,
}

/// Read at most the cap plus one byte from an open handle.
///
/// The plus-one probes the boundary: a target that grew past the cap
/// between its metadata gate and this read is classified over-cap
/// from what the take actually saw, not from the stale size.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when the bounded read itself
/// fails for an I/O reason.
async fn read_bounded(file: tokio::fs::File, full_path: &Path) -> Result<GatedBytes, ToolError> {
    let mut bytes = Vec::new();
    file.take(MAX_EXISTING_READ_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .await
        .map_err(|err| {
            ToolError::Execution(format!(
                "cannot read existing target {}: {err}",
                full_path.display()
            ))
        })?;
    if bytes.len() > usize::try_from(MAX_EXISTING_READ_BYTES).unwrap_or(usize::MAX) {
        Ok(GatedBytes::TooLarge)
    } else {
        Ok(GatedBytes::Bytes(bytes))
    }
}

/// Classify a non-regular entry by kind.
///
/// Directories answer differently from other special files because an
/// atomic replace cannot land on one, while renaming over a FIFO or a
/// device node does displace it.
fn classify_special(meta: &std::fs::Metadata) -> GatedOpen {
    if meta.is_dir() {
        GatedOpen::Directory
    } else {
        GatedOpen::NonRegular
    }
}
