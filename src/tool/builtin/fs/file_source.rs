//! The filesystem [`ContentSource`] — and the family's read-side arm.
//!
//! The shared `read` tool stays policy-free; this source is where the
//! filesystem happens. It resolves every address under the session's
//! containment policy, bounds what it reads the way the live read path
//! always has, arms the staleness baseline with the bytes it serves —
//! text, image, and binary-content reads alike — and classifies the
//! bytes for the tool: an image extension returns raw bytes for the
//! tool's native multipart rendering, decodable UTF-8 returns text,
//! and anything else returns bytes the tool refuses as binary.

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;

use tokio::io::AsyncReadExt;

use crate::tool::ToolError;
use crate::tool::builtin::read::ContentSource;
use crate::tool::builtin::read::SourceContent;

use super::FileSession;
use super::resolve;
use super::resolve::ResolvePolicy;
use super::state;

/// Extensions whose content renders as a native image part.
///
/// Kept in step with the tool-side kind detection: the tool decides an
/// address is an image by extension, so the source must hand those
/// addresses raw bytes rather than attempt a text decode.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "gif"];

/// The filesystem content source the shared `read` tool reads through.
///
/// Constructed over the same [`FileSession`] the write tools operate
/// through, so every read arms the staleness baseline the next write
/// against the same file consults. The session's resolve policy governs
/// containment exactly as it does for the write tools.
#[derive(Debug, Clone)]
pub struct FileSource {
    /// The session whose policy, anchor, and baselines govern the read.
    ///
    /// Every resolution, containment check, and baseline recording this
    /// source performs goes through the session, so reads and writes against
    /// the same file share one view of what the model knows.
    session: FileSession,
}

impl FileSource {
    /// Build a filesystem source over `session`.
    ///
    /// Clones the session handle, sharing its baseline map — reads
    /// through this source are knowledge the write tools consult.
    #[must_use]
    pub fn new(session: FileSession) -> Self {
        Self { session }
    }

    /// The session this source reads through.
    ///
    /// Exposed for hosts that construct one source and want the same
    /// session handle for attaching to contexts or further tools.
    #[must_use]
    pub fn session(&self) -> &FileSession {
        &self.session
    }
}

impl ContentSource for FileSource {
    fn read<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
        Box::pin(async move { read_addressed(&self.session, path).await })
    }

    fn size<'a>(&'a self, path: &'a str) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + 'a>> {
        Box::pin(async move {
            let full = resolve_address(&self.session, path).ok()?;
            tokio::fs::metadata(&full).await.ok().map(|meta| meta.len())
        })
    }
}

/// Resolve `address` under the session's policy.
///
/// Shared by `read` and `size` so both consult the same containment
/// rules; a URL is refused the way every family tool refuses one.
///
/// # Errors
///
/// Returns `ToolError::InvalidInput` for a URL address or a path the
/// containment policy refuses.
fn resolve_address(session: &FileSession, address: &str) -> Result<PathBuf, ToolError> {
    resolve::reject_path_url("read", address)?;
    let cwd = session.cwd().to_path_buf();
    let mut full = resolve::resolve_path(address, &cwd, session.resolve_policy())?;
    if session.resolve_policy() == ResolvePolicy::Unrestricted {
        full = resolve::canonicalize_existing(&full)?;
    }
    Ok(full)
}

/// Read the bytes at `address`, arm the baseline, and classify the content.
///
/// Every read that serves the file's bytes arms the path's staleness
/// baseline with their content hash — text and image reads deliver
/// them, and a binary-content read reports the file while still arming
/// the guard. Reads that fail before any bytes are served record
/// nothing. Arming happens before classification on purpose: content
/// the model has seen stays guarded even if the classification or a
/// later windowing step rejects the read, and a re-read re-arms it.
///
/// # Errors
///
/// Returns `ToolError::InvalidInput` for a URL address, containment
/// refusals, and a directory target; `ToolError::Execution` when
/// the file cannot be opened, verified, or read.
///
/// Classification follows the tool-side kind contract: an image
/// extension returns [`SourceContent::Bytes`] for the tool's native
/// multipart rendering; decodable UTF-8 returns
/// [`SourceContent::Text`]; anything else returns
/// [`SourceContent::Bytes`], which the tool refuses as binary by name
/// and size.
async fn read_addressed(session: &FileSession, address: &str) -> Result<SourceContent, ToolError> {
    let full = resolve_address(session, address)?;
    let metadata = match tokio::fs::metadata(&full).await {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(ToolError::Execution(format!("File not found: {address}")));
        }
        Err(err) => {
            return Err(ToolError::Execution(format!(
                "cannot read {}: {err}",
                full.display()
            )));
        }
    };
    if !metadata.is_file() {
        return Err(ToolError::Execution(format!(
            "{address} is a directory, not a file."
        )));
    }

    let bytes = read_capped(session, &full).await?;
    session.record_baseline(&full, state::observe_bytes(&bytes));

    if is_image_address(&full) {
        return Ok(SourceContent::Bytes(bytes));
    }
    match String::from_utf8(bytes) {
        Ok(text) => Ok(SourceContent::Text(text)),
        Err(error) => {
            let bytes = error.into_bytes();
            Ok(SourceContent::Bytes(bytes))
        }
    }
}

/// Read the file's bytes, capped one past the shared read cap.
///
/// The cap bounds the allocation even when the file grows between the
/// metadata check and the read. Under the contained policy the opened
/// handle is verified against the session's pinned workspace anchor,
/// so a symlink swapped onto the workspace spelling after the session
/// was constructed cannot turn the read into a byte source outside
/// the pinned workspace.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] if the file cannot be opened or
/// read, when the contained handle check fails, and when the bytes
/// exceed the shared read cap.
async fn read_capped(session: &FileSession, full: &Path) -> Result<Vec<u8>, ToolError> {
    let file = tokio::fs::File::open(full)
        .await
        .map_err(|e| ToolError::Execution(format!("Failed to open file: {e}")))?;
    if session.resolve_policy() == ResolvePolicy::Contained {
        resolve::verify_handle_inside(&file, full, session.cwd(), Some(session.anchor()))?;
    }
    let cap = super::super::read::DEFAULT_MAX_SIZE_BYTES.saturating_add(1);
    let mut buf = Vec::with_capacity(usize::try_from(cap.min(8192)).unwrap_or(8192));
    file.take(cap)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| ToolError::Execution(format!("Failed to read file: {e}")))?;
    if buf.len() as u64 > super::super::read::DEFAULT_MAX_SIZE_BYTES {
        return Err(ToolError::Execution(format!(
            "File is too large to read: {}",
            full.display()
        )));
    }
    Ok(buf)
}

/// Whether `path`'s extension marks it as an image address.
///
/// Case-insensitive, matching the tool-side kind detection the
/// classification feeds.
fn is_image_address(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}
