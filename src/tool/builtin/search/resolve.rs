//! Path resolution for the search tools.
//!
//! The search family resolves one argument — the search root — against
//! the tool context's working directory, rejecting URL spellings with
//! a redirect rather than a confusing filesystem error. Resolution is
//! lexical (no filesystem probing, so a not-yet-existing root produces
//! an honest empty walk, not an error), matching the address-opaque
//! philosophy of the [`ContentSource`](crate::tool::builtin::read::ContentSource)
//! seam these tools sit on.

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use serde_json::Value;

use crate::tool::ToolError;

/// Whether a string looks like a URL.
///
/// Case-insensitive `http://`, `https://`, or `file://` prefixes —
/// the spellings a model reaches for when it mistakes the search
/// root for a fetch target.
#[must_use]
pub fn is_url(path: &str) -> bool {
    path.get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        || path
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
        || path
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("file://"))
}

/// Whether a string looks like a `file://` URL.
///
/// The URL spelling of a local file, case-insensitive. Used by
/// [`reject_url`] to give these inputs their own redirect — pass a
/// filesystem path — since they are local files in the wrong spelling.
#[must_use]
pub fn is_file_url(path: &str) -> bool {
    path.get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("file://"))
}

/// Reject a URL where a filesystem path is required.
///
/// Every search tool guards its root argument with this check, so a
/// model that sends a URL receives a consistent error naming the tool
/// it called and asking for a filesystem path, rather than a confusing
/// filesystem error. `file://` URIs get their own message — they are
/// local files in the wrong spelling.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] naming `tool`, asking for a
/// filesystem path in both the remote-URL and `file://` cases.
pub fn reject_url(tool: &str, path: &str) -> Result<(), ToolError> {
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

/// Read the shared `path` field: present-and-string, absent as `"."`.
///
/// The family-wide input rule for the search root, so Glob and Tree
/// answer the same correction prompt the shared content parse gives
/// `Grep` and `CodeSearch` when the model sends a non-string `path`,
/// instead of silently defaulting to the whole working directory.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] naming `path` when the field
/// is present but not a string.
pub fn path_field(input: &Value) -> Result<&str, ToolError> {
    match input.get("path") {
        None => Ok("."),
        Some(Value::String(root)) => Ok(root.as_str()),
        Some(_) => Err(ToolError::InvalidInput(
            "'path' must be a string".to_string(),
        )),
    }
}

/// Resolve a possibly-relative `root` against `cwd`.
///
/// Relative roots are joined to `cwd`; absolute roots are taken as
/// given; the result is lexically normalized (`.` and `..` collapsed
/// without touching the filesystem, so a not-yet-existing root
/// resolves to the path it will occupy). No tilde expansion anywhere:
/// a leading `~` is an ordinary path component.
#[must_use]
pub fn resolve_root(root: &str, cwd: &Path) -> PathBuf {
    let path = Path::new(root);
    let joined = if path.is_relative() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    normalize_lexical(&joined)
}

/// Collapse `.` and `..` lexically, without touching the filesystem.
///
/// A `..` at the root pops nothing (the root is its own parent as far
/// as components go), matching how `Path::components` normalizes.
#[must_use]
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
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
    fn is_url_detects_http_https_and_file() {
        assert!(is_url("http://example.com"));
        assert!(is_url("https://example.com/page"));
        assert!(is_url("HTTPS://example.com/page"));
        assert!(is_url("file:///repo"));
        assert!(!is_url("/plain/path"));
        assert!(!is_url("./relative"));
    }

    #[test]
    fn is_file_url_matches_only_the_file_scheme() {
        assert!(is_file_url("file:///repo"));
        assert!(is_file_url("FILE:///repo"));
        assert!(!is_file_url("https://example.com"));
        assert!(!is_file_url("/repo/file"));
    }

    #[test]
    fn reject_url_redirects_urls_and_file_uris() {
        match reject_url("Glob", "https://x.dev") {
            Err(crate::tool::ToolError::InvalidInput(message)) => assert!(
                message.contains("URLs are not supported by the Glob tool"),
                "{message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        match reject_url("Grep", "file:///repo") {
            Err(crate::tool::ToolError::InvalidInput(message)) => assert!(
                message.contains("file:// URLs are not supported by the Grep tool"),
                "{message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        assert!(reject_url("Tree", "/repo").is_ok());
        assert!(reject_url("Glob", "./relative").is_ok());
    }

    #[test]
    fn resolve_root_joins_relative_and_keeps_absolute() {
        assert_eq!(
            resolve_root("sub", Path::new("/repo")),
            PathBuf::from("/repo/sub")
        );
        assert_eq!(
            resolve_root("/abs", Path::new("/repo")),
            PathBuf::from("/abs")
        );
    }

    #[test]
    fn resolve_root_collapses_dot_and_dot_dot_lexically() {
        assert_eq!(
            resolve_root("src/../lib", Path::new("/repo")),
            PathBuf::from("/repo/lib")
        );
        assert_eq!(
            resolve_root("./a/./b", Path::new("/repo")),
            PathBuf::from("/repo/a/b")
        );
        assert_eq!(
            resolve_root("../outside", Path::new("/repo")),
            PathBuf::from("/outside")
        );
    }

    #[test]
    fn resolve_root_does_not_expand_tilde() {
        assert_eq!(
            resolve_root("~/x", Path::new("/repo")),
            PathBuf::from("/repo/~/x")
        );
    }
}
