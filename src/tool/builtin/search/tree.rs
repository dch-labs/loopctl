//! The `Tree` tool — visual directory-tree rendering with depth limits.
//!
//! Renders a Unicode box-drawing tree (directories before files within
//! each level) up to `max_depth`, optionally including files and/or
//! filtering them by a glob pattern. Appends an `N directories, M
//! files` summary. Skips ignore-rule paths via the source's walk.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::tool::builtin::search::SearchSource;
use crate::tool::builtin::search::SourceEntry;
use crate::tool::builtin::search::output::MAX_INLINE_OUTPUT_BYTES;
use crate::tool::builtin::search::output::truncate_or_spill;
use crate::tool::builtin::search::resolve;
use crate::tool::builtin::search::walk::matches_any_glob;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Default maximum depth when the caller omits `max_depth`.
///
/// Shallow by design: a tree render is a orientation view, not an
/// inventory, and deeper structure belongs to a narrower root.
const DEFAULT_MAX_DEPTH: usize = 3;

/// Floor for a caller-supplied `max_depth`; values below this are raised.
///
/// Depth 1 shows only the direct children of the root, so a
/// clamped-to-1 request still yields a sensible listing rather
/// than an empty one.
const MIN_MAX_DEPTH: usize = 1;

/// Ceiling for a caller-supplied `max_depth`; values above this are lowered.
///
/// Bounds the walk cost and the render size; the ignore-aware
/// traversal is the expensive part, and a listing past this depth
/// stops being readable anyway.
const MAX_MAX_DEPTH: usize = 50;

/// Directory-tree rendering over any [`SearchSource`].
///
/// Displays a Unicode box-drawing tree up to `max_depth`, honoring the
/// source's ignore semantics, optionally including files and filtering
/// them by a glob pattern, with an `N directories, M files` summary.
/// Both [`is_read_only`](Tool::is_read_only) and
/// [`is_concurrency_safe`](Tool::is_concurrency_safe) are true.
pub struct TreeTool<S: SearchSource + 'static> {
    /// The source the tool walks.
    ///
    /// Shared by `Arc` so the blocking walk thread can hold a clone
    /// for the traversal's lifetime.
    source: Arc<S>,
}

impl<S: SearchSource + 'static> TreeTool<S> {
    /// Build a tree tool over `source`.
    ///
    /// The filesystem implementation is
    /// [`FsSearchSource`](crate::tool::builtin::search::FsSearchSource);
    /// any other [`SearchSource`] serves the same contract.
    #[must_use]
    pub fn new(source: S) -> Self {
        Self {
            source: Arc::new(source),
        }
    }
}

impl<S: SearchSource + 'static> Tool for TreeTool<S> {
    fn name(&self) -> &'static str {
        "Tree"
    }

    fn description(&self) -> &'static str {
        "Display directory tree structure with depth limits and filtering. \
         Respects .gitignore."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory to display, defaulting to the current working directory"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to display (default 3, clamped to 1-50)"
                    },
                    "include_files": {
                        "type": "boolean",
                        "description": "Whether to include files in addition to directories (default true)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern filtering which files are shown (directories are always kept)"
                    }
                }
            }),
        )
    }

    fn call(
        &self,
        input: Value,
        context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let source = Arc::clone(&self.source);
        let cwd = PathBuf::from(&context.cwd);
        let temp_dir = PathBuf::from(&context.temp_dir);
        Box::pin(async move { tree_inner(source, input, cwd, temp_dir).await })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// Body of the tree tool's [`Tool::call`].
///
/// Input parsing, URL rejection, and root resolution run on the async
/// side (all `O(1)`); the metadata probe, traversal, filtering,
/// rendering, and any spill run on a blocking thread via
/// [`tokio::task::spawn_blocking`] so a large tree cannot stall the
/// executor — the same split the three siblings use. A non-existent
/// root is an error-text success naming the path; a root that exists
/// but is not a directory is [`ToolError::InvalidInput`]. Everything
/// the blocking task touches is owned, so it needs no borrow of the
/// tool.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for a URL root or a root that
/// exists but is not a directory, and [`ToolError::Execution`] when
/// the root's metadata cannot be read or the blocking task joins
/// unsuccessfully.
async fn tree_inner<S: SearchSource + 'static>(
    source: Arc<S>,
    input: Value,
    cwd: PathBuf,
    temp_dir: PathBuf,
) -> Result<ToolOutput, ToolError> {
    let base_path = resolve::path_field(&input)?.to_string();
    resolve::reject_url("Tree", &base_path)?;

    let max_depth = max_depth_field(&input)?.clamp(MIN_MAX_DEPTH, MAX_MAX_DEPTH);
    let include_files = include_files_field(&input)?;
    let pattern = pattern_field(&input)?;
    let full_path = resolve::resolve_root(&base_path, &cwd);

    tokio::task::spawn_blocking(move || {
        let metadata = match std::fs::metadata(&full_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ToolOutput::error_text(format!(
                    "Path does not exist: {base_path}"
                )));
            }
            Err(error) => return Err(ToolError::Execution(error.to_string())),
        };
        if !metadata.is_dir() {
            return Err(ToolError::InvalidInput(format!(
                "Path is not a directory: {base_path}"
            )));
        }

        let entries = source.walk_entries(&full_path, Some(max_depth));
        let entries_were_empty = entries.is_empty();
        let filtered = filter_entries(entries, &full_path, include_files, pattern.as_deref());

        if filtered.is_empty() {
            let message = if entries_were_empty {
                format!("Empty directory: {base_path}")
            } else {
                format!("No matching entries in: {base_path}")
            };
            return Ok(ToolOutput::error_text(message));
        }

        let tree = format_tree(&full_path, &base_path, &filtered);
        let summary = format_summary(&filtered);

        Ok(truncate_or_spill(
            format!("{tree}\n\n{summary}"),
            "tree",
            &temp_dir,
            MAX_INLINE_OUTPUT_BYTES,
        )
        .0)
    })
    .await
    .map_err(|error| ToolError::Execution(format!("Tree walk task failed: {error}")))?
}

/// Read the `max_depth` field: absent → the default, integer → kept.
///
/// The family-wide numeric-input rule (the shared search parse and
/// the shell family's typed getters): absent is the default, a
/// non-negative integer is taken, anything else is a correction
/// prompt naming the field — a model that sends `"max_depth":
/// "10"` must be told, not silently shown depth 3. An integer
/// above the clamp ceiling — including one a narrower `usize`
/// cannot represent, as on a 32-bit host receiving `4294967296` —
/// meets the ceiling here, never a silent default.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a non-negative integer.
fn max_depth_field(input: &Value) -> Result<usize, ToolError> {
    match input.get("max_depth") {
        None => Ok(DEFAULT_MAX_DEPTH),
        Some(value) if value.is_u64() => {
            let raw = value.as_u64().unwrap_or(0);
            Ok(match usize::try_from(raw) {
                Ok(depth) => depth.min(MAX_MAX_DEPTH),
                Err(_) => MAX_MAX_DEPTH,
            })
        }
        Some(_) => Err(ToolError::InvalidInput(
            "'max_depth' must be a positive integer".to_string(),
        )),
    }
}

/// Read the `include_files` flag: absent → `true`, boolean → kept.
///
/// The same present-but-wrong-typed rule: a string `"false"` must
/// be corrected, not silently rendered with the files it asked to
/// hide.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a boolean.
fn include_files_field(input: &Value) -> Result<bool, ToolError> {
    match input.get("include_files") {
        None => Ok(true),
        Some(value) if value.is_boolean() => Ok(value.as_bool().unwrap_or(true)),
        Some(_) => Err(ToolError::InvalidInput(
            "'include_files' must be a boolean".to_string(),
        )),
    }
}

/// Read the `pattern` field: absent → `None`, string → kept.
///
/// The same present-but-wrong-typed rule: a numeric pattern must be
/// corrected, not silently dropped so the listing shows everything.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a string.
fn pattern_field(input: &Value) -> Result<Option<String>, ToolError> {
    match input.get("pattern") {
        None => Ok(None),
        Some(Value::String(pattern)) => Ok(Some(pattern.clone())),
        Some(_) => Err(ToolError::InvalidInput(
            "'pattern' must be a string".to_string(),
        )),
    }
}

/// Filter the walked entries by `include_files` and the optional pattern.
///
/// Directories are always kept (the pattern filter is files-only).
/// When the pattern contains a `/`, it matches against the path
/// relative to `root`; otherwise it matches the filename only.
fn filter_entries(
    entries: Vec<SourceEntry>,
    root: &Path,
    include_files: bool,
    pattern: Option<&str>,
) -> Vec<SourceEntry> {
    let pat_vec: Vec<String> = pattern.map_or_else(Vec::new, |p| vec![p.to_string()]);
    entries
        .into_iter()
        .filter(|entry| {
            if entry.is_dir {
                true
            } else if !include_files {
                false
            } else if pat_vec.is_empty() {
                true
            } else if pat_vec.first().is_some_and(|pattern| pattern.contains('/')) {
                let relative = entry
                    .path
                    .strip_prefix(root)
                    .ok()
                    .and_then(|rel| rel.to_str())
                    .unwrap_or("");
                matches_any_glob(relative, &pat_vec)
            } else {
                let name = entry
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                matches_any_glob(name, &pat_vec)
            }
        })
        .collect()
}

/// Render `entries` as a Unicode box-drawing tree rooted at `root`.
///
/// Directories are listed before files within each level, both groups
/// sorted alphabetically, with the standard `tree(1)`-style connectors
/// the model expects.
fn format_tree(root: &Path, display_name: &str, entries: &[SourceEntry]) -> String {
    let mut children_by_parent: HashMap<PathBuf, Vec<&SourceEntry>> = HashMap::new();
    for entry in entries {
        if let Some(parent) = entry.path.parent() {
            children_by_parent
                .entry(parent.to_path_buf())
                .or_default()
                .push(entry);
        }
    }
    for children in children_by_parent.values_mut() {
        children.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then_with(|| a.path.file_name().cmp(&b.path.file_name()))
        });
    }

    let mut output = String::new();
    output.push_str(display_name);
    output.push('/');

    let root_children: &[&SourceEntry] = children_by_parent.get(root).map_or(&[], Vec::as_slice);
    render_children(root_children, &children_by_parent, "", &mut output);
    output
}

/// Recursively render one level of the tree.
///
/// Appends each child with the correct connector and indentation
/// prefix, recursing into directories.
fn render_children(
    children: &[&SourceEntry],
    children_by_parent: &HashMap<PathBuf, Vec<&SourceEntry>>,
    prefix: &str,
    output: &mut String,
) {
    let count = children.len();
    for (index, child) in children.iter().enumerate() {
        let is_last = index == count.saturating_sub(1);
        let connector = if is_last { "└── " } else { "├── " };
        let child_prefix = if is_last { "    " } else { "│   " };
        let name = child
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("?");

        output.push('\n');
        output.push_str(prefix);
        output.push_str(connector);
        output.push_str(name);
        if child.is_dir {
            output.push('/');
            let sub_children: &[&SourceEntry] = children_by_parent
                .get(&child.path)
                .map_or(&[], Vec::as_slice);
            if !sub_children.is_empty() {
                let new_prefix = format!("{prefix}{child_prefix}");
                render_children(sub_children, children_by_parent, &new_prefix, output);
            }
        }
    }
}

/// Build the `N director(y|ies), M file(s)` summary line.
///
/// Directory and file counts of the filtered entry list, each noun
/// singularized only when its count is exactly one and pluralized
/// otherwise (including zero).
fn format_summary(filtered: &[SourceEntry]) -> String {
    let dir_count = filtered.iter().filter(|entry| entry.is_dir).count();
    let file_count = filtered.iter().filter(|entry| !entry.is_dir).count();
    let dir_word = if dir_count == 1 {
        "directory"
    } else {
        "directories"
    };
    let file_word = if file_count == 1 { "file" } else { "files" };
    format!("{dir_count} {dir_word}, {file_count} {file_word}")
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::tool::builtin::search::FsSearchSource;
    use serde_json::json;

    #[test]
    fn format_summary_singular() {
        let one_of_each = vec![
            SourceEntry {
                path: PathBuf::from("/d"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/f"),
                is_dir: false,
            },
        ];
        assert_eq!(format_summary(&one_of_each), "1 directory, 1 file");
    }

    #[test]
    fn format_summary_plural() {
        let two_dirs_three_files = vec![
            SourceEntry {
                path: PathBuf::from("/d1"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/d2"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/f1"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/f2"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/f3"),
                is_dir: false,
            },
        ];
        assert_eq!(
            format_summary(&two_dirs_three_files),
            "2 directories, 3 files"
        );
    }

    #[test]
    fn format_summary_zero() {
        assert_eq!(format_summary(&[]), "0 directories, 0 files");
    }

    #[test]
    fn format_tree_empty_entries() {
        let root = Path::new("/repo");
        let out = format_tree(root, ".", &[]);
        assert_eq!(out, "./");
    }

    #[test]
    fn format_tree_dirs_before_files() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/file_a"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/dir_b"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/file_c"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/dir_a"),
                is_dir: true,
            },
        ];
        let out = format_tree(Path::new("/repo"), ".", &entries);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "./");
        assert!(lines[1].contains("dir_a/"), "{}", lines[1]);
        assert!(lines[2].contains("dir_b/"), "{}", lines[2]);
        assert!(lines[3].contains("file_a"), "{}", lines[3]);
        assert!(lines[4].contains("file_c"), "{}", lines[4]);
    }

    #[test]
    fn format_tree_last_child_connector() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/a"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/b"),
                is_dir: false,
            },
        ];
        let out = format_tree(Path::new("/repo"), ".", &entries);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].starts_with("├── a/"), "{}", lines[1]);
        assert!(lines[2].starts_with("└── b"), "{}", lines[2]);
    }

    #[test]
    fn format_tree_nested_prefixes() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/dir"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/dir/inner"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/dir/leaf.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/top.rs"),
                is_dir: false,
            },
        ];
        let out = format_tree(Path::new("/repo"), ".", &entries);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].starts_with("├── dir/"), "{}", lines[1]);
        assert!(lines[2].starts_with("│   ├── inner/"), "{}", lines[2]);
        assert!(lines[3].starts_with("│   └── leaf.rs"), "{}", lines[3]);
        assert!(lines[4].starts_with("└── top.rs"), "{}", lines[4]);
    }

    #[test]
    fn filter_entries_keeps_all_without_pattern() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/d"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/a.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/b.txt"),
                is_dir: false,
            },
        ];
        let filtered = filter_entries(entries, Path::new("/repo"), true, None);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn filter_entries_include_files_false_keeps_only_dirs() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/d"),
                is_dir: true,
            },
            SourceEntry {
                path: PathBuf::from("/repo/a.rs"),
                is_dir: false,
            },
        ];
        let filtered = filter_entries(entries, Path::new("/repo"), false, Some("*.rs"));
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].is_dir);
    }

    #[test]
    fn filter_entries_slash_pattern_matches_relative_paths() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/src/lib.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/srcextra/leak.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/top.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/src/nested/mod.rs"),
                is_dir: false,
            },
        ];
        let filtered = filter_entries(entries, Path::new("/repo"), true, Some("src/*.rs"));
        let paths: Vec<String> = filtered
            .iter()
            .map(|entry| {
                entry
                    .path
                    .strip_prefix("/repo")
                    .map(|rel| rel.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        // `*` is the walker's char-wise wildcard and crosses `/`, so a
        // nested match under `src/` is in-contract; the discriminating
        // exclusions are the `srcextra/` prefix collision and the
        // pattern-less top-level file.
        assert_eq!(
            paths,
            vec!["src/lib.rs".to_string(), "src/nested/mod.rs".to_string()],
            "{paths:?}"
        );
    }

    #[test]
    fn filter_entries_pattern_matches_filename_only() {
        let entries = vec![
            SourceEntry {
                path: PathBuf::from("/repo/a.rs"),
                is_dir: false,
            },
            SourceEntry {
                path: PathBuf::from("/repo/b.txt"),
                is_dir: false,
            },
        ];
        let filtered = filter_entries(entries, Path::new("/repo"), true, Some("*.rs"));
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].path.ends_with("a.rs"));
    }

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut context = ToolContext::default();
        context.cwd = cwd.to_string();
        context
    }

    #[tokio::test]
    async fn call_renders_tree_with_summary_over_the_filesystem() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/lib.rs"), "").unwrap();
        std::fs::write(tmp.path().join("README.md"), "").unwrap();
        let tool = TreeTool::new(FsSearchSource);
        let output = tool
            .call(
                json!({"path": tmp.path().to_string_lossy(), "max_depth": 2}),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .expect("call");
        let text = output.text_content();
        assert!(text.contains("src/"), "{text}");
        assert!(text.contains("lib.rs"), "{text}");
        assert!(text.contains("README.md"), "{text}");
        assert!(text.contains("1 directory, 2 files"), "{text}");
    }

    #[tokio::test]
    async fn call_depth_one_lists_direct_children_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("a/b")).unwrap();
        std::fs::write(tmp.path().join("a/b/deep.rs"), "").unwrap();
        std::fs::write(tmp.path().join("top.rs"), "").unwrap();
        let tool = TreeTool::new(FsSearchSource);
        let output = tool
            .call(
                json!({"path": tmp.path().to_string_lossy(), "max_depth": 1}),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .expect("call");
        let text = output.text_content();
        assert!(text.contains("a/"), "{text}");
        assert!(text.contains("top.rs"), "{text}");
        assert!(
            !text.contains("deep.rs"),
            "depth 1 hides grandchildren: {text}"
        );
    }

    #[tokio::test]
    async fn call_missing_path_is_error_text_success() {
        let tool = TreeTool::new(FsSearchSource);
        let output = tool
            .call(json!({"path": "/nonexistent-dch-tree-path"}), &ctx_in("/"))
            .await
            .expect("call");
        assert_eq!(
            output.text_content(),
            "Path does not exist: /nonexistent-dch-tree-path"
        );
    }

    #[tokio::test]
    async fn call_path_is_a_file_is_invalid_input() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("plain.txt");
        std::fs::write(&file, "").unwrap();
        let tool = TreeTool::new(FsSearchSource);
        let error = tool
            .call(
                json!({"path": file.to_string_lossy()}),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .expect_err("file root");
        match error {
            ToolError::InvalidInput(message) => {
                assert!(message.contains("not a directory"), "{message}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_rejects_wrong_typed_max_depth_include_files_and_pattern() {
        let tool = TreeTool::new(FsSearchSource);
        for wrong in [json!("10"), json!(2.5), json!(null), json!(true)] {
            let error = tool
                .call(json!({ "max_depth": wrong }), &ctx_in("/"))
                .await
                .expect_err("a present-but-wrong-typed depth is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => {
                    assert_eq!(message, "'max_depth' must be a positive integer");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
        for wrong in [json!("true"), json!(2.5), json!(null), json!(7)] {
            let error = tool
                .call(json!({ "include_files": wrong }), &ctx_in("/"))
                .await
                .expect_err("a present-but-wrong-typed flag is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => {
                    assert_eq!(message, "'include_files' must be a boolean");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
        for wrong in [json!(123), json!(2.5), json!(null), json!(true)] {
            let error = tool
                .call(json!({ "pattern": wrong }), &ctx_in("/"))
                .await
                .expect_err("a present-but-wrong-typed pattern is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => {
                    assert_eq!(message, "'pattern' must be a string");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn absent_optional_fields_still_take_their_defaults() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "").unwrap();
        let output = TreeTool::new(FsSearchSource)
            .call(
                json!({ "path": tmp.path().to_string_lossy() }),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .unwrap();
        assert!(
            output.text_content().contains("a.rs"),
            "absent means the defaults (files included, no pattern), never an error: {}",
            output.text_content()
        );
    }

    #[test]
    fn an_absurd_max_depth_clamps_never_silently_defaults() {
        for raw in [u64::from(u32::MAX), 1u64 << 32, u64::MAX] {
            let depth = max_depth_field(&json!({ "max_depth": raw })).unwrap();
            assert_ne!(
                depth, DEFAULT_MAX_DEPTH,
                "an absurd depth must meet the clamp ceiling on every pointer width, never a silent default (raw {raw})"
            );
            assert_eq!(
                depth, MAX_MAX_DEPTH,
                "an out-of-range depth meets the documented clamp (raw {raw})"
            );
        }
    }

    #[tokio::test]
    async fn call_rejects_a_wrong_typed_path() {
        let tool = TreeTool::new(FsSearchSource);
        let error = tool
            .call(json!({"path": 42}), &ctx_in("/"))
            .await
            .expect_err("wrong-typed path");
        match error {
            ToolError::InvalidInput(message) => {
                assert!(message.contains("'path' must be a string"), "{message}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_url_root_rejected() {
        let tool = TreeTool::new(FsSearchSource);
        let error = tool
            .call(json!({"path": "https://example.com/repo"}), &ctx_in("/"))
            .await
            .expect_err("url root");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn call_empty_directory_is_error_text_success() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tool = TreeTool::new(FsSearchSource);
        let output = tool
            .call(
                json!({"path": tmp.path().to_string_lossy()}),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .expect("call");
        assert!(
            output.text_content().starts_with("Empty directory:"),
            "{}",
            output.text_content()
        );
    }

    #[tokio::test]
    async fn call_no_matching_entries_when_pattern_filters_everything() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "").unwrap();
        let tool = TreeTool::new(FsSearchSource);
        let output = tool
            .call(
                json!({
                    "path": tmp.path().to_string_lossy(),
                    "pattern": "*.nomatch"
                }),
                &ctx_in(&tmp.path().to_string_lossy()),
            )
            .await
            .expect("call");
        assert!(
            output.text_content().starts_with("No matching entries in:"),
            "{}",
            output.text_content()
        );
    }
}
