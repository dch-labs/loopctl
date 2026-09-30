//! Ready-made tools that are generally useful but never load-bearing.
//!
//! Every tool here is an ordinary [`Tool`](crate::tool::Tool) with no
//! framework coupling:
//! registering one is the only way it enters a session, and the
//! `builtin_tools` feature is the only way it compiles into the build.
//! The module exists so hosts opting into small-model or local-model
//! setups have a curated place to find these tools, not so the framework
//! grows implicit registrations.
//!
//! # Available tools
//!
//! - [`ThinkTool`] — a scratchpad the model reasons into before acting.
//! - [`ReadTool`] — line-aware reading over a pluggable
//!   [`ContentSource`], with loud truncation.
//! - With the `fs_tools` feature: the filesystem family —
//!   [`WriteTool`](fs::WriteTool), [`EditTool`](fs::EditTool),
//!   [`MultiEditTool`](fs::MultiEditTool),
//!   [`FileViewerTool`](fs::FileViewerTool), and the
//!   [`FileSource`](fs::FileSource) filesystem content source — over
//!   one shared [`FileSession`](fs::FileSession).
//!
//! # Example
//!
//! ```
//! use loopctl::tool::ToolRegistry;
//! use loopctl::tool::builtin::ThinkTool;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(ThinkTool::new());
//! assert!(registry.contains("Think"));
//! ```

pub mod read;
pub mod think;

#[cfg(feature = "fs_tools")]
pub mod fs;

pub use read::{ContentSource, ReadTool};
pub use think::ThinkTool;

#[cfg(test)]
mod tests {
    use super::read::SourceContent;
    use super::{ContentSource, ReadTool, ThinkTool};
    use crate::tool::{Tool, ToolError};
    use std::future::Future;
    use std::pin::Pin;

    /// Minimal [`ContentSource`] standing in for a real backend.
    ///
    /// The census asks tools for their names, never for content, so a
    /// source whose every read fails is enough to construct `ReadTool`
    /// without a filesystem or fixture map.
    struct CensusSource;

    impl ContentSource for CensusSource {
        fn read<'a>(
            &'a self,
            path: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
            Box::pin(async move { Err(ToolError::Execution(format!("no such content: {path}"))) })
        }
    }

    /// Whether `name` matches the builtin wire-name convention.
    ///
    /// CamelCase here is the ASCII form: an uppercase ASCII first
    /// character and an all-ASCII-alphanumeric body — no lowercase
    /// starts, underscores, hyphens, or Unicode lookalikes.
    fn is_camel_case(name: &str) -> bool {
        let mut chars = name.chars();
        chars.next().is_some_and(|first| first.is_ascii_uppercase())
            && name.chars().all(|c| c.is_ascii_alphanumeric())
    }

    /// Assert the wire-name convention for one builtin tool.
    ///
    /// Thin wrapper so census call sites state the violated contract in
    /// one place.
    fn assert_camel_case(name: &str) {
        assert!(
            is_camel_case(name),
            "builtin tool names are CamelCase (got {name})"
        );
    }

    #[test]
    fn builtin_tool_names_are_camel_case() {
        assert_camel_case(ThinkTool::new().name());
        assert_camel_case(ReadTool::new(CensusSource).name());
        #[cfg(feature = "fs_tools")]
        {
            assert_camel_case(super::fs::WriteTool::new().name());
            assert_camel_case(super::fs::EditTool::new().name());
            assert_camel_case(super::fs::MultiEditTool::new().name());
            assert_camel_case(super::fs::FileViewerTool.name());
        }
    }

    #[test]
    fn the_census_predicate_rejects_non_ascii_lookalikes() {
        for name in ["Read", "Think", "FileViewer", "MultiEdit"] {
            assert!(is_camel_case(name), "{name} is the convention's shape");
        }
        for lookalike in ["Ｒead", "Édit", "read", "Read_file", "read-tool", ""] {
            assert!(
                !is_camel_case(lookalike),
                "the census predicate is ASCII CamelCase and rejects {lookalike:?}"
            );
        }
    }

    /// Doc-line fragments that name the `ContentSource` read method by
    /// the lowercase spelling, not the tool's wire name.
    ///
    /// The sweep below is fail-closed: every other backticked old
    /// spelling on a doc line is a stale wire name and fails the pin,
    /// so a fragment joins this list only when it is genuinely a method
    /// or module reference, classified in the task record.
    const METHOD_REFERENCE_FRAGMENTS: &[&str] = &[
        "Shared by `read` and `size`",
        "a `read` call counter",
        "How many times `read` has fired",
    ];

    /// Collect every `.rs` file under `dir`, sorted per directory for
    /// stable violation messages.
    ///
    /// Directories are walked depth-first; an unreadable directory or
    /// entry fails the sweep rather than silently shrinking it.
    fn rust_sources_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("the sweep source {dir:?} must be readable: {error}"))
            .map(|entry| entry.expect("a directory entry resolves").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                rust_sources_under(&path, out);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }

    #[test]
    fn no_builtin_doc_names_a_tool_by_the_old_spelling() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut sources = Vec::new();
        rust_sources_under(&root.join("src/tool/builtin"), &mut sources);
        sources.push(root.join("src/tool/builtin.rs"));
        sources.push(root.join("src/presets.rs"));
        assert!(
            !sources.is_empty(),
            "the sweep must cover the builtin tree, not an empty set"
        );
        let mut violations = Vec::new();
        for path in &sources {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()));
            for (idx, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                let is_doc = trimmed.starts_with("///") || trimmed.starts_with("//!");
                let classified = METHOD_REFERENCE_FRAGMENTS
                    .iter()
                    .any(|frag| line.contains(frag));
                if is_doc && !classified && (line.contains("`read`") || line.contains("`think`")) {
                    violations.push(format!("{}:{}: {}", path.display(), idx + 1, trimmed));
                }
            }
        }
        assert!(
            violations.is_empty(),
            "doc lines must name builtin tools by their CamelCase wire names, \
             not the old lowercase spellings:\n{}",
            violations.join("\n")
        );
    }

    #[test]
    fn the_file_viewer_contrast_scopes_the_read_claim_to_the_default_window() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(root.join("src/tool/builtin/fs/file_viewer.rs"))
            .unwrap_or_else(|error| panic!("the viewer source must be readable: {error}"));
        // Module-doc lines are re-joined so a sentence wrapped across `//!`
        // lines still matches the phrase a reader sees.
        let rendered = text
            .lines()
            .filter_map(|line| line.trim_start().strip_prefix("//!"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            rendered.contains("a default `Read` serves one bounded window"),
            "the FileViewer contrast scopes its Read claim to the default window — Read seeks \
             arbitrary windows via offset/limit and line_range"
        );
    }
}
