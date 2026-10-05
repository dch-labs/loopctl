//! The `Glob` tool — gitignore-aware file globbing.
//!
//! Walks a source under ignore rules and returns the relative paths of
//! files matching a glob pattern. Pattern matching uses ripgrep's glob
//! engine (`ignore::overrides::Override`), which supports `*`, `?`,
//! `**`, character classes `[abc]`, and brace expansion `{a,b}`. A
//! pattern with no `/` matches the basename anywhere in the tree.

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::tool::builtin::search::SearchSource;
use crate::tool::builtin::search::output::MAX_INLINE_OUTPUT_BYTES;
use crate::tool::builtin::search::output::truncate_or_spill;
use crate::tool::builtin::search::resolve;
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Gitignore-aware file globbing over any [`SearchSource`].
///
/// Walks the target root under the source's ignore semantics and
/// returns the relative paths of files matching the glob pattern,
/// sorted alphabetically as a pretty-printed JSON array; an empty match
/// set is a success message, not an error. Both
/// [`is_read_only`](Tool::is_read_only) and
/// [`is_concurrency_safe`](Tool::is_concurrency_safe) are true.
pub struct GlobTool<S: SearchSource + 'static> {
    /// The source the tool walks.
    ///
    /// Shared by `Arc` so the blocking walk thread can hold a clone
    /// for the traversal's lifetime.
    source: Arc<S>,
}

impl<S: SearchSource + 'static> GlobTool<S> {
    /// Build a glob tool over `source`.
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

impl<S: SearchSource + 'static> Tool for GlobTool<S> {
    fn name(&self) -> &'static str {
        "Glob"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern in the specified directory. \
         Supports *, **, ? patterns. Returns matching file paths."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern selecting which files to return (e.g., '**/*.rs')"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in, defaulting to the current working directory"
                    }
                },
                "required": ["pattern"]
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
        Box::pin(async move { glob_inner(source, input, cwd, temp_dir).await })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// Body of the glob tool's [`Tool::call`].
///
/// Orchestrates: parse args → reject URL roots → resolve the root
/// against the context cwd → build the matcher → walk the source on a
/// blocking thread → sort → format. A missing `pattern` or an
/// unparseable pattern becomes [`ToolError::InvalidInput`]; an empty
/// match set is a success message. Everything the blocking walk
/// touches is owned, so the task needs no borrow of the tool.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for malformed arguments or an
/// unparseable pattern, and [`ToolError::Execution`] when the blocking
/// walk task joins unsuccessfully.
async fn glob_inner<S: SearchSource + 'static>(
    source: Arc<S>,
    input: Value,
    cwd: PathBuf,
    temp_dir: PathBuf,
) -> Result<ToolOutput, ToolError> {
    let parsed = parse_input(&input)?;
    let base_path = resolve::path_field(&input)?;
    resolve::reject_url("Glob", base_path)?;
    let base = resolve::resolve_root(&parsed.base_path, &cwd);
    let glob_override = build_glob_override(&base, &parsed.pattern)?;
    let pattern = parsed.pattern.clone();
    let matches =
        tokio::task::spawn_blocking(move || collect_matches(&*source, &base, &glob_override))
            .await
            .map_err(|error| ToolError::Execution(format!("Glob walk task failed: {error}")))?;

    if matches.is_empty() {
        return Ok(ToolOutput::text(format!(
            "No files found matching pattern: {pattern}"
        )));
    }

    let json = serde_json::to_string_pretty(&matches)
        .map_err(|error| ToolError::Execution(format!("Failed to serialize results: {error}")))?;
    Ok(truncate_or_spill(json, "glob", &temp_dir, MAX_INLINE_OUTPUT_BYTES).0)
}

/// Walk `base` on `source` and collect the relative paths of files
/// matching `glob_override`.
///
/// Runs on a blocking thread (called via `spawn_blocking`). Sorts the
/// result alphabetically before returning, as required by the tool
/// spec.
fn collect_matches(
    source: &dyn SearchSource,
    base: &Path,
    glob_override: &ignore::overrides::Override,
) -> Vec<String> {
    let mut matches = Vec::new();
    for entry in source.walk_files(base, &[], &[]) {
        let path = entry.path;
        let rel = rel_for_match(&path, base);
        if glob_override.matched(rel.as_path(), false).is_whitelist() {
            push_match_string(&mut matches, &path, base);
        }
    }
    matches.sort();
    matches
}

/// Parsed and validated glob input.
///
/// `pattern` is the user's glob, copied verbatim — no normalization;
/// the pattern feeds directly into the `ignore` matcher. `base_path`
/// is the search root, either as supplied or defaulted to `"."`.
#[derive(Debug)]
struct ParsedInput {
    /// The glob pattern supplied by the caller.
    ///
    /// Backslashes, brace expansion, and character classes all pass
    /// through unchanged.
    pattern: String,

    /// The directory to search in, defaulted to `"."` when absent.
    ///
    /// May be relative; the caller resolves it against the context
    /// cwd before walking, so relative roots reach the session's
    /// working directory rather than the process's.
    base_path: String,
}

/// Extract the `pattern` (required) and `path` (optional, `"."`).
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when `pattern` is missing or
/// not a string.
fn parse_input(input: &Value) -> Result<ParsedInput, ToolError> {
    let pattern = match input.get("pattern") {
        None => {
            return Err(ToolError::InvalidInput(
                "Missing 'pattern' field".to_string(),
            ));
        }
        Some(Value::String(pattern)) => pattern.clone(),
        Some(_) => {
            return Err(ToolError::InvalidInput(
                "'pattern' must be a string".to_string(),
            ));
        }
    };
    let base_path = input
        .get("path")
        .and_then(Value::as_str)
        .map_or_else(|| ".".to_string(), str::to_string);
    Ok(ParsedInput { pattern, base_path })
}

/// Build the whitelist matcher for one glob pattern.
///
/// The `ignore` crate's override engine gives gitignore-flavored
/// semantics — `*`, `?`, `**`, character classes, brace expansion —
/// with a `/`-free pattern matching basenames at any depth.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the pattern does not
/// parse, and [`ToolError::Execution`] when the matcher cannot be
/// built from a parsed pattern.
fn build_glob_override(
    base: &Path,
    pattern: &str,
) -> Result<ignore::overrides::Override, ToolError> {
    let mut builder = ignore::overrides::OverrideBuilder::new(base);
    builder.add(pattern).map_err(|error| {
        ToolError::InvalidInput(format!("Invalid glob pattern '{pattern}': {error}"))
    })?;
    builder
        .build()
        .map_err(|error| ToolError::Execution(format!("Failed to build glob matcher: {error}")))
}

/// The relative path to match against, falling back to the whole path.
///
/// The override matcher matches paths relative to its root; when the
/// entry is not under `base` the bare path is used so the matcher
/// still gets something to test against `/`-free patterns.
fn rel_for_match(path: &Path, base: &Path) -> PathBuf {
    path.strip_prefix(base)
        .map_or_else(|_| path.to_path_buf(), Path::to_path_buf)
}

/// Push a match string into the result set, relative to `base` when
/// possible.
///
/// Prefers the path relative to `base` (so `src/a.rs`, not the
/// absolute form); falls back to the whole path when the entry is not
/// under `base`.
fn push_match_string(matches: &mut Vec<String>, path: &Path, base: &Path) {
    let entry = path.strip_prefix(base).map_or_else(
        |_| path.to_string_lossy().into_owned(),
        |relative| relative.to_string_lossy().into_owned(),
    );
    matches.push(entry);
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
    use crate::tool::builtin::search::test_support::FakeSearchSource;
    use crate::tool::builtin::search::test_support::text;
    use serde_json::json;

    fn matcher(pattern: &str) -> ignore::overrides::Override {
        let mut builder = ignore::overrides::OverrideBuilder::new("/repo");
        builder.add(pattern).expect("valid pattern");
        builder.build().expect("builds")
    }

    #[test]
    fn matcher_recursive_star_matches_at_any_depth() {
        let override_ = matcher("**/*.rs");
        assert!(override_.matched("src/main.rs", false).is_whitelist());
        assert!(override_.matched("a/b/c.rs", false).is_whitelist());
        assert!(override_.matched("top.rs", false).is_whitelist());
    }

    #[test]
    fn matcher_extension_only_matches_basename_anywhere() {
        let override_ = matcher("*.rs");
        assert!(override_.matched("top.rs", false).is_whitelist());
        assert!(override_.matched("src/main.rs", false).is_whitelist());
        assert!(override_.matched("a/b/c.rs", false).is_whitelist());
        assert!(override_.matched("a/b/c.txt", false).is_ignore());
    }

    #[test]
    fn matcher_prefix_recursive_star_matches_zero_dirs() {
        let override_ = matcher("src/**/*.rs");
        assert!(override_.matched("src/a.rs", false).is_whitelist());
        assert!(override_.matched("src/a/b.rs", false).is_whitelist());
        assert!(override_.matched("other/a.rs", false).is_ignore());
    }

    #[test]
    fn matcher_character_class() {
        let override_ = matcher("*.[rs]s");
        assert!(override_.matched("a.rs", false).is_whitelist());
        assert!(override_.matched("a.ss", false).is_whitelist());
        assert!(override_.matched("a.cs", false).is_ignore());
    }

    #[test]
    fn matcher_brace_expansion() {
        let override_ = matcher("*.{rs,toml}");
        assert!(override_.matched("a.rs", false).is_whitelist());
        assert!(override_.matched("b.toml", false).is_whitelist());
        assert!(override_.matched("c.json", false).is_ignore());
    }

    #[test]
    fn matcher_invalid_pattern_rejected() {
        let result = build_glob_override(&PathBuf::from("/repo"), "[unclosed");
        assert!(result.is_err(), "unclosed class must error");
        match result {
            Err(ToolError::InvalidInput(message)) => assert!(
                message.contains("Invalid glob pattern"),
                "message should explain: {message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn parse_input_defaults_path_to_dot() {
        let input = json!({"pattern": "*.rs"});
        let parsed = parse_input(&input).unwrap();
        assert_eq!(parsed.pattern, "*.rs");
        assert_eq!(parsed.base_path, ".");
    }

    #[test]
    fn parse_input_uses_explicit_path() {
        let input = json!({"pattern": "*.rs", "path": "/other"});
        let parsed = parse_input(&input).unwrap();
        assert_eq!(parsed.base_path, "/other");
    }

    #[test]
    fn parse_input_rejects_a_wrong_typed_pattern_as_typed() {
        for wrong in [json!(42), json!(2.5), json!(null), json!(["x"])] {
            let error = parse_input(&json!({ "pattern": wrong }))
                .expect_err("a present-but-wrong-typed pattern is a correction prompt");
            match error {
                ToolError::InvalidInput(message) => assert_eq!(
                    message, "'pattern' must be a string",
                    "the model sent the field; it must be told the type, not that it is missing"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_input_missing_pattern_errors() {
        let input = json!({});
        assert!(parse_input(&input).is_err());
    }

    #[test]
    fn rel_for_match_strips_base_prefix() {
        let path = Path::new("/repo/src/a.rs");
        let base = Path::new("/repo");
        assert_eq!(rel_for_match(path, base), PathBuf::from("src/a.rs"));
    }

    #[test]
    fn rel_for_match_falls_back_to_absolute() {
        let path = Path::new("/other/a.rs");
        let base = Path::new("/repo");
        assert_eq!(rel_for_match(path, base), PathBuf::from("/other/a.rs"));
    }

    #[test]
    fn push_match_string_strips_base_prefix() {
        let mut out = Vec::new();
        push_match_string(&mut out, Path::new("/repo/src/a.rs"), Path::new("/repo"));
        assert_eq!(out, vec!["src/a.rs".to_string()]);
    }

    #[test]
    fn push_match_string_falls_back_to_absolute() {
        let mut out = Vec::new();
        push_match_string(&mut out, Path::new("/other/a.rs"), Path::new("/repo"));
        assert_eq!(out, vec!["/other/a.rs".to_string()]);
    }

    #[test]
    fn push_match_string_appends_across_calls() {
        let mut out = Vec::new();
        push_match_string(&mut out, Path::new("/repo/a.rs"), Path::new("/repo"));
        push_match_string(&mut out, Path::new("/repo/b.rs"), Path::new("/repo"));
        assert_eq!(out, vec!["a.rs".to_string(), "b.rs".to_string()]);
    }

    #[test]
    fn collect_matches_sorts_alphabetically() {
        let source = FakeSearchSource::with(&[
            ("/repo/zebra.rs", text("")),
            ("/repo/alpha.rs", text("")),
            ("/repo/mike.rs", text("")),
        ]);
        let override_ = matcher("*.rs");
        let got = collect_matches(&source, Path::new("/repo"), &override_);
        assert_eq!(
            got,
            vec![
                "alpha.rs".to_string(),
                "mike.rs".to_string(),
                "zebra.rs".to_string()
            ],
            "must be sorted"
        );
    }

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut context = ToolContext::default();
        context.cwd = cwd.to_string();
        context
    }

    #[tokio::test]
    async fn call_returns_sorted_pretty_json() {
        let tool = GlobTool::new(FakeSearchSource::with(&[
            ("/repo/src/b.rs", text("")),
            ("/repo/src/a.rs", text("")),
            ("/repo/Cargo.toml", text("")),
        ]));
        let output = tool
            .call(
                json!({"pattern": "**/*.rs", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<String> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed, vec!["src/a.rs".to_string(), "src/b.rs".to_string()]);
    }

    #[tokio::test]
    async fn call_empty_match_set_is_a_success_message() {
        let tool = GlobTool::new(FakeSearchSource::with(&[("/repo/a.txt", text(""))]));
        let output = tool
            .call(
                json!({"pattern": "*.rs", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        assert_eq!(
            output.text_content(),
            "No files found matching pattern: *.rs"
        );
    }

    #[tokio::test]
    async fn call_missing_pattern_is_invalid_input() {
        let tool = GlobTool::new(FakeSearchSource::with(&[]));
        let error = tool
            .call(json!({"path": "/repo"}), &ctx_in("/repo"))
            .await
            .expect_err("missing pattern");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn call_invalid_pattern_is_invalid_input() {
        let tool = GlobTool::new(FakeSearchSource::with(&[]));
        let error = tool
            .call(
                json!({"pattern": "[unclosed", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect_err("bad pattern");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn call_url_root_rejected_before_any_walk() {
        let shared = std::sync::Arc::new(FakeSearchSource::with(&[("/repo/a.rs", text(""))]));
        let tool = GlobTool::new(std::sync::Arc::clone(&shared));
        let error = tool
            .call(
                json!({"pattern": "*.rs", "path": "https://example.com/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect_err("url root");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
        assert_eq!(
            shared.walks.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a rejected root must never reach the source"
        );
    }

    #[tokio::test]
    async fn call_file_url_root_rejected_with_own_message() {
        let tool = GlobTool::new(FakeSearchSource::with(&[]));
        let error = tool
            .call(
                json!({"pattern": "*.rs", "path": "file:///repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect_err("file url root");
        match error {
            ToolError::InvalidInput(message) => assert!(
                message.contains("file:// URLs are not supported"),
                "{message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_relative_root_resolves_against_context_cwd() {
        let tool = GlobTool::new(FakeSearchSource::with(&[("/work/repo/a.rs", text(""))]));
        let output = tool
            .call(json!({"pattern": "*.rs", "path": "repo"}), &ctx_in("/work"))
            .await
            .expect("call");
        let parsed: Vec<String> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed, vec!["a.rs".to_string()]);
    }

    #[tokio::test]
    async fn call_rejects_a_wrong_typed_path() {
        let shared = std::sync::Arc::new(FakeSearchSource::with(&[("/repo/a.rs", text(""))]));
        let tool = GlobTool::new(std::sync::Arc::clone(&shared));
        let error = tool
            .call(json!({"pattern": "*.rs", "path": 42}), &ctx_in("/repo"))
            .await
            .expect_err("wrong-typed path");
        match error {
            ToolError::InvalidInput(message) => {
                assert!(message.contains("'path' must be a string"), "{message}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        assert_eq!(
            shared.walks.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a wrong-typed path must never reach the source"
        );
    }

    #[tokio::test]
    async fn call_no_pattern_leaves_source_untouched() {
        let shared = std::sync::Arc::new(FakeSearchSource::with(&[("/repo/a.rs", text(""))]));
        let tool = GlobTool::new(std::sync::Arc::clone(&shared));
        assert!(tool.call(json!({}), &ctx_in("/repo")).await.is_err());
        assert_eq!(
            shared.walks.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "input rejection must happen before the walk"
        );
    }
}
