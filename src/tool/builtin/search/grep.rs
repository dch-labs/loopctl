//! The `Grep` tool — regex content search across a source tree.
//!
//! Walks a source under ignore rules, skipping binary files, reads
//! each remaining file, and returns every line that matches the
//! user-supplied regex as a JSON object `{file, line, content}`.
//! Compiled patterns are cached process-globally through the shared
//! regex cache so repeated calls with the same pattern skip
//! recompilation.

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use regex::Regex;
use serde_json::Value;
use serde_json::json;

use crate::tool::builtin::search::SearchSource;
use crate::tool::builtin::search::content::Match;
use crate::tool::builtin::search::content::SearchJob;
use crate::tool::builtin::search::content::compile_pattern;
use crate::tool::builtin::search::content::get_usize;
use crate::tool::builtin::search::content::no_matches_message;
use crate::tool::builtin::search::content::{parse_input, relative_file, run};
use crate::tool::builtin::search::output::MAX_INLINE_OUTPUT_BYTES;
use crate::tool::builtin::search::output::truncate_or_spill;
use crate::tool::builtin::search::resolve;
use crate::tool::{DisplayHint, Retention, Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Default per-file match cap when the caller omits `max_matches`.
///
/// Guards against one pathological file flooding the result set;
/// the caller can tighten or raise it per call.
const DEFAULT_MAX_MATCHES: usize = 100;

/// Default total match cap across all files when the caller omits `max_results`.
///
/// Keeps a broad search from flooding the model's context in one
/// call; a caller wanting more asks explicitly and is still bounded
/// by [`MAX_RESULTS_CAP`].
const DEFAULT_MAX_RESULTS: usize = 1000;

/// Hard ceiling `max_results` is clamped to, whatever the caller asks.
///
/// Bounds the worst-case context spend per call — even an explicit
/// huge request stops here, the excess matches staying unlisted
/// rather than silently admitted.
const MAX_RESULTS_CAP: usize = 1000;

/// Regex content search over any [`SearchSource`].
///
/// Walks the target root under the source's ignore semantics and
/// returns every matching line as a JSON object
/// `{file, line, content}`, spilling oversized results to a file
/// under the context's temp directory. Both
/// [`is_read_only`](Tool::is_read_only) and
/// [`is_concurrency_safe`](Tool::is_concurrency_safe) are true.
pub struct GrepTool<S: SearchSource + 'static> {
    /// The source the tool searches.
    ///
    /// Shared by `Arc` so the blocking walk thread can hold a clone
    /// for the traversal's lifetime.
    source: Arc<S>,
}

impl<S: SearchSource + 'static> GrepTool<S> {
    /// Build a grep tool over `source`.
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

impl<S: SearchSource + 'static> Tool for GrepTool<S> {
    fn name(&self) -> &'static str {
        "Grep"
    }

    fn description(&self) -> &'static str {
        "Search for a regex pattern in file contents within a directory. \
         Returns matching lines with file paths and line numbers."
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
                        "description": "The regular expression to search file contents for"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in, defaulting to the current working directory"
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns restricting the search to matching files (e.g., ['*.rs'])"
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns removing files from the search (e.g., ['*.lock'])"
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum matches per file (default 100)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum total matches (default 1000, hard-capped at 1000)"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match without regard to case (default false)"
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
        Box::pin(async move { grep_inner(source, input, cwd, temp_dir).await })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// Body of the grep tool's [`Tool::call`].
///
/// Orchestrates parse → compile → walk → render. An empty match set is
/// a success message; bad arguments and invalid patterns become
/// [`ToolError::InvalidInput`]. Everything the blocking walk touches
/// is owned, so the task needs no borrow of the tool.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for malformed arguments or an
/// unparseable pattern, and [`ToolError::Execution`] when the blocking
/// walk task joins unsuccessfully.
async fn grep_inner<S: SearchSource + 'static>(
    source: Arc<S>,
    input: Value,
    cwd: PathBuf,
    temp_dir: PathBuf,
) -> Result<ToolOutput, ToolError> {
    let parsed = parse_input(&input, DEFAULT_MAX_RESULTS)?;
    let max_results = parsed.max_results.min(MAX_RESULTS_CAP);
    let max_matches = get_usize(&input, "max_matches")?
        .unwrap_or(DEFAULT_MAX_MATCHES)
        .max(1);

    resolve::reject_url("Grep", &parsed.base_path)?;

    let regex = compile_pattern(&parsed.pattern, parsed.case_insensitive)?;
    let base = resolve::resolve_root(&parsed.base_path, &cwd);
    let job = SearchJob {
        regex,
        include: parsed.include_patterns,
        exclude: parsed.exclude_patterns,
        max_results,
        per_file_cap: Some(max_matches),
        base,
        pattern: parsed.pattern.clone(),
    };
    let matches = tokio::task::spawn_blocking(move || {
        run(&*source, &job, |regex, content, file_path, base, limit| {
            scan_file(regex, content, file_path, base, limit)
        })
    })
    .await
    .map_err(|error| ToolError::Execution(format!("Grep walk task failed: {error}")))?;

    if matches.is_empty() {
        return Ok(no_matches_message(&parsed.pattern));
    }

    render(&matches, &temp_dir)
}

/// Scan one file's content for regex matches; return up to `limit`.
///
/// Computes a path relative to the search base, enumerates lines with
/// 1-indexed numbers, and pushes one [`Match`] per matching line, in
/// order, stopping once `limit` matches are collected.
fn scan_file(
    regex: &Regex,
    content: &str,
    file_path: &Path,
    base_path: &Path,
    limit: usize,
) -> Vec<Match> {
    let rel_path = relative_file(file_path, base_path);
    let mut results = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        if results.len() >= limit {
            break;
        }
        if regex.is_match(line) {
            results.push(Match {
                file: rel_path.clone(),
                line: line_num.saturating_add(1),
                content: line.to_string(),
            });
        }
    }
    results
}

/// Render `matches` as a pretty JSON array, spilling when oversized.
///
/// Maps each [`Match`] to a JSON object `{"file", "line", "content"}`
/// in collection order and hands the serialized array to
/// [`truncate_or_spill`]. The key order is the contract the model
/// expects. A serialization fault — unreachable for `String` and
/// `usize` fields in practice — surfaces as
/// [`ToolError::Execution`], the same side glob's render takes, so
/// the family never answers a failed render as success-shaped text.
///
/// # Errors
///
/// Returns [`ToolError::Execution`] when `serde_json` cannot encode
/// the match array.
fn render(matches: &[Match], temp_dir: &Path) -> Result<ToolOutput, ToolError> {
    let json_array: Vec<Value> = matches
        .iter()
        .map(|entry| {
            json!({
                "file": entry.file,
                "line": entry.line,
                "content": entry.content,
            })
        })
        .collect();
    let json = serde_json::to_string_pretty(&json_array)
        .map_err(|error| ToolError::Execution(format!("Failed to serialize results: {error}")))?;
    let (output, inline) = truncate_or_spill(json, "grep", temp_dir, MAX_INLINE_OUTPUT_BYTES);
    let output = output.with_retention(Retention::Requery);
    if inline {
        Ok(output.with_hint(DisplayHint::Json))
    } else {
        Ok(output)
    }
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
    use crate::tool::builtin::search::content::parse_input;
    use crate::tool::builtin::search::test_support::FakeFile;
    use crate::tool::builtin::search::test_support::FakeSearchSource;
    use crate::tool::builtin::search::test_support::text;
    use serde_json::json;

    #[test]
    fn parse_defaults() {
        let common = parse_input(&json!({"pattern": "foo"}), DEFAULT_MAX_RESULTS).unwrap();
        assert_eq!(common.pattern, "foo");
        assert_eq!(common.base_path, ".");
        assert!(!common.case_insensitive);
        assert_eq!(common.max_results, DEFAULT_MAX_RESULTS);
        assert!(common.include_patterns.is_empty());
        assert!(common.exclude_patterns.is_empty());
        let max_matches = get_usize(&json!({}), "max_matches")
            .unwrap()
            .unwrap_or(DEFAULT_MAX_MATCHES);
        assert_eq!(max_matches, DEFAULT_MAX_MATCHES);
    }

    #[test]
    fn parse_all_fields() {
        let input = json!({
            "pattern": "needle",
            "path": "src",
            "case_insensitive": true,
            "max_results": 10,
            "max_matches": 5,
            "include_patterns": ["*.rs"],
            "exclude_patterns": ["*.test.rs"]
        });
        let common = parse_input(&input, DEFAULT_MAX_RESULTS).unwrap();
        assert_eq!(common.base_path, "src");
        assert!(common.case_insensitive);
        assert_eq!(common.max_results, 10);
        assert_eq!(common.include_patterns, vec!["*.rs".to_string()]);
        assert_eq!(common.exclude_patterns, vec!["*.test.rs".to_string()]);
        let max_matches = get_usize(&input, "max_matches")
            .unwrap()
            .unwrap_or(DEFAULT_MAX_MATCHES);
        assert_eq!(max_matches, 5);
    }

    #[test]
    fn parse_rejects_a_wrong_typed_pattern_as_typed() {
        for wrong in [json!(42), json!(2.5), json!(null), json!(["x"])] {
            let error = parse_input(&json!({ "pattern": wrong }), DEFAULT_MAX_RESULTS)
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
    fn parse_missing_pattern_errors() {
        assert!(parse_input(&json!({}), DEFAULT_MAX_RESULTS).is_err());
    }

    #[test]
    fn parse_rejects_negative_max_matches() {
        assert!(get_usize(&json!({"pattern": "x", "max_matches": -5}), "max_matches").is_err());
    }

    #[test]
    fn parse_rejects_non_integer_max_matches() {
        assert!(
            get_usize(
                &json!({"pattern": "x", "max_matches": "abc"}),
                "max_matches"
            )
            .is_err()
        );
    }

    #[test]
    fn parse_rejects_non_array_include_patterns() {
        assert!(
            parse_input(
                &json!({"pattern": "x", "include_patterns": "foo"}),
                DEFAULT_MAX_RESULTS,
            )
            .is_err()
        );
    }

    #[test]
    fn parse_drops_non_string_array_elements() {
        let input = json!({"pattern": "x", "include_patterns": ["*.rs", 42, true, "*.toml"]});
        let common = parse_input(&input, DEFAULT_MAX_RESULTS).unwrap();
        assert_eq!(
            common.include_patterns,
            vec!["*.rs".to_string(), "*.toml".to_string()]
        );
    }

    #[test]
    fn compile_pattern_invalid_errors() {
        assert!(compile_pattern("(unclosed", false).is_err());
        assert!(compile_pattern("(unclosed", true).is_err());
    }

    #[test]
    fn scan_file_finds_matches_in_line_order() {
        let regex = regex::Regex::new("foo").unwrap();
        let content = "foo bar\nnope\nfoo again\nbaz";
        let base = Path::new("/repo");
        let file = Path::new("/repo/a.rs");
        let results = scan_file(&regex, content, file, base, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].file, "a.rs");
        assert_eq!(results[0].line, 1);
        assert_eq!(results[0].content, "foo bar");
        assert_eq!(results[1].line, 3);
        assert_eq!(results[1].content, "foo again");
    }

    #[test]
    fn scan_file_caps_at_limit() {
        let regex = regex::Regex::new("x").unwrap();
        let content = "x\nx\nx\nx\nx";
        let base = Path::new("/repo");
        let file = Path::new("/repo/a.rs");
        let results = scan_file(&regex, content, file, base, 2);
        assert_eq!(results.len(), 2, "per-file cap enforced");
    }

    #[test]
    fn scan_file_zero_limit_returns_empty() {
        let regex = regex::Regex::new("x").unwrap();
        let content = "x\nx\nx";
        let base = Path::new("/repo");
        let file = Path::new("/repo/a.rs");
        let results = scan_file(&regex, content, file, base, 0);
        assert!(results.is_empty(), "limit=0 must yield no results");
    }

    #[test]
    fn render_emits_json_array_with_correct_keys() {
        let matches = vec![
            Match {
                file: "a.rs".to_string(),
                line: 2,
                content: "needle here".to_string(),
            },
            Match {
                file: "b.rs".to_string(),
                line: 7,
                content: "another needle".to_string(),
            },
        ];
        let output = render(&matches, Path::new("/tmp")).unwrap();
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json array");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["file"], "a.rs");
        assert_eq!(parsed[0]["line"], 2);
        assert_eq!(parsed[0]["content"], "needle here");
        // Key order is the contract; assert the serialized shape too.
        assert!(
            output.text_content().contains("\"file\""),
            "file key present"
        );
    }

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut context = ToolContext::default();
        context.cwd = cwd.to_string();
        context
    }

    #[tokio::test]
    async fn call_returns_matching_lines_as_json() {
        let tool = GrepTool::new(FakeSearchSource::with(&[
            ("/repo/a.rs", text("needle one\nnothing\nneedle two\n")),
            ("/repo/b.txt", text("nothing at all\n")),
        ]));
        let output = tool
            .call(
                json!({"pattern": "needle", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed.len(), 2, "{}", output.text_content());
        assert_eq!(parsed[0]["file"], "a.rs");
        assert_eq!(parsed[0]["line"], 1);
        assert_eq!(parsed[0]["content"], "needle one");
        assert_eq!(parsed[1]["line"], 3);
    }

    #[tokio::test]
    async fn call_case_insensitive_flag_reaches_the_engine() {
        let tool = GrepTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("Needle\n"))]));
        let output = tool
            .call(
                json!({"pattern": "needle", "path": "/repo", "case_insensitive": true}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed.len(), 1, "{}", output.text_content());
        assert_eq!(parsed[0]["content"], "Needle");
    }

    #[tokio::test]
    async fn call_include_and_exclude_filter_the_walk() {
        let tool = GrepTool::new(FakeSearchSource::with(&[
            ("/repo/keep.rs", text("needle\n")),
            ("/repo/skip.lock", text("needle\n")),
            ("/repo/skip.md", text("needle\n")),
        ]));
        let output = tool
            .call(
                json!({
                    "pattern": "needle",
                    "path": "/repo",
                    "include_patterns": ["*.rs"],
                }),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed.len(), 1, "{}", output.text_content());
        assert_eq!(parsed[0]["file"], "keep.rs");
    }

    #[tokio::test]
    async fn call_tags_the_json_array_with_the_json_hint() {
        let tool = GrepTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("needle\n"))]));
        let output = tool
            .call(
                json!({"pattern": "needle", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        assert_eq!(
            output.display_hint,
            Some(crate::tool::DisplayHint::Json),
            "the body is a JSON array and must say so"
        );
    }

    #[test]
    fn the_json_hint_sits_exactly_on_the_spill_boundary() {
        let dir = tempfile::TempDir::new().unwrap();
        let matches_of = |content_len: usize| {
            vec![Match {
                file: "f.rs".to_string(),
                line: 1,
                content: "x".repeat(content_len),
            }]
        };
        let probe = render(&matches_of(100), dir.path()).unwrap();
        let overhead = probe.text_content().len() - 100;

        let at_edge = render(&matches_of(MAX_INLINE_OUTPUT_BYTES - overhead), dir.path()).unwrap();
        assert_eq!(
            at_edge.display_hint,
            Some(crate::tool::DisplayHint::Json),
            "a body exactly at the inline threshold is the JSON array itself and carries the hint"
        );

        let one_over = render(
            &matches_of(MAX_INLINE_OUTPUT_BYTES - overhead + 1),
            dir.path(),
        )
        .unwrap();
        assert_eq!(
            one_over.display_hint, None,
            "one byte past the threshold the body is spill pointer prose — the hint must follow the spill decision, not a second predicate"
        );
        assert!(one_over.text_content().contains("result too large"));
    }

    #[tokio::test]
    async fn call_no_matches_is_the_shared_message() {
        let tool = GrepTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("nothing\n"))]));
        let output = tool
            .call(
                json!({"pattern": "needle", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        assert_eq!(
            output.text_content(),
            "No matches found for pattern: needle"
        );
    }

    #[tokio::test]
    async fn call_invalid_regex_is_invalid_input() {
        let tool = GrepTool::new(FakeSearchSource::with(&[]));
        let error = tool
            .call(
                json!({"pattern": "(unclosed", "path": "/repo"}),
                &ctx_in("/repo"),
            )
            .await
            .expect_err("bad regex");
        assert!(matches!(error, ToolError::InvalidInput(_)), "{error:?}");
    }

    #[tokio::test]
    async fn call_url_root_rejected_before_any_walk() {
        let shared = std::sync::Arc::new(FakeSearchSource::with(&[("/repo/a.rs", text("x\n"))]));
        let tool = GrepTool::new(std::sync::Arc::clone(&shared));
        let error = tool
            .call(
                json!({"pattern": "x", "path": "https://example.com/repo"}),
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
    async fn call_max_results_caps_the_total() {
        let tool = GrepTool::new(FakeSearchSource::with(&[
            ("/repo/a.rs", text("x\nx\nx\n")),
            ("/repo/b.rs", text("x\nx\n")),
        ]));
        let output = tool
            .call(
                json!({"pattern": "x", "path": "/repo", "max_results": 3}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed.len(), 3, "{}", output.text_content());
    }

    #[tokio::test]
    async fn call_max_matches_caps_per_file() {
        let tool = GrepTool::new(FakeSearchSource::with(&[(
            "/repo/a.rs",
            text("x\nx\nx\nx\n"),
        )]));
        let output = tool
            .call(
                json!({"pattern": "x", "path": "/repo", "max_matches": 2}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let parsed: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json");
        assert_eq!(parsed.len(), 2, "{}", output.text_content());
    }

    #[tokio::test]
    async fn call_max_results_above_cap_is_lowered() {
        // Eleven files at the per-file default cap: 1,100 matches in
        // total, so the global ceiling — not the per-file one — is
        // the binding constraint this pin discriminates.
        let owned: Vec<(String, FakeFile)> = (0..11)
            .map(|index| (format!("/repo/f{index}.rs"), text(&"x\n".repeat(100))))
            .collect();
        let files: Vec<(&str, FakeFile)> = owned
            .iter()
            .map(|(path, file)| (path.as_str(), file.clone()))
            .collect();
        let tool = GrepTool::new(FakeSearchSource::with(&files));
        let spill_dir = tempfile::TempDir::new().unwrap();
        let context = ToolContext {
            cwd: "/repo".to_string(),
            temp_dir: spill_dir.path().to_string_lossy().into_owned(),
            ..ToolContext::default()
        };
        let output = tool
            .call(
                json!({"pattern": "x", "path": "/repo", "max_results": 99_999}),
                &context,
            )
            .await
            .expect("call");
        // A capped result this size spills; the count lives in the
        // spilled JSON body.
        let text = output.text_content();
        let start = text
            .find("written to: ")
            .map(|at| at + "written to: ".len())
            .expect("spill pointer");
        let end = text[start..]
            .find('\n')
            .map(|offset| start + offset)
            .expect("line end");
        let spilled = std::fs::read_to_string(text[start..end].trim()).expect("read spill");
        let parsed: Vec<Value> = serde_json::from_str(&spilled).expect("spilled json");
        assert_eq!(
            parsed.len(),
            MAX_RESULTS_CAP,
            "an over-cap request stops at the hard ceiling"
        );
    }

    #[tokio::test]
    async fn a_grep_output_is_stamped_requery() {
        let tool = GrepTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("Needle\n"))]));
        let output = tool
            .call(
                json!({"pattern": "needle", "path": "/repo", "case_insensitive": true}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        assert_eq!(
            output.retention,
            Some(crate::tool::Retention::Requery),
            "search hits are re-derivable — the compaction transcript withholds them"
        );
    }
}
