//! The `CodeSearch` tool — token-efficient regex code search.
//!
//! Same regex and walker engine as the grep tool, but returns grouped
//! file headers with `line: text` rows by default, and `±context_lines`
//! window snippets only when the caller opts in via `context_lines > 0`.
//! A global `max_results` cap (default 50, ceiling 200) bounds the total
//! output regardless of how many matches one file contains.

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use regex::Regex;
use serde_json::Value;

use crate::tool::Retention;
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
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// Default total match cap across all files when the caller omits `max_results`.
///
/// Keeps a broad search from flooding the model's context in one call;
/// a caller wanting more asks explicitly and is still bounded by
/// [`RESULTS_CAP`].
const DEFAULT_MAX_RESULTS: usize = 50;

/// Hard ceiling `max_results` is clamped to, whatever the caller asks.
///
/// Bounds the worst-case context spend per call — even an explicit
/// huge request stops here, the excess matches staying unlisted
/// rather than silently admitted.
const RESULTS_CAP: usize = 200;

/// Hard ceiling `context_lines` is clamped to, whatever the caller asks.
///
/// Window snippets stay narrow so each match's output remains
/// token-cheap; a wider window is what the Read tool is for.
const MAX_CONTEXT_LINES: usize = 5;

/// Token-efficient regex code search over any [`SearchSource`].
///
/// Same engine as the grep tool, but the smallest useful answer: per
/// file, every match as `  {line}: {text}` by default, and
/// `±context_lines` window snippets (with `>` on the matched line)
/// when `context_lines > 0`. Both [`is_read_only`](Tool::is_read_only)
/// and [`is_concurrency_safe`](Tool::is_concurrency_safe) are true.
pub struct CodeSearchTool<S: SearchSource + 'static> {
    /// The source the tool searches.
    ///
    /// Shared by `Arc` so the blocking walk thread can hold a clone
    /// for the traversal's lifetime.
    source: Arc<S>,
}

impl<S: SearchSource + 'static> CodeSearchTool<S> {
    /// Build a code-search tool over `source`.
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

impl<S: SearchSource + 'static> Tool for CodeSearchTool<S> {
    fn name(&self) -> &'static str {
        "CodeSearch"
    }

    fn description(&self) -> &'static str {
        "Search code with succinct, token-efficient results: every match \
         returns its content as `line: text` under a file header. Use \
         context_lines > 0 to widen each match with surrounding lines."
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
                        "description": "File patterns restricting the search to matching files"
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns removing files from the search"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match (0 = succinct lines only; max 5)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum total matches (default 50, hard-capped at 200)"
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
        Box::pin(async move { code_search_inner(source, input, cwd, temp_dir).await })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

/// Body of the code-search tool's [`Tool::call`].
///
/// Orchestrates parse → compile → walk → render. An empty match set
/// is a success message; bad arguments and invalid patterns become
/// [`ToolError::InvalidInput`]. Everything the blocking walk touches
/// is owned, so the task needs no borrow of the tool.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] for malformed arguments or an
/// unparseable pattern, and [`ToolError::Execution`] when the blocking
/// walk task joins unsuccessfully.
async fn code_search_inner<S: SearchSource + 'static>(
    source: Arc<S>,
    input: Value,
    cwd: PathBuf,
    temp_dir: PathBuf,
) -> Result<ToolOutput, ToolError> {
    let parsed = parse_input(&input, DEFAULT_MAX_RESULTS)?;
    let context_lines = get_usize(&input, "context_lines")?
        .unwrap_or(0)
        .min(MAX_CONTEXT_LINES);

    resolve::reject_url("CodeSearch", &parsed.base_path)?;

    let regex = compile_pattern(&parsed.pattern, parsed.case_insensitive)?;
    let base = resolve::resolve_root(&parsed.base_path, &cwd);
    let job = SearchJob {
        regex,
        include: parsed.include_patterns,
        exclude: parsed.exclude_patterns,
        max_results: parsed.max_results.min(RESULTS_CAP),
        per_file_cap: None,
        base,
        pattern: parsed.pattern.clone(),
    };
    let matches = tokio::task::spawn_blocking(move || {
        run(&*source, &job, |regex, content, file_path, base, limit| {
            scan_file(regex, content, file_path, base, context_lines, limit)
        })
    })
    .await
    .map_err(|error| ToolError::Execution(format!("CodeSearch walk task failed: {error}")))?;

    if matches.is_empty() {
        return Ok(no_matches_message(&parsed.pattern));
    }

    Ok(render(&matches, &parsed.pattern, context_lines, &temp_dir)
        .with_retention(Retention::Requery))
}

/// Scan one file's lines for regex matches; return up to `limit`.
///
/// Computes a path relative to the search base, enumerates lines with
/// 1-indexed numbers, and pushes one [`Match`] per matching line,
/// stopping once `limit` are collected. When `context_lines > 0`, each
/// match's content is a rendered snippet of the window with `>`
/// marking the matched line; otherwise it is the matched line's text.
fn scan_file(
    regex: &Regex,
    content: &str,
    file_path: &Path,
    base_path: &Path,
    context_lines: usize,
    limit: usize,
) -> Vec<Match> {
    let lines: Vec<&str> = content.lines().collect();
    let rel_path = relative_file(file_path, base_path);
    let mut results = Vec::new();
    for (line_num, line) in lines.iter().enumerate() {
        if results.len() >= limit {
            break;
        }
        if !regex.is_match(line) {
            continue;
        }
        let entry = if context_lines > 0 {
            render_snippet(&lines, line_num, context_lines)
        } else {
            line.to_string()
        };
        results.push(Match {
            file: rel_path.clone(),
            line: line_num.saturating_add(1),
            content: entry,
        });
    }
    results
}

/// Render the `±context_lines` window around `line_num` as a snippet.
///
/// The matched line (1-indexed) is marked `>`; its neighbors are
/// marked ` `. Each line is formatted `{marker}{lineno}: {text}` and
/// the window is joined with newlines.
fn render_snippet(lines: &[&str], line_num: usize, context_lines: usize) -> String {
    let start = line_num.saturating_sub(context_lines);
    let end = line_num
        .saturating_add(context_lines)
        .saturating_add(1)
        .min(lines.len());
    let matched_1indexed = line_num.saturating_add(1);
    let mut buffer = Vec::new();
    for (offset, line) in lines.get(start..end).unwrap_or(&[]).iter().enumerate() {
        let actual = start.saturating_add(offset).saturating_add(1);
        let marker = if actual == matched_1indexed { ">" } else { " " };
        buffer.push(format!("{marker}{actual}: {line}"));
    }
    buffer.join("\n")
}

/// Render the collected matches in one of two modes keyed on `context_lines`.
///
/// `context_lines == 0` → grouped: a `Found N match(es)` header, a
/// blank line, then per file the path and one `  {line}: {text}` row
/// per match. `context_lines > 0` → the same header followed by
/// `file:line` per match and its window snippet indented. Output
/// spills via the shared helper when it exceeds the inline limit.
fn render(matches: &[Match], pattern: &str, context_lines: usize, temp_dir: &Path) -> ToolOutput {
    let mut out = Vec::with_capacity(matches.len().saturating_add(2));
    let match_word = if matches.len() == 1 {
        "match"
    } else {
        "matches"
    };
    out.push(format!(
        "Found {} {match_word} for \"{pattern}\":",
        matches.len()
    ));
    out.push(String::new());

    if context_lines == 0 {
        let mut sorted: Vec<&Match> = matches.iter().collect();
        sorted.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.line.cmp(&b.line)));
        let mut current_file: Option<&str> = None;
        for entry in sorted {
            if current_file != Some(entry.file.as_str()) {
                out.push(entry.file.clone());
                current_file = Some(entry.file.as_str());
            }
            out.push(format!("  {}: {}", entry.line, entry.content));
        }
    } else {
        let mut sorted: Vec<&Match> = matches.iter().collect();
        sorted.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.line.cmp(&b.line)));
        for entry in sorted {
            out.push(format!("{}:{}", entry.file, entry.line));
            if !entry.content.is_empty() {
                out.push(format!("  {}", entry.content));
            }
        }
    }
    let text = out.join("\n");
    truncate_or_spill(text, "code_search", temp_dir, MAX_INLINE_OUTPUT_BYTES).0
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::tool::builtin::search::content::parse_input;
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
        let context_lines = get_usize(&json!({}), "context_lines")
            .unwrap()
            .unwrap_or(0)
            .min(MAX_CONTEXT_LINES);
        assert_eq!(context_lines, 0);
    }

    #[test]
    fn parse_clamps_max_results() {
        let common = parse_input(
            &json!({"pattern": "x", "max_results": 99_999}),
            DEFAULT_MAX_RESULTS,
        )
        .unwrap();
        assert_eq!(common.max_results.min(RESULTS_CAP), RESULTS_CAP);
    }

    #[test]
    fn parse_clamps_context_lines() {
        let clamped = get_usize(
            &json!({"pattern": "x", "context_lines": 99}),
            "context_lines",
        )
        .unwrap()
        .unwrap_or(0)
        .min(MAX_CONTEXT_LINES);
        assert_eq!(clamped, MAX_CONTEXT_LINES);
    }

    #[test]
    fn parse_rejects_negative_max_results() {
        assert!(
            parse_input(
                &json!({"pattern": "x", "max_results": -1}),
                DEFAULT_MAX_RESULTS
            )
            .is_err()
        );
    }

    #[test]
    fn parse_missing_pattern_errors() {
        assert!(parse_input(&json!({}), DEFAULT_MAX_RESULTS).is_err());
    }

    #[test]
    fn scan_file_succinct_keeps_matched_text() {
        let regex = regex::Regex::new("foo").unwrap();
        let content = "foo\nnope\nfoo";
        let results = scan_file(
            &regex,
            content,
            Path::new("/repo/a.rs"),
            Path::new("/repo"),
            0,
            100,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].line, 1);
        assert_eq!(results[1].line, 3);
        assert_eq!(results[0].content, "foo");
    }

    #[test]
    fn scan_file_context_renders_snippet() {
        let regex = regex::Regex::new("mid").unwrap();
        let content = "line1\nmid\nline3";
        let results = scan_file(
            &regex,
            content,
            Path::new("/repo/a.rs"),
            Path::new("/repo"),
            1,
            100,
        );
        assert_eq!(results.len(), 1);
        let snippet = &results[0].content;
        assert!(snippet.contains(">2: mid"), "{snippet}");
        assert!(snippet.contains(" 1: line1"), "{snippet}");
        assert!(snippet.contains(" 3: line3"), "{snippet}");
    }

    #[test]
    fn scan_file_context_at_file_start() {
        let regex = regex::Regex::new("first").unwrap();
        let content = "first\nsecond\nthird";
        let results = scan_file(
            &regex,
            content,
            Path::new("/repo/a.rs"),
            Path::new("/repo"),
            3,
            100,
        );
        assert_eq!(results.len(), 1);
        let snippet = &results[0].content;
        assert!(snippet.contains(">1: first"), "{snippet}");
        assert!(snippet.contains(" 2: second"), "{snippet}");
    }

    #[test]
    fn scan_file_respects_limit() {
        let regex = regex::Regex::new("x").unwrap();
        let content = "x\nx\nx\nx\nx";
        let results = scan_file(
            &regex,
            content,
            Path::new("/repo/a.rs"),
            Path::new("/repo"),
            0,
            2,
        );
        assert_eq!(results.len(), 2, "per-file limit stops the scan early");
    }

    #[test]
    fn render_snippet_marks_matched_line_with_gt() {
        let lines = vec!["alpha", "mid", "gamma"];
        let snippet = render_snippet(&lines, 1, 1);
        assert!(snippet.contains(">2: mid"), "{snippet}");
        assert!(snippet.contains(" 1: alpha"), "{snippet}");
        assert!(snippet.contains(" 3: gamma"), "{snippet}");
    }

    #[test]
    fn render_snippet_at_file_start_clamps_window() {
        let lines = vec!["first", "second", "third"];
        let snippet = render_snippet(&lines, 0, 3);
        assert!(snippet.contains(">1: first"), "{snippet}");
        assert!(snippet.contains(" 2: second"), "{snippet}");
        assert!(snippet.contains(" 3: third"), "{snippet}");
        assert!(!snippet.contains(" 0:"), "{snippet}");
    }

    #[test]
    fn render_snippet_at_file_end_clamps_window() {
        let lines = vec!["a", "b", "last"];
        let snippet = render_snippet(&lines, 2, 5);
        assert!(snippet.contains(">3: last"), "{snippet}");
        assert!(snippet.contains(" 2: b"), "{snippet}");
        assert!(!snippet.contains(" 4:"), "no line beyond EOF: {snippet}");
    }

    fn ctx_in(cwd: &str) -> ToolContext {
        let mut context = ToolContext::default();
        context.cwd = cwd.to_string();
        context
    }

    #[tokio::test]
    async fn call_succinct_groups_by_file_with_line_rows() {
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[
            ("/repo/b.rs", text("hit one\nmiss\nhit two\n")),
            ("/repo/a.rs", text("hit three\n")),
        ]));
        let output = tool
            .call(json!({"pattern": "hit", "path": "/repo"}), &ctx_in("/repo"))
            .await
            .expect("call");
        let text = output.text_content();
        assert!(text.starts_with("Found 3 matches for \"hit\":"), "{text}");
        let a = text.find("a.rs\n").expect("a.rs header");
        let b = text.find("b.rs\n").expect("b.rs header");
        assert!(a < b, "files sorted: {text}");
        assert!(text.contains("  1: hit one\n"), "{text}");
        // The joined text has no trailing newline; the last row ends the output.
        assert!(text.ends_with("  3: hit two"), "{text}");
        assert!(text.contains("  1: hit three\n"), "{text}");
    }

    #[tokio::test]
    async fn call_context_lines_render_snippets_under_file_line() {
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[(
            "/repo/a.rs",
            text("alpha\nneedle here\ngamma\n"),
        )]));
        let output = tool
            .call(
                json!({
                    "pattern": "needle",
                    "path": "/repo",
                    "context_lines": 1
                }),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let text = output.text_content();
        assert!(text.contains("a.rs:2\n"), "{text}");
        // Only the first snippet line gets the two-space group indent;
        // later lines keep their own markers.
        assert!(text.contains("   1: alpha"), "{text}");
        assert!(text.contains("\n>2: needle here"), "{text}");
        assert!(text.contains("\n 3: gamma"), "{text}");
    }

    #[tokio::test]
    async fn call_renders_plain_text_without_a_json_hint() {
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[(
            "/repo/a.rs",
            text("hit one\nhit two\n"),
        )]));
        let output = tool
            .call(json!({"pattern": "hit", "path": "/repo"}), &ctx_in("/repo"))
            .await
            .expect("call");
        assert!(
            output.display_hint.is_none(),
            "the grouped text body is not JSON; a Json hint would send              presentation layers into a failed parse: {:?}",
            output.display_hint
        );
    }

    #[tokio::test]
    async fn call_no_matches_is_the_shared_message() {
        let tool =
            CodeSearchTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("nothing\n"))]));
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
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[]));
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
        let tool = CodeSearchTool::new(std::sync::Arc::clone(&shared));
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
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[(
            "/repo/a.rs",
            text("x\nx\nx\nx\n"),
        )]));
        let output = tool
            .call(
                json!({"pattern": "x", "path": "/repo", "max_results": 2}),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        let text = output.text_content();
        assert!(text.starts_with("Found 2 matches for \"x\":"), "{text}");
    }

    #[tokio::test]
    async fn call_case_insensitive_flag_reaches_the_engine() {
        let tool = CodeSearchTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("Needle\n"))]));
        let output = tool
            .call(
                json!({
                    "pattern": "needle",
                    "path": "/repo",
                    "case_insensitive": true
                }),
                &ctx_in("/repo"),
            )
            .await
            .expect("call");
        assert!(
            output.text_content().contains("  1: Needle"),
            "{}",
            output.text_content()
        );
    }

    #[tokio::test]
    async fn a_code_search_output_is_stamped_requery() {
        let tool =
            CodeSearchTool::new(FakeSearchSource::with(&[("/repo/a.rs", text("hit one\n"))]));
        let output = tool
            .call(json!({"pattern": "hit", "path": "/repo"}), &ctx_in("/repo"))
            .await
            .expect("call");
        assert_eq!(
            output.retention,
            Some(crate::tool::Retention::Requery),
            "code-search hits are re-derivable — the compaction transcript withholds them"
        );
    }
}
