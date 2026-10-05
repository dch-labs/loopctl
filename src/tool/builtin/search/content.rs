//! Shared machinery for the regex content-search tools.
//!
//! Both content-search tools walk a source under ignore rules, read each
//! non-binary file, and collect regex matches as [`Match`] records. What
//! differs is how each file is scanned (one record per matched line, or a
//! context snippet attached) and how the collected matches render. This
//! module owns everything else: the walk loop, the per-file and global
//! caps, the binary and size guards, the regex cache lookup, and the
//! no-match success message.

use std::path::Path;
use std::path::PathBuf;

use regex::Regex;
use serde_json::Value;

use crate::tool::ToolError;
use crate::tool::ToolOutput;
use crate::tool::builtin::search::SearchSource;
use crate::tool::builtin::search::regex_cache::get_or_compile;
use crate::tool::builtin::search::walk;

/// One matched line, as collected by the search runner.
///
/// The common shape both content-search tools agree on: a relative file
/// path, a 1-indexed line number, and the line's text — the unit the
/// shared walk loop collects, which each tool then renders its own way.
/// The JSON renderer serializes these verbatim as
/// `{"file", "line", "content"}`; the grouped renderer collapses line
/// numbers into ranges.
#[derive(Debug, Clone)]
pub struct Match {
    /// The file path relative to the search base.
    ///
    /// Computed at scan time by stripping the walked file's path
    /// against the resolved search base; falls back to the display
    /// form (or `"unknown"` for non-UTF-8 paths) when the strip
    /// fails. Renderers echo this verbatim.
    pub file: String,

    /// 1-indexed line number of the match within the file.
    ///
    /// Source files are 1-indexed in every editor and in the model's
    /// expectation, so this is stored 1-indexed; the grouped renderer
    /// collapses consecutive values into ranges.
    pub line: usize,

    /// The matched line's text, or a rendered context snippet.
    ///
    /// The line renderer emits it as-is; the context renderer stores a
    /// rendered snippet here at scan time.
    pub content: String,
}

/// Bundles the inputs the shared walk loop needs.
///
/// Constructed by each tool's async body from its parsed input and read
/// by [`run`]. Tool-specific knobs are deliberately not here — they are
/// captured by the per-file scan closure each tool passes, so a future
/// search tool with different knobs reuses this without inheriting
/// fields it does not use.
pub struct SearchJob {
    /// The compiled regex to match against each line.
    ///
    /// Produced by [`compile_pattern`] before the job is built, so the
    /// blocking walk does no regex compilation; cheap to move
    /// (`Regex` is `Arc`-backed internally).
    pub regex: Regex,

    /// Filename-level glob filters forwarded to the walker.
    ///
    /// Empty means no include filter — every file the walker yields
    /// is read.
    pub include: Vec<String>,

    /// Filename-level glob exclusions forwarded to the walker.
    ///
    /// A file whose basename matches any entry here is skipped
    /// before it is read.
    pub exclude: Vec<String>,

    /// Total match cap across all files.
    ///
    /// Enforced as a running-total ceiling: once this many matches
    /// are collected overall, the walk stops. The per-file scan
    /// closure receives the headroom under this cap as its `limit`
    /// argument, so a single huge file cannot allocate unbounded
    /// matches before the cap discards them.
    pub max_results: usize,

    /// Optional per-file match cap.
    ///
    /// `None` means the only ceiling is
    /// [`max_results`](Self::max_results); `Some(n)` tightens each
    /// file's scan to the smaller of `n` and the remaining global
    /// headroom, stopping one huge file from saturating the result
    /// before the walker moves on.
    pub per_file_cap: Option<usize>,

    /// The directory to search, already resolved against the tool cwd.
    ///
    /// Absolute by the time it reaches the job; every walked file's
    /// path is stripped against this base to produce the relative
    /// `file` field of each [`Match`].
    pub base: PathBuf,

    /// The pattern string, exactly as the caller supplied it.
    ///
    /// Carried alongside the compiled regex only so the no-match
    /// success message can echo the original text back to the model.
    pub pattern: String,
}

/// The fields shared by every content-search tool's input.
///
/// `pattern`, `path`, `case_insensitive`, `max_results`,
/// `include_patterns`, and `exclude_patterns` parse identically in
/// every content-search tool; this struct is that shared parse, with
/// the resolved absolute base swapped in for the raw `path` string.
#[derive(Debug)]
pub struct CommonInput {
    /// The regex pattern, required and verbatim.
    ///
    /// Compiled through the shared cache by [`compile_pattern`].
    pub pattern: String,

    /// The search root as the caller wrote it, before resolution.
    ///
    /// Default `"."`; the tool resolves it against its cwd before
    /// building the job.
    pub base_path: String,

    /// Whether matching folds case, applied before caching.
    ///
    /// Defaults to `false`.
    pub case_insensitive: bool,

    /// Total match cap, before any tool-specific clamping.
    ///
    /// Defaults to the `default_max_results` argument passed to
    /// [`parse_input`]; the caller may further clamp it.
    pub max_results: usize,

    /// Filename-level glob filters forwarded to the walker.
    ///
    /// Empty means no include filter.
    pub include_patterns: Vec<String>,

    /// Filename-level glob exclusions forwarded to the walker.
    ///
    /// A file matching any entry here is skipped before it is read.
    pub exclude_patterns: Vec<String>,
}

/// Parse the six input fields shared by all content-search tools.
///
/// Extracts `pattern` (required), `path` (default `"."`),
/// `case_insensitive` (default `false`), `max_results` (default
/// `default_max_results`), `include_patterns` (default empty), and
/// `exclude_patterns` (default empty), rejecting malformed values
/// loudly. An explicit zero for `max_results` clamps to `1` so the
/// caller always gets at least one result.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when `pattern` is missing, when
/// a numeric field is present but not a non-negative integer, or when
/// an array field is present but not an array of strings.
pub fn parse_input(input: &Value, default_max_results: usize) -> Result<CommonInput, ToolError> {
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
    let base_path = match input.get("path") {
        None => ".".to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(_) => {
            return Err(ToolError::InvalidInput(
                "'path' must be a string".to_string(),
            ));
        }
    };
    let case_insensitive = match input.get("case_insensitive") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(ToolError::InvalidInput(
                "'case_insensitive' must be a boolean".to_string(),
            ));
        }
    };
    let max_results = get_usize(input, "max_results")?
        .unwrap_or(default_max_results)
        .max(1);
    let include_patterns = get_string_list(input, "include_patterns")?;
    let exclude_patterns = get_string_list(input, "exclude_patterns")?;
    Ok(CommonInput {
        pattern,
        base_path,
        case_insensitive,
        max_results,
        include_patterns,
        exclude_patterns,
    })
}

/// Read a non-negative integer field, present or absent.
///
/// The shared numeric-input rule: absent is `None` (the caller's
/// default applies), a non-negative integer is `Some`, anything else
/// is a correction prompt naming the field.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not a non-negative integer.
pub fn get_usize(input: &Value, field: &str) -> Result<Option<usize>, ToolError> {
    match input.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|raw| usize::try_from(raw).ok())
            .map(Some)
            .ok_or_else(|| {
                ToolError::InvalidInput(format!("'{field}' must be a non-negative integer"))
            }),
    }
}

/// Read an array-of-strings field, present or absent.
///
/// The shared list-input rule: absent is empty, an array yields only
/// its string elements (non-string elements are dropped, the
/// walker-tolerant shape), and anything
/// other than an array is a correction prompt.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] when the field is present but
/// not an array.
pub fn get_string_list(input: &Value, field: &str) -> Result<Vec<String>, ToolError> {
    match input.get(field) {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()),
        Some(_) => Err(ToolError::InvalidInput(format!(
            "'{field}' must be an array of strings"
        ))),
    }
}

/// Compute the display path of `file_path` relative to `base_path`.
///
/// Returns the stripped relative path when possible, the full path
/// string when `file_path` is not under `base_path`, or `"unknown"`
/// when the path cannot be rendered as UTF-8 — so the relative-path
/// computation is identical across both content-search tools.
#[must_use]
pub fn relative_file(file_path: &Path, base_path: &Path) -> String {
    file_path
        .strip_prefix(base_path)
        .ok()
        .and_then(|path| path.to_str())
        .unwrap_or_else(|| file_path.to_str().unwrap_or("unknown"))
        .to_string()
}

/// Compile `pattern` through the shared regex cache, folding case or not.
///
/// Routes the pattern through [`get_or_compile`] so a pattern compiled
/// by one tool is a cache hit in the other. When `case_insensitive` is
/// `true`, the pattern is wrapped with the `(?i)` flag (idempotently)
/// before caching.
///
/// # Errors
///
/// Returns [`ToolError::InvalidInput`] wrapping the regex engine's
/// error when the pattern fails to compile — an invalid regex is a
/// caller bug worth fixing, not a tool failure.
pub fn compile_pattern(pattern: &str, case_insensitive: bool) -> Result<Regex, ToolError> {
    get_or_compile(pattern, case_insensitive)
        .map_err(|error| ToolError::InvalidInput(format!("Invalid regex pattern: {error}")))
}

/// Walk `job.base` on `source` and collect up to `job.max_results`
/// [`Match`]es total.
///
/// The shared heart of both content-search tools: the source walk,
/// the binary-file skip, the file-size guard against OOM on huge
/// files, a size-capped read followed by UTF-8 validation (files
/// failing either are silently skipped), and both caps — the global
/// ceiling (the walk stops once reached) and the per-file limit (the
/// smaller of the per-file cap and the remaining headroom). The
/// `scan_file` closure is the per-tool specialization; its contract is
/// to return at most `limit` matches with no cap math of its own.
///
/// Unreadable files are silently skipped — a search over a tree with
/// permission-denied files still returns the matches it could collect,
/// matching the behavior of grep and ripgrep.
pub fn run<S, F>(source: &S, job: &SearchJob, scan_file: F) -> Vec<Match>
where
    S: SearchSource + ?Sized,
    F: Fn(&Regex, &str, &Path, &Path, usize) -> Vec<Match>,
{
    let mut matches = Vec::new();
    for entry in source.walk_files(&job.base, &job.include, &job.exclude) {
        if matches.len() >= job.max_results {
            break;
        }
        let path = entry.path;
        if source.file_too_large(&path) || source.likely_binary(&path) {
            continue;
        }
        let cap = usize::try_from(walk::MAX_FILE_BYTES.saturating_add(1)).unwrap_or(usize::MAX);
        let Ok(buffer) = source.read_capped(&path, cap as u64) else {
            continue;
        };
        if buffer.len() > usize::try_from(walk::MAX_FILE_BYTES).unwrap_or(usize::MAX) {
            continue;
        }
        let Ok(content) = String::from_utf8(buffer) else {
            continue;
        };
        let remaining = job.max_results.saturating_sub(matches.len());
        let limit = job.per_file_cap.map_or(remaining, |cap| cap.min(remaining));
        let file_matches = scan_file(&job.regex, &content, &path, &job.base, limit);
        matches.extend(file_matches);
    }
    matches
}

/// The "no matches" success message both tools emit for an empty set.
///
/// An empty search is not an error — the model uses "matched nothing"
/// as a signal to broaden or refine its pattern, distinct from a tool
/// failure. Returning it as a successful [`ToolOutput::text`] keeps
/// `is_error == false`, so retry logic does not fire on a
/// legitimately-empty result; the body names the pattern verbatim.
#[must_use]
pub fn no_matches_message(pattern: &str) -> ToolOutput {
    ToolOutput::text(format!("No matches found for pattern: {pattern}"))
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
    use crate::tool::builtin::search::FsSearchSource;
    use crate::tool::builtin::search::regex_cache::TEST_LOCK;
    use crate::tool::builtin::search::regex_cache::clear_cache;
    use crate::tool::builtin::search::walk::MAX_FILE_BYTES;

    // The single shared cache guard lives in regex_cache; every
    // cache-state-asserting test in the crate holds it.

    fn job(base: &Path, max_results: usize, per_file_cap: Option<usize>) -> SearchJob {
        SearchJob {
            regex: Regex::new("match").unwrap(),
            include: vec![],
            exclude: vec![],
            max_results,
            per_file_cap,
            base: base.to_path_buf(),
            pattern: "match".to_string(),
        }
    }

    /// A scan closure that emits one `Match` per line containing the
    /// regex's pattern, up to `limit` — what a real tool's `scan_file`
    /// does.
    fn scan(regex: &Regex, content: &str, file: &Path, base: &Path, limit: usize) -> Vec<Match> {
        let rel = file.strip_prefix(base).unwrap_or(file);
        let mut out = Vec::new();
        for (index, line) in content.lines().enumerate() {
            if out.len() >= limit {
                break;
            }
            if regex.is_match(line) {
                out.push(Match {
                    file: rel.to_string_lossy().into_owned(),
                    line: index.saturating_add(1),
                    content: line.to_string(),
                });
            }
        }
        out
    }

    #[test]
    fn compile_pattern_valid() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let regex = compile_pattern("foo", false).unwrap();
        assert!(regex.is_match("foobar"));
    }

    #[test]
    fn compile_pattern_case_insensitive() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let regex = compile_pattern("foo", true).unwrap();
        assert!(regex.is_match("FOO"));
        assert!(regex.is_match("foo"));
    }

    #[test]
    fn compile_pattern_invalid_returns_invalid_input() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let error = compile_pattern("(unclosed", false).unwrap_err();
        assert!(
            matches!(error, ToolError::InvalidInput(ref message) if message.contains("Invalid regex pattern")),
            "{error:?}"
        );
    }

    #[test]
    fn no_matches_message_format_and_not_error() {
        let out = no_matches_message("my_pattern");
        assert!(!out.is_error, "no-match is a success, not an error");
        assert_eq!(
            out.text_content(),
            "No matches found for pattern: my_pattern"
        );
    }

    #[test]
    fn no_matches_message_preserves_special_chars() {
        let out = no_matches_message("(?P<name>foo)");
        assert!(out.text_content().contains("(?P<name>foo)"));
    }

    #[test]
    fn run_global_cap_stops_walk() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "match\nmatch\nmatch\n").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "match\nmatch\nmatch\n").unwrap();
        std::fs::write(tmp.path().join("c.txt"), "match\nmatch\nmatch\n").unwrap();

        let j = job(tmp.path(), 4, None);
        let matches = run(&FsSearchSource, &j, scan);
        assert_eq!(matches.len(), 4, "global cap stops at 4");
    }

    #[test]
    fn run_per_file_caps_each_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "match\nmatch\nmatch\n").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "match\nmatch\n").unwrap();

        let j = job(tmp.path(), 100, Some(2));
        let matches = run(&FsSearchSource, &j, scan);
        let per_file: std::collections::HashMap<&str, usize> =
            matches
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, m| {
                    *acc.entry(m.file.as_str()).or_insert(0) += 1;
                    acc
                });
        assert_eq!(
            per_file.get("a.txt"),
            Some(&2),
            "a capped at 2: {matches:?}"
        );
        assert_eq!(per_file.get("b.txt"), Some(&2), "b has only 2: {matches:?}");
    }

    #[test]
    fn run_include_and_exclude_filter_the_walk() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("keep.rs"), "match\n").unwrap();
        std::fs::write(tmp.path().join("drop.lock"), "match\n").unwrap();
        std::fs::write(tmp.path().join("drop.md"), "match\n").unwrap();

        let mut j = job(tmp.path(), 100, None);
        j.include = vec!["*.rs".to_string()];
        j.exclude = vec!["*.md".to_string()];
        let matches = run(&FsSearchSource, &j, scan);
        assert_eq!(matches.len(), 1, "{matches:?}");
        assert_eq!(matches[0].file, "keep.rs");
    }

    #[test]
    fn run_skips_binary_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut bytes = vec![0u8; 32];
        bytes.extend_from_slice(b"match\n");
        std::fs::write(tmp.path().join("data.png"), &bytes).unwrap();
        std::fs::write(tmp.path().join("a.txt"), "match\n").unwrap();

        let j = job(tmp.path(), 100, None);
        let matches = run(&FsSearchSource, &j, scan);
        assert!(
            matches.iter().all(|m| m.file != "data.png"),
            "binary skipped"
        );
        assert!(matches.iter().any(|m| m.file == "a.txt"));
    }

    #[test]
    fn run_skips_oversized_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let big = "match ".repeat(usize::try_from(MAX_FILE_BYTES + 1).unwrap());
        std::fs::write(tmp.path().join("big.txt"), &big).unwrap();
        std::fs::write(tmp.path().join("small.txt"), "match\n").unwrap();

        let j = job(tmp.path(), 100, None);
        let matches = run(&FsSearchSource, &j, scan);
        assert!(
            matches.iter().all(|m| m.file != "big.txt"),
            "oversized file skipped"
        );
        assert!(matches.iter().any(|m| m.file == "small.txt"));
    }

    #[test]
    fn run_silently_skips_unreadable_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("not_a_file")).unwrap();
        std::fs::write(tmp.path().join("a.txt"), "match\n").unwrap();

        let j = job(tmp.path(), 100, None);
        let matches = run(&FsSearchSource, &j, scan);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, "a.txt");
    }

    #[test]
    fn run_empty_dir_returns_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let j = job(tmp.path(), 100, None);
        assert!(run(&FsSearchSource, &j, scan).is_empty());
    }

    #[test]
    fn run_match_file_and_line_are_relative_and_1_indexed() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "alpha\nBRAVO\nalpha\n").unwrap();

        let mut j = job(tmp.path(), 100, None);
        j.regex = Regex::new("BRAVO").unwrap();
        let matches = run(&FsSearchSource, &j, scan);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, "a.txt", "relative to base");
        assert_eq!(matches[0].line, 2, "1-indexed line number");
        assert_eq!(matches[0].content, "BRAVO");
    }

    #[test]
    fn relative_file_strips_base_prefix() {
        assert_eq!(
            relative_file(Path::new("/repo/src/main.rs"), Path::new("/repo")),
            "src/main.rs"
        );
    }

    #[test]
    fn relative_file_nested_subdir() {
        assert_eq!(
            relative_file(Path::new("/repo/a/b/c.rs"), Path::new("/repo")),
            "a/b/c.rs"
        );
    }

    #[test]
    fn relative_file_base_equals_file() {
        assert_eq!(relative_file(Path::new("/repo"), Path::new("/repo")), "");
    }

    #[test]
    fn relative_file_not_under_base_falls_back_to_full() {
        assert_eq!(
            relative_file(Path::new("/other/x.rs"), Path::new("/repo")),
            "/other/x.rs"
        );
    }
}
