//! Middleware that suggests alternatives when a tool is not found.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use crate::message::ToolContent;
use crate::tool::ToolRegistry;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Middleware that suggests alternatives when a tool is not found.
///
/// This middleware wraps the core dispatch. When the inner dispatch
/// produces a "tool not found" error, this middleware intercepts it,
/// computes a string-similarity score against all registered tools,
/// and appends a suggestion to the error message.
///
/// # Similarity Metric
///
/// Uses a simple normalized longest-common-subsequence ratio, which is
/// fast and effective for the common case of minor typos (e.g.
/// `"bash"` → `"basj"`).
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use loopctl::tool::registry::ToolRegistry;
/// use loopctl::middleware::unknown_tool::UnknownToolMiddleware;
///
/// let registry = Arc::new(ToolRegistry::new());
/// let mw = UnknownToolMiddleware::new(registry);
/// // If tool "basj" is not found, error message will say:
/// // "Tool 'basj' not found. Did you mean 'bash'?"
/// ```
pub struct UnknownToolMiddleware {
    /// Tool registry used to enumerate available tool names.
    registry: Arc<ToolRegistry>,
    /// Defaults to `0.4`.
    suggestion_threshold: f64,
}

impl UnknownToolMiddleware {
    /// Create a new unknown-tool middleware with default settings.
    ///
    /// `registry` is the [`ToolRegistry`] used to look up known tools and
    /// generate suggestions when an unknown tool is invoked.
    ///
    /// Uses a [`suggestion_threshold`](Self::with_threshold) of `0.4`,
    /// which balances catching common typos against false-positive
    /// suggestions.
    ///
    /// For a custom threshold see [`with_threshold`](Self::with_threshold).
    #[must_use]
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            suggestion_threshold: 0.4,
        }
    }

    /// Create with a custom similarity threshold.
    ///
    /// Lower values produce more suggestions (more false positives).
    /// Higher values require closer matches.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.suggestion_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Compute similarity between two strings using a normalized
    /// longest-common-subsequence approach.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0
    /// (identical).
    #[must_use]
    pub fn similarity(a: &str, b: &str) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        if a_lower == b_lower {
            return 1.0;
        }

        // Use longest common subsequence length as similarity metric
        let a_chars: Vec<char> = a_lower.chars().collect();
        let b_chars: Vec<char> = b_lower.chars().collect();
        let lcs_len = Self::lcs_length(&a_chars, &b_chars);

        // Tool names are short strings, so u32 is sufficient and avoids
        // usize→f64 precision loss on 64-bit targets.
        let max_len = u32::try_from(a_chars.len().max(b_chars.len())).unwrap_or(u32::MAX);
        let lcs_u32 = u32::try_from(lcs_len).unwrap_or(u32::MAX);

        // Check for prefix match bonus
        let prefix_len = a_chars
            .iter()
            .zip(b_chars.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let prefix_bonus = if prefix_len > 0 {
            let p = u32::try_from(prefix_len).unwrap_or(u32::MAX);
            f64::from(p) / f64::from(max_len) * 0.1
        } else {
            0.0
        };

        f64::from(lcs_u32) / f64::from(max_len) + prefix_bonus
    }

    /// Compute the length of the longest common subsequence of two character slices.
    ///
    /// Uses an iterative dynamic-programming approach with only two rows
    /// to keep memory usage O(min(a, b)).
    fn lcs_length(a: &[char], b: &[char]) -> usize {
        let mut prev = vec![0usize; b.len().saturating_add(1)];
        let mut curr = vec![0usize; b.len().saturating_add(1)];

        for &a_ch in a {
            for (j, &b_ch) in b.iter().enumerate() {
                let j_idx = j.saturating_add(1);
                *curr.get_mut(j_idx).unwrap_or(&mut 0) = if a_ch == b_ch {
                    prev.get(j_idx.saturating_sub(1))
                        .copied()
                        .unwrap_or(0)
                        .saturating_add(1)
                } else {
                    prev.get(j_idx)
                        .copied()
                        .unwrap_or(0)
                        .max(curr.get(j_idx.saturating_sub(1)).copied().unwrap_or(0))
                };
            }
            std::mem::swap(&mut prev, &mut curr);
            curr.fill(0);
        }

        *prev.get(b.len()).unwrap_or(&0)
    }

    /// Find the best matching tool name from a list, given a threshold.
    ///
    /// Returns the name with the highest similarity score that meets or
    /// exceeds `threshold`. Returns `None` when no candidate scores high
    /// enough.
    fn find_best_match_inner<'a>(
        requested: &str,
        available: &[&'a str],
        threshold: f64,
    ) -> Option<(&'a str, f64)> {
        let mut best: Option<(&'a str, f64)> = None;
        for &name in available {
            let score = Self::similarity(requested, name);
            if score >= threshold {
                match best {
                    Some((_, best_score)) if score <= best_score => {}
                    _ => best = Some((name, score)),
                }
            }
        }
        best
    }

    /// Check if a result looks like a "tool not found" error.
    ///
    /// Only considers single [`Text`](ToolContent::Text) results whose
    /// lowercased body mentions a tool that was not found. This is more
    /// specific than matching a bare `"not found"` so that unrelated errors
    /// like "file not found" are not mistaken for unknown tools.
    /// [`Multipart`](ToolContent::Multipart) results and non-error results
    /// always return `false`.
    fn is_tool_not_found(result: &ToolDispatchResult) -> bool {
        if !result.is_error {
            return false;
        }
        let msg = match &result.output {
            ToolContent::Text(t) => t.to_lowercase(),
            ToolContent::Multipart(_) => return false,
        };
        msg.contains("not found") && msg.contains("tool")
    }
}

impl ToolMiddleware for UnknownToolMiddleware {
    fn name(&self) -> &'static str {
        "unknown_tool"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let registry_names = self.registry.tool_names();
        let threshold = self.suggestion_threshold;

        Box::pin(async move {
            let mut result = next.dispatch(ctx).await;

            if Self::is_tool_not_found(&result) {
                // Read the tool name from ctx *after* dispatch so that any
                // redirection applied by downstream middleware is reflected
                // in the suggestion lookup.
                let tool_name = ctx.tool_name.as_str();
                let available_refs: Vec<&str> = registry_names.iter().map(String::as_str).collect();

                if let Some((suggestion, score)) =
                    Self::find_best_match_inner(tool_name, &available_refs, threshold)
                {
                    tracing::info!(
                        requested = %tool_name,
                        suggestion = %suggestion,
                        score = %score,
                        "suggesting alternative tool"
                    );
                    // Append suggestion to the error message
                    if let ToolContent::Text(ref mut msg) = result.output {
                        *msg = format!("{msg}. Did you mean '{suggestion}'?");
                    }
                }
            }

            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::UnknownToolMiddleware;

    #[test]
    fn lcs_length_identical_strings() {
        let a: Vec<char> = "hello".chars().collect();
        let b: Vec<char> = "hello".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 5);
    }

    #[test]
    fn lcs_length_completely_different() {
        let a: Vec<char> = "abc".chars().collect();
        let b: Vec<char> = "xyz".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 0);
    }

    #[test]
    fn lcs_length_empty_first() {
        let b: Vec<char> = "abc".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&[], &b), 0);
    }

    #[test]
    fn lcs_length_empty_second() {
        let a: Vec<char> = "abc".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &[]), 0);
    }

    #[test]
    fn lcs_length_both_empty() {
        assert_eq!(UnknownToolMiddleware::lcs_length(&[], &[]), 0);
    }

    #[test]
    fn lcs_length_subsequence_not_substring() {
        // "ace" is a subsequence of "abcde" but not a substring
        let a: Vec<char> = "ace".chars().collect();
        let b: Vec<char> = "abcde".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 3);
    }

    #[test]
    fn lcs_length_partial_overlap() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        // LCS is "ittn" → length 4
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 4);
    }

    #[test]
    fn lcs_length_order_independent() {
        let a: Vec<char> = "sitting".chars().collect();
        let b: Vec<char> = "kitten".chars().collect();
        assert_eq!(
            UnknownToolMiddleware::lcs_length(&a, &b),
            UnknownToolMiddleware::lcs_length(&b, &a),
        );
    }

    #[test]
    fn lcs_length_single_character_common() {
        let a: Vec<char> = "a".chars().collect();
        let b: Vec<char> = "a".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 1);
    }

    #[test]
    fn lcs_length_repeated_chars() {
        let a: Vec<char> = "aaa".chars().collect();
        let b: Vec<char> = "aa".chars().collect();
        assert_eq!(UnknownToolMiddleware::lcs_length(&a, &b), 2);
    }

    #[test]
    fn find_best_match_exact_hit() {
        let available = ["read_file", "write_file", "list_dir"];
        let (suggestion, score) =
            UnknownToolMiddleware::find_best_match_inner("read_file", &available, 0.5).unwrap();
        assert_eq!(suggestion, "read_file");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_best_match_close_typo() {
        let available = ["read_file", "write_file", "list_dir"];
        let (suggestion, _) =
            UnknownToolMiddleware::find_best_match_inner("read_fil", &available, 0.5).unwrap();
        assert_eq!(suggestion, "read_file");
    }

    #[test]
    fn find_best_match_nothing_above_threshold() {
        let available = ["read_file", "write_file", "list_dir"];
        assert!(UnknownToolMiddleware::find_best_match_inner("xyz", &available, 0.5).is_none());
    }

    #[test]
    fn find_best_match_empty_available() {
        let available: [&str; 0] = [];
        assert!(
            UnknownToolMiddleware::find_best_match_inner("read_file", &available, 0.0).is_none()
        );
    }

    #[test]
    fn find_best_match_empty_requested() {
        let available = ["read_file", "write_file"];
        // An empty request has zero overlap with everything.
        assert!(
            UnknownToolMiddleware::find_best_match_inner("", &available, 0.0).is_none()
                || UnknownToolMiddleware::find_best_match_inner("", &available, 0.0)
                    .unwrap()
                    .1
                    == 0.0
        );
    }

    #[test]
    fn find_best_match_picks_highest_score() {
        let available = ["read", "read_file", "rea"];
        // "read_file" requested → both "read" and "rea" are substrings,
        // but "read" has a higher LCS ratio.
        let (suggestion, _) =
            UnknownToolMiddleware::find_best_match_inner("read_file", &available, 0.3).unwrap();
        assert_eq!(suggestion, "read_file");
    }

    #[test]
    fn find_best_match_threshold_zero_returns_best() {
        let available = ["abc", "xyz"];
        // With threshold 0.0 even a zero-score match could pass, but "abc"
        // shares at least one char with "a" so it wins.
        let result = UnknownToolMiddleware::find_best_match_inner("a", &available, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn find_best_match_threshold_one_requires_exact() {
        let available = ["read_file", "write_file"];
        // Only an exact match can satisfy threshold 1.0.
        let (suggestion, score) =
            UnknownToolMiddleware::find_best_match_inner("read_file", &available, 1.0).unwrap();
        assert_eq!(suggestion, "read_file");
        assert!((score - 1.0).abs() < f64::EPSILON);

        assert!(
            UnknownToolMiddleware::find_best_match_inner("read_fil", &available, 1.0).is_none()
        );
    }

    use crate::message::ToolContent;
    use crate::tool::ToolDispatchResult;
    use std::time::Duration;

    #[test]
    fn is_tool_not_found_error_with_phrase() {
        let result = ToolDispatchResult::err("x", "Tool 'foo' not found".into(), Duration::ZERO);
        assert!(UnknownToolMiddleware::is_tool_not_found(&result));
    }

    #[test]
    fn is_tool_not_found_case_insensitive() {
        let result = ToolDispatchResult::err("x", "TOOL NOT FOUND".into(), Duration::ZERO);
        assert!(UnknownToolMiddleware::is_tool_not_found(&result));
    }

    #[test]
    fn is_tool_not_found_success_result() {
        let result = ToolDispatchResult::ok("x", "done".into(), Duration::ZERO);
        assert!(!UnknownToolMiddleware::is_tool_not_found(&result));
    }

    #[test]
    fn is_tool_not_found_error_without_phrase() {
        let result = ToolDispatchResult::err("x", "permission denied".into(), Duration::ZERO);
        assert!(!UnknownToolMiddleware::is_tool_not_found(&result));
    }

    #[test]
    fn is_tool_not_found_multipart_error() {
        let result = ToolDispatchResult {
            tool_call_id: String::new(),
            output: ToolContent::Multipart(vec![]),
            is_error: true,
            duration: Duration::ZERO,
            resolved_tool_name: "x".into(),
            display_hint: None,
        };
        assert!(!UnknownToolMiddleware::is_tool_not_found(&result));
    }

    use crate::cancel::CancelSignal;
    use crate::middleware::{ToolDispatchContext, ToolMiddleware, ToolPipeline};
    use crate::tool::{
        PermissionCheck, Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema,
    };
    use serde_json::{Value, json};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;

    struct ReadFileTool;
    impl Tool for ReadFileTool {
        fn name(&self) -> &'static str {
            "read_file"
        }
        fn description(&self) -> &'static str {
            "Reads a file"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "read_file".into(),
                description: "Reads a file".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            }
        }
        fn call(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Ok(ToolOutput::text("ok")) })
        }
    }

    fn make_registry() -> Arc<ToolRegistry> {
        let mut reg = ToolRegistry::new();
        reg.register(ReadFileTool);
        Arc::new(reg)
    }

    fn make_ctx(name: &str) -> ToolDispatchContext {
        ToolDispatchContext {
            tool_name: name.to_string(),
            input: json!({}),
            call_id: "call_1".to_string(),
            turn_number: 1,
            cancel: Arc::new(CancelSignal::new()),
            permission: PermissionCheck::Allow,
            tool_context: ToolContext::default(),
        }
    }

    struct RenameMiddleware {
        new_name: String,
    }

    impl ToolMiddleware for RenameMiddleware {
        fn name(&self) -> &'static str {
            "rename"
        }
        fn dispatch<'a>(
            &'a self,
            ctx: &'a mut ToolDispatchContext,
            next: &'a ToolPipeline,
        ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
            ctx.tool_name = self.new_name.clone();
            Box::pin(async move { next.dispatch(ctx).await })
        }
    }

    #[tokio::test]
    async fn dispatch_appends_suggestion_for_typo() {
        let registry = make_registry();
        let pipeline = ToolPipeline::builder()
            .with_middleware(UnknownToolMiddleware::new(Arc::clone(&registry)))
            .with_core(registry)
            .build()
            .expect("valid pipeline");

        let result = pipeline.invoke(make_ctx("read_fil")).await;
        assert!(result.is_error);
        match result.output {
            ToolContent::Text(ref msg) => {
                assert!(msg.contains("not found"), "message was: {msg}");
                assert!(
                    msg.contains("Did you mean 'read_file'?"),
                    "expected suggestion in message: {msg}"
                );
            }
            other @ ToolContent::Multipart(_) => panic!("expected Text output, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dispatch_suggestion_uses_post_dispatch_tool_name() {
        let registry = make_registry();
        let pipeline = ToolPipeline::builder()
            .with_middleware(UnknownToolMiddleware::new(Arc::clone(&registry)))
            .with_middleware(RenameMiddleware {
                new_name: "read_fil".into(),
            })
            .with_core(registry)
            .build()
            .expect("valid pipeline");

        let result = pipeline.invoke(make_ctx("xyz")).await;
        assert!(result.is_error);
        match result.output {
            ToolContent::Text(ref msg) => {
                assert!(
                    msg.contains("Did you mean 'read_file'?"),
                    "suggestion should reflect the renamed tool_name, message was: {msg}"
                );
            }
            other @ ToolContent::Multipart(_) => panic!("expected Text output, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dispatch_known_tool_no_suggestion() {
        let registry = make_registry();
        let pipeline = ToolPipeline::builder()
            .with_middleware(UnknownToolMiddleware::new(Arc::clone(&registry)))
            .with_core(registry)
            .build()
            .expect("valid pipeline");

        let result = pipeline.invoke(make_ctx("read_file")).await;
        assert!(!result.is_error, "known tool should not error");
    }
}
