//! Tool-call memoizing middleware.
//!
//! [`MemoizingMiddleware`] caches successful tool-call results keyed on
//! `(tool_name, hash(canonical input))`. On a cache hit, the middleware
//! returns the prior [`ToolDispatchResult`] with a `[cached]` marker
//! appended and skips the inner dispatch entirely — saving a token-cost
//! and context-pollution hit when a small model re-reads a file or
//! re-runs a grep it already ran.
//!
//! Cache entries are invalidated by:
//!
//! 1. **TTL** — an entry expires after `ttl_turns` turns.
//! 2. **Path-aware write invalidation** — when a write-class tool runs,
//!    entries whose paths (per the [`PathExtractor`]) intersect the
//!    write's paths are evicted. A write to file A does not evict a
//!    cached read of file B.
//!
//! # Registration
//!
//! Register *before* `.with_core(registry)` so the middleware can short-circuit
//! on a hit before the inner dispatch fires:
//!
//! ```rust,ignore
//! use loopctl::middleware::{MemoizingMiddleware, PathExtractor, ToolPipeline};
//! use std::sync::Arc;
//!
//! let extractor: Arc<dyn PathExtractor> = /* … */;
//! let pipeline = ToolPipeline::builder()
//!     .with_middleware(MemoizingMiddleware::new(
//!         vec!["Read".into(), "Grep".into()],
//!         vec!["Write".into(), "Edit".into()],
//!         extractor,
//!         5, // ttl_turns
//!     ))
//!     .with_core(registry)
//!     .build()?;
//! ```

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::message::{ToolContent, ToolContentPart};

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};

// ===================================================
// PathExtractor trait
// ===================================================

/// Extracts the filesystem paths a tool call touches, for cache invalidation.
///
/// Implement this per tool (or per tool family) to tell
/// [`MemoizingMiddleware`] which cached entries a write-class call should
/// invalidate. On a write-class call, the middleware intersects the
/// paths the write touches against each cached entry's paths, and evicts
/// any entry whose path set overlaps the write's.
///
/// # Generosity
///
/// Impls should be generous — return every path the call could affect,
/// not just the "primary" one. Over-returning is safe (just causes more
/// invalidation, never staleness); under-returning risks a stale cache
/// hit when a write should have invalidated an entry but didn't. For
/// example, a `Grep` over a directory should return every file path
/// that could be matched, so a write to any of them invalidates the
/// cached result.
///
/// # Tools that touch no paths
///
/// Return an empty `Vec` for tools whose output doesn't depend on the
/// filesystem (e.g. a pure-computation tool). Such entries are only
/// TTL-evicted, never path-evicted.
pub trait PathExtractor: Send + Sync {
    /// Paths the call touches, as canonical strings.
    ///
    /// The strings are matched verbatim against other calls' path
    /// strings — there is no glob, regex, or prefix logic in the
    /// middleware. If a caller wants prefix matching (e.g. a directory
    /// write invalidating file reads under it), the impl must emit the
    /// matching prefix form itself.
    fn paths(&self, tool_name: &str, input: &Value) -> Vec<String>;
}

/// A [`PathExtractor`] that extracts no paths.
///
/// Disables path-based cache invalidation entirely: cached entries are evicted
/// only by TTL (`ttl_turns`), never by write-class calls. Suitable when the
/// caller cannot (or does not want to) associate tool inputs with filesystem
/// paths — e.g. for a tool whose output is deterministic and path-independent.
///
/// Zero-sized; cheap to construct and `Arc`-share.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use loopctl::middleware::{MemoizingMiddleware, NoopPathExtractor};
///
/// let mw = MemoizingMiddleware::new(
///     vec!["Read".into()],
///     vec!["Write".into()],
///     Arc::new(NoopPathExtractor),
///     5, // TTL in turns — the only invalidation path with this extractor
/// );
/// ```
pub struct NoopPathExtractor;

impl PathExtractor for NoopPathExtractor {
    fn paths(&self, _tool_name: &str, _input: &Value) -> Vec<String> {
        Vec::new()
    }
}

// ===================================================
// Cache types (private)
// ===================================================

/// Cache key: the tool name plus the hash of its canonical-serialized input.
///
/// Both components are required: the same input hash could theoretically
/// collide across tools, so the tool name disambiguates.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct CacheKey {
    /// The tool name the call was issued under.
    ///
    /// Compared verbatim against the configured `tools` list to decide
    /// whether the call is memoized at all. Stored as part of the key so
    /// that the same input hash under different tool names does not
    /// produce a false cache hit.
    tool_name: String,

    /// `DefaultHasher` digest of `serde_json::to_string(&input)`.
    ///
    /// Because `serde_json::Map` is `BTreeMap`-backed (keys sorted
    /// alphabetically), two inputs that differ only in JSON key order
    /// (`{"a":1,"b":2}` vs `{"b":2,"a":1}`) produce the same canonical
    /// string and therefore the same hash — so semantically-equal calls
    /// hit the same cache entry.
    input_hash: u64,
}

/// A single cached entry: the result, when it was inserted, and what
/// paths it touched.
struct CacheEntry {
    /// The successful `ToolDispatchResult` returned on the original call.
    ///
    /// Cloned (with `[cached]` appended to its output and `duration`
    /// zeroed) on a hit. Only successful results (`is_error == false`)
    /// are cached; errors always re-run.
    result: ToolDispatchResult,

    /// The `turn_number` at which the entry was inserted.
    ///
    /// Compared against the current turn for TTL expiry via
    /// `current_turn.saturating_sub(turn_inserted) >= ttl_turns`.
    /// Stale entries are evicted lazily on lookup.
    turn_inserted: usize,

    /// Paths the original call touched, per the `PathExtractor`.
    ///
    /// Used for path-aware invalidation on write-class calls. When a
    /// write tool runs, the middleware intersects the write's paths
    /// against each entry's `paths` set and evicts any overlap. An
    /// empty vec means the call touches no paths (e.g. a pure
    /// computation tool) — such entries are only TTL-evicted, never
    /// path-evicted.
    paths: Vec<String>,
}

// ===================================================
// MemoizingMiddleware
// ===================================================

/// Deduplicates repeat tool calls by `(tool_name, hash(canonical input))`.
///
/// On a cache hit (memoized tool + key present + not TTL-expired),
/// returns the prior [`ToolDispatchResult`] with a `[cached]` marker
/// appended and skips the inner dispatch entirely. On a cache miss,
/// runs the tool and stores the result. On a write-class tool call,
/// invalidates cache entries whose paths (per the [`PathExtractor`])
/// intersect the write's paths.
///
/// Only successful results are cached (`is_error == false`); errors
/// always re-run. Entries expire after `ttl_turns` turns.
///
/// # Cache size
///
/// The cache grows with distinct `(tool, input)` pairs up to TTL
/// expiry. There is no LRU eviction in this middleware; if a long
/// session produces many distinct inputs, pair it with a tighter
/// `ttl_turns`.
///
/// # Registration
///
/// See the [module docs](self) for the registration pattern.
pub struct MemoizingMiddleware {
    /// Interior-mutable cache, guarded by a `std::sync::Mutex`.
    ///
    /// The lock is held only briefly during lookup / insert / evict —
    /// never across the inner `next.dispatch()` call (which could take
    /// seconds and would serialize all tool calls).
    cache: Mutex<HashMap<CacheKey, CacheEntry>>,

    /// Exact-match tool names whose results to cache.
    ///
    /// Tools not in this list bypass the cache entirely.
    tools: Vec<String>,

    /// Exact-match tool names that trigger path-aware invalidation.
    ///
    /// After a write tool runs, the cache evicts entries whose paths
    /// intersect the write's paths. Write results themselves are never
    /// cached.
    write_tools: Vec<String>,

    /// The shared path extractor consulted on every cache insert and
    /// every write-class invalidation.
    ///
    /// Required in the constructor; there is no blunt-clear fallback.
    /// Impls are shared via `Arc<dyn PathExtractor>` so the same path
    /// knowledge backs every cache lookup.
    path_extractor: Arc<dyn PathExtractor>,

    /// Max turns a cache entry is valid.
    ///
    /// An entry expires when
    /// `current_turn.saturating_sub(turn_inserted) >= ttl_turns`. So a
    /// `ttl_turns` of 1 means the entry is valid on the turn after
    /// insertion but expires on the second turn after.
    ttl_turns: u32,
}

impl MemoizingMiddleware {
    /// Construct a memoizing middleware.
    ///
    /// # Arguments
    ///
    /// - `tools` — exact-match names of tools whose results to cache.
    /// - `write_tools` — exact-match names of tools that trigger
    ///   path-aware invalidation.
    /// - `path_extractor` — required; supplies the per-tool path
    ///   knowledge the framework can't infer.
    /// - `ttl_turns` — max turns a cache entry is valid (entry expires
    ///   when `current_turn - turn_inserted >= ttl_turns`).
    #[must_use]
    pub fn new(
        tools: Vec<String>,
        write_tools: Vec<String>,
        path_extractor: Arc<dyn PathExtractor>,
        ttl_turns: u32,
    ) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            tools,
            write_tools,
            path_extractor,
            ttl_turns,
        }
    }
}

impl std::fmt::Debug for MemoizingMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = crate::error::recover_guard(self.cache.lock()).len();
        f.debug_struct("MemoizingMiddleware")
            .field("cache_entries", &len)
            .field("tools", &self.tools)
            .field("write_tools", &self.write_tools)
            .field("ttl_turns", &self.ttl_turns)
            .finish_non_exhaustive()
    }
}

impl ToolMiddleware for MemoizingMiddleware {
    fn name(&self) -> &'static str {
        "memoize"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let is_memoized = self.tools.iter().any(|t| t == &ctx.tool_name);
        let is_write = self.write_tools.iter().any(|t| t == &ctx.tool_name);
        let key = if is_memoized {
            Some(make_key(&ctx.tool_name, &ctx.input))
        } else {
            None
        };
        let ttl_turns = self.ttl_turns;
        let current_turn = ctx.turn_number;

        Box::pin(async move {
            if is_memoized
                && let Some(key) = key.as_ref()
                && let Some(cached) = lookup_fresh(&self.cache, key, current_turn, ttl_turns)
            {
                let mut result = cached;
                append_cached_marker(&mut result.output);
                result.duration = std::time::Duration::ZERO;
                return result;
            }

            let result = next.dispatch(ctx).await;

            if is_write && !result.is_error {
                let write_paths = self.path_extractor.paths(&ctx.tool_name, &ctx.input);
                invalidate_paths(&self.cache, &write_paths);
            } else if is_memoized && !result.is_error {
                let paths = self.path_extractor.paths(&ctx.tool_name, &ctx.input);
                if let Some(key) = key {
                    insert(&self.cache, key, result.clone(), current_turn, paths);
                }
            }

            result
        })
    }
}

/// Build a `CacheKey` for `(tool_name, input)`.
///
/// The input is canonicalized via `serde_json::to_string` (which, with
/// the default `BTreeMap`-backed `serde_json::Map`, sorts object keys
/// alphabetically — so semantically-equal inputs hash equally regardless
/// of key order) and then hashed with `DefaultHasher`.
fn make_key(tool_name: &str, input: &Value) -> CacheKey {
    let canonical = serde_json::to_string(input).unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    CacheKey {
        tool_name: tool_name.to_string(),
        input_hash: hasher.finish(),
    }
}

/// Look up a key in the cache, returning a cloned entry only if it
/// exists and is not TTL-expired. Stale entries are evicted as a side
/// effect of the lookup (lazy expiry).
fn lookup_fresh(
    cache: &Mutex<HashMap<CacheKey, CacheEntry>>,
    key: &CacheKey,
    current_turn: usize,
    ttl_turns: u32,
) -> Option<ToolDispatchResult> {
    let mut guard = crate::error::recover_guard(cache.lock());
    let expired =
        |entry: &CacheEntry| current_turn.saturating_sub(entry.turn_inserted) >= ttl_turns as usize;
    if let Some(entry) = guard.get(key)
        && expired(entry)
    {
        guard.remove(key);
        return None;
    }
    guard.get(key).map(|e| e.result.clone())
}

/// Insert a successful tool result into the cache.
///
/// Stores the result under the computed key alongside the turn it was
/// inserted and the paths the call touched. If an entry with the same
/// key already exists (e.g. the same call was made earlier and the
/// entry wasn't TTL-evicted or path-invalidated in between), it is
/// replaced — the newer result wins.
fn insert(
    cache: &Mutex<HashMap<CacheKey, CacheEntry>>,
    key: CacheKey,
    result: ToolDispatchResult,
    turn_inserted: usize,
    paths: Vec<String>,
) {
    let mut guard = crate::error::recover_guard(cache.lock());
    guard.insert(
        key,
        CacheEntry {
            result,
            turn_inserted,
            paths,
        },
    );
}

/// Evict entries whose paths intersect any of `write_paths`.
///
/// Intersection is exact string equality — no prefix or glob logic.
/// An empty `write_paths` set evicts nothing (a write that touches no
/// known paths can't path-invalidate anything).
fn invalidate_paths(cache: &Mutex<HashMap<CacheKey, CacheEntry>>, write_paths: &[String]) {
    if write_paths.is_empty() {
        return;
    }
    let write_set: HashSet<&str> = write_paths.iter().map(String::as_str).collect();
    let mut guard = crate::error::recover_guard(cache.lock());
    guard.retain(|_, entry| !entry.paths.iter().any(|p| write_set.contains(p.as_str())));
}

/// Append the `[cached]` marker to a `ToolContent`.
///
/// For `Text`, the marker is appended to the existing string. For
/// `Multipart`, a new `Text { text: "[cached]" }` part is pushed.
fn append_cached_marker(output: &mut ToolContent) {
    match output {
        ToolContent::Text(s) => s.push_str("\n[cached]"),
        ToolContent::Multipart(parts) => parts.push(ToolContentPart::Text {
            text: "[cached]".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::CancelSignal;
    use crate::message::ToolContent;
    use crate::middleware::{ToolDispatchContext, ToolPipeline};
    use crate::tool::{PermissionCheck, ToolContext, ToolRegistry};
    use std::future::Future;
    use std::sync::Arc;
    use std::time::Duration;

    /// A `PathExtractor` that reads `input.path` for memoized tools
    /// and `input.path` for write tools. Returns `[]` otherwise.
    struct PathFromInput;

    impl PathExtractor for PathFromInput {
        fn paths(&self, _tool_name: &str, input: &Value) -> Vec<String> {
            input
                .get("path")
                .and_then(Value::as_str)
                .map(std::string::ToString::to_string)
                .into_iter()
                .collect()
        }
    }

    /// A `PathExtractor` that returns `[]` for everything — no path
    /// invalidation possible. Useful for TTL-only tests.
    struct NoPaths;
    impl PathExtractor for NoPaths {
        fn paths(&self, _: &str, _: &Value) -> Vec<String> {
            Vec::new()
        }
    }

    /// A middleware that short-circuits with a fixed result, counting
    /// how many times it was dispatched.
    struct FixedOutputMiddleware {
        output: ToolContent,
        is_error: bool,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ToolMiddleware for FixedOutputMiddleware {
        fn name(&self) -> &'static str {
            "fixed_output"
        }
        fn dispatch<'a>(
            &'a self,
            _ctx: &'a mut ToolDispatchContext,
            _next: &'a ToolPipeline,
        ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
            let output = self.output.clone();
            let is_error = self.is_error;
            let count = self.call_count.clone();
            Box::pin(async move {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                ToolDispatchResult {
                    output,
                    is_error,
                    resolved_tool_name: String::new(),
                    tool_call_id: String::new(),
                    duration: Duration::ZERO,
                    display_hint: None,
                }
            })
        }
    }

    /// Like `FixedOutputMiddleware`, but returns a different result
    /// depending on whether the tool is "Read" or "Write". Needed for
    /// tests that exercise the write-invalidation path in a single
    /// shared pipeline (shared cache).
    struct RoutingFixedOutputMiddleware {
        read_output: ToolContent,
        write_output: ToolContent,
        write_is_error: bool,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ToolMiddleware for RoutingFixedOutputMiddleware {
        fn name(&self) -> &'static str {
            "routing_fixed_output"
        }
        fn dispatch<'a>(
            &'a self,
            ctx: &'a mut ToolDispatchContext,
            _next: &'a ToolPipeline,
        ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
            let (output, is_error) = if ctx.tool_name == "Write" {
                (self.write_output.clone(), self.write_is_error)
            } else {
                (self.read_output.clone(), false)
            };
            let count = self.call_count.clone();
            Box::pin(async move {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                ToolDispatchResult {
                    output,
                    is_error,
                    resolved_tool_name: String::new(),
                    tool_call_id: String::new(),
                    duration: Duration::ZERO,
                    display_hint: None,
                }
            })
        }
    }

    fn make_middleware(extractor: Arc<dyn PathExtractor>, ttl: u32) -> MemoizingMiddleware {
        MemoizingMiddleware::new(
            vec!["Read".to_string()],
            vec!["Write".to_string()],
            extractor,
            ttl,
        )
    }

    fn pipeline(
        mw: MemoizingMiddleware,
        output: ToolContent,
        is_error: bool,
    ) -> (ToolPipeline, Arc<std::sync::atomic::AtomicUsize>) {
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let registry = Arc::new(ToolRegistry::new());
        let pipeline = ToolPipeline::builder()
            .with_middleware(mw)
            .with_middleware(FixedOutputMiddleware {
                output,
                is_error,
                call_count: call_count.clone(),
            })
            .with_core(registry)
            .build()
            .expect("pipeline builds");
        (pipeline, call_count)
    }

    fn ctx_for(tool_name: &str, input: Value, turn: usize) -> ToolDispatchContext {
        ToolDispatchContext {
            tool_name: tool_name.to_string(),
            input,
            call_id: "c1".to_string(),
            turn_number: turn,
            cancel: Arc::new(CancelSignal::new()),
            permission: PermissionCheck::Allow,
            tool_context: ToolContext::default(),
        }
    }

    #[tokio::test]
    async fn repeat_read_returns_cached() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("file contents"), false);

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 0);
        let first = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 1);
        let second = pipeline.dispatch(&mut ctx).await;
        // Inner dispatch not called on a cache hit.
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "second call should hit the cache, not the inner dispatch"
        );
        assert!(
            second.output.to_string().contains("[cached]"),
            "second call output should carry [cached]: {}",
            second.output
        );
        // First call (the cache miss) should not carry [cached].
        assert!(
            !first.output.to_string().contains("[cached]"),
            "first call output should not carry [cached]"
        );
    }

    #[tokio::test]
    async fn different_input_not_cached() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("content"), false);

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "a.rs"}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "b.rs"}), 1);
        let _ = pipeline.dispatch(&mut ctx).await;

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "different input should be a cache miss — inner dispatched twice"
        );
    }

    #[tokio::test]
    async fn write_invalidates_path_matching_cache() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("ok"), false);

        // Read(foo.rs) — cached.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Write(bar.rs) — should NOT invalidate the Read(foo.rs) entry.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "bar.rs"}), 1);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Read(foo.rs) again — should still be cached (Write was to bar.rs).
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 2);
        let r = pipeline.dispatch(&mut ctx).await;
        assert!(
            r.output.to_string().contains("[cached]"),
            "Read(foo.rs) should still be cached after Write(bar.rs): {}",
            r.output
        );

        // Write(foo.rs) — should invalidate the Read(foo.rs) entry.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "foo.rs"}), 3);
        let _ = pipeline.dispatch(&mut ctx).await;

        // Read(foo.rs) again — should re-run (cache invalidated).
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 4);
        let r = pipeline.dispatch(&mut ctx).await;
        assert!(
            !r.output.to_string().contains("[cached]"),
            "Read(foo.rs) should re-run after Write(foo.rs): {}",
            r.output
        );
    }

    #[tokio::test]
    async fn ttl_expiry() {
        // ttl_turns = 2: entry inserted at turn 0 is fresh at turn 1,
        // expired at turn 2 (>= ttl_turns).
        let mw = make_middleware(Arc::new(PathFromInput), 2);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("v"), false);

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "x"}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Turn 1 — fresh, should hit.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "x"}), 1);
        let r = pipeline.dispatch(&mut ctx).await;
        assert!(
            r.output.to_string().contains("[cached]"),
            "should be fresh at turn 1"
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Turn 2 — expired (1 - 0 = 1, not >= 2; but 2 - 0 = 2 >= 2). Re-runs.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "x"}), 2);
        let r = pipeline.dispatch(&mut ctx).await;
        assert!(
            !r.output.to_string().contains("[cached]"),
            "should be expired at turn 2 (2-0 >= ttl_turns=2): {}",
            r.output
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn non_memoized_tool_passes_through() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("grep result"), false);

        // Grep is not in the memoize list (only Read is).
        let mut ctx = ctx_for("Grep", serde_json::json!({"pattern": "foo"}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let mut ctx = ctx_for("Grep", serde_json::json!({"pattern": "foo"}), 1);
        let r = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(
            !r.output.to_string().contains("[cached]"),
            "non-memoized tool should never carry [cached]"
        );
    }

    #[tokio::test]
    async fn write_tool_result_not_cached() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("wrote"), false);

        // Write is a write tool, not a memoized tool.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "x"}), 0);
        let r1 = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            !r1.output.to_string().contains("[cached]"),
            "write tool should never be cached"
        );

        let mut ctx = ctx_for("Write", serde_json::json!({"path": "x"}), 1);
        let r2 = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(
            !r2.output.to_string().contains("[cached]"),
            "write tool should never be cached"
        );
    }

    #[tokio::test]
    async fn error_result_not_cached() {
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let (pipeline, calls) = pipeline(
            mw,
            ToolContent::from_string("file not found"),
            true, // is_error
        );

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "missing"}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let mut ctx = ctx_for("Read", serde_json::json!({"path": "missing"}), 1);
        let _ = pipeline.dispatch(&mut ctx).await;
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "error result should not be cached — second call re-ran"
        );
    }

    #[tokio::test]
    async fn cached_marker_format_text_and_multipart() {
        // Text variant.
        let mw_text = make_middleware(Arc::new(NoPaths), 10);
        let (p_text, _) = pipeline(mw_text, ToolContent::from_string("t"), false);
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "a"}), 0);
        let _ = p_text.dispatch(&mut ctx).await;
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "a"}), 1);
        let r = p_text.dispatch(&mut ctx).await;
        match r.output {
            ToolContent::Text(s) => assert!(s.ends_with("\n[cached]"), "text marker: {s}"),
            ToolContent::Multipart(parts) => {
                panic!("expected Text, got Multipart with {} parts", parts.len())
            }
        }

        // Multipart variant.
        let mw2 = make_middleware(Arc::new(NoPaths), 10);
        let existing = ToolContent::from_multipart(vec![ToolContentPart::Text {
            text: "original".to_string(),
        }]);
        let (p_multi, _) = pipeline(mw2, existing, false);
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "b"}), 0);
        let _ = p_multi.dispatch(&mut ctx).await;
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "b"}), 1);
        let r = p_multi.dispatch(&mut ctx).await;
        match r.output {
            ToolContent::Multipart(parts) => {
                assert_eq!(parts.len(), 2, "should have original + [cached] part");
                match &parts[1] {
                    ToolContentPart::Text { text } => {
                        assert_eq!(text, "[cached]", "multipart marker part: {text}");
                    }
                    ToolContentPart::Image { .. } => panic!("expected Text part, got Image"),
                }
            }
            ToolContent::Text(t) => panic!("expected Multipart, got Text: {t}"),
        }
    }

    #[tokio::test]
    async fn cache_key_canonicalizes_input_key_order() {
        let mw = make_middleware(Arc::new(NoPaths), 10);
        let (pipeline, calls) = pipeline(mw, ToolContent::from_string("ok"), false);

        // Same logical input, different key order.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "x", "limit": 10}), 0);
        let _ = pipeline.dispatch(&mut ctx).await;

        let mut ctx = ctx_for("Read", serde_json::json!({"limit": 10, "path": "x"}), 1);
        let r = pipeline.dispatch(&mut ctx).await;

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "different key order, same logical input should hit the cache"
        );
        assert!(
            r.output.to_string().contains("[cached]"),
            "should be cached despite key-order difference"
        );
    }

    #[test]
    fn memoize_middleware_name() {
        let mw = MemoizingMiddleware::new(vec![], vec![], Arc::new(NoPaths), 5);
        assert_eq!(mw.name(), "memoize");
    }

    #[test]
    fn make_key_same_input_same_hash() {
        let k1 = make_key("Read", &serde_json::json!({"path": "foo.rs"}));
        let k2 = make_key("Read", &serde_json::json!({"path": "foo.rs"}));
        assert_eq!(k1, k2, "identical inputs must produce identical keys");
    }

    #[test]
    fn make_key_different_tool_different_hash() {
        let k1 = make_key("Read", &serde_json::json!({"path": "foo.rs"}));
        let k2 = make_key("Grep", &serde_json::json!({"path": "foo.rs"}));
        assert_ne!(k1, k2, "different tool names must produce different keys");
    }

    #[test]
    fn make_key_different_input_different_hash() {
        let k1 = make_key("Read", &serde_json::json!({"path": "foo.rs"}));
        let k2 = make_key("Read", &serde_json::json!({"path": "bar.rs"}));
        assert_ne!(k1, k2, "different inputs must produce different keys");
    }

    #[test]
    fn lookup_fresh_returns_none_on_miss() {
        let cache = Mutex::new(HashMap::new());
        let key = make_key("Read", &serde_json::json!({"path": "x"}));
        let result = lookup_fresh(&cache, &key, 0, 10);
        assert!(result.is_none(), "empty cache should miss");
    }

    #[test]
    fn lookup_fresh_returns_result_on_hit() {
        let cache = Mutex::new(HashMap::new());
        let key = make_key("Read", &serde_json::json!({"path": "x"}));
        insert(
            &cache,
            key.clone(),
            ToolDispatchResult {
                tool_call_id: String::new(),
                output: ToolContent::from_string("content"),
                is_error: false,
                duration: Duration::ZERO,
                resolved_tool_name: String::new(),
                display_hint: None,
            },
            0,
            vec![],
        );
        let result = lookup_fresh(&cache, &key, 1, 10);
        assert!(result.is_some(), "fresh entry should hit");
    }

    #[test]
    fn lookup_fresh_expires_and_evicts_stale_entry() {
        let cache = Mutex::new(HashMap::new());
        let key = make_key("Read", &serde_json::json!({"path": "x"}));
        insert(
            &cache,
            key.clone(),
            ToolDispatchResult {
                tool_call_id: String::new(),
                output: ToolContent::from_string("content"),
                is_error: false,
                duration: Duration::ZERO,
                resolved_tool_name: String::new(),
                display_hint: None,
            },
            0,
            vec![],
        );
        // ttl_turns=2: entry inserted at turn 0 is fresh at turn 1
        // (1-0=1 < 2), expired at turn 2 (2-0=2 >= 2).
        assert!(
            lookup_fresh(&cache, &key, 1, 2).is_some(),
            "should be fresh at turn 1"
        );
        assert!(
            lookup_fresh(&cache, &key, 2, 2).is_none(),
            "should be expired at turn 2"
        );
        // Lazy eviction: the entry should have been removed by the lookup.
        let guard = crate::error::recover_guard(cache.lock());
        assert!(
            guard.is_empty(),
            "expired entry should be evicted from cache"
        );
    }

    #[test]
    fn insert_stores_entry_and_replace_overwrites() {
        let cache = Mutex::new(HashMap::new());
        let key = make_key("Read", &serde_json::json!({"path": "x"}));
        insert(
            &cache,
            key.clone(),
            ToolDispatchResult {
                tool_call_id: String::new(),
                output: ToolContent::from_string("first"),
                is_error: false,
                duration: Duration::ZERO,
                resolved_tool_name: String::new(),
                display_hint: None,
            },
            0,
            vec![],
        );
        insert(
            &cache,
            key.clone(),
            ToolDispatchResult {
                tool_call_id: String::new(),
                output: ToolContent::from_string("second"),
                is_error: false,
                duration: Duration::ZERO,
                resolved_tool_name: String::new(),
                display_hint: None,
            },
            1,
            vec![],
        );
        let result = lookup_fresh(&cache, &key, 2, 10).unwrap();
        match result.output {
            ToolContent::Text(s) => assert_eq!(s, "second", "second insert should overwrite"),
            ToolContent::Multipart(parts) => {
                panic!("expected Text, got Multipart with {} parts", parts.len())
            }
        }
    }

    #[test]
    fn invalidate_paths_removes_overlapping_entries() {
        let cache = Mutex::new(HashMap::new());
        let key_a = make_key("Read", &serde_json::json!({"path": "a.rs"}));
        let key_b = make_key("Read", &serde_json::json!({"path": "b.rs"}));
        let result = ToolDispatchResult {
            tool_call_id: String::new(),
            output: ToolContent::from_string("content"),
            is_error: false,
            duration: Duration::ZERO,
            resolved_tool_name: String::new(),
            display_hint: None,
        };
        insert(
            &cache,
            key_a.clone(),
            result.clone(),
            0,
            vec!["a.rs".to_string()],
        );
        insert(&cache, key_b.clone(), result, 0, vec!["b.rs".to_string()]);

        // Write to a.rs should evict key_a but not key_b.
        invalidate_paths(&cache, &["a.rs".to_string()]);

        let guard = crate::error::recover_guard(cache.lock());
        assert!(
            !guard.contains_key(&key_a),
            "overlapping entry should be evicted"
        );
        assert!(
            guard.contains_key(&key_b),
            "non-overlapping entry should survive"
        );
    }

    #[test]
    fn invalidate_paths_noop_on_empty_write_paths() {
        let cache = Mutex::new(HashMap::new());
        let key = make_key("Read", &serde_json::json!({"path": "x"}));
        insert(
            &cache,
            key,
            ToolDispatchResult {
                tool_call_id: String::new(),
                output: ToolContent::from_string("content"),
                is_error: false,
                duration: Duration::ZERO,
                resolved_tool_name: String::new(),
                display_hint: None,
            },
            0,
            vec!["x".to_string()],
        );
        invalidate_paths(&cache, &[]);
        let guard = crate::error::recover_guard(cache.lock());
        assert_eq!(guard.len(), 1, "empty write_paths should evict nothing");
    }

    #[test]
    fn invalidate_paths_evicts_all_matching() {
        let cache = Mutex::new(HashMap::new());
        let result = ToolDispatchResult {
            tool_call_id: String::new(),
            output: ToolContent::from_string("content"),
            is_error: false,
            duration: Duration::ZERO,
            resolved_tool_name: String::new(),
            display_hint: None,
        };
        // Two entries both touching "shared.rs".
        let k1 = make_key(
            "Read",
            &serde_json::json!({"path": "shared.rs", "limit": 10}),
        );
        let k2 = make_key(
            "Grep",
            &serde_json::json!({"path": "shared.rs", "pattern": "foo"}),
        );
        insert(&cache, k1, result.clone(), 0, vec!["shared.rs".to_string()]);
        insert(&cache, k2, result, 0, vec!["shared.rs".to_string()]);

        invalidate_paths(&cache, &["shared.rs".to_string()]);
        let guard = crate::error::recover_guard(cache.lock());
        assert!(
            guard.is_empty(),
            "both entries touch shared.rs — both evicted"
        );
    }

    #[test]
    fn append_cached_marker_text() {
        let mut content = ToolContent::from_string("hello");
        append_cached_marker(&mut content);
        match content {
            ToolContent::Text(s) => assert!(s.ends_with("\n[cached]"), "marker appended: {s}"),
            ToolContent::Multipart(parts) => {
                panic!("expected Text, got Multipart with {} parts", parts.len())
            }
        }
    }

    #[test]
    fn append_cached_marker_multipart() {
        let mut content = ToolContent::from_multipart(vec![ToolContentPart::Text {
            text: "original".to_string(),
        }]);
        append_cached_marker(&mut content);
        match content {
            ToolContent::Multipart(parts) => {
                assert_eq!(parts.len(), 2, "should push a new part");
                match &parts[1] {
                    ToolContentPart::Text { text } => {
                        assert_eq!(text, "[cached]", "marker text: {text}");
                    }
                    ToolContentPart::Image { .. } => panic!("expected Text part, got Image"),
                }
            }
            ToolContent::Text(t) => panic!("expected Multipart, got Text: {t}"),
        }
    }

    #[test]
    fn append_cached_marker_empty_text() {
        let mut content = ToolContent::from_string("");
        append_cached_marker(&mut content);
        match content {
            ToolContent::Text(s) => assert_eq!(s, "\n[cached]", "empty text + marker"),
            ToolContent::Multipart(parts) => {
                panic!("expected Text, got Multipart with {} parts", parts.len())
            }
        }
    }

    #[tokio::test]
    async fn failed_write_does_not_invalidate_cache() {
        // A write that errors (permission denied, disk full) didn't
        // modify the filesystem, so cached reads are still accurate and
        // must NOT be invalidated. This uses a shared middleware so the
        // same cache is consulted across both Read and Write calls.
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let registry = Arc::new(ToolRegistry::new());
        let pipeline = ToolPipeline::builder()
            .with_middleware(mw)
            .with_middleware(RoutingFixedOutputMiddleware {
                read_output: ToolContent::from_string("file contents"),
                write_output: ToolContent::from_string("permission denied"),
                write_is_error: true,
                call_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })
            .with_core(registry)
            .build()
            .expect("pipeline builds");

        // Step 1: Read(foo.rs) — cache it.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 0);
        let r1 = pipeline.dispatch(&mut ctx).await;
        assert!(!r1.is_error);

        // Step 2: Write(foo.rs) fails — should NOT invalidate.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "foo.rs"}), 1);
        let wr = pipeline.dispatch(&mut ctx).await;
        assert!(wr.is_error, "write should return an error");

        // Step 3: Read(foo.rs) again — should still be cached.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 2);
        let r3 = pipeline.dispatch(&mut ctx).await;
        assert!(
            r3.output.to_string().contains("[cached]"),
            "failed write should not invalidate cached read: {}",
            r3.output
        );
    }

    #[tokio::test]
    async fn successful_write_does_invalidate_cache() {
        // The counterpart: a successful write to the same path MUST
        // invalidate the cached read.
        let mw = make_middleware(Arc::new(PathFromInput), 10);
        let registry = Arc::new(ToolRegistry::new());
        let pipeline = ToolPipeline::builder()
            .with_middleware(mw)
            .with_middleware(RoutingFixedOutputMiddleware {
                read_output: ToolContent::from_string("file contents"),
                write_output: ToolContent::from_string("wrote"),
                write_is_error: false,
                call_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            })
            .with_core(registry)
            .build()
            .expect("pipeline builds");

        // Step 1: Read(foo.rs) — cache it.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 0);
        let r1 = pipeline.dispatch(&mut ctx).await;
        assert!(!r1.is_error);

        // Step 2: Write(foo.rs) succeeds — should invalidate.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "foo.rs"}), 1);
        let wr = pipeline.dispatch(&mut ctx).await;
        assert!(!wr.is_error, "write should succeed");

        // Step 3: Read(foo.rs) again — should re-run (invalidated).
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 2);
        let r3 = pipeline.dispatch(&mut ctx).await;
        assert!(
            !r3.output.to_string().contains("[cached]"),
            "successful write should invalidate cached read: {}",
            r3.output
        );
    }

    #[test]
    fn noop_path_extractor_returns_empty() {
        let extractor = NoopPathExtractor;
        // Arbitrary tool/input combinations all yield no paths.
        assert!(
            extractor
                .paths("Read", &serde_json::json!({"path": "/etc/hosts"}))
                .is_empty()
        );
        assert!(
            extractor
                .paths("Grep", &serde_json::json!({"pattern": "x"}))
                .is_empty()
        );
        assert!(extractor.paths("Write", &serde_json::json!({})).is_empty());
    }

    #[tokio::test]
    async fn noop_path_extractor_disables_path_invalidation() {
        let mw = make_middleware(Arc::new(NoopPathExtractor), 5);
        let read_content = ToolContent::from_string("file body");
        let (pipeline, _calls) = pipeline(mw, read_content.clone(), false);

        // Step 1: Read(foo.rs) — populates the cache.
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 1);
        let r1 = pipeline.dispatch(&mut ctx).await;
        assert!(!r1.output.to_string().contains("[cached]"));

        // Step 2: Write(foo.rs) — would normally invalidate, but the
        // no-op extractor contributes no paths, so the cache survives.
        let mut ctx = ctx_for("Write", serde_json::json!({"path": "foo.rs"}), 1);
        let wr = pipeline.dispatch(&mut ctx).await;
        assert!(!wr.is_error, "write should succeed");

        // Step 3: Read(foo.rs) again — still cached (NOT invalidated).
        let mut ctx = ctx_for("Read", serde_json::json!({"path": "foo.rs"}), 2);
        let r3 = pipeline.dispatch(&mut ctx).await;
        assert!(
            r3.output.to_string().contains("[cached]"),
            "NoopPathExtractor disables path invalidation; read still cached: {}",
            r3.output
        );
    }
}
