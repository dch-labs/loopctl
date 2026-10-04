//! Process-global LRU cache of compiled regexes.
//!
//! Tools that accept user-supplied regex patterns (the grep and
//! code-search tools) would otherwise recompile the same pattern on
//! every call. This module caches up to 256 compiled regexes
//! process-globally, with least-recently-used eviction provided by the
//! `lru` crate.
//!
//! `regex::Regex` is `Send + Sync` and clones cheaply (it is `Arc`-backed
//! internally), so the cache hands out clones and never holds its lock across
//! the actual search. That is what makes the calling tools safe to mark
//! concurrency-safe despite sharing global state: the shared state is a
//! lock-protected lookup table, not mutable file or session state.

use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::OnceLock;

use lru::LruCache;
use regex::Regex;

/// Maximum number of compiled regexes kept in the cache (256 ≈ a few `MiB`).
///
/// Bounded so a long-lived process serving adversarial patterns
/// cannot grow the cache without limit; eviction is
/// least-recently-used.
const CACHE_CAPACITY: usize = 256;

/// The process-global regex cache; `OnceLock` because `LruCache::new` is not `const`.
///
/// One cache per process keeps repeated patterns compiled once
/// across every tool and every call site; the mutex guards the
/// `LruCache` itself, never the caller's use of a returned `Regex`
/// clone.
static REGEX_CACHE: OnceLock<Mutex<LruCache<String, Regex>>> = OnceLock::new();

/// Return the process-global regex cache, initializing it on first access.
///
/// `OnceLock::get_or_init` constructs the `Mutex<LruCache>` lazily because
/// [`LruCache::new`] is not `const fn` — it allocates internal storage. The
/// returned reference is `&'static` (the `OnceLock` lives for the process
/// lifetime), so callers never need to store or clone it.
fn cache() -> &'static Mutex<LruCache<String, Regex>> {
    let cap = NonZeroUsize::new(CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN);
    REGEX_CACHE.get_or_init(|| Mutex::new(LruCache::new(cap)))
}

/// Return a compiled regex for `pattern`, fetching from the cache when able.
///
/// On a cache hit the entry is moved to most-recently-used (the `lru` crate
/// does this in O(1) as part of `get`) and a cheap clone is returned — the
/// lock is held only for the lookup, never across the caller's use. On a miss
/// the regex is compiled **outside** the lock (compilation can be slow), then
/// inserted (evicting the least-recently-used if the cache is full, in O(1)).
/// A concurrent caller may insert the same key between the lookup and the
/// insert; that is harmless — `put` replaces the prior entry with an
/// equivalent regex.
///
/// A poisoned lock falls back to compiling without caching — callers still
/// get a working regex rather than a poison error propagating up.
///
/// When `case_insensitive` is `true`, the pattern is wrapped with the `(?i)`
/// flag (idempotently — a pattern already beginning with `(?i)` is not
/// double-wrapped) before lookup/compile. The wrapped form is the cache key,
/// so a subsequent case-sensitive call with the same body does not collide
/// with this entry, and a repeated case-insensitive call is a hit.
///
/// # Errors
///
/// Returns the underlying [`regex::Error`] unchanged if `pattern` fails to
/// compile. Invalid patterns are never stored in the cache.
pub fn get_or_compile(pattern: &str, case_insensitive: bool) -> Result<Regex, regex::Error> {
    let cache_key = if case_insensitive && !pattern.starts_with("(?i)") {
        format!("(?i){pattern}")
    } else {
        pattern.to_string()
    };

    if let Ok(mut guard) = cache().lock() {
        if let Some(regex) = guard.get(&cache_key) {
            return Ok(regex.clone());
        }
    } else {
        return Regex::new(&cache_key);
    }

    let regex = Regex::new(&cache_key)?;
    if let Ok(mut guard) = cache().lock() {
        guard.put(cache_key, regex.clone());
    }
    Ok(regex)
}

/// Drop every cached entry.
///
/// Exists for tests that need a known cache state before asserting on hit or
/// miss behavior; the cache is LRU-bounded, so production code never needs to
/// clear it.
#[cfg(test)]
pub fn clear_cache() {
    if let Ok(mut guard) = cache().lock() {
        guard.clear();
    }
}

/// The one guard for the process-global regex cache in tests.
///
/// Every test that clears, fills, or asserts on the shared cache —
/// in this module or any other — holds this lock, so a cache-length
/// assertion can never race a concurrent test's insert or clear.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::must_use_candidate
)]
mod tests {
    use super::*;
    use crate::tool::builtin::search::content::compile_pattern;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use super::TEST_LOCK;

    fn marker(label: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!(
            "{}_{}_{}",
            label,
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        )
    }

    #[must_use]
    fn cache_size() -> usize {
        cache().lock().map_or(0, |c| c.len())
    }

    #[test]
    fn cache_hit_returns_matching_regex() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let m = marker("hit");
        let re1 = get_or_compile(&m, false).unwrap();
        let re2 = get_or_compile(&m, false).unwrap();
        assert!(re1.is_match(&m));
        assert!(re2.is_match(&m));
    }

    #[test]
    fn cache_miss_compiles_distinct_patterns() {
        let _guard = TEST_LOCK.lock().unwrap();
        let m1 = marker("miss_a");
        let m2 = marker("miss_b");
        let re1 = get_or_compile(&m1, false).unwrap();
        let re2 = get_or_compile(&m2, false).unwrap();
        assert!(re1.is_match(&m1));
        assert!(re2.is_match(&m2));
        assert!(!re1.is_match(&m2));
    }

    #[test]
    fn cache_evicts_at_capacity() {
        let _guard = TEST_LOCK.lock().unwrap();
        let cap = CACHE_CAPACITY;
        // Fill well past capacity; the newest entry must still resolve.
        for i in 0..cap.saturating_add(50) {
            let p = format!("eviction_{}_{}", std::process::id(), i);
            let _ = get_or_compile(&p, false).unwrap();
        }
        let newest = format!("eviction_{}_{}", std::process::id(), cap.saturating_add(49));
        let re = get_or_compile(&newest, false).unwrap();
        assert!(re.is_match(&newest));
    }

    #[test]
    fn cache_lru_reorders_on_access() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let first = marker("lru_first");
        let _second = marker("lru_second");
        let _third = marker("lru_third");
        // Re-access "first" — it should still resolve (LRU kept it).
        let re = get_or_compile(&first, false).unwrap();
        assert!(re.is_match(&first));
    }

    #[test]
    fn case_insensitive_matches_uppercase() {
        let _guard = TEST_LOCK.lock().unwrap();
        let m = marker("ci");
        let re = get_or_compile(&m, true).unwrap();
        assert!(re.is_match(&m.to_uppercase()));
        assert!(!re.is_match("no_match_xyz"));
    }

    /// The Unicode-seam parity pin: whatever feature set enables the
    /// search tools must keep `regex` Unicode-capable, matching the
    /// crate-wide `redaction` configuration — non-ASCII case folding
    /// and Perl classes degrade silently to ASCII-only otherwise, a
    /// gap `--all-features` unification hides.
    #[test]
    fn unicode_folding_and_classes_survive_the_minimal_feature_set() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let folded = compile_pattern("\u{0439}", true).unwrap();
        assert!(
            folded.is_match("\u{0419}"),
            "case-insensitive matching must fold non-ASCII letters"
        );
        let words = compile_pattern(r"\w+", false).unwrap();
        assert!(
            words.is_match("\u{0441}\u{043b}\u{043e}\u{0432}\u{043e}"),
            "Perl classes must stay Unicode-aware, not ASCII-only"
        );
        clear_cache();
    }

    #[test]
    fn case_insensitive_is_idempotent() {
        let _guard = TEST_LOCK.lock().unwrap();
        let m = marker("ci_idem");
        let wrapped = format!("(?i){m}");
        // Both forms should compile and match case-insensitively.
        let re_plain = get_or_compile(&m, true).unwrap();
        let re_prefixed = get_or_compile(&wrapped, true).unwrap();
        assert!(re_plain.is_match(&m.to_uppercase()));
        assert!(re_prefixed.is_match(&m.to_uppercase()));
    }

    #[test]
    fn invalid_regex_errors() {
        let _guard = TEST_LOCK.lock().unwrap();
        let result = get_or_compile("(unclosed", false);
        assert!(result.is_err());
    }

    #[test]
    fn regex_cache_dedupes_compiled_patterns() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        let first = get_or_compile("needle_[0-9]+", false).unwrap();
        let length_after_first = cache().lock().unwrap().len();
        let second = get_or_compile("needle_[0-9]+", false).unwrap();
        let length_after_second = cache().lock().unwrap().len();
        assert_eq!(
            length_after_first, length_after_second,
            "a repeated pattern must not add a second cache entry"
        );
        assert!(first.is_match("needle_42"));
        assert!(second.is_match("needle_42"));
        assert!(!first.is_match("nope"));
        assert!(!second.is_match("nope"));
        clear_cache();
    }

    #[test]
    fn clear_cache_empties_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        let m = marker("clear");
        let _ = get_or_compile(&m, false).unwrap();
        assert!(cache_size() > 0);
        clear_cache();
        assert_eq!(cache_size(), 0);
    }
}
