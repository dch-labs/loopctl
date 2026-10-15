//! Per-turn tool advertisement: the dynamic tool surface.
//!
//! Decouples the *registered* surface (everything the
//! [`ToolRegistry`](crate::tool::ToolRegistry) knows) from the
//! *advertised* surface (the subset riding this turn's request).
//! Frontier models handle a wide tool list; small models pick the
//! wrong tool, conflate similar ones, or call tools that do not exist
//! — and every schema spends window tokens on every request. A
//! [`ToolSelector`] narrows what the model sees each turn (P8: the
//! cheap alternative to a bigger model is less confusion).
//!
//! The engine consults the configured selector at each request build
//! and threads the chosen schemas into both the request and the
//! window estimate — the figure counts what the wire carries, never
//! the whole registry. A turn whose request is rebuilt after a
//! compaction pass re-selects against the compacted conversation, so
//! the choice always describes the request it rides. The default is no selector: every registered tool
//! is advertised, byte-identical to a loop that has never heard of
//! selection. [`ToolConstraint::Strict`](crate::structured::ToolConstraint)
//! composes automatically — it constrains the model's calls to
//! whatever is advertised this turn.

use crate::message::Message;
use crate::tool::ToolSchema;

/// What the selector sees when narrowing this turn's advertisement.
///
/// Built by the engine at the turn's request build: the run's turn
/// number, the text of the prompt that opened this run, and the
/// conversation the request carries. Selectors must treat the context
/// as read-only advice — the engine threads their output into the
/// request and the window estimate as one selection.
#[derive(Debug, Clone, Copy)]
pub struct SelectionContext<'a> {
    /// The current turn number within the run.
    ///
    /// Selectors that adapt to depth — fewer tools early, more once
    /// the task's shape is clear — read it here.
    pub turn: usize,

    /// The text of the prompt that opened this run, if any input was
    /// given.
    ///
    /// The strongest selection signal a turn carries; `None` on a
    /// promptless turn, where a fail-open selector advertises
    /// everything.
    pub latest_user_message: Option<&'a str>,

    /// The conversation the request carries, in order.
    ///
    /// The engine's request-build view: transients lead, committed
    /// history follows — the same messages the model sees.
    pub conversation: &'a [Message],
}

/// Chooses which registered tools to advertise to the model this turn.
///
/// Implementations must be deterministic for a given context — the
/// engine threads the result into both the request and its window
/// estimate, so a selector whose output varies between equal contexts
/// would make the estimate unexplainable. The trait is object-safe
/// for use as `Arc<dyn ToolSelector>`.
///
/// # Contract
///
/// - The returned schemas must be members of the `all` slice the
///   engine passed in — the engine advertises the selection verbatim,
///   so a fabricated schema would advertise a tool the registry cannot
///   dispatch.
/// - An empty return advertises no tools this turn (the request omits
///   the tools field); a selector that never wants that outcome owns
///   its fail-open policy (see [`KeywordSelector`]).
/// - [`select`](ToolSelector::select) must not mutate the selector
///   (it takes `&self`).
pub trait ToolSelector: Send + Sync {
    /// The subset of `all` to advertise this turn.
    ///
    /// Returned schemas must be members of `all` — the engine
    /// advertises the selection verbatim.
    fn select(&self, ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema>;
}

/// Advertise every registered tool — the loop's default behavior as an
/// explicit knob.
///
/// Selection-free loops get this for free; the type exists for hosts
/// that want the surface configurable without special-casing `None`,
/// and as the fail-open fallback other selectors document against.
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticSelector;

impl ToolSelector for StaticSelector {
    fn select(&self, _ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema> {
        all.to_vec()
    }
}

/// Advertise the tools whose name or description shares a word with
/// the turn's prompt.
///
/// Tokenizes the prompt and each tool's name + description into
/// lowercase alphanumeric words of length 3 or more (dropping most
/// stopwords and filler), and selects every tool whose tokens overlap
/// the prompt's. A prompt sharing no token with any tool fails open to
/// the full registered surface — zero advertised tools would strand a
/// model that must act, so the failure mode is today's behavior, not
/// silence. Deterministic and cheap: allocation bounded by the
/// registry size and the prompt length.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeywordSelector;

impl ToolSelector for KeywordSelector {
    fn select(&self, ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema> {
        let Some(query) = ctx.latest_user_message else {
            return all.to_vec();
        };
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return all.to_vec();
        }
        let selected = all
            .iter()
            .filter(|schema| {
                let mut tool_text = schema.tool.clone();
                tool_text.push(' ');
                tool_text.push_str(&schema.description);
                tokenize(&tool_text)
                    .iter()
                    .any(|token| query_tokens.contains(token))
            })
            .cloned()
            .collect::<Vec<_>>();
        if selected.is_empty() {
            all.to_vec()
        } else {
            selected
        }
    }
}

/// Lowercase alphanumeric runs of length 3 or more that are not
/// stopwords, in order of first appearance.
///
/// The shared tokenizer behind [`KeywordSelector`]: punctuation and
/// casing vanish, two- and one-letter fragments (filler like "to",
/// "a", "of") drop, and so do the common English glue words that are
/// long enough to count but match everything — "the", "for", "with" —
/// a prompt's "the" must not advertise every tool whose description
/// says "the". The same word appearing twice in the query yields one
/// token — overlap is a set question, not a count.
fn tokenize(text: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "that", "this", "these", "those", "you", "your",
        "are", "was", "were", "has", "have", "had", "its", "can", "could", "will", "would",
        "should", "use", "using", "get", "gets", "all", "how", "what", "when", "where", "who",
        "why", "into", "out", "not", "but", "please",
    ];
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars().chain([' ']) {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() >= 3
                && !STOPWORDS.contains(&current.as_str())
                && !tokens.contains(&current)
            {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(name: &str, description: &str) -> ToolSchema {
        ToolSchema::new(name, description, serde_json::json!({"type": "object"}))
    }

    fn ctx(prompt: &str) -> SelectionContext<'_> {
        SelectionContext {
            turn: 0,
            latest_user_message: Some(prompt),
            conversation: &[],
        }
    }

    fn names(schemas: &[ToolSchema]) -> Vec<&str> {
        schemas.iter().map(|s| s.tool.as_str()).collect()
    }

    #[test]
    fn the_static_selector_advertises_every_registered_tool() {
        let all = vec![
            schema("read", "read a file"),
            schema("write", "write a file"),
        ];
        let selected = StaticSelector.select(&ctx("anything at all"), &all);
        assert_eq!(
            names(&selected),
            vec!["read", "write"],
            "static = the full surface"
        );
    }

    #[test]
    fn a_keyword_query_advertises_the_overlapping_tools_only() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected = KeywordSelector.select(&ctx("please search the web for rust news"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "only the overlapping tool rides"
        );
    }

    #[test]
    fn matching_is_case_insensitive_across_name_and_description() {
        let all = vec![
            schema("weather", "Report the forecast for a city"),
            schema("calc", "Evaluate an arithmetic expression"),
        ];
        let selected = KeywordSelector.select(&ctx("what is the WEATHER forecast"), &all);
        assert_eq!(
            names(&selected),
            vec!["weather"],
            "case and position never matter"
        );
    }

    #[test]
    fn a_keyword_miss_fails_open_to_the_full_surface() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
        ];
        let selected = KeywordSelector.select(&ctx("completely unrelated zebra topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "write_file"],
            "a miss is today's behavior, not silence"
        );
    }

    #[test]
    fn a_promptless_turn_fails_open_to_the_full_surface() {
        let all = vec![schema("read", "read a file")];
        let no_prompt = SelectionContext {
            turn: 3,
            latest_user_message: None,
            conversation: &[],
        };
        assert_eq!(
            names(&KeywordSelector.select(&no_prompt, &all)),
            vec!["read"],
            "nothing to match against advertises everything"
        );
    }

    #[test]
    fn short_fragments_never_match() {
        let all = vec![schema("ab", "A tool whose name is a short fragment")];
        let selected = KeywordSelector.select(&ctx("match me by the ab token please"), &all);
        assert_eq!(
            names(&selected),
            vec!["ab"],
            "the fail-open path serves a query of only-short tokens"
        );
    }

    #[test]
    fn selection_is_deterministic_for_an_unchanged_context() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let context = ctx("search the web");
        assert_eq!(
            names(&KeywordSelector.select(&context, &all)),
            names(&KeywordSelector.select(&context, &all)),
            "equal contexts select equal surfaces — the estimate must be explainable"
        );
    }

    #[test]
    fn the_selectors_are_object_safe() {
        let all = vec![schema("read", "read a file")];
        let selector: std::sync::Arc<dyn ToolSelector> = std::sync::Arc::new(KeywordSelector);
        assert_eq!(
            names(&selector.select(&ctx("read something"), &all)),
            vec!["read"],
            "Arc<dyn ToolSelector> dispatches"
        );
    }

    #[test]
    fn the_tokenizer_drops_filler_and_dedupes() {
        assert_eq!(
            tokenize("Read a FILE, read it now!"),
            vec!["read", "file", "now"],
            "lowercase runs of 3+ chars, first appearance only"
        );
    }

    #[test]
    fn glue_words_never_match() {
        assert!(
            tokenize("search the web for rust news with the api")
                .iter()
                .all(|token| !matches!(token.as_str(), "the" | "for" | "with")),
            "a prompt's glue words must not advertise every tool whose \
             description says 'the'"
        );
    }
}
