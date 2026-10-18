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
/// the prompt's — after the query's words pass through the synonym
/// table (see `DEFAULT_SYNONYM_PAIRS`): a user saying "verify",
/// "check", "find", or "look up" matches a tool described with
/// "search" or "read" without the description carrying every phrasing.
/// A [`with_core_tools`](Self::with_core_tools) set rides every
/// non-empty selection regardless of phrasing — the union of the
/// keyword matches and the core set — so a partial wrong match cannot
/// evict a tool every turn may need, while a query that matches
/// nothing still fails all the way open. A prompt whose expanded tokens share
/// nothing with any tool fails open to the full registered surface —
/// zero advertised tools would strand a model that must act, so the
/// failure mode is today's behavior, not silence. Deterministic and
/// cheap: allocation bounded by the registry size and the prompt
/// length.
#[derive(Debug, Clone, Default)]
pub struct KeywordSelector {
    /// Host-added synonym pairs, matched like the default table's.
    ///
    /// Accumulated by [`with_synonyms`](Self::with_synonyms);
    /// bidirectional and query-side, exactly like the curated
    /// defaults.
    extra_synonyms: Vec<(String, String)>,

    /// Tool names included in every selection regardless of the query.
    ///
    /// Accumulated by
    /// [`with_core_tools`](Self::with_core_tools); the union with the
    /// keyword matches is what a turn advertises.
    core_tools: Vec<String>,
}

impl KeywordSelector {
    /// Add host synonym pairs to the default table.
    ///
    /// Each pair is bidirectional and query-side, exactly like the
    /// curated defaults (see `DEFAULT_SYNONYM_PAIRS`): a query word
    /// matching either side of a pair also matches the other side
    /// against the tool text. Sides are lowercased on store, matching
    /// the tokenizer exactly, so capitalization never defeats a pair.
    /// Each side must be **one ASCII-alphanumeric token of at least 3
    /// characters that is not a stopword** — the shape the tokenizer
    /// keeps — or the pair can never fire; multiword phrasings belong
    /// to the phrase table, not the pair table. Repeated additions
    /// accumulate.
    #[must_use]
    pub fn with_synonyms(mut self, pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        self.extra_synonyms.extend(
            pairs
                .into_iter()
                .map(|(left, right)| (left.to_ascii_lowercase(), right.to_ascii_lowercase())),
        );
        self
    }

    /// Always include these tool names in every selection.
    ///
    /// The hybrid core set: the selection becomes the union of the
    /// keyword matches and these names, in registry order, so a
    /// partial wrong match cannot evict a tool every turn may need.
    /// The union composes with fail-open rather than replacing it — a
    /// query that matches nothing still fails all the way open to the
    /// full surface (core included), because a vocabulary miss must
    /// not strand the tool the prompt actually needs. Names not
    /// present in the registry are ignored.
    #[must_use]
    pub fn with_core_tools(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.core_tools.extend(names);
        self
    }

    /// The query's tokens expanded through the synonym tables.
    ///
    /// The default pairs, the host's additions, and the phrase table
    /// (checked against the raw query, since tokenizing drops the
    /// short words phrases are made of) each contribute their
    /// other-side word when a query token matches one side.
    /// Deduplicated; the original tokens always ride along.
    fn expand_query(&self, query: &str, tokens: &[String]) -> Vec<String> {
        let mut expanded = tokens.to_vec();
        let push_new = |word: &str, expanded: &mut Vec<String>| {
            if !expanded.iter().any(|seen| seen == word) {
                expanded.push(word.to_string());
            }
        };
        let defaults: Vec<(&str, &str)> = DEFAULT_SYNONYM_PAIRS.to_vec();
        let hosts: Vec<(&str, &str)> = self
            .extra_synonyms
            .iter()
            .map(|(left, right)| (left.as_str(), right.as_str()))
            .collect();
        for (left, right) in defaults.iter().chain(hosts.iter()) {
            if tokens.iter().any(|token| token == left) {
                push_new(right, &mut expanded);
            }
            if tokens.iter().any(|token| token == right) {
                push_new(left, &mut expanded);
            }
        }
        let lowered = query.to_ascii_lowercase();
        for (phrase, target) in DEFAULT_PHRASE_SYNONYMS {
            if contains_phrase(&lowered, phrase) {
                push_new(target, &mut expanded);
            }
        }
        expanded
    }
}

/// The curated synonym pairs behind keyword recall.
///
/// Bidirectional and query-side: a query containing either word also
/// matches tool text carrying the other. The pairs map the vocabulary
/// users actually use onto the vocabulary tools are described with —
/// a small, conservative set, because every pair widens recall for
/// both of its words. Hosts extend it through
/// [`with_synonyms`](KeywordSelector::with_synonyms); the tool-side
/// lever for vocabulary gaps is description wording, not expansion.
const DEFAULT_SYNONYM_PAIRS: &[(&str, &str)] = &[
    ("check", "verify"),
    ("check", "inspect"),
    ("check", "search"),
    ("find", "search"),
    ("find", "locate"),
    ("open", "read"),
    ("show", "read"),
    ("view", "read"),
    ("fix", "edit"),
    ("change", "edit"),
    ("modify", "edit"),
    ("create", "write"),
    ("make", "write"),
    ("start", "run"),
    ("launch", "run"),
];

/// The curated phrase pairs behind keyword recall.
///
/// Two-word phrases whose short words tokenizing would drop, mapped to
/// their tool-side single word; checked against the raw query with
/// word boundaries, so "look upon" does not fire the "look up" pair.
/// The phrase table is the discriminating layer for shared first
/// words: "look at" reads, "look up" and "look for" search — which is
/// exactly why no single-word `look` pair exists in
/// `DEFAULT_SYNONYM_PAIRS`.
const DEFAULT_PHRASE_SYNONYMS: &[(&str, &str)] = &[
    ("look up", "search"),
    ("look at", "read"),
    ("look for", "search"),
    ("figure out", "plan"),
];

/// Whether `lowered` contains `phrase` on word boundaries.
///
/// Every word of the phrase must appear consecutively with
/// non-alphanumeric neighbors on both sides, so a phrase never fires
/// from inside a longer word.
fn contains_phrase(lowered: &str, phrase: &str) -> bool {
    let mut haystack = lowered;
    let mut offset: usize = 0;
    while let Some(found) = haystack.find(phrase) {
        let start = offset.saturating_add(found);
        let end = start.saturating_add(phrase.len());
        let before_ok = start == 0
            || !lowered[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric);
        let after_ok = !lowered[end..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric);
        if before_ok && after_ok {
            return true;
        }
        offset = end;
        haystack = &lowered[offset..];
    }
    false
}

impl ToolSelector for KeywordSelector {
    fn select(&self, ctx: &SelectionContext<'_>, all: &[ToolSchema]) -> Vec<ToolSchema> {
        let Some(query) = ctx.latest_user_message else {
            return all.to_vec();
        };
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return all.to_vec();
        }
        let expanded = self.expand_query(query, &query_tokens);
        let matched = all
            .iter()
            .filter(|schema| {
                let mut tool_text = schema.tool.clone();
                tool_text.push(' ');
                tool_text.push_str(&schema.description);
                tokenize(&tool_text)
                    .iter()
                    .any(|token| expanded.contains(token))
            })
            .cloned()
            .collect::<Vec<_>>();
        if matched.is_empty() {
            return all.to_vec();
        }
        let matched_names: Vec<&str> = matched.iter().map(|schema| schema.tool.as_str()).collect();
        all.iter()
            .filter(|schema| {
                matched_names.contains(&schema.tool.as_str())
                    || self.core_tools.contains(&schema.tool)
            })
            .cloned()
            .collect()
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
        let selected =
            KeywordSelector::default().select(&ctx("please search the web for rust news"), &all);
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
        let selected =
            KeywordSelector::default().select(&ctx("what is the WEATHER forecast"), &all);
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
        let selected =
            KeywordSelector::default().select(&ctx("completely unrelated zebra topic"), &all);
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
            names(&KeywordSelector::default().select(&no_prompt, &all)),
            vec!["read"],
            "nothing to match against advertises everything"
        );
    }

    #[test]
    fn short_fragments_never_match() {
        let all = vec![schema("ab", "A tool whose name is a short fragment")];
        let selected =
            KeywordSelector::default().select(&ctx("match me by the ab token please"), &all);
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
            names(&KeywordSelector::default().select(&context, &all)),
            names(&KeywordSelector::default().select(&context, &all)),
            "equal contexts select equal surfaces — the estimate must be explainable"
        );
    }

    #[test]
    fn the_selectors_are_object_safe() {
        let all = vec![schema("read", "read a file")];
        let selector: std::sync::Arc<dyn ToolSelector> =
            std::sync::Arc::new(KeywordSelector::default());
        assert_eq!(
            names(&selector.select(&ctx("read something"), &all)),
            vec!["read"],
            "Arc<dyn ToolSelector> dispatches"
        );
    }

    #[test]
    fn a_synonym_only_match_selects_the_tool() {
        let all = vec![
            schema("audit_build", "Check the build for regressions"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selected =
            KeywordSelector::default().select(&ctx("verify the deployment passed"), &all);
        assert_eq!(
            names(&selected),
            vec!["audit_build"],
            "a query word matching only through the synonym table still selects — \
             fail-open would advertise everything instead"
        );
    }

    #[test]
    fn check_the_weather_selects_the_search_tool_through_synonyms() {
        let all = vec![
            schema("web_search", "Search the web for pages"),
            schema("read_file", "Read a file from disk"),
        ];
        let selected = KeywordSelector::default().select(&ctx("check the weather"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "check -> search closes the vocabulary miss without the description \
             carrying the word itself"
        );
    }

    #[test]
    fn look_up_selects_via_the_phrase_table() {
        let all = vec![
            schema("web_search", "Search the web for pages"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look up rust news"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "the two-word phrase maps to its tool-side single word"
        );
    }

    #[test]
    fn host_synonyms_extend_the_default_table() {
        let all = vec![
            schema("food_tool", "Prepare a meal"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector =
            KeywordSelector::default().with_synonyms([("ruoka".to_string(), "meal".to_string())]);
        let selected = selector.select(&ctx("ruoka time"), &all);
        assert_eq!(
            names(&selected),
            vec!["food_tool"],
            "a host pair matches exactly like a default one"
        );
    }

    #[test]
    fn unknown_synonym_words_change_nothing() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("write_file", "Write a file to disk"),
        ];
        let selector = KeywordSelector::default()
            .with_synonyms([("zebra".to_string(), "platypus".to_string())]);
        let selected = selector.select(&ctx("completely unrelated topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "write_file"],
            "an unmatched expansion is a no-op and the miss still fails open"
        );
    }

    #[test]
    fn core_tools_ride_on_every_prompt_including_misses() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selector = KeywordSelector::default().with_core_tools(["read_file".to_string()]);
        let selected = selector.select(&ctx("completely unrelated zebra topic"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "web_search"],
            "a total keyword miss fails open to the full surface — the core set \
             composes with fail-open, never replaces it: a vocabulary miss must \
             not strand the tool the prompt actually needs"
        );
    }

    #[test]
    fn the_widest_synonym_pair_does_not_pull_clearly_unrelated_tools() {
        let all = vec![
            schema("weather", "Report the weather forecast for a city"),
            schema("list_files", "List files in a directory"),
        ];
        let selected = KeywordSelector::default().select(&ctx("show me the weather"), &all);
        assert_eq!(
            names(&selected),
            vec!["weather"],
            "show widens to read but never to list — the precision guard every \
             future pair addition must argue against"
        );
    }

    #[test]
    fn a_partial_wrong_match_cannot_evict_a_core_tool() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector = KeywordSelector::default().with_core_tools(["read_file".to_string()]);
        let selected = selector.select(&ctx("what is the weather in Rome"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file", "weather"],
            "the keyword match and the core set union — a strong wrong-side match \
             cannot evict the tool every turn may need"
        );
    }

    #[test]
    fn host_synonyms_match_case_insensitively() {
        let all = vec![
            schema("query_tool", "Query the dataset"),
            schema("weather", "Report the weather forecast for a city"),
        ];
        let selector =
            KeywordSelector::default().with_synonyms([("SQL".to_string(), "query".to_string())]);
        let selected = selector.select(&ctx("sql time"), &all);
        assert_eq!(
            names(&selected),
            vec!["query_tool"],
            "pair sides normalize on store — a host pair spelled in caps still \
             expands the tokenizer's lowercase query token"
        );
    }

    #[test]
    fn look_at_selects_read_tools_without_pulling_search() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look at the config"), &all);
        assert_eq!(
            names(&selected),
            vec!["read_file"],
            "the phrase table discriminates: look at is a read intent, and the \
             word-level look pair must not drag search tools into it"
        );
    }

    #[test]
    fn look_for_keeps_the_search_intent() {
        let all = vec![
            schema("read_file", "Read a file from disk"),
            schema("web_search", "Search the web for pages"),
        ];
        let selected = KeywordSelector::default().select(&ctx("look for the bug"), &all);
        assert_eq!(
            names(&selected),
            vec!["web_search"],
            "look for is a search intent, carried by the phrase table after the \
             word pair's removal"
        );
    }

    #[test]
    fn every_default_pair_side_survives_tokenization() {
        for (left, right) in DEFAULT_SYNONYM_PAIRS {
            for side in [left, right] {
                let tokens = tokenize(side);
                assert_eq!(
                    tokens,
                    vec![side.to_string()],
                    "every curated pair side must be a single token the tokenizer \
                     keeps (>= 3 chars, non-stopword) or the pair can never fire: \
                     '{side}'"
                );
                assert!(
                    side.chars().all(|ch| ch.is_ascii_alphanumeric()),
                    "pair sides stay ASCII alphanumeric: '{side}'"
                );
            }
        }
    }

    #[test]
    fn every_default_phrase_target_survives_tokenization() {
        for (phrase, target) in DEFAULT_PHRASE_SYNONYMS {
            assert!(
                phrase.contains(' '),
                "phrase keys are multi-word — a single word belongs in the pair \
                 table: '{phrase}'"
            );
            assert!(
                phrase
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch == ' '),
                "phrase keys are lowercase, matching the lowered query: '{phrase}'"
            );
            let tokens = tokenize(target);
            assert_eq!(
                tokens,
                vec![target.to_string()],
                "every phrase target must survive tokenization — it joins the \
                 expanded set: '{target}'"
            );
        }
    }

    #[test]
    fn expansion_keeps_selection_deterministic() {
        let all = vec![
            schema("verify_build", "Check the build for regressions"),
            schema("web_search", "Search the web for pages"),
        ];
        let context = ctx("check the weather and verify the build");
        assert_eq!(
            names(&KeywordSelector::default().select(&context, &all)),
            names(&KeywordSelector::default().select(&context, &all)),
            "expansion is a pure function of the query — equal contexts select \
             equal surfaces"
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
