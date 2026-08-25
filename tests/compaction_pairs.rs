//! Randomized pairing contracts for `TruncatingCompactor` and `TokenSplitter`.
//!
//! Pins four properties over ten thousand deterministic random histories
//! per machine (reused call ids, duplicate and orphaned results, pending
//! calls, arbitrary `min_messages`/`preserve_recent` combinations):
//! every input call/result pair is either fully in the compacted output
//! or fully out; no orphaned tool result is carried into a reduced
//! output; the output is never an empty list and never contains an
//! empty message; and the splitter never separates a pair across
//! `to_compact`/`preserved`. A conversation consisting solely of
//! orphaned results is the one carve-out — it is returned as received,
//! because filtering it would produce the empty history the compactor
//! guarantees never to emit.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::str_to_string,
    clippy::unreadable_literal
)]

use loopctl::compact::ContextCompactor;
use loopctl::compact::HeuristicTokenCounter;
use loopctl::compact::TokenSplitter;
use loopctl::compact::TruncatingCompactor;
use loopctl::compact::types::{CompactReason, CompactionContext};
use loopctl::message::{Message, MessagePart, Role, ToolContent};
use std::sync::Arc;

/// Number of random histories each property test explores.
///
/// The generators are seeded per test, so raising this extends the same
/// deterministic sequence — previously recorded iteration numbers stay
/// reproducible.
const HISTORIES_PER_PROPERTY: u64 = 10_000;

/// A deterministic linear-congruential generator.
///
/// Fixed seeds keep every run over the same two thousand histories —
/// a failure is always reproducible by its iteration number.
struct Lcg(u64);

impl Lcg {
    /// Advance the generator and return the new high bits.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    /// Draw a value below `n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Build one random history: call, result, or plain-text messages with
/// ids drawn from a three-letter alphabet, so reuse and orphans occur
/// naturally.
///
/// Every call carries its message index in its input and every result
/// carries its message index in its content, so output presence is
/// checkable without content aliasing even when ids repeat.
fn gen_messages(rng: &mut Lcg) -> Vec<Message> {
    let len = 4 + rng.below(6) as usize;
    (0..len)
        .map(|i| {
            let kind = rng.below(10);
            if kind < 3 {
                let id = ["a", "b", "c"][rng.below(3) as usize];
                Message::new(
                    Role::Assistant,
                    vec![MessagePart::tool_call(
                        id,
                        "Read",
                        serde_json::json!({ "i": i }),
                    )],
                )
            } else if kind < 6 {
                let id = ["a", "b", "c"][rng.below(3) as usize];
                Message::new(
                    Role::User,
                    vec![MessagePart::tool_result(
                        id,
                        "Read",
                        ToolContent::from_string(format!("r{i}")),
                        false,
                    )],
                )
            } else if kind < 8 {
                Message::user("q")
            } else {
                Message::assistant("a")
            }
        })
        .collect()
}

/// Pair calls with results the way the implementation does: each result
/// claims the most recent preceding unconsumed call with the same id.
///
/// Returns the `(call message, result message)` index pairs plus the
/// set of result indices that found a call.
fn input_pairs(messages: &[Message]) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut pending: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    let mut pairs = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        for part in &msg.parts {
            match part {
                MessagePart::ToolCall { id, .. } => {
                    pending.entry(id.clone()).or_default().push(i);
                }
                MessagePart::ToolResult { call_id, .. } => {
                    if let Some(cm) = pending.get_mut(call_id).and_then(Vec::pop) {
                        pairs.push((cm, i));
                    }
                }
                _ => {}
            }
        }
    }
    let paired = pairs.iter().map(|(_, r)| *r).collect();
    (pairs, paired)
}

/// The tool id carried by a message's first part, for presence checks.
fn part_id(messages: &[Message], idx: usize) -> String {
    messages
        .get(idx)
        .and_then(|m| m.parts.first())
        .map_or_else(String::new, |p| match p {
            MessagePart::ToolCall { id, .. } | MessagePart::ToolResult { call_id: id, .. } => {
                id.clone()
            }
            _ => String::new(),
        })
}

/// A fixed compaction context, irrelevant to the pairing contracts.
fn ctx() -> CompactionContext {
    CompactionContext {
        tokens_before: 10_000,
        reason: CompactReason::ThresholdExceeded,
        context_window: 8_000,
        turn: 3,
        counter: Arc::new(HeuristicTokenCounter),
    }
}

#[tokio::test]
async fn compaction_invariants_hold_under_random_histories() {
    let mut rng = Lcg(0x5EED_600D);
    for iter in 0..HISTORIES_PER_PROPERTY {
        let messages = gen_messages(&mut rng);
        let compactor = TruncatingCompactor::new()
            .with_min_messages(2 + rng.below(3) as usize)
            .with_preserve_recent(1 + rng.below(5) as usize);
        let outcome = compactor.compact(messages.clone(), 1, ctx()).await;
        assert!(outcome.success, "iter {iter}");

        let (pairs, paired_results) = input_pairs(&messages);
        let out_calls: std::collections::HashSet<(String, String)> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolCall { id, input, .. } => {
                    Some((id.clone(), input.get("i")?.to_string()))
                }
                _ => None,
            })
            .collect();
        let out_results: std::collections::HashSet<(String, String)> = outcome
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                MessagePart::ToolResult {
                    call_id, output, ..
                } => Some((call_id.clone(), output.to_string())),
                _ => None,
            })
            .collect();

        for (cm, rm) in &pairs {
            let call_in = out_calls.contains(&(part_id(&messages, *cm), cm.to_string()));
            let result_in = out_results.contains(&(part_id(&messages, *rm), format!("r{rm}")));
            assert_eq!(
                call_in, result_in,
                "iter {iter}: pair ({cm},{rm}) split — call kept {call_in}, result kept {result_in}"
            );
        }
        let input_all_lone_results = paired_results.is_empty()
            && messages.iter().all(|m| {
                m.parts
                    .iter()
                    .all(|p| matches!(p, MessagePart::ToolResult { .. }))
            });
        if !input_all_lone_results {
            for (id, content) in &out_results {
                let idx: usize = content
                    .trim_start_matches('r')
                    .parse()
                    .unwrap_or(usize::MAX);
                assert!(
                    paired_results.contains(&idx),
                    "iter {iter}: orphaned result {id}/{content} carried into the output"
                );
            }
        }
        assert!(
            outcome.messages.iter().all(|m| !m.parts.is_empty()),
            "iter {iter}: empty message in the output"
        );
        assert!(
            !outcome.messages.is_empty(),
            "iter {iter}: compaction emptied the conversation"
        );
    }
}

#[test]
fn splitter_preserves_pair_closure_under_random_histories() {
    let mut rng = Lcg(0xC0FF_EE00);
    for iter in 0..HISTORIES_PER_PROPERTY {
        let messages = gen_messages(&mut rng);
        let splitter = TokenSplitter::new()
            .with_min_messages(2)
            .with_preserve_recent(1 + rng.below(5) as usize);
        let split = splitter.split(&messages);
        let (pairs, _) = input_pairs(&messages);
        for (cm, rm) in &pairs {
            assert_eq!(
                cm < &split.split_index,
                rm < &split.split_index,
                "iter {iter}: pair ({cm},{rm}) split at {}",
                split.split_index
            );
        }
    }
}
