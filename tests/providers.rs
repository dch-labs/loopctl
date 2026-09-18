//! Cassette replay suites, one module per provider.
//!
//! Each suite replays that provider's committed cassettes hermetically
//! — dummy keys, no network — and the byte-exact matching is the
//! outbound-drift guard.

#![allow(dead_code)]

#[path = "providers/openai_compat.rs"]
mod openai_compat;
