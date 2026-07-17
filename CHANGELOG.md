# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/2.0.0.html).

## [Unreleased]

### Added

- `ApiError::rate_limited(message, retry_after)` constructor for the structured
  rate-limit carrier variant.
- `LoopError::RateLimitEscalation { attempts, retry_after }` variant,
  recoverable, raised when the stream handler exhausts rate-limit retries on a
  model and escalates to the circuit breaker.
- `StreamHandlerError::RateLimitEscalation { attempts, retry_after, prior }`
  variant. After `RateLimitConfig::fallback_after_retries` rate-limit retries,
  `stream_turn` returns this instead of looping indefinitely or falling back to
  the same model's non-streaming endpoint.
- Rate-limit backoff sleeps are now clamped to the turn's `total_stream_timeout`
  so a large `Retry-After` cannot overrun the turn budget.
- The engine routes `RateLimitEscalation` to
  `FallbackManager::record_model_failure`, so a sustained rate limit on one
  model trips the circuit breaker and subsequent turns route to the fallback
  model.
- `TokenBucket` and `RateLimiter` (`stream::rate_limit`): a proactive
  client-side token-bucket rate limiter, one bucket per provider `base_url`.
  Attaches to `StreamHandler` via `with_rate_limiter`; gates each stream
  attempt before it fires, with a `max_wait` ceiling that degrades to reactive
  (proceed, risk the 429) rather than hang.
- `ApiClient::base_url()` trait method (default `""`), overridden by the
  OpenAI, Anthropic, and Gemini clients to expose their configured endpoint for
  per-provider bucket keying.
- `ParallelMode` + `ParallelDispatchConfig` in `config.rs`: opt-in parallel
  tool dispatch for independent, concurrency-safe calls within a single turn.
  `LoopConfig` is now `#[non_exhaustive]`; the new `parallel_tool_dispatch`
  field defaults to `Sequential` (v0.1.0 behaviour unchanged).
- `Tool::resource_key(&self, &Value) -> Option<String>` trait method (default
  `None`) for parallel-dispatch resource-conflict detection, plus the
  `FnTool::with_resource_key` builder.
- `MockTool::with_delay(Duration)` builder for timing-sensitive tests.
- `StructuredOutput` trait, `ResponseFormat`, `RequestOptions`, and
  `StructuredError` (`structured` module): request guaranteed-schema JSON
  responses from the model. Includes a lenient JSON extraction helper (handles
  markdown fences/prose prefixes) and a `request_structured::<T>()`
  convenience function.
- `ApiClient::stream_messages_with_options` and
  `create_message_with_options` default methods (additive — existing impls
  compile unchanged). `OpenAiClient`, `AnthropicClient`, and `GeminiClient`
  override both to inject the schema (OpenAI via native `response_format`,
  Anthropic via forced-tool tool-forcing, Gemini via `generationConfig`
  `responseMimeType` + `responseSchema`).
- `ToolOutput::structured<T>`, `structured_value()`, and
  `structured_as::<T>()` for typed tool results that round-trip through JSON.

### Changed

- Both sequential and parallel tool dispatch now check the cancel signal
  between calls. Previously, a Ctrl-C during a multi-tool batch was only
  honored at the next turn boundary; now it aborts the remaining calls in the
  batch.

## [0.1.0] - 2025-07-01

Initial crates.io release.

### Added

- Trait-based LLM loop framework with pluggable clients, tools, and memory
- Core message, error, and session types
- Convergence detection using Jaccard similarity
- Loop detector with configurable thresholds and hard-stop
- Fallback manager for model failover
- LLM streaming support
- Provider clients for OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Grok, and ZAI
- Tool dispatch with panic isolation
- Tool health monitoring and tool shield middleware
- Interactive sessions with hooks
- Context compaction with truncating compactor
- Output limit middleware for token budget enforcement
- Built-in testing utilities for writing LLM loop tests
- Example CLIs: hello, REPL, echo tool, and multi-provider chat

[0.1.0]: https://github.com/dch-labs/loopctl/releases/tag/v0.1.0
