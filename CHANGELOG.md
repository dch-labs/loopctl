# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
