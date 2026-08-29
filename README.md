# loopctl

A trait-based framework for building agent loops with pluggable LLM clients, tools, and memory.

[![crates.io](https://img.shields.io/crates/v/loopctl.svg)](https://crates.io/crates/loopctl)
[![docs.rs](https://docs.rs/loopctl/badge.svg)](https://docs.rs/loopctl)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

`loopctl` provides the core infrastructure for building LLM-based agent loops: a streaming
API client abstraction, tool registry, loop detection and convergence, fallback model chains,
cancellation, and a default loop engine (`BareLoop`). You bring your own LLM provider client
and tool implementations; the framework handles the rest.

## Modules

| Module | Description |
|--------|-------------|
| [`api`](https://docs.rs/loopctl/latest/loopctl/api/index.html) | `ApiClient` trait for LLM provider communication (streaming + non-streaming) |
| [`api::error`](https://docs.rs/loopctl/latest/loopctl/api/error/index.html) | API error types with retry classification |
| [`cancel`](https://docs.rs/loopctl/latest/loopctl/cancel/index.html) | Cooperative cancellation via `CancelSignal` (AtomicBool + tokio::Notify) |
| [`capabilities`](https://docs.rs/loopctl/latest/loopctl/capabilities/index.html) | Capability traits (`Observable`, `Detectable`, `Compactable`, etc.) |
| [`compact`](https://docs.rs/loopctl/latest/loopctl/compact/index.html) | Context compaction: `ContextCompactor` trait, `TruncatingCompactor`, `TokenSplitter` |
| [`config`](https://docs.rs/loopctl/latest/loopctl/config/index.html) | Session configuration (`LoopConfig`) |
| [`detection`](https://docs.rs/loopctl/latest/loopctl/detection/index.html) | Loop detection, convergence detection, `DetectionManager` |
| [`engine`](https://docs.rs/loopctl/latest/loopctl/engine/index.html) | `BareLoop<C>` — the default agent loop engine (stream → accumulate → dispatch tools → repeat) |
| [`error`](https://docs.rs/loopctl/latest/loopctl/error/index.html) | Central `LoopError` enum for all framework operations |
| [`fallback`](https://docs.rs/loopctl/latest/loopctl/fallback/index.html) | Circuit-breaker pattern for automatic API model fallback (`FallbackManager`) |
| [`memory`](https://docs.rs/loopctl/latest/loopctl/memory/index.html) | `LoopMemory` trait and entry types; `memory::builtin` provides `InMemoryStore` |
| [`message`](https://docs.rs/loopctl/latest/loopctl/message/index.html) | Conversation types: `Message`, `MessagePart`, `ToolContent`, roles |
| [`middleware`](https://docs.rs/loopctl/latest/loopctl/middleware/index.html) | Tool dispatch pipeline: timeouts, permissions, output limits, unknown-tool handling |
| [`observer`](https://docs.rs/loopctl/latest/loopctl/observer/index.html) | `LoopObserver` trait and `ObserverHost` for lifecycle event observation |
| [`reflection`](https://docs.rs/loopctl/latest/loopctl/reflection/index.html) | Failure reflection and recovery strategies (`Reflector`, `RecoveryStrategy`) |
| [`managers`](https://docs.rs/loopctl/latest/loopctl/managers/index.html) | `LoopManagers` — the default infrastructure bundle |
| [`stream`](https://docs.rs/loopctl/latest/loopctl/stream/index.html) | Streaming event types, accumulator, stop reasons, usage tracking |
| [`tool`](https://docs.rs/loopctl/latest/loopctl/tool/index.html) | `Tool` trait, `ToolRegistry`, `ToolSchema`, `ToolOutput`, `FnTool` adapter |
| [`hooks`](https://docs.rs/loopctl/latest/loopctl/hooks/index.html) | Bidirectional lifecycle control (allow/block/ask before tool use). *Requires `hooks` feature.* |
| [`testing`](https://docs.rs/loopctl/latest/loopctl/testing/index.html) | Mock API client, mock tools, and test fixture factories. *Requires `testing` feature.* |

## Quick Start

### Implement a Tool

```rust,no_run
use loopctl::tool::{Tool, ToolContext, ToolOutput, ToolError, ToolSchema};
use serde_json::{Value, json};
use std::pin::Pin;
use std::future::Future;

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes back the input" }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: "echo".into(),
            description: "Echoes back the input".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "message": { "type": "string" } },
                "required": ["message"]
            }),
        }
    }

    fn call(&self, input: Value, _ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
    {
        let msg = input["message"].as_str().unwrap_or("").to_string();
        Box::pin(async move { Ok(ToolOutput::text(msg)) })
    }
}
```

### Run an Agent Loop

```rust,no_run
use loopctl::engine::BareLoop;
use loopctl::engine::core::Loop;
use loopctl::engine::RunConfig;
use loopctl::tool::ToolRegistry;
use loopctl::config::SessionConfig;
use std::sync::Arc;

// 1. Bring your own API client (implements ApiClient trait)
# struct MyClient;
# use loopctl::api::ApiClient;
# impl ApiClient for MyClient {
#     fn model(&self) -> String { "llm-70b".to_string() }
#     fn stream_messages(&self, _req: &loopctl::api::StreamRequest)
#         -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>> + Send>> {
#         unimplemented!()
#     }
# }
let client = Arc::new(MyClient);

// 2. Register tools
let mut registry = ToolRegistry::new();
// registry.register(EchoTool);

// 3. Configure
let config = SessionConfig::default();

// 4. Run
let mut agent = BareLoop::new(client, registry, config);
// let result = agent.run("Use the echo tool to say hello", &RunConfig::default()).await?;
// println!("Completed in {} turns", result.turn_count());
```

### Use the Testing Module

```toml
[dev-dependencies]
loopctl = { version = "0.1", features = ["testing"] }
```

```rust,no_run
use loopctl::testing::{MockApiClient, MockTool, test_config};
use loopctl::engine::BareLoop;
use loopctl::engine::core::Loop;
use loopctl::tool::ToolRegistry;
use std::sync::Arc;

let mut client = MockApiClient::new("test-model");
client = client.with_text_response("Hello from the mock");

let mut registry = ToolRegistry::new();
registry.register(MockTool::new("demo", "A demo tool"));

let agent = BareLoop::new(
    Arc::new(client),
    registry,
    test_config(),
);
// let result = agent.run("test input").await?;
```

## Feature Flags

| Feature | Default | Depends on | Description |
|---------|---------|------------|-------------|
| `derive` | No | — | Re-exports `loopctl-derive`: `#[derive(Tool)]` generates the `Tool` impl (name, description, schema, dispatch) from a `Deserialize` input struct |
| `azure` | No | `providers`, `openai` | Azure OpenAI via the v1 API — an `OpenAiClient` profile (`AZURE_OPENAI_API_KEY`, resource-named endpoint) |
| `moonshot` | No | `providers`, `openai` | Moonshot AI (Kimi) — an `OpenAiClient` profile (`MOONSHOT_API_KEY`, `api.moonshot.ai`) |
| `bedrock` | No | `providers`, `streaming`, `anthropic`, `hmac`, `sha2`, `hex` | AWS Bedrock (`BedrockClient`) — SigV4 auth, Anthropic-native + Converse paths, binary event-stream decoding |
| `hooks` | No | — | Bidirectional lifecycle hooks (allow/block/ask before tool use, compaction) |
| `testing` | No | — | Mock clients, tools, and test fixtures |
| `tool_health` | No | — | Per-tool health monitoring, circuit breakers, and self-healing routing |
| `tool_shield` | No | `tool_health` | `ToolSafetyShield` risk evaluation (`UnixShield` reference patterns) + opt-in `SafetyShieldMiddleware` enforcement of Block decisions |
| `streaming` | No | `async-stream` | Streaming engine path: `StreamHandler` (retry, timeout, fallback), per-delta observer callbacks (`on_text_delta`, `on_thinking_delta`), `text_streamer`. Without it the engine drives each turn via `ApiClient::create_message`. |
| `providers` | No | `reqwest`, `httpdate`, `bytes` | Base HTTP provider support; enables the `provider` module |
| `openai` | No | `providers`, `streaming` | OpenAI-compatible API client (`provider::openai`) |
| `anthropic` | No | `providers`, `streaming` | Anthropic Claude API client (`provider::anthropic`) |
| `ollama` | No | `providers`, `openai` | Ollama local model client (OpenAI-compatible) |
| `deepseek` | No | `providers`, `openai` | DeepSeek API client (OpenAI-compatible) |
| `grok` | No | `providers`, `openai` | Grok (xAI) API client (OpenAI-compatible) |
| `xai` | No | `grok` | Alias for `grok` (xAI API client) |
| `gemini` | No | `providers`, `streaming` | Google Gemini API client (`provider::gemini`) |
| `zai` | No | `providers`, `anthropic` | Z.AI API client (Anthropic-compatible) |
| `grammar` | No | `providers` | Tool-call grammar providers for grammar-aware samplers (vLLM `guided_json`); enables the `Grammar` mode of `ToolConstraint` |
| `schema_validation` | No | — | JSON Schema validation of `Correction::modified_input` in `LlmReflector` (pulls `jsonschema`); when off, validation is skipped |
| `redaction` | No | `regex` | `RedactingMiddleware` — scrub secrets (bearer headers, AWS keys, PEM blocks, PATs, high-entropy tokens) from tool output as `[REDACTED:<kind>]` |
| `mcp` | No | `rmcp`, `reqwest`, `async-stream`, `jsonschema` | MCP client + server adapters — adapt foreign MCP servers' tools as `Tool` impls (`McpToolProvider`), or serve a `ToolRegistry` over MCP/stdio to any MCP client (`McpServerAdapter`; served schemas are validated with `jsonschema`, external refs refused) |

### Streaming vs non-streaming

The engine selects a turn mode at runtime via
[`TurnMode`](https://docs.rs/loopctl/latest/loopctl/engine/enum.TurnMode.html)
(`NonStreaming` or `Streaming`), set with
[`set_turn_mode`](https://docs.rs/loopctl/latest/loopctl/engine/struct.BareLoop.html#method.set_turn_mode).
The two modes are independent of *whether* the `streaming` feature is compiled
in, though the feature gates what `Streaming` can do:

- [`TurnMode::NonStreaming`](https://docs.rs/loopctl/latest/loopctl/engine/enum.TurnMode.html#variant.NonStreaming)
  drives each turn via [`ApiClient::create_message`] — a single request/response
  with no per-delta callbacks. The full assistant text surfaces through
  [`on_response`](https://docs.rs/loopctl/latest/loopctl/observer/trait.LoopObserver.html#method.on_response).
  Always available, even under `default = []`.

- [`TurnMode::Streaming`](https://docs.rs/loopctl/latest/loopctl/engine/enum.TurnMode.html#variant.Streaming)
  routes turns through [`StreamHandler`](https://docs.rs/loopctl/latest/loopctl/stream/handler/struct.StreamHandler.html)
  with retry, timeout, rate-limit detection, and `on_text_delta` /
  `on_thinking_delta` callbacks for real-time token display. Requires the
  `streaming` feature (implied by every HTTP provider).

The constructor default is `Streaming` when the `streaming` feature is enabled
and `NonStreaming` otherwise — but a constructed loop can switch to either mode
at runtime regardless of the default.

## Architecture

At the center is **BareLoop**, the default agent loop. Each turn it requests a
response from an **ApiClient** (your LLM provider) — via the streaming path
(`StreamHandler`) under `TurnMode::Streaming`, or via `create_message` under
`TurnMode::NonStreaming` — then dispatches any requested tool calls through a
**ToolRegistry**. Results are fed back into the conversation and the cycle
repeats until the model ends its turn or a configured limit is reached.

Two cross-cutting concerns run alongside the main loop:

- **Detection & Fallback** — convergence detection, loop detection, and
  automatic model/API fallback when requests fail.
- **ToolRegistry** — holds your registered tools and routes tool calls to them.

## Development

```bash
make ci        # fmt + check + clippy + test + docs (all features)
make test      # run all tests
make lint      # auto-format
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
