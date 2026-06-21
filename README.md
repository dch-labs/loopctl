# loopctl

A trait-based framework for building agent loops with pluggable LLM clients, tools, and memory.

[![crates.io](https://img.shields.io/crates/v/loopctl.svg)](https://crates.io/crates/loopctl)
[![docs.rs](https://docs.rs/loopctl/badge.svg)](https://docs.rs/loopctl)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

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
| [`builder`](https://docs.rs/loopctl/latest/loopctl/builder/index.html) | Fluent builder API with type-state generics for compile-time safety |
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
| [`runtime`](https://docs.rs/loopctl/latest/loopctl/runtime/index.html) | `LoopRuntime` — the default infrastructure bundle |
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
use loopctl::tool::ToolRegistry;
use loopctl::config::LoopConfig;
use std::sync::Arc;

// 1. Bring your own API client (implements ApiClient trait)
# struct MyClient;
# use loopctl::api::ApiClient;
# impl ApiClient for MyClient {
#     fn model(&self) -> &str { "llm-70b" }
#     fn stream_messages(&self, _req: loopctl::api::StreamRequest)
#         -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<loopctl::stream::StreamEvent, loopctl::stream::StreamError>> + Send>> {
#         unimplemented!()
#     }
# }
let client = Arc::new(MyClient);

// 2. Register tools
let mut registry = ToolRegistry::new();
// registry.register(EchoTool);

// 3. Configure
let config = LoopConfig {
    max_turns: 50,
    model: "llm-70b".into(),
    ..Default::default()
};

// 4. Run
let agent = BareLoop::new(client, registry, config);
// let result = agent.run("Use the echo tool to say hello").await?;
// println!("Completed in {} turns", result.total_turns);
```

### Use the Testing Module

```toml
[dev-dependencies]
loopctl = { version = "0.1", features = ["testing"] }
```

```rust,no_run
use loopctl::testing::{MockApiClient, MockTool, test_config};
use loopctl::engine::BareLoop;
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
| `hooks` | No | — | Bidirectional lifecycle hooks (allow/block/ask before tool use, compaction) |
| `testing` | No | — | Mock clients, tools, and test fixtures |
| `tool_health` | No | — | Per-tool health monitoring, circuit breakers, and self-healing routing |
| `tool_shield` | No | `tool_health` | Tool permission shielding and access control |

## Architecture

```text
                ┌──────────────┐
                │   ApiClient  │
                └───────┬──────┘
                        │
          ┌─────────────▼─────────────┐
          │         BareLoop          │
          │  ┌─────────────────────┐  │
          │  │  stream → accumulate│  │
          │  │  → tool dispatch    │  │
          │  │  → repeat           │  │
          │  └─────────────────────┘  │
          └─────┬──────────┬──────────┘
                │          │
   ┌────────────▼──┐  ┌────▼───────────┐
   │  ToolRegistry │  │  Detection &   │
   │  (your tools) │  │  Fallback      │
   └───────────────┘  │  • convergence │
                      │  • loop detect │
                      │  • fallback    │
                      └────────────────┘
```

## Development

```bash
make ci        # fmt + check + clippy + test + docs (all features)
make test      # run all tests
make lint      # auto-format
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
