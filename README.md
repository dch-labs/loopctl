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
| [`api_client`](https://docs.rs/loopctl/latest/loopctl/api_client/index.html)  | `ApiClient` trait for LLM provider communication (streaming + non-streaming) |
| [`api_error`](https://docs.rs/loopctl/latest/loopctl/api_error/index.html)   | API error types with retry classification |
| [`builder`](https://docs.rs/loopctl/latest/loopctl/builder/index.html)     | Fluent builder API with type-state generics for compile-time safety |
| [`builtin`](https://docs.rs/loopctl/latest/loopctl/builtin/index.html)     | Reference implementations: `InMemoryStore`, `LoggingObserver` |
| [`cancel`](https://docs.rs/loopctl/latest/loopctl/cancel/index.html)      | Cooperative cancellation via `CancelSignal` (AtomicBool + tokio::Notify) |
| [`compact`](https://docs.rs/loopctl/latest/loopctl/compact/index.html)     | Context compaction: `ContextCompactor` trait, `TruncatingCompactor`, `TokenSplitter` |
| [`core`](https://docs.rs/loopctl/latest/loopctl/core/index.html)        | Core traits (`AgentCore`, `AgentObserver`, `AgentMemory`), config, error, and state types |
| [`engine`](https://docs.rs/loopctl/latest/loopctl/engine/index.html)      | `BareLoop<C>` — the default agent loop engine (stream → accumulate → dispatch tools → repeat) |
| [`loop_control`](https://docs.rs/loopctl/latest/loopctl/loop_control/index.html)| Loop detection, convergence detection, fallback model chains, and manager bundle |
| [`message`](https://docs.rs/loopctl/latest/loopctl/message/index.html)     | Conversation types: `Message`, `MessagePart`, `ToolContent`, roles |
| [`stream`](https://docs.rs/loopctl/latest/loopctl/stream/index.html)      | Streaming event types, accumulator, stop reasons, usage tracking |
| [`tool`](https://docs.rs/loopctl/latest/loopctl/tool/index.html)        | `Tool` trait, `ToolRegistry`, `ToolSchema`, `ToolOutput`, `FnTool` adapter |
| [`testing`](https://docs.rs/loopctl/latest/loopctl/testing/index.html)     | Mock API client, mock tools, and test fixture factories (feature-gated) |

## Quick Start

### Implement a Tool

```rust,ignore
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

```rust,ignore
use loopctl::engine::bare::BareLoop;
use loopctl::tool::ToolRegistry;
use loopctl::core::types::AgentConfig;
use std::sync::Arc;

// 1. Bring your own API client (implements ApiClient trait)
let client = Arc::new(my_provider_client);

// 2. Register tools
let mut registry = ToolRegistry::new();
registry.register(EchoTool);

// 3. Configure
let config = AgentConfig {
    max_turns: 50,
    model: "gpt-4o".into(),
    ..Default::default()
};

// 4. Run
let agent = BareLoop::new(client, registry, config);
let result = agent.run("Use the echo tool to say hello").await?;
println!("Completed in {} turns", result.total_turns);
```

### Use the Testing Module

```toml
[dev-dependencies]
loopctl = { version = "0.1", features = ["testing"] }
```

```rust,ignore
use loopctl::testing::{MockApiClient, MockTool, test_config};
use loopctl::engine::bare::BareLoop;
use loopctl::tool::ToolRegistry;

let mut client = MockApiClient::new();
client.enqueue_response(/* ... */);

let mut registry = ToolRegistry::new();
registry.register(MockTool::new("demo"));

let agent = BareLoop::new(
    client.into_shared(),
    registry,
    test_config(),
);
let result = agent.run("test input").await?;
```

## Feature Flags

| Feature   | Default | Description                            |
|-----------|---------|----------------------------------------|
| `testing` | No      | Mock clients, tools, and test fixtures |

## Architecture

```text
                ┌──────────────┐
                │   ApiClient  │  ← you implement this
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
   │  ToolRegistry │  │  Loop Control  │
   │  (your tools) │  │  • convergence │
   └───────────────┘  │  • detection   │
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
