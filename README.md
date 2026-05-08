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
| [`api_client`] | `ApiClient` trait for LLM provider communication (streaming + non-streaming) |
| [`api_error`] | API error types with retry classification |
| [`builder`] | Fluent builder API with type-state generics for compile-time safety |
| [`builtin`] | Reference implementations: `InMemoryStore`, `LoggingObserver` |
| [`cancel`] | Cooperative cancellation via `CancelSignal` (AtomicBool + tokio::Notify) |
| [`core`] | Core traits (`AgentCore`, `AgentObserver`, `AgentMemory`), config, error, and state types |
| [`engine`] | `BareLoop<C>` — the default agent loop engine (stream → accumulate → dispatch tools → repeat) |
| [`loop_control`] | Loop detection, convergence detection, fallback model chains, and manager bundle |
| [`message`] | Conversation types: `Message`, `MessagePart`, `ToolContent`, roles |
| [`stream`] | Streaming event types, accumulator, stop reasons, usage tracking |
| [`tool`] | `Tool` trait, `ToolRegistry`, `ToolSchema`, `ToolOutput`, `FnTool` adapter |
| [`testing`] | Mock API client, mock tools, and test fixture factories (feature-gated) |

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

| Feature | Default | Description |
|---------|---------|-------------|
| `testing` | No | Mock clients, tools, and test fixtures |

## Architecture

```
                 ┌──────────────┐
                 │   ApiClient  │  ← you implement this
                 └──────┬───────┘
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
