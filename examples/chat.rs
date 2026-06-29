#![allow(clippy::all, clippy::pedantic, clippy::restriction)]
//! Real provider chat CLI — talk to OpenAI, Anthropic, `DeepSeek`, `Grok`, Gemini, or Ollama.
//!
//! Set environment variables to pick the provider:
//!
//! ```sh
//! # OpenAI
//! OPENAI_API_KEY=sk-... cargo run --example chat --features openai
//!
//! # DeepSeek
//! DEEPSEEK_API_KEY=... cargo run --example chat --features deepseek
//!
//! # Grok (xAI)
//! XAI_API_KEY=... cargo run --example chat --features grok
//!
//! # Gemini (Google)
//! GEMINI_API_KEY=... cargo run --example chat --features gemini
//!
//! # Anthropic
//! ANTHROPIC_API_KEY=... cargo run --example chat --features anthropic
//!
//! # Ollama (local, no API key needed)
//! OLLAMA_MODEL=llama3 cargo run --example chat --features ollama
//! ```

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use loopctl::api::ApiClient;
use loopctl::config::LoopConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::loop_core::Loop;
use loopctl::observer::{LoopObserver, ToolPostContext, ToolPreContext};
use loopctl::tool::{FnTool, ToolContext, ToolOutput, ToolRegistry};
use serde_json::json;

// ==================================================
// Type alias for tool function signatures
// ==================================================

/// Shorthand for the boxed-future signature required by [`FnTool`].
type ToolFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<ToolOutput, loopctl::tool::ToolError>>
            + Send
            + 'static,
    >,
>;

// ==================================================
// Observer
// ==================================================

/// A simple observer that prints tool calls and responses to stderr.
struct PrintingObserver;

impl LoopObserver for PrintingObserver {
    fn name(&self) -> &str {
        "chat-printer"
    }

    fn on_tool_pre(&self, ctx: &ToolPreContext) {
        eprintln!("  → calling tool: {}", ctx.tool);
    }

    fn on_tool_post(&self, ctx: &ToolPostContext) {
        let status = if ctx.is_error { "failed" } else { "completed" };
        eprintln!(
            "  ← tool {} {status} ({:.0}ms)",
            ctx.tool,
            ctx.duration.as_secs_f64() * 1000.0
        );
    }
}

// ==================================================
// Usage
// ==================================================

fn print_usage_and_exit() -> ! {
    eprintln!("No provider configured.\n");
    eprintln!("Set one of:");
    eprintln!("  OPENAI_API_KEY=<key>       — OpenAI");
    eprintln!("  DEEPSEEK_API_KEY=<key>     — DeepSeek");
    eprintln!("  XAI_API_KEY=<key>          — Grok (xAI)");
    eprintln!("  GEMINI_API_KEY=<key>       — Google Gemini");
    eprintln!("  ZAI_API_KEY=<key>          — Z.ai (ZhipuAI)");
    eprintln!("  ANTHROPIC_API_KEY=<key>    — Anthropic Claude");
    eprintln!("  OLLAMA_MODEL=<model>       — Local Ollama\n");
    eprintln!("  SELF_HOSTED_BASE_URL=<url> — Any OpenAI-compatible server");
    eprintln!("  SELF_HOSTED_MODEL=<model>  — (required with SELF_HOSTED_BASE_URL)\n");
    eprintln!(
        "Build with: --features openai | deepseek | grok | zai | gemini | anthropic | ollama\n"
    );
    std::process::exit(1);
}

// ==================================================
// Tool functions
// ==================================================

fn echo_fn(input: serde_json::Value, _ctx: &ToolContext) -> ToolFuture {
    Box::pin(async move {
        let text = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)");
        Ok(ToolOutput::text(format!("echo: {text}")))
    })
}

fn current_time_fn(_input: serde_json::Value, _ctx: &ToolContext) -> ToolFuture {
    Box::pin(async move {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(ToolOutput::text(format!("Unix timestamp: {secs}")))
    })
}

fn calculate_fn(input: serde_json::Value, _ctx: &ToolContext) -> ToolFuture {
    Box::pin(async move {
        let expr = input
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let result = simple_eval(expr);
        Ok(ToolOutput::text(result))
    })
}

/// Build the tool registry for the chat example.
fn build_tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();

    tools.register(
        FnTool::new(
            "echo".into(),
            "Echo back the provided message.".into(),
            json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The text to echo back"}
                },
                "required": ["message"]
            }),
            echo_fn,
        )
        .read_only(),
    );

    tools.register(
        FnTool::new(
            "current_time".into(),
            "Get the current wall-clock time as a Unix timestamp.".into(),
            json!({"type": "object", "properties": {}}),
            current_time_fn,
        )
        .read_only(),
    );

    tools.register(
        FnTool::new(
            "calculate".into(),
            "Evaluate a simple arithmetic expression (e.g. \"2 + 3 * 4\"). Supports +, -, *, /, parentheses.".into(),
            json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The arithmetic expression to evaluate"}
                },
                "required": ["expression"]
            }),
            calculate_fn,
        )
        .read_only(),
    );

    tools
}

// ==================================================
// Minimal arithmetic expression evaluator
// (recursive descent: + - * / and parentheses)
// ==================================================

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

fn tokenize(s: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '+' => {
                tokens.push(Token::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(Token::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '/' => {
                tokens.push(Token::Slash);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Ok(n) = num.parse::<f64>() {
                    tokens.push(Token::Num(n));
                }
            }
            _ => {
                chars.next();
            }
        }
    }
    tokens
}

fn simple_eval(expr: &str) -> String {
    let tokens = tokenize(expr);
    let mut pos = 0usize;
    match parse_expr(&tokens, &mut pos) {
        Ok(val) => format!("{val}"),
        Err(e) => format!("Error: {e}"),
    }
}

/// Helper: peek the token at `pos`, safely.
fn peek(tokens: &[Token], pos: usize) -> Option<&Token> {
    tokens.get(pos)
}

/// Helper: advance `pos` by one.
fn advance(pos: &mut usize) {
    *pos += 1;
}

fn parse_expr(tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while let Some(tok) = peek(tokens, *pos) {
        match tok {
            Token::Plus => {
                advance(pos);
                left += parse_term(tokens, pos)?;
            }
            Token::Minus => {
                advance(pos);
                left -= parse_term(tokens, pos)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term(tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_factor(tokens, pos)?;
    while let Some(tok) = peek(tokens, *pos) {
        match tok {
            Token::Star => {
                advance(pos);
                left *= parse_factor(tokens, pos)?;
            }
            Token::Slash => {
                advance(pos);
                left /= parse_factor(tokens, pos)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_factor(tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
    let Some(tok) = peek(tokens, *pos) else {
        return Err("unexpected end of expression".into());
    };
    match tok {
        Token::Num(n) => {
            let val = *n;
            advance(pos);
            Ok(val)
        }
        Token::LParen => {
            advance(pos);
            let val = parse_expr(tokens, pos)?;
            if matches!(peek(tokens, *pos), Some(Token::RParen)) {
                advance(pos);
            }
            Ok(val)
        }
        Token::Minus => {
            advance(pos);
            Ok(-parse_factor(tokens, pos)?)
        }
        other => Err(format!("unexpected token: {other:?}")),
    }
}

// ==================================================
// REPL
// ==================================================

#[allow(dead_code)]
async fn run_repl<C: ApiClient>(client: Arc<C>) {
    eprintln!("Connected to: {}\n", client.model());
    println!("Type a message and press Enter. Type 'quit' to exit.\n");

    // Create the agent once — conversation history persists across inputs.
    let config = LoopConfig {
        max_turns: 10,
        ..Default::default()
    };
    let mut agent = BareLoop::new(client, build_tools(), config);
    agent.register_observer(Arc::new(PrintingObserver));

    // Stream text deltas in real-time.
    agent.set_text_streamer(Arc::new(|delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
    }));

    let stdin = io::stdin();
    let mut total_input: u64 = 0;
    let mut total_output: u64 = 0;

    loop {
        print!("> ");
        #[allow(clippy::let_underscore_must_use)]
        let _ = io::stdout().flush();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        match agent.run(input).await {
            Ok(result) => {
                total_input += result.input_tokens;
                total_output += result.output_tokens;
                // Text was already streamed live. Just print stats.
                if result
                    .final_output
                    .as_deref()
                    .map_or(true, |s| s.is_empty())
                {
                    println!("  (empty response)");
                }
                println!(
                    "\n\n  (turns: {}, tokens: {}+{} | total: {}+{})\n",
                    result.total_turns,
                    result.input_tokens,
                    result.output_tokens,
                    total_input,
                    total_output
                );
            }
            Err(e) => {
                eprintln!("\n  Error: {e}\n");
            }
        }
    }
}

// ==================================================
// Provider detection & main
//
// Each provider produces a different concrete type, so we can't unify
// them into a single return. Instead, the `try_provider!` macro wraps
// the repeated "check env → build-or-die → run_repl → return" pattern.
// ==================================================

/// Build a client from the given expression, or print the error and exit.
macro_rules! build_or_die {
    ($provider:expr, $label:literal) => {{
        let label: &str = $label;
        $provider.unwrap_or_else(|e| {
            eprintln!("{label} client error: {e}");
            std::process::exit(1);
        })
    }};
}

#[cfg(feature = "ollama")]
fn ollama_model_from_env() -> Option<String> {
    let direct = std::env::var("OLLAMA_MODEL").or_else(|_| std::env::var("MODEL"));
    match direct {
        Ok(m) => Some(m),
        Err(_) => {
            let is_ollama_url =
                std::env::var("OPENAI_BASE_URL").is_ok_and(|u| u.contains("localhost:11434"));
            if is_ollama_url {
                Some("llama3".into())
            } else {
                None
            }
        }
    }
}

#[tokio::main]
async fn main() {
    // Ollama (OpenAI-compatible, local)
    #[cfg(feature = "ollama")]
    if let Some(model) = ollama_model_from_env() {
        let base = std::env::var("OLLAMA_BASE_URL")
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "http://localhost:11434/v1".into());

        let client = build_or_die!(
            loopctl::provider::OpenAiClient::builder()
                .api_key("ollama")
                .base_url(base)
                .model(model)
                .build(),
            "Ollama"
        );
        run_repl(Arc::new(client)).await;
        return;
    }

    // DeepSeek (OpenAI-compatible)
    #[cfg(feature = "deepseek")]
    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::deepseek(), "DeepSeek");
        run_repl(Arc::new(client)).await;
        return;
    }

    // Grok / xAI (OpenAI-compatible)
    #[cfg(feature = "grok")]
    if std::env::var("XAI_API_KEY").is_ok() || std::env::var("GROK_API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::grok(), "Grok");
        run_repl(Arc::new(client)).await;
        return;
    }

    // Gemini (Google)
    #[cfg(feature = "gemini")]
    if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::GeminiClient::from_env(), "Gemini");
        run_repl(Arc::new(client)).await;
        return;
    }

    // Z.ai (ZhipuAI / BigModel)
    #[cfg(feature = "zai")]
    if std::env::var("ZAI_API_KEY").is_ok() || std::env::var("ZHIPUAI_API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::zai(), "Z.ai");
        run_repl(Arc::new(client)).await;
        return;
    }

    // Self-hosted (vLLM, LM Studio, etc.)
    #[cfg(feature = "openai")]
    if let Ok(base) = std::env::var("SELF_HOSTED_BASE_URL") {
        let model = std::env::var("SELF_HOSTED_MODEL").unwrap_or_else(|_| "default".into());
        let client = build_or_die!(loopctl::provider::self_hosted(&base, &model), "Self-hosted");
        run_repl(Arc::new(client)).await;
        return;
    }

    // OpenAI
    #[cfg(feature = "openai")]
    if std::env::var("OPENAI_API_KEY").is_ok() || std::env::var("API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::OpenAiClient::from_env(), "OpenAI");
        run_repl(Arc::new(client)).await;
        return;
    }

    // Anthropic
    #[cfg(feature = "anthropic")]
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        let client = build_or_die!(loopctl::provider::AnthropicClient::from_env(), "Anthropic");
        run_repl(Arc::new(client)).await;
        return;
    }

    print_usage_and_exit();
}
