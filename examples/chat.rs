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
use std::sync::atomic::{AtomicBool, Ordering};

use loopctl::api::ApiClient;
use loopctl::config::SessionConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::RunConfig;
use loopctl::engine::core::Loop;
use loopctl::observer::{LoopObserver, ToolPostContext, ToolPreContext};
use loopctl::tool::{FnTool, ToolContext, ToolOutput, ToolRegistry};
use serde_json::json;

/// Shorthand for the boxed-future signature required by [`FnTool`].
type ToolFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<ToolOutput, loopctl::tool::ToolError>>
            + Send
            + 'static,
    >,
>;

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
        Ok(val) => {
            if pos < tokens.len() {
                format!(
                    "Error: unexpected token after expression: {:?}",
                    tokens[pos]
                )
            } else {
                format!("{val}")
            }
        }
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
                Ok(val)
            } else {
                Err("expected closing parenthesis".into())
            }
        }
        Token::Minus => {
            advance(pos);
            Ok(-parse_factor(tokens, pos)?)
        }
        other => Err(format!("unexpected token: {other:?}")),
    }
}

#[allow(dead_code)]
async fn run_repl<C: ApiClient>(client: Arc<C>) {
    eprintln!("Connected to: {}\n", client.model());
    println!("Type a message and press Enter. Type 'quit' to exit.\n");

    // Create the agent once — conversation history persists across inputs.
    let config = SessionConfig::default();
    let run_config = RunConfig::default().with_max_turns(10);
    let mut agent = BareLoop::new(client, build_tools(), config);
    agent.register_observer(Arc::new(PrintingObserver));

    // Pick the engine turn mode at runtime. `NO_STREAM=1` drives each turn via
    // the non-streaming `create_message` path (no per-delta callbacks, the
    // assembled response is printed after the turn). Otherwise stream text
    // deltas live as they arrive.
    let no_stream = std::env::var("NO_STREAM").map_or(false, |v| v == "1");
    #[cfg(feature = "streaming")]
    if no_stream {
        agent.set_turn_mode(loopctl::engine::TurnMode::NonStreaming);
    } else {
        agent.set_text_streamer(Arc::new(|delta| {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }));
    }
    #[cfg(not(feature = "streaming"))]
    {
        let _ = no_stream;
    }

    // Ctrl-C interrupts the in-flight turn (via loopctl's CancelSignal, which
    // `select!`s against the stream) and ends the session. The token is
    // one-shot, so the REPL exits after the interrupted turn rather than
    // continuing.
    let interrupted = Arc::new(AtomicBool::new(false));
    let cancel_signal = agent.cancel_signal();
    let interrupted_flag = Arc::clone(&interrupted);
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            interrupted_flag.store(true, Ordering::SeqCst);
            cancel_signal.cancel();
        }
    });

    let stdin = io::stdin();

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

        match agent.run(input, &run_config).await {
            Ok(result) => {
                // Under the live-streaming mode the text was already printed
                // by the streamer; under non-streaming mode print the assembled
                // response now.
                if no_stream {
                    if let Some(text) = result.output.as_deref()
                        && !text.is_empty()
                    {
                        println!("  {text}");
                    }
                } else if result.output.as_deref().map_or(true, |s| s.is_empty()) {
                    println!("  (empty response)");
                }
                println!(
                    "\n\n  (turns: {}, tokens: {}+{} | total: {}+{})\n",
                    result.turn_count(),
                    result.input_tokens(),
                    result.output_tokens(),
                    agent.session().total_input_tokens(),
                    agent.session().total_output_tokens(),
                );
            }
            Err(e) => {
                if interrupted.load(Ordering::SeqCst) {
                    eprintln!("\n  Interrupted.");
                    break;
                }
                eprintln!("\n  Error: {e}\n");
            }
        }
    }
}

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
                .with_api_key("ollama")
                .with_base_url(base)
                .with_model(model)
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
