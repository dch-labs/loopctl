//! Convert an [`OpenRouter`](https://openrouter.ai) models listing into a loopctl price table.
//!
//! The refresh tool for the crate's vendored
//! `src/pricing/curated_prices.json`: read [`OpenRouter`](https://openrouter.ai)'s public
//! `GET /api/v1/models` listing (from a file argument, or stdin),
//! convert it through
//! [`PriceTable::from_openrouter`](loopctl::pricing::PriceTable::from_openrouter),
//! and write the interchange-format JSON to stdout. The fetch is the
//! caller's deliberate act — this tool is offline and deterministic,
//! so the same listing always yields the same table:
//!
//! ```text
//! curl -s https://openrouter.ai/api/v1/models \
//!   | cargo run --example import_openrouter_prices -- --date 2026-10-25 \
//!   > src/pricing/curated_prices.json
//! ```
//!
//! With `--check <existing.json>` the tool becomes the drift probe:
//! the conversion is compared against the existing table by price
//! rows alone (vintage ignored), exit `0` means no drift (nothing is
//! written), exit `3` means a family or price moved (the fresh table
//! lands on stdout, ready to replace the file), and exit `1` is a
//! failure. The nightly workflow is this mode plus a pull request.
//!
//! Then review the diff like code: the table is pinned, dated data,
//! and a price change should be as visible as any other change.

use std::io::Read;

/// Exit status when `--check` found real drift.
///
/// Distinct from `1` (failure) so a caller can branch on "drift found"
/// without parsing stderr; the workflow treats `3` as the PR signal.
const DRIFT: i32 = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut date: Option<String> = None;
    let mut check: Option<String> = None;
    let mut input: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--date" => date = args.next(),
            "--check" => check = args.next(),
            other => input = Some(other.to_string()),
        }
    }
    let Some(date) = date else {
        eprintln!(
            "usage: import_openrouter_prices --date YYYY-MM-DD [--check existing.json] \
             [listing.json < stdin]"
        );
        std::process::exit(2);
    };
    let listing = if let Some(path) = input {
        std::fs::read_to_string(path)?
    } else {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    };
    let table = loopctl::pricing::PriceTable::from_openrouter(&listing, &date)?;
    if let Some(existing_path) = check {
        let existing = loopctl::pricing::PriceTable::from_file(&existing_path)?;
        if table.same_prices(&existing) {
            eprintln!(
                "no drift: {} families unchanged since {}",
                existing.version(),
                existing.pricing_updated()
            );
            return Ok(());
        }
        eprintln!("drift against {existing_path}: writing a table dated {date}");
        println!("{}", table.to_json()?);
        std::process::exit(DRIFT);
    }
    println!("{}", table.to_json()?);
    Ok(())
}
