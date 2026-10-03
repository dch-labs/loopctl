//! Wire-shape pins for the real embedders — request bodies, response
//! reassembly, batching, and the guards — against an in-process mock
//! server.
//!
//! Every test is deterministic and network-free: the mocks pin the exact
//! request shape (model, input array, bearer header, `dimensions`
//! parameter) and the exact response the real endpoints send, so wire
//! drift fails here instead of in production.
//!
//! Run: `cargo test --all-features --test embeddings_wire`

#![cfg(all(feature = "openai", feature = "vector_index"))]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::arithmetic_side_effects,
    clippy::single_char_pattern,
    clippy::let_underscore_must_use
)]

use loopctl::provider::embeddings::{OllamaEmbedder, OpenAiEmbedder};

/// An OpenAI embedder pointed at `server`, carrying the wire-test key.
///
/// The default geometry (1536 dimensions) is kept so the response
/// fixtures exercise the exact production shapes.
fn openai_at(server: &httpmock::MockServer) -> OpenAiEmbedder {
    OpenAiEmbedder::builder()
        .with_api_key("sk-wire-test")
        .with_base_url(server.base_url())
        .build()
        .unwrap()
}

/// An Ollama embedder pointed at `server` with a small test dimension.
///
/// Ollama models have fixed dimensions, so the wire fixtures pin a
/// three-component geometry through the builder the same way a caller
/// pins a real model's.
#[cfg(feature = "ollama")]
fn ollama_at(server: &httpmock::MockServer) -> OllamaEmbedder {
    OllamaEmbedder::builder()
        .with_model("wire-test-embed")
        .with_dim(3)
        .with_base_url(server.base_url())
        .build()
        .unwrap()
}

/// One OpenAI-shaped data row: `index` with a `marker`-leading vector.
///
/// The marker is the first component of a 1536-component vector, so one
/// number identifies which input a row answers while the body keeps the
/// production dimension.
fn openai_row(marker: f32, index: usize) -> String {
    openai_row_with(marker, index, 1536)
}

/// One OpenAI-shaped data row of `components` components.
///
/// The truncation test pins a 256-component row; everything else uses
/// the production length through [`openai_row`].
fn openai_row_with(marker: f32, index: usize, components: usize) -> String {
    format!(
        "{{\"index\":{index},\"embedding\":[{marker},{index}.0,{}]}}",
        vec!["0.0"; components.saturating_sub(2)].join(",")
    )
}

/// A full OpenAI-shaped response body for `rows`-many `(marker, index)`
/// pairs, in the given order.
fn openai_body(rows: &[(f32, usize)]) -> String {
    let rendered: Vec<String> = rows
        .iter()
        .map(|(marker, index)| openai_row(*marker, *index))
        .collect();
    format!(
        "{{\"data\":[{}],\"model\":\"text-embedding-3-small\",\"usage\":{{\"prompt_tokens\":7,\"total_tokens\":7}}}}",
        rendered.join(",")
    )
}

#[tokio::test]
async fn openai_wire_reassembles_by_index_even_when_data_arrives_shuffled() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .header("authorization", "Bearer sk-wire-test");
            then.status(200)
                .body(openai_body(&[(9.0, 2), (1.0, 0), (5.0, 1)]));
        })
        .await;
    let embedder = openai_at(&server);
    let embeddings = embedder
        .embed_texts(&["first input", "second input", "third input"])
        .await
        .unwrap();
    mock.assert_calls(1);
    assert_eq!(
        embeddings.len(),
        3,
        "one embedding per input regardless of the wire's row order"
    );
    for (position, marker) in [(0_usize, 1.0_f32), (1, 5.0), (2, 9.0)] {
        let component = embeddings[position].as_slice().first().copied();
        assert_eq!(
            component,
            Some(marker),
            "position {position} must carry input {position}'s vector — \
             reassembly keys on the data row's index field, never on array position"
        );
    }
}

#[tokio::test]
async fn openai_batch_splits_over_the_request_cap_and_preserves_order() {
    let server = httpmock::MockServer::start_async().await;
    let inputs: Vec<String> = (0..300).map(|i| format!("wire input {i:03}")).collect();
    let borrowed: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let first_rows: Vec<(f32, usize)> = (0..256).map(|index| (0.5, index)).collect();
    let second_rows: Vec<(f32, usize)> = (0..44).map(|index| (1.5, index)).collect();
    let first = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_includes("wire input 255");
            then.status(200).body(openai_body(&first_rows));
        })
        .await;
    let second = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_includes("wire input 299");
            then.status(200).body(openai_body(&second_rows));
        })
        .await;
    let embedder = openai_at(&server);
    let embeddings = embedder.embed_texts(&borrowed).await.unwrap();
    first.assert_calls(1);
    second.assert_calls(1);
    assert_eq!(
        embeddings.len(),
        300,
        "every input embeds exactly once across the split"
    );
    let boundaries = [
        (0_usize, 0.5_f32, 0.0_f32),
        (255, 0.5, 255.0),
        (256, 1.5, 0.0),
        (299, 1.5, 43.0),
    ];
    for (position, marker, batch_local) in boundaries {
        let slice = embeddings[position].as_slice();
        assert_eq!(
            (slice.first().copied(), slice.get(1).copied()),
            (Some(marker), Some(batch_local)),
            "position {position} must carry batch-local index {batch_local}'s vector — \
             concatenation preserves input order across the request split"
        );
    }
}

#[tokio::test]
async fn dimension_mismatch_fails_loudly_naming_both_numbers() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .body("{\"data\":[{\"index\":0,\"embedding\":[1.0,2.0,3.0,4.0,5.0,6.0,7.0]}]}");
        })
        .await;
    let embedder = openai_at(&server);
    let error = embedder.embed_texts(&["one text"]).await.unwrap_err();
    assert!(
        matches!(error, loopctl::api::error::ApiError::Config(_)),
        "a swapped-dimension response is a config-validation failure: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("1536") && message.contains("7"),
        "the failure names the expected and received dimensions: {message}"
    );
}

#[tokio::test]
async fn an_oversized_input_is_rejected_before_the_wire() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(openai_body(&[(1.0, 0)]));
        })
        .await;
    let embedder = openai_at(&server);
    let oversized = "x".repeat(32_765);
    let error = embedder
        .embed_texts(&["small", oversized.as_str()])
        .await
        .unwrap_err();
    assert!(
        matches!(error, loopctl::api::error::ApiError::Config(_)),
        "the over-budget input is rejected as a config failure: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("input 1"),
        "the failure names the offending input's index: {message}"
    );
    assert_eq!(
        mock.calls(),
        0,
        "no HTTP request may leave the client for a guard-rejected input"
    );
}

#[tokio::test]
async fn an_empty_input_returns_empty_without_touching_the_wire() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(openai_body(&[]));
        })
        .await;
    let embeddings = openai_at(&server).embed_texts(&[]).await.unwrap();
    assert!(
        embeddings.is_empty(),
        "an empty batch is an empty answer, not an error"
    );
    assert_eq!(mock.calls(), 0, "the empty batch must not reach the wire");
}

#[tokio::test]
async fn the_normalized_builder_l2_normalizes_the_response() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(
                "{\"data\":[{\"index\":0,\"embedding\":[3.0,4.0,".to_string()
                    + &vec!["0.0"; 1534].join(",")
                    + "]}]}",
            );
        })
        .await;
    let embedder = OpenAiEmbedder::builder()
        .with_api_key("sk-wire-test")
        .with_base_url(server.base_url())
        .normalized()
        .build()
        .unwrap();
    let embeddings = embedder.embed_texts(&["normalize me"]).await.unwrap();
    let slice = embeddings.first().unwrap().as_slice();
    let norm = slice.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "the 3-4-0… vector must arrive unit-length, got norm {norm}"
    );
    assert!(
        (slice.first().copied().unwrap_or(0.0) - 0.6).abs() < 1e-6,
        "the components keep their direction: 3/5 leads, 4/5 follows"
    );
}

#[tokio::test]
async fn the_dimensions_parameter_rides_the_request_when_set() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_includes("\"dimensions\":256");
            then.status(200)
                .body(format!("{{\"data\":[{}]}}", openai_row_with(1.0, 0, 256)));
        })
        .await;
    let embedder = OpenAiEmbedder::builder()
        .with_api_key("sk-wire-test")
        .with_base_url(server.base_url())
        .with_dimensions(256)
        .build()
        .unwrap();
    assert_eq!(
        embedder.dim(),
        256,
        "a truncation fixes the embedder's reported dimension"
    );
    let embeddings = embedder.embed_texts(&["truncated"]).await.unwrap();
    mock.assert_calls(1);
    assert_eq!(
        embeddings.first().unwrap().dim(),
        256,
        "the wire answer's length must match the truncation the request carried"
    );
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn ollama_wire_shape_roundtrips_in_input_order() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/embed")
                .body_includes("\"model\":\"wire-test-embed\"")
                .body_includes("first text");
            then.status(200).body(
                "{\"model\":\"wire-test-embed\",\"embeddings\":[[1.0,0.0,0.0],\
                 [0.0,2.0,0.0],[0.0,0.0,3.0]]}",
            );
        })
        .await;
    let embedder = ollama_at(&server);
    let embeddings = embedder
        .embed_texts(&["first text", "second text", "third text"])
        .await
        .unwrap();
    mock.assert_calls(1);
    assert_eq!(
        embeddings.len(),
        3,
        "one vector per input, in input order — /api/embed has no index field"
    );
    let expected = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    for (position, want) in expected.iter().enumerate() {
        assert_eq!(
            embeddings[position].as_slice(),
            want,
            "position {position} carries input {position}'s vector verbatim"
        );
    }
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn ollama_dimension_mismatch_fails_loudly_naming_both_numbers() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200)
                .body("{\"model\":\"wire-test-embed\",\"embeddings\":[[1.0,2.0]]}");
        })
        .await;
    let error = ollama_at(&server)
        .embed_texts(&["one text"])
        .await
        .unwrap_err();
    assert!(
        matches!(error, loopctl::api::error::ApiError::Config(_)),
        "a swapped-dimension response is a config-validation failure: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("3") && message.contains("2"),
        "the failure names the expected and received dimensions: {message}"
    );
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn ollama_transport_failures_carry_the_server_hint() {
    let embedder = OllamaEmbedder::builder()
        .with_model("wire-test-embed")
        .with_dim(3)
        .with_base_url("http://127.0.0.1:1")
        .build()
        .unwrap();
    let error = embedder.embed_texts(&["any text"]).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("is `ollama serve` running?"),
        "a transport failure must carry the actionable hint: {message}"
    );
}

/// A syntactically valid Ollama-shaped body padded past a small budget.
///
/// The filler rides inside the echoed model string, so the body parses
/// cleanly for an embedder whose budget still admits it — a refusal, not
/// a parse error, is the only way this body can fail one whose budget it
/// exceeds.
#[cfg(feature = "ollama")]
fn padded_ollama_body(total_bytes: usize) -> String {
    let wanted = total_bytes.saturating_sub(80);
    let filler = "x".repeat(wanted);
    format!("{{\"model\":\"{filler}\",\"embeddings\":[[1.0,0.0,0.0]]}}")
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn an_over_budget_content_length_is_refused_before_reading() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200)
                .body(padded_ollama_body(1024 * 1024 + 64 * 1024));
        })
        .await;
    let error = ollama_at(&server)
        .embed_texts(&["one text"])
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("bounded-body budget"),
        "a declared body over the dimension-derived budget must be refused \
         before it is read: {message}"
    );
}

/// Serve one chunked HTTP response carrying `body`, with no content-length.
///
/// httpmock always declares a truthful content-length, so the mid-stream
/// cap is driven by a hand-rolled TCP server speaking `Transfer-Encoding:
/// chunked` — the shape a hostile or misbehaving server sends to dodge a
/// header-prefetched budget check. Writes after the client aborts
/// mid-stream fail with a broken pipe and are ignored.
async fn serve_chunked(body: String) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut scratch = [0u8; 4096];
        let _ = sock.read(&mut scratch).await;
        let _ = sock
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            .await;
        for piece in body.as_bytes().chunks(64 * 1024) {
            let _ = sock
                .write_all(format!("{:x}\r\n", piece.len()).as_bytes())
                .await;
            let _ = sock.write_all(piece).await;
            let _ = sock.write_all(b"\r\n").await;
        }
        let _ = sock.write_all(b"0\r\n\r\n").await;
    });
    format!("http://{addr}")
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn a_chunked_stream_past_the_budget_is_refused_mid_stream() {
    let base = serve_chunked(padded_ollama_body(1024 * 1024 + 64 * 1024)).await;
    let embedder = OllamaEmbedder::builder()
        .with_model("wire-test-embed")
        .with_dim(3)
        .with_base_url(base)
        .build()
        .unwrap();
    let error = embedder.embed_texts(&["one text"]).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("bounded-body budget"),
        "a headerless chunked stream must hit the mid-stream cap, not parse \
         or pass: {message}"
    );
}

#[tokio::test]
async fn openai_reassembly_refuses_duplicate_indices() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(openai_body(&[(1.0, 0), (2.0, 0)]));
        })
        .await;
    let error = openai_at(&server)
        .embed_texts(&["first input", "second input"])
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("refuses to guess"),
        "two rows answering the same index must fail reassembly instead of \
         double-filling one slot: {message}"
    );
}

#[tokio::test]
async fn openai_reassembly_refuses_a_short_or_out_of_range_data_set() {
    let short_server = httpmock::MockServer::start_async().await;
    short_server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(openai_body(&[(1.0, 0)]));
        })
        .await;
    let short = openai_at(&short_server)
        .embed_texts(&["first input", "second input"])
        .await
        .unwrap_err();
    assert!(
        short.to_string().contains("refuses to guess"),
        "a data set shorter than the input list must fail reassembly: {short}"
    );

    let stray_server = httpmock::MockServer::start_async().await;
    stray_server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).body(openai_body(&[(1.0, 7)]));
        })
        .await;
    let stray = openai_at(&stray_server)
        .embed_texts(&["only input"])
        .await
        .unwrap_err();
    assert!(
        stray.to_string().contains("refuses to guess"),
        "a row answering an index the caller never sent must fail reassembly: {stray}"
    );
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn ollama_count_mismatch_fails_naming_both_numbers() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200).body(
                "{\"model\":\"wire-test-embed\",\"embeddings\":[[1.0,0.0,0.0],[0.0,2.0,0.0]]}",
            );
        })
        .await;
    let error = ollama_at(&server)
        .embed_texts(&["first text", "second text", "third text"])
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("2 embeddings") && message.contains("3 inputs"),
        "the count mismatch must name both numbers: {message}"
    );
}

#[tokio::test]
async fn dimensions_stay_absent_from_the_request_when_unset() {
    let server = httpmock::MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_excludes("dimensions");
            then.status(200).body(openai_body(&[(1.0, 0)]));
        })
        .await;
    let embeddings = openai_at(&server)
        .embed_texts(&["no truncation requested"])
        .await
        .unwrap();
    mock.assert_calls(1);
    assert_eq!(
        embeddings.len(),
        1,
        "the unset truncation still embeds — the pin is the request's absent field"
    );
}

#[tokio::test]
async fn a_three_large_geometry_round_trips_at_full_length() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .body(format!("{{\"data\":[{}]}}", openai_row_with(1.0, 0, 3072)));
        })
        .await;
    let embedder = OpenAiEmbedder::builder()
        .with_api_key("sk-wire-test")
        .with_base_url(server.base_url())
        .with_model("text-embedding-3-large")
        .with_dimensions(3072)
        .build()
        .unwrap();
    assert_eq!(
        embedder.dim(),
        3072,
        "a 3-large pairing at the parameter's maximum must build and report it"
    );
    let embeddings = embedder.embed_texts(&["full length"]).await.unwrap();
    assert_eq!(
        embeddings.first().unwrap().dim(),
        3072,
        "the full-length answer passes the dimension check untruncated"
    );
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn normalization_survives_huge_but_finite_components() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200)
                .body("{\"model\":\"wire-test-embed\",\"embeddings\":[[3e20,4e20,0.0]]}");
        })
        .await;
    let embedder = OllamaEmbedder::builder()
        .with_model("wire-test-embed")
        .with_dim(3)
        .with_base_url(server.base_url())
        .normalized()
        .build()
        .unwrap();
    let embeddings = embedder.embed_texts(&["huge but finite"]).await.unwrap();
    let slice = embeddings.first().unwrap().as_slice();
    assert!(
        (slice.first().copied().unwrap_or(0.0) - 0.6).abs() < 1e-6
            && (slice.get(1).copied().unwrap_or(0.0) - 0.8).abs() < 1e-6,
        "max-abs rescaling must normalize huge-but-finite components exactly, \
         got {slice:?}"
    );
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn an_overflowing_exponent_fails_the_embed_at_the_parse() {
    let server = httpmock::MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST).path("/api/embed");
            then.status(200)
                .body("{\"model\":\"wire-test-embed\",\"embeddings\":[[1.0,1e999,0.0]]}");
        })
        .await;
    let embedder = OllamaEmbedder::builder()
        .with_model("wire-test-embed")
        .with_dim(3)
        .with_base_url(server.base_url())
        .normalized()
        .build()
        .unwrap();
    let error = embedder
        .embed_texts(&["hostile exponent"])
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("out of range"),
        "an exponent too large for f32 must fail loudly at the parse — JSON \
         cannot carry it, so it never reaches normalization as a quiet value: {message}"
    );
}
