//! The TLS transport: the remote endpoint's encrypted carrier.
//!
//! One entry — [`connect`] — loads the trust material (the endpoint's
//! certificate authority, plus the client identity when the endpoint
//! carries one), performs the TCP connect, the TLS handshake against
//! that material, and hands back the encrypted stream as a boxed
//! [`Wire`](crate::http::Wire) the shared reader consumes. Everything
//! above the transport — the handshake document, the verbs, the event
//! stream — is identical over TLS; only the bytes are ciphertext.
//!
//! *Requires the `tls` feature.*

use std::path::Path;
use std::sync::Arc;

use tokio::net::TcpStream;

use crate::ClientError;
use crate::http::Wire;
use crate::http::io_error;

/// Connect one TLS stream to the daemon at `host:port`.
///
/// Verification uses `ca_pem` when a path is given — the shape a
/// self-hosted daemon's own certificate authority takes — and the
/// public `WebPKI` roots otherwise, for a daemon named in a
/// publicly-trusted certificate. The host doubles as the TLS server
/// name, so the daemon's certificate must cover it. When
/// `client_identity` is given, the client presents that certificate
/// chain and key for a daemon that authenticates clients by
/// certificate.
///
/// # Errors
///
/// [`ClientError::SocketAbsent`] when the TCP connect is refused;
/// [`ClientError::Transport`] for every other IO, resolution, or
/// TLS failure — including a certificate that does not verify, which
/// never degrades to an unverified connection, and an identity file
/// that cannot be read or parsed, which refuses before the connection
/// opens.
pub(crate) async fn connect(
    host: &str,
    port: u16,
    ca_pem: Option<&Path>,
    client_identity: Option<&Path>,
) -> Result<Box<dyn Wire>, ClientError> {
    let ca_bytes = match ca_pem {
        Some(path) => Some(read_pem_file(path, "the CA file").await?),
        None => None,
    };
    let identity_bytes = match client_identity {
        Some(path) => Some(read_pem_file(path, "the client identity").await?),
        None => None,
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut roots = rustls::RootCertStore::empty();
    match ca_bytes.as_deref() {
        Some(pem) => load_ca(pem, &mut roots)?,
        None => {
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        }
    }
    let identity = identity_bytes.as_deref().map(load_identity).transpose()?;
    let builder = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| ClientError::Transport(error.to_string()))?
        .with_root_certificates(roots);
    let config = match identity {
        Some((certificates, key)) => builder
            .with_client_auth_cert(certificates, key)
            .map_err(|error| ClientError::Transport(error.to_string()))?,
        None => builder.with_no_client_auth(),
    };
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let name = rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|error| ClientError::Transport(error.to_string()))?;
    let stream = TcpStream::connect((host, port))
        .await
        .map_err(|error| io_error(&error))?;
    let tls = connector
        .connect(name, stream)
        .await
        .map_err(|error| ClientError::Transport(error.to_string()))?;
    Ok(Box::new(tls))
}

/// Read one PEM file off the runtime worker.
///
/// A stalled filesystem must not block the worker — or, on a
/// current-thread runtime, everything else with it — so the read
/// rides `spawn_blocking` and the exchange deadline can still expire
/// while it waits.
///
/// # Errors
///
/// [`ClientError::Transport`] when the read fails or the blocking
/// task is lost; the diagnosis names what could not be read.
async fn read_pem_file(path: &Path, what: &str) -> Result<Vec<u8>, ClientError> {
    let path = path.to_path_buf();
    let what = what.to_string();
    tokio::task::spawn_blocking(move || std::fs::read(&path))
        .await
        .map_err(|error| ClientError::Transport(format!("{what} could not be read: {error}")))?
        .map_err(|error| ClientError::Transport(format!("{what} could not be read: {error}")))
}

/// Load one certificate-authority PEM into `roots`.
///
/// The bytes a self-hosted daemon ships for its own certificate —
/// read off the worker before this parse — each certificate in them
/// that parses is added as trusted, an entry that does not parse is
/// skipped, and bytes from which none parses refuse.
///
/// # Errors
///
/// [`ClientError::Transport`] when the bytes hold no certificates,
/// or none of them parse.
fn load_ca(pem: &[u8], roots: &mut rustls::RootCertStore) -> Result<(), ClientError> {
    let mut cursor = pem;
    let certificates = rustls_pemfile::certs(&mut cursor)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ClientError::Transport(format!("the CA file is not PEM: {error}")))?;
    if certificates.is_empty() {
        return Err(ClientError::Transport(
            "the CA file carried no certificates".into(),
        ));
    }
    let (added, _) = roots.add_parsable_certificates(certificates);
    if added == 0 {
        return Err(ClientError::Transport(
            "the CA file held no parsable certificate".into(),
        ));
    }
    Ok(())
}

/// Load one client-identity PEM: its certificate chain and key.
///
/// The bytes a mutual-TLS client presents — read off the worker
/// before this parse — every certificate they carry is offered as the
/// chain in file order, and the first private key found signs the
/// handshake. The refusal contract mirrors [`load_ca`]'s, at the
/// same fail-before-wire place.
///
/// # Errors
///
/// [`ClientError::Transport`] when the bytes are not PEM, hold no
/// certificates, or hold no private key.
fn load_identity(
    pem: &[u8],
) -> Result<
    (
        Vec<rustls::pki_types::CertificateDer<'static>>,
        rustls::pki_types::PrivateKeyDer<'static>,
    ),
    ClientError,
> {
    let mut cursor = pem;
    let certificates = rustls_pemfile::certs(&mut cursor)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            ClientError::Transport(format!("the client identity is not PEM: {error}"))
        })?;
    if certificates.is_empty() {
        return Err(ClientError::Transport(
            "the client identity carried no certificates".into(),
        ));
    }
    let mut cursor = pem;
    let key = rustls_pemfile::private_key(&mut cursor)
        .map_err(|error| {
            ClientError::Transport(format!("the client identity is not PEM: {error}"))
        })?
        .ok_or_else(|| {
            ClientError::Transport("the client identity carried no private key".into())
        })?;
    Ok((certificates, key))
}
