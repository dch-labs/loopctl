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
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut roots = rustls::RootCertStore::empty();
    match ca_pem {
        Some(path) => load_ca(path, &mut roots)?,
        None => {
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        }
    }
    let identity = client_identity.map(load_identity).transpose()?;
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

/// Load one certificate-authority PEM file into `roots`.
///
/// The file a self-hosted daemon ships for its own certificate —
/// each certificate in it that parses is added as trusted, an
/// entry that does not parse is skipped, and a file from which
/// none parses refuses.
///
/// # Errors
///
/// [`ClientError::Transport`] when the file cannot be read, holds
/// no certificates, or none of them parse.
fn load_ca(path: &Path, roots: &mut rustls::RootCertStore) -> Result<(), ClientError> {
    let pem = std::fs::read(path).map_err(|error| {
        ClientError::Transport(format!("the CA file could not be read: {error}"))
    })?;
    let mut cursor = pem.as_slice();
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

/// Load one client-identity PEM file: its certificate chain and key.
///
/// The file a mutual-TLS client presents — every certificate it
/// carries is offered as the chain in file order, and the first
/// private key found signs the handshake. The refusal contract
/// mirrors [`load_ca`]'s, at the same fail-before-wire place.
///
/// # Errors
///
/// [`ClientError::Transport`] when the file cannot be read, is not
/// PEM, holds no certificates, or holds no private key.
fn load_identity(
    path: &Path,
) -> Result<
    (
        Vec<rustls::pki_types::CertificateDer<'static>>,
        rustls::pki_types::PrivateKeyDer<'static>,
    ),
    ClientError,
> {
    let pem = std::fs::read(path).map_err(|error| {
        ClientError::Transport(format!("the client identity could not be read: {error}"))
    })?;
    let mut cursor = pem.as_slice();
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
    let mut cursor = pem.as_slice();
    let key = rustls_pemfile::private_key(&mut cursor)
        .map_err(|error| {
            ClientError::Transport(format!("the client identity is not PEM: {error}"))
        })?
        .ok_or_else(|| {
            ClientError::Transport("the client identity carried no private key".into())
        })?;
    Ok((certificates, key))
}
