use std::time::Duration;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;
use uuid::Uuid;

use super::api::{
    network_error, require_success, OAuthConfig, TokenResponse, TOKEN_ENDPOINT, YOUTUBE_SCOPE,
};
use super::{OAuthTokens, YouTubeError};

const AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(5 * 60);

pub(crate) async fn connect(
    client: &reqwest::Client,
    config: &OAuthConfig,
) -> Result<OAuthTokens, YouTubeError> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|error| YouTubeError::OAuthFailed(error.to_string()))?;
    let port = listener
        .local_addr()
        .map_err(|error| YouTubeError::OAuthFailed(error.to_string()))?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{port}/oauth/callback");
    let state = Uuid::new_v4().simple().to_string();
    let verifier = format!(
        "{}{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    );
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

    let mut auth_url =
        Url::parse(AUTH_ENDPOINT).map_err(|error| YouTubeError::OAuthFailed(error.to_string()))?;
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", &config.client_id)
        .append_pair("redirect_uri", &redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", YOUTUBE_SCOPE)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent")
        .append_pair("state", &state)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256");

    open::that(auth_url.as_str())
        .map_err(|error| YouTubeError::OAuthFailed(format!("could not open browser: {error}")))?;

    let callback = tokio::time::timeout(
        CALLBACK_TIMEOUT,
        receive_callback(listener, &state),
    )
        .await
        .map_err(|_| YouTubeError::OAuthCancelled)??;
    if callback.state.as_deref() != Some(state.as_str()) {
        return Err(YouTubeError::OAuthFailed(
            "OAuth state validation failed.".into(),
        ));
    }
    if let Some(error) = callback.error {
        return Err(if error == "access_denied" {
            YouTubeError::PermissionDenied
        } else {
            YouTubeError::OAuthFailed("Google did not grant access.".into())
        });
    }
    let code = callback
        .code
        .ok_or_else(|| YouTubeError::OAuthFailed("No authorization code was returned.".into()))?;

    let mut form = vec![
        ("client_id", config.client_id.as_str()),
        ("code", code.as_str()),
        ("code_verifier", verifier.as_str()),
        ("redirect_uri", redirect_uri.as_str()),
        ("grant_type", "authorization_code"),
    ];
    if let Some(secret) = config.client_secret.as_deref() {
        form.push(("client_secret", secret));
    }
    let response = client
        .post(TOKEN_ENDPOINT)
        .form(&form)
        .send()
        .await
        .map_err(network_error)?;
    let response = require_success(response).await?;
    let token: TokenResponse = response
        .json()
        .await
        .map_err(|error| YouTubeError::OAuthFailed(format!("invalid token response: {error}")))?;
    let refresh_token = token.refresh_token.ok_or_else(|| {
        YouTubeError::OAuthFailed(
            "Google did not return a refresh token. Revoke the app grant and connect again.".into(),
        )
    })?;
    Ok(OAuthTokens {
        access_token: token.access_token,
        refresh_token,
        expires_in: token.expires_in,
    })
}

struct Callback {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

async fn receive_callback(
    listener: TcpListener,
    expected_state: &str,
) -> Result<Callback, YouTubeError> {
    loop {
        let (mut socket, _) = listener
            .accept()
            .await
            .map_err(|error| YouTubeError::OAuthFailed(error.to_string()))?;
        let mut buffer = vec![0u8; 16 * 1024];
        let Ok(read) = socket.read(&mut buffer).await else {
            continue;
        };
        let request = String::from_utf8_lossy(&buffer[..read]);
        let Some(target) = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
        else {
            write_browser_response(&mut socket, 404, "Not found.").await;
            continue;
        };
        let Ok(callback_url) = Url::parse(&format!("http://127.0.0.1{target}")) else {
            write_browser_response(&mut socket, 404, "Not found.").await;
            continue;
        };
        let mut callback = Callback {
            code: None,
            state: None,
            error: None,
        };
        for (key, value) in callback_url.query_pairs() {
            match key.as_ref() {
                "code" => callback.code = Some(value.into_owned()),
                "state" => callback.state = Some(value.into_owned()),
                "error" => callback.error = Some(value.into_owned()),
                _ => {}
            }
        }
        if callback_url.path() != "/oauth/callback"
            || callback.state.as_deref() != Some(expected_state)
        {
            write_browser_response(&mut socket, 404, "Not found.").await;
            continue;
        }
        let successful = callback.error.is_none() && callback.code.is_some();
        let body = if successful {
            "YouTube connected. You can close this tab and return to Local Movie Translator."
        } else {
            "YouTube connection was not completed. You can close this tab and return to the app."
        };
        write_browser_response(&mut socket, 200, body).await;
        return Ok(callback);
    }
}

async fn write_browser_response(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    body: &str,
) {
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = socket.write_all(response.as_bytes()).await;
}
