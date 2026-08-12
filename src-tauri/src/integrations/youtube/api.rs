use serde::Deserialize;

use std::path::Path;

use tokio_util::sync::CancellationToken;

use super::{
    OAuthTokens, YouTubeAccount, YouTubeError, YouTubePlaylist, YouTubePrivacyStatus,
};

pub(crate) const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
// Upload is required for videos.insert; readonly is required for the
// channels.list(mine=true) call that identifies the connected channel.
pub(crate) const YOUTUBE_SCOPE: &str = concat!(
    "https://www.googleapis.com/auth/youtube.upload ",
    "https://www.googleapis.com/auth/youtube.readonly ",
    "https://www.googleapis.com/auth/youtube.force-ssl"
);

#[derive(Debug, Clone)]
pub(crate) struct OAuthConfig {
    pub client_id: String,
    pub client_secret: Option<String>,
}

impl OAuthConfig {
    pub fn load() -> Option<Self> {
        let client_id = std::env::var("LMT_YOUTUBE_CLIENT_ID")
            .ok()
            .or_else(|| option_env!("LMT_YOUTUBE_CLIENT_ID").map(str::to_owned))
            .filter(|value| !value.trim().is_empty())?;
        let client_secret = std::env::var("LMT_YOUTUBE_CLIENT_SECRET")
            .ok()
            .filter(|value| !value.trim().is_empty());
        Some(Self {
            client_id,
            client_secret,
        })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default = "default_expiry")]
    pub expires_in: u64,
}

fn default_expiry() -> u64 {
    3600
}

pub(crate) async fn refresh_access_token(
    client: &reqwest::Client,
    config: &OAuthConfig,
    refresh_token: &str,
) -> Result<OAuthTokens, YouTubeError> {
    let mut form = vec![
        ("client_id", config.client_id.as_str()),
        ("refresh_token", refresh_token),
        ("grant_type", "refresh_token"),
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
    if response.status() == reqwest::StatusCode::BAD_REQUEST {
        return Err(YouTubeError::AuthenticationRequired);
    }
    let response = require_success(response).await?;
    let token: TokenResponse = response
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid token response: {error}")))?;
    Ok(OAuthTokens {
        access_token: token.access_token,
        refresh_token: refresh_token.to_string(),
        expires_in: token.expires_in,
    })
}

pub(crate) async fn fetch_channel(
    client: &reqwest::Client,
    access_token: &str,
    account_id: String,
    connected_at: String,
) -> Result<YouTubeAccount, YouTubeError> {
    let response = client
        .get("https://www.googleapis.com/youtube/v3/channels")
        .bearer_auth(access_token)
        .query(&[
            ("part", "snippet"),
            ("mine", "true"),
            ("maxResults", "1"),
            (
                "fields",
                "items(id,snippet(title,thumbnails/default/url))",
            ),
        ])
        .send()
        .await
        .map_err(network_error)?;
    let response = require_success(response).await?;
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid channel response: {error}")))?;
    let channel = payload
        .get("items")
        .and_then(|items| items.as_array())
        .and_then(|items| items.first())
        .ok_or_else(|| YouTubeError::Api("No YouTube channel is available for this account.".into()))?;
    let thumbnail_url = channel
        .pointer("/snippet/thumbnails/default/url")
        .and_then(|value| value.as_str())
        .map(str::to_owned);
    Ok(YouTubeAccount {
        id: account_id,
        channel_id: channel.get("id").and_then(|value| value.as_str()).map(str::to_owned),
        channel_title: channel
            .pointer("/snippet/title")
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        thumbnail_url,
        connected_at,
    })
}

pub(crate) async fn fetch_playlists(
    client: &reqwest::Client,
    access_token: &str,
    cancel: &CancellationToken,
) -> Result<Vec<YouTubePlaylist>, YouTubeError> {
    let mut playlists = Vec::new();
    let mut page_token: Option<String> = None;
    loop {
        let mut request = client
            .get("https://www.googleapis.com/youtube/v3/playlists")
            .bearer_auth(access_token)
            .query(&[
                ("part", "snippet"),
                ("mine", "true"),
                ("maxResults", "50"),
                ("fields", "nextPageToken,items(id,snippet/title)"),
            ]);
        if let Some(token) = page_token.as_deref() {
            request = request.query(&[("pageToken", token)]);
        }
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
            response = request.send() => response.map_err(network_error)?,
        };
        let payload: serde_json::Value = require_success(response)
            .await?
            .json()
            .await
            .map_err(|error| YouTubeError::Api(format!("invalid playlist response: {error}")))?;
        if let Some(items) = payload.get("items").and_then(|value| value.as_array()) {
            playlists.extend(items.iter().filter_map(|item| {
                Some(YouTubePlaylist {
                    id: item.get("id")?.as_str()?.to_string(),
                    name: item.pointer("/snippet/title")?.as_str()?.to_string(),
                })
            }));
        }
        page_token = payload
            .get("nextPageToken")
            .and_then(|value| value.as_str())
            .map(str::to_owned);
        if page_token.is_none() {
            break;
        }
    }
    playlists.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(playlists)
}

pub(crate) async fn fetch_video_privacy(
    client: &reqwest::Client,
    access_token: &str,
    video_id: &str,
    cancel: &CancellationToken,
) -> Result<YouTubePrivacyStatus, YouTubeError> {
    let request = client
        .get("https://www.googleapis.com/youtube/v3/videos")
        .bearer_auth(access_token)
        .query(&[
            ("part", "status"),
            ("id", video_id),
            ("fields", "items(status/privacyStatus)"),
        ]);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    let payload: serde_json::Value = require_success(response)
        .await?
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid video status response: {error}")))?;
    match payload
        .pointer("/items/0/status/privacyStatus")
        .and_then(|value| value.as_str())
    {
        Some("private") => Ok(YouTubePrivacyStatus::Private),
        Some("unlisted") => Ok(YouTubePrivacyStatus::Unlisted),
        Some("public") => Ok(YouTubePrivacyStatus::Public),
        Some(other) => Err(YouTubeError::Api(format!(
            "YouTube returned an unsupported privacy status: {other}."
        ))),
        None => Err(YouTubeError::Network(
            "The uploaded video's status is not available yet.".into(),
        )),
    }
}

pub(crate) async fn insert_playlist_item(
    client: &reqwest::Client,
    access_token: &str,
    playlist_id: &str,
    video_id: &str,
    cancel: &CancellationToken,
) -> Result<(), YouTubeError> {
    let check = client
        .get("https://www.googleapis.com/youtube/v3/playlistItems")
        .bearer_auth(access_token)
        .query(&[
            ("part", "id"),
            ("playlistId", playlist_id),
            ("videoId", video_id),
            ("maxResults", "1"),
            ("fields", "items(id)"),
        ]);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = check.send() => response.map_err(network_error)?,
    };
    let payload: serde_json::Value = require_success(response)
        .await?
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid playlist item response: {error}")))?;
    if payload
        .get("items")
        .and_then(|items| items.as_array())
        .map(|items| !items.is_empty())
        .unwrap_or(false)
    {
        return Ok(());
    }
    let request = client
        .post("https://www.googleapis.com/youtube/v3/playlistItems")
        .bearer_auth(access_token)
        .query(&[("part", "snippet")])
        .json(&serde_json::json!({
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": { "kind": "youtube#video", "videoId": video_id }
            }
        }));
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    require_success(response).await?;
    Ok(())
}

pub(crate) async fn set_thumbnail(
    client: &reqwest::Client,
    access_token: &str,
    video_id: &str,
    path: &Path,
    cancel: &CancellationToken,
) -> Result<(), YouTubeError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    let mime = match path.extension().and_then(|value| value.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("png") => "image/png",
        _ => "image/jpeg",
    };
    let request = client
        .post("https://www.googleapis.com/upload/youtube/v3/thumbnails/set")
        .bearer_auth(access_token)
        .query(&[("videoId", video_id), ("uploadType", "media")])
        .header(reqwest::header::CONTENT_TYPE, mime)
        .body(bytes);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    require_success(response).await?;
    Ok(())
}

pub(crate) async fn insert_caption(
    client: &reqwest::Client,
    access_token: &str,
    video_id: &str,
    language: &str,
    name: &str,
    srt_path: &Path,
    cancel: &CancellationToken,
) -> Result<(), YouTubeError> {
    let check = client
        .get("https://www.googleapis.com/youtube/v3/captions")
        .bearer_auth(access_token)
        .query(&[
            ("part", "snippet"),
            ("videoId", video_id),
            ("maxResults", "50"),
            ("fields", "items(id,snippet(language,name))"),
        ]);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = check.send() => response.map_err(network_error)?,
    };
    let payload: serde_json::Value = require_success(response)
        .await?
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid caption list response: {error}")))?;
    let already_exists = payload
        .get("items")
        .and_then(|items| items.as_array())
        .map(|items| {
            items.iter().any(|item| {
                item.pointer("/snippet/language").and_then(|value| value.as_str())
                    == Some(language)
                    && item.pointer("/snippet/name").and_then(|value| value.as_str()) == Some(name)
            })
        })
        .unwrap_or(false);
    if already_exists {
        return Ok(());
    }
    let srt = tokio::fs::read(srt_path)
        .await
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    let boundary = format!("lmt-{}", uuid::Uuid::new_v4().simple());
    let metadata = serde_json::to_vec(&serde_json::json!({
        "snippet": {
            "videoId": video_id,
            "language": language,
            "name": name,
            "isDraft": false
        }
    }))
    .map_err(|error| YouTubeError::SubtitlePublish(error.to_string()))?;
    let mut body = Vec::with_capacity(metadata.len() + srt.len() + 512);
    body.extend_from_slice(format!("--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n").as_bytes());
    body.extend_from_slice(&metadata);
    body.extend_from_slice(format!("\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes());
    body.extend_from_slice(&srt);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let request = client
        .post("https://www.googleapis.com/upload/youtube/v3/captions")
        .bearer_auth(access_token)
        .query(&[("uploadType", "multipart"), ("part", "snippet")])
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/related; boundary={boundary}"),
        )
        .body(body);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    require_success(response).await?;
    Ok(())
}

pub(crate) fn network_error(error: reqwest::Error) -> YouTubeError {
    YouTubeError::Network(if error.is_timeout() {
        "The request timed out.".into()
    } else if error.is_connect() {
        "Could not connect to YouTube.".into()
    } else {
        "The connection was interrupted.".into()
    })
}

pub(crate) async fn require_success(
    response: reqwest::Response,
) -> Result<reqwest::Response, YouTubeError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(YouTubeError::AuthenticationRequired);
    }
    if status == reqwest::StatusCode::FORBIDDEN
        && (body.contains("quotaExceeded")
            || body.contains("dailyLimitExceeded")
            || body.contains("uploadLimitExceeded"))
    {
        return Err(YouTubeError::QuotaExceeded);
    }
    if status == reqwest::StatusCode::FORBIDDEN && body.contains("insufficientPermissions") {
        return Err(YouTubeError::PermissionDenied);
    }
    if status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || body.contains("rateLimitExceeded")
        || body.contains("userRateLimitExceeded")
        || body.contains("backendError")
    {
        return Err(YouTubeError::Network(format!(
            "YouTube is temporarily unavailable ({status})."
        )));
    }
    Err(YouTubeError::Api(format!(
        "YouTube rejected the request ({status})."
    )))
}
