use std::path::Path;
use std::time::SystemTime;

use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio_util::sync::CancellationToken;

use super::api::{network_error, require_success};
use super::{YouTubeError, YouTubeVideoMetadata};

const UPLOAD_ENDPOINT: &str =
    "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status";
const CHUNK_SIZE: usize = 8 * 1024 * 1024;

pub(crate) async fn create_session(
    client: &reqwest::Client,
    access_token: &str,
    metadata: &YouTubeVideoMetadata,
    file_size: u64,
    mime_type: &str,
    cancel: &CancellationToken,
) -> Result<String, YouTubeError> {
    let mut snippet = serde_json::json!({
        "title": metadata.title,
        "description": metadata.description,
        "tags": metadata.tags,
        "categoryId": metadata.category_id,
    });
    if let Some(language) = metadata.default_language.as_deref() {
        snippet["defaultLanguage"] = serde_json::Value::String(language.to_string());
    }
    let body = serde_json::json!({
        "snippet": snippet,
        "status": {
            "privacyStatus": metadata.privacy_status.as_str(),
        }
    });
    let request = client
        .post(UPLOAD_ENDPOINT)
        .bearer_auth(access_token)
        .header("X-Upload-Content-Length", file_size)
        .header("X-Upload-Content-Type", mime_type)
        .json(&body);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    let response = require_success(response).await?;
    response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .ok_or_else(|| YouTubeError::Api("YouTube did not create an upload session.".into()))
}

pub(crate) async fn query_offset(
    client: &reqwest::Client,
    access_token: &str,
    session_uri: &str,
    total: u64,
    cancel: &CancellationToken,
) -> Result<SessionPosition, YouTubeError> {
    let request = client
        .put(session_uri)
        .bearer_auth(access_token)
        .header(reqwest::header::CONTENT_LENGTH, 0)
        .header(reqwest::header::CONTENT_RANGE, format!("bytes */{total}"));
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
        response = request.send() => response.map_err(network_error)?,
    };
    if response.status().is_success() {
        return Ok(SessionPosition::Completed(parse_video_id(response).await?));
    }
    if response.status().as_u16() == 308 {
        return Ok(SessionPosition::Offset(next_offset(&response).unwrap_or(0)));
    }
    let _ = require_success(response).await?;
    Ok(SessionPosition::Offset(0))
}

pub(crate) enum SessionPosition {
    Offset(u64),
    Completed(String),
}

pub(crate) async fn upload_chunks<F>(
    client: &reqwest::Client,
    access_token: &str,
    session_uri: &str,
    file_path: &Path,
    total: u64,
    start_offset: u64,
    mime_type: &str,
    expected_modified: SystemTime,
    cancel: CancellationToken,
    mut on_progress: F,
) -> Result<String, YouTubeError>
where
    F: FnMut(u64),
{
    let mut file = tokio::fs::File::open(file_path)
        .await
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    file.seek(std::io::SeekFrom::Start(start_offset))
        .await
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    let mut offset = start_offset;
    let mut buffer = vec![0u8; CHUNK_SIZE];

    while offset < total {
        if cancel.is_cancelled() {
            return Err(YouTubeError::Cancelled);
        }
        validate_file_identity(file_path, total, expected_modified).await?;
        let wanted = ((total - offset) as usize).min(CHUNK_SIZE);
        let mut read = 0usize;
        while read < wanted {
            let count = file
                .read(&mut buffer[read..wanted])
                .await
                .map_err(|error| YouTubeError::Io(error.to_string()))?;
            if count == 0 {
                return Err(YouTubeError::InvalidVideo(
                    "The file changed or ended during upload.".into(),
                ));
            }
            read += count;
        }
        // Re-check after reading and before submitting the chunk. This
        // prevents bytes read during an in-place render rewrite from
        // being mixed into the existing resumable session.
        validate_file_identity(file_path, total, expected_modified).await?;
        let end = offset + read as u64 - 1;
        let request = client
            .put(session_uri)
            .bearer_auth(access_token)
            .header(reqwest::header::CONTENT_LENGTH, read)
            .header(reqwest::header::CONTENT_TYPE, mime_type)
            .header(
                reqwest::header::CONTENT_RANGE,
                format!("bytes {offset}-{end}/{total}"),
            )
            .body(buffer[..read].to_vec());
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
            response = request.send() => response.map_err(network_error)?,
        };

        if response.status().as_u16() == 308 {
            // Google's Range header is authoritative. It can report less
            // than the submitted chunk after a partial network write.
            offset = next_offset(&response).unwrap_or(0);
            file.seek(std::io::SeekFrom::Start(offset))
                .await
                .map_err(|error| YouTubeError::Io(error.to_string()))?;
            on_progress(offset);
            continue;
        }
        if response.status().is_success() {
            on_progress(total);
            return parse_video_id(response).await;
        }
        let _ = require_success(response).await?;
    }
    Err(YouTubeError::Api(
        "YouTube ended the upload without returning a video ID.".into(),
    ))
}

async fn validate_file_identity(
    path: &Path,
    expected_len: u64,
    expected_modified: SystemTime,
) -> Result<(), YouTubeError> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    let modified = metadata
        .modified()
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    if metadata.len() != expected_len || modified != expected_modified {
        return Err(YouTubeError::RenderChanged);
    }
    Ok(())
}

fn next_offset(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit('-').next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|last| last.saturating_add(1))
}

async fn parse_video_id(response: reqwest::Response) -> Result<String, YouTubeError> {
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|error| YouTubeError::Api(format!("invalid upload response: {error}")))?;
    payload
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .ok_or_else(|| YouTubeError::Api("YouTube did not return a video ID.".into()))
}

pub(crate) fn mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "avi" => "video/x-msvideo",
        _ => "video/mp4",
    }
}
