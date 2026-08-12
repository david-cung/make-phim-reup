use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeAccount {
    pub id: String,
    pub channel_id: Option<String>,
    pub channel_title: Option<String>,
    pub thumbnail_url: Option<String>,
    pub connected_at: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum YouTubeAccountStatus {
    Connected,
    Disconnected,
    Expired,
    AuthenticationRequired,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeConnectionState {
    pub configured: bool,
    pub status: YouTubeAccountStatus,
    pub account: Option<YouTubeAccount>,
    pub offline: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeVideoMetadata {
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub privacy_status: YouTubePrivacyStatus,
    #[serde(default = "default_category")]
    pub category_id: String,
    #[serde(default)]
    pub default_language: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubePublishOptions {
    #[serde(default)]
    pub playlist_id: Option<String>,
    #[serde(default)]
    pub thumbnail_path: Option<String>,
    #[serde(default)]
    pub publish_translated_subtitles: bool,
    #[serde(default)]
    pub publish_original_subtitles: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubePlaylist {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum YouTubeAssetStepState {
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeAssetStep {
    pub kind: String,
    pub state: YouTubeAssetStepState,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubePublishingHistoryEntry {
    pub video_id: String,
    pub title: String,
    pub privacy_status: YouTubePrivacyStatus,
    pub uploaded_at: String,
    pub channel_id: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeThumbnailResult {
    pub path: String,
    pub time_seconds: f64,
}

fn default_category() -> String {
    "22".to_string()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum YouTubePrivacyStatus {
    Private,
    Unlisted,
    Public,
}

impl YouTubePrivacyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Private => "private",
            Self::Unlisted => "unlisted",
            Self::Public => "public",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum YouTubeUploadState {
    Idle,
    Waiting,
    Connecting,
    Preparing,
    Uploading,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeUploadSnapshot {
    pub id: String,
    pub project_id: String,
    pub created_at: String,
    pub state: YouTubeUploadState,
    pub file_path: String,
    pub bytes_uploaded: u64,
    pub total_bytes: u64,
    pub progress: f64,
    pub video_id: Option<String>,
    pub url: Option<String>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub can_retry: bool,
    pub title: String,
    pub privacy_status: YouTubePrivacyStatus,
    #[serde(default)]
    pub asset_steps: Vec<YouTubeAssetStep>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YouTubeUploadProgressEvent {
    pub upload: YouTubeUploadSnapshot,
}

#[derive(Debug, Clone)]
pub(crate) struct OAuthTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
}

#[derive(Debug, Error)]
pub enum YouTubeError {
    #[error("YouTube integration is not configured")]
    NotConfigured,
    #[error("Offline Mode is enabled")]
    Offline,
    #[error("YouTube authentication was cancelled")]
    OAuthCancelled,
    #[error("YouTube authentication failed: {0}")]
    OAuthFailed(String),
    #[error("YouTube permission was denied")]
    PermissionDenied,
    #[error("YouTube authentication is required")]
    AuthenticationRequired,
    #[error("Secure credential storage is unavailable: {0}")]
    CredentialStore(String),
    #[error("No YouTube account is connected")]
    NotConnected,
    #[error("The rendered video is missing or invalid: {0}")]
    InvalidVideo(String),
    #[error("Video metadata is invalid: {0}")]
    InvalidMetadata(String),
    #[error("Upload was cancelled")]
    Cancelled,
    #[error("Upload was not found")]
    UploadNotFound,
    #[error("This upload cannot be retried")]
    NotRetryable,
    #[error("This rendered video already has an active YouTube upload")]
    UploadAlreadyActive,
    #[error("The rendered video changed after it was queued")]
    RenderChanged,
    #[error("The selected YouTube account was not found")]
    AccountNotFound,
    #[error("The selected playlist is invalid")]
    InvalidPlaylist,
    #[error("The thumbnail is invalid: {0}")]
    InvalidThumbnail(String),
    #[error("FFmpeg is required to convert or generate this thumbnail")]
    ThumbnailConversionRequired,
    #[error("Subtitle publishing failed: {0}")]
    SubtitlePublish(String),
    #[error("Publishing history could not be read or written: {0}")]
    History(String),
    #[error("YouTube quota was exceeded")]
    QuotaExceeded,
    #[error("A network error interrupted the YouTube request: {0}")]
    Network(String),
    #[error("YouTube rejected the request: {0}")]
    Api(String),
    #[error("Could not access the local video: {0}")]
    Io(String),
}

impl YouTubeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotConfigured => "YOUTUBE_NOT_CONFIGURED",
            Self::Offline => "YOUTUBE_OFFLINE",
            Self::OAuthCancelled => "YOUTUBE_OAUTH_CANCELLED",
            Self::OAuthFailed(_) => "YOUTUBE_OAUTH_FAILED",
            Self::PermissionDenied => "YOUTUBE_PERMISSION_DENIED",
            Self::AuthenticationRequired => "YOUTUBE_AUTH_REQUIRED",
            Self::CredentialStore(_) => "YOUTUBE_CREDENTIAL_STORE",
            Self::NotConnected => "YOUTUBE_NOT_CONNECTED",
            Self::InvalidVideo(_) => "YOUTUBE_INVALID_VIDEO",
            Self::InvalidMetadata(_) => "YOUTUBE_INVALID_METADATA",
            Self::Cancelled => "YOUTUBE_UPLOAD_CANCELLED",
            Self::UploadNotFound => "YOUTUBE_UPLOAD_NOT_FOUND",
            Self::NotRetryable => "YOUTUBE_NOT_RETRYABLE",
            Self::UploadAlreadyActive => "YOUTUBE_UPLOAD_ACTIVE",
            Self::RenderChanged => "YOUTUBE_RENDER_CHANGED",
            Self::AccountNotFound => "YOUTUBE_ACCOUNT_NOT_FOUND",
            Self::InvalidPlaylist => "YOUTUBE_INVALID_PLAYLIST",
            Self::InvalidThumbnail(_) => "YOUTUBE_INVALID_THUMBNAIL",
            Self::ThumbnailConversionRequired => "YOUTUBE_THUMBNAIL_CONVERSION_REQUIRED",
            Self::SubtitlePublish(_) => "YOUTUBE_SUBTITLE_PUBLISH",
            Self::History(_) => "YOUTUBE_HISTORY",
            Self::QuotaExceeded => "YOUTUBE_QUOTA_EXCEEDED",
            Self::Network(_) => "YOUTUBE_NETWORK",
            Self::Api(_) => "YOUTUBE_API",
            Self::Io(_) => "YOUTUBE_IO",
        }
    }

    pub fn recoverable(&self) -> bool {
        matches!(
            self,
            Self::Offline
                | Self::OAuthCancelled
                | Self::OAuthFailed(_)
                | Self::PermissionDenied
                | Self::AuthenticationRequired
                | Self::NotConnected
                | Self::InvalidVideo(_)
                | Self::InvalidMetadata(_)
                | Self::Cancelled
                | Self::UploadAlreadyActive
                | Self::RenderChanged
                | Self::AccountNotFound
                | Self::InvalidPlaylist
                | Self::InvalidThumbnail(_)
                | Self::ThumbnailConversionRequired
                | Self::SubtitlePublish(_)
                | Self::History(_)
                | Self::Network(_)
        )
    }
}
