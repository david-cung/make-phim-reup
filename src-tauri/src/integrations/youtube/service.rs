use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::ffmpeg::detection::FfmpegHandle;
use crate::subtitles::{self, SubtitleDoc};

use super::api::{
    fetch_channel, fetch_playlists, fetch_video_privacy, insert_caption, insert_playlist_item,
    refresh_access_token, set_thumbnail, OAuthConfig,
};
use super::upload::{self, SessionPosition};
use super::{
    OAuthTokens, YouTubeAccount, YouTubeAccountStatus, YouTubeAssetStep, YouTubeAssetStepState,
    YouTubeConnectionState, YouTubeError, YouTubePlaylist, YouTubePrivacyStatus,
    YouTubePublishOptions, YouTubePublishingHistoryEntry, YouTubeThumbnailResult,
    YouTubeUploadProgressEvent, YouTubeUploadSnapshot, YouTubeUploadState, YouTubeVideoMetadata,
};

const KEYRING_SERVICE: &str = "app.localmovietranslator.youtube";
const PLAYLIST_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_THUMBNAIL_BYTES: u64 = 2 * 1024 * 1024;
const HISTORY_FILE: &str = "youtube-publishing-history.json";
const LEGACY_HISTORY_FILE: &str = "youtube-history.json";

#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountRegistry {
    #[serde(default)]
    accounts: Vec<YouTubeAccount>,
    active_account_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct CredentialSecret {
    refresh_token: String,
}

struct CachedAccessToken {
    value: String,
    expires_at: Instant,
}

struct PlaylistCache {
    loaded_at: Instant,
    playlists: Vec<YouTubePlaylist>,
}

#[derive(Clone)]
struct RenderFileIdentity {
    canonical_path: PathBuf,
    length: u64,
    modified: SystemTime,
}

struct UploadRecord {
    snapshot: YouTubeUploadSnapshot,
    account_id: String,
    channel_id: String,
    file_path: PathBuf,
    file_identity: RenderFileIdentity,
    project_root: PathBuf,
    metadata: YouTubeVideoMetadata,
    options: YouTubePublishOptions,
    subtitles: Option<SubtitleDoc>,
    session_uri: Option<String>,
    cancel: CancellationToken,
    last_emit: Instant,
}

pub struct YouTubeService {
    app: AppHandle,
    client: reqwest::Client,
    config: Option<OAuthConfig>,
    registry_path: PathBuf,
    registry: Mutex<AccountRegistry>,
    auth_status: Mutex<HashMap<String, YouTubeAccountStatus>>,
    access_tokens: Mutex<HashMap<String, CachedAccessToken>>,
    playlist_cache: Mutex<HashMap<String, PlaylistCache>>,
    playlist_fetch: tokio::sync::Mutex<()>,
    uploads: Mutex<HashMap<String, UploadRecord>>,
    queue: Mutex<VecDeque<String>>,
    active_upload: Mutex<Option<String>>,
    network_cancel: Mutex<CancellationToken>,
    ffmpeg: Arc<FfmpegHandle>,
}

impl YouTubeService {
    pub fn new(app: AppHandle, config_dir: &Path, ffmpeg: Arc<FfmpegHandle>) -> Arc<Self> {
        let registry_path = config_dir.join("youtube-accounts.json");
        let registry = load_registry(&registry_path);
        let auth_status = registry
            .accounts
            .iter()
            .map(|account| (account.id.clone(), YouTubeAccountStatus::Connected))
            .collect();
        Arc::new(Self {
            app,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(20))
                .timeout(Duration::from_secs(10 * 60))
                .user_agent(format!(
                    "LocalMovieTranslator/{}",
                    env!("CARGO_PKG_VERSION")
                ))
                .build()
                .expect("valid YouTube HTTP client"),
            config: OAuthConfig::load(),
            registry_path,
            registry: Mutex::new(registry),
            auth_status: Mutex::new(auth_status),
            access_tokens: Mutex::new(HashMap::new()),
            playlist_cache: Mutex::new(HashMap::new()),
            playlist_fetch: tokio::sync::Mutex::new(()),
            uploads: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            active_upload: Mutex::new(None),
            network_cancel: Mutex::new(CancellationToken::new()),
            ffmpeg,
        })
    }

    pub fn connection_state(&self, offline: bool) -> YouTubeConnectionState {
        let account = {
            let registry = self.registry.lock();
            registry.active_account_id.as_ref().and_then(|id| {
                registry
                    .accounts
                    .iter()
                    .find(|account| &account.id == id)
                    .cloned()
            })
        };
        let status = account
            .as_ref()
            .and_then(|account| self.auth_status.lock().get(&account.id).copied())
            .unwrap_or(YouTubeAccountStatus::Disconnected);
        YouTubeConnectionState {
            configured: self.config.is_some(),
            status,
            account,
            offline,
        }
    }

    pub fn list_accounts(&self) -> Vec<YouTubeAccount> {
        self.registry.lock().accounts.clone()
    }

    pub fn select_account(
        &self,
        account_id: &str,
        offline: bool,
    ) -> Result<YouTubeConnectionState, YouTubeError> {
        if !self
            .registry
            .lock()
            .accounts
            .iter()
            .any(|account| account.id == account_id)
        {
            return Err(YouTubeError::AccountNotFound);
        }
        self.load_refresh_token(account_id)?;
        {
            let mut registry = self.registry.lock();
            if !registry.accounts.iter().any(|account| account.id == account_id) {
                return Err(YouTubeError::AccountNotFound);
            }
            registry.active_account_id = Some(account_id.to_string());
            save_registry(&self.registry_path, &registry)?;
        }
        self.playlist_cache.lock().clear();
        Ok(self.connection_state(offline))
    }

    pub async fn connect(&self) -> Result<YouTubeConnectionState, YouTubeError> {
        let config = self.config.as_ref().ok_or(YouTubeError::NotConfigured)?;
        let cancel = self.begin_network_activity();
        let operation = async {
            let tokens = super::oauth::connect(&self.client, config).await?;
            let temporary_id = Uuid::new_v4().to_string();
            let connected_at = Utc::now().to_rfc3339();
            let account = fetch_channel(
                &self.client,
                &tokens.access_token,
                temporary_id,
                connected_at,
            )
            .await?;
            Ok::<_, YouTubeError>((tokens, account))
        };
        let (tokens, mut account) = tokio::select! {
            _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
            result = operation => result?,
        };
        {
            let registry = self.registry.lock();
            if let Some(existing) = account.channel_id.as_ref().and_then(|channel_id| {
                registry
                    .accounts
                    .iter()
                    .find(|candidate| candidate.channel_id.as_ref() == Some(channel_id))
            }) {
                account.id = existing.id.clone();
            }
        }
        self.store_refresh_token(&account.id, &tokens.refresh_token)?;
        {
            let mut registry = self.registry.lock();
            if let Some(existing) = registry
                .accounts
                .iter_mut()
                .find(|candidate| candidate.id == account.id)
            {
                *existing = account.clone();
            } else {
                registry.accounts.push(account.clone());
            }
            registry.active_account_id = Some(account.id.clone());
            save_registry(&self.registry_path, &registry)?;
        }
        self.cache_access_token(&account.id, &tokens);
        self.auth_status
            .lock()
            .insert(account.id, YouTubeAccountStatus::Connected);
        Ok(self.connection_state(false))
    }

    pub fn disconnect(
        self: &Arc<Self>,
        offline: bool,
    ) -> Result<YouTubeConnectionState, YouTubeError> {
        let account_id = self.active_account_id()?;
        self.cancel_account_uploads(&account_id);
        let entry = keyring::Entry::new(KEYRING_SERVICE, &account_id)
            .map_err(|error| YouTubeError::CredentialStore(error.to_string()))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(error) => return Err(YouTubeError::CredentialStore(error.to_string())),
        }
        {
            let mut registry = self.registry.lock();
            registry.accounts.retain(|account| account.id != account_id);
            registry.active_account_id = registry.accounts.first().map(|account| account.id.clone());
            save_registry(&self.registry_path, &registry)?;
        }
        self.access_tokens.lock().remove(&account_id);
        self.auth_status.lock().remove(&account_id);
        self.playlist_cache.lock().remove(&account_id);
        self.schedule_next();
        Ok(self.connection_state(offline))
    }

    fn cancel_account_uploads(&self, account_id: &str) {
        let mut changed = Vec::new();
        {
            let mut uploads = self.uploads.lock();
            for record in uploads.values_mut() {
                if record.account_id != account_id {
                    continue;
                }
                match record.snapshot.state {
                    YouTubeUploadState::Waiting => {
                        record.cancel.cancel();
                        record.snapshot.state = YouTubeUploadState::Cancelled;
                        record.snapshot.error_code =
                            Some(YouTubeError::Cancelled.code().to_string());
                        record.snapshot.error_message = Some(YouTubeError::Cancelled.to_string());
                        changed.push(record.snapshot.clone());
                    }
                    YouTubeUploadState::Connecting
                    | YouTubeUploadState::Preparing
                    | YouTubeUploadState::Uploading
                    | YouTubeUploadState::Processing => record.cancel.cancel(),
                    _ => {}
                }
            }
        }
        for snapshot in changed {
            self.emit(snapshot);
        }
    }

    pub async fn list_playlists(&self) -> Result<Vec<YouTubePlaylist>, YouTubeError> {
        let account_id = self.active_account_id()?;
        if let Some(cached) = self
            .playlist_cache
            .lock()
            .get(&account_id)
            .filter(|cached| cached.loaded_at.elapsed() < PLAYLIST_CACHE_TTL)
            .map(|cached| cached.playlists.clone())
        {
            return Ok(cached);
        }
        let _fetch_guard = self.playlist_fetch.lock().await;
        if let Some(cached) = self
            .playlist_cache
            .lock()
            .get(&account_id)
            .filter(|cached| cached.loaded_at.elapsed() < PLAYLIST_CACHE_TTL)
            .map(|cached| cached.playlists.clone())
        {
            return Ok(cached);
        }
        let cancel = self.begin_network_activity();
        let access_token = tokio::select! {
            _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
            token = self.access_token(&account_id) => token?,
        };
        let playlists = fetch_playlists(&self.client, &access_token, &cancel).await?;
        self.playlist_cache.lock().insert(
            account_id,
            PlaylistCache {
                loaded_at: Instant::now(),
                playlists: playlists.clone(),
            },
        );
        Ok(playlists)
    }

    pub fn list_uploads(&self) -> Vec<YouTubeUploadSnapshot> {
        let mut uploads: Vec<_> = self
            .uploads
            .lock()
            .values()
            .map(|record| record.snapshot.clone())
            .collect();
        uploads.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        uploads
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_upload(
        self: &Arc<Self>,
        project_id: String,
        project_root: PathBuf,
        file_path: PathBuf,
        metadata: YouTubeVideoMetadata,
        options: YouTubePublishOptions,
        subtitle_doc: Option<SubtitleDoc>,
    ) -> Result<YouTubeUploadSnapshot, YouTubeError> {
        validate_metadata(&metadata)?;
        validate_publish_options(&options, subtitle_doc.as_ref())?;
        let file_path = validate_video(&file_path)?;
        let file_identity = render_file_identity(&file_path)?;
        let project_root = validate_project_root(&project_root)?;
        if let Some(path) = options.thumbnail_path.as_deref() {
            validate_thumbnail(Path::new(path))?;
        }
        let total_bytes = file_identity.length;
        let account_id = self.active_account_id()?;
        let channel_id = self
            .registry
            .lock()
            .accounts
            .iter()
            .find(|account| account.id == account_id)
            .and_then(|account| account.channel_id.clone())
            .unwrap_or_default();
        let id = Uuid::new_v4().to_string();
        let asset_steps = requested_asset_steps(&options);
        let snapshot = YouTubeUploadSnapshot {
            id: id.clone(),
            project_id,
            created_at: Utc::now().to_rfc3339(),
            state: YouTubeUploadState::Waiting,
            file_path: file_path.display().to_string(),
            bytes_uploaded: 0,
            total_bytes,
            progress: 0.0,
            video_id: None,
            url: None,
            error_code: None,
            error_message: None,
            can_retry: false,
            title: metadata.title.trim().to_string(),
            privacy_status: metadata.privacy_status,
            asset_steps,
        };
        {
            let mut uploads = self.uploads.lock();
            if uploads.values().any(|record| {
                record.file_path == file_path
                    && matches!(
                        record.snapshot.state,
                        YouTubeUploadState::Waiting
                            | YouTubeUploadState::Connecting
                            | YouTubeUploadState::Preparing
                            | YouTubeUploadState::Uploading
                            | YouTubeUploadState::Processing
                    )
            }) {
                return Err(YouTubeError::UploadAlreadyActive);
            }
            uploads.insert(
                id.clone(),
                UploadRecord {
                    snapshot: snapshot.clone(),
                    account_id,
                    channel_id,
                    file_path,
                    file_identity,
                    project_root,
                    metadata,
                    options,
                    subtitles: subtitle_doc,
                    session_uri: None,
                    cancel: CancellationToken::new(),
                    last_emit: Instant::now() - Duration::from_secs(1),
                },
            );
        }
        self.queue.lock().push_back(id);
        self.emit(snapshot.clone());
        self.schedule_next();
        Ok(snapshot)
    }

    pub fn cancel_upload(self: &Arc<Self>, upload_id: &str) -> Result<(), YouTubeError> {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let record = uploads
                .get_mut(upload_id)
                .ok_or(YouTubeError::UploadNotFound)?;
            match record.snapshot.state {
                YouTubeUploadState::Waiting => {
                    record.cancel.cancel();
                    record.snapshot.state = YouTubeUploadState::Cancelled;
                    record.snapshot.error_code = Some(YouTubeError::Cancelled.code().to_string());
                    record.snapshot.error_message = Some(YouTubeError::Cancelled.to_string());
                    Some(record.snapshot.clone())
                }
                YouTubeUploadState::Connecting
                | YouTubeUploadState::Preparing
                | YouTubeUploadState::Uploading
                | YouTubeUploadState::Processing => {
                    record.cancel.cancel();
                    None
                }
                _ => None,
            }
        };
        if let Some(snapshot) = snapshot {
            self.emit(snapshot);
            self.schedule_next();
        }
        Ok(())
    }

    pub fn cancel_all_uploads(self: &Arc<Self>) {
        self.network_cancel.lock().cancel();
        let mut changed = Vec::new();
        for record in self.uploads.lock().values_mut() {
            match record.snapshot.state {
                YouTubeUploadState::Waiting => {
                    record.cancel.cancel();
                    record.snapshot.state = YouTubeUploadState::Cancelled;
                    record.snapshot.error_code = Some(YouTubeError::Cancelled.code().to_string());
                    record.snapshot.error_message = Some(YouTubeError::Cancelled.to_string());
                    changed.push(record.snapshot.clone());
                }
                YouTubeUploadState::Connecting
                | YouTubeUploadState::Preparing
                | YouTubeUploadState::Uploading
                | YouTubeUploadState::Processing => record.cancel.cancel(),
                _ => {}
            }
        }
        for snapshot in changed {
            self.emit(snapshot);
        }
    }

    pub fn retry_upload(
        self: &Arc<Self>,
        upload_id: &str,
    ) -> Result<YouTubeUploadSnapshot, YouTubeError> {
        let (snapshot, invalidate_token_for) = {
            let mut uploads = self.uploads.lock();
            let record = uploads
                .get_mut(upload_id)
                .ok_or(YouTubeError::UploadNotFound)?;
            if record.snapshot.state != YouTubeUploadState::Failed || !record.snapshot.can_retry {
                return Err(YouTubeError::NotRetryable);
            }
            let asset_retry = record.snapshot.video_id.is_some();
            if !asset_retry {
                validate_render_identity(&record.file_identity)?;
            }
            let invalidate_token = if asset_retry {
                let top_error_code = record.snapshot.error_code.clone();
                let top_retryable = top_error_code
                    .as_deref()
                    .map(is_retryable_asset_code)
                    .unwrap_or(false);
                let mut found = top_retryable
                    && record
                        .snapshot
                        .asset_steps
                        .iter()
                        .any(|step| step.state == YouTubeAssetStepState::Pending);
                let mut auth_failed =
                    top_error_code.as_deref() == Some("YOUTUBE_AUTH_REQUIRED");
                for step in &mut record.snapshot.asset_steps {
                    if step.state == YouTubeAssetStepState::Failed
                        && step
                            .error_code
                            .as_deref()
                            .map(is_retryable_asset_code)
                            .unwrap_or(false)
                    {
                        auth_failed |= step.error_code.as_deref() == Some("YOUTUBE_AUTH_REQUIRED");
                        step.state = YouTubeAssetStepState::Pending;
                        step.error_code = None;
                        step.error_message = None;
                        found = true;
                    }
                }
                if !found {
                    return Err(YouTubeError::NotRetryable);
                }
                auth_failed.then(|| record.account_id.clone())
            } else {
                None
            };
            record.cancel = CancellationToken::new();
            record.snapshot.state = YouTubeUploadState::Waiting;
            record.snapshot.error_code = None;
            record.snapshot.error_message = None;
            record.snapshot.can_retry = false;
            (record.snapshot.clone(), invalidate_token)
        };
        if let Some(account_id) = invalidate_token_for {
            self.access_tokens.lock().remove(&account_id);
        }
        self.queue.lock().push_back(upload_id.to_string());
        self.emit(snapshot.clone());
        self.schedule_next();
        Ok(snapshot)
    }

    pub fn open_video(&self, video_id: &str) -> Result<(), YouTubeError> {
        if video_id.is_empty()
            || !video_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err(YouTubeError::Api("Invalid YouTube video ID.".into()));
        }
        open::that(format!("https://www.youtube.com/watch?v={video_id}"))
            .map_err(|error| YouTubeError::Io(error.to_string()))
    }

    pub fn list_history(
        &self,
        project_root: &Path,
    ) -> Result<Vec<YouTubePublishingHistoryEntry>, YouTubeError> {
        let root = validate_project_root(project_root)?;
        let output = project_output(&root)?;
        let current = output.join(HISTORY_FILE);
        if current.exists() {
            load_history(&current)
        } else {
            load_history(&output.join(LEGACY_HISTORY_FILE))
        }
    }

    pub fn validate_thumbnail_selection(
        &self,
        path: &Path,
    ) -> Result<YouTubeThumbnailResult, YouTubeError> {
        let path = validate_thumbnail(path)?;
        Ok(YouTubeThumbnailResult {
            path: path.display().to_string(),
            time_seconds: 0.0,
        })
    }

    pub async fn generate_thumbnail(
        &self,
        project_root: &Path,
        video_path: &Path,
        time_seconds: f64,
    ) -> Result<YouTubeThumbnailResult, YouTubeError> {
        if !time_seconds.is_finite() || time_seconds < 0.0 {
            return Err(YouTubeError::InvalidThumbnail(
                "Thumbnail time must be a finite value at or after zero.".into(),
            ));
        }
        let root = validate_project_root(project_root)?;
        let video = validate_video(video_path)?;
        let ffmpeg = self
            .ffmpeg
            .get()
            .ok_or(YouTubeError::ThumbnailConversionRequired)?;
        let output_dir = project_output(&root)?;
        let output = output_dir.join(format!(
            "youtube-thumbnail-{}.jpg",
            (time_seconds * 1000.0).round() as u64
        ));
        let result = Command::new(ffmpeg.ffmpeg())
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{time_seconds:.3}"))
            .arg("-i")
            .arg(&video)
            .arg("-frames:v")
            .arg("1")
            .arg("-q:v")
            .arg("2")
            .arg(&output)
            .output()
            .await
            .map_err(|error| YouTubeError::Io(error.to_string()))?;
        if !result.status.success() {
            return Err(YouTubeError::InvalidThumbnail(
                "FFmpeg could not extract a frame at the selected time.".into(),
            ));
        }
        let output = validate_thumbnail(&output)?;
        Ok(YouTubeThumbnailResult {
            path: output.display().to_string(),
            time_seconds,
        })
    }

    fn schedule_next(self: &Arc<Self>) {
        let next = {
            let mut active = self.active_upload.lock();
            if active.is_some() {
                return;
            }
            let mut queue = self.queue.lock();
            let uploads = self.uploads.lock();
            let next = loop {
                let Some(candidate) = queue.pop_front() else {
                    break None;
                };
                if uploads
                    .get(&candidate)
                    .map(|record| record.snapshot.state == YouTubeUploadState::Waiting)
                    .unwrap_or(false)
                {
                    break Some(candidate);
                }
            };
            *active = next.clone();
            next
        };
        if let Some(upload_id) = next {
            let service = Arc::clone(self);
            tauri::async_runtime::spawn(async move {
                service.run_upload(upload_id).await;
            });
        }
    }

    async fn run_upload(self: Arc<Self>, upload_id: String) {
        let result = self.run_upload_inner(&upload_id).await;
        match result {
            Ok(video_id) => self.finish_success(&upload_id, video_id),
            Err(error) => self.finish_error(&upload_id, error),
        }
        {
            let mut active = self.active_upload.lock();
            if active.as_deref() == Some(upload_id.as_str()) {
                *active = None;
            }
        }
        self.schedule_next();
    }

    async fn run_upload_inner(&self, upload_id: &str) -> Result<String, YouTubeError> {
        let (
            account_id,
            file_path,
            metadata,
            total,
            cancel,
            existing_session,
            completed_video_id,
            file_identity,
        ) = {
            let uploads = self.uploads.lock();
            let record = uploads
                .get(upload_id)
                .ok_or(YouTubeError::UploadNotFound)?;
            (
                record.account_id.clone(),
                record.file_path.clone(),
                record.metadata.clone(),
                record.snapshot.total_bytes,
                record.cancel.clone(),
                record.session_uri.clone(),
                record.snapshot.video_id.clone(),
                record.file_identity.clone(),
            )
        };
        if cancel.is_cancelled() || !self.begin_upload_run(upload_id) {
            return Err(YouTubeError::Cancelled);
        }
        let access_token = tokio::select! {
            _ = cancel.cancelled() => return Err(YouTubeError::Cancelled),
            token = self.access_token(&account_id) => token?,
        };
        if let Some(video_id) = completed_video_id {
            return Ok(self
                .process_uploaded_video(upload_id, &access_token, video_id, &cancel)
                .await);
        }
        validate_render_identity(&file_identity)?;
        self.set_state(upload_id, YouTubeUploadState::Preparing);
        let session_uri = match existing_session {
            Some(uri) => uri,
            None => {
                let uri = upload::create_session(
                    &self.client,
                    &access_token,
                    &metadata,
                    total,
                    upload::mime_type(&file_path),
                    &cancel,
                )
                .await?;
                if let Some(record) = self.uploads.lock().get_mut(upload_id) {
                    record.session_uri = Some(uri.clone());
                }
                uri
            }
        };
        let start_offset = match upload::query_offset(
            &self.client,
            &access_token,
            &session_uri,
            total,
            &cancel,
        )
        .await?
        {
            SessionPosition::Completed(video_id) => {
                return Ok(self
                    .process_uploaded_video(upload_id, &access_token, video_id, &cancel)
                    .await);
            }
            SessionPosition::Offset(offset) => offset,
        };
        self.update_progress(upload_id, start_offset, true);
        self.set_state(upload_id, YouTubeUploadState::Uploading);
        let video_id = upload::upload_chunks(
            &self.client,
            &access_token,
            &session_uri,
            &file_path,
            total,
            start_offset,
            upload::mime_type(&file_path),
            file_identity.modified,
            cancel.clone(),
            |uploaded| self.update_progress(upload_id, uploaded, false),
        )
        .await?;
        Ok(self
            .process_uploaded_video(upload_id, &access_token, video_id, &cancel)
            .await)
    }

    async fn process_uploaded_video(
        &self,
        upload_id: &str,
        access_token: &str,
        video_id: String,
        cancel: &CancellationToken,
    ) -> String {
        self.record_uploaded_video(upload_id, &video_id);
        if self.should_run_asset(upload_id, "status") {
            let result = if cancel.is_cancelled() {
                Err(YouTubeError::Cancelled)
            } else {
                fetch_video_privacy(&self.client, access_token, &video_id, cancel).await
            };
            match result {
                Ok(privacy) => {
                    self.set_actual_privacy(upload_id, privacy);
                    self.finish_asset_step(upload_id, "status", Ok(()));
                }
                Err(error) => self.finish_asset_step(upload_id, "status", Err(error)),
            }
        }
        self.publish_assets(upload_id, access_token, &video_id, cancel)
            .await;
        if cancel.is_cancelled() {
            self.cancel_pending_asset_steps(upload_id);
        }
        video_id
    }

    async fn publish_assets(
        &self,
        upload_id: &str,
        access_token: &str,
        video_id: &str,
        cancel: &CancellationToken,
    ) {
        let (options, project_root, subtitle_doc) = {
            let uploads = self.uploads.lock();
            let Some(record) = uploads.get(upload_id) else {
                return;
            };
            (
                record.options.clone(),
                record.project_root.clone(),
                record.subtitles.clone(),
            )
        };
        if cancel.is_cancelled() {
            return;
        }
        if self.should_run_asset(upload_id, "playlist") {
            let Some(playlist_id) = options.playlist_id.as_deref() else {
                return;
            };
            let result = insert_playlist_item(
                &self.client,
                access_token,
                playlist_id,
                video_id,
                cancel,
            )
            .await;
            self.finish_asset_step(upload_id, "playlist", result);
        }
        if cancel.is_cancelled() {
            return;
        }
        if self.should_run_asset(upload_id, "thumbnail") {
            let Some(path) = options.thumbnail_path.as_deref() else {
                return;
            };
            let result = self
                .prepare_thumbnail(upload_id, &project_root, Path::new(path))
                .await
                .and_then(|path| {
                    if cancel.is_cancelled() {
                        Err(YouTubeError::Cancelled)
                    } else {
                        Ok(path)
                    }
                });
            let result = match result {
                Ok(path) => {
                    set_thumbnail(&self.client, access_token, video_id, &path, cancel).await
                }
                Err(error) => Err(error),
            };
            self.finish_asset_step(upload_id, "thumbnail", result);
        }
        if cancel.is_cancelled() {
            return;
        }
        if let Some(doc) = subtitle_doc {
            if self.should_run_asset(upload_id, "translatedSubtitles") {
                let track_name = format!(
                    "Translated (LMT {})",
                    upload_id.get(..8).unwrap_or(upload_id)
                );
                let result = project_asset_dir(&project_root)
                    .and_then(|dir| write_srt(&dir, upload_id, "translated", &doc, true))
                    .and_then(|path| validate_caption_language(&doc.target_language).map(|_| path));
                let result = match result {
                    Ok(path) => {
                        insert_caption(
                            &self.client,
                            access_token,
                            video_id,
                            &doc.target_language,
                            &track_name,
                            &path,
                            cancel,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                };
                self.finish_asset_step(upload_id, "translatedSubtitles", result);
            }
            if cancel.is_cancelled() {
                return;
            }
            if self.should_run_asset(upload_id, "originalSubtitles") {
                let track_name =
                    format!("Original (LMT {})", upload_id.get(..8).unwrap_or(upload_id));
                let result = project_asset_dir(&project_root)
                    .and_then(|dir| write_srt(&dir, upload_id, "original", &doc, false))
                    .and_then(|path| validate_caption_language(&doc.source_language).map(|_| path));
                let result = match result {
                    Ok(path) => {
                        insert_caption(
                            &self.client,
                            access_token,
                            video_id,
                            &doc.source_language,
                            &track_name,
                            &path,
                            cancel,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                };
                self.finish_asset_step(upload_id, "originalSubtitles", result);
            }
        }
    }

    fn should_run_asset(&self, upload_id: &str, kind: &str) -> bool {
        self.uploads
            .lock()
            .get(upload_id)
            .and_then(|record| {
                record
                    .snapshot
                    .asset_steps
                    .iter()
                    .find(|step| step.kind == kind)
            })
            .map(|step| step.state == YouTubeAssetStepState::Pending)
            .unwrap_or(false)
    }

    async fn prepare_thumbnail(
        &self,
        upload_id: &str,
        project_root: &Path,
        source: &Path,
    ) -> Result<PathBuf, YouTubeError> {
        let source = validate_thumbnail(source)?;
        let is_webp = source
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("webp"))
            .unwrap_or(false);
        if !is_webp {
            return Ok(source);
        }
        let ffmpeg = self
            .ffmpeg
            .get()
            .ok_or(YouTubeError::ThumbnailConversionRequired)?;
        let dir = project_asset_dir(project_root)?;
        let output = dir.join(format!("{upload_id}-thumbnail.jpg"));
        let result = Command::new(ffmpeg.ffmpeg())
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-i")
            .arg(&source)
            .arg("-frames:v")
            .arg("1")
            .arg("-q:v")
            .arg("2")
            .arg(&output)
            .output()
            .await
            .map_err(|error| YouTubeError::Io(error.to_string()))?;
        if !result.status.success() {
            return Err(YouTubeError::InvalidThumbnail(
                "FFmpeg could not convert the WebP thumbnail to JPEG.".into(),
            ));
        }
        validate_thumbnail(&output)
    }

    async fn access_token(&self, account_id: &str) -> Result<String, YouTubeError> {
        {
            let access_tokens = self.access_tokens.lock();
            if let Some(token) = access_tokens
                .get(account_id)
                .filter(|token| token.expires_at > Instant::now())
            {
                return Ok(token.value.clone());
            }
        }
        let config = self.config.as_ref().ok_or(YouTubeError::NotConfigured)?;
        let refresh_token = self.load_refresh_token(account_id)?;
        match refresh_access_token(&self.client, config, &refresh_token).await {
            Ok(tokens) => {
                self.cache_access_token(account_id, &tokens);
                self.auth_status
                    .lock()
                    .insert(account_id.to_string(), YouTubeAccountStatus::Connected);
                Ok(tokens.access_token)
            }
            Err(error @ YouTubeError::AuthenticationRequired) => {
                self.auth_status.lock().insert(
                    account_id.to_string(),
                    YouTubeAccountStatus::AuthenticationRequired,
                );
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    fn active_account_id(&self) -> Result<String, YouTubeError> {
        self.registry
            .lock()
            .active_account_id
            .clone()
            .ok_or(YouTubeError::NotConnected)
    }

    fn begin_network_activity(&self) -> CancellationToken {
        let mut token = self.network_cancel.lock();
        if token.is_cancelled() {
            *token = CancellationToken::new();
        }
        token.clone()
    }

    fn store_refresh_token(&self, account_id: &str, refresh_token: &str) -> Result<(), YouTubeError> {
        let secret = serde_json::to_string(&CredentialSecret {
            refresh_token: refresh_token.to_string(),
        })
        .map_err(|error| YouTubeError::CredentialStore(error.to_string()))?;
        keyring::Entry::new(KEYRING_SERVICE, account_id)
            .map_err(|error| YouTubeError::CredentialStore(error.to_string()))?
            .set_password(&secret)
            .map_err(|error| YouTubeError::CredentialStore(error.to_string()))
    }

    fn load_refresh_token(&self, account_id: &str) -> Result<String, YouTubeError> {
        let secret = keyring::Entry::new(KEYRING_SERVICE, account_id)
            .map_err(|error| YouTubeError::CredentialStore(error.to_string()))?
            .get_password()
            .map_err(|error| match error {
                keyring::Error::NoEntry => YouTubeError::AuthenticationRequired,
                other => YouTubeError::CredentialStore(other.to_string()),
            })?;
        serde_json::from_str::<CredentialSecret>(&secret)
            .map(|value| value.refresh_token)
            .map_err(|error| YouTubeError::CredentialStore(error.to_string()))
    }

    fn cache_access_token(&self, account_id: &str, tokens: &OAuthTokens) {
        let lifetime = tokens.expires_in.saturating_sub(60).max(60);
        self.access_tokens.lock().insert(
            account_id.to_string(),
            CachedAccessToken {
                value: tokens.access_token.clone(),
                expires_at: Instant::now() + Duration::from_secs(lifetime),
            },
        );
    }

    fn set_state(&self, upload_id: &str, state: YouTubeUploadState) {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            record.snapshot.state = state;
            record.snapshot.clone()
        };
        self.emit(snapshot);
    }

    fn record_uploaded_video(&self, upload_id: &str, video_id: &str) {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            record.snapshot.video_id = Some(video_id.to_string());
            record.snapshot.url = Some(format!("https://www.youtube.com/watch?v={video_id}"));
            record.snapshot.bytes_uploaded = record.snapshot.total_bytes;
            record.snapshot.progress = 1.0;
            record.snapshot.state = YouTubeUploadState::Processing;
            record.snapshot.clone()
        };
        self.emit(snapshot);
    }

    fn set_actual_privacy(&self, upload_id: &str, privacy: YouTubePrivacyStatus) {
        if let Some(record) = self.uploads.lock().get_mut(upload_id) {
            record.snapshot.privacy_status = privacy;
        }
    }

    fn cancel_pending_asset_steps(&self, upload_id: &str) {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            for step in &mut record.snapshot.asset_steps {
                if step.kind != "history" && step.state == YouTubeAssetStepState::Pending {
                    step.state = YouTubeAssetStepState::Failed;
                    step.error_code = Some(YouTubeError::Cancelled.code().to_string());
                    step.error_message = Some(
                        "Skipped because publishing was cancelled after the video upload completed."
                            .into(),
                    );
                }
            }
            record.snapshot.clone()
        };
        self.emit(snapshot);
    }

    fn begin_upload_run(&self, upload_id: &str) -> bool {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return false;
            };
            if record.snapshot.state != YouTubeUploadState::Waiting
                || record.cancel.is_cancelled()
            {
                return false;
            }
            record.snapshot.state = YouTubeUploadState::Connecting;
            record.snapshot.clone()
        };
        self.emit(snapshot);
        true
    }

    fn update_progress(&self, upload_id: &str, bytes_uploaded: u64, force: bool) {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            record.snapshot.bytes_uploaded = bytes_uploaded.min(record.snapshot.total_bytes);
            record.snapshot.progress = if record.snapshot.total_bytes == 0 {
                0.0
            } else {
                record.snapshot.bytes_uploaded as f64 / record.snapshot.total_bytes as f64
            };
            let now = Instant::now();
            if !force
                && bytes_uploaded < record.snapshot.total_bytes
                && now.duration_since(record.last_emit) < Duration::from_millis(200)
            {
                return;
            }
            record.last_emit = now;
            record.snapshot.clone()
        };
        self.emit(snapshot);
    }

    fn finish_asset_step(
        &self,
        upload_id: &str,
        kind: &str,
        result: Result<(), YouTubeError>,
    ) {
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            let Some(step) = record
                .snapshot
                .asset_steps
                .iter_mut()
                .find(|step| step.kind == kind)
            else {
                return;
            };
            match result {
                Ok(()) => {
                    step.state = YouTubeAssetStepState::Completed;
                    step.error_code = None;
                    step.error_message = None;
                }
                Err(error) => {
                    step.state = YouTubeAssetStepState::Failed;
                    step.error_code = Some(error.code().to_string());
                    step.error_message = Some(error.to_string());
                }
            }
            record.snapshot.clone()
        };
        self.emit(snapshot);
    }

    fn finish_success(&self, upload_id: &str, video_id: String) {
        let (project_root, title, privacy_status, channel_id) = {
            let uploads = self.uploads.lock();
            let Some(record) = uploads.get(upload_id) else {
                return;
            };
            (
                record.project_root.clone(),
                record.snapshot.title.clone(),
                record.snapshot.privacy_status,
                record.channel_id.clone(),
            )
        };
        let url = format!("https://www.youtube.com/watch?v={video_id}");
        let entry = YouTubePublishingHistoryEntry {
            video_id: video_id.clone(),
            title,
            privacy_status,
            uploaded_at: Utc::now().to_rfc3339(),
            channel_id,
            url: url.clone(),
        };
        let history_result = project_output(&project_root)
            .and_then(|output| append_history(&output.join(HISTORY_FILE), entry));
        let snapshot = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            record.snapshot.bytes_uploaded = record.snapshot.total_bytes;
            record.snapshot.progress = 1.0;
            record.snapshot.url = Some(url);
            record.snapshot.video_id = Some(video_id);
            if let Some(step) = record
                .snapshot
                .asset_steps
                .iter_mut()
                .find(|step| step.kind == "history")
            {
                match &history_result {
                    Ok(()) => {
                        step.state = YouTubeAssetStepState::Completed;
                        step.error_code = None;
                        step.error_message = None;
                    }
                    Err(error) => {
                        step.state = YouTubeAssetStepState::Failed;
                        step.error_code = Some(error.code().to_string());
                        step.error_message = Some(error.to_string());
                    }
                }
            }
            let retryable = record
                .snapshot
                .asset_steps
                .iter()
                .find(|step| {
                    step.state == YouTubeAssetStepState::Failed
                        && step
                            .error_code
                            .as_deref()
                            .map(is_retryable_asset_code)
                            .unwrap_or(false)
                })
                .map(|step| (step.error_code.clone(), step.kind.clone()));
            if let Some((error_code, kind)) = retryable {
                record.snapshot.state = YouTubeUploadState::Failed;
                record.snapshot.error_code = error_code;
                record.snapshot.error_message = Some(format!(
                    "The video was uploaded, but the {} step was interrupted. Retry to publish only the unfinished assets.",
                    kind
                ));
                record.snapshot.can_retry = true;
            } else {
                record.snapshot.state = YouTubeUploadState::Completed;
                record.snapshot.error_code = None;
                record.snapshot.error_message = None;
                record.snapshot.can_retry = false;
            }
            record.snapshot.clone()
        };
        self.emit(snapshot);
        if let Err(error) = history_result {
            tracing::warn!(code = error.code(), "YouTube history write failed");
        }
    }

    fn finish_error(&self, upload_id: &str, error: YouTubeError) {
        let authentication_failed = matches!(&error, YouTubeError::AuthenticationRequired);
        let (snapshot, account_to_invalidate) = {
            let mut uploads = self.uploads.lock();
            let Some(record) = uploads.get_mut(upload_id) else {
                return;
            };
            record.snapshot.state = if matches!(&error, YouTubeError::Cancelled) {
                YouTubeUploadState::Cancelled
            } else {
                YouTubeUploadState::Failed
            };
            record.snapshot.error_code = Some(error.code().to_string());
            record.snapshot.error_message = Some(error.to_string());
            record.snapshot.can_retry = matches!(
                &error,
                YouTubeError::Network(_) | YouTubeError::AuthenticationRequired
            );
            (
                record.snapshot.clone(),
                authentication_failed.then(|| record.account_id.clone()),
            )
        };
        if let Some(account_id) = account_to_invalidate {
            self.access_tokens.lock().remove(&account_id);
        }
        self.emit(snapshot);
    }

    fn emit(&self, snapshot: YouTubeUploadSnapshot) {
        let _ = self.app.emit(
            "youtube://upload",
            YouTubeUploadProgressEvent { upload: snapshot },
        );
    }
}

fn requested_asset_steps(options: &YouTubePublishOptions) -> Vec<YouTubeAssetStep> {
    let mut steps = vec![pending_step("status")];
    if options.playlist_id.is_some() {
        steps.push(pending_step("playlist"));
    }
    if options.thumbnail_path.is_some() {
        steps.push(pending_step("thumbnail"));
    }
    if options.publish_translated_subtitles {
        steps.push(pending_step("translatedSubtitles"));
    }
    if options.publish_original_subtitles {
        steps.push(pending_step("originalSubtitles"));
    }
    steps.push(pending_step("history"));
    steps
}

fn pending_step(kind: &str) -> YouTubeAssetStep {
    YouTubeAssetStep {
        kind: kind.to_string(),
        state: YouTubeAssetStepState::Pending,
        error_code: None,
        error_message: None,
    }
}

fn is_retryable_asset_code(code: &str) -> bool {
    matches!(
        code,
        "YOUTUBE_NETWORK" | "YOUTUBE_AUTH_REQUIRED" | "YOUTUBE_HISTORY"
    )
}

fn validate_publish_options(
    options: &YouTubePublishOptions,
    subtitle_doc: Option<&SubtitleDoc>,
) -> Result<(), YouTubeError> {
    if let Some(playlist_id) = options.playlist_id.as_deref() {
        if playlist_id.trim().is_empty()
            || !playlist_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        {
            return Err(YouTubeError::InvalidPlaylist);
        }
    }
    if (options.publish_translated_subtitles || options.publish_original_subtitles)
        && subtitle_doc.is_none()
    {
        return Err(YouTubeError::SubtitlePublish(
            "The project has no canonical subtitle document.".into(),
        ));
    }
    Ok(())
}

fn validate_metadata(metadata: &YouTubeVideoMetadata) -> Result<(), YouTubeError> {
    let title_len = metadata.title.trim().chars().count();
    if title_len == 0 || title_len > 100 {
        return Err(YouTubeError::InvalidMetadata(
            "Title must contain 1 to 100 characters.".into(),
        ));
    }
    if metadata.title.contains('<')
        || metadata.title.contains('>')
        || metadata.description.contains('<')
        || metadata.description.contains('>')
    {
        return Err(YouTubeError::InvalidMetadata(
            "Title and description cannot contain < or > characters.".into(),
        ));
    }
    if metadata.description.len() > 5_000 {
        return Err(YouTubeError::InvalidMetadata(
            "Description cannot exceed 5,000 UTF-8 bytes.".into(),
        ));
    }
    let combined_tag_chars = metadata
        .tags
        .iter()
        .map(|tag| tag.chars().count())
        .sum::<usize>()
        .saturating_add(metadata.tags.len().saturating_sub(1));
    if metadata.tags.len() > 100
        || combined_tag_chars > 500
        || metadata.tags.iter().any(|tag| tag.trim().is_empty())
    {
        return Err(YouTubeError::InvalidMetadata(
            "Tags must be non-empty and total at most 500 characters including separators.".into(),
        ));
    }
    if metadata.category_id.is_empty()
        || !metadata.category_id.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(YouTubeError::InvalidMetadata(
            "Category ID must be numeric.".into(),
        ));
    }
    if let Some(language) = metadata.default_language.as_deref() {
        if !is_language_tag(language) {
            return Err(YouTubeError::InvalidMetadata(
                "Video language must be a valid language tag.".into(),
            ));
        }
    }
    Ok(())
}

fn validate_project_root(path: &Path) -> Result<PathBuf, YouTubeError> {
    if !path.is_absolute() {
        return Err(YouTubeError::Io(
            "The project root must be an absolute path.".into(),
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    if !canonical.is_dir() {
        return Err(YouTubeError::Io(
            "The project root is not a directory.".into(),
        ));
    }
    Ok(canonical)
}

fn project_output(project_root: &Path) -> Result<PathBuf, YouTubeError> {
    let root = validate_project_root(project_root)?;
    let output = root.join("output");
    std::fs::create_dir_all(&output).map_err(|error| YouTubeError::Io(error.to_string()))?;
    let output = output
        .canonicalize()
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    if !output.starts_with(&root) {
        return Err(YouTubeError::Io(
            "The project output directory escapes the project root.".into(),
        ));
    }
    Ok(output)
}

fn project_asset_dir(project_root: &Path) -> Result<PathBuf, YouTubeError> {
    let output = project_output(project_root)?;
    let assets = output.join("youtube-assets");
    std::fs::create_dir_all(&assets).map_err(|error| YouTubeError::Io(error.to_string()))?;
    let assets = assets
        .canonicalize()
        .map_err(|error| YouTubeError::Io(error.to_string()))?;
    if !assets.starts_with(&output) {
        return Err(YouTubeError::Io(
            "The YouTube asset directory escapes project output.".into(),
        ));
    }
    Ok(assets)
}

fn validate_video(path: &Path) -> Result<PathBuf, YouTubeError> {
    if !path.is_absolute() {
        return Err(YouTubeError::InvalidVideo(
            "The video path must be absolute.".into(),
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    let metadata = std::fs::metadata(&canonical)
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(YouTubeError::InvalidVideo(
            "Select a non-empty local video file.".into(),
        ));
    }
    let extension = canonical
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(
        extension.as_str(),
        "mp4" | "m4v" | "mkv" | "mov" | "avi" | "webm"
    ) {
        return Err(YouTubeError::InvalidVideo(
            "Supported videos: MP4, M4V, MKV, MOV, AVI, and WebM.".into(),
        ));
    }
    Ok(canonical)
}

fn render_file_identity(path: &Path) -> Result<RenderFileIdentity, YouTubeError> {
    let canonical_path = validate_video(path)?;
    let metadata = std::fs::metadata(&canonical_path)
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    let modified = metadata
        .modified()
        .map_err(|error| YouTubeError::InvalidVideo(error.to_string()))?;
    Ok(RenderFileIdentity {
        canonical_path,
        length: metadata.len(),
        modified,
    })
}

fn validate_render_identity(identity: &RenderFileIdentity) -> Result<(), YouTubeError> {
    let current = render_file_identity(&identity.canonical_path)?;
    if current.canonical_path != identity.canonical_path
        || current.length != identity.length
        || current.modified != identity.modified
    {
        return Err(YouTubeError::RenderChanged);
    }
    Ok(())
}

fn validate_thumbnail(path: &Path) -> Result<PathBuf, YouTubeError> {
    if !path.is_absolute() {
        return Err(YouTubeError::InvalidThumbnail(
            "The thumbnail path must be absolute.".into(),
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| YouTubeError::InvalidThumbnail(error.to_string()))?;
    let metadata = std::fs::metadata(&canonical)
        .map_err(|error| YouTubeError::InvalidThumbnail(error.to_string()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(YouTubeError::InvalidThumbnail(
            "Select a non-empty local image.".into(),
        ));
    }
    if metadata.len() > MAX_THUMBNAIL_BYTES {
        return Err(YouTubeError::InvalidThumbnail(
            "The image must be 2 MiB or smaller.".into(),
        ));
    }
    let extension = canonical
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "jpg" | "jpeg" | "png" | "webp") {
        return Err(YouTubeError::InvalidThumbnail(
            "Supported thumbnails: JPG, JPEG, PNG, and WebP.".into(),
        ));
    }
    let mut header = [0u8; 12];
    let read = std::fs::File::open(&canonical)
        .and_then(|mut file| file.read(&mut header))
        .map_err(|error| YouTubeError::InvalidThumbnail(error.to_string()))?;
    let signature_matches = match extension.as_str() {
        "jpg" | "jpeg" => read >= 3 && header[..3] == [0xff, 0xd8, 0xff],
        "png" => read >= 8 && header[..8] == [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
        "webp" => read >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WEBP",
        _ => false,
    };
    if !signature_matches {
        return Err(YouTubeError::InvalidThumbnail(
            "The selected file content does not match its image extension.".into(),
        ));
    }
    Ok(canonical)
}

fn validate_caption_language(language: &str) -> Result<(), YouTubeError> {
    if !is_language_tag(language) {
        return Err(YouTubeError::SubtitlePublish(
            "The project subtitle language code is invalid.".into(),
        ));
    }
    Ok(())
}

fn is_language_tag(language: &str) -> bool {
    if language != language.trim() {
        return false;
    }
    let language = language.trim();
    if language.is_empty() || language.len() > 35 {
        return false;
    }
    let mut parts = language.split('-');
    let Some(primary) = parts.next() else {
        return false;
    };
    if !(2..=8).contains(&primary.len()) || !primary.bytes().all(|byte| byte.is_ascii_alphabetic())
    {
        return false;
    }
    parts.all(|part| {
        !part.is_empty()
            && part.len() <= 8
            && part.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
}

fn write_srt(
    directory: &Path,
    upload_id: &str,
    kind: &str,
    doc: &SubtitleDoc,
    translated: bool,
) -> Result<PathBuf, YouTubeError> {
    std::fs::create_dir_all(directory).map_err(|error| YouTubeError::Io(error.to_string()))?;
    let path = directory.join(format!("{upload_id}-{kind}.srt"));
    let body = subtitles::srt::write(&doc.segments, |segment| {
        if translated {
            segment.translated_text.clone()
        } else {
            segment.source_text.clone()
        }
    });
    std::fs::write(&path, body).map_err(|error| YouTubeError::Io(error.to_string()))?;
    Ok(path)
}

fn load_registry(path: &Path) -> AccountRegistry {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn save_registry(path: &Path, registry: &AccountRegistry) -> Result<(), YouTubeError> {
    let parent = path
        .parent()
        .ok_or_else(|| YouTubeError::Io("Invalid account registry path.".into()))?;
    std::fs::create_dir_all(parent).map_err(|error| YouTubeError::Io(error.to_string()))?;
    let temporary = path.with_extension("json.tmp");
    let bytes =
        serde_json::to_vec_pretty(registry).map_err(|error| YouTubeError::Io(error.to_string()))?;
    std::fs::write(&temporary, bytes).map_err(|error| YouTubeError::Io(error.to_string()))?;
    replace_file(&temporary, path).map_err(|error| YouTubeError::Io(error.to_string()))
}

fn load_history(path: &Path) -> Result<Vec<YouTubePublishingHistoryEntry>, YouTubeError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(path).map_err(|error| YouTubeError::History(error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|error| YouTubeError::History(error.to_string()))
}

fn append_history(
    path: &Path,
    entry: YouTubePublishingHistoryEntry,
) -> Result<(), YouTubeError> {
    let parent = path
        .parent()
        .ok_or_else(|| YouTubeError::History("Invalid history path.".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| YouTubeError::History(error.to_string()))?;
    let mut history = if path.exists() {
        load_history(path)?
    } else {
        load_history(&path.with_file_name(LEGACY_HISTORY_FILE))?
    };
    history.retain(|existing| existing.video_id != entry.video_id);
    history.insert(0, entry);
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(&history)
        .map_err(|error| YouTubeError::History(error.to_string()))?;
    std::fs::write(&temporary, bytes)
        .map_err(|error| YouTubeError::History(error.to_string()))?;
    replace_file(&temporary, path).map_err(|error| YouTubeError::History(error.to_string()))
}

fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    if destination.exists() {
        std::fs::remove_file(destination)?;
    }
    std::fs::rename(source, destination)
}
