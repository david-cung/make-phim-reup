//! Structured errors returned to the frontend.
//!
//! Contract with the UI: every command returns `Result<T, AppError>`. The
//! frontend never needs to parse an anyhow chain or a Rust panic message.

use serde::Serialize;
use thiserror::Error;

use crate::audio::ExtractError;
use crate::db::DbError;
use crate::ffmpeg::FfmpegError;
use crate::jobs::JobRegistryError;
use crate::mix::MixError;
use crate::models::ModelManagerError;
use crate::paths::PathError;
use crate::projects::ProjectError;
use crate::render::RenderError;
use crate::stt::SttError;
use crate::subtitles::SubtitleError;
use crate::sync::SyncError;
use crate::translation::TranslationError;
use crate::tts::TtsError;
use crate::worker::WorkerError;

/// Serialised as `{ code, stage, message, recoverable, hint }` — see
/// `src/ipc/types.ts`.
#[derive(Debug, Clone, Serialize, Error)]
#[error("{code}: {message}")]
pub struct AppError {
    pub code: String,
    pub stage: Option<String>,
    pub message: String,
    pub recoverable: bool,
    pub hint: Option<String>,
}

impl AppError {
    pub fn new<C: Into<String>, M: Into<String>>(code: C, message: M) -> Self {
        Self {
            code: code.into(),
            stage: None,
            message: message.into(),
            recoverable: false,
            hint: None,
        }
    }

    pub fn recoverable<C: Into<String>, M: Into<String>>(code: C, message: M) -> Self {
        Self {
            code: code.into(),
            stage: None,
            message: message.into(),
            recoverable: true,
            hint: None,
        }
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl From<PathError> for AppError {
    fn from(err: PathError) -> Self {
        let code = match &err {
            PathError::DirUnavailable { .. } => "PATH_DIR_UNAVAILABLE",
            PathError::Empty => "PATH_EMPTY",
            PathError::InvalidComponent { .. } => "PATH_INVALID",
            PathError::EscapesRoot { .. } => "PATH_ESCAPES_ROOT",
            PathError::NotAbsolute { .. } => "PATH_NOT_ABSOLUTE",
            PathError::Io { .. } => "PATH_IO",
        };
        AppError::new(code, err.to_string())
    }
}

impl From<DbError> for AppError {
    fn from(err: DbError) -> Self {
        let code = match &err {
            DbError::NotFound { .. } => "DB_NOT_FOUND",
            DbError::Migration { .. } => "DB_MIGRATION",
            DbError::Sqlite { .. } => "DB_SQLITE",
            DbError::Serialization { .. } => "DB_SERIALIZATION",
            DbError::Join { .. } => "DB_JOIN",
        };
        let mut e = AppError::new(code, err.to_string());
        if matches!(err, DbError::NotFound { .. }) {
            e.recoverable = true;
        }
        e
    }
}

impl From<ProjectError> for AppError {
    fn from(err: ProjectError) -> Self {
        let code = match &err {
            ProjectError::EmptyName => "PROJECT_EMPTY_NAME",
            ProjectError::NameTooLong { .. } => "PROJECT_NAME_TOO_LONG",
            ProjectError::InvalidLanguage { .. } => "PROJECT_INVALID_LANGUAGE",
            ProjectError::InvalidSourcePath => "PROJECT_INVALID_SOURCE_PATH",
            ProjectError::SourceNotFound { .. } => "PROJECT_SOURCE_NOT_FOUND",
            ProjectError::UnsupportedSourceExtension { .. } => "PROJECT_UNSUPPORTED_EXTENSION",
            ProjectError::Path(_) => "PROJECT_PATH",
            ProjectError::Db(_) => "PROJECT_DB",
            ProjectError::Io { .. } => "PROJECT_IO",
        };
        let mut e = match err {
            ProjectError::Path(p) => AppError::from(p),
            ProjectError::Db(d) => AppError::from(d),
            other => AppError::new(code, other.to_string()),
        };
        if matches!(
            e.code.as_str(),
            "PROJECT_INVALID_SOURCE_PATH"
                | "PROJECT_SOURCE_NOT_FOUND"
                | "PROJECT_UNSUPPORTED_EXTENSION"
        ) {
            e.recoverable = true;
        }
        e
    }
}

impl From<FfmpegError> for AppError {
    fn from(err: FfmpegError) -> Self {
        let mut e = AppError::new(err.code(), err.to_string()).with_stage("media");
        e.recoverable = err.is_recoverable();
        e.hint = match &err {
            FfmpegError::NotFound { .. } | FfmpegError::ProbeNotFound { .. } => {
                Some("Install FFmpeg or set a custom path in Settings.".into())
            }
            FfmpegError::UnsupportedExtension { .. } => {
                Some("Supported: mp4, m4v, mkv, mov, avi, webm.".into())
            }
            FfmpegError::NoAudioStream => {
                Some("This file has no audio track. Try a different source.".into())
            }
            FfmpegError::DiskSpaceLow { .. } => {
                Some("Free up disk space or move the project to a larger drive.".into())
            }
            _ => None,
        };
        e
    }
}

impl From<JobRegistryError> for AppError {
    fn from(err: JobRegistryError) -> Self {
        let code = match &err {
            JobRegistryError::NotFound(_) => "JOB_NOT_FOUND",
            JobRegistryError::Conflict { .. } => "JOB_CONFLICT",
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("jobs");
        e.recoverable = true;
        e
    }
}

impl From<ExtractError> for AppError {
    fn from(err: ExtractError) -> Self {
        match err {
            ExtractError::FfmpegUnavailable { reason } => {
                let mut e =
                    AppError::new("FFMPEG_NOT_FOUND", reason).with_stage("audio_extraction");
                e.recoverable = true;
                e.hint = Some("Install FFmpeg or set a custom path in Settings.".into());
                e
            }
            ExtractError::NoSourceMedia => AppError::recoverable(
                "NO_SOURCE_MEDIA",
                "Import a video into the project before extracting audio.",
            )
            .with_stage("audio_extraction"),
            ExtractError::Ffmpeg(f) => AppError::from(f).with_stage("audio_extraction"),
            ExtractError::Db(d) => AppError::from(d).with_stage("audio_extraction"),
            ExtractError::Registry(r) => AppError::from(r).with_stage("audio_extraction"),
            ExtractError::Io { path, source } => {
                AppError::new("AUDIO_IO", format!("{path}: {source}"))
                    .with_stage("audio_extraction")
            }
        }
    }
}

impl From<WorkerError> for AppError {
    fn from(err: WorkerError) -> Self {
        // Phase 12 — the most common Spawn failure is "Python isn't
        // installed" or "the bundled worker resources are missing".
        // Distinguish the two so the UI can show install instructions
        // instead of a raw OS error.
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            WorkerError::NotRunning => ("WORKER_NOT_RUNNING", true, None),
            WorkerError::Timeout { .. } => ("WORKER_TIMEOUT", true, None),
            WorkerError::Rpc { .. } => ("WORKER_RPC", true, None),
            WorkerError::Spawn { source } => {
                if source.kind() == std::io::ErrorKind::NotFound {
                    (
                        "WORKER_PYTHON_MISSING",
                        true,
                        Some(
                            "Python 3.11+ is required for the AI worker. Install Python from python.org (or your OS package manager) and restart the app, or set LMT_PYTHON to a specific interpreter.",
                        ),
                    )
                } else if source.kind() == std::io::ErrorKind::PermissionDenied {
                    (
                        "WORKER_SPAWN_DENIED",
                        false,
                        Some(
                            "The AI worker binary is not executable. Check file permissions on the Python interpreter and retry.",
                        ),
                    )
                } else {
                    ("WORKER_SPAWN", false, None)
                }
            }
            WorkerError::Io { .. } => ("WORKER_IO", true, None),
            WorkerError::Protocol { .. } => ("WORKER_PROTOCOL", true, None),
            WorkerError::Shutdown { .. } => ("WORKER_SHUTDOWN", true, None),
            WorkerError::Cancelled => ("WORKER_CANCELLED", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("worker");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<SttError> for AppError {
    fn from(err: SttError) -> Self {
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            SttError::NoSourceMedia => (
                "STT_NO_SOURCE_MEDIA",
                true,
                Some("Import a video into the project before transcribing."),
            ),
            SttError::AudioNotExtracted => (
                "STT_AUDIO_NOT_EXTRACTED",
                true,
                Some("Extract audio for this project before transcribing."),
            ),
            SttError::UnknownModel { .. } => (
                "STT_UNKNOWN_MODEL",
                true,
                Some("Pick a different Whisper model."),
            ),
            SttError::ModelNotInstalled { .. } => (
                "STT_MODEL_NOT_INSTALLED",
                true,
                Some("Download this Whisper model first."),
            ),
            SttError::WhisperNotInstalled => (
                "STT_WHISPER_NOT_INSTALLED",
                true,
                Some("Install faster-whisper in the worker Python environment."),
            ),
            SttError::ModelLoadFailed { .. } => (
                "STT_MODEL_LOAD_FAILED",
                true,
                Some("Try a smaller model or a different compute type."),
            ),
            SttError::InvalidAudio { .. } => (
                "STT_INVALID_AUDIO",
                true,
                Some("Re-extract audio and try again."),
            ),
            SttError::OutOfMemory => (
                "STT_OUT_OF_MEMORY",
                true,
                Some("Try a smaller model or int8 compute type."),
            ),
            SttError::WorkerCrash => ("STT_WORKER_CRASH", true, None),
            SttError::Cancelled => ("STT_CANCELLED", true, None),
            SttError::DownloadFailed { .. } => ("STT_DOWNLOAD_FAILED", true, None),
            SttError::Worker(_) => ("STT_WORKER_ERROR", true, None),
            SttError::Registry(_) => ("STT_REGISTRY", true, None),
            SttError::Db(_) => ("STT_DB", true, None),
            SttError::Io { .. } => ("STT_IO", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("transcription");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<TranslationError> for AppError {
    fn from(err: TranslationError) -> Self {
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            TranslationError::NoTranscript => (
                "TRANSLATE_NO_TRANSCRIPT",
                true,
                Some("Run speech recognition before translating."),
            ),
            TranslationError::ModelNotInstalled { .. } => (
                "TRANSLATE_MODEL_NOT_INSTALLED",
                true,
                Some("Drop a GGUF model file into <models>/translation/ and refresh the list."),
            ),
            TranslationError::LlamaNotInstalled => (
                "TRANSLATE_LLAMA_NOT_INSTALLED",
                true,
                Some("Install llama-cpp-python in the worker Python environment."),
            ),
            TranslationError::ModelLoadFailed { .. } => (
                "TRANSLATE_MODEL_LOAD_FAILED",
                true,
                Some("Try a smaller quantisation or a different GGUF file."),
            ),
            TranslationError::InvalidJson { .. } => (
                "TRANSLATE_INVALID_JSON",
                true,
                Some("Retry the same chunk; if it keeps failing, try a stronger model."),
            ),
            TranslationError::IncompleteResponse { .. } => (
                "TRANSLATE_INCOMPLETE_RESPONSE",
                true,
                Some("Try a smaller chunk size or a stronger model."),
            ),
            TranslationError::OutOfMemory => (
                "TRANSLATE_OUT_OF_MEMORY",
                true,
                Some("Try a smaller/lower-quant model or reduce context size."),
            ),
            TranslationError::LlmFailure { .. } => ("TRANSLATE_LLM_FAILURE", true, None),
            TranslationError::WorkerCrash => ("TRANSLATE_WORKER_CRASH", true, None),
            TranslationError::Cancelled => ("TRANSLATE_CANCELLED", true, None),
            TranslationError::UnknownPromptVersion { .. } => {
                ("TRANSLATE_UNKNOWN_PROMPT", true, None)
            }
            TranslationError::UnknownPreset { .. } => (
                "TRANSLATE_UNKNOWN_PRESET",
                true,
                Some(
                    "The app tried to auto-download an unrecognised translation model. Update the app or install a GGUF manually.",
                ),
            ),
            TranslationError::DownloadFailed { .. } => (
                "TRANSLATE_DOWNLOAD_FAILED",
                true,
                Some(
                    "Check your network connection and disk space, then retry. Rate-limit errors from HuggingFace usually clear within a few minutes.",
                ),
            ),
            TranslationError::Worker(_) => ("TRANSLATE_WORKER_ERROR", true, None),
            TranslationError::Registry(_) => ("TRANSLATE_REGISTRY", true, None),
            TranslationError::Db(_) => ("TRANSLATE_DB", true, None),
            TranslationError::Io { .. } => ("TRANSLATE_IO", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("translation");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<SubtitleError> for AppError {
    fn from(err: SubtitleError) -> Self {
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            SubtitleError::NoTranscript => (
                "SUBTITLE_NO_TRANSCRIPT",
                true,
                Some("Run speech recognition first so subtitles have timing to attach to."),
            ),
            SubtitleError::NoSubtitles => (
                "SUBTITLE_NO_DOCUMENT",
                true,
                Some(
                    "Click \"Build subtitles\" to derive them from the transcript and translation.",
                ),
            ),
            SubtitleError::SegmentNotFound { .. } => ("SUBTITLE_SEGMENT_NOT_FOUND", true, None),
            SubtitleError::InvalidTiming { .. } => (
                "SUBTITLE_INVALID_TIMING",
                true,
                Some("End time must be strictly greater than start time."),
            ),
            SubtitleError::InvalidSplit { .. } => (
                "SUBTITLE_INVALID_SPLIT",
                true,
                Some("Pick a split time strictly between the segment's start and end."),
            ),
            SubtitleError::NoMergeTarget { .. } => (
                "SUBTITLE_NO_MERGE_TARGET",
                true,
                Some("This is the last segment; there's nothing to merge into."),
            ),
            SubtitleError::InvalidSubtitleFile { .. } => (
                "SUBTITLE_INVALID_FILE",
                true,
                Some("Confirm the file is well-formed SRT or ASS."),
            ),
            SubtitleError::UnsupportedFormat { .. } => (
                "SUBTITLE_UNSUPPORTED_FORMAT",
                true,
                Some("Only .srt, .ass and .ssa are supported."),
            ),
            SubtitleError::InvalidExportPath => (
                "SUBTITLE_INVALID_EXPORT_PATH",
                true,
                Some("Pick an absolute file path for the export."),
            ),
            SubtitleError::Db(_) => ("SUBTITLE_DB", true, None),
            SubtitleError::Io { .. } => ("SUBTITLE_IO", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("subtitles");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<TtsError> for AppError {
    fn from(err: TtsError) -> Self {
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            TtsError::NoSubtitles => (
                "TTS_NO_SUBTITLES",
                true,
                Some("Build subtitles first, then generate voice for them."),
            ),
            TtsError::EngineUnavailable { .. } => (
                "TTS_ENGINE_UNAVAILABLE",
                true,
                Some(
                    "Install the TTS engine in the worker Python environment (e.g. `pip install piper-tts`).",
                ),
            ),
            TtsError::VoiceMissing { .. } => (
                "TTS_VOICE_MISSING",
                true,
                Some("Drop a voice model into <models>/tts/<engine>/<voice_id>/ and refresh."),
            ),
            TtsError::ModelInvalid { .. } => (
                "TTS_MODEL_INVALID",
                true,
                Some("The selected voice model failed to load; try a different voice."),
            ),
            TtsError::InvalidText => (
                "TTS_INVALID_TEXT",
                true,
                Some("The subtitle text is empty; fill in a translation before generating."),
            ),
            TtsError::EngineFailure { .. } => ("TTS_ENGINE_FAILURE", true, None),
            TtsError::OutOfMemory => (
                "TTS_OUT_OF_MEMORY",
                true,
                Some("Try a smaller voice model or free RAM."),
            ),
            TtsError::DiskFull => (
                "TTS_DISK_FULL",
                true,
                Some("Free disk space, then retry generation."),
            ),
            TtsError::WorkerCrash => ("TTS_WORKER_CRASH", true, None),
            TtsError::Cancelled => ("TTS_CANCELLED", true, None),
            TtsError::SegmentNotFound { .. } => ("TTS_SEGMENT_NOT_FOUND", true, None),
            TtsError::Worker(_) => ("TTS_WORKER_ERROR", true, None),
            TtsError::Registry(_) => ("TTS_REGISTRY", true, None),
            TtsError::Db(_) => ("TTS_DB", true, None),
            TtsError::Io { .. } => ("TTS_IO", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("tts");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<SyncError> for AppError {
    fn from(err: SyncError) -> Self {
        let (code, recoverable, hint): (&'static str, bool, Option<&'static str>) = match &err {
            SyncError::NoSubtitles => (
                "SYNC_NO_SUBTITLES",
                true,
                Some("Build subtitles first, then generate TTS, then run sync."),
            ),
            SyncError::NoTts => (
                "SYNC_NO_TTS",
                true,
                Some("Generate voices for the subtitles before running sync."),
            ),
            SyncError::SegmentNotFound { .. } => ("SYNC_SEGMENT_NOT_FOUND", true, None),
            SyncError::SegmentTtsMissing { .. } => (
                "SYNC_TTS_MISSING",
                true,
                Some("Generate TTS for this segment first."),
            ),
            SyncError::InvalidTiming { .. } => (
                "SYNC_INVALID_TIMING",
                true,
                Some("End time must be strictly greater than start time."),
            ),
            SyncError::SourceMissing { .. } => (
                "SYNC_SOURCE_MISSING",
                true,
                Some("Re-run TTS to regenerate the source voice file."),
            ),
            SyncError::SourceInvalid { .. } => (
                "SYNC_SOURCE_INVALID",
                true,
                Some("Re-run TTS to regenerate the source voice file."),
            ),
            SyncError::FfmpegMissing => (
                "SYNC_FFMPEG_MISSING",
                true,
                Some("Install FFmpeg or set a custom path in Settings."),
            ),
            SyncError::EngineFailure { .. } => ("SYNC_ENGINE_FAILURE", true, None),
            SyncError::DiskFull => (
                "SYNC_DISK_FULL",
                true,
                Some("Free disk space, then retry sync."),
            ),
            SyncError::WorkerCrash => ("SYNC_WORKER_CRASH", true, None),
            SyncError::Cancelled => ("SYNC_CANCELLED", true, None),
            SyncError::Worker(_) => ("SYNC_WORKER_ERROR", true, None),
            SyncError::Registry(_) => ("SYNC_REGISTRY", true, None),
            SyncError::Db(_) => ("SYNC_DB", true, None),
            SyncError::Io { .. } => ("SYNC_IO", true, None),
        };
        let mut e = AppError::new(code, err.to_string()).with_stage("sync");
        e.recoverable = recoverable;
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<MixError> for AppError {
    fn from(err: MixError) -> Self {
        let hint: Option<&'static str> = match &err {
            MixError::NoSourceMedia => Some("Import a video into the project before mixing audio."),
            MixError::NoSubtitles => {
                Some("Build subtitles first, then generate TTS, then sync, then mix.")
            }
            MixError::NoSyncedVoices => {
                Some("Run voice sync so per-segment synced WAVs exist, then retry mix.")
            }
            MixError::FfmpegMissing => Some("Install FFmpeg or set a custom path in Settings."),
            MixError::DiskFull => Some("Free disk space, then retry mix."),
            MixError::VoiceMissing { .. } => {
                Some("Re-run voice sync to regenerate the missing WAV file.")
            }
            _ => None,
        };
        let mut e = AppError::new(err.code(), err.to_string()).with_stage("mix");
        e.recoverable = err.is_recoverable();
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<RenderError> for AppError {
    fn from(err: RenderError) -> Self {
        let hint: Option<&'static str> = match &err {
            RenderError::NoSourceMedia => Some("Import a video into the project before rendering."),
            RenderError::NoSubtitles => {
                Some("Build subtitles first — Phase 9 renders subtitles into the final movie.")
            }
            RenderError::NoMix => Some("Run audio mix before rendering the final movie."),
            RenderError::MixFileMissing { .. } => {
                Some("The mixed WAV was removed. Re-run audio mix, then retry render.")
            }
            RenderError::FfmpegMissing => Some("Install FFmpeg or set a custom path in Settings."),
            RenderError::DiskFull => Some("Free disk space, then retry render."),
            RenderError::InvalidOutputPath { .. } => {
                Some("Provide an absolute file path for the final movie.")
            }
            RenderError::OutputInvalid { .. } => {
                Some("Rendering finished but the output file is empty or unreadable. Retry.")
            }
            RenderError::ValidationMismatch { .. } => {
                Some("The rendered file is missing the video or audio stream we expected. Retry.")
            }
            _ => None,
        };
        let mut e = AppError::new(err.code(), err.to_string()).with_stage("render");
        e.recoverable = err.is_recoverable();
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<ModelManagerError> for AppError {
    fn from(err: ModelManagerError) -> Self {
        let hint: Option<&'static str> = match &err {
            ModelManagerError::UnsupportedKind { .. } => {
                Some("Supported kinds: whisper, translation, tts, voice.")
            }
            ModelManagerError::InvalidSourcePath => {
                Some("Pick an absolute file or directory path.")
            }
            ModelManagerError::SourceNotFound { .. } => {
                Some("Double-check the path and try again.")
            }
            ModelManagerError::UnsupportedSource { .. } => Some(
                "Whisper models need a CTranslate2 snapshot directory; translation models need a .gguf file; voices need a Piper voice directory.",
            ),
            ModelManagerError::MissingRequiredFile { .. } => Some(
                "The model directory is missing a required file (model.bin / config.json for Whisper).",
            ),
            ModelManagerError::Unreadable { .. } => Some("Check file permissions and retry."),
            ModelManagerError::InvalidName { .. } => {
                Some("Use a plain name without slashes, backslashes, or `..`.")
            }
            ModelManagerError::AlreadyExists { .. } => {
                Some("Remove the existing model from the models directory first.")
            }
            ModelManagerError::PermissionDenied { .. } => {
                Some("Grant write permission to the models directory or pick another location.")
            }
            ModelManagerError::ModelsDirNotWritable { .. } => {
                Some("Pick a writable models directory in Settings › AI Models.")
            }
            ModelManagerError::NetworkDisabled => Some(
                "Offline Mode is on. Turn it off in Settings, or install the model manually via `Add Local Model`.",
            ),
            ModelManagerError::Io { .. } => None,
        };
        let mut e = AppError::new(err.code(), err.to_string()).with_stage("models");
        e.recoverable = err.is_recoverable();
        if let Some(h) = hint {
            e = e.with_hint(h);
        }
        e
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::new("SERIALIZATION", err.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::new("IO", err.to_string())
    }
}
