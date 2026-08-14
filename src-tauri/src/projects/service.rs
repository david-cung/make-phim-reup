//! Project creation / lookup / deletion / media import.
//!
//! The **source movie is never modified.** All derived files live under
//! `<projects_dir>/<uuid>/`. Every path is validated with
//! `paths::validate_within` before we touch the filesystem.
//!
//! Importing media has two modes:
//!   * `Reference` (default) — store the original path + fingerprint. No
//!     data copy. Multi-GB movies stay on the disk they came from.
//!   * `Copy` — copy the source into `<project>/source/<basename>` so the
//!     project folder is fully self-contained (useful for moving it
//!     between machines).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::db::models::{
    ProjectModelPatch, ProjectRecord, ProjectStatus, ProjectSummary, SourceImportMode,
};
use crate::db::{DbError, DbHandle};
use crate::media::{fingerprint_file, is_supported_extension, SourceFingerprint};
use crate::paths::{self, AppPaths, PathError};

const MAX_NAME_LEN: usize = 200;

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("project name must not be empty")]
    EmptyName,

    #[error("project name is too long (max {max} chars)")]
    NameTooLong { max: usize },

    #[error("invalid language code `{code}` (expected 2–5 ASCII chars)")]
    InvalidLanguage { code: String },

    #[error("source media path is not absolute or is empty")]
    InvalidSourcePath,

    #[error("source media does not exist: `{path}`")]
    SourceNotFound { path: String },

    #[error("source media has an unsupported extension: `{ext}`")]
    UnsupportedSourceExtension { ext: String },

    #[error(transparent)]
    Path(#[from] PathError),

    #[error(transparent)]
    Db(#[from] DbError),

    #[error("filesystem error at `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateProjectInput {
    pub name: String,
    pub source_language: String,
    pub target_language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportMediaInput {
    pub project_id: String,
    pub source_path: String,
    /// `true` copies the file into the project's `source/` folder,
    /// `false` (default) just references the original.
    #[serde(default)]
    pub copy_into_project: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportMediaResult {
    pub project: ProjectRecord,
    pub fingerprint: SourceFingerprint,
    pub mode: SourceImportMode,
    pub source_media_path: String,
}

pub struct ProjectService {
    paths: AppPaths,
    db: DbHandle,
}

impl ProjectService {
    pub fn new(paths: AppPaths, db: DbHandle) -> Arc<Self> {
        Arc::new(Self { paths, db })
    }

    pub async fn list(self: &Arc<Self>) -> Result<Vec<ProjectSummary>, ProjectError> {
        let db = self.db.clone();
        Ok(db.run(|d| d.list_project_summaries()).await?)
    }

    pub async fn create(
        self: &Arc<Self>,
        input: CreateProjectInput,
    ) -> Result<ProjectSummary, ProjectError> {
        let name = input.name.trim().to_string();
        if name.is_empty() {
            return Err(ProjectError::EmptyName);
        }
        if name.chars().count() > MAX_NAME_LEN {
            return Err(ProjectError::NameTooLong { max: MAX_NAME_LEN });
        }
        validate_language(&input.source_language)?;
        validate_language(&input.target_language)?;

        let id = Uuid::new_v4().to_string();
        let root = paths::project_root(&self.paths.projects_dir, &id)?;
        create_project_layout(&root)?;

        let now = Utc::now();
        let rec = ProjectRecord {
            id: id.clone(),
            name,
            source_language: input.source_language,
            target_language: input.target_language,
            root_path: root.to_string_lossy().to_string(),
            source_media_path: None,
            status: ProjectStatus::Created,
            progress: Default::default(),
            created_at: now,
            updated_at: now,
            last_opened_at: None,
            source_hash: None,
            source_size: None,
            source_modified_at: None,
            source_import_mode: None,
            whisper_model: None,
            translation_model: None,
            tts_engine: None,
            tts_voice_id: None,
        };

        let rec_for_db = rec.clone();
        let db = self.db.clone();
        db.run(move |d| d.insert_project(&rec_for_db)).await?;

        // Also persist a project.json mirror on disk so the folder is
        // self-describing even if the DB is lost.
        write_project_json(&root, &rec)?;

        Ok(rec.to_summary())
    }

    pub async fn open(self: &Arc<Self>, id: String) -> Result<ProjectRecord, ProjectError> {
        // Validate id shape before touching the DB.
        paths::project_root(&self.paths.projects_dir, &id)?;

        let db = self.db.clone();
        let id_for_touch = id.clone();
        let now = Utc::now();
        db.run(move |d| d.touch_project_opened(&id_for_touch, now))
            .await?;

        let db = self.db.clone();
        let id_for_get = id.clone();
        let rec = db.run(move |d| d.get_project(&id_for_get)).await?;
        Ok(rec)
    }

    pub async fn import_media(
        self: &Arc<Self>,
        input: ImportMediaInput,
    ) -> Result<ImportMediaResult, ProjectError> {
        let source = PathBuf::from(input.source_path.trim());
        if source.as_os_str().is_empty() || !source.is_absolute() {
            return Err(ProjectError::InvalidSourcePath);
        }
        if !source.exists() {
            return Err(ProjectError::SourceNotFound {
                path: source.display().to_string(),
            });
        }
        if !is_supported_extension(&source) {
            let ext = source
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_string();
            return Err(ProjectError::UnsupportedSourceExtension { ext });
        }

        let root = paths::project_root(&self.paths.projects_dir, &input.project_id)?;

        // Prefer reference mode; only copy when explicitly requested.
        let (mode, final_source) = if input.copy_into_project {
            let filename = source
                .file_name()
                .ok_or_else(|| ProjectError::InvalidSourcePath)?;
            let dest = paths::validate_within(&root, &root.join("source").join(filename))?;
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|source| ProjectError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
            std::fs::copy(&source, &dest).map_err(|source| ProjectError::Io {
                path: dest.display().to_string(),
                source,
            })?;
            (SourceImportMode::Copy, dest)
        } else {
            (SourceImportMode::Reference, source.clone())
        };

        let fp = fingerprint_file(&final_source).map_err(|source| ProjectError::Io {
            path: final_source.display().to_string(),
            source,
        })?;

        let id = input.project_id.clone();
        let path_str = final_source.display().to_string();
        let hash = fp.hash.clone();
        let modified = fp.modified_at;
        let size_i64 = i64::try_from(fp.size_bytes).unwrap_or(i64::MAX);
        let db = self.db.clone();
        db.run(move |d| d.set_project_source(&id, &path_str, &hash, size_i64, modified, mode))
            .await?;

        let id2 = input.project_id.clone();
        let db = self.db.clone();
        let rec = db.run(move |d| d.get_project(&id2)).await?;

        // Refresh the project.json mirror.
        write_project_json(&root, &rec)?;

        Ok(ImportMediaResult {
            project: rec,
            fingerprint: fp,
            mode,
            source_media_path: final_source.display().to_string(),
        })
    }

    /// Phase 10 — persist the per-project model selection. The
    /// database is authoritative; we also refresh the `project.json`
    /// mirror so the on-disk folder stays self-describing.
    pub async fn update_models(
        self: &Arc<Self>,
        project_id: String,
        patch: ProjectModelPatch,
    ) -> Result<ProjectRecord, ProjectError> {
        // Validate the project id shape before touching the DB.
        let root = paths::project_root(&self.paths.projects_dir, &project_id)?;
        let id_for_update = project_id.clone();
        let patch_owned = patch;
        let db = self.db.clone();
        db.run(move |d| d.set_project_models(&id_for_update, &patch_owned))
            .await?;

        let id_for_get = project_id.clone();
        let db = self.db.clone();
        let rec = db.run(move |d| d.get_project(&id_for_get)).await?;
        write_project_json(&root, &rec)?;
        Ok(rec)
    }

    pub async fn delete(self: &Arc<Self>, id: String) -> Result<(), ProjectError> {
        let root = paths::project_root(&self.paths.projects_dir, &id)?;
        let db = self.db.clone();
        let id_for_delete = id.clone();
        db.run(move |d| d.delete_project(&id_for_delete)).await?;

        // Fire-and-forget: remove the directory if it exists. Missing is fine.
        if root.exists() {
            std::fs::remove_dir_all(&root).map_err(|source| ProjectError::Io {
                path: root.display().to_string(),
                source,
            })?;
        }
        Ok(())
    }
}

fn validate_language(code: &str) -> Result<(), ProjectError> {
    let trimmed = code.trim();
    if !(2..=5).contains(&trimmed.len())
        || !trimmed.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
    {
        return Err(ProjectError::InvalidLanguage {
            code: code.to_string(),
        });
    }
    Ok(())
}

fn create_project_layout(root: &Path) -> Result<(), ProjectError> {
    for sub in [
        "source",
        "audio",
        "transcription",
        "translation",
        "subtitles",
        "voices",
        "cache",
        "output",
        "logs",
    ] {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir).map_err(|source| ProjectError::Io {
            path: dir.display().to_string(),
            source,
        })?;
    }
    Ok(())
}

fn write_project_json(root: &Path, rec: &ProjectRecord) -> Result<(), ProjectError> {
    let path = root.join("project.json");
    let text = serde_json::to_string_pretty(rec).map_err(DbError::from)?;
    std::fs::write(&path, text).map_err(|source| ProjectError::Io {
        path: path.display().to_string(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use tokio::runtime::Runtime;

    fn temp_paths() -> AppPaths {
        let root = std::env::temp_dir().join(format!(
            "lmt-projtest-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let data_dir = root.join("data");
        let config_dir = root.join("config");
        let cache_dir = root.join("cache");
        let log_dir = cache_dir.join("logs");
        let projects_dir = data_dir.join("projects");
        let models_dir = data_dir.join("models");
        for d in [
            &data_dir,
            &config_dir,
            &cache_dir,
            &log_dir,
            &projects_dir,
            &models_dir,
        ] {
            std::fs::create_dir_all(d).unwrap();
        }
        AppPaths {
            db_path: data_dir.join("db.sqlite3"),
            config_file: config_dir.join("config.json"),
            data_dir,
            config_dir,
            cache_dir,
            log_dir,
            projects_dir,
            models_dir,
        }
    }

    #[test]
    fn create_open_delete_round_trip() {
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths.clone(), db);

        let summary = rt
            .block_on(svc.create(CreateProjectInput {
                name: "Movie X".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap();
        assert_eq!(summary.name, "Movie X");

        let list = rt.block_on(svc.list()).unwrap();
        assert_eq!(list.len(), 1);

        let opened = rt.block_on(svc.open(summary.id.clone())).unwrap();
        assert_eq!(opened.name, "Movie X");
        assert!(opened.last_opened_at.is_some());

        // project.json exists on disk
        let json = std::path::Path::new(&opened.root_path).join("project.json");
        assert!(json.exists());

        rt.block_on(svc.delete(summary.id)).unwrap();
        assert!(rt.block_on(svc.list()).unwrap().is_empty());
    }

    #[test]
    fn rejects_empty_name() {
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths, db);
        let err = rt
            .block_on(svc.create(CreateProjectInput {
                name: "   ".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap_err();
        assert!(matches!(err, ProjectError::EmptyName));
    }

    #[test]
    fn imports_source_by_reference_and_persists_fingerprint() {
        use std::io::Write;
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths.clone(), db);

        let summary = rt
            .block_on(svc.create(CreateProjectInput {
                name: "Ref Test".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap();

        // Fake .mkv source somewhere outside the project root.
        let ext_dir = std::env::temp_dir().join(format!("lmt-ext-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&ext_dir).unwrap();
        let src = ext_dir.join("movie.mkv");
        let mut f = std::fs::File::create(&src).unwrap();
        f.write_all(b"MKV\x1a\x45\xdf\xa3fake").unwrap();

        let result = rt
            .block_on(svc.import_media(ImportMediaInput {
                project_id: summary.id.clone(),
                source_path: src.display().to_string(),
                copy_into_project: false,
            }))
            .unwrap();

        assert!(matches!(result.mode, SourceImportMode::Reference));
        assert_eq!(result.source_media_path, src.display().to_string());
        assert_eq!(
            result.project.source_hash.as_deref(),
            Some(result.fingerprint.hash.as_str())
        );

        // Source file was NOT copied.
        let expected_copy =
            std::path::Path::new(&result.project.root_path).join("source/movie.mkv");
        assert!(!expected_copy.exists());
    }

    #[test]
    fn imports_source_by_copy_lands_in_project() {
        use std::io::Write;
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths.clone(), db);

        let summary = rt
            .block_on(svc.create(CreateProjectInput {
                name: "Copy Test".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap();

        let ext_dir = std::env::temp_dir().join(format!("lmt-ext-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&ext_dir).unwrap();
        let src = ext_dir.join("clip.mp4");
        let mut f = std::fs::File::create(&src).unwrap();
        f.write_all(b"MP4 test content").unwrap();

        let result = rt
            .block_on(svc.import_media(ImportMediaInput {
                project_id: summary.id.clone(),
                source_path: src.display().to_string(),
                copy_into_project: true,
            }))
            .unwrap();

        assert!(matches!(result.mode, SourceImportMode::Copy));
        let dest = std::path::Path::new(&result.project.root_path).join("source/clip.mp4");
        assert!(
            dest.exists(),
            "copy should have landed at {}",
            dest.display()
        );
        assert_eq!(result.source_media_path, dest.display().to_string());
    }

    #[test]
    fn rejects_unsupported_extension() {
        use std::io::Write;
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths, db);

        let summary = rt
            .block_on(svc.create(CreateProjectInput {
                name: "x".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap();

        let src = std::env::temp_dir().join(format!("lmt-bad-{}.flv", Uuid::new_v4()));
        let mut f = std::fs::File::create(&src).unwrap();
        f.write_all(b"nope").unwrap();

        let err = rt
            .block_on(svc.import_media(ImportMediaInput {
                project_id: summary.id.clone(),
                source_path: src.display().to_string(),
                copy_into_project: false,
            }))
            .unwrap_err();
        assert!(matches!(
            err,
            ProjectError::UnsupportedSourceExtension { .. }
        ));
    }

    #[test]
    fn rejects_missing_source() {
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths, db);
        let summary = rt
            .block_on(svc.create(CreateProjectInput {
                name: "y".into(),
                source_language: "en".into(),
                target_language: "vi".into(),
            }))
            .unwrap();
        let err = rt
            .block_on(svc.import_media(ImportMediaInput {
                project_id: summary.id.clone(),
                source_path: "/definitely/does/not/exist.mp4".into(),
                copy_into_project: false,
            }))
            .unwrap_err();
        assert!(matches!(err, ProjectError::SourceNotFound { .. }));
    }

    #[test]
    fn rejects_bad_language() {
        let rt = Runtime::new().unwrap();
        let paths = temp_paths();
        let db = Db::open_in_memory().unwrap();
        let svc = ProjectService::new(paths, db);
        let err = rt
            .block_on(svc.create(CreateProjectInput {
                name: "ok".into(),
                source_language: "english".into(),
                target_language: "vi".into(),
            }))
            .unwrap_err();
        assert!(matches!(err, ProjectError::InvalidLanguage { .. }));
    }
}
