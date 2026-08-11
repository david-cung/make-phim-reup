//! SQLite connection holder + typed queries for Phase 1 (projects only).

use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Row};
use thiserror::Error;
use tokio::task;

use super::migrations;
use super::models::{
    ProjectModelPatch, ProjectRecord, ProjectStatus, ProjectSummary, SourceImportMode,
};

#[derive(Debug, Error)]
pub enum DbError {
    #[error("row not found: {entity} id={id}")]
    NotFound { entity: &'static str, id: String },

    #[error("migration `{step}` failed: {source}")]
    Migration {
        step: String,
        #[source]
        source: rusqlite::Error,
    },

    #[error("sqlite error during {ctx}: {source}")]
    Sqlite {
        ctx: &'static str,
        #[source]
        source: rusqlite::Error,
    },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("db task join error: {0}")]
    Join(String),
}

fn sqlite_err(ctx: &'static str) -> impl Fn(rusqlite::Error) -> DbError {
    move |source| DbError::Sqlite { ctx, source }
}

/// Shared handle passed into every long-lived component. Cheap to clone.
pub type DbHandle = Arc<Db>;

pub struct Db {
    conn: Mutex<Connection>,
}

impl Db {
    pub fn open(path: &Path) -> Result<DbHandle, DbError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| DbError::Sqlite {
                ctx: "create db parent dir",
                source: rusqlite::Error::ToSqlConversionFailure(Box::new(e)),
            })?;
        }
        let mut conn = Connection::open(path).map_err(sqlite_err("open connection"))?;
        migrations::run(&mut conn)?;
        Ok(Arc::new(Self {
            conn: Mutex::new(conn),
        }))
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<DbHandle, DbError> {
        let mut conn = Connection::open_in_memory().map_err(sqlite_err("open in-memory"))?;
        migrations::run(&mut conn)?;
        Ok(Arc::new(Self {
            conn: Mutex::new(conn),
        }))
    }

    pub async fn run<T, F>(self: &Arc<Self>, f: F) -> Result<T, DbError>
    where
        T: Send + 'static,
        F: FnOnce(&Db) -> Result<T, DbError> + Send + 'static,
    {
        let this = self.clone();
        task::spawn_blocking(move || f(&this))
            .await
            .map_err(|e| DbError::Join(e.to_string()))?
    }

    /// Escape hatch for domain repos that live outside this module and
    /// prefer to run their own SQL. Keep the returned guard's scope
    /// tight — every millisecond it's held blocks other DB operations.
    pub fn raw(&self) -> &Mutex<Connection> {
        &self.conn
    }

    // ---------- projects ----------

    pub fn insert_project(&self, rec: &ProjectRecord) -> Result<(), DbError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO projects
                (id, name, source_language, target_language,
                 root_path, source_media_path, status, progress_json,
                 created_at, updated_at, last_opened_at,
                 source_hash, source_size, source_modified_at, source_import_mode,
                 whisper_model, translation_model, tts_engine, tts_voice_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18, ?19)",
            params![
                rec.id,
                rec.name,
                rec.source_language,
                rec.target_language,
                rec.root_path,
                rec.source_media_path,
                rec.status.as_str(),
                serde_json::to_string(&rec.progress)?,
                rec.created_at.to_rfc3339(),
                rec.updated_at.to_rfc3339(),
                rec.last_opened_at.map(|d| d.to_rfc3339()),
                rec.source_hash,
                rec.source_size,
                rec.source_modified_at.map(|d| d.to_rfc3339()),
                rec.source_import_mode.map(|m| m.as_str()),
                rec.whisper_model,
                rec.translation_model,
                rec.tts_engine,
                rec.tts_voice_id,
            ],
        )
        .map_err(sqlite_err("insert project"))?;
        Ok(())
    }

    /// Overwrite the per-project model selection. Only fields
    /// present in the patch are touched; `None` clears the value.
    pub fn set_project_models(&self, id: &str, patch: &ProjectModelPatch) -> Result<(), DbError> {
        let mut current = self.get_project(id)?;
        if let Some(v) = patch.whisper_model.clone() {
            current.whisper_model = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.translation_model.clone() {
            current.translation_model = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.tts_engine.clone() {
            current.tts_engine = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(v) = patch.tts_voice_id.clone() {
            current.tts_voice_id = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        }
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        let n = conn
            .execute(
                "UPDATE projects SET
                    whisper_model = ?1,
                    translation_model = ?2,
                    tts_engine = ?3,
                    tts_voice_id = ?4,
                    updated_at = ?5
                 WHERE id = ?6",
                params![
                    current.whisper_model,
                    current.translation_model,
                    current.tts_engine,
                    current.tts_voice_id,
                    now,
                    id,
                ],
            )
            .map_err(sqlite_err("set project models"))?;
        if n == 0 {
            return Err(DbError::NotFound {
                entity: "project",
                id: id.to_string(),
            });
        }
        Ok(())
    }

    /// Overwrite the source-video fields of a project and bump `updated_at`.
    /// Called after a successful `import_media`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_project_source(
        &self,
        id: &str,
        source_media_path: &str,
        source_hash: &str,
        source_size: i64,
        source_modified_at: DateTime<Utc>,
        import_mode: SourceImportMode,
    ) -> Result<(), DbError> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        let n = conn
            .execute(
                "UPDATE projects SET
                    source_media_path = ?1,
                    source_hash = ?2,
                    source_size = ?3,
                    source_modified_at = ?4,
                    source_import_mode = ?5,
                    updated_at = ?6
                 WHERE id = ?7",
                params![
                    source_media_path,
                    source_hash,
                    source_size,
                    source_modified_at.to_rfc3339(),
                    import_mode.as_str(),
                    now,
                    id,
                ],
            )
            .map_err(sqlite_err("set project source"))?;
        if n == 0 {
            return Err(DbError::NotFound {
                entity: "project",
                id: id.to_string(),
            });
        }
        Ok(())
    }

    pub fn list_project_summaries(&self) -> Result<Vec<ProjectSummary>, DbError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, name, source_language, target_language, status,
                        created_at, updated_at, last_opened_at
                 FROM projects
                 ORDER BY datetime(updated_at) DESC",
            )
            .map_err(sqlite_err("prepare list projects"))?;
        let rows = stmt
            .query_map([], row_to_summary)
            .map_err(sqlite_err("query list projects"))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(sqlite_err("read project row"))?);
        }
        Ok(out)
    }

    pub fn get_project(&self, id: &str) -> Result<ProjectRecord, DbError> {
        let conn = self.conn.lock();
        let raw = conn
            .query_row(
                "SELECT id, name, source_language, target_language, root_path,
                        source_media_path, status, progress_json,
                        created_at, updated_at, last_opened_at,
                        source_hash, source_size, source_modified_at, source_import_mode,
                        whisper_model, translation_model, tts_engine, tts_voice_id
                 FROM projects WHERE id = ?1",
                params![id],
                row_to_raw_project,
            )
            .optional()
            .map_err(sqlite_err("get project"))?
            .ok_or(DbError::NotFound {
                entity: "project",
                id: id.to_string(),
            })?;
        raw.into_record()
    }

    pub fn touch_project_opened(&self, id: &str, now: DateTime<Utc>) -> Result<(), DbError> {
        let conn = self.conn.lock();
        let n = conn
            .execute(
                "UPDATE projects SET last_opened_at = ?1, updated_at = ?2 WHERE id = ?3",
                params![now.to_rfc3339(), now.to_rfc3339(), id],
            )
            .map_err(sqlite_err("touch project"))?;
        if n == 0 {
            return Err(DbError::NotFound {
                entity: "project",
                id: id.to_string(),
            });
        }
        Ok(())
    }

    pub fn delete_project(&self, id: &str) -> Result<(), DbError> {
        let conn = self.conn.lock();
        let n = conn
            .execute("DELETE FROM projects WHERE id = ?1", params![id])
            .map_err(sqlite_err("delete project"))?;
        if n == 0 {
            return Err(DbError::NotFound {
                entity: "project",
                id: id.to_string(),
            });
        }
        Ok(())
    }
}

// ---- row mapping ----

fn row_to_summary(row: &Row<'_>) -> rusqlite::Result<ProjectSummary> {
    Ok(ProjectSummary {
        id: row.get(0)?,
        name: row.get(1)?,
        source_language: row.get(2)?,
        target_language: row.get(3)?,
        status: ProjectStatus::parse(&row.get::<_, String>(4)?).unwrap_or(ProjectStatus::Created),
        created_at: parse_dt(row, 5)?,
        updated_at: parse_dt(row, 6)?,
        last_opened_at: parse_dt_opt(row, 7)?,
    })
}

/// Intermediate row shape so we can defer JSON parsing to outside the
/// `rusqlite::Result` closure (rusqlite errors don't carry a good place
/// for serde failures).
struct RawProjectRow {
    id: String,
    name: String,
    source_language: String,
    target_language: String,
    root_path: String,
    source_media_path: Option<String>,
    status: ProjectStatus,
    progress_json: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    last_opened_at: Option<DateTime<Utc>>,
    source_hash: Option<String>,
    source_size: Option<i64>,
    source_modified_at: Option<DateTime<Utc>>,
    source_import_mode: Option<SourceImportMode>,
    whisper_model: Option<String>,
    translation_model: Option<String>,
    tts_engine: Option<String>,
    tts_voice_id: Option<String>,
}

impl RawProjectRow {
    fn into_record(self) -> Result<ProjectRecord, DbError> {
        let progress = serde_json::from_str(&self.progress_json)?;
        Ok(ProjectRecord {
            id: self.id,
            name: self.name,
            source_language: self.source_language,
            target_language: self.target_language,
            root_path: self.root_path,
            source_media_path: self.source_media_path,
            status: self.status,
            progress,
            created_at: self.created_at,
            updated_at: self.updated_at,
            last_opened_at: self.last_opened_at,
            source_hash: self.source_hash,
            source_size: self.source_size,
            source_modified_at: self.source_modified_at,
            source_import_mode: self.source_import_mode,
            whisper_model: self.whisper_model,
            translation_model: self.translation_model,
            tts_engine: self.tts_engine,
            tts_voice_id: self.tts_voice_id,
        })
    }
}

fn row_to_raw_project(row: &Row<'_>) -> rusqlite::Result<RawProjectRow> {
    let import_mode = row
        .get::<_, Option<String>>(14)?
        .as_deref()
        .and_then(SourceImportMode::parse);
    Ok(RawProjectRow {
        id: row.get(0)?,
        name: row.get(1)?,
        source_language: row.get(2)?,
        target_language: row.get(3)?,
        root_path: row.get(4)?,
        source_media_path: row.get(5)?,
        status: ProjectStatus::parse(&row.get::<_, String>(6)?).unwrap_or(ProjectStatus::Created),
        progress_json: row.get(7)?,
        created_at: parse_dt(row, 8)?,
        updated_at: parse_dt(row, 9)?,
        last_opened_at: parse_dt_opt(row, 10)?,
        source_hash: row.get(11)?,
        source_size: row.get(12)?,
        source_modified_at: parse_dt_opt(row, 13)?,
        source_import_mode: import_mode,
        whisper_model: row.get(15)?,
        translation_model: row.get(16)?,
        tts_engine: row.get(17)?,
        tts_voice_id: row.get(18)?,
    })
}

fn parse_dt(row: &Row<'_>, idx: usize) -> rusqlite::Result<DateTime<Utc>> {
    let s: String = row.get(idx)?;
    DateTime::parse_from_rfc3339(&s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(idx, rusqlite::types::Type::Text, Box::new(e))
        })
}

fn parse_dt_opt(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<DateTime<Utc>>> {
    let s: Option<String> = row.get(idx)?;
    Ok(match s {
        Some(v) => Some(
            DateTime::parse_from_rfc3339(&v)
                .map(|d| d.with_timezone(&Utc))
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        idx,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?,
        ),
        None => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_record(id: &str, name: &str) -> ProjectRecord {
        let now = Utc::now();
        ProjectRecord {
            id: id.to_string(),
            name: name.to_string(),
            source_language: "en".into(),
            target_language: "vi".into(),
            root_path: "/tmp/example".into(),
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
        }
    }

    #[test]
    fn insert_and_read_back() {
        let db = Db::open_in_memory().unwrap();
        let id = Uuid::new_v4().to_string();
        let rec = make_record(&id, "hello");
        db.insert_project(&rec).unwrap();

        let got = db.get_project(&id).unwrap();
        assert_eq!(got.name, "hello");
        assert_eq!(got.source_language, "en");
        assert_eq!(got.target_language, "vi");
    }

    #[test]
    fn list_orders_by_updated_desc() {
        let db = Db::open_in_memory().unwrap();
        let mut a = make_record(&Uuid::new_v4().to_string(), "a");
        a.updated_at = Utc::now() - chrono::Duration::minutes(1);
        let b = make_record(&Uuid::new_v4().to_string(), "b");
        db.insert_project(&a).unwrap();
        db.insert_project(&b).unwrap();

        let list = db.list_project_summaries().unwrap();
        assert_eq!(list[0].name, "b");
        assert_eq!(list[1].name, "a");
    }

    #[test]
    fn delete_missing_yields_not_found() {
        let db = Db::open_in_memory().unwrap();
        let err = db.delete_project("missing").unwrap_err();
        assert!(matches!(err, DbError::NotFound { .. }));
    }

    #[test]
    fn set_project_source_persists_new_fields() {
        let db = Db::open_in_memory().unwrap();
        let id = Uuid::new_v4().to_string();
        db.insert_project(&make_record(&id, "srcy")).unwrap();

        db.set_project_source(
            &id,
            "/tmp/movie.mkv",
            "sha256:deadbeef",
            9_999,
            Utc::now(),
            SourceImportMode::Reference,
        )
        .unwrap();

        let got = db.get_project(&id).unwrap();
        assert_eq!(got.source_media_path.as_deref(), Some("/tmp/movie.mkv"));
        assert_eq!(got.source_hash.as_deref(), Some("sha256:deadbeef"));
        assert_eq!(got.source_size, Some(9_999));
        assert!(matches!(
            got.source_import_mode,
            Some(SourceImportMode::Reference)
        ));
    }
}
