//! Embedded migration SQL. Kept as static strings so binaries can migrate
//! themselves without needing files at runtime.
//!
//! Rules:
//!   * Migrations are append-only. Never rewrite history — add a new one.
//!   * Every migration is wrapped in a transaction by the runner.
//!   * Version numbers are contiguous, starting at 1.

use rusqlite::{params, Connection, Transaction};

use super::DbError;

pub const MIGRATIONS: &[(i64, &str, &str)] = &[
    (1, "create_meta", SQL_001_CREATE_META),
    (2, "create_core_tables", SQL_002_CREATE_CORE),
    (3, "add_source_fingerprint", SQL_003_ADD_SOURCE_FINGERPRINT),
    (
        4,
        "add_project_model_config",
        SQL_004_ADD_PROJECT_MODEL_CONFIG,
    ),
];

pub fn run(conn: &mut Connection) -> Result<(), DbError> {
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(err_ctx("set WAL journal"))?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(err_ctx("enable foreign keys"))?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(err_ctx("set synchronous"))?;

    ensure_migrations_table(conn)?;

    let applied = applied_versions(conn)?;

    for (version, name, sql) in MIGRATIONS {
        if applied.contains(version) {
            continue;
        }
        let tx: Transaction = conn.transaction().map_err(err_ctx("start migration tx"))?;
        tx.execute_batch(sql).map_err(err_ctx(name))?;
        tx.execute(
            "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?1, ?2, datetime('now'))",
            params![version, name],
        )
        .map_err(err_ctx("record migration"))?;
        tx.commit().map_err(err_ctx("commit migration"))?;
        tracing::info!(version, name, "applied migration");
    }
    Ok(())
}

fn ensure_migrations_table(conn: &Connection) -> Result<(), DbError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
             version INTEGER PRIMARY KEY,
             name    TEXT NOT NULL,
             applied_at TEXT NOT NULL
         );",
    )
    .map_err(err_ctx("create schema_migrations"))
}

fn applied_versions(conn: &Connection) -> Result<Vec<i64>, DbError> {
    let mut stmt = conn
        .prepare("SELECT version FROM schema_migrations")
        .map_err(err_ctx("select applied versions"))?;
    let rows = stmt
        .query_map([], |r| r.get::<_, i64>(0))
        .map_err(err_ctx("query applied versions"))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(err_ctx("read applied version"))?);
    }
    Ok(out)
}

fn err_ctx(step: &'static str) -> impl Fn(rusqlite::Error) -> DbError {
    move |source| DbError::Migration {
        step: step.to_string(),
        source,
    }
}

const SQL_001_CREATE_META: &str = r#"
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
"#;

const SQL_002_CREATE_CORE: &str = r#"
CREATE TABLE IF NOT EXISTS projects (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    source_language   TEXT NOT NULL,
    target_language   TEXT NOT NULL,
    root_path         TEXT NOT NULL,
    source_media_path TEXT,
    status            TEXT NOT NULL,
    progress_json     TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_opened_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    progress      REAL NOT NULL DEFAULT 0,
    error_code    TEXT,
    error_message TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_project_status ON jobs(project_id, status);

CREATE TABLE IF NOT EXISTS subtitle_segments (
    id              INTEGER NOT NULL,
    project_id      TEXT    NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    source_text     TEXT    NOT NULL,
    translated_text TEXT,
    speaker_id      TEXT,
    voice_id        TEXT,
    status          TEXT    NOT NULL,
    PRIMARY KEY (project_id, id)
);

CREATE TABLE IF NOT EXISTS speakers (
    id               TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    display_name     TEXT NOT NULL,
    default_voice_id TEXT
);

CREATE TABLE IF NOT EXISTS voices (
    id       TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    locale   TEXT NOT NULL,
    gender   TEXT,
    path     TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,   -- stt | translation | tts
    name         TEXT NOT NULL,
    path         TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    installed_at TEXT NOT NULL
);
"#;

/// Phase 2: track how each project's source video was imported. `NULL`
/// means the project either hasn't imported a source yet, or was
/// migrated from a pre-Phase-2 DB.
const SQL_003_ADD_SOURCE_FINGERPRINT: &str = r#"
ALTER TABLE projects ADD COLUMN source_hash        TEXT;
ALTER TABLE projects ADD COLUMN source_size        INTEGER;
ALTER TABLE projects ADD COLUMN source_modified_at TEXT;
ALTER TABLE projects ADD COLUMN source_import_mode TEXT;   -- 'reference' | 'copy'
"#;

/// Phase 10: per-project selection of the AI models to use. Every
/// column is nullable so pre-Phase-10 projects keep their meaning
/// (fall back to the global settings default). No silent
/// substitutions: if the referenced model is missing at run time, the
/// per-stage service surfaces `MODEL_NOT_INSTALLED` and the UI shows
/// a "Model not installed" nudge.
const SQL_004_ADD_PROJECT_MODEL_CONFIG: &str = r#"
ALTER TABLE projects ADD COLUMN whisper_model     TEXT;
ALTER TABLE projects ADD COLUMN translation_model TEXT;
ALTER TABLE projects ADD COLUMN tts_engine        TEXT;
ALTER TABLE projects ADD COLUMN tts_voice_id      TEXT;
"#;
