//! DB-facing helpers for the `jobs` table.
//!
//! `Db` from `crate::db::store` owns the actual connection. Callers use
//! this repo from inside `db.run(|d| ...)` closures.

use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Row};

use crate::db::{Db, DbError};

use super::{JobSnapshot, JobStage, JobStatus};

/// Row-level representation (private to this module).
pub struct JobRow {
    pub id: String,
    pub project_id: String,
    pub stage: String,
    pub status: String,
    pub progress: f32,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub struct JobsRepo;

impl JobsRepo {
    pub fn get(db: &Db, id: &str) -> Result<Option<JobSnapshot>, DbError> {
        let conn = db.raw().lock();
        conn.query_row(
            "SELECT id, project_id, stage, status, progress,
                    error_code, error_message,
                    created_at, started_at, completed_at
             FROM jobs
             WHERE id = ?1
             LIMIT 1",
            params![id],
            row_to_snapshot,
        )
        .optional()
        .map_err(|source| DbError::Sqlite {
            ctx: "get job",
            source,
        })
    }

    pub fn insert(db: &Db, snap: &JobSnapshot) -> Result<(), DbError> {
        let conn = db.raw().lock();
        conn.execute(
            "INSERT INTO jobs
                (id, project_id, stage, status, progress,
                 error_code, error_message,
                 created_at, started_at, completed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                snap.id,
                snap.project_id,
                snap.stage.as_str(),
                snap.status.as_str(),
                snap.progress,
                snap.error_code,
                snap.error_message,
                snap.created_at.to_rfc3339(),
                snap.started_at.map(|d| d.to_rfc3339()),
                snap.completed_at.map(|d| d.to_rfc3339()),
            ],
        )
        .map_err(|source| DbError::Sqlite {
            ctx: "insert job",
            source,
        })?;
        Ok(())
    }

    pub fn update_status(
        db: &Db,
        id: &str,
        status: JobStatus,
        progress: Option<f32>,
        error_code: Option<&str>,
        error_message: Option<&str>,
    ) -> Result<(), DbError> {
        let conn = db.raw().lock();
        let now = Utc::now();
        let started = if matches!(status, JobStatus::Running) {
            Some(now.to_rfc3339())
        } else {
            None
        };
        let completed = if status.is_terminal() {
            Some(now.to_rfc3339())
        } else {
            None
        };
        let n = conn
            .execute(
                "UPDATE jobs SET
                status = ?1,
                progress = COALESCE(?2, progress),
                error_code = ?3,
                error_message = ?4,
                started_at = COALESCE(started_at, ?5),
                completed_at = COALESCE(completed_at, ?6)
             WHERE id = ?7",
                params![
                    status.as_str(),
                    progress,
                    error_code,
                    error_message,
                    started,
                    completed,
                    id,
                ],
            )
            .map_err(|source| DbError::Sqlite {
                ctx: "update job",
                source,
            })?;
        if n == 0 {
            return Err(DbError::NotFound {
                entity: "job",
                id: id.to_string(),
            });
        }
        Ok(())
    }

    pub fn update_progress(db: &Db, id: &str, progress: f32) -> Result<(), DbError> {
        let conn = db.raw().lock();
        conn.execute(
            "UPDATE jobs SET progress = ?1 WHERE id = ?2",
            params![progress, id],
        )
        .map_err(|source| DbError::Sqlite {
            ctx: "update job progress",
            source,
        })?;
        Ok(())
    }

    pub fn latest_by_stage(
        db: &Db,
        project_id: &str,
        stage: JobStage,
    ) -> Result<Option<JobSnapshot>, DbError> {
        let conn = db.raw().lock();
        conn.query_row(
            "SELECT id, project_id, stage, status, progress,
                    error_code, error_message,
                    created_at, started_at, completed_at
             FROM jobs
             WHERE project_id = ?1 AND stage = ?2
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
            params![project_id, stage.as_str()],
            row_to_snapshot,
        )
        .optional()
        .map_err(|source| DbError::Sqlite {
            ctx: "latest job by stage",
            source,
        })
    }

    pub fn list_active(db: &Db, project_id: &str) -> Result<Vec<JobSnapshot>, DbError> {
        let conn = db.raw().lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, stage, status, progress,
                        error_code, error_message,
                        created_at, started_at, completed_at
                 FROM jobs
                 WHERE project_id = ?1 AND status IN ('queued', 'running', 'paused')
                 ORDER BY datetime(created_at) ASC",
            )
            .map_err(|source| DbError::Sqlite {
                ctx: "prepare list_active",
                source,
            })?;
        let rows = stmt
            .query_map(params![project_id], row_to_snapshot)
            .map_err(|source| DbError::Sqlite {
                ctx: "query list_active",
                source,
            })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|source| DbError::Sqlite {
                ctx: "read list_active row",
                source,
            })?);
        }
        Ok(out)
    }

    /// Reap jobs that were marked "running" but the process no longer
    /// exists (typical after a crash). Called once at startup.
    pub fn reap_orphans(db: &Db) -> Result<usize, DbError> {
        let conn = db.raw().lock();
        let n = conn
            .execute(
                "UPDATE jobs
             SET status = 'failed',
                 error_code = COALESCE(error_code, 'JOB_ORPHANED'),
                 error_message = COALESCE(error_message, 'process no longer running'),
                 completed_at = COALESCE(completed_at, datetime('now'))
             WHERE status IN ('queued', 'running', 'paused')",
                [],
            )
            .map_err(|source| DbError::Sqlite {
                ctx: "reap orphans",
                source,
            })?;
        Ok(n)
    }

    /// Phase 12 — every job the last `reap_orphans` sweep marked as
    /// interrupted, most recent first. The Dashboard uses this to
    /// show a "resume from where you left off" hint after a crash.
    pub fn list_orphaned(db: &Db) -> Result<Vec<JobSnapshot>, DbError> {
        let conn = db.raw().lock();
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, stage, status, progress,
                        error_code, error_message,
                        created_at, started_at, completed_at
                 FROM jobs
                 WHERE error_code = 'JOB_ORPHANED'
                 ORDER BY datetime(COALESCE(completed_at, created_at)) DESC
                 LIMIT 50",
            )
            .map_err(|source| DbError::Sqlite {
                ctx: "prepare list_orphaned",
                source,
            })?;
        let rows = stmt
            .query_map([], row_to_snapshot)
            .map_err(|source| DbError::Sqlite {
                ctx: "query list_orphaned",
                source,
            })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|source| DbError::Sqlite {
                ctx: "read list_orphaned row",
                source,
            })?);
        }
        Ok(out)
    }
}

fn row_to_snapshot(row: &Row<'_>) -> rusqlite::Result<JobSnapshot> {
    let stage_str: String = row.get(2)?;
    let status_str: String = row.get(3)?;
    let created_at = parse_dt(row, 7)?;
    let started_at = parse_dt_opt(row, 8)?;
    let completed_at = parse_dt_opt(row, 9)?;
    Ok(JobSnapshot {
        id: row.get(0)?,
        project_id: row.get(1)?,
        stage: JobStage::parse(&stage_str).unwrap_or(JobStage::ExtractAudio),
        status: JobStatus::parse(&status_str).unwrap_or(JobStatus::Failed),
        progress: row.get::<_, f64>(4)? as f32,
        error_code: row.get(5)?,
        error_message: row.get(6)?,
        created_at,
        started_at,
        completed_at,
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
    use crate::db::Db;

    fn sample_snapshot(id: &str, project_id: &str) -> JobSnapshot {
        let now = Utc::now();
        JobSnapshot {
            id: id.into(),
            project_id: project_id.into(),
            stage: JobStage::Transcribe,
            status: JobStatus::Running,
            progress: 0.0,
            error_code: None,
            error_message: None,
            created_at: now,
            started_at: Some(now),
            completed_at: None,
        }
    }

    /// Regression test — the `jobs.project_id` column carries a
    /// `NOT NULL REFERENCES projects(id)` FK, so any code path that
    /// tries to persist a job with an empty (or missing) project id
    /// blows up. Model-download jobs used to hit this: they are
    /// global, not project-scoped, and therefore *must* stay
    /// in-memory. If somebody re-introduces `JobsRepo::insert(...)`
    /// for a download job, this test will fail loudly instead of
    /// only surfacing at runtime as "Cannot open project ... FOREIGN
    /// KEY constraint failed".
    #[test]
    fn insert_rejects_jobs_without_a_real_project() {
        let db = Db::open_in_memory().unwrap();
        let snap = sample_snapshot("job_download_1", "");
        let err = JobsRepo::insert(&db, &snap).unwrap_err();
        let msg = format!("{err:?}").to_lowercase();
        assert!(
            msg.contains("foreign key") || msg.contains("constraint"),
            "expected FK violation, got {err:?}",
        );
    }

    /// Companion positive test: once a matching project row exists,
    /// insert + update round-trip cleanly. Documents the intended
    /// contract that `JobsRepo::insert` is *only* for project-scoped
    /// jobs.
    #[test]
    fn insert_succeeds_for_project_scoped_job() {
        let db = Db::open_in_memory().unwrap();
        // Bypass the projects service and drop a bare row in
        // directly — we only need the FK target to satisfy the
        // constraint for this DB-level test.
        {
            let conn = db.raw().lock();
            let now = Utc::now().to_rfc3339();
            conn.execute(
                "INSERT INTO projects
                    (id, name, source_language, target_language, root_path,
                     source_media_path, status, progress_json,
                     created_at, updated_at)
                 VALUES (?1, 'test', 'en', 'vi', '/tmp/p', NULL, 'draft',
                         '{}', ?2, ?2)",
                params!["p1", now],
            )
            .unwrap();
        }
        JobsRepo::insert(&db, &sample_snapshot("job_stt_1", "p1")).unwrap();
        JobsRepo::update_status(
            &db,
            "job_stt_1",
            JobStatus::Completed,
            Some(1.0),
            None,
            None,
        )
        .unwrap();
        let latest = JobsRepo::latest_by_stage(&db, "p1", JobStage::Transcribe)
            .unwrap()
            .expect("latest job should exist");
        assert_eq!(latest.id, "job_stt_1");
        assert_eq!(latest.status, JobStatus::Completed);
    }
}
