//! In-memory registry of running jobs, indexed by job id.
//!
//! Only tracks the *live* cancel handle — durable job state (progress,
//! status, timestamps) lives in SQLite. On process crash the registry
//! is empty and `JobsRepo::reap_orphans` cleans the rows.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use thiserror::Error;

use crate::ffmpeg::extract::CancelToken;

use super::JobStage;

#[derive(Debug, Error)]
pub enum JobRegistryError {
    #[error("job `{0}` is not active")]
    NotFound(String),

    #[error("a job is already running for {stage:?} on project `{project_id}`")]
    Conflict { project_id: String, stage: JobStage },
}

#[derive(Debug, Clone)]
pub struct JobHandle {
    pub id: String,
    pub project_id: String,
    pub stage: JobStage,
    pub cancel: CancelToken,
}

#[derive(Debug, Default)]
pub struct JobRegistry {
    inner: RwLock<HashMap<String, JobHandle>>,
}

impl JobRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Reserve a slot for a new job. Fails with `Conflict` if any active
    /// job on the same project+stage already exists (Phase 2 policy:
    /// one extraction per project at a time).
    pub fn register(
        &self,
        id: String,
        project_id: String,
        stage: JobStage,
    ) -> Result<JobHandle, JobRegistryError> {
        let mut map = self.inner.write();
        if map
            .values()
            .any(|h| h.project_id == project_id && h.stage == stage)
        {
            return Err(JobRegistryError::Conflict { project_id, stage });
        }
        let handle = JobHandle {
            id: id.clone(),
            project_id,
            stage,
            cancel: CancelToken::new(),
        };
        map.insert(id, handle.clone());
        Ok(handle)
    }

    pub fn get(&self, id: &str) -> Option<JobHandle> {
        self.inner.read().get(id).cloned()
    }

    pub fn cancel(&self, id: &str) -> Result<(), JobRegistryError> {
        let handle = self
            .inner
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| JobRegistryError::NotFound(id.to_string()))?;
        handle.cancel.cancel();
        Ok(())
    }

    pub fn deregister(&self, id: &str) {
        self.inner.write().remove(id);
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.inner.read().keys().cloned().collect()
    }

    /// Phase 11 — cheap snapshot for the runtime-stats surface.
    /// Returns a cloned list of every live handle at read time; the
    /// underlying `RwLock` is held only long enough to copy the map.
    pub fn snapshot_all(&self) -> Vec<JobHandle> {
        self.inner.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_get_cancel_deregister() {
        let reg = JobRegistry::new();
        let h = reg
            .register("job-1".into(), "proj-1".into(), JobStage::ExtractAudio)
            .unwrap();
        assert!(!h.cancel.is_cancelled());

        let same = reg.get("job-1").unwrap();
        assert_eq!(same.project_id, "proj-1");

        reg.cancel("job-1").unwrap();
        assert!(reg.get("job-1").unwrap().cancel.is_cancelled());

        reg.deregister("job-1");
        assert!(reg.get("job-1").is_none());
    }

    #[test]
    fn same_project_stage_conflict() {
        let reg = JobRegistry::new();
        reg.register("a".into(), "p".into(), JobStage::ExtractAudio)
            .unwrap();
        let err = reg
            .register("b".into(), "p".into(), JobStage::ExtractAudio)
            .unwrap_err();
        assert!(matches!(err, JobRegistryError::Conflict { .. }));
    }

    #[test]
    fn cancel_missing_is_error() {
        let reg = JobRegistry::new();
        assert!(matches!(
            reg.cancel("nope"),
            Err(JobRegistryError::NotFound(_))
        ));
    }
}
