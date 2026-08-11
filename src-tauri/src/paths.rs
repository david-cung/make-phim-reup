//! OS-appropriate directories and safe path validation.
//!
//! Never hard-code `/Users/...` or `C:\Users\...`. All directories are
//! discovered through the `dirs` crate. Every path that ever touches
//! disk goes through [`validate_within`] to prevent traversal attacks.

use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

const APP_DIR: &str = "local-movie-translator";

#[derive(Debug, Error)]
pub enum PathError {
    #[error("could not determine {kind} directory for this operating system")]
    DirUnavailable { kind: &'static str },

    #[error("path is empty")]
    Empty,

    #[error("path contains an invalid component: {reason}")]
    InvalidComponent { reason: String },

    #[error("path `{candidate}` escapes allowed root `{root}`")]
    EscapesRoot { candidate: String, root: String },

    #[error("path `{path}` was expected to be absolute")]
    NotAbsolute { path: String },

    #[error("io error creating directory `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Immutable snapshot of the OS-specific directories the app owns.
#[derive(Debug, Clone, Serialize)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub log_dir: PathBuf,
    pub projects_dir: PathBuf,
    pub models_dir: PathBuf,
    pub db_path: PathBuf,
    pub config_file: PathBuf,
}

impl AppPaths {
    /// Discover OS-appropriate directories and ensure they exist.
    pub fn detect() -> Result<Self, PathError> {
        let data_dir = dirs::data_dir()
            .ok_or(PathError::DirUnavailable { kind: "data" })?
            .join(APP_DIR);
        let config_dir = dirs::config_dir()
            .ok_or(PathError::DirUnavailable { kind: "config" })?
            .join(APP_DIR);
        let cache_dir = dirs::cache_dir()
            .ok_or(PathError::DirUnavailable { kind: "cache" })?
            .join(APP_DIR);

        let log_dir = cache_dir.join("logs");
        let projects_dir = data_dir.join("projects");
        let models_dir = data_dir.join("models");
        let db_path = data_dir.join("db.sqlite3");
        let config_file = config_dir.join("config.json");

        for d in [
            &data_dir,
            &config_dir,
            &cache_dir,
            &log_dir,
            &projects_dir,
            &models_dir,
        ] {
            ensure_dir(d)?;
        }

        Ok(Self {
            data_dir,
            config_dir,
            cache_dir,
            log_dir,
            projects_dir,
            models_dir,
            db_path,
            config_file,
        })
    }

    /// All roots that are considered safe destinations for created files.
    pub fn allowed_write_roots(&self) -> Vec<&Path> {
        vec![
            self.data_dir.as_path(),
            self.cache_dir.as_path(),
            self.config_dir.as_path(),
        ]
    }
}

pub fn ensure_dir(dir: &Path) -> Result<(), PathError> {
    if dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|source| PathError::Io {
        path: dir.display().to_string(),
        source,
    })
}

/// Validate that `candidate` is safe: absolute, contains no `..` components,
/// and (after lexical normalisation) is inside `root`.
///
/// We intentionally do NOT call `canonicalize` because the candidate may not
/// exist yet. Symlink attacks are mitigated by only using this on paths we
/// ourselves construct from a validated root + trusted segments.
pub fn validate_within(root: &Path, candidate: &Path) -> Result<PathBuf, PathError> {
    if candidate.as_os_str().is_empty() {
        return Err(PathError::Empty);
    }
    if !candidate.is_absolute() {
        return Err(PathError::NotAbsolute {
            path: candidate.display().to_string(),
        });
    }
    let normalized = lexical_normalize(candidate)?;
    let root_norm = lexical_normalize(root)?;

    if !normalized.starts_with(&root_norm) {
        return Err(PathError::EscapesRoot {
            candidate: normalized.display().to_string(),
            root: root_norm.display().to_string(),
        });
    }
    Ok(normalized)
}

/// Purely lexical `..`/`.` resolution — no filesystem calls.
pub fn lexical_normalize(p: &Path) -> Result<PathBuf, PathError> {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return Err(PathError::InvalidComponent {
                        reason: "path attempts to escape its root using `..`".into(),
                    });
                }
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                out.push(c.as_os_str());
            }
        }
    }
    Ok(out)
}

/// A safe helper to build a project's on-disk root from a validated projects
/// directory and a UUID string.
pub fn project_root(projects_dir: &Path, project_id: &str) -> Result<PathBuf, PathError> {
    // UUIDs contain only [0-9a-f-]; refuse anything else.
    if !project_id
        .bytes()
        .all(|b| b.is_ascii_hexdigit() || b == b'-')
    {
        return Err(PathError::InvalidComponent {
            reason: format!("project id `{project_id}` is not a valid UUID"),
        });
    }
    validate_within(projects_dir, &projects_dir.join(project_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_relative() {
        let root = Path::new("/tmp/root");
        let err = validate_within(root, Path::new("subdir")).unwrap_err();
        assert!(matches!(err, PathError::NotAbsolute { .. }));
    }

    #[test]
    fn rejects_parent_escape() {
        let root = Path::new("/tmp/root");
        let err = validate_within(root, Path::new("/tmp/root/../etc/passwd")).unwrap_err();
        assert!(matches!(err, PathError::EscapesRoot { .. }));
    }

    #[test]
    fn allows_child() {
        let root = Path::new("/tmp/root");
        let out = validate_within(root, Path::new("/tmp/root/a/b/c")).unwrap();
        assert_eq!(out, PathBuf::from("/tmp/root/a/b/c"));
    }

    #[test]
    fn allows_dot_components() {
        let root = Path::new("/tmp/root");
        let out = validate_within(root, Path::new("/tmp/root/./a/./b")).unwrap();
        assert_eq!(out, PathBuf::from("/tmp/root/a/b"));
    }

    #[test]
    fn project_root_requires_uuid_charset() {
        let projects = Path::new("/tmp/root/projects");
        assert!(project_root(projects, "../etc").is_err());
        assert!(project_root(projects, "abc/def").is_err());
        assert!(project_root(projects, "3f2504e0-4f89-11d3-9a0c-0305e82c3301").is_ok());
    }
}
