//! Project domain: creating on-disk layout + DB rows, opening, deleting.

pub mod service;

pub use service::{
    CreateProjectInput, ImportMediaInput, ImportMediaResult, ProjectError, ProjectService,
};
