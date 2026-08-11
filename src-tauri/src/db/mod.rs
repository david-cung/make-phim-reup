//! Small SQLite persistence layer built on `rusqlite` (bundled).
//!
//! Sync `rusqlite` calls are wrapped in `tokio::task::spawn_blocking` at
//! the callsites (in `projects::service` etc.), so the async runtime never
//! blocks. The pool is a single `Mutex<Connection>` for now — the app's
//! workload is very light (metadata only). If it ever becomes contended
//! we can move to `r2d2` + `r2d2_sqlite` behind this same interface.

mod migrations;
pub mod models;
mod store;

pub use store::{Db, DbError, DbHandle};
