//! Python worker lifecycle: spawn, JSON-RPC over stdio, health, restart.

pub mod protocol;
mod supervisor;

pub use protocol::{RpcError, RpcRequestId};
pub use supervisor::{
    detect_python_bin as supervisor_python_bin, detect_worker_root as supervisor_worker_root,
    EnvInfo, NotificationCallback, PingResponse, SubscriptionId, WorkerConfig, WorkerError,
    WorkerState, WorkerStatus, WorkerSupervisor,
};
