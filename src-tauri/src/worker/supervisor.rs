//! Supervises a long-lived Python worker process.
//!
//! Responsibilities:
//!   * Locate & spawn the worker with a stable command line (no shell).
//!   * Read line-delimited JSON-RPC responses off stdout and route them to
//!     the awaiting request future.
//!   * Merge stderr lines into the app log tagged as `worker`.
//!   * Restart the worker if it dies, with an exponential back-off cap.
//!   * Emit `worker://status` events to the frontend on every transition.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::{AppHandle, Emitter};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use uuid::Uuid;

use super::protocol::{IncomingNotification, RpcError, RpcFrame, RpcNotification, RpcRequest};

const WORKER_MODULE: &str = "movie_translator_worker";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const RESTART_MIN_DELAY: Duration = Duration::from_millis(500);
const RESTART_MAX_DELAY: Duration = Duration::from_secs(15);

/// Callback signature for notification subscribers.
pub type NotificationCallback = Arc<dyn Fn(&str, &serde_json::Value) + Send + Sync + 'static>;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("worker is not running")]
    NotRunning,

    #[error("worker RPC timed out after {}s", .0.as_secs())]
    Timeout(Duration),

    #[error("worker responded with error: {0}")]
    Rpc(#[from] RpcError),

    #[error("failed to spawn worker: {source}")]
    Spawn {
        #[source]
        source: std::io::Error,
    },

    #[error("worker io: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },

    #[error("worker protocol error: {msg}")]
    Protocol { msg: String },

    #[error("worker shutdown error: {msg}")]
    Shutdown { msg: String },

    #[error("worker request was cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Starting,
    Running,
    Stopped,
    Crashed,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerStatus {
    pub state: WorkerState,
    pub pid: Option<u32>,
    pub uptime_ms: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PingResponse {
    pub pong: bool,
    pub pid: u32,
    pub uptime_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvInfo {
    pub python: String,
    pub platform: String,
    pub ffmpeg_available: bool,
    pub ffmpeg_version: Option<String>,
    pub cpu_count: u32,
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub python_bin: PathBuf,
    pub worker_root: PathBuf,
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub log_level: String,
    pub app_version: String,
    /// Phase 11 — CPU threads hint passed to inference engines. `None`
    /// means "let the engine pick". Wired into llama.cpp and (where
    /// supported) faster-whisper's `cpu_threads` argument.
    pub cpu_threads: Option<u32>,
    /// Phase 11 — allow GPU/Metal back-ends when the engine and OS
    /// support them. Effectively a capability gate; if the machine
    /// can't do GPU inference the worker silently falls back to CPU.
    pub gpu_acceleration: bool,
}

type PendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<Result<serde_json::Value, RpcError>>>>>;

type SubscriberMap = Arc<Mutex<HashMap<u64, (String, NotificationCallback)>>>;

/// Internal per-run state for a spawned worker child.
struct RunningChild {
    child: Child,
    stdin: ChildStdin,
    started_at: Instant,
    _stdout_task: JoinHandle<()>,
    _stderr_task: JoinHandle<()>,
}

pub struct WorkerSupervisor {
    cfg: WorkerConfig,
    app: AppHandle,
    counter: AtomicU64,
    sub_counter: AtomicU64,
    child: tokio::sync::Mutex<Option<RunningChild>>,
    state: Mutex<WorkerStatus>,
    pending: PendingMap,
    subscribers: SubscriberMap,
    stop_signal: Arc<Notify>,
    supervisor_task: Mutex<Option<JoinHandle<()>>>,
}

impl WorkerSupervisor {
    pub fn new(cfg: WorkerConfig, app: AppHandle) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            app,
            counter: AtomicU64::new(0),
            sub_counter: AtomicU64::new(0),
            child: tokio::sync::Mutex::new(None),
            state: Mutex::new(WorkerStatus {
                state: WorkerState::Stopped,
                pid: None,
                uptime_ms: 0,
                last_error: None,
            }),
            pending: Arc::new(Mutex::new(HashMap::new())),
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            stop_signal: Arc::new(Notify::new()),
            supervisor_task: Mutex::new(None),
        })
    }

    /// Kick off the supervisor loop. The first `spawn` happens inside it so
    /// startup failures surface via `worker://status` events instead of
    /// crashing app bootstrap.
    pub fn start(self: &Arc<Self>) {
        let this = self.clone();
        let handle = tokio::spawn(async move {
            this.run_supervisor_loop().await;
        });
        *self.supervisor_task.lock() = Some(handle);
    }

    async fn run_supervisor_loop(self: Arc<Self>) {
        let mut delay = RESTART_MIN_DELAY;
        loop {
            self.update_state(WorkerState::Starting, None, None);

            match self.spawn_child().await {
                Ok(()) => {
                    delay = RESTART_MIN_DELAY;

                    if let Err(err) = self.initialize().await {
                        tracing::error!(?err, "worker initialize failed");
                        self.update_state(WorkerState::Crashed, None, Some(err.to_string()));
                        self.terminate_child().await;
                    } else {
                        let pid = self.current_pid().await;
                        self.update_state(WorkerState::Running, pid, None);
                        tracing::info!(?pid, "worker running");
                        self.wait_for_exit().await;
                        // Fall through to restart unless stop was requested.
                    }
                }
                Err(err) => {
                    tracing::error!(%err, "worker spawn failed");
                    self.update_state(WorkerState::Crashed, None, Some(err.to_string()));
                }
            }

            self.fail_all_pending("worker exited");

            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = self.stop_signal.notified() => {
                    self.update_state(WorkerState::Stopped, None, None);
                    return;
                }
            }
            delay = std::cmp::min(delay.saturating_mul(2), RESTART_MAX_DELAY);
        }
    }

    async fn spawn_child(self: &Arc<Self>) -> Result<(), WorkerError> {
        let mut cmd = Command::new(&self.cfg.python_bin);
        cmd.arg("-u")
            .arg("-m")
            .arg(WORKER_MODULE)
            .env("PYTHONPATH", self.cfg.worker_root.join("src"))
            .env("LMT_WORKER_ROOT", &self.cfg.worker_root)
            .env("LMT_WORKER_LOG_LEVEL", self.cfg.log_level.to_uppercase())
            .env("LMT_DATA_DIR", &self.cfg.data_dir)
            .env("LMT_APP_VERSION", &self.cfg.app_version)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        tracing::debug!(python = %self.cfg.python_bin.display(), "spawning worker");

        let mut child = cmd
            .spawn()
            .map_err(|source| WorkerError::Spawn { source })?;
        let stdin = child.stdin.take().ok_or(WorkerError::Protocol {
            msg: "no stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or(WorkerError::Protocol {
            msg: "no stdout".into(),
        })?;
        let stderr = child.stderr.take().ok_or(WorkerError::Protocol {
            msg: "no stderr".into(),
        })?;

        let pending = self.pending.clone();
        let subscribers = self.subscribers.clone();
        let stdout_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        match RpcFrame::parse_line(line) {
                            Ok(RpcFrame::Response(resp)) => {
                                let sender = pending.lock().remove(&resp.id);
                                if let Some(tx) = sender {
                                    let payload = if let Some(err) = resp.error {
                                        Err(err)
                                    } else {
                                        Ok(resp.result.unwrap_or(serde_json::Value::Null))
                                    };
                                    let _ = tx.send(payload);
                                } else {
                                    tracing::warn!(id = %resp.id, "worker response for unknown request");
                                }
                            }
                            Ok(RpcFrame::Notification(notif)) => {
                                dispatch_notification(&subscribers, &notif);
                            }
                            Err(err) => {
                                tracing::warn!(%err, line = %line, "invalid RPC line from worker");
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        tracing::warn!(%err, "worker stdout read error");
                        break;
                    }
                }
            }
        });

        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "worker", "{line}");
            }
        });

        *self.child.lock().await = Some(RunningChild {
            child,
            stdin,
            started_at: Instant::now(),
            _stdout_task: stdout_task,
            _stderr_task: stderr_task,
        });
        Ok(())
    }

    async fn current_pid(&self) -> Option<u32> {
        let guard = self.child.lock().await;
        guard.as_ref().and_then(|c| c.child.id())
    }

    async fn wait_for_exit(self: &Arc<Self>) {
        loop {
            let status = {
                let mut guard = self.child.lock().await;
                let Some(rc) = guard.as_mut() else {
                    return;
                };
                match rc.child.try_wait() {
                    Ok(Some(s)) => Some(s),
                    Ok(None) => None,
                    Err(err) => {
                        tracing::warn!(%err, "worker try_wait failed");
                        None
                    }
                }
            };
            if let Some(s) = status {
                tracing::warn!(code = ?s.code(), success = s.success(), "worker exited");
                *self.child.lock().await = None;
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    async fn terminate_child(&self) {
        let mut guard = self.child.lock().await;
        if let Some(rc) = guard.as_mut() {
            let _ = rc.child.start_kill();
            let _ = rc.child.wait().await;
        }
        *guard = None;
    }

    fn fail_all_pending(&self, reason: &'static str) {
        let mut map = self.pending.lock();
        for (_id, tx) in map.drain() {
            let _ = tx.send(Err(RpcError {
                code: "WORKER_GONE".into(),
                message: reason.into(),
                data: None,
            }));
        }
    }

    async fn initialize(self: &Arc<Self>) -> Result<(), WorkerError> {
        self.request(
            "initialize",
            json!({
                "app_version": self.cfg.app_version,
                "log_level": self.cfg.log_level,
                "data_root": self.cfg.data_dir,
                "models_root": self.cfg.models_dir,
                // Phase 11 — advanced performance knobs. The worker
                // stashes these on `handlers._PERF` and providers read
                // them at model-load time.
                "perf": {
                    "cpu_threads": self.cfg.cpu_threads,
                    "gpu_acceleration": self.cfg.gpu_acceleration,
                },
            }),
        )
        .await?;
        Ok(())
    }

    pub async fn ping(self: &Arc<Self>) -> Result<PingResponse, WorkerError> {
        let v = self.request("ping", json!({})).await?;
        serde_json::from_value(v).map_err(|e| WorkerError::Protocol { msg: e.to_string() })
    }

    pub async fn env_info(self: &Arc<Self>) -> Result<EnvInfo, WorkerError> {
        let v = self.request("env_info", json!({})).await?;
        serde_json::from_value(v).map_err(|e| WorkerError::Protocol { msg: e.to_string() })
    }

    /// Phase 10 — re-run `initialize` on the running worker with a
    /// new models_root. Handlers pick up the new directory
    /// immediately, no worker restart needed.
    pub async fn reinitialize_models_root(
        self: &Arc<Self>,
        models_root: PathBuf,
    ) -> Result<(), WorkerError> {
        self.request(
            "initialize",
            json!({
                "app_version": self.cfg.app_version,
                "log_level": self.cfg.log_level,
                "data_root": self.cfg.data_dir,
                "models_root": models_root,
                "perf": {
                    "cpu_threads": self.cfg.cpu_threads,
                    "gpu_acceleration": self.cfg.gpu_acceleration,
                },
            }),
        )
        .await?;
        Ok(())
    }

    /// Phase 11 — push updated performance knobs (CPU threads / GPU
    /// acceleration) without restarting the worker. Providers reload
    /// their model on the next call so the new values take effect.
    pub async fn reinitialize_perf(
        self: &Arc<Self>,
        cpu_threads: Option<u32>,
        gpu_acceleration: bool,
    ) -> Result<(), WorkerError> {
        self.request(
            "initialize",
            json!({
                "app_version": self.cfg.app_version,
                "log_level": self.cfg.log_level,
                "data_root": self.cfg.data_dir,
                "models_root": self.cfg.models_dir,
                "perf": {
                    "cpu_threads": cpu_threads,
                    "gpu_acceleration": gpu_acceleration,
                },
            }),
        )
        .await?;
        Ok(())
    }

    pub async fn shutdown(self: &Arc<Self>) -> Result<(), WorkerError> {
        self.stop_signal.notify_waiters();
        let _ = timeout(Duration::from_secs(3), self.request("shutdown", json!({}))).await;
        self.terminate_child().await;
        self.update_state(WorkerState::Stopped, None, None);
        Ok(())
    }

    pub async fn status(self: &Arc<Self>) -> WorkerStatus {
        let mut s = self.state.lock().clone();
        if matches!(s.state, WorkerState::Running) {
            if let Some(rc) = self.child.lock().await.as_ref() {
                s.uptime_ms = rc.started_at.elapsed().as_millis() as u64;
            }
        }
        s
    }

    async fn request(
        self: &Arc<Self>,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, WorkerError> {
        let id = self.next_request_id();
        let rx = self.send_request(&id, method, params).await?;
        self.wait_response(id, rx, Some(REQUEST_TIMEOUT)).await
    }

    /// Issue a request with a caller-supplied id and no timeout so the
    /// caller can pair it with :meth:`cancel_request` for cooperative
    /// cancellation. Used by long-running methods like ``stt.transcribe``.
    pub async fn request_no_timeout_with_id(
        self: &Arc<Self>,
        request_id: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, WorkerError> {
        let rx = self.send_request(request_id, method, params).await?;
        self.wait_response(request_id.to_string(), rx, None).await
    }

    /// Reserve a new request id without sending anything yet.
    pub fn new_request_id(self: &Arc<Self>) -> String {
        self.next_request_id()
    }

    fn next_request_id(self: &Arc<Self>) -> String {
        format!(
            "{}-{}",
            self.counter.fetch_add(1, Ordering::Relaxed),
            Uuid::new_v4().simple()
        )
    }

    async fn send_request(
        self: &Arc<Self>,
        id: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<oneshot::Receiver<Result<serde_json::Value, RpcError>>, WorkerError> {
        let req = RpcRequest::new(id.to_string(), method, params);
        let mut payload =
            serde_json::to_vec(&req).map_err(|e| WorkerError::Protocol { msg: e.to_string() })?;
        payload.push(b'\n');

        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id.to_string(), tx);

        let mut guard = self.child.lock().await;
        let rc = match guard.as_mut() {
            Some(rc) => rc,
            None => {
                self.pending.lock().remove(id);
                return Err(WorkerError::NotRunning);
            }
        };
        if let Err(source) = rc.stdin.write_all(&payload).await {
            self.pending.lock().remove(id);
            return Err(WorkerError::Io { source });
        }
        if let Err(source) = rc.stdin.flush().await {
            self.pending.lock().remove(id);
            return Err(WorkerError::Io { source });
        }
        Ok(rx)
    }

    async fn wait_response(
        self: &Arc<Self>,
        id: String,
        rx: oneshot::Receiver<Result<serde_json::Value, RpcError>>,
        deadline: Option<Duration>,
    ) -> Result<serde_json::Value, WorkerError> {
        let outcome = match deadline {
            Some(d) => match timeout(d, rx).await {
                Ok(inner) => inner,
                Err(_) => {
                    self.pending.lock().remove(&id);
                    return Err(WorkerError::Timeout(d));
                }
            },
            None => rx.await,
        };
        match outcome {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(err)) => Err(WorkerError::Rpc(err)),
            Err(_) => Err(WorkerError::Cancelled),
        }
    }

    /// Send a JSON-RPC notification (no ``id``) — for control messages
    /// like `jsonrpc://cancel` that don't need a response.
    pub async fn notify(
        self: &Arc<Self>,
        method: &str,
        params: serde_json::Value,
    ) -> Result<(), WorkerError> {
        let notif = RpcNotification::new(method, params);
        let mut payload =
            serde_json::to_vec(&notif).map_err(|e| WorkerError::Protocol { msg: e.to_string() })?;
        payload.push(b'\n');
        let mut guard = self.child.lock().await;
        let rc = guard.as_mut().ok_or(WorkerError::NotRunning)?;
        rc.stdin
            .write_all(&payload)
            .await
            .map_err(|source| WorkerError::Io { source })?;
        rc.stdin
            .flush()
            .await
            .map_err(|source| WorkerError::Io { source })?;
        Ok(())
    }

    /// Ask the worker to cooperatively cancel a pending async request.
    /// The worker replies with a small ack frame, which we discard.
    pub async fn cancel_request(self: &Arc<Self>, request_id: &str) -> Result<(), WorkerError> {
        self.notify("jsonrpc://cancel", json!({ "requestId": request_id }))
            .await
    }

    /// Subscribe to notification frames for a specific method (e.g.
    /// ``"stt.progress"``). The callback runs on the stdout reader
    /// task, so keep it fast and non-blocking.
    pub fn subscribe(
        self: &Arc<Self>,
        method: &str,
        callback: NotificationCallback,
    ) -> SubscriptionId {
        let id = self.sub_counter.fetch_add(1, Ordering::Relaxed);
        self.subscribers
            .lock()
            .insert(id, (method.to_string(), callback));
        SubscriptionId(id)
    }

    pub fn unsubscribe(self: &Arc<Self>, id: SubscriptionId) {
        self.subscribers.lock().remove(&id.0);
    }

    fn update_state(&self, state: WorkerState, pid: Option<u32>, last_error: Option<String>) {
        let mut s = self.state.lock();
        *s = WorkerStatus {
            state,
            pid,
            uptime_ms: 0,
            last_error,
        };
        let payload = s.clone();
        drop(s);
        if let Err(err) = self.app.emit("worker://status", payload) {
            tracing::warn!(%err, "failed to emit worker://status");
        }
    }
}

/// Opaque handle returned by [`WorkerSupervisor::subscribe`]. The
/// caller uses it to remove the subscription when finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

fn dispatch_notification(subscribers: &SubscriberMap, notif: &IncomingNotification) {
    // Copy the (method, callback) pairs out so the callback can run
    // without holding the lock — callbacks may re-enter (`emit`,
    // logging, etc.) and we don't want to deadlock ourselves.
    let matches: Vec<NotificationCallback> = subscribers
        .lock()
        .values()
        .filter_map(|(method, cb)| {
            if method == &notif.method {
                Some(cb.clone())
            } else {
                None
            }
        })
        .collect();
    for cb in matches {
        cb(&notif.method, &notif.params);
    }
}

/// Discover a Python interpreter for the worker.
///
/// Order (Phase 12):
///   1. `LMT_PYTHON` env var (explicit override — respected in dev + prod)
///   2. `<exe>/../Resources/python-embed/bin/python3` (macOS app bundle)
///   3. `<exe>/python-embed/bin/python3` (Linux AppImage / Windows portable)
///   4. `<exe>/python-embed/python.exe` (Windows layout)
///   5. `<repo>/.venv-worker/bin/python3` (dev checkout next to the worker)
///   6. `python3` on PATH
///   7. `python` on PATH
///   8. Fallback: literal `"python3"` — spawn will fail cleanly with
///      `WORKER_PYTHON_MISSING` and the UI surfaces the install hint.
pub fn detect_python_bin() -> PathBuf {
    if let Ok(p) = std::env::var("LMT_PYTHON") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        for candidate in bundled_python_candidates(&exe) {
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    for candidate in local_venv_python_candidates() {
        if candidate.is_file() {
            return candidate;
        }
    }
    for name in ["python3", "python"] {
        if let Some(bin) = which_on_path(name) {
            return bin;
        }
    }
    PathBuf::from("python3")
}

/// Dev checkout: `setup.sh` installs extras (Piper, Whisper, …) into
/// `.venv-worker` next to the `python/` worker package. Packaged apps
/// never have this directory, so the probe is a no-op in production.
fn local_venv_python_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let worker_root = detect_worker_root();
    let Some(repo_root) = Path::new(&worker_root).parent() else {
        return out;
    };
    out.push(repo_root.join(".venv-worker").join("bin").join("python3"));
    out.push(repo_root.join(".venv-worker").join("bin").join("python"));
    #[cfg(windows)]
    {
        out.push(
            repo_root
                .join(".venv-worker")
                .join("Scripts")
                .join("python.exe"),
        );
    }
    out
}

/// Phase 12 — list every location a packaged app might have stashed a
/// self-contained Python runtime. Returned in probe order.
fn bundled_python_candidates(exe: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let exe_dir = match exe.parent() {
        Some(d) => d.to_path_buf(),
        None => return out,
    };

    // macOS `.app` layout: <bundle>/Contents/MacOS/<exe>
    //                      <bundle>/Contents/Resources/python-embed/bin/python3
    if let Some(macos_parent) = exe_dir.parent() {
        out.push(
            macos_parent
                .join("Resources")
                .join("python-embed")
                .join("bin")
                .join("python3"),
        );
    }
    // Linux AppImage / portable Windows: sibling directory.
    out.push(exe_dir.join("python-embed").join("bin").join("python3"));
    #[cfg(windows)]
    {
        out.push(exe_dir.join("python-embed").join("python.exe"));
        out.push(
            exe_dir
                .join("python-embed")
                .join("Scripts")
                .join("python.exe"),
        );
    }
    out
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Locate the Python worker package root at runtime.
///
/// Order (Phase 12):
///   1. `LMT_WORKER_ROOT` env var (explicit override).
///   2. `<exe>/../Resources/python` — macOS bundle layout that Tauri
///      lays down when the `bundle.resources` glob matches the worker
///      sources.
///   3. Walk up from the executable up to 6 levels looking for a
///      `python/src/movie_translator_worker/` sibling (dev + linux
///      appimage layout).
///   4. Compile-time fallback `<CARGO_MANIFEST_DIR>/../python`.
pub fn detect_worker_root() -> PathBuf {
    if let Ok(root) = std::env::var("LMT_WORKER_ROOT") {
        return PathBuf::from(root);
    }
    if let Ok(exe) = std::env::current_exe() {
        let module_marker = Path::new("src").join(WORKER_MODULE);

        // macOS `.app` layout: the bundled resource lives under
        // `<bundle>/Contents/Resources/python/…`. Try that first
        // so the packaged app never accidentally reaches a stale
        // dev checkout that happens to sit next to it.
        if let Some(macos_parent) = exe.parent().and_then(|p| p.parent()) {
            let candidate = macos_parent.join("Resources").join("python");
            if candidate.join(&module_marker).is_dir() {
                return candidate;
            }
        }

        let mut cur = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..6 {
            let Some(dir) = cur.take() else { break };
            let candidate = dir.join("python");
            if candidate.join(&module_marker).is_dir() {
                return candidate;
            }
            cur = dir.parent().map(|p| p.to_path_buf());
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../python")
}
