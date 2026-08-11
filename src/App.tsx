import { useEffect } from "react";
import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import Dashboard from "./screens/Dashboard";
import ProjectView from "./screens/Project";
import Settings from "./screens/Settings";
import { useAppStore } from "./state/store";

export default function App() {
  const bootstrap = useAppStore((s) => s.bootstrap);
  const bootError = useAppStore((s) => s.bootError);
  const bootReady = useAppStore((s) => s.bootReady);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="app-title">Local Movie Translator</div>
        <nav className="app-nav">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
        <div className="app-sidebar-footer">
          <WorkerBadge />
        </div>
      </aside>

      <main className="app-main">
        {!bootReady && !bootError ? (
          <div className="loading">Starting…</div>
        ) : bootError ? (
          <div className="error-panel">
            <h2>Application failed to start</h2>
            <pre>{bootError}</pre>
          </div>
        ) : (
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/projects/:id" element={<ProjectView />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        )}
      </main>
    </div>
  );
}

function WorkerBadge() {
  const worker = useAppStore((s) => s.worker);
  const dot =
    worker.state === "running" ? "ok"
    : worker.state === "starting" ? "warn"
    : worker.state === "stopped" ? "muted"
    : "err";
  return (
    <div className={`worker-badge worker-badge--${dot}`}>
      <span className="worker-dot" />
      <span className="worker-label">Worker: {worker.state}</span>
      {worker.pid ? <span className="worker-pid">pid {worker.pid}</span> : null}
    </div>
  );
}
