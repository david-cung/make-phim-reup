import { lazy, Suspense, useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./screens/Dashboard";
import { useAppStore } from "./state/store";

const ProjectView = lazy(() => import("./screens/Project"));
const Settings = lazy(() => import("./screens/Settings"));

// UI redesign — the App shell is intentionally minimal now.
//
// Previously this file also rendered the app-wide sidebar and worker
// badge; the new professional editor design gives every screen its own
// top bar (`components/TopBar.tsx`) because the Project view uses a
// completely different multi-pane editor layout than Dashboard/Settings.
// Keeping App.tsx focused on routing + bootstrap avoids double chrome.

export default function App() {
  const bootstrap = useAppStore((s) => s.bootstrap);
  const bootError = useAppStore((s) => s.bootError);
  const bootReady = useAppStore((s) => s.bootReady);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  if (!bootReady && !bootError) {
    return (
      <div className="app-shell">
        <div className="loading" style={{ margin: "auto" }}>
          Starting Local Movie Translator…
        </div>
      </div>
    );
  }

  if (bootError) {
    return (
      <div className="app-shell">
        <div className="error-panel">
          <h2>Application failed to start</h2>
          <pre>{bootError}</pre>
        </div>
      </div>
    );
  }

  return (
    <Suspense
      fallback={
        <div className="app-shell">
          <div className="loading" style={{ margin: "auto" }}>
            Loading workspace…
          </div>
        </div>
      }
    >
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/projects/:id" element={<ProjectView />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
