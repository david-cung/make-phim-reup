import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useAppStore } from "@/state/store";
import { api } from "@/ipc/bridge";
import type { AppError, CreateProjectInput, JobSnapshot } from "@/ipc/types";
import { TRANSLATION_LANGUAGES, isAppError } from "@/ipc/types";

export default function Dashboard() {
  const projects = useAppStore((s) => s.projects);
  const refresh = useAppStore((s) => s.refreshProjects);
  const createProject = useAppStore((s) => s.createProject);
  const deleteProject = useAppStore((s) => s.deleteProject);
  const settings = useAppStore((s) => s.settings);
  const localModels = useAppStore((s) => s.localModels);
  const modelDirectory = useAppStore((s) => s.modelDirectory);
  const refreshLocalModels = useAppStore((s) => s.refreshLocalModels);
  const refreshModelDirectory = useAppStore((s) => s.refreshModelDirectory);
  const markFirstRunCompleted = useAppStore((s) => s.markFirstRunCompleted);

  const [modalOpen, setModalOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [orphaned, setOrphaned] = useState<JobSnapshot[]>([]);
  const [orphanBannerDismissed, setOrphanBannerDismissed] = useState(false);

  useEffect(() => {
    void refresh();
    void refreshModelDirectory();
    void refreshLocalModels();
    // Phase 12 — surface jobs the crash-reaper marked as
    // interrupted at startup. One-shot fetch; the list is
    // stable until the user resumes/deletes the affected project.
    void api
      .listOrphanedJobs()
      .then((list) => setOrphaned(list))
      .catch(() => setOrphaned([]));
  }, [refresh, refreshModelDirectory, refreshLocalModels]);

  // Phase 10 — first-run banner. Shown once, dismissible. Never
  // pushes the user online.
  const showFirstRunBanner = useMemo(() => {
    if (!settings) return false;
    return !settings.firstRunCompleted;
  }, [settings]);

  const availableModels = useMemo(
    () => localModels.filter((m) => m.status === "available").length,
    [localModels],
  );

  const orphanedByProject = useMemo(() => {
    const byId = new Map<string, JobSnapshot>();
    for (const j of orphaned) {
      // Keep only the most recent orphaned stage per project so the
      // banner stays terse even after several bad runs.
      const prev = byId.get(j.projectId);
      if (!prev || (j.completedAt ?? j.createdAt) > (prev.completedAt ?? prev.createdAt)) {
        byId.set(j.projectId, j);
      }
    }
    return Array.from(byId.values());
  }, [orphaned]);

  return (
    <section className="dashboard">
      {!orphanBannerDismissed && orphanedByProject.length > 0 && (
        <div className="orphan-banner">
          <div>
            <strong>Some jobs didn't finish last time.</strong> Nothing was
            lost — completed segments are still on disk. Reopen the project
            below and re-run the affected stage to pick up where you left off.
            <ul className="orphan-banner-list">
              {orphanedByProject.slice(0, 5).map((j) => {
                const project = projects.find((p) => p.id === j.projectId);
                return (
                  <li key={j.id}>
                    <Link to={`/projects/${j.projectId}`}>
                      {project?.name ?? j.projectId.slice(0, 8)}
                    </Link>{" "}
                    · interrupted during <em>{j.stage}</em>
                  </li>
                );
              })}
            </ul>
          </div>
          <button
            className="btn ghost small"
            onClick={() => setOrphanBannerDismissed(true)}
          >
            Dismiss
          </button>
        </div>
      )}

      {showFirstRunBanner && (
        <div className="first-run-banner">
          <div>
            <strong>Welcome!</strong> Local Movie Translator runs entirely
            offline once models are installed. It never downloads models on
            your behalf.{" "}
            {availableModels === 0 ? (
              <>
                No models are installed yet — head to{" "}
                <Link to="/settings">Settings → AI Models</Link> to point at a
                Whisper folder, a translation GGUF, or a Piper voice.
              </>
            ) : (
              <>
                Found {availableModels} local model
                {availableModels === 1 ? "" : "s"}
                {modelDirectory ? (
                  <>
                    {" "}in <span className="mono">{modelDirectory.path}</span>
                  </>
                ) : null}
                . Manage them any time in{" "}
                <Link to="/settings">Settings → AI Models</Link>.
              </>
            )}
          </div>
          <button
            className="btn ghost small"
            onClick={() => void markFirstRunCompleted()}
          >
            Got it
          </button>
        </div>
      )}

      <header className="dashboard-header">
        <h1>Projects</h1>
        <button className="btn primary" onClick={() => setModalOpen(true)}>
          + New Project
        </button>
      </header>

      {error ? <div className="error-panel small">{error}</div> : null}

      {projects.length === 0 ? (
        <div className="empty-state">
          <p>No projects yet. Create one to get started.</p>
        </div>
      ) : (
        <table className="projects-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Source → Target</th>
              <th>Status</th>
              <th>Updated</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr key={p.id}>
                <td>
                  <Link to={`/projects/${p.id}`}>{p.name}</Link>
                </td>
                <td>
                  {p.sourceLanguage} → {p.targetLanguage}
                </td>
                <td>
                  <span className={`status status--${p.status}`}>{p.status}</span>
                </td>
                <td>{new Date(p.updatedAt).toLocaleString()}</td>
                <td>
                  <button
                    className="btn ghost danger"
                    onClick={async () => {
                      if (!confirm(`Delete "${p.name}"?`)) return;
                      try {
                        await deleteProject(p.id);
                      } catch (e) {
                        setError(formatDashboardError(e));
                      }
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {modalOpen ? (
        <NewProjectModal
          onClose={() => setModalOpen(false)}
          onCreate={async (input) => {
            setError(null);
            try {
              await createProject(input);
              setModalOpen(false);
            } catch (e) {
              setError(formatDashboardError(e));
            }
          }}
        />
      ) : null}
    </section>
  );
}

function formatDashboardError(err: unknown): string {
  if (isAppError(err)) {
    const e = err as AppError;
    const base = `${e.code}: ${e.message}`;
    return e.hint ? `${base}\n\nHint: ${e.hint}` : base;
  }
  return err instanceof Error ? err.message : JSON.stringify(err);
}

function NewProjectModal(props: {
  onClose: () => void;
  onCreate: (input: CreateProjectInput) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [targetLanguage, setTargetLanguage] = useState("vi");
  const [busy, setBusy] = useState(false);

  return (
    <div className="modal-backdrop" onClick={props.onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>New project</h2>
        <label>
          Name
          <input
            type="text"
            value={name}
            autoFocus
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. My Movie 2024"
          />
        </label>
        <div className="row">
          <label>
            Source language
            <select
              value={sourceLanguage}
              onChange={(e) => setSourceLanguage(e.target.value)}
            >
              {TRANSLATION_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Target language
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
            >
              {TRANSLATION_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="modal-actions">
          <button className="btn ghost" onClick={props.onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="btn primary"
            disabled={busy || name.trim() === ""}
            onClick={async () => {
              setBusy(true);
              try {
                await props.onCreate({
                  name: name.trim(),
                  sourceLanguage: sourceLanguage.trim() || "en",
                  targetLanguage: targetLanguage.trim() || "vi",
                });
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "Creating…" : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
}
