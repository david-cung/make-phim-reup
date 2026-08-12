import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useAppStore } from "@/state/store";
import { api } from "@/ipc/bridge";
import type { AppError, CreateProjectInput, JobSnapshot } from "@/ipc/types";
import { TRANSLATION_LANGUAGES, isAppError } from "@/ipc/types";
import { TopBar } from "../components/TopBar";
import { IconFilm, IconPlus, IconTrash } from "../components/icons";

// UI redesign — Dashboard is now the "Projects" workstation launcher.
//
// The screen renders inside the new shared shell (app-shell → topbar +
// plain body) so it matches the Project editor's chrome. Content-wise
// nothing was cut: the crash-recovery orphan banner, the first-run
// helper, the project table, and the "New project" modal all still
// bind to the same backend actions.

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
    // Phase 12 — surface jobs the crash-reaper marked as interrupted at
    // startup so the user can resume them from the orphan banner.
    void api
      .listOrphanedJobs()
      .then((list) => setOrphaned(list))
      .catch(() => setOrphaned([]));
  }, [refresh, refreshModelDirectory, refreshLocalModels]);

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
      const prev = byId.get(j.projectId);
      if (
        !prev ||
        (j.completedAt ?? j.createdAt) > (prev.completedAt ?? prev.createdAt)
      ) {
        byId.set(j.projectId, j);
      }
    }
    return Array.from(byId.values());
  }, [orphaned]);

  return (
    <div className="app-shell">
      <TopBar
        showDefaultTools={false}
        actions={
          <button
            className="btn primary"
            onClick={() => setModalOpen(true)}
            title="Create a new movie translation project"
          >
            <IconPlus size={14} />
            <span>New Project</span>
          </button>
        }
      />

      <main className="app-body plain">
        <div className="plain-scroll">
          {!orphanBannerDismissed && orphanedByProject.length > 0 && (
            <div className="orphan-banner">
              <div>
                <strong>Some jobs didn't finish last time.</strong> Nothing was
                lost — completed segments are still on disk. Reopen the project
                and re-run the affected stage to pick up where you left off.
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
                <strong>Welcome to Local Movie Translator.</strong> Everything
                runs on your machine — audio never leaves your device.{" "}
                {availableModels === 0 ? (
                  <>
                    No AI models yet — that's fine. Create a project, click{" "}
                    <em>Transcribe</em>, <em>Translate</em>, or{" "}
                    <em>Generate voice</em> and the app will download the
                    recommended Whisper / GGUF / Piper model on demand. Manage
                    them any time from{" "}
                    <Link to="/settings">Settings → AI Models</Link>.
                  </>
                ) : (
                  <>
                    Found {availableModels} local model
                    {availableModels === 1 ? "" : "s"}
                    {modelDirectory ? (
                      <>
                        {" "}
                        in{" "}
                        <span className="mono">{modelDirectory.path}</span>
                      </>
                    ) : null}
                    . Missing models auto-download on first use.
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
            <div>
              <h1>Projects</h1>
              <div className="dashboard-sub">
                {projects.length === 0
                  ? "No projects yet."
                  : `${projects.length} project${projects.length === 1 ? "" : "s"}`}
              </div>
            </div>
          </header>

          {error ? <div className="banner banner--error">{error}</div> : null}

          {projects.length === 0 ? (
            <div className="panel">
              <div className="empty-state">
                <div className="icon-64" aria-hidden="true">
                  <IconFilm size={22} />
                </div>
                <div className="empty-title">
                  Import a movie to start translating
                </div>
                <div className="empty-hint">
                  Create a project, drop in a video, and Local Movie Translator
                  will transcribe, translate, dub and re-render it — entirely
                  on-device.
                </div>
                <button
                  className="btn primary"
                  onClick={() => setModalOpen(true)}
                >
                  <IconPlus size={14} />
                  <span>New Project</span>
                </button>
              </div>
            </div>
          ) : (
            <table className="projects-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Languages</th>
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
                      <span className="mono">
                        {p.sourceLanguage} → {p.targetLanguage}
                      </span>
                    </td>
                    <td>
                      <span className={`status status--${p.status}`}>
                        {p.status}
                      </span>
                    </td>
                    <td className="muted">
                      {new Date(p.updatedAt).toLocaleString()}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        className="btn icon"
                        title={`Delete ${p.name}`}
                        aria-label={`Delete ${p.name}`}
                        onClick={async () => {
                          if (!confirm(`Delete "${p.name}"?`)) return;
                          try {
                            await deleteProject(p.id);
                          } catch (e) {
                            setError(formatDashboardError(e));
                          }
                        }}
                      >
                        <IconTrash size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>

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
    </div>
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
        <div className="modal-hint">
          Pick the source language of the movie and the language you want to
          dub it into. You can change these later.
        </div>
        <label>
          Project name
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
