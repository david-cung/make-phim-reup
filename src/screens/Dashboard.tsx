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

          {error ? <div className="banner banner--error">{error}</div> : null}

          <section className="hub-section">
            <div className="hub-section-head">
              <h2>Recent projects</h2>
              <span className="hub-count">
                {projects.length === 0
                  ? "Nothing yet"
                  : `${projects.length} project${projects.length === 1 ? "" : "s"}`}
              </span>
            </div>

            {projects.length === 0 ? (
              <button
                className="dropzone"
                onClick={() => setModalOpen(true)}
                type="button"
              >
                <span className="dropzone-icon" aria-hidden="true">
                  <IconFilm size={22} />
                </span>
                <span className="dropzone-title">Start a new project</span>
                <span className="dropzone-hint">
                  Import a movie and this studio will transcribe, translate, dub
                  and re-render it — entirely on your machine.
                </span>
                <span className="btn primary dropzone-cta">
                  <IconPlus size={14} />
                  <span>New project</span>
                </span>
              </button>
            ) : (
              <div className="hub-grid">
                {projects.map((p) => (
                  <article key={p.id} className="proj-tile">
                    <Link
                      to={`/projects/${p.id}`}
                      className="proj-thumb"
                      aria-label={`Open ${p.name}`}
                    >
                      <IconFilm size={22} />
                      <span className={`proj-chip proj-chip--${p.status}`}>
                        {p.status}
                      </span>
                    </Link>
                    <div className="proj-info">
                      <Link to={`/projects/${p.id}`} className="proj-name">
                        {p.name}
                      </Link>
                      <div className="proj-meta">
                        <span className="mono">
                          {p.sourceLanguage.toUpperCase()} →{" "}
                          {p.targetLanguage.toUpperCase()}
                        </span>
                        <span className="proj-dot" />
                        <span>{relativeTime(p.updatedAt)}</span>
                      </div>
                    </div>
                    <button
                      className="btn icon proj-del"
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
                  </article>
                ))}

                <button
                  className="proj-tile proj-tile--new"
                  onClick={() => setModalOpen(true)}
                  type="button"
                >
                  <span className="proj-new-mark">
                    <IconPlus size={18} />
                  </span>
                  <span className="proj-new-label">New project</span>
                  <span className="proj-new-hint">Import a movie to start</span>
                </button>
              </div>
            )}
          </section>
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

/** "Last edited 2h ago" style stamp used by the project hub tiles. */
function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "Edited just now";
  if (mins < 60) return `Edited ${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `Edited ${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days === 1) return "Edited yesterday";
  if (days < 7) return `Edited ${days}d ago`;
  return `Edited ${new Date(then).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  })}`;
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
