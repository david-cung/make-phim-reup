import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useAppStore } from "../state/store";
import {
  IconChevronLeft,
  IconExport,
  IconRedo,
  IconSave,
  IconSettings,
  IconUndo,
} from "./icons";

// UI redesign — professional top bar shared by every screen.
//
// The topbar collapses into three horizontal zones (left / center /
// right) at a fixed 52 px height. Each screen provides:
//   - `subject`  : brand block on the left (project name, "Dashboard",
//                  "Settings", etc.).
//   - `center`   : contextual controls (playhead time, tabs) — usually
//                  omitted; when omitted the default is Undo/Redo pair.
//   - `actions`  : right-side action cluster (Save, Export, ...).
//
// We keep the topbar drag-region enabled for macOS window movement, but
// every interactive element opts back into `no-drag` via a global rule
// in `global.css`.

export type TopBarProps = {
  subject?: ReactNode;
  center?: ReactNode;
  actions?: ReactNode;
  showBackToDashboard?: boolean;
  showSettingsLink?: boolean;
  showDefaultTools?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onSave?: () => void;
  onExport?: () => void;
  showWorker?: boolean;
};

export function TopBar(props: TopBarProps) {
  const {
    subject,
    center,
    actions,
    showBackToDashboard = false,
    showSettingsLink = true,
    showDefaultTools = true,
    onUndo,
    onRedo,
    canUndo = false,
    canRedo = false,
    onSave,
    onExport,
    showWorker = true,
  } = props;

  return (
    <header className="topbar">
      <div className="topbar-left">
        {showBackToDashboard && (
          <Link to="/" className="topbar-back" aria-label="Back to dashboard">
            <IconChevronLeft size={14} />
            Projects
          </Link>
        )}
        <Link to="/" className="app-brand" title="Local Movie Translator">
          <span className="brand-mark">LM</span>
          <span>Movie Translator</span>
        </Link>
        {subject ? (
          <>
            <span className="topbar-divider" />
            {subject}
          </>
        ) : null}
      </div>

      <div className="topbar-center">
        {center ??
          (showDefaultTools && (
            <div className="actions">
              <button
                className="btn icon"
                onClick={onUndo}
                disabled={!canUndo || !onUndo}
                title="Undo"
                aria-label="Undo"
              >
                <IconUndo size={16} />
              </button>
              <button
                className="btn icon"
                onClick={onRedo}
                disabled={!canRedo || !onRedo}
                title="Redo"
                aria-label="Redo"
              >
                <IconRedo size={16} />
              </button>
            </div>
          ))}
      </div>

      <div className="topbar-right">
        {showWorker && <WorkerBadge />}
        {actions}
        {onSave && (
          <button className="btn" onClick={onSave} title="Save (⌘S)">
            <IconSave size={14} />
            <span>Save</span>
          </button>
        )}
        {showSettingsLink && (
          <Link
            to="/settings"
            className="btn icon"
            title="Settings"
            aria-label="Settings"
          >
            <IconSettings size={16} />
          </Link>
        )}
        {onExport && (
          <button className="btn primary" onClick={onExport} title="Export">
            <IconExport size={14} />
            <span>Export</span>
          </button>
        )}
      </div>
    </header>
  );
}

function WorkerBadge() {
  const worker = useAppStore((s) => s.worker);
  const dot =
    worker.state === "running"
      ? "ok"
      : worker.state === "starting"
        ? "warn"
        : worker.state === "stopped"
          ? "muted"
          : "err";
  return (
    <div
      className={`worker-badge worker-badge--${dot}`}
      title={`Python worker: ${worker.state}${worker.pid ? ` (pid ${worker.pid})` : ""}`}
    >
      <span className="worker-dot" />
      <span className="worker-label">
        {worker.state === "running" ? "Worker ready" : `Worker ${worker.state}`}
      </span>
    </div>
  );
}
