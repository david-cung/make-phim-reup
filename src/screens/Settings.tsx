import { useEffect, useMemo, useState } from "react";
import { useAppStore } from "@/state/store";
import { api, pickDirectory, pickModelSource } from "@/ipc/bridge";
import {
  TRANSLATION_LANGUAGES,
  type EnvInfo,
  type ImportModelSpec,
  type LocalModel,
  type ModelKind,
  type PingResponse,
  type YouTubeConnectionState,
} from "@/ipc/types";
import { TopBar } from "../components/TopBar";

export default function Settings() {
  const settings = useAppStore((s) => s.settings);
  const appInfo = useAppStore((s) => s.appInfo);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const worker = useAppStore((s) => s.worker);
  const ffmpeg = useAppStore((s) => s.ffmpeg);
  const refreshFfmpeg = useAppStore((s) => s.refreshFfmpeg);
  const sttEnv = useAppStore((s) => s.sttEnv);
  const whisperModels = useAppStore((s) => s.whisperModels);
  const whisperModelsLoading = useAppStore((s) => s.whisperModelsLoading);
  const refreshSttEnv = useAppStore((s) => s.refreshSttEnv);
  const refreshWhisperModels = useAppStore((s) => s.refreshWhisperModels);
  const downloadWhisperModel = useAppStore((s) => s.downloadWhisperModel);
  const translationEnv = useAppStore((s) => s.translationEnv);
  const translationModels = useAppStore((s) => s.translationModels);
  const translationModelsLoading = useAppStore(
    (s) => s.translationModelsLoading,
  );
  const refreshTranslationEnv = useAppStore((s) => s.refreshTranslationEnv);
  const refreshTranslationModels = useAppStore(
    (s) => s.refreshTranslationModels,
  );
  const localModels = useAppStore((s) => s.localModels);
  const localModelsLoading = useAppStore((s) => s.localModelsLoading);
  const modelDirectory = useAppStore((s) => s.modelDirectory);
  const modelImportBusy = useAppStore((s) => s.modelImportBusy);
  const refreshLocalModels = useAppStore((s) => s.refreshLocalModels);
  const refreshModelDirectory = useAppStore((s) => s.refreshModelDirectory);
  const setModelDirectory = useAppStore((s) => s.setModelDirectory);
  const importLocalModel = useAppStore((s) => s.importLocalModel);
  const unloadAllModels = useAppStore((s) => s.unloadAllModels);

  const [ping, setPing] = useState<PingResponse | null>(null);
  const [env, setEnv] = useState<EnvInfo | null>(null);
  const [pingError, setPingError] = useState<string | null>(null);
  const [ffmpegPathDraft, setFfmpegPathDraft] = useState<string>(
    settings?.ffmpegPath ?? "",
  );
  const [modelError, setModelError] = useState<string | null>(null);
  const [modelFilter, setModelFilter] = useState<ModelKind | "all">("all");

  const filteredModels = useMemo(
    () =>
      modelFilter === "all"
        ? localModels
        : localModels.filter((m) => m.kind === modelFilter),
    [localModels, modelFilter],
  );

  useEffect(() => {
    void refreshEnv();
    void refreshSttEnv();
    void refreshWhisperModels();
    void refreshTranslationEnv();
    void refreshTranslationModels();
    void refreshModelDirectory();
    void refreshLocalModels();
  }, [
    refreshSttEnv,
    refreshWhisperModels,
    refreshTranslationEnv,
    refreshTranslationModels,
    refreshModelDirectory,
    refreshLocalModels,
  ]);

  useEffect(() => {
    setFfmpegPathDraft(settings?.ffmpegPath ?? "");
  }, [settings?.ffmpegPath]);

  async function refreshEnv() {
    setPingError(null);
    try {
      const [p, e] = await Promise.all([api.workerPing(), api.workerEnvInfo()]);
      setPing(p);
      setEnv(e);
    } catch (err) {
      setPingError(err instanceof Error ? err.message : JSON.stringify(err));
    }
  }

  async function applyFfmpegPath() {
    const trimmed = ffmpegPathDraft.trim();
    await updateSettings({ ffmpegPath: trimmed.length > 0 ? trimmed : null });
  }

  if (!settings || !appInfo) {
    return (
      <div className="app-shell">
        <TopBar showDefaultTools={false} showSettingsLink={false} />
        <main className="app-body plain">
          <div className="plain-scroll">
            <div className="loading">Loading settings…</div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <TopBar
        showDefaultTools={false}
        showSettingsLink={false}
        subject={
          <div className="topbar-project">
            <span className="pname">Settings</span>
            <span className="pmeta">Configure the editor and AI models</span>
          </div>
        }
      />
      <main className="app-body plain">
        <div className="plain-scroll">
          <section className="settings">

      <div className="panel">
        <h3>Application</h3>
        <Kv k="Name" v={appInfo.appName} />
        <Kv k="Version" v={appInfo.appVersion} />
        <Kv k="OS / Arch" v={`${appInfo.os} / ${appInfo.arch}`} />
        <Kv k="Data dir" v={appInfo.dataDir} mono />
        <Kv k="Config dir" v={appInfo.configDir} mono />
        <Kv k="Log dir" v={appInfo.logDir} mono />
        <Kv k="Projects dir" v={appInfo.projectsDir} mono />
        <Kv k="Models dir" v={appInfo.modelsDir} mono />
      </div>

      <StoragePanel />

      <YouTubeSettingsPanel />

      <div className="panel">
        <h3>Preferences</h3>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.offlineMode}
            onChange={(e) => void updateSettings({ offlineMode: e.target.checked })}
          />
          Offline mode (no network requests)
        </label>

        <div className="row">
          <label>
            Source language (default for new projects)
            <select
              value={settings.sourceLanguage}
              onChange={(e) =>
                void updateSettings({ sourceLanguage: e.target.value })
              }
            >
              {TRANSLATION_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
              {!TRANSLATION_LANGUAGES.some(
                (l) => l.code === settings.sourceLanguage,
              ) && (
                <option value={settings.sourceLanguage}>
                  Custom ({settings.sourceLanguage})
                </option>
              )}
            </select>
          </label>
          <label>
            Target language (default for new projects)
            <select
              value={settings.targetLanguage}
              onChange={(e) =>
                void updateSettings({ targetLanguage: e.target.value })
              }
            >
              {TRANSLATION_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
              {!TRANSLATION_LANGUAGES.some(
                (l) => l.code === settings.targetLanguage,
              ) && (
                <option value={settings.targetLanguage}>
                  Custom ({settings.targetLanguage})
                </option>
              )}
            </select>
          </label>
        </div>

        <div className="row">
          <label>
            Max concurrent jobs
            <input
              type="number"
              min={1}
              max={8}
              value={settings.maxConcurrentJobs}
              onChange={(e) =>
                void updateSettings({ maxConcurrentJobs: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Log level
            <select
              value={settings.logLevel}
              onChange={(e) =>
                void updateSettings({
                  logLevel: e.target.value as typeof settings.logLevel,
                })
              }
            >
              <option>trace</option>
              <option>debug</option>
              <option>info</option>
              <option>warn</option>
              <option>error</option>
            </select>
          </label>
        </div>
      </div>

      <div className="panel">
        <h3>Performance</h3>
        <p className="small" style={{ color: "var(--fg-muted)" }}>
          Advanced knobs for long-movie runs. Defaults are safe for
          most Apple Silicon Macs — the app will fall back to CPU
          when Metal is unavailable, and to the engine's own thread
          count when <em>CPU threads</em> is blank.
        </p>
        <div className="row">
          <label>
            Auto-unload models after (seconds)
            <input
              type="number"
              min={0}
              max={3600}
              value={settings.autoUnloadAfterSecs ?? 0}
              onChange={(e) => {
                const raw = Number(e.target.value);
                const next = Number.isFinite(raw) && raw > 0 ? raw : null;
                void updateSettings({ autoUnloadAfterSecs: next });
              }}
            />
          </label>
          <label>
            CPU threads
            <input
              type="number"
              min={0}
              max={64}
              placeholder="auto"
              value={settings.cpuThreads ?? ""}
              onChange={(e) => {
                const raw = Number(e.target.value);
                const next = Number.isFinite(raw) && raw > 0 ? raw : null;
                void updateSettings({ cpuThreads: next });
              }}
            />
          </label>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.gpuAcceleration}
            onChange={(e) =>
              void updateSettings({ gpuAcceleration: e.target.checked })
            }
          />
          GPU acceleration (Metal / CUDA when available)
        </label>
        <PerformanceMonitor />
      </div>

      <div className="panel">
        <h3>AI Models</h3>
        <p className="small" style={{ color: "var(--fg-muted)" }}>
          Local Model Manager. Models live outside the application bundle in
          the folder below. When you click <em>Transcribe</em>,{" "}
          <em>Translate</em>, or <em>Generate voice</em> on a project, missing
          models are auto-downloaded from Hugging Face on demand. Use this
          screen to browse installed models, add local files with{" "}
          <em>Add Local Model</em>, or pre-download extras before going
          offline.
        </p>

        <div className="model-dir-row">
          <div className="kv" style={{ flex: 1 }}>
            <span className="kv-k">Model directory</span>
            <span className="kv-v mono">
              {modelDirectory?.path ?? "—"}
              {modelDirectory && !modelDirectory.isDefault && (
                <em style={{ marginLeft: 6, color: "var(--fg-muted)" }}>
                  (override)
                </em>
              )}
              {modelDirectory && !modelDirectory.exists && (
                <em style={{ marginLeft: 6, color: "var(--warn)" }}>
                  (missing)
                </em>
              )}
            </span>
          </div>
          <div className="actions" style={{ marginTop: 0 }}>
            <button
              className="btn ghost small"
              onClick={async () => {
                setModelError(null);
                const picked = await pickDirectory("Select model directory");
                if (!picked) return;
                try {
                  await setModelDirectory(picked);
                } catch (err) {
                  setModelError(errorMessage(err));
                }
              }}
            >
              Change…
            </button>
            {modelDirectory && !modelDirectory.isDefault && (
              <button
                className="btn ghost small"
                onClick={async () => {
                  setModelError(null);
                  try {
                    await setModelDirectory(null);
                  } catch (err) {
                    setModelError(errorMessage(err));
                  }
                }}
              >
                Reset to default
              </button>
            )}
          </div>
        </div>

        {modelDirectory && !modelDirectory.isDefault && (
          <div className="small" style={{ color: "var(--fg-muted)" }}>
            Default: <span className="mono">{modelDirectory.defaultPath}</span>
          </div>
        )}

        <div className="model-filter">
          <label className="small">
            Show{" "}
            <select
              value={modelFilter}
              onChange={(e) =>
                setModelFilter(e.target.value as ModelKind | "all")
              }
            >
              <option value="all">All</option>
              <option value="whisper">Whisper</option>
              <option value="translation">Translation</option>
              <option value="tts">TTS</option>
              <option value="voice">Voices</option>
            </select>
          </label>
          <div className="small" style={{ color: "var(--fg-muted)" }}>
            {localModelsLoading
              ? "scanning…"
              : `${filteredModels.length} shown / ${localModels.length} total`}
          </div>
        </div>

        <div className="model-table">
          {filteredModels.length === 0 && !localModelsLoading && (
            <div className="small" style={{ color: "var(--fg-muted)" }}>
              No models yet. You can just open a project and click{" "}
              <em>Transcribe</em>, <em>Translate</em>, or{" "}
              <em>Generate voice</em> — the app will auto-download the
              recommended Whisper / GGUF / Piper model on demand. Or use{" "}
              <em>Add Local Model</em> below to register a model you already
              have on disk.
            </div>
          )}
          {filteredModels.map((m) => (
            <ModelRow key={`${m.kind}:${m.id}`} model={m} />
          ))}
        </div>

        {modelError && (
          <div className="error-panel small" style={{ marginTop: 8 }}>
            {modelError}
          </div>
        )}

        <div className="actions">
          <button
            className="btn"
            disabled={localModelsLoading}
            onClick={() => void refreshLocalModels(true)}
          >
            {localModelsLoading ? "Scanning…" : "Scan Models"}
          </button>
          <AddLocalModelButton
            busy={modelImportBusy}
            onImport={async (spec) => {
              setModelError(null);
              try {
                await importLocalModel(spec);
              } catch (err) {
                setModelError(errorMessage(err));
              }
            }}
          />
          <button
            className="btn ghost"
            onClick={async () => {
              setModelError(null);
              try {
                await unloadAllModels();
              } catch (err) {
                setModelError(errorMessage(err));
              }
            }}
          >
            Unload all
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>FFmpeg</h3>
        <Kv
          k="Status"
          v={
            ffmpeg?.available
              ? `available · ${ffmpeg.version ?? "unknown version"}`
              : `unavailable${ffmpeg?.error ? ` (${ffmpeg.error})` : ""}`
          }
        />
        {ffmpeg?.ffmpegPath && (
          <Kv k="ffmpeg path" v={ffmpeg.ffmpegPath} mono />
        )}
        {ffmpeg?.ffprobePath && (
          <Kv k="ffprobe path" v={ffmpeg.ffprobePath} mono />
        )}
        <label className="stack">
          Custom ffmpeg path (leave blank to use system PATH)
          <input
            type="text"
            value={ffmpegPathDraft}
            onChange={(e) => setFfmpegPathDraft(e.target.value)}
            placeholder="/opt/homebrew/bin/ffmpeg"
          />
        </label>
        <div className="actions">
          <button className="btn" onClick={() => void applyFfmpegPath()}>
            Save & re-detect
          </button>
          <button className="btn ghost" onClick={() => void refreshFfmpeg()}>
            Re-detect
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>Speech recognition (Whisper)</h3>
        <Kv
          k="faster-whisper"
          v={sttEnv?.whisperInstalled ? "installed" : "not installed"}
        />
        <Kv k="Default device" v={sttEnv?.defaultDevice ?? "—"} />
        <Kv k="Models root" v={sttEnv?.modelsRoot ?? "—"} mono />
        {sttEnv?.largeV3 && (
          <Kv
            k="Whisper large-v3"
            v={
              sttEnv.largeV3.canRun
                ? "available on this hardware"
                : "not available on current hardware"
            }
          />
        )}
        {sttEnv?.largeV3?.reason && !sttEnv.largeV3.canRun && (
          <div className="banner banner--warn small">{sttEnv.largeV3.reason}</div>
        )}
        {sttEnv?.devices && sttEnv.devices.length > 0 && (
          <Kv
            k="Devices"
            v={sttEnv.devices
              .map((d) => `${d.label}${d.supported ? "" : " (unsupported)"}`)
              .join(", ")}
          />
        )}
        <div className="stt-models-list">
          <div className="kv">
            <span className="kv-k">Models</span>
            <span className="kv-v">
              {whisperModelsLoading ? "loading…" : `${whisperModels.length} known`}
            </span>
          </div>
          {whisperModels.map((m) => (
            <div key={m.name} className="kv">
              <span className="kv-k">
                {m.name === "large-v3" || m.name === "large"
                  ? `${m.name} (QUALITY)`
                  : m.name === "medium"
                    ? `${m.name} (BALANCED)`
                    : m.name === "small"
                      ? `${m.name} (FAST)`
                      : m.name}
              </span>
              <span className="kv-v">
                {m.installed
                  ? `installed · ${humanBytes(m.sizeBytes ?? 0)}`
                  : m.name === "large-v3" &&
                      sttEnv?.largeV3 &&
                      sttEnv.largeV3.canRun === false
                    ? "unavailable on this hardware"
                    : "not installed"}
                {!m.installed &&
                  !(
                    m.name === "large-v3" &&
                    sttEnv?.largeV3 &&
                    sttEnv.largeV3.canRun === false
                  ) && (
                  <button
                    className="btn ghost small"
                    style={{ marginLeft: 8 }}
                    onClick={() => void downloadWhisperModel(m.name)}
                  >
                    Download
                  </button>
                )}
              </span>
            </div>
          ))}
        </div>
        <div className="actions">
          <button className="btn ghost" onClick={() => void refreshSttEnv()}>
            Refresh
          </button>
          <button className="btn ghost" onClick={() => void refreshWhisperModels()}>
            Rescan models
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>Translation (Local LLM)</h3>
        <Kv
          k="llama-cpp-python"
          v={translationEnv?.llamaInstalled ? "installed" : "not installed"}
        />
        <Kv
          k="Models directory"
          v={translationEnv?.translationRoot ?? "—"}
          mono
        />
        <Kv
          k="Default model"
          v={translationEnv?.defaultModel ?? "—"}
        />
        <Kv
          k="Prompt versions"
          v={translationEnv?.promptVersions?.join(", ") ?? "—"}
        />
        <div className="stt-models-list">
          <div className="kv">
            <span className="kv-k">Installed GGUF files</span>
            <span className="kv-v">
              {translationModelsLoading
                ? "loading…"
                : `${translationModels.length} found`}
            </span>
          </div>
          {translationModels.length === 0 && !translationModelsLoading && (
            <div className="small" style={{ color: "var(--fg-muted)" }}>
              No GGUF installed. The easiest path: open a project and click{" "}
              <em>Translate</em> — the app auto-downloads the recommended
              Qwen 2.5 3B (~2 GB) on demand. Alternatively drop your own
              GGUF into the models directory above and click <em>Rescan</em>.
            </div>
          )}
          {translationModels.map((m) => (
            <div key={m.name} className="kv">
              <span className="kv-k mono">{m.name}</span>
              <span className="kv-v">
                {humanBytes(m.sizeBytes)}
                {m.isDefault ? " · default" : ""}
              </span>
            </div>
          ))}
        </div>
        <div className="actions">
          <button
            className="btn ghost"
            onClick={() => void refreshTranslationEnv()}
          >
            Refresh
          </button>
          <button
            className="btn ghost"
            onClick={() => void refreshTranslationModels()}
          >
            Rescan
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>Python worker</h3>
        <Kv k="State" v={worker.state} />
        <Kv k="PID" v={worker.pid !== null ? String(worker.pid) : "—"} />
        <Kv k="Uptime" v={`${Math.round(worker.uptimeMs / 1000)} s`} />
        {worker.lastError ? <Kv k="Last error" v={worker.lastError} /> : null}
        {pingError ? <div className="error-panel small">{pingError}</div> : null}
        {ping ? (
          <>
            <Kv k="Ping" v={`ok · pid ${ping.pid} · uptime ${Math.round(ping.uptimeMs / 1000)}s`} />
          </>
        ) : null}
        {env ? (
          <>
            <Kv k="Python" v={env.python} />
            <Kv k="Platform" v={env.platform} />
            <Kv k="CPU cores" v={String(env.cpuCount)} />
            <Kv
              k="FFmpeg"
              v={env.ffmpegAvailable ? env.ffmpegVersion ?? "detected" : "not detected"}
            />
          </>
        ) : null}
        <button className="btn ghost" onClick={() => void refreshEnv()}>
          Ping worker
        </button>
      </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function Kv(props: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="kv">
      <span className="kv-k">{props.k}</span>
      <span className={"kv-v" + (props.mono ? " mono" : "")}>{props.v}</span>
    </div>
  );
}

function humanBytes(n: number): string {
  if (n <= 0) return "0 B";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  try {
    const obj = err as { message?: string; code?: string; hint?: string };
    if (obj?.message) {
      // Phase 12 — surface the structured `hint` field when Rust
      // provides one so users see actionable next steps instead of
      // bare error codes.
      const base = obj.code ? `${obj.code}: ${obj.message}` : obj.message;
      return obj.hint ? `${base}\n\nHint: ${obj.hint}` : base;
    }
  } catch {
    /* fall through */
  }
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

function YouTubeSettingsPanel() {
  const [state, setState] = useState<YouTubeConnectionState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.getYouTubeState().then(setState).catch((reason) => {
      setError(errorMessage(reason));
    });
  }, []);

  const disconnect = async () => {
    setBusy(true);
    setError(null);
    try {
      setState(await api.disconnectYouTube());
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h3>YouTube Account</h3>
      {state?.status === "connected" ? (
        <>
          <Kv
            k="Connected"
            v={state.account?.channelTitle ?? "YouTube channel"}
          />
          <Kv
            k="Channel ID"
            v={state.account?.channelId ?? "Unavailable"}
            mono
          />
          <div className="actions">
            <button className="btn danger" disabled={busy} onClick={disconnect}>
              {busy ? "Disconnecting…" : "Disconnect"}
            </button>
          </div>
          <p className="small muted">
            Disconnecting removes credentials from secure OS storage. Project
            publishing history is retained.
          </p>
        </>
      ) : (
        <p className="small muted">
          No YouTube account is connected. Connect from a project’s Publish
          panel.
        </p>
      )}
      {error && <div className="banner banner--error small">{error}</div>}
    </div>
  );
}

function ModelRow(props: { model: LocalModel }) {
  const { model } = props;
  const badge =
    model.status === "available"
      ? { label: "Available", cls: "ok" }
      : model.status === "missing"
        ? { label: "Missing", cls: "warn" }
        : { label: "Invalid", cls: "err" };
  return (
    <div className="model-row">
      <div className="model-row-main">
        <div className="model-row-name">
          <span className={`model-badge ${model.kind}`}>{model.kind}</span>
          <strong>{model.name}</strong>
          {model.language && (
            <span className="small" style={{ color: "var(--fg-muted)" }}>
              · {model.language}
            </span>
          )}
        </div>
        <div className="small" style={{ color: "var(--fg-muted)" }}>
          {model.engine ? `${model.engine} · ` : ""}
          {model.sizeBytes ? `${humanBytes(model.sizeBytes)} · ` : ""}
          {model.version ? `${model.version} · ` : ""}
          <span className="mono">{model.path ?? "(no path)"}</span>
        </div>
        {model.hint && (
          <div className="small" style={{ color: "var(--warn)" }}>
            {model.hint}
          </div>
        )}
      </div>
      <span className={`status-badge ${badge.cls}`}>{badge.label}</span>
    </div>
  );
}

function AddLocalModelButton(props: {
  busy: boolean;
  onImport: (spec: ImportModelSpec) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<ModelKind>("whisper");
  const [source, setSource] = useState<string>("");
  const [name, setName] = useState<string>("");

  async function pick() {
    const isDirectory = kind === "whisper" || kind === "voice";
    const filters =
      kind === "translation"
        ? [{ name: "GGUF", extensions: ["gguf"] }]
        : kind === "tts"
          ? [{ name: "TTS engine folder", extensions: [] }]
          : undefined;
    const picked = await pickModelSource(isDirectory, filters);
    if (picked) setSource(picked);
  }

  async function submit() {
    if (!source) return;
    await props.onImport({
      kind,
      sourcePath: source,
      name: name.trim() || null,
      strategy: "link",
    });
    setOpen(false);
    setSource("");
    setName("");
  }

  if (!open) {
    return (
      <button className="btn" onClick={() => setOpen(true)}>
        Add Local Model
      </button>
    );
  }
  return (
    <div className="import-model-form">
      <label className="stack small">
        Kind
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value as ModelKind)}
        >
          <option value="whisper">Whisper (folder)</option>
          <option value="translation">Translation (.gguf)</option>
          <option value="voice">Piper voice (folder)</option>
        </select>
      </label>
      <label className="stack small">
        Source path
        <div className="row" style={{ gap: 6 }}>
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="/path/to/model"
            style={{ flex: 1 }}
          />
          <button className="btn ghost small" onClick={() => void pick()}>
            Browse…
          </button>
        </div>
      </label>
      <label className="stack small">
        Name (optional)
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Derived from filename if empty"
        />
      </label>
      <div className="actions">
        <button
          className="btn"
          onClick={() => void submit()}
          disabled={!source || props.busy}
        >
          {props.busy ? "Importing…" : "Import"}
        </button>
        <button
          className="btn ghost"
          onClick={() => {
            setOpen(false);
            setSource("");
            setName("");
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

/* Phase 11 — light-touch runtime resource strip. Polls
 * `get_runtime_stats` every 3 seconds so users can see whether the
 * worker is holding on to RAM or a job is stuck. Nothing here is
 * expensive on the Rust side; the whole call is O(1) + one `ps`. */
function PerformanceMonitor() {
  const [stats, setStats] = useState<
    import("@/ipc/types").RuntimeStats | null
  >(null);
  useEffect(() => {
    let alive = true;
    const tick = () => {
      void api
        .getRuntimeStats()
        .then((s) => {
          if (alive) setStats(s);
        })
        .catch(() => {
          /* ignore transient errors */
        });
    };
    tick();
    const handle = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(handle);
    };
  }, []);
  if (!stats) {
    return (
      <div className="perf-monitor small" style={{ color: "var(--fg-muted)" }}>
        Runtime info loading…
      </div>
    );
  }
  return (
    <div className="perf-monitor">
      <div className="kv">
        <span className="kv-k">Active jobs</span>
        <span className="kv-v mono">
          {stats.activeJobs} ({stats.activeProjects} project
          {stats.activeProjects === 1 ? "" : "s"})
        </span>
      </div>
      <div className="kv">
        <span className="kv-k">App RAM</span>
        <span className="kv-v mono">{formatBytes(stats.hostRssBytes)}</span>
      </div>
      <div className="kv">
        <span className="kv-k">Worker RAM</span>
        <span className="kv-v mono">
          {formatBytes(stats.workerRssBytes)}
          {stats.workerUptimeSecs != null
            ? ` · up ${formatUptime(stats.workerUptimeSecs)}`
            : ""}
        </span>
      </div>
    </div>
  );
}

function formatBytes(v: number | null): string {
  if (v == null) return "—";
  const mb = v / (1024 * 1024);
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

function formatUptime(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${(secs / 3600).toFixed(1)}h`;
}

/* Phase 12 — Storage & Logs panel. Every action is scoped to an
 * app-owned directory (Rust enforces this), so the user can never
 * accidentally point us at their home folder. `clearCache` is
 * bytes-only (per-project data, models and outputs stay
 * untouched), `clearLogs` truncates rather than deletes so the
 * running tracing appender keeps its file handle. */
function StoragePanel() {
  const [stats, setStats] = useState<
    import("@/ipc/types").StorageStats | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<null | "cache" | "logs">(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.getStorageStats();
      setStats(s);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const reveal = async (
    kind: import("@/ipc/types").AppPathKind,
    label: string,
  ) => {
    try {
      await api.openAppPath(kind);
      setMsg(`Opened ${label} in file manager.`);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <div className="panel">
      <h3>Storage &amp; Logs</h3>
      <p className="small" style={{ color: "var(--fg-muted)" }}>
        Everything runs offline against the folders below.
        <em> Clear cache </em>
        only removes transient files — projects, models and
        rendered outputs are never touched.
      </p>

      <div className="storage-grid">
        <StorageRow
          label="App data"
          path={stats?.dataDir}
          bytes={stats?.dataBytes}
          onOpen={() => void reveal("data", "app data")}
        />
        <StorageRow
          label="Projects"
          path={stats?.projectsDir}
          bytes={stats?.projectsBytes}
          onOpen={() => void reveal("projects", "projects")}
        />
        <StorageRow
          label="Models"
          path={stats?.modelsDir}
          bytes={stats?.modelsBytes}
          onOpen={() => void reveal("models", "models")}
        />
        <StorageRow
          label="Cache"
          path={stats?.cacheDir}
          bytes={stats?.cacheBytes}
          onOpen={() => void reveal("cache", "cache")}
        />
        <StorageRow
          label="Logs"
          path={stats?.logDir}
          bytes={stats?.logBytes}
          onOpen={() => void reveal("log", "logs")}
        />
      </div>

      <div className="actions" style={{ marginTop: 8 }}>
        <button
          className="btn ghost"
          disabled={loading}
          onClick={() => void refresh()}
        >
          {loading ? "Recomputing…" : "Recompute sizes"}
        </button>
        <button
          className="btn ghost"
          disabled={busy !== null}
          onClick={async () => {
            setBusy("cache");
            setError(null);
            try {
              const freed = await api.clearCache();
              setMsg(`Cleared ${humanBytes(freed)} from cache.`);
              await refresh();
            } catch (err) {
              setError(errorMessage(err));
            } finally {
              setBusy(null);
            }
          }}
        >
          {busy === "cache" ? "Clearing…" : "Clear cache"}
        </button>
        <button
          className="btn ghost"
          disabled={busy !== null}
          onClick={async () => {
            setBusy("logs");
            setError(null);
            try {
              const n = await api.clearLogs();
              setMsg(`Truncated ${n} log file${n === 1 ? "" : "s"}.`);
              await refresh();
            } catch (err) {
              setError(errorMessage(err));
            } finally {
              setBusy(null);
            }
          }}
        >
          {busy === "logs" ? "Clearing…" : "Clear logs"}
        </button>
      </div>
      {msg && (
        <div className="small" style={{ color: "var(--ok)", marginTop: 6 }}>
          {msg}
        </div>
      )}
      {error && (
        <div className="error-panel small" style={{ marginTop: 6 }}>
          {error}
        </div>
      )}
    </div>
  );
}

function StorageRow(props: {
  label: string;
  path?: string;
  bytes?: number;
  onOpen: () => void;
}) {
  return (
    <div className="storage-row">
      <div className="storage-row-main">
        <div className="storage-row-label">{props.label}</div>
        <div className="mono small storage-row-path">{props.path ?? "—"}</div>
      </div>
      <div className="storage-row-size mono small">
        {props.bytes != null ? humanBytes(props.bytes) : "—"}
      </div>
      <button className="btn ghost small" onClick={props.onOpen}>
        Open
      </button>
    </div>
  );
}
