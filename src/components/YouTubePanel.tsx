import { useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  assetUrl,
  onYouTubeUpload,
  pickYouTubeThumbnail,
} from "@/ipc/bridge";
import {
  isAppError,
  TRANSLATION_LANGUAGES,
  type RenderSummary,
  type SubtitleSummary,
  type YouTubeAccount,
  type YouTubeConnectionState,
  type YouTubePlaylist,
  type YouTubePrivacyStatus,
  type YouTubePublishingHistoryEntry,
  type YouTubeUploadSnapshot,
} from "@/ipc/types";

interface YouTubePanelProps {
  projectId: string;
  projectName: string;
  sourceLanguage: string;
  targetLanguage: string;
  render: RenderSummary | null;
  subtitles: SubtitleSummary | null;
}

const CATEGORIES = [
  ["1", "Film & Animation"],
  ["2", "Autos & Vehicles"],
  ["10", "Music"],
  ["15", "Pets & Animals"],
  ["17", "Sports"],
  ["19", "Travel & Events"],
  ["20", "Gaming"],
  ["22", "People & Blogs"],
  ["23", "Comedy"],
  ["24", "Entertainment"],
  ["25", "News & Politics"],
  ["26", "How-to & Style"],
  ["27", "Education"],
  ["28", "Science & Technology"],
  ["29", "Nonprofits & Activism"],
] as const;

const ACTIVE_STATES = [
  "waiting",
  "connecting",
  "preparing",
  "uploading",
  "processing",
];

export function YouTubePanel(props: YouTubePanelProps) {
  const {
    projectId,
    projectName,
    sourceLanguage,
    targetLanguage,
    render,
    subtitles,
  } = props;
  const [connection, setConnection] = useState<YouTubeConnectionState | null>(
    null,
  );
  const [accounts, setAccounts] = useState<YouTubeAccount[]>([]);
  const [playlists, setPlaylists] = useState<YouTubePlaylist[]>([]);
  const [history, setHistory] = useState<YouTubePublishingHistoryEntry[]>([]);
  const [queue, setQueue] = useState<YouTubeUploadSnapshot[]>([]);
  const [title, setTitle] = useState(projectName.slice(0, 100));
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagDraft, setTagDraft] = useState("");
  const [privacy, setPrivacy] =
    useState<YouTubePrivacyStatus>("private");
  const [categoryId, setCategoryId] = useState("22");
  const [playlistId, setPlaylistId] = useState("");
  const [language, setLanguage] = useState(targetLanguage);
  const [thumbnailPath, setThumbnailPath] = useState<string | null>(null);
  const [frameTime, setFrameTime] = useState(0);
  const [translatedSubtitles, setTranslatedSubtitles] = useState(true);
  const [originalSubtitles, setOriginalSubtitles] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [starting, setStarting] = useState(false);
  const [generatingThumbnail, setGeneratingThumbnail] = useState(false);
  const [confirmPublic, setConfirmPublic] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const receivedUploadEvent = useRef(false);

  useEffect(() => {
    setTitle(projectName.slice(0, 100));
    setDescription("");
    setTags([]);
    setTagDraft("");
    setPrivacy("private");
    setCategoryId("22");
    setPlaylistId("");
    setLanguage(targetLanguage);
    setThumbnailPath(null);
    setFrameTime(0);
    setTranslatedSubtitles(true);
    setOriginalSubtitles(false);
    setQueue([]);
    receivedUploadEvent.current = false;
    setError(null);
  }, [projectId, projectName, targetLanguage]);

  useEffect(() => {
    let alive = true;
    void Promise.all([
      api.getYouTubeState(),
      api.listYouTubeAccounts(),
      api.listYouTubeUploads(),
      api.listYouTubeHistory(projectId),
    ])
      .then(async ([state, nextAccounts, uploads, nextHistory]) => {
        if (!alive) return;
        setConnection(state);
        setAccounts(nextAccounts);
        const fetched = uploads.filter(
          (upload) => upload.projectId === projectId,
        );
        setQueue((current) => {
          if (!receivedUploadEvent.current) {
            return fetched.sort((a, b) =>
              b.createdAt.localeCompare(a.createdAt),
            );
          }
          const liveIds = new Set(current.map((upload) => upload.id));
          return [
            ...current,
            ...fetched.filter((upload) => !liveIds.has(upload.id)),
          ].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
        });
        setHistory(nextHistory);
        if (state.status === "connected" && !state.offline) {
          const nextPlaylists = await api.listYouTubePlaylists();
          if (alive) setPlaylists(nextPlaylists);
        } else {
          setPlaylists([]);
        }
      })
      .catch((reason) => {
        if (alive) setError(errorMessage(reason));
      });
    const unlisten = onYouTubeUpload((next) => {
      if (next.projectId !== projectId) return;
      receivedUploadEvent.current = true;
      setQueue((current) => {
        const without = current.filter((upload) => upload.id !== next.id);
        return [next, ...without].sort((a, b) =>
          b.createdAt.localeCompare(a.createdAt),
        );
      });
      if (next.state === "completed") {
        void api.listYouTubeHistory(projectId).then(setHistory);
      }
    });
    return () => {
      alive = false;
      void unlisten.then((stop) => stop());
    };
  }, [projectId]);

  const currentUpload =
    queue.find((upload) =>
      ["connecting", "preparing", "uploading", "processing"].includes(
        upload.state,
      ),
    ) ??
    queue[0] ??
    null;
  const busy = queue.some((upload) => ACTIVE_STATES.includes(upload.state));
  const hasVideo = render?.status === "ready" && !!render.absolutePath;
  const hasTitle = title.trim().length > 0;
  const hasDescription = description.trim().length > 0;
  const descriptionBytes = new TextEncoder().encode(description).length;
  const tagCharacters = tags.join(",").length;
  const connected = connection?.status === "connected";
  const canPublish =
    connected &&
    !connection?.offline &&
    hasVideo &&
    hasTitle &&
    hasDescription &&
    !starting;
  const progressPercent = Math.round((currentUpload?.progress ?? 0) * 100);
  const translatedAvailable =
    !!subtitles && subtitles.translatedCount > 0;
  const originalAvailable = !!subtitles && subtitles.segmentCount > 0;

  const checklist = useMemo(
    () => [
      ["Video selected", hasVideo, false],
      ["Title", hasTitle, false],
      ["Description", hasDescription, false],
      ["Privacy selected", !!privacy, false],
      ["YouTube account connected", connected, false],
      ["Thumbnail valid", !!thumbnailPath, true],
      ["Playlist", !!playlistId, true],
      [
        "Subtitles",
        translatedSubtitles || originalSubtitles,
        true,
      ],
    ] as const,
    [
      connected,
      hasDescription,
      hasTitle,
      hasVideo,
      originalSubtitles,
      playlistId,
      privacy,
      thumbnailPath,
      translatedSubtitles,
    ],
  );

  const refreshAccountData = async (state: YouTubeConnectionState) => {
    setConnection(state);
    setAccounts(await api.listYouTubeAccounts());
    setPlaylists(
      state.status === "connected" && !state.offline
        ? await api.listYouTubePlaylists()
        : [],
    );
  };

  const connect = async () => {
    setConnecting(true);
    setError(null);
    try {
      await refreshAccountData(await api.connectYouTube());
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setConnecting(false);
    }
  };

  const disconnect = async () => {
    setError(null);
    try {
      await refreshAccountData(await api.disconnectYouTube());
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const selectAccount = async (accountId: string) => {
    setError(null);
    try {
      await refreshAccountData(await api.selectYouTubeAccount(accountId));
      setPlaylistId("");
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const addTag = () => {
    const next = tagDraft.trim().replace(/^#+/, "");
    const proposed = [...tags, next].join(",").length;
    if (
      next &&
      !tags.includes(next) &&
      tags.length < 100 &&
      proposed <= 500
    ) {
      setTags((current) => [...current, next]);
    }
    setTagDraft("");
  };

  const selectThumbnail = async () => {
    const path = await pickYouTubeThumbnail();
    if (!path) return;
    setError(null);
    try {
      const validated = await api.validateYouTubeThumbnail(path);
      setThumbnailPath(validated.path);
    } catch (reason) {
      setThumbnailPath(null);
      setError(errorMessage(reason));
    }
  };

  const generateThumbnail = async () => {
    setGeneratingThumbnail(true);
    setError(null);
    try {
      const result = await api.generateYouTubeThumbnail(projectId, frameTime);
      setThumbnailPath(result.path);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setGeneratingThumbnail(false);
    }
  };

  const publish = async () => {
    if (starting) return;
    setStarting(true);
    setError(null);
    try {
      const upload = await api.startYouTubeUpload(
        projectId,
        {
          title: title.trim(),
          description,
          tags,
          privacyStatus: privacy,
          categoryId,
          defaultLanguage: language || null,
        },
        {
          playlistId: playlistId || null,
          thumbnailPath,
          publishTranslatedSubtitles:
            translatedSubtitles && translatedAvailable,
          publishOriginalSubtitles: originalSubtitles && originalAvailable,
        },
      );
      setQueue((current) => [upload, ...current]);
      setConfirmPublic(false);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setStarting(false);
    }
  };

  const requestPublish = () => {
    if (privacy === "public") setConfirmPublic(true);
    else void publish();
  };

  return (
    <div className="youtube-studio">
      <header className="youtube-heading">
        <div>
          <strong>Publish to YouTube</strong>
          <div className="small muted">
            Optional publishing. Local render and export remain independent.
          </div>
        </div>
        {connected && <span className="badge badge--ok">Connected</span>}
      </header>

      {!connection?.configured && (
        <div className="banner banner--warn small">
          Configure <code>LMT_YOUTUBE_CLIENT_ID</code> to enable publishing.
        </div>
      )}
      {connection?.offline && (
        <div className="banner banner--muted small">
          YouTube is offline. Connect to the Internet to publish. Your local
          project remains available.
        </div>
      )}

      {!connected ? (
        <div className="youtube-connect">
          <span className="small muted">No YouTube account connected.</span>
          <button
            className="btn"
            onClick={connect}
            disabled={
              connecting || !connection?.configured || connection?.offline
            }
          >
            {connecting ? "Connecting…" : "Connect YouTube"}
          </button>
        </div>
      ) : (
        <>
          <section className="youtube-publish-section">
            <h4>Publishing as</h4>
            <div className="youtube-account">
              {connection.account?.thumbnailUrl && !connection.offline ? (
                <img
                  className="youtube-avatar"
                  src={connection.account.thumbnailUrl}
                  alt=""
                />
              ) : (
                <div className="youtube-avatar youtube-avatar--fallback">YT</div>
              )}
              <div>
                <strong>
                  {connection.account?.channelTitle ?? "YouTube channel"}
                </strong>
                <span className="small muted">✓ Connected account</span>
              </div>
              {accounts.length > 1 && (
                <select
                  value={connection.account?.id ?? ""}
                  disabled={busy}
                  onChange={(event) => void selectAccount(event.target.value)}
                >
                  {accounts.map((account) => (
                    <option key={account.id} value={account.id}>
                      {account.channelTitle ?? account.channelId ?? account.id}
                    </option>
                  ))}
                </select>
              )}
              <button
                className="btn ghost small"
                onClick={disconnect}
                disabled={busy}
              >
                Disconnect
              </button>
            </div>
          </section>

          <section className="youtube-publish-section">
            <h4>Video</h4>
            <div className="youtube-video-card">
              <div className="youtube-video-icon">▶</div>
              <div>
                <strong>
                  {fileName(render?.absolutePath) || "Render a movie first"}
                </strong>
                <span className="small muted">
                  {render?.sizeBytes != null
                    ? humanBytes(render.sizeBytes)
                    : "—"}{" "}
                  ·{" "}
                  {render?.durationSecs != null
                    ? formatDuration(render.durationSecs)
                    : "—"}
                </span>
                <code className="mono small">
                  {render?.absolutePath ?? ""}
                </code>
              </div>
            </div>
          </section>

          <section className="youtube-publish-section">
            <h4>Details</h4>
            <div className="youtube-form">
              <label className="youtube-wide">
                <span>
                  Title <small>{title.length}/100</small>
                </span>
                <input
                  type="text"
                  maxLength={100}
                  value={title}
                  disabled={busy}
                  onChange={(event) => setTitle(event.target.value)}
                />
              </label>
              <label className="youtube-wide">
                <span>
                  Description <small>{descriptionBytes}/5000 bytes</small>
                </span>
                <textarea
                  rows={6}
                  maxLength={5000}
                  value={description}
                  disabled={busy}
                  onChange={(event) => {
                    if (
                      new TextEncoder().encode(event.target.value).length <=
                      5000
                    ) {
                      setDescription(event.target.value);
                    }
                  }}
                />
              </label>
              <label className="youtube-wide">
                <span>
                  Tags <small>{tagCharacters}/500 characters</small>
                </span>
                <div className="youtube-tag-editor">
                  {tags.map((tag) => (
                    <span className="youtube-tag" key={tag}>
                      {tag}
                      <button
                        type="button"
                        onClick={() =>
                          setTags((current) =>
                            current.filter((value) => value !== tag),
                          )
                        }
                        aria-label={`Remove ${tag}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  <input
                    value={tagDraft}
                    disabled={busy}
                    placeholder="Add tag and press Enter"
                    onChange={(event) => setTagDraft(event.target.value)}
                    onBlur={addTag}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === ",") {
                        event.preventDefault();
                        addTag();
                      }
                    }}
                  />
                </div>
              </label>
              <label>
                <span>Category</span>
                <select
                  value={categoryId}
                  disabled={busy}
                  onChange={(event) => setCategoryId(event.target.value)}
                >
                  {CATEGORIES.map(([id, name]) => (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Privacy</span>
                <select
                  value={privacy}
                  disabled={busy}
                  onChange={(event) =>
                    setPrivacy(event.target.value as YouTubePrivacyStatus)
                  }
                >
                  <option value="private">Private</option>
                  <option value="unlisted">Unlisted</option>
                  <option value="public">Public</option>
                </select>
              </label>
              <label>
                <span>Playlist</span>
                <select
                  value={playlistId}
                  disabled={busy}
                  onChange={(event) => setPlaylistId(event.target.value)}
                >
                  <option value="">No playlist</option>
                  {playlists.map((playlist) => (
                    <option key={playlist.id} value={playlist.id}>
                      {playlist.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Video language</span>
                <select
                  value={language}
                  disabled={busy}
                  onChange={(event) => setLanguage(event.target.value)}
                >
                  {TRANSLATION_LANGUAGES.map((item) => (
                    <option key={item.code} value={item.code}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </section>

          <section className="youtube-publish-section">
            <h4>Thumbnail</h4>
            <div className="youtube-thumbnail-editor">
              <div className="youtube-thumbnail-preview">
                {thumbnailPath ? (
                  <img src={assetUrl(thumbnailPath)} alt="YouTube thumbnail" />
                ) : (
                  <span>No custom thumbnail selected</span>
                )}
              </div>
              <div className="youtube-thumbnail-actions">
                <button className="btn" onClick={selectThumbnail} disabled={busy}>
                  {thumbnailPath ? "Replace" : "Select image"}
                </button>
                {thumbnailPath && (
                  <button
                    className="btn ghost"
                    onClick={() => setThumbnailPath(null)}
                    disabled={busy}
                  >
                    Remove
                  </button>
                )}
                <label>
                  <span>Frame time (seconds)</span>
                  <input
                    type="number"
                    min={0}
                    max={render?.durationSecs ?? undefined}
                    step={0.1}
                    value={frameTime}
                    disabled={busy}
                    onChange={(event) =>
                      setFrameTime(Number(event.target.value) || 0)
                    }
                  />
                </label>
                <button
                  className="btn"
                  onClick={generateThumbnail}
                  disabled={!hasVideo || busy || generatingThumbnail}
                >
                  {generatingThumbnail
                    ? "Generating…"
                    : "Generate from video frame"}
                </button>
              </div>
            </div>
          </section>

          <section className="youtube-publish-section">
            <h4>Subtitles</h4>
            <div className="youtube-subtitle-options">
              <label className="check">
                <input
                  type="checkbox"
                  checked={translatedSubtitles && translatedAvailable}
                  disabled={!translatedAvailable || busy}
                  onChange={(event) =>
                    setTranslatedSubtitles(event.target.checked)
                  }
                />
                Upload translated subtitles ({languageName(targetLanguage)})
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={originalSubtitles && originalAvailable}
                  disabled={!originalAvailable || busy}
                  onChange={(event) =>
                    setOriginalSubtitles(event.target.checked)
                  }
                />
                Upload original subtitles ({languageName(sourceLanguage)})
              </label>
              {!subtitles && (
                <span className="small muted">
                  Build project subtitles before attaching caption tracks.
                </span>
              )}
            </div>
          </section>

          <section className="youtube-publish-section">
            <h4>Publishing checklist</h4>
            <div className="youtube-checklist">
              {checklist.map(([label, valid, optional]) => (
                <div key={label}>
                  <span className={valid ? "check-ok" : "check-empty"}>
                    {valid ? "✓" : "○"}
                  </span>
                  <span>{label}</span>
                  {optional && <small>Optional</small>}
                </div>
              ))}
            </div>
            <div className="youtube-actions">
              <button
                className="btn primary"
                onClick={requestPublish}
                disabled={!canPublish}
              >
                {starting ? "Preparing…" : "Publish"}
              </button>
            </div>
          </section>

          <UploadQueue
            queue={queue}
            onCancel={(id) => void api.cancelYouTubeUpload(id)}
            onRetry={(id) =>
              void api.retryYouTubeUpload(id).then((next) =>
                setQueue((current) => [
                  next,
                  ...current.filter((item) => item.id !== id),
                ]),
              )
            }
          />

          <PublishingHistory history={history} />
        </>
      )}

      {confirmPublic && (
        <div className="youtube-confirm" role="alertdialog" aria-modal="true">
          <div>
            <strong>Publish publicly?</strong>
            <p>
              You are about to publish this video publicly. Anyone can find
              and watch it.
            </p>
            <div className="actions">
              <button className="btn" onClick={() => setConfirmPublic(false)}>
                Cancel
              </button>
              <button className="btn primary" onClick={() => void publish()}>
                Publish Publicly
              </button>
            </div>
          </div>
        </div>
      )}

      {!hasVideo && (
        <div className="empty-state small">
          Render the final local video first. Publishing never starts
          automatically.
        </div>
      )}
      {error && <div className="banner banner--error small">{error}</div>}
      {currentUpload?.errorMessage && (
        <div className="banner banner--error small">
          {currentUpload.errorMessage}
        </div>
      )}
      {currentUpload && ACTIVE_STATES.includes(currentUpload.state) && (
        <div className="youtube-progress">
          <span className="small">{uploadStateLabel(currentUpload.state)}</span>
          <div className="progress-row">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <span className="progress-value">{progressPercent}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

function UploadQueue(props: {
  queue: YouTubeUploadSnapshot[];
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
}) {
  if (props.queue.length === 0) return null;
  return (
    <section className="youtube-publish-section">
      <h4>Upload queue</h4>
      <div className="youtube-queue">
        {props.queue.map((upload) => {
          const pct = Math.round(upload.progress * 100);
          const active = ACTIVE_STATES.includes(upload.state);
          return (
            <div className="youtube-queue-item" key={upload.id}>
              <span className={`queue-dot queue-dot--${upload.state}`} />
              <div>
                <strong>{upload.title || fileName(upload.filePath)}</strong>
                <span className="small muted">
                  {uploadStateLabel(upload.state)}
                  {upload.state === "uploading" ? ` ${pct}%` : ""}
                  {" · "}
                  {upload.privacyStatus}
                </span>
                {upload.videoId && (
                  <span className="small">
                    YouTube ID: <code>{upload.videoId}</code>
                  </span>
                )}
                {upload.assetSteps.some((step) => step.state === "failed") && (
                  <span className="small error-text">
                    Some optional publishing assets failed.
                  </span>
                )}
              </div>
              {active && (
                <button
                  className="btn ghost tiny"
                  onClick={() => props.onCancel(upload.id)}
                >
                  Cancel
                </button>
              )}
              {upload.state === "failed" && upload.canRetry && (
                <button
                  className="btn tiny"
                  onClick={() => props.onRetry(upload.id)}
                >
                  Retry
                </button>
              )}
              {upload.state === "completed" && upload.videoId && (
                <button
                  className="btn tiny"
                  onClick={() => void api.openYouTubeVideo(upload.videoId!)}
                >
                  Open
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function PublishingHistory({
  history,
}: {
  history: YouTubePublishingHistoryEntry[];
}) {
  if (history.length === 0) return null;
  return (
    <section className="youtube-publish-section">
      <h4>Publishing history</h4>
      <div className="youtube-history">
        {history.map((entry) => (
          <div key={`${entry.videoId}-${entry.uploadedAt}`}>
            <span className="check-ok">✓</span>
            <div>
              <strong>{entry.title}</strong>
              <span className="small muted">
                {capitalize(entry.privacyStatus)} ·{" "}
                {new Date(entry.uploadedAt).toLocaleDateString()}
              </span>
            </div>
            <button
              className="btn ghost tiny"
              onClick={() => void api.openYouTubeVideo(entry.videoId)}
            >
              Open on YouTube
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

function errorMessage(reason: unknown): string {
  if (isAppError(reason)) return reason.hint ?? reason.message;
  return reason instanceof Error
    ? reason.message
    : "The YouTube action could not be completed.";
}

function uploadStateLabel(state: YouTubeUploadSnapshot["state"]): string {
  const labels: Record<YouTubeUploadSnapshot["state"], string> = {
    idle: "Idle",
    waiting: "Waiting",
    connecting: "Connecting",
    preparing: "Preparing",
    uploading: "Uploading",
    processing: "Publishing assets",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
  };
  return labels[state];
}

function humanBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(
    Math.floor(Math.log(value) / Math.log(1024)),
    units.length - 1,
  );
  return `${(value / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds: number): string {
  const value = Math.max(0, Math.round(seconds));
  const h = Math.floor(value / 3600);
  const m = Math.floor((value % 3600) / 60);
  const s = value % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
    : `${m}:${s.toString().padStart(2, "0")}`;
}

function fileName(path: string | null | undefined): string {
  return path?.split(/[\\/]/).pop() ?? "";
}

function languageName(code: string): string {
  return (
    TRANSLATION_LANGUAGES.find((language) => language.code === code)?.label ??
    code
  );
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
