# YouTube integration (Phase 13)

YouTube publishing is optional. Whisper, translation, subtitles, TTS,
sync, mixing, rendering, and local export remain fully local and never
call Google.

## Google Cloud setup

1. Create or select a Google Cloud project.
2. Enable **YouTube Data API v3**.
3. Configure the OAuth consent screen. Add test users while the app is in
   testing mode. Public distribution may require Google's verification
   for the narrowly scoped YouTube permissions.
4. Create an OAuth client with application type **Desktop app**.
5. Provide the client ID as `LMT_YOUTUBE_CLIENT_ID` when building or
   launching the Tauri application.
6. If the downloaded Desktop client configuration contains a client
   secret, provide it at runtime as `LMT_YOUTUBE_CLIENT_SECRET`. It is
   intentionally never compiled into the application. Never commit it.

The app uses Google's installed-app loopback flow. It opens the system
browser and listens temporarily on a random `127.0.0.1` port. Users do
not paste tokens into the app. It requests only the publishing
permissions required by the implemented workflow:

```text
https://www.googleapis.com/auth/youtube.upload
https://www.googleapis.com/auth/youtube.readonly
https://www.googleapis.com/auth/youtube.force-ssl
```

`youtube.upload` is used only by the resumable upload. `youtube.readonly`
is used only to identify and display the connected channel through
`channels.list(mine=true)` and retrieve playlist names. `youtube.force-ssl`
is required to attach captions, set thumbnails, and add a published
video to a selected playlist.

After upgrading from Phase 13, disconnect and reconnect the YouTube
account once so Google can ask for the additional publishing permission.

Use `.env.youtube.example` only as a template. The client ID can be
supplied at runtime during development or at compile time for a packaged
build. A desktop OAuth client ID is public configuration. A client
secret is accepted only from the runtime environment; it and all user
tokens must stay out of source control and logs.

## Credential storage

Refresh tokens are stored by the Rust host in the operating system's
secure credential service:

- macOS Keychain
- Windows Credential Manager
- Linux Secret Service

Access tokens stay in backend memory and never cross Tauri IPC into
React. The local `youtube-accounts.json` registry contains only
non-secret channel metadata and an active account ID. Its schema is a
list so additional accounts can be supported later without changing the
token model.

Disconnecting deletes the selected account's refresh token and local
metadata.

## Upload workflow

After a local render completes, open **Render → Publish**, connect a
channel, enter metadata, and explicitly click **Upload to YouTube**.
Privacy defaults to **Private**.

Uploads use the official YouTube Data API resumable protocol. Rust reads
the rendered file in 8 MiB chunks; the whole movie is never loaded into
RAM. The frontend receives throttled byte counts from the backend and
shows real progress. Cancel affects only that upload.

An interrupted upload retains its resumable session in backend memory.
**Resume / Retry** asks YouTube for the committed byte offset before
continuing, so a transient failure does not restart a large upload.
Session URLs are intentionally not persisted to plaintext storage;
after a full app restart, start a new upload.

The app resolves the upload source from the project's current render
manifest. React cannot pass an arbitrary path to the upload command.

## Offline and failure behavior

When Offline Mode is enabled, connect, upload, and retry are blocked.
All local processing and local export remain available. Network,
authentication, permission, quota, metadata, and API failures are
translated to user-facing errors; raw API responses and credentials are
not shown or logged.

No upload starts automatically after rendering, and no analytics or
telemetry are included.

## Publishing Studio (Phase 14)

The project Render section includes a publishing workflow for:

- title, description, compact tags, readable category, privacy, and
  video language;
- cached playlist selection;
- local JPG, JPEG, PNG, or WebP thumbnail selection and validation;
- local thumbnail generation from an explicit FFmpeg frame time;
- translated and original caption tracks generated from the existing
  canonical subtitle document;
- a one-at-a-time upload queue, resumable retries, and per-asset status;
- lightweight project history stored at
  `<project>/output/youtube-publishing-history.json`.

WebP thumbnails are converted locally to JPEG before upload. Caption
SRT files and converted thumbnails are temporary publishing assets
under `<project>/output/youtube-assets/`; the source movie is never
copied.

Public privacy requires a separate confirmation in the editor. Private
remains the default. Publishing history contains no credentials and is
retained when an account is disconnected.
