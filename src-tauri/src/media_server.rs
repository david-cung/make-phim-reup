//! A loopback HTTP server for feeding `<video>` and `<audio>`.
//!
//! Custom URI schemes cannot play media in WebKit. Tauri's `asset://` is
//! one, and so is anything we could register ourselves: WebKit hands media
//! loading to a pipeline that never calls the scheme handler, so the
//! element fails with `MEDIA_ERR_SRC_NOT_SUPPORTED` before decoding a
//! frame. Both bugs are open and unfixed — webkit.org/b/146351 and
//! webkit.org/b/119469 — and the byte ranges media elements need are not
//! the problem, because no request is ever made.
//!
//! Ordinary HTTP over 127.0.0.1 goes through the normal loading path
//! instead, where ranges and seeking work. The Secure Contexts spec treats
//! loopback as trustworthy, so the webview does not flag it as mixed
//! content.
//!
//! The listener takes the first free port from a small reserved set on
//! 127.0.0.1. WebKit rejects wildcard ports in CSP media sources, so every
//! candidate is listed explicitly in `tauri.conf.json`. Every request must
//! also carry a token minted at startup, which keeps other local processes
//! from reading files through it. Beyond the token, reach is deliberately
//! as broad as the `assetProtocol` scope it replaces, since source videos
//! live wherever the user keeps them.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use once_cell::sync::OnceCell;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use uuid::Uuid;

/// `http://127.0.0.1:<port>/media?token=<token>`, once the server is up.
static BASE_URL: OnceCell<String> = OnceCell::new();

/// Path component every request must use.
const ROUTE: &str = "/media";

/// Explicitly mirrored in `tauri.conf.json`'s `media-src` and
/// `connect-src`. A short pool allows multiple dev/release instances
/// without relying on a wildcard port that WebKit's CSP rejects.
const MEDIA_PORTS: &[u16] = &[43120, 43121, 43122, 43123, 43124, 43125, 43126, 43127];

/// Bytes moved per write while streaming a response body.
const CHUNK_BYTES: usize = 64 * 1024;

/// Cap on the request head we are willing to buffer.
const MAX_HEAD_BYTES: usize = 16 * 1024;

/// Start the server and return its base URL.
///
/// Safe to call more than once; later calls hand back the running
/// server's URL rather than binding a second listener.
pub async fn start() -> std::io::Result<String> {
    if let Some(url) = BASE_URL.get() {
        return Ok(url.clone());
    }

    let mut listener = None;
    let mut last_error = None;
    for port in MEDIA_PORTS {
        match TcpListener::bind(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            *port,
        ))
        .await
        {
            Ok(bound) => {
                listener = Some(bound);
                break;
            }
            Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
                last_error = Some(err);
            }
            Err(err) => return Err(err),
        }
    }
    let listener = listener.ok_or_else(|| {
        last_error.unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AddrNotAvailable,
                "no media server ports are configured",
            )
        })
    })?;
    let port = listener.local_addr()?.port();
    let token = format!(
        "{}{}",
        Uuid::new_v4().simple(),
        Uuid::new_v4().simple()
    );
    let base = format!("http://127.0.0.1:{port}{ROUTE}?token={token}");

    // A second caller may have won the race; if so, keep theirs and drop
    // this listener.
    if let Err(existing) = BASE_URL.set(base.clone()) {
        return Ok(existing);
    }

    tracing::info!(port, "media server listening on loopback");
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let token = token.clone();
                    tokio::spawn(async move {
                        if let Err(err) = serve(stream, &token).await {
                            // A player that seeks or closes a tab drops the
                            // connection mid-write, which is routine.
                            tracing::debug!(%err, "media server connection ended");
                        }
                    });
                }
                Err(err) => {
                    tracing::warn!(%err, "media server accept failed");
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }
    });

    Ok(base)
}

/// The running server's base URL, if it has been started.
pub fn base_url() -> Option<String> {
    BASE_URL.get().cloned()
}

async fn serve(mut stream: TcpStream, token: &str) -> std::io::Result<()> {
    let head = match read_head(&mut stream).await? {
        Some(head) => head,
        None => return Ok(()), // client hung up before sending anything
    };

    let Some(request) = Request::parse(&head) else {
        return respond_status(&mut stream, 400, "bad request").await;
    };
    if request.token.as_deref() != Some(token) {
        tracing::warn!("media server rejected a request with a bad token");
        return respond_status(&mut stream, 403, "forbidden").await;
    }
    if request.preflight {
        tracing::info!("media server accepted browser preflight");
        return respond_preflight(&mut stream).await;
    }
    if request.probe {
        tracing::info!("WebKit media probe reached loopback server");
        return respond_bytes(&mut stream, "audio/wav", &probe_wav()).await;
    }
    let Some(path) = request.path else {
        return respond_status(&mut stream, 400, "missing path").await;
    };
    if !path.is_absolute() {
        return respond_status(&mut stream, 400, "path must be absolute").await;
    }

    let file = match tokio::fs::File::open(&path).await {
        Ok(file) => file,
        Err(err) => {
            tracing::warn!(path = %path.display(), %err, "media server open failed");
            let status = match err.kind() {
                std::io::ErrorKind::NotFound => 404,
                std::io::ErrorKind::PermissionDenied => 403,
                _ => 500,
            };
            return respond_status(&mut stream, status, "cannot open file").await;
        }
    };
    let total = file.metadata().await?.len();
    let range = request.range.as_deref().and_then(|r| parse_range(r, total));

    if request.range.is_some() && range.is_none() && total > 0 {
        // A range we cannot satisfy has its own status, and players rely on
        // it to discover the real length.
        let mut headers = format!("Content-Range: bytes */{total}\r\n");
        headers.push_str("Accept-Ranges: bytes\r\n");
        return respond(&mut stream, 416, "text/plain", 0, Some(headers), None).await;
    }

    let (start, end) = range.unwrap_or((0, total.saturating_sub(1)));
    let length = if total == 0 { 0 } else { end - start + 1 };
    let extra = range.map(|(start, end)| {
        format!(
            "Content-Range: bytes {start}-{end}/{total}\r\nAccept-Ranges: bytes\r\n"
        )
    });
    let status = if range.is_some() { 206 } else { 200 };
    tracing::debug!(
        path = %path.display(),
        status,
        range = request.range.as_deref().unwrap_or("none"),
        start,
        end,
        length,
        head = request.head_only,
        "media server serving request"
    );
    let body = if request.head_only {
        None
    } else {
        Some((file, start, length))
    };
    respond(
        &mut stream,
        status,
        mime_for(&path),
        length,
        Some(extra.unwrap_or_else(|| "Accept-Ranges: bytes\r\n".to_string())),
        body,
    )
    .await
}

/// Read up to the blank line that ends the request head.
async fn read_head(stream: &mut TcpStream) -> std::io::Result<Option<String>> {
    let mut reader = BufReader::new(stream);
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while reader.read_exact(&mut byte).await.is_ok() {
        head.push(byte[0]);
        if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
            return Ok(Some(String::from_utf8_lossy(&head).into_owned()));
        }
        if head.len() > MAX_HEAD_BYTES {
            return Ok(None);
        }
    }
    Ok(if head.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&head).into_owned())
    })
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Request {
    path: Option<PathBuf>,
    token: Option<String>,
    range: Option<String>,
    head_only: bool,
    preflight: bool,
    probe: bool,
}

impl Request {
    fn parse(head: &str) -> Option<Self> {
        let mut lines = head.lines();
        let mut parts = lines.next()?.split_whitespace();
        let method = parts.next()?;
        let target = parts.next()?;
        let head_only = match method {
            "GET" => false,
            "HEAD" => true,
            "OPTIONS" => false,
            _ => return None,
        };

        let (route, query) = target.split_once('?').unwrap_or((target, ""));
        if route != ROUTE {
            return None;
        }

        let mut request = Self {
            head_only,
            preflight: method == "OPTIONS",
            ..Default::default()
        };
        for pair in query.split('&') {
            let Some((key, value)) = pair.split_once('=') else {
                continue;
            };
            match key {
                "path" => request.path = percent_decode(value).map(PathBuf::from),
                "token" => request.token = percent_decode(value),
                "probe" => request.probe = value == "1",
                _ => {}
            }
        }

        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                if name.trim().eq_ignore_ascii_case("range") {
                    request.range = Some(value.trim().to_string());
                }
            }
        }
        Some(request)
    }
}

/// Decode `%XX` escapes. `+` stays literal: these are path values, not
/// form fields, and a filename may legitimately contain one.
fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = input.get(i + 1..i + 3)?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Parse a single-range `Range: bytes=…` header, clamped to the file.
///
/// Multi-range requests return `None`, which is answered as an ordinary
/// full-body response: players do not ask for them, and a
/// `multipart/byteranges` body is not worth carrying.
fn parse_range(header: &str, total: u64) -> Option<(u64, u64)> {
    if total == 0 {
        return None;
    }
    let spec = header.trim().strip_prefix("bytes=")?.trim();
    if spec.contains(',') {
        return None;
    }
    let (start_text, end_text) = spec.split_once('-')?;
    let last = total - 1;

    let (start, end) = match (start_text.trim(), end_text.trim()) {
        // `bytes=-500` — the final 500 bytes.
        ("", suffix) => {
            let count: u64 = suffix.parse().ok()?;
            if count == 0 {
                return None;
            }
            (total.saturating_sub(count), last)
        }
        // `bytes=500-` — from 500 to the end.
        (start, "") => (start.parse().ok()?, last),
        (start, end) => (start.parse().ok()?, end.parse::<u64>().ok()?.min(last)),
    };

    if start > last || start > end {
        return None;
    }
    Some((start, end))
}

async fn respond_status(
    stream: &mut TcpStream,
    status: u16,
    message: &str,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cross-Origin-Resource-Policy: cross-origin\r\n\
         Connection: close\r\n\r\n",
        reason = reason(status),
        len = message.len(),
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(message.as_bytes()).await?;
    stream.flush().await
}

async fn respond_preflight(stream: &mut TcpStream) -> std::io::Result<()> {
    stream
        .write_all(
            b"HTTP/1.1 204 No Content\r\n\
              Content-Length: 0\r\n\
              Access-Control-Allow-Origin: *\r\n\
              Access-Control-Allow-Methods: GET, HEAD, OPTIONS\r\n\
              Access-Control-Allow-Headers: Range\r\n\
              Access-Control-Allow-Private-Network: true\r\n\
              Access-Control-Max-Age: 600\r\n\
              Cross-Origin-Resource-Policy: cross-origin\r\n\
              Connection: close\r\n\r\n",
        )
        .await?;
    stream.flush().await
}

async fn respond_bytes(
    stream: &mut TcpStream,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {length}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Private-Network: true\r\n\
         Cross-Origin-Resource-Policy: cross-origin\r\n\
         Connection: close\r\n\r\n",
        length = body.len(),
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await
}

/// 100 ms of mono 8 kHz PCM silence. Loading it through an in-memory
/// `<audio>` element at frontend startup proves that WebKit accepted the
/// configured `media-src`, not merely that `fetch` accepted `connect-src`.
fn probe_wav() -> Vec<u8> {
    const SAMPLE_RATE: u32 = 8_000;
    const SAMPLES: u32 = 800;
    const DATA_BYTES: u32 = SAMPLES * 2;

    let mut wav = Vec::with_capacity((44 + DATA_BYTES) as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + DATA_BYTES).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&DATA_BYTES.to_le_bytes());
    wav.resize((44 + DATA_BYTES) as usize, 0);
    wav
}

/// Write a response, streaming `body` — `(file, offset, length)` — in
/// chunks so a feature-length film never lands in memory at once.
async fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    length: u64,
    extra_headers: Option<String>,
    body: Option<(tokio::fs::File, u64, u64)>,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {length}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Expose-Headers: Accept-Ranges, Content-Length, Content-Range\r\n\
         Access-Control-Allow-Private-Network: true\r\n\
         Cross-Origin-Resource-Policy: cross-origin\r\n\
         Connection: close\r\n\
         {extra}\r\n",
        reason = reason(status),
        extra = extra_headers.unwrap_or_default(),
    );
    stream.write_all(head.as_bytes()).await?;

    if let Some((mut file, offset, mut remaining)) = body {
        use tokio::io::AsyncSeekExt;
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        let mut buffer = vec![0u8; CHUNK_BYTES];
        while remaining > 0 {
            let want = remaining.min(CHUNK_BYTES as u64) as usize;
            let read = file.read(&mut buffer[..want]).await?;
            if read == 0 {
                break;
            }
            stream.write_all(&buffer[..read]).await?;
            remaining -= read as u64;
        }
    }
    stream.flush().await
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        206 => "Partial Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        _ => "Internal Server Error",
    }
}

/// Content type from the extension. The list only needs to cover what
/// this app points a media element at; anything else is left to the
/// player to sniff.
fn mime_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mp4") | Some("m4v") => "video/mp4",
        Some("mov") => "video/quicktime",
        Some("mkv") => "video/x-matroska",
        Some("webm") => "video/webm",
        Some("avi") => "video/x-msvideo",
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("m4a") => "audio/mp4",
        Some("ogg") | Some("opus") => "audio/ogg",
        Some("flac") => "audio/flac",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(target: &str, extra: &str) -> String {
        format!("GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n{extra}\r\n")
    }

    #[test]
    fn parses_a_request_the_way_the_frontend_builds_it() {
        // What `mediaUrl` produces: encodeURIComponent leaves `(`, `)`,
        // `-` and `.` alone but escapes `/` and spaces.
        let request = Request::parse(&head(
            "/media?token=abc123&path=%2FUsers%2Fdc%2FMy%20Film%20(720p).mp4",
            "",
        ))
        .expect("should parse");
        assert_eq!(request.token.as_deref(), Some("abc123"));
        assert_eq!(
            request.path,
            Some(PathBuf::from("/Users/dc/My Film (720p).mp4"))
        );
        assert!(!request.head_only);
        assert_eq!(request.range, None);
    }

    #[test]
    fn picks_up_the_range_header_whatever_its_casing() {
        for line in ["Range: bytes=0-\r\n", "range: bytes=0-\r\n", "RANGE: bytes=0-\r\n"] {
            let request =
                Request::parse(&head("/media?token=t&path=%2Fa.mp4", line)).expect("parses");
            assert_eq!(request.range.as_deref(), Some("bytes=0-"), "for {line:?}");
        }
    }

    #[test]
    fn decodes_non_ascii_paths() {
        let request =
            Request::parse(&head("/media?token=t&path=%2Ftmp%2Fphi%CC%80m.mp4", ""))
                .expect("parses");
        let path = request.path.expect("has a path");
        assert!(path.is_absolute());
        assert!(path.to_string_lossy().ends_with(".mp4"));
    }

    #[test]
    fn accepts_head_requests() {
        let request = Request::parse("HEAD /media?token=t&path=%2Fa.mp4 HTTP/1.1\r\n\r\n")
            .expect("parses");
        assert!(request.head_only);
    }

    #[test]
    fn recognizes_the_token_protected_startup_probe() {
        let request =
            Request::parse("GET /media?token=t&probe=1 HTTP/1.1\r\n\r\n").expect("parses");
        assert_eq!(request.token.as_deref(), Some("t"));
        assert!(request.probe);
        assert_eq!(request.path, None);
    }

    #[test]
    fn rejects_anything_that_is_not_our_route_or_verb() {
        assert_eq!(Request::parse(&head("/other?token=t", "")), None);
        assert_eq!(Request::parse(&head("/", "")), None);
        assert_eq!(
            Request::parse("POST /media?token=t HTTP/1.1\r\n\r\n"),
            None,
            "only GET and HEAD read files"
        );
        assert_eq!(Request::parse(""), None);
    }

    #[test]
    fn a_request_without_a_token_parses_but_carries_none() {
        // `serve` is what rejects it; parsing stays dumb on purpose.
        let request = Request::parse(&head("/media?path=%2Fa.mp4", "")).expect("parses");
        assert_eq!(request.token, None);
    }

    #[test]
    fn parses_the_open_ended_range_media_elements_open_with() {
        assert_eq!(parse_range("bytes=0-", 1000), Some((0, 999)));
        assert_eq!(parse_range("bytes=500-", 1000), Some((500, 999)));
    }

    #[test]
    fn parses_closed_and_suffix_ranges() {
        assert_eq!(parse_range("bytes=10-20", 1000), Some((10, 20)));
        assert_eq!(parse_range("bytes=-100", 1000), Some((900, 999)));
        assert_eq!(
            parse_range("bytes=990-5000", 1000),
            Some((990, 999)),
            "past the end is clamped, not refused"
        );
    }

    #[test]
    fn refuses_ranges_it_cannot_serve() {
        assert_eq!(parse_range("bytes=1000-", 1000), None, "starts past the end");
        assert_eq!(parse_range("bytes=20-10", 1000), None, "inverted");
        assert_eq!(parse_range("bytes=0-10,20-30", 1000), None, "multi-range");
        assert_eq!(parse_range("items=0-10", 1000), None, "wrong unit");
        assert_eq!(parse_range("bytes=abc-", 1000), None, "not a number");
        assert_eq!(parse_range("bytes=0-", 0), None, "empty file");
    }

    #[test]
    fn video_gets_a_video_mime_type() {
        assert_eq!(mime_for(Path::new("/a/b.mp4")), "video/mp4");
        assert_eq!(mime_for(Path::new("/a/b.MP4")), "video/mp4");
        assert_eq!(mime_for(Path::new("/a/b.mkv")), "video/x-matroska");
        assert_eq!(mime_for(Path::new("/a/b.wav")), "audio/wav");
        assert_eq!(mime_for(Path::new("/a/b.bin")), "application/octet-stream");
    }

    #[test]
    fn startup_probe_is_a_well_formed_pcm_wav() {
        let wav = probe_wav();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1, "PCM format");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1, "mono");
        assert_eq!(&wav[36..40], b"data");
        let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize;
        assert_eq!(wav.len(), 44 + data_len);
    }

    /// End to end over a real socket: the bytes a player receives for a
    /// range have to be exactly those bytes, and the token has to matter.
    #[tokio::test]
    async fn serves_ranges_over_a_real_socket() {
        use tokio::io::AsyncWriteExt as _;

        let dir = std::env::temp_dir().join(format!("media-server-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("clip.mp4");
        std::fs::write(&file, b"0123456789ABCDEF").unwrap();

        let base = start().await.expect("server starts");
        let (host, token) = base.split_once("?token=").expect("base carries a token");
        let address = host
            .trim_start_matches("http://")
            .trim_end_matches(ROUTE)
            .to_string();
        let encoded = file.to_string_lossy().replace('/', "%2F").replace(' ', "%20");

        async fn round_trip(address: &str, request: &str) -> String {
            let mut stream = TcpStream::connect(address).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.unwrap();
            String::from_utf8_lossy(&response).into_owned()
        }

        let ranged = round_trip(
            &address,
            &format!(
                "GET {ROUTE}?token={token}&path={encoded} HTTP/1.1\r\n\
                 Host: {address}\r\nRange: bytes=4-7\r\n\r\n"
            ),
        )
        .await;
        assert!(ranged.starts_with("HTTP/1.1 206"), "got: {ranged}");
        assert!(ranged.contains("Content-Range: bytes 4-7/16"), "got: {ranged}");
        assert!(ranged.contains("Content-Length: 4"), "got: {ranged}");
        assert!(ranged.ends_with("4567"), "wrong bytes in: {ranged}");

        let whole = round_trip(
            &address,
            &format!(
                "GET {ROUTE}?token={token}&path={encoded} HTTP/1.1\r\nHost: {address}\r\n\r\n"
            ),
        )
        .await;
        assert!(whole.starts_with("HTTP/1.1 200"), "got: {whole}");
        assert!(whole.contains("Content-Type: video/mp4"), "got: {whole}");
        assert!(whole.contains("Accept-Ranges: bytes"), "got: {whole}");
        assert!(whole.ends_with("0123456789ABCDEF"), "got: {whole}");

        let unsatisfiable = round_trip(
            &address,
            &format!(
                "GET {ROUTE}?token={token}&path={encoded} HTTP/1.1\r\n\
                 Host: {address}\r\nRange: bytes=99-\r\n\r\n"
            ),
        )
        .await;
        assert!(unsatisfiable.starts_with("HTTP/1.1 416"), "got: {unsatisfiable}");
        assert!(unsatisfiable.contains("Content-Range: bytes */16"));

        let no_token = round_trip(
            &address,
            &format!("GET {ROUTE}?path={encoded} HTTP/1.1\r\nHost: {address}\r\n\r\n"),
        )
        .await;
        assert!(no_token.starts_with("HTTP/1.1 403"), "got: {no_token}");

        let missing = round_trip(
            &address,
            &format!(
                "GET {ROUTE}?token={token}&path=%2Fnope%2Fmissing.mp4 HTTP/1.1\r\n\
                 Host: {address}\r\n\r\n"
            ),
        )
        .await;
        assert!(missing.starts_with("HTTP/1.1 404"), "got: {missing}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
