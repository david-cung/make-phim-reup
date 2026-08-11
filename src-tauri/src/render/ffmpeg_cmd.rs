//! Pure FFmpeg command-line builder for Phase 9 final rendering.
//!
//! No process is spawned here — that's the job of
//! [`super::service::RenderService`]. Keeping this pure makes the argv
//! trivially unit-testable and the spawner tiny.
//!
//! Overview of the shapes we produce:
//!
//! * **External** subtitles — video is `-c:v copy`, audio is
//!   re-encoded from the mixed WAV.
//! * **Burned** subtitles — video is re-encoded through the
//!   `subtitles=<path>` filter, audio is re-encoded from the mixed
//!   WAV.
//! * **None** — same as external but without the sidecar.
//!
//! In every case we `-map 0:v:0` (first video stream from the source)
//! and `-map 1:a:0` (mixed WAV) — the source's original soundtrack is
//! intentionally dropped so the final file contains exactly one
//! Vietnamese-dubbed audio track.

use std::path::Path;

use super::models::{OutputFormat, RenderSettings, SubtitleMode, VideoCodec};

/// A single FFmpeg process description ready to hand to
/// [`tokio::process::Command`]. Held as `Vec<String>` because we log
/// the exact argv and want it to be trivially comparable in tests.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderCommand {
    pub args: Vec<String>,
    /// Absolute path we asked FFmpeg to write. Same as the last argv
    /// element, but exposed separately so the caller doesn't have to
    /// poke into `args`.
    pub output: String,
}

/// Build the argv for a render run. Callers hand this straight to
/// FFmpeg.
///
/// * `source_video` — the imported movie (video + audio container).
/// * `mixed_audio`  — the Phase 8 output WAV.
/// * `burn_subtitle` — path to an SRT/ASS file that should be burned
///   into the video (only used when settings.subtitle_mode is
///   `Burned`).
/// * `output`       — absolute path of the file to write.
pub fn build_render_command(
    source_video: &Path,
    mixed_audio: &Path,
    burn_subtitle: Option<&Path>,
    settings: &RenderSettings,
    output: &Path,
) -> RenderCommand {
    let s = settings.clone().normalised();

    let mut args: Vec<String> = vec![
        "-y".into(),
        "-nostdin".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-progress".into(),
        "pipe:1".into(),
        // Input 0: original video (we keep its video track only).
        "-i".into(),
        source_video.to_string_lossy().into_owned(),
        // Input 1: mixed Vietnamese audio.
        "-i".into(),
        mixed_audio.to_string_lossy().into_owned(),
    ];

    // Map video from the source, audio from the mixed WAV. We drop
    // the source's audio deliberately so we ship exactly one dubbed
    // track and don't leak the original English/Chinese/etc.
    args.push("-map".into());
    args.push("0:v:0".into());
    args.push("-map".into());
    args.push("1:a:0".into());
    // Never pull data / attachment streams into the output — subtitle
    // decisions are handled explicitly below.
    args.push("-map_metadata".into());
    args.push("0".into());
    args.push("-map_chapters".into());
    args.push("0".into());

    // Video: burned subtitles require re-encoding through the
    // `subtitles` filter; otherwise obey the setting.
    match (&s.subtitle_mode, &s.video_codec) {
        (SubtitleMode::Burned, VideoCodec::Reencode { codec }) => {
            args.push("-vf".into());
            args.push(build_subtitles_filter(
                burn_subtitle.unwrap_or(Path::new("")),
            ));
            args.push("-c:v".into());
            args.push(codec.clone());
            push_video_reencode_defaults(&mut args, codec);
        }
        (_, VideoCodec::Copy) => {
            args.push("-c:v".into());
            args.push("copy".into());
        }
        (_, VideoCodec::Reencode { codec }) => {
            args.push("-c:v".into());
            args.push(codec.clone());
            push_video_reencode_defaults(&mut args, codec);
        }
    }

    // Audio: always re-encode from the mixed WAV.
    args.push("-c:a".into());
    args.push(s.audio_codec.codec.clone());
    if let Some(br) = &s.audio_codec.bitrate {
        args.push("-b:a".into());
        args.push(br.clone());
    }

    // MP4 wants faststart so the moov atom lands at the front — no
    // effect on other containers.
    if matches!(s.output_format, OutputFormat::Mp4) {
        args.push("-movflags".into());
        args.push("+faststart".into());
    }

    // Length: match the shorter of the two inputs. The source video
    // is authoritative, the mix should equal its duration ± a few
    // samples, so `-shortest` guarantees we never trail past the
    // video.
    args.push("-shortest".into());

    args.push(output.to_string_lossy().into_owned());

    RenderCommand {
        args,
        output: output.to_string_lossy().into_owned(),
    }
}

/// Escape a filesystem path for FFmpeg's `subtitles=` filter argument.
///
/// FFmpeg parses `filtergraph` strings with a *nested* grammar: colons
/// separate filter options, single quotes and backslashes are special
/// inside filter values. We wrap the path in single quotes and
/// backslash-escape any single quote or backslash inside it. Colons
/// are additionally escaped because they end the option value
/// otherwise.
pub fn build_subtitles_filter(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let mut escaped = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\'' => escaped.push_str("\\'"),
            ':' => escaped.push_str("\\:"),
            // Comma is a filtergraph separator, but the `subtitles=`
            // filter treats its value opaquely up to the next option
            // key. Belt-and-braces: escape commas too.
            ',' => escaped.push_str("\\,"),
            other => escaped.push(other),
        }
    }
    format!("subtitles='{escaped}'")
}

fn push_video_reencode_defaults(args: &mut Vec<String>, codec: &str) {
    // Sensible defaults for the small set of codecs the UI advertises.
    // Users who need finer control can drop into a custom pipeline
    // later — the spec explicitly asks us to hide low-level knobs.
    match codec {
        "libx264" | "libx265" => {
            args.push("-preset".into());
            args.push("medium".into());
            args.push("-crf".into());
            args.push("20".into());
            args.push("-pix_fmt".into());
            args.push("yuv420p".into());
        }
        "libvpx-vp9" => {
            args.push("-b:v".into());
            args.push("0".into());
            args.push("-crf".into());
            args.push("32".into());
        }
        _ => {}
    }
}
