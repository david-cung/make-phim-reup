//! `ffprobe -print_format json` wrapper → typed metadata.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use super::errors::FfmpegError;

/// The metadata surface we expose to the frontend.
///
/// Fields are `Option<T>` when FFprobe is allowed to legitimately omit
/// them (e.g. WebM streams without a listed bitrate), and required when
/// they are guaranteed to exist for any file we accept.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VideoMetadata {
    pub duration_secs: f64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub audio_channels: Option<u16>,
    pub audio_sample_rate: Option<u32>,
    pub format: Option<String>,
    pub file_size: u64,
    pub bit_rate: Option<u64>,
    pub audio_stream_count: u32,
    pub subtitle_stream_count: u32,
    pub video_stream_count: u32,
    pub streams: Vec<StreamSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StreamSummary {
    pub index: u32,
    pub kind: String, // "video" | "audio" | "subtitle" | "data" | "attachment"
    pub codec: Option<String>,
    pub language: Option<String>,
    pub channels: Option<u16>,
    pub sample_rate: Option<u32>,
}

// ---------- raw ffprobe JSON shapes ----------

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    #[serde(default)]
    index: u32,
    codec_type: Option<String>,
    codec_name: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    channels: Option<u16>,
    sample_rate: Option<String>,
    #[serde(default)]
    tags: Option<StreamTags>,
}

#[derive(Debug, Deserialize)]
struct StreamTags {
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    format_name: Option<String>,
    duration: Option<String>,
    size: Option<String>,
    bit_rate: Option<String>,
}

// ---------- runner ----------

pub async fn probe_video(ffprobe_bin: &Path, input: &Path) -> Result<VideoMetadata, FfmpegError> {
    if !input.exists() {
        return Err(FfmpegError::InputMissing {
            path: input.to_path_buf(),
        });
    }
    let out = Command::new(ffprobe_bin)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(input)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| FfmpegError::io("running ffprobe", source))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(FfmpegError::ProbeFailed {
            code: out.status.code().unwrap_or(-1),
            stderr,
        });
    }

    parse_ffprobe_json(&out.stdout, input)
}

pub(crate) fn parse_ffprobe_json(json: &[u8], input: &Path) -> Result<VideoMetadata, FfmpegError> {
    let raw: ProbeOutput = serde_json::from_slice(json).map_err(|e| FfmpegError::ProbeParse {
        details: e.to_string(),
    })?;

    let mut video_streams = 0u32;
    let mut audio_streams = 0u32;
    let mut subtitle_streams = 0u32;
    let mut streams: Vec<StreamSummary> = Vec::with_capacity(raw.streams.len());

    let mut video_codec = None;
    let mut audio_codec = None;
    let mut audio_channels = None;
    let mut audio_sample_rate = None;
    let mut width = None;
    let mut height = None;
    let mut fps = None;

    for s in &raw.streams {
        let kind = s.codec_type.clone().unwrap_or_else(|| "unknown".into());
        match kind.as_str() {
            "video" => {
                video_streams += 1;
                if video_codec.is_none() {
                    video_codec = s.codec_name.clone();
                    width = s.width;
                    height = s.height;
                    fps = parse_fps(&s.r_frame_rate, &s.avg_frame_rate);
                }
            }
            "audio" => {
                audio_streams += 1;
                if audio_codec.is_none() {
                    audio_codec = s.codec_name.clone();
                    audio_channels = s.channels;
                    audio_sample_rate =
                        s.sample_rate.as_deref().and_then(|v| v.parse::<u32>().ok());
                }
            }
            "subtitle" => subtitle_streams += 1,
            _ => {}
        }

        streams.push(StreamSummary {
            index: s.index,
            kind,
            codec: s.codec_name.clone(),
            language: s.tags.as_ref().and_then(|t| t.language.clone()),
            channels: s.channels,
            sample_rate: s.sample_rate.as_deref().and_then(|v| v.parse::<u32>().ok()),
        });
    }

    let format = raw.format.as_ref().and_then(|f| f.format_name.clone());
    let duration = raw
        .format
        .as_ref()
        .and_then(|f| f.duration.as_deref())
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0);
    let file_size = raw
        .format
        .as_ref()
        .and_then(|f| f.size.as_deref())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| std::fs::metadata(input).ok().map(|m| m.len()))
        .unwrap_or(0);
    let bit_rate = raw
        .format
        .as_ref()
        .and_then(|f| f.bit_rate.as_deref())
        .and_then(|s| s.parse::<u64>().ok());

    Ok(VideoMetadata {
        duration_secs: duration,
        width,
        height,
        fps,
        video_codec,
        audio_codec,
        audio_channels,
        audio_sample_rate,
        format,
        file_size,
        bit_rate,
        audio_stream_count: audio_streams,
        subtitle_stream_count: subtitle_streams,
        video_stream_count: video_streams,
        streams,
    })
}

fn parse_fps(primary: &Option<String>, fallback: &Option<String>) -> Option<f32> {
    parse_rational(primary.as_deref()).or_else(|| parse_rational(fallback.as_deref()))
}

fn parse_rational(v: Option<&str>) -> Option<f32> {
    let v = v?;
    let (n, d) = v.split_once('/')?;
    let n: f32 = n.parse().ok()?;
    let d: f32 = d.parse().ok()?;
    if d == 0.0 {
        return None;
    }
    let ratio = n / d;
    if ratio.is_finite() && ratio > 0.0 {
        Some((ratio * 1000.0).round() / 1000.0)
    } else {
        None
    }
}

/// Convenience for callers: reject clearly-unusable inputs early. Used
/// before starting an audio extraction so we don't spawn ffmpeg only to
/// have it fail with an opaque error.
pub fn require_audio_stream(meta: &VideoMetadata) -> Result<(), FfmpegError> {
    if meta.audio_stream_count == 0 {
        return Err(FfmpegError::NoAudioStream);
    }
    Ok(())
}

pub fn require_video_stream(meta: &VideoMetadata) -> Result<(), FfmpegError> {
    if meta.video_stream_count == 0 {
        return Err(FfmpegError::NoVideoStream);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MP4: &str = r#"{
        "streams": [
            {"index":0,"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"r_frame_rate":"24000/1001"},
            {"index":1,"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2,"tags":{"language":"eng"}},
            {"index":2,"codec_type":"subtitle","codec_name":"mov_text","tags":{"language":"eng"}}
        ],
        "format": {
            "format_name":"mov,mp4,m4a,3gp,3g2,mj2",
            "duration":"3600.123",
            "size":"12345678",
            "bit_rate":"27000"
        }
    }"#;

    #[test]
    fn parses_metadata() {
        let m = parse_ffprobe_json(SAMPLE_MP4.as_bytes(), std::path::Path::new("/tmp/x")).unwrap();
        assert_eq!(m.width, Some(1920));
        assert_eq!(m.height, Some(1080));
        assert_eq!(m.fps, Some(23.976));
        assert_eq!(m.video_codec.as_deref(), Some("h264"));
        assert_eq!(m.audio_codec.as_deref(), Some("aac"));
        assert_eq!(m.audio_channels, Some(2));
        assert_eq!(m.audio_sample_rate, Some(48_000));
        assert_eq!(m.file_size, 12_345_678);
        assert_eq!(m.audio_stream_count, 1);
        assert_eq!(m.subtitle_stream_count, 1);
        assert_eq!(m.video_stream_count, 1);
        assert!((m.duration_secs - 3600.123).abs() < 0.001);
    }

    #[test]
    fn missing_audio_stream_detected() {
        let json = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"h264"}],"format":{"duration":"5"}}"#;
        let m = parse_ffprobe_json(json.as_bytes(), std::path::Path::new("/tmp/x")).unwrap();
        assert_eq!(m.audio_stream_count, 0);
        assert!(matches!(
            require_audio_stream(&m),
            Err(FfmpegError::NoAudioStream)
        ));
    }

    #[test]
    fn missing_video_stream_detected() {
        let json = r#"{"streams":[{"index":0,"codec_type":"audio","codec_name":"aac","channels":2,"sample_rate":"48000"}],"format":{"duration":"5"}}"#;
        let m = parse_ffprobe_json(json.as_bytes(), std::path::Path::new("/tmp/x")).unwrap();
        assert_eq!(m.video_stream_count, 0);
        assert!(matches!(
            require_video_stream(&m),
            Err(FfmpegError::NoVideoStream)
        ));
    }

    #[test]
    fn malformed_json_is_probe_parse_error() {
        let err = parse_ffprobe_json(b"not-json", std::path::Path::new("/tmp/x")).unwrap_err();
        assert!(matches!(err, FfmpegError::ProbeParse { .. }));
    }

    #[test]
    fn division_by_zero_fps() {
        assert_eq!(parse_rational(Some("0/0")), None);
    }

    #[test]
    fn falls_back_to_avg_frame_rate() {
        let json = r#"{
            "streams":[{"index":0,"codec_type":"video","codec_name":"h264","r_frame_rate":"0/0","avg_frame_rate":"25/1"}],
            "format":{"duration":"5","size":"1000"}
        }"#;
        let m = parse_ffprobe_json(json.as_bytes(), std::path::Path::new("/tmp/x")).unwrap();
        assert_eq!(m.fps, Some(25.0));
    }
}
