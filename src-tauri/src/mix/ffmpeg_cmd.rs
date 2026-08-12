//! Pure FFmpeg command-line builder for Phase 8 mixing.
//!
//! The whole graph is a single `-filter_complex` chain, so we build it
//! carefully and unit-test the exact string. No process is spawned in
//! this module — that's the job of [`super::service::MixService`].
//!
//! Filter-graph shape (with ducking on and N voice inputs)::
//!
//! ```text
//!   [0:a]aformat=…                                     -> [orig]
//!   [1:a]adelay=t1|t1,aformat=…                        -> [v1]
//!   [2:a]adelay=t2|t2,aformat=…                        -> [v2]
//!    …
//!   [v1][v2]…amix=inputs=N:normalize=0                 -> [voice]
//!   [voice]volume=voice_vol                            -> [voice_g]
//!   [orig]volume=orig_vol                              -> [orig_g]
//!   [orig_g][voice_g]sidechaincompress=threshold=…     -> [orig_ducked]
//!   [orig_ducked][voice_g]amix=inputs=2:normalize=0    -> [mix_sum]
//!   [mix_sum]alimiter=limit=…:level=disabled           -> [mix]
//! ```
//!
//! With ducking off, the `sidechaincompress` node is skipped and the
//! two gain-stages feed straight into the final `amix`.

use std::path::Path;

use super::models::{MixSettings, MixVoiceInput};

/// Peak ceiling for the output, linear. ≈ −1 dBFS, which leaves the AAC
/// encoder in the render step a little room to overshoot without
/// clipping.
const LIMITER_CEILING: f32 = 0.891;

/// A single FFmpeg process description ready to hand to
/// [`tokio::process::Command`]. Held as `Vec<String>` because we log
/// the exact argv and want it to be trivially comparable in tests.
#[derive(Debug, Clone, PartialEq)]
pub struct MixCommand {
    pub args: Vec<String>,
    /// The absolute path we asked FFmpeg to write. Same as the last
    /// argv element, but exposed separately so the caller doesn't have
    /// to poke into `args`.
    pub output: String,
}

/// Build the argv for a mix run. Callers hand this straight to FFmpeg.
///
/// * `source_video` — the imported movie (video + audio container).
/// * `voice_segments` — every non-empty synced voice WAV, in any order.
/// * `output` — absolute path of the WAV to write.
///
/// Callers must have already filtered out `SyncStatus::Empty` entries
/// (they add nothing to the mix and just cost us extra `adelay`
/// filters). See [`MixVoiceInput::is_empty`].
pub fn build_mix_command(
    source_video: &Path,
    voice_segments: &[MixVoiceInput],
    settings: &MixSettings,
    output: &Path,
) -> MixCommand {
    let s = settings.normalised();

    let mut args: Vec<String> = vec![
        "-y".into(),
        "-nostdin".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-progress".into(),
        "pipe:1".into(),
        "-i".into(),
        source_video.to_string_lossy().into_owned(),
    ];

    // Voice inputs. Filter out empties defensively; caller should have
    // already done this but skipping silence is always safe.
    let voices: Vec<&MixVoiceInput> = voice_segments.iter().filter(|v| !v.is_empty).collect();
    for v in &voices {
        args.push("-i".into());
        args.push(v.relative_file.clone());
    }

    let filter = build_filter_graph(&voices, &s);
    args.push("-filter_complex".into());
    args.push(filter);
    args.push("-map".into());
    args.push("[mix]".into());
    args.push("-vn".into());
    args.push("-sn".into());
    args.push("-dn".into());
    // Match the source video length. The voice track only has content
    // where subtitles have voice — `amix` extends silence otherwise —
    // so this ensures we don't accidentally cut the film short.
    args.push("-shortest".into());

    if let Some(sr) = s.output_sample_rate {
        args.push("-ar".into());
        args.push(sr.to_string());
    }
    args.push("-ac".into());
    args.push(s.output_channels.to_string());
    args.push("-c:a".into());
    args.push("pcm_s16le".into());
    args.push(output.to_string_lossy().into_owned());

    MixCommand {
        args,
        output: output.to_string_lossy().into_owned(),
    }
}

/// Compose the `-filter_complex` argument only. Broken out so the
/// tests can assert on the graph shape without also matching the whole
/// argv.
pub fn build_filter_graph(voices: &[&MixVoiceInput], settings: &MixSettings) -> String {
    let s = settings.normalised();
    let ch = s.output_channels.max(1);
    let mut chain: Vec<String> = Vec::with_capacity(voices.len() + 8);

    // Original track normalised to output shape.
    chain.push(format!(
        "[0:a:0]aformat=sample_fmts=fltp:channel_layouts={ch}[orig]",
        ch = channel_layout(ch),
    ));

    // Every voice input gets an adelay to reach its subtitle start.
    for (idx, v) in voices.iter().enumerate() {
        let input_idx = idx + 1;
        let delay_ms = (v.target_start_secs.max(0.0) * 1000.0).round() as i64;
        // For stereo output we still delay just 2 channels (adelay expects
        // one value per channel of the OUTPUT stream). apad the tail so the
        // filter graph doesn't shrink the mix window to the voice length.
        chain.push(format!(
            "[{input_idx}:a:0]adelay=delays={delay}:all=1,aformat=sample_fmts=fltp:channel_layouts={layout}[v{i}]",
            input_idx = input_idx,
            delay = delay_ms,
            layout = channel_layout(ch),
            i = input_idx,
        ));
    }

    // Combine every voice into a single stream. When N=0 we don't mix
    // any voices — the mix is just the original at its configured
    // volume (allowing "mixing off" while still exercising the pipeline).
    let voice_label = if voices.is_empty() {
        None
    } else if voices.len() == 1 {
        Some("v1".to_string())
    } else {
        let inputs: String = (1..=voices.len()).map(|i| format!("[v{i}]")).collect();
        chain.push(format!(
            "{inputs}amix=inputs={n}:normalize=0:duration=longest[voice_raw]",
            inputs = inputs,
            n = voices.len(),
        ));
        Some("voice_raw".to_string())
    };

    // Apply gain stages.
    chain.push(format!(
        "[orig]volume={vol:.4}[orig_g]",
        vol = s.original_volume
    ));
    if let Some(label) = &voice_label {
        chain.push(format!(
            "[{label}]volume={vol:.4}[voice_g]",
            label = label,
            vol = s.voice_volume
        ));
    }

    // Ducking (only meaningful when we actually have voice content).
    let orig_final = if s.ducking_enabled && voice_label.is_some() {
        let ratio = duck_ratio_from_depth_db(s.ducking_depth_db);
        // `makeup` is a linear post-gain with an FFmpeg range of [1, 64];
        // 1 is unity. Ducking exists to push the original *down* while
        // the voice speaks, so we never want to add gain back — and 0 is
        // rejected outright ("Value 0.000000 for parameter 'makeup' out
        // of range").
        chain.push(format!(
            "[orig_g][voice_g]sidechaincompress=threshold={thresh:.4}:ratio={ratio:.4}:attack={attack:.2}:release={release:.2}:makeup=1[orig_ducked]",
            thresh = db_to_linear(s.ducking_threshold_db),
            ratio = ratio,
            attack = s.ducking_attack_ms,
            release = s.ducking_release_ms,
        ));
        "orig_ducked".to_string()
    } else {
        "orig_g".to_string()
    };

    // Final mix. If we have no voice content, just alias the original
    // through so the limiter below still has something to read.
    match voice_label {
        Some(_) => chain.push(format!(
            "[{orig}][voice_g]amix=inputs=2:normalize=0:duration=longest[mix_sum]",
            orig = orig_final
        )),
        None => chain.push(format!("[{orig}]anull[mix_sum]", orig = orig_final)),
    }

    // `normalize=0` makes `amix` a straight sum, so a voice line landing
    // on a loud passage pushes the result past full scale and the samples
    // clip. Catch those peaks instead of letting them square off, keeping
    // ~1 dB spare for the lossy encode the render step adds on top.
    //
    // `level=disabled` matters: the limiter's auto-level would pull quiet
    // stretches up to the ceiling, undoing the ducking we just applied.
    chain.push(format!(
        "[mix_sum]alimiter=limit={ceiling:.3}:level=disabled[mix]",
        ceiling = LIMITER_CEILING,
    ));

    chain.join(";")
}

/// FFmpeg's `channel_layouts` filter argument for a given channel count.
fn channel_layout(channels: u32) -> &'static str {
    match channels {
        1 => "mono",
        _ => "stereo",
    }
}

/// Convert a dB level to a linear amplitude ratio (10^(dB/20)).
fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Translate the user-facing "ducking depth" into a `sidechaincompress`
/// ratio. Depth `D` (in dB) tells us how much *deeper* the original
/// should sit when the voice is present — larger depth ⇒ larger ratio.
/// We keep the mapping simple and monotonic: depth 0 dB ⇒ ratio 1
/// (no compression), depth 30 dB ⇒ ratio 20 (heavy pump).
///
/// Named for `ratio` deliberately: this feeds `sidechaincompress`'s
/// `ratio`, not its `makeup`, and conflating the two is what produced an
/// invalid `makeup=0` in the graph.
pub fn duck_ratio_from_depth_db(depth_db: f32) -> f32 {
    let d = depth_db.max(0.0);
    // Empirical map — clamped to FFmpeg's ratio range [1, 20].
    (1.0 + d * (19.0 / 30.0)).clamp(1.0, 20.0)
}
