//! Phase 6 automatic dubbing preparation.
//!
//! This module is deliberately pure: given the current subtitle doc,
//! installed voices and user/default TTS knobs, it produces a
//! movie-scoped voice map plus per-segment dubbing text/voice choices.
//! The service layer persists the updated subtitle doc and manifest.

use std::collections::{BTreeMap, BTreeSet};

use crate::subtitles::{SubtitleDoc, SubtitleSegment};

use super::models::{TtsSettings, VoiceInfo, VoiceProfile};

const SAFE_MAX_TTS_SPEED: f32 = 1.12;
const TARGET_CHARS_PER_SECOND: f64 = 13.5;

#[derive(Debug, Clone)]
pub struct AutomaticDubbingPlan {
    pub doc: SubtitleDoc,
    pub voice_profiles: Vec<VoiceProfile>,
    pub changed: bool,
}

pub fn prepare_automatic_dubbing(
    doc: &SubtitleDoc,
    voices: &[VoiceInfo],
    engine: &str,
    fallback_voice_id: &str,
) -> AutomaticDubbingPlan {
    let profiles = assign_voice_profiles(doc, voices, engine, fallback_voice_id);
    let voice_by_speaker = profiles
        .iter()
        .filter_map(|profile| {
            profile
                .speaker_id
                .as_ref()
                .map(|speaker| (speaker.clone(), profile.voice_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let fallback = resolve_fallback_voice_id(voices, engine, fallback_voice_id);
    let mut next = doc.clone();
    let mut changed = false;

    for seg in &mut next.segments {
        let dubbing = dubbing_script_for_segment(seg);
        if seg.dubbing_text.trim() != dubbing {
            seg.dubbing_text = dubbing;
            changed = true;
        }

        let desired_voice = seg
            .speaker
            .as_ref()
            .and_then(|speaker| voice_by_speaker.get(speaker))
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        if !desired_voice.trim().is_empty()
            && seg.voice_id.as_deref() != Some(desired_voice.as_str())
        {
            seg.voice_id = Some(desired_voice);
            changed = true;
        }
    }

    if changed {
        next.dirty.mark_content_dirty();
        next.touch();
    }

    AutomaticDubbingPlan {
        doc: next,
        voice_profiles: profiles,
        changed,
    }
}

pub fn assign_voice_profiles(
    doc: &SubtitleDoc,
    voices: &[VoiceInfo],
    engine: &str,
    fallback_voice_id: &str,
) -> Vec<VoiceProfile> {
    let fallback = resolve_fallback_voice_id(voices, engine, fallback_voice_id);
    let mut speakers = doc
        .segments
        .iter()
        .filter_map(|seg| seg.speaker.as_ref())
        .filter(|speaker| !speaker.trim().is_empty())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    speakers.sort();
    if speakers.is_empty() && !doc.segments.is_empty() {
        speakers.push("speaker_unknown".into());
    }

    let engine_voices = voices
        .iter()
        .filter(|voice| voice.installed && voice.engine == engine)
        .collect::<Vec<_>>();
    let mut gender_buckets = BTreeMap::<String, Vec<&VoiceInfo>>::new();
    for voice in &engine_voices {
        gender_buckets
            .entry(voice.gender.to_ascii_lowercase())
            .or_default()
            .push(*voice);
    }

    speakers
        .iter()
        .enumerate()
        .map(|(idx, speaker)| {
            let preferred_gender = if idx % 2 == 0 { "male" } else { "female" };
            let chosen = gender_buckets
                .get(preferred_gender)
                .and_then(|items| items.get(idx / 2 % items.len()))
                .or_else(|| gender_buckets.get("neutral").and_then(|items| items.first()))
                .or_else(|| engine_voices.get(idx % engine_voices.len().max(1)))
                .map(|voice| voice.id.clone())
                .unwrap_or_else(|| fallback.clone());
            VoiceProfile {
                character_id: speaker_to_character_id(speaker),
                speaker_id: Some(speaker.clone()),
                voice_id: chosen,
                style: "natural".into(),
                speed: 1.0,
                confidence: if fallback.trim().is_empty() { 0.35 } else { 0.82 },
            }
        })
        .collect()
}

pub fn dubbing_script_for_segment(seg: &SubtitleSegment) -> String {
    let base = if !seg.dubbing_text.trim().is_empty() {
        seg.dubbing_text.trim()
    } else if !seg.translated_text.trim().is_empty() {
        seg.translated_text.trim()
    } else {
        seg.source_text.trim()
    };
    let mut text = normalise_spoken_text(base);
    let duration = (seg.end - seg.start).max(0.1);
    let budget = (duration * TARGET_CHARS_PER_SECOND).round() as usize;
    if budget >= 12 && text.chars().count() > budget {
        text = shorten_spoken_text(&text, budget);
    }
    text
}

pub fn effective_settings_for_segment(
    seg: &SubtitleSegment,
    base: &TtsSettings,
) -> TtsSettings {
    let mut settings = base.normalised();
    let text = pick_spoken_text(seg);
    let duration = (seg.end - seg.start).max(0.1);
    let estimated = estimate_spoken_duration_secs(&text);
    if estimated > duration * 1.08 {
        let required = (estimated / duration) as f32;
        settings.speed = settings.speed.max(required.min(SAFE_MAX_TTS_SPEED));
    }
    settings.normalised()
}

pub fn estimate_spoken_duration_secs(text: &str) -> f64 {
    let chars = text.chars().filter(|ch| !ch.is_whitespace()).count() as f64;
    let punctuation_pause = text
        .chars()
        .filter(|ch| matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | '…'))
        .count() as f64
        * 0.08;
    (chars / TARGET_CHARS_PER_SECOND + punctuation_pause).max(0.15)
}

pub fn resolve_fallback_voice_id(
    voices: &[VoiceInfo],
    engine: &str,
    requested: &str,
) -> String {
    if voices
        .iter()
        .any(|voice| voice.installed && voice.engine == engine && voice.id == requested)
    {
        return requested.to_string();
    }
    voices
        .iter()
        .find(|voice| voice.installed && voice.engine == engine && voice.gender == "neutral")
        .or_else(|| voices.iter().find(|voice| voice.installed && voice.engine == engine))
        .map(|voice| voice.id.clone())
        .unwrap_or_else(|| requested.to_string())
}

fn pick_spoken_text(seg: &SubtitleSegment) -> String {
    if !seg.dubbing_text.trim().is_empty() {
        seg.dubbing_text.trim().to_string()
    } else if !seg.translated_text.trim().is_empty() {
        seg.translated_text.trim().to_string()
    } else {
        seg.source_text.trim().to_string()
    }
}

fn normalise_spoken_text(text: &str) -> String {
    let mut out = text
        .replace('\n', " ")
        .replace("...", "…")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    while out.contains(" ,") {
        out = out.replace(" ,", ",");
    }
    while out.contains(" .") {
        out = out.replace(" .", ".");
    }
    out.trim().to_string()
}

fn shorten_spoken_text(text: &str, budget: usize) -> String {
    let mut candidate = text.to_string();
    for filler in [
        "thật sự ",
        "cơ mà ",
        "thì ",
        "mà ",
        "nhé",
        "nhỉ",
        "ạ",
    ] {
        candidate = candidate.replace(filler, "");
    }
    candidate = normalise_spoken_text(&candidate);
    if candidate.chars().count() <= budget {
        return candidate;
    }
    let mut out = String::new();
    for word in candidate.split_whitespace() {
        let next_len = out.chars().count() + word.chars().count() + usize::from(!out.is_empty());
        if next_len > budget {
            break;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    if out.trim().is_empty() {
        candidate
    } else {
        out.trim_matches(|ch: char| ch == ',' || ch == ';' || ch == ':')
            .trim()
            .to_string()
    }
}

fn speaker_to_character_id(speaker: &str) -> String {
    let suffix = speaker
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_ascii_lowercase();
    if suffix.is_empty() {
        "character_unknown".into()
    } else {
        format!("character_{suffix}")
    }
}
