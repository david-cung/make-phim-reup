//! Minimal SubRip (`*.srt`) parser and writer.
//!
//! Only the essentials — index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`,
//! multi-line text, blank-line separated blocks. HTML-ish inline tags
//! like `<i>` / `<b>` / `<font ...>` are stripped on import (Phase 5
//! is a subtitle editor, not a subtitle formatter). Coordinates,
//! X-timings and other niche extensions are ignored.

use super::models::SubtitleSegment;

const HEADER_BOM: char = '\u{feff}';

/// Parse an SRT string into `(index, start, end, text)` tuples, then
/// wrap each row in a `SubtitleSegment`. `start_id` is used for the
/// first segment; subsequent ones increment sequentially.
///
/// The `index` field on disk is preserved when it parses cleanly as
/// a u32, otherwise the loop's own counter is used.
pub fn parse(input: &str, start_id: u32) -> Result<Vec<SubtitleSegment>, String> {
    let cleaned = input.trim_start_matches(HEADER_BOM);
    // Normalise line endings so windows-authored files parse.
    let normalised = cleaned.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = Vec::new();
    let mut next_id = start_id;

    for (block_ix, raw_block) in normalised.split("\n\n").enumerate() {
        let block = raw_block.trim_matches('\n');
        if block.trim().is_empty() {
            continue;
        }
        let mut lines = block.lines();
        let first = lines.next().unwrap_or("").trim();
        // Some SRTs omit the numeric index — in that case, `first` is
        // already the timecode line.
        let (timeline, has_index) = if first.contains("-->") {
            (first, false)
        } else {
            (lines.next().unwrap_or("").trim(), true)
        };
        if !timeline.contains("-->") {
            return Err(format!(
                "block {} does not contain a timecode arrow",
                block_ix + 1
            ));
        }
        let (start, end) = parse_time_range(timeline).map_err(|e| {
            format!(
                "block {}: invalid timecode line `{}`: {}",
                block_ix + 1,
                timeline,
                e
            )
        })?;
        let text = lines.collect::<Vec<_>>().join("\n");
        let cleaned_text = strip_inline_tags(&text);

        let id = if has_index {
            first.parse::<u32>().unwrap_or_else(|_| {
                let v = next_id;
                next_id += 1;
                v
            })
        } else {
            let v = next_id;
            next_id += 1;
            v
        };
        // Keep the running counter ahead of anything we saw explicitly.
        next_id = next_id.max(id.saturating_add(1));

        out.push(SubtitleSegment {
            id,
            start,
            end,
            source_text: String::new(),
            translated_text: cleaned_text,
            dubbing_text: String::new(),
            words: None,
            speaker: None,
            voice_id: None,
        });
    }
    Ok(out)
}

/// Convert a slice of segments into an SRT string. `text_for` picks
/// which field to render (translated / source / bilingual — the
/// caller decides).
pub fn write<'a, F>(segments: &'a [SubtitleSegment], text_for: F) -> String
where
    F: Fn(&'a SubtitleSegment) -> String,
{
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        // SRT expects sequential 1-based indices, not the internal id.
        let idx = i + 1;
        out.push_str(&format!("{idx}\n"));
        out.push_str(&format!(
            "{} --> {}\n",
            format_time_srt(seg.start.max(0.0)),
            format_time_srt(seg.end.max(seg.start + 0.001))
        ));
        let text = text_for(seg);
        if text.is_empty() {
            out.push('\n');
        } else {
            out.push_str(text.trim_end_matches('\n'));
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

fn parse_time_range(line: &str) -> Result<(f64, f64), String> {
    let (l, r) = line
        .split_once("-->")
        .ok_or_else(|| "missing `-->` separator".to_string())?;
    let start = parse_time_srt(l.trim())?;
    // Anything after the end time (e.g. `X1:... X2:...` position
    // annotations) is ignored — safely.
    let end_str = r.split_whitespace().next().unwrap_or("");
    let end = parse_time_srt(end_str)?;
    Ok((start, end))
}

fn parse_time_srt(s: &str) -> Result<f64, String> {
    // Accepts `HH:MM:SS,mmm` (SRT canonical), `HH:MM:SS.mmm`
    // (occasional variant), or `MM:SS,mmm`.
    let (h, m, sec) = split_hms(s)?;
    Ok(h as f64 * 3600.0 + m as f64 * 60.0 + sec)
}

fn split_hms(s: &str) -> Result<(u32, u32, f64), String> {
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, tail) = match parts.as_slice() {
        [h, m, t] => (
            h.parse::<u32>().map_err(|_| format!("bad hours: {h}"))?,
            m.parse::<u32>().map_err(|_| format!("bad minutes: {m}"))?,
            *t,
        ),
        [m, t] => (
            0u32,
            m.parse::<u32>().map_err(|_| format!("bad minutes: {m}"))?,
            *t,
        ),
        _ => return Err(format!("expected H:M:S.ms or M:S.ms, got `{s}`")),
    };
    let sec_str = tail.replace(',', ".");
    let sec = sec_str
        .parse::<f64>()
        .map_err(|_| format!("bad seconds: {tail}"))?;
    Ok((h, m, sec))
}

pub fn format_time_srt(t: f64) -> String {
    let total_ms = (t.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn strip_inline_tags(input: &str) -> String {
    // Cheap tag stripper — walks the string once, dropping `<...>`
    // spans. Covers `<i>`, `<b>`, `<u>`, `<font ...>` and their
    // closers, which is 99% of what real subtitles use.
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_srt() {
        let input = "1\n00:00:00,500 --> 00:00:02,000\nHello\n\n2\n00:00:02,100 --> 00:00:03,500\n<i>World</i>\n";
        let segs = parse(input, 0).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].id, 1);
        assert!((segs[0].start - 0.5).abs() < 1e-6);
        assert!((segs[0].end - 2.0).abs() < 1e-6);
        assert_eq!(segs[0].translated_text, "Hello");
        assert_eq!(segs[1].translated_text, "World");
    }

    #[test]
    fn tolerates_missing_index() {
        let input = "00:00:00,000 --> 00:00:01,000\nA\n\n00:00:01,000 --> 00:00:02,000\nB\n";
        let segs = parse(input, 10).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].id, 10);
        assert_eq!(segs[1].id, 11);
    }

    #[test]
    fn writes_and_reparses_roundtrip() {
        let seg = SubtitleSegment {
            id: 5,
            start: 1.234,
            end: 3.5,
            source_text: "".into(),
            translated_text: "Xin chào".into(),
            dubbing_text: String::new(),
            words: None,
            speaker: None,
            voice_id: None,
        };
        let srt = write(std::slice::from_ref(&seg), |s| s.translated_text.clone());
        let parsed = parse(&srt, 100).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!((parsed[0].start - seg.start).abs() < 0.002);
        assert!((parsed[0].end - seg.end).abs() < 0.002);
        assert_eq!(parsed[0].translated_text, "Xin chào");
    }

    #[test]
    fn rejects_missing_arrow() {
        let input = "1\n00:00:00,000 00:00:01,000\nA\n";
        assert!(parse(input, 0).is_err());
    }
}
