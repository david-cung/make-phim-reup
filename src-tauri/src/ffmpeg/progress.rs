//! Parse `ffmpeg -progress pipe:1` output.
//!
//! FFmpeg emits `key=value\n` lines, terminated by either
//! `progress=continue` or `progress=end`. We collect a block until we
//! see one of those and return it. Callers translate `out_time_us` (or
//! the older `out_time_ms`, which is _also_ microseconds despite the
//! name) into a fraction of the media duration.

use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct ProgressBlock {
    pub raw: HashMap<String, String>,
    pub ended: bool,
}

impl ProgressBlock {
    pub fn out_time_us(&self) -> Option<u64> {
        self.raw
            .get("out_time_us")
            .and_then(|v| v.parse::<u64>().ok())
            .or_else(|| {
                self.raw
                    .get("out_time_ms")
                    .and_then(|v| v.parse::<u64>().ok())
            })
    }

    /// Fraction 0..=1 given the input's total duration in seconds.
    pub fn fraction(&self, duration_secs: f64) -> Option<f32> {
        if duration_secs <= 0.0 {
            return None;
        }
        let us = self.out_time_us()?;
        let total_us = (duration_secs * 1_000_000.0) as u64;
        if total_us == 0 {
            return None;
        }
        let f = (us as f64 / total_us as f64).clamp(0.0, 1.0) as f32;
        Some(f)
    }
}

/// Feed one line of ffmpeg's stdout at a time. Returns `Some(block)`
/// whenever a block terminator (`progress=continue|end`) is seen.
///
/// The caller is expected to keep a running `ProgressBlock` between
/// calls; typical usage:
///
/// ```ignore
/// let mut cur = ProgressBlock::default();
/// while let Some(line) = stream.next().await {
///     if let Some(done) = feed_line(&mut cur, &line) {
///         handle(done);
///     }
/// }
/// ```
pub fn feed_line(block: &mut ProgressBlock, line: &str) -> Option<ProgressBlock> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let (k, v) = line.split_once('=')?;
    let key = k.trim().to_string();
    let value = v.trim().to_string();
    if key == "progress" {
        block.ended = value == "end";
        let out = std::mem::take(block);
        return Some(out);
    }
    block.raw.insert(key, value);
    None
}

/// Convenience for tests / offline parsing: consume an entire buffer and
/// return every completed block plus any leftover partial block.
pub fn parse_progress_block(buf: &str) -> Vec<ProgressBlock> {
    let mut cur = ProgressBlock::default();
    let mut out = Vec::new();
    for line in buf.lines() {
        if let Some(done) = feed_line(&mut cur, line) {
            out.push(done);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typical_block() {
        let buf = "bitrate=32.0kbits/s\n\
                   total_size=64000\n\
                   out_time_us=2500000\n\
                   out_time_ms=2500000\n\
                   speed=1.03x\n\
                   progress=continue\n";
        let blocks = parse_progress_block(buf);
        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert!(!b.ended);
        assert_eq!(b.out_time_us(), Some(2_500_000));
        // 2.5s out of 5s → 0.5
        assert_eq!(b.fraction(5.0), Some(0.5));
    }

    #[test]
    fn end_marker_flips_ended() {
        let buf = "out_time_us=10000000\nprogress=end\n";
        let blocks = parse_progress_block(buf);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].ended);
    }

    #[test]
    fn clamps_at_one() {
        let buf = "out_time_us=999999999\nprogress=continue\n";
        let b = &parse_progress_block(buf)[0];
        assert_eq!(b.fraction(1.0), Some(1.0));
    }

    #[test]
    fn zero_duration_yields_none() {
        let buf = "out_time_us=1000\nprogress=continue\n";
        let b = &parse_progress_block(buf)[0];
        assert_eq!(b.fraction(0.0), None);
    }

    #[test]
    fn missing_out_time_yields_none() {
        let buf = "bitrate=32k\nprogress=continue\n";
        let b = &parse_progress_block(buf)[0];
        assert_eq!(b.fraction(10.0), None);
    }

    #[test]
    fn multiple_blocks() {
        let buf = "out_time_us=1000000\nprogress=continue\n\
                   out_time_us=2000000\nprogress=continue\n\
                   out_time_us=3000000\nprogress=end\n";
        let blocks = parse_progress_block(buf);
        assert_eq!(blocks.len(), 3);
        assert!(blocks[2].ended);
    }
}
