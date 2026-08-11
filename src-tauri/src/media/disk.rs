//! Cross-platform available-disk-space checks.
//!
//! Wrapped as its own module so tests can stub the estimator without
//! needing a real filesystem call. Uses `fs4` (formerly `fs2`) for the
//! `statvfs`/`GetDiskFreeSpaceEx` calls.

use std::path::Path;

use fs4::available_space;

/// Bytes of free space on the filesystem holding `path`.
pub fn free_bytes(path: &Path) -> std::io::Result<u64> {
    available_space(path)
}

/// Estimate the bytes an audio extraction will produce.
///
/// PCM S16LE @ 16 kHz mono = 32,000 bytes/s. We add a 5% margin plus a
/// 1 MiB minimum floor so short clips don't get "0 bytes needed" surprises.
pub fn estimate_pcm_wav_size(duration_secs: f64, sample_rate: u32, channels: u16) -> u64 {
    let bytes_per_sample: u64 = 2; // s16
    let per_second = u64::from(sample_rate) * u64::from(channels) * bytes_per_sample;
    let raw = (per_second as f64 * duration_secs.max(0.0) * 1.05).ceil() as u64;
    raw.max(1 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_size_estimate_matches_expected() {
        // 1s of 16 kHz mono s16 → 32,000 bytes + 5% + floor
        let s = estimate_pcm_wav_size(1.0, 16_000, 1);
        assert!(s >= 1 << 20, "min 1 MiB floor: {s}");

        // 1 hour of 16 kHz mono s16 → ~115 MB nominal
        let hr = estimate_pcm_wav_size(3600.0, 16_000, 1);
        let nominal = (16_000u64 * 2) * 3600;
        assert!(
            hr >= nominal,
            "estimate should be >= nominal: {hr} < {nominal}"
        );
    }

    #[test]
    fn free_bytes_returns_something_positive_for_tmp() {
        let free = free_bytes(&std::env::temp_dir()).unwrap();
        assert!(free > 0, "expected some free space in /tmp");
    }
}
