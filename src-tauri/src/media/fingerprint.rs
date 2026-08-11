//! Cheap, deterministic content fingerprint for cache keying.
//!
//! Fully hashing multi-GB movies is wasteful — we would spend more time
//! reading bytes than we save in avoided re-work. Instead we hash:
//!
//! ```text
//!   sha256( u64_le(size) || first_64 KiB || last_64 KiB )
//! ```
//!
//! Combined with the file size and (as a fallback) mtime, this is more
//! than sufficient to detect the two cases we actually care about:
//!
//!   * the user re-imports the same file (hash + size match → cache hit)
//!   * the user re-imports a re-encoded file that happens to have the
//!     same size (hash bytes differ → cache miss)
//!
//! Random small edits in the middle of a 20 GB file that don't touch
//! either sentinel window will not be detected. That's acceptable for a
//! cache key; a real integrity check (Phase 10) would sample more.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const WINDOW_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFingerprint {
    /// `sha256:<hex>` prefixed so future algorithms can coexist.
    pub hash: String,
    pub size_bytes: u64,
    pub modified_at: DateTime<Utc>,
}

impl SourceFingerprint {
    /// Two fingerprints of the same underlying content will match on
    /// `(hash, size_bytes)`. We deliberately ignore `modified_at` here:
    /// copying a file across a filesystem often changes mtime but the
    /// content is identical.
    pub fn matches_content(&self, other: &SourceFingerprint) -> bool {
        self.hash == other.hash && self.size_bytes == other.size_bytes
    }
}

pub fn fingerprint_file(path: &Path) -> std::io::Result<SourceFingerprint> {
    let meta = std::fs::metadata(path)?;
    let size = meta.len();
    let modified = meta
        .modified()
        .ok()
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(Utc::now);

    let mut file = File::open(path)?;

    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());

    let head_len = size.min(WINDOW_BYTES);
    if head_len > 0 {
        let mut buf = vec![0u8; head_len as usize];
        file.read_exact(&mut buf)?;
        hasher.update(&buf);
    }

    if size > 2 * WINDOW_BYTES {
        let tail_start = size - WINDOW_BYTES;
        file.seek(SeekFrom::Start(tail_start))?;
        let mut buf = vec![0u8; WINDOW_BYTES as usize];
        file.read_exact(&mut buf)?;
        hasher.update(&buf);
    }

    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(SourceFingerprint {
        hash: format!("sha256:{hex}"),
        size_bytes: size,
        modified_at: modified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempfile(contents: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lmt-fp-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("blob.bin");
        let mut f = File::create(&p).unwrap();
        f.write_all(contents).unwrap();
        p
    }

    #[test]
    fn identical_content_same_hash() {
        let a = tempfile(b"the quick brown fox jumped");
        let b = tempfile(b"the quick brown fox jumped");
        let fa = fingerprint_file(&a).unwrap();
        let fb = fingerprint_file(&b).unwrap();
        assert_eq!(fa.hash, fb.hash);
        assert!(fa.matches_content(&fb));
    }

    #[test]
    fn different_size_different_hash() {
        let a = tempfile(b"abcd");
        let b = tempfile(b"abcde");
        let fa = fingerprint_file(&a).unwrap();
        let fb = fingerprint_file(&b).unwrap();
        assert_ne!(fa.hash, fb.hash);
        assert!(!fa.matches_content(&fb));
    }

    #[test]
    fn small_file_hashes_all_bytes() {
        let a = tempfile(b"short");
        let f = fingerprint_file(&a).unwrap();
        assert!(f.hash.starts_with("sha256:"));
        assert_eq!(f.size_bytes, 5);
    }

    #[test]
    fn large_file_tail_matters() {
        // 200 KiB with different last byte → different hash even though
        // first 64 KiB is identical.
        let mut base = vec![0u8; 200 * 1024];
        for (i, b) in base.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let a = tempfile(&base);
        let mut b_bytes = base.clone();
        *b_bytes.last_mut().unwrap() ^= 0xFF;
        let b = tempfile(&b_bytes);

        let fa = fingerprint_file(&a).unwrap();
        let fb = fingerprint_file(&b).unwrap();
        assert_ne!(fa.hash, fb.hash);
    }
}
