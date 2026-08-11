//! Unit tests for helpers that don't need a live app handle.

use super::models::TranslateOptions;
use super::service::build_cache_key;

fn opts_default() -> TranslateOptions {
    TranslateOptions {
        model: "qwen2.gguf".into(),
        source_language: "en".into(),
        target_language: "vi".into(),
        prompt_version: "translation_prompt_v1".into(),
        chunk_size: 30,
        context_before: 4,
        context_after: 2,
        temperature: 0.2,
        top_p: 0.95,
        max_tokens: 2048,
    }
}

#[test]
fn cache_key_is_deterministic() {
    let a = build_cache_key("sha256:trans", "sha256:audio", &opts_default());
    let b = build_cache_key("sha256:trans", "sha256:audio", &opts_default());
    assert_eq!(a, b);
    assert!(a.starts_with("sha256:"));
}

#[test]
fn cache_key_changes_with_model() {
    let base = build_cache_key("sha256:t", "sha256:a", &opts_default());
    let mut alt = opts_default();
    alt.model = "llama3.gguf".into();
    let changed = build_cache_key("sha256:t", "sha256:a", &alt);
    assert_ne!(base, changed);
}

#[test]
fn cache_key_changes_with_target_language() {
    let base = build_cache_key("sha256:t", "sha256:a", &opts_default());
    let mut alt = opts_default();
    alt.target_language = "zh".into();
    let changed = build_cache_key("sha256:t", "sha256:a", &alt);
    assert_ne!(base, changed);
}

#[test]
fn cache_key_changes_with_prompt_version() {
    let base = build_cache_key("sha256:t", "sha256:a", &opts_default());
    let mut alt = opts_default();
    alt.prompt_version = "translation_prompt_v99".into();
    let changed = build_cache_key("sha256:t", "sha256:a", &alt);
    assert_ne!(base, changed);
}

#[test]
fn cache_key_changes_with_chunk_settings() {
    let base = build_cache_key("sha256:t", "sha256:a", &opts_default());
    let mut alt = opts_default();
    alt.chunk_size = 20;
    let changed = build_cache_key("sha256:t", "sha256:a", &alt);
    assert_ne!(base, changed);
}

#[test]
fn cache_key_changes_with_transcript_key() {
    let base = build_cache_key("sha256:t", "sha256:a", &opts_default());
    let changed = build_cache_key("sha256:different", "sha256:a", &opts_default());
    assert_ne!(base, changed);
}
