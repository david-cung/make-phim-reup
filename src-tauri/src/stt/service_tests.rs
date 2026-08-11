//! Tests for pure helpers in :mod:`super` that don't need a
//! ``WorkerSupervisor``. Full request-response integration is
//! covered by the Python-side test suite.

#[cfg(test)]
mod tests {
    use super::super::models::SttOptions;
    use super::super::service::build_cache_key;

    #[test]
    fn cache_key_is_stable_for_same_inputs() {
        let opts = SttOptions {
            model: "small".into(),
            language: Some("en".into()),
            ..SttOptions::default()
        };
        let a = build_cache_key("sha256:aa", &opts);
        let b = build_cache_key("sha256:aa", &opts);
        assert_eq!(a, b);
    }

    #[test]
    fn cache_key_changes_with_model() {
        let a = build_cache_key(
            "sha256:aa",
            &SttOptions {
                model: "small".into(),
                ..Default::default()
            },
        );
        let b = build_cache_key(
            "sha256:aa",
            &SttOptions {
                model: "medium".into(),
                ..Default::default()
            },
        );
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_changes_with_language() {
        let a = build_cache_key(
            "h",
            &SttOptions {
                language: Some("en".into()),
                ..Default::default()
            },
        );
        let b = build_cache_key(
            "h",
            &SttOptions {
                language: Some("ja".into()),
                ..Default::default()
            },
        );
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_changes_with_audio_hash() {
        let a = build_cache_key("h1", &SttOptions::default());
        let b = build_cache_key("h2", &SttOptions::default());
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_changes_with_word_timestamps() {
        let a = build_cache_key(
            "h",
            &SttOptions {
                word_timestamps: false,
                ..Default::default()
            },
        );
        let b = build_cache_key(
            "h",
            &SttOptions {
                word_timestamps: true,
                ..Default::default()
            },
        );
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_language_none_matches_lowercase_missing() {
        // `None` and explicit `Some("Auto")` should produce different
        // strings today (we don't normalize `Some("auto") -> None`),
        // but two calls with the exact same input must match.
        let opts = SttOptions {
            language: None,
            ..Default::default()
        };
        assert_eq!(build_cache_key("h", &opts), build_cache_key("h", &opts),);
    }
}
