"""Target-language validation adapters.

Core translation code asks for target-language conformity through this
adapter surface. Language-specific checks live here instead of being
hardwired into batching, ownership, or contract validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


UNTRANSLATED_SOURCE_FRAGMENT = "UNTRANSLATED_SOURCE_FRAGMENT"
FOREIGN_LANGUAGE_CONTAMINATION = "FOREIGN_LANGUAGE_CONTAMINATION"
CANDIDATE_LEAKAGE = "CANDIDATE_LEAKAGE"
TRANSLATOR_COMMENTARY_LEAK = "TRANSLATOR_COMMENTARY_LEAK"
EMPTY_TRANSLATION = "EMPTY_TRANSLATION"


@dataclass(frozen=True)
class TargetLanguageValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "details": dict(self.details),
        }


class TargetLanguageAdapter(Protocol):
    language: str

    def validate(
        self,
        text: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TargetLanguageValidation: ...


class GenericTargetLanguageAdapter:
    language = "generic"

    def validate(
        self,
        text: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TargetLanguageValidation:
        errors = [EMPTY_TRANSLATION] if not (text or "").strip() else []
        _ = source, metadata
        return TargetLanguageValidation(valid=not errors, errors=errors)


class VietnameseTargetLanguageAdapter(GenericTargetLanguageAdapter):
    language = "vi"

    def validate(
        self,
        text: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TargetLanguageValidation:
        stripped = (text or "").strip()
        errors: list[str] = []
        metadata = metadata or {}
        if not stripped:
            errors.append(EMPTY_TRANSLATION)
        cjk_matches = _CJK_RE.findall(stripped)
        if cjk_matches and not _all_preserved_tokens_allowed(stripped, metadata):
            errors.append(
                UNTRANSLATED_SOURCE_FRAGMENT
                if _CJK_RE.search(source or "")
                else FOREIGN_LANGUAGE_CONTAMINATION
            )
        if _UNEXPECTED_SCRIPT_RE.search(stripped):
            errors.append(FOREIGN_LANGUAGE_CONTAMINATION)
        if _JSON_OR_PROMPT_RE.search(stripped):
            errors.append(CANDIDATE_LEAKAGE)
        if _META_COMMENTARY_RE.search(stripped.casefold()):
            errors.append(TRANSLATOR_COMMENTARY_LEAK)
        if _looks_like_candidate_leak(stripped):
            errors.append(CANDIDATE_LEAKAGE)
        return TargetLanguageValidation(
            valid=not errors,
            errors=list(dict.fromkeys(errors)),
            details={"language": self.language},
        )


def adapter_for(target_language: str) -> TargetLanguageAdapter:
    if (target_language or "").casefold() == "vi":
        return VietnameseTargetLanguageAdapter()
    return GenericTargetLanguageAdapter()


def validate_target_language(
    target_language: str,
    text: str,
    *,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> TargetLanguageValidation:
    return adapter_for(target_language).validate(text, source=source, metadata=metadata)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_UNEXPECTED_SCRIPT_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff]")
_JSON_OR_PROMPT_RE = re.compile(
    r"(^\s*[\[{]|\"(?:translation|translated_text|segment_id|id)\"\s*:|"
    r"\b(?:translator note|note:|explanation:|option\s*[abc]\s*:|candidate\s*\d*\s*:)\b)",
    re.IGNORECASE,
)
_META_COMMENTARY_RE = re.compile(
    r"\b(cần dịch lại|không dịch được|không rõ|có thể dịch là|có lẽ nghĩa là|"
    r"bản dịch|phương án|translation|translator note)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_RE = re.compile(
    r"(\s/\s.*\s/\s|(?:^|\b)(?:Option|Candidate)\s*[ABC123]\s*:|"
    r"\b(?:or|hoac|hoặc)\b.{0,24}\b(?:or|hoac|hoặc)\b)",
    re.IGNORECASE,
)


def _looks_like_candidate_leak(text: str) -> bool:
    if _ALTERNATIVE_RE.search(text):
        return True
    slash_parts = [part.strip() for part in text.split("/") if part.strip()]
    if len(slash_parts) >= 3 and all(len(part) >= 5 for part in slash_parts):
        return True
    return False


def _all_preserved_tokens_allowed(text: str, metadata: dict[str, Any]) -> bool:
    allowed = {
        str(item)
        for item in (
            metadata.get("allowedPreservedTokens")
            or metadata.get("allowed_preserved_tokens")
            or []
        )
        if str(item)
    }
    if not allowed:
        return False
    tokens = set(_CJK_RE.findall(text))
    return bool(tokens) and tokens.issubset(allowed)
