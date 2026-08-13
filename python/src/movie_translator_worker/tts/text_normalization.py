"""Conservative text preparation for local speech synthesis.

The normalizer intentionally preserves Vietnamese diacritics and punctuation:
both Piper and F5 use punctuation as a prosody signal. It only removes control
characters, canonicalises typography, and expands a few unambiguous symbols.
"""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RE = re.compile(r"\s*\n+\s*")
_REPEATED_PUNCT_RE = re.compile(r"([!?])\1{2,}")
_ELLIPSIS_RE = re.compile(r"(?:\.\s*){3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([,;:!?])(?=\S)")
_STANDALONE_NUMBER_RE = re.compile(
    r"(?<![\w:/.,-])\d{1,4}(?![\w:/-]|[.,]\d)"
)

_TYPOGRAPHIC_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": " - ",
        "\u00a0": " ",
    }
)


def normalize_tts_text(text: str, language: str = "vi") -> str:
    value = unicodedata.normalize("NFC", text or "")
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    value = value.translate(_TYPOGRAPHIC_TRANSLATION)
    value = _BLANK_RE.sub(" ", value)
    value = _ELLIPSIS_RE.sub("… ", value)
    value = _REPEATED_PUNCT_RE.sub(r"\1\1", value)

    if (language or "").lower().startswith("vi"):
        value = value.replace("%", " phần trăm")
        value = re.sub(r"(?<=\w)\s*&\s*(?=\w)", " và ", value)
        value = _STANDALONE_NUMBER_RE.sub(
            lambda match: _vietnamese_integer(int(match.group(0))),
            value,
        )

    value = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", value)
    value = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", value)
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def _vietnamese_integer(value: int) -> str:
    digits = (
        "không",
        "một",
        "hai",
        "ba",
        "bốn",
        "năm",
        "sáu",
        "bảy",
        "tám",
        "chín",
    )

    def under_hundred(number: int) -> str:
        if number < 10:
            return digits[number]
        tens, unit = divmod(number, 10)
        prefix = "mười" if tens == 1 else f"{digits[tens]} mươi"
        if unit == 0:
            return prefix
        suffix = (
            "mốt"
            if unit == 1 and tens > 1
            else "tư"
            if unit == 4 and tens > 1
            else "lăm"
            if unit == 5
            else digits[unit]
        )
        return f"{prefix} {suffix}"

    def under_thousand(number: int) -> str:
        if number < 100:
            return under_hundred(number)
        hundreds, rest = divmod(number, 100)
        prefix = f"{digits[hundreds]} trăm"
        if rest == 0:
            return prefix
        if rest < 10:
            return f"{prefix} lẻ {digits[rest]}"
        return f"{prefix} {under_hundred(rest)}"

    if value < 1000:
        return under_thousand(value)
    thousands, rest = divmod(value, 1000)
    prefix = f"{under_thousand(thousands)} nghìn"
    if rest == 0:
        return prefix
    if rest < 10:
        return f"{prefix} lẻ {digits[rest]}"
    return f"{prefix} {under_thousand(rest)}"
