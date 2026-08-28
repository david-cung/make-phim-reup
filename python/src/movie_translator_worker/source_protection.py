"""Source reconstruction and semantic protection helpers.

The routines here are deterministic and offline. They do not translate
the source; they preserve raw text, normalize only low-risk surface
forms, split obvious merged Chinese dialogue units, and extract facts
that downstream translation must keep intact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_OR_CJK_PUNCT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff，。！？、；：]")
_CJK_SENTENCE_END_RE = re.compile(r"([。！？!?]+)")
_CJK_BOUNDARY_RE = re.compile(
    r"(?<=[啊呀吗嘛么呢吧了啦？?])(?=(?:我们|你们|他们|她们|我|你|他|她|那|这|可是|但是|然后|还有|谁|什么|怎么|为什么))"
)
_SOURCE_NEGATION_RE = re.compile(r"(不|没|沒有|没有|不是|不要|不能|别|無|未)")
_QUESTION_RE = re.compile(r"(吗|呢|谁|什么|怎么|为什么|哪|几|多少|[?？])")
_COMMAND_RE = re.compile(r"(别|不要|请|快|马上|必须)")
_DUPLICATE_RE = re.compile(r"([\u4e00-\u9fff]{1,4})\1{2,}")
_MIXED_LATIN_RE = re.compile(r"[A-Za-z]{2,}")

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_DURATION_UNITS = {
    "小时": ("hour", "giờ"),
    "鐘頭": ("hour", "giờ"),
    "钟头": ("hour", "giờ"),
    "分钟": ("minute", "phút"),
    "分鐘": ("minute", "phút"),
    "秒": ("second", "giây"),
    "天": ("day", "ngày"),
    "年": ("year", "năm"),
    "个月": ("month", "tháng"),
    "月": ("month", "tháng"),
}
_QUANTITY_UNITS = {
    "个人": ("person", "người"),
    "人": ("person", "người"),
    "个": ("generic", ""),
    "次": ("time_count", "lần"),
}
_ACTION_HINTS = {
    "面试": ("interview", "phỏng vấn"),
    "介紹": ("introduce", "giới thiệu"),
    "介绍": ("introduce", "giới thiệu"),
    "等": ("wait", "đợi"),
    "去": ("go", "đi"),
    "结婚": ("marry", "kết hôn"),
    "結婚": ("marry", "kết hôn"),
    "来": ("come", "đến"),
    "告訴": ("tell", "nói"),
    "告诉": ("tell", "nói"),
}
_PERSON_REFS = {
    "我": "speaker",
    "你": "listener",
    "他": "he",
    "她": "she",
    "我们": "we",
    "我們": "we",
    "你们": "you_plural",
    "你們": "you_plural",
    "他们": "they",
    "他們": "they",
    "她们": "they_female",
    "她們": "they_female",
}
_VI_NUMBER_WORDS = {
    0: ("0", "không"),
    1: ("1", "một", "mốt"),
    2: ("2", "hai"),
    3: ("3", "ba"),
    4: ("4", "bốn", "tư"),
    5: ("5", "năm", "lăm"),
    6: ("6", "sáu"),
    7: ("7", "bảy"),
    8: ("8", "tám"),
    9: ("9", "chín"),
    10: ("10", "mười"),
}
_VI_UNIT_WORDS = {
    "hour": ("giờ", "tiếng"),
    "minute": ("phút",),
    "second": ("giây",),
    "day": ("ngày", "hôm"),
    "year": ("năm",),
    "month": ("tháng",),
    "person": ("người",),
    "time_count": ("lần",),
}


@dataclass(frozen=True)
class SourceUnit:
    source_segment_id: str
    sub_segment_id: str
    text_cn: str
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_segment_id": self.source_segment_id,
            "sourceSegmentId": self.source_segment_id,
            "sub_segment_id": self.sub_segment_id,
            "subSegmentId": self.sub_segment_id,
            "text_cn": self.text_cn,
            "textCn": self.text_cn,
        }
        if self.start is not None:
            payload["start"] = round(float(self.start), 3)
        if self.end is not None:
            payload["end"] = round(float(self.end), 3)
        return payload


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def normalize_source_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def split_logical_source_units(text: str) -> list[str]:
    normalized = normalize_source_text(text)
    if not normalized:
        return []
    if not contains_cjk(normalized):
        return [normalized]

    pieces: list[str] = []
    start = 0
    for match in _CJK_SENTENCE_END_RE.finditer(normalized):
        end = match.end()
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    tail = normalized[start:].strip()
    if tail:
        pieces.append(tail)
    if len(pieces) <= 1:
        pieces = [part for part in _CJK_BOUNDARY_RE.split(normalized) if part.strip()]

    cleaned = [p.strip(" ，,") for p in pieces if p.strip(" ，,")]
    if len(cleaned) <= 1:
        return cleaned or [normalized]
    # Avoid micro-splits such as interjections becoming standalone subtitle rows.
    merged: list[str] = []
    for piece in cleaned:
        if merged and len(_CJK_RE.findall(piece)) <= 1:
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def source_quality(text: str, *, start: float | None = None, end: float | None = None) -> dict[str, Any]:
    normalized = normalize_source_text(text)
    flags: list[str] = []
    if contains_cjk(normalized) and _MIXED_LATIN_RE.search(normalized):
        flags.append("MIXED_LANGUAGE_FRAGMENT")
    if _DUPLICATE_RE.search(normalized):
        flags.append("DUPLICATED_WORDS")
    cjk_chars = _CJK_RE.findall(normalized)
    unusual = [
        ch for ch in normalized
        if ord(ch) > 127 and not _CJK_OR_CJK_PUNCT_RE.match(ch) and not ch.isspace()
    ]
    if unusual:
        flags.append("UNUSUAL_CHARACTERS")
    if len(cjk_chars) >= 24 and not re.search(r"[。！？!?，,；;]", normalized):
        flags.append("LONG_UNPUNCTUATED_CHINESE")
    if start is not None and end is not None and end < start:
        flags.append("BAD_TIMING")
    confidence = max(0.2, 0.96 - 0.14 * len(set(flags)))
    return {
        "source_confidence": round(confidence, 3),
        "sourceConfidence": round(confidence, 3),
        "quality_flags": list(dict.fromkeys(flags)),
        "qualityFlags": list(dict.fromkeys(flags)),
    }


def semantic_analysis(text: str) -> dict[str, Any]:
    normalized = normalize_source_text(text)
    numbers = _extract_numbers(normalized)
    actions = [
        {"source": key, "action": action, "vi_hint": vi_hint, "viHint": vi_hint}
        for key, (action, vi_hint) in _ACTION_HINTS.items()
        if key in normalized
    ]
    negations = list(dict.fromkeys(_SOURCE_NEGATION_RE.findall(normalized)))
    person_refs = [
        {"source": key, "role": role}
        for key, role in _PERSON_REFS.items()
        if key in normalized
    ]
    flags: list[str] = []
    if _QUESTION_RE.search(normalized) and not _non_question_negated_difficulty(normalized):
        flags.append("QUESTION")
    if _COMMAND_RE.search(normalized):
        flags.append("COMMAND")
    if negations:
        flags.append("NEGATION")
    if numbers:
        flags.append("NUMERIC_FACTS")
    if actions:
        flags.append("ACTION_HINTS")
    return {
        "subject": None,
        "action": actions[0]["action"] if actions else None,
        "actions": actions,
        "object": None,
        "time": [n for n in numbers if n["kind"] == "duration"],
        "quantity": [n for n in numbers if n["kind"] == "quantity"],
        "numbers": numbers,
        "person_references": person_refs,
        "personReferences": person_refs,
        "negation": negations,
        "is_question": "QUESTION" in flags,
        "isQuestion": "QUESTION" in flags,
        "is_command": "COMMAND" in flags,
        "isCommand": "COMMAND" in flags,
        "emotion": _emotion_hint(normalized),
        "meaning_confidence": _meaning_confidence(normalized, flags),
        "meaningConfidence": _meaning_confidence(normalized, flags),
        "protection_flags": flags,
        "protectionFlags": flags,
    }


def source_protection_payload(
    *,
    segment_id: int | str,
    text: str,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, Any]:
    raw = text or ""
    normalized = normalize_source_text(raw)
    units = split_logical_source_units(normalized)
    quality = source_quality(normalized, start=start, end=end)
    subsegments = [
        SourceUnit(str(segment_id), _sub_id(segment_id, idx), unit).to_dict()
        for idx, unit in enumerate(units)
    ]
    segmentation_flags: list[str] = []
    if len(units) > 1:
        segmentation_flags.append("MULTIPLE_DIALOGUE_UNITS")
    if contains_cjk(normalized) and len(_CJK_RE.findall(normalized)) >= 24 and not re.search(r"[。！？!?，,；;]", normalized):
        segmentation_flags.append("LONG_UNPUNCTUATED_CHINESE")
    if start is not None and end is not None and end < start:
        segmentation_flags.append("OVERLAPPING_OR_BAD_TIMING")
    return {
        "raw_source": raw,
        "rawSource": raw,
        "normalized_source": normalized,
        "normalizedSource": normalized,
        "source_segment_id": str(segment_id),
        "sourceSegmentId": str(segment_id),
        "logical_subsegments": subsegments,
        "logicalSubsegments": subsegments,
        "segmentation_flags": segmentation_flags,
        "segmentationFlags": segmentation_flags,
        "semantic": semantic_analysis(normalized),
        "source_quality": quality,
        "sourceQuality": quality,
    }


def split_source_segment(
    *,
    segment_id: int | str,
    text: str,
    start: float,
    end: float,
) -> list[SourceUnit]:
    normalized = normalize_source_text(text)
    units = split_logical_source_units(normalized)
    if len(units) <= 1:
        return [SourceUnit(str(segment_id), _sub_id(segment_id, 0), normalized, start, end)]
    duration = max(0.001, end - start)
    weights = [max(1, len(_CJK_RE.findall(unit)) or len(unit)) for unit in units]
    total = sum(weights)
    cursor = start
    out: list[SourceUnit] = []
    for index, (unit, weight) in enumerate(zip(units, weights)):
        if index == len(units) - 1:
            unit_end = end
        else:
            unit_end = cursor + duration * (weight / total)
        out.append(SourceUnit(str(segment_id), _sub_id(segment_id, index), unit, cursor, unit_end))
        cursor = unit_end
    return out


def validate_translation_against_source(
    *,
    source: str,
    translation: str,
    protection: dict[str, Any] | None = None,
) -> list[str]:
    payload = protection or source_protection_payload(segment_id="source", text=source)
    semantic = payload.get("semantic") if isinstance(payload, dict) else {}
    if not isinstance(semantic, dict):
        semantic = semantic_analysis(source)
    issues: list[str] = []
    if contains_cjk(translation):
        issues.append("UNTRANSLATED_CHINESE")
    for fact in semantic.get("numbers") or []:
        if isinstance(fact, dict):
            issues.extend(_number_fact_issues(fact, translation))
    if semantic.get("negation") and not _source_negation_rendered(source, translation):
        issues.append("MISSING_NEGATION")
    if semantic.get("is_question") or semantic.get("isQuestion"):
        if not _looks_like_vi_question(translation):
            issues.append("QUESTION_CHANGED_TO_STATEMENT")
    for action in semantic.get("actions") or []:
        if isinstance(action, dict):
            vi_hint = str(action.get("vi_hint") or action.get("viHint") or "")
            if vi_hint and vi_hint not in _normalise_vi(translation):
                issues.append("MISSING_ACTION")
    return list(dict.fromkeys(issues))


def _extract_numbers(text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    unit_pattern = "|".join(sorted(map(re.escape, list(_DURATION_UNITS) + list(_QUANTITY_UNITS)), key=len, reverse=True))
    for match in re.finditer(rf"(?P<num>\d+(?:\.\d+)?)\s*(?:个)?(?P<unit>{unit_pattern})", text):
        value = float(match.group("num"))
        if value.is_integer():
            value = int(value)
        unit = match.group("unit")
        facts.append(_number_fact(match.group(0), value, unit))
    cn_number = "".join(_CN_DIGITS) + "".join(_CN_UNITS)
    for match in re.finditer(rf"(?P<num>[{cn_number}]+)\s*(?:个)?(?P<unit>{unit_pattern})", text):
        value = chinese_number_to_int(match.group("num"))
        if value is None:
            continue
        # Avoid duplicating a fact already matched through a normalized Arabic number.
        if any(f["source"] == match.group(0) for f in facts):
            continue
        facts.append(_number_fact(match.group(0), value, match.group("unit")))
    return facts


def _number_fact(source: str, value: int | float, unit: str) -> dict[str, Any]:
    if unit in _DURATION_UNITS:
        normalized_unit, vi_unit = _DURATION_UNITS[unit]
        kind = "duration"
    else:
        normalized_unit, vi_unit = _QUANTITY_UNITS[unit]
        kind = "quantity"
    return {
        "source": source,
        "value": value,
        "unit": normalized_unit,
        "source_unit": unit,
        "sourceUnit": unit,
        "vi_unit": vi_unit,
        "viUnit": vi_unit,
        "kind": kind,
        "must_preserve": True,
        "mustPreserve": True,
    }


def chinese_number_to_int(text: str) -> int | None:
    if not text:
        return None
    if all(ch in _CN_DIGITS for ch in text):
        value = 0
        for ch in text:
            value = value * 10 + _CN_DIGITS[ch]
        return value
    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def _number_fact_issues(fact: dict[str, Any], translation: str) -> list[str]:
    issues: list[str] = []
    value = fact.get("value")
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        int_value = -1
    normal = _normalise_vi(translation)
    number_forms = _VI_NUMBER_WORDS.get(int_value, (str(value),))
    has_number = any(re.search(rf"\b{re.escape(form)}\b", normal) for form in number_forms if form)
    if not has_number:
        issues.append("NUMBER_MISMATCH")
    unit = str(fact.get("unit") or "")
    expected_units = _VI_UNIT_WORDS.get(unit, ())
    if expected_units and not any(word in normal for word in expected_units):
        issues.append("DURATION_MISMATCH" if fact.get("kind") == "duration" else "QUANTITY_MISMATCH")
    if unit == "hour" and any(word in normal for word in ("năm", "tháng", "ngày")):
        issues.append("DURATION_MISMATCH")
    if unit == "year" and any(word in normal for word in ("giờ", "tiếng", "phút")):
        issues.append("DURATION_MISMATCH")
    if unit == "person" and "người" not in normal:
        issues.append("QUANTITY_MISMATCH")
    return issues


def _missing_vi_negation(translation: str) -> bool:
    normal = _normalise_vi(translation)
    return not any(word in normal.split() for word in ("không", "chưa", "đừng", "chẳng", "chả", "khỏi"))


def _source_negation_rendered(source: str, translation: str) -> bool:
    normal = _normalise_vi(translation)
    if _non_question_negated_difficulty(normalize_source_text(source)):
        return any(phrase in normal for phrase in ("đơn giản", "dễ", "khó gì", "không khó"))
    return not _missing_vi_negation(translation)


def _non_question_negated_difficulty(source: str) -> bool:
    return bool(re.search(r"没\s*什么\s*难|沒有\s*什麼\s*難|没什么难|没有什么难|沒什麼難", source))


def _looks_like_vi_question(translation: str) -> bool:
    normal = _normalise_vi(translation)
    return "?" in translation or any(word in normal.split() for word in ("à", "ư", "sao", "hả", "chứ", "không", "chưa", "ai", "gì", "nào", "đâu"))


def _normalise_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _emotion_hint(text: str) -> str:
    if re.search(r"[!！]|别|不要|怎么|为什么", text):
        return "urgent_or_emphatic"
    if _QUESTION_RE.search(text):
        return "questioning"
    return "neutral"


def _meaning_confidence(text: str, flags: Iterable[str]) -> float:
    base = 0.86 if contains_cjk(text) else 0.78
    if "NUMERIC_FACTS" in flags:
        base += 0.04
    if "ACTION_HINTS" in flags:
        base += 0.03
    return round(min(0.96, base), 3)


def _sub_id(segment_id: int | str, index: int) -> str:
    suffix = chr(ord("a") + index) if 0 <= index < 26 else str(index + 1)
    return f"{segment_id}{suffix}"
