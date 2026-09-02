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
_SOURCE_FINAL_PARTICLE_RE = re.compile(r"[啊呀吧呢嘛哦喔啦咯哇呗]$")
_UNCERTAINTY_RE = re.compile(r"(可能|也许|也許|大概|应该|應該|好像|似乎|恐怕)")
_STRONG_CERTAINTY_RE = re.compile(r"(一定|肯定|必须|必須|绝对|絕對)")
_ASPECT_ALREADY_RE = re.compile(r"(已经|已經|了)$|已经|已經")
_ASPECT_STILL_RE = re.compile(r"(还|還|仍然|一直)")
_ASPECT_NOT_YET_RE = re.compile(r"(还没|還沒|还没有|還沒有|尚未|未)")
_VI_META_COMMENTARY_RE = re.compile(
    r"\b(cần dịch lại|không dịch được|không rõ|có thể dịch là|có lẽ nghĩa là|"
    r"bản dịch|phương án|option|candidate|translation|translator note)\b",
    re.IGNORECASE,
)
_VI_QUESTION_FINAL_RE = re.compile(r"(?:\b(?:à|á|hả|sao|ư|chứ|nhỉ)\s*\?)\s*$", re.IGNORECASE)
_VI_REQUEST_PARTICLE_RE = re.compile(r"\b(?:nhé|nha)\s*[.!…]*$", re.IGNORECASE)
_VI_UNCERTAINTY_RE = re.compile(r"\b(chắc|có lẽ|hình như|có thể|dường như|chắc là)\b", re.IGNORECASE)
_VI_UNSUPPORTED_RELATION_RE = re.compile(
    r"\b(bạn trai|bạn gái|người yêu|hẹn hò|tìm bạn gái|tìm bạn trai|lấy chồng|lấy vợ)\b",
    re.IGNORECASE,
)

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
    "相亲": ("matchmaking", "xem mắt"),
    "相親": ("matchmaking", "xem mắt"),
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
    speech_act = _classify_speech_act(normalized)
    question_type = _question_type(normalized, speech_act)
    polarity = _polarity(normalized, negations)
    predicate = _predicate_hint(normalized, actions)
    events = _event_hints(normalized, actions)
    certainty = _certainty(normalized)
    aspect = _aspect(normalized)
    source_particles = _source_final_particles(normalized)
    naturalization_budget = _naturalization_budget(normalized, speech_act)
    must_preserve = _must_preserve(
        speech_act=speech_act,
        question_type=question_type,
        predicate=predicate,
        polarity=polarity,
        numbers=numbers,
        events=events,
        certainty=certainty,
    )
    semantic_anchors = {
        "speech_act": speech_act,
        "speechAct": speech_act,
        "question_type": question_type,
        "questionType": question_type,
        "predicate": predicate,
        "polarity": polarity,
        "numbers": numbers,
        "events": events,
        "certainty": certainty,
        "aspect": aspect,
        "must_preserve": must_preserve,
        "mustPreserve": must_preserve,
    }
    literal_baseline = _literal_baseline_hint(
        speech_act=speech_act,
        question_type=question_type,
        predicate=predicate,
        numbers=numbers,
    )
    flags: list[str] = []
    if speech_act in {
        "QUESTION",
        "RHETORICAL_QUESTION",
        "CONFIRMATION_SEEKING",
    }:
        flags.append("QUESTION")
    if speech_act in {"COMMAND", "REQUEST"}:
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
        "speech_act": speech_act,
        "speechAct": speech_act,
        "sentence_mood": speech_act,
        "sentenceMood": speech_act,
        "question_type": question_type,
        "questionType": question_type,
        "is_question": "QUESTION" in flags,
        "isQuestion": "QUESTION" in flags,
        "is_command": "COMMAND" in flags,
        "isCommand": "COMMAND" in flags,
        "polarity": polarity,
        "predicate": predicate,
        "events": events,
        "certainty": certainty,
        "aspect": aspect,
        "source_particles": source_particles,
        "sourceParticles": source_particles,
        "semantic_anchors": semantic_anchors,
        "semanticAnchors": semantic_anchors,
        "naturalization_budget": naturalization_budget,
        "naturalizationBudget": naturalization_budget,
        "literal_baseline_hint": literal_baseline,
        "literalBaselineHint": literal_baseline,
        "must_preserve": must_preserve,
        "mustPreserve": must_preserve,
        "emotion": _emotion_hint(normalized),
        "meaning_confidence": _meaning_confidence(normalized, flags),
        "meaningConfidence": _meaning_confidence(normalized, flags),
        "protection_flags": flags,
        "protectionFlags": flags,
    }


def _classify_speech_act(text: str) -> str:
    if not text:
        return "STATEMENT"
    stripped = text.rstrip()
    if stripped.endswith(("...", "…", "——", "—")):
        return "UNFINISHED"
    if _rhetorical_question(text):
        return "RHETORICAL_QUESTION"
    if _confirmation_question(text):
        return "CONFIRMATION_SEEKING"
    if _is_question_text(text):
        return "QUESTION"
    if _second_person_directive(text):
        return "REQUEST"
    if re.search(r"(建议|建議|不如|要不|还是|還是).{0,8}(吧)?$", text):
        return "SUGGESTION"
    if re.search(r"(请|請|麻烦|麻煩|拜托)", text):
        return "REQUEST"
    if _COMMAND_RE.search(text):
        return "COMMAND"
    if stripped.endswith(("!", "！")):
        return "EXCLAMATION"
    if re.fullmatch(r"(嗯|哦|好|是|对|對|不是|没有|沒有|行|可以)[。.!！]?", stripped):
        return "RESPONSE"
    return "STATEMENT"


def _is_question_text(text: str) -> bool:
    if _non_question_negated_difficulty(text):
        return False
    return bool(_QUESTION_RE.search(text))


def _rhetorical_question(text: str) -> bool:
    if "难道" in text or "難道" in text:
        return True
    return bool(re.search(r"(怎么会|怎麼會|不是.+吗|不是.+嗎|哪有)", text))


def _confirmation_question(text: str) -> bool:
    if re.search(r"(对不对|對不對|是不是|是吧|对吧|對吧)", text):
        return True
    if text.endswith(("吧?", "吧？")):
        return True
    return False


def _second_person_directive(text: str) -> bool:
    return bool(
        re.search(
            r"(你|您).{0,6}(早点|早點|快点|快點|过来|過來|回来|回來|听|聽|说|說|"
            r"走|去|等一下|等等|记得|記得)",
            text,
        )
    )


def _question_type(text: str, speech_act: str) -> str | None:
    if speech_act not in {"QUESTION", "RHETORICAL_QUESTION", "CONFIRMATION_SEEKING"}:
        return None
    if speech_act == "CONFIRMATION_SEEKING":
        return "CONFIRMATION"
    if speech_act == "RHETORICAL_QUESTION":
        return "RHETORICAL"
    if "为什么" in text or "為什麼" in text or "为啥" in text or "为什麽" in text:
        return "WHY"
    if re.search(r"(怎么|怎麼).{0,4}(没|沒有|没有|不)", text):
        return "WHY"
    if "谁" in text or "誰" in text:
        return "WHO"
    if "什么时候" in text or "什麼時候" in text or "何时" in text or "幾時" in text:
        return "WHEN"
    if re.search(r"(哪里|哪裡|哪儿|哪兒|在哪|去哪)", text):
        return "WHERE"
    if "多少" in text or "几" in text or "幾" in text:
        return "HOW_MANY"
    if "什么" in text or "什麼" in text:
        return "WHAT"
    if "怎么" in text or "怎麼" in text or "如何" in text or "怎么样" in text or "怎麼樣" in text:
        return "HOW"
    if re.search(r"(吗|嗎|有没有|有沒有|能不能|可不可以|要不要)", text):
        return "YES_NO"
    return "YES_NO" if text.endswith(("?", "？")) else None


def _polarity(text: str, negations: list[str]) -> str:
    if re.search(r"(不是不|不能不|不得不|没有不|沒有不)", text):
        return "double_negative"
    return "negative" if negations else "positive"


def _predicate_hint(text: str, actions: list[dict[str, str]]) -> str | None:
    if _non_question_negated_difficulty(text):
        return "not_difficult"
    if re.search(r"(很难|很難|困难|困難)", text):
        return "difficult"
    if re.search(r"(结婚|結婚)", text):
        return "marriage"
    if re.search(r"(相亲|相親)", text):
        return "matchmaking"
    if actions:
        return actions[0].get("action")
    return None


def _event_hints(text: str, actions: list[dict[str, str]]) -> list[str]:
    events = [str(item.get("action")) for item in actions if item.get("action")]
    if re.search(r"(结婚|結婚)", text) and "marriage" not in events:
        events.append("marriage")
    if re.search(r"(相亲|相親)", text) and "matchmaking" not in events:
        events.append("matchmaking")
    return list(dict.fromkeys(events))


def _certainty(text: str) -> str:
    if _UNCERTAINTY_RE.search(text):
        return "possible"
    if _STRONG_CERTAINTY_RE.search(text):
        return "certain_emphatic"
    return "certain"


def _aspect(text: str) -> str | None:
    if _ASPECT_NOT_YET_RE.search(text):
        return "not_yet"
    if _ASPECT_ALREADY_RE.search(text):
        return "completed_or_current_result"
    if _ASPECT_STILL_RE.search(text):
        return "ongoing_or_still"
    return None


def _source_final_particles(text: str) -> list[dict[str, str]]:
    particles: list[dict[str, str]] = []
    stripped = text.rstrip("。！？!?… ")
    match = _SOURCE_FINAL_PARTICLE_RE.search(stripped)
    if match:
        particles.append(
            {
                "source": match.group(0),
                "rule": "analyze sentence function first; do not map directly to a Vietnamese particle",
            }
        )
    return particles


def _naturalization_budget(text: str, speech_act: str) -> str:
    risky = bool(
        _SOURCE_NEGATION_RE.search(text)
        or _extract_numbers(text)
        or _is_question_text(text)
        or re.search(r"(相亲|相親|结婚|結婚|关系|關係)", text)
    )
    if speech_act in {"UNFINISHED", "RESPONSE"}:
        return "LEVEL_0_1"
    if risky:
        return "LEVEL_1"
    return "LEVEL_1_2"


def _must_preserve(
    *,
    speech_act: str,
    question_type: str | None,
    predicate: str | None,
    polarity: str,
    numbers: list[dict[str, Any]],
    events: list[str],
    certainty: str,
) -> list[str]:
    anchors = [f"speech_act:{speech_act}", f"polarity:{polarity}", f"certainty:{certainty}"]
    if question_type:
        anchors.append(f"question_type:{question_type}")
    if predicate:
        anchors.append(f"predicate:{predicate}")
    anchors.extend(f"event:{event}" for event in events)
    for fact in numbers:
        anchors.append(f"number:{fact.get('value')}:{fact.get('unit')}")
    return list(dict.fromkeys(anchors))


def _literal_baseline_hint(
    *,
    speech_act: str,
    question_type: str | None,
    predicate: str | None,
    numbers: list[dict[str, Any]],
) -> str | None:
    if predicate == "marriage" and speech_act == "STATEMENT":
        return "Chúng tôi đã kết hôn."
    if predicate == "not_difficult":
        return "Chuyện đó cũng không có gì khó."
    if question_type == "WHY":
        return "Sao lại như vậy?"
    if numbers:
        fact = numbers[0]
        return f"Preserve {fact.get('value')} {fact.get('unit')} exactly."
    return None


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
    if _translator_commentary_leak(translation):
        issues.append("TRANSLATOR_COMMENTARY_LEAK")
    speech_act = str(semantic.get("speechAct") or semantic.get("speech_act") or "STATEMENT")
    source_question_type = semantic.get("questionType") or semantic.get("question_type")
    vi_speech_act, vi_question_type = _vi_speech_act_and_question_type(translation)
    if speech_act == "STATEMENT" and vi_speech_act in {
        "QUESTION",
        "CONFIRMATION_SEEKING",
    }:
        issues.append("STATEMENT_TO_QUESTION_ERROR")
        if _VI_QUESTION_FINAL_RE.search(translation):
            issues.append("UNSUPPORTED_PARTICLE_INSERTION")
    if speech_act in {"QUESTION", "RHETORICAL_QUESTION", "CONFIRMATION_SEEKING"}:
        if vi_speech_act == "STATEMENT":
            issues.append("QUESTION_TO_STATEMENT_ERROR")
        elif (
            source_question_type
            and vi_question_type
            and str(source_question_type) != vi_question_type
            and not (str(source_question_type) == "CONFIRMATION" and vi_question_type == "YES_NO")
        ):
            issues.append("QUESTION_TYPE_ERROR")
    if speech_act == "STATEMENT" and _unsupported_request_particle(translation, semantic):
        issues.append("UNSUPPORTED_PARTICLE_INSERTION")
    for fact in semantic.get("numbers") or []:
        if isinstance(fact, dict):
            numeric = _number_fact_issues(fact, translation)
            issues.extend(numeric)
            if numeric:
                issues.append("NUMERIC_FIDELITY_ERROR")
    if semantic.get("negation") and not _source_negation_rendered(source, translation):
        issues.append("MISSING_NEGATION")
        issues.append("POLARITY_ERROR")
    if semantic.get("is_question") or semantic.get("isQuestion"):
        if not _looks_like_vi_question(translation):
            issues.append("QUESTION_CHANGED_TO_STATEMENT")
    for action in semantic.get("actions") or []:
        if isinstance(action, dict):
            vi_hint = str(action.get("vi_hint") or action.get("viHint") or "")
            action_name = str(action.get("action") or "")
            if vi_hint and not _action_rendered(action_name, vi_hint, translation):
                issues.append("MISSING_ACTION")
                issues.append("PREDICATE_ERROR")
    if _unsupported_semantic_addition(source, translation, semantic):
        issues.append("SEMANTIC_ADDITION")
    if _predicate_changed(source, translation, semantic):
        issues.append("PREDICATE_ERROR")
    if _certainty_changed(translation, semantic):
        issues.append("CERTAINTY_ERROR")
    if _aspect_changed(translation, semantic):
        issues.append("PREDICATE_ERROR")
    return list(dict.fromkeys(issues))


def _translator_commentary_leak(translation: str) -> bool:
    normal = _normalise_vi(translation)
    return bool(_VI_META_COMMENTARY_RE.search(normal))


def _vi_speech_act_and_question_type(translation: str) -> tuple[str, str | None]:
    stripped = (translation or "").strip()
    normal = _normalise_vi(stripped)
    accentless = _strip_vi_accents(normal)
    has_question_mark = "?" in stripped
    if not stripped:
        return "STATEMENT", None
    if _vi_why_question(accentless):
        return "QUESTION", "WHY"
    if not has_question_mark and not _vi_confirmation_question(stripped, accentless):
        return "STATEMENT", None
    if re.search(r"\b(ai)\b", accentless):
        return "QUESTION", "WHO"
    if re.search(r"\b(khi nao|bao gio|luc nao)\b", accentless):
        return "QUESTION", "WHEN"
    if re.search(r"\b(o dau|dau|noi nao|cho nao)\b", accentless):
        return "QUESTION", "WHERE"
    if re.search(r"\b(bao nhieu|may)\b", accentless):
        return "QUESTION", "HOW_MANY"
    if re.search(r"\b(cai gi|gi|chuyen gi)\b", accentless):
        return "QUESTION", "WHAT"
    if re.search(r"\b(nhu the nao|the nao|lam sao|bang cach nao)\b", accentless):
        return "QUESTION", "HOW"
    if _vi_confirmation_question(stripped, accentless):
        return "CONFIRMATION_SEEKING", "YES_NO"
    if "?" in stripped:
        return "QUESTION", "YES_NO"
    return "STATEMENT", None


def _vi_why_question(accentless: str) -> bool:
    if re.search(r"\b(tai sao|vi sao|sao lai)\b", accentless):
        return True
    # Sentence-initial "Sao ..." asks why/how come. Clause-final
    # "... sao?" is a yes/no confirmation particle.
    return bool(re.match(r"^\s*sao\b", accentless))


def _vi_confirmation_question(original: str, accentless: str) -> bool:
    if _VI_QUESTION_FINAL_RE.search(original):
        return True
    if re.search(r"\b(phai khong|dung khong|duoc khong|co khong)\s*\??$", accentless):
        return True
    if "?" not in original:
        return False
    if re.search(r"\b(chua|khong)\s*\?$", accentless):
        return True
    return bool(re.search(r"\b(co|da|dang|se)\b.{0,40}\b(khong|chua)\b", accentless))


def _unsupported_request_particle(translation: str, semantic: dict[str, Any]) -> bool:
    predicate = str(semantic.get("predicate") or "")
    if predicate == "not_difficult":
        return False
    if _VI_REQUEST_PARTICLE_RE.search(_normalise_vi(translation)):
        return True
    return False


def _action_rendered(action: str, vi_hint: str, translation: str) -> bool:
    normal = _normalise_vi(translation)
    accentless = _strip_vi_accents(normal)
    synonyms = {
        "marry": ("ket hon", "cuoi", "hon nhan"),
        "matchmaking": ("xem mat", "mai moi", "mai mối"),
        "interview": ("phong van",),
        "introduce": ("gioi thieu",),
        "wait": ("doi", "cho"),
        "go": ("di",),
        "come": ("den", "toi"),
        "tell": ("noi", "bao", "ke"),
    }.get(action, (_strip_vi_accents(vi_hint),))
    return any(synonym and synonym in accentless for synonym in synonyms)


def _unsupported_semantic_addition(
    source: str,
    translation: str,
    semantic: dict[str, Any],
) -> bool:
    normal = _normalise_vi(translation)
    accentless = _strip_vi_accents(normal)
    normalized_source = normalize_source_text(source)
    if not _VI_UNSUPPORTED_RELATION_RE.search(normal) and "chịu khó" not in normal and "tổng tài" not in normal:
        _ = semantic
        return False
    if "chịu khó" in normal and not re.search(r"(努力|辛苦|忍|坚持|堅持| chịu khó)", normalized_source):
        return True
    if re.search(r"\b(ban trai|ban gai|nguoi yeu|hen ho|tim ban gai|tim ban trai)\b", accentless):
        if not re.search(r"(男朋友|女朋友|朋友|恋爱|戀愛|约会|約會|对象|對象)", normalized_source):
            return True
    if re.search(r"\b(lay chong|lay vo)\b", accentless):
        if not re.search(r"(老公|老婆|丈夫|妻子|嫁|娶)", normalized_source):
            return True
    if "tổng tài" in normal and "总裁" not in normalized_source:
        return True
    _ = semantic
    return False


def _predicate_changed(
    source: str,
    translation: str,
    semantic: dict[str, Any],
) -> bool:
    predicate = str(semantic.get("predicate") or "")
    normal = _normalise_vi(translation)
    accentless = _strip_vi_accents(normal)
    if predicate == "not_difficult":
        has_difficulty = (
            "kho" in accentless
            or "don gian" in accentless
            or "de" in accentless.split()
        )
        if not has_difficulty:
            return True
        if "chịu khó" in normal or re.search(r"\b(co gang|rang|rang len|co lay)\b", accentless):
            return True
    if predicate in {"marriage", "matchmaking"} and re.search(
        r"\b(tim ban gai|tim ban trai|hen ho)\b",
        accentless,
    ):
        return True
    _ = source
    return False


def _certainty_changed(translation: str, semantic: dict[str, Any]) -> bool:
    certainty = str(semantic.get("certainty") or "certain")
    if certainty in {"possible", "speculative"}:
        return False
    return bool(_VI_UNCERTAINTY_RE.search(_normalise_vi(translation)))


def _aspect_changed(translation: str, semantic: dict[str, Any]) -> bool:
    aspect = semantic.get("aspect")
    if not aspect:
        return False
    normal = _strip_vi_accents(_normalise_vi(translation))
    if aspect in {"completed_or_current_result", "ongoing_or_still"} and re.search(r"\bse\b", normal):
        return True
    if aspect == "not_yet" and not re.search(r"\b(chua|van chua|chua tung)\b", normal):
        return True
    return False


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
        return any(
            phrase in normal
            for phrase in ("đơn giản", "dễ", "khó gì", "không khó", "không có gì khó")
        )
    return not _missing_vi_negation(translation)


def _non_question_negated_difficulty(source: str) -> bool:
    return bool(re.search(r"没\s*什么\s*难|沒有\s*什麼\s*難|没什么难|没有什么难|沒什麼難", source))


def _looks_like_vi_question(translation: str) -> bool:
    normal = _normalise_vi(translation)
    return "?" in translation or any(word in normal.split() for word in ("à", "ư", "sao", "hả", "chứ", "không", "chưa", "ai", "gì", "nào", "đâu"))


def _normalise_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _strip_vi_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d")


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
