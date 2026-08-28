"""Versioned translation prompts.

Prompts are kept out of the UI and out of the provider so we can:

* audit the exact instructions the model saw for any given output
  (``prompt_version`` is baked into the cache key);
* iterate on the wording without touching provider code;
* bump the version safely — old cached translations stay valid until
  the user chooses to regenerate.

Adding a new prompt is a code change plus a version bump; the on-disk
translation.json records which version produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

TRANSLATION_PROMPT_V1 = "translation_prompt_v1"
TRANSLATION_PROMPT_V2 = "translation_prompt_v2"
TRANSLATION_PROMPT_V3 = "translation_prompt_v3"
TRANSLATION_PROMPT_V4 = "translation_prompt_v4"
TRANSLATION_PROMPT_V5 = "translation_prompt_v5"

# ISO-639-1 → human name. Only entries here get pretty-printed in the
# prompt; everything else falls back to the code itself.
_LANG_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "yue": "Cantonese Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
}


def language_name(code: str) -> str:
    return _LANG_NAMES.get((code or "").lower(), code or "unknown")


@dataclass(frozen=True)
class PromptMessage:
    role: str  # "system" | "user"
    content: str


def system_prompt_v1(source_lang: str, target_lang: str) -> str:
    """The immutable system prompt for translation_prompt_v1.

    Kept as an unindented block so token counts are stable across
    minor whitespace changes.
    """
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    return (
        f"You are a professional subtitle translator translating movie "
        f"dialogue from {src} into {tgt}.\n\n"
        "RULES (must follow):\n"
        f"- Translate ONLY the segments in the CHUNK, into natural, idiomatic {tgt}.\n"
        "- Preserve the speaker's meaning, tone, register, and intent.\n"
        "- Keep character names, place names, and important terminology intact.\n"
        "- Match the level of formality and politeness of the source.\n"
        "- Preserve profanity/vulgarity level when present.\n"
        "- Avoid overly literal translation; produce dialogue that sounds native.\n"
        "- Keep subtitles concise — one or two short lines per segment.\n"
        "- Do NOT invent dialogue, do NOT omit meaning, do NOT add explanations.\n"
        "- Do NOT include translator notes or bracketed asides.\n"
        "- Do NOT modify the segment id, start, end, or the CONTEXT segments.\n"
        "- Use the PREVIOUS and NEXT context ONLY to understand the scene; "
        "do not return those segments in your output.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a single JSON object of the exact shape:\n"
        '{ "segments": [ { "id": <int>, "translation": "<string>" }, ... ] }\n'
        "One object per CHUNK segment, in the same order. No Markdown, "
        "no code fences, no prose before or after the JSON."
    )


def user_prompt_v1(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
) -> str:
    """Render the per-chunk user message.

    Each ``dict`` should have ``id`` and ``text`` keys. Context segments
    are shown to the model but must not appear in the output.
    """
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    before = json.dumps(
        list(context_before), ensure_ascii=False, separators=(",", ":")
    )
    now = json.dumps(list(chunk), ensure_ascii=False, separators=(",", ":"))
    after = json.dumps(
        list(context_after), ensure_ascii=False, separators=(",", ":")
    )
    hint_line = ""
    if hint:
        hint_line = f"\nTRANSLATOR HINT (do not translate this hint):\n{hint}\n"
    return (
        f"Translate from {src} into {tgt}.\n"
        f"{hint_line}\n"
        "PREVIOUS CONTEXT (do NOT translate; scene lead-in only):\n"
        f"{before}\n\n"
        "CHUNK TO TRANSLATE (translate every segment below):\n"
        f"{now}\n\n"
        "NEXT CONTEXT (do NOT translate; scene follow-up only):\n"
        f"{after}\n\n"
        f'Return JSON: {{ "segments": [ {{"id": ..., "translation": "..."}} ] }} '
        f"for the CHUNK only, translated into {tgt}."
    )


def render_chunk_messages_v1(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
) -> list[PromptMessage]:
    return [
        PromptMessage("system", system_prompt_v1(source_lang, target_lang)),
        PromptMessage(
            "user",
            user_prompt_v1(
                source_lang=source_lang,
                target_lang=target_lang,
                context_before=context_before,
                chunk=chunk,
                context_after=context_after,
                hint=hint,
            ),
        ),
    ]


def system_prompt_v2(source_lang: str, target_lang: str) -> str:
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    spoken = "Vietnamese movie dialogue" if (target_lang or "").lower() == "vi" else f"spoken {tgt}"
    return (
        f"You are a professional movie dialogue translator working from {src} into {tgt}.\n\n"
        "PRIORITY ORDER:\n"
        "1. Meaning of the CURRENT line\n"
        "2. Natural spoken " + spoken + "\n"
        "3. Character tone and register\n"
        "4. Surrounding PREVIOUS / NEXT lines\n"
        "5. Conciseness that still fits a subtitle\n\n"
        "RULES:\n"
        "- Translate ONLY the CURRENT / CHUNK lines. Use PREVIOUS and NEXT only as context.\n"
        "- Sound like people talking in a film, not a document.\n"
        "- Prefer short, idiomatic spoken lines over literal word-by-word rendering.\n"
        "- Keep names, places, and important terms.\n"
        "- Match formality, intimacy, and profanity of the source.\n"
        "- Do not add explanations, honorifics, or translator notes.\n"
        "- Do not make every line overly formal Vietnamese (avoid newspaper style).\n"
        "- Keep each line short enough to speak in the same time as the original.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a single JSON object:\n"
        '{ "segments": [ { "id": <int>, "translation": "<spoken subtitle>" } ] }\n'
        "No Markdown, no code fences, no extra prose."
    )


def system_prompt_v3(source_lang: str, target_lang: str) -> str:
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    spoken = "Vietnamese movie dialogue" if (target_lang or "").lower() == "vi" else f"spoken {tgt}"
    extra = ""
    if (target_lang or "").lower() == "vi":
        extra = (
            "\nVIETNAMESE TARGET CHECKS:\n"
            "- Every translation MUST be Vietnamese written with Latin Vietnamese characters.\n"
            "- Do NOT leave Chinese/Japanese/Korean characters in the translation.\n"
            "- Translate names phonetically or keep proper names only when they are names.\n"
            "- If unsure, still produce a natural Vietnamese line instead of copying the source.\n"
            "- Vietnamese pronouns MUST follow explicit pronounContext relationshipRule when present.\n"
            "- Treat speaker as the person saying the CURRENT line and addressees as people being addressed.\n"
            "- Do NOT reverse selfReference/addressTerm direction.\n"
            "- If speaker/addressee/relationship is unknown, prefer neutral wording, 'tôi', or omitted pronouns where natural; do not force anh/chị/em.\n"
        )
    return (
        f"You are a professional movie dialogue translator working from {src} into {tgt}.\n\n"
        "PRIORITY ORDER:\n"
        "1. Meaning of the CURRENT line\n"
        "2. Natural spoken " + spoken + "\n"
        "3. Character tone and register\n"
        "4. Surrounding PREVIOUS / NEXT lines\n"
        "5. Conciseness that still fits a subtitle\n\n"
        "RULES:\n"
        "- Translate ONLY the CURRENT / CHUNK lines. Use PREVIOUS and NEXT only as context.\n"
        "- Sound like people talking in a film, not a document.\n"
        "- Prefer short, idiomatic spoken lines over literal word-by-word rendering.\n"
        "- Keep names, places, and important terms.\n"
        "- Match formality, intimacy, and profanity of the source.\n"
        "- Do not add explanations, honorifics, or translator notes.\n"
        "- Never copy the source line as the translation unless the source is already in the target language.\n"
        "- Keep each line short enough to speak in the same time as the original.\n"
        f"{extra}\n"
        "OUTPUT FORMAT:\n"
        "Return a single JSON object:\n"
        '{ "translations": [ { "id": <int>, "translated_text": "<spoken subtitle>", "confidence": 0.0-1.0, "reason_flags": [] } ] }\n'
        "No Markdown, no code fences, no extra prose."
    )


def user_prompt_v2(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
) -> str:
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    before = json.dumps(
        list(context_before), ensure_ascii=False, separators=(",", ":")
    )
    now = json.dumps(list(chunk), ensure_ascii=False, separators=(",", ":"))
    after = json.dumps(
        list(context_after), ensure_ascii=False, separators=(",", ":")
    )
    hint_line = f"\nNOTE:\n{hint}\n" if hint else ""
    return (
        f"Translate movie dialogue from {src} into natural spoken {tgt}.\n"
        f"{hint_line}\n"
        "PREVIOUS:\n"
        f"{before}\n\n"
        "CURRENT (translate these):\n"
        f"{now}\n\n"
        "NEXT:\n"
        f"{after}\n\n"
        'Return JSON: { "segments": [ {"id": ..., "translation": "..."} ] }'
    )


def user_prompt_v3(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
    translation_memory: Optional[dict] = None,
) -> str:
    src = language_name(source_lang)
    tgt = language_name(target_lang)
    before = json.dumps(
        list(context_before), ensure_ascii=False, separators=(",", ":")
    )
    now = json.dumps(list(chunk), ensure_ascii=False, separators=(",", ":"))
    after = json.dumps(
        list(context_after), ensure_ascii=False, separators=(",", ":")
    )
    memory = json.dumps(
        translation_memory or {}, ensure_ascii=False, separators=(",", ":")
    )
    hint_line = f"\nNOTE:\n{hint}\n" if hint else ""
    guard = ""
    if (target_lang or "").lower() == "vi":
        guard = (
            "\nIMPORTANT: Output Vietnamese only. If any CURRENT line is Chinese, "
            "translate it to Vietnamese; do not copy Chinese characters into the answer.\n"
        )
    return (
        f"Translate movie dialogue from {src} into natural spoken {tgt}.\n"
        f"{guard}"
        f"{hint_line}\n"
        "Each row may include pronounContext with speaker, addressees, relationshipRule, and reviewFlags.\n"
        "Rows may include automaticPronounPlan with speaker, listener, relationship, self_pronoun, target_pronoun, confidence, enforce, and context_request.\n"
        "When automaticPronounPlan.enforce is true, treat self_pronoun and target_pronoun as strong constraints for the CURRENT speaker's line.\n"
        "When automaticPronounPlan.context_request is true, use broader dialogue context and avoid hard-coding a risky pronoun; omit pronouns naturally if needed.\n"
        "Rows may also include speakerId such as speaker_001/speaker_002. Use speakerId only as a stable dialogue turn label; do not infer real names, gender, or relationships from it.\n"
        "TRANSLATION_MEMORY may include movieSummary, currentScenes, characters, characterGraph, relationshipFact rows, addressPatterns, sceneRelationshipOverrides, pronounPlans, speakerCharacterMapping, knownNames, and previous translationMemory rows. Use it only to keep names, aliases, speaker continuity, scene meaning, and pronouns consistent.\n"
        "Character and relationship memory is uncertain unless confidence is high. Do not invent gender, relationships, or real character names that are not present in the memory or dialogue.\n"
        "Rows include compact sourceProtection. Treat it as hard constraints: preserve numbers, units, durations, quantities, negation, question/command status, actions, and unit order. Never turn hours into years or negation into affirmation.\n"
        "If sourceProtection.units has multiple items, translate them in order without changing their meaning. If sourceProtection.quality.confidence is low, stay conservative and do not invent corrected source text.\n"
        "Priority for Vietnamese pronouns: current source evidence, explicit relationshipRule, verified relationshipFact/source context, high-confidence automaticPronounPlan/addressPatterns, scene overrides, confirmed dialogue context, recentAddressHistory style hints, then model inference last. surfaceRealizationSuggestion and previous Vietnamese translations are not semantic evidence.\n"
        "Resolve first-person and second-person together as one speaker-listener social configuration. If listener or relationship is unknown, omit the Vietnamese pronoun when natural, or use neutral wording; do not guess anh/chị/em/chú/ông.\n"
        "For a relationshipRule, selfReference is what the CURRENT speaker calls themselves; addressTerm is what the CURRENT speaker calls the addressee.\n"
        "Never swap speaker and addressee pronouns.\n\n"
        "Use TRANSLATION_MEMORY to keep names, terms, scene context, and nearby pronoun patterns consistent. Do not translate memory rows again.\n\n"
        "TRANSLATION_MEMORY:\n"
        f"{memory}\n\n"
        "PREVIOUS:\n"
        f"{before}\n\n"
        "CURRENT (translate these):\n"
        f"{now}\n\n"
        "NEXT:\n"
        f"{after}\n\n"
        f'Return JSON in {tgt}: {{ "translations": [ {{"id": ..., "translated_text": "...", "confidence": 0.0, "reason_flags": []}} ] }}'
    )


def system_prompt_v4(source_lang: str, target_lang: str) -> str:
    return system_prompt_v3(source_lang, target_lang)


def user_prompt_v4(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
    translation_memory: Optional[dict] = None,
) -> str:
    retry_note = ""
    if hint:
        retry_note = (
            "\nRETRY_OR_VALIDATION_NOTE:\n"
            f"{hint}\n"
            "Re-evaluate the CURRENT dialogue using the expanded conversation context. "
            "Pay attention to omitted subjects, pronouns, forms of address, speaker intent, and who is being addressed.\n"
        )
    return user_prompt_v3(
        source_lang=source_lang,
        target_lang=target_lang,
        context_before=context_before,
        chunk=chunk,
        context_after=context_after,
        hint=retry_note or None,
        translation_memory=translation_memory,
    )


def system_prompt_v5(source_lang: str, target_lang: str) -> str:
    base = system_prompt_v3(source_lang, target_lang)
    if (target_lang or "").lower() != "vi":
        return base
    return (
        base
        + "\nPHASE 8 SEMANTIC REALIZATION:\n"
        "- Preserve source meaning before naturalness or brevity.\n"
        "- Treat relationship/address terms as semantic facts, not fixed Vietnamese words.\n"
        "- Resolve speaker, listener, referent, possession, direct address, and third-person reference before choosing Vietnamese pronouns.\n"
        "- Do not infer anh/chị/em/con/cháu/mẹ/bố/etc. without source or high-confidence memory evidence.\n"
        "- Unknown speaker-listener relationship: prefer neutral wording such as tôi or explicit kinship descriptions; do not invent intimacy, age, hierarchy, or family facts.\n"
        "- Relationship memory stores source facts and evidence. Previous Vietnamese translations are style hints only and must not override current source evidence.\n"
        "- For high-risk rows, you may reason internally over 2-3 candidates, but return only the best JSON translation.\n"
    )


def user_prompt_v5(
    *,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
    translation_memory: Optional[dict] = None,
) -> str:
    phase8_note = (
        "Rows may include semanticRepresentation. It is source meaning, not a Vietnamese word map. "
        "Use realizationGuidance to decide whether social pronouns are justified. "
        "If ambiguityScore is high, preserve referents/possession/relationship with conservative Vietnamese instead of guessing.\n"
        "Never learn new relationship facts from previous Vietnamese translations; use them only as weak style continuity.\n"
        "If characterGraph.addressPatterns shows a high-confidence pair, keep social consistency unless current source, scene, or discourse role justifies a shift. Consistency must never override explicit source meaning.\n"
    )
    combined_hint = phase8_note if not hint else f"{phase8_note}\n{hint}"
    return user_prompt_v4(
        source_lang=source_lang,
        target_lang=target_lang,
        context_before=context_before,
        chunk=chunk,
        context_after=context_after,
        hint=combined_hint,
        translation_memory=translation_memory,
    )


def render_chunk_messages(
    *,
    prompt_version: str,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
    translation_memory: Optional[dict] = None,
) -> list[PromptMessage]:
    if prompt_version == TRANSLATION_PROMPT_V2:
        return [
            PromptMessage("system", system_prompt_v2(source_lang, target_lang)),
            PromptMessage(
                "user",
                user_prompt_v2(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_before=context_before,
                    chunk=chunk,
                    context_after=context_after,
                    hint=hint,
                ),
            ),
        ]
    if prompt_version == TRANSLATION_PROMPT_V3:
        return [
            PromptMessage("system", system_prompt_v3(source_lang, target_lang)),
            PromptMessage(
                "user",
                user_prompt_v3(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_before=context_before,
                    chunk=chunk,
                    context_after=context_after,
                    hint=hint,
                    translation_memory=translation_memory,
                ),
            ),
        ]
    if prompt_version == TRANSLATION_PROMPT_V4:
        return [
            PromptMessage("system", system_prompt_v4(source_lang, target_lang)),
            PromptMessage(
                "user",
                user_prompt_v4(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_before=context_before,
                    chunk=chunk,
                    context_after=context_after,
                    hint=hint,
                    translation_memory=translation_memory,
                ),
            ),
        ]
    if prompt_version == TRANSLATION_PROMPT_V5:
        return [
            PromptMessage("system", system_prompt_v5(source_lang, target_lang)),
            PromptMessage(
                "user",
                user_prompt_v5(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_before=context_before,
                    chunk=chunk,
                    context_after=context_after,
                    hint=hint,
                    translation_memory=translation_memory,
                ),
            ),
        ]
    return render_chunk_messages_v1(
        source_lang=source_lang,
        target_lang=target_lang,
        context_before=context_before,
        chunk=chunk,
        context_after=context_after,
        hint=hint,
    )


def prompt_versions() -> list[str]:
    return [
        TRANSLATION_PROMPT_V1,
        TRANSLATION_PROMPT_V2,
        TRANSLATION_PROMPT_V3,
        TRANSLATION_PROMPT_V4,
        TRANSLATION_PROMPT_V5,
    ]


def is_known_version(name: str) -> bool:
    return name in prompt_versions()
