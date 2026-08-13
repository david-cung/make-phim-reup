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

# ISO-639-1 → human name. Only entries here get pretty-printed in the
# prompt; everything else falls back to the code itself.
_LANG_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese",
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
    before = json.dumps(list(context_before), ensure_ascii=False, indent=2)
    now = json.dumps(list(chunk), ensure_ascii=False, indent=2)
    after = json.dumps(list(context_after), ensure_ascii=False, indent=2)
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
    before = json.dumps(list(context_before), ensure_ascii=False, indent=2)
    now = json.dumps(list(chunk), ensure_ascii=False, indent=2)
    after = json.dumps(list(context_after), ensure_ascii=False, indent=2)
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


def render_chunk_messages(
    *,
    prompt_version: str,
    source_lang: str,
    target_lang: str,
    context_before: Iterable[dict],
    chunk: Iterable[dict],
    context_after: Iterable[dict],
    hint: Optional[str] = None,
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
    return render_chunk_messages_v1(
        source_lang=source_lang,
        target_lang=target_lang,
        context_before=context_before,
        chunk=chunk,
        context_after=context_after,
        hint=hint,
    )


def prompt_versions() -> list[str]:
    return [TRANSLATION_PROMPT_V1, TRANSLATION_PROMPT_V2]


def is_known_version(name: str) -> bool:
    return name in prompt_versions()
