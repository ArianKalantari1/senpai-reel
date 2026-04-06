"""
Phase 7 / Phase 10 — Content generation.

Phase 10: swapped from GPT-4o-mini (OpenAI) to Groq LLaMA 3.3 70B.
`openai_api_key` parameter is still accepted but ignored — Groq reads its
key from providers.factory.get_generation_llm().
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.db import get_connection
from analysis.prompts import (
    CAPTION_SYSTEM, CAPTION_USER,
    HOOKS_SYSTEM, HOOKS_USER,
    SCRIPT_SYSTEM, SCRIPT_USER,
    format_reference_context,
)

_MODEL = "llama-3.3-70b-versatile"   # Groq model name


@dataclass
class GeneratedContent:
    gen_id: str
    content_type: str   # caption | hooks | script
    topic: str
    output_text: str
    model: str
    tokens_used: int
    cost_usd: float
    created_at: datetime


def _call_gpt(
    system: str,
    user: str,
    api_key: str = "",
    model: str = _MODEL,
    stream: bool = False,
    max_tokens: int = 1000,
) -> tuple[str, int, float]:
    """
    Generate text via Groq LLaMA 3.3 70B.
    `api_key` is ignored — key is read from providers.factory.
    Returns (output_text, tokens_used, cost_usd).
    """
    from providers.factory import get_generation_llm
    gen_llm = get_generation_llm()
    text, tokens, cost = gen_llm.generate(system, user, max_tokens=max_tokens)
    return text, tokens, cost


def generate_caption(
    topic: str,
    angle: str,
    tone: str,
    reference_units: list,
    openai_api_key: str,
) -> GeneratedContent:
    context = format_reference_context(reference_units)
    user_msg = CAPTION_USER.format(topic=topic, tone=tone, angle=angle, reference_context=context)
    text, tokens, cost = _call_gpt(CAPTION_SYSTEM, user_msg, openai_api_key, max_tokens=600)

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="caption",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
    )
    _save(result, reference_units)
    return result


def generate_hooks(
    topic: str,
    angle: str,
    reference_units: list,
    openai_api_key: str,
    count: int = 5,
) -> GeneratedContent:
    system = HOOKS_SYSTEM.format(count=count)
    context = format_reference_context(reference_units)
    user_msg = HOOKS_USER.format(topic=topic, angle=angle, reference_context=context, count=count)
    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=400)

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="hooks",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
    )
    _save(result, reference_units)
    return result


def generate_script(
    topic: str,
    duration_sec: int,
    tone: str,
    reference_units: list,
    openai_api_key: str,
) -> GeneratedContent:
    word_count = int(duration_sec / 60 * 130)
    system = SCRIPT_SYSTEM.format(duration_sec=duration_sec, word_count=word_count, tone=tone)
    context = format_reference_context(reference_units)
    user_msg = SCRIPT_USER.format(
        topic=topic, duration_sec=duration_sec, tone=tone, reference_context=context
    )
    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=800)

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="script",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
    )
    _save(result, reference_units)
    return result


def _save(content: GeneratedContent, reference_units: list):
    """Persist generated content to DB."""
    source_ids = [getattr(u, "unit_id", "") for u in reference_units]
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO generated_content
                (gen_id, created_at, topic, content_type, output_text, model,
                 source_units, tokens_used, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                content.gen_id, content.created_at, content.topic, content.content_type,
                content.output_text, content.model, source_ids,
                content.tokens_used, content.cost_usd,
            ],
        )
    finally:
        conn.close()
