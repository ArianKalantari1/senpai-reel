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
    HOOKS_SYSTEM, HOOKS_USER, HOOKS_REMIX_USER,
    SCRIPT_SYSTEM, SCRIPT_USER,
    IDEAS_SYSTEM, IDEAS_USER,
    format_hooks, format_ideas, format_remix_hooks,
)
from analysis.search import get_hook_examples, get_topic_ideas

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
    openai_api_key: str = "",
) -> GeneratedContent:
    """
    Generate a caption. Auto-fetches hook examples + idea landscape from corpus.
    No manual unit selection required.
    """
    hooks = get_hook_examples(topic)
    ideas = get_topic_ideas(topic)
    system = CAPTION_SYSTEM.format(tone=tone)
    user_msg = CAPTION_USER.format(
        angle=angle or topic,
        topic=topic,
        hook_examples=format_hooks(hooks),
        idea_landscape=format_ideas(ideas),
    )
    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=600)

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
    _save(result, hooks + ideas)
    return result


def generate_hooks(
    topic: str,
    angle: str,
    count: int = 5,
    remix_examples: Optional[list] = None,
    openai_api_key: str = "",
) -> GeneratedContent:
    """
    Generate opening hooks.
    - Normal mode: auto-fetches hook examples + ideas from corpus.
    - Remix mode: remix_examples is a list of user-selected corpus hooks to draw
      structural inspiration from while delivering the angle.
    """
    system = HOOKS_SYSTEM.format(count=count)

    if remix_examples:
        # Remix mode — the user picked specific hooks they like the structure of
        user_msg = HOOKS_REMIX_USER.format(
            angle=angle or topic,
            topic=topic,
            count=count,
            remix_hooks=format_remix_hooks(remix_examples),
        )
        source_units = remix_examples
    else:
        hooks = get_hook_examples(topic)
        ideas = get_topic_ideas(topic)
        user_msg = HOOKS_USER.format(
            angle=angle or topic,
            topic=topic,
            hook_examples=format_hooks(hooks),
            idea_landscape=format_ideas(ideas),
            count=count,
        )
        source_units = hooks + ideas

    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=500)

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
    _save(result, source_units)
    return result


def generate_script(
    topic: str,
    duration_sec: int,
    tone: str,
    angle: str = "",
    openai_api_key: str = "",
) -> GeneratedContent:
    """
    Generate a reel script. Auto-fetches hook examples + ideas from corpus.
    """
    word_count = int(duration_sec / 60 * 130)
    hooks = get_hook_examples(topic)
    ideas = get_topic_ideas(topic)
    system = SCRIPT_SYSTEM.format(duration_sec=duration_sec, word_count=word_count, tone=tone)
    user_msg = SCRIPT_USER.format(
        topic=topic,
        angle=angle or topic,
        duration_sec=duration_sec,
        tone=tone,
        hook_examples=format_hooks(hooks),
        idea_landscape=format_ideas(ideas),
    )
    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=900)

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
    _save(result, hooks + ideas)
    return result


def generate_ideas(
    topic: str,
    n: int = 5,
    openai_api_key: str = "",
) -> GeneratedContent:
    """
    Recommend fresh content ideas based on what's circulating in the corpus.
    No angle needed — the LLM proposes the angles.
    """
    ideas = get_topic_ideas(topic, limit=20)
    system = IDEAS_SYSTEM.format(n=n)
    user_msg = IDEAS_USER.format(
        topic=topic,
        idea_landscape=format_ideas(ideas),
        n=n,
    )
    text, tokens, cost = _call_gpt(system, user_msg, openai_api_key, max_tokens=900)

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="ideas",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
    )
    _save(result, ideas)
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
