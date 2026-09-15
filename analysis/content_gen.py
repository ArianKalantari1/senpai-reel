"""
Phase 7 — Content generation via GPT-4o.

Uses extracted message units as grounding context for all generated content.
All outputs are saved to the generated_content table with cost tracking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection
from analysis.unit_role import ROLES_EXCLUDED_FROM_GENERATION
from analysis.originality import (
    ReferenceLeakError,
    find_overlap,
)
from analysis.prompts import (
    CAPTION_SYSTEM, CAPTION_USER,
    HOOKS_SYSTEM, HOOKS_USER,
    SCRIPT_SYSTEM, SCRIPT_USER,
    format_reference_context,
)

_GPT4O_INPUT_COST = 2.50 / 1_000_000   # $2.50 / 1M input tokens
_GPT4O_OUTPUT_COST = 10.00 / 1_000_000  # $10.00 / 1M output tokens

# Use gpt-4o-mini for lower cost; generation quality is still excellent here
_MODEL = "gpt-4o-mini"
_MINI_INPUT_COST = 0.15 / 1_000_000
_MINI_OUTPUT_COST = 0.60 / 1_000_000


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
    client_id: str = DEFAULT_CLIENT_ID
    generation_run_id: Optional[str] = None
    generation_condition: Optional[str] = None


def _with_client_context(user_msg: str, client_context: Optional[dict]) -> str:
    if not client_context:
        return user_msg

    lines = []
    for label, key in [
        ("Client", "name"),
        ("Niche", "niche"),
        ("Target audience", "target_audience"),
        ("Brand voice", "brand_voice_notes"),
        ("Notes", "notes"),
    ]:
        value = (client_context.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")

    if not lines:
        return user_msg
    return f"{user_msg}\n\nClient context:\n" + "\n".join(lines)


def _reference_units_for_generation(reference_units: list) -> list:
    """Drop units that are research evidence, not generation source material.

    `hasattr`, not `getattr(unit, "unit_role", None)`. The default erases the
    difference between two very different units:

      - role present and None  -> ALLOW. Deliberate: every legacy row is NULL,
        and excluding NULL starves generation. See analysis/unit_role.py.
      - no `unit_role` attribute at all -> RAISE. A caller passing a shape that
        cannot carry a role is a programming error, not a unit to wave through.

    Collapsing those two was a real defect: `SearchResult` had no `unit_role`
    field, so `getattr` returned None for every unit search produced, nothing
    was ever filtered, and the whole suite stayed green. Absent is not zero,
    and it is not "safe" either.
    """
    kept = []
    for unit in reference_units:
        if not hasattr(unit, "unit_role"):
            raise TypeError(
                f"{type(unit).__name__} has no `unit_role`, so the generation "
                "role firewall cannot be applied to it. Add the field rather "
                "than passing the unit through unchecked."
            )
        role = (unit.unit_role or "").strip().lower()
        if role not in ROLES_EXCLUDED_FROM_GENERATION:
            kept.append(unit)
    return kept


def _call_gpt(
    system: str,
    user: str,
    api_key: str,
    model: str = _MODEL,
    stream: bool = False,
    max_tokens: int = 1000,
) -> tuple[str, int, float]:
    """
    Call GPT and return (output_text, tokens_used, cost_usd).
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code == 401:
        raise PermissionError("Invalid OpenAI API key")
    resp.raise_for_status()

    data = resp.json()
    usage = data["usage"]
    in_tokens = usage.get("prompt_tokens", 0)
    out_tokens = usage.get("completion_tokens", 0)
    cost = in_tokens * _MINI_INPUT_COST + out_tokens * _MINI_OUTPUT_COST
    text = data["choices"][0]["message"]["content"].strip()
    return text, in_tokens + out_tokens, round(cost, 6)


def _call_gpt_original(
    system: str,
    user: str,
    api_key: str,
    reference_units: list,
    max_tokens: int = 1000,
) -> tuple[str, int, float]:
    """Call GPT and reject output that reuses competitor wording.

    One retry with the offending phrases named explicitly, then fail closed.
    Returning leaked copy silently is worse than returning an error: the
    client publishes it under their own name and nobody finds out.

    See creative-director-ai #17.
    """
    text, tokens, cost = _call_gpt(system, user, api_key, max_tokens=max_tokens)
    overlap = find_overlap(text, reference_units)
    if not overlap:
        return text, tokens, cost

    banned = "\n".join(f"- {p}" for p in overlap[:10])
    retry_user = (
        f"{user}\n\n"
        "Your previous attempt reused wording from the reference material. "
        "Rewrite it completely. These exact phrases must not appear:\n"
        f"{banned}"
    )
    text2, tokens2, cost2 = _call_gpt(system, retry_user, api_key, max_tokens=max_tokens)

    total_tokens = tokens + tokens2
    total_cost = round(cost + cost2, 6)

    overlap2 = find_overlap(text2, reference_units)
    if overlap2:
        raise ReferenceLeakError(overlap2)
    return text2, total_tokens, total_cost


def generate_caption(
    topic: str,
    angle: str,
    tone: str,
    reference_units: list,
    openai_api_key: str,
    client_id: str = DEFAULT_CLIENT_ID,
    client_context: Optional[dict] = None,
    generation_run_id: Optional[str] = None,
    generation_condition: Optional[str] = None,
) -> GeneratedContent:
    reference_units = _reference_units_for_generation(reference_units)
    context = format_reference_context(reference_units)
    user_msg = CAPTION_USER.format(topic=topic, tone=tone, angle=angle, reference_context=context)
    user_msg = _with_client_context(user_msg, client_context)
    text, tokens, cost = _call_gpt_original(
        CAPTION_SYSTEM, user_msg, openai_api_key, reference_units, max_tokens=600
    )

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="caption",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
        client_id=client_id,
        generation_run_id=generation_run_id,
        generation_condition=generation_condition,
    )
    _save(result, reference_units, client_id)
    return result


def generate_hooks(
    topic: str,
    angle: str,
    reference_units: list,
    openai_api_key: str,
    count: int = 5,
    client_id: str = DEFAULT_CLIENT_ID,
    client_context: Optional[dict] = None,
    generation_run_id: Optional[str] = None,
    generation_condition: Optional[str] = None,
) -> GeneratedContent:
    reference_units = _reference_units_for_generation(reference_units)
    system = HOOKS_SYSTEM.format(count=count)
    context = format_reference_context(reference_units)
    user_msg = HOOKS_USER.format(topic=topic, angle=angle, reference_context=context, count=count)
    user_msg = _with_client_context(user_msg, client_context)
    text, tokens, cost = _call_gpt_original(
        system, user_msg, openai_api_key, reference_units, max_tokens=400
    )

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="hooks",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
        client_id=client_id,
        generation_run_id=generation_run_id,
        generation_condition=generation_condition,
    )
    _save(result, reference_units, client_id)
    return result


def generate_script(
    topic: str,
    duration_sec: int,
    tone: str,
    reference_units: list,
    openai_api_key: str,
    client_id: str = DEFAULT_CLIENT_ID,
    client_context: Optional[dict] = None,
    generation_run_id: Optional[str] = None,
    generation_condition: Optional[str] = None,
) -> GeneratedContent:
    reference_units = _reference_units_for_generation(reference_units)
    word_count = int(duration_sec / 60 * 130)
    system = SCRIPT_SYSTEM.format(duration_sec=duration_sec, word_count=word_count, tone=tone)
    context = format_reference_context(reference_units)
    user_msg = SCRIPT_USER.format(
        topic=topic, duration_sec=duration_sec, tone=tone, reference_context=context
    )
    user_msg = _with_client_context(user_msg, client_context)
    text, tokens, cost = _call_gpt_original(
        system, user_msg, openai_api_key, reference_units, max_tokens=800
    )

    result = GeneratedContent(
        gen_id=str(uuid.uuid4()),
        content_type="script",
        topic=topic,
        output_text=text,
        model=_MODEL,
        tokens_used=tokens,
        cost_usd=cost,
        created_at=datetime.utcnow(),
        client_id=client_id,
        generation_run_id=generation_run_id,
        generation_condition=generation_condition,
    )
    _save(result, reference_units, client_id)
    return result


def _save(
    content: GeneratedContent,
    reference_units: list,
    client_id: str = DEFAULT_CLIENT_ID,
):
    """Persist generated content to DB."""
    source_ids = [getattr(u, "unit_id", "") for u in reference_units]
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO generated_content
                (gen_id, client_id, created_at, topic, content_type, output_text, model,
                 source_units, tokens_used, cost_usd, generation_run_id, generation_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                content.gen_id, client_id, content.created_at, content.topic, content.content_type,
                content.output_text, content.model, source_ids,
                content.tokens_used, content.cost_usd, content.generation_run_id,
                content.generation_condition,
            ],
        )
    finally:
        conn.close()
