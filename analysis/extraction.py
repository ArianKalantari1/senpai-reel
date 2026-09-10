"""
Phase 4 — Knowledge extraction from transcripts.

Uses GPT-4o-mini to extract structured "message units" from reel transcripts.
Each unit is a single concrete tip, warning, stat, or other insight.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection
from analysis.taxonomy import (
    TOPICS,
    CONTENT_TYPES,
    topics_for_prompt,
    content_types_for_prompt,
    validate_topic,
    validate_content_type,
)

logger = logging.getLogger(__name__)

# GPT-4o-mini pricing: $0.15/1M input + $0.60/1M output tokens (as of 2025)
_GPT_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
_GPT_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

_SYSTEM_PROMPT = f"""You are an expert analyst specialising in Australian job market content on Instagram.

Your task: extract structured "message units" from Instagram reel transcripts.

A message unit is ONE concrete, self-contained idea. Extract 1-8 units per transcript.

For each unit, return a JSON object with these fields:
  - text:         The key sentence or claim (verbatim or closely paraphrased from transcript)
  - claim:        The core assertion in one sentence (your words)
  - advice:       Actionable instruction if any (null if not applicable)
  - topic:        ONE of the following topics:
{topics_for_prompt()}
  - subtopic:     One-word refinement (e.g. "keywords", "salary_negotiation") or null
  - content_type: ONE of the following types:
{content_types_for_prompt()}
  - confidence:   Your extraction confidence 0.0–1.0

Return a JSON object: {{"units": [...]}}

Rules:
- Only extract concrete, specific insights — not vague generalities
- Skip filler, small talk, and generic CTAs ("follow me", "like this video")
- Keep text under 200 characters
- confidence < 0.5 means you're unsure — still include but flag
"""


@dataclass
class MessageUnit:
    unit_id: str
    post_id: str
    text: str
    claim: str
    advice: Optional[str]
    topic: str
    subtopic: Optional[str]
    content_type: str
    confidence: float
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    extracted_at: Optional[datetime] = None
    model: str = "gpt-4o-mini"
    client_id: str = DEFAULT_CLIENT_ID


def extract_message_units(
    transcript_text: str,
    post_id: str,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> tuple[List[MessageUnit], float]:
    """
    Extract message units from a transcript via GPT.

    Returns:
        (units, cost_usd)
    """
    if not transcript_text or len(transcript_text.strip()) < 30:
        return [], 0.0

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript_text[:4000]}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 2000,
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
    usage = data.get("usage", {})
    cost = (
        usage.get("prompt_tokens", 0) * _GPT_INPUT_COST_PER_TOKEN
        + usage.get("completion_tokens", 0) * _GPT_OUTPUT_COST_PER_TOKEN
    )

    raw_content = data["choices"][0]["message"]["content"]
    parsed = json.loads(raw_content)
    raw_units = parsed.get("units", [])

    units = []
    now = datetime.utcnow()
    for ru in raw_units:
        try:
            unit = MessageUnit(
                unit_id=str(uuid.uuid4()),
                post_id=post_id,
                text=str(ru.get("text", ""))[:300],
                claim=str(ru.get("claim", ""))[:300],
                advice=ru.get("advice") or None,
                topic=validate_topic(ru.get("topic", "General")),
                subtopic=ru.get("subtopic") or None,
                content_type=validate_content_type(ru.get("content_type", "other")),
                confidence=float(ru.get("confidence", 0.5)),
                extracted_at=now,
                model=model,
            )
            units.append(unit)
        except Exception as e:
            logger.warning("Skipping malformed unit: %s — %s", ru, e)

    return units, round(cost, 6)


def save_message_units(units: List[MessageUnit], client_id: str = DEFAULT_CLIENT_ID):
    """Persist extracted message units to DB."""
    if not units:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, advice, topic, subtopic,
                content_type, confidence, extracted_at, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (unit_id) DO NOTHING
            """,
            [
                (
                    u.unit_id,
                    getattr(u, "client_id", None) or client_id,
                    u.post_id,
                    u.text,
                    u.claim,
                    u.advice,
                    u.topic,
                    u.subtopic,
                    u.content_type, u.confidence, u.extracted_at, u.model,
                )
                for u in units
            ],
        )
    finally:
        conn.close()
