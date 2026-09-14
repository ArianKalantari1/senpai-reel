"""
Phase 4 — Knowledge extraction from transcripts.

Uses GPT-4o-mini to extract structured "message units" from reel transcripts.
Each unit is a single concrete tip, warning, stat, or other insight.
"""

from __future__ import annotations

import json
import logging
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection
from analysis.unit_role import role_from_rules
from analysis.taxonomy import (
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

_SYSTEM_PROMPT_TEMPLATE = """You are an expert analyst specialising in Australian job market content on Instagram.

Your task: extract structured "message units" from Instagram reel transcripts.

A message unit is ONE concrete, self-contained idea. Extract 1-8 units per transcript.

For each unit, return a JSON object with these fields:
  - text:         The key sentence or claim (verbatim or closely paraphrased from transcript)
  - claim:        The core assertion in one sentence (your words)
  - advice:       Actionable instruction if any (null if not applicable)
  - topic:        ONE of the following topics:
{topics}
  - subtopic:     One-word refinement (e.g. "keywords", "salary_negotiation") or null
  - content_type: ONE of the following types:
{content_types}
  - confidence:   Your extraction confidence 0.0–1.0

Return a JSON object: {{"units": [...]}}

Rules:
- Only extract concrete, specific insights — not vague generalities
- Skip filler, small talk, and generic CTAs ("follow me", "like this video")
- Keep text under 200 characters
- confidence < 0.5 means you're unsure — still include but flag

SPEAKER STANCE — read this before extracting anything.

A transcript is not a list of things the creator believes. Creators quote bad
advice to mock it, act out scripted examples, read out a comment they
disagree with, and stage debates between opposing views. A sentence appearing
in the transcript is NOT evidence the creator asserts it.

Before extracting a unit, ask: **is the speaker asserting this, or performing
it?** Extract only what is asserted. If a sentence is performed rather than
asserted, either skip it or extract the creator's actual point instead.

The five cases that go wrong most often:

1. SATIRE AND MOCKERY. "Every LinkedIn post should start with 'It is with
   mixed emotions that I announce...'" is a joke about a cliché. The creator's
   assertion is that the cliché is tired — not that you should use it.

2. NEGATIVE EXAMPLES. Creators quote a bad message or a bad resume line to
   show what not to do. "I'd love to share some content ideas with you" shown
   as a bad networking DM is not a recommendation. If it is introduced as a
   mistake, extract it as a warning or skip it.

3. SCRIPTED DEMONSTRATIONS. Many videos act out a good answer to an interview
   question. The speaker in that script is a character, not the creator.
   "I'm currently looking for roles with more ownership over strategy" is a
   demo line — it is NOT a fact about the creator, and the unit must not say
   "the speaker is seeking...". Extract the technique being demonstrated, or
   skip.

4. DEBATE MONTAGES. Some videos cut between opposing views — "I tell my work
   friends everything" against "I'd never trust anyone at work" — without
   endorsing either. Do not turn one side into advice. Extract the tension
   itself, or skip.

5. SINGLE ANECDOTES. "An engineer I know left tech to start a company" is one
   story. Do not generalise it into "tech professionals are leaving stable
   jobs". Keep the scope the speaker actually gave it.

When stance is genuinely ambiguous, lower `confidence` rather than guessing.
A confident unit that inverts the speaker's meaning is worse than no unit:
downstream, nothing can tell it apart from a real finding.
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
    topic_id: Optional[str] = None
    taxonomy_id: Optional[str] = None
    # None means "not classified yet" — never default to "subject". See #27.
    unit_role: Optional[str] = None
    unit_role_source: Optional[str] = None
    unit_role_confidence: Optional[float] = None
    extraction_run_id: Optional[str] = None
    prompt_version: Optional[str] = None
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    extracted_at: Optional[datetime] = None
    model: str = "gpt-4o-mini"
    client_id: str = DEFAULT_CLIENT_ID


def build_system_prompt(client_id: str) -> str:
    """Build the extraction prompt against one client's topic vocabulary.

    Previously a module-level f-string, which baked the Jobs-AU topics into
    every extraction regardless of persona — so a second niche classified
    almost entirely as the catch-all. See creative-director-ai #25.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        topics=topics_for_prompt(client_id),
        content_types=content_types_for_prompt(),
    )


def prompt_version_for_text(system_prompt: str) -> str:
    """Short hash of the rendered extraction prompt."""
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def prompt_version_for_client(client_id: str = DEFAULT_CLIENT_ID) -> str:
    return prompt_version_for_text(build_system_prompt(client_id))


def extract_message_units(
    transcript_text: str,
    post_id: str,
    openai_api_key: str,
    client_id: str = DEFAULT_CLIENT_ID,
    model: str = "gpt-4o-mini",
    extraction_run_id: Optional[str] = None,
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

    system_prompt = build_system_prompt(client_id)
    prompt_version = prompt_version_for_text(system_prompt)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
            _topic_id, _topic_name = validate_topic(client_id, ru.get("topic", ""))
            _content_type = validate_content_type(ru.get("content_type", "other"))
            # Rules first. What they cannot decide stays NULL for the backfill
            # tool to classify, rather than being guessed at here.
            _role = role_from_rules(
                text=str(ru.get("text", "")),
                claim=str(ru.get("claim", "")),
                content_type=_content_type,
            )
            unit = MessageUnit(
                unit_id=str(uuid.uuid4()),
                post_id=post_id,
                text=str(ru.get("text", ""))[:300],
                claim=str(ru.get("claim", ""))[:300],
                advice=ru.get("advice") or None,
                topic=_topic_name,
                topic_id=_topic_id,
                subtopic=ru.get("subtopic") or None,
                content_type=_content_type,
                confidence=float(ru.get("confidence", 0.5)),
                unit_role=_role,
                unit_role_source="rule" if _role else None,
                extraction_run_id=extraction_run_id,
                prompt_version=prompt_version,
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

    # Pin this batch to the taxonomy version in force now, so a later version
    # can be added alongside rather than silently reinterpreting these rows.
    from core.taxonomy import get_active_taxonomy
    _active = get_active_taxonomy(client_id)
    _taxonomy_id = _active["taxonomy_id"] if _active else None

    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, advice, topic, topic_id,
                taxonomy_id, subtopic, content_type, confidence, extracted_at, model,
                unit_role, unit_role_source, unit_role_confidence,
                extraction_run_id, prompt_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    u.topic_id,
                    getattr(u, "taxonomy_id", None) or _taxonomy_id,
                    u.subtopic,
                    u.content_type, u.confidence, u.extracted_at, u.model,
                    getattr(u, "unit_role", None),
                    getattr(u, "unit_role_source", None),
                    getattr(u, "unit_role_confidence", None),
                    getattr(u, "extraction_run_id", None),
                    getattr(u, "prompt_version", None),
                )
                for u in units
            ],
        )
    finally:
        conn.close()
