"""
LLM providers.

CerebrasProvider   — primary for extraction (LLaMA 3.3 70B @ ~2,100 tok/sec, 1,000 req/day free)
GroqLLMProvider    — used for Content Studio generation (14,400 req/day free)

Both use OpenAI-compatible chat completions format.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _with_backoff(fn, max_retries: int = 5, base_delay: float = 15.0):
    """Call fn(), retrying on 429 / queue_exceeded with exponential backoff."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            is_rate = (
                "429" in msg
                or "too_many_requests" in msg.lower()
                or "queue_exceeded" in msg.lower()
                or "rate_limit" in msg.lower()
                or "high traffic" in msg.lower()
            )
            if is_rate and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — waiting %.0fs: %s",
                    attempt + 1, max_retries, wait, msg[:120],
                )
                time.sleep(wait)
            else:
                raise

# Cerebras pricing (after free tier)
_CEREBRAS_INPUT_COST  = 0.85 / 1_000_000   # $0.85/1M input tokens
_CEREBRAS_OUTPUT_COST = 1.20 / 1_000_000   # $1.20/1M output tokens

# Groq LLM pricing (after free tier)
_GROQ_INPUT_COST  = 0.59 / 1_000_000
_GROQ_OUTPUT_COST = 0.79 / 1_000_000


class CerebrasProvider:
    """
    LLM extraction via Cerebras Qwen-3 235B.

    Speed:     ~2,100 tokens/sec (wafer-scale hardware)
    Free tier: 1,000 req/day, 30 req/min
    Signup:    cerebras.ai (no credit card)

    Install: pip install cerebras-cloud-sdk
    """

    def __init__(self, api_key: str, model: str = "qwen-3-235b-a22b-instruct-2507"):
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY is required")
        from cerebras.cloud.sdk import Cerebras
        self._client = Cerebras(api_key=api_key)
        self._model = model

    def extract(
        self,
        transcript_text: str,
        post_id: str,
        ocr_text: str = "",
        system_prompt: Optional[str] = None,
    ) -> Tuple[list, float]:
        """
        Extract message units from transcript text.

        Returns:
            (List[MessageUnit], cost_usd)  — same contract as analysis.extraction.extract_message_units()
        """
        from analysis.extraction import _SYSTEM_PROMPT, MessageUnit
        from analysis.taxonomy import validate_topic, validate_content_type

        if not transcript_text or len(transcript_text.strip()) < 30:
            return [], 0.0

        prompt = system_prompt or _SYSTEM_PROMPT
        user_content = f"Transcript:\n{transcript_text[:4000]}"
        if ocr_text:
            user_content += f"\n\nOn-screen text (from video frames):\n{ocr_text[:800]}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content},
        ]

        response = _with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000,
            )
        )

        usage = response.usage
        cost = (
            (usage.prompt_tokens or 0) * _CEREBRAS_INPUT_COST
            + (usage.completion_tokens or 0) * _CEREBRAS_OUTPUT_COST
        )

        raw_content = response.choices[0].message.content

        # Qwen3 / reasoning models prepend <think>…</think> blocks before the JSON.
        # Strip them before parsing so json.loads doesn't fail.
        clean = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        if not clean:
            clean = raw_content  # nothing was stripped — use original

        parsed = None
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Last resort: find the outermost JSON object in the text
            m = re.search(r'\{.*\}', clean, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        if parsed is None:
            logger.warning("Cerebras returned non-JSON for %s — raw: %.200s", post_id, raw_content)
            return [], round(cost, 6)

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
                    model=self._model,
                )
                units.append(unit)
            except Exception as e:
                logger.warning("Skipping malformed unit: %s — %s", ru, e)

        return units, round(cost, 6)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> Tuple[str, int, float]:
        """
        Generate text (captions, hooks, scripts).

        Returns:
            (output_text, tokens_used, cost_usd)
        """
        response = _with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
        )
        usage = response.usage
        tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        cost = (
            (usage.prompt_tokens or 0) * _CEREBRAS_INPUT_COST
            + (usage.completion_tokens or 0) * _CEREBRAS_OUTPUT_COST
        )
        return response.choices[0].message.content.strip(), tokens, round(cost, 6)


class GroqLLMProvider:
    """
    LLM generation via Groq LLaMA 3.3 70B.

    Speed:     ~200 tokens/sec (fast enough for Content Studio generation)
    Free tier: 14,400 req/day (LLM tier is less restricted than Whisper audio)
    Signup:    console.groq.com

    Used for: Content Studio (captions, hooks, scripts)
    NOT used for extraction critical path — Cerebras is faster there.

    Install: pip install groq
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = model

    def extract(
        self,
        transcript_text: str,
        post_id: str,
        ocr_text: str = "",
        system_prompt: Optional[str] = None,
    ) -> Tuple[list, float]:
        """Extraction via Groq — fallback if Cerebras is unavailable."""
        from analysis.extraction import _SYSTEM_PROMPT, MessageUnit
        from analysis.taxonomy import validate_topic, validate_content_type

        if not transcript_text or len(transcript_text.strip()) < 30:
            return [], 0.0

        prompt = system_prompt or _SYSTEM_PROMPT
        user_content = f"Transcript:\n{transcript_text[:4000]}"
        if ocr_text:
            user_content += f"\n\nOn-screen text (from video frames):\n{ocr_text[:800]}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_content},
        ]

        response = _with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000,
            )
        )

        usage = response.usage
        cost = (
            (usage.prompt_tokens or 0) * _GROQ_INPUT_COST
            + (usage.completion_tokens or 0) * _GROQ_OUTPUT_COST
        )

        raw_content = response.choices[0].message.content
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            return [], round(cost, 6)

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
                    model=self._model,
                )
                units.append(unit)
            except Exception as e:
                logger.warning("Skipping malformed unit: %s — %s", ru, e)

        return units, round(cost, 6)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> Tuple[str, int, float]:
        """
        Generate text for Content Studio.

        Returns:
            (output_text, tokens_used, cost_usd)
        """
        response = _with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
        )
        usage = response.usage
        tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        cost = (
            (usage.prompt_tokens or 0) * _GROQ_INPUT_COST
            + (usage.completion_tokens or 0) * _GROQ_OUTPUT_COST
        )
        return response.choices[0].message.content.strip(), tokens, round(cost, 6)
