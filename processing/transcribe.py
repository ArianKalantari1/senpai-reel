"""
Phase 3 — Deepgram transcription engine.

Provides:
  - DeepgramTranscriber   — primary, Nova-2 with word-level timestamps
  - WhisperTranscriber    — fallback via OpenAI Whisper-1
  - TranscriptResult      — shared dataclass
  - transcribe_post()     — convenience wrapper that persists to DB
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection

logger = logging.getLogger(__name__)

# Deepgram pricing: $0.0058 / minute for Nova-2 (pre-recorded)
_DEEPGRAM_COST_PER_MIN = 0.0058


@dataclass
class WordTimestamp:
    word: str
    start_sec: float
    end_sec: float
    confidence: float


@dataclass
class TranscriptResult:
    post_id: str
    provider: str
    model: str
    transcript: str
    language: str
    confidence: float
    duration_sec: float
    words: List[WordTimestamp] = field(default_factory=list)
    cost_usd: float = 0.0
    raw_response: dict = field(default_factory=dict)
    client_id: str = DEFAULT_CLIENT_ID

    @property
    def word_count(self) -> int:
        return len(self.transcript.split())


class DeepgramTranscriber:
    """Transcribe audio using Deepgram Nova-2."""

    API_URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")
        self._headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        }

    def transcribe(self, audio_path: str, post_id: str) -> TranscriptResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        params = {
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "false",
            "words": "true",
            "language": "en-AU",
        }

        with open(path, "rb") as f:
            audio_bytes = f.read()

        resp = requests.post(
            self.API_URL,
            headers=self._headers,
            params=params,
            data=audio_bytes,
            timeout=120,
        )

        if resp.status_code == 401:
            raise PermissionError("Invalid Deepgram API key")
        resp.raise_for_status()

        raw = resp.json()
        channel = raw["results"]["channels"][0]["alternatives"][0]
        transcript_text = channel.get("transcript", "")
        confidence = channel.get("confidence", 0.0)
        duration = raw["metadata"].get("duration", 0.0)
        cost = (duration / 60.0) * _DEEPGRAM_COST_PER_MIN

        words = [
            WordTimestamp(
                word=w["word"],
                start_sec=w["start"],
                end_sec=w["end"],
                confidence=w.get("confidence", 0.0),
            )
            for w in channel.get("words", [])
        ]

        return TranscriptResult(
            post_id=post_id,
            provider="deepgram",
            model="nova-2",
            transcript=transcript_text,
            language="en-AU",
            confidence=confidence,
            duration_sec=duration,
            words=words,
            cost_usd=round(cost, 6),
            raw_response=raw,
        )


class WhisperTranscriber:
    """Fallback: OpenAI Whisper-1."""

    API_URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._api_key = api_key

    def transcribe(self, audio_path: str, post_id: str) -> TranscriptResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        headers = {"Authorization": f"Bearer {self._api_key}"}
        with open(path, "rb") as f:
            resp = requests.post(
                self.API_URL,
                headers=headers,
                files={"file": (path.name, f, "audio/wav")},
                data={"model": "whisper-1", "language": "en"},
                timeout=120,
            )
        resp.raise_for_status()
        raw = resp.json()

        # Whisper returns size in bytes; cost = $0.006 per minute (≈ 1 byte ≈ 0.0625ms for 16kHz mono)
        file_size = path.stat().st_size
        duration_sec = file_size / (16000 * 2)  # 16kHz, 16-bit
        cost = (duration_sec / 60.0) * 0.006

        return TranscriptResult(
            post_id=post_id,
            provider="openai",
            model="whisper-1",
            transcript=raw.get("text", ""),
            language="en",
            confidence=0.0,
            duration_sec=round(duration_sec, 1),
            words=[],  # Whisper-1 doesn't return word timestamps
            cost_usd=round(cost, 6),
            raw_response=raw,
        )


def save_transcript(result: TranscriptResult):
    """Persist a TranscriptResult to the transcripts + transcript_words tables."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO transcripts (
                post_id, client_id, provider, model, transcript, language,
                confidence, duration_sec, word_count, transcribed_at, cost_usd, raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (post_id) DO UPDATE SET
                client_id      = COALESCE(transcripts.client_id, excluded.client_id),
                transcript     = excluded.transcript,
                provider       = excluded.provider,
                model          = excluded.model,
                confidence     = excluded.confidence,
                duration_sec   = excluded.duration_sec,
                word_count     = excluded.word_count,
                transcribed_at = excluded.transcribed_at,
                cost_usd       = excluded.cost_usd,
                raw_response   = excluded.raw_response
            """,
            [
                result.post_id,
                result.client_id,
                result.provider,
                result.model,
                result.transcript,
                result.language,
                result.confidence,
                result.duration_sec,
                result.word_count,
                datetime.utcnow(),
                result.cost_usd,
                json.dumps(result.raw_response),
            ],
        )

        if result.words:
            conn.execute("DELETE FROM transcript_words WHERE post_id = ?", [result.post_id])
            conn.executemany(
                """
                INSERT INTO transcript_words (
                    client_id, post_id, word_index, word, start_sec, end_sec, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.client_id,
                        result.post_id,
                        i,
                        w.word,
                        w.start_sec,
                        w.end_sec,
                        w.confidence,
                    )
                    for i, w in enumerate(result.words)
                ],
            )
    finally:
        conn.close()


def transcribe_post(
    post_id: str,
    api_key: str,
    provider: str = "deepgram",
    client_id: str = DEFAULT_CLIENT_ID,
) -> TranscriptResult:
    """
    Fetch audio path from DB, transcribe, and save result.

    Args:
        post_id:  The post to transcribe
        api_key:  API key for the chosen provider
        provider: 'deepgram' (default) or 'whisper'

    Returns:
        TranscriptResult
    """
    conn = get_connection()
    row = conn.execute(
        """
        SELECT p.local_audio_path
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        WHERE p.post_id = ? AND cp.client_id = ?
        """,
        [post_id, client_id],
    ).fetchone()
    conn.close()

    if not row or not row[0]:
        raise ValueError(f"No audio file for post_id {post_id} — run audio extraction first")

    audio_path = row[0]
    transcriber = (
        DeepgramTranscriber(api_key) if provider == "deepgram" else WhisperTranscriber(api_key)
    )
    result = transcriber.transcribe(audio_path, post_id)
    result.client_id = client_id
    save_transcript(result)
    return result
