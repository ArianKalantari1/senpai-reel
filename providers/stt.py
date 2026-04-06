"""
STT providers.

AssemblyAIProvider — primary (100 hr/month free, Universal-2 model)
Deepgram / Whisper paths remain in processing/transcribe.py for rollback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from processing.transcribe import TranscriptResult, WordTimestamp

logger = logging.getLogger(__name__)

_ASSEMBLYAI_COST_PER_MIN = 0.0028  # Universal-2, as of 2026


class AssemblyAIProvider:
    """
    Transcription via AssemblyAI Universal-2.

    Free tier: 100 hours/month (~4,800 reels/month).
    Latency: ~5-8 seconds per reel (async submit + poll).

    Install: pip install assemblyai
    Signup:  assemblyai.com (no credit card for free tier)
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is required")
        import assemblyai as aai
        aai.settings.api_key = api_key
        self._aai = aai

    def transcribe(self, audio_path: str, post_id: str) -> TranscriptResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        config = self._aai.TranscriptionConfig(
            language_code="en",
            punctuate=True,
            format_text=True,
            word_boost=["ATS", "resume", "LinkedIn", "STAR", "KPI"],  # domain hints
        )
        transcriber = self._aai.Transcriber(config=config)
        transcript = transcriber.transcribe(str(path))

        if transcript.status == self._aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI error for {post_id}: {transcript.error}")

        duration_sec = (transcript.audio_duration or 0.0)
        cost = (duration_sec / 60.0) * _ASSEMBLYAI_COST_PER_MIN

        words = []
        if transcript.words:
            for i, w in enumerate(transcript.words):
                words.append(WordTimestamp(
                    word=w.text,
                    start_sec=w.start / 1000.0,  # AssemblyAI returns milliseconds
                    end_sec=w.end / 1000.0,
                    confidence=w.confidence or 0.0,
                ))

        return TranscriptResult(
            post_id=post_id,
            provider="assemblyai",
            model="universal-2",
            transcript=transcript.text or "",
            language="en",
            confidence=transcript.confidence or 0.0,
            duration_sec=round(duration_sec, 1),
            words=words,
            cost_usd=round(cost, 6),
            raw_response={},  # skip large raw blob
        )
