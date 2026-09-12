"""
Phase 3 — Deepgram transcription engine.

Provides:
  - DeepgramTranscriber   — primary, Nova-2 with word-level timestamps
  - WhisperTranscriber    — fallback via OpenAI Whisper-1
  - AssemblyAIProvider    — AssemblyAI Universal-2
  - TranscriptResult      — shared dataclass
  - transcribe_post()     — convenience wrapper that persists to DB
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional, Protocol, runtime_checkable
from datetime import datetime

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection
from core.secrets import get_secret
from processing.concurrency import run_db_write
from processing.media_archive import archive_video_if_ready

logger = logging.getLogger(__name__)

# Deepgram pricing: $0.0058 / minute for Nova-2 (pre-recorded)
_DEEPGRAM_COST_PER_MIN = 0.0058


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Provider seam for speech-to-text implementations."""

    name: ClassVar[str]
    secret_name: ClassVar[str]

    def transcribe(self, audio_path: str, post_id: str) -> "TranscriptResult":
        ...


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

    name = "deepgram"
    secret_name = "DEEPGRAM_API_KEY"
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

    name = "whisper"
    secret_name = "OPENAI_API_KEY"
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


class _AssemblyAIClient:
    """Small REST client for local-file AssemblyAI transcription."""

    BASE_URL = "https://api.assemblyai.com"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout_sec: int = 120,
        poll_interval_sec: float = 3.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = {"authorization": api_key}
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        speech_models: list[str],
        language_code: Optional[str],
        word_boost: list[str],
    ) -> dict:
        upload_url = self._upload(audio_path)
        payload = {
            "audio_url": upload_url,
            "speech_models": speech_models,
        }
        if language_code:
            payload["language_code"] = language_code
        if word_boost:
            payload["word_boost"] = word_boost

        submitted = self._post_json("/v2/transcript", payload)
        transcript_id = submitted["id"]
        return self._poll(transcript_id)

    def _upload(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self._base_url}/v2/upload",
                headers=self._headers,
                data=f,
                timeout=self._timeout_sec,
            )
        self._raise_for_status(resp)
        return resp.json()["upload_url"]

    def _post_json(self, path: str, payload: dict) -> dict:
        resp = requests.post(
            f"{self._base_url}{path}",
            headers={**self._headers, "content-type": "application/json"},
            json=payload,
            timeout=self._timeout_sec,
        )
        self._raise_for_status(resp)
        return resp.json()

    def _poll(self, transcript_id: str) -> dict:
        deadline = time.monotonic() + self._timeout_sec
        url = f"{self._base_url}/v2/transcript/{transcript_id}"
        while True:
            resp = requests.get(url, headers=self._headers, timeout=self._timeout_sec)
            self._raise_for_status(resp)
            payload = resp.json()
            status = payload.get("status")
            if status == "completed":
                return payload
            if status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {payload.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"AssemblyAI transcription timed out: {transcript_id}")
            time.sleep(self._poll_interval_sec)

    @staticmethod
    def _raise_for_status(resp):
        if resp.status_code == 401:
            raise PermissionError("Invalid AssemblyAI API key")
        resp.raise_for_status()


class AssemblyAIProvider:
    """Transcribe local audio files using AssemblyAI Universal-2."""

    name = "assemblyai"
    secret_name = "ASSEMBLYAI_API_KEY"
    MODEL = "universal-2"

    def __init__(
        self,
        api_key: str,
        *,
        client: Optional[_AssemblyAIClient] = None,
        language_code: str = "en_au",
        word_boost: Optional[list[str]] = None,
    ):
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is required")
        self._client = client or _AssemblyAIClient(api_key)
        self._language_code = language_code
        self._word_boost = list(word_boost or [])

    def transcribe(self, audio_path: str, post_id: str) -> TranscriptResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        raw = self._client.transcribe_file(
            path,
            speech_models=[self.MODEL],
            language_code=self._language_code,
            word_boost=self._word_boost,
        )

        words = [
            WordTimestamp(
                word=word.get("text") or word.get("word"),
                start_sec=_ms_to_sec(word.get("start")),
                end_sec=_ms_to_sec(word.get("end")),
                confidence=word.get("confidence"),
            )
            for word in _assemblyai_words(raw)
            if word.get("text") or word.get("word")
        ]

        return TranscriptResult(
            post_id=post_id,
            provider="assemblyai",
            model=raw.get("speech_model_used") or self.MODEL,
            transcript=raw.get("text") or "",
            language=raw.get("language_code"),
            confidence=raw.get("confidence"),
            duration_sec=raw.get("audio_duration"),
            words=words,
            cost_usd=None,
            raw_response=raw,
        )


def _ms_to_sec(value):
    if value is None:
        return None
    return value / 1000.0


def _assemblyai_words(raw: dict) -> list[dict]:
    words = raw.get("words") or []
    if words:
        return words

    utterance_words = []
    for utterance in raw.get("utterances") or []:
        utterance_words.extend(utterance.get("words") or [])
    return utterance_words


TRANSCRIPTION_PROVIDERS: dict[str, type[TranscriptionProvider]] = {
    DeepgramTranscriber.name: DeepgramTranscriber,
    WhisperTranscriber.name: WhisperTranscriber,
    AssemblyAIProvider.name: AssemblyAIProvider,
}


def _normalise_provider_name(provider: str) -> str:
    return str(provider or "").strip().lower()


def get_transcription_provider_class(provider: str) -> type[TranscriptionProvider]:
    provider_name = _normalise_provider_name(provider)
    try:
        return TRANSCRIPTION_PROVIDERS[provider_name]
    except KeyError as exc:
        available = ", ".join(sorted(TRANSCRIPTION_PROVIDERS))
        raise ValueError(
            f"Unknown transcription provider `{provider}`. Available providers: {available}."
        ) from exc


def resolve_transcription_api_key(
    provider: str,
    api_key: Optional[str] = None,
    st_module=None,
) -> str:
    """Resolve the API key for the requested provider, preserving explicit overrides."""
    provider_name = _normalise_provider_name(provider)
    provider_class = get_transcription_provider_class(provider_name)
    secret_name = provider_class.secret_name
    resolved = str(api_key).strip() if api_key is not None else get_secret(secret_name, st_module)
    if not resolved:
        raise ValueError(
            f"`{secret_name}` is required for transcription provider `{provider_name}`."
        )
    return resolved


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
    api_key: Optional[str] = None,
    provider: str = "deepgram",
    client_id: str = DEFAULT_CLIENT_ID,
) -> TranscriptResult:
    """
    Fetch audio path from DB, transcribe, and save result.

    Args:
        post_id:  The post to transcribe
        api_key:  Optional API key override for the chosen provider
        provider: 'deepgram' (default), 'whisper', or 'assemblyai'

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
    result = transcribe_audio_file(audio_path, post_id, api_key, provider, client_id)
    run_db_write(save_transcript, result)
    archive_video_if_ready(post_id, client_id=client_id)
    return result


def transcribe_audio_file(
    audio_path: str,
    post_id: str,
    api_key: Optional[str] = None,
    provider: str = "deepgram",
    client_id: str = DEFAULT_CLIENT_ID,
) -> TranscriptResult:
    """Transcribe an already-resolved audio path without reading the DB."""
    provider_name = _normalise_provider_name(provider)
    transcriber_class = get_transcription_provider_class(provider_name)
    resolved_api_key = resolve_transcription_api_key(provider_name, api_key)
    transcriber = transcriber_class(resolved_api_key)
    result = transcriber.transcribe(audio_path, post_id)
    result.client_id = client_id
    return result
