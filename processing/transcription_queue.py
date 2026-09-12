"""
Phase 3 — Transcription queue.

Processes all posts that have audio but no transcript yet.
Designed to be run incrementally — persists after each transcription
so partial runs don't lose work.

Phase 10 update: when provider == 'assemblyai', delegates to the streaming
pipeline (pipeline.stream_runner) for concurrent, overlapping STT → extract →
embed execution.  Legacy 'deepgram' / 'whisper' paths remain unchanged.
"""

import logging
from typing import Callable, Optional

from core.db import get_connection
from processing.transcribe import transcribe_post

logger = logging.getLogger(__name__)


def get_transcription_stats() -> dict:
    """Return counts useful for UI progress display."""
    conn = get_connection()
    try:
        total_with_audio = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE local_audio_path IS NOT NULL AND local_audio_path != ''"
        ).fetchone()[0]
        transcribed = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM transcripts").fetchone()[0]
        pending = total_with_audio - transcribed
        return {
            "total_with_audio": total_with_audio,
            "transcribed": transcribed,
            "pending": max(pending, 0),
            "total_cost_usd": round(total_cost, 4),
        }
    finally:
        conn.close()


def run_transcription_queue(
    api_key: str,
    provider: str = "deepgram",
    batch_size: int = 10,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Transcribe posts with audio but no transcript yet.

    Args:
        api_key:           API key — AssemblyAI key when provider='assemblyai',
                           Deepgram key when provider='deepgram', ignored for 'whisper'
        provider:          'assemblyai' | 'deepgram' | 'whisper'
        batch_size:        Max posts to process per run
        progress_callback: Optional callable(done, total, post_id, error)

    Returns:
        dict: done, failed, skipped, total_cost_usd
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.post_id,
               COALESCE(
                   NULLIF(p.local_audio_path, ''),
                   NULLIF(p.audio_url, ''),
                   p.video_url
               ) AS audio_source
        FROM posts p
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE (
              (p.local_audio_path IS NOT NULL AND p.local_audio_path != '')
           OR (p.audio_url       IS NOT NULL AND p.audio_url       != '')
           OR (p.video_url       IS NOT NULL AND p.video_url       != '')
        )
          AND t.post_id IS NULL
        LIMIT ?
        """,
        [batch_size],
    ).fetchall()
    conn.close()

    total = len(rows)

    # --- Phase 10: AssemblyAI path — streaming pipeline ---
    if provider == "assemblyai":
        from providers.stt import AssemblyAIProvider
        from providers.factory import get_llm, get_embeddings
        from pipeline.stream_runner import run_stream_pipeline

        stt = AssemblyAIProvider(api_key=api_key)
        llm = get_llm()
        emb = get_embeddings()

        stats = run_stream_pipeline(
            post_rows=rows,
            stt_provider=stt,
            llm_provider=llm,
            embed_provider=emb,
            progress_callback=progress_callback,
        )
        return {
            "done": stats.stt_done,
            "failed": stats.stt_failed,
            "total": total,
            "total_cost_usd": round(stats.total_cost_usd, 4),
            "errors": stats.errors,
        }

    # --- Legacy path: Deepgram / Whisper ---
    done = failed = 0
    total_cost = 0.0
    errors = []

    for i, (post_id, audio_path) in enumerate(rows):
        try:
            result = transcribe_post(post_id, api_key, provider)
            done += 1
            total_cost += result.cost_usd
            if progress_callback:
                progress_callback(i + 1, total, post_id, None)
        except Exception as e:
            failed += 1
            err_msg = str(e)
            errors.append({"post_id": post_id, "error": err_msg})
            logger.warning("Transcription failed for %s: %s", post_id, err_msg)
            if progress_callback:
                progress_callback(i + 1, total, post_id, err_msg)

    return {
        "done": done,
        "failed": failed,
        "total": total,
        "total_cost_usd": round(total_cost, 4),
        "errors": errors,
    }
