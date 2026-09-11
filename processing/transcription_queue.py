"""
Phase 3 — Transcription queue.

Processes all posts that have audio but no transcript yet.
Designed to be run incrementally — persists after each transcription
so partial runs don't lose work.
"""

import logging
from typing import Callable, Optional

from core.db import DEFAULT_CLIENT_ID, get_connection
from processing.concurrency import io_worker_count, run_bounded, run_db_write
from processing.transcribe import save_transcript, transcribe_audio_file

logger = logging.getLogger(__name__)


def get_transcription_stats(client_id: str = DEFAULT_CLIENT_ID) -> dict:
    """Return counts useful for UI progress display."""
    conn = get_connection()
    try:
        total_with_audio = conn.execute(
            """
            SELECT COUNT(*)
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            WHERE cp.client_id = ?
              AND p.local_audio_path IS NOT NULL
              AND p.local_audio_path != ''
            """,
            [client_id],
        ).fetchone()[0]
        transcribed = conn.execute(
            """
            SELECT COUNT(*)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        total_cost = conn.execute(
            """
            SELECT COALESCE(SUM(t.cost_usd), 0)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
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
    client_id: str = DEFAULT_CLIENT_ID,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Transcribe posts with audio but no transcript yet.

    Args:
        api_key:           API key (Deepgram or OpenAI depending on provider)
        provider:          'deepgram' or 'whisper'
        batch_size:        Max posts to process per run
        progress_callback: Optional callable(done, total, post_id, error)

    Returns:
        dict: done, failed, skipped, total_cost_usd
    """
    if not api_key:
        key_name = "DEEPGRAM_API_KEY" if provider == "deepgram" else "OPENAI_API_KEY"
        return {
            "done": 0,
            "failed": 0,
            "total": 0,
            "total_cost_usd": 0.0,
            "errors": [
                {
                    "post_id": None,
                    "error": f"Transcription is not configured. Add `{key_name}` in Settings.",
                }
            ],
        }

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.post_id, p.local_audio_path
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE cp.client_id = ?
          AND p.local_audio_path IS NOT NULL
          AND p.local_audio_path != ''
          AND t.post_id IS NULL
        LIMIT ?
        """,
        [client_id, batch_size],
    ).fetchall()
    conn.close()

    total = len(rows)
    done = failed = 0
    total_cost = 0.0
    errors = []

    def _transcribe(row):
        post_id, audio_path = row
        result = transcribe_audio_file(audio_path, post_id, api_key, provider, client_id)
        run_db_write(save_transcript, result)
        return result

    def _progress(count, count_total, row, result, error):
        if progress_callback:
            progress_callback(count, count_total, row[0], str(error) if error else None)

    completed = run_bounded(
        rows,
        _transcribe,
        max_workers=io_worker_count(max_workers),
        progress_callback=_progress,
    )

    for item in completed:
        post_id = item.item[0]
        if item.error:
            failed += 1
            err_msg = str(item.error)
            errors.append({"post_id": post_id, "error": err_msg})
            logger.warning("Transcription failed for %s: %s", post_id, err_msg)
        else:
            done += 1
            total_cost += item.result.cost_usd

    return {
        "done": done,
        "failed": failed,
        "total": total,
        "total_cost_usd": round(total_cost, 4),
        "errors": errors,
    }
