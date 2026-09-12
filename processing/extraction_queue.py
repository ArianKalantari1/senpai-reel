"""
Phase 4 — Extraction queue.

Processes all transcripts that don't have message_units extracted yet.
"""

import logging
from typing import Callable, Optional

from core.db import DEFAULT_CLIENT_ID, get_connection
from analysis.extraction import extract_message_units, save_message_units
from processing.concurrency import io_worker_count, run_bounded, run_db_write

logger = logging.getLogger(__name__)


def get_extraction_stats(client_id: str = DEFAULT_CLIENT_ID) -> dict:
    conn = get_connection()
    try:
        total_transcripts = conn.execute(
            """
            SELECT COUNT(*)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        extracted_posts = conn.execute(
            """
            SELECT COUNT(DISTINCT mu.post_id)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ? AND mu.client_id = ?
            """,
            [client_id, client_id],
        ).fetchone()[0]
        total_units = conn.execute(
            """
            SELECT COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ? AND mu.client_id = ?
            """,
            [client_id, client_id],
        ).fetchone()[0]
        total_cost = conn.execute(
            """
            SELECT COALESCE(SUM(t.extraction_cost_usd), 0)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        return {
            "total_transcripts": total_transcripts,
            "extracted_posts": extracted_posts,
            "pending": max(total_transcripts - extracted_posts, 0),
            "total_units": total_units,
            "total_cost_usd": round(total_cost, 4),
        }
    except Exception:
        return {"total_transcripts": 0, "extracted_posts": 0, "pending": 0,
                "total_units": 0, "total_cost_usd": 0.0}
    finally:
        conn.close()


def run_extraction_queue(
    openai_api_key: str,
    batch_size: int = 10,
    progress_callback: Optional[Callable] = None,
    client_id: str = DEFAULT_CLIENT_ID,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Extract message units from transcripts not yet processed.

    Returns:
        dict with done, failed, total, total_units_extracted, total_cost_usd
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT t.post_id, t.transcript
        FROM transcripts t
        JOIN client_posts cp ON t.post_id = cp.post_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM message_units mu
            WHERE mu.post_id = t.post_id AND mu.client_id = ?
        )
          AND cp.client_id = ?
        LIMIT ?
        """,
        [client_id, client_id, batch_size],
    ).fetchall()
    conn.close()

    total = len(rows)
    done = failed = 0
    total_units = 0
    total_cost = 0.0

    def _extract(row):
        post_id, transcript = row
        units, cost = extract_message_units(
            transcript, post_id, openai_api_key, client_id=client_id
        )
        for unit in units:
            unit.client_id = client_id
        run_db_write(_save_extraction_result, units, client_id, post_id, cost)
        return {"units": len(units), "cost": cost}

    def _progress(count, count_total, row, result, error):
        unit_count = result["units"] if result else 0
        if progress_callback:
            progress_callback(count, count_total, row[0], str(error) if error else None, unit_count)

    completed = run_bounded(
        rows,
        _extract,
        max_workers=io_worker_count(max_workers),
        progress_callback=_progress,
    )

    for item in completed:
        post_id = item.item[0]
        if item.error:
            failed += 1
            logger.warning("Extraction failed for %s: %s", post_id, item.error)
        else:
            done += 1
            total_units += item.result["units"]
            total_cost += item.result["cost"]

    return {
        "done": done,
        "failed": failed,
        "total": total,
        "total_units_extracted": total_units,
        "total_cost_usd": round(total_cost, 6),
    }


def _update_extraction_cost(post_id: str, cost: float):
    try:
        conn = get_connection()
        conn.execute(
            "UPDATE transcripts SET extraction_cost_usd = ? WHERE post_id = ?",
            [cost, post_id],
        )
        conn.close()
    except Exception:
        pass  # column might not exist yet — migration will add it


def _save_extraction_result(units, client_id: str, post_id: str, cost: float):
    save_message_units(units, client_id)
    # Track cost in transcripts table
    _update_extraction_cost(post_id, cost)
