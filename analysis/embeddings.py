"""
Phase 5 / Phase 10 — Text embeddings.

Phase 10: swapped from OpenAI text-embedding-3-small (1536-dim) to
Voyage voyage-3-lite (512-dim) via providers.VoyageProvider.

Legacy `api_key` parameter is accepted for backwards compatibility but
ignored — VoyageProvider reads its key from providers.factory.
"""

from __future__ import annotations

import logging
from typing import List

from core.db import get_connection

logger = logging.getLogger(__name__)


def _get_provider():
    from providers.factory import get_embeddings
    return get_embeddings()


def embed_text(text: str, api_key: str = "") -> List[float]:
    """Embed a single text string. Returns 512-dim float list (Phase 10)."""
    return _get_provider().embed(text)


def embed_batch(texts: List[str], api_key: str = "") -> List[List[float]]:
    """
    Batch-embed texts via Voyage voyage-3-lite.
    Returns a list of 512-dim float vectors.
    """
    return _get_provider().embed_batch(texts)


def embed_pending_units(
    api_key: str = "",
    batch_size: int = 128,
    progress_callback=None,
) -> dict:
    """
    Embed all message_units that don't have an embedding yet.

    Returns:
        dict with done, failed, total counts
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT unit_id, text FROM message_units
        WHERE embedding IS NULL
        LIMIT ?
        """,
        [batch_size],
    ).fetchall()
    conn.close()

    if not rows:
        return {"done": 0, "failed": 0, "total": 0}

    unit_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    total = len(rows)

    try:
        embeddings = embed_batch(texts)
    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        return {"done": 0, "failed": total, "total": total, "error": str(e)}

    conn = get_connection()
    done = failed = 0
    try:
        for i, (unit_id, embedding) in enumerate(zip(unit_ids, embeddings)):
            try:
                conn.execute(
                    "UPDATE message_units SET embedding = ?, embedded_at = CURRENT_TIMESTAMP WHERE unit_id = ?",
                    [embedding, unit_id],
                )
                done += 1
            except Exception as e:
                failed += 1
                logger.warning("Failed to save embedding for %s: %s", unit_id, e)
            if progress_callback:
                progress_callback(i + 1, total, unit_id)
    finally:
        conn.close()

    return {"done": done, "failed": failed, "total": total}
