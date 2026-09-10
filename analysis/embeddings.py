"""
Phase 5 — OpenAI text embeddings.

Uses text-embedding-3-small (1536 dims, ~$0.02/1M tokens — essentially free).
Stores FLOAT[1536] in message_units.embedding.
"""

from __future__ import annotations

import logging
from typing import List

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection

logger = logging.getLogger(__name__)

_EMBED_COST_PER_TOKEN = 0.02 / 1_000_000
_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 50


def embed_text(text: str, api_key: str) -> List[float]:
    """Embed a single text string, returns 1536-dim float list."""
    return embed_batch([text], api_key)[0]


def embed_batch(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Batch-embed up to _BATCH_SIZE texts in a single API call.
    Automatically splits larger lists into sub-batches.
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={"model": _MODEL, "input": batch},
            timeout=60,
        )
        if resp.status_code == 401:
            raise PermissionError("Invalid OpenAI API key")
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend([d["embedding"] for d in data["data"]])

    return all_embeddings


def embed_pending_units(
    api_key: str,
    batch_size: int = 50,
    progress_callback=None,
    client_id: str = DEFAULT_CLIENT_ID,
) -> dict:
    """
    Embed all message_units that don't have an embedding yet.

    Returns:
        dict with done, failed, total counts
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT mu.unit_id, mu.text
        FROM message_units mu
        JOIN client_posts cp ON mu.post_id = cp.post_id
        WHERE cp.client_id = ?
          AND mu.embedding IS NULL
        LIMIT ?
        """,
        [client_id, batch_size],
    ).fetchall()
    conn.close()

    if not rows:
        return {"done": 0, "failed": 0, "total": 0}

    unit_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    total = len(rows)

    try:
        embeddings = embed_batch(texts, api_key)
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
