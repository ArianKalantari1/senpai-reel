"""
Phase 5 — OpenAI text embeddings.

Uses text-embedding-3-small (1536 dims, ~$0.02/1M tokens — essentially free).
Stores FLOAT[1536] in message_units.embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Tuple

import requests

from core.db import DEFAULT_CLIENT_ID, get_connection

logger = logging.getLogger(__name__)

_EMBED_COST_PER_TOKEN = 0.02 / 1_000_000
_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 50


@dataclass(frozen=True)
class _EmbeddingBatchResult:
    embeddings: List[List[float]]
    total_tokens: int
    total_cost: float
    model: str
    dimensions: int | None


def _resolved_model() -> str:
    if not isinstance(_MODEL, str) or not _MODEL.strip():
        raise ValueError(
            "Embedding model is not configured; refusing to write unattributed vectors."
        )
    return _MODEL.strip()


def embed_text(text: str, api_key: str) -> List[float]:
    """Embed a single text string, returns 1536-dim float list."""
    return embed_batch([text], api_key)[0]


def embed_batch(texts: List[str], api_key: str,
                dimensions: int | None = None) -> List[List[float]]:
    embeddings, _, _ = embed_batch_with_usage(texts, api_key, dimensions)
    return embeddings


def embed_batch_with_usage(texts: List[str], api_key: str,
                           dimensions: int | None = None) -> Tuple[List[List[float]], int, float]:
    result = _embed_batch_with_metadata(texts, api_key, dimensions)
    return result.embeddings, result.total_tokens, result.total_cost


def _embed_batch_with_metadata(
    texts: List[str],
    api_key: str,
    dimensions: int | None = None,
) -> _EmbeddingBatchResult:
    """
    Batch-embed up to _BATCH_SIZE texts in a single API call.
    Automatically splits larger lists into sub-batches.
    """
    if not texts:
        return _EmbeddingBatchResult([], 0, 0.0, "", dimensions)

    all_embeddings: List[List[float]] = []
    total_tokens = 0
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = _resolved_model()

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            # text-embedding-3-small returns 1536 dimensions unless asked for
            # fewer. The stored column dictates the width, so the caller passes
            # it rather than the model deciding and the write failing per row.
            json=({"model": model, "input": batch}
                  | ({"dimensions": dimensions} if dimensions else {})),
            timeout=60,
        )
        if resp.status_code == 401:
            raise PermissionError("Invalid OpenAI API key")
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend([d["embedding"] for d in data["data"]])
        total_tokens += int(data.get("usage", {}).get("total_tokens") or 0)

    total_cost = round(total_tokens * _EMBED_COST_PER_TOKEN, 8)
    return _EmbeddingBatchResult(
        all_embeddings,
        total_tokens,
        total_cost,
        model,
        dimensions,
    )


def embed_pending_units(
    api_key: str,
    batch_size: int = 50,
    progress_callback=None,
    client_id: str = DEFAULT_CLIENT_ID,
    dimensions: int | None = None,
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
        batch_result = _embed_batch_with_metadata(
            texts, api_key, dimensions)
    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        return {"done": 0, "failed": total, "total": total, "error": str(e)}

    embeddings = batch_result.embeddings
    conn = get_connection()
    done = failed = 0
    unit_cost = round(batch_result.total_cost / len(embeddings), 8) if embeddings else 0.0
    try:
        for i, (unit_id, embedding) in enumerate(zip(unit_ids, embeddings)):
            try:
                conn.execute(
                    """
                    UPDATE message_units
                    SET embedding = ?,
                        embedded_at = CURRENT_TIMESTAMP,
                        embedding_cost_usd = ?,
                        embedding_model = ?
                    WHERE unit_id = ?
                    """,
                    [embedding, unit_cost, batch_result.model, unit_id],
                )
                done += 1
            except Exception as e:
                failed += 1
                logger.warning("Failed to save embedding for %s: %s", unit_id, e)
            if progress_callback:
                progress_callback(i + 1, total, unit_id)
    finally:
        conn.close()

    return {
        "done": done,
        "failed": failed,
        "total": total,
        "total_tokens": batch_result.total_tokens,
        "total_cost_usd": round(unit_cost * done, 8),
        "embedding_model": batch_result.model if done else None,
        "embedding_dimensions": (
            batch_result.dimensions
            if batch_result.dimensions is not None
            else (len(embeddings[0]) if done else None)
        ),
    }
