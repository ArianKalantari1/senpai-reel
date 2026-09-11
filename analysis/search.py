"""
Phase 5 — Semantic search over message units.

Uses DuckDB's list_cosine_similarity() for vector search —
no external vector DB required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.db import get_connection
from analysis.embeddings import embed_text


@dataclass
class SearchResult:
    unit_id: str
    post_id: str
    username: str
    topic: str
    content_type: str
    text: str
    claim: str
    score: float
    video_url: Optional[str] = None
    posted_at: Optional[str] = None


def _require_client_id(client_id: Optional[str]) -> str:
    if not client_id:
        raise ValueError("client_id is required")
    return client_id


def semantic_search(
    query: str,
    openai_api_key: str,
    client_id: str,
    topic_filter: Optional[str] = None,
    content_type_filter: Optional[str] = None,
    top_k: int = 20,
) -> List[SearchResult]:
    """
    Search message_units by semantic similarity.

    Args:
        query:                Natural language search query
        openai_api_key:       For embedding the query
        client_id:            Active client scope
        topic_filter:         Optional topic to restrict search (from this client's active taxonomy)
        content_type_filter:  Optional content_type to restrict search
        top_k:                Number of results to return

    Returns:
        List of SearchResult sorted by cosine similarity descending
    """
    client_id = _require_client_id(client_id)
    query_vec = embed_text(query, openai_api_key)

    conn = get_connection()
    try:
        where_clauses = ["mu.embedding IS NOT NULL"]
        where_params = []

        if topic_filter and topic_filter != "All":
            where_clauses.append("mu.topic = ?")
            where_params.append(topic_filter)

        if content_type_filter and content_type_filter != "All":
            where_clauses.append("mu.content_type = ?")
            where_params.append(content_type_filter)

        where_clauses.append(
            "EXISTS (SELECT 1 FROM client_posts cp WHERE cp.post_id = mu.post_id AND cp.client_id = ?)"
        )
        where_params.append(client_id)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Parameter order must match the SQL left-to-right:
        # 1. query_vec  → list_cosine_similarity(?, ...)  in SELECT
        # 2. where_params → WHERE mu.topic = ?, etc.
        # 3. top_k      → LIMIT ?
        params = [query_vec] + where_params + [top_k]

        rows = conn.execute(
            f"""
            SELECT
                mu.unit_id,
                mu.post_id,
                COALESCE(ca.username, 'unknown') AS username,
                mu.topic,
                mu.content_type,
                mu.text,
                mu.claim,
                list_cosine_similarity(mu.embedding, ?::FLOAT[1536]) AS score,
                p.video_url,
                CAST(p.posted_at AS TEXT) AS posted_at
            FROM message_units mu
            LEFT JOIN posts p ON mu.post_id = p.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            {where_sql}
            ORDER BY score DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            SearchResult(
                unit_id=r[0],
                post_id=r[1],
                username=r[2],
                topic=r[3],
                content_type=r[4],
                text=r[5],
                claim=r[6],
                score=float(r[7]) if r[7] is not None else 0.0,
                video_url=r[8],
                posted_at=r[9],
            )
            for r in rows
        ]
    finally:
        conn.close()


def keyword_search(
    keyword: str,
    client_id: str,
    topic_filter: Optional[str] = None,
    top_k: int = 50,
    content_type_filter: Optional[str] = None,
) -> List[SearchResult]:
    """Fast keyword search (no embedding needed) — fallback when no API key."""
    client_id = _require_client_id(client_id)
    conn = get_connection()
    try:
        where_clauses = ["(LOWER(mu.text) LIKE ? OR LOWER(mu.claim) LIKE ?)"]
        params = [f"%{keyword.lower()}%", f"%{keyword.lower()}%"]

        if topic_filter and topic_filter != "All":
            where_clauses.append("mu.topic = ?")
            params.append(topic_filter)

        if content_type_filter and content_type_filter != "All":
            where_clauses.append("mu.content_type = ?")
            params.append(content_type_filter)

        where_clauses.append(
            "EXISTS (SELECT 1 FROM client_posts cp WHERE cp.post_id = mu.post_id AND cp.client_id = ?)"
        )
        params.append(client_id)

        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.append(top_k)

        rows = conn.execute(
            f"""
            SELECT
                mu.unit_id, mu.post_id,
                COALESCE(ca.username, 'unknown') AS username,
                mu.topic, mu.content_type, mu.text, mu.claim,
                mu.confidence AS score,
                p.video_url,
                CAST(p.posted_at AS TEXT) AS posted_at
            FROM message_units mu
            LEFT JOIN posts p ON mu.post_id = p.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            {where_sql}
            ORDER BY mu.confidence DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            SearchResult(
                unit_id=r[0], post_id=r[1], username=r[2],
                topic=r[3], content_type=r[4], text=r[5], claim=r[6],
                score=float(r[7]) if r[7] else 0.0, video_url=r[8], posted_at=r[9],
            )
            for r in rows
        ]
    finally:
        conn.close()
