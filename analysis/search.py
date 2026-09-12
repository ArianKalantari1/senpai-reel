"""
Phase 5 / Phase 10 — Semantic search over message units.

Uses DuckDB's list_cosine_similarity() for vector search —
no external vector DB required.

Phase 10: embedding dimension changed from 1536 (OpenAI) to 512 (Voyage).
`openai_api_key` is accepted for backwards compatibility but ignored.
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


def semantic_search(
    query: str,
    openai_api_key: str = "",
    topic_filter: Optional[str] = None,
    content_type_filter: Optional[str] = None,
    top_k: int = 20,
) -> List[SearchResult]:
    """
    Search message_units by semantic similarity.

    Args:
        query:                Natural language search query
        openai_api_key:       Ignored (Phase 10 — Voyage reads key from factory)
        topic_filter:         Optional topic to restrict search
        content_type_filter:  Optional content_type to restrict search
        top_k:                Number of results to return

    Returns:
        List of SearchResult sorted by cosine similarity descending
    """
    query_vec = embed_text(query)

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
                list_cosine_similarity(mu.embedding, ?::FLOAT[512]) AS score,
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
    topic_filter: Optional[str] = None,
    top_k: int = 50,
) -> List[SearchResult]:
    """Fast keyword search (no embedding needed) — fallback when no API key."""
    conn = get_connection()
    try:
        where_clauses = ["(LOWER(mu.text) LIKE ? OR LOWER(mu.claim) LIKE ?)"]
        params = [f"%{keyword.lower()}%", f"%{keyword.lower()}%"]

        if topic_filter and topic_filter != "All":
            where_clauses.append("mu.topic = ?")
            params.append(topic_filter)

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


def get_hook_examples(topic: str = "All", limit: int = 8) -> List[SearchResult]:
    """
    Pull real hook lines from the corpus ordered by confidence.
    Falls back to all topics when fewer than 3 match the requested topic —
    so there's always something to show the LLM as creative reference.
    """
    conn = get_connection()
    try:
        def _fetch(apply_topic: bool) -> list:
            where = "WHERE mu.content_type = 'hook'"
            params: list = []
            if apply_topic and topic != "All":
                where += " AND mu.topic = ?"
                params.append(topic)
            params.append(limit)
            return conn.execute(
                f"""
                SELECT mu.unit_id, mu.post_id,
                       COALESCE(ca.username, 'unknown') AS username,
                       mu.topic, mu.content_type, mu.text, mu.claim,
                       mu.confidence AS score, p.video_url,
                       CAST(p.posted_at AS TEXT) AS posted_at
                FROM message_units mu
                LEFT JOIN posts p ON mu.post_id = p.post_id
                LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
                {where}
                ORDER BY mu.confidence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        rows = _fetch(apply_topic=True)
        if len(rows) < 3 and topic != "All":
            rows = _fetch(apply_topic=False)

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


def get_topic_ideas(topic: str = "All", limit: int = 12) -> List[SearchResult]:
    """
    Pull a random sample of tips, stats, warnings, and myths for the topic.
    These represent the intellectual landscape — what the audience cares about.
    Falls back to all topics when fewer than 3 match.
    """
    conn = get_connection()
    try:
        def _fetch(apply_topic: bool) -> list:
            where = "WHERE mu.content_type IN ('tip', 'stat', 'warning', 'myth')"
            params: list = []
            if apply_topic and topic != "All":
                where += " AND mu.topic = ?"
                params.append(topic)
            params.append(limit)
            return conn.execute(
                f"""
                SELECT mu.unit_id, mu.post_id,
                       COALESCE(ca.username, 'unknown') AS username,
                       mu.topic, mu.content_type, mu.text, mu.claim,
                       mu.confidence AS score, p.video_url,
                       CAST(p.posted_at AS TEXT) AS posted_at
                FROM message_units mu
                LEFT JOIN posts p ON mu.post_id = p.post_id
                LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
                {where}
                ORDER BY RANDOM()
                LIMIT ?
                """,
                params,
            ).fetchall()

        rows = _fetch(apply_topic=True)
        if len(rows) < 3 and topic != "All":
            rows = _fetch(apply_topic=False)

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
