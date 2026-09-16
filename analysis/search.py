"""
Phase 5 — Semantic search over message units.

Uses DuckDB's list_cosine_similarity() for vector search —
no external vector DB required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import core.db as db_mod
from core.db import get_connection
from analysis.embeddings import _MODEL as EMBEDDING_MODEL, embed_batch
from tools.embed_units import embedding_dimension, width_of


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
    # Carried so generation can enforce the role firewall on what search hands
    # it. Without this field the filter in analysis/content_gen.py reads None
    # for every unit and silently passes competitor technique through.
    # NULL stays None: not classified is not the same as safe.
    unit_role: Optional[str] = None


def _require_client_id(client_id: Optional[str]) -> str:
    if not client_id:
        raise ValueError("client_id is required")
    return client_id


def _message_units_has_column(conn, column: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'message_units'
          AND column_name = ?
        LIMIT 1
        """,
        [column],
    ).fetchone()
    return row is not None


def _embedding_width_or_refuse() -> int:
    width = embedding_dimension(db_mod.DB_PATH)
    if width is None:
        raise RuntimeError(
            "Could not read a fixed width for message_units.embedding; "
            "refusing to guess a query embedding dimension."
        )
    return width


def semantic_search(
    query: str,
    openai_api_key: str,
    client_id: str,
    topic_filter: Optional[str] = None,
    content_type_filter: Optional[str] = None,
    top_k: int = 20,
    unit_role_filter: Optional[str] = None,
    exclude_roles: Optional[Sequence[str]] = None,
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
    width = _embedding_width_or_refuse()

    conn = get_connection()
    try:
        if not _message_units_has_column(conn, "embedding_model"):
            return []

        query_vec = embed_batch([query], openai_api_key, dimensions=width)[0]

        where_clauses = ["mu.embedding IS NOT NULL", "mu.embedding_model = ?"]
        where_params = [EMBEDDING_MODEL]

        if topic_filter and topic_filter != "All":
            where_clauses.append("mu.topic = ?")
            where_params.append(topic_filter)

        if content_type_filter and content_type_filter != "All":
            where_clauses.append("mu.content_type = ?")
            where_params.append(content_type_filter)

        # Second axis (#27). "Unclassified" is selectable on purpose: it is how
        # you see how much of the corpus has actually been reviewed.
        if unit_role_filter and unit_role_filter != "All":
            if unit_role_filter == "Unclassified":
                where_clauses.append("mu.unit_role IS NULL")
            else:
                where_clauses.append("mu.unit_role = ?")
                where_params.append(unit_role_filter)

        # NULL is deliberately NOT excluded — see ROLES_EXCLUDED_FROM_GENERATION.
        # An unclassified unit is unproven, not disqualified, and excluding it
        # would empty the reference pool until the backfill runs.
        if exclude_roles:
            placeholders = ", ".join(["?"] * len(exclude_roles))
            where_clauses.append(
                f"(mu.unit_role IS NULL OR mu.unit_role NOT IN ({placeholders}))")
            where_params.extend(exclude_roles)

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
                list_cosine_similarity(mu.embedding, ?::FLOAT[{width}]) AS score,
                p.video_url,
                CAST(p.posted_at AS TEXT) AS posted_at,
                mu.unit_role
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
                unit_role=r[10],
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
    unit_role_filter: Optional[str] = None,
    exclude_roles: Optional[Sequence[str]] = None,
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

        if unit_role_filter and unit_role_filter != "All":
            if unit_role_filter == "Unclassified":
                where_clauses.append("mu.unit_role IS NULL")
            else:
                where_clauses.append("mu.unit_role = ?")
                params.append(unit_role_filter)

        # NULL is deliberately NOT excluded — see ROLES_EXCLUDED_FROM_GENERATION.
        # An unclassified unit is unproven, not disqualified, and excluding it
        # would empty the reference pool until the backfill runs.
        if exclude_roles:
            placeholders = ", ".join(["?"] * len(exclude_roles))
            where_clauses.append(
                f"(mu.unit_role IS NULL OR mu.unit_role NOT IN ({placeholders}))")
            params.extend(exclude_roles)

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
                CAST(p.posted_at AS TEXT) AS posted_at,
                mu.unit_role
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
                unit_role=r[10],
            )
            for r in rows
        ]
    finally:
        conn.close()
