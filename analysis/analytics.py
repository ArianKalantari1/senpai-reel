"""
Phase 6 — Analytics queries.

All analytics queries are in one place so the Streamlit page stays thin.
"""

from __future__ import annotations

import pandas as pd

from core.db import DEFAULT_CLIENT_ID, get_connection


def _client_scope(alias: str, client_id: str | None, params: list) -> str:
    if not client_id:
        return ""
    params.append(client_id)
    return (
        f"EXISTS (SELECT 1 FROM client_posts cp "
        f"WHERE cp.post_id = {alias}.post_id AND cp.client_id = ?)"
    )


def get_creator_leaderboard(
    limit: int = 20,
    client_id: str | None = DEFAULT_CLIENT_ID,
) -> pd.DataFrame:
    conn = get_connection()
    try:
        params = []
        scope = _client_scope("p", client_id, params)
        where_sql = f"WHERE {scope}" if scope else ""
        params.append(limit)
        return conn.execute(
            """
            SELECT
                ca.username,
                COUNT(p.post_id)           AS total_posts,
                ROUND(AVG(p.engagement_rate), 2) AS avg_engagement,
                MAX(p.engagement_rate)     AS max_engagement,
                SUM(p.views)               AS total_views,
                SUM(p.likes)               AS total_likes,
                ROUND(AVG(p.duration_sec), 0) AS avg_duration_sec
            FROM posts p
            JOIN creator_accounts ca ON p.account_id = ca.account_id
            {where_sql}
            GROUP BY ca.username
            ORDER BY avg_engagement DESC
            LIMIT ?
            """.format(where_sql=where_sql),
            params,
        ).df()
    finally:
        conn.close()


def get_topic_distribution(client_id: str | None = DEFAULT_CLIENT_ID) -> pd.DataFrame:
    conn = get_connection()
    try:
        params = []
        scope = _client_scope("mu", client_id, params)
        where_sql = f"WHERE {scope}" if scope else ""
        return conn.execute(
            """
            SELECT mu.topic, COUNT(*) AS unit_count
            FROM message_units mu
            {where_sql}
            GROUP BY mu.topic
            ORDER BY unit_count DESC
            """.format(where_sql=where_sql),
            params,
        ).df()
    finally:
        conn.close()


def get_content_gap_matrix(client_id: str | None = DEFAULT_CLIENT_ID) -> pd.DataFrame:
    """Returns a pivot table: topic (rows) × content_type (cols) = unit count."""
    conn = get_connection()
    try:
        params = []
        scope = _client_scope("mu", client_id, params)
        where_sql = f"WHERE {scope}" if scope else ""
        df = conn.execute(
            """
            SELECT mu.topic, mu.content_type, COUNT(*) AS cnt
            FROM message_units mu
            {where_sql}
            GROUP BY mu.topic, mu.content_type
            """.format(where_sql=where_sql),
            params,
        ).df()
    finally:
        conn.close()

    if df.empty:
        return df
    pivot = df.pivot(index="topic", columns="content_type", values="cnt").fillna(0).astype(int)
    return pivot


def get_top_posts(
    topic: str = "All",
    limit: int = 20,
    client_id: str | None = DEFAULT_CLIENT_ID,
) -> pd.DataFrame:
    conn = get_connection()
    try:
        params = []
        where_clauses = []
        scope = _client_scope("p", client_id, params)
        if scope:
            where_clauses.append(scope)
        topic_join = ""
        if topic != "All":
            topic_join = "LEFT JOIN message_units mu ON p.post_id = mu.post_id"
            where_clauses.append("mu.topic = ?")
            params.append(topic)
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        params.append(limit)
        return conn.execute(
            """
            SELECT DISTINCT
                p.post_id,
                ca.username,
                p.caption,
                p.likes,
                p.views,
                ROUND(p.engagement_rate, 2) AS engagement_rate,
                p.duration_sec,
                CAST(p.posted_at AS TEXT) AS posted_at,
                p.video_url,
                p.hashtags
            FROM posts p
            JOIN creator_accounts ca ON p.account_id = ca.account_id
            {topic_join}
            {where_sql}
            ORDER BY p.engagement_rate DESC NULLS LAST
            LIMIT ?
            """.format(topic_join=topic_join, where_sql=where_sql),
            params,
        ).df()
    finally:
        conn.close()


def get_hashtag_intelligence(
    limit: int = 30,
    client_id: str | None = DEFAULT_CLIENT_ID,
) -> pd.DataFrame:
    """Most-used hashtags across all posts."""
    conn = get_connection()
    try:
        params = []
        scope = _client_scope("p", client_id, params)
        where_sql = "WHERE p.hashtags IS NOT NULL"
        if scope:
            where_sql += f" AND {scope}"
        df = conn.execute(
            f"SELECT p.hashtags FROM posts p {where_sql}",
            params,
        ).df()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=["hashtag", "count"])

    from collections import Counter
    counter: Counter = Counter()
    for tags in df["hashtags"]:
        if tags is not None and len(tags) > 0:
            for tag in tags:
                if tag:
                    counter[tag.lower().strip("#")] += 1

    top = counter.most_common(limit)
    return pd.DataFrame(top, columns=["hashtag", "count"])


def get_posting_cadence(client_id: str | None = DEFAULT_CLIENT_ID) -> pd.DataFrame:
    """Posts per week per creator."""
    conn = get_connection()
    try:
        params = []
        scope = _client_scope("p", client_id, params)
        where_clauses = ["p.posted_at IS NOT NULL"]
        if scope:
            where_clauses.append(scope)
        return conn.execute(
            """
            SELECT
                ca.username,
                DATE_TRUNC('week', p.posted_at) AS week,
                COUNT(*) AS posts
            FROM posts p
            JOIN creator_accounts ca ON p.account_id = ca.account_id
            WHERE {where_sql}
            GROUP BY ca.username, week
            ORDER BY week DESC, posts DESC
            """.format(where_sql=" AND ".join(where_clauses)),
            params,
        ).df()
    finally:
        conn.close()
