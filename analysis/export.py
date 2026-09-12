"""
Shared CSV export module.

Provides reusable report builders for the Posts + Transcripts and
Knowledge Units reports.  Used by both the Data Viewer page and
the Full Analysis pipeline.
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from core.db import get_connection

EXPORTS_DIR = Path("data/exports")
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_posts_report_df(account_filter: str = "All") -> pd.DataFrame:
    """One row per reel — metrics + transcript + knowledge-unit summary."""
    account_where = (
        "" if account_filter == "All"
        else f"WHERE p.account_id = '{account_filter}'"
    )
    conn = get_connection()
    try:
        df = conn.execute(f"""
            SELECT
                p.post_id,
                COALESCE(ca.username, p.account_id)        AS username,
                ca.full_name,
                ca.followers,
                ca.following,
                ca.post_count                              AS creator_total_posts,
                ca.is_verified,
                ca.category                               AS creator_category,
                ca.bio,
                p.likes,
                p.views,
                p.comments_count,
                ROUND(p.engagement_rate, 4)               AS engagement_rate_pct,
                ROUND(p.duration_sec, 1)                  AS duration_sec,
                CAST(p.posted_at AS TEXT)                 AS posted_at,
                CAST(p.scraped_at AS TEXT)                AS scraped_at,
                p.caption_clean                           AS caption,
                array_to_string(p.hashtags,  ', ')        AS hashtags,
                array_to_string(p.mentions,  ', ')        AS mentions,
                p.is_pinned,
                p.is_sponsored,
                p.video_url,
                p.download_status,
                t.transcript,
                t.language                               AS transcript_language,
                t.word_count                             AS transcript_word_count,
                ROUND(t.confidence, 3)                   AS transcript_confidence,
                t.provider                               AS transcript_provider,
                CAST(t.transcribed_at AS TEXT)           AS transcribed_at,
                COUNT(mu.unit_id)                        AS knowledge_units_extracted,
                string_agg(DISTINCT mu.topic,        ', ' ORDER BY mu.topic)
                                                         AS topics_covered,
                string_agg(DISTINCT mu.content_type, ', ' ORDER BY mu.content_type)
                                                         AS content_types_found,
                string_agg(DISTINCT mu.subtopic,     ' | ' ORDER BY mu.subtopic)
                                                         AS subtopics
            FROM posts p
            LEFT JOIN creator_accounts ca  ON p.account_id = ca.account_id
            LEFT JOIN transcripts t         ON p.post_id    = t.post_id
            LEFT JOIN message_units mu      ON p.post_id    = mu.post_id
            {account_where}
            GROUP BY
                p.post_id, p.account_id, ca.username, ca.full_name, ca.followers, ca.following,
                ca.post_count, ca.is_verified, ca.category, ca.bio,
                p.likes, p.views, p.comments_count, p.engagement_rate,
                p.duration_sec, p.posted_at, p.scraped_at,
                p.caption_clean, p.hashtags, p.mentions,
                p.is_pinned, p.is_sponsored, p.video_url, p.download_status,
                t.transcript, t.language, t.word_count, t.confidence,
                t.provider, t.transcribed_at
            ORDER BY p.posted_at DESC NULLS LAST
        """).df()
    finally:
        conn.close()
    return df


def build_units_report_df(account_filter: str = "All") -> pd.DataFrame:
    """One row per extracted insight — with source reel metrics."""
    account_where = (
        "" if account_filter == "All"
        else f"WHERE p.account_id = '{account_filter}'"
    )
    conn = get_connection()
    try:
        df = conn.execute(f"""
            SELECT
                mu.unit_id,
                mu.post_id,
                COALESCE(ca.username, p.account_id)    AS username,
                ca.full_name,
                ca.followers,
                ca.is_verified,
                ca.category                           AS creator_category,
                p.likes,
                p.views,
                ROUND(p.engagement_rate, 4)           AS engagement_rate_pct,
                CAST(p.posted_at AS TEXT)             AS posted_at,
                mu.topic,
                mu.subtopic,
                mu.content_type,
                mu.text                               AS insight_text,
                mu.claim,
                mu.advice,
                ROUND(mu.confidence, 3)               AS extraction_confidence,
                CAST(mu.extracted_at AS TEXT)         AS extracted_at,
                array_to_string(p.hashtags, ', ')     AS reel_hashtags,
                LEFT(p.caption_clean, 200)            AS reel_caption_preview,
                LEFT(t.transcript, 500)               AS transcript_preview
            FROM message_units mu
            LEFT JOIN posts p             ON mu.post_id    = p.post_id
            LEFT JOIN creator_accounts ca ON p.account_id  = ca.account_id
            LEFT JOIN transcripts t       ON mu.post_id    = t.post_id
            {account_where}
            ORDER BY p.posted_at DESC NULLS LAST, mu.content_type, mu.topic
        """).df()
    finally:
        conn.close()
    return df


def export_to_csv(df: pd.DataFrame, name_prefix: str) -> str:
    """Write a DataFrame to data/exports/ with a timestamped filename. Returns the path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{name_prefix}_{ts}.csv"
    path = EXPORTS_DIR / fname
    df.to_csv(path, index=False)
    return str(path)
