"""Per-client cost rollup (creative-director-ai #14).

The commercial question this answers is not "what did one API call cost?" but
"what does this client cost me per month, and what should I charge them?"

Most of the data was already being recorded per row and simply never rolled
up. The exception was Apify, which had no cost column at all — see
`apify_rate_usd()` below for why its number is treated carefully.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from core.db import get_connection, init_db

# Line items, in pipeline order.
LINES = [
    ("scraping", "Scraping (Apify)"),
    ("transcription", "Transcription (Deepgram)"),
    ("extraction", "Extraction + embeddings (OpenAI)"),
    ("generation", "Generation (OpenAI)"),
]


def apify_rate_usd() -> Optional[float]:
    """USD per scraped result, from APIFY_USD_PER_RESULT.

    Returns None when unset, and callers must render that as *unknown* rather
    than zero. Apify prices in platform credits and the rate depends on the
    plan, so there is no correct default to hardcode — a guessed rate would
    silently produce a confident, wrong number in the one place that has to be
    trustworthy.
    """
    raw = os.getenv("APIFY_USD_PER_RESULT", "").strip()
    if not raw:
        return None
    try:
        rate = float(raw)
    except ValueError:
        return None
    return rate if rate >= 0 else None


def _month_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _period_sum(conn, sql: str, params: list, since: Optional[datetime] = None) -> float:
    period_filter = ""
    query_params = list(params)
    if since is not None:
        period_filter = "WHERE cost_at >= ?"
        query_params.append(since)

    row = conn.execute(
        f"""
        SELECT COALESCE(SUM(cost_usd), 0)
        FROM ({sql}) costs
        {period_filter}
        """,
        query_params,
    ).fetchone()
    return float(row[0] or 0.0)


def _unpriced_count(conn, sql: str, params: list, since: Optional[datetime] = None) -> int:
    period_filter = ""
    query_params = list(params)
    if since is not None:
        period_filter = "AND cost_at >= ?"
        query_params.append(since)

    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM ({sql}) costs
        WHERE cost_usd IS NULL {period_filter}
        """,
        query_params,
    ).fetchone()
    return int(row[0] or 0)


def _line(conn, sql: str, params: list, since: datetime, known: bool = True, month_known: bool = True) -> dict:
    all_time = round(_period_sum(conn, sql, params), 4)
    month = round(_period_sum(conn, sql, params, since), 4)
    return {
        "usd": all_time,
        "all_time_usd": all_time,
        "month_usd": month,
        "known": known,
        "month_known": month_known,
    }


def client_costs(client_id: str, now: Optional[datetime] = None) -> dict:
    """Cost breakdown for one client.

    Each line carries `usd` (None when genuinely unknown) and `known`, so the
    UI can distinguish "cost nothing" from "cost not recorded".
    """
    init_db()
    since = _month_start(now)
    conn = get_connection()
    try:
        scrape_sql = """
            SELECT cost_usd, COALESCE(finished_at, started_at) AS cost_at
            FROM scrape_jobs
            WHERE client_id = ?
        """
        transcription_sql = """
            SELECT cost_usd, transcribed_at AS cost_at
            FROM transcripts
            WHERE client_id = ?
        """
        extraction_sql = """
            SELECT extraction_cost_usd AS cost_usd, transcribed_at AS cost_at
            FROM transcripts
            WHERE client_id = ?
        """
        embedding_sql = """
            SELECT embedding_cost_usd AS cost_usd, embedded_at AS cost_at
            FROM message_units
            WHERE client_id = ?
        """
        generation_sql = """
            SELECT cost_usd, created_at AS cost_at
            FROM generated_content
            WHERE client_id = ?
        """

        unpriced_scrapes = _unpriced_count(conn, scrape_sql, [client_id])
        month_unpriced_scrapes = _unpriced_count(conn, scrape_sql, [client_id], since)
        scraping = _line(
            conn,
            scrape_sql,
            [client_id],
            since,
            known=unpriced_scrapes == 0,
            month_known=month_unpriced_scrapes == 0,
        )
        transcription = _line(conn, transcription_sql, [client_id], since)
        extraction = _line(
            conn,
            f"{extraction_sql} UNION ALL {embedding_sql}",
            [client_id, client_id],
            since,
        )
        generation = _line(conn, generation_sql, [client_id], since)

        reels = conn.execute(
            "SELECT COUNT(*) FROM client_posts WHERE client_id = ?", [client_id]
        ).fetchone()[0]
        pieces = conn.execute(
            "SELECT COUNT(*) FROM generated_content WHERE client_id = ?", [client_id]
        ).fetchone()[0]
        month_pieces = conn.execute(
            """
            SELECT COUNT(*)
            FROM generated_content
            WHERE client_id = ? AND created_at >= ?
            """,
            [client_id, since],
        ).fetchone()[0]
    finally:
        conn.close()

    lines = {
        "scraping": scraping,
        "transcription": transcription,
        "extraction": extraction,
        "generation": generation,
    }

    known_total = round(sum(v["all_time_usd"] or 0.0 for v in lines.values()), 4)
    month_total = round(sum(v["month_usd"] or 0.0 for v in lines.values()), 4)
    complete = all(v["known"] for v in lines.values())
    month_complete = all(v["month_known"] for v in lines.values())

    return {
        "client_id": client_id,
        "lines": lines,
        "month_total_usd": month_total,
        "month_total_is_complete": month_complete,
        "total_usd": known_total,
        "all_time_total_usd": known_total,
        "total_is_complete": complete,
        "unpriced_scrape_jobs": unpriced_scrapes,
        "month_unpriced_scrape_jobs": month_unpriced_scrapes,
        "reels_analysed": reels,
        "pieces_generated": pieces,
        "month_pieces_generated": month_pieces,
        # The number that gets compared against a client's tool subscriptions.
        "cost_per_piece": round(known_total / pieces, 4) if pieces else None,
        "month_cost_per_piece": round(month_total / month_pieces, 4) if month_pieces else None,
    }


def estimate_transcription_usd(minutes: float, rate_per_min: float = 0.0058) -> float:
    """Estimate before spending. Default is the Deepgram Nova-2 rate recorded
    in docs/COST_ANALYSIS.md; pass an explicit rate if the plan differs."""
    return round(max(minutes, 0.0) * rate_per_min, 4)
