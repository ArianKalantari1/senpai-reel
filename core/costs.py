"""Per-client cost rollup (creative-director-ai #14).

The commercial question this answers is not "what did one API call cost?" but
"what does this client cost me per month, and what should I charge them?"

Most of the data was already being recorded per row and simply never rolled
up. The exception was Apify, which had no cost column at all — see
`apify_rate_usd()` below for why its number is treated carefully.
"""
from __future__ import annotations

import os
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


def _scalar(conn, sql: str, params: list):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row and row[0] is not None else None


def client_costs(client_id: str) -> dict:
    """Cost breakdown for one client.

    Each line carries `usd` (None when genuinely unknown) and `known`, so the
    UI can distinguish "cost nothing" from "cost not recorded".
    """
    init_db()
    conn = get_connection()
    try:
        scrape_cost = _scalar(
            conn,
            "SELECT SUM(cost_usd) FROM scrape_jobs WHERE client_id = ?",
            [client_id],
        )
        unpriced_scrapes = conn.execute(
            "SELECT COUNT(*) FROM scrape_jobs WHERE client_id = ? AND cost_usd IS NULL",
            [client_id],
        ).fetchone()[0]

        transcription = _scalar(
            conn,
            "SELECT SUM(cost_usd) FROM transcripts WHERE client_id = ?",
            [client_id],
        )
        extraction = _scalar(
            conn,
            "SELECT SUM(extraction_cost_usd) FROM transcripts WHERE client_id = ?",
            [client_id],
        )
        generation = _scalar(
            conn,
            "SELECT SUM(cost_usd) FROM generated_content WHERE client_id = ?",
            [client_id],
        )

        reels = conn.execute(
            "SELECT COUNT(*) FROM client_posts WHERE client_id = ?", [client_id]
        ).fetchone()[0]
        pieces = conn.execute(
            "SELECT COUNT(*) FROM generated_content WHERE client_id = ?", [client_id]
        ).fetchone()[0]
    finally:
        conn.close()

    lines = {
        "scraping": {"usd": scrape_cost, "known": unpriced_scrapes == 0},
        "transcription": {"usd": transcription, "known": True},
        "extraction": {"usd": extraction, "known": True},
        "generation": {"usd": generation, "known": True},
    }

    known_total = sum(v["usd"] or 0.0 for v in lines.values())
    complete = all(v["known"] for v in lines.values())

    return {
        "client_id": client_id,
        "lines": lines,
        "total_usd": round(known_total, 4),
        "total_is_complete": complete,
        "unpriced_scrape_jobs": unpriced_scrapes,
        "reels_analysed": reels,
        "pieces_generated": pieces,
        # The number that gets compared against a client's tool subscriptions.
        "cost_per_piece": round(known_total / pieces, 4) if pieces else None,
    }


def estimate_transcription_usd(minutes: float, rate_per_min: float = 0.0058) -> float:
    """Estimate before spending. Default is the Deepgram Nova-2 rate recorded
    in docs/COST_ANALYSIS.md; pass an explicit rate if the plan differs."""
    return round(max(minutes, 0.0) * rate_per_min, 4)
