"""
Scraping engine — wraps Apify's instagram-reel-scraper actor.
Handles: API call, response parsing, DB upsert, audit logging.
"""
from datetime import datetime
from typing import List, Tuple, Optional

from core.db import (
    DEFAULT_CLIENT_ID,
    save_raw_scrape,
    upsert_creator_account,
    upsert_post,
    start_scrape_job,
    finish_scrape_job,
    _insert_comments_recursive,
    save_structured_scrape,
)
import duckdb

ACTOR_ID = "apify~instagram-reel-scraper"
DB_PATH = "reels.duckdb"


from collection.sources.apify import ApifyError, ApifySource
from collection.sources.base import ReelRecord


class ScraperError(Exception):
    pass


def scrape_account(
    username: str,
    apify_token: str,
    max_items: int = 50,
    retries: int = 2,
    client_id: str = DEFAULT_CLIENT_ID,
) -> Tuple[List[dict], str]:
    """
    Scrape reels for a single Instagram account.

    Returns:
        (items, job_id) — raw Apify items list and the DB job_id

    Raises:
        ScraperError on non-retryable failures.
    """
    job_id = start_scrape_job(username, client_id)
    source = ApifySource(apify_token, retries=retries)
    try:
        items = source.fetch_raw(username, max_items)
    except ApifyError as exc:
        finish_scrape_job(job_id, 0, 0, "failed", str(exc))
        raise ScraperError(str(exc)) from exc
    return items, job_id


def fetch_records(
    username: str,
    apify_token: str,
    max_items: int = 50,
    retries: int = 2,
) -> List[ReelRecord]:
    """Fetch reels as normalised ReelRecords.

    Prefer this over scrape_account() for new code: it is source-agnostic and
    distinguishes a field the source did not supply from a genuine zero.
    Does not touch the database.
    """
    return ApifySource(apify_token, retries=retries).fetch_account_reels(
        username, max_items
    )


def process_and_store(
    username: str,
    items: List[dict],
    job_id: str,
    client_id: str = DEFAULT_CLIENT_ID,
) -> Tuple[int, int]:
    """
    Persist scraped items to DB:
      1. raw_scrapes (deduped by shortCode)
      2. creator_accounts (upsert)
      3. posts (upsert)
      4. legacy reels/comments/tagged_users tables (for backward compat)
      5. Mark scrape_job as done

    Returns:
        (reels_found, reels_new)
    """
    if not items:
        finish_scrape_job(job_id, 0, 0, "done")
        return 0, 0

    # 1. Raw store (deduped)
    save_raw_scrape(username, items, client_id)

    # 2. Canonical tables
    account_id = None
    reels_new = 0
    for item in items:
        # Upsert creator on first item (all items share same owner)
        if account_id is None:
            account_id = upsert_creator_account(username, item)

        is_new = upsert_post(account_id, item, client_id)
        if is_new:
            reels_new += 1

    # 3. Legacy structured tables (keeps existing pages working)
    save_structured_scrape(username, items, client_id)

    reels_found = len(items)
    finish_scrape_job(job_id, reels_found, reels_new, "done")
    return reels_found, reels_new


def scrape_and_store(
    username: str,
    apify_token: str,
    max_items: int = 50,
    client_id: str = DEFAULT_CLIENT_ID,
) -> dict:
    """
    Full pipeline: scrape + store for one account.
    Returns a result dict suitable for display in the UI.
    """
    try:
        items, job_id = scrape_account(username, apify_token, max_items, client_id=client_id)
        reels_found, reels_new = process_and_store(username, items, job_id, client_id)
        return {
            "username": username,
            "status": "done",
            "reels_found": reels_found,
            "reels_new": reels_new,
            "job_id": job_id,
            "error": None,
        }
    except ScraperError as e:
        return {
            "username": username,
            "status": "failed",
            "reels_found": 0,
            "reels_new": 0,
            "job_id": None,
            "error": str(e),
        }
