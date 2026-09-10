"""
Scraping engine — wraps Apify's instagram-reel-scraper actor.
Handles: API call, response parsing, DB upsert, audit logging.
"""
import requests
import time
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
    url = (
        f"https://api.apify.com/v2/acts/{ACTOR_ID}"
        f"/run-sync-get-dataset-items?token={apify_token}"
    )
    payload = {
        "username": [f"https://www.instagram.com/{username}/"],
        "resultsLimit": max_items,
        "skipPinnedPosts": False,
        "includeSharesCount": False,
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            items = resp.json()
            if not isinstance(items, list):
                raise ScraperError(f"Unexpected response format: {type(items)}")
            return items, job_id
        except requests.exceptions.HTTPError as e:
            last_error = str(e)
            if resp.status_code in (401, 403):
                # Auth errors — no point retrying
                finish_scrape_job(job_id, 0, 0, "failed", last_error)
                raise ScraperError(f"Auth error for @{username}: {last_error}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # 1s, 2s backoff
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(2 ** attempt)

    finish_scrape_job(job_id, 0, 0, "failed", last_error)
    raise ScraperError(f"Failed to scrape @{username} after {retries+1} attempts: {last_error}")


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
