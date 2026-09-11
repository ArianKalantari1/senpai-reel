"""Apify implementation of ReelSource.

Wraps the `apify~instagram-reel-scraper` actor. Behaviour is unchanged from
the original inline implementation in `collection/scraper.py`; this only
moves it behind the interface.

Licensing note: Apify's terms do not permit use in a paid service delivered
to third parties. This source is for research use. See creative-director-ai
#16 and #20.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from collection.sources.base import ReelRecord

ACTOR_ID = "apify~instagram-reel-scraper"
NAME = "apify"


class ApifyError(RuntimeError):
    """Apify call failed in a way the caller should handle."""


def _opt_int(value: Any) -> Optional[int]:
    """Coerce to int, preserving None.

    Deliberately not `int(value or 0)`: a missing count and a genuine zero
    are different facts and must stay distinguishable.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_record(item: dict, handle: str = "") -> ReelRecord:
    """Map one raw Apify item to a ReelRecord."""
    return ReelRecord(
        post_id=item.get("shortCode") or "",
        source=NAME,
        handle=handle or (item.get("ownerUsername") or ""),
        caption=item.get("caption"),
        likes=_opt_int(item.get("likesCount")),
        views=_opt_int(item.get("videoViewCount")),
        comments_count=_opt_int(item.get("commentsCount")),
        duration_sec=_opt_float(item.get("videoDuration")),
        posted_at=item.get("timestamp"),
        video_url=item.get("videoUrl"),
        audio_url=item.get("audioUrl"),
        thumbnail_url=item.get("displayUrl"),
        raw=item,
    )


class ApifySource:
    """Fetches reels via the Apify actor."""

    name = NAME

    def __init__(self, token: str, retries: int = 2, timeout: int = 120):
        if not token:
            raise ApifyError("Apify token is required")
        self.token = token
        self.retries = retries
        self.timeout = timeout

    def fetch_raw(self, handle: str, limit: int = 50) -> list[dict]:
        """Return the actor's native items, with the original retry policy."""
        url = (
            f"https://api.apify.com/v2/acts/{ACTOR_ID}"
            f"/run-sync-get-dataset-items?token={self.token}"
        )
        payload = {
            "username": [f"https://www.instagram.com/{handle}/"],
            "resultsLimit": limit,
            "skipPinnedPosts": False,
            "includeSharesCount": False,
        }

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                items = resp.json()
                if not isinstance(items, list):
                    raise ApifyError(f"Unexpected response format: {type(items)}")
                return items
            except requests.exceptions.HTTPError as exc:
                last_error = str(exc)
                if resp.status_code in (401, 403):
                    # Auth errors will not resolve on retry.
                    raise ApifyError(f"Auth error for @{handle}: {last_error}")
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
            except ApifyError:
                raise
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        raise ApifyError(
            f"Failed to fetch @{handle} after {self.retries + 1} attempts: {last_error}"
        )

    def fetch_account_reels(self, handle: str, limit: int = 50) -> list[ReelRecord]:
        return [to_record(item, handle) for item in self.fetch_raw(handle, limit)]
