"""Acquisition source interface.

The pipeline should not care where reels come from. Apify is the only
implementation today, but its terms do not permit use in a paid service,
so replacing it is the single change that moves this project from research
to sellable. This seam is what makes that swap cheap.

See creative-director-ai #15 (this adapter) and #20 (the eventual swap).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# Fields every source is expected to attempt. A source that cannot supply one
# leaves it None — see the note on `unavailable` below.
NUMERIC_FIELDS = ("likes", "views", "comments_count", "duration_sec")


@dataclass
class ReelRecord:
    """One reel, normalised across sources.

    **None means "this source did not supply it", not zero.**

    Sources differ in coverage: Instagram's Graph API may not expose view
    counts for accounts you do not own, while Apify does. Collapsing a
    missing view count to 0 would corrupt every engagement calculation
    downstream and would do it invisibly, because the numbers still look
    plausible. Callers must branch on `is None`, never on falsiness.
    """

    post_id: str
    source: str                      # which implementation produced this
    handle: str = ""

    caption: Optional[str] = None
    likes: Optional[int] = None
    views: Optional[int] = None
    comments_count: Optional[int] = None
    duration_sec: Optional[float] = None
    posted_at: Optional[str] = None

    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # The source's own payload, kept so existing consumers that expect the
    # native shape keep working while the adapter is introduced.
    raw: dict[str, Any] = field(default_factory=dict)

    def unavailable(self) -> list[str]:
        """Fields this source did not supply. Empty means full coverage."""
        return [f for f in NUMERIC_FIELDS if getattr(self, f) is None]

    def has(self, field_name: str) -> bool:
        """True when the source supplied this field, including a real zero."""
        return getattr(self, field_name, None) is not None


@runtime_checkable
class ReelSource(Protocol):
    """Anything that can fetch reels for an account.

    Implementations raise their own errors; the caller decides retry policy.
    """

    name: str

    def fetch_account_reels(self, handle: str, limit: int) -> list[ReelRecord]:
        ...
