"""Absent-preserving metric helpers for the archived video analysis page."""

from __future__ import annotations

import os
import math
from collections.abc import Iterable

from core.db import engagement_rate_of, opt_number


def ratio_percent(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator * 100, 2)


def round_or_none(value, digits=2):
    if value is None:
        return None
    return round(value, digits)


def video_file_size_mb(path: str):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None


def known_values(values: Iterable):
    return [
        value for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]


def known_total(values: Iterable):
    known = known_values(values)
    if not known:
        return None
    return sum(known)


def known_mean(values: Iterable):
    known = known_values(values)
    if not known:
        return None
    return sum(known) / len(known)


def format_count_or_unknown(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return f"{int(value):,}"


def format_percent_or_unknown(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return f"{value:.{digits}f}%"


def build_video_analysis_result(metadata: dict, short_code: str, file_size_mb=None):
    likes = opt_number(metadata.get("likesCount"), int)
    views = opt_number(metadata.get("videoViewCount"), int)
    plays = opt_number(metadata.get("videoPlayCount"), int)
    comments = opt_number(metadata.get("commentsCount"), int)
    duration = opt_number(metadata.get("videoDuration"), float)
    caption = metadata.get("caption", "") or ""
    owner = metadata.get("ownerUsername", "") or ""
    location = metadata.get("locationName", "") or ""
    timestamp = metadata.get("timestamp", "")

    engagement_rate = engagement_rate_of(likes, views)
    comments_rate = ratio_percent(comments, likes)
    play_completion = ratio_percent(plays, views)

    caption_lower = caption.lower()
    location_lower = location.lower()
    if any(word in caption_lower + location_lower for word in ["food", "restaurant", "bar", "drink", "ramen", "eat"]):
        category = "Food & Beverage"
    elif any(word in caption_lower for word in ["party", "celebration", "anniversary", "event"]):
        category = "Events & Celebrations"
    elif any(word in caption_lower for word in ["style", "aesthetic", "vibe", "mood"]):
        category = "Lifestyle"
    elif location_lower:
        category = "Location-based"
    else:
        category = "General Content"

    virality_score = 0
    if engagement_rate is not None and engagement_rate > 5:
        virality_score += 2
    if comments_rate is not None and comments_rate > 10:
        virality_score += 1
    if duration is not None and 15 <= duration <= 60:
        virality_score += 2
    if len(caption) > 100:
        virality_score += 1
    if "@" in caption:
        virality_score += 1
    if location:
        virality_score += 1
    if play_completion is not None and play_completion > 80:
        virality_score += 2
    virality_score = min(10, virality_score)

    return {
        "short_code": short_code,
        "owner": owner,
        "category": category,
        "file_size_mb": round_or_none(file_size_mb, 2),
        "duration": duration,
        "likes": likes,
        "views": views,
        "plays": plays,
        "comments": comments,
        "engagement_rate": round_or_none(engagement_rate, 2),
        "comments_rate": comments_rate,
        "play_completion": play_completion,
        "virality_potential": virality_score,
        "caption_length": len(caption),
        "has_location": bool(location),
        "mentions_count": len(metadata.get("mentions", []) or []),
        "hashtags_count": len(metadata.get("hashtags", []) or []),
        "timestamp": timestamp[:10] if timestamp else "Unknown",
        "location": location or "Not specified",
    }
