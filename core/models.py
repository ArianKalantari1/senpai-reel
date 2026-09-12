"""
Dataclass definitions for the canonical data model.
These mirror the DB tables exactly and are used throughout the codebase.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class CreatorAccount:
    account_id: str          # Instagram user ID
    username: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    post_count: Optional[int] = None
    is_verified: bool = False
    profile_pic_url: Optional[str] = None
    external_url: Optional[str] = None
    category: Optional[str] = None   # e.g. "HR/Recruitment"
    first_seen_at: Optional[datetime] = None
    last_scraped_at: Optional[datetime] = None


@dataclass
class Post:
    post_id: str             # Instagram shortCode
    account_id: str          # FK → creator_accounts.account_id
    caption: Optional[str] = None
    caption_clean: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    likes: int = 0
    views: int = 0
    comments_count: int = 0
    duration_sec: float = 0.0
    posted_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    is_pinned: bool = False
    is_sponsored: bool = False
    engagement_rate: float = 0.0
    local_video_path: Optional[str] = None
    archived_video_path: Optional[str] = None
    video_archived_at: Optional[datetime] = None
    local_audio_path: Optional[str] = None
    keyframes_dir: Optional[str] = None
    keyframes_extracted_at: Optional[datetime] = None
    keyframe_count: Optional[int] = None
    download_status: str = "pending"   # pending | done | failed | skipped
    raw_json: Optional[dict] = None


@dataclass
class ScrapeJob:
    job_id: str
    username: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    reels_found: int = 0
    reels_new: int = 0
    status: str = "running"    # running | done | failed
    error_msg: Optional[str] = None
