"""
Phase 2 — Video downloader.

Uses direct HTTP download (fast, no CDN auth issues) for Apify-sourced URLs.
Falls back to yt-dlp if the direct download returns a non-video content type.
"""

import os
import logging
import time
from pathlib import Path
import requests
import yt_dlp

from core.db import DEFAULT_CLIENT_ID, get_connection

logger = logging.getLogger(__name__)

DOWNLOADS_DIR = Path("downloads")
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Respect CDN servers — don't hammer with concurrent retries
_RETRY_DELAY = 2  # seconds between retries
_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _direct_download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download via requests streaming. Returns True on success."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.instagram.com/",
    }
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=timeout)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "video" not in content_type and "octet-stream" not in content_type:
            return False  # not a video — caller will try yt-dlp
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return dest.stat().st_size > 10_000  # > 10 KB = valid
    except Exception as e:
        logger.debug("Direct download failed for %s: %s", url, e)
        return False


def _ytdlp_download(url: str, dest: Path) -> bool:
    """Download via yt-dlp. Returns True on success."""
    ydl_opts = {
        "outtmpl": str(dest),
        "format": "mp4/bestvideo+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "extractor_retries": 1,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return dest.exists() and dest.stat().st_size > 10_000
    except Exception as e:
        logger.debug("yt-dlp failed for %s: %s", url, e)
        return False


def download_video(post_id: str, video_url: str, output_dir: str = "downloads") -> dict:
    """
    Download a single video.

    Args:
        post_id:    DB post_id (used as filename stem)
        video_url:  CDN URL from Apify response
        output_dir: Directory to save file (default: downloads/)

    Returns:
        dict with keys: success (bool), path (str|None), size_mb (float), error (str|None)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{post_id}.mp4"

    # Already downloaded — skip
    if dest.exists() and dest.stat().st_size > 10_000:
        _update_db(post_id, "done", str(dest), dest.stat().st_size / 1_048_576)
        return {"success": True, "path": str(dest), "size_mb": dest.stat().st_size / 1_048_576, "error": None}

    _update_db(post_id, "downloading", None, None)

    success = _direct_download(video_url, dest) or _ytdlp_download(video_url, dest)

    if success:
        size_mb = dest.stat().st_size / 1_048_576
        _update_db(post_id, "done", str(dest), size_mb)
        return {"success": True, "path": str(dest), "size_mb": round(size_mb, 2), "error": None}
    else:
        if dest.exists():
            dest.unlink()  # remove partial file
        _update_db(post_id, "failed", None, None)
        return {"success": False, "path": None, "size_mb": 0, "error": "All download methods failed"}


def _update_db(post_id: str, status: str, path, size_mb):
    """Write download status back to posts table."""
    try:
        conn = get_connection()
        if path:
            conn.execute(
                """
                UPDATE posts
                SET download_status = ?,
                    local_video_path = ?,
                    file_size_mb = ?,
                    downloaded_at = CURRENT_TIMESTAMP
                WHERE post_id = ?
                """,
                [status, path, size_mb, post_id],
            )
        else:
            conn.execute(
                "UPDATE posts SET download_status = ? WHERE post_id = ?",
                [status, post_id],
            )
        conn.close()
    except Exception as e:
        logger.error("DB update failed for post %s: %s", post_id, e)


def download_pending_posts(
    batch_size: int = 20,
    progress_callback=None,
    client_id: str = DEFAULT_CLIENT_ID,
) -> dict:
    """
    Download all posts with download_status = 'pending'.

    Args:
        batch_size:        Max number of posts to process in one call
        progress_callback: Optional callable(done, total, post_id) for UI updates

    Returns:
        dict with done, failed, skipped counts
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.post_id, p.video_url
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        WHERE cp.client_id = ?
          AND p.download_status = 'pending'
          AND p.video_url IS NOT NULL
          AND p.video_url != ''
        LIMIT ?
        """,
        [client_id, batch_size],
    ).fetchall()
    conn.close()

    total = len(rows)
    done = failed = 0

    for i, (post_id, video_url) in enumerate(rows):
        result = download_video(post_id, video_url)
        if result["success"]:
            done += 1
        else:
            failed += 1
        if progress_callback:
            progress_callback(i + 1, total, post_id)
        if i < total - 1:
            time.sleep(0.3)  # small delay between downloads

    return {"done": done, "failed": failed, "skipped": 0, "total": total}
