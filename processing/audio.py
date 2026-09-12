"""
Phase 2 — Audio extractor.

Converts downloaded MP4/video files to 16kHz mono WAV for Deepgram transcription.
Uses ffmpeg subprocess — no Python audio library dependency.
"""

import subprocess
import logging
from pathlib import Path

from core.db import DEFAULT_CLIENT_ID, get_connection
from processing.concurrency import cpu_worker_count, run_bounded, run_db_write
from processing.media_archive import extract_keyframes, keyframes_available, record_keyframes

logger = logging.getLogger(__name__)

AUDIO_DIR = Path("audio_extracts")
AUDIO_DIR.mkdir(exist_ok=True)


def extract_audio(video_path: str, output_dir: str = "audio_extracts") -> dict:
    """
    Extract audio from a video file as 16kHz mono WAV.

    Args:
        video_path: Path to the downloaded .mp4 file
        output_dir: Directory to write the .wav file

    Returns:
        dict with keys: success (bool), path (str|None), error (str|None)
    """
    src = Path(video_path)
    if not src.exists():
        return {"success": False, "path": None, "error": f"Video not found: {video_path}"}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.stem}.wav"

    # Already extracted — skip
    if dest.exists() and dest.stat().st_size > 1_000:
        return {"success": True, "path": str(dest), "error": None}

    cmd = [
        "ffmpeg",
        "-y",               # overwrite without prompting
        "-i", str(src),
        "-vn",              # no video stream
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16 kHz sample rate (Deepgram requirement)
        "-ac", "1",         # mono
        str(dest),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return {"success": False, "path": None, "error": result.stderr[-300:]}

        if not dest.exists() or dest.stat().st_size < 1_000:
            return {"success": False, "path": None, "error": "Output WAV is empty"}

        return {"success": True, "path": str(dest), "error": None}

    except subprocess.TimeoutExpired:
        return {"success": False, "path": None, "error": "ffmpeg timed out after 120s"}
    except FileNotFoundError:
        return {"success": False, "path": None, "error": "ffmpeg not found — install with: brew install ffmpeg"}


def extract_audio_for_post(post_id: str, client_id: str = DEFAULT_CLIENT_ID) -> dict:
    """
    Extract audio for a specific post_id.
    Pulls local_video_path from DB, runs extraction, writes local_audio_path back.
    """
    conn = get_connection()
    row = conn.execute(
        """
        SELECT p.local_video_path, p.local_audio_path, p.keyframes_dir
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        WHERE p.post_id = ? AND cp.client_id = ?
        """,
        [post_id, client_id],
    ).fetchone()
    conn.close()

    if not row:
        return {"success": False, "path": None, "error": f"post_id {post_id} not found"}

    video_path, existing_audio, existing_keyframes = row
    audio_exists = bool(existing_audio and Path(existing_audio).exists())
    if audio_exists and keyframes_available(existing_keyframes):
        return {"success": True, "path": existing_audio, "error": None}

    if not video_path or not Path(video_path).exists():
        if audio_exists:
            return {"success": True, "path": existing_audio, "error": None}
        return {"success": False, "path": None, "error": "No local video file for this post"}

    audio_result = {"success": True, "path": existing_audio, "error": None}
    if not audio_exists:
        audio_result = extract_audio(video_path)
        if audio_result["success"]:
            _update_audio_path(post_id, audio_result["path"])

    keyframe_result = {"success": True, "path": existing_keyframes, "count": None, "error": None}
    if not keyframes_available(existing_keyframes):
        keyframe_result = extract_keyframes(video_path, post_id)
        if keyframe_result["success"]:
            record_keyframes(post_id, keyframe_result["path"], keyframe_result["count"])

    if audio_result["success"] and keyframe_result["success"]:
        return {"success": True, "path": audio_result["path"], "error": None}
    return {
        "success": False,
        "path": audio_result.get("path"),
        "error": audio_result.get("error") or keyframe_result.get("error"),
    }


def _update_audio_path(post_id: str, audio_path: str):
    run_db_write(_write_audio_path, post_id, audio_path)


def _write_audio_path(post_id: str, audio_path: str):
    try:
        conn = get_connection()
        conn.execute(
            "UPDATE posts SET local_audio_path = ? WHERE post_id = ?",
            [audio_path, post_id],
        )
        conn.close()
    except Exception as e:
        logger.error("Failed to update audio path for %s: %s", post_id, e)


def extract_audio_for_downloaded_posts(
    progress_callback=None,
    client_id: str = DEFAULT_CLIENT_ID,
    max_workers: int | None = None,
) -> dict:
    """
    Extract audio for all posts that are downloaded but missing audio.

    Returns:
        dict with done, failed, skipped counts
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.post_id, p.local_video_path, p.local_audio_path, p.keyframes_dir
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        WHERE cp.client_id = ?
          AND p.download_status = 'done'
          AND p.local_video_path IS NOT NULL
          AND p.local_video_path != ''
          AND (
              p.local_audio_path IS NULL
              OR p.local_audio_path = ''
              OR p.keyframes_dir IS NULL
              OR p.keyframes_dir = ''
          )
        """,
        [client_id],
    ).fetchall()
    conn.close()

    total = len(rows)
    done = failed = 0

    def _extract(row):
        post_id, video_path, existing_audio, existing_keyframes = row
        audio_exists = bool(existing_audio and Path(existing_audio).exists())
        audio_result = {"success": True, "path": existing_audio, "error": None}
        if not audio_exists:
            audio_result = extract_audio(video_path)
            if audio_result["success"]:
                _update_audio_path(post_id, audio_result["path"])

        keyframe_result = {"success": True, "path": existing_keyframes, "count": None, "error": None}
        if not keyframes_available(existing_keyframes):
            keyframe_result = extract_keyframes(video_path, post_id)
            if keyframe_result["success"]:
                record_keyframes(post_id, keyframe_result["path"], keyframe_result["count"])

        if audio_result["success"] and keyframe_result["success"]:
            return {"success": True, "path": audio_result["path"], "error": None}
        return {
            "success": False,
            "path": audio_result.get("path"),
            "error": audio_result.get("error") or keyframe_result.get("error"),
        }

    def _progress(count, count_total, row, result, error):
        if progress_callback:
            progress_callback(count, count_total, row[0])

    completed = run_bounded(
        rows,
        _extract,
        max_workers=cpu_worker_count(max_workers),
        progress_callback=_progress,
    )

    for item in completed:
        post_id = item.item[0]
        result = item.result or {}
        if item.error:
            failed += 1
            logger.warning("Audio extraction failed for %s: %s", post_id, item.error)
        elif result.get("success"):
            done += 1
        else:
            failed += 1
            logger.warning("Audio extraction failed for %s: %s", post_id, result.get("error"))

    return {"done": done, "failed": failed, "total": total}
