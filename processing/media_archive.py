"""Keyframe extraction and source video archiving."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from core.db import DEFAULT_CLIENT_ID, get_connection, init_db
from processing.concurrency import run_db_write

logger = logging.getLogger(__name__)

ARCHIVE_ENV = "SENPAI_ARCHIVE_DIR"
KEYFRAME_HWACCEL_ENV = "SENPAI_FFMPEG_HWACCEL"
KEYFRAMES_DIR = Path("keyframes")
DEFAULT_KEYFRAME_COUNT = 12


def configured_archive_dir(value: Optional[str] = None) -> Optional[Path]:
    raw = os.getenv(ARCHIVE_ENV, "") if value is None else value
    raw = raw.strip() if raw else ""
    return Path(raw).expanduser() if raw else None


def _safe_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe or "unknown"


def _count_keyframes(path: str | Path | None) -> int:
    if not path:
        return 0
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return 0
    return len([p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def keyframes_available(path: str | Path | None) -> bool:
    return _count_keyframes(path) > 0


def _probe_duration(video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def extract_keyframes(
    video_path: str,
    post_id: str,
    output_dir: str = "keyframes",
    target_count: int = DEFAULT_KEYFRAME_COUNT,
    hwaccel: Optional[str] = None,
) -> dict:
    """Extract a compact local keyframe set for later visual/OCR work."""

    src = Path(video_path)
    if not src.exists():
        return {"success": False, "path": None, "count": None, "error": f"Video not found: {video_path}"}

    frame_count = max(int(target_count or DEFAULT_KEYFRAME_COUNT), 1)
    dest_dir = Path(output_dir) / _safe_segment(post_id)
    existing = _count_keyframes(dest_dir)
    if existing:
        return {"success": True, "path": str(dest_dir), "count": existing, "error": None}

    dest_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(src)
    fps_expr = f"{frame_count}/{duration:.3f}" if duration else "1/3"
    vf = f"fps={fps_expr},scale='min(540,iw)':-2"
    selected_hwaccel = hwaccel
    if selected_hwaccel is None:
        selected_hwaccel = os.getenv(KEYFRAME_HWACCEL_ENV, "").strip() or None

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if selected_hwaccel:
        cmd.extend(["-hwaccel", selected_hwaccel])
    cmd.extend(
        [
            "-i",
            str(src),
            "-vf",
            vf,
            "-frames:v",
            str(frame_count),
            "-q:v",
            "4",
            str(dest_dir / "frame_%03d.jpg"),
        ]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"success": False, "path": None, "count": None, "error": "ffmpeg keyframe extraction timed out"}
    except FileNotFoundError:
        return {"success": False, "path": None, "count": None, "error": "ffmpeg not found"}

    count = _count_keyframes(dest_dir)
    if result.returncode != 0 or count == 0:
        error = result.stderr[-300:] if result.stderr else "No keyframes were written"
        return {"success": False, "path": None, "count": None, "error": error}

    return {"success": True, "path": str(dest_dir), "count": count, "error": None}


def record_keyframes(post_id: str, keyframes_dir: str, count: int):
    run_db_write(_write_keyframes, post_id, keyframes_dir, count)


def _write_keyframes(post_id: str, keyframes_dir: str, count: int):
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE posts
            SET keyframes_dir = ?,
                keyframes_extracted_at = CURRENT_TIMESTAMP,
                keyframe_count = ?
            WHERE post_id = ?
            """,
            [keyframes_dir, count, post_id],
        )
    finally:
        conn.close()


def _post_archive_state(post_id: str, client_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                p.local_video_path,
                p.archived_video_path,
                p.local_audio_path,
                p.keyframes_dir,
                p.keyframe_count,
                t.post_id IS NOT NULL AS has_transcript
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            LEFT JOIN transcripts t ON t.post_id = p.post_id
            WHERE p.post_id = ? AND cp.client_id = ?
            """,
            [post_id, client_id],
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    return {
        "local_video_path": row[0],
        "archived_video_path": row[1],
        "local_audio_path": row[2],
        "keyframes_dir": row[3],
        "keyframe_count": row[4],
        "has_transcript": bool(row[5]),
    }


def _mark_archived(post_id: str, archived_path: str):
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE posts
            SET local_video_path = NULL,
                archived_video_path = ?,
                video_archived_at = CURRENT_TIMESTAMP
            WHERE post_id = ?
            """,
            [archived_path, post_id],
        )
    finally:
        conn.close()


def _archive_destination(root: Path, client_id: str, src: Path) -> Path:
    dest_dir = root / _safe_segment(client_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    stem = src.stem
    suffix = src.suffix
    for index in range(2, 10_000):
        candidate = dest_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose an archive path for {src.name}")


def _existing_archive_match(root: Path, client_id: str, src: Path) -> Optional[Path]:
    """Find where a previously moved source file landed without touching files."""

    dest_dir = root / _safe_segment(client_id)
    if not dest_dir.exists() or not dest_dir.is_dir():
        return None

    exact = dest_dir / src.name
    if exact.exists() and exact.is_file():
        return exact

    pattern = re.compile(rf"^{re.escape(src.stem)}_(\d+){re.escape(src.suffix)}$")
    candidates = []
    for candidate in dest_dir.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.match(candidate.name)
        if match:
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def reconcile_orphaned_archives(
    client_id: str = DEFAULT_CLIENT_ID,
    archive_dir: Optional[str] = None,
    ensure_schema: bool = True,
) -> dict:
    """Record archive paths for videos moved before their DB update completed."""

    if ensure_schema:
        init_db()

    root = configured_archive_dir(archive_dir)
    if root is None:
        return {
            "status": "skipped",
            "reason": "archive_unset",
            "reconciled": 0,
            "unmatched": 0,
            "total": 0,
            "results": [],
        }
    if not root.exists() or not root.is_dir():
        return {
            "status": "skipped",
            "reason": "archive_unreachable",
            "reconciled": 0,
            "unmatched": 0,
            "total": 0,
            "results": [],
        }

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.post_id, p.local_video_path
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            JOIN transcripts t ON t.post_id = p.post_id
            WHERE cp.client_id = ?
              AND p.archived_video_path IS NULL
              AND p.local_video_path IS NOT NULL
              AND p.local_video_path != ''
              AND p.local_audio_path IS NOT NULL
              AND p.local_audio_path != ''
              AND p.keyframes_dir IS NOT NULL
              AND p.keyframes_dir != ''
            """,
            [client_id],
        ).fetchall()
    finally:
        conn.close()

    reconciled = unmatched = 0
    results = []
    for post_id, local_video_path in rows:
        src = Path(local_video_path)
        if src.exists():
            continue

        match = _existing_archive_match(root, client_id, src)
        if match is None:
            unmatched += 1
            results.append({"post_id": post_id, "status": "unmatched", "path": None})
            continue

        run_db_write(_mark_archived, post_id, str(match))
        reconciled += 1
        results.append({"post_id": post_id, "status": "reconciled", "path": str(match)})

    return {
        "status": "done",
        "reason": None,
        "reconciled": reconciled,
        "unmatched": unmatched,
        "total": reconciled + unmatched,
        "results": results,
    }


def archive_video_if_ready(
    post_id: str,
    client_id: str = DEFAULT_CLIENT_ID,
    archive_dir: Optional[str] = None,
    ensure_schema: bool = True,
) -> dict:
    """Archive a source video only after all derived local assets exist."""

    if ensure_schema:
        init_db()
    state = _post_archive_state(post_id, client_id)
    if not state:
        return {"status": "skipped", "reason": "post_not_found", "post_id": post_id}
    if state["archived_video_path"]:
        return {
            "status": "archived",
            "reason": "already_archived",
            "post_id": post_id,
            "path": state["archived_video_path"],
        }
    if not state["local_audio_path"]:
        return {"status": "skipped", "reason": "audio_missing", "post_id": post_id}
    if not state["has_transcript"]:
        return {"status": "skipped", "reason": "transcript_missing", "post_id": post_id}
    if not state["local_video_path"]:
        return {"status": "skipped", "reason": "source_path_missing", "post_id": post_id}

    src = Path(state["local_video_path"])
    if not src.exists():
        return {"status": "skipped", "reason": "source_file_missing", "post_id": post_id}

    if not keyframes_available(state["keyframes_dir"]):
        keyframes = extract_keyframes(str(src), post_id)
        if not keyframes["success"]:
            return {
                "status": "skipped",
                "reason": "keyframes_failed",
                "post_id": post_id,
                "error": keyframes["error"],
            }
        record_keyframes(post_id, keyframes["path"], keyframes["count"])

    root = configured_archive_dir(archive_dir)
    if root is None:
        return {"status": "skipped", "reason": "archive_unset", "post_id": post_id}
    if not root.exists() or not root.is_dir():
        return {"status": "skipped", "reason": "archive_unreachable", "post_id": post_id}

    try:
        dest = _archive_destination(root, client_id, src)
        if src.resolve() == dest.resolve():
            return {"status": "skipped", "reason": "archive_matches_source", "post_id": post_id}
        shutil.move(str(src), str(dest))
        run_db_write(_mark_archived, post_id, str(dest))
    except Exception as exc:
        logger.warning("Video archive skipped for %s: %s", post_id, exc)
        return {"status": "skipped", "reason": "archive_failed", "post_id": post_id, "error": str(exc)}

    return {"status": "archived", "reason": None, "post_id": post_id, "path": str(dest)}


def archive_ready_videos(
    client_id: str = DEFAULT_CLIENT_ID,
    archive_dir: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """Archive all currently eligible videos for one client."""

    init_db()
    reconcile_result = reconcile_orphaned_archives(
        client_id=client_id,
        archive_dir=archive_dir,
        ensure_schema=False,
    )
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.post_id
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            JOIN transcripts t ON t.post_id = p.post_id
            WHERE cp.client_id = ?
              AND p.archived_video_path IS NULL
              AND p.local_video_path IS NOT NULL
              AND p.local_video_path != ''
              AND p.local_audio_path IS NOT NULL
              AND p.local_audio_path != ''
            """,
            [client_id],
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    archived = skipped = 0
    results = []
    for index, (post_id,) in enumerate(rows, start=1):
        result = archive_video_if_ready(
            post_id,
            client_id=client_id,
            archive_dir=archive_dir,
            ensure_schema=False,
        )
        results.append(result)
        if result["status"] == "archived":
            archived += 1
        else:
            skipped += 1
        if progress_callback:
            progress_callback(index, total, post_id, result)

    return {
        "archived": archived,
        "skipped": skipped,
        "total": total,
        "results": results,
        "reconciled": reconcile_result["reconciled"],
        "reconcile": reconcile_result,
    }
