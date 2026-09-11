from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional

from core.clients import get_accounts_with_limits
from core.db import get_connection, init_db


PIPELINE_LOCK_NAME = "full_pipeline"
LOCK_STALE_AFTER = timedelta(hours=4)
CDN_EXPIRY_WARNING_HOURS = 18
CDN_EXPIRY_HIGH_RISK_HOURS = 36


ProgressCallback = Optional[Callable[[dict], None]]


def _now() -> datetime:
    return datetime.utcnow()


def _elapsed(started_at: float) -> float:
    return round(time.monotonic() - started_at, 1)


def _stage_result(
    stage: str,
    status: str,
    done: int = 0,
    failed: int = 0,
    total: int = 0,
    message: str = "",
    cost_usd: float = 0.0,
    errors: Optional[list] = None,
) -> dict:
    return {
        "stage": stage,
        "status": status,
        "done": done,
        "failed": failed,
        "total": total,
        "message": message,
        "cost_usd": round(cost_usd, 6),
        "errors": errors or [],
    }


def _emit(
    callback: ProgressCallback,
    started_at: float,
    stage: str,
    done: int,
    total: int,
    item: str = "",
    message: str = "",
    error: Optional[str] = None,
):
    if callback:
        callback(
            {
                "stage": stage,
                "done": done,
                "total": total,
                "item": item,
                "message": message,
                "error": error,
                "elapsed_sec": _elapsed(started_at),
            }
        )


def _row_to_lock(row) -> Optional[dict]:
    if not row:
        return None
    return {
        "lock_name": row[0],
        "run_id": row[1],
        "client_id": row[2],
        "started_at": row[3],
        "heartbeat_at": row[4],
        "stage": row[5],
    }


def get_pipeline_lock() -> Optional[dict]:
    init_db()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT lock_name, run_id, client_id, started_at, heartbeat_at, stage
            FROM pipeline_locks
            WHERE lock_name = ?
            """,
            [PIPELINE_LOCK_NAME],
        ).fetchone()
        return _row_to_lock(row)
    finally:
        conn.close()


def try_start_pipeline_run(client_id: str) -> dict:
    init_db()
    run_id = f"pipeline_{uuid.uuid4().hex[:12]}"
    now = _now()
    cutoff = now - LOCK_STALE_AFTER
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM pipeline_locks WHERE lock_name = ? AND heartbeat_at < ?",
            [PIPELINE_LOCK_NAME, cutoff],
        )
        existing = conn.execute(
            """
            SELECT lock_name, run_id, client_id, started_at, heartbeat_at, stage
            FROM pipeline_locks
            WHERE lock_name = ?
            """,
            [PIPELINE_LOCK_NAME],
        ).fetchone()
        if existing:
            return {"acquired": False, "run_id": None, "lock": _row_to_lock(existing), "error": None}

        conn.execute(
            """
            INSERT INTO pipeline_locks (
                lock_name, run_id, client_id, started_at, heartbeat_at, stage
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [PIPELINE_LOCK_NAME, run_id, client_id, now, now, "Starting"],
        )
        return {"acquired": True, "run_id": run_id, "lock": None, "error": None}
    except Exception as exc:
        return {
            "acquired": False,
            "run_id": None,
            "lock": None,
            "error": (
                "The pipeline is busy or DuckDB is accepting another write. "
                "Wait for the current operation to finish, then try again."
            ),
            "raw_error": str(exc),
        }
    finally:
        conn.close()


def update_pipeline_lock(run_id: str, stage: str):
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE pipeline_locks
            SET heartbeat_at = ?, stage = ?
            WHERE lock_name = ? AND run_id = ?
            """,
            [_now(), stage, PIPELINE_LOCK_NAME, run_id],
        )
    finally:
        conn.close()


def release_pipeline_lock(run_id: str):
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM pipeline_locks WHERE lock_name = ? AND run_id = ?",
            [PIPELINE_LOCK_NAME, run_id],
        )
    finally:
        conn.close()


def get_pending_download_expiry(client_id: str) -> dict:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.post_id, COALESCE(ca.username, p.account_id) AS username, p.scraped_at
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            WHERE cp.client_id = ?
              AND p.download_status = 'pending'
              AND p.video_url IS NOT NULL
              AND p.video_url != ''
            ORDER BY p.scraped_at ASC NULLS LAST
            """,
            [client_id],
        ).fetchall()
    finally:
        conn.close()

    now = _now()
    items = []
    for post_id, username, scraped_at in rows:
        if not scraped_at:
            continue
        age_hours = max((now - scraped_at).total_seconds() / 3600, 0)
        items.append(
            {
                "post_id": post_id,
                "username": username,
                "scraped_at": scraped_at,
                "age_hours": round(age_hours, 1),
            }
        )

    warning_items = [
        item for item in items if item["age_hours"] >= CDN_EXPIRY_WARNING_HOURS
    ]
    high_risk_items = [
        item for item in items if item["age_hours"] >= CDN_EXPIRY_HIGH_RISK_HOURS
    ]
    return {
        "pending_with_urls": len(rows),
        "warning_count": len(warning_items),
        "high_risk_count": len(high_risk_items),
        "oldest_age_hours": max((item["age_hours"] for item in items), default=0),
        "items": warning_items[:10],
    }


def get_pipeline_snapshot(client_id: str) -> dict:
    init_db()
    conn = get_connection()
    try:
        accounts = conn.execute(
            "SELECT COUNT(*) FROM client_accounts WHERE client_id = ?",
            [client_id],
        ).fetchone()[0]
        post_counts = conn.execute(
            """
            SELECT
                COUNT(*) AS total_posts,
                SUM(CASE WHEN p.download_status = 'pending' THEN 1 ELSE 0 END) AS pending_downloads,
                SUM(CASE WHEN p.download_status = 'done' THEN 1 ELSE 0 END) AS downloaded,
                SUM(CASE WHEN p.download_status = 'failed' THEN 1 ELSE 0 END) AS failed_downloads,
                SUM(CASE WHEN p.download_status = 'done'
                          AND p.local_video_path IS NOT NULL
                          AND p.local_video_path != ''
                         THEN 1 ELSE 0 END) AS local_videos,
                SUM(CASE WHEN p.local_audio_path IS NOT NULL
                          AND p.local_audio_path != ''
                         THEN 1 ELSE 0 END) AS audio_done,
                SUM(CASE WHEN p.download_status = 'done'
                          AND p.local_video_path IS NOT NULL
                          AND p.local_video_path != ''
                          AND (p.local_audio_path IS NULL OR p.local_audio_path = '')
                         THEN 1 ELSE 0 END) AS audio_pending
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()
        transcript_counts = conn.execute(
            """
            SELECT
                COUNT(DISTINCT t.post_id) AS transcribed,
                COALESCE(SUM(t.cost_usd), 0) AS transcription_cost
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()
        extraction_counts = conn.execute(
            """
            SELECT
                COUNT(DISTINCT mu.post_id) AS extracted_posts,
                COUNT(mu.unit_id) AS total_units,
                SUM(CASE WHEN mu.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded_units
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()
        generated = conn.execute(
            "SELECT COUNT(*) FROM generated_content WHERE client_id = ?",
            [client_id],
        ).fetchone()[0]
        last_scrape = conn.execute(
            """
            SELECT MAX(finished_at)
            FROM scrape_jobs
            WHERE client_id = ? AND status = 'done'
            """,
            [client_id],
        ).fetchone()[0]
    finally:
        conn.close()

    total_posts = post_counts[0] or 0
    audio_done = post_counts[5] or 0
    audio_pending = post_counts[6] or 0
    transcribed = transcript_counts[0] or 0
    extracted_posts = extraction_counts[0] or 0
    total_units = extraction_counts[1] or 0
    embedded_units = extraction_counts[2] or 0
    expiry = get_pending_download_expiry(client_id)

    return {
        "accounts": accounts,
        "scrape": {
            "accounts": accounts,
            "posts": total_posts,
            "last_finished_at": last_scrape,
        },
        "download": {
            "done": post_counts[2] or 0,
            "pending": post_counts[1] or 0,
            "failed": post_counts[3] or 0,
            "local_videos": post_counts[4] or 0,
            "expiry": expiry,
        },
        "audio": {
            "done": audio_done,
            "pending": audio_pending,
            "ready": post_counts[4] or 0,
        },
        "transcription": {
            "done": transcribed,
            "pending": max(audio_done - transcribed, 0),
            "ready": audio_done,
            "cost_usd": round(transcript_counts[1] or 0, 4),
        },
        "extraction": {
            "done": extracted_posts,
            "pending": max(transcribed - extracted_posts, 0),
            "total_units": total_units,
        },
        "embedding": {
            "done": embedded_units,
            "pending": max(total_units - embedded_units, 0),
            "total_units": total_units,
        },
        "content": {
            "generated": generated,
        },
    }


def run_scrape_stage(
    client_id: str,
    apify_token: str,
    max_items: Optional[int] = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    stage = "Scrape"
    started = time.monotonic()

    accounts = get_accounts_with_limits(client_id)
    if not accounts:
        return _stage_result(stage, "skipped", message="No competitor accounts are configured.")

    if not apify_token:
        return _stage_result(
            stage,
            "failed",
            message="Scraping needs an Apify token. Add `APIFY_TOKEN` in Settings.",
        )

    from collection.scraper import scrape_and_store

    done = failed = reels_found = reels_new = 0
    errors = []
    for index, (handle, stored_limit) in enumerate(accounts, start=1):
        limit = int(max_items or stored_limit or 30)
        _emit(progress_callback, started, stage, index - 1, len(accounts), handle, f"Scraping @{handle}")
        result = scrape_and_store(handle, apify_token, limit, client_id)
        if result.get("status") == "done":
            done += 1
            reels_found += result.get("reels_found", 0)
            reels_new += result.get("reels_new", 0)
        else:
            failed += 1
            errors.append({"account": handle, "error": result.get("error") or "Scrape failed"})
        _emit(progress_callback, started, stage, index, len(accounts), handle, f"Finished @{handle}")

    status = "failed" if failed else "done"
    return _stage_result(
        stage,
        status,
        done=done,
        failed=failed,
        total=len(accounts),
        message=f"{reels_found} reels found, {reels_new} new.",
        errors=errors,
    )


def run_download_stage(
    client_id: str,
    batch_size: int = 500,
    progress_callback: ProgressCallback = None,
) -> dict:
    stage = "Download"
    started = time.monotonic()

    from processing.download import download_pending_posts

    def _download_progress(done, total, post_id):
        _emit(progress_callback, started, stage, done, total, post_id, f"Downloading {post_id}")

    result = download_pending_posts(batch_size=batch_size, progress_callback=_download_progress, client_id=client_id)
    status = "failed" if result["total"] and result["failed"] and not result["done"] else "done"
    if result["total"] == 0:
        status = "skipped"
    return _stage_result(
        stage,
        status,
        done=result["done"],
        failed=result["failed"],
        total=result["total"],
        message="Downloaded pending media.",
    )


def run_audio_stage(client_id: str, progress_callback: ProgressCallback = None) -> dict:
    stage = "Audio"
    started = time.monotonic()

    from processing.audio import extract_audio_for_downloaded_posts

    def _audio_progress(done, total, post_id):
        _emit(progress_callback, started, stage, done, total, post_id, f"Extracting audio for {post_id}")

    result = extract_audio_for_downloaded_posts(progress_callback=_audio_progress, client_id=client_id)
    status = "failed" if result["total"] and result["failed"] and not result["done"] else "done"
    if result["total"] == 0:
        status = "skipped"
    return _stage_result(
        stage,
        status,
        done=result["done"],
        failed=result["failed"],
        total=result["total"],
        message="Prepared audio for transcription.",
    )


def run_transcription_stage(
    client_id: str,
    deepgram_api_key: str,
    provider: str = "deepgram",
    batch_size: int = 500,
    progress_callback: ProgressCallback = None,
) -> dict:
    stage = "Transcribe"
    started = time.monotonic()
    pending = get_pipeline_snapshot(client_id)["transcription"]["pending"]
    if pending == 0:
        return _stage_result(stage, "skipped", message="No audio is waiting for transcription.")

    if provider == "deepgram" and not deepgram_api_key:
        return _stage_result(
            stage,
            "failed",
            message="Transcription is not configured. Add `DEEPGRAM_API_KEY` in Settings.",
        )

    from processing.transcription_queue import run_transcription_queue

    def _transcription_progress(done, total, post_id, err):
        _emit(progress_callback, started, stage, done, total, post_id, f"Transcribing {post_id}", err)

    result = run_transcription_queue(
        deepgram_api_key,
        provider=provider,
        batch_size=batch_size,
        progress_callback=_transcription_progress,
        client_id=client_id,
    )
    status = "failed" if result["total"] and result["failed"] and not result["done"] else "done"
    if result["total"] == 0:
        status = "skipped"
    return _stage_result(
        stage,
        status,
        done=result["done"],
        failed=result["failed"],
        total=result["total"],
        message=f"Transcribed audio. Cost this run: ${result['total_cost_usd']:.4f}.",
        cost_usd=result["total_cost_usd"],
        errors=result.get("errors", []),
    )


def run_extraction_stage(
    client_id: str,
    openai_api_key: str,
    batch_size: int = 500,
    progress_callback: ProgressCallback = None,
) -> dict:
    stage = "Extract"
    started = time.monotonic()
    pending = get_pipeline_snapshot(client_id)["extraction"]["pending"]
    if pending == 0:
        return _stage_result(stage, "skipped", message="No transcripts are waiting for extraction.")

    if not openai_api_key:
        return _stage_result(
            stage,
            "failed",
            message="Extraction needs an OpenAI API key. Add `OPENAI_API_KEY` in Settings.",
        )

    from processing.extraction_queue import run_extraction_queue

    def _extraction_progress(done, total, post_id, err, n_units):
        message = f"Extracting {post_id}" if err else f"Extracted {n_units} units from {post_id}"
        _emit(progress_callback, started, stage, done, total, post_id, message, err)

    result = run_extraction_queue(
        openai_api_key,
        batch_size=batch_size,
        progress_callback=_extraction_progress,
        client_id=client_id,
    )
    status = "failed" if result["total"] and result["failed"] and not result["done"] else "done"
    if result["total"] == 0:
        status = "skipped"
    return _stage_result(
        stage,
        status,
        done=result["done"],
        failed=result["failed"],
        total=result["total"],
        message=(
            f"{result['total_units_extracted']} knowledge units extracted. "
            f"Cost this run: ${result['total_cost_usd']:.4f}."
        ),
        cost_usd=result["total_cost_usd"],
    )


def run_embedding_stage(
    client_id: str,
    openai_api_key: str,
    batch_size: int = 1000,
    progress_callback: ProgressCallback = None,
) -> dict:
    stage = "Embed"
    started = time.monotonic()
    pending = get_pipeline_snapshot(client_id)["embedding"]["pending"]
    if pending == 0:
        return _stage_result(stage, "skipped", message="No knowledge units are waiting for embeddings.")

    if not openai_api_key:
        return _stage_result(
            stage,
            "failed",
            message="Embeddings need an OpenAI API key. Add `OPENAI_API_KEY` in Settings.",
        )

    from analysis.embeddings import embed_pending_units

    def _embedding_progress(done, total, unit_id):
        _emit(progress_callback, started, stage, done, total, unit_id, f"Embedding {unit_id}")

    result = embed_pending_units(
        openai_api_key,
        batch_size=batch_size,
        progress_callback=_embedding_progress,
        client_id=client_id,
    )
    status = "failed" if result["total"] and result["failed"] and not result["done"] else "done"
    if result["total"] == 0:
        status = "skipped"
    return _stage_result(
        stage,
        status,
        done=result["done"],
        failed=result["failed"],
        total=result["total"],
        message=result.get("error") or "Embedded pending knowledge units.",
    )


def run_everything_pending(
    client_id: str,
    apify_token: str,
    deepgram_api_key: str,
    openai_api_key: str,
    scrape_max_items: Optional[int] = None,
    download_batch_size: int = 500,
    transcription_batch_size: int = 500,
    extraction_batch_size: int = 500,
    embedding_batch_size: int = 1000,
    progress_callback: ProgressCallback = None,
) -> dict:
    lock = try_start_pipeline_run(client_id)
    if not lock["acquired"]:
        return {
            "status": "busy",
            "message": lock.get("error") or "A pipeline run is already in progress.",
            "lock": lock.get("lock"),
            "results": [],
        }

    run_id = lock["run_id"]
    results = []
    try:
        stage_calls = [
            (
                "Scrape",
                lambda: run_scrape_stage(client_id, apify_token, scrape_max_items, progress_callback),
            ),
            (
                "Download",
                lambda: run_download_stage(client_id, download_batch_size, progress_callback),
            ),
            (
                "Audio",
                lambda: run_audio_stage(client_id, progress_callback),
            ),
            (
                "Transcribe",
                lambda: run_transcription_stage(
                    client_id,
                    deepgram_api_key,
                    batch_size=transcription_batch_size,
                    progress_callback=progress_callback,
                ),
            ),
            (
                "Extract",
                lambda: run_extraction_stage(
                    client_id,
                    openai_api_key,
                    batch_size=extraction_batch_size,
                    progress_callback=progress_callback,
                ),
            ),
            (
                "Embed",
                lambda: run_embedding_stage(
                    client_id,
                    openai_api_key,
                    batch_size=embedding_batch_size,
                    progress_callback=progress_callback,
                ),
            ),
        ]

        for stage, call in stage_calls:
            update_pipeline_lock(run_id, stage)
            result = call()
            results.append(result)
            if result["status"] == "failed":
                return {
                    "status": "failed",
                    "message": f"{stage} failed: {result['message']}",
                    "results": results,
                }

        return {
            "status": "done",
            "message": "Pipeline run complete.",
            "results": results,
        }
    finally:
        release_pipeline_lock(run_id)
