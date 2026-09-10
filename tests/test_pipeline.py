from datetime import datetime, timedelta

import duckdb
import pytest


@pytest.fixture
def pipeline_db(tmp_path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "pipeline.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def test_pipeline_snapshot_is_client_scoped_and_flags_expiring_downloads(pipeline_db):
    from core.clients import create_client
    from core.pipeline import get_pipeline_snapshot

    alpha_id = create_client("Alpha", "Careers")
    beta_id = create_client("Beta", "Healthcare")
    now = datetime.utcnow()
    old = now - timedelta(hours=22)

    conn = duckdb.connect(pipeline_db.DB_PATH)
    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc1','alpha_creator',NULL,NULL,NULL,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        [now, now],
    )
    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc2','beta_creator',NULL,NULL,NULL,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        [now, now],
    )
    rows = [
        ("alpha_pending", alpha_id, "acc1", "pending", old, "", None),
        ("alpha_video", alpha_id, "acc1", "done", now, "downloads/alpha_video.mp4", None),
        ("alpha_audio", alpha_id, "acc1", "done", now, "downloads/alpha_audio.mp4", "audio/alpha_audio.wav"),
        ("beta_pending", beta_id, "acc2", "pending", old, "", None),
    ]
    for post_id, client_id, account_id, status, scraped_at, video_path, audio_path in rows:
        conn.execute(
            """
            INSERT INTO posts (
                post_id, client_id, account_id, download_status, scraped_at,
                video_url, local_video_path, local_audio_path, hashtags, mentions
            )
            VALUES (?, ?, ?, ?, ?, 'https://cdn.example/video.mp4', ?, ?, [], [])
            """,
            [post_id, client_id, account_id, status, scraped_at, video_path, audio_path],
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [client_id, post_id, now],
        )

    conn.execute(
        """
        INSERT INTO transcripts (
            post_id, client_id, provider, model, transcript, language,
            confidence, duration_sec, word_count, transcribed_at, cost_usd
        )
        VALUES ('alpha_audio', ?, 'deepgram', 'nova-2', 'hello', 'en', 0.9, 5, 1, ?, 0.001)
        """,
        [alpha_id, now],
    )
    conn.execute(
        """
        INSERT INTO message_units (
            unit_id, client_id, post_id, text, claim, topic, content_type,
            confidence, extracted_at, model, embedding
        )
        VALUES ('unit_alpha', ?, 'alpha_audio', 'hello', 'claim', 'General', 'tip',
                0.9, ?, 'gpt-4o-mini', NULL)
        """,
        [alpha_id, now],
    )
    conn.close()

    alpha = get_pipeline_snapshot(alpha_id)
    beta = get_pipeline_snapshot(beta_id)

    assert alpha["scrape"]["posts"] == 3
    assert alpha["download"]["pending"] == 1
    assert alpha["download"]["done"] == 2
    assert alpha["audio"]["pending"] == 1
    assert alpha["audio"]["done"] == 1
    assert alpha["transcription"]["done"] == 1
    assert alpha["extraction"]["total_units"] == 1
    assert alpha["embedding"]["pending"] == 1
    assert alpha["download"]["expiry"]["warning_count"] == 1
    assert alpha["download"]["expiry"]["items"][0]["post_id"] == "alpha_pending"

    assert beta["scrape"]["posts"] == 1
    assert beta["download"]["pending"] == 1
    assert beta["download"]["expiry"]["items"][0]["post_id"] == "beta_pending"


def test_pipeline_lock_reports_busy_until_released(pipeline_db):
    from core.pipeline import get_pipeline_lock, release_pipeline_lock, try_start_pipeline_run

    first = try_start_pipeline_run(pipeline_db.DEFAULT_CLIENT_ID)
    assert first["acquired"] is True
    assert get_pipeline_lock()["run_id"] == first["run_id"]

    second = try_start_pipeline_run("another_client")
    assert second["acquired"] is False
    assert second["lock"]["run_id"] == first["run_id"]

    release_pipeline_lock(first["run_id"])
    assert get_pipeline_lock() is None


def test_transcription_queue_missing_key_returns_named_error(pipeline_db):
    from processing.transcription_queue import run_transcription_queue

    result = run_transcription_queue("", client_id=pipeline_db.DEFAULT_CLIENT_ID)

    assert result["done"] == 0
    assert result["failed"] == 0
    assert result["errors"]
    assert "Transcription is not configured" in result["errors"][0]["error"]
    assert "DEEPGRAM_API_KEY" in result["errors"][0]["error"]
