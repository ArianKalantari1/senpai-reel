"""
Tests for processing/extraction_queue.py — stats and batch processing.
"""
import json
import uuid
import threading
import time
import pytest
import duckdb
from unittest.mock import patch, MagicMock
from datetime import datetime


@pytest.fixture
def queue_db(tmp_path):
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "queue.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def _seed_transcript(db_mod, post_id="post_t1", transcript="Resume tips for job seekers. Apply these now."):
    conn = duckdb.connect(db_mod.DB_PATH)
    now = datetime.utcnow()
    client_id = db_mod.DEFAULT_CLIENT_ID
    conn.execute("""
        INSERT INTO posts (post_id, client_id, account_id, engagement_rate, download_status,
            scraped_at, hashtags, mentions)
        VALUES (?, ?, 'acc1', 0, 'done', ?, [], [])
    """, (post_id, client_id, now))
    conn.execute(
        "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
        [client_id, post_id, now],
    )
    conn.execute("""
        INSERT INTO transcripts (post_id, client_id, provider, model, transcript, language,
            confidence, duration_sec, word_count, transcribed_at, cost_usd)
        VALUES (?, ?, 'deepgram', 'nova-2', ?, 'en', 0.95, 30.0, 8, ?, 0.003)
    """, (post_id, client_id, transcript, now))
    conn.close()


# ── get_extraction_stats ──────────────────────────────────────────────────────

class TestGetExtractionStats:
    def test_empty_returns_zeros(self, queue_db):
        from processing.extraction_queue import get_extraction_stats
        stats = get_extraction_stats()
        assert stats["total_transcripts"] == 0
        assert stats["extracted_posts"] == 0
        assert stats["pending"] == 0
        assert stats["total_units"] == 0
        assert stats["total_cost_usd"] >= 0.0

    def test_counts_transcript(self, queue_db):
        from processing.extraction_queue import get_extraction_stats
        _seed_transcript(queue_db, "t1")
        stats = get_extraction_stats()
        assert stats["total_transcripts"] == 1
        assert stats["pending"] == 1

    def test_pending_decreases_after_extraction(self, queue_db):
        from processing.extraction_queue import get_extraction_stats
        _seed_transcript(queue_db, "t2")

        # Add a message_unit for t2
        conn = duckdb.connect(queue_db.DB_PATH)
        conn.execute("""
            INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, ?, 't2', 'tip', 'claim', 'Resume', 'tip', 0.9, CURRENT_TIMESTAMP, 'gpt-4o-mini')
        """, (str(uuid.uuid4()), queue_db.DEFAULT_CLIENT_ID))
        conn.close()

        stats = get_extraction_stats()
        assert stats["extracted_posts"] == 1
        assert stats["pending"] == 0

    def test_returns_dict_keys(self, queue_db):
        from processing.extraction_queue import get_extraction_stats
        stats = get_extraction_stats()
        assert "total_transcripts" in stats
        assert "extracted_posts" in stats
        assert "pending" in stats
        assert "total_units" in stats
        assert "total_cost_usd" in stats


# ── run_extraction_queue ──────────────────────────────────────────────────────

def _mock_extract_response(num_units=2):
    """Return a mock for extract_message_units."""
    from analysis.extraction import MessageUnit
    units = [
        MessageUnit(
            unit_id=str(uuid.uuid4()),
            post_id="post_x",
            text=f"tip {i}",
            claim=f"claim {i}",
            advice=None,
            topic="Resume",
            subtopic=None,
            content_type="tip",
            confidence=0.9,
            extracted_at=datetime.utcnow(),
        )
        for i in range(num_units)
    ]
    return units, 0.0005


def _mock_units_for(post_id, num_units=1):
    from analysis.extraction import MessageUnit

    return [
        MessageUnit(
            unit_id=str(uuid.uuid4()),
            post_id=post_id,
            text=f"tip {i}",
            claim=f"claim {i}",
            advice=None,
            topic="Resume",
            subtopic=None,
            content_type="tip",
            confidence=0.9,
            extracted_at=datetime.utcnow(),
        )
        for i in range(num_units)
    ]


class TestRunExtractionQueue:
    def test_no_transcripts_returns_zero(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        result = run_extraction_queue("fake_key")
        assert result["total"] == 0
        assert result["done"] == 0

    def test_processes_pending_transcript(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        _seed_transcript(queue_db, "proc1", "A " * 50)  # long enough for extraction

        with patch("processing.extraction_queue.extract_message_units", return_value=_mock_extract_response(2)):
            result = run_extraction_queue("fake_key")

        assert result["total"] == 1
        assert result["done"] == 1
        assert result["total_units_extracted"] == 2

    def test_already_extracted_post_skipped(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        _seed_transcript(queue_db, "skip1")

        # Pre-seed a message_unit so this post is already extracted
        conn = duckdb.connect(queue_db.DB_PATH)
        conn.execute("""
            INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, ?, 'skip1', 'text', 'claim', 'Resume', 'tip', 0.9, CURRENT_TIMESTAMP, 'gpt-4o-mini')
        """, (str(uuid.uuid4()), queue_db.DEFAULT_CLIENT_ID))
        conn.close()

        with patch("processing.extraction_queue.extract_message_units") as mock_extract:
            result = run_extraction_queue("fake_key")

        mock_extract.assert_not_called()
        assert result["total"] == 0

    def test_exception_counts_as_failed(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        _seed_transcript(queue_db, "fail1", "A " * 50)

        with patch("processing.extraction_queue.extract_message_units",
                   side_effect=Exception("API error")):
            result = run_extraction_queue("fake_key")

        assert result["failed"] == 1
        assert result["done"] == 0

    def test_progress_callback_called(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        _seed_transcript(queue_db, "cb1", "A " * 50)

        calls = []
        with patch("processing.extraction_queue.extract_message_units",
                   return_value=_mock_extract_response(1)):
            run_extraction_queue("fake_key", progress_callback=lambda *a: calls.append(a))

        assert len(calls) == 1

    def test_batch_size_limits_processing(self, queue_db):
        from processing.extraction_queue import run_extraction_queue
        for i in range(5):
            _seed_transcript(queue_db, f"batch_{i}", "A " * 50)

        with patch("processing.extraction_queue.extract_message_units",
                   return_value=_mock_extract_response(1)):
            result = run_extraction_queue("fake_key", batch_size=3)

        assert result["total"] == 3  # only 3 of 5 processed

    def test_partial_failure_keeps_successful_item_writes(self, queue_db):
        from processing.extraction_queue import run_extraction_queue

        _seed_transcript(queue_db, "ok_1", "A " * 50)
        _seed_transcript(queue_db, "fail_1", "FAIL")
        _seed_transcript(queue_db, "ok_2", "B " * 50)

        def fake_extract(transcript, post_id, api_key):
            if transcript == "FAIL":
                raise RuntimeError("API error")
            return _mock_units_for(post_id), 0.0005

        with patch("processing.extraction_queue.extract_message_units", side_effect=fake_extract):
            result = run_extraction_queue("fake_key", max_workers=3)

        assert result["total"] == 3
        assert result["done"] == 2
        assert result["failed"] == 1

        conn = duckdb.connect(queue_db.DB_PATH)
        saved_posts = conn.execute(
            "SELECT DISTINCT post_id FROM message_units ORDER BY post_id"
        ).fetchall()
        failed_cost = conn.execute(
            "SELECT extraction_cost_usd FROM transcripts WHERE post_id = 'fail_1'"
        ).fetchone()[0]
        conn.close()

        assert [row[0] for row in saved_posts] == ["ok_1", "ok_2"]
        assert failed_cost is None

    def test_parallel_extraction_serializes_saves(self, queue_db):
        from processing.extraction_queue import run_extraction_queue

        for i in range(5):
            _seed_transcript(queue_db, f"serial_{i}", "A " * 50)

        active_writes = 0
        max_active_writes = 0
        lock = threading.Lock()

        def fake_extract(transcript, post_id, api_key):
            return _mock_units_for(post_id), 0.0005

        def slow_save(units, client_id):
            nonlocal active_writes, max_active_writes
            with lock:
                active_writes += 1
                max_active_writes = max(max_active_writes, active_writes)
            time.sleep(0.02)
            with lock:
                active_writes -= 1

        with (
            patch("processing.extraction_queue.extract_message_units", side_effect=fake_extract),
            patch("processing.extraction_queue.save_message_units", side_effect=slow_save),
        ):
            result = run_extraction_queue("fake_key", max_workers=5)

        assert result["done"] == 5
        assert max_active_writes == 1
