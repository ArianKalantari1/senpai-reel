"""
Tests for collection/scraper.py — deduplication logic and Apify response parsing.
All HTTP calls are mocked; DB uses a temp file so tests are isolated.
"""
import json
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import duckdb


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Patch DB_PATH in every affected module to point to a temp DuckDB file."""
    db_file = str(tmp_path / "test.duckdb")
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = db_file
    db_mod.init_db()
    yield db_file
    db_mod.DB_PATH = old_path


# ── Sample Apify item ─────────────────────────────────────────────────────────

def make_apify_item(
    short_code="ABC123",
    reel_id="111",
    username="testuser",
    likes=100,
    views=1000,
):
    return {
        "id": reel_id,
        "shortCode": short_code,
        "caption": "Test caption #jobs @recruiter",
        "likesCount": likes,
        "videoViewCount": views,
        "commentsCount": 5,
        "videoUrl": "https://example.com/video.mp4",
        "audioUrl": "https://example.com/audio.mp3",
        "images": ["https://example.com/thumb.jpg"],
        "timestamp": "2025-01-15T10:00:00Z",
        "videoDuration": 30.0,
        "isPinned": False,
        "isSponsored": False,
        "ownerUsername": username,
        "ownerId": "999",
    }


# ── save_raw_scrape deduplication ─────────────────────────────────────────────

class TestSaveRawScrapeDedup:
    def test_new_item_is_saved(self, tmp_db):
        from core.db import save_raw_scrape
        items = [make_apify_item("UNIQUE1")]
        result = save_raw_scrape("testuser", items)
        assert "1 new" in result

        conn = duckdb.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM raw_scrapes").fetchone()[0]
        conn.close()
        assert count == 1

    def test_duplicate_shortcode_is_skipped(self, tmp_db):
        from core.db import save_raw_scrape
        items = [make_apify_item("DUP1")]
        save_raw_scrape("testuser", items)
        result = save_raw_scrape("testuser", items)
        assert "0 new" in result
        assert "1 already existed" in result

        conn = duckdb.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM raw_scrapes").fetchone()[0]
        conn.close()
        assert count == 1

    def test_mixed_new_and_duplicate(self, tmp_db):
        from core.db import save_raw_scrape
        save_raw_scrape("testuser", [make_apify_item("FIRST")])
        items = [make_apify_item("FIRST"), make_apify_item("SECOND", reel_id="222")]
        result = save_raw_scrape("testuser", items)
        assert "1 new" in result
        assert "1 already existed" in result

    def test_empty_list_returns_safely(self, tmp_db):
        from core.db import save_raw_scrape
        result = save_raw_scrape("testuser", [])
        assert "0 new" in result


# ── upsert_post ──────────────────────────────────────────────────────────────

class TestUpsertPost:
    def test_new_post_returns_true(self, tmp_db):
        from core.db import upsert_post
        item = make_apify_item("NEW1")
        is_new = upsert_post("acct_001", item)
        assert is_new is True

    def test_duplicate_post_returns_false(self, tmp_db):
        from core.db import upsert_post
        item = make_apify_item("DUP2")
        upsert_post("acct_001", item)
        is_new = upsert_post("acct_001", item)
        assert is_new is False

    def test_hashtags_extracted(self, tmp_db):
        from core.db import upsert_post
        item = make_apify_item("HASH1")
        item["caption"] = "Get hired #jobs #resume"
        upsert_post("acct_001", item)
        conn = duckdb.connect(tmp_db)
        row = conn.execute(
            "SELECT hashtags FROM posts WHERE post_id = 'HASH1'"
        ).fetchone()
        conn.close()
        assert "jobs" in row[0]
        assert "resume" in row[0]

    def test_engagement_rate_calculated(self, tmp_db):
        from core.db import upsert_post
        item = make_apify_item("ENG1", likes=200, views=1000)
        upsert_post("acct_001", item)
        conn = duckdb.connect(tmp_db)
        row = conn.execute(
            "SELECT engagement_rate FROM posts WHERE post_id = 'ENG1'"
        ).fetchone()
        conn.close()
        assert abs(row[0] - 20.0) < 0.01

    def test_zero_views_no_division_error(self, tmp_db):
        from core.db import upsert_post
        item = make_apify_item("ZERO1", likes=0, views=0)
        is_new = upsert_post("acct_001", item)
        assert is_new is True


# ── upsert_creator_account ────────────────────────────────────────────────────

class TestUpsertCreatorAccount:
    def test_new_creator_is_inserted(self, tmp_db):
        from core.db import upsert_creator_account
        item = make_apify_item()
        account_id = upsert_creator_account("testuser", item)
        assert account_id == "999"

        conn = duckdb.connect(tmp_db)
        row = conn.execute(
            "SELECT username FROM creator_accounts WHERE account_id = '999'"
        ).fetchone()
        conn.close()
        assert row[0] == "testuser"

    def test_existing_creator_updates_last_scraped(self, tmp_db):
        from core.db import upsert_creator_account
        item = make_apify_item()
        upsert_creator_account("testuser", item)
        upsert_creator_account("testuser", item)  # second call → UPDATE
        conn = duckdb.connect(tmp_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM creator_accounts WHERE username = 'testuser'"
        ).fetchone()[0]
        conn.close()
        assert count == 1   # only one row, not two


# ── process_and_store ─────────────────────────────────────────────────────────

class TestProcessAndStore:
    def test_empty_items_returns_zero_zero(self, tmp_db):
        from core.db import start_scrape_job
        from collection.scraper import process_and_store
        job_id = start_scrape_job("emptyuser")
        found, new = process_and_store("emptyuser", [], job_id)
        assert found == 0
        assert new == 0

    def test_new_items_counted(self, tmp_db):
        from core.db import start_scrape_job
        from collection.scraper import process_and_store
        job_id = start_scrape_job("testuser")
        items = [make_apify_item("SC1"), make_apify_item("SC2", reel_id="222")]
        found, new = process_and_store("testuser", items, job_id)
        assert found == 2
        assert new == 2

    def test_second_scrape_counts_zero_new(self, tmp_db):
        from core.db import start_scrape_job
        from collection.scraper import process_and_store
        items = [make_apify_item("SC3")]
        job1 = start_scrape_job("testuser")
        process_and_store("testuser", items, job1)
        job2 = start_scrape_job("testuser")
        found, new = process_and_store("testuser", items, job2)
        assert found == 1
        assert new == 0


# ── ScraperError on HTTP failure ──────────────────────────────────────────────

class TestScraperError:
    def test_auth_error_raises_scraper_error(self, tmp_db):
        from collection.scraper import scrape_account, ScraperError
        import requests as req_lib

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        http_err = req_lib.exceptions.HTTPError("401 Unauthorized")
        mock_resp.raise_for_status.side_effect = http_err

        with patch("collection.sources.apify.requests.post", return_value=mock_resp):
            with pytest.raises(ScraperError, match="Auth error"):
                scrape_account("baduser", "fake_token", max_items=5, retries=0)

    def test_successful_scrape_returns_list(self, tmp_db):
        from collection.scraper import scrape_account

        items = [make_apify_item("X1"), make_apify_item("X2", reel_id="222")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = items

        with patch("collection.sources.apify.requests.post", return_value=mock_resp):
            result, job_id = scrape_account("gooduser", "fake_token", max_items=5, retries=0)

        assert len(result) == 2
        assert job_id is not None
