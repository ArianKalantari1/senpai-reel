"""
Tests for core/db.py — DB helper functions using a temp database.
"""
import json
import pytest
import duckdb
from datetime import datetime


@pytest.fixture
def tmp_db(tmp_path):
    """Patch DB_PATH and initialise schema in a temp file."""
    db_file = str(tmp_path / "test.duckdb")
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = db_file
    db_mod.init_db()
    yield db_file, db_mod
    db_mod.DB_PATH = old_path


# ── init_db creates expected tables ──────────────────────────────────────────

class TestInitDb:
    EXPECTED_TABLES = {
        "clients", "client_accounts", "client_posts",
        "profiles", "raw_scrapes", "reels", "comments", "tagged_users",
        "creator_accounts", "posts", "scrape_jobs", "transcripts",
        "transcript_words", "message_units", "generated_content",
    }

    def test_all_tables_created(self, tmp_db):
        db_file, _ = tmp_db
        conn = duckdb.connect(db_file)
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        conn.close()
        assert self.EXPECTED_TABLES.issubset(tables)

    def test_init_db_is_idempotent(self, tmp_db):
        """Calling init_db() a second time should not raise."""
        _, db_mod = tmp_db
        db_mod.init_db()  # second call — must not crash

    def test_demo_client_seeded(self, tmp_db):
        db_file, db_mod = tmp_db
        conn = duckdb.connect(db_file)
        client = conn.execute(
            "SELECT name, niche FROM clients WHERE client_id = ?",
            [db_mod.DEFAULT_CLIENT_ID],
        ).fetchone()
        accounts = conn.execute(
            "SELECT COUNT(*) FROM client_accounts WHERE client_id = ?",
            [db_mod.DEFAULT_CLIENT_ID],
        ).fetchone()[0]
        conn.close()
        assert client == (db_mod.DEFAULT_CLIENT_NAME, db_mod.DEFAULT_CLIENT_NICHE)
        assert accounts > 0

    def test_existing_posts_are_backfilled_to_demo_client(self, tmp_path):
        import core.db as db_mod

        db_file = str(tmp_path / "legacy.duckdb")
        conn = duckdb.connect(db_file)
        now = datetime.utcnow()
        conn.execute("""
            CREATE TABLE posts (
                post_id TEXT PRIMARY KEY,
                account_id TEXT,
                scraped_at TIMESTAMP,
                download_status TEXT,
                engagement_rate DOUBLE
            )
        """)
        conn.execute(
            "INSERT INTO posts VALUES ('legacy_post', 'acc1', ?, 'done', 7.5)",
            [now],
        )
        conn.close()

        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = db_file
        try:
            db_mod.init_db()
            conn = duckdb.connect(db_file)
            post_client = conn.execute(
                "SELECT client_id FROM posts WHERE post_id = 'legacy_post'"
            ).fetchone()[0]
            link = conn.execute(
                """
                SELECT COUNT(*)
                FROM client_posts
                WHERE client_id = ? AND post_id = 'legacy_post'
                """,
                [db_mod.DEFAULT_CLIENT_ID],
            ).fetchone()[0]
            row_count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            conn.close()
        finally:
            db_mod.DB_PATH = old_path

        assert post_client == db_mod.DEFAULT_CLIENT_ID
        assert link == 1
        assert row_count == 1


# ── start_scrape_job / finish_scrape_job ─────────────────────────────────────

class TestScrapeJobs:
    def test_start_creates_running_job(self, tmp_db):
        _, db_mod = tmp_db
        job_id = db_mod.start_scrape_job("user1")
        assert job_id is not None

        conn = duckdb.connect(db_mod.DB_PATH)
        row = conn.execute(
            "SELECT status, username FROM scrape_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "running"
        assert row[1] == "user1"

    def test_finish_updates_status(self, tmp_db):
        _, db_mod = tmp_db
        job_id = db_mod.start_scrape_job("user2")
        db_mod.finish_scrape_job(job_id, 10, 5, "done")

        conn = duckdb.connect(db_mod.DB_PATH)
        row = conn.execute(
            "SELECT status, reels_found, reels_new FROM scrape_jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "done"
        assert row[1] == 10
        assert row[2] == 5

    def test_finish_with_error(self, tmp_db):
        _, db_mod = tmp_db
        job_id = db_mod.start_scrape_job("user3")
        db_mod.finish_scrape_job(job_id, 0, 0, "failed", "timeout")

        conn = duckdb.connect(db_mod.DB_PATH)
        row = conn.execute(
            "SELECT status, error_msg FROM scrape_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "failed"
        assert row[1] == "timeout"

    def test_get_scrape_history_returns_list(self, tmp_db):
        _, db_mod = tmp_db
        job_id = db_mod.start_scrape_job("user4")
        db_mod.finish_scrape_job(job_id, 3, 2, "done")
        history = db_mod.get_scrape_history()
        assert isinstance(history, list)
        assert len(history) >= 1


# ── get_posts_stats ───────────────────────────────────────────────────────────

class TestGetPostsStats:
    def test_empty_returns_zeros(self, tmp_db):
        _, db_mod = tmp_db
        stats = db_mod.get_posts_stats()
        assert stats["total_posts"] == 0
        assert stats["accounts"] == 0
        assert stats["downloaded"] == 0

    def test_counts_posts_correctly(self, tmp_db):
        _, db_mod = tmp_db
        # Insert two posts directly
        conn = duckdb.connect(db_mod.DB_PATH)
        now = datetime.utcnow()
        for i, post_id in enumerate(["p1", "p2"]):
            conn.execute(
                """INSERT INTO posts (post_id, client_id, account_id, engagement_rate, download_status,
                   scraped_at, hashtags, mentions)
                   VALUES (?, ?, 'acc1', 5.0, 'pending', ?, [], [])""",
                (post_id, db_mod.DEFAULT_CLIENT_ID, now)
            )
            conn.execute(
                "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
                [db_mod.DEFAULT_CLIENT_ID, post_id, now],
            )
        conn.execute(
            "UPDATE posts SET download_status = 'done' WHERE post_id = 'p1'"
        )
        conn.close()

        stats = db_mod.get_posts_stats()
        assert stats["total_posts"] == 2
        assert stats["downloaded"] == 1


# ── _extract_hashtags / _extract_mentions / _clean_caption ───────────────────

class TestCaptionHelpers:
    def test_extract_hashtags(self, tmp_db):
        _, db_mod = tmp_db
        tags = db_mod._extract_hashtags("Get hired #jobs #resume tips #ATS")
        assert set(tags) == {"jobs", "resume", "ATS"}

    def test_extract_hashtags_empty(self, tmp_db):
        _, db_mod = tmp_db
        assert db_mod._extract_hashtags(None) == []
        assert db_mod._extract_hashtags("no hashtags here") == []

    def test_extract_mentions(self, tmp_db):
        _, db_mod = tmp_db
        mentions = db_mod._extract_mentions("DM @recruiter or @careers_au")
        assert set(mentions) == {"recruiter", "careers_au"}

    def test_clean_caption(self, tmp_db):
        _, db_mod = tmp_db
        cleaned = db_mod._clean_caption("Get hired #jobs #resume via @recruiter")
        assert "#jobs" not in cleaned
        assert "@recruiter" not in cleaned
        assert "Get hired" in cleaned

    def test_clean_caption_none(self, tmp_db):
        _, db_mod = tmp_db
        assert db_mod._clean_caption(None) is None


# ── get_connection ─────────────────────────────────────────────────────────────

class TestGetConnection:
    def test_returns_connection(self, tmp_db):
        _, db_mod = tmp_db
        conn = db_mod.get_connection()
        assert conn is not None
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        assert "posts" in tables
        conn.close()
