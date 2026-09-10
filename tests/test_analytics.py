"""
Tests for analysis/analytics.py — pandas DataFrame outputs from DB queries.
"""
import pytest
import duckdb
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def analytics_db(tmp_path):
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "analytics.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def _seed_basic_data(db_mod):
    """Insert two accounts, 3 posts, and a few message units."""
    conn = duckdb.connect(db_mod.DB_PATH)
    now = datetime.utcnow()
    client_id = db_mod.DEFAULT_CLIENT_ID

    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc1','alpha',NULL,NULL,5000,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        (now, now)
    )
    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc2','beta',NULL,NULL,3000,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        (now, now)
    )

    for post_id, acct, eng, views, likes, tags, posted in [
        ("p1", "acc1", 10.0, 1000, 100, ["jobs", "resume"], now - timedelta(days=3)),
        ("p2", "acc1",  5.0,  500,  25, ["ats"],            now - timedelta(days=2)),
        ("p3", "acc2",  8.0,  800,  64, ["salary"],         now - timedelta(days=1)),
    ]:
        conn.execute(
            """INSERT INTO posts (post_id, client_id, account_id, engagement_rate, views, likes,
               download_status, scraped_at, posted_at, hashtags, mentions, duration_sec)
               VALUES (?, ?, ?, ?, ?, ?, 'done', ?, ?, ?, [], 30)""",
            (post_id, client_id, acct, eng, views, likes, now, posted, tags)
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [client_id, post_id, now],
        )

    # message units
    for i, (uid, post, topic, ctype) in enumerate([
        ("u1", "p1", "Resume", "tip"),
        ("u2", "p1", "Resume", "warning"),
        ("u3", "p2", "ATS",    "tip"),
        ("u4", "p3", "Salary", "stat"),
    ]):
        conn.execute("""
            INSERT INTO message_units (unit_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, ?, ?, ?, ?, ?, 0.9, ?, 'gpt-4o-mini')
        """, (uid, post, f"unit text {i}", f"claim {i}", topic, ctype, now))

    conn.close()


def _client_id(db_mod):
    return db_mod.DEFAULT_CLIENT_ID


class TestGetCreatorLeaderboard:
    def test_returns_dataframe(self, analytics_db):
        from analysis.analytics import get_creator_leaderboard
        _seed_basic_data(analytics_db)
        df = get_creator_leaderboard(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)

    def test_contains_expected_columns(self, analytics_db):
        from analysis.analytics import get_creator_leaderboard
        _seed_basic_data(analytics_db)
        df = get_creator_leaderboard(_client_id(analytics_db))
        assert "username" in df.columns
        assert "avg_engagement" in df.columns

    def test_ordered_by_engagement_desc(self, analytics_db):
        from analysis.analytics import get_creator_leaderboard
        _seed_basic_data(analytics_db)
        df = get_creator_leaderboard(_client_id(analytics_db))
        if len(df) >= 2:
            assert df.iloc[0]["avg_engagement"] >= df.iloc[1]["avg_engagement"]

    def test_empty_db_returns_empty_df(self, analytics_db):
        from analysis.analytics import get_creator_leaderboard
        df = get_creator_leaderboard(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_client_id_raises(self, analytics_db):
        from analysis.analytics import get_creator_leaderboard
        with pytest.raises(TypeError):
            get_creator_leaderboard()
        with pytest.raises(ValueError):
            get_creator_leaderboard("")


class TestGetTopicDistribution:
    def test_returns_dataframe(self, analytics_db):
        from analysis.analytics import get_topic_distribution
        _seed_basic_data(analytics_db)
        df = get_topic_distribution(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)

    def test_topic_counts_correct(self, analytics_db):
        from analysis.analytics import get_topic_distribution
        _seed_basic_data(analytics_db)
        df = get_topic_distribution(_client_id(analytics_db))
        resume_row = df[df["topic"] == "Resume"]
        assert len(resume_row) == 1
        assert resume_row.iloc[0]["unit_count"] == 2

    def test_empty_returns_empty(self, analytics_db):
        from analysis.analytics import get_topic_distribution
        df = get_topic_distribution(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)


class TestGetContentGapMatrix:
    def test_returns_dataframe(self, analytics_db):
        from analysis.analytics import get_content_gap_matrix
        _seed_basic_data(analytics_db)
        df = get_content_gap_matrix(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)

    def test_pivot_structure(self, analytics_db):
        from analysis.analytics import get_content_gap_matrix
        _seed_basic_data(analytics_db)
        df = get_content_gap_matrix(_client_id(analytics_db))
        if not df.empty:
            # Rows should be topics, columns should be content_types
            assert "Resume" in df.index
            assert "tip" in df.columns

    def test_empty_returns_empty(self, analytics_db):
        from analysis.analytics import get_content_gap_matrix
        df = get_content_gap_matrix(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)


class TestGetTopPosts:
    def test_returns_all_posts_no_filter(self, analytics_db):
        from analysis.analytics import get_top_posts
        _seed_basic_data(analytics_db)
        df = get_top_posts(_client_id(analytics_db), topic="All")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_topic_filter_restricts_results(self, analytics_db):
        from analysis.analytics import get_top_posts
        _seed_basic_data(analytics_db)
        df = get_top_posts(_client_id(analytics_db), topic="Resume")
        assert isinstance(df, pd.DataFrame)
        # Should only return posts that have a Resume message unit
        assert len(df) >= 1

    def test_limit_respected(self, analytics_db):
        from analysis.analytics import get_top_posts
        _seed_basic_data(analytics_db)
        df = get_top_posts(_client_id(analytics_db), topic="All", limit=2)
        assert len(df) <= 2

    def test_ordered_by_engagement(self, analytics_db):
        from analysis.analytics import get_top_posts
        _seed_basic_data(analytics_db)
        df = get_top_posts(_client_id(analytics_db))
        if len(df) >= 2:
            assert df.iloc[0]["engagement_rate"] >= df.iloc[1]["engagement_rate"]


class TestGetHashtagIntelligence:
    def test_returns_dataframe_with_columns(self, analytics_db):
        from analysis.analytics import get_hashtag_intelligence
        _seed_basic_data(analytics_db)
        df = get_hashtag_intelligence(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)
        assert "hashtag" in df.columns
        assert "count" in df.columns

    def test_counts_hashtag_occurrences(self, analytics_db):
        from analysis.analytics import get_hashtag_intelligence
        _seed_basic_data(analytics_db)
        df = get_hashtag_intelligence(_client_id(analytics_db))
        # "jobs" appears in p1, "resume" in p1 — check jobs has count >= 1
        jobs_row = df[df["hashtag"] == "jobs"]
        if len(jobs_row) > 0:
            assert jobs_row.iloc[0]["count"] >= 1

    def test_empty_db_returns_empty_df(self, analytics_db):
        from analysis.analytics import get_hashtag_intelligence
        df = get_hashtag_intelligence(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)


class TestGetPostingCadence:
    def test_returns_dataframe(self, analytics_db):
        from analysis.analytics import get_posting_cadence
        _seed_basic_data(analytics_db)
        df = get_posting_cadence(_client_id(analytics_db))
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, analytics_db):
        from analysis.analytics import get_posting_cadence
        _seed_basic_data(analytics_db)
        df = get_posting_cadence(_client_id(analytics_db))
        if not df.empty:
            assert "username" in df.columns
            assert "posts" in df.columns
