"""
Tests for analysis/search.py — semantic search with known vectors +
keyword fallback search.
"""
import uuid
import math
import pytest
import duckdb
from unittest.mock import patch, MagicMock
from datetime import datetime


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def search_db(tmp_path):
    """
    Set up a temp DB with two message_units that have known embeddings.
    Unit A: embedding = [1, 0, 0, ...]  (Resume topic)
    Unit B: embedding = [0, 1, 0, ...]  (Interview topic)
    """
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "search_test.duckdb")
    db_mod.init_db()

    dim = 1536
    vec_a = [1.0] + [0.0] * (dim - 1)
    vec_b = [0.0, 1.0] + [0.0] * (dim - 2)

    conn = duckdb.connect(db_mod.DB_PATH)
    now = datetime.utcnow()
    client_id = db_mod.DEFAULT_CLIENT_ID

    # Insert two creator accounts and posts first (for JOINs)
    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc1','user1',NULL,NULL,NULL,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        (now, now)
    )
    conn.execute("""
        INSERT INTO posts (post_id, client_id, account_id, engagement_rate, download_status,
                           scraped_at, hashtags, mentions, video_url)
        VALUES ('post_a', ?, 'acc1', 5.0, 'done', ?, [], [], 'http://vid_a')
    """, (client_id, now))
    conn.execute("""
        INSERT INTO posts (post_id, client_id, account_id, engagement_rate, download_status,
                           scraped_at, hashtags, mentions, video_url)
        VALUES ('post_b', ?, 'acc1', 3.0, 'done', ?, [], [], 'http://vid_b')
    """, (client_id, now))
    conn.executemany(
        "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
        [(client_id, "post_a", now), (client_id, "post_b", now)],
    )

    # Insert message units with embeddings
    uid_a = str(uuid.uuid4())
    uid_b = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO message_units
            (unit_id, client_id, post_id, text, claim, topic, content_type, confidence,
             extracted_at, model, embedding, embedded_at)
        VALUES (?, ?, 'post_a', 'Resume keyword tips', 'Keywords matter', 'Resume', 'tip',
                0.9, ?, 'gpt-4o-mini', ?::FLOAT[1536], ?)
    """, (uid_a, client_id, now, vec_a, now))
    conn.execute("""
        INSERT INTO message_units
            (unit_id, client_id, post_id, text, claim, topic, content_type, confidence,
             extracted_at, model, embedding, embedded_at)
        VALUES (?, ?, 'post_b', 'Interview STAR method', 'Structure your answers', 'Interview', 'tip',
                0.85, ?, 'gpt-4o-mini', ?::FLOAT[1536], ?)
    """, (uid_b, client_id, now, vec_b, now))

    conn.close()
    yield db_mod.DB_PATH, uid_a, uid_b
    db_mod.DB_PATH = old_path


# ── keyword_search ─────────────────────────────────────────────────────────────

class TestKeywordSearch:
    def test_finds_matching_text(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("resume")
        assert any("Resume" in r.text or "resume" in r.text.lower() for r in results)

    def test_no_match_returns_empty(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("xyzzy_impossible_match_99")
        assert results == []

    def test_topic_filter_works(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("tips", topic_filter="Resume")
        assert all(r.topic == "Resume" for r in results)

    def test_top_k_respected(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("method", top_k=1)
        assert len(results) <= 1

    def test_result_has_expected_fields(self, search_db):
        from analysis.search import keyword_search, SearchResult
        results = keyword_search("keyword")
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.unit_id is not None
        assert r.post_id is not None
        assert r.topic in ("Resume", "Interview", "General")


# ── semantic_search ───────────────────────────────────────────────────────────

class TestSemanticSearch:
    def _mock_embed(self, vec):
        """Return a mock embed_text that returns vec."""
        mock = MagicMock(return_value=vec)
        return mock

    def test_query_aligned_with_unit_a_scores_higher(self, search_db):
        """
        Query vector = [1, 0, 0, ...] should score unit A (Resume) > unit B (Interview).
        """
        from analysis.search import semantic_search

        dim = 1536
        query_vec = [1.0] + [0.0] * (dim - 1)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("resume tips", "fake_key", top_k=10)

        assert len(results) >= 2
        # First result should be unit A (identical vector → cosine = 1.0)
        assert results[0].topic == "Resume"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_topic_filter_excludes_other_topics(self, search_db):
        from analysis.search import semantic_search

        dim = 1536
        query_vec = [1.0] + [0.0] * (dim - 1)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("anything", "fake_key", topic_filter="Interview", top_k=10)

        assert all(r.topic == "Interview" for r in results)

    def test_content_type_filter_works(self, search_db):
        from analysis.search import semantic_search

        dim = 1536
        query_vec = [0.5] + [0.5] + [0.0] * (1534)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("test", "k", content_type_filter="tip", top_k=10)

        assert all(r.content_type == "tip" for r in results)

    def test_returns_search_result_objects(self, search_db):
        from analysis.search import semantic_search, SearchResult

        dim = 1536
        query_vec = [1.0] + [0.0] * (dim - 1)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("test", "k", top_k=5)

        for r in results:
            assert isinstance(r, SearchResult)
            assert r.score is not None

    def test_empty_db_returns_empty_list(self, tmp_path):
        from analysis.search import semantic_search
        import core.db as db_mod

        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "empty.duckdb")
        db_mod.init_db()

        dim = 1536
        with patch("analysis.search.embed_text", return_value=[0.0] * dim):
            results = semantic_search("anything", "key", top_k=5)

        assert results == []
        db_mod.DB_PATH = old_path
