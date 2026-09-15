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
    yield db_mod.DB_PATH, uid_a, uid_b, client_id
    db_mod.DB_PATH = old_path


def _client_id(search_db):
    return search_db[3]


# ── keyword_search ─────────────────────────────────────────────────────────────

class TestKeywordSearch:
    def test_finds_matching_text(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("resume", _client_id(search_db))
        assert any("Resume" in r.text or "resume" in r.text.lower() for r in results)

    def test_no_match_returns_empty(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("xyzzy_impossible_match_99", _client_id(search_db))
        assert results == []

    def test_topic_filter_works(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("tips", _client_id(search_db), topic_filter="Resume")
        assert all(r.topic == "Resume" for r in results)

    def test_top_k_respected(self, search_db):
        from analysis.search import keyword_search
        results = keyword_search("method", _client_id(search_db), top_k=1)
        assert len(results) <= 1

    def test_result_has_expected_fields(self, search_db):
        from analysis.search import keyword_search, SearchResult
        results = keyword_search("keyword", _client_id(search_db))
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.unit_id == search_db[1]
        assert r.post_id == "post_a"
        assert r.username == "user1"
        assert r.topic == "Resume"
        assert r.content_type == "tip"
        assert r.text == "Resume keyword tips"
        assert r.claim == "Keywords matter"

    def test_missing_client_id_raises(self, search_db):
        from analysis.search import keyword_search
        with pytest.raises(TypeError):
            keyword_search("resume")
        with pytest.raises(ValueError):
            keyword_search("resume", "")


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
            results = semantic_search("resume tips", "fake_key", _client_id(search_db), top_k=10)

        assert len(results) >= 2
        # First result should be unit A (identical vector → cosine = 1.0)
        assert results[0].topic == "Resume"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_topic_filter_excludes_other_topics(self, search_db):
        from analysis.search import semantic_search

        dim = 1536
        query_vec = [1.0] + [0.0] * (dim - 1)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search(
                "anything",
                "fake_key",
                _client_id(search_db),
                topic_filter="Interview",
                top_k=10,
            )

        assert all(r.topic == "Interview" for r in results)

    def test_content_type_filter_works(self, search_db):
        from analysis.search import semantic_search

        dim = 1536
        query_vec = [0.5] + [0.5] + [0.0] * (1534)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search(
                "test",
                "k",
                _client_id(search_db),
                content_type_filter="tip",
                top_k=10,
            )

        assert all(r.content_type == "tip" for r in results)

    def test_returns_search_result_objects(self, search_db):
        from analysis.search import semantic_search, SearchResult

        dim = 1536
        query_vec = [1.0] + [0.0] * (dim - 1)

        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("test", "k", _client_id(search_db), top_k=5)

        assert len(results) >= 2
        first = results[0]
        assert isinstance(first, SearchResult)
        assert first.unit_id == search_db[1]
        assert first.post_id == "post_a"
        assert first.username == "user1"
        assert first.topic == "Resume"
        assert first.content_type == "tip"
        assert first.text == "Resume keyword tips"
        assert first.claim == "Keywords matter"
        assert first.score == pytest.approx(1.0, abs=0.01)

    def test_empty_db_returns_empty_list(self, tmp_path):
        from analysis.search import semantic_search
        import core.db as db_mod

        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "empty.duckdb")
        db_mod.init_db()

        dim = 1536
        with patch("analysis.search.embed_text", return_value=[0.0] * dim):
            results = semantic_search("anything", "key", db_mod.DEFAULT_CLIENT_ID, top_k=5)

        assert results == []
        db_mod.DB_PATH = old_path

    def test_missing_client_id_raises_before_embedding(self, search_db):
        from analysis.search import semantic_search
        with pytest.raises(TypeError):
            semantic_search("resume", "fake_key")
        with patch("analysis.search.embed_text") as embed_mock:
            with pytest.raises(ValueError):
                semantic_search("resume", "fake_key", "")
        embed_mock.assert_not_called()


class TestSearchCarriesUnitRole:
    """unit_role is the one SearchResult field with a shipped regression.

    It was added so analysis/content_gen.py could keep competitor technique out
    of generation. The first version of that filter read the role off a
    SearchResult that had no such field, so every unit came back None and the
    firewall passed everything through — with the whole suite green. Asserting
    the other fields does not cover it, because a dropped column reads as NULL
    and NULL is the legitimate value for every legacy row.
    """

    def _set_role(self, db_path, unit_id, role):
        conn = duckdb.connect(db_path)
        try:
            conn.execute(
                "UPDATE message_units SET unit_role = ? WHERE unit_id = ?",
                [role, unit_id],
            )
        finally:
            conn.close()

    def test_keyword_search_carries_the_stored_role(self, search_db):
        from analysis.search import keyword_search

        db_path, uid_a, _uid_b, client_id = search_db
        self._set_role(db_path, uid_a, "technique")

        results = keyword_search("resume", client_id)
        by_id = {r.unit_id: r for r in results}
        assert by_id[uid_a].unit_role == "technique"

    def test_semantic_search_carries_the_stored_role(self, search_db):
        from analysis.search import semantic_search

        db_path, uid_a, _uid_b, client_id = search_db
        self._set_role(db_path, uid_a, "technique")

        query_vec = [1.0] + [0.0] * 1535
        with patch("analysis.search.embed_text", return_value=query_vec):
            results = semantic_search("test", "k", client_id, top_k=5)

        by_id = {r.unit_id: r for r in results}
        assert by_id[uid_a].unit_role == "technique"

    def test_an_unclassified_unit_stays_none_rather_than_defaulting(self, search_db):
        """Absent is not a role.

        Every row predating the classifier is NULL. If search substituted a
        default here, the generation filter would start judging rows on a value
        nobody assigned.
        """
        from analysis.search import keyword_search

        _db_path, _uid_a, uid_b, client_id = search_db
        results = keyword_search("interview", client_id)
        by_id = {r.unit_id: r for r in results}
        assert by_id[uid_b].unit_role is None
