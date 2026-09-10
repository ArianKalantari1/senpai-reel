"""
Tests for analysis/embeddings.py — mocked OpenAI embeddings API.
"""
import uuid
import pytest
import duckdb
from unittest.mock import patch, MagicMock
from datetime import datetime


def _openai_embed_response(texts):
    """Build a minimal OpenAI embeddings response."""
    dim = 1536
    return {
        "data": [
            {"embedding": [0.1] * dim, "index": i}
            for i in range(len(texts))
        ],
        "usage": {"prompt_tokens": len(texts) * 5, "total_tokens": len(texts) * 5},
    }


# ── embed_batch ───────────────────────────────────────────────────────────────

class TestEmbedBatch:
    def test_empty_list_returns_empty(self):
        from analysis.embeddings import embed_batch
        result = embed_batch([], "api_key")
        assert result == []

    def test_single_text_returns_vector(self):
        from analysis.embeddings import embed_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_embed_response(["hello"])

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            result = embed_batch(["hello"], "fake_key")

        assert len(result) == 1
        assert len(result[0]) == 1536

    def test_multiple_texts_returns_multiple_vectors(self):
        from analysis.embeddings import embed_batch

        texts = ["text one", "text two", "text three"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_embed_response(texts)

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            result = embed_batch(texts, "fake_key")

        assert len(result) == 3

    def test_401_raises_permission_error(self):
        from analysis.embeddings import embed_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.return_value = None

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            with pytest.raises(PermissionError):
                embed_batch(["text"], "bad_key")

    def test_large_batch_split_into_sub_batches(self):
        """If texts > _BATCH_SIZE (50), should make multiple API calls."""
        from analysis.embeddings import embed_batch, _BATCH_SIZE

        texts = [f"text {i}" for i in range(_BATCH_SIZE + 5)]

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            batch = kwargs["json"]["input"]
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.raise_for_status.return_value = None
            mock_r.json.return_value = _openai_embed_response(batch)
            return mock_r

        with patch("analysis.embeddings.requests.post", side_effect=side_effect):
            result = embed_batch(texts, "key")

        assert call_count[0] == 2  # ceil((50+5) / 50) = 2 calls
        assert len(result) == len(texts)


# ── embed_text ────────────────────────────────────────────────────────────────

class TestEmbedText:
    def test_returns_single_vector(self):
        from analysis.embeddings import embed_text

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_embed_response(["test"])

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            vec = embed_text("test", "key")

        assert isinstance(vec, list)
        assert len(vec) == 1536


# ── embed_pending_units ───────────────────────────────────────────────────────

@pytest.fixture
def embed_db(tmp_path):
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "embed.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


class TestEmbedPendingUnits:
    def test_no_pending_returns_zeros(self, embed_db):
        from analysis.embeddings import embed_pending_units
        result = embed_pending_units("key")
        assert result["done"] == 0
        assert result["total"] == 0

    def test_embeds_pending_units(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        client_id = embed_db.DEFAULT_CLIENT_ID
        for i in range(3):
            conn.execute(
                "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
                [client_id, f"post_{i}", now],
            )
            conn.execute("""
                INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic, content_type,
                    confidence, extracted_at, model)
                VALUES (?, ?, ?, ?, ?, 'Resume', 'tip', 0.9, ?, 'gpt-4o-mini')
            """, (str(uuid.uuid4()), client_id, f"post_{i}", f"text {i}", f"claim {i}", now))
        conn.close()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_embed_response(["a", "b", "c"])

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            result = embed_pending_units("key", batch_size=10)

        assert result["total"] == 3
        assert result["done"] == 3
        assert result["failed"] == 0

    def test_api_failure_returns_failed_count(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        client_id = embed_db.DEFAULT_CLIENT_ID
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, 'post_err', ?)",
            [client_id, now],
        )
        conn.execute("""
            INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, ?, 'post_err', 'text', 'claim', 'Resume', 'tip', 0.9, ?, 'gpt-4o-mini')
        """, (str(uuid.uuid4()), client_id, now))
        conn.close()

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.return_value = None

        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            result = embed_pending_units("bad_key")

        assert result["failed"] == 1
        assert result["done"] == 0

    def test_progress_callback_called(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        client_id = embed_db.DEFAULT_CLIENT_ID
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, 'post_cb', ?)",
            [client_id, now],
        )
        conn.execute("""
            INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, ?, 'post_cb', 'text', 'claim', 'Resume', 'tip', 0.9, ?, 'gpt-4o-mini')
        """, (str(uuid.uuid4()), client_id, now))
        conn.close()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_embed_response(["a"])

        calls = []
        with patch("analysis.embeddings.requests.post", return_value=mock_resp):
            embed_pending_units("key", progress_callback=lambda *a: calls.append(a))

        assert len(calls) == 1
