"""
Tests for analysis/embeddings.py — Phase 10: Voyage provider.

All outbound calls are mocked via the _get_provider() helper so no real
API credentials are required during testing.
"""
import uuid
import pytest
import duckdb
from unittest.mock import patch, MagicMock
from datetime import datetime


def _mock_voyage_provider(dim=512):
    """Return a drop-in mock for VoyageProvider."""
    provider = MagicMock()
    provider.dimensions = dim
    provider.embed.side_effect = lambda text: [0.1] * dim
    provider.embed_batch.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    return provider


# ── embed_batch ───────────────────────────────────────────────────────────────

class TestEmbedBatch:
    def test_empty_list_returns_empty(self):
        from analysis.embeddings import embed_batch
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            result = embed_batch([])
        assert result == []

    def test_single_text_returns_512_dim_vector(self):
        from analysis.embeddings import embed_batch
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            result = embed_batch(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 512

    def test_multiple_texts_returns_multiple_vectors(self):
        from analysis.embeddings import embed_batch
        texts = ["text one", "text two", "text three"]
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            result = embed_batch(texts)
        assert len(result) == 3

    def test_provider_error_propagates(self):
        from analysis.embeddings import embed_batch
        provider = _mock_voyage_provider()
        provider.embed_batch.side_effect = RuntimeError("API down")
        with patch("analysis.embeddings._get_provider", return_value=provider):
            with pytest.raises(RuntimeError):
                embed_batch(["text"])


# ── embed_text ────────────────────────────────────────────────────────────────

class TestEmbedText:
    def test_returns_single_512_dim_vector(self):
        from analysis.embeddings import embed_text
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            vec = embed_text("test")
        assert isinstance(vec, list)
        assert len(vec) == 512


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
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            result = embed_pending_units()
        assert result["done"] == 0
        assert result["total"] == 0

    def test_embeds_pending_units(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        for i in range(3):
            conn.execute("""
                INSERT INTO message_units (unit_id, post_id, text, claim, topic, content_type,
                    confidence, extracted_at, model)
                VALUES (?, ?, ?, ?, 'Resume', 'tip', 0.9, ?, 'llama-3.3-70b')
            """, (str(uuid.uuid4()), f"post_{i}", f"text {i}", f"claim {i}", now))
        conn.close()

        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            result = embed_pending_units(batch_size=10)

        assert result["total"] == 3
        assert result["done"] == 3
        assert result["failed"] == 0

    def test_provider_failure_returns_failed_count(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        conn.execute("""
            INSERT INTO message_units (unit_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, 'post_err', 'text', 'claim', 'Resume', 'tip', 0.9, ?, 'llama-3.3-70b')
        """, (str(uuid.uuid4()), now))
        conn.close()

        provider = _mock_voyage_provider()
        provider.embed_batch.side_effect = RuntimeError("API error")

        with patch("analysis.embeddings._get_provider", return_value=provider):
            result = embed_pending_units()

        assert result["failed"] == 1
        assert result["done"] == 0

    def test_progress_callback_called(self, embed_db):
        from analysis.embeddings import embed_pending_units

        conn = duckdb.connect(embed_db.DB_PATH)
        now = datetime.utcnow()
        conn.execute("""
            INSERT INTO message_units (unit_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model)
            VALUES (?, 'post_cb', 'text', 'claim', 'Resume', 'tip', 0.9, ?, 'llama-3.3-70b')
        """, (str(uuid.uuid4()), now))
        conn.close()

        calls = []
        with patch("analysis.embeddings._get_provider", return_value=_mock_voyage_provider()):
            embed_pending_units(progress_callback=lambda *a: calls.append(a))

        assert len(calls) == 1

