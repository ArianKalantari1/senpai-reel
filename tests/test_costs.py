"""Per-client cost rollup (creative-director-ai #14)."""
import pytest

import core.costs as costs_mod
from core.costs import apify_rate_usd, client_costs, estimate_transcription_usd


@pytest.fixture
def db(tmp_path, monkeypatch):
    import core.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "t.duckdb"))
    db_mod.init_db()
    return db_mod


def _seed(db_mod, client_id, scrape_cost=None, jobs=1):
    conn = db_mod.get_connection()
    try:
        for i in range(jobs):
            conn.execute(
                "INSERT INTO scrape_jobs (job_id, client_id, username, reels_found, "
                "status, cost_usd) VALUES (?,?,?,?,?,?)",
                [f"j{i}_{client_id}", client_id, "acct", 10, "done", scrape_cost],
            )
        conn.execute(
            "INSERT INTO transcripts (post_id, client_id, cost_usd, extraction_cost_usd) "
            "VALUES (?,?,?,?)", [f"p_{client_id}", client_id, 0.05, 0.01]
        )
        conn.execute(
            "INSERT INTO generated_content (gen_id, client_id, content_type, output_text, "
            "cost_usd) VALUES (?,?,?,?,?)", [f"g_{client_id}", client_id, "caption", "x", 0.002]
        )
        conn.execute("INSERT INTO client_posts (client_id, post_id) VALUES (?,?)",
                     [client_id, f"p_{client_id}"])
    finally:
        conn.close()


class TestApifyRate:
    def test_unset_is_none_not_zero(self, monkeypatch):
        monkeypatch.delenv("APIFY_USD_PER_RESULT", raising=False)
        assert apify_rate_usd() is None

    def test_garbage_is_none_not_zero(self, monkeypatch):
        monkeypatch.setenv("APIFY_USD_PER_RESULT", "free")
        assert apify_rate_usd() is None

    def test_negative_rejected(self, monkeypatch):
        monkeypatch.setenv("APIFY_USD_PER_RESULT", "-1")
        assert apify_rate_usd() is None

    def test_valid_rate_parsed(self, monkeypatch):
        monkeypatch.setenv("APIFY_USD_PER_RESULT", "0.0023")
        assert apify_rate_usd() == pytest.approx(0.0023)


class TestRollup:
    def test_unpriced_scrape_marks_total_incomplete(self, db):
        _seed(db, "c1", scrape_cost=None)
        out = client_costs("c1")
        assert out["lines"]["scraping"]["known"] is False
        assert out["total_is_complete"] is False, "an unpriced scrape is not a free scrape"
        assert out["unpriced_scrape_jobs"] == 1

    def test_priced_scrape_completes_the_total(self, db):
        _seed(db, "c2", scrape_cost=0.023)
        out = client_costs("c2")
        assert out["total_is_complete"] is True
        assert out["lines"]["scraping"]["usd"] == pytest.approx(0.023)
        assert out["total_usd"] == pytest.approx(0.023 + 0.05 + 0.01 + 0.002)

    def test_cost_per_piece(self, db):
        _seed(db, "c3", scrape_cost=0.01)
        out = client_costs("c3")
        assert out["pieces_generated"] == 1
        assert out["cost_per_piece"] == pytest.approx(out["total_usd"])

    def test_cost_per_piece_none_when_nothing_generated(self, db):
        conn = db.get_connection()
        conn.execute("INSERT INTO scrape_jobs (job_id, client_id, reels_found, status, cost_usd) "
                     "VALUES (?,?,?,?,?)", ["j", "c4", 5, "done", 0.01])
        conn.close()
        assert client_costs("c4")["cost_per_piece"] is None

    def test_costs_are_scoped_to_one_client(self, db):
        _seed(db, "cA", scrape_cost=1.0)
        _seed(db, "cB", scrape_cost=99.0)
        assert client_costs("cA")["lines"]["scraping"]["usd"] == pytest.approx(1.0)
        assert client_costs("cB")["lines"]["scraping"]["usd"] == pytest.approx(99.0)

    def test_empty_client_is_zero_not_error(self, db):
        out = client_costs("nobody")
        assert out["total_usd"] == 0
        assert out["reels_analysed"] == 0


class TestEstimate:
    def test_estimate_before_spending(self):
        assert estimate_transcription_usd(100) == pytest.approx(0.58)

    def test_negative_minutes_clamped(self):
        assert estimate_transcription_usd(-5) == 0
