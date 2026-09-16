"""Tests for tools/embed_units.py — the loop analysis.embeddings does not have."""
import importlib.util
import pathlib
import sys

import duckdb
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _eu():
    spec = importlib.util.spec_from_file_location("eu", ROOT / "tools" / "embed_units.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["eu"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def embed_db(tmp_path, monkeypatch):
    import core.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "embed.duckdb"))
    db_mod.init_db()
    return db_mod.DB_PATH


def _seed_units(db_path, client_id, n, embedded=0):
    conn = duckdb.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO posts (post_id, client_id, account_id, engagement_rate,
               download_status, scraped_at, hashtags, mentions, video_url)
               VALUES ('p1', ?, 'a1', 4.0, 'done', now(), [], [], 'u')""", [client_id])
        conn.execute("INSERT INTO client_posts (client_id, post_id, added_at) "
                     "VALUES (?, 'p1', now())", [client_id])
        for i in range(n):
            conn.execute(
                """INSERT INTO message_units (unit_id, client_id, post_id, text, claim,
                   topic, content_type, confidence, extracted_at, model)
                   VALUES (?,?, 'p1', ?, ?, 'Resume', 'tip', 0.9, now(), 'm')""",
                [f"u{i}", client_id, f"source text number {i}", f"claim number {i}"])
        for i in range(embedded):
            conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                         "WHERE unit_id = ?", [[0.1] * 1536, f"u{i}"])
    finally:
        conn.close()


def test_pending_count_separates_embedded_from_missing(embed_db):
    eu = _eu()
    _seed_units(embed_db, "c1", 10, embedded=3)

    pending, total = eu.pending_count(embed_db, "c1")

    assert (pending, total) == (7, 10)


def test_dry_run_sends_nothing_and_writes_nothing(embed_db, capsys, monkeypatch):
    eu = _eu()
    _seed_units(embed_db, "c1", 5)

    def explode(*a, **k):
        raise AssertionError("--dry-run called the embedding API")

    monkeypatch.setattr(eu, "embed_pending_units", explode)

    assert eu.cmd_run(embed_db, "c1", None, dry_run=True) == 0
    assert eu.pending_count(embed_db, "c1")[0] == 5
    assert "would embed 5" in capsys.readouterr().out


def test_it_loops_until_nothing_is_pending(embed_db, capsys, monkeypatch):
    """embed_pending_units does ONE batch. Calling it once leaves the rest."""
    eu = _eu()
    _seed_units(embed_db, "c1", 120)
    calls = []

    def fake(api_key, batch_size=50, client_id="c1", **kw):
        pending, _ = eu.pending_count(embed_db, client_id)
        n = min(batch_size, pending)
        calls.append(n)
        conn = duckdb.connect(embed_db)
        try:
            ids = [r[0] for r in conn.execute(
                "SELECT unit_id FROM message_units WHERE embedding IS NULL LIMIT ?",
                [n]).fetchall()]
            for unit_id in ids:
                conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                             "WHERE unit_id = ?", [[0.2] * 1536, unit_id])
        finally:
            conn.close()
        return {"done": n, "failed": 0, "total": n, "total_cost_usd": 0.000001 * n}

    monkeypatch.setattr(eu, "embed_pending_units", fake)
    monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

    eu.cmd_run(embed_db, "c1", None, dry_run=False)

    assert sum(calls) == 120, f"stopped early after {calls}"
    assert eu.pending_count(embed_db, "c1")[0] == 0
    assert "0 still without a vector" in capsys.readouterr().out


def test_out_of_credit_stops_the_loop_and_keeps_what_was_embedded(embed_db, capsys, monkeypatch):
    """Paid work already written must survive, and the reason must be named."""
    eu = _eu()
    _seed_units(embed_db, "c1", 120)
    calls = []

    def fake(api_key, batch_size=50, client_id="c1", **kw):
        calls.append(1)
        if len(calls) > 1:
            raise eu.OutOfCreditError("no credit remaining on the account.")
        conn = duckdb.connect(embed_db)
        try:
            ids = [r[0] for r in conn.execute(
                "SELECT unit_id FROM message_units WHERE embedding IS NULL LIMIT 50"
            ).fetchall()]
            for unit_id in ids:
                conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                             "WHERE unit_id = ?", [[0.2] * 1536, unit_id])
        finally:
            conn.close()
        return {"done": 50, "failed": 0, "total": 50, "total_cost_usd": 0.00005}

    monkeypatch.setattr(eu, "embed_pending_units", fake)
    monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

    eu.cmd_run(embed_db, "c1", None, dry_run=False)
    out = capsys.readouterr().out

    assert eu.pending_count(embed_db, "c1")[0] == 70, "embedded rows were lost"
    assert "no credit remaining" in out
    assert "70 still without a vector" in out


class TestCostIsNeverAFalseZero:
    def test_an_unpriced_batch_makes_the_total_unknown_not_zero(self, embed_db, capsys, monkeypatch):
        """The unpriced-scrape bug in a new place.

        A run that embedded rows and reported $0.000000 claims the work was
        free. If any batch comes back without a cost, the total is unknowable
        and must say so.
        """
        eu = _eu()
        _seed_units(embed_db, "c1", 60)
        calls = []

        def fake(api_key, batch_size=50, client_id="c1", **kw):
            calls.append(1)
            conn = duckdb.connect(embed_db)
            try:
                ids = [r[0] for r in conn.execute(
                    "SELECT unit_id FROM message_units WHERE embedding IS NULL LIMIT ?",
                    [batch_size]).fetchall()]
                for unit_id in ids:
                    conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                                 "WHERE unit_id = ?", [[0.2] * 1536, unit_id])
            finally:
                conn.close()
            cost = 0.00005 if len(calls) == 1 else None
            return {"done": len(ids), "failed": 0, "total": len(ids),
                    "total_cost_usd": cost}

        monkeypatch.setattr(eu, "embed_pending_units", fake)
        monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

        eu.cmd_run(embed_db, "c1", None, dry_run=False)
        out = capsys.readouterr().out

        assert "Cost unknown" in out
        assert "$0.000000" not in out

    def test_a_fully_priced_run_reports_the_real_total(self, embed_db, capsys, monkeypatch):
        eu = _eu()
        _seed_units(embed_db, "c1", 50)

        def fake(api_key, batch_size=50, client_id="c1", **kw):
            # Must reflect pending state. A fake that always claims 50 rows
            # loops forever, which is how the no-progress guard in cmd_run got
            # written.
            pending, _ = eu.pending_count(embed_db, client_id)
            if not pending:
                return {"done": 0, "failed": 0, "total": 0}
            conn = duckdb.connect(embed_db)
            try:
                conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536]",
                             [[0.2] * 1536])
            finally:
                conn.close()
            return {"done": pending, "failed": 0, "total": pending,
                    "total_cost_usd": 0.00025}

        monkeypatch.setattr(eu, "embed_pending_units", fake)
        monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

        eu.cmd_run(embed_db, "c1", None, dry_run=False)

        assert "$0.000250" in capsys.readouterr().out


def test_a_batch_that_never_resolves_anything_stops_instead_of_spinning(
        embed_db, capsys, monkeypatch):
    """cmd_run loops until nothing is pending, which trusts the caller.

    A batch reporting units but resolving none of them would loop against the
    API forever. Found by writing a fake that did exactly that — the test run
    hung rather than failed, which is the worst way for this to surface.
    """
    eu = _eu()
    _seed_units(embed_db, "c1", 60)
    calls = []

    def stuck(*a, **k):
        calls.append(1)
        # Bounded on purpose. Without this the test can only fail by hanging,
        # which is how the missing guard surfaced in the first place — a
        # mutation run timed out instead of reporting. A test whose failure
        # mode is "never finishes" tells you nothing at 3am.
        if len(calls) > 5:
            raise AssertionError(
                f"cmd_run called embed_pending_units {len(calls)} times without "
                "progress — the no-progress guard is missing")
        return {"done": 0, "failed": 0, "total": 50}

    monkeypatch.setattr(eu, "embed_pending_units", stuck)
    monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

    eu.cmd_run(embed_db, "c1", None, dry_run=False)

    assert len(calls) == 1, f"kept calling after no progress: {len(calls)} times"
    assert "reported units but embedded none" in capsys.readouterr().out
