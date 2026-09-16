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


def test_a_batch_that_saves_nothing_stops_instead_of_spinning(
        embed_db, capsys, monkeypatch):
    """Forward progress means rows LEFT the pending set.

    The first version of this guard only fired when done == 0 AND failed == 0.
    On a real database whose embedding column rejected every write, every batch
    came back done=0 failed=50 — so the guard never fired, the same fifty rows
    were selected again, and the loop ran 190 times paying for 9,500
    embeddings and saving none of them.
    """
    eu = _eu()
    _seed_units(embed_db, "c1", 500)
    calls = []

    def all_fail(*a, **k):
        calls.append(1)
        # Bounded: without it the only failure mode is hanging, which is how
        # the original gap surfaced — a mutation run timed out rather than
        # reporting.
        if len(calls) > 5:
            raise AssertionError(
                f"called {len(calls)} times while saving nothing — the loop "
                "would charge for the same rows forever")
        return {"done": 0, "failed": 50, "total": 50, "total_cost_usd": 0.00005}

    monkeypatch.setattr(eu, "embed_pending_units", all_fail)
    monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

    eu.cmd_run(embed_db, "c1", None, dry_run=False)
    out = capsys.readouterr().out

    assert len(calls) == 1, f"kept paying after saving nothing: {len(calls)} batches"
    assert "50 row(s) in this batch failed and none were saved" in out
    assert "same rows again" in out


def test_a_batch_reporting_rows_but_doing_nothing_also_stops(embed_db, capsys, monkeypatch):
    eu = _eu()
    _seed_units(embed_db, "c1", 60)
    calls = []

    def stuck(*a, **k):
        calls.append(1)
        if len(calls) > 5:
            raise AssertionError("no-progress guard is missing")
        return {"done": 0, "failed": 0, "total": 50}

    monkeypatch.setattr(eu, "embed_pending_units", stuck)
    monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

    eu.cmd_run(embed_db, "c1", None, dry_run=False)

    assert len(calls) == 1


class TestDimensionPreflight:
    """The stored column decides the width, and it is not always 1536.

    A real database declares FLOAT[512]. creative-director-ai#29 recorded a
    Voyage adapter at 512 dimensions against a FLOAT[1536] column and noted the
    resize migration was never written — it had in fact been applied to that
    database, and nothing in the code knew.
    """

    def test_it_reads_the_declared_width(self, embed_db):
        eu = _eu()
        assert eu.embedding_dimension(embed_db) == 1536

    def test_a_narrow_column_is_read_and_requested(self, tmp_path, monkeypatch, capsys):
        eu = _eu()
        narrow = str(tmp_path / "narrow.duckdb")
        conn = duckdb.connect(narrow)
        try:
            conn.execute("CREATE TABLE message_units (unit_id TEXT, client_id TEXT, "
                         "post_id TEXT, text TEXT, embedding FLOAT[512])")
            conn.execute("CREATE TABLE client_posts (client_id TEXT, post_id TEXT)")
            conn.execute("INSERT INTO client_posts VALUES ('c1','p1')")
            conn.execute("INSERT INTO message_units VALUES ('u1','c1','p1','t',NULL)")
        finally:
            conn.close()

        assert eu.embedding_dimension(narrow) == 512

        seen = {}

        def fake(api_key, batch_size=50, client_id="c1", dimensions=None, **kw):
            seen["dimensions"] = dimensions
            return {"done": 0, "failed": 0, "total": 0}

        monkeypatch.setattr(eu, "embed_pending_units", fake)
        monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")
        eu.cmd_run(narrow, "c1", None, dry_run=False)

        assert seen["dimensions"] == 512, "asked the API for the wrong width"
        out = capsys.readouterr().out
        assert "FLOAT[512]" in out
        assert "NOT comparable" in out, "the width mismatch was not called out"

    def test_an_unreadable_width_refuses_to_spend(self, tmp_path, monkeypatch):
        eu = _eu()
        empty = str(tmp_path / "empty.duckdb")
        conn = duckdb.connect(empty)
        try:
            # A variable-length list column: realistic, readable by
            # pending_count, but carries no declared width to request.
            conn.execute("CREATE TABLE message_units (unit_id TEXT, client_id TEXT, "
                         "post_id TEXT, text TEXT, embedding FLOAT[])")
            conn.execute("CREATE TABLE client_posts (client_id TEXT, post_id TEXT)")
            conn.execute("INSERT INTO client_posts VALUES ('c1','p1')")
            conn.execute("INSERT INTO message_units VALUES ('u1','c1','p1','t',NULL)")
        finally:
            conn.close()

        monkeypatch.setattr(eu, "embed_pending_units",
                            lambda *a, **k: pytest.fail("spent without knowing the width"))
        monkeypatch.setattr(eu, "get_secret", lambda name: "sk-test")

        with pytest.raises(SystemExit) as exc:
            eu.cmd_run(empty, "c1", None, dry_run=False)
        assert "Refusing to spend on a guess" in str(exc.value)


class TestWidthParsing:
    """Tested on type strings directly: the discriminating cases are ones the
    schema in this repo does not currently produce, which is exactly why a
    database-only test let a loose regex survive mutation.
    """

    def test_only_a_bracketed_number_counts(self):
        eu = _eu()
        assert eu.width_of("FLOAT[512]") == 512
        assert eu.width_of("FLOAT[1536]") == 1536

    def test_a_digit_in_the_type_name_is_not_the_width(self):
        """DuckDB reports REAL as FLOAT4.

        A search for any digit reads FLOAT4[512] as 4, and the tool would then
        request four-dimension vectors — wrong, and wrong in a way that writes
        successfully into a column of the wrong shape.
        """
        eu = _eu()
        assert eu.width_of("FLOAT4[512]") == 512
        assert eu.width_of("FLOAT4") is None

    def test_a_variable_length_list_has_no_declared_width(self):
        eu = _eu()
        assert eu.width_of("FLOAT[]") is None

    def test_absent_is_none_not_a_default(self):
        eu = _eu()
        assert eu.width_of(None) is None
        assert eu.width_of("") is None
