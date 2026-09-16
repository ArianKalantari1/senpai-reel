"""Tests for tools/migrate_embedding_width.py."""
import importlib.util
import pathlib
import sys

import duckdb
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _mw():
    spec = importlib.util.spec_from_file_location(
        "mw", ROOT / "tools" / "migrate_embedding_width.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["mw"] = m
    spec.loader.exec_module(m)
    return m


def _narrow_db(tmp_path, vectors=3, width=512):
    path = str(tmp_path / "narrow.duckdb")
    conn = duckdb.connect(path)
    try:
        conn.execute(f"""CREATE TABLE message_units (
            unit_id TEXT, client_id TEXT, post_id TEXT, text TEXT,
            embedding FLOAT[{width}], embedded_at TIMESTAMP,
            embedding_cost_usd DOUBLE)""")
        for i in range(vectors):
            conn.execute(
                "INSERT INTO message_units VALUES (?, 'c1', 'p1', 't', "
                "?::FLOAT[%d], now(), NULL)" % width,
                [f"u{i}", [0.1 * (i + 1)] * width])
        conn.execute("INSERT INTO message_units VALUES ('bare','c1','p1','t',"
                     "NULL, NULL, NULL)")
    finally:
        conn.close()
    return path


class TestDryRunChangesNothing:
    def test_it_reports_without_touching_the_column(self, tmp_path, capsys):
        mw = _mw()
        db = _narrow_db(tmp_path)

        assert mw.cmd_main(db, 1536, apply=False) == 0

        conn = duckdb.connect(db, read_only=True)
        try:
            assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[512]"
            with pytest.raises(duckdb.CatalogException):
                conn.execute("SELECT * FROM embedding_archive").fetchall()
        finally:
            conn.close()
        assert "DRY RUN" in capsys.readouterr().out

    def test_an_already_correct_column_is_left_alone(self, tmp_path, capsys):
        mw = _mw()
        db = _narrow_db(tmp_path, width=1536)

        assert mw.cmd_main(db, 1536, apply=True) == 0
        assert "already the target width" in capsys.readouterr().out

        conn = duckdb.connect(db, read_only=True)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL"
            ).fetchone()[0] == 3, "vectors were cleared despite no migration"
        finally:
            conn.close()


class TestVectorsAreArchivedBeforeAnythingIsDropped:
    """Somebody paid for those vectors. Unusable is not the same as worthless
    to keep — this project does not delete work to make a migration tidy.
    """

    def test_every_vector_survives_in_the_archive(self, tmp_path):
        mw = _mw()
        db = _narrow_db(tmp_path, vectors=3)

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db, read_only=True)
        try:
            rows = conn.execute(
                "SELECT unit_id, embedding, note FROM embedding_archive ORDER BY unit_id"
            ).fetchall()
            assert [r[0] for r in rows] == ["u0", "u1", "u2"]
            assert all(len(r[1]) == 512 for r in rows), "archived vectors were truncated"
            assert all("model unrecorded" in r[2] for r in rows)
        finally:
            conn.close()

    def test_the_column_is_replaced_at_the_target_width(self, tmp_path):
        mw = _mw()
        db = _narrow_db(tmp_path)

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db)
        try:
            assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[1536]"
            # And it now accepts what the model actually returns.
            conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                         "WHERE unit_id = 'u0'", [[0.5] * 1536])
        finally:
            conn.close()

    def test_rows_without_a_vector_are_untouched(self, tmp_path):
        mw = _mw()
        db = _narrow_db(tmp_path, vectors=3)

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db, read_only=True)
        try:
            assert conn.execute("SELECT COUNT(*) FROM message_units").fetchone()[0] == 4
            assert conn.execute(
                "SELECT COUNT(*) FROM embedding_archive").fetchone()[0] == 3
        finally:
            conn.close()


def test_embedding_model_column_is_added(tmp_path):
    """The defect underneath all of this: nothing recorded which model made a
    vector, so 730 of them became unattributable and therefore unusable.
    """
    mw = _mw()
    db = _narrow_db(tmp_path)

    mw.cmd_main(db, 1536, apply=True)

    conn = duckdb.connect(db, read_only=True)
    try:
        assert mw.column_type(conn, "message_units", "embedding_model") == "VARCHAR"
    finally:
        conn.close()


def test_plan_is_read_only(tmp_path):
    mw = _mw()
    db = _narrow_db(tmp_path)

    state = mw.plan(db, 1536)

    assert state == {"current_type": "FLOAT[512]", "target": "FLOAT[1536]",
                     "vectors": 3, "already_archived": 0, "needed": True}
    conn = duckdb.connect(db, read_only=True)
    try:
        assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[512]"
    finally:
        conn.close()


class TestArchiveCompleteness:
    """Tested directly because the integration path cannot fail on demand.

    A mutation removing this check survived the database-level tests: the
    archive always succeeded there, so nothing ever exercised the refusal.
    """

    def test_a_short_archive_is_refused(self):
        mw = _mw()
        assert mw.archive_is_complete(before=730, archived=730) is True
        assert mw.archive_is_complete(before=730, archived=729) is False
        assert mw.archive_is_complete(before=730, archived=0) is False

    def test_an_empty_source_needs_no_archive(self):
        mw = _mw()
        assert mw.archive_is_complete(before=0, archived=0) is True

    def test_the_refusal_says_nothing_was_changed(self):
        """An operator reading this must know the database is still intact."""
        mw = _mw()
        message = mw.archive_shortfall_message(before=730, archived=12)
        assert "730" in message and "12" in message
        assert "Nothing was changed" in message
        assert "still holds every vector" in message


def _indexed_db(tmp_path, vectors=3):
    """A column indexed AFTER embedding — the real database's shape.

    DuckDB refuses to drop a column when an index references a later one,
    because dropping shifts every subsequent column's position. core/db.py
    declares three message_units indexes and all sit on columns BEFORE
    embedding, so the schema alone never reproduces this.
    """
    path = str(tmp_path / "indexed.duckdb")
    conn = duckdb.connect(path)
    try:
        conn.execute("""CREATE TABLE message_units (
            unit_id TEXT, client_id TEXT, post_id TEXT, text TEXT,
            embedding FLOAT[512], embedded_at TIMESTAMP,
            embedding_cost_usd DOUBLE, extraction_run_id TEXT)""")
        conn.execute("CREATE INDEX idx_mu_client ON message_units(client_id)")
        conn.execute("CREATE INDEX idx_mu_run ON message_units(extraction_run_id)")
        for i in range(vectors):
            conn.execute(
                "INSERT INTO message_units VALUES (?, 'c1', 'p1', 't', "
                "?::FLOAT[512], now(), NULL, 'r1')", [f"u{i}", [0.1] * 512])
    finally:
        conn.close()
    return path


class TestIndexesThatBlockTheColumnDrop:
    def test_the_bare_drop_really_does_fail(self, tmp_path):
        """Proves the fixture reproduces the operator's error.

        Without this the other tests could pass against a database that never
        had the problem.
        """
        db = _indexed_db(tmp_path)
        conn = duckdb.connect(db)
        try:
            with pytest.raises(duckdb.CatalogException, match="index depends on a column"):
                conn.execute("ALTER TABLE message_units DROP COLUMN embedding")
        finally:
            conn.close()

    def test_migration_succeeds_and_puts_every_index_back(self, tmp_path, capsys):
        mw = _mw()
        db = _indexed_db(tmp_path)

        assert mw.cmd_main(db, 1536, apply=True) == 0

        conn = duckdb.connect(db, read_only=True)
        try:
            assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[1536]"
            names = {n for n, _ in mw.indexes_on(conn, "message_units")}
            assert names == {"idx_mu_client", "idx_mu_run"}, names
        finally:
            conn.close()
        assert "rebuilt 2 index(es)" in capsys.readouterr().out

    def test_the_rebuilt_index_has_the_same_definition(self, tmp_path):
        mw = _mw()
        db = _indexed_db(tmp_path)

        before = dict(mw.indexes_on(duckdb.connect(db, read_only=True), "message_units"))
        mw.cmd_main(db, 1536, apply=True)
        after = dict(mw.indexes_on(duckdb.connect(db, read_only=True), "message_units"))

        assert after == before, "an index came back with a different definition"

    def test_vectors_still_reach_the_archive_when_indexes_are_involved(self, tmp_path):
        mw = _mw()
        db = _indexed_db(tmp_path, vectors=3)

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db, read_only=True)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM embedding_archive").fetchone()[0] == 3
        finally:
            conn.close()

    def test_an_index_with_no_recorded_sql_refuses_rather_than_losing_it(
            self, tmp_path, monkeypatch):
        """Silently losing an index is the worst outcome available here.

        Nobody notices until a query is slow months later, and by then the
        cause is long gone from anyone's memory.
        """
        mw = _mw()
        db = _indexed_db(tmp_path)
        real = mw.indexes_on

        def blank_sql(conn, table):
            return [(name, None) for name, _ in real(conn, table)]

        monkeypatch.setattr(mw, "indexes_on", blank_sql)

        with pytest.raises(SystemExit) as exc:
            mw.cmd_main(db, 1536, apply=True)
        assert "cannot be recreated" in str(exc.value)
        assert "Nothing was dropped" in str(exc.value)

        conn = duckdb.connect(db, read_only=True)
        try:
            # The refusal has to come BEFORE the first DROP INDEX. After that
            # there is no way back: the drops must be committed for DuckDB to
            # let the column go, so the swap cannot be one transaction.
            assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[512]"
            assert conn.execute("SELECT COUNT(*) FROM message_units "
                                "WHERE embedding IS NOT NULL").fetchone()[0] == 3
            names = {n for n, _ in mw.indexes_on(conn, "message_units")}
            assert names == {"idx_mu_client", "idx_mu_run"}, names
        finally:
            conn.close()


def _seed_archive(db, rows):
    """Put rows into embedding_archive before the migration runs.

    Simulates a re-run: a first attempt that archived successfully and then
    died during the schema swap, which cannot be rolled back.
    """
    conn = duckdb.connect(db)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS embedding_archive (
            unit_id TEXT, embedding FLOAT[], embedded_at TIMESTAMP,
            embedding_cost_usd DOUBLE, archived_at TIMESTAMP, note TEXT)""")
        for unit_id in rows:
            conn.execute(
                "INSERT INTO embedding_archive VALUES (?, [0.1], now(), NULL, "
                "now(), 'earlier attempt')", [unit_id])
    finally:
        conn.close()


class TestReRunningAfterAFailedSwap:
    """The swap cannot be one transaction, so a half-finished run is possible.

    DuckDB's column-drop dependency check does not see uncommitted index
    drops, so the drops must commit before the column can go. The archive is
    committed first to make that survivable — which means a second run has to
    cope with an archive that is already populated.
    """

    def test_a_vector_already_archived_is_not_stored_twice(self, tmp_path):
        mw = _mw()
        db = _narrow_db(tmp_path, vectors=3)
        _seed_archive(db, ["u0"])

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db, read_only=True)
        try:
            total, distinct = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT unit_id) FROM embedding_archive"
            ).fetchone()
            assert (total, distinct) == (3, 3), "u0 was archived twice"
        finally:
            conn.close()

    def test_a_null_unit_id_in_the_archive_does_not_block_the_copy(self, tmp_path):
        """`unit_id NOT IN (SELECT unit_id FROM archive)` is never true once
        the archive holds one NULL, so the dedupe would copy nothing and the
        migration would refuse — on a database that was perfectly fine.
        """
        mw = _mw()
        db = _narrow_db(tmp_path, vectors=3)
        _seed_archive(db, [None])

        mw.cmd_main(db, 1536, apply=True)

        conn = duckdb.connect(db, read_only=True)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM embedding_archive "
                "WHERE unit_id IS NOT NULL").fetchone()[0] == 3
        finally:
            conn.close()


class TestFailureMessagesSayWhatIsTrue:
    """A message that claims a rollback that did not happen is worse than no
    message: the operator stops looking at exactly the wrong moment."""

    def test_the_rebuild_failure_prints_the_sql_to_run_by_hand(self):
        mw = _mw()
        saved = [("idx_a", "CREATE INDEX idx_a ON t(a);"),
                 ("idx_b", "CREATE INDEX idx_b ON t(b);")]

        msg = mw.index_rebuild_message(["idx_b"], saved)

        assert "CREATE INDEX idx_b ON t(b);" in msg
        assert "CREATE INDEX idx_a ON t(a);" not in msg, "named an index that came back"
        assert "Nothing was changed" not in msg, "claimed a rollback that did not happen"
        assert "embedding_archive" in msg

    def test_the_missing_sql_refusal_says_nothing_was_dropped(self):
        mw = _mw()
        msg = mw.index_sql_missing_message(["idx_a"])
        assert "cannot be recreated" in msg
        assert "Nothing was dropped" in msg


class TestGuardsTheIntegrationPathCannotReach:
    """Each recreate either works or raises, so the "did every index come
    back?" branch is unreachable from a real database. Tested directly, or it
    is decoration."""

    def test_an_index_that_did_not_come_back_is_named(self):
        mw = _mw()
        saved = [("idx_a", "sql_a"), ("idx_b", "sql_b")]
        assert mw.indexes_not_rebuilt(saved, {"idx_a"}) == ["idx_b"]

    def test_all_present_is_empty(self):
        mw = _mw()
        saved = [("idx_a", "sql_a"), ("idx_b", "sql_b")]
        assert mw.indexes_not_rebuilt(saved, {"idx_a", "idx_b"}) == []

    def test_an_extra_index_is_not_reported_as_missing(self):
        """Recreating can leave more indexes than were saved — an ART index
        DuckDB builds for a constraint, say. That is not a loss."""
        mw = _mw()
        assert mw.indexes_not_rebuilt([("idx_a", "sql_a")], {"idx_a", "idx_x"}) == []


class _ConnWithoutInserts:
    """Passes every statement through except INSERT, which it drops.

    Reproduces the one failure the archive check exists for: a copy that
    silently writes nothing while the archive already holds rows from an
    earlier attempt. Counting archive rows would see enough rows and let the
    column be dropped; counting coverage sees that no CURRENT vector is in
    there. Bounded — it does not hang, it returns.
    """

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *args):
        if sql.strip().upper().startswith("INSERT"):
            return self._conn.execute("SELECT 1")
        return self._conn.execute(sql, *args)


class TestTheArchiveCheckCountsCoverageNotRows:
    def test_a_copy_that_wrote_nothing_is_caught_even_with_a_full_archive(
            self, tmp_path):
        mw = _mw()
        db = _narrow_db(tmp_path, vectors=3)
        # Three rows already in the archive, none of them for a unit that
        # currently holds a vector: an earlier run at a different width.
        _seed_archive(db, ["old0", "old1", "old2"])

        conn = duckdb.connect(db)
        try:
            with pytest.raises(SystemExit) as exc:
                mw._archive_vectors(_ConnWithoutInserts(conn), 1536)
            assert "still holds every vector it did" in str(exc.value)
        finally:
            conn.close()

        conn = duckdb.connect(db, read_only=True)
        try:
            assert mw.column_type(conn, "message_units", "embedding") == "FLOAT[512]"
            assert conn.execute("SELECT COUNT(*) FROM message_units "
                                "WHERE embedding IS NOT NULL").fetchone()[0] == 3
        finally:
            conn.close()

    def test_the_swap_actually_consults_the_guard(self, tmp_path):
        """indexes_not_rebuilt being correct is worth nothing if _swap_column
        ignores what it returns. Deleting the `if missing` branch survives
        every other test in this file.
        """
        mw = _mw()
        db = _indexed_db(tmp_path)
        real = mw.indexes_on
        calls = {"n": 0}

        def vanishing(conn, table):
            # First call is the save; every later call is the verification,
            # answered as if the rebuild had silently done nothing.
            calls["n"] += 1
            return real(conn, table) if calls["n"] == 1 else []

        conn = duckdb.connect(db)
        try:
            mw.indexes_on = vanishing
            with pytest.raises(SystemExit) as exc:
                mw._swap_column(conn, 1536)
        finally:
            mw.indexes_on = real
            conn.close()
        assert "did not come back" in str(exc.value)
        assert "idx_mu_client" in str(exc.value)
        assert "CREATE INDEX idx_mu_client" in str(exc.value), "no SQL to recover with"
