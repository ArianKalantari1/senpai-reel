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
