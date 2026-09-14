from datetime import datetime

import duckdb
import pytest


def _seed_reextract_db(tmp_path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "reextract.duckdb")
    db_mod.init_db()
    db_path = db_mod.DB_PATH
    client_id = db_mod.DEFAULT_CLIENT_ID

    conn = duckdb.connect(db_path)
    now = datetime.utcnow()
    conn.execute(
        """
        INSERT INTO posts (
            post_id, client_id, account_id, engagement_rate, download_status,
            scraped_at, hashtags, mentions
        )
        VALUES ('post_1', ?, 'acc1', 0, 'done', ?, [], [])
        """,
        [client_id, now],
    )
    conn.execute(
        "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, 'post_1', ?)",
        [client_id, now],
    )
    conn.execute(
        """
        INSERT INTO transcripts (
            post_id, client_id, provider, model, transcript, language,
            confidence, duration_sec, word_count, transcribed_at, cost_usd
        )
        VALUES (
            'post_1', ?, 'deepgram', 'nova-2',
            'Tailor your resume with evidence before applying.', 'en',
            0.95, 30, 7, ?, 0.001
        )
        """,
        [client_id, now],
    )
    conn.execute(
        """
        INSERT INTO message_units (
            unit_id, client_id, post_id, text, claim, topic, content_type,
            confidence, extracted_at, model
        )
        VALUES (
            'old_unit', ?, 'post_1', 'old text', 'old claim',
            'Resume', 'tip', 0.9, ?, 'gpt-4o-mini'
        )
        """,
        [client_id, now],
    )
    conn.close()

    return db_path, db_mod, old_path


def _legacy_unmigrated_db(tmp_path):
    path = str(tmp_path / "legacy.duckdb")
    conn = duckdb.connect(path)
    conn.execute(
        """
        CREATE TABLE message_units (
            unit_id TEXT PRIMARY KEY,
            client_id TEXT,
            post_id TEXT,
            text TEXT,
            claim TEXT,
            content_type TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE transcripts (
            post_id TEXT,
            client_id TEXT,
            transcript TEXT
        )
        """
    )
    conn.execute("CREATE TABLE client_posts (client_id TEXT, post_id TEXT, added_at TIMESTAMP)")
    conn.close()
    return path


def _row(db_path, unit_id):
    conn = duckdb.connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM message_units WHERE unit_id = ?",
            [unit_id],
        ).fetchone()
    finally:
        conn.close()


def test_dry_run_on_unmigrated_db_explains_migration(tmp_path):
    from tools.reextract import main

    db_path = _legacy_unmigrated_db(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", db_path, "--posts", "1", "--run-id", "sample", "--dry-run"])

    message = str(exc.value)
    assert "BinderException" not in message
    assert "extraction_run_id" in message
    assert "prompt_version" in message
    assert "init_db" in message
    assert "python -c" in message


def test_dry_run_uses_read_only_path_and_writes_nothing(tmp_path):
    from tools.reextract import main

    db_path, db_mod, old_path = _seed_reextract_db(tmp_path)
    try:
        before = _row(db_path, "old_unit")
        assert main(["run", db_path, "--posts", "1", "--run-id", "sample", "--dry-run"]) == 0
        after = _row(db_path, "old_unit")

        conn = duckdb.connect(db_path)
        run_rows = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE extraction_run_id = 'sample'"
        ).fetchone()[0]
        conn.close()
    finally:
        db_mod.DB_PATH = old_path

    assert after == before
    assert run_rows == 0


def test_run_appends_tagged_units_and_preserves_existing_rows(tmp_path, monkeypatch):
    from analysis.extraction import MessageUnit
    from tools import reextract

    db_path, db_mod, old_path = _seed_reextract_db(tmp_path)
    calls = []

    def fake_extract(transcript, post_id, api_key, **kwargs):
        calls.append((transcript, post_id, api_key, kwargs))
        return [
            MessageUnit(
                unit_id="new_unit",
                post_id=post_id,
                text="new text",
                claim="new claim",
                advice=None,
                topic="Resume",
                subtopic=None,
                content_type="tip",
                confidence=0.8,
                extraction_run_id=kwargs["extraction_run_id"],
                prompt_version="sha256:testprompt",
                extracted_at=datetime.utcnow(),
            )
        ], 0.000123

    monkeypatch.setattr(reextract, "extract_message_units", fake_extract)

    try:
        before = _row(db_path, "old_unit")
        assert reextract.main([
            "run",
            db_path,
            "--posts",
            "1",
            "--run-id",
            "sample",
            "--api-key",
            "key",
        ]) == 0
        after = _row(db_path, "old_unit")

        conn = duckdb.connect(db_path)
        old_count = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE extraction_run_id IS NULL"
        ).fetchone()[0]
        new_row = conn.execute(
            """
            SELECT extraction_run_id, prompt_version, claim
            FROM message_units
            WHERE unit_id = 'new_unit'
            """
        ).fetchone()
        conn.close()
    finally:
        db_mod.DB_PATH = old_path

    assert after == before, "pre-existing rows must be byte-identical"
    assert old_count == 1
    assert new_row == ("sample", "sha256:testprompt", "new claim")
    assert calls[0][3]["extraction_run_id"] == "sample"


def test_compare_prints_old_and_new_units_side_by_side(tmp_path, capsys):
    from tools.reextract import cmd_compare

    db_path, db_mod, old_path = _seed_reextract_db(tmp_path)
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        INSERT INTO message_units (
            unit_id, client_id, post_id, text, claim, topic, content_type,
            confidence, extracted_at, model, extraction_run_id, prompt_version
        )
        VALUES (
            'new_unit', ?, 'post_1', 'new text', 'new claim',
            'Resume', 'tip', 0.8, CURRENT_TIMESTAMP, 'gpt-4o-mini',
            'sample', 'sha256:testprompt'
        )
        """,
        [db_mod.DEFAULT_CLIENT_ID],
    )
    conn.close()

    try:
        assert cmd_compare(db_path, "sample", client_id=db_mod.DEFAULT_CLIENT_ID) == 0
    finally:
        db_mod.DB_PATH = old_path

    out = capsys.readouterr().out
    assert "Post post_1: before 1 unit(s), run 1 unit(s)" in out
    assert "old_unit" in out
    assert "new_unit" in out
    assert "old claim" in out
    assert "new claim" in out
