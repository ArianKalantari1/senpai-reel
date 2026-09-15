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


def _seed_reextract_sample_db(
    tmp_path,
    post_ids=("post_a", "post_b", "post_c", "post_d", "post_e", "post_f"),
):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "reextract_sample.duckdb")
    db_mod.init_db()
    db_path = db_mod.DB_PATH
    client_id = db_mod.DEFAULT_CLIENT_ID

    conn = duckdb.connect(db_path)
    now = datetime.utcnow()
    for post_id in post_ids:
        conn.execute(
            """
            INSERT INTO posts (
                post_id, client_id, account_id, engagement_rate, download_status,
                scraped_at, hashtags, mentions
            )
            VALUES (?, ?, 'acc1', 0, 'done', ?, [], [])
            """,
            [post_id, client_id, now],
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [client_id, post_id, now],
        )
        conn.execute(
            """
            INSERT INTO transcripts (
                post_id, client_id, provider, model, transcript, language,
                confidence, duration_sec, word_count, transcribed_at, cost_usd
            )
            VALUES (?, ?, 'deepgram', 'nova-2', ?, 'en', 0.95, 30, 7, ?, 0.001)
            """,
            [post_id, client_id, f"Transcript for {post_id}", now],
        )
        conn.execute(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model
            )
            VALUES (
                ?, ?, ?, 'old text', 'old claim', 'Resume', 'tip',
                0.9, ?, 'gpt-4o-mini'
            )
            """,
            [f"unit_{post_id}", client_id, post_id, now],
        )
    conn.close()

    return db_path, db_mod, old_path, list(post_ids)


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


def _selected_posts(output: str) -> list[str]:
    return [
        line.strip().split(":", 1)[0]
        for line in output.splitlines()
        if line.startswith("  post_")
    ]


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


def test_dry_run_same_run_id_selects_same_posts(tmp_path, capsys):
    from tools.reextract import main

    db_path, db_mod, old_path, _post_ids = _seed_reextract_sample_db(tmp_path)
    try:
        assert main(["run", db_path, "--posts", "3", "--run-id", "sample", "--dry-run"]) == 0
        first = _selected_posts(capsys.readouterr().out)

        assert main(["run", db_path, "--posts", "3", "--run-id", "sample", "--dry-run"]) == 0
        second = _selected_posts(capsys.readouterr().out)
    finally:
        db_mod.DB_PATH = old_path

    assert first == second
    assert len(first) == 3


def test_selection_is_seeded_sample_not_first_posts_by_id(tmp_path, capsys):
    from tools.reextract import main

    db_path, db_mod, old_path, post_ids = _seed_reextract_sample_db(tmp_path)
    try:
        assert main(["run", db_path, "--posts", "3", "--run-id", "sample", "--dry-run"]) == 0
        selected = _selected_posts(capsys.readouterr().out)
    finally:
        db_mod.DB_PATH = old_path

    assert selected
    assert selected != sorted(post_ids)[:3]


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


def test_compare_prints_old_and_new_units_as_independent_lists(tmp_path, capsys):
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
    assert "  # | before | run" not in out
    assert "\n  1 |" not in out
    assert "  before:" in out
    assert "  run:" in out
    assert "old_unit" in out
    assert "new_unit" in out
    assert "old claim" in out
    assert "new claim" in out


def _seed_resumable_db(tmp_path, n=5, client="acme"):
    import core.db as db_mod
    old = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "resume.duckdb")
    db_mod.init_db()
    path = db_mod.DB_PATH
    now = datetime.utcnow()
    conn = duckdb.connect(path)
    for i in range(n):
        pid = f"p{i}"
        conn.execute(
            "INSERT INTO posts (post_id, client_id, account_id, scraped_at, hashtags, mentions)"
            " VALUES (?,?,'acct',?,[],[])", [pid, client, now])
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?,?,?)",
            [client, pid, now])
        conn.execute(
            "INSERT INTO transcripts (post_id, client_id, provider, model, transcript,"
            " language, confidence, duration_sec, word_count, transcribed_at, cost_usd)"
            " VALUES (?,?,'dg','n2',?,'en',0.9,30,7,?,0.001)",
            [pid, client, f"transcript for {pid}, long enough to extract from", now])
        conn.execute(
            "INSERT INTO message_units (unit_id, client_id, post_id, text, claim, topic,"
            " content_type, confidence, extracted_at, model)"
            " VALUES (?,?,?,'old text','old claim','careers','tip',0.9,?,'m')",
            [f"old{i}", client, pid, now])
    conn.close()
    return path, db_mod, old, client


class TestAPartlyFinishedRunCanBeFinished:
    """A 429 on one call must not cost a whole validation round.

    Units are saved per post inside the loop, so a run that dies at post 3
    leaves two posts written. Before --resume, re-running the same id hit the
    duplicate guard: the run could not be finished and could not be restarted
    under the name its sample was drawn for. The id was burned.
    """

    def _rx(self):
        import importlib.util, pathlib, sys
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("rx2", root / "tools" / "reextract.py")
        m = importlib.util.module_from_spec(spec); sys.modules["rx2"] = m
        spec.loader.exec_module(m)
        return m

    @staticmethod
    def _unit_for(post_id, run_id):
        from analysis.extraction import MessageUnit
        import uuid as _u
        return MessageUnit(
            unit_id=str(_u.uuid4()), post_id=post_id, text="new", claim="new claim",
            advice=None, topic="careers", subtopic=None, content_type="insight",
            confidence=0.9, extraction_run_id=run_id, prompt_version="sha256:test",
            extracted_at=datetime.utcnow(), model="m")

    def _run_dying_at(self, rx, path, client, die_on):
        from analysis.extraction import RateLimitedError
        calls = {"n": 0}

        def flaky(transcript, post_id, key, client_id="demo", model="m", extraction_run_id=None):
            calls["n"] += 1
            if calls["n"] == die_on:
                raise RateLimitedError("simulated")
            return [self._unit_for(post_id, extraction_run_id)], 0.0001

        rx.extract_message_units = flaky
        with pytest.raises(RateLimitedError):
            rx.cmd_run(path, 5, "RUN-A", dry_run=False, client_id=client, api_key="k")

    def test_resume_finishes_the_run_without_writing_anything_twice(self, tmp_path, capsys):
        rx = self._rx()
        path, db_mod, old, client = _seed_resumable_db(tmp_path)
        try:
            self._run_dying_at(rx, path, client, die_on=3)
            conn = duckdb.connect(path)
            partial = conn.execute(
                "SELECT COUNT(DISTINCT post_id) FROM message_units"
                " WHERE extraction_run_id = 'RUN-A'").fetchone()[0]
            conn.close()
            assert partial == 2, "fixture must actually leave a partial run"

            rx.extract_message_units = lambda t, pid, k, client_id="demo", model="m", extraction_run_id=None: (
                [self._unit_for(pid, extraction_run_id)], 0.0001)
            assert rx.cmd_run(path, 5, "RUN-A", dry_run=False,
                              client_id=client, api_key="k", resume=True) == 0

            conn = duckdb.connect(path)
            done = conn.execute(
                "SELECT COUNT(DISTINCT post_id) FROM message_units"
                " WHERE extraction_run_id = 'RUN-A'").fetchone()[0]
            worst = conn.execute(
                "SELECT MAX(n) FROM (SELECT post_id, COUNT(*) n FROM message_units"
                " WHERE extraction_run_id = 'RUN-A' GROUP BY post_id)").fetchone()[0]
            originals = conn.execute(
                "SELECT COUNT(*) FROM message_units WHERE extraction_run_id IS NULL").fetchone()[0]
            conn.close()
            assert done == 5, "resume must finish every selected post"
            assert worst == 1, "a resumed post must not be written twice"
            assert originals == 5, "originals must be untouched"
        finally:
            db_mod.DB_PATH = old

    def test_same_run_id_without_resume_is_still_refused(self, tmp_path):
        rx = self._rx()
        path, db_mod, old, client = _seed_resumable_db(tmp_path)
        try:
            self._run_dying_at(rx, path, client, die_on=3)
            with pytest.raises(SystemExit) as exc:
                rx.cmd_run(path, 5, "RUN-A", dry_run=False, client_id=client, api_key="k")
            # Refusing by default is deliberate: silently appending to an
            # existing run would mix two prompt versions under one id.
            assert "--resume" in str(exc.value)
        finally:
            db_mod.DB_PATH = old

    def test_resuming_a_finished_run_does_nothing_and_says_so(self, tmp_path, capsys):
        rx = self._rx()
        path, db_mod, old, client = _seed_resumable_db(tmp_path)
        try:
            rx.extract_message_units = lambda t, pid, k, client_id="demo", model="m", extraction_run_id=None: (
                [self._unit_for(pid, extraction_run_id)], 0.0001)
            rx.cmd_run(path, 5, "RUN-A", dry_run=False, client_id=client, api_key="k")
            capsys.readouterr()
            assert rx.cmd_run(path, 5, "RUN-A", dry_run=False,
                              client_id=client, api_key="k", resume=True) == 0
            assert "Nothing to do" in capsys.readouterr().out
        finally:
            db_mod.DB_PATH = old
