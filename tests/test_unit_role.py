"""unit_role: subject vs technique vs meta (creative-director-ai #27).

content_type conflated "what shape is this idea" with "is this an idea at all".
These tests pin the second axis, and in particular pin the two things that make
it safe: the rules never guess, and NULL never becomes "subject".
"""
import pytest

from analysis.taxonomy import CONTENT_TYPES
from analysis.unit_role import (
    UNIT_ROLES,
    role_from_rules,
    validate_unit_role,
    unit_roles_for_prompt,
)


class TestRulesAreExhaustiveOverContentType:
    @pytest.mark.parametrize("ctype", CONTENT_TYPES)
    def test_every_content_type_is_handled_without_error(self, ctype):
        result = role_from_rules(text="some ordinary sentence", claim="", content_type=ctype)
        assert result in UNIT_ROLES or result is None

    @pytest.mark.parametrize("ctype", ["hook", "cta"])
    def test_delivery_shapes_are_technique_with_no_model_call(self, ctype):
        assert role_from_rules(text="anything at all", content_type=ctype) == "technique"

    @pytest.mark.parametrize("ctype", ["tip", "warning", "stat", "myth", "story", "other"])
    def test_subject_bearing_shapes_defer_rather_than_guess(self, ctype):
        # These could be subject or offtopic; the rules cannot tell, and saying
        # so is the correct answer.
        assert role_from_rules(text="tailor your resume to the role", content_type=ctype) is None

    def test_case_and_whitespace_do_not_defeat_the_rule(self):
        assert role_from_rules(text="x", content_type="  CTA  ") == "technique"


class TestMetaMarkers:
    @pytest.mark.parametrize("phrase", [
        "link in bio for the template",
        "follow for more career tips",
        "smash that like button",
        "comment RESUME below and I'll send it",
        "check out my course on interviewing",
        "this video is sponsored by NordVPN",
        "use code ARI for 20% off",
        "DM me if you want the checklist",
    ])
    def test_channel_chatter_is_meta(self, phrase):
        assert role_from_rules(text=phrase, content_type="tip") == "meta"

    def test_marker_in_the_claim_also_counts(self):
        assert role_from_rules(text="", claim="follow for more", content_type="tip") == "meta"

    def test_content_type_wins_over_a_meta_marker(self):
        # A cta saying "link in bio" is still technique — that is what a cta IS.
        assert role_from_rules(text="link in bio", content_type="cta") == "technique"

    @pytest.mark.parametrize("phrase", [
        "recruiters follow a checklist when screening",
        "the hiring manager will comment on your portfolio",
        "my course of action was to apply anyway",
    ])
    def test_narrow_enough_not_to_eat_real_ideas(self, phrase):
        # A false `meta` silently removes a real idea from the generation pool,
        # so the markers must not fire on ordinary use of the same words.
        assert role_from_rules(text=phrase, content_type="tip") is None


class TestValidationNeverGuesses:
    @pytest.mark.parametrize("value", UNIT_ROLES)
    def test_known_roles_round_trip(self, value):
        assert validate_unit_role(value) == value
        assert validate_unit_role(value.upper()) == value

    @pytest.mark.parametrize("value", ["", None, "subjekt", "idea", "SUBJECT_MATTER"])
    def test_unknown_reads_as_unknown_not_subject(self, value):
        # Falling back to "subject" would put unvetted material into the
        # generation pool — the exact failure this axis exists to prevent.
        assert validate_unit_role(value) is None

    def test_prompt_lists_every_role(self):
        rendered = unit_roles_for_prompt()
        for role in UNIT_ROLES:
            assert role in rendered


@pytest.fixture
def db(tmp_path, monkeypatch):
    import core.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "t.duckdb"))
    db_mod.init_db()
    return db_mod


class TestMigration:
    def test_columns_exist_and_running_twice_is_safe(self, db):
        db.init_db()  # second time must not error or clobber
        conn = db.get_connection()
        try:
            cols = {r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='message_units'").fetchall()}
        finally:
            conn.close()
        assert {"unit_role", "unit_role_source", "unit_role_confidence"} <= cols

    def test_existing_rows_start_null_not_subject(self, db):
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO message_units (unit_id, client_id, post_id, text, claim, "
                "content_type) VALUES ('u1','c1','p1','t','c','tip')")
            role, source = conn.execute(
                "SELECT unit_role, unit_role_source FROM message_units "
                "WHERE unit_id='u1'").fetchone()
        finally:
            conn.close()
        # "not classified yet" and "classified as subject" must stay different
        # facts, or we can never tell how much of the corpus was reviewed.
        assert role is None
        assert source is None


def _seed(db, rows):
    conn = db.get_connection()
    try:
        for uid, cid, text, ctype, role, source in rows:
            conn.execute(
                "INSERT INTO message_units (unit_id, client_id, post_id, text, claim, "
                "content_type, unit_role, unit_role_source) VALUES (?,?,?,?,?,?,?,?)",
                [uid, cid, f"p_{uid}", text, text, ctype, role, source])
            # search.py scopes clients through client_posts, not through
            # message_units.client_id, so a unit is invisible to search without
            # this link even when its own client_id is set.
            conn.execute(
                "INSERT INTO client_posts (client_id, post_id, added_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)", [cid, f"p_{uid}"])
    finally:
        conn.close()


def _roles(db, client_id=None):
    conn = db.get_connection()
    try:
        sql = "SELECT unit_id, unit_role, unit_role_source FROM message_units"
        params = []
        if client_id:
            sql += " WHERE client_id = ?"
            params.append(client_id)
        return {r[0]: (r[1], r[2]) for r in conn.execute(sql, params).fetchall()}
    finally:
        conn.close()


def _run(db, *args):
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, str(root / "tools" / "classify_unit_roles.py"), "--db", db.DB_PATH, *args],
        capture_output=True, text=True, cwd=str(root))


class TestBackfill:
    def test_dry_run_writes_nothing(self, db):
        _seed(db, [("u1", "c1", "link in bio", "tip", None, None),
                   ("u2", "c1", "tailor your resume", "cta", None, None)])
        before = _roles(db)
        result = _run(db, "--dry-run")
        assert result.returncode == 0, result.stderr
        # Assert on the database, not on stdout. A tool that prints "DRY RUN"
        # while writing is exactly the failure this guards.
        assert _roles(db) == before
        assert all(role is None for role, _ in _roles(db).values())

    def test_apply_writes_the_rule_decided_rows(self, db):
        _seed(db, [("u1", "c1", "link in bio", "tip", None, None),
                   ("u2", "c1", "anything", "cta", None, None),
                   ("u3", "c1", "tailor your resume to the role", "tip", None, None)])
        assert _run(db, "--apply").returncode == 0
        after = _roles(db)
        assert after["u1"] == ("meta", "rule")
        assert after["u2"] == ("technique", "rule")
        # The rules could not decide u3, so it stays unclassified rather than
        # being guessed into the generation pool.
        assert after["u3"] == (None, None)

    def test_a_human_verdict_is_never_overwritten(self, db):
        # u1 would be rule-classified as meta; a human said it is subject.
        _seed(db, [("u1", "c1", "link in bio", "tip", "subject", "human")])
        assert _run(db, "--apply").returncode == 0
        assert _roles(db)["u1"] == ("subject", "human")

    def test_client_isolation(self, db):
        _seed(db, [("u1", "c1", "link in bio", "tip", None, None),
                   ("u2", "c2", "link in bio", "tip", None, None)])
        assert _run(db, "--apply", "--client", "c1").returncode == 0
        after = _roles(db)
        assert after["u1"] == ("meta", "rule")
        assert after["u2"] == (None, None), "another client's rows must not be touched"


class TestSearchFilter:
    def test_role_filter_narrows_results(self, db):
        from analysis.search import keyword_search
        _seed(db, [("u1", "c1", "resume keyword matching", "tip", "subject", "rule"),
                   ("u2", "c1", "resume keyword matching", "cta", "technique", "rule"),
                   ("u3", "c1", "resume keyword matching", "tip", None, None)])
        all_ids = {r.unit_id for r in keyword_search("resume", "c1", top_k=50)}
        assert {"u1", "u2", "u3"} <= all_ids, "no filter must not filter"

        subject = {r.unit_id for r in keyword_search("resume", "c1", top_k=50,
                                                     unit_role_filter="subject")}
        assert subject == {"u1"}

        unclassified = {r.unit_id for r in keyword_search("resume", "c1", top_k=50,
                                                          unit_role_filter="Unclassified")}
        assert unclassified == {"u3"}, "seeing what has NOT been reviewed is the point"


class TestPreMigrationDatabase:
    """The case CI never exercised: a database that predates the columns.

    Every other test builds its database through init_db(), where unit_role
    always exists. A real database does not, and --dry-run opens it read-only
    so it cannot add the columns itself. That combination produced a raw
    BinderException the first time this ran against real data.
    """

    def _legacy_db(self, tmp_path):
        import duckdb
        path = str(tmp_path / "legacy.duckdb")
        conn = duckdb.connect(path)
        # message_units as it looked before the unit_role migration
        conn.execute(
            "CREATE TABLE message_units (unit_id TEXT PRIMARY KEY, client_id TEXT, "
            "post_id TEXT, text TEXT, claim TEXT, content_type TEXT, confidence DOUBLE)")
        conn.execute("INSERT INTO message_units VALUES ('u1','c1','p1','link in bio','x','tip',0.9)")
        conn.close()
        return path

    def _run(self, db_path, *args):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, str(root / "tools" / "classify_unit_roles.py"),
             "--db", db_path, *args],
            capture_output=True, text=True, cwd=str(root))

    def test_dry_run_explains_instead_of_crashing(self, tmp_path):
        result = self._run(self._legacy_db(tmp_path), "--dry-run")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "BinderException" not in combined, "a stack trace is not an error message"
        assert "unit_role" in combined
        # The message has to carry the fix, not just the diagnosis.
        assert "init_db" in combined

    def test_apply_explains_too(self, tmp_path):
        result = self._run(self._legacy_db(tmp_path), "--apply")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "BinderException" not in combined

    def test_a_migrated_database_is_unaffected(self, tmp_path):
        import duckdb
        path = self._legacy_db(tmp_path)
        conn = duckdb.connect(path)
        for col, typ in (("unit_role", "TEXT"), ("unit_role_source", "TEXT"),
                         ("unit_role_confidence", "DOUBLE")):
            conn.execute(f"ALTER TABLE message_units ADD COLUMN {col} {typ}")
        conn.close()
        result = self._run(path, "--dry-run")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DRY RUN" in result.stdout
