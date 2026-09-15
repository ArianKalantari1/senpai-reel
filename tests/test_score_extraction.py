from pathlib import Path

import duckdb
import pytest


def _make_score_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "score.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        CREATE TABLE message_units (
            unit_id TEXT,
            post_id TEXT,
            claim TEXT,
            text TEXT,
            topic TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE transcripts (
            post_id TEXT,
            transcript TEXT
        )
        """
    )
    conn.close()
    return db_path


def _insert_unit(conn, unit_id, post_id, claim, text, topic="General"):
    conn.execute(
        """
        INSERT INTO message_units (unit_id, post_id, claim, text, topic)
        VALUES (?, ?, ?, ?, ?)
        """,
        [unit_id, post_id, claim, text, topic],
    )


def _make_duplicate_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "duplicates.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        CREATE TABLE message_units (
            unit_id TEXT,
            post_id TEXT,
            claim TEXT,
            text TEXT,
            topic TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE client_posts (
            client_id TEXT,
            post_id TEXT,
            added_at TIMESTAMP
        )
        """
    )
    conn.close()
    return db_path


def _insert_duplicate_unit(
    conn,
    unit_id,
    post_id,
    claim,
    client_id="client_a",
    text="source transcript",
):
    conn.execute(
        "INSERT INTO client_posts VALUES (?, ?, CURRENT_TIMESTAMP)",
        [client_id, post_id],
    )
    conn.execute(
        """
        INSERT INTO message_units (unit_id, post_id, claim, text, topic)
        VALUES (?, ?, ?, ?, 'CV')
        """,
        [unit_id, post_id, claim, text],
    )


def test_load_left_joins_transcripts(tmp_path):
    from tools.score_extraction import load

    db_path = _make_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_unit(
        conn,
        "u1",
        "p1",
        "Use explicit salary evidence before asking for a raise",
        "Bring salary evidence into the raise conversation",
    )
    _insert_unit(
        conn,
        "u2",
        "p2",
        "Ask for feedback after interviews",
        "Ask for feedback after interviews",
    )
    conn.execute(
        "INSERT INTO transcripts (post_id, transcript) VALUES ('p1', 'Use explicit salary evidence before asking for a raise today')"
    )
    conn.close()

    rows = load(db_path)

    assert rows == [
        (
            "u1",
            "Use explicit salary evidence before asking for a raise",
            "Bring salary evidence into the raise conversation",
            "Use explicit salary evidence before asking for a raise today",
            "General",
        ),
        (
            "u2",
            "Ask for feedback after interviews",
            "Ask for feedback after interviews",
            None,
            "General",
        ),
    ]


def test_score_transcript_reports_claim_lifted_and_text_match_separately():
    from tools.score_extraction import score_transcript

    transcript = (
        "Use explicit salary evidence before asking for a raise, then pause "
        "so your manager has room to respond."
    )

    claim_lifted = score_transcript(
        "Use explicit salary evidence before asking for a raise",
        "Bring concrete compensation examples to the conversation",
        transcript,
    )
    text_match = score_transcript(
        "Ask with supporting proof",
        "Use explicit salary evidence before asking for a raise",
        transcript,
    )

    assert claim_lifted["flags"] == ["LIFTED"]
    assert claim_lifted["claim_lifted"] is True
    assert claim_lifted["text_matches_transcript"] is False
    assert text_match["flags"] == ["OK"]
    assert text_match["claim_lifted"] is False
    assert text_match["text_matches_transcript"] is True


def test_score_summary_uses_transcript_coverage_denominator(tmp_path, capsys):
    from tools.score_extraction import cmd_score

    db_path = _make_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_unit(
        conn,
        "u1",
        "p1",
        "Use explicit salary evidence before asking for a raise",
        "Bring concrete compensation examples to the conversation",
    )
    _insert_unit(
        conn,
        "u2",
        "p2",
        "Ask with supporting proof",
        "Use explicit salary evidence before asking for a raise",
    )
    _insert_unit(
        conn,
        "u3",
        "p3",
        "Follow up after an interview",
        "Follow up after an interview",
    )
    conn.execute(
        """
        INSERT INTO transcripts VALUES
        ('p1', 'Use explicit salary evidence before asking for a raise today'),
        ('p2', 'Use explicit salary evidence before asking for a raise today')
        """
    )
    conn.close()

    cmd_score(db_path)
    out = capsys.readouterr().out

    assert "Structural signals" in out
    assert "transcript coverage 2/3" in out
    assert "LIFTED                    1/2" in out
    assert "text matches source       1/2" in out


def test_blind_writes_transcript_excerpt(tmp_path):
    from tools.score_extraction import cmd_blind

    db_path = _make_score_db(tmp_path)
    long_transcript = (
        "opening filler " * 80
        + "Use explicit salary evidence before asking for a raise "
        + "closing filler " * 80
    )
    conn = duckdb.connect(db_path)
    _insert_unit(
        conn,
        "u1",
        "p1",
        "Use explicit salary evidence before asking for a raise",
        "Bring salary evidence into the raise conversation",
    )
    conn.execute("INSERT INTO transcripts VALUES ('p1', ?)", [long_transcript])
    conn.close()

    out_path = tmp_path / "sample.tsv"
    cmd_blind(db_path, 1, str(out_path))

    text = out_path.read_text(encoding="utf-8")
    header, row = [line for line in text.splitlines() if not line.startswith("#")]
    assert header == "unit_id\tVERDICT\tclaim\tsource\ttranscript_excerpt"
    assert "Use explicit salary evidence before asking for a raise" in row
    assert len(row) < len(long_transcript)


def test_blind_can_sample_one_extraction_run(tmp_path):
    from tools.score_extraction import cmd_blind

    db_path = _make_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    conn.execute("ALTER TABLE message_units ADD COLUMN extraction_run_id TEXT")
    _insert_unit(
        conn,
        "old_unit",
        "p1",
        "old claim",
        "old source",
    )
    _insert_unit(
        conn,
        "new_unit",
        "p2",
        "new claim",
        "new source",
    )
    conn.execute(
        "UPDATE message_units SET extraction_run_id = 'sample-run' WHERE unit_id = 'new_unit'"
    )
    conn.close()

    out_path = tmp_path / "sample.tsv"
    cmd_blind(db_path, 10, str(out_path), run_id="sample-run")

    text = out_path.read_text(encoding="utf-8")
    assert "new_unit" in text
    assert "new claim" in text
    assert "old_unit" not in text
    assert "old claim" not in text


def test_blind_cli_can_sample_one_extraction_run(tmp_path):
    import pathlib
    import subprocess
    import sys

    db_path = _make_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    conn.execute("ALTER TABLE message_units ADD COLUMN extraction_run_id TEXT")
    _insert_unit(conn, "old_unit", "p1", "old claim", "old source")
    _insert_unit(conn, "new_unit", "p2", "new claim", "new source")
    conn.execute(
        "UPDATE message_units SET extraction_run_id = 'sample-run' WHERE unit_id = 'new_unit'"
    )
    conn.close()

    root = pathlib.Path(__file__).resolve().parent.parent
    out_path = tmp_path / "sample.tsv"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "score_extraction.py"),
            "blind",
            db_path,
            "10",
            str(out_path),
            "--run-id",
            "sample-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )

    assert result.returncode == 0, result.stderr
    text = out_path.read_text(encoding="utf-8")
    assert "new_unit" in text
    assert "old_unit" not in text
    assert "Sample restricted to extraction_run_id='sample-run'." in result.stdout


def test_duplicates_flags_redundant_cv_template_claims(tmp_path, capsys):
    from tools.score_extraction import cmd_duplicates

    transcript = (
        "CV mistakes you should avoid. Let's make pink, add a cute selfie... "
        "let's take this standardized template. Black and white. Simple, easy to scan."
    )
    db_path = _make_duplicate_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_duplicate_unit(
        conn,
        "u_colour_selfie",
        "post_cv",
        "Colorful templates and selfies are not suitable for a professional CV",
        text=transcript,
    )
    _insert_duplicate_unit(
        conn,
        "u_bw_effective",
        "post_cv",
        "A black and white template is more effective for a CV than a colorful one",
        text=transcript,
    )
    _insert_duplicate_unit(
        conn,
        "u_standardized_scan",
        "post_cv",
        "A standardized black and white template is easier to scan and more professional",
        text=transcript,
    )
    conn.close()

    cmd_duplicates(db_path, "client_a")
    out = capsys.readouterr().out

    assert "Near-duplicate units 3/3 scored (100.0% of scored, 100.0% of all visible units)" in out
    assert "Same-post duplicate pairs: 2" in out
    assert "Cross-post duplicate pairs: 0" in out
    assert "post_cv" in out
    assert "u_bw_effective <-> u_colour_selfie" in out
    assert "u_bw_effective <-> u_standardized_scan" in out
    assert "Colorful templates and selfies are not suitable" in out
    assert "standardized black and white template is easier to scan" in out
    assert "3 redundant unit(s)" in out


def test_duplicates_reports_cross_post_pairs(tmp_path, capsys):
    from tools.score_extraction import cmd_duplicates

    db_path = _make_duplicate_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_duplicate_unit(
        conn,
        "u1",
        "post_a",
        "Tailor your resume to match the job description",
    )
    _insert_duplicate_unit(
        conn,
        "u2",
        "post_b",
        "A resume should be tailored to the job description",
    )
    conn.close()

    cmd_duplicates(db_path, "client_a")
    out = capsys.readouterr().out

    assert "Same-post duplicate pairs: 0" in out
    assert "Cross-post duplicate pairs: 1" in out
    assert "post_a <-> post_b" in out
    assert "u1 <-> u2" in out


def test_duplicates_scopes_to_one_client(tmp_path, capsys):
    from tools.score_extraction import cmd_duplicates

    db_path = _make_duplicate_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_duplicate_unit(conn, "client_a_unique", "post_a", "Ask for interview feedback")
    _insert_duplicate_unit(
        conn,
        "client_b_dup_1",
        "post_b1",
        "Tailor your resume to match the job description",
        client_id="client_b",
    )
    _insert_duplicate_unit(
        conn,
        "client_b_dup_2",
        "post_b2",
        "A resume should be tailored to the job description",
        client_id="client_b",
    )
    conn.close()

    cmd_duplicates(db_path, "client_a")
    out = capsys.readouterr().out

    assert "Near-duplicate units 0/1 scored" in out
    assert "client_b_dup" not in out
    assert "post_b1" not in out
    assert "post_b2" not in out


def test_duplicates_respects_threshold(tmp_path, capsys):
    from tools.score_extraction import cmd_duplicates

    db_path = _make_duplicate_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_duplicate_unit(
        conn,
        "u1",
        "post_a",
        "Tailor your resume to match the job description",
    )
    _insert_duplicate_unit(
        conn,
        "u2",
        "post_b",
        "A resume should be tailored to the job description",
    )
    conn.close()

    cmd_duplicates(db_path, "client_a", paraphrase_threshold=0.95)
    out = capsys.readouterr().out

    assert "Near-duplicate units 0/2 scored" in out
    assert "Cross-post duplicate pairs: 0" in out


def test_duplicates_reports_claim_coverage_without_treating_missing_as_unique(tmp_path, capsys):
    from tools.score_extraction import cmd_duplicates

    db_path = _make_duplicate_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_duplicate_unit(conn, "with_claim", "post_a", "Ask for interview feedback")
    _insert_duplicate_unit(conn, "missing_claim", "post_b", None)
    conn.close()

    cmd_duplicates(db_path, "client_a")
    out = capsys.readouterr().out

    assert "Scored claims 1/2 units (50.0%)" in out
    assert "Missing/empty claims are unknown, not unique." in out
    assert "Near-duplicate units 0/1 scored (0.0% of scored, 0.0% of all visible units)" in out


def test_duplicates_cli_rejects_bad_threshold(tmp_path):
    import pathlib
    import subprocess
    import sys

    db_path = _make_duplicate_db(tmp_path)
    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "score_extraction.py"),
            "duplicates",
            db_path,
            "--client",
            "client_a",
            "--paraphrase-threshold",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )

    assert result.returncode != 0
    assert "--paraphrase-threshold" in result.stderr
    assert "outside 0-1" in result.stderr


def test_compare_counts_lifted_as_lazy_not_wrong(tmp_path, capsys):
    from tools.score_extraction import cmd_compare

    marked = tmp_path / "marked.tsv"
    marked.write_text(
        "\n".join(
            [
                "unit_id\tVERDICT\tclaim\tsource\ttranscript_excerpt",
                (
                    "lazy_lifted\tlazy\t"
                    "salary evidence improves timing\t"
                    "compensation proof helps negotiation\t"
                    "salary evidence improves timing in raise conversations"
                ),
                (
                    "wrong_lifted\twrong\t"
                    "salary evidence improves timing\t"
                    "compensation proof helps negotiation\t"
                    "salary evidence improves timing in raise conversations"
                ),
            ]
        ),
        encoding="utf-8",
    )

    cmd_compare(str(marked))
    out = capsys.readouterr().out

    assert "Agreement with your reading: 1/2" in out
    assert "you said wrong" in out
    assert "LIFTED" in out


def test_compare_can_use_full_transcript_from_db(tmp_path, capsys):
    from tools.score_extraction import cmd_compare

    db_path = _make_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_unit(
        conn,
        "u1",
        "p1",
        "salary evidence improves timing",
        "compensation proof helps negotiation",
    )
    conn.execute(
        "INSERT INTO transcripts VALUES ('p1', 'salary evidence improves timing in raise conversations')"
    )
    conn.close()

    marked = tmp_path / "marked_db.tsv"
    marked.write_text(
        "\n".join(
            [
                "unit_id\tVERDICT\tclaim\tsource\ttranscript_excerpt",
                "u1\tlazy\tsalary evidence improves timing\tcompensation proof helps negotiation\topening excerpt only",
            ]
        ),
        encoding="utf-8",
    )

    cmd_compare(str(marked), db_path)
    out = capsys.readouterr().out

    assert "Agreement with your reading: 1/1" in out


class TestCompareWarnsWhenTranscriptIsDegraded:
    """Omitting the db path scores LIFTED against a truncated excerpt.

    transcript_excerpt() anchors on the earliest matching claim word, so a
    phrase lifted from later in the transcript falls outside the window and
    LIFTED does not fire. Demonstrated: the same row scores 100% agreement with
    the database and 0% without it, and the tool attributes the gap to its own
    unreliability. Silence there is a wrong number with a verdict attached.
    """

    def _marked_file(self, tmp_path):
        import importlib.util, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("se", root / "tools" / "score_extraction.py")
        se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)

        lifted = "quantified measurable achievements outperform generic responsibility statements"
        claim = f"recruiters prefer {lifted}"
        text = "some unrelated snippet"
        transcript = ("recruiters are busy people. "
                      + " ".join(["padding filler words about nothing in particular here"] * 60)
                      + f" {lifted}")
        excerpt = se.transcript_excerpt(transcript, claim, text)
        assert lifted[:30] not in excerpt, "fixture must actually truncate the lifted phrase"

        path = tmp_path / "marked.tsv"
        path.write_text(
            "unit_id\tVERDICT\tclaim\tsource\ttranscript_excerpt\n"
            f"u1\tlazy\t{se.tsv_cell(claim)}\t{se.tsv_cell(text)}\t{se.tsv_cell(excerpt)}\n",
            encoding="utf-8")
        return str(path)

    def _run(self, *args):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, str(root / "tools" / "score_extraction.py"), "compare", *args],
            capture_output=True, text=True, cwd=str(root))

    def test_warns_when_no_database_is_given(self, tmp_path):
        out = self._run(self._marked_file(tmp_path)).stdout
        assert "WARNING" in out
        assert "not trustworthy" in out
        assert "compare <marked.tsv> <db>" in out

    def test_the_warning_names_the_actual_risk(self, tmp_path):
        out = self._run(self._marked_file(tmp_path)).stdout
        # Naming LIFTED matters: a generic "results may vary" would not tell the
        # reader which signal to distrust.
        assert "LIFTED" in out
        assert "truncated excerpt" in out


class TestSemanticLaziness:
    """A claim that restates its source IN DIFFERENT WORDS used to score clean.

    That was the single biggest reason `compare` agreed with a careful human
    read only 57% of the time on 2026-09-13: seven of the fifteen disagreements
    were units a human called "lazy" and the tool called OK.

    The old PARAPHRASE check needed 0.60 raw-token overlap. Restating something
    in your own words shares almost no raw tokens, so it never fired.
    """

    def _se(self):
        import importlib.util, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("se", root / "tools" / "score_extraction.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    # Real pairs from the 2026-09-13 blind sample, marked "lazy" by hand.
    LAZY = [
        ("Cracks in life come from living fully, not from failure.",
         "Careers crack, confidence cracks, relationship crack. Not because you "
         "failed, but because you lived."),
        ("Taking action is the most important step in building high agency.",
         "The number one thing you can do to practice high agency is develop a "
         "bias for action."),
        ("A simple method can save job seekers significant money on resume-boosting certificates.",
         "I'm going to save you thousands of dollars on certificates for your "
         "resume using this one very simple hack."),
        ("The AI agent autonomously finds job roles matching your criteria.",
         "Now you have an AI agent that'll automatically search for the best fit "
         "roles for you."),
    ]

    # Marked "good" by hand. These must NOT start flagging — a false
    # PARAPHRASE costs trust in a tool whose whole job is being trustworthy.
    GOOD = [
        ("Unfamiliarity, not dislike, is the real barrier for AI brands.",
         "Lack of familiarity was the most common reason for rejection."),
        ("Corporate environments make high achievers dependent and vulnerable to systemic shocks.",
         "Most smart, ambitious people have been fragilized by corporate systems."),
        ("Regular confidence-building reduces the depth of confidence dips.",
         "The more you work on your confidence, the higher you can raise your threshold."),
        ("Automated hiring processes feel so impersonal that they seem AI-generated.",
         "Wait, is this chatgpt?"),
        ("Inspiration involves adapting ideas creatively, not replicating them verbatim.",
         "Imitation is the highest form of flattery, but there's a difference "
         "between inspiration and copying."),
    ]

    def test_stemming_collapses_inflections(self):
        se = self._se()
        assert se.stem("lived") == se.stem("living")
        assert se.stem("cracks") == se.stem("crack")
        assert se.stem("customers") == se.stem("customer")

    def test_derivational_morphology_is_out_of_scope(self):
        se = self._se()
        # "failed" -> "fail" but "failure" stays put: verb-to-noun derivation
        # needs "-ure" stripped, which would maul "measure" -> "meas" and
        # "figure" -> "fig". Inflection is worth collapsing; derivation is not
        # worth the collateral damage at this sample size.
        assert se.stem("failed") != se.stem("failure")

    def test_stemming_does_not_maul_short_words(self):
        se = self._se()
        # Stripping below three characters produces collisions that would
        # quietly inflate every overlap score.
        for w in ("is", "was", "key", "day", "buy"):
            assert len(se.stem(w)) >= 3 or se.stem(w) == w

    def test_it_now_catches_restatement_in_different_words(self):
        se = self._se()
        caught = sum(1 for c, t in self.LAZY
                     if "PARAPHRASE" in se.score_pair(c, t)["flags"])
        # Not all of them — this is an improvement, not a solution, and the
        # docstring in the tool says so. Pinning the floor we calibrated to.
        assert caught >= 3, f"only caught {caught}/{len(self.LAZY)}"

    def test_the_old_threshold_caught_none_of_them(self):
        se = self._se()
        caught = sum(1 for c, t in self.LAZY
                     if "PARAPHRASE" in se.score_pair(c, t, paraphrase_threshold=0.60)["flags"])
        # The regression this fixes, kept executable rather than described.
        assert caught == 0

    def test_good_abstractions_stay_clean(self):
        se = self._se()
        for claim, text in self.GOOD:
            flags = se.score_pair(claim, text)["flags"]
            assert "PARAPHRASE" not in flags, f"false positive on: {claim[:60]}"

    def test_threshold_is_tunable_without_a_code_change(self):
        se = self._se()
        claim, text = self.LAZY[0]
        assert "PARAPHRASE" in se.score_pair(claim, text, paraphrase_threshold=0.10)["flags"]
        assert "PARAPHRASE" not in se.score_pair(claim, text, paraphrase_threshold=0.95)["flags"]

    def test_both_overlaps_are_reported(self):
        se = self._se()
        r = se.score_pair(*self.LAZY[0])
        # Keeping the raw figure visible means the effect of stemming stays
        # auditable instead of being folded invisibly into one number.
        assert r["overlap"] >= r["raw_overlap"]


class TestParaphraseThresholdFlag:
    """The threshold is only retunable if it reaches score and compare."""

    def _se(self):
        import importlib.util, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("se", root / "tools" / "score_extraction.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    @pytest.mark.parametrize("value", ["abc", "0", "1.5", "-0.2"])
    def test_unusable_values_are_refused_not_absorbed(self, value):
        import argparse
        se = self._se()
        # A measuring tool that silently accepts a nonsense threshold reports a
        # number that looks fine and means nothing. Fail at the boundary.
        with pytest.raises(argparse.ArgumentTypeError):
            se._paraphrase_threshold(value)

    def test_one_point_zero_is_allowed(self):
        se = self._se()
        # Strict, but meaningful: every stemmed content word of the claim
        # present in the source. Only 0 is useless, and only 0 is refused.
        assert se._paraphrase_threshold("1.0") == 1.0

    def test_bare_invocation_prints_docstring_and_exits_2(self):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, str(root / "tools" / "score_extraction.py")],
            capture_output=True, text=True, cwd=str(root))
        assert result.returncode == 2
        assert "KNOWN GAPS" in result.stdout
        assert "no API key, no spend, no LLM judging an LLM" in result.stdout

    def test_blind_rejects_threshold_and_names_compare(self):
        import subprocess, sys, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "score_extraction.py"),
                "blind",
                "reels.duckdb",
                "--paraphrase-threshold",
                "0.3",
            ],
            capture_output=True, text=True, cwd=str(root))
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "--paraphrase-threshold" in combined
        assert "compare" in combined

    # One real pair from the 2026-09-13 blind sample, hand-marked "lazy".
    # Stemmed overlap 0.33 — above the shipped 0.30, so it is a live case on
    # both sides of the threshold rather than a constructed one.
    _LAZY_CLAIM = "Cracks in life come from living fully, not from failure."
    _LAZY_SOURCE = ("Careers crack, confidence cracks, relationship crack. "
                    "Not because you failed, but because you lived.")

    def _db_with_the_lazy_pair(self, tmp_path):
        db_path = _make_score_db(tmp_path)
        conn = duckdb.connect(db_path)
        conn.execute(
            "INSERT INTO message_units VALUES (?, ?, ?, ?, ?)",
            ["u1", "p1", self._LAZY_CLAIM, self._LAZY_SOURCE, "career"],
        )
        conn.close()
        return db_path

    @staticmethod
    def _flags_tallied(out: str) -> set[str]:
        """Flag names from the tally block only.

        Not a substring search on the whole report: it closes with a legend
        naming every flag, so `"PARAPHRASE" in out` is true whatever the
        threshold does. That false-negative-proof version of this test passed
        against a threshold that was never applied.
        """
        flags = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isupper() and parts[1].rstrip(",").isdigit():
                flags.add(parts[0])
        return flags

    def test_score_actually_uses_the_threshold(self, tmp_path, capsys):
        se = self._se()
        db_path = self._db_with_the_lazy_pair(tmp_path)

        se.cmd_score(db_path, 0.10)
        assert "PARAPHRASE" in self._flags_tallied(capsys.readouterr().out)

        se.cmd_score(db_path, 0.95)
        assert "PARAPHRASE" not in self._flags_tallied(capsys.readouterr().out)

    def test_compare_actually_uses_the_threshold(self, tmp_path, capsys):
        se = self._se()
        marked = tmp_path / "marked.tsv"
        marked.write_text(
            "unit_id\tVERDICT\tclaim\tsource\n"
            f"u1\tlazy\t{self._LAZY_CLAIM}\t{self._LAZY_SOURCE}\n",
            encoding="utf-8",
        )

        # Human said "lazy". A permissive threshold makes the tool agree; a
        # strict one makes it disagree. If the threshold never reaches
        # score_pair, both runs report the same figure.
        se.cmd_compare(str(marked), None, 0.10)
        assert "1/1" in capsys.readouterr().out

        se.cmd_compare(str(marked), None, 0.95)
        assert "0/1" in capsys.readouterr().out


class TestBlindRefusesToWriteAnEmptySample:
    """"Wrote 0 units" followed by instructions is a quiet failure.

    A real run hit this: `blind --run-id stance-check` was called before the
    write run had succeeded, so nothing carried that id. The tool wrote an
    empty file, printed "Wrote 0 units to stance.tsv", and then told the
    operator to go and mark it.
    """

    def _se(self):
        import importlib.util, pathlib, sys
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("se2", root / "tools" / "score_extraction.py")
        m = importlib.util.module_from_spec(spec); sys.modules["se2"] = m
        spec.loader.exec_module(m)
        return m

    def test_unknown_run_id_exits_and_names_the_id(self, tmp_path):
        se = self._se()
        db_path = _make_score_db(tmp_path)
        conn = duckdb.connect(db_path)
        # The column has to exist or the migration guard fires first with a
        # different (also correct) message. A fixture that cannot reach the
        # code under test is not testing it.
        conn.execute("ALTER TABLE message_units ADD COLUMN extraction_run_id TEXT")
        conn.execute(
            "INSERT INTO message_units (unit_id, post_id, claim, text, topic) "
            "VALUES ('u1','p1','a claim','a source','careers')"
        )
        conn.close()
        out = tmp_path / "empty.tsv"
        with pytest.raises(SystemExit) as exc:
            se.cmd_blind(db_path, 10, str(out), run_id="never-written")
        assert "never-written" in str(exc.value)
        # And it must not leave a misleading empty file behind.
        assert not out.exists()

    def test_empty_database_exits_rather_than_writing_nothing(self, tmp_path):
        se = self._se()
        db_path = _make_score_db(tmp_path)
        out = tmp_path / "empty.tsv"
        with pytest.raises(SystemExit):
            se.cmd_blind(db_path, 10, str(out))
        assert not out.exists()

    def test_a_real_sample_still_writes(self, tmp_path, capsys):
        se = self._se()
        db_path = _make_score_db(tmp_path)
        conn = duckdb.connect(db_path)
        conn.execute(
            "INSERT INTO message_units VALUES ('u1','p1','a claim','a source','careers')"
        )
        conn.close()
        out = tmp_path / "sample.tsv"
        se.cmd_blind(db_path, 10, str(out))
        assert out.exists()
        # Asking for more than exist is not an error, but it is worth saying.
        assert "only 1 were available" in capsys.readouterr().out


class TestTheCliCarriesFlagsThroughToTheCommands:
    """The dispatcher is the thing argparse replaced, so test the dispatcher.

    TestParaphraseThresholdFlag above calls cmd_score/cmd_compare directly, so it
    stays green when main() parses a flag correctly and then forgets to pass
    it on. Three separate mutations of the dispatch lines survived that suite
    untouched. These tests run the real command line instead.
    """

    _CLAIM = "Cracks in life come from living fully, not from failure."
    _SOURCE = ("Careers crack, confidence cracks, relationship crack. "
               "Not because you failed, but because you lived.")

    def _run(self, args, cwd):
        import subprocess, sys
        return subprocess.run(
            [sys.executable, str(cwd / "tools" / "score_extraction.py"), *args],
            capture_output=True, text=True, cwd=str(cwd))

    def _root(self):
        import pathlib
        return pathlib.Path(__file__).resolve().parent.parent

    def test_compare_threshold_reaches_the_agreement_figure(self, tmp_path):
        marked = tmp_path / "marked.tsv"
        marked.write_text(
            "unit_id\tVERDICT\tclaim\tsource\n"
            f"u1\tlazy\t{self._CLAIM}\t{self._SOURCE}\n",
            encoding="utf-8")
        root = self._root()

        permissive = self._run(
            ["compare", str(marked), "--paraphrase-threshold", "0.10"], root)
        strict = self._run(
            ["compare", str(marked), "--paraphrase-threshold", "0.95"], root)

        assert "1/1" in permissive.stdout, permissive.stdout + permissive.stderr
        assert "0/1" in strict.stdout, strict.stdout + strict.stderr

    def test_score_threshold_reaches_the_tally(self, tmp_path):
        db_path = _make_score_db(tmp_path)
        conn = duckdb.connect(db_path)
        conn.execute("INSERT INTO message_units VALUES (?, ?, ?, ?, ?)",
                     ["u1", "p1", self._CLAIM, self._SOURCE, "career"])
        conn.close()
        root = self._root()

        permissive = self._run(
            ["score", db_path, "--paraphrase-threshold", "0.10"], root)
        strict = self._run(
            ["score", db_path, "--paraphrase-threshold", "0.95"], root)

        tallied = TestParaphraseThresholdFlag._flags_tallied
        assert "PARAPHRASE" in tallied(permissive.stdout), permissive.stdout
        assert "PARAPHRASE" not in tallied(strict.stdout), strict.stdout

    def test_score_by_account_flag_reaches_the_account_report(self, tmp_path):
        db_path = _make_account_score_db(tmp_path)
        conn = duckdb.connect(db_path)
        _insert_account_post(conn, "client_a", "acc_cli", "cli_creator", "cli_post")
        _insert_unit(
            conn,
            "cli_unit",
            "cli_post",
            "Specific metrics make resume bullets easier to judge",
            "Use numbers in resume bullets so recruiters see evidence",
        )
        conn.close()

        result = self._run(
            ["score", db_path, "--by-account", "--client", "client_a"],
            self._root(),
        )

        assert result.returncode == 0, result.stderr
        assert "Extraction quality by account" in result.stdout
        assert "@cli_creator" in result.stdout

    def test_run_id_on_score_names_blind_rather_than_just_refusing(self, tmp_path):
        result = self._run(["score", "reels.duckdb", "--run-id", "x"], self._root())
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        # Not `"blind" in combined`: argparse's own usage line lists every
        # subcommand, so that assertion passes with the guard deleted. It has
        # to be the sentence that tells the operator where the flag belongs.
        assert "only `blind` selects an extraction run" in combined, combined


def _make_account_score_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "account_score.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        CREATE TABLE creator_accounts (
            account_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE posts (
            post_id TEXT PRIMARY KEY,
            client_id TEXT,
            account_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE client_posts (
            client_id TEXT,
            post_id TEXT,
            added_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_units (
            unit_id TEXT,
            post_id TEXT,
            claim TEXT,
            text TEXT,
            topic TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE transcripts (
            post_id TEXT,
            transcript TEXT
        )
        """
    )
    conn.close()
    return db_path
def _insert_account_post(conn, client_id, account_id, username, post_id):
    conn.execute(
        "INSERT OR IGNORE INTO creator_accounts VALUES (?, ?)",
        [account_id, username],
    )
    conn.execute(
        "INSERT INTO posts VALUES (?, ?, ?)",
        [post_id, client_id, account_id],
    )
    conn.execute(
        "INSERT INTO client_posts VALUES (?, ?, CURRENT_TIMESTAMP)",
        [client_id, post_id],
    )
def _line_for_account(out: str, username: str) -> str:
    return next(line for line in out.splitlines() if f"@{username}" in line)
def _account_columns(out: str, username: str) -> list[str]:
    return _line_for_account(out, username).split()
def test_score_by_account_reports_flags_and_sorts_cleanest_first(tmp_path, capsys):
    from tools.score_extraction import cmd_score

    db_path = _make_account_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_account_post(conn, "client_a", "acc_clean", "clean_creator", "clean_1")
    _insert_account_post(conn, "client_a", "acc_clean", "clean_creator", "clean_2")
    _insert_account_post(conn, "client_a", "acc_bad", "bad_creator", "bad_1")
    _insert_account_post(conn, "client_a", "acc_empty", "empty_creator", "empty_1")
    _insert_unit(
        conn,
        "clean_u1",
        "clean_1",
        "Specific metrics make resume bullets easier to judge",
        "Use numbers in resume bullets so recruiters see evidence",
    )
    _insert_unit(
        conn,
        "clean_u2",
        "clean_2",
        "Follow-up emails should add context instead of repeating thanks",
        "Send interview follow-ups with a useful detail from the conversation",
    )
    _insert_unit(
        conn,
        "copied_u",
        "bad_1",
        "Use explicit salary evidence before asking for a raise",
        "Use explicit salary evidence before asking for a raise",
    )
    _insert_unit(
        conn,
        "paraphrase_u",
        "bad_1",
        "Cracks in life come from living fully, not from failure.",
        "Careers crack, confidence cracks, relationship crack. Not because you failed, but because you lived.",
    )
    _insert_unit(
        conn,
        "vague_u",
        "bad_1",
        "This demonstrates important effective valuable insights for professionals",
        "Specific salary negotiation scripts require numbers",
    )
    _insert_unit(
        conn,
        "lifted_u",
        "bad_1",
        "Portfolio projects prove practical skill",
        "Show work samples during the hiring process",
    )
    conn.execute(
        """
        INSERT INTO transcripts VALUES
        ('clean_1', 'interview preparation requires concrete examples and calm delivery'),
        ('clean_2', 'follow up with one useful detail after the interview'),
        ('bad_1', 'portfolio projects prove practical skill during hiring')
        """
    )
    conn.close()

    cmd_score(db_path, by_account=True, client_id="client_a")
    out = capsys.readouterr().out

    assert out.index("@clean_creator") < out.index("@bad_creator") < out.index("@empty_creator")
    assert _account_columns(out, "clean_creator") == [
        "@clean_creator",
        "2",
        "2",
        "1.00",
        "100.0%",
        "0.0%",
        "0.0%",
        "0.0%",
        "0.0%",
    ]
    assert _account_columns(out, "bad_creator") == [
        "@bad_creator",
        "4",
        "1",
        "4.00",
        "0.0%",
        "25.0%",
        "25.0%",
        "25.0%",
        "25.0%",
    ]
    empty_columns = _account_columns(out, "empty_creator")
    assert empty_columns[:4] == ["@empty_creator", "0", "1", "0.00"]
    assert empty_columns[4:] == ["no", "data", "no", "data", "no", "data", "no", "data", "no", "data"]
    assert "0.0%" not in _line_for_account(out, "empty_creator")
def test_score_by_account_scopes_to_client_posts(tmp_path, capsys):
    from tools.score_extraction import cmd_score

    db_path = _make_account_score_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_account_post(conn, "client_a", "acc_a", "client_a_creator", "a_post")
    _insert_account_post(conn, "client_b", "acc_b", "client_b_creator", "b_post")
    _insert_unit(
        conn,
        "a_unit",
        "a_post",
        "Specific metrics make resume bullets easier to judge",
        "Use numbers in resume bullets so recruiters see evidence",
    )
    _insert_unit(
        conn,
        "b_unit",
        "b_post",
        "Use explicit salary evidence before asking for a raise",
        "Use explicit salary evidence before asking for a raise",
    )
    conn.execute(
        """
        INSERT INTO transcripts VALUES
        ('a_post', 'interview preparation requires concrete examples'),
        ('b_post', 'Use explicit salary evidence before asking for a raise')
        """
    )
    conn.close()

    cmd_score(db_path, by_account=True, client_id="client_a")
    out = capsys.readouterr().out

    assert "@client_a_creator" in out
    assert "@client_b_creator" not in out
    assert "b_unit" not in out
def test_score_by_account_requires_client(tmp_path):
    from tools.score_extraction import cmd_score

    db_path = _make_account_score_db(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cmd_score(db_path, by_account=True)
    assert "--client CLIENT" in str(exc.value)


def test_duplicate_pairs_stem_each_claim_once(monkeypatch):
    """The stemming must not live inside the O(n^2) loop.

    Comparison is pairwise, so anything in the inner loop runs ~14 million
    times on the real 5,354-unit corpus. Stemming both claims there meant
    every claim was re-tokenised once per other unit — 28 million stemming
    calls for 5,354 claims, and a projected 13.5-minute run that the fix
    brought to 32 seconds with identical output.

    Asserting on wall-clock would be flaky, so this asserts the property that
    causes it: each claim is stemmed exactly once.
    """
    import importlib.util, pathlib, sys
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("se_perf", root / "tools" / "score_extraction.py")
    se = importlib.util.module_from_spec(spec); sys.modules["se_perf"] = se
    spec.loader.exec_module(se)

    units = [
        se.DuplicateUnit(f"u{i}", f"p{i // 3}", f"Tailor your resume for role number {i}")
        for i in range(30)
    ]

    calls = []
    real = se.claim_stems
    monkeypatch.setattr(se, "claim_stems", lambda claim: calls.append(claim) or real(claim))

    se.find_duplicate_pairs(units, se.DEFAULT_PARAPHRASE_THRESHOLD)

    # 30 units: once each. The pre-fix version made 2 * (30*29/2) = 870 calls.
    assert len(calls) == len(units), f"stemmed {len(calls)} times for {len(units)} units"


def test_duplicate_pairs_find_the_same_pairs_as_pairwise_restemming():
    """The fast path must agree with the obvious slow one, exactly."""
    import importlib.util, pathlib, sys, random
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("se_equiv", root / "tools" / "score_extraction.py")
    se = importlib.util.module_from_spec(spec); sys.modules["se_equiv"] = se
    spec.loader.exec_module(se)

    random.seed(7)
    vocab = ("resume interview recruiter linkedin salary cover letter ats keyword "
             "tailor skill experience manager hiring role apply job search").split()
    units = [
        se.DuplicateUnit(f"u{i}", f"p{i // 4}", " ".join(random.sample(vocab, 8)))
        for i in range(80)
    ]

    fast = {(p.left.unit_id, p.right.unit_id, round(p.overlap, 12))
            for p in se.find_duplicate_pairs(units, se.DEFAULT_PARAPHRASE_THRESHOLD)}
    slow = set()
    for i, left in enumerate(units):
        for right in units[i + 1:]:
            overlap = se.shared_stem_overlap(left.claim, right.claim)
            if overlap is not None and overlap >= se.DEFAULT_PARAPHRASE_THRESHOLD:
                slow.add((left.unit_id, right.unit_id, round(overlap, 12)))

    assert fast == slow
    assert slow, "fixture produced no duplicate pairs, so the comparison proves nothing"
