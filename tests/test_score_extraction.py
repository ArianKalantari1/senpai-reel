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
    """The threshold is only retunable 'without a code change' if it can be
    reached from a terminal. A keyword argument on score_pair() cannot be:
    the CLI takes its arguments positionally, so the flag has to be lifted out
    of argv before dispatch or it shifts <db> into the next slot and the
    failure surfaces as a bad database path.
    """

    def _se(self):
        import importlib.util, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("se", root / "tools" / "score_extraction.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    def test_absent_flag_leaves_argv_untouched(self):
        se = self._se()
        assert se.take_threshold(["score", "reels.duckdb"]) == (["score", "reels.duckdb"], None)

    def test_flag_is_removed_so_positionals_keep_their_slots(self):
        se = self._se()
        rest, value = se.take_threshold(
            ["compare", "marked.tsv", "reels.duckdb", "--paraphrase-threshold", "0.25"]
        )
        assert rest == ["compare", "marked.tsv", "reels.duckdb"]
        assert value == 0.25

    def test_flag_may_appear_before_the_subcommand(self):
        se = self._se()
        rest, value = se.take_threshold(["--paraphrase-threshold=0.5", "score", "reels.duckdb"])
        assert rest == ["score", "reels.duckdb"]
        assert value == 0.5

    @pytest.mark.parametrize("argv", [
        ["score", "db", "--paraphrase-threshold", "abc"],
        ["score", "db", "--paraphrase-threshold", "0"],
        ["score", "db", "--paraphrase-threshold", "1.5"],
        ["score", "db", "--paraphrase-threshold", "-0.2"],
        ["score", "db", "--paraphrase-threshold"],
    ])
    def test_unusable_values_are_refused_not_absorbed(self, argv):
        se = self._se()
        # A measuring tool that silently accepts a nonsense threshold reports a
        # number that looks fine and means nothing. Fail at the boundary.
        with pytest.raises(SystemExit):
            se.take_threshold(argv)

    def test_one_point_zero_is_allowed(self):
        se = self._se()
        # Strict, but meaningful: every stemmed content word of the claim
        # present in the source. Only 0 is useless, and only 0 is refused.
        assert se.take_threshold(["score", "db", "--paraphrase-threshold", "1.0"])[1] == 1.0

    def test_the_commands_accept_the_threshold(self):
        se = self._se()
        import inspect
        # Signature only. This test alone is NOT enough — a command can take
        # the argument and never pass it on, and this still passes. The two
        # tests below are the ones that catch that, and they were written
        # because a mutation that made cmd_compare ignore the threshold left
        # this assertion green.
        for fn in (se.cmd_score, se.cmd_compare):
            assert "paraphrase_threshold" in inspect.signature(fn).parameters
        # `blind` deliberately does not: it withholds every scorer opinion.
        assert "paraphrase_threshold" not in inspect.signature(se.cmd_blind).parameters

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
