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
