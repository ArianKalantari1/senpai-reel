import csv
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb


def _openai_response(content: str, in_tokens: int = 100, out_tokens: int = 50):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
    }
    return mock_resp


def _seed_generation_db(tmp_path: Path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "generation.duckdb")
    db_mod.init_db()
    db_path = db_mod.DB_PATH
    client_id = db_mod.DEFAULT_CLIENT_ID
    now = datetime.utcnow()

    conn = duckdb.connect(db_path)
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
        INSERT INTO message_units (
            unit_id, client_id, post_id, text, claim, topic, content_type,
            confidence, extracted_at, model, unit_role
        )
        VALUES
          (
            'allowed_unit', ?, 'post_1',
            'Allowed source wording that must not be needed for bare generation',
            'Allowed reference claim', 'Resume', 'tip', 0.9, ?, 'gpt-4o-mini',
            NULL
          ),
          (
            'technique_unit', ?, 'post_1',
            'Competitor CTA wording', 'Technique must stay out',
            'Resume', 'cta', 0.9, ?, 'gpt-4o-mini', 'technique'
          )
        """,
        [client_id, now, client_id, now],
    )
    conn.close()
    return db_path, db_mod, old_path


def test_generation_ab_columns_are_nullable_for_legacy_rows(tmp_path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "legacy_generation.duckdb")
    try:
        db_mod.init_db()
        conn = duckdb.connect(db_mod.DB_PATH)
        conn.execute(
            """
            INSERT INTO generated_content (
                gen_id, client_id, created_at, topic, content_type, output_text,
                model, source_units, tokens_used, cost_usd
            )
            VALUES ('legacy', ?, CURRENT_TIMESTAMP, 'Resume', 'caption',
                    'old output', 'gpt-4o-mini', [], 12, 0.000001)
            """,
            [db_mod.DEFAULT_CLIENT_ID],
        )
        row = conn.execute(
            """
            SELECT generation_run_id, generation_condition
            FROM generated_content
            WHERE gen_id = 'legacy'
            """
        ).fetchone()
        conn.close()
    finally:
        db_mod.DB_PATH = old_path

    assert row == (None, None)


def test_run_keeps_conditions_identical_except_reference_units(tmp_path, monkeypatch):
    from tools import score_generation as sg

    db_path, db_mod, old_path = _seed_generation_db(tmp_path)
    monkeypatch.setattr(sg, "_default_run_id", lambda: "generation-test")
    monkeypatch.setattr(sg, "get_secret", lambda name: "fake-key")

    try:
        with patch(
            "analysis.content_gen.requests.post",
            side_effect=[
                _openai_response("Grounded candidate", in_tokens=200, out_tokens=50),
                _openai_response("Bare candidate", in_tokens=80, out_tokens=50),
            ],
        ) as post:
            assert sg.cmd_run(
                db_path,
                client_id=db_mod.DEFAULT_CLIENT_ID,
                pairs=1,
                content_type="caption",
            ) == 0
    finally:
        db_mod.DB_PATH = old_path

    grounded_payload = post.call_args_list[0].kwargs["json"]
    bare_payload = post.call_args_list[1].kwargs["json"]
    assert grounded_payload["model"] == bare_payload["model"]
    assert grounded_payload["temperature"] == bare_payload["temperature"]
    assert grounded_payload["messages"][0] == bare_payload["messages"][0]
    assert "Allowed reference claim" in grounded_payload["messages"][1]["content"]
    assert "Technique must stay out" not in grounded_payload["messages"][1]["content"]
    assert "Competitor CTA wording" not in grounded_payload["messages"][1]["content"]
    assert "Allowed reference claim" not in bare_payload["messages"][1]["content"]
    assert "Allowed source wording" not in bare_payload["messages"][1]["content"]
    assert "Client context:" in grounded_payload["messages"][1]["content"]
    assert "Client context:" in bare_payload["messages"][1]["content"]

    conn = duckdb.connect(db_path)
    rows = conn.execute(
        """
        SELECT generation_run_id, generation_condition, topic, content_type, model,
               source_units, output_text
        FROM generated_content
        WHERE generation_run_id = 'generation-test'
        ORDER BY generation_condition
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    assert {row[0] for row in rows} == {"generation-test"}
    assert [row[1] for row in rows] == ["bare", "grounded"]
    assert {row[2] for row in rows} == {"Resume"}
    assert {row[3] for row in rows} == {"caption"}
    assert {row[4] for row in rows} == {"gpt-4o-mini"}
    assert [row[5] for row in rows] == [[], ["allowed_unit"]]


def _insert_generated_pair(
    conn,
    *,
    run_id="generation-test",
    client_id="jobs_au_demo",
    grounded_source_units=None,
    bare_source_units=None,
    grounded_text="Candidate alpha",
    bare_text="Candidate beta",
):
    grounded_source_units = ["unit_a"] if grounded_source_units is None else grounded_source_units
    bare_source_units = [] if bare_source_units is None else bare_source_units
    conn.execute(
        """
        INSERT INTO generated_content (
            gen_id, client_id, created_at, topic, content_type, output_text,
            model, source_units, tokens_used, cost_usd, generation_run_id,
            generation_condition
        )
        VALUES
          (?, ?, CURRENT_TIMESTAMP, 'Resume', 'caption',
           ?, 'gpt-4o-mini', ?, 100, 0.010000, ?, 'grounded'),
          (?, ?, CURRENT_TIMESTAMP, 'Resume', 'caption',
           ?, 'gpt-4o-mini', ?, 50, 0.002000, ?, 'bare')
        """,
        [
            f"{client_id}_gen_a",
            client_id,
            grounded_text,
            grounded_source_units,
            run_id,
            f"{client_id}_gen_b",
            client_id,
            bare_text,
            bare_source_units,
            run_id,
        ],
    )


def _condition_by_gen_id(key_text: str) -> dict[str, str]:
    rows = list(csv.DictReader(key_text.splitlines(), delimiter="\t"))
    conditions = {}
    for row in rows:
        for side in ("left", "right"):
            conditions[row[f"{side}_gen_id"]] = row[f"{side}_condition"]
    return conditions


def test_blind_file_hides_condition_and_ids_while_key_stores_mapping(tmp_path):
    from tools import score_generation as sg

    db_path, db_mod, old_path = _seed_generation_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_generated_pair(conn)
    conn.close()

    out = tmp_path / "blind.tsv"
    out_again = tmp_path / "blind_again.tsv"
    try:
        assert sg.cmd_blind(db_path, db_mod.DEFAULT_CLIENT_ID, "generation-test", str(out)) == 0
        assert sg.cmd_blind(db_path, db_mod.DEFAULT_CLIENT_ID, "generation-test", str(out_again)) == 0
    finally:
        db_mod.DB_PATH = old_path

    marked = out.read_text(encoding="utf-8")
    marked_again = out_again.read_text(encoding="utf-8")
    key = sg._key_path(str(out)).read_text(encoding="utf-8")

    assert "pair_id\tPREFERENCE\tleft\tright" in marked
    assert "grounded" not in marked.lower()
    assert "bare" not in marked.lower()
    assert "jobs_au_demo_gen_a" not in marked
    assert "jobs_au_demo_gen_b" not in marked
    assert "Candidate alpha" in marked
    assert "Candidate beta" in marked
    assert marked == marked_again
    assert _condition_by_gen_id(key) == {
        "jobs_au_demo_gen_a": "grounded",
        "jobs_au_demo_gen_b": "bare",
    }


def test_blind_uses_explicit_condition_not_source_units(tmp_path):
    from tools import score_generation as sg

    db_path, db_mod, old_path = _seed_generation_db(tmp_path)
    conn = duckdb.connect(db_path)
    _insert_generated_pair(
        conn,
        grounded_source_units=[],
        bare_source_units=["unit_that_must_not_define_condition"],
    )
    conn.close()

    out = tmp_path / "blind.tsv"
    try:
        assert sg.cmd_blind(db_path, db_mod.DEFAULT_CLIENT_ID, "generation-test", str(out)) == 0
    finally:
        db_mod.DB_PATH = old_path

    key = sg._key_path(str(out)).read_text(encoding="utf-8")

    assert _condition_by_gen_id(key) == {
        "jobs_au_demo_gen_a": "grounded",
        "jobs_au_demo_gen_b": "bare",
    }


def test_blind_scopes_generation_run_to_client(tmp_path):
    from tools import score_generation as sg

    db_path, db_mod, old_path = _seed_generation_db(tmp_path)
    other_client = "other_client"
    conn = duckdb.connect(db_path)
    _insert_generated_pair(conn, client_id=db_mod.DEFAULT_CLIENT_ID)
    _insert_generated_pair(
        conn,
        client_id=other_client,
        grounded_text="Other client alpha",
        bare_text="Other client beta",
    )
    conn.close()

    out = tmp_path / "blind.tsv"
    try:
        assert sg.cmd_blind(db_path, db_mod.DEFAULT_CLIENT_ID, "generation-test", str(out)) == 0
    finally:
        db_mod.DB_PATH = old_path

    marked = out.read_text(encoding="utf-8")
    key = sg._key_path(str(out)).read_text(encoding="utf-8")

    assert "Candidate alpha" in marked
    assert "Candidate beta" in marked
    assert "Other client alpha" not in marked
    assert "Other client beta" not in marked
    assert "other_client_gen_a" not in key
    assert "other_client_gen_b" not in key


def test_blind_rejects_run_without_one_of_each_condition(tmp_path):
    from tools import score_generation as sg

    db_path, db_mod, old_path = _seed_generation_db(tmp_path)
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        INSERT INTO generated_content (
            gen_id, client_id, created_at, topic, content_type, output_text,
            model, source_units, tokens_used, cost_usd, generation_run_id,
            generation_condition
        )
        VALUES
          ('gen_a', ?, CURRENT_TIMESTAMP, 'Resume', 'caption',
           'Candidate alpha', 'gpt-4o-mini', ['unit_a'], 100, 0.010000,
           'generation-test', 'grounded'),
          ('gen_b', ?, CURRENT_TIMESTAMP, 'Resume', 'caption',
           'Candidate beta', 'gpt-4o-mini', [], 50, 0.002000,
           'generation-test', 'grounded')
        """,
        [db_mod.DEFAULT_CLIENT_ID, db_mod.DEFAULT_CLIENT_ID],
    )
    conn.close()

    try:
        try:
            sg.cmd_blind(db_path, db_mod.DEFAULT_CLIENT_ID, "generation-test", str(tmp_path / "blind.tsv"))
        except SystemExit as exc:
            assert "not pair-complete" in str(exc)
        else:
            raise AssertionError("cmd_blind should reject an unbalanced generation condition set")
    finally:
        db_mod.DB_PATH = old_path


def test_compare_reports_small_sample_and_per_condition_cost(tmp_path, capsys):
    from tools import score_generation as sg

    marked = tmp_path / "marked.tsv"
    key = sg._key_path(str(marked))

    marked_lines = ["pair_id\tPREFERENCE\tleft\tright"]
    key_lines = [
        "pair_id\tleft_condition\tleft_gen_id\tleft_cost_usd\t"
        "right_condition\tright_gen_id\tright_cost_usd"
    ]
    for index in range(1, 11):
        pair_id = f"pair_{index:03d}"
        preference = "left" if index <= 6 else "right"
        marked_lines.append(f"{pair_id}\t{preference}\tCandidate A\tCandidate B")
        key_lines.append(
            f"{pair_id}\tgrounded\tg{index}a\t0.010000\tbare\tg{index}b\t0.002000"
        )

    marked.write_text("\n".join(marked_lines) + "\n", encoding="utf-8")
    key.write_text("\n".join(key_lines) + "\n", encoding="utf-8")

    assert sg.cmd_compare(str(marked)) == 0
    out = capsys.readouterr().out

    assert "grounded win rate 6/10" in out
    assert "sample n=10" in out
    assert "SAMPLE TOO SMALL" in out
    assert "Minimum for a directional read is 20" in out
    assert "Cost by condition" in out
    assert "grounded $0.100000 total" in out
    assert "bare     $0.020000 total" in out
    assert "5.0x grounded/bare" in out
