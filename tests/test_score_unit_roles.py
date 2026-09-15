from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def role_db(tmp_path, monkeypatch):
    import core.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "roles.duckdb"))
    db_mod.init_db()
    return db_mod.DB_PATH


def _seed_unit(
    db_path: str,
    unit_id: str,
    client_id: str = "client_a",
    text: str = "Tailor the resume to the job description",
    claim: str | None = None,
    content_type: str = "tip",
    topic: str = "Hiring",
    unit_role: str | None = None,
    unit_role_source: str | None = None,
):
    post_id = f"post_{unit_id}"
    conn = duckdb.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            [client_id, post_id],
        )
        conn.execute(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, content_type, topic,
                unit_role, unit_role_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                unit_id,
                client_id,
                post_id,
                text,
                claim if claim is not None else text,
                content_type,
                topic,
                unit_role,
                unit_role_source,
            ],
        )
    finally:
        conn.close()


def _data_lines(path: Path) -> list[list[str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return [line.split("\t") for line in lines]


def test_blind_withholds_model_role_and_writes_sidecar_key(role_db, tmp_path, monkeypatch):
    from tools import score_unit_roles

    _seed_unit(role_db, "u1")
    _seed_unit(role_db, "u2", text="Interview feedback should be requested quickly")

    def fake_call(unit, api_key):
        role = "subject" if unit.unit_id == "u1" else "technique"
        return score_unit_roles.ModelVerdict(
            role=role,
            confidence=0.81,
            cost_usd=0.00001,
            raw=f'{{"role":"{role}","confidence":0.81}}',
        )

    monkeypatch.setattr(score_unit_roles, "_call_model", fake_call)
    out = tmp_path / "roles.tsv"

    score_unit_roles.cmd_blind(role_db, "client_a", 2, str(out), seed="same", api_key="fake")

    marked = _data_lines(out)
    key = _data_lines(Path(score_unit_roles._key_path(str(out))))
    assert marked[0] == ["unit_id", "HUMAN_ROLE", "content_type", "topic", "claim", "source"]
    assert key[0] == ["unit_id", "model_role", "model_confidence", "model_cost_usd", "model_raw"]
    assert {row[1] for row in marked[1:]} == {"?"}
    assert {row[1] for row in key[1:]} == {"subject", "technique"}
    assert "model_role" not in marked[0]
    assert "model_confidence" not in marked[0]


def test_blind_samples_only_rule_deferred_units(role_db, tmp_path, monkeypatch):
    from tools import score_unit_roles

    _seed_unit(role_db, "rule_cta", content_type="cta")
    _seed_unit(role_db, "rule_meta", text="link in bio for the guide")
    _seed_unit(role_db, "deferred", content_type="tip")
    calls = []

    def fake_call(unit, api_key):
        calls.append(unit.unit_id)
        return score_unit_roles.ModelVerdict("subject", 0.9, None, '{"role":"subject"}')

    monkeypatch.setattr(score_unit_roles, "_call_model", fake_call)
    out = tmp_path / "roles.tsv"

    score_unit_roles.cmd_blind(role_db, "client_a", 10, str(out), api_key="fake")

    text = out.read_text(encoding="utf-8")
    assert calls == ["deferred"]
    assert "deferred" in text
    assert "rule_cta" not in text
    assert "rule_meta" not in text


def test_blind_is_seeded_and_reproducible(role_db, tmp_path, monkeypatch):
    from tools import score_unit_roles

    for i in range(8):
        _seed_unit(role_db, f"u{i}", text=f"Resume idea {i}")

    def fake_call(unit, api_key):
        return score_unit_roles.ModelVerdict("subject", 0.7, None, '{"role":"subject"}')

    monkeypatch.setattr(score_unit_roles, "_call_model", fake_call)
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"

    score_unit_roles.cmd_blind(role_db, "client_a", 4, str(first), seed="fixed", api_key="fake")
    score_unit_roles.cmd_blind(role_db, "client_a", 4, str(second), seed="fixed", api_key="fake")

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    first_key = Path(score_unit_roles._key_path(str(first))).read_text(encoding="utf-8")
    second_key = Path(score_unit_roles._key_path(str(second))).read_text(encoding="utf-8")
    assert first_key == second_key


def test_blind_scopes_to_client(role_db, tmp_path, monkeypatch):
    from tools import score_unit_roles

    _seed_unit(role_db, "client_a_unit", client_id="client_a")
    _seed_unit(role_db, "client_b_unit", client_id="client_b")
    calls = []

    def fake_call(unit, api_key):
        calls.append((unit.client_id, unit.unit_id))
        return score_unit_roles.ModelVerdict("subject", 0.7, None, '{"role":"subject"}')

    monkeypatch.setattr(score_unit_roles, "_call_model", fake_call)
    out = tmp_path / "roles.tsv"

    score_unit_roles.cmd_blind(role_db, "client_a", 10, str(out), api_key="fake")

    assert calls == [("client_a", "client_a_unit")]
    assert "client_a_unit" in out.read_text(encoding="utf-8")
    assert "client_b_unit" not in out.read_text(encoding="utf-8")


def test_compare_reports_overall_and_per_role(tmp_path, capsys):
    from tools import score_unit_roles

    marked = tmp_path / "roles.tsv"
    marked.write_text(
        "\n".join(
            [
                "unit_id\tHUMAN_ROLE\tcontent_type\ttopic\tclaim\tsource",
                "u1\tsubject\ttip\tHiring\tc\ts",
                "u2\ttechnique\ttip\tHiring\tc\ts",
                "u3\tmeta\ttip\tHiring\tc\ts",
                "u4\tofftopic\ttip\tHiring\tc\ts",
                "u5\t?\ttip\tHiring\tc\ts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    Path(score_unit_roles._key_path(str(marked))).write_text(
        "\n".join(
            [
                "unit_id\tmodel_role\tmodel_confidence\tmodel_cost_usd\tmodel_raw",
                "u1\tsubject\t0.9\t0.000001\t{}",
                "u2\tsubject\t0.9\t0.000001\t{}",
                "u3\tmeta\t0.9\t0.000001\t{}",
                "u4\tofftopic\t0.9\t0.000001\t{}",
                "u5\tsubject\t0.9\t0.000001\t{}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    score_unit_roles.cmd_compare(str(marked))
    out = capsys.readouterr().out

    assert "Unit role agreement with your reading: 3/4  (75%)" in out
    assert "subject    1/1  (100%)" in out
    assert "technique  0/1  (0%)" in out
    assert "meta       1/1  (100%)" in out
    assert "offtopic   1/1  (100%)" in out
    assert "Skipped unmarked rows: 1" in out


def test_compare_rejects_invalid_human_role(tmp_path):
    from tools import score_unit_roles

    marked = tmp_path / "roles.tsv"
    marked.write_text(
        "unit_id\tHUMAN_ROLE\tcontent_type\ttopic\tclaim\tsource\n"
        "u1\tidea\ttip\tHiring\tc\ts\n",
        encoding="utf-8",
    )
    Path(score_unit_roles._key_path(str(marked))).write_text(
        "unit_id\tmodel_role\tmodel_confidence\tmodel_cost_usd\tmodel_raw\n"
        "u1\tsubject\t0.9\t0.000001\t{}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="HUMAN_ROLE 'idea' is not valid"):
        score_unit_roles.cmd_compare(str(marked))


def test_call_model_parses_json_without_live_api(monkeypatch):
    from tools import score_unit_roles

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"role":"TECHNIQUE","confidence":0.82}'}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }

    sent = {}

    def fake_post(url, headers, json, timeout):
        sent["url"] = url
        sent["headers"] = headers
        sent["json"] = json
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(score_unit_roles.requests, "post", fake_post)
    verdict = score_unit_roles._call_model(
        score_unit_roles.UnitCandidate(
            unit_id="u1",
            client_id="client_a",
            text="Open with a sharp contrast",
            claim="Open with contrast",
            content_type="tip",
            topic="Hooks",
        ),
        api_key="fake-key",
    )

    assert verdict.role == "technique"
    assert verdict.confidence == 0.82
    assert verdict.cost_usd is not None
    assert sent["url"] == score_unit_roles.OPENAI_CHAT_COMPLETIONS_URL
    assert sent["headers"]["Authorization"] == "Bearer fake-key"
    assert sent["json"]["model"] == score_unit_roles.MODEL
