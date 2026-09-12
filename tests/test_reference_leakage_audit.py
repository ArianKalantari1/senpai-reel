"""Regression tests for the reference leakage audit."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from tools.audit_reference_leakage import DEFAULT_N, audit, load_rows


COMPETITOR_PHRASE = (
    "never use a colourful resume template because recruiters reject gimmicks"
)


@pytest.fixture
def leakage_db(tmp_path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "reference_leakage.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def _openai_response(content: str, in_tokens: int = 100, out_tokens: int = 50):
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
    }
    return response


def _seed_reference_unit(db_mod, unit_id: str = "unit_leakage"):
    now = datetime.utcnow()
    client_id = db_mod.DEFAULT_CLIENT_ID
    post_id = "post_leakage"

    conn = duckdb.connect(db_mod.DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO posts (
                post_id, client_id, account_id, scraped_at, hashtags, mentions
            )
            VALUES (?, ?, 'account_leakage', ?, [], [])
            """,
            [post_id, client_id, now],
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [client_id, post_id, now],
        )
        conn.execute(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model
            )
            VALUES (?, ?, ?, ?, ?, 'Resume', 'warning', 0.95, ?, 'test')
            """,
            [
                unit_id,
                client_id,
                post_id,
                COMPETITOR_PHRASE,
                "Keep resume design clean and professional.",
                now,
            ],
        )
    finally:
        conn.close()

    return SimpleNamespace(
        unit_id=unit_id,
        text=COMPETITOR_PHRASE,
        claim="Keep resume design clean and professional.",
        topic="Resume",
        content_type="warning",
    )


def _assert_audit_clean(db_path: str):
    rows = load_rows(db_path)
    findings, counted = audit(rows, DEFAULT_N)
    assert counted > 0, "test setup must include at least one cited reference"
    assert findings == [], f"source n-gram leaked into generated output: {findings}"


def test_generation_output_has_no_audited_reference_ngram(leakage_db):
    from analysis.content_gen import generate_caption

    reference = _seed_reference_unit(leakage_db)
    leaked_first_attempt = f"Did you know {COMPETITOR_PHRASE} today?"
    clean_retry = (
        "Use a simple resume layout so hiring teams can scan your experience quickly."
    )

    with patch(
        "analysis.content_gen.requests.post",
        side_effect=[
            _openai_response(leaked_first_attempt),
            _openai_response(clean_retry),
        ],
    ) as mock_post:
        result = generate_caption(
            topic="Resume",
            angle="design mistakes",
            tone="professional",
            reference_units=[reference],
            openai_api_key="fake",
            client_id=leakage_db.DEFAULT_CLIENT_ID,
        )

    assert mock_post.call_count == 2, "the first leaked generation should trigger one retry"
    assert result.output_text == clean_retry
    _assert_audit_clean(leakage_db.DB_PATH)


def test_reference_leakage_audit_positive_control_fails(leakage_db):
    reference = _seed_reference_unit(leakage_db, unit_id="unit_positive_control")
    now = datetime.utcnow()

    conn = duckdb.connect(leakage_db.DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO generated_content (
                gen_id, client_id, created_at, topic, content_type, output_text,
                model, source_units, tokens_used, cost_usd
            )
            VALUES (
                'gen_positive_control', ?, ?, 'Resume', 'caption', ?, 'test',
                ?, 10, 0.001
            )
            """,
            [
                leakage_db.DEFAULT_CLIENT_ID,
                now,
                f"Here is copied advice: {COMPETITOR_PHRASE}.",
                [reference.unit_id],
            ],
        )
    finally:
        conn.close()

    with pytest.raises(AssertionError, match="source n-gram leaked"):
        _assert_audit_clean(leakage_db.DB_PATH)

    findings, counted = audit(load_rows(leakage_db.DB_PATH), DEFAULT_N)
    assert counted == 1
    assert findings
    assert any(overlap in COMPETITOR_PHRASE for overlap in findings[0]["overlaps"])
