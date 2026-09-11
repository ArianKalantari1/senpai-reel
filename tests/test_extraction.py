"""
Tests for analysis/extraction.py — GPT mocking + taxonomy validation +
message unit parsing.
"""
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


# ── Taxonomy validation ───────────────────────────────────────────────────────

class TestTaxonomy:
    def test_validate_topic_exact(self):
        from analysis.taxonomy import validate_topic
        from core.db import DEFAULT_CLIENT_ID
        _, name = validate_topic(DEFAULT_CLIENT_ID, "Resume")
        assert name == "Resume"

    def test_validate_topic_case_insensitive(self):
        from analysis.taxonomy import validate_topic
        from core.db import DEFAULT_CLIENT_ID
        assert validate_topic(DEFAULT_CLIENT_ID, "resume")[1] == "Resume"
        assert validate_topic(DEFAULT_CLIENT_ID, "LINKEDIN")[1] == "LinkedIn"

    def test_validate_topic_unknown_falls_back_to_catch_all(self):
        """Unknown topics land in the client's own catch-all, not a hardcoded name."""
        from analysis.taxonomy import validate_topic
        from core.db import DEFAULT_CLIENT_ID
        from core.taxonomy import CATCH_ALL_TOPIC_ID
        topic_id, name = validate_topic(DEFAULT_CLIENT_ID, "Nonsense")
        assert topic_id == CATCH_ALL_TOPIC_ID
        assert name == "General"  # the demo client happens to call it that

    def test_validate_content_type_exact(self):
        from analysis.taxonomy import validate_content_type
        assert validate_content_type("tip") == "tip"

    def test_validate_content_type_case_insensitive(self):
        from analysis.taxonomy import validate_content_type
        assert validate_content_type("TIP") == "tip"
        assert validate_content_type("Warning") == "warning"

    def test_validate_content_type_unknown_returns_other(self):
        from analysis.taxonomy import validate_content_type
        assert validate_content_type("unknown_type") == "other"

    def test_all_topics_validate(self):
        """Topics resolve against the client's own taxonomy, not a global list."""
        from analysis.taxonomy import validate_topic
        from core.db import DEFAULT_CLIENT_ID
        from core.taxonomy import topic_names
        for name in topic_names(DEFAULT_CLIENT_ID):
            topic_id, resolved = validate_topic(DEFAULT_CLIENT_ID, name)
            assert resolved == name
            assert topic_id

    def test_all_content_types_validate(self):
        from analysis.taxonomy import CONTENT_TYPES, validate_content_type
        for ct in CONTENT_TYPES:
            assert validate_content_type(ct) == ct

    def test_topics_for_prompt_includes_all(self):
        from analysis.taxonomy import topics_for_prompt
        from core.db import DEFAULT_CLIENT_ID
        from core.taxonomy import topic_names
        prompt_str = topics_for_prompt(DEFAULT_CLIENT_ID)
        for topic in topic_names(DEFAULT_CLIENT_ID):
            assert topic in prompt_str

    def test_content_types_for_prompt_includes_all(self):
        from analysis.taxonomy import CONTENT_TYPES, content_types_for_prompt
        prompt_str = content_types_for_prompt()
        for ct in CONTENT_TYPES:
            assert ct in prompt_str


# ── extract_message_units ─────────────────────────────────────────────────────

def _make_openai_response(units: list) -> dict:
    """Build a mock OpenAI chat completions response."""
    content = json.dumps({"units": units})
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 200},
    }


def _sample_unit(
    text="Tailor your resume for each job",
    claim="ATS filters require keyword matching",
    topic="Resume",
    content_type="tip",
    confidence=0.9,
):
    return {
        "text": text,
        "claim": claim,
        "advice": "Add relevant keywords from the job description",
        "topic": topic,
        "subtopic": "keywords",
        "content_type": content_type,
        "confidence": confidence,
    }


class TestExtractMessageUnits:
    def test_empty_transcript_returns_empty(self):
        from analysis.extraction import extract_message_units
        units, cost = extract_message_units("", "post_001", "api_key")
        assert units == []
        assert cost == 0.0

    def test_short_transcript_returns_empty(self):
        from analysis.extraction import extract_message_units
        units, cost = extract_message_units("Hi.", "post_001", "api_key")
        assert units == []
        assert cost == 0.0

    def test_successful_extraction(self):
        from analysis.extraction import extract_message_units, MessageUnit

        raw_units = [_sample_unit()]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, cost = extract_message_units(
                "Always tailor your resume for each job. ATS systems filter by keywords.",
                "post_001",
                "fake_key",
            )

        assert len(units) == 1
        u = units[0]
        assert isinstance(u, MessageUnit)
        assert u.topic == "Resume"
        assert u.content_type == "tip"
        assert u.confidence == pytest.approx(0.9)
        assert u.post_id == "post_001"
        assert u.unit_id is not None

    def test_cost_is_calculated(self):
        from analysis.extraction import extract_message_units

        raw_units = [_sample_unit()]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"units": raw_units})}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        }

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            _, cost = extract_message_units(
                "A " * 50,  # long enough to pass the 30-char check
                "post_x",
                "fake_key",
            )

        # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
        assert cost == pytest.approx(0.75, rel=0.01)

    def test_invalid_topic_falls_back_to_general(self):
        from analysis.extraction import extract_message_units

        raw_units = [_sample_unit(topic="NotATopic")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, _ = extract_message_units("A " * 50, "p1", "k")

        assert units[0].topic == "General"

    def test_invalid_content_type_falls_back_to_other(self):
        from analysis.extraction import extract_message_units

        raw_units = [_sample_unit(content_type="badtype")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, _ = extract_message_units("A " * 50, "p1", "k")

        assert units[0].content_type == "other"

    def test_multiple_units_extracted(self):
        from analysis.extraction import extract_message_units

        raw_units = [_sample_unit(text=f"Tip {i}") for i in range(5)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, _ = extract_message_units("A " * 50, "p1", "k")

        assert len(units) == 5

    def test_401_raises_permission_error(self):
        from analysis.extraction import extract_message_units

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.return_value = None

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            with pytest.raises(PermissionError):
                extract_message_units("A " * 50, "p1", "bad_key")

    def test_malformed_unit_is_skipped(self):
        """A unit missing required fields should be skipped, not crash."""
        from analysis.extraction import extract_message_units

        raw_units = [
            {"text": None, "claim": None},   # will trigger str() cast but survive
            _sample_unit(text="Good unit"),
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, _ = extract_message_units("A " * 50, "p1", "k")

        # At least the valid unit survives
        assert any(u.text == "Good unit" for u in units)

    def test_text_truncated_to_300_chars(self):
        from analysis.extraction import extract_message_units

        long_text = "x" * 500
        raw_units = [_sample_unit(text=long_text)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _make_openai_response(raw_units)

        with patch("analysis.extraction.requests.post", return_value=mock_resp):
            units, _ = extract_message_units("A " * 50, "p1", "k")

        assert len(units[0].text) <= 300


# ── save_message_units ────────────────────────────────────────────────────────

class TestSaveMessageUnits:
    def test_saves_to_db(self, tmp_path):
        import duckdb
        import core.db as db_mod
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "test.duckdb")
        db_mod.init_db()

        from analysis.extraction import MessageUnit, save_message_units

        unit = MessageUnit(
            unit_id=str(uuid.uuid4()),
            post_id="post_save_test",
            text="Resume tip",
            claim="Tailor to role",
            advice="Use keywords",
            topic="Resume",
            subtopic="keywords",
            content_type="tip",
            confidence=0.85,
            extracted_at=datetime.utcnow(),
        )
        save_message_units([unit])

        conn = duckdb.connect(db_mod.DB_PATH)
        count = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE post_id = 'post_save_test'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

        db_mod.DB_PATH = old_path

    def test_empty_list_does_nothing(self):
        from analysis.extraction import save_message_units
        # Should not raise
        save_message_units([])
