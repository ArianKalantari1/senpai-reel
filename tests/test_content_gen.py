"""
Tests for analysis/content_gen.py — mocked GPT + prompt rendering.
"""
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


# ── format_reference_context ─────────────────────────────────────────────────

class TestFormatReferenceContext:
    def test_empty_list_returns_string(self):
        from analysis.prompts import format_reference_context
        result = format_reference_context([])
        assert isinstance(result, str)

    def test_uses_claim_and_excludes_verbatim_text(self):
        """Competitor expression must not reach the generator (issue #17).

        `text` is captured verbatim from the competitor transcript, `claim`
        is the extractor's own abstraction. Only the abstraction may cross.
        """
        from analysis.prompts import format_reference_context

        class FakeUnit:
            text = "Always tailor your resume"
            topic = "Resume"
            content_type = "tip"
            claim = "Customise to each role"

        result = format_reference_context([FakeUnit()])
        assert "Customise to each role" in result
        assert "Always tailor your resume" not in result

    def test_falls_back_to_truncated_text_when_claim_missing(self):
        from analysis.prompts import format_reference_context

        class FakeUnit:
            text = "x" * 400
            topic = "Resume"
            content_type = "tip"
            claim = ""

        result = format_reference_context([FakeUnit()])
        assert "x" in result
        assert result.count("x") == 120, "fallback text must be truncated"

    def test_skips_units_with_no_usable_body(self):
        from analysis.prompts import format_reference_context

        class Empty:
            text = ""
            topic = "Resume"
            content_type = "tip"
            claim = ""

        assert format_reference_context([Empty()]) == "(no reference units selected)"

    def test_all_system_prompts_carry_originality_constraint(self):
        from analysis.prompts import CAPTION_SYSTEM, HOOKS_SYSTEM, SCRIPT_SYSTEM

        for prompt in (CAPTION_SYSTEM, HOOKS_SYSTEM, SCRIPT_SYSTEM):
            assert "your own words" in prompt
            assert "Do not reuse phrases" in prompt

    def test_includes_topic(self):
        from analysis.prompts import format_reference_context

        class FakeUnit:
            text = "Salary tip"
            topic = "Salary"
            content_type = "tip"
            claim = "Negotiate"

        result = format_reference_context([FakeUnit()])
        assert "Salary" in result


# ── CAPTION / HOOKS / SCRIPT prompt templates ─────────────────────────────────

class TestPromptTemplates:
    def test_caption_user_formats(self):
        from analysis.prompts import CAPTION_USER
        rendered = CAPTION_USER.format(
            topic="Resume", tone="professional", angle="common mistakes",
            reference_context="- Tailor to role"
        )
        assert "Resume" in rendered
        assert "professional" in rendered

    def test_hooks_system_formats_count(self):
        from analysis.prompts import HOOKS_SYSTEM
        rendered = HOOKS_SYSTEM.format(count=5)
        assert "5" in rendered

    def test_hooks_user_formats(self):
        from analysis.prompts import HOOKS_USER
        rendered = HOOKS_USER.format(
            topic="Interview", angle="STAR method",
            reference_context="- Practice answers", count=3
        )
        assert "Interview" in rendered
        assert "3" in rendered

    def test_script_user_formats(self):
        from analysis.prompts import SCRIPT_USER
        rendered = SCRIPT_USER.format(
            topic="LinkedIn", duration_sec=30, tone="casual",
            reference_context="- Optimise profile"
        )
        assert "LinkedIn" in rendered
        assert "30" in rendered


# ── _call_gpt ─────────────────────────────────────────────────────────────────

def _openai_response(content="Generated text", in_tokens=100, out_tokens=50):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
    }


class TestCallGpt:
    def test_returns_text_tokens_cost(self):
        from analysis.content_gen import _call_gpt

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response("Hello from GPT")

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            text, tokens, cost = _call_gpt("sys", "user", "api_key")

        assert text == "Hello from GPT"
        assert tokens == 150
        assert cost > 0

    def test_401_raises_permission_error(self):
        from analysis.content_gen import _call_gpt

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.return_value = None

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            with pytest.raises(PermissionError):
                _call_gpt("sys", "user", "bad_key")

    def test_cost_scales_with_tokens(self):
        from analysis.content_gen import _call_gpt, _MINI_INPUT_COST, _MINI_OUTPUT_COST

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response(in_tokens=1_000_000, out_tokens=1_000_000)

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            _, _, cost = _call_gpt("s", "u", "k")

        expected = 1_000_000 * _MINI_INPUT_COST + 1_000_000 * _MINI_OUTPUT_COST
        assert cost == pytest.approx(expected, rel=0.01)


# ── generate_caption ──────────────────────────────────────────────────────────

@pytest.fixture
def gen_db(tmp_path):
    import core.db as db_mod
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "gen_test.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


class TestGenerateCaption:
    def test_returns_generated_content(self, gen_db):
        from analysis.content_gen import generate_caption, GeneratedContent

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response("Great caption here #jobs")

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            result = generate_caption(
                topic="Resume",
                angle="common mistakes",
                tone="professional",
                reference_units=[],
                openai_api_key="fake",
            )

        assert isinstance(result, GeneratedContent)
        assert result.content_type == "caption"
        assert result.output_text == "Great caption here #jobs"
        assert result.topic == "Resume"
        assert result.cost_usd >= 0

    def test_saved_to_db(self, gen_db):
        import duckdb

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response("Caption saved")

        from analysis.content_gen import generate_caption
        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            result = generate_caption("LinkedIn", "profile tips", "casual", [], "k")

        conn = duckdb.connect(gen_db.DB_PATH)
        row = conn.execute(
            "SELECT output_text FROM generated_content WHERE gen_id = ?", (result.gen_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "Caption saved"


# ── generate_hooks ────────────────────────────────────────────────────────────

class TestGenerateHooks:
    def test_returns_hooks_content_type(self, gen_db):
        from analysis.content_gen import generate_hooks

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response("1. Hook one\n2. Hook two")

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            result = generate_hooks(
                topic="Interview",
                angle="before you walk in",
                reference_units=[],
                openai_api_key="fake",
                count=2,
            )

        assert result.content_type == "hooks"
        assert "Hook" in result.output_text


# ── generate_script ───────────────────────────────────────────────────────────

class TestGenerateScript:
    def test_returns_script_content_type(self, gen_db):
        from analysis.content_gen import generate_script

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _openai_response("HOOK: Did you know...\nSETUP: ...")

        with patch("analysis.content_gen.requests.post", return_value=mock_resp):
            result = generate_script(
                topic="Salary",
                tone="confident",
                duration_sec=30,
                reference_units=[],
                openai_api_key="fake",
            )

        assert result.content_type == "script"
        assert result.topic == "Salary"
