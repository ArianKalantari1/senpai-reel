"""
Tests for analysis/content_gen.py — Phase 10: Groq-backed generation.

_call_gpt now delegates to providers.factory.get_generation_llm().generate().
All tests mock at the provider factory level so no real API keys are required.
"""
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


def _mock_generation_llm(text="Generated text", tokens=150, cost=0.0001):
    """Return a mock get_generation_llm() result."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (text, tokens, cost)
    return mock_llm

class TestFormatReferenceContext:
    def test_empty_list_returns_string(self):
        from analysis.prompts import format_reference_context
        result = format_reference_context([])
        assert isinstance(result, str)

    def test_includes_unit_text(self):
        from analysis.prompts import format_reference_context

        class FakeUnit:
            text = "Always tailor your resume"
            topic = "Resume"
            content_type = "tip"
            claim = "Customise to each role"

        result = format_reference_context([FakeUnit()])
        assert "Always tailor your resume" in result

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

class TestCallGpt:
    def test_returns_text_tokens_cost(self):
        from analysis.content_gen import _call_gpt

        with patch("providers.factory.get_generation_llm", return_value=_mock_generation_llm("Hello from Groq", 150, 0.0001)):
            text, tokens, cost = _call_gpt("sys", "user")

        assert text == "Hello from Groq"
        assert tokens == 150
        assert cost > 0

    def test_provider_error_propagates(self):
        from analysis.content_gen import _call_gpt

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("rate limit")

        with patch("providers.factory.get_generation_llm", return_value=mock_llm):
            with pytest.raises(RuntimeError):
                _call_gpt("sys", "user")

    def test_different_max_tokens_passed_through(self):
        from analysis.content_gen import _call_gpt

        mock_llm = _mock_generation_llm("OK", 50, 0.0)

        with patch("providers.factory.get_generation_llm", return_value=mock_llm):
            _call_gpt("s", "u", max_tokens=800)

        mock_llm.generate.assert_called_once_with("s", "u", max_tokens=800)


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

        with patch("analysis.content_gen._call_gpt", return_value=("Great caption here #jobs", 150, 0.0001)):
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

        from analysis.content_gen import generate_caption
        with patch("analysis.content_gen._call_gpt", return_value=("Caption saved", 100, 0.0)):
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

        with patch("analysis.content_gen._call_gpt", return_value=("1. Hook one\n2. Hook two", 80, 0.0)):
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

        with patch("analysis.content_gen._call_gpt", return_value=("HOOK: Did you know...\nSETUP: ...", 200, 0.0)):
            result = generate_script(
                topic="Salary",
                tone="confident",
                duration_sec=30,
                reference_units=[],
                openai_api_key="fake",
            )

        assert result.content_type == "script"
        assert result.topic == "Salary"
