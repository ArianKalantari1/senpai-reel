"""Tests for the originality gate (creative-director-ai issue #17)."""
from unittest.mock import patch

import pytest

from analysis.originality import ReferenceLeakError, find_overlap


class FakeUnit:
    def __init__(self, text="", claim=""):
        self.text = text
        self.claim = claim
        self.topic = "ATS"
        self.content_type = "stat"


VERBATIM = "Ninety one percent of recruiters use an ATS to filter resumes"


class TestFindOverlap:
    def test_detects_reused_span(self):
        units = [FakeUnit(text=VERBATIM, claim="Most resumes are screened first")]
        got = find_overlap(f"Did you know {VERBATIM} before anyone reads them", units)
        assert got, "verbatim reuse should be detected"
        assert any("recruiters" in g for g in got)

    def test_same_idea_reworded_is_clean(self):
        units = [FakeUnit(text=VERBATIM, claim="Most resumes are screened first")]
        assert find_overlap(
            "Software screens most applications before a person ever sees them", units
        ) == []

    def test_filler_only_overlap_is_ignored(self):
        """A span of pure filler is idiom, not copying."""
        units = [FakeUnit(text="you can do it if you are ready")]
        assert find_overlap("you can do it if you want results", units) == []

    def test_content_words_make_a_span_meaningful(self):
        """The gate deliberately errs toward over-flagging.

        "do it for your resume" is a common phrase, but `resume` is a content
        word, so it counts. A false positive costs one retry (~$0.0003); a
        false negative costs a client publishing competitor wording.
        """
        units = [FakeUnit(text="this is how you can do it for your resume")]
        assert find_overlap("this is how you can do it for your resume today", units)

    def test_checks_claim_as_well_as_text(self):
        units = [FakeUnit(text="", claim="tailor every resume to the job description")]
        assert find_overlap("You should tailor every resume to the job description", units)

    def test_no_units_means_no_overlap(self):
        assert find_overlap("anything at all goes here", []) == []

    def test_short_output_cannot_match(self):
        assert find_overlap("too short", [FakeUnit(text=VERBATIM)]) == []


class TestGateBehaviour:
    """The gate retries once, then fails closed rather than returning leaked copy."""

    def _units(self):
        return [FakeUnit(text=VERBATIM, claim="Most resumes are screened first")]

    def test_clean_output_passes_through_without_retry(self):
        from analysis import content_gen

        with patch.object(content_gen, "_call_gpt", return_value=("Original words entirely", 10, 0.1)) as m:
            text, tokens, cost = content_gen._call_gpt_original(
                "sys", "user", "key", self._units()
            )
        assert text == "Original words entirely"
        assert m.call_count == 1, "clean output must not trigger a retry"
        assert (tokens, cost) == (10, 0.1)

    def test_leaked_output_retries_and_returns_clean_second_attempt(self):
        from analysis import content_gen

        responses = [
            (f"Did you know {VERBATIM} today", 10, 0.1),
            ("Software screens applications before a person sees them", 12, 0.2),
        ]
        with patch.object(content_gen, "_call_gpt", side_effect=responses) as m:
            text, tokens, cost = content_gen._call_gpt_original(
                "sys", "user", "key", self._units()
            )
        assert m.call_count == 2
        assert VERBATIM.lower() not in text.lower()
        assert tokens == 22, "token cost of the retry must be accounted for"
        assert cost == pytest.approx(0.3)

    def test_retry_prompt_names_the_offending_phrases(self):
        from analysis import content_gen

        responses = [
            (f"Did you know {VERBATIM} today", 10, 0.1),
            ("Completely different wording here entirely", 12, 0.2),
        ]
        with patch.object(content_gen, "_call_gpt", side_effect=responses) as m:
            content_gen._call_gpt_original("sys", "user", "key", self._units())
        retry_user = m.call_args_list[1][0][1]
        assert "must not appear" in retry_user
        assert "recruiters" in retry_user

    def test_fails_closed_when_retry_also_leaks(self):
        from analysis import content_gen

        leaked = (f"Did you know {VERBATIM} today", 10, 0.1)
        with patch.object(content_gen, "_call_gpt", side_effect=[leaked, leaked]) as m:
            with pytest.raises(ReferenceLeakError) as exc:
                content_gen._call_gpt_original("sys", "user", "key", self._units())
        assert m.call_count == 2, "must not retry more than once"
        assert exc.value.phrases, "the error should carry the offending phrases"
