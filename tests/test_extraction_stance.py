"""Speaker stance in the extraction prompt (from the 2026-09-13 blind marking).

A 40-unit blind sample was marked by hand. Eight units were judged wrong —
contradicting, inventing, or misreading their source — and SEVEN of the eight
were the same failure: the extractor treating a performed sentence as an
asserted one.

That is one missing capability appearing seven times, not eight scattered
errors, so the fix belongs in the prompt rather than in a downstream filter.

WHAT THESE TESTS CAN AND CANNOT DO
    They pin the specification: the rules are present, survive per-client
    templating, and name each observed failure mode.

    They CANNOT verify the model obeys them. Only re-extracting against the
    real corpus and re-marking a sample can do that, and the corpus is on the
    owner's machine. Do not read a green suite here as evidence the bug is
    fixed — read it as evidence the instruction is in place.
"""
import pytest

from analysis.extraction import build_system_prompt
from core.db import DEFAULT_CLIENT_ID


def _flat(text: str) -> str:
    """Collapse whitespace. The prompt is wrapped prose, so an exact-match
    assertion on a phrase breaks the moment a line wrap lands inside it —
    which is a test fragility, not a prompt defect."""
    return " ".join(text.split())


@pytest.fixture
def prompt():
    return build_system_prompt(DEFAULT_CLIENT_ID)


class TestStanceRulesArePresent:
    def test_the_section_exists(self, prompt):
        assert "SPEAKER STANCE" in prompt

    def test_it_states_the_core_distinction(self, prompt):
        # The single idea the whole section exists to convey.
        assert "asserting this, or performing it" in _flat(prompt)

    @pytest.mark.parametrize("case", [
        "SATIRE AND MOCKERY",
        "NEGATIVE EXAMPLES",
        "SCRIPTED DEMONSTRATIONS",
        "DEBATE MONTAGES",
        "SINGLE ANECDOTES",
    ])
    def test_each_observed_failure_mode_is_named(self, prompt, case):
        assert case in prompt

    def test_ambiguity_lowers_confidence_rather_than_guessing(self, prompt):
        assert "lower `confidence` rather than guessing" in _flat(prompt)


class TestStanceSurvivesTemplating:
    """The prompt is built per client. A rule that only reaches one persona
    is not a rule — the per-persona taxonomy work made this a live risk."""

    def test_present_for_the_default_client(self, prompt):
        assert "SPEAKER STANCE" in prompt

    def test_present_for_an_unknown_client(self):
        # An unknown client falls back to its own catch-all taxonomy; the
        # stance rules must not be collateral damage of that fallback.
        other = build_system_prompt("no-such-client")
        assert "SPEAKER STANCE" in other
        assert "SCRIPTED DEMONSTRATIONS" in other

    def test_templating_did_not_break_the_json_braces(self, prompt):
        # _SYSTEM_PROMPT_TEMPLATE is .format()ted, so a stray brace in the new
        # prose would raise or corrupt the output contract.
        assert '{"units": [...]}' in prompt


class TestTheRecordedFailures:
    """The seven real units the rules were written from.

    Kept as data rather than prose so the evidence stays attached to the fix.
    If someone later decides a rule is unnecessary, this is what they have to
    argue with.
    """

    FAILURES = [
        ("9e0f4af5", "satire", "a mocked LinkedIn cliche read as a usage fact"),
        ("d661e835", "negative example", "a bad networking DM read as a recommendation"),
        ("2f4b6348", "scripted demo", "a demo interview answer attributed to the creator"),
        ("9eba2fae", "scripted demo", "a negotiation gambit read as 'decline the offer'"),
        ("135db81f", "satire", "a comedy sketch's debunked CV line read as fact"),
        ("32a62bf8", "debate montage", "one side of a montage read as advice"),
        ("a97ebe06", "single anecdote", "one person's story generalised to a trend"),
    ]

    def test_seven_failures_of_one_kind(self):
        assert len(self.FAILURES) == 7

    def test_every_failure_maps_to_a_named_rule(self, prompt):
        covered = {
            "satire": "SATIRE AND MOCKERY",
            "negative example": "NEGATIVE EXAMPLES",
            "scripted demo": "SCRIPTED DEMONSTRATIONS",
            "debate montage": "DEBATE MONTAGES",
            "single anecdote": "SINGLE ANECDOTES",
        }
        for _uid, kind, _desc in self.FAILURES:
            assert kind in covered, f"unmapped failure kind: {kind}"
            assert covered[kind] in prompt, f"rule missing for: {kind}"
