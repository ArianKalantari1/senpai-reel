"""Unit role: what *kind* of thing a message unit is (creative-director-ai #27).

`content_type` conflates two questions. `tip`/`warning`/`stat`/`myth`/`story`
describe a claim about the world; `hook`/`cta` describe how a message is
delivered. One column, two axes, so nothing downstream can tell a competitor's
call-to-action from an idea worth reusing.

`unit_role` is the second axis. It sits *beside* `content_type` — `hook` and
`cta` stay exactly where they are — and answers a different question:

    content_type  ->  what shape is this idea?
    unit_role     ->  is this an idea at all, or a delivery technique?

Note what this is NOT for: filtering CTAs out as noise. A CTA is valuable, as
technique. The bug was filing it as subject matter.
"""

from __future__ import annotations

import re
from typing import Optional

# ── The axis ──────────────────────────────────────────────────────────────────
UNIT_ROLES = ["subject", "technique", "meta", "offtopic"]

UNIT_ROLE_DESCRIPTIONS = {
    "subject":   "Asserts something about the world. Eligible to feed generation.",
    "technique": "Describes HOW a message is delivered — hook, CTA, structure, "
                 "framing device, pacing. The pattern is reusable; the wording never is.",
    "meta":      "About the creator or channel itself — sponsor read, follow-for-more, "
                 "link in bio, course or product plug.",
    "offtopic":  "Unrelated to the client's niche.",
}

# How a role was decided. A human correction must never be silently overwritten
# by a later backfill run, so the source is stored alongside the value.
UNIT_ROLE_SOURCES = ["rule", "model", "human"]

# ── Deterministic rules ───────────────────────────────────────────────────────
# content_type values that already answer the question. `hook` and `cta` are
# descriptions of delivery, so they are technique by definition — no model call,
# no cost, and testable without an API key.
_TECHNIQUE_CONTENT_TYPES = {"hook", "cta"}

# Phrases that mark a unit as being about the channel rather than the subject.
# Deliberately narrow: a marker fires only when it is unambiguous, because a
# false `meta` silently removes a real idea from the generation pool. Anything
# doubtful is left for the model rather than guessed at here.
_META_PATTERNS = [
    r"\blink in (?:my )?bio\b",
    r"\bfollow (?:me|us|for more|along)\b",
    r"\bsubscribe\b",
    r"\bhit the (?:follow|like|bell)\b",
    r"\bsmash that (?:like|follow)\b",
    r"\bcomment \w+ (?:below )?and (?:i|we)(?:'ll| will)\b",
    r"\bdm me\b",
    r"\bcheck out my (?:course|program|masterclass|ebook|guide|link)\b",
    # A bare "my course" is not enough: "my course of action" is an ordinary
    # sentence, and a false `meta` silently deletes a real idea from the
    # generation pool. Require a verb that makes it a plug.
    r"\b(?:join|enrol|enroll|buy|grab|inside|link to) (?:my |the )?"
    r"(?:course|program|masterclass|bootcamp)\b",
    r"\b(?:this video is )?sponsored by\b",
    r"\bthanks to .{1,40} for sponsoring\b",
    r"\buse (?:my )?code \w+\b",
    r"\bsign up (?:at|via|using) (?:my|the) link\b",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


def role_from_rules(text: str, claim: str = "", content_type: str = "") -> Optional[str]:
    """Decide a unit's role from deterministic signals, or None to defer.

    Pure: text in, role or None out. No network, no database, no model. Returning
    None is a real answer — it means "the rules cannot tell", and guessing here
    would be worse than paying for a model call.
    """
    if content_type and content_type.strip().lower() in _TECHNIQUE_CONTENT_TYPES:
        return "technique"

    haystack = f"{text or ''}\n{claim or ''}"
    if _META_RE.search(haystack):
        return "meta"

    return None


def validate_unit_role(value: str) -> Optional[str]:
    """Normalise a model-produced role. Unrecognised reads as unknown, not a guess.

    Deliberately returns None rather than falling back to "subject": a wrong
    `subject` puts unvetted material into the generation pool, which is the
    failure this whole axis exists to prevent.
    """
    if not value:
        return None
    candidate = value.strip().lower()
    return candidate if candidate in UNIT_ROLES else None


def unit_roles_for_prompt() -> str:
    """Formatted role list for model prompts."""
    return "\n".join(f"  - {r}: {UNIT_ROLE_DESCRIPTIONS[r]}" for r in UNIT_ROLES)
