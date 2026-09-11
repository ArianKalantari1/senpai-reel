"""Taxonomy: per-persona topics, shared content types.

Topics are **per-client and versioned** — see creative-director-ai #25 and
`docs/design/persona-discovery.md`. The Jobs-AU lists below are *seed data*
for the demo client, not the runtime source of truth: a second persona in
another niche gets its own topics, or everything it says classifies as
"General".

`content_type` stays shared. It describes the *shape* of an idea (tip,
warning, stat, myth...) rather than its subject, and shape is
niche-independent.
"""

from __future__ import annotations
from typing import Optional

# ── Topics ─────────────────────────────────────────────────────────────────────
SEED_TOPICS_JOBS_AU = [
    "ATS",
    "Resume",
    "CoverLetter",
    "Interview",
    "LinkedIn",
    "JobSearch",
    "Salary",
    "CareerChange",
    "Recruiter",
    "Visa",
    "General",   # fallback for content that doesn't fit a specific topic
]

# ── Content Types ──────────────────────────────────────────────────────────────
CONTENT_TYPES = [
    "tip",         # Actionable advice
    "warning",     # Common mistake to avoid
    "stat",        # Statistic or data point
    "myth",        # Debunking a misconception
    "story",       # Personal experience / case study
    "hook",        # Opening hook (first 3 seconds)
    "cta",         # Call to action
    "other",       # Catch-all fallback
]

# ── Descriptions used in GPT system prompt ────────────────────────────────────
SEED_TOPIC_DESCRIPTIONS_JOBS_AU = {
    "ATS": "Applicant Tracking Systems — keywords, resume parsing, rejection",
    "Resume": "Resume writing, formatting, sections, keywords, length",
    "CoverLetter": "Cover letter strategy, structure, personalisation",
    "Interview": "Interview prep, common questions, STAR method, follow-up",
    "LinkedIn": "LinkedIn profile optimisation, networking, InMail, content",
    "JobSearch": "Job search strategy, job boards, cold outreach, referrals",
    "Salary": "Salary negotiation, benchmarks, market rates",
    "CareerChange": "Pivoting industries, upskilling, transferable skills",
    "Recruiter": "Working with recruiters, agency vs in-house, recruiter tips",
    "Visa": "Working visa, sponsorship Australia, 482, 189, 190 visas",
    "General": "Career advice that doesn't fit a specific topic",
}

CONTENT_TYPE_DESCRIPTIONS = {
    "tip": "Actionable advice the viewer can apply immediately",
    "warning": "A common mistake or thing to avoid",
    "stat": "A specific statistic or data point cited",
    "myth": "Debunking a widely-held misconception",
    "story": "A personal experience, case study, or anecdote",
    "hook": "Opening hook or attention-grabbing statement (first 3 seconds)",
    "cta": "Call to action — follow, like, comment, download",
    "other": "Content that doesn't fit the above types",
}


def validate_content_type(value: str) -> str:
    """Normalise and validate a content_type string. Returns 'other' if unrecognised."""
    for ct in CONTENT_TYPES:
        if ct.lower() == value.lower().strip():
            return ct
    return "other"


def content_types_for_prompt() -> str:
    """Return a formatted string of content types for GPT system prompts."""
    return "\n".join(f"  - {ct}: {CONTENT_TYPE_DESCRIPTIONS[ct]}" for ct in CONTENT_TYPES)


# ── Per-client topics ─────────────────────────────────────────────────────────
# These delegate to core.taxonomy. Imported lazily to avoid a circular import
# (core.db -> core.clients -> analysis.* in some call paths).

def topics_for_prompt(client_id: str) -> str:
    """Formatted topic list for this client's active taxonomy."""
    from core.taxonomy import topics_for_prompt as _impl
    return _impl(client_id)


def validate_topic(client_id: str, value: str):
    """Resolve a model-produced topic name against this client's active taxonomy.

    Returns (topic_id, name). Falls back to the client's own catch-all rather
    than a hardcoded "General", because a persona may not have one by that name.
    """
    from core.taxonomy import validate_topic as _impl
    return _impl(client_id, value)
