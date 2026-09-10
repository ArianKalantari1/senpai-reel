"""
Curated list of Jobs-in-Australia competitor Instagram accounts to track.
Kept as seed data for the demo client. Runtime account lists now live in DuckDB
(`client_accounts`) and are looked up by client_id.

Categories:
  - Resume & Career Coaches
  - Recruitment Agencies
  - HR & ATS Tips
  - Interview Prep
  - LinkedIn / Job Search Strategy
"""

# fmt: off
JOBS_AU_SEED_ACCOUNTS = [
    # ── Resume & Career Coaches ──────────────────────────────────────────────
    "resumeworded",          # AI-powered resume + LinkedIn feedback
    "careersidekick",        # Career advice, job search tips
    "the.career.strategist", # Australian career coach
    "iamhannah.co",          # Resume + job search for AU/NZ
    "careerwithsam",         # Interview + resume AU-focused
    "jobsearchcoach",        # General job search advice
    "theresumewriter",       # Professional resume tips
    "careercoachmelbourne",  # Melbourne-based career coach

    # ── Recruitment & HR ─────────────────────────────────────────────────────
    "hays.australia",        # Hays Recruitment AU
    "robertwaltersau",       # Robert Walters Australia
    "michaelpageaustralia",  # Michael Page AU
    "reedrecruitment",       # Reed Recruitment
    "seek.com.au",           # SEEK Australia (main job board)
    "hrmonline",             # HR news and insights AU

    # ── ATS & Resume Tips ────────────────────────────────────────────────────
    "tealau",                # Teal – resume builder / ATS
    "kickresume",            # Resume / cover letter builder
    "resumetricks",          # ATS resume hacks

    # ── Interview Prep ───────────────────────────────────────────────────────
    "interviewguru",         # Interview tips
    "theinterviewcoach",     # STAR method + behavioural tips
    "lindseypollak",         # Workplace + career advice

    # ── LinkedIn & Job Strategy ──────────────────────────────────────────────
    "linkedinau",            # LinkedIn Australia official
    "joshuafluke",           # LinkedIn personal branding
    "austinbelcak",          # Job search strategy, networking
]
# fmt: on

# Backwards compatibility for older scripts/tests that import the original name.
COMPETITOR_ACCOUNTS = JOBS_AU_SEED_ACCOUNTS

# Default max reels to scrape per account in batch mode
DEFAULT_MAX_ITEMS = 30

# Higher limits for key competitor deep-dives
PRIORITY_ACCOUNTS = {
    "resumeworded": 50,
    "seek.com.au": 50,
    "hays.australia": 50,
}


def get_accounts_with_limits(client_id: str | None = None) -> list[tuple[str, int]]:
    """Return (username, max_items) tuples for a client-backed scrape batch."""
    if client_id:
        from core.clients import get_accounts_with_limits as _get_client_accounts_with_limits

        rows = _get_client_accounts_with_limits(client_id)
        if rows:
            return rows

    return [
        (username, PRIORITY_ACCOUNTS.get(username, DEFAULT_MAX_ITEMS))
        for username in JOBS_AU_SEED_ACCOUNTS
    ]
