# Claude working notes

**Read `AGENTS.md` first.** Everything there applies here — where code lives, the
draft-PR workflow, the house rules (absent is not zero; do not invent numbers you
cannot verify), client scoping, and the test discipline.

This file covers what is specific to reviewing rather than building.

## Context you would otherwise re-derive

- This repo is the codebase. `ArianKalantari1/creative-director-ai` holds issues,
  research and design docs.
- `docs/existing-systems/senpai-reel-inventory.md` (planning repo) is the map of
  what already exists. Read it before proposing anything that sounds new.
- Phase 0 (client-readiness) is complete. Phase A (Creative DNA validation,
  issues #1–#7) is paused. The commercial blocker is acquisition rights — Apify's
  terms do not permit use in a paid service, so #20 is the gate to selling.

## Reviewing

**Verify, do not read.** Check the branch out, run the suite, and exercise the
behaviour. Several real findings this project has produced came from running code
that looked correct in the diff:

- an unpriced scrape rendering as `$0.0000` rather than unknown
- a lock heartbeat that only refreshed at stage boundaries, so a healthy
  20-minute scrape would have been flagged abandoned mid-run
- two tests passing only because two modules happened to share the same
  `requests` module object

**Check whether the bug is new.** Before reporting a regression, test the same
thing against `main`. A false regression report costs trust and wastes a round.
This has already happened once here — a signature change made a stale test call
look like a crash.

**Separate what was verified from what was inferred.** Say which findings were
reproduced and which are reasoning. If something could not be checked — a blocked
domain, a missing database — say so rather than writing around it.

**Credit real improvements plainly.** Several of Codex's changes have been better
than what they replaced. Saying so is part of an accurate review, not politeness.

## Environment limits

- Most external domains are blocked by the egress proxy: vendor docs, Australian
  regulator and legislation sites, `api.github.com`. `WebSearch` works.
- Branch deletion fails with HTTP 403, so merged branches accumulate. GitHub does
  not auto-retarget a stacked PR when its base merges, because that depends on the
  base branch being deleted — retarget stacked PRs manually before merging.
- `reels.duckdb` is not in the repo, so anything needing real data has to be run
  by a human.

When something cannot be verified from here, say so and keep the claim out of the
deliverable. A document that looks sourced but is not is worse than an open
question — that judgement is the origin of half the decisions recorded in the
planning repo.
