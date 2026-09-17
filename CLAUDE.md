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

## Ari's other project: Remmie

Remmie is Ari's separate product — a pre-consultation patient-history tool for
Australian GP practices. Two repos: `remmie-webdev` (landing site) and the
product backend. Not part of this project, but its architecture is a candidate
to borrow from, and Ari has said so explicitly.

**The mechanism, in one line:** the model proposes facts with quotes;
deterministic code re-verifies every quote against its cited source, blocks
anything it cannot confirm, and binds each surviving fact to a span in the
original document. Grounding is enforced outside the model, not self-reported
by it.

The parts worth knowing when this repo's extraction comes up:

- `GroundedField` — no bare facts. Every value travels with `Evidence`
  (`doc_id`, `page`, verbatim `quote`).
- `verify_quote()` — pure Python re-check that the claimed quote is really in
  the cited document. Exact match, then fuzzy alignment for OCR noise.
- `_critical_tokens_ok` — the part that matters most here. Fuzziness must not
  wash out numbers or negation: "3.2 cm" must not match "5.2 cm", and
  "no tear" must not match "tear".
- Abstention gate — unverified values are blanked and rendered as "Not stated
  in the documents supplied". This is *absent is not zero*, applied per fact.
- Provenance labels over confidence scores — the doctor sees QUOTED /
  GP_SUPPLIED / DRAFTED / ABSENT / GP_EDITED, never an ML number.

**Where it maps onto senpai-reel.** The schema already has the plumbing and
none of it is wired: `transcripts.transcript` holds full text,
`transcript_words` holds word-level timing, and `message_units.source_start` /
`source_end` are DOUBLE — almost certainly seconds into the video — but
nothing writes or reads them. Meanwhile `message_units.confidence` is a model
self-report ("Your extraction confidence 0.0-1.0"), which is exactly what
Remmie refuses to trust, and the extraction prompt permits "verbatim or
closely paraphrased", so no claim is guaranteed quotable.

The extraction prompt already warns that creators quote bad advice to argue
against it, so polarity inversion is a known failure mode here — currently
defended with a prompt instruction rather than a check. That is the gap
`_critical_tokens_ok` fills.

**What does not transfer:** the contradiction/red-flag sweep and the HITL gate
(domain-specific; Ari is n=1 and is himself the human in the loop), and the
no-opinion policy — marketing copy should extrapolate, so that constraint is
wrong here.

**Do the measurement before porting anything.** What fraction of the ~5,361
existing claims actually verify against their transcript is unknown. If most
verify, none of this is worth building.
