# Working agreements

Standing rules for agents working in this repository. They apply to every task
without being restated in a prompt. If a prompt contradicts this file, the
prompt wins — but say so, so the file can be corrected.

## Where things live

| | |
|---|---|
| **This repo** (`senpai-reel`) | all application code |
| **`ArianKalantari1/creative-director-ai`** | planning, research, issues, design docs |

Issues are filed in `creative-director-ai` and reference code here. Do not create
application code in the planning repo, and do not copy this codebase into it.

Useful context there: `docs/existing-systems/senpai-reel-inventory.md` (what
already exists) and `docs/design/` (where things are heading).

## Workflow

1. **Read the issue first.** Tasks carry their context in the GitHub issue, not
   in the prompt. The prompt will usually just name one.
2. **Open the PR as a draft** while you work.
3. **Mark it Ready for review when it is actually done** — tests green, self-reviewed.
   Review happens on that transition, so an early flip wastes a round and a late
   one stalls the work.
4. **Disagree on the issue, before building.** If a ticket's approach looks wrong,
   say so there rather than quietly doing something else. A stated disagreement is
   useful; a silent deviation is a surprise in a diff.

## Before every push

- **Run the full suite**: `python -m pytest tests/ -q --ignore=tests/test.py`
- **Check the exit code, not the output.** `pytest ... | tail -3` returns *tail's*
  status, so a failing suite can look like it passed. Use `${PIPESTATUS[0]}` or do
  not pipe.
- **Confirm client isolation still holds** if you touched anything that reads the
  database. It is the foundation everything else sits on, and a regression there is
  the one that costs a client relationship.
- **Re-read your own diff adversarially** before marking ready.

## House rules

### Absent is not zero

The most repeated bug in this codebase. It has appeared three separate times:
the acquisition adapter's field mapping, `or 0` in `upsert_post`, and
`COALESCE(SUM(...), 0)` in the cost meter.

```python
views = item.get("videoViewCount") or 0   # WRONG — missing becomes 0
views = _opt_number(item.get("videoViewCount"), int)   # None stays None
```

A missing value that becomes zero is wrong *invisibly*, because zero is plausible.
Nobody notices until a decision has been made on it.

- Keep `None` for "not supplied", including for unparseable input
- Never render a number the system cannot vouch for as a confident figure —
  show `—`, or say "unknown"
- The same applies to derived values: if an input is unknown, the derivative is
  unknown, not zero. An engagement rate of `0.0` claims nobody engaged

### Do not invent numbers you cannot verify

If a rate, price or constant cannot be checked, do not pick a plausible default.
Record it as unconfigured and say so in the UI. A guessed number is worse than a
missing one, because it looks like an answer.

`APIFY_USD_PER_RESULT` is the worked example: unset means unknown, and the cost
meter reports the total as incomplete rather than understating it.

### Everything is per-client

Every query that touches posts, transcripts, message units, generated content or
costs must be scoped to a client. `client_id` is a required leading argument on
scoped functions, deliberately — so a forgotten argument is an error rather than
silently returning the demo client's data.

Prefer `client_posts` visibility checks over direct `posts.client_id` checks.

### Taxonomy is not universal

`analysis/taxonomy.py` currently hardcodes Jobs-AU topics. That is a known
limitation, not a model to copy — see `docs/design/persona-discovery.md` in the
planning repo. Do not add more niche-specific vocabulary as module constants.

## Commits and PRs

- Explain **why**, not just what. The diff already says what.
- Note anything you found but did not fix, and where it is filed.
- Say what you verified and how. "Tests pass" is weaker than the actual numbers.
- Flag your own uncertainty. A PR that says which part is shakiest gets a better
  review than one that sounds uniformly confident.

## Costs

Roughly ⅓ of a cent per 30-second reel all-in (Deepgram + GPT-4o-mini +
embeddings). Acquisition is the only meaningful recurring cost. Keeping it cheap
is a product requirement, not an afterthought — so prefer local, deterministic
processing over API calls where the quality is comparable.
