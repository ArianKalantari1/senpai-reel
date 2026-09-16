#!/usr/bin/env python3
"""Where is the industry loud, and where is it only repeating itself?

The corpus measures SUPPLY — what competitors publish. On its own that cannot
find a gap, because "nobody talks about X" and "nobody cares about X" look
identical from the supply side. This tool joins supply to the one demand
signal already in the database: the engagement of the posts a topic appears in.

    topic_gaps.py <db> --client CLIENT [--by topic|subtopic]
                                       [--dedupe-threshold 0.70]

A gap is the combination the columns are arranged to show: engagement is high,
distinct accounts are few, and the ideas behind those accounts are fewer still.

Two counting rules, both learned the hard way on this corpus:

  - Ideas, not units. 22.7% of units are near-duplicates of another unit at
    0.70 overlap, and 1,617 duplicate pairs sit inside a single post. One
    90-second video sliced nineteen ways is one idea, not nineteen, and
    counting units lets an extraction artefact masquerade as market attention.

  - Accounts, not posts. One creator posting six times about resumes is one
    voice. Saturation is how many DIFFERENT people are covering something.

Engagement is never averaged over unknowns. `posts.engagement_rate` is NULL
whenever likes or views were missing (see engagement_rate_of in core/db.py),
and a topic whose posts all lack it reports `no data`, not 0.0%. The column
declares DEFAULT 0, so this is worth stating: a zero here would read as
"nobody engaged" with a topic that may simply never have been measured.

Mechanical throughout — no API key, no spend. It cannot tell you whether a
topic is worth writing about. It can tell you where to look.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_claim_helpers():
    """Borrow the calibrated stemming from score_extraction.py.

    tools/ is not a package, so this loads by path rather than importing.
    Deliberately reusing that module's stemmer instead of writing a second
    one: two similarity measures that drift apart would make `duplicates`
    and this tool disagree about what counts as the same idea, and the
    whole point here is that they agree.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "score_extraction.py"
    spec = importlib.util.spec_from_file_location("_score_extraction_for_gaps", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.claim_stems


claim_stems = _load_claim_helpers()

DEFAULT_DEDUPE_THRESHOLD = 0.70
UNKNOWN = "no data"


# Engagement is likes as a percentage of views. Negative is impossible and
# above 100% means likes exceeded counted views. The real corpus contains
# both: values down to -0.64% and a post at 135.71%. They are not measurements
# and must not sort to the top or bottom of a ranking as though they were.
MAX_PLAUSIBLE_ENGAGEMENT = 100.0


def implausible(value: float | None) -> bool:
    return value is not None and (value < 0.0 or value > MAX_PLAUSIBLE_ENGAGEMENT)


def _connect(db_path: str):
    import duckdb

    return duckdb.connect(db_path, read_only=True)


def _require_schema(conn, db_path: str) -> None:
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    missing = sorted({"message_units", "posts", "client_posts"} - tables)
    if missing:
        raise SystemExit(
            f"\n  topic_gaps needs {', '.join(missing)} for client-scoped reporting.\n"
        )


def load_rows(db_path: str, client_id: str, level: str = "topic") -> list[tuple]:
    """(topic, account, post_id, unit_id, claim, engagement_rate, embedding)."""
    conn = _connect(db_path)
    try:
        _require_schema(conn, db_path)
        has_accounts = "creator_accounts" in {
            r[0] for r in conn.execute("SHOW TABLES").fetchall()
        }
        account = "COALESCE(ca.username, p.account_id, 'unknown')" if has_accounts else \
                  "COALESCE(p.account_id, 'unknown')"
        join = "LEFT JOIN creator_accounts ca ON ca.account_id = p.account_id" if has_accounts else ""
        if level not in ("topic", "subtopic"):
            raise SystemExit(f"--by must be topic or subtopic, not {level!r}")
        return conn.execute(
            f"""
            SELECT
                COALESCE(NULLIF(TRIM(mu.{level}), ''), '(unclassified)') AS topic,
                {account} AS account,
                p.post_id,
                mu.unit_id,
                mu.claim,
                p.engagement_rate,
                mu.embedding
            FROM posts p
            JOIN client_posts cp ON cp.post_id = p.post_id AND cp.client_id = ?
            JOIN message_units mu ON mu.post_id = p.post_id AND mu.client_id = ?
            {join}
            ORDER BY topic, account, p.post_id, mu.unit_id
            """,
            [client_id, client_id],
        ).fetchall()
    finally:
        conn.close()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def distinct_ideas(claims: list[str], threshold: float,
                   embeddings: list | None = None) -> tuple[int, str]:
    """Roughly how many separate ideas these claims represent, and by what measure.

    Greedy single-pass clustering either way: a claim joins the first kept
    claim it is close enough to, otherwise it starts a new cluster. Not
    optimal clustering and it does not need to be — the question is "roughly
    how many separate ideas".

    Embeddings are used when every claim has one, because stem overlap cannot
    answer this question on a single-domain corpus. Measured on three real
    units extracted from one 10-second CV video, which a human reads as one
    idea stated three ways:

        "Colorful templates and selfies are not suitable for a professional CV"
        "A black and white template is more effective for a CV than a colorful one"
        "A standardized black and white template is easier to scan and professional"

    Their pairwise stem overlaps are 0.43, 0.29 and 0.43. The threshold that
    collapses them (0.40) also flags 81.5% of the corpus as duplicated; the
    threshold that leaves the corpus intact (0.70) leaves all three standing.
    There is no cut-off that separates "same idea" from "same topic" here,
    because these are the same idea in genuinely different words — which is a
    semantic question, and stem overlap is a lexical instrument.

    The embeddings are already computed and paid for (message_units.embedding),
    so when they are present this uses them. The returned measure name says
    which was used, because a number produced by the fallback means less.
    """
    usable = (embeddings is not None
              and len(embeddings) == len(claims)
              and all(e is not None and len(e) for e in embeddings))
    if usable:
        kept_vectors: list = []
        for vector in embeddings:
            for seen in kept_vectors:
                if _cosine(vector, seen) >= threshold:
                    break
            else:
                kept_vectors.append(vector)
        return len(kept_vectors), "embedding"

    kept: list[set[str]] = []
    for claim in claims:
        stems = claim_stems(claim or "")
        if not stems:
            continue
        for seen in kept:
            denominator = min(len(stems), len(seen))
            if denominator and len(stems & seen) / denominator >= threshold:
                break
        else:
            kept.append(stems)
    return len(kept), "stem overlap"


def topic_rows(rows: list[tuple], threshold: float) -> list[dict]:
    grouped: dict[str, dict] = {}
    for topic, account, post_id, _unit_id, claim, engagement, embedding in rows:
        g = grouped.setdefault(topic, {
            "topic": topic, "accounts": set(), "posts": set(),
            "units": 0, "claims": [], "embeddings": [], "engagements": {},
            "implausible": set(),
        })
        g["accounts"].add(account)
        g["posts"].add(post_id)
        g["units"] += 1
        if claim and claim.strip():
            g["claims"].append(claim)
            g["embeddings"].append(list(embedding) if embedding is not None else None)
        # Per post, not per unit: a post with nine units would otherwise weigh
        # nine times in the engagement figure for its topic.
        if engagement is not None:
            value = float(engagement)
            if implausible(value):
                g["implausible"].add(post_id)
            else:
                g["engagements"][post_id] = value

    out = []
    for g in grouped.values():
        values = list(g["engagements"].values())
        ideas, measure = distinct_ideas(g["claims"], threshold, g["embeddings"])
        out.append({
            "topic": g["topic"],
            "accounts": len(g["accounts"]),
            "posts": len(g["posts"]),
            "units": g["units"],
            "ideas": ideas,
            "measure": measure,
            # None, never 0.0 — a topic nobody measured is not a topic nobody
            # engaged with.
            "median_engagement": statistics.median(values) if values else None,
            "engagement_known": len(values),
            "engagement_implausible": len(g["implausible"]),
        })
    # Unknown engagement sorts last rather than sorting as zero.
    out.sort(key=lambda r: (r["median_engagement"] is None,
                            -(r["median_engagement"] or 0.0), r["topic"]))
    return out


def _pct(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:.2f}%"


def cmd_report(db_path: str, client_id: str, threshold: float,
               level: str = "topic", min_posts: int = 1) -> int:
    rows = load_rows(db_path, client_id, level)
    if not rows:
        raise SystemExit(f"\n  No client-visible units found for {client_id!r}.\n")

    topics = topic_rows(rows, threshold)
    # One post's engagement is an anecdote. Ranking by it puts whatever single
    # video went viral at the top of a report about market structure: the real
    # corpus produced five different subtopics all reading 135.71%, which is
    # one post counted five times.
    hidden = [r for r in topics if r["posts"] < min_posts]
    topics = [r for r in topics if r["posts"] >= min_posts]
    if not topics:
        raise SystemExit(
            f"\n  No {level}(s) reach --min-posts {min_posts}. "
            f"The most any has is {max((r['posts'] for r in hidden), default=0)}.\n")
    print(f"\n{level.capitalize()} supply vs engagement — client {client_id}")
    measures = {r["measure"] for r in topics}
    print(f"  Ideas collapse near-duplicate claims at >= {threshold:.2f}, "
          f"by {' and '.join(sorted(measures))}.")
    if "stem overlap" in measures:
        print("  WARNING: stem overlap is a lexical measure and under-counts")
        print("  duplication — three real units from one CV video, which a human")
        print("  reads as one idea, do not collapse at any usable threshold.")
        print("  Embed the units (analysis/embeddings.py) for a real figure.")
    print("  Engagement is the median across posts with a known rate; topics")
    print("  where none is known read 'no data' rather than 0.\n")
    print(f"  {level[:18]:<18}{'accts':>6}{'posts':>7}{'units':>7}{'ideas':>7}"
          f"{'units/idea':>12}{'median engagement':>20}{'known':>8}")
    for r in topics:
        # A unit whose claim is blank counts in `units` but yields no idea, so
        # ideas can be 0 while units is not. Dividing there would either crash
        # or invent a ratio for a row that has no usable claim at all.
        per_idea = f"{r['units'] / r['ideas']:.1f}" if r["ideas"] else UNKNOWN
        known = f"{r['engagement_known']}/{r['posts']}"
        print(f"  {r['topic'][:18]:<18}{r['accounts']:>6}{r['posts']:>7}{r['units']:>7}"
              f"{r['ideas']:>7}{per_idea:>12}{_pct(r['median_engagement']):>20}{known:>8}")

    dropped = sum(r["engagement_implausible"] for r in topics)
    if dropped:
        print(f"  {dropped} post(s) excluded for an impossible engagement rate "
              "(negative, or above 100%).")
        print("  Those are data errors, not low performers, and are not counted"
              " either way.")
    if hidden:
        print(f"  {len(hidden)} {level}(s) hidden by --min-posts {min_posts}.")
    measured = [r for r in topics if r["median_engagement"] is not None]
    print(f"\n  {len(topics)} {level}(s); {len(measured)} with any engagement data.")
    if measured:
        print("  Look for high engagement with few accounts and fewer ideas —")
        print("  that is loud demand meeting thin, repetitive supply.")
    else:
        print("  No engagement data at all, so this is a supply census only.")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db")
    parser.add_argument("--client", required=True)
    parser.add_argument("--by", choices=("topic", "subtopic"), default="topic",
                        help="grouping level; subtopic is the finer grain the "
                             "extractor already records")
    parser.add_argument("--dedupe-threshold", type=float, default=DEFAULT_DEDUPE_THRESHOLD)
    parser.add_argument("--min-posts", type=int, default=1,
                        help="hide rows with fewer posts; one post's engagement "
                             "is an anecdote, not a rate")
    args = parser.parse_args(argv)
    if not 0 < args.dedupe_threshold <= 1:
        raise SystemExit("--dedupe-threshold must be between 0 and 1.")
    if args.min_posts < 1:
        raise SystemExit("--min-posts must be at least 1.")
    return cmd_report(args.db, args.client, args.dedupe_threshold, args.by,
                      args.min_posts)


if __name__ == "__main__":
    raise SystemExit(main())
