#!/usr/bin/env python3
"""Embed the message units that do not have a vector yet.

    embed_units.py <db> --client CLIENT [--limit N] [--dry-run]

`analysis.embeddings.embed_pending_units` does one batch of 50 and returns,
which is right for a Streamlit progress bar and useless from a terminal. This
loops until nothing is pending, reports what it actually spent, and can be
stopped and restarted — the query selects `embedding IS NULL`, so finished
work is never redone.

Why it matters: on the real corpus only 666 of 5,361 units are embedded (12%),
and the subtopic clustering in tools/topic_gaps.py collapsed 2,156 labels into
1,902 — a 12% reduction that matched the embedding coverage exactly. A
subtopic with no vector cannot be placed in a cluster, so it maps to itself.
The clustering was not too strict; it was starved.

text-embedding-3-small costs $0.02 per million tokens, so the whole backlog is
under a cent. The reason to check first is not the money.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.db as db_mod  # noqa: E402
from analysis.embeddings import embed_pending_units  # noqa: E402
from analysis.extraction import OutOfCreditError, RateLimitedError  # noqa: E402
from core.config import get_secret  # noqa: E402

DEFAULT_BATCH = 50


def pending_count(db_path: str, client_id: str) -> tuple[int, int]:
    """(pending, total) units visible to this client."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    try:
        return conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE mu.embedding IS NULL), COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON cp.post_id = mu.post_id AND cp.client_id = ?
            """,
            [client_id],
        ).fetchone()
    finally:
        conn.close()


def _format_cost(value: float | None) -> str:
    # Never $0.0000 for an unknown. A run that embedded rows and reported no
    # spend is the unpriced-scrape bug wearing different clothes.
    return "unknown" if value is None else f"${value:,.6f}"


def cmd_run(db_path: str, client_id: str, limit: int | None, dry_run: bool) -> int:
    pending, total = pending_count(db_path, client_id)
    print(f"\nEmbedding — client {client_id}")
    print(f"  {pending:,} of {total:,} units have no vector "
          f"({pending / total * 100:.0f}%)" if total else "  No units.")
    if not pending:
        print("  Nothing to do.\n")
        return 0
    if dry_run:
        print(f"  --dry-run: would embed {min(pending, limit or pending):,} unit(s). "
              "Nothing was sent and nothing was written.\n")
        return 0

    api_key = get_secret("OPENAI_API_KEY").strip()
    if not api_key:
        raise SystemExit("\n  OPENAI_API_KEY is required to embed units.\n")

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    done = failed = 0
    spent: float | None = 0.0
    try:
        while True:
            if limit is not None and done >= limit:
                break
            try:
                result = embed_pending_units(api_key, batch_size=DEFAULT_BATCH,
                                             client_id=client_id)
            except OutOfCreditError as exc:
                print(f"\n  {exc}")
                break
            except RateLimitedError as exc:
                print(f"\n  {exc}")
                break
            if not result["total"]:
                break
            if not result["done"] and not result["failed"]:
                # A batch reporting units but resolving none of them would loop
                # against the API forever. The tool cannot tell whether the
                # caller is stuck or the rows are unembeddable, so it stops and
                # says so rather than spinning. Found by writing a fake that
                # did exactly this.
                print("  stopped: a batch reported units but embedded none.")
                break
            done += result["done"]
            failed += result["failed"]
            batch_cost = result.get("total_cost_usd")
            # A batch that reports no cost makes the running total unknowable.
            # Saying so beats printing a total that silently excludes it.
            spent = None if batch_cost is None or spent is None else spent + batch_cost
            print(f"  {done:,} embedded, {failed:,} failed  ({_format_cost(spent)})",
                  flush=True)
            if result.get("error"):
                print(f"  stopped: {result['error']}")
                break
    finally:
        db_mod.DB_PATH = old_path

    remaining, _ = pending_count(db_path, client_id)
    print(f"\n  Embedded {done:,}, failed {failed:,}. Cost {_format_cost(spent)}.")
    print(f"  {remaining:,} still without a vector.")
    print("  Re-run to continue — finished units are never redone.\n")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db")
    parser.add_argument("--client", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after roughly this many units")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is pending without sending anything")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    return cmd_run(args.db, args.client, args.limit, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
