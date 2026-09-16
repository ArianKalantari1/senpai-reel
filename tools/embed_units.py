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
# What text-embedding-3-small returns unless a narrower width is requested.
MODEL_NATIVE_DIMENSIONS = 1536


def embedding_dimension(db_path: str) -> int | None:
    """The width message_units.embedding actually holds, or None if unreadable.

    The repo schema declares FLOAT[1536]. A real database was found declaring
    FLOAT[512] — the resize that creative-director-ai#29 records as never
    written had in fact been applied somewhere, and the code had no idea. Every
    write failed with a per-row cast error while the loop kept paying for more.
    """
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    try:
        row = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'message_units' AND column_name = 'embedding'"
        ).fetchone()
    finally:
        conn.close()
    return width_of(row[0] if row else None)


def width_of(data_type: str | None) -> int | None:
    """Fixed array width from a DuckDB type string, or None if it has none.

    Only a bracketed number counts. DuckDB reports REAL as FLOAT4, so a looser
    search for any digit would read FLOAT4[512] as a 4-wide column and then
    request four-dimension vectors from the API — wrong, and wrong in a way
    that still writes successfully into the wrong shape.

    FLOAT[] — a variable-length list — has no declared width, and that is a
    None rather than a guess.
    """
    import re

    if not data_type:
        return None
    match = re.search(r"\[(\d+)\]", str(data_type))
    return int(match.group(1)) if match else None


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

    width = embedding_dimension(db_path)
    if width is None:
        raise SystemExit(
            "\n  Could not read the width of message_units.embedding, so this "
            "would\n  guess at what the column accepts. Refusing to spend on a "
            "guess.\n")
    print(f"  Column holds FLOAT[{width}]; requesting {width}-dimension vectors.")
    if width != MODEL_NATIVE_DIMENSIONS:
        print(f"  NOTE: the model's native width is {MODEL_NATIVE_DIMENSIONS}. "
              f"Vectors are being\n        truncated to {width} by the API, which "
              "is supported but means these\n        vectors are NOT comparable "
              "with any stored at a different width.")

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
                                             client_id=client_id,
                                             dimensions=width)
            except OutOfCreditError as exc:
                print(f"\n  {exc}")
                break
            except RateLimitedError as exc:
                print(f"\n  {exc}")
                break
            if not result["total"]:
                break
            if not result["done"]:
                # Forward progress means rows LEFT the pending set. A batch
                # where every row failed does not: the same rows are selected
                # again next time and charged again. The first version of this
                # guard only caught done == 0 AND failed == 0, so a database
                # whose column rejected every write looped 190 times, paying
                # for 9,500 embeddings and saving none of them.
                print(f"  stopped: {result['failed']:,} row(s) in this batch "
                      "failed and none were saved.")
                print("  Re-running would select and pay for the same rows again.")
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
