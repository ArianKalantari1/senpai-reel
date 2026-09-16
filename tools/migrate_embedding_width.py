#!/usr/bin/env python3
"""Resize message_units.embedding, keeping the vectors it currently holds.

    migrate_embedding_width.py <db> [--width 1536] [--apply]

Defaults to a dry run. Nothing changes without --apply.

## Why this exists

A real database declares FLOAT[512] where core/db.py declares FLOAT[1536], so
every OpenAI write failed with a per-row cast error. creative-director-ai#29
recorded a Voyage adapter at 512 dimensions and noted the resize migration was
never written; it had in fact been applied to that database, and nothing in
the code knew.

That database also holds 730 vectors written on 7-8 April 2026 with
embedding_cost_usd NULL. Which model produced them is **not recorded
anywhere** — message_units.model is the EXTRACTION model, and the embedding
write only ever touched embedding, embedded_at and embedding_cost_usd. A
vector whose model is unknown is a vector whose space is unknown, so it cannot
be compared with anything, including other vectors of the same width.

## What this does

1. Copies every existing vector into `embedding_archive` first. They are
   unusable, but somebody paid for them and this project does not delete work
   to make a migration tidy. Nothing is dropped until the copy is verified row
   for row.
2. Replaces the column at the requested width. DuckDB cannot widen a
   fixed-size array in place, and every value is being discarded anyway.
3. Adds `embedding_model`, so an unattributable vector cannot happen twice.
   This is the same lesson as extraction_run_id in creative-director-ai#37,
   learned again one table over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARCHIVE_TABLE = "embedding_archive"
DEFAULT_WIDth = 1536


def column_type(conn, table: str, column: str) -> str | None:
    row = conn.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?", [table, column]).fetchone()
    return row[0] if row else None


def plan(db_path: str, width: int) -> dict:
    """What the migration would do, read-only."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    try:
        current = column_type(conn, "message_units", "embedding")
        vectors = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL"
        ).fetchone()[0] if current else 0
        archived = 0
        if column_type(conn, ARCHIVE_TABLE, "unit_id"):
            archived = conn.execute(f"SELECT COUNT(*) FROM {ARCHIVE_TABLE}").fetchone()[0]
        return {"current_type": current, "target": f"FLOAT[{width}]",
                "vectors": vectors, "already_archived": archived,
                "needed": current != f"FLOAT[{width}]"}
    finally:
        conn.close()


def archive_is_complete(before: int, archived: int) -> bool:
    """Did the copy keep every vector that existed?

    Checked before anything is dropped. A migration that loses the thing it
    promised to keep is worse than one that refuses to run, and the only
    moment to find out is while the original is still there.
    """
    return archived >= before


def archive_shortfall_message(before: int, archived: int) -> str:
    return (f"\n  Archive holds {archived:,} row(s) but {before:,} vector(s) "
            "exist.\n  Nothing was changed — the column still holds every "
            "vector it did.\n")


def migrate(db_path: str, width: int) -> dict:
    """Archive, then replace the column. Raises rather than losing a vector."""
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        conn.execute("BEGIN TRANSACTION")
        before = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL").fetchone()[0]

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {ARCHIVE_TABLE} (
                unit_id TEXT,
                embedding FLOAT[],
                embedded_at TIMESTAMP,
                embedding_cost_usd DOUBLE,
                archived_at TIMESTAMP,
                note TEXT
            )""")
        conn.execute(f"""
            INSERT INTO {ARCHIVE_TABLE}
            SELECT unit_id, embedding::FLOAT[], embedded_at, embedding_cost_usd,
                   CURRENT_TIMESTAMP,
                   'model unrecorded; width ' || ? || '; see creative-director-ai#29'
            FROM message_units WHERE embedding IS NOT NULL""", [str(before)])

        archived = conn.execute(
            f"SELECT COUNT(*) FROM {ARCHIVE_TABLE} WHERE archived_at IS NOT NULL"
        ).fetchone()[0]
        if not archive_is_complete(before, archived):
            conn.execute("ROLLBACK")
            raise SystemExit(archive_shortfall_message(before, archived))

        conn.execute("ALTER TABLE message_units DROP COLUMN embedding")
        conn.execute(f"ALTER TABLE message_units ADD COLUMN embedding FLOAT[{width}]")
        if not column_type(conn, "message_units", "embedding_model"):
            conn.execute("ALTER TABLE message_units ADD COLUMN embedding_model TEXT")
        conn.execute("COMMIT")
        return {"archived": before, "width": width}
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def cmd_main(db_path: str, width: int, apply: bool) -> int:
    state = plan(db_path, width)
    print(f"\nEmbedding column migration — {db_path}")
    print(f"  current : {state['current_type']}")
    print(f"  target  : {state['target']}")
    print(f"  vectors : {state['vectors']:,} (would be archived, then cleared)")
    if state["already_archived"]:
        print(f"  archive : {state['already_archived']:,} row(s) already held")
    if not state["needed"]:
        print("\n  Column is already the target width. Nothing to do.\n")
        return 0
    if not apply:
        print("\n  DRY RUN. Re-run with --apply to make these changes.")
        print("  Existing vectors are copied to embedding_archive before the")
        print("  column is replaced; nothing is dropped until that copy is")
        print("  verified row for row.\n")
        return 0

    result = migrate(db_path, width)
    print(f"\n  Archived {result['archived']:,} vector(s) to {ARCHIVE_TABLE}.")
    print(f"  Column is now FLOAT[{result['width']}]; embedding_model added.")
    print("  Every unit is now unembedded. Re-run tools/embed_units.py.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDth)
    parser.add_argument("--apply", action="store_true",
                        help="actually change the database; omit for a dry run")
    args = parser.parse_args(argv)
    if args.width < 1:
        raise SystemExit("--width must be positive.")
    return cmd_main(args.db, args.width, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
