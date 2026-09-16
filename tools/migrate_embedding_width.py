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
2. Takes down any index on message_units, replaces the column at the
   requested width, and puts every index back exactly as it was. DuckDB
   cannot widen a fixed-size array in place, and every value is being
   discarded anyway.
3. Adds `embedding_model`, so an unattributable vector cannot happen twice.
   This is the same lesson as extraction_run_id in creative-director-ai#37,
   learned again one table over.

## Why it is not one transaction

DuckDB refuses to drop a column while an index references a column positioned
after it, and its dependency check does NOT see uncommitted index drops —
dropping the indexes and the column inside one transaction still fails with
`Cannot drop this column: an index depends on a column after it!` even though
duckdb_indexes() reports them gone. The drops have to be committed first, so
step 2 cannot be atomic.

Step 1 therefore commits on its own, before anything is taken down. If step 2
dies halfway, no vector is lost — they are all in the archive — and re-running
is safe: the copy skips units already archived. The one thing a half-finished
run can leave behind is a missing index, and the failure message prints the
exact statement that puts it back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARCHIVE_TABLE = "embedding_archive"
DEFAULT_WIDTH = 1536


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


def indexes_on(conn, table: str) -> list[tuple[str, str]]:
    """(name, recreate SQL) for every index on a table.

    DuckDB refuses to drop a column while an index references a column
    positioned after it, because dropping shifts every later column's position:

        Cannot drop this column: an index depends on a column after it!

    The real database hit this and core/db.py does not explain why — its three
    declared message_units indexes are all on columns BEFORE embedding. So the
    offending index is one the schema never declared, and enumerating is the
    only honest way to find it. duckdb_indexes() hands back the exact
    statement that created each one, so they can be put back as they were
    rather than reconstructed from a guess.
    """
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = ?",
        [table]).fetchall()]


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


def index_sql_missing_message(names: list[str]) -> str:
    """Refusal text for an index that cannot be faithfully recreated.

    Silently losing an index is the worst outcome available here: nobody
    notices until a query is slow months later, and by then the cause is long
    gone from anyone's memory.
    """
    listed = ", ".join(repr(n) for n in names)
    subject = "index" if len(names) == 1 else "indexes"
    return (f"\n  No recorded SQL for {subject} {listed} on message_units, so it\n"
            "  cannot be recreated after the column swap.\n"
            "  Nothing was dropped — the column still holds every vector it did.\n")


def indexes_not_rebuilt(saved: list[tuple[str, str]], rebuilt: set[str]) -> list[str]:
    """Names that were taken down and did not come back.

    Extracted because the integration path cannot produce its failure: every
    recreate either works or raises, so the caller's `if missing` branch is
    unreachable from a real database and a mutation deleting it survives the
    whole suite. A guard whose failure condition the tests cannot create is
    not tested, however many tests surround it.
    """
    return sorted({n for n, _ in saved} - rebuilt)


def index_rebuild_message(missing: list[str], saved: list[tuple[str, str]]) -> str:
    """What to tell an operator when the column swapped but an index did not
    come back.

    This is the one failure that cannot be rolled back, so it must not be
    reported as if it had been. The vectors are safe in the archive and the
    column is the new width; what is missing is an index, and the statement
    that recreates it is known exactly. Printing it beats describing it.
    """
    sql = "\n".join(f"    {s}" for n, s in saved if n in set(missing))
    return (f"\n  The column was replaced, but these indexes did not come "
            f"back: {', '.join(missing)}.\n"
            "  Vectors are safe in " + ARCHIVE_TABLE + " and the column is the "
            "new width.\n  Recreate the indexes by running:\n\n" + sql + "\n")


def _archive_vectors(conn, width: int) -> int:
    """Copy every vector into the archive and verify it landed. Transactional.

    Returns the number archived. Raises SystemExit without changing anything
    if the copy is short.

    Rows already in the archive are skipped, so a re-run after a failed schema
    swap does not store a second copy of the same vector. The skip is written
    as NOT EXISTS rather than NOT IN: a single NULL unit_id anywhere in the
    archive makes `NOT IN` never true, which would copy nothing at all.
    """
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
        FROM message_units mu
        WHERE mu.embedding IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM {ARCHIVE_TABLE} a
                          WHERE a.unit_id = mu.unit_id)""",
                 [str(width)])

    # Coverage, not row count: every unit that has a vector must have a row in
    # the archive. Counting archive rows alone would pass on a re-run that
    # copied nothing new but also had nothing to copy for the newest units.
    archived = conn.execute(f"""
        SELECT COUNT(*) FROM message_units mu
        WHERE mu.embedding IS NOT NULL
          AND EXISTS (SELECT 1 FROM {ARCHIVE_TABLE} a
                      WHERE a.unit_id = mu.unit_id)""").fetchone()[0]
    if not archive_is_complete(before, archived):
        conn.execute("ROLLBACK")
        raise SystemExit(archive_shortfall_message(before, archived))
    conn.execute("COMMIT")
    return before


def _swap_column(conn, width: int) -> list[str]:
    """Replace the embedding column, taking the blocking indexes down with it.

    Deliberately NOT inside a transaction. DuckDB's column-drop dependency
    check does not see uncommitted index drops: dropping both indexes and then
    the column inside one transaction still fails with

        Cannot drop this column: an index depends on a column after it!

    even though duckdb_indexes() reports them gone. The drops have to be
    committed first, which means this sequence cannot be atomic. That is
    survivable only because the archive is already committed before this runs,
    so no vector depends on this completing.

    Returns the names of the indexes taken down and put back.
    """
    saved = indexes_on(conn, "message_units")
    unrecorded = [name for name, sql in saved if not sql]
    if unrecorded:
        # Refuse while everything is still standing. After the first DROP INDEX
        # there is no way back.
        raise SystemExit(index_sql_missing_message(unrecorded))

    for name, _sql in saved:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.execute("ALTER TABLE message_units DROP COLUMN embedding")
    conn.execute(f"ALTER TABLE message_units ADD COLUMN embedding FLOAT[{width}]")
    for _name, sql in saved:
        conn.execute(sql)

    rebuilt = {n for n, _ in indexes_on(conn, "message_units")}
    missing = indexes_not_rebuilt(saved, rebuilt)
    if missing:
        raise SystemExit(index_rebuild_message(missing, saved))
    if not column_type(conn, "message_units", "embedding_model"):
        conn.execute("ALTER TABLE message_units ADD COLUMN embedding_model TEXT")
    return [n for n, _ in saved]


def migrate(db_path: str, width: int) -> dict:
    """Archive, then replace the column. Raises rather than losing a vector."""
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        archived = _archive_vectors(conn, width)
        rebuilt = _swap_column(conn, width)
        return {"archived": archived, "width": width, "indexes_rebuilt": rebuilt}
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
    rebuilt = result.get("indexes_rebuilt") or []
    if rebuilt:
        print(f"  Dropped and rebuilt {len(rebuilt)} index(es): {', '.join(rebuilt)}.")
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
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--apply", action="store_true",
                        help="actually change the database; omit for a dry run")
    args = parser.parse_args(argv)
    if args.width < 1:
        raise SystemExit("--width must be positive.")
    return cmd_main(args.db, args.width, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
