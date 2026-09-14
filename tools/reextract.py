#!/usr/bin/env python3
"""Sample re-extraction tool for prompt validation.

Re-extraction here means "append a small labelled comparison run". It never
deletes, overwrites, or cleans up existing message_units rows.

    reextract.py run <db> --posts N [--run-id NAME] [--dry-run]
    reextract.py compare <db> --run-id NAME
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import zip_longest

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

import core.db as db_mod  # noqa: E402
from analysis.extraction import (  # noqa: E402
    extract_message_units,
    prompt_version_for_client,
    save_message_units,
)
from core.config import get_secret  # noqa: E402
from core.db import DEFAULT_CLIENT_ID  # noqa: E402


_REQUIRED_COLUMNS = (
    "unit_role",
    "unit_role_source",
    "unit_role_confidence",
    "topic_id",
    "taxonomy_id",
    "extraction_run_id",
    "prompt_version",
)


def _connect(db_path: str, read_only: bool):
    import duckdb
    return duckdb.connect(db_path, read_only=read_only)


@contextmanager
def _using_db_path(db_path: str):
    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    try:
        yield
    finally:
        db_mod.DB_PATH = old_path


def _migration_command(db_path: str) -> str:
    return f"python -c \"import core.db as db; db.DB_PATH={db_path!r}; db.init_db()\""


def _require_schema(conn, db_path: str):
    present = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'message_units'"
        ).fetchall()
    }
    missing = [c for c in _REQUIRED_COLUMNS if c not in present]
    if not missing:
        return
    raise SystemExit(
        f"\n  This database predates required message_units columns: {', '.join(missing)}.\n\n"
        "  This tool will not write around that, and --dry-run opens the database\n"
        "  read-only, so it cannot migrate it for you. Run the additive migration\n"
        "  once, then re-run this command:\n\n"
        f"      {_migration_command(db_path)}\n"
    )


def _default_run_id() -> str:
    return "reextract-" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _select_posts(conn, client_id: str, limit: int):
    return conn.execute(
        """
        SELECT
            t.post_id,
            t.transcript,
            COUNT(mu.unit_id) AS existing_units
        FROM transcripts t
        JOIN client_posts cp
          ON cp.post_id = t.post_id
         AND cp.client_id = ?
        JOIN message_units mu
          ON mu.post_id = t.post_id
         AND mu.client_id = ?
        WHERE t.transcript IS NOT NULL
          AND LENGTH(TRIM(t.transcript)) > 0
        GROUP BY t.post_id, t.transcript
        ORDER BY t.post_id
        LIMIT ?
        """,
        [client_id, client_id, limit],
    ).fetchall()


def _run_id_exists(conn, run_id: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM message_units WHERE extraction_run_id = ? LIMIT 1",
            [run_id],
        ).fetchone()
    )


def cmd_run(
    db_path: str,
    posts: int,
    run_id: str | None,
    *,
    dry_run: bool,
    client_id: str,
    api_key: str | None = None,
) -> int:
    if posts < 1:
        raise SystemExit("--posts must be at least 1")
    run_id = run_id or _default_run_id()

    conn = _connect(db_path, read_only=dry_run)
    try:
        _require_schema(conn, db_path)
        if _run_id_exists(conn, run_id):
            raise SystemExit(
                f"\n  extraction_run_id {run_id!r} already exists. Pick a new --run-id.\n"
            )
        rows = _select_posts(conn, client_id, posts)
    finally:
        conn.close()

    with _using_db_path(db_path):
        prompt_version = prompt_version_for_client(client_id)

    print(f"\nreextract run {run_id!r} — client {client_id}")
    print(f"Prompt version: {prompt_version}")
    print(f"Selected posts: {len(rows)}/{posts}\n")
    for post_id, _transcript, existing_units in rows:
        print(f"  {post_id}: {existing_units} existing unit(s)")

    if dry_run:
        print("\nDRY RUN — opened read-only and wrote nothing.\n")
        return 0

    resolved_key = (api_key or get_secret("OPENAI_API_KEY")).strip()
    if not resolved_key:
        raise SystemExit(
            "\n  OPENAI_API_KEY is required for a write run. "
            "Pass --api-key or set the environment/secrets value.\n"
        )

    total_units = 0
    total_cost = 0.0
    with _using_db_path(db_path):
        for post_id, transcript, _existing_units in rows:
            units, cost = extract_message_units(
                transcript,
                post_id,
                resolved_key,
                client_id=client_id,
                extraction_run_id=run_id,
            )
            for unit in units:
                unit.client_id = client_id
                unit.extraction_run_id = run_id
            save_message_units(units, client_id)
            total_units += len(units)
            total_cost += cost
            print(f"  wrote {len(units)} unit(s) for {post_id}")

    print(f"\nWrote {total_units} new unit(s) tagged extraction_run_id={run_id!r}.")
    print(f"Existing rows were not updated or deleted. Extraction cost: ${total_cost:.6f}\n")
    return 0


def _rows_for_compare(conn, post_id: str, client_id: str, run_id: str, new: bool):
    predicate = "extraction_run_id = ?" if new else "extraction_run_id IS DISTINCT FROM ?"
    return conn.execute(
        f"""
        SELECT unit_id, topic, content_type, claim, text
        FROM message_units
        WHERE post_id = ?
          AND client_id = ?
          AND {predicate}
        ORDER BY extracted_at NULLS FIRST, unit_id
        """,
        [post_id, client_id, run_id],
    ).fetchall()


def _cell(row) -> str:
    if row is None:
        return ""
    unit_id, topic, content_type, claim, text = row
    body = claim or text or ""
    body = " ".join(str(body).split())
    if len(body) > 90:
        body = body[:87] + "..."
    return f"{unit_id} [{topic or '-'} / {content_type or '-'}] {body}"


def cmd_compare(db_path: str, run_id: str, *, client_id: str) -> int:
    conn = _connect(db_path, read_only=True)
    try:
        _require_schema(conn, db_path)
        post_ids = [
            r[0]
            for r in conn.execute(
                """
                SELECT DISTINCT post_id
                FROM message_units
                WHERE extraction_run_id = ?
                  AND client_id = ?
                ORDER BY post_id
                """,
                [run_id, client_id],
            ).fetchall()
        ]
        if not post_ids:
            raise SystemExit(f"\n  No message_units found for extraction_run_id={run_id!r}.\n")

        print(f"\nreextract compare — run {run_id!r}, client {client_id}\n")
        for post_id in post_ids:
            old_rows = _rows_for_compare(conn, post_id, client_id, run_id, new=False)
            new_rows = _rows_for_compare(conn, post_id, client_id, run_id, new=True)
            print(f"Post {post_id}: before {len(old_rows)} unit(s), run {len(new_rows)} unit(s)")
            print("  # | before | run")
            for idx, (old, new) in enumerate(zip_longest(old_rows, new_rows), start=1):
                print(f"  {idx} | {_cell(old)} | {_cell(new)}")
            print()
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="append a sampled re-extraction run")
    run.add_argument("db")
    run.add_argument("--posts", type=int, required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--client", default=DEFAULT_CLIENT_ID)
    run.add_argument("--api-key", default=None)

    compare = sub.add_parser("compare", help="print old and new units side by side")
    compare.add_argument("db")
    compare.add_argument("--run-id", required=True)
    compare.add_argument("--client", default=DEFAULT_CLIENT_ID)

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(
            args.db,
            args.posts,
            args.run_id,
            dry_run=args.dry_run,
            client_id=args.client,
            api_key=args.api_key,
        )
    if args.command == "compare":
        return cmd_compare(args.db, args.run_id, client_id=args.client)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
