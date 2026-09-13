#!/usr/bin/env python3
"""Backfill `unit_role` over existing message units (creative-director-ai #27).

Classifies in place. Does NOT re-extract: units keep their unit_id, their text,
their provenance and their taxonomy_id, because re-extraction costs money and
loses the audit trail tying a unit to the transcript it came from.

    classify_unit_roles.py --dry-run                 show the split, write nothing
    classify_unit_roles.py --apply                   write the rule-decided rows
    classify_unit_roles.py --apply --with-model      also classify the remainder

Rules run first and cost nothing. Only what the rules cannot decide is worth
paying a model for, and that share is exactly what --dry-run tells you.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

from analysis.unit_role import role_from_rules  # noqa: E402

# gpt-4o-mini, the rate analysis/extraction.py documents
_IN, _OUT = 0.15 / 1_000_000, 0.60 / 1_000_000
_EST_IN, _EST_OUT = 220, 12   # per unit: short prompt, one-word answer


def _connect(db_path: str, read_only: bool):
    import duckdb
    return duckdb.connect(db_path, read_only=read_only)


_REQUIRED_COLUMNS = ("unit_role", "unit_role_source", "unit_role_confidence")


def _require_schema(conn, db_path: str):
    """Fail with an instruction, not a BinderException.

    --dry-run opens the database read-only on purpose, so it cannot add the
    columns itself. An existing database predates them, and every test builds a
    fresh one through init_db() where they always exist — which is exactly why
    this was not caught before a real database hit it.
    """
    present = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'message_units'").fetchall()}
    missing = [c for c in _REQUIRED_COLUMNS if c not in present]
    if not missing:
        return
    raise SystemExit(
        f"\n  This database predates the unit_role columns: {', '.join(missing)}.\n\n"
        "  --dry-run opens it read-only, so it cannot add them itself. Run the\n"
        "  migration once — it is additive (ALTER TABLE ADD COLUMN), and it is\n"
        "  the same thing the app does on every launch:\n\n"
        f"      python -c \"import core.db as db; db.DB_PATH='{db_path}'; db.init_db()\"\n\n"
        "  Then re-run this command.\n"
    )


def _fetch(conn, client_id):
    # No OR here, deliberately. This clause used to read
    #   "unit_role IS NULL OR unit_role_source IS DISTINCT FROM 'human'"
    # which, joined to "client_id = ?" with AND, parses as
    #   unit_role IS NULL OR (source IS DISTINCT FROM 'human' AND client_id = ?)
    # because AND binds tighter than OR — so every unclassified row of every
    # OTHER client was swept in too. A test caught it. Client isolation is the
    # foundation everything else sits on, so the fix removes the precedence
    # hazard rather than parenthesising around it.
    where = ["unit_role_source IS DISTINCT FROM 'human'"]
    params = []
    if client_id:
        where.append("client_id = ?")
        params.append(client_id)
    return conn.execute(
        "SELECT unit_id, client_id, text, claim, content_type, unit_role, unit_role_source "
        f"FROM message_units WHERE {' AND '.join(where)}", params
    ).fetchall()


def _triage(rows):
    """Split rows into rule-decided, already-human, and needs-a-model."""
    decided, human, deferred = [], [], []
    for uid, cid, text, claim, ctype, role, source in rows:
        if source == "human":
            human.append((uid, role))
            continue
        verdict = role_from_rules(text=text or "", claim=claim or "", content_type=ctype or "")
        if verdict:
            decided.append((uid, verdict))
        else:
            deferred.append((uid, cid, text, claim, ctype))
    return decided, human, deferred


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="reels.duckdb")
    ap.add_argument("--client", default=None, help="restrict to one client_id")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--with-model", action="store_true",
                    help="with --apply, also classify what the rules could not")
    args = ap.parse_args()

    conn = _connect(args.db, read_only=args.dry_run)
    try:
        _require_schema(conn, args.db)
        rows = _fetch(conn, args.client)
        if not rows:
            print("\nNo units to classify. Either the table is empty or every row is "
                  "already marked by a human.\n")
            return
        decided, human, deferred = _triage(rows)

        n = len(rows)
        print(f"\nunit_role backfill — {n:,} units"
              + (f" for client {args.client}" if args.client else "") + "\n")

        tally = Counter(role for _uid, role in decided)
        print(f"  Decided by rules        {len(decided):>7,}  {len(decided)/n*100:5.1f}%")
        for role, count in tally.most_common():
            print(f"      {role:18} {count:>7,}")
        print(f"  Already marked human    {len(human):>7,}  {len(human)/n*100:5.1f}%  (never overwritten)")
        print(f"  Needs a model           {len(deferred):>7,}  {len(deferred)/n*100:5.1f}%")

        cost = len(deferred) * (_EST_IN * _IN + _EST_OUT * _OUT)
        print(f"\n  Estimated model cost    ${cost:,.2f}  "
              f"({len(deferred):,} x ~{_EST_IN}+{_EST_OUT} tokens @ gpt-4o-mini)")

        if args.dry_run:
            print("\n  DRY RUN — nothing was written.\n")
            return

        conn.executemany(
            "UPDATE message_units SET unit_role = ?, unit_role_source = 'rule' "
            "WHERE unit_id = ? AND unit_role_source IS DISTINCT FROM 'human'",
            [(role, uid) for uid, role in decided],
        )
        print(f"\n  Wrote {len(decided):,} rule-decided roles.")
        if args.with_model:
            print("  --with-model is not implemented yet: the model pass needs the "
                  "prompt calibrated against a marked sample first, and guessing a\n"
                  "  role is worse than leaving it NULL. The rows above stay unclassified.")
        print(f"  {len(deferred):,} units remain NULL — not classified, which is not "
              "the same as 'subject'.\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
