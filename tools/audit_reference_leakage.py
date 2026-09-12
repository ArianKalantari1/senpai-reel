#!/usr/bin/env python3
"""Measure whether competitor wording survives into generated client content.

Read-only: opens the senpai-reel DuckDB, compares every generated output
against the competitor message units it actually cited, and reports verbatim
n-gram overlap.

    python tools/audit_reference_leakage.py /path/to/reels.duckdb
    python tools/audit_reference_leakage.py reels.duckdb --n 5 --show 10 --json out.json

Exit codes: 0 clean, 1 overlap found, 2 could not run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

DEFAULT_N = 5

# Words too common for an overlap to mean anything. A shared 5-gram of these
# is idiom, not copying.
STOPish = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for",
    "with", "your", "you", "it", "is", "are", "be", "this", "that", "at", "as",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation and emoji, collapse to a token list."""
    return TOKEN_RE.findall((text or "").lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def is_generic(gram: tuple[str, ...]) -> bool:
    """True when every token is a filler word, so the match is not meaningful."""
    return all(tok in STOPish for tok in gram)


def load_rows(db_path: str):
    """Return generation rows with the reference unit text they cited."""
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)
    try:
        gens = conn.execute(
            """
            SELECT gen_id, content_type, output_text, source_units
            FROM generated_content
            WHERE output_text IS NOT NULL
            """
        ).fetchall()

        units = {
            uid: txt
            for uid, txt in conn.execute(
                "SELECT unit_id, text FROM message_units"
            ).fetchall()
        }
    finally:
        conn.close()

    rows = []
    for gen_id, ctype, output, source_units in gens:
        cited = [(u, units.get(u, "")) for u in (source_units or []) if units.get(u)]
        rows.append((gen_id, ctype, output, cited))
    return rows


def audit(rows, n: int = DEFAULT_N):
    findings = []
    counted = 0
    for gen_id, ctype, output, cited in rows:
        if not cited:
            continue
        counted += 1
        out_grams = ngrams(normalise(output), n)
        if not out_grams:
            continue
        for unit_id, unit_text in cited:
            shared = out_grams & ngrams(normalise(unit_text), n)
            shared = {g for g in shared if not is_generic(g)}
            if shared:
                findings.append(
                    {
                        "gen_id": gen_id,
                        "content_type": ctype,
                        "unit_id": unit_id,
                        "overlaps": sorted(" ".join(g) for g in shared),
                        "unit_text": unit_text,
                    }
                )
    return findings, counted


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("db", help="path to reels.duckdb")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="n-gram length (default 5)")
    ap.add_argument("--show", type=int, default=5, help="how many findings to print in full")
    ap.add_argument("--json", metavar="PATH", help="also write full findings as JSON")
    args = ap.parse_args()

    try:
        rows = load_rows(args.db)
    except Exception as exc:
        print(f"Could not read {args.db}: {exc}", file=sys.stderr)
        return 2

    findings, counted = audit(rows, args.n)

    print(f"Reference leakage audit  (n={args.n})")
    print(f"  generations in database : {len(rows)}")
    print(f"  with cited references   : {counted}")

    if not counted:
        print("\nNo generation cited any reference unit. Nothing to measure -")
        print("generate content with reference units selected, then re-run.")
        return 0

    affected = {f["gen_id"] for f in findings}
    rate = len(affected) / counted * 100
    print(f"  containing source n-grams: {len(affected)}  ({rate:.1f}%)")

    if not findings:
        print("\nCLEAN - no non-generic source n-gram survived into any output.")
        return 0

    worst = Counter()
    for finding in findings:
        for overlap in finding["overlaps"]:
            worst[overlap] += 1

    print(f"\n{len(findings)} generation/reference pairs share wording.\n")
    for finding in findings[: args.show]:
        print(f"  [{finding['content_type']}] {finding['gen_id']}  <- unit {finding['unit_id']}")
        for overlap in finding["overlaps"][:4]:
            print(f'      "{overlap}"')
        print(f"      reference: {finding['unit_text'][:110]}")
        print()

    if len(findings) > args.show:
        print(f"  ... {len(findings) - args.show} more (use --show or --json)\n")

    print("Most repeated overlaps:")
    for gram, count in worst.most_common(5):
        print(f'  {count:3}x  "{gram}"')

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n": args.n,
                    "generations": len(rows),
                    "with_references": counted,
                    "affected": len(affected),
                    "findings": findings,
                },
                fh,
                indent=2,
            )
        print(f"\nFull findings written to {args.json}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
