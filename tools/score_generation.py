#!/usr/bin/env python3
"""Score generation quality with a blind human A/B preference test.

This tool asks one narrow question: does grounding generation in the corpus
beat the same brief without reference units?

It does not judge output quality itself. The model writes the candidates; a
human chooses left, right, or tie in a blind TSV; `compare` only counts that
human judgement and reports the cost of each condition.

    score_generation.py run <db> --client X --pairs N [--content-type caption|hooks|script]
    score_generation.py blind <db> --run-id NAME [out.tsv]
    score_generation.py compare <marked.tsv>
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

import core.db as db_mod  # noqa: E402
from analysis.content_gen import (  # noqa: E402
    GeneratedContent,
    generate_caption,
    generate_hooks,
    generate_script,
)
from analysis.unit_role import ROLES_EXCLUDED_FROM_GENERATION  # noqa: E402
from core.clients import get_client  # noqa: E402
from core.config import get_secret  # noqa: E402


REFERENCE_UNITS_PER_PAIR = 8
MIN_MEANINGFUL_PAIRS = 20
DEFAULT_CONTENT_TYPE = "caption"
DEFAULT_BLIND_OUT = "generation_sample.tsv"


@dataclass(frozen=True)
class ReferenceUnit:
    unit_id: str
    topic: str
    content_type: str
    text: str
    claim: str
    unit_role: str | None = None


@dataclass(frozen=True)
class GeneratedRow:
    gen_id: str
    client_id: str
    topic: str
    content_type: str
    output_text: str
    model: str
    source_units: list[str]
    cost_usd: float | None
    created_at: object

    @property
    def condition(self) -> str:
        return "grounded" if self.source_units else "bare"


@dataclass(frozen=True)
class Pair:
    pair_no: int
    grounded: GeneratedRow
    bare: GeneratedRow


def tsv_cell(value) -> str:
    return " ".join(str(value or "").split()).replace("\t", " ")


def _default_run_id() -> str:
    return "generation-" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _key_path(marked_path: str) -> Path:
    path = Path(marked_path)
    return path.with_suffix(path.suffix + ".key.tsv") if path.suffix else path.with_name(path.name + ".key.tsv")


def _cost_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _parse_cost(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


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


def _require_generated_schema(conn, db_path: str):
    present = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'generated_content'"
        ).fetchall()
    }
    if "generation_run_id" in present:
        return
    raise SystemExit(
        "\n  This database predates generated_content.generation_run_id.\n\n"
        "  `blind` opens the database read-only, so it cannot migrate it for you. "
        "Run the additive migration once, then re-run this command:\n\n"
        f"      {_migration_command(db_path)}\n"
    )


def _init_db_path(db_path: str):
    with _using_db_path(db_path):
        db_mod.init_db()


def _client_context(db_path: str, client_id: str) -> dict:
    with _using_db_path(db_path):
        client = get_client(client_id)
    if not client:
        raise SystemExit(f"\n  No client found for {client_id!r}.\n")
    return client


def _load_reference_units(db_path: str, client_id: str) -> list[ReferenceUnit]:
    conn = _connect(db_path, read_only=True)
    try:
        role_placeholders = ", ".join(["?"] * len(ROLES_EXCLUDED_FROM_GENERATION))
        rows = conn.execute(
            f"""
            SELECT
                mu.unit_id,
                mu.topic,
                mu.content_type,
                mu.text,
                mu.claim,
                mu.unit_role
            FROM message_units mu
            WHERE mu.client_id = ?
              AND (mu.unit_role IS NULL OR mu.unit_role NOT IN ({role_placeholders}))
              AND COALESCE(NULLIF(TRIM(mu.claim), ''), NULLIF(TRIM(mu.text), '')) IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM client_posts cp
                  WHERE cp.client_id = ?
                    AND cp.post_id = mu.post_id
              )
            ORDER BY mu.topic NULLS LAST, mu.unit_id
            """,
            [client_id, *ROLES_EXCLUDED_FROM_GENERATION, client_id],
        ).fetchall()
    finally:
        conn.close()

    return [
        ReferenceUnit(
            unit_id=r[0],
            topic=r[1] or "General",
            content_type=r[2] or "",
            text=r[3] or "",
            claim=r[4] or "",
            unit_role=r[5],
        )
        for r in rows
    ]


def _group_by_topic(units: list[ReferenceUnit]) -> dict[str, list[ReferenceUnit]]:
    grouped: dict[str, list[ReferenceUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.topic or "General", []).append(unit)
    return grouped


def _generate(
    content_type: str,
    topic: str,
    reference_units: list[ReferenceUnit],
    api_key: str,
    client_id: str,
    client_context: dict,
    run_id: str,
) -> GeneratedContent:
    if content_type == "caption":
        return generate_caption(
            topic,
            topic,
            "professional",
            reference_units,
            api_key,
            client_id=client_id,
            client_context=client_context,
            generation_run_id=run_id,
        )
    if content_type == "hooks":
        return generate_hooks(
            topic,
            topic,
            reference_units,
            api_key,
            count=5,
            client_id=client_id,
            client_context=client_context,
            generation_run_id=run_id,
        )
    if content_type == "script":
        return generate_script(
            topic,
            45,
            "professional",
            reference_units,
            api_key,
            client_id=client_id,
            client_context=client_context,
            generation_run_id=run_id,
        )
    raise SystemExit(f"Unknown content type: {content_type}")


def _sum_costs(values: list[float | None]) -> tuple[float | None, int]:
    missing = sum(1 for value in values if value is None)
    if missing:
        return None, missing
    return sum(value for value in values if value is not None), 0


def _print_cost_summary(grounded_costs: list[float | None], bare_costs: list[float | None]):
    grounded_total, grounded_missing = _sum_costs(grounded_costs)
    bare_total, bare_missing = _sum_costs(bare_costs)

    print("\nCost by condition:")
    for label, total, missing, count in [
        ("grounded", grounded_total, grounded_missing, len(grounded_costs)),
        ("bare", bare_total, bare_missing, len(bare_costs)),
    ]:
        if total is None or count == 0:
            print(f"  {label:8} — unknown ({missing}/{count} missing)")
        else:
            print(f"  {label:8} ${total:.6f} total  (${total / count:.6f}/pair)")

    if grounded_total is not None and bare_total is not None and bare_total > 0:
        print(f"  ratio    {grounded_total / bare_total:.1f}x grounded/bare")
    else:
        print("  ratio    —")


def cmd_run(
    db_path: str,
    *,
    client_id: str,
    pairs: int,
    content_type: str = DEFAULT_CONTENT_TYPE,
) -> int:
    if pairs < 1:
        raise SystemExit("--pairs must be at least 1")

    _init_db_path(db_path)
    api_key = get_secret("OPENAI_API_KEY").strip()
    if not api_key:
        raise SystemExit(
            "\n  OPENAI_API_KEY is required to generate A/B pairs. "
            "Add it to Settings, `.streamlit/secrets.toml`, or the environment.\n"
        )

    client_context = _client_context(db_path, client_id)
    units = _load_reference_units(db_path, client_id)
    grouped = _group_by_topic(units)
    topics = sorted(topic for topic, topic_units in grouped.items() if topic_units)
    if not topics:
        raise SystemExit(
            "\n  No eligible reference units found for this client. "
            "Need at least one non-technique/meta/offtopic unit.\n"
        )

    run_id = _default_run_id()
    print(f"\ngeneration run {run_id!r} — client {client_id}")
    print(f"Content type: {content_type}")
    print(f"Pairs requested: {pairs}\n")

    grounded_costs: list[float | None] = []
    bare_costs: list[float | None] = []
    with _using_db_path(db_path):
        for index in range(pairs):
            topic = topics[index % len(topics)]
            refs = grouped[topic][:REFERENCE_UNITS_PER_PAIR]
            print(f"  pair {index + 1:>3}: {topic} ({len(refs)} reference unit(s))")
            grounded = _generate(content_type, topic, refs, api_key, client_id, client_context, run_id)
            bare = _generate(content_type, topic, [], api_key, client_id, client_context, run_id)
            grounded_costs.append(grounded.cost_usd)
            bare_costs.append(bare.cost_usd)

    print(f"\nWrote {pairs * 2} generated row(s) tagged generation_run_id={run_id!r}.")
    _print_cost_summary(grounded_costs, bare_costs)
    print(f"\nBlind them with: score_generation.py blind {db_path} --run-id {run_id}\n")
    return 0


def _row_from_db(row) -> GeneratedRow:
    source_units = list(row[6] or [])
    return GeneratedRow(
        gen_id=row[0],
        client_id=row[1],
        topic=row[2] or "General",
        content_type=row[3] or "",
        output_text=row[4] or "",
        model=row[5] or "",
        source_units=source_units,
        cost_usd=row[7],
        created_at=row[8],
    )


def _load_pairs(db_path: str, run_id: str) -> list[Pair]:
    conn = _connect(db_path, read_only=True)
    try:
        _require_generated_schema(conn, db_path)
        rows = [
            _row_from_db(row)
            for row in conn.execute(
                """
                SELECT gen_id, client_id, topic, content_type, output_text, model,
                       source_units, cost_usd, created_at
                FROM generated_content
                WHERE generation_run_id = ?
                ORDER BY created_at, gen_id
                """,
                [run_id],
            ).fetchall()
        ]
    finally:
        conn.close()

    grouped: dict[tuple[str, str, str, str], dict[str, list[GeneratedRow]]] = {}
    for row in rows:
        key = (row.client_id, row.content_type, row.topic, row.model)
        grouped.setdefault(key, {"grounded": [], "bare": []})[row.condition].append(row)

    pairs: list[Pair] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        grounded = bucket["grounded"]
        bare = bucket["bare"]
        if len(grounded) != len(bare):
            raise SystemExit(
                f"\n  Run {run_id!r} is not pair-complete for {key}: "
                f"{len(grounded)} grounded row(s), {len(bare)} bare row(s).\n"
            )
        for grounded_row, bare_row in zip(grounded, bare):
            pairs.append(Pair(len(pairs) + 1, grounded_row, bare_row))

    if not pairs:
        raise SystemExit(f"\n  No generated_content rows found for generation_run_id={run_id!r}.\n")
    return pairs


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict], comments: list[str] | None = None):
    with path.open("w", encoding="utf-8", newline="") as fh:
        for comment in comments or []:
            fh.write(f"# {comment}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cmd_blind(db_path: str, run_id: str, out: str = DEFAULT_BLIND_OUT) -> int:
    pairs = _load_pairs(db_path, run_id)
    rng = random.Random(run_id)
    mark_rows = []
    key_rows = []

    for pair in pairs:
        if rng.random() < 0.5:
            left, right = pair.grounded, pair.bare
        else:
            left, right = pair.bare, pair.grounded
        pair_id = f"pair_{pair.pair_no:03d}"
        mark_rows.append(
            {
                "pair_id": pair_id,
                "PREFERENCE": "?",
                "left": tsv_cell(left.output_text),
                "right": tsv_cell(right.output_text),
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "left_condition": left.condition,
                "left_gen_id": left.gen_id,
                "left_cost_usd": _cost_cell(left.cost_usd),
                "right_condition": right.condition,
                "right_gen_id": right.gen_id,
                "right_cost_usd": _cost_cell(right.cost_usd),
            }
        )

    out_path = Path(out)
    key_path = _key_path(out)
    _write_tsv(
        out_path,
        ["pair_id", "PREFERENCE", "left", "right"],
        mark_rows,
        comments=[
            "Mark PREFERENCE as left / right / tie.",
            "Do not change any other column. Save when done.",
        ],
    )
    _write_tsv(
        key_path,
        [
            "pair_id",
            "left_condition",
            "left_gen_id",
            "left_cost_usd",
            "right_condition",
            "right_gen_id",
            "right_cost_usd",
        ],
        key_rows,
    )

    print(f"\nWrote {len(mark_rows)} pair(s) to {out_path}")
    print(f"Wrote the hidden key to {key_path}")
    print("No labels or generated-content ids are included in the marking file.")
    print(f"\nMark PREFERENCE, then: score_generation.py compare {out_path}\n")
    return 0


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        lines = [line for line in fh if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def _condition_costs(key_rows: list[dict[str, str]]) -> tuple[list[float | None], list[float | None]]:
    costs = {"grounded": [], "bare": []}
    for row in key_rows:
        for side in ("left", "right"):
            condition = row[f"{side}_condition"]
            costs[condition].append(_parse_cost(row.get(f"{side}_cost_usd")))
    return costs["grounded"], costs["bare"]


def cmd_compare(marked: str) -> int:
    marked_path = Path(marked)
    rows = _read_tsv(marked_path)
    if not rows:
        raise SystemExit("\n  No rows found. Fill in PREFERENCE with left/right/tie.\n")

    key_path = _key_path(marked)
    key_rows = _read_tsv(key_path)
    if not key_rows:
        raise SystemExit(f"\n  Missing or empty hidden key file: {key_path}\n")
    key_by_pair = {row["pair_id"]: row for row in key_rows}

    grounded_wins = bare_wins = ties = skipped = 0
    used_keys: list[dict[str, str]] = []
    for row in rows:
        pair_id = row.get("pair_id", "")
        key = key_by_pair.get(pair_id)
        if not key:
            raise SystemExit(f"\n  No hidden key row for pair_id={pair_id!r}.\n")
        used_keys.append(key)

        preference = (row.get("PREFERENCE") or "").strip().lower()
        if preference in ("", "?"):
            skipped += 1
            continue
        if preference == "tie":
            ties += 1
            continue
        if preference not in ("left", "right"):
            raise SystemExit(
                f"\n  Invalid PREFERENCE for {pair_id}: {preference!r}. "
                "Use left, right, or tie.\n"
            )

        winner = key[f"{preference}_condition"]
        if winner == "grounded":
            grounded_wins += 1
        elif winner == "bare":
            bare_wins += 1
        else:
            raise SystemExit(f"\n  Hidden key has invalid condition {winner!r} for {pair_id}.\n")

    marked_count = grounded_wins + bare_wins + ties
    decisive = grounded_wins + bare_wins
    if marked_count == 0:
        raise SystemExit("\n  No marked rows found. Fill in PREFERENCE with left/right/tie.\n")

    print(f"\nGeneration preference — {marked_count} marked pair(s)")
    if skipped:
        print(f"  skipped unmarked rows: {skipped}")
    if decisive:
        pct = grounded_wins / decisive * 100
        print(
            f"  grounded win rate {grounded_wins}/{decisive} decisive "
            f"preference(s) ({pct:.0f}%), sample n={marked_count}"
        )
    else:
        print(f"  grounded win rate — no decisive preferences, sample n={marked_count}")
    print(f"  bare wins {bare_wins}/{decisive}" if decisive else "  bare wins —")
    print(f"  ties {ties}/{marked_count}")

    if marked_count < MIN_MEANINGFUL_PAIRS:
        print(
            f"\n  SAMPLE TOO SMALL: n={marked_count}. Minimum for a directional read "
            f"is {MIN_MEANINGFUL_PAIRS} marked pairs, so treat this as raw notes, "
            "not a finding."
        )
    elif grounded_wins > bare_wins:
        print("\n  Grounded leads this marked sample. Read that alongside cost.")
    elif bare_wins > grounded_wins:
        print("\n  Bare leads this marked sample. That is a real finding if it holds.")
    else:
        print("\n  No preference lead in this marked sample.")

    grounded_costs, bare_costs = _condition_costs(used_keys)
    _print_cost_summary(grounded_costs, bare_costs)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("db")
    run.add_argument("--client", required=True)
    run.add_argument("--pairs", type=int, required=True)
    run.add_argument(
        "--content-type",
        choices=["caption", "hooks", "script"],
        default=DEFAULT_CONTENT_TYPE,
    )

    blind = subparsers.add_parser("blind")
    blind.add_argument("db")
    blind.add_argument("--run-id", required=True)
    blind.add_argument("out", nargs="?", default=DEFAULT_BLIND_OUT)

    compare = subparsers.add_parser("compare")
    compare.add_argument("marked")

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(
            args.db,
            client_id=args.client,
            pairs=args.pairs,
            content_type=args.content_type,
        )
    if args.command == "blind":
        return cmd_blind(args.db, args.run_id, args.out)
    if args.command == "compare":
        return cmd_compare(args.marked)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
