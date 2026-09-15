#!/usr/bin/env python3
"""Calibrate unit_role model classifications against blind human labels.

The model pass should only happen after we know whether the model tracks human
judgement on the rows the deterministic rules cannot decide. This tool creates
that blind sample and compares the marked sample back to the withheld model
opinion.

    score_unit_roles.py blind <db> --client CLIENT [n] [out.tsv] [--seed SEED]
                                                write n deferred units for marking
    score_unit_roles.py compare <marked.tsv>     report agreement by role and overall

`blind` writes two files:
  - out.tsv for human marking, with no model role or confidence
  - out.key.tsv beside it, with the withheld model answer for compare
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

from analysis.extraction import (  # noqa: E402
    _RATE_LIMIT_ATTEMPTS,
    _openai_error_code,
    _retry_after_seconds,
    OutOfCreditError,
    RateLimitedError,
)
from analysis.unit_role import (  # noqa: E402
    UNIT_ROLES,
    role_from_rules,
    unit_roles_for_prompt,
    validate_unit_role,
)
from core.config import get_secret  # noqa: E402

MODEL = "gpt-4o-mini"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

# gpt-4o-mini rates also used by classify_unit_roles.py.
_IN, _OUT = 0.15 / 1_000_000, 0.60 / 1_000_000

_REQUIRED_COLUMNS = (
    "unit_id",
    "client_id",
    "post_id",
    "text",
    "claim",
    "content_type",
    "topic",
    "unit_role",
    "unit_role_source",
    "unit_role_confidence",
)


@dataclass(frozen=True)
class UnitCandidate:
    unit_id: str
    client_id: str
    text: str
    claim: str
    content_type: str
    topic: str


@dataclass(frozen=True)
class ModelVerdict:
    role: str | None
    confidence: float | None
    cost_usd: float | None
    raw: str


def tsv_cell(value: Any) -> str:
    return " ".join(str(value or "").split()).replace("\t", " ")


def _connect(db_path: str):
    import duckdb

    return duckdb.connect(db_path, read_only=True)


def _require_schema(conn, db_path: str) -> None:
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
        f"\n  This database cannot support unit_role calibration yet: "
        f"{', '.join(missing)} missing from message_units.\n\n"
        "  Run the additive migration once:\n\n"
        f"      python -c \"import core.db as db; db.DB_PATH='{db_path}'; db.init_db()\"\n"
    )


def _load_deferred_units(db_path: str, client_id: str) -> list[UnitCandidate]:
    conn = _connect(db_path)
    try:
        _require_schema(conn, db_path)
        rows = conn.execute(
            """
            SELECT mu.unit_id, mu.client_id, mu.text, mu.claim, mu.content_type, mu.topic
            FROM message_units mu
            JOIN client_posts cp
              ON cp.client_id = ? AND cp.post_id = mu.post_id
            WHERE mu.client_id = ?
              AND mu.unit_role_source IS DISTINCT FROM 'human'
              AND (mu.text IS NOT NULL OR mu.claim IS NOT NULL)
            ORDER BY mu.unit_id
            """,
            [client_id, client_id],
        ).fetchall()
    finally:
        conn.close()

    deferred: list[UnitCandidate] = []
    for unit_id, row_client_id, text, claim, content_type, topic in rows:
        verdict = role_from_rules(
            text=text or "",
            claim=claim or "",
            content_type=content_type or "",
        )
        if verdict:
            continue
        deferred.append(
            UnitCandidate(
                unit_id=unit_id,
                client_id=row_client_id,
                text=text or "",
                claim=claim or "",
                content_type=content_type or "",
                topic=topic or "",
            )
        )
    return deferred


def _key_path(out: str) -> str:
    path = Path(out)
    if path.suffix:
        return str(path.with_suffix(path.suffix + ".key.tsv"))
    return str(path.with_name(path.name + ".key.tsv"))


def _format_float(value: float | None, places: int = 4) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _model_messages(unit: UnitCandidate) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify one extracted message unit into exactly one unit_role. "
                "Return JSON only with keys role and confidence."
            ),
        },
        {
            "role": "user",
            "content": (
                "Roles:\n"
                f"{unit_roles_for_prompt()}\n\n"
                f"content_type: {unit.content_type or '[missing]'}\n"
                f"topic: {unit.topic or '[missing]'}\n"
                f"text: {unit.text or '[missing]'}\n"
                f"claim: {unit.claim or '[missing]'}\n\n"
                "Return a JSON object like "
                '{"role":"subject","confidence":0.82}.'
            ),
        },
    ]


def _safe_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= confidence <= 1:
        return confidence
    return None


def _cost_from_usage(usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return float(prompt_tokens) * _IN + float(completion_tokens) * _OUT


def _call_model(unit: UnitCandidate, api_key: str) -> ModelVerdict:
    """One classification call, with the same 429 split as extraction.

    OpenAI answers both "slow down" and "you have no money" with 429. The
    classification here reuses analysis/extraction.py rather than repeating
    it, so the two spending paths cannot drift apart: a transient limit is
    retried, an exhausted balance fails immediately and says so. Without this,
    a run that hits either one dies on a raw HTTPError traceback partway
    through a paid loop.
    """
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        response = requests.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": _model_messages(unit),
            },
            timeout=60,
        )
        if response.status_code != 429:
            break
        code = _openai_error_code(response)
        if "quota" in code or "credit" in code:
            raise OutOfCreditError(
                "OpenAI rejected the request: no credit remaining on the account.\n"
                "Add credit at https://platform.openai.com/settings/organization/billing/\n"
                "then re-run."
            )
        if attempt == _RATE_LIMIT_ATTEMPTS - 1:
            raise RateLimitedError(
                f"Rate limited by OpenAI after {_RATE_LIMIT_ATTEMPTS} attempts. "
                "Wait a minute and re-run — the sample seed is stable, so the "
                "same units are drawn again."
            )
        time.sleep(_retry_after_seconds(response, attempt))

    response.raise_for_status()
    payload = response.json()
    raw = payload["choices"][0]["message"]["content"].strip()
    cost = _cost_from_usage(payload.get("usage"))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ModelVerdict(role=None, confidence=None, cost_usd=cost, raw=raw)
    return ModelVerdict(
        role=validate_unit_role(str(parsed.get("role", ""))),
        confidence=_safe_confidence(parsed.get("confidence")),
        cost_usd=cost,
        raw=raw,
    )


def _write_blind_rows(out: str, sample: list[UnitCandidate]) -> None:
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Mark HUMAN_ROLE as subject / technique / meta / offtopic.\n")
        fh.write("# Leave ? only if you cannot judge the row.\n")
        fh.write("# Do not change any other column. Save when done.\n")
        fh.write("# The model role is withheld in the sidecar key file on purpose.\n")
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["unit_id", "HUMAN_ROLE", "content_type", "topic", "claim", "source"])
        for unit in sample:
            writer.writerow(
                [
                    unit.unit_id,
                    "?",
                    tsv_cell(unit.content_type),
                    tsv_cell(unit.topic),
                    tsv_cell(unit.claim),
                    tsv_cell(unit.text),
                ]
            )


def _write_key_rows(out: str, verdicts: dict[str, ModelVerdict]) -> None:
    with open(_key_path(out), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["unit_id", "model_role", "model_confidence", "model_cost_usd", "model_raw"])
        for unit_id, verdict in verdicts.items():
            writer.writerow(
                [
                    unit_id,
                    verdict.role or "",
                    _format_float(verdict.confidence, places=3),
                    _format_float(verdict.cost_usd, places=6),
                    tsv_cell(verdict.raw),
                ]
            )


def cmd_blind(
    db_path: str,
    client_id: str,
    n: int,
    out: str,
    seed: str = "unit-role-calibration",
    api_key: str | None = None,
) -> None:
    if n < 1:
        raise SystemExit("n must be at least 1.")
    key = api_key if api_key is not None else get_secret("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required to sample model unit_role verdicts.")

    units = _load_deferred_units(db_path, client_id)
    rng = random.Random(seed)
    sample = rng.sample(units, min(n, len(units)))
    if not sample:
        raise SystemExit(
            f"\n  No rule-deferred units found for client {client_id!r}, so {out} "
            "would be empty.\n\n"
            "  The calibration harness only samples rows the deterministic unit_role "
            "rules cannot decide.\n"
        )

    # Written incrementally rather than as a comprehension. Every call in this
    # loop is money already spent; a comprehension that raises on unit 30 of 60
    # throws away 29 paid answers and leaves no files behind.
    verdicts: dict[str, ModelVerdict] = {}
    stopped_by: Exception | None = None
    for unit in sample:
        try:
            verdicts[unit.unit_id] = _call_model(unit, key)
        except (OutOfCreditError, RateLimitedError) as exc:
            stopped_by = exc
            break

    sample = [unit for unit in sample if unit.unit_id in verdicts]
    if sample:
        _write_blind_rows(out, sample)
        _write_key_rows(out, verdicts)

    if stopped_by is not None:
        kept = (
            f"  {len(sample)} unit(s) were classified before it stopped and are "
            f"saved in {out}.\n  Mark those, or re-run for a full sample once "
            "the cause is cleared.\n"
            if sample
            else "  Nothing was classified, so no files were written.\n"
        )
        raise SystemExit(f"\n{stopped_by}\n\n{kept}")

    costs = [v.cost_usd for v in verdicts.values() if v.cost_usd is not None]
    print(f"\nWrote {len(sample)} rule-deferred units to {out}")
    print(f"Wrote withheld model answers to {_key_path(out)}")
    print(f"Sample seed: {seed!r}")
    if len(sample) < n:
        print(f"NOTE: you asked for {n} and only {len(sample)} were available.")
    if len(costs) == len(verdicts):
        print(f"Model cost from API usage: ${sum(costs):,.4f}")
    else:
        print("Model cost unavailable: API usage fields were not returned for every row.")
    print("No model roles are included in the marking file, so your reading is not anchored.")
    print(f"\nMark HUMAN_ROLE, then: score_unit_roles.py compare {out}\n")


def _read_tsv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        filtered = (line for line in fh if not line.startswith("#"))
        reader = csv.DictReader(filtered, delimiter="\t")
        for row in reader:
            rows.append({k: v for k, v in row.items() if k is not None})
    return rows


def _normalise_human_role(value: str) -> str | None:
    stripped = (value or "").strip()
    if stripped in ("", "?"):
        return None
    role = validate_unit_role(stripped)
    if role:
        return role
    raise SystemExit(
        f"HUMAN_ROLE {value!r} is not valid. Use one of: {', '.join(UNIT_ROLES)}."
    )


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def cmd_compare(marked: str) -> None:
    marked_rows = _read_tsv(marked)
    key_rows = _read_tsv(_key_path(marked))
    key_by_unit = {row.get("unit_id", ""): row for row in key_rows}

    total = agree = skipped = 0
    by_human_role: dict[str, Counter[str]] = defaultdict(Counter)
    misses: list[tuple[str, str, str]] = []

    for row in marked_rows:
        unit_id = row.get("unit_id", "")
        human_role = _normalise_human_role(row.get("HUMAN_ROLE", ""))
        if human_role is None:
            skipped += 1
            continue
        if unit_id not in key_by_unit:
            raise SystemExit(f"No withheld model answer found for unit_id={unit_id!r}.")

        model_role = validate_unit_role(key_by_unit[unit_id].get("model_role", ""))
        model_label = model_role or "unclassified"
        by_human_role[human_role][model_label] += 1
        total += 1
        if model_role == human_role:
            agree += 1
        else:
            misses.append((unit_id, human_role, model_label))

    if not total:
        raise SystemExit(
            "No marked rows found. Fill HUMAN_ROLE with subject/technique/meta/offtopic."
        )

    print(f"\nUnit role agreement with your reading: {agree}/{total}  ({_pct(agree, total):.0f}%)")
    if skipped:
        print(f"Skipped unmarked rows: {skipped}")
    print("\nAgreement by human role:")
    for role in UNIT_ROLES:
        counts = by_human_role[role]
        role_total = sum(counts.values())
        role_agree = counts[role]
        if role_total:
            predicted = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            print(f"  {role:10} {role_agree}/{role_total}  ({_pct(role_agree, role_total):.0f}%)  {predicted}")
        else:
            print(f"  {role:10} 0/0  (n/a)")

    if misses:
        print("\nWhere it disagreed with you:")
        for unit_id, human_role, model_role in misses[:15]:
            print(f"  {unit_id}: you said {human_role}, model said {model_role}")
    print()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    blind = sub.add_parser("blind", help="write a blind sample for human marking")
    blind.add_argument("db")
    blind.add_argument("n", nargs="?", type=int, default=100)
    blind.add_argument("out", nargs="?", default="unit_role_sample.tsv")
    blind.add_argument("--client", required=True, help="client_id to sample from")
    blind.add_argument("--seed", default="unit-role-calibration")

    compare = sub.add_parser("compare", help="compare marked roles with withheld model roles")
    compare.add_argument("marked")

    args = parser.parse_args(argv)
    if args.cmd == "blind":
        cmd_blind(args.db, args.client, args.n, args.out, seed=args.seed)
    elif args.cmd == "compare":
        cmd_compare(args.marked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
