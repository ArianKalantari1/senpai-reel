#!/usr/bin/env python3
"""Score extracted message units mechanically.

This tool reports two separate source-closeness checks:
  - claim vs text snippet: `text` and `claim` are both extractor outputs, so
    COPIED means "the claim restates its own snippet".
  - claim/text vs transcript: `transcripts.transcript` is joined by post_id, so
    LIFTED means "the claim reused a source-video phrase".

Every check here is mechanical — no API key, no spend, no LLM judging an LLM.
That is a deliberate limit, not an oversight: these signals catch structural
failures (copying, inversion, invented content, filler) and cannot judge whether
a claim is *true*. Only a reader can do that, which is what `blind` is for.

KNOWN GAPS:
  - the agreement figure from `compare` is only as good as the sample you mark

    score_extraction.py score <db>              measure the corpus
    score_extraction.py blind <db> [n] [out] [--run-id NAME]
                                                write n units for human marking
    score_extraction.py compare <marked.tsv> [db]  agreement between you and it

  --paraphrase-threshold <0..1> retunes the PARAPHRASE cut-off for `score` and
  `compare` without editing code. See DEFAULT_PARAPHRASE_THRESHOLD for how the
  shipped value was calibrated and what a larger marked sample should change.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

TOKEN = re.compile(r"[a-z0-9']+")

# Words whose presence flips meaning. An inversion usually shows up as one of
# these appearing on one side of the pair and not the other.
NEGATION = {
    "not", "no", "never", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "wont", "won't", "isnt", "isn't", "arent", "aren't",
    "shouldnt", "shouldn't", "avoid", "stop", "without", "rarely", "unnecessary",
}

# Filler that lets a claim say nothing while sounding like something.
HEDGE = {
    "demonstrates", "showcases", "highlights", "emphasizes", "underscores",
    "important", "essential", "crucial", "key", "valuable", "effective",
    "helps", "enables", "allows", "ensures", "reflects", "indicates",
}

STOPish = {
    "the","a","an","and","or","but","if","to","of","in","on","for","with","your",
    "you","it","is","are","be","this","that","at","as","was","were","from","by",
    "can","will","their","them","they","he","she","his","her","its","have","has",
    "had","do","does","did","so","than","then","when","what","which","who","how",
    "about","into","out","up","down","more","most","some","any","all","just",
}


def toks(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


# Suffix stripping, deliberately crude. It only has to collapse morphological
# variants of the same word ("lived"/"living", "failed"/"failure"), not be
# linguistically correct. Longest suffixes first so "ations" beats "s".
_SUFFIXES = (
    "ational", "ization", "iveness", "fulness", "ousness", "ations", "ement",
    "ments", "ingly", "edly", "ance", "ence", "ible", "able", "ings", "ment",
    "ness", "tion", "sion", "ies", "ing", "ers", "est", "ed", "es", "ly",
    "er", "s", "y",
)


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# CALIBRATED 2026-09-13 against 12 hand-marked pairs — 7 judged "lazy" by a
# careful human read, 5 judged good abstractions. The old threshold of 0.60 on
# raw tokens caught ZERO of the seven, which is why `compare` agreed with a
# human only 57% of the time: the tool could see lexical laziness but not
# semantic laziness, and a claim that restates its source in different words
# scored clean.
#
#   threshold  stemmed   caught      false positives
#       0.60      no      0/7             0/5     <- what shipped
#       0.40      no      2/7             0/5
#       0.40     yes      3/7             0/5
#       0.30     yes      4/7             0/5     <- chosen
#       0.25     yes      6/7             0/5
#
# 0.25 catches more and still showed no false positives, but the highest-scoring
# good abstraction in the control sits at 0.17 — an 0.08 margin on a 12-item
# set. 0.30 leaves 0.13. While the calibration set is this small, under-flagging
# is the safer error: a false PARAPHRASE on a good unit costs trust in a tool
# whose whole purpose is being trustworthy about itself.
#
# Re-run the sweep when a larger marked sample exists. --paraphrase-threshold
# exists so that does not require a code change.
DEFAULT_PARAPHRASE_THRESHOLD = 0.30


def ngrams(t: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)} if len(t) >= n else set()


def content_toks(text: str) -> list[str]:
    return [w for w in toks(text) if w not in STOPish]


def shared_content_ngrams(left: str, right: str, n: int = 4) -> set[tuple[str, ...]]:
    """Shared n-grams after filler words are removed."""
    left_content = content_toks(left)
    if len(left_content) < n:
        return set()
    right_content = content_toks(right)
    if len(right_content) < n:
        return set()
    return ngrams(left_content, n) & ngrams(right_content, n)


def score_pair(claim: str, text: str,
               paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD) -> dict:
    """Return structural signals for one (claim, source) pair."""
    c, t = toks(claim), toks(text)
    if not c or not t:
        return {"flags": ["EMPTY"], "overlap": None, "novel": None}

    cset, tset = set(c), set(t)
    flags = []

    # 1. Copying. A shared 4-gram of non-filler words is borrowed phrasing,
    #    not abstraction. This is the firewall the project is built on.
    shared4 = {g for g in (ngrams(c, 4) & ngrams(t, 4)) if not all(w in STOPish for w in g)}
    if shared4:
        flags.append("COPIED")

    # 2. Near-paraphrase. High content-word overlap means it restated rather
    #    than generalised. Weaker than COPIED but still low value.
    content_c = {w for w in cset if w not in STOPish}
    content_t = {w for w in tset if w not in STOPish}
    # Stemmed, so "not because you failed, but because you lived" registers
    # against "come from living fully, not from failure". Raw tokens miss it.
    stem_c = {stem(w) for w in content_c}
    stem_t = {stem(w) for w in content_t}
    overlap = len(stem_c & stem_t) / len(stem_c) if stem_c else 0.0
    raw_overlap = len(content_c & content_t) / len(content_c) if content_c else 0.0
    if not shared4 and overlap >= paraphrase_threshold:
        flags.append("PARAPHRASE")

    # 3. Possible inversion. Negation on one side only flips the meaning.
    #    Crude — "avoid X" vs "X is unnecessary" agree while tripping this —
    #    so it is a flag for a human to check, never a verdict.
    if bool(cset & NEGATION) != bool(tset & NEGATION):
        flags.append("NEGATION_MISMATCH")

    # 4. Novel content. Content words in the claim absent from the source.
    #    MEASURED LIMITATION: this flag does NOT separate invention from good
    #    abstraction, because both introduce new words. Tested against eight
    #    hand-judged real pairs, it caught both genuine hallucinations and
    #    also fired on a correct abstraction ("I got a 20 cent raise" -> "high
    #    performance doesn't always lead to meaningful pay increases").
    #    Treat it as "a human should look at this", never as a verdict.
    novel = {w for w in content_c if w not in content_t and len(w) > 4}
    novel = {w for w in novel if not any(w.startswith(s[:5]) for s in content_t if len(s) > 4)}
    if len(novel) >= 5:
        flags.append("NOVEL_CONTENT")

    # 5. Filler. Hedge words with little borrowed substance is a platitude.
    if (cset & HEDGE) and overlap < 0.3:
        flags.append("VAGUE")

    return {
        "flags": flags or ["OK"],
        "overlap": round(overlap, 2),
        "raw_overlap": round(raw_overlap, 2),
        "novel": sorted(novel)[:6],
    }


def score_transcript(claim: str, text: str, transcript: str | None) -> dict:
    """Return transcript-comparison signals without judging truth."""
    if not transcript:
        return {
            "has_transcript": False,
            "flags": [],
            "claim_lifted": None,
            "text_matches_transcript": None,
        }

    claim_lifted = bool(shared_content_ngrams(claim, transcript))
    text_matches_transcript = bool(shared_content_ngrams(text, transcript))
    return {
        "has_transcript": True,
        "flags": ["LIFTED"] if claim_lifted else ["OK"],
        "claim_lifted": claim_lifted,
        "text_matches_transcript": text_matches_transcript,
    }


def transcript_excerpt(transcript: str | None, claim: str = "", text: str = "", limit: int = 700) -> str:
    """Short display excerpt, anchored near the snippet/claim when possible."""
    if not transcript:
        return "[missing transcript]"

    compact = " ".join(str(transcript).split())
    if len(compact) <= limit:
        return compact

    lower = compact.lower()
    anchors = [w for w in content_toks(text) + content_toks(claim) if len(w) > 4]
    positions = [lower.find(anchor) for anchor in anchors if lower.find(anchor) >= 0]
    pos = min(positions) if positions else 0

    start = max(0, pos - limit // 3)
    end = min(len(compact), start + limit)
    start = max(0, end - limit)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def tsv_cell(value) -> str:
    return " ".join(str(value or "").split()).replace("\t", " ")


def load(db_path: str, run_id: str | None = None):
    import duckdb
    conn = duckdb.connect(db_path, read_only=True)
    try:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='message_units'").fetchall()]
        if "claim" not in cols or "text" not in cols:
            raise SystemExit("message_units needs both `claim` and `text` columns")
        if run_id and "extraction_run_id" not in cols:
            raise SystemExit(
                "\n  This database predates extraction_run_id, so `blind --run-id` "
                "cannot target a run.\n\n"
                "  Run the additive migration once:\n\n"
                f"      python -c \"import core.db as db; db.DB_PATH='{db_path}'; db.init_db()\"\n"
            )
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        has_transcripts = "transcripts" in tables
        sel = (
            "mu.unit_id, mu.claim, mu.text, t.transcript"
            + (", mu.topic" if "topic" in cols else ", NULL")
        )
        join = "LEFT JOIN transcripts t ON t.post_id = mu.post_id" if has_transcripts else ""
        if not has_transcripts:
            sel = sel.replace("t.transcript", "NULL")
        where = ["mu.claim IS NOT NULL", "mu.text IS NOT NULL"]
        params = []
        if run_id:
            where.append("mu.extraction_run_id = ?")
            params.append(run_id)
        return conn.execute(
            f"SELECT {sel} FROM message_units mu "
            f"{join} "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY mu.unit_id",
            params,
        ).fetchall()
    finally:
        conn.close()


def load_transcripts_by_unit(db_path: str, unit_ids: list[str]) -> dict[str, str | None]:
    if not unit_ids:
        return {}

    import duckdb
    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        if "transcripts" not in tables:
            return {unit_id: None for unit_id in unit_ids}
        placeholders = ", ".join(["?"] * len(unit_ids))
        rows = conn.execute(
            f"""
            SELECT mu.unit_id, t.transcript
            FROM message_units mu
            LEFT JOIN transcripts t ON t.post_id = mu.post_id
            WHERE mu.unit_id IN ({placeholders})
            """,
            unit_ids,
        ).fetchall()
        found = {unit_id: transcript for unit_id, transcript in rows}
        return {unit_id: found.get(unit_id) for unit_id in unit_ids}
    finally:
        conn.close()


def cmd_score(db_path: str,
              paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD):
    rows = load(db_path)
    tally, transcript_tally, overlaps = Counter(), Counter(), []
    covered = text_matches_transcript = 0
    for _uid, claim, text, transcript, _topic in rows:
        r = score_pair(claim, text, paraphrase_threshold)
        for f in r["flags"]:
            tally[f] += 1
        if r["overlap"] is not None:
            overlaps.append(r["overlap"])

    n = len(rows)
    print(f"\nExtraction quality — {n:,} units\n")
    if n == 0:
        print("  No scored units found.\n")
        return

    for _uid, claim, text, transcript, _topic in rows:
        tr = score_transcript(claim, text, transcript)
        if tr["has_transcript"]:
            covered += 1
            for f in tr["flags"]:
                transcript_tally[f] += 1
            if tr["text_matches_transcript"]:
                text_matches_transcript += 1

    print("  Structural signals (a unit can raise more than one):")
    for flag, count in tally.most_common():
        print(f"    {flag:20} {count:>7,}  {count / n * 100:5.1f}%")
    if overlaps:
        overlaps.sort()
        print(f"\n  Content-word overlap, claim vs source:")
        print(f"    median {overlaps[len(overlaps)//2]:.2f} | "
              f"p90 {overlaps[int(len(overlaps)*0.9)]:.2f}")

    print("\n  Transcript comparison:")
    print(f"    transcript coverage {covered:,}/{n:,}  ({covered / n * 100:5.1f}%)")
    if covered:
        lifted = transcript_tally["LIFTED"]
        print(f"    LIFTED              {lifted:>7,}/{covered:<7,}  {lifted / covered * 100:5.1f}%")
        print(
            f"    text matches source {text_matches_transcript:>7,}/{covered:<7,}  "
            f"{text_matches_transcript / covered * 100:5.1f}%"
        )
    else:
        print("    LIFTED                    —  no transcript rows")
        print("    text matches source       —  no transcript rows")
    print("\n  RELIABLE for lazy/restated claims: COPIED, PARAPHRASE, VAGUE, LIFTED.")
    print("  LIFTED is reliable for laziness, not wrongness: it mechanically")
    print("  detects source wording in the abstraction when a transcript exists.")
    print("  ADVISORY only: NOVEL_CONTENT, NEGATION_MISMATCH — these fire on")
    print("  good abstractions too, because abstracting introduces new words.")
    print("  Nothing here judges whether a claim is TRUE. Run `blind` for that.\n")


def cmd_blind(db_path: str, n: int, out: str, run_id: str | None = None):
    import random
    rows = load(db_path, run_id=run_id)
    random.seed(7)  # reproducible, so the scorer cannot be tuned to the sample
    sample = random.sample(rows, min(n, len(rows)))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Mark each line's VERDICT column: good / wrong / lazy\n")
        fh.write("#   good  = faithful and adds something over the source\n")
        fh.write("#   wrong = contradicts, invents, or misreads the source\n")
        fh.write("#   lazy  = accurate but just restates the source\n")
        fh.write("# Do not change any other column. Save when done.\n")
        fh.write("unit_id\tVERDICT\tclaim\tsource\ttranscript_excerpt\n")
        for uid, claim, text, transcript, _t in sample:
            c = tsv_cell(claim)
            s = tsv_cell(text)
            te = tsv_cell(transcript_excerpt(transcript, claim, text))
            fh.write(f"{uid}\t?\t{c}\t{s}\t{te}\n")
    print(f"\nWrote {len(sample)} units to {out}")
    if run_id:
        print(f"Sample restricted to extraction_run_id={run_id!r}.")
    print("No verdicts included — the scorer's opinion is withheld on purpose,")
    print("so your reading is not anchored by it.")
    print(f"\nMark the VERDICT column, then: score_extraction.py compare {out} {db_path}\n")


def cmd_compare(marked: str, db_path: str | None = None,
                paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD):
    rows = []
    with open(marked, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("unit_id\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            uid, verdict, claim, text = parts[0], parts[1].strip().lower(), parts[2], parts[3]
            transcript = parts[4] if len(parts) > 4 else None
            if verdict in ("good", "wrong", "lazy"):
                rows.append((uid, verdict, claim, text, transcript))

    if not rows:
        raise SystemExit("No marked rows found. Fill in the VERDICT column with good/wrong/lazy.")

    transcript_by_unit = load_transcripts_by_unit(db_path, [r[0] for r in rows]) if db_path else {}
    if not db_path and any(r[4] for r in rows):
        # Without the database, the only transcript available is the truncated
        # excerpt in the TSV. transcript_excerpt() anchors its window on the
        # EARLIEST matching claim word, so a phrase lifted from later in the
        # transcript falls outside it and LIFTED silently does not fire. The
        # agreement number then comes out lower and the tool blames itself —
        # a wrong number delivered with a confident verdict attached.
        print("\n  WARNING: no database path given, so LIFTED is being scored against the")
        print("  truncated excerpt in the file, not the full transcript. It will be")
        print("  under-reported and the agreement figure below is not trustworthy.")
        print(f"  Re-run as: score_extraction.py compare <marked.tsv> <db>\n")
    agree = disagree = 0
    misses = []
    for uid, human, claim, text, transcript in rows:
        transcript = transcript_by_unit.get(uid, transcript)
        if transcript == "[missing transcript]":
            transcript = None
        flags = set(score_pair(claim, text, paraphrase_threshold)["flags"])
        flags |= set(score_transcript(claim, text, transcript)["flags"]) - {"OK"}
        # COPIED, PARAPHRASE and VAGUE proved reliable on the hand-judged set.
        # LIFTED joins them for "lazy", not "wrong": exact reuse of transcript
        # phrasing is source-closeness evidence, not evidence of falsity.
        # NOVEL_CONTENT and NEGATION_MISMATCH proved advisory only, so a "good"
        # unit is not counted as a disagreement merely for raising one of those.
        reliable = flags & {"COPIED", "PARAPHRASE", "VAGUE", "LIFTED"}
        advisory = flags & {"NOVEL_CONTENT", "NEGATION_MISMATCH"}
        if human == "good":
            machine_ok = not reliable
        elif human == "lazy":
            # "lazy" means accurate but merely restated — which is exactly what
            # the reliable flags detect.
            machine_ok = bool(reliable)
        else:  # wrong
            # Only the advisory flags point at "contradicts / invents / misreads".
            # COPIED and PARAPHRASE mean the opposite — the claim stuck close to
            # its source — so counting them as agreement here would let the tool
            # score a hit for saying something the human did not say. That is how
            # a calibration number flatters itself, and this tool is worthless if
            # its own honesty check is lenient in its own favour.
            machine_ok = bool(advisory)
        if machine_ok:
            agree += 1
        else:
            disagree += 1
            misses.append((uid, human, sorted(flags), claim[:70]))

    total = agree + disagree
    pct = agree / total * 100
    print(f"\nAgreement with your reading: {agree}/{total}  ({pct:.0f}%)\n")
    if misses:
        print("  Where it disagreed with you:")
        for uid, human, flags, claim in misses[:15]:
            print(f"    you said {human:5}  it said {','.join(flags):28} {claim}")
    print()
    if pct >= 85:
        print("  >=85%: the scorer tracks your judgement. Its corpus number means something.")
    elif pct >= 70:
        print("  70-85%: partially useful. Treat its number as a rough signal, not a measure.")
    else:
        print("  <70%: the scorer does NOT track your judgement. Do not trust its number.")
        print("  The disagreements above are the spec for fixing it.")
    print()


_THRESHOLD_FLAG = "--paraphrase-threshold"
_RUN_ID_FLAG = "--run-id"


def take_threshold(argv: list[str]) -> tuple[list[str], float | None]:
    """Pull --paraphrase-threshold out of argv, returning the rest unchanged.

    The commands take their arguments positionally, so the flag has to be
    removed before dispatch rather than parsed alongside — otherwise passing it
    shifts <db> into the <n> slot and the failure looks like a bad database
    path. Accepts both "--paraphrase-threshold 0.25" and "=0.25".
    """
    rest: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == _THRESHOLD_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{_THRESHOLD_FLAG} needs a value between 0 and 1.")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(_THRESHOLD_FLAG + "="):
            value = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1

    if value is None:
        return rest, None
    try:
        threshold = float(value)
    except ValueError:
        raise SystemExit(f"{_THRESHOLD_FLAG}: {value!r} is not a number.")
    # 0 flags every unit that has no shared 4-gram, which is silently useless
    # rather than loudly wrong — the worst way for a measuring tool to fail.
    # 1.0 is strict but meaningful (total overlap), so it stays allowed.
    if not 0 < threshold <= 1:
        raise SystemExit(
            f"{_THRESHOLD_FLAG}: {threshold} is outside 0-1. "
            f"The calibrated default is {DEFAULT_PARAPHRASE_THRESHOLD}."
        )
    return rest, threshold


def take_run_id(argv: list[str]) -> tuple[list[str], str | None]:
    """Pull --run-id out of argv, returning the rest unchanged."""
    rest: list[str] = []
    run_id: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == _RUN_ID_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{_RUN_ID_FLAG} needs a value.")
            run_id = argv[i + 1].strip()
            i += 2
            continue
        if arg.startswith(_RUN_ID_FLAG + "="):
            run_id = arg.split("=", 1)[1].strip()
            i += 1
            continue
        rest.append(arg)
        i += 1
    if run_id == "":
        raise SystemExit(f"{_RUN_ID_FLAG} needs a non-empty value.")
    return rest, run_id


if __name__ == "__main__":
    a, override = take_threshold(sys.argv[1:])
    a, run_id = take_run_id(a)
    threshold = DEFAULT_PARAPHRASE_THRESHOLD if override is None else override
    if len(a) >= 2 and a[0] == "score":
        if run_id is not None:
            raise SystemExit(f"{_RUN_ID_FLAG} only applies to `blind`.")
        cmd_score(a[1], threshold)
    elif len(a) >= 2 and a[0] == "blind":
        if override is not None:
            # `blind` withholds every scorer opinion on purpose. Accepting the
            # flag here and ignoring it would let someone believe they had
            # changed a sample they had not.
            raise SystemExit(
                f"{_THRESHOLD_FLAG} does not apply to `blind` — it writes units "
                "for you to mark and never scores them. Pass it to `compare`."
            )
        cmd_blind(
            a[1],
            int(a[2]) if len(a) > 2 else 100,
            a[3] if len(a) > 3 else "extraction_sample.tsv",
            run_id=run_id,
        )
    elif len(a) >= 2 and a[0] == "compare":
        if run_id is not None:
            raise SystemExit(f"{_RUN_ID_FLAG} only applies to `blind`.")
        cmd_compare(a[1], a[2] if len(a) > 2 else None, threshold)
    else:
        print(__doc__)
        sys.exit(2)
