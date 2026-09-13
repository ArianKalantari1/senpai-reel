#!/usr/bin/env python3
"""Score a unit's `claim` against the `text` snippet it was abstracted from.

WHAT IS ACTUALLY COMPARED, because it is easy to assume otherwise: both sides
are the extractor's own output. `text` is the near-verbatim snippet the model
pulled from the transcript; `claim` is the model's one-sentence restatement of
it. This tool does NOT read the transcript, so COPIED here means "the claim
restates its own snippet", not "the unit was lifted from the source video".
The transcript is reachable (`transcripts.transcript`, joined on `post_id`) and
comparing against it would be the stronger check — see KNOWN GAPS below.

Every check here is mechanical — no API key, no spend, no LLM judging an LLM.
That is a deliberate limit, not an oversight: these signals catch structural
failures (copying, inversion, invented content, filler) and cannot judge whether
a claim is *true*. Only a reader can do that, which is what `blind` is for.

KNOWN GAPS:
  - does not compare against the transcript (see above)
  - the agreement figure from `compare` is only as good as the sample you mark

    score_extraction.py score <db>              measure the corpus
    score_extraction.py blind <db> [n] [out]    write n units for human marking
    score_extraction.py compare <marked.tsv>    agreement between you and it
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


def ngrams(t: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)} if len(t) >= n else set()


def score_pair(claim: str, text: str) -> dict:
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
    overlap = len(content_c & content_t) / len(content_c) if content_c else 0.0
    if not shared4 and overlap >= 0.6:
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
        "novel": sorted(novel)[:6],
    }


def load(db_path: str):
    import duckdb
    conn = duckdb.connect(db_path, read_only=True)
    try:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='message_units'").fetchall()]
        if "claim" not in cols or "text" not in cols:
            raise SystemExit("message_units needs both `claim` and `text` columns")
        sel = "unit_id, claim, text" + (", topic" if "topic" in cols else ", NULL")
        return conn.execute(
            f"SELECT {sel} FROM message_units "
            "WHERE claim IS NOT NULL AND text IS NOT NULL").fetchall()
    finally:
        conn.close()


def cmd_score(db_path: str):
    rows = load(db_path)
    tally, overlaps = Counter(), []
    for _uid, claim, text, _topic in rows:
        r = score_pair(claim, text)
        for f in r["flags"]:
            tally[f] += 1
        if r["overlap"] is not None:
            overlaps.append(r["overlap"])

    n = len(rows)
    print(f"\nExtraction quality — {n:,} units\n")
    print("  Structural signals (a unit can raise more than one):")
    for flag, count in tally.most_common():
        print(f"    {flag:20} {count:>7,}  {count / n * 100:5.1f}%")
    if overlaps:
        overlaps.sort()
        print(f"\n  Content-word overlap, claim vs source:")
        print(f"    median {overlaps[len(overlaps)//2]:.2f} | "
              f"p90 {overlaps[int(len(overlaps)*0.9)]:.2f}")
    print("\n  RELIABLE on a hand-judged set: COPIED, PARAPHRASE, VAGUE.")
    print("  ADVISORY only: NOVEL_CONTENT, NEGATION_MISMATCH — these fire on")
    print("  good abstractions too, because abstracting introduces new words.")
    print("  Nothing here judges whether a claim is TRUE. Run `blind` for that.\n")


def cmd_blind(db_path: str, n: int, out: str):
    import random
    rows = load(db_path)
    random.seed(7)  # reproducible, so the scorer cannot be tuned to the sample
    sample = random.sample(rows, min(n, len(rows)))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Mark each line's VERDICT column: good / wrong / lazy\n")
        fh.write("#   good  = faithful and adds something over the source\n")
        fh.write("#   wrong = contradicts, invents, or misreads the source\n")
        fh.write("#   lazy  = accurate but just restates the source\n")
        fh.write("# Do not change any other column. Save when done.\n")
        fh.write("unit_id\tVERDICT\tclaim\tsource\n")
        for uid, claim, text, _t in sample:
            c = " ".join(str(claim).split())
            s = " ".join(str(text).split())
            fh.write(f"{uid}\t?\t{c}\t{s}\n")
    print(f"\nWrote {len(sample)} units to {out}")
    print("No verdicts included — the scorer's opinion is withheld on purpose,")
    print("so your reading is not anchored by it.")
    print(f"\nMark the VERDICT column, then: score_extraction.py compare {out}\n")


def cmd_compare(marked: str):
    rows = []
    with open(marked, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("unit_id\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            uid, verdict, claim, text = parts[0], parts[1].strip().lower(), parts[2], parts[3]
            if verdict in ("good", "wrong", "lazy"):
                rows.append((uid, verdict, claim, text))

    if not rows:
        raise SystemExit("No marked rows found. Fill in the VERDICT column with good/wrong/lazy.")

    agree = disagree = 0
    misses = []
    for uid, human, claim, text in rows:
        flags = set(score_pair(claim, text)["flags"])
        # COPIED, PARAPHRASE and VAGUE proved reliable on the hand-judged set.
        # NOVEL_CONTENT and NEGATION_MISMATCH proved advisory only, so a "good"
        # unit is not counted as a disagreement merely for raising one of those.
        reliable = flags & {"COPIED", "PARAPHRASE", "VAGUE"}
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


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "score":
        cmd_score(a[1])
    elif len(a) >= 2 and a[0] == "blind":
        cmd_blind(a[1], int(a[2]) if len(a) > 2 else 100, a[3] if len(a) > 3 else "extraction_sample.tsv")
    elif len(a) >= 2 and a[0] == "compare":
        cmd_compare(a[1])
    else:
        print(__doc__)
        sys.exit(2)
