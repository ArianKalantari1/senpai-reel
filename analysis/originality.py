"""Originality gate for generated content.

Reference units are competitor research, not source material. This module
checks that generated output does not reproduce wording from the references
it was grounded in.

See creative-director-ai issue #17 and the strategy-abstraction firewall in
`research/2026-09-marketing-intelligence-creative-os-v2.md` section 5.
"""
from __future__ import annotations

import re

DEFAULT_N = 5

# Words too common for a shared span to be meaningful. An n-gram made only of
# these is idiom, not copying.
_FILLER = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for",
    "with", "your", "you", "it", "is", "are", "be", "this", "that", "at", "as",
    "do", "not", "can", "will", "how", "what", "why", "when", "get", "make",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _meaningful(gram: tuple[str, ...]) -> bool:
    return not all(tok in _FILLER for tok in gram)


def reference_strings(units: list) -> list[str]:
    """Every piece of competitor-derived text a unit carries.

    Checks `text` as well as `claim`: `text` holds verbatim transcript
    wording, so even though it no longer reaches the prompt it is exactly
    what output must not match.
    """
    out = []
    for u in units or []:
        for attr in ("text", "claim"):
            val = (getattr(u, attr, "") or "").strip()
            if val:
                out.append(val)
    return out


def find_overlap(output: str, units: list, n: int = DEFAULT_N) -> list[str]:
    """Return meaningful n-grams shared by `output` and any reference unit."""
    out_grams = _ngrams(_tokens(output), n)
    if not out_grams:
        return []
    shared: set[tuple[str, ...]] = set()
    for ref in reference_strings(units):
        shared |= out_grams & _ngrams(_tokens(ref), n)
    return sorted(" ".join(g) for g in shared if _meaningful(g))


class ReferenceLeakError(RuntimeError):
    """Generated output reproduced wording from its reference units."""

    def __init__(self, phrases: list[str]):
        self.phrases = phrases
        preview = "; ".join(f'"{p}"' for p in phrases[:3])
        super().__init__(
            f"Generated content reused wording from competitor references ({preview}). "
            "Try different reference units, or rephrase the angle."
        )
