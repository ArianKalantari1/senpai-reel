"""
Phase 10 — 3-stage streaming pipeline.

Architecture:
  posts → [STT workers] ──callback──▶ [Extract workers] ──callback──▶ [Embed workers]
                                                                              │
                                                                    [DB Writer (sequential)]

Stages overlap: as soon as reel 1 finishes STT, extraction begins
while reel 2 is still being transcribed.  Only one DB writer thread
exists at all times — DuckDB cannot handle concurrent writes.

Usage:
    from pipeline.stream_runner import run_stream_pipeline, PipelineStats
    stats = run_stream_pipeline(
        post_rows,          # [(post_id, audio_path), ...]
        stt_provider,
        llm_provider,
        embed_provider,
        progress_callback=lambda done, total, post_id, err: ...,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.
    acquire() blocks until a slot is available.

    Args:
        max_per_minute: maximum requests allowed per 60-second window
    """

    def __init__(self, max_per_minute: int):
        self._interval = 60.0 / max(max_per_minute, 1)
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                wait = self._next_allowed - now
            time.sleep(min(wait, 0.05))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total: int = 0
    stt_done: int = 0
    stt_failed: int = 0
    extract_done: int = 0
    extract_failed: int = 0
    embed_done: int = 0
    embed_failed: int = 0
    total_cost_usd: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def fully_done(self) -> int:
        """Posts that completed the full pipeline (including embed)."""
        return self.embed_done + self.embed_failed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stream_pipeline(
    post_rows: List[Tuple[str, str]],
    stt_provider,
    llm_provider,
    embed_provider,
    ocr_provider=None,
    stt_rate_per_min: int = 20,
    llm_rate_per_min: int = 30,
    emb_rate_per_min: int = 180,
    max_workers: int = 8,
    progress_callback: Optional[Callable] = None,
) -> PipelineStats:
    """
    Run the 3-stage streaming pipeline over a list of posts.

    Args:
        post_rows:          list of (post_id, audio_path) tuples
        stt_provider:       AssemblyAIProvider (or any provider with .transcribe())
        llm_provider:       CerebrasProvider (or any with .extract())
        embed_provider:     VoyageProvider (or any with .embed_batch())
        ocr_provider:       optional GeminiOCRProvider; skipped if None
        stt_rate_per_min:   rate limit for STT stage (default: 20 = AssemblyAI free tier)
        llm_rate_per_min:   rate limit for LLM stage (default: 30 = Cerebras free tier)
        emb_rate_per_min:   rate limit for embed stage (default: 180 = Voyage ~3/sec)
        max_workers:        max threads per stage
        progress_callback:  callable(done, total, post_id, error_or_None)

    Returns:
        PipelineStats with per-stage counts, costs, and error list
    """
    stats = PipelineStats(total=len(post_rows))
    if not post_rows:
        return stats

    stt_limiter = RateLimiter(stt_rate_per_min)
    llm_limiter = RateLimiter(llm_rate_per_min)
    emb_limiter = RateLimiter(emb_rate_per_min)

    # Thread-safe accumulators for the DB write pass
    _lock              = threading.Lock()
    transcript_store   = []     # TranscriptResult
    units_store        = []     # MessageUnit (all posts combined)
    embed_store        = []     # (unit_id, embedding) pairs
    cost_map           = {}     # post_id → extraction cost_usd

    # Countdown latch: each post decrements once it exits the last stage
    _remaining = [len(post_rows)]
    _done_event = threading.Event()

    def _countdown(n: int = 1):
        with _lock:
            _remaining[0] -= n
            if _remaining[0] <= 0:
                _done_event.set()

    # --- Stage 3: Embed ---

    def _do_embed(units):
        emb_limiter.acquire()
        texts = [u.text for u in units]
        return embed_provider.embed_batch(texts)

    def _on_embed_done(future, units):
        try:
            embeddings = future.result()
            pairs = [(u.unit_id, emb) for u, emb in zip(units, embeddings)]
            with _lock:
                embed_store.extend(pairs)
                stats.embed_done += 1
        except Exception as exc:
            logger.warning("Embed stage failed: %s", exc)
            with _lock:
                stats.embed_failed += 1
                stats.errors.append({"stage": "embed", "error": str(exc)})
        finally:
            _countdown()

    # --- Stage 2: Extract ---

    def _do_extract(stt_result, ocr_text: str):
        llm_limiter.acquire()
        return llm_provider.extract(
            stt_result.transcript,
            stt_result.post_id,
            ocr_text=ocr_text,
        )

    def _on_extract_done(future, post_id, executor):
        try:
            units, cost = future.result()
            with _lock:
                units_store.extend(units)
                cost_map[post_id] = cost
                stats.extract_done += 1
                stats.total_cost_usd = round(stats.total_cost_usd + cost, 6)

            if units:
                ef = executor.submit(_do_embed, units)
                ef.add_done_callback(lambda f: _on_embed_done(f, units))
                return  # countdown happens inside _on_embed_done
        except Exception as exc:
            logger.warning("Extract stage failed for %s: %s", post_id, exc)
            with _lock:
                stats.extract_failed += 1
                stats.errors.append({"stage": "extract", "post_id": post_id, "error": str(exc)})

        _countdown()  # no embed submitted — count down here

    # --- Stage 1: STT (+ optional OCR) ---

    def _do_stt(post_id, audio_path):
        stt_limiter.acquire()
        return stt_provider.transcribe(audio_path, post_id)

    def _do_ocr_for_post(post_id) -> str:
        """Sample frames and extract OCR text if ocr_provider is configured."""
        if ocr_provider is None:
            return ""
        try:
            from processing.frames import sample_frames
            frame_paths = sample_frames(post_id, n_frames=4)
            texts = [ocr_provider.extract(fp, post_id) for fp in frame_paths]
            return " ".join(t for t in texts if t).strip()
        except Exception as exc:
            logger.debug("OCR skipped for %s: %s", post_id, exc)
            return ""

    def _on_stt_done(future, post_id, executor):
        try:
            stt_result = future.result()
            with _lock:
                transcript_store.append(stt_result)
                stats.stt_done += 1

            if progress_callback:
                try:
                    progress_callback(stats.stt_done, stats.total, post_id, None)
                except Exception:
                    pass

            # Fetch OCR text (non-blocking — returns "" quickly if no frames exist)
            ocr_text = _do_ocr_for_post(post_id)

            ef = executor.submit(_do_extract, stt_result, ocr_text)
            ef.add_done_callback(lambda f: _on_extract_done(f, post_id, executor))
        except Exception as exc:
            logger.warning("STT stage failed for %s: %s", post_id, exc)
            with _lock:
                stats.stt_failed += 1
                stats.errors.append({"stage": "stt", "post_id": post_id, "error": str(exc)})

            if progress_callback:
                try:
                    progress_callback(stats.stt_done, stats.total, post_id, str(exc))
                except Exception:
                    pass

            _countdown()  # post exits pipeline at STT failure

    # --- Launch ---

    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline")
    try:
        for post_id, audio_path in post_rows:
            f = executor.submit(_do_stt, post_id, audio_path)
            f.add_done_callback(
                lambda fut, pid=post_id, path=audio_path: _on_stt_done(fut, pid, executor)
            )

        # Block until every post has exited the last stage (or failed)
        _done_event.wait(timeout=7200)   # 2-hour hard timeout

    finally:
        executor.shutdown(wait=True)

    # --- Sequential DB writes ---
    _write_results(transcript_store, units_store, embed_store, cost_map)

    return stats


# ---------------------------------------------------------------------------
# DB write (single writer — no concurrent DuckDB access)
# ---------------------------------------------------------------------------

def _write_results(transcript_store, units_store, embed_store, cost_map):
    from processing.transcribe import save_transcript
    from analysis.extraction import save_message_units
    from core.db import get_connection

    # 1. Transcripts
    for result in transcript_store:
        try:
            save_transcript(result)
        except Exception as exc:
            logger.warning("DB write failed for transcript %s: %s", result.post_id, exc)

    # 2. Message units
    if units_store:
        try:
            save_message_units(units_store)
        except Exception as exc:
            logger.warning("DB write failed for message units: %s", exc)

    # 3. Embeddings
    if embed_store:
        try:
            conn = get_connection()
            for unit_id, embedding in embed_store:
                conn.execute(
                    "UPDATE message_units "
                    "SET embedding = ?, embedded_at = CURRENT_TIMESTAMP "
                    "WHERE unit_id = ?",
                    [embedding, unit_id],
                )
            conn.close()
        except Exception as exc:
            logger.warning("DB write failed for embeddings: %s", exc)

    # 4. Extraction costs (best-effort)
    if cost_map:
        try:
            conn = get_connection()
            for post_id, cost in cost_map.items():
                try:
                    conn.execute(
                        "UPDATE transcripts SET extraction_cost_usd = ? WHERE post_id = ?",
                        [cost, post_id],
                    )
                except Exception:
                    pass
            conn.close()
        except Exception as exc:
            logger.warning("DB write failed for extraction costs: %s", exc)
