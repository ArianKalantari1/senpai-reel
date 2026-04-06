"""
Phase 4 — Extraction queue.

Processes all transcripts that don't have message_units extracted yet.

Phase 10 update: when called without an explicit openai_api_key (or when
providers.factory is configured), delegates to CerebrasProvider via the
streaming pipeline helpers.  The function signature is unchanged for
backwards compatibility.
"""

import logging
from typing import Callable, Optional

from core.db import get_connection
from analysis.extraction import extract_message_units, save_message_units

logger = logging.getLogger(__name__)


def get_extraction_stats() -> dict:
    conn = get_connection()
    try:
        total_transcripts = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        extracted_posts = conn.execute(
            "SELECT COUNT(DISTINCT post_id) FROM message_units"
        ).fetchone()[0]
        total_units = conn.execute("SELECT COUNT(*) FROM message_units").fetchone()[0]
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(extraction_cost_usd), 0) FROM transcripts"
        ).fetchone()[0]
        return {
            "total_transcripts": total_transcripts,
            "extracted_posts": extracted_posts,
            "pending": max(total_transcripts - extracted_posts, 0),
            "total_units": total_units,
            "total_cost_usd": round(total_cost, 4),
        }
    except Exception:
        return {"total_transcripts": 0, "extracted_posts": 0, "pending": 0,
                "total_units": 0, "total_cost_usd": 0.0}
    finally:
        conn.close()


def run_extraction_queue(
    openai_api_key: str = "",
    batch_size: int = 10,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Extract message units from transcripts not yet processed.

    When openai_api_key is empty (Phase 10 default), CerebrasProvider is used
    via providers.factory.  Pass a non-empty key to force the legacy OpenAI path.

    Returns:
        dict with done, failed, total, total_units_extracted, total_cost_usd
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT t.post_id, t.transcript
        FROM transcripts t
        WHERE NOT EXISTS (
            SELECT 1 FROM message_units mu WHERE mu.post_id = t.post_id
        )
        LIMIT ?
        """,
        [batch_size],
    ).fetchall()
    conn.close()

    total = len(rows)
    done = failed = 0
    total_units = 0
    total_cost = 0.0

    # --- Phase 10: use Cerebras when no OpenAI key provided ---
    if not openai_api_key:
        from providers.factory import get_llm, get_embeddings
        from pipeline.stream_runner import run_stream_pipeline

        # We only have transcripts here (no audio paths), so use a lightweight
        # extraction-only adapter that wraps TranscriptResult-like objects.
        from processing.transcribe import TranscriptResult

        pseudo_results = [
            TranscriptResult(
                post_id=post_id,
                provider="cached",
                model="cached",
                transcript=transcript,
                language="en",
                confidence=1.0,
                duration_sec=0.0,
                words=[],
                cost_usd=0.0,
                raw_response={},
            )
            for post_id, transcript in rows
        ]

        llm = get_llm()
        emb = get_embeddings()

        stats = _run_extract_embed_only(pseudo_results, llm, emb, progress_callback)
        return {
            "done": stats["done"],
            "failed": stats["failed"],
            "total": total,
            "total_units_extracted": stats["units"],
            "total_cost_usd": round(stats["cost"], 6),
        }

    # --- Legacy: OpenAI path ---
    for i, (post_id, transcript) in enumerate(rows):
        try:
            units, cost = extract_message_units(transcript, post_id, openai_api_key)
            save_message_units(units)
            _update_extraction_cost(post_id, cost)
            done += 1
            total_units += len(units)
            total_cost += cost
            if progress_callback:
                progress_callback(i + 1, total, post_id, None, len(units))
        except Exception as e:
            failed += 1
            logger.warning("Extraction failed for %s: %s", post_id, e)
            if progress_callback:
                progress_callback(i + 1, total, post_id, str(e), 0)

    return {
        "done": done,
        "failed": failed,
        "total": total,
        "total_units_extracted": total_units,
        "total_cost_usd": round(total_cost, 6),
    }


def _run_extract_embed_only(pseudo_results, llm_provider, embed_provider, progress_callback) -> dict:
    """
    Run extract + embed stages only (no STT) for posts that are already transcribed.
    Used by extraction_queue when transcripts already exist in DB.
    """
    import threading
    from pipeline.stream_runner import RateLimiter

    llm_limiter = RateLimiter(30)
    emb_limiter = RateLimiter(180)

    total = len(pseudo_results)
    done = failed = units_total = 0
    cost_total = 0.0
    all_units = []
    embed_pairs = []
    _lock = threading.Lock()

    def process_one(result, idx):
        nonlocal done, failed, units_total, cost_total
        try:
            llm_limiter.acquire()
            units, cost = llm_provider.extract(result.transcript, result.post_id)
            with _lock:
                all_units.extend(units)
                cost_total += cost

            if units:
                emb_limiter.acquire()
                texts = [u.text for u in units]
                embeddings = embed_provider.embed_batch(texts)
                with _lock:
                    embed_pairs.extend(zip([u.unit_id for u in units], embeddings))

            with _lock:
                done += 1
                units_total += len(units)

            if progress_callback:
                progress_callback(idx + 1, total, result.post_id, None, len(units))
        except Exception as e:
            with _lock:
                failed += 1
            logger.warning("Extraction/embed failed for %s: %s", result.post_id, e)
            if progress_callback:
                progress_callback(idx + 1, total, result.post_id, str(e), 0)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(process_one, r, i) for i, r in enumerate(pseudo_results)]
        concurrent.futures.wait(futs)

    # DB write
    save_message_units(all_units)

    from core.db import get_connection
    if embed_pairs:
        conn = get_connection()
        try:
            for unit_id, embedding in embed_pairs:
                conn.execute(
                    "UPDATE message_units SET embedding = ?, embedded_at = CURRENT_TIMESTAMP WHERE unit_id = ?",
                    [embedding, unit_id],
                )
        finally:
            conn.close()

    return {"done": done, "failed": failed, "units": units_total, "cost": cost_total}


def _update_extraction_cost(post_id: str, cost: float):
    try:
        conn = get_connection()
        conn.execute(
            "UPDATE transcripts SET extraction_cost_usd = ? WHERE post_id = ?",
            [cost, post_id],
        )
        conn.close()
    except Exception:
        pass  # column might not exist yet — migration will add it
