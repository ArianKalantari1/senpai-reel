"""
Batch transcription runner — transcribes all untranscribed posts.

Uses AssemblyAI (from remote audio_url / video_url) → Cerebras extraction → Voyage embeddings.
Processes in batches to allow incremental progress and safe interruption.
"""
import os
import sys
import time
import logging

# ── Load secrets ───────────────────────────────────────────────────────────
try:
    import tomllib
except ImportError:
    import tomli as tomllib

with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

os.environ["ASSEMBLYAI_API_KEY"] = secrets.get("ASSEMBLYAI_API_KEY", "")
os.environ["CEREBRAS_API_KEY"] = secrets.get("CEREBRAS_API_KEY", "")
os.environ["VOYAGE_API_KEY"] = secrets.get("VOYAGE_API_KEY", "")
os.environ["GROQ_API_KEY"] = secrets.get("GROQ_API_KEY", "")

# Monkey-patch st.secrets so providers.factory works
import streamlit as st

class FakeSecrets(dict):
    def get(self, key, default=None):
        return super().get(key, default)
    def __getitem__(self, key):
        return super().__getitem__(key)

st.secrets = FakeSecrets(secrets)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_transcribe")

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 50           # posts per pipeline run
ASSEMBLYAI_KEY = secrets.get("ASSEMBLYAI_API_KEY", "")

def count_pending():
    """Count posts that still need transcription."""
    from core.db import get_connection
    conn = get_connection()
    n = conn.execute("""
        SELECT COUNT(*)
        FROM posts p
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE (
              (p.audio_url  IS NOT NULL AND p.audio_url  != '')
           OR (p.video_url  IS NOT NULL AND p.video_url  != '')
        )
          AND t.post_id IS NULL
    """).fetchone()[0]
    conn.close()
    return n


def get_pending_with_audio_url():
    """Count posts that have a dedicated audio_url (most reliable)."""
    from core.db import get_connection
    conn = get_connection()
    n = conn.execute("""
        SELECT COUNT(*)
        FROM posts p
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE p.audio_url IS NOT NULL AND p.audio_url != ''
          AND t.post_id IS NULL
    """).fetchone()[0]
    conn.close()
    return n


def progress_cb(done, total, post_id, error):
    if error:
        logger.warning(f"  [{done}/{total}] FAIL {post_id}: {error}")
    else:
        logger.info(f"  [{done}/{total}] OK   {post_id}")


def get_pending_rows(batch_size, prefer_audio_url=True):
    """Get a batch of (post_id, audio_source) tuples, preferring audio_url."""
    from core.db import get_connection
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.post_id,
               COALESCE(
                   NULLIF(p.audio_url, ''),
                   p.video_url
               ) AS audio_source
        FROM posts p
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE (
              (p.audio_url  IS NOT NULL AND p.audio_url  != '')
           OR (p.video_url  IS NOT NULL AND p.video_url  != '')
        )
          AND t.post_id IS NULL
        ORDER BY
            CASE WHEN p.audio_url IS NOT NULL AND p.audio_url != '' THEN 0 ELSE 1 END,
            p.post_id
        LIMIT ?
    """, [batch_size]).fetchall()
    conn.close()
    return rows


def main():
    from providers.stt import AssemblyAIProvider
    from providers.factory import get_llm
    from pipeline.stream_runner import run_stream_pipeline

    stt = AssemblyAIProvider(api_key=ASSEMBLYAI_KEY)
    llm = get_llm()
    emb = None  # Skip embeddings — Voyage free-tier rate limit (3 RPM) causes failures

    pending = count_pending()
    logger.info(f"=== Starting batch transcription: {pending} posts pending ===")
    logger.info(f"    (of which {get_pending_with_audio_url()} have audio_url)")

    if pending == 0:
        logger.info("Nothing to transcribe — all posts already have transcripts.")
        return

    batch_num = 0
    total_done = 0
    total_failed = 0
    total_cost = 0.0
    all_errors = []
    start = time.time()

    while True:
        rows = get_pending_rows(BATCH_SIZE)
        if not rows:
            break

        batch_num += 1
        logger.info(f"\n--- Batch {batch_num}: processing {len(rows)} posts ---")

        try:
            stats = run_stream_pipeline(
                post_rows=rows,
                stt_provider=stt,
                llm_provider=llm,
                embed_provider=emb,
                progress_callback=progress_cb,
            )

            done = stats.stt_done
            failed = stats.stt_failed
            cost = stats.total_cost_usd

            total_done += done
            total_failed += failed
            total_cost += cost
            all_errors.extend(stats.errors)

            logger.info(f"  Batch {batch_num} complete: {done} done, {failed} failed, ${cost:.4f}")
            logger.info(f"  Running totals: {total_done} done, {total_failed} failed, ${total_cost:.4f}")

            # If the entire batch failed, stop to avoid infinite loop
            if done == 0 and failed > 0:
                logger.error("Entire batch failed — stopping to avoid loop.")
                break

        except Exception as e:
            logger.error(f"Batch {batch_num} crashed: {e}")
            import traceback
            traceback.print_exc()
            all_errors.append({"batch": batch_num, "error": str(e)})
            break

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed/60:.1f} minutes")
    logger.info(f"  Transcribed: {total_done}")
    logger.info(f"  Failed:      {total_failed}")
    logger.info(f"  Total cost:  ${total_cost:.4f}")
    logger.info(f"  Remaining:   {count_pending()}")

    if all_errors:
        logger.info(f"\nErrors ({len(all_errors)}):")
        for e in all_errors[:20]:
            logger.info(f"  {e}")
        if len(all_errors) > 20:
            logger.info(f"  ... and {len(all_errors) - 20} more")


if __name__ == "__main__":
    main()
