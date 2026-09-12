import streamlit as st
import duckdb
import pandas as pd
import time

st.set_page_config(page_title="Corpus Explorer", page_icon="📝", layout="wide")
st.title("📝 Corpus Explorer")
st.caption("Transcribe, extract, and embed reels — powered by AssemblyAI + Cerebras + Voyage")

DB_PATH = "reels.duckdb"


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH)


conn = get_conn()


def get_transcription_stats():
    try:
        total_audio = conn.execute(
            """SELECT COUNT(*) FROM posts
               WHERE (local_audio_path IS NOT NULL AND local_audio_path != '')
                  OR (audio_url IS NOT NULL AND audio_url != '')
                  OR (video_url IS NOT NULL AND video_url != '')"""
        ).fetchone()[0]
        transcribed = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM transcripts").fetchone()[0]
        pending = max(total_audio - transcribed, 0)
        extracted = conn.execute(
            "SELECT COUNT(DISTINCT post_id) FROM message_units"
        ).fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(DISTINCT post_id) FROM message_units WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        return {
            "transcribed": transcribed,
            "pending": pending,
            "total_audio": total_audio,
            "total_cost_usd": round(total_cost, 4),
            "extracted": extracted,
            "embedded": embedded,
        }
    except Exception:
        return {
            "transcribed": 0, "pending": 0, "total_audio": 0, "total_cost_usd": 0.0,
            "extracted": 0, "embedded": 0,
        }


stats = get_transcription_stats()

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Transcribed", stats["transcribed"])
c2.metric("Pending", stats["pending"])
c3.metric("Transcribable", stats["total_audio"])
c4.metric("Extracted", stats["extracted"])
c5.metric("Embedded", stats["embedded"])
c6.metric("Pipeline Cost", f"${stats['total_cost_usd']:.4f}")

st.markdown("---")

# ── Phase 10 — Unified pipeline (AssemblyAI → Cerebras → Voyage) ──────────────
with st.expander("🚀 Run Full Pipeline  (AssemblyAI → Cerebras → Voyage)", expanded=True):
    st.caption(
        "Transcribes audio with AssemblyAI, extracts message units with Cerebras Qwen-3 235B, "
        "and embeds with Voyage voyage-3-lite.  Stages run concurrently — ~7-9 min for 100 reels."
    )

    # API key check
    assemblyai_key = st.secrets.get("ASSEMBLYAI_API_KEY", "")
    cerebras_key   = st.secrets.get("CEREBRAS_API_KEY", "")
    voyage_key     = st.secrets.get("VOYAGE_API_KEY", "")

    missing = [k for k, v in [
        ("ASSEMBLYAI_API_KEY", assemblyai_key),
        ("CEREBRAS_API_KEY", cerebras_key),
        ("VOYAGE_API_KEY", voyage_key),
    ] if not v]

    if missing:
        st.warning(
            f"⚠️ Missing keys in `.streamlit/secrets.toml`: {', '.join(f'`{k}`' for k in missing)}"
        )

    batch = st.number_input("Batch size", min_value=1, max_value=200, value=20, key="full_batch")

    if st.button("▶ Run full pipeline", type="primary", disabled=bool(missing)):
        from processing.transcription_queue import run_transcription_queue

        st.info("Pipeline started — stages run concurrently. Progress below:")

        prog          = st.progress(0.0)
        transcribed_n = st.empty()
        pills_row     = st.columns(3)
        stt_box       = pills_row[0].empty()
        ext_box       = pills_row[1].empty()
        emb_box       = pills_row[2].empty()

        # Fetch total count
        total_pending = conn.execute(
            """
            SELECT COUNT(*) FROM posts p
            LEFT JOIN transcripts t ON p.post_id = t.post_id
            WHERE (
                  (p.local_audio_path IS NOT NULL AND p.local_audio_path != '')
               OR (p.audio_url       IS NOT NULL AND p.audio_url       != '')
               OR (p.video_url       IS NOT NULL AND p.video_url       != '')
            )
              AND t.post_id IS NULL
            """
        ).fetchone()[0]
        total_run = min(total_pending, int(batch))

        _stt_count = [0]

        def _cb(done, total, post_id, error):
            _stt_count[0] = done
            pct = done / total if total else 0
            prog.progress(pct)
            if error:
                transcribed_n.warning(f"⚠️ `{post_id}`: {error}")
            else:
                transcribed_n.caption(f"Transcribing… {done}/{total}")
            stt_box.metric("Transcribed", done)

        result = run_transcription_queue(assemblyai_key, "assemblyai", int(batch), _cb)

        prog.progress(1.0)
        transcribed_n.empty()

        # Final stats refresh
        final_stats = get_transcription_stats()
        stt_box.metric("Transcribed", result["done"])
        ext_box.metric("Extracted", final_stats["extracted"])
        emb_box.metric("Embedded", final_stats["embedded"])

        if result["failed"] > 0:
            st.warning(f"{result['failed']} STT posts failed — see errors below")
        all_errors = result.get("errors", [])
        if all_errors:
            with st.expander(f"⚠️ {len(all_errors)} pipeline error(s) — click to inspect", expanded=True):
                st.json(all_errors)

        # Clean up local video/audio files to save disk space
        from pathlib import Path
        cleaned = 0
        for d in [Path("downloads"), Path("audio_extracts")]:
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
                        cleaned += 1

        cost_msg = f"Total cost this batch: **${result['total_cost_usd']:.4f}**"
        cleanup_msg = f"  Cleaned up {cleaned} local files." if cleaned else ""
        st.success(
            f"Pipeline complete — {result['done']} transcribed / {result['failed']} failed.  "
            f"{cost_msg}{cleanup_msg}"
        )
        st.rerun()

st.markdown("---")

# ── Re-extract from existing transcripts (for when extraction silently failed) ──
with st.expander("🔁 Re-extract & Re-embed (fix missing knowledge units)", expanded=False):
    st.caption(
        "Transcripts already in DB that have **no message units** will be re-processed through "
        "Cerebras → Voyage.  Use this if the pipeline showed 0 Extracted after transcription ran."
    )

    cerebras_key = st.secrets.get("CEREBRAS_API_KEY", "")
    voyage_key_re = st.secrets.get("VOYAGE_API_KEY", "")
    re_missing = [k for k, v in [("CEREBRAS_API_KEY", cerebras_key), ("VOYAGE_API_KEY", voyage_key_re)] if not v]
    if re_missing:
        st.warning(f"⚠️ Missing keys: {', '.join(f'`{k}`' for k in re_missing)}")

    re_batch = st.number_input("Batch size", min_value=1, max_value=500, value=50, key="re_batch")

    if st.button("🔁 Re-extract missing units", disabled=bool(re_missing)):
        pending_rows = conn.execute(
            """
            SELECT t.post_id, t.transcript
            FROM transcripts t
            LEFT JOIN message_units mu ON t.post_id = mu.post_id
            WHERE mu.post_id IS NULL
              AND t.transcript IS NOT NULL
              AND LENGTH(t.transcript) > 30
            LIMIT ?
            """,
            [int(re_batch)],
        ).fetchall()

        if not pending_rows:
            st.info("All transcripts already have message units — nothing to do.")
        else:
            from providers.factory import get_llm, get_embeddings
            from analysis.extraction import save_message_units
            from core.db import get_connection as _gc

            llm_p = get_llm()
            emb_p = get_embeddings()

            prog_re = st.progress(0.0)
            status_re = st.empty()
            done_re = fail_re = 0
            all_units = []

            # ── Stage 1: Extract all units (Cerebras — no tight rate limit) ──────
            status_re.caption(f"Extracting from {len(pending_rows)} transcripts via Cerebras…")
            for i, (post_id, transcript_text) in enumerate(pending_rows):
                status_re.caption(f"Extracting {i+1}/{len(pending_rows)} — `{post_id}`")
                try:
                    units, _ = llm_p.extract(transcript_text, post_id)
                    if units:
                        all_units.extend(units)
                        done_re += 1
                    else:
                        st.warning(f"No units extracted for `{post_id}` — check Cerebras logs")
                        fail_re += 1
                except Exception as exc:
                    st.error(f"Extraction failed for `{post_id}`: {exc}")
                    fail_re += 1
                prog_re.progress(0.5 * (i + 1) / len(pending_rows))
                time.sleep(2)  # pace at ~30 req/min to stay under Cerebras free-tier limit

            # ── Stage 2: Save units to DB ─────────────────────────────────────────
            if all_units:
                save_message_units(all_units)

            # ── Stage 3: Embed ALL units in one batched call (respects 3 RPM) ────
            all_embed_pairs = []
            if all_units:
                total_batches = (len(all_units) + 127) // 128
                status_re.caption(
                    f"Embedding {len(all_units)} units in {total_batches} batch(es) "
                    f"— respecting Voyage 3 RPM limit (~20 s between batches)…"
                )
                try:
                    embeddings = emb_p.embed_batch([u.text for u in all_units], rate_per_min=3)
                    all_embed_pairs = [(u.unit_id, emb) for u, emb in zip(all_units, embeddings)]
                except Exception as emb_err:
                    st.error(f"Embedding failed: {emb_err}")

            prog_re.progress(0.9)

            if all_embed_pairs:
                conn_w = _gc()
                for unit_id, embedding in all_embed_pairs:
                    conn_w.execute(
                        "UPDATE message_units SET embedding = ?, embedded_at = CURRENT_TIMESTAMP WHERE unit_id = ?",
                        [embedding, unit_id],
                    )
                conn_w.close()

            status_re.empty()
            prog_re.progress(1.0)
            st.success(
                f"Re-extraction complete — {done_re} posts produced units / {fail_re} failed. "
                f"{len(all_units)} knowledge units saved, {len(all_embed_pairs)} embedded."
            )
            st.rerun()

st.markdown("---")

# ── Transcript list ────────────────────────────────────────────────────────────
search = st.text_input("🔍 Search transcripts", placeholder="e.g., ATS, resume, STAR method")

try:
    if search:
        df = conn.execute(
            """
            SELECT t.post_id, p.account_id, t.transcript, t.confidence,
                   t.duration_sec, t.word_count, t.cost_usd, t.transcribed_at
            FROM transcripts t
            LEFT JOIN posts p ON t.post_id = p.post_id
            WHERE LOWER(t.transcript) LIKE ?
            ORDER BY t.transcribed_at DESC
            LIMIT 100
            """,
            [f"%{search.lower()}%"],
        ).df()
    else:
        df = conn.execute(
            """
            SELECT t.post_id, p.account_id, t.transcript, t.confidence,
                   t.duration_sec, t.word_count, t.cost_usd, t.transcribed_at
            FROM transcripts t
            LEFT JOIN posts p ON t.post_id = p.post_id
            ORDER BY t.transcribed_at DESC
            LIMIT 100
            """,
        ).df()
except Exception:
    df = pd.DataFrame()

if df.empty:
    st.info("No transcripts yet. Run a scrape, download videos, extract audio, then transcribe.")
else:
    st.caption(f"{len(df)} transcripts")
    df["transcript_preview"] = df["transcript"].str.slice(0, 200) + "…"
    display_df = df.drop(columns=["transcript"])
    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 column_config={
                     "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                     "duration_sec": st.column_config.NumberColumn("Duration (s)", format="%.0f"),
                     "cost_usd": st.column_config.NumberColumn("Cost $", format="%.4f"),
                 })

    # Full transcript viewer
    st.markdown("---")
    st.subheader("Full Transcript Viewer")
    selected_post = st.selectbox("Select post", df["post_id"].tolist())
    if selected_post:
        row = df[df["post_id"] == selected_post].iloc[0]
        st.write(f"**@{row.get('account_id', '?')}** — {row.get('duration_sec', 0):.0f}s — "
                 f"confidence {row.get('confidence', 0):.2f} — ${row.get('cost_usd', 0):.4f}")
        st.text_area("Full Transcript", value=row["transcript"], height=250, disabled=True)

        # Word timestamps
        try:
            words_df = conn.execute(
                "SELECT word_index, word, start_sec, end_sec, confidence "
                "FROM transcript_words WHERE post_id = ? ORDER BY word_index",
                [selected_post],
            ).df()
            if not words_df.empty:
                with st.expander("📊 Word Timestamps"):
                    st.dataframe(words_df, use_container_width=True, hide_index=True)
        except Exception:
            pass
