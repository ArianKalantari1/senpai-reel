import streamlit as st
import pandas as pd

from core.client_context import render_client_selector
from core.config import get_secret, missing_secret_message
from core.db import get_connection, init_db
from core.navigation import render_page_link
from core.pipeline import get_pipeline_lock

st.set_page_config(page_title="Corpus Explorer", page_icon="📝", layout="wide")
st.title("Corpus Explorer")
init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Search transcripts for {active_client['name']}, view full text, track transcription cost")
pipeline_lock = get_pipeline_lock()
pipeline_busy = pipeline_lock is not None

if pipeline_busy:
    st.warning(
        f"Pipeline busy. Current stage: {pipeline_lock.get('stage') or 'Starting'}. "
        "Transcription actions are paused until it finishes."
    )


def get_transcription_stats():
    conn = get_connection()
    try:
        total_audio = conn.execute(
            """
            SELECT COUNT(*)
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            WHERE cp.client_id = ?
              AND p.local_audio_path IS NOT NULL
              AND p.local_audio_path != ''
            """,
            [client_id],
        ).fetchone()[0]
        transcribed = conn.execute(
            """
            SELECT COUNT(*)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        total_cost = conn.execute(
            """
            SELECT COALESCE(SUM(t.cost_usd), 0)
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        pending = max(total_audio - transcribed, 0)
        return {
            "transcribed": transcribed,
            "pending": pending,
            "total_audio": total_audio,
            "total_cost_usd": round(total_cost, 4),
        }
    except Exception:
        return {"transcribed": 0, "pending": 0, "total_audio": 0, "total_cost_usd": 0.0}
    finally:
        conn.close()


stats = get_transcription_stats()

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transcribed", stats["transcribed"])
c2.metric("Pending", stats["pending"])
c3.metric("Total Audio Files", stats["total_audio"])
# Free Deepgram credit = $200; $0.0058/min
remaining_free = max(200.0 - stats["total_cost_usd"], 0)
c4.metric("Total Cost", f"${stats['total_cost_usd']:.4f}", delta=f"${remaining_free:.2f} free remains")

st.markdown("---")

# ── Transcription queue trigger ────────────────────────────────────────────────
with st.expander("⚙️ Run Transcription Queue"):
    prov = st.radio("Provider", ["deepgram", "whisper"], horizontal=True)
    batch = st.number_input("Batch size", min_value=1, max_value=50, value=10)

    api_key_field = "DEEPGRAM_API_KEY" if prov == "deepgram" else "OPENAI_API_KEY"
    api_key = get_secret(api_key_field, st)
    key_ok = bool(api_key)

    if not key_ok:
        st.warning(missing_secret_message(api_key_field, "Transcription"))
        render_page_link(st, "Settings.py", "Open Settings")
    else:
        if st.button(f"▶️ Transcribe {batch} posts via {prov}", type="primary", disabled=pipeline_busy):
            from processing.transcription_queue import run_transcription_queue

            prog = st.progress(0)
            status_txt = st.empty()

            def _cb(done, total, post_id, err):
                prog.progress(done / total)
                if err:
                    status_txt.warning(f"⚠️ {post_id}: {err}")
                else:
                    status_txt.info(f"✅ {done}/{total}: `{post_id}`")

            result = run_transcription_queue(api_key, prov, batch, _cb, client_id)
            status_txt.empty()
            prog.empty()
            st.success(
                f"Done — {result['done']} transcribed, {result['failed']} failed. "
                f"Cost this batch: ${result['total_cost_usd']:.4f}"
            )
            if result["errors"]:
                st.json(result["errors"])
            st.rerun()

st.markdown("---")

# ── Single-post transcribe ─────────────────────────────────────────────────────
with st.expander("🎙️ Transcribe a single post"):
    pid = st.text_input("Post ID", placeholder="e.g., ABC123")
    prov_single = st.radio("Provider", ["deepgram", "whisper"], key="single_prov", horizontal=True)
    if st.button("▶️ Transcribe", key="btn_single_tx", disabled=pipeline_busy):
        key_field = "DEEPGRAM_API_KEY" if prov_single == "deepgram" else "OPENAI_API_KEY"
        key = get_secret(key_field, st)
        if not key:
            st.error(missing_secret_message(key_field, "Transcription"))
            render_page_link(st, "Settings.py", "Open Settings")
        elif not pid.strip():
            st.error("Enter a post ID.")
        else:
            from processing.transcribe import transcribe_post
            with st.spinner("Transcribing…"):
                try:
                    res = transcribe_post(pid.strip(), key, prov_single, client_id)
                    st.success(f"Done — {res.word_count} words, ${res.cost_usd:.4f}")
                    st.write(res.transcript)
                except Exception as e:
                    st.error(str(e))

st.markdown("---")

# ── Transcript list ────────────────────────────────────────────────────────────
search = st.text_input("🔍 Search transcripts", placeholder="e.g., ATS, resume, STAR method")

try:
    conn = get_connection()
    if search:
        df = conn.execute(
            """
            SELECT t.post_id, COALESCE(ca.username, p.account_id) AS account_id,
                   t.transcript, t.confidence,
                   t.duration_sec, t.word_count, t.cost_usd, t.transcribed_at
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            LEFT JOIN posts p ON t.post_id = p.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            WHERE LOWER(t.transcript) LIKE ?
              AND cp.client_id = ?
            ORDER BY t.transcribed_at DESC
            LIMIT 100
            """,
            [f"%{search.lower()}%", client_id],
        ).df()
    else:
        df = conn.execute(
            """
            SELECT t.post_id, COALESCE(ca.username, p.account_id) AS account_id,
                   t.transcript, t.confidence,
                   t.duration_sec, t.word_count, t.cost_usd, t.transcribed_at
            FROM transcripts t
            JOIN client_posts cp ON t.post_id = cp.post_id
            LEFT JOIN posts p ON t.post_id = p.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            WHERE cp.client_id = ?
            ORDER BY t.transcribed_at DESC
            LIMIT 100
            """,
            [client_id],
        ).df()
except Exception:
    df = pd.DataFrame()
finally:
    try:
        conn.close()
    except Exception:
        pass

if df.empty:
    st.info("No transcripts yet. Run stage 3 on the Pipeline page after scraping, downloading, and extracting audio.")
    render_page_link(st, "Pipeline.py", "Open Pipeline")
else:
    st.caption(f"{len(df)} transcripts")
    # Truncate for table display
    df["transcript_preview"] = df["transcript"].str.slice(0, 200) + "…"
    display_df = df.drop(columns=["transcript"])
    st.dataframe(display_df, width="stretch", hide_index=True,
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
            conn = get_connection()
            words_df = conn.execute(
                "SELECT word_index, word, start_sec, end_sec, confidence "
                "FROM transcript_words WHERE post_id = ? AND client_id = ? ORDER BY word_index",
                [selected_post, client_id],
            ).df()
            if not words_df.empty:
                with st.expander("📊 Word Timestamps"):
                    st.dataframe(words_df, width="stretch", hide_index=True)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
