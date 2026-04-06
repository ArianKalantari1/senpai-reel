import streamlit as st
import duckdb
import pandas as pd

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
            "SELECT COUNT(*) FROM posts WHERE local_audio_path IS NOT NULL AND local_audio_path != ''"
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
c3.metric("Total Audio Files", stats["total_audio"])
c4.metric("Extracted", stats["extracted"])
c5.metric("Embedded", stats["embedded"])
c6.metric("Pipeline Cost", f"${stats['total_cost_usd']:.4f}")

st.markdown("---")

# ── Phase 10 — Unified pipeline (AssemblyAI → Cerebras → Voyage) ──────────────
with st.expander("🚀 Run Full Pipeline  (AssemblyAI → Cerebras → Voyage)", expanded=True):
    st.caption(
        "Transcribes audio with AssemblyAI, extracts message units with Cerebras LLaMA 3.3 70B, "
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
            WHERE p.local_audio_path IS NOT NULL
              AND p.local_audio_path != ''
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
            st.warning(f"{result['failed']} posts failed — see errors below")
            st.json(result.get("errors", []))

        st.success(
            f"Pipeline complete — {result['done']} transcribed / {result['failed']} failed.  "
            f"Total cost this batch: **${result['total_cost_usd']:.4f}**"
        )
        st.rerun()

st.markdown("---")

# ── Legacy queue triggers (Deepgram / Whisper rollback) ────────────────────────
with st.expander("⚙️ Legacy: Deepgram / Whisper transcription"):
    prov = st.radio("Provider", ["deepgram", "whisper"], horizontal=True)
    batch_leg = st.number_input("Batch size", min_value=1, max_value=50, value=10, key="leg_batch")

    api_key_field = "DEEPGRAM_API_KEY" if prov == "deepgram" else "OPENAI_API_KEY"
    try:
        api_key = st.secrets[api_key_field]
        key_ok = bool(api_key)
    except Exception:
        api_key = ""
        key_ok = False

    if not key_ok:
        st.warning(f"⚠️ `{api_key_field}` not found in `.streamlit/secrets.toml`")
    else:
        if st.button(f"▶️ Transcribe {batch_leg} posts via {prov}", type="secondary"):
            from processing.transcription_queue import run_transcription_queue

            prog = st.progress(0)
            status_txt = st.empty()

            def _cb_leg(done, total, post_id, err):
                prog.progress(done / total)
                if err:
                    status_txt.warning(f"⚠️ {post_id}: {err}")
                else:
                    status_txt.info(f"✅ {done}/{total}: `{post_id}`")

            result = run_transcription_queue(api_key, prov, batch_leg, _cb_leg)
            status_txt.empty()
            prog.empty()
            st.success(
                f"Done — {result['done']} transcribed, {result['failed']} failed. "
                f"Cost this batch: ${result['total_cost_usd']:.4f}"
            )
            if result.get("errors"):
                st.json(result["errors"])
            st.rerun()

st.markdown("---")

# ── Single-post transcribe ─────────────────────────────────────────────────────
with st.expander("🎙️ Transcribe a single post"):
    pid = st.text_input("Post ID", placeholder="e.g., ABC123")
    prov_single = st.radio("Provider", ["deepgram", "whisper"], key="single_prov", horizontal=True)
    if st.button("▶️ Transcribe", key="btn_single_tx"):
        key_field = "DEEPGRAM_API_KEY" if prov_single == "deepgram" else "OPENAI_API_KEY"
        try:
            key = st.secrets[key_field]
        except Exception:
            key = ""
        if not key:
            st.error(f"`{key_field}` not set in secrets.")
        elif not pid.strip():
            st.error("Enter a post ID.")
        else:
            from processing.transcribe import transcribe_post
            with st.spinner("Transcribing…"):
                try:
                    res = transcribe_post(pid.strip(), key, prov_single)
                    st.success(f"Done — {res.word_count} words, ${res.cost_usd:.4f}")
                    st.write(res.transcript)
                except Exception as e:
                    st.error(str(e))

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

DB_PATH = "reels.duckdb"


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH)


conn = get_conn()


def get_transcription_stats():
    try:
        total_audio = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE local_audio_path IS NOT NULL AND local_audio_path != ''"
        ).fetchone()[0]
        transcribed = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM transcripts").fetchone()[0]
        pending = max(total_audio - transcribed, 0)
        return {
            "transcribed": transcribed,
            "pending": pending,
            "total_audio": total_audio,
            "total_cost_usd": round(total_cost, 4),
        }
    except Exception:
        return {"transcribed": 0, "pending": 0, "total_audio": 0, "total_cost_usd": 0.0}


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
    try:
        api_key = st.secrets[api_key_field]
        key_ok = bool(api_key)
    except Exception:
        api_key = ""
        key_ok = False

    if not key_ok:
        st.warning(f"⚠️ `{api_key_field}` not found in `.streamlit/secrets.toml`")
    else:
        if st.button(f"▶️ Transcribe {batch} posts via {prov}", type="primary"):
            from processing.transcription_queue import run_transcription_queue

            prog = st.progress(0)
            status_txt = st.empty()

            def _cb(done, total, post_id, err):
                prog.progress(done / total)
                if err:
                    status_txt.warning(f"⚠️ {post_id}: {err}")
                else:
                    status_txt.info(f"✅ {done}/{total}: `{post_id}`")

            result = run_transcription_queue(api_key, prov, batch, _cb)
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
    if st.button("▶️ Transcribe", key="btn_single_tx"):
        key_field = "DEEPGRAM_API_KEY" if prov_single == "deepgram" else "OPENAI_API_KEY"
        try:
            key = st.secrets[key_field]
        except Exception:
            key = ""
        if not key:
            st.error(f"`{key_field}` not set in secrets.")
        elif not pid.strip():
            st.error("Enter a post ID.")
        else:
            from processing.transcribe import transcribe_post
            with st.spinner("Transcribing…"):
                try:
                    res = transcribe_post(pid.strip(), key, prov_single)
                    st.success(f"Done — {res.word_count} words, ${res.cost_usd:.4f}")
                    st.write(res.transcript)
                except Exception as e:
                    st.error(str(e))

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
            words_df = conn.execute(
                "SELECT word_index, word, start_sec, end_sec, confidence "
                "FROM transcript_words WHERE post_id = ? ORDER BY word_index",
                [selected_post],
            ).df()
            if not words_df.empty:
                with st.expander("📊 Word Timestamps"):
                    st.dataframe(words_df, width="stretch", hide_index=True)
        except Exception:
            pass
