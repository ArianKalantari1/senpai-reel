import streamlit as st
import pandas as pd

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search")

from analysis.taxonomy import TOPICS, CONTENT_TYPES
from analysis.search import semantic_search, keyword_search
from core.client_context import render_client_selector
from core.db import get_connection, init_db


init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Search {active_client['name']}'s extracted knowledge units by meaning, not just keywords")


def _unit_count(client_id: str) -> int:
    try:
        conn = get_connection()
        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [client_id],
        ).fetchone()[0]
        embedded = conn.execute(
            """
            SELECT COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ?
              AND mu.embedding IS NOT NULL
            """,
            [client_id],
        ).fetchone()[0]
        conn.close()
        return total, embedded
    except Exception:
        return 0, 0


total_units, embedded_units = _unit_count(client_id)

c1, c2 = st.columns(2)
c1.metric("Total Knowledge Units", total_units)
c2.metric("Embedded (semantic search ready)", embedded_units)

if total_units == 0:
    st.warning("No knowledge units yet. Run: Scrape → Download → Extract Audio → Transcribe → Extract Units.")
    st.stop()

st.markdown("---")

# ── Search controls ────────────────────────────────────────────────────────────
col_q, col_topic, col_ct, col_k = st.columns([3, 1, 1, 1])

with col_q:
    query = st.text_input("Search query", placeholder="e.g., how to pass ATS screening")

with col_topic:
    topic_filter = st.selectbox("Topic", ["All"] + TOPICS)

with col_ct:
    ct_filter = st.selectbox("Content type", ["All"] + CONTENT_TYPES)

with col_k:
    top_k = st.number_input("Results", min_value=5, max_value=100, value=20)

search_mode = "keyword"
try:
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    if openai_key and embedded_units > 0:
        search_mode = "semantic"
except Exception:
    openai_key = ""

if search_mode == "keyword":
    st.caption("⚡ Keyword search mode (set OPENAI_API_KEY + run embedding pipeline for semantic search)")
else:
    st.caption("🧠 Semantic search mode (cosine similarity on embeddings)")

if query.strip():
    with st.spinner("Searching…"):
        if search_mode == "semantic":
            results = semantic_search(query, openai_key, client_id, topic_filter, ct_filter, top_k)
        else:
            results = keyword_search(query, client_id, topic_filter, top_k, ct_filter)

    if not results:
        st.info("No results found. Try a different query or remove filters.")
    else:
        st.success(f"{len(results)} results")

        # ── Embed pipeline trigger (sidebar) ───────────────────────────────────
        with st.sidebar:
            st.header("⚙️ Embedding Pipeline")
            st.write(f"{embedded_units}/{total_units} units embedded")
            if openai_key and embedded_units < total_units:
                if st.button("▶️ Embed pending units"):
                    from analysis.embeddings import embed_pending_units
                    with st.spinner("Embedding…"):
                        r = embed_pending_units(openai_key, batch_size=50, client_id=client_id)
                    st.success(f"Done — {r['done']} embedded, {r['failed']} failed")
                    st.rerun()
            elif not openai_key:
                st.warning("Set `OPENAI_API_KEY` in secrets.toml to enable embedding.")

        # ── Results cards ──────────────────────────────────────────────────────
        for i, r in enumerate(results):
            with st.container():
                col_score, col_content = st.columns([1, 8])
                with col_score:
                    score_pct = int(r.score * 100) if search_mode == "semantic" else "—"
                    st.metric("Score", f"{score_pct}%" if isinstance(score_pct, int) else score_pct)
                    st.caption(f"`{r.topic}`")
                    st.caption(f"`{r.content_type}`")

                with col_content:
                    st.markdown(f"**@{r.username}**  ·  {r.posted_at or ''}")
                    st.write(f"📌 **{r.text}**")
                    if r.claim and r.claim != r.text:
                        st.caption(f"Claim: {r.claim}")
                    if r.video_url:
                        st.markdown(f"[🎥 Watch video]({r.video_url})")

                st.divider()
else:
    st.info("Enter a search query above to find knowledge units.")
    # Show embedding pipeline in sidebar always
    with st.sidebar:
        st.header("⚙️ Embedding Pipeline")
        st.write(f"{embedded_units}/{total_units} units embedded")

        try:
            openai_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            openai_key = ""

        if openai_key and embedded_units < total_units:
            if st.button("▶️ Embed pending units"):
                from analysis.embeddings import embed_pending_units
                with st.spinner("Embedding…"):
                    r = embed_pending_units(openai_key, batch_size=50, client_id=client_id)
                st.success(f"Done — {r['done']} embedded, {r['failed']} failed")
                st.rerun()
        elif not openai_key:
            st.warning("Set `OPENAI_API_KEY` in secrets.toml to enable semantic search.")

# ── Extraction queue trigger (bottom) ─────────────────────────────────────────
with st.expander("⚙️ Run Extraction Queue (Phase 4)"):
    try:
        openai_key2 = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        openai_key2 = ""

    batch = st.number_input("Batch size", 1, 50, 10, key="ext_batch")
    if not openai_key2:
        st.warning("OPENAI_API_KEY not set.")
    elif st.button("▶️ Extract knowledge units from transcripts", type="primary"):
        from processing.extraction_queue import run_extraction_queue

        prog = st.progress(0)
        status = st.empty()

        def _cb(done, total, post_id, err, n_units):
            prog.progress(done / total)
            if err:
                status.warning(f"⚠️ {post_id}: {err}")
            else:
                status.info(f"✅ {done}/{total} — {n_units} units from `{post_id}`")

        result = run_extraction_queue(openai_key2, batch, _cb, client_id)
        status.empty()
        prog.empty()
        st.success(
            f"Done — {result['done']} posts processed, "
            f"{result['total_units_extracted']} units extracted, "
            f"${result['total_cost_usd']:.4f} cost"
        )
        st.rerun()
