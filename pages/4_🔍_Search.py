import streamlit as st
import pandas as pd

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search")
st.caption("Search across all extracted knowledge units by meaning, not just keywords")

from analysis.taxonomy import TOPICS, CONTENT_TYPES
from analysis.search import semantic_search, keyword_search
from core.db import get_connection


def _unit_count() -> int:
    try:
        conn = get_connection()
        total = conn.execute("SELECT COUNT(*) FROM message_units").fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        return total, embedded
    except Exception:
        return 0, 0


total_units, embedded_units = _unit_count()

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
    voyage_key = st.secrets.get("VOYAGE_API_KEY", "")
    if voyage_key and embedded_units > 0:
        search_mode = "semantic"
except Exception:
    voyage_key = ""

if search_mode == "keyword":
    st.caption("⚡ Keyword search mode (set VOYAGE_API_KEY + run the full pipeline to enable semantic search)")
else:
    st.caption("🧠 Semantic search mode (cosine similarity on Voyage embeddings)")

if query.strip():
    with st.spinner("Searching…"):
        if search_mode == "semantic":
            results = semantic_search(query, "", topic_filter, ct_filter, top_k)
        else:
            results = keyword_search(query, topic_filter, top_k)

    if not results:
        st.info("No results found. Try a different query or remove filters.")
    else:
        st.success(f"{len(results)} results")

        # ── Embed pipeline trigger (sidebar) ───────────────────────────────────
        with st.sidebar:
            st.header("⚙️ Embedding Pipeline")
            st.write(f"{embedded_units}/{total_units} units embedded")
            if voyage_key and embedded_units < total_units:
                if st.button("▶️ Embed pending units"):
                    from analysis.embeddings import embed_pending_units
                    with st.spinner("Embedding…"):
                        r = embed_pending_units(batch_size=50)
                    st.success(f"Done — {r['done']} embedded, {r['failed']} failed")
                    st.rerun()
            elif not voyage_key:
                st.warning("Set `VOYAGE_API_KEY` in secrets.toml to enable semantic search.")

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

        if voyage_key and embedded_units < total_units:
            if st.button("▶️ Embed pending units"):
                from analysis.embeddings import embed_pending_units
                with st.spinner("Embedding…"):
                    r = embed_pending_units(batch_size=50)
                st.success(f"Done — {r['done']} embedded, {r['failed']} failed")
                st.rerun()
        elif not voyage_key:
            st.warning("Set `VOYAGE_API_KEY` in secrets.toml to enable semantic search.")


