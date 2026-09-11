import streamlit as st

st.set_page_config(page_title="Search", page_icon="🔍", layout="wide")
st.title("Search")

from analysis.taxonomy import CONTENT_TYPES
from core.taxonomy import topic_names
from analysis.search import semantic_search, keyword_search
from core.client_context import render_client_selector
from core.config import get_secret, missing_secret_message
from core.db import get_connection, init_db
from core.navigation import render_page_link
from core.pipeline import get_pipeline_lock


init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Search {active_client['name']}'s extracted knowledge units by meaning, not just keywords")
pipeline_lock = get_pipeline_lock()
pipeline_busy = pipeline_lock is not None

if pipeline_busy:
    st.warning(
        f"Pipeline busy. Current stage: {pipeline_lock.get('stage') or 'Starting'}. "
        "Extraction and embedding actions are paused until it finishes."
    )


def _unit_count(active_client_id: str) -> int:
    try:
        conn = get_connection()
        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ?
            """,
            [active_client_id],
        ).fetchone()[0]
        embedded = conn.execute(
            """
            SELECT COUNT(*)
            FROM message_units mu
            JOIN client_posts cp ON mu.post_id = cp.post_id
            WHERE cp.client_id = ?
              AND mu.embedding IS NOT NULL
            """,
            [active_client_id],
        ).fetchone()[0]
        conn.close()
        return total, embedded
    except Exception:
        return 0, 0


def _render_embedding_sidebar(openai_key: str, total_units: int, embedded_units: int):
    with st.sidebar:
        st.header("Embedding Pipeline")
        st.write(f"{embedded_units}/{total_units} units embedded")

        if openai_key and embedded_units < total_units:
            if st.button("Embed pending units", disabled=pipeline_busy):
                from analysis.embeddings import embed_pending_units

                with st.spinner("Embedding…"):
                    r = embed_pending_units(openai_key, batch_size=50, client_id=client_id)
                st.success(f"Done — {r['done']} embedded, {r['failed']} failed")
                st.rerun()
        elif not openai_key:
            st.warning(missing_secret_message("OPENAI_API_KEY", "Semantic search"))
            render_page_link(st, "Settings.py", "Open Settings")


def _render_search(openai_key: str, total_units: int, embedded_units: int):
    st.markdown("---")
    col_q, col_topic, col_ct, col_k = st.columns([3, 1, 1, 1])

    with col_q:
        query = st.text_input("Search query", placeholder="e.g., how to pass ATS screening")

    with col_topic:
        topic_filter = st.selectbox("Topic", ["All"] + topic_names(client_id))

    with col_ct:
        ct_filter = st.selectbox("Content type", ["All"] + CONTENT_TYPES)

    with col_k:
        top_k = st.number_input("Results", min_value=5, max_value=100, value=20)

    search_mode = "semantic" if openai_key and embedded_units > 0 else "keyword"
    if search_mode == "keyword":
        st.caption("Keyword search mode. Add `OPENAI_API_KEY` and run embeddings for semantic search.")
    else:
        st.caption("Semantic search mode.")

    if query.strip():
        with st.spinner("Searching…"):
            if search_mode == "semantic":
                results = semantic_search(query, openai_key, client_id, topic_filter, ct_filter, top_k)
            else:
                results = keyword_search(query, client_id, topic_filter, top_k, ct_filter)

        if not results:
            st.info("No results found. Try a different query or remove filters.")
            return

        st.success(f"{len(results)} results")
        for result in results:
            with st.container():
                col_score, col_content = st.columns([1, 8])
                with col_score:
                    score_pct = int(result.score * 100) if search_mode == "semantic" else "—"
                    st.metric("Score", f"{score_pct}%" if isinstance(score_pct, int) else score_pct)
                    st.caption(f"`{result.topic}`")
                    st.caption(f"`{result.content_type}`")

                with col_content:
                    st.markdown(f"**@{result.username}**  ·  {result.posted_at or ''}")
                    st.write(f"**{result.text}**")
                    if result.claim and result.claim != result.text:
                        st.caption(f"Claim: {result.claim}")
                    if result.video_url:
                        st.markdown(f"[Watch video]({result.video_url})")

                st.divider()
    else:
        st.info("Enter a search query above to find knowledge units.")


total_units, embedded_units = _unit_count(client_id)

c1, c2 = st.columns(2)
c1.metric("Total Knowledge Units", total_units)
c2.metric("Embedded", embedded_units)

openai_key = get_secret("OPENAI_API_KEY", st)

if total_units == 0:
    st.warning("No knowledge units yet. Run stage 4 on the Pipeline page after transcription is complete.")
    render_page_link(st, "Pipeline.py", "Open Pipeline")
else:
    _render_embedding_sidebar(openai_key, total_units, embedded_units)
    _render_search(openai_key, total_units, embedded_units)

# ── Extraction queue trigger (bottom) ─────────────────────────────────────────
with st.expander("Run Extraction Queue"):
    openai_key2 = get_secret("OPENAI_API_KEY", st)

    batch = st.number_input("Batch size", 1, 50, 10, key="ext_batch")
    if not openai_key2:
        st.warning(missing_secret_message("OPENAI_API_KEY", "Extraction"))
        render_page_link(st, "Settings.py", "Open Settings")
    elif st.button("Extract knowledge units from transcripts", type="primary", disabled=pipeline_busy):
        from processing.extraction_queue import run_extraction_queue

        prog = st.progress(0)
        status = st.empty()

        def _cb(done, total, post_id, err, n_units):
            prog.progress(done / total)
            if err:
                status.warning(f"{post_id}: {err}")
            else:
                status.info(f"{done}/{total} — {n_units} units from `{post_id}`")

        result = run_extraction_queue(openai_key2, batch, _cb, client_id)
        status.empty()
        prog.empty()
        st.success(
            f"Done — {result['done']} posts processed, "
            f"{result['total_units_extracted']} units extracted, "
            f"${result['total_cost_usd']:.4f} cost"
        )
        st.rerun()
