import streamlit as st
import pandas as pd

st.set_page_config(page_title="Content Studio", page_icon="✍️", layout="wide")
st.title("Content Studio")

from core.taxonomy import topic_names
from analysis.search import keyword_search, semantic_search
from core.client_context import render_client_selector
from core.config import get_secret, missing_secret_message
from core.db import get_connection, init_db
from core.navigation import render_page_link
from core.pipeline import get_pipeline_lock

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Generate Instagram captions, hooks, and scripts for {active_client['name']}")

openai_key = get_secret("OPENAI_API_KEY", st)
pipeline_lock = get_pipeline_lock()
pipeline_busy = pipeline_lock is not None

if pipeline_busy:
    st.warning(
        f"Pipeline busy. Current stage: {pipeline_lock.get('stage') or 'Starting'}. "
        "Generation actions are paused until it finishes."
    )


def _render_history():
    st.subheader("Generation History")
    conn = get_connection()
    try:
        df = conn.execute(
            """
            SELECT gen_id, created_at, topic, content_type,
                   LEFT(output_text, 100) AS preview, model, tokens_used, cost_usd
            FROM generated_content
            WHERE client_id = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            [client_id],
        ).df()
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        st.info("No generated content yet. Run the Pipeline, then come back here to create the first draft.")
        render_page_link(st, "Pipeline.py", "Open Pipeline")
        return

    total_cost = df["cost_usd"].sum() if "cost_usd" in df else 0
    st.caption(f"{len(df)} items generated — total cost: ${total_cost:.4f}")
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "cost_usd": st.column_config.NumberColumn("Cost $", format="%.4f"),
            "preview": st.column_config.TextColumn("Preview", width="large"),
        },
    )

    selected_id = st.selectbox("View full output", df["gen_id"].tolist())
    if selected_id:
        conn2 = get_connection()
        row = conn2.execute(
            "SELECT output_text FROM generated_content WHERE gen_id = ? AND client_id = ?",
            [selected_id, client_id],
        ).fetchone()
        conn2.close()
        if row:
            st.text_area("Full output", value=row[0], height=400, disabled=True)


if not openai_key:
    st.error(missing_secret_message("OPENAI_API_KEY", "Content generation"))
    render_page_link(st, "Settings.py", "Open Settings")
    st.info("History is still available below. Generation, extraction, embeddings, and semantic reference search need OpenAI.")
    st.markdown("---")
    _render_history()
else:
    # ── Generation controls ──────────────────────────────────────────────────
    col_topic, col_tone, col_angle = st.columns([2, 1, 3])

    with col_topic:
        topic = st.selectbox("Topic", topic_names(client_id))

    with col_tone:
        tone = st.selectbox("Tone", ["professional", "friendly", "bold", "educational"])

    with col_angle:
        angle = st.text_input(
            "Specific angle / hook idea",
            placeholder="e.g., ATS rejects tables & columns",
        )

    st.markdown("---")

    # ── Reference units ──────────────────────────────────────────────────────
    st.subheader("Reference Knowledge Units")
    st.caption("Search for reference insights to ground generation.")

    ref_search = st.text_input("Search for reference insights", placeholder="e.g., ATS keywords")
    selected_units = []

    if ref_search:
        try:
            results = semantic_search(
                ref_search,
                openai_key,
                client_id,
                topic_filter=topic,
                top_k=10,
            )
        except Exception:
            results = keyword_search(ref_search, client_id, topic, 10)

        if results:
            st.write(f"Found {len(results)} insights — select which to use as reference:")
            for r in results:
                checked = st.checkbox(
                    f"[{r.topic}/{r.content_type}] {r.text}",
                    key=f"ref_{r.unit_id}",
                )
                if checked:
                    selected_units.append(r)
        else:
            st.info("No matching insights. Generation can still proceed without selected references.")

    if selected_units:
        st.success(f"{len(selected_units)} reference units selected")

    st.markdown("---")

    # ── Generation tabs ──────────────────────────────────────────────────────
    gen_caption_tab, gen_hooks_tab, gen_script_tab, history_tab = st.tabs([
        "Caption", "Hooks", "Script", "History"
    ])

    with gen_caption_tab:
        st.subheader("Generate Instagram Caption")
        if st.button("Generate Caption", type="primary", key="btn_caption", disabled=pipeline_busy):
            if not angle.strip():
                st.warning("Add a specific angle above for the best results.")
            with st.spinner("Generating caption…"):
                from analysis.content_gen import generate_caption

                try:
                    result = generate_caption(
                        topic,
                        angle or topic,
                        tone,
                        selected_units,
                        openai_key,
                        client_id,
                        active_client,
                    )
                    st.success(f"Generated. Cost: ${result.cost_usd:.4f}, tokens: {result.tokens_used}.")
                    st.text_area("Caption", value=result.output_text, height=300, key="caption_output")
                    st.caption(f"Saved to DB — ID: `{result.gen_id}`")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    with gen_hooks_tab:
        st.subheader("Generate Opening Hooks")
        hook_count = st.number_input("Number of hooks", min_value=3, max_value=10, value=5)
        if st.button("Generate Hooks", type="primary", key="btn_hooks", disabled=pipeline_busy):
            with st.spinner("Generating hooks…"):
                from analysis.content_gen import generate_hooks

                try:
                    result = generate_hooks(
                        topic,
                        angle or topic,
                        selected_units,
                        openai_key,
                        hook_count,
                        client_id,
                        active_client,
                    )
                    st.success(f"Generated. Cost: ${result.cost_usd:.4f}.")
                    st.text_area("Hooks", value=result.output_text, height=250, key="hooks_output")
                    st.caption(f"Saved — ID: `{result.gen_id}`")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    with gen_script_tab:
        st.subheader("Generate Reel Script")
        duration_sec = st.slider("Duration (seconds)", 15, 90, 45, step=5)
        word_estimate = int(duration_sec / 60 * 130)
        st.caption(f"Target: ~{word_estimate} words at 130 wpm")

        if st.button("Generate Script", type="primary", key="btn_script", disabled=pipeline_busy):
            with st.spinner("Generating script…"):
                from analysis.content_gen import generate_script

                try:
                    result = generate_script(
                        topic,
                        duration_sec,
                        tone,
                        selected_units,
                        openai_key,
                        client_id,
                        active_client,
                    )
                    st.success(f"Generated. Cost: ${result.cost_usd:.4f}.")
                    st.text_area("Script", value=result.output_text, height=400, key="script_output")
                    st.caption(f"Saved — ID: `{result.gen_id}`")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    with history_tab:
        _render_history()
