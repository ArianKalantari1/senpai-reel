import streamlit as st
import pandas as pd

st.set_page_config(page_title="Content Studio", page_icon="✍️", layout="wide")
st.title("✍️ Content Studio")
st.caption("AI-powered creative engine — informed by your corpus, not limited by it")

from analysis.taxonomy import TOPICS
from analysis.search import get_hook_examples
from core.db import get_connection

try:
    groq_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    groq_key = ""

if not groq_key:
    st.error("⚠️ `GROQ_API_KEY` is not set in `.streamlit/secrets.toml`. Content generation requires it.")
    st.stop()

# ── Top controls ────────────────────────────────────────────────────────────────
col_topic, col_tone, col_angle = st.columns([2, 1, 3])

with col_topic:
    topic = st.selectbox("Topic", TOPICS)

with col_tone:
    tone = st.selectbox("Tone", ["professional", "friendly", "bold", "educational"])

with col_angle:
    angle = st.text_input(
        "Your angle / hook idea",
        placeholder="e.g., ATS silently rejects tables and columns",
        help="The creative brief. Everything the LLM generates will serve this idea.",
    )

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────────
ideas_tab, caption_tab, hooks_tab, script_tab, history_tab = st.tabs([
    "💡 Content Ideas", "📝 Caption", "🎣 Hooks", "🎬 Script", "📋 History"
])

# ── Ideas ────────────────────────────────────────────────────────────────────────
with ideas_tab:
    st.subheader("💡 What should I make next?")
    st.caption(
        "Scans your corpus and recommends fresh content angles you haven't made yet. "
        "No angle needed — the LLM proposes what to create."
    )
    n_ideas = st.number_input("Number of ideas", min_value=3, max_value=10, value=5, key="n_ideas")
    if st.button("🔍 Generate Content Ideas", type="primary", key="btn_ideas"):
        with st.spinner(f"Scanning corpus for '{topic}' ideas…"):
            from analysis.content_gen import generate_ideas
            try:
                result = generate_ideas(topic, n=n_ideas)
                st.success(f"Done! (${result.cost_usd:.4f}, {result.tokens_used} tokens)")
                st.markdown(result.output_text)
                st.caption(f"Saved — ID: `{result.gen_id}`")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ── Caption ──────────────────────────────────────────────────────────────────────
with caption_tab:
    st.subheader("📝 Generate Caption")
    st.caption(
        "Auto-pulls hook style + idea landscape from your corpus. "
        "Set your angle above — the LLM uses the corpus as creative fuel, not a script."
    )
    if not angle.strip():
        st.info("💡 Enter an angle above (e.g. 'most recruiters spend 6 seconds on your resume') for a focused caption.")

    if st.button("✨ Generate Caption", type="primary", key="btn_caption"):
        with st.spinner("Writing caption from your corpus…"):
            from analysis.content_gen import generate_caption
            try:
                result = generate_caption(topic, angle or topic, tone)
                st.success(f"Generated! (${result.cost_usd:.4f}, {result.tokens_used} tokens)")
                st.text_area("Caption", value=result.output_text, height=320, key="caption_output")
                st.caption(f"Saved — ID: `{result.gen_id}`")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ── Hooks ────────────────────────────────────────────────────────────────────────
with hooks_tab:
    st.subheader("🎣 Generate Hooks")
    hook_count = st.number_input("Number of hooks", min_value=3, max_value=10, value=5, key="hook_count")

    if st.button("✨ Generate Hooks", type="primary", key="btn_hooks"):
        if not angle.strip():
            st.warning("⚠️ Enter an angle above — hooks need a specific creative direction.")
        else:
            with st.spinner("Generating hooks from your corpus…"):
                from analysis.content_gen import generate_hooks
                try:
                    result = generate_hooks(topic, angle, count=hook_count)
                    st.success(f"Generated! (${result.cost_usd:.4f})")
                    st.text_area("Hooks", value=result.output_text, height=270, key="hooks_output")
                    st.caption(f"Saved — ID: `{result.gen_id}`")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    st.markdown("---")

    # ── Hook Remixer ─────────────────────────────────────────────────────────────
    with st.expander("🎛️ Hook Remixer — mix & match corpus hooks to spark new ones"):
        st.caption(
            "Pick 2-5 real hooks from your corpus that have the energy or structure you want. "
            "The LLM creates brand-new hooks that match the *technique* — not the words — "
            "and deliver your angle. Set your angle above first."
        )
        remix_count = st.number_input("Hooks to generate", min_value=3, max_value=10, value=5, key="remix_count")

        if st.button("🔄 Load corpus hooks for this topic", key="btn_load_hooks"):
            loaded = get_hook_examples(topic, limit=15)
            st.session_state["corpus_hooks"] = loaded
            st.session_state["corpus_hooks_topic"] = topic

        corpus_hooks = st.session_state.get("corpus_hooks", [])
        current_topic = st.session_state.get("corpus_hooks_topic", "")
        selected_remix: list = []

        if corpus_hooks:
            if current_topic != topic:
                st.info(f"Showing hooks loaded for '{current_topic}'. Click 'Load corpus hooks' to refresh for '{topic}'.")
            st.write(f"**{len(corpus_hooks)} hooks found** — tick the ones whose energy or structure you like:")
            for h in corpus_hooks:
                label = f"[{h.topic}] {h.text}"
                if st.checkbox(label, key=f"remix_{h.unit_id}"):
                    selected_remix.append(h)
        elif "corpus_hooks" in st.session_state:
            st.warning(
                f"No hook-type units found for '{topic}'. "
                "Try a different topic, or run more reels through the pipeline to build the hook library."
            )

        if selected_remix:
            st.success(f"✅ {len(selected_remix)} hook(s) selected as style reference")

        remix_disabled = len(selected_remix) == 0
        if st.button("🎛️ Remix These Hooks", type="primary", key="btn_remix", disabled=remix_disabled):
            if not angle.strip():
                st.warning("⚠️ Enter an angle above — the remixed hooks need a creative direction.")
            else:
                with st.spinner("Remixing…"):
                    from analysis.content_gen import generate_hooks
                    try:
                        result = generate_hooks(topic, angle, count=remix_count, remix_examples=selected_remix)
                        st.success(f"Generated! (${result.cost_usd:.4f})")
                        st.text_area("Remixed hooks", value=result.output_text, height=270, key="remix_output")
                        st.caption(f"Saved — ID: `{result.gen_id}`")
                    except Exception as e:
                        st.error(f"Remix failed: {e}")

# ── Script ───────────────────────────────────────────────────────────────────────
with script_tab:
    st.subheader("🎬 Generate Reel Script")
    st.caption("Auto-uses hook patterns + ideas from your corpus. Add an angle above for a focused script.")
    duration_sec = st.slider("Duration (seconds)", 15, 90, 45, step=5, key="duration")
    word_estimate = int(duration_sec / 60 * 130)
    st.caption(f"Target: ~{word_estimate} words at 130 wpm")

    if not angle.strip():
        st.info("💡 Enter an angle above to give the script a specific focus.")

    if st.button("✨ Generate Script", type="primary", key="btn_script"):
        with st.spinner("Writing script from your corpus…"):
            from analysis.content_gen import generate_script
            try:
                result = generate_script(topic, duration_sec, tone, angle=angle)
                st.success(f"Generated! (${result.cost_usd:.4f})")
                st.text_area("Script", value=result.output_text, height=420, key="script_output")
                st.caption(f"Saved — ID: `{result.gen_id}`")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ── History ───────────────────────────────────────────────────────────────────────
with history_tab:
    st.subheader("📋 Generation History")
    conn = get_connection()
    try:
        df = conn.execute(
            """
            SELECT gen_id, created_at, topic, content_type,
                   LEFT(output_text, 100) AS preview, model, tokens_used, cost_usd
            FROM generated_content
            ORDER BY created_at DESC
            LIMIT 50
            """
        ).df()
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        st.info("No generated content yet.")
    else:
        total_cost = df["cost_usd"].sum() if "cost_usd" in df else 0
        st.caption(f"{len(df)} items generated — total cost: ${total_cost:.4f}")
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "cost_usd": st.column_config.NumberColumn("Cost $", format="%.4f"),
                         "preview": st.column_config.TextColumn("Preview", width="large"),
                     })

        selected_id = st.selectbox("View full output", df["gen_id"].tolist(), key="history_select")
        if selected_id:
            conn2 = get_connection()
            row = conn2.execute(
                "SELECT output_text FROM generated_content WHERE gen_id = ?", [selected_id]
            ).fetchone()
            conn2.close()
            if row:
                st.text_area("Full output", value=row[0], height=400, disabled=True, key="history_full")

