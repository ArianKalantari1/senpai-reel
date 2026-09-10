import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Analytics Dashboard")

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from analysis.analytics import (
    get_creator_leaderboard,
    get_topic_distribution,
    get_content_gap_matrix,
    get_top_posts,
    get_hashtag_intelligence,
)
from analysis.taxonomy import TOPICS
from core.client_context import render_client_selector
from core.db import init_db

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Creator performance, content topics, gaps, and trends for {active_client['name']}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboard",
    "🏷️ Topic Distribution",
    "🗺️ Content Gap Map",
    "🔥 Top Reels",
    "#️⃣ Hashtags",
])

# ── Tab 1: Creator Leaderboard ─────────────────────────────────────────────────
with tab1:
    st.subheader("Creator Leaderboard")
    df_lb = get_creator_leaderboard(client_id, 30)
    if df_lb.empty:
        st.info("No posts scraped yet.")
    else:
        if HAS_PLOTLY:
            fig = px.bar(
                df_lb.head(15),
                x="username",
                y="avg_engagement",
                color="total_posts",
                title="Average Engagement Rate by Creator",
                labels={"avg_engagement": "Avg Engagement %", "username": "Creator"},
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, width="stretch")
        st.dataframe(df_lb, width="stretch", hide_index=True,
                     column_config={
                         "avg_engagement": st.column_config.NumberColumn("Avg Eng %", format="%.2f"),
                         "max_engagement": st.column_config.NumberColumn("Max Eng %", format="%.2f"),
                         "total_views": st.column_config.NumberColumn("Total Views", format="%d"),
                     })

# ── Tab 2: Topic Distribution ──────────────────────────────────────────────────
with tab2:
    st.subheader("Topic Distribution (from extracted knowledge units)")
    df_topics = get_topic_distribution(client_id)
    if df_topics.empty:
        st.info("No knowledge units yet — run extraction pipeline first.")
    else:
        col_pie, col_bar = st.columns(2)
        if HAS_PLOTLY:
            with col_pie:
                fig_pie = px.pie(df_topics, names="topic", values="unit_count",
                                  title="Topics by Unit Count", hole=0.3)
                st.plotly_chart(fig_pie, width="stretch")
            with col_bar:
                fig_bar = px.bar(df_topics, x="topic", y="unit_count",
                                  title="Units per Topic",
                                  labels={"unit_count": "# Units", "topic": "Topic"},
                                  color="unit_count", color_continuous_scale="Teal")
                st.plotly_chart(fig_bar, width="stretch")
        else:
            st.dataframe(df_topics, width="stretch", hide_index=True)

# ── Tab 3: Content Gap Map ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Content Gap Map — Topic × Content Type")
    st.caption("Low numbers = opportunity (content type rarely covered for that topic)")
    pivot = get_content_gap_matrix(client_id)
    if pivot.empty:
        st.info("No knowledge units yet.")
    else:
        if HAS_PLOTLY:
            fig_heat = px.imshow(
                pivot,
                title="Message Unit Density (dark = more content)",
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
            )
            fig_heat.update_layout(height=500)
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.dataframe(pivot, width="stretch")

# ── Tab 4: Top Reels ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("Top Performing Reels")
    topic_sel = st.selectbox("Filter by topic", ["All"] + TOPICS)
    df_top = get_top_posts(client_id, topic_sel, 50)
    if df_top.empty:
        st.info("No posts yet.")
    else:
        st.dataframe(
            df_top,
            width="stretch",
            hide_index=True,
            column_config={
                "video_url": st.column_config.LinkColumn("Watch", display_text="🎥"),
                "engagement_rate": st.column_config.NumberColumn("Eng %", format="%.2f"),
                "caption": st.column_config.TextColumn("Caption", width="large"),
            },
        )

# ── Tab 5: Hashtags ────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Hashtag Intelligence")
    df_ht = get_hashtag_intelligence(client_id, 40)
    if df_ht.empty:
        st.info("No hashtag data yet.")
    else:
        if HAS_PLOTLY:
            fig_ht = px.bar(
                df_ht.head(25), x="hashtag", y="count",
                title="Top 25 Hashtags",
                labels={"count": "Usage count", "hashtag": "Hashtag"},
                color="count", color_continuous_scale="Purples",
            )
            fig_ht.update_xaxes(tickangle=45)
            st.plotly_chart(fig_ht, width="stretch")
        st.dataframe(df_ht, width="stretch", hide_index=True)
