import streamlit as st
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from archive.video_analysis_metrics import (
    build_video_analysis_result,
    format_count_or_unknown,
    format_percent_or_unknown,
    known_mean,
    known_total,
    video_file_size_mb,
)
from core.client_context import render_client_selector
from core.db import get_connection, init_db

st.set_page_config(page_title="AI Video Analysis", page_icon="🤖", layout="wide")
st.title("🤖 AI Video Analysis Dashboard")
st.write("AI-powered insights from your Instagram reel videos")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]

@st.cache_data
def analyze_videos(client_id):
    """Run AI analysis on downloaded videos"""
    conn = get_connection()
    downloads_path = "downloads"
    
    if not os.path.exists(downloads_path):
        conn.close()
        return []
    
    video_files = [f for f in os.listdir(downloads_path) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        conn.close()
        return []
    
    results = []
    
    for video_file in video_files:
        video_path = os.path.join(downloads_path, video_file)
        short_code = os.path.splitext(video_file)[0].split('_')[-1]
        
        try:
            # Get metadata
            metadata_result = conn.execute("""
                SELECT raw FROM raw_scrapes
                WHERE client_id = ? AND raw LIKE ?
                LIMIT 1
            """, [client_id, f'%{short_code}%']).fetchone()
            
            if not metadata_result:
                continue
                
            metadata = json.loads(metadata_result[0])
            
        except Exception as e:
            continue
        
        result = build_video_analysis_result(
            metadata,
            short_code,
            file_size_mb=video_file_size_mb(video_path),
        )
        
        results.append(result)
    
    conn.close()
    return results

# Run analysis
with st.spinner("🤖 Running AI analysis on videos..."):
    analysis_results = analyze_videos(client_id)

if not analysis_results:
    st.warning("⚠️ No videos found for analysis. Please download some videos first!")
    st.info("💡 Use the Data Viewer page to download videos, then come back here for AI analysis.")
    st.stop()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(analysis_results)

# Main metrics
st.subheader("📊 Overview Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📹 Total Videos", len(df))
with col2:
    st.metric("👁️ Total Views", format_count_or_unknown(known_total(df['views'])))
with col3:
    st.metric("❤️ Total Likes", format_count_or_unknown(known_total(df['likes'])))
with col4:
    st.metric("⭐ Avg Engagement", format_percent_or_unknown(known_mean(df['engagement_rate'])))
with col5:
    avg_virality = known_mean(df['virality_potential'])
    st.metric("🔥 Avg Virality", "unknown" if avg_virality is None else f"{avg_virality:.1f}/10")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Content Categories")
    category_data = df.groupby('category').agg({
        'short_code': 'count',
        'engagement_rate': 'mean',
        'virality_potential': 'mean'
    }).round(2)
    category_data.columns = ['Videos', 'Avg Engagement %', 'Avg Virality']
    
    fig_cat = px.pie(
        values=category_data['Videos'], 
        names=category_data.index,
        title="Distribution of Content Types"
    )
    st.plotly_chart(fig_cat, width="stretch")

with col2:
    st.subheader("🎯 Engagement vs Virality")
    fig_scatter = px.scatter(
        df, 
        x='engagement_rate', 
        y='virality_potential',
        size='views',
        color='category',
        hover_data=['short_code', 'owner'],
        title="Engagement Rate vs Virality Potential"
    )
    fig_scatter.update_xaxis(title="Engagement Rate (%)")
    fig_scatter.update_yaxis(title="Virality Potential (0-10)")
    st.plotly_chart(fig_scatter, width="stretch")

# Performance analysis
st.subheader("🏆 Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("**Top Performers by Engagement:**")
    known_engagement_df = df.dropna(subset=['engagement_rate'])
    if known_engagement_df.empty:
        st.write("No known engagement rates to rank.")
    else:
        top_engagement = known_engagement_df.nlargest(5, 'engagement_rate')[['short_code', 'owner', 'engagement_rate', 'category']]
        for idx, row in top_engagement.iterrows():
            st.write(f"🥇 **{row['short_code']}** (@{row['owner']}) - {row['engagement_rate']:.1f}% ({row['category']})")

with col2:
    st.write("**Highest Virality Potential:**")
    top_virality = df.nlargest(5, 'virality_potential')[['short_code', 'owner', 'virality_potential', 'category']]
    for idx, row in top_virality.iterrows():
        st.write(f"🔥 **{row['short_code']}** (@{row['owner']}) - {row['virality_potential']}/10 ({row['category']})")

# Detailed analysis
st.markdown("---")
st.subheader("🔍 Detailed Video Analysis")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    category_filter = st.selectbox("Filter by Category", ["All"] + list(df['category'].unique()))
with col2:
    known_engagement = df['engagement_rate'].dropna()
    if known_engagement.empty:
        min_engagement = None
        st.caption("Engagement filter unavailable: engagement rates are unknown.")
    else:
        min_engagement = st.slider("Minimum Engagement %", 0, int(known_engagement.max()), 0)
with col3:
    min_virality = st.slider("Minimum Virality Score", 0, 10, 0)

# Apply filters
filtered_df = df.copy()
if category_filter != "All":
    filtered_df = filtered_df[filtered_df['category'] == category_filter]
if min_engagement is not None:
    filtered_df = filtered_df[filtered_df['engagement_rate'] >= min_engagement]
filtered_df = filtered_df[filtered_df['virality_potential'] >= min_virality]

# Display filtered results
display_columns = ['short_code', 'owner', 'category', 'engagement_rate', 'virality_potential', 
                  'views', 'likes', 'comments', 'duration', 'has_location']

st.dataframe(
    filtered_df[display_columns].round(2),
    width="stretch",
    column_config={
        "short_code": "Short Code",
        "owner": "Owner",
        "category": "Category", 
        "engagement_rate": st.column_config.NumberColumn("Engagement %", format="%.1f%%"),
        "virality_potential": st.column_config.NumberColumn("Virality", format="%.0f/10"),
        "views": st.column_config.NumberColumn("Views", format="%d"),
        "likes": st.column_config.NumberColumn("Likes", format="%d"),
        "comments": st.column_config.NumberColumn("Comments", format="%d"),
        "duration": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
        "has_location": st.column_config.CheckboxColumn("Has Location")
    }
)
st.caption("Blank numeric cells mean the source did not supply enough data to calculate the value.")

# Insights and recommendations
st.markdown("---")
st.subheader("💡 AI Insights & Recommendations")

# Calculate insights
category_engagement = df.dropna(subset=['engagement_rate']).groupby('category')['engagement_rate'].mean()
duration_known = df.dropna(subset=['duration', 'engagement_rate'])
duration_analysis = duration_known.groupby(duration_known['duration'] < 30)['engagement_rate'].mean()
location_analysis = df.dropna(subset=['engagement_rate']).groupby('has_location')['engagement_rate'].mean()

col1, col2 = st.columns(2)

with col1:
    st.write("🎯 **Key Insights:**")
    if category_engagement.empty:
        st.write("• **Best performing category:** unknown (engagement rates are unavailable)")
    else:
        best_category = category_engagement.idxmax()
        best_category_score = category_engagement.max()
        st.write(f"• **Best performing category:** {best_category} ({best_category_score:.1f}% avg engagement)")
    
    if len(duration_analysis) == 2:
        short_eng = duration_analysis.get(True)
        long_eng = duration_analysis.get(False)
        st.write(
            "• **Duration impact:** "
            f"Short videos ({format_percent_or_unknown(short_eng)}) vs "
            f"Long videos ({format_percent_or_unknown(long_eng)})"
        )
    else:
        st.write("• **Duration impact:** unknown (duration or engagement is unavailable)")
    
    if len(location_analysis) == 2:
        with_loc = location_analysis.get(True)
        without_loc = location_analysis.get(False)
        st.write(
            "• **Location tags:** "
            f"With location ({format_percent_or_unknown(with_loc)}) vs "
            f"Without ({format_percent_or_unknown(without_loc)})"
        )
    else:
        st.write("• **Location tags:** unknown (engagement is unavailable)")

with col2:
    st.write("🚀 **Recommendations:**")
    if category_engagement.empty:
        st.write("• Focus category: **unknown** until engagement rates are available")
    else:
        st.write(f"• Focus on **{best_category.lower()}** content")
    st.write("• Optimize video length: **15-60 seconds**")
    st.write("• Always add **location tags** when relevant")
    st.write("• Use **detailed captions** (100+ characters)")
    st.write("• Include **user mentions** to increase reach")

# Technical details
with st.expander("🔧 Technical Analysis Details"):
    st.write("**Analysis Methods Used:**")
    st.write("- Content categorization based on caption and location keywords")
    st.write("- Engagement rate calculation: (Likes ÷ Views) × 100")
    st.write("- Virality prediction using multiple factors:")
    st.write("  - Engagement rate > 5% (+2 points)")
    st.write("  - Comment rate > 10% (+1 point)")
    st.write("  - Optimal duration 15-60s (+2 points)")
    st.write("  - Detailed caption >100 chars (+1 point)")
    st.write("  - User mentions (+1 point)")
    st.write("  - Location tag (+1 point)")
    st.write("  - High play completion >80% (+2 points)")

# Export analysis
if st.button("📥 Export Analysis Results"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"ai_video_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
