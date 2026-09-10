import streamlit as st
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
        
        # Get file info
        try:
            file_size = os.path.getsize(video_path) / (1024 * 1024)
        except:
            file_size = 0
        
        # Extract metrics
        likes = metadata.get('likesCount', 0) or 0
        views = metadata.get('videoViewCount', 0) or 0
        plays = metadata.get('videoPlayCount', 0) or 0
        comments = metadata.get('commentsCount', 0) or 0
        duration = metadata.get('videoDuration', 0) or 0
        caption = metadata.get('caption', '') or ''
        owner = metadata.get('ownerUsername', '') or ''
        location = metadata.get('locationName', '') or ''
        timestamp = metadata.get('timestamp', '')
        
        # Calculate metrics
        engagement_rate = (likes / views * 100) if views > 0 else 0
        comments_rate = (comments / likes * 100) if likes > 0 else 0
        play_completion = (plays / views * 100) if views > 0 else 0
        
        # Content categorization
        caption_lower = caption.lower()
        location_lower = location.lower()
        
        if any(word in caption_lower + location_lower for word in ['food', 'restaurant', 'bar', 'drink', 'ramen', 'eat']):
            category = 'Food & Beverage'
        elif any(word in caption_lower for word in ['party', 'celebration', 'anniversary', 'event']):
            category = 'Events & Celebrations'
        elif any(word in caption_lower for word in ['style', 'aesthetic', 'vibe', 'mood']):
            category = 'Lifestyle'
        elif location_lower:
            category = 'Location-based'
        else:
            category = 'General Content'
        
        # Virality prediction
        virality_score = 0
        if engagement_rate > 5: virality_score += 2
        if comments_rate > 10: virality_score += 1
        if 15 <= duration <= 60: virality_score += 2
        if len(caption) > 100: virality_score += 1
        if '@' in caption: virality_score += 1
        if location: virality_score += 1
        if play_completion > 80: virality_score += 2
        virality_score = min(10, virality_score)
        
        result = {
            'short_code': short_code,
            'owner': owner,
            'category': category,
            'file_size_mb': round(file_size, 2),
            'duration': duration,
            'likes': likes,
            'views': views,
            'plays': plays,
            'comments': comments,
            'engagement_rate': round(engagement_rate, 2),
            'comments_rate': round(comments_rate, 2),
            'play_completion': round(play_completion, 2),
            'virality_potential': virality_score,
            'caption_length': len(caption),
            'has_location': bool(location),
            'mentions_count': len(metadata.get('mentions', [])),
            'hashtags_count': len(metadata.get('hashtags', [])),
            'timestamp': timestamp[:10] if timestamp else 'Unknown',
            'location': location or 'Not specified'
        }
        
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
    st.metric("👁️ Total Views", f"{df['views'].sum():,}")
with col3:
    st.metric("❤️ Total Likes", f"{df['likes'].sum():,}")
with col4:
    st.metric("⭐ Avg Engagement", f"{df['engagement_rate'].mean():.1f}%")
with col5:
    st.metric("🔥 Avg Virality", f"{df['virality_potential'].mean():.1f}/10")

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
    top_engagement = df.nlargest(5, 'engagement_rate')[['short_code', 'owner', 'engagement_rate', 'category']]
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
    min_engagement = st.slider("Minimum Engagement %", 0, int(df['engagement_rate'].max()), 0)
with col3:
    min_virality = st.slider("Minimum Virality Score", 0, 10, 0)

# Apply filters
filtered_df = df.copy()
if category_filter != "All":
    filtered_df = filtered_df[filtered_df['category'] == category_filter]
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

# Insights and recommendations
st.markdown("---")
st.subheader("💡 AI Insights & Recommendations")

# Calculate insights
best_category = df.groupby('category')['engagement_rate'].mean().idxmax()
best_category_score = df.groupby('category')['engagement_rate'].mean().max()

duration_analysis = df.groupby(df['duration'] < 30)['engagement_rate'].mean()
location_analysis = df.groupby('has_location')['engagement_rate'].mean()

col1, col2 = st.columns(2)

with col1:
    st.write("🎯 **Key Insights:**")
    st.write(f"• **Best performing category:** {best_category} ({best_category_score:.1f}% avg engagement)")
    
    if len(duration_analysis) == 2:
        short_eng = duration_analysis[True] if True in duration_analysis else 0
        long_eng = duration_analysis[False] if False in duration_analysis else 0
        st.write(f"• **Duration impact:** Short videos ({short_eng:.1f}%) vs Long videos ({long_eng:.1f}%)")
    
    if len(location_analysis) == 2:
        with_loc = location_analysis[True] if True in location_analysis else 0
        without_loc = location_analysis[False] if False in location_analysis else 0
        st.write(f"• **Location tags:** With location ({with_loc:.1f}%) vs Without ({without_loc:.1f}%)")

with col2:
    st.write("🚀 **Recommendations:**")
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
