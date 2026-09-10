import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from core.client_context import render_client_selector
from core.db import get_connection, init_db

st.set_page_config(page_title="Analytics", page_icon="🤖", layout="wide")

st.title("AI-Powered Analytics")
st.write("Advanced analytics and insights from your Instagram reel data")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]

# Database connection
@st.cache_resource
def get_db_connection():
    return get_connection()

def load_data(query, params=None):
    conn = get_db_connection()
    if params:
        return conn.execute(query, params).df()
    else:
        return conn.execute(query).df()

# Check if we have data
try:
    reels_count = load_data(
        "SELECT COUNT(*) as count FROM reels WHERE client_id = ?",
        [client_id],
    ).iloc[0]['count']
    if reels_count == 0:
        st.warning("⚠️ No processed reels found. Please scrape some data and process it first.")
        st.stop()
except:
    st.error("❌ Database connection failed. Please run the scraper first.")
    st.stop()

# Sidebar - Profile Selection
st.sidebar.header("🔧 Analytics Options")

try:
    profiles_df = load_data(
        "SELECT DISTINCT profile FROM reels WHERE client_id = ? ORDER BY profile",
        [client_id],
    )
    available_profiles = ['All'] + profiles_df['profile'].tolist() if not profiles_df.empty else ['All']
except:
    available_profiles = ['All']

selected_profile = st.sidebar.selectbox("Select Profile for Analysis", available_profiles)

# Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Analytics", "⏰ Timing Insights", "💬 Engagement Analysis", "🏷️ Tagging Patterns"])

with tab1:
    st.header("📊 Content Performance Analytics")
    
    try:
        # Get performance data
        query = """
            SELECT reel_id, shortcode, caption, likes, views, video_play_count, 
                   comments_count, duration, timestamp, is_pinned
            FROM reels 
            WHERE client_id = ? AND likes IS NOT NULL AND views IS NOT NULL
        """
        params = [client_id]
        if selected_profile != 'All':
            query += " AND profile = ?"
            params.append(selected_profile)
        
        performance_data = load_data(query, params)
        
        if not performance_data.empty:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_likes = performance_data['likes'].sum()
                st.metric("Total Likes", f"{total_likes:,}")
            
            with col2:
                total_views = performance_data['views'].sum()
                st.metric("Total Views", f"{total_views:,}")
            
            with col3:
                avg_engagement = (performance_data['likes'] / performance_data['views'].replace(0, 1)).mean() * 100
                st.metric("Avg Engagement Rate", f"{avg_engagement:.2f}%")
            
            with col4:
                total_comments = performance_data['comments_count'].sum()
                st.metric("Total Comments", f"{total_comments:,}")
            
            # Performance charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Top Performing Reels by Likes")
                top_reels = performance_data.nlargest(10, 'likes')
                fig = px.bar(top_reels, x='shortcode', y='likes', 
                           title="Top 10 Reels by Likes",
                           hover_data=['views', 'comments_count'])
                fig.update_xaxis(title="Reel Shortcode")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")
            
            with chart_col2:
                st.subheader("Likes vs Views Correlation")
                fig = px.scatter(performance_data, x='views', y='likes',
                               size='comments_count', hover_data=['shortcode'],
                               title="Engagement Correlation")
                fig.update_xaxis(title="Views")
                fig.update_yaxis(title="Likes")
                st.plotly_chart(fig, width="stretch")
            
            # Duration analysis
            if 'duration' in performance_data.columns:
                st.subheader("📹 Content Duration Analysis")
                duration_col1, duration_col2 = st.columns(2)
                
                with duration_col1:
                    # Duration distribution
                    fig = px.histogram(performance_data, x='duration', nbins=20,
                                     title="Video Duration Distribution")
                    fig.update_xaxis(title="Duration (seconds)")
                    st.plotly_chart(fig, width="stretch")
                
                with duration_col2:
                    # Duration vs engagement
                    performance_data['engagement_rate'] = (performance_data['likes'] / performance_data['views'].replace(0, 1)) * 100
                    fig = px.scatter(performance_data, x='duration', y='engagement_rate',
                                   title="Duration vs Engagement Rate",
                                   trendline="ols")
                    fig.update_xaxis(title="Duration (seconds)")
                    fig.update_yaxis(title="Engagement Rate (%)")
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("No performance data available for analysis.")
            
    except Exception as e:
        st.error(f"Error in performance analysis: {e}")

with tab2:
    st.header("⏰ Posting Time & Timing Insights")
    
    try:
        # Get timing data
        query = """
            SELECT timestamp, likes, views, comments_count,
                   EXTRACT(HOUR FROM timestamp) as hour,
                   EXTRACT(DOW FROM timestamp) as day_of_week,
                   EXTRACT(MONTH FROM timestamp) as month
            FROM reels 
            WHERE client_id = ? AND timestamp IS NOT NULL AND likes IS NOT NULL
        """
        params = [client_id]
        if selected_profile != 'All':
            query += " AND profile = ?"
            params.append(selected_profile)
        
        timing_data = load_data(query, params)
        
        if not timing_data.empty:
            # Day of week mapping
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            timing_data['day_name'] = timing_data['day_of_week'].map(lambda x: day_names[int(x)])
            
            timing_col1, timing_col2 = st.columns(2)
            
            with timing_col1:
                st.subheader("Best Posting Hours")
                hourly_performance = timing_data.groupby('hour').agg({
                    'likes': 'mean',
                    'views': 'mean',
                    'comments_count': 'mean'
                }).round(2)
                
                fig = px.line(hourly_performance, y=['likes', 'views'], 
                            title="Average Engagement by Hour of Day")
                fig.update_xaxis(title="Hour (24h format)")
                fig.update_yaxis(title="Average Count")
                st.plotly_chart(fig, width="stretch")
            
            with timing_col2:
                st.subheader("Best Posting Days")
                daily_performance = timing_data.groupby('day_name').agg({
                    'likes': 'mean',
                    'views': 'mean',
                    'comments_count': 'mean'
                }).round(2)
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_performance = daily_performance.reindex(day_order)
                
                fig = px.bar(daily_performance, y=['likes', 'views'],
                           title="Average Engagement by Day of Week")
                fig.update_xaxis(title="Day of Week")
                fig.update_yaxis(title="Average Count")
                st.plotly_chart(fig, width="stretch")
            
            # Best performing time slots
            st.subheader("🎯 Optimal Posting Recommendations")
            
            # Calculate engagement rate by hour
            timing_data['engagement_rate'] = (timing_data['likes'] / timing_data['views'].replace(0, 1)) * 100
            best_hours = timing_data.groupby('hour')['engagement_rate'].mean().nlargest(5)
            best_days = timing_data.groupby('day_name')['engagement_rate'].mean().nlargest(3)
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.write("**🕐 Best Hours to Post:**")
                for hour, rate in best_hours.items():
                    st.write(f"• {int(hour):02d}:00 - Avg engagement: {rate:.2f}%")
            
            with rec_col2:
                st.write("**📅 Best Days to Post:**")
                for day, rate in best_days.items():
                    st.write(f"• {day} - Avg engagement: {rate:.2f}%")
                    
        else:
            st.info("No timing data available for analysis.")
            
    except Exception as e:
        st.error(f"Error in timing analysis: {e}")

with tab3:
    st.header("💬 Engagement & Comments Analysis")
    
    try:
        # Get comments data
        comments_query = """
            SELECT c.reel_id, c.username, c.text, c.likes as comment_likes,
                   r.likes as reel_likes, r.views, r.shortcode, r.profile
            FROM comments c
            JOIN reels r ON c.reel_id = r.reel_id
            WHERE c.client_id = ? AND r.client_id = ?
        """
        params = [client_id, client_id]
        if selected_profile != 'All':
            comments_query += " AND r.profile = ?"
            params.append(selected_profile)
            
        comments_data = load_data(comments_query, params)
        
        if not comments_data.empty:
            engagement_col1, engagement_col2 = st.columns(2)
            
            with engagement_col1:
                st.subheader("👥 Top Commenters")
                top_commenters = comments_data['username'].value_counts().head(10)
                fig = px.bar(x=top_commenters.values, y=top_commenters.index, 
                           orientation='h', title="Most Active Commenters")
                fig.update_xaxis(title="Number of Comments")
                fig.update_yaxis(title="Username")
                st.plotly_chart(fig, width="stretch")
            
            with engagement_col2:
                st.subheader("💭 Comment Sentiment Overview")
                # Simple sentiment analysis based on text length and keywords
                comments_data['text_length'] = comments_data['text'].str.len()
                avg_comment_length = comments_data['text_length'].mean()
                
                # Count positive indicators (emojis and positive words)
                positive_indicators = ['❤️', '🔥', '👏', '😍', '🥰', 'love', 'amazing', 'great', 'awesome', 'perfect']
                comments_data['positive_count'] = comments_data['text'].str.lower().str.count('|'.join(positive_indicators))
                
                avg_positivity = comments_data['positive_count'].mean()
                
                st.metric("Avg Comment Length", f"{avg_comment_length:.0f} chars")
                st.metric("Avg Positive Indicators", f"{avg_positivity:.2f}")
                
                # Show sample comments
                st.write("**Recent Comments Sample:**")
                sample_comments = comments_data.head(5)[['username', 'text', 'comment_likes']]
                for _, row in sample_comments.iterrows():
                    st.write(f"**@{row['username']}** (♥️ {row['comment_likes']}): {row['text'][:100]}...")
        else:
            st.info("No comments data available for analysis.")
            
    except Exception as e:
        st.error(f"Error in engagement analysis: {e}")

with tab4:
    st.header("🏷️ Tagging Patterns & Network Analysis")
    
    try:
        # Get tagging data
        tagging_query = """
            SELECT t.username, t.full_name, r.profile, r.shortcode, r.likes, r.views
            FROM tagged_users t
            JOIN reels r ON t.reel_id = r.reel_id
            WHERE t.client_id = ? AND r.client_id = ?
        """
        params = [client_id, client_id]
        if selected_profile != 'All':
            tagging_query += " AND r.profile = ?"
            params.append(selected_profile)
            
        tagging_data = load_data(tagging_query, params)
        
        if not tagging_data.empty:
            tag_col1, tag_col2 = st.columns(2)
            
            with tag_col1:
                st.subheader("🏷️ Most Tagged Users")
                most_tagged = tagging_data['username'].value_counts().head(15)
                fig = px.bar(x=most_tagged.values, y=most_tagged.index, 
                           orientation='h', title="Most Frequently Tagged Users")
                fig.update_xaxis(title="Times Tagged")
                fig.update_yaxis(title="Username")
                st.plotly_chart(fig, width="stretch")
            
            with tag_col2:
                st.subheader("📈 Tagging Impact on Performance")
                # Analyze if tagging affects performance
                reel_tag_counts = tagging_data.groupby('shortcode').agg({
                    'username': 'count',
                    'likes': 'first',
                    'views': 'first'
                }).rename(columns={'username': 'tag_count'})
                
                if len(reel_tag_counts) > 1:
                    fig = px.scatter(reel_tag_counts, x='tag_count', y='likes',
                                   size='views', title="Tags vs Likes Correlation",
                                   trendline="ols")
                    fig.update_xaxis(title="Number of Tagged Users")
                    fig.update_yaxis(title="Likes")
                    st.plotly_chart(fig, width="stretch")
            
            # Tagging network insights
            st.subheader("🌐 Tagging Network Insights")
            unique_tagged = tagging_data['username'].nunique()
            avg_tags_per_reel = tagging_data.groupby('shortcode')['username'].count().mean()
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            with insight_col1:
                st.metric("Unique Tagged Users", unique_tagged)
            with insight_col2:
                st.metric("Avg Tags per Reel", f"{avg_tags_per_reel:.1f}")
            with insight_col3:
                collaboration_rate = (tagging_data.groupby('shortcode')['username'].count() > 0).mean() * 100
                st.metric("Collaboration Rate", f"{collaboration_rate:.1f}%")
                
        else:
            st.info("No tagging data available for analysis.")
            
    except Exception as e:
        st.error(f"Error in tagging analysis: {e}")

# AI Insights Section
st.markdown("---")
st.header("🤖 AI-Generated Insights")

try:
    # Get summary stats for AI insights
    summary_query = """
        SELECT 
            COUNT(*) as total_reels,
            AVG(likes) as avg_likes,
            AVG(views) as avg_views,
            AVG(comments_count) as avg_comments,
            AVG(duration) as avg_duration
        FROM reels
        WHERE client_id = ? AND likes IS NOT NULL AND views IS NOT NULL
    """
    params = [client_id]
    if selected_profile != 'All':
        summary_query += " AND profile = ?"
        params.append(selected_profile)
        
    summary_data = load_data(summary_query, params)
    
    if not summary_data.empty and summary_data.iloc[0]['total_reels'] > 0:
        stats = summary_data.iloc[0]
        
        insights = []
        
        # Engagement rate insight
        engagement_rate = (stats['avg_likes'] / stats['avg_views']) * 100 if stats['avg_views'] > 0 else 0
        if engagement_rate > 5:
            insights.append(f"🎉 **High Engagement**: Your content has an excellent engagement rate of {engagement_rate:.2f}%")
        elif engagement_rate > 2:
            insights.append(f"👍 **Good Engagement**: Your engagement rate is {engagement_rate:.2f}%, which is above average")
        else:
            insights.append(f"📈 **Growth Opportunity**: Your engagement rate is {engagement_rate:.2f}%. Consider posting at optimal times or using trending hashtags")
        
        # Content length insight
        if stats['avg_duration']:
            if stats['avg_duration'] < 15:
                insights.append("⚡ **Short & Sweet**: Your reels are perfectly sized for quick consumption - great for engagement!")
            elif stats['avg_duration'] > 45:
                insights.append("📚 **Storyteller**: Your longer-form content suggests deep storytelling - perfect for building strong connections")
            else:
                insights.append("🎯 **Perfect Length**: Your reel duration is in the sweet spot for maximum retention")
        
        # Comments ratio insight
        if stats['avg_comments'] and stats['avg_likes']:
            comment_ratio = (stats['avg_comments'] / stats['avg_likes']) * 100
            if comment_ratio > 5:
                insights.append("💬 **Community Builder**: High comment-to-like ratio shows you're building a engaged community!")
            else:
                insights.append("💡 **Engagement Tip**: Consider asking questions in your captions to boost comments")
        
        # Display insights
        for insight in insights:
            st.write(insight)
            
        # Recommendations
        st.subheader("🎯 AI Recommendations")
        
        recommendations = [
            "📅 **Consistency**: Post regularly to maintain audience engagement",
            "🏷️ **Hashtags**: Use 5-10 relevant hashtags per post",
            "⏰ **Timing**: Post when your audience is most active",
            "💬 **Engage**: Respond to comments within 24 hours",
            "🎨 **Quality**: Maintain good lighting and clear audio",
            "📊 **Analyze**: Review these analytics weekly to optimize your strategy"
        ]
        
        for rec in recommendations:
            st.write(rec)
            
    else:
        st.info("Not enough data for AI insights. Scrape more content to see personalized recommendations!")
        
except Exception as e:
    st.error(f"Error generating AI insights: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🤖 This AI analytics page provides automated insights based on your scraped Instagram data.")
