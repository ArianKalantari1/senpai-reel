import streamlit as st
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
import re
from datetime import datetime

from core.client_context import render_client_selector
from core.db import get_connection, init_db

st.set_page_config(page_title="Graph Network Analysis", page_icon="🕸️", layout="wide")
st.title("🕸️ Instagram Content Graph Network")
st.write("Free graph analysis - discover content relationships and influence patterns without expensive AI APIs")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]

@st.cache_data
def build_content_network(client_id):
    """Build the content network graph"""
    conn = get_connection()
    
    # Get all posts
    results = conn.execute(
        "SELECT raw FROM raw_scrapes WHERE client_id = ? ORDER BY scraped_at DESC",
        [client_id],
    ).fetchall()
    conn.close()
    
    posts = []
    graph = nx.Graph()
    
    for (raw,) in results:
        try:
            data = json.loads(raw)
            
            caption = data.get('caption', '') or ''
            hashtags = re.findall(r'#\w+', caption.lower())
            mentions = re.findall(r'@\w+', caption.lower())
            
            post_data = {
                'short_code': data.get('shortCode', ''),
                'owner': data.get('ownerUsername', ''),
                'caption': caption,
                'hashtags': hashtags,
                'mentions': mentions,
                'location': data.get('locationName', ''),
                'likes': data.get('likesCount', 0) or 0,
                'views': data.get('videoViewCount', 0) or 0,
                'comments': data.get('commentsCount', 0) or 0,
                'duration': data.get('videoDuration', 0) or 0,
                'timestamp': data.get('timestamp', '')
            }
            posts.append(post_data)
            
            # Add node to graph
            engagement_rate = (post_data['likes']/post_data['views']*100) if post_data['views'] > 0 else 0
            graph.add_node(
                post_data['short_code'],
                owner=post_data['owner'],
                engagement=engagement_rate,
                likes=post_data['likes'],
                views=post_data['views'],
                hashtag_count=len(hashtags),
                mention_count=len(mentions)
            )
            
        except Exception as e:
            continue
    
    # Create edges based on similarity
    for i, post1 in enumerate(posts):
        for post2 in posts[i+1:]:
            similarity = calculate_similarity(post1, post2)
            if similarity > 0.3:
                graph.add_edge(post1['short_code'], post2['short_code'], weight=similarity)
    
    return posts, graph

def calculate_similarity(post1, post2):
    """Calculate content similarity"""
    score = 0
    
    # Same owner
    if post1['owner'] == post2['owner']:
        score += 0.5
    
    # Shared hashtags
    common_hashtags = set(post1['hashtags']) & set(post2['hashtags'])
    if common_hashtags and (post1['hashtags'] or post2['hashtags']):
        score += 0.4 * len(common_hashtags) / max(len(post1['hashtags']), len(post2['hashtags']), 1)
    
    # Similar engagement
    eng1 = (post1['likes']/post1['views']*100) if post1['views'] > 0 else 0
    eng2 = (post2['likes']/post2['views']*100) if post2['views'] > 0 else 0
    if abs(eng1 - eng2) < 5:
        score += 0.2
    
    # Cross mentions
    if any(mention in post2['caption'].lower() for mention in post1['mentions']):
        score += 0.3
    
    return min(1.0, score)

@st.cache_data
def analyze_hashtags(posts):
    """Analyze hashtag performance and relationships"""
    hashtag_data = defaultdict(lambda: {'posts': 0, 'total_likes': 0, 'total_views': 0, 'owners': set()})
    
    for post in posts:
        for hashtag in post['hashtags']:
            hashtag_data[hashtag]['posts'] += 1
            hashtag_data[hashtag]['total_likes'] += post['likes']
            hashtag_data[hashtag]['total_views'] += post['views']
            hashtag_data[hashtag]['owners'].add(post['owner'])
    
    # Convert to dataframe
    hashtag_df = []
    for hashtag, data in hashtag_data.items():
        if data['posts'] >= 2:  # Only hashtags used in 2+ posts
            avg_engagement = (data['total_likes']/data['total_views']*100) if data['total_views'] > 0 else 0
            hashtag_df.append({
                'hashtag': hashtag,
                'posts': data['posts'],
                'avg_engagement': avg_engagement,
                'total_likes': data['total_likes'],
                'unique_owners': len(data['owners'])
            })
    
    return pd.DataFrame(hashtag_df).sort_values('avg_engagement', ascending=False)

@st.cache_data
def analyze_user_network(posts):
    """Analyze user influence and mention patterns"""
    user_data = defaultdict(lambda: {'posts': 0, 'total_likes': 0, 'total_views': 0, 'mentioned_by': [], 'mentions_others': []})
    
    for post in posts:
        owner = post['owner']
        user_data[owner]['posts'] += 1
        user_data[owner]['total_likes'] += post['likes']
        user_data[owner]['total_views'] += post['views']
        
        # Track mentions
        for mention in post['mentions']:
            mentioned_user = mention[1:]  # Remove @
            user_data[owner]['mentions_others'].append(mentioned_user)
            user_data[mentioned_user]['mentioned_by'].append(owner)
    
    # Convert to dataframe
    user_df = []
    for user, data in user_data.items():
        if data['posts'] > 0:  # Only users who have posts
            avg_engagement = (data['total_likes']/data['total_views']*100) if data['total_views'] > 0 else 0
            influence_score = data['posts'] * avg_engagement * 0.01
            
            user_df.append({
                'user': user,
                'posts': data['posts'],
                'avg_engagement': avg_engagement,
                'influence_score': influence_score,
                'mentioned_by_count': len(set(data['mentioned_by'])),
                'mentions_others_count': len(set(data['mentions_others']))
            })
    
    return pd.DataFrame(user_df).sort_values('influence_score', ascending=False)

def create_network_visualization(graph, posts):
    """Create interactive network visualization"""
    if len(graph.nodes) == 0:
        return None
    
    # Calculate layout
    pos = nx.spring_layout(graph, k=3, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(graph[edge[0]][edge[1]].get('weight', 1))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        node_info = graph.nodes[node]
        engagement = node_info.get('engagement', 0)
        likes = node_info.get('likes', 0)
        owner = node_info.get('owner', '')
        
        node_text.append(f"{node}<br>@{owner}<br>Engagement: {engagement:.1f}%<br>Likes: {likes:,}")
        node_colors.append(engagement)
        node_sizes.append(max(10, min(50, likes/10)))  # Scale size by likes
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertemplate='%{text}<extra></extra>',
        text=[node for node in graph.nodes()],
        textposition="middle center",
        textfont=dict(size=8),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Engagement %"),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                    title="Instagram Content Network - Posts connected by similarity",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Node size = likes, Color = engagement rate, Connections = content similarity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="gray", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                ))
    
    return fig

# Main analysis
with st.spinner("🔗 Building content network graph..."):
    posts, graph = build_content_network(client_id)

if not posts:
    st.warning("⚠️ No posts found for analysis!")
    st.stop()

# Overview metrics
st.subheader("📊 Network Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📹 Posts", len(posts))
with col2:
    st.metric("🔗 Connections", len(graph.edges))
with col3:
    density = nx.density(graph) if len(graph.nodes) > 1 else 0
    st.metric("🕸️ Network Density", f"{density:.3f}")
with col4:
    communities = list(nx.connected_components(graph))
    st.metric("👥 Communities", len(communities))

# Network visualization
st.subheader("🕸️ Content Relationship Network")
network_fig = create_network_visualization(graph, posts)
if network_fig:
    st.plotly_chart(network_fig, use_container_width=True)
else:
    st.info("Not enough connections to display network graph")

# Analysis tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Hashtag Analysis", "👥 User Influence", "🔍 Network Metrics", "💡 Insights"])

with tab1:
    st.subheader("🏷️ Hashtag Performance & Relationships")
    
    hashtag_df = analyze_hashtags(posts)
    
    if not hashtag_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top hashtags by engagement
            st.write("**Top Performing Hashtags:**")
            top_hashtags = hashtag_df.head(10)
            fig_hashtags = px.bar(
                top_hashtags, 
                x='avg_engagement', 
                y='hashtag',
                title="Average Engagement Rate by Hashtag",
                orientation='h'
            )
            st.plotly_chart(fig_hashtags, use_container_width=True)
        
        with col2:
            # Hashtag usage frequency
            st.write("**Most Used Hashtags:**")
            fig_usage = px.scatter(
                hashtag_df,
                x='posts',
                y='avg_engagement',
                size='total_likes',
                hover_data=['hashtag'],
                title="Hashtag Usage vs Performance"
            )
            st.plotly_chart(fig_usage, use_container_width=True)
        
        # Hashtag table
        st.write("**Detailed Hashtag Analysis:**")
        st.dataframe(hashtag_df, use_container_width=True)
    else:
        st.info("No hashtags found with sufficient usage (2+ posts)")

with tab2:
    st.subheader("👥 User Influence & Mention Network")
    
    user_df = analyze_user_network(posts)
    
    if not user_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top influencers
            st.write("**Most Influential Users:**")
            top_users = user_df.head(10)
            fig_influence = px.bar(
                top_users,
                x='influence_score',
                y='user',
                title="User Influence Score",
                orientation='h'
            )
            st.plotly_chart(fig_influence, use_container_width=True)
        
        with col2:
            # Mention patterns
            st.write("**Mention Relationships:**")
            fig_mentions = px.scatter(
                user_df,
                x='mentioned_by_count',
                y='mentions_others_count', 
                size='posts',
                color='avg_engagement',
                hover_data=['user'],
                title="Mention Patterns (size = posts, color = engagement)"
            )
            st.plotly_chart(fig_mentions, use_container_width=True)
        
        # User table
        st.write("**User Analysis Details:**")
        st.dataframe(user_df, use_container_width=True)

with tab3:
    st.subheader("🔍 Advanced Network Metrics")
    
    if len(graph.nodes) > 1:
        # Centrality measures
        degree_centrality = nx.degree_centrality(graph)
        
        # Most central posts
        central_posts = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        st.write("**Most Connected Posts (High Centrality):**")
        central_df = pd.DataFrame(central_posts, columns=['Short Code', 'Centrality Score'])
        
        # Add post details
        post_details = []
        for short_code, centrality in central_posts:
            post = next((p for p in posts if p['short_code'] == short_code), {})
            post_details.append({
                'Short Code': short_code,
                'Owner': post.get('owner', ''),
                'Centrality': f"{centrality:.3f}",
                'Likes': post.get('likes', 0),
                'Engagement %': f"{(post.get('likes', 0)/post.get('views', 1)*100):.1f}%"
            })
        
        central_detailed_df = pd.DataFrame(post_details)
        st.dataframe(central_detailed_df, use_container_width=True)
        
        # Network statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Network Statistics:**")
            avg_clustering = nx.average_clustering(graph)
            st.write(f"• Average Clustering Coefficient: {avg_clustering:.3f}")
            st.write(f"• Network Density: {density:.3f}")
            st.write(f"• Connected Components: {len(communities)}")
            
            if communities:
                largest_component = max(communities, key=len)
                st.write(f"• Largest Component Size: {len(largest_component)} posts")
        
        with col2:
            st.write("**Community Distribution:**")
            community_sizes = [len(c) for c in communities]
            fig_communities = px.histogram(
                x=community_sizes,
                title="Community Size Distribution",
                labels={'x': 'Community Size', 'y': 'Number of Communities'}
            )
            st.plotly_chart(fig_communities, use_container_width=True)

with tab4:
    st.subheader("💡 Strategic Insights & Recommendations")
    
    # Generate insights
    insights = []
    
    # Network insights
    if density > 0.5:
        insights.append("🔗 **High Content Cohesion**: Your posts are highly interconnected, showing strong thematic consistency")
    elif density < 0.2:
        insights.append("🎭 **Diverse Content Strategy**: Your posts cover diverse topics - consider more thematic clustering")
    
    # Community insights
    if len(communities) > 1:
        largest = max(communities, key=len)
        largest_posts = [p for p in posts if p['short_code'] in largest]
        avg_engagement = sum((p['likes']/p['views']*100) if p['views'] > 0 else 0 for p in largest_posts) / len(largest_posts)
        insights.append(f"👥 **Content Clusters**: Found {len(communities)} distinct content groups. Largest cluster has {avg_engagement:.1f}% avg engagement")
    
    # Hashtag insights
    hashtag_df = analyze_hashtags(posts)
    if not hashtag_df.empty:
        best_hashtag = hashtag_df.iloc[0]
        insights.append(f"🏷️ **Top Hashtag**: {best_hashtag['hashtag']} performs best with {best_hashtag['avg_engagement']:.1f}% avg engagement")
    
    # User insights
    user_df = analyze_user_network(posts)
    if not user_df.empty:
        top_user = user_df.iloc[0]
        insights.append(f"👑 **Key Influencer**: @{top_user['user']} has highest influence score ({top_user['influence_score']:.1f})")
    
    # Display insights
    for insight in insights:
        st.write(insight)
    
    st.markdown("---")
    st.subheader("🎯 Actionable Recommendations")
    
    recommendations = [
        "**Content Strategy**: Focus on your highest-performing hashtags and content themes",
        "**Network Growth**: Engage more with central posts and influential users in your network", 
        "**Community Building**: Strengthen connections within your largest content clusters",
        "**Hashtag Optimization**: Use hashtags that appear in your top-performing content groups",
        "**Collaboration**: Reach out to users with high mention activity for potential partnerships"
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")

# Footer with cost info
st.markdown("---")
st.info("""
💰 **Cost-Effective Analysis**: This entire graph network analysis uses only your existing data and free algorithms. 
No expensive vision APIs required! The insights are generated from:
- Content similarity patterns
- Hashtag co-occurrence networks  
- User mention relationships
- Engagement correlation analysis
""")

# Export option
if st.button("📥 Export Network Analysis"):
    # Prepare export data
    export_data = {
        'posts': posts,
        'network_metrics': {
            'nodes': len(graph.nodes),
            'edges': len(graph.edges), 
            'density': density,
            'communities': len(communities)
        },
        'hashtag_analysis': hashtag_df.to_dict('records') if not hashtag_df.empty else [],
        'user_analysis': user_df.to_dict('records') if not user_df.empty else []
    }
    
    json_str = json.dumps(export_data, indent=2, default=str)
    st.download_button(
        label="Download Analysis JSON",
        data=json_str,
        file_name=f"network_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )
