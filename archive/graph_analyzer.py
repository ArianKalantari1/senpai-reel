import json
import duckdb
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict

from core.db import DEFAULT_CLIENT_ID

class InstagramGraphAnalyzer:
    """
    Create rich graph networks from Instagram data WITHOUT expensive vision APIs
    Focus on relationships, patterns, and network analysis
    """
    
    def __init__(self, db_path="reels.duckdb", client_id=DEFAULT_CLIENT_ID):
        self.conn = duckdb.connect(db_path)
        self.client_id = client_id
        self.graph = nx.Graph()
        
    def extract_hashtags_mentions(self, text):
        """Extract hashtags and mentions from text"""
        if not text:
            return [], []
        
        hashtags = re.findall(r'#\w+', text.lower())
        mentions = re.findall(r'@\w+', text.lower())
        return hashtags, mentions
    
    def build_content_similarity_graph(self):
        """Build graph based on content similarity (keywords, hashtags, etc.)"""
        print("🔗 Building content similarity graph...")
        
        # Get all posts with metadata
        results = self.conn.execute("""
            SELECT raw FROM raw_scrapes 
            WHERE client_id = ?
            ORDER BY scraped_at DESC
        """, [self.client_id]).fetchall()
        
        posts = []
        for (raw,) in results:
            try:
                data = json.loads(raw)
                
                caption = data.get('caption', '') or ''
                hashtags, mentions = self.extract_hashtags_mentions(caption)
                
                post_data = {
                    'short_code': data.get('shortCode', ''),
                    'owner': data.get('ownerUsername', ''),
                    'caption': caption,
                    'hashtags': hashtags,
                    'mentions': mentions,
                    'location': data.get('locationName', ''),
                    'likes': data.get('likesCount', 0) or 0,
                    'views': data.get('videoViewCount', 0) or 0,
                    'timestamp': data.get('timestamp', ''),
                    'duration': data.get('videoDuration', 0) or 0
                }
                posts.append(post_data)
            except:
                continue
        
        # Add nodes (posts)
        for post in posts:
            self.graph.add_node(
                post['short_code'],
                node_type='post',
                owner=post['owner'],
                likes=post['likes'],
                views=post['views'],
                engagement_rate=(post['likes']/post['views']*100) if post['views'] > 0 else 0,
                hashtags=len(post['hashtags']),
                mentions=len(post['mentions']),
                has_location=bool(post['location'])
            )
        
        # Create edges based on content similarity
        for i, post1 in enumerate(posts):
            for post2 in posts[i+1:]:
                similarity_score = self.calculate_content_similarity(post1, post2)
                
                if similarity_score > 0.3:  # Threshold for connection
                    self.graph.add_edge(
                        post1['short_code'], 
                        post2['short_code'],
                        weight=similarity_score,
                        connection_type='content_similarity'
                    )
        
        return posts
    
    def calculate_content_similarity(self, post1, post2):
        """Calculate similarity between two posts based on multiple factors"""
        score = 0
        
        # Same owner
        if post1['owner'] == post2['owner']:
            score += 0.4
        
        # Shared hashtags
        common_hashtags = set(post1['hashtags']) & set(post2['hashtags'])
        if common_hashtags:
            score += 0.3 * (len(common_hashtags) / max(len(post1['hashtags']), len(post2['hashtags']), 1))
        
        # Similar engagement patterns
        eng1 = (post1['likes']/post1['views']*100) if post1['views'] > 0 else 0
        eng2 = (post2['likes']/post2['views']*100) if post2['views'] > 0 else 0
        if abs(eng1 - eng2) < 5:  # Similar engagement rates
            score += 0.2
        
        # Similar duration
        if abs(post1['duration'] - post2['duration']) < 10:  # Within 10 seconds
            score += 0.1
        
        # Cross-mentions
        if any(mention in post2['caption'].lower() for mention in post1['mentions']):
            score += 0.4
        
        return min(1.0, score)
    
    def build_hashtag_network(self, posts):
        """Create a separate graph for hashtag relationships"""
        hashtag_graph = nx.Graph()
        hashtag_posts = defaultdict(list)
        
        # Collect posts for each hashtag
        for post in posts:
            for hashtag in post['hashtags']:
                hashtag_posts[hashtag].append(post)
        
        # Filter hashtags that appear in at least 2 posts
        popular_hashtags = {h: posts for h, posts in hashtag_posts.items() if len(posts) >= 2}
        
        # Add hashtag nodes
        for hashtag, related_posts in popular_hashtags.items():
            total_engagement = sum((p['likes']/p['views']*100) if p['views'] > 0 else 0 for p in related_posts)
            avg_engagement = total_engagement / len(related_posts)
            
            hashtag_graph.add_node(
                hashtag,
                node_type='hashtag',
                post_count=len(related_posts),
                avg_engagement=avg_engagement,
                total_likes=sum(p['likes'] for p in related_posts)
            )
        
        # Create edges between hashtags that co-occur
        hashtag_list = list(popular_hashtags.keys())
        for i, hashtag1 in enumerate(hashtag_list):
            for hashtag2 in hashtag_list[i+1:]:
                # Count co-occurrences
                cooccurrence = sum(
                    1 for post in posts 
                    if hashtag1 in post['hashtags'] and hashtag2 in post['hashtags']
                )
                
                if cooccurrence > 0:
                    hashtag_graph.add_edge(
                        hashtag1, hashtag2,
                        weight=cooccurrence,
                        cooccurrence_count=cooccurrence
                    )
        
        return hashtag_graph
    
    def build_user_influence_network(self, posts):
        """Build network showing user influence and interactions"""
        user_graph = nx.DiGraph()  # Directed graph for influence
        
        # Collect user data
        user_data = defaultdict(lambda: {'posts': 0, 'total_likes': 0, 'total_views': 0, 'mentioned_by': []})
        
        for post in posts:
            owner = post['owner']
            user_data[owner]['posts'] += 1
            user_data[owner]['total_likes'] += post['likes']
            user_data[owner]['total_views'] += post['views']
            
            # Track who mentions whom
            for mention in post['mentions']:
                mentioned_user = mention[1:]  # Remove @
                user_data[mentioned_user]['mentioned_by'].append(owner)
        
        # Add user nodes
        for user, data in user_data.items():
            avg_engagement = (data['total_likes']/data['total_views']*100) if data['total_views'] > 0 else 0
            influence_score = data['posts'] * avg_engagement * 0.01  # Simple influence metric
            
            user_graph.add_node(
                user,
                node_type='user',
                posts=data['posts'],
                avg_engagement=avg_engagement,
                influence_score=influence_score,
                mention_count=len(data['mentioned_by'])
            )
        
        # Add edges for mentions (influence flow)
        for user, data in user_data.items():
            for mentioner in data['mentioned_by']:
                if mentioner in user_graph and user in user_graph:
                    if user_graph.has_edge(mentioner, user):
                        user_graph[mentioner][user]['weight'] += 1
                    else:
                        user_graph.add_edge(mentioner, user, weight=1, connection_type='mention')
        
        return user_graph
    
    def analyze_network_metrics(self):
        """Calculate network analysis metrics"""
        if len(self.graph) == 0:
            return {}
        
        metrics = {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
        }
        
        # Central nodes
        if len(self.graph) > 1:
            centrality = nx.degree_centrality(self.graph)
            metrics['most_central_posts'] = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if nx.is_connected(self.graph):
                metrics['diameter'] = nx.diameter(self.graph)
                betweenness = nx.betweenness_centrality(self.graph)
                metrics['bridge_posts'] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return metrics
    
    def detect_communities(self):
        """Find communities/clusters in the network"""
        if len(self.graph) < 3:
            return []
        
        try:
            # Use Louvain community detection
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            return dict(communities)
        except ImportError:
            # Fallback: simple connected components
            return {i: list(component) for i, component in enumerate(nx.connected_components(self.graph))}
    
    def generate_insights(self, posts, hashtag_graph, user_graph):
        """Generate actionable insights from the network analysis"""
        insights = []
        
        # Content clustering insights
        communities = self.detect_communities()
        if communities:
            insights.append(f"📊 Found {len(communities)} content clusters in your network")
            
            # Analyze largest cluster
            largest_cluster = max(communities.values(), key=len)
            if len(largest_cluster) > 1:
                cluster_posts = [p for p in posts if p['short_code'] in largest_cluster]
                avg_engagement = sum((p['likes']/p['views']*100) if p['views'] > 0 else 0 for p in cluster_posts) / len(cluster_posts)
                insights.append(f"🎯 Your largest content cluster has {len(largest_cluster)} posts with {avg_engagement:.1f}% avg engagement")
        
        # Hashtag insights
        if hashtag_graph.nodes:
            hashtag_metrics = [(node, data['avg_engagement']) for node, data in hashtag_graph.nodes(data=True)]
            best_hashtag = max(hashtag_metrics, key=lambda x: x[1])
            insights.append(f"🏷️ Most effective hashtag: {best_hashtag[0]} ({best_hashtag[1]:.1f}% avg engagement)")
        
        # User influence insights
        if user_graph.nodes:
            influence_metrics = [(node, data['influence_score']) for node, data in user_graph.nodes(data=True)]
            top_influencer = max(influence_metrics, key=lambda x: x[1])
            insights.append(f"👑 Most influential account: @{top_influencer[0]} (influence score: {top_influencer[1]:.1f})")
        
        # Network density insight
        network_metrics = self.analyze_network_metrics()
        if 'density' in network_metrics:
            density = network_metrics['density']
            if density > 0.5:
                insights.append("🔗 High content interconnectedness - your posts form a cohesive brand narrative")
            elif density < 0.2:
                insights.append("🎭 Diverse content strategy - consider more thematic consistency for stronger branding")
        
        return insights

if __name__ == "__main__":
    analyzer = InstagramGraphAnalyzer()
    
    print("🕸️ Instagram Graph Network Analyzer")
    print("=" * 50)
    print("Building content similarity graph...")
    
    posts = analyzer.build_content_similarity_graph()
    hashtag_graph = analyzer.build_hashtag_network(posts)
    user_graph = analyzer.build_user_influence_network(posts)
    
    print(f"✅ Built network with {len(analyzer.graph.nodes)} posts and {len(analyzer.graph.edges)} connections")
    
    # Analyze metrics
    metrics = analyzer.analyze_network_metrics()
    communities = analyzer.detect_communities()
    insights = analyzer.generate_insights(posts, hashtag_graph, user_graph)
    
    print(f"\n📊 Network Analysis Results:")
    print(f"  Nodes: {metrics.get('total_nodes', 0)}")
    print(f"  Edges: {metrics.get('total_edges', 0)}")
    print(f"  Network Density: {metrics.get('density', 0):.3f}")
    print(f"  Communities Found: {len(communities)}")
    
    print(f"\n💡 Key Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    if 'most_central_posts' in metrics:
        print(f"\n🎯 Most Connected Posts:")
        for post, centrality in metrics['most_central_posts']:
            print(f"  {post}: {centrality:.3f} centrality")
    
    print(f"\n🏷️ Hashtag Network:")
    print(f"  Popular hashtags: {len(hashtag_graph.nodes)}")
    print(f"  Hashtag connections: {len(hashtag_graph.edges)}")
    
    print(f"\n👥 User Influence Network:")
    print(f"  Users: {len(user_graph.nodes)}")
    print(f"  Mention relationships: {len(user_graph.edges)}")
