"""
Integration Script: Connect Video Graph Analysis with Instagram Data
Bridges the gap between scraped Instagram data and graph-based video analysis
"""

import duckdb
import sqlite3
import json
from pathlib import Path
import subprocess
from graph.video_graph_builder import VideoGraphBuilder
from graph.video_graph_query_engine import VideoGraphQueryEngine
import pandas as pd

from core.db import DEFAULT_CLIENT_ID

class InstagramVideoGraphIntegrator:
    """
    Integrates Instagram video analysis with graph-based representation
    Converts scraped Instagram videos into queryable graph networks
    """
    
    def __init__(
        self,
        duckdb_path="reels.duckdb",
        graph_db_path="video_graphs.db",
        client_id=DEFAULT_CLIENT_ID,
    ):
        self.duckdb_path = duckdb_path
        self.graph_db_path = graph_db_path
        self.client_id = client_id
        self.duckdb_conn = duckdb.connect(duckdb_path)
        
        # Initialize graph builder and query engine
        self.graph_builder = VideoGraphBuilder(graph_db_path)
        self.query_engine = VideoGraphQueryEngine(graph_db_path)
    
    def get_instagram_videos_for_analysis(self, limit=10):
        """Get Instagram videos that need graph analysis"""
        query = """
        SELECT r.post_id, r.post_url, r.video_url, r.caption, r.like_count, r.play_count,
               r.music_info, r.duration, r.hashtags
        FROM reels r
        WHERE r.client_id = ?
          AND r.video_url IS NOT NULL
          AND r.video_url != ''
          AND r.post_id NOT IN (
              SELECT DISTINCT video_id FROM video_analysis 
              WHERE analysis_type = 'graph_signature'
          )
        ORDER BY r.like_count DESC
        LIMIT ?
        """
        
        result = self.duckdb_conn.execute(query, [self.client_id, limit]).fetchdf()
        return result
    
    def download_instagram_video(self, post_url, output_dir="downloads"):
        """Download Instagram video using post URL"""
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(exist_ok=True)
            
            # Use yt-dlp to download the video
            output_template = f"{output_dir}/%(id)s.%(ext)s"
            cmd = [
                "yt-dlp",
                "--extract-flat", "false",
                "--format", "best[ext=mp4]",
                "--output", output_template,
                post_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Extract the downloaded filename from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Destination:" in line or "has already been downloaded" in line:
                        # Extract filename from the line
                        parts = line.split()
                        for part in parts:
                            if part.endswith('.mp4'):
                                return part
                
                # Fallback: try to find the file by pattern
                post_id = post_url.split('/')[-2]  # Extract post ID from URL
                potential_files = list(Path(output_dir).glob(f"{post_id}*.mp4"))
                if potential_files:
                    return str(potential_files[0])
            
            return None
            
        except subprocess.TimeoutExpired:
            print(f"Download timeout for {post_url}")
            return None
        except Exception as e:
            print(f"Download error for {post_url}: {e}")
            return None
    
    def process_instagram_video_to_graph(self, video_data):
        """Convert Instagram video to graph representation"""
        try:
            post_id = video_data['post_id']
            post_url = video_data['post_url']
            
            print(f"Processing video: {post_id}")
            
            # Step 1: Download the video
            video_path = self.download_instagram_video(post_url)
            if not video_path:
                print(f"Failed to download video: {post_id}")
                return None
            
            print(f"Downloaded: {video_path}")
            
            # Step 2: Build graph representation
            graph_result = self.graph_builder.build_video_graph(video_path, post_id)
            if not graph_result:
                print(f"Failed to build graph for: {post_id}")
                return None
            
            print(f"Graph built with {graph_result['total_frames']} frames")
            
            # Step 3: Generate video signature
            signature = self.query_engine.generate_video_signature(post_id)
            
            # Step 4: Store analysis results in DuckDB
            self.store_graph_analysis_results(post_id, video_data, graph_result, signature)
            
            # Source videos are research material. The archive pipeline owns
            # moving them out of the working directory; graph analysis only reads.
            print(f"Video file retained: {video_path}")
            
            return {
                'post_id': post_id,
                'graph_result': graph_result,
                'signature': signature
            }
            
        except Exception as e:
            print(f"Error processing video {video_data['post_id']}: {e}")
            return None
    
    def store_graph_analysis_results(self, post_id, video_data, graph_result, signature):
        """Store graph analysis results back into DuckDB"""
        
        # Store in video_analysis table
        analysis_data = {
            'post_id': post_id,
            'analysis_type': 'graph_signature',
            'confidence_score': 1.0,  # Graph analysis is deterministic
            'analysis_results': json.dumps({
                'signature': signature,
                'graph_stats': {
                    'total_frames': graph_result['total_frames'],
                    'total_edges': graph_result['total_edges'],
                    'scene_transitions': graph_result['scene_transitions'],
                    'motion_segments': graph_result['motion_segments']
                }
            })
        }
        
        # Insert into video_analysis table
        insert_query = """
        INSERT INTO video_analysis (post_id, analysis_type, confidence_score, analysis_results, created_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        self.duckdb_conn.execute(insert_query, [
            analysis_data['post_id'],
            analysis_data['analysis_type'],
            analysis_data['confidence_score'],
            analysis_data['analysis_results']
        ])
        
        print(f"Stored graph analysis for {post_id}")
    
    def batch_process_videos(self, batch_size=5):
        """Process multiple videos in batch"""
        videos_to_process = self.get_instagram_videos_for_analysis(batch_size)
        
        if videos_to_process.empty:
            print("No videos need processing!")
            return []
        
        print(f"Processing {len(videos_to_process)} videos...")
        
        results = []
        for _, video in videos_to_process.iterrows():
            result = self.process_instagram_video_to_graph(video.to_dict())
            if result:
                results.append(result)
        
        return results
    
    def get_analysis_summary(self):
        """Get summary of graph analysis results"""
        query = """
        SELECT 
            COUNT(*) as total_analyzed,
            AVG(confidence_score) as avg_confidence,
            MIN(created_at) as first_analysis,
            MAX(created_at) as latest_analysis
        FROM video_analysis 
        WHERE analysis_type = 'graph_signature'
        """
        
        result = self.duckdb_conn.execute(query).fetchone()
        
        return {
            'total_analyzed': result[0],
            'avg_confidence': result[1],
            'first_analysis': result[2],
            'latest_analysis': result[3]
        }
    
    def find_similar_videos(self, post_id, similarity_threshold=0.7):
        """Find similar videos based on graph signatures"""
        # Get the signature of the target video
        query = """
        SELECT analysis_results 
        FROM video_analysis 
        WHERE post_id = ? AND analysis_type = 'graph_signature'
        """
        
        result = self.duckdb_conn.execute(query, [post_id]).fetchone()
        if not result:
            return []
        
        target_analysis = json.loads(result[0])
        target_signature = target_analysis['signature']
        
        # Get all other video signatures
        all_videos_query = """
        SELECT r.post_id, r.caption, r.like_count, r.hashtags, va.analysis_results
        FROM video_analysis va
        JOIN reels r ON va.post_id = r.post_id
        WHERE va.analysis_type = 'graph_signature' AND va.post_id != ?
        """
        
        all_videos = self.duckdb_conn.execute(all_videos_query, [post_id]).fetchall()
        
        similar_videos = []
        
        for video in all_videos:
            other_post_id, caption, like_count, hashtags, analysis_json = video
            other_analysis = json.loads(analysis_json)
            other_signature = other_analysis['signature']
            
            # Calculate similarity using the query engine's comparison method
            similarity_scores = self.calculate_signature_similarity(target_signature, other_signature)
            overall_similarity = similarity_scores['overall_similarity']
            
            if overall_similarity >= similarity_threshold:
                similar_videos.append({
                    'post_id': other_post_id,
                    'caption': caption,
                    'like_count': like_count,
                    'hashtags': hashtags,
                    'similarity': overall_similarity,
                    'similarity_breakdown': similarity_scores
                })
        
        # Sort by similarity
        similar_videos.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_videos
    
    def calculate_signature_similarity(self, sig1, sig2):
        """Calculate similarity between two video signatures"""
        # Use the query engine's comparison methods
        structural_sim = self.query_engine.calculate_structural_similarity(
            sig1['structural_fingerprint'], 
            sig2['structural_fingerprint']
        )
        
        visual_sim = self.query_engine.calculate_visual_similarity(
            sig1['visual_fingerprint'], 
            sig2['visual_fingerprint']
        )
        
        motion_sim = self.query_engine.calculate_motion_similarity(
            sig1['motion_fingerprint'], 
            sig2['motion_fingerprint']
        )
        
        overall_similarity = (structural_sim + visual_sim + motion_sim) / 3
        
        return {
            'overall_similarity': overall_similarity,
            'structural_similarity': structural_sim,
            'visual_similarity': visual_sim,
            'motion_similarity': motion_sim
        }
    
    def export_graph_insights(self, output_file="graph_insights.json"):
        """Export graph-based insights for all analyzed videos"""
        query = """
        SELECT r.post_id, r.caption, r.like_count, r.play_count, r.hashtags,
               r.music_info, va.analysis_results
        FROM video_analysis va
        JOIN reels r ON va.post_id = r.post_id
        WHERE va.analysis_type = 'graph_signature'
        ORDER BY r.like_count DESC
        """
        
        results = self.duckdb_conn.execute(query).fetchall()
        
        insights = []
        for row in results:
            post_id, caption, like_count, play_count, hashtags, music_info, analysis_json = row
            
            analysis = json.loads(analysis_json)
            signature = analysis['signature']
            
            insights.append({
                'post_id': post_id,
                'caption': caption,
                'engagement': {
                    'likes': like_count,
                    'plays': play_count
                },
                'content_features': {
                    'hashtags': hashtags,
                    'music_info': music_info
                },
                'visual_signature': signature
            })
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"Exported insights for {len(insights)} videos to {output_file}")
        
        return insights

def main():
    """Example usage of the integration system"""
    integrator = InstagramVideoGraphIntegrator()
    
    print("🎬 Instagram Video Graph Integration")
    print("=" * 50)
    
    # Check current status
    summary = integrator.get_analysis_summary()
    print(f"Current Analysis Status:")
    print(f"  Total Analyzed: {summary['total_analyzed']}")
    print(f"  Average Confidence: {summary['avg_confidence']:.2f}")
    
    if summary['total_analyzed'] == 0:
        print("\n🚀 Starting batch processing...")
        # Process a small batch for demo
        results = integrator.batch_process_videos(batch_size=3)
        print(f"✅ Processed {len(results)} videos successfully")
    
    # Export insights
    insights = integrator.export_graph_insights()
    print(f"📊 Exported insights for analysis")
    
    # Example: Find similar videos
    if insights:
        sample_post_id = insights[0]['post_id']
        similar = integrator.find_similar_videos(sample_post_id, similarity_threshold=0.6)
        print(f"🔍 Found {len(similar)} similar videos to {sample_post_id}")

if __name__ == "__main__":
    main()
