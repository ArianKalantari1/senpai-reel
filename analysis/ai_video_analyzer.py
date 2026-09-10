import json
import duckdb
import os
import openai
from datetime import datetime
import base64
import cv2
import tempfile
import subprocess

from core.db import DEFAULT_CLIENT_ID

class VideoAIAnalyzer:
    def __init__(self, db_path="reels.duckdb", client_id=None):
        self.conn = duckdb.connect(db_path)
        self.client_id = client_id or os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
        self.downloads_path = "downloads"
        self.analysis_path = "ai_analysis"
        
        # Create analysis table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS video_analysis (
                id INTEGER PRIMARY KEY,
                client_id VARCHAR,
                short_code VARCHAR,
                video_filepath VARCHAR,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Visual Analysis
                scene_description TEXT,
                object_detection JSON,
                text_extraction TEXT,
                color_analysis JSON,
                
                -- Audio Analysis  
                audio_transcript TEXT,
                background_music TEXT,
                audio_sentiment VARCHAR,
                
                -- Content Analysis
                content_category VARCHAR,
                engagement_prediction FLOAT,
                brand_mentions JSON,
                emotion_analysis JSON,
                
                -- Technical Metrics
                video_quality_score FLOAT,
                frame_analysis JSON,
                
                -- AI Model Info
                ai_model_used VARCHAR,
                confidence_score FLOAT,
                processing_time_seconds FLOAT
            )
        """)
        try:
            self.conn.execute("ALTER TABLE video_analysis ADD COLUMN client_id VARCHAR")
            self.conn.execute(
                "UPDATE video_analysis SET client_id = ? WHERE client_id IS NULL",
                [DEFAULT_CLIENT_ID],
            )
        except Exception:
            pass
        
        os.makedirs(self.analysis_path, exist_ok=True)
    
    def extract_frames(self, video_path, num_frames=5):
        """Extract key frames from video for analysis"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert to base64 for API calls
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append({
                    'timestamp': i / cap.get(cv2.CAP_PROP_FPS),
                    'frame_data': frame_base64
                })
            
            if len(frames) >= num_frames:
                break
        
        cap.release()
        return frames
    
    def analyze_with_openai_vision(self, frames, metadata):
        """Use OpenAI GPT-4 Vision for video analysis"""
        # This would require OpenAI API key
        try:
            client = openai.OpenAI()
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a video content analyzer. Analyze the provided video frames and metadata to extract:
                    1. Scene description and visual content
                    2. Objects and people detected
                    3. Text visible in frames
                    4. Color schemes and aesthetics
                    5. Content category (food, lifestyle, entertainment, etc.)
                    6. Brand mentions or logos
                    7. Emotional tone and engagement factors
                    8. Predicted engagement level (1-10)
                    
                    Return analysis as structured JSON."""
                },
                {
                    "role": "user", 
                    "content": f"""
                    Analyze this Instagram reel:
                    
                    Metadata:
                    - Short Code: {metadata.get('shortCode')}
                    - Caption: {metadata.get('caption', '')[:500]}
                    - Owner: @{metadata.get('ownerUsername')}
                    - Location: {metadata.get('locationName')}
                    - Duration: {metadata.get('videoDuration')} seconds
                    - Current Likes: {metadata.get('likesCount')}
                    - Current Views: {metadata.get('videoViewCount')}
                    
                    I've provided {len(frames)} key frames from the video.
                    """
                }
            ]
            
            # Add frame images to the message
            for i, frame in enumerate(frames):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Frame {i+1} (at {frame['timestamp']:.1f}s):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame['frame_data']}"
                            }
                        }
                    ]
                })
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=1500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"OpenAI Vision analysis failed: {e}")
            return None
    
    def analyze_with_local_models(self, video_path, metadata):
        """Use local AI models for privacy/cost reasons"""
        analysis = {
            "scene_description": "Local analysis placeholder",
            "content_category": "entertainment",
            "engagement_prediction": 7.5,
            "processing_method": "local"
        }
        
        # You could integrate:
        # - YOLO for object detection
        # - CLIP for scene understanding
        # - Whisper for audio transcription
        # - Local LLMs for content analysis
        
        return analysis
    
    def analyze_video_file(self, video_filepath, use_cloud_ai=True):
        """Analyze a single video file"""
        start_time = datetime.now()
        short_code = os.path.splitext(os.path.basename(video_filepath))[0]
        
        print(f"🔍 Analyzing {short_code}...")
        
        # Get metadata from database
        metadata_result = self.conn.execute("""
            SELECT raw FROM raw_scrapes
            WHERE client_id = ? AND raw LIKE ?
            LIMIT 1
        """, [self.client_id, f'%{short_code}%']).fetchone()
        
        metadata = {}
        if metadata_result:
            try:
                metadata = json.loads(metadata_result[0])
            except:
                pass
        
        # Extract frames
        frames = self.extract_frames(video_filepath)
        
        if not frames:
            print(f"❌ Could not extract frames from {short_code}")
            return None
        
        # Choose analysis method
        if use_cloud_ai:
            analysis = self.analyze_with_openai_vision(frames, metadata)
        else:
            analysis = self.analyze_with_local_models(video_filepath, metadata)
        
        if not analysis:
            print(f"❌ Analysis failed for {short_code}")
            return None
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store results in database
        self.conn.execute("""
            INSERT INTO video_analysis (
                client_id, short_code, video_filepath, scene_description,
                object_detection, text_extraction, content_category,
                engagement_prediction, ai_model_used, confidence_score,
                processing_time_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            self.client_id,
            short_code,
            video_filepath,
            analysis.get('scene_description', ''),
            json.dumps(analysis.get('object_detection', {})),
            analysis.get('text_extraction', ''),
            analysis.get('content_category', ''),
            analysis.get('engagement_prediction', 0.0),
            'gpt-4-vision' if use_cloud_ai else 'local',
            analysis.get('confidence_score', 0.8),
            processing_time
        ])
        
        print(f"✅ Analysis complete for {short_code} ({processing_time:.1f}s)")
        return analysis
    
    def batch_analyze_downloaded_videos(self, limit=None, use_cloud_ai=True):
        """Analyze all downloaded videos"""
        video_files = []
        
        for filename in os.listdir(self.downloads_path):
            if filename.endswith(('.mp4', '.mov', '.avi')):
                filepath = os.path.join(self.downloads_path, filename)
                video_files.append(filepath)
        
        if limit:
            video_files = video_files[:limit]
        
        print(f"🚀 Starting AI analysis of {len(video_files)} videos...")
        
        results = []
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}]", end=" ")
            
            try:
                result = self.analyze_video_file(video_path, use_cloud_ai)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {os.path.basename(video_path)}: {e}")
        
        print(f"\n🎉 Analysis complete! {len(results)} videos analyzed successfully.")
        return results
    
    def get_analysis_insights(self):
        """Get insights from all analyzed videos"""
        results = self.conn.execute("""
            SELECT 
                COUNT(*) as total_analyzed,
                AVG(engagement_prediction) as avg_engagement_prediction,
                content_category,
                COUNT(*) as category_count,
                AVG(processing_time_seconds) as avg_processing_time
            FROM video_analysis 
            WHERE client_id = ?
            GROUP BY content_category
            ORDER BY category_count DESC
        """, [self.client_id]).fetchall()
        
        print("\n📊 AI Analysis Insights:")
        print("=" * 50)
        
        for row in results:
            print(f"Content Category: {row[2]}")
            print(f"  Videos: {row[3]}")
            print(f"  Avg Engagement Prediction: {row[1]:.1f}/10")
            print(f"  Avg Processing Time: {row[4]:.1f}s")
            print()

if __name__ == "__main__":
    analyzer = VideoAIAnalyzer()
    
    print("🤖 Video AI Analyzer")
    print("=" * 40)
    
    choice = input("""
Choose analysis method:
1. Cloud AI (OpenAI Vision) - More accurate, costs money
2. Local AI - Free, less accurate
3. View existing analysis insights

Enter choice (1-3): """).strip()
    
    if choice == "1":
        print("⚠️  Note: This will use OpenAI API and incur costs")
        confirm = input("Continue? (y/n): ").lower().startswith('y')
        if confirm:
            analyzer.batch_analyze_downloaded_videos(limit=3, use_cloud_ai=True)
        
    elif choice == "2":
        analyzer.batch_analyze_downloaded_videos(limit=5, use_cloud_ai=False)
        
    elif choice == "3":
        analyzer.get_analysis_insights()
    
    else:
        print("Invalid choice")
