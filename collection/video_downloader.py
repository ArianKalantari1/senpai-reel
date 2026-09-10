import json
import requests
import duckdb
import os
from datetime import datetime
import re

from core.db import DEFAULT_CLIENT_ID

class VideoDownloader:
    def __init__(self, db_path="reels.duckdb", client_id=None):
        self.conn = duckdb.connect(db_path)
        self.client_id = client_id or os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
        self.download_folder = "downloads"
        os.makedirs(self.download_folder, exist_ok=True)
    
    def get_fresh_video_url(self, post_url):
        """
        Extract fresh video URL from Instagram post URL
        This uses the same technique as your scraper
        """
        try:
            # You can adapt your existing scraping logic here
            # For now, this is a placeholder that shows the concept
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(post_url, headers=headers)
            
            # Extract video URL from the page source
            # This would need your scraping logic to get the fresh videoUrl
            # For demonstration, returning None - you'd implement the extraction
            return None
            
        except Exception as e:
            print(f"Error getting fresh URL: {e}")
            return None
    
    def download_video(self, video_url, filename):
        """Download video from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(video_url, headers=headers, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(self.download_folder, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    def download_videos_from_db(self, profile=None, limit=10):
        """Download videos from database records"""
        query = "SELECT raw, scraped_at FROM raw_scrapes WHERE client_id = ?"
        params = [self.client_id]
        
        if profile:
            query += " AND profile = ?"
            params.append(profile)
        
        query += f" ORDER BY scraped_at DESC LIMIT {limit}"
        
        results = self.conn.execute(query, params).fetchall()
        
        downloaded = []
        
        for raw_data, scraped_at in results:
            try:
                data = json.loads(raw_data)
                short_code = data.get('shortCode', 'unknown')
                post_url = data.get('url')
                old_video_url = data.get('videoUrl')
                
                print(f"Processing {short_code}...")
                
                # Try the old URL first (might still work)
                filename = f"{short_code}_{scraped_at.strftime('%Y%m%d')}.mp4"
                
                if old_video_url and old_video_url != 'N/A':
                    result = self.download_video(old_video_url, filename)
                    if result:
                        downloaded.append((short_code, result))
                        print(f"✅ Downloaded {short_code}")
                        continue
                
                # If old URL failed, try to get fresh URL from post
                if post_url:
                    fresh_url = self.get_fresh_video_url(post_url)
                    if fresh_url:
                        result = self.download_video(fresh_url, filename)
                        if result:
                            downloaded.append((short_code, result))
                            print(f"✅ Downloaded {short_code} (fresh URL)")
                        else:
                            print(f"❌ Failed to download {short_code}")
                    else:
                        print(f"⚠️ Couldn't get fresh URL for {short_code}")
                
            except Exception as e:
                print(f"Error processing record: {e}")
        
        return downloaded

if __name__ == "__main__":
    downloader = VideoDownloader()
    
    # Download videos for a specific profile
    profile = input("Enter profile name (or press Enter for all): ").strip()
    profile = profile if profile else None
    
    limit = int(input("How many videos to download (default 10): ") or "10")
    
    print(f"\n🚀 Starting download...")
    downloaded = downloader.download_videos_from_db(profile, limit)
    
    print(f"\n✨ Download complete!")
    print(f"Successfully downloaded {len(downloaded)} videos:")
    for short_code, filepath in downloaded:
        print(f"  - {short_code}: {filepath}")
