#!/usr/bin/env python3
import json
import duckdb
import os
import subprocess
import sys

from core.db import DEFAULT_CLIENT_ID

def download_all_videos(client_id=None):
    """Download all Instagram videos from the database using yt-dlp"""
    client_id = client_id or os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
    
    # Connect to database
    conn = duckdb.connect("reels.duckdb")
    
    # Create downloads directory
    os.makedirs("downloads", exist_ok=True)
    
    # Get all post URLs
    results = conn.execute("""
        SELECT raw FROM raw_scrapes 
        WHERE client_id = ?
        ORDER BY scraped_at DESC
    """, [client_id]).fetchall()
    
    videos_to_download = []
    
    for (raw,) in results:
        try:
            data = json.loads(raw)
            post_url = data.get('url')
            short_code = data.get('shortCode', 'unknown')
            owner = data.get('ownerUsername', 'unknown')
            
            if post_url:
                videos_to_download.append({
                    'url': post_url,
                    'short_code': short_code,
                    'owner': owner,
                    'filename': f"{owner}_{short_code}"
                })
        except:
            continue
    
    print(f"🎯 Found {len(videos_to_download)} videos to download")
    
    # Download each video
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos_to_download, 1):
        print(f"\n📥 [{i}/{len(videos_to_download)}] Downloading {video['short_code']} by @{video['owner']}...")
        
        try:
            # Use yt-dlp to download
            cmd = [
                "/Users/ariankalantari/senpai-reel/venv/bin/python", "-m", "yt_dlp",
                video['url'],
                "-o", f"downloads/{video['filename']}.%(ext)s",
                "--cookies-from-browser", "chrome"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Success: {video['short_code']}")
                successful += 1
            else:
                print(f"❌ Failed: {video['short_code']}")
                print(f"   Error: {result.stderr[:100]}...")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error downloading {video['short_code']}: {e}")
            failed += 1
    
    print(f"\n🎉 Download Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📂 Videos saved in: downloads/")
    
    return successful, failed

if __name__ == "__main__":
    print("🚀 Instagram Video Downloader")
    print("=" * 40)
    
    try:
        successful, failed = download_all_videos()
        
        if successful > 0:
            print(f"\n🎊 Successfully downloaded {successful} videos!")
            print("You can find them in the downloads/ folder")
        else:
            print("\n😞 No videos were downloaded successfully")
            print("Try the manual method using online services")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
