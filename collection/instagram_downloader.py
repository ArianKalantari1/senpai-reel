import json
import duckdb
import requests
import os
from urllib.parse import quote

from core.db import DEFAULT_CLIENT_ID

class InstagramVideoDownloader:
    def __init__(self, client_id=None):
        self.conn = duckdb.connect("reels.duckdb")
        self.client_id = client_id or os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
        os.makedirs("downloads", exist_ok=True)
    
    def get_post_urls(self, limit=10, profile=None):
        """Get Instagram post URLs from database"""
        query = "SELECT raw FROM raw_scrapes WHERE client_id = ?"
        params = [self.client_id]
        
        if profile:
            query += " AND profile = ?"
            params.append(profile)
        
        query += f" ORDER BY scraped_at DESC LIMIT {limit}"
        
        results = self.conn.execute(query, params).fetchall()
        
        post_data = []
        for (raw,) in results:
            try:
                data = json.loads(raw)
                post_url = data.get('url')
                short_code = data.get('shortCode', 'unknown')
                title = data.get('caption', '')[:50] if data.get('caption') else short_code
                
                if post_url:
                    post_data.append({
                        'url': post_url,
                        'short_code': short_code,
                        'title': title
                    })
            except:
                continue
        
        return post_data
    
    def download_via_online_service(self, post_urls):
        """
        Use online Instagram downloader services
        These services work with post URLs and don't require re-scraping
        """
        services = [
            {
                'name': 'SnapInsta',
                'url': 'https://snapinsta.app/',
                'method': 'Paste Instagram URL and download'
            },
            {
                'name': 'SaveFrom.net',
                'url': 'https://en.savefrom.net/18/',
                'method': 'Paste URL, click download'
            },
            {
                'name': 'Instagram Video Downloader',
                'url': 'https://igram.world/en/',
                'method': 'Paste URL and download'
            },
            {
                'name': 'InstaDownloader',
                'url': 'https://instadownloader.co/',
                'method': 'Paste URL and get download link'
            }
        ]
        
        print("🌐 Online Download Services:")
        print("=" * 50)
        
        for service in services:
            print(f"\n📱 {service['name']}")
            print(f"   URL: {service['url']}")
            print(f"   Method: {service['method']}")
        
        print(f"\n📋 Your Instagram Post URLs:")
        print("=" * 50)
        
        for i, post in enumerate(post_urls, 1):
            print(f"{i:2d}. {post['short_code']}: {post['url']}")
            if post['title']:
                print(f"     Caption: {post['title']}...")
        
        print("\n💡 Instructions:")
        print("1. Copy any of the Instagram URLs above")
        print("2. Go to one of the download services")
        print("3. Paste the URL and download the video")
        print("4. Repeat for all videos you want")
        
        return post_urls
    
    def create_batch_download_script(self, post_urls):
        """Create a script for batch downloading using yt-dlp"""
        script_content = """#!/bin/bash
# Batch Instagram Video Downloader
# Install yt-dlp first: pip install yt-dlp

mkdir -p downloads
cd downloads

echo "🚀 Starting Instagram video downloads..."

"""
        
        for post in post_urls:
            script_content += f'echo "Downloading {post["short_code"]}..."\n'
            script_content += f'/Users/ariankalantari/senpai-reel/venv/bin/python -m yt_dlp "{post["url"]}" -o "{post["short_code"]}.%(ext)s"\n'
            script_content += f'echo "✅ Done: {post["short_code"]}"\n\n'
        
        script_content += 'echo "🎉 All downloads complete!"\n'
        
        with open("batch_download.sh", "w") as f:
            f.write(script_content)
        
        os.chmod("batch_download.sh", 0o755)
        
        print("\n📝 Created batch_download.sh script!")
        print("To use it:")
        print("1. Install yt-dlp: pip install yt-dlp")
        print("2. Run: ./batch_download.sh")
        print("3. Videos will be saved in downloads/ folder")

def main():
    downloader = InstagramVideoDownloader()
    
    profile = input("Enter profile name (or press Enter for all): ").strip()
    profile = profile if profile else None
    
    limit = int(input("How many videos (default 10): ") or "10")
    
    print(f"\n🔍 Getting Instagram post URLs...")
    post_urls = downloader.get_post_urls(limit, profile)
    
    if not post_urls:
        print("❌ No post URLs found!")
        return
    
    print(f"\n✨ Found {len(post_urls)} Instagram posts")
    
    choice = input("""
Choose download method:
1. Online services (manual but reliable)
2. Create batch script (requires yt-dlp)

Enter choice (1 or 2): """).strip()
    
    if choice == "1":
        downloader.download_via_online_service(post_urls)
    elif choice == "2":
        downloader.create_batch_download_script(post_urls)
    else:
        print("Invalid choice. Showing online services:")
        downloader.download_via_online_service(post_urls)

if __name__ == "__main__":
    main()
