import json
import requests
import duckdb
from datetime import datetime
import concurrent.futures
import os

from core.db import DEFAULT_CLIENT_ID

def test_url(url_data):
    """Test if a video URL is still valid"""
    url, short_code = url_data
    try:
        response = requests.head(url, timeout=10)
        return short_code, url, response.status_code == 200
    except:
        return short_code, url, False

def check_existing_urls(client_id=None):
    """Check which video URLs from the database still work"""
    conn = duckdb.connect("reels.duckdb")
    client_id = client_id or os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
    
    # Get all video URLs from the database
    results = conn.execute("""
        SELECT raw FROM raw_scrapes 
        WHERE client_id = ?
        ORDER BY scraped_at DESC
    """, [client_id]).fetchall()
    
    url_data = []
    for (raw,) in results:
        try:
            data = json.loads(raw)
            video_url = data.get('videoUrl')
            short_code = data.get('shortCode', 'unknown')
            
            if video_url and video_url != 'N/A':
                url_data.append((video_url, short_code))
        except:
            continue
    
    print(f"Testing {len(url_data)} video URLs...")
    
    # Test URLs in parallel for speed
    working_urls = []
    broken_urls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(test_url, url_data)
        
        for short_code, url, is_working in results:
            if is_working:
                working_urls.append((short_code, url))
                print(f"✅ {short_code}: Working")
            else:
                broken_urls.append((short_code, url))
                print(f"❌ {short_code}: Expired/Broken")
    
    print(f"\n📊 Results:")
    print(f"Working URLs: {len(working_urls)}")
    print(f"Broken URLs: {len(broken_urls)}")
    print(f"Success Rate: {len(working_urls)/(len(working_urls)+len(broken_urls))*100:.1f}%")
    
    return working_urls, broken_urls

if __name__ == "__main__":
    working, broken = check_existing_urls()
    
    if working:
        print(f"\n🎉 {len(working)} URLs still work! You can download these immediately.")
        
        download = input("\nDownload working videos now? (y/n): ").lower().startswith('y')
        if download:
            import os
            os.makedirs("downloads", exist_ok=True)
            
            for short_code, url in working:
                try:
                    print(f"Downloading {short_code}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    filepath = f"downloads/{short_code}.mp4"
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"✅ Saved: {filepath}")
                except Exception as e:
                    print(f"❌ Failed {short_code}: {e}")
    else:
        print("\n😞 No working URLs found. You'll need to use the post URLs to get fresh video links.")
