import json
import duckdb
import os

from core.db import DEFAULT_CLIENT_ID

# Connect to database and get all Instagram post URLs
conn = duckdb.connect("reels.duckdb")
client_id = os.getenv("SENPAI_CLIENT_ID", DEFAULT_CLIENT_ID)
results = conn.execute(
    "SELECT raw FROM raw_scrapes WHERE client_id = ? ORDER BY scraped_at DESC",
    [client_id],
).fetchall()

print("📋 Instagram Post URLs for Manual Download")
print("=" * 60)
print()

for i, (raw,) in enumerate(results, 1):
    try:
        data = json.loads(raw)
        post_url = data.get('url')
        short_code = data.get('shortCode', 'unknown')
        owner = data.get('ownerUsername', 'unknown')
        caption = data.get('caption', '')[:100] if data.get('caption') else 'No caption'
        
        if post_url:
            print(f"{i:2d}. {short_code} by @{owner}")
            print(f"    URL: {post_url}")
            print(f"    Caption: {caption}...")
            print()
    except:
        continue

print("\n" + "=" * 60)
print("💡 How to download:")
print("1. Copy any Instagram URL above")
print("2. Go to one of these sites:")
print("   • https://snapinsta.app/")
print("   • https://instadownloader.co/")
print("   • https://savefrom.net/18/")
print("3. Paste URL and download")
print("4. Repeat for all videos you want")
