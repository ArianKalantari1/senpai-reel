import duckdb, json
conn = duckdb.connect('reels.duckdb')
total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
print(f"Total posts: {total}")

rows = conn.execute("""
  SELECT post_id,
         (video_url IS NOT NULL AND video_url != '') as has_video_url,
         (audio_url IS NOT NULL AND audio_url != '') as has_audio_url,
         local_audio_path,
         download_status,
         LEFT(video_url, 80) as video_url_preview
  FROM posts LIMIT 5
""").fetchall()
for r in rows:
    print(r)

# Pipeline eligibility check
eligible = conn.execute("""
  SELECT COUNT(*) FROM posts p
  LEFT JOIN transcripts t ON p.post_id = t.post_id
  WHERE (
    (p.local_audio_path IS NOT NULL AND p.local_audio_path != '')
    OR (p.audio_url IS NOT NULL AND p.audio_url != '')
    OR (p.video_url IS NOT NULL AND p.video_url != '')
  )
  AND t.post_id IS NULL
""").fetchone()[0]
print(f"\nPipeline-eligible (untranscribed with any URL): {eligible}")

trans = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
print(f"Transcripts: {trans}")
conn.close()
