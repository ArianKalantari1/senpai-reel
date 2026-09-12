import duckdb

conn = duckdb.connect('reels.duckdb', read_only=True)

total = conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
print(f'Total posts: {total}')

transcribed = conn.execute('SELECT COUNT(DISTINCT post_id) FROM transcripts').fetchone()[0]
print(f'Transcribed posts: {transcribed}')
print(f'Untranscribed posts: {total - transcribed}')

with_audio = conn.execute("SELECT COUNT(*) FROM posts WHERE local_audio_path IS NOT NULL AND local_audio_path != ''").fetchone()[0]
print(f'Posts with local_audio_path: {with_audio}')

with_video = conn.execute("SELECT COUNT(*) FROM posts WHERE local_video_path IS NOT NULL AND local_video_path != ''").fetchone()[0]
print(f'Posts with video downloaded: {with_video}')

cols = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='posts' ORDER BY ordinal_position").fetchall()
print(f'\nPosts columns: {[c[0] for c in cols]}')

# Counts of audio_url vs video_url for untranscribed
r = conn.execute("""
    SELECT
        COUNT(*) FILTER (WHERE p.audio_url IS NOT NULL AND p.audio_url != '') as with_audio_url,
        COUNT(*) FILTER (WHERE p.video_url IS NOT NULL AND p.video_url != '') as with_video_url,
        COUNT(*) FILTER (WHERE (p.audio_url IS NULL OR p.audio_url = '') AND (p.video_url IS NOT NULL AND p.video_url != '')) as video_only
    FROM posts p
    LEFT JOIN transcripts t ON p.post_id = t.post_id
    WHERE t.post_id IS NULL
""").fetchone()
print(f'\nUntranscribed with audio_url: {r[0]}')
print(f'Untranscribed with video_url: {r[1]}')
print(f'Untranscribed video-only (no audio_url): {r[2]}')

# Existing transcript providers
existing = conn.execute('SELECT provider, COUNT(*) FROM transcripts GROUP BY provider').fetchall()
print(f'\nExisting transcripts by provider: {existing}')

# Sample transcribed post
sample = conn.execute("""
    SELECT p.post_id, p.local_audio_path, LEFT(p.audio_url, 60), LEFT(p.video_url, 60)
    FROM posts p
    JOIN transcripts t ON p.post_id = t.post_id
    LIMIT 3
""").fetchall()
print(f'\nSample transcribed posts:')
for s in sample:
    print(f'  post_id={s[0]}, local_audio={s[1]}, audio_url={s[2]}, video_url={s[3]}')

# Sample untranscribed post URLs
print(f'\nSample untranscribed post URLs:')
rows2 = conn.execute("""
    SELECT p.post_id, LEFT(p.audio_url, 80), LEFT(p.video_url, 80)
    FROM posts p
    LEFT JOIN transcripts t ON p.post_id = t.post_id
    WHERE t.post_id IS NULL
    LIMIT 5
""").fetchall()
for r2 in rows2:
    print(f'  post_id={r2[0]}')
    print(f'    audio_url: {r2[1]}')
    print(f'    video_url: {r2[2]}')

conn.close()
