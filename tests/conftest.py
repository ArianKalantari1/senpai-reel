"""
Shared pytest fixtures for all test modules.
Uses an in-memory DuckDB database so tests never touch reels.duckdb.
"""
import pytest
import duckdb
from unittest.mock import patch


@pytest.fixture
def mem_conn():
    """Provide a fresh in-memory DuckDB connection with the full schema."""
    conn = duckdb.connect(":memory:")
    _create_all_tables(conn)
    yield conn
    conn.close()


def _create_all_tables(conn: duckdb.DuckDBPyConnection):
    """Create the full schema that matches core/db.py init_db()."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        client_id TEXT PRIMARY KEY, name TEXT NOT NULL, niche TEXT,
        created_at TIMESTAMP, notes TEXT, brand_voice_notes TEXT,
        target_audience TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS client_accounts (
        client_id TEXT, instagram_handle TEXT, category TEXT,
        added_at TIMESTAMP, max_items INTEGER DEFAULT 30,
        PRIMARY KEY (client_id, instagram_handle)
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS client_posts (
        client_id TEXT, post_id TEXT, added_at TIMESTAMP,
        PRIMARY KEY (client_id, post_id)
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        client_id TEXT, profile TEXT, scraped_at TIMESTAMP, total_reels INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS raw_scrapes (
        client_id TEXT, id INTEGER, profile TEXT, raw JSON, scraped_at TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS reels (
        reel_id TEXT PRIMARY KEY, client_id TEXT, profile TEXT, shortcode TEXT, caption TEXT,
        likes INTEGER, views INTEGER, video_play_count INTEGER,
        comments_count INTEGER, video_url TEXT, audio_url TEXT,
        thumbnail_url TEXT, display_url TEXT, all_images JSON,
        timestamp TIMESTAMP, location_name TEXT, location_id TEXT,
        duration DOUBLE, is_pinned BOOLEAN, is_sponsored BOOLEAN,
        owner_username TEXT, owner_id TEXT, raw JSON, scraped_at TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        client_id TEXT, reel_id TEXT, comment_id TEXT, parent_id TEXT, username TEXT,
        text TEXT, likes INTEGER, timestamp TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tagged_users (
        client_id TEXT, reel_id TEXT, username TEXT, full_name TEXT,
        user_id TEXT, profile_pic_url TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS creator_accounts (
        account_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        full_name TEXT, bio TEXT, followers INTEGER, following INTEGER,
        post_count INTEGER, is_verified BOOLEAN DEFAULT FALSE,
        profile_pic_url TEXT, external_url TEXT, category TEXT,
        first_seen_at TIMESTAMP, last_scraped_at TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id TEXT PRIMARY KEY, client_id TEXT, account_id TEXT, caption TEXT,
        caption_clean TEXT, hashtags TEXT[], mentions TEXT[],
        likes INTEGER DEFAULT 0, views INTEGER DEFAULT 0,
        comments_count INTEGER DEFAULT 0, duration_sec DOUBLE DEFAULT 0,
        posted_at TIMESTAMP, scraped_at TIMESTAMP, video_url TEXT,
        audio_url TEXT, thumbnail_url TEXT, is_pinned BOOLEAN DEFAULT FALSE,
        is_sponsored BOOLEAN DEFAULT FALSE, engagement_rate DOUBLE DEFAULT 0,
        local_video_path TEXT, local_audio_path TEXT,
        download_status TEXT DEFAULT 'pending',
        downloaded_at TIMESTAMP, file_size_mb DOUBLE, raw_json JSON
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scrape_jobs (
        job_id TEXT PRIMARY KEY, client_id TEXT, username TEXT, started_at TIMESTAMP,
        finished_at TIMESTAMP, reels_found INTEGER DEFAULT 0,
        reels_new INTEGER DEFAULT 0, status TEXT DEFAULT 'running',
        error_msg TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        post_id TEXT PRIMARY KEY, client_id TEXT, provider TEXT, model TEXT, transcript TEXT,
        language TEXT, confidence DOUBLE, duration_sec DOUBLE, word_count INTEGER,
        transcribed_at TIMESTAMP, cost_usd DOUBLE, raw_response JSON,
        extraction_cost_usd DOUBLE
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcript_words (
        client_id TEXT, post_id TEXT, word_index INTEGER, word TEXT,
        start_sec DOUBLE, end_sec DOUBLE, confidence DOUBLE,
        PRIMARY KEY (post_id, word_index)
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS message_units (
        unit_id TEXT PRIMARY KEY, client_id TEXT, post_id TEXT, text TEXT, claim TEXT,
        advice TEXT, topic TEXT, subtopic TEXT, content_type TEXT,
        confidence DOUBLE, source_start DOUBLE, source_end DOUBLE,
        extracted_at TIMESTAMP, model TEXT,
        embedding FLOAT[1536], embedded_at TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS generated_content (
        gen_id TEXT PRIMARY KEY, client_id TEXT, created_at TIMESTAMP, topic TEXT,
        content_type TEXT, output_text TEXT, model TEXT,
        source_units TEXT[], tokens_used INTEGER, cost_usd DOUBLE
    )""")
