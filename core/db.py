import duckdb
import json
import uuid
import re
from datetime import datetime
from typing import List, Optional

DB_PATH = "reels.duckdb"

def init_db():
    conn = duckdb.connect(DB_PATH)

    # Profile summary
    conn.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        profile TEXT,
        scraped_at TIMESTAMP,
        total_reels INTEGER
    )
    """)

    # Raw scrapes (NEW)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS raw_scrapes (
        id INTEGER,
        profile TEXT,
        raw JSON,
        scraped_at TIMESTAMP
    )
    """)

    # Reels table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS reels (
        reel_id TEXT PRIMARY KEY,
        profile TEXT,
        shortcode TEXT,
        caption TEXT,
        likes INTEGER,
        views INTEGER,
        video_play_count INTEGER,
        comments_count INTEGER,
        video_url TEXT,
        audio_url TEXT,
        thumbnail_url TEXT,
        display_url TEXT,
        all_images JSON,
        timestamp TIMESTAMP,
        location_name TEXT,
        location_id TEXT,
        duration DOUBLE,
        is_pinned BOOLEAN,
        is_sponsored BOOLEAN,
        owner_username TEXT,
        owner_id TEXT,
        raw JSON,
        scraped_at TIMESTAMP
    )
    """)

    # Comments
    conn.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        reel_id TEXT,
        comment_id TEXT,
        parent_id TEXT,
        username TEXT,
        text TEXT,
        likes INTEGER,
        timestamp TIMESTAMP
    )
    """)

    # Tagged users
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tagged_users (
        reel_id TEXT,
        username TEXT,
        full_name TEXT,
        user_id TEXT,
        profile_pic_url TEXT
    )
    """)

    # ---- Phase 1: Canonical tables ----

    conn.execute("""
    CREATE TABLE IF NOT EXISTS creator_accounts (
        account_id      TEXT PRIMARY KEY,
        username        TEXT UNIQUE NOT NULL,
        full_name       TEXT,
        bio             TEXT,
        followers       INTEGER,
        following       INTEGER,
        post_count      INTEGER,
        is_verified     BOOLEAN DEFAULT FALSE,
        profile_pic_url TEXT,
        external_url    TEXT,
        category        TEXT,
        first_seen_at   TIMESTAMP,
        last_scraped_at TIMESTAMP
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id          TEXT PRIMARY KEY,
        account_id       TEXT,
        caption          TEXT,
        caption_clean    TEXT,
        hashtags         TEXT[],
        mentions         TEXT[],
        likes            INTEGER DEFAULT 0,
        views            INTEGER DEFAULT 0,
        comments_count   INTEGER DEFAULT 0,
        duration_sec     DOUBLE DEFAULT 0,
        posted_at        TIMESTAMP,
        scraped_at       TIMESTAMP,
        video_url        TEXT,
        audio_url        TEXT,
        thumbnail_url    TEXT,
        is_pinned        BOOLEAN DEFAULT FALSE,
        is_sponsored     BOOLEAN DEFAULT FALSE,
        engagement_rate  DOUBLE DEFAULT 0,
        local_video_path TEXT,
        local_audio_path TEXT,
        download_status  TEXT DEFAULT 'pending',
        downloaded_at    TIMESTAMP,
        file_size_mb     DOUBLE,
        raw_json         JSON
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS scrape_jobs (
        job_id       TEXT PRIMARY KEY,
        username     TEXT,
        started_at   TIMESTAMP,
        finished_at  TIMESTAMP,
        reels_found  INTEGER DEFAULT 0,
        reels_new    INTEGER DEFAULT 0,
        status       TEXT DEFAULT 'running',
        error_msg    TEXT
    )
    """)

    # Phase 3 — Transcripts
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        post_id        TEXT PRIMARY KEY,
        provider       TEXT,
        model          TEXT,
        transcript     TEXT,
        language       TEXT,
        confidence     DOUBLE,
        duration_sec   DOUBLE,
        word_count     INTEGER,
        transcribed_at TIMESTAMP,
        cost_usd       DOUBLE,
        raw_response   JSON
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcript_words (
        post_id    TEXT,
        word_index INTEGER,
        word       TEXT,
        start_sec  DOUBLE,
        end_sec    DOUBLE,
        confidence DOUBLE,
        PRIMARY KEY (post_id, word_index)
    )
    """)

    # Phase 4 — Message units (knowledge extraction)
    # Phase 10 note: embedding column is FLOAT[512] (Voyage voyage-3-lite)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS message_units (
        unit_id      TEXT PRIMARY KEY,
        post_id      TEXT,
        text         TEXT,
        claim        TEXT,
        advice       TEXT,
        topic        TEXT,
        subtopic     TEXT,
        content_type TEXT,
        confidence   DOUBLE,
        source_start DOUBLE,
        source_end   DOUBLE,
        extracted_at TIMESTAMP,
        model        TEXT,
        embedding    FLOAT[512],
        embedded_at  TIMESTAMP
    )
    """)

    # Phase 7 — Generated content
    conn.execute("""
    CREATE TABLE IF NOT EXISTS generated_content (
        gen_id       TEXT PRIMARY KEY,
        created_at   TIMESTAMP,
        topic        TEXT,
        content_type TEXT,
        output_text  TEXT,
        model        TEXT,
        source_units TEXT[],
        tokens_used  INTEGER,
        cost_usd     DOUBLE
    )
    """)

    # Migration: add Phase 2 columns if they don't exist yet
    for col, typedef in [
        ("downloaded_at", "TIMESTAMP"),
        ("file_size_mb", "DOUBLE"),
        # Phase 4
        ("extraction_cost_usd", "DOUBLE"),
    ]:
        try:
            conn.execute(f"ALTER TABLE transcripts ADD COLUMN {col} {typedef}")
        except Exception:
            pass

    for col, typedef in [
        ("downloaded_at", "TIMESTAMP"),
        ("file_size_mb", "DOUBLE"),
        # Phase 10 — OCR text from video frames
        ("ocr_text", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists

    # Phase 10 — Migrate embedding column from FLOAT[1536] to FLOAT[512]
    # DuckDB does not support ALTER COLUMN type, so we recreate the table.
    # This clears existing embeddings; they must be re-embedded after migration.
    try:
        col_info = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'message_units' AND column_name = 'embedding'"
        ).fetchone()
        if col_info and "1536" in str(col_info[0]):
            conn.execute("""
                CREATE TABLE message_units_new AS
                SELECT unit_id, post_id, text, claim, advice, topic, subtopic,
                       content_type, confidence, source_start, source_end,
                       extracted_at, model,
                       NULL::FLOAT[512] AS embedding,
                       NULL::TIMESTAMP   AS embedded_at
                FROM message_units
            """)
            conn.execute("DROP TABLE message_units")
            conn.execute("ALTER TABLE message_units_new RENAME TO message_units")
    except Exception:
        pass  # table may not exist yet or already migrated

    # Phase 9 — Indexes for query performance
    for ddl in [
        "CREATE INDEX IF NOT EXISTS idx_posts_account_id    ON posts(account_id)",
        "CREATE INDEX IF NOT EXISTS idx_posts_posted_at     ON posts(posted_at)",
        "CREATE INDEX IF NOT EXISTS idx_posts_download_status ON posts(download_status)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_message_units_unit_id  ON message_units(unit_id)",
        "CREATE INDEX IF NOT EXISTS idx_message_units_topic ON message_units(topic)",
        "CREATE INDEX IF NOT EXISTS idx_message_units_post_id ON message_units(post_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcripts_post_id ON transcripts(post_id)",
    ]:
        try:
            conn.execute(ddl)
        except Exception:
            pass   # index may already exist

    conn.close()


def get_connection(read_only: bool = False, retries: int = 5, retry_delay: float = 1.0):
    """
    Return a new DuckDB connection.

    Args:
        read_only:   Open in read-only mode (allows concurrent readers alongside a writer).
        retries:     Number of times to retry on lock conflict before raising.
        retry_delay: Seconds to wait between retries.

    Caller is responsible for closing it.
    """
    import time as _time
    last_err = None
    for attempt in range(retries):
        try:
            return duckdb.connect(DB_PATH, read_only=read_only)
        except Exception as e:
            if "lock" in str(e).lower() or "IO Error" in str(e):
                last_err = e
                if attempt < retries - 1:
                    _time.sleep(retry_delay)
            else:
                raise
    raise last_err


# --------------------------------------------------------
#  RAW SCRAPE SAVER
# --------------------------------------------------------
def save_raw_scrape(profile, items):
    """Stores the raw JSON array exactly as returned. Skips duplicates by shortCode."""
    conn = duckdb.connect(DB_PATH)

    saved, skipped = 0, 0
    for idx, item in enumerate(items):
        shortcode = item.get("shortCode")

        # Dedup: skip if this shortcode already exists in raw_scrapes
        if shortcode:
            existing = conn.execute(
                "SELECT COUNT(*) FROM raw_scrapes WHERE json_extract_string(raw, '$.shortCode') = ?",
                (shortcode,)
            ).fetchone()[0]
            if existing > 0:
                skipped += 1
                continue

        conn.execute("""
            INSERT INTO raw_scrapes VALUES (?, ?, ?, ?)
        """, (
            idx,
            profile,
            json.dumps(item),
            datetime.utcnow()
        ))
        saved += 1

    conn.close()
    return f"{saved} new, {skipped} already existed"


# --------------------------------------------------------
#  STRUCTURED SCRAPE SAVER
# --------------------------------------------------------
def save_structured_scrape(profile, items):
    """Wrapper matching your old function name."""
    save_scrape(profile, items)


def fix_timestamp(ts):
    if not ts:
        return None
    # Remove Z and milliseconds
    ts = ts.replace("Z", "")
    if "." in ts:
        ts = ts.split(".")[0]
    return ts.replace("T", " ")

# --------------------------------------------------------
#  STRUCTURED LOADER (YOUR ORIGINAL)
# --------------------------------------------------------
def save_scrape(profile, items):
    conn = duckdb.connect(DB_PATH)

    conn.execute("""
        INSERT INTO profiles VALUES (?, ?, ?)
    """, (profile, datetime.utcnow(), len(items)))

    for item in items:
        reel_id = item.get("id")

        conn.execute("""
            INSERT OR REPLACE INTO reels VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            reel_id,                                    # reel_id
            profile,                                    # profile
            item.get("shortCode"),                      # shortcode
            item.get("caption"),                        # caption
            item.get("likesCount"),                     # likes
            item.get("videoViewCount"),                 # views
            item.get("videoPlayCount"),                 # video_play_count
            item.get("commentsCount"),                  # comments_count
            item.get("videoUrl"),                       # video_url
            item.get("audioUrl"),                       # audio_url
            item["images"][0] if item.get("images") else None,  # thumbnail_url
            item.get("displayUrl"),                     # display_url
            json.dumps(item.get("images", [])),         # all_images
            fix_timestamp(item.get("timestamp")),       # timestamp
            item.get("locationName"),                   # location_name
            item.get("locationId"),                     # location_id
            item.get("videoDuration"),                  # duration
            bool(item.get("isPinned")),                 # is_pinned
            bool(item.get("isSponsored")),              # is_sponsored
            item.get("ownerUsername"),                  # owner_username
            item.get("ownerId"),                        # owner_id
            json.dumps(item),                           # raw
            datetime.utcnow()                           # scraped_at
        ))

        # tagged users
        for u in item.get("taggedUsers", []):
            conn.execute("""
                INSERT INTO tagged_users VALUES (?, ?, ?, ?, ?)
            """, (
                reel_id,
                u.get("username"),
                u.get("full_name"),
                u.get("id"),
                u.get("profile_pic_url")
            ))

        # comments
        _insert_comments_recursive(conn, reel_id, item.get("latestComments", []))

    conn.close()


def _insert_comments_recursive(conn, reel_id, comments, parent_id=None):
    for c in comments:
        conn.execute("""
            INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            reel_id,
            c.get("id"),
            parent_id,
            c.get("ownerUsername"),
            c.get("text"),
            c.get("likesCount"),
            fix_timestamp(c.get("timestamp"))
        ))

        if isinstance(c.get("replies"), list):
            _insert_comments_recursive(conn, reel_id, c["replies"], c.get("id"))


# --------------------------------------------------------
#  PHASE 1: CANONICAL SAVERS
# --------------------------------------------------------

def _extract_hashtags(caption: Optional[str]) -> List[str]:
    if not caption:
        return []
    return re.findall(r"#(\w+)", caption)


def _extract_mentions(caption: Optional[str]) -> List[str]:
    if not caption:
        return []
    return re.findall(r"@(\w+)", caption)


def _clean_caption(caption: Optional[str]) -> Optional[str]:
    if not caption:
        return None
    cleaned = re.sub(r"#\w+", "", caption)
    cleaned = re.sub(r"@\w+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def upsert_creator_account(username: str, item: dict) -> str:
    """Insert or update a creator account from a scraped Apify item.
    Returns the account_id (Instagram user ID)."""
    conn = duckdb.connect(DB_PATH)
    account_id = str(item.get("ownerId") or item.get("owner", {}).get("id") or username)
    now = datetime.utcnow()

    existing = conn.execute(
        "SELECT account_id FROM creator_accounts WHERE username = ?", (username,)
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE creator_accounts
            SET last_scraped_at = ?, followers = COALESCE(?, followers)
            WHERE username = ?
        """, (now, item.get("followersCount"), username))
    else:
        conn.execute("""
            INSERT INTO creator_accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            account_id,
            username,
            item.get("ownerFullName"),
            item.get("biography"),
            item.get("followersCount"),
            item.get("followsCount"),
            item.get("postsCount"),
            bool(item.get("verified")),
            item.get("profilePicUrl"),
            item.get("externalUrl"),
            "Jobs/Recruitment",
            now,
            now,
        ))

    conn.close()
    return account_id


def upsert_post(account_id: str, item: dict) -> bool:
    """Insert or update a post from a scraped Apify item.
    Returns True if this was a new post, False if updated."""
    conn = duckdb.connect(DB_PATH)
    post_id = item.get("shortCode") or item.get("id")
    if not post_id:
        conn.close()
        return False

    caption = item.get("caption")
    hashtags = _extract_hashtags(caption)
    mentions = _extract_mentions(caption)
    caption_clean = _clean_caption(caption)
    likes = item.get("likesCount") or 0
    views = item.get("videoViewCount") or 0
    engagement_rate = round((likes / views * 100), 4) if views > 0 else 0.0
    now = datetime.utcnow()

    thumbnail = None
    images = item.get("images")
    if images and isinstance(images, list) and len(images) > 0:
        thumbnail = images[0]

    existing = conn.execute(
        "SELECT post_id FROM posts WHERE post_id = ?", (post_id,)
    ).fetchone()

    is_new = existing is None

    if existing:
        # Update engagement stats only (keep download status etc)
        conn.execute("""
            UPDATE posts
            SET likes = ?, views = ?, comments_count = ?, engagement_rate = ?,
                scraped_at = ?, video_url = ?, audio_url = ?
            WHERE post_id = ?
        """, (likes, views, item.get("commentsCount") or 0, engagement_rate,
              now, item.get("videoUrl"), item.get("audioUrl"), post_id))
    else:
        conn.execute("""
            INSERT INTO posts (
                post_id, account_id, caption, caption_clean,
                hashtags, mentions, likes, views,
                comments_count, duration_sec, posted_at, scraped_at,
                video_url, audio_url, thumbnail_url,
                is_pinned, is_sponsored, engagement_rate,
                local_video_path, local_audio_path,
                download_status, downloaded_at, file_size_mb,
                raw_json, ocr_text
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?
            )
        """, (
            post_id,
            account_id,
            caption,
            caption_clean,
            hashtags,
            mentions,
            likes,
            views,
            item.get("commentsCount") or 0,
            item.get("videoDuration") or 0.0,
            fix_timestamp(item.get("timestamp")),
            now,
            item.get("videoUrl"),
            item.get("audioUrl"),
            thumbnail,
            bool(item.get("isPinned")),
            bool(item.get("isSponsored")),
            engagement_rate,
            None,    # local_video_path
            None,    # local_audio_path
            "pending",
            None,    # downloaded_at
            None,    # file_size_mb
            json.dumps(item),
            None,    # ocr_text (Phase 10)
        ))

    conn.close()
    return is_new


def start_scrape_job(username: str) -> str:
    """Create a scrape_jobs row and return the job_id."""
    conn = duckdb.connect(DB_PATH)
    job_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO scrape_jobs VALUES (?, ?, ?, NULL, 0, 0, 'running', NULL)
    """, (job_id, username, datetime.utcnow()))
    conn.close()
    return job_id


def finish_scrape_job(job_id: str, reels_found: int, reels_new: int,
                      status: str = "done", error_msg: Optional[str] = None):
    """Mark a scrape job as finished."""
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        UPDATE scrape_jobs
        SET finished_at = ?, reels_found = ?, reels_new = ?, status = ?, error_msg = ?
        WHERE job_id = ?
    """, (datetime.utcnow(), reels_found, reels_new, status, error_msg, job_id))
    conn.close()


def get_scrape_history() -> list:
    """Return recent scrape jobs ordered by newest first."""
    conn = duckdb.connect(DB_PATH)
    rows = conn.execute("""
        SELECT job_id, username, started_at, finished_at,
               reels_found, reels_new, status, error_msg
        FROM scrape_jobs ORDER BY started_at DESC LIMIT 50
    """).fetchall()
    conn.close()
    return rows


def get_posts_stats() -> dict:
    """Return aggregate stats for the posts table."""
    conn = duckdb.connect(DB_PATH)
    row = conn.execute("""
        SELECT COUNT(*) as total,
               COUNT(DISTINCT account_id) as accounts,
               AVG(engagement_rate) as avg_engagement,
               SUM(CASE WHEN download_status = 'done' THEN 1 ELSE 0 END) as downloaded
        FROM posts
    """).fetchone()
    conn.close()
    return {
        "total_posts": row[0] or 0,
        "accounts": row[1] or 0,
        "avg_engagement": round(row[2] or 0, 2),
        "downloaded": row[3] or 0,
    }

