import duckdb
import json
import uuid
import re
from datetime import datetime
from typing import Iterable, List, Optional

DB_PATH = "reels.duckdb"

DEFAULT_CLIENT_ID = "jobs_au_demo"
DEFAULT_CLIENT_NAME = "Jobs AU (demo)"
DEFAULT_CLIENT_NICHE = "Jobs in Australia"


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = ? AND column_name = ?
        """,
        [table, column],
    ).fetchone()
    return bool(row and row[0])


def _add_column_if_missing(conn, table: str, column: str, typedef: str):
    if not _column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")


def _seed_demo_client(conn):
    now = datetime.utcnow()
    conn.execute(
        """
        INSERT INTO clients (
            client_id, name, niche, created_at, notes,
            brand_voice_notes, target_audience
        )
        SELECT ?, ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM clients WHERE client_id = ?)
        """,
        [
            DEFAULT_CLIENT_ID,
            DEFAULT_CLIENT_NAME,
            DEFAULT_CLIENT_NICHE,
            now,
            "Seed client for the original Jobs-in-Australia competitor corpus.",
            "Helpful, direct, evidence-backed career advice.",
            "Job seekers and career switchers in Australia.",
            DEFAULT_CLIENT_ID,
        ],
    )

    try:
        from collection.account_list import JOBS_AU_SEED_ACCOUNTS, PRIORITY_ACCOUNTS
    except Exception:
        JOBS_AU_SEED_ACCOUNTS = []
        PRIORITY_ACCOUNTS = {}

    for handle in JOBS_AU_SEED_ACCOUNTS:
        conn.execute(
            """
            INSERT INTO client_accounts (
                client_id, instagram_handle, category, added_at, max_items
            )
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM client_accounts
                WHERE client_id = ? AND instagram_handle = ?
            )
            """,
            [
                DEFAULT_CLIENT_ID,
                handle,
                "Jobs/Recruitment",
                now,
                PRIORITY_ACCOUNTS.get(handle, 30),
                DEFAULT_CLIENT_ID,
                handle,
            ],
        )


def _backfill_client_id(conn, tables: Iterable[str]):
    for table in tables:
        _add_column_if_missing(conn, table, "client_id", "TEXT")
        conn.execute(
            f"UPDATE {table} SET client_id = ? WHERE client_id IS NULL",
            [DEFAULT_CLIENT_ID],
        )


def _backfill_client_post_links(conn):
    conn.execute(
        """
        INSERT INTO client_posts (client_id, post_id, added_at)
        SELECT DISTINCT
            COALESCE(p.client_id, ?),
            p.post_id,
            COALESCE(p.scraped_at, CURRENT_TIMESTAMP)
        FROM posts p
        WHERE p.post_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM client_posts cp
              WHERE cp.client_id = COALESCE(p.client_id, ?)
                AND cp.post_id = p.post_id
          )
        """,
        [DEFAULT_CLIENT_ID, DEFAULT_CLIENT_ID],
    )

    conn.execute(
        """
        INSERT INTO client_accounts (client_id, instagram_handle, category, added_at, max_items)
        SELECT DISTINCT
            COALESCE(p.client_id, ?),
            ca.username,
            COALESCE(ca.category, 'Competitor'),
            CURRENT_TIMESTAMP,
            30
        FROM posts p
        JOIN creator_accounts ca ON p.account_id = ca.account_id
        WHERE ca.username IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM client_accounts cacc
              WHERE cacc.client_id = COALESCE(p.client_id, ?)
                AND cacc.instagram_handle = ca.username
          )
        """,
        [DEFAULT_CLIENT_ID, DEFAULT_CLIENT_ID],
    )

def init_db():
    conn = duckdb.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        client_id         TEXT PRIMARY KEY,
        name              TEXT NOT NULL,
        niche             TEXT,
        created_at        TIMESTAMP,
        notes             TEXT,
        brand_voice_notes TEXT,
        target_audience   TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS client_accounts (
        client_id        TEXT,
        instagram_handle TEXT,
        category         TEXT,
        added_at         TIMESTAMP,
        max_items        INTEGER DEFAULT 30,
        PRIMARY KEY (client_id, instagram_handle)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS client_posts (
        client_id TEXT,
        post_id   TEXT,
        added_at  TIMESTAMP,
        PRIMARY KEY (client_id, post_id)
    )
    """)

    # Profile summary
    conn.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        client_id TEXT,
        profile TEXT,
        scraped_at TIMESTAMP,
        total_reels INTEGER
    )
    """)

    # Raw scrapes (NEW)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS raw_scrapes (
        client_id TEXT,
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
        client_id TEXT,
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
        client_id TEXT,
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
        client_id TEXT,
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
        client_id        TEXT,
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
        client_id    TEXT,
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
        client_id      TEXT,
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
        client_id  TEXT,
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
    conn.execute("""
    CREATE TABLE IF NOT EXISTS message_units (
        unit_id      TEXT PRIMARY KEY,
        client_id    TEXT,
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
        embedding    FLOAT[1536],
        embedded_at  TIMESTAMP
    )
    """)

    # Phase 7 — Generated content
    conn.execute("""
    CREATE TABLE IF NOT EXISTS generated_content (
        gen_id       TEXT PRIMARY KEY,
        client_id    TEXT,
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

    # Idempotent migrations for databases created before Phase 0 multi-client work.
    for col, typedef in [
        ("downloaded_at", "TIMESTAMP"),
        ("file_size_mb", "DOUBLE"),
        ("extraction_cost_usd", "DOUBLE"),
    ]:
        _add_column_if_missing(conn, "transcripts", col, typedef)

    for col, typedef in [
        ("downloaded_at", "TIMESTAMP"),
        ("file_size_mb", "DOUBLE"),
    ]:
        _add_column_if_missing(conn, "posts", col, typedef)

    _backfill_client_id(
        conn,
        [
            "profiles",
            "raw_scrapes",
            "reels",
            "comments",
            "tagged_users",
            "posts",
            "scrape_jobs",
            "transcripts",
            "transcript_words",
            "message_units",
            "generated_content",
        ],
    )
    _seed_demo_client(conn)
    _backfill_client_post_links(conn)

    # Phase 9 — Indexes for query performance
    for ddl in [
        "CREATE INDEX IF NOT EXISTS idx_client_accounts_client ON client_accounts(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_client_posts_client ON client_posts(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_client_posts_post ON client_posts(post_id)",
        "CREATE INDEX IF NOT EXISTS idx_posts_client_id ON posts(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_posts_account_id    ON posts(account_id)",
        "CREATE INDEX IF NOT EXISTS idx_posts_posted_at     ON posts(posted_at)",
        "CREATE INDEX IF NOT EXISTS idx_posts_download_status ON posts(download_status)",
        "CREATE INDEX IF NOT EXISTS idx_scrape_jobs_client_id ON scrape_jobs(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_generated_content_client_id ON generated_content(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_message_units_client_id ON message_units(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_message_units_topic ON message_units(topic)",
        "CREATE INDEX IF NOT EXISTS idx_message_units_post_id ON message_units(post_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcripts_client_id ON transcripts(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcripts_post_id ON transcripts(post_id)",
    ]:
        try:
            conn.execute(ddl)
        except Exception:
            pass   # index may already exist

    conn.close()


def get_connection():
    """Return a new DuckDB connection. Caller is responsible for closing it."""
    return duckdb.connect(DB_PATH)


# --------------------------------------------------------
#  RAW SCRAPE SAVER
# --------------------------------------------------------
def save_raw_scrape(profile, items, client_id: str = DEFAULT_CLIENT_ID):
    """Stores the raw JSON array exactly as returned. Skips duplicates by shortCode."""
    conn = duckdb.connect(DB_PATH)

    saved, skipped = 0, 0
    for idx, item in enumerate(items):
        shortcode = item.get("shortCode")

        # Dedup: skip if this shortcode already exists in raw_scrapes
        if shortcode:
            existing = conn.execute(
                """
                SELECT COUNT(*) FROM raw_scrapes
                WHERE client_id = ?
                  AND json_extract_string(raw, '$.shortCode') = ?
                """,
                (client_id, shortcode)
            ).fetchone()[0]
            if existing > 0:
                skipped += 1
                continue

        conn.execute("""
            INSERT INTO raw_scrapes (client_id, id, profile, raw, scraped_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            client_id,
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
def save_structured_scrape(profile, items, client_id: str = DEFAULT_CLIENT_ID):
    """Wrapper matching your old function name."""
    save_scrape(profile, items, client_id)


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
def save_scrape(profile, items, client_id: str = DEFAULT_CLIENT_ID):
    conn = duckdb.connect(DB_PATH)

    conn.execute("""
        INSERT INTO profiles (client_id, profile, scraped_at, total_reels)
        VALUES (?, ?, ?, ?)
    """, (client_id, profile, datetime.utcnow(), len(items)))

    for item in items:
        reel_id = item.get("id")

        conn.execute("""
            INSERT OR REPLACE INTO reels (
                reel_id, client_id, profile, shortcode, caption, likes, views,
                video_play_count, comments_count, video_url, audio_url,
                thumbnail_url, display_url, all_images, timestamp, location_name,
                location_id, duration, is_pinned, is_sponsored, owner_username,
                owner_id, raw, scraped_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            reel_id,                                    # reel_id
            client_id,                                  # client_id
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
                INSERT INTO tagged_users (
                    client_id, reel_id, username, full_name, user_id, profile_pic_url
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                client_id,
                reel_id,
                u.get("username"),
                u.get("full_name"),
                u.get("id"),
                u.get("profile_pic_url")
            ))

        # comments
        _insert_comments_recursive(conn, reel_id, item.get("latestComments", []), client_id=client_id)

    conn.close()


def _insert_comments_recursive(conn, reel_id, comments, parent_id=None, client_id: str = DEFAULT_CLIENT_ID):
    for c in comments:
        conn.execute("""
            INSERT INTO comments (
                client_id, reel_id, comment_id, parent_id, username, text, likes, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            client_id,
            reel_id,
            c.get("id"),
            parent_id,
            c.get("ownerUsername"),
            c.get("text"),
            c.get("likesCount"),
            fix_timestamp(c.get("timestamp"))
        ))

        if isinstance(c.get("replies"), list):
            _insert_comments_recursive(conn, reel_id, c["replies"], c.get("id"), client_id)


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


def _link_post_to_client(conn, client_id: str, post_id: str):
    conn.execute(
        """
        INSERT INTO client_posts (client_id, post_id, added_at)
        SELECT ?, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM client_posts WHERE client_id = ? AND post_id = ?
        )
        """,
        [client_id, post_id, datetime.utcnow(), client_id, post_id],
    )


def upsert_post(account_id: str, item: dict, client_id: str = DEFAULT_CLIENT_ID) -> bool:
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
                scraped_at = ?, video_url = ?, audio_url = ?,
                client_id = COALESCE(client_id, ?)
            WHERE post_id = ?
        """, (likes, views, item.get("commentsCount") or 0, engagement_rate,
              now, item.get("videoUrl"), item.get("audioUrl"), client_id, post_id))
    else:
        conn.execute("""
            INSERT INTO posts (
                post_id, client_id, account_id, caption, caption_clean, hashtags,
                mentions, likes, views, comments_count, duration_sec, posted_at,
                scraped_at, video_url, audio_url, thumbnail_url, is_pinned,
                is_sponsored, engagement_rate, local_video_path, local_audio_path,
                download_status, downloaded_at, file_size_mb, raw_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            post_id,
            client_id,
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
        ))

    _link_post_to_client(conn, client_id, post_id)
    conn.close()
    return is_new


def start_scrape_job(username: str, client_id: str = DEFAULT_CLIENT_ID) -> str:
    """Create a scrape_jobs row and return the job_id."""
    conn = duckdb.connect(DB_PATH)
    job_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO scrape_jobs (
            job_id, client_id, username, started_at, finished_at,
            reels_found, reels_new, status, error_msg
        )
        VALUES (?, ?, ?, ?, NULL, 0, 0, 'running', NULL)
    """, (job_id, client_id, username, datetime.utcnow()))
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


def get_scrape_history(client_id: Optional[str] = DEFAULT_CLIENT_ID) -> list:
    """Return recent scrape jobs ordered by newest first."""
    conn = duckdb.connect(DB_PATH)
    if client_id:
        rows = conn.execute("""
            SELECT job_id, username, started_at, finished_at,
                   reels_found, reels_new, status, error_msg
            FROM scrape_jobs
            WHERE client_id = ?
            ORDER BY started_at DESC LIMIT 50
        """, [client_id]).fetchall()
    else:
        rows = conn.execute("""
            SELECT job_id, username, started_at, finished_at,
                   reels_found, reels_new, status, error_msg
            FROM scrape_jobs ORDER BY started_at DESC LIMIT 50
        """).fetchall()
    conn.close()
    return rows


def get_posts_stats(client_id: Optional[str] = DEFAULT_CLIENT_ID) -> dict:
    """Return aggregate stats for the posts table."""
    conn = duckdb.connect(DB_PATH)
    if client_id:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT p.account_id) as accounts,
                   AVG(p.engagement_rate) as avg_engagement,
                   SUM(CASE WHEN p.download_status = 'done' THEN 1 ELSE 0 END) as downloaded
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            WHERE cp.client_id = ?
        """, [client_id]).fetchone()
    else:
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
