import streamlit as st
import duckdb
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="Data Viewer", page_icon="📊", layout="wide")
st.title("📊 Data Viewer")
st.caption("All scraped reels — sorted by engagement rate")


PAGE_SIZE = 50  # rows per page


@st.cache_resource
def get_conn():
    return duckdb.connect("reels.duckdb")


# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

conn = get_conn()

# Account filter — prefer posts table, fall back to raw_scrapes
try:
    accounts_df = conn.execute(
        "SELECT DISTINCT account_id as profile FROM posts ORDER BY account_id"
    ).df()
    source = "posts"
except Exception:
    accounts_df = conn.execute(
        "SELECT DISTINCT profile FROM raw_scrapes ORDER BY profile"
    ).df()
    source = "raw_scrapes"

available_accounts = ["All"] + accounts_df.iloc[:, 0].tolist()
selected_account = st.sidebar.selectbox("Account", available_accounts)
sort_by = st.sidebar.selectbox("Sort by", ["engagement_rate", "views", "likes", "posted_at"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: `{source}` table")

# ── Page state ─────────────────────────────────────────────────────────────────
if "data_page" not in st.session_state:
    st.session_state.data_page = 0

# Reset to page 0 when filters change
filter_key = f"{selected_account}_{sort_by}"
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state.data_page = 0
    st.session_state["_last_filter_key"] = filter_key


# ── Load from posts table (canonical) ─────────────────────────────────────────
def _total_posts(account_filter):
    where = "" if account_filter == "All" else f"WHERE account_id = '{account_filter}'"
    try:
        return conn.execute(f"SELECT COUNT(*) FROM posts {where}").fetchone()[0]
    except Exception:
        return 0


def load_posts(account_filter, order_by, page):
    where = "" if account_filter == "All" else f"WHERE account_id = '{account_filter}'"
    order_col = order_by if order_by in ("engagement_rate", "views", "likes", "posted_at") else "engagement_rate"
    offset = page * PAGE_SIZE
    try:
        df = conn.execute(f"""
            SELECT
                post_id,
                account_id,
                caption,
                likes,
                views,
                comments_count,
                duration_sec,
                posted_at,
                hashtags,
                mentions,
                engagement_rate,
                video_url,
                thumbnail_url,
                is_pinned,
                is_sponsored,
                download_status
            FROM posts
            {where}
            ORDER BY {order_col} DESC NULLS LAST
            LIMIT {PAGE_SIZE} OFFSET {offset}
        """).df()
        return df, "posts"
    except Exception:
        return pd.DataFrame(), "empty"


def load_raw_fallback(account_filter, page):
    """Fall back to raw_scrapes if posts table is empty."""
    where = "" if account_filter == "All" else f"WHERE profile = '{account_filter}'"
    offset = page * PAGE_SIZE
    df = conn.execute(f"""
        SELECT profile, raw, scraped_at FROM raw_scrapes
        {where} ORDER BY scraped_at DESC LIMIT {PAGE_SIZE} OFFSET {offset}
    """).df()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        try:
            r = json.loads(row["raw"])
            likes = r.get("likesCount") or 0
            views = r.get("videoViewCount") or 0
            rows.append({
                "post_id": r.get("shortCode", ""),
                "account_id": row["profile"],
                "caption": str(r.get("caption", "") or "")[:120],
                "likes": likes,
                "views": views,
                "comments_count": r.get("commentsCount") or 0,
                "duration_sec": r.get("videoDuration") or 0,
                "posted_at": r.get("timestamp", "")[:10],
                "engagement_rate": round(likes / views * 100, 2) if views > 0 else 0,
                "video_url": r.get("videoUrl", ""),
                "download_status": "pending",
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


with st.spinner("Loading…"):
    total_rows = _total_posts(selected_account)
    df, source_used = load_posts(selected_account, sort_by, st.session_state.data_page)
    if df.empty and st.session_state.data_page == 0:
        df = load_raw_fallback(selected_account, 0)
        source_used = "raw_scrapes (fallback)"

if df.empty and st.session_state.data_page == 0:
    st.warning("⚠️ No data yet. Go to the **Scraper** page (home) and run a scrape first.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
current_page = st.session_state.data_page
first_row = current_page * PAGE_SIZE + 1
last_row = min(first_row + len(df) - 1, total_rows)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total reels", f"{total_rows:,}")
c2.metric("Avg Likes", f"{df['likes'].mean():,.0f}" if "likes" in df else "—")
c3.metric("Avg Views", f"{df['views'].mean():,.0f}" if "views" in df else "—")
c4.metric("Avg Engagement", f"{df['engagement_rate'].mean():.2f}%" if "engagement_rate" in df else "—")
c5.metric("Source", source_used)

st.markdown("---")

# ── Search ─────────────────────────────────────────────────────────────────────
search = st.text_input("🔍 Search captions or account", placeholder="e.g., ATS, resume, interview")
if search and "caption" in df.columns:
    mask = (
        df["caption"].str.contains(search, case=False, na=False) |
        df["account_id"].str.contains(search, case=False, na=False)
    )
    df = df[mask]
    st.caption(f"{len(df)} results for '{search}'")

# ── Main table ─────────────────────────────────────────────────────────────────
col_cfg = {}
if "video_url" in df.columns:
    col_cfg["video_url"] = st.column_config.LinkColumn("Video", display_text="🎥 Watch")
if "engagement_rate" in df.columns:
    col_cfg["engagement_rate"] = st.column_config.NumberColumn("Engagement %", format="%.2f")
if "duration_sec" in df.columns:
    col_cfg["duration_sec"] = st.column_config.NumberColumn("Duration (s)", format="%.0f")

# Truncate caption for table display
if "caption" in df.columns:
    df = df.copy()
    df["caption"] = df["caption"].str.slice(0, 100)

st.dataframe(df, width="stretch", hide_index=True, column_config=col_cfg)

# ── Pagination controls ────────────────────────────────────────────────────────
st.markdown("---")
pg_left, pg_mid, pg_right = st.columns([1, 2, 1])

with pg_left:
    if st.button("◀ Previous", disabled=(current_page == 0), width="stretch"):
        st.session_state.data_page -= 1
        st.rerun()

with pg_mid:
    st.markdown(
        f"<div style='text-align:center;padding-top:8px'>Page {current_page + 1} of {total_pages} &nbsp;·&nbsp; rows {first_row}–{last_row} of {total_rows:,}</div>",
        unsafe_allow_html=True,
    )

with pg_right:
    if st.button("Next ▶", disabled=(current_page >= total_pages - 1), width="stretch"):
        st.session_state.data_page += 1
        st.rerun()

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Download current page:")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False),
        file_name=f"reels_{selected_account}_p{current_page + 1}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
    )
with dl2:
    st.download_button(
        "📥 Download JSON",
        data=df.to_json(orient="records", indent=2),
        file_name=f"reels_{selected_account}_p{current_page + 1}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        width="stretch",
    )

# ── Full Report Export ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Full Report Export")
st.caption(
    "Complete dataset — all accounts, all rows, joins metadata + transcripts + extracted knowledge units. "
    "Designed for sharing with clients as a competitor content analysis report."
)

exp_account = st.selectbox(
    "Filter by account (or All)",
    available_accounts,
    key="export_account",
)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("**📄 Posts + Transcripts report**")
    st.caption(
        "One row per reel. Includes: creator profile, engagement metrics, "
        "caption, hashtags, full transcript, transcript stats, and a summary of "
        "extracted knowledge units (count, topics, content types)."
    )
    if st.button("⚙️ Build Posts Report", key="btn_build_posts"):
        with st.spinner("Querying database…"):
            try:
                account_where = "" if exp_account == "All" else f"WHERE p.account_id = '{exp_account}'"
                posts_df = conn.execute(f"""
                    SELECT
                        p.post_id,
                        COALESCE(ca.username, p.account_id)        AS username,
                        ca.full_name,
                        ca.followers,
                        ca.following,
                        ca.post_count                              AS creator_total_posts,
                        ca.is_verified,
                        ca.category                               AS creator_category,
                        ca.bio,
                        p.likes,
                        p.views,
                        p.comments_count,
                        ROUND(p.engagement_rate, 4)               AS engagement_rate_pct,
                        ROUND(p.duration_sec, 1)                  AS duration_sec,
                        CAST(p.posted_at AS TEXT)                 AS posted_at,
                        CAST(p.scraped_at AS TEXT)                AS scraped_at,
                        p.caption_clean                           AS caption,
                        array_to_string(p.hashtags,  ', ')        AS hashtags,
                        array_to_string(p.mentions,  ', ')        AS mentions,
                        p.is_pinned,
                        p.is_sponsored,
                        p.video_url,
                        p.download_status,
                        t.transcript,
                        t.language                               AS transcript_language,
                        t.word_count                             AS transcript_word_count,
                        ROUND(t.confidence, 3)                   AS transcript_confidence,
                        t.provider                               AS transcript_provider,
                        CAST(t.transcribed_at AS TEXT)           AS transcribed_at,
                        COUNT(mu.unit_id)                        AS knowledge_units_extracted,
                        string_agg(DISTINCT mu.topic,        ', ' ORDER BY mu.topic)
                                                                 AS topics_covered,
                        string_agg(DISTINCT mu.content_type, ', ' ORDER BY mu.content_type)
                                                                 AS content_types_found,
                        string_agg(DISTINCT mu.subtopic,     ' | ' ORDER BY mu.subtopic)
                                                                 AS subtopics
                    FROM posts p
                    LEFT JOIN creator_accounts ca  ON p.account_id = ca.account_id
                    LEFT JOIN transcripts t         ON p.post_id    = t.post_id
                    LEFT JOIN message_units mu      ON p.post_id    = mu.post_id
                    {account_where}
                    GROUP BY
                        p.post_id, p.account_id, ca.username, ca.full_name, ca.followers, ca.following,
                        ca.post_count, ca.is_verified, ca.category, ca.bio,
                        p.likes, p.views, p.comments_count, p.engagement_rate,
                        p.duration_sec, p.posted_at, p.scraped_at,
                        p.caption_clean, p.hashtags, p.mentions,
                        p.is_pinned, p.is_sponsored, p.video_url, p.download_status,
                        t.transcript, t.language, t.word_count, t.confidence,
                        t.provider, t.transcribed_at
                    ORDER BY p.posted_at DESC NULLS LAST
                """).df()
                st.session_state["export_posts_df"] = posts_df
                st.success(f"✅ {len(posts_df):,} rows · {len(posts_df.columns)} columns")
            except Exception as e:
                st.error(f"Query failed: {e}")

    if "export_posts_df" in st.session_state:
        df_p = st.session_state["export_posts_df"]
        fname = f"competitor_analysis_posts_{exp_account}_{datetime.now().strftime('%Y%m%d')}.csv"
        st.download_button(
            "📥 Download Posts Report CSV",
            data=df_p.to_csv(index=False),
            file_name=fname,
            mime="text/csv",
            key="dl_posts_report",
        )
        with st.expander("Preview first 5 rows"):
            st.dataframe(df_p.head(5), use_container_width=True, hide_index=True)

with col_exp2:
    st.markdown("**🧠 Knowledge Units report**")
    st.caption(
        "One row per extracted insight. Each reel may have multiple rows — one per "
        "tip, stat, warning, myth, or hook pulled from the transcript. "
        "Includes the source reel's metrics so you can rank insights by reach."
    )
    if st.button("⚙️ Build Knowledge Units Report", key="btn_build_units"):
        with st.spinner("Querying database…"):
            try:
                account_where_mu = "" if exp_account == "All" else f"WHERE p.account_id = '{exp_account}'"
                units_df = conn.execute(f"""
                    SELECT
                        mu.unit_id,
                        mu.post_id,
                        COALESCE(ca.username, p.account_id)    AS username,
                        ca.full_name,
                        ca.followers,
                        ca.is_verified,
                        ca.category                           AS creator_category,
                        p.likes,
                        p.views,
                        ROUND(p.engagement_rate, 4)           AS engagement_rate_pct,
                        CAST(p.posted_at AS TEXT)             AS posted_at,
                        mu.topic,
                        mu.subtopic,
                        mu.content_type,
                        mu.text                               AS insight_text,
                        mu.claim,
                        mu.advice,
                        ROUND(mu.confidence, 3)               AS extraction_confidence,
                        CAST(mu.extracted_at AS TEXT)         AS extracted_at,
                        array_to_string(p.hashtags, ', ')     AS reel_hashtags,
                        LEFT(p.caption_clean, 200)            AS reel_caption_preview,
                        LEFT(t.transcript, 500)               AS transcript_preview
                    FROM message_units mu
                    LEFT JOIN posts p         ON mu.post_id    = p.post_id
                    LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
                    LEFT JOIN transcripts t    ON mu.post_id   = t.post_id
                    {account_where_mu}
                    ORDER BY p.posted_at DESC NULLS LAST, mu.content_type, mu.topic
                """).df()
                st.session_state["export_units_df"] = units_df
                st.success(f"✅ {len(units_df):,} rows · {len(units_df.columns)} columns")
            except Exception as e:
                st.error(f"Query failed: {e}")

    if "export_units_df" in st.session_state:
        df_u = st.session_state["export_units_df"]
        fname_u = f"competitor_analysis_knowledge_units_{exp_account}_{datetime.now().strftime('%Y%m%d')}.csv"
        st.download_button(
            "📥 Download Knowledge Units CSV",
            data=df_u.to_csv(index=False),
            file_name=fname_u,
            mime="text/csv",
            key="dl_units_report",
        )
        with st.expander("Preview first 5 rows"):
            st.dataframe(df_u.head(5), use_container_width=True, hide_index=True)


