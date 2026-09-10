import streamlit as st
import pandas as pd
import json
from datetime import datetime

from core.client_context import render_client_selector
from core.db import get_connection, init_db

st.set_page_config(page_title="Data Viewer", page_icon="📊", layout="wide")
st.title("📊 Data Viewer")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]
st.caption(f"Scraped reels for {active_client['name']} — sorted by engagement rate")


PAGE_SIZE = 50  # rows per page


# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

conn = get_connection()

# Account filter — prefer posts table, fall back to raw_scrapes
try:
    accounts_df = conn.execute(
        """
        SELECT DISTINCT COALESCE(ca.username, p.account_id) AS profile
        FROM posts p
        JOIN client_posts cp ON p.post_id = cp.post_id
        LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
        WHERE cp.client_id = ?
        ORDER BY profile
        """,
        [client_id],
    ).df()
    source = "posts"
except Exception:
    accounts_df = conn.execute(
        """
        SELECT DISTINCT profile
        FROM raw_scrapes
        WHERE client_id = ?
        ORDER BY profile
        """,
        [client_id],
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
filter_key = f"{client_id}_{selected_account}_{sort_by}"
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state.data_page = 0
    st.session_state["_last_filter_key"] = filter_key


# ── Load from posts table (canonical) ─────────────────────────────────────────
def _total_posts(account_filter):
    where_clauses = ["cp.client_id = ?"]
    params = [client_id]
    if account_filter != "All":
        where_clauses.append("COALESCE(ca.username, p.account_id) = ?")
        params.append(account_filter)
    where = "WHERE " + " AND ".join(where_clauses)
    try:
        return conn.execute(
            f"""
            SELECT COUNT(*)
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            {where}
            """,
            params,
        ).fetchone()[0]
    except Exception:
        return 0


def load_posts(account_filter, order_by, page):
    where_clauses = ["cp.client_id = ?"]
    params = [client_id]
    if account_filter != "All":
        where_clauses.append("COALESCE(ca.username, p.account_id) = ?")
        params.append(account_filter)
    where = "WHERE " + " AND ".join(where_clauses)
    order_col = order_by if order_by in ("engagement_rate", "views", "likes", "posted_at") else "engagement_rate"
    offset = page * PAGE_SIZE
    params.extend([PAGE_SIZE, offset])
    try:
        df = conn.execute(f"""
            SELECT
                p.post_id,
                COALESCE(ca.username, p.account_id) AS account_id,
                p.caption,
                p.likes,
                p.views,
                p.comments_count,
                p.duration_sec,
                p.posted_at,
                p.hashtags,
                p.mentions,
                p.engagement_rate,
                p.video_url,
                p.thumbnail_url,
                p.is_pinned,
                p.is_sponsored,
                p.download_status
            FROM posts p
            JOIN client_posts cp ON p.post_id = cp.post_id
            LEFT JOIN creator_accounts ca ON p.account_id = ca.account_id
            {where}
            ORDER BY {order_col} DESC NULLS LAST
            LIMIT ? OFFSET ?
        """, params).df()
        return df, "posts"
    except Exception:
        return pd.DataFrame(), "empty"


def load_raw_fallback(account_filter, page):
    """Fall back to raw_scrapes if posts table is empty."""
    where_clauses = ["client_id = ?"]
    params = [client_id]
    if account_filter != "All":
        where_clauses.append("profile = ?")
        params.append(account_filter)
    where = "WHERE " + " AND ".join(where_clauses)
    offset = page * PAGE_SIZE
    params.extend([PAGE_SIZE, offset])
    df = conn.execute(f"""
        SELECT profile, raw, scraped_at FROM raw_scrapes
        {where} ORDER BY scraped_at DESC LIMIT ? OFFSET ?
    """, params).df()
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
    conn.close()
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

conn.close()
