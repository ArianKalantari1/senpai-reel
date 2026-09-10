from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Iterable, Optional

from core.db import (
    DEFAULT_CLIENT_ID,
    DEFAULT_CLIENT_NAME,
    DEFAULT_CLIENT_NICHE,
    get_connection,
    init_db,
)


def normalize_handle(handle: str) -> str:
    """Return an Instagram handle without @, URL parts, or surrounding whitespace."""
    handle = (handle or "").strip()
    handle = re.sub(r"^https?://(www\.)?instagram\.com/", "", handle)
    handle = handle.strip().strip("/").split("/")[0]
    return handle.lstrip("@").strip().lower()


def list_clients() -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT client_id, name, niche, created_at, notes,
                   brand_voice_notes, target_audience
            FROM clients
            ORDER BY
                CASE WHEN client_id = ? THEN 0 ELSE 1 END,
                created_at,
                name
            """,
            [DEFAULT_CLIENT_ID],
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "client_id": r[0],
            "name": r[1],
            "niche": r[2],
            "created_at": r[3],
            "notes": r[4],
            "brand_voice_notes": r[5],
            "target_audience": r[6],
        }
        for r in rows
    ]


def get_client(client_id: str = DEFAULT_CLIENT_ID) -> Optional[dict]:
    init_db()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT client_id, name, niche, created_at, notes,
                   brand_voice_notes, target_audience
            FROM clients
            WHERE client_id = ?
            """,
            [client_id],
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    return {
        "client_id": row[0],
        "name": row[1],
        "niche": row[2],
        "created_at": row[3],
        "notes": row[4],
        "brand_voice_notes": row[5],
        "target_audience": row[6],
    }


def create_client(
    name: str,
    niche: str,
    notes: str = "",
    brand_voice_notes: str = "",
    target_audience: str = "",
    accounts: Optional[Iterable[str]] = None,
) -> str:
    init_db()
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Client name is required")

    client_id = f"client_{uuid.uuid4().hex[:12]}"
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO clients (
                client_id, name, niche, created_at, notes,
                brand_voice_notes, target_audience
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                client_id,
                clean_name,
                (niche or "").strip(),
                datetime.utcnow(),
                notes.strip(),
                brand_voice_notes.strip(),
                target_audience.strip(),
            ],
        )
    finally:
        conn.close()

    for handle in accounts or []:
        add_client_account(client_id, handle)

    return client_id


def add_client_account(
    client_id: str,
    instagram_handle: str,
    category: str = "Competitor",
    max_items: int = 30,
):
    init_db()
    handle = normalize_handle(instagram_handle)
    if not handle:
        raise ValueError("Instagram handle is required")

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO client_accounts (
                client_id, instagram_handle, category, added_at, max_items
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (client_id, instagram_handle) DO UPDATE SET
                category = excluded.category,
                max_items = excluded.max_items
            """,
            [client_id, handle, category.strip() or "Competitor", datetime.utcnow(), int(max_items)],
        )
    finally:
        conn.close()


def remove_client_account(client_id: str, instagram_handle: str):
    init_db()
    handle = normalize_handle(instagram_handle)
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM client_accounts WHERE client_id = ? AND instagram_handle = ?",
            [client_id, handle],
        )
    finally:
        conn.close()


def get_client_accounts(client_id: str = DEFAULT_CLIENT_ID) -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT instagram_handle, category, added_at, max_items
            FROM client_accounts
            WHERE client_id = ?
            ORDER BY instagram_handle
            """,
            [client_id],
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "instagram_handle": r[0],
            "category": r[1],
            "added_at": r[2],
            "max_items": r[3],
        }
        for r in rows
    ]


def get_accounts_with_limits(client_id: str = DEFAULT_CLIENT_ID) -> list[tuple[str, int]]:
    accounts = get_client_accounts(client_id)
    return [
        (row["instagram_handle"], int(row.get("max_items") or 30))
        for row in accounts
    ]


def delete_client(client_id: str):
    init_db()
    if client_id == DEFAULT_CLIENT_ID:
        raise ValueError(f"{DEFAULT_CLIENT_NAME} is protected because it owns the seeded corpus")

    conn = get_connection()
    try:
        orphan_posts = [
            row[0]
            for row in conn.execute(
                """
                SELECT cp.post_id
                FROM client_posts cp
                WHERE cp.client_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM client_posts other
                      WHERE other.post_id = cp.post_id
                        AND other.client_id != ?
                  )
                """,
                [client_id, client_id],
            ).fetchall()
        ]

        conn.execute(
            """
            UPDATE posts
            SET client_id = (
                SELECT MIN(other.client_id)
                FROM client_posts other
                WHERE other.post_id = posts.post_id
                  AND other.client_id != ?
            )
            WHERE client_id = ?
              AND EXISTS (
                  SELECT 1 FROM client_posts other
                  WHERE other.post_id = posts.post_id
                    AND other.client_id != ?
              )
            """,
            [client_id, client_id, client_id],
        )

        for table in ["transcripts", "transcript_words", "message_units"]:
            conn.execute(
                f"""
                UPDATE {table}
                SET client_id = (
                    SELECT MIN(other.client_id)
                    FROM client_posts other
                    WHERE other.post_id = {table}.post_id
                      AND other.client_id != ?
                )
                WHERE client_id = ?
                  AND EXISTS (
                      SELECT 1 FROM client_posts other
                      WHERE other.post_id = {table}.post_id
                        AND other.client_id != ?
                  )
                """,
                [client_id, client_id, client_id],
            )

        for table in [
            "generated_content",
            "scrape_jobs",
            "profiles",
            "raw_scrapes",
            "reels",
            "comments",
            "tagged_users",
        ]:
            conn.execute(f"DELETE FROM {table} WHERE client_id = ?", [client_id])

        conn.execute("DELETE FROM client_accounts WHERE client_id = ?", [client_id])
        conn.execute("DELETE FROM client_posts WHERE client_id = ?", [client_id])

        if orphan_posts:
            placeholders = ", ".join(["?"] * len(orphan_posts))
            for table in ["message_units", "transcript_words", "transcripts", "posts"]:
                conn.execute(
                    f"DELETE FROM {table} WHERE post_id IN ({placeholders})",
                    orphan_posts,
                )

        conn.execute(
            """
            DELETE FROM creator_accounts
            WHERE account_id NOT IN (
                SELECT DISTINCT account_id FROM posts WHERE account_id IS NOT NULL
            )
              AND username NOT IN (
                SELECT DISTINCT instagram_handle FROM client_accounts
              )
            """
        )
        conn.execute("DELETE FROM clients WHERE client_id = ?", [client_id])
    finally:
        conn.close()


__all__ = [
    "DEFAULT_CLIENT_ID",
    "DEFAULT_CLIENT_NAME",
    "DEFAULT_CLIENT_NICHE",
    "add_client_account",
    "create_client",
    "delete_client",
    "get_accounts_with_limits",
    "get_client",
    "get_client_accounts",
    "list_clients",
    "normalize_handle",
    "remove_client_account",
]
