"""Per-client, versioned topic taxonomies (creative-director-ai #25).

Topics were a module constant hardcoded to one niche. A second persona in a
different niche had everything classify as "General", which collapsed topic
search, the content gap map, and Content Studio's grounding all at once.

**Versioning is the point, not bookkeeping.** Re-classifying is cheap —
transcripts are retained and extraction runs from stored text, so ~5,000 reels
costs roughly $1.50 and one queue run. That only stays true if prior
assignments survive the change. Overwrite them and you can no longer tell
whether a gap closed or a label simply moved, which is the longitudinal view
that makes a large corpus worth keeping.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

from core.db import DEFAULT_CLIENT_ID, get_connection, init_db

# Every taxonomy needs somewhere to put ideas that fit nothing else. The name
# is per-client, but the role is universal, so the id is fixed.
CATCH_ALL_TOPIC_ID = "general"
CATCH_ALL_DEFAULT_NAME = "General"


def slugify_topic(name: str) -> str:
    """Stable id from a display name. Renaming keeps the original id."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or f"topic_{uuid.uuid4().hex[:8]}"


def _active_taxonomy(conn, client_id: str) -> Optional[dict]:
    """Read the active taxonomy on an existing connection.

    Separate from get_active_taxonomy() because seeding runs inside init_db(),
    and anything it calls must not call init_db() again.
    """
    row = conn.execute(
        """
        SELECT taxonomy_id, client_id, version, created_at, notes
        FROM taxonomy_versions
        WHERE client_id = ? AND is_active = TRUE
        ORDER BY version DESC
        LIMIT 1
        """,
        [client_id],
    ).fetchone()
    if not row:
        return None
    return {
        "taxonomy_id": row[0],
        "client_id": row[1],
        "version": row[2],
        "created_at": row[3],
        "notes": row[4],
    }


def get_active_taxonomy(client_id: str) -> Optional[dict]:
    init_db()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT taxonomy_id, client_id, version, created_at, notes
            FROM taxonomy_versions
            WHERE client_id = ? AND is_active = TRUE
            ORDER BY version DESC
            LIMIT 1
            """,
            [client_id],
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "taxonomy_id": row[0],
        "client_id": row[1],
        "version": row[2],
        "created_at": row[3],
        "notes": row[4],
    }


def list_topics(client_id: str, taxonomy_id: Optional[str] = None) -> list[dict]:
    """Topics for a taxonomy version, defaulting to the client's active one."""
    if taxonomy_id is None:
        active = get_active_taxonomy(client_id)
        if not active:
            return []
        taxonomy_id = active["taxonomy_id"]

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT topic_id, name, description, sort_order
            FROM topics
            WHERE taxonomy_id = ?
            ORDER BY sort_order, name
            """,
            [taxonomy_id],
        ).fetchall()
    finally:
        conn.close()
    return [
        {"topic_id": r[0], "name": r[1], "description": r[2], "sort_order": r[3]}
        for r in rows
    ]


def topic_names(client_id: str) -> list[str]:
    """Display names for the active taxonomy — what a dropdown shows."""
    return [t["name"] for t in list_topics(client_id)]


def create_taxonomy_version(
    client_id: str,
    topics: list[dict],
    notes: str = "",
    activate: bool = True,
    _init: bool = True,
) -> str:
    """Add a new taxonomy version. Never mutates an existing one.

    `topics` entries take `name`, optional `description`, and optional
    `topic_id`. Carry the topic_id forward when renaming so the topic keeps its
    lineage across versions; omit it for a genuinely new topic.
    """
    if _init:
        init_db()
    if not topics:
        raise ValueError("A taxonomy needs at least one topic")

    conn = get_connection()
    try:
        prev = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM taxonomy_versions WHERE client_id = ?",
            [client_id],
        ).fetchone()[0]
        version = int(prev or 0) + 1
        taxonomy_id = f"tax_{uuid.uuid4().hex[:12]}"

        if activate:
            conn.execute(
                "UPDATE taxonomy_versions SET is_active = FALSE WHERE client_id = ?",
                [client_id],
            )

        conn.execute(
            """
            INSERT INTO taxonomy_versions
                (taxonomy_id, client_id, version, created_at, is_active, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [taxonomy_id, client_id, version, datetime.utcnow(), activate, notes or ""],
        )

        seen = set()
        for i, t in enumerate(topics):
            name = (t.get("name") or "").strip()
            if not name:
                continue
            topic_id = (t.get("topic_id") or slugify_topic(name)).strip()
            if topic_id in seen:
                continue  # a version cannot hold the same topic twice
            seen.add(topic_id)
            conn.execute(
                """
                INSERT INTO topics
                    (taxonomy_id, topic_id, client_id, name, description, sort_order)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [taxonomy_id, topic_id, client_id, name, t.get("description") or "", i],
            )

        if CATCH_ALL_TOPIC_ID not in seen:
            conn.execute(
                """
                INSERT INTO topics
                    (taxonomy_id, topic_id, client_id, name, description, sort_order)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    taxonomy_id,
                    CATCH_ALL_TOPIC_ID,
                    client_id,
                    CATCH_ALL_DEFAULT_NAME,
                    "Content that does not fit a specific topic",
                    len(topics) + 1,
                ],
            )
    finally:
        conn.close()
    return taxonomy_id


def topics_for_prompt(client_id: str) -> str:
    """Formatted topic list for the extraction prompt."""
    topics = list_topics(client_id)
    if not topics:
        return f"  - {CATCH_ALL_DEFAULT_NAME}: no topics defined for this client yet"
    return "\n".join(
        f"  - {t['name']}: {t['description'] or t['name']}" for t in topics
    )


def validate_topic(client_id: str, value: str) -> tuple[str, str]:
    """Resolve a model-produced topic name to (topic_id, name).

    Falls back to the client's own catch-all rather than a hardcoded "General",
    because a persona may have named it something else — or not have one.
    """
    wanted = (value or "").strip().lower()
    topics = list_topics(client_id)
    for t in topics:
        if t["name"].lower() == wanted or t["topic_id"].lower() == wanted:
            return t["topic_id"], t["name"]
    for t in topics:
        if t["topic_id"] == CATCH_ALL_TOPIC_ID:
            return t["topic_id"], t["name"]
    return CATCH_ALL_TOPIC_ID, CATCH_ALL_DEFAULT_NAME


def ensure_seed_taxonomy(client_id: str = DEFAULT_CLIENT_ID) -> Optional[str]:
    """Give the demo client its original Jobs-AU topics as version 1.

    Idempotent, and only seeds the demo client — a new persona starts empty
    rather than inheriting somebody else's vocabulary.
    """
    if client_id != DEFAULT_CLIENT_ID:
        return None

    conn = get_connection()
    try:
        if _active_taxonomy(conn, client_id):
            return None  # already seeded
    finally:
        conn.close()

    from analysis.taxonomy import (
        SEED_TOPIC_DESCRIPTIONS_JOBS_AU,
        SEED_TOPICS_JOBS_AU,
    )

    topics = [
        {
            "topic_id": slugify_topic(name) if name != "General" else CATCH_ALL_TOPIC_ID,
            "name": name,
            "description": SEED_TOPIC_DESCRIPTIONS_JOBS_AU.get(name, ""),
        }
        for name in SEED_TOPICS_JOBS_AU
    ]
    return create_taxonomy_version(
        client_id,
        topics,
        notes="Seeded from the original Jobs-AU taxonomy",
        _init=False,  # we are already inside init_db()
    )
