from datetime import datetime

import duckdb
import pytest


@pytest.fixture
def client_db(tmp_path):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "clients.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def test_create_two_clients_with_different_accounts(client_db):
    from core.clients import add_client_account, create_client, get_client_accounts, list_clients

    cafe_id = create_client("Cafe Pilot", "Hospitality")
    clinic_id = create_client("Clinic Pilot", "Healthcare")

    add_client_account(cafe_id, "@melbourne.cafe", "Local cafe", 20)
    add_client_account(clinic_id, "https://www.instagram.com/health.creator/", "Healthcare", 15)

    clients = {row["client_id"]: row for row in list_clients()}
    assert cafe_id in clients
    assert clinic_id in clients
    assert clients[cafe_id]["niche"] == "Hospitality"
    assert clients[clinic_id]["niche"] == "Healthcare"
    assert get_client_accounts(cafe_id)[0]["instagram_handle"] == "melbourne.cafe"
    assert get_client_accounts(clinic_id)[0]["instagram_handle"] == "health.creator"


def test_delete_client_removes_private_data_and_preserves_shared_post(client_db):
    from core.clients import create_client, delete_client

    alpha_id = create_client("Alpha", "Fitness")
    beta_id = create_client("Beta", "Education")
    now = datetime.utcnow()

    conn = duckdb.connect(client_db.DB_PATH)
    conn.execute(
        "INSERT INTO creator_accounts VALUES ('acc1','creator',NULL,NULL,NULL,NULL,NULL,FALSE,NULL,NULL,NULL,?,?)",
        [now, now],
    )
    for post_id in ["shared_post", "private_post"]:
        conn.execute(
            """
            INSERT INTO posts (
                post_id, client_id, account_id, engagement_rate, download_status,
                scraped_at, hashtags, mentions
            )
            VALUES (?, ?, 'acc1', 3.0, 'done', ?, [], [])
            """,
            [post_id, alpha_id, now],
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [alpha_id, post_id, now],
        )
        conn.execute(
            """
            INSERT INTO transcripts (
                post_id, client_id, provider, model, transcript, language,
                confidence, duration_sec, word_count, transcribed_at, cost_usd
            )
            VALUES (?, ?, 'deepgram', 'nova-2', 'hello', 'en', 0.9, 5, 1, ?, 0.001)
            """,
            [post_id, alpha_id, now],
        )
        conn.execute(
            """
            INSERT INTO message_units (
                unit_id, client_id, post_id, text, claim, topic, content_type,
                confidence, extracted_at, model
            )
            VALUES (?, ?, ?, 'text', 'claim', 'General', 'tip', 0.9, ?, 'gpt-4o-mini')
            """,
            [f"unit_{post_id}", alpha_id, post_id, now],
        )

    conn.execute(
        "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, 'shared_post', ?)",
        [beta_id, now],
    )
    conn.execute(
        """
        INSERT INTO generated_content (
            gen_id, client_id, created_at, topic, content_type, output_text,
            model, source_units, tokens_used, cost_usd
        )
        VALUES ('gen_alpha', ?, ?, 'General', 'caption', 'copy', 'gpt-4o-mini', [], 10, 0.001)
        """,
        [alpha_id, now],
    )
    conn.close()

    delete_client(alpha_id)

    conn = duckdb.connect(client_db.DB_PATH)
    alpha_exists = conn.execute(
        "SELECT COUNT(*) FROM clients WHERE client_id = ?",
        [alpha_id],
    ).fetchone()[0]
    shared_exists = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE post_id = 'shared_post'"
    ).fetchone()[0]
    private_exists = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE post_id = 'private_post'"
    ).fetchone()[0]
    beta_link = conn.execute(
        """
        SELECT COUNT(*) FROM client_posts
        WHERE client_id = ? AND post_id = 'shared_post'
        """,
        [beta_id],
    ).fetchone()[0]
    generated_exists = conn.execute(
        "SELECT COUNT(*) FROM generated_content WHERE client_id = ?",
        [alpha_id],
    ).fetchone()[0]
    conn.close()

    assert alpha_exists == 0
    assert shared_exists == 1
    assert private_exists == 0
    assert beta_link == 1
    assert generated_exists == 0
