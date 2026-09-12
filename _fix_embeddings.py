#!/usr/bin/env python3
"""Fix embedding column dimension and re-embed all message_units."""
import sys
sys.path.insert(0, ".")

import duckdb
import tomllib

with open(".streamlit/secrets.toml", "rb") as f:
    s = tomllib.load(f)

voyage_key = s.get("VOYAGE_API_KEY", "")
if not voyage_key:
    print("ERROR: VOYAGE_API_KEY missing")
    sys.exit(1)

conn = duckdb.connect("reels.duckdb")

# Check current column type
col_info = conn.execute(
    "SELECT data_type FROM information_schema.columns "
    "WHERE table_name = 'message_units' AND column_name = 'embedding'"
).fetchone()
print(f"Current embedding column type: {col_info[0] if col_info else 'NOT FOUND'}")

# Fix if needed
if col_info and "1536" in str(col_info[0]):
    print("Migrating embedding column from FLOAT[1536] to FLOAT[512]...")
    # DuckDB may have index dependencies — recreate table with correct schema
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
    # Recreate indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_message_units_topic ON message_units(topic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_message_units_post_id ON message_units(post_id)")
    print("Migration done.")
elif col_info and "512" in str(col_info[0]):
    print("Column is already FLOAT[512] — good.")
else:
    print(f"Unknown column state: {col_info}")

# Get all units needing embeddings
rows = conn.execute(
    "SELECT unit_id, text FROM message_units WHERE embedding IS NULL"
).fetchall()
conn.close()
print(f"Units needing embeddings: {len(rows)}")

if not rows:
    print("Nothing to embed.")
    sys.exit(0)

from providers.embeddings import VoyageProvider
emb = VoyageProvider(api_key=voyage_key)

unit_ids = [r[0] for r in rows]
texts = [r[1] for r in rows]

print(f"Embedding {len(texts)} units with voyage-3-lite...")
try:
    embeddings = emb.embed_batch(texts)
    print(f"Got {len(embeddings)} embeddings, each dim={len(embeddings[0])}")
except Exception as e:
    print(f"Embedding FAILED: {e}")
    sys.exit(1)

conn = duckdb.connect("reels.duckdb")
for uid, emb_vec in zip(unit_ids, embeddings):
    conn.execute(
        "UPDATE message_units SET embedding = ?, embedded_at = CURRENT_TIMESTAMP WHERE unit_id = ?",
        [emb_vec, uid],
    )
conn.close()
print("Embeddings saved.")

# Final check
conn = duckdb.connect("reels.duckdb")
total = conn.execute("SELECT COUNT(*) FROM message_units").fetchone()[0]
embedded = conn.execute("SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL").fetchone()[0]
conn.close()
print(f"\nFinal: {total} message_units, {embedded} with embeddings")
