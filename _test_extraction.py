#!/usr/bin/env python3
"""Test Cerebras extraction and Voyage embeddings, then fill missing message_units."""
import sys, os
sys.path.insert(0, ".")

import tomllib
with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

cerebras_key = secrets.get("CEREBRAS_API_KEY", "")
voyage_key = secrets.get("VOYAGE_API_KEY", "")
print(f"Cerebras key: {'SET' if cerebras_key else 'MISSING'}")
print(f"Voyage key:   {'SET' if voyage_key else 'MISSING'}")

if not cerebras_key:
    print("ERROR: CEREBRAS_API_KEY not found in secrets.toml — extraction will always fail!")
    sys.exit(1)

# Test extraction
from providers.llm import CerebrasProvider
prov = CerebrasProvider(api_key=cerebras_key)
print(f"Cerebras model: {prov._model}")

test_transcript = (
    "First tip: always tailor your resume with keywords from the job description. "
    "Second tip: quantify achievements with numbers. "
    "Third tip: use the STAR method in behavioural interviews."
)
print(f"\nTesting extraction on sample transcript ({len(test_transcript)} chars)...")
try:
    units, cost = prov.extract(test_transcript, "test-post-id")
    print(f"OK — {len(units)} units, cost ${cost:.6f}")
    for u in units[:3]:
        print(f"  [{u.topic}] {u.text[:80]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nExtraction is working. Now filling missing message_units from existing transcripts...")

import duckdb
from analysis.extraction import save_message_units

conn = duckdb.connect("reels.duckdb")
rows = conn.execute("""
    SELECT t.post_id, t.transcript
    FROM transcripts t
    LEFT JOIN message_units m ON t.post_id = m.post_id
    WHERE m.post_id IS NULL
      AND LENGTH(t.transcript) > 30
""").fetchall()
conn.close()

print(f"Found {len(rows)} transcripts without message_units")

all_units = []
for i, (post_id, transcript) in enumerate(rows, 1):
    print(f"  [{i}/{len(rows)}] Extracting {post_id[:12]}... ", end="", flush=True)
    try:
        units, cost = prov.extract(transcript, post_id)
        print(f"{len(units)} units  (${cost:.4f})")
        all_units.extend(units)
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nSaving {len(all_units)} total units to DB...")
save_message_units(all_units)

# Now embed
if voyage_key and all_units:
    from providers.embeddings import VoyageProvider
    emb_prov = VoyageProvider(api_key=voyage_key)
    print(f"Embedding {len(all_units)} units with Voyage...")
    texts = [u.text for u in all_units]
    try:
        embeddings = emb_prov.embed_batch(texts)
        conn = duckdb.connect("reels.duckdb")
        for unit, emb in zip(all_units, embeddings):
            conn.execute(
                "UPDATE message_units SET embedding = ?, embedded_at = CURRENT_TIMESTAMP WHERE unit_id = ?",
                [emb, unit.unit_id],
            )
        conn.close()
        print(f"Embeddings saved for {len(all_units)} units")
    except Exception as e:
        print(f"Embedding FAILED: {e}")
else:
    if not voyage_key:
        print("VOYAGE_API_KEY missing — skipping embeddings")

# Final check
conn = duckdb.connect("reels.duckdb")
units_count = conn.execute("SELECT COUNT(*) FROM message_units").fetchone()[0]
embedded_count = conn.execute("SELECT COUNT(*) FROM message_units WHERE embedding IS NOT NULL").fetchone()[0]
conn.close()
print(f"\nFinal DB state: {units_count} message_units, {embedded_count} with embeddings")
