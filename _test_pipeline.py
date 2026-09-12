"""
Test the full pipeline path end-to-end with 1 post.
Runs all 3 provider imports and transcription queue call to surface any errors.
"""
import os, sys, traceback

# Load secrets from .streamlit/secrets.toml manually
try:
    import tomllib
except ImportError:
    import tomli as tomllib

secrets_path = ".streamlit/secrets.toml"
with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)

os.environ["ASSEMBLYAI_API_KEY"] = secrets.get("ASSEMBLYAI_API_KEY", "")
os.environ["CEREBRAS_API_KEY"] = secrets.get("CEREBRAS_API_KEY", "")
os.environ["VOYAGE_API_KEY"] = secrets.get("VOYAGE_API_KEY", "")

assemblyai_key = secrets.get("ASSEMBLYAI_API_KEY", "")
cerebras_key = secrets.get("CEREBRAS_API_KEY", "")
voyage_key = secrets.get("VOYAGE_API_KEY", "")

print(f"AssemblyAI key present: {bool(assemblyai_key)}")
print(f"Cerebras key present: {bool(cerebras_key)}")
print(f"Voyage key present: {bool(voyage_key)}")

# Monkey-patch st.secrets
import types
import streamlit as st

# Create a fake secrets object
class FakeSecrets(dict):
    def get(self, key, default=None):
        return super().get(key, default)
    def __getitem__(self, key):
        return super().__getitem__(key)

st.secrets = FakeSecrets(secrets)

print("\n--- Testing provider imports ---")
try:
    from providers.stt import AssemblyAIProvider
    stt = AssemblyAIProvider(api_key=assemblyai_key)
    print("✅ AssemblyAIProvider OK")
except Exception as e:
    print(f"❌ AssemblyAIProvider: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from providers.factory import get_llm, get_embeddings
    llm = get_llm()
    print(f"✅ LLM provider: {type(llm).__name__}")
except Exception as e:
    print(f"❌ LLM provider: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    emb = get_embeddings()
    print(f"✅ Embeddings provider: {type(emb).__name__}")
except Exception as e:
    print(f"❌ Embeddings provider: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n--- Testing pipeline query ---")
try:
    from core.db import get_connection
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.post_id,
               COALESCE(
                   NULLIF(p.local_audio_path, ''),
                   NULLIF(p.audio_url, ''),
                   p.video_url
               ) AS audio_source
        FROM posts p
        LEFT JOIN transcripts t ON p.post_id = t.post_id
        WHERE (
              (p.local_audio_path IS NOT NULL AND p.local_audio_path != '')
           OR (p.audio_url       IS NOT NULL AND p.audio_url       != '')
           OR (p.video_url       IS NOT NULL AND p.video_url       != '')
        )
          AND t.post_id IS NULL
        LIMIT 1
    """).fetchall()
    conn.close()
    print(f"✅ Query found {len(rows)} row(s)")
    if rows:
        print(f"   post_id={rows[0][0]}, audio_source preview={rows[0][1][:60]}...")
except Exception as e:
    print(f"❌ Pipeline query: {e}")
    traceback.print_exc()
    sys.exit(1)

if not rows:
    print("No rows to process — exiting.")
    sys.exit(0)

print("\n--- Testing STT on 1 post ---")
post_id, audio_source = rows[0]
print(f"Transcribing: {post_id}")
print(f"Source URL: {audio_source[:80]}...")

try:
    result = stt.transcribe(audio_source, post_id)
    print(f"✅ Transcription done!")
    print(f"   Duration: {result.duration_sec}s")
    print(f"   Words: {result.word_count}")
    print(f"   Cost: ${result.cost_usd:.4f}")
    print(f"   Preview: {result.transcript[:200]}")
except Exception as e:
    print(f"❌ STT failed: {e}")
    traceback.print_exc()
