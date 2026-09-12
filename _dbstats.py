import sys; sys.path.insert(0, ".")
import duckdb

conn = duckdb.connect("reels.duckdb")

accounts = conn.execute("SELECT account_id, COUNT(*) as posts FROM posts GROUP BY account_id ORDER BY posts DESC").fetchall()
print("=== Accounts scraped ===")
for a in accounts:
    print(f"  {a[0]}: {a[1]} posts")

trans = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
print(f"\nTranscripts: {trans}")

print("\n=== Knowledge units by topic ===")
topics = conn.execute("SELECT topic, COUNT(*) as n FROM message_units GROUP BY topic ORDER BY n DESC").fetchall()
for t in topics:
    print(f"  {t[0]}: {t[1]}")

print("\n=== By content_type ===")
cts = conn.execute("SELECT content_type, COUNT(*) as n FROM message_units GROUP BY content_type ORDER BY n DESC").fetchall()
for c in cts:
    print(f"  {c[0]}: {c[1]}")

print("\n=== Sample message units ===")
samples = conn.execute("SELECT topic, content_type, text FROM message_units LIMIT 10").fetchall()
for s in samples:
    print(f"  [{s[0]}/{s[1]}] {s[2][:90]}")

conn.close()
