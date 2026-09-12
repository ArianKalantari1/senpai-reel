"""Clear all data tables while keeping schema intact."""
import sys
sys.path.insert(0, ".")
import duckdb

conn = duckdb.connect("reels.duckdb")

# List actual tables in the DB
tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
print("Tables found:", tables)

# Data tables to truncate (order matters for any FK constraints)
to_clear = [
    "message_units",
    "transcript_words",
    "transcripts",
    "generated_content",
    "raw_scrapes",
    "profiles",
    "posts",
    "reels",
    "comments",
    "tagged_users",
]

for t in to_clear:
    if t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        conn.execute(f"DELETE FROM {t}")
        print(f"  Cleared {t}: {count} rows deleted")
    else:
        print(f"  Skipped {t} (not in DB)")

conn.close()
print("\nDone. Schema intact, all data cleared.")
