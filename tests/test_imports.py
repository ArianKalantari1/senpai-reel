from core.db import init_db, get_connection
init_db()
print('init_db OK')

from analysis.taxonomy import CONTENT_TYPES
from core.taxonomy import topic_names
from core.db import DEFAULT_CLIENT_ID
print('taxonomy OK —', len(topic_names(DEFAULT_CLIENT_ID)), 'topics for the demo client')

from analysis.extraction import extract_message_units
from processing.extraction_queue import get_extraction_stats
print('extraction stats:', get_extraction_stats())

from analysis.embeddings import embed_batch
print('embeddings OK')

from analysis.search import keyword_search
print('search OK')

from analysis.analytics import get_creator_leaderboard
print('analytics OK')

from analysis.prompts import format_reference_context
from analysis.content_gen import generate_caption
print('content_gen OK')

conn = get_connection()
tables = sorted([r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()])
print('tables:', tables)
conn.close()
print('ALL OK')
