# Multi-Client Data Model

Senpai Reel uses one DuckDB file with explicit client scoping.

## Decision

The Phase 0 migration keeps `reels.duckdb` as a single analytical store and adds:

- `clients`: client profile, niche, notes, brand voice, and target audience
- `client_accounts`: competitor Instagram handles per client
- `client_posts`: many-to-many visibility links between clients and posts
- `client_id` columns on scrape jobs, posts, transcripts, transcript words, message units, generated content, and legacy scrape tables

This was chosen over one DuckDB file per client because:

- existing scraped/transcribed/extracted rows were paid for and must be preserved
- shared competitors are normal, and a single store lets clients reuse the same post/transcript/embedding without paying twice
- future cross-client learning remains possible without a later data merge
- Streamlit pages can still enforce client isolation by filtering through `client_posts`

## Seed Migration

`init_db()` is idempotent. On first run against an older database it:

1. Creates the client tables.
2. Adds missing `client_id` columns.
3. Creates the protected `jobs_au_demo` client named `Jobs AU (demo)`.
4. Seeds the original Jobs-in-Australia handles into `client_accounts`.
5. Assigns existing rows to `jobs_au_demo`.
6. Links all existing posts into `client_posts`.

No schema reset is required.

## Scoping Rule

UI pages read the active client from `core.client_context.render_client_selector()`.
Queries that surface posts, transcripts, message units, analytics, or generated content must filter by the active `client_id`.

For post-derived artifacts, prefer `client_posts` visibility checks over direct `posts.client_id` checks. `posts.client_id`, `transcripts.client_id`, and `message_units.client_id` record the client that first created the artifact, while `client_posts` controls which clients can see it.

## Deletion

`core.clients.delete_client()` deletes non-demo client-owned rows and their generated content. Posts and expensive derived artifacts are deleted only when no other client is linked to the post. Shared posts are retained for the remaining clients.

The seeded demo client is protected because it owns the original corpus.
