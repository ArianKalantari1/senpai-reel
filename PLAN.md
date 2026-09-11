# Senpai Reel — Attacking Plan
> **Domain:** Jobs in Australia (HR, recruitment, resumes, interviews, ATS systems)
> **Goal:** Scrape competitor Instagram reels → extract rich content context → store in structured DB → enable semantic search + content generation
> **UI:** Streamlit only (no FastAPI)
> **DB:** DuckDB (primary), DuckDB cosine similarity for vector search (no external vector DB)
> **Transcription:** Deepgram Nova-2 (primary, ~$0.006/min, $200 free credits ≈ 34k min free)
> **LLM:** GPT-4o-mini (extraction + generation), text-embedding-3-small (embeddings)

---

## How to Use This Document
- Every phase has **Acceptance Criteria** — a phase is DONE only when ALL criteria are met
- Every phase ends with a **git commit + push** (see [Git Workflow](#git-workflow) at bottom)
- Copilot MUST read this file at the start of every session to understand current state
- Update the `[ ]` checkboxes as tasks complete, then commit the updated PLAN.md

---

## Current Status

| Phase | Status | Branch |
|-------|--------|--------|
| Phase 0 — Foundation | `[x] Done` | `phase-0-foundation` |
| Phase 1 — Scraping Engine | `[x] Done` | `phase-1-scraping` |
| Phase 2 — Media Acquisition | `[x] Done` | `phase-2-media` |
| Phase 3 — Transcription | `[x] Done` | `phase-3-transcription` |
| Phase 4 — Knowledge Extraction | `[x] Done` | `phase-4-extraction` |
| Phase 5 — Vector Search | `[x] Done` | `phase-5-search` |
| Phase 6 — Analytics | `[x] Done` | `phase-6-analytics` |
| Phase 7 — Content Studio | `[x] Done` | `phase-7-content` |
| Phase 8 — QA & Testing | `[x] Done` | `phase-8-qa` |
| Phase 9 — Optimisation | `[x] Done` | `phase-9-optimisation` |

---

## Phase 0 — Foundation & Cleanup
**Goal:** Get the codebase clean, consistent, and git-tracked before touching anything new.
**Est. effort:** 0.5–1 day

### Tasks
- [ ] Merge `pipeline/transform.py` into `core/db.py` — they are identical duplicates. Delete transform.py after merge.
- [ ] Delete `pipeline/etl_basic.py` — empty file, adds confusion
- [ ] Add deduplication to `raw_scrapes` insert — check `shortcode` before inserting (avoid double-scraping)
- [ ] Make structured processing automatic — remove opt-in checkbox from `app.py` or default it to `True`
- [ ] Archive graph pages — move `pages/4_🕸️_Graph_Network.py` and `pages/5_🔍_Graph_Query_Engine.py` to `archive/` folder (keep code, remove from sidebar)
- [ ] Create `secrets.toml.example` so new devs know what keys to add
- [ ] Add `APIFY_TOKEN`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY` to secrets.toml.example
- [ ] Verify Streamlit app boots cleanly with `streamlit run app.py`

### Acceptance Criteria
- [ ] `python -c "from core.db import init_db; init_db()"` runs without error
- [ ] No duplicate function definitions across the codebase
- [ ] Scraping a profile twice does NOT create duplicate rows in raw_scrapes
- [ ] App boots and all remaining pages render

### Files touched
- `core/db.py` — deduplication logic
- `app.py` — remove opt-in checkbox
- `pipeline/transform.py` — DELETE
- `pipeline/etl_basic.py` — DELETE
- `pages/4_*.py`, `pages/5_*.py` — MOVE to `archive/`

---

## Phase 1 — Scraping Engine (HIGHEST PRIORITY)
**Goal:** A rock-solid scraping pipeline that collects all reels from a list of competitor accounts and stores them in a canonical structured database. This is the foundation everything else depends on.
**Est. effort:** 2–3 days

### Why this is phase 1
You cannot transcribe, analyse, or generate content without data. The richer and more complete the scrape, the better every downstream feature becomes. Time spent here is the highest-ROI investment in the project.

### New DB Tables Required
```sql
-- Canonical creator accounts
CREATE TABLE creator_accounts (
    account_id     TEXT PRIMARY KEY,    -- Instagram user ID
    username       TEXT UNIQUE NOT NULL,
    full_name      TEXT,
    bio            TEXT,
    followers      INTEGER,
    following      INTEGER,
    post_count     INTEGER,
    is_verified    BOOLEAN,
    profile_pic_url TEXT,
    external_url   TEXT,
    category       TEXT,               -- e.g. "HR/Recruitment"
    first_seen_at  TIMESTAMP,
    last_scraped_at TIMESTAMP
);

-- Canonical posts table (one row per reel)
CREATE TABLE posts (
    post_id        TEXT PRIMARY KEY,   -- Instagram shortcode
    account_id     TEXT REFERENCES creator_accounts(account_id),
    caption        TEXT,
    caption_clean  TEXT,               -- stripped of hashtags/mentions
    hashtags       TEXT[],             -- array of hashtags
    mentions       TEXT[],             -- array of @mentions
    likes          INTEGER,
    views          INTEGER,
    comments_count INTEGER,
    duration_sec   DOUBLE,
    posted_at      TIMESTAMP,
    scraped_at     TIMESTAMP,
    video_url      TEXT,               -- CDN URL (expires ~48h)
    audio_url      TEXT,
    thumbnail_url  TEXT,
    is_pinned      BOOLEAN,
    is_sponsored   BOOLEAN,
    engagement_rate DOUBLE,            -- (likes + comments) / views * 100
    raw_json       JSON               -- full Apify payload for reference
);

-- Track scrape jobs (audit log)
CREATE TABLE scrape_jobs (
    job_id         TEXT PRIMARY KEY,
    username       TEXT,
    started_at     TIMESTAMP,
    finished_at    TIMESTAMP,
    reels_found    INTEGER,
    reels_new      INTEGER,
    status         TEXT,              -- running | done | failed
    error_msg      TEXT
);
```

### Tasks
- [ ] Create `core/models.py` — dataclass definitions for `CreatorAccount`, `Post`, `ScrapeJob`
- [ ] Update `core/db.py` — add new table schemas above, keep `raw_scrapes` as backup store
- [ ] Build `collection/scraper.py` — clean rewrite of scraping logic:
  - Apify actor call with configurable `maxItems` and `resultsType`
  - Deduplication: check `posts` table before inserting (keyed on shortcode)
  - Upsert pattern: update engagement stats if post already exists
  - Rate limiting / retry with exponential backoff
  - ScrapeJob audit log (start/finish/error)
- [ ] Build `collection/account_list.py` — curated list of Jobs-AU competitor accounts to track
- [ ] Update `app.py` scraper UI:
  - Multi-account batch input (textarea with one handle per line)
  - Scrape job progress bar (poll Apify run status)
  - Show "X new reels found" after each run
  - Display last scraped date per account
- [ ] Add `pages/Data_Viewer.py` — switch from `raw_scrapes` to `posts` table

### Acceptance Criteria
- [ ] Batch scrape 5 accounts in a single Streamlit run without error
- [ ] Re-scraping the same account does NOT create duplicate posts
- [ ] Engagement rate is auto-calculated on insert
- [ ] ScrapeJob row is written with correct start/finish time and reel count
- [ ] Data Viewer displays data from `posts` table with all key columns

### Files created/modified
- `core/models.py` — NEW
- `core/db.py` — updated schemas
- `collection/scraper.py` — NEW (replaces logic spread across app.py)
- `collection/account_list.py` — NEW
- `app.py` — updated UI

---

## Phase 2 — Media Acquisition
**Goal:** Download video files immediately after scraping (CDN URLs expire ~48h), extract audio, and track download status.
**Est. effort:** 1–2 days

### Why this matters
Deepgram needs audio. The video CDN URLs in Apify responses expire within ~48 hours. Without immediate download, transcription becomes impossible for anything scraped > 2 days ago.

### New DB Columns/Tables
```sql
-- Add to posts table
ALTER TABLE posts ADD COLUMN local_video_path TEXT;  -- e.g. downloads/shortcode.mp4
ALTER TABLE posts ADD COLUMN local_audio_path TEXT;  -- e.g. audio_extracts/shortcode.mp3
ALTER TABLE posts ADD COLUMN download_status TEXT;   -- pending | downloading | done | failed
ALTER TABLE posts ADD COLUMN downloaded_at TIMESTAMP;
ALTER TABLE posts ADD COLUMN file_size_mb DOUBLE;
```

### Tasks
- [ ] Create `processing/download.py`:
  - `download_video(post_id, video_url, output_dir)` using `yt-dlp` or direct HTTP download
  - Fallback: try `video_url` first, then `audio_url` from Apify payload
  - Set `download_status` = `downloading` before starting, `done` after, `failed` on error
  - Skip if `local_video_path` already exists and file size > 0
- [ ] Create `processing/audio.py`:
  - `extract_audio(video_path)` using `ffmpeg` subprocess call
  - Output: 16kHz mono WAV (required by Deepgram)
  - Update `local_audio_path` in DB after extraction
- [ ] Integrate into scraping flow: after each post is saved to DB, queue download
- [ ] Add download queue to Streamlit UI (show pending/done counts)
- [ ] Add `downloads/` and `audio_extracts/` to `.gitignore` (already done in Phase 0)

### Acceptance Criteria
- [ ] After scraping, all new posts show `download_status = pending`
- [ ] Running the download job converts `pending` → `done` for reachable URLs
- [ ] Audio file at `local_audio_path` plays correctly (16kHz WAV)
- [ ] Re-running download job skips already-downloaded posts
- [ ] Expired-URL posts are marked `failed` with descriptive error

### Files created/modified
- `processing/__init__.py` — NEW
- `processing/download.py` — NEW
- `processing/audio.py` — NEW
- `core/db.py` — new columns
- `app.py` — download queue UI

---

## Phase 3 — Transcription Pipeline
**Goal:** Transcribe all downloaded audio files using Deepgram Nova-2 and store full transcripts with word-level timestamps in DuckDB.
**Est. effort:** 2–3 days

### Stack Decision
| Provider | Cost/min | Free Tier | Accuracy | Word Timestamps |
|----------|----------|-----------|----------|-----------------|
| **Deepgram Nova-2** ✅ | $0.0058 | $200 (~34k min) | Excellent (casual speech) | Yes (word-level) |
| OpenAI Whisper-1 | $0.020 | None | Good | No |
| AssemblyAI | $0.0035 | $50 (~240h) | Good | Yes |

Use Deepgram Nova-2 as primary. Implement a `BaseTranscriber` protocol so switching providers is a 1-line change.

### New DB Tables
```sql
CREATE TABLE transcripts (
    post_id       TEXT PRIMARY KEY REFERENCES posts(post_id),
    provider      TEXT,             -- deepgram | whisper | assemblyai
    model         TEXT,             -- nova-2 | whisper-1
    transcript    TEXT,             -- full transcript text
    language      TEXT,
    confidence    DOUBLE,           -- average confidence score 0-1
    duration_sec  DOUBLE,
    word_count    INTEGER,
    transcribed_at TIMESTAMP,
    cost_usd      DOUBLE,           -- actual cost for this transcription
    raw_response  JSON             -- full provider response
);

CREATE TABLE transcript_words (
    post_id       TEXT REFERENCES posts(post_id),
    word_index    INTEGER,
    word          TEXT,
    start_sec     DOUBLE,
    end_sec       DOUBLE,
    confidence    DOUBLE,
    PRIMARY KEY   (post_id, word_index)
);
```

### Tasks
- [ ] Add `DEEPGRAM_API_KEY` to `.streamlit/secrets.toml`
- [ ] Create `processing/transcribe.py`:
  - `BaseTranscriber` — abstract protocol with `transcribe(audio_path) -> TranscriptResult`
  - `DeepgramTranscriber` — calls Nova-2, parses word-level JSON, stores cost
  - `WhisperTranscriber` — OpenAI Whisper-1 fallback
  - `TranscriptResult` dataclass: text, words, confidence, cost_usd, raw_response
- [ ] Create `processing/transcription_queue.py`:
  - Reads posts with `download_status = done` and no transcript yet
  - Batches 10 at a time (configurable)
  - Updates DB after each transcription (not at end of batch — so partial runs succeed)
  - Tracks running cost total
- [ ] Add `pages/Corpus_Explorer.py` — new Streamlit page:
  - List all transcribed posts with search (keyword)
  - Full transcript view per post (side panel)
  - Transcription cost tracker (total spent, remaining free credits estimate)
  - Manual "Transcribe Now" button for a single post

### Acceptance Criteria
- [ ] 10 audio files transcribed end-to-end without error
- [ ] `transcripts` table populated with full text
- [ ] `transcript_words` table populated with word timestamps
- [ ] Corpus Explorer page shows searchable list of transcripts
- [ ] Running cost is visible and correct in UI
- [ ] Re-running queue skips already-transcribed posts

### Files created/modified
- `processing/transcribe.py` — NEW
- `processing/transcription_queue.py` — NEW
- `pages/Corpus_Explorer.py` — NEW
- `core/db.py` — new tables

---

## Phase 4 — Knowledge Extraction
**Goal:** Extract structured "message units" from transcripts — key claims, advice, tips, hooks — tagged with the Jobs-AU domain taxonomy.
**Est. effort:** 2–3 days

### What is a "Message Unit"?
A message unit is a single coherent idea extracted from a reel transcript. Example:

> **Transcript:** "Most ATS systems will reject your resume if you use tables or columns. Keep it to a single column with plain text."
>
> **Message unit:** `{claim: "ATS systems reject resumes with tables/columns", advice: "use single column plain text", topic: "ATS", subtopic: "resume_formatting", content_type: "warning"}`

### Jobs-AU Domain Taxonomy
```
Topics:
  ATS            — Applicant Tracking Systems
  Resume         — Resume writing, formatting, keywords
  CoverLetter    — Cover letter strategy
  Interview      — Interview prep, questions, STAR method
  LinkedIn       — LinkedIn profile, networking, InMail
  JobSearch      — Job search strategy, boards, cold outreach
  Salary         — Salary negotiation, benchmarks
  CareerChange   — Pivoting industries, upskilling
  Recruiter      — Working with recruiters, agency vs in-house
  Visa           — Working visa, sponsorship, 482, 189, 190

Content Types:
  tip            — Actionable advice
  warning        — Common mistake to avoid
  stat           — Statistic or data point
  myth           — Debunking a misconception
  story          — Personal experience / case study
  hook           — Opening hook (first 3 seconds)
  cta            — Call to action
```

### New DB Tables
```sql
CREATE TABLE message_units (
    unit_id       TEXT PRIMARY KEY,   -- uuid
    post_id       TEXT REFERENCES posts(post_id),
    text          TEXT,               -- the extracted sentence/claim
    claim         TEXT,               -- core claim (GPT extracted)
    advice        TEXT,               -- actionable advice (if any)
    topic         TEXT,               -- from taxonomy above
    subtopic      TEXT,
    content_type  TEXT,               -- tip | warning | stat | myth | story | hook | cta
    confidence    DOUBLE,             -- extraction confidence 0-1
    source_start  DOUBLE,             -- timestamp in video where this appears
    source_end    DOUBLE,
    extracted_at  TIMESTAMP,
    model         TEXT                -- gpt-4o-mini | manual
);
```

### Tasks
- [ ] Create `analysis/extraction.py`:
  - `extract_message_units(transcript_text, post_id)` — GPT-4o-mini call with structured JSON output
  - System prompt: Jobs-AU domain expert, extract only concrete tips/claims
  - Parse structured JSON response into `MessageUnit` dataclass
  - Store in `message_units` table
- [ ] Create `analysis/taxonomy.py` — topic + content_type constants, validation helpers
- [ ] Create `processing/extraction_queue.py`:
  - Reads `transcripts` with no `message_units` yet
  - Runs extraction in batches
  - Stores cost per post (GPT-4o-mini is cheap: ~$0.002–0.005/reel)
- [ ] Add taxonomy filter to Corpus Explorer (Phase 3 page)

### Acceptance Criteria
- [ ] 20 transcripts processed → message_units populated
- [ ] Each unit has topic, content_type, confidence
- [ ] Topic distribution visible in UI (bar chart)
- [ ] Corpus Explorer filterable by topic or content_type

### Files created/modified
- `analysis/extraction.py` — NEW
- `analysis/taxonomy.py` — NEW
- `processing/extraction_queue.py` — NEW
- `pages/Corpus_Explorer.py` — updated with filters

---

## Phase 5 — Vector Search & Retrieval (CONSULTANT PRIORITY #1)
**Goal:** Embed all message units and enable semantic "search by meaning" — the core intelligence layer of the product.
**Est. effort:** 2–3 days

### Architecture
```
message_units.text  →  text-embedding-3-small (1536 dims)  →  FLOAT[1536] column in DuckDB
                                                                      ↓
Query: "how to get past ATS"  →  embed query  →  cosine similarity  →  top-K results
```

No external vector DB. DuckDB handles cosine similarity natively on FLOAT[] columns. Simple, zero-ops, free.

### New DB Changes
```sql
-- Add to message_units table
ALTER TABLE message_units ADD COLUMN embedding FLOAT[1536];
ALTER TABLE message_units ADD COLUMN embedded_at TIMESTAMP;

-- Materialised view for search convenience
CREATE TABLE search_index (
    unit_id       TEXT PRIMARY KEY,
    post_id       TEXT,
    username      TEXT,
    posted_at     TIMESTAMP,
    topic         TEXT,
    content_type  TEXT,
    text          TEXT,
    embedding     FLOAT[1536]
);
```

### Cosine Similarity Query Pattern
```sql
-- Given a query embedding as ?query_vec (FLOAT[1536]):
SELECT
    unit_id, post_id, username, topic, content_type, text,
    list_cosine_similarity(embedding, ?query_vec) AS score
FROM search_index
WHERE topic = ?topic_filter  -- optional
ORDER BY score DESC
LIMIT 20;
```

### Tasks
- [ ] Create `analysis/embeddings.py`:
  - `embed_text(text) -> list[float]` — calls OpenAI text-embedding-3-small
  - `embed_batch(texts) -> list[list[float]]` — batch embed, 50 at a time
  - Cache: skip if `embedded_at` already set
- [ ] Create `processing/embedding_queue.py`:
  - Reads message_units with no embedding yet
  - Runs batch embedding, stores FLOAT[] in DB
  - Cost tracker (~$0.0001 per 1k tokens for text-embedding-3-small, essentially free)
- [ ] Create `analysis/search.py`:
  - `semantic_search(query, topic_filter, top_k) -> list[SearchResult]`
  - `SearchResult` dataclass: unit_id, post_id, username, topic, text, score
- [ ] Add `pages/Search.py` — semantic search UI:
  - Query input box
  - Topic filter dropdown (taxonomy)
  - Results list with post thumbnail, creator, score, full text
  - Link to original Instagram post
  - "Use this as inspiration" button (wires to Phase 7 Content Studio)

### Acceptance Criteria
- [ ] All message_units have embeddings stored
- [ ] Semantic search returns topically relevant results (manual QA: 10 test queries)
- [ ] Results ranked by cosine similarity score descending
- [ ] Topic filter narrows results correctly
- [ ] Search latency < 2 seconds for corpus of 500 units

### Files created/modified
- `analysis/embeddings.py` — NEW
- `analysis/search.py` — NEW
- `processing/embedding_queue.py` — NEW
- `pages/Search.py` — NEW

---

## Phase 6 — Analytics Dashboard
**Goal:** Give the user a clear view of what competitors are posting, what performs well, and what content gaps exist.
**Est. effort:** 2 days

### Key Views
1. **Creator Leaderboard** — engagement rate by creator, views/likes over time
2. **Topic Distribution** — which topics dominate the content landscape (pie + bar)
3. **Content Gap Map** — which topic × content_type combinations have FEW posts (opportunity map)
4. **Top Performing Reels** — by views, by engagement rate, by topic
5. **Hashtag Intelligence** — most-used hashtags by top-performing creators
6. **Posting Cadence** — posts per week per creator, best days/times

### Tasks
- [ ] Add computed column `engagement_rate` to `posts` (if not done in Phase 1)
- [ ] Create `analysis/analytics.py`:
  - `get_topic_distribution()` → DataFrame
  - `get_content_gap_matrix()` → DataFrame (topic × content_type heatmap)
  - `get_creator_leaderboard()` → DataFrame
  - `get_top_posts(topic, limit)` → DataFrame
- [ ] Add `pages/Analytics.py` — 4-tab Streamlit dashboard:
  - Tab 1: Creator Leaderboard (bar chart + table)
  - Tab 2: Topic Distribution (pie + bar)
  - Tab 3: Content Gap Map (Plotly heatmap)
  - Tab 4: Top Reels (cards with thumbnail + stats)
- [ ] Retire old `pages/2_🤖_AI_Analytics.py` (replace or merge)

### Acceptance Criteria
- [ ] All 4 dashboard tabs load without error with real data
- [ ] Content gap map shows at least one clear opportunity (low-count cell)
- [ ] Creator leaderboard sortable by different metrics

### Files created/modified
- `analysis/analytics.py` — NEW
- `pages/Analytics.py` — NEW
- `pages/2_🤖_AI_Analytics.py` — retire or merge

---

## Phase 7 — Content Generation Studio
**Goal:** Use message units + top-performing reels as context to generate Instagram-ready content: captions, hook lines, and full scripts.
**Est. effort:** 3–4 days

### Features
1. **Caption Generator** — given a topic and tone, generate an Instagram caption with hashtags
2. **Hook Generator** — generate 5 alternative opening lines for a reel on a given topic
3. **Reel Script Generator** — 30-60 second talking-point script with hook → body → CTA structure

### Output Format
```
Hook:     "Did you know 75% of resumes never reach a human?"
Body:     "Here's what most people miss about ATS: ..."
          "Step 1: ..., Step 2: ..., Step 3: ..."
CTA:      "Follow for daily job search tips for Australians."
Caption:  "75% of resumes never reach human eyes 😱 ..."
Hashtags: #JobsAustralia #ATSTips #ResumeAdvice ...
```

### Tasks
- [ ] Create `analysis/content_gen.py`:
  - `generate_caption(topic, angle, reference_posts)` — GPT-4o call
  - `generate_hooks(topic, count=5)` — GPT-4o call
  - `generate_script(topic, duration_sec=45, reference_units)` — GPT-4o call
  - All functions accept `reference_units: list[MessageUnit]` for context grounding
- [ ] Create `analysis/prompts.py` — all system/user prompt templates (separate from logic)
- [ ] Add `pages/Content_Studio.py`:
  - Topic selector
  - Tone selector (professional / friendly / bold / educational)
  - Reference posts selector (multi-select from search results)
  - Generate button → streamed GPT output
  - Copy to clipboard button
  - Save to `generated_content` table
- [ ] New DB table:
  ```sql
  CREATE TABLE generated_content (
      gen_id       TEXT PRIMARY KEY,
      created_at   TIMESTAMP,
      topic        TEXT,
      content_type TEXT,   -- caption | hooks | script
      output_text  TEXT,
      model        TEXT,
      source_units  TEXT[],  -- unit_ids used as context
      tokens_used  INTEGER,
      cost_usd     DOUBLE
  );
  ```

### Acceptance Criteria
- [ ] Caption generator produces Instagram-ready output in < 10 sec
- [ ] Hook generator produces 5 diverse, non-repetitive hooks
- [ ] Script generator output fits within ~45-second speaking time (≈ 90-110 words)
- [ ] Generated content is saved to DB with cost tracking
- [ ] All outputs are grounded with reference to real scraped content

### Files created/modified
- `analysis/content_gen.py` — NEW
- `analysis/prompts.py` — NEW
- `pages/Content_Studio.py` — NEW
- `core/db.py` — `generated_content` table

---

## Phase 8 — Quality Assurance & Testing
**Goal:** Make the pipeline reliable enough to run unattended without manual fixing.
**Est. effort:** 2 days (spread across phases, not a single block)

### Tasks
- [x] `tests/test_scraper.py` — unit tests for deduplication logic, Apify response parsing
- [x] `tests/test_db.py` — test all DB insert/upsert functions with mock data
- [x] `tests/test_transcription.py` — mock Deepgram API, test transcript parsing
- [x] `tests/test_extraction.py` — mock GPT, test message unit parsing + taxonomy validation
- [x] `tests/test_search.py` — test cosine similarity with known vectors
- [x] `tests/test_content_gen.py` — mock GPT, test prompt rendering
- [x] Add `pytest` and `pytest-cov` to `requirements.txt`
- [x] Minimum 70% code coverage on `core/`, `processing/`, `analysis/`

### Acceptance Criteria
- [x] `pytest tests/` passes with 0 failures (132 tests)
- [x] Coverage report shows ≥ 70% on key modules (all Phase 1-8 files ≥ 75%)

---

## Phase 9 — Optimisation & Scale
**Goal:** Handle 100+ creators, 10k+ posts, 10k+ message units without performance degradation.
**Est. effort:** 1–2 days

### Tasks
- [ ] Add DuckDB indexes on `posts.account_id`, `posts.posted_at`, `message_units.topic`
- [ ] Implement cursor-based pagination in Data Viewer (currently loads all rows)
- [ ] Batch embedding: send up to 100 texts per OpenAI API call
- [ ] Transcription: process up to 5 files concurrently (ThreadPoolExecutor)
- [ ] Add DuckDB WAL checkpoint job (prevent WAL file growing unbounded)
- [ ] Profile Streamlit page load times — any page > 3s gets a `@st.cache_data` decorator
- [ ] Cost dashboard: show total spend across Deepgram + OpenAI APIs

### Acceptance Criteria
- [ ] 1000 message units embedded in < 5 minutes
- [ ] Data Viewer loads page 1 in < 1 second
- [ ] All Streamlit pages load in < 3 seconds with 1000+ posts

---

## Git Workflow

### Rule: Every phase commit and push
```bash
# 1. Create branch for the phase
git checkout -b phase-N-name

# 2. Work on the phase (multiple small commits ok)
git add .
git commit -m "phase N: short description of change"

# 3. When phase is DONE (all acceptance criteria met):
# Update PLAN.md — mark phase as [x] Done
git add PLAN.md
git commit -m "phase N complete: mark done in PLAN.md"
git push origin phase-N-name

# 4. Merge to main
git checkout main
git merge phase-N-name
git push origin main
```

### Commit message format
```
phase N: <imperative verb> <what changed>

Examples:
phase 1: add deduplication to raw_scrapes insert
phase 2: implement yt-dlp video downloader
phase 3: add Deepgram Nova-2 transcriber
```

### Branch naming
```
phase-0-foundation
phase-1-scraping
phase-2-media
phase-3-transcription
phase-4-extraction
phase-5-search
phase-6-analytics
phase-7-content
phase-8-qa
phase-9-optimisation
```

---

## Architecture Overview

```
Instagram (Apify) ──► raw_scrapes ──► posts ──► creator_accounts
                                         │
                              ┌──────────┼──────────┐
                              │          │          │
                        video files   CDN URLs   metadata
                              │
                        audio extract (ffmpeg)
                              │
                        Deepgram Nova-2
                              │
                        transcripts + transcript_words
                              │
                        GPT-4o-mini extraction
                              │
                        message_units (topic/type tagged)
                              │
                        text-embedding-3-small
                              │
                        message_units.embedding (FLOAT[1536])
                              │
               ┌──────────────┴──────────────────┐
               │                                  │
        DuckDB cosine search              GPT-4o generation
               │                                  │
        Search UI (Page 4)           Content Studio (Page 6)
                                              │
                                     generated_content (DB)
```

---

## Environment Setup

```bash
# Clone & activate
git clone <repo-url>
cd senpai-reel
python -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Set up secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and fill in your API keys

# Run
streamlit run app.py
```

### Required secrets (`.streamlit/secrets.toml`)
```toml
APIFY_TOKEN = "apify_api_..."
OPENAI_API_KEY = "sk-..."
DEEPGRAM_API_KEY = "..."
```

---

## Cost Estimates (per 100 reels)

| Service | Cost | Notes |
|---------|------|-------|
| Apify scraping | ~$0.25 | $0.0025/reel |
| Deepgram Nova-2 | ~$0.58 | avg 100sec/reel @ $0.0058/min |
| GPT-4o-mini extraction | ~$0.10 | ~50k input tokens/100 reels |
| text-embedding-3-small | ~$0.01 | essentially free |
| GPT-4o content gen | ~$0.05 | per generation run |
| **Total/100 reels** | **~$1.00** | First ~3400 reels free (Deepgram $200 credit) |

---

*Last updated: 2026-04-06*
*Maintained by: GitHub Copilot + Arian Kalantari*
*Copilot: Read this file at the start of every session to understand project state.*
