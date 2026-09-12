# Senpai Reel — System Guide
*Last updated: 6 April 2026*

---

## What this app does (in one sentence)

Senpai Reel scrapes up to 23 competitor Instagram accounts in the **Jobs-in-Australia niche**, analyses what they say in their reels, and generates content ideas grounded in that competitive intelligence.

---

## The full pipeline (what happens under the hood)

Data flows forward through the operational stages below. Each stage only works
on the active client.

```
Instagram Accounts
      │
      ▼
[Stage 1] SCRAPE          Apify API → raw JSON + structured DB rows
      │
      ▼
[Stage 2] DOWNLOAD        Direct HTTP (or yt-dlp fallback) → .mp4 files on disk
      │
      ▼
[Stage 2b] EXTRACT AUDIO  ffmpeg → 16kHz mono .wav files on disk
      │
      ▼
[Stage 2c] EXTRACT FRAMES ffmpeg → compact JPEG keyframes on disk
      │
      ▼
[Stage 3] TRANSCRIBE      Deepgram Nova-2 → full text + word timestamps in DB
      │
      ▼
[Stage 3b] ARCHIVE VIDEO  move processed source .mp4 to external archive, if configured
      │
      ▼
[Stage 4] EXTRACT         GPT-4o-mini → "message units" (structured knowledge) in DB
      │
      ▼
[Stage 4b] EMBED          OpenAI text-embedding-3-small → 1536-dim vectors in DB
      │
      ▼
[Stage 5] USE             Search · Analytics · Content Studio
```

---

## Stage-by-stage technical breakdown

### Stage 1 — Scraping (`collection/scraper.py`)

**What it does:**
Calls the Apify actor `apify~instagram-reel-scraper` for one or more Instagram usernames. 
The actor returns a JSON array of reel objects. Each object has: 
`shortCode`, `caption`, `likesCount`, `videoViewCount`, `commentsCount`, `videoUrl`, 
`audioUrl`, `timestamp`, `videoDuration`, `ownerUsername`, `ownerId`, `images[]`, etc.

**What gets saved to DB:**
1. Raw JSON blob per reel → `raw_scrapes` table (for debugging / auditing)
2. Creator profile → `creator_accounts` table (upserted by username)
3. Individual reel → `posts` table (upserted by `shortCode`)
4. Legacy copies → `reels`, `comments`, `tagged_users` tables (kept for backward compat)
5. Job audit record → `scrape_jobs` table (with status, counts, timestamps)

**Deduplication:**  
The `shortCode` field is used as the primary key for posts. If a reel already exists, only engagement stats (likes, views) are updated — download status, audio path, transcript, etc. are never overwritten.

**Retry logic:**  
On HTTP errors the scraper tries up to 3 times with exponential backoff (1s, 2s). Auth errors (401/403) abort immediately — no point retrying.

**Accounts tracked (`collection/account_list.py`):**  
23 curated Jobs-AU competitor accounts across 5 categories:
- Resume & Career Coaches: `resumeworded`, `careersidekick`, `the.career.strategist`, `iamhannah.co`, `careerwithsam`, `jobsearchcoach`, `theresumewriter`, `careercoachmelbourne`
- Recruitment & HR: `hays.australia`, `robertwaltersau`, `michaelpageaustralia`, `reedrecruitment`, `seek.com.au`, `hrmonline`
- ATS & Resume Tips: `tealau`, `kickresume`, `resumetricks`
- Interview Prep: `interviewguru`, `theinterviewcoach`
- LinkedIn & Job Search: and ~4 more

**Performance:**  
Each account scrape takes 10–60 seconds (Apify cloud actor spins up). A full 23-account batch = ~15–30 minutes.

---

### Stage 2 — Video download (`processing/download.py`)

**What it does:**  
For every post with `download_status = 'pending'`, downloads the `video_url` CDN link to `downloads/{post_id}.mp4`.

**Two-method approach (with automatic fallback):**
1. **Direct HTTP** (`requests` streaming, 1 MB chunks) — fast, works ~70% of the time when the Apify CDN URL is still valid
2. **yt-dlp** — used if the direct download gets a non-video content type or fails; slower but more reliable

**Success criteria:** File must be > 10 KB. Anything smaller is treated as a failed partial download.

**DB writes on each attempt:**
- `download_status` set to `'downloading'` at start, then `'done'` or `'failed'`
- `local_video_path`, `file_size_mb`, `downloaded_at` written on success

**Important timing issue:**  
Apify CDN URLs expire after ~24–48 hours. If you scrape but don't download within that window, the direct HTTP method will fail. yt-dlp may still work by fetching a fresh URL, but is slower and less reliable.

**Audio and keyframe extraction (`processing/audio.py`, `processing/media_archive.py`):**
Runs as a sub-step after download. Calls `ffmpeg` to convert `.mp4` → 16kHz mono `.wav` in `audio_extracts/`. This specific format is required by Deepgram. It also extracts a compact local keyframe set to `keyframes/{post_id}/` so later visual/OCR work can use the editing reference without keeping the source video in the working directory.
**ffmpeg must be installed:** `brew install ffmpeg` (path: `/opt/homebrew/bin/ffmpeg`)

Once audio, keyframes, and transcript exist, the source video is eligible for archiving. If `SENPAI_ARCHIVE_DIR` is unset or the external drive is disconnected, archiving is skipped and the `.mp4` remains in `downloads/`. The pipeline continues normally and does not mark the post archived. When the archive target is reachable, the source video is moved to the archive and `posts.archived_video_path` records where it went.

---

### Stage 3 — Transcription (`processing/transcribe.py`, `processing/transcription_queue.py`)

**What it does:**  
For every post with a `.wav` audio file but no transcript yet, sends the audio to the Deepgram API and saves the result.

**Primary provider: Deepgram Nova-2**
- `en-AU` language model (Australian English)
- Returns full transcript text + word-level timestamps (start/end seconds + confidence per word)
- Cost: **$0.0058 per minute** of audio
- You have $200 free credits ≈ 34,000 minutes of audio

**Fallback provider: OpenAI Whisper-1**
- Slower and more expensive than Nova-2 for this use case
- Only used if you pass `provider='whisper'` explicitly

**What gets saved:**
- `transcripts` table: full text, language, confidence score, duration, word count, cost, raw API response JSON
- `transcript_words` table: every word with its start/end timestamp and confidence (enables future karaoke-style UI or search within timestamps)

**The queue:**  
`run_transcription_queue()` finds up to N posts with audio but no transcript, processes them one by one, and persists after each one. If it crashes partway through, already-transcribed posts are never re-processed.

---

### Stage 4 — Knowledge extraction (`analysis/extraction.py`, `processing/extraction_queue.py`)

**What it does:**  
For every transcript that hasn't been processed, sends the transcript text to GPT-4o-mini with a structured prompt and extracts 1–8 "message units" per transcript.

**What is a message unit?**  
A message unit is one concrete, self-contained idea from a reel. Examples:
- *"91% of recruiters use ATS to filter resumes before a human reads them"* (type: stat, topic: ATS)
- *"Don't put your photo on an Australian resume"* (type: warning, topic: Resume)
- *"Use the STAR method for every behavioural interview question"* (type: tip, topic: Interview)

**The taxonomy (`analysis/taxonomy.py`):**

Topics (11):
`ATS` · `Resume` · `CoverLetter` · `Interview` · `LinkedIn` · `JobSearch` · `Salary` · `CareerChange` · `Recruiter` · `Visa` · `General`

Content types (8):
`tip` · `warning` · `stat` · `myth` · `story` · `hook` · `cta` · `other`

**GPT output per unit:**
- `text` — verbatim or close paraphrase (≤200 chars)
- `claim` — core assertion in one sentence (GPT's words)
- `advice` — actionable instruction, or null
- `topic` — one of the 11 topics above
- `subtopic` — freeform refinement (e.g. "salary_negotiation")
- `content_type` — one of the 8 types above
- `confidence` — 0.0–1.0 (GPT's certainty in the extraction)

**Cost:** GPT-4o-mini = $0.15/1M input + $0.60/1M output tokens. A typical 60-second reel transcript (~200 words) costs ~$0.0003. Processing 1,000 reels ≈ $0.30 total.

**DB write:**  
Each unit is written to `message_units` with its own UUID. The extraction cost is also written back to the `transcripts.extraction_cost_usd` column.

---

### Stage 4b — Embedding (`analysis/embeddings.py`)

**What it does:**  
For every message unit without an embedding, calls OpenAI `text-embedding-3-small` to generate a 1536-dimensional float vector and stores it in `message_units.embedding` as `FLOAT[1536]`.

**Why:** DuckDB can compute cosine similarity between two vectors natively (`list_cosine_similarity()`), enabling semantic search without an external vector database.

**Cost:** $0.02/1M tokens — essentially free. 10,000 units ≈ $0.002.

**Batch size:** 50 texts per API call to stay well within rate limits.

---

### Stage 5 — Usage (visible pages)

#### Pipeline (`pages/Pipeline.py`)

Client-scoped pipeline dashboard with one-click operation:

| Stage | Trigger |
|-------|---------|
| Scrape | Scrape configured competitor accounts |
| Download | Download pending Apify media URLs |
| Audio | Extract WAV audio and keyframes from downloaded MP4s |
| Transcribe | Run the Deepgram queue |
| Archive | Move processed source MP4s to `SENPAI_ARCHIVE_DIR`, if reachable |
| Extract units | Run knowledge extraction |
| Embed | Run semantic embeddings |

The **Run Everything Pending** button chains those stages in dependency order. It uses a DuckDB-backed `pipeline_locks` row so only one write-heavy pipeline run can operate at a time, and it shows a clear busy state on other write pages. If a process dies mid-run, the Pipeline page shows lock age, last heartbeat, and a force-release control once the heartbeat is stale.

Item-level queues run with bounded workers. I/O-bound work (downloads, Deepgram, OpenAI extraction) defaults to 8 workers via `SENPAI_IO_WORKERS`; CPU-heavy ffmpeg extraction defaults to 2 workers via `SENPAI_CPU_WORKERS`. DuckDB writes are serialized behind a shared writer lock.

The page also warns when pending downloads are approaching Apify CDN expiry. URLs older than roughly 18 hours are flagged because media links commonly expire within 24–48 hours.

#### Onboarding (`pages/Onboarding.py`)

Guided first-run path:

1. Name the client and niche.
2. Add competitor Instagram accounts.
3. Open the Pipeline page.
4. Open Content Studio after extraction.

#### Settings (`pages/Settings.py`)

Shows whether `APIFY_TOKEN`, `DEEPGRAM_API_KEY`, and `OPENAI_API_KEY` are configured. Users can paste keys for the current Streamlit session. Session keys are not persisted; permanent setup still uses `.streamlit/secrets.toml` or environment variables.

Apify scrape cost uses `APIFY_USD_PER_RESULT` when configured. If that value is missing, scrape cost is shown as unknown rather than zero.

#### Costs (`pages/Costs.py`)

Per-client cost meter with this-month and all-time totals for Apify scraping, Deepgram transcription, OpenAI extraction + embeddings, and OpenAI generation.

The page also shows reels scraped, reels transcribed, reels analysed, pieces generated, and all-time cost per generated piece. That cost-per-piece value is the number to compare against the client's existing tool subscriptions.

#### Home / Scraper (`app.py`)

The operational control centre. Three tabs:

| Tab | What it does |
|-----|-------------|
| **Single Account** | Scrape one Instagram handle on demand (10–60 sec) |
| **Batch Scrape** | Edit the 23-account list and scrape all at once |
| **Download Queue** | Download pending videos + shows pending/done/failed counts |

Also shows a scrape history expander (last 20 jobs with status, timestamps, counts).

If `APIFY_TOKEN` is missing, the page shows a plain configuration message and links to Settings. The page no longer breaks during first run.

---

#### Data Viewer (`pages/Data_Viewer.py`)

Paginated table (50 rows/page) of all posts in the DB. 

Sidebar filters: account, sort column (engagement/views/likes/date).  
Top bar: total reels, avg likes, avg views, avg engagement, avg duration.  
Columns shown: username, caption (truncated to 100 chars), likes, views, engagement %, posted date, duration, download status.  
Export: CSV download button, JSON download button.  
Pagination: Previous / Next buttons with "Showing rows X–Y of Z" counter.

Falls back to `raw_scrapes` table if `posts` is empty (useful in early stages).

---

#### Corpus Explorer (`pages/Corpus_Explorer.py`)

Browse and search transcripts.

Top stats: how many posts are transcribed / pending / cost so far.  
**Trigger transcription queue** — available in an expandable section. Requires `DEEPGRAM_API_KEY`; if it is missing, the page says transcription is not configured and links to Settings.
Search bar: filter transcripts by keyword.  
Result list: clicking a row shows the full transcript text + word timestamps.

---

#### Search (`pages/Search.py`)

Semantic and keyword search over all extracted message units.

Top stats: total units extracted, how many are embedded.  
Search mode auto-selects:
- If `OPENAI_API_KEY` is set **and** units are embedded → semantic search (cosine similarity)
- Otherwise → keyword search (SQL `LIKE`)

Filters: topic (dropdown), content type (dropdown), result count.  
Results shown as cards: score, topic badge, content type badge, text, claim, creator username, link to source video.

**Also hides pipeline controls here:**  
- "Run Extraction Queue" expander (requires `OPENAI_API_KEY`)
- "Run Embedding Pipeline" trigger in sidebar

---

#### Analytics (`pages/Analytics.py`)

5-tab competitor intelligence dashboard (requires data in `message_units`):

| Tab | Shows |
|-----|-------|
| Leaderboard | Bar chart: creator accounts ranked by post count + total engagement |
| Topic Distribution | Pie + bar chart: which topics are covered most across all reels |
| Content Gap Map | Heatmap pivot: creator × topic, showing who covers what (and what nobody covers) |
| Top Reels | Table of highest-engagement posts, filterable by topic, with video URL links |
| Hashtags | Bar chart of most-used hashtags across all posts |

Requires `plotly` for charts — degrades gracefully to text tables if unavailable.

---

#### Content Studio (`pages/Content_Studio.py`)

AI-powered content generation. **Requires `OPENAI_API_KEY` for new output.** Without it, the page explains what is unavailable and still shows generation history.

Controls: Topic (dropdown), Tone (professional/friendly/bold/educational), Angle (free text).  
Reference units: search for relevant competitor insights to ground the generation (optional but strongly recommended — improves output quality).  

4 tabs:
| Tab | Generates | Model | Cost estimate |
|-----|-----------|-------|---------------|
| Caption | Full Instagram caption with hashtags | gpt-4o-mini | ~$0.001 |
| Hooks | 5 opening hooks for a reel | gpt-4o-mini | ~$0.001 |
| Script | Full 60-second reel script | gpt-4o-mini | ~$0.002 |
| History | All previously generated content with costs | — | — |

All outputs are saved to the `generated_content` table automatically.

---

## Database schema (16 core tables)

| Table | Stage | What it holds |
|-------|-------|---------------|
| `clients` | 0 | Client profile, niche, notes, brand voice, and target audience |
| `client_accounts` | 0 | Competitor Instagram handles per client |
| `client_posts` | 0 | Client-to-post visibility links |
| `raw_scrapes` | 1 | Raw JSON blobs as returned by Apify |
| `profiles` | 1 | Legacy: one row per scrape run summary |
| `creator_accounts` | 1 | One row per Instagram account (bio, followers, etc.) |
| `posts` | 1–3b | One row per reel (canonical, includes download, audio, keyframe, and archive paths) |
| `scrape_jobs` | 1 | Audit log of every scrape run (status, counts, errors, Apify cost) |
| `reels` | 1 | Legacy structured reels (kept for old pages compatibility) |
| `comments` | 1 | Reel comments (recursive, supports replies) |
| `tagged_users` | 1 | Users tagged in reels |
| `transcripts` | 3 | One row per transcribed post (text, confidence, cost) |
| `transcript_words` | 3 | One row per word with start/end timestamps |
| `message_units` | 4 | One row per extracted knowledge unit (text, topic, type, embedding, embedding cost) |
| `generated_content` | 5 | All AI-generated captions/hooks/scripts with cost tracking |
| `pipeline_locks` | Ops | Single-row lock for write-heavy pipeline runs |

**Storage location:** `reels.duckdb` (single file in project root, no server needed)  
**WAL file:** `reels.duckdb.wal` — this grows over time. DuckDB checkpoints it automatically, but if it gets large (>100 MB), run `CHECKPOINT;` from a DuckDB shell.

---

## Required credentials (`.streamlit/secrets.toml`)

```toml
APIFY_TOKEN = "apify_api_..."          # Required for Stage 1 scraping
DEEPGRAM_API_KEY = "..."               # Required for Stage 3 transcription
OPENAI_API_KEY = "sk-..."             # Required for Stages 4, 4b, and Content Studio
APIFY_USD_PER_RESULT = "0.0025"      # Optional; used for scrape-job cost capture
```

Keys can also be pasted on the Settings page for the current Streamlit session.

Optional worker-limit environment variables:

```bash
SENPAI_IO_WORKERS=8     # Downloads, Deepgram, OpenAI extraction
SENPAI_CPU_WORKERS=2    # ffmpeg audio/keyframe extraction
SENPAI_ARCHIVE_DIR=/Volumes/SenpaiArchive  # Optional external source-video archive
SENPAI_FFMPEG_HWACCEL=videotoolbox         # Optional; benchmark before enabling
```

If any key is missing:
- No `APIFY_TOKEN` → Scrape buttons explain that scraping needs Apify and link to Settings.
- No `DEEPGRAM_API_KEY` → Transcription says it is not configured and links to Settings.
- No `OPENAI_API_KEY` → Search falls back to keyword mode; extraction, embeddings, and generation explain what needs OpenAI.

---

## Current user journey

### Recommended path:

```
1. Open Onboarding.
2. Name the client and add competitor accounts.
3. Open Pipeline and click Run Everything Pending.
4. Use Search, Analytics, Costs, or Content Studio.
```

The older per-stage controls remain on Home, Corpus Explorer, and Search while the chained pipeline proves itself.

Empty states on each page point back to the Pipeline stage that fills the missing data.

---

## Known issues & things to watch for

| Issue | Severity | Notes |
|-------|----------|-------|
| Apify CDN URLs expire in ~24–48h | Medium | Pipeline and Download Queue warn when pending media URLs are getting old |
| External archive drive disconnected | Low | Archive stage skips cleanly and leaves source videos in `downloads/`; it never records an archive path unless the move succeeds. |
| DuckDB lock conflict | Medium | Pipeline runs use `pipeline_locks`; write pages show a busy state while a run is active. Pipeline can force-release a stale heartbeat. |
| `use_container_width` warnings | Low | Fixed in latest commit — cosmetic only |
| WAL file growth | Low | `reels.duckdb.wal` will grow with each write session. No auto-checkpoint UI. |
