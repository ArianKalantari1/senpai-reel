# Senpai Reel — Full Project Documentation

> **Purpose of this document:** A detailed, consultant-ready breakdown of everything that has been built, how it works behind the scenes, what the current state is, and where the gaps are. Use this as the planning foundation for the next phase.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [The Database — What Gets Stored](#4-the-database--what-gets-stored)
5. [Module Breakdown](#5-module-breakdown)
   - [core/ — Database Layer](#51-core--database-layer)
   - [collection/ — Video Download Tools](#52-collection--video-download-tools)
   - [pipeline/ — ETL Pipeline](#53-pipeline--etl-pipeline)
   - [analysis/ — AI & ML Analysis Layer](#54-analysis--ai--ml-analysis-layer)
   - [graph/ — Graph Network Engine](#55-graph--graph-network-engine)
6. [The Streamlit App — Page by Page](#6-the-streamlit-app--page-by-page)
   - [app.py — Home (Scraper)](#61-apppy--home-scraper)
   - [Page 0 — Test](#62-page-0--test)
   - [Page 1 — Data Viewer](#63-page-1--data-viewer)
   - [Page 2 — AI Analytics](#64-page-2--ai-analytics)
   - [Page 3 — AI Video Analysis](#65-page-3--ai-video-analysis)
   - [Page 4 — Graph Network](#66-page-4--graph-network)
   - [Page 5 — Graph Query Engine](#67-page-5--graph-query-engine)
7. [External Integrations & APIs Used](#7-external-integrations--apis-used)
8. [Tech Stack & Dependencies](#8-tech-stack--dependencies)
9. [Current State: What Works vs What Doesn't](#9-current-state-what-works-vs-what-doesnt)
10. [Known Gaps & Issues](#10-known-gaps--issues)
11. [Directory Structure Explained](#11-directory-structure-explained)
12. [How to Run the App](#12-how-to-run-the-app)

---

## 1. What Is This Project?

**Senpai Reel** is an Instagram Reels intelligence platform. Its goal is to scrape, store, analyse, and surface insights from Instagram Reels content — helping content creators and strategists understand:

- What content performs well (engagement, virality, timing)
- How content is structured (video pacing, camera motion, scene cuts)
- What relationships exist between posts (hashtag networks, user influence, similarity)
- What drives engagement (ML-predicted features, caption patterns, music, location)

The application is built as a **multi-page Streamlit web app** that runs locally. It combines:
- **Apify** for Instagram data scraping (no direct Instagram API required)
- **DuckDB** as a fast, local analytical database
- **NetworkX + Plotly** for graph visualisation
- **OpenCV + scikit-learn** for local video and ML analysis
- **OpenAI GPT-4 Vision** (optional, not yet wired to the UI) for deep video understanding

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│   app.py (scraper) + 6 pages in pages/                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ reads/writes
┌────────────────────────────▼────────────────────────────────────┐
│                    DATA LAYER (reels.duckdb)                    │
│   raw_scrapes │ reels │ comments │ tagged_users │ profiles       │
│               + video_analysis (analysis results table)         │
└──────┬─────────────────────────────────┬──────────────────────-─┘
       │                                 │
┌──────▼──────────┐             ┌────────▼───────────────────────┐
│  APIFY API      │             │   GRAPH LAYER (SQLite)         │
│  (Instagram     │             │   video_graphs.db              │
│   scraping)     │             │   demo_video_graphs.db         │
│                 │             │   frame_nodes, frame_edges     │
│  Actor:         │             └────────────────────────────────┘
│  instagram-     │
│  reel-scraper   │             ┌────────────────────────────────┐
└─────────────────┘             │   LOCAL FILES                  │
                                │   downloads/  (mp4 videos)     │
┌────────────────────────┐      │   audio_extracts/ (wav files)  │
│  ML/AI LAYER           │      │   ai_analysis/ (results)       │
│  analysis/             │      └────────────────────────────────┘
│  - engagement_predictor│
│    (scikit-learn RF)   │
│  - ai_video_analyzer   │
│    (OpenAI Vision)     │
│  - audio_analyzer      │
│    (Whisper + librosa) │
└────────────────────────┘
```

---

## 3. Data Flow — End to End

Below is the exact sequence of events from button click to data appearing on screen.

### Step 1 — Scraping (app.py)

1. User enters an Instagram username and a maximum reel count in the UI.
2. On "Run Scraper" click, the app calls `run_scraper()` which hits the **Apify REST API**:
   ```
   POST https://api.apify.com/v2/acts/apify~instagram-reel-scraper/run-sync-get-dataset-items?token=<APIFY_TOKEN>
   ```
   with a payload containing the full Instagram profile URL and result limit.
3. Apify runs the scrape synchronously (takes ~10–30 seconds) and returns a **JSON array** where each item represents one reel. Each object contains ~40 fields including:
   - `id`, `shortCode`, `url` (post URL)
   - `caption`, `hashtags`, `mentions`
   - `likesCount`, `videoViewCount`, `videoPlayCount`, `commentsCount`
   - `videoUrl` (a CDN URL, expires after ~24–48 hours)
   - `videoDuration`, `timestamp`
   - `ownerUsername`, `ownerId`
   - `musicInfo` (artist, song name)
   - `locationName`, `locationId`
   - `isSponsored`, `isPinned`
   - `latestComments` (nested array of comment objects)
   - `taggedUsers` (array of tagged account objects)

### Step 2 — Raw Storage (`core/db.py → save_raw_scrape`)

4. The entire JSON array is saved **as-is** into the `raw_scrapes` DuckDB table. Each item is a separate row with:
   - An index `id`
   - The `profile` (username)
   - The full `raw` JSON blob
   - A `scraped_at` UTC timestamp

   This is the **primary source of truth**. All downstream analysis reads from here.

### Step 3 — Optional Structured Parsing (`core/db.py → save_structured_scrape`)

5. (Optional, toggled by a checkbox in the UI) The raw JSON is further parsed and **denormalised** into typed relational tables:
   - `reels` — one row per reel with all metrics as typed columns
   - `comments` — flattened recursively (replies become regular rows with a `parent_id`)
   - `tagged_users` — one row per tagged account per reel

   This step is required for **Page 2 (AI Analytics)** to work, because that page queries these structured tables.

### Step 4 — Analysis (multiple paths)

Once data is in the database, three independent analysis paths are available:

**Path A — Metadata Analytics (Page 1 & 2):**  
Read from `raw_scrapes` or `reels`, compute engagement rates, timing, hashtag counts, and display via Plotly charts.

**Path B — ML Engagement Prediction (`analysis/engagement_predictor.py`):**  
A **Random Forest Regressor** is trained on existing post features (caption length, duration, hashtag count, day of week, hour, whether it has music/location, owner historical avg engagement). It predicts engagement rate for new posts and can run "what-if" optimisation scenarios.

**Path C — Video-Level Analysis (Page 3):**  
Requires `.mp4` files to be in the `downloads/` folder. The page runs `analyze_videos()` which reads file metadata + cross-references the database, computes virality scores via heuristic rules, and displays breakdowns by category, engagement, and timing.

**Path D — Deep AI Vision (`analysis/ai_video_analyzer.py`):**  
The most powerful path, **not yet wired to the Streamlit UI**. Extracts N key frames from a `.mp4` file using OpenCV, encodes them as base64 JPEGs, and sends them to `gpt-4-vision-preview` together with reel metadata. The response (scene description, objects, text, emotion, category, engagement prediction) is stored in the `video_analysis` DuckDB table.

**Path E — Graph Network Analysis (Page 4 & 5):**  
All scrape data is loaded from `raw_scrapes`, posts become graph nodes, and edges are created based on a multi-factor **content similarity score** (shared hashtags, same owner, similar engagement, cross-mentions, similar duration). NetworkX is used for the graph, Plotly for visualisation. A separate SQLite-based graph system (`video_graphs.db`) stores frame-level graphs for individual videos.

---

## 4. The Database — What Gets Stored

### `reels.duckdb` (Main Database — DuckDB)

| Table | What it stores | How it's populated |
|---|---|---|
| `raw_scrapes` | Full raw JSON per reel, exact Apify response | Every scrape run |
| `reels` | Structured, typed reel metrics | Optional step after scraping |
| `comments` | Flattened comment tree (threaded to flat) | Optional step after scraping |
| `tagged_users` | All tagged accounts per reel | Optional step after scraping |
| `profiles` | Scrape summary per username per run | Optional step after scraping |
| `video_analysis` | AI analysis results (GPT-4 Vision output) | `ai_video_analyzer.py` (not yet in UI) |

### `reels` Table — Full Schema

```sql
reel_id TEXT PRIMARY KEY,    -- Instagram internal post ID
profile TEXT,                -- Scraped username
shortcode TEXT,              -- Short code (e.g. "DRt0DAPEcv1")
caption TEXT,
likes INTEGER,
views INTEGER,
video_play_count INTEGER,
comments_count INTEGER,
video_url TEXT,              -- CDN URL (expires ~48h)
audio_url TEXT,
thumbnail_url TEXT,
display_url TEXT,
all_images JSON,             -- Array of image URLs
timestamp TIMESTAMP,         -- Post publication time
location_name TEXT,
location_id TEXT,
duration DOUBLE,             -- Video length in seconds
is_pinned BOOLEAN,
is_sponsored BOOLEAN,
owner_username TEXT,
owner_id TEXT,
raw JSON,                    -- Original full JSON kept for reference
scraped_at TIMESTAMP
```

### `video_graphs.db` and `demo_video_graphs.db` (SQLite — Graph Databases)

These are separate SQLite databases used by the graph engine. They store:

| Table | What it stores |
|---|---|
| `frame_nodes` | One row per video frame: timestamp, brightness, edge density, dominant colours, motion vectors |
| `frame_edges` | Relationships between frames: temporal (every frame pairs), scene_transition (cuts/fades), similarity (repeated visual patterns) |

---

## 5. Module Breakdown

### 5.1 `core/` — Database Layer

**`core/db.py`** — The central data access layer. Provides:

- `init_db()` — Creates all DuckDB tables if they don't exist. Called once at app startup.
- `save_raw_scrape(profile, items)` — Stores the raw Apify JSON response exactly as received. This is the most important function — it's the gatekeeper of all raw data.
- `save_structured_scrape(profile, items)` — Wrapper that calls `save_scrape()`.
- `save_scrape(profile, items)` — Full structured parse: iterates every reel item, extracts typed fields, inserts into `reels`, `comments` (recursively), `tagged_users`, and `profiles`.
- `fix_timestamp(ts)` — Converts ISO 8601 `"2024-01-15T18:30:00Z"` to `"2024-01-15 18:30:00"` for DuckDB compatibility.
- `_insert_comments_recursive(conn, reel_id, comments, parent_id)` — Flattens threaded comment trees (where replies are nested under parent comments) into a flat table with a `parent_id` column for hierarchy reconstruction.

---

### 5.2 `collection/` — Video Download Tools

These are **helper scripts, not integrated into the Streamlit UI**.

| File | Purpose |
|---|---|
| `get_urls.py` | Reads `raw_scrapes`, prints all post URLs to terminal for manual copy-paste to download sites |
| `url_checker.py` | Tests whether stored CDN video URLs are still valid (they expire) using concurrent HTTP HEAD requests |
| `video_downloader.py` | Attempts to download videos directly from stored CDN URLs or re-fetch fresh URLs from the post page. CDN URLs expire quickly so this often fails. |
| `instagram_downloader.py` | A more complete downloader that supports both direct CDN download and creating a `batch_download.sh` shell script using `yt-dlp` |
| `download_all_videos.py` | Batch download script |
| `batch_download.sh` | Generated bash script for `yt-dlp` batch downloads |

**The core challenge:** Instagram CDN video URLs expire within ~24–48 hours. Once the Apify scrape is done, the `videoUrl` in the database becomes a dead link. To re-download videos, you need to either: (a) scrape again immediately, or (b) use `yt-dlp` with the post URL which handles auth/redirect internally.

---

### 5.3 `pipeline/` — ETL Pipeline

| File | Status | Purpose |
|---|---|---|
| `transform.py` | ✅ Implemented | A duplicate of the structured scrape logic from `core/db.py`. Contains `save_scrape()` and `init_db()`. Was an earlier standalone version before being moved to `core/`. |
| `etl_basic.py` | ❌ Empty | Intended entry point for a full ETL pipeline. **Currently blank.** |

**Note:** There are two parallel implementations of the structured save logic — one in `core/db.py` and one in `pipeline/transform.py`. They do the same thing. The `core/db.py` version is the one actually used by the app. The `pipeline/` module was intended to be a more formal ETL pipeline but was never fully built.

---

### 5.4 `analysis/` — AI & ML Analysis Layer

#### `analysis/engagement_predictor.py` — `EngagementPredictor` class

A **scikit-learn Random Forest Regressor** for predicting engagement rate.

**How it works:**
1. `prepare_training_data()` — Reads all posts from `raw_scrapes`, extracts feature vectors, uses `likesCount / videoViewCount * 100` as the target Y variable.
2. Feature extraction (`extract_features(post_data)`):
   - `duration` — Video length in seconds
   - `caption_length` — Number of characters
   - `mentions_count` / `hashtags_count` — Extracted from the data
   - `has_exclamation` / `has_question` / `has_emoji` — Boolean flags from caption text
   - `is_food_content` / `is_event_content` / `is_location_content` — Keyword-based categorisation
   - `hour_of_day` / `day_of_week` / `is_weekend` — Parsed from the post timestamp
   - `owner_avg_engagement` — Historical average for that owner, computed from the database
   - `has_music` / `is_sponsored` — Boolean flags
3. `train_model()` — 70/30 train-test split, trains the model, reports MAE and R², saves to `engagement_prediction_model.pkl`
4. `predict_engagement(post_data)` — Runs a single prediction
5. `analyze_content_optimization(sample_post)` — Runs "what-if" scenarios (e.g. `what if I add a location? what if duration = 30s?`) and reports predicted engagement lift
6. **Not yet integrated into the Streamlit UI.** It runs as a standalone Python script.

#### `analysis/ai_video_analyzer.py` — `VideoAIAnalyzer` class

**Deep AI video analysis using OpenAI GPT-4 Vision.**

**How it works:**
1. `extract_frames(video_path, num_frames=5)` — Uses OpenCV to open the `.mp4` file, calculates uniform sampling intervals, reads N frames, converts each to a JPEG and then to base64 string.
2. `analyze_with_openai_vision(frames, metadata)` — Constructs a multi-turn OpenAI message: first a system prompt defining the analysis task, then reel metadata as context, then each base64 frame as an image message. Calls `gpt-4-vision-preview` with `max_tokens=1500`. Expects JSON response with: `scene_description`, `object_detection`, `text_extraction`, `color_analysis`, `content_category`, `engagement_prediction`, `brand_mentions`, `emotion_analysis`.
3. `analyze_with_local_models()` — Fallback placeholder for local YOLO/CLIP/Whisper integration (not yet implemented).
4. `analyze_video_file(video_filepath)` — Orchestration function: load metadata from DB, extract frames, run analysis, calculate processing time, save to `video_analysis` DuckDB table.
5. `batch_analyze_downloaded_videos()` — Loops over all `.mp4` files in `downloads/`, calls `analyze_video_file()` for each.
6. **Requires OpenAI API key. Not yet connected to Streamlit UI.**

#### `analysis/audio_analyzer.py` — `VideoAudioAnalyzer` class

**Multi-modal audio understanding using Whisper + librosa.**

**How it works:**
1. `extract_audio_from_video(video_path)` — Uses MoviePy to extract the audio track from an `.mp4` and save as a `.wav` file in `audio_extracts/`.
2. Audio features extracted using **librosa**: MFCC coefficients (spectral fingerprints), spectral centroid, spectral rolloff, zero-crossing rate, tempo.
3. Speech transcription via **OpenAI Whisper** (local model): identifies speech segments, start/end timestamps, confidence, language.
4. Results stored in a separate SQLite DB (`video_audio_analysis.db`) with tables: `audio_metadata`, `speech_transcription`, `audio_features`.
5. **Heavy dependencies (whisper, moviepy, librosa). Not yet integrated into the Streamlit UI.**

#### `analysis/simple_audio_analyzer.py`

A lightweight alternative to `audio_analyzer.py`. Uses only `ffmpeg`/`ffprobe` subprocesses instead of Python libraries. Can:
- Extract audio to WAV using ffmpeg
- Query audio properties (duration, sample rate, channels, codec) via ffprobe JSON output
- Detect speech activity using ffmpeg's `silencedetect` and `astats` filters
- **No heavy Python dependencies.** Practical fallback if Whisper/librosa aren't installed.

---

### 5.5 `graph/` — Graph Network Engine

This is the most architecturally sophisticated module. It has two sub-systems:

#### Sub-system A — Instagram Post-Level Graph (uses `reels.duckdb`)

**`graph/graph_analyzer.py` — `InstagramGraphAnalyzer` class**

Builds three types of graph networks from the scraped Instagram metadata:

1. **Content Similarity Graph** (`build_content_similarity_graph()`):
   - Every post is a **node** with attributes: engagement rate, likes, views, hashtag count, mention count, has_location
   - Two posts get an **edge** if their content similarity score > 0.3
   - Similarity score is computed via `calculate_content_similarity()`:
     - Same owner → +0.4
     - Shared hashtags → +0.3 × (shared / max hashtag count)
     - Similar engagement rate (within 5%) → +0.2
     - Similar duration (within 10 seconds) → +0.1
     - Cross-mentions → +0.4
     - Max: 1.0

2. **Hashtag Network** (`build_hashtag_network()`):
   - Every hashtag that appears in ≥ 2 posts becomes a **node** with: post_count, avg_engagement, total_likes
   - Two hashtags get an **edge** if they co-occur in the same post (weight = co-occurrence count)
   - Reveals which hashtag clusters travel together

3. **User Influence Network** (`build_user_influence_network()`):
   - Users are **nodes** with: post count, avg_engagement, influence_score, mention_count
   - **Directed edges** from mentioner to mentioned (influence flows via mentions)
   - `influence_score = posts × avg_engagement × 0.01`

#### Sub-system B — Frame-Level Video Graph (uses `video_graphs.db`)

**`graph/video_graph_builder.py` — `VideoGraphBuilder` class**

Converts a raw `.mp4` file into a detailed mathematical graph where every frame is a node:

**Per-frame features extracted by `extract_frame_features()`:**
- `frame_hash` — Perceptual hash (MD5 of 8×8 grayscale of frame) for similarity detection
- `dominant_colors` — K-means clustering (k=3) in RGB space to find 3 dominant colour triplets
- `brightness` — Mean of the grayscale frame
- `edge_density` — Percentage of pixels flagged as edges by Canny edge detection
- `motion_vectors` — Lucas-Kanade optical flow from previous frame (magnitude + direction)

**Edge types in the graph:**
- `temporal` — Connects every frame to the next frame. Weight = frame similarity score.
- `scene_transition` — Added when a major visual change is detected (cut: similarity < 0.3; fade: brightness change > 50; camera movement: avg motion magnitude > 5.0)

**`graph/video_graph_query_engine.py` — `VideoGraphQueryEngine` class**

Runs complex queries against the SQLite graph databases:

- `query_frame_similarity_patterns(video_id, threshold)` — Find repeated visual patterns (same frame appearing multiple times = loop or reused content)
- `analyze_camera_movement_patterns(video_id)` — Build a camera motion intensity timeline and classify into: static shots, pan sequences, quick cuts, zoom sequences
- `analyze_color_evolution(video_id)` — Track how dominant colours and brightness change across the video timeline
- `detect_color_transitions(color_timeline)` — Find significant palette shifts (colour Jaccard similarity < 0.5)
- `query_scene_structure(video_id)` — Count shots, calculate average shot length and variance, classify pacing as "fast"/"medium"/"slow"

**`graph/instagram_graph_integration.py` — `InstagramVideoGraphIntegrator` class**

The bridge between the Instagram data pipeline and the video graph pipeline:
- Queries DuckDB for reels that haven't been graph-analysed yet
- Downloads the video via `yt-dlp`
- Passes it to `VideoGraphBuilder` to generate the frame graph
- Stores results to `video_graphs.db`
- **Not yet integrated into the Streamlit UI.**

**`graph/video_graph_visualizer.py`**

Standalone visualiser for a graph stored in `demo_video_graphs.db`. Used by Page 5 (Graph Query Engine).

---

## 6. The Streamlit App — Page by Page

### 6.1 `app.py` — Home (Scraper)

**What it does:** The main entry point. A form to scrape Instagram reels.

**UI Elements:**
- Text input: Instagram username
- Number input: max reels (1–200)
- "Run Scraper" button
- Sidebar: "View RAW scrapes" button, "Run Analytics Pipeline" button

**Behind the scenes:**
1. `init_db()` is called on every page load to ensure tables exist.
2. On scrape, calls Apify REST API synchronously.
3. Displays raw JSON response in the UI (useful for debugging).
4. Calls `save_raw_scrape()` — always saves raw data.
5. Optional checkbox: "Also process structured data" — calls `save_structured_scrape()`.
6. Shows quick stats (reel count, comment count, tagged user count).
7. Provides a "Download JSON" button for local export.

**Status:** ✅ Fully working. Requires `APIFY_TOKEN`; if it is missing, the app shows setup guidance instead of breaking.

---

### 6.2 Page 0 — Test

A single-line validation page. Confirms the Streamlit multi-page setup is working. No logic.

---

### 6.3 Data Viewer (`pages/Data_Viewer.py`)

**What it does:** A rich, filterable, searchable table of all scraped reels.

**Data source:** `raw_scrapes` table (reads raw JSON and parses on-the-fly).

**Behind the scenes:**
1. Sidebar filter by profile name, limit (50/100/200/500/1000 rows).
2. `load_raw_data()` queries `raw_scrapes` with optional profile filter.
3. `process_raw_data_for_table()` parses each raw JSON blob and extracts:
   - Short Code, Caption Preview (truncated to 80 chars)
   - Likes (formatted with commas), Views, Play Count, Comments
   - Engagement % = `likes/views × 100`
   - Duration (seconds)
   - Posted date, Location, Owner, Music (artist - song)
   - Hashtag count, Mention count
   - Sponsored flag (✓/✗)
   - Clickable Video URL, Clickable Post URL
   - Scraped timestamp
4. Top summary metrics bar: Total Reels, Avg Likes, Avg Views, Avg Plays, Avg Engagement %.
5. Full-text search across: caption, short code, owner, music, location.
6. Download as CSV or raw JSON.
7. `Video URL` and `Post URL` are rendered as clickable `st.column_config.LinkColumn`.

**Status:** ✅ Fully working. Works from raw data only — does not need the structured pipeline.

---

### 6.4 Page 2 — AI Analytics (`pages/2_🤖_AI_Analytics.py`)

**What it does:** Advanced analytics dashboard with Plotly charts across 4 tabs.

**Data source:** `reels`, `comments`, `tagged_users` tables (the structured tables).

**⚠️ Dependency:** Requires the structured pipeline to have run (the "Also process structured data" checkbox on the scraper page). If the `reels` table is empty, the page shows an error and stops.

**Tab 1 — Performance Analytics:**
- Metrics: Total Likes, Total Views, Avg Engagement Rate, Total Comments
- Bar chart: Top 10 reels by likes
- Scatter: Likes vs Views (bubble size = comments)
- Duration histogram + Duration vs Engagement scatter (with OLS trendline)

**Tab 2 — Timing Insights:**
- Line chart: Avg engagement by hour of day (24h clock)
- Bar chart: Avg engagement by day of week
- "Optimal Posting Recommendations": top 5 posting hours + top 3 posting days by engagement rate

**Tab 3 — Engagement & Comments Analysis:**
- Bar chart: Top 10 most active commenters
- Sentiment overview: avg comment length, positive indicator count
- Sample comment display (recent 5 comments with like count)

**Tab 4 — Tagging Patterns:**
- Bar chart: Most tagged users
- Tagging network: which users appear tagged most frequently on which profiles

**Status:** ✅ Built and functional. ❌ Requires `reels` table populated — only happens if the structured processing checkbox was used.

---

### 6.5 Page 3 — AI Video Analysis (`pages/3_🤖_AI_Video_Analysis.py`)

**What it does:** Analyses downloaded `.mp4` files in the `downloads/` folder.

**Data source:** `.mp4` files in `downloads/` + cross-reference with `raw_scrapes` for metadata.

**How it works:**
1. Scans `downloads/` for `.mp4`, `.mov`, `.avi` files.
2. For each file, extracts `short_code` from the filename.
3. Queries `raw_scrapes` for metadata matching that short code.
4. Computes heuristic-based metrics:
   - `engagement_rate = likes/views × 100`
   - `comments_rate = comments/likes × 100`
   - `play_completion = plays/views × 100`
   - **Content category** via keyword matching in caption + location:
     - Food & Beverage (food, restaurant, bar, drink, ramen, eat)
     - Events & Celebrations (party, celebration, anniversary, event)
     - Lifestyle (style, aesthetic, vibe, mood)
     - Location-based (has locationName but no other category)
     - General Content (fallback)
   - **Virality score (0–10)** via additive heuristic rules:
     - Engagement > 5% → +2
     - Comments rate > 10% → +1
     - Duration 15–60 seconds → +2
     - Caption > 100 chars → +1
     - Has mentions (@) → +1
     - Has location → +1
     - Play completion > 80% → +2

5. Displays: overview metrics, pie chart of content categories, scatter of engagement vs virality (bubble = views, colour = category), top performers by engagement and virality, performance data table with all metrics.

**Status:** ✅ Logic is complete. ❌ Requires `.mp4` files in `downloads/` — currently empty. ❌ Does **not** use OpenAI Vision yet — the GPT-4 analysis in `analysis/ai_video_analyzer.py` is not yet wired in.

---

### 6.6 Page 4 — Graph Network (`pages/4_🕸️_Graph_Network.py`)

**What it does:** Builds and visualises the Instagram content similarity graph in real time.

**Data source:** `raw_scrapes` table (reads all posts).

**Behind the scenes:**
1. `@st.cache_data` wraps `build_content_network()` so the expensive graph computation only runs once per session.
2. Reads all raw posts, extracts hashtags and mentions via regex (`#\w+`, `@\w+`).
3. Adds every post as a NetworkX Graph node with: owner, engagement, likes, views, hashtag_count, mention_count.
4. O(n²) edge loop: compares every pair of posts using `calculate_similarity()` (same owner +0.5, shared hashtags +0.3 normalised, similar engagement +0.2, cross-mentions +0.4). Edges added when similarity > 0.3.
5. Uses Plotly `go.Scatter` with `mode='markers+text'` for the network rendering using a spring layout.

**Status:** ✅ Works on any scraped data. Can be slow on large datasets (O(n²) similarity loop).

---

### 6.7 Page 5 — Graph Query Engine (`pages/5_🔍_Graph_Query_Engine.py`)

**What it does:** Interactive interface for the frame-level video graph queries against `demo_video_graphs.db`.

**Data source:** `demo_video_graphs.db` (SQLite, pre-built demo data).

**Behind the scenes:**
1. `@st.cache_resource` wraps `VideoGraphQueryEngine("demo_video_graphs.db")` for a persistent connection.
2. Provides interactive controls to run:
   - `query_frame_similarity_patterns()` — Find repeated frames
   - `analyze_camera_movement_patterns()` — Camera motion timeline (`plot_motion_timeline()`)
   - `analyze_color_evolution()` — Brightness and colour timeline (`plot_color_evolution()`)
   - `query_scene_structure()` — Shot count, avg shot length, pacing
3. All charts rendered via Plotly with custom colour schemes.

**Status:** ✅ UI is built. Works only with `demo_video_graphs.db`. The real-video pipeline (`instagram_graph_integration.py`) is not yet connected to populate `video_graphs.db` from actual downloaded videos.

---

## 7. External Integrations & APIs Used

| Integration | Purpose | Required? | Key |
|---|---|---|---|
| **Apify** (`apify~instagram-reel-scraper`) | Instagram data scraping | ✅ Required for all data | `APIFY_TOKEN` in `.streamlit/secrets.toml` |
| **OpenAI GPT-4 Vision** | Deep video frame analysis | ❌ Optional (code built, not wired to UI) | `OPENAI_API_KEY` |
| **OpenAI Whisper** (local) | Speech transcription from audio | ❌ Optional (code built, not in UI) | No key needed (local model) |
| **yt-dlp** | Video download from Instagram post URLs | ❌ Optional CLI tool | None |
| **ffmpeg / ffprobe** | Audio extraction from videos | ❌ Optional (for audio analysis) | System install |

---

## 8. Tech Stack & Dependencies

### Core Framework
- **Streamlit** — The entire UI and page routing
- **DuckDB** — Primary analytical database (file-based, no server needed)
- **SQLite** — Secondary database for graph data (`video_graphs.db`)

### Data & Analytics
- **pandas** — DataFrames for all tabular operations
- **plotly** (express + graph_objects) — All charts and network visualisations
- **networkx** — Graph construction, layout algorithms (spring, circular, kamada-kawai), centrality metrics

### Machine Learning
- **scikit-learn** — `RandomForestRegressor` for engagement prediction
- **joblib** — Model serialisation to `.pkl`

### Video & Image Processing
- **OpenCV (cv2)** — Frame extraction, colour analysis, optical flow, edge detection, k-means clustering
- **hashlib** — Perceptual frame hashing
- **numpy** — All matrix/array operations in video processing

### Audio (optional, heavy dependencies)
- **whisper** (OpenAI local model) — Speech transcription
- **moviepy** — Audio extraction from video
- **librosa** — Audio feature extraction (MFCC, spectral centroid, tempo)
- **soundfile** — WAV file I/O

### HTTP & API
- **requests** — Apify API calls and direct video URL downloads

### Secrets
- `.streamlit/secrets.toml` — Stores `APIFY_TOKEN` (and optionally `OPENAI_API_KEY`)

---

## 9. Current State: What Works vs What Doesn't

### ✅ What Works Right Now (Out of the Box)

| Feature | How to Use |
|---|---|
| Scrape Instagram reels via Apify | Home page → enter username → click Run |
| Store raw JSON data | Automatic on every scrape |
| View scraped data in table | Page 1 — Data Viewer |
| Filter/search scraped data | Page 1 — Data Viewer sidebar + search bar |
| Download CSV / raw JSON | Page 1 — Data Viewer download buttons |
| Content similarity graph | Page 4 — Graph Network (auto-builds from scraped data) |
| Frame-level graph query demo | Page 5 — Graph Query Engine (reads demo DB) |
| Multi-page navigation | Streamlit sidebar (auto-generated from `pages/` folder) |

### ⚠️ What Works But Has Preconditions

| Feature | Precondition |
|---|---|
| Page 2 — AI Analytics (Plotly charts) | Must tick "Also process structured data" on scraper page first |
| Page 3 — AI Video Analysis | Must have `.mp4` files in `downloads/` folder |

### ❌ What Is Built But Not Yet Connected to the UI

| Feature | File | What's Missing |
|---|---|---|
| OpenAI GPT-4 Vision analysis | `analysis/ai_video_analyzer.py` | Needs a Streamlit UI, `OPENAI_API_KEY`, and downloaded videos |
| Random Forest engagement predictor | `analysis/engagement_predictor.py` | Needs a Streamlit page to train/run/display it |
| Audio/speech transcription | `analysis/audio_analyzer.py` | Needs ffmpeg + whisper + moviepy installed, and a UI |
| Real video → graph pipeline | `graph/instagram_graph_integration.py` | Needs yt-dlp download step + UI trigger |
| Batch video download | `collection/instagram_downloader.py` | Needs Streamlit UI or to be run as a CLI script |

### ❌ What Is Empty / Incomplete

| Item | Status |
|---|---|
| `pipeline/etl_basic.py` | Completely empty |
| `models/` directory | Empty (likely intended for saved ML models) |
| `data/raw/` and `data/processed/` | Empty folders |
| `ai_analysis/` directory | Empty (intended for AI results output) |

---

## 10. Known Gaps & Issues

### Gap 1 — The Structured Pipeline Is Opt-In and Easy to Miss
The `reels` table (required for Page 2 analytics) is only populated if the user ticks a checkbox. If someone scrapes data multiple times without ticking the box, Page 2 has zero data while Page 1 shows everything correctly. The structured processing should be the default or automatic.

### Gap 2 — Video URLs Expire Quickly
Instagram CDN URLs stored in the `videoUrl` field expire in roughly 24–48 hours. There is no automated re-fetch mechanism. The `collection/url_checker.py` can detect expired URLs but there is no UI-side solution for re-downloading.

### Gap 3 — The ETL Pipeline Is Effectively Non-Existent
`pipeline/etl_basic.py` is empty. The `pipeline/transform.py` is a duplicate of `core/db.py`. There is no scheduled, automated, or batch ETL process. All data loading is manual and on-demand.

### Gap 4 — No Authentication / Multi-User Support
The app is single-user localhost. There is no login, no user separation, and no API rate limiting.

### Gap 5 — Page 2 Analytics Only Work on `reels` Table
Because Page 2 queries the typed `reels` table, its charts only show data that went through the structured pipeline. Meanwhile all raw data is accessible in Page 1. These two data sources can fall out of sync.

### Gap 6 — Graph O(n²) Similarity is Slow at Scale
The content similarity graph in Page 4 computes all pairs of posts. For 200 posts this is 20,000 comparisons. For 1,000 posts it is 500,000. At scale this will be very slow. Needs indexing, approximate nearest-neighbour, or subsampling.

### Gap 7 — The Video Graph Is Demo-Only
Page 5 only reads from `demo_video_graphs.db` which contains hardcoded demo data. The pipeline to process real downloaded videos into frame-level graphs (`instagram_graph_integration.py`) is built but never called from the UI.

### Gap 8 — No Duplicate Detection in `raw_scrapes`
Every scrape run inserts all items again with new `scraped_at` timestamps. If you scrape the same profile twice, duplicate rows accumulate. There is no deduplication logic.

---

## 11. Directory Structure Explained

```
senpai-reel/
│
├── app.py                          # Home page: Instagram scraper UI
├── reels.duckdb                    # Main analytical database (DuckDB)
├── reels.duckdb.wal                # DuckDB write-ahead log
├── video_graphs.db                 # Frame-level video graph DB (SQLite)
├── demo_video_graphs.db            # Demo frame graph data (SQLite)
│
├── .streamlit/                     # Streamlit configuration
│   └── secrets.toml                # API keys (APIFY_TOKEN, etc.) — NOT in git
│
├── pages/                          # Streamlit multi-page app pages
│   ├── Onboarding.py               # First-run client/account setup
│   ├── Pipeline.py                 # One-click scrape-to-embeddings pipeline
│   ├── Settings.py                 # Session-scoped API key setup
│   ├── Costs.py                    # Per-client cost meter
│   ├── Data_Viewer.py              # Main data exploration table
│   ├── Corpus_Explorer.py          # Transcripts and word timestamps
│   ├── Search.py                   # Keyword and semantic search
│   ├── Analytics.py                # Competitor intelligence dashboard
│   └── Content_Studio.py           # Grounded content generation
│
├── core/
│   └── db.py                       # All DuckDB table definitions + CRUD functions
│
├── pipeline/
│   ├── etl_basic.py                # ❌ EMPTY — intended ETL pipeline entry point
│   └── transform.py                # Older version of db.py structured save logic
│
├── analysis/
│   ├── ai_video_analyzer.py        # OpenAI Vision frame analysis (not in UI)
│   ├── engagement_predictor.py     # scikit-learn RandomForest predictor (not in UI)
│   ├── audio_analyzer.py           # Whisper + librosa audio analysis (not in UI)
│   └── simple_audio_analyzer.py    # ffmpeg-based lightweight audio analysis
│
├── graph/
│   ├── graph_analyzer.py           # Instagram post-level graph builder (3 graph types)
│   ├── video_graph_builder.py      # Frame-level video graph builder (OpenCV-based)
│   ├── video_graph_query_engine.py # SQL + NetworkX queries on video graphs
│   ├── video_graph_visualizer.py   # Plotly visualisation for video graphs
│   └── instagram_graph_integration.py  # Bridge: Instagram data → video graph pipeline
│
├── collection/
│   ├── get_urls.py                  # Print post URLs from DB to terminal
│   ├── url_checker.py               # Test if CDN video URLs are still valid
│   ├── video_downloader.py          # Attempt direct video downloads from CDN URLs
│   ├── instagram_downloader.py      # Download via online services or yt-dlp script
│   ├── download_all_videos.py       # Batch download script
│   └── batch_download.sh            # Generated yt-dlp shell script
│
├── downloads/                       # Target folder for downloaded .mp4 files (empty)
├── audio_extracts/                  # Target folder for extracted .wav files (empty)
├── ai_analysis/                     # Target folder for AI analysis outputs (empty)
├── models/                          # Target folder for saved ML models (empty)
├── data/
│   ├── raw/                         # Empty
│   └── processed/                   # Empty
│
├── docs/
│   ├── AI_ANALYSIS_GUIDE.md         # Architecture guide for AI analysis options
│   └── COST_ANALYSIS.md             # Cost comparison: cloud AI vs. free approaches
│
├── tests/
│   └── test.py                      # Basic test script
│
└── venv/                            # Python virtual environment
```

---

## 12. How to Run the App

### Prerequisites
- Python 3.10+ with a virtual environment activated
- An `.streamlit/secrets.toml` file with:
  ```toml
  APIFY_TOKEN = "your_apify_token_here"
  ```

### Start the App
```bash
cd /Users/ariankalantari/senpai-reel
source venv/bin/activate
streamlit run app.py
```

The app runs on **http://localhost:8501**.

### First-Time Usage Flow
1. Open the app → enter an Instagram username → click "Run Scraper"
2. Wait ~10–30 seconds for data to come back
3. Tick "Also process structured data" if you want Page 2 analytics
4. Navigate to Page 1 (Data Viewer) to see your data in table form
5. Navigate to Page 2 for charts (only if you ticked the structured processing)
6. Navigate to Page 4 to see the content similarity graph

### To Enable Video Analysis (Page 3)
1. Get some `.mp4` files into the `downloads/` folder
2. Easiest method: copy Instagram post URLs from Page 1 → run `yt-dlp` manually:
   ```bash
   cd downloads
   yt-dlp "https://www.instagram.com/p/SHORTCODE/"
   ```
3. Return to Page 3 to see analysis

---

*Last updated: April 2026*
