"""
Full analysis pipeline — orchestrates the complete workflow:
  Scrape → Download → Audio Extract → Transcribe + Extract + Embed → Export CSV.

Designed to be called from the UI with a progress callback.
Each stage is fault-tolerant: if one account/video fails, the rest continue.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import streamlit as st

from collection.scraper import scrape_and_store
from processing.download import download_pending_posts
from processing.audio import extract_audio_for_downloaded_posts
from processing.transcription_queue import run_transcription_queue
from analysis.export import (
    build_posts_report_df,
    build_units_report_df,
    export_to_csv,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    # Scrape
    accounts_attempted: int = 0
    accounts_succeeded: int = 0
    reels_found: int = 0
    reels_new: int = 0
    scrape_errors: list = field(default_factory=list)

    # Download
    videos_downloaded: int = 0
    videos_failed: int = 0

    # Audio
    audio_extracted: int = 0
    audio_failed: int = 0

    # Pipeline (transcribe + extract + embed)
    transcribed: int = 0
    transcribe_failed: int = 0
    pipeline_cost_usd: float = 0.0
    pipeline_errors: list = field(default_factory=list)

    # Export
    csv_paths: list = field(default_factory=list)


def run_full_analysis(
    accounts: List[str],
    max_reels: int = 100,
    download_batch_size: int = 500,
    pipeline_batch_size: int = 200,
    progress_cb: Optional[Callable] = None,
) -> PipelineResult:
    """
    Run the complete analysis pipeline.

    Args:
        accounts:            list of Instagram usernames to scrape
        max_reels:           max reels to request per account (Apify will return
                             whatever exists if fewer)
        download_batch_size: how many videos to download per batch
        pipeline_batch_size: how many to run through STT→Extract→Embed
        progress_cb:         callable(stage, message, pct) for live UI updates
                             stage: "scrape" | "download" | "audio" | "pipeline" | "export"
    """
    result = PipelineResult()
    apify_token = st.secrets.get("APIFY_TOKEN", "")

    def _emit(stage: str, msg: str, pct: float = 0.0):
        if progress_cb:
            progress_cb(stage, msg, pct)

    # ── Stage 1: Scrape ──────────────────────────────────────────────────────
    _emit("scrape", f"Scraping {len(accounts)} accounts (up to {max_reels} reels each)…", 0.0)
    result.accounts_attempted = len(accounts)

    for i, username in enumerate(accounts):
        _emit("scrape", f"Scraping @{username}  ({i+1}/{len(accounts)})…",
              (i / len(accounts)))
        try:
            r = scrape_and_store(username, apify_token, max_reels)
            if r["status"] == "done":
                result.accounts_succeeded += 1
                result.reels_found += r["reels_found"]
                result.reels_new += r["reels_new"]
            else:
                result.scrape_errors.append({"account": username, "error": r["error"]})
        except Exception as e:
            result.scrape_errors.append({"account": username, "error": str(e)})

    _emit("scrape", f"Scraping complete — {result.reels_found} reels found", 1.0)

    # ── Stage 2: Download ────────────────────────────────────────────────────
    _emit("download", "Downloading videos…", 0.0)

    def _dl_progress(done, total, post_id):
        _emit("download", f"Downloading {done}/{total}…", done / total if total else 1.0)

    dl = download_pending_posts(batch_size=download_batch_size, progress_callback=_dl_progress)
    result.videos_downloaded = dl["done"]
    result.videos_failed = dl["failed"]

    _emit("download", f"Downloads complete — {dl['done']} videos", 1.0)

    # ── Stage 3: Audio extraction ────────────────────────────────────────────
    _emit("audio", "Extracting audio from videos…", 0.0)

    def _audio_progress(done, total, post_id):
        _emit("audio", f"Extracting audio {done}/{total}…", done / total if total else 1.0)

    audio = extract_audio_for_downloaded_posts(progress_callback=_audio_progress)
    result.audio_extracted = audio["done"]
    result.audio_failed = audio["failed"]

    _emit("audio", f"Audio extraction complete — {audio['done']} files", 1.0)

    # ── Stage 4: Transcribe + Extract + Embed ────────────────────────────────
    _emit("pipeline", "Running AI pipeline (transcribe → extract → embed)…", 0.0)

    assemblyai_key = st.secrets.get("ASSEMBLYAI_API_KEY", "")

    def _pipe_progress(done, total, post_id, error):
        _emit("pipeline", f"Pipeline {done}/{total}…", done / total if total else 1.0)

    pipe = run_transcription_queue(
        api_key=assemblyai_key,
        provider="assemblyai",
        batch_size=pipeline_batch_size,
        progress_callback=_pipe_progress,
    )
    result.transcribed = pipe["done"]
    result.transcribe_failed = pipe["failed"]
    result.pipeline_cost_usd = pipe.get("total_cost_usd", 0.0)
    result.pipeline_errors = pipe.get("errors", [])

    _emit("pipeline", f"Pipeline complete — {pipe['done']} processed", 1.0)

    # ── Stage 5: Export CSVs ─────────────────────────────────────────────────
    _emit("export", "Building CSV reports…", 0.0)

    posts_df = build_posts_report_df("All")
    posts_path = export_to_csv(posts_df, "competitor_posts_report")
    result.csv_paths.append(posts_path)

    _emit("export", "Posts report ready, building units report…", 0.5)

    units_df = build_units_report_df("All")
    units_path = export_to_csv(units_df, "competitor_units_report")
    result.csv_paths.append(units_path)

    _emit("export", "Export complete!", 1.0)

    return result
