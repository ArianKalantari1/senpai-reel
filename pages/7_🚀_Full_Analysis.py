"""
🚀 Full Analysis — one-click competitor analysis pipeline.

Chains: Batch Scrape → Download → Audio Extract → Transcribe+Extract+Embed → Export CSV.
The user configures accounts + max reels, clicks Run, and walks away.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from collection.account_list import COMPETITOR_ACCOUNTS, DEFAULT_MAX_ITEMS
from core.db import init_db

init_db()

st.set_page_config(page_title="Full Analysis", page_icon="🚀", layout="wide")
st.title("🚀 Full Analysis Pipeline")
st.caption(
    "Scrape → Download → Transcribe → Extract → Embed → Export CSV.  "
    "Configure below, click Run, and walk away."
)

# ── Settings ─────────────────────────────────────────────────────────────────
st.subheader("⚙️ Settings")

col_left, col_right = st.columns([3, 1])

with col_left:
    default_handles = "\n".join(COMPETITOR_ACCOUNTS)
    handles_input = st.text_area(
        "Accounts to scrape (one per line)",
        value=default_handles,
        height=300,
        help="All competitor accounts pre-filled. Add or remove as needed — no @ needed.",
    )

with col_right:
    max_reels = st.number_input(
        "Max reels per account",
        min_value=1,
        max_value=500,
        value=100,
        help=(
            "If an account has fewer reels than this, "
            "it'll just grab what's available and move on."
        ),
    )
    pipeline_batch = st.number_input(
        "Pipeline batch size",
        min_value=1,
        max_value=500,
        value=200,
        help="How many reels to process through STT → Extract → Embed per run.",
    )

handles = [h.strip().lstrip("@") for h in handles_input.splitlines() if h.strip()]
st.caption(f"**{len(handles)}** accounts configured  ·  up to **{max_reels}** reels each")

# ── API key check ────────────────────────────────────────────────────────────
apify_ok = bool(st.secrets.get("APIFY_TOKEN", ""))
assemblyai_ok = bool(st.secrets.get("ASSEMBLYAI_API_KEY", ""))
cerebras_ok = bool(st.secrets.get("CEREBRAS_API_KEY", ""))
voyage_ok = bool(st.secrets.get("VOYAGE_API_KEY", ""))

missing_keys = []
if not apify_ok:
    missing_keys.append("APIFY_TOKEN")
if not assemblyai_ok:
    missing_keys.append("ASSEMBLYAI_API_KEY")
if not cerebras_ok:
    missing_keys.append("CEREBRAS_API_KEY")
if not voyage_ok:
    missing_keys.append("VOYAGE_API_KEY")

if missing_keys:
    st.warning(
        f"⚠️ Missing API keys in `.streamlit/secrets.toml`: "
        f"{', '.join(f'`{k}`' for k in missing_keys)}"
    )

# ── Run button ───────────────────────────────────────────────────────────────
st.markdown("---")

can_run = bool(handles) and not missing_keys

if st.button(
    "🚀 Run Full Analysis",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
):
    from pipeline.full_pipeline import run_full_analysis

    # Progress UI
    stage_names = {
        "scrape": "1/5 — Scraping",
        "download": "2/5 — Downloading",
        "audio": "3/5 — Audio Extraction",
        "pipeline": "4/5 — AI Pipeline",
        "export": "5/5 — Exporting",
    }

    stage_header = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    stage_header.subheader("⏳ Running…")

    # Stage metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    scrape_metric = m1.empty()
    download_metric = m2.empty()
    audio_metric = m3.empty()
    pipeline_metric = m4.empty()
    export_metric = m5.empty()

    _current_stage = [None]

    def on_progress(stage, message, pct):
        stage_label = stage_names.get(stage, stage)
        if _current_stage[0] != stage:
            _current_stage[0] = stage
            # Update stage-level metric badges
            if stage == "scrape":
                scrape_metric.metric("Scrape", "▶ running")
            elif stage == "download":
                scrape_metric.metric("Scrape", "✅")
                download_metric.metric("Download", "▶ running")
            elif stage == "audio":
                download_metric.metric("Download", "✅")
                audio_metric.metric("Audio", "▶ running")
            elif stage == "pipeline":
                audio_metric.metric("Audio", "✅")
                pipeline_metric.metric("Pipeline", "▶ running")
            elif stage == "export":
                pipeline_metric.metric("Pipeline", "✅")
                export_metric.metric("Export", "▶ running")

        stage_header.subheader(f"⏳ {stage_label}")
        status_text.caption(message)
        # Overall progress: each stage is 20% of the bar
        stage_offsets = {"scrape": 0, "download": 0.2, "audio": 0.4, "pipeline": 0.6, "export": 0.8}
        overall = stage_offsets.get(stage, 0) + pct * 0.2
        progress_bar.progress(min(overall, 1.0))

    result = run_full_analysis(
        accounts=handles,
        max_reels=int(max_reels),
        download_batch_size=500,
        pipeline_batch_size=int(pipeline_batch),
        progress_cb=on_progress,
    )

    # Final state
    progress_bar.progress(1.0)
    stage_header.subheader("✅ Analysis Complete!")
    status_text.empty()
    export_metric.metric("Export", "✅")

    # ── Summary ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Results Summary")

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Accounts Scraped", f"{result.accounts_succeeded}/{result.accounts_attempted}")
    r2.metric("Reels Found", f"{result.reels_found} ({result.reels_new} new)")
    r3.metric("Videos Downloaded", result.videos_downloaded)
    r4.metric("Transcribed", result.transcribed)
    r5.metric("Pipeline Cost", f"${result.pipeline_cost_usd:.4f}")

    # Errors
    all_errors = result.scrape_errors + result.pipeline_errors
    if all_errors:
        with st.expander(f"⚠️ {len(all_errors)} error(s)", expanded=False):
            st.json(all_errors)

    # ── Download buttons ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Download Reports")

    for csv_path in result.csv_paths:
        p = Path(csv_path)
        if p.exists():
            with open(p, "r") as f:
                csv_data = f.read()
            st.download_button(
                f"📥 {p.name}",
                data=csv_data,
                file_name=p.name,
                mime="text/csv",
                key=f"dl_{p.stem}",
            )

    # Store result in session state so it persists on the page
    st.session_state["last_analysis_result"] = {
        "accounts_succeeded": result.accounts_succeeded,
        "accounts_attempted": result.accounts_attempted,
        "reels_found": result.reels_found,
        "reels_new": result.reels_new,
        "videos_downloaded": result.videos_downloaded,
        "transcribed": result.transcribed,
        "pipeline_cost_usd": result.pipeline_cost_usd,
        "csv_paths": result.csv_paths,
        "timestamp": datetime.now().isoformat(),
    }

# ── Previous Exports ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Previous Exports")

exports_dir = Path("data/exports")
if exports_dir.exists():
    csvs = sorted(exports_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        for csv_file in csvs[:20]:
            size_kb = csv_file.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            col_a, col_b = st.columns([4, 1])
            col_a.caption(f"**{csv_file.name}**  ·  {size_kb:.0f} KB  ·  {mod_time}")
            with open(csv_file, "r") as f:
                col_b.download_button(
                    "📥",
                    data=f.read(),
                    file_name=csv_file.name,
                    mime="text/csv",
                    key=f"prev_{csv_file.stem}",
                )
    else:
        st.info("No exports yet. Run an analysis above to generate CSV reports.")
else:
    st.info("No exports yet. Run an analysis above to generate CSV reports.")
