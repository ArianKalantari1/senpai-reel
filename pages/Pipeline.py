import time

import streamlit as st

from core.client_context import render_client_selector
from core.config import get_secret, missing_secret_message, secret_status
from core.db import init_db
from core.navigation import render_page_link
from core.pipeline import (
    get_pipeline_lock,
    get_pipeline_snapshot,
    release_pipeline_lock,
    run_audio_stage,
    run_download_stage,
    run_embedding_stage,
    run_everything_pending,
    run_extraction_stage,
    run_scrape_stage,
    run_transcription_stage,
    try_start_pipeline_run,
    update_pipeline_lock,
)


st.set_page_config(page_title="Pipeline", page_icon="▶️", layout="wide")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]

st.title("Pipeline")
st.caption(f"Run the full reel pipeline for {active_client['name']} from one place.")


def _settings_link():
    render_page_link(st, "Settings.py", "Open Settings")


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _progress_callback(progress_bar, status_slot, run_started):
    def _callback(event: dict):
        total = max(int(event.get("total") or 0), 1)
        done = min(int(event.get("done") or 0), total)
        ratio = done / total
        elapsed = float(event.get("elapsed_sec") or (time.monotonic() - run_started))
        eta = ""
        if done > 0 and done < total:
            remaining = (elapsed / done) * (total - done)
            eta = f" · ETA {_format_duration(remaining)}"
        progress_bar.progress(ratio)
        item = event.get("item") or ""
        item_text = f" · {item}" if item else ""
        status_slot.info(
            f"{event.get('stage', 'Pipeline')}: {done}/{total}{item_text} "
            f"· elapsed {_format_duration(elapsed)}{eta}"
        )

    return _callback


def _run_stage_with_lock(stage_name: str, stage_fn):
    lock = try_start_pipeline_run(client_id)
    if not lock["acquired"]:
        st.error(lock.get("error") or "Pipeline busy. Wait for the current run to finish, then try again.")
        if lock.get("lock"):
            st.caption(f"Current stage: {lock['lock'].get('stage') or 'Starting'}")
        return None

    run_id = lock["run_id"]
    progress_bar = st.progress(0)
    status_slot = st.empty()
    started = time.monotonic()
    try:
        update_pipeline_lock(run_id, stage_name)
        result = stage_fn(_progress_callback(progress_bar, status_slot, started))
    finally:
        release_pipeline_lock(run_id)
    progress_bar.empty()
    status_slot.empty()
    return result


def _render_result(result):
    if not result:
        return
    if result["status"] == "failed":
        st.error(result["message"])
        if result.get("errors"):
            st.json(result["errors"])
    elif result["status"] == "skipped":
        st.info(result["message"])
    else:
        st.success(result["message"])


def _render_stage_row(number, stage, status, action_label, disabled, action):
    c_num, c_stage, c_status, c_action = st.columns([0.5, 2.2, 4.2, 2])
    c_num.markdown(f"**{number}**")
    c_stage.markdown(f"**{stage}**")
    c_status.write(status)
    if c_action.button(action_label, disabled=disabled, key=f"pipeline_{stage}"):
        _render_result(action())


snapshot = get_pipeline_snapshot(client_id)
lock = get_pipeline_lock()
is_busy = lock is not None

if is_busy:
    st.warning(
        f"Pipeline busy for client `{lock['client_id']}`. "
        f"Current stage: {lock.get('stage') or 'Starting'}."
    )

configured = {row["name"]: row["configured"] for row in secret_status(st)}
missing = [name for name, ok in configured.items() if not ok]
if missing:
    st.info(
        "Some stages need configuration before they can run: "
        + ", ".join(f"`{name}`" for name in missing)
        + "."
    )
    _settings_link()

if snapshot["accounts"] == 0:
    st.info("No competitor accounts yet. Add accounts in Onboarding or Client setup, then run the pipeline.")
    render_page_link(st, "Onboarding.py", "Open Onboarding")

expiry = snapshot["download"]["expiry"]
if expiry["warning_count"]:
    st.warning(
        f"{expiry['warning_count']} pending downloads are approaching Apify CDN expiry. "
        f"The oldest pending media URL is about {expiry['oldest_age_hours']:.1f} hours old; "
        "download soon before the URLs stop working."
    )
    with st.expander("Pending media near expiry"):
        for item in expiry["items"]:
            st.write(
                f"`{item['post_id']}` from @{item['username']} "
                f"was scraped {item['age_hours']:.1f} hours ago."
            )

st.subheader("Stages")

opts = st.expander("Run options", expanded=False)
with opts:
    scrape_max_items = st.number_input("Max reels per account", min_value=1, max_value=200, value=30)
    download_batch = st.number_input("Download batch size", min_value=1, max_value=2000, value=500)
    transcription_batch = st.number_input("Transcription batch size", min_value=1, max_value=2000, value=500)
    extraction_batch = st.number_input("Extraction batch size", min_value=1, max_value=2000, value=500)
    embedding_batch = st.number_input("Embedding batch size", min_value=1, max_value=5000, value=1000)

apify_token = get_secret("APIFY_TOKEN", st)
deepgram_key = get_secret("DEEPGRAM_API_KEY", st)
openai_key = get_secret("OPENAI_API_KEY", st)

_render_stage_row(
    "1",
    "Scrape",
    f"{snapshot['scrape']['accounts']} accounts · {snapshot['scrape']['posts']} reels in this client",
    "Scrape now",
    is_busy,
    lambda: _run_stage_with_lock(
        "Scrape",
        lambda cb: run_scrape_stage(client_id, apify_token, scrape_max_items, cb),
    ),
)
_render_stage_row(
    "2",
    "Download",
    (
        f"{snapshot['download']['done']} done · {snapshot['download']['pending']} pending · "
        f"{snapshot['download']['failed']} failed"
    ),
    "Download pending",
    is_busy,
    lambda: _run_stage_with_lock(
        "Download",
        lambda cb: run_download_stage(client_id, download_batch, cb),
    ),
)
_render_stage_row(
    "2b",
    "Audio",
    f"{snapshot['audio']['done']} extracted · {snapshot['audio']['pending']} pending",
    "Extract audio",
    is_busy,
    lambda: _run_stage_with_lock(
        "Audio",
        lambda cb: run_audio_stage(client_id, cb),
    ),
)
_render_stage_row(
    "3",
    "Transcribe",
    f"{snapshot['transcription']['done']} done · {snapshot['transcription']['pending']} pending",
    "Transcribe pending",
    is_busy,
    lambda: _run_stage_with_lock(
        "Transcribe",
        lambda cb: run_transcription_stage(client_id, deepgram_key, batch_size=transcription_batch, progress_callback=cb),
    ),
)
_render_stage_row(
    "4",
    "Extract units",
    (
        f"{snapshot['extraction']['done']} transcripts extracted · "
        f"{snapshot['extraction']['pending']} pending · {snapshot['extraction']['total_units']} units"
    ),
    "Extract pending",
    is_busy,
    lambda: _run_stage_with_lock(
        "Extract",
        lambda cb: run_extraction_stage(client_id, openai_key, extraction_batch, cb),
    ),
)
_render_stage_row(
    "4b",
    "Embed",
    f"{snapshot['embedding']['done']} embedded · {snapshot['embedding']['pending']} pending",
    "Embed pending",
    is_busy,
    lambda: _run_stage_with_lock(
        "Embed",
        lambda cb: run_embedding_stage(client_id, openai_key, embedding_batch, cb),
    ),
)

st.markdown("---")
if st.button("Run Everything Pending", type="primary", disabled=is_busy):
    progress_bar = st.progress(0)
    status_slot = st.empty()
    started = time.monotonic()
    result = run_everything_pending(
        client_id=client_id,
        apify_token=apify_token,
        deepgram_api_key=deepgram_key,
        openai_api_key=openai_key,
        scrape_max_items=scrape_max_items,
        download_batch_size=download_batch,
        transcription_batch_size=transcription_batch,
        extraction_batch_size=extraction_batch,
        embedding_batch_size=embedding_batch,
        progress_callback=_progress_callback(progress_bar, status_slot, started),
    )
    progress_bar.empty()
    status_slot.empty()

    if result["status"] == "busy":
        st.error(result["message"])
    elif result["status"] == "failed":
        st.error(result["message"])
    else:
        st.success(result["message"])

    if result.get("results"):
        st.table(
            [
                {
                    "Stage": row["stage"],
                    "Status": row["status"],
                    "Done": row["done"],
                    "Failed": row["failed"],
                    "Total": row["total"],
                    "Cost": f"${row['cost_usd']:.4f}",
                    "Message": row["message"],
                }
                for row in result["results"]
            ]
        )

if missing:
    st.caption("Configured stages still run; missing credentials only stop stages that have pending work.")
    for name in missing:
        st.caption(missing_secret_message(name))
