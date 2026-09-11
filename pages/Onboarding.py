import streamlit as st

from core.client_context import render_client_selector
from core.clients import add_client_account, create_client, get_client_accounts
from core.config import secret_status
from core.db import init_db
from core.navigation import render_page_link


st.set_page_config(page_title="Onboarding", page_icon="🧭", layout="wide")

init_db()
active_client = render_client_selector()
client_id = active_client["client_id"]

st.title("Onboarding")
st.caption("Set up a client, choose competitors, run the pipeline, then generate the first content.")


def _suggested_handles(niche: str) -> list[str]:
    text = (niche or "").lower()
    if any(word in text for word in ["job", "career", "resume", "recruit"]):
        return ["resumeworded", "careersaus", "jobsearchau", "seekjobs", "thecareersacademy"]
    if any(word in text for word in ["fitness", "gym", "health", "wellness"]):
        return ["kayla_itsines", "chontelduncan", "keepitcleaner", "centrfit"]
    if any(word in text for word in ["cafe", "restaurant", "hospitality", "food"]):
        return ["broadsheet_melb", "timeoutmelbourne", "melbournefoodiehub", "urbanlistmelb"]
    if any(word in text for word in ["clinic", "dental", "medical", "skin"]):
        return ["thegoodgp", "skin.software", "qandaskin", "healthdirectau"]
    return []


def _page_link(target: str, label: str):
    render_page_link(st, target.removeprefix("pages/"), label)


accounts = get_client_accounts(client_id)
configured_count = sum(1 for row in secret_status(st) if row["configured"])

step1, step2, step3, step4 = st.columns(4)
step1.metric("1. Client", "Ready" if active_client else "Needed")
step2.metric("2. Competitors", len(accounts))
step3.metric("3. API Keys", f"{configured_count}/3")
step4.metric("4. Content", "Ready" if accounts else "Waiting")

st.markdown("---")

with st.expander("1. Name Your Client And Niche", expanded=not active_client):
    with st.form("onboarding_create_client", clear_on_submit=True):
        name = st.text_input("Client name")
        niche = st.text_input("Niche", placeholder="e.g., careers for new migrants in Australia")
        target = st.text_input("Target audience")
        voice = st.text_area("Brand voice notes", height=80)
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Create Client", type="primary")

    if submitted:
        try:
            new_client_id = create_client(
                name,
                niche,
                target_audience=target,
                brand_voice_notes=voice,
                notes=notes,
            )
            st.session_state["active_client_id"] = new_client_id
            st.success("Client created.")
            st.rerun()
        except Exception as exc:
            st.error(f"Client setup failed: {exc}")

with st.expander("2. Add Competitor Accounts", expanded=len(accounts) == 0):
    suggestions = _suggested_handles(active_client.get("niche", ""))
    if suggestions:
        st.write("Suggested starting points for this niche:")
        st.write(", ".join(f"@{handle}" for handle in suggestions))

    default_text = "\n".join(suggestions)
    handles_text = st.text_area(
        "Paste Instagram handles",
        value=default_text if not accounts else "",
        height=140,
        placeholder="@account\nhttps://www.instagram.com/another_account/",
    )
    max_items = st.number_input("Max reels per account", min_value=1, max_value=200, value=30)
    if st.button("Add Accounts", type="primary"):
        handles = [line.strip() for line in handles_text.splitlines() if line.strip()]
        if not handles:
            st.warning("Paste at least one Instagram handle.")
        else:
            added = 0
            for handle in handles:
                try:
                    add_client_account(client_id, handle, max_items=max_items)
                    added += 1
                except Exception as exc:
                    st.warning(f"Could not add `{handle}`: {exc}")
            st.success(f"Added {added} accounts.")
            st.rerun()

    if accounts:
        st.write("Current accounts:")
        st.table(
            [
                {
                    "Handle": f"@{row['instagram_handle']}",
                    "Category": row["category"],
                    "Max reels": row["max_items"],
                }
                for row in accounts
            ]
        )

with st.expander("3. Run The Pipeline", expanded=len(accounts) > 0):
    if len(accounts) == 0:
        st.info("Add competitor accounts first. The pipeline needs accounts to scrape.")
    else:
        st.write("Run scraping, download, audio, transcription, extraction, and embeddings from one page.")
        _page_link("pages/Pipeline.py", "Open Pipeline")

with st.expander("4. Generate First Content", expanded=False):
    st.write("After the pipeline has extracted knowledge units, use the Content Studio to draft captions, hooks, or scripts.")
    _page_link("pages/Content_Studio.py", "Open Content Studio")
