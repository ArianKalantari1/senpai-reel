import streamlit as st

from core.config import REQUIRED_SECRETS, get_secret, secret_status, session_secret_key
from core.db import init_db


st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

init_db()

st.title("Settings")
st.caption("Add API keys for this browser session, or keep using `.streamlit/secrets.toml` and environment variables.")

st.subheader("API Keys")
rows = secret_status(st)
for row in rows:
    label = "Configured" if row["configured"] else "Missing"
    st.write(f"**{row['label']}** · {label} · {row['source']}")
    st.caption(row["purpose"])

with st.form("runtime_keys_form"):
    st.write("Session keys")
    values = {}
    for name, spec in REQUIRED_SECRETS.items():
        current = get_secret(name, st)
        placeholder = "Already configured" if current else name
        values[name] = st.text_input(
            spec.label,
            value="",
            placeholder=placeholder,
            type="password",
            help=f"Used for {spec.purpose}. Stored only in Streamlit session state.",
        )

    save_clicked = st.form_submit_button("Use Keys For This Session", type="primary")

if save_clicked:
    for name, value in values.items():
        clean = value.strip()
        if clean:
            st.session_state[session_secret_key(name)] = clean
    st.success("Session keys updated.")
    st.rerun()

if st.button("Clear Session Keys"):
    for name in REQUIRED_SECRETS:
        st.session_state.pop(session_secret_key(name), None)
    st.success("Session keys cleared.")
    st.rerun()

st.info(
    "Session keys are not written to disk. For a permanent local setup, add the same names to "
    "`.streamlit/secrets.toml` or export them as environment variables before launching Streamlit."
)
