from __future__ import annotations

from core.clients import (
    DEFAULT_CLIENT_ID,
    add_client_account,
    create_client,
    delete_client,
    get_client,
    get_client_accounts,
    list_clients,
)


def render_client_selector() -> dict:
    """Render the global Streamlit client selector and return the active client."""
    import streamlit as st

    clients = list_clients()
    if not clients:
        raise RuntimeError("No clients exist after database initialization")

    clients_by_id = {row["client_id"]: row for row in clients}
    ids = list(clients_by_id.keys())
    current_id = st.session_state.get("active_client_id", DEFAULT_CLIENT_ID)
    if current_id not in clients_by_id:
        current_id = ids[0]

    def _label(client_id: str) -> str:
        client = clients_by_id[client_id]
        niche = client.get("niche") or "No niche"
        return f"{client['name']} ({niche})"

    with st.sidebar:
        st.header("Client")
        selected_id = st.selectbox(
            "Active client",
            ids,
            index=ids.index(current_id),
            format_func=_label,
            key="active_client_select",
        )
        st.session_state["active_client_id"] = selected_id
        active = clients_by_id[selected_id]

        account_count = len(get_client_accounts(selected_id))
        st.caption(f"{account_count} competitor accounts")

        with st.expander("Client setup"):
            with st.form("create_client_form", clear_on_submit=True):
                new_name = st.text_input("Client name")
                new_niche = st.text_input("Niche")
                new_target = st.text_input("Target audience")
                new_voice = st.text_area("Brand voice notes", height=80)
                new_notes = st.text_area("Notes", height=80)
                accounts_text = st.text_area(
                    "Competitor handles",
                    placeholder="one handle per line",
                    height=100,
                )
                submitted = st.form_submit_button("Create client")

            if submitted:
                try:
                    handles = [line for line in accounts_text.splitlines() if line.strip()]
                    client_id = create_client(
                        new_name,
                        new_niche,
                        notes=new_notes,
                        brand_voice_notes=new_voice,
                        target_audience=new_target,
                        accounts=handles,
                    )
                    st.session_state["active_client_id"] = client_id
                    st.success("Client created")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            with st.form("add_client_account_form", clear_on_submit=True):
                handle = st.text_input("Add competitor handle")
                category = st.text_input("Category", value="Competitor")
                max_items = st.number_input("Max reels", min_value=1, max_value=200, value=30)
                add_clicked = st.form_submit_button("Add account")

            if add_clicked:
                try:
                    add_client_account(selected_id, handle, category, max_items)
                    st.success("Account added")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            if selected_id != DEFAULT_CLIENT_ID:
                confirm = st.checkbox("Confirm delete active client", key=f"confirm_delete_{selected_id}")
                if st.button("Delete active client", disabled=not confirm):
                    try:
                        delete_client(selected_id)
                        st.session_state["active_client_id"] = DEFAULT_CLIENT_ID
                        st.success("Client deleted")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    refreshed = get_client(st.session_state["active_client_id"])
    return refreshed or active
