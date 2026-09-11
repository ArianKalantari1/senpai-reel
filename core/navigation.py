from __future__ import annotations


def render_page_link(st_module, page_filename: str, label: str):
    """Render a Streamlit page link from app.py or from direct page tests."""
    if not hasattr(st_module, "page_link"):
        st_module.caption(label)
        return

    for target in (f"pages/{page_filename}", page_filename):
        try:
            st_module.page_link(target, label=label)
            return
        except Exception:
            continue

    st_module.caption(label)
