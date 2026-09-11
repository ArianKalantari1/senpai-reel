from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


SESSION_SECRET_PREFIX = "runtime_secret_"


@dataclass(frozen=True)
class RequiredSecret:
    name: str
    label: str
    purpose: str


REQUIRED_SECRETS = {
    "APIFY_TOKEN": RequiredSecret(
        "APIFY_TOKEN",
        "Apify token",
        "scraping Instagram reels",
    ),
    "DEEPGRAM_API_KEY": RequiredSecret(
        "DEEPGRAM_API_KEY",
        "Deepgram API key",
        "transcribing downloaded audio",
    ),
    "OPENAI_API_KEY": RequiredSecret(
        "OPENAI_API_KEY",
        "OpenAI API key",
        "extracting insights, embeddings, semantic search, and content generation",
    ),
}


def session_secret_key(name: str) -> str:
    return f"{SESSION_SECRET_PREFIX}{name}"


def get_secret(name: str, st_module=None) -> str:
    """
    Read a runtime credential from Streamlit session state, st.secrets, or env.

    Session state wins so a first-run user can paste keys in Settings without
    editing local files. Values are intentionally not persisted by this helper.
    """
    if st_module is not None:
        session_value = st_module.session_state.get(session_secret_key(name), "")
        if session_value:
            return str(session_value).strip()

        try:
            value = st_module.secrets.get(name, "")
            if value:
                return str(value).strip()
        except Exception:
            pass

    return os.environ.get(name, "").strip()


def has_secret(name: str, st_module=None) -> bool:
    return bool(get_secret(name, st_module))


def secret_status(st_module=None) -> list[dict]:
    rows = []
    for name, spec in REQUIRED_SECRETS.items():
        configured = has_secret(name, st_module)
        source = "Not configured"
        if configured and st_module is not None and st_module.session_state.get(session_secret_key(name), ""):
            source = "Current session"
        elif configured:
            source = "Secrets or environment"

        rows.append(
            {
                "name": name,
                "label": spec.label,
                "purpose": spec.purpose,
                "configured": configured,
                "source": source,
            }
        )
    return rows


def missing_secret_message(name: str, action: Optional[str] = None) -> str:
    spec = REQUIRED_SECRETS.get(name)
    purpose = spec.purpose if spec else name
    label = spec.label if spec else name
    if action:
        return f"{action} needs a {label}. Add `{name}` in Settings, `.streamlit/secrets.toml`, or the environment."
    return f"{label} is not configured. Add `{name}` in Settings, `.streamlit/secrets.toml`, or the environment to enable {purpose}."
