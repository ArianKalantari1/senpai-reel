from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


SESSION_SECRET_PREFIX = "runtime_secret_"


@dataclass(frozen=True)
class RequiredSecret:
    name: str
    label: str
    purpose: str
    required: bool = True


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
    "ASSEMBLYAI_API_KEY": RequiredSecret(
        "ASSEMBLYAI_API_KEY",
        "AssemblyAI API key",
        "transcribing downloaded audio with AssemblyAI",
        required=False,
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

    env_value = os.environ.get(name, "").strip()
    if env_value:
        return env_value

    return _secret_from_toml(name)


# README tells people to put their keys in .streamlit/secrets.toml, and the app
# reads them from there via st.secrets. The CLI tools call get_secret() without
# a Streamlit module, so before this fallback existed they saw only the
# environment — which meant the documented place to put a key was a place half
# the codebase could not read it. `reextract.py` refused to run for exactly
# that reason on a machine where the key was correctly configured.
#
# Environment still wins. An export is a deliberate act for one command; the
# file is ambient configuration. This is purely a fallback, so nothing that
# worked before changes behaviour.
# Streamlit reads secrets from TWO files, and a key configured in either one
# works in the app. The first version of this fallback checked only the
# project-local file, so a key living in the user-level file was invisible to
# the CLI while the app found it happily — the same failure as before, one
# directory over. If the CLI does not look everywhere the app looks, it will
# keep reporting a missing credential that is not missing.
#
# Project-local is listed second because it is the more specific of the two and
# should win a conflict. That ordering matches Streamlit's own documented
# behaviour, but it is a claim about another project's semantics rather than
# something verified here — if the two files ever disagree in practice, check
# it before trusting this comment.
# Both are module-level so a test can point them at a tmp_path. Reading them
# inside the function rather than capturing at import time is what makes that
# patching work.
_GLOBAL_SECRETS_TOML = Path.home() / ".streamlit" / "secrets.toml"
_SECRETS_TOML = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"


def _secrets_toml_paths() -> list[Path]:
    return [_GLOBAL_SECRETS_TOML, _SECRETS_TOML]


def _load_toml(path: Path) -> dict:
    """Parse one TOML file. Never raises — unreadable means "nothing here"."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return {}
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        return {}


def _secret_from_toml(name: str, path: Path | None = None) -> str:
    """Read one key from Streamlit's secrets files. Never raises.

    A missing file, unreadable file, malformed TOML or absent key all mean the
    same thing to the caller: not configured. Returns "" so `has_secret` stays
    honest rather than a crash masquerading as a missing credential.

    Pass `path` to read exactly one file. Otherwise both of Streamlit's
    locations are consulted, later entries winning.
    """
    paths = [path] if path is not None else _secrets_toml_paths()
    found = ""
    for candidate in paths:
        value = _load_toml(candidate).get(name, "")
        if value:
            found = str(value).strip()
    return found


def has_secret(name: str, st_module=None) -> bool:
    return bool(get_secret(name, st_module))


def secret_status(st_module=None) -> list[dict]:
    rows = []
    for name, spec in REQUIRED_SECRETS.items():
        configured = has_secret(name, st_module)
        status = "configured"
        source = "Secrets or environment"
        if configured and st_module is not None and st_module.session_state.get(session_secret_key(name), ""):
            source = "Current session"
        elif not configured and spec.required:
            status = "missing-required"
            source = "Missing required key"
        elif not configured:
            status = "not-in-use"
            source = "Optional provider not configured"

        rows.append(
            {
                "name": name,
                "label": spec.label,
                "purpose": spec.purpose,
                "required": spec.required,
                "configured": configured,
                "status": status,
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
