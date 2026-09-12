"""Runtime secret access.

This module keeps the secret helpers available under the name used by newer
transcription code while preserving `core.config` as the current source of
truth for existing callers.
"""

from core.config import (
    REQUIRED_SECRETS,
    get_secret,
    has_secret,
    missing_secret_message,
    secret_status,
    session_secret_key,
)

__all__ = [
    "REQUIRED_SECRETS",
    "get_secret",
    "has_secret",
    "missing_secret_message",
    "secret_status",
    "session_secret_key",
]
