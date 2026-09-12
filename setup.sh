#!/usr/bin/env bash
# One-command setup. Safe to re-run.
#
#   ./setup.sh
#
set -euo pipefail

GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; RED=$'\033[0;31m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
ok()   { echo "${GREEN}✓${OFF} $1"; }
warn() { echo "${YELLOW}!${OFF} $1"; }
fail() { echo "${RED}✗${OFF} $1"; }

echo "${BOLD}Senpai Reel setup${OFF}"
echo

# ── Python ────────────────────────────────────────────────────────────────────
# The floor is 3.10 for syntax. The CEILING matters just as much: several
# dependencies (pyarrow via streamlit, duckdb) ship prebuilt wheels only up to
# a given Python, and pip silently falls back to compiling from source on
# anything newer. That fails minutes later with "command 'cmake' failed",
# which looks like a broken toolchain rather than the wrong interpreter.
#
# So rather than accept whatever `python3` happens to be, look for one that
# actually works. Override with SENPAI_PYTHON=/path/to/python3.12 if needed.
PY_MIN_MINOR=10
PY_MAX_MINOR=13   # highest MINOR known to have wheels for every dependency

_py_ok() {
  # usable = exists, and 3.PY_MIN_MINOR <= version <= 3.PY_MAX_MINOR
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c "import sys
lo, hi = ${PY_MIN_MINOR}, ${PY_MAX_MINOR}
sys.exit(0 if sys.version_info[0] == 3 and lo <= sys.version_info[1] <= hi else 1)" 2>/dev/null
}

PYTHON=""
for candidate in "${SENPAI_PYTHON:-}" python3.12 python3.11 python3.13 python3.10 python3; do
  [ -n "$candidate" ] || continue
  if _py_ok "$candidate"; then PYTHON="$candidate"; break; fi
done

if [ -z "$PYTHON" ]; then
  fail "No suitable Python found. Need 3.${PY_MIN_MINOR} to 3.${PY_MAX_MINOR}."
  if command -v python3 >/dev/null 2>&1; then
    echo "    Your default python3 is $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')."
    echo "    Newer is not better here — dependencies have no wheels for it yet,"
    echo "    so pip tries to compile them from source and fails on cmake."
  fi
  if [[ "$OSTYPE" == darwin* ]]; then
    echo "    Install one with:  brew install python@3.12"
  else
    echo "    Install one with:  sudo apt install python3.12 python3.12-venv"
  fi
  echo "    Or point at an existing one:  SENPAI_PYTHON=/path/to/python3.12 ./setup.sh"
  exit 1
fi
PY_VER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ok "python ${PY_VER} (${PYTHON})"

# ── ffmpeg (system dependency, not pip) ───────────────────────────────────────
# Needed to turn downloaded video into the 16kHz mono wav Deepgram expects.
# Without it, scraping and downloading work but transcription silently has
# nothing to transcribe.
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
else
  fail "ffmpeg not found — audio extraction will fail."
  if [[ "$OSTYPE" == darwin* ]]; then
    echo "    Install it with:  brew install ffmpeg"
  else
    echo "    Install it with:  sudo apt install ffmpeg"
  fi
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  ok "created .venv with python ${PY_VER}"
else
  # An existing .venv may have been built with a different interpreter — the
  # check above says nothing about it. Verify the one we are about to install
  # into, since that is what pip will actually use.
  VENV_VER=$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")
  if .venv/bin/python -c "import sys
lo, hi = ${PY_MIN_MINOR}, ${PY_MAX_MINOR}
sys.exit(0 if sys.version_info[0] == 3 and lo <= sys.version_info[1] <= hi else 1)" 2>/dev/null; then
    ok ".venv already exists (python ${VENV_VER})"
  else
    fail "Existing .venv uses python ${VENV_VER}, outside 3.${PY_MIN_MINOR}-3.${PY_MAX_MINOR}."
    echo "    Installing into it would fail while building wheels from source."
    echo "    Rebuild it:  rm -rf .venv && ./setup.sh"
    exit 1
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python3 -m pip install --quiet --upgrade pip
echo "  installing dependencies…"
python3 -m pip install --quiet -r requirements.txt
ok "dependencies installed"

# ── Secrets ───────────────────────────────────────────────────────────────────
mkdir -p .streamlit
if [ ! -f .streamlit/secrets.toml ]; then
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
  warn "created .streamlit/secrets.toml from the example — add your API keys"
  NEEDS_KEYS=1
else
  if grep -q "YOUR_TOKEN_HERE\|YOUR_KEY_HERE" .streamlit/secrets.toml; then
    warn ".streamlit/secrets.toml still has placeholder values"
    NEEDS_KEYS=1
  else
    ok "secrets.toml present"
    NEEDS_KEYS=0
  fi
fi

# ── Database ──────────────────────────────────────────────────────────────────
python3 -c "from core.db import init_db; init_db()"
ok "database ready (reels.duckdb)"

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "  running tests…"
if python3 -m pytest tests/ -q --ignore=tests/test.py >/dev/null 2>&1; then
  ok "test suite passes"
else
  warn "some tests failed — the app will probably still run, but worth a look:"
  echo "    .venv/bin/python -m pytest tests/ -q --ignore=tests/test.py"
fi

echo
echo "${BOLD}Ready.${OFF}"
if [ "${NEEDS_KEYS:-0}" = "1" ]; then
  echo
  echo "Before scraping, add your keys to ${BOLD}.streamlit/secrets.toml${OFF}:"
  echo "  APIFY_TOKEN       required to scrape"
  echo "  DEEPGRAM_API_KEY  required to transcribe"
  echo "  OPENAI_API_KEY    required to extract and generate"
fi
echo
echo "Start the app:"
echo "  ${BOLD}source .venv/bin/activate && streamlit run app.py${OFF}"
