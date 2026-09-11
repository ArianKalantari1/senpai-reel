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
if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found. Install Python 3.10+ and re-run."
  exit 1
fi
PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  fail "Python ${PY_VER} found, but 3.10+ is required."
  exit 1
fi
ok "python3 ${PY_VER}"

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
  python3 -m venv .venv
  ok "created .venv"
else
  ok ".venv already exists"
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
