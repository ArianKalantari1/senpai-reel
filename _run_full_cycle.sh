#!/bin/zsh
# Full cycle: kill Streamlit → run transcriptions → restart Streamlit
set -e

cd /Users/ariankalantari/senpai-reel
source venv/bin/activate

echo "=== Killing Streamlit ==="
pkill -f "streamlit" 2>/dev/null || true
sleep 2

echo "=== Starting batch transcription ==="
python _run_transcriptions.py 2>&1
TRANSCRIBE_EXIT=$?

echo ""
echo "=== Transcription finished (exit code: $TRANSCRIBE_EXIT) ==="
echo "=== Restarting Streamlit ==="
nohup streamlit run app.py --server.headless true > /dev/null 2>&1 &
sleep 2
echo "=== Streamlit restarted (PID: $!) ==="
echo "=== Done! ==="
