#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  dev.sh — Mega Millions Engine Dev Launcher
#  Starts the local web server and opens the dashboard.
#
#  Usage:
#    ./dev.sh           # start server on port 8765 (default)
#    ./dev.sh 9000      # start on a custom port
#    ./dev.sh --stop    # kill any running server
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8765}"
PID_FILE="$SCRIPT_DIR/.dev-server.pid"

# ── --stop flag ───────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"
      rm -f "$PID_FILE"
      echo "  ✓ Server (PID $PID) stopped."
    else
      echo "  Server was not running (stale PID file removed)."
      rm -f "$PID_FILE"
    fi
  else
    echo "  No running server found."
  fi
  exit 0
fi

# ── Kill any previous server on same port ────────────────────
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

# ── Banner ────────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   🎰  Mega Millions Engine — Dev Start   ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── Start HTTP server ─────────────────────────────────────────
cd "$SCRIPT_DIR"
python3.13 -m http.server "$PORT" \
  --bind 127.0.0.1 \
  > /tmp/mm-server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Brief pause to confirm the server started
sleep 0.4
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "  ✗ Server failed to start. Check /tmp/mm-server.log"
  rm -f "$PID_FILE"
  exit 1
fi

URL="http://localhost:$PORT"

echo "  ✓  Web server running"
echo "  ✓  Serving from: $SCRIPT_DIR"
echo "  ✓  PID: $SERVER_PID  (stop with: ./dev.sh --stop)"
echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │  Dashboard → $URL"
echo "  └─────────────────────────────────────────┘"
echo ""

# ── Open browser ──────────────────────────────────────────────
if command -v open &>/dev/null; then
  open "$URL"           # macOS
elif command -v xdg-open &>/dev/null; then
  xdg-open "$URL"       # Linux
fi

echo "  Press Ctrl+C to stop the server."
echo ""

# ── Keep alive and relay server logs ─────────────────────────
trap "kill $SERVER_PID 2>/dev/null; rm -f '$PID_FILE'; echo '  Server stopped.'; exit 0" INT TERM

wait "$SERVER_PID"
