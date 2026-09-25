#!/usr/bin/env bash
# Stop the HAProxy started by start_haproxy.sh (runs ON the support pod).
#   ./stop_haproxy.sh llamacpp_2x16
#   ./stop_haproxy.sh --all
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARG="${1:-}"

if [ "$ARG" = "--all" ] || [ -z "$ARG" ]; then
  pkill -TERM -u "$(id -u)" -x haproxy 2>/dev/null && echo "sent TERM to all haproxy owned by $(whoami)"
  sleep 2
  pkill -KILL -u "$(id -u)" -x haproxy 2>/dev/null
else
  PIDFILE="$SCRIPT_DIR/run/$(basename "$ARG" .conf)/lb/haproxy.pid"
  pid="$(cat "$PIDFILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null; sleep 2
    kill -KILL "$pid" 2>/dev/null
    echo "stopped haproxy (pid $pid)"
  else
    echo "no running haproxy for $ARG"
  fi
  rm -f "$PIDFILE"
fi
echo "remaining haproxy: $(pgrep -u "$(id -u)" -x haproxy 2>/dev/null | wc -l)"
exit 0
