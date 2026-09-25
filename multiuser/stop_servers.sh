#!/usr/bin/env bash
# Stop the server instances started by start_servers.sh (runs ON the inference pod).
#
#   ./stop_servers.sh configs/llamacpp_2x16.conf   # or just: llamacpp_2x16
#   ./stop_servers.sh --all                        # every llama-server / vllm on this pod
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARG="${1:-}"

kill_tree() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  # Servers were started with setsid, so the pid is its own process-group leader.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
  return 0
}

if [ "$ARG" = "--all" ] || [ -z "$ARG" ]; then
  echo "stopping ALL llama-server / vllm processes owned by $(whoami) on $(hostname)"
  pkill -TERM -u "$(id -u)" -f 'llama-server|vllm.*serve|vllm\.entrypoints' 2>/dev/null
  sleep 5
  pkill -KILL -u "$(id -u)" -f 'llama-server|vllm.*serve|vllm\.entrypoints' 2>/dev/null
else
  CONFIG="$ARG"
  [ -f "$CONFIG" ] || CONFIG="$SCRIPT_DIR/configs/${ARG%.conf}.conf"
  CONFIG_NAME="$(basename "$CONFIG" .conf)"
  RUN_DIR="$SCRIPT_DIR/run/$CONFIG_NAME"

  if [ ! -d "$RUN_DIR" ]; then
    echo "no run dir for '$CONFIG_NAME' ($RUN_DIR); use --all to sweep everything" >&2
    exit 1
  fi
  shopt -s nullglob
  pidfiles=("$RUN_DIR"/inst*.pid)
  shopt -u nullglob
  if [ ${#pidfiles[@]} -eq 0 ]; then
    echo "no pid files in $RUN_DIR; use --all" >&2
    exit 1
  fi
  for pf in "${pidfiles[@]}"; do
    pid="$(cat "$pf" 2>/dev/null)"
    [ -n "$pid" ] || continue
    if kill_tree "$pid"; then
      echo "stopped $(basename "$pf" .pid) (pid $pid)"
    else
      echo "$(basename "$pf" .pid) (pid $pid) was not running"
    fi
    rm -f "$pf"
  done
fi

sleep 2
left="$(pgrep -u "$(id -u)" -f 'llama-server|vllm.*serve' 2>/dev/null | wc -l)"
echo "remaining server processes: $left"
[ "$left" -eq 0 ] && echo "all inference cores released."
exit 0
