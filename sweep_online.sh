#!/usr/bin/env bash
# Online-serving pp sweep (tg pinned low, effectively "tg 0" -- neither
# server API accepts max_tokens/n_predict=0, so 1 is the practical minimum,
# same convention used for the offline pp sweeps) against an already-running
# vllm or llama.cpp server, over the network (server on one pod, this script
# run from another -- see bench_client.py for the single-request/warmup
# methodology and cache-defeat details).
#
# Usage:
#   ./sweep_online.sh --backend vllm      --url http://192.168.7.40:8000 --model /path/to/model
#   ./sweep_online.sh --backend llamacpp  --url http://192.168.7.40:8080
#
# Resumable: re-running skips any pp whose output json already exists.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND=""
URL=""
MODEL="default"
RESULTS_DIR=""
TG=1
NUM_ITERS=3
WARMUP_ITERS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --tg) TG="$2"; shift 2 ;;
    --num-iters) NUM_ITERS="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$BACKEND" ] || [ -z "$URL" ]; then
  echo "usage: $0 --backend vllm|llamacpp --url http://HOST:PORT [--model NAME] [--results-dir DIR] [--tg N] [--num-iters N] [--warmup-iters N]" >&2
  exit 1
fi

if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "llamacpp" ]; then
  echo "--backend must be vllm or llamacpp" >&2
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/$BACKEND}"
mkdir -p "$RESULTS_DIR"

PROMPT_SIZES=(1 2 4 8 16 32 48 64 96 128 256 384 512 768 1024 1536 2048 3072 4096)

for pp in "${PROMPT_SIZES[@]}"; do
  out_json="$RESULTS_DIR/online_${BACKEND}_${pp}.json"
  out_log="$RESULTS_DIR/online_${BACKEND}_${pp}.log"

  if [ -s "$out_json" ]; then
    echo "skip pp=$pp (exists: $out_json)"
    continue
  fi

  echo "========================================"
  echo "Running online pp=$pp tg=$TG backend=$BACKEND"
  echo "========================================"

  (
    python3 "$SCRIPT_DIR/bench_client.py" \
      --backend "$BACKEND" \
      --url "$URL" \
      --model "$MODEL" \
      --pp "$pp" \
      --tg "$TG" \
      --num-iters "$NUM_ITERS" \
      --warmup-iters "$WARMUP_ITERS" \
      --output-json "$out_json"
  ) 2>&1 | tee "$out_log"

  if [ -s "$out_json" ]; then
    echo "Completed pp=$pp"
  else
    echo "FAILED pp=$pp (see $out_log)"
  fi
  echo
done

echo "done. results in $RESULTS_DIR"
