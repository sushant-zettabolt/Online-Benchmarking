#!/usr/bin/env bash
# One-off driver for the 20-value pp list (includes pp=3, unlike sweep_online.sh's
# hardcoded 19-value list). Does not modify sweep_online.sh itself.
set -uo pipefail
BACKEND="$1"; URL="$2"; MODEL="$3"; RESULTS_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"
for pp in 1 2 3 4 8 16 32 48 64 96 128 256 384 512 768 1024 1536 2048 3072 4096; do
  out="$RESULTS_DIR/online_${BACKEND}_${pp}.json"
  if [ -s "$out" ]; then echo "skip pp=$pp (exists)"; continue; fi
  echo "=== pp=$pp backend=$BACKEND ==="
  python3 "$SCRIPT_DIR/bench_client.py" --backend "$BACKEND" --url "$URL" --model "$MODEL" \
    --pp "$pp" --tg 1 --num-iters 3 --warmup-iters 1 --output-json "$out" 2>&1 | tail -5
done
echo "done: $RESULTS_DIR"
