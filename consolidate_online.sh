#!/usr/bin/env bash
# Consolidates online-serving sweep result JSONs (sweep_online.sh output --
# online_<backend>_<pp>.json) into a single TSV, same pp/prompt_tok/ttft_s/tok_s
# style as the offline consolidate_results.sh table. Works for either
# backend's results dir (vllm or llamacpp) since both write mean_ttft_s/tok_s
# into their JSON already.
set -uo pipefail

RESULTS_DIR="${1:-/proj/rdi/staff/sacsharm/online_bench/results/vllm}"
OUT_TSV="${2:-$RESULTS_DIR/summary.tsv}"

shopt -s nullglob
files=("$RESULTS_DIR"/online_*_[0-9]*.json)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
  echo "no result files found matching $RESULTS_DIR/online_*_<pp>.json" >&2
  exit 1
fi

{
  printf "pp\tprompt_tok\tttft_s\ttok_s\n"
  for f in "${files[@]}"; do
    python3 -c "
import json
d = json.load(open('$f'))
pp = d['pp']
iters = d.get('iters') or []
prompt_tok = None
if iters:
    prompt_tok = iters[0].get('prompt_tokens') or iters[0].get('prompt_n')
if prompt_tok is None:
    prompt_tok = pp
ttft = d.get('mean_ttft_s')
tok_s = d.get('tok_s')
if ttft is None or tok_s is None:
    raise SystemExit(f'missing mean_ttft_s/tok_s in $f')
print(f'{pp}\t{prompt_tok}\t{ttft:.3f}\t{tok_s:.1f}')
"
  done | sort -t$'\t' -k1,1n
} | tee "$OUT_TSV"

echo "written to $OUT_TSV" >&2
