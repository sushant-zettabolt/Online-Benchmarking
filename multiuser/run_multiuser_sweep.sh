#!/usr/bin/env bash
# Multi-user Locust sweep (runs ON the support pod, turin-xcovoid0021-pod-5).
#
# For each user count in USERS, ramps that many simulated users up against the
# local HAProxy and holds them for DURATION seconds. Every user issues a
# 1024-token prompt / 128-token completion every 30 seconds (see locustfile.py),
# so a 5-minute run yields ~5 requests per user.
#
#   ./run_multiuser_sweep.sh llamacpp_2x16
#   ./run_multiuser_sweep.sh vllm_1x32 --users "2 4 8 12 16" --duration 300
#
# Per run it saves: Locust CSVs, the raw per-request JSONL, and before/after
# snapshots of HAProxy's stats CSV (which give the true per-instance request
# split, i.e. proof that the weighted round robin did what it claimed).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCUST="$SCRIPT_DIR/.venv/bin/locust"

CONFIG_NAME="${1:-}"
shift || true

USERS="2 4 8 12 16"
DURATION=300
SPAWN_RATE=1
PACING=30
PP=1024
TG=128
FE_PORT=9000
STATS_PORT=9001
COOLDOWN=30
WARMUP=""

while [ $# -gt 0 ]; do
  case "$1" in
    --users)      USERS="$2"; shift 2 ;;
    --duration)   DURATION="$2"; shift 2 ;;
    --spawn-rate) SPAWN_RATE="$2"; shift 2 ;;
    --pacing)     PACING="$2"; shift 2 ;;
    --pp)         PP="$2"; shift 2 ;;
    --tg)         TG="$2"; shift 2 ;;
    --port)       FE_PORT="$2"; shift 2 ;;
    --stats-port) STATS_PORT="$2"; shift 2 ;;
    --cooldown)   COOLDOWN="$2"; shift 2 ;;
    --warmup)     WARMUP="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$CONFIG_NAME" ]; then
  echo "usage: $0 <config-name> [--users \"2 4 8 12 16\"] [--duration 300] ..." >&2
  exit 1
fi
CONFIG_NAME="$(basename "$CONFIG_NAME" .conf)"

MANIFEST="$SCRIPT_DIR/run/$CONFIG_NAME/manifest.json"
if [ ! -f "$MANIFEST" ]; then
  echo "manifest not found: $MANIFEST (start the servers on the inference pod first)" >&2
  exit 1
fi
if [ ! -x "$LOCUST" ]; then
  echo "locust not found at $LOCUST" >&2; exit 1
fi

SRV_HOST="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['host'])")"
SRV_PORTS="$(python3 -c "import json;print(' '.join(str(i['port']) for i in json.load(open('$MANIFEST'))['instances']))")"

# Snapshot each instance's Prometheus endpoint so every run gets a SERVER-SIDE
# view (vLLM: queue/prefill/decode/TTFT/TPOT histograms) to sit beside the
# client-side numbers. Diffed before/after so the stats cover only this run.
snap_prom() {
  local phase="$1" i=0 p
  for p in $SRV_PORTS; do
    python3 "$SCRIPT_DIR/prom_metrics.py" snapshot \
      "http://${SRV_HOST}:${p}/metrics" "$RUN_DIR/prom_${phase}_inst${i}.txt" 2>/dev/null
    i=$((i + 1))
  done
}

BACKEND="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['backend'])")"
MODEL="$(python3 -c "import json;m=json.load(open('$MANIFEST'));print(m['served_model_name'] or 'default')")"
NINST="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['instances_n'])")"
[ -n "$WARMUP" ] || WARMUP=$(( NINST * 2 ))

RESULTS="$SCRIPT_DIR/results/$CONFIG_NAME"
mkdir -p "$RESULTS"
cp "$MANIFEST" "$RESULTS/manifest.json"

if ! (exec 3<>"/dev/tcp/127.0.0.1/$FE_PORT") 2>/dev/null; then
  echo "nothing listening on 127.0.0.1:$FE_PORT -- run ./start_haproxy.sh $CONFIG_NAME first" >&2
  exit 1
fi
exec 3<&- 2>/dev/null

cat <<EOF
=======================================================================
 multi-user sweep : $CONFIG_NAME
 backend          : $BACKEND   ($NINST instance(s))
 load per user    : pp=$PP  tg=$TG  at 1 request / ${PACING}s
 user counts      : $USERS
 duration         : ${DURATION}s per run (spawn rate ${SPAWN_RATE}/s)
 target           : http://127.0.0.1:${FE_PORT}  (HAProxy, weighted round robin)
 results          : $RESULTS
=======================================================================
EOF

# ----------------------------------------------------------------- warmup --
# Touch every instance a couple of times so the first measured request of the
# first run isn't paying one-off allocation/page-fault costs.
echo
echo "warmup: $WARMUP request(s) through the LB..."
for (( w=0; w<WARMUP; w++ )); do
  if [ "$BACKEND" = "llamacpp" ]; then
    body="$(python3 -c "import random;print('{\"prompt\":[%s],\"n_predict\":8,\"temperature\":0,\"cache_prompt\":false,\"stream\":false}' % ','.join(str(random.randint(10,127000)) for _ in range($PP)))")"
    url="http://127.0.0.1:${FE_PORT}/completion"
  else
    body="$(python3 -c "import random;print('{\"model\":\"$MODEL\",\"prompt\":[%s],\"max_tokens\":8,\"temperature\":0}' % ','.join(str(random.randint(10,127000)) for _ in range($PP)))")"
    url="http://127.0.0.1:${FE_PORT}/v1/completions"
  fi
  sb="$(curl -s --max-time 300 -o /dev/null -D - -H 'Content-Type: application/json' \
        -d "$body" "$url" | grep -i '^x-served-by:' | tr -d '\r' | awk '{print $2}')"
  echo "  warmup $((w+1))/$WARMUP -> ${sb:-?}"
done

# ------------------------------------------------------------------ sweep --
for U in $USERS; do
  RUN_DIR="$RESULTS/users_${U}"
  mkdir -p "$RUN_DIR"
  rm -f "$RUN_DIR/samples.jsonl"

  echo
  echo "-----------------------------------------------------------------------"
  echo " run: ${U} users x ${DURATION}s  (expect ~$(( U * DURATION / PACING )) requests)"
  echo "-----------------------------------------------------------------------"

  curl -s --max-time 10 "http://127.0.0.1:${STATS_PORT}/;csv" > "$RUN_DIR/haproxy_before.csv" 2>/dev/null
  snap_prom before

  BENCH_BACKEND="$BACKEND" \
  BENCH_MODEL="$MODEL" \
  BENCH_PP="$PP" \
  BENCH_TG="$TG" \
  BENCH_PACING_S="$PACING" \
  BENCH_PHASE_JITTER=1 \
  BENCH_RUN_LABEL="${CONFIG_NAME}/users_${U}" \
  BENCH_JSONL="$RUN_DIR/samples.jsonl" \
  "$LOCUST" \
      -f "$SCRIPT_DIR/locustfile.py" \
      --headless \
      --host "http://127.0.0.1:${FE_PORT}" \
      --users "$U" \
      --spawn-rate "$SPAWN_RATE" \
      --run-time "${DURATION}s" \
      --csv "$RUN_DIR/locust" \
      --csv-full-history \
      --only-summary \
      --exit-code-on-error 0 \
    2>&1 | tee "$RUN_DIR/locust.log" | tail -20

  snap_prom after
  curl -s --max-time 10 "http://127.0.0.1:${STATS_PORT}/;csv" > "$RUN_DIR/haproxy_after.csv" 2>/dev/null

  n="$(wc -l < "$RUN_DIR/samples.jsonl" 2>/dev/null || echo 0)"
  echo " -> $n samples written to $RUN_DIR/samples.jsonl"

  if [ "$U" != "${USERS##* }" ]; then
    echo " cooldown ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo
echo "sweep complete. consolidating..."
python3 "$SCRIPT_DIR/consolidate_multiuser.py" "$RESULTS"
