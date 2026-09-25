#!/usr/bin/env bash
# Runs ONE overnight benchmark job end-to-end, synchronously (this script IS
# meant to be launched detached/backgrounded by dispatcher.sh, not the other
# way around). Executes on the devbox (has kubectl); every pod-local action
# goes through `kubectl exec` into either the inference pod (server + local
# resource monitor) or the control pod (HAProxy + Locust for multiuser, or the
# bench_client.py driver for ppsweep -- both need network route to the
# inference pod's overlay IP, which only pod-resident processes have).
#
# Usage:
#   ./run_job.sh <mode> <layout> <instances> <cores_per_instance> <backend> \
#                <variant> <quant> <model_name> <infer_pod> <control_pod> \
#                <worker_slot> <job_id>
set -uo pipefail

MODE="$1"; LAYOUT="$2"; INSTANCES="$3"; CORES_PER_INSTANCE="$4"
BACKEND="$5"; VARIANT="$6"; QUANT="$7"; MODEL_NAME="$8"
INFER_POD="$9"; CONTROL_POD="${10}"; WORKER_SLOT="${11}"; JOB_ID="${12}"

# Overridable for smoke-testing (real overnight runs use the defaults):
#   SWEEP_USERS="2" SWEEP_DURATION=60 PP_LIST="1 128 1024" BASELINE_S=5 ./run_job.sh ...
SWEEP_USERS="${SWEEP_USERS:-2 4 8 12 16}"
SWEEP_DURATION="${SWEEP_DURATION:-300}"
PP_LIST_OVERRIDE="${PP_LIST:-}"
BASELINE_S="${BASELINE_S:-20}"

OB_DIR="/proj/rdi/staff/sacsharm/online_bench"
MU_DIR="$OB_DIR/multiuser"
OVERNIGHT_DIR="$OB_DIR/overnight"
TMP_SCRIPTS="$OVERNIGHT_DIR/tmp_scripts"
RESULTS_DIR="$OVERNIGHT_DIR/results/$JOB_ID"
LEDGER="$OVERNIGHT_DIR/RUN_LEDGER.jsonl"
KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml
export KUBECONFIG
NS=zendnn

mkdir -p "$RESULTS_DIR" "$TMP_SCRIPTS"
STARTED_AT="$(date -Iseconds)"

log() { echo "[$(date '+%H:%M:%S')] [$JOB_ID] $*"; }

ledger_line() {  # $1=status $2=note
  python3 - "$JOB_ID" "$MODE" "$LAYOUT" "$BACKEND" "$VARIANT" "$QUANT" "$INFER_POD" \
    "$CONTROL_POD" "$STARTED_AT" "$1" "$2" "$RESULTS_DIR" <<'PY'
import json, sys, datetime
(job_id, mode, layout, backend, variant, quant, infer_pod, control_pod,
 started, status, note, results_dir) = sys.argv[1:13]
rec = {"job_id": job_id, "mode": mode, "layout": layout, "backend": backend,
       "variant": variant, "quant": quant, "infer_pod": infer_pod,
       "control_pod": control_pod, "started_at": started,
       "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "status": status, "note": note, "results_dir": results_dir}
print(json.dumps(rec))
PY
}

fail() {
  log "FAILED: $*"
  ledger_line "FAILED" "$*" >> "$LEDGER"
  exit 1
}

# Run a script FILE (not an inline -c string, to sidestep the && / & chaining
# quoting gotcha entirely) on a pod as the sacsharm user. Blocks until it exits.
pod_run() {  # $1=pod $2=script_path
  local pod="$1" script="$2"
  timeout 1800 kubectl exec "$pod" -n "$NS" -- bash -c "
    u=\$(stat -Lc '%u' /proj/rdi/staff/sacsharm); g=\$(stat -Lc '%g' /proj/rdi/staff/sacsharm)
    setpriv --reuid \"\$u\" --regid \"\$g\" --clear-groups env HOME=/proj/rdi/staff/sacsharm bash '$script'
  "
}

numa_of() { kubectl get pod "$1" -n "$NS" -o jsonpath='{.metadata.labels.numa}'; }

# ---------------------------------------------------------------- model lookup --
read -r MODEL_PATH SERVED_NAME < <(awk -v n="$MODEL_NAME" -v b="$BACKEND" -v q="$QUANT" \
  '!/^#/ && NF>=5 && $1==n && $2==b && $3==q {print $4, $5; found=1} END{if(!found) exit 1}' \
  "$OVERNIGHT_DIR/models.conf") || fail "no models.conf row for $MODEL_NAME/$BACKEND/$QUANT"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
DST_MODEL_PATH="/tmp/models/$MODEL_BASENAME"

# ------------------------------------------------------------------- binaries --
case "$BACKEND-$VARIANT" in
  llamacpp-zendnn)   LLAMA_SERVER=/proj/rdi/staff/sacsharm/llama.cpp/build_zendnn/bin/llama-server
                      LLAMA_BATCHED=/proj/rdi/staff/sacsharm/llama.cpp/build_zendnn/bin/llama-batched-bench ;;
  llamacpp-nozendnn) LLAMA_SERVER=/proj/rdi/staff/sacsharm/llama.cpp/build_release/bin/llama-server
                      LLAMA_BATCHED=/proj/rdi/staff/sacsharm/llama.cpp/build_release/bin/llama-batched-bench ;;
  vllm-zentorch)      VLLM_BIN=/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm
                       VLLM_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so ;;
  vllm-nozentorch)    VLLM_BIN=/proj/rdi/staff/sacsharm/vllm/.venv_nozentorch/bin/vllm
                       VLLM_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so ;;
  *) fail "unknown backend/variant combo $BACKEND-$VARIANT" ;;
esac
LLAMA_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5

# ---------------------------------------------------------------- pod topology --
NUMA="$(numa_of "$INFER_POD")" || fail "could not read numa label for $INFER_POD"
[ -n "$NUMA" ] || fail "empty numa label for $INFER_POD"
CORE_START=$(( NUMA * 32 ))
CORE_END=$(( CORE_START + 31 ))
CRANGE="${CORE_START}-${CORE_END}"
log "infer pod $INFER_POD -> numa=$NUMA cores=$CRANGE"

# --------------------------------------------------------------- stage model --
STAGE_SCRIPT="$TMP_SCRIPTS/stage_${JOB_ID}.sh"
cat > "$STAGE_SCRIPT" <<EOF
set -uo pipefail
mkdir -p /tmp/models
if [ -e "$DST_MODEL_PATH" ]; then
  if [ -f "$MODEL_PATH" ]; then
    ssz=\$(stat -c %s "$MODEL_PATH" 2>/dev/null || echo -1)
    dsz=\$(stat -c %s "$DST_MODEL_PATH" 2>/dev/null || echo -2)
    [ "\$ssz" = "\$dsz" ] && { echo "already staged"; exit 0; }
  else
    [ -d "$DST_MODEL_PATH" ] && [ -n "\$(ls -A "$DST_MODEL_PATH" 2>/dev/null)" ] && { echo "already staged (dir)"; exit 0; }
  fi
fi
echo "staging $MODEL_PATH -> $DST_MODEL_PATH"
if [ -d "$MODEL_PATH" ]; then
  rm -rf "${DST_MODEL_PATH}.part"; cp -r "$MODEL_PATH" "${DST_MODEL_PATH}.part" && mv -f "${DST_MODEL_PATH}.part" "$DST_MODEL_PATH"
else
  cp -f "$MODEL_PATH" "${DST_MODEL_PATH}.part" && mv -f "${DST_MODEL_PATH}.part" "$DST_MODEL_PATH"
fi
EOF
log "staging model on $INFER_POD (skip if already present)..."
pod_run "$INFER_POD" "$STAGE_SCRIPT" || fail "model staging failed"

# ------------------------------------------------------------- start monitor --
PHASEFILE="/tmp/phase_${JOB_ID}.txt"
PIDFILE="/tmp/pid_${JOB_ID}.txt"
MONITOR_CSV="$RESULTS_DIR/resource_usage.csv"
MONITOR_SCRIPT="$TMP_SCRIPTS/monitor_${JOB_ID}.sh"
cat > "$MONITOR_SCRIPT" <<EOF
echo before > "$PHASEFILE"
cd "$OVERNIGHT_DIR"
: > "$PIDFILE"
setsid nohup ./monitor_resources.sh "$CRANGE" 2 "$MONITOR_CSV" "$PHASEFILE" "$PIDFILE" > /tmp/monlog_${JOB_ID}.log 2>&1 < /dev/null &
disown
EOF
log "starting resource monitor (before phase)..."
pod_run "$INFER_POD" "$MONITOR_SCRIPT" || fail "monitor launch failed"
sleep "$BASELINE_S"

set_phase() { # $1=phase
  local s="$TMP_SCRIPTS/phase_${JOB_ID}_$1.sh"
  echo "echo $1 > '$PHASEFILE'" > "$s"
  pod_run "$INFER_POD" "$s"
}

set_track_pid() {  # $1=pid -- RSS tracking is instance-0-only (a representative
                    # sample), not summed across every instance of a multi-instance layout.
  local s="$TMP_SCRIPTS/setpid_${JOB_ID}.sh"
  echo "echo $1 > '$PIDFILE'" > "$s"
  pod_run "$INFER_POD" "$s"
}

stop_monitor() {
  local s="$TMP_SCRIPTS/stopmon_${JOB_ID}.sh"
  echo "pkill -u \$(id -u) -f 'monitor_resources[.]sh.*${JOB_ID}' 2>/dev/null; pkill -u \$(id -u) -f \"$MONITOR_CSV\" 2>/dev/null; true" > "$s"
  pod_run "$INFER_POD" "$s"
}

CONFIG_NAME="overnight_${JOB_ID}"
CONFIG_PATH="$MU_DIR/configs/${CONFIG_NAME}.conf"
STATUS="OK"; NOTE=""; MANIFEST=""

if [ "$MODE" = "batched" ]; then
  # -------------------------------------------------------- offline batched --
  # No server/manifest for offline batched-bench -- write a synthetic
  # single-instance manifest so analyze_resources.py can still group cores.
  MANIFEST="$RESULTS_DIR/manifest.json"
  printf '{"instances": [{"idx": 0, "cores": "%s"}]}\n' "$CRANGE" > "$MANIFEST"
  set_phase during
  BATCH_OUT="$RESULTS_DIR/batched_bench.md"
  RUN_SCRIPT="$TMP_SCRIPTS/run_${JOB_ID}.sh"
  if [ "$BACKEND" = "llamacpp" ]; then
    cat > "$RUN_SCRIPT" <<EOF
export LD_PRELOAD="$LLAMA_LD_PRELOAD"
export GOMP_CPU_AFFINITY="$CRANGE"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_LRU_CACHE_CAPACITY=1024
numactl --physcpubind=$CRANGE --membind=$NUMA \
  "$LLAMA_BATCHED" -m "$DST_MODEL_PATH" \
  -c 70000 -b 2048 -ub 512 -t 32 -tb 32 \
  -Cr $CRANGE -Crb $CRANGE --cpu-strict 1 --cpu-strict-batch 1 \
  -fa on -npp 128,512,1024,2048,4096 -ntg 128 -npl 1,2,4,8,12,16 \
  --output-format md > "$BATCH_OUT" 2>"$RESULTS_DIR/batched_bench.log"
EOF
  else
    cat > "$RUN_SCRIPT" <<EOF
export LD_PRELOAD="$VLLM_LD_PRELOAD"
export VLLM_CPU_OMP_THREADS_BIND="$CRANGE"
export OMP_NUM_THREADS=32
export VLLM_LOGGING_LEVEL=WARNING
numactl --physcpubind=$CRANGE --membind=$NUMA bash -c '
  for npl in 1 2 4 8 12 16; do
    for pp in 128 512 1024 2048 4096; do
      echo "=== npl=\$npl pp=\$pp ===" >> "$RESULTS_DIR/batched_bench.log"
      "$VLLM_BIN" bench latency --model "$DST_MODEL_PATH" --dtype bfloat16 \
        --max-model-len 8192 --batch-size "\$npl" --input-len "\$pp" --output-len 128 \
        --num-iters 3 --num-iters-warmup 1 \
        --output-json "$RESULTS_DIR/latency_npl\${npl}_pp\${pp}.json" >> "$RESULTS_DIR/batched_bench.log" 2>&1
    done
  done
'
EOF
  fi
  log "running offline batched-bench ($BACKEND/$VARIANT)..."
  pod_run "$INFER_POD" "$RUN_SCRIPT" || { STATUS="FAILED"; NOTE="batched-bench run failed"; }

elif [ "$MODE" = "ppsweep" ]; then
  # ---------------------------------------------- single-request pp-sweep gap-fill --
  RUN_MODE_IS_SERVER=1
elif [ "$MODE" = "multiuser" ]; then
  RUN_MODE_IS_SERVER=1
else
  fail "unknown mode $MODE"
fi

if [ "${RUN_MODE_IS_SERVER:-0}" = "1" ]; then
  PARALLEL=16; SLOT_CTX=2048
  [ "$MODE" = "ppsweep" ] && { PARALLEL=1; SLOT_CTX=$([ "$BACKEND" = "llamacpp" ] && echo 8192 || echo 4096); }
  KV_GB=$(( 16 / INSTANCES )); [ "$KV_GB" -lt 1 ] && KV_GB=1
  PORT=$([ "$BACKEND" = "llamacpp" ] && echo 8080 || echo 8000)

  {
    echo "BACKEND=$BACKEND"
    echo "INSTANCES=$INSTANCES"
    echo "CORES_PER_INSTANCE=$CORES_PER_INSTANCE"
    echo "CORE_START=$CORE_START"
    echo "NUMA_NODE=$NUMA"
    echo "BASE_PORT=$PORT"
    echo "PARALLEL=$PARALLEL"
    echo "SLOT_CTX=$SLOT_CTX"
    echo 'WEIGHTS=""'
    if [ "$BACKEND" = "llamacpp" ]; then
      echo "LLAMA_SERVER=$LLAMA_SERVER"
      echo "LLAMA_MODEL=$DST_MODEL_PATH"
      echo "LOAD_MODE=mmap"
    else
      echo "VLLM_BIN=$VLLM_BIN"
      echo "VLLM_MODEL=$DST_MODEL_PATH"
      echo "SERVED_MODEL_NAME=${SERVED_NAME:-llama31-8b}"
      echo "KVCACHE_SPACE_GB=$KV_GB"
      echo "ENFORCE_EAGER=0"
    fi
  } > "$CONFIG_PATH"
  log "generated config $CONFIG_PATH"

  set_phase during

  START_SCRIPT="$TMP_SCRIPTS/start_${JOB_ID}.sh"
  cat > "$START_SCRIPT" <<EOF
cd "$MU_DIR"
./stop_servers.sh "$CONFIG_NAME" >/dev/null 2>&1
./start_servers.sh "${CONFIG_NAME}.conf" --wait-secs 900
EOF
  log "starting servers on $INFER_POD ($CONFIG_NAME)..."
  if ! pod_run "$INFER_POD" "$START_SCRIPT"; then
    STATUS="FAILED"; NOTE="server failed to become healthy"
  else
    MANIFEST="$MU_DIR/run/$CONFIG_NAME/manifest.json"
    INFER_IP="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['host'])" 2>/dev/null)"
    INST0_PID="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['instances'][0]['pid'])" 2>/dev/null)"
    [ -n "$INST0_PID" ] && set_track_pid "$INST0_PID"

    if [ "$MODE" = "multiuser" ]; then
      FE_PORT=$(( 9000 + WORKER_SLOT * 10 ))
      STATS_PORT=$(( 9001 + WORKER_SLOT * 10 ))
      SWEEP_SCRIPT="$TMP_SCRIPTS/sweep_${JOB_ID}.sh"
      cat > "$SWEEP_SCRIPT" <<EOF
cd "$MU_DIR"
./stop_haproxy.sh "$CONFIG_NAME" >/dev/null 2>&1
./start_haproxy.sh "$CONFIG_NAME" --port $FE_PORT --stats-port $STATS_PORT
./run_multiuser_sweep.sh "$CONFIG_NAME" --users "$SWEEP_USERS" --duration "$SWEEP_DURATION" \
  --port $FE_PORT --stats-port $STATS_PORT
./stop_haproxy.sh "$CONFIG_NAME" >/dev/null 2>&1
EOF
      log "running multiuser sweep via control pod $CONTROL_POD (fe=$FE_PORT stats=$STATS_PORT)..."
      pod_run "$CONTROL_POD" "$SWEEP_SCRIPT" || { STATUS="FAILED"; NOTE="multiuser sweep failed"; }
      mkdir -p "$RESULTS_DIR"
      cp -r "$MU_DIR/results/$CONFIG_NAME" "$RESULTS_DIR/multiuser_sweep" 2>/dev/null
    else
      # ppsweep gap-fill
      PP_LIST="${PP_LIST_OVERRIDE:-1 2 3 4 8 16 32 48 64 96 128 256 384 512 768 1024 1536 2048 3072 4096}"
      BENCH_MODEL="${SERVED_NAME:-default}"
      [ "$BACKEND" = "llamacpp" ] && BENCH_MODEL="default"
      SWEEP_SCRIPT="$TMP_SCRIPTS/sweep_${JOB_ID}.sh"
      cat > "$SWEEP_SCRIPT" <<EOF
cd "$OB_DIR"
mkdir -p "$RESULTS_DIR/ppsweep"
for pp in $PP_LIST; do
  out="$RESULTS_DIR/ppsweep/online_${BACKEND}_\${pp}.json"
  [ -s "\$out" ] && continue
  python3 bench_client.py --backend "$BACKEND" --url "http://${INFER_IP}:${PORT}" \
    --model "$BENCH_MODEL" --pp "\$pp" --tg 1 --num-iters 3 --warmup-iters 1 \
    --output-json "\$out"
done
EOF
      log "running pp-sweep gap-fill via control pod $CONTROL_POD..."
      pod_run "$CONTROL_POD" "$SWEEP_SCRIPT" || { STATUS="FAILED"; NOTE="pp-sweep failed"; }
    fi

    STOP_SCRIPT="$TMP_SCRIPTS/stop_${JOB_ID}.sh"
    cat > "$STOP_SCRIPT" <<EOF
cd "$MU_DIR"
./stop_servers.sh "$CONFIG_NAME" >/dev/null 2>&1
true
EOF
    pod_run "$INFER_POD" "$STOP_SCRIPT"
  fi
fi

set_phase after
sleep "$BASELINE_S"
stop_monitor
python3 "$OVERNIGHT_DIR/analyze_resources.py" "$MONITOR_CSV" \
  "${MANIFEST:-$MU_DIR/run/$CONFIG_NAME/manifest.json}" > "$RESULTS_DIR/coreusage_report.txt" 2>&1 || true

ledger_line "$STATUS" "$NOTE" >> "$LEDGER"
log "done, status=$STATUS ${NOTE:+note=$NOTE}"
[ "$STATUS" = "OK" ]
