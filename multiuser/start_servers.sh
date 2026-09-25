#!/usr/bin/env bash
# Multi-instance LLM server launcher (runs ON the inference pod, turin-xcovoid0014-pod-6).
#
# Reads an instance-layout config, carves the pod's physical core pool into N
# consecutive slices, and starts one fully-detached server instance per slice
# (numactl --physcpubind=<slice> --membind=<node>). Instance i listens on
# BASE_PORT+i.
#
#   INSTANCES=1 CORES_PER_INSTANCE=32 -> inst0: 192-223            port 8080
#   INSTANCES=2 CORES_PER_INSTANCE=16 -> inst0: 192-207 :8080
#                                        inst1: 208-223 :8081
#   INSTANCES=4 CORES_PER_INSTANCE=8  -> 192-199 / 200-207 / 208-215 / 216-223
#
# The launcher SELF-TERMINATES once every instance answers /health: the servers
# are re-parented to init via setsid, so after this script returns no part of
# the benchmark harness holds any of the inference cores. HAProxy and Locust
# run on a separate pod (turin-xcovoid0021-pod-5) precisely so that all 32
# cores here stay dedicated to inference.
#
# Writes run/<config>/manifest.json describing host/ports/cores/weights; the
# support pod reads that file over the shared /proj filesystem to build its
# HAProxy backend.
#
# Usage:
#   ./start_servers.sh configs/llamacpp_2x16.conf
#   ./start_servers.sh configs/vllm_4x8.conf --wait-secs 900
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
WAIT_SECS=900
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --wait-secs) WAIT_SECS="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)           CONFIG="$1"; shift ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "usage: $0 <config-file> [--wait-secs N] [--force]" >&2
  echo "available configs:" >&2
  ls "$SCRIPT_DIR/configs"/*.conf 2>/dev/null | sed 's/^/  /' >&2
  exit 1
fi
[ -f "$CONFIG" ] || CONFIG="$SCRIPT_DIR/configs/$CONFIG"
if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2; exit 1
fi

# ---------------------------------------------------------------- defaults --
BACKEND=llamacpp
INSTANCES=1
CORES_PER_INSTANCE=32
CORE_START=192
NUMA_NODE=6
BASE_PORT=8080
PARALLEL=8
SLOT_CTX=2048
WEIGHTS=""

# llama.cpp
LLAMA_SERVER=/proj/rdi/staff/sacsharm/llama.cpp/build_zendnn/bin/llama-server
LLAMA_MODEL=/tmp/models/Llama-3.1-8B-Instruct-BF16.gguf
LOAD_MODE=mmap
LD_PRELOAD_LIBS=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5
ZENDNNL_MATMUL_ALGO=1
ZENDNNL_LRU_CACHE_CAPACITY=1024

# vLLM
VLLM_BIN=/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm
VLLM_MODEL=/tmp/models/Llama-3.1-8B-Instruct
VLLM_LD_PRELOAD_LIBS=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so
SERVED_MODEL_NAME=llama31-8b
KVCACHE_SPACE_GB=16
MAX_NUM_BATCHED_TOKENS=4096
# 1 = pass --enforce-eager (the established command's setting). On CPU this
# disables torch.compile/inductor, so it can materially change decode speed --
# configs exist for both settings so the effect can be reported rather than
# assumed.
ENFORCE_EAGER=1

# shellcheck disable=SC1090
source "$CONFIG"

CONFIG_NAME="$(basename "$CONFIG" .conf)"
RUN_DIR="$SCRIPT_DIR/run/$CONFIG_NAME"
CORE_END=$(( CORE_START + INSTANCES * CORES_PER_INSTANCE - 1 ))

# ------------------------------------------------------------- validation --
POOL_LIST="$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo "")"
echo "config          : $CONFIG_NAME"
echo "backend         : $BACKEND"
echo "instances       : $INSTANCES x ${CORES_PER_INSTANCE} cores"
echo "core span       : ${CORE_START}-${CORE_END}   (pod pool: ${POOL_LIST:-unknown})"
echo "numa node       : $NUMA_NODE"
echo "ports           : ${BASE_PORT}-$(( BASE_PORT + INSTANCES - 1 ))"
echo

if [ "$CORE_END" -ge $(( CORE_START + 256 )) ]; then
  echo "refusing: core span looks impossible" >&2; exit 1
fi

# Refuse to oversubscribe the pod's cpuset (each instance must get disjoint cores).
if [ -n "$POOL_LIST" ]; then
  pool_n=$(python3 - "$POOL_LIST" <<'PY'
import sys
n=0
for part in sys.argv[1].split(','):
    if '-' in part:
        a,b=part.split('-'); n += int(b)-int(a)+1
    elif part.strip():
        n += 1
print(n)
PY
)
  need=$(( INSTANCES * CORES_PER_INSTANCE ))
  if [ "$need" -gt "$pool_n" ]; then
    echo "refusing: layout needs $need cores but pod only owns $pool_n" >&2; exit 1
  fi
fi

# Refuse to start on top of a live deployment unless --force.
for (( i=0; i<INSTANCES; i++ )); do
  p=$(( BASE_PORT + i ))
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
    exec 3<&- 2>/dev/null
    if [ "$FORCE" -eq 1 ]; then
      echo "port $p busy -- --force given, stopping existing instances first"
      "$SCRIPT_DIR/stop_servers.sh" "$CONFIG" >/dev/null 2>&1
      sleep 5
      break
    fi
    echo "refusing: port $p already in use. Run ./stop_servers.sh $CONFIG_NAME first (or pass --force)." >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/*.pid "$RUN_DIR"/manifest.json

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$HOST_IP" ] || HOST_IP=127.0.0.1

# --------------------------------------------------------------- weights ---
# Default WRR weight = core count, so a heterogeneous layout (e.g. 16+8+8)
# receives traffic in proportion to the compute each instance actually owns.
read -r -a WEIGHT_ARR <<< "$WEIGHTS"
if [ "${#WEIGHT_ARR[@]}" -ne "$INSTANCES" ]; then
  WEIGHT_ARR=()
  for (( i=0; i<INSTANCES; i++ )); do WEIGHT_ARR+=( "$CORES_PER_INSTANCE" ); done
fi

# ---------------------------------------------------------------- launch ---
declare -a PIDS PORTS RANGES

for (( i=0; i<INSTANCES; i++ )); do
  cstart=$(( CORE_START + i * CORES_PER_INSTANCE ))
  cend=$(( cstart + CORES_PER_INSTANCE - 1 ))
  crange="${cstart}-${cend}"
  port=$(( BASE_PORT + i ))
  log="$RUN_DIR/inst${i}.log"

  echo "launching inst${i}: cores ${crange}  port ${port}  weight ${WEIGHT_ARR[$i]}"

  # Each instance is launched from its own subshell that *exports* the tuning
  # vars before exec'ing, rather than prefixing them with env(1).
  if [ "$BACKEND" = "llamacpp" ]; then
    ctx=$(( SLOT_CTX * PARALLEL ))
    (
      export LD_PRELOAD="$LD_PRELOAD_LIBS"
      # This node boots with isolcpus=domain,...,16-255, so the scheduler never
      # load-balances tasks onto these cores -- every thread stays wherever it
      # was created unless it is pinned explicitly. libggml-cpu links libgomp
      # (GOMP_parallel), which ignores KMP_AFFINITY entirely; GOMP_CPU_AFFINITY
      # is the variable it actually reads. Measured on 16 cores: KMP_AFFINITY
      # -> 1.00 cores in use, GOMP_CPU_AFFINITY -> 15.78.
      export GOMP_CPU_AFFINITY="$crange"
      export OMP_NUM_THREADS="$CORES_PER_INSTANCE"
      export OMP_DYNAMIC=FALSE
      export OMP_WAIT_POLICY=ACTIVE
      export ZENDNNL_MATMUL_ALGO="$ZENDNNL_MATMUL_ALGO"
      export ZENDNNL_LRU_CACHE_CAPACITY="$ZENDNNL_LRU_CACHE_CAPACITY"
      setsid nohup numactl --physcpubind="$crange" --membind="$NUMA_NODE" \
        "$LLAMA_SERVER" \
          -m "$LLAMA_MODEL" \
          -t "$CORES_PER_INSTANCE" \
          -fa on \
          -ctk f16 -ctv f16 \
          -c "$ctx" \
          -np "$PARALLEL" \
          --cont-batching \
          --no-context-shift \
          --cache-ram 0 \
          --cache-reuse 0 \
          --load-mode "$LOAD_MODE" \
          --metrics \
          --host 0.0.0.0 \
          --port "$port" \
        > "$log" 2>&1 < /dev/null &
      echo $! > "$RUN_DIR/inst${i}.pid"
    )
  else
    (
      # Mirrors the established working vLLM invocation (Intel libiomp5 from
      # the venv, zendnn matmul cache, mp executor, eager). vLLM does its own
      # explicit per-thread binding via VLLM_CPU_OMP_THREADS_BIND, which is what
      # makes it safe under this node's isolcpus setting.
      export LD_PRELOAD="$VLLM_LD_PRELOAD_LIBS"
      export VLLM_CPU_OMP_THREADS_BIND="$crange"
      export OMP_NUM_THREADS="$CORES_PER_INSTANCE"
      export VLLM_CPU_KVCACHE_SPACE="$KVCACHE_SPACE_GB"
      export ZENDNNL_MATMUL_ALGO="$ZENDNNL_MATMUL_ALGO"
      export ZENDNNL_MATMUL_WEIGHT_CACHE=1
      export ZENDNNL_LRU_CACHE_CAPACITY="$ZENDNNL_LRU_CACHE_CAPACITY"
      export VLLM_LOGGING_LEVEL=WARNING
      eager_flag=""
      [ "$ENFORCE_EAGER" = "1" ] && eager_flag="--enforce-eager"
      # shellcheck disable=SC2086
      setsid nohup numactl --physcpubind="$crange" --membind="$NUMA_NODE" \
        "$VLLM_BIN" serve "$VLLM_MODEL" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --distributed-executor-backend mp \
          $eager_flag \
          --dtype bfloat16 \
          --max-model-len "$(( SLOT_CTX * 2 ))" \
          --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
          --max-num-seqs "$PARALLEL" \
          --no-enable-prefix-caching \
          --uvicorn-log-level warning \
          --host 0.0.0.0 \
          --port "$port" \
        > "$log" 2>&1 < /dev/null &
      echo $! > "$RUN_DIR/inst${i}.pid"
    )
  fi

  pid="$(cat "$RUN_DIR/inst${i}.pid")"
  PIDS+=( "$pid" ); PORTS+=( "$port" ); RANGES+=( "$crange" )
done

# -------------------------------------------------------- wait for health --
echo
echo "waiting up to ${WAIT_SECS}s for all $INSTANCES instance(s) to become healthy..."
deadline=$(( $(date +%s) + WAIT_SECS ))
declare -a READY
for (( i=0; i<INSTANCES; i++ )); do READY+=( 0 ); done

all_ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  all_ready=1
  for (( i=0; i<INSTANCES; i++ )); do
    [ "${READY[$i]}" -eq 1 ] && continue
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
              "http://127.0.0.1:${PORTS[$i]}/health" 2>/dev/null)"
    if [ "$code" = "200" ]; then
      READY[$i]=1
      echo "  inst${i} (port ${PORTS[$i]}, cores ${RANGES[$i]}) ready after $(( WAIT_SECS - (deadline - $(date +%s)) ))s"
    else
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        echo "  inst${i} DIED during startup -- tail of $RUN_DIR/inst${i}.log:" >&2
        tail -25 "$RUN_DIR/inst${i}.log" >&2
        exit 1
      fi
      all_ready=0
    fi
  done
  [ "$all_ready" -eq 1 ] && break
  sleep 3
done

if [ "$all_ready" -ne 1 ]; then
  echo "TIMEOUT: not all instances healthy after ${WAIT_SECS}s" >&2
  for (( i=0; i<INSTANCES; i++ )); do
    [ "${READY[$i]}" -eq 0 ] && echo "  inst${i} port ${PORTS[$i]} NOT ready (see $RUN_DIR/inst${i}.log)" >&2
  done
  exit 1
fi

# --------------------------------------------------------------- manifest --
if [ "$BACKEND" = "llamacpp" ]; then
  MANIFEST_MODEL="$LLAMA_MODEL"; MANIFEST_SERVED=""
else
  MANIFEST_MODEL="$VLLM_MODEL";  MANIFEST_SERVED="$SERVED_MODEL_NAME"
fi

python3 <<PY
import json, datetime
ports   = "${PORTS[*]}".split()
ranges  = "${RANGES[*]}".split()
pids    = "${PIDS[*]}".split()
weights = "${WEIGHT_ARR[*]}".split()
m = {
    "config_name": "$CONFIG_NAME",
    "backend": "$BACKEND",
    "host": "$HOST_IP",
    "pod": "$(hostname)",
    "model": "$MANIFEST_MODEL",
    "served_model_name": "$MANIFEST_SERVED",
    "instances_n": $INSTANCES,
    "cores_per_instance": $CORES_PER_INSTANCE,
    "parallel_slots": $PARALLEL,
    "slot_ctx": $SLOT_CTX,
    "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "instances": [
        {"idx": i, "port": int(ports[i]), "cores": ranges[i],
         "ncores": $CORES_PER_INSTANCE, "weight": int(weights[i]), "pid": int(pids[i])}
        for i in range($INSTANCES)
    ],
}
with open("$RUN_DIR/manifest.json", "w") as f:
    json.dump(m, f, indent=2)
print(json.dumps(m, indent=2))
PY

echo
echo "all instances healthy. manifest: $RUN_DIR/manifest.json"
echo "launcher exiting now -- servers are detached (setsid); this script holds no cores."
exit 0
