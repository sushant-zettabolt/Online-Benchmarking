#!/usr/bin/env bash
# Diagnostic: on a node booted with isolcpus (no scheduler load balancing onto
# cores 16-255), which env actually spreads llama-server's OpenMP compute
# threads across its core slice?
#
# Measures the only metric that matters: aggregate CPU-seconds consumed by the
# server process per wall-second during a real prompt-eval ("cores in use").
set -uo pipefail

S=/proj/rdi/staff/sacsharm/llama.cpp/build_zendnn/bin/llama-server
M=/tmp/models/Llama-3.1-8B-Instruct-BF16.gguf
CR="${CR:-192-207}"
NT="${NT:-16}"
PORT=8099
OUT=/tmp/diag_aff
mkdir -p $OUT

probe() {
  local name="$1"; shift
  echo "=================================================================="
  echo "VARIANT: $name"
  echo "  $*"
  echo "------------------------------------------------------------------"

  env "$@" setsid nohup numactl --physcpubind="$CR" --membind=6 \
      "$S" -m "$M" -t "$NT" -fa on -ctk f16 -ctv f16 -c 4096 -np 1 \
           --load-mode mmap --host 127.0.0.1 --port $PORT \
      > "$OUT/$name.log" 2>&1 < /dev/null &
  local pid=$!

  local up=0
  for _ in $(seq 1 150); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health)" = "200" ] && { up=1; break; }
    kill -0 $pid 2>/dev/null || { echo "  DIED:"; tail -4 "$OUT/$name.log"; return 1; }
    sleep 2
  done
  [ "$up" = 1 ] || { echo "  never became healthy"; kill -KILL -- -$pid 2>/dev/null; return 1; }

  local body
  body="$(python3 -c "import random;print(','.join(str(random.randint(10,127000)) for _ in range(512)))")"
  curl -s --max-time 180 -H 'Content-Type: application/json' \
    -d "{\"prompt\":[$body],\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false,\"ignore_eos\":true,\"stream\":false}" \
    http://127.0.0.1:$PORT/completion > "$OUT/$name.json" 2>&1 &
  local cpid=$!

  # aggregate CPU-time of the whole process over a 5s window mid-request
  python3 - "$pid" <<'PY'
import os, sys, time
CLK = os.sysconf('SC_CLK_TCK'); pid = sys.argv[1]
def cpu():
    f = open(f'/proc/{pid}/stat').read().split()
    return (int(f[13]) + int(f[14])) / CLK
time.sleep(2)
t0 = time.time(); a = cpu(); time.sleep(5); b = cpu(); t1 = time.time()
cpus = set()
for tid in os.listdir(f'/proc/{pid}/task'):
    try:
        st = open(f'/proc/{pid}/task/{tid}/stat').read().split()
        if st[2] == 'R': cpus.add(st[38])
    except Exception: pass
print("  CORES IN USE: %.2f      distinct cpus running: %d" % ((b - a) / (t1 - t0), len(cpus)))
PY

  wait $cpid 2>/dev/null
  python3 -c "
import json
try:
    t=json.load(open('$OUT/$name.json'))['timings']
    print('  pp=%.1f tok/s   tg=%.2f tok/s' % (t['prompt_per_second'], t['predicted_per_second']))
except Exception as e:
    print('  no timings (%s)' % e)
"
  kill -KILL -- "-$pid" 2>/dev/null; kill -KILL $pid 2>/dev/null
  sleep 3
  echo
}

TC=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
IOMP=/usr/lib/x86_64-linux-gnu/libomp.so.5
Z=(ZENDNNL_MATMUL_ALGO=1 ZENDNNL_LRU_CACHE_CAPACITY=1024 OMP_NUM_THREADS=$NT OMP_DYNAMIC=FALSE)

probe V1_current   LD_PRELOAD=$TC:$IOMP "${Z[@]}" OMP_WAIT_POLICY=ACTIVE KMP_AFFINITY=granularity=fine,compact,1,0
probe V2_gomp_only LD_PRELOAD=$TC       "${Z[@]}" OMP_WAIT_POLICY=ACTIVE GOMP_CPU_AFFINITY="$CR"
probe V3_gomp_iomp LD_PRELOAD=$TC:$IOMP "${Z[@]}" OMP_WAIT_POLICY=ACTIVE GOMP_CPU_AFFINITY="$CR"
probe V4_places    LD_PRELOAD=$TC       "${Z[@]}" OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_PLACES="$(python3 -c "
lo,hi=map(int,'$CR'.split('-')); print(','.join('{%d}'%c for c in range(lo,hi+1)))")"

echo "logs: $OUT"
