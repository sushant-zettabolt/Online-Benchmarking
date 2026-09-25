# Exact commands and flags used per experiment

This is the literal reference for "what command actually ran" for every
build variant and every experiment config in this repo — reconstructed from
`start_servers.sh`'s templates (substituted with each config's real values)
and, where sweeps have already been run, verified against the actual
recorded `.log`/`manifest.json` files in `results/`/`run/` (gitignored, but
still present on the pods' NFS share as of this writing). If you need to
re-derive a command yourself: source the `.conf` file's variables into the
relevant template below, or just read `start_servers.sh` directly — it is
the single source of truth these commands are generated from.

## 1. Build / environment setup (one-time, per backend variant)

### llama.cpp — ZenDNN vs non-ZenDNN

Both builds are the same source tree, same every other CMake flag; the
*only* difference (confirmed via `diff` of the two `CMakeCache.txt` files)
is `GGML_ZENDNN` / `ZENDNN_ROOT`:

```bash
# ZenDNN-accelerated build -> llama.cpp/build_zendnn/bin/llama-server
cmake -B build_zendnn -DCMAKE_BUILD_TYPE=Release \
  -DGGML_ZENDNN=ON -DZENDNN_ROOT=/proj/rdi/staff/sacsharm/ZenDNN/build/install
cmake --build build_zendnn --config Release -j 32

# Plain (non-ZenDNN) build -> llama.cpp/build_release/bin/llama-server
cmake -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release -j 32
```

Both have `GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON`, `GGML_OPENMP=ON`,
`GGML_BLAS=OFF`. `build_release` was already present when the non-ZenDNN
comparison was done this project; it was rebuilt incrementally
(`cmake --build build_release --config Release -j 32`, a no-op on unchanged
files) rather than reconfigured from scratch, to pick up any source drift.

### vLLM — zentorch vs non-zentorch

vLLM's ZenDNN integration (`zentorch`) is a **runtime plugin, not a
compile-time flag** — `vllm/platforms/__init__.py::cpu_platform_plugin()`
selects `ZenCpuPlatform` if `import zentorch` succeeds, else falls back to
plain `CpuPlatform` (same compiled `_C*.so` extensions either way). So
there is no separate "vLLM build" — just whether `zentorch` is importable
in the venv that launches the server.

`vllm/.venv` (zentorch installed, used by every config below that isn't
suffixed `_nozentorch`) predates this benchmarking work — installed
versions, for reference/reproducibility: `vllm==0.28.1rc1.dev485+g58ad1f3b8.cpu`,
`torch==2.13.0+cpu`, `zentorch==2.13.0.0`.

`vllm/.venv_nozentorch` (used by every `_nozentorch*` config) was created
this project by copying `.venv` and removing the `zentorch` package:

```bash
cd /proj/rdi/staff/sacsharm/vllm
cp -a .venv .venv_nozentorch
rm -rf .venv_nozentorch/lib/python3.12/site-packages/zentorch \
       .venv_nozentorch/lib/python3.12/site-packages/zentorch-2.13.0.0.dist-info
```

**Critical extra step, do not skip**: `cp -a` does NOT make a venv
self-contained — every `bin/*` entry-point script's shebang stays hardcoded
to the *original* venv's interpreter path. Without this fix, every server
launched via `.venv_nozentorch/bin/vllm` silently still runs on `.venv`'s
python (which still has zentorch importable) — this happened for real, see
`HANDOFF.md` gotcha #5. Fix, required after every `cp -a` of a venv:

```bash
cd /proj/rdi/staff/sacsharm/vllm/.venv_nozentorch/bin
grep -rl '^#!/proj/aigstaff/sacsharm/vllm/[.]venv/bin/python' . | \
  xargs sed -i '1s#/proj/aigstaff/sacsharm/vllm/[.]venv/bin/python#/proj/aigstaff/sacsharm/vllm/.venv_nozentorch/bin/python#'
```

Verify: `ps` on the live server PID shows the interpreter as
`.venv_nozentorch/bin/python3` (not `.venv/bin/python3`), and
`grep -c 'site-packages/zentorch/' /proc/<pid>/maps` reads `0`.

### Locust (multi-user load generator)

```bash
cd multiuser && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # locust==2.46.6
```

## 2. Server launch commands (`start_servers.sh`)

`start_servers.sh` builds one command per instance from a config file's
variables. Every instance is launched via
`numactl --physcpubind=<core-range> --membind=<numa-node>` in front of the
actual server binary, fully detached (`setsid nohup ... &`, re-parented to
init) so the launcher itself holds no cores after it exits.

### llama.cpp instance template

```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5"
export GOMP_CPU_AFFINITY="<core-range>"      # e.g. 192-223 -- NOT KMP_AFFINITY, silent no-op (libgomp, not libomp)
export OMP_NUM_THREADS="<cores-per-instance>"
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_LRU_CACHE_CAPACITY=1024

numactl --physcpubind="<core-range>" --membind="<numa-node>" \
  <LLAMA_SERVER> \
    -m <LLAMA_MODEL> \
    -t <cores-per-instance> \
    -fa on \
    -ctk f16 -ctv f16 \
    -c $((SLOT_CTX * PARALLEL)) \
    -np <PARALLEL> \
    --cont-batching \
    --no-context-shift \
    --cache-ram 0 \
    --cache-reuse 0 \
    --load-mode <LOAD_MODE> \
    --metrics \
    --host 0.0.0.0 \
    --port <port>
```

**Fully expanded — llama.cpp ZenDNN, single instance, 32 cores**
(`configs/llamacpp_1x32.conf`; instance 0, cores 192-223, port 8080):

```bash
GOMP_CPU_AFFINITY=192-223 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE \
ZENDNNL_MATMUL_ALGO=1 ZENDNNL_LRU_CACHE_CAPACITY=1024 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5 \
numactl --physcpubind=192-223 --membind=6 \
  /proj/rdi/staff/sacsharm/llama.cpp/build_zendnn/bin/llama-server \
    -m /tmp/models/Llama-3.1-8B-Instruct-BF16.gguf \
    -t 32 -fa on -ctk f16 -ctv f16 \
    -c 32768 -np 16 --cont-batching --no-context-shift \
    --cache-ram 0 --cache-reuse 0 --load-mode mmap --metrics \
    --host 0.0.0.0 --port 8080
```

**Fully expanded — llama.cpp non-ZenDNN, BF16, single instance, 32 cores**
(`configs/llamacpp_1x32_nozendnn.conf`, pp-sweep config: `PARALLEL=1
SLOT_CTX=8192` so `-c 8192 -np 1`):

```bash
GOMP_CPU_AFFINITY=192-223 OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE \
ZENDNNL_MATMUL_ALGO=1 ZENDNNL_LRU_CACHE_CAPACITY=1024 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5 \
numactl --physcpubind=192-223 --membind=6 \
  /proj/rdi/staff/sacsharm/llama.cpp/build_release/bin/llama-server \
    -m /tmp/models/Llama-3.1-8B-Instruct-BF16.gguf \
    -t 32 -fa on -ctk f16 -ctv f16 \
    -c 8192 -np 1 --cont-batching --no-context-shift \
    --cache-ram 0 --cache-reuse 0 --load-mode mmap --metrics \
    --host 0.0.0.0 --port 8080
```

**llama.cpp non-ZenDNN, Q8_0** (`configs/llamacpp_1x32_nozendnn_q8.conf`):
identical to the above except `-m /tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf`
(same `build_release` binary, same everything else — quantization is a
property of the GGUF file, not a server flag).

The `ZENDNNL_*` env vars are harmless-but-inert on `build_release` (no
ZenDNN code path present to read them) — left set for template consistency,
they do not affect the non-ZenDNN runs.

### vLLM instance template

```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:<venv>/lib/libiomp5.so"
export VLLM_CPU_OMP_THREADS_BIND="<core-range>"   # explicit per-thread binding -- isolcpus-safe by itself
export OMP_NUM_THREADS="<cores-per-instance>"
export VLLM_CPU_KVCACHE_SPACE="<KVCACHE_SPACE_GB>"
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_MATMUL_WEIGHT_CACHE=1
export ZENDNNL_LRU_CACHE_CAPACITY=1024
export VLLM_LOGGING_LEVEL=WARNING

numactl --physcpubind="<core-range>" --membind="<numa-node>" \
  <VLLM_BIN> serve <VLLM_MODEL> \
    --served-model-name <SERVED_MODEL_NAME> \
    --distributed-executor-backend mp \
    [--enforce-eager]                          # only if ENFORCE_EAGER=1
    --dtype bfloat16 \
    --max-model-len $((SLOT_CTX * 2)) \
    --max-num-batched-tokens 4096 \
    --max-num-seqs <PARALLEL> \
    --no-enable-prefix-caching \
    --uvicorn-log-level warning \
    --host 0.0.0.0 \
    --port <port>
```

**Fully expanded — vLLM zentorch, non-eager, single instance, 32 cores**
(`configs/vllm_1x32_noeager.conf`; instance 0, cores 192-223, port 8000):

```bash
VLLM_CPU_OMP_THREADS_BIND=192-223 OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=16 \
ZENDNNL_MATMUL_ALGO=1 ZENDNNL_MATMUL_WEIGHT_CACHE=1 ZENDNNL_LRU_CACHE_CAPACITY=1024 \
VLLM_LOGGING_LEVEL=WARNING \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so \
numactl --physcpubind=192-223 --membind=6 \
  /proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm serve /tmp/models/Llama-3.1-8B-Instruct \
    --served-model-name llama31-8b \
    --distributed-executor-backend mp \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching \
    --uvicorn-log-level warning \
    --host 0.0.0.0 --port 8000
```
(`configs/vllm_1x32.conf` is identical but `ENFORCE_EAGER=1`, i.e. adds
`--enforce-eager` — this is the eager/non-eager A/B pair; non-eager won at
every user count and is the standing convention for every config created
after it.)

**Fully expanded — vLLM non-zentorch, BF16, single instance, 32 cores**
(`configs/vllm_1x32_nozentorch.conf`, pp-sweep config: `PARALLEL=1
SLOT_CTX=4096` so `--max-num-seqs 1 --max-model-len 8192`):

```bash
VLLM_CPU_OMP_THREADS_BIND=192-223 OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=16 \
ZENDNNL_MATMUL_ALGO=1 ZENDNNL_MATMUL_WEIGHT_CACHE=1 ZENDNNL_LRU_CACHE_CAPACITY=1024 \
VLLM_LOGGING_LEVEL=WARNING \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv_nozentorch/lib/libiomp5.so \
numactl --physcpubind=192-223 --membind=6 \
  /proj/rdi/staff/sacsharm/vllm/.venv_nozentorch/bin/vllm serve /tmp/models/Llama-3.1-8B-Instruct \
    --served-model-name llama31-8b \
    --distributed-executor-backend mp \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --uvicorn-log-level warning \
    --host 0.0.0.0 --port 8000
```
(No `--enforce-eager` — non-eager, matching the standing convention.)

**vLLM non-zentorch, W8A8** (`configs/vllm_1x32_nozentorch_w8a8.conf`):
identical to the above except `serve
/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` and
`--served-model-name llama31-8b-w8a8` (checkpoint downloaded pre-quantized
from HF Hub, `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`,
compressed-tensors INT8 W8A8 format — vLLM auto-detects the quantization
scheme from the checkpoint's `quantization_config`, no extra CLI flag
needed). `--dtype bfloat16` still applies to any non-quantized tensors
(embeddings etc.); the quantized linear layers run INT8×INT8.

### Multi-instance layouts (2×16 / 3×10 / 4×8 / 8×4)

Same per-instance templates as above, just repeated once per instance with
`CORE_START + i*CORES_PER_INSTANCE` and `BASE_PORT + i`. E.g. for
`configs/llamacpp_4x8.conf` (4 instances × 8 cores, ports 8080-8083):

| instance | cores | port |
|---|---|---|
| 0 | 192-199 | 8080 |
| 1 | 200-207 | 8081 |
| 2 | 208-215 | 8082 |
| 3 | 216-223 | 8083 |

Every instance gets its own `GOMP_CPU_AFFINITY`/`VLLM_CPU_OMP_THREADS_BIND`
set to its own slice — this is what keeps 4 independent server processes
from fighting over the same cores under `isolcpus`. See the full parameter
table in the top-level `README.md` for every config's instance count / core
count / model / build.

## 3. Multi-user sweep commands (`run_multiuser_sweep.sh` + HAProxy)

Every multi-user config in the table above was (or, for the ones still
outstanding, would be) driven identically from the support pod:

```bash
./start_haproxy.sh <config_name>
setsid nohup ./run_multiuser_sweep.sh <config_name> --users "2 4 8 12 16" --duration 300 \
  > run/sweep_<config_name>.log 2>&1 < /dev/null &
```

`run_multiuser_sweep.sh` defaults (all used as-is, never overridden, for
every sweep run so far): `--spawn-rate 1`, `--pacing 30`, `--pp 1024
--tg 128`, `--cooldown 30`. `--pacing 30` (1 request every 30s per
simulated user, via Locust's `constant_pacing`) was changed from an
earlier default of 60s partway through this project — every sweep listed
as "30s" in `HANDOFF.md`'s run table used the current default; the three
listed as "60s" (`llamacpp_1x32`, `vllm_1x32`, `vllm_1x32_noeager`) predate
that change and were run with `--pacing 60` (the then-current default, not
an explicit override either).

Teardown after a sweep: `./stop_haproxy.sh <config_name>` (support pod),
`./stop_servers.sh <config_name>` (inference pod).

Core-usage verification alongside any new config, run concurrently with the
sweep (inference pod):

```bash
setsid nohup ./monitor_cores.sh 192-223 5 /tmp/monitor_<config_name>.csv &
# ... let the sweep run ...
python3 analyze_coreusage.py /tmp/monitor_<config_name>.csv run/<config_name>/manifest.json
```

Consolidate: `python3 consolidate_multiuser.py results/<config_a> results/<config_b> ...`

## 4. Single-request pp-sweep commands (`sweep_online.sh` / `bench_client.py`)

Run from `online_bench/` (not `multiuser/`) against an already-running
single-instance server (started the same way as above, just one instance,
32 cores, `PARALLEL=1` config variant):

```bash
setsid nohup ./sweep_online.sh --backend <llamacpp|vllm> --url http://192.168.7.60:<port> \
  [--model <served-model-name>] --results-dir results/<name> \
  > /tmp/sweep_<name>.log 2>&1 < /dev/null &
./consolidate_online.sh results/<name>
```

Every point in every completed pp-sweep used the tool's defaults —
`--tg 1` (practical stand-in for tg=0; neither server API accepts 0),
`--num-iters 3`, `--warmup-iters 1` — never overridden. Prompt sizes swept
(fixed list, hardcoded in `sweep_online.sh`, same for every run):
`1 2 4 8 16 32 48 64 96 128 256 384 512 768 1024 1536 2048 3072 4096`.

Exact invocations used for the six completed results directories (`--model`
matters for vLLM, which routes by served name; llama.cpp ignores it —
recorded literally in each result JSON's `"model"` field):

| results dir | command |
|---|---|
| `results/llamacpp` | `./sweep_online.sh --backend llamacpp --url http://192.168.7.60:8080 --results-dir results/llamacpp` |
| `results/llamacpp_nozendnn` | `./sweep_online.sh --backend llamacpp --url http://192.168.7.60:8080 --results-dir results/llamacpp_nozendnn` |
| `results/llamacpp_nozendnn_q8` | `./sweep_online.sh --backend llamacpp --url http://192.168.7.60:8080 --results-dir results/llamacpp_nozendnn_q8` |
| `results/vllm` | `./sweep_online.sh --backend vllm --url http://192.168.7.60:8000 --model llama31-8b --results-dir results/vllm` |
| `results/vllm_nozentorch` | `./sweep_online.sh --backend vllm --url http://192.168.7.60:8000 --model llama31-8b --results-dir results/vllm_nozentorch` |
| `results/vllm_nozentorch_w8a8` | `./sweep_online.sh --backend vllm --url http://192.168.7.60:8000 --model llama31-8b-w8a8 --results-dir results/vllm_nozentorch_w8a8` |

(`--model` defaults to the literal string `"default"` when omitted — fine
for llama.cpp, which has exactly one model loaded per instance and ignores
the field entirely; every recorded llama.cpp result JSON shows
`"model": "default"`.)

Each server instance backing these results was started exactly per the
"fully expanded" commands in section 2 (same config files:
`llamacpp_1x32.conf`/`llamacpp_1x32_nozendnn.conf`/`llamacpp_1x32_nozendnn_q8.conf`
for the llama.cpp rows, `vllm_1x32_noeager.conf`/`vllm_1x32_nozentorch.conf`/
`vllm_1x32_nozentorch_w8a8.conf` for the vLLM rows — note `results/vllm`
was served by the zentorch venv at whatever eager/non-eager setting was
live at the time; see `HANDOFF.md` for the exact caveat on that one
pre-existing result).

## 5. Where to find the literal argv/env of a specific past run

`run/<config_name>/inst<i>.log` (gitignored, present on the pods) has the
server's own startup log, which echoes effective settings (context size,
thread count, etc.) even though it does not print its own invoking argv
verbatim. `run/<config_name>/manifest.json` records the resolved
host/port/core-range/weight/pid per instance for that run.
`start_servers.sh`'s own stdout (also tee'd nowhere by default — capture it
yourself if you need a transcript) prints the config name, backend,
instance/core layout, and per-instance launch line as it runs.
