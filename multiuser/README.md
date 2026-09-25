# Multi-user online-serving benchmark (vLLM + llama.cpp)

Measures how a fixed 32-core budget behaves under a realistic multi-user load
when it is split into **N server instances pinned to consecutive core slices**
and fronted by **HAProxy weighted round robin**.

## Pod split

| Pod | Role | Cores | Runs |
|---|---|---|---|
| `turin-xcovoid0014-pod-6` (`192.168.7.60`) | **inference** | 192-223, NUMA node 6 | only the server instances |
| `turin-xcovoid0021-pod-5` (`192.168.13.19`) | **support** | 160-191, NUMA node 5 | HAProxy + Locust |

The load generator and the load balancer live on the *support* pod on purpose:
all 32 inference cores stay dedicated to the servers. `start_servers.sh`
detaches every instance with `setsid` and then **exits**, so once it returns no
part of the harness is resident on the inference pod.

The two pods share `/proj/rdi/staff/sacsharm` over NFS, which is how the
inference pod's `run/<config>/manifest.json` reaches the support pod.

## Load profile

Exactly as specified:

* prompt **1024 tokens**, generation capped at **128 tokens**
* **1 request every 30 seconds per user** (Locust `constant_pacing(30)`)
* each run lasts **5 minutes** → ~5 requests per user
* user counts swept **separately**: **2, 4, 8, 12, 16**
* users ramp in at 1/s, then each takes a random phase offset inside the
  30 s window so N independent users don't collapse into one synchronised burst

Every request's prompt is 1024 **freshly drawn random token ids**, so neither
llama.cpp's per-slot prompt cache nor vLLM's automatic prefix caching can ever
produce a hit — the same cache-defeat requirement as the single-user sweep, but
without spending an extra `/tokenize` round trip that would skew the HAProxy
request distribution.

## Instance layouts

`configs/<backend>_<N>x<cores>.conf`. `start_servers.sh` carves the pool into
consecutive slices automatically:

```
INSTANCES=1 CORES_PER_INSTANCE=32  ->  inst0 192-223                                  :8080
INSTANCES=2 CORES_PER_INSTANCE=16  ->  inst0 192-207  inst1 208-223                   :8080 :8081
INSTANCES=4 CORES_PER_INSTANCE=8   ->  inst0 192-199  inst1 200-207  ...               :8080..:8083
```

Available: `llamacpp_1x32`, `llamacpp_2x16`, `llamacpp_4x8`, `llamacpp_8x4`,
`vllm_1x32`, `vllm_2x16`, `vllm_4x8`.

HAProxy's `balance roundrobin` is a *dynamic weighted* round robin. Each
server's `weight` is set to its core count, so equal slices split traffic
evenly and a heterogeneous layout (e.g. `WEIGHTS="16 8 8"`) splits 50/25/25.

## Running a sweep

**On the inference pod** (`turin-xcovoid0014-pod-6`):

```bash
cd /proj/rdi/staff/sacsharm/online_bench/multiuser
./stage_models.sh                       # once per pod restart -- /tmp is wiped
./start_servers.sh configs/llamacpp_2x16.conf
# ...script exits; servers keep running detached
```

**On the support pod** (`turin-xcovoid0021-pod-5`):

```bash
cd /proj/rdi/staff/sacsharm/online_bench/multiuser
./start_haproxy.sh llamacpp_2x16        # reads the manifest, builds the WRR config
./run_multiuser_sweep.sh llamacpp_2x16  # 2,4,8,12,16 users x 5 min each (~28 min)
```

Teardown:

```bash
./stop_haproxy.sh llamacpp_2x16         # support pod
./stop_servers.sh llamacpp_2x16         # inference pod  (or --all)
```

## Output

```
results/<config>/summary.tsv        one row per user count
results/<config>/distribution.tsv   requests served per instance vs configured WRR share
results/<config>/users_<N>/samples.jsonl   one JSON object per request
results/<config>/users_<N>/locust_*.csv    Locust's own stats
run/<config>/inst<i>.log            server logs
run/<config>/lb/haproxy.{cfg,log}   generated LB config + access log
```

`summary.tsv` columns: `users reqs fail req_per_min ttft_p50 ttft_p90 ttft_p99
tpot_ms_p50 tpot_ms_p90 e2e_p50 e2e_p90 e2e_p99 out_tok_s prompt_tok`.

`prompt_tok` is the server-reported prompt length — it should read 1024, which
verifies the intended load rather than assuming it.

Re-consolidate at any time (and compare layouts side by side):

```bash
python3 consolidate_multiuser.py results/llamacpp_1x32 results/llamacpp_2x16 results/llamacpp_4x8
```

## Core pinning on this node — important

`turin-xcovoid0014` boots with:

```
isolcpus=domain,managed_irq,16-255   nohz_full=16-255   rcu_nocbs=16-255
```

`isolcpus=domain` removes cores 16-255 from the scheduler's load-balancing
domains. The kernel **never migrates a runnable task onto them** — a thread
stays on whatever core it was created on, and forks inherit that core. So
"affinity says 192-223" is not enough; every compute thread has to be pinned
*individually* or the whole process collapses onto one core while 31 sit idle.

Measured on this pod: 32 busy-loop processes with affinity `192-223` consumed a
combined **1.01 cores** over 5 wall-seconds. Pinning one spinner per core with
`taskset` instead gave **16.30 cores** for 16 spinners.

This is why `start_servers.sh` sets **`GOMP_CPU_AFFINITY=<lo>-<hi>`** for
llama.cpp. `libggml-cpu.so.0` links **libgomp** and dispatches through
`GOMP_parallel`; libgomp does not read `KMP_AFFINITY` (an Intel/LLVM-libomp
variable), so that setting is silently a no-op here. Because ggml is built with
OpenMP, llama.cpp's own `--cpu-range` / `--cpu-mask` flags are also ignored for
compute. Single-instance, 16 cores, 512-token prompt:

| env | cores in use | pp tok/s |
|---|---|---|
| `KMP_AFFINITY=granularity=fine,compact,1,0` | 1.00 | request timed out |
| `GOMP_CPU_AFFINITY=192-207` | 15.78 | 269.8 |
| `GOMP_CPU_AFFINITY` + libomp preloaded | 15.78 | 270.7 |
| `OMP_PROC_BIND=close` + explicit `OMP_PLACES` | 15.78 | — |

vLLM needs no equivalent change: `VLLM_CPU_OMP_THREADS_BIND=<lo>-<hi>` already
performs explicit per-thread binding, which is isolcpus-safe.

Verify placement any time with `diag_affinity.sh` (starts one server per env
variant and reports cores actually consumed during a real prompt eval).

## Request duty cycle — worth knowing before reading results

On 16 cores, one 1024-prompt / 128-token request takes **~32 s** end to end
(~3.3 s prefill, ~226 ms/token decode) — already longer than the 30 s pacing
interval. That means even a *single* user is essentially back-to-back busy
(`constant_pacing` fires the next request immediately once the wait interval
has already elapsed), and every user count in the sweep — not just 12/16 —
will show real queueing rather than sitting idle between requests. That is a
property of the requested load shape, not a harness artefact — but it means
TTFT at every point, including the low end of the sweep, will include real
queue wait, and cluster utilization (`mean_conc` in
`summary_throughput.tsv`) should read close to the full user count rather
than diluted by idle gaps.

## Notes

* `stage_models.sh` copies the models from NFS to the pod's local overlay disk.
  The llama.cpp configs use `--load-mode mmap` so N instances share one
  page-cache copy of the 16 GiB model; if that mmap were NFS-backed, a decode
  page fault would become a network round trip and pollute the latency numbers.
* HAProxy sets `option http-no-delay` — without it small SSE frames get
  coalesced and the measured TTFT/TPOT would be wrong.
* HAProxy adds an `X-Served-By` response header naming the backend instance, so
  the per-instance split can be confirmed from the client side as well as from
  HAProxy's own counters.
* HAProxy needed `apt-get install haproxy` (root). Everything else runs as
  `sacsharm`; Locust lives in `./.venv`.
