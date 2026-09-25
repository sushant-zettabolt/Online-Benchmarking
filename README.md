# online_bench — vLLM vs llama.cpp CPU inference benchmarking harness

Tooling to benchmark **vLLM** and **llama.cpp** serving Llama-3.1-8B-Instruct
on AMD Zen (Turin) CPU pods, across two independent axes:

1. **Multi-user load harness** (`multiuser/`) — N server instances pinned to
   disjoint core slices, fronted by HAProxy weighted round robin, driven by a
   Locust load profile (paced per-user requests, swept across user counts).
   Answers "how does a fixed core budget behave under concurrent load, and
   how should it be split into instances?"
2. **Single-request pp-sweep** (this directory) — one server, one request at
   a time, prompt length swept from 1 to 4096 tokens. Answers "what is the
   raw per-request prefill/decode throughput, isolated from any queueing or
   scheduling effects?"

Both axes have been run for several build/quantization variants: ZenDNN
(llama.cpp) / zentorch (vLLM) accelerated builds vs plain non-accelerated
builds, and BF16 vs INT8 (Q8_0 for llama.cpp, W8A8 for vLLM) quantization.

**For current run status, numbers, and hard-won gotchas, read
[`multiuser/HANDOFF.md`](multiuser/HANDOFF.md) — it is the living status doc
and takes precedence over anything below if they ever disagree.** This
README is the stable "how the harness works and how to run it" reference.
**For the exact literal command + env vars used for every build variant and
every experiment config (e.g. "what command was ZenDNN llama.cpp vs
non-ZenDNN llama.cpp launched with"), see
[`multiuser/COMMANDS.md`](multiuser/COMMANDS.md).**

## Environment / pod topology

| Pod | Role | Cores | NUMA node | Runs |
|---|---|---|---|---|
| `turin-xcovoid0014-pod-6` (`192.168.7.60`) | **inference** | 192-223 (32 cores) | 6 | LLM server instances only |
| `turin-xcovoid0021-pod-5` (`192.168.13.19`) | **support** | 160-191 (32 cores) | 5 | HAProxy + Locust (multi-user only) |

- Both pods share `/proj/rdi/staff/sacsharm` over NFS (also mounted as
  `/proj/aigstaff/sacsharm` — **same files, same inode, two path aliases**;
  any absolute path baked into a file at install/copy time — e.g. a venv's
  script shebangs — will consistently use the `/proj/aigstaff/...` form).
- `KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml` for all `kubectl exec`
  calls against these pods (both in namespace `zendnn`).
- The inference pod boots with `isolcpus=domain,managed_irq,16-255`
  (`nohz_full`/`rcu_nocbs` too) — cores 16-255 are pulled out of the kernel's
  load-balancing domains, so **a thread never migrates onto them on its
  own**. Every compute thread must be pinned individually or the whole
  process collapses onto one core while the rest sit idle. This is why:
  - llama.cpp is launched with `GOMP_CPU_AFFINITY=<lo>-<hi>` (not
    `KMP_AFFINITY` — `libggml-cpu.so.0` links libgomp, not Intel libomp, so
    `KMP_AFFINITY` is a silent no-op here).
  - vLLM is launched with `VLLM_CPU_OMP_THREADS_BIND=<lo>-<hi>`, which does
    correct explicit per-thread binding already.
  - Both are already wired into `multiuser/start_servers.sh` — you shouldn't
    need to touch this unless adding a genuinely new launch path. Verify
    with `multiuser/monitor_cores.sh` + `multiuser/analyze_coreusage.py`
    (samples `/proc/stat` per core, flags any core reading near-zero) any
    time a new binary/config combination is tried for the first time.
    `multiuser/diag_affinity.sh` reproduces the underlying A/B if this ever
    needs re-demonstrating from scratch.

## Repo layout

```
online_bench/
├── README.md                 this file
├── bench_client.py            single-request HTTP client (pp-sweep)
├── sweep_online.sh             pp-sweep driver: --backend {llamacpp,vllm} --url ... --results-dir ...
├── consolidate_online.sh       turns a pp-sweep results dir into summary.tsv
├── results/                    pp-sweep output (gitignored, regenerate by running)
└── multiuser/
    ├── README.md                multi-user harness design & usage reference
    ├── HANDOFF.md                living status doc: what's been run, numbers, gotchas
    ├── start_servers.sh / stop_servers.sh     multi-instance launcher (inference pod)
    ├── start_haproxy.sh / stop_haproxy.sh     WRR load balancer (support pod)
    ├── run_multiuser_sweep.sh                 sweeps user counts, snapshots metrics
    ├── locustfile.py                           load profile (paced per-user requests)
    ├── consolidate_multiuser.py                 results dir -> summary TSVs
    ├── prom_metrics.py                           vLLM /metrics scraping helper
    ├── monitor_cores.sh / analyze_coreusage.py    per-core busy% verification
    ├── diag_affinity.sh                           isolcpus pinning A/B repro
    ├── stage_models.sh                            NFS -> pod local overlay disk
    ├── requirements.txt                            locust==2.46.6 (for .venv/)
    ├── configs/*.conf                               one file per experiment (see below)
    ├── .venv/                                       Locust venv (gitignored)
    ├── results/                                     sweep output (gitignored)
    └── run/                                         server logs, manifests, HAProxy cfg (gitignored)
```

`results/`, `run/`, and all `.venv*` directories are intentionally
gitignored — they're regenerated by running the harness, not source. Model
weights (BF16/Q8_0/W8A8 GGUF/HF checkpoints) live outside this repo entirely
(NFS + pod-local `/tmp/models` overlay) — see `stage_models.sh` and
`HANDOFF.md`'s Environment section for exact paths.

## Setup

```bash
cd multiuser
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Locust, on the support pod
./stage_models.sh                                                     # NFS -> /tmp/models on the inference pod, once per pod restart
```

The vLLM and llama.cpp binaries themselves are **not** part of this repo —
they're separate builds living at `/proj/rdi/staff/sacsharm/{vllm,llama.cpp}`
(upstream source checkouts with their own build directories:
llama.cpp's `build_zendnn` (`GGML_ZENDNN=ON`) vs `build_release`
(`GGML_ZENDNN=OFF`); vLLM's `.venv` (has `zentorch` installed, auto-selects
`ZenCpuPlatform`) vs `.venv_nozentorch` (zentorch removed, falls back to
plain `CpuPlatform`) — same compiled extensions either way, since vLLM's
ZenDNN integration is a runtime plugin, not a compile-time flag). Configs
point `LLAMA_SERVER=`/`VLLM_BIN=` at whichever build/venv the experiment
needs.

### Running commands on the pods

All scripts below are meant to run **as the `sacsharm` user** via `kubectl
exec`. The `login <user>` heredoc wrapper on these pods is unreliable
(intermittent hangs / false "in use" lock rejections) — use this pattern
instead:

```bash
kubectl exec <pod> -n zendnn -- bash -c '
  u=$(stat -Lc "%u" /proj/rdi/staff/sacsharm); g=$(stat -Lc "%g" /proj/rdi/staff/sacsharm)
  setpriv --reuid "$u" --regid "$g" --clear-groups env HOME=/proj/rdi/staff/sacsharm \
    bash -c "cd /proj/rdi/staff/sacsharm/online_bench/multiuser && <command>"
'
```

If `<command>` needs to launch something detached (`setsid nohup ... &`),
put `cd`/`mkdir`/etc. on their own lines inside the inner script and only
let the final `setsid nohup ... &` line end with `&` — chaining
`cd && setsid nohup ... & echo $!` on one line backgrounds the *entire*
preceding `&&` chain as a single job, so `cd` doesn't persist to whatever
comes after.

## Multi-user harness — quick start

**On the inference pod:**

```bash
cd online_bench/multiuser
./stage_models.sh                             # once per pod restart, /tmp is wiped
./start_servers.sh configs/llamacpp_2x16.conf --wait-secs 900
# script exits once healthy; servers keep running detached
```

**On the support pod:**

```bash
cd online_bench/multiuser
./start_haproxy.sh llamacpp_2x16               # reads run/<config>/manifest.json, builds WRR config
setsid nohup ./run_multiuser_sweep.sh llamacpp_2x16 --users "2 4 8 12 16" --duration 300 \
  > run/sweep_llamacpp_2x16.log 2>&1 < /dev/null &    # ~28 min for 5 user counts
```

**Teardown:**

```bash
./stop_haproxy.sh llamacpp_2x16      # support pod
./stop_servers.sh llamacpp_2x16      # inference pod (or --all)
```

**Consolidate / re-consolidate at any time:**

```bash
python3 consolidate_multiuser.py results/llamacpp_2x16 results/vllm_2x16_noeager
```

Load profile (`locustfile.py`): 1024-token random-token-id prompt (fresh
tokenization per request — defeats both llama.cpp's per-slot prompt cache
and vLLM's automatic prefix caching), 128-token completion, one request
every 30 seconds per simulated user (`constant_pacing`), 5-minute run per
user count. See `multiuser/README.md` for the full design rationale
(why 30s pacing already means back-to-back load at every user count, WRR
weighting, HAProxy tuning flags, etc).

### Experiment configs (`configs/*.conf`)

Naming: `<backend>_<instances>x<cores-per-instance>[_<variant>].conf`.
`start_servers.sh` carves the 32-core pool into that many consecutive
slices automatically (e.g. `4x8` → cores 192-199, 200-207, 208-215, 216-223).

| config | backend | layout | build | model | notes |
|---|---|---|---|---|---|
| `llamacpp_1x32` | llama.cpp | 1×32 | `build_zendnn` | BF16 GGUF | baseline |
| `llamacpp_2x16` | llama.cpp | 2×16 | `build_zendnn` | BF16 GGUF | |
| `llamacpp_3x10` | llama.cpp | 3×10 | `build_zendnn` | BF16 GGUF | 30 of 32 cores |
| `llamacpp_4x8` | llama.cpp | 4×8 | `build_zendnn` | BF16 GGUF | |
| `llamacpp_8x4` | llama.cpp | 8×4 | `build_zendnn` | BF16 GGUF | not yet benchmarked |
| `llamacpp_1x32_nozendnn` | llama.cpp | 1×32 | `build_release` (no ZenDNN) | BF16 GGUF | single-request pp-sweep only |
| `llamacpp_1x32_nozendnn_q8` | llama.cpp | 1×32 | `build_release` (no ZenDNN) | Q8_0 GGUF | single-request pp-sweep only |
| `vllm_1x32` | vLLM | 1×32 | `.venv` (zentorch), eager | BF16 HF | superseded by `_noeager` |
| `vllm_1x32_noeager` | vLLM | 1×32 | `.venv` (zentorch), non-eager | BF16 HF | non-eager is the standing convention (see below) |
| `vllm_2x16_noeager` | vLLM | 2×16 | `.venv` (zentorch), non-eager | BF16 HF | |
| `vllm_3x10_noeager` | vLLM | 3×10 | `.venv` (zentorch), non-eager | BF16 HF | |
| `vllm_4x8` / `vllm_4x8_noeager` | vLLM | 4×8 | `.venv` (zentorch) | BF16 HF | use the `_noeager` one |
| `vllm_1x32_nozentorch` | vLLM | 1×32 | `.venv_nozentorch`, non-eager | BF16 HF | single-request pp-sweep only |
| `vllm_1x32_nozentorch_w8a8` | vLLM | 1×32 | `.venv_nozentorch`, non-eager | INT8 W8A8 HF | single-request pp-sweep only |

**Non-eager convention**: every vLLM config created after the initial eager
A/B (`ENFORCE_EAGER=0`, i.e. `--enforce-eager` NOT passed, torch.compile
enabled) — non-eager won at every user count tested, gap widening with
load. Plain (no-suffix) `vllm_2x16.conf`/`vllm_4x8.conf` still exist and
default to eager; prefer the `_noeager` variant unless deliberately
re-testing eager mode.

To add a new layout, copy the closest existing `.conf`, adjust
`INSTANCES`/`CORES_PER_INSTANCE`/`CORE_START`, and give it a name matching
the convention above.

## Single-request pp-sweep — quick start

No HAProxy/Locust needed — one server, `bench_client.py` issues requests
directly:

```bash
cd online_bench
setsid nohup ./sweep_online.sh --backend llamacpp --url http://192.168.7.60:8080 \
  --results-dir results/llamacpp_nozendnn > /tmp/sweep.log 2>&1 < /dev/null &
./consolidate_online.sh results/llamacpp_nozendnn
```

Sweeps the fixed prompt-length list `1 2 4 8 16 32 48 64 96 128 256 384 512
768 1024 1536 2048 3072 4096` tokens, `tg=1` (practical stand-in for
`tg=0` — neither server API accepts 0), 1 warmup + 3 measured requests per
point, fresh random-nonsense-text prompt per request (full cache defeat).
Resumable — re-running skips any `online_*_<pp>.json` that already exists.
`consolidate_online.sh <dir>` turns that into a `pp/prompt_tok/ttft_s/tok_s`
`summary.tsv`. The one-instance-32-core `configs/*_nozendnn*`/`*_nozentorch*`
harness configs above are launched with `start_servers.sh` purely as a
single-instance launcher (no HAProxy/Locust) to feed a URL into this tool.

## Known gotchas (condensed — see `multiuser/HANDOFF.md` for full detail)

1. **isolcpus pinning** — see Environment section above.
2. **`login <user>` heredoc is unreliable** — use the `setpriv` pattern above.
3. **Never start a sweep without confirming nothing stale is already
   running** (`pgrep -f 'llama-server|vllm|haproxy|locustfile|
   run_multiuser_sweep|sweep_online'` on both pods should read 0).
4. **Copying a venv with `cp -a` does NOT make it self-contained** —
   entry-point script shebangs (`bin/vllm`, etc.) stay hardcoded to the
   *original* venv's interpreter path. `.venv_nozentorch` was built this
   way and silently ran with zentorch still loaded for two full sweeps
   before this was caught by checking the *live process's* actual
   interpreter (`ps`) and loaded libraries (`grep -c
   'site-packages/zentorch/' /proc/<pid>/maps` — not the bare word
   "zentorch", which false-positives against the venv's own directory
   name). Fix: rewrite every affected shebang after any `cp -a` of a venv.
   General lesson: verify a runtime claim (build flag, backend, config in
   effect) against the **live running process**, not an isolated
   re-creation of the same command.
5. **vLLM server-side Prometheus metrics in `consolidate_multiuser.py`
   only read instance 0** for multi-instance configs — a known bug in
   `prom_server_side()`, not yet fixed. Treat multi-instance
   `summary_server.tsv` `n` counts with suspicion until fixed.

## Status, results, and full run history

See [`multiuser/HANDOFF.md`](multiuser/HANDOFF.md) for: every experiment run
so far and its headline numbers, what's currently live on the pods, what's
still outstanding, and the full gotcha write-ups. See
[`multiuser/README.md`](multiuser/README.md) for the multi-user harness's
detailed design rationale (request duty cycle, HAProxy tuning, output file
formats). See [`multiuser/COMMANDS.md`](multiuser/COMMANDS.md) for the exact
reconstructed command line (env vars + argv) behind every config file, and
the exact `sweep_online.sh`/`run_multiuser_sweep.sh` invocations used to
produce every results directory. Raw result data (`results/`, `run/`) is not
committed to this repo — it's regenerated by running the sweeps, and lives
on the NFS share alongside the code.
