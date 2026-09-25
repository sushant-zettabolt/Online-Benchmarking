# HANDOFF — vLLM vs llama.cpp benchmarking (multi-user + single-request non-ZenDNN/quant)

Rewritten 2026-09-24 for continuity across a context compaction. Read this
first before touching anything. See also `README.md` in this directory for
the multi-user harness design/usage reference (still accurate). This file is
the "what's actually been done and what's live right now" layer on top of it,
covering BOTH the multi-user harness (this directory) and the single-request
pp-sweep tooling in the parent `online_bench/` directory.

## Environment

- Inference pod: `turin-xcovoid0014-pod-6` (IP `192.168.7.60`), 32 cores
  `192-223`, NUMA node 6. Runs the LLM servers only.
- Support pod: `turin-xcovoid0021-pod-5` (IP `192.168.13.19`), 32 cores
  `160-191`, NUMA node 5. Runs HAProxy + Locust for multi-user sweeps.
- Both share `/proj/rdi/staff/sacsharm` over NFS (`== /proj/aigstaff/sacsharm`,
  same path, aliased mount — **shebangs and other embedded absolute paths use
  the `/proj/aigstaff/...` form**, don't assume `/proj/rdi/staff/...` is what
  you'll find baked into a file).
- `KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml` for all `kubectl` calls.
- Models, all staged to `/tmp/models` on pod-6 (local overlay disk, **wiped on
  pod restart** — re-run `stage_models.sh` or re-copy if so):
  - `Llama-3.1-8B-Instruct-BF16.gguf` / `Llama-3.1-8B-Instruct` (HF) — BF16,
    unquantized, verified (16,060,556,376 / 16,068,895,872 bytes).
  - `Llama-3.1-8B-Instruct-Q8_0.gguf` — llama.cpp Q8_0 quant (source:
    `/proj/rdi/staff/sacsharm/models/gguf/`).
  - `Meta-Llama-3.1-8B-Instruct-quantized.w8a8` — genuine INT8 W8A8
    (compressed-tensors format: per-channel static INT8 weights, dynamic
    per-token INT8 activations), downloaded from HF Hub
    (`RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`, not gated) rather
    than quantized locally — see gotcha #5.

## CRITICAL gotchas (do not rediscover these the hard way again)

1. **This node boots with `isolcpus=domain,managed_irq,16-255`.** Cores
   16-255 are removed from the kernel's load-balancing domains, so a thread
   never migrates onto them on its own. **llama.cpp**: use
   `GOMP_CPU_AFFINITY=<lo>-<hi>` (not `KMP_AFFINITY`, a silent no-op since
   `libggml-cpu.so.0` links libgomp not Intel libomp) — already wired into
   `start_servers.sh`. **vLLM**: `VLLM_CPU_OMP_THREADS_BIND=<lo>-<hi>` already
   does explicit per-thread binding, already correct. `diag_affinity.sh`
   reproduces the A/B if this ever needs re-verifying. **Verification tool**:
   `monitor_cores.sh <lo>-<hi> <interval_s> <out.csv>` (samples per-core
   busy% from `/proc/stat` at a fixed interval, run detached alongside a
   sweep/request) + `analyze_coreusage.py <csv> <manifest.json>` (per-instance
   mean/min/max, flags any core reading <5% mean). Used throughout this
   session to confirm every multi-instance layout (1×32, 2×16, 3×10,
   partial 4×8) and every non-ZenDNN single-instance run actually spread
   across all pinned cores — always came back clean, no collapse.

2. **The `login <user>` wrapper on these pods is unreliable** (heredoc stdin
   sometimes not delivered, causing hangs or false "IN USE" lock rejections).
   **Workaround**: bypass `login` entirely.
   ```bash
   kubectl exec <pod> -n zendnn -- bash -c '
     u=$(stat -Lc "%u" /proj/rdi/staff/sacsharm); g=$(stat -Lc "%g" /proj/rdi/staff/sacsharm)
     setpriv --reuid "$u" --regid "$g" --clear-groups env HOME=/proj/rdi/staff/sacsharm \
       bash -c "cd /proj/rdi/staff/sacsharm/online_bench/multiuser && <command>"
   '
   ```
   100% reliable all session. **Sub-gotcha found this segment**: if the
   command run via this pattern needs to itself launch something detached
   with `setsid nohup ... &`, do NOT chain `cd && setsid nohup ... & echo $! > file`
   on one line — the trailing `&` backgrounds the *entire preceding `&&`
   chain* as one job, so `cd` runs in a forked subshell and does not persist
   to the next statement, and a relative-path `echo $! > file` afterward
   fails silently (wrong cwd). Fix: put `cd`, `mkdir`, etc. on their own
   lines (real newlines inside the double-quoted inner script), and only
   the final `setsid nohup ... &` line ends with `&` — no need for `$!`
   capture at all if you don't need the pid (use `pkill -f <pattern>` later).
   **Also**: `pkill -f 'foo'`/`grep foo` can self-match if the literal string
   `foo` appears anywhere in your *own* invoking command line (e.g. an
   English sentence in a fallback `echo`, or the venv directory itself being
   named e.g. `.venv_nozentorch`). Obfuscate one character
   (`pkill -f 'monitor_cores[.]sh'`) or avoid writing the bare word in
   fallback text/comments in the same command.

3. **Never run two sweeps concurrently against the same server** — always
   verify `pgrep -f run_multiuser_sweep`/`locustfile`/`sweep_online` reads 0
   before starting a new one. Launch detached (`setsid nohup ... &`) so a
   dropped local `kubectl exec` doesn't leave it silently running unattended.

4. **A token-counting bug was found and fixed in `locustfile.py`**: count SSE
   chunks structurally (`is_token_step()`, checks for `content`/`choices`
   field), not by non-empty text — random-token prompts produce partial-UTF8
   fragments that detokenize to `""` and would undercount ~2x. Already fixed,
   preserve this if `locustfile.py` is ever rewritten.

5. **[NEW, this segment — the big one] Copying a venv with `cp -a` does NOT
   make it self-contained — entry-point script shebangs stay hardcoded to the
   ORIGINAL venv's interpreter.** Built `.venv_nozentorch` as
   `cp -a .venv .venv_nozentorch` then deleted the `zentorch` package from the
   copy, intending an isolated non-ZenDNN vLLM. **This silently failed**:
   `bin/vllm`'s shebang was still `#!/proj/aigstaff/sacsharm/vllm/.venv/bin/python3`
   (the *original* venv), so every server launched via
   `.venv_nozentorch/bin/vllm` actually ran on the *original* interpreter,
   which still had `zentorch` importable → `ZenCpuPlatform` was still
   selected. Two full sweeps (`vllm_nozentorch` BF16 and `vllm_nozentorch_w8a8`)
   were run and reported to the user as "non-ZenDNN" before this was caught —
   **the user asked "are you sure this ran without zendnn?" and that
   skepticism is what caught it**; don't skip verifying next time either.
   - **How it was caught**: checked the *live process's* actual invoked
     interpreter via `ps` (showed `.venv/bin/python3`, not
     `.venv_nozentorch/bin/python3`) and `grep -c 'site-packages/zentorch/'
     /proc/<pid>/maps` (nonzero = real zentorch `.so`/`.py` files mapped into
     that process's address space — this is the reliable live-process check,
     NOT just testing the venv's own python binary in isolation, which can
     look fine while the actually-launched server uses a different
     interpreter via its shebang).
   - **Pitfall while checking**: a bare `grep -c zentorch /proc/<pid>/maps`
     gave a false positive of 560 matches, because the venv itself is named
     `.venv_nozentorch` (contains the substring "zentorch"!) — every mapped
     file under that venv matched. Always grep for the actual package path
     (`site-packages/zentorch/`), not the bare word, when the venv/directory
     name itself might collide.
   - **The fix**: after `cp -a`-ing a venv, rewrite every script's shebang:
     `grep -rl '^#!/proj/aigstaff/sacsharm/vllm/[.]venv/bin/python' .venv_nozentorch/bin |
      xargs sed -i '1s#/proj/aigstaff/sacsharm/vllm/[.]venv/bin/python#/proj/aigstaff/sacsharm/vllm/.venv_nozentorch/bin/python#'`
     (59 scripts fixed). Verified after: `ps` shows the *nozentorch* venv's
     own python as the invoked interpreter, `grep -c 'site-packages/zentorch/'
     /proc/<pid>/maps` = 0, and `current_platform` resolves to `CpuPlatform`.
   - **Both affected sweeps were fully re-run after the fix** and the
     corrected numbers are what's in `results/vllm_nozentorch/summary.tsv`
     and `results/vllm_nozentorch_w8a8/summary.tsv` today (see numbers below;
     the old contaminated json files were `rm -rf`'d before re-running, so
     there's no stale contaminated data sitting in these directories).
   - llama.cpp is NOT affected by any version of this bug — `build_release`
     is a real separately-compiled binary, not a venv/shebang situation.

## What's built

**Multi-user harness (this directory, `online_bench/multiuser/`)**:
- `start_servers.sh` / `stop_servers.sh` — multi-instance launcher, reads
  `configs/*.conf`, carves the 32-core pool into N disjoint slices, detached,
  self-exits after health-checking, writes `run/<config>/manifest.json`.
- `start_haproxy.sh` / `stop_haproxy.sh` — weighted-round-robin LB from the
  manifest (weight = core count), on the support pod.
- `locustfile.py` — load profile: 1024-token random-token-id prompts,
  128-token completions, `constant_pacing(PACING_S)` per user — **default
  changed 2026-09-24 from 60s to 30s** (`llamacpp_1x32`/`vllm_1x32`/
  `vllm_1x32_noeager` results used 60s; every sweep after that date used 30s
  — see "Request duty cycle" note in README.md, rewritten for the 30s case).
- `run_multiuser_sweep.sh` — sweeps `--users "2 4 8 12 16"`, 300s/run, 30s
  cooldown, snapshots vLLM Prometheus `/metrics` before/after each run.
- `monitor_cores.sh` / `analyze_coreusage.py` — **new this segment**, see
  gotcha #1.
- `prom_metrics.py`, `consolidate_multiuser.py`, `stage_models.sh`,
  `diag_affinity.sh` — unchanged from before, see prior sections below /
  README.md.
- Configs: `llamacpp_{1x32,2x16,3x10,4x8,8x4}.conf`,
  `vllm_{1x32,1x32_noeager,2x16_noeager,3x10_noeager,4x8_noeager}.conf`.
  `2x16_noeager`/`3x10_noeager`/`4x8_noeager` are **new this segment**
  (`vllm_2x16.conf`/`vllm_4x8.conf` without `_noeager` in the name still
  exist but default to `ENFORCE_EAGER=1` — don't use those, use the
  `_noeager` variants, matching the established non-eager convention).

**Single-request pp-sweep tooling (`online_bench/`, parent of `multiuser/`)**:
- `bench_client.py` — single-request/single-user HTTP client. 1 warmup + N
  measured requests per pp point, random-nonsense-text prompts (fresh
  tokenization per request via `/tokenize`, full cache-defeat). Pure client,
  no opinion on server build — works against anything with a URL.
- `sweep_online.sh --backend {llamacpp,vllm} --url ... --results-dir ...` —
  sweeps the fixed pp list `1 2 4 8 16 32 48 64 96 128 256 384 512 768 1024
  1536 2048 3072 4096`, tg=1 (practical stand-in for tg=0, neither API
  accepts 0), resumable (skips existing output json).
- `consolidate_online.sh <results-dir>` — turns `online_*_<pp>.json` into a
  `pp/prompt_tok/ttft_s/tok_s` `summary.tsv`.
- These were NOT modified this segment — reused as-is for the non-ZenDNN and
  quantization comparisons (see below).

**Non-ZenDNN builds (new this segment)**:
- **llama.cpp**: `/proj/rdi/staff/sacsharm/llama.cpp/build_release` already
  existed with `GGML_ZENDNN=OFF` and every other CMake flag identical to
  `build_zendnn` — exactly what a from-scratch `cmake -B build && cmake
  --build build --config Release -j 32` would produce. Did an incremental
  `cmake --build build_release --config Release -j 32` to pick up any source
  drift (no-op if nothing changed) rather than reconfigure a fresh dir.
  `cmake`/`ninja-build` had to be `apt-get install`'d on pod-6 first (root,
  not previously present).
- **vLLM**: `.venv_nozentorch` — see gotcha #5 for the full story and fix.
  Once fixed, verified clean (no zentorch, `CpuPlatform` not `ZenCpuPlatform`)
  via live-process `/proc/<pid>/maps` inspection, not just testing the venv
  binary in isolation.
- New single-instance (1×32 core) configs in `multiuser/configs/` (used with
  `start_servers.sh` purely as a launcher, no HAProxy/Locust needed for a
  single-client sweep — `PARALLEL=1`, larger `SLOT_CTX` than the multi-user
  configs since pp goes up to 4096+tg here and the multi-user configs'
  `SLOT_CTX=2048` would truncate that):
  - `llamacpp_1x32_nozendnn.conf` (BF16 GGUF, `build_release` binary)
  - `llamacpp_1x32_nozendnn_q8.conf` (Q8_0 GGUF, `build_release` binary)
  - `vllm_1x32_nozentorch.conf` (BF16 HF, `.venv_nozentorch`, `ENFORCE_EAGER=0`)
  - `vllm_1x32_nozentorch_w8a8.conf` (w8a8 checkpoint, `.venv_nozentorch`,
    `ENFORCE_EAGER=0`)

## What's been run, and the numbers

### A. Multi-user sweeps (this directory, `results/`)

All: pp=1024/tg=128 per user, prefix cache off, WRR verified even (±2% of
target share) at every point, `monitor_cores.sh` confirmed no core collapse
for every layout tested.

| config | pacing | users tested | status |
|---|---|---|---|
| `llamacpp_1x32` | 60s | 2/4/8/12/16 | complete |
| `vllm_1x32` (eager) | 60s | 2/4/8/12/16 | complete |
| `vllm_1x32_noeager` | 60s | 2/4/8/12/16 | complete |
| `llamacpp_2x16` | 30s | 2/4/8/12/16 | complete |
| `vllm_2x16_noeager` | 30s | 2/4/8/12/16 | complete |
| `llamacpp_3x10` | 30s | 2/4/8/12/16 | complete |
| `vllm_3x10_noeager` | 30s | 2/4/8/12/16 | complete |
| `llamacpp_4x8` | 30s | 2/4/8/12 complete, **16 truncated** | **incomplete** |
| `vllm_4x8_noeager` | — | — | **not started** |

**`llamacpp_4x8` detail**: user asked to pause this sweep mid-run
("stop the 4x8 sweep for now"); `users_16`'s `samples.jsonl` has only 22
samples (expected up to ~80 at 30s pacing/300s) — a truncated/killed run, do
not treat it as a real data point. `users_2/4/8/12` each show sample counts
consistent with having fully completed their own 300s window before the
sweep was interrupted (sweep runs user-counts strictly sequentially), so
those four ARE usable if needed, but `users_16` needs a fresh re-run and
`vllm_4x8_noeager` was never even started.

**Eager A/B (1×32, 60s pacing)**: non-eager wins at every user count, gap
widening with load (TPOT 279.8ms eager vs 222.9ms non-eager at 16 users).
Non-eager (`ENFORCE_EAGER=0`) has been the standing convention for every
vLLM config created since — treat this as adopted, not just recommended.

**llama.cpp vs vLLM (non-eager), headline across every completed layout**:
roughly even at 2-4 users; vLLM pulls ahead clearly from 8 users on, and the
gap widens as core-count-per-instance shrinks (1×32 → 2×16 → 3×10). llama.cpp
saturates/degrades hard under load (e.g. 3×10: TPOT p90 spikes to 1460ms and
e2e p99 to ~199s at 16 users); vLLM degrades far more gracefully and
completes a much higher fraction of the offered load at every layout tested.

Known unresolved: **llama.cpp's 12-user point (from the original 60s-pacing
1×32 sweep) was non-monotonic vs its own 16-user point** (TPOT 539ms vs
376ms) — investigated (stable within its own run, not a warmup transient)
but never re-run fresh to check run-to-run variance. Still open.

Full tables: `results/<config>/summary_{throughput,latency,server}.tsv`.

### B. Single-request pp-sweep, non-ZenDNN / quantization (`online_bench/results/`)

All: single instance, 32 cores, single request at a time (no concurrency),
same pp list, tg=1, num_iters=3/warmup=1 (verified this matches what the
*original* ZenDNN pp-sweep used too, in `results/{llamacpp,vllm}/*.json`).

| results dir | backend | build | model | status |
|---|---|---|---|---|
| `llamacpp` | llama.cpp | ZenDNN (`build_zendnn`) | BF16 | complete (pre-existing) |
| `vllm` | vLLM | zentorch (`.venv`) | BF16 | complete (pre-existing) |
| `llamacpp_nozendnn` | llama.cpp | non-ZenDNN (`build_release`) | BF16 | complete |
| `llamacpp_nozendnn_q8` | llama.cpp | non-ZenDNN (`build_release`) | Q8_0 | complete |
| `vllm_nozentorch` | vLLM | non-zentorch (`.venv_nozentorch`, **fixed**) | BF16 | complete, re-run after gotcha #5 fix |
| `vllm_nozentorch_w8a8` | vLLM | non-zentorch (`.venv_nozentorch`, **fixed**) | w8a8 INT8 | complete, re-run after gotcha #5 fix |

**Headline numbers at pp=4096 (tok/s)**: llama.cpp BF16 ZenDNN 553.3 vs
non-ZenDNN 345.4 (**ZenDNN +60%** — real, substantial, grows with pp).
vLLM BF16 zentorch ~572 (pre-existing result, not re-verified against the
same live-process check used later — treat with slightly less confidence
than the numbers below) vs non-zentorch (fixed) 605.1 (**zentorch was
*not* a net win here** once eager/non-eager and everything else is held
equal — plausible since `ENFORCE_EAGER=0` already lets torch.compile
generate a good generic kernel). llama.cpp Q8_0 non-ZenDNN 271.2 vs its own
BF16 non-ZenDNN 345.4 (**Q8_0 is SLOWER** without ZenDNN's dequant-fused
GEMM path — only wins at pp≤64). vLLM w8a8 non-zentorch 801.6 vs its own
BF16 non-zentorch 605.1 (**w8a8 is ~30-60% faster**, real INT8×INT8 AVX512-
VNNI acceleration, peaks even higher at mid-pp: 1163.9 vs 730.2 tok/s at
pp=768).

Full tables: `online_bench/results/*/summary.tsv`.

## Currently live on the pods

**Both pods confirmed clean** (checked immediately before writing this
handoff) — no servers, no HAProxy, no monitor, no sweep processes anywhere.

## Leftover disk cruft (safe to delete, not cleaned up yet)

- `/proj/rdi/staff/sacsharm/vllm/.venv_quantize` (~5.9G) — a venv created to
  install `llmcompressor` for producing a w8a8 checkpoint locally; abandoned
  mid-install in favor of downloading the pre-quantized
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` checkpoint from HF Hub
  instead (much faster, same INT8 W8A8 compressed-tensors format). Not
  needed for anything currently. Delete if disk space matters, or leave it.
- `.venv_nozentorch` (~3.3G) — kept intentionally, this one IS still needed
  (and now fixed/verified) for any future non-zentorch vLLM run.

## Outstanding / not yet done

1. **`llamacpp_4x8` users_16 re-run + `vllm_4x8_noeager` full sweep** — configs
   exist, cores are free, just needs launching (both backends, 4×8 layout).
2. **8×4 layout** — `llamacpp_8x4.conf` exists; no `vllm_8x4*.conf` created
   yet; neither has been benchmarked.
3. **llama.cpp 12-user reproducibility re-check** (original 60s-pacing 1×32
   sweep) — still open, see above.
4. **Busy-only / prefill-only-busy cluster throughput metrics** — discussed
   at length earlier in the project, computed ad-hoc, never wired into
   `consolidate_multiuser.py` as permanent columns. A prior plan to add them
   was explicitly superseded by the user ("we will include gap time in
   cluster numbers, just do this change [pacing], ignore prev instruction")
   — so **this is deliberately not planned work anymore**, not just
   forgotten. Only revisit if explicitly asked again.
5. **vLLM server-side Prometheus metrics only read instance 0** in
   `consolidate_multiuser.py::prom_server_side()` — a real latent bug for any
   multi-instance vLLM config's `summary_server.tsv` (silently reports half+
   the cluster as if it were the whole thing). Known, not fixed (was
   planned once, then explicitly superseded per point 4 above alongside the
   busy-time metrics work). Fix before trusting `vllm_2x16_noeager`'s or
   `vllm_3x10_noeager`'s server-side table `n` counts.

## Quick reference: how to resume a sweep cleanly

```bash
# 1. Make sure nothing stale is running (both pods)
kubectl exec <pod> -n zendnn -- bash -c "pgrep -u \$(id -u) -f 'llama-server|vllm|haproxy|locustfile|run_multiuser_sweep|sweep_online'"

# 2. Inference pod: stage models if /tmp was wiped, then start servers
./stage_models.sh   # only if /tmp/models is empty
./stop_servers.sh --all
./start_servers.sh configs/<name>.conf --wait-secs 900

# 3. (multi-user only) Support pod: HAProxy, then sweep (detached!)
./stop_haproxy.sh --all
./start_haproxy.sh <name>
setsid nohup ./run_multiuser_sweep.sh <name> --users "2 4 8 12 16" --duration 300 \
  > run/sweep_<name>.log 2>&1 < /dev/null &

# 3b. (single-request pp-sweep instead) from online_bench/, no HAProxy needed
setsid nohup ./sweep_online.sh --backend <b> --url http://<pod6-ip>:<port> \
  --results-dir results/<name> > /tmp/sweep_<name>.log 2>&1 < /dev/null &

# 4. Poll (don't hold a blocking kubectl exec open for minutes)
kubectl exec <pod> -n zendnn -- bash -c "pgrep -u \$(id -u) -f run_multiuser_sweep | wc -l"

# 5. Core-usage sanity check alongside any new run (see gotcha #1)
setsid nohup ./monitor_cores.sh <lo>-<hi> 5 <out.csv> & 
# ... fire a request or let the sweep run ...
python3 analyze_coreusage.py <out.csv> run/<name>/manifest.json

# 6. Consolidate (use setpriv form, not login heredoc, per gotcha #2)
python3 consolidate_multiuser.py results/<name>          # multi-user
./consolidate_online.sh results/<name>                    # single-request
```
