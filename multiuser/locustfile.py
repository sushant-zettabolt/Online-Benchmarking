#!/usr/bin/env python3
"""
Locust load profile for multi-user LLM serving benchmarks (runs on the support
pod, turin-xcovoid0021-pod-5, against HAProxy on localhost).

Load shape (per the benchmark spec):
  * every simulated user sends a PROMPT of BENCH_PP tokens (default 1024)
  * and asks for BENCH_TG generated tokens (default 128)
  * at a fixed rate of ONE REQUEST EVERY 30 SECONDS PER USER (constant_pacing)
  * user counts are swept separately: 2, 4, 8, 12, 16 (see run_multiuser_sweep.sh)
  * each run lasts BENCH_DURATION seconds (default 300 = 5 min)

Cache defeat
------------
Both backends will happily make a repeated (or prefix-sharing) prompt look
artificially fast -- llama.cpp via its per-slot prompt cache, vLLM via
automatic prefix caching. Rather than round-tripping /tokenize for every
request (which would inject extra traffic into HAProxy and skew the weighted
round-robin distribution we are trying to measure), every request builds its
prompt as BENCH_PP *freshly drawn random token ids* from a safe, non-special
slice of the vocabulary. Two requests sharing even a 2-token prefix is already
a ~1e-10 event, so no KV block can ever be reused across requests, users, or
runs. The server-reported prompt token count is recorded on every sample so
the 1024 figure can be verified rather than assumed.

Phase jitter
------------
Users are spawned a second or so apart, which would otherwise leave all N
users firing inside the same ~N-second window of every 60s period -- a
synchronised burst, not N independent users. With BENCH_PHASE_JITTER=1
(default) each user sleeps a uniform [0, pacing) before its first request, so
arrivals spread across the whole minute the way independent users would.

Metrics
-------
Streaming is used throughout so TTFT is measured properly. Every sample is
appended to BENCH_JSONL (ttft, end-to-end, TPOT, token counts, which HAProxy
backend served it); consolidate_multiuser.py turns those into percentiles.
Locust's own stats get three entries: POST/completion (end-to-end ms),
TTFT/completion, and TPOT/completion.
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

from locust import User, constant_pacing, events, task

# ------------------------------------------------------------------ config --
BACKEND      = os.environ.get("BENCH_BACKEND", "llamacpp")
MODEL        = os.environ.get("BENCH_MODEL", "default")
PP           = int(os.environ.get("BENCH_PP", "1024"))
TG           = int(os.environ.get("BENCH_TG", "128"))
PACING_S     = float(os.environ.get("BENCH_PACING_S", "30"))
TIMEOUT_S    = float(os.environ.get("BENCH_TIMEOUT_S", "600"))
JSONL_PATH   = os.environ.get("BENCH_JSONL", "")
PHASE_JITTER = os.environ.get("BENCH_PHASE_JITTER", "1") == "1"
RUN_LABEL    = os.environ.get("BENCH_RUN_LABEL", "")

# Llama-3.1 vocab is 0..128255 with 128000+ reserved for special tokens.
# Staying inside [10, 127000] keeps every drawn id an ordinary text token.
VOCAB_LO = int(os.environ.get("BENCH_VOCAB_LO", "10"))
VOCAB_HI = int(os.environ.get("BENCH_VOCAB_HI", "127000"))

_jsonl_lock = threading.Lock()
_jsonl_fh = None


@events.test_start.add_listener
def _open_jsonl(environment, **_kw):
    global _jsonl_fh
    if JSONL_PATH:
        os.makedirs(os.path.dirname(JSONL_PATH) or ".", exist_ok=True)
        _jsonl_fh = open(JSONL_PATH, "a", buffering=1)


@events.test_stop.add_listener
def _close_jsonl(environment, **_kw):
    global _jsonl_fh
    if _jsonl_fh:
        _jsonl_fh.flush()
        _jsonl_fh.close()
        _jsonl_fh = None


def _emit(sample):
    if _jsonl_fh is None:
        return
    with _jsonl_lock:
        _jsonl_fh.write(json.dumps(sample) + "\n")


def random_prompt_tokens(n):
    return [random.randint(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def build_payload(prompt_tokens):
    if BACKEND == "llamacpp":
        return "/completion", {
            "prompt": prompt_tokens,
            "n_predict": TG,
            "temperature": 0,
            "cache_prompt": False,
            "ignore_eos": True,     # always generate exactly TG tokens
            "stream": True,
        }
    return "/v1/completions", {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": TG,
        "min_tokens": TG,          # keep every sample the same generation length
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def is_token_step(chunk):
    """
    True if this SSE chunk represents one generated token.

    Deliberately NOT keyed on the chunk's text being non-empty. The prompts are
    random token ids, so a large share of generated tokens are partial-UTF-8
    byte fragments that the server's detokenizer cannot emit as text yet -- it
    buffers them and returns "" for that step. Counting only non-empty text
    undercounted vLLM by ~2x (128 reported tokens, 64 non-empty chunks), which
    would double every ITL/TPOT and halve per-user tg throughput.
    """
    if BACKEND == "llamacpp":
        # every streamed chunk carries a content field; the terminal chunk
        # (stop=true) is bookkeeping, not an extra token
        return "content" in chunk and not chunk.get("stop")
    # vLLM: a chunk with a choices entry is one decode step; the final
    # usage-only chunk has choices == []
    return bool(chunk.get("choices"))


class LLMUser(User):
    """One simulated end user: 1024-token prompt, 128-token completion, 1 req/30s."""

    wait_time = constant_pacing(PACING_S)

    def on_start(self):
        self.user_seq = 0
        if PHASE_JITTER:
            # de-synchronise this user's minute relative to the others
            time.sleep(random.uniform(0, PACING_S))

    @task
    def completion(self):
        path, payload = build_payload(random_prompt_tokens(PP))
        url = self.host.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )

        wall_start = time.time()
        t0 = time.perf_counter()
        tok_times = []            # perf_counter at each streamed token chunk
        prompt_tokens = None
        out_tokens = None
        srv_prompt_ms = None
        srv_predicted_ms = None
        served_by = None
        err = None

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                served_by = resp.headers.get("X-Served-By")
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[len("data:"):].strip()
                    if body == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    if is_token_step(chunk):
                        tok_times.append(time.perf_counter())

                    # authoritative counts / server-side split from the backend
                    timings = chunk.get("timings")
                    if timings:
                        prompt_tokens = timings.get("prompt_n", prompt_tokens)
                        out_tokens = timings.get("predicted_n", out_tokens)
                        srv_prompt_ms = timings.get("prompt_ms", srv_prompt_ms)
                        srv_predicted_ms = timings.get("predicted_ms", srv_predicted_ms)
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        out_tokens = usage.get("completion_tokens", out_tokens)
        except Exception as exc:                        # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        e2e = time.perf_counter() - t0
        chunk_tokens = len(tok_times)
        if out_tokens is None:
            out_tokens = chunk_tokens
        ttft = (tok_times[0] - t0) if tok_times else e2e

        # Inter-token latencies: the gaps between consecutive streamed tokens.
        # TPOT is their mean; ITL percentiles come from the pooled gaps.
        itl_ms = [(tok_times[i] - tok_times[i - 1]) * 1000.0 for i in range(1, len(tok_times))]
        decode_s = (tok_times[-1] - tok_times[0]) if len(tok_times) > 1 else 0.0
        tpot_ms = (decode_s / len(itl_ms) * 1000.0) if itl_ms else None

        # per-user (single-stream) throughputs, as this user experienced them
        pp_tps_user = (prompt_tokens / ttft) if (prompt_tokens and ttft > 0) else None
        tg_tps_user = (len(itl_ms) / decode_s) if decode_s > 0 else None

        self.user_seq += 1
        _emit({
            "run_label": RUN_LABEL,
            "backend": BACKEND,
            "users": self.environment.runner.user_count if self.environment.runner else None,
            "wall_start": wall_start,
            "wall_end": time.time(),
            "seq": self.user_seq,
            "ok": err is None,
            "error": err,
            "pp_requested": PP,
            "tg_requested": TG,
            "prompt_tokens": prompt_tokens,
            "out_tokens": out_tokens,
            "chunk_tokens": chunk_tokens,
            "ttft_s": ttft,
            "e2e_s": e2e,
            "decode_s": decode_s,
            "tpot_ms": tpot_ms,
            "itl_ms": itl_ms,
            "pp_tps_user": pp_tps_user,
            "tg_tps_user": tg_tps_user,
            "srv_prompt_ms": srv_prompt_ms,
            "srv_predicted_ms": srv_predicted_ms,
            "served_by": served_by,
        })

        exc = Exception(err) if err else None
        events.request.fire(request_type="POST", name="completion",
                            response_time=e2e * 1000.0, response_length=out_tokens or 0,
                            exception=exc, context={})
        if err is None:
            events.request.fire(request_type="TTFT", name="completion",
                                response_time=ttft * 1000.0, response_length=0,
                                exception=None, context={})
            if tpot_ms is not None:
                events.request.fire(request_type="TPOT", name="completion",
                                    response_time=tpot_ms, response_length=0,
                                    exception=None, context={})
