#!/usr/bin/env python3
"""
Single-request online-serving benchmark client, backend-agnostic (vllm /
llama.cpp server). Measures one (pp, tg) point: 1 warmup request (discarded)
then N measured requests (default 1), single user / single request at a time
(no concurrency) so numbers reflect pure per-request serving latency.

Prompt-cache defeat: both llama.cpp (--cache-prompt) and vLLM (automatic
prefix caching, on by default in V1) will silently make a *repeated*
identical prompt look artificially fast on the 2nd+ request -- and this
applies across different requests that merely *share a prefix*, not just
byte-identical ones. A naive "slice a window out of one repeated filler
phrase" scheme is NOT enough: the tokenized filler is periodic (repeats
every ~39 tokens for a short phrase), so different offsets alias onto the
same content, and content from a smaller pp point becomes a literal prefix
of a larger one, letting cached KV blocks leak across both iterations of
the same pp *and* across different pp points in the sweep. (Verified live:
10 random offset draws out of a 256-token window produced only 8 distinct
sequences -- 2 were byte-identical.)

The fix here: every single request (including every warmup) gets its own
freshly generated, independently-randomized nonsense text (random
lowercase letter blocks) tokenized on the fly via the server's own
/tokenize endpoint. No two requests -- same pp or different pp, same run or
a different run entirely -- can ever share a cacheable prefix.
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.request
import urllib.error


def http_post_json(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def http_post_stream(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            yield time.perf_counter(), payload_str


def random_nonsense_text(approx_chars):
    words = []
    remaining = approx_chars
    while remaining > 0:
        wlen = random.randint(3, 9)
        words.append("".join(random.choices(string.ascii_lowercase, k=wlen)))
        remaining -= wlen + 1
    return " ".join(words)


def tokenize(base_url, backend, model, text):
    if backend == "llamacpp":
        resp = http_post_json(f"{base_url}/tokenize", {"content": text, "add_special": False})
    else:
        resp = http_post_json(
            f"{base_url}/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}
        )
    return resp["tokens"]


def get_prompt_tokens(base_url, backend, model, pp):
    approx_chars = pp * 6 + 64
    for _ in range(6):
        text = random_nonsense_text(approx_chars)
        tokens = tokenize(base_url, backend, model, text)
        if len(tokens) >= pp:
            return tokens[:pp]
        approx_chars *= 2
    raise RuntimeError(f"could not generate a prompt with >= {pp} tokens")


def run_llamacpp_request(base_url, prompt_tokens, tg):
    payload = {
        "prompt": prompt_tokens,
        "n_predict": tg,
        "temperature": 0,
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = http_post_json(f"{base_url}/completion", payload)
    t1 = time.perf_counter()

    timings = resp.get("timings", {})
    return {
        "client_wall_s": t1 - t0,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "ttft_s": (timings["prompt_ms"] / 1000.0) if timings.get("prompt_ms") is not None else None,
    }


def run_vllm_request(base_url, model, prompt_tokens, tg):
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": tg,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    usage = None
    for t_chunk, chunk_str in http_post_stream(f"{base_url}/v1/completions", payload):
        t_last = t_chunk
        if chunk_str == "[DONE]":
            continue
        try:
            chunk = json.loads(chunk_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        has_text = bool(choices) and choices[0].get("text")
        if has_text and t_first is None:
            t_first = t_chunk
        if chunk.get("usage") is not None:
            usage = chunk["usage"]

    if t_first is None:
        t_first = t_last

    return {
        "client_wall_s": (t_last - t0) if t_last else None,
        "ttft_s": (t_first - t0) if t_first else None,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["vllm", "llamacpp"])
    ap.add_argument("--url", required=True, help="server base url, e.g. http://192.168.7.40:8000")
    ap.add_argument("--model", default="default", help="model field for vllm's OpenAI request body")
    ap.add_argument("--pp", type=int, required=True, help="prompt token count")
    ap.add_argument("--tg", type=int, default=1, help="tokens to generate (min 1; true 0 unsupported by either API)")
    ap.add_argument("--num-iters", type=int, default=3, help="measured iterations, averaged")
    ap.add_argument("--warmup-iters", type=int, default=1, help="warmup iterations, discarded")
    ap.add_argument("--seed", type=int, default=None, help="random seed for nonsense-text generation")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    base_url = args.url.rstrip("/")

    def do_request():
        prompt_tokens = get_prompt_tokens(base_url, args.backend, args.model, args.pp)
        if args.backend == "llamacpp":
            return run_llamacpp_request(base_url, prompt_tokens, args.tg)
        return run_vllm_request(base_url, args.model, prompt_tokens, args.tg)

    for i in range(args.warmup_iters):
        print(f"[warmup {i+1}/{args.warmup_iters}] pp={args.pp} tg={args.tg}", file=sys.stderr)
        do_request()

    iters = []
    for i in range(args.num_iters):
        print(f"[measure {i+1}/{args.num_iters}] pp={args.pp} tg={args.tg}", file=sys.stderr)
        iters.append(do_request())

    ttfts = [it["ttft_s"] for it in iters if it.get("ttft_s") is not None]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else None

    result = {
        "backend": args.backend,
        "url": base_url,
        "model": args.model,
        "pp": args.pp,
        "tg": args.tg,
        "num_iters": args.num_iters,
        "warmup_iters": args.warmup_iters,
        "iters": iters,
        "mean_ttft_s": mean_ttft,
        "tok_s": (args.pp / mean_ttft) if mean_ttft else None,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
