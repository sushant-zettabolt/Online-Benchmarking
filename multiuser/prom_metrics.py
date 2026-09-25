#!/usr/bin/env python3
"""
Snapshot / diff a Prometheus /metrics endpoint so a benchmark run gets
SERVER-SIDE latency numbers to sit beside the client-side ones.

  python3 prom_metrics.py snapshot http://127.0.0.1:8000/metrics out.prom
  python3 prom_metrics.py diff before.prom after.prom            # -> json on stdout

Why this exists
---------------
llama.cpp returns per-request server timings in the response body
(timings.prompt_ms / predicted_ms), so its server-side view is free. vLLM's
OpenAI endpoint returns nothing, but vLLM exports Prometheus histograms that
break the request down further than llama.cpp does:

  vllm:request_queue_time_seconds    time waiting before the scheduler ran it
  vllm:request_prefill_time_seconds  prefill compute
  vllm:request_decode_time_seconds   decode compute
  vllm:time_to_first_token_seconds   server-side TTFT (queue + prefill)
  vllm:time_per_output_token_seconds server-side TPOT

These are cumulative, so we snapshot before and after a run and diff. A
histogram diff gives an exact mean (sum/count) and bucket-interpolated
percentiles for just that run's requests.

NOTE: vLLM only populates these when stat logging is enabled -- do NOT pass
--disable-log-stats to the server.
"""
import json
import sys
import urllib.request


def fetch(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse(text):
    """-> {metric_name: {'buckets': {le: cum_count}, 'sum': float, 'count': float}}"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, val = line.rsplit(" ", 1)
            val = float(val)
        except ValueError:
            continue
        if "{" in left:
            name, labels = left.split("{", 1)
            labels = labels.rstrip("}")
        else:
            name, labels = left, ""
        base = name
        kind = None
        if name.endswith("_bucket"):
            base, kind = name[:-7], "bucket"
        elif name.endswith("_sum"):
            base, kind = name[:-4], "sum"
        elif name.endswith("_count"):
            base, kind = name[:-6], "count"
        e = out.setdefault(base, {"buckets": {}, "sum": 0.0, "count": 0.0})
        if kind == "bucket":
            le = None
            for part in labels.split(","):
                part = part.strip()
                if part.startswith("le="):
                    le = part.split("=", 1)[1].strip('"')
            if le is not None:
                try:
                    lef = float(le)
                except ValueError:
                    lef = float("inf")
                e["buckets"][lef] = e["buckets"].get(lef, 0.0) + val
        elif kind == "sum":
            e["sum"] += val
        elif kind == "count":
            e["count"] += val
    return out


def hist_pct(buckets, p, total):
    """Linear-interpolated percentile from cumulative histogram buckets."""
    if not buckets or total <= 0:
        return None
    items = sorted(buckets.items())
    target = total * p / 100.0
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= target:
            if le == float("inf"):
                return prev_le
            if c == prev_c:
                return le
            frac = (target - prev_c) / (c - prev_c)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_c = le, c
    return items[-1][0] if items[-1][0] != float("inf") else prev_le


def diff(before, after):
    res = {}
    for name, a in after.items():
        b = before.get(name, {"buckets": {}, "sum": 0.0, "count": 0.0})
        dcount = a["count"] - b["count"]
        dsum = a["sum"] - b["sum"]
        dbuckets = {}
        for le, c in a["buckets"].items():
            d = c - b["buckets"].get(le, 0.0)
            if d < 0:
                d = 0.0
            dbuckets[le] = d
        if dcount <= 0:
            continue
        res[name] = {
            "count": dcount,
            "mean": dsum / dcount if dcount else None,
            "p50": hist_pct(dbuckets, 50, dcount),
            "p90": hist_pct(dbuckets, 90, dcount),
            "p99": hist_pct(dbuckets, 99, dcount),
        }
    return res


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "snapshot":
        url, out = sys.argv[2], sys.argv[3]
        try:
            open(out, "w").write(fetch(url))
        except Exception as exc:                        # noqa: BLE001
            print(f"prom snapshot failed ({exc})", file=sys.stderr)
            open(out, "w").write("")
    elif cmd == "diff":
        b = parse(open(sys.argv[2]).read()) if len(sys.argv) > 2 else {}
        a = parse(open(sys.argv[3]).read()) if len(sys.argv) > 3 else {}
        print(json.dumps(diff(b, a), indent=2, default=str))
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
