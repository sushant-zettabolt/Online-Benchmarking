#!/usr/bin/env python3
"""
Consolidate a multi-user sweep into TSV tables.

  python3 consolidate_multiuser.py results/llamacpp_1x32
  python3 consolidate_multiuser.py results/llamacpp_1x32 results/vllm_1x32   # side by side

Reads each results/<config>/users_<N>/samples.jsonl (written by locustfile.py)
plus the HAProxy stats snapshots taken around each run.

Metric definitions
------------------
Everything is measured client-side from the streamed response, so vLLM and
llama.cpp are compared on identical terms.

  TTFT  time from request send to the first streamed token (includes queue wait)
  ITL   gap between consecutive streamed tokens; percentiles are over the
        pooled gaps from every request in the run
  TPOT  mean ITL within one request (= decode_span / (tokens-1))

  per-user throughput  what one user's own stream achieved:
        pp_tps = prompt_tokens / TTFT
        tg_tps = (tokens-1) / decode_span      ( = 1000 / TPOT )

  cluster throughput (*_wallavg)  aggregate over the whole run window:
        pp_tps = sum(prompt_tokens) / window
        tg_tps = sum(output_tokens) / window
        where window spans the first request start to the last request end.
        This is a WALL-CLOCK AVERAGE, not a saturation number: below ~1 mean
        concurrency it reads *lower* than user_tg_tps/user_pp_tps, because
        most of the window has no request in flight (the load is paced
        per-user, so idle time is real and gets averaged in). The
        identity cl_tg_tps_wallavg = (out_tok/req / e2e/req) * mean_conc holds
        exactly -- see mean_conc for how loaded the server actually was, and
        compare user_pp_tps/user_tg_tps across backends for a like-for-like
        single-stream number that isn't diluted by idle time.
"""
import json
import os
import statistics
import sys


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def mean(xs):
    return statistics.fmean(xs) if xs else None


def load_samples(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def haproxy_stot(csv_path):
    counts = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return counts
    header = lines[0].lstrip("# ").split(",")
    try:
        i_px, i_sv, i_stot = header.index("pxname"), header.index("svname"), header.index("stot")
    except ValueError:
        return counts
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(i_px, i_sv, i_stot):
            continue
        if parts[i_px] != "be_llm" or parts[i_sv] in ("BACKEND", "FRONTEND"):
            continue
        try:
            counts[parts[i_sv]] = int(parts[i_stot] or 0)
        except ValueError:
            pass
    return counts


def prom_server_side(run_dir):
    """
    Server-side latency for a run, from the Prometheus before/after snapshots.

    Only vLLM populates these. Histogram percentiles are bucket-interpolated
    and therefore coarse; the mean (sum/count) is exact, so means are what we
    report here.
    """
    before = os.path.join(run_dir, "prom_before_inst0.txt")
    after = os.path.join(run_dir, "prom_after_inst0.txt")
    if not (os.path.exists(before) and os.path.exists(after)):
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import prom_metrics
        d = prom_metrics.diff(prom_metrics.parse(open(before).read()),
                              prom_metrics.parse(open(after).read()))
    except Exception:                                   # noqa: BLE001
        return None
    if not d:
        return None

    def m(name):
        e = d.get("vllm:" + name)
        return e["mean"] if e and e.get("mean") is not None else None

    return {
        "n": (d.get("vllm:e2e_request_latency_seconds") or {}).get("count"),
        "queue_ms": (m("request_queue_time_seconds") or 0) * 1000.0
                    if m("request_queue_time_seconds") is not None else None,
        "prefill_ms": (m("request_prefill_time_seconds") or 0) * 1000.0
                      if m("request_prefill_time_seconds") is not None else None,
        "decode_s": m("request_decode_time_seconds"),
        "ttft_ms": (m("time_to_first_token_seconds") or 0) * 1000.0
                   if m("time_to_first_token_seconds") is not None else None,
        "tpot_ms": (m("request_time_per_output_token_seconds") or 0) * 1000.0
                   if m("request_time_per_output_token_seconds") is not None else None,
        "itl_ms": (m("inter_token_latency_seconds") or 0) * 1000.0
                  if m("inter_token_latency_seconds") is not None else None,
    }


def user_dirs(results_dir):
    out = []
    for name in os.listdir(results_dir):
        if name.startswith("users_") and os.path.isdir(os.path.join(results_dir, name)):
            try:
                out.append((int(name.split("_", 1)[1]), os.path.join(results_dir, name)))
            except ValueError:
                pass
    return sorted(out)


THROUGHPUT_HDR = ("users\treqs\tfail\tmean_conc\treq_per_min"
                  "\tcl_pp_tps_wallavg\tcl_tg_tps_wallavg\tcl_tot_tps_wallavg"
                  "\tuser_pp_tps\tuser_tg_tps\tprompt_tok\tout_tok")
LATENCY_HDR = ("users\tttft_p50\tttft_p90\tttft_p99\tttft_mean"
               "\ttpot_p50\ttpot_p90\ttpot_mean"
               "\titl_p50\titl_p90\titl_p99\titl_mean"
               "\te2e_p50\te2e_p90\te2e_p99"
               "\tsrv_prefill_p50\tsrv_tpot_p50\tqueue_p50")


def process(results_dir):
    results_dir = os.path.abspath(results_dir.rstrip("/"))
    cfg = os.path.basename(results_dir)
    manifest = {}
    mpath = os.path.join(results_dir, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))

    runs = user_dirs(results_dir)
    if not runs:
        print(f"no users_* runs found under {results_dir}", file=sys.stderr)
        return

    tput_rows = [THROUGHPUT_HDR]
    lat_rows = [LATENCY_HDR]
    srv_rows = ["users\tsource\tn\tqueue_ms\tprefill_ms\tdecode_s\tttft_ms\ttpot_ms\titl_ms"]
    dist_rows = ["users\tinstance\tcores\tweight\twrr_share_pct\tserved\tserved_pct"]

    for users, d in runs:
        s = load_samples(os.path.join(d, "samples.jsonl"))
        ok = [x for x in s if x.get("ok")]
        fail = len(s) - len(ok)

        ttft = [x["ttft_s"] for x in ok if x.get("ttft_s") is not None]
        e2e = [x["e2e_s"] for x in ok if x.get("e2e_s") is not None]
        tpot = [x["tpot_ms"] for x in ok if x.get("tpot_ms") is not None]
        itl = [v for x in ok for v in (x.get("itl_ms") or [])]
        upp = [x["pp_tps_user"] for x in ok if x.get("pp_tps_user")]
        utg = [x["tg_tps_user"] for x in ok if x.get("tg_tps_user")]
        ptok = [x["prompt_tokens"] for x in ok if x.get("prompt_tokens")]
        otok = [x["out_tokens"] for x in ok if x.get("out_tokens")]

        if ok:
            window = max(x["wall_end"] for x in ok) - min(x["wall_start"] for x in ok)
        else:
            window = 0.0
        # NOTE: these are wall-clock averages over the whole run window,
        # including any time a user was idle between its paced requests --
        # NOT a saturation/capacity number. At low concurrency (mean_conc < 1)
        # this reads *lower* than user_tg_tps/user_pp_tps because most of the
        # window has 0 requests in flight; that crossover is expected, not a
        # bug (see identity below). Compare user_*_tps across backends for a
        # like-for-like single-stream number; use these + mean_conc together
        # to see whether the offered load is actually being sustained.
        #   cl_tg_tps_wallavg == (out_tok/req / e2e/req) * mean_conc   (exact)
        sum_p = sum(ptok)
        sum_o = sum(otok)
        cl_pp = sum_p / window if window > 0 else None
        cl_tg = sum_o / window if window > 0 else None
        cl_tot = (sum_p + sum_o) / window if window > 0 else None
        req_min = len(ok) / window * 60.0 if window > 0 else None
        mean_conc = sum(e2e) / window if window > 0 else None

        tput_rows.append("\t".join([
            str(users), str(len(ok)), str(fail),
            fmt(mean_conc, 2), fmt(req_min, 2),
            fmt(cl_pp, 1), fmt(cl_tg, 2), fmt(cl_tot, 1),
            fmt(mean(upp), 1), fmt(mean(utg), 2),
            str(int(statistics.median(ptok))) if ptok else "-",
            str(int(statistics.median(otok))) if otok else "-",
        ]))

        # Server-reported split, when the backend provides one (llama.cpp does;
        # vLLM's OpenAI endpoint does not -> these read "-"). srv_prefill is
        # pure prefill compute measured inside the server, so
        #   queue = client TTFT - server prefill
        # is the request's queue wait + network (network measured at ~0.37 ms
        # through HAProxy, i.e. negligible). This is exactly what server-side
        # timings would hide if we reported them instead of client-side.
        srv_pref = [x["srv_prompt_ms"] for x in ok if x.get("srv_prompt_ms") is not None]
        srv_tpot = [x["srv_predicted_ms"] / (x["out_tokens"] - 1)
                    for x in ok
                    if x.get("srv_predicted_ms") is not None and (x.get("out_tokens") or 0) > 1]
        queue = [x["ttft_s"] * 1000.0 - x["srv_prompt_ms"]
                 for x in ok
                 if x.get("srv_prompt_ms") is not None and x.get("ttft_s") is not None]

        lat_rows.append("\t".join([
            str(users),
            fmt(pct(ttft, 50)), fmt(pct(ttft, 90)), fmt(pct(ttft, 99)), fmt(mean(ttft)),
            fmt(pct(tpot, 50), 1), fmt(pct(tpot, 90), 1), fmt(mean(tpot), 1),
            fmt(pct(itl, 50), 1), fmt(pct(itl, 90), 1), fmt(pct(itl, 99), 1), fmt(mean(itl), 1),
            fmt(pct(e2e, 50), 2), fmt(pct(e2e, 90), 2), fmt(pct(e2e, 99), 2),
            fmt(pct(srv_pref, 50), 1), fmt(pct(srv_tpot, 50), 1), fmt(pct(queue, 50), 1),
        ]))

        # ---- server-side view -------------------------------------------
        prom = prom_server_side(d)
        if prom and prom.get("n"):
            srv_rows.append("\t".join([
                str(users), "vllm:prometheus", f"{prom['n']:.0f}",
                fmt(prom["queue_ms"], 1), fmt(prom["prefill_ms"], 1),
                fmt(prom["decode_s"], 2), fmt(prom["ttft_ms"], 1),
                fmt(prom["tpot_ms"], 1), fmt(prom["itl_ms"], 1),
            ]))
        elif srv_pref:
            # llama.cpp: per-request timings from the response body
            dec = [x["srv_predicted_ms"] / 1000.0 for x in ok
                   if x.get("srv_predicted_ms") is not None]
            srv_rows.append("\t".join([
                str(users), "llamacpp:timings", str(len(srv_pref)),
                fmt(pct(queue, 50), 1), fmt(pct(srv_pref, 50), 1),
                fmt(pct(dec, 50), 2), "-",
                fmt(pct(srv_tpot, 50), 1), fmt(pct(srv_tpot, 50), 1),
            ]))

        before = haproxy_stot(os.path.join(d, "haproxy_before.csv"))
        after = haproxy_stot(os.path.join(d, "haproxy_after.csv"))
        served = {k: after.get(k, 0) - before.get(k, 0) for k in after} if after else {}
        if not served:
            served = {}
            for x in ok:
                sb = x.get("served_by")
                if sb:
                    served[sb] = served.get(sb, 0) + 1
        total_served = sum(served.values())
        insts = manifest.get("instances") or []
        wsum = sum(i["weight"] for i in insts) or 1
        for i in insts:
            name = f"inst{i['idx']}"
            n = served.get(name, 0)
            dist_rows.append("\t".join([
                str(users), name, i["cores"], str(i["weight"]),
                f"{100.0 * i['weight'] / wsum:.1f}", str(n),
                f"{100.0 * n / total_served:.1f}" if total_served else "-",
            ]))

    title = (f"{cfg}  ({manifest.get('backend','?')}, "
             f"{manifest.get('instances_n','?')} x {manifest.get('cores_per_instance','?')} cores)")
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print("\n-- THROUGHPUT (cl_* = whole cluster, user_* = single user's own stream) --")
    print("\n".join(tput_rows))
    print("\n-- LATENCY (ttft/e2e in seconds, tpot/itl in ms) --")
    print("\n".join(lat_rows))

    if len(srv_rows) > 1:
        print("\n-- SERVER-SIDE (vLLM: Prometheus histogram means; llama.cpp: per-request response timings, medians) --")
        print("\n".join(srv_rows))
        with open(os.path.join(results_dir, "summary_server.tsv"), "w") as f:
            f.write("\n".join(srv_rows) + "\n")

    out_t = os.path.join(results_dir, "summary_throughput.tsv")
    out_l = os.path.join(results_dir, "summary_latency.tsv")
    with open(out_t, "w") as f:
        f.write("\n".join(tput_rows) + "\n")
    with open(out_l, "w") as f:
        f.write("\n".join(lat_rows) + "\n")
    written = [out_t, out_l]

    if len(dist_rows) > 1 and len(manifest.get("instances") or []) > 1:
        print("\n-- weighted-round-robin distribution --")
        print("\n".join(dist_rows))
        dout = os.path.join(results_dir, "distribution.tsv")
        with open(dout, "w") as f:
            f.write("\n".join(dist_rows) + "\n")
        written.append(dout)

    print("\nwritten: " + "\n         ".join(written), file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        process(path)
