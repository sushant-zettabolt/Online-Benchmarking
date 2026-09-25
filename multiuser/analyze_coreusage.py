#!/usr/bin/env python3
"""
Summarize a monitor_cores.sh CSV: mean/min/max busy% per core over the whole
sweep, grouped by instance (from manifest.json), with a pass/fail read on
whether threads spread across every pinned core or collapsed onto one.

  python3 analyze_coreusage.py run/llamacpp_2x16/coreusage.csv run/llamacpp_2x16/manifest.json
"""
import csv
import json
import statistics
import sys


def main(csv_path, manifest_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    cores = [int(h[3:]) for h in header[1:]]
    samples = {c: [] for c in cores}
    for r in rows:
        for c, v in zip(cores, r[1:]):
            try:
                samples[c].append(float(v))
            except ValueError:
                pass

    manifest = json.load(open(manifest_path))
    insts = manifest.get("instances") or [{"idx": 0, "cores": f"{min(cores)}-{max(cores)}"}]

    print(f"{len(rows)} samples over {cores[0]}-{cores[-1]} ({len(cores)} cores)\n")
    bad = []
    for inst in insts:
        lo, hi = (int(x) for x in inst["cores"].split("-"))
        print(f"-- inst{inst['idx']} cores {lo}-{hi} --")
        print("core\tmean%\tmin%\tmax%")
        for c in range(lo, hi + 1):
            xs = samples.get(c, [])
            if not xs:
                print(f"{c}\tNO DATA")
                continue
            m, mn, mx = statistics.fmean(xs), min(xs), max(xs)
            flag = " <-- LOW (possible collapse)" if m < 5.0 else ""
            if flag:
                bad.append(c)
            print(f"{c}\t{m:.1f}\t{mn:.1f}\t{mx:.1f}{flag}")
        print()

    if bad:
        print(f"WARNING: cores with <5% mean utilization (likely idle/collapsed): {bad}")
    else:
        print("OK: every pinned core shows >=5% mean utilization -- no single-core collapse.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
