#!/usr/bin/env python3
"""
Summarize a monitor_resources.sh CSV (per-core CPU busy% + RAM, phase-tagged):
mean/min/max busy% per core grouped by instance (from manifest.json) and by
phase (before/during/after), flags any pinned core with <5% mean busy% DURING
the active phase (possible collapse -- before/after are expected to be near
0% by design, so the collapse check only applies to the "during" phase), and
reports RAM (system used + tracked process RSS) per phase for a sanity check
that the server actually loaded/held memory only during "during".

  python3 analyze_resources.py <csv> <manifest.json>
"""
import csv
import json
import statistics
import sys


def main(csv_path, manifest_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    core_cols = [h for h in header if h.startswith("cpu")]
    cores = [int(h[3:]) for h in core_cols]
    core_idx = {h: i for i, h in enumerate(header)}

    by_phase = {}
    for r in rows:
        phase = r[core_idx["phase"]]
        by_phase.setdefault(phase, []).append(r)

    manifest = json.load(open(manifest_path))
    insts = manifest.get("instances") or [{"idx": 0, "cores": f"{min(cores)}-{max(cores)}"}]

    print(f"{len(rows)} samples over {cores[0]}-{cores[-1]} ({len(cores)} cores), "
          f"phases: {', '.join(f'{p}={len(v)}' for p, v in by_phase.items())}\n")

    bad = []
    during_rows = by_phase.get("during", [])
    if not during_rows:
        print("WARNING: no 'during' phase samples found -- can't check for core collapse.")
    else:
        for inst in insts:
            lo, hi = (int(x) for x in inst["cores"].split("-"))
            print(f"-- inst{inst['idx']} cores {lo}-{hi} (during phase only) --")
            print("core\tmean%\tmin%\tmax%")
            for c in range(lo, hi + 1):
                col = f"cpu{c}"
                if col not in core_idx:
                    print(f"{c}\tNO DATA")
                    continue
                xs = [float(r[core_idx[col]]) for r in during_rows if r[core_idx[col]]]
                if not xs:
                    print(f"{c}\tNO DATA")
                    continue
                m, mn, mx = statistics.fmean(xs), min(xs), max(xs)
                flag = " <-- LOW (possible collapse)" if m < 5.0 else ""
                if flag:
                    bad.append(c)
                print(f"{c}\t{m:.1f}\t{mn:.1f}\t{mx:.1f}{flag}")
            print()

    print("-- RAM by phase --")
    print("phase\tn\tmem_used_mb(mean)\tproc_rss_mb(mean)\tproc_rss_mb(max)")
    for phase, prows in by_phase.items():
        used = [float(r[core_idx["mem_used_mb"]]) for r in prows if r[core_idx["mem_used_mb"]]]
        rss = [float(r[core_idx["proc_rss_mb"]]) for r in prows if r[core_idx["proc_rss_mb"]]]
        used_m = statistics.fmean(used) if used else 0.0
        rss_m = statistics.fmean(rss) if rss else 0.0
        rss_x = max(rss) if rss else 0.0
        print(f"{phase}\t{len(prows)}\t{used_m:.0f}\t{rss_m:.0f}\t{rss_x:.0f}")

    print()
    if bad:
        print(f"WARNING: cores with <5% mean utilization during 'during' phase (likely idle/collapsed): {sorted(set(bad))}")
    else:
        print("OK: every pinned core shows >=5% mean utilization during the active phase -- no single-core collapse.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
