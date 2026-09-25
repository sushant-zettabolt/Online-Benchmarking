#!/usr/bin/env bash
# Extends multiuser/monitor_cores.sh: per-core CPU busy% (same /proc/stat method)
# PLUS system RAM (/proc/meminfo) PLUS per-process RSS of a given PID PLUS a
# phase label, sampled together at a fixed interval, for as long as this
# process lives. Meant to run detached (setsid nohup ... &) on the INFERENCE
# pod spanning before/during/after a job (see run_job.sh), so the whole
# lifecycle (idle baseline -> server up+load -> idle baseline again) is one
# continuous, easy-to-graph CSV.
#
# Phase is read from a small marker file each sample tick (default "unknown"
# if missing/empty) so the caller can flip phases (before/during/after)
# without restarting this process: `echo during > <phasefile>`. The PID to
# track for RSS is read the same way from <pidfile> each tick (empty/missing
# until the caller knows it -- e.g. the server PID isn't known until AFTER
# start_servers.sh returns, which is after this monitor already started its
# "before" baseline) so it can be filled in later without restarting either.
#
#   ./monitor_resources.sh <lo>-<hi> <interval_s> <out.csv> <phasefile> <pidfile>
#
# CSV columns: ts,phase,cpu<lo>,...,cpu<hi>,mem_used_mb,mem_total_mb,proc_rss_mb
# proc_rss_mb is 0 if pidfile is empty/missing or the PID isn't running.
set -uo pipefail

RANGE="${1:?usage: monitor_resources.sh <lo>-<hi> <interval_s> <out.csv> <phasefile> <pidfile>}"
INTERVAL="${2:-2}"
OUT="${3:?usage: monitor_resources.sh <lo>-<hi> <interval_s> <out.csv> <phasefile> <pidfile>}"
PHASEFILE="${4:?usage: monitor_resources.sh <lo>-<hi> <interval_s> <out.csv> <phasefile> <pidfile>}"
PIDFILE="${5:?usage: monitor_resources.sh <lo>-<hi> <interval_s> <out.csv> <phasefile> <pidfile>}"
[ -f "$PIDFILE" ] || : > "$PIDFILE"

LO="${RANGE%-*}"
HI="${RANGE#*-}"

mkdir -p "$(dirname "$OUT")"
[ -f "$PHASEFILE" ] || echo "before" > "$PHASEFILE"

{
  printf "ts,phase"
  for ((c = LO; c <= HI; c++)); do printf ",cpu%s" "$c"; done
  printf ",mem_used_mb,mem_total_mb,proc_rss_mb\n"
} > "$OUT"

read_stat() {
  awk -v lo="$LO" -v hi="$HI" '
    $1 ~ /^cpu[0-9]+$/ {
      n = substr($1, 4) + 0
      if (n >= lo && n <= hi) {
        idle = $5 + $6
        total = 0
        for (i = 2; i <= 11; i++) total += $i
        print n, total, idle
      }
    }' /proc/stat
}

read_mem() {
  awk '
    /^MemTotal:/     { total = $2 }
    /^MemAvailable:/ { avail = $2 }
    END { printf "%.0f,%.0f", (total-avail)/1024, total/1024 }
  ' /proc/meminfo
}

read_rss() {
  local pid="$1"
  [ -n "$pid" ] || { echo 0; return; }
  awk '/^VmRSS:/ { printf "%.0f", $2/1024; found=1 } END { if (!found) print 0 }' "/proc/$pid/status" 2>/dev/null || echo 0
}

declare -A PREV_TOTAL PREV_IDLE CUR

while read -r n t i; do
  PREV_TOTAL[$n]=$t
  PREV_IDLE[$n]=$i
done < <(read_stat)

while true; do
  sleep "$INTERVAL"
  ts=$(date +%s)
  phase="$(cat "$PHASEFILE" 2>/dev/null || echo unknown)"
  [ -n "$phase" ] || phase="unknown"

  while read -r n t i; do
    pt=${PREV_TOTAL[$n]:-$t}
    pi=${PREV_IDLE[$n]:-$i}
    dt=$((t - pt))
    di=$((i - pi))
    if [ "$dt" -gt 0 ]; then
      CUR[$n]=$(awk -v dt="$dt" -v di="$di" 'BEGIN{printf "%.1f", 100*(dt-di)/dt}')
    else
      CUR[$n]="0.0"
    fi
    PREV_TOTAL[$n]=$t
    PREV_IDLE[$n]=$i
  done < <(read_stat)

  line="$ts,$phase"
  for ((c = LO; c <= HI; c++)); do
    line="$line,${CUR[$c]:-0.0}"
  done
  mem="$(read_mem)"
  track_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  rss="$(read_rss "$track_pid")"
  echo "$line,$mem,$rss" >> "$OUT"
done
