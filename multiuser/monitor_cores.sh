#!/usr/bin/env bash
# Sample per-core CPU busy% from /proc/stat over a core range, at a fixed
# interval, for as long as this process lives. Meant to run detached
# (setsid nohup ... &) alongside a sweep on the inference pod, so a full
# multi-instance run can be checked afterwards for the isolcpus core-collapse
# failure mode (see HANDOFF.md gotcha #1): every pinned core should show
# meaningful busy% while a sweep is in flight, not just the first core of
# each instance's range.
#
#   ./monitor_cores.sh 192-223 5 run/llamacpp_2x16/coreusage.csv &
#   echo $! > run/llamacpp_2x16/monitor.pid
#   ...
#   kill $(cat run/llamacpp_2x16/monitor.pid)
set -uo pipefail

RANGE="${1:?usage: monitor_cores.sh <lo>-<hi> <interval_s> <out.csv>}"
INTERVAL="${2:-5}"
OUT="${3:?usage: monitor_cores.sh <lo>-<hi> <interval_s> <out.csv>}"

LO="${RANGE%-*}"
HI="${RANGE#*-}"

mkdir -p "$(dirname "$OUT")"
{
  printf "ts"
  for ((c = LO; c <= HI; c++)); do printf ",cpu%s" "$c"; done
  printf "\n"
} > "$OUT"

read_stat() {
  # /proc/stat per-cpu line: cpuN user nice system idle iowait irq softirq steal guest guest_nice
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

declare -A PREV_TOTAL PREV_IDLE CUR

while read -r n t i; do
  PREV_TOTAL[$n]=$t
  PREV_IDLE[$n]=$i
done < <(read_stat)

while true; do
  sleep "$INTERVAL"
  ts=$(date +%s)
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
  line="$ts"
  for ((c = LO; c <= HI; c++)); do
    line="$line,${CUR[$c]:-0.0}"
  done
  echo "$line" >> "$OUT"
done
