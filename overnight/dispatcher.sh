#!/usr/bin/env bash
# Top-level overnight orchestrator. Runs on the devbox (needs kubectl), meant
# to be launched detached: `setsid nohup ./dispatcher.sh <control_pod> > dispatcher.log 2>&1 & disown`
# so it survives independent of whatever invoked it.
#
# - Builds queue.tsv from matrix.conf (once, skipped if it already exists so
#   a restart resumes rather than re-queuing everything).
# - Claims the control pod + every currently-free turin-* pod, launches one
#   worker.sh per inference pod (each drains the shared queue until empty or
#   the 07:00 Asia/Kolkata cutoff).
# - Every ~10 min: re-discovers newly-freed turin-* pods (people log off
#   through the night) and launches additional workers for them too, up to
#   MAX_INFER_PODS.
# - At the cutoff: writes the STOP marker (workers finish their CURRENT job,
#   don't pick up new ones, then release their own pod locks and exit),
#   waits for all workers to actually stop, then writes RUN_LEDGER.md and
#   releases the control pod claim.
set -uo pipefail

OVERNIGHT_DIR="/proj/rdi/staff/sacsharm/online_bench/overnight"
MATRIX="$OVERNIGHT_DIR/matrix.conf"
QUEUE="$OVERNIGHT_DIR/queue.tsv"
STOP_MARKER="$OVERNIGHT_DIR/STOP"
CONTROL_POD_FILE="$OVERNIGHT_DIR/.control_pod"
CLAIMED_FILE="$OVERNIGHT_DIR/.claimed_pods"
LEDGER_JSONL="$OVERNIGHT_DIR/RUN_LEDGER.jsonl"
LEDGER_MD="$OVERNIGHT_DIR/RUN_LEDGER.md"
export KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml
SLOTS_DIR=/proj/zendnn/k8
LOCK="$SLOTS_DIR/pod-lock.sh"
MAX_INFER_PODS="${MAX_INFER_PODS:-6}"
CUTOFF_HHMM="${CUTOFF_HHMM:-07:00}"
CUTOFF_TZ="Asia/Kolkata"

DISPATCH_LOG="$OVERNIGHT_DIR/dispatcher.log"
log() { echo "[$(date '+%F %H:%M:%S')] [dispatcher] $*" | tee -a "$DISPATCH_LOG"; }

rm -f "$STOP_MARKER"

# ------------------------------------------------------------- build queue --
if [ ! -f "$QUEUE" ]; then
  log "building queue.tsv from matrix.conf"
  {
    printf "status\tmode\tlayout\tinstances\tcores\tbackend\tvariant\tquant\tpriority\tmodel\tjob_id\tpod\n"
    awk '!/^#/ && NF>=9' "$MATRIX" | while read -r mode layout instances cores backend variant quant priority model; do
      job_id="${model}__${layout}__${backend}_${variant}__${quant}"
      printf "PENDING\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\n" \
        "$mode" "$layout" "$instances" "$cores" "$backend" "$variant" "$quant" "$priority" "$model" "$job_id"
    done
  } > "$QUEUE"
  log "queue built: $(($(wc -l < "$QUEUE") - 1)) jobs"
else
  log "queue.tsv already exists, resuming ($(grep -c PENDING "$QUEUE" 2>/dev/null || echo 0) pending, $(grep -c RUNNING "$QUEUE" 2>/dev/null || echo 0) running left over)"
  # Only reset a RUNNING row to PENDING if its assigned pod has no live
  # worker.sh process -- a normal dispatcher restart (this branch runs every
  # time, not just after a crash) must NOT touch rows whose worker is still
  # legitimately mid-job, or a second worker could later grab the same
  # job_id and both write into the same results dir concurrently.
  python3 - "$QUEUE" <<'PY'
import subprocess, sys

def worker_alive(pod):
    r = subprocess.run(["pgrep", "-f", f"worker.sh {pod} "], capture_output=True)
    return r.returncode == 0

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()
header, rows = lines[0], lines[1:]
for i, line in enumerate(rows):
    cols = line.rstrip("\n").split("\t")
    if cols[0] == "RUNNING" and not worker_alive(cols[11]):
        cols[0] = "PENDING"
        rows[i] = "\t".join(cols) + "\n"
with open(path, "w") as f:
    f.writelines([header] + rows)
PY
fi

# ------------------------------------------------------------- claim pods --
claim_pod() {  # $1=pod -> 0 if claimed (or already ours)
  bash "$LOCK" claim "$1" sacsharm >/dev/null 2>&1
}

free_turin_pods() {
  # turin-* pods with CLAIM column FREE-or-ours AND USER column showing no
  # OTHER real user. CLAIM=="-" alone is NOT sufficient -- USER can carry a
  # comma-separated list of live process owners (e.g. "50112475,isshreya")
  # even when nobody holds the advisory CLAIM lock, and kubectl-exec'd
  # commands of ours show up there too, as our raw numeric UID rather than
  # "sacsharm" -- both must be stripped before deciding "no one else is here".
  # (Learned the hard way: an early claim of xcovoid0023-pod-2 grabbed a pod
  # with CLAIM="-" but USER="50112475,isshreya" -- isshreya was already on
  # it. Caught and backed out within ~1 minute, before any server/core work
  # started, only a harmless model-staging copy was killed mid-flight.)
  (cd "$SLOTS_DIR" && timeout 90 bash slots.sh 2>/dev/null) | \
    awk '$1 ~ /^turin-/ && $3=="Running" && ($5=="-" || $5=="sacsharm") {print $1, $6}' | \
    python3 -c '
import sys
my_uid = str(__import__("os").getuid())
for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    pod = parts[0]
    users = parts[1].split(",") if len(parts) > 1 else []
    others = [u for u in users if u not in ("sacsharm", my_uid, "-", "FREE")]
    if not others:
        print(pod)
'
}

if [ ! -s "$CONTROL_POD_FILE" ]; then
  CONTROL_CANDIDATES="turin-xcovoid0021-pod-5 $(free_turin_pods)"
  CONTROL_POD=""
  for p in $CONTROL_CANDIDATES; do
    if claim_pod "$p"; then CONTROL_POD="$p"; break; fi
  done
  [ -n "$CONTROL_POD" ] || { log "FATAL: could not claim any control pod"; exit 1; }
  echo "$CONTROL_POD" > "$CONTROL_POD_FILE"
  log "claimed control pod: $CONTROL_POD"
else
  CONTROL_POD="$(cat "$CONTROL_POD_FILE")"
  claim_pod "$CONTROL_POD"
  log "control pod (from prior run): $CONTROL_POD"
fi

touch "$CLAIMED_FILE"
NEXT_SLOT_FILE="$OVERNIGHT_DIR/.next_slot"
[ -s "$NEXT_SLOT_FILE" ] || echo 0 > "$NEXT_SLOT_FILE"
launch_worker_for() {  # $1=pod -- worker slot persisted to disk (not just this
                       # process's memory) so a dispatcher restart can't hand
                       # out a slot/port range already in use by a still-live
                       # worker from before the restart.
  local pod="$1"
  grep -qx "$pod" "$CLAIMED_FILE" 2>/dev/null && return 1
  [ "$pod" = "$CONTROL_POD" ] && return 1
  claim_pod "$pod" || return 1
  echo "$pod" >> "$CLAIMED_FILE"
  NEXT_SLOT="$(cat "$NEXT_SLOT_FILE")"
  echo $((NEXT_SLOT + 1)) > "$NEXT_SLOT_FILE"
  log "claimed inference pod $pod, launching worker (slot $NEXT_SLOT)"
  ( cd "$OVERNIGHT_DIR" && setsid nohup bash ./worker.sh "$pod" "$NEXT_SLOT" \
      > "$OVERNIGHT_DIR/worker_${pod}_launch.log" 2>&1 < /dev/null & disown )
  return 0
}

log "initial pod discovery..."
for p in $(free_turin_pods); do
  n_claimed=$(($(wc -l < "$CLAIMED_FILE") ))
  [ "$n_claimed" -ge "$MAX_INFER_PODS" ] && break
  launch_worker_for "$p"
done
log "initial workers launched: $(cat "$CLAIMED_FILE" | wc -l) inference pod(s)"

# ------------------------------------------------------------------- loop --
seconds_to_cutoff() {
  python3 - "$CUTOFF_HHMM" "$CUTOFF_TZ" <<'PY'
import sys, datetime
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(sys.argv[2])
except Exception:
    tz = None
hh, mm = (int(x) for x in sys.argv[1].split(":"))
now = datetime.datetime.now(tz) if tz else datetime.datetime.now()
target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
if target <= now:
    target += datetime.timedelta(days=1)
print(int((target - now).total_seconds()))
PY
}

while true; do
  SEC_LEFT=$(seconds_to_cutoff)
  if [ "$SEC_LEFT" -le 0 ]; then
    log "cutoff ($CUTOFF_HHMM $CUTOFF_TZ) reached -- writing STOP marker"
    touch "$STOP_MARKER"
    break
  fi
  log "status: $(grep -c DONE "$QUEUE") done, $(grep -c FAILED "$QUEUE") failed, $(grep -c RUNNING "$QUEUE") running, $(grep -c PENDING "$QUEUE") pending; $(($(wc -l < "$CLAIMED_FILE"))) inference pod(s) active; ${SEC_LEFT}s to cutoff"

  n_claimed=$(($(wc -l < "$CLAIMED_FILE")))
  if [ "$n_claimed" -lt "$MAX_INFER_PODS" ]; then
    for p in $(free_turin_pods); do
      n_claimed=$(($(wc -l < "$CLAIMED_FILE")))
      [ "$n_claimed" -ge "$MAX_INFER_PODS" ] && break
      launch_worker_for "$p" && log "picked up newly-free pod $p"
    done
  fi

  SLEEP_S=600
  [ "$SEC_LEFT" -lt "$SLEEP_S" ] && SLEEP_S=$SEC_LEFT
  sleep "$SLEEP_S"
done

# --------------------------------------------------------- wind-down/report --
log "waiting for active workers to finish their current job..."
DEADLINE=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RUNNING="$(grep -c RUNNING "$QUEUE" 2>/dev/null || echo 0)"
  [ "$RUNNING" -eq 0 ] && break
  log "still waiting: $RUNNING job(s) running"
  sleep 120
done

while read -r pod; do
  bash "$LOCK" release "$pod" sacsharm >/dev/null 2>&1
done < "$CLAIMED_FILE"
bash "$LOCK" release "$CONTROL_POD" sacsharm >/dev/null 2>&1
log "all pod locks released"

python3 - "$QUEUE" "$LEDGER_JSONL" > "$LEDGER_MD" <<'PY'
import sys, csv, json, collections

queue_path, ledger_jsonl = sys.argv[1], sys.argv[2]
with open(queue_path) as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

by_status = collections.Counter(r["status"] for r in rows)
print("# Overnight run ledger\n")
print(f"Totals: {dict(by_status)}\n")
print("| job_id | mode | status |")
print("|---|---|---|")
for r in rows:
    print(f"| {r['job_id']} | {r['mode']} | {r['status']} |")

print("\n## Per-job detail (from RUN_LEDGER.jsonl, run_job.sh's own records)\n")
try:
    with open(ledger_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            print(f"- **{rec['job_id']}**: {rec['status']} ({rec['started_at']} -> {rec['ended_at']})"
                  + (f" -- {rec['note']}" if rec.get("note") else ""))
except FileNotFoundError:
    print("(no per-job ledger entries yet)")
PY
log "wrote $LEDGER_MD"
log "dispatcher done."
