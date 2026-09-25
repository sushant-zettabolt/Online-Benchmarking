#!/usr/bin/env bash
# One worker per claimed inference pod. Pops jobs off the shared queue.tsv
# (flock-protected) and runs them one at a time via run_job.sh until the
# queue is empty or the STOP marker (07:00 IST cutoff, written by
# dispatcher.sh) appears. Releases its pod-lock claim when it exits either
# way. Meant to be launched detached: `setsid nohup ./worker.sh <pod> <slot> &`
set -uo pipefail

INFER_POD="$1"; WORKER_SLOT="$2"
OVERNIGHT_DIR="/proj/rdi/staff/sacsharm/online_bench/overnight"
QUEUE="$OVERNIGHT_DIR/queue.tsv"
LOCKFILE="$OVERNIGHT_DIR/.queue.lock"
STOP_MARKER="$OVERNIGHT_DIR/STOP"
CONTROL_POD_FILE="$OVERNIGHT_DIR/.control_pod"
export KUBECONFIG=/proj/zendnn/k8/dev-kubeconfig.yaml
LOCK="/proj/zendnn/k8/pod-lock.sh"

CONTROL_POD="$(cat "$CONTROL_POD_FILE")"
LOG="$OVERNIGHT_DIR/worker_${INFER_POD}.log"
log() { echo "[$(date '+%F %H:%M:%S')] [worker:$INFER_POD] $*" >> "$LOG"; }

log "worker starting, slot=$WORKER_SLOT, control_pod=$CONTROL_POD"

pop_next_job() {
  # Prints a TSV line for the next PENDING job (lowest priority number first,
  # first-seen order within a priority) and marks it RUNNING+this pod, or
  # prints nothing if none pending. Flock-serialized against other workers.
  python3 - "$QUEUE" "$INFER_POD" <<'PY'
import fcntl, sys

queue_path, pod = sys.argv[1], sys.argv[2]
with open(queue_path, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.readlines()
    header, rows = lines[0], lines[1:]
    best_i, best_pri = None, None
    for i, line in enumerate(rows):
        cols = line.rstrip("\n").split("\t")
        status, pri = cols[0], int(cols[8])
        if status == "PENDING" and (best_pri is None or pri < best_pri):
            best_i, best_pri = i, pri
    if best_i is None:
        fcntl.flock(f, fcntl.LOCK_UN)
        sys.exit(1)
    cols = rows[best_i].rstrip("\n").split("\t")
    cols[0] = "RUNNING"
    cols[11] = pod
    rows[best_i] = "\t".join(cols) + "\n"
    f.seek(0); f.writelines([header] + rows); f.truncate()
    fcntl.flock(f, fcntl.LOCK_UN)
    print("\t".join(cols))
PY
}

mark_status() {  # $1=job_id $2=status
  python3 - "$QUEUE" "$1" "$2" <<'PY'
import fcntl, sys
queue_path, job_id, status = sys.argv[1], sys.argv[2], sys.argv[3]
with open(queue_path, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.readlines()
    header, rows = lines[0], lines[1:]
    for i, line in enumerate(rows):
        cols = line.rstrip("\n").split("\t")
        if cols[10] == job_id:
            cols[0] = status
            rows[i] = "\t".join(cols) + "\n"
            break
    f.seek(0); f.writelines([header] + rows); f.truncate()
    fcntl.flock(f, fcntl.LOCK_UN)
PY
}

while true; do
  if [ -f "$STOP_MARKER" ]; then
    log "STOP marker present, exiting"
    break
  fi
  JOB_LINE="$(pop_next_job)" || { log "queue empty, exiting"; break; }
  IFS=$'\t' read -r STATUS MODE LAYOUT INSTANCES CORES BACKEND VARIANT QUANT PRIORITY MODEL JOB_ID POD <<< "$JOB_LINE"
  log "starting job $JOB_ID (mode=$MODE layout=$LAYOUT $BACKEND/$VARIANT/$QUANT)"

  bash "$OVERNIGHT_DIR/run_job.sh" "$MODE" "$LAYOUT" "$INSTANCES" "$CORES" "$BACKEND" \
    "$VARIANT" "$QUANT" "$MODEL" "$INFER_POD" "$CONTROL_POD" "$WORKER_SLOT" "$JOB_ID" \
    >> "$LOG" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ]; then
    mark_status "$JOB_ID" "DONE"
    log "job $JOB_ID DONE"
  else
    mark_status "$JOB_ID" "FAILED"
    log "job $JOB_ID FAILED (rc=$RC)"
  fi

  bash "$LOCK" refresh "$INFER_POD" sacsharm >/dev/null 2>&1
done

bash "$LOCK" release "$INFER_POD" sacsharm >/dev/null 2>&1
log "worker exiting, released pod lock"
