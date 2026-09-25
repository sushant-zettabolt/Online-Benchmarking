#!/usr/bin/env bash
# Copy the benchmark models from the shared NFS mount to the inference pod's
# LOCAL overlay disk (runs ON the inference pod).
#
# This matters for measurement validity, not just speed: the llama.cpp configs
# load with --load-mode mmap so that N instances share one page-cache copy of
# the model. If the mmap is backed by NFS, a page fault during decode becomes a
# network round trip and shows up as latency that has nothing to do with the
# core pinning we are trying to measure. /tmp is the pod's overlay fs.
#
# /tmp is wiped when the pod restarts, so re-run this after any restart.
#   ./stage_models.sh          # both models
#   ./stage_models.sh gguf     # llama.cpp only
#   ./stage_models.sh hf       # vLLM only
set -uo pipefail

SRC=/proj/rdi/staff/sacsharm/models
DST=/tmp/models
WHAT="${1:-all}"

mkdir -p "$DST"

stage_file() {
  local src="$1" dst="$2"
  if [ -f "$dst" ] && [ "$(stat -c %s "$src")" = "$(stat -c %s "$dst")" ]; then
    echo "already staged: $dst ($(du -h "$dst" | cut -f1))"
    return 0
  fi
  echo "copying $src -> $dst ..."
  cp -f "$src" "$dst.part" && mv -f "$dst.part" "$dst"
  echo "done: $dst ($(du -h "$dst" | cut -f1))"
}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "gguf" ]; then
  stage_file "$SRC/gguf/Llama-3.1-8B-Instruct-BF16.gguf" "$DST/Llama-3.1-8B-Instruct-BF16.gguf"
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "hf" ]; then
  if [ -d "$DST/Llama-3.1-8B-Instruct" ] && \
     [ -n "$(ls -A "$DST/Llama-3.1-8B-Instruct" 2>/dev/null)" ]; then
    echo "already staged: $DST/Llama-3.1-8B-Instruct"
  else
    echo "copying $SRC/hf/Llama-3.1-8B-Instruct -> $DST/ ..."
    cp -r "$SRC/hf/Llama-3.1-8B-Instruct" "$DST/"
  fi
  du -sh "$DST/Llama-3.1-8B-Instruct" 2>/dev/null
fi

echo
df -h /tmp | tail -1
ls -la "$DST"
