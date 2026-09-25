set -uo pipefail
mkdir -p /tmp/models
if [ -e "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ]; then
  if [ -f "/proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ]; then
    ssz=$(stat -c %s "/proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" 2>/dev/null || echo -1)
    dsz=$(stat -c %s "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" 2>/dev/null || echo -2)
    [ "$ssz" = "$dsz" ] && { echo "already staged"; exit 0; }
  else
    [ -d "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ] && [ -n "$(ls -A "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" 2>/dev/null)" ] && { echo "already staged (dir)"; exit 0; }
  fi
fi
echo "staging /proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 -> /tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
if [ -d "/proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ]; then
  rm -rf "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.part"; cp -r "/proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.part" && mv -f "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.part" "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
else
  cp -f "/proj/rdi/staff/sacsharm/models/hf/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.part" && mv -f "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.part" "/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
fi
