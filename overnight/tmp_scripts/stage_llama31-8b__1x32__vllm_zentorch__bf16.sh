set -uo pipefail
mkdir -p /tmp/models
if [ -e "/tmp/models/Llama-3.1-8B-Instruct" ]; then
  if [ -f "/proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct" ]; then
    ssz=$(stat -c %s "/proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct" 2>/dev/null || echo -1)
    dsz=$(stat -c %s "/tmp/models/Llama-3.1-8B-Instruct" 2>/dev/null || echo -2)
    [ "$ssz" = "$dsz" ] && { echo "already staged"; exit 0; }
  else
    [ -d "/tmp/models/Llama-3.1-8B-Instruct" ] && [ -n "$(ls -A "/tmp/models/Llama-3.1-8B-Instruct" 2>/dev/null)" ] && { echo "already staged (dir)"; exit 0; }
  fi
fi
echo "staging /proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct -> /tmp/models/Llama-3.1-8B-Instruct"
if [ -d "/proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct" ]; then
  rm -rf "/tmp/models/Llama-3.1-8B-Instruct.part"; cp -r "/proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct" "/tmp/models/Llama-3.1-8B-Instruct.part" && mv -f "/tmp/models/Llama-3.1-8B-Instruct.part" "/tmp/models/Llama-3.1-8B-Instruct"
else
  cp -f "/proj/rdi/staff/sacsharm/models/hf/Llama-3.1-8B-Instruct" "/tmp/models/Llama-3.1-8B-Instruct.part" && mv -f "/tmp/models/Llama-3.1-8B-Instruct.part" "/tmp/models/Llama-3.1-8B-Instruct"
fi
