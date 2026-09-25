set -uo pipefail
mkdir -p /tmp/models
if [ -e "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf" ]; then
  if [ -f "/proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf" ]; then
    ssz=$(stat -c %s "/proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf" 2>/dev/null || echo -1)
    dsz=$(stat -c %s "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf" 2>/dev/null || echo -2)
    [ "$ssz" = "$dsz" ] && { echo "already staged"; exit 0; }
  else
    [ -d "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf" ] && [ -n "$(ls -A "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf" 2>/dev/null)" ] && { echo "already staged (dir)"; exit 0; }
  fi
fi
echo "staging /proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf -> /tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf"
if [ -d "/proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf" ]; then
  rm -rf "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf.part"; cp -r "/proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf" "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf.part" && mv -f "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf.part" "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf"
else
  cp -f "/proj/rdi/staff/sacsharm/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf" "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf.part" && mv -f "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf.part" "/tmp/models/Llama-3.1-8B-Instruct-Q8_0.gguf"
fi
