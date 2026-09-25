cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_servers.sh "overnight_llama31-8b__4x8__vllm_zentorch__bf16" >/dev/null 2>&1
./start_servers.sh "overnight_llama31-8b__4x8__vllm_zentorch__bf16.conf" --wait-secs 900
