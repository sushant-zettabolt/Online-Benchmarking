cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_servers.sh "overnight_llama31-8b__1x32__vllm_zentorch__w8a8" >/dev/null 2>&1
./start_servers.sh "overnight_llama31-8b__1x32__vllm_zentorch__w8a8.conf" --wait-secs 900
