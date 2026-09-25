cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_haproxy.sh "overnight_llama31-8b__4x8__llamacpp_nozendnn__bf16" >/dev/null 2>&1
./start_haproxy.sh "overnight_llama31-8b__4x8__llamacpp_nozendnn__bf16" --port 9050 --stats-port 9051
./run_multiuser_sweep.sh "overnight_llama31-8b__4x8__llamacpp_nozendnn__bf16" --users "2 4 8 12 16" --duration "300"   --port 9050 --stats-port 9051
./stop_haproxy.sh "overnight_llama31-8b__4x8__llamacpp_nozendnn__bf16" >/dev/null 2>&1
