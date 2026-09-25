echo before > "/tmp/phase_llama31-8b__8x4__llamacpp_zendnn__bf16.txt"
cd "/proj/rdi/staff/sacsharm/online_bench/overnight"
: > "/tmp/pid_llama31-8b__8x4__llamacpp_zendnn__bf16.txt"
setsid nohup ./monitor_resources.sh "224-255" 2 "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__8x4__llamacpp_zendnn__bf16/resource_usage.csv" "/tmp/phase_llama31-8b__8x4__llamacpp_zendnn__bf16.txt" "/tmp/pid_llama31-8b__8x4__llamacpp_zendnn__bf16.txt" > /tmp/monlog_llama31-8b__8x4__llamacpp_zendnn__bf16.log 2>&1 < /dev/null &
disown
