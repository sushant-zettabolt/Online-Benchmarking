echo before > "/tmp/phase_llama31-8b__1x32__llamacpp_zendnn__q8.txt"
cd "/proj/rdi/staff/sacsharm/online_bench/overnight"
: > "/tmp/pid_llama31-8b__1x32__llamacpp_zendnn__q8.txt"
setsid nohup ./monitor_resources.sh "32-63" 2 "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__llamacpp_zendnn__q8/resource_usage.csv" "/tmp/phase_llama31-8b__1x32__llamacpp_zendnn__q8.txt" "/tmp/pid_llama31-8b__1x32__llamacpp_zendnn__q8.txt" > /tmp/monlog_llama31-8b__1x32__llamacpp_zendnn__q8.log 2>&1 < /dev/null &
disown
