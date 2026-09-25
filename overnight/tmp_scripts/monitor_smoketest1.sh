echo before > "/tmp/phase_smoketest1.txt"
cd "/proj/rdi/staff/sacsharm/online_bench/overnight"
setsid nohup ./monitor_resources.sh "192-223" 2 "/proj/rdi/staff/sacsharm/online_bench/overnight/results/smoketest1/resource_usage.csv" "/tmp/phase_smoketest1.txt" > /tmp/monlog_smoketest1.log 2>&1 < /dev/null &
disown
