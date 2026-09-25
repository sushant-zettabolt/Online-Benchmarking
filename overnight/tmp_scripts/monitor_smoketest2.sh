echo before > "/tmp/phase_smoketest2.txt"
cd "/proj/rdi/staff/sacsharm/online_bench/overnight"
: > "/tmp/pid_smoketest2.txt"
setsid nohup ./monitor_resources.sh "192-223" 2 "/proj/rdi/staff/sacsharm/online_bench/overnight/results/smoketest2/resource_usage.csv" "/tmp/phase_smoketest2.txt" "/tmp/pid_smoketest2.txt" > /tmp/monlog_smoketest2.log 2>&1 < /dev/null &
disown
