echo before > "/tmp/phase_smoketest_batched.txt"
cd "/proj/rdi/staff/sacsharm/online_bench/overnight"
: > "/tmp/pid_smoketest_batched.txt"
setsid nohup ./monitor_resources.sh "192-223" 2 "/proj/rdi/staff/sacsharm/online_bench/overnight/results/smoketest_batched/resource_usage.csv" "/tmp/phase_smoketest_batched.txt" "/tmp/pid_smoketest_batched.txt" > /tmp/monlog_smoketest_batched.log 2>&1 < /dev/null &
disown
