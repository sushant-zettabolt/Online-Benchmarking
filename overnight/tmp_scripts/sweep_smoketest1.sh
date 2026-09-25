cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_haproxy.sh "overnight_smoketest1" >/dev/null 2>&1
./start_haproxy.sh "overnight_smoketest1" --port 9000 --stats-port 9001
./run_multiuser_sweep.sh "overnight_smoketest1" --users "2" --duration "60"   --port 9000 --stats-port 9001
./stop_haproxy.sh "overnight_smoketest1" >/dev/null 2>&1
