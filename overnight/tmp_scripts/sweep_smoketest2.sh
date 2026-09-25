cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_haproxy.sh "overnight_smoketest2" >/dev/null 2>&1
./start_haproxy.sh "overnight_smoketest2" --port 9000 --stats-port 9001
./run_multiuser_sweep.sh "overnight_smoketest2" --users "2" --duration "45"   --port 9000 --stats-port 9001
./stop_haproxy.sh "overnight_smoketest2" >/dev/null 2>&1
