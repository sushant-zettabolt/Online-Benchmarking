cd "/proj/rdi/staff/sacsharm/online_bench/multiuser"
./stop_servers.sh "overnight_smoketest2" >/dev/null 2>&1
./start_servers.sh "overnight_smoketest2.conf" --wait-secs 900
