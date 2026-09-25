pkill -u $(id -u) -f 'monitor_resources[.]sh.*smoketest1' 2>/dev/null; pkill -u $(id -u) -f "/proj/rdi/staff/sacsharm/online_bench/overnight/results/smoketest1/resource_usage.csv" 2>/dev/null; true
