export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/lib/x86_64-linux-gnu/libomp.so.5"
export GOMP_CPU_AFFINITY="192-223"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_LRU_CACHE_CAPACITY=1024
numactl --physcpubind=192-223 --membind=6   "/proj/rdi/staff/sacsharm/llama.cpp/build_release/bin/llama-batched-bench" -m "/tmp/models/Llama-3.1-8B-Instruct-BF16.gguf"   -c 70000 -b 2048 -ub 512 -t 32 -tb 32   -Cr 192-223 -Crb 192-223 --cpu-strict 1 --cpu-strict-batch 1   -fa on -npp 128,512,1024,2048,4096 -ntg 128 -npl 1,2,4,8,12,16   --output-format md > "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__llamacpp_nozendnn__bf16/batched_bench.md" 2>"/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__llamacpp_nozendnn__bf16/batched_bench.log"
