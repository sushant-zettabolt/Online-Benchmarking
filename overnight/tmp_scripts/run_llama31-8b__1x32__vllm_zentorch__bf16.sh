export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so"
export VLLM_CPU_OMP_THREADS_BIND="128-159"
export OMP_NUM_THREADS=32
export VLLM_LOGGING_LEVEL=WARNING
numactl --physcpubind=128-159 --membind=4 bash -c '
  for npl in 1 2 4 8 12 16; do
    for pp in 128 512 1024 2048 4096; do
      echo "=== npl=$npl pp=$pp ===" >> "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__vllm_zentorch__bf16/batched_bench.log"
      "/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm" bench latency --model "/tmp/models/Llama-3.1-8B-Instruct" --dtype bfloat16         --max-model-len 8192 --batch-size "$npl" --input-len "$pp" --output-len 128         --num-iters 3 --num-iters-warmup 1         --output-json "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__vllm_zentorch__bf16/latency_npl${npl}_pp${pp}.json" >> "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__vllm_zentorch__bf16/batched_bench.log" 2>&1
    done
  done
'
