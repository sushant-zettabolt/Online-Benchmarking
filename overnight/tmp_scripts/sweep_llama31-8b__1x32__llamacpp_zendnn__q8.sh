cd "/proj/rdi/staff/sacsharm/online_bench"
mkdir -p "/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__llamacpp_zendnn__q8/ppsweep"
for pp in 1 2 3 4 8 16 32 48 64 96 128 256 384 512 768 1024 1536 2048 3072 4096; do
  out="/proj/rdi/staff/sacsharm/online_bench/overnight/results/llama31-8b__1x32__llamacpp_zendnn__q8/ppsweep/online_llamacpp_${pp}.json"
  [ -s "$out" ] && continue
  python3 bench_client.py --backend "llamacpp" --url "http://192.168.6.8:8080"     --model "default" --pp "$pp" --tg 1 --num-iters 3 --warmup-iters 1     --output-json "$out"
done
