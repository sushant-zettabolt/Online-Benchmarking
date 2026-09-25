
llama_batched_bench: n_kv_max = 73728, n_batch = 2048, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 32, n_threads_batch = 32

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    0.376 |   340.05 |   15.370 |     8.33 |   15.746 |    16.26 |
