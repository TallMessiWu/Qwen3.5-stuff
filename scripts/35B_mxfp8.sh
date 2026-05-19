export ASCEND_RT_VISIBLE_DEVICES=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1


# export VLLM_ASCEND_BALANCE_SCHEDULING=1
vllm serve /mnt/weight/Qwen3.5-35B-A3B-W8A8-MXFP8-FULL-QUANT \
--host 127.0.0.1 \
--port 6969 \
--quantization ascend \
--served-model-name qwen3.5 \
--data-parallel-size 1 \
--tensor-parallel-size 1 \
--max-num-seqs=64 \
--max-model-len=16384 \
--max-num-batched-tokens 16384  \
--gpu-memory-utilization 0.43 \
--trust-remote-code \
--async-scheduling \
--additional-config '{"enable_cpu_binding":true}' \
--default-chat-template-kwargs '{"enable_thinking": false}' \
--no-enable-prefix-caching \
--mm-processor-cache-gb 0 \
--reasoning-parser qwen3 \
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile_offline", "torch_profiler_with_stack": true}' \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 2}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# --compilation-config '{"cudagraph_capture_sizes":[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60], "cudagraph_mode":"FULL"}'
# --enforce-eager
# --max-model-len=262144 \
# --compilation-config '{"cudagraph_mode":"FULL"}'
#--gpu-memory-utilization 0.9 \
# --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile_offline"}' \
# --compilation-config '{"cudagraph_mode":"FULL"}'
# --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,10], "cudagraph_mode":"FULL"}'
# 2>&1 | tee model.log
# --language-model-only