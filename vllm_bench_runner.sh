#!/bin/bash

# Internal benchmarking script that runs INSIDE the vllm-benchmark container
# This script is called by start.sh

# Read arguments passed from host
test=${BENCH_TEST:-0}
fast=${BENCH_FAST:-0}

# Server connection details (from docker network)
server_host=${VLLM_SERVER_HOST:-vllm-server}
server_port=${VLLM_SERVER_PORT:-8000}

# Get model from environment (set by .env file)
model=${VLLM_MODEL:-amd/Llama-3.3-70B-Instruct-FP8-KV}
tp=${VLLM_TENSOR_PARALLEL:-1}

# Output directory (mounted from host)
log_loc=/workspace
timestamp=$(date +"%y%m%d_%Hh%M")

echo "[BENCH INTERNAL] Starting benchmarks"
echo "[BENCH INTERNAL] Model: $model"
echo "[BENCH INTERNAL] Tensor Parallel: $tp"
echo "[BENCH INTERNAL] Server: $server_host:$server_port"
echo "[BENCH INTERNAL] Test mode: $test, Fast mode: $fast"

# Wait for server to be ready (extra safety even though depends_on health check exists)
echo "[BENCH INTERNAL] Verifying server is ready..."
max_wait=3600 # 1 hour max wait time to avoid infinite loop
elapsed=0
until curl -sSf "http://${server_host}:${server_port}/v1/models" >/dev/null 2>&1; do
    if [[ $elapsed -ge $max_wait ]]; then
        echo "[BENCH INTERNAL ERROR] Server not ready after ${max_wait}s"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "[BENCH INTERNAL] Server is ready!"

# Make model name friendly for file naming
mdl=$(echo "$model" | tr '/' '_')

# Run benchmarks
for max_concurrency in 1 4 8 16 32 64 128; do
    # use a number that will exercise max concurrency
    # num_prompts should be 2x max_concurrency for steady-state measurements
    if [[ $max_concurrency -eq 1 ]]; then
        num_prompts=2
    elif [[ $max_concurrency -eq 4 ]]; then
        num_prompts=8
    elif [[ $max_concurrency -eq 8 ]]; then
        num_prompts=16
    elif [[ $max_concurrency -eq 16 ]]; then
        num_prompts=32
    elif [[ $max_concurrency -eq 32 ]]; then
        num_prompts=64
    elif [[ $max_concurrency -eq 64 ]]; then
        num_prompts=128
    elif [[ $max_concurrency -eq 128 ]]; then
        num_prompts=256
    fi
    
    # cycle through different IL/OL combos
    for j in {1..4}; do
        if [[ $j -eq 1 ]]; then
            il=1024
            ol=4096
        elif [[ $j -eq 2 ]]; then
            il=4096
            ol=1024
        elif [[ $j -eq 3 ]]; then
            il=1024
            ol=1024
        else
            il=4096
            ol=4096
        fi
        
        for i in {1..3} ; do
            filename=benchsvr_${mdl}_tp${tp}_${il}-${ol}_mc${max_concurrency}_np${num_prompts}_t${i}
            log=${log_loc}/${filename}.log
            json=${log_loc}/${filename}.json
            
            echo "[BENCH] Running: IL=$il OL=$ol MC=$max_concurrency NP=$num_prompts (iteration $i/3)" | tee -a $log
            echo "Date: $(date)" >> $log
            
            vllm bench serve \
                --model $model \
                --dataset-name random \
                --random-input-len $il \
                --random-output-len $ol \
                --num-warmups ${NUM_WARMUPS:-5} \
                --max-concurrency $max_concurrency \
                --ignore-eos \
                --num-prompts $num_prompts \
                --percentile-metrics ttft,tpot,itl,e2el \
                --save-result \
                --result-dir $log_loc \
                --result-filename ${filename}.json \
                --host $server_host \
                --port $server_port \
                2>&1 | tee -a $log
                        
            # exit after 1 iteration for test run
            if [[ $test -eq 1 ]]; then
                echo "[BENCH INTERNAL] Test mode - exiting after first benchmark"
                break 3
            fi
            # only run once per config
            if [[ $fast -eq 1 ]]; then
                break
            fi
        done
        # exit after 1 iteration for test run (already handled above)
        if [[ $test -eq 1 ]]; then
            break
        fi
    done
    # exit after 1 iteration for test run (already handled above)
    if [[ $test -eq 1 ]]; then
        break
    fi
done

echo "[BENCH INTERNAL] Benchmarks complete!"
