#!/bin/bash

# Docker Compose-based vLLM benchmarking orchestrator
# This script runs on the host and manages docker compose services
# All configuration comes from .env file
# 
# Example usage:
#   ./start.sh -test -fast
#   ./start.sh --logs /path/to/logs

# default values
test=0
fast=0
log_loc=$PWD

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -test|--test)
            # Test a single run only
            test=1
            shift 1
            ;;
        -fast|--fast)
            # Don't repeat runs 3 times
            fast=1
            shift 1
            ;;
        -logs|--logs)
            # Specify log location
            log_loc="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--fast] [--logs <path>]"
            echo ""
            echo "Options:"
            echo "  --test    Run a single benchmark iteration only"
            echo "  --fast    Skip repeated runs (run each config once)"
            echo "  --logs    Specify log directory (default: current directory)"
            echo ""
            echo "Configuration is loaded from .env file"
            exit 1
            ;;
    esac
done

# record env
timestamp=$(date +"%y%m%d_%Hh%M")
env_log=${log_loc}/env_${timestamp}.log

# Check if docker compose is available and set the command to use
if command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "[BENCH ERROR] Neither 'docker compose' nor 'docker-compose' command found"
    exit 1
fi

echo "[BENCH] Using command: $DOCKER_COMPOSE"

# Record environment info from host
echo "=== Benchmark Environment Info ===" > $env_log
echo "Timestamp: $timestamp" >> $env_log
echo "Test Mode: $test" >> $env_log
echo "Fast Mode: $fast" >> $env_log
echo "Log Location: $log_loc" >> $env_log
echo "" >> $env_log

# Collect ROCm info if available
if command -v rocm-smi &> /dev/null; then
    echo -e "\n$ rocm-smi --showdriverversion" >> $env_log
    rocm-smi --showdriverversion >> $env_log 2>&1
    echo -e "\n$ rocm-smi --showvbios" >> $env_log
    rocm-smi --showvbios >> $env_log 2>&1
fi

if command -v amd-smi &> /dev/null; then
    echo -e "\n$ amd-smi firmware | head -n45" >> $env_log
    amd-smi firmware | head -n45 >> $env_log 2>&1
fi

echo "[BENCH] Configuration loaded from .env file"
echo "[BENCH] Starting services with docker compose..."

# Export test/fast flags for the internal benchmark script
export BENCH_TEST=$test
export BENCH_FAST=$fast

# Start services: vllm-server and vllm-benchmark
# The benchmark container will run vllm_bench_runner.sh
$DOCKER_COMPOSE up --abort-on-container-exit --exit-code-from vllm-benchmark vllm-server vllm-benchmark 2>&1 | tee ${log_loc}/start.log

# Capture exit code
exit_code=${PIPESTATUS[0]}

# log container info
echo -e "\n=== Docker container info ===" >> $env_log
$DOCKER_COMPOSE ps >> $env_log 2>&1
echo -e "\n=== Server container logs ===" >> $env_log
$DOCKER_COMPOSE logs --tail=100 vllm-server >> $env_log 2>&1
echo -e "\n=== Benchmark container logs ===" >> $env_log
$DOCKER_COMPOSE logs --tail=100 vllm-benchmark >> $env_log 2>&1

# shutdown services
echo "[BENCH] Shutting down services..."
$DOCKER_COMPOSE down

# Move benchmark results to log directory if it's different from current directory
if [[ "$(realpath $log_loc)" != "$(realpath $PWD)" ]]; then
    echo "[BENCH] Moving benchmark results to $log_loc..."
    mkdir -p "$log_loc"
    
    # Move all benchsvr JSON and log files
    if ls benchsvr_*.json 1> /dev/null 2>&1; then
        mv benchsvr_*.json "$log_loc/" 2>/dev/null || true
    fi
    if ls benchsvr_*.log 1> /dev/null 2>&1; then
        mv benchsvr_*.log "$log_loc/" 2>/dev/null || true
    fi
    
    echo "[BENCH] Benchmark files moved to $log_loc"
fi

if [[ $exit_code -eq 0 ]]; then
    echo "[BENCH] Benchmark complete! Logs saved to $log_loc"
    echo "[BENCH] Environment log: $env_log"
else
    echo "[BENCH ERROR] Benchmark failed with exit code $exit_code"
    echo "[BENCH ERROR] Check logs in $log_loc"
    exit $exit_code
fi
