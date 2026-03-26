#!/bin/bash

# vLLM Benchmark Sweep Script
# Uses vllm bench sweep serve to automatically test multiple configurations
# All configuration comes from .env file

set -e

# Load .env file
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please create it from .env.example"
    exit 1
fi

# Source the .env file to get configuration
source .env

# Default values
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"benchmark_$(date +%y%m%d_%H%M)"}
OUTPUT_DIR=${OUTPUT_DIR:-"./Results"}
BENCH_PARAMS=${BENCH_PARAMS:-"./bench_params.json"}
SERVE_PARAMS=${SERVE_PARAMS:-"./serve_params.json"}
SERVER_READY_TIMEOUT=600 # How long to wait for the vllm server to be ready (in seconds)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bench-params)
            BENCH_PARAMS="$2"
            shift 2
            ;;
        --serve-params)
            SERVE_PARAMS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment-name NAME  Name for this benchmark run (default: benchmark_<timestamp>)"
            echo "  --output-dir DIR        Directory to save results (default: ./Results)"
            echo "  --bench-params FILE     JSON file with benchmark parameters (default: ./bench_params.json)"
            echo "  --serve-params FILE     JSON file with server parameters (default: ./serve_params.json)"
            echo "  --trials N              Number of trials per configuration (default: 3)"
            echo ""
            echo "Configuration is loaded from .env file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "VLLM_IMAGE" ]]; then
    echo "[ERROR] VLLM_IMAGE not set in .env file"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "vLLM Benchmark Sweep Configuration"
echo "========================================="
echo "Experiment Name:    $EXPERIMENT_NAME"
echo "Docker Image:       $VLLM_IMAGE"
echo "Output Directory:   $OUTPUT_DIR"
echo "========================================="
echo ""


# Run the benchmark sweep in Docker
echo "[INFO] Starting benchmark sweep..."
echo ""

docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --network=host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "$(pwd):/workspace" \
    -w /workspace \
    --env-file .env \
    --entrypoint "" \
    "${VLLM_IMAGE}" \
    vllm bench sweep serve \
        --serve-cmd "vllm serve" \
        --server-ready-timeout 7200 \
        --bench-cmd "vllm bench serve" \
        --serve-params "${SERVE_PARAMS}" \
        --bench-params "${BENCH_PARAMS}" \
        --output-dir "${OUTPUT_DIR}" \
        --experiment-name "${EXPERIMENT_NAME}" \


EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "========================================="
    echo "[SUCCESS] Benchmark sweep complete!"
    echo "========================================="
    echo "Results saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
    echo "Configuration log: ${CONFIG_LOG}"
    echo ""
    echo "To analyze results:"
    echo "  python3 parse_json_to_csv.py --base-path ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
else
    echo ""
    echo "========================================="
    echo "[ERROR] Benchmark sweep failed!"
    echo "========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check logs in: ${OUTPUT_DIR}"
    exit $EXIT_CODE
fi
