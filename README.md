# vLLM Docker Compose Benchmarking

This directory contains a Docker Compose setup for running vLLM benchmarks with ROCm.

## Architecture

The benchmarking system uses two containers:
1. **vllm-server**: Runs the vLLM inference server
2. **vllm-benchmark**: Runs benchmark tests against the server

All configuration is managed via the `.env` file. The orchestrator script (`start.sh`) runs on the host and manages both containers. The actual benchmarks run inside the `vllm-benchmark` container via `vllm_bench_runner.sh`.

## Setup

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and set your HF_TOKEN and other configurations
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x start.sh vllm_bench_runner.sh
   ```

## Usage

### Running Benchmarks

**Basic benchmark run:**
```bash
./start.sh
```

**Test run (single iteration):**
```bash
./start.sh --test --fast
```

**Custom log location:**
```bash
./start.sh --logs /path/to/logs
```

### Command-Line Options

The script accepts only these arguments:
- `--test`: Run a single benchmark iteration only
- `--fast`: Skip repeated runs (run each config once instead of 3 times)
- `--logs <path>`: Specify log directory (default: current directory)

**All other configuration** (model, tensor parallel size, quantization, etc.) is set in the `.env` file.

### Manual Container Management

**Start only the vLLM server:**
```bash
docker compose up -d vllm-server
```

**Check server health:**
```bash
docker compose ps
curl http://localhost:8000/v1/models
```

**Stop the server:**
```bash
docker compose down
```

**Interactive shell in benchmark container:**
```bash
docker compose run --rm --entrypoint /bin/bash vllm-benchmark
```

## How It Works

1. **docker-compose.yml**: Defines two services:
   - `vllm-server`: The vLLM server with ROCm support, exposed on port 8000
   - `vllm-benchmark`: Benchmark runner that connects to vllm-server

2. **start.sh** (host orchestrator):
   - Runs on your host machine
   - Loads configuration from `.env` file
   - Starts both docker-compose services
   - Waits for benchmarks to complete
   - Collects logs and shuts down services

3. **vllm_bench_runner.sh** (container script):
   - Runs inside the `vllm-benchmark` container
   - Connects to vllm-server via docker network
   - Executes benchmark tests with various configurations
   - Writes results to mounted workspace directory

4. **.env file**: Contains all configuration:
   - Model selection
   - Tensor parallel size
   - Quantization settings
   - vLLM optimization flags

## Configuration

All configuration is managed via the `.env` file (see [.env.example](.env.example)):

```bash
# vLLM Docker Compose Environment Configuration

# Docker image configuration (full image path)
VLLM_IMAGE=vllm/vllm-openai-rocm:latest

# Hardware Info [for parsing]
GPU_MODEL=MI355X
HIP_VISIBLE_DEVICES=7
# Hugging Face token for model downloads
HF_TOKEN=<your_hugging_face_token_here>

# Model configuration
VLLM_MODEL=Qwen/Qwen3-1.7B-FP8 #e.g. unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8
VLLM_TENSOR_PARALLEL=1
VLLM_PORT=8002
VLLM_KV_CACHE_DTYPE=fp8
VLLM_MAX_MODEL_LENGTH=8192 
NUM_WARMUPS=5

# Optional: vLLM optimization flags
# Uncomment and set these to enable specific optimizations
VLLM_ROCM_USE_AITER=1
VLLM_USE_AITER_TRITON_ROPE=1
VLLM_ROCM_USE_AITER_RMSNORM=1
VLLM_ROCM_USE_AITER_MHA=0
```

To change configuration, edit your `.env` file and run the benchmark script again.


## Output

Benchmarks generate:
- Individual run logs: `benchsvr_*.log`
- JSON results: `benchsvr_*.json`
- Combined run log: `start.log`
- Environment snapshot: `env_<timestamp>.log`

All files are created in the log directory (current directory by default, or specified via `--logs`).

## Troubleshooting

**Server not starting:**
```bash
docker compose logs vllm-server
```

**Benchmark container errors:**
```bash
docker compose logs vllm-benchmark
```

**GPU not accessible:**
Ensure you have ROCm drivers installed and `/dev/kfd` and `/dev/dri` devices are available:
```bash
ls -la /dev/kfd /dev/dri
```

**Network connection issues:**
The benchmark container connects to the server container via docker network using hostname `vllm-server`. Check connectivity:
```bash
docker compose run --rm vllm-benchmark curl http://vllm-server:8000/v1/models
```

Or use the env var:
```bash
docker compose run --rm vllm-benchmark sh -c "curl http://vllm-server:\$VLLM_PORT/v1/models"
```

**DOCKER_COMPOSE not set error:**
Make sure you have either `docker compose` (Docker Compose V2) or `docker-compose` (V1) installed.

