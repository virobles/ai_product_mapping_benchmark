#!/usr/bin/env python3
"""
Script to parse vLLM benchmark sweep results and export to CSV.

This script searches through benchmark experiment folders (e.g., Results/benchmark_260324_1550)
for summary.json files and consolidates all the data into a CSV file matching the format
of example_results.csv.

Usage:
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 --gpu "MI355X" --output results.csv
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 --gpu "MI355X" --precision "FP8" --docker-image "vllm/vllm-openai-rocm:latest"
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 --gpu "MI355X" --precision "FP8" --docker-image "vllm/vllm-openai-rocm:latest" --output benchmark_results_mi355x.csv
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Model mapping: base model -> MI355X variant
MODEL_MAPPING = {
    "openai/gpt-oss-120b": "amd/gpt-oss-120b-w-mxfp4-a-fp8",
    "meta-llama/Llama-3.3-70B-Instruct": "amd/Llama-3.3-70B-Instruct-FP8-KV",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    "google/gemma-3-27b-it": "RedHatAI/gemma-3-27b-it-FP8-dynamic",
    "meta-llama/Llama-3.1-8B": "amd/Llama-3.1-8B-Instruct-FP8-KV",
    "Qwen/Qwen3-4B": "Qwen/Qwen3-4B-FP8",
    "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B-FP8",
}


def extract_model_parts(model_id: str) -> tuple:
    """
    Extract base model and model variant from model_id using MODEL_MAPPING.
    
    Examples:
        "amd/Llama-3.1-8B-Instruct-FP8-KV" -> ("meta-llama/Llama-3.1-8B", "amd/Llama-3.1-8B-Instruct-FP8-KV")
        "Qwen/Qwen3-1.7B-FP8" -> ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-FP8")
    """
    # Check if model_id is a known variant in our mapping
    for base_model, variant in MODEL_MAPPING.items():
        if model_id == variant or variant in model_id:
            return base_model, variant
    
    # Try to infer base model from variant
    if "Llama-3.1-8B" in model_id:
        base_model = "meta-llama/Llama-3.1-8B"
    elif "Llama-3.3-70B" in model_id:
        base_model = "meta-llama/Llama-3.3-70B-Instruct"
    elif "Qwen3-1.7B" in model_id:
        base_model = "Qwen/Qwen3-1.7B"
    elif "Qwen3-4B" in model_id:
        base_model = "Qwen/Qwen3-4B"
    elif "gemma-3-27b" in model_id:
        base_model = "google/gemma-3-27b-it"
    elif "Mistral-Small-3.2-24B" in model_id:
        base_model = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    elif "gpt-oss-120b" in model_id:
        base_model = "openai/gpt-oss-120b"
    else:
        # Default: use the model_id as both
        base_model = model_id
    
    # Try to get the correct variant from mapping
    variant = MODEL_MAPPING.get(base_model, model_id)
    
    return base_model, variant


def find_summary_files(benchmark_dir: Path) -> List[Path]:
    """Find all summary.json files in the benchmark directory."""
    return list(benchmark_dir.rglob("summary.json"))


def load_summary_data(summary_file: Path) -> List[Dict[str, Any]]:
    """Load summary.json file and return the list of run results."""
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
            # summary.json should be an array of run results
            if isinstance(data, list):
                return data
            else:
                print(f"Warning: {summary_file} is not a list, wrapping in list")
                return [data]
    except Exception as e:
        print(f"Error loading {summary_file}: {e}")
        return []


def process_benchmark_folder(
    benchmark_dir: Path,
    gpu: str,
    precision: str = "",
    docker_image: str = "",
    model_override: str = "",
    model_variant_override: str = ""
) -> List[Dict[str, Any]]:
    """
    Process all summary.json files in the benchmark directory and return rows for CSV.
    
    Each summary.json contains multiple runs. This function uses only run #3 (run_number=2).
    """
    summary_files = find_summary_files(benchmark_dir)
    
    if not summary_files:
        print(f"No summary.json files found in {benchmark_dir}")
        return []
    
    print(f"Found {len(summary_files)} summary.json files")
    
    rows = []
    
    for summary_file in summary_files:
        runs = load_summary_data(summary_file)
        
        for run in runs:
            # Only use run #3 (run_number == 2, since 0-indexed)
            if run.get("run_number") != 2:
                continue
            
            # Extract model info
            model_id = run.get("model_id", run.get("model", ""))
            
            if model_override and model_variant_override:
                base_model = model_override
                model_variant = model_variant_override
            elif model_override:
                base_model = model_override
                model_variant = model_id
            else:
                base_model, model_variant = extract_model_parts(model_id)
            
            # Try to infer precision if not provided
            inferred_precision = precision
            if not inferred_precision:
                if "FP8" in model_id or "fp8" in model_id.lower():
                    inferred_precision = "FP8"
                elif "INT8" in model_id or "int8" in model_id.lower():
                    inferred_precision = "INT8"
                elif "FP16" in model_id or "fp16" in model_id.lower():
                    inferred_precision = "FP16"
                else:
                    inferred_precision = "Unknown"
            
            # Create row
            row = {
                "Model": base_model,
                "Model Variant": model_variant,
                "GPU": gpu,
                "ISL": run.get("random_input_len", ""),
                "OSL": run.get("random_output_len", ""),
                "Tokens/sec": run.get("output_throughput", ""),
                "Mean TTFT": run.get("mean_ttft_ms", ""),
                "Median TTFT": run.get("median_ttft_ms", ""),
                "P99 TTFT": run.get("p99_ttft_ms", ""),
                "Mean TPOT": run.get("mean_tpot_ms", ""),
                "Median TPOT": run.get("median_tpot_ms", ""),
                "P99 TPOT": run.get("p99_tpot_ms", ""),
                "Mean ITL": run.get("mean_itl_ms", ""),
                "Median ITL": run.get("median_itl_ms", ""),
                "P99 ITL": run.get("p99_itl_ms", ""),
                "Concurrency": run.get("max_concurrency", ""),
                "Precision": inferred_precision,
                "Docker Image": docker_image,
            }
            
            rows.append(row)
    
    return rows


def write_csv(rows: List[Dict[str, Any]], output_file: str):
    """Write rows to CSV file."""
    if not rows:
        print("No data to write to CSV")
        return
    
    
    fieldnames = [
        "Model",
        "Model Variant",
        "GPU",
        "ISL",
        "OSL",
        "Tokens/sec",
        "Mean TTFT",
        "Median TTFT",
        "P99 TTFT",
        "Mean TPOT",
        "Median TPOT",
        "P99 TPOT",
        "Mean ITL",
        "Median ITL",
        "P99 ITL",
        "Concurrency",
        "Precision",
        "Docker Image",
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWrote {len(rows)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM benchmark sweep results and export to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with GPU name
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 --gpu "MI355X"
    
    # With all metadata
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 \\
        --gpu "Radeon 9700 Pro" \\
        --precision "FP8" \\
        --docker-image "vllm/vllm-openai-rocm:latest"
    
    # Override model names
    python3 benchmark_to_csv.py Results/benchmark_260324_1550 \\
        --gpu "MI355X" \\
        --model "meta-llama/Llama-3.1-8B" \\
        --model-variant "amd/Llama-3.1-8B-Instruct-FP8-KV"
        """
    )
    
    parser.add_argument(
        'benchmark_dir',
        type=str,
        help='Path to benchmark results directory (e.g., Results/benchmark_260324_1550)'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        required=True,
        help='GPU name (e.g., "MI355X", "Radeon 9700 Pro")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.csv',
        help='Output CSV filename (default: benchmark_results.csv)'
    )
    
    parser.add_argument(
        '--precision',
        type=str,
        default='',
        help='Precision/quantization (e.g., "FP8", "FP16", "INT8"). Will be inferred from model name if not provided.'
    )
    
    parser.add_argument(
        '--docker-image',
        type=str,
        default='',
        help='Docker image used for benchmarks (e.g., "vllm/vllm-openai-rocm:latest")'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='',
        help='Override base model name (e.g., "meta-llama/Llama-3.1-8B")'
    )
    
    parser.add_argument(
        '--model-variant',
        type=str,
        default='',
        help='Override model variant name (e.g., "amd/Llama-3.1-8B-Instruct-FP8-KV")'
    )
    
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    
    if not benchmark_dir.exists():
        print(f"Error: Directory {benchmark_dir} does not exist")
        return 1
    
    if not benchmark_dir.is_dir():
        print(f"Error: {benchmark_dir} is not a directory")
        return 1
    
    print(f"Processing benchmark directory: {benchmark_dir}")
    print(f"GPU: {args.gpu}")
    if args.precision:
        print(f"Precision: {args.precision}")
    if args.docker_image:
        print(f"Docker Image: {args.docker_image}")
    if args.model:
        print(f"Model Override: {args.model}")
    if args.model_variant:
        print(f"Model Variant Override: {args.model_variant}")
    print()
    
    rows = process_benchmark_folder(
        benchmark_dir,
        args.gpu,
        args.precision,
        args.docker_image,
        args.model,
        args.model_variant
    )
    
    write_csv(rows, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
