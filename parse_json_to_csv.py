#!/usr/bin/env python3
"""
Script to parse JSON benchmark files and export all values to a CSV file.

This script searches through specified folders for JSON files containing benchmark data
and consolidates all the data into a single CSV file with additional metadata columns
for folder name and parsed filename components.

Parent folder name can be used to designate the GPU that was used.  e.g. add all json files
into a MI300X folder for results from that GPU.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


# Configure which folders to search for JSON files
# SEARCH_FOLDERS = [
#     "mi300",
#     "mi350",
#     "mi355",
# ]
SEARCH_FOLDERS = [
    ".",
]

# Mapping from source column names in the all-data CSV to summary CSV column names
SUMMARY_COLUMN_MAP = {
    "model_id": "Model",
    "folder_name": "GPU",
    "input_token_length": "ISL",
    "output_token_length": "OSL",
    "output_throughput": "Tokens/sec",
    "mean_ttft_ms": "Mean TTFT",
    "median_ttft_ms": "Median TTFT",
    "p99_ttft_ms": "P99 TTFT",
    "mean_tpot_ms": "Mean TPOT",
    "median_tpot_ms": "Median TPOT",
    "p99_tpot_ms": "P99 TPOT",
    "mean_itl_ms": "Mean ITL",
    "median_itl_ms": "Median ITL",
    "p99_itl_ms": "P99 ITL",
    "max_concurrency": "Concurrency",
    "datatype": "Precision",
    "docker_image": "Docker Image",}


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename pattern like:
    benchsvr_Llama-3.3-70B-Instruct-FP8-KV_tp1_128-2048_mc128_t1.json
    
    Returns dictionary with parsed components.
    """
    metadata = {}
    
    # Remove file extension
    name_without_ext = filename.replace('.json', '')
    
    # Split by underscores to get components
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        metadata['benchmark_type'] = parts[0]  # e.g., "benchsvr"
        metadata['model_name'] = parts[1] if len(parts) > 1 else ""
        
        # Parse remaining parts for tp, input-output, mc, and trial info
        for part in parts[2:]:
            if part.startswith('tp'):
                metadata['tensor_parallel'] = part
            elif '-' in part and part.replace('-', '').isdigit():
                # This looks like input-output token pattern
                metadata['input_output_tokens'] = part
                input_tokens, output_tokens = part.split('-', 1)
                metadata['input_token_length'] = input_tokens
                metadata['output_token_length'] = output_tokens
            elif part.startswith('mc'):
                metadata['max_concurrency'] = part
            elif part.startswith('t') and part[1:].isdigit():
                metadata['trial_number'] = part
            elif part.startswith('np') and part[2:].isdigit():
                metadata['num_prompts'] = part
    
    return metadata


def load_env_file(env_path: str = '.env') -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    Returns dictionary of key-value pairs.
    """
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: .env file not found at {env_path}")
    except Exception as e:
        print(f"Warning: Error reading .env file: {e}")
    
    return env_vars


def find_json_files(base_path: str, folders: List[str]) -> List[tuple]:
    """
    Find all JSON files in the specified folders.
    
    Returns list of tuples: (folder_name, file_path)
    """
    json_files = []
    base_path = Path(base_path)

    def get_folder_label(json_file: Path) -> str:
        parent_dir = json_file.parent
        if parent_dir == base_path:
            return "."
        return parent_dir.name
    
    for folder in folders:
        folder_path = base_path / folder
        if folder_path.exists() and folder_path.is_dir():
            # Search recursively for JSON files
            for json_file in folder_path.rglob("*.json"):
                folder_label = get_folder_label(json_file)
                json_files.append((folder_label, str(json_file)))
        else:
            print(f"Warning: Folder '{folder}' not found in {base_path}")
    
    return json_files


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load and return JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def collect_all_keys(json_files: List[tuple]) -> set:
    """
    Collect all unique keys from all JSON files to ensure comprehensive CSV headers.
    """
    all_keys = set()
    
    for folder, file_path in json_files:
        data = load_json_data(file_path)
        all_keys.update(data.keys())
    
    return all_keys


def get_all_output_filename(output_file: str) -> str:
    """Return all-data CSV name by converting <name>.csv to <name>.all.csv."""
    if output_file.lower().endswith('.csv'):
        return output_file[:-4] + '.all.csv'
    return output_file + '.all.csv'


def export_all_to_csv(json_files: List[tuple], output_file: str, datatype: str = "", docker_image: str = "", gpu_name: str = ""):
    """
    Export all JSON data to an all-data CSV file.
    """
    if not json_files:
        print("No JSON files found to process.")
        return None
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    # Collect all possible keys from all files
    json_keys = collect_all_keys(json_files)
    
    # Metadata columns that we'll add
    metadata_columns = [
        'folder_name', 
        'tensor_parallel',
        'input_output_tokens',
        'input_token_length',
        'output_token_length',
        'max_concurrency_parsed',
        'trial_number',
        'num_prompts_parsed',
        'datatype',
        'docker_image'
    ]
    
    # Create comprehensive header with metadata first, then all JSON keys
    headers = metadata_columns + sorted(list(json_keys))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for folder, file_path in json_files:
            try:
                # Load JSON data
                json_data = load_json_data(file_path)
                
                if not json_data:
                    continue
                
                # Parse filename metadata
                filename = os.path.basename(file_path)
                filename_metadata = parse_filename_metadata(filename)
                
                # Create row with metadata
                # Use gpu_name if provided, otherwise use folder name
                gpu_label = gpu_name if gpu_name else folder
                row = {
                    'folder_name': gpu_label,
                    'tensor_parallel': filename_metadata.get('tensor_parallel', ''),
                    'input_output_tokens': filename_metadata.get('input_output_tokens', ''),
                    'input_token_length': filename_metadata.get('input_token_length', ''),
                    'output_token_length': filename_metadata.get('output_token_length', ''),
                    'max_concurrency_parsed': filename_metadata.get('max_concurrency', ''),
                    'trial_number': filename_metadata.get('trial_number', ''),
                    'num_prompts_parsed': filename_metadata.get('num_prompts', ''),
                    'datatype': datatype,
                    'docker_image': docker_image
                }
                
                # Add all JSON data to the row
                row.update(json_data)
                
                writer.writerow(row)
                print(f"Processed: {folder}/{filename}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"\nData exported to: {output_file}")
    print(f"Total columns: {len(headers)}")
    return output_file


def export_summary_csv_from_all(all_csv_file: str, summary_output_file: str, column_map: Dict[str, str]):
    """
    Create a summary CSV as a subset of columns from the all-data CSV.
    """
    with open(all_csv_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile)

        missing_columns = [column for column in column_map.keys() if column not in (reader.fieldnames or [])]
        if missing_columns:
            print(
                "Warning: Missing columns in all-data CSV for summary export: "
                + ", ".join(missing_columns)
            )

        summary_headers = list(column_map.values())
        with open(summary_output_file, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=summary_headers)
            writer.writeheader()

            for row in reader:
                summary_row = {
                    output_name: row.get(source_name, '')
                    for source_name, output_name in column_map.items()
                }
                writer.writerow(summary_row)

    print(f"Summary data exported to: {summary_output_file}")
    print(f"Summary columns: {', '.join(summary_headers)}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse JSON benchmark files and export to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default folders to search: {', '.join(SEARCH_FOLDERS)}

Example usage:
    python parse_json_to_csv.py
    python parse_json_to_csv.py --output benchmark_results.csv
    python parse_json_to_csv.py --base-path /path/to/data --output results.csv
        """
    )
    
    parser.add_argument(
        '--base-path',
        default='.',
        help='Base directory to search for folders (default: current directory)'
    )
    
    parser.add_argument(
        '--output', 
        default='benchmark_results.csv',
        help='Output CSV filename (default: benchmark_results.csv)'
    )
    
    parser.add_argument(
        '--folders',
        nargs='*',
        default=SEARCH_FOLDERS,
        help=f'Space separated list of folders to search.  Folder names are also added as a CSV field. (default: {" ".join(SEARCH_FOLDERS)})'
    )
    
    parser.add_argument(
        '--list-folders',
        action='store_true',
        help='List available folders and exit'
    )
    
    parser.add_argument(
        '--datatype',
        default=None,
        help='Data type/precision label to add as metadata (e.g., FP8, FP16, BF16). If not specified, reads from VLLM_QUANTIZATION in .env file'
    )
    
    parser.add_argument(
        '--docker-image',
        default=None,
        help='Docker image name/tag to add as metadata (e.g., vllm/vllm-openai-rocm:latest). If not specified, reads from VLLM_IMAGE in .env file'
    )
    
    parser.add_argument(
        '--gpu',
        default=None,
        help='GPU model name to add as metadata (e.g., MI300X, MI350X). If not specified, reads from VLLM_GPU in .env file or uses folder name'
    )
    
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file for reading VLLM_QUANTIZATION, VLLM_IMAGE, and VLLM_GPU (default: .env)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    env_vars = load_env_file(args.env_file)
    
    # Use values from .env if not provided via command line
    datatype = args.datatype if args.datatype is not None else env_vars.get('VLLM_QUANTIZATION', '')
    docker_image = args.docker_image if args.docker_image is not None else env_vars.get('VLLM_IMAGE', '')
    gpu_name = args.gpu if args.gpu is not None else env_vars.get('GPU_MODEL', '')
    
    # Convert datatype to uppercase for consistency (fp8 -> FP8)
    if datatype:
        datatype = datatype.upper()
    
    # List folders option
    if args.list_folders:
        base_path = Path(args.base_path)
        print(f"Available folders in {base_path}:")
        for item in sorted(base_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                json_count = len(list(item.rglob("*.json")))
                print(f"  {item.name} ({json_count} JSON files)")
        return
    
    # Find all JSON files in specified folders
    json_files = find_json_files(args.base_path, args.folders)
    
    if not json_files:
        print("No JSON files found in the specified folders.")
        print("Use --list-folders to see available folders.")
        return
    
    # Export full data to <name>.all.csv
    all_output_file = get_all_output_filename(args.output)
    exported_all_file = export_all_to_csv(json_files, all_output_file, datatype, docker_image, gpu_name)

    # Export summary data to expected output filename
    if exported_all_file:
        export_summary_csv_from_all(exported_all_file, args.output, SUMMARY_COLUMN_MAP)
        
    # Print configuration used
    print(f"\nConfiguration used:")
    print(f"  GPU: {gpu_name if gpu_name else '(using folder names)'}")
    print(f"  Precision: {datatype if datatype else '(not set)'}")
    print(f"  Docker Image: {docker_image if docker_image else '(not set)'}")


if __name__ == "__main__":
    main()