#!/usr/bin/env python3
"""
Generate manifest.json for GitHub Pages benchmark viewer.

This script scans the results directory and creates a manifest.json file
that lists all devices and their available benchmarks. The manifest is
used by the index.html page to dynamically display benchmark results.

Usage:
    python generate_manifest.py [--results-dir PATH]

The script will:
1. Scan for device directories (e.g., jetson, pastup)
2. Find all benchmark folders containing report.html
3. Identify the latest benchmark for each device
4. Generate manifest.json in the results directory
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def find_benchmarks(device_path: Path) -> list[str]:
    """Find all benchmark directories containing report.html."""
    benchmarks = []
    
    for item in device_path.iterdir():
        if item.is_dir() and item.name != 'latest':
            report_path = item / 'report.html'
            if report_path.exists():
                benchmarks.append(item.name)
    
    # Sort benchmarks by name (which includes timestamp) in descending order
    benchmarks.sort(reverse=True)
    return benchmarks


def get_latest_benchmark(device_path: Path, benchmarks: list[str]) -> str | None:
    """Determine the latest benchmark for a device."""
    # Check if there's a 'latest' symlink
    latest_link = device_path / 'latest'
    if latest_link.is_symlink():
        target = latest_link.resolve()
        if target.exists():
            return target.name
    
    # Otherwise, return the first (most recent) benchmark
    return benchmarks[0] if benchmarks else None


def parse_benchmark_timestamp(name: str) -> datetime | None:
    """Parse timestamp from benchmark folder name."""
    # Expected format: benchmark_YYYYMMDD_HHMMSS
    try:
        if name.startswith('benchmark_'):
            timestamp_str = name[10:]  # Remove 'benchmark_' prefix
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        pass
    return None


def generate_manifest(results_dir: Path) -> dict:
    """Generate the benchmark manifest."""
    manifest = {}
    
    for device_dir in results_dir.iterdir():
        if device_dir.is_dir() and not device_dir.name.startswith('.'):
            # Skip if this is not a device directory (e.g., if it's a benchmark dir)
            if device_dir.name.startswith('benchmark_'):
                continue
                
            benchmarks = find_benchmarks(device_dir)
            
            if benchmarks:
                latest = get_latest_benchmark(device_dir, benchmarks)
                
                manifest[device_dir.name] = {
                    'benchmarks': benchmarks,
                    'latest': latest,
                    'count': len(benchmarks)
                }
                
                # Add metadata for each benchmark
                benchmark_metadata = {}
                for benchmark in benchmarks:
                    timestamp = parse_benchmark_timestamp(benchmark)
                    metadata = {
                        'has_report': True
                    }
                    if timestamp:
                        metadata['timestamp'] = timestamp.isoformat()
                        metadata['display_name'] = timestamp.strftime('%Y-%m-%d %H:%M')
                    
                    # Check for additional files
                    benchmark_path = device_dir / benchmark
                    if (benchmark_path / 'results.json').exists():
                        metadata['has_results_json'] = True
                    if (benchmark_path / 'plots').is_dir():
                        metadata['has_plots'] = True
                    
                    benchmark_metadata[benchmark] = metadata
                
                manifest[device_dir.name]['metadata'] = benchmark_metadata
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Generate manifest.json for benchmark viewer'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path(__file__).parent,
        help='Path to the results directory (default: script directory)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for manifest.json (default: results-dir/manifest.json)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        default=True,
        help='Pretty print JSON output (default: True)'
    )
    
    args = parser.parse_args()
    
    results_dir = args.results_dir.resolve()
    output_path = args.output or (results_dir / 'manifest.json')
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning results directory: {results_dir}")
    
    manifest = generate_manifest(results_dir)
    
    if not manifest:
        print("Warning: No benchmarks found!", file=sys.stderr)
    else:
        device_count = len(manifest)
        benchmark_count = sum(d['count'] for d in manifest.values())
        print(f"Found {device_count} device(s) with {benchmark_count} total benchmark(s)")
        
        for device, data in manifest.items():
            print(f"  - {device}: {data['count']} benchmark(s), latest: {data['latest']}")
    
    # Write manifest
    with open(output_path, 'w') as f:
        if args.pretty:
            json.dump(manifest, f, indent=2)
        else:
            json.dump(manifest, f)
    
    print(f"Manifest written to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
