#!/usr/bin/env python3
"""
Generate benchmark comparison report from existing AirSplatMap outputs.

Usage:
    python scripts/generate_benchmark_report.py --output-root ./output
    python scripts/generate_benchmark_report.py --output-root ./output --output benchmark_report.md
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import (
    load_existing_results,
    generate_comparison_report,
    BenchmarkResult
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from existing outputs"
    )
    parser.add_argument(
        "--output-root", "-o",
        type=str,
        default="./output",
        help="Root directory containing engine outputs"
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        default=None,
        help="Output path for report (default: print to stdout)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Also save results to JSON file"
    )
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    if not output_root.exists():
        print(f"Error: Output root '{output_root}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading results from {output_root}...", file=sys.stderr)
    results = load_existing_results(output_root)
    
    print(f"Found {len(results.results)} results:", file=sys.stderr)
    for engine in results.get_engines():
        datasets = [r.dataset_name for r in results.get_by_engine(engine)]
        print(f"  - {engine}: {', '.join(datasets)}", file=sys.stderr)
    
    if args.format == "json":
        import json
        output = json.dumps([r.to_dict() for r in results.results], indent=2)
    else:
        output = generate_comparison_report(results)
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Saved report to {args.output}", file=sys.stderr)
    else:
        print(output)
    
    if args.save_json:
        results.save(args.save_json)


if __name__ == "__main__":
    main()
