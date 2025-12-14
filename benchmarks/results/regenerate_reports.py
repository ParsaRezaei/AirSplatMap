#!/usr/bin/env python3
"""Regenerate HTML reports for all benchmark runs with results.json."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.visualization.html_report import generate_html_report

def main():
    results_base = Path(__file__).parent
    success = 0
    failed = 0
    skipped = 0

    for host_dir in sorted(results_base.iterdir()):
        if not host_dir.is_dir() or host_dir.name.startswith('.'):
            continue
        if host_dir.name in ['manifest.json', 'index.html', 'README.md']:
            continue
        
        print(f'\n=== {host_dir.name} ===')
        
        for benchmark_dir in sorted(host_dir.iterdir()):
            if not benchmark_dir.is_dir() or benchmark_dir.name == 'latest' or benchmark_dir.is_symlink():
                continue
            
            results_json = benchmark_dir / 'results.json'
            if not results_json.exists():
                print(f'  - {benchmark_dir.name}: no results.json')
                skipped += 1
                continue
            
            try:
                with open(results_json) as f:
                    data = json.load(f)
                
                output_html = benchmark_dir / 'report.html'
                generate_html_report(
                    data.get('pose', []),
                    data.get('depth', []),
                    data.get('gs', []),
                    data.get('pipeline', []),
                    data.get('hardware', {}),
                    output_html
                )
                print(f'  ✓ {benchmark_dir.name}')
                success += 1
            except Exception as e:
                print(f'  ✗ {benchmark_dir.name}: {e}')
                failed += 1

    print(f'\n=== Summary ===')
    print(f'  Generated: {success}')
    print(f'  Failed: {failed}')
    print(f'  Skipped (no results.json): {skipped}')

if __name__ == '__main__':
    main()
