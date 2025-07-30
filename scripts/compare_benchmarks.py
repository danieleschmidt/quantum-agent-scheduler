#!/usr/bin/env python3
"""
Compare benchmark results between baseline and PR.

This script analyzes performance differences and generates a markdown report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def load_benchmark_data(filepath: str) -> Dict:
    """Load benchmark JSON data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def analyze_performance_change(baseline: Dict, current: Dict) -> Dict:
    """Analyze performance changes between baseline and current benchmarks."""
    results = {}
    
    baseline_benchmarks = {b['name']: b for b in baseline.get('benchmarks', [])}
    current_benchmarks = {b['name']: b for b in current.get('benchmarks', [])}
    
    for name in set(baseline_benchmarks.keys()) | set(current_benchmarks.keys()):
        if name in baseline_benchmarks and name in current_benchmarks:
            baseline_time = baseline_benchmarks[name]['stats']['mean']
            current_time = current_benchmarks[name]['stats']['mean']
            
            change_percent = ((current_time - baseline_time) / baseline_time) * 100
            
            results[name] = {
                'baseline_time': baseline_time,
                'current_time': current_time,
                'change_percent': change_percent,
                'status': 'regression' if change_percent > 5 else 'improvement' if change_percent < -5 else 'stable'
            }
        elif name in baseline_benchmarks:
            results[name] = {
                'baseline_time': baseline_benchmarks[name]['stats']['mean'],
                'current_time': None,
                'change_percent': None,
                'status': 'removed'
            }
        else:
            results[name] = {
                'baseline_time': None,
                'current_time': current_benchmarks[name]['stats']['mean'],
                'change_percent': None,
                'status': 'new'
            }
    
    return results

def generate_markdown_report(analysis: Dict) -> str:
    """Generate markdown report from analysis."""
    report = []
    
    # Summary
    regressions = [name for name, data in analysis.items() if data['status'] == 'regression']
    improvements = [name for name, data in analysis.items() if data['status'] == 'improvement']
    new_tests = [name for name, data in analysis.items() if data['status'] == 'new']
    removed_tests = [name for name, data in analysis.items() if data['status'] == 'removed']
    
    report.append("### Performance Summary")
    report.append("")
    report.append(f"- 🔴 **Regressions**: {len(regressions)}")
    report.append(f"- 🟢 **Improvements**: {len(improvements)}")
    report.append(f"- 🆕 **New tests**: {len(new_tests)}")
    report.append(f"- ❌ **Removed tests**: {len(removed_tests)}")
    report.append("")
    
    # Detailed results
    if regressions:
        report.append("### 🔴 Performance Regressions")
        report.append("")
        report.append("| Test | Baseline (s) | Current (s) | Change |")
        report.append("|------|--------------|-------------|---------|")
        
        for name in regressions:
            data = analysis[name]
            report.append(f"| `{name}` | {data['baseline_time']:.4f} | {data['current_time']:.4f} | +{data['change_percent']:.1f}% |")
        report.append("")
    
    if improvements:
        report.append("### 🟢 Performance Improvements")
        report.append("")
        report.append("| Test | Baseline (s) | Current (s) | Change |")
        report.append("|------|--------------|-------------|---------|")
        
        for name in improvements:
            data = analysis[name]
            report.append(f"| `{name}` | {data['baseline_time']:.4f} | {data['current_time']:.4f} | {data['change_percent']:.1f}% |")
        report.append("")
    
    if new_tests:
        report.append("### 🆕 New Benchmarks")
        report.append("")
        for name in new_tests:
            data = analysis[name]
            report.append(f"- `{name}`: {data['current_time']:.4f}s")
        report.append("")
    
    if removed_tests:
        report.append("### ❌ Removed Benchmarks")
        report.append("")
        for name in removed_tests:
            report.append(f"- `{name}`")
        report.append("")
    
    # Recommendations
    if regressions:
        report.append("### 📋 Recommendations")
        report.append("")
        report.append("Performance regressions detected. Consider:")
        report.append("- Reviewing algorithmic changes in affected functions")
        report.append("- Checking for memory leaks or excessive allocations")
        report.append("- Profiling the code to identify bottlenecks")
        report.append("- Adding performance optimization as a follow-up task")
        report.append("")
    
    return "\n".join(report)

def main():
    """Main comparison function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <baseline.json> <current.json>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    
    baseline_data = load_benchmark_data(baseline_file)
    current_data = load_benchmark_data(current_file)
    
    analysis = analyze_performance_change(baseline_data, current_data)
    report = generate_markdown_report(analysis)
    
    print(report)

if __name__ == '__main__':
    main()