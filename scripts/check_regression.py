#!/usr/bin/env python3
import subprocess
import json
import re
import sys
import os

def load_baselines():
    with open('performance_baselines.json', 'r') as f:
        return json.load(f)

def run_benchmark():
    print("Running benchmark suite...")
    # Run with --release for accurate performance numbers
    result = subprocess.run(
        ['cargo', 'run', '--release', '--bin', 'benchmark'], 
        capture_output=True, 
        text=True
    )
    if result.returncode != 0:
        print("Benchmark failed to run:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def parse_results(output):
    results = {}
    
    # Parse Throughput: 49387.39 items/sec
    throughput_match = re.search(r'Throughput: ([\d\.]+) items/sec', output)
    if throughput_match:
        results['batching_throughput'] = float(throughput_match.group(1))
        
    # Parse Average latency: 5.82 µs/op
    latency_match = re.search(r'Average latency: ([\d\.]+) µs/op', output)
    if latency_match:
        results['kv_cache_latency'] = float(latency_match.group(1))
        
    return results

def check_regression(baselines, results):
    failed = False
    
    print("\n--- Regression Check ---")
    
    # Check Throughput (Higher is better)
    min_throughput = baselines['batching_throughput_min_items_per_sec']
    actual_throughput = results.get('batching_throughput', 0)
    
    if actual_throughput >= min_throughput:
        print(f"✅ Batching Throughput: {actual_throughput:.2f} >= {min_throughput:.2f} items/sec")
    else:
        print(f"❌ Batching Throughput: {actual_throughput:.2f} < {min_throughput:.2f} items/sec")
        failed = True
        
    # Check Latency (Lower is better)
    max_latency = baselines['kv_cache_latency_max_micros']
    actual_latency = results.get('kv_cache_latency', float('inf'))
    
    if actual_latency <= max_latency:
        print(f"✅ KV Cache Latency: {actual_latency:.2f} <= {max_latency:.2f} µs/op")
    else:
        print(f"❌ KV Cache Latency: {actual_latency:.2f} > {max_latency:.2f} µs/op")
        failed = True
        
    return not failed

def main():
    if not os.path.exists('performance_baselines.json'):
        print("Error: performance_baselines.json not found")
        sys.exit(1)
        
    baselines = load_baselines()
    output = run_benchmark()
    results = parse_results(output)
    
    if not results:
        print("Error: Could not parse benchmark results")
        print("Output was:")
        print(output)
        sys.exit(1)
        
    success = check_regression(baselines, results)
    
    if success:
        print("\n✨ Regression check PASSED")
        sys.exit(0)
    else:
        print("\n💀 Regression check FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
