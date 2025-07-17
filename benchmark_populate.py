#!/usr/bin/env python3
"""
Benchmark script to test PopulateDefault performance improvements
"""

import time
import numpy as np
import pandas as pd
from pangadfs.populate import PopulateDefault
from pangadfs.pospool import PospoolDefault
from pangadfs.pool import PoolDefault

def create_test_data(n_players=1000):
    """Create test data for benchmarking"""
    positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    
    # Create a larger test pool with balanced positions
    data = []
    for i in range(n_players):
        # Ensure we have players in each position
        pos = positions[i % len(positions)]
        data.append({
            'id': i,
            'name': f'Player_{i}',
            'pos': pos,
            'salary': np.random.randint(3000, 12000),
            'proj': np.random.uniform(5, 25),
            'prob': np.random.uniform(0.01, 0.1)
        })
    
    return pd.DataFrame(data)

def benchmark_populate(population_sizes=[50, 100, 500, 1000]):
    """Benchmark the PopulateDefault performance"""
    print("Creating test data...")
    df = create_test_data(1000)
    
    # Create pospool with reasonable filter values
    posfilter = {'QB': 20, 'RB': 20, 'WR': 20, 'TE': 20, 'DST': 20}
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1}
    
    pospool = PospoolDefault().pospool(pool=df, posfilter=posfilter, column_mapping={})
    populate = PopulateDefault()
    
    print("\nBenchmarking PopulateDefault performance:")
    print("=" * 60)
    
    for pop_size in population_sizes:
        print(f"\nPopulation size: {pop_size}")
        
        # Warm up
        _ = populate.populate(pospool=pospool, posmap=posmap, population_size=10)
        
        # Benchmark
        times = []
        for _ in range(5):  # Run 5 times for average
            start_time = time.perf_counter()
            result = populate.populate(pospool=pospool, posmap=posmap, population_size=pop_size)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Result shape: {result.shape}")
        print(f"  Time per individual: {avg_time/pop_size*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_populate()
