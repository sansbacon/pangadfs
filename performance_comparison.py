#!/usr/bin/env python3
"""
Performance comparison between original and optimized PopulateDefault
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to path to import both versions
sys.path.insert(0, os.getcwd())

from pangadfs.pospool import PospoolDefault

# Import original version
from pangadfs.populate_original import PopulateDefault as PopulateOriginal

# Import optimized version
from pangadfs.populate import PopulateDefault as PopulateOptimized

def create_realistic_test_data():
    """Create realistic test data similar to actual usage"""
    # Use the actual test data structure
    data = []
    positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    
    # Create realistic data with proper distributions
    for i in range(200):
        pos = positions[i % len(positions)]
        data.append({
            'player': f'Player_{i}',
            'team': 'TEAM',
            'pos': pos,
            'salary': np.random.randint(3000, 12000),
            'proj': np.random.uniform(5, 25)
        })
    
    return pd.DataFrame(data)

def run_performance_comparison():
    """Compare performance between original and optimized versions"""
    print("Creating realistic test data...")
    df = create_realistic_test_data()
    
    # Use the same setup as the actual tests
    posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
    
    pospool = PospoolDefault().pospool(pool=df, posfilter=posfilter, column_mapping={})
    
    print(f"Pospool created with positions: {list(pospool.keys())}")
    for pos, pool in pospool.items():
        print(f"  {pos}: {len(pool)} players")
    
    population_sizes = [10, 50, 100]
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    for pop_size in population_sizes:
        print(f"\nPopulation size: {pop_size}")
        print("-" * 40)
        
        # Test original version
        populate_orig = PopulateOriginal()
        times_orig = []
        
        try:
            # Warm up
            _ = populate_orig.populate(pospool=pospool, posmap=posmap, population_size=5, probcol='proj')
            
            # Benchmark original
            for _ in range(10):
                start_time = time.perf_counter()
                result_orig = populate_orig.populate(pospool=pospool, posmap=posmap, population_size=pop_size, probcol='proj')
                end_time = time.perf_counter()
                times_orig.append(end_time - start_time)
            
            avg_time_orig = np.mean(times_orig)
            std_time_orig = np.std(times_orig)
            
            print(f"ORIGINAL:")
            print(f"  Average time: {avg_time_orig:.6f}s ± {std_time_orig:.6f}s")
            print(f"  Result shape: {result_orig.shape}")
            print(f"  Time per individual: {avg_time_orig/pop_size*1000:.3f}ms")
            
        except Exception as e:
            print(f"ORIGINAL: ERROR - {e}")
            avg_time_orig = float('inf')
        
        # Test optimized version
        populate_opt = PopulateOptimized()
        times_opt = []
        
        try:
            # Warm up
            _ = populate_opt.populate(pospool=pospool, posmap=posmap, population_size=5, probcol='proj')
            
            # Benchmark optimized
            for _ in range(10):
                start_time = time.perf_counter()
                result_opt = populate_opt.populate(pospool=pospool, posmap=posmap, population_size=pop_size, probcol='proj')
                end_time = time.perf_counter()
                times_opt.append(end_time - start_time)
            
            avg_time_opt = np.mean(times_opt)
            std_time_opt = np.std(times_opt)
            
            print(f"OPTIMIZED:")
            print(f"  Average time: {avg_time_opt:.6f}s ± {std_time_opt:.6f}s")
            print(f"  Result shape: {result_opt.shape}")
            print(f"  Time per individual: {avg_time_opt/pop_size*1000:.3f}ms")
            
            # Compare results
            if avg_time_orig != float('inf'):
                speedup = avg_time_orig / avg_time_opt
                print(f"COMPARISON:")
                if speedup > 1:
                    print(f"  Speedup: {speedup:.2f}x FASTER")
                else:
                    print(f"  Slowdown: {1/speedup:.2f}x SLOWER")
            
        except Exception as e:
            print(f"OPTIMIZED: ERROR - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_performance_comparison()
