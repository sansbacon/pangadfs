#!/usr/bin/env python3
"""
Direct comparison between original and new approach
"""

import time
import numpy as np
import pandas as pd
from pangadfs.populate import PopulateDefault
from pangadfs.populate_original import PopulateDefault as PopulateOriginal
from pangadfs.validate import FlexDuplicatesValidate, DuplicatesValidate
from pangadfs.pospool import PospoolDefault

def create_test_data():
    """Create realistic test data"""
    data = []
    positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    
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

def test_comparison():
    """Compare original vs new approach"""
    print("Creating test data...")
    df = create_test_data()
    
    # Setup
    posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
    
    pospool = PospoolDefault().pospool(pool=df, posfilter=posfilter, column_mapping={})
    
    # Initialize both approaches
    populate_original = PopulateOriginal()
    populate_new = PopulateDefault()
    flex_validator = FlexDuplicatesValidate()
    dup_validator = DuplicatesValidate()
    
    population_sizes = [50, 100, 500]
    
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs NEW APPROACH")
    print("="*80)
    
    for pop_size in population_sizes:
        print(f"\nPopulation size: {pop_size}")
        print("-" * 50)
        
        # Test original approach
        times_original = []
        result_original = None
        
        try:
            for _ in range(10):
                start_time = time.perf_counter()
                result_original = populate_original.populate(
                    pospool=pospool, 
                    posmap=posmap, 
                    population_size=pop_size, 
                    probcol='proj'
                )
                end_time = time.perf_counter()
                times_original.append(end_time - start_time)
            
            avg_time_original = np.mean(times_original)
            std_time_original = np.std(times_original)
            
            print(f"ORIGINAL APPROACH:")
            print(f"  Average time: {avg_time_original:.6f}s ± {std_time_original:.6f}s")
            print(f"  Result shape: {result_original.shape}")
            print(f"  Time per individual: {avg_time_original/pop_size*1000:.3f}ms")
            
        except Exception as e:
            print(f"ORIGINAL APPROACH: ERROR - {e}")
            avg_time_original = float('inf')
        
        # Test new approach
        times_new = []
        result_new = None
        
        try:
            for _ in range(10):
                start_time = time.perf_counter()
                
                # Fast population creation
                population = populate_new.populate(
                    pospool=pospool, 
                    posmap=posmap, 
                    population_size=pop_size, 
                    probcol='proj'
                )
                
                # Validation step (optional - can be skipped for pure speed)
                # population = dup_validator.validate(population=population)
                # population = flex_validator.validate(population=population, posmap=posmap)
                result_new = population
                
                end_time = time.perf_counter()
                times_new.append(end_time - start_time)
            
            avg_time_new = np.mean(times_new)
            std_time_new = np.std(times_new)
            
            print(f"NEW APPROACH (Fast Populate):")
            print(f"  Average time: {avg_time_new:.6f}s ± {std_time_new:.6f}s")
            print(f"  Result shape: {result_new.shape}")
            print(f"  Time per individual: {avg_time_new/pop_size*1000:.3f}ms")
            
            # Calculate speedup
            if avg_time_original != float('inf'):
                speedup = avg_time_original / avg_time_new
                print(f"SPEEDUP: {speedup:.2f}x FASTER")
            
        except Exception as e:
            print(f"NEW APPROACH: ERROR - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_comparison()
