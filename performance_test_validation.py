#!/usr/bin/env python3
"""
Performance test comparing populate with and without duplicate handling
"""

import time
import numpy as np
import pandas as pd
from pangadfs.populate import PopulateDefault
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

def test_performance():
    """Test performance of new approach"""
    print("Creating test data...")
    df = create_test_data()
    
    # Setup
    posfilter = {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6}
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 7}
    
    pospool = PospoolDefault().pospool(pool=df, posfilter=posfilter, column_mapping={})
    populate = PopulateDefault()
    flex_validator = FlexDuplicatesValidate()
    dup_validator = DuplicatesValidate()
    
    population_sizes = [50, 100, 500, 1000]
    
    print("\n" + "="*80)
    print("PERFORMANCE TEST: POPULATE + VALIDATION APPROACH")
    print("="*80)
    
    for pop_size in population_sizes:
        print(f"\nPopulation size: {pop_size}")
        print("-" * 40)
        
        # Test new approach: fast populate + validation
        times_new = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            # Fast population creation (no duplicate checking)
            population = populate.populate(
                pospool=pospool, 
                posmap=posmap, 
                population_size=pop_size, 
                probcol='proj'
            )
            
            # Validate duplicates in separate step
            population = dup_validator.validate(population=population)
            population = flex_validator.validate(population=population, posmap=posmap)
            
            end_time = time.perf_counter()
            times_new.append(end_time - start_time)
        
        avg_time_new = np.mean(times_new)
        std_time_new = np.std(times_new)
        
        print(f"NEW APPROACH (Populate + Validation):")
        print(f"  Average time: {avg_time_new:.6f}s ± {std_time_new:.6f}s")
        print(f"  Final population size: {len(population)}")
        print(f"  Time per individual: {avg_time_new/pop_size*1000:.3f}ms")
        
        # Show breakdown
        # Test just populate
        times_populate = []
        for _ in range(10):
            start_time = time.perf_counter()
            population = populate.populate(
                pospool=pospool, 
                posmap=posmap, 
                population_size=pop_size, 
                probcol='proj'
            )
            end_time = time.perf_counter()
            times_populate.append(end_time - start_time)
        
        avg_populate = np.mean(times_populate)
        
        # Test just validation
        population = populate.populate(
            pospool=pospool, 
            posmap=posmap, 
            population_size=pop_size, 
            probcol='proj'
        )
        
        times_validate = []
        for _ in range(10):
            start_time = time.perf_counter()
            validated = dup_validator.validate(population=population)
            validated = flex_validator.validate(population=validated, posmap=posmap)
            end_time = time.perf_counter()
            times_validate.append(end_time - start_time)
        
        avg_validate = np.mean(times_validate)
        
        print(f"  Breakdown:")
        print(f"    Populate only: {avg_populate:.6f}s ({avg_populate/avg_time_new*100:.1f}%)")
        print(f"    Validate only: {avg_validate:.6f}s ({avg_validate/avg_time_new*100:.1f}%)")

if __name__ == "__main__":
    test_performance()
