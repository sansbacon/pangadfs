#!/usr/bin/env python3
"""
Debug script to understand the populate issue
"""

import numpy as np
import pandas as pd
from pangadfs.populate import PopulateDefault
from pangadfs.pospool import PospoolDefault

def debug_populate():
    """Debug the PopulateDefault issue"""
    
    # Create simple test data
    data = []
    positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    
    for i in range(100):
        pos = positions[i % len(positions)]
        data.append({
            'id': i,
            'name': f'Player_{i}',
            'pos': pos,
            'salary': np.random.randint(3000, 12000),
            'proj': np.random.uniform(5, 25),
            'prob': np.random.uniform(0.01, 0.1)
        })
    
    df = pd.DataFrame(data)
    print("Created dataframe:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Positions: {df['pos'].value_counts()}")
    
    # Create pospool
    posfilter = {'QB': 20, 'RB': 20, 'WR': 20, 'TE': 20, 'DST': 20}
    posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
    
    print(f"\nPosfilter: {posfilter}")
    print(f"Posmap: {posmap}")
    
    pospool = PospoolDefault().pospool(pool=df, posfilter=posfilter, column_mapping={})
    
    print(f"\nPospool keys: {list(pospool.keys())}")
    for pos, pool in pospool.items():
        print(f"  {pos}: {len(pool)} players")
        if len(pool) > 0:
            print(f"    Sample: {pool.head(2)}")
    
    # Test populate
    populate = PopulateDefault()
    try:
        result = populate.populate(pospool=pospool, posmap=posmap, population_size=5)
        print(f"\nResult shape: {result.shape}")
        print(f"Result:\n{result}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_populate()
