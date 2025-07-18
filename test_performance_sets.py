#!/usr/bin/env python3

"""
Quick performance test for the improved set-based multilineup optimization
"""

import time
import logging
from pathlib import Path
import sys

# Add pangadfs to path
sys.path.insert(0, str(Path(__file__).parent))

from pangadfs.ga import GeneticAlgorithm
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_performance():
    """Test the performance of the improved set-based optimizer"""
    
    # Configuration for set-based optimizer
    config = {
        'ga_settings': {
            'csvpth': Path('pangadfs/app/appdata/pool.csv'),
            'population_size': 100,  # Small population for quick test
            'n_generations': 5,      # Few generations for quick test
            'target_lineups': 20,    # Small number of lineups
            'lineup_pool_size': 10000,  # Smaller pool for testing
            'set_diversity_weight': 0.3,
            'tournament_size': 3,
            'mutation_rate': 0.05,
            'elite_divisor': 5,
            'stop_criteria': 10,
            'crossover_method': 'uniform',
            'select_method': 'tournament',
            'elite_method': 'fittest',
            'verbose': True,
            'enable_profiling': False,
            'points_column': 'proj',
            'position_column': 'pos',
            'salary_column': 'salary'
        },
        'site_settings': {
            'salary_cap': 50000,
            'lineup_size': 9,
            'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1},
            'flex_positions': ('RB', 'WR', 'TE'),
            'posfilter': {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'FLEX': 0}
        }
    }
    
    # Set up driver managers
    dmgrs = {}
    emgrs = {}
    
    for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
        pns = f'pangadfs.{ns}'
        if ns == 'validate':
            emgrs['validate'] = NamedExtensionManager(
                namespace=pns, 
                names=['validate_salary', 'validate_duplicates'], 
                invoke_on_load=True, 
                name_order=True)
        elif ns == 'optimize':
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name='optimize_multilineup_sets',  # Use the new set-based optimizer
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    print("Testing improved set-based multilineup optimization performance...")
    print("Configuration:")
    print(f"  - Population size: {config['ga_settings']['population_size']}")
    print(f"  - Generations: {config['ga_settings']['n_generations']}")
    print(f"  - Target lineups: {config['ga_settings']['target_lineups']}")
    print(f"  - Lineup pool size: {config['ga_settings']['lineup_pool_size']}")
    print()
    
    # Run optimization and measure time
    start_time = time.time()
    
    try:
        ga = GeneticAlgorithm(ctx=config, driver_managers=dmgrs, extension_managers=emgrs)
        results = ga.optimize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("✅ Optimization completed successfully!")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print()
        
        # Analyze results
        if 'lineups' in results and results['lineups']:
            lineups = results['lineups']
            print("📊 Results Analysis:")
            print(f"  - Generated {len(lineups)} lineups")
            print(f"  - Best score: {results.get('best_score', 'N/A')}")
            
            # Calculate diversity metrics
            if len(lineups) > 1:
                from pangadfs.misc import calculate_jaccard_diversity
                
                total_overlap = 0
                comparisons = 0
                
                for i, item in enumerate(lineups):
                    for j in range(i + 1, len(lineups)):
                        diversity = calculate_jaccard_diversity(item, lineups[j])
                        overlap = 1.0 - diversity
                        total_overlap += overlap
                        comparisons += 1
                
                avg_overlap = total_overlap / comparisons if comparisons > 0 else 0
                print(f"  - Average overlap: {avg_overlap:.3f}")
                print(f"  - Average diversity: {1.0 - avg_overlap:.3f}")
        
        print()
        print("🎉 Performance test completed successfully!")
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"❌ Optimization failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    test_performance()
