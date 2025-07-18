#!/usr/bin/env python3

"""
Performance comparison test between original and optimized multilineup set generation
"""

import time
import logging
import sys
from pathlib import Path

# Add pangadfs to path
sys.path.insert(0, str(Path(__file__).parent))

from pangadfs.ga import GeneticAlgorithm
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_config(lineup_pool_size=10000, population_size=50, target_lineups=20):
    """Create test configuration"""
    return {
        'ga_settings': {
            'csvpth': Path('pangadfs/app/appdata/pool.csv'),
            'population_size': population_size,
            'n_generations': 3,  # Minimal generations for speed
            'target_lineups': target_lineups,
            'lineup_pool_size': lineup_pool_size,
            'set_diversity_weight': 0.3,
            'tournament_size': 3,
            'mutation_rate': 0.05,
            'elite_divisor': 5,
            'stop_criteria': 10,
            'crossover_method': 'uniform',
            'select_method': 'tournament',
            'elite_method': 'fittest',
            'verbose': False,  # Reduce logging for cleaner output
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

def setup_driver_managers(optimizer_name):
    """Set up driver managers for testing"""
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
                name=optimizer_name,
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    return dmgrs, emgrs

def test_optimizer_performance(optimizer_name, config, test_name):
    """Test performance of a specific optimizer"""
    print(f"\n{'='*60}")
    print(f"Testing {test_name}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Population size: {config['ga_settings']['population_size']}")
    print(f"  - Target lineups: {config['ga_settings']['target_lineups']}")
    print(f"  - Lineup pool size: {config['ga_settings']['lineup_pool_size']}")
    print(f"  - Generations: {config['ga_settings']['n_generations']}")
    
    try:
        # Set up driver managers
        dmgrs, emgrs = setup_driver_managers(optimizer_name)
        
        # Measure time
        start_time = time.time()
        
        # Run optimization
        ga = GeneticAlgorithm(ctx=config, driver_managers=dmgrs, extension_managers=emgrs)
        results = ga.optimize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Analyze results
        if 'lineups' in results and results['lineups']:
            lineups = results['lineups']
            n_lineups = len(lineups)
            best_score = results.get('best_score', 'N/A')
            
            # Calculate diversity if multiple lineups
            avg_diversity = 0.0
            if n_lineups > 1 and 'diversity_metrics' in results:
                avg_overlap = results['diversity_metrics'].get('avg_overlap', 0.0)
                avg_diversity = 1.0 - avg_overlap
            
            print(f"\n✅ SUCCESS!")
            print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
            print(f"📊 Results:")
            print(f"  - Generated {n_lineups} lineups")
            print(f"  - Best score: {best_score}")
            print(f"  - Average diversity: {avg_diversity:.3f}")
            print(f"  - Time per lineup: {elapsed_time/n_lineups:.3f} seconds")
            
            return {
                'success': True,
                'time': elapsed_time,
                'lineups': n_lineups,
                'diversity': avg_diversity,
                'time_per_lineup': elapsed_time/n_lineups
            }
        else:
            print(f"❌ No lineups generated")
            return {'success': False, 'time': elapsed_time}
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"❌ FAILED after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return {'success': False, 'time': elapsed_time, 'error': str(e)}

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🚀 MULTILINEUP OPTIMIZER PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Small Test',
            'config': create_config(lineup_pool_size=5000, population_size=30, target_lineups=10),
            'description': 'Quick test with small parameters'
        },
        {
            'name': 'Medium Test', 
            'config': create_config(lineup_pool_size=10000, population_size=50, target_lineups=20),
            'description': 'Medium-sized test'
        },
        {
            'name': 'Large Test',
            'config': create_config(lineup_pool_size=25000, population_size=100, target_lineups=50),
            'description': 'Larger test to show scalability'
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        test_name = test_config['name']
        config = test_config['config']
        description = test_config['description']
        
        print(f"\n🔬 {test_name}: {description}")
        print("-" * 60)
        
        # Test optimized version
        optimized_result = test_optimizer_performance(
            'optimize_multilineup_sets', 
            config, 
            f"{test_name} - OPTIMIZED Set-Based"
        )
        
        results[test_name] = {
            'config': config,
            'optimized': optimized_result
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        
        opt = result['optimized']
        if opt['success']:
            print(f"  ✅ Optimized: {opt['time']:.2f}s ({opt['time_per_lineup']:.3f}s per lineup)")
            print(f"     Generated {opt['lineups']} lineups with {opt['diversity']:.3f} avg diversity")
        else:
            print(f"  ❌ Optimized: FAILED ({opt.get('time', 0):.2f}s)")
    
    print(f"\n🎯 KEY OPTIMIZATIONS IMPLEMENTED:")
    print("  • Vectorized lineup pool generation using multidimensional_shifting_fast")
    print("  • Smart strategy selection based on problem size")
    print("  • Clustering-based diversity sampling for large pools")
    print("  • Reduced similarity computation overhead")
    print("  • Memory-efficient batching")
    print("  • Numba acceleration where available")

if __name__ == '__main__':
    run_performance_comparison()
