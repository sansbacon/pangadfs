# simple_test_sets.py
# Simple test of the new set-based multilineup optimization

import logging
from pathlib import Path
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager
from pangadfs.ga import GeneticAlgorithm


def simple_test():
    """Simple test with small parameters"""
    logging.basicConfig(level=logging.INFO)
    
    print("Simple Set-Based Multilineup Test")
    print("=" * 40)
    
    # Configuration with smaller parameters for faster testing
    ctx = {
        'ga_settings': {
            'crossover_method': 'tournament_within_sets',
            'csvpth': Path(__file__).parent / 'pangadfs' / 'app' / 'appdata' / 'pool.csv',
            'elite_divisor': 5,
            'elite_method': 'fittest',
            'mutation_rate': 0.1,
            'n_generations': 3,  # Very few generations
            'points_column': 'proj',
            'population_size': 10,  # Very small population
            'position_column': 'pos',
            'salary_column': 'salary',
            'select_method': 'tournament',
            'stop_criteria': 5,
            'verbose': True,
            'enable_profiling': False,
            
            # Set-based multilineup settings
            'target_lineups': 3,             # Just 3 lineups
            'diversity_weight': 0.3,         
            'diversity_method': 'jaccard',   
            'tournament_size': 2,            
        },

        'site_settings': {
            'flex_positions': ('RB', 'WR', 'TE'),
            'lineup_size': 9,
            'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
            'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 1},
            'salary_cap': 50000
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
            # Use the new set-based optimizer
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name='optimize_multilineup_sets',
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    # Set up GeneticAlgorithm object
    ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
    
    # Run the optimization
    print(f"Running optimization for {ctx['ga_settings']['target_lineups']} lineups...")
    results = ga.optimize()

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"Generated {len(results['lineups'])} lineups")
    print(f"Best individual lineup score: {results['best_score']:.2f}")
    print(f"Average diversity overlap: {results['diversity_metrics']['avg_overlap']:.3f}")
    
    # Show all lineups briefly
    print(f"\n=== ALL {len(results['lineups'])} LINEUPS ===")
    for i, (lineup, score) in enumerate(zip(results['lineups'], results['scores'])):
        print(f"Lineup {i+1} (Score: {score:.2f}): {len(lineup)} players")
    
    print("\n✅ Set-based optimization completed successfully!")
    return results


if __name__ == '__main__':
    results = simple_test()
