# test_sets_multilineup.py
# Test the new set-based multilineup optimization

import logging
from pathlib import Path
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager
from pangadfs.ga import GeneticAlgorithm


def test_sets_multilineup():
    """Test the new set-based multilineup optimization"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Set-Based Multilineup Optimization")
    print("=" * 50)
    
    # Configuration for set-based optimization
    ctx = {
        'ga_settings': {
            'crossover_method': 'tournament_within_sets',
            'csvpth': Path(__file__).parent / 'pangadfs' / 'app' / 'appdata' / 'pool.csv',
            'elite_divisor': 5,
            'elite_method': 'fittest',
            'mutation_rate': 0.1,
            'n_generations': 10,  # Fewer generations for testing
            'points_column': 'proj',
            'population_size': 1000,  # Smaller population for testing
            'position_column': 'pos',
            'salary_column': 'salary',
            'select_method': 'tournament',
            'stop_criteria': 5,
            'verbose': True,
            'enable_profiling': True,
            
            # Set-based multilineup settings
            'target_lineups': 10,            # Generate 10 lineups
            'diversity_weight': 0.3,         # Weight for diversity penalty
            'diversity_method': 'jaccard',   # Use Jaccard similarity
            'tournament_size': 3,            # Tournament size for crossover
        },

        'site_settings': {
            'flex_positions': ('RB', 'WR', 'TE'),
            'lineup_size': 9,
            'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
            'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 1},
            'salary_cap': 50000
        }
    }

    # Set up driver managers to use the new set-based optimizer
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
                name='optimize_multilineup_sets',  # New set-based optimizer
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    # Set up GeneticAlgorithm object
    ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
    
    # Run the set-based optimization
    print(f"Running set-based optimization for {ctx['ga_settings']['target_lineups']} lineups...")
    results = ga.optimize()

    # Display results
    print("\n" + "=" * 60)
    print("SET-BASED MULTILINEUP OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"Generated {len(results['lineups'])} lineups")
    print(f"Best individual lineup score: {results['best_score']:.2f}")
    print(f"Average diversity overlap: {results['diversity_metrics']['avg_overlap']:.3f}")
    print(f"Minimum diversity overlap: {results['diversity_metrics']['min_overlap']:.3f}")
    
    # Show all lineups
    print(f"\n=== ALL {len(results['lineups'])} LINEUPS ===")
    for i, (lineup, score) in enumerate(zip(results['lineups'], results['scores'])):
        print(f"\nLineup {i+1} (Score: {score:.2f}):")
        print(lineup[['player', 'team', 'pos', 'salary', 'proj']].to_string(index=False))
    
    # Display profiling results if enabled
    if ga.profiler.enabled:
        print("\n" + "=" * 60)
        ga.profiler.print_profiling_results()
    
    return results


def compare_optimizers():
    """Compare the old post-processing approach vs new set-based approach"""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print("COMPARING OPTIMIZATION APPROACHES")
    print("=" * 70)
    
    # Common configuration
    base_ctx = {
        'ga_settings': {
            'crossover_method': 'uniform',
            'csvpth': Path(__file__).parent / 'pangadfs' / 'app' / 'appdata' / 'pool.csv',
            'elite_divisor': 5,
            'elite_method': 'fittest',
            'mutation_rate': 0.05,
            'n_generations': 10,
            'points_column': 'proj',
            'population_size': 5000,
            'position_column': 'pos',
            'salary_column': 'salary',
            'select_method': 'tournament',
            'stop_criteria': 5,
            'verbose': False,  # Reduce output for comparison
            'enable_profiling': True,
            
            'target_lineups': 5,
            'diversity_weight': 0.3,
            'diversity_method': 'jaccard',
        },

        'site_settings': {
            'flex_positions': ('RB', 'WR', 'TE'),
            'lineup_size': 9,
            'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
            'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 1},
            'salary_cap': 50000
        }
    }
    
    # Test 1: Old post-processing approach
    print("\n1. Testing OLD post-processing approach...")
    ctx_old = base_ctx.copy()
    
    dmgrs_old = {}
    emgrs_old = {}
    for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
        pns = f'pangadfs.{ns}'
        if ns == 'validate':
            emgrs_old['validate'] = NamedExtensionManager(
                namespace=pns, names=['validate_salary', 'validate_duplicates'], 
                invoke_on_load=True, name_order=True)
        elif ns == 'optimize':
            dmgrs_old[ns] = DriverManager(
                namespace=pns, name='optimize_multilineup', invoke_on_load=True)
        else:
            dmgrs_old[ns] = DriverManager(
                namespace=pns, name=f'{ns}_default', invoke_on_load=True)
    
    ga_old = GeneticAlgorithm(ctx=ctx_old, driver_managers=dmgrs_old, extension_managers=emgrs_old)
    results_old = ga_old.optimize()
    
    # Test 2: New set-based approach
    print("\n2. Testing NEW set-based approach...")
    ctx_new = base_ctx.copy()
    ctx_new['ga_settings']['tournament_size'] = 3
    
    dmgrs_new = {}
    emgrs_new = {}
    for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
        pns = f'pangadfs.{ns}'
        if ns == 'validate':
            emgrs_new['validate'] = NamedExtensionManager(
                namespace=pns, names=['validate_salary', 'validate_duplicates'], 
                invoke_on_load=True, name_order=True)
        elif ns == 'optimize':
            dmgrs_new[ns] = DriverManager(
                namespace=pns, name='optimize_multilineup_sets', invoke_on_load=True)
        else:
            dmgrs_new[ns] = DriverManager(
                namespace=pns, name=f'{ns}_default', invoke_on_load=True)
    
    ga_new = GeneticAlgorithm(ctx=ctx_new, driver_managers=dmgrs_new, extension_managers=emgrs_new)
    results_new = ga_new.optimize()
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"{'Metric':<30} {'Old Approach':<15} {'New Approach':<15} {'Improvement'}")
    print("-" * 70)
    
    # Number of lineups generated
    old_count = len(results_old['lineups'])
    new_count = len(results_new['lineups'])
    print(f"{'Lineups Generated':<30} {old_count:<15} {new_count:<15} {'+' if new_count > old_count else ''}{new_count - old_count}")
    
    # Total points
    old_total = sum(results_old['scores'])
    new_total = sum(results_new['scores'])
    print(f"{'Total Points':<30} {old_total:<15.1f} {new_total:<15.1f} {'+' if new_total > old_total else ''}{new_total - old_total:.1f}")
    
    # Average diversity
    old_div = results_old['diversity_metrics']['avg_overlap']
    new_div = results_new['diversity_metrics']['avg_overlap']
    print(f"{'Avg Overlap (lower=better)':<30} {old_div:<15.3f} {new_div:<15.3f} {'-' if new_div < old_div else '+'}{abs(new_div - old_div):.3f}")
    
    # Best individual score
    old_best = results_old['best_score']
    new_best = results_new['best_score']
    print(f"{'Best Individual Score':<30} {old_best:<15.1f} {new_best:<15.1f} {'+' if new_best > old_best else ''}{new_best - old_best:.1f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if new_count >= old_count and new_total >= old_total and new_div <= old_div:
        print("✅ NEW SET-BASED APPROACH IS SUPERIOR!")
        print("   - Generated same or more lineups")
        print("   - Achieved same or higher total points")
        print("   - Maintained same or better diversity")
    else:
        print("⚠️  Results are mixed - further tuning may be needed")
    
    return results_old, results_new


if __name__ == '__main__':
    # Test the new set-based approach
    results = test_sets_multilineup()
    
    # Compare old vs new approaches
    old_results, new_results = compare_optimizers()
