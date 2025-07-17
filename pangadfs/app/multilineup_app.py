# pangadfs/app/multilineup_app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging

from pangadfs.ga import GeneticAlgorithm
from pathlib import Path
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager


def run_multilineup():
    """Example application using pangadfs multilineup optimization"""
    logging.basicConfig(level=logging.INFO)

    ctx = {
        'ga_settings': {
            'crossover_method': 'uniform',
            'csvpth': Path(__file__).parent / 'appdata' / 'pool.csv',
            'elite_divisor': 5,
            'elite_method': 'fittest',
            'mutation_rate': .05,
            'n_generations': 20,
            'points_column': 'proj',
            'population_size': 30000,
            'position_column': 'pos',
            'salary_column': 'salary',
            'select_method': 'tournament',
            'stop_criteria': 10,
            'verbose': True,
            'enable_profiling': True,
            
            # Multilineup-specific settings
            'target_lineups': 150,           # Generate 150 lineups
            'diversity_weight': 0.3,         # Weight for diversity penalty
            'min_overlap_threshold': 0.3,    # Minimum allowed overlap between lineups
            'diversity_method': 'jaccard',   # Diversity calculation method
        },

        'site_settings': {
            'flex_positions': ('RB', 'WR', 'TE'),
            'lineup_size': 9,
            'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
            'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
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
            # Use the multilineup optimizer instead of default
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name='optimize_multilineup',  # Changed from 'optimize_default'
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    # Set up GeneticAlgorithm object
    ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
    
    # Run optimizer
    results = ga.optimize()

    # Show results
    print("\n=== MULTILINEUP OPTIMIZATION RESULTS ===")
    print(f"Generated {len(results['lineups'])} lineups")
    print(f"Best lineup score: {results['best_score']}")
    print(f"Average diversity overlap: {results['diversity_metrics']['avg_overlap']:.3f}")
    print(f"Minimum diversity overlap: {results['diversity_metrics']['min_overlap']:.3f}")
    
    # Show top 5 lineups
    print("\n=== TOP 5 LINEUPS ===")
    for i, (lineup, score) in enumerate(zip(results['lineups'][:5], results['scores'][:5])):
        print(f"\nLineup {i+1} (Score: {score:.2f}):")
        print(lineup[['player', 'team', 'pos', 'salary', 'proj']].to_string(index=False))
    
    # Display profiling results if enabled
    if ga.profiler.enabled:
        ga.profiler.print_profiling_results()


def run_single_lineup():
    """Example showing backward compatibility - single lineup mode"""
    logging.basicConfig(level=logging.INFO)

    ctx = {
        'ga_settings': {
            'crossover_method': 'uniform',
            'csvpth': Path(__file__).parent / 'appdata' / 'pool.csv',
            'elite_divisor': 5,
            'elite_method': 'fittest',
            'mutation_rate': .05,
            'n_generations': 20,
            'points_column': 'proj',
            'population_size': 30000,
            'position_column': 'pos',
            'salary_column': 'salary',
            'select_method': 'tournament',
            'stop_criteria': 10,
            'verbose': True,
            'enable_profiling': True,
            
            # Single lineup mode (backward compatibility)
            'target_lineups': 1,  # Only generate 1 lineup
        },

        'site_settings': {
            'flex_positions': ('RB', 'WR', 'TE'),
            'lineup_size': 9,
            'posfilter': {'QB': 14, 'RB': 8, 'WR': 8, 'TE': 5, 'DST': 4, 'FLEX': 8},
            'posmap': {'DST': 1, 'QB': 1, 'TE': 1, 'RB': 2, 'WR': 3, 'FLEX': 7},
            'salary_cap': 50000
        }
    }

    # Set up driver managers (using multilineup optimizer in single mode)
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
                name='optimize_multilineup',  # Using multilineup in single mode
                invoke_on_load=True)
        else:
            dmgrs[ns] = DriverManager(
                namespace=pns, 
                name=f'{ns}_default', 
                invoke_on_load=True)
    
    # Set up GeneticAlgorithm object
    ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
    
    # Run optimizer
    results = ga.optimize()

    # Show that it works exactly like the original optimizer
    print("\n=== SINGLE LINEUP MODE (BACKWARD COMPATIBILITY) ===")
    print(f"Generated {len(results['lineups'])} lineup")
    print(f"Best lineup score: {results['best_score']}")
    
    # These should work exactly as before
    print("\nBest lineup (backward compatibility):")
    print(results['best_lineup'][['player', 'team', 'pos', 'salary', 'proj']].to_string(index=False))
    
    # New multilineup fields also available
    print("\nFirst lineup from lineups array:")
    print(results['lineups'][0][['player', 'team', 'pos', 'salary', 'proj']].to_string(index=False))


if __name__ == '__main__':
    print("Running multilineup optimization example...")
    run_multilineup()
    
    print("\n" + "="*60)
    print("Running single lineup mode example...")
    run_single_lineup()
