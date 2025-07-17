#!/usr/bin/env python3
# test_multilineup.py
# Simple test script to verify multilineup optimization works

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pangadfs'))
import numpy as np
import pandas as pd
from pathlib import Path

from pangadfs.ga import GeneticAlgorithm
from stevedore.driver import DriverManager
from stevedore.named import NamedExtensionManager


def test_multilineup_plugin():
    """Test the multilineup optimization plugin"""
    print("Testing OptimizeMultilineup plugin...")
    
    # Create a simple test dataset
    test_data = {
        'player': [f'Player_{i}' for i in range(50)],
        'team': [f'Team_{i%10}' for i in range(50)],
        'pos': ['QB'] * 5 + ['RB'] * 15 + ['WR'] * 20 + ['TE'] * 5 + ['DST'] * 5,
        'salary': np.random.randint(3000, 12000, 50),
        'proj': np.random.uniform(5, 30, 50)
    }
    
    # Create test CSV
    test_df = pd.DataFrame(test_data)
    test_csv_path = Path('test_pool.csv')
    test_df.to_csv(test_csv_path, index=False)
    
    try:
        # Test configuration
        ctx = {
            'ga_settings': {
                'crossover_method': 'uniform',
                'csvpth': test_csv_path,
                'elite_divisor': 5,
                'elite_method': 'fittest',
                'mutation_rate': .05,
                'n_generations': 5,  # Short test
                'points_column': 'proj',
                'population_size': 1000,  # Small population for test
                'position_column': 'pos',
                'salary_column': 'salary',
                'select_method': 'tournament',
                'stop_criteria': 3,
                'verbose': False,  # Quiet for test
                'enable_profiling': False,
                
                # Multilineup settings
                'target_lineups': 10,  # Small number for test
                'diversity_weight': 0.3,
                'min_overlap_threshold': 0.3,
                'diversity_method': 'jaccard',
            },

            'site_settings': {
                'flex_positions': ('RB', 'WR', 'TE'),
                'lineup_size': 9,
                'posfilter': {'QB': 5, 'RB': 15, 'WR': 20, 'TE': 5, 'DST': 5, 'FLEX': 40},
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
                # Test the multilineup optimizer
                dmgrs[ns] = DriverManager(
                    namespace=pns, 
                    name='optimize_multilineup',
                    invoke_on_load=True)
            else:
                dmgrs[ns] = DriverManager(
                    namespace=pns, 
                    name=f'{ns}_default', 
                    invoke_on_load=True)
        
        # Create GA and run optimization
        ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
        results = ga.optimize()
        
        # Verify results
        print("✓ Plugin loaded and executed successfully")
        
        # Check result structure
        required_keys = ['population', 'fitness', 'best_lineup', 'best_score', 'lineups', 'scores', 'diversity_metrics']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        print("✓ Result structure is correct")
        
        # Check multilineup results
        assert len(results['lineups']) == 10, f"Expected 10 lineups, got {len(results['lineups'])}"
        assert len(results['scores']) == 10, f"Expected 10 scores, got {len(results['scores'])}"
        print("✓ Generated correct number of lineups")
        
        # Check diversity metrics
        diversity_keys = ['avg_overlap', 'min_overlap', 'diversity_matrix']
        for key in diversity_keys:
            assert key in results['diversity_metrics'], f"Missing diversity metric: {key}"
        print("✓ Diversity metrics are present")
        
        # Check backward compatibility
        assert isinstance(results['best_lineup'], pd.DataFrame), "best_lineup should be DataFrame"
        assert isinstance(results['best_score'], (int, float)), "best_score should be numeric"
        print("✓ Backward compatibility maintained")
        
        # Check that lineups are different
        lineup_sets = [set(lineup.index) for lineup in results['lineups']]
        unique_lineups = len({frozenset(s) for s in lineup_sets})
        print(f"✓ Generated {unique_lineups} unique lineups out of {len(results['lineups'])}")
        
        print("\nTest Results Summary:")
        print(f"- Generated {len(results['lineups'])} lineups")
        print(f"- Best score: {results['best_score']:.2f}")
        print(f"- Average overlap: {results['diversity_metrics']['avg_overlap']:.3f}")
        print(f"- Minimum overlap: {results['diversity_metrics']['min_overlap']:.3f}")
        
        print("\n✅ All tests passed! OptimizeMultilineup plugin is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        if test_csv_path.exists():
            test_csv_path.unlink()
    
    return True


def test_single_lineup_mode():
    """Test backward compatibility with single lineup mode"""
    print("\nTesting single lineup mode (backward compatibility)...")
    
    # Create a simple test dataset with more players
    test_data = {
        'player': [f'Player_{i}' for i in range(60)],
        'team': [f'Team_{i%8}' for i in range(60)],
        'pos': ['QB'] * 8 + ['RB'] * 16 + ['WR'] * 20 + ['TE'] * 8 + ['DST'] * 8,
        'salary': np.random.randint(3000, 12000, 60),
        'proj': np.random.uniform(5, 30, 60)
    }
    
    # Create test CSV
    test_df = pd.DataFrame(test_data)
    test_csv_path = Path('test_pool_single.csv')
    test_df.to_csv(test_csv_path, index=False)
    
    try:
        # Test configuration for single lineup
        ctx = {
            'ga_settings': {
                'crossover_method': 'uniform',
                'csvpth': test_csv_path,
                'elite_divisor': 5,
                'elite_method': 'fittest',
                'mutation_rate': .05,
                'n_generations': 3,
                'points_column': 'proj',
                'population_size': 500,
                'position_column': 'pos',
                'salary_column': 'salary',
                'select_method': 'tournament',
                'stop_criteria': 2,
                'verbose': False,
                'enable_profiling': False,
                
                # Single lineup mode
                'target_lineups': 1,
            },

            'site_settings': {
                'flex_positions': ('RB', 'WR', 'TE'),
                'lineup_size': 9,
                'posfilter': {'QB': 3, 'RB': 8, 'WR': 12, 'TE': 4, 'DST': 3, 'FLEX': 24},
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
                dmgrs[ns] = DriverManager(
                    namespace=pns, 
                    name='optimize_multilineup',  # Using multilineup in single mode
                    invoke_on_load=True)
            else:
                dmgrs[ns] = DriverManager(
                    namespace=pns, 
                    name=f'{ns}_default', 
                    invoke_on_load=True)
        
        # Create GA and run optimization
        ga = GeneticAlgorithm(ctx=ctx, driver_managers=dmgrs, extension_managers=emgrs)
        results = ga.optimize()
        
        # Verify single lineup mode works like original
        assert len(results['lineups']) == 1, f"Expected 1 lineup, got {len(results['lineups'])}"
        assert len(results['scores']) == 1, f"Expected 1 score, got {len(results['scores'])}"
        
        # Check that best_lineup and lineups[0] are the same
        best_lineup_players = set(results['best_lineup'].index)
        first_lineup_players = set(results['lineups'][0].index)
        assert best_lineup_players == first_lineup_players, "best_lineup and lineups[0] should be identical"
        
        print("✓ Single lineup mode works correctly")
        print("✓ Backward compatibility maintained")
        print(f"✓ Generated lineup with score: {results['best_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Single lineup test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        if test_csv_path.exists():
            test_csv_path.unlink()
    
    return True


if __name__ == '__main__':
    print("Running OptimizeMultilineup plugin tests...\n")
    
    success1 = test_multilineup_plugin()
    success2 = test_single_lineup_mode()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The OptimizeMultilineup plugin is ready to use.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
