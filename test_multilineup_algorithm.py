import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pangadfs.ga import GeneticAlgorithm
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_multilineup():
    """Test the multilineup algorithm directly"""
    
    # Create GA instance with multilineup settings
    ga = GeneticAlgorithm()
    
    # Configure for multilineup
    ga.ctx = {
        'ga_settings': {
            'csvpth': 'pangadfs/app/appdata/pool.csv',
            'population_size': 1000,
            'n_generations': 50,
            'stop_criteria': 10,
            'points_column': 'proj',
            'position_column': 'pos',
            'salary_column': 'salary',
            'target_lineups': 15,  # Request 15 lineups
            'diversity_weight': 0.6,  # High diversity
            'min_overlap_threshold': 0.6,  # High diversity requirement
            'diversity_method': 'jaccard',
            'verbose': True
        },
        'site_settings': {
            'salary_cap': 50000,
            'posfilter': {
                'QB': 1,
                'RB': 2,
                'WR': 3,
                'TE': 1,
                'FLEX': 1,
                'DST': 1
            },
            'posmap': {
                'QB': ['QB'],
                'RB': ['RB'],
                'WR': ['WR'],
                'TE': ['TE'],
                'FLEX': ['RB', 'WR', 'TE'],
                'DST': ['DST']
            },
            'flex_positions': ['RB', 'WR', 'TE']
        }
    }
    
    # Run optimization
    print("Starting multilineup optimization test...")
    print(f"Requesting {ga.ctx['ga_settings']['target_lineups']} lineups")
    print(f"Diversity weight: {ga.ctx['ga_settings']['diversity_weight']}")
    print(f"Min overlap threshold: {ga.ctx['ga_settings']['min_overlap_threshold']}")
    
    results = ga.optimize()
    
    # Check results
    lineups = results.get('lineups', [])
    scores = results.get('scores', [])
    diversity_metrics = results.get('diversity_metrics', {})
    
    print("\n=== RESULTS ===")
    print(f"Requested lineups: {ga.ctx['ga_settings']['target_lineups']}")
    print(f"Generated lineups: {len(lineups)}")
    print(f"Success rate: {len(lineups) / ga.ctx['ga_settings']['target_lineups'] * 100:.1f}%")
    
    if lineups:
        print(f"Best score: {max(scores):.2f}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")
        print(f"Average overlap: {diversity_metrics.get('avg_overlap', 0):.3f}")
        
        # Check if lineups contain TE
        print("\n=== LINEUP ANALYSIS ===")
        for i, lineup in enumerate(lineups[:3]):  # Check first 3 lineups
            print(f"\nLineup {i+1} (Score: {scores[i]:.2f}):")
            positions = lineup['pos'].value_counts()
            print(f"  Positions: {dict(positions)}")
            
            # Check for TE specifically
            te_players = lineup[lineup['pos'] == 'TE']
            if len(te_players) > 0:
                print(f"  TE players: {te_players['player'].tolist()}")
            else:
                print("  TE players: NONE FOUND!")
    
    return results

if __name__ == "__main__":
    test_multilineup()
