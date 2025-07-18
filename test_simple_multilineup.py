import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pangadfs.app.multilineup_app import MultilineupApp
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_multilineup_simple():
    """Test the multilineup algorithm using the existing app structure"""
    
    # Create multilineup app
    app = MultilineupApp()
    
    # Configure settings
    settings = {
        'csvpth': 'pangadfs/app/appdata/pool.csv',
        'population_size': 1000,
        'n_generations': 50,
        'stop_criteria': 10,
        'target_lineups': 15,  # Request 15 lineups
        'diversity_weight': 0.6,  # High diversity
        'min_overlap_threshold': 0.6,  # High diversity requirement
        'diversity_method': 'jaccard',
        'verbose': True
    }
    
    print("Starting multilineup optimization test...")
    print(f"Requesting {settings['target_lineups']} lineups")
    print(f"Diversity weight: {settings['diversity_weight']}")
    print(f"Min overlap threshold: {settings['min_overlap_threshold']}")
    
    try:
        # Run optimization
        results = app.run(settings)
        
        # Check results
        lineups = results.get('lineups', [])
        scores = results.get('scores', [])
        diversity_metrics = results.get('diversity_metrics', {})
        
        print(f"\n=== RESULTS ===")
        print(f"Requested lineups: {settings['target_lineups']}")
        print(f"Generated lineups: {len(lineups)}")
        print(f"Success rate: {len(lineups) / settings['target_lineups'] * 100:.1f}%")
        
        if lineups:
            print(f"Best score: {max(scores):.2f}")
            print(f"Average score: {sum(scores) / len(scores):.2f}")
            print(f"Average overlap: {diversity_metrics.get('avg_overlap', 0):.3f}")
            
            # Check if lineups contain TE
            print(f"\n=== LINEUP ANALYSIS ===")
            for i, lineup in enumerate(lineups[:3]):  # Check first 3 lineups
                print(f"\nLineup {i+1} (Score: {scores[i]:.2f}):")
                positions = lineup['pos'].value_counts()
                print(f"  Positions: {dict(positions)}")
                
                # Check for TE specifically
                te_players = lineup[lineup['pos'] == 'TE']
                if len(te_players) > 0:
                    print(f"  TE players: {te_players['player'].tolist()}")
                else:
                    print(f"  TE players: NONE FOUND!")
                    
                # Show all players in lineup
                print(f"  All players: {lineup['player'].tolist()}")
        
        return results
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_multilineup_simple()
