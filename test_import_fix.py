#!/usr/bin/env python3

"""
Simple test to verify the calculate_jaccard_diversity import fix
"""

import sys
from pathlib import Path

# Add pangadfs to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that we can import the function"""
    try:
        from pangadfs.misc import calculate_jaccard_diversity
        print("✅ Successfully imported calculate_jaccard_diversity")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_function():
    """Test that the function works correctly"""
    try:
        from pangadfs.misc import calculate_jaccard_diversity
        
        # Test with sample lineups
        lineup1 = [1, 2, 3, 4, 5]
        lineup2 = [1, 2, 6, 7, 8]
        
        diversity = calculate_jaccard_diversity(lineup1, lineup2)
        expected = 1.0 - (2/8)  # 2 intersection, 8 union -> 0.75 diversity
        
        print("Test lineups:")
        print(f"  Lineup 1: {lineup1}")
        print(f"  Lineup 2: {lineup2}")
        print(f"  Calculated diversity: {diversity:.3f}")
        print(f"  Expected diversity: {expected:.3f}")
        
        if abs(diversity - expected) < 0.001:
            print("✅ Function works correctly")
            return True
        else:
            print("❌ Function returned incorrect result")
            return False
            
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        return False

def main():
    print("Testing calculate_jaccard_diversity import fix...")
    print()
    
    # Test import
    import_success = test_import()
    
    if import_success:
        # Test function
        function_success = test_function()
        
        if function_success:
            print()
            print("🎉 All tests passed! The import issue is fixed.")
        else:
            print()
            print("⚠️  Import works but function has issues.")
    else:
        print()
        print("❌ Import still failing.")

if __name__ == '__main__':
    main()
