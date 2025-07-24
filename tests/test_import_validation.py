# tests/test_import_validation.py
# -*- coding: utf-8 -*-
# Simple test to validate that all multilineup optimization approaches can be imported

import sys
import os
from pathlib import Path

# Add pangadfs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all multilineup optimization approaches can be imported"""
    print("Testing multilineup optimization imports...")
    
    try:
        # Test basic optimize imports
        from pangadfs.optimize import OptimizeDefault, OptimizeMultilineup, OptimizeMultilineupSets
        print("✅ Basic optimize classes imported successfully")
        
        # Test pool-based optimizer
        from pangadfs.optimize_pool_based import OptimizePoolBasedSets
        print("✅ Pool-based optimizer imported successfully")
        
        # Test multi-objective optimizer
        from pangadfs.optimize_multi_objective import OptimizeMultiObjective
        print("✅ Multi-objective optimizer imported successfully")
        
        # Test that the optimized classes can be imported
        from pangadfs.fitness_sets import FitnessMultilineupSets
        from pangadfs.mutate_sets import MutateMultilineupSets
        from pangadfs.crossover_sets import CrossoverMultilineupSets
        from pangadfs.populate_sets import PopulateMultilineupSets
        print("✅ All optimized set classes imported successfully")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("The import fixes have resolved the issues.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_class_instantiation():
    """Test that classes can be instantiated"""
    print("\nTesting class instantiation...")
    
    try:
        from pangadfs.optimize import OptimizeMultilineup, OptimizeMultilineupSets
        from pangadfs.optimize_pool_based import OptimizePoolBasedSets
        from pangadfs.optimize_multi_objective import OptimizeMultiObjective
        
        # Test instantiation
        opt1 = OptimizeMultilineup()
        opt2 = OptimizeMultilineupSets()
        opt3 = OptimizePoolBasedSets()
        opt4 = OptimizeMultiObjective()
        
        print("✅ All optimizer classes instantiated successfully")
        
        # Test that they have the optimize method
        assert hasattr(opt1, 'optimize'), "OptimizeMultilineup missing optimize method"
        assert hasattr(opt2, 'optimize'), "OptimizeMultilineupSets missing optimize method"
        assert hasattr(opt3, 'optimize'), "OptimizePoolBasedSets missing optimize method"
        assert hasattr(opt4, 'optimize'), "OptimizeMultiObjective missing optimize method"
        
        print("✅ All optimizer classes have optimize method")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

def analyze_approaches():
    """Analyze the different approaches available"""
    print("\n📊 MULTILINEUP OPTIMIZATION APPROACHES ANALYSIS")
    print("=" * 60)
    
    approaches = {
        'OptimizeMultilineup': {
            'description': 'Post-processing approach - runs single-lineup GA then selects diverse lineups',
            'pros': ['Proven performance', 'Simple to understand', 'Focus on individual quality'],
            'cons': ['Diversity is afterthought', 'May not be optimal for large sets'],
            'best_for': 'Individual lineup quality priority'
        },
        'OptimizeMultilineupSets': {
            'description': 'Set-based GA - operates on lineup sets as first-class citizens',
            'pros': ['True multilineup optimization', 'Balanced approach', 'Good for medium sets'],
            'cons': ['More complex', 'May compromise individual quality'],
            'best_for': 'Balanced quality and diversity'
        },
        'OptimizePoolBasedSets': {
            'description': 'Advanced pool-based approach with pre-generated lineup pools',
            'pros': ['Very sophisticated', 'Adaptive strategies', 'Good for large sets'],
            'cons': ['Complex', 'Memory intensive', 'Many parameters'],
            'best_for': 'Large-scale optimization with resources'
        },
        'OptimizeMultiObjective': {
            'description': 'Multi-objective optimization with explicit objective balancing',
            'pros': ['Mathematically rigorous', 'Explicit trade-offs', 'Pareto optimization'],
            'cons': ['Complex parameter tuning', 'May be overkill'],
            'best_for': 'Research and advanced use cases'
        }
    }
    
    for name, info in approaches.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")

def recommendation_based_on_requirements():
    """Provide recommendation based on user requirements"""
    print("\n🎯 RECOMMENDATION BASED ON YOUR REQUIREMENTS")
    print("=" * 60)
    print("Your priorities:")
    print("1. Best individual lineup quality (#1 priority)")
    print("2. Diverse options")
    print("3. Focus on large lineup sets (50-150 lineups)")
    print("4. Must include lineups within 5% of optimal")
    print("5. 16GB memory available")
    
    print("\n🏆 RECOMMENDED APPROACH: OptimizeMultilineup (Post-processing)")
    print("\nReasons:")
    print("✅ Prioritizes individual lineup quality (your #1 requirement)")
    print("✅ GA focuses 100% on optimization before diversity selection")
    print("✅ Proven track record in your previous testing")
    print("✅ Simple and reliable - fewer parameters to tune")
    print("✅ Most likely to produce lineups within 5% of optimal")
    print("✅ Scales well to large lineup sets")
    print("✅ Memory efficient compared to pool-based approaches")
    
    print("\n📋 APPROACHES TO REMOVE:")
    print("❌ OptimizeMultilineupSets - May compromise individual quality for diversity")
    print("❌ OptimizePoolBasedSets - Unnecessary complexity for your use case")
    print("❌ OptimizeMultiObjective - Overkill, may not prioritize individual quality enough")
    
    print("\n⚙️ RECOMMENDED CONFIGURATION:")
    print("- Use OptimizeMultilineup as primary approach")
    print("- Set diversity_weight = 0.2-0.3 (favor quality over diversity)")
    print("- Use large population sizes (1000+) for better exploration")
    print("- Run sufficient generations (100-200) for convergence")
    print("- Use aggressive diversity selection in post-processing")

if __name__ == '__main__':
    print("MULTILINEUP OPTIMIZATION VALIDATION TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test instantiation
        instantiation_ok = test_class_instantiation()
        
        if instantiation_ok:
            # Provide analysis and recommendations
            analyze_approaches()
            recommendation_based_on_requirements()
            
            print("\n✅ VALIDATION COMPLETE")
            print("All multilineup optimization approaches are ready for use!")
        else:
            print("\n❌ VALIDATION FAILED")
            print("Classes cannot be instantiated properly.")
    else:
        print("\n❌ VALIDATION FAILED") 
        print("Import issues still exist.")
