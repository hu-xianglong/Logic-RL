#!/usr/bin/env python3
"""
Simple test that doesn't require full CAMEL import.
"""

import json
import os
import pandas as pd

def test_data_access():
    """Test that we can access the required data files."""
    data_path = "/Users/xianglonghu/CascadeProjects/Logic-RL/data/loong/zebra_raw_levels/n2_m2/train.parquet"
    results_path = "../loong_logic/results/zebra/aggregated_results.json"
    
    print("Testing data file access...")
    
    # Test parquet file
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded parquet with {len(df)} rows")
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
        return False
    
    # Test results file
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return False
    
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        successful_count = 0
        for result in results_data.get('results', []):
            if result.get('success', False) and result.get('verification', {}).get('is_correct', False):
                successful_count += 1
        
        print(f"✅ Found {successful_count} successful puzzles in results")
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False
    
    return True

def test_constraint_library():
    """Test that python-constraint works."""
    try:
        from constraint import Problem, AllDifferentConstraint
        
        # Simple test
        problem = Problem()
        problem.addVariable("a", [1, 2])
        problem.addVariable("b", [1, 2])
        problem.addConstraint(AllDifferentConstraint(), ["a", "b"])
        
        solutions = problem.getSolutions()
        
        if len(solutions) == 2:
            print("✅ python-constraint working correctly")
            return True
        else:
            print(f"❌ Unexpected number of solutions: {len(solutions)}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import constraint library: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing constraint library: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple Zebra Evolution Tests")
    print("=" * 40)
    
    tests = [
        ("Data Access", test_data_access),
        ("Constraint Library", test_constraint_library),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name} Test:")
        print("-" * 20)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 Basic tests passed! Ready to install remaining dependencies.")
    else:
        print("\n⚠️ Some tests failed. Please fix issues first.")
    
    return all_passed

if __name__ == "__main__":
    main()