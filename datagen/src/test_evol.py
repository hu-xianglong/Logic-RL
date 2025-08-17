#!/usr/bin/env python3
"""
Test script for zebra puzzle evolution - runs a minimal example.
"""

import json
import sys
import os
from pathlib import Path

from evol_zebra_puzzles import ZebraEvolInstructTemplates, ZebraScorer, load_seed_puzzles

def test_templates():
    """Test that templates are working correctly."""
    templates = ZebraEvolInstructTemplates()
    
    print("Evolution Methods:")
    for key, value in templates.EVOL_METHODS.items():
        print(f"  {key}: {value[:50]}...")
    
    print("\nStrategies:")
    for key, value in templates.STRATEGY.items():
        print(f"  {key}: {list(value.keys())}")
    
    return True

def test_scorer():
    """Test the scorer on a sample puzzle."""
    scorer = ZebraScorer()
    
    sample_puzzle = """
    You are given an Einstein logic puzzle. In this puzzle, there are 2 people.
    
    Setup:
    The puzzle involves the following categories:
    1. Nationality: British, French
    2. Pet: Dog, Cat
    
    Clues:
    1. The British person is immediately to the left of the person with a cat.
    2. The person with a dog is at position 1.
    
    Expected Output Format:
    Your solution should be a dictionary where each key is a category name.
    """
    
    scores = scorer.score(sample_puzzle)
    print("\nScorer test results:")
    for key, value in scores.items():
        print(f"  {key}: {value:.2f}")
    
    return scores["overall"] > 0.5

def test_data_loading():
    """Test loading seed puzzles."""
    data_path = "/Users/xianglonghu/CascadeProjects/Logic-RL/data/loong/zebra_raw_levels/n2_m2/train.parquet"
    results_path = "../loong_logic/results/zebra/aggregated_results.json"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return False
    
    puzzles = load_seed_puzzles(data_path, results_path, num_samples=2)
    print(f"\nLoaded {len(puzzles)} seed puzzles with solutions")
    
    if puzzles:
        print("\nFirst puzzle preview:")
        print(f"Question: {puzzles[0]['question'][:200]}...")
        print(f"Has code: {len(puzzles[0]['code']) > 0}")
        print(f"Has answer: {len(puzzles[0]['final_answer']) > 0}")
    
    return len(puzzles) > 0

def main():
    """Run all tests."""
    print("Testing Zebra Evolution Components")
    print("=" * 50)
    
    tests = [
        ("Templates", test_templates),
        ("Scorer", test_scorer),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name} Test:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((name, result))
            print(f"✓ {name} test {'passed' if result else 'failed'}")
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready to run evolution.")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running evolution.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)