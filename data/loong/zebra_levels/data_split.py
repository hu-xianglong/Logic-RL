#!/usr/bin/env python3
"""
Script to split zebra puzzle data into 80/20 train/test sets.
Filters out puzzles where is_correct is false.
Processes 2x3 and 3x3 puzzle levels.
"""

import json
import random
import os
from pathlib import Path
from typing import Dict, List, Any

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def filter_correct_solutions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter results to only include puzzles where is_correct is true.

    Args:
        results: List of puzzle results

    Returns:
        Filtered list containing only correct solutions
    """
    filtered = []
    for result in results:
        # Check if verification exists and is_correct is true
        verification = result.get("verification", {})
        if verification.get("is_correct", False) == True:
            filtered.append(result)
    return filtered

def split_data(data: List[Any], train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    Split data into train and test sets.

    Args:
        data: List of data items to split
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data

def process_level(level_dir: Path, level_name: str) -> None:
    """
    Process a single difficulty level directory.

    Args:
        level_dir: Path to the level directory
        level_name: Name of the level (e.g., "2x3", "3x3")
    """
    aggregated_file = level_dir / "aggregated_results.json"

    if not aggregated_file.exists():
        print(f"  ⚠️  No aggregated_results.json found for {level_name}")
        return

    print(f"Processing {level_name}...")

    # Load data
    data = load_json_data(aggregated_file)
    results = data.get("results", [])

    if not results:
        print(f"  ⚠️  No results found in {level_name}")
        return

    print(f"  Found {len(results)} total puzzles")

    # Filter to only keep correct solutions
    filtered_results = filter_correct_solutions(results)
    print(f"  Filtered to {len(filtered_results)} correct solutions (removed {len(results) - len(filtered_results)} incorrect)")

    if not filtered_results:
        print(f"  ⚠️  No correct solutions found in {level_name}")
        return

    # Split the data
    train_data, test_data = split_data(filtered_results)

    print(f"  Split into {len(train_data)} training and {len(test_data)} test puzzles")

    # Save train and test sets
    train_file = level_dir / "train.json"
    test_file = level_dir / "test.json"

    save_json_data(train_data, train_file)
    save_json_data(test_data, test_file)

    print(f"  ✓ Saved train.json ({len(train_data)} puzzles)")
    print(f"  ✓ Saved test.json ({len(test_data)} puzzles)")

def main():
    """Main function to process all levels."""
    script_dir = Path(__file__).parent

    # Define levels to process: 2x3 and 3x3
    levels_to_process = [
        ("2x3", script_dir / "2x3"),
        ("3x3", script_dir / "3x3"),
    ]

    print("Starting data split process...")
    print(f"Train/Test ratio: 80/20")
    print(f"Random seed: 42")
    print(f"Filtering: Only including puzzles where is_correct=true\n")

    processed_count = 0

    for level_name, level_path in levels_to_process:
        if level_path.exists():
            process_level(level_path, level_name)
            processed_count += 1
            print()
        else:
            print(f"Skipping {level_name} (directory not found)")
            print()

    print(f"✅ Processing complete! Processed {processed_count} level(s)")

if __name__ == "__main__":
    main()