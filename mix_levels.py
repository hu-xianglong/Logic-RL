#!/usr/bin/env python3
"""
Script to mix all levels (3ppl, 4ppl, 5ppl, 6ppl, 7ppl) together
for both train and test datasets.

This creates combined datasets with all difficulty levels mixed together.
"""

import pandas as pd
import os
from pathlib import Path

def mix_all_levels():
    """Mix all levels (3ppl through 7ppl) for both train and test data."""
    
    # Define the levels and base directory
    levels = ['3ppl', '4ppl', '5ppl', '6ppl', '7ppl']
    base_dir = Path('data/kk/instruct')
    output_dir = base_dir / 'mixed'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Mix train data
    print("Mixing training data...")
    train_dfs = []
    for level in levels:
        train_file = base_dir / level / 'train.parquet'
        if train_file.exists():
            df = pd.read_parquet(train_file)
            # Add level information to help track the source
            df['level'] = level
            train_dfs.append(df)
            print(f"  Loaded {level}: {len(df)} samples")
        else:
            print(f"  Warning: {train_file} not found")
    
    # Combine all training data
    combined_train = pd.concat(train_dfs, ignore_index=True)
    print(f"Combined training data: {len(combined_train)} samples")
    
    # Shuffle the data to mix levels
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined training data
    train_output = output_dir / 'train.parquet'
    combined_train.to_parquet(train_output, index=False)
    print(f"Saved mixed training data to: {train_output}")
    
    # Mix test data
    print("\nMixing test data...")
    test_dfs = []
    for level in levels:
        test_file = base_dir / level / 'test.parquet'
        if test_file.exists():
            df = pd.read_parquet(test_file)
            # Add level information to help track the source
            df['level'] = level
            test_dfs.append(df)
            print(f"  Loaded {level}: {len(df)} samples")
        else:
            print(f"  Warning: {test_file} not found")
    
    # Combine all test data
    combined_test = pd.concat(test_dfs, ignore_index=True)
    print(f"Combined test data: {len(combined_test)} samples")
    
    # Shuffle the data to mix levels
    combined_test = combined_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined test data
    test_output = output_dir / 'test.parquet'
    combined_test.to_parquet(test_output, index=False)
    print(f"Saved mixed test data to: {test_output}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Mixed training data: {len(combined_train)} samples")
    print(f"Mixed test data: {len(combined_test)} samples")
    print(f"Total samples: {len(combined_train) + len(combined_test)}")
    
    print("\nLevel distribution in training data:")
    print(combined_train['level'].value_counts().sort_index())
    
    print("\nLevel distribution in test data:")
    print(combined_test['level'].value_counts().sort_index())
    
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")

if __name__ == "__main__":
    mix_all_levels()