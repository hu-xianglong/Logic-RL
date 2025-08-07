#!/usr/bin/env python3
"""
Data preparation script to convert loong logic domain data to training format.
Creates 90/10 train/test split and converts JSON to parquet format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random
import os


def load_json_data(json_path):
    """Load data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_training_format(data, domain_name="logic"):
    """Convert JSON data to training format matching existing parquet structure."""
    converted_data = []
    
    for idx, item in enumerate(data):
        # Extract components from the original format
        question = item.get('question', '')
        final_answer = item.get('final_answer', '')
        rationale = item.get('rationale', '')
        metadata = item.get('metadata', {})
        
        # Create prompt in the expected format
        prompt_content = f"<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state your solution within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"
        
        # Create the training row
        row = {
            'quiz': question,
            'names': np.array(['placeholder']),  # Placeholder for compatibility
            'knight_knave': {'placeholder': 'placeholder'},  # Placeholder for compatibility
            'solution': np.array([True]),  # Placeholder for compatibility
            'solution_text': final_answer,
            'solution_text_format': final_answer,
            'cot_head': "Let's think step by step to solve this logic problem.",
            'cot_repeat_steps': np.array([rationale]) if rationale else np.array(['reasoning steps']),
            'cot_foot': 'This leads to the solution.',
            'statements': 'logic_problem',  # Placeholder for compatibility
            'index': idx,
            'data_source': f'loong_{domain_name}',
            'prompt': np.array([{'content': prompt_content, 'role': 'user'}]),
            'ability': domain_name,
            'reward_model': {
                'ground_truth': {
                    'solution_text_format': final_answer,
                    'statements': 'logic_problem'
                },
                'style': 'rule'
            },
            'extra_info': {
                'index': idx,
                'split': 'pending',  # Will be set during split
                'metadata': metadata
            }
        }
        converted_data.append(row)
    
    return converted_data


def create_train_test_split(data, test_size=0.1, random_state=42):
    """Create 90/10 train/test split."""
    random.seed(random_state)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * (1 - test_size))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    
    # Update split information
    for item in train_data:
        item['extra_info']['split'] = 'train'
    
    for item in test_data:
        item['extra_info']['split'] = 'test'
    
    return train_data, test_data


def save_as_parquet(data, output_path):
    """Save data as parquet file."""
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from loong domain')
    parser.add_argument('--input_path', type=str, 
                       default='loong/data/logic/seed_dataset.json',
                       help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='data/loong/logic',
                       help='Output directory for parquet files')
    parser.add_argument('--domain', type=str, default='logic',
                       help='Domain name for this dataset')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Test set size (default: 0.1 for 90/10 split)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducible splits')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_path}")
    json_data = load_json_data(args.input_path)
    print(f"Loaded {len(json_data)} samples")
    
    # Convert to training format
    print("Converting to training format...")
    converted_data = convert_to_training_format(json_data, args.domain)
    
    # Create train/test split
    print(f"Creating {1-args.test_size:.0%}/{args.test_size:.0%} train/test split...")
    train_data, test_data = create_train_test_split(
        converted_data, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Save as parquet files
    train_path = os.path.join(args.output_dir, 'train.parquet')
    test_path = os.path.join(args.output_dir, 'test.parquet')
    
    save_as_parquet(train_data, train_path)
    save_as_parquet(test_data, test_path)
    
    print(f"\nData preparation completed!")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")


if __name__ == "__main__":
    main()