#!/usr/bin/env python3
"""Preprocess Evolved Zebra Puzzle dataset for Einstein logic puzzles."""

import os
import json
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path


def make_prefix(question, template_type):
    """Create the prompt prefix based on template type."""
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n\nUser: {question}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def load_evolved_puzzles(evol_file_path):
    """Load evolved puzzles from the generated JSON file."""
    
    if not os.path.exists(evol_file_path):
        raise FileNotFoundError(f"File not found: {evol_file_path}")
    
    with open(evol_file_path, 'r', encoding='utf-8') as f:
        puzzles = json.load(f)
    
    print(f"Loaded {len(puzzles)} evolved puzzles")
    
    # Analyze evolution methods
    method_counts = {}
    for puzzle in puzzles:
        method = puzzle.get('evolution_method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("\nEvolution method distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} puzzles")
    
    return puzzles


def process_evolved_puzzles(puzzles, args):
    """Process evolved puzzles and create train/test datasets."""
    
    # Prepare puzzle data in standardized format
    processed_puzzles = []
    
    for idx, puzzle in enumerate(puzzles):
        # Convert solution format to JSON string if it's a dict
        solution = puzzle.get('solution', {})
        if isinstance(solution, dict):
            solution_str = json.dumps(solution)
        else:
            solution_str = str(solution)
        
        puzzle_data = {
            'id': puzzle.get('id', f'evolved_{idx}'),
            'question': puzzle.get('puzzle_text', ''),
            'final_answer': solution_str,
            'evolution_method': puzzle.get('evolution_method', 'unknown'),
            'categories': puzzle.get('categories', []),
            'items': puzzle.get('items', {}),
            'code': puzzle.get('code', ''),
            'execution_success': puzzle.get('execution_success', False),
            'expected_answer': puzzle.get('expected_answer', ''),
            'verification_status': puzzle.get('verification_status', ''),
            'seed_index': puzzle.get('seed_index', -1),
            'generation_time': puzzle.get('evolution_intermediate', {}).get('generation_time_seconds', 0),
            'verification_time': puzzle.get('verification_intermediate', {}).get('verification_time_seconds', 0)
        }
        processed_puzzles.append(puzzle_data)
    
    # Calculate split
    total_puzzles = len(processed_puzzles)
    train_size = min(args.train_size, int(total_puzzles * 0.8))
    test_size = min(args.test_size, total_puzzles - train_size)
    
    print(f"\nDataset split: {train_size} train, {test_size} test samples")
    
    # Create dataset
    raw_dataset = Dataset.from_list(processed_puzzles)
    
    # Split into train and test
    train_dataset = raw_dataset.select(range(train_size))
    test_dataset = raw_dataset.select(range(train_size, train_size + test_size))
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['question'], template_type=args.template_type)
            
            ground_truth = example['final_answer']
            
            # Create the standardized data format
            data = {
                "data_source": f"evol_zebra_n2_m2",
                "prompt": [{
                    "role": "user", 
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                }
            }
            return data
        return process_fn
    
    print(f"Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print(f"Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    return train_dataset, test_dataset, processed_puzzles


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_human_readable_json(train_parquet_path, test_parquet_path, output_path):
    """Save a human-readable version of the puzzles from parquet files for examination."""
    
    # Load samples from parquet files
    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)
    
    human_readable = {
        "description": "Sample puzzles from the processed parquet files",
        "train_samples": [],
        "test_samples": []
    }
    
    # Process first 5 train samples
    for idx in range(min(5, len(train_df))):
        row = train_df.iloc[idx].to_dict()
        sample = {
            "sample_number": idx + 1,
            "data_source": convert_to_serializable(row.get('data_source', 'unknown')),
            "prompt": convert_to_serializable(row.get('prompt', [])),
            "ability": convert_to_serializable(row.get('ability', 'unknown')),
            "reward_model": convert_to_serializable(row.get('reward_model', {}))
        }
        
        # Extract and format the question text for readability
        if sample['prompt'] and len(sample['prompt']) > 0:
            content = sample['prompt'][0].get('content', '')
            # Find the actual puzzle text after the template prefix
            if '<|im_start|>user' in content:
                puzzle_part = content.split('<|im_start|>user\n')[-1]
                puzzle_part = puzzle_part.split('\n<|im_end|>')[0]
                sample['puzzle_text_extracted'] = puzzle_part
        
        human_readable["train_samples"].append(sample)
    
    # Process first 5 test samples
    for idx in range(min(5, len(test_df))):
        row = test_df.iloc[idx].to_dict()
        sample = {
            "sample_number": idx + 1,
            "data_source": convert_to_serializable(row.get('data_source', 'unknown')),
            "prompt": convert_to_serializable(row.get('prompt', [])),
            "ability": convert_to_serializable(row.get('ability', 'unknown')),
            "reward_model": convert_to_serializable(row.get('reward_model', {}))
        }
        
        # Extract and format the question text for readability
        if sample['prompt'] and len(sample['prompt']) > 0:
            content = sample['prompt'][0].get('content', '')
            # Find the actual puzzle text after the template prefix
            if '<|im_start|>user' in content:
                puzzle_part = content.split('<|im_start|>user\n')[-1]
                puzzle_part = puzzle_part.split('\n<|im_end|>')[0]
                sample['puzzle_text_extracted'] = puzzle_part
        
        human_readable["test_samples"].append(sample)
    
    # Add summary statistics
    human_readable["summary"] = {
        "total_train_samples": len(train_df),
        "total_test_samples": len(test_df),
        "samples_shown": {
            "train": len(human_readable["train_samples"]),
            "test": len(human_readable["test_samples"])
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(human_readable, f, indent=2, ensure_ascii=False)
    
    print(f"Saved human-readable sample from parquet to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./results/evol_zebra_50_complete.json',
                        help='Path to evolved puzzles JSON file')
    parser.add_argument('--output_dir', default='../data/loong/evol_zebra_levels/n2_m2',
                        help='Output directory for processed datasets')
    parser.add_argument('--train_size', type=int, default=40, 
                        help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=10, 
                        help='Number of test samples')
    parser.add_argument('--template_type', type=str, default='qwen-instruct',
                        help='Template type for prompt formatting')
    
    args = parser.parse_args()
    
    # Load evolved puzzles
    print(f"Loading evolved puzzles from {args.input_file}")
    puzzles = load_evolved_puzzles(args.input_file)
    
    # Process puzzles
    train_dataset, test_dataset, processed_puzzles = process_evolved_puzzles(puzzles, args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_path = output_dir / 'train.parquet'
    test_path = output_dir / 'test.parquet'
    
    print(f"\nSaving train dataset to {train_path}")
    train_dataset.to_parquet(str(train_path))
    
    print(f"Saving test dataset to {test_path}")
    test_dataset.to_parquet(str(test_path))
    
    # Save complete processed puzzles as JSON
    complete_json_path = output_dir / 'evol_zebra_50_complete.json'
    print(f"Saving complete puzzles to {complete_json_path}")
    with open(complete_json_path, 'w', encoding='utf-8') as f:
        json.dump(puzzles, f, indent=2)
    
    # Save human-readable sample from parquet files
    human_readable_path = output_dir / 'sample_puzzles_from_parquet.json'
    save_human_readable_json(train_path, test_path, human_readable_path)
    
    # Save metadata
    metadata = {
        'source_file': args.input_file,
        'total_puzzles': len(puzzles),
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'template_type': args.template_type,
        'evolution_methods': {},
        'all_solvable': True,  # All puzzles were verified as solvable
        'n_attributes': 2,
        'm_objects': 2,
        'difficulty_level': 'n2_m2'
    }
    
    # Count evolution methods
    for puzzle in puzzles:
        method = puzzle.get('evolution_method', 'unknown')
        metadata['evolution_methods'][method] = metadata['evolution_methods'].get(method, 0) + 1
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - train.parquet ({len(train_dataset)} samples)")
    print(f"  - test.parquet ({len(test_dataset)} samples)")
    print(f"  - evol_zebra_50_complete.json (complete puzzles)")
    print(f"  - sample_puzzles_from_parquet.json (human-readable sample from parquet)")
    print(f"  - metadata.json (dataset metadata)")


if __name__ == '__main__':
    main()