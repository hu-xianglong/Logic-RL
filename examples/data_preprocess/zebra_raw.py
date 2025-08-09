""" Preprocess Zebra Puzzle (Raw) dataset from loong_logic submodule for Einstein logic puzzles """

import os
import json
import glob
from datasets import Dataset
from tqdm import tqdm
import argparse
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt


def parse_solution_from_markdown(table_markdown):
    """Parse the solution from markdown table format to dictionary format."""
    lines = table_markdown.strip().split('\n')
    if len(lines) < 3:
        return None
    
    # Skip header separator line
    header_line = lines[0]
    data_lines = lines[2:]  # Skip the separator line
    
    # Extract position headers (1, 2, 3, etc.)
    positions = header_line.split('|')[2:-1]  # Skip first empty and last empty
    positions = [pos.strip() for pos in positions]
    
    solution = {}
    for line in data_lines:
        parts = line.split('|')
        if len(parts) < 3:
            continue
        category = parts[1].strip()
        values = [part.strip() for part in parts[2:-1]]  # Skip first empty and last empty
        if category and values:
            solution[category] = values
    
    return solution


def make_prefix(question, template_type):
    """Create the prompt prefix based on template type."""
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n\nUser: {question}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def load_puzzles_from_batch_dirs(einstein_batch_dir, max_puzzles_per_batch=50):
    """Load puzzles from all batch directories in einstein_batch."""
    all_puzzles = []
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(einstein_batch_dir, "n*_m*"))
    batch_dirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(batch_dirs)} batch directories:")
    for batch_dir in batch_dirs:
        print(f"  {os.path.basename(batch_dir)}")
    
    for batch_dir in tqdm(batch_dirs, desc="Processing batch directories"):
        batch_name = os.path.basename(batch_dir)
        puzzles_file = os.path.join(batch_dir, "puzzles.json")
        
        if not os.path.exists(puzzles_file):
            print(f"Warning: {puzzles_file} not found, skipping {batch_name}")
            continue
        
        try:
            with open(puzzles_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            puzzles = batch_data.get('puzzles', [])
            # Take max puzzles per batch
            puzzles = puzzles[:max_puzzles_per_batch]
            
            print(f"Loaded {len(puzzles)} puzzles from {batch_name}")
            
            for puzzle in puzzles:
                puzzle_data = {
                    'id': puzzle.get('id'),
                    'question': puzzle.get('problem'),
                    'final_answer': puzzle.get('ground_truth_solution', {}),
                    'batch_source': batch_name,
                    'metadata': puzzle.get('metadata', {}),
                    'clues': puzzle.get('clues', [])
                }
                all_puzzles.append(puzzle_data)
                
        except Exception as e:
            print(f"Error loading {puzzles_file}: {e}")
            continue
    
    print(f"Total puzzles loaded: {len(all_puzzles)}")
    return all_puzzles


def analyze_token_distribution(puzzles, tokenizer_name="microsoft/DialoGPT-medium", template_type='qwen-instruct'):
    """Analyze token distribution of prompts."""
    print(f"\nAnalyzing token distribution with tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        print("Skipping token analysis")
        return
    
    token_counts = []
    
    for i, puzzle in enumerate(tqdm(puzzles, desc="Tokenizing prompts")):
        question = puzzle['question']
        if not question:
            continue
            
        prompt = make_prefix(question, template_type)
        tokens = tokenizer.encode(str(prompt), add_special_tokens=True)
        token_count = len(tokens)
        token_counts.append(token_count)
        
        # Show a few examples
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"Batch: {puzzle['batch_source']}")
            print(f"Question preview: {question[:200]}...")
            print(f"Token count: {token_count}")
    
    if not token_counts:
        print("No valid token counts found")
        return
    
    # Calculate statistics
    stats = {
        'count': len(token_counts),
        'mean': np.mean(token_counts),
        'std': np.std(token_counts),
        'min': np.min(token_counts),
        'max': np.max(token_counts),
        'median': np.median(token_counts),
        'p25': np.percentile(token_counts, 25),
        'p75': np.percentile(token_counts, 75),
        'p90': np.percentile(token_counts, 90),
        'p95': np.percentile(token_counts, 95),
        'p99': np.percentile(token_counts, 99)
    }
    
    print("\n" + "="*80)
    print("TOKEN DISTRIBUTION STATISTICS")
    print("="*80)
    print(f"Count:      {stats['count']:,}")
    print(f"Mean:       {stats['mean']:.1f}")
    print(f"Std:        {stats['std']:.1f}")
    print(f"Min:        {stats['min']:,}")
    print(f"Max:        {stats['max']:,}")
    print(f"Median:     {stats['median']:.1f}")
    print(f"25th %ile:  {stats['p25']:.1f}")
    print(f"75th %ile:  {stats['p75']:.1f}")
    print(f"90th %ile:  {stats['p90']:.1f}")
    print(f"95th %ile:  {stats['p95']:.1f}")
    print(f"99th %ile:  {stats['p99']:.1f}")
    
    # Plot distribution
    try:
        plt.figure(figsize=(12, 6))
        plt.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
        plt.axvline(stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.1f}')
        plt.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.1f}')
        
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.title(f'Zebra Raw Puzzles Token Distribution\n(n={stats["count"]:,}, template={template_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = 'zebra_raw_token_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/loong/zebra_raw')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--einstein_batch_dir', default='loong_logic/data/generated/einstein_batch')
    parser.add_argument('--max_puzzles_per_batch', type=int, default=50)
    parser.add_argument('--train_size', type=int, default=400)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--tokenizer', default='microsoft/DialoGPT-medium')
    parser.add_argument('--analyze_tokens_only', action='store_true', help='Only analyze tokens, dont save datasets')
    
    args = parser.parse_args()
    
    data_source = 'zebra_raw'
    
    # Load puzzles from all batch directories
    print(f"Loading puzzles from {args.einstein_batch_dir}")
    all_puzzles = load_puzzles_from_batch_dirs(
        args.einstein_batch_dir, 
        max_puzzles_per_batch=args.max_puzzles_per_batch
    )
    
    if not all_puzzles:
        print("No puzzles found!")
        exit(1)
    
    # Analyze token distribution
    analyze_token_distribution(all_puzzles, args.tokenizer, args.template_type)
    
    if args.analyze_tokens_only:
        print("Token analysis complete. Exiting (--analyze_tokens_only flag set)")
        exit(0)
    
    # Check if we have enough puzzles
    total_needed = args.train_size + args.test_size
    if len(all_puzzles) < total_needed:
        print(f"Warning: Only {len(all_puzzles)} puzzles available, need {total_needed}")
        print(f"Adjusting sizes: train={min(args.train_size, len(all_puzzles)-args.test_size)}, test={min(args.test_size, len(all_puzzles)//4)}")
        args.train_size = min(args.train_size, len(all_puzzles) - args.test_size)
        args.test_size = min(args.test_size, len(all_puzzles) - args.train_size)
    
    # Create dataset from the puzzles
    raw_dataset = Dataset.from_list(all_puzzles)
    
    # Split into train and test
    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['question'], template_type=args.template_type)
            
            ground_truth = example['final_answer']
            
            # Create the standardized data format
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user", 
                    "content": question,
                }],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'batch_source': example['batch_source'],
                    'puzzle_id': example['id'],
                    'metadata': example['metadata']
                }
            }
            return data
        return process_fn

    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Expand local directory path
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving train dataset to {os.path.join(local_dir, 'train.parquet')}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    print(f"Saving test dataset to {os.path.join(local_dir, 'test.parquet')}")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Zebra Raw dataset processing complete!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Print batch distribution
    print(f"\nBatch distribution in training set:")
    batch_counts = {}
    for item in train_dataset:
        batch = item['extra_info']['batch_source']
        batch_counts[batch] = batch_counts.get(batch, 0) + 1
    
    for batch, count in sorted(batch_counts.items()):
        print(f"  {batch}: {count} puzzles")

    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}")
        print("Note: HDFS copy functionality requires verl.utils.hdfs_io module")
        # makedirs(hdfs_dir)
        # copy(src=local_dir, dst=hdfs_dir)