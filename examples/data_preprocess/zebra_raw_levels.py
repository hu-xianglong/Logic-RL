""" Preprocess Zebra Puzzle (Raw) dataset from loong_logic submodule for Einstein logic puzzles - Level by Level """

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


def extract_level_info(batch_name):
    """Extract level information from batch directory name (e.g., 'n3_m4' -> n_attributes=3, m_objects=4)."""
    parts = batch_name.split('_')
    if len(parts) != 2:
        return None, None
    
    try:
        n_attributes = int(parts[0][1:])  # Remove 'n' prefix
        m_objects = int(parts[1][1:])     # Remove 'm' prefix
        return n_attributes, m_objects
    except (ValueError, IndexError):
        return None, None


def calculate_difficulty_score(n_attributes, m_objects):
    """Calculate a difficulty score based on n_attributes and m_objects."""
    # Difficulty scoring: n×m format
    return n_attributes * m_objects


def load_puzzles_by_level(einstein_batch_dir, max_puzzles_per_batch=50):
    """Load puzzles organized by difficulty level."""
    levels = {}
    
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
        
        # Extract level information
        n_attributes, m_objects = extract_level_info(batch_name)
        if n_attributes is None or m_objects is None:
            print(f"Warning: Could not parse level info from {batch_name}, skipping")
            continue
        
        difficulty_score = calculate_difficulty_score(n_attributes, m_objects)
        
        try:
            with open(puzzles_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            puzzles = batch_data.get('puzzles', [])
            # Take max puzzles per batch
            puzzles = puzzles[:max_puzzles_per_batch]
            
            print(f"Loaded {len(puzzles)} puzzles from {batch_name} (difficulty: {difficulty_score})")
            
            level_key = f"n{n_attributes}_m{m_objects}"
            
            level_puzzles = []
            for puzzle in puzzles:
                puzzle_data = {
                    'id': puzzle.get('id'),
                    'question': puzzle.get('problem'),
                    'final_answer': puzzle.get('ground_truth_solution', {}),
                    'batch_source': batch_name,
                    'n_attributes': n_attributes,
                    'm_objects': m_objects,
                    'difficulty_score': difficulty_score,
                    'metadata': puzzle.get('metadata', {}),
                    'clues': puzzle.get('clues', [])
                }
                level_puzzles.append(puzzle_data)
            
            levels[level_key] = {
                'puzzles': level_puzzles,
                'n_attributes': n_attributes,
                'm_objects': m_objects,
                'difficulty_score': difficulty_score,
                'batch_name': batch_name
            }
                
        except Exception as e:
            print(f"Error loading {puzzles_file}: {e}")
            continue
    
    return levels


def analyze_token_distribution_by_level(levels, tokenizer_name="microsoft/DialoGPT-medium", template_type='qwen-instruct'):
    """Analyze token distribution for each level separately."""
    print(f"\nAnalyzing token distribution by level with tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        print("Skipping token analysis")
        return
    
    level_stats = {}
    
    for level_key, level_data in levels.items():
        puzzles = level_data['puzzles']
        token_counts = []
        
        print(f"\nProcessing level {level_key}...")
        
        for i, puzzle in enumerate(tqdm(puzzles, desc=f"Tokenizing {level_key}")):
            question = puzzle['question']
            if not question:
                continue
                
            prompt = make_prefix(question, template_type)
            tokens = tokenizer.encode(str(prompt), add_special_tokens=True)
            token_count = len(tokens)
            token_counts.append(token_count)
            
            # Show one example per level
            if i == 0:
                print(f"\nExample from {level_key}:")
                print(f"N attributes: {puzzle['n_attributes']}, M objects: {puzzle['m_objects']}")
                print(f"Difficulty score: {puzzle['difficulty_score']}")
                print(f"Question preview: {question[:200]}...")
                print(f"Token count: {token_count}")
        
        if not token_counts:
            print(f"No valid token counts found for {level_key}")
            continue
        
        # Calculate statistics for this level
        stats = {
            'level': level_key,
            'n_attributes': level_data['n_attributes'],
            'm_objects': level_data['m_objects'],
            'difficulty_score': level_data['difficulty_score'],
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
        
        level_stats[level_key] = stats
        
        print(f"\n{level_key} Statistics:")
        print(f"Count:      {stats['count']:,}")
        print(f"Mean:       {stats['mean']:.1f}")
        print(f"Median:     {stats['median']:.1f}")
        print(f"Min/Max:    {stats['min']:,} / {stats['max']:,}")
        print(f"95th %ile:  {stats['p95']:.1f}")
    
    # Create comparative plot
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Token distribution by level
        levels_sorted = sorted(level_stats.keys(), key=lambda x: level_stats[x]['difficulty_score'])
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels_sorted)))
        
        for i, level_key in enumerate(levels_sorted):
            stats = level_stats[level_key]
            # Get token counts for this level
            puzzles = levels[level_key]['puzzles']
            token_counts = []
            for puzzle in puzzles:
                prompt = make_prefix(puzzle['question'], template_type)
                tokens = tokenizer.encode(str(prompt), add_special_tokens=True)
                token_counts.append(len(tokens))
            
            ax1.hist(token_counts, bins=30, alpha=0.6, label=f"{level_key} (diff: {stats['difficulty_score']})", 
                    color=colors[i], density=True)
        
        ax1.set_xlabel('Token Count')
        ax1.set_ylabel('Density')
        ax1.set_title('Token Distribution by Difficulty Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean token count vs difficulty
        difficulties = [level_stats[level]['difficulty_score'] for level in levels_sorted]
        means = [level_stats[level]['mean'] for level in levels_sorted]
        
        ax2.scatter(difficulties, means, s=100, alpha=0.7)
        for i, level_key in enumerate(levels_sorted):
            ax2.annotate(level_key, (difficulties[i], means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Difficulty Score (n_attributes × m_objects)')
        ax2.set_ylabel('Mean Token Count')
        ax2.set_title('Token Count vs Difficulty Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = 'zebra_raw_token_distribution_by_level.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComparative plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error creating comparative plot: {e}")
    
    return level_stats


def process_level_datasets(levels, args):
    """Process and save datasets for each level separately."""
    
    for level_key, level_data in levels.items():
        puzzles = level_data['puzzles']
        
        if not puzzles:
            print(f"No puzzles found for level {level_key}, skipping")
            continue
        
        print(f"\nProcessing level {level_key} with {len(puzzles)} puzzles")
        
        # Calculate train/test split for this level
        total_puzzles = len(puzzles)
        level_train_size = min(args.train_size, int(total_puzzles * 0.8))
        level_test_size = min(args.test_size, total_puzzles - level_train_size)
        
        if level_train_size + level_test_size > total_puzzles:
            level_test_size = total_puzzles - level_train_size
        
        print(f"Level {level_key}: {level_train_size} train, {level_test_size} test samples")
        
        # Create dataset from the puzzles
        raw_dataset = Dataset.from_list(puzzles)
        
        # Split into train and test
        train_dataset = raw_dataset.select(range(level_train_size))
        test_dataset = raw_dataset.select(range(level_train_size, level_train_size + level_test_size))

        def make_map_fn(split, level_info):
            def process_fn(example, idx):
                question = make_prefix(example['question'], template_type=args.template_type)
                
                ground_truth = example['final_answer']
                
                # Create the standardized data format
                data = {
                    "data_source": f"zebra_raw_{level_key}",
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
                        'level': level_key,
                        'n_attributes': example['n_attributes'],
                        'm_objects': example['m_objects'],
                        'difficulty_score': example['difficulty_score'],
                        'metadata': example['metadata']
                    }
                }
                return data
            return process_fn

        print(f"Processing train dataset for {level_key}...")
        train_dataset = train_dataset.map(function=make_map_fn('train', level_data), with_indices=True)
        
        print(f"Processing test dataset for {level_key}...")
        test_dataset = test_dataset.map(function=make_map_fn('test', level_data), with_indices=True)

        # Create level-specific directory
        level_dir = os.path.join(args.local_dir, level_key)
        os.makedirs(level_dir, exist_ok=True)

        # Save datasets
        train_path = os.path.join(level_dir, 'train.parquet')
        test_path = os.path.join(level_dir, 'test.parquet')
        
        print(f"Saving train dataset to {train_path}")
        train_dataset.to_parquet(train_path)
        
        print(f"Saving test dataset to {test_path}")
        test_dataset.to_parquet(test_path)
        
        # Save level metadata
        metadata = {
            'level': level_key,
            'n_attributes': level_data['n_attributes'],
            'm_objects': level_data['m_objects'],
            'difficulty_score': level_data['difficulty_score'],
            'batch_source': level_data['batch_name'],
            'total_puzzles': len(puzzles),
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset)
        }
        
        metadata_path = os.path.join(level_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Level {level_key} processing complete:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Difficulty score: {level_data['difficulty_score']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/loong/zebra_raw_levels')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--einstein_batch_dir', default='loong_logic/data/generated/einstein_batch')
    parser.add_argument('--max_puzzles_per_batch', type=int, default=50)
    parser.add_argument('--train_size', type=int, default=400, help='Max train samples per level')
    parser.add_argument('--test_size', type=int, default=100, help='Max test samples per level')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--tokenizer', default='microsoft/DialoGPT-medium')
    parser.add_argument('--analyze_tokens_only', action='store_true', help='Only analyze tokens, dont save datasets')
    
    args = parser.parse_args()
    
    # Load puzzles organized by level
    print(f"Loading puzzles from {args.einstein_batch_dir}")
    levels = load_puzzles_by_level(
        args.einstein_batch_dir, 
        max_puzzles_per_batch=args.max_puzzles_per_batch
    )
    
    if not levels:
        print("No levels found!")
        exit(1)
    
    print(f"\nFound {len(levels)} difficulty levels:")
    for level_key, level_data in sorted(levels.items(), key=lambda x: x[1]['difficulty_score']):
        print(f"  {level_key}: {len(level_data['puzzles'])} puzzles (difficulty: {level_data['difficulty_score']})")
    
    # Analyze token distribution by level
    level_stats = analyze_token_distribution_by_level(levels, args.tokenizer, args.template_type)
    
    if args.analyze_tokens_only:
        print("Token analysis complete. Exiting (--analyze_tokens_only flag set)")
        exit(0)
    
    # Create main local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Process and save datasets for each level
    process_level_datasets(levels, args)
    
    print(f"\nZebra Raw Levels dataset processing complete!")
    print(f"Data saved to: {local_dir}")
    print(f"Levels processed: {len(levels)}")
    
    # Create overall summary
    summary = {
        'total_levels': len(levels),
        'levels': {}
    }
    
    for level_key, level_data in levels.items():
        summary['levels'][level_key] = {
            'n_attributes': level_data['n_attributes'],
            'm_objects': level_data['m_objects'],
            'difficulty_score': level_data['difficulty_score'],
            'total_puzzles': len(level_data['puzzles'])
        }
    
    summary_path = os.path.join(local_dir, 'levels_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        print("Note: HDFS copy functionality requires verl.utils.hdfs_io module")
        # makedirs(hdfs_dir)
        # copy(src=local_dir, dst=hdfs_dir)
