#!/usr/bin/env python3
"""
Script to analyze the token distribution of zebra puzzle prompts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import seaborn as sns
from pathlib import Path
import argparse

def load_zebra_data(data_dir="data/loong/zebra"):
    """Load zebra puzzle data from parquet files."""
    data_dir = Path(data_dir)
    
    datasets = {}
    for split in ["train", "test"]:
        file_path = data_dir / f"{split}.parquet"
        if file_path.exists():
            print(f"Loading {split} data from {file_path}")
            datasets[split] = pd.read_parquet(file_path)
            print(f"  {split}: {len(datasets[split])} samples")
        else:
            print(f"Warning: {file_path} not found")
    
    return datasets

def analyze_token_distribution(datasets, tokenizer_name="microsoft/DialoGPT-medium", prompt_key="prompt"):
    """Analyze token distribution of prompts."""
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    for split, df in datasets.items():
        print(f"\nAnalyzing {split} data...")
        
        # Check available columns
        print(f"Available columns: {list(df.columns)}")
        
        # Try to find prompt column
        if prompt_key in df.columns:
            prompts = df[prompt_key].tolist()
        elif 'question' in df.columns:
            prompts = df['question'].tolist()
            prompt_key = 'question'
        elif 'input' in df.columns:
            prompts = df['input'].tolist()
            prompt_key = 'input'
        else:
            print(f"Could not find prompt column. Available: {list(df.columns)}")
            continue
        
        print(f"Using column '{prompt_key}' as prompts")
        
        # Tokenize all prompts
        token_counts = []
        for i, prompt in enumerate(prompts):
            if pd.isna(prompt):
                continue
            tokens = tokenizer.encode(str(prompt), add_special_tokens=True)
            token_counts.append(len(tokens))
            
            # Show a few examples
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"Prompt: {str(prompt)[:200]}...")
                print(f"Token count: {len(tokens)}")
        
        results[split] = {
            'token_counts': token_counts,
            'stats': {
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
        }
    
    return results

def print_statistics(results):
    """Print token distribution statistics."""
    print("\n" + "="*80)
    print("TOKEN DISTRIBUTION STATISTICS")
    print("="*80)
    
    for split, data in results.items():
        stats = data['stats']
        print(f"\n{split.upper()} SET:")
        print(f"  Count:      {stats['count']:,}")
        print(f"  Mean:       {stats['mean']:.1f}")
        print(f"  Std:        {stats['std']:.1f}")
        print(f"  Min:        {stats['min']:,}")
        print(f"  Max:        {stats['max']:,}")
        print(f"  Median:     {stats['median']:.1f}")
        print(f"  25th %ile:  {stats['p25']:.1f}")
        print(f"  75th %ile:  {stats['p75']:.1f}")
        print(f"  90th %ile:  {stats['p90']:.1f}")
        print(f"  95th %ile:  {stats['p95']:.1f}")
        print(f"  99th %ile:  {stats['p99']:.1f}")

def plot_distribution(results, save_path="zebra_token_distribution.png"):
    """Plot token distribution."""
    plt.figure(figsize=(15, 10))
    
    # Determine number of subplots
    n_splits = len(results)
    if n_splits == 1:
        rows, cols = 1, 1
    elif n_splits == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    
    for i, (split, data) in enumerate(results.items(), 1):
        token_counts = data['token_counts']
        stats = data['stats']
        
        plt.subplot(rows, cols, i)
        
        # Histogram
        plt.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
        plt.axvline(stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.1f}')
        plt.axvline(stats['p95'], color='green', linestyle='--', label=f'95th %ile: {stats["p95"]:.1f}')
        
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.title(f'{split.title()} Set Token Distribution\n(n={stats["count"]:,})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    # Also create a combined plot if multiple splits
    if len(results) > 1:
        plt.figure(figsize=(12, 6))
        
        for split, data in results.items():
            token_counts = data['token_counts']
            plt.hist(token_counts, bins=50, alpha=0.6, label=f'{split.title()} (n={len(token_counts):,})', edgecolor='black')
        
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.title('Zebra Puzzle Token Distribution - All Splits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        combined_path = save_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {combined_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze zebra prompt token distribution')
    parser.add_argument('--data_dir', default='data/loong/zebra', help='Directory containing zebra data')
    parser.add_argument('--tokenizer', default='microsoft/DialoGPT-medium', help='Tokenizer to use')
    parser.add_argument('--prompt_key', default='prompt', help='Column name containing prompts')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Load data
    datasets = load_zebra_data(args.data_dir)
    
    if not datasets:
        print("No data found!")
        return
    
    # Analyze token distribution
    results = analyze_token_distribution(datasets, args.tokenizer, args.prompt_key)
    
    if not results:
        print("No results generated!")
        return
    
    # Print statistics
    print_statistics(results)
    
    # Plot distribution
    if not args.no_plot:
        try:
            plot_distribution(results)
            plt.show()
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Continuing without plots...")

if __name__ == "__main__":
    main()