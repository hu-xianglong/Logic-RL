""" Preprocess Zebra Puzzle (Wikipedia) dataset for Einstein logic puzzles """

import os
import json
from datasets import Dataset
from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs  # Optional HDFS support
import shutil
import argparse
import re


def make_prefix(question, template_type):
    """Create the prompt prefix based on template type."""
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n\nUser: {question}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve an Einstein logic puzzle. After thinking, when you finally reach a conclusion, clearly state the solution within <answer> </answer> tags in the required format.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/loong/zebra_puzzle')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='loong_data_separated/zebra_puzzle.json')
    parser.add_argument('--train_size', type=int, default=290)
    parser.add_argument('--test_size', type=int, default=32)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'zebra_puzzle'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load the separated Zebra Puzzle JSON data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Total records: {len(raw_data)}")
    
    # Create dataset from the raw data
    raw_dataset = Dataset.from_list(raw_data)

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE, f"Dataset size {len(raw_dataset)} is smaller than required {TRAIN_SIZE + TEST_SIZE}"
    
    # Split into train and test
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

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
                    "ground_truth": {
                        "final_answer": ground_truth,
                        "rationale": example['rationale']
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx
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

    print(f"Zebra Puzzle dataset processing complete!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}")
        print("Note: HDFS copy functionality requires verl.utils.hdfs_io module")
        # makedirs(hdfs_dir)
        # copy(src=local_dir, dst=hdfs_dir)
