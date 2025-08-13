#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import numpy as np
import random
import torch
import time
import re
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string."""
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure."""
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def parse_dict_answer(answer_text: str) -> Optional[Dict]:
    """Parse dictionary-style answer from text."""
    print("\n[Dictionary Parsing]")
    
    # Try to find dictionary patterns in the text
    dict_patterns = [
        r'\{[^{}]*\}',  # Simple dict pattern
        r'\{.*?\}',     # Greedy dict pattern
    ]
    
    for pattern in dict_patterns:
        matches = re.findall(pattern, answer_text, re.DOTALL)
        for match in matches:
            try:
                # Try to parse as JSON first
                parsed_dict = json.loads(match.replace("'", '"'))
                print(f"  Successfully parsed JSON: {parsed_dict}")
                return parsed_dict
            except json.JSONDecodeError:
                try:
                    # Try to parse as Python literal
                    parsed_dict = ast.literal_eval(match)
                    print(f"  Successfully parsed Python literal: {parsed_dict}")
                    return parsed_dict
                except (ValueError, SyntaxError):
                    print(f"  Failed to parse: {match}")
                    continue
    
    print("  [Error] No valid dictionary found in answer")
    return None

def normalize_ground_truth(ground_truth: Any) -> Optional[Dict]:
    """Normalize ground truth to consistent dictionary format."""
    print("\n[Ground Truth Normalization]")
    
    if isinstance(ground_truth, dict):
        print(f"  Ground truth is already dict: {ground_truth}")
        return ground_truth
    elif isinstance(ground_truth, str):
        try:
            # Try to parse as JSON
            parsed_dict = json.loads(ground_truth.replace("'", '"'))
            print(f"  Parsed JSON ground truth: {parsed_dict}")
            return parsed_dict
        except json.JSONDecodeError:
            try:
                # Try to parse as Python literal
                parsed_dict = ast.literal_eval(ground_truth)
                print(f"  Parsed Python literal ground truth: {parsed_dict}")
                return parsed_dict
            except (ValueError, SyntaxError):
                print(f"  [Error] Failed to parse ground truth: {ground_truth}")
                return None
    else:
        print(f"  [Error] Unsupported ground truth type: {type(ground_truth)}")
        return None

def compare_solutions(model_answer: Dict, ground_truth: Dict) -> Tuple[bool, Dict]:
    """Compare model answer with ground truth solution."""
    print("\n[Solution Comparison]")
    comparison_details = {
        'categories_match': False,
        'exact_match': False,
        'partial_matches': {},
        'missing_categories': [],
        'extra_categories': []
    }
    
    # Check if inputs are dictionaries, return False if not
    if not isinstance(model_answer, dict) or not isinstance(ground_truth, dict):
        print(f"  [Error] Invalid types - model_answer: {type(model_answer)}, ground_truth: {type(ground_truth)}")
        return False, comparison_details
    
    # Check if categories match
    model_categories = set(model_answer.keys()) if model_answer else set()
    truth_categories = set(ground_truth.keys()) if ground_truth else set()
    
    comparison_details['missing_categories'] = list(truth_categories - model_categories)
    comparison_details['extra_categories'] = list(model_categories - truth_categories)
    comparison_details['categories_match'] = model_categories == truth_categories
    
    print(f"  Model categories: {sorted(model_categories)}")
    print(f"  Truth categories: {sorted(truth_categories)}")
    print(f"  Categories match: {comparison_details['categories_match']}")
    
    if comparison_details['missing_categories']:
        print(f"  Missing categories: {comparison_details['missing_categories']}")
    if comparison_details['extra_categories']:
        print(f"  Extra categories: {comparison_details['extra_categories']}")
    
    # Compare each category
    exact_match = True
    for category in truth_categories:
        if category not in model_answer:
            comparison_details['partial_matches'][category] = False
            exact_match = False
            continue
            
        model_list = model_answer[category]
        truth_list = ground_truth[category]
        
        # For Einstein puzzles, order matters (position-based)
        # Compare case-insensitively by converting both lists to lowercase
        def normalize_list(lst):
            if isinstance(lst, list):
                return [str(item).lower().strip() if item is not None else '' for item in lst]
            else:
                return [str(lst).lower().strip() if lst is not None else '']
        
        model_list_normalized = normalize_list(model_list)
        truth_list_normalized = normalize_list(truth_list)
        category_match = model_list_normalized == truth_list_normalized
        comparison_details['partial_matches'][category] = category_match
        
        print(f"  {category}:")
        print(f"    Model: {model_list}")
        print(f"    Truth: {truth_list}")
        print(f"    Match: {category_match}")
        
        if not category_match:
            exact_match = False
    
    comparison_details['exact_match'] = exact_match
    print(f"  Overall exact match: {exact_match}")
    
    return exact_match, comparison_details

def compute_score(solution_str: str, ground_truth: Any, 
                 format_reward: int = 1,
                 answer_reward: float = 1.0) -> float:
    """Compute reward score for zebra puzzle solutions using the same system as kk.py."""
    print("="*60)
    print("ZEBRA PUZZLE SCORING")
    print("="*60)
    
    # Extract model answer
    extracted_answer, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    
    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    
    # Normalize ground truth
    normalized_truth = normalize_ground_truth(ground_truth)
    if normalized_truth is None:
        print("[Final Score] Failed to normalize ground truth")
        return -abs(format_reward) + (-2)
    
    # Validate answer content
    answer_score = 0
    if format_correct and extracted_answer:
        model_answer = parse_dict_answer(extracted_answer)
        if model_answer:
            print(f"\n[Content Validation]")
            is_correct, comparison_details = compare_solutions(model_answer, normalized_truth)
            
            if is_correct:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print("  Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*60)
    print(f" Final Score ".center(60, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*60)

    return total_score

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

def write_jsonl(output_file, data):
    """Write data to a JSONL file."""
    with open(output_file, "w+") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

def init_seed(seed=42):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_zebra_data(data_path, split="test"):
    """Load zebra puzzle data from parquet file."""
    parquet_path = os.path.join(data_path, f"{split}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records from {parquet_path}")
    
    # Convert to list of dictionaries
    records = df.to_dict('records')
    return records

def create_zebra_prompt(question, system_message=None):
    """Create a prompt for zebra puzzle evaluation."""
    if system_message is None:
        system_message = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, provide your answer as a dictionary within the <answer> tags."""
    
    prompt = f"""<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>"""
    
    return prompt

def eval_zebra_subject(args, level, generator, test_records, exist_result_records):
    """Evaluate zebra puzzles for one difficulty level."""
    scores = []
    start_index = len(exist_result_records)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing {level} starting from index {start_index}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total records: {len(test_records)}, Existing results: {len(exist_result_records)}")

    # Prepare all prompts
    prompts = []
    for i in range(start_index, len(test_records)):
        record = test_records[i]
        question = record.get('question', '')
        prompt = create_zebra_prompt(question)
        prompts.append(prompt)

    if not prompts:
        print(f"No new prompts to process for {level}")
        return [], 0.0, exist_result_records

    print(f"Generating {len(prompts)} responses...")
    responses = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        try:
            result = generator(
                prompt,
                max_new_tokens=args.max_token,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True if args.temperature > 0 else False,
                return_full_text=False
            )
            response = result[0]['generated_text']
            responses.append(response)
        except Exception as e:
            print(f"Error generating response for prompt {i+1}: {e}")
            responses.append("")

    # Process results and compute scores
    for i, (prompt, response) in enumerate(zip(prompts, responses), start=start_index):
        record = test_records[i]
        
        # Extract ground truth answer
        ground_truth = record.get('final_answer', '')
        
        # Compute score using zebra.py scoring system
        full_response = prompt + response
        score = compute_score(full_response, ground_truth)
        scores.append(score)
        
        # Create result record - convert any numpy types to Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            return obj
        
        new_item = {
            'id': convert_to_json_serializable(record.get('id', '')),
            'index': i,
            'question': convert_to_json_serializable(record.get('question', '')),
            'ground_truth': convert_to_json_serializable(ground_truth),
            'model_response': response,
            'full_prompt': prompt,
            'score': float(score),
            'correct': score > 0,  # Consider positive scores as correct
            'level': level,
            'n_attributes': convert_to_json_serializable(record.get('n_attributes', 0)),
            'm_objects': convert_to_json_serializable(record.get('m_objects', 0)),
            'difficulty_score': convert_to_json_serializable(record.get('difficulty_score', 0)),
            'clues': convert_to_json_serializable(record.get('clues', [])),
        }
        
        exist_result_records.append(new_item)

    avg_score = np.mean(scores) if scores else 0.0
    accuracy = np.mean([score > 0 for score in scores]) if scores else 0.0
    print(f"Level: {level}, Average Score: {avg_score:.3f}, Accuracy: {accuracy:.3f}")
    return scores, avg_score, exist_result_records

def save_final_results(all_scores, results, fname):
    """Save final results to a file."""
    if all_scores:
        overall_avg_score = np.mean(np.concatenate(all_scores))
        overall_accuracy = np.mean([score > 0 for score in np.concatenate(all_scores)])
        results["overall_average_score"] = overall_avg_score
        results["overall_accuracy"] = overall_accuracy
        print(f"Overall Average Score: {overall_avg_score:.3f}")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)

def load_previous_results(fname):
    """Load previous results."""
    results = {"levels": {}}
    if os.path.isfile(fname):
        with open(fname, 'r', encoding='utf-8') as file:
            results = json.load(file)
    return results

def main(args):
    model_short_name = "/".join(args.model.split("/")[-2:])
    prefix = os.path.join(args.save_dir, model_short_name)
    
    config = f"zebra_{args.level}_token{args.max_token}"
    output_folder = os.path.join(prefix, config)
    results_fname = os.path.join(prefix, f"results_{config}.json")
    os.makedirs(output_folder, exist_ok=True)

    print("Configuration:", config)
    print("Output Folder:", output_folder)
    print("Results File:", results_fname)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting zebra evaluation process...")

    try:
        # Load zebra data
        data_path = os.path.join(args.data_dir, "loong", "zebra_raw_levels", args.level)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading zebra data from {data_path}")
        test_records = load_zebra_data(data_path, split="test")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Loaded {len(test_records)} test records")

        # Initialize model and tokenizer
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
        print(f"  Model path: {args.model}")
        print(f"  Max tokens: {args.max_token}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model and pipeline initialized successfully")
        
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Load previous results
    previous_results = load_previous_results(results_fname)
    
    # Prepare output file
    result_outfile = os.path.join(output_folder, f"{args.level}_results.jsonl")
    exist_result_records = load_jsonl(result_outfile)
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        test_records = test_records[:args.limit]
        print(f"Limited to {len(test_records)} test records")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation...")
    
    try:
        scores, avg_score, result_records = eval_zebra_subject(
            args, args.level, generator, test_records, exist_result_records
        )
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Writing {len(result_records)} results to {result_outfile}")
        write_jsonl(result_outfile, result_records)
        
        # Update results
        accuracy = np.mean([score > 0 for score in scores]) if scores else 0.0
        previous_results["levels"][args.level] = {
            "average_score": avg_score,
            "accuracy": accuracy,
            "total_samples": len(scores)
        }
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Completed {args.level}: avg_score = {avg_score:.4f}, accuracy = {accuracy:.4f}")
        
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ ERROR evaluating {args.level}: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving final results...")
    save_final_results([scores], previous_results, results_fname)
    
    # Show final results summary
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === RESULTS SUMMARY ===")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Results directory: {output_folder}")
    if os.path.exists(result_outfile):
        with open(result_outfile, 'r') as f:
            lines = sum(1 for _ in f)
        file_size = os.path.getsize(result_outfile) / 1024  # KB
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   📄 {os.path.basename(result_outfile)}: {lines} results, {file_size:.1f} KB")
    
    if os.path.exists(results_fname):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Results summary: {results_fname}")
        try:
            with open(results_fname, 'r') as f:
                results_data = json.load(f)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Overall average score: {results_data.get('overall_average_score', 'N/A')}")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Overall accuracy: {results_data.get('overall_accuracy', 'N/A')}")
        except:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Could not read results file")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Zebra evaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for Zebra puzzles")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", "-s", type=str, default="eval_zebra_results", help="Save directory")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name or path")
    parser.add_argument("--level", "-l", type=str, default="n2_m2", help="Zebra puzzle difficulty level (e.g., n2_m2)")
    parser.add_argument("--max_token", type=int, default=8192, help="Maximum number of tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    init_seed()
    main(args)