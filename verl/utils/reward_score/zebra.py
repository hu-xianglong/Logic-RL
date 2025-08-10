import re
import json
import ast
from typing import Dict, Any, Optional, Tuple

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
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
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
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
    """Parse dictionary-style answer from text.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
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
    """Normalize ground truth to consistent dictionary format.
    
    Args:
        ground_truth: Ground truth in various formats
        
    Returns:
        Normalized dictionary or None if parsing fails
    """
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
    """Compare model answer with ground truth solution.
    
    Args:
        model_answer: Parsed model answer dictionary
        ground_truth: Parsed ground truth dictionary
        
    Returns:
        Tuple of (is_correct, comparison_details)
    """
    print("\n[Solution Comparison]")
    comparison_details = {
        'categories_match': False,
        'exact_match': False,
        'partial_matches': {},
        'missing_categories': [],
        'extra_categories': []
    }
    
    # Debug: Check types
    print(f"  model_answer type: {type(model_answer)}, value: {model_answer}")
    print(f"  ground_truth type: {type(ground_truth)}, value: {ground_truth}")
    
    # Handle edge case where model_answer might be a set (error case)
    if isinstance(model_answer, set):
        print("  [Error] model_answer is a set, converting to empty dict")
        model_answer = {}
    elif not isinstance(model_answer, dict):
        print(f"  [Error] model_answer is not a dict: {type(model_answer)}, converting to empty dict")
        model_answer = {}
    
    if isinstance(ground_truth, set):
        print("  [Error] ground_truth is a set, converting to empty dict")
        ground_truth = {}
    elif not isinstance(ground_truth, dict):
        print(f"  [Error] ground_truth is not a dict: {type(ground_truth)}, converting to empty dict")
        ground_truth = {}
    
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
    """Compute reward score for zebra puzzle solutions using the same system as kk.py.
    
    Args:
        solution_str: The complete solution string from the model
        ground_truth: Ground truth answer (dict or string representation)
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
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
        print(f"  parse_dict_answer returned type: {type(model_answer)}, value: {model_answer}")
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