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
                    print(f"  Successfully parsed literal: {parsed_dict}")
                    return parsed_dict
                except (ValueError, SyntaxError):
                    continue
    
    print("  [Error] Failed to parse dictionary from answer")
    return None

def compare_solutions(predicted: Dict, ground_truth: str) -> float:
    """Compare predicted solution with ground truth.
    
    Args:
        predicted: Predicted solution dictionary
        ground_truth: Ground truth solution string
        
    Returns:
        Score based on correctness (0.0 to 1.0)
    """
    print("\n[Solution Comparison]")
    
    try:
        # Parse ground truth
        if isinstance(ground_truth, str):
            gt_dict = json.loads(ground_truth.replace("'", '"'))
        else:
            gt_dict = ground_truth
            
        print(f"  Ground truth: {gt_dict}")
        print(f"  Predicted: {predicted}")
        
        # Check if structures match
        if set(predicted.keys()) != set(gt_dict.keys()):
            print("  [Error] Key mismatch")
            return 0.0
            
        # Check each category
        total_items = 0
        correct_items = 0
        
        for category in gt_dict:
            if category not in predicted:
                continue
                
            gt_items = gt_dict[category]
            pred_items = predicted[category]
            
            if len(gt_items) != len(pred_items):
                print(f"  [Error] Length mismatch in {category}")
                continue
                
            # Check positional accuracy
            for i, (gt_item, pred_item) in enumerate(zip(gt_items, pred_items)):
                total_items += 1
                if gt_item == pred_item:
                    correct_items += 1
                    
        if total_items == 0:
            return 0.0
            
        accuracy = correct_items / total_items
        print(f"  Accuracy: {correct_items}/{total_items} = {accuracy:.3f}")
        return accuracy
        
    except Exception as e:
        print(f"  [Error] Comparison failed: {e}")
        return 0.0

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, Any],
                 format_weight: float = 0.3,
                 content_weight: float = 0.7) -> float:
    """Computes comprehensive score for logic problem solution.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_weight: Weight for format correctness
        content_weight: Weight for answer correctness
        
    Returns:
        Total score (0.0 to 1.0)
    """
    print("\n" + "="*80)
    print(" Processing Logic Problem ".center(80, '='))
    
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str[:200]}...")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = 1.0 if format_correct else 0.0
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    # Validate answer content
    content_score = 0.0
    if format_correct and answer_text:
        predicted_dict = parse_dict_answer(answer_text)
        if predicted_dict:
            # Get ground truth solution
            gt_solution = ground_truth.get('solution_text_format', '')
            if not gt_solution:
                gt_solution = ground_truth.get('execution_result', '')
            
            content_score = compare_solutions(predicted_dict, gt_solution)
        else:
            print("  [Error] Failed to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # Calculate weighted final score
    final_score = format_weight * format_score + content_weight * content_score
    
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score:.3f} (weight: {format_weight})")
    print(f"  Content: {content_score:.3f} (weight: {content_weight})")
    print(f"  Total: {final_score:.3f}")
    print("="*80 + "\n")

    return final_score