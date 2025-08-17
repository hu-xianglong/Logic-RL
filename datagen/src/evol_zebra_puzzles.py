#!/usr/bin/env python3
"""
Generate evolved zebra puzzles using CAMEL's EvolInstruct pipeline.
This script takes existing zebra puzzles and evolves them to create more challenging variants.
"""

import json
import logging
import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Optional
import random
import io
from contextlib import redirect_stdout, redirect_stderr

from camel.agents import ChatAgent
from camel.datagen.evol_instruct import EvolInstructPipeline
from camel.datagen.evol_instruct.scorer import BaseScorer
from camel.datagen.evol_instruct.templates import BaseEvolInstructTemplates
from camel.logger import enable_logging, get_logger, set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


@dataclass(frozen=True)
class ZebraEvolInstructTemplates(BaseEvolInstructTemplates):
    """Templates for evolving zebra logic puzzles."""
    
    @property
    def EVOL_METHODS(self) -> Dict[str, str]:
        """Evolution methods for zebra puzzles."""
        return {
            "add_categories": (
                "Add one more category (e.g., if the puzzle has Nationality and Pet, "
                "add a third category like Beverage or Sport). Make sure all categories "
                "have the same number of items as there are positions."
            ),
            "add_positions": (
                "Increase the number of positions/people by 1 (e.g., from 2 to 3 people). "
                "Ensure all categories have items for the new position."
            ),
            "complexify_clues": (
                "Replace simple clues with more complex ones. For example:\n"
                "- Change 'X is at position 1' to 'X is immediately left of Y'\n"
                "- Change 'X is same as Y' to 'X is between Y and Z'\n"
                "- Add negation clues like 'X is not Y' or 'X is not at position 1'\n"
                "Make sure the puzzle remains uniquely solvable."
            ),
            "add_indirect_clues": (
                "Add more clues that require multi-step reasoning. For example:\n"
                "- 'The person who likes X is next to the person who has Y'\n"
                "- 'Either X is at position 1 or Y is at position 2, but not both'\n"
                "- 'The person with X and the person with Y are not adjacent'"
            ),
            "increase_clue_distance": (
                "Make clues refer to items that are further apart logically. Instead of "
                "directly linking items, use transitive relationships that require more "
                "deduction steps to connect."
            ),
            "add_numeric_constraints": (
                "Add clues involving numeric relationships like:\n"
                "- 'X is at an odd-numbered position'\n"
                "- 'X and Y have positions that sum to 3'\n"
                "- 'The distance between X and Y is exactly 2 positions'"
            ),
            "clarify": (
                "Remove any ambiguous language and make clues more precise and clear "
                "without changing the logical structure."
            )
        }
    
    @property
    def STRATEGY(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """Evolution strategies for zebra puzzles."""
        return {
            "in-depth": {
                "instruction": (
                    "Please act as an expert Logic Puzzle Designer.\n"
                    "Your objective is to evolve the given zebra/Einstein logic puzzle "
                    "into a more challenging version that requires deeper logical reasoning.\n"
                    "The evolved puzzle must:\n"
                    "1. Remain uniquely solvable (exactly one valid solution)\n"
                    "2. Be understandable by humans\n"
                    "3. Maintain the Einstein puzzle format\n"
                    "4. Include all necessary setup information\n\n"
                    "Use the following method to evolve the puzzle:\n{method}\n\n"
                    "Make sure to:\n"
                    "- Clearly state the number of positions/people\n"
                    "- List all categories and their items\n"
                    "- Provide clear, unambiguous clues\n"
                    "- Include the expected output format\n"
                    "- Ensure internal consistency\n"
                    "- NEVER use ambiguous clues like '0 is at position X'\n\n"
                    "#Given Puzzle#:\n{prompt}\n\n"
                    "#Evolved Puzzle#:\n"
                ),
                "methods": ["complexify_clues", "add_indirect_clues", "increase_clue_distance", "add_numeric_constraints"]
            },
            "in-breadth": {
                "instruction": (
                    "Please act as an expert Logic Puzzle Designer.\n"
                    "Your objective is to create a new zebra/Einstein logic puzzle inspired by "
                    "the given one, but with different content and structure.\n"
                    "The new puzzle should:\n"
                    "1. Have a unique solution\n"
                    "2. Use different categories and items than the original\n"
                    "3. Maintain similar difficulty level\n"
                    "4. Follow the Einstein puzzle format\n\n"
                    "Use the following method to create variation:\n{method}\n\n"
                    "Make sure to:\n"
                    "- Use fresh, interesting categories (avoid repeating the original)\n"
                    "- Create logical, solvable clues\n"
                    "- Clearly state all setup information\n"
                    "- Provide the expected output format\n"
                    "- NEVER use ambiguous clues like '0 is at position X'\n\n"
                    "#Given Puzzle#:\n{prompt}\n\n"
                    "#New Puzzle#:\n"
                ),
                "methods": ["add_categories", "add_positions"]
            },
            "condense": {
                "instruction": (
                    "Please act as a Logic Puzzle Editor.\n"
                    "Your task is to refine and clarify the given zebra puzzle.\n"
                    "Focus on:\n"
                    "1. Removing any ambiguous language\n"
                    "2. Ensuring clues are precise and clear\n"
                    "3. Maintaining the exact same logical structure\n"
                    "4. Improving readability without changing difficulty\n"
                    "5. NEVER use ambiguous clues like '0 is at position X'\n\n"
                    "#Given Puzzle#:\n{prompt}\n\n"
                    "#Refined Puzzle#:\n"
                ),
                "methods": ["clarify"]
            }
        }


class ZebraScorer(BaseScorer):
    """Scorer for zebra logic puzzles."""
    
    def score(self, instruction: str) -> Dict[str, float]:
        """Score a zebra puzzle based on various criteria."""
        scores = {}
        
        # Check for proper structure
        has_setup = "Setup:" in instruction or "categories" in instruction.lower()
        has_clues = "Clues:" in instruction or "clue" in instruction.lower()
        has_format = "Expected Output Format:" in instruction or "solution should be" in instruction
        
        scores["structure"] = (has_setup + has_clues + has_format) / 3.0
        
        # Count number of clues (complexity indicator)
        clue_count = instruction.lower().count("clue") + instruction.count("\n-") + instruction.count("\n*")
        clue_count += len([line for line in instruction.split('\n') if line.strip() and line.strip()[0].isdigit()])
        scores["complexity"] = min(clue_count / 10.0, 1.0)
        
        # Check for various clue types
        clue_types = 0
        if "immediately" in instruction or "adjacent" in instruction:
            clue_types += 1
        if "same person" in instruction or "same as" in instruction:
            clue_types += 1
        if "left of" in instruction or "right of" in instruction:
            clue_types += 1
        if "position" in instruction and any(str(i) in instruction for i in range(1, 6)):
            clue_types += 1
        if "between" in instruction:
            clue_types += 1
        if "not" in instruction or "different" in instruction:
            clue_types += 1
            
        scores["variety"] = min(clue_types / 4.0, 1.0)
        
        # Check clarity (no ambiguous terms)
        ambiguous_terms = ["0 is at", "maybe", "possibly", "might", "could be", "sometimes"]
        has_ambiguity = any(term in instruction.lower() for term in ambiguous_terms)
        scores["clarity"] = 0.0 if has_ambiguity else 1.0
        
        # Overall quality score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores


def load_seed_puzzles(data_path: str, results_path: str, num_samples: int = 10) -> List[Dict]:
    """Load sample puzzles with their solutions from successful results."""
    # Load original parquet data
    df = pd.read_parquet(data_path)
    
    # Load successful results
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    # Create a mapping of successful puzzle IDs to their data
    successful_puzzles = {}
    for result in results_data.get('results', []):
        if result.get('success', False) and result.get('verification', {}).get('is_correct', False):
            puzzle_id = result['puzzle_id']
            successful_puzzles[puzzle_id] = {
                'question': result['puzzle']['problem'],
                'code': result['code'],
                'final_answer': result['puzzle']['ground_truth_solution'],
                'execution_result': result['execution_result']
            }
    
    # Sample good puzzles with their solutions
    good_puzzles = []
    for _, row in df.iterrows():
        puzzle_id = row['id']
        if puzzle_id in successful_puzzles and "0 is at" not in row['question']:
            puzzle_data = successful_puzzles[puzzle_id]
            good_puzzles.append({
                'question': puzzle_data['question'],
                'code': puzzle_data['code'],
                'final_answer': puzzle_data['final_answer'],
                'original_id': puzzle_id
            })
            if len(good_puzzles) >= num_samples:
                break
    
    return good_puzzles


def execute_code(code: str) -> Tuple[bool, str, Optional[Dict]]:
    """Execute constraint code and return success status, output, and parsed result."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Create a local namespace for execution
            local_namespace = {}
            # Execute the code in the local namespace
            exec(code, globals(), local_namespace)
        
        stdout_output = stdout_buffer.getvalue().strip()
        stderr_output = stderr_buffer.getvalue().strip()
        
        if stderr_output:
            return False, f"Error: {stderr_output}", None
        
        if not stdout_output:
            return False, "No output generated", None
        
        # Try to parse the output as a dictionary
        try:
            import ast
            result_dict = ast.literal_eval(stdout_output)
            if isinstance(result_dict, dict):
                return True, stdout_output, result_dict
            else:
                return False, f"Output is not a dictionary: {stdout_output}", None
        except (SyntaxError, ValueError) as e:
            return False, f"Could not parse output as dictionary: {e}", None
            
    except Exception as e:
        return False, f"Execution error: {str(e)}", None


def generate_solution_code(agent: ChatAgent, puzzle_text: str) -> Tuple[str, str]:
    """Generate both the puzzle and its solution code."""
    code_prompt = f"""
Given this evolved zebra puzzle, generate Python code using the python-constraint library to solve it.

{puzzle_text}

Requirements:
1. Use 'from constraint import Problem, AllDifferentConstraint'
2. Follow the exact structure from the original solver
3. Print the solution as a dictionary
4. Make sure the code is executable

Return ONLY the executable Python code, no explanations.
"""
    
    response = agent.step(code_prompt)
    code = response.msg.content.strip()
    
    # Clean up code (remove markdown if present)
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    # Ensure constraint import
    if "from constraint import" not in code:
        code = "from constraint import Problem, AllDifferentConstraint\n\n" + code
    
    return code


def save_evolved_puzzles(results: List, output_path: str, agent: ChatAgent):
    """Save evolved puzzles with generated and validated code."""
    evolved_data = []
    successful_count = 0
    
    for i, generations in enumerate(results):
        print(f"Processing evolved puzzle {i+1}/{len(results)}...")
        
        # Get the best puzzle from the last generation
        last_gen_key = max(generations.keys())
        last_generation = generations[last_gen_key]
        
        # Find the best candidate
        best_candidate = max(
            last_generation,
            key=lambda x: x["scores"]["overall"] if x["scores"] else 0
        )
        
        puzzle_text = best_candidate["instruction"]
        
        # Generate solution code
        try:
            print(f"  Generating code...")
            code = generate_solution_code(agent, puzzle_text)
            
            # Execute and validate the code
            print(f"  Validating code...")
            success, output, result_dict = execute_code(code)
            
            if success:
                successful_count += 1
                print(f"  ✓ Code executed successfully")
                
                # Extract clues
                clues = []
                if "Clues:" in puzzle_text:
                    clues_section = puzzle_text.split("Clues:")[1].split("Expected Output")[0]
                    clues = [line.strip() for line in clues_section.split("\n") 
                            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))]
                
                evolved_data.append({
                    "id": f"evolved_{i}_{hash(puzzle_text) % 10000000}",
                    "question": puzzle_text,
                    "clues": clues,
                    "data_source": "evolved_zebra",
                    "code": code,
                    "final_answer": json.dumps(result_dict),
                    "execution_result": output,
                    "success": True,
                    "difficulty_score": best_candidate["scores"]["complexity"] * 10 if best_candidate["scores"] else 5,
                    "scores": best_candidate["scores"]
                })
            else:
                print(f"  ✗ Code execution failed: {output}")
                evolved_data.append({
                    "id": f"evolved_{i}_{hash(puzzle_text) % 10000000}",
                    "question": puzzle_text,
                    "clues": [],
                    "data_source": "evolved_zebra",
                    "code": code,
                    "final_answer": "",
                    "execution_result": output,
                    "success": False,
                    "difficulty_score": best_candidate["scores"]["complexity"] * 10 if best_candidate["scores"] else 5,
                    "scores": best_candidate["scores"]
                })
                
        except Exception as e:
            print(f"  ✗ Error generating/executing code: {e}")
            evolved_data.append({
                "id": f"evolved_{i}_{hash(puzzle_text) % 10000000}",
                "question": puzzle_text,
                "clues": [],
                "data_source": "evolved_zebra",
                "code": "",
                "final_answer": "",
                "execution_result": f"Error: {str(e)}",
                "success": False,
                "difficulty_score": best_candidate["scores"]["complexity"] * 10 if best_candidate["scores"] else 5,
                "scores": best_candidate["scores"]
            })
    
    print(f"\nGenerated {len(evolved_data)} puzzles, {successful_count} with working code ({successful_count/len(evolved_data)*100:.1f}%)")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(evolved_data)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")
    
    # Also save as JSON for inspection
    json_path = output_path.replace('.parquet', '.json')
    with open(json_path, 'w') as f:
        json.dump(evolved_data, f, indent=2)
    print(f"Also saved to {json_path} for inspection")


def main():
    """Generate evolved zebra puzzles."""
    # Configuration
    input_data_path = "/Users/xianglonghu/CascadeProjects/Logic-RL/data/loong/zebra_raw_levels/n2_m2/train.parquet"
    results_path = "../loong_logic/results/zebra/aggregated_results.json"
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load seed puzzles with solutions
    print("Loading seed puzzles with solutions...")
    seed_puzzle_data = load_seed_puzzles(input_data_path, results_path, num_samples=5)
    print(f"Loaded {len(seed_puzzle_data)} successful puzzles with solutions")
    
    # Extract just the questions for evolution
    seed_puzzles = [data['question'] for data in seed_puzzle_data]
    
    # Initialize model and agent
    print("Initializing model...")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,  # Using GPT-4O-mini for cost efficiency
        model_config_dict={"temperature": 0.7, "max_tokens": 2000},
    )
    agent = ChatAgent(model=model)
    
    # Initialize pipeline
    print("Setting up evolution pipeline...")
    pipeline = EvolInstructPipeline(
        agent=agent,
        templates=ZebraEvolInstructTemplates(),
    )
    
    # Evolution configuration
    num_generations = 2  # Number of evolution iterations
    
    # Strategy 1: Make puzzles more complex
    print("\n=== Generating complex puzzles ===")
    evol_spec = ["in-depth", "condense"]
    
    results_complex = pipeline.generate(
        prompts=seed_puzzles,
        evolution_spec=evol_spec,
        num_generations=num_generations,
        scorer=ZebraScorer(),
    )
    
    save_evolved_puzzles(
        results_complex, 
        os.path.join(output_dir, "evolved_complex.parquet"),
        agent
    )
    
    # Strategy 2: Create variations
    print("\n=== Generating puzzle variations ===")
    evol_spec = ["in-breadth", "in-depth", "condense"]
    
    results_variations = pipeline.generate(
        prompts=seed_puzzles[:3],  # Use fewer seeds for variations
        evolution_spec=evol_spec,
        num_generations=num_generations,
        scorer=ZebraScorer(),
    )
    
    save_evolved_puzzles(
        results_variations,
        os.path.join(output_dir, "evolved_variations.parquet"),
        agent
    )
    
    print("\n=== Evolution complete! ===")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Set up logging
    enable_logging()
    set_log_level(logging.INFO)
    logger = get_logger("zebra-evol")
    
    logger.info("Starting zebra puzzle evolution...")
    try:
        main()
        logger.info("Evolution completed successfully!")
    except Exception as e:
        logger.error(f"Evolution failed: {str(e)}")
        raise