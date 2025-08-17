#!/usr/bin/env python3
"""
Complete EvolInstruct pipeline for zebra puzzle generation and verification.
"""

import json
import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
import openai

# Load environment variables
load_dotenv()

class ZebraPuzzleGenerator:
    """Generates 2x2 zebra puzzles using EvolInstruct methodology."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def load_seed_puzzles(self, num_samples: int = 3) -> List[str]:
        """Load successful 2x2 puzzle texts as seeds."""
        results_path = "../loong_logic/results/zebra/aggregated_results.json"
        
        if not os.path.exists(results_path):
            # Fallback to manual seeds if results not available
            return self._get_manual_seeds()
        
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Extract successful 2x2 puzzle texts
        successful_puzzles = []
        for result in results_data.get('results', []):
            if result.get('success', False) and result.get('verification', {}).get('is_correct', False):
                puzzle_text = result['puzzle']['problem']
                # Only 2x2 puzzles and avoid problematic ones
                if "2 people" in puzzle_text and "2 different categories" in puzzle_text and "0 is at" not in puzzle_text:
                    successful_puzzles.append(puzzle_text)
                    if len(successful_puzzles) >= num_samples:
                        break
        
        return successful_puzzles if successful_puzzles else self._get_manual_seeds()
    
    def _get_manual_seeds(self) -> List[str]:
        """Fallback manual seed puzzles."""
        return [
            """You are given an Einstein logic puzzle. In this puzzle, there are 2 people living in the same community. Each person has exactly one attribute from each of 2 different categories. No two people share the same attribute from any category.

Setup:
The puzzle involves the following categories, each with 2 unique items:
1. Nationality: polish, chinese
2. Sport: swimming, water-polo

Clues:
1. the person who plays swimming is immediately to the left of the chinese person

Expected Output Format:
Your solution should be a dictionary where each key is a category name and each value is a list of items, ordered by position from left to right.""",

            """You are given an Einstein logic puzzle. In this puzzle, there are 2 people living in the same community. Each person has exactly one attribute from each of 2 different categories. No two people share the same attribute from any category.

Setup:
The puzzle involves the following categories, each with 2 unique items:
1. Color: red, blue
2. Pet: cat, dog

Clues:
1. the person with the red attribute has a cat

Expected Output Format:
Your solution should be a dictionary where each key is a category name and each value is a list of items, ordered by position from left to right."""
        ]
    
    def evolve_puzzle(self, seed_puzzle: str, evolution_method: str = "complexity") -> str:
        """Evolve a puzzle using different EvolInstruct methods."""
        
        if evolution_method == "complexity":
            prompt = f"""Please act as an expert Puzzle Creator.
Your objective is to rewrite the given puzzle into a more complex version while keeping it solvable.
The rewritten puzzle must be reasonable and solvable by humans.
Make it more challenging by adding nuanced relationships or requiring deeper logical reasoning.

Original puzzle:
{seed_puzzle}

Requirements:
1. Keep exactly 2 people and 2 categories
2. Use DIFFERENT categories and items than the original
3. Make the clue more sophisticated but still clear
4. Ensure unique solvability

Create a NEW puzzle following the same format. Return only the puzzle text."""

        elif evolution_method == "diversity":
            prompt = f"""Please act as an expert Puzzle Creator.
Your objective is to create a completely new puzzle inspired by the given one.
Use a different domain/context while maintaining the logical structure.

Original puzzle:
{seed_puzzle}

Requirements:
1. Keep exactly 2 people and 2 categories
2. Use COMPLETELY DIFFERENT categories (e.g., Food+Music, Job+Vehicle, etc.)
3. Create a clear, unambiguous clue
4. Ensure the puzzle has exactly one solution

Create a NEW puzzle following the same format. Return only the puzzle text."""

        elif evolution_method == "reasoning":
            prompt = f"""Please act as an expert Puzzle Creator.
Your objective is to enhance the given puzzle to require multi-step logical reasoning.
Make solvers think through intermediate steps.

Original puzzle:
{seed_puzzle}

Requirements:
1. Keep exactly 2 people and 2 categories
2. Use different categories and items
3. Create a clue that requires inference or elimination reasoning
4. Ensure unique solvability

Create a NEW puzzle following the same format. Return only the puzzle text."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_evolved_puzzles(self, num_puzzles: int = 6) -> List[Dict[str, Any]]:
        """Generate multiple evolved puzzles using different methods."""
        seed_puzzles = self.load_seed_puzzles()
        evolution_methods = ["complexity", "diversity", "reasoning"]
        
        evolved_puzzles = []
        
        for i in range(num_puzzles):
            seed = seed_puzzles[i % len(seed_puzzles)]
            method = evolution_methods[i % len(evolution_methods)]
            
            print(f"Generating puzzle {i+1}/{num_puzzles} using {method} evolution...")
            
            try:
                evolved_text = self.evolve_puzzle(seed, method)
                
                evolved_puzzles.append({
                    "id": f"evolved_{i}",
                    "puzzle_text": evolved_text,
                    "evolution_method": method,
                    "seed_index": i % len(seed_puzzles),
                    "status": "generated"
                })
                
                print(f"  ✓ Generated puzzle using {method}")
                
            except Exception as e:
                print(f"  ✗ Failed to generate puzzle: {e}")
                evolved_puzzles.append({
                    "id": f"evolved_{i}",
                    "puzzle_text": "",
                    "evolution_method": method,
                    "seed_index": i % len(seed_puzzles),
                    "status": "failed",
                    "error": str(e)
                })
        
        return evolved_puzzles


class ZebraPuzzleVerifier:
    """Verifies zebra puzzles by generating code and solutions."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def extract_puzzle_structure(self, puzzle_text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract categories and items from puzzle text."""
        lines = puzzle_text.split('\n')
        categories = []
        items = {}
        
        in_setup = False
        for line in lines:
            if "Setup:" in line or "setup:" in line:
                in_setup = True
                continue
            elif "Clues:" in line or "clues:" in line:
                break
            
            if in_setup and line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                # Parse category line like "1. Sport: soccer, tennis" or "- Color: red, blue"
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        # Extract category name (remove number/bullet)
                        category_part = parts[0].strip()
                        if '.' in category_part:
                            category = category_part.split('.')[-1].strip()
                        elif '-' in category_part:
                            category = category_part.replace('-', '').strip()
                        else:
                            category = category_part.strip()
                        
                        # Extract items
                        items_text = ':'.join(parts[1:]).strip()
                        items_list = [item.strip() for item in items_text.split(',')]
                        
                        if len(items_list) == 2:  # Only accept 2x2
                            categories.append(category)
                            items[category] = items_list
        
        return categories, items
    
    def generate_solution_code(self, puzzle_text: str, categories: List[str], items: Dict[str, List[str]]) -> str:
        """Generate Python constraint code to solve the puzzle."""
        
        prompt = f"""Generate Python code using python-constraint library to solve this 2x2 Einstein logic puzzle:

{puzzle_text}

The puzzle has these categories and items:
Categories: {categories}
Items: {items}

Requirements:
1. Use 'from constraint import Problem, AllDifferentConstraint'
2. Create variables for each item with positions 0,1
3. Add AllDifferentConstraint for each category
4. Parse the clue and add appropriate constraint
5. Return solution as dictionary format

Template:
```python
from constraint import Problem, AllDifferentConstraint

def solve_puzzle():
    problem = Problem()
    
    categories = {categories}
    items = {items}
    
    # Add variables
    for category in categories:
        for item in items[category]:
            var_name = f"{{category}}:{{item}}"
            problem.addVariable(var_name, [0, 1])
    
    # Category constraints
    for category in categories:
        cat_vars = [f"{{category}}:{{item}}" for item in items[category]]
        problem.addConstraint(AllDifferentConstraint(), cat_vars)
    
    # Add clue constraint here based on the specific clue
    # [ANALYZE THE CLUE AND ADD APPROPRIATE CONSTRAINT]
    
    solutions = problem.getSolutions()
    if len(solutions) != 1:
        raise ValueError(f"Expected 1 solution, got {{len(solutions)}}")
    
    solution = solutions[0]
    result = {{}}
    for category in categories:
        result[category] = ["", ""]
        for item in items[category]:
            pos = solution[f"{{category}}:{{item}}"]
            result[category][pos] = item
    
    return result

print(solve_puzzle())
```

Generate the complete working code with the correct constraint for the clue. Return only executable Python code."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        
        code = response.choices[0].message.content.strip()
        
        # Clean up code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    def execute_code(self, code: str) -> Tuple[bool, str, Dict]:
        """Execute the generated code and return results."""
        try:
            import io
            from contextlib import redirect_stdout
            
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                exec(code, {})
            
            output = stdout_buffer.getvalue().strip()
            
            # Parse output as dictionary
            import ast
            result_dict = ast.literal_eval(output)
            
            if isinstance(result_dict, dict) and len(result_dict) == 2:
                return True, output, result_dict
            else:
                return False, f"Invalid output format: {output}", {}
                
        except Exception as e:
            return False, f"Execution error: {str(e)}", {}
    
    def generate_expected_answer(self, puzzle_text: str, categories: List[str], items: Dict[str, List[str]]) -> str:
        """Generate the expected answer using LLM reasoning."""
        
        prompt = f"""Solve this 2x2 Einstein logic puzzle step by step:

{puzzle_text}

Categories: {categories}
Items: {items}

Think through this logically:
1. Identify what the clue tells us
2. Use process of elimination
3. Determine the unique solution

Provide your reasoning and then the final answer in this exact format:
{{"Category1": ["item_at_position_0", "item_at_position_1"], "Category2": ["item_at_position_0", "item_at_position_1"]}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    def verify_puzzle(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete verification of a puzzle: extract structure, generate code, test execution."""
        
        puzzle_text = puzzle_data["puzzle_text"]
        
        if not puzzle_text.strip():
            return {
                **puzzle_data,
                "verification_status": "failed",
                "error": "Empty puzzle text",
                "categories": [],
                "items": {},
                "code": "",
                "execution_success": False,
                "solution": {},
                "expected_answer": ""
            }
        
        try:
            # Extract structure
            categories, items = self.extract_puzzle_structure(puzzle_text)
            
            if len(categories) != 2 or any(len(items.get(cat, [])) != 2 for cat in categories):
                return {
                    **puzzle_data,
                    "verification_status": "invalid_structure",
                    "error": f"Invalid 2x2 structure: {categories}, {items}",
                    "categories": categories,
                    "items": items,
                    "code": "",
                    "execution_success": False,
                    "solution": {},
                    "expected_answer": ""
                }
            
            # Generate solution code
            code = self.generate_solution_code(puzzle_text, categories, items)
            
            # Execute code
            execution_success, execution_output, solution = self.execute_code(code)
            
            # Generate expected answer
            expected_answer = self.generate_expected_answer(puzzle_text, categories, items)
            
            return {
                **puzzle_data,
                "verification_status": "success" if execution_success else "execution_failed",
                "categories": categories,
                "items": items,
                "code": code,
                "execution_success": execution_success,
                "execution_output": execution_output,
                "solution": solution,
                "expected_answer": expected_answer,
                "solvable": execution_success
            }
            
        except Exception as e:
            return {
                **puzzle_data,
                "verification_status": "error",
                "error": str(e),
                "categories": [],
                "items": {},
                "code": "",
                "execution_success": False,
                "solution": {},
                "expected_answer": ""
            }


def main():
    """Main pipeline: generate evolved puzzles and verify them."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    print("🧩 Starting EvolInstruct Zebra Puzzle Pipeline...")
    
    # Initialize components
    generator = ZebraPuzzleGenerator(api_key)
    verifier = ZebraPuzzleVerifier(api_key)
    
    # Generate evolved puzzles
    print("\n=== Phase 1: Generating Evolved Puzzles ===")
    evolved_puzzles = generator.generate_evolved_puzzles(num_puzzles=3)  # Reduced for testing
    
    generated_count = len([p for p in evolved_puzzles if p["status"] == "generated"])
    print(f"Generated {generated_count}/{len(evolved_puzzles)} puzzles successfully")
    
    # Verify puzzles
    print("\n=== Phase 2: Verifying Puzzles ===")
    verified_puzzles = []
    
    for i, puzzle in enumerate(evolved_puzzles):
        print(f"Verifying puzzle {i+1}/{len(evolved_puzzles)}...")
        
        if puzzle["status"] != "generated":
            print(f"  ⚠️ Skipping failed generation")
            verified_puzzles.append(puzzle)
            continue
        
        verified_puzzle = verifier.verify_puzzle(puzzle)
        verified_puzzles.append(verified_puzzle)
        
        status = verified_puzzle["verification_status"]
        if status == "success":
            print(f"  ✅ Verification successful - puzzle is solvable")
        else:
            print(f"  ❌ Verification failed: {status}")
    
    # Results summary
    print("\n=== Results Summary ===")
    successful_verifications = [p for p in verified_puzzles if p.get("verification_status") == "success"]
    solvable_puzzles = [p for p in verified_puzzles if p.get("solvable", False)]
    
    print(f"Total puzzles: {len(verified_puzzles)}")
    print(f"Successfully generated: {generated_count}")
    print(f"Successfully verified: {len(successful_verifications)}")
    print(f"Solvable puzzles: {len(solvable_puzzles)}")
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    
    # Save detailed results
    output_file = "./results/evol_zebra_complete.json"
    with open(output_file, 'w') as f:
        json.dump(verified_puzzles, f, indent=2)
    print(f"Saved detailed results to {output_file}")
    
    # Save summary of successful puzzles
    if solvable_puzzles:
        summary_file = "./results/evol_zebra_summary.json"
        summary_data = []
        
        for puzzle in solvable_puzzles:
            summary_data.append({
                "id": puzzle["id"],
                "evolution_method": puzzle["evolution_method"],
                "categories": puzzle["categories"],
                "items": puzzle["items"],
                "puzzle_text": puzzle["puzzle_text"][:200] + "...",
                "solution": puzzle["solution"]
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to {summary_file}")
        
        # Show example
        example = solvable_puzzles[0]
        print(f"\n=== Example Solvable Puzzle ===")
        print(f"Method: {example['evolution_method']}")
        print(f"Categories: {example['categories']}")
        print(f"Items: {example['items']}")
        print(f"Solution: {example['solution']}")


if __name__ == "__main__":
    main()