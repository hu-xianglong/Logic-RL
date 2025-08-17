# Zebra Puzzle Evolution with CAMEL

This project uses CAMEL's EvolInstruct pipeline to generate evolved zebra/Einstein logic puzzles with validated code solutions.

## Features

- **Evolution Pipeline**: Uses CAMEL's EvolInstruct to evolve existing puzzles into more challenging variants
- **Code Generation**: Automatically generates Python constraint programming code for each evolved puzzle
- **Code Validation**: Executes generated code and validates solutions
- **Quality Scoring**: Scores puzzles based on structure, complexity, variety, and clarity
- **Seed Data**: Uses successful puzzles from n2_m2 dataset with 92.5% success rate

## Setup

```bash
# Create virtual environment
uv venv .venv

# Install dependencies (requires additional packages)
uv pip install --python .venv/bin/python rouge-score transformers datasets accelerate

# Test basic functionality
.venv/bin/python simple_test.py
```

## Usage

The main script `evol_zebra_puzzles.py` contains:

1. **ZebraEvolInstructTemplates**: Custom evolution templates for logic puzzles
2. **ZebraScorer**: Quality scorer that evaluates puzzle structure and clarity  
3. **Code Generation**: Automatic generation and validation of constraint programming solutions
4. **Data Pipeline**: Loads successful puzzles from existing results and evolves them

### Evolution Strategies

- **in-depth**: Makes puzzles more complex with advanced clue types
- **in-breadth**: Creates variations with different categories/items
- **condense**: Refines and clarifies existing puzzles

### Key Improvements

- Explicitly avoids generating ambiguous clues like "0 is at position X"
- Validates that generated code executes successfully
- Ensures evolved puzzles have working solutions
- Uses successful puzzles (92.5% success rate) as seed data

## Files

- `evol_zebra_puzzles.py`: Main evolution script
- `simple_test.py`: Basic functionality test
- `test_evol.py`: Full test suite (requires all dependencies)
- `run_evol.py`: Simple runner script
- `pyproject.toml`: Project dependencies

## Data Sources

- **Input**: `/Users/xianglonghu/CascadeProjects/Logic-RL/data/loong/zebra_raw_levels/n2_m2/train.parquet`
- **Results**: `../loong_logic/results/zebra/aggregated_results.json`
- **Output**: `./results/evolved_*.parquet`

## Next Steps

To run the full evolution pipeline:

1. Install remaining dependencies (rouge-score, transformers, etc.)
2. Set OPENAI_API_KEY environment variable
3. Run: `.venv/bin/python evol_zebra_puzzles.py`

The pipeline will generate evolved puzzles with validated code solutions, avoiding the data quality issues found in the n2_m3 dataset.