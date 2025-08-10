# Curriculum Training Script Usage Examples

## Basic Usage

### Train all levels from the beginning:
```bash
./train_curriculum_zebra_v2.sh
```

### Start training from a specific level:
```bash
# Start from n2_m3 level (will automatically use checkpoint from n2_m2)
./train_curriculum_zebra_v2.sh --start-level n2_m3

# Start from n3_m4 level (will automatically use checkpoint from n3_m3)
./train_curriculum_zebra_v2.sh --start-level n3_m4
```

### Get help:
```bash
./train_curriculum_zebra_v2.sh --help
```

## Available Levels (in training order)

1. `n2_m2` - 2 attributes, 2 objects (difficulty: 2×2=4)
2. `n2_m3` - 2 attributes, 3 objects (difficulty: 2×3=6)
3. `n2_m4` - 2 attributes, 4 objects (difficulty: 2×4=8)
4. `n3_m3` - 3 attributes, 3 objects (difficulty: 3×3=9)
5. `n2_m5` - 2 attributes, 5 objects (difficulty: 2×5=10)
6. `n3_m4` - 3 attributes, 4 objects (difficulty: 3×4=12)
7. `n4_m3` - 4 attributes, 3 objects (difficulty: 4×3=12)
8. `n3_m5` - 3 attributes, 5 objects (difficulty: 3×5=15)
9. `n5_m3` - 5 attributes, 3 objects (difficulty: 5×3=15)
10. `n4_m4` - 4 attributes, 4 objects (difficulty: 4×4=16)
11. `n4_m5` - 4 attributes, 5 objects (difficulty: 4×5=20)
12. `n5_m4` - 5 attributes, 4 objects (difficulty: 5×4=20)

## How it Works

When you specify `--start-level`, the script will:

1. **Validate** the level name against available levels
2. **Find the checkpoint** from the previous level (if not starting from first level)
3. **Skip earlier levels** in the training loop
4. **Start training** from your specified level using the previous level's latest checkpoint
5. **Continue training** all subsequent levels in order

## Error Handling

- Invalid level names are caught early with helpful error messages
- Missing checkpoints from previous levels trigger warnings but allow fallback to base model
- Missing data files for specific levels are skipped with warnings

## Examples for Your Setup

Based on your current situation:

```bash
# Since you have n2_m2 trained up to global_step_300, start from n2_m3:
./train_curriculum_zebra_v2.sh --start-level n2_m3

# Or if you want to continue from a later level:
./train_curriculum_zebra_v2.sh --start-level n3_m4
```

The script will automatically detect and use the latest checkpoint (`global_step_300`) from the previous level.
