#!/bin/bash
# Curriculum training script for Zebra puzzles - Generate individual scripts for each level
set -e  # Exit on any error

# Configuration
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
export VLLM_ATTENTION_BACKEND=XFORMERS
EPOCHS_PER_LEVEL=30
BASE_DATA_DIR="data/loong/zebra_raw_levels"
BASE_CHECKPOINT_DIR="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum"
BASE_LOG_DIR="./logs/zebra_curriculum"

# Create directories
mkdir -p $BASE_CHECKPOINT_DIR
mkdir -p $BASE_LOG_DIR

# Define levels in order of difficulty (sorted by difficulty score n×m)
declare -a LEVELS=(
    "n2_m2:2x2"      # 2 attributes, 2 objects, difficulty 2×2=4
    "n2_m3:2x3"      # 2 attributes, 3 objects, difficulty 2×3=6
    "n2_m4:2x4"      # 2 attributes, 4 objects, difficulty 2×4=8
    "n3_m3:3x3"      # 3 attributes, 3 objects, difficulty 3×3=9
    "n2_m5:2x5"      # 2 attributes, 5 objects, difficulty 2×5=10
    "n3_m4:3x4"      # 3 attributes, 4 objects, difficulty 3×4=12
    "n4_m3:4x3"      # 4 attributes, 3 objects, difficulty 4×3=12
    "n3_m5:3x5"      # 3 attributes, 5 objects, difficulty 3×5=15
    "n5_m3:5x3"      # 5 attributes, 3 objects, difficulty 5×3=15
    "n4_m4:4x4"      # 4 attributes, 4 objects, difficulty 4×4=16
    "n4_m5:4x5"      # 4 attributes, 5 objects, difficulty 4×5=20
    "n5_m4:5x4"      # 5 attributes, 4 objects, difficulty 5×4=20
)

# Function to get the latest checkpoint from a directory
get_latest_checkpoint() {
    local checkpoint_dir=$1
    echo "DEBUG get_latest_checkpoint: Checking directory: $checkpoint_dir" >&2
    
    if [ -d "$checkpoint_dir" ]; then
        echo "DEBUG get_latest_checkpoint: Directory exists" >&2
        
        # Look for actor/global_step_* directories (actual checkpoint structure)
        actor_dir="$checkpoint_dir/actor"
        if [ -d "$actor_dir" ]; then
            echo "DEBUG get_latest_checkpoint: Found actor directory: $actor_dir" >&2
            global_step_dirs=$(find "$actor_dir" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null | sort -t_ -k3 -n)
            echo "DEBUG get_latest_checkpoint: Found global_step dirs: $global_step_dirs" >&2
            
            if [ -n "$global_step_dirs" ]; then
                latest_checkpoint=$(echo "$global_step_dirs" | tail -1)
                echo "DEBUG get_latest_checkpoint: Latest checkpoint: $latest_checkpoint" >&2
                echo "$latest_checkpoint"
                return
            fi
        fi
        
        # Fallback: Look for epoch_* directories
        epoch_dirs=$(find "$checkpoint_dir" -maxdepth 1 -name "epoch_*" -type d 2>/dev/null | sort -V)
        echo "DEBUG get_latest_checkpoint: Found epoch dirs: $epoch_dirs" >&2
        
        if [ -n "$epoch_dirs" ]; then
            latest_checkpoint=$(echo "$epoch_dirs" | tail -1)
            echo "DEBUG get_latest_checkpoint: Latest checkpoint: $latest_checkpoint" >&2
            echo "$latest_checkpoint"
        else
            echo "DEBUG get_latest_checkpoint: No checkpoints found" >&2
            echo ""
        fi
    else
        echo "DEBUG get_latest_checkpoint: Directory does not exist: $checkpoint_dir" >&2
        echo ""
    fi
}

# Function to generate and run training script for a single level
generate_and_run_level() {
    local level_name=$1
    local difficulty=$2
    local resume_checkpoint=$3
    
    echo "============================================================"
    echo "Training Level: $level_name (Difficulty: $difficulty)"
    echo "Data: $BASE_DATA_DIR/$level_name/"
    echo "Checkpoint Dir: $BASE_CHECKPOINT_DIR/$level_name/"
    echo "Resume from: $resume_checkpoint"
    echo "============================================================"
    
    # Prepare checkpoint path
    local model_path=""
    if [ -n "$resume_checkpoint" ]; then
        if [ -d "$resume_checkpoint" ]; then
            echo "Resuming from checkpoint: $resume_checkpoint"
            model_path="$resume_checkpoint"
        else
            echo "ERROR: Expected checkpoint does not exist: $resume_checkpoint"
            echo "This indicates a problem with the curriculum training chain."
            echo "Cannot continue without proper checkpoint."
            exit 1
        fi
    else
        echo "Starting from base model: $MODEL_PATH"
        model_path="$MODEL_PATH"
    fi
    
    # Create level-specific directories
    mkdir -p "$BASE_CHECKPOINT_DIR/$level_name"
    mkdir -p "$BASE_LOG_DIR"
    
    # Generate training script for this level
    local script_file="train_${level_name}_${difficulty}.sh"
    echo "Generating training script: $script_file"
    
    cat > "$script_file" << EOF
#!/bin/bash
# Auto-generated training script for curriculum level: $level_name ($difficulty)
# Generated on: $(date)
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "============================================================"
echo "Starting training for level $level_name (difficulty: $difficulty)"
echo "Model path: $model_path"
echo "Data: $BASE_DATA_DIR/$level_name/"
echo "Epochs: $EPOCHS_PER_LEVEL"
echo "Checkpoint will be saved to: $BASE_CHECKPOINT_DIR/$level_name/"
echo "============================================================"

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files="$BASE_DATA_DIR/$level_name/train.parquet" \\
    data.val_files="$BASE_DATA_DIR/$level_name/test.parquet" \\
    data.train_batch_size=8 \\
    data.val_batch_size=8 \\
    data.max_prompt_length=2000 \\
    data.max_response_length=4096 \\
    actor_rollout_ref.model.path="$model_path" \\
    actor_rollout_ref.actor.optim.lr=3e-7 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \\
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.temperature=0.7 \\
    actor_rollout_ref.rollout.n=16 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    algorithm.kl_ctrl.kl_coef=0.001 \\
    trainer.critic_warmup=0 \\
    trainer.logger=['wandb'] \\
    trainer.project_name='GRPO_Zebra_Curriculum' \\
    trainer.experiment_name="Qwen-7B-${level_name}-${difficulty}" \\
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=1 \\
    trainer.default_local_dir="$BASE_CHECKPOINT_DIR/$level_name" \\
    trainer.default_hdfs_dir=null \\
    trainer.save_freq=5 \\
    trainer.test_freq=5 \\
    trainer.total_epochs=$EPOCHS_PER_LEVEL \\
    \\\$@ 2>&1 | tee "$BASE_LOG_DIR/training_${level_name}_${difficulty}.log"

training_exit_code=\\\${PIPESTATUS[0]}
if [ \\\$training_exit_code -ne 0 ]; then
    echo "ERROR: Training failed for level $level_name with exit code \\\$training_exit_code"
    exit \\\$training_exit_code
fi

echo "============================================================"
echo "Successfully completed training for level $level_name"
echo "Checking for saved checkpoints..."
ls -la "$BASE_CHECKPOINT_DIR/$level_name/" || echo "Checkpoint directory not found"
if [ -d "$BASE_CHECKPOINT_DIR/$level_name/actor" ]; then
    echo "Actor checkpoints:"
    ls -la "$BASE_CHECKPOINT_DIR/$level_name/actor/"
fi
echo "============================================================"
EOF
    
    # Make script executable
    chmod +x "$script_file"
    
    echo "Running training script: $script_file"
    echo "Command: ./$script_file"
    
    # Run the generated script
    ./"$script_file"
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Training failed for level $level_name with exit code $exit_code"
        exit $exit_code
    fi
    
    echo "Completed training for level $level_name"
    echo ""
}

# Main training loop
echo "Starting Curriculum Training for Zebra Puzzles"
echo "Total levels: ${#LEVELS[@]}"
echo "Epochs per level: $EPOCHS_PER_LEVEL"
echo "Base model: $MODEL_PATH"
echo ""

# Check if data directory exists
if [ ! -d "$BASE_DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $BASE_DATA_DIR"
    echo "Please run the zebra_raw_levels.py preprocessor first to generate level-separated data."
    exit 1
fi

# Initialize previous checkpoint variable
previous_checkpoint=""

# Train each level in sequence
for level_info in "${LEVELS[@]}"; do
    # Parse level name and difficulty
    IFS=':' read -r level_name difficulty <<< "$level_info"
    
    # Check if data exists for this level
    if [ ! -f "$BASE_DATA_DIR/$level_name/train.parquet" ]; then
        echo "WARNING: Data not found for level $level_name, skipping..."
        continue
    fi
    
    # Generate and run training for this level
    generate_and_run_level "$level_name" "$difficulty" "$previous_checkpoint"
    
    # Update previous checkpoint to the latest from this level
    echo "DEBUG: Looking for checkpoints in: $BASE_CHECKPOINT_DIR/$level_name"
    ls -la "$BASE_CHECKPOINT_DIR/$level_name" || echo "Directory does not exist"
    
    previous_checkpoint=$(get_latest_checkpoint "$BASE_CHECKPOINT_DIR/$level_name")
    echo "DEBUG: get_latest_checkpoint returned: '$previous_checkpoint'"
    
    if [ -z "$previous_checkpoint" ]; then
        echo "ERROR: No checkpoint found after training $level_name"
        echo "DEBUG: Checking if any files exist in checkpoint dir:"
        find "$BASE_CHECKPOINT_DIR/$level_name" -type f 2>/dev/null | head -10
        echo "This is a critical error - training should have saved a checkpoint."
        echo "Cannot continue curriculum training without proper checkpoints."
        exit 1
    else
        echo "Next level will resume from: $previous_checkpoint"
        echo "DEBUG: Verifying checkpoint exists:"
        if [ -d "$previous_checkpoint" ]; then
            ls -la "$previous_checkpoint"
            echo "✓ Checkpoint verified and ready for next level"
        else
            echo "ERROR: Checkpoint path does not exist: $previous_checkpoint"
            echo "Cannot continue curriculum training."
            exit 1
        fi
    fi
    
    echo "----------------------------------------"
done

echo "============================================================"
echo "Curriculum Training Complete!"
echo "============================================================"
echo "All levels have been trained for $EPOCHS_PER_LEVEL epochs each."
echo "Final model checkpoint: $previous_checkpoint"
echo "Logs available in: $BASE_LOG_DIR"
echo "Checkpoints available in: $BASE_CHECKPOINT_DIR"
echo "Generated training scripts available in current directory"

# Create a summary file
summary_file="$BASE_LOG_DIR/curriculum_training_summary.txt"
echo "Zebra Puzzle Curriculum Training Summary" > "$summary_file"
echo "=======================================" >> "$summary_file"
echo "Completion Time: $(date)" >> "$summary_file"
echo "Base Model: $MODEL_PATH" >> "$summary_file"
echo "Epochs per Level: $EPOCHS_PER_LEVEL" >> "$summary_file"
echo "Total Levels: ${#LEVELS[@]}" >> "$summary_file"
echo "" >> "$summary_file"
echo "Training Order:" >> "$summary_file"
for level_info in "${LEVELS[@]}"; do
    IFS=':' read -r level_name difficulty <<< "$level_info"
    echo "  $level_name (difficulty: $difficulty)" >> "$summary_file"
done
echo "" >> "$summary_file"
echo "Final Checkpoint: $previous_checkpoint" >> "$summary_file"

echo "Summary saved to: $summary_file"
