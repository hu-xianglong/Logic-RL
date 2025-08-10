#!/bin/bash
# Curriculum training script for Zebra puzzles - train each level in order for 5 epochs
set -e  # Exit on any error

# Configuration
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
export VLLM_ATTENTION_BACKEND=XFORMERS
EPOCHS_PER_LEVEL=5
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
    if [ -d "$checkpoint_dir" ]; then
        latest_checkpoint=$(find "$checkpoint_dir" -name "epoch_*" -type d | sort -V | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            echo "$latest_checkpoint"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Function to train a single level
train_level() {
    local level_name=$1
    local difficulty=$2
    local resume_checkpoint=$3
    
    echo "============================================================"
    echo "Training Level: $level_name (Difficulty: $difficulty)"
    echo "Data: $BASE_DATA_DIR/$level_name/"
    echo "Checkpoint Dir: $BASE_CHECKPOINT_DIR/$level_name/"
    echo "Resume from: $resume_checkpoint"
    echo "============================================================"
    
    # Prepare checkpoint arguments
    local checkpoint_args=""
    if [ -n "$resume_checkpoint" ] && [ -d "$resume_checkpoint" ]; then
        echo "Resuming from checkpoint: $resume_checkpoint"
        checkpoint_args="actor_rollout_ref.model.path=$resume_checkpoint"
        
        # Log checkpoint info to Python for verification
        python3 -c "
import os
import json
from datetime import datetime

checkpoint_path = '$resume_checkpoint'
level_name = '$level_name'
difficulty = '$difficulty'

print(f'[CHECKPOINT INFO] Level: {level_name}')
print(f'[CHECKPOINT INFO] Difficulty: {difficulty}')
print(f'[CHECKPOINT INFO] Using checkpoint: {checkpoint_path}')

if os.path.exists(checkpoint_path):
    # Check if it's a valid checkpoint directory
    config_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f'[CHECKPOINT INFO] Checkpoint config found: {config_path}')
        except:
            print(f'[CHECKPOINT INFO] Warning: Could not read checkpoint config')
    
    # List contents of checkpoint directory
    contents = os.listdir(checkpoint_path)
    print(f'[CHECKPOINT INFO] Checkpoint contents: {contents}')
    
    # Check modification time
    mtime = os.path.getmtime(checkpoint_path)
    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[CHECKPOINT INFO] Checkpoint last modified: {mtime_str}')
else:
    print(f'[CHECKPOINT INFO] ERROR: Checkpoint path does not exist!')
    
print(f'[CHECKPOINT INFO] ==========================================')
"
    else
        echo "Starting from base model: $MODEL_PATH"
        checkpoint_args="actor_rollout_ref.model.path=$MODEL_PATH"
        
        # Log base model info
        python3 -c "
model_path = '$MODEL_PATH'
level_name = '$level_name'
difficulty = '$difficulty'

print(f'[CHECKPOINT INFO] Level: {level_name}')
print(f'[CHECKPOINT INFO] Difficulty: {difficulty}')
print(f'[CHECKPOINT INFO] Using base model: {model_path}')
print(f'[CHECKPOINT INFO] This is the first level in curriculum')
print(f'[CHECKPOINT INFO] ==========================================')
"
    fi
    
    # Create level-specific directories
    mkdir -p "$BASE_CHECKPOINT_DIR/$level_name"
    mkdir -p "$BASE_LOG_DIR"
    
    # Run training for this level
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$BASE_DATA_DIR/$level_name/train.parquet" \
        data.val_files="$BASE_DATA_DIR/$level_name/test.parquet" \
        data.train_batch_size=8 \
        data.val_batch_size=8 \
        data.max_prompt_length=2000 \
        data.max_response_length=4096 \
        $checkpoint_args \
        actor_rollout_ref.actor.optim.lr=3e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['wandb'] \
        trainer.project_name='GRPO_Zebra_Curriculum' \
        trainer.experiment_name="Qwen-7B-${level_name}-${difficulty}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.default_local_dir="$BASE_CHECKPOINT_DIR/$level_name" \
        trainer.default_hdfs_dir=null \
        trainer.save_freq=1 \
        trainer.test_freq=1 \
        trainer.total_epochs=$EPOCHS_PER_LEVEL \
        2>&1 | tee "$BASE_LOG_DIR/training_${level_name}_${difficulty}.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Training failed for level $level_name"
        exit 1
    fi
    
    echo "Completed training for level $level_name"
    
    # Log completion info and final checkpoint
    python3 -c "
import os
import glob
from datetime import datetime

level_name = '$level_name'
difficulty = '$difficulty'
checkpoint_dir = '$BASE_CHECKPOINT_DIR/$level_name'

print(f'[TRAINING COMPLETE] Level: {level_name} (difficulty: {difficulty})')
print(f'[TRAINING COMPLETE] Checkpoint directory: {checkpoint_dir}')

if os.path.exists(checkpoint_dir):
    # Find all epoch checkpoints
    epoch_dirs = glob.glob(os.path.join(checkpoint_dir, 'epoch_*'))
    if epoch_dirs:
        # Sort to get the latest epoch
        epoch_dirs.sort()
        latest_epoch = epoch_dirs[-1]
        epoch_num = os.path.basename(latest_epoch)
        
        # Check modification time
        mtime = os.path.getmtime(latest_epoch)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f'[TRAINING COMPLETE] Latest checkpoint: {latest_epoch}')
        print(f'[TRAINING COMPLETE] Final epoch: {epoch_num}')
        print(f'[TRAINING COMPLETE] Saved at: {mtime_str}')
        
        # List checkpoint contents
        if os.path.isdir(latest_epoch):
            contents = os.listdir(latest_epoch)
            print(f'[TRAINING COMPLETE] Checkpoint files: {contents}')
    else:
        print(f'[TRAINING COMPLETE] WARNING: No epoch checkpoints found!')
        all_contents = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
        print(f'[TRAINING COMPLETE] Directory contents: {all_contents}')
else:
    print(f'[TRAINING COMPLETE] ERROR: Checkpoint directory does not exist!')

print(f'[TRAINING COMPLETE] ==========================================')
"
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
    
    # Train the level
    train_level "$level_name" "$difficulty" "$previous_checkpoint"
    
    # Update previous checkpoint to the latest from this level
    previous_checkpoint=$(get_latest_checkpoint "$BASE_CHECKPOINT_DIR/$level_name")
    
    if [ -z "$previous_checkpoint" ]; then
        echo "WARNING: No checkpoint found after training $level_name"
        previous_checkpoint=""
    else
        echo "Next level will resume from: $previous_checkpoint"
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

# Create a summary file
summary_file="$BASE_LOG_DIR/curriculum_training_summary.txt"
echo "Zebra Puzzle Curriculum Training Summary" > "$summary_file"
echo "=======================================" >> "$summary_file"
echo "Start Time: $(date)" >> "$summary_file"
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
echo "Completion Time: $(date)" >> "$summary_file"

echo "Summary saved to: $summary_file"
