#!/bin/bash
# Training script for curriculum level: n2_m3 (2x3)
# Generated to fix checkpoint detection issue
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

echo "============================================================"
echo "Starting training for level n2_m3 (difficulty: 2x3)"
echo "Model path: /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m2/actor/global_step_300"
echo "Data: data/loong/zebra_raw_levels/n2_m3/"
echo "Epochs: 30"
echo "Checkpoint will be saved to: /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m3/"
echo "============================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="data/loong/zebra_raw_levels/n2_m3/train.parquet" \
    data.val_files="data/loong/zebra_raw_levels/n2_m3/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2000 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m2/actor/global_step_300" \
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
    trainer.experiment_name="Qwen-7B-n2_m3-2x3" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m3" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=30 \
    $@ 2>&1 | tee "./logs/zebra_curriculum/training_n2_m3_2x3.log"

training_exit_code=${PIPESTATUS[0]}
if [ $training_exit_code -ne 0 ]; then
    echo "ERROR: Training failed for level n2_m3 with exit code $training_exit_code"
    exit $training_exit_code
fi

echo "============================================================"
echo "Successfully completed training for level n2_m3"
echo "Checking for saved checkpoints..."
ls -la "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m3/" || echo "Checkpoint directory not found"
if [ -d "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m3/actor" ]; then
    echo "Actor checkpoints:"
    ls -la "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di35qir2/camel/loong_checkpoints/zebra_curriculum/n2_m3/actor/"
fi
echo "============================================================"
