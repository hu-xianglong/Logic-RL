set -x
echo "=== Memory Check ==="
echo "Physical node memory (from /proc/meminfo):"
cat /proc/meminfo | grep MemTotal

echo "Cgroup memory limit (Slurm allocation):"
CGROUP_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
echo "$CGROUP_LIMIT bytes ($(echo "$CGROUP_LIMIT / 1024 / 1024 / 1024" | bc) GB)"

echo "What Python/psutil sees:"
python3 -c "import psutil; m=psutil.virtual_memory(); print(f'Total: {m.total/1024**3:.1f} GB, Available: {m.available/1024**3:.1f} GB, Used: {m.used/1024**3:.1f} GB ({m.percent}% used)')"

export RAY_disable_cgroup_memory_limit=true
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
# Use your persistent storage location
# In your batch script, use your project directory for cache
export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export HF_DATASETS_CACHE=/workspace/.hf_cache


echo "HF Home:"
echo $HF_HOME
echo "Transformers Cache:"
echo $TRANSFORMERS_CACHE
echo "HF Datasets Cache:"
echo $HF_DATASETS_CACHE

echo "===================="


MODEL_PATH=./checkpoints/5ppl/actor/global_step_336
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/kk/instruct/5ppl/train.parquet \
    data.val_files=data/kk/instruct/5ppl/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=4096\
    actor_rollout_ref.model.path=$MODEL_PATH\
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen-7B-5PPL' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=./checkpoints/5ppl/second_run \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 $@ 2>&1 | tee grpo_5ppl.log

