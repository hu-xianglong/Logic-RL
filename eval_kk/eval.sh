model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed/actor/global_step_575" #model path
config="vllm"
num_limit=100
max_token=8192
ntrain=0
split="test"
log_path="log/mixed_575"

mkdir -p ${log_path}

for eval_nppl in 7; do
    log_file="${log_path}/${eval_nppl}.log"
    echo "Starting job for eval_nppl: $eval_nppl, logging to $log_file"

    source ../.venv/bin/activate

    CUDA_VISIBLE_DEVICES=$((eval_nppl - 5)) PYTHONUNBUFFERED=1 python main_eval_instruct.py --batch_size 8 --model ${model} --max_token ${max_token} \
    --ntrain ${ntrain} --config ${config} --limit ${num_limit} --split ${split} --temperature 1.0  --top_p 1.0 \
    --problem_type "clean" --eval_nppl ${eval_nppl} > "$log_file" 2>&1 &
done &  