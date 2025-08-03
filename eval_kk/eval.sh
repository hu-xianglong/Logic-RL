model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw/actor/global_step_1575" #model path
log_path="log/mixed_raw_1575"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_3/actor/global_step_325" #model path
log_path="log/mixed_3_325"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_4/actor/global_step_350" #model path
log_path="log/mixed_4_1500"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw_2050/actor/global_step_350" #model path
log_path="log/mixed_raw_2050"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_5_1500/actor/global_step_600" #model path
log_path="log/mixed_5_1500"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw_2050/actor/global_step_100" #model path
log_path="log/mixed_raw_2050_100"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw_1900/actor/global_step_150" #model path
log_path="log/mixed_raw_1900_150"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw/actor/global_step_1900" #model path
log_path="log/mixed_raw_1900"
model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL/checkpoints/mixed_raw_2050/actor/global_step_350" #model path
log_path="log/mixed_raw_2050"
config="vllm"
num_limit=100
max_token=8192
ntrain=0
split="test"

mkdir -p ${log_path}

for eval_nppl in 3 4 5 6; do
    log_file="${log_path}/${eval_nppl}.log"
    echo "Starting job for eval_nppl: $eval_nppl, logging to $log_file"

    source ../.venv/bin/activate

    CUDA_VISIBLE_DEVICES=$((eval_nppl - 3)) PYTHONUNBUFFERED=1 python main_eval_instruct.py --batch_size 8 --model ${model} --max_token ${max_token} \
    --ntrain ${ntrain} --config ${config} --limit ${num_limit} --split ${split} --temperature 1.0  --top_p 1.0 \
    --problem_type "clean" --eval_nppl ${eval_nppl} > "$log_file" 2>&1 &
done &  