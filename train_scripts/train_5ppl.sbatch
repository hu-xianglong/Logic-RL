#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH -q mcml
#SBATCH --mem=0   
#SBATCH --time=20:00:00      
#SBATCH -o train_5ppl_%j.out
#SBATCH -e train_5ppl_%j.err
mkdir -p /workspace/.hf_cache
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di35qir2/camel/Logic-RL
enroot create --name pytorch-ready pytorch-cu121-py39-vllm.sqsh
chmod +x train_5ppl.sh
enroot start --root --mount $PWD:/workspace pytorch-ready \
    bash -c 'eval "$(/workspace/.conda/bin/conda shell.bash hook)" && \
             conda activate /workspace/.env39 && \
             ./train_5ppl.sh'
