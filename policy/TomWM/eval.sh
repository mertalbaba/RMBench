#!/bin/bash
# Requires the WM server to be up first (cosmos env, GPU 7):
#   cd /home/mealbaba/tom-wm/cosmos-predict2.5 && \
#   CUDA_VISIBLE_DEVICES=7 CUDA_HOME=/home/mealbaba/cuda128 \
#   HF_HOME=/local/mealbaba/tomwm/hf .venv/bin/python -m tom_wm.runtime.server \
#       --ckpt /local/mealbaba/tomwm/checkpoints/a0_gate1 --port ${port}

policy_name=TomWM
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
port=${6:-6001}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --port ${port}
