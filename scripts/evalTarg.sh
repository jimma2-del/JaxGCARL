#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

method="$1"
env=ant
eval_env="$4"
run_name="$2"
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
for seed in 6 7 8 9 10 11 12; do
  
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python run.py "$method" \
    --wandb_project_name test --wandb_group ${run_name} --exp_name "${run_name}_${seed}" --num_evals 50 --num_eval_envs 2048 \
    --seed ${seed} --total_env_steps 10000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env ${env} --eval_env ${eval_env} --checkpoint_logdir "checkpoints_${run_name}/${seed}" --save_interval 5 \
    --eval_only_path "checkpoints_${run_name}/${seed}/$3" \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn bwd_infonce --energy_fn norm \
    --train_step_multiplier 1 --log_wandb 
  done
echo "All runs have finished."
