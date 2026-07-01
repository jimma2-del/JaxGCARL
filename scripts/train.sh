#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate jaxgcrl

method=carl
env=ant_custom_forces
eval_env=ant

for seed in 1; do
  # --eval_only_path "checkpoints/42/AdamG1_damping0.1_seed4.pkl"
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python run.py "$method" \
    --wandb_project_name test --wandb_group first_run --exp_name test --num_evals 50 \
    --seed ${seed} --total_env_steps 10000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env ${env} --eval_env ${eval_env} --checkpoint_logdir "checkpoints/${seed}" --save_interval 5 \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn bwd_infonce --energy_fn norm \
    --train_step_multiplier 1 --log_wandb 
  done

echo "All runs have finished."



# BELOW is the code to run just eval_env for a variety of saved models. Change --eval_only_path to the correct path to the saved models.
# Change the seeds to whichever seeds as necessary.
# You may increase num_eval_envs from 2048 to 8192, it will decrease the variance, but 2048 already mostly minimizes it.
# After running this full script for a set of 10/50 runs, change the name of the resultant data.txt file. (Its actually a csv file).


##!/bin/bash
#
# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.
#
# NOTE: If you run into OOM issues, try reducing --num_envs
#
#eval "$(conda shell.bash hook)"
#conda activate jaxgcrl
#
#method=carl
#env=ant
#eval_env=ant_custom_masses
#
#for seed in 1 2 3 4 5 6 ; do
#  # 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
#  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python run.py "$method" \
#    --wandb_project_name test --wandb_group first_run --exp_name test --num_evals 50 --num_eval_envs 2048 \
#    --seed ${seed} --total_env_steps 10000000 --batch_size 256 --num_envs 512 \
#    --discounting 0.99 --action_repeat 1 --env ${env} --eval_env ${eval_env} --checkpoint_logdir "checkpoints/${seed}" --save_interval 5 \
#    --eval_only_path "../SmallRunsJaxGCARL/checkpointsCarl/${seed}/final_8475648.pkl"
#    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
#    --contrastive_loss_fn bwd_infonce --energy_fn norm \
#    --train_step_multiplier 1 --log_wandb 
#  done
#
#echo "All runs have finished."
