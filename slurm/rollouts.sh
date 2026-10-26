#!/bin/bash
#SBATCH --job-name=rdt-roll
#SBATCH --account=ufl135
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/%x-%A_%a.out
# Closed-loop rollouts, parallel over (task, episode chunk).
# Array task i -> task TASKS[i / CHUNKS], episodes [c*PER, (c+1)*PER).
#   default: 5 tasks x 10 chunks x 5 episodes = 250 episodes
#   submit:  sbatch slurm/rollouts.sh            (override: EPISODES_PER_TASK=20 sbatch --array=0-19 ...)
source slurm/env.sh
TASKS=(PickCube-v1 StackCube-v1 PushCube-v1 PegInsertionSide-v1 PlugCharger-v1)
PER=${PER:-5}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-50}
CHUNKS=$(( EPISODES_PER_TASK / PER ))
i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$(( i / CHUNKS ))]}
A=$(( (i % CHUNKS) * PER )); B=$(( A + PER ))
echo "task=$TASK episodes=$A:$B cpus=$SLURM_CPUS_PER_TASK"
python scripts/run_rollouts.py --task $TASK --episodes $A:$B --out runs/rollouts --threads $SLURM_CPUS_PER_TASK
