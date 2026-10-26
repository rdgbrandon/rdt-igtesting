#!/bin/bash
#SBATCH --job-name=rdt-attr
#SBATCH --account=ufl135
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Attribution + faithfulness, parallel over episodes (one array task per episode dir).
#   ls -d runs/rollouts/*_ep* > runs/attr_list.txt
#   sbatch --array=0-$(( $(wc -l < runs/attr_list.txt) - 1 ))%50 slurm/attribution.sh
# CALLS: which decision points per episode (default every 3rd), METHODS / N_EVALS as in run_attribution.py
source slurm/env.sh
EP=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" runs/attr_list.txt)
echo "episode=$EP cpus=$SLURM_CPUS_PER_TASK"
python scripts/run_attribution.py --episodes "$EP" --only-success \
    --calls ${CALLS:-every:3} --methods ${METHODS:-random,attn,grad,gxi,ig_zero,ig_bg,blurig,eig} \
    --n-evals ${N_EVALS:-16} --batch ${BATCH:-8} ${EXTRA:-} \
    --threads $SLURM_CPUS_PER_TASK --out runs/attr
