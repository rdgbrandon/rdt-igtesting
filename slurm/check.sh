#!/bin/bash
#SBATCH --job-name=rdt-check
#SBATCH --account=ufl135
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Smoke test: environment, cached-sampler equivalence + timing, ManiSkill rendering.
# Cost: <= 16 core-hours.
source slurm/env.sh
echo "== node"; hostname; nproc; grep -m1 "model name" /proc/cpuinfo; cat /etc/os-release | head -2
echo "== python"; python -c "import torch,transformers,diffusers,mani_skill,sapien;print(torch.__version__,transformers.__version__,diffusers.__version__,sapien.__version__,'threads',torch.get_num_threads())"
echo "== equivalence"; python tests/check_policy_equivalence.py --threads $SLURM_CPUS_PER_TASK 2>&1 | grep -vi warn
echo "== render"; python tests/check_render.py 2>&1 | grep -vi warn | tail -15
echo "== done"
