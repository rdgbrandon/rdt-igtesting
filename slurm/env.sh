# Source from every Expanse job: sets paths and activates the conda env
# created by the one-time setup (~/rdt_setup.sh).
export PROJ=${PROJ:-/expanse/lustre/projects/ufl135/$USER}
source $PROJ/miniforge3/bin/activate
conda activate $PROJ/envs/rdt
export RDT_REPO=$PROJ/RoboticsDiffusionTransformer
export RDT_DATA=$PROJ/rdt-data
export HF_HOME=$PROJ/hf-cache
export CODE=$PROJ/rdt-igtesting
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
# software Vulkan (lavapipe) for SAPIEN rendering on GPU-less nodes, if installed
if [ -f $PROJ/vulkan/lvp_icd.json ]; then
    export VK_ICD_FILENAMES=$PROJ/vulkan/lvp_icd.json
    export LD_LIBRARY_PATH=$PROJ/vulkan/lib64:$LD_LIBRARY_PATH
fi
cd $CODE
