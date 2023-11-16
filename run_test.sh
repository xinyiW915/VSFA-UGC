#!/bin/sh
#SBATCH --account=eeng028284
#SBATCH --job-name=test
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-UGC/test.log
#SBATCH --mem=100GB

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

# Load modules required for runtime e.g.##
module load languages/anaconda3/2020.02-tflow-1.15
module load apps/ffmpeg/4.3
module load CUDA
# Activate virtualenv
source activate reproducibleresearch    #pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# test
python test_demo.py

## Deactivate virtualenv
conda deactivate