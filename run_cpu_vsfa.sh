#!/bin/bash

#SBATCH --job-name=vfsakonvid
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-UGC/logs/vfsakonvid.out
#SBATCH --mem-per-cpu=80G

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
#conda create -n reproducibleresearch pip python=3.6

# Activate virtualenv
source activate reproducibleresearch
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script

# VFSA prediction
CUDA_VISIBLE_DEVICES=2 python VSFA.py --database=KoNViD_1k --exp_id=0


## Deactivate virtualenv
conda deactivate

