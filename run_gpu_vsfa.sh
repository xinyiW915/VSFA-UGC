#!/bin/bash

#SBATCH --job-name=vfsa720
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-master/logs/vfsa720P.out
#SBATCH --mem=50GB

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
source activate reproducibleresearch        #pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script

# VFSA prediction
CUDA_VISIBLE_DEVICES=2 python VSFA.py --database=YOUTUBE_UGC_720P --exp_id=0

## Deactivate virtualenv
conda deactivate

#/mnt/storage/home/um20242/scratch/VSFA-master/CNN_features_720P/2_resnet-50_res5c.npy