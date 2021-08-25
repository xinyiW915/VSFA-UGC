#!/bin/bash

#SBATCH --job-name=cnn2160
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-master/logs/cnn2160.out
#SBATCH --mem=80GB

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
source activate reproducibleresearch        #pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# CNN features extraction
CUDA_VISIBLE_DEVICES=2 python CNNfeaturesUGC.py --database=YOUTUBE_UGC_2160P --frame_batch_size=32

## Deactivate virtualenv
conda deactivate