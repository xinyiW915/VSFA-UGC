#!/bin/bash

#SBATCH --job-name=cnnALL
#SBATCH --partition=cpu
#SBATCH --nodes=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-UGC/logs/cnnALL.out
#SBATCH --mem-per-cpu=90G

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

# CNN features extraction
#nohup python -u CNNfeatures.py>test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python CNNfeaturesUGC.py --database=YOUTUBE_UGC_ALL_p2 --frame_batch_size=32


## Deactivate virtualenv
conda deactivate

