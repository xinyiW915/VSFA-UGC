#!/bin/bash

#SBATCH --job-name=cnn1080
#SBATCH --partition=cpu
#SBATCH --nodes=16
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-master/logs/cnn1080.out
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
source activate reproducibleresearch        #pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script

# test demo video sequence
#python test_demo.py --video_format=RGB --video_path=/mnt/storage/home/um20242/scratch/test_sequence/original_test.mp4
#python test_demo.py --video_format=RGB --video_path=/mnt/storage/home/um20242/scratch/test_sequence/decoded_test.mp4
#python test_demo.py --video_format=RGB --video_path=/mnt/storage/home/um20242/scratch/VSFA-master/test.mp4

# CNN features extraction
#nohup python -u CNNfeatures.py>test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python CNNfeaturesUGC.py --database=YOUTUBE_UGC_1080P --frame_batch_size=32


## Deactivate virtualenv
conda deactivate

