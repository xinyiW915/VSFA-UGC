#!/bin/bash

#SBATCH --job-name=kfold_v1080
#SBATCH --partition=cpu
#SBATCH --nodes=6
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -o /mnt/storage/home/um20242/scratch/VSFA-UGC/logs/kfold_vfsa1080.out
#SBATCH --mem-per-cpu=50G

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

module load languages/anaconda3/2020.02-tflow-1.15
#conda create -n reproducibleresearch pip python=3.6

# Activate virtualenv
source activate reproducibleresearch
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script
#python < kfold_train_test_svr_export_model.py
python < kfold_ugc.py

## Deactivate virtualenv
conda deactivate