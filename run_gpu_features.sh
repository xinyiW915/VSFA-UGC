#!/bin/bash
#SBATCH --account=eeng028284
#SBATCH --job-name=vsfacnn
##SBATCH --nodes=1
##SBATCH --gres=gpu:2
#SBATCH --partition gpu_veryshort
#SBATCH --time=6:00:00
#SBATCH -o /user/work/um20242/VSFA-UGC/logs/vsfacnn_runtime.out
#SBATCH --mem=100GB

# Load modules required for runtime e.g.##
module add apps/matlab/2018a
module load apps/ffmpeg/4.3
module load languages/anaconda3/2020.02-tflow-1.15
module load CUDA
# Activate virtualenv
source activate reproducibleresearch

cd $SLURM_SUBMIT_DIR

# CNN features extraction
#CUDA_VISIBLE_DEVICES=2 python CNNfeaturesUGC.py --database=YOUTUBE_UGC_2160P --frame_batch_size=32
CUDA_VISIBLE_DEVICES=2 python CNNfeaturesUGC.py --database=YOUTUBE_UGC_1080P_test --frame_batch_size=32

## Deactivate virtualenv
conda deactivate