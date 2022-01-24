#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=9:30:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/RGB_h1-0_sum0_lapT.%j.out
#SBATCH --error=bch_outs/RGB_h1-0_sum0_lapT.%j.err
# tune lap only wrp h1-0
source activate pch1.5
python train.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model vis2PM --mod_src RGB --mod_tar PMarray --n_phy 1 \
--type_whtL pwrs --h_mode 1 --h_base 0 --kdeMode 0 --lambda_lap ${1} --lambda_sum 0 --lambda_ssim 0 --lambda_D 0 \





