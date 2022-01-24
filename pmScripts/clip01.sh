#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=9:30:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/vis2PM_RGB_clip01.%j.out
#SBATCH --error=bch_outs/vis2PM_RGB_clip01.%j.err
# input first for modalities. All change the option name first.
# base one with mse n_phy1, no others
source activate pch1.5
python train.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model vis2PM --mod_src ${1} --mod_tar PMarray --n_phy 1 \
--type_whtL n --lambda_sum 0 --lambda_ssim 0 --lambda_D 0 \