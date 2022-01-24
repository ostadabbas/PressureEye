#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=9:30:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/vis2PM_RGB_clip01_as_beta2.%j.out
#SBATCH --error=bch_outs/vis2PM_RGB_clip01_as_beta2.%j.err
source activate pch1.5
python train.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model vis2PM --mod_src ${1} --mod_tar PMarray --n_phy 2 \
--type_whtL pwrs --lambda_sum 1e-6 --lambda_ssim 0 --lambda_D 0 \







