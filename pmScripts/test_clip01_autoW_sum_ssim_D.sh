#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=9:30:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/vis2PM_RGB_clip01_autoW_sum_ssim_D.%j.out
#SBATCH --error=bch_outs/vis2PM_RGB_clip01_autoW_sum_ssim_D.%j.err
#source activate pch1.5
python test.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model vis2PM --mod_src ${1} --mod_tar PMarray --n_phy 1 \
--type_whtL pwrs --lambda_sum 1e-6 --lambda_ssim 10 --lambda_D 1






